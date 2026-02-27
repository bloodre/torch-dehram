"""Conjugate Gradient solver and preconditioner abstractions.

This module provides a generic, framework-independent implementation of the
Preconditioned Conjugate Gradient (PCG) algorithm for solving symmetric
positive definite linear systems A x = b.

It contains no dependencies on chain complexes or inner products: those
modules consume this one by composing their own operators and passing them
to cg_solve.
"""
# pylint: disable=invalid-name

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable, Optional, Union

import torch
from torch import Tensor
from torch_sparse import SparseTensor

from ..utils import extract_sparse_diagonal


# ------------------------------------------------------------------
# Type alias for linear operators
# ------------------------------------------------------------------

LinearOperator = Union[Tensor, SparseTensor, Callable[[Tensor], Tensor]]


def _matvec(A: LinearOperator, x: Tensor) -> Tensor:
    """Apply linear operator A to vector x.

    Args:
        A (LinearOperator): dense/sparse matrix or callable operator.
        x (Tensor): input tensor.

    Returns:
        A(x) as Tensor.
    """
    if callable(A):
        return A(x)
    return A @ x


# ------------------------------------------------------------------
# Preconditioner abstraction
# ------------------------------------------------------------------


class Preconditioner(ABC):
    """Abstract base class for preconditioners.

    A preconditioner M approximates A^{-1}. Given a residual r,
    apply(r) returns an approximation of A^{-1} r, which improves
    the conditioning of the CG iteration.
    """

    @abstractmethod
    def apply(self, r: Tensor) -> Tensor:
        """Apply preconditioner to residual r.

        Args:
            r (Tensor): residual tensor, same shape as x and b.

        Returns:
            Approximation of A^{-1} r, same shape as r.
        """


class JacobiPreconditioner(Preconditioner):
    """Jacobi preconditioner based on the diagonal of A, with optional damping.

    Applies ω * M^{-1} r = ω * (r / diag), where ω ∈ (0, 1] is the
    damping parameter. Damping can improve stability and convergence
    for ill-conditioned systems. Effective when the diagonal captures
    most of the conditioning of A.

    Args:
        diag: (n,) tensor of diagonal entries of A.
        omega: Damping parameter (default: 1.0, no damping).
    """

    def __init__(self, diag: Tensor, omega: float = 1.0):
        if not 0 < omega <= 1.0:
            raise ValueError("Damping parameter omega must be in (0, 1]")

        self._omega = omega

        if diag.ndim == 1:
            self._diag = diag.unsqueeze(-1)
        elif diag.ndim == 2 and diag.size(1) == 1:
            self._diag = diag
        else:
            raise ValueError("Diagonal must be a vector or a 1-column matrix.")

    def apply(self, r: Tensor) -> Tensor:
        """Apply damped diagonal preconditioning.

        Args:
            r: (n,) or (n, d) residual tensor.

        Returns:
            ω * (r / diag), same shape as r.
        """
        # Apply ω * M⁻¹_approx = ω * diag(M)⁻¹
        return self._omega * (r / self._diag)


def estimate_condition_number_heuristic(M: SparseTensor) -> int:
    """Fast heuristics for condition number estimation.

    Uses diagonal dominance and sparsity patterns to estimate condition number
    without expensive eigenvalue computations.

    Args:
        M: SparseTensor with shape (n, n)

    Returns:
        Estimated condition number (order of magnitude).
    """
    # Method 1: Diagonal dominance ratio
    diag = extract_sparse_diagonal(M)
    off_diagonal_sum = torch.sum(M.storage.value()) - torch.sum(diag)
    dominance_ratio = torch.sum(diag) / off_diagonal_sum

    # Method 2: Matrix size and sparsity
    n = M.size(0)
    nnz = len(M.storage.value())
    sparsity = nnz / (n * n)

    # Heuristic mapping
    if dominance_ratio > 10 and sparsity < 0.05:
        return 10  # Well-conditioned
    elif dominance_ratio > 5 and sparsity < 0.1:
        return 100  # Moderate
    elif dominance_ratio > 2:
        return 1000  # Ill-conditioned
    else:
        return 10000  # Very ill-conditioned


def simple_condition_estimate(M: SparseTensor) -> int:
    """Ultra-fast condition number guess based on sparsity.

    Very rough estimate using only sparsity pattern.

    Args:
        M: SparseTensor with shape (n, n)

    Returns:
        Estimated condition number (order of magnitude).
    """
    n = M.size(0)
    nnz_ratio = M.storage.value().numel() / (n * n)

    if nnz_ratio < 0.01:      # Very sparse
        return 1000
    elif nnz_ratio < 0.05:    # Moderately sparse
        return 500
    else:                     # Denser
        return 100


def suggest_omega(condition_number: int) -> float:
    """Suggest damping parameter omega based on condition number.

    Maps condition number to appropriate damping parameter for Jacobi
    preconditioning. Higher condition numbers require more damping.

    Args:
        condition_number: Estimated condition number (order of magnitude).

    Returns:
        Recommended omega damping parameter in (0, 1].
    """
    if condition_number <= 10:
        return 1.0      # Well-conditioned - no damping needed
    elif condition_number <= 100:
        return 0.9      # Light damping
    elif condition_number <= 1000:
        return 0.8      # Moderate damping
    elif condition_number <= 10000:
        return 0.7      # Strong damping
    else:
        return 0.6      # Heavy damping for very ill-conditioned


class OperatorPreconditioner(Preconditioner):
    """Preconditioner wrapping an arbitrary linear operator.

    Allows using any dense/sparse matrix or callable as a preconditioner,
    without imposing specific structure.

    Args:
        operator (LinearOperator): dense/sparse matrix or callable P such
            that P(r) approximates A^{-1} r.
    """

    def __init__(self, operator: LinearOperator):
        self._op = operator

    def apply(self, r: Tensor) -> Tensor:
        """Apply operator preconditioner to residual r.

        Args:
            r (Tensor): (n,) or (n, d) residual tensor.

        Returns:
            P(r), same shape as r.
        """
        return _matvec(self._op, r)


# ------------------------------------------------------------------
# CG solver
# ------------------------------------------------------------------


def cg_solve(
    A: LinearOperator,
    b: Tensor,
    x0: Optional[Tensor] = None,
    preconditioner: Optional[Preconditioner] = None,
    tol: float = 1e-8,
    maxiter: Optional[int] = None,
) -> tuple[Tensor, dict]:
    """Solve the SPD linear system A x = b via Preconditioned Conjugate Gradient.

    Supports both vector (n,) and batched (n, d) right-hand sides. When b has
    shape (n, d), all d systems are solved simultaneously.

    Args:
        A (LinearOperator): symmetric positive definite operator.
        b (Tensor): right-hand side, shape (n,) or (n, d).
        x0 (Tensor | None): initial guess, defaults to zero.
        preconditioner (Preconditioner | None): optional preconditioner M.
            When None, CG runs without preconditioning (M = I).
        tol (float): stopping tolerance on the residual norm.
        maxiter (int | None): maximum number of iterations. Defaults to n.

    Returns:
        x (Tensor): approximate solution, same shape as b.
        info (dict): convergence info with keys:
            - "converged" (bool)
            - "iterations" (int)
            - "residual_norm" (float): max residual norm across features.
    """
    n = b.shape[0]
    _maxiter = maxiter if maxiter is not None else n

    # Initial guess
    x = torch.zeros_like(b) if x0 is None else x0.clone()

    # Initial residual: r = b - A x
    r = b - _matvec(A, x)

    # Initial preconditioned residual
    z = preconditioner.apply(r) if preconditioner is not None else r.clone()

    p = z.clone()

    # Inner products along the n dimension: scalar or (d,) if batched
    rz_old = (r * z).sum(dim=0)

    converged = False
    it = 0

    for it in range(_maxiter):
        Ap = _matvec(A, p)

        # Step size alpha = (r^T z) / (p^T A p)
        pAp = (p * Ap).sum(dim=0)
        alpha = rz_old / pAp

        # Update solution and residual
        x = x + alpha * p
        r = r - alpha * Ap

        # Check convergence
        residual_norm = r.norm(dim=0)
        if residual_norm.max() < tol:
            converged = True
            break

        z = preconditioner.apply(r) if preconditioner is not None else r.clone()

        rz_new = (r * z).sum(dim=0)

        # Direction update: p = z + beta * p
        beta = rz_new / rz_old
        p = z + beta * p
        rz_old = rz_new

    info = {
        "converged": converged,
        "iterations": it + 1,
        "residual_norm": r.norm(dim=0).max().cpu().detach().item(),
    }

    return x, info
