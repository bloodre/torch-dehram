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

from ..utils.matrix import extract_sparse_diagonal


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
