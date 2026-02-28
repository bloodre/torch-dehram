"""FEEC inner product from assembled Whitney mass matrices.

FEECInnerProduct provides a concrete InnerProduct implementation using
assembled sparse mass matrices from Whitney finite element spaces. Unlike
DEC (diagonal), these mass matrices are full but sparse and SPD.

Supports both per-degree (k given) and global (k=None) operations when used
with ContiguousChainComplex, analogous to DECInnerProduct.
"""
# pylint: disable=invalid-name

from __future__ import annotations

from typing import Optional, Union

import torch
from torch import Tensor
from torch_sparse import SparseTensor

from ..complex.chain import ChainComplex, ContiguousChainComplex
from ..metric.inner_product import InnerProduct
from ..solvers.cg import Preconditioner, cg_solve
from .geometry import SimplicialGeometry
from .mass import assemble_global_mass_all, is_diagonal_matrix


class FEECInnerProduct(InnerProduct):
    """Whitney FEEC inner product with sparse mass matrices.

    For each degree k, stores a sparse SPD matrix M_k representing the
    L² inner product on Whitney k-forms:

        <x, y>_k = x^T M_k y

    Supports global operations (k=None) for ContiguousChainComplex by
    assembling a block-diagonal matrix with offsets.

    Args:
        masses (dict[int, SparseTensor]): per-degree mass matrices.
        chain_complex (ChainComplex | ContiguousChainComplex): complex
            providing offsets for global mode.
        preconditioner (Preconditioner | None): optional preconditioner
            for iterative solves (default: None → unpreconditioned CG).
        check_diagonality (bool): whether to check if mass matrices are
            diagonal (default: True).
    """

    def __init__(
        self,
        masses: dict[int, SparseTensor],
        chain_complex: Union[ChainComplex, ContiguousChainComplex],
        preconditioner: Optional[Preconditioner] = None,
        check_diagonality: bool = True,
    ):
        """Initialize with mass matrices and diagonal detection.

        Args:
            masses: Per-degree mass matrices.
            chain_complex: Topology for global operations.
            preconditioner: Optional preconditioner for CG solves.
        """
        self._masses = masses
        self._chain_complex = chain_complex
        self._is_contiguous = isinstance(chain_complex, ContiguousChainComplex)
        self._preconditioner = preconditioner

        self._mass_global: Optional[SparseTensor] = None

        # Precompute diagonal status for all mass matrices
        self._is_diagonal: dict[int, bool] = {}
        if check_diagonality:
            self._is_diagonal = {
                k: is_diagonal_matrix(M)
                for k, M in masses.items()
            }

    @classmethod
    def from_geometry(
        cls,
        chain_complex: ChainComplex,
        geometry: SimplicialGeometry,
        quad_degree: int = 2,
        preconditioner: Optional[Preconditioner] = None,
    ) -> FEECInnerProduct:
        """Build FEEC inner product from mesh geometry and quadrature.

        Args:
            chain_complex (ChainComplex): simplicial mesh topology.
            geometry (SimplicialGeometry): embedded vertex positions.
            quad_degree (int): polynomial degree for quadrature (default 2).
            preconditioner (Preconditioner | None): optional preconditioner.

        Returns:
            FEECInnerProduct with assembled mass matrices.
        """
        masses = assemble_global_mass_all(chain_complex, geometry, quad_degree)
        return cls(masses, chain_complex, preconditioner)

    def _assemble_global_mass(self) -> SparseTensor:
        """Assemble global block-diagonal mass matrix (cached)."""
        if self._mass_global is not None:
            return self._mass_global

        if not self._is_contiguous:
            raise ValueError(
                "Global mass matrix requires ContiguousChainComplex"
            )

        offsets = self._chain_complex.offsets.cpu()
        n_total = int(offsets[-1].item())
        n_degrees = len(offsets) - 1

        rows = []
        cols = []
        vals = []

        for k in range(n_degrees):
            offset_k = int(offsets[k].item())
            M_k = self._masses[k]

            row_k, col_k, val_k = M_k.coo()
            rows.append(row_k + offset_k)
            cols.append(col_k + offset_k)
            vals.append(val_k)

        row_global = torch.cat(rows, dim=0)
        col_global = torch.cat(cols, dim=0)
        val_global = torch.cat(vals, dim=0)

        self._mass_global = SparseTensor(
            row=row_global,
            col=col_global,
            value=val_global,
            sparse_sizes=(n_total, n_total),
        ).coalesce()

        return self._mass_global

    def _invalidate_global_cache(self) -> None:
        """Invalidate global mass cache after device movement."""
        self._mass_global = None

    def matrix(self, k: Optional[int] = None) -> SparseTensor:
        """Return mass matrix M_k or global M.

        Args:
            k (int | None): degree, or None for global operator.

        Returns:
            SparseTensor mass matrix.

        Raises:
            ValueError: if k=None but complex is not ContiguousChainComplex.
        """
        if k is None:
            if not self._is_contiguous:
                raise ValueError(
                    "k=None (global matrix) requires ContiguousChainComplex"
                )
            return self._assemble_global_mass()
        return self._masses[k]

    def diagonal(self, k: Optional[int] = None) -> Tensor:
        """Return diagonal of M_k or global M.

        Args:
            k (int | None): degree, or None for global operator.

        Returns:
            (n_k,) or (n_total,) diagonal entries.

        Raises:
            ValueError: if k=None but complex is not ContiguousChainComplex.
        """
        M = self.matrix(k)
        return M.get_diag()

    def apply(self, x: Tensor, k: Optional[int] = None) -> Tensor:
        """Apply M_k to cochain values x.

        Args:
            x (Tensor): (n_k, d) or (n_total, d) cochain values.
            k (int | None): degree, or None for global operator.

        Returns:
            M_k @ x, same shape as x.

        Raises:
            ValueError: if k=None but complex is not ContiguousChainComplex.
        """
        M = self.matrix(k)
        if x.ndim == 1:
            return M @ x
        return M @ x

    def pair(self, x: Tensor, y: Tensor, k: Optional[int] = None) -> Tensor:
        """Compute inner product <x, y>_k = x^T M_k y.

        Args:
            x (Tensor): (n_k,) or (n_k, d) tensor of cochain values.
            y (Tensor): (n_k,) or (n_k, d) tensor of cochain values.
            k (int | None): cochain degree, or None for global operator.

        Returns:
            Scalar or (d,) tensor of inner product values.
        """
        My = self.apply(y, k)
        if x.ndim == 1:
            return torch.dot(x, My)
        return torch.sum(x * My, dim=0)

    def solve(
        self,
        b: Tensor,
        k: Optional[int] = None,
        tol: float = 1e-6,
        maxiter: int = 30,
    ) -> Tensor:
        """Solve M_k @ x = b for x using optimal method.

        Automatically routes to diagonal solve when M_k is diagonal,
        otherwise uses preconditioned CG for full sparse matrices.

        Args:
            b: Right-hand side with shape (n_k, d) or (n_total, d).
            k: Form degree, or None for global operator.
            tol: Tolerance for CG (ignored for diagonal solves).
            maxiter: Maximum CG iterations (ignored for diagonal solves).

        Returns:
            Solution x with same shape as b.

        Raises:
            ValueError: If k=None but complex is not ContiguousChainComplex.
        """
        # Check if we can use fast diagonal solve
        if k is not None and self._is_diagonal.get(k, False):
            return self._diag_solve(b, k)
        else:
            return self._cg_solve(b, k, tol, maxiter)

    def _diag_solve(self, b: Tensor, k: int) -> Tensor:
        """Solve diagonal system M_k @ x = b using O(n) operations.

        For diagonal matrices, the solution is simply x_i = b_i / M_ii.
        This provides exact solution in linear time without iteration.

        Args:
            b: Right-hand side with shape (n_k, d) or (n_k,).
            k: Form degree for the diagonal mass matrix.

        Returns:
            Solution x with same shape as b.
        """
        return b / self.matrix(k)

    def _cg_solve(
        self,
        b: Tensor,
        k: Optional[int] = None,
        tol: float = 1e-6,
        maxiter: int = 30,
    ) -> Tensor:
        """Solve M_k @ x = b for x using preconditioned conjugate gradient.

        Used when the mass matrix is not diagonal. Iteratively solves
        the sparse symmetric positive definite system.

        Args:
            b: Right-hand side with shape (n_k, d) or (n_total, d).
            k: Form degree, or None for global operator.
            tol: Convergence tolerance for CG iterations.
            maxiter: Maximum number of CG iterations.

        Returns:
            Solution x with same shape as b.

        Raises:
            ValueError: If k=None but complex is not ContiguousChainComplex.
        """
        M = self.matrix(k)

        x, _ = cg_solve(
            M,
            b,
            preconditioner=self._preconditioner,
            tol=tol,
            maxiter=maxiter,
        )
        return x

    def to(self, *args, **kwargs) -> FEECInnerProduct:
        """Move all tensors to specified device/dtype."""
        self._masses = {
            k: M.to(*args, **kwargs) for k, M in self._masses.items()
        }
        self._invalidate_global_cache()
        return self

    def cpu(self) -> FEECInnerProduct:
        """Move to CPU."""
        return self.to("cpu")

    def cuda(self) -> FEECInnerProduct:
        """Move to CUDA."""
        return self.to("cuda")
