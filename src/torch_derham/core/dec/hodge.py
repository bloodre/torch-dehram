"""DEC inner product from circumcentric dual cell measures.

DECInnerProduct is a diagonal inner product on k-cochains defined by:

    <x, y>_k = x^T M_k y,   M_k = diag(dual_measure[k] / primal_measure[k])

For a circumcentric DEC complex (Delaunay mesh), M_k is symmetric positive
definite and the codifferential is exactly adjoint to the coboundary.

For ContiguousChainComplex, the global diagonal is assembled by concatenating
per-degree diagonals according to the offsets layout, enabling k=None operators.
"""
# pylint: disable=invalid-name

from __future__ import annotations

from typing import Optional, Union

import torch
from torch import Tensor

from ..complex.chain import ChainComplex, ContiguousChainComplex
from ..metric.inner_product import InnerProduct
from .dual_cells import SimplicialDualCellMeasure


class DECInnerProduct(InnerProduct):
    """Diagonal DEC inner product from circumcentric dual cell measures.

    For each degree k, the mass matrix is diagonal:

        M_k[σ] = dual_measure[k][σ]

    where dual_measure[k][σ] is the circumcentric dual cell volume of σ.

    Note: in classical DEC, the diagonal also involves a primal volume factor
    (|σ| in the denominator). That factor is 1 for 0-cochains (vertices have
    no intrinsic measure). Subclasses or factory methods may supply primal
    volumes for the full ratio; this class stores what is provided.

    Supports both per-degree (k given) and global (k=None) operations
    when the chain complex is a ContiguousChainComplex.

    Args:
        dual (SimplicialDualCellMeasure): circumcentric dual measures.
        chain_complex (ChainComplex | ContiguousChainComplex): complex
            providing offsets for global mode.
    """

    def __init__(
        self,
        dual: SimplicialDualCellMeasure,
        chain_complex: Union[ChainComplex, ContiguousChainComplex],
    ):
        self._dual = dual
        self._chain_complex = chain_complex
        self._is_contiguous = isinstance(chain_complex, ContiguousChainComplex)

        # Per-degree diagonal cache
        self._diag_k: dict[int, Tensor] = {}

        # Global diagonal (k=None), built lazily for ContiguousChainComplex
        self._diag_global: Optional[Tensor] = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _diagonal_k(self, k: int) -> Tensor:
        """Return (n_k,) diagonal tensor for degree k, cached."""
        if k not in self._diag_k:
            self._diag_k[k] = self._dual.measure(k)
        return self._diag_k[k]

    def _diagonal_global(self) -> Tensor:
        """Return (n_total,) global diagonal for ContiguousChainComplex, cached."""
        if self._diag_global is None:
            offsets = self._chain_complex.offsets.cpu()
            n_total = int(offsets[-1].item())
            device = self._dual.device()
            dtype = self._diagonal_k(0).dtype

            diag = torch.empty(n_total, dtype=dtype, device=device)
            for k in range(self._dual.dim + 1):
                start = int(offsets[k].item())
                end = int(offsets[k + 1].item())
                diag[start:end] = self._diagonal_k(k)

            self._diag_global = diag

        return self._diag_global

    def _invalidate_global_cache(self) -> None:
        """Invalidate global diagonal cache (e.g. after device movement)."""
        self._diag_global = None

    # ------------------------------------------------------------------
    # InnerProduct interface
    # ------------------------------------------------------------------

    def diagonal(self, k: Optional[int] = None) -> Tensor:
        """Return diagonal of M_k or global diagonal when k=None.

        Args:
            k (int | None): cochain degree, or None for global operator
                (requires ContiguousChainComplex).

        Returns:
            (n_k,) or (n_total,) diagonal tensor.

        Raises:
            ValueError: if k=None but complex is not ContiguousChainComplex.
        """
        if k is None:
            if not self._is_contiguous:
                raise ValueError(
                    "k=None (global diagonal) requires ContiguousChainComplex"
                )
            return self._diagonal_global()
        return self._diagonal_k(k)

    def apply(self, k: Optional[int], x: Tensor) -> Tensor:
        """Apply M_k to cochain values x.

        Args:
            k (int | None): cochain degree, or None for global operator.
            x (Tensor): (n_k, d) or (n_total, d) cochain values.

        Returns:
            M_k @ x = diag_k * x, same shape as x.

        Raises:
            ValueError: if k=None but complex is not ContiguousChainComplex.
        """
        d = self.diagonal(k)
        if x.ndim == 1:
            return d * x
        return d.unsqueeze(-1) * x

    def solve(self, k: Optional[int], b: Tensor) -> Tensor:
        """Solve M_k @ x = b for x.

        Args:
            k (int | None): cochain degree, or None for global operator.
            b (Tensor): (n_k, d) or (n_total, d) right-hand side.

        Returns:
            x = b / diag_k, same shape as b.

        Raises:
            ValueError: if k=None but complex is not ContiguousChainComplex.
        """
        d = self.diagonal(k)
        if b.ndim == 1:
            return b / d
        return b / d.unsqueeze(-1)

    def matrix(self, k: Optional[int] = None) -> None:
        """DEC mass matrix is diagonal; use diagonal() instead.

        Returns None: a sparse identity-scaled matrix would be wasteful.
        The diagonal form is the canonical representation.
        """  # pylint: disable=unused-argument
        return None

    # ------------------------------------------------------------------
    # Device movement
    # ------------------------------------------------------------------

    def to(self, *args, **kwargs) -> DECInnerProduct:
        """Move all tensors to the specified device/dtype."""
        self._diag_k = {k: v.to(*args, **kwargs) for k, v in self._diag_k.items()}
        self._dual = self._dual.to(*args, **kwargs)
        self._invalidate_global_cache()
        return self

    def cpu(self) -> DECInnerProduct:
        """Move all tensors to CPU."""
        return self.to("cpu")

    def cuda(self) -> DECInnerProduct:
        """Move all tensors to CUDA."""
        return self.to("cuda")
