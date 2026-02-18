"""Hodge star operator abstraction.

The Hodge star is a musical isomorphism ★: Ω^k → Ω^{n-k} that depends on
the metric structure (inner product) and the volume form. It satisfies:

    <α, β>_k vol = α ∧ (★β)

where n is the ambient dimension and vol is the volume form.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

from torch import Tensor
from torch_sparse import SparseTensor

from ..complex.chain import ChainComplex, ContiguousChainComplex
from .inner_product import InnerProduct


class HodgeStar(ABC):
    """Abstract base class for Hodge star operators.

    The Hodge star ★_k: C^k → C^{n-k} is a metric-dependent isomorphism
    between k-forms and (n-k)-forms. It encodes the relationship between
    the inner product and the wedge product:

        <α, β>_k vol = α ∧ (★β)

    For diagonal inner products (DEC), the Hodge star is diagonal and
    computed directly from dual cell measures.

    For general inner products (FEEC), the Hodge star may require
    solving linear systems and is typically assembled as a sparse matrix.

    Global operator support (k=None):
    When k=None, the operator works on the full graded space, applying
    the grade-reversing involution across all degrees simultaneously.
    This is useful for ContiguousChainComplex representations.

    Args:
        chain_complex (ChainComplex | ContiguousChainComplex): chain complex
            providing dimension info.
        inner_product (InnerProduct): inner product defining the metric.
    """

    def __init__(
        self,
        chain_complex: ChainComplex | ContiguousChainComplex,
        inner_product: InnerProduct,
    ):
        self.chain_complex = chain_complex
        self.inner_product = inner_product
        self._is_contiguous = hasattr(chain_complex, 'data')

    @property
    def dim(self) -> int:
        """Ambient dimension (maximum degree)."""
        return self.chain_complex.dim

    @abstractmethod
    def apply(self, x: Tensor, k: Optional[int] = None) -> Tensor:
        """Apply Hodge star operator ★_k: C^k → C^{n-k}.

        Args:
            x (Tensor): (n_k, d) or (n_total, d) tensor of cochain values.
            k (int | None): cochain degree. Required for ChainComplex,
                optional for ContiguousChainComplex (None gives global operator).

        Returns:
            ★_k(x) as (n_{n-k}, d) or (n_total, d) tensor.
        """

    def matrix(self, k: Optional[int] = None) -> Optional[SparseTensor]:  # pylint: disable=unused-argument
        """Return explicit sparse matrix for ★_k (if available).

        Args:
            k (int | None): form degree, or None for global operator.

        Returns:
            (n_{n-k}, n_k) or (n_total, n_total) SparseTensor, or None.
        """
        return None

    def diagonal(self, k: Optional[int] = None) -> Optional[Tensor]:  # pylint: disable=unused-argument
        """Extract diagonal of ★_k (if diagonal).

        Args:
            k (int | None): form degree, or None for global operator.

        Returns:
            Diagonal entries as (n_k,) or (n_total,) tensor, or None.
        """
        return None


class DiagonalHodgeStar(HodgeStar):
    """Hodge star for diagonal inner products.

    For DEC-style diagonal inner products, the Hodge star is also diagonal
    and can be computed efficiently from the ratio of primal/dual volumes.

    Subclasses should implement _compute_diagonal(k) to compute the
    diagonal entries for degree k.
    """

    def __init__(
        self,
        chain_complex: ChainComplex | ContiguousChainComplex,
        inner_product: InnerProduct,
    ):
        super().__init__(chain_complex, inner_product)
        self._diag_cache: dict[Optional[int], Tensor] = {}

    @abstractmethod
    def _compute_diagonal(self, k: Optional[int] = None) -> Tensor:
        """Compute diagonal entries of ★_k.

        Args:
            k (int | None): form degree. Required for ChainComplex,
                optional for ContiguousChainComplex.

        Returns:
            (n_k,) or (n_total,) tensor of diagonal entries.
        """

    def diagonal(self, k: Optional[int] = None) -> Tensor:
        """Extract diagonal of ★_k.

        Args:
            k (int | None): form degree. Required for ChainComplex,
                optional for ContiguousChainComplex.

        Returns:
            (n_k,) or (n_total,) tensor of diagonal entries.

        Raises:
            ValueError: if k is None but complex is not ContiguousChainComplex.
        """
        if k is None and not self._is_contiguous:
            raise ValueError(
                "k=None (global operator) requires ContiguousChainComplex"
            )
        if k not in self._diag_cache:
            self._diag_cache[k] = self._compute_diagonal(k)
        return self._diag_cache[k]

    def apply(self, x: Tensor, k: Optional[int] = None) -> Tensor:
        """Apply diagonal Hodge star operator.

        Args:
            x (Tensor): (n_k, d) or (n_total, d) tensor of cochain values.
            k (int | None): form degree. Required for ChainComplex,
                optional for ContiguousChainComplex.

        Returns:
            ★_k(x) as (n_{n-k}, d) or (n_total, d) tensor.

        Raises:
            ValueError: if k is None but complex is not ContiguousChainComplex.
        """
        diag = self.diagonal(k)
        if x.ndim == 1:
            return diag * x
        return diag.unsqueeze(-1) * x
