"""Inner product abstraction for discrete differential forms.

This module defines the base interface for inner products on k-cochains.
An inner product on k-cochains defines <x,y>_k = x^T M_k y where M_k
is a symmetric positive definite operator.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

from torch import Tensor
from torch_sparse import SparseTensor


class InnerProduct(ABC):
    """Abstract base class for inner products on k-cochains.

    An inner product on k-cochains defines:
        <x, y>_k = x^T M_k y

    where M_k is a symmetric positive definite operator.

    Implementations must provide:
    - apply(x, k): compute M_k @ x for specific degree k
    - solve(b, k): solve M_k @ x = b for x

    Global operator support (k=None):
    When k=None, methods operate on the full graded space (all degrees).
    This is useful for ContiguousChainComplex where all k-cells are indexed
    in a single global space. Implementations may raise NotImplementedError
    if global operations are not supported.

    Optionally may provide:
    - diagonal(k): extract diagonal of M_k (for diagonal operators)
    - matrix(k): explicit sparse matrix (for assembly)
    """

    @abstractmethod
    def apply(self, x: Tensor, k: Optional[int] = None) -> Tensor:
        """Apply inner product operator M_k to vector x.

        Args:
            x (Tensor): (n_k, d) or (n_total, d) tensor of cochain values.
            k (int | None): cochain degree, or None for global operator.

        Returns:
            M_k @ x as (n_k, d) or (n_total, d) tensor.
        """

    @abstractmethod
    def solve(self, b: Tensor, k: Optional[int] = None) -> Tensor:
        """Solve M_k @ x = b for x.

        Args:
            b (Tensor): (n_k, d) or (n_total, d) tensor of right-hand side.
            k (int | None): cochain degree, or None for global operator.

        Returns:
            x as (n_k, d) or (n_total, d) tensor satisfying M_k @ x = b.
        """

    def diagonal(self, k: Optional[int] = None) -> Optional[Tensor]:  # pylint: disable=unused-argument
        """Extract diagonal of M_k (if available).

        Args:
            k (int | None): cochain degree, or None for global operator.

        Returns:
            (n_k,) or (n_total,) tensor of diagonal entries, or None if not available.
        """
        return None

    def matrix(self, k: Optional[int] = None) -> Optional[SparseTensor]:  # pylint: disable=unused-argument
        """Return explicit sparse matrix M_k (if available).

        Args:
            k (int | None): cochain degree, or None for global operator.

        Returns:
            (n_k, n_k) or (n_total, n_total) SparseTensor, or None if not available.
        """
        return None
