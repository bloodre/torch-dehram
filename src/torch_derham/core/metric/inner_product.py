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
    - apply(k, x): compute M_k @ x
    - solve(k, b): solve M_k @ x = b for x

    Optionally may provide:
    - diagonal(k): extract diagonal of M_k (for diagonal operators)
    - matrix(k): explicit sparse matrix (for assembly)
    """

    @abstractmethod
    def apply(self, k: int, x: Tensor) -> Tensor:
        """Apply inner product operator M_k to vector x.

        Args:
            k (int): cochain degree.
            x (Tensor): (n_k, d) tensor of cochain values.

        Returns:
            M_k @ x as (n_k, d) tensor.
        """

    @abstractmethod
    def solve(self, k: int, b: Tensor) -> Tensor:
        """Solve M_k @ x = b for x.

        Args:
            k (int): cochain degree.
            b (Tensor): (n_k, d) tensor of right-hand side.

        Returns:
            x as (n_k, d) tensor satisfying M_k @ x = b.
        """

    def diagonal(self, _k: int) -> Optional[Tensor]:
        """Extract diagonal of M_k (if available).

        Args:
            _k (int): cochain degree.

        Returns:
            (n_k,) tensor of diagonal entries, or None if not available.
        """
        return None

    def matrix(self, _k: int) -> Optional[SparseTensor]:
        """Return explicit sparse matrix M_k (if available).

        Args:
            _k (int): cochain degree.

        Returns:
            (n_k, n_k) SparseTensor, or None if not available.
        """
        return None
