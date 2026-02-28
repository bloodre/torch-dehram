"""Discrete interior product operators.

This module defines the abstract interface for discrete interior product
operators ``i_X: C^k -> C^{k-1}`` that contract differential forms with
a fixed vector field X.

The interior product (also called contraction) is a fundamental operation
in differential geometry that reduces the degree of a differential form
by one. Given a vector field X and a k-form α, the interior product i_X(α)
is a (k-1)-form defined pointwise by:

    (i_X α)(v_1, ..., v_{k-1}) = α(X, v_1, ..., v_{k-1})

Implementations should provide:
- apply(alpha): compute i_X(α) for a k-cochain α
- vector_field: access the underlying vector field X
"""
from abc import ABC, abstractmethod
from typing import Callable

from torch import Tensor

from .cochain import CoChain


class InteriorProduct(ABC):
    """
    Discrete interior product i_X with a fixed vector field X.

    Provides operators i_X : k-forms -> (k-1)-forms
    """

    @abstractmethod
    def apply(self, alpha: CoChain) -> CoChain:
        """
        Apply i_X to a discrete k-form.

        Args:
            alpha (CoChain): cochain of degree k of shape (n_k, d)

        Returns:
            CoChain: cochain of degree k-1 of shape (n_{k-1}, d)
        """

    @abstractmethod
    def _vector_field(self) -> CoChain | Tensor:
        """
        Return the underlying discrete vector field X, which
        is either a CoChain or a Tensor.

        When CoChain is used, the degree should be 0.
        """
        raise NotImplementedError


    def vector_field(self) -> CoChain | Tensor:
        """
        Return the underlying discrete vector field X.

        Returns:
            CoChain | Tensor: the vector field X
        """
        if isinstance(self._vector_field(), CoChain):
            if self._vector_field().k != 0:
                raise ValueError("When CoChain is used, the degree should be 0.")
        return self._vector_field()


InteriorProductBuilder = Callable[[CoChain | Tensor], InteriorProduct]
"""Type alias for a factory that builds an InteriorProduct from a vector field.

A builder takes a vector field (as a 0-cochain or tensor) and returns
an InteriorProduct operator configured for that vector field.
"""
