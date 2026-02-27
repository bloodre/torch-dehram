"""Discrete Lie derivative operators.

This module defines the abstract interface for discrete Lie derivative
operators ``L_X: C^k -> C^k`` that compute the directional derivative of
differential forms along a fixed vector field X.

The Lie derivative measures how a differential form changes as it is
transported along the flow of a vector field. Unlike the exterior derivative
and interior product, the Lie derivative preserves the degree of forms.

Implementations may use different discretization strategies:
- Cartan formula: L_X = i_X d + d i_X (composition of interior product and exterior derivative)
- Flow-based pullback: differentiate the pullback along the flow of X
- Semi-Lagrangian transport: advect form values along characteristics
- Extrusion/DEC swept volume: integrate over swept regions

The Cartan formula is provided as the default implementation via
``CartanLieDerivative``, which requires an interior product operator
and a chain complex for the exterior derivative.
"""

from abc import ABC, abstractmethod
from torch import Tensor

from .cochain import CoChain
from .interior_product import InteriorProduct
from .complex import ChainComplex, ContiguousChainComplex


class LieDerivative(ABC):
    """Discrete Lie derivative ``L_X`` with a fixed vector field X.

    Provides degree-preserving operators ``L_X: C^k -> C^k`` that compute
    the directional derivative of discrete differential forms along a
    vector field X.

    The Lie derivative satisfies key properties:
    - Linearity: ``L_X(α + β) = L_X(α) + L_X(β)``
    - Leibniz rule on wedge products: ``L_X(α ∧ β) = L_X(α) ∧ β + α ∧ L_X(β)``
    - Commutes with exterior derivative: ``L_X(dα) = d(L_X(α))``

    Implementations should provide:
    - ``apply(alpha)``: compute ``L_X(α)`` for a k-cochain α
    - ``vector_field``: access the underlying vector field X
    """

    @abstractmethod
    def apply(self, alpha: CoChain) -> CoChain:
        """Apply ``L_X`` to a discrete k-form.

        Args:
            alpha (CoChain): Input k-cochain of shape ``(n_k, d)``.

        Returns:
            CoChain: Output k-cochain of shape ``(n_k, d)`` representing
                the Lie derivative ``L_X(α)``.
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


class CartanLieDerivative(LieDerivative):
    """Lie derivative using the Cartan formula: ``L_X = i_X d + d i_X``.

    The Cartan formula expresses the Lie derivative as a composition of
    the interior product and exterior derivative. This is the standard
    discrete implementation when both operators are available.

    For a k-form α:
        L_X(α) = i_X(dα) + d(i_X(α))

    where:
        - d: exterior derivative (coboundary operator)
        - i_X: interior product (contraction) with vector field X

    Args:
        interior_product: Interior product operator ``i_X``.
        chain: Chain complex providing exterior derivative operators.
    """

    def __init__(
        self,
        interior_product: InteriorProduct,
        chain: ChainComplex | ContiguousChainComplex,
    ):
        self.interior_product = interior_product
        self.chain = chain

    def apply(self, alpha: CoChain) -> CoChain:
        """Apply ``L_X`` to a discrete k-form using the Cartan formula.

        Computes ``L_X(α) = i_X(dα) + d(i_X(α))`` by:
        1. Computing ``i_X(α)`` → (k-1)-form
        2. Computing ``d(i_X(α))`` → k-form
        3. Computing ``dα`` → (k+1)-form
        4. Computing ``i_X(dα)`` → k-form
        5. Summing the results

        Args:
            alpha: Input k-cochain of shape ``(n_k, d)``.

        Returns:
            Output k-cochain of shape ``(n_k, d)`` representing ``L_X(α)``.
        """
        # i_X(alpha) -> (k-1)-form
        interior_result = self.interior_product.apply(alpha)

        # d(i_X(alpha)) -> k-form
        exterior_result = self.chain.d(alpha.k - 1) @ interior_result

        # d(alpha) -> (k+1)-form
        exterior_alpha = self.chain.d(alpha.k) @ alpha

        # i_X(d(alpha)) -> k-form
        interior_exterior = self.interior_product.apply(exterior_alpha)

        # L_X(alpha) = d(i_X(alpha)) + i_X(d(alpha))
        return exterior_result + interior_exterior

    def _vector_field(self) -> CoChain | Tensor:
        return self.interior_product.vector_field()
