"""Discrete exterior calculus operators derived from inner products.

This module provides the DiscreteCalculus class which combines a chain
complex (providing differential operators d_k) with an inner product
(providing metric M_k) to derive adjoint and Laplacian operators.
"""

from __future__ import annotations

from typing import Optional

from torch import Tensor
from torch_sparse import SparseTensor

from ..complex.chain import ChainComplex, ContiguousChainComplex
from .inner_product import InnerProduct


class DiscreteCalculus:
    """Discrete exterior calculus combining differential and metric structure.

    Given a chain complex (d_k operators) and an inner product (M_k operators),
    this class provides derived operators:
    - d_k: exterior derivative (coboundary)
    - δ_k: codifferential (adjoint of d)
    - Δ_k: Hodge-Laplacian

    The codifferential is defined as:
        δ_k = M_{k-1}^{-1} d_{k-1}^T M_k

    The Hodge-Laplacian is:
        Δ_k = δ_{k+1} d_k + d_{k-1} δ_k

    Global operator support (k=None):
    When k=None, operators work on the full graded space. This is useful
    for ContiguousChainComplex representations.

    Args:
        chain_complex (ChainComplex | ContiguousChainComplex): chain complex
            providing boundary operators.
        inner_product (InnerProduct): inner product providing metric M_k.
    """

    def __init__(
        self,
        chain_complex: ChainComplex | ContiguousChainComplex,
        inner_product: InnerProduct,
    ):
        self.chain_complex = chain_complex
        self.inner_product = inner_product
        self._is_contiguous = isinstance(chain_complex, ContiguousChainComplex)

    @property
    def dim(self) -> int:
        """Maximum degree of the chain complex."""
        return self.chain_complex.dim

    def d(self, k: Optional[int] = None) -> SparseTensor:
        """Exterior derivative (coboundary) operator d_k: C^k → C^{k+1}.

        Args:
            k (int | None): cochain degree. Required for ChainComplex,
                optional for ContiguousChainComplex (None gives global operator).

        Returns:
            Sparse matrix d_k of shape (n_{k+1}, n_k) or global operator.

        Raises:
            ValueError: if k is None but complex is not ContiguousChainComplex.
        """
        if k is None:
            if not self._is_contiguous:
                raise ValueError(
                    "k=None (global operator) requires ContiguousChainComplex"
                )
            return self.chain_complex.data
        return self.chain_complex.coboundary(k)

    def delta_apply(self, x: Tensor, k: Optional[int] = None) -> Tensor:
        """Apply codifferential operator δ_k: C^k → C^{k-1}.

        The codifferential is the adjoint of d with respect to the inner product:
            δ_k = M_{k-1}^{-1} d_{k-1}^T M_k

        For ContiguousChainComplex with k=None, applies δ across full graded space.

        Args:
            x (Tensor): (n_k, d) or (n_total, d) tensor of cochain values.
            k (int | None): cochain degree (k > 0). Required for ChainComplex,
                optional for ContiguousChainComplex.

        Returns:
            δ_k(x) as (n_{k-1}, d) or (n_total, d) tensor.

        Raises:
            ValueError: if k is None but complex is not ContiguousChainComplex,
                or if k is int and k <= 0.
        """
        if k is None and not self._is_contiguous:
            raise ValueError(
                "k=None (global operator) requires ContiguousChainComplex"
            )
        if k is not None and k <= 0:
            raise ValueError(f"codifferential δ_k requires k > 0, got k={k}")

        y = self.inner_product.apply(k, x)
        if k is None:
            d_op = self.d(None)
        else:
            d_op = self.d(k - 1)
        z = d_op.t() @ y
        if k is None:
            return self.inner_product.solve(None, z)
        return self.inner_product.solve(k - 1, z)

    def laplacian_apply(self, x: Tensor, k: Optional[int] = None) -> Tensor:
        """Apply Hodge-Laplacian operator Δ_k: C^k → C^k.

        The Hodge-Laplacian is defined as:
            Δ_k = δ_{k+1} d_k + d_{k-1} δ_k

        For ContiguousChainComplex with k=None, applies Δ across full graded space.

        Args:
            x (Tensor): (n_k, d) or (n_total, d) tensor of cochain values.
            k (int | None): cochain degree. Required for ChainComplex,
                optional for ContiguousChainComplex.

        Returns:
            Δ_k(x) as (n_k, d) or (n_total, d) tensor.

        Raises:
            ValueError: if k is None but complex is not ContiguousChainComplex.
        """
        if k is None and not self._is_contiguous:
            raise ValueError(
                "k=None (global operator) requires ContiguousChainComplex"
            )

        out = x.new_zeros(x.shape)

        if k is None:
            d_op = self.d(None)
            dx = d_op @ x
            out = out + self.delta_apply(dx)
            delta_x = self.delta_apply(x)
            out = out + d_op.t() @ delta_x
        else:
            if k < self.dim:
                dx = self.d(k) @ x
                out = out + self.delta_apply(dx, k + 1)

            if k > 0:
                delta_x = self.delta_apply(x, k)
                out = out + self.d(k - 1) @ delta_x

        return out
