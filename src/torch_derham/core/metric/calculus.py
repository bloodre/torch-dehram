"""Discrete exterior calculus operators derived from inner products.

This module provides the DiscreteCalculus class which combines a chain
complex (providing differential operators d_k) with an inner product
(providing metric M_k) to derive adjoint and Laplacian operators.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from torch import Tensor
from torch_sparse import SparseTensor

if TYPE_CHECKING:
    from ..complex.chain import ChainComplex
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

    Args:
        complex (ChainComplex): chain complex providing boundary operators.
        inner_product (InnerProduct): inner product providing metric M_k.
    """

    def __init__(
        self,
        chain_complex: ChainComplex,
        inner_product: InnerProduct,
    ):
        self.chain_complex = chain_complex
        self.inner_product = inner_product

    @property
    def dim(self) -> int:
        """Maximum degree of the chain complex."""
        return self.chain_complex.dim

    def d(self, k: int) -> SparseTensor:
        """Exterior derivative (coboundary) operator d_k: C^k → C^{k+1}.

        Args:
            k (int): cochain degree.

        Returns:
            Sparse matrix d_k of shape (n_{k+1}, n_k).
        """
        return self.chain_complex.coboundary(k)

    def delta_apply(self, k: int, x: Tensor) -> Tensor:
        """Apply codifferential operator δ_k: C^k → C^{k-1}.

        The codifferential is the adjoint of d with respect to the inner product:
            δ_k = M_{k-1}^{-1} d_{k-1}^T M_k

        Args:
            k (int): cochain degree (k > 0).
            x (Tensor): (n_k, d) tensor of k-cochain values.

        Returns:
            δ_k(x) as (n_{k-1}, d) tensor.

        Raises:
            ValueError: if k <= 0.
        """
        if k <= 0:
            raise ValueError(f"codifferential δ_k requires k > 0, got k={k}")

        y = self.inner_product.apply(k, x)
        d_km1 = self.d(k - 1)
        z = d_km1.t() @ y
        return self.inner_product.solve(k - 1, z)

    def laplacian_apply(self, k: int, x: Tensor) -> Tensor:
        """Apply Hodge-Laplacian operator Δ_k: C^k → C^k.

        The Hodge-Laplacian is defined as:
            Δ_k = δ_{k+1} d_k + d_{k-1} δ_k

        Args:
            k (int): cochain degree.
            x (Tensor): (n_k, d) tensor of k-cochain values.

        Returns:
            Δ_k(x) as (n_k, d) tensor.
        """
        out = x.new_zeros(x.shape)

        if k < self.dim:
            dx = self.d(k) @ x
            out = out + self.delta_apply(k + 1, dx)

        if k > 0:
            delta_x = self.delta_apply(k, x)
            out = out + self.d(k - 1) @ delta_x

        return out
