"""Affine geometry for simplicial finite elements.

Provides tools for handling the affine map F_T : Δⁿ → T from the reference
simplex to a physical simplex T in ℝ^m.

For an affine simplex with vertices x_0, ..., x_n in ℝ^m:
    F_T(λ) = Σ λ_i x_i

The Jacobian J_T is constant and given by:
    J_T = [x_1 - x_0, ..., x_n - x_0]  (m × n matrix)

For pullback of differential forms, we need:
    - det(J_T) for volume scaling (n = m case),
    - J_T^{-T} for pulling back covectors (1-forms),
    - appropriate wedge products for higher-degree forms.
"""
# pylint: disable=invalid-name

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import Tensor

if TYPE_CHECKING:
    from ..complex.simplicial import SimplicialChainComplex


class SimplicialGeometry:
    """Geometry data for a simplicial complex embedded in ℝ^m.

    Stores vertex positions and provides Jacobian computation for affine maps
    from reference simplices to physical simplices.

    Args:
        vertex_positions (Tensor): (n_vertices, m) coordinates in ℝ^m.
        top_cells (Tensor): (N_n, n+1) vertex indices of top simplices.
        n (int): topological dimension (simplex degree).
    """

    def __init__(
        self,
        vertex_positions: Tensor,
        top_cells: Tensor,
        n: int,
    ):
        self.vertex_positions = vertex_positions
        self.top_cells = top_cells
        self.n = n
        self.m = vertex_positions.shape[1]

        self._jacobians: Tensor | None = None
        self._det_jacobians: Tensor | None = None
        self._inv_jacobians_T: Tensor | None = None

    @classmethod
    def from_complex(
        cls,
        chain_complex: SimplicialChainComplex,
        vertex_positions: Tensor,
    ) -> SimplicialGeometry:
        """Build geometry from a simplicial complex and vertex positions.

        Args:
            chain_complex (SimplicialChainComplex): simplicial mesh.
            vertex_positions (Tensor): (n_vertices, m) coordinates.

        Returns:
            SimplicialGeometry with top cells from the complex.
        """
        n = chain_complex.dim
        top_cells = chain_complex.cells(n)
        return cls(vertex_positions, top_cells, n)

    def _compute_jacobians(self) -> None:
        """Compute Jacobians for all top simplices (lazy, cached)."""
        if self._jacobians is not None:
            return

        # Vertex coordinates: (N_n, n+1, m)
        verts = self.vertex_positions[self.top_cells]

        # J_T = [x_1 - x_0, ..., x_n - x_0]: (N_n, m, n)
        x0 = verts[:, 0:1, :]  # (N_n, 1, m)
        edges = verts[:, 1:, :] - x0  # (N_n, n, m)
        self._jacobians = edges.transpose(1, 2)  # (N_n, m, n)

    def jacobians(self) -> Tensor:
        """Return Jacobians for all top simplices.

        Returns:
            (N_n, m, n) Jacobian matrices J_T.
        """
        self._compute_jacobians()
        return self._jacobians

    def det_jacobians(self) -> Tensor:
        """Return determinants of Gram matrices for volume scaling.

        For embedded simplices (m ≥ n), the measure scaling factor is:
            |det J_T| = sqrt(det(J_T^T J_T))

        For square Jacobians (m = n), this is just |det(J_T)|.

        Returns:
            (N_n,) determinant values (unsigned).
        """
        if self._det_jacobians is not None:
            return self._det_jacobians

        J = self.jacobians()  # (N_n, m, n)
        _, m, n = J.shape

        if m == n:
            # Square case: det(J)
            sign, logdet = torch.linalg.slogdet(J)
            self._det_jacobians = (sign * torch.exp(logdet)).abs()
        else:
            # Gram determinant: sqrt(det(J^T J))
            G = J.transpose(1, 2) @ J  # (N_n, n, n)
            sign, logdet = torch.linalg.slogdet(G)
            self._det_jacobians = (sign * torch.exp(0.5 * logdet)).abs()

        return self._det_jacobians

    def inv_jacobians_T(self) -> Tensor:
        """Return transpose of inverse Jacobians for pullback of 1-forms.

        For square Jacobians (m = n), computes J_T^{-T}.
        For m > n (embedded), computes the pseudoinverse transpose.

        Returns:
            (N_n, n, m) matrices (J_T^{-1})^T for pulling back covectors.

        Raises:
            ValueError: if n > m (simplex dimension exceeds ambient dimension).
        """
        if self._inv_jacobians_T is not None:
            return self._inv_jacobians_T

        J = self.jacobians()  # (N_n, m, n)
        _, m, n = J.shape

        if n > m:
            raise ValueError(
                f"Cannot compute inverse for n={n} > m={m} (underdetermined)"
            )

        if m == n:
            # Square: direct inverse transpose
            J_inv = torch.linalg.inv(J)  # (N_n, n, n)
            self._inv_jacobians_T = J_inv.transpose(1, 2)  # (N_n, n, n)
        else:
            # Pseudoinverse: (J^T J)^{-1} J^T, then transpose
            G = J.transpose(1, 2) @ J  # (N_n, n, n)
            G_inv = torch.linalg.inv(G)  # (N_n, n, n)
            J_pinv = G_inv @ J.transpose(1, 2)  # (N_n, n, m)
            self._inv_jacobians_T = J_pinv.transpose(1, 2)  # (N_n, m, n)

        return self._inv_jacobians_T

    def volume_scaling(self) -> Tensor:
        """Volume scaling factor for integration over physical simplices.

        Returns:
            (N_n,) factors such that ∫_T f ≈ Σ_q w_q |scaling| f(x_q).
        """
        return self.det_jacobians()

    def to(self, *args, **kwargs) -> SimplicialGeometry:
        """Move all tensors to specified device/dtype."""
        self.vertex_positions = self.vertex_positions.to(*args, **kwargs)
        self.top_cells = self.top_cells.to(*args, **kwargs)

        if self._jacobians is not None:
            self._jacobians = self._jacobians.to(*args, **kwargs)
        if self._det_jacobians is not None:
            self._det_jacobians = self._det_jacobians.to(*args, **kwargs)
        if self._inv_jacobians_T is not None:
            self._inv_jacobians_T = self._inv_jacobians_T.to(*args, **kwargs)

        return self

    def cpu(self) -> SimplicialGeometry:
        """Move to CPU."""
        return self.to("cpu")

    def cuda(self) -> SimplicialGeometry:
        """Move to CUDA."""
        return self.to("cuda")
