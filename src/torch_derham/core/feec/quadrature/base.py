"""Quadrature rule for numerical integration.

Provides a simple interface for evaluating functions at quadrature points
and computing weighted integrals. The quadrature points and weights are
stored as tensors, allowing for batched operations on GPU.
"""

from __future__ import annotations

from typing import Callable

import torch

from torch import Tensor, nn


class Quadrature:
    """Quadrature rule for numerical integration.

    Stores quadrature points and weights for a domain, with methods
    to evaluate functions at quadrature points and compute weighted integrals.

    Args:
        points: Tensor of shape (n_points, dim) containing quadrature points.
        weights: Tensor of shape (n_points,) containing quadrature weights.

    Attributes:
        points: Quadrature points tensor.
        weights: Quadrature weights tensor.
        n_points: Number of quadrature points.
        dim: Dimension of the points.
    """

    def __init__(self, points: Tensor, weights: Tensor):
        """Initialize quadrature rule with points and weights."""
        self._points = points
        self._weights = weights

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def points(self) -> Tensor:
        """Get quadrature points."""
        return self._points

    @property
    def weights(self) -> Tensor:
        """Get quadrature weights."""
        return self._weights

    @property
    def n_points(self) -> int:
        """Get number of quadrature points."""
        return self._points.size(0)

    @property
    def dim(self) -> int:
        """Get dimension of the points."""
        return self._points.size(1)

    # ------------------------------------------------------------------
    # Evaluation and integration
    # ------------------------------------------------------------------

    def evaluate(self, f: Callable[[Tensor], Tensor]) -> Tensor:
        """Evaluate function at quadrature points.

        Args:
            f: Function that takes points tensor and returns values.

        Returns:
            Function values evaluated at quadrature points.
        """
        return f(self._points)

    def integrate(self, values: Tensor) -> Tensor:
        """Compute weighted integral of values at quadrature points.

        Args:
            values: Tensor of values at quadrature points. Shape (n_points, ...).

        Returns:
            Weighted sum of values (approximation of integral).
        """
        return (values * self._weights).sum(dim=0)

    # ------------------------------------------------------------------
    # Device movement
    # ------------------------------------------------------------------

    def to(self, target) -> "Quadrature":
        """Move quadrature rule to target device/dtype."""
        self._points = self._points.to(target)
        self._weights = self._weights.to(target)
        return self

    def cpu(self) -> "Quadrature":
        """Move quadrature rule to CPU."""
        return self.to("cpu")

    def cuda(self) -> "Quadrature":
        """Move quadrature rule to CUDA."""
        return self.to("cuda")

    def half(self) -> "Quadrature":
        """Move quadrature rule to half precision."""
        return self.to(torch.float16)

    def float(self) -> "Quadrature":
        """Move quadrature rule to float precision."""
        return self.to(torch.float32)

    def double(self) -> "Quadrature":
        """Move quadrature rule to double precision."""
        return self.to(torch.float64)


class AdaptiveIntegrator(Quadrature, nn.Module):
    """Integration with learnable weights.

    Inherits quadrature interface from Quadrature but makes weights learnable
    through nn.Parameter. Points remain fixed (e.g., from Xiao-Gimbutas rules).

    Args:
        points: Fixed quadrature points (e.g., from Xiao-Gimbutas).
        initial_weights: Initial values for learnable weights.

    Note:
        This class is ONNX-compatible as it doesn't create new objects during
        forward passes - weights are static parameters that can be optimized.
    """

    def __init__(self, points: Tensor, initial_weights: Tensor):
        """Initialize adaptive integrator with learnable weights."""
        # Initialize with points and initial weights
        super().__init__(points, initial_weights)
        # Override weights with learnable parameter
        self._weights = nn.Parameter(initial_weights)

    @property
    def weights(self) -> nn.Parameter:
        """Get learnable weight parameters."""
        return self._weights

    def forward(self, values: Tensor) -> Tensor:
        """Compute weighted integral of values at quadrature points.

        Args:
            values: Tensor of values at quadrature points.

        Returns:
            Weighted sum of values (approximation of integral).

        Example:
            >>> integrator = AdaptiveIntegrator(points, initial_weights)
            >>> values = integrator.evaluate(f)
            >>> result = integrator(values)
        """
        return self.integrate(values)

    @classmethod
    def from_quadrature(cls, quadrature: Quadrature) -> "AdaptiveIntegrator":
        """Create adaptive integrator from existing quadrature rule."""
        return cls(quadrature.points, quadrature.weights)

    def to_quadrature(self) -> Quadrature:
        """Convert adaptive integrator back to fixed quadrature."""
        return Quadrature(self.points, self.weights.detach().clone())

    @classmethod
    def from_points(cls, points: Tensor) -> "AdaptiveIntegrator":
        """Create adaptive integrator with uniform weights from points."""
        weights = torch.ones_like(points[..., 0])
        return cls(points, weights)
