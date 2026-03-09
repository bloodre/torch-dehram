"""Quadrature rules for numerical integration over simplices.

This module provides quadrature rules for various simplex types and dimensions:

- 1-simplex (edges): Gauss-Legendre rules
- 2-simplex (triangles): Dunavant, Strang-Fix, and Xiao-Gimbutas rules
- 3-simplex (tetrahedra): Keast and Xiao-Gimbutas rules
- n-simplex (n ≥ 4): Grundmann-Möller and symmetric rules

The quadrature rules are organized into separate modules:

- edges.py: 1D Gauss-Legendre quadrature
- triangles.py: 2D triangle quadrature rules
- tetrahedron.py: 3D tetrahedron quadrature rules
- xiao_gimbutas.py: Modern optimal rules for 2D/3D
- simplex_nd.py: Symmetric rules for high dimensions
- grundmann_moeller.py: Algorithmic rules for arbitrary dimensions
- base.py: Base Quadrature class and AdaptiveIntegrator

All rules use barycentric coordinates for the reference simplex with vertices at:
- 1-simplex: (1,0), (0,1)
- 2-simplex: (1,0,0), (0,1,0), (0,0,1)
- 3-simplex: (1,0,0,0), (0,1,0,0), (0,0,1,0), (0,0,0,1)
- n-simplex: (0,...,0,1,0,...,0) with 1 at each position

Example usage:
    >>> from torch_derham.core.feec.quadrature import (
    ...     xiao_gimbutas_triangle_degree_5,
    ...     grundmann_moeller_rule,
    ...     AdaptiveIntegrator
    ... )
    >>>
    >>> # Get Xiao-Gimbutas triangle rule
    >>> points, weights = xiao_gimbutas_triangle_degree_5()
    >>>
    >>> # Get Grundmann-Möller rule for 4-simplex
    >>> points, weights = grundmann_moeller_rule(dimension=4, degree=3)
    >>>
    >>> # Create adaptive integrator
    >>> integrator = AdaptiveIntegrator(points, weights)
"""

from __future__ import annotations

# Base classes
from .base import Quadrature, AdaptiveIntegrator

# 1D rules
from .edges import (
    gauss_legendre_degree_1,
    gauss_legendre_degree_3,
    gauss_legendre_degree_5,
    gauss_legendre_degree_7,
    gauss_legendre_degree_9,
)

# 2D rules
from .triangles import (
    dunavant_degree_1,
    dunavant_degree_2,
    dunavant_degree_3,
    dunavant_degree_4,
    dunavant_degree_5,
    strang_fix_degree_2,
    strang_fix_degree_3,
)

# 3D rules
from .tetrahedron import (
    keast_degree_1,
    keast_degree_2,
    keast_degree_3,
    keast_degree_4,
    keast_degree_5,
)

# Modern optimal rules (2D/3D)
from .xiao_gimbutas import (
    xiao_gimbutas_triangle_degree_1,
    xiao_gimbutas_triangle_degree_2,
    xiao_gimbutas_triangle_degree_3,
    xiao_gimbutas_triangle_degree_4,
    xiao_gimbutas_triangle_degree_5,
    xiao_gimbutas_tetrahedron_degree_1,
    xiao_gimbutas_tetrahedron_degree_2,
    xiao_gimbutas_tetrahedron_degree_3,
    xiao_gimbutas_tetrahedron_degree_4,
    xiao_gimbutas_tetrahedron_degree_5,
)

# High-dimensional rules
from .simplex_nd import (
    simplex_4d_degree_1,
    simplex_4d_degree_2,
    simplex_4d_degree_3,
    simplex_5d_degree_1,
    simplex_5d_degree_2,
    simplex_5d_degree_3,
)

# Algorithmic rules
from .grundmann_moeller import (
    grundmann_moeller_points,
    grundmann_moeller_weights,
    grundmann_moeller_rule,
    grundmann_moeller_info,
)

__all__ = [
    # Base classes
    "Quadrature",
    "AdaptiveIntegrator",

    # 1D rules
    "gauss_legendre_degree_1",
    "gauss_legendre_degree_3",
    "gauss_legendre_degree_5",
    "gauss_legendre_degree_7",
    "gauss_legendre_degree_9",

    # 2D rules
    "dunavant_degree_1",
    "dunavant_degree_2",
    "dunavant_degree_3",
    "dunavant_degree_4",
    "dunavant_degree_5",
    "strang_fix_degree_2",
    "strang_fix_degree_3",

    # 3D rules
    "keast_degree_1",
    "keast_degree_2",
    "keast_degree_3",
    "keast_degree_4",
    "keast_degree_5",

    # Modern optimal rules
    "xiao_gimbutas_triangle_degree_1",
    "xiao_gimbutas_triangle_degree_2",
    "xiao_gimbutas_triangle_degree_3",
    "xiao_gimbutas_triangle_degree_4",
    "xiao_gimbutas_triangle_degree_5",
    "xiao_gimbutas_tetrahedron_degree_1",
    "xiao_gimbutas_tetrahedron_degree_2",
    "xiao_gimbutas_tetrahedron_degree_3",
    "xiao_gimbutas_tetrahedron_degree_4",
    "xiao_gimbutas_tetrahedron_degree_5",

    # High-dimensional rules
    "simplex_4d_degree_1",
    "simplex_4d_degree_2",
    "simplex_4d_degree_3",
    "simplex_5d_degree_1",
    "simplex_5d_degree_2",
    "simplex_5d_degree_3",

    # Algorithmic rules
    "grundmann_moeller_points",
    "grundmann_moeller_weights",
    "grundmann_moeller_rule",
    "grundmann_moeller_info",
]
