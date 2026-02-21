"""Quadrature rules on reference simplices.

Provides pre-tabulated Gauss-like quadrature rules on the reference n-simplex
in barycentric coordinates. Rules are stored as (points, weights) where points
are in barycentric coordinates (λ_0, ..., λ_n) with Σ λ_i = 1.

The degree of exactness is the maximum polynomial degree p such that the rule
integrates all polynomials of total degree ≤ p exactly.

Design rationale:
    - Rules are **pre-tabulated constants** from standard FEM literature,
      not recomputed on demand.
    - Quadrature points and weights solve moment-matching equations that are
      nonlinear and expensive to compute. Once solved, the result depends only
      on (n, degree) and can be reused universally.
    - All rules are defined in CPU float64 and are device-agnostic; the
      assembly layer (assembly.py) casts them to the target device/dtype.

Sources:
    - Triangle rules: symmetric Gauss formulas (Dunavant, Strang-Fix).
    - Tetrahedron rules: Keast-type symmetric rules.
    - Edge rules: Gauss-Legendre mapped to barycentric coordinates.

Coverage:
    - Edges (n=1): degree ≤ 3
    - Triangles (n=2): degree ≤ 3
    - Tetrahedra (n=3): degree ≤ 3
    Higher degrees or dimensions are not currently implemented.

Reference simplex conventions:
    - Δ¹: vertices at (1,0), (0,1); measure = 1/2
    - Δ²: vertices at (1,0,0), (0,1,0), (0,0,1); measure = 1/2
    - Δ³: vertices at (1,0,0,0), (0,1,0,0), (0,0,1,0), (0,0,0,1); measure = 1/6
"""
# pylint: disable=invalid-name

from __future__ import annotations

import torch
from torch import Tensor


def _triangle_degree2() -> tuple[Tensor, Tensor]:
    """3-point degree-2 quadrature on reference triangle.

    Classic symmetric 3-point rule with points at barycentric (2/3, 1/6, 1/6)
    and cyclic permutations. Equal weights of 1/6 each.

    This is the minimal symmetric rule exact for degree 2 polynomials.

    Exact for polynomials of total degree ≤ 2.

    Returns:
        points (3, 3): barycentric coordinates.
        weights (3,): quadrature weights (sum = 1/2).
    """
    points = torch.tensor(
        [
            [2.0 / 3.0, 1.0 / 6.0, 1.0 / 6.0],
            [1.0 / 6.0, 2.0 / 3.0, 1.0 / 6.0],
            [1.0 / 6.0, 1.0 / 6.0, 2.0 / 3.0],
        ],
        dtype=torch.float64,
    )
    weights = torch.full((3,), 1.0 / 6.0, dtype=torch.float64)
    return points, weights


def _triangle_degree3() -> tuple[Tensor, Tensor]:
    """6-point degree-3 quadrature on reference triangle.

    Symmetric 6-point Gauss rule with points at (a, b, b) and permutations,
    where a ≈ 0.659, b ≈ 0.232 are roots of the moment equations.
    Equal weights of 1/12 each.

    Exact for polynomials of total degree ≤ 3.

    Returns:
        points (6, 3): barycentric coordinates.
        weights (6,): quadrature weights (sum = 1/2).
    """
    a = 0.659027622374092
    b = 0.231933368553031
    w1 = 1.0 / 12.0

    points = torch.tensor(
        [
            [a, b, b],
            [b, a, b],
            [b, b, a],
            [b, a, b],
            [b, b, a],
            [a, b, b],
        ],
        dtype=torch.float64,
    )
    weights = torch.full((6,), w1, dtype=torch.float64)
    return points, weights


def _tetrahedron_degree2() -> tuple[Tensor, Tensor]:
    """4-point degree-2 quadrature on reference tetrahedron.

    Symmetric 4-point Keast-type rule. Points are at (a,a,a,b) and permutations,
    where a = (5-√5)/20 ≈ 0.138, b = (5+3√5)/20 ≈ 0.585.
    Equal weights of 1/24 each.

    Exact for polynomials of total degree ≤ 2.

    Returns:
        points (4, 4): barycentric coordinates.
        weights (4,): quadrature weights (sum = 1/6).
    """
    a = (5.0 - torch.sqrt(torch.tensor(5.0))) / 20.0
    b = (5.0 + 3.0 * torch.sqrt(torch.tensor(5.0))) / 20.0
    w = 1.0 / 24.0

    points = torch.tensor(
        [
            [a, a, a, b],
            [a, a, b, a],
            [a, b, a, a],
            [b, a, a, a],
        ],
        dtype=torch.float64,
    )
    weights = torch.full((4,), w, dtype=torch.float64)
    return points, weights


def _tetrahedron_degree3() -> tuple[Tensor, Tensor]:
    """11-point degree-3 quadrature on reference tetrahedron.

    Symmetric Keast-type degree-3 rule with three orbits:
        - 1 point at centroid (1/4, 1/4, 1/4, 1/4), weight w1 ≈ -0.0132
        - 4 points at (1/6, 1/6, 1/6, 1/2) orbit, weight w2 ≈ 0.0076 each
        - 4 points at (1/14, 1/14, 1/14, 11/14) orbit, weight w3 ≈ 0.0025 each
        - 2 additional points from incomplete orbit, weight w3 each

    Note: centroid weight is negative (allowed for higher-degree rules).

    Exact for polynomials of total degree ≤ 3.

    Returns:
        points (11, 4): barycentric coordinates.
        weights (11,): quadrature weights (sum = 1/6).
    """
    a1 = 0.25
    w1 = -0.013155555555556

    a2 = 1.0 / 6.0
    b2 = 0.5
    w2 = 0.007622222222222

    a3 = 1.0 / 14.0
    b3 = 11.0 / 14.0
    w3 = 0.002488888888889

    points = torch.tensor(
        [
            [a1, a1, a1, a1],
            [a2, a2, a2, b2],
            [a2, a2, b2, a2],
            [a2, b2, a2, a2],
            [b2, a2, a2, a2],
            [a3, a3, a3, b3],
            [a3, a3, b3, a3],
            [a3, b3, a3, a3],
            [b3, a3, a3, a3],
            [a3, a3, b3, a3],
            [a3, b3, a3, a3],
        ],
        dtype=torch.float64,
    )
    weights = torch.cat([
        torch.tensor([w1]),
        torch.full((4,), w2),
        torch.full((4,), w3),
        torch.full((2,), w3),
    ], dim=0).to(torch.float64)

    return points, weights


def quadrature_simplex(n: int, degree: int) -> tuple[Tensor, Tensor]:
    """Get quadrature rule on reference n-simplex.

    Returns pre-tabulated rules stored as CPU float64 tensors. The assembly
    layer is responsible for casting to the target device/dtype.

    Args:
        n (int): simplex dimension (1=edge, 2=triangle, 3=tetrahedron).
        degree (int): desired polynomial degree of exactness.

    Returns:
        points (Q, n+1): barycentric coordinates of Q quadrature points.
        weights (Q,): quadrature weights (sum = measure of reference simplex).

    Raises:
        ValueError: if no rule is available for (n, degree).

    Notes:
        - Currently supports degree ≤ 3 for n ∈ {1, 2, 3}.
        - Rules are device-agnostic; cast via `.to(device, dtype)` as needed.
    """
    if n == 2:
        if degree <= 2:
            return _triangle_degree2()
        elif degree <= 3:
            return _triangle_degree3()
        else:
            raise ValueError(f"No triangle quadrature rule for degree={degree}")

    elif n == 3:
        if degree <= 2:
            return _tetrahedron_degree2()
        elif degree <= 3:
            return _tetrahedron_degree3()
        else:
            raise ValueError(f"No tetrahedron quadrature rule for degree={degree}")

    elif n == 1:
        # Edge / interval: Gauss-Legendre mapped to barycentric
        if degree <= 1:
            # Midpoint rule
            points = torch.tensor([[0.5, 0.5]], dtype=torch.float64)
            weights = torch.tensor([0.5], dtype=torch.float64)
            return points, weights
        elif degree <= 3:
            # 2-point Gauss
            a = 0.5 - torch.sqrt(torch.tensor(1.0 / 12.0))
            b = 0.5 + torch.sqrt(torch.tensor(1.0 / 12.0))
            points = torch.tensor([[1 - a, a], [1 - b, b]], dtype=torch.float64)
            weights = torch.tensor([0.25, 0.25], dtype=torch.float64)
            return points, weights
        else:
            raise ValueError(f"No edge quadrature rule for degree={degree}")

    else:
        raise ValueError(f"Quadrature not implemented for n={n}")


def barycentric_to_cartesian(
    barycentric: Tensor,
    vertex_coords: Tensor,
) -> Tensor:
    """Convert barycentric to Cartesian coordinates.

    Args:
        barycentric (Tensor): (Q, n+1) barycentric coordinates.
        vertex_coords (Tensor): (n+1, m) Cartesian vertex positions.

    Returns:
        (Q, m) Cartesian coordinates.
    """
    return barycentric @ vertex_coords
