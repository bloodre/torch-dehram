"""Gauss-Legendre quadrature rules for edges (1-simplex).

Provides pre-tabulated Gauss-Legendre quadrature points and weights for
numerical integration on the reference edge [0, 1] in barycentric coordinates.

Sources:
    - Wikipedia: Gauss-Legendre quadrature
      https://en.wikipedia.org/wiki/Gauss%E2%80%93Legendre_quadrature
    - Wolfram MathWorld: Legendre-Gauss Quadrature
      https://mathworld.wolfram.com/Legendre-GaussQuadrature.html
    - Numerical tables from Abramowitz and Stegun

Reference:
    Abramowitz, M. and Stegun, I. A., Handbook of Mathematical Functions,
    Dover Publications, 1972, Section 25.4.
"""

from __future__ import annotations


def gauss_legendre_degree_1() -> tuple[list[list[float]], list[float]]:
    """1-point Gauss-Legendre rule (midpoint).

    Exact for polynomials of degree 1.

    Returns:
        points: List of barycentric coordinates [[λ₀, λ₁]].
        weights: List of quadrature weights.
    """
    points = [
        [0.5, 0.5],
    ]
    weights = [0.5]
    return points, weights


def gauss_legendre_degree_3() -> tuple[list[list[float]], list[float]]:
    """2-point Gauss-Legendre rule.

    Exact for polynomials of degree 3.

    Points at x = (1 ± 1/√3) / 2 in [0,1].

    Returns:
        points: List of barycentric coordinates [[λ₀, λ₁], ...].
        weights: List of quadrature weights.
    """
    a = 0.5 - 0.5 / 1.732050807568877  # (1 - 1/√3) / 2
    b = 0.5 + 0.5 / 1.732050807568877  # (1 + 1/√3) / 2

    points = [
        [1.0 - a, a],
        [1.0 - b, b],
    ]
    weights = [0.25, 0.25]
    return points, weights


def gauss_legendre_degree_5() -> tuple[list[list[float]], list[float]]:
    """3-point Gauss-Legendre rule.

    Exact for polynomials of degree 5.

    Points at x = (1 ± √(3/5)) / 2 and x = 1/2 in [0,1].

    Returns:
        points: List of barycentric coordinates [[λ₀, λ₁], ...].
        weights: List of quadrature weights.
    """
    a = 0.5 - 0.5 * 0.774596669241483  # (1 - √(3/5)) / 2
    b = 0.5
    c = 0.5 + 0.5 * 0.774596669241483  # (1 + √(3/5)) / 2

    points = [
        [1.0 - a, a],
        [1.0 - b, b],
        [1.0 - c, c],
    ]
    w1 = 5.0 / 36.0
    w2 = 4.0 / 18.0
    weights = [w1, w2, w1]
    return points, weights


def gauss_legendre_degree_7() -> tuple[list[list[float]], list[float]]:
    """4-point Gauss-Legendre rule.

    Exact for polynomials of degree 7.

    Returns:
        points: List of barycentric coordinates [[λ₀, λ₁], ...].
        weights: List of quadrature weights.
    """
    x1 = 0.5 - 0.5 * 0.861136311594053
    x2 = 0.5 - 0.5 * 0.339981043584856
    x3 = 0.5 + 0.5 * 0.339981043584856
    x4 = 0.5 + 0.5 * 0.861136311594053

    points = [
        [1.0 - x1, x1],
        [1.0 - x2, x2],
        [1.0 - x3, x3],
        [1.0 - x4, x4],
    ]

    w1 = 0.5 * 0.347854845137454
    w2 = 0.5 * 0.652145154862546
    weights = [w1, w2, w2, w1]
    return points, weights


def gauss_legendre_degree_9() -> tuple[list[list[float]], list[float]]:
    """5-point Gauss-Legendre rule.

    Exact for polynomials of degree 9.

    Returns:
        points: List of barycentric coordinates [[λ₀, λ₁], ...].
        weights: List of quadrature weights.
    """
    x1 = 0.5 - 0.5 * 0.906179845938664
    x2 = 0.5 - 0.5 * 0.538469310105683
    x3 = 0.5
    x4 = 0.5 + 0.5 * 0.538469310105683
    x5 = 0.5 + 0.5 * 0.906179845938664

    points = [
        [1.0 - x1, x1],
        [1.0 - x2, x2],
        [1.0 - x3, x3],
        [1.0 - x4, x4],
        [1.0 - x5, x5],
    ]

    w1 = 0.5 * 0.236926885056189
    w2 = 0.5 * 0.478628670499366
    w3 = 0.5 * 0.568888888888889
    weights = [w1, w2, w3, w2, w1]
    return points, weights
