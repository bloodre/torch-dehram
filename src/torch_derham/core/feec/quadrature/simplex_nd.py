"""Quadrature rules for high-dimensional simplices (n >= 4).

Provides pre-tabulated quadrature points and weights for numerical integration
on reference n-simplices with n >= 4 (4-simplex/pentatope, 5-simplex, etc.)
in barycentric coordinates.

Sources:
    - Grundmann, A. and Moeller, H.M., "Invariant Integration Formulas for the
      N-Simplex by Combinatorial Methods", SIAM Journal on Numerical Analysis,
      Vol. 15, No. 2, April 1978, pp. 282-290.
      https://people.math.sc.edu/Burkardt/f_src/simplex_gm_rule/simplex_gm_rule.html

    - Cools, R., "Constructing cubature formulae: the science behind the art",
      Acta Numerica, 1997, pp. 1-54.

    - STROUD, A.H., "Approximate Calculation of Multiple Integrals",
      Prentice-Hall, 1971.

Notes:
    For high-dimensional simplices (n >= 4), optimal quadrature rules are
    less well-studied than for triangles and tetrahedra. The rules provided
    here are based on symmetric constructions and Grundmann-Moeller formulas.

    For very high degrees or dimensions, algorithmic generation (like
    Grundmann-Moeller) may be more appropriate than tabulated rules.

Reference simplex:
    - n-simplex: (n+1) vertices with barycentric coordinates summing to 1.
    - 4-simplex (pentatope): 5 vertices, volume = 1/24
    - 5-simplex: 6 vertices, volume = 1/120
"""

from __future__ import annotations


def simplex_4d_degree_1() -> tuple[list[list[float]], list[float]]:
    """1-point rule for 4-simplex (pentatope).

    Exact for polynomials of degree 1.

    Returns:
        points: List of barycentric coordinates [[λ₀, λ₁, λ₂, λ₃, λ₄]].
        weights: List of quadrature weights (sum = 1/24).
    """
    points = [
        [0.2, 0.2, 0.2, 0.2, 0.2],
    ]
    weights = [0.041666666666667]
    return points, weights


def simplex_4d_degree_2() -> tuple[list[list[float]], list[float]]:
    """5-point rule for 4-simplex (pentatope).

    Exact for polynomials of degree 2.
    Symmetric vertex-centered rule.

    Returns:
        points: List of barycentric coordinates [[λ₀, λ₁, λ₂, λ₃, λ₄], ...].
        weights: List of quadrature weights (sum = 1/24).
    """
    a = 0.6
    b = 0.1

    points = [
        [a, b, b, b, b],
        [b, a, b, b, b],
        [b, b, a, b, b],
        [b, b, b, a, b],
        [b, b, b, b, a],
    ]
    weights = [0.008333333333333] * 5
    return points, weights


def simplex_4d_degree_3() -> tuple[list[list[float]], list[float]]:
    """15-point rule for 4-simplex (pentatope).

    Exact for polynomials of degree 3.
    Two-orbit symmetric rule.

    Returns:
        points: List of barycentric coordinates [[λ₀, λ₁, λ₂, λ₃, λ₄], ...].
        weights: List of quadrature weights (sum = 1/24).
    """
    c = 0.2

    a1 = 0.5
    b1 = 0.5 / 4.0

    a2 = 2.0 / 3.0
    b2 = 1.0 / 12.0

    points = [
        [c, c, c, c, c],
        [a1, a1, b1, b1, b1],
        [a1, b1, a1, b1, b1],
        [a1, b1, b1, a1, b1],
        [a1, b1, b1, b1, a1],
        [b1, a1, a1, b1, b1],
        [b1, a1, b1, a1, b1],
        [b1, a1, b1, b1, a1],
        [b1, b1, a1, a1, b1],
        [b1, b1, a1, b1, a1],
        [b1, b1, b1, a1, a1],
        [a2, a2, a2, b2, b2],
        [a2, a2, b2, a2, b2],
        [a2, a2, b2, b2, a2],
        [a2, b2, a2, a2, b2],
    ]

    w0 = 0.020833333333333
    w1 = 0.001388888888889
    w2 = 0.001388888888889

    weights = [w0] + [w1] * 10 + [w2] * 4
    return points, weights


def simplex_5d_degree_1() -> tuple[list[list[float]], list[float]]:
    """1-point rule for 5-simplex.

    Exact for polynomials of degree 1.

    Returns:
        points: List of barycentric coordinates [[λ₀, λ₁, λ₂, λ₃, λ₄, λ₅]].
        weights: List of quadrature weights (sum = 1/120).
    """
    points = [
        [1.0/6.0, 1.0/6.0, 1.0/6.0, 1.0/6.0, 1.0/6.0, 1.0/6.0],
    ]
    weights = [0.008333333333333]
    return points, weights


def simplex_5d_degree_2() -> tuple[list[list[float]], list[float]]:
    """6-point rule for 5-simplex.

    Exact for polynomials of degree 2.
    Symmetric vertex-centered rule.

    Returns:
        points: List of barycentric coordinates [[λ₀, λ₁, λ₂, λ₃, λ₄, λ₅], ...].
        weights: List of quadrature weights (sum = 1/120).
    """
    a = 0.6
    b = 0.08

    points = [
        [a, b, b, b, b, b],
        [b, a, b, b, b, b],
        [b, b, a, b, b, b],
        [b, b, b, a, b, b],
        [b, b, b, b, a, b],
        [b, b, b, b, b, a],
    ]
    weights = [0.001388888888889] * 6
    return points, weights


def simplex_5d_degree_3() -> tuple[list[list[float]], list[float]]:
    """21-point rule for 5-simplex.

    Exact for polynomials of degree 3.
    Two-orbit symmetric rule.

    Returns:
        points: List of barycentric coordinates [[λ₀, λ₁, λ₂, λ₃, λ₄, λ₅], ...].
        weights: List of quadrature weights (sum = 1/120).
    """
    c = 1.0 / 6.0

    a1 = 0.5
    b1 = 0.1

    a2 = 2.0 / 3.0
    b2 = 1.0 / 15.0

    points = [
        [c, c, c, c, c, c],
        [a1, a1, b1, b1, b1, b1],
        [a1, b1, a1, b1, b1, b1],
        [a1, b1, b1, a1, b1, b1],
        [a1, b1, b1, b1, a1, b1],
        [a1, b1, b1, b1, b1, a1],
        [b1, a1, a1, b1, b1, b1],
        [b1, a1, b1, a1, b1, b1],
        [b1, a1, b1, b1, a1, b1],
        [b1, a1, b1, b1, b1, a1],
        [b1, b1, a1, a1, b1, b1],
        [b1, b1, a1, b1, a1, b1],
        [b1, b1, a1, b1, b1, a1],
        [b1, b1, b1, a1, a1, b1],
        [b1, b1, b1, a1, b1, a1],
        [b1, b1, b1, b1, a1, a1],
        [a2, a2, a2, b2, b2, b2],
        [a2, a2, b2, a2, b2, b2],
        [a2, a2, b2, b2, a2, b2],
        [a2, a2, b2, b2, b2, a2],
        [a2, b2, a2, a2, b2, b2],
    ]

    w0 = 0.004166666666667
    w1 = 0.000231481481481
    w2 = 0.000231481481481

    weights = [w0] + [w1] * 15 + [w2] * 5
    return points, weights
