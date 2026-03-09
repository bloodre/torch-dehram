"""Triangle quadrature rules (2-simplex).

Provides pre-tabulated symmetric quadrature points and weights for numerical
integration on the reference triangle in barycentric coordinates.

Sources:
    - Dunavant, D.A., "High Degree Efficient Symmetrical Gaussian Quadrature
      Rules for the Triangle", International Journal for Numerical Methods in
      Engineering, Vol. 21, 1985, pp. 1129-1148.
      https://people.math.sc.edu/Burkardt/cpp_src/triangle_dunavant_rule/triangle_dunavant_rule.html

    - Strang, G. and Fix, G.J., "An Analysis of the Finite Element Method",
      Prentice-Hall, 1973.
      http://www.math.uci.edu/~chenlong/iFEM/fem/html/quadpts.html

Reference simplex:
    Vertices at (1,0,0), (0,1,0), (0,0,1) in barycentric coordinates.
    Area = 1/2.
"""

from __future__ import annotations


def dunavant_degree_1() -> tuple[list[list[float]], list[float]]:
    """1-point rule (centroid).

    Exact for polynomials of degree 1.

    Returns:
        points: List of barycentric coordinates [[λ₀, λ₁, λ₂]].
        weights: List of quadrature weights (sum = 1/2).
    """
    points = [
        [0.333333333333333, 0.333333333333333, 0.333333333333333],
    ]
    weights = [0.5]
    return points, weights


def dunavant_degree_2() -> tuple[list[list[float]], list[float]]:
    """3-point symmetric rule.

    Exact for polynomials of degree 2.

    Returns:
        points: List of barycentric coordinates [[λ₀, λ₁, λ₂], ...].
        weights: List of quadrature weights (sum = 1/2).
    """
    points = [
        [0.666666666666667, 0.166666666666667, 0.166666666666667],
        [0.166666666666667, 0.666666666666667, 0.166666666666667],
        [0.166666666666667, 0.166666666666667, 0.666666666666667],
    ]
    weights = [0.166666666666667, 0.166666666666667, 0.166666666666667]
    return points, weights


def dunavant_degree_3() -> tuple[list[list[float]], list[float]]:
    """6-point symmetric rule.

    Exact for polynomials of degree 3.

    Returns:
        points: List of barycentric coordinates [[λ₀, λ₁, λ₂], ...].
        weights: List of quadrature weights (sum = 1/2).
    """
    a = 0.659027622374092
    b = 0.231933368553031

    points = [
        [a, b, b],
        [b, a, b],
        [b, b, a],
        [b, a, b],
        [b, b, a],
        [a, b, b],
    ]
    weights = [0.083333333333333] * 6
    return points, weights


def dunavant_degree_4() -> tuple[list[list[float]], list[float]]:
    """6-point symmetric rule.

    Exact for polynomials of degree 4.

    Returns:
        points: List of barycentric coordinates [[λ₀, λ₁, λ₂], ...].
        weights: List of quadrature weights (sum = 1/2).
    """
    a1 = 0.108103018168070
    b1 = 0.445948490915965

    a2 = 0.816847572980459
    b2 = 0.091576213509771

    points = [
        [a1, b1, b1],
        [b1, a1, b1],
        [b1, b1, a1],
        [a2, b2, b2],
        [b2, a2, b2],
        [b2, b2, a2],
    ]

    w1 = 0.111690794839005
    w2 = 0.054975871827661
    weights = [w1, w1, w1, w2, w2, w2]
    return points, weights


def dunavant_degree_5() -> tuple[list[list[float]], list[float]]:
    """7-point symmetric rule.

    Exact for polynomials of degree 5.

    Returns:
        points: List of barycentric coordinates [[λ₀, λ₁, λ₂], ...].
        weights: List of quadrature weights (sum = 1/2).
    """
    a1 = 0.333333333333333

    a2 = 0.059715871789770
    b2 = 0.470142064105115

    a3 = 0.797426985353087
    b3 = 0.101286507323456

    points = [
        [a1, a1, a1],
        [a2, b2, b2],
        [b2, a2, b2],
        [b2, b2, a2],
        [a3, b3, b3],
        [b3, a3, b3],
        [b3, b3, a3],
    ]

    w1 = 0.112500000000000
    w2 = 0.066197076394253
    w3 = 0.062969590272413
    weights = [w1, w2, w2, w2, w3, w3, w3]
    return points, weights


def strang_fix_degree_2() -> tuple[list[list[float]], list[float]]:
    """Strang-Fix 3-point rule.

    Exact for polynomials of degree 2.
    Alternative to Dunavant degree 2 with same structure.

    Returns:
        points: List of barycentric coordinates [[λ₀, λ₁, λ₂], ...].
        weights: List of quadrature weights (sum = 1/2).
    """
    return dunavant_degree_2()


def strang_fix_degree_3() -> tuple[list[list[float]], list[float]]:
    """Strang-Fix 4-point rule.

    Exact for polynomials of degree 3.

    Returns:
        points: List of barycentric coordinates [[λ₀, λ₁, λ₂], ...].
        weights: List of quadrature weights (sum = 1/2).
    """
    a = 0.6
    b = 0.2

    points = [
        [a, b, b],
        [b, a, b],
        [b, b, a],
        [0.333333333333333, 0.333333333333333, 0.333333333333333],
    ]

    w1 = 0.104166666666667
    w2 = -0.28125
    weights = [w1, w1, w1, w2]
    return points, weights
