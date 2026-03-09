"""Tetrahedron quadrature rules (3-simplex).

Provides pre-tabulated symmetric quadrature points and weights for numerical
integration on the reference tetrahedron in barycentric coordinates.

Sources:
    - Keast, P., "Moderate Degree Tetrahedral Quadrature Formulas",
      Computer Methods in Applied Mechanics and Engineering, Vol. 55, No. 3,
      May 1986, pp. 339-348.
      https://people.math.sc.edu/Burkardt/f_src/tetrahedron_keast_rule/tetrahedron_keast_rule.html

    - Burkardt, J., "TETRAHEDRON_KEAST_RULE - Quadrature Rules for a Tetrahedron"
      https://people.sc.fsu.edu/~jburkardt/f_src/tetrahedron_keast_rule/tetrahedron_keast_rule.html

Reference simplex:
    Vertices at (1,0,0,0), (0,1,0,0), (0,0,1,0), (0,0,0,1) in barycentric coordinates.
    Volume = 1/6.
"""

from __future__ import annotations


def keast_degree_1() -> tuple[list[list[float]], list[float]]:
    """1-point rule (centroid).

    Exact for polynomials of degree 1.

    Returns:
        points: List of barycentric coordinates [[λ₀, λ₁, λ₂, λ₃]].
        weights: List of quadrature weights (sum = 1/6).
    """
    points = [
        [0.25, 0.25, 0.25, 0.25],
    ]
    weights = [0.166666666666667]
    return points, weights


def keast_degree_2() -> tuple[list[list[float]], list[float]]:
    """4-point symmetric rule.

    Exact for polynomials of degree 2.

    Returns:
        points: List of barycentric coordinates [[λ₀, λ₁, λ₂, λ₃], ...].
        weights: List of quadrature weights (sum = 1/6).
    """
    a = 0.585410196624969
    b = 0.138196601125011

    points = [
        [a, b, b, b],
        [b, a, b, b],
        [b, b, a, b],
        [b, b, b, a],
    ]
    weights = [0.041666666666667] * 4
    return points, weights


def keast_degree_3() -> tuple[list[list[float]], list[float]]:
    """10-point symmetric rule.

    Exact for polynomials of degree 3.

    Returns:
        points: List of barycentric coordinates [[λ₀, λ₁, λ₂, λ₃], ...].
        weights: List of quadrature weights (sum = 1/6).
    """
    a1 = 0.25

    a2 = 0.5
    b2 = 0.166666666666667

    points = [
        [a1, a1, a1, a1],
        [a2, b2, b2, b2],
        [b2, a2, b2, b2],
        [b2, b2, a2, b2],
        [b2, b2, b2, a2],
        [b2, b2, a2, b2],
        [b2, a2, b2, b2],
        [a2, b2, b2, b2],
        [b2, b2, b2, a2],
        [a1, a1, a1, a1],
    ]

    w1 = -0.013155555555556
    w2 = 0.007622222222222
    weights = [w1, w2, w2, w2, w2, w2, w2, w2, w2, w1]
    return points, weights


def keast_degree_4() -> tuple[list[list[float]], list[float]]:
    """11-point symmetric rule.

    Exact for polynomials of degree 4.

    Returns:
        points: List of barycentric coordinates [[λ₀, λ₁, λ₂, λ₃], ...].
        weights: List of quadrature weights (sum = 1/6).
    """
    a1 = 0.25

    a2 = 0.071428571428571
    b2 = 0.785714285714286

    a3 = 0.399403576166799
    b3 = 0.100596423833201

    points = [
        [a1, a1, a1, a1],
        [a2, a2, a2, b2],
        [a2, a2, b2, a2],
        [a2, b2, a2, a2],
        [b2, a2, a2, a2],
        [a3, a3, b3, b3],
        [a3, b3, a3, b3],
        [a3, b3, b3, a3],
        [b3, a3, a3, b3],
        [b3, a3, b3, a3],
        [b3, b3, a3, a3],
    ]

    w1 = -0.013155555555556
    w2 = 0.007622222222222
    w3 = 0.024888888888889
    weights = [w1, w2, w2, w2, w2, w3, w3, w3, w3, w3, w3]
    return points, weights


def keast_degree_5() -> tuple[list[list[float]], list[float]]:
    """14-point symmetric rule.

    Exact for polynomials of degree 5.

    Returns:
        points: List of barycentric coordinates [[λ₀, λ₁, λ₂, λ₃], ...].
        weights: List of quadrature weights (sum = 1/6).
    """
    a1 = 0.25

    a2 = 0.0
    b2 = 0.333333333333333

    a3 = 0.727272727272727
    b3 = 0.090909090909091

    a4 = 0.066550153573664
    b4 = 0.433449846426336

    points = [
        [a1, a1, a1, a1],
        [a2, b2, b2, b2],
        [b2, a2, b2, b2],
        [b2, b2, a2, b2],
        [b2, b2, b2, a2],
        [a3, b3, b3, b3],
        [b3, a3, b3, b3],
        [b3, b3, a3, b3],
        [b3, b3, b3, a3],
        [a4, a4, b4, b4],
        [a4, b4, a4, b4],
        [a4, b4, b4, a4],
        [b4, a4, a4, b4],
        [b4, a4, b4, a4],
    ]

    w1 = 0.030283678097089
    w2 = 0.006026785714286
    w3 = 0.011645249086029
    w4 = 0.010949141561386
    weights = [w1, w2, w2, w2, w2, w3, w3, w3, w3, w4, w4, w4, w4, w4]
    return points, weights
