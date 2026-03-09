"""Grundmann-Moeller quadrature rules for n-dimensional simplices.

Algorithmic generation of quadrature rules for arbitrary dimension and degree.
These rules are exact for odd polynomial degrees and work for any n-simplex.

Sources:
    - Grundmann, A. and Moeller, H.M., "Invariant Integration Formulas for the
      N-Simplex by Combinatorial Methods", SIAM Journal on Numerical Analysis,
      Vol. 15, No. 2, April 1978, pp. 282-290.
      https://doi.org/10.1137/0715019

    - Burkardt's implementation:
      https://people.math.sc.edu/Burkardt/f_src/simplex_gm_rule/simplex_gm_rule.html

    - Julia package:
      https://github.com/eschnett/GrundmannMoeller.jl

Notes:
    Grundmann-Moeller rules are the standard method for high-dimensional simplex
    quadrature. They work for arbitrary dimension n and odd degrees (1, 3, 5, ...).

    Key characteristics:
    - Exact for odd polynomial degrees
    - Points lie on regular barycentric grid
    - All weights are positive
    - Number of points: C(s+n, n) where s = (degree+1)/2
    - Computationally efficient to generate

    For even degrees, use degree+1 (e.g., degree 4 → use degree 5 rule).

Reference simplex:
    - n-simplex: (n+1) vertices with barycentric coordinates summing to 1.
    - Volume of unit n-simplex: 1/n!
"""

from __future__ import annotations

from math import factorial
from typing import Iterator


def integer_combinations_with_sum(target_sum: int, length: int) -> Iterator[list[int]]:
    """Generate all tuples of non-negative integers with given sum and length.

    Args:
        target_sum: Target sum for each tuple.
        length: Number of elements in each tuple.

    Yields:
        Lists of non-negative integers [k₀, k₁, ..., k_{length-1}] where ∑kᵢ = target_sum.

    Example:
        >>> list(integer_combinations_with_sum(2, 3))
        [[2, 0, 0], [1, 1, 0], [1, 0, 1], [0, 2, 0], [0, 1, 1], [0, 0, 2]]
    """
    if length == 1:
        yield [target_sum]
    else:
        for i in range(target_sum + 1):
            for rest in integer_combinations_with_sum(target_sum - i, length - 1):
                yield [i] + rest


def grundmann_moeller_points(dimension: int, degree: int) -> list[list[float]]:
    """Generate Grundmann-Moeller quadrature points.

    Args:
        dimension: Dimension of the simplex (n).
        degree: Polynomial degree of exactness (must be odd).

    Returns:
        List of barycentric coordinates [[λ₀, λ₁, ..., λₙ], ...].

    Raises:
        ValueError: If degree is even.

    Example:
        >>> points = grundmann_moeller_points(dimension=2, degree=3)
        >>> len(points)
        6
    """
    if degree % 2 == 0:
        raise ValueError(
            f"Grundmann-Moeller rules only work for odd degrees. "
            f"Got degree={degree}. Use degree={degree+1} instead."
        )

    n = dimension
    s = (degree + 1) // 2

    points = []
    for combo in integer_combinations_with_sum(s, n + 1):
        point = [x / (degree + 1) for x in combo]
        points.append(point)

    return points


def grundmann_moeller_weights(
    dimension: int,
    degree: int,
    points: list[list[float]] | None = None
) -> list[float]:
    """Generate Grundmann-Moeller quadrature weights.

    Args:
        dimension: Dimension of the simplex (n).
        degree: Polynomial degree of exactness (must be odd).
        points: Optional pre-computed points. If None, points are generated internally.

    Returns:
        List of quadrature weights (sum = volume of unit n-simplex = 1/n!).

    Raises:
        ValueError: If degree is even.

    Example:
        >>> weights = grundmann_moeller_weights(dimension=2, degree=3)
        >>> abs(sum(weights) - 0.5) < 1e-10
        True
    """
    if degree % 2 == 0:
        raise ValueError(
            f"Grundmann-Moeller rules only work for odd degrees. "
            f"Got degree={degree}. Use degree={degree+1} instead."
        )

    n = dimension
    d = degree
    s = (d + 1) // 2

    if points is None:
        points = grundmann_moeller_points(dimension, degree)

    weights = []

    for point in points:
        combo = [int(round(coord * (d + 1))) for coord in point]

        weight = ((-1) ** s) * (2 ** (-2 * s))
        weight *= factorial(n) * factorial(d + n + 1)

        for beta in range(s + 1):
            term = ((-1) ** beta) / factorial(beta)

            for i in range(n + 1):
                k_i = combo[i]
                numerator = factorial(2 * s - beta + k_i)
                denominator = factorial(s - beta + k_i)
                term *= numerator / denominator

            weight += term

        weight /= factorial(s) * factorial(d + 1)

        weights.append(weight)

    return weights


def grundmann_moeller_rule(
    dimension: int,
    degree: int
) -> tuple[list[list[float]], list[float]]:
    """Generate complete Grundmann-Moeller quadrature rule.

    Args:
        dimension: Dimension of the simplex (n).
        degree: Polynomial degree of exactness (must be odd).

    Returns:
        points: List of barycentric coordinates [[λ₀, λ₁, ..., λₙ], ...].
        weights: List of quadrature weights (sum = 1/n!).

    Raises:
        ValueError: If degree is even.

    Example:
        >>> points, weights = grundmann_moeller_rule(dimension=2, degree=3)
        >>> len(points) == len(weights)
        True
        >>> abs(sum(weights) - 0.5) < 1e-10
        True
    """
    points = grundmann_moeller_points(dimension, degree)
    weights = grundmann_moeller_weights(dimension, degree, points)
    return points, weights


def grundmann_moeller_info(dimension: int, degree: int) -> dict[str, int | float]:
    """Get information about a Grundmann-Moeller rule without generating it.

    Args:
        dimension: Dimension of the simplex (n).
        degree: Polynomial degree of exactness.

    Returns:
        Dictionary with rule information:
        - num_points: Number of quadrature points
        - simplex_volume: Volume of unit n-simplex
        - degree: Polynomial degree of exactness

    Example:
        >>> info = grundmann_moeller_info(dimension=4, degree=3)
        >>> info['num_points']
        15
    """
    n = dimension
    d = degree if degree % 2 == 1 else degree + 1
    s = (d + 1) // 2

    num_points = 1
    for i in range(n):
        num_points *= (s + i + 1)
        num_points //= (i + 1)

    simplex_volume = 1.0 / factorial(n)

    return {
        'num_points': num_points,
        'simplex_volume': simplex_volume,
        'degree': d,
    }
