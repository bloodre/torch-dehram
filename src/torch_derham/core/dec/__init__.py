"""Discrete Exterior Calculus (DEC) for simplicial complexes.

This module provides circumcentric dual cell measures and the associated
diagonal inner product for PL-simplicial complexes embedded in Euclidean space.

Provides:
- SimplicialDualCellMeasure: circumcentric dual measures for all degrees k.
- DECInnerProduct: diagonal mass matrix from dual measures.
"""

from .dual_cells import SimplicialDualCellMeasure
from .hodge import DECInnerProduct

__all__ = [
    "DECInnerProduct",
    "SimplicialDualCellMeasure",
]
