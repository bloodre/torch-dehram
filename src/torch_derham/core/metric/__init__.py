"""Metric structures for discrete differential forms.

This module provides:
- InnerProduct: abstract base class for inner products on k-cochains.
- DiscreteCalculus: derived operators (codifferential, Laplacian) from inner products.
- HodgeStar: abstract base class for Hodge star operators.
- DiagonalHodgeStar: specialized Hodge star for diagonal inner products.
"""

from .calculus import DiscreteCalculus
from .hodge import DiagonalHodgeStar, HodgeStar
from .inner_product import InnerProduct

__all__ = [
    "DiagonalHodgeStar",
    "DiscreteCalculus",
    "HodgeStar",
    "InnerProduct",
]
