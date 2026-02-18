"""Metric structures for discrete differential forms.

This module provides:
- InnerProduct: abstract base class for inner products on k-cochains.
- DiscreteCalculus: derived operators (codifferential, Laplacian) from inner products.
"""

from .calculus import DiscreteCalculus
from .inner_product import InnerProduct

__all__ = [
    "DiscreteCalculus",
    "InnerProduct",
]
