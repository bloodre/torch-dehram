"""Abstract base classes for cell complexes and face lookup.

This module defines the core interfaces for cell complex representations:
- CellIndexing: base interface for topological cell complexes.
- FaceLookup: mixin interface for complexes supporting face lookup.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import torch
from torch import Tensor

if TYPE_CHECKING:
    from .chain import ChainComplex


class CellIndexing(ABC):
    """Abstract base class for cell complex representations.

    A cell complex stores k-cells for k=0..dim and can compute
    boundary operators on demand.
    """

    @property
    @abstractmethod
    def dim(self) -> int:
        """Topological dimension (maximum k)."""

    @abstractmethod
    def n_cells(self, k: int) -> int:
        """Number of k-cells.

        Args:
            k: cell degree.

        Returns:
            Number of k-cells in the complex.
        """

    @abstractmethod
    def sizes(self) -> list[int]:
        """List of cell counts [n_0, n_1, ..., n_dim]."""

    @abstractmethod
    def device(self) -> torch.device:
        """Return the device of the cell data."""

    @abstractmethod
    def to(self, *args, **kwargs) -> CellIndexing:
        """Move cell data to the specified device."""

    @abstractmethod
    def cpu(self) -> CellIndexing:
        """Move cell data to CPU."""

    @abstractmethod
    def cuda(self) -> CellIndexing:
        """Move cell data to GPU."""

    @abstractmethod
    def get_chain(self, compute_order: bool = True) -> ChainComplex:
        """Build a ChainComplex with explicit boundary operators.

        Args:
            compute_order: if True, computes order tensors for boundaries.

        Returns:
            ChainComplex with boundary operators for k=1..dim.
        """


class FaceLookup(ABC):
    """Mixin interface for complexes supporting face lookup.

    This interface allows looking up cell indices by their vertex/face
    representation, enabling operations like cup products.
    """

    @abstractmethod
    def lookup_faces(self, faces: Tensor, k: int) -> Tensor:
        """Lookup indices of faces in the k-cell list.

        Args:
            faces: (m, k+1) tensor of cell vertex representations.
            k: degree of cells to lookup.

        Returns:
            (m,) tensor of cell indices in [0, n_k).

        Raises:
            ValueError: if any face is not found in the complex.
        """


__all__ = [
    "CellIndexing",
    "FaceLookup",
]
