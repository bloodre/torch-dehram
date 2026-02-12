"""Core orientation-aware incidence storage.

We store oriented boundary incidences as triplets:
  (parent_k_cell_index, child_(k-1)_cell_index, sign)

This supports arbitrary number of children by repeating parent indices.
"""

from __future__ import annotations

import torch

from torch import Tensor
from torch_sparse import SparseTensor

_INT_DTYPES = (torch.long, torch.int, torch.int16, torch.int8)


def _sort_order(parent: Tensor, child: Tensor, order: Tensor | None) -> Tensor | None:
    """Return the child cells order of the incidence matrix after sorting."""
    # Return as is if None
    if order is None:
        return order
    # permutation that torch_sparse effectively applies (row-major then col)
    # stable=True is important for determinism
    perm = child.argsort(stable=True)
    perm = perm[parent[perm].argsort(stable=True)]
    return order[perm]


def _parse_incidence(
        inc: Tensor | SparseTensor,
        sparse_sizes: tuple[int, int] | None = None,
        order: Tensor | None = None,
    ) -> tuple[SparseTensor, Tensor | None]:
    """Return incidence as SparseTensor with shape (n_children, n_parents)
    and the cells order after sorting."""
    # Return SparseTensor as-is
    if isinstance(inc, SparseTensor):
        return inc, order

    # Check convention: (nnz,2/3)
    assert isinstance(inc, Tensor)
    assert inc.dtype == torch.long, "Incidence must be torch.long indices/signs."
    assert inc.ndim == 2 and inc.size(1) in (2, 3), "Incidence must be (nnz,2) or (nnz,3)."

    # Extract parent/child/signed
    parent = inc[:, 0]
    child = inc[:, 1]
    if inc.size(1) == 3:
        sign = inc[:, 2].to(torch.float32)
    else:
        sign = torch.ones_like(parent, dtype=torch.float32)

    # store as (row=child, col=parent)
    sp = SparseTensor(row=child, col=parent, value=sign, sparse_sizes=sparse_sizes)
    return sp, _sort_order(parent, child, order)


class BoundaryIncidence:
    """Oriented incidence (boundary) between k-cells and (k-1)-cells.

    This object stores the sparse matrix for the boundary operator

        ∂_k : C_k → C_{k-1}

    as a sparse representation with shape (n_children, n_parents), where:
      - columns index k-cells (parents)
      - rows index (k-1)-cells (children)
      - values are signed incidence coefficients (typically ±1)

    Args:
        incidence (Tensor | SparseTensor):
            Incidence data.

            Accepted conventions:
              - If SparseTensor: must satisfy
                    row = child_(k-1) cell index
                    col = parent_k cell index
                    value = sign (±1)
                and have sparse shape (n_children, n_parents).

              - If dense Tensor: must be an integer tensor of shape (nnz, 2) or (nnz, 3):
                    incidence[:, 0] = parent_k index
                    incidence[:, 1] = child_(k-1) index
                    incidence[:, 2] = sign (±1)   (only if provided)
                If the sign column is omitted, all signs are taken as +1.

              - If torch.sparse COO tensor: interpreted as a matrix with
                    indices()[0] = child indices (rows)
                    indices()[1] = parent indices (cols)
                    values()      = signs (±1)
                and shape (n_children, n_parents).

        k (int):
            Dimension of the parent cells (the operator is ∂_k). Must be > 0.

        order (Tensor | None):
            Order of the cells. Allows to represent parent cells as a list
            [c_0, ..., c_m] where c_i is the child cells. When a parent cell
            is composed of m + 1 child cells, it is expected that the associated
            order values assemble into the range [0, m], hence representing a permutation.
            -> Required for cup products.
            When incidence is provided as a tensor, internal conversion to a SparseTensor
            operates a row-major then col-major sort: order is automatically adjusted.
            It is not the case for SparseTensor and we expect the order to be provided as-is.

        n_parents (int | None):
            Number of k-cells (columns). If None, may be inferred from indices
            (note: inference misses isolated cells).
            Default: None.

        n_children (int | None):
            Number of (k-1)-cells (rows). If None, may be inferred from indices
            (note: inference misses isolated cells).
            Default: None.

        validate (bool):
            If True, checks index bounds, sign values, and (optionally) that each
            parent k-cell has at least k incident children (for k>0).
            Default: True.
    """

    def __init__(
        self,
        incidence: Tensor | SparseTensor,
        k: int,
        order: Tensor | None = None,
        n_parents: int | None = None,
        n_children: int | None = None,
        validate: bool = True,
    ):
        assert k > 0
        self.k = k

        # Set incidence tensor
        sparse_sizes = (
            (n_children, n_parents)
            if n_children is not None and n_parents is not None
            else None
        )
        self.inc, self.order = _parse_incidence(incidence, sparse_sizes, order)

        # Infer sizes if not provided
        if sparse_sizes is None:
            row = self.inc.storage.row()
            col = self.inc.storage.col()
            n_parents = int(col.max().item()) + 1 if col.numel() else 0
            n_children = int(row.max().item()) + 1 if row.numel() else 0

        self._shape = (n_children, n_parents)
        self._ordered = order is not None

        # Validate data
        if validate:
            self._validate_cells()

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def shape(self) -> tuple[int, int]:
        """Shape of the incidence matrix (N_{k-1}, N_k) / (n_children, n_parents)."""
        return self._shape

    @property
    def ordered(self) -> bool:
        """Whether the cells are ordered."""
        return self._ordered

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def _validate_cells(self) -> None:
        """Validate the incidence matrix for the following criteria:
          - row/col must be long
          - values must be ±1
          - bounds: row < N_{k-1}, col < N_k
          - each k-cell has at least k boundary children
        """
        row = self.inc.storage.row()
        col = self.inc.storage.col()
        val = self.inc.storage.value()

        assert row.dtype == torch.long and col.dtype == torch.long, "row/col must be long"
        assert val is not None, "Incidence must have values (signs)."
        # Value type: keep small ints, but float is also ok; only check content
        v = val.to(torch.int8)
        u = torch.unique(v)
        assert torch.all((u == 1) | (u == -1)), "Signs must be ±1"

        # Bounds
        if row.numel():
            assert int(row.min()) >= 0 and int(row.max()) < self.shape[0]
            assert int(col.min()) >= 0 and int(col.max()) < self.shape[1]

        # Sanity: each k-cell has at least k boundary children
        if self.shape[1] > 0:
            counts = torch.bincount(col, minlength=self.shape[1])
            assert torch.all(counts >= self.k), f"Some {self.k}-cells have < {self.k} children."

        # Order
        if self.ordered:
            assert self.order.dtype in _INT_DTYPES, "Must be an integer dtype."
            assert self.order.ndim == 1, "Must be a 1D tensor."
            assert self.order.numel() == self.shape[1], "Must match cell numbers."
            # Note: should check correct permutations representation?

    # ------------------------------------------------------------------
    # Boundary / coboundary accessors
    # ------------------------------------------------------------------

    def boundary(self) -> SparseTensor:
        """Return ∂_k as SparseTensor: C_k -> C_{k-1} (rows=children, cols=parents)."""
        return self.inc

    def coboundary(self) -> SparseTensor:
        """Return d_{k-1} = (∂_k)^T as SparseTensor: C^{k-1} -> C^k."""
        return self.inc.t()

    # ------------------------------------------------------------------
    # Device movement
    # ------------------------------------------------------------------

    def device(self) -> torch.device:
        """Return the used device."""
        return self.inc.device()

    def to(self, *args, **kwargs) -> "BoundaryIncidence":
        """Move the incidence matrix to the specified device."""
        self.inc.to(*args, **kwargs)
        if self.ordered:
            self.order.to(*args, **kwargs)
        return self

    def cpu(self) -> "BoundaryIncidence":
        """Move the incidence matrix to CPU."""
        self.inc.cpu()
        if self.ordered:
            self.order.cpu()
        return self

    def cuda(self) -> "BoundaryIncidence":
        """Move the incidence matrix to GPU."""
        self.inc.cuda()
        if self.ordered:
            self.order.cuda()
        return self

    def pin_memory(self) -> "BoundaryIncidence":
        """Move the incidence matrix to pinned memory."""
        self.inc.pin_memory()
        if self.ordered:
            self.order.pin_memory()
        return self

    # ------------------------------------------------------------------
    # Type casting
    # ------------------------------------------------------------------

    def to_value_dtype(self, dtype: torch.dtype) -> "BoundaryIncidence":
        """Cast incidence tensor values to `dtype`.

        Example: conversion to float16
        """
        row = self.inc.storage.row()
        col = self.inc.storage.col()
        val = self.inc.storage.value()

        self.inc = SparseTensor(
            row=row,
            col=col,
            value=val.to(dtype),
            sparse_sizes=self.shape,
        )
        return self

    def half(self) -> "BoundaryIncidence":
        """Cast incidence tensor values to float16."""
        return self.to_value_dtype(torch.float16)

    def float(self) -> "BoundaryIncidence":
        """Cast incidence tensor values to float32."""
        return self.to_value_dtype(torch.float32)

    def double(self) -> "BoundaryIncidence":
        """Cast incidence tensor values to float64."""
        return self.to_value_dtype(torch.float64)
