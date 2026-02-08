"""Core orientation-aware incidence storage.

We store oriented boundary incidences as triplets:
  (parent_k_cell_index, child_(k-1)_cell_index, sign)

This supports arbitrary number of children by repeating parent indices.
"""

from __future__ import annotations

import torch

from torch import Tensor
from torch_sparse import SparseTensor


def _parse_incidence(
        inc: Tensor | SparseTensor,
        sparse_sizes: tuple[int, int] | None = None,
    ) -> SparseTensor:
    """Return incidence as SparseTensor with shape (n_children, n_parents)."""
    # Return SparseTensor as-is
    if isinstance(inc, SparseTensor):
        return inc

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
    return SparseTensor(row=child, col=parent, value=sign, sparse_sizes=sparse_sizes)


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
            Dimension of the parent cells (the operator is ∂_k).

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
        n_parents: int | None = None,
        n_children: int | None = None,
        validate: bool = True,
    ):
        assert k >= 0
        self.k = k

        # Set incidence tensor
        sparse_sizes = (
            (n_children, n_parents)
            if n_children is not None and n_parents is not None
            else None
        )
        self.inc = _parse_incidence(incidence, sparse_sizes)

        # Infer sizes if not provided
        if sparse_sizes is None:
            row = self.inc.storage.row()
            col = self.inc.storage.col()
            n_parents = int(col.max().item()) + 1 if col.numel() else 0
            n_children = int(row.max().item()) + 1 if row.numel() else 0

        self._shape = (n_children, n_parents)

        # Validate data
        if validate:
            self._validate_cells()

    @property
    def shape(self) -> tuple[int, int]:
        """Shape of the incidence matrix (N_{k-1}, N_k) / (n_children, n_parents)."""
        return self._shape

    def _validate_cells(self) -> None:
        """Validate the incidence matrix for the following criteria:
          - row/col must be long
          - values must be ±1
          - bounds: row < N_{k-1}, col < N_k
          - each k-cell has at least k boundary children (except k=0)
        """
        row = self.inc.storage.row()
        col = self.inc.storage.col()
        val = self.inc.storage.value()

        assert row.dtype == torch.long and col.dtype == torch.long, "row/col must be long"
        assert val is not None, "Incidence must have values (signs)."
        # value type: keep small ints, but float is also ok; only check content
        v = val.to(torch.int8)
        u = torch.unique(v)
        assert torch.all((u == 1) | (u == -1)), f"Signs must be ±1 (got {u.tolist()})"

        # bounds
        if row.numel():
            assert int(row.min()) >= 0 and int(row.max()) < self.shape[0]
            assert int(col.min()) >= 0 and int(col.max()) < self.shape[1]

        # sanity: each k-cell has at least k boundary children (except k=0)
        if self.k > 0 and self.shape[1] > 0:
            counts = torch.bincount(col, minlength=self.shape[1])
            assert torch.all(counts >= self.k), f"Some {self.k}-cells have < {self.k} children."

    def boundary(self) -> SparseTensor:
        """Return ∂_k as SparseTensor: C_k -> C_{k-1} (rows=children, cols=parents)."""
        return self.inc

    def coboundary(self) -> SparseTensor:
        """Return d_{k-1} = (∂_k)^T as SparseTensor: C^{k-1} -> C^k."""
        return self.inc.t()

    def to(self, *args, **kwargs):
        """Move the incidence matrix to the specified device."""
        self.inc.to(*args, **kwargs)

    def cpu(self):
        """Move the incidence matrix to CPU."""
        self.inc.cpu()

    def cuda(self):
        """Move the incidence matrix to GPU."""
        self.inc.cuda()

    def pin_memory(self):
        """Move the incidence matrix to pinned memory."""
        self.inc.pin_memory()

    def to_value_dtype_(self, dtype: torch.dtype):
        """Cast SparseTensor values to `dtype` (reuses row/col, replaces value).

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

    def half_(self):
        """Cast SparseTensor values to float16 (reuses row/col, replaces value)."""
        return self.to_value_dtype_(torch.float16)

    def float_(self):
        """Cast SparseTensor values to float32 (reuses row/col, replaces value)."""
        return self.to_value_dtype_(torch.float32)

    def double_(self):
        """Cast SparseTensor values to float64 (reuses row/col, replaces value)."""
        return self.to_value_dtype_(torch.float64)
