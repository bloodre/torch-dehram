"""Core chain complex container built from BoundaryIncidence blocks.

Stores boundary operators ∂_k : C_k -> C_{k-1} for k=1..dim, and exposes
coboundaries d_k : C^k -> C^{k+1} via transposes.
"""

from __future__ import annotations

from typing import Sequence

import torch
from torch import Tensor
from torch_sparse import SparseTensor

from .incidence import BoundaryIncidence


class ChainComplex:
    """Finite-dimensional chain complex (cell complex) via boundary incidences.

    Args:
        boundaries: sequence of BoundaryIncidence for k=1..dim (in increasing k).
            boundaries[k-1] must have .k == k and shape (N_{k-1}, N_k).
        validate: if True, checks dimension consistency and ∂_{k-1} ∘ ∂_k = 0
            (the latter is optional and can be expensive; see validate_idempotent).
        validate_idempotent: if True, validates that d∘d = 0 (equivalently ∂∘∂ = 0).
            Defaults to False.
    """
    boundaries: list[BoundaryIncidence]

    def __init__(
        self,
        boundaries: Sequence[BoundaryIncidence],
        validate: bool = True,
        validate_idempotent: bool = False,
    ):
        self.boundaries = list(boundaries)
        if validate:
            self._validate(validate_idempotent=validate_idempotent)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def dim(self) -> int:
        """Topological dimension (max k)."""
        return len(self.boundaries)

    def n_cells(self, k: int) -> int:
        """Number of k-cells, inferred from boundary shapes."""
        if k < 0 or k > self.dim:
            raise ValueError(f"k out of range: {k}")
        if k == 0:
            return self.boundaries[0].shape[0] if self.dim >= 1 else 0
        return self.boundaries[k - 1].shape[1]

    # ------------------------------------------------------------------
    # Boundary / coboundary accessors
    # ------------------------------------------------------------------

    def boundary(self, k: int) -> SparseTensor:
        """Return ∂_k : C_k -> C_{k-1}. Defined only for k>=1."""
        if k < 1 or k > self.dim:
            raise ValueError(f"boundary(k) defined for 1<=k<=dim, got k={k}")
        return self.boundaries[k - 1].boundary()

    def coboundary(self, k: int) -> SparseTensor:
        """Return d_k : C^k -> C^{k+1} where d_k = (∂_{k+1})^T.

        Defined for 0<=k<dim.
        """
        if k < 0 or k >= self.dim:
            raise ValueError(f"coboundary(k) defined for 0<=k<dim, got k={k}")
        return self.boundaries[k].coboundary()  # boundary for k+1 stored at index k

    # alias
    d = coboundary

    # ------------------------------------------------------------------
    # Device movement
    # ------------------------------------------------------------------

    def to(self, *args, **kwargs) -> "ChainComplex":
        """Move the chain complex to the specified device."""
        for b in self.boundaries:
            b.to(*args, **kwargs)
        return self

    def cpu(self) -> "ChainComplex":
        """Move the chain complex to CPU."""
        for b in self.boundaries:
            b.cpu()
        return self

    def cuda(self) -> "ChainComplex":
        """Move the chain complex to GPU."""
        for b in self.boundaries:
            b.cuda()
        return self

    def pin_memory(self) -> "ChainComplex":
        """Move the chain complex to pinned memory."""
        for b in self.boundaries:
            b.pin_memory()
        return self

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------

    def contiguous(self) -> "ContiguousChainComplex":
        """Build a contiguous representation of this chain complex.

        All k-cell index spaces are merged into a single global index space,
        and all boundary operators are packed into one SparseTensor.
        """
        return ContiguousChainComplex.from_boundaries(self.boundaries)

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def _validate(self, validate_idempotent: bool = False) -> None:
        """Validate the chain complex for the following criteria:
          - k labels + shape chaining
          - ∂_{k-1} ∘ ∂_k = 0 (optional)
        """
        # k labels + shape chaining
        for i, b in enumerate(self.boundaries, start=1):
            assert b.k == i, f"Expected boundary block with k={i}, got k={b.k}"
            if i > 1:
                prev = self.boundaries[i - 2]
                assert prev.shape[1] == b.shape[0], (
                    f"Shape mismatch: ∂_{i-1} is {prev.shape}, ∂_{i} is {b.shape} "
                    f"(need N_{i-1} to match)"
                )

        if validate_idempotent and self.dim >= 2:
            # Check ∂_{k-1} ∘ ∂_k = 0 for k=2..dim
            # i.e., boundary(k-1) @ boundary(k) == 0
            # (SparseTensor matmul exists; keep this as an optional expensive check.)
            for k in range(2, self.dim + 1):
                bkm1 = self.boundary(k - 1)  # (N_{k-2}, N_{k-1})
                bk = self.boundary(k)        # (N_{k-1}, N_k)
                c = bkm1 @ bk                # (N_{k-2}, N_k)
                nnz = c.nnz() if hasattr(c, "nnz") else c.storage.value().numel()
                assert nnz == 0, f"Nonzero entries in ∂_{k-1}∘∂_{k} (k={k})"


class ContiguousChainComplex:
    """Chain complex stored as a single block-bidiagonal SparseTensor.

    Instead of holding one SparseTensor per boundary operator ∂_k, this
    representation packs *all* k-cell index spaces into a single global
    index space and stores the full boundary operator as one SparseTensor.

    Global indexing layout (given offsets tensor of length dim+2):
        k-cells occupy global indices [offsets[k], offsets[k+1]).

    The single SparseTensor ``D`` has shape (N_total, N_total) where
    N_total = sum of all N_k.  The block for ∂_k lives at:
        rows in [offsets[k-1], offsets[k])
        cols in [offsets[k],   offsets[k+1])

    This layout is lighter in metadata (one sparse object instead of dim)
    and enables efficient global operations across degrees.

    Args:
        data: single SparseTensor of shape (N_total, N_total) containing
            all boundary blocks.
        offsets: 1-D int32 or int64 tensor of length dim+2.
            offsets[k] is the first global index of the k-cell space.
        validate_idempotent: whether to validate that ∂_{k-1} ∘ ∂_k = 0
            for k=2..dim. This is an expensive check (O(N^2) per boundary).
            Defaults to False.
    """

    def __init__(
        self,
        data: SparseTensor,
        offsets: Tensor,
        validate_idempotent: bool = False,
    ):
        self._data = data
        if not hasattr(self._data, "colptr") or self._data.colptr() is None:
            self._data.fill_cache_()
        self._offsets = offsets
        if validate_idempotent:
            self._validate_idempotent()

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def data(self) -> SparseTensor:
        """The global block-bidiagonal boundary SparseTensor."""
        return self._data

    @property
    def offsets(self) -> Tensor:
        """Offsets tensor of length dim+2 delimiting each k-cell space."""
        return self._offsets

    @property
    def dim(self) -> int:
        """Topological dimension (max k)."""
        # offsets has dim+2 entries: [0-cells, 1-cells, ..., dim-cells, end]
        return self._offsets.numel() - 2

    @property
    def n_total(self) -> int:
        """Total number of cells across all dimensions."""
        return int(self._offsets[-1].item())

    def n_cells(self, k: int) -> int:
        """Number of k-cells."""
        if k < 0 or k > self.dim:
            raise ValueError(f"k out of range: {k}")
        return int(
            (self._offsets[k + 1] - self._offsets[k]).cpu().item()
        )

    def sizes(self) -> list[int]:
        """List of cell counts [N_0, N_1, ..., N_dim]."""
        return [self.n_cells(k) for k in range(self.dim + 1)]

    # ------------------------------------------------------------------
    # Boundary / coboundary accessors
    # ------------------------------------------------------------------

    def boundary(self, k: int) -> SparseTensor:
        """Return ∂_k as a SparseTensor of shape (N_{k-1}, N_k).

        Slices the global tensor at:
            rows: [offsets[k-1], offsets[k])
            cols: [offsets[k],   offsets[k+1])
        Row/col indices are shifted back to local [0, N_{k-1}) / [0, N_k).
        """
        if k < 1 or k > self.dim:
            raise ValueError(f"boundary(k) defined for 1<=k<=dim, got k={k}")

        row_start = self._offsets[k - 1]
        row_end = self._offsets[k]
        col_start = self._offsets[k]
        col_end = self._offsets[k + 1]

        # Extract COO entries from the global tensor
        g_row = self._data.storage.row()
        g_col = self._data.storage.col()
        g_val = self._data.storage.value()

        # Mask for entries that belong to the ∂_k block
        mask = (
            (g_row >= row_start) & (g_row < row_end)
            & (g_col >= col_start) & (g_col < col_end)
        )

        # Shift to local indices
        local_row = g_row[mask] - row_start
        local_col = g_col[mask] - col_start
        local_val = g_val[mask]

        # Sparse size computation
        sparse_sizes = (
            (row_end - row_start).cpu().item(),
            (col_end - col_start).cpu().item(),
        )

        return SparseTensor(
            row=local_row,
            col=local_col,
            value=local_val,
            sparse_sizes=sparse_sizes,
        )

    def coboundary(self, k: int) -> SparseTensor:
        """Return d_k = (∂_{k+1})^T as SparseTensor: C^k -> C^{k+1}.

        Defined for 0 <= k < dim.
        """
        if k < 0 or k >= self.dim:
            raise ValueError(f"coboundary(k) defined for 0<=k<dim, got k={k}")
        return self.boundary(k + 1).t()

    # alias
    d = coboundary

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------

    @classmethod
    def from_boundaries(
        cls,
        boundaries: Sequence[BoundaryIncidence],
    ) -> ContiguousChainComplex:
        """Build a ContiguousChainComplex from a sequence of BoundaryIncidence.

        Each BoundaryIncidence for ∂_k has shape (N_{k-1}, N_k).
        The global index space is laid out as:
            [0-cells | 1-cells | ... | dim-cells]
        with offsets computed from the boundary shapes.

        Args:
            boundaries: sequence of BoundaryIncidence for k=1..dim
                (in increasing k order).
        """
        if not boundaries:
            raise ValueError("Need at least one boundary operator.")

        dim = len(boundaries)

        # Compute per-dimension sizes: N_0, N_1, ..., N_dim
        # N_0 = boundaries[0].shape[0]  (number of 0-cells)
        # N_k = boundaries[k-1].shape[1] for k >= 1
        cell_counts = [boundaries[0].shape[0]]
        for b in boundaries:
            cell_counts.append(b.shape[1])

        # Build offsets: offsets[k] = sum(N_0 .. N_{k-1})
        offsets = torch.zeros(dim + 2, dtype=torch.long)
        for i, count in enumerate(cell_counts):
            offsets[i + 1] = offsets[i] + count

        # Collect shifted COO triplets from each ∂_k
        all_rows: list[Tensor] = []
        all_cols: list[Tensor] = []
        all_vals: list[Tensor] = []

        for k_idx, b in enumerate(boundaries):
            # k_idx=0 corresponds to ∂_1, k_idx=1 to ∂_2, etc.
            k = k_idx + 1
            sp = b.boundary()

            row = sp.storage.row()
            col = sp.storage.col()
            val = sp.storage.value()

            # Shift row indices into the (k-1)-cell global range
            row_offset = int(offsets[k - 1].item())
            # Shift col indices into the k-cell global range
            col_offset = int(offsets[k].item())

            all_rows.append(row + row_offset)
            all_cols.append(col + col_offset)
            all_vals.append(val)

        n_total = int(offsets[-1].item())

        # Assemble the single global SparseTensor
        data = SparseTensor(
            row=torch.cat(all_rows),
            col=torch.cat(all_cols),
            value=torch.cat(all_vals),
            sparse_sizes=(n_total, n_total),
        )

        return cls(data=data, offsets=offsets)

    # ------------------------------------------------------------------
    # Device movement
    # ------------------------------------------------------------------

    def to(self, *args, **kwargs) -> ContiguousChainComplex:
        """Move the contiguous chain complex to the specified device."""
        self._data = self._data.to(*args, **kwargs)
        self._offsets = self._offsets.to(*args, **kwargs)
        return self

    def cpu(self) -> ContiguousChainComplex:
        """Move the contiguous chain complex to CPU."""
        self._data = self._data.cpu()
        self._offsets = self._offsets.cpu()
        return self

    def cuda(self) -> ContiguousChainComplex:
        """Move the contiguous chain complex to GPU."""
        self._data = self._data.cuda()
        self._offsets = self._offsets.cuda()
        return self

    def pin_memory(self) -> ContiguousChainComplex:
        """Move the contiguous chain complex to pinned memory."""
        self._data = self._data.pin_memory()
        self._offsets = self._offsets.pin_memory()
        return self

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def _validate_idempotent(self) -> None:
        for k in range(2, self.dim + 1):
            bkm1 = self.boundary(k - 1)  # (N_{k-2}, N_{k-1})
            bk = self.boundary(k)      # (N_{k-1}, N_k)
            c = bkm1 @ bk
            nnz = c.nnz() if hasattr(c, "nnz") else c.storage.value().numel()
            assert nnz == 0, f"Nonzero entries in ∂_{k-1}∘∂_{k} (k={k})"
