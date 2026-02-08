"""Core chain complex container built from BoundaryIncidence blocks.

Stores boundary operators ∂_k : C_k -> C_{k-1} for k=1..dim, and exposes
coboundaries d_k : C^k -> C^{k+1} via transposes.
"""

from __future__ import annotations

from typing import Sequence

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
                Bkm1 = self.boundary(k - 1)  # (N_{k-2}, N_{k-1})
                Bk = self.boundary(k)        # (N_{k-1}, N_k)
                C = Bkm1 @ Bk                # (N_{k-2}, N_k)
                nnz = C.nnz() if hasattr(C, "nnz") else C.storage.value().numel()
                assert nnz == 0, f"Nonzero entries in ∂_{k-1}∘∂_{k} (k={k})"
