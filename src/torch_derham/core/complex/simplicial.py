"""Simplicial chain complex with implicit boundary structure.

A simplicial complex stores k-simplices as vertex lists. The boundary
operator is implicit: ∂[v0, ..., vk] = Σ (-1)^i [v0, ..., v̂i, ..., vk].
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

import torch
from torch import Tensor

from ..cochain import CoChain
from ..index import row
from .incidence import BoundaryIncidence
from .chain import ChainComplex


def _generate_faces(parents: Tensor, k: int) -> Tensor:
    """Generate all (k-1)-faces by omitting each vertex slot.

    Args:
        parents (Tensor): (N_k, k+1) tensor, vertices strictly increasing per row.
        k (int): degree of parent simplices.

    Returns:
        (N_k*(k+1), k) tensor of face vertices.
    """
    parents = parents.contiguous()
    n = parents.size(0)
    all_idx = torch.arange(k + 1, device=parents.device)
    keep = torch.stack(
        [torch.cat([all_idx[:i], all_idx[i + 1 :]]) for i in range(k + 1)],
        dim=0,
    )
    faces = parents[:, keep]
    return faces.reshape(n * (k + 1), k).contiguous()


@torch.no_grad()
def _build_boundary_incidence_k(
    parents_k: Tensor,
    children_km1: Tensor,
    k: int,
    compute_order: bool = True,
    order_dtype: torch.dtype = torch.int32,
) -> Tuple[Tensor, Optional[Tensor]]:
    """Build (incidence, order) for ∂_k from simplex vertex tables.

    Args:
        parents_k (Tensor): (N_k, k+1) integer tensor, strictly increasing rows.
        children_km1 (Tensor): (N_{k-1}, k) integer tensor, strictly increasing rows.
        k (int): simplex dimension.
        compute_order (bool): whether to return face-slot order tensor.
        order_dtype (torch.dtype): dtype for the order tensor.

    Returns:
        incidence: (nnz, 3) tensor [parent, child, sign] with same dtype as parents_k.
        order: (nnz,) integer tensor giving omitted-vertex slot i in [0..k],
            OR None if compute_order=False.
    """
    device = parents_k.device
    idx_dtype = parents_k.dtype
    n_parents = parents_k.size(0)

    if n_parents == 0 or children_km1.size(0) == 0:
        incidence = torch.empty((0, 3), dtype=idx_dtype, device=device)
        order = (
            torch.empty(0, dtype=order_dtype, device=device)
            if compute_order
            else None
        )
        return incidence, order

    faces = _generate_faces(parents_k, k)

    child_keys_sorted, child_perm = row.build_sorted_row_index(children_km1, seed=0)
    child_idx = row.lookup_row_indices(
        faces,
        children_km1,
        child_keys_sorted,
        child_perm,
    )

    parent_idx = torch.arange(n_parents, dtype=idx_dtype).repeat_interleave(
        k + 1
    )

    face_pos = torch.arange(k + 1, dtype=idx_dtype).repeat(n_parents)
    sign = 1 - 2 * (face_pos & 1)

    nnz = parent_idx.numel()
    incidence = torch.empty((nnz, 3), dtype=idx_dtype)
    incidence[:, 0] = parent_idx
    incidence[:, 1] = child_idx.to(idx_dtype)
    incidence[:, 2] = sign.to(idx_dtype)

    order = None
    if compute_order:
        order = face_pos.to(order_dtype)

    incidence = incidence.to(device=device)
    order = order.to(device=device) if order is not None else None
    return incidence, order


class SimplicialChainComplex:
    """Simplicial chain complex with implicit boundary structure.

    Stores k-simplices for k=0..dim as vertex lists. The boundary operator
    is not stored explicitly but computed on demand from the vertex structure.

    A k-simplex is represented by its (k+1) vertices in sorted order:
    [v0, v1, ..., vk] where v0 < v1 < ... < vk.

    Args:
        cells: sequence of tensors for k=0..dim.
            cells[k] has shape (n_k, k+1) where each row is the sorted
            vertices [v0, ..., vk] of a k-simplex.
        validate: if True, checks that vertices are sorted and unique per row.
    """

    def __init__(
        self,
        cells: Sequence[Tensor],
        validate: bool = True,
    ):
        assert len(cells) > 0, "Need at least 0-cells."
        self._cells = [c.contiguous() for c in cells]

        # Cache for boundary lookups
        self._lookup_cache: dict[int, tuple[Tensor, Tensor]] = {}

        if validate:
            self._validate()

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def dim(self) -> int:
        """Topological dimension (max k)."""
        return len(self._cells) - 1

    def n_cells(self, k: int) -> int:
        """Number of k-simplices."""
        if k < 0 or k > self.dim:
            raise ValueError(f"k out of range: {k}")
        return self._cells[k].shape[0]

    def cells(self, k: int) -> Tensor:
        """Return the k-simplices as a (n_k, k+1) tensor of vertices."""
        if k < 0 or k > self.dim:
            raise ValueError(f"k out of range: {k}")
        return self._cells[k]

    def sizes(self) -> list[int]:
        """List of simplex counts [n_0, n_1, ..., n_dim]."""
        return [c.shape[0] for c in self._cells]

    # ------------------------------------------------------------------
    # Device movement
    # ------------------------------------------------------------------

    def device(self) -> torch.device:
        """Return the device of the cell tensors."""
        return self._cells[0].device

    def to(self, *args, **kwargs) -> "SimplicialChainComplex":
        """Move all cell tensors to the specified device."""
        self._cells = [c.to(*args, **kwargs) for c in self._cells]
        self._lookup_cache.clear()
        return self

    def cpu(self) -> "SimplicialChainComplex":
        """Move all cell tensors to CPU."""
        self._cells = [c.cpu() for c in self._cells]
        self._lookup_cache.clear()
        return self

    def cuda(self) -> "SimplicialChainComplex":
        """Move all cell tensors to GPU."""
        self._cells = [c.cuda() for c in self._cells]
        self._lookup_cache.clear()
        return self

    # ------------------------------------------------------------------
    # Chain complex conversion
    # ------------------------------------------------------------------

    def get_chain(self, compute_order: bool = True) -> ChainComplex:
        """Build a ChainComplex from this simplicial complex.

        Computes the boundary operators ∂_k for k=1..dim by finding
        which (k-1)-simplex each face of each k-simplex corresponds to,
        along with the incidence signs.

        Args:
            compute_order: if True, computes the order tensor for each
                boundary operator (required for cup products).
                Defaults to True.

        Returns:
            ChainComplex with boundary operators for k=1..dim.
        """
        boundaries: list[BoundaryIncidence] = []

        for k in range(1, self.dim + 1):
            inc_data, order = self._compute_boundary_k(k, compute_order)
            boundaries.append(
                BoundaryIncidence(
                    incidence=inc_data,
                    k=k,
                    order=order,
                    n_parents=self.n_cells(k),
                    n_children=self.n_cells(k - 1),
                    validate=False,
                )
            )

        return ChainComplex(boundaries, validate=False)

    def _compute_boundary_k(
        self,
        k: int,
        compute_order: bool,
    ) -> tuple[Tensor, Optional[Tensor]]:
        """Compute the boundary data for ∂_k.

        For each k-simplex σ = [v0, ..., vk], the boundary is:
            ∂σ = Σ_{i=0}^{k} (-1)^i · [v0, ..., v̂i, ..., vk]

        where v̂i means vi is omitted.

        Args:
            k: the degree of the boundary operator.
            compute_order: whether to compute the order tensor.

        Returns:
            inc_data: (nnz, 3) tensor of (parent, child, sign) triplets.
            order: (nnz,) tensor of face positions, or None.
        """
        return _build_boundary_incidence_k(
            parents_k=self._cells[k],
            children_km1=self._cells[k - 1],
            k=k,
            compute_order=compute_order,
        )

    # ------------------------------------------------------------------
    # Face lookup
    # ------------------------------------------------------------------

    def _ensure_lookup_cache(self, k: int) -> Tuple[Tensor, Tensor]:
        """Ensure lookup cache exists for k-simplices, return (keys_sorted, perm)."""
        if k not in self._lookup_cache:
            keys_sorted, perm = row.build_sorted_row_index(self._cells[k])
            self._lookup_cache[k] = (keys_sorted, perm)
        return self._lookup_cache[k]

    def lookup_faces(self, faces: Tensor, k: int) -> Tensor:
        """Lookup indices of faces in the k-simplex list.

        Args:
            faces (Tensor): (m, k+1) tensor of face vertices.
            k (int): degree of simplices to lookup.

        Returns:
            (m,) tensor of simplex indices.
        """
        if k < 0 or k > self.dim:
            raise ValueError(f"k out of range: {k}")
        keys_sorted, perm = self._ensure_lookup_cache(k)
        return row.lookup_row_indices(
            faces,
            self._cells[k],
            keys_sorted,
            perm,
        )

    # ------------------------------------------------------------------
    # Alexander-Whitney cup product
    # ------------------------------------------------------------------

    def aw_cup(
        self,
        cochain_p: CoChain,
        cochain_q: CoChain,
        chain: Tensor,
    ) -> Tensor:
        """Alexander-Whitney cup product evaluation.

        For a (p+q)-chain represented as vertex lists, evaluate the cup product:
            (α ∪ β)(σ) = α(front_p(σ)) · β(back_q(σ))

        where:
            front_p(σ) = [v0, ..., vp]
            back_q(σ) = [vp, ..., vk]

        Args:
            cochain_p (CoChain): cochain of degree p with data (n_p, d_p).
            cochain_q (CoChain): cochain of degree q with data (n_q, d_q).
            chain (Tensor): (m, p+q+1) tensor of (p+q)-simplex vertices.

        Returns:
            (m, d_p * d_q) tensor of cup product values.
        """
        p = cochain_p.k
        q = cochain_q.k
        k = p + q

        if chain.ndim != 2 or chain.size(1) != k + 1:
            raise ValueError(
                f"chain must be (m, {k+1}) for p={p}, q={q}, got shape {tuple(chain.shape)}"
            )

        front_faces = chain[:, : p + 1]
        back_faces = chain[:, p :]

        front_indices = self.lookup_faces(front_faces, p)
        back_indices = self.lookup_faces(back_faces, q)

        alpha_vals = cochain_p.data[front_indices]
        beta_vals = cochain_q.data[back_indices]

        result = alpha_vals.unsqueeze(-1) * beta_vals.unsqueeze(-2)
        return result.reshape(chain.size(0), -1)

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def _validate(self) -> None:
        """Validate the simplicial complex structure."""
        for k, cells in enumerate(self._cells):
            assert cells.ndim == 2, f"cells[{k}] must be 2D"
            assert cells.shape[1] == k + 1, (
                f"cells[{k}] must have shape (n, {k + 1}), "
                f"got shape {tuple(cells.shape)}"
            )

            # Check vertices are sorted within each simplex
            if k > 0:
                diffs = cells[:, 1:] - cells[:, :-1]
                assert torch.all(diffs > 0), (
                    f"Vertices in cells[{k}] must be strictly increasing "
                    "within each row"
                )
