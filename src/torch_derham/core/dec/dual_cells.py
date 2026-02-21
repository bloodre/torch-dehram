"""Circumcentric dual cell measures for simplicial complexes.

Implements the standard DEC (Discrete Exterior Calculus) circumcentric dual
for a PL-simplicial complex embedded in Euclidean space.

For each primal k-simplex σ, the dual measure is:

    dual_measure[k][σ] = Σ_T Σ_{flags σ=F_k ⊂ ... ⊂ F_n=T} vol_{n-k}(c(F_k),...,c(F_n))

where the outer sum is over incident top n-simplices T, the inner sum is over
all (n-k)! ordered flags from σ to T, c(F_j) is the circumcenter of F_j, and
vol_{n-k} is the Euclidean (n-k)-simplex volume.

Note: circumcentric duals are exact for Delaunay complexes (circumcenters
inside top simplices). For general complexes, dual volumes may be negative
locally; absolute values are taken per flag, consistent with the Euclidean
circumcentric approximation of the dual cell measure.
"""
# pylint: disable=invalid-name

from __future__ import annotations

from itertools import combinations, permutations
from typing import TYPE_CHECKING

import torch
from torch import Tensor

from ..cochain import CoChain
from ..ops.index import row

if TYPE_CHECKING:
    from ..complex.simplicial import SimplicialChainComplex


# ------------------------------------------------------------------
# Geometric helpers (batched, no Python loops over simplices)
# ------------------------------------------------------------------


def _circumcenter(verts: Tensor) -> Tensor:
    """Compute circumcenters of a batch of j-simplices in ℝ^m.

    Args:
        verts (Tensor): (N, j+1, m) vertex coordinates.

    Returns:
        (N, m) circumcenters, equidistant from all j+1 vertices.
    """
    # Circumcenter condition: 2(p_i - p_0)·c = ||p_i||² - ||p_0||², i=1..j
    # Let c' = c - p_0; solve  A c' = b  in least-squares sense.
    p0 = verts[:, 0:1, :]          # (N, 1, m)
    A = 2.0 * (verts[:, 1:, :] - p0)    # (N, j, m)
    b = (verts[:, 1:, :] ** 2 - verts[:, 0:1, :] ** 2).sum(dim=-1)  # (N, j)

    # torch.linalg.lstsq: A is (..., j, m), b is (..., j, 1)
    c_prime = torch.linalg.lstsq(
        A,
        b.unsqueeze(-1),
    ).solution.squeeze(-1)  # (N, m)

    return c_prime + verts[:, 0, :]   # (N, m)


def _simplex_volume(verts: Tensor) -> Tensor:
    """Compute unsigned volumes of a batch of j-simplices in ℝ^m.

    Uses the Gram determinant: vol = sqrt(det(E^T E)) / j!
    where E = [p_1-p_0, ..., p_j-p_0] is (N, m, j).

    For j = 0 (single point): returns 1 by convention.

    Args:
        verts (Tensor): (N, j+1, m) vertex coordinates.

    Returns:
        (N,) unsigned volumes.
    """
    N, jp1, _ = verts.shape
    j = jp1 - 1

    if j == 0:
        return verts.new_ones(N)

    # Edge matrix: (N, m, j)
    E = (verts[:, 1:, :] - verts[:, 0:1, :]).transpose(1, 2)

    # Gram matrix: (N, j, j)
    G = E.transpose(1, 2) @ E

    # vol = sqrt(det(G)) / j!
    sign, logdet = torch.linalg.slogdet(G)
    gram_det = sign * torch.exp(0.5 * logdet)  # (N,)

    factorial_j = float(torch.arange(1, j + 1).prod().item())
    return gram_det.abs() / factorial_j


# ------------------------------------------------------------------
# Flag enumeration (CPU, combinatorial, tiny for small n)
# ------------------------------------------------------------------


def _enumerate_flags(n: int, k: int) -> list[tuple[tuple[int, ...], list[tuple[int, ...]]]]:
    """Enumerate all (k-face, flag) pairs for a standard n-simplex.

    A flag from σ to T is a chain σ = F_k ⊂ F_{k+1} ⊂ ... ⊂ F_n = T,
    represented as a list of tuples of local vertex indices (into {0,...,n}).

    Args:
        n (int): top simplex dimension.
        k (int): primal face dimension.

    Returns:
        List of (sigma_local, flag) pairs where:
            sigma_local: (k+1,) tuple of local vertex indices.
            flag: list of (j+1,) tuples for j=k,...,n.
    """
    result = []
    top = tuple(range(n + 1))

    for sigma in combinations(top, k + 1):
        complement = [v for v in top if v not in sigma]

        for perm in permutations(complement):
            flag = [sigma]
            current = list(sigma)
            for v in perm:
                current = sorted(current + [v])
                flag.append(tuple(current))
            result.append((sigma, flag))

    return result


# ------------------------------------------------------------------
# Per-degree dual measure computation
# ------------------------------------------------------------------


def _compute_dual_measure_k(
    top_cells: Tensor,
    k_cells: Tensor,
    vertex_pos: Tensor,
    n: int,
    k: int,
) -> Tensor:
    """Compute circumcentric dual measure for all k-simplices.

    For each k-simplex σ, accumulates contributions from all incident top
    n-simplices T via the flag simplex formula (see module docstring).

    Args:
        top_cells (Tensor): (N_n, n+1) global vertex indices of top simplices.
        k_cells (Tensor): (N_k, k+1) global vertex indices of k-simplices.
        vertex_pos (Tensor): (n_0, m) embedded vertex coordinates.
        n (int): top dimension.
        k (int): primal face degree.

    Returns:
        (N_k,) tensor of unsigned dual measures for each k-simplex.
    """
    N_k = k_cells.shape[0]
    device = vertex_pos.device
    dtype = vertex_pos.dtype

    # Top case: dual of a top simplex is a point (measure = 1 by convention)
    if k == n:
        return vertex_pos.new_ones(N_k)

    dual_measure = torch.zeros(N_k, dtype=dtype, device=device)

    # Build hash index for fast global id lookup of k-simplices
    keys_sorted, perm = row.build_sorted_row_index(k_cells)

    # Enumerate all (sigma_local, flag) pairs for the standard n-simplex
    flags = _enumerate_flags(n, k)

    for sigma_local, flag in flags:
        # Local indices into the top simplex vertex list
        sigma_idx = torch.tensor(sigma_local, dtype=torch.long, device=device)

        # Global vertex ids of σ for each top simplex: (N_n, k+1)
        sigma_verts_global = top_cells[:, sigma_idx]

        # Sort rows to canonical form for lookup
        sigma_verts_sorted, _ = sigma_verts_global.sort(dim=-1)

        # Lookup global k-simplex ids: (N_n,)
        sigma_global_ids = row.lookup_row_indices(
            sigma_verts_sorted.to(torch.long),
            k_cells.to(torch.long),
            keys_sorted,
            perm,
        )

        # Compute circumcenters of all faces in this flag
        # flag has n-k+1 entries (j = k, k+1, ..., n)
        # circumcenters: list of (N_n, m) tensors
        circumcenters = []
        for face_local in flag:
            face_idx = torch.tensor(face_local, dtype=torch.long, device=device)
            face_verts = vertex_pos[top_cells[:, face_idx]]   # (N_n, j+1, m)
            cc = _circumcenter(face_verts)                     # (N_n, m)
            circumcenters.append(cc)

        # Stack into (N_n, n-k+1, m) and compute flag simplex volume
        cc_tensor = torch.stack(circumcenters, dim=1)   # (N_n, n-k+1, m)
        vol = _simplex_volume(cc_tensor)                # (N_n,)

        # Accumulate: dual_measure[sigma_id] += vol  for each top simplex
        dual_measure.scatter_add_(0, sigma_global_ids, vol)

    return dual_measure


# ------------------------------------------------------------------
# Main class
# ------------------------------------------------------------------


class SimplicialDualCellMeasure:
    """Circumcentric dual cell measures for a PL-simplicial complex.

    Stores, for each degree k, a (n_k,) tensor of dual cell measures
    computed via the Euclidean circumcentric construction. The dual
    complex is the standard DEC circumcentric dual.

    This class is specific to simplicial complexes embedded in
    Euclidean space. For general CW complexes, additional geometric
    information per cell would be required.

    Args:
        measures (dict[int, Tensor]): mapping from degree k to (n_k,) tensor.
        n (int): topological dimension of the complex.
    """

    def __init__(
        self,
        measures: dict[int, Tensor],
        n: int,
    ):
        self._measures = measures
        self._n = n

    @property
    def dim(self) -> int:
        """Topological dimension."""
        return self._n

    def measure(self, k: int) -> Tensor:
        """Return dual cell measures for degree k.

        Args:
            k (int): simplex degree.

        Returns:
            (n_k,) tensor of dual cell volumes.

        Raises:
            KeyError: if degree k has no precomputed measure.
        """
        if k not in self._measures:
            raise KeyError(
                f"No dual measure precomputed for k={k}. "
                f"Available degrees: {sorted(self._measures)}"
            )
        return self._measures[k]

    def device(self) -> torch.device:
        """Device of the measure tensors."""
        return next(iter(self._measures.values())).device

    def to(self, *args, **kwargs) -> SimplicialDualCellMeasure:
        """Move all measure tensors to the specified device/dtype."""
        measures = {k: v.to(*args, **kwargs) for k, v in self._measures.items()}
        return SimplicialDualCellMeasure(measures, self._n)

    def cpu(self) -> SimplicialDualCellMeasure:
        """Move all measure tensors to CPU."""
        return self.to("cpu")

    def cuda(self) -> SimplicialDualCellMeasure:
        """Move all measure tensors to CUDA."""
        return self.to("cuda")

    @classmethod
    def from_circumcentric(
        cls,
        indexing: SimplicialChainComplex,
        vertex_positions: CoChain,
    ) -> SimplicialDualCellMeasure:
        """Build circumcentric dual measures from an embedded simplicial complex.

        Computes dual measures for all degrees k=0..n using the standard
        DEC circumcentric construction: the dual of each k-simplex is a
        polytope whose volume is computed as a sum over flag simplices built
        from circumcenters of nested sub-simplices.

        Note: this is an approximation for non-Delaunay complexes, where
        some circumcenters may fall outside top simplices.

        Args:
            indexing (SimplicialChainComplex): simplicial complex providing
                cell vertex tables for all degrees.
            vertex_positions (CoChain): k=0 cochain with data of shape
                (n_vertices, m) containing Euclidean vertex coordinates in ℝ^m.

        Returns:
            SimplicialDualCellMeasure with measures for k=0..n.

        Raises:
            ValueError: if vertex_positions.k != 0.
        """
        if vertex_positions.k != 0:
            raise ValueError(
                f"vertex_positions must be a 0-cochain, got k={vertex_positions.k}"
            )

        n = indexing.dim
        pos = vertex_positions.data   # (n_0, m)

        # Top simplices drive the whole construction
        top_cells = indexing.cells(n).to(torch.long)

        measures: dict[int, Tensor] = {}
        for k in range(n + 1):
            k_cells = indexing.cells(k).to(torch.long)
            measures[k] = _compute_dual_measure_k(
                top_cells=top_cells,
                k_cells=k_cells,
                vertex_pos=pos,
                n=n,
                k=k,
            )

        return cls(measures, n)
