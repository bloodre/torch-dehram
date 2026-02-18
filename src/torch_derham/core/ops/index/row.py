"""Hashing and row-index utilities for simplex vertex tables.

This module provides fast, GPU-friendly row hashing and exact row lookup via
(hash -> candidate range -> verify) without leaving PyTorch.

Currently uses SplitMix64-style mixing.

Key design goals:
- No Python for-loops in hashing or lookup.
- CPU/CUDA support.
- Fast-path when every query has a unique candidate (common case).
- Collision-safe exactness via verification when candidate ranges are > 1.

Notes:
- The hash is not required to be collision-free: verification guarantees correctness.
- We cache small uint64 constants and per-(device, width) salts to avoid recreating
  tensors every call. This is lightweight and avoids the complexity of nn.Module
  buffers (which are more useful when parameters/buffers must move with .to()).

Example
-------
Build an index for a table of (k-1)-simplices (children), then map faces to ids:

    >>> import torch
    >>> from torch_derham.core.index.row import (
    ...     build_sorted_row_index,
    ...     lookup_row_indices,
    ... )
    >>>
    >>> # children: (n_children, k) vertex lists, sorted within each row
    >>> children = torch.tensor([[0, 1],
    ...                         [0, 2],
    ...                         [1, 2],
    ...                         [2, 3]], device="cuda", dtype=torch.int64)
    >>>
    >>> keys_sorted, perm = build_sorted_row_index(children, seed=0)
    >>>
    >>> # faces to lookup (m, k)
    >>> faces = torch.tensor([[1, 2],
    ...                       [0, 2],
    ...                       [2, 3]], device="cuda", dtype=torch.int64)
    >>>
    >>> idx = lookup_row_indices(faces, children, keys_sorted, perm, seed=0)
    >>> idx
    tensor([2, 1, 3], device='cuda:0')

Notes
-----
- All inputs are assumed to represent exact integer rows; floating point is not
  supported.
- `seed` must match between `build_sorted_row_index` and `lookup_row_indices`.
- If `rows` contains duplicates, lookups may raise an error (ambiguous match).
"""

from __future__ import annotations

import torch

from torch import Tensor


# ----------------------------------------------------------------------
# Small device caches (constants, salts)
# ----------------------------------------------------------------------

_U64_CONST_CACHE: dict[torch.device, dict[str, Tensor]] = {}
_SALT_CACHE: dict[tuple[torch.device, int], tuple[Tensor, Tensor]] = {}


def _u64(device: torch.device, value: int) -> Tensor:
    """Create (or reuse) a uint64 scalar tensor on device."""
    # Scalars are tiny; caching avoids repeated allocations in hot paths.
    key = device
    d = _U64_CONST_CACHE.get(key)
    if d is None:
        d = {}
        _U64_CONST_CACHE[key] = d

    name = f"c_{value & 0xFFFFFFFFFFFFFFFF:016x}"
    t = d.get(name)
    if t is None:
        t = torch.tensor(value & 0xFFFFFFFFFFFFFFFF, dtype=torch.uint64, device=device)
        d[name] = t
    return t


def _get_mix_constants(device: torch.device) -> tuple[Tensor, Tensor, Tensor]:
    """SplitMix64 constants as cached uint64 scalars."""
    c1 = _u64(device, 0x9E3779B97F4A7C15)  # Golden ratio
    c2 = _u64(device, 0xBF58476D1CE4E5B9)  # Mixing multiplier
    c3 = _u64(device, 0x94D049BB133111EB)  # Mixing multiplier
    return c1, c2, c3


def _splitmix64(x: Tensor) -> Tensor:
    """SplitMix64 mixing function (vectorized, uint64)."""
    if x.dtype != torch.uint64:
        raise ValueError(f"expected uint64, got {x.dtype}")

    c1, c2, c3 = _get_mix_constants(x.device)
    x = x + c1
    x = (x ^ (x >> 30)) * c2
    x = (x ^ (x >> 27)) * c3
    x = x ^ (x >> 31)
    return x


def _get_salt_and_weights(device: torch.device, width: int) -> tuple[Tensor, Tensor]:
    """Get (salt, weights) for a given row width, cached per device.

    Args:
        device: CPU/CUDA.
        width: number of columns.

    Returns:
        salt: (width,) uint64
        weights: (width,) uint64
    """
    key = (device, int(width))
    cached = _SALT_CACHE.get(key)
    if cached is not None:
        return cached

    # Column-dependent salt -> order sensitivity.
    j = torch.arange(width, device=device, dtype=torch.uint64)

    # Two arbitrary, fixed constants to diversify streams.
    c_salt = _u64(device, 0xD6E8FEB86659FD93)
    c_w = _u64(device, 0xA5A3564E27F3A7D1)

    salt = _splitmix64(j + c_salt)
    weights = _splitmix64(salt ^ c_w)

    _SALT_CACHE[key] = (salt, weights)
    return salt, weights


# ----------------------------------------------------------------------
# Hashing
# ----------------------------------------------------------------------


def row_hash_u64(rows: Tensor, seed: int = 0) -> Tensor:
    """Compute a vectorized 64-bit hash key per row (no Python loops).

    Args:
        rows: (n, w) integer tensor.
        seed: optional seed (int) folded into the final key.

    Returns:
        (n,) uint64 tensor of row hash keys on the same device.
    """
    if rows.ndim != 2:
        raise ValueError(f"expected 2D tensor, got {rows.ndim}D")

    x = rows.to(dtype=torch.uint64)
    n, w = x.shape
    if w == 0:
        return torch.full(
            (n,),
            fill_value=(seed & 0xFFFFFFFFFFFFFFFF),
            dtype=torch.uint64,
            device=x.device,
        )

    salt, weights = _get_salt_and_weights(x.device, w)

    # Mix each element with its column salt (order-sensitive).
    m = _splitmix64(x ^ salt)  # (n, w)

    # Combine columns in one shot. Wraparound is modulo 2^64 by dtype.
    h = (m * weights).sum(dim=1)  # (n,)

    # Fold in seed and width, then final mix.
    seed_u = _u64(x.device, seed)
    w_u = _u64(x.device, w)
    return _splitmix64(h ^ seed_u ^ w_u)


def u64_to_ordered_i64(keys: Tensor) -> Tensor:
    """Convert uint64 keys to int64 keys with the same total order.

    This allows sort/searchsorted on int64 while preserving uint64 ordering.
    """
    if keys.dtype != torch.uint64:
        raise ValueError(f"expected uint64, got {keys.dtype}")
    sign_bit = _u64(keys.device, 0x8000000000000000)
    return (keys ^ sign_bit).to(dtype=torch.int64)


# ----------------------------------------------------------------------
# Row index + exact lookup (hash + verify)
# ----------------------------------------------------------------------


def build_sorted_row_index(
    rows: Tensor,
    *,
    stable: bool = True,
    seed: int = 0,
) -> tuple[Tensor, Tensor]:
    """Build a sorted key index for (n,w) rows.

    Args:
        rows: (n, w) integer tensor.
        stable: whether to request stable sorting.
        seed: hash seed.

    Returns:
        keys_sorted: (n,) int64 tensor of sorted keys (order-preserving).
        perm: (n,) int64 tensor mapping sorted positions -> original row indices.
    """
    keys = u64_to_ordered_i64(row_hash_u64(rows, seed=seed))
    perm = torch.argsort(keys, stable=stable)
    return keys[perm], perm.to(dtype=torch.int64)


def lookup_row_indices(
    queries: Tensor,
    rows: Tensor,
    keys_sorted: Tensor,
    perm: Tensor,
    *,
    seed: int = 0,
    validate: bool = False,
) -> Tensor:
    """Exact lookup of query rows in a row table using hash + verify.

    Fast path: if each query's candidate range has length 1, returns after a
    single batched verification (no ragged expansion).

    Slow path: only for ambiguous keys (candidate range length > 1), expands
    those candidates and verifies in a single vectorized pass (no Python loops).

    Args:
        queries: (m, w) integer tensor of query rows.
        rows: (n, w) integer tensor of indexed rows (same width as queries).
        keys_sorted: (n,) int64 sorted keys from build_sorted_row_index.
        perm: (n,) int64 permutation from build_sorted_row_index.
        seed: hash seed (must match build_sorted_row_index).
        validate: whether to validate the input.

    Returns:
        (m,) int64 tensor of row indices in `rows` for each query.

    Raises:
        ValueError: if any query row is not found or matches multiple rows.
    """
    if validate:
        if queries.ndim != 2 or rows.ndim != 2:
            raise ValueError("queries and rows must be 2D tensors")
        if queries.size(1) != rows.size(1):
            raise ValueError(
                f"width mismatch: queries w={queries.size(1)} vs rows w={rows.size(1)}"
            )
        if keys_sorted.ndim != 1 or perm.ndim != 1:
            raise ValueError("keys_sorted and perm must be 1D tensors")
        if keys_sorted.numel() != rows.size(0) or perm.numel() != rows.size(0):
            raise ValueError("keys_sorted/perm size must match number of rows")
        if keys_sorted.dtype != torch.int64 or perm.dtype != torch.int64:
            raise ValueError("keys_sorted and perm must be int64")

    device = rows.device
    q = queries.to(device=device)

    q_keys = u64_to_ordered_i64(row_hash_u64(q, seed=seed))

    left = torch.searchsorted(keys_sorted, q_keys, right=False)
    right = torch.searchsorted(keys_sorted, q_keys, right=True)
    lens = (right - left).to(dtype=torch.int64)

    # Any missing?
    if (lens == 0).any():
        bad = int(torch.nonzero(lens == 0, as_tuple=False)[0, 0].cpu().item())
        raise ValueError(f"1 face not found in rows (example idx={bad})")

    m = q.size(0)
    out = torch.empty((m,), dtype=torch.int64, device=device)

    # ------------------------------------------------------------------
    # Fast path: unique candidate for all (or most) queries.
    # ------------------------------------------------------------------
    uniq_mask = lens == 1
    if uniq_mask.all():
        sorted_pos = left
        cand = perm[sorted_pos]
        ok = (rows.index_select(0, cand) == q).all(dim=1)
        if not ok.all():
            bad = int(torch.nonzero(~ok, as_tuple=False)[0, 0].cpu().item())
            raise ValueError(f"1 face not found in rows (example idx={bad})")
        return cand

    # Fill unique ones in a batched way.
    uniq_idx = torch.nonzero(uniq_mask, as_tuple=False).flatten()
    sorted_pos_u = left.index_select(0, uniq_idx)
    cand_u = perm.index_select(0, sorted_pos_u)
    ok_u = (rows.index_select(0, cand_u) == q.index_select(0, uniq_idx)).all(dim=1)
    if not ok_u.all():
        bad_local = int(torch.nonzero(~ok_u, as_tuple=False)[0, 0].cpu().item())
        bad = int(uniq_idx[bad_local].cpu().item())
        raise ValueError(f"1 face not found in rows (example idx={bad})")
    out.index_copy_(0, uniq_idx, cand_u)

    # ------------------------------------------------------------------
    # Slow path: ambiguous candidate ranges only (lens > 1), vectorized.
    # ------------------------------------------------------------------
    amb_mask = ~uniq_mask  # lens >= 2
    amb_idx = torch.nonzero(amb_mask, as_tuple=False).flatten()  # (m_amb,)

    left_a = left.index_select(0, amb_idx)
    lens_a = lens.index_select(0, amb_idx)
    m_a = amb_idx.numel()

    # Expand candidate positions for ambiguous queries:
    # For each i in [0..m_a), generate positions left_a[i] + t for t in [0..lens_a[i)-1].
    total = int(lens_a.sum().cpu().item())

    # qid in [0..m_a) repeated by lens_a.
    qid = torch.repeat_interleave(torch.arange(m_a, device=device), lens_a)  # (total,)

    # pos_in_run computed from global arange minus repeated prefix offsets.
    prefix = torch.cumsum(lens_a, dim=0) - lens_a  # (m_a,)
    pos_in_run = torch.arange(total, device=device) - torch.repeat_interleave(prefix, lens_a)
    sorted_pos = left_a.index_select(0, qid) + pos_in_run  # (total,)

    cand = perm.index_select(0, sorted_pos)  # (total,)
    cand_rows = rows.index_select(0, cand)   # (total, w)
    q_rows = q.index_select(0, amb_idx.index_select(0, qid))  # (total, w)

    eq = (cand_rows == q_rows).all(dim=1)  # (total,)

    # Each ambiguous query must match exactly one candidate.
    counts = torch.zeros((m_a,), dtype=torch.int64, device=device)
    counts.scatter_add_(0, qid, eq.to(dtype=torch.int64))
    if (counts != 1).any():
        bad_local = int(torch.nonzero(counts != 1, as_tuple=False)[0, 0].cpu().item())
        bad = int(amb_idx[bad_local].cpu().item())
        raise ValueError(
            f"query idx={bad} has {int(counts[bad_local].cpu().item())} matches "
            "(missing or duplicate rows)"
        )

    # Pick matching candidate index per ambiguous query using scatter_reduce.
    picked = torch.full((m_a,), -1, dtype=torch.int64, device=device)
    picked.scatter_reduce_(
        0,
        qid,
        torch.where(eq, cand, torch.full_like(cand, -1)),
        reduce="amax",
        include_self=True,
    )

    out.index_copy_(0, amb_idx, picked)
    return out
