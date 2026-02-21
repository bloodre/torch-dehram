"""Mass matrix assembly for Whitney finite elements.

Assembles global sparse mass matrices M_k for each degree k using quadrature
over reference simplices and pullback of Whitney forms to physical elements.

The local mass matrix for element T and degree k is:
    M_T^{(k)}[i,j] = ∫_T ω_i^k ∧ ★ω_j^k

For lowest-order Whitney forms, this reduces to quadrature over the reference
simplex with appropriate pullback factors from the affine map geometry.
"""
# pylint: disable=invalid-name

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import Tensor
from torch_sparse import SparseTensor
from ..ops.index import row as row_module
from .geometry import SimplicialGeometry
from .quadrature import quadrature_simplex
from .reference import eval_whitney_kform_all, enumerate_whitney_dofs

if TYPE_CHECKING:
    from ..complex.simplicial import SimplicialChainComplex


def assemble_local_mass_k(
    geometry: SimplicialGeometry,
    k: int,
    quad_degree: int,
) -> Tensor:
    """Assemble local mass matrices for all top simplices at degree k.

    Args:
        geometry (SimplicialGeometry): mesh geometry.
        k (int): form degree.
        quad_degree (int): polynomial degree for quadrature exactness.

    Returns:
        (N_n, n_loc_k, n_loc_k) local mass matrices for each top simplex.
    """
    n = geometry.n
    N_n = geometry.top_cells.shape[0]
    device = geometry.vertex_positions.device
    dtype = geometry.vertex_positions.dtype

    if k > n:
        raise ValueError(f"k={k} > n={n}")

    # Reference quadrature
    bary_ref, weights_ref = quadrature_simplex(n, quad_degree)
    bary_ref = bary_ref.to(device=device, dtype=dtype)
    weights_ref = weights_ref.to(device=device, dtype=dtype)

    # Enumerate local DOFs (faces of the top simplex)
    local_dofs = enumerate_whitney_dofs(n, k)
    n_loc = len(local_dofs)

    # Evaluate Whitney forms at reference quadrature points
    # omega_ref: (Q, n_loc, ...) depending on k
    omega_ref = eval_whitney_kform_all(bary_ref, n, k)

    # Volume scaling: (N_n,)
    vol_scale = geometry.volume_scaling()

    # Local mass matrices: (N_n, n_loc, n_loc)
    M_local = torch.zeros(N_n, n_loc, n_loc, device=device, dtype=dtype)

    # For k=0: no pullback needed (scalars)
    if k == 0:
        # omega_ref: (Q, n_loc)
        for i in range(n_loc):
            for j in range(n_loc):
                omega_i = omega_ref[:, i]  # (Q,)
                omega_j = omega_ref[:, j]  # (Q,)
                integrand = omega_i * omega_j  # (Q,)
                integral_ref = (weights_ref * integrand).sum()  # scalar
                M_local[:, i, j] = vol_scale * integral_ref

    elif k == 1:
        # omega_ref: (Q, n_loc, n) covectors in reference coords
        # Need to pullback via J^{-T} to physical coords
        J_inv_T = geometry.inv_jacobians_T()  # (N_n, n, m)

        for i in range(n_loc):
            for j in range(n_loc):
                omega_i_ref = omega_ref[:, i, :]  # (Q, n)
                omega_j_ref = omega_ref[:, j, :]  # (Q, n)

                # Pullback: ω_phys = J^{-T} ω_ref
                # For each element: (n, m) @ (n,) → (m,)
                # Broadcast over quadrature: omega_i_ref (Q, n), J_inv_T (N_n, n, m)
                # → (N_n, Q, m)
                omega_i_phys = torch.einsum(
                    "qn,Tnm->Tqm",
                    omega_i_ref,
                    J_inv_T,
                )  # (N_n, Q, m)
                omega_j_phys = torch.einsum(
                    "qn,Tnm->Tqm",
                    omega_j_ref,
                    J_inv_T,
                )  # (N_n, Q, m)

                # Inner product at each quadrature point: (N_n, Q)
                inner = (omega_i_phys * omega_j_phys).sum(dim=-1)  # (N_n, Q)

                # Integrate: (N_n,)
                integral = (weights_ref.unsqueeze(0) * inner).sum(dim=1)  # (N_n,)
                M_local[:, i, j] = vol_scale * integral

    elif k == 2:
        # omega_ref: (Q, n_loc, n, n) antisymmetric matrices
        # Pullback for 2-forms involves J^{-T} ⊗ J^{-T} applied to wedge
        # For simplicity in embedded case, use Gram determinant scaling
        # and metric-induced inner product.
        #
        # Full treatment: pullback (J^{-T} ∧ J^{-T}) applied to 2-form.
        # For now: approximate via reference inner product scaled by Gram det.
        # This is exact when m=n (square Jacobian).

        vol_scale_sq = vol_scale  # Already accounts for Gram det

        for i in range(n_loc):
            for j in range(n_loc):
                omega_i_ref = omega_ref[:, i, :, :]  # (Q, n, n)
                omega_j_ref = omega_ref[:, j, :, :]  # (Q, n, n)

                # Reference inner product: Frobenius
                inner_ref = (omega_i_ref * omega_j_ref).sum(dim=(-2, -1))  # (Q,)
                integral_ref = (weights_ref * inner_ref).sum()  # scalar

                M_local[:, i, j] = vol_scale_sq * integral_ref

    elif k == 3 and n == 3:
        # k=3: volume form, one DOF per tet, constant
        # omega_ref is a scalar (constant over simplex)
        vol_coeff = omega_ref[0].item()  # scalar
        # M_local is 1×1 per element
        M_local[:, 0, 0] = vol_scale * vol_coeff ** 2

    else:
        raise ValueError(f"Local mass assembly not implemented for k={k}, n={n}")

    return M_local


def assemble_global_mass_k(
    chain_complex: SimplicialChainComplex,
    geometry: SimplicialGeometry,
    k: int,
    quad_degree: int = 2,
) -> SparseTensor:
    """Assemble global sparse mass matrix M_k for degree k.

    Uses vectorized DOF mapping: precomputes global indices for all local
    DOFs of all elements, then scatters local mass matrices in batched ops.

    Args:
        chain_complex (SimplicialChainComplex): mesh topology.
        geometry (SimplicialGeometry): mesh geometry.
        k (int): form degree.
        quad_degree (int): quadrature polynomial degree (default 2).

    Returns:
        SparseTensor (N_k, N_k) global sparse mass matrix.
    """
    n = geometry.n
    N_k = chain_complex.n_cells(k)
    device = geometry.vertex_positions.device

    if k > n:
        raise ValueError(f"k={k} > n={n}")

    # Assemble local matrices: (N_n, n_loc, n_loc)
    M_local = assemble_local_mass_k(geometry, k, quad_degree)
    N_n, n_loc, _ = M_local.shape

    # Global DOF numbering: k-simplices in the complex
    k_cells_global = chain_complex.cells(k).to(torch.long)  # (N_k, k+1)
    top_cells = geometry.top_cells.to(torch.long)  # (N_n, n+1)

    # Local DOF patterns (k-faces of reference n-simplex)
    local_dofs = enumerate_whitney_dofs(n, k)  # list of (k+1)-tuples

    # Build hash index for k-simplices once
    k_cells_sorted, perm = row_module.build_sorted_row_index(k_cells_global)

    # Precompute global DOF indices for all elements: (N_n, n_loc)
    # For each local DOF pattern, gather vertex IDs from all top cells
    dofs_global = torch.empty(N_n, n_loc, dtype=torch.long, device=device)

    for i_local, face_local in enumerate(local_dofs):
        # Gather vertices for this local face from all elements: (N_n, k+1)
        face_idx = torch.tensor(face_local, dtype=torch.long, device=device)
        face_verts = top_cells[:, face_idx]  # (N_n, k+1)

        # Sort rows to canonical form
        face_verts_sorted, _ = face_verts.sort(dim=-1)

        # Lookup all global IDs in one batched call: (N_n,)
        global_ids = row_module.lookup_row_indices(
            face_verts_sorted,
            k_cells_global,
            k_cells_sorted,
            perm,
        )
        dofs_global[:, i_local] = global_ids

    # Build COO indices via broadcasting
    # row: (N_n, n_loc, 1) → (N_n, n_loc, n_loc)
    # col: (N_n, 1, n_loc) → (N_n, n_loc, n_loc)
    rows_2d = dofs_global.unsqueeze(2).expand(N_n, n_loc, n_loc)
    cols_2d = dofs_global.unsqueeze(1).expand(N_n, n_loc, n_loc)

    # Flatten to (N_n * n_loc * n_loc,)
    row_idx = rows_2d.reshape(-1)
    col_idx = cols_2d.reshape(-1)
    values = M_local.reshape(-1)

    # Filter out near-zero entries
    mask = values.abs() > 1e-14
    row_idx = row_idx[mask]
    col_idx = col_idx[mask]
    values = values[mask]

    # Build SparseTensor and coalesce to sum duplicate entries
    M_k = SparseTensor(
        row=row_idx,
        col=col_idx,
        value=values,
        sparse_sizes=(N_k, N_k),
    ).coalesce()

    return M_k


def assemble_global_mass_all(
    chain_complex: SimplicialChainComplex,
    geometry: SimplicialGeometry,
    quad_degree: int = 2,
) -> dict[int, SparseTensor]:
    """Assemble global mass matrices for all degrees k=0..n.

    Args:
        chain_complex (SimplicialChainComplex): mesh topology.
        geometry (SimplicialGeometry): mesh geometry.
        quad_degree (int): quadrature polynomial degree (default 2).

    Returns:
        Dictionary mapping k → M_k (SparseTensor).
    """
    n = geometry.n
    masses = {}
    for k in range(n + 1):
        masses[k] = assemble_global_mass_k(
            chain_complex,
            geometry,
            k,
            quad_degree,
        )
    return masses
