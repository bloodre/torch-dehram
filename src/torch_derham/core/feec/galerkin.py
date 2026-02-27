"""Galerkin FEEC interior product via Whitney form reconstruction and quadrature.

High-level definition
---------------------
We implement the discrete interior product operator ``i_X: C^k -> C^{k-1}`` as

    ``C_k(X) = M_{k-1}^{-1} B_k(X)``

where ``B_k(X)`` is a coupling matrix assembled by quadrature using Whitney
forms on each top cell ``T``:

    ``(B_k(X))_{a,b} = ∫_T < i_X w_b^{(k)}, w_a^{(k-1)} > dV``

Representation and conventions
------------------------------
- The vector field ``X`` is represented as a Whitney 0-form (piecewise affine),
  i.e. vertex samples interpolated barycentrically inside each element.
- Whitney k-forms are evaluated on the reference simplex, then pulled back to
  physical coordinates using Jacobians from ``SimplicialGeometry``.
- Inner products follow the same conventions as the Whitney mass matrix, so
  k-forms of order 2 use Frobenius products of their antisymmetric matrices.

Important implementation notes
------------------------------
- Orientation: when mapping local DOFs to global DOF IDs, we must apply the
  orientation sign induced by the local vertex order relative to the global
  canonical order of the corresponding oriented k-cell. We compute a ±1 factor
  per local DOF and multiply into the local tensor before scattering.
- Pullback: a covariant k-form pulls back with ``J^{-T}`` applied to each
  covariant index. For 2-forms in particular, this is
  ``ω^phys_ab = Σ_ij (J^{-T})_ai (J^{-T})_bj ω^ref_ij``.
- Quadrature: we integrate per element using pre-tabulated reference rules and
  multiply by the element volume scaling factor.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import Tensor
from torch_sparse import SparseTensor

from ..ops.index import row as row_module
from ..cochain import CoChain
from .geometry import SimplicialGeometry
from .quadrature import quadrature_simplex
from .reference import enumerate_whitney_dofs, eval_whitney_kform_all

if TYPE_CHECKING:
    from ..complex.simplicial import SimplicialChainComplex
    from .inner_product import FEECInnerProduct


def _ensure_J_inv_T_shape(
    geometry: SimplicialGeometry,
) -> Tensor:
    """Return J^{-T} with shape (n_elems, m, n).

    Some codebases store this as (n_elems, n, m). We normalize here.

    Args:
        geometry: Mesh geometry providing Jacobian information.

    Returns:
        Tensor with shape (n_elems, m, n) representing J^{-T}.

    Raises:
        ValueError: If inv_jacobians_T is not rank-3 or has unexpected shape.
    """
    J_inv_T = geometry.inv_jacobians_T()
    n = geometry.n
    m = geometry.vertex_positions.shape[1]

    if J_inv_T.ndim != 3:
        raise ValueError(f"inv_jacobians_T must be rank-3, got shape {tuple(J_inv_T.shape)}")

    # Expect either (T, m, n) or (T, n, m) (square case collapses ambiguity)
    if J_inv_T.shape[1] == m and J_inv_T.shape[2] == n:
        return J_inv_T
    if J_inv_T.shape[1] == n and J_inv_T.shape[2] == m:
        return J_inv_T.transpose(1, 2)

    raise ValueError(
        "inv_jacobians_T has unexpected shape. "
        f"Expected (T,{m},{n}) or (T,{n},{m}), got {tuple(J_inv_T.shape)}"
    )


def _parity_sign_local_to_global(
    local_verts: Tensor,
    global_verts: Tensor,
) -> Tensor:
    """Compute orientation sign for local-to-global DOF mapping.

    Explanation
    -----------
    Whitney DOFs are associated with oriented faces (k-simplices). When we map
    a local face (given by the order of its vertices inside a top cell) to a
    global face (stored in canonical, sorted order), we must include the sign
    of the permutation that takes the local ordering to the global ordering.
    This parity is the orientation sign and must be applied when scattering
    element tensors to the global matrix.

    Args:
        local_verts: Tensor with shape ``(N, L)`` listing the local vertex
            indices for each of ``N`` faces as they occur inside the element.
        global_verts: Tensor with shape ``(N, L)`` listing the same vertex
            indices for each face but in canonical (stored) order.

    Returns:
        Tensor with shape ``(N,)`` of values in ``{+1, -1}``.
    """
    if local_verts.shape != global_verts.shape:
        raise ValueError(
            f"local_verts and global_verts must have same shape, got "
            f"{tuple(local_verts.shape)} vs {tuple(global_verts.shape)}"
        )

    L = local_verts.size(1)
    device = local_verts.device

    # pos[i, r] = index in global_verts[i, :] where local_verts[i, r] occurs
    # Shape: (N, L)
    matches = (global_verts.unsqueeze(1) == local_verts.unsqueeze(2))  # (N, L, L)
    # Each local vertex should match exactly one global position
    pos = matches.to(torch.long).argmax(dim=2)

    # Count inversions in pos for each row
    # inv = #{(r,s): r<s and pos[r] > pos[s]}
    pos_r = pos.unsqueeze(2)  # (N, L, 1)
    pos_s = pos.unsqueeze(1)  # (N, 1, L)
    inv_mat = pos_r > pos_s  # (N, L, L)

    # Only count upper-triangular pairs (r < s)
    triu_mask = torch.triu(torch.ones(L, L, device=device, dtype=torch.bool), diagonal=1)
    inv = (inv_mat & triu_mask.unsqueeze(0)).sum(dim=(1, 2))  # (N,)

    # (-1)^inv
    sign = torch.where((inv % 2) == 0, 1.0, -1.0).to(torch.float32)
    return sign


def _contract_1forms(
    omega_1: Tensor,
    x_phys: Tensor,
) -> Tensor:
    """Contract a 1-form with a vector field (scalar result).

    Computes the pointwise contraction ``i_X(ω) = ω(X)`` for each quadrature
    point, element, and local k-form basis function.

    Args:
        omega_1: Tensor with shape ``(T, Q, n_loc_k, m)`` of 1-forms in physical coords.
        x_phys: Tensor with shape ``(T, Q, m)`` of the vector field in physical coords.

    Returns:
        Tensor with shape ``(T, Q, n_loc_k)`` of scalar results.
    """
    return torch.einsum("Tqlm,Tqm->Tql", omega_1, x_phys)


def _contract_2forms_first_slot(
    omega_2: Tensor,
    x_phys: Tensor,
) -> Tensor:
    """Contract a 2-form with a vector field (1-form result).

    We represent a 2-form by its antisymmetric matrix ``A`` so that
    ``(i_X A)_b = Σ_a A_ab X^a``. This matches the mass matrix convention
    where 2-forms use Frobenius inner products of their matrix representations.

    Args:
        omega_2: Tensor with shape ``(T, Q, n_loc_k, m, m)`` containing
            antisymmetric matrices.
        x_phys: Tensor with shape ``(T, Q, m)`` containing the vector field.

    Returns:
        Tensor with shape ``(T, Q, n_loc_k, m)`` of 1-forms.
    """
    # contract on the FIRST index of omega_2: sum_a omega_{a b} X^a
    return torch.einsum("Tqlab,Tqa->Tqlb", omega_2, x_phys)


def _contract_3form_volume(
    coeff: Tensor,
    x_phys: Tensor,
) -> Tensor:
    """Contract a 3-form with a vector field (2-form result).

    In 3D, the volume form is ``vol = dx ∧ dy ∧ dz`` and a 3-form is
    ``ω = c * vol``. The contraction satisfies ``i_X(ω) = c * ⋆X``, i.e., the
    2-form which is the Hodge dual of ``X`` scaled by ``c``. We produce an
    antisymmetric matrix representation of this 2-form.

    Args:
        coeff: Tensor with shape ``(T,)`` of scalar coefficients (one per element,
            in physical coordinates).
        x_phys: Tensor with shape ``(T, Q, 3)`` of vector values.

    Returns:
        Tensor with shape ``(T, Q, 1, 3, 3)`` of antisymmetric matrices
        representing 2-forms.
    """
    T, Q, _ = x_phys.shape
    device = x_phys.device
    dtype = x_phys.dtype

    out = torch.zeros(T, Q, 1, 3, 3, device=device, dtype=dtype)

    c = coeff.view(T, 1)  # (T,1)
    x0 = x_phys[:, :, 0]
    x1 = x_phys[:, :, 1]
    x2 = x_phys[:, :, 2]

    out[:, :, 0, 0, 1] = c * x2
    out[:, :, 0, 0, 2] = -c * x1
    out[:, :, 0, 1, 2] = c * x0

    # antisym completion
    out[:, :, 0, 1, 0] = -out[:, :, 0, 0, 1]
    out[:, :, 0, 2, 0] = -out[:, :, 0, 0, 2]
    out[:, :, 0, 2, 1] = -out[:, :, 0, 1, 2]

    return out


def assemble_local_B_k(
    geometry: SimplicialGeometry,
    X_vertex: Tensor,
    k: int,
    quad_degree: int,
) -> Tensor:
    """Assemble local coupling tensors ``B_k(X)`` for all top simplices.

    Definition
    ----------
    For each element ``T`` (top simplex), assemble the matrix
    ``B_local[T, a, b]`` defined by the integral
    ``∫_T < i_X w_b^{(k)}, w_a^{(k-1)} > dV``.

    Implementation outline
    ----------------------
    1. Tabulate Whitney k-forms on the reference simplex at quadrature points.
    2. Pull back to physical coordinates via ``J^{-T}`` (once per covariant index).
    3. Interpolate ``X`` at quadrature points using Whitney-0 barycentric
       interpolation of the vertex values ``X_vertex`` gathered per element.
    4. Compute pointwise contractions ``i_X w_b^{(k)}``.
    5. Compute inner products with ``w_a^{(k-1)}`` (already in physical coords).
    6. Integrate with quadrature weights and multiply by ``volume_scaling``.

    Shapes
    ------
    - Returns a tensor of shape ``(n_elems, n_loc_{k-1}, n_loc_k)``.

    Args:
        geometry: Mesh geometry (Jacobian, volume scaling, top cells).
        X_vertex: Tensor with shape ``(n_vertices, m)`` of vertex samples of ``X``.
        k: Form degree (``k >= 1``).
        quad_degree: Reference quadrature degree.

    Returns:
        Tensor with shape ``(n_elems, n_loc_{k-1}, n_loc_k)`` of local entries.
    """
    n = geometry.n
    n_elems = geometry.top_cells.shape[0]
    device = geometry.vertex_positions.device
    dtype = geometry.vertex_positions.dtype
    m = geometry.vertex_positions.shape[1]

    if k < 1 or k > n:
        raise ValueError(f"k must be in [1, {n}], got {k}")

    # Quadrature on reference n-simplex
    bary_ref, weights_ref = quadrature_simplex(n, quad_degree)
    bary_ref = bary_ref.to(device=device, dtype=dtype)        # (Q, n+1)
    weights_ref = weights_ref.to(device=device, dtype=dtype)  # (Q,)

    # Local DOFs on reference simplex (counts = C(n+1,k+1))
    local_dofs_k = enumerate_whitney_dofs(n, k)
    local_dofs_km1 = enumerate_whitney_dofs(n, k - 1)
    n_loc_k = len(local_dofs_k)
    n_loc_km1 = len(local_dofs_km1)

    # Reference Whitney basis evaluations at quadrature points
    omega_k_ref = eval_whitney_kform_all(bary_ref, n, k)
    omega_km1_ref = eval_whitney_kform_all(bary_ref, n, k - 1)

    # Element volume scaling (must match how your mass matrix integrates)
    vol_scale = geometry.volume_scaling().to(device=device, dtype=dtype)  # (T,)

    # Whitney-0 interpolation of X at quadrature points
    x_verts = X_vertex[geometry.top_cells]  # (T, n+1, m)
    x_quad = torch.einsum("qi,Tia->Tqa", bary_ref, x_verts)  # (T, Q, m)

    # J^{-T} normalized shape (T, m, n)
    J_inv_T = _ensure_J_inv_T_shape(geometry).to(device=device, dtype=dtype)

    b_local = torch.zeros(n_elems, n_loc_km1, n_loc_k, device=device, dtype=dtype)

    if k == 1:
        # omega_k_ref: (Q, n_loc_k, n)
        # Pull back to physical: omega_1_phys[T,q,l,a] = Σ_i omega_ref[q,l,i] * (J^{-T})[T,a,i]
        omega_1_phys = torch.einsum("qli,Tai->Tqla", omega_k_ref, J_inv_T)  # (T,Q,n_loc_k,m)

        # Contract: contracted[T,q,l] = omega_1_phys · x_quad
        contracted = _contract_1forms(omega_1_phys, x_quad)  # (T,Q,n_loc_k)

        # omega_km1_ref is k-1=0 forms: (Q, n_loc_km1)
        omega_0 = omega_km1_ref  # (Q, n_loc_km1)

        integrand = contracted[:, :, None, :] * omega_0[None, :, :, None]  # (T,Q,n_loc_km1,n_loc_k)
        integral = (
            weights_ref[None, :, None, None] * integrand
        ).sum(dim=1)  # (T,n_loc_km1,n_loc_k)
        b_local = vol_scale[:, None, None] * integral

    elif k == 2:
        # omega_k_ref: (Q, n_loc_k, n, n)
        # Pull back 2-forms:
        # omega_2_phys[T,q,l,a,b] = Σ_ij (J^{-T})[T,a,i](J^{-T})[T,b,j] omega_ref[q,l,i,j]
        omega_2_phys = torch.einsum(
            "Tai,Tbj,qlij->Tqlab", J_inv_T, J_inv_T, omega_k_ref
        )  # (T,Q,n_loc_k,m,m)

        # Contract first slot: (i_X ω)_b = Σ_a ω_{ab} X^a
        contracted_1 = _contract_2forms_first_slot(omega_2_phys, x_quad)  # (T,Q,n_loc_k,m)

        # omega_km1_ref is 1-forms: (Q, n_loc_km1, n)
        omega_1_phys = torch.einsum("qpi,Tai->Tqpa", omega_km1_ref, J_inv_T)  # (T,Q,n_loc_km1,m)

        # Inner product of 1-forms (Euclidean): ⟨α,β⟩ = Σ_a α_a β_a
        integrand = torch.einsum(
            "Tqlm,Tqpm->Tqpl", contracted_1, omega_1_phys
        )  # (T,Q,n_loc_km1,n_loc_k)
        integral = (
            weights_ref[None, :, None, None] * integrand
        ).sum(dim=1)  # (T,n_loc_km1,n_loc_k)
        b_local = vol_scale[:, None, None] * integral

    elif k == 3 and n == 3:
        if m != 3:
            raise NotImplementedError(
                "k=3 contraction here assumes ambient dimension m=3 "
            "and 2-forms as 3x3 antisymmetric matrices."
            )

        # eval_whitney_kform_all for k=3 returns (1,) (single local DOF, constant on reference)
        # Interpret this as coefficient in reference coordinates.
        coeff_ref = omega_k_ref.reshape(-1)[0].to(dtype=dtype)  # scalar

        # Physical coefficient via det(J^{-T}) (covariant 3-form transform for m=n=3)
        # This choice must match your mass convention for 3-forms.
        det_JinvT = torch.linalg.det(J_inv_T)  # (T,)
        coeff_phys = coeff_ref * det_JinvT  # (T,)

        contracted_2 = _contract_3form_volume(coeff_phys, x_quad)  # (T,Q,1,3,3)

        # omega_km1_ref for k-1=2: (Q, n_loc_km1, 3, 3) in reference coords
        omega_2_phys = torch.einsum(
            "Tai,Tbj,qlij->Tqlab", J_inv_T, J_inv_T, omega_km1_ref
        )  # (T,Q,n_loc_km1,3,3)

        # Frobenius inner product of antisym matrices
        integrand = torch.einsum(
            "Tqxab,Tqfab->Tqfx", contracted_2, omega_2_phys
        )  # (T,Q,n_loc_km1,1)
        integral = (weights_ref[None, :, None, None] * integrand).sum(dim=1)  # (T,n_loc_km1,1)
        b_local = vol_scale[:, None, None] * integral

    else:
        raise ValueError(f"Local B_k assembly not implemented for k={k}, n={n}")

    return b_local


def assemble_global_B_k(
    chain_complex: SimplicialChainComplex,
    geometry: SimplicialGeometry,
    X_vertex: Tensor,
    k: int,
    quad_degree: int = 2,
) -> SparseTensor:
    """Assemble global coupling matrix ``B_k(X)`` for degree ``k``.

    Workflow
    --------
    1. Assemble local tensors ``b_local[T, a, b]`` on each top element.
    2. Map local ``(k-1)`` and ``k`` DOFs to global DOF IDs using the vertex
       patterns of the corresponding faces, with a batched lookup.
    3. Compute orientation signs for both sides and multiply them into
       ``b_local`` before scattering (sign product rule).
    4. Broadcast rows/cols and flatten to COO vectors, filter near-zero values,
       and build a ``torch_sparse.SparseTensor`` which is then coalesced.

    Shapes
    ------
    - Output has shape ``(N_{k-1}, N_k)``.

    Args:
        chain_complex: Mesh topology for global DOF lists per degree.
        geometry: Mesh geometry for elementwise transforms and top cells.
        X_vertex: Tensor with shape ``(n_vertices, m)`` of vertex samples of ``X``.
        k: Form degree (``k >= 1``).
        quad_degree: Reference quadrature degree used in local assembly.

    Returns:
        Sparse ``torch_sparse.SparseTensor`` of shape ``(N_{k-1}, N_k)``.
    """
    n = geometry.n
    n_cells_k = chain_complex.n_cells(k)
    n_cells_km1 = chain_complex.n_cells(k - 1)
    device = geometry.vertex_positions.device

    if k < 1 or k > n:
        raise ValueError(f"k must be in [1, {n}], got {k}")

    b_local = assemble_local_B_k(geometry, X_vertex, k, quad_degree)
    n_elems, n_loc_km1, n_loc_k = b_local.shape

    # (N_k, k+1) & GLOBAL ORIENTATION
    k_cells_global = chain_complex.cells(k).to(torch.long)
    # (N_{k-1}, k) & GLOBAL ORIENTATION
    km1_cells_global = chain_complex.cells(k - 1).to(torch.long)
    # (T, n+1) & TOP-SIMPLEX ORIENTATION
    top_cells = geometry.top_cells.to(torch.long)

    local_dofs_k = enumerate_whitney_dofs(n, k)
    local_dofs_km1 = enumerate_whitney_dofs(n, k - 1)

    # For lookup by vertex set, use sorted rows
    k_cells_sorted, k_perm = row_module.build_sorted_row_index(k_cells_global)
    km1_cells_sorted, km1_perm = row_module.build_sorted_row_index(km1_cells_global)

    dofs_global_k = torch.empty(n_elems, n_loc_k, dtype=torch.long, device=device)
    signs_k = torch.empty(n_elems, n_loc_k, dtype=torch.float32, device=device)

    dofs_global_km1 = torch.empty(n_elems, n_loc_km1, dtype=torch.long, device=device)
    signs_km1 = torch.empty(n_elems, n_loc_km1, dtype=torch.float32, device=device)

    # Map local k-faces of each top simplex to global k-cells (+ orientation sign)
    for i_local, face_local in enumerate(local_dofs_k):
        face_idx = torch.tensor(face_local, dtype=torch.long, device=device)
        local_face_verts = top_cells[:, face_idx]  # (T, k+1) in LOCAL induced orientation

        sorted_face_verts, _ = local_face_verts.sort(dim=-1)

        global_ids = row_module.lookup_row_indices(
            sorted_face_verts,
            k_cells_global,
            k_cells_sorted,
            k_perm,
        )
        global_verts = k_cells_global[global_ids]  # (T, k+1) in GLOBAL orientation

        dofs_global_k[:, i_local] = global_ids
        signs_k[:, i_local] = _parity_sign_local_to_global(local_face_verts, global_verts)

    # Map local (k-1)-faces similarly
    for i_local, face_local in enumerate(local_dofs_km1):
        face_idx = torch.tensor(face_local, dtype=torch.long, device=device)
        local_face_verts = top_cells[:, face_idx]  # (T, k) local induced

        sorted_face_verts, _ = local_face_verts.sort(dim=-1)

        global_ids = row_module.lookup_row_indices(
            sorted_face_verts,
            km1_cells_global,
            km1_cells_sorted,
            km1_perm,
        )
        global_verts = km1_cells_global[global_ids]  # (T, k) global orientation

        dofs_global_km1[:, i_local] = global_ids
        signs_km1[:, i_local] = _parity_sign_local_to_global(local_face_verts, global_verts)

    # Apply signs: B_local_signed[T,a,b] = sign_{k-1}(T,a) * sign_k(T,b) * B_local[T,a,b]
    b_local_signed = b_local * (signs_km1[:, :, None] * signs_k[:, None, :]).to(b_local.dtype)

    # Build COO indices by broadcasting local dof ids
    rows_3d = dofs_global_km1[:, :, None].expand(n_elems, n_loc_km1, n_loc_k)
    cols_3d = dofs_global_k[:, None, :].expand(n_elems, n_loc_km1, n_loc_k)

    row_idx = rows_3d.reshape(-1)
    col_idx = cols_3d.reshape(-1)
    values = b_local_signed.reshape(-1)

    mask = values.abs() > 1e-14
    row_idx = row_idx[mask]
    col_idx = col_idx[mask]
    values = values[mask]

    return SparseTensor(
        row=row_idx,
        col=col_idx,
        value=values,
        sparse_sizes=(n_cells_km1, n_cells_k),
    ).coalesce()


class GalerkinInteriorProduct:
    """Galerkin FEEC interior product operator ``i_X: C^k -> C^{k-1}``.

    This operator exposes two responsibilities:
    - Assemble and cache coupling matrices ``B_k(X)`` for ``k = 1..n`` using
      quadrature of Whitney reconstructions and geometric pullbacks.
    - Delegate the projection step ``M_{k-1}^{-1} (B_k x)`` to a provided
      inner product solver (e.g., Conjugate Gradient on the Whitney mass).
    """

    def __init__(
        self,
        chain_complex: SimplicialChainComplex,
        B_matrices: dict[int, SparseTensor],
        inner_product: FEECInnerProduct,
    ):
        """Initialize with pre-assembled coupling matrices.

        Args:
            chain_complex: Mesh topology; provides global DOF counts per degree.
            B_matrices: Dictionary mapping degree ``k`` to ``B_k(X)`` (as a
                ``torch_sparse.SparseTensor`` with shape ``(N_{k-1}, N_k)``).
            inner_product: Inner product helper exposing the mass matrices and
                a ``solve`` routine to apply ``M_{k-1}^{-1}``.
        """
        self.chain_complex = chain_complex
        self.B_matrices = B_matrices
        self.inner_product = inner_product
        self.n = chain_complex.dim

    @classmethod
    def from_vector_field(
        cls,
        chain_complex: SimplicialChainComplex,
        geometry: SimplicialGeometry,
        X0: CoChain,
        inner_product: FEECInnerProduct,
        quad_degree: int = 2,
    ) -> GalerkinInteriorProduct:
        """Assemble interior product from a vector field ``X``.

        The vector field ``X`` is specified as a 0-cochain over vertices.
        This constructor assembles and caches all coupling matrices
        ``B_k(X)`` for ``k=1..n`` and returns a ready-to-use operator.

        Args:
            chain_complex: Mesh topology.
            geometry: Mesh geometry.
            X0: Vector field as 0-cochain whose data has shape ``(n_vertices, m)``.
            inner_product: Inner product helper for mass matrices and solves.
            quad_degree: Quadrature polynomial degree (default 2).

        Returns:
            A ``GalerkinInteriorProduct`` with ``B_k(X)`` assembled.
        """
        if X0.k != 0:
            raise ValueError(f"Vector field must be a 0-cochain, got k={X0.k}")

        n = chain_complex.dim
        X_vertex = X0.data

        B_matrices: dict[int, SparseTensor] = {}
        for k in range(1, n + 1):
            B_matrices[k] = assemble_global_B_k(
                chain_complex=chain_complex,
                geometry=geometry,
                X_vertex=X_vertex,
                k=k,
                quad_degree=quad_degree,
            )

        return cls(chain_complex, B_matrices, inner_product)

    def coupling(self, k: int) -> SparseTensor:
        """Return the coupling matrix ``B_k(X)``.

        Args:
            k: Form degree.

        Returns:
            Sparse ``torch_sparse.SparseTensor`` with shape ``(N_{k-1}, N_k)``.
        """
        if k not in self.B_matrices:
            raise ValueError(f"No coupling matrix for k={k}")
        return self.B_matrices[k]

    def apply(self, cochain_k: CoChain) -> CoChain:
        """Apply interior product ``i_X`` to a ``k``-cochain.

        Computes ``y = M_{k-1}^{-1} (B_k x)`` where ``x`` is the data of the
        input cochain. Uses ``SparseTensor.matmul`` for ``B_k x`` and delegates
        the solve to the provided inner product.

        Args:
            cochain_k: Input ``k``-cochain.

        Returns:
            Output ``(k-1)``-cochain with the projected result.
        """

        k = cochain_k.k
        if k < 1 or k > self.n:
            raise ValueError(f"k must be in [1, {self.n}], got {k}")

        B_k = self.coupling(k)
        rhs = B_k.matmul(cochain_k.data)  # (N_{k-1}, d)

        # Solve M_{k-1} y = rhs using the provided inner product object.
        y = self.inner_product.solve(rhs, k=k - 1)

        return CoChain(k=k - 1, data=y)

    def to(self, *args, **kwargs) -> GalerkinInteriorProduct:
        """Move all internal sparse matrices to the specified device/dtype.

        Args:
            *args: Positional arguments passed to ``torch.Tensor.to``.
            **kwargs: Keyword arguments passed to ``torch.Tensor.to``.

        Returns:
            Self for method chaining.
        """
        self.B_matrices = {k: B.to(*args, **kwargs) for k, B in self.B_matrices.items()}
        return self

    def cpu(self) -> GalerkinInteriorProduct:
        """Move all tensors to CPU."""
        return self.to("cpu")

    def cuda(self) -> GalerkinInteriorProduct:
        """Move all tensors to CUDA."""
        return self.to("cuda")
