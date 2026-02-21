"""Whitney basis forms on the reference simplex.

Lowest-order Whitney k-forms on the reference n-simplex Δⁿ in barycentric
coordinates. These basis forms are canonical in FEEC and provide one DOF per
k-dimensional sub-simplex (face).

For the reference simplex Δⁿ with vertices at standard basis vectors:
    v_0 = origin, v_i = e_i for i=1..n

Barycentric coordinates λ = (λ_0, ..., λ_n) satisfy:
    - Σ λ_i = 1
    - λ_i(v_j) = δ_ij
    - x = Σ λ_i(x) v_i

Whitney k-forms (lowest order):
    - k=0: ω_i^0 = λ_i  (nodal, one per vertex)
    - k=1: ω_{ij}^1 = λ_i dλ_j - λ_j dλ_i  (one per edge)
    - k=2: ω_{ijk}^2 = λ_i dλ_j ∧ dλ_k + cyclic permutations (one per face)
    - k=n: ω^n = n! dλ_1 ∧ ... ∧ dλ_n (one per top simplex)

These are expressed as differential forms (alternating tensors) at each point.
For computation in embedded ℝ^m, they are pulled back via the affine map
F_T : Δⁿ → T, which requires the inverse Jacobian.
"""
# pylint: disable=invalid-name

from __future__ import annotations

from itertools import combinations

import torch
from torch import Tensor


def barycentric_gradients_reference(n: int, device: torch.device) -> Tensor:
    """Gradients of barycentric coordinates on the reference simplex.

    For Δⁿ with vertices v_0=origin, v_i=e_i, the gradient of λ_i in
    reference (ξ) coordinates is:
        ∇λ_0 = -ones, ∇λ_i = e_i for i=1..n.

    Args:
        n (int): simplex dimension.
        device (torch.device): device for tensors.

    Returns:
        (n+1, n) tensor where row i is ∇λ_i in ℝⁿ.
    """
    grad_lambda = torch.zeros(n + 1, n, device=device)
    grad_lambda[0, :] = -1.0
    for i in range(1, n + 1):
        grad_lambda[i, i - 1] = 1.0
    return grad_lambda


def eval_whitney_0form(
    barycentric: Tensor,
    vertex_index: int,
) -> Tensor:
    """Evaluate Whitney 0-form at barycentric points.

    ω_i^0(λ) = λ_i (scalar per point).

    Args:
        barycentric (Tensor): (N, n+1) barycentric coordinates.
        vertex_index (int): index i ∈ {0, ..., n}.

    Returns:
        (N,) scalars.
    """
    return barycentric[:, vertex_index]


def eval_whitney_1form(
    barycentric: Tensor,
    grad_lambda: Tensor,
    edge: tuple[int, int],
) -> Tensor:
    """Evaluate Whitney 1-form at barycentric points.

    ω_{ij}^1 = λ_i dλ_j - λ_j dλ_i, represented as a covector (1, n) in
    reference coordinates.

    Args:
        barycentric (Tensor): (N, n+1) barycentric coordinates.
        grad_lambda (Tensor): (n+1, n) gradients ∇λ_k.
        edge (tuple[int, int]): ordered vertex indices (i, j).

    Returns:
        (N, n) covectors (1-forms as row vectors in ℝⁿ).
    """
    i, j = edge
    lambda_i = barycentric[:, i:i+1]   # (N, 1)
    lambda_j = barycentric[:, j:j+1]   # (N, 1)

    grad_i = grad_lambda[i:i+1, :]     # (1, n)
    grad_j = grad_lambda[j:j+1, :]     # (1, n)

    return lambda_i * grad_j - lambda_j * grad_i  # (N, n)


def eval_whitney_2form(
    barycentric: Tensor,
    grad_lambda: Tensor,
    face: tuple[int, int, int],
) -> Tensor:
    """Evaluate Whitney 2-form at barycentric points.

    ω_{ijk}^2 = 2(λ_i dλ_j ∧ dλ_k + λ_j dλ_k ∧ dλ_i + λ_k dλ_i ∧ dλ_j),

    represented as an antisymmetric matrix (wedge product) in ℝⁿ coordinates.
    For n=2, this is a scalar (volume form coefficient).
    For n=3, this is an antisymmetric 3×3 matrix → 3-vector via Hodge dual.

    Args:
        barycentric (Tensor): (N, n+1) barycentric coordinates.
        grad_lambda (Tensor): (n+1, n) gradients ∇λ_k.
        face (tuple[int, int, int]): ordered vertex indices (i, j, k).

    Returns:
        (N, n, n) antisymmetric matrices representing 2-forms.
    """
    i, j, k = face
    n = grad_lambda.shape[1]
    N = barycentric.shape[0]

    lambda_vals = [
        barycentric[:, i],
        barycentric[:, j],
        barycentric[:, k],
    ]
    grads = [
        grad_lambda[i, :],
        grad_lambda[j, :],
        grad_lambda[k, :],
    ]

    result = torch.zeros(N, n, n, device=barycentric.device, dtype=barycentric.dtype)

    # ω = 2 Σ_{cyclic} λ_p dλ_q ∧ dλ_r
    # dλ_q ∧ dλ_r = antisymmetric outer product
    cyclic = [(0, 1, 2), (1, 2, 0), (2, 0, 1)]
    for p_idx, q_idx, r_idx in cyclic:
        lam_p = lambda_vals[p_idx].unsqueeze(-1).unsqueeze(-1)  # (N, 1, 1)
        g_q = grads[q_idx].unsqueeze(0)  # (1, n)
        g_r = grads[r_idx].unsqueeze(0)  # (1, n)

        # dλ_q ∧ dλ_r as antisymmetric matrix: outer(g_q, g_r) - outer(g_r, g_q)
        wedge = g_q.unsqueeze(-1) * g_r.unsqueeze(-2) - g_r.unsqueeze(-1) * g_q.unsqueeze(-2)
        # wedge: (1, n, n)

        result += lam_p * wedge

    return 2.0 * result


def eval_whitney_3form(
    grad_lambda: Tensor,
) -> Tensor:
    """Evaluate Whitney 3-form (volume form) on reference tetrahedron.

    ω^3 = 6 dλ_1 ∧ dλ_2 ∧ dλ_3, represented as det([∇λ_1, ∇λ_2, ∇λ_3]).

    For Δ³, the space is 1-dimensional; the basis 3-form is constant.

    Args:
        grad_lambda (Tensor): (4, 3) gradients ∇λ_k.

    Returns:
        Scalar value of the 3-form coefficient (constant over the simplex).
    """
    # dλ_1 ∧ dλ_2 ∧ dλ_3 = det of the 3×3 matrix [∇λ_1, ∇λ_2, ∇λ_3]
    grad_matrix = grad_lambda[1:, :]  # (3, 3)
    det_val = torch.linalg.det(grad_matrix)
    return 6.0 * det_val


def enumerate_whitney_dofs(n: int, k: int) -> list[tuple[int, ...]]:
    """Enumerate all k-faces of the reference n-simplex.

    Each k-face is a (k+1)-tuple of vertex indices in sorted order.

    Args:
        n (int): simplex dimension.
        k (int): form degree.

    Returns:
        List of (k+1)-tuples of vertex indices.
    """
    vertices = list(range(n + 1))
    return [tuple(sorted(face)) for face in combinations(vertices, k + 1)]


def eval_whitney_kform_all(
    barycentric: Tensor,
    n: int,
    k: int,
) -> Tensor:
    """Evaluate all Whitney k-forms at given barycentric points.

    Returns a tensor suitable for mass matrix integration.

    Args:
        barycentric (Tensor): (N, n+1) barycentric coordinates.
        n (int): simplex dimension.
        k (int): form degree.

    Returns:
        For k=0: (N, n_dof) scalars.
        For k=1: (N, n_dof, n) covectors.
        For k=2: (N, n_dof, n, n) antisymmetric matrices.
        For k=3: (n_dof,) constant (if n=3).

    Raises:
        ValueError: if k > n or k not in {0, 1, 2, 3}.
    """
    if k > n:
        raise ValueError(f"k={k} > n={n} is invalid")

    device = barycentric.device
    grad_lambda = barycentric_gradients_reference(n, device)
    dofs = enumerate_whitney_dofs(n, k)
    n_dof = len(dofs)
    N = barycentric.shape[0]

    if k == 0:
        result = torch.empty(N, n_dof, device=device, dtype=barycentric.dtype)
        for idx, (vertex,) in enumerate(dofs):
            result[:, idx] = eval_whitney_0form(barycentric, vertex)
        return result

    elif k == 1:
        result = torch.empty(N, n_dof, n, device=device, dtype=barycentric.dtype)
        for idx, edge in enumerate(dofs):
            result[:, idx, :] = eval_whitney_1form(barycentric, grad_lambda, edge)
        return result

    elif k == 2:
        result = torch.empty(N, n_dof, n, n, device=device, dtype=barycentric.dtype)
        for idx, face in enumerate(dofs):
            result[:, idx, :, :] = eval_whitney_2form(barycentric, grad_lambda, face)
        return result

    elif k == 3 and n == 3:
        # Only one DOF for k=3, constant over simplex
        vol_coeff = eval_whitney_3form(grad_lambda)
        return vol_coeff.unsqueeze(0)  # (1,)

    else:
        raise ValueError(f"eval_whitney_kform_all not implemented for k={k}, n={n}")
