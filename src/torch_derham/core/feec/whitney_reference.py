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
    vertices: Tensor,
) -> Tensor:
    """Evaluate Whitney 0-forms at barycentric points.

    ω_i^0(λ) = λ_i (scalar per point).

    Args:
        barycentric (Tensor): (N, n+1) barycentric coordinates.
        vertices (Tensor): (M,) or (M, 1) vertex indices.

    Returns:
        (N, M) scalars.
    """
    if vertices.dim() == 1:
        vertices = vertices.unsqueeze(-1)
    vertices = vertices.squeeze(-1)
    return barycentric[:, vertices]


def eval_whitney_1form(
    barycentric: Tensor,
    grad_lambda: Tensor,
    edges: Tensor,
) -> Tensor:
    """Evaluate Whitney 1-forms at barycentric points.

    ω_{ij}^1 = λ_i dλ_j - λ_j dλ_i, represented as a covector (1, n) in
    reference coordinates.

    Args:
        barycentric (Tensor): (N, n+1) barycentric coordinates.
        grad_lambda (Tensor): (n+1, n) gradients ∇λ_k.
        edges (Tensor): (M, 2) vertex index pairs (i, j).

    Returns:
        (N, M, n) covectors (1-forms as row vectors in ℝⁿ).
    """
    i_idx = edges[:, 0]
    j_idx = edges[:, 1]

    lambda_i = barycentric[:, i_idx]
    lambda_j = barycentric[:, j_idx]

    grad_i = grad_lambda[i_idx, :]
    grad_j = grad_lambda[j_idx, :]

    result = (
        lambda_i.unsqueeze(-1) * grad_j.unsqueeze(0)
        - lambda_j.unsqueeze(-1) * grad_i.unsqueeze(0)
    )

    return result


def eval_whitney_2form(
    barycentric: Tensor,
    grad_lambda: Tensor,
    faces: Tensor,
) -> Tensor:
    """Evaluate Whitney 2-forms at barycentric points.

    ω_{ijk}^2 = 2(λ_i dλ_j ∧ dλ_k + λ_j dλ_k ∧ dλ_i + λ_k dλ_i ∧ dλ_j),

    represented as an antisymmetric matrix (wedge product) in ℝⁿ coordinates.
    For n=2, this is a scalar (volume form coefficient).
    For n=3, this is an antisymmetric 3×3 matrix → 3-vector via Hodge dual.

    Args:
        barycentric (Tensor): (N, n+1) barycentric coordinates.
        grad_lambda (Tensor): (n+1, n) gradients ∇λ_k.
        faces (Tensor): (M, 3) vertex index triples (i, j, k).

    Returns:
        (N, M, n, n) antisymmetric matrices representing 2-forms.
    """
    N = barycentric.shape[0]
    M = faces.shape[0]
    n = grad_lambda.shape[1]

    i_idx = faces[:, 0]
    j_idx = faces[:, 1]
    k_idx = faces[:, 2]

    lambda_i = barycentric[:, i_idx]
    lambda_j = barycentric[:, j_idx]
    lambda_k = barycentric[:, k_idx]

    grad_i = grad_lambda[i_idx, :]
    grad_j = grad_lambda[j_idx, :]
    grad_k = grad_lambda[k_idx, :]

    result = torch.zeros(N, M, n, n, device=barycentric.device, dtype=barycentric.dtype)

    cyclic_lambda = [lambda_i, lambda_j, lambda_k]
    cyclic_grads = [grad_i, grad_j, grad_k]
    cyclic_indices = [(0, 1, 2), (1, 2, 0), (2, 0, 1)]

    for p_idx, q_idx, r_idx in cyclic_indices:
        lam_p = cyclic_lambda[p_idx]
        g_q = cyclic_grads[q_idx]
        g_r = cyclic_grads[r_idx]

        wedge = (
            g_q.unsqueeze(1).unsqueeze(-1) * g_r.unsqueeze(1).unsqueeze(-2)
            - g_r.unsqueeze(1).unsqueeze(-1) * g_q.unsqueeze(1).unsqueeze(-2)
        )

        result += lam_p.unsqueeze(-1).unsqueeze(-1).unsqueeze(1) * wedge

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
    grad_matrix = grad_lambda[1:, :]
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
    dofs_tensor = torch.tensor(dofs, dtype=torch.long, device=device)

    if k == 0:
        return eval_whitney_0form(barycentric, dofs_tensor)
    elif k == 1:
        return eval_whitney_1form(barycentric, grad_lambda, dofs_tensor)
    elif k == 2:
        return eval_whitney_2form(barycentric, grad_lambda, dofs_tensor)
    elif k == 3 and n == 3:
        vol_coeff = eval_whitney_3form(grad_lambda)
        return vol_coeff.unsqueeze(0)
    else:
        raise ValueError(f"eval_whitney_kform_all not implemented for k={k}, n={n}")
