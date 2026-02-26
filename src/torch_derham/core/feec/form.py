"""Whitney form evaluator and quadrature utilities.

Provides a unified class for evaluating Whitney forms on reference simplices
and applying quadrature rules. This module separates reference-space evaluation
from geometry pullback, keeping the evaluator pure and reusable.
"""

from __future__ import annotations

import torch
from torch import Tensor

from .whitney_reference import (
    barycentric_gradients_reference,
    enumerate_whitney_dofs,
    eval_whitney_0form,
    eval_whitney_1form,
    eval_whitney_2form,
    eval_whitney_3form,
)


class WhitneyFormEvaluator:
    """Unified evaluator for Whitney forms on reference simplices.

    Provides batched evaluation of Whitney k-forms at barycentric points
    and quadrature integration utilities. Geometry pullback is handled
    by callers (e.g., mass assembly), not this class.
    """

    def __init__(self, n: int, device: torch.device | None = None):
        """Initialize evaluator for n-simplices.

        Args:
            n (int): simplex dimension.
            device (torch.device | None): device for tensors.
        """
        self.n = n
        self.device = device or torch.device("cpu")
        self._grad_lambda = barycentric_gradients_reference(n, self.device)

    def evaluate(
        self,
        k: int,
        barycentric: Tensor,
        dofs: Tensor,
    ) -> Tensor:
        """Evaluate Whitney k-forms at barycentric points for specific DOFs.

        Args:
            k (int): form degree.
            barycentric (Tensor): (N, n+1) barycentric coordinates.
            dofs (Tensor): (M, k+1) vertex indices for DOFs to evaluate.

        Returns:
            Tensor of form values:
                k=0: (N, M) scalars
                k=1: (N, M, n) covectors
                k=2: (N, M, n, n) antisymmetric matrices
                k=3: (M,) scalars (if n=3)
        """
        if k > self.n:
            raise ValueError(f"k={k} > n={self.n}")

        if k == 0:
            return eval_whitney_0form(barycentric, dofs)
        elif k == 1:
            return eval_whitney_1form(barycentric, self._grad_lambda, dofs)
        elif k == 2:
            return eval_whitney_2form(barycentric, self._grad_lambda, dofs)
        elif k == 3 and self.n == 3:
            vol_coeff = eval_whitney_3form(self._grad_lambda)
            return vol_coeff.expand(dofs.shape[0])
        else:
            raise ValueError(f"Evaluation not implemented for k={k}, n={self.n}")

    def evaluate_all(
        self,
        k: int,
        barycentric: Tensor,
    ) -> Tensor:
        """Evaluate all Whitney k-forms at barycentric points.

        Args:
            k (int): form degree.
            barycentric (Tensor): (N, n+1) barycentric coordinates.

        Returns:
            Tensor of form values for all local DOFs.
        """
        dofs = enumerate_whitney_dofs(self.n, k)
        dofs_tensor = torch.tensor(dofs, dtype=torch.long, device=self.device)
        return self.evaluate(k, barycentric, dofs_tensor)

    def apply_quadrature(
        self,
        form_values: Tensor,
        weights: Tensor,
    ) -> Tensor:
        """Apply quadrature weights to form values.

        Args:
            form_values (Tensor): values at quadrature points.
            weights (Tensor): quadrature weights.

        Returns:
            Integrated quantities (scalar or per-DOF values).
        """
        if form_values.dim() == 1:
            return (weights * form_values).sum()
        elif form_values.dim() == 2:
            return (weights.unsqueeze(-1) * form_values).sum(dim=0)
        elif form_values.dim() == 3:
            return (weights.unsqueeze(-1).unsqueeze(-1) * form_values).sum(dim=0)
        elif form_values.dim() == 4:
            return (
                weights.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) * form_values
            ).sum(dim=0)
        else:
            raise ValueError(f"Unsupported form_values shape: {form_values.shape}")
