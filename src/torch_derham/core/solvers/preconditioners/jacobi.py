"""Jacobi preconditioner implementations."""

from __future__ import annotations

from torch import Tensor

from ..cg import Preconditioner


class JacobiPreconditioner(Preconditioner):
    """Jacobi preconditioner based on the diagonal of A, with optional damping.

    Applies ω * M^{-1} r = ω * (r / diag), where ω ∈ (0, 1] is the
    damping parameter. Damping can improve stability and convergence
    for ill-conditioned systems. Effective when the diagonal captures
    most of the conditioning of A.

    Args:
        diag: (n,) or (n, 1) tensor of diagonal entries of A.
        omega: Damping parameter (default: 1.0, no damping).
    """

    def __init__(self, diag: Tensor, omega: float = 1.0):
        if not 0 < omega <= 1.0:
            raise ValueError("Damping parameter omega must be in (0, 1]")

        self._omega = omega

        if diag.ndim == 1:
            # (n,) -> (n, 1)
            self._diag = diag.unsqueeze(-1)
        elif diag.ndim == 2 and diag.size(1) == 1:
            self._diag = diag
        else:
            raise ValueError("Diagonal must be a vector or a 1-column matrix.")

    def apply(self, r: Tensor) -> Tensor:
        """Apply damped diagonal preconditioning.

        Args:
            r: (n,) or (n, d) residual tensor.

        Returns:
            ω * (r / diag), same shape as r.
        """
        # Apply ω * M⁻¹_approx = ω * diag(M)⁻¹
        return self._omega * (r / self._diag)


def suggest_omega(condition_number: int) -> float:
    """Suggest damping parameter omega based on condition number.

    Maps condition number to appropriate damping parameter for Jacobi
    preconditioning. Higher condition numbers require more damping.

    Args:
        condition_number: Estimated condition number (order of magnitude).

    Returns:
        Recommended omega damping parameter in (0, 1].
    """
    if condition_number <= 10:
        return 1.0      # Well-conditioned - no damping needed
    elif condition_number <= 100:
        return 0.9      # Light damping
    elif condition_number <= 1000:
        return 0.8      # Moderate damping
    elif condition_number <= 10000:
        return 0.7      # Strong damping
    else:
        return 0.6      # Heavy damping for very ill-conditioned
