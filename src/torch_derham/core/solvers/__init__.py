"""Generic linear solvers for discrete differential operators.

This module provides:
- Preconditioner: abstract base class for preconditioners.
- JacobiPreconditioner: Jacobi (diagonal) preconditioner.
- OperatorPreconditioner: wraps an arbitrary linear operator.
- cg_solve: Preconditioned Conjugate Gradient solver.
"""

from .cg import (
    JacobiPreconditioner,
    OperatorPreconditioner,
    Preconditioner,
    cg_solve,
)

__all__ = [
    "JacobiPreconditioner",
    "OperatorPreconditioner",
    "Preconditioner",
    "cg_solve",
]
