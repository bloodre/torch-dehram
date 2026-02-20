"""Generic linear solvers for discrete differential operators.

This module provides:
- Preconditioner: abstract base class for preconditioners.
- DiagonalPreconditioner: Jacobi (diagonal) preconditioner.
- OperatorPreconditioner: wraps an arbitrary linear operator.
- cg_solve: Preconditioned Conjugate Gradient solver.
"""

from .cg import (
    DiagonalPreconditioner,
    OperatorPreconditioner,
    Preconditioner,
    cg_solve,
)

__all__ = [
    "DiagonalPreconditioner",
    "OperatorPreconditioner",
    "Preconditioner",
    "cg_solve",
]
