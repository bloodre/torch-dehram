"""Generic linear solvers for discrete differential operators.

This module provides:
- Preconditioner: abstract base class for preconditioners.
- JacobiPreconditioner: Jacobi (diagonal) preconditioner.
- OperatorPreconditioner: wraps an arbitrary linear operator.
- cg_solve: Preconditioned Conjugate Gradient solver.
"""

from .cg import (
    OperatorPreconditioner,
    Preconditioner,
    cg_solve,
)
from .preconditioners import (
    JacobiPreconditioner,
    MLPNeuralPreconditioner,
)

__all__ = [
    "JacobiPreconditioner",
    "MLPNeuralPreconditioner",
    "OperatorPreconditioner",
    "Preconditioner",
    "cg_solve",
]
