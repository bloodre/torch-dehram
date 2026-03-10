"""Utility functions for sparse matrix operations."""

import torch
from torch import Tensor
from torch_sparse import SparseTensor


def extract_sparse_diagonal(M: SparseTensor) -> Tensor:
    """Extract diagonal from torch_sparse.SparseTensor.

    Args:
        M: SparseTensor with shape (n, n)

    Returns:
        Dense tensor with shape (n,) containing diagonal entries
    """
    # Get sparse storage components
    row_idx = M.storage.row()
    col_idx = M.storage.col()
    values = M.storage.value()

    # Find diagonal entries (where row == col)
    diagonal_mask = row_idx == col_idx
    diagonal_indices = row_idx[diagonal_mask]
    diagonal_values = values[diagonal_mask]

    # Create full diagonal tensor (fill missing entries with zeros)
    n = M.size(0)
    diagonal = torch.zeros(n, device=M.device, dtype=M.dtype())
    diagonal[diagonal_indices] = diagonal_values

    return diagonal


@torch.no_grad()
def is_diagonal_matrix(M: SparseTensor, tol: float = 1e-12) -> bool:
    """Check if a sparse matrix is effectively diagonal.

    Determines if all off-diagonal entries are below the given tolerance.
    Useful for detecting when a mass matrix can be solved with O(n)
    operations instead of iterative methods.

    Args:
        M: SparseTensor to check.
        tol: Tolerance for considering off-diagonal entries as zero.

    Returns:
        True if matrix is diagonal within tolerance, False otherwise.
    """
    if M.size(0) != M.size(1):
        return False

    # Get row and column indices
    row_idx = M.storage.row()
    col_idx = M.storage.col()
    values = M.storage.value()

    # Check for off-diagonal entries
    off_diagonal_mask = row_idx != col_idx
    if off_diagonal_mask.any():
        off_diagonal_values = values[off_diagonal_mask]
        if torch.max(torch.abs(off_diagonal_values)) > tol:
            return False

    return True
