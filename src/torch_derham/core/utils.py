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
