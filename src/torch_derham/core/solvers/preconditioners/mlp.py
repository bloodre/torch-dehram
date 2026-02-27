"""MLP-based neural preconditioner implementations."""

from __future__ import annotations

import torch.nn as nn
from torch import Tensor

from ..cg import Preconditioner


class MLPNeuralPreconditioner(Preconditioner, nn.Module):
    """MLP-based neural preconditioner for fixed geometry sparse matrices.

    Implements the approach from Krishnapriyan et al. (2021):
    "Neural Preconditioners for Iterative Linear Solvers"

    Learns mapping: r → M^{-1} @ r using a configurable MLP.

    Args:
        matrix_dim: Dimension of the matrix (and vectors)
        hidden_dims: List of hidden layer dimensions
        activation: Activation function ('relu', 'gelu', 'tanh')
        dropout: Dropout rate
        layer_norm: Whether to use layer normalization
        use_residual: Whether to use residual connection
    """

    def __init__(self,
                 matrix_dim: int,
                 hidden_dims: list[int] = [256, 256, 256],
                 activation: str = 'relu',
                 dropout: float = 0.0,
                 layer_norm: bool = True,
                 use_residual: bool = False):
        super().__init__()
        self.matrix_dim = matrix_dim
        self.use_residual = use_residual

        # Build MLP dynamically
        layers = []
        input_dim = matrix_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))

            # Activation
            if activation == 'relu':
                layers.append(nn.ReLU())
            elif activation == 'gelu':
                layers.append(nn.GELU())
            elif activation == 'tanh':
                layers.append(nn.Tanh())

            # Regularization
            if dropout > 0:
                layers.append(nn.Dropout(dropout))

            # Normalization
            if layer_norm:
                layers.append(nn.LayerNorm(hidden_dim))

            input_dim = hidden_dim

        # Final layer
        layers.append(nn.Linear(input_dim, matrix_dim))

        self.network = nn.Sequential(*layers)

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        """Initialize network weights for better training."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # Xavier initialization
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def apply(self, r: Tensor) -> Tensor:
        """Apply neural preconditioning to residual."""
        out = self.network(r)

        if self.use_residual:
            # Residual connection: identity + learned correction
            return r + out

        return out
