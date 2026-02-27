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
        matrix_dim (int): Dimension of the matrix (and vectors)
        hidden_dims (list[int]): List of hidden layer dimensions
        activation (str | nn.Module): Activation function ('relu', 'gelu', 'tanh')
            or nn.Module instance
        dropout (float): Dropout rate
        layer_norm (bool): Whether to use layer normalization
        use_residual (bool): Whether to use residual connection
    """

    def __init__(self,
                 matrix_dim: int,
                 hidden_dims: list[int] = [256, 256, 256],
                 activation: str | nn.Module = 'relu',
                 dropout: float = 0.0,
                 layer_norm: bool = True,
                 use_residual: bool = False):
        super().__init__()
        self.matrix_dim = matrix_dim
        self.use_residual = use_residual

        # Resolve activation
        def get_activation() -> nn.Module:
            if isinstance(activation, nn.Module):
                return activation
            if activation == 'relu':
                return nn.ReLU()
            elif activation == 'gelu':
                return nn.GELU()
            elif activation == 'tanh':
                return nn.Tanh()
            else:
                raise ValueError(f"Unknown activation: {activation}")

        # Build MLP dynamically
        layers = []
        input_dim = matrix_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(get_activation())

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
