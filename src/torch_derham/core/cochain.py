"""
Cochain: efficient per-degree feature storage on a cell complex.

- Supports different channel counts per degree k (d_k can vary).
- Stores ONE tensor per k: x[k] has shape (N_k, d_k).
- O(1) degree views; no padding, no global concatenation required.

This is "features as (vector-valued) cochains":
  x[k] ∈ C^k(K; R^{d_k}) ≅ R^{N_k × d_k}
No sheaf compatibility is enforced here.
"""

from __future__ import annotations

from typing import Optional, Sequence

import torch
from torch import Tensor


def _as_int(x) -> int:
    if isinstance(x, int):
        return x
    if isinstance(x, Tensor):
        return int(x.cpu().item())
    return int(x)


class Cochain:
    """Cochain: efficient per-degree feature storage on a cell complex.
    
    Supports different channel counts per degree k (d_k can vary).
    Stores ONE tensor per k: x[k] has shape (N_k, d_k).
    O(1) degree views; no padding, no global concatenation required.
    
    This is "features as (vector-valued) cochains":
      x[k] ∈ C^k(K; R^{d_k}) ≅ R^{N_k × d_k}
    No sheaf compatibility is enforced here.
    """

    def __init__(
        self,
        x: dict[int, Tensor],
        dim: Optional[int] = None,
        validate: bool = True,
    ):
        """
        Args:
            x: dict k -> Tensor (N_k, d_k) cell features.
            dim: optional maximum k. If None, inferred from keys in x.
            validate: if True, checks rank/dtypes, and that k are nonnegative.
        """
        self.x = x

        if dim is None and not x.keys():
            raise ValueError("Cannot infer dimension from empty data")
        elif dim is None:
            self.dim = max(x.keys())
        else:
            self.dim = dim

        if validate:
            self._validate_basic()

    # -----------------------
    # Basic access
    # -----------------------
    def has_k(self, k: int) -> bool:
        """Check if k-degree features are present."""
        return k in self.x

    def x_k(self, k: int) -> Tensor:
        """Get k-degree features."""
        return self.x.get(k)

    def d_k(self, k: int) -> int:
        """Get dimension of k-degree features."""
        t = self.x.get(k)
        return t.size(1)

    def n_k(self, k: int) -> int:
        """Get number of k-cells."""
        t = self.x.get(k)
        return t.size(0)

    # -----------------------
    # Validation
    # -----------------------
    def _validate_basic(self) -> None:
        for k, t in self.x.items():
            if not isinstance(t, Tensor) or t.ndim != 2:
                raise ValueError(
                    f"x[{k}] must be rank-2 Tensor (N_k, d_k), "
                    f"got {type(t)} {getattr(t,'shape',None)}"
                )
            if k < 0:
                raise ValueError(f"Degree k must be >=0, got {k}")

    def validate_against_complex(self, complex_obj, *, allow_missing: bool = True) -> None:
        """Check x[k].shape[0] == complex.n_cells(k) for all stored k.

        Does not require all degrees to be present unless allow_missing=False.
        """
        # Check present keys
        for k, t in self.x.items():
            n_k = _as_int(complex_obj.n_cells(k))
            if t.size(0) != n_k:
                raise ValueError(f"x[{k}] has N={t.size(0)} but complex has N_k={n_k}")

        if not allow_missing:
            dim = _as_int(complex_obj.dim)
            for k in range(dim + 1):
                if k not in self.x:
                    raise ValueError(f"Missing x[{k}] for k={k}")

    # -----------------------
    # Device / dtype
    # -----------------------
    def to(self, *args, **kwargs) -> "Cochain":
        """Move tensors to device/dtype."""
        self.x = {k: t.to(*args, **kwargs) for k, t in self.x.items()}
        return self

    def cpu(self) -> "Cochain":
        """Move tensors to CPU."""
        return self.to("cpu")

    def cuda(self) -> "Cochain":
        """Move tensors to CUDA."""
        return self.to("cuda")

    def pin_memory(self) -> "Cochain":
        """Move tensors to pinned memory."""
        self.x = {k: t.pin_memory() for k, t in self.x.items()}
        return self

    # -----------------------
    # Convenience constructors
    # -----------------------
    @classmethod
    def zeros_from_complex(
        cls,
        complex_obj,
        d_by_k: dict[int, int],
        dtype: torch.dtype = torch.float32,
        device: Optional[torch.device] = None,
    ) -> "Cochain":
        """Allocate zero features for the specified degrees."""
        x: dict[int, Tensor] = {}
        for k, d in d_by_k.items():
            n_k = _as_int(complex_obj.n_cells(k))
            x[k] = torch.zeros((n_k, d), dtype=dtype, device=device)
        dim = max(d_by_k.keys()) if d_by_k else _as_int(complex_obj.dim)
        return cls(x=x, dim=dim, validate=True)

    @classmethod
    def from_tensors(
        cls,
        x_by_k: Sequence[Optional[Tensor]],
        validate: bool = True,
    ) -> "Cochain":
        """Build from list/tuple where index is k."""
        x: dict[int, Tensor] = {k: t for k, t in enumerate(x_by_k) if t is not None}
        dim = max(x.keys()) if x else -1
        return cls(x=x, dim=dim, validate=validate)
