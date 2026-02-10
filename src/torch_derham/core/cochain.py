"""Cochain containers and graded cochains.

This module provides two small abstractions around feature tensors
defined on the cells of a chain complex:

  - CoChain: a single k-degree cochain x^k with data in R^{N_k × d_k}.
  - GradedCochain: a graded cochain x = (x^0, ..., x^K) collecting
    one CoChain per degree.

The intent is to offer a lightweight, explicit representation for
"features as (vector-valued) cochains" without enforcing any sheaf
compatibility or PDE-like structure.
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


class CoChain:
    """Single-degree cochain x^k with feature tensor.

    This object pairs a degree label ``k`` with a feature tensor
    ``data`` of shape ``(N_k, d_k)``.  It does **not** know about any
    surrounding chain complex; it is purely a typed container.

    Args:
        k: cochain degree (k >= 0).
        data: feature tensor of shape (N_k, d_k).
    """

    def __init__(
        self,
        k: int,
        data: Tensor,
    ):
        if k < 0:
            raise ValueError(f"Cochain degree k must be >= 0, got {k}")
        if not isinstance(data, Tensor):
            raise TypeError(
                "Cochain data must be a Tensor, "
                f"got {type(data)}",
            )
        if data.ndim != 2:
            raise ValueError(
                "Cochain data must be rank-2 Tensor (N_k, d_k), "
                f"got ndim={data.ndim}",
            )
        if data.numel() == 0:
            raise ValueError("Cochain data must be non-empty")

        self._k = int(k)
        self._data = data

    # -----------------------
    # Basic properties
    # -----------------------
    @property
    def k(self) -> int:
        """Cochain degree k (non-negative integer)."""
        return self._k

    @property
    def data(self) -> Tensor:
        """Underlying feature tensor of shape (N_k, d_k)."""
        return self._data

    @property
    def n_cells(self) -> int:
        """Number of k-cells N_k (first dimension of data)."""
        return self._data.size(0)

    @property
    def channels(self) -> int:
        """Number of feature channels d_k (second dimension of data)."""
        return self._data.size(1)

    # -----------------------
    # Device / dtype movement
    # -----------------------
    def to(self, *args, **kwargs) -> "CoChain":
        """Move the underlying tensor to the specified device / dtype."""
        self._data = self._data.to(*args, **kwargs)
        return self

    def cpu(self) -> "CoChain":
        """Move the underlying tensor to CPU."""
        self._data = self._data.cpu()
        return self

    def cuda(self) -> "CoChain":
        """Move the underlying tensor to CUDA."""
        self._data = self._data.cuda()
        return self

    def pin_memory(self) -> "CoChain":
        """Move the underlying tensor to pinned host memory."""
        self._data = self._data.pin_memory()
        return self


class GradedCochain:
    """Graded cochain x = (x^0, ..., x^dim) on a cell complex.

    This aggregates one :class:`CoChain` per degree, each with its own
    number of cells N_k and channels d_k.  Degrees may be sparse (some
    k between 0 and dim may be missing); the missing degrees are
    interpreted as the zero cochain.

    Internally, cochains are stored in a dense list of length
    ``dim + 1`` where the entry at index ``k`` is either a ``CoChain``
    instance for degree k, or ``None`` if that degree is absent.

    Args:
        cochains: sequence of CoChain objects (not necessarily ordered
            or dense in k).
        dim: optional maximum degree.  If None, inferred as
            ``max(c.k for c in cochains)``.
        validate: if True, run consistency checks on degrees and shapes.
    """

    def __init__(
        self,
        cochains: Sequence[CoChain],
        dim: Optional[int] = None,
        validate: bool = True,
    ):
        if not cochains:
            raise ValueError("Cannot create GradedCochain from empty sequence")

        # Infer dimension if not provided
        max_k = max(c.k for c in cochains)
        self.dim = int(dim) if dim is not None else int(max_k)

        # Build dense storage: one slot per 0 <= k <= dim
        self._by_k: list[Optional[CoChain]] = [None] * (self.dim + 1)
        for c in cochains:
            k = c.k
            if k < 0:
                raise ValueError(f"Cochain degree k must be >= 0, got {k}")
            if k > self.dim:
                raise ValueError(
                    "Cochain degree k must satisfy k <= dim, "
                    f"got k={k}, dim={self.dim}",
                )
            if self._by_k[k] is not None:
                raise ValueError(
                    f"Multiple cochains specified for degree k={k}; "
                    "expected at most one per degree.",
                )
            self._by_k[k] = c

        if validate:
            self._validate_basic()

    # -----------------------
    # Basic access
    # -----------------------
    def has_k(self, k: int) -> bool:
        """Check if a k-degree cochain is present (non-zero)."""
        if k < 0 or k > self.dim:
            return False
        return self._by_k[k] is not None

    def cochain(self, k: int) -> Optional[CoChain]:
        """Return the CoChain at degree k, or None if absent."""
        if k < 0 or k > self.dim:
            raise ValueError(f"Degree k must satisfy 0 <= k <= dim, got k={k}")
        return self._by_k[k]

    def x_k(self, k: int) -> Optional[Tensor]:
        """Get k-degree feature tensor, or None if absent."""
        c = self.cochain(k)
        return c.data if c is not None else None

    def d_k(self, k: int) -> int:
        """Get number of channels d_k for degree k.

        Raises if the degree is missing.
        """
        c = self.cochain(k)
        if c is None:
            raise ValueError(f"No cochain stored for degree k={k}")
        return c.channels

    def n_k(self, k: int) -> int:
        """Get number of k-cells N_k for degree k.

        Raises if the degree is missing.
        """
        c = self.cochain(k)
        if c is None:
            raise ValueError(f"No cochain stored for degree k={k}")
        return c.n_cells

    # -----------------------
    # Validation
    # -----------------------
    def _validate_basic(self) -> None:
        """Basic sanity checks on degree layout and tensor shapes.

        This enforces that all stored degrees satisfy 0 <= k <= dim and
        that there is **at most one** cochain per degree.  Missing
        degrees between 0 and dim are allowed and interpreted as the
        zero cochain.
        """
        # dim constraint: already ensured in __init__ via construction
        for k, c in enumerate(self._by_k):
            if c is None:
                continue
            if c.k != k:
                raise RuntimeError(
                    "Internal invariant violated: stored CoChain has "
                    f"k={c.k} but lives at index {k}",
                )
            data = c.data
            if data.ndim != 2:
                raise ValueError(
                    f"Cochain at k={k} must be rank-2 Tensor (N_k, d_k), "
                    f"got ndim={data.ndim}",
                )
            if data.numel() == 0:
                raise ValueError(f"Cochain at k={k} must be non-empty")

    def validate_against_complex(self, complex_obj, allow_missing: bool = True) -> None:
        """Check x[k].shape[0] == complex.n_cells(k) for all stored k.

        Does not require all degrees to be present unless allow_missing=False.
        """
        # Check present degrees
        for k, c in enumerate(self._by_k):
            if c is None:
                continue
            n_k = _as_int(complex_obj.n_cells(k))
            if c.n_cells != n_k:
                raise ValueError(
                    f"Cochain at k={k} has N={c.n_cells} but complex has N_k={n_k}",
                )

        if not allow_missing:
            dim = _as_int(complex_obj.dim)
            for k in range(dim + 1):
                if not self.has_k(k):
                    raise ValueError(f"Missing cochain for k={k}")

    # -----------------------
    # Device / dtype
    # -----------------------
    def to(self, *args, **kwargs) -> "GradedCochain":
        """Move tensors to device/dtype."""
        for c in self._by_k:
            if c is not None:
                c.to(*args, **kwargs)
        return self

    def cpu(self) -> "GradedCochain":
        """Move tensors to CPU."""
        return self.to("cpu")

    def cuda(self) -> "GradedCochain":
        """Move tensors to CUDA."""
        return self.to("cuda")

    def pin_memory(self) -> "GradedCochain":
        """Move tensors to pinned memory."""
        for c in self._by_k:
            if c is not None:
                c.pin_memory()
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
    ) -> "GradedCochain":
        """Allocate zero features for the specified degrees."""
        cochains: list[CoChain] = []
        for k, d in d_by_k.items():
            n_k = _as_int(complex_obj.n_cells(k))
            data = torch.zeros(
                (n_k, d),
                dtype=dtype,
                device=device,
            )
            cochains.append(CoChain(k=k, data=data))

        dim = max(d_by_k.keys()) if d_by_k else _as_int(complex_obj.dim)
        return cls(cochains=cochains, dim=dim, validate=True)

    @classmethod
    def from_tensors(
        cls,
        x_by_k: Sequence[Optional[Tensor]],
        validate: bool = True,
    ) -> "GradedCochain":
        """Build from list/tuple where index is k."""
        cochains: list[CoChain] = []
        for k, t in enumerate(x_by_k):
            if t is None:
                continue
            cochains.append(CoChain(k=k, data=t))

        dim = max((c.k for c in cochains), default=-1)
        return cls(cochains=cochains, dim=dim, validate=validate)
