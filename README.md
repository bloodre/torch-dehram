# About

Torch de Rham is an early-stage PyTorch toolbox for working with **cochains on cell complexes** (graphs/meshes/simplicial complexes) and building operators inspired by **discrete exterior calculus (DEC)** and **finite element exterior calculus (FEEC)**.

This repository is a starting point and the API is expected to change.

## Planned scope (draft)
- **Complex / cochain primitives**
  - Oriented complexes (initial focus: simplicial)
  - Cochains as tensors attached to k-cells
  - Coboundary operator `d` derived from topology (with basic consistency checks)

- **Inner products / Hodge (pluggable)**
  - Common interface for inner products / “Hodge” operators
  - DEC-style diagonal choices as a first baseline
  - FEEC-flavored choices (e.g., Whitney) as optional backends

- **Algebraic operators (research-oriented)**
  - Hodge Laplacians built from `d` and a chosen inner product
  - Experimental wedge / contraction operators (design TBD)

- **Utilities**
  - Helpers to construct complexes from simplices / meshes (TBD)
  - Optional interoperability with PyTorch Geometric (TBD)

## Installation

### Prerequisites

- **Poetry** (dependency manager): Install from https://python-poetry.org/docs/#installation
- **Make** (build tool): Available by default on Linux and macOS, or install via Homebrew on macOS

### Installation

```bash
make install
# Or for development
make install-dev
```

### Makefile Environment Variables

Override defaults when needed:

```bash
# Different PyTorch version (2.0.0<=TORCH_VER<=2.8.0)
make install TORCH_VER=2.7.0

# Different CUDA version (cu118, cu121, cpu)
# Should be available for torch-sparse and torch-scatter
make install CUDA=cpu

# Custom Poetry command
make install POETRY=poetry1
```

### Makefile Targets

- `make install` - Install base dependencies
- `make install-dev` - Install with development dependencies
- `make clean` - Remove Poetry virtual environment
- `make build` - Build distribution wheel

## Status
- Not yet stable
