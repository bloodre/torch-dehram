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

## Status
- Not yet stable


