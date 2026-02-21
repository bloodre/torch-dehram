"""Finite Element Exterior Calculus (FEEC) for simplicial complexes.

Provides lowest-order Whitney finite element spaces on simplicial meshes,
with quadrature-based assembly of sparse mass matrices and inner products.

Main components:
- SimplicialGeometry: affine simplex geometry (Jacobians, det, pullback).
- Whitney basis evaluators on reference simplex (reference.py).
- Quadrature rules on reference simplices (quadrature.py).
- Mass matrix assembly (assembly.py).
- FEECInnerProduct: sparse mass-based inner product with CG solver.

Example usage:
    ```python
    from torch_derham.core import SimplicialChainComplex
    from torch_derham.core.feec import SimplicialGeometry, FEECInnerProduct
    from torch_derham.core.cochain import CoChain
    
    # Build complex and geometry
    complex = SimplicialChainComplex(...)
    positions = CoChain(k=0, data=vertex_coords)
    geometry = SimplicialGeometry.from_complex(complex, positions.data)
    
    # Assemble FEEC inner product with quadrature degree 2
    feec_ip = FEECInnerProduct.from_geometry(complex, geometry, quad_degree=2)
    
    # Use in DiscreteCalculus for codifferential, Laplacian, etc.
    from torch_derham.core.metric import DiscreteCalculus
    calculus = DiscreteCalculus(complex, feec_ip)
    ```
"""

from .assembly import (
    assemble_global_mass_all,
    assemble_global_mass_k,
    assemble_local_mass_k,
)
from .geometry import SimplicialGeometry
from .inner_product import FEECInnerProduct
from .quadrature import quadrature_simplex
from .reference import (
    enumerate_whitney_dofs,
    eval_whitney_kform_all,
)

__all__ = [
    "FEECInnerProduct",
    "SimplicialGeometry",
    "assemble_global_mass_all",
    "assemble_global_mass_k",
    "assemble_local_mass_k",
    "enumerate_whitney_dofs",
    "eval_whitney_kform_all",
    "quadrature_simplex",
]
