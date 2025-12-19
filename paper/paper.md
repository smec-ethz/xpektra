---
title: 'xpektra: A modular and differentiable framework for spectral methods in solid mechanics'
tags:
  - Python
  - mechanics
  - homogenization
  - FFT
  - JAX
  - differentiable programming
  - solid mechanics
authors:
  - name: Mohit Pundir
    orcid: 0000-0002-3652-3297
    equal_contrib: true
    affiliation: 1
affiliations:
 - name: ETH Zurich, Switzerland
   index: 1
date: 13 August 2025
bibliography: paper.bib
---

# Summary

`xpektra` is a Python library designed to provide a modular and differentiable framework for solving partial differential equations (PDEs) using spectral methods, with a primary focus on computational homogenization in solid mechanics. Built on top of JAX [@jax2018github] and Equinox [@kidger2021equinox], `xpektra` leverages automatic differentiation and just-in-time (JIT) compilation to offer high-performance solvers that are easily extensible. Unlike traditional monolithic implementations of Fast Fourier Transform (FFT)-based homogenization, `xpektra` decomposes the solution process into independent building blocks—operators, spaces, discretization schemes, and solvers—allowing researchers to mix and match components to construct custom spectral algorithms.

# Statement of need

Homogenization of heterogeneous materials is a critical task in computational mechanics, essential for predicting the effective properties of composites, polycrystals, and metamaterials. Since the seminal work of Moulinec and Suquet [@moulinec1998numerical], FFT-based methods have become a standard tool due to their computational efficiency compared to Finite Element Methods (FEM).

However, the ecosystem of FFT-based software faces significant challenges:
1.  **Monolithic Design:** Many existing codes are highly specialized, hard-coding specific discretization schemes (e.g., only "basic scheme" or only "staggered grid") or material laws, making it difficult to benchmark new theoretical developments.
2.  **Lack of Differentiability:** Most high-performance implementations are written in C++ or Fortran. While fast, they lack support for automatic differentiation (AD). This makes gradient-based tasks—such as inverse homogenization, topology optimization, or material parameter identification—arduous, often requiring manual derivation of adjoint states.

`xpektra` addresses these gaps by providing a fully differentiable, modular framework. By utilizing JAX, `xpektra` enables users to compute gradients of the homogenized response with respect to any input parameter (e.g., microstructure geometry or material properties) automatically. This capability is crucial for the emerging fields of data-driven mechanics and differentiable physics. Furthermore, its modular design allows users to seamlessly switch between formulations (e.g., Lippmann-Schwinger vs. Galerkin) and discretization schemes (e.g., central difference vs. spectral derivatives) without rewriting the core solver logic.

# Methodology

`xpektra` abstracts the spectral solution process into orthogonal choices, allowing users to reconstruct classic methods or invent new ones.

## 1. Formulation
The library supports three distinct mathematical formulations for the homogenization problem:

* **The Lippmann-Schwinger Approach:** Implemented via the `MoulinecSuquetProjection` class, this formulation uses a reference material to define a periodic Green's operator. It corresponds to the classic "basic scheme" [@moulinec1998numerical].
* **The Variational (Galerkin) Approach:** Implemented via the `GalerkinProjection` class, this formulation enforces compatibility and equilibrium using a projection operator derived from the weak form of the PDE [@vondrejc2014fft; @brisard2010fft]. It does not strictly require a reference material and is often more robust for high-contrast materials.
* **The Displacement-Based Approach (DBFFT):** Unlike the stress-strain based formulations above, `xpektra` also supports the Displacement-Based FFT framework. Here, the governing equation is solved directly for the displacement field $\boldsymbol{u}$: $\nabla \cdot (\boldsymbol{C} : \nabla^s \boldsymbol{u}) = 0$. This formulation produces a full-rank linear system, enabling the effective use of standard preconditioners which is often difficult with strain-based methods.

## 2. Discretization (The Scheme)
To address the well-known "Gibbs ringing" artifacts associated with spectral derivatives, `xpektra` decouples the derivative calculation from the solver. The library provides a rich suite of discretization schemes via the `Scheme` abstract base class:

* **Spectral Derivatives:** The classic `FourierScheme`, accurate for smooth fields but prone to ringing at sharp interfaces.
* **Rotated Finite Differences:** The `RotatedDifference` scheme (equivalent to linear finite elements with reduced integration [@willot2015fourier]), which eliminates ringing and improves stability for porous materials.
* **Standard Finite Differences:** A comprehensive collection of difference stencils, including `CentralDifference` (2nd order), `ForwardDifference`, `BackwardDifference`, as well as high-order approximations (`FourthOrderCentralDifference`, `SixthOrderCentralDifference`, and `EighthOrderCentralDifference`).

## 3. Extensibility
A core philosophy of `xpektra` is its "abstract-or-final" design. The library defines clear API contracts via abstract base classes (`Scheme`, `ProjectionOperator`). This allows researchers to implement entirely new methods without modifying the core library code.

* **New Schemes:** Users can implement custom finite difference stencils (e.g., the TETRA2 scheme) simply by subclassing `Scheme` and defining the discrete gradient operator.
* **New Solvers:** Users can plug in custom optimization strategies (e.g., Anderson Acceleration or L-BFGS) by interacting with the JIT-compiled residual functions provided by the library.

# Example Usage

The following example demonstrates setting up a linear elasticity problem using the Fourier-Galerkin method with a rotated finite-difference discretization scheme.

```python
import jax.numpy as jnp
from xpektra import SpectralSpace, SpectralOperator, make_field
from xpektra.projection_operator import GalerkinProjection
from xpektra.scheme import RotatedDifference
from xpektra.transform import FFTTransform
from xpektra.solvers.nonlinear import NewtonSolver, newton_krylov_solver

# 1. Define the Spectral Space and Discretization Scheme
ndim, N = 2, 64
fft_transform = FFTTransform(dim=ndim)
# Define a 64x64 grid
space = SpectralSpace(lengths=(1.0, 1.0), shape=(N, N), transform=fft_transform)
# Use Rotated Difference scheme to avoid Gibbs ringing
scheme = RotatedDifference(space=space)

# 2. Define the Operator
# The Galerkin operator projects stress onto compatible strain fields
operator = GalerkinProjection(scheme=scheme)

# 3. Define the Residual Function (Physics)
def residual_fn(strain_fluctuation_flat, macro_strain):
    # Reshape flattened input to grid
    strain_fluc = strain_fluctuation_flat.reshape(shape_tensor)
    total_strain = strain_fluc + macro_strain
    
    # Compute stress (User-defined constitutive law)
    sigma = compute_stress(total_strain) 
    
    # Apply Projection: G : sigma = 0 (Equilibrium)
    # operator.project applies the FFT-based projection
    residual = operator.project(sigma) 
    return residual.flatten()

# 4. Solve using Newton-Krylov
solver = NewtonSolver(krylov_solver=newton_krylov_solver)
solution = solver.solve(x=initial_guess, f=residual_fn)
```

![Stress field ($\sigma_{xx}$) computed for a circular inclusion problem using the Fourier-Galerkin method. The sharp interface is handled using the rotated finite difference scheme.](figure1.png){ width=80% }

# Performance

`xpektra` is designed for high-performance computing. By leveraging JAX's Just-In-Time (JIT) compilation and hardware acceleration, the solvers scale efficiently on modern hardware.

![Computational time per iteration vs. grid size ($N$) for the linear elasticity solver. The JAX-based implementation on a GPU shows significant speedup compared to standard CPU execution.](figure2.png){ width=70% }