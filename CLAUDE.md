# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## About

`xpektra` is a Python library providing modular building blocks for spectral methods in solid mechanics. It is built on JAX and Equinox, enabling differentiable spectral computations. Key use cases include computational homogenization, multiscale simulations, and FFT-based micromechanics solvers.

## Commands

**Install for development:**
```bash
pip install -e '.[dev]'
```

**Run all tests:**
```bash
pytest
```

**Run a single test:**
```bash
pytest tests/test_schemes.py::test_fourier_exactness -v
```

**Build docs:**
```bash
pip install -e '.[docs]'
mkdocs build
```

**Versioning / changelog (uses commitizen):**
```bash
cz bump        # bump version and update changelog
cz changelog   # generate changelog only
```

## Architecture

The library is organized around a composition pattern where small, focused modules are combined to build complex spectral solvers.

### Core abstractions (in `xpektra/`)

**`Transform`** (`transform.py`) — Abstract base for FFT-like operations. `FFTTransform` is the concrete implementation (JAX `fftn`/`ifftn`). The transform handles forward/inverse transforms and wavenumber vector generation.

**`SpectralSpace`** (`space.py`) — Equinox module tying together a grid `shape`, physical `lengths`, and a `Transform`. Provides `get_wavenumber_mesh()` used downstream by schemes and operators.

**`Scheme`** (`scheme.py`) — Abstract base for discretization strategies. `DiagonalScheme` handles Cartesian grids where differentiation is diagonal in Fourier space. It stores a pre-computed `gradient_operator` array and provides `apply_gradient`, `apply_divergence`, `apply_symmetric_gradient`, and `apply_laplacian`. Concrete schemes differ only in their `formula()` method:
- `FourierScheme` — exact spectral derivative (`iξ`)
- `CentralDifference`, `ForwardDifference`, `BackwardDifference` — finite-difference variants
- `RotatedDifference` — Willot/HEX8R rotated scheme (2D+ only)
- `FourthOrderCentralDifference`, `SixthOrderCentralDifference`, `EighthOrderCentralDifference` — higher-order FD

**`SpectralOperator`** (`spectral_operator.py`) — Combines `SpectralSpace` + `Scheme` + `TensorOperator` into a user-facing API. Exposes `grad`, `div`, `sym_grad`, `laplacian`, `forward`/`inverse`, and tensor ops (`dot`, `ddot`, `trace`, `trans`, `dyad`). All methods are JIT-compiled via `@eqx.filter_jit`.

**`TensorOperator`** (`tensor_operator.py`) — Handles rank-aware tensor algebra using dispatch tables (`DOT_EINSUM_DISPATCH`, `DDOT_EINSUM_DISPATCH`, etc.). Tensor rank is inferred from array shape minus `dim` spatial dimensions. Array layout is `(spatial..., tensor...)`.

**`ProjectionOperator`** (`projection_operator.py`) — Two implementations:
- `GalerkinProjection` — matrix-free projection using the scheme's gradient operator; memory-efficient
- `MoulinecSuquetProjection` — classical Green's operator for isotropic reference materials (requires Lamé parameters)

**`make_field`** (`__init__.py`) — Creates zero-filled arrays with layout `(spatial..., tensor...)` for rank-0 (scalar), rank-1 (vector), or rank-2 (tensor) fields.

### Green's functions (`xpektra/green_functions/`)

Contains specialized operator implementations: `fourier_galerkin.py`, `displacement_based_operators.py`, `spatial.py`, `tensor.py`. These are more specialized than the core projection operators.

### Solvers (`xpektra/solvers/`)

`linear.py` provides JAX-native conjugate gradient variants (`conjugate_gradient_while`, `conjugate_gradient_scan`, `bound_conjugate_gradient`). `petsc_solvers.py` provides PETSc-based alternatives.

### Data flow for a typical FFT-based solver

```
SpectralSpace + FFTTransform
        |
     Scheme (e.g. FourierScheme)
        |
  SpectralOperator  ←→  TensorOperator
        |
 ProjectionOperator (GalerkinProjection or MoulinecSuquetProjection)
        |
    Solver (CG or Newton-Krylov via scipy/jaxopt)
```

### JAX / Equinox conventions

- All core classes are `eqx.Module` (pytree-compatible, frozen after construction).
- `static=True` fields (shape, lengths, transform, dim) are compile-time constants for JIT.
- Computationally intensive methods use `@eqx.filter_jit`.
- Enable 64-bit precision with `jax.config.update("jax_enable_x64", True)` (done in tests; required for numerical accuracy in examples).
- Fields and operators use `(spatial..., tensor...)` axis ordering throughout.
