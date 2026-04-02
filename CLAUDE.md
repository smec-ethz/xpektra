# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## About

`xpektra` is a Python library providing modular building blocks for spectral methods in solid mechanics. It is built on JAX, enabling differentiable spectral computations. Key use cases include computational homogenization, multiscale simulations, and FFT-based micromechanics solvers.

## Commands

**Install for development:**
```bash
uv pip install -e '.[dev]'
```

**Run all tests:**
```bash
uv run pytest
```

**Run a single test:**
```bash
uv run pytest tests/test_schemes.py::test_fourier_exactness -v
```

**Build docs:**
```bash
uv pip install -e '.[docs]'
uv run mkdocs build
```

**Versioning / changelog (uses commitizen):**
```bash
cz bump        # bump version and update changelog
cz changelog   # generate changelog only
```

## Architecture

The library is organized around a composition pattern where small, focused modules are combined to build complex spectral solvers.

### Core abstractions (in `xpektra/`)

**`Transform`** (`transform.py`) — Abstract base for FFT-like operations, implemented as a frozen dataclass registered as a JAX pytree. `FFTTransform` is the concrete implementation (JAX `fftn`/`ifftn`), also a registered frozen dataclass. The transform handles forward/inverse transforms and wavenumber vector generation.

**`SpectralSpace`** (`space.py`) — Frozen dataclass registered as a JAX pytree via `@jax.tree_util.register_dataclass`. Ties together a grid `shape`, physical `lengths`, and a `Transform`. Provides `get_wavenumber_mesh()` used downstream by schemes and operators.

**`Scheme`** (`scheme.py`) — Abstract base class (ABC) for discretization strategies. `DiagonalScheme` is registered as a JAX pytree via `@jax.tree_util.register_pytree_node_class` with `gradient_operator` as the dynamic child and `dim`/`space` as static aux_data. Immutability is enforced via `__setattr__`. Each concrete scheme is also registered as a separate pytree node class and inherits `tree_flatten`/`tree_unflatten` from `DiagonalScheme`. Concrete schemes differ only in their `formula()` method:
- `FourierScheme` — exact spectral derivative (`iξ`)
- `CentralDifference`, `ForwardDifference`, `BackwardDifference` — finite-difference variants
- `RotatedDifference` — Willot/HEX8R rotated scheme (2D+ only)
- `FourthOrderCentralDifference`, `SixthOrderCentralDifference`, `EighthOrderCentralDifference` — higher-order FD

**`SpectralOperator`** (`spectral_operator.py`) — Frozen dataclass registered as a JAX pytree via `@jax.tree_util.register_dataclass`. Combines `SpectralSpace` + `Scheme` + `TensorOperator` into a user-facing API. Exposes `grad`, `div`, `sym_grad`, `laplacian`, `forward`/`inverse`, and tensor ops (`dot`, `ddot`, `trace`, `trans`, `dyad`). All methods are JIT-compiled via `@jax.jit`.

**`TensorOperator`** (`tensor_operator.py`) — Registered as a JAX pytree via `@jax.tree_util.register_pytree_node_class`. Handles rank-aware tensor algebra using dispatch tables (`DOT_EINSUM_DISPATCH`, `DDOT_EINSUM_DISPATCH`, etc.). Tensor rank is inferred from array shape minus `dim` spatial dimensions. Array layout is `(spatial..., tensor...)`. Immutability is enforced via `__setattr__`; structural `__eq__`/`__hash__` are implemented for correct JAX cache behavior.

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

### JAX pytree conventions

- Classes use one of two pytree registration patterns:
  - **Frozen dataclasses** (`@jax.tree_util.register_dataclass` + `@dataclass(frozen=True)`): Used by `Transform`, `FFTTransform`, `SpectralSpace`, `SpectralOperator`. Fields use `metadata=dict(static=True)` to mark compile-time constants.
  - **Manual pytree registration** (`@jax.tree_util.register_pytree_node_class`): Used by `TensorOperator`, `DiagonalScheme`, and all concrete schemes. These implement `tree_flatten`/`tree_unflatten` manually. Immutability is enforced via `__setattr__` with an `_initialized` sentinel.
- For class hierarchies (e.g. `DiagonalScheme` → `FourierScheme`), each concrete class must have its own `@register_pytree_node_class` decorator. `tree_unflatten` uses `@classmethod` with `cls` so subclasses inherit it and get the correct type back.
- Computationally intensive methods use `@jax.jit`.
- Enable 64-bit precision with `jax.config.update("jax_enable_x64", True)` (done in tests; required for numerical accuracy in examples).
- Fields and operators use `(spatial..., tensor...)` axis ordering throughout.
