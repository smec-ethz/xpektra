## v0.3.2 (2025-12-02)

### Fix

- **solvers**: adds newton solver with implicit differentiation using matrix free solve (db184ea)
- modify multiscale problem to be more optimized and computationally efficient, corrected inconsistenty in typing (d21abd8)
- attempt at multiscale by coupling tatva and xpektra, add cg with implicit differentiation (85d3797)
- **spectral_operator**: add laplacian operator (48662ac)
- **scheme**: adapt scheme for computing gradients for 1D domain, adds phasefield locaization as an example (b585c4e)

## v0.3.1 (2025-11-13)

### Fix

- transform now agnostic of shape, examples adapted to new design (8e2fb12)
- **examples**: adpating examples based on restructed projectors (2edc262)
- **projection operator**: reimplements fourier-galerkin projection operator as matrix-free for better memory utilization (3aa78d0)
- removing tensor operator as imput to Galerkin projection (6c63e2b)

## v0.3.0 (2025-11-08)

### Feat

- adds Moulinec-Suquet projection operator, divergence operator, symmetric gradient operator and preconditioned cg solver (a2e223c)

### Fix

- **example**: working moulinec suquet formulation with fixed-point iteration (4c8483c)
- **operator**: makes projection operator pure abstract, moves scheme and tensor to inherited class (0b43992)
- bump jax version to 0.8.0 (0aff0aa)

## v0.2.0 (2025-10-18)

### Feat

- restructuring and refactoring code for modularity and extension (541ec39)
- adds mkdocs for the documentation (73aaafd)
- add new library name (559e16a)
- add operators and space for modularity (736abe5)

### Fix

- changes the internal memory layout of representing fields (b5b3dcc)
- **operators**: change operators to green function (ab0da52)
- **docs**: tests different setting for mkdocs (efbae99)
- **docs**: corrects path to the logo in site and readme (9114595)
- update the example on linear and elastoplasticity (13158a4)
