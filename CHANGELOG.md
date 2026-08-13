## v0.3.3 (2025-12-05)

### Fix

- removes tatva as tool (42db8f7)

## [0.5.0](https://github.com/smec-ethz/xpektra/compare/v0.4.1...v0.5.0) (2026-08-13)


### Features

* **scheme:** add TETRA2 scheme, change base scheme for modularity ([d25ec2d](https://github.com/smec-ethz/xpektra/commit/d25ec2d97e83bbddf542a7499da3c23e96b5c86e))
* **transform:** add slab and pencil decomposition for parallel FFTs ([9bee2fb](https://github.com/smec-ethz/xpektra/commit/9bee2fb3f44d90b188a1282912e8d5b6192419e8))
* **transform:** add slab decomposition for 3D FFT ([ae5734d](https://github.com/smec-ethz/xpektra/commit/ae5734d50a55cd6c504fa4af46f699ce58815705))


### Bug Fixes

* adapt examples, but need to be moved ([8da6366](https://github.com/smec-ethz/xpektra/commit/8da6366d47a5042d628597ca2585f83e5d062f1c))
* remove twice computation of operator in symmetric gradient ([7a88c0b](https://github.com/smec-ethz/xpektra/commit/7a88c0b5d430d0a6b784146a07d92d9a299bbef2))
* **scheme:** add stencils for forward, backeard and central scheme ([8da6366](https://github.com/smec-ethz/xpektra/commit/8da6366d47a5042d628597ca2585f83e5d062f1c))
* **scheme:** add sympy based derivation of fourier operator from FD stencil in real space ([220a332](https://github.com/smec-ethz/xpektra/commit/220a3326d41afcc4115669ffe232c191a8392153))
* **tensor:** replace einsum with broadcast rule for no GEMM under vmap ([5567ddf](https://github.com/smec-ethz/xpektra/commit/5567ddf7c41adbf911162247b92e89d1e3e8d1b1))
* **tensor:** use jnp.trace for trace operator ([23b1dac](https://github.com/smec-ethz/xpektra/commit/23b1dacd4ece3a02366cf7a9dab7baeb63ffc5ff))

## [0.4.1](https://github.com/smec-ethz/xpektra/compare/v0.4.0...v0.4.1) (2026-04-12)


### Bug Fixes

* make soldis, tatva as example dependency ([daac7c5](https://github.com/smec-ethz/xpektra/commit/daac7c56b8c743bdfeb95f1ca4c4be92a26cac29))

## [0.4.0](https://github.com/smec-ethz/xpektra/compare/v0.3.3...v0.4.0) (2026-04-04)


### Features

* single operator as entry point, examples now use soldis as solver ([4847be4](https://github.com/smec-ethz/xpektra/commit/4847be4159ea60e36582b4b1ca43d4093ca39f51))


### Bug Fixes

* add hyperelastic example, remove example on Gent ([80c1d9c](https://github.com/smec-ethz/xpektra/commit/80c1d9c739051613aa64630b1bb839ed988f4971))
* move soldis as example dependency ([4652f54](https://github.com/smec-ethz/xpektra/commit/4652f54bf42a487d604095e687f8aab21e87a9cc))
* removing equinox dependency from the core modules ([7919e7c](https://github.com/smec-ethz/xpektra/commit/7919e7c81c720c1feea5a269637c4642e7d8e535))
* **tests:** adds more test suite ([18e0deb](https://github.com/smec-ethz/xpektra/commit/18e0deb423b19fcb629a1775e1e307a5b7a2cc76))
* update linear elastic example, add tests for schemes ([afbcaeb](https://github.com/smec-ethz/xpektra/commit/afbcaeb6050aaa32bee4a73ea57a714f3d890eba))

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
