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
