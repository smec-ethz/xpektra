import jax

jax.config.update("jax_compilation_cache_dir", "/cluster/scratch/mpundir/jax-cache")
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)

import jax.numpy as jnp  # type: ignore

jax.config.update("jax_enable_x64", True)  # use double-precision
jax.config.update("jax_platforms", "cpu")

# -----------------------------GRID TENSOR OPERATIONS -----------------------------


# tensor operations / products: np.einsum enables index notation, avoiding loops
# e.g. ddot42 performs $C_ij = A_ijkl B_lk$ for the entire grid
@jax.jit
def trans2(A2):
    return jnp.einsum("ijxy->jixy  ", A2)


@jax.jit
def trace2(A2):
    return jnp.einsum("iixy         ->xy    ", A2)


@jax.jit
def dot(A, B):
    return jnp.einsum("ij,ji->ij", A, B)


@jax.jit
def dot21(A, v):
    return jnp.einsum("ij...,j...  ->i...", A, v, optimize="optimal")


@jax.jit
def ddot22(A2, B2):
    return jnp.einsum("ijxy  ,jixy  ->xy    ", A2, B2)


@jax.jit
def ddot42(A4, B2):
    return jnp.einsum("ijklxy,lkxy  ->ijxy  ", A4, B2)


@jax.jit
def ddot44(A4, B4):
    return jnp.einsum("ijklxy,lkmnxy->ijmnxy", A4, B4)


@jax.jit
def dot11(A1, B1):
    return jnp.einsum("ixy   ,ixy   ->xy    ", A1, B1)


@jax.jit
def dot22(A2, B2):
    return jnp.einsum("ijxy  ,jkxy  ->ikxy  ", A2, B2)


@jax.jit
def dot24(A2, B4):
    return jnp.einsum("ijxy  ,jkmnxy->ikmnxy", A2, B4)


@jax.jit
def dot42(A4, B2):
    return jnp.einsum("ijklxy,lmxy  ->ijkmxy", A4, B2)


@jax.jit
def dyad22(A2, B2):
    return jnp.einsum("ijxy  ,klxy  ->ijklxy", A2, B2)


@jax.jit
def dyad11(A1, B1):
    return jnp.einsum("ixy   ,jxy   ->ijxy  ", A1, B1)
