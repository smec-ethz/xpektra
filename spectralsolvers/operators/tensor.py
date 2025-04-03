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
    return jnp.einsum("ij...->ji...  ", A2, optimize="optimal")


@jax.jit
def trace2(A2):
    return jnp.einsum("ii...->...    ", A2, optimize="optimal")


@jax.jit
def dot(A, B):
    return jnp.einsum("ij,ji->ij", A, B, optimize="optimal")


@jax.jit
def dot21(A, v):
    return jnp.einsum("ij...,j...  ->i...", A, v, optimize="optimal")


@jax.jit
def ddot22(A2, B2):
    return jnp.einsum("ij...  ,ji...  ->...    ", A2, B2, optimize="optimal")


@jax.jit
def ddot42(A4, B2):
    return jnp.einsum("ijkl...,lk...  ->ij...  ", A4, B2, optimize="optimal")


@jax.jit
def ddot44(A4, B4):
    return jnp.einsum("ijkl...,lkmn...->ijmn...", A4, B4, optimize="optimal")


@jax.jit
def dot11(A1, B1):
    return jnp.einsum("i...,i... ->...    ", A1, B1, optimize="optimal")


@jax.jit
def dot22(A2, B2):
    return jnp.einsum("ij...  ,jk...  ->ik...  ", A2, B2, optimize="optimal")


@jax.jit
def dot24(A2, B4):
    return jnp.einsum("ij...  ,jkmn...->ikmn...", A2, B4, optimize="optimal")


@jax.jit
def dot42(A4, B2):
    return jnp.einsum("ijkl...,lm...  ->ijkm...", A4, B2, optimize="optimal")


@jax.jit
def dyad22(A2, B2):
    return jnp.einsum("ij...  ,kl...  ->ijkl...", A2, B2, optimize="optimal")


@jax.jit
def dyad11(A1, B1):
    return jnp.einsum("i...   ,j...   ->ij...  ", A1, B1, optimize="optimal")
