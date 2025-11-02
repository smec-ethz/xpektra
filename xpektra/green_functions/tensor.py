import jax

jax.config.update("jax_enable_x64", True)  # use double-precision


import jax.numpy as jnp  # type: ignore


# -----------------------------GRID TENSOR OPERATIONS -----------------------------


# tensor operations / products: np.einsum enables index notation, avoiding loops
# e.g. ddot42 performs $C_ij = A_ijkl B_lk$ for the entire grid
# @jax.jit
def trans2(A2: jnp.ndarray) -> jnp.ndarray:
    """Transpose a second-order tensor.

    Args:
        A2: Second-order tensor of shape (i,j,...)

    Returns:
        Transposed tensor of shape (j,i,...)
    """
    return jnp.einsum("ij...->ji...  ", A2, optimize="optimal")


def trace2(A2: jnp.ndarray) -> jnp.ndarray:
    """Compute the trace of a second-order tensor.

    Args:
        A2: Second-order tensor of shape (i,i,...)

    Returns:
        Scalar trace of shape (...)
    """
    return jnp.einsum("ii...->...    ", A2, optimize="optimal")


@jax.jit
def dot(A: jnp.ndarray, B: jnp.ndarray) -> jnp.ndarray:
    """Compute dot product of two matrices.

    Args:
        A: First matrix of shape (i,j)
        B: Second matrix of shape (j,i)

    Returns:
        Resulting matrix of shape (i,j)
    """
    return jnp.einsum("ij,ji->ij", A, B, optimize="optimal")


@jax.jit
def dot21(A: jnp.ndarray, v: jnp.ndarray) -> jnp.ndarray:
    """Compute dot product between a second-order tensor and a vector.

    Args:
        A: Second-order tensor of shape (i,j,...)
        v: Vector of shape (j,...)

    Returns:
        Resulting vector of shape (i,...)
    """
    return jnp.einsum("ij...,j...  ->i...", A, v, optimize="optimal")


def ddot22(A2: jnp.ndarray, B2: jnp.ndarray) -> jnp.ndarray:
    """Compute double dot product between two second-order tensors.

    Args:
        A2: First second-order tensor of shape (i,j,...)
        B2: Second second-order tensor of shape (j,i,...)

    Returns:
        Scalar result of shape (...)
    """
    return jnp.einsum("ij...  ,ji...  ->...    ", A2, B2, optimize="optimal")


def ddot42(A4: jnp.ndarray, B2: jnp.ndarray) -> jnp.ndarray:
    """Compute double dot product between a fourth-order and second-order tensor.

    Args:
        A4: Fourth-order tensor of shape (i,j,k,l,...)
        B2: Second-order tensor of shape (l,k,...)

    Returns:
        Second-order tensor of shape (i,j,...)
    """
    return jnp.einsum("ijkl...,lk...  ->ij...  ", A4, B2, optimize="optimal")


@jax.jit
def ddot44(A4: jnp.ndarray, B4: jnp.ndarray) -> jnp.ndarray:
    """Compute double dot product between two fourth-order tensors.

    Args:
        A4: First fourth-order tensor of shape (i,j,k,l,...)
        B4: Second fourth-order tensor of shape (l,k,m,n,...)

    Returns:
        Fourth-order tensor of shape (i,j,m,n,...)
    """
    return jnp.einsum("ijkl...,lkmn...->ijmn...", A4, B4, optimize="optimal")


@jax.jit
def dot11(A1: jnp.ndarray, B1: jnp.ndarray) -> jnp.ndarray:
    """Compute dot product between two vectors.

    Args:
        A1: First vector of shape (i,...)
        B1: Second vector of shape (i,...)

    Returns:
        Scalar result of shape (...)
    """
    return jnp.einsum("i...,i... ->...    ", A1, B1, optimize="optimal")


@jax.jit
def dot22(A2: jnp.ndarray, B2: jnp.ndarray) -> jnp.ndarray:
    """Compute dot product between two second-order tensors.

    Args:
        A2: First second-order tensor of shape (i,j,...)
        B2: Second second-order tensor of shape (j,k,...)

    Returns:
        Second-order tensor of shape (i,k,...)
    """
    return jnp.einsum("ij...  ,jk...  ->ik...  ", A2, B2, optimize="optimal")


@jax.jit
def dot24(A2: jnp.ndarray, B4: jnp.ndarray) -> jnp.ndarray:
    """Compute dot product between a second-order and fourth-order tensor.

    Args:
        A2: Second-order tensor of shape (i,j,...)
        B4: Fourth-order tensor of shape (j,k,m,n,...)

    Returns:
        Fourth-order tensor of shape (i,k,m,n,...)
    """
    return jnp.einsum("ij...  ,jkmn...->ikmn...", A2, B4, optimize="optimal")


@jax.jit
def dot42(A4: jnp.ndarray, B2: jnp.ndarray) -> jnp.ndarray:
    """Compute dot product between a fourth-order and second-order tensor.

    Args:
        A4: Fourth-order tensor of shape (i,j,k,l,...)
        B2: Second-order tensor of shape (l,m,...)

    Returns:
        Fourth-order tensor of shape (i,j,k,m,...)
    """
    return jnp.einsum("ijkl...,lm...  ->ijkm...", A4, B2, optimize="optimal")


@jax.jit
def dyad22(A2: jnp.ndarray, B2: jnp.ndarray) -> jnp.ndarray:
    """Compute dyadic product between two second-order tensors.

    Args:
        A2: First second-order tensor of shape (i,j,...)
        B2: Second second-order tensor of shape (k,l,...)

    Returns:
        Fourth-order tensor of shape (i,j,k,l,...)
    """
    return jnp.einsum("ij...  ,kl...  ->ijkl...", A2, B2, optimize="optimal")


@jax.jit
def dyad11(A1: jnp.ndarray, B1: jnp.ndarray) -> jnp.ndarray:
    """Compute dyadic product between two vectors.

    Args:
        A1: First vector of shape (i,...)
        B1: Second vector of shape (j,...)

    Returns:
        Second-order tensor of shape (i,j,...)
    """
    return jnp.einsum("i...   ,j...   ->ij...  ", A1, B1, optimize="optimal")
