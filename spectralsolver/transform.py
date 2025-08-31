import jax
jax.config.update("jax_enable_x64", True)  # use double-precision
import jax.numpy as jnp  # type: ignore
import equinox as eqx


def _fft(x, N, ndim):
    return jnp.fft.fftshift(
        jnp.fft.fftn(
            jnp.fft.ifftshift(x),
            [
                N,
            ]
            * ndim,
        )
    )


def _ifft(x, N, ndim):
    return jnp.fft.fftshift(
        jnp.fft.ifftn(
            jnp.fft.ifftshift(x),
            [
                N,
            ]
            * ndim,
        )
    )
