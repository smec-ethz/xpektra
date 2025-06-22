from functools import partial
import jax
jax.config.update("jax_enable_x64", True)  # use double-precision
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)


import jax.numpy as jnp  # type: ignore



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
