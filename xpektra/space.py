import jax
import jax.numpy as jnp  # type: ignore
from jax import Array
import equinox as eqx
from typing import Optional

from xpektra.transform import Transform

class SpectralSpace(eqx.Module):
    """Defines the spectral space

    Args:
        size: The size of the spectral space.
        dim: The dimension of the spectral space.
        length: The length of the spectral space.
        iota: The imaginary unit.

    Returns:
        The spectral space.

    Example:
        >>> space = SpectralSpace(size=10, dim=1, length=1.0)
        >>> space.fft(jnp.array([1.0, 2.0, 3.0, 4.0]))
        >>> space.ifft(jnp.array([ 5. +0.j, -5.+10.j], dtype=complex64))
        >>> space.frequency_vector()
        >>> space.wavenumber_vector()
        >>> space.differential_vector(jnp.array([1.0, 2.0, 3.0, 4.0]), "forward_difference")
    """

    lengths: tuple[float, ...] = eqx.field(static=True)
    transform: Transform = eqx.field(static=True)


    def get_wavenumber_mesh(self) -> list[Array]:
        """Creates a list of coordinate arrays for the wavenumbers."""
        #ξ = self.transform.get_wavenumber_vector()
        #if self.dim == 1:
        #    return [ξ]
        #else:
        #    return list(
        #        jnp.meshgrid(*([ξ] * self.dim), indexing="ij")
        #    )

        k_vecs = [
            self.transform.get_wavenumber_vector(n, l) 
            for n, l in zip(self.transform.shape, self.lengths)
        ]
        return list(jnp.meshgrid(*k_vecs, indexing='ij'))