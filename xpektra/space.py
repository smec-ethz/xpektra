import jax
import jax.numpy as jnp  # type: ignore
from jax import Array
import equinox as eqx
from typing import Optional


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

    size: int
    dim: int
    length: float
    iota: Optional[complex] = 1j

    def fft(self, x: Array) -> Array:
        axes_to_transform = range(self.dim)
        return jnp.fft.fftshift(
            jnp.fft.fftn(
                jnp.fft.ifftshift(x),
                s=[self.size] * self.dim,
                axes=axes_to_transform,
            )
        )

    def ifft(self, x: Array) -> Array:
        axes_to_transform = range(self.dim)
        return jnp.fft.fftshift(
            jnp.fft.ifftn(
                jnp.fft.ifftshift(x),
                s=[self.size] * self.dim,
                axes=axes_to_transform,
            )
        )

    def frequency_vector(self) -> Array:
        freq = (
            jnp.arange(-(self.size - 1) / 2, +(self.size + 1) / 2, dtype="int64")
            / self.length
        )
        return freq

    def wavenumber_vector(self) -> Array:
        freq = self.frequency_vector()
        return 2 * jnp.pi * freq