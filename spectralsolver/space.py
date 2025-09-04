import jax
import jax.numpy as jnp  # type: ignore
from jax import Array
import equinox as eqx
from typing import Optional


# --- Define the gradient modes ---
class DifferentialMode:
    fourier = "fourier"
    forward_difference = "forward_difference"
    central_difference = "central_difference"
    backward_difference = "backward_difference"
    rotated_difference = "rotated_difference"
    four_central_difference = "four_central_difference"
    six_central_difference = "six_central_difference"
    eight_central_difference = "eight_central_difference"


class SpectralSpace(eqx.Module):
    """Defines the spectral space

    Args:
        size (int): Number of grid points.
        dim (int): Dimension of the space. Defaults to 1.
        length (float): Length of the space. Defaults to 1.
        iota (complex): Imaginary unit. Defaults to 1j. Optional

    Methods:
        fft: Fast Fourier Transform.
        ifft: Inverse Fast Fourier Transform.
        frequency_vector: get the frequency vector.
        wavenumber_vector: get the wavenumber vector.
        differential_vector: get the differential vector.
    Raises:
        RuntimeError: Rotated difference is not defined for 1D.
        RuntimeError: Differential scheme not defined.

    Returns:
        Array: Transformed array.
    """

    size: int
    dim: int
    length: float
    iota: Optional[complex] = 1j

    def fft(self, x: Array) -> Array:
        return jnp.fft.fftshift(
            jnp.fft.fftn(
                jnp.fft.ifftshift(x),
                [
                    self.size,
                ]
                * self.dim,
            )
        )

    def ifft(self, x: Array) -> Array:
        return jnp.fft.fftshift(
            jnp.fft.ifftn(
                jnp.fft.ifftshift(x),
                [
                    self.size,
                ]
                * self.dim,
            )
        )

    def frequency_vector(self):
        freq = (
            jnp.arange(-(self.size - 1) / 2, +(self.size + 1) / 2, dtype="int64")
            / self.length
        )
        return freq  # 2*pi*(n)/samplingspace/n https://arxiv.org/pdf/1412.8398

    def wavenumber_vector(self):
        freq = self.frequency_vector()
        return 2 * jnp.pi * freq

    def differential_vector(
        self, xi: Array, diff_mode: DifferentialMode, factor: float = 1.0
    ) -> Array:
        iota = 1j
        dx = self.length / self.size
        if self.dim == 1 and diff_mode == DifferentialMode.rotated_difference:
            raise RuntimeError("Rotated difference is not defined for 1D")

        if diff_mode == DifferentialMode.fourier:
            return iota * xi
        elif diff_mode == DifferentialMode.forward_difference:
            return (jnp.exp(iota * xi * dx) - 1) / dx
        elif diff_mode == DifferentialMode.central_difference:
            return iota * jnp.sin(xi * dx) / dx
        elif diff_mode == DifferentialMode.backward_difference:
            return (1 - jnp.exp(-iota * xi * dx)) / dx
        elif diff_mode == DifferentialMode.rotated_difference:
            return 2 * iota * jnp.tan(xi * dx / 2) * factor / dx
        elif diff_mode == DifferentialMode.four_central_difference:
            return iota * (
                8 * jnp.sin(xi * dx) / (6 * dx) - jnp.sin(2 * xi * dx) / (6 * dx)
            )
        elif diff_mode == DifferentialMode.six_central_difference:
            return iota * (
                9 * jnp.sin(xi * dx) / (6 * dx)
                - 3 * jnp.sin(2 * xi * dx) / (10 * dx)
                + jnp.sin(3 * xi * dx) / (30 * dx)
            )
        elif diff_mode == DifferentialMode.eight_central_difference:
            return iota * (
                8 * jnp.sin(xi * dx) / (5 * dx)
                - 2 * jnp.sin(2 * xi * dx) / (5 * dx)
                + 8 * jnp.sin(3 * xi * dx) / (105 * dx)
                - jnp.sin(4 * xi * dx) / (140 * dx)
            )
        else:
            raise RuntimeError("Differential scheme not defined")
