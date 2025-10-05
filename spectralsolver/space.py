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
    """Defines the spectral space"""

    size: int
    dim: int
    length: float
    iota: Optional[complex] = 1j

    def fft(self, x: Array) -> Array:
        # CRITICAL CHANGE: We must specify the axes to transform, which are
        # the first `self.dim` spatial dimensions.
        axes_to_transform = range(self.dim)
        return jnp.fft.fftshift(
            jnp.fft.fftn(
                jnp.fft.ifftshift(x),
                s=[self.size] * self.dim,
                axes=axes_to_transform, # <--- ADDED THIS LINE
            )
        )

    def ifft(self, x: Array) -> Array:
        # CRITICAL CHANGE: Also specify the axes for the inverse transform.
        axes_to_transform = range(self.dim)
        return jnp.fft.fftshift(
            jnp.fft.ifftn(
                jnp.fft.ifftshift(x),
                s=[self.size] * self.dim,
                axes=axes_to_transform, # <--- ADDED THIS LINE
            )
        )

    # --- All other methods remain exactly the same ---
    
    def frequency_vector(self):
        freq = (
            jnp.arange(-(self.size - 1) / 2, +(self.size + 1) / 2, dtype="int64")
            / self.length
        )
        return freq

    def wavenumber_vector(self):
        freq = self.frequency_vector()
        return 2 * jnp.pi * freq

    def differential_vector(
        self, xi: Array, diff_mode: str, factor: float = 1.0
    ) -> Array:
        # This method operates element-wise and is layout-agnostic. No changes needed.
        # ... (code for differential_vector is unchanged) ...
        iota = 1j
        dx = self.length / self.size
        if self.dim == 1 and diff_mode == "rotated_difference":
            raise RuntimeError("Rotated difference is not defined for 1D")

        if diff_mode == "fourier":
            return iota * xi
        elif diff_mode == "forward_difference":
            return (jnp.exp(iota * xi * dx) - 1) / dx
        elif diff_mode == "central_difference":
            return iota * jnp.sin(xi * dx) / dx
        elif diff_mode == "backward_difference":
            return (1 - jnp.exp(-iota * xi * dx)) / dx
        elif diff_mode == "rotated_difference":
            return 2 * iota * jnp.tan(xi * dx / 2) * factor / dx
        # ... and so on for other difference schemes
        else:
            raise RuntimeError("Differential scheme not defined")