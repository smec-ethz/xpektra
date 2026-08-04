from abc import abstractmethod
from dataclasses import dataclass, field

import jax
import jax.numpy as jnp
from jax import Array
from jax.sharding import PartitionSpec as P


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class Transform:
    """Abstract base class for all spectral transforms."""

    dim: int | None = field(metadata=dict(static=True), default=None)

    @abstractmethod
    def forward(self, x: Array) -> Array:
        """Perform the forward transform (e.g., FFT, DCT)."""
        raise NotImplementedError

    @abstractmethod
    def inverse(self, x_hat: Array) -> Array:
        """Perform the inverse transform (e.g., iFFT, iDCT)."""
        raise NotImplementedError

    @abstractmethod
    def get_wavenumber_vector(self, size: int, length: float) -> Array:
        """Get the 1D vector of wavenumbers (e.g., ξ for FFT, k for DCT)."""
        raise NotImplementedError


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class FFTTransform(Transform):
    """
    The standard, JAX-native Fast Fourier Transform.

    ***Arguments***
    - dim: Number of spatial dimensions to transform over.

    ***Returns***
    - The FFT transform object.

    Example:

    ```
    fft_transform = FFTTransform(dim=2)
    x_hat = fft_transform.forward(x)
    x = fft_transform.inverse(x_hat)
    ```

    """

    def forward(self, x: Array) -> Array:
        """
        Computes the centered FFT.

        ***Arguments***
        - x: Input array of shape (Nx, Ny, ..., d, d)
        ***Returns***
        - x_hat: Transformed array of the same shape
        """

        # Transform only the spatial axes (0 to dim-1)
        axes = range(self.dim)
        return jnp.fft.fftn(x, axes=axes)

    def inverse(self, x_hat: Array) -> Array:
        """
        Computes the inverse centered FFT.

        ***Arguments***
        - x_hat: Input array in frequency space of shape (Nx, Ny, ..., d, d)
        ***Returns***
        - x: Inverse transformed array of the same shape
        """
        axes = range(self.dim)
        return jnp.fft.ifftn(x_hat, axes=axes)

    def get_wavenumber_vector(self, size: int, length: float) -> Array:
        """
        Returns the real-valued wavenumber ξ.

        For an FFT on N points over length L, the wavenumbers are:
        ξ = 2π * [0, 1, ..., N/2-1, -N/2, ..., -1] / L

        ***Arguments***
        - size: Number of points in the spatial dimension.

        - length: Length of the spatial domain.
        ***Returns***
        - k: Real-valued wavenumber vector of shape (size,).

        """

        # Standard FFT frequencies: [0, 1, ..., -N/2, ..., -1]
        freqs = jnp.fft.fftfreq(size, d=length / size)
        return freqs * 2 * jnp.pi


class SlabFFTTransform2D(FFTTransform):
    device_mesh: jax.sharding.Mesh

    def __init__(self, dim: int, device_mesh: jax.sharding.Mesh):
        self.device_mesh = device_mesh
        super().__init__(dim)

    def forward(self, x):
        """Forward 2-D FFT. Input sharded on axis 0, output sharded on axis 1."""

        @jax.shard_map(
            out_specs=P(None, "x"), in_specs=P("x", None), mesh=self.device_mesh
        )
        def local(xl):
            xl = jnp.fft.fft(xl, axis=1)  # axis 1 complete -> local
            xl = jax.lax.all_to_all(xl, "x", split_axis=1, concat_axis=0, tiled=True)
            return jnp.fft.fft(xl, axis=0)  # axis 0 now complete -> local

        return local(x)

    def inverse(self, x_hat):
        """Inverse. Input sharded on axis 1, output sharded on axis 0 — exactly reversed."""

        @jax.shard_map(
            out_specs=P("x", None), in_specs=P(None, "x"), mesh=self.device_mesh
        )
        def local(xl):
            xl = jnp.fft.ifft(xl, axis=0)  # axis 0 complete -> local
            xl = jax.lax.all_to_all(xl, "x", split_axis=0, concat_axis=1, tiled=True)
            return jnp.fft.ifft(xl, axis=1)

        return local(x_hat)
