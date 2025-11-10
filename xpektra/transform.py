import equinox as eqx
from jax import Array
import jax.numpy as jnp
from abc import abstractmethod
from jax.scipy.fft import dctn, idctn


class Transform(eqx.Module):
    """Abstract base class for all spectral transforms."""

    @abstractmethod
    def forward(self, x: Array) -> Array:
        """Perform the forward transform (e.g., FFT, DCT)."""
        raise NotImplementedError

    @abstractmethod
    def inverse(self, x_hat: Array) -> Array:
        """Perform the inverse transform (e.g., iFFT, iDCT)."""
        raise NotImplementedError

    @abstractmethod
    def wavenumber_vector(self) -> Array:
        """Get the 1D vector of wavenumbers (e.g., ξ for FFT, k for DCT)."""
        raise NotImplementedError


class FFTTransform(Transform):
    """The standard, JAX-native Fast Fourier Transform."""

    size: int = eqx.field(static=True)
    dim: int = eqx.field(static=True)
    length: float = eqx.field(static=True)

    @eqx.filter_jit
    def forward(self, x: Array) -> Array:
        axes = range(self.dim)
        return jnp.fft.fftshift(
            jnp.fft.fftn(jnp.fft.ifftshift(x), s=[self.size] * self.dim, axes=axes)
        )

    @eqx.filter_jit
    def backward(self, x_hat: Array) -> Array:
        axes = range(self.dim)
        return jnp.fft.fftshift(
            jnp.fft.ifftn(jnp.fft.ifftshift(x_hat), s=[self.size] * self.dim, axes=axes)
        )

    def wavenumber_vector(self) -> Array:
        """Returns the real-valued wavenumber ξ."""
        freq = (
            jnp.arange(-(self.size - 1) / 2, +(self.size + 1) / 2, dtype="int64")
            / self.length
        )
        return 2 * jnp.pi * freq


class DCTTransform(Transform):
    """
    The Discrete Cosine Transform (Type-II), using JAX-native functions.

    This transform is fully JIT-compatible and is the correct choice
    for problems with Neumann (zero-flux) boundary conditions.
    """

    size: int = eqx.field(static=True)
    dim: int = eqx.field(static=True)
    length: float = eqx.field(static=True)
    norm: str = eqx.field(static=True, default=None)

    @eqx.filter_jit
    def forward(self, x: Array) -> Array:
        """Performs the forward DCT-II."""
        axes = range(self.dim)
        # We use type=2 and norm='ortho' to match the standard
        # definitions used in many physics/math contexts.
        return dctn(x, type=2, norm=self.norm, axes=axes)

    @eqx.filter_jit
    def backward(self, x_hat: Array) -> Array:
        """Performs the backward DCT-II (which is the DCT-III)."""
        axes = range(self.dim)
        return idctn(x_hat, type=2, norm=self.norm, axes=axes)

    def wavenumber_vector(self) -> Array:
        """
        Returns the real-valued wavenumbers 'k' for the DCT-II.

        For a DCT-II on N points, the wavenumbers are k = 0, 1, ..., N-1.
        """
        return jnp.arange(self.size, dtype=jnp.float64)
