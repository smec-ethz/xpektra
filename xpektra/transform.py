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
    def get_wavenumber_vector(self) -> Array:
        """Get the 1D vector of wavenumbers (e.g., ξ for FFT, k for DCT)."""
        raise NotImplementedError


class FFTTransform(Transform):
    """The standard, JAX-native Fast Fourier Transform."""

    shape: tuple[int, ...]
    dim: int = eqx.field(static=True)

    def __init__(self, shape: tuple[int, ...]):
        self.shape = shape
        self.dim = len(shape)

    @eqx.filter_jit
    def forward(self, x: Array) -> Array:
        """
        Computes the centered FFT.
        
        Args:
            x: Input array of shape (Nx, Ny, ..., d, d)
        Returns:
            x_hat: Transformed array of the same shape
        """
        
        # Transform only the spatial axes (0 to dim-1)
        axes = range(self.dim)
        
        # ifftshift moves zero-freq from center to corners (standard FFT input)
        # fftn performs the transform
        # fftshift moves zero-freq back to center (for visualization/filtering)
        return jnp.fft.fftshift(
            jnp.fft.fftn(
                jnp.fft.ifftshift(x, axes=axes), 
                s=self.shape, 
                axes=axes
            ),
            axes=axes
        )

    @eqx.filter_jit
    def inverse(self, x_hat: Array) -> Array:
        """
        Computes the inverse centered FFT.

        Args:
            x_hat: Input array in frequency space of shape (Nx, Ny, ..., d, d)
        Returns:
            x: Inverse transformed array of the same shape
        """
        axes = range(self.dim)
        return jnp.fft.fftshift(
            jnp.fft.ifftn(
                jnp.fft.ifftshift(x_hat, axes=axes), 
                s=self.shape, 
                axes=axes
            ),
            axes=axes
        )


    def get_wavenumber_vector(self, size, length) -> Array:
        """
        Returns the real-valued wavenumber ξ. 
        
        For an FFT on N points over length L, the wavenumbers are:
        ξ = 2π * [0, 1, ..., N/2-1, -N/2, ..., -1] / L

        Args:
            size: Number of points in the spatial dimension.
            length: Length of the spatial domain.
        Returns:
            k: Real-valued wavenumber vector of shape (size,).

        """

        # Standard FFT frequencies: [0, 1, ..., -N/2, ..., -1]
        freqs = jnp.fft.fftfreq(size, d=length/size)
        
        # Shifted frequencies: [-N/2, ..., 0, ..., N/2]
        # This aligns the k-vector with the data returned by forward()
        k = jnp.fft.fftshift(freqs) * 2 * jnp.pi
        return k


'''
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
'''