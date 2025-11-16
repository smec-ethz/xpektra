import equinox as eqx
import jax.numpy as jnp  # type: ignore
from jax import Array

from xpektra.transform import Transform


class SpectralSpace(eqx.Module):
    """Defines the spectral space

    Args:
        shape: The shape of the spectral space.
        lengths: The lengths of the spectral space.
        transform: The transform to be used in the spectral space.

    Returns:
        The spectral space.

    Example:
        >>> space = SpectralSpace(shape=(10,), lengths=(1.0,), transform=FFTTransform(dim=1))
        >>> space.get_wavenumber_vector()
    """

    lengths: tuple[float, ...] = eqx.field(static=True)
    shape: tuple[int, ...] = eqx.field(static=True)
    transform: Transform = eqx.field(static=True)

    def get_wavenumber_mesh(self) -> list[Array]:
        """
        Creates a list of coordinate arrays for the wavenumbers.

        Returns:
            A list of arrays representing the wavenumber meshgrid.
        """
        k_vecs = [
            self.transform.get_wavenumber_vector(size=n, length=length)
            for n, length in zip(self.shape, self.lengths)
        ]
        return list(jnp.meshgrid(*k_vecs, indexing="ij"))
