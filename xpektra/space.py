from dataclasses import dataclass, field

import jax
import jax.numpy as jnp  # type: ignore
from jax import Array

from xpektra.transform import Transform

__all__ = ["SpectralSpace"]


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class SpectralSpace:
    """Defines the spectral space

    ***Arguments***

    - shape: The shape of the spectral space.
    - lengths: The lengths of the spectral space.
    - transform: The transform to be used in the spectral space.

    ***Returns***
    - The spectral space.

    ```
    space = SpectralSpace(shape=(10,), lengths=(1.0,), transform=FFTTransform(dim=1))
    space.get_wavenumber_vector()
    ```

    """

    lengths: tuple[float, ...] = field(metadata=dict(static=True))
    shape: tuple[int, ...] = field(metadata=dict(static=True))
    transform: Transform = field(metadata=dict(static=True))

    def get_wavenumber_mesh(self) -> list[Array]:
        """
        Creates a list of coordinate arrays for the wavenumbers.

        ***Returns***
        - A list of arrays representing the wavenumber meshgrid.

        """
        return self.transform.get_wavenumber_mesh(self.shape, self.lengths)
