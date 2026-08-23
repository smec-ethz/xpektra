# Copyright (C) 2025 ETH Zurich (SMEC)
#
# This file is part of xpektra.
#
# xpektra is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# xpektra is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with xpektra.  If not, see <https://www.gnu.org/licenses/>.

from dataclasses import dataclass, field

import jax
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

    lengths: tuple[float, ...] = field(metadata={"static": True})
    shape: tuple[int, ...] = field(metadata={"static": True})
    transform: Transform = field(metadata={"static": True})

    def get_wavenumber_mesh(self) -> list[Array]:
        """
        Creates a list of coordinate arrays for the wavenumbers.

        ***Returns***
        - A list of arrays representing the wavenumber meshgrid.

        """
        return self.transform.get_wavenumber_mesh(self.shape, self.lengths)
