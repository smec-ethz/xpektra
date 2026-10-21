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

import math
from dataclasses import dataclass, field, replace

import jax
from jax import Array

from xpektra.transform import RFFTTransform, Transform

__all__ = ["AxisTuple", "SpectralSpace"]

type Shape = tuple[int, ...]
type Lengths = tuple[float, ...]

_AXIS_NAMES = ("x", "y", "z")


class AxisTuple(tuple):
    """A tuple of axis indices that also supports membership testing by axis name ('x', 'y', 'z')."""

    def __contains__(self, item):
        if isinstance(item, str):
            item_l = item.lower()
            if item_l in _AXIS_NAMES:
                return super().__contains__(_AXIS_NAMES.index(item_l))
            return False
        elif isinstance(item, int):
            return super().__contains__(item)
        return False


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class SpectralSpace:
    """Defines the spectral space domain.

    ***Args***
    - lengths: The lengths of the spatial domain.
    - shape: The physical voxel resolution shape of the domain.
    - transform: The transform to be used in the spectral space.
    - non_periodic_axes: Non-periodic axes (names 'x', 'y', 'z' or indices),
      which are mirrored to 2N on the extended computational grid.
    """

    lengths: Lengths = field(metadata={"static": True})
    shape: Shape = field(metadata={"static": True})
    transform: Transform = field(metadata={"static": True})
    extended_lengths: Lengths = field(metadata={"static": True})
    extended_shape: Shape = field(metadata={"static": True})
    physical_slices: tuple[slice, ...] = field(metadata={"static": True})
    non_periodic_axes: AxisTuple = field(
        metadata={"static": True}, default_factory=AxisTuple
    )

    def __init__(
        self,
        lengths: Lengths,
        shape: Shape,
        transform: Transform,
        non_periodic_axes: tuple[str | int, ...]
        | list[str | int]
        | str
        | int
        | None = None,
        extended_lengths: Lengths | None = None,
        extended_shape: Shape | None = None,
        physical_slices: tuple[slice, ...] | None = None,
    ):
        ndim = len(shape)

        if non_periodic_axes is not None:
            raw_axes = (
                (non_periodic_axes,)
                if isinstance(non_periodic_axes, (str, int))
                else tuple(non_periodic_axes)
            )
            normalized = []
            for ax in raw_axes:
                if isinstance(ax, str):
                    ax_l = ax.lower()
                    if ax_l in _AXIS_NAMES[:ndim]:
                        normalized.append(_AXIS_NAMES.index(ax_l))
                    else:
                        raise ValueError(f"Invalid axis name '{ax}' for {ndim}D space.")
                elif isinstance(ax, int):
                    normalized.append(ax % ndim)
                else:
                    raise TypeError(
                        f"Axis must be str or int, got {type(ax).__name__}."
                    )
            np_axes = AxisTuple(sorted(set(normalized)))
        else:
            np_axes = AxisTuple()

        if extended_shape is not None and extended_lengths is not None:
            ext_shape = tuple(extended_shape)
            ext_lengths = tuple(float(l) for l in extended_lengths)
        else:
            ext_shape = tuple(
                2 * shape[i] if i in np_axes else shape[i] for i in range(ndim)
            )
            ext_lengths = tuple(
                2.0 * float(lengths[i]) if i in np_axes else float(lengths[i])
                for i in range(ndim)
            )

        if physical_slices is not None:
            phys_slices = tuple(physical_slices)
        else:
            phys_slices = tuple(
                slice(0, shape[i]) if i in np_axes else slice(None) for i in range(ndim)
            )

        object.__setattr__(self, "shape", tuple(shape))
        object.__setattr__(self, "lengths", tuple(float(l) for l in lengths))
        object.__setattr__(self, "extended_shape", ext_shape)
        object.__setattr__(self, "extended_lengths", ext_lengths)
        if isinstance(transform, RFFTTransform) and transform.shape is None:
            transform = replace(transform, shape=ext_shape[: transform.dim])
        object.__setattr__(self, "transform", transform)
        object.__setattr__(self, "physical_slices", phys_slices)
        object.__setattr__(self, "non_periodic_axes", np_axes)

    @property
    def dim(self) -> int:
        """Spatial dimension of the domain."""
        return len(self.shape)

    def extend_cell_field(self, field: Array, rank: int = 0) -> Array:
        """Extend a cell field (material property, eigenstrain, stiffness) using DCT-2
        cell-centred reflection with tensor parity signs along non-periodic axes.
        """
        from xpektra.extension import extend_cell_field as _extend_cell_field

        return _extend_cell_field(
            field,
            axes=tuple(self.non_periodic_axes),
            rank=rank,
            spatial_dim=self.dim,
        )

    def get_wavenumber_mesh(self) -> list[Array]:
        """Creates a list of coordinate arrays for the wavenumbers.

        ***Returns***
        - A list of arrays representing the wavenumber meshgrid.
        """
        return self.transform.get_wavenumber_mesh(
            self.extended_shape, self.extended_lengths
        )

    @property
    def cell_volume(self) -> float:
        """The voxel / grid cell volume of the computational domain."""
        return math.prod(
            length / n for length, n in zip(self.extended_lengths, self.extended_shape)
        )
