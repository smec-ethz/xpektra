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
import jax.numpy as jnp
from jax import Array

from xpektra.transform import Transform

__all__ = ["SpectralSpace"]


def _extend_domain(
    shape: tuple[int, ...],
    lengths: tuple[float, ...],
    axes: dict[int, str | tuple[str, ...]],
) -> tuple[tuple[int, ...], tuple[float, ...]]:
    ndim = len(shape)
    new_shape = list(shape)
    new_lengths = list(lengths)

    for axis, kind in axes.items():
        norm_axis = axis % ndim
        kinds = (kind,) if isinstance(kind, str) else tuple(kind)
        shapes_for_axis = []
        lengths_for_axis = []

        for k in kinds:
            k_str = k.lower()
            n = shape[norm_axis]
            length = lengths[norm_axis]

            if k_str == "dst1":
                shapes_for_axis.append(2 * (n + 1))
                lengths_for_axis.append(2.0 * length)
            elif k_str in ("dct2", "dst2"):
                shapes_for_axis.append(2 * n)
                lengths_for_axis.append(2.0 * length)
            elif k_str == "dct1":
                shapes_for_axis.append(2 * (n - 1))
                lengths_for_axis.append(2.0 * length)
            else:
                raise ValueError(
                    f"Unknown extension type '{k}'. Supported: 'dst1', 'dct2', 'dst2', 'dct1'."
                )

        if len(set(shapes_for_axis)) > 1:
            raise ValueError(
                f"Axis {axis} has incompatible extension shapes for kinds {kinds}: {shapes_for_axis}."
            )

        new_shape[norm_axis] = shapes_for_axis[0]
        new_lengths[norm_axis] = lengths_for_axis[0]

    return tuple(new_shape), tuple(new_lengths)


def _extend_axis_single(res: Array, norm_axis: int, kind: str) -> Array:
    kind = kind.lower()
    if kind == "dst1":
        zeros = jnp.zeros(
            res.shape[:norm_axis] + (1,) + res.shape[norm_axis + 1 :],
            dtype=res.dtype,
        )
        return jnp.concatenate(
            [zeros, res, zeros, -jnp.flip(res, axis=norm_axis)],
            axis=norm_axis,
        )
    elif kind == "dct2":
        return jnp.concatenate(
            [res, jnp.flip(res, axis=norm_axis)],
            axis=norm_axis,
        )
    elif kind == "dst2":
        return jnp.concatenate(
            [res, -jnp.flip(res, axis=norm_axis)],
            axis=norm_axis,
        )
    elif kind == "dct1":
        flipped_interior = jnp.flip(
            jnp.take(res, jnp.arange(1, res.shape[norm_axis] - 1), axis=norm_axis),
            axis=norm_axis,
        )
        return jnp.concatenate([res, flipped_interior], axis=norm_axis)
    else:
        raise ValueError(
            f"Unknown extension type '{kind}'. Supported: 'dst1', 'dct2', 'dst2', 'dct1'."
        )


def _extend_field(
    field: Array,
    axes: dict[int, str | tuple[str, ...]],
    spatial_dim: int | None = None,
) -> Array:
    res = field
    ndim = spatial_dim if spatial_dim is not None else res.ndim
    for axis, kind in axes.items():
        norm_axis = axis % ndim
        if isinstance(kind, (tuple, list)):
            if res.ndim == ndim:
                # Scalar field: use the first specified kind
                res = _extend_axis_single(res, norm_axis, kind[0])
            elif res.ndim == ndim + 1:
                # Vector field: extend each component along norm_axis
                n_comps = res.shape[-1]
                if len(kind) != n_comps:
                    raise ValueError(
                        f"Axis {axis} specifies {len(kind)} extension types, "
                        f"but field has {n_comps} components."
                    )
                comps = [
                    _extend_axis_single(res[..., c], norm_axis, kind[c])
                    for c in range(n_comps)
                ]
                res = jnp.stack(comps, axis=-1)
            else:
                raise ValueError(
                    f"Field with {res.ndim} dimensions cannot be extended with "
                    f"per-component rules for {ndim} spatial dimensions."
                )
        else:
            res = _extend_axis_single(res, norm_axis, kind)
    return res


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class SpectralSpace:
    """Defines the spectral space.

    ***Arguments***
    - lengths: The lengths of the spatial domain.
    - shape: The grid resolution shape of the spatial domain.
    - transform: The transform to be used in the spectral space.
    - extend_axes: Optional mapping from axis to extension type, e.g. `{1: "dst1"}`
      or per-component tuple `{1: ("dct2", "dst2")}` for vector fields.
      If provided, `shape` and `lengths` define the physical domain, and the
      periodized computational domain is computed automatically.

    ***Attributes***
    - shape: The computational (periodized) grid shape used by operators and transforms.
    - lengths: The computational (periodized) domain lengths.
    - physical_shape: The physical interior domain shape before extension.
    - physical_lengths: The physical domain lengths before extension.
    - extend_axes: The mapping from axis to extension type.

    ***Methods***
    - extend_field(field, axes=None): Extends an array field. If `axes` is None,
      defaults to `self.extend_axes`.
    - get_wavenumber_mesh(): Returns the wavenumber meshgrid for the space.
    """

    lengths: tuple[float, ...] = field(metadata={"static": True})
    shape: tuple[int, ...] = field(metadata={"static": True})
    transform: Transform = field(metadata={"static": True})
    physical_lengths: tuple[float, ...] = field(metadata={"static": True})
    physical_shape: tuple[int, ...] = field(metadata={"static": True})
    extended_lengths: tuple[float, ...] = field(metadata={"static": True})
    extended_shape: tuple[int, ...] = field(metadata={"static": True})
    physical_slices: tuple[slice, ...] = field(metadata={"static": True})
    extend_axes: dict[int, str | tuple[str, ...]] = field(
        metadata={"static": True}, default_factory=dict
    )

    def __init__(
        self,
        lengths: tuple[float, ...],
        shape: tuple[int, ...],
        transform: Transform,
        extend_axes: dict[int, str | tuple[str, ...]] | None = None,
        physical_lengths: tuple[float, ...] | None = None,
        physical_shape: tuple[int, ...] | None = None,
    ):
        phys_shape = tuple(shape) if physical_shape is None else tuple(physical_shape)
        phys_lengths = (
            tuple(float(l) for l in lengths)
            if physical_lengths is None
            else tuple(float(l) for l in physical_lengths)
        )
        ext_axes = {} if extend_axes is None else dict(extend_axes)

        if ext_axes and (physical_shape is None or physical_lengths is None):
            comp_shape, comp_lengths = _extend_domain(
                phys_shape, phys_lengths, ext_axes
            )
        else:
            comp_shape = tuple(shape)
            comp_lengths = tuple(float(l) for l in lengths)

        ndim = len(phys_shape)
        ext_set = {ax % ndim for ax in ext_axes}
        phys_slices = tuple(
            slice(0, comp_shape[i] // 2) if i % ndim in ext_set else slice(None)
            for i in range(ndim)
        )

        object.__setattr__(self, "shape", comp_shape)
        object.__setattr__(self, "lengths", comp_lengths)
        object.__setattr__(self, "extended_shape", comp_shape)
        object.__setattr__(self, "extended_lengths", comp_lengths)
        object.__setattr__(self, "transform", transform)
        object.__setattr__(self, "physical_shape", phys_shape)
        object.__setattr__(self, "physical_lengths", phys_lengths)
        object.__setattr__(self, "physical_slices", phys_slices)
        object.__setattr__(self, "extend_axes", ext_axes)

    def extend_field(
        self, field: Array, axes: dict[int, str | tuple[str, ...]] | None = None
    ) -> Array:
        """Extend an array field along specified axes according to reflection symmetries.

        If `axes` is None, uses `self.extend_axes`. Supported types:
        - 'dst1': DST-I odd reflection with boundary zeros, length becomes 2*(N+1).
        - 'dct2': DCT-II even cell-boundary reflection, length becomes 2*N.
        - 'dst2': DST-II odd cell-boundary reflection, length becomes 2*N.
        - 'dct1': DCT-I even node-sharing reflection, length becomes 2*(N-1).
        """
        target_axes = self.extend_axes if axes is None else axes
        if not target_axes:
            return field
        return _extend_field(
            field, axes=target_axes, spatial_dim=len(self.physical_lengths)
        )

    def get_wavenumber_mesh(self) -> list[Array]:
        """Creates a list of coordinate arrays for the wavenumbers.

        ***Returns***
        - A list of arrays representing the wavenumber meshgrid.
        """
        return self.transform.get_wavenumber_mesh(self.shape, self.lengths)
