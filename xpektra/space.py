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
import jax.numpy as jnp
from jax import Array

from xpektra.transform import RFFTTransform, Transform

__all__ = ["SpectralSpace"]

type Shape = tuple[int, ...]
type Lengths = tuple[float, ...]
type AxisExtensionKind = str | tuple[str, ...]
type AxisExtensionMap = dict[int, AxisExtensionKind]

_SUPPORTED_EXTENSIONS = ("dst1", "dct2", "dst2", "dct1")
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


def _extend_domain(
    shape: Shape,
    lengths: Lengths,
    axes: AxisExtensionMap,
) -> tuple[Shape, Lengths]:
    """Compute the extended domain shape and lengths based on specified axes and extension types.

    ***Args***
    - shape: The original shape of the domain.
    - lengths: The original lengths of the domain.
    - axes: A dictionary mapping axis indices to extension types (e.g., 'dst1', 'dct2', 'dst2', 'dct1').

    ***Returns***
    - A tuple containing the new shape and new lengths after applying the specified extensions.

    """
    ndim = len(shape)
    new_shape = list(shape)
    new_lengths = list(lengths)

    for axis, kind in axes.items():
        norm_axis = axis % ndim
        k = kind if isinstance(kind, str) else kind[0]
        k_str = k.lower()
        n = shape[norm_axis]
        length = lengths[norm_axis]

        if k_str == "dst1":
            new_shape[norm_axis] = 2 * (n + 1)
            new_lengths[norm_axis] = 2.0 * length
        elif k_str in ("dct2", "dst2"):
            new_shape[norm_axis] = 2 * n
            new_lengths[norm_axis] = 2.0 * length
        elif k_str == "dct1":
            new_shape[norm_axis] = 2 * (n - 1)
            new_lengths[norm_axis] = 2.0 * length
        else:
            raise ValueError(
                f"Unknown extension type '{k}'. Supported: {', '.join(f"'{s}'" for s in _SUPPORTED_EXTENSIONS)}."
            )

    return tuple(new_shape), tuple(new_lengths)


def _extend_axis_single(res: Array, norm_axis: int, kind: str) -> Array:
    kind = kind.lower()
    if kind == "dst1":
        sl_zero = [slice(None)] * res.ndim
        sl_zero[norm_axis] = slice(0, 1)
        zeros = jnp.zeros_like(res[tuple(sl_zero)])
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
            f"Unknown extension type '{kind}'. Supported: {', '.join(f"'{s}'" for s in _SUPPORTED_EXTENSIONS)}."
        )


def _extend_field(
    field: Array,
    axes: AxisExtensionMap,
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


def _restrict_axis_single(res: Array, norm_axis: int, kind: str) -> Array:
    kind = kind.lower()
    axis_len = res.shape[norm_axis]
    if kind == "dst1":
        # Extended length is 2*(N+1) = 2N + 2. Physical interior length is N = axis_len // 2 - 1.
        n = axis_len // 2 - 1
        sl_pos = [slice(None)] * res.ndim
        sl_pos[norm_axis] = slice(1, n + 1)
        sl_neg = [slice(None)] * res.ndim
        sl_neg[norm_axis] = slice(n + 2, 2 * n + 2)
        return res[tuple(sl_pos)] - jnp.flip(res[tuple(sl_neg)], axis=norm_axis)
    elif kind == "dct2":
        # Extended length is 2*N. Physical length is N = axis_len // 2.
        n = axis_len // 2
        sl_pos = [slice(None)] * res.ndim
        sl_pos[norm_axis] = slice(0, n)
        sl_neg = [slice(None)] * res.ndim
        sl_neg[norm_axis] = slice(n, 2 * n)
        return res[tuple(sl_pos)] + jnp.flip(res[tuple(sl_neg)], axis=norm_axis)
    elif kind == "dst2":
        # Extended length is 2*N. Physical length is N = axis_len // 2.
        n = axis_len // 2
        sl_pos = [slice(None)] * res.ndim
        sl_pos[norm_axis] = slice(0, n)
        sl_neg = [slice(None)] * res.ndim
        sl_neg[norm_axis] = slice(n, 2 * n)
        return res[tuple(sl_pos)] - jnp.flip(res[tuple(sl_neg)], axis=norm_axis)
    elif kind == "dct1":
        # Extended length is 2*(N-1) = 2N - 2. Physical length is N = axis_len // 2 + 1.
        n = axis_len // 2 + 1
        sl_pos = [slice(None)] * res.ndim
        sl_pos[norm_axis] = slice(0, n)
        r_first = res[tuple(sl_pos)]
        sl_neg = [slice(None)] * res.ndim
        sl_neg[norm_axis] = slice(n, 2 * n - 2)
        r_mirrored = jnp.flip(res[tuple(sl_neg)], axis=norm_axis)
        sl_interior = [slice(None)] * res.ndim
        sl_interior[norm_axis] = slice(1, n - 1)
        return r_first.at[tuple(sl_interior)].add(r_mirrored)
    else:
        raise ValueError(
            f"Unknown extension type '{kind}'. Supported: {', '.join(f"'{s}'" for s in _SUPPORTED_EXTENSIONS)}."
        )


def _restrict_field(
    field: Array,
    axes: AxisExtensionMap,
    spatial_dim: int | None = None,
) -> Array:
    res = field
    ndim = spatial_dim if spatial_dim is not None else res.ndim
    for axis, kind in axes.items():
        norm_axis = axis % ndim
        if isinstance(kind, (tuple, list)):
            if res.ndim > ndim:
                n_comps = res.shape[-1]
                if len(kind) != n_comps:
                    raise ValueError(
                        f"Axis {axis} has {len(kind)} extension rules, "
                        f"but field has {n_comps} components."
                    )
                comps = [
                    _restrict_axis_single(res[..., c], norm_axis, k)
                    for c, k in enumerate(kind)
                ]
                res = jnp.stack(comps, axis=-1)
            else:
                raise ValueError(
                    f"Field with {res.ndim} dimensions cannot be restricted with "
                    f"per-component rules for {ndim} spatial dimensions."
                )
        else:
            res = _restrict_axis_single(res, norm_axis, kind)
    return res


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class SpectralSpace:
    """Defines the spectral space.

    ***Args***
    - lengths: The lengths of the spatial domain.
    - shape: The grid resolution shape of the spatial domain.
    - transform: The transform to be used in the spectral space.
    - extend_axes: Optional mapping from axis to extension type, e.g. `{1: "dst1"}`
      or per-component tuple `{1: ("dct2", "dst2")}` for vector fields.
      If provided, `shape` and `lengths` define the physical domain, and the
      periodized computational domain is computed automatically.

    """

    lengths: Lengths = field(metadata={"static": True})
    shape: Shape = field(metadata={"static": True})
    transform: Transform = field(metadata={"static": True})
    extended_lengths: Lengths = field(metadata={"static": True})
    extended_shape: Shape = field(metadata={"static": True})
    physical_slices: tuple[slice, ...] = field(metadata={"static": True})
    extend_axes: AxisExtensionMap = field(
        metadata={"static": True}, default_factory=dict
    )

    def __init__(
        self,
        lengths: Lengths,
        shape: Shape,
        transform: Transform,
        extend_axes: AxisExtensionMap | None = None,
        extended_lengths: Lengths | None = None,
        extended_shape: Shape | None = None,
        physical_slices: tuple[slice, ...] | None = None,
    ):
        ext_axes = {} if extend_axes is None else dict(extend_axes)

        if extended_shape is not None and extended_lengths is not None:
            ext_shape = tuple(extended_shape)
            ext_lengths = tuple(float(l) for l in extended_lengths)
        elif ext_axes:
            ext_shape, ext_lengths = _extend_domain(shape, lengths, ext_axes)
        else:
            ext_shape = tuple(shape)
            ext_lengths = tuple(float(l) for l in lengths)

        ndim = len(shape)
        if physical_slices is not None:
            phys_slices = tuple(physical_slices)
        else:
            ext_set = {ax % ndim for ax in ext_axes}
            phys_slices = tuple(
                slice(0, ext_shape[i] // 2) if i in ext_set else slice(None)
                for i in range(ndim)
            )

        object.__setattr__(self, "shape", shape)
        object.__setattr__(self, "lengths", tuple(float(l) for l in lengths))
        object.__setattr__(self, "extended_shape", ext_shape)
        object.__setattr__(self, "extended_lengths", ext_lengths)
        if isinstance(transform, RFFTTransform) and transform.shape is None:
            transform = replace(transform, shape=ext_shape[: transform.dim])
        object.__setattr__(self, "transform", transform)
        object.__setattr__(self, "physical_slices", phys_slices)
        object.__setattr__(self, "extend_axes", ext_axes)

    def extend_primal(
        self, field: Array, axes: AxisExtensionMap | None = None
    ) -> Array:
        """Extend a primal field (e.g. displacement u) along specified axes according to boundary symmetries.

        If `axes` is None, uses `self.extend_axes`. Supported types:
        - 'dst1': DST-I odd reflection with boundary zeros, length becomes 2*(N+1).
        - 'dct2': DCT-II even cell-boundary reflection, length becomes 2*N.
        - 'dst2': DST-II odd cell-boundary reflection, length becomes 2*N.
        - 'dct1': DCT-I even node-sharing reflection, length becomes 2*(N-1).
        """
        target_axes = self.extend_axes if axes is None else axes
        if not target_axes:
            return field
        return _extend_field(field, axes=target_axes, spatial_dim=len(self.lengths))

    def restrict_dual(
        self, field: Array, axes: AxisExtensionMap | None = None
    ) -> Array:
        """Restrict an extended dual field (e.g. force or residual) back to the physical domain.

        Applies the exact mathematical adjoint E^T of the extension operator E.
        If `axes` is None, uses `self.extend_axes`.
        """
        target_axes = self.extend_axes if axes is None else axes
        if not target_axes:
            return field
        return _restrict_field(field, axes=target_axes, spatial_dim=len(self.lengths))

    def extend_material_field(
        self,
        field: Array,
        axes: tuple[int, ...] | list[int] | dict[int, str] | None = None,
    ) -> Array:
        """Extend a material field (e.g. lam, mu) using even (DCT-2) symmetry along extended axes.

        By default, applies 'dct2' cell-boundary reflection along all axes present in `self.extend_axes`.
        """
        if axes is None:
            target_axes = {ax: "dct2" for ax in self.extend_axes}
        elif isinstance(axes, dict):
            target_axes = axes
        else:
            target_axes = {ax: "dct2" for ax in axes}

        if not target_axes:
            return field
        return _extend_field(field, axes=target_axes, spatial_dim=len(self.lengths))

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
