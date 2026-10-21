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
from dataclasses import dataclass
from functools import cached_property
from typing import Literal, cast, get_args

import jax
import jax.numpy as jnp
from jax import Array
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P

from xpektra.space import SpectralSpace
from xpektra.extension import (
    ExtensionKind,
    extend_1d,
    get_extension_kind,
    mask_1d,
    restrict_1d,
    weights_1d,
)

type BCType = Literal["periodic", "dirichlet", "neumann"]
type FacetPair = tuple[BCType, BCType]
type FaceTMap = dict[str, BCType]
type FacetBC = BCType | FacetPair | FaceTMap
type ComponentBC = dict[int, FacetBC]
type AxisBC = FacetBC | ComponentBC | None

_VALID_BCS = frozenset(get_args(BCType.__value__))
PERIODIC_PAIR: FacetPair = ("periodic", "periodic")


def _check_bc_value(bc: str) -> BCType:
    """Check if the boundary condition value is valid.

    Args:
        bc (str): The boundary condition value.

    Returns:
        BCType: The boundary condition value if valid.
    """
    if not isinstance(bc, str):
        raise TypeError(
            f"Boundary condition must be a string, got {type(bc).__name__}."
        )
    norm = bc.strip().lower()
    if norm not in _VALID_BCS:
        raise ValueError(
            f"Invalid boundary condition '{bc}'. Valid options are: {_VALID_BCS}."
        )

    return cast(BCType, norm)


def _check_mix_bc(min_bc: str | BCType, max_bc: str | BCType) -> None:
    """Check if the boundary conditions for a face are valid.

    Args:
        min_bc (BCType): The boundary condition for the minimum side of the face.
        max_bc (BCType): The boundary condition for the maximum side of the face.

    Raises:
        ValueError: If the boundary conditions are not valid.
    """
    if (min_bc == "periodic") != (max_bc == "periodic"):
        raise ValueError(
            f"Cannot mix periodic and non-periodic boundary conditions on the same face, got min={min_bc} and max={max_bc}."
        )


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class Field:
    """A field in the solution space.

    Attributes:
        shape (tuple[int, ...]): The shape of the field.
        x (BoundaryCondition): The boundary condition in the x direction. Default is None which means periodic boundary condition.
        y (BoundaryCondition): The boundary condition in the y direction. Default is None which means periodic boundary condition.
        z (BoundaryCondition): The boundary condition in the z direction. Default is None, which means periodic boundary condition.
        name (str): The name of the field.

    """

    shape: tuple[int, ...] = jax.tree.static()
    x: AxisBC = jax.tree.static(default=None)
    y: AxisBC = jax.tree.static(default=None)
    z: AxisBC = jax.tree.static(default=None)
    name: str = jax.tree.static(default="field")

    def __post_init__(self):
        """Post-initialization checks for the Field class."""
        self._check_field_shape()
        self._check_field_bcs()

    def _check_field_bcs(self) -> None:
        """Validate boundary conditions and enforce v1 restrictions."""
        # Trigger parsing to validate all axis BCs
        parsed_bcs = self.bcs

        # v1 Restriction (§6.5): Non-periodic BCs on tensor fields (rank > 1) -> error
        if self.rank > 1:
            for comp_pairs in parsed_bcs.values():
                for min_bc, max_bc in comp_pairs:
                    if min_bc != "periodic" or max_bc != "periodic":
                        raise ValueError(
                            f"Non-periodic boundary conditions are not supported for "
                            f"tensor fields (rank={self.rank}) in v1."
                        )

    def _check_field_shape(self) -> None:
        """Check if the shape of the field is valid.

        Args:
            shape (tuple[int, ...]): The shape of the field.

        Raises:
            ValueError: If the shape is not valid.
        """
        if not isinstance(self.shape, tuple):
            raise TypeError("Shape must be a tuple.")

        if not all(isinstance(s, int) and s > 0 for s in self.shape):
            raise ValueError(
                f"All shape dimensions must be positive integers, got {self.shape}."
            )

        if self.shape == (1,):
            raise ValueError("For scalar fields, use shape=() instead of shape=(1,).")

    @property
    def n_components(self) -> int:
        return math.prod(self.shape) if self.shape else 1

    @property
    def rank(self) -> int:
        return len(self.shape)

    @property
    def bcs(self) -> dict[str, tuple[FacetPair, ...]]:
        """Canonical (min, max) boundary conditions for all axes and components."""
        return {
            "x": self._parse_axis_bc(self.x),
            "y": self._parse_axis_bc(self.y),
            "z": self._parse_axis_bc(self.z),
        }

    @property
    def non_periodic_axes(self) -> tuple[str, ...]:
        """Sorted tuple of axis indices that have non-periodic boundary conditions."""
        return tuple(
            sorted(
                ax
                for ax, comp_pairs in self.bcs.items()
                if any(
                    min_bc != "periodic" or max_bc != "periodic"
                    for min_bc, max_bc in comp_pairs
                )
            )
        )

    def _parse_facet_bc(self, facet_bc: FacetBC) -> tuple[BCType, BCType]:
        """Parse the boundary condition for a face.

        Args:
            facet_bc (FacetBC): The boundary condition for the face.

        Returns:
            tuple[BCType, BCType]: The parsed boundary condition for the face.
        """

        if isinstance(facet_bc, str):
            bc = _check_bc_value(facet_bc)
            return bc, bc

        elif isinstance(facet_bc, (tuple, list)):
            if len(facet_bc) != 2:
                raise ValueError(
                    f"Face boundary condition must have 2 (min, max), got {facet_bc}."
                )
            bc1 = _check_bc_value(facet_bc[0])
            bc2 = _check_bc_value(facet_bc[1])
            _check_mix_bc(bc1, bc2)
            return bc1, bc2

        elif isinstance(facet_bc, dict):
            if "min" not in facet_bc or "max" not in facet_bc:
                raise ValueError(
                    f"Face boundary condition dict must have 'min' and 'max' keys, got {list(facet_bc.keys())}."
                )
            min_bc = _check_bc_value(facet_bc["min"])
            max_bc = _check_bc_value(facet_bc["max"])
            _check_mix_bc(min_bc, max_bc)
            return min_bc, max_bc

        else:
            raise TypeError(
                f"Invalid FacetBC type {type(facet_bc).__name__} : {facet_bc}."
            )

    def _parse_axis_bc(self, axis_bc: AxisBC) -> tuple[FacetPair, ...]:
        """Parse the boundary condition for an axis.

        Args:
            axis_bc (AxisBC): The boundary condition for the axis.

        Returns:
            tuple[BCType, BCType]: The parsed boundary condition for the axis.
        """
        # check if axis_bc is None, which means periodic boundary condition
        if axis_bc is None:
            return (PERIODIC_PAIR,) * self.n_components

        # check if axis_bc is a FacetBC
        if isinstance(axis_bc, (str, tuple, list)) or (
            isinstance(axis_bc, dict) and ("min" in axis_bc or "max" in axis_bc)
        ):
            pair = self._parse_facet_bc(cast(FacetPair, axis_bc))
            return (pair,) * self.n_components

        # check if axis_bc is a ComponentBC
        comp_bcs: list[FacetPair] = [PERIODIC_PAIR] * self.n_components
        for key, value in axis_bc.items():
            if isinstance(key, int):
                if key < 0 or key >= self.n_components:
                    raise ValueError(
                        f"Component index {key} is out of bounds for field with {self.n_components} components."
                    )
                comp_bcs[key] = self._parse_facet_bc(value)
            else:
                raise TypeError(
                    f"Invalid component key type {type(key).__name__}, must be int or str."
                )

        return tuple(comp_bcs)

    def get_axis_bcs(self, axis: str) -> tuple[FacetPair, ...]:
        """Return canonical (min, max) boundary conditions for all components along an axis."""
        if axis not in self.bcs:
            raise ValueError(
                f"Invalid axis '{axis}', must be one of {list(self.bcs.keys())}."
            )
        return self.bcs.get(axis, (PERIODIC_PAIR,) * self.n_components)

    def get_bc(self, axis: str, component: int = 0) -> FacetPair:
        """Return (min, max) boundary condition for a specific axis and component."""
        axis_bcs = self.get_axis_bcs(axis)
        if component < 0 or component >= self.n_components:
            raise ValueError(
                f"Component index {component} out of range (0..{self.n_components - 1})."
            )
        return axis_bcs[component]


@jax.tree_util.register_pytree_node_class
class State:
    _RESERVED = {"_dofs"}

    def __init__(self, **dofs: Array):
        """Initialize the state with fields.

        Args:
            **dofs (Array): The dofs associated with the state.
        """
        if bad := self._RESERVED & dofs.keys():
            raise ValueError(
                f"Field names {bad} are reserved and cannot be used as field names."
            )
        object.__setattr__(self, "_dofs", dofs)

    def __getattr__(self, name: str) -> Array:
        try:
            return self._dofs[name]
        except KeyError:
            raise AttributeError(f"Dof '{name}' not found in state.")

    def __iter__(self):
        return iter(self._dofs.values())

    def __len__(self):
        return len(self._dofs)

    def tree_flatten(self):
        return tuple(self._dofs.values()), tuple(self._dofs.keys())

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(**dict(zip(aux_data, children)))


@dataclass(frozen=True)
class FieldView:
    """View of a single field within the solution space."""

    name: str
    field: Field
    channel_slice: slice
    node_shape: tuple[int, ...]
    n_free: int

    @property
    def shape(self) -> tuple[int, ...]:
        """Physical shape of this field: (*node_shape, *field.shape)."""
        return self.node_shape + self.field.shape


class SolutionSpace:
    """A solution space for one or multiple fields.

    Handles storage layout, constraint masking, extension, dual restriction,
    and preconditioning for spectral solvers.

    Attributes:
        space (SpectralSpace): The underlying domain resolution and mirrored axes.
        fields (dict[str, Field]): Declared fields.
        node_shape (tuple[int, ...]): Spatial node grid shape (N+1 on mirrored axes, N on periodic).
        shape (tuple[int, ...]): Solution array shape (*node_shape, C).
        n_free (int): Number of unconstrained degrees of freedom.
    """

    _RESERVED = {
        "space",
        "fields",
        "node_shape",
        "shape",
        "mask",
        "n_free",
        "zeros",
        "apply_mask",
        "primal_to_extended",
        "dual_to_physical",
        "primal_to_physical",
        "preconditioner",
        "_field_views",
        "_field_slices",
        "_axis_masks",
    }

    def __init__(self, space: SpectralSpace, **fields: Field):
        if not fields:
            raise ValueError("SolutionSpace requires at least one Field.")

        if bad := self._RESERVED & fields.keys():
            raise ValueError(
                f"Field names {bad} are reserved and cannot be used in SolutionSpace."
            )

        ndim = space.dim
        axis_names = ("x", "y", "z")

        # Validation Rule 1 (§6.5): Field non-periodic BCs must belong to space.non_periodic_axes
        for name, field in fields.items():
            for ax_name in field.non_periodic_axes:
                if ax_name not in space.non_periodic_axes:
                    raise ValueError(
                        f"Field '{name}' specifies non-periodic BCs along axis '{ax_name}', "
                        f"but '{ax_name}' is not in space.non_periodic_axes ({space.non_periodic_axes})."
                    )

        # Validation Rule 5 (§6.5): Sharded axes cannot be non-periodic in v1
        if space.transform.device_mesh is not None:
            phys_spec = getattr(space.transform, "physical_spec", None)
            if phys_spec is not None:
                for i, spec in enumerate(phys_spec):
                    if spec is not None and i in space.non_periodic_axes:
                        raise ValueError(
                            f"Sharding a non-periodic axis ({i}) is not supported in v1."
                        )

        # Compute node_shape: N+1 for non-periodic axes, N for periodic axes
        node_shape = tuple(
            space.shape[i] + 1 if i in space.non_periodic_axes else space.shape[i]
            for i in range(ndim)
        )

        # Channel layout & slices
        field_slices: dict[str, slice] = {}
        curr_channel = 0
        for name, field in fields.items():
            n_c = field.n_components
            field_slices[name] = slice(curr_channel, curr_channel + n_c)
            curr_channel += n_c

        total_channels = curr_channel
        full_shape = node_shape + (total_channels,)

        # Build lightweight broadcastable 1D masks and inv_weights per non-periodic axis
        axis_masks: dict[int, Array] = {}
        axis_inv_weights: dict[int, Array] = {}
        for ax_idx in space.non_periodic_axes:
            ax_name = axis_names[ax_idx]
            n_nodes = node_shape[ax_idx]
            comp_masks_1d = []
            comp_inv_w_1d = []
            for name, field in fields.items():
                for comp_idx in range(field.n_components):
                    bc = field.get_bc(ax_name, comp_idx)
                    comp_masks_1d.append(mask_1d(n_nodes, bc))
                    w = weights_1d(n_nodes, bc)
                    inv_w = jnp.where(w > 0, 1.0 / w, 0.0)
                    comp_inv_w_1d.append(inv_w)
            stacked_m = jnp.stack(comp_masks_1d, axis=-1)
            stacked_w = jnp.stack(comp_inv_w_1d, axis=-1)
            unsqueeze_shape = [1] * (ndim + 1)
            unsqueeze_shape[ax_idx] = n_nodes
            unsqueeze_shape[-1] = total_channels
            axis_masks[ax_idx] = stacked_m.reshape(tuple(unsqueeze_shape))
            axis_inv_weights[ax_idx] = stacked_w.reshape(tuple(unsqueeze_shape))

        # Compute n_free via separability without materializing full 3D mask
        field_views: dict[str, FieldView] = {}
        total_n_free = 0

        for name, field in fields.items():
            field_n_free = 0
            for comp_idx in range(field.n_components):
                comp_free = 1
                for i in range(ndim):
                    if i in space.non_periodic_axes:
                        bc = field.get_bc(axis_names[i], comp_idx)
                        m_1d = mask_1d(node_shape[i], bc)
                        comp_free *= int(jnp.sum(m_1d))
                    else:
                        comp_free *= space.shape[i]
                field_n_free += comp_free

            sl = field_slices[name]
            view = FieldView(
                name=name,
                field=field,
                channel_slice=sl,
                node_shape=node_shape,
                n_free=field_n_free,
            )
            field_views[name] = view
            setattr(self, name, view)
            total_n_free += field_n_free

        self.space = space
        self.fields = fields
        self.node_shape = node_shape
        self.shape = full_shape
        self.n_free = total_n_free
        self._field_slices = field_slices
        self._field_views = field_views
        self._axis_masks = axis_masks
        self._axis_inv_weights = axis_inv_weights

    def apply_mask(self, u: Array) -> Array:
        """Apply constraint mask to solution array u with zero communication."""
        for ax_mask in self._axis_masks.values():
            u = u * ax_mask
        return u

    def apply_inv_weights(self, u: Array) -> Array:
        """Apply W⁻¹ = diag(Eᵀ E)⁻¹ to solution array u with zero communication."""
        for ax_inv_w in self._axis_inv_weights.values():
            u = u * ax_inv_w
        return u

    @property
    def mask(self) -> Array:
        """On-demand construction of the mask array (shape V.shape)."""
        sharding = getattr(self.space.transform, "physical_spec", None)
        mesh = getattr(self.space.transform, "device_mesh", None)

        if mesh is not None and sharding is not None:
            full_spec = NamedSharding(mesh, P(*sharding, None))
            full_mask = jnp.ones(self.shape, dtype=bool)
            full_mask = jax.device_put(full_mask, full_spec)
        else:
            full_mask = jnp.ones(self.shape, dtype=bool)

        for ax_mask in self._axis_masks.values():
            full_mask = full_mask & ax_mask
        return full_mask

    def __getitem__(self, name: str) -> FieldView:
        return self._field_views[name]

    def __contains__(self, name: str) -> bool:
        return name in self._field_views

    def __len__(self) -> int:
        return len(self.fields)

    def __iter__(self):
        return iter(self._field_views.values())

    def zeros(self, sharding: NamedSharding | None = None) -> Array:
        """Return a solution array initialized to zero of shape V.shape, optionally sharded."""
        arr = jnp.zeros(self.shape, dtype=float)
        if sharding is not None:
            arr = jax.device_put(arr, sharding)
        return arr

    def primal_to_extended(self, u: Array) -> State:
        """Convert primal solution degrees of freedom to extended state (E).

        Applies mask, then extends along non-periodic axes with boundary reflection.

        Args:
            u (Array): Primal solution array of shape (*node_shape, total_channels).

        Returns:
            State: Extended state where each field has shape (*extended_shape, *field.shape).
        """
        u_masked = self.apply_mask(u)
        axis_names = ("x", "y", "z")
        ext_fields: dict[str, Array] = {}

        for name, field in self.fields.items():
            sl = self._field_slices[name]
            field_arr = u_masked[..., sl]

            comp_arrays = []
            for comp_idx in range(field.n_components):
                comp_arr = field_arr[..., comp_idx]
                for ax_idx in self.space.non_periodic_axes:
                    ax_name = axis_names[ax_idx]
                    bc = field.get_bc(ax_name, comp_idx)
                    kind = get_extension_kind(bc)
                    comp_arr = extend_1d(comp_arr, kind=kind, axis=ax_idx)
                comp_arrays.append(comp_arr)

            if field.shape == ():
                # Scalar field: (*extended_shape,) with no trailing 1
                field_ext = comp_arrays[0]
            else:
                field_ext = jnp.stack(comp_arrays, axis=-1)
                if len(field.shape) > 1:
                    field_ext = field_ext.reshape(self.space.extended_shape + field.shape)

            ext_fields[name] = field_ext

        return State(**ext_fields)

    def dual_to_physical(self, *fields: Array | State, **kwargs: Array) -> Array:
        """Convert extended dual forces back to physical solution array (Eᵀ).

        Restricts extended dual fields along non-periodic axes, stacks them into
        V.shape, and applies the constraint mask.

        Args:
            *fields: A State instance, or positional extended arrays matching field order.
            **kwargs: Extended arrays passed by field name.

        Returns:
            Array: Physical dual force array of shape (*node_shape, total_channels).
        """
        if len(fields) == 1 and isinstance(fields[0], State):
            ext_dict = dict(fields[0]._dofs)
        elif len(fields) == 1 and isinstance(fields[0], dict):
            ext_dict = fields[0]
        elif fields:
            ext_dict = dict(zip(self.fields.keys(), fields))
        elif kwargs:
            ext_dict = kwargs
        else:
            raise ValueError("dual_to_physical requires dual fields to be passed.")

        missing = set(self.fields.keys()) - set(ext_dict.keys())
        if missing:
            raise ValueError(f"Missing dual fields: {missing}")

        ndim = self.space.dim
        axis_names = ("x", "y", "z")
        all_restricted_comps: list[Array] = []

        for name, field in self.fields.items():
            f_ext = ext_dict[name]

            if f_ext.ndim == 1:
                f_ext = f_ext.reshape(self.space.extended_shape + field.shape)

            if field.shape == ():
                if f_ext.ndim == ndim + 1 and f_ext.shape[-1] == 1:
                    f_ext = f_ext[..., 0]

            for comp_idx in range(field.n_components):
                if field.shape == ():
                    comp_f = f_ext
                elif field.rank == 1:
                    comp_f = f_ext[..., comp_idx]
                else:
                    comp_f = f_ext.reshape(self.space.extended_shape + (-1,))[..., comp_idx]

                for ax_idx in reversed(self.space.non_periodic_axes):
                    ax_name = axis_names[ax_idx]
                    bc = field.get_bc(ax_name, comp_idx)
                    kind = get_extension_kind(bc)
                    comp_f = restrict_1d(comp_f, kind=kind, axis=ax_idx)

                all_restricted_comps.append(comp_f)

        f_stacked = jnp.stack(all_restricted_comps, axis=-1)
        return self.apply_mask(f_stacked)

    def primal_to_physical(self, u: Array) -> State:
        """Convert primal solution array to physical fields (State).

        Fills periodic wrap nodes (u_N := u_0) along doubled axes and splits into fields.

        Args:
            u (Array): Primal solution array of shape (*node_shape, total_channels).

        Returns:
            State: Physical state where each field has shape (*node_shape, *field.shape).
        """
        u_masked = self.apply_mask(u)
        axis_names = ("x", "y", "z")
        phys_fields: dict[str, Array] = {}

        for name, field in self.fields.items():
            sl = self._field_slices[name]
            field_arr = u_masked[..., sl]

            comp_arrays = []
            for comp_idx in range(field.n_components):
                comp_arr = field_arr[..., comp_idx]
                for ax_idx in self.space.non_periodic_axes:
                    ax_name = axis_names[ax_idx]
                    bc = field.get_bc(ax_name, comp_idx)
                    if bc == ("periodic", "periodic"):
                        # Fill periodic wrap duplicate: u_N := u_0
                        sl_last = [slice(None)] * comp_arr.ndim
                        sl_last[ax_idx] = slice(-1, None)
                        sl_first = [slice(None)] * comp_arr.ndim
                        sl_first[ax_idx] = slice(0, 1)
                        comp_arr = comp_arr.at[tuple(sl_last)].set(comp_arr[tuple(sl_first)])
                comp_arrays.append(comp_arr)

            if field.shape == ():
                phys_field = comp_arrays[0]
            else:
                phys_field = jnp.stack(comp_arrays, axis=-1)
                if len(field.shape) > 1:
                    phys_field = phys_field.reshape(self.node_shape + field.shape)

            phys_fields[name] = phys_field

        return State(**phys_fields)
