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
from typing import Literal, cast, get_args

import jax
from jax import Array
from jax.sharding import NamedSharding

from xpektra import SpectralSpace

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
    _RESERVED = {"_fields"}

    def __init__(self, **fields: Field):
        """Initialize the state with fields.

        Args:
            **fields (Field): The fields associated with the state.
        """
        if bad := self._RESERVED & fields.keys():
            raise ValueError(
                f"Field names {bad} are reserved and cannot be used as field names."
            )
        object.__setattr__(self, "_fields", fields)

    def __getattr__(self, name: str) -> Field:
        """Get a field by name.

        Args:
            name (str): The name of the field.

        Returns:
            Field: The field associated with the name.
        """
        try:
            return self._fields[name]
        except KeyError:
            raise AttributeError(f"Field '{name}' not found in state.")

    def __iter__(self):
        """Iterate over the fields in the state.

        Yields:
            tuple[str, Field]: The name and field of each field in the state.
        """
        yield from self._fields.items()

    def __len__(self):
        """Return the number of fields in the state.

        Returns:
            int: The number of fields in the state.
        """
        return len(self._fields)

    def tree_flatten(self):
        """Flatten the state for JAX transformations.

        Returns:
            tuple: A tuple containing the flattened fields and auxiliary data.
        """
        return tuple(self._fields.values()), tuple(self._fields.keys())

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        """Unflatten the state for JAX transformations.

        Args:
            aux_data (tuple): Auxiliary data containing field names.
            children (tuple): Flattened fields.

        Returns:
            State: The unflattened state.
        """
        return cls(**dict(zip(aux_data, children)))


class SolutionSpace:
    """A solution space for a field or multiple fields.

    Attributes:
        field (Field): The field associated with the solution space.
    """

    def __init__(self, space: SpectralSpace, **fields: Field):
        """Initialize the solution space.

        Args:
            space (SpectralSpace): The spectral space associated with the solution space.
            **fields (Field): The fields associated with the solution space.
        """
        self.space = space
        for name, field in fields.items():
            setattr(self, name, field)

    def to_extended(self, dofs: Array) -> State:
        """Convert flattened degrees of freedom to an extended state.

        Args:
            dofs_flat (Array): The flattened degrees of freedom.

        Returns:
            State: The extended state.
        """
        # Implementation needed

    def to_physical(self, dofs: Array) -> State:
        """Convert flattened degrees of freedom to a physical state.

        Args:
            dofs_flat (Array): The flattened degrees of freedom.

        Returns:
            State: The physical state.
        """
        # Implementation needed
        #

    def zeros(self, sharding: NamedSharding | None = None) -> Array:
        """Return a state with all fields initialized to zero.

        Args:
            sharding (NamedSharding | None): Optional sharding for the arrays.

        Returns:
            State: A state with all fields initialized to zero.
        """
        # Implementation needed

    @property
    def mask(self) -> Array:
        """Return a mask for the solution space.

        Returns:
            Array: A mask for the solution space.
        """
        # Implementation needed
