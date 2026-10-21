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

from typing import Literal

import jax.numpy as jnp
from jax import Array

type BCType = Literal["periodic", "dirichlet", "neumann"]
type FacetPair = tuple[BCType, BCType]
type ExtensionKind = Literal["odd", "even", "tile"]

__all__ = [
    "ExtensionKind",
    "get_extension_kind",
    "mask_1d",
    "weights_1d",
    "extend_1d",
    "restrict_1d",
    "extend_cell_1d",
    "extend_cell_field",
]


def get_extension_kind(bc: FacetPair) -> ExtensionKind:
    """Determine the extension kind based on the boundary conditions.

    Rule: reflect odd only if both faces are Dirichlet, otherwise even;
    tiled if periodic along a doubled axis.

    Args:
        bc (FacetPair): The boundary conditions for the face.

    Returns:
        ExtensionKind: The extension kind ("odd", "even", or "tile").
    """
    min_bc, max_bc = bc
    if min_bc == "dirichlet" and max_bc == "dirichlet":
        return "odd"
    elif min_bc == "periodic" and max_bc == "periodic":
        return "tile"
    else:
        return "even"


def mask_1d(n_nodes: int, bc: FacetPair) -> Array:
    """Generate a 1D constraint mask along a doubled axis of n_nodes = N + 1.

    Zeroes constrained Dirichlet faces and periodic wrap duplicate nodes; True elsewhere.

    Args:
        n_nodes (int): Number of physical nodes along the axis (N + 1).
        bc (FacetPair): Canonical (min, max) boundary conditions.

    Returns:
        Array: 1D boolean mask array of shape (n_nodes,).
    """
    if n_nodes < 2:
        raise ValueError(f"n_nodes along a doubled axis must be >= 2, got {n_nodes}.")

    min_bc, max_bc = bc
    mask = jnp.ones(n_nodes, dtype=bool)

    if min_bc == "dirichlet":
        mask = mask.at[0].set(False)
    if max_bc == "dirichlet":
        mask = mask.at[-1].set(False)
    if min_bc == "periodic" and max_bc == "periodic":
        mask = mask.at[-1].set(False)

    return mask


def weights_1d(n_nodes: int, bc: FacetPair) -> Array:
    """Generate 1D diagonal weights W = diag(Eᵀ E) along a doubled axis of n_nodes = N + 1.

    Args:
        n_nodes (int): Number of physical nodes along the axis (N + 1).
        bc (FacetPair): Canonical (min, max) boundary conditions.

    Returns:
        Array: 1D weight array of shape (n_nodes,).
    """
    if n_nodes < 2:
        raise ValueError(f"n_nodes along a doubled axis must be >= 2, got {n_nodes}.")

    min_bc, max_bc = bc
    if min_bc == "periodic" and max_bc == "periodic":
        # Periodic tile: W_k = 2 for 0 <= k < N, W_N = 0 (wrap node)
        return jnp.full(n_nodes, 2.0).at[-1].set(0.0)
    # Reflections (odd & even): W_0 = 1, W_N = 1, W_k = 2 for 1 <= k < N
    return jnp.full(n_nodes, 2.0).at[0].set(1.0).at[-1].set(1.0)


def extend_1d(
    u: Array,
    kind: ExtensionKind,
    axis: int = 0,
) -> Array:
    """Extend array u along specified axis from N + 1 nodes to 2N extended grid points.

    Args:
        u (Array): Input array with shape along axis equal to N + 1.
        kind (ExtensionKind): Extension kind:
            - 'odd': odd reflection [u, -flip(u_{1:N})] -> DST-1
            - 'even': even reflection [u, flip(u_{1:N})] -> DCT-1
            - 'tile': periodic tile [u_{0:N}, u_{0:N}]
        axis (int): Axis along which to extend (default 0).

    Returns:
        Array: Extended array with length 2N along specified axis.
    """
    axis = axis % u.ndim
    n_nodes = u.shape[axis]
    if n_nodes < 2:
        raise ValueError(f"Length along axis {axis} must be >= 2, got {n_nodes}.")
    N = n_nodes - 1

    if kind == "tile":
        sl = [slice(None)] * u.ndim
        sl[axis] = slice(0, N)
        u_periodic = u[tuple(sl)]
        return jnp.concatenate([u_periodic, u_periodic], axis=axis)
    elif kind in ("odd", "even"):
        sl = [slice(None)] * u.ndim
        sl[axis] = slice(1, N)
        reflected = jnp.flip(u[tuple(sl)], axis=axis)
        if kind == "odd":
            reflected = -reflected
        return jnp.concatenate([u, reflected], axis=axis)
    else:
        raise ValueError(
            f"Unknown extension kind: '{kind}'. Expected 'odd', 'even', or 'tile'."
        )


def restrict_1d(
    f: Array,
    kind: ExtensionKind,
    axis: int = 0,
) -> Array:
    """Dual restriction Eᵀ along specified axis from 2N extended grid points to N + 1 nodes.

    Args:
        f (Array): Extended array with shape along axis equal to 2N.
        kind (ExtensionKind): Extension kind:
            - 'odd': fold-and-sum with parity s = -1
            - 'even': fold-and-sum with parity s = +1
            - 'tile': periodic fold-and-sum for k < N, zero at index N
        axis (int): Axis along which to restrict (default 0).

    Returns:
        Array: Restricted array with length N + 1 along specified axis.
    """
    axis = axis % f.ndim
    len_ext = f.shape[axis]
    if len_ext < 2 or len_ext % 2 != 0:
        raise ValueError(
            f"Extended length along axis {axis} must be an even integer >= 2, got {len_ext}."
        )
    N = len_ext // 2

    if kind == "tile":
        sl_first = [slice(None)] * f.ndim
        sl_first[axis] = slice(0, N)
        sl_second = [slice(None)] * f.ndim
        sl_second[axis] = slice(N, 2 * N)

        f_sum = f[tuple(sl_first)] + f[tuple(sl_second)]

        sl_zero = [slice(None)] * f.ndim
        sl_zero[axis] = slice(0, 1)
        zeros = jnp.zeros_like(f[tuple(sl_zero)])
        return jnp.concatenate([f_sum, zeros], axis=axis)

    elif kind in ("odd", "even"):
        s = -1.0 if kind == "odd" else 1.0

        sl_0 = [slice(None)] * f.ndim
        sl_0[axis] = slice(0, 1)

        sl_mid = [slice(None)] * f.ndim
        sl_mid[axis] = slice(1, N)

        sl_tail = [slice(None)] * f.ndim
        sl_tail[axis] = slice(N + 1, 2 * N)

        sl_N = [slice(None)] * f.ndim
        sl_N[axis] = slice(N, N + 1)

        f_mid = f[tuple(sl_mid)] + s * jnp.flip(f[tuple(sl_tail)], axis=axis)
        return jnp.concatenate([f[tuple(sl_0)], f_mid, f[tuple(sl_N)]], axis=axis)
    else:
        raise ValueError(
            f"Unknown extension kind: '{kind}'. Expected 'odd', 'even', or 'tile'."
        )


def extend_cell_1d(
    field: Array,
    axis: int,
    rank: int = 0,
    spatial_dim: int | None = None,
) -> Array:
    """Extend a cell field along specified axis using cell-centred (DCT-2) reflection with tensor parity.

    When reflecting across axis a, component (i, j, ...) is multiplied by (-1)^{#{indices == a}}.

    Args:
        field (Array): Input array with shape (*spatial_shape, *tensor_shape).
        axis (int): Spatial axis along which to mirror.
        rank (int): Rank of tensor at each cell (0 for scalar, 1 for vector, 2 for tensor, etc.).
        spatial_dim (int | None): Number of spatial dimensions. If None, inferred as field.ndim - rank.

    Returns:
        Array: Extended cell field with doubled length (2N) along specified axis.
    """
    spatial_ndim = (field.ndim - rank) if spatial_dim is None else spatial_dim
    axis = axis % spatial_ndim

    flipped = jnp.flip(field, axis=axis)

    if rank == 0:
        return jnp.concatenate([field, flipped], axis=axis)

    tensor_shape = field.shape[spatial_ndim:]
    coords = jnp.indices(tensor_shape)  # shape (rank, *tensor_shape)
    matches = coords == axis
    count = jnp.sum(matches, axis=0)  # shape (*tensor_shape)
    parity = jnp.where(count % 2 == 1, -1.0, 1.0).astype(field.dtype)

    flipped_with_parity = flipped * parity
    return jnp.concatenate([field, flipped_with_parity], axis=axis)


def extend_cell_field(
    field: Array,
    axes: tuple[int, ...],
    rank: int = 0,
    spatial_dim: int | None = None,
) -> Array:
    """Extend a cell field along multiple axes using cell-centred (DCT-2) reflection with tensor parity.

    Args:
        field (Array): Input array with shape (*spatial_shape, *tensor_shape).
        axes (tuple[int, ...]): Spatial axes along which to mirror.
        rank (int): Rank of tensor at each cell.
        spatial_dim (int | None): Number of spatial dimensions.

    Returns:
        Array: Extended cell field.
    """
    spatial_ndim = (field.ndim - rank) if spatial_dim is None else spatial_dim
    for ax in axes:
        norm_ax = ax % spatial_ndim
        field = extend_cell_1d(field, axis=norm_ax, rank=rank, spatial_dim=spatial_ndim)
    return field

