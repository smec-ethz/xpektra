import itertools
from collections.abc import Callable
from typing import Literal

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from xpektra.scheme import QuadFullScheme, Scheme
from xpektra.space import SpectralSpace


_AXIS_NAMES = ("x", "y", "z")


def _normalize_axis(axis: str, dim: int) -> int:
    """Normalize a boundary normal axis name ('x', 'y', 'z') to an index."""
    if not isinstance(axis, str):
        raise TypeError(
            f"axis must be one of {_AXIS_NAMES[:dim]}, got {type(axis).__name__}."
        )
    ax_l = axis.strip().lower()
    if ax_l not in _AXIS_NAMES[:dim]:
        raise ValueError(
            f"Invalid axis {axis!r} for {dim}D space; expected one of {_AXIS_NAMES[:dim]}."
        )
    return _AXIS_NAMES.index(ax_l)


def make_boundary_projection(
    scheme: Scheme,
    quad_points: Array | np.ndarray,
    axis: Literal["x", "y", "z"],
    side: Literal["min", "max"] = "min",
    kind: Literal["grad", "eval"] = "grad",
    rule: Literal["auto", "lagrange", "average"] = "auto",
) -> Callable[[Array], Array]:
    """Build a local boundary projector from scheme stencils to boundary quadrature points.

    Evaluates only on the 1-cell boundary strip of a nodal field ``u`` (without running
    a global FFT), supporting ``QuadFullScheme``, ``Quad1RScheme``, ``Hex1RScheme``,
    and ``Tetra2Scheme``.

    Args:
        scheme: Discretization scheme (provides ``space`` and cell stencils).
        quad_points: Reference quadrature points on the boundary face, shape
            ``(n_face_q, dim - 1)`` in ``[-1, 1]^(dim - 1)``.
        axis: Normal axis of the boundary (``"x"``, ``"y"``, or ``"z"``).
        side: Boundary side along ``axis`` (``"min"`` or ``"max"``).
        kind: ``"grad"`` to project the spatial gradient (uses ``support_stencils``)
            or ``"eval"`` to project field values (uses ``support_value_stencils``).
        rule: ``"auto"`` (Lagrange interpolation for ``QuadFullScheme``, uniform
            average otherwise), ``"lagrange"``, or ``"average"``.

    Returns:
        A JIT-compatible function ``project(u)`` returning
        ``(n_face_elems, n_face_q, *val_shape, dim)`` for ``kind="grad"`` or
        ``(n_face_elems, n_face_q, *val_shape)`` for ``kind="eval"``.
    """
    if not isinstance(side, str):
        raise TypeError(f"side must be 'min' or 'max', got {type(side).__name__}.")
    side_norm = side.strip().lower()
    if side_norm not in ("min", "max"):
        raise ValueError(f"side must be 'min' or 'max', got {side!r}")
    if kind not in ("grad", "eval"):
        raise ValueError(f"kind must be 'grad' or 'eval', got {kind!r}")

    shape = scheme.space.shape
    dim = scheme.space.dim
    axis_idx = _normalize_axis(axis, dim)
    n_bulk_q = scheme.n_quads
    face_qp = np.asarray(quad_points, dtype=float)
    if face_qp.ndim == 1:
        face_qp = face_qp[:, None]
    n_face_q = face_qp.shape[0]
    tan_axes = [ax for ax in range(dim) if ax != axis_idx]

    use_rule = (
        ("lagrange" if isinstance(scheme, QuadFullScheme) else "average")
        if rule == "auto"
        else rule
    )

    if use_rule == "lagrange":
        g = 1.0 / np.sqrt(3.0)
        signs = np.array(list(itertools.product((-1.0, 1.0), repeat=dim)))
        face_xi = np.empty((n_face_q, dim))
        face_xi[:, axis_idx] = -1.0 if side_norm == "min" else 1.0
        for j, ax in enumerate(tan_axes):
            face_xi[:, ax] = face_qp[:, j]
        weights = np.prod(
            0.5 * (1.0 + signs[None, :, :] * (face_xi[:, None, :] / g)),
            axis=-1,
        )
    elif use_rule == "average":
        weights = np.full((n_face_q, n_bulk_q), 1.0 / n_bulk_q)
    else:
        raise ValueError(f"Unknown projection rule {rule!r}")

    corners = list(itertools.product((0, 1), repeat=dim))
    n_corners = len(corners)
    h_subs = {
        f"h_{ax + 1}": float(scheme.space.lengths[ax]) / shape[ax]
        for ax in range(dim)
    }

    def _eval_weight(weight) -> float:
        if hasattr(weight, "subs"):
            subs = {s: h_subs[s.name] for s in weight.free_symbols if s.name in h_subs}
            return float(weight.subs(subs))
        return float(weight)

    if kind == "grad":
        stencil_arr = np.zeros((n_bulk_q, dim, n_corners))
        for q, dir_stencils in enumerate(scheme.support_stencils):
            for d_ax, entries in enumerate(dir_stencils):
                for offset, weight in entries:
                    stencil_arr[q, d_ax, corners.index(tuple(offset))] += _eval_weight(
                        weight
                    )
        S_face = jnp.asarray(np.einsum("qa,adc->qdc", weights, stencil_arr))
    else:
        stencil_arr = np.zeros((n_bulk_q, n_corners))
        for q, entries in enumerate(scheme.support_value_stencils):
            for offset, weight in entries:
                stencil_arr[q, corners.index(tuple(offset))] += _eval_weight(weight)
        S_face = jnp.asarray(np.einsum("qa,ac->qc", weights, stencil_arr))

    cell_start = 0 if side_norm == "min" else shape[axis_idx] - 1

    def _corner_slice(u: Array, offset: tuple[int, ...]) -> Array:
        sl = tuple(
            cell_start + offset[ax]
            if ax == axis_idx
            else slice(offset[ax], shape[ax] + offset[ax])
            for ax in range(dim)
        )
        return u[sl]

    if kind == "grad":
        def project_cell(u_corners: Array) -> Array:
            w = S_face.reshape((n_face_q, dim, n_corners) + (1,) * (u_corners.ndim - 1))
            out = jnp.sum(w * u_corners[None, None, ...], axis=2)
            return jnp.moveaxis(out, 1, -1)
    else:
        def project_cell(u_corners: Array) -> Array:
            w = S_face.reshape((n_face_q, n_corners) + (1,) * (u_corners.ndim - 1))
            return jnp.sum(w * u_corners[None, ...], axis=1)

    _project_cells = jax.vmap(project_cell)

    def project(u: Array) -> Array:
        corners_u = jnp.stack([_corner_slice(u, c) for c in corners], axis=len(tan_axes))
        val_shape = u.shape[dim:]
        corners_flat = corners_u.reshape(-1, n_corners, *val_shape)
        return _project_cells(corners_flat)

    return project


def make_boundary_mesh(
    space: SpectralSpace,
    axis: Literal["x", "y", "z"],
    side: Literal["min", "max"] = "min",
) -> tuple[Array, Array]:
    """Build boundary node coordinates and element connectivity for a domain face.

    Uses the same tangential-axis ``ij`` row-major ordering as
    :func:`make_boundary_projection` and boundary slices of ``u``, so element ``e``
    and node ``n`` align directly with projected boundary fields and sliced DOFs.
    Periodic tangential axes wrap their last element back to node ``0``.

    Args:
        space: Spectral domain providing ``shape``, ``lengths``, ``node_shape``,
            and ``node_coords``.
        axis: Normal axis of the boundary (``"x"``, ``"y"``, or ``"z"``).
        side: Boundary side along ``axis`` (``"min"`` or ``"max"``).

    Returns:
        ``(coords, elements)`` where ``coords`` has shape ``(n_face_nodes, dim)``
        with physical ``(x, y)`` or ``(x, y, z)`` coordinates, and ``elements`` has
        shape ``(n_face_elems, 2)`` (``Line2``) in 2D or ``(n_face_elems, 4)``
        (``Quad4``, CCW ``[n00, n10, n11, n01]``) in 3D.
    """
    if not isinstance(side, str):
        raise TypeError(f"side must be 'min' or 'max', got {type(side).__name__}.")
    side_norm = side.strip().lower()
    if side_norm not in ("min", "max"):
        raise ValueError(f"side must be 'min' or 'max', got {side!r}")

    dim = space.dim
    if dim not in (2, 3):
        raise ValueError(f"make_boundary_mesh requires 2D or 3D space, got {dim}D.")

    axis_idx = _normalize_axis(axis, dim)
    tan_axes = [ax for ax in range(dim) if ax != axis_idx]

    tan_1d = [space.node_coords(_AXIS_NAMES[ax]) for ax in tan_axes]
    tan_grids = jnp.meshgrid(*tan_1d, indexing="ij")
    n_nodes = tan_grids[0].size
    normal_val = 0.0 if side_norm == "min" else float(space.lengths[axis_idx])

    cols: list[Array] = [jnp.empty(0)] * dim
    cols[axis_idx] = jnp.full(n_nodes, normal_val)
    for grid, ax in zip(tan_grids, tan_axes):
        cols[ax] = grid.ravel()
    coords = jnp.stack(cols, axis=1)

    if dim == 2:
        t0 = tan_axes[0]
        i0 = jnp.arange(space.shape[t0])
        i1 = (i0 + 1) % space.node_shape[t0]
        elements = jnp.stack([i0, i1], axis=1)
    else:
        t0, t1 = tan_axes
        n_n0, n_n1 = space.node_shape[t0], space.node_shape[t1]
        ii, jj = jnp.meshgrid(
            jnp.arange(space.shape[t0]), jnp.arange(space.shape[t1]), indexing="ij"
        )
        ii1 = (ii + 1) % n_n0
        jj1 = (jj + 1) % n_n1
        n00 = (ii * n_n1 + jj).ravel()
        n10 = (ii1 * n_n1 + jj).ravel()
        n11 = (ii1 * n_n1 + jj1).ravel()
        n01 = (ii * n_n1 + jj1).ravel()
        elements = jnp.stack([n00, n10, n11, n01], axis=1)

    return coords, elements
