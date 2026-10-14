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

import functools
import inspect
from collections.abc import Callable
from dataclasses import dataclass, field

import jax
from jax import Array

from xpektra.scheme import Scheme
from xpektra.space import SpectralSpace

__all__ = ["SpectralOperator"]


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class SpectralOperator:
    """
    A spectral operator defined by a spectral space and a differential scheme.

    It provides the differential operators (``grad``, ``div``, ``sym_grad``,
    ``laplacian``), the transforms, and the quadrature boundary (``auto_vmap``,
    ``integrate``).

    It deliberately provides no tensor algebra.  Pointwise algebra lives in
    ``xpektra.linalg`` and is written inside an ``auto_vmap``-ed law, where every
    field is ``(*spatial, *tensor)`` and the quadrature axis is absent.  Rank is
    named by the caller there -- ``linalg.contract("...ij,...ji->...", a, b)`` --
    rather than inferred from ``ndim``, which could not distinguish a node field
    ``(*spatial, dim)`` from a centre field ``(*spatial, n_quads)`` and silently
    dispatched the wrong rule.

    ***Args***
    - space: The spectral space.
    - scheme: The differential scheme.

    ***Returns***
    - The spectral operator.

    Example:

    ```
    op = SpectralOperator(scheme=scheme, space=space)
    grad_u = op.grad(u)
    div_v = op.div(v)
    ```
    """

    scheme: Scheme
    space: SpectralSpace = field(metadata={"static": True})

    def _check_spatial_shape(self, field: Array, name: str = "field") -> None:
        """Validate that the field's leading dimensions match the computational domain."""
        space_dim = len(self.space.lengths)
        if field.ndim < space_dim:
            raise ValueError(
                f"{name} has {field.ndim} dimensions, but space requires at least "
                f"{space_dim} spatial dimensions."
            )
        field_spatial = field.shape[:space_dim]
        if field_spatial != self.space.extended_shape:
            if (
                self.space.shape != self.space.extended_shape
                and field_spatial == self.space.shape
            ):
                raise ValueError(
                    f"Input '{name}' has physical shape {field_spatial}, but the operator "
                    f"expects extended shape {self.space.extended_shape}. "
                    f"Did you forget to call `space.extend_primal({name})`?"
                )
            raise ValueError(
                f"Input '{name}' spatial shape {field_spatial} does not match "
                f"space.extended_shape {self.space.extended_shape}."
            )

    @jax.jit
    def grad(self, u: Array) -> Array:
        """Applies the gradient operator to the input real-valued array u.

        ***Arguments***
        - u: A real-valued node field of shape (N,)*dim + tensor axes, e.g.
          (N,)*dim for a scalar or (N,)*dim + (dim,) for a vector.

        ***Returns***
        - The gradient of u, a real-valued array of shape
          (N,)*dim + (n_quads,) + tensor axes + (dim,).

        The derivative index comes last: for a vector field
        ``grad(u)[..., i, j] == d_j u_i``.  This matches the
        continuum-mechanics ``grad u``, so a deformation gradient is
        ``I + grad(u)``.
        """
        self._check_spatial_shape(u, "u")
        u_hat = self.space.transform.forward(u)
        grad_u_hat = self.scheme.apply_gradient(u_hat)
        grad_u = self.space.transform.inverse(grad_u_hat)
        return grad_u.real

    @jax.jit
    def div(self, v: Array) -> Array:
        """Applies the divergence operator to the input real-valued array v.

        ***Arguments***
        - v: A real-valued centre field of shape (N,)*dim + (n_quads, dim, dim).
        ***Returns***
        - The divergence of v, a real-valued array of shape (N,)*dim + (dim,).
        """
        self._check_spatial_shape(v, "v")
        v_hat = self.space.transform.forward(v)
        div_v_hat = self.scheme.apply_divergence(v_hat)
        div_v = self.space.transform.inverse(div_v_hat)
        return div_v.real

    @jax.jit
    def sym_grad(self, u: Array) -> Array:
        """Applies the symmetric gradient operator to the input real-valued array u.

        ***Arguments***
        - u: A real-valued array of shape (N,)*dim.

        ***Returns***
        - The symmetric gradient of u, a real-valued array of shape
          (N,)*dim + (n_quads,) + (dim, dim).  The quadrature axis is always
          present, length 1 for a single-support scheme.
        """
        self._check_spatial_shape(u, "u")
        u_hat = self.space.transform.forward(u)
        sym_grad_u_hat = self.scheme.apply_symmetric_gradient(u_hat)
        # the quadrature axis is trailing, so the
        # transform's ``axes=range(dim)`` batches over it in a single call.
        return self.space.transform.inverse(sym_grad_u_hat).real

    @jax.jit
    def eval(self, u: Array) -> Array:
        """Evaluates a node field at the quadrature points.

        ***Arguments***
        - u: A real-valued node field of shape (N,)*dim + tensor axes.

        ***Returns***
        - The field at the quadrature points, shape
          (N,)*dim + (n_quads,) + tensor axes -- the same points ``grad`` and
          ``sym_grad`` evaluate at, so the result can go straight into an
          ``auto_vmap``-ed law or ``integrate``.
        """
        self._check_spatial_shape(u, "u")
        u_hat = self.space.transform.forward(u)
        u_q_hat = self.scheme.apply_interpolation(u_hat)
        return self.space.transform.inverse(u_q_hat).real

    @jax.jit
    def laplacian(self, u: Array) -> Array:
        """Applies the Laplacian operator to the input real-valued array u.

        ***Arguments***
        - u: A real-valued array of shape (N,)*dim.
        ***Returns***
        - The Laplacian of u, a real-valued array of shape (N,)*dim.
        """
        self._check_spatial_shape(u, "u")
        u_hat = self.space.transform.forward(u)
        lap_u_hat = self.scheme.apply_laplacian(u_hat)
        lap_u = self.space.transform.inverse(lap_u_hat)
        return lap_u.real

    @jax.jit
    def forward(self, u: Array) -> Array:
        """Applies the forward transform to the input real-valued array u.

        ***Arguments***
        - u: A real-valued array of shape (N,)*dim.
        ***Returns***
        - The transformed array u_hat, a complex-valued array of shape (N,)*dim.
        """
        self._check_spatial_shape(u, "u")
        return self.space.transform.forward(u)

    @jax.jit
    def inverse(self, u_hat: Array) -> Array:
        """Applies the inverse transform to the input complex-valued array u_hat.

        ***Arguments***
        - u_hat: A complex-valued array of shape (N,)*dim.
        ***Returns***
        - The inverse transformed array u, a real-valued array of shape (N,)*dim.
        """
        return self.space.transform.inverse(u_hat).real

    # ------------------------------------------------------------------
    # The quadrature boundary
    # ------------------------------------------------------------------

    def auto_vmap[F: Callable](self, **ranks: int) -> Callable[[F], F]:
        """Decorator: lift a per-quadrature-point law to a full centre field.

        A constitutive law is written for **one** quadrature point but for the
        whole grid, so its arguments are ``(*spatial, *tensor)`` with no
        quadrature axis.  This maps it over that axis.  The spatial axes are
        *not* mapped: they stay vectorised inside the law, which is what keeps
        a pointwise ``@`` from becoming a batched GEMM over the grid.

        Each argument's tensor ``rank`` is declared by name; undeclared
        arguments default to rank 0, which covers material fields and other
        scalar fields.  From the rank, ``ndim - len(spatial) - rank`` decides:
        ``1`` means a centre field and it is mapped; ``0`` or less means it
        cannot be carrying a quadrature axis -- a per-voxel field, or a uniform
        constant such as ``jnp.asarray(mu0)`` -- and it is broadcast.  Two or
        more raises, so forgetting a declaration is an error rather than a
        silent broadcast: ``(N, N)`` against ``(N, N, 1)`` otherwise yields
        ``(N, N, N)`` with no complaint.

        Non-arrays (a scheme, a python float) are always broadcast.

        This differs from ``jax_autovmap`` in ``tatva``, which batches over
        *leading* axes.  Here exactly one axis, at a known position, is ever
        mapped.

        Args:
            **ranks: tensor rank of each parameter that is not a scalar field,
                keyed by parameter name.

        Returns:
            A decorator.  The wrapped function takes full centre fields and
            returns a field with the quadrature axis restored in place.

        Example:
            >>> @op.auto_vmap(eps=2)
            ... def energy(eps, alpha, lam, mu):   # eps is (*spatial, d, d)
            ...     tr = linalg.trace(eps)
            ...     return 0.5 * lam * tr**2 + mu * linalg.contract(
            ...         "...ij,...ji->...", eps, eps)
            >>> psi = energy(eps_field, alpha, lam, mu)   # (*spatial, n_quads)
        """
        space_dim = len(self.space.lengths)
        n_quads = self.scheme.n_quads

        def decorator(fn: F) -> F:
            signature = inspect.signature(fn)
            names = list(signature.parameters)
            unknown = set(ranks) - set(names)
            if unknown:
                raise TypeError(
                    f"auto_vmap got ranks for parameters {', '.join(f"'{p}'" for p in sorted(unknown))}, "
                    f"which {fn.__name__} does not take."
                )

            @functools.wraps(fn)
            def wrapper(*args, **kwargs):
                bound = signature.bind(*args, **kwargs)
                bound.apply_defaults()

                values, in_axes = [], []
                for name in names:
                    value = bound.arguments[name]
                    values.append(value)
                    if not hasattr(value, "ndim"):
                        in_axes.append(None)  # scheme, python scalars, ...
                        continue
                    rank = ranks.get(name, 0)
                    extra = value.ndim - space_dim - rank
                    if extra == 0:
                        # Full field on the spatial grid without quadrature axis (e.g. material field).
                        if (
                            self.space.shape != self.space.extended_shape
                            and value.shape[:space_dim] != self.space.extended_shape
                        ):
                            raise ValueError(
                                f"Argument '{name}' passed to {fn.__name__} has spatial shape "
                                f"{value.shape[:space_dim]}, but operator requires extended shape "
                                f"{self.space.extended_shape}. For material fields (e.g. lam, mu), "
                                f"did you forget to call `space.extend_material_field({name})`?"
                            )
                        in_axes.append(None)
                    elif extra < 0:
                        # Fewer axes than a full field: a uniform constant (such as jnp.asarray(mu0)
                        # or a uniform tensor eps0) that is broadcast across the grid.
                        in_axes.append(None)
                    elif extra == 1 and value.shape[space_dim] == n_quads:
                        if (
                            self.space.shape != self.space.extended_shape
                            and value.shape[:space_dim] != self.space.extended_shape
                        ):
                            raise ValueError(
                                f"Argument '{name}' passed to {fn.__name__} has spatial shape "
                                f"{value.shape[:space_dim]}, but operator requires extended shape "
                                f"{self.space.extended_shape}."
                            )
                        in_axes.append(space_dim)
                    else:
                        raise ValueError(
                            f"{fn.__name__}({name}=...): shape {value.shape} has "
                            f"{extra} axes more than a rank-{rank} field on "
                            f"{space_dim} spatial axes. Expected either that, or "
                            f"one extra axis of length {n_quads} at position "
                            f"{space_dim} for a centre field. Declare its rank "
                            "in auto_vmap."
                        )

                if all(axis is None for axis in in_axes):
                    return fn(*values)
                return jax.vmap(fn, in_axes=tuple(in_axes), out_axes=space_dim)(*values)

            return wrapper

        return decorator

    @jax.jit
    def integrate(self, density: Array) -> Array:
        """Integrate over the cell: voxel volume x mean over quadrature x sum over space.

        This is a true integral, ``int f dV``, so energies built with it are
        physical energies.  The differential operators (``grad``, ``div``,
        ``laplacian``) stay pointwise and carry no volume, so for an energy
        ``E = integrate(psi(sym_grad(u)))``

            jax.grad(E)(u) == -V * div(sigma),   V = prod(lengths / shape).

        ``jax.grad(E)`` is the nodal force; ``div(sigma)`` is the force density.
        Take residuals and tangents from the energy, not from ``div``.

        The *mean* over quadrature points is what pairs with the ``1/n_quads``
        inside ``apply_divergence``.  Using ``sum`` instead scales the energy by
        ``n_quads`` and breaks the identity above.

        Tensor axes are preserved, so this returns a scalar for a scalar density
        and a summed tensor for a tensor-valued one.

        Args:
            density: field of shape ``(*spatial, n_quads, *tensor)``.

        Returns:
            Shape ``(*tensor,)``.
        """
        space_dim = len(self.space.lengths)
        n_quads = self.scheme.n_quads
        if density.ndim <= space_dim or density.shape[space_dim] != n_quads:
            raise ValueError(
                f"integrate expects a quadrature axis of length {n_quads} at "
                f"position {space_dim}, got shape {density.shape}."
            )
        if density.shape[:space_dim] != self.space.extended_shape:
            if (
                self.space.shape != self.space.extended_shape
                and density.shape[:space_dim] == self.space.shape
            ):
                raise ValueError(
                    f"integrate received density with physical spatial shape "
                    f"{density.shape[:space_dim]} instead of extended shape "
                    f"{self.space.extended_shape}."
                )
            raise ValueError(
                f"integrate expects density spatial shape {self.space.extended_shape}, "
                f"got {density.shape[:space_dim]}."
            )
        cell_volume = self.space.cell_volume
        density_phys = density[self.space.physical_slices]
        return cell_volume * density_phys.mean(axis=space_dim).sum(
            axis=tuple(range(space_dim))
        )
