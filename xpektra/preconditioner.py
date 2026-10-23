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

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field

import jax
import jax.numpy as jnp
from jax import Array
from jax.sharding import NamedSharding

from xpektra import linalg
from xpektra.solution_space import SolutionSpace
from xpektra.space import SpectralSpace

__all__ = [
    "AdaptedGreenPreconditioner",
    "GenericGreenPreconditioner",
    "GreenPreconditioner",
    "IsotropicGreenPreconditioner",
    "SpectralView",
    "adapt_green_preconditioner",
    "make_generic_preconditioner",
    "make_isotropic_preconditioner",
]


type ResidualFn = Callable[[Array], Array]


class GreenPreconditioner(ABC):
    """``M^-1`` as a Fourier symbol, plus the real-space wrapper around it.

    This base declares no dataclass fields of its own, only the two attributes it
    needs from every subclass.  A non-dataclass base contributes nothing to
    ``dataclasses.fields``, so each concrete class keeps its own field list --
    and its own ``@register_dataclass``, since registration is not inherited.

    Args:
        solution_space: the physical or computational SolutionSpace.
    """

    solution_space: SolutionSpace

    @property
    def space(self) -> SpectralSpace:
        """The underlying SpectralSpace."""
        return self.solution_space.space

    @property
    def d(self) -> int:
        """Total components per node."""
        return self.solution_space.nb_components

    @abstractmethod
    def apply_hat(self, r_hat: Array) -> Array:
        """``z_hat = M^-1_hat r_hat``: the symbol, applied where it lives.
        Applies the preconditioner to Fourier coefficients of the residual,
        returning Fourier coefficients of the preconditioned residual.  The shape is
        ``(*spatial, d)``, complex, and the transform is not involved.

        Args:
            r_hat: Fourier coefficients of the residual, shape ``(*spatial, d)``,
                complex.
        Returns:
            Fourier coefficients of the preconditioned residual, shape
                ``(*spatial, d)``, complex.

        """

    @jax.jit
    def __call__(self, r_in: Array) -> Array:
        """Applies ``M^-1`` to a residual, returning a preconditioned residual.

        Args:
            r_in: Flat or shaped residual array, real.
        Returns:
            Preconditioned residual, matching input shape, real.
        """
        is_flat = r_in.ndim == 1
        canonical_shape = self.solution_space.canonical_shape
        r = r_in.reshape(canonical_shape)
        if getattr(self, "sharding", None) is not None:
            r = jax.reshard(r, self.sharding)
        r_hat = self.solution_space.space.transform.forward(r)
        out = self.solution_space.space.transform.inverse(
            self.apply_hat(r_hat)
        ).real
        if is_flat:
            return out.reshape(-1)
        if (
            len(self.solution_space.fields) == 1
            and next(iter(self.solution_space.fields.values())).shape == ()
            and r_in.shape == self.solution_space.node_shape
        ):
            return out[..., 0]
        return out

    def in_fourier(self) -> "SpectralView":
        """This preconditioner as a *callable* on Fourier coefficients.

        Returns:
            A callable that applies the preconditioner to Fourier coefficients.
        """
        return SpectralView(inner=self)

    def adapt(
        self, solution_space: SolutionSpace
    ) -> "GreenPreconditioner | AdaptedGreenPreconditioner":
        """Adapt this extended Green preconditioner to a physical SolutionSpace with boundary conditions.

        Args:
            solution_space: Physical SolutionSpace with boundary conditions.

        Returns:
            GreenPreconditioner | AdaptedGreenPreconditioner: Physical-domain preconditioner.
        """
        return adapt_green_preconditioner(self, solution_space)


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class SpectralView:
    """A preconditioner rebound so that ``__call__`` is the Fourier-space apply.

    A separate *type* rather than a mode flag on the preconditioner itself.  The
    two applies differ in shape and dtype -- flat real against ``(*spatial, d)``
    complex -- so a single entry point covering both could not be honestly named
    or annotated, and the mismatch that matters fails *silently*: handing an
    already-transformed ``r_hat`` to the real-space path finds the reshape a
    no-op, transforms a second time, drops an imaginary part, and returns wrong
    numbers with no error.  Distinct types cannot be confused that way, and both
    stay available from a single construction.

    ``inner`` is a data field, so the symbol's arrays remain leaves and survive
    being stored by a solver.
    """

    inner: GreenPreconditioner

    @jax.jit
    def __call__(self, r_hat: Array) -> Array:
        return self.inner.apply_hat(r_hat)


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class GenericGreenPreconditioner(GreenPreconditioner):
    """``z = M^-1 r`` for a generic reference operator, applied in Fourier space.

    ``G_hat`` is the pointwise inverse of the reference tangent's Fourier symbol,
    built once by :func:`make_generic_preconditioner`.  It is the only dynamic
    field, so rebuilding the preconditioner for a new reference state yields a
    pytree with an unchanged treedef -- new data, no recompile in whatever solver
    holds it.

    Both operands are *node* fields, ``(*spatial, d)``: there is no quadrature
    axis anywhere in the preconditioner, because the reference operator maps
    nodes to nodes.

    Attributes:
        G_hat: ``inv(K_hat)``, shape ``(*spatial, d, d)``, complex.
        solution_space: the SolutionSpace supplying grid shape and channels.
    """

    G_hat: Array
    solution_space: SolutionSpace = field(metadata={"static": True})
    sharding: NamedSharding | None = field(default=None, metadata={"static": True})

    @jax.jit
    def apply_hat(self, r_hat: Array) -> Array:
        """One contraction per mode: the symbol is stored inverted already."""
        return linalg.contract("...ij,...j->...i", self.G_hat, r_hat)


def make_generic_preconditioner(
    residual_ref_fn: ResidualFn,
    solution_space: SolutionSpace,
    rtol: float = 1e-12,
    sharding: NamedSharding | None = None,
) -> "GenericGreenPreconditioner | AdaptedGreenPreconditioner":
    """Builds ``inv(K_hat)`` from ``d`` impulse responses of the reference tangent.

    Args:
        residual_ref_fn: applies the reference operator to a flat vector of
            length ``prod(space.shape) * d``.  Must be linear and translation
            invariant; it need not be self-adjoint.
        space: the spectral space.
        d: components per node.
        rtol: a mode is treated as null when the Frobenius norm of its symbol
            falls below ``rtol`` times the largest such norm.
        sharding: optional sharding for parallel distributed environments.
        solution_space: optional physical SolutionSpace with boundary conditions.
            When provided, returns an adapted physical preconditioner M(r).

    Returns:
        The preconditioner (or adapted physical preconditioner if solution_space is provided).
    """

    nb_comp = solution_space.nb_components
    space = solution_space.space
    canonical_shape = solution_space.canonical_shape
    zero_node = (0,) * len(space.extended_shape)

    def _init_field(arr: Array) -> Array:
        if sharding is not None:
            return jax.device_put(arr, sharding)
        return arr

    u_ref = _init_field(jnp.zeros(canonical_shape)).reshape(-1)

    columns = []
    for i in range(nb_comp):
        seed_i = _init_field(jnp.zeros(canonical_shape).at[zero_node + (i,)].set(1.0))
        K_ref_i = jax.jvp(residual_ref_fn, (u_ref,), (seed_i.reshape(-1),))[1]
        K_ref_i_field = K_ref_i.reshape(canonical_shape)
        if sharding is not None:
            K_ref_i_field = jax.reshard(K_ref_i_field, sharding)
        columns.append(space.transform.forward(K_ref_i_field))

    K_hat = jnp.stack(columns, axis=-1)

    # Gate on the *magnitude* of the symbol, not on its determinant.  See the
    # docstring: a determinant threshold is `rtol**(1/d)` in the eigenvalues, so
    # the old `rtol=1e-12` behaved like `1e-4` for d=3 and discarded 8 legitimate
    # modes for Hex1R at N=21.
    scale = jnp.sqrt(jnp.sum(jnp.abs(K_hat) ** 2, axis=(-2, -1)))
    ok = scale > rtol * jnp.max(scale)

    eye = jnp.eye(nb_comp, dtype=K_hat.dtype)
    K_safe = jnp.where(ok[..., None, None], K_hat, eye)
    G_hat = jnp.where(ok[..., None, None], linalg.inv(K_safe), 0.0)

    precond = GenericGreenPreconditioner(
        G_hat=G_hat, solution_space=solution_space, sharding=sharding
    )
    return precond.adapt(solution_space)


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class IsotropicGreenPreconditioner(GreenPreconditioner):
    """``z = M^-1 r`` with ``M = D^T C0 D`` for an *isotropic* reference material.

    Matrix free: where :class:`GenericGreenPreconditioner` stores a full
    ``(d, d)`` complex block per mode, this stores the two real vectors ``g`` and
    ``h``

    Args:
        g: ``Re(D_1)``, shape ``(*spatial, d)``.
        h: ``Im(D_1)``, shape ``(*spatial, d)``.
        alpha: ``mu0 (|g|^2 + |h|^2)``, already floored at the null modes so
            ``1/alpha`` stays finite there.
        ok: ``False`` at the null modes, where the result is zeroed.
        lam0: first Lame parameter of the reference material, times the voxel
            volume.
        mu0: shear modulus of the reference material, times the voxel volume.
        solution_space: the SolutionSpace supplying grid shape and channels.
        rtol: relative threshold below which a mode is treated as null.
    """

    g: Array
    h: Array
    alpha: Array
    ok: Array
    lam0: Array
    mu0: Array
    solution_space: SolutionSpace = field(metadata={"static": True})
    rtol: float = field(default=1e-12, metadata={"static": True})

    @jax.jit
    def apply_hat(self, r_hat: Array) -> Array:
        """``z_hat = (I - G S^-1 G^T) r_hat / alpha`` -- the Woodbury apply."""
        g, h = self.g, self.h
        gg = linalg.contract("...i,...i->...", g, g)
        hh = linalg.contract("...i,...i->...", h, h)
        gh = linalg.contract("...i,...i->...", g, h)

        # ( alpha/beta I2 + G^T G )^-1 with G = [g, h]
        a = self.alpha / (self.lam0 + self.mu0)
        s11, s22, s12 = a + gg, a + hh, gh
        det = s11 * s22 - s12 * s12

        p = linalg.contract("...i,...i->...", g, r_hat)  # G^T r_hat
        q = linalg.contract("...i,...i->...", h, r_hat)
        c1 = (s22 * p - s12 * q) / det
        c2 = (s11 * q - s12 * p) / det

        z_hat = (r_hat - c1[..., None] * g - c2[..., None] * h) / self.alpha[..., None]
        return jnp.where(self.ok[..., None], z_hat, 0.0)


def make_isotropic_preconditioner(
    scheme,
    lam0: Array,
    mu0: Array,
    solution_space: SolutionSpace,
    rtol: float = 1e-12,
) -> "IsotropicGreenPreconditioner | AdaptedGreenPreconditioner":
    """Builds the isotropic Green preconditioner from a scheme's gradient symbol.

    Args:
        scheme: supplies ``gradient_operator``, ``space`` and ``dim``.
        lam0: first Lame parameter of the reference material.
        mu0: shear modulus of the reference material.
        rtol: relative threshold below which a mode is treated as null.
        solution_space: optional physical SolutionSpace with boundary conditions.
            When provided, returns an adapted physical preconditioner M(r).

    Returns:
        The preconditioner (or adapted physical preconditioner if solution_space is provided).
    """
    a1 = scheme.gradient_operator[..., 0, :]
    g, h = a1.real, a1.imag
    # ``op.integrate`` carries the voxel volume, so the tangent of an energy is
    # ``V D^T C0 D``.  Scaling both moduli by V inverts that; the ratio inside
    # the Woodbury solve is unchanged.
    cell_volume = scheme.space.cell_volume
    lam0, mu0 = cell_volume * jnp.asarray(lam0), cell_volume * jnp.asarray(mu0)

    gg = linalg.contract("...i,...i->...", g, g)
    hh = linalg.contract("...i,...i->...", h, h)
    alpha = mu0 * (gg + hh)

    # The xi=0 mode is exactly zero, but the Nyquist corner comes out at ~1e-29
    # rather than 0 -- round-off in the stencil symbol.  An ``alpha > 0`` test
    # lets it through and 1/alpha then blows up to ~1e28.  The spectral gap is
    # vast (the next mode is O(10)), so a relative threshold separates them
    # cleanly.
    ok = alpha > rtol * jnp.max(alpha)

    precond = IsotropicGreenPreconditioner(
        g=g,
        h=h,
        alpha=jnp.where(ok, alpha, 1.0),  # keep 1/alpha finite at the null modes
        ok=ok,
        lam0=lam0,
        mu0=mu0,
        solution_space=solution_space,
        rtol=rtol,
    )
    return precond.adapt(solution_space)


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class AdaptedGreenPreconditioner:
    """Adapted physical preconditioner for a SolutionSpace with non-periodic boundary conditions.

    Wraps an extended periodic GreenPreconditioner with boundary masks,
    weights, extension E, and transpose restriction Eᵀ.
    """

    G: GreenPreconditioner
    solution_space: SolutionSpace = field(metadata={"static": True})
    scale: float = field(metadata={"static": True})
    is_single_scalar: bool = field(metadata={"static": True})

    @jax.jit
    def __call__(self, r: Array) -> Array:
        is_flat = r.ndim == 1
        if is_flat:
            r_arr = r.reshape(self.solution_space.shape)
        elif self.is_single_scalar and r.shape == self.solution_space.node_shape:
            r_arr = r[..., None]
        else:
            r_arr = r

        # 1. Mask residual
        r_masked = self.solution_space.apply_mask(r_arr)

        # 2. Scale by W⁻¹
        r_scaled = self.solution_space.apply_inv_weights(r_masked)

        # 3. Extend to extended grid (E)
        r_ext_state = self.solution_space.primal_to_extended(r_scaled)

        # 4. Apply Green preconditioner
        if len(self.solution_space.fields) == 1:
            (r_ext,) = r_ext_state
            z_ext = self.G(r_ext)
            z_phys = self.solution_space.dual_to_physical(z_ext)
        else:
            r_ext_stacked = jnp.concatenate(
                [
                    getattr(r_ext_state, name).reshape(
                        self.solution_space.space.extended_shape + (-1,)
                    )
                    for name in self.solution_space.fields
                ],
                axis=-1,
            )
            z_ext_stacked = self.G(r_ext_stacked)
            z_phys = self.solution_space.dual_to_physical(z_ext_stacked)

        # 5. Scale by 2^{n_d} W⁻¹
        z_scaled = self.scale * self.solution_space.apply_inv_weights(z_phys)

        # 6. Mask
        z_out = self.solution_space.apply_mask(z_scaled)

        if is_flat:
            return z_out.reshape(-1)
        if self.is_single_scalar and r.shape == self.solution_space.node_shape:
            return z_out[..., 0]
        return z_out


def adapt_green_preconditioner(
    G: GreenPreconditioner,
    solution_space: SolutionSpace,
) -> "GreenPreconditioner | AdaptedGreenPreconditioner":
    """Adapt extended-grid GreenPreconditioner to a physical SolutionSpace with boundary conditions.

    Applies the exact symmetric projection:
        M(r) = mask · 2^{n_d} W⁻¹ Eᵀ G(E W⁻¹ mask · r)

    Args:
        G (GreenPreconditioner):
            Preconditioner on the extended periodic grid.
        solution_space (SolutionSpace): Physical solution space with boundary conditions.

    Returns:
        GreenPreconditioner | AdaptedGreenPreconditioner:
            Physical-domain preconditioner mapping residual to search direction.
    """
    if not solution_space.space.non_periodic_axes:
        return G  # No adaptation needed for fully periodic spaces

    if not isinstance(G, GreenPreconditioner):
        raise TypeError(
            f"Expected a GreenPreconditioner instance, got {type(G).__name__}. "
            "Only Fourier-space Green preconditioners on the extended grid can be adapted "
            "to a SolutionSpace."
        )

    if G.solution_space.space.extended_shape != solution_space.space.extended_shape:
        raise ValueError(
            f"Preconditioner grid shape {G.solution_space.space.extended_shape} does not match "
            f"SolutionSpace extended grid shape {solution_space.space.extended_shape}."
        )

    scale = float(2 ** len(solution_space.space.non_periodic_axes))
    is_single_scalar = (
        len(solution_space.fields) == 1
        and next(iter(solution_space.fields.values())).shape == ()
    )

    return AdaptedGreenPreconditioner(
        G=G,
        solution_space=solution_space,
        scale=scale,
        is_single_scalar=is_single_scalar,
    )
