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
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field

import jax
import jax.numpy as jnp
from jax import Array

from xpektra import linalg
from xpektra.space import SpectralSpace

__all__ = [
    "GenericGreenPreconditioner",
    "GreenPreconditioner",
    "IsotropicGreenPreconditioner",
    "SpectralView",
    "make_generic_preconditioner",
    "make_isotropic_preconditioner",
]


class GreenPreconditioner(ABC):
    """``M^-1`` as a Fourier symbol, plus the real-space wrapper around it.

    This base declares no dataclass fields of its own, only the two attributes it
    needs from every subclass.  A non-dataclass base contributes nothing to
    ``dataclasses.fields``, so each concrete class keeps its own field list --
    and its own ``@register_dataclass``, since registration is not inherited.

    Args:
        space: the spectral space, supplying the transform.
        d: components per node -- 1 for a scalar field, ``ndim`` for a vector.
    """

    space: SpectralSpace
    d: int

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
    def __call__(self, r_flat: Array) -> Array:
        """Applies ``M^-1`` to a flattened residual, returning a flat vector in real space.

        Args:
            r_flat: Flattened residual, shape ``(n,)``, real.
        Returns:
            Flattened preconditioned residual, shape ``(n,)``, real.
        """
        canonical_shape = self.space.shape + (self.d,)
        r_hat = self.space.transform.forward(r_flat.reshape(canonical_shape))
        return self.space.transform.inverse(self.apply_hat(r_hat)).real.reshape(-1)

    def in_fourier(self) -> "SpectralView":
        """This preconditioner as a *callable* on Fourier coefficients.

        Returns:
            A callable that applies the preconditioner to Fourier coefficients.
        """
        return SpectralView(inner=self)


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
        space: the spectral space, supplying the transform.
        d: components per node -- 1 for a scalar field, ``ndim`` for a vector.
    """

    G_hat: Array
    space: SpectralSpace = field(metadata={"static": True})
    d: int = field(metadata={"static": True})

    @jax.jit
    def apply_hat(self, r_hat: Array) -> Array:
        """One contraction per mode: the symbol is stored inverted already."""
        return linalg.contract("...ij,...j->...i", self.G_hat, r_hat)


def make_generic_preconditioner(
    residual_ref_fn: Callable[[Array], Array],
    space: SpectralSpace,
    d: int,
    rtol: float = 1e-12,
) -> GenericGreenPreconditioner:
    """Builds ``inv(K_hat)`` from ``d`` impulse responses of the reference tangent.

    Args:
        residual_ref_fn: applies the reference operator to a flat vector of
            length ``prod(space.shape) * d``.  Must be linear and translation
            invariant; it need not be self-adjoint.
        space: the spectral space.
        d: components per node.
        rtol: a mode is treated as null when the Frobenius norm of its symbol
            falls below ``rtol`` times the largest such norm.

    Returns:
        The preconditioner.
    """
    canonical_shape = space.shape + (d,)
    zero_node = (0,) * len(space.shape)

    seeds = jnp.stack(
        [
            jnp.zeros(canonical_shape).at[zero_node + (i,)].set(1.0).reshape(-1)
            for i in range(d)
        ]
    )
    u_ref = jnp.zeros(canonical_shape).reshape(-1)

    # action of the tangent on a vector v, i.e. K v
    def mv(v: Array) -> Array:
        return jax.jvp(residual_ref_fn, (u_ref,), (v,))[1]

    K_refs = jax.vmap(mv)(seeds)
    columns = [
        space.transform.forward(K_refs[i].reshape(canonical_shape)) for i in range(d)
    ]
    K_hat = jnp.stack(columns, axis=-1)

    # Gate on the *magnitude* of the symbol, not on its determinant.  See the
    # docstring: a determinant threshold is `rtol**(1/d)` in the eigenvalues, so
    # the old `rtol=1e-12` behaved like `1e-4` for d=3 and discarded 8 legitimate
    # modes for Hex1R at N=21.
    scale = jnp.sqrt(jnp.sum(jnp.abs(K_hat) ** 2, axis=(-2, -1)))
    ok = scale > rtol * jnp.max(scale)

    eye = jnp.eye(d, dtype=K_hat.dtype)
    K_safe = jnp.where(ok[..., None, None], K_hat, eye)
    G_hat = jnp.where(ok[..., None, None], linalg.inv(K_safe), 0.0)

    return GenericGreenPreconditioner(G_hat=G_hat, space=space, d=d)


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
        space: the spectral space, supplying the transform.
        d: components per node.
        rtol: relative threshold below which a mode is treated as null.
    """

    g: Array
    h: Array
    alpha: Array
    ok: Array
    lam0: Array
    mu0: Array
    space: SpectralSpace = field(metadata={"static": True})
    d: int = field(metadata={"static": True})
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
    scheme, lam0: Array, mu0: Array, rtol: float = 1e-12
) -> IsotropicGreenPreconditioner:
    """Builds the isotropic Green preconditioner from a scheme's gradient symbol.

    Args:
        scheme: supplies ``gradient_operator``, ``space`` and ``dim``.
        lam0: first Lame parameter of the reference material.
        mu0: shear modulus of the reference material.
        rtol: relative threshold below which a mode is treated as null.

    Returns:
        The preconditioner.
    """
    a1 = scheme.gradient_operator[..., 0, :]
    g, h = a1.real, a1.imag
    # ``op.integrate`` carries the voxel volume, so the tangent of an energy is
    # ``V D^T C0 D``.  Scaling both moduli by V inverts that; the ratio inside
    # the Woodbury solve is unchanged.
    cell_volume = math.prod(
        length / n for length, n in zip(scheme.space.lengths, scheme.space.shape)
    )
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

    return IsotropicGreenPreconditioner(
        g=g,
        h=h,
        alpha=jnp.where(ok, alpha, 1.0),  # keep 1/alpha finite at the null modes
        ok=ok,
        lam0=lam0,
        mu0=mu0,
        space=scheme.space,
        d=scheme.dim,
        rtol=rtol,
    )
