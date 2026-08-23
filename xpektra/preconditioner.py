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

from collections.abc import Callable
from dataclasses import dataclass, field

import jax
import jax.numpy as jnp
from jax import Array

from xpektra import linalg
from xpektra.space import SpectralSpace

__all__ = [
    "GenericGreenPreconditioner",
    "IsotropicGreenPreconditioner",
    "make_generic_preconditioner",
    "make_isotropic_preconditioner",
]


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class GenericGreenPreconditioner:
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
    def __call__(self, r_flat: Array) -> Array:
        """Applies ``M^-1`` to a flattened residual, returning a flat vector.

        Flat in and flat out because Krylov solvers work on vectors; the reshape
        to ``(*spatial, d)`` is what the transform needs.
        """
        canonical_shape = self.space.shape + (self.d,)
        r_hat = self.space.transform.forward(r_flat.reshape(canonical_shape))
        z_hat = linalg.contract("...ij,...j->...i", self.G_hat, r_hat)
        return self.space.transform.inverse(z_hat).real.reshape(-1)


def make_generic_preconditioner(
    residual_ref_fn: Callable[[Array], Array],
    space: SpectralSpace,
    d: int,
    rtol: float = 1e-12,
) -> GenericGreenPreconditioner:
    """Builds ``inv(K_hat)`` from ``d`` impulse responses of the reference tangent.

    The reference operator is translation invariant, so its action is a
    convolution and column ``i`` of the symbol is the FFT of the response to a
    unit impulse at node 0 in direction ``i``.  Stacking the columns on
    ``axis=-1`` puts response ``i`` at ``K_hat[..., :, i]``, so component ``j``
    of response ``i`` lands at ``K_hat[..., j, i]``.

    ``G_hat`` costs ``d*d`` complex numbers per pixel, which is the minimal
    faithful representation of a *generic* symbol.
    :class:`IsotropicGreenPreconditioner` gets away with ``O(d)`` only because
    its symbol is identity-plus-rank-two, so Woodbury reduces the inverse to a
    2x2 solve; nothing of the sort holds here.  Building it
    inside ``__call__`` instead measured ~8x slower per apply, since every apply
    would redo ``d`` tangent applies and ``d`` forward FFTs for a symbol that is
    constant within a solve.

    Call this again whenever the reference state changes (a new load increment,
    say): the result has the same treedef, so no recompile follows.

    Null modes are handled explicitly.  ``K_hat`` is singular at ``xi = 0``
    (rigid translation) and, for a reduced-integration stencil, at the Nyquist
    corner.  Their symbols come out at round-off (~1e-30) rather than exactly
    zero, so an ``== 0`` test lets them through and the closed-form inverse then
    blows up to ~1e29.  The spectral gap is vast (the next ``|det|`` is O(1)), so
    a *relative* threshold separates them cleanly: the identity is inverted in
    their place and the result is then zeroed there.

    Args:
        residual_ref_fn: applies the reference operator to a flat vector of
            length ``prod(space.shape) * d``.  Must be linear and translation
            invariant; it need not be self-adjoint.
        space: the spectral space.
        d: components per node.
        rtol: relative threshold on ``|det(K_hat)|`` below which a mode is
            treated as null.

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

    det = linalg.det(K_hat)
    ok = jnp.abs(det) > rtol * jnp.max(jnp.abs(det))

    eye = jnp.eye(d, dtype=K_hat.dtype)
    K_safe = jnp.where(ok[..., None, None], K_hat, eye)
    G_hat = jnp.where(ok[..., None, None], linalg.inv(K_safe), 0.0)

    return GenericGreenPreconditioner(G_hat=G_hat, space=space, d=d)


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class IsotropicGreenPreconditioner:
    """``z = M^-1 r`` with ``M = D^T C0 D`` for an *isotropic* reference material.

    Matrix free: where :class:`GenericGreenPreconditioner` stores a full
    ``(d, d)`` complex block per mode, this stores only the two real vectors
    ``g`` and ``h`` -- the real and imaginary parts of the scheme's gradient
    symbol -- and two moduli.  For ``d = 3`` that is 6 reals per voxel against 18,
    and there is no build cost at all: no impulse responses, no ``d`` tangent
    applies, no ``d`` forward FFTs.

    The saving comes from structure.  The isotropic reference symbol is
    ``alpha I + beta G G^T`` with ``G = [g, h]`` and ``alpha = mu0 (|g|^2 +
    |h|^2)`` -- an identity plus a rank-*two* update -- so Sherman-Morrison-
    Woodbury reduces the inverse to a 2x2 solve regardless of ``d``.  Nothing of
    the sort holds for a general anisotropic or non-symmetric reference, which is
    what :class:`GenericGreenPreconditioner` is for.

    Attributes:
        g: ``Re(D_1)``, shape ``(*spatial, d)``.
        h: ``Im(D_1)``, shape ``(*spatial, d)``.
        lam0: first Lame parameter of the reference material.
        mu0: shear modulus of the reference material.
        space: the spectral space, supplying the transform.
        d: components per node.
        rtol: relative threshold below which a mode is treated as null.
    """

    g: Array
    h: Array
    lam0: Array
    mu0: Array
    space: SpectralSpace = field(metadata={"static": True})
    d: int = field(metadata={"static": True})
    rtol: float = field(default=1e-12, metadata={"static": True})

    @jax.jit
    def __call__(self, r_flat: Array) -> Array:
        g, h = self.g, self.h
        beta = self.lam0 + self.mu0

        gg = linalg.contract("...i,...i->...", g, g)
        hh = linalg.contract("...i,...i->...", h, h)
        gh = linalg.contract("...i,...i->...", g, h)
        alpha = self.mu0 * (gg + hh)

        # The xi=0 mode is exactly zero, but the Nyquist corner comes out at
        # ~1e-29 rather than 0 -- round-off in the stencil symbol.  An
        # ``alpha > 0`` test lets it through and 1/alpha then blows up to ~1e28.
        # The spectral gap is vast (the next mode is O(10)), so a relative
        # threshold separates them cleanly.
        ok = alpha > self.rtol * jnp.max(alpha)
        safe = jnp.where(ok, alpha, 1.0)  # keep 1/alpha finite at the null modes

        r_hat = self.space.transform.forward(
            r_flat.reshape(self.space.shape + (self.d,))
        )

        # ( alpha/beta I2 + G^T G )^-1 with G = [g, h]
        a = safe / beta
        s11, s22, s12 = a + gg, a + hh, gh
        det = s11 * s22 - s12 * s12

        p = linalg.contract("...i,...i->...", g, r_hat)  # G^T r_hat
        q = linalg.contract("...i,...i->...", h, r_hat)
        c1 = (s22 * p - s12 * q) / det
        c2 = (s11 * q - s12 * p) / det

        z_hat = (r_hat - c1[..., None] * g - c2[..., None] * h) / safe[..., None]
        z_hat = jnp.where(ok[..., None], z_hat, 0.0)

        return self.space.transform.inverse(z_hat).real.reshape(-1)


def make_isotropic_preconditioner(
    scheme, lam0: Array, mu0: Array, rtol: float = 1e-12
) -> IsotropicGreenPreconditioner:
    """Builds the isotropic Green preconditioner from a scheme's gradient symbol.

    Nothing is precomputed beyond splitting the symbol into real and imaginary
    parts, so this is cheap enough to rebuild whenever the reference moduli
    change.

    Only the *first* support's symbol is used, which is not the approximation it
    looks like.  The true quadrature-averaged acoustic tensor is

    ``A = mean_r [ mu0 |a_r|^2 I + lam0 conj(a_r) a_r^T + mu0 a_r a_r^H ]``

    and the closed form below drops an antisymmetric part
    ``i (lam0 - mu0) (g h^T - h g^T)``.  That part vanishes for every scheme in
    the library: for the single-support schemes because their symbol is a common
    complex factor times a real vector, so ``g`` and ``h`` are parallel; and for
    TETRA2 because its supports satisfy ``a_2 = phi(xi) conj(a_1)`` with
    ``|phi| = 1``, which makes the average real symmetric.  Verified to ~1e-16
    relative by ``test_isotropic_symbol_is_the_exact_acoustic_tensor``.

    A future scheme satisfying neither condition would silently degrade this from
    exact to approximate -- still a usable preconditioner, since one only has to
    be spectrally close, but no longer the exact inverse of its reference.

    Args:
        scheme: supplies ``gradient_operator``, ``space`` and ``dim``.
        lam0: first Lame parameter of the reference material.
        mu0: shear modulus of the reference material.
        rtol: relative threshold below which a mode is treated as null.

    Returns:
        The preconditioner.
    """
    a1 = scheme.gradient_operator[..., 0, :]
    return IsotropicGreenPreconditioner(
        g=a1.real,
        h=a1.imag,
        lam0=jnp.asarray(lam0),
        mu0=jnp.asarray(mu0),
        space=scheme.space,
        d=scheme.dim,
        rtol=rtol,
    )
