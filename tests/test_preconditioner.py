"""``make_generic_preconditioner`` recovers the inverse symbol of a reference operator.

The construction assumes only that the reference operator is linear and
translation invariant -- not that it is self-adjoint.  These tests exercise both
a non-singular reference (where ``M^-1 K`` is exactly the identity) and a
singular one (where the null modes must be projected out rather than inverted).
"""

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import pytest

from xpektra import SpectralSpace, linalg
from xpektra.preconditioner import (
    make_generic_preconditioner,
    make_isotropic_preconditioner,
)
from xpektra.scheme import (
    FourierScheme,
    Hex1RScheme,
    Quad1RScheme,
    Tetra2Scheme,
)
from xpektra.spectral_operator import SpectralOperator
from xpektra.transform import FFTTransform

N = 8

SCHEMES = [(Quad1RScheme, 2), (Hex1RScheme, 3), (FourierScheme, 2)]
IDS = [f"{c.__name__}-{d}d" for c, d in SCHEMES]


def _operator(cls, dim):
    space = SpectralSpace(
        lengths=(1.0,) * dim, shape=(N,) * dim, transform=FFTTransform(dim=dim)
    )
    return SpectralOperator(scheme=cls(space=space), space=space), space


def _helmholtz(op, space, dim):
    """``u - laplacian(u)`` on a scalar field: linear, shift invariant, no null mode.

    The mass term is what makes it invertible everywhere, so ``M^-1 K`` should be
    the identity to round-off with nothing to project out.
    """

    def apply(u_flat):
        u = u_flat.reshape(space.shape)
        return (u - op.laplacian(u)).reshape(-1)

    return apply


@pytest.mark.parametrize(("cls", "dim"), SCHEMES, ids=IDS)
def test_inverts_a_nonsingular_reference(cls, dim):
    """``M^-1 K v == v`` when the reference has no null modes."""
    op, space = _operator(cls, dim)
    K = _helmholtz(op, space, dim)
    M_inv = make_generic_preconditioner(K, space, d=1)

    v = jax.random.normal(jax.random.PRNGKey(0), (N**dim,))
    np.testing.assert_allclose(M_inv(K(v)), v, atol=1e-10)


@pytest.mark.parametrize(("cls", "dim"), SCHEMES, ids=IDS)
def test_null_modes_are_projected_not_inverted(cls, dim):
    """A singular reference must give 0 on its null space, not ``inf``.

    The Laplacian annihilates a constant field.  Its symbol there is round-off
    rather than exactly zero, so an ``== 0`` guard would let it through and the
    closed-form inverse would return ~1e29.
    """
    op, space = _operator(cls, dim)

    def K(u_flat):
        return (-op.laplacian(u_flat.reshape(space.shape))).reshape(-1)

    M_inv = make_generic_preconditioner(K, space, d=1)

    assert jnp.all(jnp.isfinite(M_inv.G_hat)), "null mode produced a non-finite inverse"

    constant = jnp.ones((N**dim,))
    np.testing.assert_allclose(M_inv(constant), 0.0, atol=1e-12)


@pytest.mark.parametrize(("cls", "dim"), SCHEMES, ids=IDS)
def test_is_a_left_inverse_off_the_null_space(cls, dim):
    """``M^-1 K`` is the identity on the range, i.e. ``M^-1 K M^-1 == M^-1``."""
    op, space = _operator(cls, dim)

    def K(u_flat):
        return (-op.laplacian(u_flat.reshape(space.shape))).reshape(-1)

    M_inv = make_generic_preconditioner(K, space, d=1)

    r = jax.random.normal(jax.random.PRNGKey(1), (N**dim,))
    z = M_inv(r)
    np.testing.assert_allclose(M_inv(K(z)), z, atol=1e-10)


def test_handles_a_vector_field_reference():
    """``d = ndim``: the symbol is a full ``(d, d)`` block per mode."""
    dim = 2
    op, space = _operator(Quad1RScheme, dim)
    n_u = N**dim * dim

    def K(u_flat):
        u = u_flat.reshape(space.shape + (dim,))
        return (u - op.laplacian(u)).reshape(-1)

    M_inv = make_generic_preconditioner(K, space, d=dim)
    assert M_inv.G_hat.shape == space.shape + (dim, dim)

    v = jax.random.normal(jax.random.PRNGKey(2), (n_u,))
    np.testing.assert_allclose(M_inv(K(v)), v, atol=1e-10)


def test_makes_no_symmetry_assumption():
    """A non-self-adjoint reference must invert correctly too.

    This is the property non-associated plasticity depends on: its tangent is
    not symmetric, so a construction that assumed self-adjointness would be
    silently wrong rather than merely slower.
    """
    dim = 2
    op, space = _operator(Quad1RScheme, dim)

    def K(u_flat):
        """``u - laplacian(u) + c . grad(u)``: the advection term breaks symmetry."""
        u = u_flat.reshape(space.shape)
        g = op.grad(u)  # (*spatial, n_quads, dim)
        drift = g[..., 0, 0] + 0.5 * g[..., 0, 1]
        return (u - op.laplacian(u) + 0.7 * drift).reshape(-1)

    M_inv = make_generic_preconditioner(K, space, d=1)

    # the symbol really is non-Hermitian, so the test is not vacuous
    K_hat_asym = jnp.max(jnp.abs(M_inv.G_hat - jnp.conj(M_inv.G_hat)))
    assert float(K_hat_asym) > 1e-6, "reference came out Hermitian; test is vacuous"

    v = jax.random.normal(jax.random.PRNGKey(3), (N**dim,))
    np.testing.assert_allclose(M_inv(K(v)), v, atol=1e-10)


def test_treedef_is_stable_across_rebuilds():
    """Rebuilding for a new reference state must not force a recompile.

    `G_hat` is the only dynamic field, so two builds differing in data alone have
    identical treedefs -- that is what lets a solver hold the preconditioner as a
    pytree child and swap it between increments.
    """
    dim = 2
    op, space = _operator(Quad1RScheme, dim)

    def make(scale):
        def K(u_flat):
            u = u_flat.reshape(space.shape)
            return (scale * u - op.laplacian(u)).reshape(-1)

        return make_generic_preconditioner(K, space, d=1)

    a, b = make(1.0), make(2.5)
    assert jax.tree_util.tree_structure(a) == jax.tree_util.tree_structure(b)
    assert not np.allclose(a.G_hat, b.G_hat), "different references gave identical data"


def test_survives_a_jit_boundary():
    """The stored symbol crosses `jit` as a pytree child."""
    dim = 2
    op, space = _operator(Quad1RScheme, dim)
    K = _helmholtz(op, space, dim)
    M_inv = make_generic_preconditioner(K, space, d=1)

    r = jax.random.normal(jax.random.PRNGKey(4), (N**dim,))
    out = jax.jit(lambda m, x: m(x))(M_inv, r)
    np.testing.assert_allclose(out, M_inv(r), atol=1e-14)


# ---------------------------------------------------------------------------
# The isotropic (matrix-free) preconditioner
# ---------------------------------------------------------------------------

ISO_SCHEMES = [(Quad1RScheme, 2), (Hex1RScheme, 3)]
ISO_IDS = [f"{c.__name__}-{d}d" for c, d in ISO_SCHEMES]

LAM0, MU0 = 1.3, 0.7


def _isotropic_reference(op, space, dim):
    """``K = D^T C0 D`` for a homogeneous isotropic material, via the energy."""
    u_shape = space.shape + (dim,)

    def energy(u_flat):
        eps = op.sym_grad(u_flat.reshape(u_shape))
        tr = linalg.trace(eps)
        psi = 0.5 * LAM0 * tr**2 + MU0 * linalg.contract("...ij,...ji->...", eps, eps)
        return op.integrate(psi)

    return jax.jit(jax.grad(energy))


@pytest.mark.parametrize(("cls", "dim"), ISO_SCHEMES, ids=ISO_IDS)
def test_isotropic_matches_the_generic_preconditioner(cls, dim):
    """The closed form and the impulse-response build agree.

    This is the load-bearing test: the two arrive at the same operator by
    entirely different routes -- Woodbury on an identity-plus-rank-two symbol
    versus ``d`` impulse responses followed by an explicit ``d x d`` inverse.
    An error in the 2x2 algebra could not survive it.
    """
    op, space = _operator(cls, dim)
    K = _isotropic_reference(op, space, dim)

    M_iso = make_isotropic_preconditioner(op.scheme, LAM0, MU0)
    M_gen = make_generic_preconditioner(K, space, d=dim)

    r = K(jax.random.normal(jax.random.PRNGKey(0), (N**dim * dim,)))
    np.testing.assert_allclose(M_iso(r), M_gen(r), atol=1e-12)


@pytest.mark.parametrize(("cls", "dim"), ISO_SCHEMES, ids=ISO_IDS)
def test_isotropic_is_a_left_inverse_on_the_range(cls, dim):
    """``M^-1 K`` is the identity on the range: ``M K M == M``.

    Stated this way rather than ``M K v == v`` because the reference operator is
    genuinely singular -- rigid translation, plus the Nyquist corner for a
    reduced-integration stencil -- so a generic ``v`` is not recovered.
    """
    op, space = _operator(cls, dim)
    K = _isotropic_reference(op, space, dim)
    M_inv = make_isotropic_preconditioner(op.scheme, LAM0, MU0)

    r = jax.random.normal(jax.random.PRNGKey(1), (N**dim * dim,))
    z = M_inv(r)
    np.testing.assert_allclose(M_inv(K(z)), z, atol=1e-10)


@pytest.mark.parametrize(("cls", "dim"), ISO_SCHEMES, ids=ISO_IDS)
def test_isotropic_projects_out_rigid_translation(cls, dim):
    """The ``xi = 0`` mode is annihilated, so the output has zero mean per component."""
    op, space = _operator(cls, dim)
    M_inv = make_isotropic_preconditioner(op.scheme, LAM0, MU0)

    r = jax.random.normal(jax.random.PRNGKey(2), (N**dim * dim,))
    z = M_inv(r).reshape(space.shape + (dim,))
    np.testing.assert_allclose(jnp.mean(z, axis=tuple(range(dim))), 0.0, atol=1e-12)
    assert jnp.all(jnp.isfinite(z)), "null mode produced a non-finite result"


@pytest.mark.parametrize(("cls", "dim"), ISO_SCHEMES, ids=ISO_IDS)
def test_isotropic_stores_less_than_the_generic(cls, dim):
    """Matrix free: ``O(d)`` reals per voxel rather than ``O(d^2)`` complex.

    The point of keeping both. If this ever inverts, the isotropic form has lost
    its reason to exist.
    """
    op, space = _operator(cls, dim)
    K = _isotropic_reference(op, space, dim)

    def nbytes(tree):
        return sum(
            leaf.nbytes
            for leaf in jax.tree_util.tree_leaves(tree)
            if hasattr(leaf, "nbytes")
        )

    iso = nbytes(make_isotropic_preconditioner(op.scheme, LAM0, MU0))
    gen = nbytes(make_generic_preconditioner(K, space, d=dim))
    assert iso < gen, f"isotropic stored {iso} B, generic {gen} B"


def test_isotropic_survives_a_jit_boundary():
    op, space = _operator(Quad1RScheme, 2)
    M_inv = make_isotropic_preconditioner(op.scheme, LAM0, MU0)

    r = jax.random.normal(jax.random.PRNGKey(3), (N**2 * 2,))
    out = jax.jit(lambda m, x: m(x))(M_inv, r)
    np.testing.assert_allclose(out, M_inv(r), atol=1e-14)


@pytest.mark.parametrize(
    ("cls", "dim"),
    [(Quad1RScheme, 2), (Hex1RScheme, 3), (Tetra2Scheme, 3), (FourierScheme, 2)],
    ids=["Quad1R", "Hex1R", "Tetra2", "Fourier"],
)
def test_isotropic_symbol_is_the_exact_acoustic_tensor(cls, dim):
    """The closed form equals the true quadrature-averaged acoustic tensor.

    ``A = mean_r [ mu0 |a_r|^2 I + lam0 conj(a_r) a_r^T + mu0 a_r a_r^H ]``

    The closed form uses only support 0 and drops an antisymmetric
    ``i (lam0 - mu0) (g h^T - h g^T)``.  That term vanishes here -- ``g`` and
    ``h`` are parallel for the single-support schemes, and TETRA2's mirror
    relation cancels it across supports -- so using one support is exact rather
    than approximate.  ``lam0 != mu0`` below, or the dropped term would be zero
    by construction and the test vacuous.
    """
    space = SpectralSpace(
        lengths=(1.0,) * dim, shape=(N,) * dim, transform=FFTTransform(dim=dim)
    )
    D = cls(space=space).gradient_operator  # (*spatial, n_quads, dim)
    eye = jnp.eye(dim)

    norm_sq = jnp.sum(D * jnp.conj(D), axis=-1).real
    outer_c = jnp.conj(D)[..., :, None] * D[..., None, :]
    outer_h = D[..., :, None] * jnp.conj(D)[..., None, :]
    exact = (
        MU0 * norm_sq[..., None, None] * eye + LAM0 * outer_c + MU0 * outer_h
    ).mean(axis=-3)

    a1 = D[..., 0, :]
    g, h = a1.real, a1.imag
    gg = jnp.sum(g * g, -1) + jnp.sum(h * h, -1)
    outer = g[..., :, None] * g[..., None, :] + h[..., :, None] * h[..., None, :]
    closed = MU0 * gg[..., None, None] * eye + (LAM0 + MU0) * outer

    scale = float(jnp.max(jnp.abs(exact)))
    np.testing.assert_allclose(closed, exact, atol=1e-12 * scale)
