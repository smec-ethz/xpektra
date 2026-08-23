"""``op.auto_vmap`` and ``op.integrate``: the quadrature boundary.

A constitutive law is written for one quadrature point but the whole grid, so
inside it every field is ``(*spatial, *tensor)`` and nothing carries a
quadrature axis.  ``auto_vmap`` is what strips and restores that axis.

The failure it exists to prevent is not an exception.  A material field at
``(N, N)`` against a per-quadrature invariant at ``(N, N, 1)`` right-aligns to
``(N, N, N)`` -- no error, N times the memory, wrong physics.  Several tests
here assert that a mis-declared argument *raises* rather than broadcasting.
"""

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import pytest

from xpektra import SpectralSpace
from xpektra.linalg import contract, trace
from xpektra.scheme import Hex1RScheme, Quad1RScheme, Tetra2Scheme
from xpektra.spectral_operator import SpectralOperator
from xpektra.transform import FFTTransform

N = 8

# (scheme, dim) -- n_quads is 1 for the first two, 2 for Tetra2.
SCHEMES = [(Quad1RScheme, 2), (Hex1RScheme, 3), (Tetra2Scheme, 3)]
IDS = [c.__name__ for c, _ in SCHEMES]


def _op(cls, dim):
    space = SpectralSpace(
        lengths=(1.0,) * dim, shape=(N,) * dim, transform=FFTTransform(dim=dim)
    )
    return SpectralOperator(scheme=cls(space=space), space=space)


def _fields(op, dim, seed=0):
    nq = op.scheme.n_quads
    k = jax.random.split(jax.random.PRNGKey(seed), 4)
    eps = jax.random.normal(k[0], (N,) * dim + (nq, dim, dim))
    eps = 0.5 * (eps + jnp.swapaxes(eps, -1, -2))
    lam = jnp.abs(jax.random.normal(k[1], (N,) * dim)) + 0.5
    mu = jnp.abs(jax.random.normal(k[2], (N,) * dim)) + 0.5
    alpha = jax.random.uniform(k[3], (N,) * dim)
    return eps, lam, mu, alpha


# ---------------------------------------------------------------------------
# auto_vmap
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(("cls", "dim"), SCHEMES, ids=IDS)
def test_maps_only_the_quadrature_axis(cls, dim):
    """Inside the law, arguments have no quadrature axis; outside, they do."""
    op = _op(cls, dim)
    eps, lam, mu, alpha = _fields(op, dim)
    seen = {}

    @op.auto_vmap(eps=2)
    def density(eps, alpha, lam, mu):
        seen["eps"] = eps.shape
        seen["lam"] = lam.shape
        return lam * trace(eps) ** 2 + mu * alpha

    out = density(eps, alpha, lam, mu)
    assert seen["eps"] == (N,) * dim + (dim, dim)  # quadrature axis stripped
    assert seen["lam"] == (N,) * dim  # per-voxel, broadcast not mapped
    assert out.shape == (N,) * dim + (op.scheme.n_quads,)


@pytest.mark.parametrize(("cls", "dim"), SCHEMES, ids=IDS)
def test_matches_the_hand_written_form(cls, dim):
    """The mapped law equals the same algebra written over the whole field."""
    op = _op(cls, dim)
    eps, lam, mu, alpha = _fields(op, dim)

    @op.auto_vmap(eps=2)
    def density(eps, alpha, lam, mu):
        return (
            lam * trace(eps) ** 2 + mu * contract("...ij,...ji->...", eps, eps) * alpha
        )

    # by hand: the material fields need the quadrature axis added explicitly
    manual = (
        lam[..., None] * trace(eps) ** 2
        + mu[..., None] * contract("...ij,...ji->...", eps, eps) * alpha[..., None]
    )

    np.testing.assert_allclose(density(eps, alpha, lam, mu), manual, atol=1e-13)


@pytest.mark.parametrize(("cls", "dim"), SCHEMES, ids=IDS)
def test_gradients_match_the_hand_written_form(cls, dim):
    """Newton differentiates through this, so the derivative must agree too."""
    op = _op(cls, dim)
    eps, lam, mu, _ = _fields(op, dim)

    @op.auto_vmap(eps=2)
    def density(eps, lam, mu):
        return lam * trace(eps) ** 2 + mu * contract("...ij,...ji->...", eps, eps)

    def manual(e):
        return (
            lam[..., None] * trace(e) ** 2
            + mu[..., None] * contract("...ij,...ji->...", e, e)
        ).sum()

    g_auto = jax.grad(lambda e: density(e, lam, mu).sum())(eps)
    np.testing.assert_allclose(g_auto, jax.grad(manual)(eps), atol=1e-12)


@pytest.mark.parametrize(("cls", "dim"), SCHEMES, ids=IDS)
def test_tensor_valued_law_keeps_the_axis_in_place(cls, dim):
    """A law returning a stress puts the quadrature axis back where it was.

    ``out_axes`` must be ``len(spatial)``, not ``-1``; those coincide only for a
    scalar-valued law.
    """
    op = _op(cls, dim)
    eps, lam, mu, _ = _fields(op, dim)

    @op.auto_vmap(eps=2)
    def stress(eps, lam, mu):
        return (lam * trace(eps))[..., None, None] * jnp.eye(dim) + 2 * mu[
            ..., None, None
        ] * eps

    assert stress(eps, lam, mu).shape == (N,) * dim + (op.scheme.n_quads, dim, dim)


@pytest.mark.parametrize(("cls", "dim"), SCHEMES, ids=IDS)
def test_rank_0_centre_field_is_detected(cls, dim):
    """A per-quadrature scalar needs no declaration: rank 0 is the default."""
    op = _op(cls, dim)
    nq = op.scheme.n_quads
    psi = jax.random.normal(jax.random.PRNGKey(5), (N,) * dim + (nq,))
    lam = jnp.ones((N,) * dim)

    @op.auto_vmap()
    def scaled(psi, lam):
        assert psi.shape == (N,) * dim  # mapped
        return psi * lam

    assert scaled(psi, lam).shape == (N,) * dim + (nq,)


@pytest.mark.parametrize(("cls", "dim"), SCHEMES, ids=IDS)
def test_undeclared_rank_raises_instead_of_broadcasting(cls, dim):
    """The whole point: a missing declaration must not silently broadcast."""
    op = _op(cls, dim)
    eps, lam, _, _ = _fields(op, dim)

    @op.auto_vmap()  # eps is rank 2 but left at the default 0
    def density(eps, lam):
        return trace(eps) * lam

    with pytest.raises(ValueError, match="Declare its rank"):
        density(eps, lam)


@pytest.mark.parametrize(("cls", "dim"), SCHEMES, ids=IDS)
def test_wrong_rank_raises(cls, dim):
    op = _op(cls, dim)
    eps, lam, _, _ = _fields(op, dim)

    @op.auto_vmap(eps=1)
    def density(eps, lam):
        return trace(eps) * lam

    with pytest.raises(ValueError, match="Declare its rank"):
        density(eps, lam)


def test_rank_for_an_unknown_parameter_is_rejected():
    op = _op(Hex1RScheme, 3)
    with pytest.raises(TypeError, match="does not take"):

        @op.auto_vmap(sigma=2)
        def density(eps, lam):
            return eps * lam


@pytest.mark.parametrize(("cls", "dim"), SCHEMES, ids=IDS)
def test_non_array_arguments_pass_through(cls, dim):
    """Python scalars and other non-arrays are never mapped."""
    op = _op(cls, dim)
    eps, _, _, _ = _fields(op, dim)

    @op.auto_vmap(eps=2)
    def density(eps, factor):
        return factor * trace(eps)

    assert density(eps, 2.5).shape == (N,) * dim + (op.scheme.n_quads,)


@pytest.mark.parametrize(("cls", "dim"), SCHEMES, ids=IDS)
def test_law_with_no_centre_field_is_called_directly(cls, dim):
    """Nothing to map: the function runs untouched, with no quadrature axis."""
    op = _op(cls, dim)
    _, lam, mu, _ = _fields(op, dim)

    @op.auto_vmap()
    def density(lam, mu):
        return lam + mu

    assert density(lam, mu).shape == (N,) * dim


# ---------------------------------------------------------------------------
# integrate
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(("cls", "dim"), SCHEMES, ids=IDS)
def test_integrate_averages_quadrature_and_sums_space(cls, dim):
    op = _op(cls, dim)
    nq = op.scheme.n_quads
    density = jnp.ones((N,) * dim + (nq,))
    # mean over quadrature is 1, summed over N**dim voxels
    np.testing.assert_allclose(op.integrate(density), float(N**dim), rtol=1e-13)


@pytest.mark.parametrize(("cls", "dim"), SCHEMES, ids=IDS)
def test_integrate_uses_the_mean_not_the_sum(cls, dim):
    """The ``1/n_quads`` is what pairs with ``apply_divergence``.

    With ``sum`` the total would scale with ``n_quads`` and ``jax.grad`` of an
    energy would no longer equal the discrete divergence of its stress.
    """
    op = _op(cls, dim)
    nq = op.scheme.n_quads
    density = jnp.ones((N,) * dim + (nq,))
    assert float(op.integrate(density)) == pytest.approx(N**dim)  # not nq * N**dim


@pytest.mark.parametrize(("cls", "dim"), SCHEMES, ids=IDS)
def test_integrate_preserves_tensor_axes(cls, dim):
    op = _op(cls, dim)
    nq = op.scheme.n_quads
    field = jnp.ones((N,) * dim + (nq, dim, dim))
    assert op.integrate(field).shape == (dim, dim)


@pytest.mark.parametrize(("cls", "dim"), SCHEMES, ids=IDS)
def test_integrate_rejects_a_missing_quadrature_axis(cls, dim):
    op = _op(cls, dim)
    with pytest.raises(ValueError, match="quadrature axis"):
        op.integrate(jnp.ones((N,) * dim))


@pytest.mark.parametrize(("cls", "dim"), SCHEMES, ids=IDS)
def test_uniform_constants_are_broadcast(cls, dim):
    """A 0-d array is a valid material constant, not a malformed field.

    A homogeneous reference passes ``jnp.asarray(mu0)`` where a heterogeneous
    problem passes a ``(*spatial,)`` field.  It has too few axes to be carrying
    a quadrature axis, so broadcasting is the only available reading.
    """
    op = _op(cls, dim)
    eps, _, _, _ = _fields(op, dim)

    @op.auto_vmap(eps=2)
    def density(eps, lam, mu):
        return lam * trace(eps) ** 2 + mu * trace(eps)

    uniform = density(eps, jnp.asarray(1.3), jnp.asarray(0.7))
    assert uniform.shape == (N,) * dim + (op.scheme.n_quads,)

    # a field of that same constant must give the same answer
    ones = jnp.ones((N,) * dim)
    np.testing.assert_allclose(
        uniform, density(eps, 1.3 * ones, 0.7 * ones), atol=1e-13
    )


@pytest.mark.parametrize(("cls", "dim"), SCHEMES, ids=IDS)
def test_broadcasting_scalars_does_not_weaken_the_guard(cls, dim):
    """Allowing under-shaped arguments must not let an undeclared field through."""
    op = _op(cls, dim)
    eps, lam, _, _ = _fields(op, dim)

    @op.auto_vmap()  # eps is rank 2 but left at the default 0
    def density(eps, lam):
        return trace(eps) * lam

    with pytest.raises(ValueError, match="Declare its rank"):
        density(eps, lam)
