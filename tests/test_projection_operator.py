import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import pytest

from xpektra import SpectralSpace
from xpektra.projection_operator import GalerkinProjection, MoulinecSuquetProjection
from xpektra.scheme import FourierScheme
from xpektra.transform import FFTTransform


def make_space(dim, N=16):
    lengths = (1.0,) * dim
    shape = (N,) * dim
    transform = FFTTransform(dim=dim)
    return SpectralSpace(lengths=lengths, shape=shape, transform=transform)


# ---------------------------------------------------------------------------
# GalerkinProjection
# ---------------------------------------------------------------------------


class TestGalerkinProjection:
    def test_output_shape_2d(self):
        N, dim = 16, 2
        space = make_space(dim, N)
        scheme = FourierScheme(space=space)
        proj = GalerkinProjection(scheme=scheme)

        rng = jax.random.PRNGKey(0)
        sigma_hat = jax.random.normal(rng, (N, N, 1, dim, dim)) + 0j
        result = proj(sigma_hat)
        assert result.shape == (N, N, 1, dim, dim)

    @pytest.mark.parametrize("dim", [1, 2])
    def test_idempotency(self, dim):
        """project(project(x)) == project(x) to machine precision."""
        N = 16
        space = make_space(dim, N)
        scheme = FourierScheme(space=space)
        proj = GalerkinProjection(scheme=scheme)

        rng = jax.random.PRNGKey(1)
        shape = (N,) * dim + (scheme.n_quads, dim, dim)
        sigma_hat = jax.random.normal(rng, shape) + 0j

        once = proj(sigma_hat)
        twice = proj(once)
        np.testing.assert_allclose(
            np.abs(twice - once),
            0.0,
            atol=1e-12,
            err_msg="GalerkinProjection is not idempotent",
        )

    def test_zero_frequency_mode_is_zero(self):
        """A field with only the DC component projects to zero."""
        N, dim = 16, 2
        space = make_space(dim, N)
        scheme = FourierScheme(space=space)
        proj = GalerkinProjection(scheme=scheme)

        # Only the zero-frequency mode is nonzero
        sigma_hat = jnp.zeros((N, N, 1, dim, dim), dtype=complex)
        sigma_hat = sigma_hat.at[0, 0, 0, :, :].set(jnp.eye(dim))

        result = proj(sigma_hat)
        np.testing.assert_allclose(
            np.abs(result), 0.0, atol=1e-14, err_msg="DC mode should project to zero"
        )


# ---------------------------------------------------------------------------
# MoulinecSuquetProjection
# ---------------------------------------------------------------------------


class TestMoulinecSuquetProjection:
    def test_output_shape_2d(self):
        N, dim = 8, 2
        space = make_space(dim, N)
        proj = MoulinecSuquetProjection(lambda0=1.0, mu0=1.0, space=space)
        assert proj._operator.shape == (N, N, dim, dim, dim, dim)

    def test_output_shape_3d(self):
        N, dim = 4, 3
        space = make_space(dim, N)
        proj = MoulinecSuquetProjection(lambda0=1.0, mu0=1.0, space=space)
        assert proj._operator.shape == (N, N, N, dim, dim, dim, dim)

    def test_zero_frequency_mode(self):
        """The DC mode (zero wavenumber) of Ghat should be zero."""
        N, dim = 8, 2
        space = make_space(dim, N)
        proj = MoulinecSuquetProjection(lambda0=1.0, mu0=1.0, space=space)
        dc = proj._operator[0, 0, ...]
        np.testing.assert_allclose(
            np.abs(dc),
            0.0,
            atol=1e-14,
            err_msg="DC (zero-frequency) mode of Ghat must be zero",
        )

    def test_major_symmetry(self):
        """G_{khij} == G_{ijkh} (major symmetry of the Green's operator)."""
        N, dim = 8, 2
        space = make_space(dim, N)
        proj = MoulinecSuquetProjection(lambda0=1.0, mu0=1.0, space=space)
        Ghat = proj._operator
        G_khij = Ghat
        G_ijkh = jnp.einsum("...khij->...ijkh", Ghat)
        np.testing.assert_allclose(
            G_khij,
            G_ijkh,
            atol=1e-12,
            err_msg="Ghat does not satisfy major symmetry G_{khij} == G_{ijkh}",
        )

    def test_minor_symmetry_first_pair(self):
        """G_{khij} == G_{hkij} (symmetry in first index pair)."""
        N, dim = 8, 2
        space = make_space(dim, N)
        proj = MoulinecSuquetProjection(lambda0=1.0, mu0=1.0, space=space)
        Ghat = proj._operator
        G_khij = Ghat
        G_hkij = jnp.einsum("...khij->...hkij", Ghat)
        np.testing.assert_allclose(
            G_khij,
            G_hkij,
            atol=1e-12,
            err_msg="Ghat does not satisfy minor symmetry G_{khij} == G_{hkij}",
        )

    @pytest.mark.parametrize("lambda0,mu0", [(0.5, 1.0), (1.0, 0.5), (2.0, 3.0)])
    def test_material_parameters(self, lambda0, mu0):
        """Ghat can be computed for different material parameters without error."""
        N, dim = 8, 2
        space = make_space(dim, N)
        proj = MoulinecSuquetProjection(lambda0=lambda0, mu0=mu0, space=space)
        Ghat = proj._operator
        assert Ghat.shape == (N, N, dim, dim, dim, dim)
        assert jnp.all(jnp.isfinite(Ghat))

    def test_call_after_construction(self):
        """The operator is built in __init__, so the instance is callable at once."""
        N, dim = 8, 2
        space = make_space(dim, N)
        # no scheme needed: MS ignores the gradient operator entirely
        proj = MoulinecSuquetProjection(lambda0=1.0, mu0=1.0, space=space)

        rng = jax.random.PRNGKey(0)
        field_hat = jax.random.normal(rng, (N, N, 1, dim, dim)) + 0j
        result = proj(field_hat)
        assert result.shape == (N, N, 1, dim, dim)

    def test_survives_a_jit_boundary(self):
        """The precomputed operator crosses jit as a pytree child."""
        N, dim = 8, 2
        space = make_space(dim, N)
        proj = MoulinecSuquetProjection(lambda0=1.0, mu0=1.0, space=space)

        rng = jax.random.PRNGKey(0)
        field_hat = jax.random.normal(rng, (N, N, 1, dim, dim)) + 0j
        out = jax.jit(lambda p, f: p(f))(proj, field_hat)
        np.testing.assert_allclose(out, proj(field_hat), atol=1e-14)


# ---------------------------------------------------------------------------
# Multi-support behaviour -- the projector couples the quadrature points
# ---------------------------------------------------------------------------


class TestGalerkinMultiSupport:
    """The compatible subspace is ``eps_q,ij = D_q,j u_i`` for a *single-valued* u.

    Both properties below hold for any true projector onto that subspace, and
    together they pin the quadrature averaging: dropping the ``mean_q`` from either
    the numerator or ``norm_sq`` breaks idempotency, and using only one support's
    ``D`` breaks the fixed-point property.
    """

    @staticmethod
    def _schemes():
        from xpektra.scheme import Tetra2Scheme

        space = make_space(3, 8)
        return [FourierScheme(space=space), Tetra2Scheme(space=space)]

    def _field(self, scheme, N=8):
        shape = (N, N, N, scheme.n_quads, 3, 3)
        k1, k2 = jax.random.split(jax.random.PRNGKey(3))
        return jax.random.normal(k1, shape) + 1j * jax.random.normal(k2, shape)

    def test_idempotent(self):
        for scheme in self._schemes():
            proj = GalerkinProjection(scheme=scheme)
            f = self._field(scheme)
            once = proj(f)
            twice = proj(once)
            np.testing.assert_allclose(
                twice,
                once,
                atol=1e-12,
                err_msg=f"not idempotent for {type(scheme).__name__}",
            )

    def test_discrete_gradients_are_fixed_points(self):
        """A field already of the form ``D_q,j u_i`` must come back unchanged."""
        for scheme in self._schemes():
            proj = GalerkinProjection(scheme=scheme)
            k1, k2 = jax.random.split(jax.random.PRNGKey(7))
            u = jax.random.normal(k1, (8, 8, 8, 3)) + 1j * jax.random.normal(
                k2, (8, 8, 8, 3)
            )
            g = jnp.einsum("...qj,...i->...qij", scheme.gradient_operator, u)
            np.testing.assert_allclose(
                proj(g),
                g,
                atol=1e-12,
                err_msg=f"gradients are not fixed for {type(scheme).__name__}",
            )

    def test_shape_is_preserved(self):
        for scheme in self._schemes():
            f = self._field(scheme)
            assert GalerkinProjection(scheme=scheme)(f).shape == f.shape
