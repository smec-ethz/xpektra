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
        proj = GalerkinProjection()

        rng = jax.random.PRNGKey(0)
        sigma_hat = jax.random.normal(rng, (N, N, dim, dim)) + 0j
        result = proj.project(sigma_hat, scheme.gradient_operator, space)
        assert result.shape == (N, N, dim, dim)

    @pytest.mark.parametrize("dim", [1, 2])
    def test_idempotency(self, dim):
        """project(project(x)) == project(x) to machine precision."""
        N = 16
        space = make_space(dim, N)
        scheme = FourierScheme(space=space)
        proj = GalerkinProjection()

        rng = jax.random.PRNGKey(1)
        shape = (N,) * dim + (dim, dim)
        sigma_hat = jax.random.normal(rng, shape) + 0j

        grad_op = scheme.gradient_operator
        once = proj.project(sigma_hat, grad_op, space)
        twice = proj.project(once, grad_op, space)
        np.testing.assert_allclose(
            np.abs(twice - once), 0.0, atol=1e-12,
            err_msg="GalerkinProjection is not idempotent"
        )

    def test_zero_frequency_mode_is_zero(self):
        """A field with only the DC component projects to zero."""
        N, dim = 16, 2
        space = make_space(dim, N)
        scheme = FourierScheme(space=space)
        proj = GalerkinProjection()

        # Only the zero-frequency mode is nonzero
        sigma_hat = jnp.zeros((N, N, dim, dim), dtype=complex)
        sigma_hat = sigma_hat.at[0, 0, :, :].set(jnp.eye(dim))

        result = proj.project(sigma_hat, scheme.gradient_operator, space)
        np.testing.assert_allclose(
            np.abs(result), 0.0, atol=1e-14,
            err_msg="DC mode should project to zero"
        )



# ---------------------------------------------------------------------------
# MoulinecSuquetProjection
# ---------------------------------------------------------------------------

class TestMoulinecSuquetProjection:
    def test_output_shape_2d(self):
        N, dim = 8, 2
        space = make_space(dim, N)
        proj = MoulinecSuquetProjection(lambda0=1.0, mu0=1.0).build(space)
        assert proj._operator.shape == (N, N, dim, dim, dim, dim)

    def test_output_shape_3d(self):
        N, dim = 4, 3
        space = make_space(dim, N)
        proj = MoulinecSuquetProjection(lambda0=1.0, mu0=1.0).build(space)
        assert proj._operator.shape == (N, N, N, dim, dim, dim, dim)

    def test_zero_frequency_mode(self):
        """The DC mode (zero wavenumber) of Ghat should be zero."""
        N, dim = 8, 2
        space = make_space(dim, N)
        proj = MoulinecSuquetProjection(lambda0=1.0, mu0=1.0).build(space)
        dc = proj._operator[0, 0, ...]
        np.testing.assert_allclose(
            np.abs(dc), 0.0, atol=1e-14,
            err_msg="DC (zero-frequency) mode of Ghat must be zero"
        )

    def test_major_symmetry(self):
        """G_{khij} == G_{ijkh} (major symmetry of the Green's operator)."""
        N, dim = 8, 2
        space = make_space(dim, N)
        proj = MoulinecSuquetProjection(lambda0=1.0, mu0=1.0).build(space)
        Ghat = proj._operator
        G_khij = Ghat
        G_ijkh = jnp.einsum("...khij->...ijkh", Ghat)
        np.testing.assert_allclose(
            G_khij, G_ijkh, atol=1e-12,
            err_msg="Ghat does not satisfy major symmetry G_{khij} == G_{ijkh}"
        )

    def test_minor_symmetry_first_pair(self):
        """G_{khij} == G_{hkij} (symmetry in first index pair)."""
        N, dim = 8, 2
        space = make_space(dim, N)
        proj = MoulinecSuquetProjection(lambda0=1.0, mu0=1.0).build(space)
        Ghat = proj._operator
        G_khij = Ghat
        G_hkij = jnp.einsum("...khij->...hkij", Ghat)
        np.testing.assert_allclose(
            G_khij, G_hkij, atol=1e-12,
            err_msg="Ghat does not satisfy minor symmetry G_{khij} == G_{hkij}"
        )

    @pytest.mark.parametrize("lambda0,mu0", [(0.5, 1.0), (1.0, 0.5), (2.0, 3.0)])
    def test_material_parameters(self, lambda0, mu0):
        """Ghat can be computed for different material parameters without error."""
        N, dim = 8, 2
        space = make_space(dim, N)
        proj = MoulinecSuquetProjection(lambda0=lambda0, mu0=mu0).build(space)
        Ghat = proj._operator
        assert Ghat.shape == (N, N, dim, dim, dim, dim)
        assert jnp.all(jnp.isfinite(Ghat))

    def test_project_precomputed(self):
        """project() works after build()."""
        N, dim = 8, 2
        space = make_space(dim, N)
        scheme = FourierScheme(space=space)
        proj = MoulinecSuquetProjection(lambda0=1.0, mu0=1.0).build(space)

        rng = jax.random.PRNGKey(0)
        field_hat = jax.random.normal(rng, (N, N, dim, dim)) + 0j
        result = proj.project(field_hat, scheme.gradient_operator, space)
        assert result.shape == (N, N, dim, dim)

    def test_build_via_spectral_operator(self):
        """SpectralOperator.__post_init__ triggers build() automatically."""
        from xpektra.spectral_operator import SpectralOperator

        N, dim = 8, 2
        space = make_space(dim, N)
        scheme = FourierScheme(space=space)
        proj = MoulinecSuquetProjection(lambda0=1.0, mu0=1.0)
        assert proj._operator is None

        op = SpectralOperator(scheme=scheme, space=space, projection=proj)
        assert op.projection._operator is not None
        assert op.projection._operator.shape == (N, N, dim, dim, dim, dim)
