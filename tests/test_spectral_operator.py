import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import pytest

from xpektra import SpectralSpace, make_field
from xpektra.scheme import FourierScheme
from xpektra.spectral_operator import SpectralOperator
from xpektra.transform import FFTTransform


def make_operator(dim, N=32, length=1.0):
    lengths = (length,) * dim
    shape = (N,) * dim
    transform = FFTTransform(dim=dim)
    space = SpectralSpace(lengths=lengths, shape=shape, transform=transform)
    scheme = FourierScheme(space=space)
    return SpectralOperator(space=space, scheme=scheme), N, length


# ---------------------------------------------------------------------------
# forward / inverse roundtrip
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("dim", [1, 2])
def test_forward_inverse_roundtrip(dim):
    op, N, _ = make_operator(dim)
    rng = jax.random.PRNGKey(0)
    u = jax.random.normal(rng, (N,) * dim)
    recovered = op.inverse(op.forward(u))
    np.testing.assert_allclose(recovered, u, atol=1e-13)


# ---------------------------------------------------------------------------
# laplacian
# ---------------------------------------------------------------------------

def test_laplacian_1d():
    """laplacian(sin(kx)) == -k^2 sin(kx) (exact for Fourier scheme)."""
    op, N, L = make_operator(dim=1, N=64, length=2 * np.pi)
    k = 3.0
    x = np.linspace(0, L, N, endpoint=False)
    u = jnp.array(np.sin(k * x))
    lap_u = op.laplacian(u)
    expected = -(k**2) * np.sin(k * x)
    np.testing.assert_allclose(lap_u, expected, atol=1e-11)


def test_laplacian_2d():
    """laplacian(sin(kx)*sin(ky)) == -2k^2 sin(kx)*sin(ky)."""
    op, N, L = make_operator(dim=2, N=32, length=2 * np.pi)
    k = 2.0
    x = np.linspace(0, L, N, endpoint=False)
    X, Y = np.meshgrid(x, x, indexing="ij")
    u = jnp.array(np.sin(k * X) * np.sin(k * Y))
    lap_u = op.laplacian(u)
    expected = -2 * k**2 * np.sin(k * X) * np.sin(k * Y)
    np.testing.assert_allclose(lap_u, expected, atol=1e-10)


@pytest.mark.parametrize("dim", [1, 2])
def test_laplacian_shape(dim):
    op, N, _ = make_operator(dim)
    u = make_field(dim=dim, shape=(N,) * dim, rank=0)
    result = op.laplacian(u)
    assert result.shape == (N,) * dim


# ---------------------------------------------------------------------------
# div
# ---------------------------------------------------------------------------

def test_div_2d_shape():
    """div of a rank-2 tensor field returns a vector field."""
    op, N, _ = make_operator(dim=2)
    sigma = make_field(dim=2, shape=(N, N), rank=2)
    result = op.div(sigma)
    assert result.shape == (N, N, 2)


def test_div_2d_correctness():
    """
    sigma[j, i] = sin(2pi*x/L) * delta_{j,0} * delta_{i,0}
    => div(sigma)_i = sum_j d sigma_{ji} / dx_j
    => div(sigma)_0 = d sigma_{00} / dx_0 = (2pi/L) cos(2pi*x/L)
    => div(sigma)_1 = 0
    """
    op, N, L = make_operator(dim=2, N=64, length=1.0)
    x = np.linspace(0, L, N, endpoint=False)
    X, _ = np.meshgrid(x, x, indexing="ij")
    k = 2 * np.pi / L

    sigma = np.zeros((N, N, 2, 2))
    sigma[:, :, 0, 0] = np.sin(k * X)
    sigma = jnp.array(sigma)

    result = op.div(sigma)

    expected_0 = k * np.cos(k * X)
    expected_1 = np.zeros((N, N))

    np.testing.assert_allclose(result[..., 0], expected_0, atol=1e-10)
    np.testing.assert_allclose(result[..., 1], expected_1, atol=1e-10)


# ---------------------------------------------------------------------------
# sym_grad
# ---------------------------------------------------------------------------

def test_sym_grad_shape_2d():
    """sym_grad of a 2D vector field returns shape (N, N, 2, 2)."""
    op, N, _ = make_operator(dim=2)
    u = jnp.zeros((N, N, 2))
    result = op.sym_grad(u)
    assert result.shape == (N, N, 2, 2)


def test_sym_grad_symmetry_2d():
    """sym_grad output is symmetric: eps_ij == eps_ji."""
    op, N, _ = make_operator(dim=2, N=32)
    rng = jax.random.PRNGKey(42)
    u = jax.random.normal(rng, (N, N, 2))
    eps = op.sym_grad(u)
    np.testing.assert_allclose(eps[..., 0, 1], eps[..., 1, 0], atol=1e-12)


def test_sym_grad_1d_matches_grad():
    """In 1D, sym_grad of a scalar field matches grad (library treats (N,) as the 1D vector)."""
    op, N, L = make_operator(dim=1, N=64, length=2 * np.pi)
    k = 2.0
    x = np.linspace(0, L, N, endpoint=False)
    # In the 1D special case, sym_grad expects input shape (N,) and returns (N,)
    u = jnp.array(np.sin(k * x))
    eps = op.sym_grad(u)
    expected = k * np.cos(k * x)
    np.testing.assert_allclose(eps, expected, atol=1e-11)
