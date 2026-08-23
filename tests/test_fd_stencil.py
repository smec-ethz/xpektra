import jax
import jax.numpy as jnp
import numpy as np

from xpektra.scheme import (
    ForwardScheme,
    Hex1RScheme,
    Quad1RScheme,
)
from xpektra.space import SpectralSpace
from xpektra.spectral_operator import SpectralOperator
from xpektra.transform import FFTTransform

jax.config.update("jax_enable_x64", True)  # use double-precision


def _space(N=65, length=1.0, dim=2):
    return SpectralSpace(
        lengths=(length,) * dim, shape=(N,) * dim, transform=FFTTransform(dim=dim)
    )


def _spacings(space):
    return [space.lengths[i] / space.shape[i] for i in range(len(space.lengths))]


def _rotated_reference(space):
    """Closed-form Willot symbol, ``D_i = 2i tan(xi_i h_i / 2) / h_i * prod_j f_j``.

    Kept as an independent hand-written formula so the stencil machinery is
    still checked against something other than itself.
    """
    k_vals = space.get_wavenumber_mesh()
    h_vals = _spacings(space)

    factor = 1.0
    for j, h in enumerate(h_vals):
        factor = factor * 0.5 * (1 + jnp.exp(1j * k_vals[j] * h))

    return jnp.stack(
        [2j * jnp.tan(k_vals[i] * h / 2) * factor / h for i, h in enumerate(h_vals)],
        axis=-1,
    )


def _forward_reference(space):
    """Closed-form forward-difference symbol, ``D_i = (exp(i xi_i h_i) - 1) / h_i``."""
    k_vals = space.get_wavenumber_mesh()
    h_vals = _spacings(space)
    return jnp.stack(
        [(jnp.exp(1j * k_vals[i] * h) - 1) / h for i, h in enumerate(h_vals)],
        axis=-1,
    )


def test_stencil_matches_rotated_difference():
    """Stencil-built Fourier symbol reproduces the closed-form Willot symbol."""
    space = _space()
    k_vals = space.get_wavenumber_mesh()
    dx = space.lengths[0] / space.shape[0]
    dy = space.lengths[1] / space.shape[1]

    reference = _rotated_reference(space)

    quad_1r = Quad1RScheme(space=space)

    for axis, stencil in enumerate(quad_1r.stencils):
        _, Z_func = quad_1r.build_fourier_operator(stencil=stencil)
        Z = Z_func(k_vals[0], k_vals[1], dx, dy)
        np.testing.assert_allclose(Z, reference[..., axis], atol=1e-12)


def test_stencil_matches_rotated_difference_3d():
    N = 65
    space = _space(dim=3, N=N)
    k_vals = space.get_wavenumber_mesh()
    h_vals = [space.lengths[i] / space.shape[i] for i in range(3)]

    reference = _rotated_reference(space)

    hex1r = Hex1RScheme(space=space)

    for axis, stencil in enumerate(hex1r.stencils):
        _, Z_func = hex1r.build_fourier_operator(stencil=stencil)
        Z = Z_func(*k_vals, *h_vals)
        np.testing.assert_allclose(Z, reference[..., axis], atol=1e-12)

    op = SpectralOperator(scheme=hex1r, space=space)

    # get Analytical Data
    # f, f_prime_exact = gaussian_field(N, length)
    rng = jax.random.PRNGKey(0)
    u = jax.random.normal(rng, (N,) * 3)
    recovered = op.inverse(op.forward(u))
    np.testing.assert_allclose(recovered, u, atol=1e-13)

    op.grad(u)
    op.laplacian(u)


def test_stencil_match_forward():
    """Stencil-built Fourier symbol reproduces the closed-form forward difference."""

    space = _space()
    k_vals = space.get_wavenumber_mesh()
    dx = space.lengths[0] / space.shape[0]
    dy = space.lengths[1] / space.shape[1]

    reference = _forward_reference(space)
    forward_scheme = ForwardScheme(space=space)

    for axis, stencil in enumerate(forward_scheme.stencils):
        _, Z_func = forward_scheme.build_fourier_operator(stencil=stencil)
        Z = Z_func(k_vals[0], k_vals[1], dx, dy)
        np.testing.assert_allclose(Z, reference[..., axis], atol=1e-12)


if __name__ == "__main__":
    # test_stencil_matches_rotated_difference()
    test_stencil_matches_rotated_difference_3d()
    # test_stencil_match_forward()
