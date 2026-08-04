import jax
import jax.numpy as jnp
import numpy as np

from xpektra.scheme import (
    ForwardDifference,
    ForwardScheme,
    Hex1RScheme,
    Quad1RScheme,
    RotatedDifference,
)
from xpektra.space import SpectralSpace
from xpektra.spectral_operator import SpectralOperator
from xpektra.transform import FFTTransform

jax.config.update("jax_enable_x64", True)  # use double-precision


def _space(N=64, length=1.0, dim=2):
    return SpectralSpace(
        lengths=(length,) * dim, shape=(N,) * dim, transform=FFTTransform(dim=dim)
    )


def test_stencil_matches_rotated_difference():
    """Stencil-built Fourier symbol reproduces the hand-coded RotatedDifference."""
    space = _space()
    k_vals = space.get_wavenumber_mesh()
    dx = space.lengths[0] / space.shape[0]
    dy = space.lengths[1] / space.shape[1]

    reference = RotatedDifference(space=space).gradient_operator

    quad_1r = Quad1RScheme(space=space)

    for axis, stencil in enumerate(quad_1r.stencils):
        _, Z_func = quad_1r.build_fourier_operator(stencil=stencil)
        Z = Z_func(k_vals[0], k_vals[1], dx, dy)
        np.testing.assert_allclose(Z, reference[..., axis], atol=1e-12)


def test_stencil_matches_rotated_difference_3d():
    N = 64
    space = _space(dim=3, N=N)
    k_vals = space.get_wavenumber_mesh()
    h_vals = [space.lengths[i] / space.shape[i] for i in range(3)]

    import time

    start_time = time.perf_counter()
    reference = RotatedDifference(space=space).gradient_operator
    end_time = time.perf_counter()
    print(
        f"Time taken to compute reference gradient operator: {end_time - start_time:.6f} seconds"
    )

    hex1r = Hex1RScheme(space=space)

    for axis, stencil in enumerate(hex1r.stencils):
        start_time = time.perf_counter()
        _, Z_func = hex1r.build_fourier_operator(stencil=stencil)
        end_time = time.perf_counter()
        print(
            f"Time taken to build Fourier operator for axis {axis}: {end_time - start_time:.6f} seconds"
        )
        start_time_eval = time.perf_counter()
        Z = Z_func(*k_vals, *h_vals)
        end_time_eval = time.perf_counter()
        print(
            f"Time taken to evaluate Fourier operator for axis {axis}: {end_time_eval - start_time_eval:.6f} seconds"
        )
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
    """Stencil-built Fourier symbol reproduces the hand-coded ForwardDifference."""

    space = _space()
    k_vals = space.get_wavenumber_mesh()
    dx = space.lengths[0] / space.shape[0]
    dy = space.lengths[1] / space.shape[1]

    reference = ForwardDifference(space=space).gradient_operator
    forward_scheme = ForwardScheme(space=space)

    for axis, stencil in enumerate(forward_scheme.stencils):
        _, Z_func = forward_scheme.build_fourier_operator(stencil=stencil)
        Z = Z_func(k_vals[0], k_vals[1], dx, dy)
        np.testing.assert_allclose(Z, reference[..., axis], atol=1e-12)


if __name__ == "__main__":
    # test_stencil_matches_rotated_difference()
    test_stencil_matches_rotated_difference_3d()
    # test_stencil_match_forward()
