import jax
import jax.numpy as jnp
import numpy as np
import sympy as sp

from xpektra.scheme import Hex1R, Quad1R, RotatedDifference
from xpektra.space import SpectralSpace
from xpektra.transform import FFTTransform

jax.config.update("jax_enable_x64", True)  # use double-precision


def _space(N=64, length=1.0, dim=2):
    return SpectralSpace(
        lengths=(length,) * dim, shape=(N,) * dim, transform=FFTTransform(dim=dim)
    )


def test_stencil_matches_rotated_difference():
    """Stencil-built Fourier symbol reproduces the hand-coded RotatedDifference."""

    h1, h2 = sp.symbols("h_1 h_2", real=True)

    willot_dx_stencil = [
        ((1, 1), 1 / (2 * h1)),
        ((0, 1), -1 / (2 * h1)),
        ((1, 0), 1 / (2 * h1)),
        ((0, 0), -1 / (2 * h1)),
    ]

    willot_dy_stencil = [
        ((1, 1), 1 / (2 * h2)),
        ((1, 0), -1 / (2 * h2)),
        ((0, 1), 1 / (2 * h2)),
        ((0, 0), -1 / (2 * h2)),
    ]

    space = _space()
    k_vals = space.get_wavenumber_mesh()
    dx = space.lengths[0] / space.shape[0]
    dy = space.lengths[1] / space.shape[1]

    reference = RotatedDifference(space=space).gradient_operator

    quad_1r = Quad1R(space=space)

    for axis, stencil in enumerate([willot_dx_stencil, willot_dy_stencil]):
        Z_sym, Z_func = quad_1r.build_fourier_operator(stencil=stencil)
        Z = Z_func(k_vals[0], k_vals[1], dx, dy)
        np.testing.assert_allclose(Z, reference[..., axis], atol=1e-12)


def test_stencil_matches_rotated_difference_3d():
    h1, h2, h3 = sp.symbols("h_1 h_2 h_3", real=True)
    willot_dx_stencil = [
        ((1, 0, 0), 1 / (4 * h1)),
        ((0, 0, 0), -1 / (4 * h1)),
        ((1, 1, 0), 1 / (4 * h1)),
        ((0, 1, 0), -1 / (4 * h1)),
        ((1, 0, 1), 1 / (4 * h1)),
        ((0, 0, 1), -1 / (4 * h1)),
        ((1, 1, 1), 1 / (4 * h1)),
        ((0, 1, 1), -1 / (4 * h1)),
    ]

    willot_dy_stencil = [
        ((0, 1, 0), 1 / (4 * h2)),
        ((0, 0, 0), -1 / (4 * h2)),
        ((1, 1, 0), 1 / (4 * h2)),
        ((1, 0, 0), -1 / (4 * h2)),
        ((0, 1, 1), 1 / (4 * h2)),
        ((0, 0, 1), -1 / (4 * h2)),
        ((1, 1, 1), 1 / (4 * h2)),
        ((1, 0, 1), -1 / (4 * h2)),
    ]

    willot_dz_stencil = [
        ((0, 0, 1), 1 / (4 * h3)),
        ((0, 0, 0), -1 / (4 * h3)),
        ((1, 0, 1), 1 / (4 * h3)),
        ((1, 0, 0), -1 / (4 * h3)),
        ((0, 1, 1), 1 / (4 * h3)),
        ((0, 1, 0), -1 / (4 * h3)),
        ((1, 1, 1), 1 / (4 * h3)),
        ((1, 1, 0), -1 / (4 * h3)),
    ]

    space = _space(dim=3)
    k_vals = space.get_wavenumber_mesh()
    h_vals = [space.lengths[i] / space.shape[i] for i in range(3)]

    import time

    start_time = time.perf_counter()
    reference = RotatedDifference(space=space).gradient_operator
    end_time = time.perf_counter()
    print(
        f"Time taken to compute reference gradient operator: {end_time - start_time:.6f} seconds"
    )

    hex1r = Hex1R(space=space)

    for axis, stencil in enumerate(
        [willot_dx_stencil, willot_dy_stencil, willot_dz_stencil]
    ):
        start_time = time.perf_counter()
        Z_sym, Z_func = hex1r.build_fourier_operator(stencil=stencil)
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


if __name__ == "__main__":
    test_stencil_matches_rotated_difference()
    test_stencil_matches_rotated_difference_3d()
