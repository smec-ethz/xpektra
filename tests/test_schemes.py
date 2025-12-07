import sys

import jax

jax.config.update("jax_enable_x64", True)  # use double-precision

import jax.numpy as jnp
import numpy as np
import pytest

from xpektra import SpectralSpace, make_field
from xpektra.scheme import (
    CentralDifference,
    EighthOrderCentralDifference,
    FourierScheme,
    FourthOrderCentralDifference,
    SixthOrderCentralDifference,
)
from xpektra.spectral_operator import SpectralOperator
from xpektra.transform import FFTTransform


def gaussian_field(N, length=1.0):
    """
    Generates a Gaussian pulse and its analytical derivative.
    The pulse is centered and narrow enough to vanish at boundaries.
    """
    mu = length / 2.0
    sigma = length / 20.0
    x = np.linspace(0, length, N, endpoint=False)

    f = np.exp(-((x - mu) ** 2) / (2 * sigma**2))
    f_prime = -((x - mu) / sigma**2) * f
    return f, f_prime


def compute_relative_error(N, SchemeClass):
    """
    Runs a simulation for a given N and Scheme, returning the L2 relative error.
    """
    length = 1.0

    # use dim=1 for testing convergence
    fft_transform = FFTTransform(dim=1)
    space = SpectralSpace(lengths=(length,), shape=(N,), transform=fft_transform)
    scheme = SchemeClass(space=space)
    op = SpectralOperator(scheme=scheme, space=space)

    # get Analytical Data
    f, f_prime_exact = gaussian_field(N, length)

    # compute Numerical Derivative
    # convert numpy array to JAX array for the library
    f_prime_num = op.grad(jnp.array(f))

    # compute Relative Error
    # relative_error = ||num - exact|| / ||exact||
    error = np.linalg.norm(f_prime_num - f_prime_exact) / np.linalg.norm(f_prime_exact)
    return error


@pytest.mark.parametrize(
    "SchemeClass, expected_order",
    [
        (CentralDifference, 2),
        (FourthOrderCentralDifference, 4),
        (SixthOrderCentralDifference, 6),
        (EighthOrderCentralDifference, 8),
    ],
)
def test_finite_difference_convergence(SchemeClass, expected_order):
    """
    Verifies that Finite Difference schemes converge at the expected polynomial rate.
    We check if the error drops by roughly 2^order when resolution doubles.
    """
    resolutions = [64, 128, 256]
    errors = []

    for N in resolutions:
        err = compute_relative_error(N, SchemeClass)
        errors.append(err)

    # calculate empirical convergence rates between consecutive resolutions
    # rate = log2(Error_coarse / Error_fine)
    measured_rates = []
    for i in range(len(errors) - 1):
        rate = np.log2(errors[i] / errors[i + 1])
        measured_rates.append(rate)

    avg_rate = np.mean(measured_rates)

    print(f"\nScheme: {SchemeClass.__name__}")
    print(f"Errors: {errors}")
    print(f"Measured Rates: {measured_rates}")
    print(f"Average Rate: {avg_rate:.2f} (Target: {expected_order})")

    # assert that the rate is close to the theoretical order.
    # allow a small tolerance (0.5) because asymptotic convergence
    # isn't perfect at finite N.
    assert avg_rate >= expected_order - 0.5, (
        f"{SchemeClass.__name__} converging too slowly! Expected ~{expected_order}, got {avg_rate:.2f}"
    )


def test_fourier_exactness():
    """
    Verifies that the Fourier scheme achieves machine precision
    for a band-limited function (sine wave).
    """
    N = 64
    length = 2 * np.pi

    # setup
    fft_transform = FFTTransform(dim=1)
    space = SpectralSpace(lengths=(length,), shape=(N,), transform=fft_transform)
    scheme = FourierScheme(space=space)
    op = SpectralOperator(scheme=scheme, space=space)

    # data: sin(x) -> cos(x)
    x = np.linspace(0, length, N, endpoint=False)
    f = np.sin(x)
    f_prime_exact = np.cos(x)

    # compute
    f_prime_num = op.grad(jnp.array(f))

    # check error
    error = np.linalg.norm(f_prime_num - f_prime_exact)

    print(f"\nFourier Error: {error}")
    # should be close to machine epsilon (e.g. 1e-15)
    assert error < 1e-13, "Fourier scheme should be exact for sin(x)"


def test_gradient_shapes():
    """
    Smoke test to ensure 1D and 2D gradients return the correct array shapes.
    (Verifies the fix for the (N,1) vs (N,) issue).
    """
    N = 32

    # 1D Case
    space1 = SpectralSpace(lengths=(1.0,), shape=(N,), transform=FFTTransform(dim=1))
    op1 = SpectralOperator(scheme=CentralDifference(space1), space=space1)
    u1 = make_field(dim=1, shape=(N,), rank=0)  # Shape (N,)
    grad1 = op1.grad(u1)
    assert grad1.shape == (N,), (
        f"1D Grad shape mismatch: got {grad1.shape}, expected {(N,)}"
    )

    # 2D Case
    space2 = SpectralSpace(
        lengths=(1.0, 1.0), shape=(N, N), transform=FFTTransform(dim=2)
    )
    op2 = SpectralOperator(scheme=CentralDifference(space2), space=space2)
    u2 = make_field(dim=2, shape=(N, N), rank=0)  # Shape (N, N)
    grad2 = op2.grad(u2)
    assert grad2.shape == (N, N, 2), (
        f"2D Grad shape mismatch: got {grad2.shape}, expected {(N, N, 2)}"
    )


if __name__ == "__main__":
    # allow running this script directly
    sys.exit(pytest.main(["-v", __file__]))
