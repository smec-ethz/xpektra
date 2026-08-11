"""Adjointness and Laplacian definiteness for every finite-difference scheme.

These properties are what make the displacement formulation ``K = D^T C D``
symmetric positive semi-definite.  They are checked here rather than only through
``GalerkinProjection``, which conjugates internally and is therefore insensitive to
the half-voxel phase that ``divergence_operator`` corrects.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from xpektra.scheme import (
    BackwardScheme,
    CentralScheme,
    ForwardScheme,
    Hex1RScheme,
    Quad1RScheme,
    Tetra2Scheme,
)
from xpektra.space import SpectralSpace
from xpektra.transform import FFTTransform

jax.config.update("jax_enable_x64", True)

N = 8

# (factory, dim) -- Quad1R is 2D only, Hex1R and Tetra2 are 3D only.
SCHEMES = [
    (ForwardScheme, 3),
    (BackwardScheme, 3),
    (CentralScheme, 3),
    (Quad1RScheme, 2),
    (Hex1RScheme, 3),
    (Tetra2Scheme, 3),
]
IDS = [cls.__name__ for cls, _ in SCHEMES]


def _space(dim, n=N):
    return SpectralSpace(
        lengths=(1.0,) * dim, shape=(n,) * dim, transform=FFTTransform(dim=dim)
    )


def _random_fields(scheme, dim, seed=0):
    """A vector field and a *symmetric* tensor field, both in Fourier space.

    ``sig`` always carries a leading quadrature axis, even for single-support
    schemes, so the inner products below need no special-casing.
    """
    k1, k2 = jax.random.split(jax.random.PRNGKey(seed))
    u = jnp.fft.fftn(
        jax.random.normal(k1, (N,) * dim + (dim,)), axes=tuple(range(dim))
    )
    sig = jax.random.normal(k2, (scheme.n_quads,) + (N,) * dim + (dim, dim))
    sig = jnp.fft.fftn(
        0.5 * (sig + jnp.swapaxes(sig, -1, -2)), axes=tuple(range(1, dim + 1))
    )
    return u, sig


def _sym_grad(scheme, u_hat):
    """Symmetric gradient, always with a leading quadrature axis."""
    eps = scheme.apply_symmetric_gradient(u_hat)
    return eps[None] if scheme.n_quads == 1 else eps


def _divergence(scheme, sig_hat):
    """Divergence, taking the leading-quadrature-axis convention above."""
    return scheme.apply_divergence(sig_hat[0] if scheme.n_quads == 1 else sig_hat)


@pytest.mark.parametrize(("cls", "dim"), SCHEMES, ids=IDS)
def test_divergence_operator_is_negative_conjugate_of_gradient(cls, dim):
    """``div = -conj(grad)`` -- the discrete adjoint, Eq. (19)_2."""
    scheme = cls(space=_space(dim))
    np.testing.assert_allclose(
        scheme.divergence_operator, -jnp.conj(scheme.gradient_operator), atol=1e-14
    )


@pytest.mark.parametrize(("cls", "dim"), SCHEMES, ids=IDS)
def test_divergence_is_adjoint_of_symmetric_gradient(cls, dim):
    """``<div sigma, u> == -<sigma, eps(u)>`` for symmetric sigma.

    This is the property the strain-based Galerkin path never needed and the
    displacement formulation depends on.  It fails if ``apply_divergence``
    contracts against ``gradient_operator`` instead of ``divergence_operator``.
    """
    scheme = cls(space=_space(dim))
    u, sig = _random_fields(scheme, dim)

    lhs = jnp.vdot(_divergence(scheme, sig), u)
    rhs = -jnp.sum(jnp.conj(sig) * _sym_grad(scheme, u)) / scheme.n_quads

    np.testing.assert_allclose(lhs, rhs, rtol=1e-10)


@pytest.mark.parametrize(("cls", "dim"), SCHEMES, ids=IDS)
def test_laplacian_symbol_is_real_and_negative_semidefinite(cls, dim):
    """``lap = -||D||^2``: real, and <= 0 on every mode (Eq. 19_3 / 31)."""
    scheme = cls(space=_space(dim))
    ones = jnp.ones((N,) * dim)
    lap = scheme.apply_laplacian(ones)  # symbol, since u_hat == 1

    assert lap.shape == (N,) * dim
    np.testing.assert_allclose(lap.imag, 0.0, atol=1e-12)
    assert lap.real.max() <= 1e-12, f"positive modes: {(lap.real > 1e-12).sum()}"


@pytest.mark.parametrize(("cls", "dim"), SCHEMES, ids=IDS)
def test_constant_field_has_zero_gradient(cls, dim):
    """Stencil weights sum to zero, so a constant field differentiates to zero."""
    scheme = cls(space=_space(dim))
    const_hat = jnp.fft.fftn(jnp.ones((N,) * dim), axes=tuple(range(dim)))
    grad = jnp.fft.ifftn(
        scheme.apply_gradient(const_hat),
        axes=tuple(range(1, dim + 1)) if scheme.n_quads > 1 else tuple(range(dim)),
    )
    np.testing.assert_allclose(jnp.abs(grad).max(), 0.0, atol=1e-11)
