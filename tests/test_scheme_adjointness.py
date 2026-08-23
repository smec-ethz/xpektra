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

    ``sig`` follows the library layout ``(*spatial, n_quads, dim, dim)``.  Because
    the quadrature axis is trailing, the FFT is the same ``axes=range(dim)`` call
    whatever ``n_quads`` is, and the operators below need no special-casing.
    """
    k1, k2 = jax.random.split(jax.random.PRNGKey(seed))
    u = jnp.fft.fftn(jax.random.normal(k1, (N,) * dim + (dim,)), axes=tuple(range(dim)))
    q = (scheme.n_quads,)
    sig = jax.random.normal(k2, (N,) * dim + q + (dim, dim))
    sig = jnp.fft.fftn(0.5 * (sig + jnp.swapaxes(sig, -1, -2)), axes=tuple(range(dim)))
    return u, sig


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

    lhs = jnp.vdot(scheme.apply_divergence(sig), u)
    rhs = -jnp.sum(jnp.conj(sig) * scheme.apply_symmetric_gradient(u)) / scheme.n_quads

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
        axes=tuple(range(dim)),
    )
    np.testing.assert_allclose(jnp.abs(grad).max(), 0.0, atol=1e-11)


# ---------------------------------------------------------------------------
# Gradient of a higher-rank field -- index convention
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(("cls", "dim"), SCHEMES, ids=IDS)
def test_gradient_of_vector_shape(cls, dim):
    """``grad`` grows a ``(n_quads, dim)`` pair in front of the field's own axes."""
    scheme = cls(space=_space(dim))
    u = jnp.zeros((N,) * dim + (dim,), dtype=complex)
    assert scheme.apply_gradient(u).shape == (N,) * dim + (scheme.n_quads, dim, dim)

    # rank 0 and rank 2 follow the same rule
    a = jnp.zeros((N,) * dim, dtype=complex)
    assert scheme.apply_gradient(a).shape == (N,) * dim + (scheme.n_quads, dim)
    t = jnp.zeros((N,) * dim + (dim, dim), dtype=complex)
    assert scheme.apply_gradient(t).shape == (N,) * dim + (
        scheme.n_quads,
        dim,
        dim,
        dim,
    )


@pytest.mark.parametrize(("cls", "dim"), SCHEMES, ids=IDS)
def test_symmetric_gradient_is_symmetrised_gradient(cls, dim):
    """``sym_grad(u) == 0.5 * (g + g^T)`` -- pins the derivative-first convention.

    Under the transposed (continuum-mechanics) convention ``apply_gradient``
    would return ``g^T``, which still satisfies this identity; what it would
    *not* satisfy is the derivative index sitting on the axis that
    ``apply_divergence`` contracts.  That is covered by
    :func:`test_divergence_of_gradient_is_laplacian`.
    """
    scheme = cls(space=_space(dim))
    k = jax.random.PRNGKey(11)
    u = jnp.fft.fftn(jax.random.normal(k, (N,) * dim + (dim,)), axes=tuple(range(dim)))

    g = scheme.apply_gradient(u)
    np.testing.assert_allclose(
        scheme.apply_symmetric_gradient(u),
        0.5 * (g + jnp.swapaxes(g, -1, -2)),
        atol=1e-12,
    )


@pytest.mark.parametrize(("cls", "dim"), SCHEMES, ids=IDS)
def test_divergence_of_gradient_is_laplacian(cls, dim):
    """``div(grad(u)) == laplacian(u)`` component-wise, for a vector field.

    This is the property that actually fixes the index order: it holds only if
    the derivative index of ``grad`` lands on axis ``-2``, the one
    ``apply_divergence`` contracts.  With the transposed convention this would
    compute ``grad(div u)`` instead.
    """
    scheme = cls(space=_space(dim))
    k = jax.random.PRNGKey(13)
    u = jnp.fft.fftn(jax.random.normal(k, (N,) * dim + (dim,)), axes=tuple(range(dim)))

    np.testing.assert_allclose(
        scheme.apply_divergence(scheme.apply_gradient(u)),
        scheme.apply_laplacian(u),
        atol=1e-12,
    )
