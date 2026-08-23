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

N = 9

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


# ---------------------------------------------------------------------------
# Reduced-integration schemes need an odd grid
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("cls", "dim"), [(Quad1RScheme, 2), (Hex1RScheme, 3)], ids=["Quad1R", "Hex1R"]
)
def test_even_grid_is_refused(cls, dim):
    """An even grid puts the Nyquist frequency on the grid, where the symbol is 0."""
    with pytest.raises(ValueError, match="odd grid"):
        cls(space=_space(dim, n=8))


@pytest.mark.parametrize(
    ("cls", "dim"), [(Quad1RScheme, 2), (Hex1RScheme, 3)], ids=["Quad1R", "Hex1R"]
)
def test_odd_grid_has_only_the_rigid_translation_null_mode(cls, dim):
    """The reason for the check: at odd N there is exactly one null mode.

    At ``N = 8`` Hex1R has 23 of them and the smallest non-zero symbol is at
    round-off; at ``N = 9`` it has 1 and the gap is ~5e-2 of the maximum.  Those
    spurious modes are undetermined, so a solve leaves them wherever the initial
    guess put them.
    """
    scheme = cls(space=_space(dim, n=9))
    norm = jnp.sqrt(jnp.sum(jnp.abs(scheme.gradient_operator) ** 2, axis=(-2, -1)))
    scale = float(jnp.max(norm))

    assert int(jnp.sum(norm < 1e-10 * scale)) == 1, "expected only the xi=0 mode"
    # and the spectral gap is real, not round-off
    assert float(jnp.sort(norm.reshape(-1))[1]) / scale > 1e-2


def test_the_check_is_specific_to_reduced_integration():
    """``FourierScheme`` has no such factor, so an even grid is fine for it."""
    from xpektra.scheme import FourierScheme

    scheme = FourierScheme(space=_space(3, n=8))  # must not raise
    norm = jnp.sqrt(jnp.sum(jnp.abs(scheme.gradient_operator) ** 2, axis=(-2, -1)))
    assert int(jnp.sum(norm < 1e-10 * float(jnp.max(norm)))) == 1


@pytest.mark.parametrize(
    ("cls", "dim"), [(Quad1RScheme, 2), (Hex1RScheme, 3)], ids=["Quad1R", "Hex1R"]
)
def test_even_grid_can_be_opted_into(cls, dim):
    """`q1_ringing_2d.py` studies the Nyquist null mode, so it needs an even grid."""
    scheme = cls(space=_space(dim, n=8), allow_even_grid=True)
    norm = jnp.sqrt(jnp.sum(jnp.abs(scheme.gradient_operator) ** 2, axis=(-2, -1)))
    # the spurious modes really are there -- that is what makes the opt-out useful
    assert int(jnp.sum(norm < 1e-10 * float(jnp.max(norm)))) > 1


def test_the_opt_out_survives_a_pytree_roundtrip():
    """It is aux_data, so a scheme crossing `jit` must not silently re-arm the check."""
    scheme = Quad1RScheme(space=_space(2, n=8), allow_even_grid=True)
    leaves, treedef = jax.tree_util.tree_flatten(scheme)
    assert jax.tree_util.tree_unflatten(treedef, leaves).allow_even_grid
