"""TETRA2-specific structure: the two supports, their mirror relation, and the
crossed pairing that follows from it.

Reference: Amouzou-adoun et al. (2026), §2.5.2, Eqs. (20)-(31).
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from xpektra.scheme import (
    Tetra2Scheme,
    tetra_t1_stencils,
    tetra_t2_stencils,
)
from xpektra.space import SpectralSpace
from xpektra.transform import FFTTransform

jax.config.update("jax_enable_x64", True)

N = 8


def _space(dim=3, n=N):
    return SpectralSpace(
        lengths=(1.0,) * dim, shape=(n,) * dim, transform=FFTTransform(dim=dim)
    )


@pytest.fixture
def scheme():
    return Tetra2Scheme(space=_space())


def _centre_phase(space):
    """exp(-i xi.h / 2): undoes the node->centre half shift the stencils carry."""
    h = [space.lengths[i] / space.shape[i] for i in range(3)]
    k = space.get_wavenumber_mesh()
    return jnp.exp(-0.5j * sum(k[i] * h[i] for i in range(3)))[..., None]


# --------------------------------------------------------------------------
# Stencil structure -- guards against transcription errors in Eqs. (20)-(22)
# --------------------------------------------------------------------------


def test_supports_are_the_two_vertex_parity_classes():
    """T1 uses only even-parity cube vertices, T2 only odd-parity ones.

    A single mistyped offset breaks this, and it is the error mode these
    hand-copied stencils are most prone to.
    """
    for stencils, parity in ((tetra_t1_stencils(), 0), (tetra_t2_stencils(), 1)):
        for direction in stencils:
            offsets = {offset for offset, _ in direction}
            assert len(offsets) == 4, "each tetrahedron has four vertices"
            assert all(sum(o) % 2 == parity for o in offsets), (
                f"offsets {offsets} are not all parity {parity}"
            )


def test_each_direction_uses_all_four_vertices():
    """Every derivative on a tetrahedron involves its four nodes exactly once."""
    for stencils in (tetra_t1_stencils(), tetra_t2_stencils()):
        vertices = {offset for offset, _ in stencils[0]}
        for direction in stencils:
            assert {offset for offset, _ in direction} == vertices


# --------------------------------------------------------------------------
# Fourier symbols
# --------------------------------------------------------------------------


def test_n_quads_and_operator_shape(scheme):
    assert scheme.n_quads == 2
    assert scheme.gradient_operator.shape == (2, N, N, N, 3)


def test_mirror_identity(scheme):
    """Eq. (29): referenced to the voxel centre, ``Z_T2 == -conj(Z_T1)``.

    Equivalent to the statement that T2 is T1 mirrored through the voxel faces.
    """
    phase = _centre_phase(scheme.space)
    z1, z2 = scheme.gradient_operator * phase
    np.testing.assert_allclose(z2, -jnp.conj(z1), atol=1e-12)


def test_cross_product_is_minus_modulus_squared(scheme):
    """``sum_m Z_T1,m Z_T2,m == -||Z_T1||^2`` -- real, negative, and a modulus.

    Pairing each support with itself instead gives ``Re(Z^2)``, which changes sign
    across roughly half the spectrum; that is the failure this pins down.
    """
    phase = _centre_phase(scheme.space)
    z1, z2 = scheme.gradient_operator * phase

    crossed = jnp.einsum("...m,...m->...", z1, z2)
    np.testing.assert_allclose(crossed.imag, 0.0, atol=1e-12)
    np.testing.assert_allclose(
        crossed.real, -jnp.sum(jnp.abs(z1) ** 2, axis=-1), atol=1e-12
    )

    straight = 0.5 * jnp.einsum("...m,...m->...", z1, z1) + 0.5 * jnp.einsum(
        "...m,...m->...", z2, z2
    )
    assert (straight.real > 1e-12).any(), (
        "self-pairing should be sign-indefinite; if it is not, this test is vacuous"
    )


def test_both_supports_are_first_order_consistent():
    """Each support reproduces ``i xi`` at low frequency (checks sign and the 1/2h).

    Uses a finer grid than the other tests so the lowest mode is well resolved:
    the leading error is ``O((xi h)^2)``, ~0.6% at N=16 and ~2.5% at N=8.
    """
    space = _space(n=16)
    scheme = Tetra2Scheme(space=space)
    k = space.get_wavenumber_mesh()
    ops = scheme.gradient_operator * _centre_phase(space)

    idx = (0, 1, 0)  # lowest non-zero mode, along axis 1
    for r in range(scheme.n_quads):
        for m in range(3):
            np.testing.assert_allclose(
                ops[r][idx][m], 1j * k[m][idx], atol=1e-12, rtol=1e-2
            )


# --------------------------------------------------------------------------
# Operator application
# --------------------------------------------------------------------------


def test_gradient_and_divergence_shapes(scheme):
    u_scalar = jnp.zeros((N, N, N))
    u_vector = jnp.zeros((N, N, N, 3))
    sigma = jnp.zeros((2, N, N, N, 3, 3))

    assert scheme.apply_gradient(u_scalar).shape == (2, N, N, N, 3)
    assert scheme.apply_symmetric_gradient(u_vector).shape == (2, N, N, N, 3, 3)
    assert scheme.apply_divergence(sigma).shape == (N, N, N, 3)
    assert scheme.apply_laplacian(u_scalar).shape == (N, N, N)
    assert scheme.apply_laplacian(u_vector).shape == (N, N, N, 3)


def test_divergence_rejects_unstacked_input(scheme):
    """A plain centre field would unpack along a *spatial* axis and return garbage."""
    with pytest.raises(ValueError, match="quadrature points"):
        scheme.apply_divergence(jnp.zeros((N, N, N, 3, 3)))


def test_rejects_non_3d():
    with pytest.raises(ValueError, match="only compatible with 3D"):
        Tetra2Scheme(space=_space(dim=2))


def test_pytree_roundtrip_preserves_type_and_operator(scheme):
    leaves, treedef = jax.tree_util.tree_flatten(scheme)
    restored = jax.tree_util.tree_unflatten(treedef, leaves)

    assert type(restored) is Tetra2Scheme
    assert restored.n_quads == 2
    np.testing.assert_array_equal(restored.gradient_operator, scheme.gradient_operator)


def test_usable_under_jit(scheme):
    """The scheme crosses a jit boundary as a pytree and still divides correctly."""

    @jax.jit
    def apply(s, sigma):
        return s.apply_divergence(sigma)

    key = jax.random.PRNGKey(0)
    sigma = jax.random.normal(key, (2, N, N, N, 3, 3))
    np.testing.assert_allclose(
        apply(scheme, sigma), scheme.apply_divergence(sigma), atol=1e-12
    )
