import jax
import jax.numpy as jnp
import pytest

jax.config.update("jax_enable_x64", True)

from xpektra.extension import (
    extend_1d,
    extend_cell_field,
    get_extension_kind,
    mask_1d,
    restrict_1d,
    weights_1d,
)


def test_get_extension_kind():
    assert get_extension_kind(("dirichlet", "dirichlet")) == "odd"
    assert get_extension_kind(("neumann", "neumann")) == "even"
    assert get_extension_kind(("dirichlet", "neumann")) == "even"
    assert get_extension_kind(("neumann", "dirichlet")) == "even"
    assert get_extension_kind(("periodic", "periodic")) == "tile"


def test_mask_1d():
    # Dirichlet both ends: False at ends
    m_dd = mask_1d(5, ("dirichlet", "dirichlet"))
    assert jnp.array_equal(m_dd, jnp.array([False, True, True, True, False]))

    # Neumann both ends: all True
    m_nn = mask_1d(5, ("neumann", "neumann"))
    assert jnp.array_equal(m_nn, jnp.array([True, True, True, True, True]))

    # Mixed (D, N)
    m_dn = mask_1d(5, ("dirichlet", "neumann"))
    assert jnp.array_equal(m_dn, jnp.array([False, True, True, True, True]))

    # Mixed (N, D)
    m_nd = mask_1d(5, ("neumann", "dirichlet"))
    assert jnp.array_equal(m_nd, jnp.array([True, True, True, True, False]))

    # Periodic tile on doubled axis: False at wrap node (last element)
    m_pp = mask_1d(5, ("periodic", "periodic"))
    assert jnp.array_equal(m_pp, jnp.array([True, True, True, True, False]))


def test_weights_1d():
    # Odd & Even: [1, 2, 2, 2, 1]
    w_dd = weights_1d(5, ("dirichlet", "dirichlet"))
    assert jnp.allclose(w_dd, jnp.array([1.0, 2.0, 2.0, 2.0, 1.0]))

    w_nn = weights_1d(5, ("neumann", "neumann"))
    assert jnp.allclose(w_nn, jnp.array([1.0, 2.0, 2.0, 2.0, 1.0]))

    # Periodic tile: [2, 2, 2, 2, 0]
    w_pp = weights_1d(5, ("periodic", "periodic"))
    assert jnp.allclose(w_pp, jnp.array([2.0, 2.0, 2.0, 2.0, 0.0]))


def test_extend_1d_explicit_values():
    u = jnp.array([0.0, 1.0, 2.0, 3.0, 0.0])

    # Odd: [u, -flip(u_{1:N})] -> [0, 1, 2, 3, 0, -3, -2, -1]
    ext_odd = extend_1d(u, "odd")
    expected_odd = jnp.array([0.0, 1.0, 2.0, 3.0, 0.0, -3.0, -2.0, -1.0])
    assert jnp.allclose(ext_odd, expected_odd)

    # Even: [u, flip(u_{1:N})] -> [0, 1, 2, 3, 0, 3, 2, 1]
    ext_even = extend_1d(u, "even")
    expected_even = jnp.array([0.0, 1.0, 2.0, 3.0, 0.0, 3.0, 2.0, 1.0])
    assert jnp.allclose(ext_even, expected_even)

    # Tile: [u_{0:N}, u_{0:N}] -> [0, 1, 2, 3, 0, 1, 2, 3]
    ext_tile = extend_1d(u, "tile")
    expected_tile = jnp.array([0.0, 1.0, 2.0, 3.0, 0.0, 1.0, 2.0, 3.0])
    assert jnp.allclose(ext_tile, expected_tile)


@pytest.mark.parametrize("kind", ["odd", "even", "tile"])
@pytest.mark.parametrize("axis", [0, 1, -1])
def test_adjoint_property(kind, axis):
    """Verify <E u, f> == <u, Eᵀ f> for all kinds and across axes."""
    key = jax.random.PRNGKey(42)
    k1, k2 = jax.random.split(key)

    shape_u = [5, 6, 7]
    u = jax.random.normal(k1, tuple(shape_u))

    Eu = extend_1d(u, kind=kind, axis=axis)

    shape_f = list(Eu.shape)
    f = jax.random.normal(k2, tuple(shape_f))

    Et_f = restrict_1d(f, kind=kind, axis=axis)

    inner1 = jnp.sum(Eu * f)
    inner2 = jnp.sum(u * Et_f)

    assert jnp.allclose(inner1, inner2, atol=1e-12, rtol=1e-12)


@pytest.mark.parametrize("kind", ["odd", "even", "tile"])
def test_et_e_equals_weights(kind):
    """Verify Eᵀ(E u) == W * u for all modes."""
    key = jax.random.PRNGKey(101)
    n_nodes = 7
    u = jax.random.normal(key, (n_nodes,))

    Eu = extend_1d(u, kind=kind)
    Et_Eu = restrict_1d(Eu, kind=kind)

    bc = (
        ("dirichlet", "dirichlet")
        if kind == "odd"
        else (("periodic", "periodic") if kind == "tile" else ("neumann", "neumann"))
    )
    W = weights_1d(n_nodes, bc)

    assert jnp.allclose(Et_Eu, W * u, atol=1e-12)


@pytest.mark.parametrize("kind", ["odd", "even", "tile"])
def test_autodiff_matches_dual(kind):
    """JAX automatic reverse-mode VJP of extend_1d matches restrict_1d."""
    key = jax.random.PRNGKey(202)
    k1, k2 = jax.random.split(key)
    u = jax.random.normal(k1, (8,))
    Eu = extend_1d(u, kind=kind)
    f = jax.random.normal(k2, Eu.shape)

    def loss(x):
        return jnp.sum(extend_1d(x, kind=kind) * f)

    grad_u = jax.grad(loss)(u)
    manual_et = restrict_1d(f, kind=kind)

    assert jnp.allclose(grad_u, manual_et, atol=1e-12)


# ---------------------------------------------------------------------------
# Cell Field Extension: DCT-2 & Tensor Parity
# ---------------------------------------------------------------------------


def test_extend_cell_field_scalar():
    # 2D scalar field (3, 4), non-periodic along axis 1 (y)
    lam = jnp.arange(12, dtype=float).reshape(3, 4)
    lam_ext = extend_cell_field(lam, axes=(1,), rank=0)
    assert lam_ext.shape == (3, 8)
    # Physical slice unchanged
    assert jnp.allclose(lam_ext[:, :4], lam)
    # Reflected half is even mirror
    assert jnp.allclose(lam_ext[:, 4:], lam[:, ::-1])


def test_extend_cell_field_tensor_rank2_parity():
    # 2D strain tensor field (Nx=2, Ny=3, 2, 2)
    Nx, Ny = 2, 3
    eps = jnp.ones((Nx, Ny, 2, 2))
    # Mirror across y (axis 1)
    eps_ext = extend_cell_field(eps, axes=(1,), rank=2)
    assert eps_ext.shape == (Nx, 2 * Ny, 2, 2)

    # In reflected half:
    # eps_xx (indices 0, 0): even (+1)
    assert jnp.allclose(eps_ext[:, Ny:, 0, 0], 1.0)
    # eps_yy (indices 1, 1): even (+1, since (-1)^2 = +1)
    assert jnp.allclose(eps_ext[:, Ny:, 1, 1], 1.0)
    # eps_xy (indices 0, 1): odd (-1) -> flips sign!
    assert jnp.allclose(eps_ext[:, Ny:, 0, 1], -1.0)
    # eps_yx (indices 1, 0): odd (-1) -> flips sign!
    assert jnp.allclose(eps_ext[:, Ny:, 1, 0], -1.0)


def test_extend_cell_field_tensor_rank4_parity():
    # 2D stiffness tensor C_ijkl of shape (2, 2, 2, 2, 2, 2)
    Nx, Ny = 2, 2
    C = jnp.ones((Nx, Ny, 2, 2, 2, 2))
    C_ext = extend_cell_field(C, axes=(1,), rank=4)
    assert C_ext.shape == (Nx, 2 * Ny, 2, 2, 2, 2)

    # C_xxxx (0 ones) -> +1
    assert jnp.allclose(C_ext[:, Ny:, 0, 0, 0, 0], 1.0)
    # C_xxxy (1 one) -> -1
    assert jnp.allclose(C_ext[:, Ny:, 0, 0, 0, 1], -1.0)
    # C_xxyy (2 ones) -> +1
    assert jnp.allclose(C_ext[:, Ny:, 0, 0, 1, 1], 1.0)
    # C_xyyy (3 ones) -> -1
    assert jnp.allclose(C_ext[:, Ny:, 0, 1, 1, 1], -1.0)
    # C_yyyy (4 ones) -> +1
    assert jnp.allclose(C_ext[:, Ny:, 1, 1, 1, 1], 1.0)


def test_spectral_space_non_periodic_axes():
    from xpektra import FFTTransform, SpectralSpace

    space = SpectralSpace(
        lengths=(1.0, 2.0),
        shape=(32, 32),
        non_periodic_axes=("y",),
        transform=FFTTransform(dim=2),
    )
    assert space.shape == (32, 32)
    assert space.extended_shape == (32, 64)
    assert space.extended_lengths == (1.0, 4.0)
    assert space.dim == 2
    assert "y" in space.non_periodic_axes
    assert 1 in space.non_periodic_axes
    assert "x" not in space.non_periodic_axes
    assert space.physical_slices == (slice(None), slice(0, 32))

    # Reject integer axis inputs
    with pytest.raises(TypeError, match="Axis must be a string"):
        SpectralSpace(
            lengths=(1.0, 1.0),
            shape=(16, 16),
            transform=FFTTransform(dim=2),
            non_periodic_axes=(1,),
        )

    # Cell field extension directly via space
    lam = jnp.arange(32 * 32, dtype=float).reshape(32, 32)
    lam_ext = space.extend_cell_field(lam, rank=0)
    assert lam_ext.shape == (32, 64)
    assert jnp.allclose(lam_ext[:, :32], lam)
    assert jnp.allclose(lam_ext[:, 32:], lam[:, ::-1])


def test_spectral_space_pytree():
    from xpektra import FFTTransform, SpectralSpace

    space = SpectralSpace(
        lengths=(1.0, 2.0),
        shape=(32, 32),
        non_periodic_axes=("y",),
        transform=FFTTransform(dim=2),
    )
    leaves, treedef = jax.tree_util.tree_flatten(space)
    assert len(leaves) == 0  # pure metadata, zero array leaves
    space_rec = jax.tree_util.tree_unflatten(treedef, leaves)
    assert space == space_rec

