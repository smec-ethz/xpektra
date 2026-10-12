import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.fft import dst as scipy_dst
from xpektra.transform import DSTTransform, FFTTransform

jax.config.update("jax_enable_x64", True)


def test_dst_pytree():
    t = DSTTransform(dim=2, axes=(0, 1))
    leaves, treedef = jax.tree_util.tree_flatten(t)
    t_unflattened = jax.tree_util.tree_unflatten(treedef, leaves)
    assert t == t_unflattened


def test_dst_matches_scipy_1d():
    N = 15
    key = jax.random.PRNGKey(0)
    x = jax.random.normal(key, (N,))

    t = DSTTransform(dim=1, axes=0)
    x_hat = t.forward(x)

    # Scipy unnormalized type-1 DST uses factor of 2:
    # y_k = 2 * sum x_n sin(...)
    # Our DST-I is sum x_n sin(...) = 0.5 * scipy_dst(x, type=1)
    expected = 0.5 * scipy_dst(np.array(x), type=1)
    np.testing.assert_allclose(np.array(x_hat), expected, atol=1e-12, rtol=1e-12)


def test_dst_matches_scipy_2d():
    Nx, Ny = 11, 13
    key = jax.random.PRNGKey(1)
    x = jax.random.normal(key, (Nx, Ny))

    t = DSTTransform(dim=2, axes=(0, 1))
    x_hat = t.forward(x)

    # 2D DST-I via scipy
    expected = 0.25 * scipy_dst(scipy_dst(np.array(x), type=1, axis=0), type=1, axis=1)
    np.testing.assert_allclose(np.array(x_hat), expected, atol=1e-12, rtol=1e-12)


@pytest.mark.parametrize(
    "dim,axes,shape",
    [
        (1, 0, (16,)),
        (1, (0,), (23,)),
        (2, (0, 1), (12, 14)),
        (2, 0, (12, 14)),
        (2, 1, (12, 14)),
        (3, (0, 1, 2), (6, 8, 7)),
        (3, (0, 2), (6, 8, 7)),
        (3, 1, (6, 8, 7)),
    ],
)
def test_dst_forward_inverse_roundtrip(dim, axes, shape):
    key = jax.random.PRNGKey(42)
    x = jax.random.normal(key, shape)

    t = DSTTransform(dim=dim, axes=axes)
    x_hat = t.forward(x)
    recovered = t.inverse(x_hat)

    np.testing.assert_allclose(np.array(recovered), np.array(x), atol=1e-12, rtol=1e-12)


def test_dst_tensor_field():
    key = jax.random.PRNGKey(99)
    # 2D vector field with shape (Nx, Ny, 2)
    x = jax.random.normal(key, (10, 12, 2))

    t = DSTTransform(dim=2, axes=0)
    x_hat = t.forward(x)
    recovered = t.inverse(x_hat)

    np.testing.assert_allclose(np.array(recovered), np.array(x), atol=1e-12, rtol=1e-12)


def test_dst_jit():
    key = jax.random.PRNGKey(7)
    x = jax.random.normal(key, (8, 10))

    t = DSTTransform(dim=2, axes=(0, 1))

    fwd = jax.jit(t.forward)
    inv = jax.jit(t.inverse)

    x_hat = fwd(x)
    recovered = inv(x_hat)
    np.testing.assert_allclose(np.array(recovered), np.array(x), atol=1e-12, rtol=1e-12)


def test_wavenumbers():
    L = 2.0
    N = 16
    t = DSTTransform(dim=1, axes=0)
    k_vec = t.get_wavenumber_vector(size=N, length=L)

    expected = np.arange(1, N + 1) * (np.pi / L)
    np.testing.assert_allclose(np.array(k_vec), expected, atol=1e-12)

    # 2D mixed wavenumbers mesh
    t2 = DSTTransform(dim=2, axes=0)
    mesh = t2.get_wavenumber_mesh(shape=(N, 8), lengths=(L, 1.0))
    assert len(mesh) == 2
    assert mesh[0].shape == (N, 8)
    assert mesh[1].shape == (N, 8)


def test_spectral_space_extend_axes():
    from xpektra import SpectralSpace

    # 2D space with DST-1 along axis 1
    space = SpectralSpace(
        lengths=(1.0, 1.0),
        shape=(31, 31),
        extend_axes={1: "dst1"},
        transform=FFTTransform(dim=2),
    )
    assert space.shape == (31, 31)
    assert space.lengths == (1.0, 1.0)
    assert space.extended_shape == (31, 64)
    assert space.extended_lengths == (1.0, 2.0)

    # Fluctuation field (default extend_axes: dst1 along axis 1)
    u = jnp.ones((31, 31, 2))
    u_ext = space.extend_primal(u)
    assert u_ext.shape == (31, 64, 2)
    # Check boundary zeros: u_ext at y=0 (index 0) and y=32 (index N+1) must be 0
    np.testing.assert_allclose(u_ext[:, 0, :], 0.0)
    np.testing.assert_allclose(u_ext[:, 32, :], 0.0)

    # Dual restriction (adjoint)
    key = jax.random.PRNGKey(42)
    r_ext = jax.random.normal(key, (31, 64, 2))
    u_rand = jax.random.normal(key, (31, 31, 2))
    r_phys = space.restrict_dual(r_ext)
    assert r_phys.shape == (31, 31, 2)
    # Adjoint identity: <r_ext, E u> == <E^T r_ext, u>
    inner_ext = jnp.sum(r_ext * space.extend_primal(u_rand))
    inner_phys = jnp.sum(r_phys * u_rand)
    np.testing.assert_allclose(inner_ext, inner_phys, atol=1e-12, rtol=1e-12)

    # Material field (extend_material_field automatically uses dct2)
    lam = jnp.arange(31 * 32, dtype=float).reshape(31, 32)
    lam_ext = space.extend_material_field(lam)
    assert lam_ext.shape == (31, 64)
    # Check even reflection: lam_ext[:, 32:] == lam_ext[:, :32][:, ::-1]
    np.testing.assert_allclose(lam_ext[:, 32:], lam_ext[:, :32][:, ::-1])

    # PyTree flattening of space
    leaves, treedef = jax.tree_util.tree_flatten(space)
    space_rec = jax.tree_util.tree_unflatten(treedef, leaves)
    assert space == space_rec


def test_operator_shape_checks():
    from xpektra import SpectralOperator, SpectralSpace
    from xpektra.scheme import QuadFullScheme

    space = SpectralSpace(
        lengths=(1.0, 1.0),
        shape=(31, 31),
        extend_axes={1: "dst1"},
        transform=FFTTransform(dim=2),
    )
    scheme = QuadFullScheme(space=space)
    op = SpectralOperator(scheme=scheme, space=space)

    # 1. Unextended displacement passed to op.sym_grad / op.grad
    u_phys = jnp.ones((31, 31, 2))
    with pytest.raises(ValueError, match="extend_primal"):
        op.sym_grad(u_phys)

    with pytest.raises(ValueError, match="extend_primal"):
        op.grad(u_phys)

    # 2. Unextended material field passed to auto_vmap
    @op.auto_vmap(eps=2)
    def law(eps, lam):
        return lam * eps

    u_ext = space.extend_primal(u_phys)
    eps = op.sym_grad(u_ext)
    lam_phys = jnp.ones((31, 32))
    with pytest.raises(ValueError, match="extend_material_field"):
        law(eps, lam_phys)

    # 3. Unextended density passed to op.integrate
    density_phys = jnp.ones((31, 31, scheme.n_quads))
    with pytest.raises(ValueError, match="integrate received density with physical spatial shape"):
        op.integrate(density_phys)
