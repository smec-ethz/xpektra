import jax
import jax.numpy as jnp
import pytest

jax.config.update("jax_enable_x64", True)

from xpektra.solution_space import Field, FieldView, SolutionSpace, State

from xpektra import FFTTransform, SpectralSpace


def test_field_shape_and_rank():
    scalar = Field(shape=())
    assert scalar.shape == ()
    assert scalar.rank == 0
    assert scalar.n_components == 1

    vec = Field(shape=(2,))
    assert vec.shape == (2,)
    assert vec.rank == 1
    assert vec.n_components == 2

    tensor = Field(shape=(3, 3))
    assert tensor.shape == (3, 3)
    assert tensor.rank == 2
    assert tensor.n_components == 9

    # Rejections
    with pytest.raises(ValueError, match=r"shape=\(\) instead of shape=\(1,\)"):
        Field(shape=(1,))

    with pytest.raises(TypeError):
        Field(shape=2)  # type: ignore[arg-type]

    with pytest.raises(ValueError):
        Field(shape=(-2,))


def test_field_defaults():
    f = Field(shape=())
    assert f.name == "field"
    assert f.bcs["x"] == (("periodic", "periodic"),)
    assert f.bcs["y"] == (("periodic", "periodic"),)
    assert f.bcs["z"] == (("periodic", "periodic"),)
    assert f.non_periodic_axes == ()


def test_field_face_formats():
    # String format (both faces)
    f_str = Field(shape=(2,), x="dirichlet")
    assert f_str.get_axis_bcs("x") == (
        ("dirichlet", "dirichlet"),
        ("dirichlet", "dirichlet"),
    )
    assert f_str.non_periodic_axes == ("x",)

    # Tuple format (min, max)
    f_tup = Field(shape=(), y=("dirichlet", "neumann"))
    assert f_tup.get_axis_bcs("y") == (("dirichlet", "neumann"),)
    assert f_tup.non_periodic_axes == ("y",)

    # Dict format (min, max)
    f_dict = Field(shape=(), z={"min": "neumann", "max": "dirichlet"})
    assert f_dict.get_axis_bcs("z") == (("neumann", "dirichlet"),)
    assert f_dict.non_periodic_axes == ("z",)


def test_field_component_dict():
    # Roller on y
    u = Field(shape=(2,), y={0: "neumann", 1: "dirichlet"})
    assert u.get_bc("y", 0) == ("neumann", "neumann")
    assert u.get_bc("y", 1) == ("dirichlet", "dirichlet")
    assert u.get_axis_bcs("x") == (
        ("periodic", "periodic"),
        ("periodic", "periodic"),
    )
    assert u.non_periodic_axes == ("y",)

    # Component index out of range
    with pytest.raises(ValueError, match="out of bounds"):
        Field(shape=(2,), y={2: "dirichlet"})

    # Non-integer component key
    with pytest.raises(TypeError, match="must be int"):
        Field(shape=(2,), y={"ux": "dirichlet"})  # type: ignore[dict-item]


def test_field_periodic_consistency():
    with pytest.raises(ValueError, match="Cannot mix periodic"):
        Field(shape=(), x=("dirichlet", "periodic"))

    with pytest.raises(ValueError, match="Cannot mix periodic"):
        Field(shape=(), y={"min": "periodic", "max": "neumann"})


def test_field_tensor_v1_restriction():
    # Periodic tensor is allowed
    Field(shape=(2, 2))

    # Non-periodic tensor must raise in v1
    with pytest.raises(ValueError, match="tensor fields"):
        Field(shape=(2, 2), x="dirichlet")


def test_field_jax_pytree():
    f = Field(shape=(2,), name="u", y={0: "neumann", 1: "dirichlet"})
    leaves, treedef = jax.tree.flatten(f)

    # Pure metadata -> zero dynamic array leaves
    assert len(leaves) == 0

    f_unflattened = jax.tree.unflatten(treedef, leaves)
    assert f_unflattened == f
    assert f_unflattened.bcs == f.bcs
    assert f_unflattened.non_periodic_axes == f.non_periodic_axes


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------


def test_state_construction():
    s = State(u=jnp.array([1.0, 2.0]), T=jnp.array([3.0]))
    assert set(s._dofs.keys()) == {"u", "T"}
    assert len(s._dofs) == 2


def test_state_named_access():
    s = State(u=jnp.array([1.0, 2.0]), T=jnp.array([3.0, 4.0]))
    assert jnp.allclose(s.u, jnp.array([1.0, 2.0]))
    assert jnp.allclose(s.T, jnp.array([3.0, 4.0]))


def test_state_iter():
    state = State(u=jnp.array([1.0, 2.0]), T=jnp.array([3.0, 4.0]))
    (u, T) = state
    assert jnp.allclose(u, jnp.array([1.0, 2.0]))
    assert jnp.allclose(T, jnp.array([3.0, 4.0]))


def test_state_len():
    s = State(a=jnp.ones(5))
    assert len(s) == 1

    s2 = State(a=jnp.ones(1), b=jnp.ones(2), c=jnp.ones(3))
    assert len(s2) == 3


def test_state_reserved_name():
    with pytest.raises(ValueError, match="reserved"):
        State(_dofs=jnp.array(0))


def test_state_missing_key():
    s = State(u=jnp.zeros(3))
    with pytest.raises(AttributeError, match="not found"):
        _ = s.z


def test_state_pytree():
    s = State(u=jnp.array([1.0, 2.0]), T=jnp.array([3.0, 4.0]))
    leaves, treedef = jax.tree.flatten(s)
    s2 = jax.tree.unflatten(treedef, leaves)
    assert jnp.allclose(s2.u, s.u)
    assert jnp.allclose(s2.T, s.T)


def test_state_jit():
    s = State(u=jnp.array([1.0, 2.0]), T=jnp.array([3.0, 4.0]))

    @jax.jit
    def fn(s):
        return s.u + s.T

    result = fn(s)
    assert jnp.allclose(result, jnp.array([4.0, 6.0]))


def test_state_grad():
    s = State(u=jnp.array([1.0, 2.0]), T=jnp.array([3.0, 4.0]))

    @jax.grad
    def energy(s):
        return jnp.sum(s.u**2) + jnp.sum(s.T**3)

    g = energy(s)
    assert isinstance(g, State)
    assert jnp.allclose(g.u, 2 * s.u)
    assert jnp.allclose(g.T, 3 * s.T**2)


def test_state_map_structure():
    s = State(u=jnp.array([1.0, 2.0]), T=jnp.array([3.0, 4.0]))
    s2 = jax.tree.map(lambda x: x * 2, s)
    assert jnp.allclose(s2.u, jnp.array([2.0, 4.0]))
    assert jnp.allclose(s2.T, jnp.array([6.0, 8.0]))


def test_state_single_dof():
    """State with only one named array."""
    s = State(u=jnp.array([1.0, 2.0, 3.0]))
    assert jnp.allclose(s.u, jnp.array([1.0, 2.0, 3.0]))
    assert len(s) == 1
    leaves, treedef = jax.tree.flatten(s)
    s2 = jax.tree.unflatten(treedef, leaves)
    assert jnp.allclose(s2.u, s.u)


# ---------------------------------------------------------------------------
# SolutionSpace: Layout, Slices, Mask, n_free, Views, and Validation
# ---------------------------------------------------------------------------


def test_solution_space_periodic():
    space = SpectralSpace(
        lengths=(1.0, 1.0), shape=(32, 32), transform=FFTTransform(dim=2)
    )
    V = SolutionSpace(space, u=Field(shape=(2,)))

    assert V.node_shape == (32, 32)
    assert V.shape == (32, 32, 2)
    assert V.n_free == 32 * 32 * 2
    assert jnp.all(V.mask)

    assert isinstance(V.u, FieldView)
    assert V.u.shape == (32, 32, 2)
    assert V.u.n_free == 32 * 32 * 2
    assert V.u.channel_slice == slice(0, 2)

    assert len(V) == 1
    assert "u" in V
    assert V["u"] == V.u

    z = V.zeros()
    assert z.shape == (32, 32, 2)
    assert jnp.all(z == 0.0)


def test_solution_space_roller():
    # 2D space with y non-periodic (doubled)
    space = SpectralSpace(
        lengths=(1.0, 1.0),
        shape=(32, 32),
        non_periodic_axes=("y",),
        transform=FFTTransform(dim=2),
    )
    # Roller BC: ux is Neumann on y, uy is Dirichlet on y
    V = SolutionSpace(space, u=Field(shape=(2,), y={0: "neumann", 1: "dirichlet"}))

    assert V.node_shape == (32, 33)
    assert V.shape == (32, 33, 2)

    # ux free nodes = 32 * 33 = 1056; uy free nodes = 32 * 31 = 992; total = 2048
    assert V.n_free == 2048
    assert V.u.n_free == 2048
    assert V.u.shape == (32, 33, 2)

    # Apply mask test
    ones = jnp.ones(V.shape)
    masked = V.apply_mask(ones)
    # ux (channel 0) is Neumann: unconstrained everywhere
    assert jnp.all(masked[:, :, 0] == 1.0)
    # uy (channel 1) is Dirichlet: zeros at y=0 and y=-1, 1 elsewhere
    assert jnp.all(masked[:, 0, 1] == 0.0)
    assert jnp.all(masked[:, -1, 1] == 0.0)
    assert jnp.all(masked[:, 1:-1, 1] == 1.0)

    # On-demand mask property matches n_free exactly
    assert int(jnp.sum(V.mask)) == V.n_free


def test_solution_space_multi_field():
    space = SpectralSpace(
        lengths=(1.0, 1.0),
        shape=(32, 32),
        non_periodic_axes=("y",),
        transform=FFTTransform(dim=2),
    )
    V = SolutionSpace(
        space,
        u=Field(shape=(2,), y={0: "neumann", 1: "dirichlet"}),
        T=Field(shape=(), y=("dirichlet", "neumann")),
    )

    assert V.node_shape == (32, 33)
    assert V.shape == (32, 33, 3)

    assert V.u.shape == (32, 33, 2)
    assert V.u.channel_slice == slice(0, 2)
    assert V.u.n_free == 2048

    assert V.T.shape == (32, 33)
    assert V.T.channel_slice == slice(2, 3)
    # T has 1 Dirichlet face (y=0) and 1 Neumann face (y=-1): 32 * 32 = 1024 free
    assert V.T.n_free == 1024

    assert V.n_free == 2048 + 1024
    assert int(jnp.sum(V.mask)) == V.n_free


def test_solution_space_validation():
    space = SpectralSpace(
        lengths=(1.0, 1.0),
        shape=(32, 32),
        non_periodic_axes=("y",),
        transform=FFTTransform(dim=2),
    )

    # Non-periodic BC along axis 'x' when only 'y' is in space.non_periodic_axes
    with pytest.raises(ValueError, match="not in space.non_periodic_axes"):
        SolutionSpace(space, u=Field(shape=(), x="dirichlet"))

    # Reserved name check
    with pytest.raises(ValueError, match="reserved"):
        SolutionSpace(space, mask=Field(shape=()))

    # Empty fields check
    with pytest.raises(ValueError, match="at least one Field"):
        SolutionSpace(space)


# ---------------------------------------------------------------------------
# Step 3.2 Tests: primal_to_extended, dual_to_physical, primal_to_physical
# ---------------------------------------------------------------------------


def test_primal_to_extended_shapes():
    space = SpectralSpace(
        lengths=(1.0, 1.0),
        shape=(32, 32),
        non_periodic_axes=("y",),
        transform=FFTTransform(dim=2),
    )
    V = SolutionSpace(
        space,
        u=Field(shape=(2,), y={0: "neumann", 1: "dirichlet"}),
        T=Field(shape=(), y=("dirichlet", "neumann")),
    )

    u_sol = V.zeros()
    state_ext = V.primal_to_extended(u_sol)

    # u has shape (*extended_shape, 2) = (32, 64, 2)
    assert state_ext.u.shape == (32, 64, 2)
    # T is scalar: (*extended_shape,) = (32, 64) with no trailing 1
    assert state_ext.T.shape == (32, 64)


def test_multi_dimensional_adjoint():
    """Verify <V.primal_to_extended(u), z> == <u, V.dual_to_physical(z)> for random arrays."""
    key = jax.random.PRNGKey(42)
    k1, k2, k3 = jax.random.split(key, 3)

    space = SpectralSpace(
        lengths=(1.0, 1.0),
        shape=(8, 12),
        non_periodic_axes=("y",),
        transform=FFTTransform(dim=2),
    )
    V = SolutionSpace(
        space,
        u=Field(shape=(2,), y={0: "neumann", 1: "dirichlet"}),
        T=Field(shape=(), y=("dirichlet", "neumann")),
    )

    u_sol = jax.random.normal(k1, V.shape)
    z_u = jax.random.normal(k2, (8, 24, 2))
    z_T = jax.random.normal(k3, (8, 24))
    z_ext = State(u=z_u, T=z_T)

    # Primal E
    u_ext = V.primal_to_extended(u_sol)

    # Dual Eᵀ via State
    dual_phys = V.dual_to_physical(z_ext)

    # Dual Eᵀ via positional and kwargs
    dual_phys_pos = V.dual_to_physical(z_u, z_T)
    dual_phys_kw = V.dual_to_physical(u=z_u, T=z_T)
    assert jnp.allclose(dual_phys, dual_phys_pos)
    assert jnp.allclose(dual_phys, dual_phys_kw)

    # Inner products
    inner_ext = jnp.sum(u_ext.u * z_u) + jnp.sum(u_ext.T * z_T)
    inner_phys = jnp.sum(u_sol * dual_phys)

    assert jnp.allclose(inner_ext, inner_phys, atol=1e-12, rtol=1e-12)


def test_primal_to_physical_periodic_wrap_fill():
    # Doubled axis y, but scalar field has periodic BC
    space = SpectralSpace(
        lengths=(1.0, 1.0),
        shape=(16, 16),
        non_periodic_axes=("y",),
        transform=FFTTransform(dim=2),
    )
    V = SolutionSpace(space, phi=Field(shape=(), y="periodic"))

    u_sol = jnp.ones(V.shape)
    # Give index 0 along y a distinct value 5.0, index -1 a value 0.0
    u_sol = u_sol.at[:, 0, 0].set(5.0)
    u_sol = u_sol.at[:, -1, 0].set(0.0)

    phys_state = V.primal_to_physical(u_sol)
    # The wrap node at y=-1 must be filled with the value from y=0 (5.0)
    assert jnp.all(phys_state.phi[:, 0] == 5.0)
    assert jnp.all(phys_state.phi[:, -1] == 5.0)


def test_solution_space_jit():
    space = SpectralSpace(
        lengths=(1.0, 1.0),
        shape=(16, 16),
        non_periodic_axes=("y",),
        transform=FFTTransform(dim=2),
    )
    V = SolutionSpace(space, u=Field(shape=(2,), y={0: "neumann", 1: "dirichlet"}))

    @jax.jit
    def fwd(u):
        return V.primal_to_extended(u)

    @jax.jit
    def bwd(z):
        return V.dual_to_physical(z)

    u0 = V.zeros()
    ext = fwd(u0)
    rec = bwd(ext)
    assert rec.shape == V.shape

