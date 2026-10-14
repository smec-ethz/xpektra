import jax
import pytest

from xpektra.solution_space import Field


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
