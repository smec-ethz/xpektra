"""``op.eval``: node field -> quadrature points, and the lumped mass it implies."""

import itertools

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import pytest

from xpektra import SpectralSpace
from xpektra.scheme import (
    CentralScheme,
    FourierScheme,
    Hex1RScheme,
    Quad1RScheme,
    QuadFullScheme,
    Tetra2Scheme,
)
from xpektra.spectral_operator import SpectralOperator
from xpektra.transform import FFTTransform

N = 9
SCHEMES = [
    (CentralScheme, 2),
    (FourierScheme, 2),
    (Quad1RScheme, 2),
    (QuadFullScheme, 2),
    (Hex1RScheme, 3),
    (Tetra2Scheme, 3),
]
IDS = [c.__name__ for c, _ in SCHEMES]


def _op(cls, dim):
    lengths = (2.0, 0.5, 3.0)[:dim]
    space = SpectralSpace(
        lengths=lengths, shape=(N,) * dim, transform=FFTTransform(dim=dim)
    )
    return SpectralOperator(scheme=cls(space=space), space=space)


def _at(u, offset):
    """``u`` at node ``x + offset``, as a field over ``x``."""
    return jnp.roll(u, tuple(-o for o in offset), axis=tuple(range(len(offset))))


@pytest.mark.parametrize(("cls", "dim"), SCHEMES, ids=IDS)
def test_partition_of_unity(cls, dim):
    """Constants are reproduced at every quadrature point, so mass is conserved."""
    op = _op(cls, dim)
    u = jnp.ones((N,) * dim + (dim,))
    np.testing.assert_allclose(
        op.eval(u), jnp.ones((N,) * dim + (op.scheme.n_quads, dim)), atol=1e-13
    )


def test_custom_scheme_without_value_stencils_still_differentiates():
    """Only ``eval`` needs value stencils; a scheme that omits them must still build."""

    class TwoPointCentral(CentralScheme):
        @property
        def support_stencils(self):
            return (self.stencils, self.stencils)

    space = SpectralSpace(lengths=(1.0, 1.0), shape=(N, N), transform=FFTTransform(dim=2))
    op = SpectralOperator(scheme=TwoPointCentral(space=space), space=space)
    assert op.grad(jnp.ones((N, N))).shape == (N, N, 2, 2)
    with pytest.raises(NotImplementedError, match="support_value_stencils"):
        op.eval(jnp.ones((N, N)))


def test_quad1r_is_the_corner_average():
    op = _op(Quad1RScheme, 2)
    u = jax.random.normal(jax.random.PRNGKey(0), (N, N))
    corners = sum(_at(u, o) for o in itertools.product((0, 1), repeat=2)) / 4
    np.testing.assert_allclose(op.eval(u)[..., 0], corners, atol=1e-13)


def test_tetra2_is_the_vertex_average_of_each_tetrahedron():
    op = _op(Tetra2Scheme, 3)
    u = jax.random.normal(jax.random.PRNGKey(0), (N, N, N))
    t1 = ((0, 0, 0), (1, 1, 0), (0, 1, 1), (1, 0, 1))
    t2 = ((1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 1))
    for q, tet in enumerate((t1, t2)):
        expected = sum(_at(u, v) for v in tet) / 4
        np.testing.assert_allclose(op.eval(u)[..., q], expected, atol=1e-13)


@pytest.mark.parametrize("cls", [Quad1RScheme, QuadFullScheme])
def test_lumped_mass_from_the_mass_functional(cls):
    """``grad_w integrate(rho * eval(w))`` is the row-sum lumped Q1 mass."""
    op = _op(cls, 2)
    rho = jax.random.uniform(jax.random.PRNGKey(1), (N, N))  # per cell
    rho_q = jnp.broadcast_to(rho[..., None], (N, N, op.scheme.n_quads))

    mass = jax.grad(lambda w: op.integrate(rho_q * op.eval(w)))(jnp.zeros((N, N)))

    # node (i, j) collects a quarter of cells (i-1..i, j-1..j)
    cell_volume = 2.0 * 0.5 / N**2
    expected = cell_volume * sum(
        jnp.roll(rho, o, axis=(0, 1)) for o in itertools.product((0, 1), repeat=2)
    ) / 4
    np.testing.assert_allclose(mass, expected, atol=1e-13)
    np.testing.assert_allclose(mass.sum(), op.integrate(rho_q), rtol=1e-13)
