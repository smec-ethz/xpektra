import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import pytest

from xpektra.tensor_operator import TensorOperator


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _identity_field(dim, spatial_shape):
    """Return an identity rank-2 tensor field, shape spatial_shape + (dim, dim)."""
    I = jnp.eye(dim)
    return jnp.broadcast_to(I, spatial_shape + (dim, dim))


def _random_symmetric(dim, spatial_shape, key=0):
    """Return a random symmetric rank-2 tensor field."""
    rng = jax.random.PRNGKey(key)
    A = jax.random.normal(rng, spatial_shape + (dim, dim))
    return 0.5 * (A + jnp.swapaxes(A, -1, -2))


def _random_field(dim, spatial_shape, rank, key=0):
    rng = jax.random.PRNGKey(key)
    tensor_shape = (dim,) * rank
    return jax.random.normal(rng, spatial_shape + tensor_shape)


# ---------------------------------------------------------------------------
# dot
# ---------------------------------------------------------------------------

class TestDot:
    @pytest.mark.parametrize("dim", [1, 2, 3])
    def test_dot_vector_vector_shape(self, dim):
        spatial = (4,) * dim
        op = TensorOperator(dim=dim)
        u = _random_field(dim, spatial, rank=1)
        v = _random_field(dim, spatial, rank=1, key=1)
        result = op.dot(u, v)
        assert result.shape == spatial

    @pytest.mark.parametrize("dim", [1, 2, 3])
    def test_dot_vector_vector_value(self, dim):
        spatial = (4,) * dim
        op = TensorOperator(dim=dim)
        u = _random_field(dim, spatial, rank=1)
        v = _random_field(dim, spatial, rank=1, key=1)
        result = op.dot(u, v)
        expected = jnp.einsum("...i,...i->...", u, v)
        np.testing.assert_allclose(result, expected, atol=1e-12)

    def test_dot_tensor_vector_shape(self):
        dim, spatial = 2, (4, 4)
        op = TensorOperator(dim=dim)
        A = _random_field(dim, spatial, rank=2)
        v = _random_field(dim, spatial, rank=1)
        result = op.dot(A, v)
        assert result.shape == spatial + (dim,)

    def test_dot_tensor_vector_value(self):
        dim, spatial = 2, (4, 4)
        op = TensorOperator(dim=dim)
        A = _random_field(dim, spatial, rank=2)
        v = _random_field(dim, spatial, rank=1)
        result = op.dot(A, v)
        expected = jnp.einsum("...ij,...j->...i", A, v)
        np.testing.assert_allclose(result, expected, atol=1e-12)

    def test_dot_tensor_tensor_shape(self):
        dim, spatial = 2, (4, 4)
        op = TensorOperator(dim=dim)
        A = _random_field(dim, spatial, rank=2)
        B = _random_field(dim, spatial, rank=2, key=1)
        result = op.dot(A, B)
        assert result.shape == spatial + (dim, dim)

    def test_dot_tensor_tensor_value(self):
        dim, spatial = 2, (4, 4)
        op = TensorOperator(dim=dim)
        A = _random_field(dim, spatial, rank=2)
        B = _random_field(dim, spatial, rank=2, key=1)
        result = op.dot(A, B)
        expected = jnp.einsum("...ij,...jk->...ik", A, B)
        np.testing.assert_allclose(result, expected, atol=1e-12)


# ---------------------------------------------------------------------------
# ddot
# ---------------------------------------------------------------------------

class TestDdot:
    def test_ddot_22_shape(self):
        dim, spatial = 2, (4, 4)
        op = TensorOperator(dim=dim)
        A = _random_field(dim, spatial, rank=2)
        B = _random_field(dim, spatial, rank=2, key=1)
        result = op.ddot(A, B)
        assert result.shape == spatial

    def test_ddot_22_value(self):
        dim, spatial = 2, (4, 4)
        op = TensorOperator(dim=dim)
        A = _random_field(dim, spatial, rank=2)
        B = _random_field(dim, spatial, rank=2, key=1)
        result = op.ddot(A, B)
        expected = jnp.einsum("...ij,...ji->...", A, B)
        np.testing.assert_allclose(result, expected, atol=1e-12)

    def test_ddot_22_non_negative_self(self):
        """A:A >= 0 for any real A."""
        dim, spatial = 2, (4, 4)
        op = TensorOperator(dim=dim)
        A = _random_field(dim, spatial, rank=2)
        result = op.ddot(A, jnp.swapaxes(A, -1, -2))  # A : A^T = sum A_ij^2
        assert jnp.all(result >= 0)

    def test_ddot_42_shape(self):
        dim, spatial = 2, (4, 4)
        op = TensorOperator(dim=dim)
        A4 = _random_field(dim, spatial, rank=4)
        B2 = _random_field(dim, spatial, rank=2)
        result = op.ddot(A4, B2)
        assert result.shape == spatial + (dim, dim)

    def test_ddot_42_value(self):
        dim, spatial = 2, (4, 4)
        op = TensorOperator(dim=dim)
        A4 = _random_field(dim, spatial, rank=4)
        B2 = _random_field(dim, spatial, rank=2)
        result = op.ddot(A4, B2)
        expected = jnp.einsum("...ijkl,...lk->...ij", A4, B2)
        np.testing.assert_allclose(result, expected, atol=1e-12)

    def test_ddot_equals_trace_dot(self):
        """ddot(A, B) == trace(dot(A, B)) for rank-2 tensors."""
        dim, spatial = 2, (4, 4)
        op = TensorOperator(dim=dim)
        A = _random_field(dim, spatial, rank=2)
        B = _random_field(dim, spatial, rank=2, key=1)
        ddot_result = op.ddot(A, B)
        trace_dot_result = op.trace(op.dot(A, B))
        np.testing.assert_allclose(ddot_result, trace_dot_result, atol=1e-12)


# ---------------------------------------------------------------------------
# trace
# ---------------------------------------------------------------------------

class TestTrace:
    def test_trace_identity(self):
        """trace of identity field equals dim everywhere."""
        dim, spatial = 2, (4, 4)
        op = TensorOperator(dim=dim)
        I_field = _identity_field(dim, spatial)
        result = op.trace(I_field)
        np.testing.assert_allclose(result, dim, atol=1e-12)

    @pytest.mark.parametrize("dim", [1, 2, 3])
    def test_trace_shape(self, dim):
        spatial = (4,) * dim
        op = TensorOperator(dim=dim)
        A = _random_field(dim, spatial, rank=2)
        result = op.trace(A)
        assert result.shape == spatial

    def test_trace_diagonal(self):
        """trace of diag([a, b]) == a + b."""
        dim, spatial = 2, (4, 4)
        op = TensorOperator(dim=dim)
        rng = jax.random.PRNGKey(0)
        diag_vals = jax.random.normal(rng, spatial + (dim,))
        A = jnp.zeros(spatial + (dim, dim))
        for i in range(dim):
            A = A.at[..., i, i].set(diag_vals[..., i])
        result = op.trace(A)
        expected = jnp.sum(diag_vals, axis=-1)
        np.testing.assert_allclose(result, expected, atol=1e-12)


# ---------------------------------------------------------------------------
# trans
# ---------------------------------------------------------------------------

class TestTrans:
    @pytest.mark.parametrize("dim", [1, 2, 3])
    def test_trans_involution(self, dim):
        """trans(trans(A)) == A."""
        spatial = (4,) * dim
        op = TensorOperator(dim=dim)
        A = _random_field(dim, spatial, rank=2)
        result = op.trans(op.trans(A))
        np.testing.assert_allclose(result, A, atol=1e-12)

    def test_trans_shape(self):
        dim, spatial = 2, (4, 4)
        op = TensorOperator(dim=dim)
        A = _random_field(dim, spatial, rank=2)
        result = op.trans(A)
        assert result.shape == spatial + (dim, dim)

    def test_trans_swaps_indices(self):
        """trans(A)[..., i, j] == A[..., j, i]."""
        dim, spatial = 2, (4, 4)
        op = TensorOperator(dim=dim)
        A = _random_field(dim, spatial, rank=2)
        At = op.trans(A)
        np.testing.assert_allclose(At[..., 0, 1], A[..., 1, 0], atol=1e-12)
        np.testing.assert_allclose(At[..., 1, 0], A[..., 0, 1], atol=1e-12)

    def test_dot_transpose_identity(self):
        """dot(A, v) computed two ways: directly, and via transpose."""
        dim, spatial = 2, (4, 4)
        op = TensorOperator(dim=dim)
        A = _random_field(dim, spatial, rank=2)
        v = _random_field(dim, spatial, rank=1)
        # A v = (A^T)^T v
        result1 = op.dot(A, v)
        result2 = op.dot(op.trans(op.trans(A)), v)
        np.testing.assert_allclose(result1, result2, atol=1e-12)


# ---------------------------------------------------------------------------
# dyad
# ---------------------------------------------------------------------------

class TestEqualityAndHash:
    def test_equal_default_operators(self):
        """Operators with the same dim and default rules are equal."""
        a = TensorOperator(dim=2)
        b = TensorOperator(dim=2)
        assert a == b
        assert hash(a) == hash(b)

    def test_different_dim_not_equal(self):
        """Operators with different dim are not equal."""
        a = TensorOperator(dim=2)
        b = TensorOperator(dim=3)
        assert a != b

    def test_equal_custom_rules(self):
        """Operators with the same custom rules are equal."""
        rule = {(1, 0): "...i,...->...i"}
        a = TensorOperator(dim=2, dot_rules=rule)
        b = TensorOperator(dim=2, dot_rules=rule)
        assert a == b
        assert hash(a) == hash(b)

    def test_different_custom_rules_not_equal(self):
        """Operators with different custom rules are not equal."""
        a = TensorOperator(dim=2, dot_rules={(1, 0): "...i,...->...i"})
        b = TensorOperator(dim=2)
        assert a != b

    def test_immutability(self):
        """Attribute assignment after init raises AttributeError."""
        op = TensorOperator(dim=2)
        with pytest.raises(AttributeError, match="Cannot modify frozen"):
            op.dim = 3


class TestCustomRules:
    def test_extra_dot_rule(self):
        """Users can register additional einsum rules at construction time."""
        dim, spatial = 2, (4, 4)
        # Add a rank-(1, 0) dot rule: scale a vector by a scalar field
        op = TensorOperator(dim=dim, dot_rules={(1, 0): "...i,...->...i"})
        rng = jax.random.PRNGKey(0)
        v = jax.random.normal(rng, spatial + (dim,))
        s = jax.random.normal(jax.random.PRNGKey(1), spatial)
        result = op.dot(v, s)
        expected = jnp.einsum("...i,...->...i", v, s)
        np.testing.assert_allclose(result, expected, atol=1e-12)

    def test_extra_rule_does_not_override_defaults(self):
        """Adding a custom rule does not affect other built-in rules."""
        dim, spatial = 2, (4, 4)
        op = TensorOperator(dim=dim, dot_rules={(1, 0): "...i,...->...i"})
        rng = jax.random.PRNGKey(0)
        A = jax.random.normal(rng, spatial + (dim, dim))
        v = jax.random.normal(jax.random.PRNGKey(1), spatial + (dim,))
        result = op.dot(A, v)
        expected = jnp.einsum("...ij,...j->...i", A, v)
        np.testing.assert_allclose(result, expected, atol=1e-12)

    def test_extra_ddot_rule(self):
        """Custom ddot rule for rank-(2, 2) override (user can override defaults too)."""
        dim, spatial = 2, (4, 4)
        # Override ddot22 with Frobenius inner product instead of trace(AB)
        op = TensorOperator(dim=dim, ddot_rules={(2, 2): "...ij,...ij->..."})
        rng = jax.random.PRNGKey(0)
        A = jax.random.normal(rng, spatial + (dim, dim))
        B = jax.random.normal(jax.random.PRNGKey(1), spatial + (dim, dim))
        result = op.ddot(A, B)
        expected = jnp.einsum("...ij,...ij->...", A, B)
        np.testing.assert_allclose(result, expected, atol=1e-12)


class TestDyad:
    def test_dyad_vector_vector_shape(self):
        dim, spatial = 2, (4, 4)
        op = TensorOperator(dim=dim)
        u = _random_field(dim, spatial, rank=1)
        v = _random_field(dim, spatial, rank=1, key=1)
        result = op.dyad(u, v)
        assert result.shape == spatial + (dim, dim)

    def test_dyad_vector_vector_value(self):
        dim, spatial = 2, (4, 4)
        op = TensorOperator(dim=dim)
        u = _random_field(dim, spatial, rank=1)
        v = _random_field(dim, spatial, rank=1, key=1)
        result = op.dyad(u, v)
        expected = jnp.einsum("...i,...j->...ij", u, v)
        np.testing.assert_allclose(result, expected, atol=1e-12)

    def test_dyad_tensor_tensor_shape(self):
        dim, spatial = 2, (4, 4)
        op = TensorOperator(dim=dim)
        A = _random_field(dim, spatial, rank=2)
        B = _random_field(dim, spatial, rank=2, key=1)
        result = op.dyad(A, B)
        assert result.shape == spatial + (dim, dim, dim, dim)

    def test_trace_of_dyad_equals_dot(self):
        """trace(dyad(u, v)) == dot(u, v)."""
        dim, spatial = 2, (4, 4)
        op = TensorOperator(dim=dim)
        u = _random_field(dim, spatial, rank=1)
        v = _random_field(dim, spatial, rank=1, key=1)
        trace_dyad = op.trace(op.dyad(u, v))
        dot_uv = op.dot(u, v)
        np.testing.assert_allclose(trace_dyad, dot_uv, atol=1e-12)
