from collections.abc import Callable

import jax
import jax.numpy as jnp
from jax import Array

__all__ = ["TensorOperator"]


# --- Define the broadcast rules for dot product (spatial dims first) ---
def _dot(A: Array, B: Array) -> Array:
    """00 dot product: scalar-scalar"""
    return jnp.einsum("..., ...->...", A, B)


def _dot11(A: Array, B: Array) -> Array:
    """11 dot product: vector-vector"""
    return jnp.einsum("...i, ...i->...", A, B)


def _dot12(A: Array, B: Array) -> Array:
    """12 dot product: vector-tensor"""
    return jnp.einsum("...i, ...ij->...j", A, B)


def _dot21(A: Array, B: Array) -> Array:
    """21 dot product: tensor-vector"""
    return jnp.sum(A[..., :, :] * B[..., None, :], axis=-1)


def _dot22(A: Array, B: Array) -> Array:
    """22 broadcast dot product: tensor-tensor to avoid GEMM inside vmap."""
    return jnp.sum(A[..., :, :, None] * B[..., None, :, :], axis=-2)


def _dot24(A: Array, B: Array) -> Array:
    """24 broadcast dot product: tensor2-tensor4 to avoid GEMM inside vmap."""
    return jnp.sum(A[..., :, :, None, None, None] * B[..., None, :, :, :, :], axis=-4)


def _dot42(A: Array, B: Array) -> Array:
    """42 broadcast dot product: tensor4-tensor to avoid GEMM inside vmap."""
    return jnp.sum(A[..., :, :, :, :, None] * B[..., None, None, None, :, :], axis=-2)


DOT_EINSUM_DISPATCH: dict[tuple[int, int], Callable[[Array, Array], Array]] = {
    (0, 0): _dot,  # scalar-scalar
    (1, 1): _dot11,  # dot11: vector-vector
    (1, 2): _dot12,  # dot12: vector-tensor
    (2, 1): _dot21,  # dot21: tensor-vector
    (2, 2): _dot22,  # dot22: tensor-tensor
    (2, 4): _dot24,  # dot24: tensor-tensor4
    (4, 2): _dot42,  # dot42: tensor4-tensor
}


# --- Define the broadcast rules for ddot (spatial dims first) ---
def _ddot22(A: Array, B: Array) -> Array:
    """Double dot product: tensor-tensor."""
    return jnp.einsum("...ij,...ji->...", A, B)


def _ddot42(A: Array, B: Array) -> Array:
    """...ijkl,...lk->...ij -- B's contracted axes are (l, k), A's are (k, l)."""
    Bs = jnp.swapaxes(B, -2, -1)
    return jnp.sum(A[..., :, :, :, :] * Bs[..., None, None, :, :], axis=(-2, -1))


def _ddot44(A: Array, B: Array) -> Array:
    """...ijkl,...lkmn->...ijmn -- B's contracted axes are (l, k), A's are (k, l)."""
    Bs = jnp.swapaxes(B, -4, -3)
    return jnp.sum(
        A[..., :, :, :, :, None, None] * Bs[..., None, None, :, :, :, :], axis=(-4, -3)
    )


DDOT_EINSUM_DISPATCH: dict[tuple[int, int], Callable[[Array, Array], Array]] = {
    (2, 2): _ddot22,  # ddot22: tensor-tensor
    (4, 2): _ddot42,  # ddot42: tensor4-tensor
    (4, 4): _ddot44,  # ddot44: tensor4-tensor4
}


# --- Define the einsum rules for dyad (spatial dims first) ---
def _ddyad22(A: Array, B: Array) -> Array:
    """...ij,...kl->...ijkl -- dyad22: tensor-tensor."""
    return jnp.einsum("...ij,...kl->...ijkl", A, B)


def _ddyad11(A: Array, B: Array) -> Array:
    """...i,...j->...ij -- dyad11: vector-vector."""
    return jnp.einsum("...i,...j->...ij", A, B)


DYAD_EINSUM_DISPATCH: dict[tuple[int, int], Callable[[Array, Array], Array]] = {
    (2, 2): _ddyad22,  # dyad22: tensor-tensor
    (1, 1): _ddyad11,  # dyad11: vector-vector
}


# --- Define the broadcast rules for trace (spatial dims first) ---
def _trace2(A: Array) -> Array:
    """Trace of a rank-2 tensor: A_ii."""
    return jnp.trace(A, axis1=-2, axis2=-1)


def _trace4(A: Array) -> Array:
    """Trace of a rank-4 tensor: A_ijij (pairs axes -4/-2 and -3/-1)."""
    return jnp.trace(jnp.trace(A, axis1=-4, axis2=-2), axis1=-2, axis2=-1)


# --- Define the trace rules (spatial dims first) ---
# These are callables rather than einsum strings.  An einsum with a repeated
# index inside a single operand ("...ii->...") has no XLA primitive, so it is
# emulated with a mask-and-select whose mask is built replicated; that clashes
# with a sharded operand under explicit sharding.  jnp.trace has no such issue.
TRACE_DISPATCH: dict[int, Callable[[Array], Array]] = {
    2: _trace2,  # trace of a rank-2 tensor
    4: _trace4,  # trace of a rank-4 tensor (e.g., for identity)
}


# --- Define the einsum rules for transpose (spatial dims first) ---
def _transpose2(A: Array) -> Array:
    return jnp.einsum("...ij->...ji", A)


TRANS_EINSUM_DISPATCH: dict[int, Callable[[Array], Array]] = {
    2: _transpose2,  # transpose of a rank-2 tensor
}


def _det11(A: Array) -> Array:
    """Determinant of a 1x1 block, shape ``(spatial...,)``."""
    return A[..., 0, 0]


def _det22(A: Array) -> Array:
    """Determinant of a 2x2 block, shape ``(spatial...,)``."""
    return A[..., 0, 0] * A[..., 1, 1] - A[..., 0, 1] * A[..., 1, 0]


def _det33(A: Array) -> Array:
    """Determinant of a 3x3 block as the scalar triple product of its columns."""
    a1 = A[..., :, 0]
    a2 = A[..., :, 1]
    a3 = A[..., :, 2]
    return jnp.sum(a1 * jnp.cross(a2, a3), axis=-1)


# --- Define the determinant rules (spatial dims first) ---
# Keyed on the block size ``A.shape[-1]``, matching INVERSE_DISPATCH.  The
# expressions duplicate the ones inside the ``_inverse`` functions rather than
# being shared: the callers that need both are typically thresholding on ``det``
# to decide *whether* to invert, so the two are not evaluated together.
DET_DISPATCH: dict[int, Callable[[Array], Array]] = {
    1: _det11,  # determinant of a 1x1 block
    2: _det22,  # determinant of a 2x2 block
    3: _det33,  # determinant of a 3x3 block
}


def _inverse11(A: Array) -> Array:
    det = A[..., 0, 0]
    A_inv = 1 / det
    return A_inv[..., None, None]


def _inverse22(A: Array) -> Array:
    a = A[..., 0, 0]
    b = A[..., 0, 1]
    c = A[..., 1, 0]
    e = A[..., 1, 1]
    det = a * e - b * c

    row1 = jnp.stack([e, -b], axis=-1)
    row2 = jnp.stack([-c, a], axis=-1)
    adj = jnp.stack([row1, row2], axis=-2)

    return adj / det[..., None, None]


def _inverse33(A: Array) -> Array:
    """Reciprocal-basis form: the rows of ``det * A^-1`` are the cross products
    of the columns of ``A``, which builds the adjugate already transposed."""
    a1 = A[..., :, 0]
    a2 = A[..., :, 1]
    a3 = A[..., :, 2]

    c1 = jnp.cross(a2, a3)
    c2 = jnp.cross(a3, a1)
    c3 = jnp.cross(a1, a2)

    det = jnp.sum(a1 * c1, axis=-1)
    return jnp.stack([c1, c2, c3], axis=-2) / det[..., None, None]


# --- Define the inverse rules (spatial dims first) ---
# Keyed on the block size ``A.shape[-1]``, not on the tensor rank: the rank is
# always 2 here, and it is the size that selects the closed-form expression.
# The formulas are unguarded -- a singular block yields inf/nan rather than an
# error.  Null-space handling belongs to the caller, which knows which modes are
# expected to be singular (e.g. the xi=0 mode of a Green's operator).
INVERSE_DISPATCH: dict[int, Callable[[Array], Array]] = {
    1: _inverse11,  # inverse of a 1x1 block
    2: _inverse22,  # inverse of a 2x2 block
    3: _inverse33,  # inverse of a 3x3 block
}


@jax.tree_util.register_pytree_node_class
class TensorOperator:
    """Tensor algebra operator for fields with layout (spatial..., tensor...).

    By default, supports the standard rank combinations defined in the module-level
    dispatch tables. Additional einsum rules can be registered at construction time
    via the ``dot_rules``, ``ddot_rules``, ``dyad_rules``, ``trace_rules``,
    ``trans_rules``, ``inverse_rules``, and ``det_rules`` arguments, allowing advanced
    users to extend the operator without modifying library source.  ``inverse_rules``
    and ``det_rules`` are keyed on the block size ``A.shape[-1]``; all others are keyed
    on tensor rank.

    Example — adding a rank-(3, 2) dot rule:

    ```python
    op = TensorOperator(dim=3, dot_rules={(3, 2): lambda A, B: jnp.einsum("...ijk,...kl->...ijl", A, B)})
    ```
    """

    _dot_rules: dict[tuple[int, int], Callable[[Array, Array], Array]]
    _ddot_rules: dict[tuple[int, int], Callable[[Array, Array], Array]]
    _dyad_rules: dict[tuple[int, int], Callable[[Array, Array], Array]]
    _trace_rules: dict[int, Callable[[Array], Array]]
    _trans_rules: dict[int, Callable[[Array], Array]]
    _inverse_rules: dict[int, Callable[[Array], Array]]
    _det_rules: dict[int, Callable[[Array], Array]]
    dim: int

    def __eq__(self, other):
        """Structural equality so JAX static-field comparison works correctly."""
        if type(self) is not type(other):
            return NotImplemented
        return (
            self.dim == other.dim
            and self._dot_rules == other._dot_rules
            and self._ddot_rules == other._ddot_rules
            and self._dyad_rules == other._dyad_rules
            and self._trace_rules == other._trace_rules
            and self._trans_rules == other._trans_rules
            and self._inverse_rules == other._inverse_rules
            and self._det_rules == other._det_rules
        )

    def __hash__(self):
        return hash(
            (
                self.dim,
                tuple(sorted(self._dot_rules.items())),
                tuple(sorted(self._ddot_rules.items())),
                tuple(sorted(self._dyad_rules.items())),
                tuple(sorted(self._trace_rules.items())),
                tuple(sorted(self._trans_rules.items())),
                tuple(sorted(self._inverse_rules.items())),
                tuple(sorted(self._det_rules.items())),
            )
        )

    def __setattr__(self, name, value):
        """Enforce immutability after initialization.

        Attribute assignment is only allowed during ``__init__`` (before
        ``_initialized`` is set).  Any attempt to mutate the instance
        afterwards raises ``AttributeError``, mirroring the guarantees
        previously provided by ``eqx.Module``.
        """
        if hasattr(self, "_initialized"):
            raise AttributeError(f"Cannot modify frozen {type(self).__name__}")
        object.__setattr__(self, name, value)

    def __init__(
        self,
        dim: int,
        dot_rules: dict[tuple[int, int], Callable[[Array, Array], Array]] | None = None,
        ddot_rules: dict[tuple[int, int], Callable[[Array, Array], Array]]
        | None = None,
        dyad_rules: dict[tuple[int, int], Callable[[Array, Array], Array]]
        | None = None,
        trace_rules: dict[int, Callable[[Array], Array]] | None = None,
        trans_rules: dict[int, Callable[[Array], Array]] | None = None,
        inverse_rules: dict[int, Callable[[Array], Array]] | None = None,
        det_rules: dict[int, Callable[[Array], Array]] | None = None,
    ):
        self.dim = dim
        self._dot_rules = {**DOT_EINSUM_DISPATCH, **(dot_rules or {})}
        self._ddot_rules = {**DDOT_EINSUM_DISPATCH, **(ddot_rules or {})}
        self._dyad_rules = {**DYAD_EINSUM_DISPATCH, **(dyad_rules or {})}
        self._trace_rules = {**TRACE_DISPATCH, **(trace_rules or {})}
        self._trans_rules = {**TRANS_EINSUM_DISPATCH, **(trans_rules or {})}
        self._inverse_rules = {**INVERSE_DISPATCH, **(inverse_rules or {})}
        self._det_rules = {**DET_DISPATCH, **(det_rules or {})}
        object.__setattr__(self, "_initialized", True)

    def tree_flatten(self):
        # No dynamic fields, so we return empty list and the static fields as metadata
        children = []
        aux_data = {
            "dim": self.dim,
            "dot_rules": self._dot_rules,
            "ddot_rules": self._ddot_rules,
            "dyad_rules": self._dyad_rules,
            "trace_rules": self._trace_rules,
            "trans_rules": self._trans_rules,
            "inverse_rules": self._inverse_rules,
            "det_rules": self._det_rules,
        }
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data: dict, children: list) -> "TensorOperator":
        return TensorOperator(
            dim=aux_data["dim"],
            dot_rules=aux_data["dot_rules"],
            ddot_rules=aux_data["ddot_rules"],
            dyad_rules=aux_data["dyad_rules"],
            trace_rules=aux_data["trace_rules"],
            trans_rules=aux_data["trans_rules"],
            inverse_rules=aux_data["inverse_rules"],
            det_rules=aux_data["det_rules"],
        )

    def _get_rank(self, A: Array) -> int:
        """Returns the tensor rank of A (total ndim minus spatial dims)."""
        rank = len(A.shape) - self.dim
        if rank < 0:
            raise ValueError(
                f"Array with shape {A.shape} has fewer dimensions than the "
                f"number of spatial dimensions ({self.dim})."
            )
        return rank

    @jax.jit
    def dot(self, A: Array, B: Array) -> Array:
        """Computes the dot product between tensors A and B."""
        rank_A = self._get_rank(A)
        rank_B = self._get_rank(B)
        dot_fn = self._dot_rules.get((rank_A, rank_B))
        if dot_fn is None:
            raise NotImplementedError(
                f"No dot product implemented for tensor ranks ({rank_A}, {rank_B})."
            )
        return dot_fn(A, B)

    @jax.jit
    def ddot(self, A: Array, B: Array) -> Array:
        """Computes the double dot product between tensors A and B."""
        rank_A = self._get_rank(A)
        rank_B = self._get_rank(B)
        ddot_fn = self._ddot_rules.get((rank_A, rank_B))
        if ddot_fn is None:
            raise NotImplementedError(
                f"No double dot product implemented for tensor ranks ({rank_A}, {rank_B})."
            )
        return ddot_fn(A, B)

    @jax.jit
    def trace(self, A: Array) -> Array:
        """Computes the trace of tensor A."""
        rank_A = self._get_rank(A)
        trace_fn = self._trace_rules.get(rank_A)
        if trace_fn is None:
            raise NotImplementedError(
                f"No trace implemented for tensor rank ({rank_A})."
            )
        return trace_fn(A)

    @jax.jit
    def trans(self, A: Array) -> Array:
        """Computes the transpose of tensor A."""
        rank_A = self._get_rank(A)
        trans_fn = self._trans_rules.get(rank_A)
        if trans_fn is None:
            raise NotImplementedError(
                f"No transpose implemented for tensor rank ({rank_A})."
            )
        return trans_fn(A)

    def _get_block_size(self, A: Array, op_name: str) -> int:
        """Returns ``d`` for a rank-2 field of square ``(d, d)`` blocks."""
        rank_A = self._get_rank(A)
        if rank_A != 2:
            raise NotImplementedError(
                f"{op_name} is only defined for rank-2 tensors, got rank ({rank_A})."
            )
        rows, cols = A.shape[-2], A.shape[-1]
        if rows != cols:
            raise ValueError(
                f"{op_name} requires square tensor blocks, got ({rows}, {cols})."
            )
        return cols

    @jax.jit
    def inv(self, A: Array) -> Array:
        """Pointwise inverse of a rank-2 tensor field, shape ``(spatial..., d, d)``.

        Each ``(d, d)`` block is inverted independently by a closed-form
        expression, so no batched LU decomposition is dispatched.  Singular
        blocks yield ``inf``/``nan`` rather than raising -- see
        ``INVERSE_DISPATCH``.  Use ``det`` to mask them out beforehand.
        """
        d = self._get_block_size(A, "Inverse")
        inverse_fn = self._inverse_rules.get(d)
        if inverse_fn is None:
            raise NotImplementedError(
                f"No inverse implemented for tensor blocks of size ({d}, {d})."
            )
        return inverse_fn(A)

    @jax.jit
    def det(self, A: Array) -> Array:
        """Pointwise determinant of a rank-2 tensor field, shape ``(spatial...,)``.

        Companion to ``inv``: thresholding on ``abs(det)`` is how a caller
        identifies the blocks that ``inv`` cannot handle.
        """
        d = self._get_block_size(A, "Determinant")
        det_fn = self._det_rules.get(d)
        if det_fn is None:
            raise NotImplementedError(
                f"No determinant implemented for tensor blocks of size ({d}, {d})."
            )
        return det_fn(A)

    @jax.jit
    def dyad(self, A: Array, B: Array) -> Array:
        """Computes the dyadic product between tensors A and B."""
        rank_A = self._get_rank(A)
        rank_B = self._get_rank(B)
        dyad_fn = self._dyad_rules.get((rank_A, rank_B))
        if dyad_fn is None:
            raise NotImplementedError(
                f"No dyad implemented for tensor ranks ({rank_A}, {rank_B})."
            )
        return dyad_fn(A, B)
