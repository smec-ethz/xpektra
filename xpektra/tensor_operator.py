from collections.abc import Callable

import jax
import jax.numpy as jnp
from jax import Array


# --- Define the broadcast rules for dot product (spatial dims first) ---
def _dot(A: Array, B: Array) -> Array:
    """00 dot product: scalar-scalar"""
    return jnp.einsum("..., ...->...", A, B)


def _dot11(A: Array, B: Array) -> Array:
    """11 dot product: vector-vector"""
    return jnp.einsum("...i, ...i->...", A, B)


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


@jax.tree_util.register_pytree_node_class
class TensorOperator:
    """Tensor algebra operator for fields with layout (spatial..., tensor...).

    By default, supports the standard rank combinations defined in the module-level
    dispatch tables. Additional einsum rules can be registered at construction time
    via the ``dot_rules``, ``ddot_rules``, ``dyad_rules``, ``trace_rules``, and
    ``trans_rules`` arguments, allowing advanced users to extend the operator without
    modifying library source.

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
    ):
        self.dim = dim
        self._dot_rules = {**DOT_EINSUM_DISPATCH, **(dot_rules or {})}
        self._ddot_rules = {**DDOT_EINSUM_DISPATCH, **(ddot_rules or {})}
        self._dyad_rules = {**DYAD_EINSUM_DISPATCH, **(dyad_rules or {})}
        self._trace_rules = {**TRACE_DISPATCH, **(trace_rules or {})}
        self._trans_rules = {**TRANS_EINSUM_DISPATCH, **(trans_rules or {})}
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
