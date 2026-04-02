from typing import Dict, List, Tuple

import jax
import jax.numpy as jnp  # type: ignore
from jax import Array

# --- Define the einsum rules for dot product (spatial dims first) ---
DOT_EINSUM_DISPATCH: Dict[Tuple[int, int], str] = {
    (0, 0): "...,...->...",  # scalar-scalar
    (1, 1): "...i,...i->...",  # dot11: vector-vector
    (2, 1): "...ij,...j->...i",  # dot21: tensor-vector
    (2, 2): "...ij,...jk->...ik",  # dot22: tensor-tensor
    (2, 4): "...ij,...jkmn->...ikmn",  # dot24: tensor-tensor4
    (4, 2): "...ijkl,...lm->...ijkm",  # dot42: tensor4-tensor
}

# --- Define the einsum rules for double dot product (spatial dims first) ---
DDOT_EINSUM_DISPATCH: Dict[Tuple[int, int], str] = {
    (2, 2): "...ij,...ji->...",  # ddot22: tensor-tensor
    (4, 2): "...ijkl,...lk->...ij",  # ddot42: tensor4-tensor
    (4, 4): "...ijkl,...lkmn->...ijmn",  # ddot44: tensor4-tensor4
}

# --- Define the einsum rules for dyad (spatial dims first) ---
DYAD_EINSUM_DISPATCH: Dict[Tuple[int, int], str] = {
    (2, 2): "...ij,...kl->...ijkl",  # dyad22: tensor-tensor
    (1, 1): "...i,...j->...ij",  # dyad11: vector-vector
}

# --- Define the einsum rules for trace (spatial dims first) ---
TRACE_EINSUM_DISPATCH: Dict[int, str] = {
    2: "...ii->...",  # trace of a rank-2 tensor
    4: "...ijij->...",  # trace of a rank-4 tensor (e.g., for identity)
}

# --- Define the einsum rules for transpose (spatial dims first) ---
TRANS_EINSUM_DISPATCH: Dict[int, str] = {
    2: "...ij->...ji",  # transpose of a rank-2 tensor
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
    op = TensorOperator(dim=3, dot_rules={(3, 2): "...ijk,...kl->...ijl"})
    ```
    """

    _dot_rules: Dict[Tuple[int, int], str]
    _ddot_rules: Dict[Tuple[int, int], str]
    _dyad_rules: Dict[Tuple[int, int], str]
    _trace_rules: Dict[int, str]
    _trans_rules: Dict[int, str]
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
        return hash((
            self.dim,
            tuple(sorted(self._dot_rules.items())),
            tuple(sorted(self._ddot_rules.items())),
            tuple(sorted(self._dyad_rules.items())),
            tuple(sorted(self._trace_rules.items())),
            tuple(sorted(self._trans_rules.items())),
        ))

    def __setattr__(self, name, value):
        """Enforce immutability after initialization.

        Attribute assignment is only allowed during ``__init__`` (before
        ``_initialized`` is set).  Any attempt to mutate the instance
        afterwards raises ``AttributeError``, mirroring the guarantees
        previously provided by ``eqx.Module``.
        """
        if hasattr(self, "_initialized"):
            raise AttributeError(
                f"Cannot modify frozen {type(self).__name__}"
            )
        object.__setattr__(self, name, value)

    def __init__(
        self,
        dim: int,
        dot_rules: Dict[Tuple[int, int], str] | None = None,
        ddot_rules: Dict[Tuple[int, int], str] | None = None,
        dyad_rules: Dict[Tuple[int, int], str] | None = None,
        trace_rules: Dict[int, str] | None = None,
        trans_rules: Dict[int, str] | None = None,
    ):
        self.dim = dim
        self._dot_rules = {**DOT_EINSUM_DISPATCH, **(dot_rules or {})}
        self._ddot_rules = {**DDOT_EINSUM_DISPATCH, **(ddot_rules or {})}
        self._dyad_rules = {**DYAD_EINSUM_DISPATCH, **(dyad_rules or {})}
        self._trace_rules = {**TRACE_EINSUM_DISPATCH, **(trace_rules or {})}
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
    def tree_unflatten(cls, aux_data: Dict, children: List) -> "TensorOperator":
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
        einsum_str = self._dot_rules.get((rank_A, rank_B))
        if einsum_str is None:
            raise NotImplementedError(
                f"No dot product implemented for tensor ranks ({rank_A}, {rank_B})."
            )
        return jnp.einsum(einsum_str, A, B, optimize="optimal")

    @jax.jit
    def ddot(self, A: Array, B: Array) -> Array:
        """Computes the double dot product between tensors A and B."""
        rank_A = self._get_rank(A)
        rank_B = self._get_rank(B)
        einsum_str = self._ddot_rules.get((rank_A, rank_B))
        if einsum_str is None:
            raise NotImplementedError(
                f"No double dot product implemented for tensor ranks ({rank_A}, {rank_B})."
            )
        return jnp.einsum(einsum_str, A, B, optimize="optimal")

    @jax.jit
    def trace(self, A: Array) -> Array:
        """Computes the trace of tensor A."""
        rank_A = self._get_rank(A)
        einsum_str = self._trace_rules.get(rank_A)
        if einsum_str is None:
            raise NotImplementedError(
                f"No trace implemented for tensor rank ({rank_A})."
            )
        return jnp.einsum(einsum_str, A, optimize="optimal")

    @jax.jit
    def trans(self, A: Array) -> Array:
        """Computes the transpose of tensor A."""
        rank_A = self._get_rank(A)
        einsum_str = self._trans_rules.get(rank_A)
        if einsum_str is None:
            raise NotImplementedError(
                f"No transpose implemented for tensor rank ({rank_A})."
            )
        return jnp.einsum(einsum_str, A, optimize="optimal")

    @jax.jit
    def dyad(self, A: Array, B: Array) -> Array:
        """Computes the dyadic product between tensors A and B."""
        rank_A = self._get_rank(A)
        rank_B = self._get_rank(B)
        einsum_str = self._dyad_rules.get((rank_A, rank_B))
        if einsum_str is None:
            raise NotImplementedError(
                f"No dyad implemented for tensor ranks ({rank_A}, {rank_B})."
            )
        return jnp.einsum(einsum_str, A, B, optimize="optimal")
