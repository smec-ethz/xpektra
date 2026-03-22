from typing import Dict, Tuple

import equinox as eqx
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


class TensorOperator(eqx.Module):
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

    dim: int = eqx.field(static=True)
    _dot_rules: Dict[Tuple[int, int], str] = eqx.field(static=True)
    _ddot_rules: Dict[Tuple[int, int], str] = eqx.field(static=True)
    _dyad_rules: Dict[Tuple[int, int], str] = eqx.field(static=True)
    _trace_rules: Dict[int, str] = eqx.field(static=True)
    _trans_rules: Dict[int, str] = eqx.field(static=True)

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

    def _get_rank(self, A: Array) -> int:
        """Returns the tensor rank of A (total ndim minus spatial dims)."""
        rank = len(A.shape) - self.dim
        if rank < 0:
            raise ValueError(
                f"Array with shape {A.shape} has fewer dimensions than the "
                f"number of spatial dimensions ({self.dim})."
            )
        return rank

    @eqx.filter_jit
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

    @eqx.filter_jit
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

    @eqx.filter_jit
    def trace(self, A: Array) -> Array:
        """Computes the trace of tensor A."""
        rank_A = self._get_rank(A)
        einsum_str = self._trace_rules.get(rank_A)
        if einsum_str is None:
            raise NotImplementedError(
                f"No trace implemented for tensor rank ({rank_A})."
            )
        return jnp.einsum(einsum_str, A, optimize="optimal")

    @eqx.filter_jit
    def trans(self, A: Array) -> Array:
        """Computes the transpose of tensor A."""
        rank_A = self._get_rank(A)
        einsum_str = self._trans_rules.get(rank_A)
        if einsum_str is None:
            raise NotImplementedError(
                f"No transpose implemented for tensor rank ({rank_A})."
            )
        return jnp.einsum(einsum_str, A, optimize="optimal")

    @eqx.filter_jit
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
