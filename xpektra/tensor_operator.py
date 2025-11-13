import jax.numpy as jnp  # type: ignore
from jax import Array
import equinox as eqx

from typing import Dict, Tuple


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
    dim: int  # Number of spatial dimensions, e.g., 2 for (nx, ny)

    @eqx.filter_jit
    def _get_rank(self, A: Array) -> int:
        """Helper to explicitly calculate the tensor rank."""
        # The number of tensor dimensions is the total number of dimensions
        # minus the number of spatial dimensions.
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
        einsum_str = DOT_EINSUM_DISPATCH.get((rank_A, rank_B))
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
        einsum_str = DDOT_EINSUM_DISPATCH.get((rank_A, rank_B))
        if einsum_str is None:
            raise NotImplementedError(
                f"No double dot product implemented for tensor ranks ({rank_A}, {rank_B})."
            )
        return jnp.einsum(einsum_str, A, B, optimize="optimal")

    @eqx.filter_jit
    def trace(self, A: Array) -> Array:
        """Computes the trace of tensor A."""

        rank_A = self._get_rank(A)
        einsum_str = TRACE_EINSUM_DISPATCH.get(rank_A)
        if einsum_str is None:
            raise NotImplementedError(
                f"No trace implemented for tensor rank ({rank_A})."
            )
        return jnp.einsum(einsum_str, A, optimize="optimal")

    @eqx.filter_jit
    def trans(self, A: Array) -> Array:
        """Computes the transpose of tensor A."""

        rank_A = self._get_rank(A)
        einsum_str = TRANS_EINSUM_DISPATCH.get(rank_A)
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
        einsum_str = DYAD_EINSUM_DISPATCH.get((rank_A, rank_B))
        if einsum_str is None:
            raise NotImplementedError(
                f"No dyad implemented for tensor ranks ({rank_A}, {rank_B})."
            )
        return jnp.einsum(einsum_str, A, B, optimize="optimal")
