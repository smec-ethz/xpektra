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
        rank_A = self._get_rank(A)
        rank_B = self._get_rank(B)
        einsum_str = DOT_EINSUM_DISPATCH.get((rank_A, rank_B))
        if einsum_str is None:
            raise NotImplementedError(
                f"No dot product implemented for tensor ranks ({rank_A}, {rank_B})."
            )
        return jnp.einsum(einsum_str, A, B, optimize="optimal")

    # The other methods (ddot, trace, etc.) follow the exact same pattern
    # Just replace the dispatch dictionary and the error message string.

    @eqx.filter_jit
    def ddot(self, A: Array, B: Array) -> Array:
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
        rank_A = self._get_rank(A)
        einsum_str = TRACE_EINSUM_DISPATCH.get(rank_A)
        if einsum_str is None:
            raise NotImplementedError(
                f"No trace implemented for tensor rank ({rank_A})."
            )
        return jnp.einsum(einsum_str, A, optimize="optimal")

    @eqx.filter_jit
    def trans(self, A: Array) -> Array:
        rank_A = self._get_rank(A)
        einsum_str = TRANS_EINSUM_DISPATCH.get(rank_A)
        if einsum_str is None:
            raise NotImplementedError(
                f"No transpose implemented for tensor rank ({rank_A})."
            )
        return jnp.einsum(einsum_str, A, optimize="optimal")

    @eqx.filter_jit
    def dyad(self, A: Array, B: Array) -> Array:
        rank_A = self._get_rank(A)
        rank_B = self._get_rank(B)
        einsum_str = DYAD_EINSUM_DISPATCH.get((rank_A, rank_B))
        if einsum_str is None:
            raise NotImplementedError(
                f"No dyad implemented for tensor ranks ({rank_A}, {rank_B})."
            )
        return jnp.einsum(einsum_str, A, B, optimize="optimal")


# --- Define the operator ---
"""class SpectralOperator(eqx.Module):
    space: SpectralSpace
    diff_mode: DifferentialMode
    tensor: TensorOperator
    grad_op: Array
    lap_op: Array
    div_op: Array

    def __init__(self, space: SpectralSpace, diff_mode: DifferentialMode):
        self.space = space
        self.diff_mode = diff_mode
        self.tensor = TensorOperator(dim=space.dim)
        self.grad_op = self.gradient_operator()
        self.lap_op = self.tensor.dot(self.grad_op, self.grad_op)
        self.div_op = self.divergence_operator()

    def divergence_operator(self):
        div_op = jnp.zeros(
            (
                self.space.dim,
                self.space.dim,
                self.space.dim,
                self.space.size,
                self.space.size,
            ),
            dtype="complex",
        )
        for i, j, k in itertools.product(range(self.space.dim), repeat=3):
            div_op = div_op.at[i, j, k, :, :].set(self.grad_op[k] * δ(i, j))
        return div_op

    def gradient_operator(self):
        Δ = self.space.length / self.space.size

        ξ = self.space.wavenumber_vector()

        if self.space.dim == 1:
            Dξ = np.zeros(
                [self.space.dim, self.space.size], dtype="complex"
            )  # frequency vectors
            wavenumbers = [ξ]

            kmax_dealias = ξ.max() * 2.0 / 3.0  # The Nyquist mode
            dealias = np.array(np.abs(wavenumbers[0]) < kmax_dealias, dtype=bool)

        elif self.space.dim == 2:
            Dξ = np.zeros(
                [self.space.dim, self.space.size, self.space.size], dtype="complex"
            )  # frequency vectors
            ξx, ξy = np.meshgrid(ξ, ξ)
            wavenumbers = [ξx, ξy]

            kmax_dealias = ξx.max() * 2.0 / 3.0  # The Nyquist mode
            dealias = np.array(
                (np.abs(wavenumbers[0]) < kmax_dealias)
                * (np.abs(wavenumbers[1]) < kmax_dealias),
                dtype=bool,
            )

        elif self.space.dim == 3:
            Dξ = np.zeros(
                [self.space.dim, self.space.size, self.space.size, self.space.size],
                dtype="complex",
            )  # frequency vectors
            ξx, ξy, ξz = np.meshgrid(ξ, ξ, ξ)
            wavenumbers = [ξx, ξy, ξz]

            kmax_dealias = ξx.max() * 2.0 / 3.0  # The Nyquist mode
            dealias = np.array(
                (np.abs(wavenumbers[0]) < kmax_dealias)
                * (np.abs(wavenumbers[1]) < kmax_dealias)
                * (np.abs(wavenumbers[2]) < kmax_dealias),
                dtype=bool,
            )

        factor = 1.0

        if self.space.dim > 1:
            for j in range(self.space.dim):
                factor *= 0.5 * (1 + np.exp(self.space.iota * wavenumbers[j] * Δ))

        for i in range(self.space.dim):
            ξ = wavenumbers[i]
            Dξ[i] = self.space.differential_vector(
                xi=ξ, diff_mode=self.diff_mode, factor=factor
            )

        if self.space.dim == 1:
            return Dξ[0]  # , dealias
        else:
            return Dξ  # , dealias

    @eqx.filter_jit
    def grad(self, A: Array) -> Array:
        rank = len(A.shape[: -self.space.dim])
        if rank != 0:
            raise ValueError("Gradient is not defined for non-scalar fields")
        return jnp.real(self.space.ifft(self.grad_op * self.space.fft(A)))

    @eqx.filter_jit
    def laplace(self, A: Array) -> Array:
        return jnp.real(
            self.space.ifft(self.tensor.ddot(self.lap_op, self.space.fft(A)))
        )

    @eqx.filter_jit
    def div(self, A: Array) -> Array:
        return self.space.fft(
            jnp.einsum(
                "ijkxy, jkxy->ixy", self.div_op, self.space.fft(A), optimize="optimal"
            )
        )
"""
