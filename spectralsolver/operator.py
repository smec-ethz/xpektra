import jax.numpy as jnp  # type: ignore
from jax import Array
import equinox as eqx

from typing import Dict, Tuple
import numpy as np
from itertools import repeat
import itertools


from spectralsolver.space import SpectralSpace, DifferentialMode

# --- Define the Kronecker delta function ---
δ = lambda i, j: float(i == j)  # noqa: E731


# --- Define the einsum rules for dot product ---
DOT_EINSUM_DISPATCH: Dict[Tuple[int, int], str] = {
    (0, 0): "ij,ij->ij",
    (1, 1): "i...,i...->...",  # dot11: vector-vector
    (2, 1): "ij...,j...->i...",  # dot21: tensor-vector
    (2, 2): "ij...,jk...->ik...",  # dot22: tensor-tensor
    (2, 4): "ij...,jkmn...->ikmn...",  # dot24: tensor-tensor4
    (4, 2): "ijkl...,lm...->ijkm...",  # dot42: tensor4-tensor
}

# --- Define the einsum rules for double dot product ---
DDOT_EINSUM_DISPATCH: Dict[Tuple[int, int], str] = {
    (2, 2): "ij...,ji...->...",
    (4, 2): "ijkl...,lk...->ij...",
    (4, 4): "ijkl...,lkmn...->ijmn...",
}

# --- Define the einsum rules for dyad ---
DYAD_EINSUM_DISPATCH: Dict[Tuple[int, int], str] = {
    (2, 2): "ij...,kl...->ijkl...",
    (1, 1): "i...,j...->ij...",
}

# --- Define the einsum rules for trace ---
TRACE_EINSUM_DISPATCH: Dict[int, str] = {
    2: "ii...->...",
}

# --- Define the einsum rules for transpose ---
TRANS_EINSUM_DISPATCH: Dict[int, str] = {
    2: "ij...->ji...",
}


# --- Define the tensor operator ---
class TensorOperator(eqx.Module):
    dim: int

    @eqx.filter_jit
    def dot(self, A: Array, B: Array) -> Array:
        rank_A = len(A.shape[: -self.dim])
        rank_B = len(B.shape[: -self.dim])
        einsum_str = DOT_EINSUM_DISPATCH.get((rank_A, rank_B))
        if einsum_str is None:
            raise NotImplementedError(
                f"No dot product implemented for tensor ranks ({rank_A}, {rank_B}) "
                f"derived from shapes {A.shape} and {B.shape}."
            )
        return jnp.einsum(einsum_str, A, B, optimize="optimal")

    @eqx.filter_jit
    def ddot(self, A: Array, B: Array) -> Array:
        rank_A = len(A.shape[: -self.dim])
        rank_B = len(B.shape[: -self.dim])
        einsum_str = DDOT_EINSUM_DISPATCH.get((rank_A, rank_B))
        if einsum_str is None:
            raise NotImplementedError(
                f"No double dot product implemented for tensor ranks ({rank_A}, {rank_B}) "
                f"derived from shapes {A.shape} and {B.shape}."
            )
        return jnp.einsum(einsum_str, A, B, optimize="optimal")

    @eqx.filter_jit
    def trace(self, A: Array) -> Array:
        rank_A = len(A.shape[: -self.dim])
        einsum_str = TRACE_EINSUM_DISPATCH.get(rank_A)
        if einsum_str is None:
            raise NotImplementedError(
                f"No trace implemented for tensor rank ({rank_A}) derived from shape {A.shape}."
            )
        return jnp.einsum(einsum_str, A, optimize="optimal")

    @eqx.filter_jit
    def trans(self, A: Array) -> Array:
        rank_A = len(A.shape[: -self.dim])
        einsum_str = TRANS_EINSUM_DISPATCH.get(rank_A)
        if einsum_str is None:
            raise NotImplementedError(
                f"No transpose implemented for tensor rank ({rank_A}) derived from shape {A.shape}."
            )
        return jnp.einsum(einsum_str, A, optimize="optimal")

    @eqx.filter_jit
    def dyad(self, A: Array, B: Array) -> Array:
        rank_A = len(A.shape[: -self.dim])
        rank_B = len(B.shape[: -self.dim])
        einsum_str = DYAD_EINSUM_DISPATCH.get((rank_A, rank_B))
        if einsum_str is None:
            raise NotImplementedError(
                f"No dyad implemented for tensor ranks ({rank_A}, {rank_B}) derived from shapes {A.shape} and {B.shape}."
            )
        return jnp.einsum(einsum_str, A, B, optimize="optimal")


# --- Define the operator ---
class SpectralOperator(eqx.Module):
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
