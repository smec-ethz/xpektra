import jax

jax.config.update("jax_enable_x64", True)  # use double-precision
import jax.numpy as jnp  # type: ignore
from jax import Array
import equinox as eqx

from typing import Dict, Tuple
import numpy as np
from itertools import repeat
import itertools


# --- Define the Kronecker delta function ---
δ = lambda i, j: float(i == j)


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

# --- Define the einsum rules for trace ---
TRACE_EINSUM_DISPATCH: Dict[int, str] = {
    2: "ii...->...",
}

# --- Define the einsum rules for transpose ---
TRANS_EINSUM_DISPATCH: Dict[int, str] = {
    2: "ij...->ji...",
}


# --- Define the fft and ifft functions ---
def _fft(x : Array, N : int, ndim : int) -> Array:
    return jnp.fft.fftshift(
        jnp.fft.fftn(
            jnp.fft.ifftshift(x),
            [
                N,
            ]
            * ndim,
        )
    )


def _ifft(x : Array, N : int, ndim : int) -> Array:
    return jnp.fft.fftshift(
        jnp.fft.ifftn(
            jnp.fft.ifftshift(x),
            [
                N,
            ]
            * ndim,
        )
    )

# --- Define the gradient modes ---
class GradientMode:
    fourier = "fourier"
    forward_difference = "forward_difference"
    central_difference = "central_difference"
    backward_difference = "backward_difference"
    rotated_difference = "rotated_difference"
    four_central_difference = "four_central_difference"
    six_central_difference = "six_central_difference"


# --- Define the tensor operator ---
class TensorOperator(eqx.Module):
    dim: int
    size: int

    @eqx.filter_jit
    def fft(self, A: Array) -> Array:
        return _fft(A, self.size, self.dim)

    @eqx.filter_jit
    def ifft(self, A: Array) -> Array:
        return _ifft(A, self.size, self.dim)

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


# --- Define the operator ---
class Operator(eqx.Module):
    gradient_operator: Array
    dim: int
    size: int
    length: float
    mode: GradientMode
    tensor: TensorOperator
    grad_op: Array
    lap_op: Array
    div_op: Array

    def __init__(
        self,
        dim: int,
        size: int,
        length: float=1.0,
        mode: GradientMode = GradientMode.fourier,
    ):
        self.mode = mode
        self.dim = dim
        self.size = size
        self.length = length
        self.mode = mode
        self.tensor = TensorOperator(dim=dim, size=size)
        self.grad_op = self.gradient_operator()
        self.lap_op = self.tensor.dot(self.grad_op, self.grad_op)
        self.div_op = self.divergence_operator()

    def divergence_operator(self):
        div_op = jnp.zeros(
            (self.dim, self.dim, self.dim, self.size, self.size), dtype="complex"
        )
        for i, j, k in itertools.product(range(self.dim), repeat=3):
            div_op = div_op.at[i, j, k, :, :].set(self.grad_op[k] * δ(i, j))
        return div_op

    def gradient_operator(self):
        Δ = self.length / self.size

        freq = (
            np.arange(-(self.size - 1) / 2, +(self.size + 1) / 2, dtype="int64")
            / self.length
        )
        ξ = 2 * np.pi * freq  # 2*pi*(n)/samplingspace/n https://arxiv.org/pdf/1412.8398

        if self.dim == 1:
            Dξ = np.zeros([self.dim, self.size], dtype="complex")  # frequency vectors
            wavenumbers = [ξ]

            kmax_dealias = ξ.max() * 2.0 / 3.0  # The Nyquist mode
            dealias = np.array(np.abs(wavenumbers[0]) < kmax_dealias, dtype=bool)

        elif self.dim == 2:
            Dξ = np.zeros(
                [self.dim, self.size, self.size], dtype="complex"
            )  # frequency vectors
            ξx, ξy = np.meshgrid(ξ, ξ)
            wavenumbers = [ξx, ξy]

            kmax_dealias = ξx.max() * 2.0 / 3.0  # The Nyquist mode
            dealias = np.array(
                (np.abs(wavenumbers[0]) < kmax_dealias)
                * (np.abs(wavenumbers[1]) < kmax_dealias),
                dtype=bool,
            )

        elif self.dim == 3:
            Dξ = np.zeros(
                [self.dim, self.size, self.size, self.size], dtype="complex"
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

        shape = [
            self.size,
        ] * self.dim  # number of voxels in all directions

        factor = 1.0
        ι = 1j

        if self.dim > 1:
            for j in range(self.dim):
                factor *= 0.5 * (1 + np.exp(ι * wavenumbers[j] * Δ))

        for i in range(self.dim):
            ξ = wavenumbers[i]

            if self.mode == GradientMode.fourier:
                Dξ[i] = ι * ξ

            elif self.mode == GradientMode.forward_difference:
                Dξ[i] = (np.exp(ι * ξ * Δ) - 1) / Δ

            elif self.mode == GradientMode.central_difference:
                Dξ[i] = ι * np.sin(ξ * Δ) / Δ

            elif self.mode == GradientMode.backward_difference:
                Dξ[i] = (1 - np.exp(-ι * ξ * Δ)) / Δ

            elif self.mode == GradientMode.rotated_difference and self.dim > 1:
                Dξ[i] = 2 * ι * np.tan(ξ * Δ / 2) * factor / Δ

            elif self.mode == GradientMode.four_central_difference:
                Dξ[i] = ι * (8 * np.sin(ξ * Δ) / (6 * Δ) - np.sin(2 * ξ * Δ) / (6 * Δ))

            elif self.mode == GradientMode.six_central_difference:
                Dξ[i] = ι * (
                    9 * np.sin(ξ * Δ) / (6 * Δ)
                    - 3 * np.sin(2 * ξ * Δ) / (10 * Δ)
                    + np.sin(3 * ξ * Δ) / (30 * Δ)
                )

            elif self.mode == GradientMode.eight_central_difference:
                Dξ[i] = ι * (
                    8 * np.sin(ξ * Δ) / (5 * Δ)
                    - 2 * np.sin(2 * ξ * Δ) / (5 * Δ)
                    + 8 * np.sin(3 * ξ * Δ) / (105 * Δ)
                    - np.sin(4 * ξ * Δ) / (140 * Δ)
                )

            else:
                raise RuntimeError("Gradient mode not defined")

        if self.dim == 1:
            return Dξ[0]  # , dealias
        else:
            return Dξ  # , dealias

    @eqx.filter_jit
    def grad(self, A: Array) -> Array:
        rank = len(A.shape[: -self.dim])
        if rank != 0:
            raise ValueError("Gradient is not defined for non-scalar fields")
        return jnp.real(self.tensor.ifft(self.grad_op * self.tensor.fft(A)))

    @eqx.filter_jit
    def laplace(self, A: Array) -> Array:
        return jnp.real(self.tensor.ifft(self.tensor.ddot(self.lap_op, self.tensor.fft(A))))

    @eqx.filter_jit
    def div(self, A: Array) -> Array:
        return self.tensor.fft(
            jnp.einsum(
                "ijkxy, jkxy->ixy", self.div_op, self.tensor.fft(A), optimize="optimal"
            )
        )
    
    def fft(self, A: Array) -> Array:
        return self.tensor.fft(A)
    
    def ifft(self, A: Array) -> Array:
        return jnp.real(self.tensor.ifft(A))
    