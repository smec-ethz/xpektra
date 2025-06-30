import jax  # type: ignore

jax.config.update("jax_enable_x64", True)  # use double-precision
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
import jax.numpy as jnp  # type: ignore

import numpy as np
import functools

import itertools

from spectralsolvers.operators.spatial import Operator


def compute_differential_operator(
    ind: jnp.ndarray,
    freq: jnp.ndarray,
    operator: Operator,
    ndim: int,
    dx: float,
) -> jnp.ndarray:
    Δ = dx

    ι = 1j

    ξ = jnp.empty(ndim, dtype="complex")
    Dξ = jnp.empty(ndim, dtype="complex")
    factor = 1.0
    for jj in range(ndim):
        index = ind.at[jj].get()
        freq_jj = freq.at[jj].get()

        factor *= 0.5 * (1 + jnp.exp(ι * 2 * jnp.pi * freq_jj.at[index].get() * Δ))
    for ii in range(ndim):
        index = ind.at[ii].get()
        freq_ii = freq.at[ii].get()
        ξ = ξ.at[ii].set(2 * jnp.pi * freq_ii.at[index].get())
        if operator == Operator.fourier:
            Dξ = Dξ.at[ii].set(ι * ξ.at[ii].get())
        elif operator == Operator.forward_difference:
            Dξ = Dξ.at[ii].set((jnp.exp(ι * ξ.at[ii].get() * Δ) - 1) / Δ)
        elif operator == Operator.central_difference:
            Dξ = Dξ.at[ii].set(ι * jnp.sin(ξ.at[ii].get() * Δ) / Δ)
        elif operator == Operator.four_central_difference:
            Dξ = Dξ.at[ii].set(
                ι
                * (
                    8 * jnp.sin(ξ[ii] * Δ) / (6 * Δ)
                    - jnp.sin(2 * ξ.at[ii].get() * Δ) / (6 * Δ)
                )
            )
        elif operator == Operator.six_central_difference:
            Dξ = Dξ.at[ii].set(
                ι
                * (
                    9 * jnp.sin(ξ.at[ii].get() * Δ) / (6 * Δ)
                    - 3 * jnp.sin(2 * ξ.at[ii].get() * Δ) / (10 * Δ)
                    + jnp.sin(3 * ξ.at[ii].get() * Δ) / (30 * Δ)
                )
            )
        elif operator == Operator.eight_central_difference:
            Dξ = Dξ.at[ii].set(
                ι
                * (
                    8 * jnp.sin(ξ.at[ii].get() * Δ) / (5 * Δ)
                    - 2 * jnp.sin(2 * ξ.at[ii].get() * Δ) / (5 * Δ)
                    + 8 * jnp.sin(3 * ξ.at[ii].get() * Δ) / (105 * Δ)
                    - jnp.sin(4 * ξ.at[ii].get() * Δ) / (140 * Δ)
                )
            )
        elif operator == Operator.rotated_difference:
            Dξ = Dξ.at[ii].set(2 * ι * jnp.tan(ξ.at[ii].get() * Δ / 2) * factor / Δ)

    return Dξ


def optimized_projection_fill(
    G: np.ndarray, Dξs: np.ndarray, grid_size: tuple[int, ...]
) -> np.ndarray:
    ndim = len(grid_size)
    shape = grid_size
    N = np.prod(shape)

    # Flatten Dξs into shape (N, ndim)
    Dξs = Dξs.reshape(N, ndim)
    norm_sq = np.einsum("ni,ni->n", Dξs, np.conj(Dξs))  # shape (N,)

    # Avoid division by zero
    valid_mask = norm_sq != 0
    Dξ_inv = np.zeros_like(Dξs, dtype=np.complex128)
    Dξ_inv[valid_mask] = np.conj(Dξs[valid_mask]) / norm_sq[valid_mask, None]

    # Precompute grid indices
    grid_indices = list(itertools.product(*[range(n) for n in shape]))

    δ = lambda i, j: float(i == j)

    for i, j, l, m in itertools.product(range(ndim), repeat=4):
        if δ(i, m) == 0:
            continue  # skip computation entirely

        term = Dξs[:, j] * Dξ_inv[:, l]  # shape (N,)
        term[~valid_mask] = 0.0

        # Assign into G
        for index, ind in enumerate(grid_indices):
            G[i, j, l, m][ind] = δ(i, m) * term[index]

    return G


def compute_projection_operator(
    grid_size: tuple[int, ...],
    length: float = 1.0,
    operator: Operator = Operator.fourier,
) -> np.ndarray:
    ndim = len(grid_size)
    dx = length / grid_size[0]

    G = np.zeros((ndim, ndim, ndim, ndim) + grid_size, dtype="complex")

    freq = jnp.array(
        [
            np.arange(
                -(grid_size[ii] - 1) / 2.0, +(grid_size[ii] + 1) / 2.0, dtype="int64"
            )
            / length
            for ii in range(ndim)
        ]
    )

    grid_indices = np.array(list(itertools.product(*[range(n) for n in grid_size])))

    _map = jax.vmap(compute_differential_operator, in_axes=(0, None, None, None, None))
    Dξs = _map(jnp.array(grid_indices), freq, operator, ndim, dx)
    Dξs = np.array(Dξs)

    G = optimized_projection_fill(G, Dξs, grid_size)

    return G


@functools.partial(jax.jit, static_argnames=["grid_size", "length", "operator"])
def compute_projection_operator_legacy(
    grid_size, length=1, operator=Operator.forward_difference
):
    ndim = len(grid_size)
    Δ = length / grid_size[0]

    # projection operator
    𝔾 = np.zeros(
        (ndim, ndim, ndim, ndim) + grid_size, dtype="complex"
    )  # zero initialize

    # frequencies
    freq = [
        np.arange(-(grid_size[ii] - 1) / 2.0, +(grid_size[ii] + 1) / 2.0, dtype="int64")
        / length
        for ii in range(ndim)
    ]

    # Dirac delta function
    δ = lambda i, j: float(i == j)

    ι = 1j  # iota

    for i, j, l, m in itertools.product(range(ndim), repeat=4):
        for ind in itertools.product(*[range(n) for n in grid_size]):
            ξ = np.empty(ndim, dtype="complex")
            Dξ = np.empty(ndim, dtype="complex")

            factor = 1.0
            for jj in range(ndim):
                factor *= 0.5 * (1 + np.exp(ι * 2 * np.pi * freq[jj][ind[jj]] * Δ))

            for ii in range(ndim):
                ξ[ii] = (
                    2 * np.pi * freq[ii][ind[ii]]
                )  ## frequency vector # 2*pi*(n)/samplingspace/n https://arxiv.org/pdf/1412.8398

                if operator == Operator.fourier:
                    Dξ[ii] = ι * ξ[ii]  ## fourier operator
                elif operator == Operator.forward_difference:
                    Dξ[ii] = (np.exp(ι * ξ[ii] * Δ) - 1) / Δ
                elif operator == Operator.central_difference:
                    Dξ[ii] = ι * np.sin(ξ[ii] * Δ) / Δ
                elif operator == Operator.four_central_difference:
                    Dξ[ii] = ι * (
                        8 * np.sin(ξ[ii] * Δ) / (6 * Δ)
                        - np.sin(2 * ξ[ii] * Δ) / (6 * Δ)
                    )
                elif operator == Operator.six_central_difference:
                    Dξ[ii] = ι * (
                        9 * np.sin(ξ[ii] * Δ) / (6 * Δ)
                        - 3 * np.sin(2 * ξ[ii] * Δ) / (10 * Δ)
                        + np.sin(3 * ξ[ii] * Δ) / (30 * Δ)
                    )
                elif operator == Operator.eight_central_difference:
                    Dξ[ii] = ι * (
                        8 * np.sin(ξ[ii] * Δ) / (5 * Δ)
                        - 2 * np.sin(2 * ξ[ii] * Δ) / (5 * Δ)
                        + 8 * np.sin(3 * ξ[ii] * Δ) / (105 * Δ)
                        - np.sin(4 * ξ[ii] * Δ) / (140 * Δ)
                    )
                elif operator == Operator.rotated_difference:
                    Dξ[ii] = 2 * ι * np.tan(ξ[ii] * Δ / 2) * factor / Δ

            if not Dξ.dot(np.conjugate(Dξ)) == 0:  # zero freq. -> mean
                Dξ_inverse = np.conjugate(Dξ) / (Dξ.dot(np.conjugate(Dξ)))
                𝔾[i, j, l, m][ind] = δ(i, m) * Dξ[j] * Dξ_inverse[l]
    return 𝔾


@functools.partial(jax.jit, static_argnames=["N", "length", "operator"])
def compute_Ghat_2_1(N, length=1, operator=Operator.forward_difference):
    """
    Compute the projection operator for the 2nd order 1st derivative.
    """
    
    ndim = len(N)
    Δ = length / N[0]

    # PROJECTION IN FOURIER SPACE #############################################
    Ghat2_1 = np.zeros((ndim, ndim) + N, dtype="complex")  # zero initialize
    freq = [
        np.arange(-(N[ii] - 1) / 2.0, +(N[ii] + 1) / 2.0) / length for ii in range(ndim)
    ]

    for i, j in itertools.product(range(ndim), repeat=2):
        for ind in itertools.product(*[range(n) for n in N]):
            q = np.empty(ndim, dtype="complex")
            Dξ = np.empty(ndim, dtype="complex")
            for ii in range(ndim):
                q[ii] = 2 * np.pi * freq[ii][ind[ii]]  ## frequency vector
                if operator == Operator.fourier:
                    Dξ[ii] = 1j * q[ii]
                elif operator == Operator.central_difference:
                    Dξ[ii] = 1j * np.sin(q[ii] * Δ) / Δ
                elif operator == Operator.four_central_difference:
                    Dξ[ii] = 1j * (
                        8 * np.sin(q[ii] * Δ) / (6 * Δ)
                        - np.sin(2 * q[ii] * Δ) / (6 * Δ)
                    )
                elif operator == Operator.eight_central_difference:
                    Dξ[ii] = 1j * (
                        8 * np.sin(q[ii] * Δ) / (5 * Δ)
                        - 2 * np.sin(2 * q[ii] * Δ) / (5 * Δ)
                        + 8 * np.sin(3 * q[ii] * Δ) / (105 * Δ)
                        - np.sin(4 * q[ii] * Δ) / (140 * Δ)
                    )
                elif operator == Operator.forward_difference:
                    Dξ[ii] = (np.exp(1j * q[ii] * Δ) - 1) / Δ
                else:
                    raise RuntimeError("operator incorrectly defined")

            if not Dξ.dot(np.conjugate(Dξ)) == 0:  # zero freq. -> mean
                Dξ_inverse = np.conjugate(Dξ) / (Dξ.dot(np.conjugate(Dξ)))
                Ghat2_1[i, j][ind] = Dξ[i] * Dξ_inverse[j]

    return Ghat2_1
