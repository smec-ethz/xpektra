import jax  # type: ignore
import jax.numpy as jnp  # type: ignore
import numpy as np
import functools

import itertools

from spectralsolver.space import SpectralSpace, DifferentialMode


def compute_differential_operator(
    ind: jnp.ndarray,
    space: SpectralSpace,
    diff_mode: DifferentialMode,
) -> jnp.ndarray:
    freq = jnp.array([space.frequency_vector() for ii in range(space.dim)])
    Δ = space.length / space.size

    ξ = jnp.empty(space.dim, dtype="complex")
    Dξ = jnp.empty(space.dim, dtype="complex")

    factor = 1.0
    for jj in range(space.dim):
        index = ind.at[jj].get()
        freq_jj = freq.at[jj].get()

        factor *= 0.5 * (
            1 + jnp.exp(space.iota * 2 * jnp.pi * freq_jj.at[index].get() * Δ)
        )
    for ii in range(space.dim):
        index = ind.at[ii].get()
        freq_ii = freq.at[ii].get()
        ξ = ξ.at[ii].set(2 * jnp.pi * freq_ii.at[index].get())
        Dξ = Dξ.at[ii].set(space.differential_vector(ξ.at[ii].get(), diff_mode, factor))

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

    δ = lambda i, j: float(i == j)  # noqa: E731

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
    space: SpectralSpace,
    diff_mode: DifferentialMode = DifferentialMode.fourier,
) -> np.ndarray:
    ndim = space.dim
    grid_size = (space.size,) * ndim
    G = np.zeros((ndim, ndim, ndim, ndim) + grid_size, dtype="complex")

    grid_indices = np.array(list(itertools.product(*[range(n) for n in grid_size])))
    partial_compute_differential_operator = functools.partial(
        compute_differential_operator, space=space, diff_mode=diff_mode
    )

    _map = jax.vmap(partial_compute_differential_operator)
    Dξs = _map(
        jnp.array(grid_indices),
    )
    Dξs = np.array(Dξs)

    G = optimized_projection_fill(G, Dξs, grid_size)

    return G


@functools.partial(jax.jit, static_argnames=["grid_size", "length", "diff_mode"])
def compute_projection_operator_legacy(
    grid_size, length=1, diff_mode=DifferentialMode.forward_difference
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
    δ = lambda i, j: float(i == j)  # noqa: E731

    iota = 1j  # iota

    for i, j, l, m in itertools.product(range(ndim), repeat=4):
        for ind in itertools.product(*[range(n) for n in grid_size]):
            ξ = np.empty(ndim, dtype="complex")
            Dξ = np.empty(ndim, dtype="complex")

            factor = 1.0
            for jj in range(ndim):
                factor *= 0.5 * (1 + np.exp(iota * 2 * np.pi * freq[jj][ind[jj]] * Δ))

            for ii in range(ndim):
                ξ[ii] = (
                    2 * np.pi * freq[ii][ind[ii]]
                )  ## frequency vector # 2*pi*(n)/samplingspace/n https://arxiv.org/pdf/1412.8398

                if diff_mode == DifferentialMode.fourier:
                    Dξ[ii] = iota * ξ[ii]  ## fourier operator
                elif diff_mode == DifferentialMode.forward_difference:
                    Dξ[ii] = (np.exp(iota * ξ[ii] * Δ) - 1) / Δ
                elif diff_mode == DifferentialMode.central_difference:
                    Dξ[ii] = iota * np.sin(ξ[ii] * Δ) / Δ
                elif diff_mode == DifferentialMode.four_central_difference:
                    Dξ[ii] = iota * (
                        8 * np.sin(ξ[ii] * Δ) / (6 * Δ)
                        - np.sin(2 * ξ[ii] * Δ) / (6 * Δ)
                    )
                elif diff_mode == DifferentialMode.six_central_difference:
                    Dξ[ii] = iota * (
                        9 * np.sin(ξ[ii] * Δ) / (6 * Δ)
                        - 3 * np.sin(2 * ξ[ii] * Δ) / (10 * Δ)
                        + np.sin(3 * ξ[ii] * Δ) / (30 * Δ)
                    )
                elif diff_mode == DifferentialMode.eight_central_difference:
                    Dξ[ii] = iota * (
                        8 * np.sin(ξ[ii] * Δ) / (5 * Δ)
                        - 2 * np.sin(2 * ξ[ii] * Δ) / (5 * Δ)
                        + 8 * np.sin(3 * ξ[ii] * Δ) / (105 * Δ)
                        - np.sin(4 * ξ[ii] * Δ) / (140 * Δ)
                    )
                elif diff_mode == DifferentialMode.rotated_difference:
                    Dξ[ii] = 2 * iota * np.tan(ξ[ii] * Δ / 2) * factor / Δ

            if not Dξ.dot(np.conjugate(Dξ)) == 0:  # zero freq. -> mean
                Dξ_inverse = np.conjugate(Dξ) / (Dξ.dot(np.conjugate(Dξ)))
                𝔾[i, j, l, m][ind] = δ(i, m) * Dξ[j] * Dξ_inverse[l]
    return 𝔾


@functools.partial(jax.jit, static_argnames=["N", "length", "diff_mode"])
def compute_Ghat_2_1(N, length=1, diff_mode=DifferentialMode.forward_difference):
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
                if diff_mode == DifferentialMode.fourier:
                    Dξ[ii] = 1j * q[ii]
                elif diff_mode == DifferentialMode.central_difference:
                    Dξ[ii] = 1j * np.sin(q[ii] * Δ) / Δ
                elif diff_mode == DifferentialMode.four_central_difference:
                    Dξ[ii] = 1j * (
                        8 * np.sin(q[ii] * Δ) / (6 * Δ)
                        - np.sin(2 * q[ii] * Δ) / (6 * Δ)
                    )
                elif diff_mode == DifferentialMode.eight_central_difference:
                    Dξ[ii] = 1j * (
                        8 * np.sin(q[ii] * Δ) / (5 * Δ)
                        - 2 * np.sin(2 * q[ii] * Δ) / (5 * Δ)
                        + 8 * np.sin(3 * q[ii] * Δ) / (105 * Δ)
                        - np.sin(4 * q[ii] * Δ) / (140 * Δ)
                    )
                elif diff_mode == DifferentialMode.forward_difference:
                    Dξ[ii] = (np.exp(1j * q[ii] * Δ) - 1) / Δ
                else:
                    raise RuntimeError("diff_mode incorrectly defined")

            if not Dξ.dot(np.conjugate(Dξ)) == 0:  # zero freq. -> mean
                Dξ_inverse = np.conjugate(Dξ) / (Dξ.dot(np.conjugate(Dξ)))
                Ghat2_1[i, j][ind] = Dξ[i] * Dξ_inverse[j]

    return Ghat2_1
