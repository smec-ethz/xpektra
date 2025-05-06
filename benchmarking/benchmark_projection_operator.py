import os
import time
import numpy as np
import jax

jax.config.update("jax_compilation_cache_dir", os.environ["JAX_CACHE_DIR"])
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)  # use double-precision
if os.environ["JAX_PLATFORM"] == "cpu":
    jax.config.update("jax_platforms", "cpu")
# jax.config.update("jax_traceback_filtering", "off")

print(jax.devices())

import jax.numpy as jnp
import itertools
import functools



# ----------------------------
# ORIGINAL (slow) implementation
# ----------------------------
def compute_differential_operator(ind, freq, operator, ndim, dx):
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
        if operator == "fourier":
            Dξ = Dξ.at[ii].set(ι * ξ.at[ii].get())
        elif operator == "forward-difference":
            Dξ = Dξ.at[ii].set((jnp.exp(ι * ξ.at[ii].get() * Δ) - 1) / Δ)
        elif operator == "central-difference":
            Dξ = Dξ.at[ii].set(ι * jnp.sin(ξ.at[ii].get() * Δ) / Δ)
        elif operator == "4-central-difference":
            Dξ = Dξ.at[ii].set(
                ι
                * (
                    8 * jnp.sin(ξ[ii] * Δ) / (6 * Δ)
                    - jnp.sin(2 * ξ.at[ii].get() * Δ) / (6 * Δ)
                )
            )
        elif operator == "6-central-difference":
            Dξ = Dξ.at[ii].set(
                ι
                * (
                    9 * jnp.sin(ξ.at[ii].get() * Δ) / (6 * Δ)
                    - 3 * jnp.sin(2 * ξ.at[ii].get() * Δ) / (10 * Δ)
                    + jnp.sin(3 * ξ.at[ii].get() * Δ) / (30 * Δ)
                )
            )
        elif operator == "8-central-difference":
            Dξ = Dξ.at[ii].set(
                ι
                * (
                    8 * jnp.sin(ξ.at[ii].get() * Δ) / (5 * Δ)
                    - 2 * jnp.sin(2 * ξ.at[ii].get() * Δ) / (5 * Δ)
                    + 8 * jnp.sin(3 * ξ.at[ii].get() * Δ) / (105 * Δ)
                    - jnp.sin(4 * ξ.at[ii].get() * Δ) / (140 * Δ)
                )
            )
        elif operator == "rotated-difference":
            Dξ = Dξ.at[ii].set(2 * ι * jnp.tan(ξ.at[ii].get() * Δ / 2) * factor / Δ)

    return Dξ


def optimized_projection_fill(G, Dξs, grid_size):
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


def compute_projection_operator_modified(grid_size, length=1, operator="fourier"):
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
    Dξs = jnp.array(Dξs)

    G = optimized_projection_fill(G, Dξs, grid_size)

    # should try distirbuted layout for cpu and gpus
    # https://colab.research.google.com/github/phlippe/uvadlc_notebooks/blob/master/docs/tutorial_notebooks/scaling/JAX/data_parallel_intro.ipynb

    return G


# ----------------------------
# OPTIMIZED JAX version
# ----------------------------
def compute_projection_operator_original(grid_size, length=1, operator="fourier"):
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

                if operator == "fourier":
                    Dξ[ii] = ι * ξ[ii]  ## fourier operator
                elif operator == "forward-difference":
                    Dξ[ii] = (np.exp(ι * ξ[ii] * Δ) - 1) / Δ
                elif operator == "central-difference":
                    Dξ[ii] = ι * np.sin(ξ[ii] * Δ) / Δ
                elif operator == "4-central-difference":
                    Dξ[ii] = ι * (
                        8 * np.sin(ξ[ii] * Δ) / (6 * Δ)
                        - np.sin(2 * ξ[ii] * Δ) / (6 * Δ)
                    )
                elif operator == "6-central-difference":
                    Dξ[ii] = ι * (
                        9 * np.sin(ξ[ii] * Δ) / (6 * Δ)
                        - 3 * np.sin(2 * ξ[ii] * Δ) / (10 * Δ)
                        + np.sin(3 * ξ[ii] * Δ) / (30 * Δ)
                    )
                elif operator == "8-central-difference":
                    Dξ[ii] = ι * (
                        8 * np.sin(ξ[ii] * Δ) / (5 * Δ)
                        - 2 * np.sin(2 * ξ[ii] * Δ) / (5 * Δ)
                        + 8 * np.sin(3 * ξ[ii] * Δ) / (105 * Δ)
                        - np.sin(4 * ξ[ii] * Δ) / (140 * Δ)
                    )
                elif operator == "rotated-difference":
                    Dξ[ii] = 2 * ι * np.tan(ξ[ii] * Δ / 2) * factor / Δ

            if not Dξ.dot(np.conjugate(Dξ)) == 0:  # zero freq. -> mean
                Dξ_inverse = np.conjugate(Dξ) / (Dξ.dot(np.conjugate(Dξ)))
                𝔾[i, j, l, m][ind] = δ(i, m) * Dξ[j] * Dξ_inverse[l]
    return 𝔾


# ----------------------------
# Benchmark and Validate
# ----------------------------
def benchmark_and_validate(grid_size):
    print(f"\n--- Grid size: {grid_size} ---")

    t0 = time.time()
    G_ref = compute_projection_operator_original(
        grid_size, operator="rotated-difference"
    )
    t1 = time.time()
    print(f"Original version time: {t1 - t0:.2f} s")

    t2 = time.time()
    G_jax = compute_projection_operator_modified(
        grid_size, operator="rotated-difference"
    )
    t3 = time.time()
    print(f"JAX optimized version time: {t3 - t2:.2f} s")

    # Compare outputs
    diff = np.max(np.abs(G_ref - G_jax))
    print(f"Max abs difference: {diff:.8e}")


# ----------------------------
# Run for 2D and 3D
# ----------------------------
benchmark_and_validate((31, 31))
benchmark_and_validate((31, 31, 31))
