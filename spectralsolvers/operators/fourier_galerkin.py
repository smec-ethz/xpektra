import jax  # type: ignore

jax.config.update("jax_compilation_cache_dir", "/cluster/scratch/mpundir/jax-cache")
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
import jax.numpy as jnp  # type: ignore

jax.config.update("jax_enable_x64", True)  # use double-precision
jax.config.update("jax_platforms", "cpu")
import numpy as np
import functools

import itertools


@functools.partial(jax.jit, static_argnames=["grid_size", "length", "operator"])
def compute_projection_operator(grid_size, length=1, operator="forward-difference"):
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


@functools.partial(jax.jit, static_argnames=["N", "length", "operator"])
def compute_Ghat_2_1(N, length=1, operator="forward-difference"):
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
                if operator == "fourier":
                    Dξ[ii] = 1j * q[ii]
                elif operator == "central-difference":
                    Dξ[ii] = 1j * np.sin(q[ii] * Δ) / Δ
                elif operator == "4-order-cd":
                    Dξ[ii] = 1j * (
                        8 * np.sin(q[ii] * Δ) / (6 * Δ)
                        - np.sin(2 * q[ii] * Δ) / (6 * Δ)
                    )
                elif operator == "8-order-cd":
                    Dξ[ii] = 1j * (
                        8 * np.sin(q[ii] * Δ) / (5 * Δ)
                        - 2 * np.sin(2 * q[ii] * Δ) / (5 * Δ)
                        + 8 * np.sin(3 * q[ii] * Δ) / (105 * Δ)
                        - np.sin(4 * q[ii] * Δ) / (140 * Δ)
                    )
                elif operator == "forward-difference":
                    Dξ[ii] = (np.exp(1j * q[ii] * Δ) - 1) / Δ
                else:
                    raise RuntimeError("operator incorrectly defined")

            if not Dξ.dot(np.conjugate(Dξ)) == 0:  # zero freq. -> mean
                Dξ_inverse = np.conjugate(Dξ) / (Dξ.dot(np.conjugate(Dξ)))
                Ghat2_1[i, j][ind] = Dξ[i] * Dξ_inverse[j]

    return Ghat2_1
