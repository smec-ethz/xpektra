import numpy as np
import jax
import equinox as eqx



class Operator:
    fourier = "fourier"
    forward_difference = "forward-difference"
    central_difference = "central-difference"
    backward_difference = "backward-difference"
    rotated_difference = "rotated-difference"
    four_central_difference = "4-central-difference"
    six_central_difference = "6-central-difference"
    eight_central_difference = "8-central-difference"

def gradient_operator(N, ndim, length=1.0, operator=Operator.fourier):
    Δ = length / N

    freq = np.arange(-(N - 1) / 2, +(N + 1) / 2, dtype="int64") / length
    ξ = 2 * np.pi * freq  # 2*pi*(n)/samplingspace/n https://arxiv.org/pdf/1412.8398

    if ndim == 1:
        Dξ = np.zeros([ndim, N], dtype="complex")  # frequency vectors
        wavenumbers = [ξ]

        kmax_dealias = ξ.max() * 2.0 / 3.0  # The Nyquist mode
        dealias = np.array(np.abs(wavenumbers[0]) < kmax_dealias, dtype=bool)

    elif ndim == 2:
        Dξ = np.zeros([ndim, N, N], dtype="complex")  # frequency vectors
        ξx, ξy = np.meshgrid(ξ, ξ)
        wavenumbers = [ξx, ξy]

        kmax_dealias = ξx.max() * 2.0 / 3.0  # The Nyquist mode
        dealias = np.array(
            (np.abs(wavenumbers[0]) < kmax_dealias)
            * (np.abs(wavenumbers[1]) < kmax_dealias),
            dtype=bool,
        )

    elif ndim == 3:
        Dξ = np.zeros([ndim, N, N, N], dtype="complex")  # frequency vectors
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
        N,
    ] * ndim  # number of voxels in all directions

    factor = 1.0
    ι = 1j

    if ndim > 1:
        for j in range(ndim):
            factor *= 0.5 * (1 + np.exp(ι * wavenumbers[j] * Δ))

    for i in range(ndim):
        ξ = wavenumbers[i]

        if operator == Operator.fourier:
            Dξ[i] = ι * ξ

        elif operator == Operator.forward_difference:
            Dξ[i] = (np.exp(ι * ξ * Δ) - 1) / Δ

        elif operator == Operator.central_difference:
            Dξ[i] = ι * np.sin(ξ * Δ) / Δ

        elif operator == Operator.backward_difference:
            Dξ[i] = (1 - np.exp(-ι * ξ * Δ)) / Δ

        elif operator == Operator.rotated_difference and ndim > 1:
            Dξ[i] = 2 * ι * np.tan(ξ * Δ / 2) * factor / Δ

        elif operator == Operator.four_central_difference:
            Dξ[i] = ι * (8 * np.sin(ξ * Δ) / (6 * Δ) - np.sin(2 * ξ * Δ) / (6 * Δ))

        elif operator == Operator.six_central_difference:
            Dξ[i] = ι * (
                9 * np.sin(ξ * Δ) / (6 * Δ)
                - 3 * np.sin(2 * ξ * Δ) / (10 * Δ)
                + np.sin(3 * ξ * Δ) / (30 * Δ)
            )

        elif operator == Operator.eight_central_difference:
            Dξ[i] = ι * (
                8 * np.sin(ξ * Δ) / (5 * Δ)
                - 2 * np.sin(2 * ξ * Δ) / (5 * Δ)
                + 8 * np.sin(3 * ξ * Δ) / (105 * Δ)
                - np.sin(4 * ξ * Δ) / (140 * Δ)
            )

        else:
            raise RuntimeError("Gradient operator not defined")

    if ndim == 1:
        return Dξ[0], dealias
    else:
        return Dξ, dealias
