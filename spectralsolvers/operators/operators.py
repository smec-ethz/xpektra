import numpy as np


def gradient_operator(N, ndim, length=1.0, operator="fourier"):
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

        if operator == "fourier":
            Dξ[i] = ι * ξ

        elif operator == "forward-difference":
            Dξ[i] = (np.exp(ι * ξ * Δ) - 1) / Δ

        elif operator == "central-difference":
            Dξ[i] = ι * np.sin(ξ * Δ) / Δ

        elif operator == "backward-difference":
            Dξ[i] = (1 - np.exp(-ι * ξ * Δ)) / Δ

        elif operator == "rotated-difference" and ndim > 1:
            Dξ[i] = 2 * ι * np.tan(ξ * Δ / 2) * factor / Δ

        elif operator == "4-central-difference":
            Dξ[i] = ι * (8 * np.sin(ξ * Δ) / (6 * Δ) - np.sin(2 * ξ * Δ) / (6 * Δ))

        elif operator == "6-central-difference":
            Dξ[i] = ι * (
                9 * np.sin(ξ * Δ) / (6 * Δ)
                - 3 * np.sin(2 * ξ * Δ) / (10 * Δ)
                + np.sin(3 * ξ * Δ) / (30 * Δ)
            )

        elif operator == "8-central-difference":
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
