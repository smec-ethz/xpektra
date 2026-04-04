# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.18.1
#   kernelspec:
#     display_name: xpektra
#     language: python
#     name: python3
# ---

# %%
from re import A

import jax

jax.config.update("jax_enable_x64", True)

# %%
import random

import equinox as eqx
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from jax import Array
from scipy.spatial.distance import cdist
from soldis.linear import CG
from soldis.newton import (
    LineSearchNewtonSolver,
    LineSearchNewtonSolverOptions,
    NewtonSolver,
    NewtonSolverOptions,
)

# %%
from xpektra import (
    FFTTransform,
    GalerkinProjection,
    SpectralOperator,
    SpectralSpace,
    make_field,
)
from xpektra.scheme import RotatedDifference

random.seed(1)


def place_circle(matrix, n, r, x_center, y_center):
    for i in range(n):
        for j in range(n):
            if (i - x_center) ** 2 + (j - y_center) ** 2 <= r**2:
                matrix[i][j] = 1


def generate_matrix_with_circles(n, x, r):
    if r >= n:
        raise ValueError("Radius r must be less than the size of the matrix n")

    matrix = np.zeros((n, n), dtype=int)
    placed_circles = 0

    while placed_circles < x:
        x_center = random.randint(0, n - 1)
        y_center = random.randint(0, n - 1)

        # Check if the circle fits within the matrix bounds
        if (
            x_center + r < n
            and y_center + r < n
            and x_center - r >= 0
            and y_center - r >= 0
        ):
            previous_matrix = matrix.copy()
            place_circle(matrix, n, r, x_center, y_center)
            if not np.array_equal(previous_matrix, matrix):
                placed_circles += 1

    return matrix


N = 99
ndim = 2
length = 1.0

x = 1
r = 20
structure = generate_matrix_with_circles(N, x, r)


cb = plt.imshow(structure, cmap="viridis")
plt.colorbar(cb)
plt.show()


# %%
# Helper to map properties to grid
def map_prop(structure, val_soft, val_hard):
    return val_hard * structure + val_soft * (1 - structure)


# Properties
phase_contrast = 2.0
K_field = map_prop(structure, 0.833, phase_contrast * 0.833)
mu_field = map_prop(structure, 0.386, phase_contrast * 0.386)
H_field = map_prop(structure, 0.01, phase_contrast * 0.01)  # Normalized
sigma_y_field = map_prop(structure, 0.003, phase_contrast * 0.003)  # Normalized
n_exponent = 1.0


# %%
fft_transform = FFTTransform(dim=ndim)
space = SpectralSpace(lengths=(length, length), shape=(N, N), transform=fft_transform)
op = SpectralOperator(
    scheme=RotatedDifference(space=space), space=space, projection=GalerkinProjection()
)

dofs_shape = make_field(dim=ndim, shape=structure.shape, rank=2).shape

# %%
# Pre-compute Identity Tensors for the grid
# I2: (N,N,2,2), I4_dev: (N,N,2,2,2,2)

i = jnp.eye(ndim)
I = make_field(dim=ndim, shape=structure.shape, rank=ndim) * i  # Broadcasted Identity
II = op.dyad(I, I)  # Fourth-order Identity


class J2Plasticity(eqx.Module):
    """
    Encapsulates the J2 Plasticity constitutive law and return mapping.
    """

    K: Array
    mu: Array
    H: Array
    sigma_y: Array
    n: float

    def yield_stress(self, ep: Array) -> Array:
        return self.sigma_y + self.H * (ep**self.n)

    @jax.jit
    def compute_response(
        self, eps_total: Array, state_prev: tuple[Array, ...]
    ) -> tuple:
        """
        Computes stress and new state variables given total strain and history.
        state_prev = (eps_total_t, eps_elastic_t, ep_t)
        """
        eps_t, epse_t, ep_t = state_prev

        # Trial State (assume elastic step)
        # Delta eps = eps_total - eps_t
        # Trial elastic strain = old elastic strain + Delta eps
        epse_trial = epse_t + (eps_total - eps_t)

        # Volumetric / Deviatoric Split, 2D plane strain
        trace_epse = op.trace(epse_trial)
        epse_dev = epse_trial - (trace_epse[..., None, None] / 2.0) * jnp.eye(2)

        # Note: Be careful with 2D trace. If plane strain, tr=e11+e22.
        # If plane stress, e33 is non-zero. Assuming plane strain for simplicity.

        # Trial Stress
        # sigma_vol = K * trace_epse * I
        # sigma_dev = 2 * mu * epse_dev
        sigma_vol = self.K[..., None, None] * trace_epse[..., None, None] * jnp.eye(2)
        sigma_dev = 2.0 * self.mu[..., None, None] * epse_dev
        sigma_trial = sigma_vol + sigma_dev

        # Mises Stress
        # sig_eq = sqrt(3/2 * s:s)
        norm_s = jnp.sqrt(op.ddot(sigma_dev, sigma_dev))
        sig_eq_trial = jnp.sqrt(1.5) * norm_s

        # 2. Check Yield Condition
        sig_y_current = self.yield_stress(ep_t)
        phi = sig_eq_trial - sig_y_current

        # 3. Return Mapping (if plastic)
        # Mask for plastic points
        is_plastic = phi > 0

        # Plastic Multiplier Delta_gamma
        # Denom = 3*mu + H
        denom = 3.0 * self.mu + self.H  # (Linear hardening H' = H)
        d_gamma = jnp.where(is_plastic, phi / denom, 0.0)

        # Update State
        # Normal vector n = s_trial / |s_trial|
        # s_new = s_trial - 2*mu*d_gamma * n
        # This simplifies to scaling s_trial
        scale_factor = jnp.where(
            is_plastic, 1.0 - (3.0 * self.mu * d_gamma) / sig_eq_trial, 1.0
        )

        sigma_dev_new = sigma_dev * scale_factor[..., None, None]
        sigma_new = sigma_vol + sigma_dev_new

        # Update plastic strain
        ep_new = ep_t + d_gamma

        # Update elastic strain (back-calculate from stress)
        # eps_e_new = eps_e_trial - d_gamma * n * sqrt(3/2) ...
        # Easier: eps_e_new = C_inv : sigma_new
        # Or just update deviatoric part
        epse_dev_new = epse_dev * scale_factor[..., None, None]
        epse_vol_new = trace_epse[..., None, None] * jnp.eye(2)  # Volumetric is elastic
        epse_new = epse_dev_new + epse_vol_new

        return sigma_new, (eps_total, epse_new, ep_new)


# Instantiate Material
material = J2Plasticity(K_field, mu_field, H_field, sigma_y_field, n_exponent)


@jax.jit
def residual_fn(
    eps_fluc_flat: Array,
    macro_strain: Array,
    state_prev: tuple[Array, ...],
    material: J2Plasticity,
) -> Array:
    eps_fluc = eps_fluc_flat.reshape(dofs_shape)
    eps_macro = jnp.zeros(dofs_shape)
    eps_macro = eps_macro.at[:, :, 0, 0].set(macro_strain)
    eps_macro = eps_macro.at[:, :, 1, 1].set(-macro_strain)
    eps_total = eps_fluc + eps_macro

    sigma, _ = material.compute_response(eps_total, state_prev)
    residual_field = op.inverse(op.project(op.forward(sigma.reshape(dofs_shape))))
    return jnp.real(residual_field).reshape(-1)


"""
solver = LineSearchNewtonSolver(
    residual_fn,
    lin_solver=CG(tol=1e-5, maxiter=50),
    options=LineSearchNewtonSolverOptions(tol=1e-8, maxiter=20, verbose=True),
)
"""

solver = NewtonSolver(
    residual_fn,
    lin_solver=CG(tol=1e-5, maxiter=50),
    options=NewtonSolverOptions(tol=1e-8, maxiter=20, verbose=True),
)

# Initialize Fields
# Layout: (N, N, 2, 2)
eps_total = make_field(dim=ndim, shape=structure.shape, rank=2)
eps_elastic = make_field(dim=ndim, shape=structure.shape, rank=2)
ep_accum = make_field(dim=ndim, shape=structure.shape, rank=0)  # Scalar plastic strain
state_current = (eps_total, eps_elastic, ep_accum)
eps_fluc_init = make_field(
    dim=ndim, shape=structure.shape, rank=2
)  # Initial guess for fluctuation

# History storage
stress_history = []

# Load steps
n_steps = 200
max_strain = 0.1 * jnp.sqrt(3) / 2
strain_steps = jnp.linspace(0, max_strain, n_steps)

print("Starting Plasticity Simulation...")

for inc, macro_strain in enumerate(strain_steps[1:13]):
    state = solver.root(
        eps_fluc_init.reshape(-1), macro_strain, state_current, material
    )
    eps_fluc = state.value.reshape(dofs_shape)
    eps_fluc_init = eps_fluc  # initial guess for next step

    # Reconstruct total strain to update history variables
    eps_macro = jnp.zeros(dofs_shape)
    eps_macro = eps_macro.at[:, :, 0, 1].set(macro_strain)
    eps_macro = eps_macro.at[:, :, 1, 0].set(macro_strain)
    eps_total = eps_fluc + eps_macro
    final_sigma, state_current = material.compute_response(eps_total, state_current)

    avg_stress = jnp.mean(final_sigma, axis=(0, 1))
    stress_history.append(avg_stress[0, 1])


# Plot
plt.plot(strain_steps[1:13], stress_history, "-o")
plt.xlabel("Macroscopic Shear Strain")
plt.ylabel("Macroscopic Shear Stress")
plt.title("J2 Plasticity: Stress-Strain Curve")
plt.grid()
plt.show()
