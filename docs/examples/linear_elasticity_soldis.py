# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
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

# %% [markdown]
# Linear elasticity example using the `soldis` Newton-Krylov solver.
# This replaces xpektra's built-in Newton solver with soldis's `NewtonSolver` + `CG`,
# passing the macro strain as a dynamic argument to avoid recompilation per load step.

# %%
import jax

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platforms", "cpu")

import jax.numpy as jnp
import numpy as np
from jax import Array

# %%
from xpektra import SpectralSpace, make_field, GalerkinProjection
from xpektra.scheme import RotatedDifference
from xpektra.spectral_operator import SpectralOperator
from xpektra.transform import FFTTransform

from soldis.linear import CG
from soldis.newton import NewtonSolver, NewtonSolverOptions

# %% [markdown]
# ## Setup: grid, operators, materials

# %%
N = 199
ndim = 2
length = 1

x = np.linspace(-0.5, 0.5, N)
X, Y = np.meshgrid(x, x, indexing="ij")
phase = jnp.where(X**2 + Y**2 <= (0.2 / np.pi), 1.0, 0.0)

# %%
fft_transform = FFTTransform(dim=ndim)
space = SpectralSpace(
    lengths=(length,) * ndim, shape=phase.shape, transform=fft_transform
)
rotated_scheme = RotatedDifference(space=space)
op = SpectralOperator(scheme=rotated_scheme, space=space, projection=GalerkinProjection())

# %%
lambda1, lambda2 = 10.0, 1000.0
mu1, mu2 = 0.25, 2.5
lambdas = lambda1 * (1.0 - phase) + lambda2 * phase
mu = mu1 * (1.0 - phase) + mu2 * phase

# %% [markdown]
# ## Strain energy and stress

# %%
dofs_shape = make_field(dim=ndim, shape=phase.shape, rank=2).shape


@jax.jit
def strain_energy(eps_flat: Array) -> Array:
    eps = eps_flat.reshape(dofs_shape)
    eps_sym = 0.5 * (eps + op.trans(eps))
    energy = 0.5 * jnp.multiply(lambdas, op.trace(eps_sym) ** 2) + jnp.multiply(
        mu, op.trace(op.dot(eps_sym, eps_sym))
    )
    return energy.sum()


compute_stress = jax.jacrev(strain_energy)

# %% [markdown]
# ## Residual function
#
# The residual takes the fluctuation strain and the macro strain as separate arguments.
# By keeping `macro_strain` as a positional arg (not baked in via `partial`),
# it flows through `solver.root(y0, macro_strain)` as dynamic data — no recompilation
# when the macro strain changes between load steps.


# %%
@jax.jit
def residual_fn(eps_fluc_flat: Array, macro_strain: Array) -> Array:
    eps_fluc = eps_fluc_flat.reshape(dofs_shape)
    eps_macro = jnp.zeros(dofs_shape)
    eps_macro = eps_macro.at[:, :, 0, 0].set(macro_strain)
    eps_macro = eps_macro.at[:, :, 1, 1].set(macro_strain)
    eps_total = eps_fluc + eps_macro
    sigma = compute_stress(eps_total.reshape(-1))
    residual_field = op.inverse(op.project(op.forward(sigma.reshape(dofs_shape))))
    return jnp.real(residual_field).reshape(-1)


# %% [markdown]
# ## Explicit Jacobian
#
# For linear elasticity, the Jacobian is state-independent: ``dR/dε · δε`` is just the
# residual operator applied to the perturbation ``δε``.  Providing this explicitly avoids
# a nested differentiation (JVP-of-jacrev) that the auto-generated Jacobian would require.


# %%
def jac_fn(x: Array, macro_strain: Array):
    """Return the Jacobian matvec — independent of x and macro_strain for linear elasticity."""

    def mv(v: Array) -> Array:
        dsigma = compute_stress(v)
        jvp_field = op.inverse(op.project(op.forward(dsigma.reshape(dofs_shape))))
        return jnp.real(jvp_field).reshape(-1)

    return mv


# %% [markdown]
# ## Solve with soldis Newton-Krylov
#
# `NewtonSolver` is constructed once with:
# - `residual_fn` as the root-finding target
# - `CG()` as the matrix-free inner linear solver (Krylov)
# - `jac_fn` as the explicit Jacobian (avoids nested auto-diff)
#
# The solver is a JAX pytree. `fn`, `jac`, and `linear_solver` sit in aux_data (static),
# so their identity must remain stable across calls. Since we construct the solver once
# and pass `macro_strain` dynamically via `*args`, JAX compiles the while-loop only once.

# %%
solver = NewtonSolver(
    residual_fn,
    lin_solver=CG(),
    jac=jac_fn,
    options=NewtonSolverOptions(tol=1e-8, maxiter=20, verbose=True),
)

applied_strains = jnp.diff(jnp.linspace(0, 1e-2, num=5))
eps_fluc = jnp.array(make_field(dim=2, shape=phase.shape, rank=2))

for inc, macro_strain in enumerate(applied_strains):
    state = solver.root(eps_fluc.reshape(-1), macro_strain)
    eps_fluc = state.value.reshape(dofs_shape)
    print(f"Increment {inc}: converged={state.converged}, iterations={state.iteration}")

# total strain at final load level
total_macro_strain = jnp.sum(applied_strains)
eps = eps_fluc + jnp.eye(2)[None, None, :, :] * total_macro_strain

# %%
sig = compute_stress(eps.reshape(-1)).reshape(dofs_shape)

# %%
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(8, 3), layout="constrained")
cb1 = ax1.imshow(sig.at[:, :, 0, 0].get(), cmap="managua_r")
divider = make_axes_locatable(ax1)
cax = divider.append_axes("top", size="10%", pad=0.2)
fig.colorbar(
    cb1, cax=cax, label=r"$\sigma_{xx}$", orientation="horizontal", location="top"
)

cb2 = ax2.imshow(eps.at[:, :, 0, 1].get(), cmap="managua_r")
divider = make_axes_locatable(ax2)
cax = divider.append_axes("top", size="10%", pad=0.2)
fig.colorbar(
    cb2, cax=cax, label=r"$\varepsilon_{xy}$", orientation="horizontal", location="top"
)

ax3.plot(eps.at[:, :, 0, 0].get()[:, int(N / 2)])
ax_twin = ax3.twinx()
ax_twin.plot(phase[int(N / 2), :], color="gray")
plt.show()
