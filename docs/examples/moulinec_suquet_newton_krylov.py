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

# %% [markdown]
# # Moulinec-Suquet as Newton-Krylov solver
#
# In this tutorial, we will solve a linear elasticity problem using Moulinec-Suquet's Green's operator but recasted the Lippmann-Schwinger equation as a Newton-Krylov solver.

# %%
from collections.abc import Callable
from functools import partial

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from jax import Array

jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
jax.config.update("jax_enable_x64", True)  # use double-precision
jax.config.update("jax_platforms", "cpu")

# %%
from soldis.linear import CG
from soldis.newton import NewtonSolver, NewtonSolverOptions
from xpektra.scheme import FourierScheme
from xpektra.spectral_operator import SpectralOperator
from xpektra.transform import FFTTransform

from xpektra import (
    MoulinecSuquetProjection,
    SpectralSpace,
    make_field,
)

# %% [markdown]
# Let us start by defining the RVE geometry. We will consider a 2D square RVE with a circular inclusion.

# %%
N = 99
ndim = 2
length = 1


# Create phase indicator (cylinder)
x = np.linspace(-0.5, 0.5, N)

if ndim == 3:
    Y, X, Z = np.meshgrid(x, x, x, indexing="ij")  # (N, N, N) grid
    phase = jnp.where(X**2 + Z**2 <= (0.2 / np.pi), 1.0, 0.0)  # 20% vol frac
else:
    X, Y = np.meshgrid(x, x, indexing="ij")  # (N, N) grid
    phase = jnp.where(X**2 + Y**2 <= (0.2 / np.pi), 1.0, 0.0)


plt.figure(figsize=(3, 3))
cb = plt.imshow(phase, origin="lower")
plt.colorbar(cb, label="Phase indicator")
plt.xlabel("$x$")
plt.ylabel("$y$")
plt.show()

# %% [markdown]
# Based on the phase indicator, we can now define the material parameters. We will consider a linear elastic material with different properties in the inclusion and the matrix.

# %%
# Material parameters [grids of scalars, shape (N,N,N)]
lambda1, lambda2 = 10.0, 1000.0
mu1, mu2 = 0.25, 2.5
lambdas = lambda1 * (1.0 - phase) + lambda2 * phase
mu = mu1 * (1.0 - phase) + mu2 * phase


# %% [markdown]
# ## Defining TensorOperator and SpectralSpace
#
#

# %%
fft_transform = FFTTransform(dim=ndim)
space = SpectralSpace(
    lengths=(length,) * ndim, shape=phase.shape, transform=fft_transform
)
fourier_scheme = FourierScheme(space=space)

op = SpectralOperator(
    scheme=fourier_scheme,
    space=space,
)


# %% [markdown]
# ## Defining the constitutive law


# %%
dofs_shape = make_field(dim=ndim, shape=phase.shape, rank=2).shape


@jax.jit
def _strain_energy(eps_flat: Array, lambdas: Array, mu: Array) -> Array:
    eps = eps_flat.reshape(dofs_shape)

    eps_sym = 0.5 * (eps + op.trans(eps))
    energy = 0.5 * jnp.multiply(lambdas, op.trace(eps_sym) ** 2) + jnp.multiply(
        mu, op.trace(op.dot(eps_sym, eps_sym))
    )
    return energy.sum()


# %% [markdown]
# ## Defining the reference material for Moulinec-Suquet projection
#
# To define the reference material, we will use the average properties of the material. We make use of `jax.jacrev` to compute the stress tensor as a function of the strain tensor. This way we do not need to store the reference material tensor in memory.

# %%
# Use average properties for the reference material
lambda0 = (lambda1 + lambda2) / 2.0
mu0 = (mu1 + mu2) / 2.0

material_energy = jax.jit(partial(_strain_energy, lambdas=lambdas, mu=mu))
reference_energy = jax.jit(partial(_strain_energy, lambdas=lambda0, mu=mu0))

compute_stress = jax.jacrev(material_energy)
compute_reference_stress = jax.jacrev(reference_energy)

# %% [markdown]
# To check the correctness of our reference material, we can compare the stress computed using the reference material tensor with the stress computed using the average properties.

# %%
i = jnp.eye(ndim)
I = make_field(dim=ndim, shape=(N, N), rank=2) + i  # Add i to broadcast

I4 = jnp.einsum("il,jk->ijkl", i, i)
I4rt = jnp.einsum("ik,jl->ijkl", i, i)
I4s = (I4 + I4rt) / 2.0
II = jnp.einsum("...ij,...kl->...ijkl", I, I)

# Build the constant C0 reference tensor [shape (3,3,3,3)]
C0 = lambda0 * II + 2.0 * mu0 * I4s

assert np.allclose(op.ddot(C0, I), compute_reference_stress(I)), (
    "Reference stress computation is incorrect"
)


# %% [markdown]
# We can now define the Moulinec-Suquet projection operator.

# %%
op = SpectralOperator(
    scheme=fourier_scheme,
    space=space,
    projection=MoulinecSuquetProjection(lambda0=lambda0, mu0=mu0),
)


# %% [markdown]
# ## Defining the residual and Jacobian


# %%


@jax.jit
def residual_fn(eps_fluc_flat: Array, macro_strain: Array) -> Array:
    """
    This makes instances of this class behave like a function.
    It takes only the flattened vector of unknowns, as required by the solver.
    """
    eps_fluc = eps_fluc_flat.reshape(dofs_shape)
    eps_macro = jnp.zeros(dofs_shape)
    eps_macro = eps_macro.at[:, :, 0, 0].set(macro_strain)
    eps_macro = eps_macro.at[:, :, 1, 1].set(macro_strain)
    eps_total = eps_fluc + eps_macro
    sigma = compute_stress(eps_total.reshape(-1))
    residual_field = op.inverse(op.project(op.forward(sigma.reshape(dofs_shape))))
    return jnp.real(residual_field).reshape(-1)


def jac_fn(x: Array, macro_strain: Array) -> Callable[[Array], Array]:

    @jax.jit
    def mv(dx: Array) -> Array:
        eps_macro = jnp.zeros(dofs_shape)
        eps_macro = eps_macro.at[:, :, 0, 0].set(macro_strain)
        eps_macro = eps_macro.at[:, :, 1, 1].set(macro_strain)
        x_total = x + eps_macro.reshape(-1)
        dsigma = jax.jvp(compute_stress, (x_total,), (dx,))[1]
        jvp_field = op.inverse(op.project(op.forward(dsigma.reshape(dofs_shape))))
        return jnp.real(jvp_field).reshape(-1)

    return mv


# %%
applied_strains = jnp.linspace(0, 1e-2, num=5)
eps_fluc_init = make_field(dim=2, shape=phase.shape, rank=2)

solver = NewtonSolver(
    residual_fn,
    jac=jac_fn,
    lin_solver=CG(),
    options=NewtonSolverOptions(tol=1e-8, maxiter=20, verbose=True),
)


for inc, macro_strain in enumerate(applied_strains):
    state = solver.root(eps_fluc_init.reshape(-1), macro_strain)
    deps_fluc = state.value.reshape(dofs_shape)
    # update fluctuation strain
    eps_fluc = eps_fluc_init + deps_fluc.reshape(dofs_shape)

    # update initial guess for next increment
    eps_fluc_init = eps_fluc

    # total strain
    eps = eps_fluc + jnp.eye(2)[None, None, :, :] * macro_strain

sig = compute_stress(eps)


# %%
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

ax3.plot(sig.at[:, :, 0, 0].get()[:, int(N / 2)])
ax_twin = ax3.twinx()
ax_twin.plot(phase[int(N / 2), :], color="gray")
plt.show()
