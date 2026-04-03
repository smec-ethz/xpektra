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
# # Homogenization
#
# In this example, we demonstrate how to perform computational homogenization using the spectral solver framework. We will define a microstructure, apply macroscopic strains, and compute the effective material properties.
#
# We will make use of the differentiable nature of `xpektra` to compute the tangent stiffness matrix via automatic differentiation using `JAX`.

# %%
import jax

jax.config.update("jax_enable_x64", True)  # use double-precision
import time

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from jax import Array
from skimage.morphology import disk
from soldis.linear import CG
from soldis.newton import NewtonSolver, NewtonSolverOptions

# %% [markdown]
# We start by importing the necessary libraries and configuring JAX for double-precision computations.
# %%
from xpektra import (
    FFTTransform,
    GalerkinProjection,
    SpectralOperator,
    SpectralSpace,
    make_field,
)
from xpektra.scheme import RotatedDifference

# %% [markdown]
# ## Define Microstructure
#
# We define a simple microstructure consisting of a circular inclusion in a matrix material.

# %%
volume_fraction_percentage = 0.007

length = 0.1
H, L = (199, 199)

dx = length / H
dy = length / L

Hmid = int(H / 2)
Lmid = int(L / 2)
vol_inclusion = volume_fraction_percentage * (length * length)
r = (
    int(np.sqrt(vol_inclusion / np.pi) / dx) + 1
)  # Since the rounding off leads to smaller fraction therefore we add 1.

structure = jnp.zeros((H, L))
structure = structure.at[Hmid - r : Hmid + 1 + r, Lmid - r : Lmid + 1 + r].add(disk(r))

ndim = len(structure.shape)

plt.figure(figsize=(4, 4))
plt.imshow(structure, cmap="gray")
plt.colorbar()
plt.show()

# %% [markdown]
# ## Define Material Properties
#
# We assume the inclusion to be stiffer than the matrix.

# %%
# material parameters, lames constant
lambda_matrix = 2.0
mu_matrix = 1.0

lambda_inclusion = 10.0
mu_inclusion = 5.0

lambda_field = lambda_matrix * (1 - structure) + lambda_inclusion * (structure)
mu_field = mu_matrix * (1 - structure) + mu_inclusion * (structure)

# %% [markdown]
# ## Defining Spectral functions

# %%
fft_transform = FFTTransform(dim=ndim)
space = SpectralSpace(
    lengths=(length,) * ndim, shape=structure.shape, transform=fft_transform
)
diff_scheme = RotatedDifference(space=space)

op = SpectralOperator(
    scheme=diff_scheme,
    space=space,
    projection=GalerkinProjection(),
)

# %%
dofs_shape = make_field(dim=ndim, shape=structure.shape, rank=2).shape


# %%
@jax.jit
def strain_energy(eps_flat: Array) -> Array:
    eps = eps_flat.reshape(dofs_shape)
    eps_sym = 0.5 * (eps + op.trans(eps))
    energy = 0.5 * jnp.multiply(lambda_field, op.trace(eps_sym) ** 2) + jnp.multiply(
        mu_field, op.trace(op.dot(eps_sym, eps_sym))
    )
    return energy.sum()


compute_stress = jax.jit(jax.jacrev(strain_energy))

# %% [markdown]
# We now define the Galerkin projection operator.


# %%
@jax.jit
def residual_fn(eps_fluc_flat: Array, macro_strain: Array) -> Array:
    """
    A function that computes the residual of the problem based on the given macro strain.
    It takes only the flattened vector of fluctuation strain and a macro strain.

    Args:
        eps_fluc_flat: Flattened vector of fluctuation strain.
        macro_strain: Macro strain.

    Returns:
        Residual field.
    """

    eps_fluc = eps_fluc_flat.reshape(dofs_shape)
    eps_macro = jnp.zeros(dofs_shape)
    eps_macro = eps_macro.at[:, :, 0, 0].set(macro_strain[0])
    eps_macro = eps_macro.at[:, :, 1, 1].set(macro_strain[1])
    eps_macro = eps_macro.at[:, :, 0, 1].set(macro_strain[2] / 2.0)
    eps_macro = eps_macro.at[:, :, 1, 0].set(macro_strain[2] / 2.0)

    eps_total = eps_fluc + eps_macro
    eps_flat = eps_total.reshape(-1)
    sigma = compute_stress(eps_flat)
    residual_field = op.inverse(op.project(op.forward(sigma.reshape(dofs_shape))))
    return jnp.real(residual_field).reshape(-1)


def jac_fn(x: Array, macro_strain: Array):
    """Return the Jacobian matvec — independent of x and macro_strain for linear elasticity.

    Providing this explicitly avoids nested differentiation (JVP-of-jacrev) that the
    auto-generated Jacobian would require. This is especially important here because
    ``jax.jacfwd(local_constitutive_update)`` differentiates through the entire solver,
    so one fewer differentiation level significantly reduces the XLA graph size.
    """

    @jax.jit
    def mv(v: Array) -> Array:
        dsigma = compute_stress(v)
        jvp_field = op.inverse(op.project(op.forward(dsigma.reshape(dofs_shape))))
        return jnp.real(jvp_field).reshape(-1)

    return mv


# %% [markdown]
# We can define a function to compute the local constitutive response given a macroscopic strain. The function will solve the local problem using a Newton-Krylov solver and return the homogenized stress.


# %%

solver = NewtonSolver(
    residual_fn,
    lin_solver=CG(),
    jac=jac_fn,
    options=NewtonSolverOptions(tol=1e-8, maxiter=20, verbose=True),
)


@jax.jit
def local_constitutive_update(macro_strain):
    # initialize the initial guess for the local strain field
    eps_init = jnp.array(make_field(dim=2, shape=structure.shape, rank=2))

    # solve for the fluctuation strain field, the residual is the
    # right hand side is defined based on the initial guess
    state = solver.root(eps_init.reshape(-1), macro_strain)
    eps_fluc = state.value.reshape(dofs_shape)

    # compute the actual micro strain field
    # eps fluctuation is added to the initial guess
    eps_macro = eps_init.at[:, :, 0, 0].set(macro_strain[0])
    eps_macro = eps_macro.at[:, :, 1, 1].set(macro_strain[1])
    eps_macro = eps_macro.at[:, :, 0, 1].set(macro_strain[2] / 2.0)
    eps_macro = eps_macro.at[:, :, 1, 0].set(macro_strain[2] / 2.0)

    eps = eps_fluc + eps_macro

    # compute the actual micro stress field
    sig = compute_stress(eps.reshape(-1)).reshape(dofs_shape)

    # get the macro stress
    macro_sigma = jnp.array(
        [
            jnp.sum(sig.at[:, :, 0, 0].get() * dx * dy),
            jnp.sum(sig.at[:, :, 1, 1].get() * dx * dy),
            0.5
            * (
                jnp.sum(sig.at[:, :, 1, 0].get() * dx * dy)
                + jnp.sum(sig.at[:, :, 0, 1].get() * dx * dy)
            ),
        ]
    )
    macro_sigma = macro_sigma / length**2

    return macro_sigma, (macro_sigma, sig, eps)


# %% [markdown]
# We use the `jax.jacfwd` to differentiate the stress computation function to obtain the tangent operator.
#
# $$
# \mathbb{C} = \frac{\partial \sigma(\varepsilon_\text{macro})}{\partial \varepsilon_\text{macro}}
# $$

# %%
tangent_operator_and_state = jax.jacfwd(local_constitutive_update, has_aux=True)

# %%
deps = jnp.array([1.2, 1.0, 1])

start_time = time.time()
tangent, state = tangent_operator_and_state(deps)
end_time = time.time()
print(f"Time taken: {end_time - start_time} seconds")

# %% [markdown]
# The homogenized tangent stiffness matrix is thus computed as:
#

# %%
print(f"tangent: {tangent}")


# %% [markdown]
# Plotting the micro stress field

# %%
plt.figure(figsize=(4, 4))
plt.imshow(state[1][:, :, 0, 0], cmap="berlin", origin="lower")
plt.colorbar(label=r"$\sigma_{xx}$")
plt.title("Micro Stress Field")
plt.xlabel("X")
plt.ylabel("Y")
plt.show()

# %%
