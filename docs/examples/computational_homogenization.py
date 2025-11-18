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
jax.config.update("jax_platforms", "cpu")
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
import time

import equinox as eqx
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from jax import Array
from skimage.morphology import disk

# %% [markdown]
# We start by importing the necessary libraries and configuring JAX for double-precision computations.
# %%
from xpektra import FFTTransform, SpectralOperator, SpectralSpace, make_field
from xpektra.projection_operator import GalerkinProjection
from xpektra.scheme import RotatedDifference
from xpektra.solvers.nonlinear import (  # noqa: E402
    conjugate_gradient,
    implicit_newton_solver,
    newton_krylov_solver,
)

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
)

# %%
dofs_shape = make_field(dim=ndim, shape=structure.shape, rank=2).shape


# %%
@eqx.filter_jit
def strain_energy(eps_flat: Array) -> Array:
    eps = eps_flat.reshape(dofs_shape)
    eps_sym = 0.5 * (eps + op.trans(eps))
    energy = 0.5 * jnp.multiply(lambda_field, op.trace(eps_sym) ** 2) + jnp.multiply(
        mu_field, op.trace(op.dot(eps_sym, eps_sym))
    )
    return energy.sum()


compute_stress = jax.jit(jax.jacrev(strain_energy))

# %%
Ghat = GalerkinProjection(scheme=diff_scheme)


# %%
@eqx.filter_jit
def residual_fn(eps_flat: Array) -> Array:
    """
    This makes instances of this class behave like a function.
    It takes only the flattened vector of unknowns, as required by the solver.
    """
    eps_flat = eps_flat.reshape(-1)
    sigma = compute_stress(eps_flat)
    residual_field = op.inverse(Ghat.project(op.forward(sigma.reshape(dofs_shape))))
    return jnp.real(residual_field).reshape(-1)


@eqx.filter_jit
def jacobian_fn(deps_flat: Array) -> Array:
    """
    The Jacobian is a linear operator, so its __call__ method
    represents the Jacobian-vector product.
    """

    deps_flat = deps_flat.reshape(-1)
    dsigma = compute_stress(deps_flat)
    jvp_field = op.inverse(Ghat.project(op.forward(dsigma.reshape(dofs_shape))))
    return jnp.real(jvp_field).reshape(-1)


# %% [markdown]
# We can define a function to compute the local constitutive response given a macroscopic strain. The function will solve the local problem using a Newton-Krylov solver and return the homogenized stress.


# %%
from xpektra.solvers.nonlinear import NewtonSolver


@eqx.filter_jit
def local_constitutive_update(macro_strain):
    # set macroscopic loading
    deps = jnp.array(make_field(dim=2, shape=structure.shape, rank=2))
    deps = deps.at[:, :, 0, 0].set(macro_strain[0])
    deps = deps.at[:, :, 1, 1].set(macro_strain[1])
    deps = deps.at[:, :, 0, 1].set(macro_strain[2] / 2.0)
    deps = deps.at[:, :, 1, 0].set(macro_strain[2] / 2.0)

    # initial residual: distribute "deps" over grid
    b = -residual_fn(deps)

    solver = NewtonSolver(
        b=-residual_fn(deps),
        jacobian=jacobian_fn,
        tol=1e-8,
        max_iter=20,
        krylov_solver=conjugate_gradient,
        krylov_tol=1e-8,
        krylov_max_iter=20,
    )

    """
    eps = implicit_newton_solver(
        x=deps.reshape(-1),
        b=b,
        gradient=residual_fn,
        jacobian=jacobian_fn,
        tol=1e-8,
        max_iter=20,
        krylov_solver=conjugate_gradient,
        krylov_tol=1e-8,
        krylov_max_iter=20,
    )
    """
    eps = solver._solve(x=deps.reshape(-1), f=residual_fn)
    sig = compute_stress(eps).reshape(dofs_shape)

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
print(tangent)
