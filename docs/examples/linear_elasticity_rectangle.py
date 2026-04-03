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
# This tutorial used Fourier-Galerkin method to solve a linear elasticity problem of a circular inclusion in a square matrix. The inclusion is a material with a different elastic properties than the matrix.

# %%
import jax

jax.config.update("jax_enable_x64", True)  # use double-precision
import jax.numpy as jnp
import numpy as np
from jax import Array
from soldis.linear import CG
from soldis.newton import NewtonSolver, NewtonSolverOptions

# %% [markdown]
#
# In this example, we solve a linear elasticity problem of a circular inclusion in a square matrix. The inclusion is a material with a different elastic properties than the matrix. We use the Fourier-Galerkin method to solve the problem.
#
# We import the necessary modules and set up the environment. The module `xpektra` contains the operators and solvers for the Fourier-Galerkin method. We import the `SpectralSpace`, `TensorOperator`, `make_field`, `RotatedDifference`, `Fourier`, `ForwardDifference`, `GalerkinProjection`, `conjugate_gradient_while`, and `newton_krylov_solver` modules to create the operators and solvers.
#
# %%
from xpektra import (
    SpectralSpace,
    make_field,
)
from xpektra.projection_operator import GalerkinProjection
from xpektra.scheme import RotatedDifference
from xpektra.spectral_operator import SpectralOperator
from xpektra.transform import FFTTransform

# %% [markdown]
# To simplify the execution of the code, we define a `ElasticityOperator` class that contains the Fourier-Galerkin operator, the spatial operators, the tensor operators, and the FFT and IFFT operators. The `__init__` method initializes the operator and the `__call__` method computes the stresses in the real space given as
#
# $$
# \mathcal{F}^{-1} \left( \mathbb{G}:\mathcal{F}(\mathbf{\sigma}) \right) = \mathbf{0}
# $$

# %% [markdown]
# We define the grid size and the length of the RVE and construct the structure of the RVE.

# %%
Nx = 151
Ny = 199
ndim = 2
lx = 0.75
ly = 1.0


# Create phase indicator (cylinder)
x = np.linspace(-lx / 2, lx / 2, Nx)
y = np.linspace(-ly / 2, ly / 2, Ny)


if ndim == 3:
    Y, X, Z = np.meshgrid(x, y, x, indexing="ij")  # (N, N, N) grid
    phase = jnp.where(X**2 + Z**2 <= (0.2 / np.pi), 1.0, 0.0)  # 20% vol frac
else:
    X, Y = np.meshgrid(x, y, indexing="ij")  # (N, N) grid
    phase = jnp.where(X**2 + Y**2 <= (0.2 / np.pi), 1.0, 0.0)

# %% [markdown]
# ## Definin the tensor operator and the spectral space

# %%
# tensor = TensorOperator(dim=ndim)
# space = SpectralSpace(size=N, dim=ndim, length=length)

fft_transform = FFTTransform(dim=ndim)
space = SpectralSpace(lengths=(lx, ly), shape=phase.shape, transform=fft_transform)
rotated_scheme = RotatedDifference(space=space)

op = SpectralOperator(
    scheme=rotated_scheme,
    space=space,
    projection=GalerkinProjection(),
)

# %% [markdown]
# Next, we define the material parameters.

# %%
# Material parameters [grids of scalars, shape (N,N,N)]
lambda1, lambda2 = 10.0, 1000.0
mu1, mu2 = 0.25, 2.5
lambdas = lambda1 * (1.0 - phase) + lambda2 * phase
mu = mu1 * (1.0 - phase) + mu2 * phase


# %% [markdown]
# The linear elasticity strain energy is given as
#
# $$
# W = \frac{1}{2} \int_{\Omega}  (\lambda \text{tr}(\epsilon)^2+ \mu \text{tr}(\epsilon : \epsilon ) ) d\Omega
# $$
#
# We define a python function to compute the strain energy and then use the `jax.jacrev` function to compute the stress tensor.

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

    sigma = compute_stress(eps_total)
    residual_field = op.inverse(op.project(op.forward(sigma.reshape(dofs_shape))))
    return jnp.real(residual_field).reshape(-1)


def jac_fn(x: Array, macro_strain: Array) -> Array:

    @jax.jit
    def mv(deps_flat: Array) -> Array:
        """
        The Jacobian is a linear operator, so its __call__ method
        represents the Jacobian-vector product.
        """

        deps_flat = deps_flat.reshape(-1)
        dsigma = compute_stress(deps_flat)
        jvp_field = op.inverse(op.project(op.forward(dsigma.reshape(dofs_shape))))
        return jnp.real(jvp_field).reshape(-1)

    return mv


solver = NewtonSolver(
    residual_fn,
    jac=jac_fn,
    lin_solver=CG(),
    options=NewtonSolverOptions(tol=1e-8, maxiter=20, verbose=True),
)


# %%
applied_strains = jnp.diff(jnp.linspace(0, 1e-2, num=5))
eps_fluc_init = make_field(dim=2, shape=phase.shape, rank=2)

# %%

for inc, macro_strain in enumerate(applied_strains):
    # solving for elasticity
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

ax3.plot(sig.at[:, :, 0, 0].get()[int(Nx / 2), :])
ax_twin = ax3.twinx()
ax_twin.plot(phase[int(Nx / 2), :], color="gray")
plt.show()


# %%
