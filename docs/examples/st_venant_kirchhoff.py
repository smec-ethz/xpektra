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
import jax

jax.config.update("jax_enable_x64", True)  # use double-precision

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from jax import Array
from jax._src.export.shape_poly import DimSize
from skimage.morphology import rectangle
from soldis.linear import CG
from soldis.newton import NewtonSolver, NewtonSolverOptions

# %%
from xpektra import SpectralSpace, make_field
from xpektra.projection_operator import GalerkinProjection
from xpektra.scheme import RotatedDifference
from xpektra.spectral_operator import SpectralOperator
from xpektra.transform import FFTTransform

# %% [markdown]
# ## constructing a dual phase RVE

# %%
ndim = 2
N = 199
length = 1.0

r = int(N / 3)

structure = np.zeros((N, N))
structure[:r, -r:] += rectangle(r, r)

plt.figure(figsize=(3, 3))
cb = plt.imshow(structure, origin="lower")
plt.colorbar(cb)
plt.show()

# %%
fft_transform = FFTTransform(dim=ndim)
space = SpectralSpace(
    lengths=(length,) * ndim, shape=structure.shape, transform=fft_transform
)
rotated_scheme = RotatedDifference(space=space)

op = SpectralOperator(
    scheme=rotated_scheme,
    space=space,
    projection=GalerkinProjection(),
)

dofs_shape = make_field(dim=ndim, shape=structure.shape, rank=2).shape

# %%
E1 = 0.57
E2 = 5.7
nu1 = 0.386
nu2 = 0.386

lambda1 = E1 * nu1 / ((1 + nu1) * (1 - 2 * nu1))
mu1 = E1 / (2 * (1 + nu1))

lambda2 = E2 * nu2 / ((1 + nu2) * (1 - 2 * nu2))
mu2 = E2 / (2 * (1 + nu2))


lambdas = lambda1 * (1.0 - structure) + lambda2 * structure
mus = mu1 * (1.0 - structure) + mu2 * structure

# %%
i = jnp.eye(ndim)
I = make_field(dim=ndim, shape=structure.shape, rank=2) + i


@jax.jit
def green_lagrange_strain(F: Array) -> Array:
    return 0.5 * (op.dot(op.trans(F), F) - I)


@jax.jit
def strain_energy(F_flat: Array) -> float:
    F = F_flat.reshape(dofs_shape)
    E = green_lagrange_strain(F)
    E = 0.5 * (E + op.trans(E))
    energy = 0.5 * jnp.multiply(lambdas, op.trace(E) ** 2) + jnp.multiply(
        mus, op.trace(op.dot(E, E))
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
    sigma = compute_stress(eps_total.reshape(-1))
    residual_field = op.inverse(op.project(op.forward(sigma.reshape(dofs_shape))))
    return jnp.real(residual_field).reshape(-1)


def jac_fn(x: Array, macro_strain: Array) -> Array:

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


solver = NewtonSolver(
    residual_fn,
    jac=jac_fn,
    lin_solver=CG(),
    options=NewtonSolverOptions(tol=1e-8, maxiter=20, verbose=True),
)

# %%
applied_strains = jnp.diff(jnp.linspace(0, 0.1, num=6))
F_fluc_init = jnp.array(make_field(dim=ndim, shape=structure.shape, rank=2))
F_fluc_init = F_fluc_init.at[:, :, 0, 0].set(1.0)
F_fluc_init = F_fluc_init.at[:, :, 1, 1].set(1.0)


for inc, macro_defo in enumerate(applied_strains):
    state = solver.root(F_fluc_init.reshape(-1), macro_defo)
    dF_fluc = state.value.reshape(dofs_shape)

    F_fluc = (
        F_fluc_init + dF_fluc.reshape(dofs_shape) - jnp.eye(ndim)[None, None, :, :]
    )  # remove identity part
    F_fluc_init = F_fluc

    F = F_fluc + jnp.eye(ndim)[None, None, :, :] * (macro_defo)

    print(f"Increment {inc}: converged={state.converged}, iterations={state.iteration}")


P = compute_stress(F.reshape(-1)).reshape(dofs_shape)


# %%
fig = plt.figure(figsize=(5, 5))
ax = plt.axes()
cax = ax.imshow(P.at[:, :, 0, 1].get(), cmap="managua_r", origin="lower")
fig.colorbar(cax, label=r"$P_{xy}$", orientation="horizontal", location="top")
plt.show()


# %%
