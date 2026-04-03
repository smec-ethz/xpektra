import jax

jax.config.update("jax_enable_x64", True)  # use double-precision
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from jax import Array
from soldis.linear import CG
from soldis.newton import NewtonSolver, NewtonSolverOptions

from xpektra import (
    FFTTransform,
    RotatedDifference,
    SpectralOperator,
    SpectralSpace,
    make_field,
)
from xpektra.projection_operator import GalerkinProjection

phase = np.load("structure.npy")

N = phase.shape[0]
ndim = 2
length = 1


space = SpectralSpace(
    lengths=(length,) * ndim, shape=phase.shape, transform=FFTTransform(ndim)
)
op = SpectralOperator(
    scheme=RotatedDifference(space), space=space, projection=GalerkinProjection()
)


# Material parameters [grids of scalars, shape (N,N,N)]
lambda1, lambda2 = 10.0, 1000.0
mu1, mu2 = 0.25, 2.5
lambdas = lambda1 * (1.0 - phase) + lambda2 * phase
mu = mu1 * (1.0 - phase) + mu2 * phase


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


@jax.jit
def residual_fn(eps_fluc_flat: Array, macro_strain: Array) -> Array:
    """
    This makes instances of this class behave like a function.
    It takes only the flattened vector of unknowns, as required by the solver.
    """
    eps_fluc = eps_fluc_flat.reshape(dofs_shape)
    eps_macro = jnp.zeros(dofs_shape)
    eps_macro = eps_macro.at[..., 0, 0].set(macro_strain)
    eps_macro = eps_macro.at[..., 1, 1].set(macro_strain)

    eps_total = eps_fluc + eps_macro

    sigma = compute_stress(eps_total)
    residual_field = op.inverse(op.project(op.forward(sigma.reshape(dofs_shape))))
    return jnp.real(residual_field).reshape(-1)


solver = NewtonSolver(
    residual_fn,
    lin_solver=CG(),
    options=NewtonSolverOptions(tol=1e-8, maxiter=20, verbose=True),
)


applied_strains = jnp.diff(jnp.linspace(0, 1e-2, num=5))
eps_fluc_init = make_field(dim=2, shape=phase.shape, rank=2)

for inc, macro_strain in enumerate(applied_strains):
    # solving for elasticity

    state = solver.root(eps_fluc_init.reshape(-1), macro_strain)
    deps_fluc = state.value.reshape(dofs_shape)

    # update fluctuation strain
    eps_fluc = eps_fluc_init + deps_fluc.reshape(dofs_shape)

    # total strain
    eps = eps_fluc + jnp.eye(ndim)[None, None, :, :] * macro_strain


sig = compute_stress(eps).reshape(dofs_shape)

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(8, 3), layout="constrained")
cb1 = ax1.imshow(sig.at[:, :, 0, 0].get(), cmap="managua_r")

fig.colorbar(cb1, ax=ax1)
cb2 = ax2.imshow(eps.at[:, :, 0, 1].get(), cmap="managua_r")

fig.colorbar(cb2, ax=ax2)
ax3.plot(sig.at[:, :, 0, 0].get()[:, int(N / 2)])

ax_twin = ax3.twinx()
ax_twin.plot(phase[int(N / 2), :], color="gray")
plt.show()
