import os
os.environ["XLA_FLAGS"] = '--xla_force_host_platform_device_count=8' # Use 8 CPU devices

import jax

jax.config.update("jax_enable_x64", True)  # use double-precision
jax.config.update("jax_platforms", "cpu")
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.sharding import Mesh, PartitionSpec as P

import equinox as eqx


from functools import partial

import matplotlib.pyplot as plt
from skimage.morphology import disk
from xpektra import (
    DifferentialMode,
    SpectralSpace,
    TensorOperator,
    make_field,
)
from xpektra.scheme import RotatedDifference, Fourier
from xpektra.green_functions import fourier_galerkin
from xpektra.projection_operator import GalerkinProjection
from xpektra.solvers.nonlinear import (  # noqa: E402
    conjugate_gradient_while,
    newton_krylov_solver,
)

from jax_autovmap import autovmap
import time

N = 8
shape = (N, N)
length = 1.0
ndim = 2


def create_structure(N):
    Hmid = int(N / 2)
    Lmid = int(N / 2)
    r = int(N / 4)

    structure = np.ones((N, N))
    structure[Hmid - r : Hmid + r + 1, Lmid - r : Lmid + r + 1] -= disk(r)

    return structure


structure = create_structure(N)

print(structure)

mask = structure == 1
mask_eps = mask[..., None, None]

tensor = TensorOperator(dim=ndim)
space = SpectralSpace(size=N, dim=ndim, length=length)


def param(X, inclusion, solid):
    props = inclusion * jnp.ones_like(X) * (1 - X) + solid * jnp.ones_like(X) * (X)
    return props


phase_contrast = 1.0 / 1e3

# lames constant
lambda_modulus = {"solid": 1.0, "inclusion": phase_contrast}
shear_modulus = {"solid": 1.0, "inclusion": phase_contrast}

bulk_modulus = {}
bulk_modulus["solid"] = lambda_modulus["solid"] + 2 * shear_modulus["solid"] / 3
bulk_modulus["inclusion"] = (
    lambda_modulus["inclusion"] + 2 * shear_modulus["inclusion"] / 3
)

λ0 = param(
    structure, inclusion=lambda_modulus["inclusion"], solid=lambda_modulus["solid"]
)  # lame parameter
μ0 = param(
    structure, inclusion=shear_modulus["inclusion"], solid=shear_modulus["solid"]
)  # lame parameter
K0 = param(structure, inclusion=bulk_modulus["inclusion"], solid=bulk_modulus["solid"])


dofs_shape = make_field(dim=ndim, N=N, rank=2).shape

@eqx.filter_jit
def strain_energy_flattened(eps_flat):
    eps = eps_flat.reshape(dofs_shape)
    eps_unmasked = jnp.where(mask_eps, 0, eps)
    eps_sym = 0.5 * (eps_unmasked + tensor.trans(eps_unmasked))
    energy = 0.5 * jnp.multiply(λ0, tensor.trace(eps_sym) ** 2) + jnp.multiply(
        μ0, tensor.trace(tensor.dot(eps_sym, eps_sym))
    )
    return energy.sum()


@eqx.filter_jit
def strain_energy_masked(eps_flat):
    eps = eps_flat.reshape(dofs_shape)
    eps_masked = jnp.where(mask_eps, eps, 0)
    eps_sym = 0.5 * (eps_masked + tensor.trans(eps_masked))
    energy = 0.5 * jnp.multiply(λ0, tensor.trace(eps_sym) ** 2) + jnp.multiply(
        μ0, tensor.trace(tensor.dot(eps_sym, eps_sym))
    )
    return energy.sum()


@eqx.filter_jit
def total_strain_energy(eps_flat):
    return strain_energy_masked(eps_flat) + strain_energy_flattened(eps_flat)


@eqx.filter_jit
def strain_energy(eps):
    eps_sym = 0.5 * (eps + tensor.trans(eps))
    energy = 0.5 * jnp.multiply(λ0, tensor.trace(eps_sym) ** 2) + jnp.multiply(
        μ0, tensor.trace(tensor.dot(eps_sym, eps_sym))
    )
    return energy.sum()


I = make_field(dim=ndim, N=N, rank=2)
I[:, :, 0, 0] = 1
I[:, :, 1, 1] = 1

mask_eps

print(I[mask])

I_flat = I.reshape(-1)
num_devices = len(jax.devices())
jax_mesh = jax.make_mesh((num_devices,), ("batch",))
data_sharding = jax.sharding.NamedSharding(jax_mesh, jax.sharding.PartitionSpec("batch"))


compute_stress = jax.jacrev(strain_energy)
compute_stress_flattened = jax.jacrev(total_strain_energy)


start_time = time.time()
compute_stress(I)
end_time = time.time()
print(f"Time taken: {end_time - start_time} seconds")

start_time = time.time()
compute_stress(I)
end_time = time.time()
print(f"Time taken: {end_time - start_time} seconds")

start_time = time.time()
compute_stress_flattened(I_flat)
end_time = time.time()
print(f"Time taken flattened: {end_time - start_time} seconds")

start_time = time.time()
compute_stress_flattened(I_flat)
end_time = time.time()
print(f"Time taken flattened: {end_time - start_time} seconds")


#print(compute_stress(I).shape)
#print_shape(I)
#mapped_print_shape = jax.vmap(print_shape, in_axes=(0,))

#print(mapped_print_shape(jnp.array(I)).shape)

'''
Ghat = GalerkinProjection(
    scheme=RotatedDifference(space=space), tensor_op=tensor
).compute_operator()

eps = make_field(dim=2, N=N, rank=2)


class Residual(eqx.Module):
    """A callable module that computes the residual vector."""

    Ghat: Array
    space: SpectralSpace = eqx.field(static=True)
    tensor_op: TensorOperator = eqx.field(static=True)
    dofs_shape: tuple = eqx.field(static=True)

    # We can even pre-define the stress function if it's always the same
    # For this example, we'll keep your original `compute_stress` function
    # available in the global scope.

    @eqx.filter_jit
    def __call__(self, eps_flat: Array) -> Array:
        """
        This makes instances of this class behave like a function.
        It takes only the flattened vector of unknowns, as required by the solver.
        """
        eps = eps_flat.reshape(self.dofs_shape)
        sigma = compute_stress(eps)  # Assumes compute_stress is defined elsewhere
        residual_field = self.space.ifft(
            self.tensor_op.ddot(self.Ghat, self.space.fft(sigma))
        )
        return jnp.real(residual_field).reshape(-1)


class Jacobian(eqx.Module):
    """A callable module that represents the Jacobian operator (tangent)."""

    Ghat: Array
    space: SpectralSpace = eqx.field(static=True)
    tensor_op: TensorOperator = eqx.field(static=True)
    dofs_shape: tuple = eqx.field(static=True)

    @eqx.filter_jit
    def __call__(self, deps_flat: Array) -> Array:
        """
        The Jacobian is a linear operator, so its __call__ method
        represents the Jacobian-vector product.
        """
        deps = deps_flat.reshape(self.dofs_shape)
        # Assuming linear elasticity, the tangent is the same as the residual operator
        dsigma = compute_stress(deps)
        jvp_field = self.space.ifft(
            self.tensor_op.ddot(self.Ghat, self.space.fft(dsigma))
        )
        return jnp.real(jvp_field).reshape(-1)


residual_fn = Residual(Ghat=Ghat, space=space, tensor_op=tensor, dofs_shape=eps.shape)
jacobian_fn = Jacobian(Ghat=Ghat, space=space, tensor_op=tensor, dofs_shape=eps.shape)


applied_strains = jnp.diff(jnp.linspace(0, 2e-1, num=20))

deps = make_field(dim=2, N=N, rank=2)

for inc, deps_avg in enumerate(applied_strains):
    # solving for elasticity
    deps[:, :, 0, 0] = deps_avg
    b = -residual_fn(deps)
    eps = eps + deps

    final_state = newton_krylov_solver(
        state=(deps, b, eps),
        gradient=residual_fn,
        jacobian=jacobian_fn,
        tol=1e-8,
        max_iter=20,
        krylov_solver=conjugate_gradient_while,
        krylov_tol=1e-8,
        krylov_max_iter=20,
    )
    eps = final_state[2]

sig = compute_stress(final_state[2])

plt.figure(figsize=(4, 3))
ax = plt.axes()
ax.plot(sig.at[:, :, 1, 1].get()[int(N / 2), :])


ax2 = ax.twinx()
ax2.plot(structure[int(N / 2), :], color="gray")
plt.show()
'''