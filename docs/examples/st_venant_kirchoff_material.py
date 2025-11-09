import jax

jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
jax.config.update("jax_enable_x64", True)  # use double-precision
jax.config.update("jax_platforms", "cpu")

import jax.numpy as jnp
from jax import Array

import numpy as np


import matplotlib.pyplot as plt
from skimage.morphology import disk, rectangle

from xpektra import (
    SpectralSpace,
    TensorOperator,
    make_field,
)
from xpektra.scheme import RotatedDifference, Fourier
from xpektra.projection_operator import GalerkinProjection
from xpektra.solvers.nonlinear import (  # noqa: E402
    conjugate_gradient_while,
    newton_krylov_solver,
)

import equinox as eqx

import time


ndim = 2
N = 299
length = 1.0

r = int(N / 3)

structure = np.zeros((N, N))
structure[:r, -r:] += rectangle(r, r)

tensor = TensorOperator(dim=ndim)
space = SpectralSpace(size=N, dim=ndim, length=length)

"""
# identity tensor (single tensor)
i = jnp.eye(ndim)

# identity tensors (grid)
I = jnp.einsum("ij,xy", i, jnp.ones([N, N]))  # 2nd order Identity tensor
I4 = jnp.einsum(
    "ijkl,xy->ijklxy", jnp.einsum("il,jk", i, i), jnp.ones([N, N])
)  # 4th order Identity tensor
I4rt = jnp.einsum("ijkl,xy->ijklxy", jnp.einsum("ik,jl", i, i), jnp.ones([N, N]))
I4s = (I4 + I4rt) / 2.0

II = tensor.dyad(I, I)
"""


def param(X, soft, hard):
    return soft * jnp.ones_like(X) * (X) + hard * jnp.ones_like(X) * (1 - X)


# %%
# material parameters
elastic_modulus = {"hard": 5.7, "soft": 0.57}  # N/mm2
poisson_modulus = {"hard": 0.386, "soft": 0.386}

# lames constant
lambda_modulus = {}
shear_modulus = {}
bulk_modulus = {}

for key in elastic_modulus.keys():
    lambda_modulus[key] = (
        poisson_modulus[key]
        * elastic_modulus[key]
        / ((1 + poisson_modulus[key]) * (1 - 2 * poisson_modulus[key]))
    )

    shear_modulus[key] = elastic_modulus[key] / (2 * (1 + poisson_modulus[key]))

    bulk_modulus[key] = lambda_modulus[key] + 2 * shear_modulus[key] / 3

# %%
# material parameters
K = param(
    structure, soft=bulk_modulus["soft"], hard=bulk_modulus["hard"]
)  # bulk      modulus
μ0 = param(
    structure, soft=shear_modulus["soft"], hard=shear_modulus["hard"]
)  # shear     modulus
λ0 = param(
    structure, soft=lambda_modulus["soft"], hard=lambda_modulus["hard"]
)  # shear     modulus


Ghat = GalerkinProjection(
    scheme=RotatedDifference(space=space), tensor_op=tensor
).compute_operator()

I = make_field(dim=ndim, N=N, rank=2)
I[:, :, 0, 0] = 1
I[:, :, 1, 1] = 1

dofs_shape = make_field(dim=ndim, N=N, rank=2).shape


@eqx.filter_jit
def green_lagrange_strain(F: Array) -> Array:
    return 0.5 * (tensor.dot(tensor.trans(F), F) - I)


@eqx.filter_jit
def strain_energy(F_flat: Array) -> float:
    F = F_flat.reshape(dofs_shape)
    E = green_lagrange_strain(F)
    E = 0.5 * (E + tensor.trans(E))
    energy = 0.5 * jnp.multiply(λ0, tensor.trace(E) ** 2) + jnp.multiply(
        μ0, tensor.trace(tensor.dot(E, E))
    )
    return energy.sum()


compute_stress = jax.jacrev(strain_energy)


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
    def __call__(self, F_flat: Array) -> Array:
        """
        This makes instances of this class behave like a function.
        It takes only the flattened vector of unknowns, as required by the solver.
        """
        start_time = time.time()
        sigma = compute_stress(F_flat)  # Assumes compute_stress is defined elsewhere
        end_time = time.time()
        jax.debug.print(
            "Time taken to compute sigma: {:.14f} seconds", end_time - start_time
        )

        residual_field = self.space.ifft(
            self.tensor_op.ddot(
                self.Ghat, self.space.fft(sigma.reshape(self.dofs_shape))
            )
        )
        return jnp.real(residual_field).reshape(-1)


class Jacobian(eqx.Module):
    """A callable module that represents the Jacobian operator (tangent)."""

    Ghat: Array
    space: SpectralSpace = eqx.field(static=True)
    tensor_op: TensorOperator = eqx.field(static=True)
    dofs_shape: tuple = eqx.field(static=True)

    @eqx.filter_jit
    def __call__(self, dF_flat: Array, F_flat: Array) -> Array:
        """
        The Jacobian is a linear operator, so its __call__ method
        represents the Jacobian-vector product.
        """
        # Assuming linear elasticity, the tangent is the same as the residual operator
        start_time = time.time()
        tangents = jax.jvp(compute_stress, (F_flat,), (dF_flat,))[1]
        end_time = time.time()
        jax.debug.print(
            "Time taken to compute tangents: {:.14f} seconds", end_time - start_time
        )

        start_time = time.time()
        jvp_field = self.space.ifft(
            self.tensor_op.ddot(
                self.Ghat, self.space.fft(tangents.reshape(self.dofs_shape))
            )
        )
        end_time = time.time()
        jax.debug.print(
            "Time taken to compute jvp_field: {:.14f} seconds", end_time - start_time
        )
        return jnp.real(jvp_field).reshape(-1)


F = make_field(dim=ndim, N=N, rank=2)
F[:, :, 0, 0] = 1
F[:, :, 1, 1] = 1

residual_fn = Residual(Ghat=Ghat, space=space, tensor_op=tensor, dofs_shape=F.shape)
jacobian_fn = Jacobian(Ghat=Ghat, space=space, tensor_op=tensor, dofs_shape=F.shape)


dF = make_field(dim=ndim, N=N, rank=2)

applied_strains = jnp.diff(jnp.linspace(0, 0.1, num=2))
print(applied_strains)

for inc, dF_avg in enumerate(applied_strains):
    # solving for elasticity
    dF[:, :, 0, 1] = dF_avg
    b = -residual_fn(dF.reshape(-1))
    F = F + dF

    jacobian_partial = eqx.Partial(jacobian_fn, F_flat=F.reshape(-1))

    start_time = time.time()
    final_state = newton_krylov_solver(
        state=(dF, b, F),
        gradient=residual_fn,
        jacobian=jacobian_partial,
        tol=1e-6,
        max_iter=20,
        krylov_solver=conjugate_gradient_while,
        krylov_tol=1e-6,
        krylov_max_iter=20,
    )
    end_time = time.time()
    print(f"Time taken to solve for step {inc}: {end_time - start_time} seconds")
    F = final_state[2]

    print("step", inc, "time", inc)


P = compute_stress(F.reshape(-1)).reshape(dofs_shape)

import matplotlib.pyplot as plt  # noqa: E402

fig = plt.figure(figsize=(5, 5))
ax = plt.axes()
cax = ax.imshow(P.at[:, :, 0, 1].get(), cmap="managua_r", origin="lower")
fig.colorbar(cax, label=r"$\epsilon_{xy}$", orientation="horizontal", location="top")
plt.show()
