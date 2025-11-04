import jax

jax.config.update("jax_enable_x64", True)  # use double-precision
jax.config.update("jax_platforms", "cpu")
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
import jax.numpy as jnp
import numpy as np
from jax import Array
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


N = 199
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
def strain_energy(eps_flat: Array) -> Array:
    eps = eps_flat.reshape(dofs_shape)
    eps_sym = 0.5 * (eps + tensor.trans(eps))
    energy = 0.5 * jnp.multiply(λ0, tensor.trace(eps_sym) ** 2) + jnp.multiply(
        μ0, tensor.trace(tensor.dot(eps_sym, eps_sym))
    )
    return energy.sum()


compute_stress = jax.jacrev(strain_energy)


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
        # eps = eps_flat.reshape(self.dofs_shape)
        sigma = compute_stress(eps_flat)  # Assumes compute_stress is defined elsewhere
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
    def __call__(self, deps_flat: Array) -> Array:
        """
        The Jacobian is a linear operator, so its __call__ method
        represents the Jacobian-vector product.
        """

        # Assuming linear elasticity, the tangent is the same as the residual operator
        dsigma = compute_stress(deps_flat)
        jvp_field = self.space.ifft(
            self.tensor_op.ddot(
                self.Ghat, self.space.fft(dsigma.reshape(self.dofs_shape))
            )
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

sig = compute_stress(final_state[2]).reshape(dofs_shape)

plt.figure(figsize=(4, 3))
ax = plt.axes()
ax.plot(sig.at[:, :, 1, 1].get()[int(N / 2), :])


ax2 = ax.twinx()
ax2.plot(structure[int(N / 2), :], color="gray")
plt.show()
