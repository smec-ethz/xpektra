import jax

jax.config.update("jax_enable_x64", True)  # use double-precision
jax.config.update("jax_platforms", "cpu")
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
import jax.numpy as jnp
import numpy as np
from jax import Array

from functools import partial


from skimage.morphology import disk
from xpektra import (
    SpectralSpace,
    make_field,
)

from xpektra import SpectralSpace, make_field
from xpektra.transform import FFTTransform
from xpektra.scheme import RotatedDifference, FourierScheme
from xpektra.spectral_operator import SpectralOperator
from xpektra.projection_operator import GalerkinProjection
from xpektra.solvers.nonlinear import (  # noqa: E402
    conjugate_gradient_while,
    newton_krylov_solver,
)
import equinox as eqx

import time

volume_fraction_percentage = 0.007

# %%
length = 0.1
H, L = (299, 299)

dx = length / H
dy = length / L

Hmid = int(H / 2)
Lmid = int(L / 2)
vol_inclusion = volume_fraction_percentage * (length * length)
r = (
    int(np.sqrt(vol_inclusion / np.pi) / dx) + 1
)  # Since the rounding off leads to smaller fraction therefore we add 1.


structure = np.zeros((H, L))
structure[Hmid - r : Hmid + 1 + r, Lmid - r : Lmid + 1 + r] += disk(r)

ndim = len(structure.shape)
N = structure.shape[0]


def param(X, inclusion, solid):
    return inclusion * jnp.ones_like(X) * (X) + solid * jnp.ones_like(X) * (1 - X)


# material parameters
# lames constant
lambda_modulus = {"solid": 2.0, "inclusion": 10}
shear_modulus = {"solid": 1.0, "inclusion": 5}

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


fft_transform = FFTTransform(dim=ndim)
space = SpectralSpace(
    lengths=(length,) * ndim, shape=structure.shape, transform=fft_transform
)
diff_scheme = RotatedDifference(space=space)

op = SpectralOperator(
    scheme=diff_scheme,
    space=space,
)

dofs_shape = make_field(dim=2, shape=structure.shape, rank=2).shape

@eqx.filter_jit
def strain_energy(eps_flat: Array) -> Array:
    eps = eps_flat.reshape(dofs_shape)
    eps_sym = 0.5 * (eps + op.trans(eps))
    energy = 0.5 * jnp.multiply(λ0, op.trace(eps_sym) ** 2) + jnp.multiply(
        μ0, op.trace(op.dot(eps_sym, eps_sym))
    )
    return energy.sum()


compute_stress = jax.jit(jax.jacrev(strain_energy))

Ghat = GalerkinProjection(scheme=diff_scheme)


eps = make_field(dim=2, shape=structure.shape, rank=2)

class Residual(eqx.Module):
    """A callable module that computes the residual vector."""

    Ghat: Array
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
        eps_flat = eps_flat.reshape(-1)
        sigma = compute_stress(eps_flat)
        residual_field = op.inverse(
            Ghat.project(op.forward(sigma.reshape(self.dofs_shape)))
        )
        return jnp.real(residual_field).reshape(-1)


class Jacobian(eqx.Module):
    """A callable module that represents the Jacobian operator (tangent)."""

    Ghat: Array
    dofs_shape: tuple = eqx.field(static=True)

    @eqx.filter_jit
    def __call__(self, deps_flat: Array) -> Array:
        """
        The Jacobian is a linear operator, so its __call__ method
        represents the Jacobian-vector product.
        """

        deps_flat = deps_flat.reshape(-1)
        dsigma = compute_stress(deps_flat)
        jvp_field = op.inverse(
            Ghat.project(op.forward(dsigma.reshape(self.dofs_shape)))
        )
        return jnp.real(jvp_field).reshape(-1)

residual_fn = Residual(Ghat=Ghat, dofs_shape=eps.shape)
jacobian_fn = Jacobian(Ghat=Ghat, dofs_shape=eps.shape)


@eqx.filter_jit
def local_constitutive_update(macro_strain):
    # ----------------------------- NEWTON ITERATIONS -----------------------------
    # initialize stress and strain tensor                         [grid of tensors]
    eps = make_field(dim=2, shape=structure.shape, rank=2)
    # set macroscopic loading
    DE = jnp.zeros_like(eps)
    DE = DE.at[:, :, 0, 0].set(macro_strain[0])
    DE = DE.at[:, :, 1, 1].set(macro_strain[1])
    DE = DE.at[:, :, 0, 1].set(macro_strain[2] / 2.0)
    DE = DE.at[:, :, 1, 0].set(macro_strain[2] / 2.0)

    # initial residual: distribute "DE" over grid using "K4"
    b = -residual_fn(DE)
    eps = jax.lax.add(eps, DE)
    En = jnp.linalg.norm(eps)

    final_state = newton_krylov_solver(
        state=(DE, b, eps),
        gradient=residual_fn,
        jacobian=jacobian_fn,
        tol=1e-8,
        max_iter=20,
        krylov_solver=conjugate_gradient_while,
        krylov_tol=1e-8,
        krylov_max_iter=20,
    )

    DE, b, eps = final_state
    sig = compute_stress(eps)

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


tangent_operator_and_state = jax.jit(
    jax.jacfwd(local_constitutive_update, argnums=0, has_aux=True)
)

deps = jnp.array([1.2, 1.0, 1])
start_time = time.time()
tangent, state = tangent_operator_and_state(deps)
end_time = time.time()
print(f"Time taken: {end_time - start_time} seconds")

print(tangent)
