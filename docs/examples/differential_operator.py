# %%
import jax

jax.config.update("jax_compilation_cache_dir", "./jax-cache")
jax.config.update("jax_enable_x64", True)  # use double-precision
jax.config.update("jax_platforms", "cpu")

import jax.numpy as jnp
import numpy as np
import functools

from jax import Array
import equinox as eqx

import matplotlib.pyplot as plt
from skimage.morphology import disk, rectangle, ellipse
import itertools
from matplotlib.gridspec import GridSpec

from xpektra import (
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


def param(X, soft, hard):
    return soft * jnp.ones_like(X) * (X) + hard * jnp.ones_like(X) * (1 - X)


def test_microstructure(N, operator, length):
    H, L = (N, N)
    r = int(H / 2)

    structure = np.zeros((H, L))
    structure[:r, -r:] += rectangle(r, r)
    structure = np.flipud(structure)
    structure = np.fliplr(structure)

    ndim = len(structure.shape)
    N = structure.shape[0]

    tensor = TensorOperator(dim=ndim)
    space = SpectralSpace(size=N, dim=ndim, length=length)

    # material parameters
    phase_contrast = 1000.0

    # lames constant
    lambda_modulus = {"soft": 1.0, "hard": phase_contrast}
    shear_modulus = {"soft": 1.0, "hard": phase_contrast}

    bulk_modulus = {}
    bulk_modulus["soft"] = lambda_modulus["soft"] + 2 * shear_modulus["soft"] / 3
    bulk_modulus["hard"] = lambda_modulus["hard"] + 2 * shear_modulus["hard"] / 3

    # material parameters
    μ0 = param(
        structure, soft=shear_modulus["soft"], hard=shear_modulus["hard"]
    )  # shear     modulus
    λ0 = param(
        structure, soft=lambda_modulus["soft"], hard=lambda_modulus["hard"]
    )  # shear     modulus

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
            eps_flat = eps_flat.reshape(-1)
            sigma = compute_stress(
                eps_flat
            )  # Assumes compute_stress is defined elsewhere
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

            deps_flat = deps_flat.reshape(-1)
            dsigma = compute_stress(deps_flat)
            jvp_field = self.space.ifft(
                self.tensor_op.ddot(
                    self.Ghat, self.space.fft(dsigma.reshape(self.dofs_shape))
                )
            )
            return jnp.real(jvp_field).reshape(-1)

    eps = make_field(dim=ndim, N=N, rank=2)
    residual_fn = Residual(
        Ghat=Ghat, space=space, tensor_op=tensor, dofs_shape=eps.shape
    )
    jacobian_fn = Jacobian(
        Ghat=Ghat, space=space, tensor_op=tensor, dofs_shape=eps.shape
    )


    deps = make_field(dim=ndim, N=N, rank=2)
    deps[:, :, 0, 1] = 5e-1
    deps[:, :, 1, 0] = 5e-1

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

    sig = compute_stress(eps.reshape(-1)).reshape(dofs_shape)

    return sig.at[:, :,0, 1].get(), structure


N = 699
length = 1

sig_xy, structure = test_microstructure(N=N, operator="rotated", length=length)

# %%
dx = length / N
N_inset = int(0.04 / dx)

plt.imshow(sig_xy, origin="lower", cmap="Spectral", vmax=2000)
plt.xlim(int(N / 2) - N_inset, int(N / 2) + N_inset)
plt.ylim(int(N / 2) - N_inset, int(N / 2) + N_inset)
plt.plot(
    [int(N / 2) - N_inset, int(N / 2)], [int(N / 2), int(N / 2)], color="k", zorder=20
)
plt.plot(
    [int(N / 2), int(N / 2)],
    [int(N / 2), int(N / 2) + N_inset],
    color="k",
    zorder=20,
)

plt.colorbar()

plt.show()


length = 1
dx = length / N
N_inset = int(0.04 / dx)

plt.imshow(sig_xy, origin="lower", cmap="Spectral", vmax=2000)
plt.xlim(int(N / 2) - N_inset, int(N / 2) + N_inset)
plt.ylim(int(N / 2) - N_inset, int(N / 2) + N_inset)
plt.plot(
    [int(N / 2) - N_inset, int(N / 2)], [int(N / 2), int(N / 2)], color="k", zorder=20
)
plt.plot(
    [int(N / 2), int(N / 2)],
    [int(N / 2), int(N / 2) + N_inset],
    color="k",
    zorder=20,
)

plt.colorbar()
plt.show()