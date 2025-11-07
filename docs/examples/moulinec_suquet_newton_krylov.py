import jax

jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
jax.config.update("jax_enable_x64", True)  # use double-precision
jax.config.update("jax_platforms", "cpu")

import jax.numpy as jnp
from jax import Array

import numpy as np

from xpektra import (
    SpectralSpace,
    TensorOperator,
    make_field,
)
from xpektra.projection_operator import ProjectionOperator
from xpektra.solvers.nonlinear import (  # noqa: E402
    conjugate_gradient_while,
    newton_krylov_solver,
)


import equinox as eqx

import matplotlib.pyplot as plt


class MoulinecSuquetProjection(ProjectionOperator):
    """
    A 'final' class implementing the Moulinec-Suquet (MS) Green's operator,
    which depends on a homogeneous isotropic reference material (C0).
    """

    space: SpectralSpace
    lambda0: Array
    mu0: Array

    def compute_operator(self) -> Array:
        """
        Implements the vectorized formula for the MS Green's operator (Ghat = Γ^0).
        This replaces the slow, nested loops from the original script.
        """
        # Get grid properties from the scheme's space
        ndim = self.space.dim

        # Get the frequency grid 'q' (which is ξ), shape (N, N, N, 3)
        freq_vec = self.space.frequency_vector()
        meshes = np.meshgrid(*([freq_vec] * ndim), indexing="ij")
        q = jnp.stack(meshes, axis=-1)

        # Pre-calculate scalar terms, ensuring no division by zero
        q_dot_q = jnp.sum(q * q, axis=-1, keepdims=True)
        # Use jnp.where to safely handle the zero-frequency mode
        q_dot_q_safe = jnp.where(q_dot_q == 0, 1.0, q_dot_q)

        # Calculate the first term (T1) of the Green's operator
        # T1_num = (δ_ki ξ_h ξ_j + δ_hi ξ_k ξ_j + δ_kj ξ_h ξ_i + δ_hj ξ_k ξ_i)
        i = jnp.eye(ndim)
        t1_A = jnp.einsum("ki,...h,...j->...khij", i, q, q)
        t1_B = jnp.einsum("hi,...k,...j->...khij", i, q, q)
        t1_C = jnp.einsum("kj,...h,...i->...khij", i, q, q)
        t1_D = jnp.einsum("hj,...k,...i->...khij", i, q, q)

        T1_num = t1_A + t1_B + t1_C + t1_D
        T1 = T1_num / (4.0 * self.mu0 * q_dot_q_safe[..., None, None, None])

        # Calculate the second term (T2) of the Green's operator
        # T2 = const * (ξ_k ξ_h ξ_i ξ_j) / |ξ|^4
        const = (self.lambda0 + self.mu0) / (self.mu0 * (self.lambda0 + 2.0 * self.mu0))
        q4 = jnp.einsum("...k,...h,...i,...j->...khij", q, q, q, q)
        T2 = const * q4 / (q_dot_q_safe**2)[..., None, None, None]

        # Combine and set the zero-frequency mode to zero
        Ghat = T1 - T2
        Ghat = jnp.where(q_dot_q[..., None, None, None] == 0, 0.0, Ghat)

        return Ghat


N = 251
ndim = 2
length = 1

tensor = TensorOperator(dim=ndim)
space = SpectralSpace(size=N, dim=ndim, length=length)


# Create phase indicator (cylinder)
x = np.linspace(-0.5, 0.5, N)

if ndim == 3:
    Y, X, Z = np.meshgrid(x, x, x, indexing="ij")  # (N, N, N) grid
    phase = jnp.where(X**2 + Z**2 <= (0.2 / np.pi), 1.0, 0.0)  # 20% vol frac
else:
    X, Y = np.meshgrid(x, x, indexing="ij")  # (N, N) grid
    phase = jnp.where(X**2 + Y**2 <= (0.2 / np.pi), 1.0, 0.0)


# Material parameters [grids of scalars, shape (N,N,N)]
lambda1, lambda2 = 10.0, 1000.0
mu1, mu2 = 0.25, 2.5
lambdas = lambda1 * (1.0 - phase) + lambda2 * phase
mu = mu1 * (1.0 - phase) + mu2 * phase


dofs_shape = make_field(dim=ndim, N=N, rank=2).shape


@eqx.filter_jit
def _strain_energy(eps: Array, lambdas: Array, mu: Array) -> Array:
    eps_sym = 0.5 * (eps + tensor.trans(eps))
    energy = 0.5 * jnp.multiply(lambdas, tensor.trace(eps_sym) ** 2) + jnp.multiply(
        mu, tensor.trace(tensor.dot(eps_sym, eps_sym))
    )
    return energy.sum()


# Use average properties for the reference material
lambda0 = (lambda1 + lambda2) / 2.0
mu0 = (mu1 + mu2) / 2.0

material_energy = eqx.Partial(_strain_energy, lambdas=lambdas, mu=mu)
reference_energy = eqx.Partial(_strain_energy, lambdas=lambda0, mu=mu0)

compute_stress = jax.jacrev(material_energy)
compute_reference_stress = jax.jacrev(reference_energy)


i = jnp.eye(ndim)
I = make_field(dim=ndim, N=N, rank=2) + i  # Add i to broadcast

I4 = jnp.einsum("il,jk->ijkl", i, i)
I4rt = jnp.einsum("ik,jl->ijkl", i, i)
I4s = (I4 + I4rt) / 2.0
II = jnp.einsum("...ij,...kl->...ijkl", I, I)

# Build the constant C0 reference tensor [shape (3,3,3,3)]
C0 = lambda0 * II + 2.0 * mu0 * I4s

assert np.allclose(tensor.ddot(C0, I), compute_reference_stress(I)), "Reference stress computation is incorrect"

Ghat = MoulinecSuquetProjection(
    space=space, lambda0=lambda0, mu0=mu0
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
    def __call__(self, eps_flat: Array, eps_macro: Array) -> Array:
        """
        This makes instances of this class behave like a function.
        It takes only the flattened vector of unknowns, as required by the solver.
        """
        eps = eps_flat.reshape(self.dofs_shape)
        sigma = compute_stress(eps) 
        sigma0 = compute_reference_stress(eps) 
        tau = sigma - sigma0
        eps_fluc = self.space.ifft(self.tensor_op.ddot(self.Ghat, self.space.fft(tau)))

        residual_field = eps - eps_macro + jnp.real(eps_fluc)

        return residual_field.reshape(-1)


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

        dsigma = compute_stress(deps)
        dsigma0 = compute_reference_stress(deps) 
        dtau = dsigma - dsigma0
        jvp_field = self.space.ifft(
            self.tensor_op.ddot(self.Ghat, self.space.fft(dtau))
        )
        jvp_field = jnp.real(jvp_field) + deps
        return jvp_field.reshape(-1)


applied_strains = jnp.linspace(0, 1e-2, num=5)

eps = make_field(dim=2, N=N, rank=2)
deps = make_field(dim=2, N=N, rank=2)
eps_macro = make_field(dim=2, N=N, rank=2)

residual_fn = Residual(Ghat=Ghat, space=space, tensor_op=tensor, dofs_shape=eps.shape)
jacobian_fn = Jacobian(Ghat=Ghat, space=space, tensor_op=tensor, dofs_shape=eps.shape)


for inc, eps_avg in enumerate(applied_strains):
    # solving for elasticity
    eps_macro[:, :, 0, 0] = eps_avg
    eps_macro[:, :, 1, 1] = eps_avg

    residual_partial = eqx.Partial(residual_fn, eps_macro=eps_macro)

    b = -residual_partial(eps)
    # eps = eps + deps

    final_state = newton_krylov_solver(
        state=(deps, b, eps),
        gradient=residual_partial,
        jacobian=jacobian_fn,
        tol=1e-8,
        max_iter=20,
        krylov_solver=conjugate_gradient_while,
        krylov_tol=1e-8,
        krylov_max_iter=20,
    )
    eps = final_state[2]

sig = compute_stress(final_state[2])

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(4, 3))
ax1.imshow(sig.at[:, :, 0, 0].get(), cmap="managua_r")


ax2.plot(sig.at[:, :, 0, 0].get()[:, int(N / 2)])


ax_twin = ax2.twinx()
ax_twin.plot(phase[int(N / 2), :], color="gray")
plt.show()
