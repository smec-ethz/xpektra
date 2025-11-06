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
from xpektra.scheme import RotatedDifference, Fourier, Scheme
from xpektra.projection_operator import ProjectionOperator

import equinox as eqx
from functools import partial

import time


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
        N = self.space.size

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



N = 15
ndim = 3
length = 1

# Create phase indicator (cylinder)
x = np.linspace(-0.5, 0.5, N)
Y, X, Z = np.meshgrid(x, x, x, indexing="ij")  # (N, N, N) grid
phase = jnp.where(X**2 + Z**2 <= (0.2 / np.pi), 1.0, 0.0)  # 20% vol frac

# Material parameters [grids of scalars, shape (N,N,N)]
lambda1, lambda2 = 10.0, 100.0
mu1, mu2 = 0.25, 2.5
lambdas = lambda1 * (1.0 - phase) + lambda2 * phase
mu = mu1 * (1.0 - phase) + mu2 * phase


tensor = TensorOperator(dim=ndim)
space = SpectralSpace(size=N, dim=ndim, length=length)

i = jnp.eye(ndim)
I = make_field(dim=ndim, N=N, rank=2) + i  # Add i to broadcast

I4 = jnp.einsum("il,jk->ijkl", i, i)
I4rt = jnp.einsum("ik,jl->ijkl", i, i)
I4s = (I4 + I4rt) / 2.0
II = jnp.einsum("...ij,...kl->...ijkl", I, I)

# Broadcast scalars to the 4th-order tensor shape
C4 = (
    lambdas[..., None, None, None, None] * II
    + (2.0 * mu[..., None, None, None, None]) * I4s
)

# Use average properties for the reference material
lambda0 = (lambda1 + lambda2) / 2.0
mu0 = (mu1 + mu2) / 2.0

# Build the constant C0 reference tensor [shape (3,3,3,3)]
C0 = lambda0 * II + 2.0 * mu0 * I4s


Ghat = MoulinecSuquetProjection(space=space, lambda0=lambda0, mu0=mu0).compute_operator()


# --- fixed-point iteration ---

@partial(jax.jit, static_argnames=['max_iter', 'tol'])
def solve_ms_fft(E_macro: Array, eps_guess: Array, max_iter: int, tol: float) -> Array:
    """Solves the Lippmann-Schwinger equation via fixed-point iteration."""
    
    eps = eps_guess
    
    def cond_fun(state):
        eps_k, eps_prev, k = state
        err = jnp.linalg.norm(eps_k - eps_prev) / jnp.linalg.norm(E_macro)
        return jnp.logical_and(err > tol, k < max_iter)

    def body_fun(state):
        eps_k, _, k = state
        
        # Calculate stress and polarization
        sigma = tensor.ddot(C4, eps_k)
        sigma0 = tensor.ddot(C0, eps_k)
        tau = sigma - sigma0 # Polarization field tau = σ - C0:ε
        
        # Apply Green's operator: ε_fluc = G^0 * tau
        tau_hat = space.fft(tau)
        eps_fluc_hat = tensor.ddot(Ghat, tau_hat) #project(Ghat, tau_hat)
        eps_fluc = jnp.real(space.ifft(eps_fluc_hat))
        
        # Update total strain: ε_new = E_macro - ε_fluc
        eps_new = E_macro - eps_fluc
        
        return (eps_new, eps_k, k + 1)

    # Run the while loop
    (eps_final, _, num_iters) = jax.lax.while_loop(
        cond_fun, 
        body_fun, 
        (eps, eps, 0)
    )
    
    # jax.debug.print("Converged in {i} iterations", i=num_iters)
    return eps_final

# --- solve for 6 load cases & homogenize ---

# Create 6 macroscopic strain load cases
E_list = [
    jnp.array([[1, 0, 0], [0, 0, 0], [0, 0, 0]],), # E_xx
    jnp.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]],), # E_yy
    jnp.array([[0, 0, 0], [0, 0, 0], [0, 0, 1]],), # E_zz
    jnp.array([[0, 1, 0], [1, 0, 0], [0, 0, 0]],), # 2E_xy
    jnp.array([[0, 0, 0], [0, 0, 1], [0, 1, 0]],), # 2E_yz
    jnp.array([[0, 0, 1], [1, 0, 0], [0, 0, 0]],)  # 2E_xz
]

homogenized_stiffness = jnp.zeros((6, 6))
voigt_indices = [(0, 0), (1, 1), (2, 2), (0, 1), (1, 2), (0, 2)]

print("Starting homogenization...")
for i, E_voigt in enumerate(E_list):
    # Create the full E_macro field (broadcasts E_voigt)
    E_macro = make_field(dim=ndim, N=N, rank=2) + E_voigt
    
    # Solve the RVE problem
    eps_final = solve_ms_fft(E_macro, E_macro, max_iter=200, tol=1e-8)
    
    # Compute the final stress field
    sig_final = tensor.ddot(C4, eps_final)
    
    # Homogenize (average over the volume)
    avg_stress = jnp.mean(sig_final, axis=(0, 1, 2))
    
    # Store in Voigt notation
    for j, (row, col) in enumerate(voigt_indices):
        homogenized_stiffness = homogenized_stiffness.at[j, i].set(avg_stress[row, col])

print("Homogenized Stiffness (Voigt): \n", homogenized_stiffness)