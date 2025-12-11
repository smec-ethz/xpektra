# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.18.1
#   kernelspec:
#     display_name: venv
#     language: python
#     name: python3
# ---

# %%
import jax

jax.config.update("jax_enable_x64", True)  # use double-precision
jax.config.update("jax_platforms", "cpu")
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
import jax.numpy as jnp
import numpy as np

# %%
from functools import partial

import matplotlib.pyplot as plt
from skimage.morphology import disk, ellipse, rectangle
from spectralsolver import (
    DifferentialMode,
    SpectralSpace,
    TensorOperator,
    make_field,
)
from spectralsolver.operators import fourier_galerkin
from typing import Callable

# %% [markdown]
# In this notebook, we implement the small-strain `J2` plasticity. We use the automatic differentiation to compute the alogriothmic tangent stiffness matrix (which ia function of bbothe elastic strain and plastic strain and yield function). It is necessary for `Newton-Raphson` iteration. 

# %% [markdown]
# ## constructing an RVE

# %%
import random

random.seed(1)


def place_circle(matrix, n, r, x_center, y_center):
    for i in range(n):
        for j in range(n):
            if (i - x_center) ** 2 + (j - y_center) ** 2 <= r**2:
                matrix[i][j] = 1


def generate_matrix_with_circles(n, x, r):
    if r >= n:
        raise ValueError("Radius r must be less than the size of the matrix n")

    matrix = np.zeros((n, n), dtype=int)
    placed_circles = 0

    while placed_circles < x:
        x_center = random.randint(0, n - 1)
        y_center = random.randint(0, n - 1)

        # Check if the circle fits within the matrix bounds
        if (
            x_center + r < n
            and y_center + r < n
            and x_center - r >= 0
            and y_center - r >= 0
        ):
            previous_matrix = matrix.copy()
            place_circle(matrix, n, r, x_center, y_center)
            if not np.array_equal(previous_matrix, matrix):
                placed_circles += 1

    return matrix


# Example usage
N = 199
shape = (N, N)
length = 1.0
ndim = 2


x = 10
r = 20
structure = generate_matrix_with_circles(N, x, r)

# %%
grid_size = (N,) * ndim
elasticity_dof_shape = (ndim, ndim) + grid_size


# %% [markdown]
# ## assigning material parameters 
# We assign material parameters to the two phases. The two phases within the RVE are denoted as
# - Soft = 0
# - Hard = 1

# %%
# material parameters + function to convert to grid of scalars
def param(X, soft, hard):
    return hard * jnp.ones_like(X) * (X) + soft * jnp.ones_like(X) * (1 - X)


# %% [markdown]
# We consider a `linear isotropic hardening law` for both the phases

# %%
# material parameters
phase_constrast = 2

K = param(structure, soft=0.833, hard=phase_constrast * 0.833)  # bulk      modulus
μ = param(structure, soft=0.386, hard=phase_constrast * 0.386)  # shear     modulus
H = param(
    structure, soft=2000.0e6 / 200.0e9, hard=phase_constrast * 2000.0e6 / 200.0e9
)  # hardening modulus
sigma_y = param(
    structure, soft=600.0e6 / 200.0e9, hard=phase_constrast * 600.0e6 / 200.0e9
)  # initial yield stress

n = 1.0

# %% [markdown]
# ## plasticity basics
#
# Now we define the basics of plasticity implementation:
#
# - yield surface
#
# $$
# \Phi(\sigma_{ij}, \varepsilon^p_{ij}) = \underbrace{\sqrt{\dfrac{3}{2}\sigma^{dev}_{ij}\sigma^{dev}_{jk}}}_{\sigma^{eq}} - (\sigma_{0} + H\varepsilon^{p})
# $$
#
# - return mappping algorithm
#
# $$
# \Delta \varepsilon =  \dfrac{\langle \Phi(\sigma_{ij}, \varepsilon_{p}) \rangle_{+}}{3\mu + H}
# $$
#
# - tangent stiffness operator
#   
# $$
# \mathbb{C} = \dfrac{\partial \sigma^{t+1}}{\partial \varepsilon^{t+1}} 
# $$

# %% [markdown]
# We also define certain Identity tensor for each grid point.
#
# - $\mathbf{I}$ = 2 order Identity tensor with shape `(2, 2, N, N)` 
# - $\mathbb{I4}$ = 4 order Identity tensor with shape `(2, 2, 2, 2, N, N)`
#

# %%
tensor = TensorOperator(dim=ndim)
space = SpectralSpace(size=N, dim=ndim, length=length)

# %%
# identity tensor (single tensor)
i = jnp.eye(ndim)

# identity tensors (grid)
I = jnp.einsum(
    "ij,xy",
    i,
    jnp.ones(
        [
            N,
        ]
        * ndim
    ),
)  # 2nd order Identity tensor
I4 = jnp.einsum(
    "ijkl,xy->ijklxy",
    jnp.einsum("il,jk", i, i),
    jnp.ones(
        [
            N,
        ]
        * ndim
    ),
)  # 4th order Identity tensor
I4rt = jnp.einsum(
    "ijkl,xy->ijklxy",
    jnp.einsum("ik,jl", i, i),
    jnp.ones(
        [
            N,
        ]
        * ndim
    ),
)
I4s = (I4 + I4rt) / 2.0

II = tensor.dyad(I, I)
I4d = I4s - II / 3.0

Ghat = fourier_galerkin.compute_projection_operator(
    space=space, diff_mode=DifferentialMode.rotated_difference
)

# %%
import equinox as eqx


# %%
@jax.jit
def yield_function(ep: jnp.ndarray):
    return sigma_y + H * ep**n


@jax.jit
def compute_stress(eps: jnp.ndarray, args: tuple):
    eps_t, epse_t, ep_t = args

    # elastic stiffness tensor
    C4e = K * II + 2.0 * μ * I4d

    # trial state
    epse_s = epse_t + (eps - eps_t)
    sig_s = tensor.ddot(C4e, epse_s)
    sigm_s = tensor.ddot(sig_s, I) / 3.0
    sigd_s = sig_s - sigm_s * I
    sigeq_s = jnp.sqrt(3.0 / 2.0 * tensor.ddot(sigd_s, sigd_s))

    # avoid zero division below ("phi_s" is corrected below)
    Z = jnp.where(sigeq_s == 0, True, False)
    sigeq_s = jnp.where(Z, 1, sigeq_s)

    # evaluate yield surface, set to zero if elastic (or stress-free)
    sigy = yield_function(ep_t)
    phi_s = sigeq_s - sigy
    phi_s = 1.0 / 2.0 * (phi_s + jnp.abs(phi_s))
    phi_s = jnp.where(Z, 0.0, phi_s)
    elastic_pt = jnp.where(phi_s <= 0, True, False)

    # plastic multiplier, based on non-linear hardening
    # - initialize
    dep = phi_s / (3 * μ + H)

    # return map algorithm
    N = 3.0 / 2.0 * sigd_s / sigeq_s
    ep = ep_t + dep
    sig = sig_s - dep * N * 2.0 * μ
    epse = epse_s - dep * N

    return sig, epse, ep


@eqx.filter_jit
def compute_residual(sigma: jnp.ndarray) -> jnp.ndarray:
    return jnp.real(space.ifft(tensor.ddot(Ghat, space.fft(sigma)))).reshape(-1)


@eqx.filter_jit
def compute_tangents(deps: jnp.ndarray, args: tuple):
    deps = deps.reshape(ndim, ndim, N, N)
    eps, eps_t, epse_t, ep_t = args
    primal, tangents = jax.jvp(
        partial(compute_stress, args=(eps_t, epse_t, ep_t)), (eps,), (deps,)
    )
    return compute_residual(tangents[0])


# partial_compute_tangent = partial(compute_tangents, sigma=sigma)

# %%
from spectralsolver.solvers.nonlinear import (
    conjugate_gradient_while,
    newton_krylov_solver,
)


# %%
@jax.jit
def newton_solver(state, n):
    deps, b, eps, eps_t, epse_t, ep_t, En, sig = state

    error = jnp.linalg.norm(deps) / En
    jax.debug.print("residual={}", jnp.linalg.norm(deps) / En)

    def true_fun(state):
        deps, b, eps, eps_t, epse_t, ep_t, En, sig = state

        partial_compute_tangent = jax.jit(
            partial(compute_tangents, args=(eps, eps_t, epse_t, ep_t))
        )

        deps, iiter = conjugate_gradient_while(
            atol=1e-8,
            A=partial_compute_tangent,
            b=b,
        )  # solve linear system using CG

        deps = deps.reshape(eps.shape)
        eps = jax.lax.add(eps, deps)  # update DOFs (array -> tensor.grid)
        sig, epse, ep = compute_stress(eps, (eps_t, epse_t, ep_t))
        b = -compute_residual(sig)  # compute residual

        jax.debug.print("CG iteration {}", iiter)

        return (deps, b, eps, eps_t, epse, ep, En, sig)

    def false_fun(state):
        return state

    return jax.lax.cond(error > 1e-8, true_fun, false_fun, state), n


# %%
# initialize: stress and strain tensor, and history
sig = make_field(dim=ndim, N=N, rank=2)
eps = make_field(dim=ndim, N=N, rank=2)
eps_t = make_field(dim=ndim, N=N, rank=2)
epse_t = make_field(dim=ndim, N=N, rank=2)
ep_t = make_field(dim=ndim, N=N, rank=2)
deps = make_field(dim=ndim, N=N, rank=2)

# define incremental macroscopic strain
ninc = 100
epsbar = 0.12
deps[0, 0] = jnp.sqrt(3.0) / 2.0 * epsbar / float(ninc)
deps[1, 1] = -jnp.sqrt(3.0) / 2.0 * epsbar / float(ninc)

b = -compute_tangents(deps, (eps, eps_t, epse_t, ep_t))
eps = jax.lax.add(eps, deps)
En = jnp.linalg.norm(eps)

# %%
state = (deps, b, eps, eps_t, epse_t, ep_t, En, sig)
final_state, xs = jax.lax.scan(newton_solver, init=state, xs=jnp.arange(0, 10))
