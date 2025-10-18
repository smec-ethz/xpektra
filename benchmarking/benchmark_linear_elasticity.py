import os

import jax

jax.config.update("jax_compilation_cache_dir", os.environ["JAX_CACHE_DIR"])
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)  # use double-precision
if os.environ["JAX_PLATFORM"] == "cpu":
    jax.config.update("jax_platforms", "cpu")

print(jax.devices())


import numpy as np
from functools import partial

from spectralsolver import (
    DifferentialMode,
    SpectralSpace,
    TensorOperator,
    make_field,
)
from spectralsolver.green_functions import fourier_galerkin

from skimage.morphology import disk, rectangle, ellipse
import timeit

import argparse


from spectralsolver.solvers.nonlinear import (
    conjugate_gradient_while,
    newton_krylov_solver,
)

def create_structure_3d(N):
    Hmid = int(N / 2)
    Lmid = int(N / 2)
    Tmid = int(N / 2)

    r = int(N / 10)

    structure = np.zeros((N, N, N))
    structure[
        Hmid : Hmid + 1, Lmid - 2 * r : Lmid + 2 * r, Tmid - 2 * r : Tmid + 2 * r
    ] += np.ones((1, 4 * r, 4 * r))

    return structure


def create_structure(N):
    Hmid = int(N / 2)
    Lmid = int(N / 2)
    r = int(N / 10)

    structure = np.zeros((N, N))
    structure[Hmid : Hmid + 1, Lmid - 2 * r : Lmid + 2 * r] += rectangle(
        nrows=1, ncols=4 * r
    )

    return structure


def param(X, crack, solid):
    return crack * jnp.ones_like(X) * (X) + solid * jnp.ones_like(X) * (1 - X)


def run(args):
    N = args.N
    ndim = args.ndim
    length = args.length

    if ndim == 3:
        structure = create_structure_3d(N)
    elif ndim == 2:
        structure = create_structure(N)
    else:
        raise RuntimeError(f"Operations are not defined for dim {ndim}")

    structure = jax.device_put(structure)

    tensor = TensorOperator(dim=ndim)
    space = SpectralSpace(size=N, dim=ndim, length=length)


    
    start_time = timeit.default_timer()

    Ghat = fourier_galerkin.compute_projection_operator(
    space=space, diff_mode=DifferentialMode.rotated_difference
)
    Ghat = jax.device_put(Ghat)

    print('Ghat execution time : ', timeit.default_timer() - start_time )

    # material parameters
    elastic_modulus = {"solid": 1.0, "crack": 1e-3}  # N/mm2
    poisson_modulus = {"solid": 0.2, "crack": 0.2}

    # lames constant
    lambda_modulus = {}
    lambda_modulus["solid"] = (
        poisson_modulus["solid"]
        * elastic_modulus["solid"]
        / ((1 + poisson_modulus["solid"]) * (1 - 2 * poisson_modulus["solid"]))
    )
    lambda_modulus["crack"] = (
        poisson_modulus["crack"]
        * elastic_modulus["crack"]
        / ((1 + poisson_modulus["crack"]) * (1 - 2 * poisson_modulus["crack"]))
    )

    shear_modulus = {}
    shear_modulus["solid"] = elastic_modulus["solid"] / (2 * (1 + poisson_modulus["solid"]))
    shear_modulus["crack"] = elastic_modulus["crack"] / (2 * (1 + poisson_modulus["crack"]))

    bulk_modulus = {}
    bulk_modulus["solid"] = lambda_modulus["solid"] + 2 * shear_modulus["solid"] / 3
    bulk_modulus["crack"] = lambda_modulus["crack"] + 2 * shear_modulus["crack"] / 3


    λ0 = param(
        structure, crack=lambda_modulus["crack"], solid=lambda_modulus["solid"]
    )  # lame parameter
    μ0 = param(
        structure, crack=shear_modulus["crack"], solid=shear_modulus["solid"]
    )  # lame parameter
    K0 = param(structure, crack=bulk_modulus["crack"], solid=bulk_modulus["solid"])

    μ0 = jax.device_put(μ0)
    λ0 = jax.device_put(λ0)
    K0 = jax.device_put(K0)

    eps = make_field(dim=ndim, N=N, rank=2)

    @jax.jit
    def strain_energy(eps):
        eps_sym = 0.5 * (eps + tensor.trans(eps))
        energy = 0.5 * jnp.multiply(λ0, tensor.trace(eps_sym) ** 2) + jnp.multiply(
            μ0, tensor.trace(tensor.dot(eps_sym, eps_sym))
        )
        return energy.sum()


    compute_stress = jax.jacrev(strain_energy)

    @jax.jit
    def _tangent(deps, Ghat, dofs_shape):
        deps = deps.reshape(dofs_shape)
        dsigma = compute_stress(deps)
        return jnp.real(space.ifft(tensor.ddot(Ghat, space.fft(dsigma)))).reshape(-1)

    @jax.jit
    def _residual(deps, Ghat, dofs_shape):
        deps = deps.reshape(dofs_shape)
        sigma = compute_stress(deps)
        return jnp.real(space.ifft(tensor.ddot(Ghat, space.fft(sigma)))).reshape(-1)


    residual = jax.jit(partial(_residual, Ghat=Ghat, dofs_shape=eps.shape))
    jacobian = jax.jit(partial(_tangent, Ghat=Ghat, dofs_shape=eps.shape))


    applied_strains = jnp.diff(jnp.linspace(0, 2e-1, num=20))

    deps = make_field(dim=ndim, N=N, rank=2)

    for_start_time = timeit.default_timer()

    for inc, deps_avg in enumerate(applied_strains):
        # solving for elasticity
        deps[0, 0] = deps_avg
        b = -residual(deps)
        eps = eps + deps

        final_state = newton_krylov_solver(
            state=(deps, b, eps),
            gradient=residual,
            jacobian=jacobian,
            tol=1e-8,
            max_iter=20,
            krylov_solver=conjugate_gradient_while,
            krylov_tol=1e-8,
            krylov_max_iter=20,
        )
        eps = final_state[2]

    print("execution time for :", timeit.default_timer() - for_start_time)
    


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="benchmark simulations")
    
    parser.add_argument("--N", type=int, default=499, help="pixel size")
    parser.add_argument ("--ndim", type=int, default=2, help="dimension of rve")
    parser.add_argument ("--length", type=float, default=1, help="length of rve")
 
    args = parser.parse_args()

    run(args)