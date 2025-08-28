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

from spectralsolvers.operators import spatial, tensor, fourier_galerkin
from spectralsolvers.fft.transform import _fft, _ifft
from spectralsolvers.solvers.linear import conjugate_gradient_while
from spectralsolvers.solvers.nonlinear import newton_krylov_solver

from skimage.morphology import disk, rectangle, ellipse
import timeit

import argparse

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


@partial(jax.jit, static_argnames=["crack", "solid"])
def param(X, crack, solid):
    return crack * jnp.ones_like(X) * (X) + solid * jnp.ones_like(X) * (1 - X)


def run(args):
    N = args.N
    ndim = args.ndim
    length = args.length

    grid_size = (N,) * ndim
    elasticity_dof_shape = (ndim, ndim) + grid_size

    if ndim == 3:
        structure = create_structure_3d(N)
    elif ndim == 2:
        structure = create_structure(N)
    else:
        raise RuntimeError(f"Operations are not defined for dim {ndim}")

    structure = jax.device_put(structure)


    fft = jax.jit(partial(_fft, N=N, ndim=ndim))
    ifft = jax.jit(partial(_ifft, N=N, ndim=ndim))

    start_time = timeit.default_timer()

    Ghat = fourier_galerkin.compute_projection_operator_legacy(
        grid_size=grid_size, operator=spatial.Operator.rotated_difference, length=length
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


    # identity tensors (grid)
    if ndim == 2:
        I2 = jnp.einsum(
            "ij,xy",
            jnp.eye(ndim),
            jnp.ones(
                [
                    N,
                ]
                * ndim
            ),
        )
    elif ndim == 3:
        I2 = jnp.einsum(
            "ij,xyz",
            jnp.eye(ndim),
            jnp.ones(
                [
                    N,
                ]
                * ndim
            ),
        )
    else:
        raise RuntimeError(f"Operations are not defined for dim {ndim}")




    @jax.jit
    def strain_energy(eps, args=None):
        eps_sym = 0.5 * (eps + tensor.trans2(eps))
        energy = 0.5 * jnp.multiply(λ0, tensor.trace2(eps_sym) ** 2) + jnp.multiply(
            μ0, tensor.trace2(tensor.dot22(eps_sym, eps_sym))
        )
        return energy.sum()


    sigma = jax.jit(jax.jacrev(strain_energy, argnums=0))


    # functions for the projection 'G', and the product 'G : K : eps'
    @jax.jit
    def G(A2):
        return jnp.real(ifft(tensor.ddot42(Ghat, fft(A2)))).reshape(-1)


    @jax.jit
    def G_K_deps(depsm, args=None):
        depsm = depsm.reshape(elasticity_dof_shape)
        return G(sigma(depsm, args))



    applied_strains = np.diff(np.linspace(0, 2e-1, num=20))
    eps = jnp.zeros(elasticity_dof_shape)
    deps = jnp.zeros(elasticity_dof_shape)

    eps = jax.device_put(eps)
    deps = jax.device_put(deps)


    for_start_time = timeit.default_timer()

    for inc, deps_avg in enumerate(applied_strains):

        # solving for elasticity
        deps = deps.at[0, 0].set(deps_avg)

        b = -G_K_deps(deps, None)
        eps = jax.lax.add(eps, deps)

        start_time = timeit.default_timer()


        final_state = newton_krylov_solver(
            state=(deps, b, eps),
            A=G_K_deps,
            tol=1e-8,
            max_iter=20,
            krylov_solver=conjugate_gradient_while,
            krylov_tol=1e-8,
            krylov_max_iter=20,
            additionals=None,
        )
        print("execution time :", timeit.default_timer() - start_time)


        eps = final_state[2]

    print("execution time for :", timeit.default_timer() - for_start_time)
    


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="benchmark simulations")
    
    parser.add_argument("--N", type=int, default=499, help="pixel size")
    parser.add_argument ("--ndim", type=int, default=2, help="dimension of rve")
    parser.add_argument ("--length", type=float, default=1, help="length of rve")
 
    args = parser.parse_args()

    run(args)