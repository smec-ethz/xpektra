import jax

jax.config.update("jax_enable_x64", True)  # use double-precision
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)


import jax.numpy as jnp
import numpy as np
import functools
import equinox as eqx

"""
@functools.partial(
    jax.jit,
    static_argnames=[
        "A",
        "krylov_solver",
        "tol",
        "max_iter",
        "krylov_tol",
        "krylov_max_iter",
    ],
)"""
@eqx.filter_jit
def newton_krylov_solver(
    state, A, additionals, krylov_solver, tol, max_iter, krylov_tol, krylov_max_iter
):

    def newton_raphson(state, n):
        dF, b, F = state
        error = jnp.linalg.norm(b)

        def true_fun(state):
            dF, b, F = state

            dF, iiter = krylov_solver(
                A=A,
                b=b,
                atol=krylov_tol,
                max_iter=krylov_max_iter,
                additionals=additionals,
            )  # solve linear system
            
            dF = dF.reshape(F.shape)
            F = jax.lax.add(F, dF)
            b = -A(F, additionals)  # compute residual

            return (dF, b, F)

        def false_fun(state):
            return state

        return jax.lax.cond(error > tol, true_fun, false_fun, state), n

    final_state, xs = jax.lax.scan(
        newton_raphson, init=state, xs=jnp.arange(0, max_iter),
    )

    def not_converged(residual):
        jax.debug.print("Didnot converge, Residual value : {}", residual)
        return residual

    def converged(residual):
        jax.debug.print("Converged, Residual value : {}", residual)
        return residual

    residual = jnp.linalg.norm(final_state[1])
    _ = jax.lax.cond(residual > tol, not_converged, converged, residual)

    return final_state
