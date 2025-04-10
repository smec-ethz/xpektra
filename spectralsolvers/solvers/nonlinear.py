import jax
import jax.numpy as jnp
import numpy as np
import functools


@functools.partial(jax.jit, static_argnames=["A", "krylov_solver", "tol"])
def newton_krylov_solver(state, A, krylov_solver, tol):

    def newton_raphson(state, n):
        dF, b, F = state
        error = jnp.linalg.norm(b)
        # jnp.linalg.norm(dF) / jnp.linalg.norm(F)
        jax.debug.print("residual={}", error)

        def true_fun(state):
            dF, b, F = state

            dF, iiter = krylov_solver(A=A, b=b, atol=1e-8)  # solve linear system

            dF = dF.reshape(F.shape)
            F = jax.lax.add(F, dF)
            b = -A(F)  # compute residual

            return (dF, b, F)

        def false_fun(state):
            return state

        return jax.lax.cond(error > tol, true_fun, false_fun, state), n

    final_state, xs = jax.lax.scan(newton_raphson, init=state, xs=jnp.arange(0, 20))
    return final_state
