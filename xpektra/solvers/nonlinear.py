import jax
import jax.numpy as jnp
import equinox as eqx
from typing import Callable


@eqx.filter_jit
def conjugate_gradient_while(A, b, atol=1e-8, max_iter=100):
    iiter = 0

    def body_fun(state):
        b, p, r, rsold, x, iiter = state
        Ap = A(p)
        alpha = rsold / jnp.vdot(p, Ap)
        x = x + jnp.dot(alpha, p)
        r = r - jnp.dot(alpha, Ap)
        rsnew = jnp.vdot(r, r)
        p = r + (rsnew / rsold) * p
        rsold = rsnew
        iiter = iiter + 1
        return (b, p, r, rsold, x, iiter)

    def cond_fun(state):
        b, p, r, rsold, x, iiter = state
        return jnp.logical_and(jnp.sqrt(rsold) > atol, iiter < max_iter)

    x = jnp.full_like(b, fill_value=0.0)
    r = b - A(x)
    p = r
    rsold = jnp.vdot(r, r)
    jax.debug.print("CG error = {:.14f}", rsold)

    b, p, r, rsold, x, iiter = jax.lax.while_loop(
        cond_fun, body_fun, (b, p, r, rsold, x, iiter)
    )
    return x, iiter


@eqx.filter_jit
def newton_krylov_solver(
    state: tuple,
    gradient: Callable,
    jacobian: Callable,
    krylov_solver: Callable,
    tol: float,
    max_iter: int,
    krylov_tol: float,
    krylov_max_iter: int,
):
    def newton_raphson(state, n):
        dF, b, F = state
        error = jnp.linalg.norm(b)

        def true_fun(state):
            dF, b, F = state

            #jacobian_partial = eqx.Partial(jacobian, F_flat=F.reshape(-1))

            dF, iiter = krylov_solver(
                A=jacobian,
                b=b,
                atol=krylov_tol,
                max_iter=krylov_max_iter,
            )  # solve linear system

            dF = dF.reshape(F.shape)
            F = jax.lax.add(F, dF)
            b = -gradient(F)  # compute residual

            return (dF, b, F)

        def false_fun(state):
            return state

        return jax.lax.cond(error > tol, true_fun, false_fun, state), n

    final_state, xs = jax.lax.scan(
        newton_raphson,
        init=state,
        xs=jnp.arange(0, max_iter),
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