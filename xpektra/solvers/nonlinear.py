import jax
import jax.numpy as jnp
from jax import Array
import equinox as eqx
from typing import Callable


@eqx.filter_jit
def _cg_solver_impl(A: Callable, b: Array, atol=1e-8, max_iter=100) -> Array:
    """
    This is the internal implementation of the CG solve.
    It will be passed to `custom_linear_solve`.
    """
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

    b, p, r, rsold, x, iiter = jax.lax.while_loop(
        cond_fun, body_fun, (b, p, r, rsold, x, iiter)
    )
    # The solver only needs to return the solution vector 'x'
    return x


@eqx.filter_jit
def conjugate_gradient(
    A: Callable, b: Array, atol=1e-8, max_iter=100
) -> tuple[Array, int]:
    """
    Solves Ax = b using CG with an implicit gradient.

    A must be a symmetric positive-definite linear operator (as a JIT-able callable).
    """

    # We define the solver function to be passed.
    # It must have the signature solve(matvec, b)
    def solver(matvec_fn, b_vec):
        return _cg_solver_impl(matvec_fn, b_vec, atol, max_iter)

    # For CG, the operator A must be symmetric, so A = Aᵀ.
    # This means the solve_fun and transpose_solve_fun are identical.
    x = jax.lax.custom_linear_solve(
        matvec=A,
        b=b,
        solve=solver,
        transpose_solve=solver,  # Assumes A is symmetric
        symmetric=True,
    )

    # We return a dummy 0 for the iteration count to match your old API
    return x, 0


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
def conjugate_gradient_scan(A, b, atol, max_iter):
    x = jnp.full_like(b, fill_value=0.0)

    r = b - A(x)
    p = r
    rsold = jnp.vdot(r, r)

    state = (b, p, r, rsold, x)

    def conjugate_gradient(state, n):
        b, p, r, rsold, x = state

        def true_fun(state):
            b, p, r, rsold, x = state
            Ap = A(p)
            alpha = rsold / jnp.vdot(p, Ap)
            x = x + jnp.dot(alpha, p)
            r = r - jnp.dot(alpha, Ap)
            rsnew = jnp.vdot(r, r)
            p = r + (rsnew / rsold) * p
            rsold = rsnew
            return (b, p, r, rsold, x)

        def false_fun(state):
            return state

        return (
            jax.lax.cond(
                jnp.sqrt(rsold) > atol,
                true_fun,
                false_fun,
                state,
            ),
            n,
        )

    final_state, xs = jax.lax.scan(
        conjugate_gradient, init=state, xs=jnp.arange(0, max_iter)
    )

    return final_state[-1], None


@eqx.filter_jit
def preconditioned_conjugate_gradient(A, b, M_inv, atol=1e-8, max_iter=100):
    iiter = 0
    x = jnp.full_like(b, fill_value=0.0)

    r = b - A(x)
    z = M_inv(r)  # Apply preconditioner
    p = z
    rsold = jnp.vdot(r, z)  # z^T * r

    def body_fun(state):
        b, p, r, rsold, x, z, iiter = state
        Ap = A(p)
        alpha = rsold / jnp.vdot(p, Ap)
        x = x + alpha * p
        r = r - alpha * Ap
        z = M_inv(r)  # Apply preconditioner
        rsnew = jnp.vdot(r, z)  # z^T * r

        p = z + (rsnew / rsold) * p
        rsold = rsnew
        iiter = iiter + 1
        return (b, p, r, rsold, x, z, iiter)

    def cond_fun(state):
        b, p, r, rsold, x, z, iiter = state
        return jnp.logical_and(jnp.sqrt(rsold) > atol, iiter < max_iter)

    b, p, r, rsold, x, z, iiter = jax.lax.while_loop(
        cond_fun, body_fun, (b, p, r, rsold, x, z, iiter)
    )
    jax.debug.print("CG error = {:.14f}", rsold)

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

            # jacobian_partial = eqx.Partial(jacobian, F_flat=F.reshape(-1))

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
