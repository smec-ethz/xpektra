from typing import Callable

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import Array


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
    # jax.debug.print("CG error = {:.14f}", rsold)

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

    # We return a dummy 0 for the iteration count
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

    return final_state[-1]  # , None


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
    x: Array,
    b: Array,
    gradient: Callable,
    jacobian: Callable,
    krylov_solver: Callable,
    tol: float,
    max_iter: int,
    krylov_tol: float,
    krylov_max_iter: int,
):
    def newton_raphson(state, n):
        x, b = state
        error = jnp.linalg.norm(b)

        def true_fun(state):
            x, b = state
            dF, iiter = krylov_solver(
                A=jacobian,
                b=b,
                atol=krylov_tol,
                max_iter=krylov_max_iter,
            )  # solve linear system

            dF = dF.reshape(x.shape)
            x = jax.lax.add(x, dF)
            b = -gradient(x)  # compute residual

            return (x, b)

        def false_fun(state):
            return state

        return jax.lax.cond(error > tol, true_fun, false_fun, state), n

    final_state, xs = jax.lax.scan(
        newton_raphson,
        init=(x, b),
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

    return final_state[0]


def _solve_linear_system(J, b):
    dx = jnp.linalg.solve(J, b)
    return dx


@eqx.filter_jit
def implicit_newton_solver(
    x: Array,
    b: Array,
    gradient: Callable,
    jacobian: Callable,
    krylov_solver: Callable,
    tol: float,
    max_iter: int,
    krylov_tol: float,
    krylov_max_iter: int,
):
    partial_newton_krylov_solver = eqx.Partial(
        newton_krylov_solver,
        b=b,
        jacobian=jacobian,
        tol=tol,
        max_iter=max_iter,
        krylov_solver=krylov_solver,
        krylov_tol=krylov_tol,
        krylov_max_iter=krylov_max_iter,
    )

    def solve(f, x):
        return partial_newton_krylov_solver(x=x, gradient=f)

    # def tangent_solve(g, y):
    #    return _solve_linear_system(jax.jacfwd(g)(y), y)
    def tangent_solve(g, y):
        """
        Solve J u = y for u using only JVP/VJP of g (matrix-free).
        We form the normal-system operator: v -> J^T (J v)
        and solve (J^T J) u = J^T y with conjugate gradient.
        """

        # Precompute a pullback closure at y for efficient repeated J^T applications.
        # vjp_fun will accept a vector 'w' and return tuple (cotangent,)
        _, vjp_fun = jax.vjp(g, y)

        # matvec for normal eq: v -> J^T (J v)
        def normal_matvec(v):
            # compute J v using jax.jvp
            jv = jax.jvp(g, (y,), (v,))[1]
            # apply pullback to get J^T jv; vjp_fun returns a tuple of cotangents
            jt_jv = vjp_fun(jv)[0]
            return jt_jv

        # right-hand side: J^T y
        # but 'y' here is the tangent right-hand side (same shape as output of g)
        jT_y = vjp_fun(y)[0]

        # solve (J^T J) u = J^T y
        u = _cg_solver_impl(
            normal_matvec, jT_y, atol=krylov_tol, max_iter=krylov_max_iter
        )

        return u

    x_sol = jax.lax.custom_root(gradient, x, solve, tangent_solve, has_aux=False)

    return x_sol
