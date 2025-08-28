import jax

jax.config.update("jax_enable_x64", True)  # use double-precision
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
import os


import jax.numpy as jnp
import numpy as np
import functools
import equinox as eqx

#@functools.partial(jax.jit, static_argnums=(0,))
@eqx.filter_jit
def conjugate_gradient_while(A, b, additionals, atol=1e-8, max_iter=100):

    iiter = 0

    def body_fun(state):
        b, p, r, rsold, x, iiter = state
        Ap = A(p, additionals)
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
    r = b - A(x, additionals)
    p = r
    rsold = jnp.vdot(r, r)
    # jax.debug.print("CG error = {}", rsold)

    b, p, r, rsold, x, iiter = jax.lax.while_loop(
        cond_fun, body_fun, (b, p, r, rsold, x, iiter)
    )
    return x, iiter


#@functools.partial(jax.jit, static_argnames=["A", "max_iter"])
@eqx.filter_jit
def conjugate_gradient_scan(A, b, additionals, atol, max_iter):
    x = jnp.full_like(b, fill_value=0.0)

    r = b - A(x, additionals)
    p = r
    rsold = jnp.vdot(r, r)

    state = (b, p, r, rsold, x)

    def conjugate_gradient(state, n):
        b, p, r, rsold, x = state

        def true_fun(state):
            b, p, r, rsold, x = state
            Ap = A(p, additionals)
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

    final_state, xs = jax.lax.scan(conjugate_gradient, init=state, xs=jnp.arange(0, max_iter))

    return final_state[-1], None


@functools.partial(jax.jit, static_argnums=(0,))
def bound_conjugate_gradient(
    A, b, lower_bound, upper_bound, additional, x0=None, atol=1e-5
):

    b, lower_bound, upper_bound, x0, additionals = jax.device_put(
        (b, lower_bound, upper_bound, x0, additional)
    )
    small = 1e-8

    def true():
        return True

    def false():
        return False

    def body_fun(state):
        b, x1, g1, r_prev, p1, msk_bnd, changed, iiter = state

        iiter += 1
        # jax.debug.print('msk = {}', msk_bnd)

        # 3. construct a search direction that is zero in bound variables
        r1 = -g1
        r1 = jnp.where(msk_bnd, 0.0, r1)

        def update_direction():
            beta = jnp.vdot(r1, r1) / jnp.vdot(r_prev, r_prev)
            p2 = r1 + beta * p1
            return p2

        def donot_update_direction():
            return r_prev

        p2 = jax.lax.cond(
            jnp.logical_or(changed, iiter <= 1),
            donot_update_direction,
            update_direction,
        )  # , true_fun, false_fun)
        p2 = jnp.where(msk_bnd, 0.0, p2)

        # jax.debug.print('p2 = {}', p2)

        # 4. compute tildex, optimization ignoring the bounds
        q2 = A(additionals, p2)
        # q2 = jnp.where(msk_bnd, 1e5, q2)
        # jax.debug.print('q2 = {}', q2)

        alpha = jnp.vdot(r1, p2) / jnp.vdot(p2, q2)
        x2_trial = x1 + alpha * p2
        # jax.debug.print('x2_trial = {}', x2_trial)

        # 7. check convergence of inner iteration
        rms_xk = jnp.linalg.norm(x2_trial)
        rms_upd = jnp.linalg.norm(x2_trial - x1)
        upd = rms_upd / rms_xk

        # 5. / 2. project x onto feasible domain, bind variables
        outer_it = True  # upd < 1e-10
        # jax.debug.print('upd = {}', upd)

        msk_low_prj = jnp.where(
            outer_it,
            jnp.where(x2_trial < lower_bound, True, False),
            jnp.zeros_like(lower_bound, dtype="bool"),
        )
        msk_upp_prj = jnp.where(
            outer_it,
            jnp.where(x2_trial > upper_bound, True, False),
            jnp.zeros_like(lower_bound, dtype="bool"),
        )

        x2 = jnp.where(msk_low_prj, lower_bound, x2_trial)
        x2 = jnp.where(msk_upp_prj, upper_bound, x2)

        # jax.debug.print('x2 = {}', x2)

        msk_bnd = jnp.where(msk_low_prj, True, msk_bnd)
        msk_bnd = jnp.where(msk_upp_prj, True, msk_bnd)

        changed = jax.lax.cond(
            jnp.count_nonzero(msk_low_prj) + jnp.count_nonzero(msk_upp_prj) > 0,
            true,
            false,
        )

        # 6. update or recompute the gradient
        def increment_residual():
            return g1 + alpha * q2

        def solve_for_residual():
            return A(additionals, x2) - b

        g2 = jax.lax.cond(
            jnp.logical_or(changed, outer_it), solve_for_residual, increment_residual
        )

        # 2. if desired: release constraints with negative gradient
        check_grad = jnp.logical_and(outer_it, changed == False)

        msk_rel = jnp.where(
            check_grad,
            jnp.where(jnp.logical_and(msk_bnd, g2 < -small), True, False),
            jnp.zeros_like(lower_bound, dtype="bool"),
        )
        msk_bnd = jnp.where(msk_rel, False, msk_bnd)

        changed = jax.lax.cond(jnp.count_nonzero(msk_rel) > 0, true, false)

        return (b, x2, g2, r1, p2, msk_bnd, changed, iiter)

    def cond_fun(state):
        b, x2, g2, r1, p2, msk_bnd, changed, iiter = state
        res = jnp.where(msk_bnd, 0, g2)
        jax.debug.print("inner residual = {}", jnp.linalg.norm(res.reshape(-1)))
        return jax.lax.cond(
            jnp.logical_and(jnp.linalg.norm(res.reshape(-1)) > atol, iiter < 100),
            true,
            false,
        )

    x0 = x0.reshape(-1)
    b = b.reshape(-1)

    # jax.debug.print('x0 = {}', x0)
    # jax.debug.print('b = {}', b)

    g0 = A(additionals, x0) - b
    msk_bnd = jnp.where(
        jnp.logical_and(jnp.logical_or(x0 <= lower_bound, x0 >= upper_bound), g0 >= 0),
        True,
        False,
    )
    # jax.debug.print('msk_bnd = {}', msk_bnd)

    iiter = 0
    changed = False

    r0 = -g0
    r0 = jnp.where(msk_bnd, 0.0, r0)
    # jax.debug.print('r0 = {}', r0)

    p1 = r0

    b, x1, g1, r1, p1, msk_bnd, changed, iiter = jax.lax.while_loop(
        cond_fun, body_fun, (b, x0, g0, r0, p1, msk_bnd, changed, iiter)
    )
    return x1


@functools.partial(jax.jit, static_argnums=(0,))
def enhanced_bound_conjugate_gradient(
    A, b, lower_bound, upper_bound, additional, x0=None, atol=1e-5
):

    b, lower_bound, upper_bound, x0, additionals = jax.device_put(
        (b, lower_bound, upper_bound, x0, additional)
    )
    small = 1e-8

    def true():
        return True

    def false():
        return False

    def body_fun(state):
        b, x1, g1, r_prev, p1, msk_bnd, changed, iiter = state

        iiter += 1
        # jax.debug.print('msk = {}', msk_bnd)

        # 3. construct a search direction that is zero in bound variables
        r1 = -g1
        r1 = jnp.where(msk_bnd, 0.0, r1)

        def update_direction():
            beta = (jnp.vdot(r1, r1) - jnp.vdot(r1, r_prev)) / jnp.vdot(r_prev, r_prev)
            p2 = r1 + jnp.maximum(0, beta) * p1
            return p2

        def donot_update_direction():
            return r_prev

        p2 = jax.lax.cond(
            jnp.logical_or(changed, iiter <= 1),
            donot_update_direction,
            update_direction,
        )  # , true_fun, false_fun)
        p2 = jnp.where(msk_bnd, 0.0, p2)

        # jax.debug.print('p2 = {}', p2)

        # 4. compute tildex, optimization ignoring the bounds
        q2 = A(additionals, p2)
        # q2 = jnp.where(msk_bnd, 1e5, q2)
        # jax.debug.print('q2 = {}', q2)

        alpha = jnp.vdot(r1, p2) / jnp.vdot(p2, q2)
        x2_trial = x1 + alpha * p2
        # jax.debug.print('x2_trial = {}', x2_trial)

        # 7. check convergence of inner iteration
        rms_xk = jnp.linalg.norm(x2_trial)
        rms_upd = jnp.linalg.norm(x2_trial - x1)
        upd = rms_upd / rms_xk

        # 5. / 2. project x onto feasible domain, bind variables
        outer_it = upd < 1e-10
        # jax.debug.print('upd = {}', upd)

        msk_low_prj = jnp.where(
            outer_it,
            jnp.where(x2_trial < lower_bound, True, False),
            jnp.zeros_like(lower_bound, dtype="bool"),
        )
        msk_upp_prj = jnp.where(
            outer_it,
            jnp.where(x2_trial > upper_bound, True, False),
            jnp.zeros_like(lower_bound, dtype="bool"),
        )

        x2 = jnp.where(msk_low_prj, lower_bound, x2_trial)
        x2 = jnp.where(msk_upp_prj, upper_bound, x2)

        # jax.debug.print('x2 = {}', x2)

        msk_bnd = jnp.where(msk_low_prj, True, msk_bnd)
        msk_bnd = jnp.where(msk_upp_prj, True, msk_bnd)

        changed = jax.lax.cond(
            jnp.count_nonzero(msk_low_prj) + jnp.count_nonzero(msk_upp_prj) > 0,
            true,
            false,
        )

        # 6. update or recompute the gradient
        def increment_residual():
            return g1 + alpha * q2

        def solve_for_residual():
            return A(additionals, x2) - b

        g2 = jax.lax.cond(
            jnp.logical_or(changed, outer_it), solve_for_residual, increment_residual
        )

        # 2. if desired: release constraints with negative gradient
        check_grad = jnp.logical_and(outer_it, changed == False)

        msk_rel = jnp.where(
            check_grad,
            jnp.where(jnp.logical_and(msk_bnd, g2 < -small), True, False),
            jnp.zeros_like(lower_bound, dtype="bool"),
        )
        msk_bnd = jnp.where(msk_rel, False, msk_bnd)

        changed = jax.lax.cond(jnp.count_nonzero(msk_rel) > 0, true, false)

        return (b, x2, g2, r1, p2, msk_bnd, changed, iiter)

    def cond_fun(state):
        b, x2, g2, r1, p2, msk_bnd, changed, iiter = state
        res = jnp.where(msk_bnd, 0, g2)
        jax.debug.print("inner residual = {}", jnp.linalg.norm(res.reshape(-1)))
        return jax.lax.cond(
            jnp.logical_and(jnp.linalg.norm(res.reshape(-1)) > atol, iiter < 100),
            true,
            false,
        )

    x0 = x0.reshape(-1)
    b = b.reshape(-1)

    # jax.debug.print('x0 = {}', x0)
    # jax.debug.print('b = {}', b)

    g0 = A(additionals, x0) - b
    msk_bnd = jnp.where(
        jnp.logical_and(jnp.logical_or(x0 <= lower_bound, x0 >= upper_bound), g0 >= 0),
        True,
        False,
    )
    # jax.debug.print('msk_bnd = {}', msk_bnd)

    iiter = 0
    changed = False

    r0 = -g0
    r0 = jnp.where(msk_bnd, 0.0, r0)
    # jax.debug.print('r0 = {}', r0)

    p1 = r0

    b, x1, g1, r1, p1, msk_bnd, changed, iiter = jax.lax.while_loop(
        cond_fun, body_fun, (b, x0, g0, r0, p1, msk_bnd, changed, iiter)
    )
    return x1
