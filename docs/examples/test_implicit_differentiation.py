import jax
import jax.numpy as jnp
from jax.scipy.sparse import linalg as jax_sparse_linalg


# --- 1. Define the Problem (Residual Function) ---
# We want to find x* such that R(x*, theta) = 0
def residual(x, theta):
    return x**2 - theta


# --- 2. Define the Solver (Forward Pass) ---
# This is decorated to tell JAX to use a custom gradient
@jax.custom_vjp
def find_root(theta):
    """
    Finds x* such that x*^2 - theta = 0 using Newton's method.
    """
    # Use jax.lax.stop_gradient on the *initial guess*
    # This is important! We don't want to differentiate through the guess.
    x_guess = jax.lax.stop_gradient(theta)  # A reasonable guess, e.g., x_guess = 1.0

    # Simple Newton-Raphson loop
    # We use jax.lax.fori_loop for JIT-compatibility
    def body_fun(i, x):
        # R(x) = x^2 - theta
        # J_x = dR/dx = 2x
        # Update: x_new = x - R(x) / J_x
        x_new = x - (x**2 - theta) / (2.0 * x)
        return x_new

    # Run for a fixed number of iterations
    # In a real solver, you'd use a convergence criteria (e.g., in a lax.while_loop)
    x_star = jax.lax.fori_loop(0, 10, body_fun, x_guess)
    return x_star


# --- 3. Define the VJP (Backward Pass) ---


# First, define the *forward* part for the custom VJP
# It must return the result AND any values needed for the backward pass
def find_root_fwd(theta):
    # Run the solver
    x_star = find_root(theta)
    # Save x_star and theta for the backward pass
    residuals = (x_star, theta)
    return x_star, residuals


# Now, define the *backward* part
def find_root_bwd(residuals, g):
    """
    Computes the Vector-Jacobian Product (VJP)
    g: incoming gradient, (dL/dx*)
    residuals: values saved from the forward pass (x*, theta)
    """
    x_star, theta = residuals

    # We need to solve for g_theta = g * (dx*/dtheta)
    # From our formula: dx*/dtheta = - (J_x)^-1 * J_theta
    # So: g_theta = g * [- (J_x)^-1 * J_theta]
    #
    # We rearrange to solve a linear system:
    # Let lambda^T = g * (J_x)^-1  =>  J_x^T * lambda = g^T
    # Then: g_theta = -lambda^T * J_theta

    # --- Step A: Compute Jacobians at the converged solution (x*, theta) ---

    # J_x = dR/dx = 2x
    # We use jacfwd (or jacrev) on our residual function
    # (jnp.array([x_star]) is used because jacobians expect arrays)
    J_x_fn = jax.jacfwd(residual, argnums=0)
    J_x = J_x_fn(x_star, theta)  # This will be 2*x_star

    # J_theta = dR/dtheta = -1
    J_theta_fn = jax.jacfwd(residual, argnums=1)
    J_theta = J_theta_fn(x_star, theta)  # This will be -1.0

    # --- Step B: Solve the linear system J_x^T * lambda = g ---
    # In 1D, this is just lambda = g / J_x
    # In general, use jax.numpy.linalg.solve
    # Note: J_x is a scalar (2*x_star), g is the incoming gradient (dL/dx*)
    lambda_val = g / J_x  # jax.numpy.linalg.solve(J_x, g)  # Equivalent to g / J_x

    # --- Step C: Compute the final gradient w.r.t. theta ---
    # g_theta = -lambda * J_theta
    # Note: We must return a tuple, one gradient per input (here, just theta)
    g_theta = -lambda_val * J_theta

    return (g_theta,)  # Must be a tuple


# THIS IS THE MODIFIED BACKWARD PASS
def find_root_bwd_cg(residuals, v):
    """
    Computes the VJP using a matrix-free iterative solver (GMRES).
    v: incoming gradient, (dL/dx*)
    """
    x_star, theta = residuals

    # --- Step 1: Solve J_x^T * lambda = v ---

    # Define the linear operator A(p) = J_x^T * p
    # This is a VJP of the residual w.r.t. x
    # We "peel off" the vjp function for R(x) at x_star
    _, vjp_fn_x = jax.vjp(lambda x: residual(x, theta), x_star)

    # op_A is the function GMRES needs.
    def op_A(p):
        return vjp_fn_x(p)[0]  # vjp_fn returns a tuple

    # Solve A(lambda) = v for lambda
    # We use GMRES because J_x^T is not guaranteed to be SPD
    lambda_val, _ = jax_sparse_linalg.gmres(op_A, v)

    # --- Step 2: Compute g_theta = - (J_theta^T * lambda) ---

    # This is a VJP of the residual w.r.t. theta
    _, vjp_fn_theta = jax.vjp(lambda t: residual(x_star, t), theta)

    # The [0] is to select the gradient for the first (and only) arg
    g_theta = -vjp_fn_theta(lambda_val)[0]

    return (g_theta,)  # Must be a tuple of gradients


# --- 4. "Define" the VJP for the function ---
find_root.defvjp(find_root_fwd, find_root_bwd_cg)


# --- 5. Test it! ---


# Let's create a simple loss function L(theta) = find_root(theta)
# We expect dL/dtheta = d(find_root)/dtheta = 1 / (2*x*)
def loss_fn(theta):
    return find_root(theta)


# Get the gradient of the loss
grad_loss = jax.grad(loss_fn)

# Test at a specific value
theta_val = 9.0
x_star_val = find_root(theta_val)  # This will be ~3.0
gradient_val = grad_loss(theta_val)

print(f"Theta: {theta_val}")
print(f"Converged x* (sqrt(theta)): {x_star_val}")
print(f"Gradient d(x*)/d(theta): {gradient_val}")

# Check against the analytical answer
# expected = 1 / (2 * x*) = 1 / (2 * 3.0) = 1/6 = 0.1666...
expected_grad = 1.0 / (2.0 * x_star_val)
print(f"Expected Gradient (1 / (2*x*)): {expected_grad}")
print(
    f"JAX-computed and expected gradients match: {jnp.allclose(gradient_val, expected_grad)}"
)
