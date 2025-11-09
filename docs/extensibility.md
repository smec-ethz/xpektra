# Extensibility: Building Your Own Methods

The modular, "abstract-or-final" design of `xpektra` is not just an internal feature; it's an open invitation for you to extend the library. You can implement entirely new schemes, formulations, or solvers without modifying any of the core `xpektra` code.

The library's abstract classes (`Scheme`, `CartesianScheme`, `ProjectionOperator`) define a clear API "contract." To add new functionality, you simply create a new class that inherits from one of these base classes and provides the required methods.

Here are a few examples of how you could extend the library.

### Example 1: Implementing a New Discretization `Scheme`

**Goal:** You want to implement a specific finite difference scheme, like the `TETRA2` method, which is known for its stability.

**How:** You create a new class that inherits from `CartesianScheme`. Because the `TETRA2` logic is complex and non-separable, you would override the entire `_compute_gradient_operator` method to implement its unique mixing formula.

```python
from xpektra.scheme import CartesianScheme

class TETRA2(CartesianScheme):
    """
    Implements the TETRA2 finite difference scheme by overriding
    the gradient operator computation.
    """
    def _compute_gradient_operator(self) -> Array:
        # 1. Get wavenumber meshes from the base class
        xi, yi, zi = self._wavenumbers_mesh
        
        # 2. Implement the private methods for the T1 and T2 operators
        # D_T1 = self._operator_T1(...)
        # D_T2 = self._operator_T2(...)
        
        # 3. Implement the mixing logic
        # D_mixed = 0.5 * D_T1 + 0.5 * D_T2
        
        # 4. Stack and return the final operator
        # return jnp.stack([D_mixed_x, D_mixed_y, D_mixed_z], axis=-1)
        pass # Your implementation here

# --- How you use it ---
# space = SpectralSpace(dim=3, size=128)
# my_scheme = TETRA2(space)
# projection = GalerkinProjection(scheme=my_scheme, tensor_op=tensor_op)
```

The rest of the library (`GalerkinProjection`, `NewtonKrylovSolver`) will now use your new scheme without any changes.

### Example 2: Implementing a New `ProjectionOperator`

**Goal:** You want to implement an accelerated fixed-point solver, like the `Eyre-Milton (EM)` or `Augmented Lagrangian (ADMM)` method. These methods use a different Green's operator $\Gamma^\gamma$ that is "polarized" by a parameter $\gamma$.

**How:** You create a new class that inherits from `ProjectionOperator` and implements the `_compute_operator` method to build this new $\Gamma^\gamma$ tensor.

```python
from xpektra.projection_operator import ProjectionOperator

class EyreMiltonProjection(ProjectionOperator):
    """
    Implements the polarized Green's operator for the
    Eyre-Milton (EM) accelerated fixed-point scheme.
    """
    def __init__(self, scheme, tensor_op, gamma):
        self.gamma = gamma # Store the acceleration parameter
        super().__init__(scheme, tensor_op)

    def _compute_operator(self) -> Array:
        # Implement the specific formula for the EM operator,
        # which depends on self.gamma and the gradient operator.
        pass # Your implementation here
```

### Example 3: Implementing a New Solver Strategy

**Goal:** You aren't satisfied with the basic fixed-point or Newton-Krylov solvers and want to use `Anderson Acceleration` to solve the root-finding problem $R(\varepsilon) = 0$.

**How:** `xpektra` provides the residual calculation as a self-contained, JIT-able object (like the `Residual` class in the DBFFT example). You don't need to change any `xpektra` class. You simply write your own solver function that accepts this `Residual` object as an argument.

```python
import jax

# Your custom solver function
def anderson_solver(residual_fn: Callable, eps_initial: Array, max_iter: int):
    """
    A custom solver that takes a residual function and finds its root
    using Anderson acceleration.
    """
    # 1. Get the residual (a JIT-able function-like object)
    # R = residual_fn
    
    # 2. Implement the Anderson acceleration logic
    # ... your solver loop ...
    # eps_new = ... R(eps_old) ...
    pass

# --- How you use it ---
# residual_fn = Residual(...)
# eps_final = anderson_solver(residual_fn, eps_initial, max_iter=50)
```

This powerful, modular design is the central philosophy of `xpektra`, allowing you to focus on the novel parts of your research while relying on the library's stable, optimized building blocks.