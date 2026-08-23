# Extensibility: Building Your Own Methods

The modular, "abstract-or-final" design of `xpektra` is not just an internal feature; it's an open invitation for you to extend the library. You can implement entirely new schemes, formulations, or solvers without modifying any of the core `xpektra` code.

The library's abstract classes (`Scheme`, `FiniteDifferenceScheme`, `ProjectionOperator`) define a clear API "contract." To add new functionality, you simply create a new class that inherits from one of these base classes and provides the required methods.

Here are a few examples of how you could extend the library.

!!! example "Implementing a New Discretization `Scheme`"

    **Goal:** You want to implement a specific finite difference scheme, for example a wider central difference.

    **How:** Almost always you inherit from `FiniteDifferenceScheme` and declare only the `stencils` — a list of `(offset, weight)` pairs per spatial direction. The base class turns each stencil into its Fourier symbol symbolically, and the four differential operations come for free from `Scheme`.

```python
import sympy as sp
from xpektra.scheme import FiniteDifferenceScheme, _unit_offset

class FourthOrderCentralScheme(FiniteDifferenceScheme):
    """Fourth-order central difference, built from its stencil."""

    @property
    def stencils(self):
        h = sp.symbols(f"h_1:{self.dim + 1}", real=True)
        stencils = []
        for i in range(self.dim):
            stencils.append([
                (_unit_offset(i, self.dim, -2),  1 / (12 * h[i])),
                (_unit_offset(i, self.dim, -1), -8 / (12 * h[i])),
                (_unit_offset(i, self.dim,  1),  8 / (12 * h[i])),
                (_unit_offset(i, self.dim,  2), -1 / (12 * h[i])),
            ])
        return stencils

# --- How you use it ---
# space = SpectralSpace(lengths=(1.0,) * 3, shape=(128,) * 3, transform=FFTTransform(dim=3))
# my_scheme = FourthOrderCentralScheme(space)
# projection = GalerkinProjection(scheme=my_scheme, tensor_op=tensor_op)
```

The rest of the library (`GalerkinProjection`, `NewtonKrylovSolver`) will now use your new scheme without any changes.

!!! example "Implementing a New `ProjectionOperator`"

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

!!! example "Implementing a New Solver Strategy"

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
