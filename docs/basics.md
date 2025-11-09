# Building Blocks of `xpektra`

The `xpektra` library is built on a set of modular, JAX-native components. The core philosophy is to provide a "toolkit" of these blocks, allowing you to assemble different types of spectral solvers (Galerkin, Moulinec-Suquet, displacement-based) with minimal, reusable code.

These blocks are:

1.  `Tensor Fields`: The data, representing physical quantities on a grid.
2.  `TensorOperator`: The tools to perform tensor algebra on those fields.
3.  `SpectralSpace`: The "canvas" that defines the grid and FFT operations.
4.  `Scheme`: The rules for defining derivatives (e.g., Fourier, Finite Difference).
5.  `ProjectionOperator`: The core "engine" that enforces mechanical laws.


## Tensor Fields

In **Xpektra**, all physical fields (like stress, strain, or displacement) are represented as JAX or NumPy arrays. The library follows a consistent `(spatial..., tensor...)` memory layout for all fields.

This means the spatial grid dimensions `(N_x, N_y, ...)` always come first, followed by the tensor component dimensions `(d, d, ...)`.

  * **Rank 0 (Scalar) Field** in 2D: `(N, N)`
  * **Rank 1 (Vector) Field** in 2D: `(N, N, 2)`
  * **Rank 2 (Tensor) Field** in 2D: `(N, N, 2, 2)`
  * **Rank 2 (Tensor) Field** in 3D: `(N, N, N, 3, 3)`

**Xpektra** provides a convenience function `make_field` to create an empty tensor field with the correct shape.

```python
from xpektra import make_field  

# Create a rank-2 (2x2) tensor field on a 128x128 grid
stress_field = make_field(dim=2, N=128, rank=2)
print(stress_field.shape)
# Output: (128, 128, 2, 2)

# Create a rank-1 (vector) field on a 64x64x64 grid
displacement_field = make_field(dim=3, N=64, rank=1)
print(displacement_field.shape)
# Output: (64, 64, 64, 3)
```


## Tensor Operations on Grid

To perform tensor algebra on these fields, **Xpektra** provides the `TensorOperator` class. This class is a lightweight helper that provides methods like `dot`, `ddot`, `trace`, etc., which automatically handle the spatial dimensions using optimized `einsum` operations.

You only need to initialize it with the spatial dimension `dim`.

```python
from xpektra import TensorOperator, make_field

tensor_op = TensorOperator(dim=2)
```

The `TensorOperator` uses pre-defined `einsum` rules to perform the correct contraction based on the ranks of the input fields.

### Common Operations

#### Dot Product: `.dot()`

Performs a single tensor contraction.

  * **Tensor-Tensor (`...ij,...jk->...ik`)**:
    ```python
    A = make_field(dim=2, N=12, rank=2)
    B = make_field(dim=2, N=12, rank=2)
    C = tensor_op.dot(A, B)
    # C.shape = (12, 12, 2, 2)
    ```
  * **Tensor-Vector (`...ij,...j->...i`)**:
    ```python
    A = make_field(dim=2, N=12, rank=2)
    v = make_field(dim=2, N=12, rank=1)
    w = tensor_op.dot(A, v)
    # w.shape = (12, 12, 2)
    ```

#### Double Dot Product: `.ddot()`

Performs a double tensor contraction.

  * **4th-Order-2nd-Order (`...ijkl,...lk->...ij`)**:
    ```python
    C4 = make_field(dim=2, N=12, rank=4)
    eps = make_field(dim=2, N=12, rank=2)
    sig = tensor_op.ddot(C4, eps)
    # sig.shape = (12, 12, 2, 2)
    ```

#### Dyadic Product: `.dyad()`

Creates a higher-rank tensor from two lower-rank tensors.

  * **Vector-Vector (`...i,...j->...ij`)**:
    ```python
    v1 = make_field(dim=2, N=12, rank=1)
    v2 = make_field(dim=2, N=12, rank=1)
    A = tensor_op.dyad(v1, v2)
    # A.shape = (12, 12, 2, 2)
    ```

#### Trace: `.trace()`

Calculates the trace of a tensor.

  * **Trace of Rank-2 Tensor (`...ii->...`)**:
    ```python
    A = make_field(dim=2, N=12, rank=2)
    tr_A = tensor_op.trace(A)
    # tr_A.shape = (12, 12)
    ```

#### Transpose: `.trans()`

Calculates the transpose of a tensor.

  * **Transpose of Rank-2 Tensor (`...ij->...ji`)**:
    ```python
    A = make_field(dim=2, N=12, rank=2)
    A_T = tensor_op.trans(A)
    # A_T.shape = (12, 12, 2, 2)
    ```


## Spectral Space

The `SpectralSpace` class is the canvas for all operations. It defines the grid's properties (size, dimensions, length) and provides the JAX-native `fft` and `ifft` methods.

These methods are crucial as they correctly apply the transform along the spatial axes (`(N, N, ...)`) while leaving the tensor component axes (`(..., d, d)`) untouched.

```python
from xpektra import SpectralSpace

space = SpectralSpace(dim=2, size=128, length=1.0)
```

Its key attributes and methods are:

  * `dim`: The number of spatial dimensions (e.g., 2).
  * `size`: The number of grid points per dimension (e.g., 128).
  * `length`: The physical length of the domain.
  * `wavenumber_vector()`: Returns the real-valued wavenumber vector $\xi$.
  * `frequency_vector()`: Returns the integer frequency vector.
  * `fft(field)`: Performs the forward FFT on a tensor field.
  * `ifft(field_hat)`: Performs the inverse FFT on a tensor field.


## Discretization Schemes

Discretization is handled by `Scheme` objects. These objects are the "rulebook" for how to calculate derivatives.

The library uses an **abstract-or-final** design pattern. As a user, you will typically just instantiate one of the **final** schemes provided. The scheme object's primary purpose is to provide the **gradient operator** (`.gradient_operator`) in Fourier space, which is used to build the final projection operator.

Available schemes include:

  * `Fourier`: The standard spectral derivative $D_k = i \xi_k$.
  * `CentralDifference`: Finite difference $D_k = i \sin(\xi_k \Delta x) / \Delta x$.
  * `RotatedDifference`: The scheme used by Willot (2015).
  * And other common finite difference stencils.


```python
from xpektra import SpectralSpace
from xpektra.scheme import Fourier, CentralDifference

space = SpectralSpace(dim=2, size=128)

# --- Choose a discretization scheme ---

# Use the standard spectral derivative
scheme_fourier = Fourier(space)

# Use a central difference (LFE-equivalent) scheme
scheme_fd = CentralDifference(space)

# Get the gradient operator in Fourier space (shape (128, 128, 2))
grad_op = scheme_fd.gradient_operator
```

## Green's Operators (Projection Operators)

The `ProjectionOperator` is the core "engine" of the solver. It's an abstract class for operators that enforce the mechanical constraints (like equilibrium and compatibility). These operators are pre-computed in Fourier space and are represented as a 4th-order tensor field `Ghat`.

You will typically instantiate one of the **final** implementations.

### Fourier-Galerkin Operator

This is the standard, material-independent projection operator $G_{hat}$ used in the Galerkin formulation. It is implemented by the `GalerkinProjection` class.

It is constructed simply by passing it the `Scheme` and `TensorOperator` you've already defined.

```python
from xpektra.projection_operator import GalerkinProjection

# Build the Galerkin operator from the chosen scheme
projection = GalerkinProjection(scheme=scheme_fd, tensor_op=tensor_op)

# Get the pre-computed 4th-order operator
Ghat = projection.Ghat
# Ghat.shape = (128, 128, 2, 2, 2, 2)
```

This `projection` object can now be used in a solver to project any stress field $\hat{\sigma}$ onto the compatible strain space: `eps_hat = projection.project(sigma_hat)`.

### Moulinec-Suquet Operator

This is the Green's operator $\Gamma^0$ used in the Lippmann-Schwinger equation. It depends on a homogeneous isotropic reference material $C_0$ (defined by Lamé parameters $\lambda_0$ and $\mu_0$). It is implemented by the `MoulinecSuquetProjection` class.

```python
from xpektra.projection_operator import MoulinecSuquetProjection

# Define reference material properties
lambda0 = 100.0
mu0 = 25.0

# Build the MS operator
ms_operator = MoulinecSuquetProjection(
    scheme=scheme_fourier, 
    tensor_op=tensor_op, 
    lambda0=lambda0, 
    mu0=mu0
)

# Get the pre-computed operator
Ghat_ms = ms_operator.Ghat
```

This `ms_operator` object can be used to solve the Lippmann-Schwinger equation: $\varepsilon^{k+1} = \bar{E} - \Gamma^0 * (\sigma(\varepsilon^k) - C_0 : \varepsilon^k)$.