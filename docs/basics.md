# Building Blocks of `xpektra`

The `xpektra` library is built on a set of modular, JAX-native components. The core philosophy is to provide a "toolkit" that separates the **geometry** (grids), the **calculus** (differentiation rules), and the **physics** (equilibrium constraints).

The library is organized into a hierarchy. As a user, you will primarily interact with the **`SpectralOperator`**, which acts as a facade for the underlying machinery.

1.  **`TensorFields`**: The underlying data structures and algebra engine.
2.  **`SpectralSpace & Transform`**: The "canvas" that defines the grid and the transform (FFT, DCT).
3.  **`Scheme`**: The "rulebook" for how derivatives are calculated.
4.  **`ProjectionOperator`**: The physics engine used to build Green's operators for specific formulations.
5.  **`SpectralOperator`**: The **main interface** that combines space and scheme to provide high-level operations (`grad`, `div`, `fft`).


## The Data: `Tensor Fields`

In **`xpektra`**, physical quantities like stress, strain, or displacement are represented as JAX arrays. The library enforces a consistent **`(spatial..., tensor...)`** memory layout.

This means the spatial grid dimensions `(N_x, N_y, ...)` always come first, followed by the tensor component dimensions `(d, d, ...)`.

  * **`Scalar Field (Rank 0)`** in 2D: `(N, N)`
  * **`Vector Field (Rank 1)`** in 2D: `(N, N, 2)`
  * **`Tensor Field (Rank 2)`** in 2D: `(N, N, 2, 2)`

**`xpektra`** provides a helper function `make_field` to create fields with the correct shape.

```python
from xpektra import make_field

# Create a rank-2 (2x2) tensor field on a 128x128 grid
stress = make_field(dim=2, N=128, rank=2)
print(stress.shape) # Output: (128, 128, 2, 2)
```

## The Foundation: `SpectralSpace` and `Transform`

The `SpectralSpace` class defines the geometry of your problem. It holds the physical dimensions, the grid resolution, and the **transform strategy** (e.g., FFT or DCT) used to move between real and spectral space. We define the transform strategy in the `Transform` class.

```python
from xpektra import SpectralSpace
from xpektra.transform import FFTTransform

# 1. Choose a Transform Strategy
transform = FFTTransform(dim=2)

# 2. Define the Space
space = SpectralSpace(lengths =(1, 10), shape=(64, 256), transform=transform)
```

!!! tip "Non-Square/Non-Cube Grids"

    The `SpectralSpace` class is defined for rectangular/square grids in 2D and cuboid grids in 3D.

    
## The Calculus: `Scheme`

Discretization is handled by `Scheme` objects. These define *how* derivatives are computed. In `xpektra` we have divided the scheme based on the type of grid and how the differentiation looks in Fourier space. 

Currently, we support cartersian based schemes with diagonalized differentiation operator in Fourier space.Some of the available schemes include:

  * **`Fourier`**: The standard spectral derivative ($D_k = i \xi_k$). Accurate but prone to Gibbs ringing.
  * **`CentralDifference`**: A robust finite difference scheme ($D_k = i \sin(\xi_k h)/h$). Equivalent to Linear Finite Elements; eliminates ringing.
  * **`RotatedDifference`**: An advanced finite difference scheme (Willot, 2015) offering high stability.



```python
from xpektra.scheme import RotatedDifference

# Create a scheme attached to your space
scheme = RotatedDifference(space=space)
```

!!! tip "Extending to Non-Cartesian grids or non-diagonalized differentiation"

    We are actively working on extending `xpektra` to support non-Cartesian grids and non-diagonalized differentiation operators. But one can easily implement their own scheme by subclassing the `Scheme` class as shown in the documentation.


## The Interface: `SpectralOperator`

The **`SpectralOperator`** is the heart of the library. It combines the `Space` and the `Scheme` into a single, powerful toolkit.

Instead of managing transforms and derivatives manually, you use this operator to perform high-level mathematical operations on your fields.

```python
from xpektra.spectral_operator import SpectralOperator

# Initialize the main operator
op = SpectralOperator(space=space, scheme=scheme)

# --- Calculus Operations ---
grad_u = op.grad(u)       # Computes gradient of scalar u
div_v  = op.div(v)        # Computes divergence of vector v
sym_grad = op.sym_grad(u) # Computes symmetric gradient (strain)

# --- Transform Operations ---
u_hat = op.forward(u)     # Forward transform (FFT/DCT)
u_real = op.inverse(u_hat) # Inverse transform
```

The `SpectralOperator` is "smart" that means it delegates the math to the specific `Scheme` you chose, ensuring consistency.

## The Algebra: `TensorOperator`

The `TensorOperator` is the low-level engine handling tensor contractions (dot products, traces) on the grid.

While it powers the library internally, you rarely need to instantiate it yourself. The **`SpectralOperator`** exposes the most common tensor operations directly for convenience:

```python
# Dot product (contraction)
C = op.dot(A, B) 

# Double dot product (A : B)
energy = op.ddot(sigma, epsilon)

# Transpose
grad_u_T = op.trans(grad_u)
```

If you need advanced tensor manipulations, the underlying engine is available via `op.tensor_op`.


## The Physics: `ProjectionOperator`

For solvers based on the Lippmann-Schwinger equation or Galerkin methods, you need a **`ProjectionOperator`**. This component pre-computes the 4th-order Green's operator ($\hat{\mathbb{\Gamma}}^0$ or $\hat{\mathbb{G}}$) in spectral space, which enforces equilibrium constraints.

You typically instantiate a specific type of projection based on your formulation:

### Galerkin Projection

Used for the material-independent variational formulation (recommended for Newton-Krylov solvers).

```python
from xpektra.projection_operator import GalerkinProjection

# Build the operator from your scheme
projection = GalerkinProjection(scheme=scheme)

# Apply projection: P(sigma_hat)
residual_hat = projection.project(sigma_hat)
```
!!! tip "Matrix-free Galerkin Projection"

    `xpektra` implements a matrix-free `GalerkinProjection` that means the projection is applied without explicitly forming the matrix, which can be memory-intensive for large problems.


### Moulinec-Suquet Projection

Used for the classic fixed-point scheme with a reference material ($\mathbb{C}_0$). This Moulinec-Suquet projection is used to solve the Lippmann-Schwinger equation: $\varepsilon^{k+1} = \bar{E} - \Gamma^0 * (\sigma(\varepsilon^k) - C_0 : \varepsilon^k)$.


```python
from xpektra.projection_operator import MoulinecSuquetProjection

# Build with reference material properties
ms_proj = MoulinecSuquetProjection(
    scheme=scheme, 
    lambda0=100.0, 
    mu0=25.0
)
```
