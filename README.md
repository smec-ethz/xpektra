
<div align="center">
<img src="docs/assets/xpektra-trans.png" alt="drawing" width="400"/>
<h3 align="center">xpektra : Modular framework for spectral methods</h3>


`xpektra` is a Python library that provides a modularframework for spectral methods.  `xpektra` provide fundamental mathematical building blocks which can be used to construct complex spectral methods. It is built on top of JAX and Equinox, making it easy to use spectral methods in a differentiable way.


</div>

## License
`xpektra` is distributed under the GNU Lesser General Public License v3.0 or later. See `COPYING` and `COPYING.LESSER` for the complete terms. © 2025 ETH Zurich (Mohit Pundir).

## Features

- Functional programming interface for spectral methods
- Differentiable operations using JAX
- Support for FFT and other spectral transforms
- Easy integration with machine learning frameworks

## Installation
Install the current release from PyPI:

```bash
pip install xpektra
```
For development work, clone the repository and install it in editable mode (use your preferred virtual environment tool such as `uv` or `venv`):

```bash
git clone https://gitlab.ethz.ch/smec/software/xpektra.git
cd xpektra
pip install -e .
```

## Usage

`xpektra` provides modular blocks by defining a few operators and spaces that can be used to construct complex spectral methods. These operators and spaces are:

- `SpectralSpace`: Defines the spectral space on which the methods are defined, this includes the FFT, IFFT, and other spectral transforms.
- `TensorOperator`: Defines the tensor operator which performs tensor operations on quantities living on the grid, the operator automatically figures out the order of the tensor.
- `Scheme`: Defines the scheme for discretization which is then used to construct the gradient operator. Currently, `CartesianScheme` is the only scheme available but one can easily define new schemes by subclassing the `Scheme` class.
- `make_field`: Defines the field on which the methods are defined, this includes the field operations and the field creation.
- `ProjectionOperator`: Defines the projection operator which projects the stress field onto the spectral space. Currently, `GalerkinProjection` is the only projection operator available but one can easily define new projection operators by subclassing the `ProjectionOperator` class.

```python
from xpektra import (
    SpectralSpace,
    TensorOperator,
    make_field,
)
from xpektra.scheme import RotatedDifference, Fourier
from xpektra.projection_operator import GalerkinProjection
from xpektra.solvers.nonlinear import (  # noqa: E402
    conjugate_gradient_while,
    newton_krylov_solver,
)


N = 199
shape = (N, N)
length = 1.0
ndim = 2


def create_structure(N):
    Hmid = int(N / 2)
    Lmid = int(N / 2)
    r = int(N / 4)

    structure = np.ones((N, N))
    structure[Hmid - r : Hmid + r + 1, Lmid - r : Lmid + r + 1] -= disk(r)

    return structure


structure = create_structure(N)

tensor = TensorOperator(dim=ndim)
space = SpectralSpace(size=N, dim=ndim, length=length)


def param(X, inclusion, solid):
    props = inclusion * jnp.ones_like(X) * (1 - X) + solid * jnp.ones_like(X) * (X)
    return props


phase_contrast = 1./1e3

# lames constant
lambda_modulus = {"solid": 1.0, "inclusion": phase_contrast}
shear_modulus = {"solid": 1.0, "inclusion": phase_contrast}

bulk_modulus = {}
bulk_modulus["solid"] = lambda_modulus["solid"] + 2 * shear_modulus["solid"] / 3
bulk_modulus["inclusion"] = (
    lambda_modulus["inclusion"] + 2 * shear_modulus["inclusion"] / 3
)

λ0 = param(
    structure, inclusion=lambda_modulus["inclusion"], solid=lambda_modulus["solid"]
)  # lame parameter
μ0 = param(
    structure, inclusion=shear_modulus["inclusion"], solid=shear_modulus["solid"]
)  # lame parameter
K0 = param(structure, inclusion=bulk_modulus["inclusion"], solid=bulk_modulus["solid"])


@eqx.filter_jit
def strain_energy(eps):
    eps_sym = 0.5 * (eps + tensor.trans(eps))
    energy = 0.5 * jnp.multiply(λ0, tensor.trace(eps_sym) ** 2) + jnp.multiply(
        μ0, tensor.trace(tensor.dot(eps_sym, eps_sym))
    )
    return energy.sum()


I = make_field(dim=ndim, N=N, rank=2)
I[:, :, 0, 0] = 1
I[:, :, 1, 1] = 1


def compute_stress(eps):
    return jnp.einsum("..., ...ij->...ij", λ0 * tensor.trace(eps), I) + 2 * jnp.einsum(
        "..., ...ij->...ij", μ0, eps
    )


Ghat = GalerkinProjection(
    scheme=RotatedDifference(space=space), tensor_op=tensor
).compute_operator()

eps = make_field(dim=2, N=N, rank=2)


class Residual(eqx.Module):
    """A callable module that computes the residual vector."""

    Ghat: Array
    space: SpectralSpace = eqx.field(static=True)
    tensor_op: TensorOperator = eqx.field(static=True)
    dofs_shape: tuple = eqx.field(static=True)

    # We can even pre-define the stress function if it's always the same
    # For this example, we'll keep your original `compute_stress` function
    # available in the global scope.

    @eqx.filter_jit
    def __call__(self, eps_flat: Array) -> Array:
        """
        This makes instances of this class behave like a function.
        It takes only the flattened vector of unknowns, as required by the solver.
        """
        eps = eps_flat.reshape(self.dofs_shape)
        sigma = compute_stress(eps)  # Assumes compute_stress is defined elsewhere
        residual_field = self.space.ifft(
            self.tensor_op.ddot(self.Ghat, self.space.fft(sigma))
        )
        return jnp.real(residual_field).reshape(-1)


class Jacobian(eqx.Module):
    """A callable module that represents the Jacobian operator (tangent)."""

    Ghat: Array
    space: SpectralSpace = eqx.field(static=True)
    tensor_op: TensorOperator = eqx.field(static=True)
    dofs_shape: tuple = eqx.field(static=True)

    @eqx.filter_jit
    def __call__(self, deps_flat: Array) -> Array:
        """
        The Jacobian is a linear operator, so its __call__ method
        represents the Jacobian-vector product.
        """
        deps = deps_flat.reshape(self.dofs_shape)
        # Assuming linear elasticity, the tangent is the same as the residual operator
        dsigma = compute_stress(deps)
        jvp_field = self.space.ifft(
            self.tensor_op.ddot(self.Ghat, self.space.fft(dsigma))
        )
        return jnp.real(jvp_field).reshape(-1)


residual_fn = Residual(Ghat=Ghat, space=space, tensor_op=tensor, dofs_shape=eps.shape)
jacobian_fn = Jacobian(Ghat=Ghat, space=space, tensor_op=tensor, dofs_shape=eps.shape)

```
