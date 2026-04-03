from dataclasses import dataclass, field

import jax
from jax import Array

from xpektra.projection_operator import ProjectionOperator
from xpektra.scheme import Scheme
from xpektra.space import SpectralSpace
from xpektra.tensor_operator import TensorOperator


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class SpectralOperator:
    """
    A spectral operator defined by spectral space, differential scheme.
    It provides methods to compute gradient, divergence, symmetric gradient,
    and also tensor operations like dot, ddot, trace, transpose, and dyadic product.

    ***Arguuments**
    - space: The spectral space.
    - scheme: The differential scheme.

    ***Returns***
    - The spectral operator.

    Example:

    ```
    op = SpectralOperator(scheme=scheme, space=space, projection=GalerkinProjection())
    grad_u = op.grad(u)
    div_v = op.div(v)
    ```

    To swap a single attribute (e.g. the projection) on a frozen instance, use
    ``dataclasses.replace``::

        from dataclasses import replace

        new_op = replace(op, projection=MoulinecSuquetProjection(lambda0=10.0, mu0=1.0))

    ``__post_init__`` runs on the new instance, so ``build(space)`` is called
    automatically for the new projection.
    """

    scheme: Scheme
    space: SpectralSpace = field(metadata=dict(static=True))
    projection: ProjectionOperator | None = field(default=None)
    tensor: TensorOperator | None = field(default=None, metadata=dict(static=True))

    def __post_init__(self):
        object.__setattr__(self, "tensor", TensorOperator(dim=len(self.space.lengths)))
        if self.projection is not None:
            object.__setattr__(
                self, "projection", self.projection.build(self.space)
            )

    @jax.jit
    def grad(self, u: Array) -> Array:
        """Applies the gradient operator to the input real-valued array u.

        ***Arguments***
        - u: A real-valued array of shape (N,)*dim.

        ***Returns***
        - The gradient of u, a real-valued array of shape (N,)*dim + (dim,).

        """
        u_hat = self.space.transform.forward(u)
        grad_u_hat = self.scheme.apply_gradient(u_hat)
        grad_u = self.space.transform.inverse(grad_u_hat)
        return grad_u.real

    @jax.jit
    def div(self, v: Array) -> Array:
        """Applies the divergence operator to the input real-valued array v.

        ***Arguments***
        - v: A real-valued array of shape (N,)*dim + (dim,).
        ***Returns***
        - The divergence of v, a real-valued array of shape (N,)*dim.
        """

        v_hat = self.space.transform.forward(v)
        div_v_hat = self.scheme.apply_divergence(v_hat)
        div_v = self.space.transform.inverse(div_v_hat)
        return div_v.real

    @jax.jit
    def sym_grad(self, u: Array) -> Array:
        """Applies the symmetric gradient operator to the input real-valued array u.

        ***Arguments***
        - u: A real-valued array of shape (N,)*dim.

        ***Returns***
        - The symmetric gradient of u, a real-valued array of shape (N,)*dim + (dim, dim).
        """
        u_hat = self.space.transform.forward(u)
        sym_grad_u_hat = self.scheme.apply_symmetric_gradient(u_hat)
        sym_grad_u = self.space.transform.inverse(sym_grad_u_hat)
        return sym_grad_u.real

    @jax.jit
    def laplacian(self, u: Array) -> Array:
        """Applies the Laplacian operator to the input real-valued array u.

        ***Arguments***
        - u: A real-valued array of shape (N,)*dim.
        ***Returns***
        - The Laplacian of u, a real-valued array of shape (N,)*dim.
        """
        u_hat = self.space.transform.forward(u)
        lap_u_hat = self.scheme.apply_laplacian(u_hat)
        lap_u = self.space.transform.inverse(lap_u_hat)
        return lap_u.real

    @jax.jit
    def forward(self, u: Array) -> Array:
        """Applies the forward transform to the input real-valued array u.

        ***Arguments***
        - u: A real-valued array of shape (N,)*dim.
        ***Returns***
        - The transformed array u_hat, a complex-valued array of shape (N,)*dim.
        """
        return self.space.transform.forward(u)

    @jax.jit
    def inverse(self, u_hat: Array) -> Array:
        """Applies the inverse transform to the input complex-valued array u_hat.

        ***Arguments***
        - u_hat: A complex-valued array of shape (N,)*dim.
        ***Returns***
        - The inverse transformed array u, a real-valued array of shape (N,)*dim.
        """
        return self.space.transform.inverse(u_hat).real

    @jax.jit
    def project(self, field_hat: Array) -> Array:
        """Apply the projection operator to a Fourier-space field.

        Delegates to ``self.projection.project()``, passing the scheme's
        gradient operator and the spectral space so that large arrays are
        not duplicated across pytree leaves.

        Args:
            field_hat: Field in Fourier space.
        Returns:
            The projected field in Fourier space.
        Raises:
            ValueError: If no projection operator is set.
        """
        if self.projection is None:
            raise ValueError("No projection operator set on this SpectralOperator")
        return self.projection.project(
            field_hat, self.scheme.gradient_operator, self.space
        )

    def ddot(self, A: Array, B: Array) -> Array:
        """Applies the double dot product to the input arrays A and B.

        ***Arguments***
        - A: A real-valued array of shape (N,)*dim.
        - B: A real-valued array of shape (N,)*dim.
        ***Returns***
        - The double dot product of A and B, a real-valued array of shape (N,)*dim.
        """
        return self.tensor.ddot(A, B)

    def trace(self, A: Array) -> Array:
        """Applies the trace operator to the input array A.

        ***Arguments***
        - A: A real-valued array of shape (N,)*dim.
        ***Returns***
        - The trace of A, a real-valued array of shape (N,).
        """
        return self.tensor.trace(A)

    def trans(self, A: Array) -> Array:
        """Applies the transpose operator to the input array A.

        ***Arguments***
        - A: A real-valued array of shape (N,)*dim.
        ***Returns***
        - The transpose of A, a real-valued array of shape (N,)*dim.
        """
        return self.tensor.trans(A)

    def dyad(self, A: Array, B: Array) -> Array:
        """Applies the dyadic product to the input arrays A and B.

        ***Arguments***
        - A: A real-valued array of shape (N,)*dim.
        - B: A real-valued array of shape (N,)*dim.
        ***Returns***
        - The dyadic product of A and B, a real-valued array of shape (N,)*dim.
        """
        return self.tensor.dyad(A, B)

    def dot(self, A: Array, B: Array) -> Array:
        """Applies the dot product to the input arrays A and B.

        ***Arguments***
        - A: A real-valued array of shape (N,)*dim.
        - B: A real-valued array of shape (N,)*dim.
        ***Returns***
        - The dot product of A and B, a real-valued array of shape (N,)*dim.
        """
        return self.tensor.dot(A, B)
