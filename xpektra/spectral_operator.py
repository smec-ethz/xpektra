import equinox as eqx
from jax import Array

from xpektra.scheme import Scheme
from xpektra.space import SpectralSpace
from xpektra.tensor_operator import TensorOperator


class SpectralOperator(eqx.Module):
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
    operator = SpectralOperator(space, scheme)
    grad_u = operator.grad(u)
    div_v = operator.div(v)
    sym_grad_u = operator.sym_grad(u)
    ```

    """

    space: SpectralSpace = eqx.field(static=True)
    scheme: Scheme
    tensor: TensorOperator = eqx.field(static=True)

    def __init__(
        self,
        space: SpectralSpace,
        scheme: Scheme,
    ):
        self.space = space
        self.scheme = scheme
        self.tensor = TensorOperator(dim=len(self.space.lengths))

    @eqx.filter_jit
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

    @eqx.filter_jit
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

    @eqx.filter_jit
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

    @eqx.filter_jit
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

    @eqx.filter_jit
    def forward(self, u: Array) -> Array:
        """Applies the forward transform to the input real-valued array u.

        ***Arguments***
        - u: A real-valued array of shape (N,)*dim.
        ***Returns***
        - The transformed array u_hat, a complex-valued array of shape (N,)*dim.
        """
        return self.space.transform.forward(u)

    @eqx.filter_jit
    def inverse(self, u_hat: Array) -> Array:
        """Applies the inverse transform to the input complex-valued array u_hat.

        ***Arguments***
        - u_hat: A complex-valued array of shape (N,)*dim.
        ***Returns***
        - The inverse transformed array u, a real-valued array of shape (N,)*dim.
        """
        return self.space.transform.inverse(u_hat).real

    @eqx.filter_jit
    def ddot(self, A: Array, B: Array) -> Array:
        """Applies the double dot product to the input arrays A and B.

        ***Arguments***
        - A: A real-valued array of shape (N,)*dim.
        - B: A real-valued array of shape (N,)*dim.
        ***Returns***
        - The double dot product of A and B, a real-valued array of shape (N,)*dim.
        """
        return self.tensor.ddot(A, B)

    @eqx.filter_jit
    def trace(self, A: Array) -> Array:
        """Applies the trace operator to the input array A.

        ***Arguments***
        - A: A real-valued array of shape (N,)*dim.
        ***Returns***
        - The trace of A, a real-valued array of shape (N,).
        """
        return self.tensor.trace(A)

    @eqx.filter_jit
    def trans(self, A: Array) -> Array:
        """Applies the transpose operator to the input array A.

        ***Arguments***
        - A: A real-valued array of shape (N,)*dim.
        ***Returns***
        - The transpose of A, a real-valued array of shape (N,)*dim.
        """
        return self.tensor.trans(A)

    @eqx.filter_jit
    def dyad(self, A: Array, B: Array) -> Array:
        """Applies the dyadic product to the input arrays A and B.

        ***Arguments***
        - A: A real-valued array of shape (N,)*dim.
        - B: A real-valued array of shape (N,)*dim.
        ***Returns***
        - The dyadic product of A and B, a real-valued array of shape (N,)*dim.
        """
        return self.tensor.dyad(A, B)

    @eqx.filter_jit
    def dot(self, A: Array, B: Array) -> Array:
        """Applies the dot product to the input arrays A and B.

        ***Arguments***
        - A: A real-valued array of shape (N,)*dim.
        - B: A real-valued array of shape (N,)*dim.
        ***Returns***
        - The dot product of A and B, a real-valued array of shape (N,)*dim.
        """
        return self.tensor.dot(A, B)
