import jax.numpy as jnp
import numpy as np
from jax import Array
import equinox as eqx

from xpektra.scheme import Scheme
from xpektra.space import SpectralSpace
from xpektra.tensor_operator import TensorOperator



class SpectralOperator(eqx.Module):
    """
    A spectral operator defined by its size, dimension, length, transform, and scheme.
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
        
        Args:
            u: A real-valued array of shape (N,)*dim.
        Returns:
            The gradient of u, a real-valued array of shape (N,)*dim + (dim,).
    
        """
        u_hat = self.space.transform.forward(u)
        grad_u_hat = self.scheme.apply_gradient(u_hat)
        grad_u = self.space.transform.inverse(grad_u_hat)
        return grad_u.real
    
    @eqx.filter_jit
    def div(self, v: Array) -> Array:
        """Applies the divergence operator to the input real-valued array v.
        
        Args:
            v: A real-valued array of shape (N,)*dim + (dim,).
        Returns:
            The divergence of v, a real-valued array of shape (N,)*dim.
        """
        v_hat = self.space.transform.forward(v)
        div_v_hat = self.scheme.apply_divergence(v_hat)
        div_v = self.space.transform.inverse(div_v_hat)
        return div_v.real
    
    @eqx.filter_jit
    def sym_grad(self, u: Array) -> Array:
        """Applies the symmetric gradient operator to the input real-valued array u.
        
        Args:
            u: A real-valued array of shape (N,)*dim.
        Returns:
            The symmetric gradient of u, a real-valued array of shape (N,)*dim + (dim, dim).
        """
        u_hat = self.space.transform.forward(u)
        sym_grad_u_hat = self.scheme.apply_symmetric_gradient(u_hat)
        sym_grad_u = self.space.transform.inverse(sym_grad_u_hat)
        return sym_grad_u.real
    

    @eqx.filter_jit
    def forward(self, u: Array) -> Array:
        """Applies the forward transform to the input real-valued array u.
        
        Args:
            u: A real-valued array of shape (N,)*dim.
        Returns:
            The transformed array u_hat, a complex-valued array of shape (N,)*dim.
        """
        return self.space.transform.forward(u)
    
    @eqx.filter_jit
    def inverse(self, u_hat: Array) -> Array:
        """Applies the inverse transform to the input complex-valued array u_hat.
        
        Args:
            u_hat: A complex-valued array of shape (N,)*dim.
        Returns:
            The inverse transformed array u, a real-valued array of shape (N,)*dim.
        """
        return self.space.transform.inverse(u_hat).real
    
    @eqx.filter_jit
    def ddot(self, A: Array, B: Array) -> Array:
        return self.tensor.ddot(A, B)
    
    @eqx.filter_jit
    def trace(self, A: Array) -> Array:
        return self.tensor.trace(A) 
    
    @eqx.filter_jit
    def trans(self, A: Array) -> Array:
        return self.tensor.trans(A)
    
    @eqx.filter_jit
    def dyad(self, A: Array, B: Array) -> Array:
        return self.tensor.dyad(A, B)   
    
    @eqx.filter_jit
    def dot(self, A: Array, B: Array) -> Array:
        return self.tensor.dot(A, B)