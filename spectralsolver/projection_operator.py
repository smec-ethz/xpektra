import jax.numpy as jnp
from jax import Array
import equinox as eqx
from abc import abstractmethod

from spectralsolver.scheme import CartesianScheme
from spectralsolver import TensorOperator

class ProjectionOperator(eqx.Module):
    """
    An 'abstract' base class for operators that project fields.
    
    It uses a `Scheme` to construct a 4th-order tensor (`Ghat`) that
    enforces mechanical constraints in Fourier space.
    """
    scheme: CartesianScheme
    tensor_op: TensorOperator

    def __init__(self, scheme: CartesianScheme, tensor_op: TensorOperator):
        self.scheme = scheme
        self.tensor_op = tensor_op


    @abstractmethod
    def compute_operator(self) -> Array:
        """Abstract method to compute the 4th-order operator tensor."""
        raise NotImplementedError


class GalerkinProjection(ProjectionOperator):
    """
    A 'final' class implementing the material-independent Galerkin projection.
    """
    def compute_operator(self) -> Array:
        """Implements the formula: Ghat_ijlm = δ_im * Dξ_j * Dξ_inv_l."""
        Dξs = self.scheme.gradient_operator
        ndim = self.scheme.space.dim

        # 1. Calculate the inverse of the gradient operator field
        norm_sq = jnp.sum(Dξs * jnp.conj(Dξs), axis=-1, keepdims=True)
        Dξ_inv = jnp.zeros_like(Dξs, dtype=jnp.complex128)
        
        # Avoid division by zero at the zero-frequency mode
        valid_mask = (norm_sq > 1e-12).squeeze()
        Dξ_inv = Dξ_inv.at[valid_mask].set(
            jnp.conj(Dξs[valid_mask]) / norm_sq[valid_mask]
        )

        # 2. Construct the 4th-order tensor using einsum
        identity = jnp.eye(ndim)
        Ghat = jnp.einsum(
            'im,...j,...l->...ijlm', 
            identity, 
            Dξs, 
            Dξ_inv, 
            optimize=True
        )
        return Ghat