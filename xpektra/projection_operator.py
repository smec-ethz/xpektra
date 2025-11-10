"""
Projection operators for the spectral methods.
"""

import jax.numpy as jnp
from jax import Array
import equinox as eqx
from abc import abstractmethod
import numpy as np

from xpektra.scheme import CartesianScheme
from xpektra import TensorOperator
from xpektra.scheme import SpectralSpace

class ProjectionOperator(eqx.Module):
    """
    An 'abstract' base class for operators that project fields.

    It uses a `Scheme` to construct a 4th-order tensor that
    enforces mechanical constraints in Fourier space.

    """

    @abstractmethod
    def compute_operator(self) -> Array:
        """Abstract method to compute the 4th-order operator tensor."""
        raise NotImplementedError


class GalerkinProjection(ProjectionOperator):
    """
    A subclass implementing the material-independent Galerkin projection.

    """

    scheme: CartesianScheme
    tensor_op: TensorOperator

    def compute_operator(self) -> Array:
        """Implements the Fourier Galerkin projection operator.
        
        ```python
        Ghat = GalerkinProjection(scheme, tensor_op).compute_operator()
        ```
        
        """
        Dξs = self.scheme.gradient_operator
        ndim = self.scheme.space.dim

        # Calculate the inverse of the gradient operator field
        norm_sq = jnp.sum(Dξs * jnp.conj(Dξs), axis=-1, keepdims=True)
        Dξ_inv = jnp.zeros_like(Dξs, dtype=jnp.complex128)

        # Avoid division by zero at the zero-frequency mode
        valid_mask = (norm_sq > 1e-12).squeeze()
        Dξ_inv = Dξ_inv.at[valid_mask].set(
            jnp.conj(Dξs[valid_mask]) / norm_sq[valid_mask]
        )

        # Construct the 4th-order tensor using einsum
        identity = jnp.eye(ndim)
        Ghat = jnp.einsum("im,...j,...l->...ijlm", identity, Dξs, Dξ_inv, optimize=True)
        return Ghat


class MoulinecSuquetProjection(ProjectionOperator):
    """
    A subclass of `ProjectionOperator` implementing the Moulinec-Suquet (MS) Green's operator,
    which depends on a homogeneous isotropic reference material.
    """

    space: SpectralSpace
    lambda0: Array
    mu0: Array

    def compute_operator(self) -> Array:
        """
        Implements the Moulinec-Suquet projection operator.

        ```python
        Ghat = MoulinecSuquetProjection(space, lambda0, mu0).compute_operator()
        ```
        """
        # Get grid properties from the scheme's space
        ndim = self.space.dim

        # Get the frequency grid 'q' (which is ξ), shape (N, N, N, 3)
        freq_vec = self.space.frequency_vector()
        meshes = np.meshgrid(*([freq_vec] * ndim), indexing="ij")
        q = jnp.stack(meshes, axis=-1)

        # Pre-calculate scalar terms, ensuring no division by zero
        q_dot_q = jnp.sum(q * q, axis=-1, keepdims=True)
        # Use jnp.where to safely handle the zero-frequency mode
        q_dot_q_safe = jnp.where(q_dot_q == 0, 1.0, q_dot_q)

        # Calculate the first term (T1) of the Green's operator
        # T1_num = (δ_ki ξ_h ξ_j + δ_hi ξ_k ξ_j + δ_kj ξ_h ξ_i + δ_hj ξ_k ξ_i)
        i = jnp.eye(ndim)
        t1_A = jnp.einsum("ki,...h,...j->...khij", i, q, q)
        t1_B = jnp.einsum("hi,...k,...j->...khij", i, q, q)
        t1_C = jnp.einsum("kj,...h,...i->...khij", i, q, q)
        t1_D = jnp.einsum("hj,...k,...i->...khij", i, q, q)

        T1_num = t1_A + t1_B + t1_C + t1_D
        T1 = T1_num / (4.0 * self.mu0 * q_dot_q_safe[..., None, None, None])

        # Calculate the second term (T2) of the Green's operator
        # T2 = const * (ξ_k ξ_h ξ_i ξ_j) / |ξ|^4
        const = (self.lambda0 + self.mu0) / (self.mu0 * (self.lambda0 + 2.0 * self.mu0))
        q4 = jnp.einsum("...k,...h,...i,...j->...khij", q, q, q, q)
        T2 = const * q4 / (q_dot_q_safe**2)[..., None, None, None]

        # Combine and set the zero-frequency mode to zero
        Ghat = T1 - T2
        Ghat = jnp.where(q_dot_q[..., None, None, None] == 0, 0.0, Ghat)

        return Ghat
