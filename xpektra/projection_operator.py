"""
Projection operators for the spectral methods.
"""

import jax.numpy as jnp
from jax import Array
import equinox as eqx
from abc import abstractmethod
import numpy as np

from xpektra.scheme import DiagonalScheme
from xpektra.space import SpectralSpace
#from xpektra.scheme import SpectralSpace


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


class GalerkinProjection(eqx.Module):
    """
    A 'final' class implementing the material-independent Galerkin projection.

    This implementation is 'matrix-free'. It does not materialize
    the full 4th-order Ghat tensor. Instead, it stores the
    gradient operator (Dξs) and its inverse (Dξ_inv) and computes
    the projection on the fly. This saves a massive amount of memory.

    Args:
        scheme: The differential scheme providing the gradient operator.
    
    Returns:
        The Galerkin projection operator.

    Example:
        >>> projection = GalerkinProjection(scheme)
        >>> eps_hat = projection.project(field_hat)
    
    """

    scheme: DiagonalScheme
    
    @eqx.filter_jit
    def project(self, field_hat: Array) -> Array:
        """
        Applies the projection on the fly. This is the core of the class.

        Computes: eps_hat = [δ_im * Dξ_j * Dξ_inv_l] * sigma_hat_lm

        Args:
            field_hat: The input field in Fourier space, shape (..., dim, dim).
        Returns:
            The projected field in Fourier space, shape (..., dim, dim).
        """
        Dξs = self.scheme.gradient_operator

        # Calculate and store the inverse
        norm_sq = jnp.einsum("...i,...i->...", Dξs, jnp.conj(Dξs)).real[..., None]

        # Create a safe denominator to avoid 0/0
        norm_sq_safe = jnp.where(norm_sq == 0, 1.0, norm_sq)

        # Dξ_inv_l = conj(Dξ_l) / ||Dξ||²
        Dξ_inv = jnp.conj(Dξs) / norm_sq_safe

        # Manually set the zero-frequency mode to 0.0
        Dξ_inv = jnp.where(
            norm_sq == 0, 0.0, Dξ_inv
        )  # Compute inner term: temp_i = Dξ_inv_l * sigma_hat_il
        temp_i = jnp.einsum("...l,...il->...i", Dξ_inv, field_hat)

        # Compute outer term: eps_hat_ij = Dξ_j * temp_i
        eps_hat = jnp.einsum("...j,...i->...ij", Dξs, temp_i)

        del temp_i

        return eps_hat


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
