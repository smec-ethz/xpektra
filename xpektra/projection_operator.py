import jax
import jax.numpy as jnp
from jax import Array
import equinox as eqx
from abc import abstractmethod
import numpy as np
import itertools
from itertools import repeat

from xpektra.scheme import CartesianScheme
from xpektra import TensorOperator
from xpektra.scheme import SpectralSpace


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


'''
    def optimized_projection_fill(
        self,
        G: np.ndarray, Dξs: np.ndarray, grid_size: tuple[int, ...]
    ) -> np.ndarray:
        """
        ONLY ONE LINE IS CHANGED HERE.
        The assignment indexing is updated to match the new (spatial..., tensor...) layout.
        """
        ndim = len(grid_size)
        shape = grid_size
        N = np.prod(shape)
    
        # Flatten Dξs into shape (N, ndim)
        Dξs = Dξs.reshape(N, ndim)
        norm_sq = np.einsum("ni,ni->n", Dξs, np.conj(Dξs))  # shape (N,)
    
        # Avoid division by zero
        valid_mask = norm_sq != 0
        Dξ_inv = np.zeros_like(Dξs, dtype=np.complex128)
        Dξ_inv[valid_mask] = np.conj(Dξs[valid_mask]) / norm_sq[valid_mask, None]
    
        # Precompute grid indices
        grid_indices = list(itertools.product(*[range(n) for n in shape]))
    
        δ = lambda i, j: float(i == j)  # noqa: E731
    
        for i, j, l, m in itertools.product(range(ndim), repeat=4):
            if δ(i, m) == 0:
                continue  # skip computation entirely
            
            term = Dξs[:, j] * Dξ_inv[:, l]  # shape (N,)
            term[~valid_mask] = 0.0
    
            # Assign into G
            for index, ind in enumerate(grid_indices):
                # --- THIS IS THE ONLY LINE CHANGED IN THIS FUNCTION ---
                # Old indexing: G[i, j, l, m][ind]
                # New indexing: G[ind + (i, j, l, m)]
                # `ind` is a tuple like (x,y,z), so we concatenate it with the tensor indices.
                G[ind + (i, j, l, m)] = δ(i, m) * term[index]
    
        return G


    def compute_operator(self) -> Array:
        """
        ONLY ONE LINE IS CHANGED HERE.
        The shape of the initial `G` array is changed to the new layout.
        """
        ndim = self.scheme.space.dim
        grid_size = (self.scheme.space.size,) * ndim

        G = np.zeros(grid_size + (ndim, ndim, ndim, ndim), dtype="complex")

        Dξs = self.scheme.gradient_operator
        G = self.optimized_projection_fill(G, Dξs, grid_size)

        return G'''
