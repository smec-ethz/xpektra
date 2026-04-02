from abc import ABC, abstractmethod

import jax
import jax.numpy as jnp
from jax import Array

from xpektra.space import SpectralSpace
from xpektra.transform import FFTTransform

iota = 1j  # Imaginary unit


class Scheme(ABC):
    """
    Abstract base class for a complete discretization strategy.

    A Scheme is a self-contained object responsible for generating the
    discrete gradient operator based on a given spectral space.
    """

    @abstractmethod
    def compute_gradient_operator(self, wavenumbers_mesh: list[Array]) -> Array:
        """
        The primary output of any scheme. The gradient operator field has shape ( (N,)*dim, (dim,)*rank).
        """
        raise NotImplementedError

    @abstractmethod
    def is_compatible(self, transform) -> bool:
        """
        Checks if the scheme is compatible with the given transform.
        """
        raise NotImplementedError

    @abstractmethod
    def apply_gradient(self, u_hat: Array) -> Array:
        """
        Applies the gradient operator on the fly.
        """
        raise NotImplementedError

    @abstractmethod
    def apply_divergence(self, u_hat: Array) -> Array:
        """
        Applies the gradient operator on the fly.
        """
        raise NotImplementedError

    @abstractmethod
    def apply_symmetric_gradient(self, u_hat: Array) -> Array:
        """
        Applies the symmetric gradient operator on the fly.
        """
        raise NotImplementedError

    @abstractmethod
    def apply_laplacian(self, u_hat: Array) -> Array:
        """
        Applies the Laplacian operator on the fly.
        """
        raise NotImplementedError


@jax.tree_util.register_pytree_node_class
class DiagonalScheme(Scheme):
    """
    Base class for schemes operating on a uniform Cartesian grid
    where the differentiation is diagonal in Fourier space.
    """

    dim: int
    space: SpectralSpace
    gradient_operator: Array

    def __setattr__(self, name, value):
        """Enforce immutability after initialization.

        Attribute assignment is only allowed during ``__init__`` (before
        ``_initialized`` is set).  Any attempt to mutate the instance
        afterwards raises ``AttributeError``, mirroring the guarantees
        previously provided by ``eqx.Module``.
        """
        if hasattr(self, "_initialized"):
            raise AttributeError(f"Cannot modify frozen {type(self).__name__}")
        object.__setattr__(self, name, value)

    def __init__(self, space: SpectralSpace):
        if not self.is_compatible(space.transform):
            raise ValueError(
                "The provided scheme is not compatible with the spectral space's transform."
            )

        self.space = space
        self.dim = len(self.space.lengths)
        wavenumbers_mesh = space.get_wavenumber_mesh()
        self.gradient_operator = self.compute_gradient_operator(
            wavenumbers_mesh=wavenumbers_mesh
        )
        object.__setattr__(self, "_initialized", True)

    def tree_flatten(self):
        children = [self.gradient_operator]
        aux_data = {"dim": self.dim, "space": self.space}
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        obj = object.__new__(cls)
        object.__setattr__(obj, "gradient_operator", children[0])
        object.__setattr__(obj, "dim", aux_data["dim"])
        object.__setattr__(obj, "space", aux_data["space"])
        object.__setattr__(obj, "_initialized", True)
        return obj

    def is_compatible(self, transform):
        return isinstance(transform, FFTTransform)

    @jax.jit
    def apply_symmetric_gradient(self, u_hat: Array) -> Array:  # pyright: ignore[reportIncompatibleMethodOverride]
        """
        Applies the symmetric gradient operator on the fly.
        Computes: eps_hat_ij = 0.5 * (Dξ_i * u_hat_j + Dξ_j * u_hat_i)
        """
        Dξs = self.gradient_operator
        if self.dim == 1:
            return Dξs * u_hat  # In 1D, symmetric gradient is just the gradient

        term1 = jnp.einsum("...i,...j->...ij", Dξs, u_hat)  # D_i * u_j
        term2 = jnp.einsum("...j,...i->...ij", Dξs, u_hat)  # D_j * u_i
        return 0.5 * (term1 + term2)

    @jax.jit
    def apply_divergence(self, u_hat: Array) -> Array:  # pyright: ignore[reportIncompatibleMethodOverride]
        """
        Applies the divergence operator on the fly.
        Computes: div_hat_i = Dξ_j * u_hat_ji
        """
        Dξs = self.gradient_operator
        if self.dim == 1:
            return Dξs * u_hat

        # Note: We must transpose sigma_hat for the ddot
        return jnp.einsum("...j,...ji->...i", Dξs, u_hat)

    @jax.jit
    def apply_gradient(self, u_hat: Array) -> Array:  # pyright: ignore[reportIncompatibleMethodOverride]
        """
        Applies the gradient operator on the fly.
        Computes: grad_hat_ij = Dξ_i * u_hat_j
        """
        Dξs = self.gradient_operator
        if self.dim == 1:
            return Dξs * u_hat

        return Dξs * u_hat[..., None]

    @jax.jit
    def apply_laplacian(self, u_hat: Array) -> Array:  # pyright: ignore[reportIncompatibleMethodOverride]
        """
        Applies the Laplacian operator on the fly.
        Computes: lap_hat = -|Dξ|^2 * u_hat
        """
        Dξs = self.gradient_operator
        if self.dim == 1:
            lap_op_hat = Dξs * Dξs  # |Dξ|^2
            return lap_op_hat * u_hat

        lap_op_hat = jnp.einsum("...i,...i->...", Dξs, Dξs)  # |Dξ|^2
        return lap_op_hat * u_hat

    def compute_gradient_operator(self, wavenumbers_mesh) -> Array:
        """Builds the full gradient operator field using the scheme's formula."""
        # This factor is needed for certain schemes like 'rotated_difference'

        factor = 1.0
        if self.dim > 1:
            # Note: A scheme's formula must handle this factor if it needs it.
            for j in range(self.dim):
                Δ = self.space.lengths[j] / self.space.shape[j]
                factor *= 0.5 * (1 + jnp.exp(iota * wavenumbers_mesh[j] * Δ))

        diff_vectors = []
        for i in range(self.dim):
            Dξ_i = self.formula(
                xi=wavenumbers_mesh[i],
                dx=self.space.lengths[i] / self.space.shape[i],
                iota=iota,
                factor=factor,
            )
            diff_vectors.append(Dξ_i)

        if self.dim == 1:
            return diff_vectors[0]
        else:
            return jnp.stack(diff_vectors, axis=-1)

    @abstractmethod
    def formula(self, xi, dx, iota, factor):
        """
        The core formula for the discrete derivative in Fourier space.
        Must be implemented by concrete schemes.
        """
        raise NotImplementedError


@jax.tree_util.register_pytree_node_class
class FourierScheme(DiagonalScheme):
    """
    Class implementing the standard spectral 'Fourier' derivative.
    """

    def formula(self, xi, dx, iota, factor):
        return iota * xi


@jax.tree_util.register_pytree_node_class
class CentralDifference(DiagonalScheme):
    """Implements the standard central difference scheme."""

    def formula(self, xi, dx, iota, factor):
        return iota * jnp.sin(xi * dx) / dx


@jax.tree_util.register_pytree_node_class
class ForwardDifference(DiagonalScheme):
    """Implements the forward difference scheme."""

    def formula(self, xi, dx, iota, factor):
        return (jnp.exp(iota * xi * dx) - 1) / dx


@jax.tree_util.register_pytree_node_class
class BackwardDifference(DiagonalScheme):
    """Implements the backward difference scheme."""

    def formula(self, xi, dx, iota, factor):
        return (1 - jnp.exp(-iota * xi * dx)) / dx


@jax.tree_util.register_pytree_node_class
class RotatedDifference(DiagonalScheme):
    """Implements the rotated finite difference scheme (Willot/HEX8R)."""

    def formula(self, xi, dx, iota, factor):
        if self.dim == 1:
            raise RuntimeError("Rotated difference is not defined for 1D")
        return 2 * iota * jnp.tan(xi * dx / 2) * factor / dx


@jax.tree_util.register_pytree_node_class
class FourthOrderCentralDifference(DiagonalScheme):
    """Implements the fourth order difference scheme."""

    def formula(self, xi, dx, iota, factor):
        return iota * (
            8 * jnp.sin(xi * dx) / (6 * dx) - jnp.sin(2 * xi * dx) / (6 * dx)
        )


@jax.tree_util.register_pytree_node_class
class SixthOrderCentralDifference(DiagonalScheme):
    """Implements the sixth order difference scheme."""

    def formula(self, xi, dx, iota, factor):
        return iota * (
            9 * jnp.sin(xi * dx) / (6 * dx)
            - 3 * jnp.sin(2 * xi * dx) / (10 * dx)
            + jnp.sin(3 * xi * dx) / (30 * dx)
        )


@jax.tree_util.register_pytree_node_class
class EighthOrderCentralDifference(DiagonalScheme):
    """Implements the eighth order difference scheme."""

    def formula(self, xi, dx, iota, factor):
        return iota * (
            8 * jnp.sin(xi * dx) / (5 * dx)
            - 2 * jnp.sin(2 * xi * dx) / (5 * dx)
            + 8 * jnp.sin(3 * xi * dx) / (105 * dx)
            - jnp.sin(4 * xi * dx) / (140 * dx)
        )


'''
class MatrixScheme(Scheme):
    """Base for Chebyshev, Legendre, etc."""
    # Shape: (Nx, Nx, dim) assuming separable dimensions for simplicity
    # or (N_total, N_total) for full density.
    diff_matrices: Array
    space: SpectralSpace = eqx.field(static=True)


    def apply_gradient(self, u_hat: Array) -> Array:
        # Matrix multiplication logic
        # Example: Contraction along specific axes
        # This is pseudo-code, real implementation depends on tensor layout
        grad_x = jnp.einsum("ij, j... -> i...", self.diff_matrices[0], u_hat)
        grad_y = jnp.einsum("ij, ...j -> ...i", self.diff_matrices[1], u_hat)
        return jnp.stack([grad_x, grad_y], axis=-1)

    def apply_divergence(self, v_hat: Array) -> Array:
        # Matrix multiplication + Sum
        div_x = jnp.einsum("ij, j... -> i...", self.diff_matrices[0], v_hat[..., 0])
        div_y = jnp.einsum("ij, ...j -> ...i", self.diff_matrices[1], v_hat[..., 1])
        return div_x + div_y
'''
