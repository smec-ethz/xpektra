from abc import abstractmethod
from typing import List
import numpy as np
import jax.numpy as jnp
from jax import Array
import equinox as eqx

from xpektra.space import SpectralSpace
from xpektra.transform import Transform


class Scheme(eqx.Module):
    """
    Abstract base class for a complete discretization strategy.

    A Scheme is a self-contained object responsible for generating the
    discrete gradient operator based on a given spectral space.
    """

    # space: eqx.AbstractVar[SpectralSpace]

    @abstractmethod
    def compute_gradient_operator(self) -> Array:
        """
        The primary output of any scheme. The gradient operator field has shape ( (N,)*dim, (dim,)*rank).
        """
        raise NotImplementedError

    @abstractmethod
    def create_wavenumber_mesh(self) -> List[Array]:
        """
        Creates a list of coordinate arrays for the wavenumbers.
        """
        raise NotImplementedError


class CartesianScheme(Scheme):
    """
    Base class for schemes operating on a uniform Cartesian grid.

    It handles the wavenumber_mesh generation. Subclasses only need to
    provide the mathematical formula for differentiation.
    """

    space: SpectralSpace
    gradient_operator: Array
    # wavenumbers_mesh: List[Array]

    def __init__(self, space: SpectralSpace):
        self.space = space
        self.gradient_operator = self.compute_gradient_operator()
        # self.wavenumbers_mesh = self.create_wavenumber_mesh()

    '''
    @property
    def symmetric_gradient_operator(self) -> Array:
        """The symmetric gradient operator field."""
        nabla = self.gradient_operator
        kronecker_delta = np.eye(self.space.dim, dtype=complex)

        nabla_sym = 0.5 * (
            jnp.einsum("...j,ik->...ijk", nabla, kronecker_delta, optimize="optimal")
            + jnp.einsum("...i,jk->...ijk", nabla, kronecker_delta, optimize="optimal")
        )
        return nabla_sym

    @property
    def divergence_operator(self) -> Array:
        """The divergence operator field."""
        nabla = self.gradient_operator
        kronecker_delta = jnp.eye(self.space.dim, dtype=complex)
        div = jnp.einsum("...k,ij->...ijk", nabla, kronecker_delta, optimize="optimal")
        return div
    '''

    @eqx.filter_jit
    def symm_grad(self, u_hat: Array) -> Array:
        """
        Applies the symmetric gradient operator on the fly.
        Computes: eps_hat_ij = 0.5 * (Dξ_i * u_hat_j + Dξ_j * u_hat_i)
        """
        Dξs = self.gradient_operator
        term1 = jnp.einsum('...i,...j->...ij', Dξs, u_hat) # D_i * u_j
        term2 = jnp.einsum('...j,...i->...ij', Dξs, u_hat) # D_j * u_i
        return 0.5 * (term1 + term2)

    @eqx.filter_jit
    def div(self, sigma_hat: Array) -> Array:
        """
        Applies the divergence operator on the fly.
        Computes: div_hat_i = Dξ_j * sigma_hat_ji
        """
        Dξs = self.gradient_operator
        # Note: We must transpose sigma_hat for the ddot
        return jnp.einsum('...j,...ji->...i', Dξs, sigma_hat)

    @abstractmethod
    def formula(self, xi: Array, dx: float, iota: complex, factor: float) -> Array:
        """
        Subclasses must implement their specific differentiation formula here.
        This defines the discrete operator D(ξ) for a single dimension.
        """
        raise NotImplementedError

    def create_wavenumber_mesh(self) -> List[Array]:
        """Creates a list of coordinate arrays for the wavenumbers."""
        ξ = self.space.wavenumber_vector()
        if self.space.dim == 1:
            return [ξ]
        else:
            return list(np.meshgrid(*([ξ] * self.space.dim), indexing="ij"))

    def compute_gradient_operator(self) -> Array:
        """Builds the full gradient operator field using the scheme's formula."""
        # This factor is needed for certain schemes like 'rotated_difference'

        wavenumbers_mesh = self.create_wavenumber_mesh()
        factor = 1.0
        Δ = self.space.length / self.space.size
        if self.space.dim > 1:
            # Note: A scheme's formula must handle this factor if it needs it.
            for j in range(self.space.dim):
                factor *= 0.5 * (1 + np.exp(self.space.iota * wavenumbers_mesh[j] * Δ))

        diff_vectors = []
        for i in range(self.space.dim):
            Dξ_i = self.formula(
                xi=wavenumbers_mesh[i],
                dx=self.space.length / self.space.size,
                iota=self.space.iota,
                factor=factor,
            )
            diff_vectors.append(Dξ_i)

        return np.stack(diff_vectors, axis=-1)


# --- Concrete Finite Difference Implementations ---


class Fourier(CartesianScheme):
    """
    Implements the standard spectral 'Fourier' derivative.
    """

    def formula(self, xi, dx, iota, factor):
        return iota * xi


class CentralDifference(CartesianScheme):
    """Implements the standard central difference scheme."""

    def formula(self, xi, dx, iota, factor):
        return iota * jnp.sin(xi * dx) / dx


class ForwardDifference(CartesianScheme):
    """Implements the forward difference scheme."""

    def formula(self, xi, dx, iota, factor):
        return (jnp.exp(iota * xi * dx) - 1) / dx


class BackwardDifference(CartesianScheme):
    """Implements the backward difference scheme."""

    def formula(self, xi, dx, iota, factor):
        return (1 - jnp.exp(-iota * xi * dx)) / dx


class RotatedDifference(CartesianScheme):
    """Implements the rotated finite difference scheme (Willot/HEX8R)."""

    def formula(self, xi, dx, iota, factor):
        if self.space.dim == 1:
            raise RuntimeError("Rotated difference is not defined for 1D")
        return 2 * iota * jnp.tan(xi * dx / 2) * factor / dx


class FourthOrderCentralDifference(CartesianScheme):
    """Implements the fourth order difference scheme."""

    def formula(self, xi, dx, iota, factor):
        return iota * (
            8 * jnp.sin(xi * dx) / (6 * dx) - jnp.sin(2 * xi * dx) / (6 * dx)
        )


class SixthOrderCentralDifference(CartesianScheme):
    """Implements the sixth order difference scheme."""

    def formula(self, xi, dx, iota, factor):
        return iota * (
            9 * jnp.sin(xi * dx) / (6 * dx)
            - 3 * jnp.sin(2 * xi * dx) / (10 * dx)
            + jnp.sin(3 * xi * dx) / (30 * dx)
        )


class EighthOrderCentralDifference(CartesianScheme):
    """Implements the eighth order difference scheme."""

    def formula(self, xi, dx, iota, factor):
        return iota * (
            8 * jnp.sin(xi * dx) / (5 * dx)
            - 2 * jnp.sin(2 * xi * dx) / (5 * dx)
            + 8 * jnp.sin(3 * xi * dx) / (105 * dx)
            - jnp.sin(4 * xi * dx) / (140 * dx)
        )
