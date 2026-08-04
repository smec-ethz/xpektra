from abc import ABC, abstractmethod

import jax
import jax.numpy as jnp
import sympy as sp
from jax import Array
from numpy import gradient

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
    def is_compatible(self):
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


def _unit_offset(axis: int, dim: int, offset: int) -> tuple[int, ...]:
    """Helper function to create a unit offset tuple for a given axis."""
    return tuple(offset if i == axis else 0 for i in range(dim))


class FiniteDifferenceScheme(Scheme):
    """
    Base class for schemes operating on a uniform Cartesian grid
    where the differentiation is not diagonal in Fourier space.

    Cannot be instantiated directly — use a concrete subclass
    (e.g. CentralDifference, ForwardDifference).
    """

    dim: int
    space: SpectralSpace
    gradient_operator: Array

    def __init__(self, space: SpectralSpace):
        self.space = space
        self.dim = len(space.lengths)

        # check compatibility of the scheme
        self.is_compatible()

        self.gradient_operator = self.compute_gradient_operator(
            wavenumbers_mesh=space.get_wavenumber_mesh()
        )

        object.__setattr__(self, "_initialized", True)

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

    def __init_subclass__(cls) -> None:
        """Automatically register all subclasses as PyTrees."""
        jax.tree_util.register_pytree_node_class(cls)

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

    def is_compatible(self):
        if not isinstance(self.space.transform, FFTTransform):
            raise ValueError(  # noqa: TRY004
                "FiniteDifferenceScheme is only compatible with FFTTransform."
            )

    @property
    def stencils(self):
        raise NotImplementedError

    def build_fourier_operator(self, stencil: list, modules: str = "jax"):
        """
        Factory method to create a finite difference scheme from a given stencil.

        The symbol is returned in exponential form, ``Z(k) = sum_a w_a exp(i k.a h)``.
        That is what the stencil literally says, it needs no symbolic
        simplification to build, and its conjugate -- required for the adjoint
        (divergence) operator -- is a term-by-term sign flip.

        Args:
            stencil: The finite difference stencil, as ``(offset, weight)`` pairs.
            modules: Backend for ``lambdify``.  The default ``"jax"`` keeps the
                returned callable traceable under ``jit``/``grad``; ``"numpy"``
                does not.

        Returns:
            A callable that computes the Fourier representation of the finite difference operator.
        """

        I = sp.I  # Imaginary unit

        # define symbolic wavevectors
        k_syms = sp.symbols(f"K_1:{self.dim + 1}", real=True)
        h_syms = sp.symbols(f"h_1:{self.dim + 1}", real=True)

        Z_symbolic = 0
        for offset, weight in stencil:
            if len(offset) != self.dim:
                raise ValueError(
                    f"Stencil offset {offset} does not match the specified dimension {self.dim}."
                )
            phase = sum(
                sp.Rational(offset[i]) * k_syms[i] * h_syms[i] for i in range(self.dim)
            )
            Z_symbolic += weight * sp.exp(I * phase)

        # Convert symbolic expression to a numerical function
        args = k_syms + h_syms
        Z_func = sp.lambdify(args, Z_symbolic, modules, cse=True)

        return Z_symbolic, Z_func

    def compute_gradient_operator(self, wavenumbers_mesh: list[Array]):
        """Builds the full gradient operator field using the scheme's stencils.

        Args:
            wavenumbers_mesh: A list of arrays representing the meshgrid of wavenumbers.

        Returns:
            An array representing the gradient operator in Fourier space, with shape ( (N,)*dim, (dim,)*rank).
        """
        spacings = [
            self.space.lengths[i] / self.space.shape[i] for i in range(self.dim)
        ]

        Zs = []
        for stencil in self.stencils:
            _, Z_func = self.build_fourier_operator(stencil=stencil)
            Z = Z_func(*wavenumbers_mesh, *spacings)
            Zs.append(Z)
        return Zs[0] if self.dim == 1 else jnp.stack(Zs, axis=-1)

    @jax.jit
    def apply_symmetric_gradient(self, u_hat: Array) -> Array:
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
    def apply_divergence(self, u_hat: Array) -> Array:
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
    def apply_gradient(self, u_hat: Array) -> Array:
        """
        Applies the gradient operator on the fly.
        Computes: grad_hat_ij = Dξ_i * u_hat_j
        """
        Dξs = self.gradient_operator
        if self.dim == 1:
            return Dξs * u_hat

        return Dξs * u_hat[..., None]

    @jax.jit
    def apply_laplacian(self, u_hat: Array) -> Array:
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


class ForwardScheme(FiniteDifferenceScheme):
    """Represents a forward difference scheme in Fourier space."""

    def is_compatible(self):
        super().is_compatible()

    @property
    def stencils(self):
        h_syms = sp.symbols(f"h_1:{self.dim + 1}", real=True)
        stencils = []
        for i in range(self.dim):
            stencil = [
                (_unit_offset(i, self.dim, 1), 1 / h_syms[i]),
                ((0,) * self.dim, -1 / h_syms[i]),
            ]
            stencils.append(stencil)
        return stencils


class BackwardScheme(FiniteDifferenceScheme):
    """Represents a backward difference scheme in Fourier space."""

    def is_compatible(self):
        super().is_compatible()

    @property
    def stencils(self):
        h_syms = sp.symbols(f"h_1:{self.dim + 1}", real=True)
        stencils = []
        for i in range(self.dim):
            stencil = [
                ((0,) * self.dim, 1 / h_syms[i]),
                (_unit_offset(i, self.dim, -1), -1 / h_syms[i]),
            ]
            stencils.append(stencil)
        return stencils


class CentralScheme(FiniteDifferenceScheme):
    """Represents a central difference scheme in Fourier space."""

    def is_compatible(self):
        super().is_compatible()

    @property
    def stencils(self):
        h_syms = sp.symbols(f"h_1:{self.dim + 1}", real=True)
        stencils = []
        for i in range(self.dim):
            stencil = [
                (_unit_offset(i, self.dim, -1), -1 / (2 * h_syms[i])),
                (_unit_offset(i, self.dim, 1), 1 / (2 * h_syms[i])),
            ]
            stencils.append(stencil)
        return stencils


class Quad1RScheme(FiniteDifferenceScheme):
    """Represents a 1st-order quadrature scheme in Fourier space using Willot's method."""

    def is_compatible(self):
        if self.dim != 2:
            raise ValueError("Quad1R scheme is only compatible with 2D space.")

        super().is_compatible()

    @property
    def stencils(self):
        h1, h2 = sp.symbols("h_1 h_2", real=True)

        dx_stencil = [
            ((1, 1), 1 / (2 * h1)),
            ((0, 1), -1 / (2 * h1)),
            ((1, 0), 1 / (2 * h1)),
            ((0, 0), -1 / (2 * h1)),
        ]

        dy_stencil = [
            ((1, 1), 1 / (2 * h2)),
            ((1, 0), -1 / (2 * h2)),
            ((0, 1), 1 / (2 * h2)),
            ((0, 0), -1 / (2 * h2)),
        ]

        stencils = [dx_stencil, dy_stencil]
        return stencils


class Hex1RScheme(FiniteDifferenceScheme):
    """Represents a 1st-order hexagonal scheme in Fourier space using Willot's method."""

    def is_compatible(self):
        if self.dim != 3:
            raise ValueError("Hex1R scheme is only compatible with 3D space.")

        super().is_compatible()

    @property
    def stencils(self):
        h1, h2, h3 = sp.symbols("h_1 h_2 h_3", real=True)
        dx_stencil = [
            ((1, 0, 0), 1 / (4 * h1)),
            ((0, 0, 0), -1 / (4 * h1)),
            ((1, 1, 0), 1 / (4 * h1)),
            ((0, 1, 0), -1 / (4 * h1)),
            ((1, 0, 1), 1 / (4 * h1)),
            ((0, 0, 1), -1 / (4 * h1)),
            ((1, 1, 1), 1 / (4 * h1)),
            ((0, 1, 1), -1 / (4 * h1)),
        ]

        dy_stencil = [
            ((0, 1, 0), 1 / (4 * h2)),
            ((0, 0, 0), -1 / (4 * h2)),
            ((1, 1, 0), 1 / (4 * h2)),
            ((1, 0, 0), -1 / (4 * h2)),
            ((0, 1, 1), 1 / (4 * h2)),
            ((0, 0, 1), -1 / (4 * h2)),
            ((1, 1, 1), 1 / (4 * h2)),
            ((1, 0, 1), -1 / (4 * h2)),
        ]

        dz_stencil = [
            ((0, 0, 1), 1 / (4 * h3)),
            ((0, 0, 0), -1 / (4 * h3)),
            ((1, 0, 1), 1 / (4 * h3)),
            ((1, 0, 0), -1 / (4 * h3)),
            ((0, 1, 1), 1 / (4 * h3)),
            ((0, 1, 0), -1 / (4 * h3)),
            ((1, 1, 1), 1 / (4 * h3)),
            ((1, 1, 0), -1 / (4 * h3)),
        ]

        return [dx_stencil, dy_stencil, dz_stencil]


class DiagonalScheme(Scheme, ABC):
    """
    Base class for schemes operating on a uniform Cartesian grid
    where the differentiation is diagonal in Fourier space.

    Cannot be instantiated directly — use a concrete subclass
    (e.g. FourierScheme, CentralDifference).
    """

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

    def __init_subclass__(cls) -> None:
        """Automatically register all subclasses as PyTrees."""
        jax.tree_util.register_pytree_node_class(cls)

    def __init__(self, space: SpectralSpace):
        self.space = space
        self.dim = len(self.space.lengths)
        self.is_compatible()
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

    def is_compatible(self):
        if not isinstance(self.space.transform, FFTTransform):
            raise ValueError(  # noqa: TRY004
                "The provided scheme is not compatible with the spectral space's transform."
            )

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


class FourierScheme(DiagonalScheme):
    """
    Class implementing the standard spectral 'Fourier' derivative.
    """

    def formula(self, xi, dx, iota, factor):
        return iota * xi


class CentralDifference(DiagonalScheme):
    """Implements the standard central difference scheme."""

    def formula(self, xi, dx, iota, factor):
        return iota * jnp.sin(xi * dx) / dx


class ForwardDifference(DiagonalScheme):
    """Implements the forward difference scheme."""

    def formula(self, xi, dx, iota, factor):
        return (jnp.exp(iota * xi * dx) - 1) / dx


class BackwardDifference(DiagonalScheme):
    """Implements the backward difference scheme."""

    def formula(self, xi, dx, iota, factor):
        return (1 - jnp.exp(-iota * xi * dx)) / dx


class RotatedDifference(DiagonalScheme):
    """Implements the rotated finite difference scheme (Willot/HEX8R)."""

    def formula(self, xi, dx, iota, factor):
        if self.dim == 1:
            raise RuntimeError("Rotated difference is not defined for 1D")
        return 2 * iota * jnp.tan(xi * dx / 2) * factor / dx


class FourthOrderCentralDifference(DiagonalScheme):
    """Implements the fourth order difference scheme."""

    def formula(self, xi, dx, iota, factor):
        return iota * (
            8 * jnp.sin(xi * dx) / (6 * dx) - jnp.sin(2 * xi * dx) / (6 * dx)
        )


class SixthOrderCentralDifference(DiagonalScheme):
    """Implements the sixth order difference scheme."""

    def formula(self, xi, dx, iota, factor):
        return iota * (
            9 * jnp.sin(xi * dx) / (6 * dx)
            - 3 * jnp.sin(2 * xi * dx) / (10 * dx)
            + jnp.sin(3 * xi * dx) / (30 * dx)
        )


class EighthOrderCentralDifference(DiagonalScheme):
    """Implements the eighth order difference scheme."""

    def formula(self, xi, dx, iota, factor):
        return iota * (
            8 * jnp.sin(xi * dx) / (5 * dx)
            - 2 * jnp.sin(2 * xi * dx) / (5 * dx)
            + 8 * jnp.sin(3 * xi * dx) / (105 * dx)
            - jnp.sin(4 * xi * dx) / (140 * dx)
        )
