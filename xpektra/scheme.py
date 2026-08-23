# Copyright (C) 2025 ETH Zurich (SMEC)
#
# This file is part of xpektra.
#
# xpektra is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# xpektra is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with xpektra.  If not, see <https://www.gnu.org/licenses/>.

from abc import ABC, abstractmethod

import jax
import jax.numpy as jnp
import sympy as sp
from jax import Array

from xpektra.linalg import contract
from xpektra.space import SpectralSpace
from xpektra.transform import FFTTransform

__all__ = [
    "BackwardScheme",
    "CentralScheme",
    "ForwardScheme",
    "FourierScheme",
    "Hex1RScheme",
    "Quad1RScheme",
    "Tetra2Scheme",
]

iota = 1j  # Imaginary unit

# einsum subscripts for the tensor axes of a field whose gradient is taken.
# Must avoid "q" (quadrature) and "i" (the derivative index).
_TENSOR_SUBSCRIPTS = "jklmn"


class Scheme(ABC):
    """
    Abstract base class for a complete discretization strategy.

    A Scheme is a self-contained object responsible for generating the
    discrete gradient operator based on a given spectral space, and for
    applying that operator (and its adjoint) to fields in Fourier space.

    Subclasses supply only :meth:`compute_gradient_operator` -- either from
    finite-difference stencils (:class:`FiniteDifferenceScheme`) or from a
    closed-form symbol (:class:`FourierScheme`).  All four ``apply_*``
    operations live here, so every scheme shares one convention:
    ``divergence_operator = -conj(gradient_operator)``, which is what makes
    ``div`` the adjoint of ``sym_grad`` and hence ``D^T C D`` symmetric.

    ``n_quads`` is the number of derivation supports (quadrature points) per
    voxel; multi-support schemes such as :class:`Tetra2Scheme` derive it from
    their stencils.  ``gradient_operator`` is always ``(*spatial, n_quads, dim)``
    -- the quadrature axis is present even when ``n_quads == 1`` -- so centre
    fields have the same shape whichever scheme is in use, and none of the
    operations below needs a branch.
    """

    n_quads: int = 1

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

        self.n_quads = self._n_supports()

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
        aux_data = {"dim": self.dim, "space": self.space, "n_quads": self.n_quads}
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        obj = object.__new__(cls)
        object.__setattr__(obj, "gradient_operator", children[0])
        object.__setattr__(obj, "dim", aux_data["dim"])
        object.__setattr__(obj, "space", aux_data["space"])
        object.__setattr__(obj, "n_quads", aux_data["n_quads"])
        object.__setattr__(obj, "_initialized", True)
        return obj

    def _n_supports(self) -> int:
        """Number of derivation supports (quadrature points) per voxel."""
        return 1

    def is_compatible(self):
        """
        Checks if the scheme is compatible with the given transform.
        """
        if not isinstance(self.space.transform, FFTTransform):
            raise ValueError(  # noqa: TRY004
                f"{type(self).__name__} is only compatible with FFTTransform."
            )

    @abstractmethod
    def compute_gradient_operator(self, wavenumbers_mesh: list[Array]) -> Array:
        """
        The primary output of any scheme. The gradient operator field has shape ( (N,)*dim, (dim,)*rank).
        """
        raise NotImplementedError

    @property
    def divergence_operator(self):
        """Returns the divergence operator in Fourier space."""
        return -jnp.conj(self.gradient_operator)

    @jax.jit
    def apply_symmetric_gradient(self, u_hat: Array) -> Array:
        """
        Applies the symmetric gradient operator on the fly.
        Computes: eps_hat_qij = 0.5 * (D_qi * u_hat_j + D_qj * u_hat_i)

        Args:
            u_hat: Node field in Fourier space, shape ``(*spatial, dim)``.

        Returns:
            Centre field in Fourier space, shape ``(*spatial, n_quads, dim, dim)``.
        """
        t = jnp.einsum("...qi,...j->...qij", self.gradient_operator, u_hat)
        return 0.5 * (t + jnp.swapaxes(t, -1, -2))

    @jax.jit
    def apply_divergence(self, u_hat: Array) -> Array:
        """
        Applies the quadrature-averaged divergence operator on the fly.
        Computes: div_hat_i = 1/n_q * sum_q -conj(D_qj) * u_hat_qji

        Uses ``divergence_operator``, not ``gradient_operator``: the input lives at
        the voxel centres, so the half-voxel phase is conjugated (Eq. 19₂ of
        Amouzou-adoun et al., 2026).  That is what makes ``div`` the adjoint of
        ``sym_grad``, and hence ``D^T C D`` symmetric.

        For a multi-support scheme ``divergence_operator[..., q, :]`` is the
        *mirror* support's symbol referenced to the nodes, so pairing each
        quadrature point with its own entry here already performs the cross
        derivation of Eq. (38) -- including the centre->node phase that
        contracting against the other support's ``D`` directly would omit.

        Args:
            u_hat: Centre field in Fourier space, shape
                ``(*spatial, n_quads, dim, dim)``.

        Returns:
            Node field in Fourier space, shape ``(*spatial, dim)``.
        """
        if u_hat.ndim < 3 or u_hat.shape[-3] != self.n_quads:
            raise ValueError(
                f"expected {self.n_quads} quadrature points on axis -3, "
                f"got shape {u_hat.shape}"
            )

        # Note: We must transpose sigma_hat for the ddot
        return (
            contract("...qj,...qji->...i", self.divergence_operator, u_hat)
            / self.n_quads
        )

    @jax.jit
    def apply_gradient(self, u_hat: Array) -> Array:
        """
        Applies the gradient operator on the fly, for a node field of any rank.
        Computes: grad_hat_qi... = D_qi * u_hat_...

        The derivative index comes **first**: for a vector field
        ``grad_qij = D_qi * u_j = d_i u_j``.  That matches
        :meth:`apply_symmetric_gradient`, which builds the same product before
        symmetrising, and :meth:`apply_divergence`, which contracts the operator
        against axis ``-2`` of its input.  So ``div(grad(u))`` is the vector
        Laplacian, and ``sym_grad(u) == 0.5 * (g + g^T)`` holds exactly.

        Note this is the transpose of the continuum-mechanics ``grad u``
        convention, in which ``(grad u)_ij = d_j u_i``; a deformation gradient is
        therefore ``I + swapaxes(grad(u), -1, -2)``.

        Args:
            u_hat: Node field in Fourier space, shape ``(*spatial, *tensor)``.
                Rank 0 (scalar) and rank 1 (vector) are the usual cases.

        Returns:
            Centre field in Fourier space, shape
            ``(*spatial, n_quads, dim, *tensor)``.
        """
        rank = u_hat.ndim - self.dim
        if rank < 0:
            raise ValueError(
                f"field has {u_hat.ndim} axes, fewer than the {self.dim} spatial "
                f"dimensions; got shape {u_hat.shape}"
            )
        if rank > len(_TENSOR_SUBSCRIPTS):
            raise ValueError(
                f"gradient of a rank-{rank} field is not supported "
                f"(maximum {len(_TENSOR_SUBSCRIPTS)})"
            )

        # rank 0 -> "...qi,...->...qi";  rank 1 -> "...qi,...j->...qij"
        subs = _TENSOR_SUBSCRIPTS[:rank]
        return jnp.einsum(
            f"...qi,...{subs}->...qi{subs}", self.gradient_operator, u_hat
        )

    @jax.jit
    def apply_laplacian(self, u_hat: Array) -> Array:
        """
        Applies the Laplacian operator on the fly.
        Computes: lap_hat = 1/n_q * sum_q sum_i D_qi * -conj(D_qi) = -1/n_q sum_q ||D_q||^2

        Real and negative semi-definite.  For a multi-support scheme this is the
        cross derivation of Eq. (31): as in :meth:`apply_divergence`, ``_Dd`` is
        the mirror support's symbol referenced to the nodes, so the centre->node
        phase is included.  Contracting the two supports' ``D`` against each other
        directly leaves a residual ``exp(i xi.h)``; pairing each support with
        *itself* without the conjugate is sign-indefinite over roughly half the
        spectrum.

        Node -> node: the quadrature axis is summed away, not grown, so this
        accepts a field of any tensor rank.
        """
        # -1/n_q sum_q ||D_q||^2
        lap_op_hat = (
            contract(
                "...qi,...qi->...", self.gradient_operator, self.divergence_operator
            )
            / self.n_quads
        )
        return (
            jnp.expand_dims(lap_op_hat, tuple(range(lap_op_hat.ndim, u_hat.ndim)))
            * u_hat
        )


def _unit_offset(axis: int, dim: int, offset: int) -> tuple[int, ...]:
    """Helper function to create a unit offset tuple for a given axis."""
    return tuple(offset if i == axis else 0 for i in range(dim))


class FiniteDifferenceScheme(Scheme):
    """
    Base class for schemes operating on a uniform Cartesian grid
    where the differentiation is not diagonal in Fourier space.

    Cannot be instantiated directly — use a concrete subclass
    (e.g. CentralScheme, ForwardScheme).
    """

    def _n_supports(self) -> int:
        return len(self.support_stencils)

    @property
    def stencils(self):
        raise NotImplementedError

    @property
    def support_stencils(self):
        return (self.stencils,)

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

    def build_support_operator(self, stencils: tuple, wavenumber_mesh: list[Array]):
        spacings = [
            self.space.lengths[i] / self.space.shape[i] for i in range(self.dim)
        ]
        Zs = [
            self.build_fourier_operator(stencil=s)[1](*wavenumber_mesh, *spacings)
            for s in stencils
        ]

        return jnp.stack(Zs, axis=-1)

    def compute_gradient_operator(self, wavenumbers_mesh: list[Array]):
        """Builds the full gradient operator field using the scheme's stencils.

        Args:
            wavenumbers_mesh: A list of arrays representing the meshgrid of wavenumbers.

        Returns:
            The gradient operator in Fourier space, shape
            ``(*spatial, n_quads, dim)``.  The quadrature axis sits *after* the
            spatial axes so that ``transform.forward``/``inverse``, which act on
            ``axes=range(dim)``, and the sharded transforms, whose
            ``PartitionSpec`` objects are indexed from the left, both keep working
            unchanged.  It is always present, length 1 for a single-support
            scheme, so that every operation is a single branch-free einsum.
        """
        ops = [
            self.build_support_operator(stencils=s, wavenumber_mesh=wavenumbers_mesh)
            for s in self.support_stencils
        ]
        return jnp.stack(ops, axis=-2)


class ForwardScheme(FiniteDifferenceScheme):
    """Represents a forward difference scheme in Fourier space."""

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


def tetra_t1_stencils() -> list[list]:
    """Derivative stencils on tetrahedron T1 (Eqs. 20-22, Amouzou-adoun et al., 2026).

    T1 spans the four *even-parity* vertices of the voxel:
    ``(0,0,0), (1,1,0), (0,1,1), (1,0,1)``.
    """
    h1, h2, h3 = sp.symbols("h_1 h_2 h_3", real=True)
    dx_stencil = [
        ((1, 1, 0), 1 / (2 * h1)),
        ((0, 1, 1), -1 / (2 * h1)),
        ((1, 0, 1), 1 / (2 * h1)),
        ((0, 0, 0), -1 / (2 * h1)),
    ]
    dy_stencil = [
        ((1, 1, 0), 1 / (2 * h2)),
        ((0, 0, 0), -1 / (2 * h2)),
        ((0, 1, 1), 1 / (2 * h2)),
        ((1, 0, 1), -1 / (2 * h2)),
    ]
    dz_stencil = [
        ((0, 1, 1), 1 / (2 * h3)),
        ((0, 0, 0), -1 / (2 * h3)),
        ((1, 0, 1), 1 / (2 * h3)),
        ((1, 1, 0), -1 / (2 * h3)),
    ]
    return [dx_stencil, dy_stencil, dz_stencil]


def tetra_t2_stencils() -> list[list]:
    """Derivative stencils on tetrahedron T2 (Eqs. 20-22, Amouzou-adoun et al., 2026).

    T2 spans the four *odd-parity* vertices of the voxel:
    ``(1,0,0), (0,1,0), (0,0,1), (1,1,1)``.  It is the mirror of T1 through the
    voxel faces, which is why its Fourier symbol is ``-conj(Z_T1)`` when both are
    referenced to the voxel centre (Eq. 29).
    """
    h1, h2, h3 = sp.symbols("h_1 h_2 h_3", real=True)
    dx_stencil = [
        ((1, 0, 0), 1 / (2 * h1)),
        ((0, 0, 1), -1 / (2 * h1)),
        ((1, 1, 1), 1 / (2 * h1)),
        ((0, 1, 0), -1 / (2 * h1)),
    ]
    dy_stencil = [
        ((0, 1, 0), 1 / (2 * h2)),
        ((1, 0, 0), -1 / (2 * h2)),
        ((1, 1, 1), 1 / (2 * h2)),
        ((0, 0, 1), -1 / (2 * h2)),
    ]
    dz_stencil = [
        ((0, 0, 1), 1 / (2 * h3)),
        ((0, 1, 0), -1 / (2 * h3)),
        ((1, 1, 1), 1 / (2 * h3)),
        ((1, 0, 0), -1 / (2 * h3)),
    ]
    return [dx_stencil, dy_stencil, dz_stencil]


class Tetra2Scheme(FiniteDifferenceScheme):
    """Double-tetrahedron scheme (TETRA2), Finel (2025); Amouzou-adoun et al. (2026).

    Two derivation supports per voxel, so ``gradient_operator`` has shape
    ``(*spatial, 2, 3)`` and strain/stress fields carry two values per voxel
    (``n_quads = 2``; §2.7.1).  The displacement stays single-valued.

    Shapes are therefore *not* symmetric between gradient and divergence:
    ``apply_gradient``/``apply_symmetric_gradient`` map node -> 2x centre, while
    ``apply_divergence`` maps 2x centre -> node.  ``apply_laplacian`` is node ->
    node and does not grow the axis, because the two supports combine there.

    All four operations are inherited unchanged from :class:`Scheme`: the
    quadrature averaging of Eqs. (31) and (38) is already implied by
    ``divergence_operator = -conj(gradient_operator)`` holding *per support*, so
    this class only has to declare its stencils.
    """

    def is_compatible(self):
        if self.dim != 3:
            raise ValueError("Tetra2 scheme is only compatible with 3D space.")

        super().is_compatible()

    @property
    def support_stencils(self):
        return (tetra_t1_stencils(), tetra_t2_stencils())


class FourierScheme(Scheme):
    """Exact spectral derivative, ``D_i = i xi_i``.

    The only scheme whose symbol is not a finite stencil, so it builds
    ``gradient_operator`` directly rather than through
    :class:`FiniteDifferenceScheme`.  It is also the one case where the
    adjoint convention costs nothing: ``-conj(i xi) = i xi``, so the shared
    ``divergence_operator`` reduces to the gradient symbol itself and the
    inherited ``apply_*`` reproduce the classical spectral operators exactly.
    """

    def compute_gradient_operator(self, wavenumbers_mesh: list[Array]) -> Array:
        """Builds the spectral gradient operator, shape ``(*spatial, 1, dim)``.

        The length-1 quadrature axis is inserted for the same reason as in
        :meth:`FiniteDifferenceScheme.compute_gradient_operator`: it keeps the
        inherited ``apply_*`` branch-free.
        """
        Ds = jnp.stack([iota * xi for xi in wavenumbers_mesh], axis=-1)
        return Ds[..., None, :]
