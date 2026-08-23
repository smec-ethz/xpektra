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
from jax import Array

from xpektra.linalg import contract
from xpektra.scheme import Scheme
from xpektra.space import SpectralSpace

__all__ = [
    "GalerkinProjection",
    "MoulinecSuquetProjection",
    "ProjectionOperator",
]


class ProjectionOperator(ABC):
    """Abstract base for projection operators.

    Subclasses implement a specific projection strategy (Galerkin,
    Moulinec-Suquet, etc.) and are called directly on a Fourier-space field.
    """

    @abstractmethod
    def __call__(self, field_hat: Array) -> Array:
        """Project a Fourier-space field.

        Args:
            field_hat: Input field in Fourier space, shape
                ``(*spatial, n_quads, dim, dim)``.  The quadrature axis is
                always present, length 1 for a single-support scheme.

        Returns:
            The projected field in Fourier space, same shape.
        """
        raise NotImplementedError


@jax.tree_util.register_pytree_node_class
class GalerkinProjection(ProjectionOperator):
    """
    Material-independent Galerkin projection (matrix-free).

    Projects a field onto the space of discrete gradients, ``eps_q,ij = D_q,j u_i``
    for a single-valued ``u``.  The full 4th-order ``Ghat`` is never materialised;
    the scheme is held directly, so its gradient operator is not duplicated.

    The least-squares solution for ``u`` averages over the quadrature points --
    ``u`` is shared by all of them, which is precisely the coupling that keeps this
    off :class:`~xpektra.SpectralOperator`::

        norm_sq  = mean_q sum_i |D_q,i|^2
        temp_i   = mean_q conj(D_q,l) sigma_q,il / norm_sq
        eps_q,ij = D_q,j * temp_i

    For ``n_quads == 1`` this collapses to the classical single-support form.

    Args:
        scheme: The discretization scheme supplying the gradient operator.

    Example:
        >>> proj = GalerkinProjection(scheme=scheme)
        >>> projected = proj(field_hat)
    """

    def __init__(self, scheme: Scheme) -> None:
        self.scheme = scheme

    def tree_flatten(self):
        return (self.scheme,), None

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(scheme=children[0])

    @jax.jit
    def __call__(self, field_hat: Array) -> Array:
        """
        Applies the Galerkin projection on the fly.

        Args:
            field_hat: The input field in Fourier space, shape
                ``(*spatial, n_quads, dim, dim)``.

        Returns:
            The projected field in Fourier space, same shape.
        """
        scheme = self.scheme
        n_quads = scheme.n_quads

        Dqs = scheme.gradient_operator  # (*spatial, n_quads, dim)
        sigma = field_hat

        # norm_sq = mean_q ||D_q||^2, real by construction
        norm_sq = contract("...qi,...qi->...", Dqs, jnp.conj(Dqs)).real / n_quads

        # Create a safe denominator to avoid 0/0 at the null modes
        norm_sq_safe = jnp.where(norm_sq == 0, 1.0, norm_sq)

        # temp_i = mean_q conj(D_q,l) sigma_q,il / norm_sq
        temp_i = (
            contract("...ql,...qil->...i", jnp.conj(Dqs), sigma)
            / n_quads
            / norm_sq_safe[..., None]
        )
        temp_i = jnp.where(norm_sq[..., None] == 0, 0.0, temp_i)

        # eps_q,ij = D_q,j * temp_i
        eps_hat = jnp.einsum("...qj,...i->...qij", Dqs, temp_i)

        del temp_i

        return eps_hat


@jax.tree_util.register_pytree_node_class
class MoulinecSuquetProjection(ProjectionOperator):
    """
    Moulinec-Suquet (MS) Green's operator for isotropic reference materials.

    The operator tensor is precomputed at construction, so changing the reference
    material means building a new instance.

    Note:
        ``Ghat`` is derived for the exact Fourier symbol ``i*xi``.  Applying it to
        a multi-support scheme's field is supported mechanically -- the same
        ``Ghat`` is broadcast across the quadrature axis -- but it is *not* the
        consistent projector for that discretization; use
        :class:`GalerkinProjection` there.

    Args:
        lambda0: First Lamé parameter of the reference material.
        mu0: Shear modulus of the reference material.
        space: The spectral space, used to build the operator.

    Example:
        >>> proj = MoulinecSuquetProjection(lambda0=10.0, mu0=1.0, space=space)
        >>> projected = proj(field_hat)
    """

    lambda0: Array
    mu0: Array
    _operator: Array

    def __init__(self, lambda0: Array, mu0: Array, space: SpectralSpace) -> None:
        self.lambda0 = jnp.asarray(lambda0)
        self.mu0 = jnp.asarray(mu0)
        self._operator = self._build_operator(space)

    def tree_flatten(self):
        children = (self.lambda0, self.mu0, self._operator)
        return children, None

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        obj = object.__new__(cls)
        obj.lambda0 = children[0]
        obj.mu0 = children[1]
        obj._operator = children[2]
        return obj

    @jax.jit
    def __call__(self, field_hat: Array) -> Array:
        """Apply the Moulinec-Suquet projection.

        Args:
            field_hat: Input field in Fourier space, shape
                ``(*spatial, n_quads, dim, dim)``.

        Returns:
            The projected field in Fourier space, same shape.
        """
        # Give Ghat a length-1 quadrature axis so it broadcasts over the
        # supports; it is the same operator for every support.
        Ghat = self._operator[..., None, :, :, :, :]
        return contract("...khij,...ij->...kh", Ghat, field_hat)

    def _build_operator(self, space: SpectralSpace) -> Array:
        """Build the full Green's operator tensor.

        Returns:
            The Green's operator Ghat, shape (..., dim, dim, dim, dim).
        """
        ndim = len(space.shape)

        meshes = space.get_wavenumber_mesh()
        q = jnp.stack(meshes, axis=-1)

        q_dot_q = jnp.sum(q * q, axis=-1, keepdims=True)
        q_dot_q_safe = jnp.where(q_dot_q == 0, 1.0, q_dot_q)

        i = jnp.eye(ndim)
        t1_A = jnp.einsum("ki,...h,...j->...khij", i, q, q)
        t1_B = jnp.einsum("hi,...k,...j->...khij", i, q, q)
        t1_C = jnp.einsum("kj,...h,...i->...khij", i, q, q)
        t1_D = jnp.einsum("hj,...k,...i->...khij", i, q, q)

        T1_num = t1_A + t1_B + t1_C + t1_D
        T1 = T1_num / (4.0 * self.mu0 * q_dot_q_safe[..., None, None, None])

        const = (self.lambda0 + self.mu0) / (self.mu0 * (self.lambda0 + 2.0 * self.mu0))
        q4 = jnp.einsum("...k,...h,...i,...j->...khij", q, q, q, q)
        T2 = const * q4 / (q_dot_q_safe**2)[..., None, None, None]

        Ghat = T1 - T2
        Ghat = jnp.where(q_dot_q[..., None, None, None] == 0, 0.0, Ghat)

        return Ghat
