"""
Projection operators for the spectral methods.
"""

from abc import ABC, abstractmethod

import jax
import jax.numpy as jnp
from jax import Array

from xpektra.space import SpectralSpace


class ProjectionOperator(ABC):
    """Abstract base for projection operators.

    Subclasses implement a specific projection strategy (Galerkin, Moulinec-Suquet, etc.)
    and are passed to :class:`SpectralOperator` which delegates ``project()`` calls,
    supplying ``gradient_operator`` and ``space`` at call time so that the scheme data
    lives in a single place.

    When a projection is attached to a :class:`SpectralOperator`, its
    :meth:`build` method is called with the spectral space.  Subclasses that
    need precomputation (e.g. materialising a Green's operator) should override
    :meth:`build` and return a new instance with the precomputed data.
    """

    @abstractmethod
    def project(
        self, field_hat: Array, gradient_operator: Array, space: SpectralSpace
    ) -> Array:
        """Project a Fourier-space field.

        Args:
            field_hat: Input field in Fourier space.
            gradient_operator: The gradient operator array from the scheme.
            space: The spectral space (for wavenumber mesh access).

        Returns:
            The projected field in Fourier space.
        """
        raise NotImplementedError

    def build(self, space: SpectralSpace) -> "ProjectionOperator":
        """Precompute any data that depends on the spectral space.

        Called by :class:`SpectralOperator` when the projection is attached.
        The default implementation returns ``self`` unchanged (suitable for
        stateless projections like :class:`GalerkinProjection`).

        Subclasses that need precomputation should override this method and
        return a **new instance** with the precomputed data populated.
        """
        return self


@jax.tree_util.register_pytree_node_class
class GalerkinProjection(ProjectionOperator):
    """
    Material-independent Galerkin projection (matrix-free).

    This implementation does not materialize the full 4th-order Ghat tensor.
    It receives the gradient operator at call time from the owning
    :class:`SpectralOperator`, avoiding duplication of the large array.

    Example:
        >>> op = SpectralOperator(scheme=scheme, space=space, projection=GalerkinProjection())
        >>> projected = op.project(field_hat)
    """

    def tree_flatten(self):
        return (), None

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls()

    @jax.jit
    def project(
        self, field_hat: Array, gradient_operator: Array, space: SpectralSpace
    ) -> Array:
        """
        Applies the Galerkin projection on the fly.

        Computes: eps_hat = [δ_im * Dξ_j * Dξ_inv_l] * sigma_hat_lm

        Args:
            field_hat: The input field in Fourier space, shape (..., dim, dim).
            gradient_operator: The gradient operator from the scheme.
            space: Unused (accepted for interface compatibility).
        Returns:
            The projected field in Fourier space, shape (..., dim, dim).
        """
        Dξs = gradient_operator

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


@jax.tree_util.register_pytree_node_class
class MoulinecSuquetProjection(ProjectionOperator):
    """
    Moulinec-Suquet (MS) Green's operator for isotropic reference materials.

    Construction takes only the reference material parameters.  The operator
    tensor is precomputed automatically when the projection is attached to a
    :class:`SpectralOperator` (via :meth:`build`).

    To change the reference material, use ``dataclasses.replace`` to swap the
    projection on an existing :class:`SpectralOperator`::

        from dataclasses import replace

        new_op = replace(op, projection=MoulinecSuquetProjection(lambda0=new_lam, mu0=new_mu))

    Args:
        lambda0: First Lamé parameter of the reference material.
        mu0: Shear modulus of the reference material.

    Example:
        >>> proj = MoulinecSuquetProjection(lambda0=10.0, mu0=1.0)
        >>> op = SpectralOperator(scheme=scheme, space=space, projection=proj)
        >>> projected = op.project(field_hat)
    """

    lambda0: Array
    mu0: Array
    _operator: Array | None

    def __init__(self, lambda0: Array, mu0: Array) -> None:
        self.lambda0 = jnp.asarray(lambda0)
        self.mu0 = jnp.asarray(mu0)
        self._operator = None

    def build(self, space: SpectralSpace) -> "MoulinecSuquetProjection":
        """Precompute the Green's operator tensor from the spectral space.

        Returns a new instance with the precomputed operator populated.
        Called automatically by :class:`SpectralOperator.__post_init__`.
        """
        obj = object.__new__(MoulinecSuquetProjection)
        obj.lambda0 = self.lambda0
        obj.mu0 = self.mu0
        obj._operator = self._build_operator(space)
        return obj

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
    def project(
        self, field_hat: Array, gradient_operator: Array, space: SpectralSpace
    ) -> Array:
        """Apply the Moulinec-Suquet projection.

        Args:
            field_hat: Input field in Fourier space, shape (..., dim, dim).
            gradient_operator: Unused (accepted for interface compatibility).
            space: Unused (operator is precomputed via :meth:`build`).

        Returns:
            The projected field in Fourier space, shape (..., dim, dim).
        """
        return jnp.einsum("...khij,...ij->...kh", self._operator, field_hat)

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

        const = (self.lambda0 + self.mu0) / (
            self.mu0 * (self.lambda0 + 2.0 * self.mu0)
        )
        q4 = jnp.einsum("...k,...h,...i,...j->...khij", q, q, q, q)
        T2 = const * q4 / (q_dot_q_safe**2)[..., None, None, None]

        Ghat = T1 - T2
        Ghat = jnp.where(q_dot_q[..., None, None, None] == 0, 0.0, Ghat)

        return Ghat
