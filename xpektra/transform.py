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

from abc import abstractmethod
from dataclasses import dataclass, field
from typing import ClassVar

import jax
import jax.numpy as jnp
from jax import Array
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P

__all__ = [
    "DSTTransform",
    "FFTTransform",
    "PencilFFTTransform",
    "SlabFFTTransform2D",
    "SlabFFTTransform3D",
]


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class Transform:
    """Abstract base class for all spectral transforms."""

    dim: int | None = field(metadata={"static": True}, default=None)
    device_mesh: jax.sharding.Mesh | None = field(
        metadata={"static": True}, default=None
    )

    @abstractmethod
    def forward(self, x: Array) -> Array:
        """Perform the forward transform (e.g., FFT, DCT)."""
        raise NotImplementedError

    @abstractmethod
    def inverse(self, x_hat: Array) -> Array:
        """Perform the inverse transform (e.g., iFFT, iDCT)."""
        raise NotImplementedError

    @abstractmethod
    def get_wavenumber_vector(self, size: int, length: float) -> Array:
        """Get the 1D vector of wavenumbers (e.g., ξ for FFT, k for DCT)."""
        raise NotImplementedError

    @abstractmethod
    def get_wavenumber_mesh(
        self, shape: tuple[int, ...], lengths: tuple[float, ...]
    ) -> list[Array]:
        """Get the wavenumber meshgrid for the given shape and lengths."""
        raise NotImplementedError


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class FFTTransform(Transform):
    """
    The standard, JAX-native Fast Fourier Transform.

    ***Arguments***
    - dim: Number of spatial dimensions to transform over.

    ***Returns***
    - The FFT transform object.

    Example:

    ```
    fft_transform = FFTTransform(dim=2)
    x_hat = fft_transform.forward(x)
    x = fft_transform.inverse(x_hat)
    ```

    """

    # Layout of a field before and after the transform.  ``None`` means the
    # transform is single-device.  Sharded subclasses declare both specs here
    # and every ``shard_map`` below is driven from them, so the transpose
    # sequence and the wavenumber mesh can never disagree about the layout.
    physical_spec: ClassVar[P | None] = None
    spectral_spec: ClassVar[P | None] = None

    def _validate_mesh(self) -> None:
        """Check the mesh carries every axis name this transform's specs reference."""
        name = type(self).__name__
        assert self.device_mesh is not None, f"{name} requires a device_mesh."
        required = tuple(
            dict.fromkeys(
                axis
                for spec in (self.physical_spec, self.spectral_spec)
                for axis in spec
                if axis is not None
            )
        )
        missing = tuple(a for a in required if a not in self.device_mesh.axis_names)
        assert not missing, (
            f"{name} requires mesh axes {required}, but the mesh provides "
            f"{self.device_mesh.axis_names} (missing {missing})."
        )

    def forward(self, x: Array) -> Array:
        """
        Computes the centered FFT.

        ***Arguments***
        - x: Input array of shape (Nx, Ny, ..., d, d)
        ***Returns***
        - x_hat: Transformed array of the same shape
        """

        # Transform only the spatial axes (0 to dim-1)
        axes = range(self.dim)
        return jnp.fft.fftn(x, axes=axes)

    def inverse(self, x_hat: Array) -> Array:
        """
        Computes the inverse centered FFT.

        ***Arguments***
        - x_hat: Input array in frequency space of shape (Nx, Ny, ..., d, d)
        ***Returns***
        - x: Inverse transformed array of the same shape
        """
        axes = range(self.dim)
        return jnp.fft.ifftn(x_hat, axes=axes)

    def get_wavenumber_vector(self, size: int, length: float) -> Array:
        """
        Returns the real-valued wavenumber ξ.

        For an FFT on N points over length L, the wavenumbers are:
        ξ = 2π * [0, 1, ..., N/2-1, -N/2, ..., -1] / L

        ***Arguments***
        - size: Number of points in the spatial dimension.

        - length: Length of the spatial domain.
        ***Returns***
        - k: Real-valued wavenumber vector of shape (size,).

        """

        # Standard FFT frequencies: [0, 1, ..., -N/2, ..., -1]
        freqs = jnp.fft.fftfreq(size, d=length / size)
        return freqs * 2 * jnp.pi

    def get_wavenumber_mesh(
        self, shape: tuple[int, ...], lengths: tuple[float, ...]
    ) -> list[Array]:
        """
        Returns the wavenumber meshgrid for the given shape and lengths.

        When ``spectral_spec`` is set the meshgrid is built one shard at a time,
        so the full ``(N0, ..., Nd)`` array is never materialised on any device.
        """
        k_vecs = [
            self.get_wavenumber_vector(size=n, length=length)
            for n, length in zip(shape, lengths)
        ]

        if self.spectral_spec is None:
            return list(jnp.meshgrid(*k_vecs, indexing="ij"))

        # Axis i of the spectral array is sharded over ``spectral_spec[i]``, so
        # the 1-D vector feeding that axis carries exactly that one mesh axis.
        in_specs = tuple(P(axis) for axis in self.spectral_spec)

        @jax.shard_map(
            in_specs=in_specs, out_specs=self.spectral_spec, mesh=self.device_mesh
        )
        def local(*k_local):
            return jnp.meshgrid(*k_local, indexing="ij")

        # shard_map requires each input to already carry the sharding declared
        # in in_specs; it will not reshard implicitly.
        k_vecs = [
            jax.device_put(k, NamedSharding(self.device_mesh, spec))
            for k, spec in zip(k_vecs, in_specs)
        ]

        return local(*k_vecs)


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class DSTTransform(Transform):
    """Discrete Sine Transform (DST) using FFT.

    The DST is computed using the Fast Fourier Transform by embedding the
    interior domain into an odd (antisymmetric) periodic extension.

    For an N-point input along axis `a`, the odd extension constructs an array
    of length 2*(N+1):
        [0, u_0, ..., u_{N-1}, 0, -u_{N-1}, ..., -u_0]

    Along a single DST-I axis, the transform satisfies:
        dstI(actual_domain) = -0.5 * imag(FFT(full_domain)[actual_domain])
    where `actual_domain` corresponds to indices 1 to N along that axis.

    ***Arguments***
    - dim: Number of spatial dimensions to transform over.
    - axes: Axis or tuple of axes along which to apply the DST (e.g. 0, (0, 1), or (0, 1, 2)).
      Any spatial axes (in 0 .. dim-1) not in `axes` are transformed using standard FFT.
    - dst_axis: Alias for `axes`.
    - fft_transform: The underlying FFTTransform to delegate the FFT to (can be
      standard FFTTransform, SlabFFTTransform2D, SlabFFTTransform3D, or PencilFFTTransform).
    - dst_type: The type of DST (default "DST-I").
    """

    axes: tuple[int, ...] = field(metadata={"static": True}, default=(0,))
    fft_transform: FFTTransform = field(default_factory=FFTTransform)
    dst_type: str = field(metadata={"static": True}, default="DST-I")

    def __init__(
        self,
        dim: int | None = None,
        axes: int | tuple[int, ...] | list[int] | None = None,
        dst_axis: int | tuple[int, ...] | list[int] | None = None,
        fft_transform: FFTTransform | None = None,
        dst_type: str = "DST-I",
        device_mesh: jax.sharding.Mesh | None = None,
    ):
        if axes is None and dst_axis is None:
            norm_axes = (0,)
        elif axes is None:
            norm_axes = (dst_axis,) if isinstance(dst_axis, int) else tuple(dst_axis)
        else:
            norm_axes = (axes,) if isinstance(axes, int) else tuple(axes)

        if fft_transform is None:
            if dim is None:
                dim = max(norm_axes) + 1 if norm_axes else 1
            fft_transform = FFTTransform(dim=dim, device_mesh=device_mesh)
        else:
            if dim is None:
                dim = fft_transform.dim
            if device_mesh is None:
                device_mesh = fft_transform.device_mesh

        norm_axes = tuple(a % dim if a < 0 else a for a in norm_axes)

        object.__setattr__(self, "dim", dim)
        object.__setattr__(self, "device_mesh", device_mesh)
        object.__setattr__(self, "axes", norm_axes)
        object.__setattr__(self, "fft_transform", fft_transform)
        object.__setattr__(self, "dst_type", dst_type)

    @property
    def dst_axis(self) -> tuple[int, ...]:
        return self.axes

    @property
    def physical_spec(self) -> P | None:
        return self.fft_transform.physical_spec

    @property
    def spectral_spec(self) -> P | None:
        return self.fft_transform.spectral_spec

    def _extend_odd(self, u_interior: Array, axis: int) -> Array:
        """Extend an interior field with odd symmetry along `axis` to the full periodic domain.

        For an N-point input along `axis`, constructs an array of length 2*(N+1):
            [0, u_0, ..., u_{N-1}, 0, -u_{N-1}, ..., -u_0]
        """
        axis = axis % u_interior.ndim
        zeros = jnp.zeros(
            u_interior.shape[:axis] + (1,) + u_interior.shape[axis + 1 :],
            dtype=u_interior.dtype,
        )
        return jnp.concatenate(
            [zeros, u_interior, zeros, -jnp.flip(u_interior, axis=axis)],
            axis=axis,
        )

    def forward(self, x: Array) -> Array:
        """Compute the forward DST along `self.axes` via the underlying FFTTransform.

        ***Arguments***
        - x: Input array of shape (Nx, Ny, ..., *tensor_axes)
        ***Returns***
        - Transformed array of the same shape
        """
        x_ext = x
        for axis in self.axes:
            x_ext = self._extend_odd(x_ext, axis=axis)

        # Delegate the FFT of the extended domain to the underlying FFTTransform
        fft_full = self.fft_transform.forward(x_ext)

        # Slice the actual interior domain modes: [1 : N + 1] along DST axes
        slices = tuple(
            slice(1, x.shape[i] + 1) if i in self.axes else slice(None)
            for i in range(x.ndim)
        )
        x_actual = fft_full[slices]

        # Each odd extension introduces (-2i), so scale by (i/2) per DST axis
        factor = (0.5j) ** len(self.axes)
        x_hat = factor * x_actual

        if len(self.axes) == self.dim and not jnp.iscomplexobj(x):
            return x_hat.real
        return x_hat

    def inverse(self, x_hat: Array) -> Array:
        """Compute the inverse DST along `self.axes` via the underlying FFTTransform.

        ***Arguments***
        - x_hat: Transformed array of shape (Nx, Ny, ..., *tensor_axes)
        ***Returns***
        - Inverse transformed array of the same shape
        """
        x_hat_ext = x_hat
        for axis in self.axes:
            x_hat_ext = self._extend_odd(x_hat_ext, axis=axis)

        fft_dst = jnp.fft.fftn(x_hat_ext, axes=self.axes)
        slices = tuple(
            slice(1, x_hat.shape[i] + 1) if i in self.axes else slice(None)
            for i in range(x_hat.ndim)
        )
        x_dst = fft_dst[slices]

        # Scale factor for inverse DST: prod( 2 / (N_a + 1) * (i / 2) ) = prod( i / (N_a + 1) )
        scale = 1.0
        for a in self.axes:
            scale = scale * (1j / (x_hat.shape[a] + 1))
        x_dst = x_dst * scale

        # iFFT along remaining spatial axes (if any)
        fft_axes = tuple(i for i in range(self.dim) if i not in self.axes)
        if fft_axes:
            x_rec = jnp.fft.ifftn(x_dst, axes=fft_axes)
        else:
            x_rec = x_dst

        if len(self.axes) == self.dim and not jnp.iscomplexobj(x_hat):
            return x_rec.real
        return x_rec

    def get_wavenumber_vector(self, size: int, length: float) -> Array:
        """Returns the real-valued wavenumber vector for DST-I.

        For DST-I on N points over length L, the wavenumbers are:
        k = π * [1, 2, ..., N] / L
        """
        return jnp.arange(1, size + 1) * (jnp.pi / length)

    def get_wavenumber_mesh(
        self, shape: tuple[int, ...], lengths: tuple[float, ...]
    ) -> list[Array]:
        """Returns the wavenumber meshgrid for the given shape and lengths."""
        k_vecs = []
        for i, (n, length) in enumerate(zip(shape, lengths)):
            if i in self.axes:
                k_vecs.append(self.get_wavenumber_vector(size=n, length=length))
            else:
                k_vecs.append(
                    self.fft_transform.get_wavenumber_vector(size=n, length=length)
                )
        return list(jnp.meshgrid(*k_vecs, indexing="ij"))


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class SlabFFTTransform2D(FFTTransform):
    physical_spec: ClassVar[P] = P("x", None)
    spectral_spec: ClassVar[P] = P(None, "x")

    def __post_init__(self):
        assert self.dim == 2, "Slab decomposition 2D is only implemented for 2D FFT."
        self._validate_mesh()

    def forward(self, x):
        """Forward 2-D FFT. Input sharded on axis 0, output sharded on axis 1."""

        @jax.shard_map(
            in_specs=self.physical_spec,
            out_specs=self.spectral_spec,
            mesh=self.device_mesh,
        )
        def local(xl):
            xl = jnp.fft.fft(xl, axis=1)  # axis 1 complete -> local
            xl = jax.lax.all_to_all(xl, "x", split_axis=1, concat_axis=0, tiled=True)
            return jnp.fft.fft(xl, axis=0)  # axis 0 now complete -> local

        return local(x)

    def inverse(self, x_hat):
        """Inverse. Input sharded on axis 1, output sharded on axis 0 — exactly reversed."""

        @jax.shard_map(
            in_specs=self.spectral_spec,
            out_specs=self.physical_spec,
            mesh=self.device_mesh,
        )
        def local(xl):
            xl = jnp.fft.ifft(xl, axis=0)  # axis 0 complete -> local
            xl = jax.lax.all_to_all(xl, "x", split_axis=0, concat_axis=1, tiled=True)
            return jnp.fft.ifft(xl, axis=1)

        return local(x_hat)


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class SlabFFTTransform3D(FFTTransform):
    """3-D FFT with a 1-D device mesh.

    Only axis 0 (physical) or axis 1 (spectral) is ever distributed, so each
    transform costs a single transpose and every collective spans the whole
    mesh.  Usable while the device count does not exceed the grid size; beyond
    that, :class:`PencilDecomposition` is required.
    """

    physical_spec: ClassVar[P] = P("x", None, None)
    spectral_spec: ClassVar[P] = P(None, "x", None)

    def __post_init__(self):
        assert self.dim == 3, "SlabFFTTransform3D is only implemented for 3D FFT."
        self._validate_mesh()

    def forward(self, x):
        """Forward 3-D FFT. Input sharded on axis 0, output sharded on axis 1."""

        @jax.shard_map(
            in_specs=self.physical_spec,
            out_specs=self.spectral_spec,
            mesh=self.device_mesh,
        )
        def local(xl):
            xl = jnp.fft.fftn(xl, axes=(1, 2))  # axes 1, 2 complete -> local
            xl = jax.lax.all_to_all(xl, "x", split_axis=1, concat_axis=0, tiled=True)
            return jnp.fft.fft(xl, axis=0)  # axis 0 now complete -> local

        return local(x)

    def inverse(self, x_hat):
        """Inverse. Input sharded on axis 1, output sharded on axis 0 — exactly reversed."""

        @jax.shard_map(
            in_specs=self.spectral_spec,
            out_specs=self.physical_spec,
            mesh=self.device_mesh,
        )
        def local(xl):
            xl = jnp.fft.ifft(xl, axis=0)  # axis 0 complete -> local
            xl = jax.lax.all_to_all(xl, "x", split_axis=0, concat_axis=1, tiled=True)
            return jnp.fft.ifftn(xl, axes=(1, 2))

        return local(x_hat)


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class PencilFFTTransform(FFTTransform):
    physical_spec: ClassVar[P] = P("x", None, "z")
    spectral_spec: ClassVar[P] = P("z", "x", None)

    def __post_init__(self):
        assert self.dim == 3, "Pencil decomposition is only implemented for 3D FFT."
        self._validate_mesh()

    def forward(self, x):
        """Forward 3-D FFT. Input sharded on axis 0 and 2, output sharded on axis 0, 1."""

        @jax.shard_map(
            in_specs=self.physical_spec,
            out_specs=self.spectral_spec,
            mesh=self.device_mesh,
        )
        def local(xl):
            xl = jnp.fft.fft(xl, axis=1)  # axis 1 complete -> local
            xl = jax.lax.all_to_all(
                xl, "x", split_axis=1, concat_axis=0, tiled=True
            )  # it tells which device axis splits the data
            xl = jnp.fft.fft(xl, axis=0)
            xl = jax.lax.all_to_all(xl, "z", split_axis=0, concat_axis=2, tiled=True)
            return jnp.fft.fft(xl, axis=2)  # axis 0 now complete -> local

        return local(x)

    def inverse(self, x_hat):
        """Inverse. Input sharded on axis 0, 1, output sharded on axis 0 and 2 — exactly reversed."""

        @jax.shard_map(
            in_specs=self.spectral_spec,
            out_specs=self.physical_spec,
            mesh=self.device_mesh,
        )
        def local(xl):
            xl = jnp.fft.ifft(xl, axis=2)  # axis 2 complete -> local
            xl = jax.lax.all_to_all(xl, "z", split_axis=2, concat_axis=0, tiled=True)
            xl = jnp.fft.ifft(xl, axis=0)  # axis 0 complete -> local
            xl = jax.lax.all_to_all(xl, "x", split_axis=0, concat_axis=1, tiled=True)
            return jnp.fft.ifft(xl, axis=1)

        return local(x_hat)
