"""The sharded transforms must handle a quadrature-carrying field unchanged.

This is the property that decided the field layout.  ``physical_spec`` and
``spectral_spec`` are ``PartitionSpec``\\s indexed *from the left*
(``P("x", None, None)``), and the ``all_to_all`` calls inside each transform name
axes 0/1/2 explicitly.  With the quadrature axis trailing --
``(*spatial, n_quads, dim, dim)`` -- those specs leave it replicated and keep
working untouched.  Had it been leading, they would have sharded the quadrature
axis over the device mesh and replicated a spatial one instead.
"""

import os

os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=2")

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

from xpektra.transform import FFTTransform, SlabFFTTransform3D

N = 8


@pytest.fixture(scope="module")
def mesh():
    devices = jax.devices()
    if len(devices) < 2:
        pytest.skip(f"needs >= 2 devices, got {len(devices)}")
    return Mesh(np.asarray(devices[:2]).reshape(2), axis_names=("x",))


@pytest.mark.parametrize("tensor_shape", [(), (3,), (3, 3), (2, 3, 3)])
def test_slab3d_roundtrip_over_trailing_axes(mesh, tensor_shape):
    """Forward then inverse is the identity for any trailing-axis structure.

    ``(2, 3, 3)`` is the TETRA2 case: two quadrature points of a rank-2 tensor.
    """
    transform = SlabFFTTransform3D(dim=3, device_mesh=mesh)

    shape = (N, N, N) + tensor_shape
    key = jax.random.PRNGKey(0)
    x = jax.random.normal(key, shape) + 0j
    x = jax.device_put(x, NamedSharding(mesh, P("x", None, None)))

    recovered = transform.inverse(transform.forward(x))
    np.testing.assert_allclose(recovered, x, atol=1e-12)


def test_slab3d_matches_unsharded_fft_on_a_quadrature_field(mesh):
    """The sharded transform agrees with the single-device one, quadrature included."""
    shape = (N, N, N, 2, 3, 3)
    key = jax.random.PRNGKey(1)
    x = jax.random.normal(key, shape) + 0j

    plain = FFTTransform(dim=3).forward(x)

    sharded = SlabFFTTransform3D(dim=3, device_mesh=mesh).forward(
        jax.device_put(x, NamedSharding(mesh, P("x", None, None)))
    )

    np.testing.assert_allclose(jnp.asarray(sharded), plain, atol=1e-10)


def test_spatial_axes_stay_leading(mesh):
    """A guard on the invariant itself: the spec must not touch the quadrature axis."""
    spec = SlabFFTTransform3D.physical_spec
    assert len(spec) == 3, "physical_spec indexes the three spatial axes only"
    assert spec[0] == "x", "axis 0 must be the sharded spatial axis"
