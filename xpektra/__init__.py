import numpy as np

from xpektra.space import SpectralSpace
from xpektra.spectral_operator import SpectralOperator
from xpektra.tensor_operator import TensorOperator
from xpektra.transform import FFTTransform


def make_field(
    dim: int, shape: tuple, rank: int, dtype: np.dtype = float
) -> np.ndarray:
    """
    Creates a zero-filled tensor field with the (spatial..., tensor...) memory layout.

    Args:
        dim: The number of spatial dimensions (e.g., 3 for a 3D grid).
        N: The number of grid points along each spatial dimension.
        rank: The rank of the tensor at each grid point (e.g., 0 for scalar, 1 for vector, 2 for tensor).

    Returns:
        A NumPy array with the correct shape.
    """
    spatial_shape = shape  # Shape of the spatial dimensions
    tensor_shape = (dim,) * rank  # Assumes tensor dimensions are size `dim`

    return np.zeros(spatial_shape + tensor_shape, dtype=dtype)
