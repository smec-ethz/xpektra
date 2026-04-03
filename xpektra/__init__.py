import numpy as np

from xpektra.space import SpectralSpace
from xpektra.spectral_operator import SpectralOperator
from xpektra.tensor_operator import TensorOperator
from xpektra.transform import FFTTransform
from xpektra.scheme import (
    FourierScheme,
    CentralDifference,
    ForwardDifference,
    BackwardDifference,
    RotatedDifference,
    FourthOrderCentralDifference,
    SixthOrderCentralDifference,
    EighthOrderCentralDifference,
)
from xpektra.projection_operator import (
    GalerkinProjection,
    MoulinecSuquetProjection,
    ProjectionOperator,
)


def make_field(
    dim: int, shape: tuple, rank: int, dtype: np.dtype = float, fill_value: float = 0
) -> np.ndarray:
    """
    Creates a tensor field with the (spatial..., tensor...) memory layout.

    Args:
        dim: The number of spatial dimensions (e.g., 3 for a 3D grid).
        shape: The shape of the spatial dimensions.
        rank: The rank of the tensor at each grid point (e.g., 0 for scalar, 1 for vector, 2 for tensor).
        dtype: The data type of the field. Defaults to float.
        fill_value: The value to fill the field with. Defaults to 0.

    Returns:
        A NumPy array with the correct shape.
    """
    spatial_shape = shape
    tensor_shape = (dim,) * rank

    field = np.empty(spatial_shape + tensor_shape, dtype=dtype)
    field.fill(fill_value)
    return field
