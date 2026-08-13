import numpy as np

from xpektra.projection_operator import (
    GalerkinProjection,
    MoulinecSuquetProjection,
    ProjectionOperator,
)
from xpektra.scheme import (
    BackwardDifference,
    BackwardScheme,
    CentralDifference,
    CentralScheme,
    EighthOrderCentralDifference,
    ForwardDifference,
    ForwardScheme,
    FourierScheme,
    FourthOrderCentralDifference,
    Hex1RScheme,
    Quad1RScheme,
    RotatedDifference,
    SixthOrderCentralDifference,
    Tetra2Scheme,
)
from xpektra.space import SpectralSpace
from xpektra.spectral_operator import SpectralOperator
from xpektra.tensor_operator import TensorOperator
from xpektra.transform import (
    FFTTransform,
    PencilFFTTransform,
    SlabFFTTransform2D,
    SlabFFTTransform3D,
)

__all__ = [
    # scheme
    "BackwardDifference",
    "BackwardScheme",
    "CentralDifference",
    "CentralScheme",
    "EighthOrderCentralDifference",
    # transform
    "FFTTransform",
    "ForwardDifference",
    "ForwardScheme",
    "FourierScheme",
    "FourthOrderCentralDifference",
    # projection_operator
    "GalerkinProjection",
    "Hex1RScheme",
    "MoulinecSuquetProjection",
    "PencilFFTTransform",
    "ProjectionOperator",
    "Quad1RScheme",
    "RotatedDifference",
    "SixthOrderCentralDifference",
    "SlabFFTTransform2D",
    "SlabFFTTransform3D",
    "SpectralOperator",
    # space / operators
    "SpectralSpace",
    "TensorOperator",
    "Tetra2Scheme",
    # helpers
    "make_field",
]


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
