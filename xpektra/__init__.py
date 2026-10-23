from xpektra.preconditioner import (
    AdaptedGreenPreconditioner,
    GenericGreenPreconditioner,
    IsotropicGreenPreconditioner,
    adapt_green_preconditioner,
    make_generic_preconditioner,
    make_isotropic_preconditioner,
)
from xpektra.projection_operator import (
    GalerkinProjection,
    MoulinecSuquetProjection,
    ProjectionOperator,
)
from xpektra.scheme import (
    BackwardScheme,
    CentralScheme,
    ForwardScheme,
    FourierScheme,
    Hex1RScheme,
    Quad1RScheme,
    QuadFullScheme,
    Tetra2Scheme,
)
from xpektra.solution_space import Field, SolutionSpace, State
from xpektra.space import SpectralSpace
from xpektra.spectral_operator import SpectralOperator
from xpektra.transform import (
    FFTTransform,
    PencilFFTTransform,
    RFFTTransform,
    SlabFFTTransform2D,
    SlabFFTTransform3D,
    SlabRFFTTransform2D,
    SlabRFFTTransform3D,
)

__all__ = [
    # scheme
    "BackwardScheme",
    "CentralScheme",
    # solution_space
    "Field",
    "SolutionSpace",
    "State",
    # transform
    "FFTTransform",
    "ForwardScheme",
    "FourierScheme",
    # projection_operator
    "GalerkinProjection",
    "GenericGreenPreconditioner",
    "Hex1RScheme",
    "IsotropicGreenPreconditioner",
    "MoulinecSuquetProjection",
    "PencilFFTTransform",
    "ProjectionOperator",
    "Quad1RScheme",
    "QuadFullScheme",
    "RFFTTransform",
    "SlabFFTTransform2D",
    "SlabFFTTransform3D",
    "SlabRFFTTransform2D",
    "SlabRFFTTransform3D",
    "SpectralOperator",
    # space / operators
    "SpectralSpace",
    "Tetra2Scheme",
    # helpers
    "AdaptedGreenPreconditioner",
    "adapt_green_preconditioner",
    "make_generic_preconditioner",
    "make_isotropic_preconditioner",
]
