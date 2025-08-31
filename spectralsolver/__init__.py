from spectralsolver.operator import Operator, GradientMode, TensorOperator
import numpy as np

def make_field(dim: int, N: int, rank: int) -> np.ndarray:
    return np.zeros((rank,) * rank + (N,) * dim)