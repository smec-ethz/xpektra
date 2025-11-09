## Tensor fields

In **Xpektra**, a tensor such as stress, strain, displacement, etc. is represented as a tensor at each grid point in space. A tensor field on a grid is thus given as:

$$
\varepsilon_{xyij}
$$

where ``x, y`` are the indices of the grid points and ``i, j`` are the indices of the tensor.

For example, a stress tensor in $\mathbb{R}^2$ is represented as a tensor of shape ``(2, 2)`` at each grid point in space. The way we store this tensor grig field is using `numpy` or `jax.numpy` array. The shape of the array is ``( (N,)*dim, (dim,)*rank)`` where ``N`` is the number of grid points in space, ``dim`` is the dimension of the space, and ``rank`` is the rank of the tensor. Therefore, a rank 2 tensor in $\mathbb{R}^2$ is represented as a tensor of shape ``(N, N, 2, 2)`` and similarly a rank 1 tensor in $\mathbb{R}^2$ is represented as a tensor of shape ``(N, N, 2)``.

**Xpektra** provides a function `make_field` to create a tensor field. The function takes the following arguments:

- `dim`: the dimension of the space
- `N`: the number of grid points in space
- `rank`: the rank of the tensor


```python
from xpektra import make_field  

stress = make_field(dim=2, N=12, rank=2)
```

## Tensor operations on grid

The tensor operations on the entire grid are mathematically given as:


### Dot product

For a **rank 0** tensor and a **rank 0** tensor, the dot product is given as:

$$
a_{xy} \cdot b_{xy} = c_{xy}
$$

For a **rank 1** tensor and a **rank 1** tensor, the dot product is given as:

$$
a_{xyi} \cdot b_{xyj} = c_{xyij}
$$

For a **rank 2** tensor and a **rank 1** tensor, the dot product is given as:

$$
a_{xyij} \cdot b_{xyj} = c_{xyi}
$$

For a **rank 2** tensor and a **rank 2** tensor, the dot product is given as:

$$
a_{xyij} \cdot b_{xyjk} = c_{xyik}
$$

### Double dot product

For a **rank 2** tensor and a **rank 2** tensor, the double dot product is given as:

$$
a_{xyij} \cdot b_{xyji} = c_{xy}
$$

For a **rank 4** tensor and a **rank 2** tensor, the double dot product is given as:

$$
a_{xyijkl} \cdot b_{xylk} = c_{xyij}
$$

For a **rank 4** tensor and a **rank 4** tensor, the double dot product is given as:

$$
a_{xyijkl} \cdot b_{xylkmn} = c_{xyijmn}
$$


### Dyadic product

For a **rank 1** tensor and a **rank 1** tensor, the dyadic product is given as:

$$
a_{xyi} \otimes b_{xyj} = c_{xyij}
$$

For a **rank 2** tensor and a **rank 1** tensor, the dyadic product is given as:


All of these operations are implemented in the `xpektra.TensorOperator` module. To use these operations, we need to create a `TensorOperator` object. The constructor takes the following arguments:
- `dim`: the dimension of the space



```python
from xpektra import TensorOperator

tensor = TensorOperator(dim=2)
```

Once we have defined the `TensorOperator` object, we can use it to perform the tensor operations on the grid. For example, to perform the dot product of two tensors, we can use the `dot` method of the `TensorOperator` object.

```python
a = make_field(dim=2, N=12, rank=2)
b = make_field(dim=2, N=12, rank=2)
dot_product = tensor.dot(a, b)
```

The `TensorOperator` object automatically detects the rank of the tensors and performs the appropriate operation. This is the reason why we only need to define the `dim` argument in the `TensorOperator` constructor.


## Spectral space

In **Xpektra**, we define a spectral space which allows us to correctly define various 
```python
from xpektra import SpectralSpace

spectral_space = SpectralSpace(dim=2, N=12)
```

The `SpectralSpace` object has the following attributes:

- `dim`: the dimension of the space
- `size`: the size of the spectral space
- `wavenumber_vector`: the wavenumber vector
- `frequency_vector`: the frequency vector


## Discretization schemes

In **Xpektra**, we define a discretization scheme which allows us to correctly define various differentiation operators. The discretization scheme is defined by the `Scheme` class. The `Scheme` class is a base class for all the discretization schemes. The `Scheme` class has the following methods:

- `compute_gradient_operator`: computes the gradient operator
- `create_wavenumber_mesh`: creates the wavenumber mesh





## Green operators

### Fourier-Galerkin operator


### Moulinec-Suquet operator


