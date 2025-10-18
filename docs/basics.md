## Tensor fields

In **Xpektra**, a tensor such as stress, strain, displacement, etc. is represented as a tensor at each grid point in space. A tensor field on a grid is thus given as:

$$
\varepsilon_{ijxy} 
$$

where ``i, j`` are the indices of the tensor and ``x, y`` are the indices of the grid points.

For example, a stress tensor in $\mathbb{R}^2$ is represented as a tensor of shape ``(2, 2)`` at each grid point in space. The way we store this tensor grig field is using `numpy` or `jax.numpy` array. The shape of the array is ``((dim,)*rank, (N,)*dim)`` where ``N`` is the number of grid points in space, ``dim`` is the dimension of the space, and ``rank`` is the rank of the tensor. Therefore, a rank 2 tensor in $\mathbb{R}^2$ is represented as a tensor of shape ``(2, 2, N, N)`` and similarly a rank 1 tensor in $\mathbb{R}^2$ is represented as a tensor of shape ``(2, N, N)``.

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
a_{ixy} \cdot b_{jxy} = c_{ijxy}
$$

For a **rank 2** tensor and a **rank 1** tensor, the dot product is given as:

$$
a_{ijxy} \cdot b_{jxy} = c_{ixy}
$$

For a **rank 2** tensor and a **rank 2** tensor, the dot product is given as:

$$
a_{ijxy} \cdot b_{jkxy} = c_{ikxy}
$$

### Double dot product

For a **rank 2** tensor and a **rank 2** tensor, the double dot product is given as:

$$
a_{ijxy} \cdot b_{jiy} = c_{xy}
$$
For a **rank 4** tensor and a **rank 2** tensor, the double dot product is given as:

$$
a_{ijklxy} \cdot b_{lkxy} = c_{ijxy}
$$

For a **rank 4** tensor and a **rank 4** tensor, the double dot product is given as:

$$
a_{ijklxy} \cdot b_{lkmnxy} = c_{ijmnxy}
$$


### Dyadic product

For a **rank 1** tensor and a **rank 1** tensor, the dyadic product is given as:

$$
a_{ixy} \otimes b_{jxy} = c_{ijxy}
$$

For a **rank 2** tensor and a **rank 1** tensor, the dyad is given as:


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


## Green operators

### Fourier-Galerkin operator