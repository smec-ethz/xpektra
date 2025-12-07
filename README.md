
<div align="center">
<img src="docs/assets/xpektra-trans-low.png" alt="drawing" width="400"/>
<h3 align="center">xpektra : Modular framework for spectral methods</h3>


`xpektra` is a Python library that provides a modularframework for spectral methods.  `xpektra` provide fundamental mathematical building blocks which can be used to construct complex spectral methods. It is built on top of JAX and Equinox, making it easy to use spectral methods in a differentiable way.


</div>

[![Documentation](https://github.com/smec-ethz/xpektra/actions/workflows/build_docs.yml/badge.svg)](https://github.com/smec-ethz/xpektra/actions/workflows/build_docs.yml)
[![Tests](https://github.com/smec-ethz/xpektra/actions/workflows/run_tests.yml/badge.svg)](https://github.com/smec-ethz/xpektra/actions/workflows/run_tests.yml)

## License
`xpektra` is distributed under the GNU Lesser General Public License v3.0 or later. See `COPYING` and `COPYING.LESSER` for the complete terms. © 2025 ETH Zurich (Mohit Pundir).

## Features

- Modular building blocks for spectral methods which can be easily combined to create complex solid mechanics problems.
- Extensible design allowing users to define their own operators and spaces such as Fourier-Galerkin, Moulinec and Suquet, Displacement-based, etc.
- Differentiable operations using JAX
- Implicit differentiation support which allows for computationally efficient Homogenization and Multiscale simulations.


## Installation
Install the current release from PyPI:

```bash
pip install xpektra
```
For development work, clone the repository and install it in editable mode (use your preferred virtual environment tool such as `uv` or `venv`):

```bash
git clone https://github.com/smec-ethz/xpektra.git
cd xpektra
pip install -e .
```

## Usage

`xpektra` provides modular blocks by defining a few operators and spaces that can be used to construct complex spectral methods. These operators and spaces are:

- `SpectralSpace`: Defines the spectral space on which the methods are defined, this includes the FFT, IFFT, and other spectral transforms.
- `TensorOperator`: Defines the tensor operator which performs tensor operations on quantities living on the grid, the operator automatically figures out the order of the tensor.
- `Scheme`: Defines the scheme for discretization which is then used to construct the gradient operator. Currently, `CartesianScheme` is the only scheme available but one can easily define new schemes by subclassing the `Scheme` class.
- `make_field`: Defines the field on which the methods are defined, this includes the field operations and the field creation.
- `ProjectionOperator`: Defines the projection operator which projects the stress field onto the spectral space. Currently, `GalerkinProjection` is the only projection operator available but one can easily define new projection operators by subclassing the `ProjectionOperator` class.


## 👉 Where to contribute

If you have a suggestion that would make this better, please fork the repo and create a pull request on [**github.com/smec-ethz/xpektra**](https://github.com/smec-ethz/xpektra). Please use that repository to open issues or submit merge requests. You can also simply open an issue with the tag "enhancement". Don't forget to give the project a star! Thanks again!

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request
