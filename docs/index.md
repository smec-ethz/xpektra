#  Getting Started

<div align="center">

<img src="assets/xpektra-trans-low.png" alt="drawing" width="400" height="100"/>

<h3 align="center">xpektra: Modular framework for spectral methods</h3>


`xpektra` is a Python library that provides a modularframework for spectral methods.  `xpektra` provide fundamental mathematical building blocks which can be used to construct complex spectral methods. It is built on top of JAX and Equinox, making it easy to use spectral methods in a differentiable way.



</div>

## License

`xpektra` is distributed under the GNU Lesser General Public License v3.0 or later. See `COPYING` and `COPYING.LESSER` for the complete terms. © 2025 ETH Zurich (Mohit Pundir).


## Features

- Modular building blocks for spectral methods which can be easily combined to create complex solid mechanics problems.
- Extensible design allowing users to define their own operators and spaces such as Fourier-Galerkin, Moulinec and Suquet, Displacement-based, etc.
- Differentiable operations using JAX
- Implicit differentiation support which allows for computationally efficient Homogenization and Multiscale simulations.


## Installation

Requires Python 3.10+.

```bash
pip install xpektra
```


## Quick Example
