# Spectral Operator

In **`xpektra`**, spectral operators are defined to perform various operations in the spectral domain. These operators are essential for implementing spectral methods for solving partial differential equations. The spectral operators are built upon the discretization schemes defined in the `scheme` module and the spectral spaces defined in the `space` module.

The ``SpectralOperator`` class serves as a unified interface for various spectral operations, including gradient, symmetric gradient, divergence, and Laplacian operations. It also provides methods for forward and inverse Fourier transforms, as well as tensor operations like double dot product, dot product, trace, and dyadic product.

::: xpektra.spectral_operator.SpectralOperator
    options:
        members: 
            - scheme
            - space
            - tensor_op
            - grad
            - sym_grad
            - div
            - laplacian
            - forward
            - inverse   
            - ddot
            - dot
            - trace
            - dyad
