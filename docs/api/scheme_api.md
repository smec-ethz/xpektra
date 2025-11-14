# Discretization Schemes

In **Xpektra**, we define a discretization scheme which allows us to correctly define various differentiation operators. To do so, we need two information:

- The underlying grid in the physical space _i.e_ if regular, staggered, etc.
- The differentiation formula to be used.

In order to facilitate this, we define a base class `Scheme` which provides the necessary infrastructure to define a discretization scheme.

::: xpektra.scheme.Scheme
    options:
        members: 
            - compute_gradient_operator
            - is_compatible


The `Scheme` class is a base class for all the discretization schemes. One can create different discretization schemes by subclassing the `Scheme` class and implementing the `formula` method. The `formula` method should return the gradient operator field for a given wavenumber and grid spacing. In **Xpektra**, we have implemented the `CartesianScheme` which takes a regular grid in physical space and returns the gradient operator field in spectral space.

::: xpektra.scheme.DiagonalScheme
    options:
        members: 
            - __init__
            - apply_gradient
            - apply_symmetric_gradient
            - apply_divergence
            - apply_laplacian
            - compute_gradient_operator
            - is_compatible
            - formula

To define the differentiation formula, we need to implement the `formula` method. The `formula` method should return the gradient operator field for a given wavenumber and grid spacing. In **Xpektra**, we have various differentiation schemes available which can be used to define the differentiation formula.

::: xpektra.scheme.FourierScheme
    options:
        members: 
            - formula  

The formula is given by:

$$D(\xi) = \iota \xi$$

where $\iota$ is the imaginary unit and $\xi$ is the wavenumber.
 
::: xpektra.scheme.CentralDifference
    options:
        members: 
            - formula  
            
The formula is given by:

$$D(\xi) = \iota \frac{\sin(\xi \Delta x)}{\Delta x}$$

where $\iota$ is the imaginary unit, $\xi$ is the wavenumber and $\Delta x$ is the grid spacing.
 
::: xpektra.scheme.ForwardDifference
    options:
        members: 
            - formula    
            
The formula is given by:

$$D(\xi) = \frac{\exp(\iota \xi \Delta x) - 1}{\Delta x}$$

where $\iota$ is the imaginary unit, $\xi$ is the wavenumber and $\Delta x$ is the grid spacing.

::: xpektra.scheme.BackwardDifference
    options:
        members: 
            - formula
            
The formula is given by:

$$D(\xi) = \frac{1 - \exp(-\iota \xi \Delta x)}{\Delta x}$$

where $\iota$ is the imaginary unit, $\xi$ is the wavenumber and $\Delta x$ is the grid spacing.


::: xpektra.scheme.RotatedDifference
    options:
        members: 
            - formula

The formula is given by:

$$D(\xi) = \frac{2 \iota \tan(\xi \Delta x / 2) \Delta x}{2 \Delta x}$$

where $\iota$ is the imaginary unit, $\xi$ is the wavenumber and $\Delta x$ is the grid spacing.


::: xpektra.scheme.FourthOrderCentralDifference
    options:
        members: 
            - formula

The formula is given by:

$$D(\xi) = \iota \frac{8 \sin(\xi \Delta x) - 2 \sin(2 \xi \Delta x) + 8 \sin(3 \xi \Delta x) - \sin(4 \xi \Delta x)}{6 \Delta x}$$

where $\iota$ is the imaginary unit, $\xi$ is the wavenumber and $\Delta x$ is the grid spacing.


::: xpektra.scheme.SixthOrderCentralDifference
    options:
        members: 
            - formula

The formula is given by:

$$D(\xi) = \iota \frac{9 \sin(\xi \Delta x) - 3 \sin(2 \xi \Delta x) + \sin(3 \xi \Delta x)}{6 \Delta x}$$

where $\iota$ is the imaginary unit, $\xi$ is the wavenumber and $\Delta x$ is the grid spacing.

::: xpektra.scheme.EighthOrderCentralDifference
    options:
        members: 
            - formula

The formula is given by:

$$D(\xi) = \iota \frac{8 \sin(\xi \Delta x) - 2 \sin(2 \xi \Delta x) + 8 \sin(3 \xi \Delta x) - \sin(4 \xi \Delta x)}{12 \Delta x}$$

where $\iota$ is the imaginary unit, $\xi$ is the wavenumber and $\Delta x$ is the grid spacing.
