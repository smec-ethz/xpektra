# Discretization Schemes

In **`xpektra`**, we define a discretization scheme which allows us to correctly define various differentiation operators. To do so, we need two information:

- The underlying grid in the physical space _i.e_ if regular, staggered, etc.
- The differentiation formula to be used.

In order to facilitate this, we define a base class `Scheme` which provides the necessary infrastructure to define a discretization scheme.

`Scheme` owns the four differential operations, so every scheme shares a single
adjoint convention: the divergence operator is $-\overline{D(\xi)}$, which is
what makes `div` the adjoint of `sym_grad` and hence $D^{T} C D$ symmetric. A
subclass supplies only `compute_gradient_operator`.

`apply_gradient` accepts a node field of any rank and places the **derivative
index first**, so for a vector field $(\nabla u)_{ij} = \partial_i u_j$. That is
the axis `apply_divergence` contracts, which is why
`div(grad(u)) == laplacian(u)`. Note this is the transpose of the
continuum-mechanics convention $(\nabla u)_{ij} = \partial_j u_i$; a deformation
gradient is therefore `I + swapaxes(grad(u), -1, -2)`.

::: xpektra.scheme.Scheme
    options:
        members: 
            - __init__
            - compute_gradient_operator
            - is_compatible
            - apply_gradient
            - apply_symmetric_gradient
            - apply_divergence
            - apply_laplacian

## Finite difference schemes

Most schemes are defined by a *stencil* — a list of `(offset, weight)` pairs —
rather than by a closed-form symbol. `FiniteDifferenceScheme` turns a stencil
into its Fourier symbol $Z(\xi) = \sum_a w_a \exp(\iota\, \xi \cdot a h)$
symbolically, so a new scheme only has to declare its `stencils`.

A scheme may also carry more than one *derivation support* per voxel. The
number of supports is `n_quads`, and the gradient operator is always
`(*spatial, n_quads, dim)` — the quadrature axis is present even for a
single-support scheme, so centre fields (strain, stress) have the same shape
whichever scheme is in use.

::: xpektra.scheme.FiniteDifferenceScheme
    options:
        members: 
            - stencils
            - support_stencils
            - build_fourier_operator
            - compute_gradient_operator

::: xpektra.scheme.CentralScheme
    options:
        members: 
            - stencils

The symbol is given by:

$$D(\xi) = \iota \frac{\sin(\xi \Delta x)}{\Delta x}$$

where $\iota$ is the imaginary unit, $\xi$ is the wavenumber and $\Delta x$ is the grid spacing.

::: xpektra.scheme.ForwardScheme
    options:
        members: 
            - stencils

The symbol is given by:

$$D(\xi) = \frac{\exp(\iota \xi \Delta x) - 1}{\Delta x}$$

::: xpektra.scheme.BackwardScheme
    options:
        members: 
            - stencils

The symbol is given by:

$$D(\xi) = \frac{1 - \exp(-\iota \xi \Delta x)}{\Delta x}$$

### Rotated (Willot) schemes

`Quad1RScheme` (2D) and `Hex1RScheme` (3D) are the rotated finite difference
schemes of Willot. Their symbol is

$$D_i(\xi) = \frac{2 \iota \tan(\xi_i \Delta x_i / 2)}{\Delta x_i} \prod_j \frac{1 + \exp(\iota \xi_j \Delta x_j)}{2}$$

built here from the corresponding voxel-corner stencils rather than from the
closed form.

::: xpektra.scheme.Quad1RScheme
    options:
        members: 
            - stencils
            - is_compatible

::: xpektra.scheme.Hex1RScheme
    options:
        members: 
            - stencils
            - is_compatible

### Multi-support schemes

::: xpektra.scheme.Tetra2Scheme
    options:
        members: 
            - support_stencils
            - is_compatible

## Spectral scheme

`FourierScheme` is the one scheme whose symbol is not a finite stencil, so it
builds its gradient operator directly.

::: xpektra.scheme.FourierScheme
    options:
        members: 
            - compute_gradient_operator

The symbol is given by:

$$D(\xi) = \iota \xi$$

where $\iota$ is the imaginary unit and $\xi$ is the wavenumber. Note that
$-\overline{\iota \xi} = \iota \xi$, so for this scheme the shared adjoint
convention reduces to the classical spectral operators.
