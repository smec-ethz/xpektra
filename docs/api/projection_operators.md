# Projection Operators

::: xpektra.projection_operator.ProjectionOperator
    options:
        members: 
            - compute_operator
            - scheme
            - tensor_op

::: xpektra.projection_operator.GalerkinProjection
    options:
        members: 
            - compute_operator

The formula is given by: <br>

$$\hat{G}_{ijlm} = \delta_{im} * D\xi_{j} * D\xi^{-1}_{l}$$

where $\delta_{im}$ is the Kronecker delta, $D\xi_{j}$ is the gradient operator and $D\xi^{-1}_{l}$ is the inverse of the gradient operator.

::: xpektra.projection_operator.MoulinecSuquetProjection
    options:
        members: 
            - compute_operator
            - space
            - lambda0
            - mu0