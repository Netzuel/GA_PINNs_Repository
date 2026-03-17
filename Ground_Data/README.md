# Gradient-Annihilated PINNs (GA-PINNs) for Solving Riemann Problems: Application to Relativistic Hydrodynamics
## Problems considered.

The configuration and information of the different problems considered is explained below. These are the problems proposed in the original published manuscript about [GA PINNs](https://www.sciencedirect.com/science/article/pii/S0045782524001622).


### Problem 1. Sod Shock tube.

- $\Gamma=5/3$,
- $(t,x)\in[0,0.5]\times[0,1]$.

Initial conditions:

$$
U(0,x) := U_0(x) =
\begin{cases}
\rho_0(x) = 1.0 & \text{if } x \leq 0.5, \quad \rho_0(x) = 0.125 \ \text{otherwise} \\
u_0(x) = 0.5 & \forall x \\
p_0(x) = 1.0 & \text{if } x \leq 0.5, \quad p_0(x) = 0.1 \ \text{otherwise}
\end{cases}
$$
