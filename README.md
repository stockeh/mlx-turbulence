# Turbulent Flows

The unsteady two-dimensional incompressible Navier-Stokes equation for a viscous, incompressible fluid with Kolmogorov forcing in the vorticity form is given by:

$$
\begin{cases}
\partial_t \omega + \mathbf{u} \cdot \nabla \omega = \nu \Delta \omega + f(x, y), & (x, y) \in (0, 2\pi)^2, \, t \in (0, t_{\text{final}}], \\
f(x, y) = \chi \left(\sin(2\pi(x + y)) + \cos(2\pi(x + y))\right), & (x, y) \in (0, 2\pi)^2, \\
\nabla \cdot \mathbf{u} = 0, & (x, y) \in (0, 2\pi)^2, \, t \in (0, t_{\text{final}}], \\
\omega(x, y, 0) = \omega_0, & (x, y) \in (0, 2\pi)^2.
\end{cases}
$$

where $\chi = 0.1$, $\omega$ is the vorticity, $\mathbf{u}$ is the velocity field, $\nu=1\text{e}-6$ is the kinematic viscosity, and $\Delta$ is the two-dimensional Laplacian operator. Initial condition $\omega_0(x) \sim \mathcal{N}\left(0, 14^{1/2}\left(-\Delta + 196 I\right)^{-3/2}\right)$


Exisiting datasets from [ixScience's Google Drive](https://drive.google.com/drive/folders/1UnbQh2WWc6knEHbLn-ZaXrKUZhp7pjt-) and corresponding repo [ixScience/fourier_neural_operator](https://github.com/ixScience/fourier_neural_operator).