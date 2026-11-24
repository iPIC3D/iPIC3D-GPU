# Simulation cases

[Documentation index](../README.md#documentation)

The `Case` key selects the initial fields and particles.

## Supported cases

| `Case` | Description | Example deck in `share/inputfiles` |
|---|---|---|
| `GEM` | GEM reconnection challenge: Harris sheet $B_x = B_{0x}\tanh((y-L_y/2)/\delta)$ with a localised Gaussian X-point perturbation of flux amplitude $0.4\,B_{0x}\delta$ at the box centre. Species with nonzero `u0` or `w0` form the sheet (density $\propto\operatorname{sech}^2$) and carry its current; the others are a uniform background | `magneticReconnection/testGEM3Dsmall.inp` |
| `GEMnoPert` | The `GEM` equilibrium without the perturbation | |
| `GEMDoubleHarris` | Two opposite sheets at $L_y/4$ and $3L_y/4$, carried by the drifting species: density $\operatorname{sech}^2$ of both sheets, drift $(u_0,v_0,w_0)$ below $L_y/2$ and $(u_0,v_0,-w_0)$ above. The X-point perturbation is on the lower sheet | `turbulence/DebyeScaleTurbulence.inp` |
| `GEMHarris` | Generalised Harris sheet with optional GEM (`pertGEM`) and hump (`pertHump`) perturbations, Ampère-consistent currents (`currentFromAmpere`) and a varying sheet temperature (`spatiallyVaryingThermal`), see [below](#gemharris) | `magneticReconnection/inpLe2DGEMHarris.inp` |
| `ForceFree` | Uniform density, Harris $B_x$ and $B_z = B_{0z}/\cosh((y-L_y/2)/\delta)$, force-free when $B_{0z}=B_{0x}$ and $B_{0y}=0$, plus a GEM-type perturbation of flux amplitude $B_{0x}/10$. $\mathbf J=(c/4\pi)\nabla\times\mathbf B$ is split among the species by `u0`, `v0`, `w0`; particles drift with $J_s/(\operatorname{sign}(q_s)\rho_s)$ | `magneticReconnection/testForceFree2D.inp` |
| `Dipole` | 3D planetary magnetosphere: a dipole with equatorial field `B1z` at `Planet_radius` R ($B\propto B_{1z}(R/r)^3$), an external field that is zero inside the planet, on top of the uniform $\mathbf B_0$; the initial E is [E0](boundary-conditions.md#boundary-codes) everywhere. See the [planet surface](boundary-conditions.md#planet-surface-dipole-dipole2d) | `magnetosphere/testMagnetosphere3Dsmall.inp` |
| `Dipole2D` | The same in the $xz$-plane, $B\propto B_{1z}(R/r)^2$ | `magnetosphere/Magnetosphere2D.inp`, `magnetosphere/testMagnetosphere2Dsmall.inp` |
| `NullPoints` | Periodic null-point field $B_x=-B_{0x}(L_x/L_y)\cos(2\pi x/L_x)\sin(2\pi y/L_y)$, $B_y=B_{0x}\cos(2\pi y/L_y)[\sin(2\pi x/L_x)-2\sin(2\pi z/L_z)]$, $B_z=2B_{0x}(L_z/L_y)\sin(2\pi y/L_y)\cos(2\pi z/L_z)$, divergence-free for any box. Species 0 carries $(c/4\pi)\nabla\times\mathbf B$ and species 1 is at rest, so `ns` must be at least 2 (not checked at startup) | `NullPoints.inp` |
| `TaylorGreen` | Every species flows with $\mathbf u=u_0(\sin x\cos y\cos z,\,-(L_y/L_x)\cos x\sin y\cos z,\,0)$ in $\mathbf B=B_{0x}(\cos x\sin y\sin z,\,\sin x\cos y\sin z,\,-(L_z/L_x+L_z/L_y)\sin x\sin y\cos z)$, with `u0` of species 0 and coordinates scaled by $2\pi/L$. Both are divergence-free; the flow carries no net current when $\sum_s\operatorname{sign}(q_s)$`rhoINIT`$_s=0$ | `turbulence/TaylorGreen.inp` |
| `RandomCase` | Divergence-free in-plane field $\sum B_{0x}(-k_y,k_x)\cos(\mathbf k\cdot\mathbf x+\phi_{mn})$ over $\mathbf k=2\pi(m/L_x,n/L_y)$, $\lvert m\rvert,\lvert n\rvert\le3$, with fixed pseudo-random phases, plus a uniform $B_{0z}$; uniform density | |
| `HumpPert` | Uniform density; each component of $\mathbf B_0$ is multiplied by $1+\tfrac12\operatorname{sech}^2\operatorname{sech}^2$ of the two transverse coordinates from the domain midpoint, with width `delta`. `pertHump`, `deltaxHump` and `deltayHump` are **not** used here | |
| `BATSRUS` | Coupling with the BATS-R-US MHD code. Needs `USE_BATSRUS=ON`. Field initialisation works, but **GPU particle initialisation aborts and restarts are not supported**. Without the flag it behaves as an unknown case | |

Any other `Case` prints a warning and uses the default initialisation: uniform density, constant $\mathbf B$, zero $\mathbf E$.

## GEMHarris

- With `currentFromAmpere = 1`:
  - the Harris current $J_z=-(c/4\pi)(B_{0x}/\delta)\operatorname{sech}^2$ is split by the charge-sign-weighted `w0` fractions, a uniform drift per species;
  - the current of the perturbations, $(c/4\pi)\nabla\times\delta\mathbf B$, is carried by the drifting electrons in proportion to their density, and the run stops at startup if no electron species drifts;
  - a hump wider than about the sheet would need drifts above $c$ where the sheet density vanishes, and the run stops at startup.
- The Harris equilibrium needs drifts in the ratio $u_e/u_i=-T_e/T_i$. With `currentFromAmpere = 1`, the run warns when the `w0` split differs.
- The GEM perturbation fills the box, so `pertGEM` needs `currentFromAmpere = 0`.

## Case-specific keys

| Key | Default | Used by | Description |
|---|---|---|---|
| `delta` | `0.5` | `GEM`, `GEMnoPert`, `GEMDoubleHarris`, `ForceFree`, `GEMHarris`, `HumpPert` | Sheet half-thickness |
| `pertGEM` | `0.0` | `GEMHarris` | GEM flux perturbation (Birn et al. 2001): $\psi=$ `pertGEM` $B_{0x}L_x/(2\pi)\cos(2\pi x/L_x)\cos(\pi y/L_y)$ from the box centre, so $\delta B_y$ peaks at `pertGEM` $B_{0x}$ and vanishes on the $y$ faces |
| `pertHump` | `0.0` | `GEMHarris` | Hump flux $A=(\texttt{pertHump}\,B_{0x}/k_y)\,e^{-x^2/\Delta_x^2-y^2/\Delta_y^2}\cos(k_xx)\cos(k_yy)$ from the box centre |
| `deltaxHump`, `deltayHump` | `8.0`, `4.0` | `GEMHarris` | Hump widths $\Delta_x$, $\Delta_y$ in units of `delta` |
| `kxHump`, `kyHump` | `-1.0` | `GEMHarris` | Hump wave numbers; a negative value gives $2\pi/L_x$ and $\pi/L_y$. `kyHump` must not be 0 (not checked at startup) |
| `currentFromAmpere` | `0` | `GEMHarris` | `1`: the Harris current split by the `w0` weights plus the perturbation current $(c/4\pi)\nabla\times\delta\mathbf B$, see [above](#gemharris); `0`: the uniform drifts `u0`, `v0`, `w0` |
| `spatiallyVaryingThermal` | `0` | `GEMHarris` | `1`: sheet species get $T_s e^{-2A_p/(B_{0x}\delta)}$, with $A_p$ the flux function of the perturbations ($B_x=\partial_yA$, $B_y=-\partial_xA$), so their pressure follows the Harris law $p_0e^{-2A/(B_{0x}\delta)}$ along the perturbed field lines; the background stays isothermal. Needs `currentFromAmpere = 1`, `B0x` ≠ 0 and `B0y` = 0 |
| `u0`, `v0`, `w0` | | `ForceFree`; `GEMHarris` with `currentFromAmpere = 1` (`w0` only) | Current weights: species $s$ carries the share $\operatorname{sign}(q_s)u_{0,s}/\sum_r\operatorname{sign}(q_r)u_{0,r}$ of $J_x$ ($v_0$ for $J_y$, $w_0$ for $J_z$). A component whose weights sum to 0 gets no current |
| `B1z` | `0.0` | `Dipole`, `Dipole2D` | Dipole strength: the equatorial field at `Planet_radius` |
| `x_center_dipole`, `y_center_dipole`, `z_center_dipole` | `5.0` | `Dipole`, `Dipole2D` | Dipole centre |

In every case, the run stops at startup if a loaded drift reaches $c$.

## Wall forcing

The GEM and ForceFree wall fixes force B on the non-periodic $y$ faces and apply only with `ForceWallB = 1`. By default (`0`) the $y$ faces keep only their [boundary conditions](boundary-conditions.md#boundary-codes), and B there evolves by Faraday's law. The forcing is an external source of energy and reconnected flux: in a 2D GEM it gave +13 % to +19 % total energy and a flux of 2.64 against 2.21 B0 d_i by tΩci = 19.

| `Case` | With `ForceWallB = 1` |
|---|---|
| `GEM`, `GEMnoPert`, `GEMDoubleHarris` | On the three cell and node planes next to each non-periodic $y$ face, $B_x=B_{0x}\tanh((y_g-L_y/2)/\delta)$, with $y_g$ the centre of the ghost cell, and $B_z=B_{0z}$. On the ghost plane also $B_y=B_{0y}$, except on a conductor, whose rule for the normal $B_y$ comes after it |
| `ForceFree` | On the three cell planes next to each non-periodic $y$ face, $B_z=B_{0z}/\cosh((y-L_y/2)/\delta)$ at each plane's centre. On the ghost plane also $B_x=B_{0x}\tanh((y-L_y/2)/\delta)$ and $B_y=B_{0y}$, except on a conductor |

`Dipole` and `Dipole2D` use the [planet surface](boundary-conditions.md#planet-surface-dipole-dipole2d) rules. The other cases have no runtime boundary fix.
