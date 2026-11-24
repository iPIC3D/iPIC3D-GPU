# Units and normalisation

[Documentation index](../README.md#documentation)

## Unit system

iPIC3D uses **normalised Gaussian CGS** units, based on ion plasma parameters. For the default `c = 1`:

| Quantity | Normalisation | Value |
|---|---|---|
| Length | Ion inertial length $d_i=c/\omega_{pi}$ | 1 |
| Time | $\omega_{pi}^{-1}$ | 1 |
| Speed | Speed of light $c$ | 1 |
| Mass | Ion mass $m_i$ | 1 |
| Charge | Elementary charge $e$ | 1 |
| Energy | $m_ic^2$ | 1 |
| Magnetic field | $B_{\text{ref}}$, see [Dimensional fields](#dimensional-fields) | |

The normalisation $\omega_{pi}^2=4\pi n_0e^2/m_i=1$ gives a background **number density** $n_0=1/(4\pi)\approx0.0796$.

The equations are the Gaussian Maxwell equations, with `FourPI` explicit in the code:

$$\nabla\times\mathbf B=\frac{4\pi}{c}\mathbf J+\frac1c\frac{\partial\mathbf E}{\partial t},\quad \nabla\times\mathbf E=-\frac1c\frac{\partial\mathbf B}{\partial t},\quad \nabla\cdot\mathbf E=4\pi\rho,\quad \nabla\cdot\mathbf B=0$$

### Speed of light input c

Any `c > 0` works: the field solve, both movers and the drift field $\mathbf E_0=-\mathbf u_0\times\mathbf B_0/c$ take $c$ as a parameter. `rhoINIT` and `qom` alone fix $\omega_{pi}=1$, so time stays in $\omega_{pi}^{-1}$. The input `c` sets the speed unit $V_0=c_{\text{phys}}/c$:

| Quantity | Unit for any `c` |
|---|---|
| Speed | $V_0=c_{\text{phys}}/c$ |
| Length | $V_0/\omega_{pi}=d_i/c$ |
| Time | $\omega_{pi}^{-1}$ |
| Energy | $m_iV_0^2$ |
| E, B | $B_{\text{ref}}=\sqrt{4\pi n_{\text{ref}}m_iV_0^2}$ |

The ion inertial length is therefore $c$ code lengths. Thermal and drift speeds, `dt`, the box and the fields in the input are all in these units. Equilibria involving $c$ must be set for the chosen value, such as the Harris drifts $\sum_sq_sn_sw_{0,s}=-cB_{0x}/(4\pi\delta)$.

### Dimensional fields

Code values are ratios $B_{\text{code}}=B_{\text{CGS}}/B_{\text{ref}}$, with $E_{\text{ref}}=B_{\text{ref}}$ since E and B share dimensions in Gaussian CGS. Here $n_{\text{ref}}$ [cm⁻³] is the physical ion density for `rhoINIT = 1`, $m_i$ [g] the ion mass and $V_0$ [cm/s] the speed unit. For example, `B0x = 0.0195` means $0.0195\,B_{\text{ref}}$ in Gauss.

## Input parameters

| Key | Meaning |
|---|---|
| `rhoINIT` | $4\pi n_s$, an unsigned number-density magnitude; the code seeds the density `rhoINIT`/4π |
| `rhoINJECT` | $4\pi n_{\text{inject},s}$, unsigned |
| `B0x`, `B0y`, `B0z` | Background $\mathbf B_0$; sets the [Alfvén speed](#alfvén-speed) |
| `uth`, `vth`, `wth` | One-component thermal speeds $\sqrt{kT/m}$ |
| `u0`, `v0`, `w0` | Drifts per species. Species 0 also sets the `TaylorGreen` amplitude and $\mathbf E_0$ ([field inflow faces](boundary-conditions.md#boundary-codes), initial E of `Dipole` and `Dipole2D`) |
| `qom` | $q/m$, sign included. The species sign enters through it when particle charges are assigned |

For `uth = vth = wth` and no drift, the Maxwellian sampler, which scales unit Gaussians by the thermal speeds, gives the distribution of the speed $u$:

$$F\,du=4\pi\left(\frac{1}{2\pi u_{th}^2}\right)^{3/2}u^2e^{-u^2/(2u_{th}^2)}N_s\,du,\quad N_s=\texttt{npcelx}\cdot\texttt{npcely}\cdot\texttt{npcelz}$$

## Alfvén speed

In Gaussian units $v_A=B/\sqrt{4\pi\rho_m}$, with $\rho_m=\sum_sm_sn_s$ the mass density. The charge density of a species is `rhoINIT`/4π (its number density for unit charge), and its mass density is that charge density divided by |`qom`|. Hence, in code units:

$$v_A=\frac{B}{\sqrt{\sum_s\texttt{rhoINIT}_s/\lvert\texttt{qom}_s\rvert}}$$

- $B$ is the field magnitude, or the component of interest, such as the reconnecting `B0x` of a Harris sheet.
- The result is in the speed unit $V_0$ ($c$ for `c = 1`), so $v_A/c_{\text{phys}}$ is the code value divided by the input `c`.
- For ions alone with `rhoINIT = 1` and `qom = 1`, $v_A=B$. This is why the input $B_0$ is often read as the Alfvén speed. The ion cyclotron frequency is $\Omega_{ci}=\texttt{qom}\,B/c$, so $v_A/\Omega_{ci}=d_i$.
- Electrons add their density divided by their |`qom`|: with `qom = -64` they add 1/64 of the ion mass density.
- The formula assumes $v_A\ll c$. Near $c$, the wave speed is $v_A/\sqrt{1+v_A^2/c^2}$.

| Plasma | Input | $v_A$ |
|---|---|---|
| Uniform ions and electrons | `rhoINIT = 1 1`, `qom = -64 1`, `B0z = 0.0195` | $0.0195/\sqrt{1+1/64}\approx0.0193$ |
| Lobes of `testGEM2Dsmall.inp` (background species 2 and 3; the sheet species vanish there) | `rhoINIT = 0.1 0.1`, `qom = -64 1`, `B0x = 0.0195` | $0.0195/\sqrt{0.1+0.1/64}\approx0.0612$ |

**Local value from the output.** With the species densities written (`MomentsOutputTag = rho`) and the field (`FieldOutputTag = B`), at every node:

$$v_A=\frac{\lvert\mathbf B\rvert}{\sqrt{\sum_s\lvert\rho_{s,\text{disk}}\rvert/\lvert q/m\rvert_s}}$$

The written $\rho_{s,\text{disk}}$ already carries the factor 4π.

## Internal storage

Moments are stored in the physical normalised units above. After deposition, `rho` is a **signed** charge density.

**Particle charge.** The loader uses $q=\operatorname{sign}(q/m)\,\Delta V\,\rho_{cs}/N_s$, with $\rho_{cs}$ the seeded density at the cell centre. Boundary injection uses $q=\operatorname{sign}(q/m)(\texttt{rhoINJECT}/4\pi/N_s)\,\Delta V$.

**Deposit.** Each particle adds to the 8 nodes of its cell, with $S_p\in[0,1]$ its trilinear weight on the node:

$$\rho_s=\sum_pq_pS_p/\Delta V,\quad \mathbf J_s=\sum_pq_p\mathbf v_pS_p/\Delta V,\quad P_{ij,s}=\sum_pq_pv_{i,p}v_{j,p}S_p/\Delta V$$

No scaling is applied between the deposit and the solver, and the species sums are plain sums.

**4π in the field solver** ($\delta$ = `c * th * dt`, $\Omega=(q/m)\Delta t\,B/(2c)$):

| Term | Formula |
|---|---|
| Ampère source | $-(4\pi/c)\hat J$ |
| Poisson source | $-\delta^2\,4\pi\nabla\hat\rho$ |
| Susceptibility | $(4\pi/2)\,\delta\,(\Delta t/c)(q/m)\rho_s/(1+\Omega^2)$ |
| ∇·E correction | $\nabla\cdot\mathbf E-4\pi\rho_c$ |
| Hat density | $\hat\rho=\rho_c-\Delta t\,\theta\,\nabla\cdot\hat J$, no $4\pi$ (continuity equation) |
| Rotation | $1/(1+\Omega^2)$, no $4\pi$ |

## Written quantities

Everything is written as stored, except the **charge density, multiplied by 4π** on output to match the input convention. Checkpoints store the unscaled $\rho_s$.

- The written $\rho_s$ and $\rho_{\text{total}}$ are signed, unlike the unsigned `rhoINIT`.
- The written B includes any external field, such as a planet's dipole.
- $P_{ij,s}$ is charge-weighted: it carries the charge sign, includes the bulk flow and is not mass-weighted.
- The heat flux $Q_{ijk,s}$ is mass-weighted and in the species' bulk frame, $\sum_p(q_p/(q/m)_s)\,c_ic_jc_k\,S_p/\Delta V$ with $\mathbf c=\mathbf v_p-\mathbf J_s/\rho_s$, and 0 at nodes where $\lvert\rho_s\rvert$ is below a floor. Do not divide it by $q/m$.
- Particle positions are in $d_i/c$, velocities in $V_0$, and the charge carries the $1/(4\pi)$ factor.

## Derived quantities

Use $\rho_{\text{phys}}=\rho_{\text{disk}}/(4\pi)$ wherever density appears.

| Quantity | Formula with on-disk values | Note |
|---|---|---|
| Bulk velocity | $\mathbf V_s=\mathbf J_s/(\rho_{s,\text{disk}}/4\pi)$ | |
| Hall E | $-\mathbf J\times\mathbf B/(c\,\rho_{e,\text{disk}}/4\pi)$ | Signed electron density |
| E·J dissipation | $\mathbf E\cdot\mathbf J$ | |
| Momentum flux | $P_{\text{disk}}/(q/m)_s$ | |
| Thermal pressure tensor | $(P_{ij,\text{disk}}-J_{i,s}J_{j,s}/\rho_s)/(q/m)_s$ | $\rho_s=\rho_{s,\text{disk}}/4\pi$ |
| Species gas pressure | $p_s=v_{th,s}^2\lvert\rho_{s,\text{disk}}\rvert/(4\pi\lvert q/m\rvert_s)$ | One velocity component |
| Total gas pressure | $p_{\text{gas}}=\sum_sp_s$ | |
| Magnetic pressure | $p_B=B^2/(8\pi)$ | |
| Plasma beta | $\beta=8\pi p_{\text{gas}}/B^2$ | |
| Gauss's law | $\nabla\cdot\mathbf E=\rho_{\text{disk}}$ | $\rho_{\text{disk}}$ already is $4\pi\rho_{\text{phys}}$ |
