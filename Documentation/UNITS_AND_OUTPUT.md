# iPIC3D-GPU: Units, Normalization Convention and Output Data Reference

## Table of Contents

- [1. Unit System](#1-unit-system)
- [2. Maxwell's Equations (Gaussian CGS Form)](#2-maxwells-equations-gaussian-cgs-form)
- [3. Internal Storage Convention](#3-internal-storage-convention)
  - [3.1 Density Initialization](#31-density-initialization)
  - [3.2 Particle Weight](#32-particle-weight)
  - [3.3 GPU Moment Accumulation](#33-gpu-moment-accumulation)
  - [3.4 Moments on the GPU](#34-moments-on-the-gpu)
  - [3.5 Species Summation](#35-species-summation)
- [4. Field Solver: Gaussian 4π Factors](#4-field-solver-gaussian-4π-factors)
- [5. Input File Parameters](#5-input-file-parameters)
- [6. Output Backends](#6-output-backends)
- [7. Output Data Convention](#7-output-data-convention)
  - [7.1 Fields](#71-fields)
  - [7.2 Charge Density](#72-charge-density)
  - [7.3 Current Density](#73-current-density)
  - [7.4 Pressure Tensor](#74-pressure-tensor)
  - [7.5 Particles](#75-particles)
  - [7.6 Energy Diagnostics](#76-energy-diagnostics)
- [8. PVTK Output Files](#8-pvtk-output-files)
  - [8.1 Field Output (FieldOutputTag)](#81-field-output-fieldoutputtag)
  - [8.2 Moments Output (MomentsOutputTag)](#82-moments-output-momentsoutputtag)
- [9. PHDF5 Output Files](#9-phdf5-output-files)
- [10. SHDF5 Output Files](#10-shdf5-output-files)
- [11. Derived Quantities and Usage Notes](#11-derived-quantities-and-usage-notes)
  - [11.1 Correct Formulas Using On-Disk Values](#111-correct-formulas-using-on-disk-values)
  - [11.2 Quick Reference for Conversions](#112-quick-reference-for-conversions)
- [12. Summary Table](#12-summary-table)

---

## 1. Unit System

iPIC3D uses **normalized Gaussian CGS** units. All quantities are normalized to
ion plasma parameters:

| Quantity         | Normalization                                  | Value |
|------------------|------------------------------------------------|-------|
| Length           | Ion inertial length, $d_i = c/\omega_{pi}$     | 1     |
| Time             | Inverse ion plasma frequency, $\omega_{pi}^{-1}$| 1     |
| Speed            | Speed of light, $c$                            | 1     |
| Mass             | Ion mass, $m_i$                                | 1     |
| Charge           | Elementary charge, $e$                         | 1     |
| Magnetic field   | $B_0$ in input file = Alfvén speed $v_A$       | —     |
| Energy           | $m_i c^2$                                      | 1     |

The key consequence of the $\omega_{pi}$ normalization is:

$$\omega_{pi}^2 = \frac{4\pi\, n_0\, e^2}{m_i} = 1 \quad \Longrightarrow \quad n_0 = \frac{1}{4\pi} \approx 0.0796$$

This means the physical background **number density** in normalized units is
$1/(4\pi)$, not 1. The input file parameters `rhoINIT` and `rhoINJECT` are
convenience quantities equal to $4\pi \times n_s$ for each species. They are
input number-density magnitudes, not signed charge densities; the species sign
enters later through `qom` when particle charges are assigned and moments are
deposited.

---

## 2. Maxwell's Equations (Gaussian CGS Form)

iPIC3D solves the full Gaussian CGS Maxwell's equations:

$$\nabla \times \mathbf{B} = \frac{4\pi}{c}\mathbf{J} + \frac{1}{c}\frac{\partial \mathbf{E}}{\partial t}$$

$$\nabla \times \mathbf{E} = -\frac{1}{c}\frac{\partial \mathbf{B}}{\partial t}$$

$$\nabla \cdot \mathbf{E} = 4\pi\rho$$

$$\nabla \cdot \mathbf{B} = 0$$

The constant `FourPI = 16 * atan(1.0)` $\approx 12.5664$ is a member of
`EMfields3D`, set in its constructor, and appears explicitly wherever Gaussian
Maxwell's equations require it.

---

## 3. Internal Storage Convention

All particle moments (ρ, J, pressure tensor) are stored in the physical
normalized units defined in [Section 1](#1-unit-system). Internally, densities
are not kept in the `rhoINIT`/`rhoINJECT` input scaling. After particle loading
and deposition, `rho` arrays represent signed charge density. The special case
is the input convention: `rhoINIT` and `rhoINJECT` themselves are unsigned
species number-density magnitudes.

### 3.1 Density Initialization

The case initialisers seed the species density from `rhoINIT`, e.g. the
uniform background in `EMfields3D::initGEM()`:

```cpp
rhons[is][i][j][k] = rhoINIT[is] / FourPI;   // physical density
```

(the drifting Harris species get `rhoINIT[is] / FourPI` times their
$\operatorname{sech}^2$ profile). The input `rhoINIT` is divided by $4\pi$ to
obtain the seeded species number density magnitude on the grid. The charge sign is not encoded in `rhoINIT`; it
is applied later through `qom` when particle weights are formed.

### 3.2 Particle Weight

In `ParticleSoAHost::maxwellian()`:

```cpp
chargeFactor = sign(qom) * VOL / npcel;
q = chargeFactor * EMf->getRHOcs(i, j, k, species);
```

`getRHOcs()` returns the center-interpolated seeded density magnitude. The
factor `sign(qom)` applies the species sign, so the particle charge `q` and the
deposited `rho_s` become signed charge-density quantities.

For the boundary injection (`ParticleCommInjection`, used by the GPU injection
kernel), `rhoINJECT` is explicitly divided by `FourPI`:

```cpp
chargePerParticle = sign(qom) * (rhoINJECT / FourPI / npcel) * VOL;
```

### 3.3 GPU Moment Accumulation

In `depositParticleMomentsToNodes()` (momentKernel.cu, used by the moment
kernels), the particle weight includes `invVOL` **twice**:

```cpp
invVOLqi = grid->invVOL * qi;
weights[c] = invVOLqi * xi * eta * zeta * grid->invVOL;
```

- First `invVOL`: converts particle charge to density (charge/volume).
- Second `invVOL`: normalization of the tri-linear shape function.

The resulting accumulated moments are:

$$\rho_s = \sum_p q_p \cdot W_p \cdot (\Delta V)^{-2}$$

$$\mathbf{J}_s = \sum_p q_p \cdot \mathbf{v}_p \cdot W_p \cdot (\Delta V)^{-2}$$

$$P_{ij,s} = \sum_p q_p \cdot v_{i,p} \cdot v_{j,p} \cdot W_p \cdot (\Delta V)^{-2}$$

These accumulated moments remain in the same internal physical units used
elsewhere in the code.

The velocity moments array stores:
- `velmoments[0] = 1` → charge density $\rho_s$
- `velmoments[1] = u` → current $J_{x,s}$
- `velmoments[2] = v` → current $J_{y,s}$
- `velmoments[3] = w` → current $J_{z,s}$
- `velmoments[4..9] = uu, uv, uw, vv, vw, ww` → pressure tensor $P_{ij,s}$

### 3.4 Moments on the GPU

The moments stay on the GPU: `gpuScatterMomentsD2D` copies them
device-to-device into the field solver's per-species arrays. `gpuSolverSyncD2H`
copies them to the host arrays (`rhons[is]`, `Jxs[is]`, `Jys[is]`, `Jzs[is]`,
`pXXsn`, `pXYsn`, ...) only when output, diagnostics, a restart or Catalyst need
them.
**No scaling is applied** on either path.

### 3.5 Species Summation

`gpuSumOverSpecies()` and `gpuSumOverSpeciesJ()` on the GPU, and the host
`sumOverSpeciesJ()` called by the PHDF5 and SHDF5 writers, are **plain sums**
over species indices. No $4\pi$ factor is applied.

---

## 4. Field Solver: Gaussian 4π Factors

The implicit field solver uses the physical CGS moments with explicit $4\pi$
factors from Gaussian Maxwell's equations:

| Physics                | Location                  | Code formula                                      |
|------------------------|---------------------------|--------------------------------------------------|
| Ampère source          | `gpuMaxwellSource()`      | `-FourPI/c * Ĵ`                                  |
| Poisson source         | `gpuMaxwellSource()`      | `-(cθΔt)² × FourPI × ∇ρ̂`                        |
| Susceptibility (MUdot) | `gpuMUdot()`              | `FourPI/2 × cθΔt × Δt/c × (q/m) × ρ_s / (1+Ω²)` |
| Divergence cleaning    | `gpuPoissonCorrection()`  | `div(E) - FourPI × ρ_c`                          |
| Hat functions          | `gpuCalculateHatFunctions()` | `ρ̂ = ρ_c - Δt·θ·div(Ĵ)` — no FourPI (consistent) |
| Rotation (PIdot)       | `gpuPIdot()`              | `1/(1+Ω²)` — no FourPI (pure rotation)           |

`cθΔt` is the code's `delt = c * th * dt`.

The hat function `ρ̂ = ρ_c - dt·θ·div(Ĵ)` has **no** explicit $4\pi$ because
both $\rho_c$ and $\mathbf{J}$ are in the same physical units; the continuity
equation $\partial\rho/\partial t + \nabla\cdot\mathbf{J} = 0$ has no $4\pi$.

---

## 5. Input File Parameters

Key parameters and their relationship to physical units:

| Input parameter  | Physical meaning                           | Internal value            |
|------------------|--------------------------------------------|---------------------------|
| `rhoINIT`        | $4\pi \times n_s$ (species number-density magnitude) | Seeds `rhons = rhoINIT / 4π` before deposition |
| `rhoINJECT`      | $4\pi \times n_{\text{inject},s}$ (injected number-density magnitude) | `q ∝ rhoINJECT / 4π`, sign from `qom` |
| `B0x, B0y, B0z`  | Physical $\mathbf{B}_0$ (= Alfvén speed)   | Used directly             |
| `uth, vth, wth`  | Thermal speeds $v_{th} = \sqrt{kT/m}$      | Used directly             |
| `u0, v0, w0`     | Drift velocities (per species); current-split weights for `ForceFree` and `GEMHarris` with `currentFromAmpere = 1`; for species 0 also the `TaylorGreen` flow amplitude and $E_0 = -\mathbf{u}_0\times\mathbf{B}_0$ (open faces, `Dipole`) | Used directly             |
| `qom`            | Charge-to-mass ratio $q/m$ (sign matters)  | Used directly             |
| `dt`             | Time step in $\omega_{pi}^{-1}$            | Used directly             |
| `Lx, Ly, Lz`    | Box size in $d_i$                          | Used directly             |
| `c`              | Speed of light (should be 1.0)             | Used directly             |

`rhoINIT` and `rhoINJECT` set density magnitudes only. They do not encode the
sign of the species charge.

---

## 6. Output Backends

iPIC3D-GPU supports multiple field output backends, selected by `WriteMethod`
in the input file:

| WriteMethod  | Format                    | Field output   | Moments output | Total ρ          |
|--------------|---------------------------|----------------|----------------|------------------|
| `pvtk`       | Parallel VTK (MPI-IO)     | ✓             | ✓             | ✓ (× 4π)        |
| `nbcvtk`     | Non-blocking VTK (MPI-IO) | ✓             | ✓             | ✓ (× 4π)        |
| `phdf5`      | Parallel HDF5             | ✓             | ✓ (same file)  | ✓ (× 4π)        |
| `shdf5`      | Serial HDF5 (per-process) | ✓             | ✓             | ✓ (× 4π)        |
| `adios2`     | ADIOS2 format             | ✗ (throws)    | ✗             | ✗               |

Output is controlled by two tag strings in the input file:
- **`FieldOutputTag`**: grid fields (`B`, `E`), the species currents `Je`/`Ji`/`Je2`/`Ji3` (species 0–3) and `rho` (the total charge density)
- **`MomentsOutputTag`**: moments (e.g. `"rho+PXX+PXY+PXZ+PYY+PYZ+PZZ"`)

Tags are separated by `+`, trimmed and matched exactly against the tokens listed
in the README; an unknown token prints a warning and is ignored.

---

## 7. Output Data Convention

Fields, currents, pressure tensor components, heat flux and particle
quantities are written exactly as stored internally. Charge density is the
exception: the species and total densities are multiplied by `4π` on output
to match the input-file convention.

### 7.1 Fields

| Field | Internal value | On-disk value | Units |
|-------|---------------|---------------|-------|
| $E_x, E_y, E_z$ | Physical $\mathbf{E}$ | Physical $\mathbf{E}$ | normalized CGS |
| $B_x, B_y, B_z$ | Physical $\mathbf{B}$ | Physical $\mathbf{B}$ | normalized CGS |

**What "physical normalized CGS" means for E and B:**

In Gaussian CGS, E and B share dimensions (statV/cm = Gauss). Code values are
dimensionless ratios $B_{\text{code}} = B_{\text{CGS}} / B_{\text{ref}}$, where
the dimensional reference quantities are:

$$B_{\text{ref}} = \sqrt{4\pi\, n_{\text{ref}}\, m_i\, c^2}, \qquad E_{\text{ref}} = B_{\text{ref}}$$

with $n_{\text{ref}}$ [$\text{cm}^{-3}$] the physical ion density for `rhoINIT = 1`,
$m_i$ [g] the ion mass, and $c$ [cm/s] the speed of light.
$E_{\text{ref}} = B_{\text{ref}}$ because E and B share dimensions in Gaussian CGS
and the velocity normalization is $c$. In the fully normalized system
($n_0 = 1/(4\pi)$, $m_i = 1$, $c = 1$) these reduce to $B_{\text{ref}} = E_{\text{ref}} = 1$,
so code values are directly the dimensionless field strengths. For example,
`B0x = 0.0195` means $B_{0x} = 0.0195 \times B_{\text{ref}}$ in Gauss.

Because $n_0 = 1/(4\pi)$, the Alfvén speed simplifies to $v_A = B_0$ (see
[Section 11](#11-derived-quantities-and-usage-notes) for derivation). No
additional scaling is applied on write.

Note: B output uses `getBxTot/getByTot/getBzTot` which includes any external
field components.

### 7.2 Charge Density

**ρ is stored on disk as a signed charge density.**

| Density | Internal value | On-disk value | Scale |
|---------|---------------|---------------|-------|
| $\rho_s$ (per-species) | signed $\rho_{\text{phys}} = q_s n_s$ | $4\pi \times \rho_{\text{phys}}$ | × 4π |
| $\rho_{\text{total}}$ | $\sum_s \rho_{s,\text{phys}}$ | $4\pi \times \sum_s \rho_{s,\text{phys}}$ | × 4π |

Restart files store the unscaled $\rho_s$.

This uses the same numerical $4\pi$ scaling as `rhoINIT`, but it is not the
same physical quantity: `rhoINIT` is an unsigned input number-density
magnitude, while the written `rho` datasets are signed deposited charge
density.

### 7.3 Current Density

| Current | Internal value | On-disk value | Scale |
|---------|---------------|---------------|-------|
| $J_{x,s}, J_{y,s}, J_{z,s}$ | $J_{\text{phys}}$ | $J_{\text{phys}}$ | none |
| $J_x, J_y, J_z$ (total) | $J_{\text{phys}}$ | $J_{\text{phys}}$ | none |

### 7.4 Pressure Tensor

| Tensor component | Internal value | On-disk value | Scale |
|-----------------|---------------|---------------|-------|
| $P_{xx}, P_{xy}, \ldots, P_{zz}$ | $\sum_p q_p v_{i,p} v_{j,p} W_p (\Delta V)^{-2}$ | same | none |

The stored tensor is the charge-weighted second velocity moment: it carries the
sign of the species charge, includes the bulk flow and is not mass-weighted.
The momentum-flux tensor is $P_{\text{disk}}/(q/m)_s$, and the thermal pressure
tensor is $\bigl(P_{ij,\text{disk}} - J_{i,s}J_{j,s}/\rho_s\bigr)/(q/m)_s$ with
$\rho_s = \rho_{s,\text{disk}}/4\pi$.

See [Section 3.3](#33-gpu-moment-accumulation) for the accumulation formula
and [Section 11](#11-derived-quantities-and-usage-notes) for pressure calculations
from on-disk values.

### 7.5 Particles

Particle output (when `ParticlesOutputCycle > 0` and `ParticlesOutputTag` is
not empty) writes:

| Quantity | Tag | Description | Units |
|----------|-----|-------------|-------|
| Position | `position` | $x, y, z$ | $d_i$ |
| Velocity | `velocity` | $u, v, w$ | $c$ |
| Charge | `q` | Particle charge weight | physical (carries $1/(4\pi)$ factor) |
| ID | `ID` | Particle tracking ID (species with `TrackParticleID` only) | integer |

### 7.6 Energy Diagnostics

Every `DiagnosticsOutputCycle` cycles, the last MPI rank (nprocs − 1) appends a
row to `SaveDirName/ConservedQuantities.txt` with the columns `Cycle`,
`Total_Energy`, `Momentum`, `Eenergy`, `Benergy`, `Kenergy`, `Total_Charge`,
then `Ke[0…ns−1]`, `BulkE[0…ns−1]`, `Mom[0…ns−1]` and `Q[0…ns−1]`.

`Eenergy`, `Benergy` (with the external field) and `BulkE` are node sums
$\sum \tfrac12 \Delta V\,|\mathbf{E}|^2/4\pi$, $\sum \tfrac12 \Delta V\,|\mathbf{B}|^2/4\pi$ and
$\sum \tfrac12 \Delta V\,|\mathbf{J}_s|^2/(\rho_s\,q_s/m_s)$. Each rank sums the nodes it
owns (`EMfields3D::ownedNodeEnds`): the node shared with the next rank, or with
the periodic image, belongs to that neighbour, and on a non-periodic direction
the last rank also counts its boundary node, so every node is counted once.

---

## 8. PVTK Output Files

### 8.1 Field Output (FieldOutputTag)

Available tags and output files:

| Tag | Filename pattern | Format | Data | Scaling |
|-----|-----------------|--------|------|---------|
| `B` | `{SimName}_B_{cycle}.vtk` | 3-component vector | $B_{x,y,z}^{\text{tot}}$ | none |
| `E` | `{SimName}_E_{cycle}.vtk` | 3-component vector | $E_{x,y,z}$ | none |
| `Je`, `Ji`, `Je2`, `Ji3` | `{SimName}_J0_{cycle}.vtk` … `{SimName}_J3_{cycle}.vtk` | 3-component vector | $J_{x,y,z}$ of species 0, 1, 2, 3 | none |
| `rho` | `{SimName}_rho_tot_{cycle}.vtk` | scalar | total charge density | × 4π |

Example input: `FieldOutputTag = B+E+Je+Ji+rho`

VTK files are binary STRUCTURED_POINTS format with grid dimensions matching
`(nxc+1) × (nyc+1) × (nzc+1)` (node-centered, excluding ghost cells).

### 8.2 Moments Output (MomentsOutputTag)

A bare token writes every species, a token with a species index (`rho0`,
`PXX2`, `Qxyz1`) one species, and a `_tot` token the sum over species:

| Tag | Filename pattern | Format | Scaling |
|-----|-----------------|--------|---------|
| `rho`, `rho<s>` | `{SimName}_rho{s}_{cycle}.vtk` | scalar | × 4π |
| `rho_tot` | `{SimName}_rho_tot_{cycle}.vtk` | scalar | × 4π |
| `J`, `J<s>` | `{SimName}_J{s}_{cycle}.vtk` | 3-component vector | none |
| `J_tot` | `{SimName}_J_tot_{cycle}.vtk` | 3-component vector | none |
| `PXX` … `PZZ`, `PXX<s>` … | `{SimName}_PXX{s}_{cycle}.vtk` … | scalar | none |
| `PXX_tot` … `PZZ_tot` | `{SimName}_PXX_tot_{cycle}.vtk` … | scalar | none |
| `P`, `P<s>`, `P_tot` | the six components above | scalar | none |
| `Qxxx` … `Qzzz`, `Q`, with `<s>` or `_tot` | `{SimName}_Qxxx{s}_{cycle}.vtk` … | scalar | none |

Example input: `MomentsOutputTag = rho+PXX+PXY+PXZ+PYY+PYZ+PZZ`

---

## 9. PHDF5 Output Files

Parallel HDF5 writes the fields and moments of one cycle into a single file,
`{SaveDirName}/{SimName}_{cycle:05d}.h5`. All datasets are under the
`"Fields"` group, on the global node grid `(nxc+1) × (nyc+1) × (nzc+1)`:

| Dataset | Data source | Scaling |
|---------|-------------|---------|
| `Ex`, `Ey`, `Ez` | Electric field | none |
| `Bx`, `By`, `Bz` | Magnetic field, $\mathbf{B}^{\text{tot}}$ | none |
| `rho_{s}` | Charge density, species `s` | × 4π |
| `rho_tot` | Total charge density | × 4π |
| `Jx_{s}`, `Jy_{s}`, `Jz_{s}` / `Jx_tot` … | Current density, species `s` / total | none |
| `pXX_{s}` … `pZZ_{s}` / `pXX_tot` … | Pressure tensor, species `s` / total | none |
| `Qxxx_{s}` … / `Qxxx_tot` … | Heat-flux tensor, species `s` / total | none |

The datasets written are selected by the same tags as in
[Section 8](#8-pvtk-output-files).

---

## 10. SHDF5 Output Files

Serial HDF5 writes one file per MPI process, `SaveDirName/proc<rank>.hdf`,
with the rank's local nodes. It uses the same `FieldOutputTag` and
`MomentsOutputTag` tokens as the other backends:

| Tag | HDF5 path | Scaling |
|-----|-----------|---------|
| `B` | `/fields/B{x,y,z}/cycle_{n}` ($\mathbf{B}^{\text{tot}}$) | none |
| `E` | `/fields/E{x,y,z}/cycle_{n}` | none |
| `rho`, `rho<s>` (MomentsOutputTag) | `/moments/species_{s}/rho/cycle_{n}` | × 4π |
| `rho` (FieldOutputTag), `rho_tot` | `/moments/rho/cycle_{n}` | × 4π |
| `J`, `J<s>`, `Je` … | `/moments/species_{s}/J{x,y,z}/cycle_{n}` | none |
| `J_tot` | `/moments/J{x,y,z}/cycle_{n}` | none |
| `P`, `PXX` … | `/moments/species_{s}/p{XX,XY,...,ZZ}/cycle_{n}` | none |
| `PXX_tot` … | `/moments/pXX_tot/cycle_{n}` … | none |
| `Q`, `Qxxx` … | `/moments/species_{s}/Qxxx/cycle_{n}`, `/moments/Qxxx_tot/cycle_{n}` | none |

---

## 11. Derived Quantities and Usage Notes

When post-processing output, convert density back to its internal physical form
with $\rho_{\text{phys}} = \rho_{\text{disk}}/(4\pi)$ whenever it appears in a
formula. J, E and B can be used directly; P is the charge-weighted second moment (see
[Section 7.4](#74-pressure-tensor)). Note that `rho_disk` is a signed charge
density, not the unsigned `rhoINIT` input quantity.

### 11.1 Correct Formulas Using On-Disk Values

| Derived quantity | Formula using disk values | Notes |
|-----------------|--------------------------|-------|
| **Bulk velocity** | $\mathbf{V}_s = \mathbf{J}_s / ({\rho} _{s,\text{disk}} / 4\pi)$ | Uses species density |
| **Hall E-field** | $\mathbf{J} \times \mathbf{B} / (\rho_{e,\text{disk}} / 4\pi)$ | Uses electron density |
| **E·J dissipation** | $\mathbf{E} \cdot \mathbf{J}$ | Both physical, use directly |
| **Alfvén speed** | $v_A = B_0 / \sqrt{\lvert \rho_{\text{disk}} \rvert / \lvert q/m \rvert}$ | Uses density magnitude. For ions ($\lvert q/m \rvert=1$, `rhoINIT=1`): $v_A = B_0$ |
| **Species gas pressure** | $p_s = v_{th,s}^2 \times \lvert \rho_{s,\text{disk}} \rvert / (4\pi\,\lvert q/m \rvert_s)$ | Use density magnitude because on-disk $\rho_s$ is signed |
| **Total gas pressure** | $p_{\text{gas}} = \sum_s p_s$ | Sum species contributions |
| **Plasma beta** | $\beta = 8\pi\, p_{\text{gas}} / B^2$ | Equivalent to the usual Gaussian CGS definition |
| **Magnetic pressure** | $p_B = B^2 / (8\pi)$ | Standard Gaussian CGS |
| **Gauss's law check** | $\nabla \cdot \mathbf{E} = \rho_{\text{disk}}$ | $\rho_{\text{disk}}$ already is $4\pi\rho_{\text{phys}}$ |

### 11.2 Quick Reference for Conversions

```
ρ_physical = ρ_disk / (4π)
J_physical = J_disk           (already physical)
E_physical = E_disk           (already physical)
B_physical = B_disk           (already physical)
momentum flux = P_disk / (q/m)_s   (P_disk is charge-weighted)
```

---

## 12. Summary Table

| Quantity | Internal storage | On-disk value | Input convention |
|----------|-----------------|---------------|------------------|
| $\rho_s$ | signed $\rho_s = q_s n_s$ after deposition | $4\pi \times \rho_s$ | `rhoINIT` sets unsigned $4\pi n_s$ input magnitude |
| $\rho_{\text{total}}$ | $\sum_s \rho_s$ | $4\pi \times \sum_s \rho_s$ | — |
| $\mathbf{J}_s$ | $J_{\text{phys}}$ | $J_{\text{phys}}$ | — |
| $\mathbf{E}$ | Physical | Physical | — |
| $\mathbf{B}$ | Physical (incl. external) | Physical | `B0x, B0y, B0z` = physical |
| $P_{ij,s}$ | charge-weighted $\sum q v_i v_j$ | same | — |
| Particle $q$ | Physical | Physical | — |

Only ρ uses the same numerical `4π` scaling as `rhoINIT`; unlike `rhoINIT`, the
written ρ datasets are signed charge densities. J, E, B, P, and particle
weights are written in the internal normalized units.
