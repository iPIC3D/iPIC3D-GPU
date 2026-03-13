# iPIC3D-GPU: Units, Normalization Convention and Output Data Reference

## Table of Contents

- [1. Unit System](#1-unit-system)
- [2. Maxwell's Equations (Gaussian CGS Form)](#2-maxwells-equations-gaussian-cgs-form)
- [3. Internal Storage Convention](#3-internal-storage-convention)
  - [3.1 Density Initialization](#31-density-initialization)
  - [3.2 Particle Weight](#32-particle-weight)
  - [3.3 GPU Moment Accumulation](#33-gpu-moment-accumulation)
  - [3.4 GPU→CPU Transfer](#34-gpucpu-transfer)
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

This means the **physical background charge density** in normalized units is
$1/(4\pi)$, not 1. The input file parameter `rhoINIT` is a convenience quantity
equal to $4\pi \times \rho_{\text{physical}}$ so that the background value is 1.

---

## 2. Maxwell's Equations (Gaussian CGS Form)

iPIC3D solves the full Gaussian CGS Maxwell's equations:

$$\nabla \times \mathbf{B} = \frac{4\pi}{c}\mathbf{J} + \frac{1}{c}\frac{\partial \mathbf{E}}{\partial t}$$

$$\nabla \times \mathbf{E} = -\frac{1}{c}\frac{\partial \mathbf{B}}{\partial t}$$

$$\nabla \cdot \mathbf{E} = 4\pi\rho$$

$$\nabla \cdot \mathbf{B} = 0$$

The constant `FourPI = 16 * atan(1.0)` $\approx 12.5664$ is defined in
`EMfields3D.cpp` (line 288) and appears explicitly wherever Gaussian Maxwell's
equations require it.

---

## 3. Internal Storage Convention

All particle moments (ρ, J, pressure tensor) are stored in the physical
normalized units defined in [Section 1](#1-unit-system). Internally, densities
are not kept in the `rhoINIT`/`rhoINJECT` input scaling.

### 3.1 Density Initialization

In `EMfields3D::initGEM()` (EMfields3D.cpp, line 3283):

```cpp
rhons[is][i][j][k] = rhoINIT[is] / FourPI;   // physical density
```

The input `rhoINIT` is divided by $4\pi$ to obtain the physical charge density
used internally.

### 3.2 Particle Weight

In `Particles3D::maxwellian()` (Particles3D.cpp, line 135):

```cpp
q = sign(qom) * (VOL / npcel) * getRHOcs(i, j, k, ns);
```

`getRHOcs()` returns the center-interpolated physical density, so the particle
charge `q` is consistent with the internal storage convention.

For boundary repopulation (Particles3D.cpp, line 457), `Ninj` (= `rhoINJECT`
from input) is explicitly divided by `FourPI`:

```cpp
q_per_particle = sign(qom) * (Ninj / FourPI / npcel) * VOL;
```

### 3.3 GPU Moment Accumulation

In `momentKernelStayed()` (momentKernel.cu, lines 111–126), the particle weight
includes `invVOL` **twice**:

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

### 3.4 GPU→CPU Transfer

In `c_Solver::copyMomentsD2H()` (iPIC3Dlib.cu, line 677):

Raw `cudaMemcpyAsync` copies GPU moments directly into `rhons[is]`,
`Jxs[is]`, `Jys[is]`, `Jzs[is]`, `pXXsn`, `pXYsn`, etc. — **no scaling
is applied**.

### 3.5 Species Summation

`sumOverSpecies()` and `sumOverSpeciesJ()` (EMfields3D.cpp, lines 2033–2057)
are **plain sums** over species indices. No $4\pi$ factor is applied.

---

## 4. Field Solver: Gaussian 4π Factors

The implicit field solver uses the physical CGS moments with explicit $4\pi$
factors from Gaussian Maxwell's equations:

| Physics                | Location                  | Code formula                                      |
|------------------------|---------------------------|--------------------------------------------------|
| Ampère source          | `calculateE()`, L644      | `-FourPI/c * Jxh`                                |
| Poisson source         | `calculateE()`, L660      | `-θ²Δt² × FourPI × ∇ρ̂`                          |
| Susceptibility (MUdot) | `MUdot()`, L975           | `FourPI/2 × θ × dt/c × (q/m) × ρ_s / (1+Ω²)`  |
| Divergence cleaning    | `calculateE()`, L544      | `div(E) - FourPI × ρ_c`                          |
| Hat functions           | `calculateHatFunctions()` | `ρ̂ = ρ_c - dt·θ·div(Ĵ)` — no FourPI (consistent)|
| Rotation (PIdot)       | `PIdot()`, L928           | `1/(1+Ω²)` — no FourPI (pure rotation)           |

The hat function `ρ̂ = ρ_c - dt·θ·div(Ĵ)` has **no** explicit $4\pi$ because
both $\rho_c$ and $\mathbf{J}$ are in the same physical units; the continuity
equation $\partial\rho/\partial t + \nabla\cdot\mathbf{J} = 0$ has no $4\pi$.

---

## 5. Input File Parameters

Key parameters and their relationship to physical units:

| Input parameter  | Physical meaning                           | Internal value            |
|------------------|--------------------------------------------|---------------------------|
| `rhoINIT`        | $4\pi \times \rho_{\text{phys}}$           | `rhons = rhoINIT / 4π`   |
| `rhoINJECT`      | $4\pi \times \rho_{\text{inject,phys}}$    | `q ∝ rhoINJECT / 4π`    |
| `B0x, B0y, B0z`  | Physical $\mathbf{B}_0$ (= Alfvén speed)   | Used directly             |
| `uth, vth, wth`  | Thermal speeds $v_{th} = \sqrt{kT/m}$      | Used directly             |
| `u0, v0, w0`     | Drift velocities (per species)             | Used directly             |
| `qom`            | Charge-to-mass ratio $q/m$ (sign matters)  | Used directly             |
| `dt`             | Time step in $\omega_{pi}^{-1}$            | Used directly             |
| `Lx, Ly, Lz`    | Box size in $d_i$                          | Used directly             |
| `c`              | Speed of light (should be 1.0)             | Used directly             |

---

## 6. Output Backends

iPIC3D-GPU supports multiple output backends, selected by `WriteMethod` in the
input file:

| WriteMethod  | Format                    | Field output   | Moments output | Total ρ  |
|--------------|---------------------------|----------------|----------------|----------|
| `pvtk`       | Parallel VTK (MPI-IO)     | ✓             | ✓             | ✓       |
| `nbcvtk`     | Non-blocking VTK (MPI-IO) | ✓             | ✓             | ✗       |
| `phdf5`      | Parallel HDF5             | ✓             | (in Fields)   | ✗       |
| `shdf5`      | Serial HDF5 (per-process) | ✓             | ✓             | ✗       |
| `H5hut`      | H5hut format              | ✓             | ✗             | ✗       |
| `adios2`     | ADIOS2 format             | ✓             | ✗             | ✗       |

Output is controlled by two tag strings in the input file:
- **`FieldOutputTag`**: Controls field/current/density output (e.g. `"B+E+Je+Ji+rho"`)
- **`MomentsOutputTag`**: Controls moments output (e.g. `"rho+PXX+PXY+PXZ+PYY+PYZ+PZZ"`)

Tags are separated by `+` and matched via substring search.

---

## 7. Output Data Convention

The output rule is simple: fields, currents, pressure tensor components, and
particle quantities are written exactly as stored internally. Charge density is
the only systematic exception: it is multiplied by `4π` on output to match the
input-file convention.

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

**ρ is written multiplied by $4\pi$ (= `rhoINIT`-scale).**

| Density | Internal value | On-disk value | Scale |
|---------|---------------|---------------|-------|
| $\rho_s$ (per-species) | $\rho_{\text{phys}}$ | $4\pi \times \rho_{\text{phys}}$ | × 4π |
| $\rho_{\text{total}}$ | $\sum_s \rho_{s,\text{phys}}$ | $4\pi \times \sum_s \rho_{s,\text{phys}}$ | × 4π |

This means the on-disk density matches the `rhoINIT` input convention.

### 7.3 Current Density

| Current | Internal value | On-disk value | Scale |
|---------|---------------|---------------|-------|
| $J_{x,s}, J_{y,s}, J_{z,s}$ | $J_{\text{phys}}$ | $J_{\text{phys}}$ | none |
| $J_x, J_y, J_z$ (total) | $J_{\text{phys}}$ | $J_{\text{phys}}$ | none |

### 7.4 Pressure Tensor

| Tensor component | Internal value | On-disk value | Scale |
|-----------------|---------------|---------------|-------|
| $P_{xx}, P_{xy}, \ldots, P_{zz}$ | $P_{\text{phys}}$ | $P_{\text{phys}}$ | none |

See [Section 3.3](#33-gpu-moment-accumulation) for the accumulation formula
and [Section 11](#11-derived-quantities-and-usage-notes) for pressure calculations
from on-disk values.

### 7.5 Particles

Particle output (when enabled via `ParticlesOutputCycle > 0`) writes:

| Quantity | Tag | Description | Units |
|----------|-----|-------------|-------|
| Position | `position` | $x, y, z$ | $d_i$ |
| Velocity | `velocity` | $u, v, w$ | $c$ |
| Charge | `q` | Particle charge weight | physical (carries $1/(4\pi)$ factor) |
| ID | `ID` | Particle tracking ID | integer |

### 7.6 Energy Diagnostics

Available in SHDF5 backend:

| Quantity | Tag | Formula | Notes |
|----------|-----|---------|-------|
| Kinetic energy | `k_energy` | $\frac{1}{2} \sum_p \|q_p\| \mathbf{v}_p^2 / \|q/m\|$ | Per species |
| Magnetic energy | `B_energy` | $\sum (B_x^2 + B_y^2 + B_z^2) \cdot \text{VOL}$ | Global |

---

## 8. PVTK Output Files

### 8.1 Field Output (FieldOutputTag)

Available tags and output files:

| Tag | Filename pattern | Format | Data | Scaling |
|-----|-----------------|--------|------|---------|
| `B` | `{SimName}_B_{cycle}.vtk` | 3-component vector | $B_{x,y,z}^{\text{tot}}$ | none |
| `E` | `{SimName}_E_{cycle}.vtk` | 3-component vector | $E_{x,y,z}$ | none |
| `Je` | `{SimName}_Je_{cycle}.vtk` | 3-component vector | $J_{x,y,z}$ species 0 | none |
| `Ji` | `{SimName}_Ji_{cycle}.vtk` | 3-component vector | $J_{x,y,z}$ species 1 | none |
| `Je2` | `{SimName}_Je2_{cycle}.vtk` | 3-component vector | $J_{x,y,z}$ species 2 | none |
| `Ji3` | `{SimName}_Ji3_{cycle}.vtk` | 3-component vector | $J_{x,y,z}$ species 3 | none |
| `rho` | multiple files per cycle | scalar | writes `rhoe`, `rhoi`, and `rho_total` | × 4π |

Example input: `FieldOutputTag = B+E+Je+Ji+Je2+Ji3+rho`

VTK files are binary STRUCTURED_POINTS format with grid dimensions matching
`(nxc+1) × (nyc+1) × (nzc+1)` (node-centered, excluding ghost cells).

### 8.2 Moments Output (MomentsOutputTag)

Written for **all species**.

| Tag | Filename pattern | Scaling |
|-----|-----------------|---------|
| `rho` | `{SimName}_rho{e\|i}{species}_{cycle}.vtk` | × 4π |
| `PXX` | `{SimName}_PXX{e\|i}{species}_{cycle}.vtk` | none |
| `PXY` | `{SimName}_PXY{e\|i}{species}_{cycle}.vtk` | none |
| `PXZ` | `{SimName}_PXZ{e\|i}{species}_{cycle}.vtk` | none |
| `PYY` | `{SimName}_PYY{e\|i}{species}_{cycle}.vtk` | none |
| `PYZ` | `{SimName}_PYZ{e\|i}{species}_{cycle}.vtk` | none |
| `PZZ` | `{SimName}_PZZ{e\|i}{species}_{cycle}.vtk` | none |

Species naming: even-indexed species get `e` (electrons), odd-indexed get `i` (ions).
Examples for 4 species: `rhoe0`, `rhoi1`, `rhoe2`, `rhoi3`, `PXXe0`, `PXXi1`, etc.

Example input: `MomentsOutputTag = rho+PXX+PXY+PXZ+PYY+PYZ+PZZ`

---

## 9. PHDF5 Output Files

Parallel HDF5 writes all fields and moments into a single file per cycle.
All datasets are under the `"Fields"` group:

| Dataset | Data source | Grid | Scaling |
|---------|-------------|------|---------|
| `Ex`, `Ey`, `Ez` | Electric field | cell-centered | none |
| `Bx`, `By`, `Bz` | Magnetic field | cell-centered | none |
| `Rho_{is}` | Charge density, species `is` | cell-centered | × 4π |
| `Jx_{is}`, `Jy_{is}`, `Jz_{is}` | Current density, species `is` | cell-centered | none |

No pressure tensor components are written in the PHDF5 path.

---

## 10. SHDF5 Output Files

Serial HDF5 (one file per MPI process). Tags are parsed from both
`FieldOutputTag` and `MomentsOutputTag`. Available quantities:

**Fields** (from FieldOutputTag):

| Tag | HDF5 path | Scaling |
|-----|-----------|---------|
| `Ball` or `Bx/By/Bz` | `/fields/B{x,y,z}/cycle_{n}` | none |
| `Eall` or `Ex/Ey/Ez` | `/fields/E{x,y,z}/cycle_{n}` | none |
| `Jall` or `Jx/Jy/Jz` | `/moments/J{x,y,z}/cycle_{n}` | none |
| `Jsall` or `Jxs/Jys/Jzs` | `/moments/species_{s}/J{x,y,z}/cycle_{n}` | none |
| `rhos` | `/moments/species_{s}/rho/cycle_{n}` | × 4π |

**Moments** (from MomentsOutputTag):

| Tag | HDF5 path | Scaling |
|-----|-----------|---------|
| `pressure` | `/moments/species_{s}/p{XX,XY,...,ZZ}/cycle_{n}` | none |
| `k_energy` | `/energy/kinetic/species_{s}/cycle_{n}` | scalar |
| `B_energy` | `/energy/magnetic/cycle_{n}` | scalar |

---

## 11. Derived Quantities and Usage Notes

When post-processing output, convert density back to its internal physical form
with $\rho_{\text{phys}} = \rho_{\text{disk}}/(4\pi)$ whenever it appears in a
formula. J, E, B, and P can be used directly.

### 11.1 Correct Formulas Using On-Disk Values

| Derived quantity | Formula using disk values | Notes |
|-----------------|--------------------------|-------|
| **Bulk velocity** | $\mathbf{V}_s = \mathbf{J}_s / (\rho_{s,\text{disk}} / 4\pi)$ | Uses species density |
| **Hall E-field** | $\mathbf{J} \times \mathbf{B} / (\rho_{e,\text{disk}} / 4\pi)$ | Uses electron density |
| **E·J dissipation** | $\mathbf{E} \cdot \mathbf{J}$ | Both physical, use directly |
| **Alfvén speed** | $v_A = B_0 / \sqrt{\rho_{\text{disk}} / \|q/m\|}$ | $4\pi$ in $v_A = B/\sqrt{4\pi\rho_m}$ cancels with $\rho_m = \rho_{\text{disk}}/(4\pi\cdot\|q/m\|)$. For ions ($\|q/m\|=1$, `rhoINIT=1`): $v_A = B_0$ |
| **Plasma beta** | $\beta = 2 v_{th}^2 \rho_{\text{disk}} / B^2$ | $4\pi$ cancels between $p_{\text{gas}} = v_{th}^2 \rho_{\text{disk}}/(4\pi)$ and $p_B = B^2/(8\pi)$ |
| **Gas pressure** | $p = v_{th}^2 \times \rho_{\text{disk}} / (4\pi)$ | Mass density $\rho_m = \rho_{\text{disk}}/(4\pi)$ for ions ($m_i=1$) |
| **Magnetic pressure** | $p_B = B^2 / (8\pi)$ | Standard Gaussian CGS |
| **Gauss's law check** | $\nabla \cdot \mathbf{E} = \rho_{\text{disk}}$ | $\rho_{\text{disk}}$ already is $4\pi\rho_{\text{phys}}$ |

### 11.2 Quick Reference for Conversions

```
ρ_physical = ρ_disk / (4π)
J_physical = J_disk           (already physical)
E_physical = E_disk           (already physical)
B_physical = B_disk           (already physical)
P_physical = P_disk           (already physical)
```

---

## 12. Summary Table

| Quantity | Internal storage | On-disk value | Input convention |
|----------|-----------------|---------------|------------------|
| $\rho_s$ | $\rho_{\text{phys}} = \rho_{\text{init}} / 4\pi$ | $\rho_{\text{init}}$ (× 4π) | `rhoINIT` = $4\pi\rho_{\text{phys}}$ |
| $\mathbf{J}_s$ | $J_{\text{phys}}$ | $J_{\text{phys}}$ | — |
| $\mathbf{E}$ | Physical | Physical | — |
| $\mathbf{B}$ | Physical (incl. external) | Physical | `B0x, B0y, B0z` = physical |
| $P_{ij,s}$ | Physical | Physical | — |
| Particle $q$ | Physical | Physical | — |

Only ρ is written in the input-style `4πρ` scaling; J, E, B, P, and particle
weights are written in the internal physical normalized units.
