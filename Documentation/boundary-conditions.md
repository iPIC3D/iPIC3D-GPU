# Boundary conditions

[Documentation index](../README.md#documentation)

## Boundary codes

Every face has a field code (`bcEMface*`, required, also on periodic faces, where it is unused) and a particle code (`bcPface*`, default `1`).

| `bcEMface*` | Field boundary |
|---|---|
| `0` | Perfect electric conductor at rest, a *conductor* in these pages |
| `1` | First-order perfect magnetic conductor: ghost E_n = B_t = 0, ghost E_t and B_n copy the face node (cell B_n: the first cell), no boundary rows |
| `2` | Open face (inflow, outflow or lateral) |

| `bcPface*` | Particle boundary |
|---|---|
| `0` | Exit: the particle is removed |
| `1` | [Specular reflection](#particles-on-a-reflecting-face) |
| `2` | [Open inflow](#open-inflow-faces): deleted near the face; species with `rhoINJECT > 0` are refilled with a Maxwellian |
| `3` | [Open outflow](#open-outflow-faces) |

Any other field or particle code stops the run at startup.

An **open inflow face** is a face with `bcPface* = 2`. If it is also open for the fields (`bcEMface* = 2`) and is not X-right, it is a **field inflow face**: its fields are tied to E0 = −(u0×B0)/c, with `u0` the drift of species 0, the field in which the inflowing plasma drifts. The field rules below that name E0 apply to field inflow faces only.

## Boundary keys

| Key | Default | Description |
|---|---|---|
| `yes_sal` | `1` | [Absorbing layer](#absorbing-layer-on-open-faces) on open faces |
| `n_layers_sal` | `3` | Layers of the absorbing layer, or, with `yes_sal = 0`, of the fixed and copied layers. At least 1 with the layer on; a count that does not fit in the ranks on an open face (about their cells along the normal) stops the run |
| `sal_tangential_B` | `1` | `1`: blend only the tangential B. `0`: all three components |
| `ApplyInflowBcsEImage` | `1` | Rows of the field inflow faces in the Maxwell image, see [below](#boundary-rows-of-the-implicit-solve) |
| `OutflowLayerFields` | `solved` | Fields on the [X-right outflow layer](#open-outflow-faces): `solved` or `copy` |
| `ForceWallB` | `0` | [GEM and ForceFree wall forcing](cases.md#wall-forcing) |

## Boundary rows of the implicit solve

Each boundary condition is split into a linear part in the image row and a constant part in the right-hand-side row, so that the image M·v the Krylov solvers evaluate stays linear.

| Face | Image row | Right-hand-side row |
|---|---|---|
| Field inflow, with `ApplyInflowBcsEImage = 1` | v | E0 |
| Conductor, tangential components | v_t | 0 |
| Conductor, normal component | The interior row, read through the conductor image | The interior right-hand side, read through the image |
| Every other face | The operator's | The physical right-hand side |

Where a field inflow face meets a conductor, the inflow rows win.

## Fields on a conductor

The tangential E on a conductor at rest is zero. The identity rows impose E_t = 0 on the face nodes; where conductor faces meet, every component is tangential to one of them, so all three are 0. Since GMRES meets them only to `GMREStol`, the solved E(n+θ) and the initial E are set to E_t = 0 there. The wall E is not E0: a wall at rest does not carry the drift field of the plasma.

Beyond the wall the fields are their image in the conductor, which keeps E_t = 0 and reverses the charges and tangential currents:

- E_n and B_t are even about the wall: ghost node = node 2, the first inside.
- E_t and B_n are odd about their wall values (0 for E_t, the frozen W for B_n): ghost node = 2·face − node 2.

The mover reads these ghost nodes for a push ending beyond the wall.

**Cell B.** Since E_t is fixed on the wall, Faraday's law freezes the normal B there:

- Tangential B: ghost = first cell.
- Normal B: its change is odd, so the wall value stays at its initial W: ghost = 2W − first cell.

W is the mean of the node B_n over the 4 face nodes of each cell face, taken from the initial field or the checkpoint when the solver starts. Each face node then carries the mean of the W of its 4 cell faces for the whole run, and Faraday's law conserves divC2N(B) on them. A restart takes W from the checkpoint's node B, which averages W once more along the wall; this is exact for a wall field that is uniform or linear along the wall.

**Normal E on the wall.** The face node's normal row is the [interior row](field-solver.md#scheme) read through the conductor image, in which the image charge is opposite:

- *Ghost cells:* the gradients of E have the image of E, with ∂_n E_n and ∂_t E_t odd. The charge-like ∇·(μE) and ρ̂ are odd, so the face row's Gauss term ∂_n(·) is twice the wall-cell value over Δ.
- *Cell charge:* the solve's ρ_c reads the face-node charge as 0 (real plus image). The output densities keep the folded charge.
- *Wall-cell divergences of Ĵ and μE:* they read the face's tangential components as 0 and its normal component as stored. The face Ĵ_n is the image current, the change of the induced wall charge.
- *Ĵ on the face:* its pressure term −(Δt/2)∇·Π has the ghosts of a current's image, so its tangential part interpolates to 0 on the face node. Ĵ_t is set to 0 after it is formed, and the Ĵ smoothing keeps it 0 through its odd ghosts, which makes it the adjoint of the E smoothing.

With these, the Gauss term of the interior rows sums by parts up to the wall, as energy consistency requires. A face row without it, or with a copy ghost of the charge, leaves a wall term that drives a growing wall mode at θ = ½. The vacuum standing wave between the walls conserves energy at θ = ½ to round-off (unit test `maxwell_conductor_modes`).

## Particles on a reflecting face

With `bcPface* = 1`, a particle is reflected specularly on the face, inside its push:

- A push that would end beyond the face is shortened until it ends on it, to 10⁻⁶ of the smallest cell size, in at most 5 refinements of its duration.
- The normal velocity is reversed there; on an edge or corner, for every reflecting face reached.
- The rest of the step is pushed from the face, and further bounces in the same step are treated alike.

Every part of the step is the mover's own implicit push, with the fields at its own mid position, so no force acts across the face. This holds for the standard push and for each subcycle of the relativistic one, on any face and in any decomposition.

A step takes at most 10 pushes. After them, the rest of the step is one free push, and a particle left beyond the face is mirrored back. Only reflecting faces closer than one step's flight, or a gyration the step does not resolve (ΩΔt well above 1), reach that limit.

## Open inflow faces

With `bcPface* = 2` (any face, X-right included, whatever its `bcEMface*`), every particle of every species that ends a push in the last 3 cells before the face is deleted, so planetary species leave the domain there. For every species with `rhoINJECT > 0`, the GPU then refills those cells with a drifting Maxwellian (`rhoINJECT`, the species' drift and thermal speeds), every cycle, with the same particles per cell as the initial loading. A drawn velocity above c is drawn again.

Each injected particle has its own Philox stream, set by the cycle, species, global cell and sub-cell, so ranks inject different plasma, the injection does not depend on `XLEN`, `YLEN` or `ZLEN`, and a restart repeats it.

## Open outflow faces

**Fields (X-right).** With `bcEMfaceXright = 2`, `bcPfaceXright = 3` and `PERIODICX = 0`, the last L = max(3, `n_layers_sal`) cells before the X-right face form the outflow layer, beyond the reference plane x_s, L cells inside the face. The solve sees the layer as neutral, with the current and pressure of x_s. This is always on.

- `OutflowLayerFields` sets the fields on the layer after each solve:
  - `solved` (default): the solver's E and B are kept, so they satisfy Faraday's and Ampère's laws as inside, with the layer's neutral charge and copied current. Only the ghost node and ghost cell beyond the face are copied from the face, the zero-gradient continuation the solver assumes.
  - `copy`: E and B are copied from x_s, a flat continuation. This holds the field at the face at its x_s value, removing the outward push of a tangential field that weakens toward the face. Where slow plasma drains through the face along field lines tied elsewhere, the flux then piles up behind x_s.
- Before each solve, the species densities on the layer are set to the x_s values, rescaled to equal positive and negative charge, (ρ₊ + |ρ₋|)/2 each. Where a node of x_s holds charge of one sign only, the layer nodes in its line are emptied. J_s and the pressure tensors are copied from x_s.
- The predicted charge ρ̂ is zero on every cell beyond x_s, including the cell between x_s and the first layer node.
- Inside the solver, the face is closed with a zero-gradient ghost node.
- With `divBCorrection`, the cleaning corrects B on every cell, the layer included.

The written `rho`, `J` and pressure on the layer are these neutralised or copied values; the written E and B are the copied or solved fields.

**Particles (every face with `bcPface* = 3`).** A particle ending a push beyond the face is deleted. The plasma beyond the face is the last 3 cells shifted 3 cells outward, moving as its sources, so a particle that crosses the plane 3 cells inside the face inward during the push re-enters: a copy with its new velocity and charge is appended at its end position shifted 3 cells outward, and pushed from the next cycle.

Beyond an edge or corner between such faces the plasma is shifted across each of them, so a particle crossing these planes of two faces in one push gives three copies, and of three faces seven. A copy is kept if it lies inside the domain and outside the injection layers of the open inflow faces. A copy beyond a periodic boundary is wrapped by the particle exchange.

## Absorbing layer on open faces

With `yes_sal = 1` (the *SAL* of the run log), every cycle blends B toward B0 on the cell layers 0…`n_layers_sal` of every open face but X-right, after Faraday's law and before B is interpolated to the nodes; an open X-right face copies its B layers from the next layer inward instead (on an [outflow face](#open-outflow-faces), as `OutflowLayerFields` sets). Field inflow faces also blend E toward E0 on their node layers 0…`n_layers_sal`, after the solve. The weight s rises linearly from 0 on the ghost layer by 1/`n_layers_sal` per layer to 1 inside, so the face node of E and the first cell of B get 1/`n_layers_sal`.

- `sal_tangential_B = 1`: only B_t is blended; B_n follows Faraday's law. Blending B_n would create ∇·B of about (B_n − B0_n)/(`n_layers_sal` Δx) every cycle; the B_t blend creates only (1 − s) ∂_n B_n, which the [∇·B cleaning](field-solver.md#b-cleaning) removes.
- `sal_tangential_B = 0`: all three components are blended.

Without the layer (`yes_sal = 0`):

- Every field inflow face holds E0 on its node layers 0…`n_layers_sal` after the solve and B0 (all components) on its cell layers after Faraday's law.
- An open X-left face gets B0 whatever its `bcPface*`.
- On other open faces, E is left as solved (on the X-right outflow face, as `OutflowLayerFields` sets) and B on those layers is copied from the next cell layer inward.

The B treatment of an open face runs only on the ranks on that face with at least 9 cells along its normal, a gate meant for the flat direction of a 2D run. On a thinner rank of any other direction it is skipped, and the code and `scripts/ipic_preflight.py` warn: use fewer ranks along that direction.

## Planet surface (`Dipole`, `Dipole2D`)

| Key | Default | Description |
|---|---|---|
| `Planet_radius` | `5.0` | Planet radius R |
| `x_center_planet`, `y_center_planet`, `z_center_planet` | `5.0` | Planet centre; `Dipole2D` ignores `y_center_planet`, except in the [exosphere source](particles.md#exosphere-ionisation), whose distance is three-dimensional |
| `planetReflectionType` | `1` | `1`: diffuse reflection; `0`: specular. Any other value stops the run at startup |
| `PlanetDiagnosticsCycle` | `DiagnosticsOutputCycle` | Interval of `PlanetBC.txt`; `<= 0` disables it |

The dipole field has its own keys, see [Simulation cases](cases.md#case-specific-keys).

Particles that enter the planet are processed on the GPU every cycle, on each rank:

- **Ions** are all absorbed.
- **Electrons** of all species are ranked together by the relativistic kinetic energy of one physical electron and absorbed from the most energetic down, until the absorbed charge best matches the charge to neutralise: the ions absorbed in this cycle plus the imbalance carried from earlier cycles. The remainder, at most half an electron macroparticle unless too few electrons hit, carries to the next cycle. It is not in the checkpoints, so a restart starts it from zero.
- **Other electrons** are reflected about the outward normal at the point where the push left them (radial; in 2D, the in-plane normal of the cylinder):
  - An electron that ended at depth d is put on the same ray at R + d, but at least at 1.05 R (in 2D, y is kept). On the first cycle of a run, the electrons loaded inside the planet are processed, and those not absorbed are placed at up to 2R.
  - A specular reflection reverses only an inward normal velocity. A diffuse one keeps the speed and draws the direction by the Lambert cosine law. Either way the electron leaves with v·n ≥ 0.
  - It may land on another rank, which the particle exchange handles. One placed beyond a domain face is handled by the particle exchange too: removed on an exit face (`0`), mirrored on a reflecting face (`1`), mirrored with a new inward thermal velocity on an open inflow face (`2`), and removed with a warning on an open outflow face (`3`).

Every cycle, before the field solve, each species' density on the nodes inside the planet is reset to its initial value.

Every `PlanetDiagnosticsCycle` cycles, rank 0 appends the surface balance to `SaveDirName/PlanetBC.txt`: the cycles the row covers (`NCycles`); per species, the macroparticles and charge that hit and were absorbed; the carried charge, summed and its extremes; the mean energy separating absorbed from reflected electrons; and the rank-cycles with hits and with too few electrons to balance the ions. A row labelled N covers the hits since the previous row, up to cycle N−1. A last row at the end of the run, labelled with the resume cycle, covers the rest. A fresh run starts a new file; a restart appends.
