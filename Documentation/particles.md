# Particles

[Documentation index](../README.md#documentation)

Species keys are in [Input file](input-file.md#species), and face behaviour in [Boundary conditions](boundary-conditions.md).

## Loading

On a fresh start, each species is loaded on a regular sub-grid of `npcelx`, `npcely`, `npcelz` particles per cell, with Maxwellian velocities about the density and drift its [case](cases.md) sets.

## Mover

| Key | Default | Description |
|---|---|---|
| `NiterMover` | `3` | Most iterations of the implicit predictor-corrector push, at least 1. The push stops earlier when the mean velocity changes by less than 10⁻⁶ of itself |

- The standard mover is the non-relativistic implicit push.
- `Dipole` and `Dipole2D` use a relativistic push, subcycled by |B| at the particle's start so that each subcycle gyrates by at most π/4 there, with at most 1024 subcycles.
- Both movers cap |v| at 0.9999 c. A particle whose velocity or position is not finite after the push is deleted, and so is one that starts a relativistic push at |v| ≥ c.
- Reflection on `bcPface* = 1` faces is part of the push, see [Boundary conditions](boundary-conditions.md#particles-on-a-reflecting-face).

## Sorting

| Key | Default | Description |
|---|---|---|
| `SortingCycle` | `0` | Sort the particles by cell every N cycles. `0` disables it |

On a sorting cycle the moments of the sorted particles use a cell-aware deposit kernel, and particles appended after the sort use the unsorted one. Other cycles use the unsorted kernel. Merging, the macrocell analyses, capped particle output and checkpoints sort the species they need on their own cycles.

## MPI exchange

After the mover, departing particles are collected into buffers and exchanged with the neighbouring ranks until none remain in flight. Particles still in flight after the first two rounds and 2(`XLEN` + `YLEN` + `ZLEN`) more stop the run. The device arrays grow as described in [Architecture](architecture.md#memory).

## Particle IDs

With `TrackParticleID`, each particle of the species has a persistent ID, kept through exchange, sorting and restarts. Injected particles, open-outflow copies, exosphere particles and the second particle of a split get new IDs. A merged particle keeps the ID of one of the pair.

## Particle count control (splitting and merging)

Flows move particles between ranks, and open boundaries and the planet remove them, so a rank's count drifts from its loaded number. Two options, both off by default, keep it in a band, after Lapenta and Brackbill, J. Comput. Phys. 115, 213 (1994).

| Key | Default | Description |
|---|---|---|
| `ParticleSplitting` | `0` | Split particles of a species on a rank holding fewer than `ParticleSplittingThreshold` times its reference count |
| `ParticleMerging` | `0` | Merge particles of a species on a rank holding more than `ParticleMergingThreshold` times its reference count |
| `ParticleSplittingThreshold` | `0.95` | Lower edge of the band |
| `ParticleMergingThreshold` | `1.05` | Upper edge of the band |

The **reference count** of a species on a rank is the number the input loads there and the planet leaves: `npcelx * npcely * npcelz` times the rank's cells, in `Dipole` and `Dipole2D` only those whose centre is outside the planet. It does not depend on the state of the run, so a restart continues with the same reference. A rank that loads another number prints a warning at a fresh start. The [planetary species](#exosphere-ionisation) are not controlled. With only one option on, the run warns that the count is bounded on one side only.

The control acts on the count of a rank, not of a cell: it does not refill depleted cells of a rank whose total is inside the band.

**Splitting.**

- Before the mover, a rank below the band splits as many particles as it misses, each once per cycle at most, so a rank below half its reference at most doubles per cycle.
- A rank that has lost all particles of a species is not refilled.
- Parents are spread evenly over the rank's particles and change from cycle to cycle.
- A parent becomes two particles of half its charge and the same velocity, displaced by ±δ along one grid axis (never an axis with a single cell), inside the parent's cell. δ is half the loading interval of that axis, less for a parent near a cell face.
- Along one axis the node weight is linear, so the pair deposits exactly the parent's charge, current and pressure on every node: the fields do not see the split.
- A species is not split in a cycle in which it is merged. The run warns when `ParticleSplittingThreshold` is not below `ParticleMergingThreshold`: the two would undo each other.

**Merging.**

- After the moments, the species above the band that was merged longest ago is chosen, one per cycle and rank. Before the next mover it is sorted by cell.
- In every cell holding more than `npcelx * npcely * npcelz` particles, each particle looks for the nearest later particle in velocity. The two merge if |v₁ − v₂|² is at most 0.9 % of the cell's velocity variance, taken about its mean velocity so the test is frame-independent.
- The merged particle has the total charge, the centre of charge and the mean velocity of the pair. A particle merges at most once, and a cell stops at `npcelx * npcely * npcelz` particles.
- A merge keeps the charge, the momentum and the centre of charge exactly. It loses the kinetic energy of the relative motion, m₁m₂|v₁ − v₂|² / (2(m₁ + m₂)): at most 0.23 % of the mean thermal energy of a pair of equal particles.
- It keeps the ten node moments only for a pair along a grid axis with equal velocities. Otherwise it moves charge, current and pressure between the eight nodes of the cell, keeping their total charge.
- The search compares all pairs of a cell, so its cost grows with the square of the particles per cell.

## Exosphere ionisation

With `AddExosphereInjection = 1`, each cycle injects photoionised particles into the planetary species, those from index `ns_solar_wind` on.

| Key | Default | Description |
|---|---|---|
| `AddExosphereInjection` | `0` | `1` enables the source |
| `ns_solar_wind` | `ns` | Number of solar-wind species |
| `ns_planetary` | `0` | Number of planetary species; with the source on, `ns - ns_solar_wind` |
| `RmaxExosphereInjection` | `3.0` | Outer radius of the source, from the planet centre. It must exceed `Planet_radius` (default `5.0`), else nothing is injected |
| `MaxExosphereParticlesPerSpecies` | `0` | Particles injected per species, rank and cycle at most; `0`: no limit |
| `NeutralSurfaceDensity` | required if `ns_planetary > 0` | N, per planetary species |
| `ExosphericScaleHeight` | required if `ns_planetary > 0` | H, per planetary species |
| `PhotoionizationFrequency` | required if `ns_planetary > 0` | ν, per planetary species |
| `MacroParticleWeightRatio` | required if `ns_planetary > 0` | w, per planetary species |

In every active cell whose centre lies between `Planet_radius` (R) and `RmaxExosphereInjection`:

- The neutral density follows the Chamberlain profile n(r) = N exp(−(r − R)/H).
- The expected number of new particles is `npcel` · w · Δt · ν · n(r), with `npcel` the particles per cell of species 0. Its integer part is always injected, and the remainder is the probability of one more.
- Each has charge sign(q/m) · V_cell / (`npcel` · 4π · w), a uniform position in the cell and a Maxwellian velocity with the species' thermal and drift speeds.

Quasi-neutrality comes from the input. List the planetary species in pairs (the first and second, the third and fourth, ...) and give a pair identical neutral parameters: a pair shares its random numbers, so it injects the same number of particles at the same positions. When less than 30 % of the GPU memory is free, each planetary species is further limited to its share of 80 % of the free memory.

## Test particles

`nsTestPart` species are loaded at startup after the `ns` species, with the kinetic energy `energy` in units of m c² (γ = 1 + `energy`), the pitch angle `pitch_angle` (radians) relative to z (B is assumed along z), a random gyrophase, and the charge following the density of species 0. `TestPartOutputCycle` writes them, see [Output](output.md#test-particles).

> **Limitation.** The time loop does not advance test particles: every output holds their initial state. Serial-HDF5 checkpoints write them, but a restart does not read them back: it loads them again from the input.
