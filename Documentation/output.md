# Output

[Documentation index](../README.md#documentation)

Physical units and the scaling of written quantities are in [Units](units.md). Checkpoints are in [Restart](restart.md), and the velocity analyses in [Data analysis](data-analysis.md).

## Overview

| Output | Interval | Content | Backend |
|---|---|---|---|
| Fields and moments | `FieldOutputCycle` | `FieldOutputTag`, `MomentsOutputTag` | `FieldWriteMethod` (required) |
| Particles | `ParticlesOutputCycle` | `ParticlesOutputTag`, species of `ParticlesOutputSpecies` | `ParticlesWriteMethod` |
| Checkpoints | `RestartOutputCycle`, and at the end with `CallFinalize` | Fields and all particles (test particles only with `shdf5`) | `RestartWriteMethod` |
| Energy diagnostics | `DiagnosticsOutputCycle` | Energies, momentum, charge | Text |

These outputs go to `SaveDirName`, the checkpoints to `RestartDirName`, and the [data-analysis](data-analysis.md) records to `DataAnalysisOutputDir`.

## Output keys

Example for four species:

```text
SaveDirName    = data            # output folder
SimulationName = myrun           # prefix of VTK and parallel-HDF5 files

FieldWriteMethod = pvtk
FieldOutputCycle = 100
FieldOutputTag   = B+E
MomentsOutputTag = rho+J+P_tot

ParticlesOutputCycle   = 1000
ParticlesOutputTag     = position+velocity+q
ParticlesOutputSpecies = 0 1 0 1 # species 1 and 3 only
TrackParticleID        = 0 1 0 0 # IDs for species 1
ParticlesOutputTrackedFraction = -1 0.1 -1 -1
ParticlesOutputMaxPerCell      = 0 0 0 16
ParticlesWriteMethod   = adios2

DiagnosticsOutputCycle = 100
```

| Key | Default | Description |
|---|---|---|
| `SaveDirName` | `data` | [Output folder](running.md#output-folder) |
| `FieldOutputCycle` | `100` | Field and moment interval |
| `FieldOutputTag`, `MomentsOutputTag` | empty | [Tags](#field-and-moment-tags) |
| `ParticlesOutputCycle` | `0` | Particle interval |
| `ParticlesOutputTag` | empty | [Tags](#particle-tags) |
| `ParticlesOutputSpecies` | all `1` | Per-species mask, nonzero = written. With particle output on, at least one species |
| `ParticlesOutputTrackedFraction` | `-1` | [Fraction](#tracked-fraction) of a species with `TrackParticleID` that is written: 0 to 1, or `-1` for off |
| `ParticlesOutputMaxPerCell` | `0` | [Cap](#cap-per-cell) of written particles per cell and species; `0`: all |
| `ParticlesWriteMethod` | `adios2` if built, else `shdf5` | Particle backend |
| `AsyncOutput` | `1` | [Deferred writing](#deferred-writing) |
| `DiagnosticsOutputCycle` | `FieldOutputCycle` | [Energy diagnostics](#energy-diagnostics) interval |
| `TestPartOutputCycle` | `0` | [Test-particle](#test-particles) interval |
| `ParaviewScriptPath` | empty | ParaView Catalyst script, with `USE_CATALYST` |

**When.** An output of interval `N > 0` is written at the multiples of `N`, cycle 0 included; the field output also at the first cycle of a restart. `N <= 0` switches it off.

**What.** A tag joins tokens with `+`, and spaces around a token are ignored. An empty tag writes nothing. An unknown field or moment token, or one naming a species at or above `ns`, is ignored with a warning. An unknown particle token stops the run.

**Checked at startup.** The run stops, naming the key, for:

- a missing `FieldWriteMethod`, one other than `shdf5`, `phdf5`, `pvtk` or `nbcvtk`, or an HDF5 one the build lacks;
- a `ParticlesWriteMethod` or `RestartWriteMethod` other than `adios2` or `shdf5`;
- a selection key with neither one nor `ns` values, or a value outside the range given in the table above.

Two settings without effect only warn, for a written species: a fraction other than `-1` on a species without `TrackParticleID`, and a cap on a species whose fraction is on.

## Backends

| `FieldWriteMethod` | Format | Needs |
|---|---|---|
| `pvtk` | VTK, blocking collective MPI-IO | |
| `nbcvtk` | VTK, non-blocking collective MPI-IO, in flight until the next output cycle | |
| `shdf5` | Serial HDF5, one file per rank | `USE_HDF5=ON` |
| `phdf5` | Parallel HDF5, one file per cycle, fields only | `USE_PHDF5=ON` |

`adios2` is not implemented for fields. `FieldWriteMethod` is checked also with the field output off.

| `ParticlesWriteMethod`, `RestartWriteMethod` | Format | Needs |
|---|---|---|
| `adios2` | ADIOS2 BP5, serial, one `.bp` directory per rank | `USE_ADIOS2=ON` |
| `shdf5` | Serial HDF5, one file per rank | `USE_HDF5=ON` |

The three methods are independent. A particle or checkpoint backend the build lacks is replaced by the other, and rank 0 warns once if such data are to be written. A build with neither writes no particles or checkpoints, warns, and runs on.

## Field and moment tags

`FieldOutputTag`, for example `B+E+rho`:

| Token | Data |
|---|---|
| `B`, `E` | Magnetic (including any external field) and electric field |
| `Je`, `Ji`, `Je2`, `Ji3` | Current of species 0, 1, 2, 3 |
| `rho` | Total charge density |

`MomentsOutputTag`, for example `rho+J+PXX+PYY+PZZ`. A bare token writes every species, a token with an index `<s>` one species, and a `_tot` token the sum over species:

| Token | Variants | Data |
|---|---|---|
| `rho` | `rho<s>`, `rho_tot` | Charge density |
| `J` | `J<s>`, `J_tot` | Current density |
| `P` | `P<s>`, `P_tot` | All 6 pressure components |
| `PXX`, `PXY`, `PXZ`, `PYY`, `PYZ`, `PZZ` | `PXX<s>`, `PXX_tot`, ... | One pressure component |
| `Q` | `Q<s>`, `Q_tot` | All 10 heat-flux components |
| `Qxxx`, `Qxxy`, `Qxxz`, `Qxyy`, `Qxyz`, `Qxzz`, `Qyyy`, `Qyyz`, `Qyzz`, `Qzzz` | `Qxyz<s>`, `Qxxx_tot`, ... | One heat-flux component |

## Particle tags

`ParticlesOutputTag`, for example `position+velocity+q+ID`:

| Token | Data |
|---|---|
| `position` | $x, y, z$ |
| `velocity` | $u, v, w$ |
| `q` | Charge |
| `ID` | Particle ID, for species with `TrackParticleID` |

Only the arrays and species written are copied from the GPU, and the mover waits for that copy, so a smaller particle output takes less time.

## Selecting the particles written

A species uses at most one selection. With `TrackParticleID` and a fraction other than `-1`, it writes its fraction and the cap has no effect. Any other species follows `ParticlesOutputMaxPerCell`.

Both use one number per particle: `u`, a 64-bit hash of the ID divided by `2^64`, uniform on [0, 1) and uncorrelated with position and velocity. Each species has its own hash, so species are selected independently. Since the ID never changes, neither does `u`. A species without IDs hashes the position and first velocity component instead, so each output writes another random sample.

Positions, velocities and IDs are written unchanged. The files hold the same variables and datasets as without a selection, and `particleNum<rank>.csv` keeps counting every particle.

### Tracked fraction

With `ParticlesOutputTrackedFraction = f`, a species with `TrackParticleID` writes the particles with `u < f`. `0` writes none (its variables and datasets are written empty) and `1` writes all.

- **Which.** About `f n` particles, give or take `sqrt(f (1 - f) n)`, each with probability `f` whatever its position or velocity. The same particles are written at every output until they leave the domain, and across restarts. A new particle (injected or split) is written if its `u < f`. With `ID` in the tag, trajectories can be followed from file to file.
- **Charge.** Each written particle has its charge divided by `f`. Densities, currents and pressures computed from them are on average those of the simulation; a cell holding `m` written particles gives its density within about `1 / sqrt(m)`.
- **Cost.** No sort: particles are compacted on the GPU and written in their order there. GPU memory: two buffers of 12 bytes in all per written particle of the largest selected species, plus a quarter for growth.

### Cap per cell

With `ParticlesOutputMaxPerCell = K > 0`, a cell holding `n` particles of a species writes `min(n, K)`: all when `n <= K`, otherwise the `K` with the smallest `u`.

- **Which.** A uniform random sample of the cell: each particle is written with probability `K / n`, and every set of `K` is equally likely, whatever their order on the GPU. With IDs, the same particles are written from one output to the next while the cell's population does not change.
- **Charge.** Each written particle has its charge times `Q / Q_kept`, the charges of all and of the written particles of the cell (`n / K` for equal charges). The charge of each cell is exact: binned by cell, the density is that of the simulation; interpolated to nodes it is close but not exact, because the written particles sit elsewhere in the cell. Currents and pressures are the cell's charge times the charge-weighted means of the written particles.
- **Cost.** The species must be sorted by cell, so a particle output cycle sorts it once more, as a checkpoint does (one sort when both fall on the same cycle). A cell's particles are written together. GPU memory: the sort's buffers, 12 bytes per particle of capacity plus three integers per cell, which a run with checkpoints or `SortingCycle` holds already; otherwise the first capped output allocates them for the rest of the run.

## Files

Fields are written on the global node grid `(nxc+1) × (nyc+1) × (nzc+1)`, except `shdf5`, which holds each rank's local nodes.

| Output | Backend | Files |
|---|---|---|
| Fields, moments | `pvtk`, `nbcvtk` | `<SimulationName>_<quantity>_<cycle>.vtk`: binary `STRUCTURED_POINTS`, one per quantity and cycle |
| | `phdf5` | `<SimulationName>_<cycle:05d>.h5`, datasets in group `Fields` |
| | `shdf5` | `proc<rank>.hdf` with `settings.hdf`, datasets `/fields/...`, `/moments/...` |
| Particles | `adios2` | `particle_<rank>.bp`: one step per output cycle, variables `cycle`, `part<s>PositionX`, ..., `part<s>VelocityU`, ..., `part<s>charge`, `part<s>ID` |
| | `shdf5` | `proc<rank>.hdf`: `/particles/species_<s>/{x,y,z,u,v,w,q,ID}/cycle_<N>` |
| Test particles | serial HDF5 | `proc<rank>.hdf`: `/testparticles/species_<s>/{x,y,z,u,v,w,ID}/cycle_<N>`, with `s` counting on after the `ns` species |
| Checkpoints | `adios2`, `shdf5` | See [Restart](restart.md#checkpoint-slots) |
| Energy diagnostics | text | `ConservedQuantities.txt` |
| Planet surface balance | text | `PlanetBC.txt`, see [Planet surface](boundary-conditions.md#planet-surface-dipole-dipole2d) |
| Run record | text | `SimulationData.txt`, `particleNum<rank>.csv` (particles of each species, every cycle) |

A species outside `ParticlesOutputSpecies` has no particle variables or datasets, and `ID` exists only for species with `TrackParticleID`.

Names of the field datasets:

| Tag | VTK `<quantity>` | `phdf5` dataset | `shdf5` path (with `/cycle_<N>`) |
|---|---|---|---|
| `B` | `B` | `Bx`, `By`, `Bz` | `/fields/B{x,y,z}` |
| `E` | `E` | `Ex`, `Ey`, `Ez` | `/fields/E{x,y,z}` |
| `rho`, `rho<s>` (moments) | `rho<s>` | `rho_<s>` | `/moments/species_<s>/rho` |
| `rho` (fields), `rho_tot` | `rho_tot` | `rho_tot` | `/moments/rho` |
| `J`, `J<s>`, `Je`, ... | `J<s>` | `Jx_<s>`, `Jy_<s>`, `Jz_<s>` | `/moments/species_<s>/J{x,y,z}` |
| `J_tot` | `J_tot` | `Jx_tot`, ... | `/moments/J{x,y,z}` |
| `PXX`, `P`, ... | `PXX<s>`, ... | `pXX_<s>`, ... | `/moments/species_<s>/p{XX,XY,...,ZZ}` |
| `PXX_tot`, ... | `PXX_tot`, ... | `pXX_tot`, ... | `/moments/pXX_tot` |
| `Qxxx`, `Q`, ... | `Qxxx<s>`, ... | `Qxxx_<s>`, ... | `/moments/species_<s>/Qxxx` |
| `Qxxx_tot`, ... | `Qxxx_tot`, ... | `Qxxx_tot`, ... | `/moments/Qxxx_tot` |

## Restarts and existing files

A restart appends to the files of the run it continues. It runs again from its checkpoint cycle, so it writes the outputs of that cycle and later again.

- **VTK and parallel-HDF5 files**, one per cycle, are overwritten.
- **`particle_<rank>.bp`:** the repeated cycles are appended as further steps. The last step of a cycle is the one of the restart. This also holds on another decomposition, so a rank's file then holds steps of two subdomains.
- **`SimulationData.txt`** is rewritten with the restart's record; `ConservedQuantities.txt` and `particleNum<rank>.csv` are appended to, see [Energy diagnostics](#energy-diagnostics).
- **`proc<rank>.hdf`**, with `FieldWriteMethod = shdf5` or when the restart writes serial-HDF5 particles or test particles:
  - Before the first cycle, each rank removes its datasets `cycle_<N>` with `N` at or after the restart cycle, including those of outputs no longer written, so each file holds one history. To keep the earlier outputs, restart into another `SaveDirName`. HDF5 may keep the freed space (`h5repack` reclaims it).
  - On another decomposition or periodicity, the files of the earlier run, `settings.hdf` included, are kept as `*_before_cycle<N>.hdf` (`_2`, `_3`, ... if such a file exists), new files start and rank 0 warns.
  - A rank file the run would continue but cannot open or rewrite stops the run before the first cycle.

## Deferred writing

With `AsyncOutput = 1`, the particle outputs, checkpoints and `shdf5` field outputs are written by writer threads while the run goes on. With `0` they are written at their cycle on the main thread.

- A cycle waits only for the copy of its data to host memory, except that a checkpoint first waits for every pending write, and a `phdf5` field or test-particle output for the pending HDF5 writes. One writer thread for HDF5 and one for ADIOS2 then write, at low priority, in cycle order.
- The host buffers keep the data until written, so a later copy into them waits for the writer, and an output may take until the next copy of its data to reach the disk.
- The end of the run waits for the last writes and prints the time the main thread waited for the writers.
- A writer error stops the run: an ADIOS2 one at the next cycle, naming the output; a serial-HDF5 one at once.
- The `pvtk`, `nbcvtk` and `phdf5` outputs (collective MPI writes, which only the main thread may make), the energy diagnostics and the test particles are written at their cycle, on the host while the GPU mover of that cycle runs.

## Energy diagnostics

Every `DiagnosticsOutputCycle` cycles, the last MPI rank appends a row to `ConservedQuantities.txt`, with the columns `Cycle`, `Total_Energy`, `Momentum`, `Eenergy`, `Benergy`, `Kenergy`, `Total_Charge`, then `Ke[0…ns−1]`, `BulkE[0…ns−1]`, `Mom[0…ns−1]`, `Q[0…ns−1]`, and `Px`, `Py`, `Pz`. A restart appends, so the rows of the cycles it repeats appear twice; `particleNum<rank>.csv` is appended to the same way, with a header line from each restart. A file with other columns is renamed `ConservedQuantities.txt.before_cycle_<cycle>` and a new one starts.

- `Total_Energy` is `Eenergy + Benergy + Kenergy`. `Q[s]` is the sum of the particle charges of species s, and `Total_Charge` their sum.
- `Eenergy` is the node sum $\sum w\,\tfrac12\Delta V|\mathbf E|^2/4\pi$.
- `Benergy` is the cell sum $\sum\tfrac12\Delta V|\mathbf B_c+\mathbf B_{\text{ext}}|^2/4\pi$, with the cell B that Faraday's law advances and the node $\mathbf B_{\text{ext}}$ averaged over the 8 corners. E on nodes and B on cells are the pair the discrete energy balance holds for; the node B is an interpolation that filters the grid scale.
- At a restart `Benergy` drops, see [Restart](restart.md#checkpoint-content).
- Node sums count each global node once. The trapezoidal weight $w$ is ½ per non-periodic domain face the node lies on.
- `Ke[s]`, with $m=q/(q/m)$: $\sum(\gamma-1)mc^2$ in `Dipole` and `Dipole2D`, whose mover is relativistic, and $\sum\tfrac12mv^2$ otherwise. `Kenergy` is their sum. In the relativistic cases a particle with $|v|\ge c$, which only loaded data can hold and the mover deletes, is left out of `Ke[s]` and `Mom[s]`.
- `Mom[s]` is the magnitude of $\sum\gamma m\mathbf v$ ($\gamma=1$ in the non-relativistic cases). `Momentum` is the magnitude of the total, with components `Px`, `Py`, `Pz`.
- `BulkE[s]` is the node sum $\sum w\,\tfrac12\Delta V|\mathbf J_s|^2/(\rho_s\,q_s/m_s)$.

## Test particles

`TestPartOutputCycle` writes their positions, velocities and IDs with the serial-HDF5 writer into `proc<rank>.hdf`, whatever the write methods. A build without HDF5 warns and does not write them. See the [limitation](particles.md#test-particles).
