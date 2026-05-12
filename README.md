# iPIC3D-GPU

> iPIC3D with GPU acceleration, supporting multi-node multi-GPU.
```                                                                       
          ,-.----.                           .--,-``-.                   
          \    /  \      ,---,   ,----..    /   /     '.       ,---,     
  ,--,    |   :    \  ,`--.' |  /   /   \  / ../        ;    .'  .' `\   
,--.'|    |   |  .\ : |   :  : |   :     : \ ``\  .`-    ' ,---.'     \  
|  |,     .   :  |: | :   |  ' .   |  ;. /  \___\/   \   : |   |  .`\  | 
`--'_     |   |   \ : |   :  | .   ; /--`        \   :   | :   : |  '  | 
,' ,'|    |   : .   / '   '  ; ;   | ;           /  /   /  |   ' '  ;  : 
'  | |    ;   | |`-'  |   |  | |   : |           \  \   \  '   | ;  .  | 
|  | :    |   | ;     '   :  ; .   | '___    ___ /   :   | |   | :  |  ' 
'  : |__  :   ' |     |   |  ' '   ; : .'|  /   /\   /   : '   : | /  ;  
|  | '.'| :   : :     '   :  | '   | '/  : / ,,/  ',-    . |   | '` ,/   
;  :    ; |   | :     ;   |.'  |   :    /  \ ''\        ;  ;   :  .'     
|  ,   /  `---'.|     '---'     \   \ .'    \   \     .'   |   ,.'       
 ---`-'     `---`                `---`       `--`-,,-'     '---'         
                                                                         
```

## Citation
 - Markidis, S., Hu, A., Peng, I., Pennati, L., Lumsden, I., Yokelson, D., Brink, S., Pearce, O., Scogland, T.R., de Supinski, B.R. and Delzanno, G.L., 2025. Exascale Implicit Kinetic Plasma Simulations on El~ Capitan for Solving the Micro-Macro Coupling in Magnetospheric Physics. arXiv preprint arXiv:2507.20719.
 - Markidis, Stefano, and Giovanni Lapenta. "Multi-scale simulations of plasma with iPIC3D." Mathematics and Computers in Simulation 80.7 (2010): 1509-1519.


## Usage

### Requirement
To install and run iPIC3D-GPU, you need: 
- CUDA/HIP compatible hardware, CUDA capabiliy 7.5 or higher 
- cmake, MPI(MPICH and OpenMPI are tested) and HDF5 (optional), C/C++ compiler supporting C++ 17 standard

**To meet the requirements of compatability between CUDA and compiler, it's recommended to use a relatively new compiler version e.g. GCC 12**

If you are on a super-computer or cluster, it's highly possible that you can use tools like `module` to change the compiler, MPI or libraries used.

### Get the code

Git clone this repository or download the zip file of default branch. For example:

``` shell
git clone https://github.com/iPIC3D/iPIC3D-GPU.git iPIC3D-GPU
cd ./iPIC3D-GPU
```
Now you are in the project folder

### Build

- Create a build directory
``` shell
mkdir build && cd build
```
- Use `CMake` to generate the make files
``` shell
# use .. here because CMakeList.txt would be under project root 
cmake .. # using CUDA by default
cmake -DHIP_ON=ON .. # use HIP
```

#### CMake options

| Option | Default | Description |
|--------|---------|-------------|
| `CUDA_ARCH` | `75` | CUDA compute capability (minimum `75` for double-precision `atomicAdd`). Example: `-DCUDA_ARCH=80` for A100 |
| `HIP_ON` | `OFF` | Set `ON` to compile with HIP instead of CUDA |
| `HIP_ARCH` | `gfx90a` | HIP GPU architecture (e.g. `gfx90a` for MI250X, `gfx940` for MI300). Example: `-DHIP_ARCH=gfx940` |
| `USE_HDF5` | `ON` | Compile HDF5-based I/O backends (serial HDF5, parallel HDF5, H5hut) |
| `USE_PHDF5` | `ON` | Enable parallel HDF5 field output (requires `USE_HDF5=ON` and MPI-enabled HDF5). Auto-disabled when `USE_HDF5=OFF` |
| `USE_ADIOS2` | `ON` | Compile ADIOS2 backend for particles and restarts |
| `USE_CATALYST` | `OFF` | Enable ParaView Catalyst in-situ visualization (requires ParaView ≥ 5.7) |
| `USE_BATSRUS` | `OFF` | Enable BATS-R-US MHD coupling (adds `BATSRUS` compile definition) |
| `USE_OPENMP` | `ON` | Enable OpenMP in the CPU solver. **Delete CMake cache when changing this.** |
| `BENCH_MARK` | `OFF` | Print per-task timing (`LOG_TASKS_TOTAL_TIME`) |
| `BUILD_SHARED_LIBS` | `ON` | Build shared libraries (`OFF` for static) |
| `SITE` | `default` | Select a predefined site configuration from `cmake/sites/` |

Example with explicit architecture and selected backends:
```shell
cmake -DCUDA_ARCH=80 -DUSE_ADIOS2=ON -DUSE_HDF5=ON -DUSE_PHDF5=ON ..
# or for HIP:
cmake -DHIP_ON=ON -DHIP_ARCH=gfx90a -DUSE_ADIOS2=ON ..
```


- Compile with `make` if successful, you will find an executable named `iPIC3D` in build directory
``` shell
make # you can use this single-threaded compile command, but slow
make -j4 # build with 4 threads
make -j # build with max threads, fast, recommended
```

### Run

iPIC3D uses inputfiles to control the simulation, we pass this text file as the only command line argument:

``` shell
export OMP_NUM_THREADS=2
mpirun -np 8 ./iPIC3D ../share/inputfiles/magneticReconnection/testGEM3Dsmall.inp
```

With this command, you are using 8 MPI ranks, 2 OpenMP threads per rank.

**Important:** make sure `number of MPI process = XLEN x YLEN x ZLEN` as specified in the input file.

**Critical:** OpenMP is enabled by default, make sure the number of thread every process is reasonable. Refer to [OpenMP](#openmp) for more details.

If you are on a super-computer, especially a multi-node system, it's likely that you should use `srun` to launch the program. 

#### Multi-node and Multi-GPU

Assigning MPI processes to nodes and GPUs are vital in performance, for it decides the pipeline and subdomains in the program.

It's fine to use more than 1 MPI process per GPU, but the **number of MPI ranks per node must be evenly divisible by the number of GPUs on that node**. The following example uses 4 nodes, each equipped with 4 GPU:

``` shell
# 1 MPI process per GPU
srun --nodes=4 --ntasks=16 --ntasks-per-node=4 ./iPIC3D ../share/benchmark/GEM3Dsmall_4x2x2_100/testGEM3Dsmall.inp 

# 2 MPI processes per GPU
srun --nodes=4 --ntasks=32 --ntasks-per-node=8 ./iPIC3D ../share/benchmark/GEM3Dsmall_4x4x2_100/testGEM3Dsmall.inp  
```


### Result

This iPIC3D-GPU will create folder (usually named `data`) for the output results if it doesn't exist. On a **fresh start**, existing contents of the output folder are deleted. On a **restart**, the output folder is left intact and new data is appended.


## Build Options

### Debug

By default, the software is built with `Release` build type, which means highly optimized by the compiler. If you'd like to debug, use:

``` shell
cmake -DCMAKE_BUILD_TYPE=Debug ..
```
instead, and you'll have `iPIC3D_d`. If you'd like to just have an unoptimized (slow) version:
``` shell
cmake -DCMAKE_BUILD_TYPE=Default ..
```

### OpenMP

In this iPIC3D-GPU, the Solver stays on the CPU side, which means the number of MPI process will not only affect the GPU but also the Solver's performance. 

To speedup the CPU part, the OpenMP is enabled by default:
``` shell
cmake .. # default

cmake -DUSE_OPENMP=OFF .. # if you'd like to disable OpenMP

# set OpenMP threads for each MPI process
export OMP_NUM_THREADS=4
```
The solver on CPU will be benefited from OpenMP now, and this option is ON by default. It's important to control the number of threads per MPI process, make sure it's in a reasonable range.

## Restart

iPIC3D-GPU supports restarting a simulation from checkpoint files. Restart files store electromagnetic field arrays, moment arrays, and particle data. A restart checkpoint label is the loop cycle that will be executed first after restart.

### Launching a restart

Add the `restart` keyword to the command line alongside the input file (order does not matter):

```shell
mpirun -np 8 ./iPIC3D restart ../share/inputfiles/myInput.inp
# or equivalently
mpirun -np 8 ./iPIC3D ../share/inputfiles/myInput.inp restart
```

Without `restart`, the simulation always starts fresh from the initial conditions.

### Restart files

`RestartDirName` is the restart root directory. New checkpoints are written into two reusable slots under that root:

```text
RestartDirName/
  latest_restart.json
  restart_A/
  restart_B/
```

Each checkpoint rewrites the slot that is not currently marked latest. After every rank closes its rank file, the code enters an MPI barrier; then rank 0 writes `restart_A/manifest.json` or `restart_B/manifest.json` and atomically replaces `latest_restart.json`.

The rank-file format depends on the compile-time backend:

- **ADIOS2** (`USE_ADIOS2=ON`, default): `restart_A/restart_<rank>.bp` or `restart_B/restart_<rank>.bp`.
- **Serial HDF5** (`USE_ADIOS2=OFF`, `USE_HDF5=ON`): `restart_A/restart<rank>.hdf` or `restart_B/restart<rank>.hdf`.

On restart, the reader uses `latest_restart.json` to select the newest valid slot. If that slot is incomplete, it tries the previous slot recorded in the metadata. If no A/B metadata exists, it falls back to the legacy flat layout directly inside `RestartDirName`.

**Important:** you must restart with the **same number of MPI processes** as the original run, since each rank reads its own file.

### Input file parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `RestartDirName` | `data` | Restart root containing `latest_restart.json` and `restart_A` / `restart_B` |
| `RestartOutputCycle` | `5000` | Write a restart checkpoint every N cycles. Set to `0` to disable periodic checkpoints. |
| `CallFinalize` | `1` | If `1`, write a final restart checkpoint when the simulation ends. Requires `RestartOutputCycle > 0` — if `RestartOutputCycle` is `0`, no final restart is written even with `CallFinalize = 1`. |
| `ncycles` | — | Number of **new** cycles to run from the restart point (not an absolute cycle number). |

### How the cycle counter works

The cycle variable `i` in the main loop is a **global/absolute counter**. On restart, the checkpoint cycle label is used as the first loop cycle:

```
first_cycle = restart_cycle_label
LastCycle    = first_cycle + ncycles
loop:  i = first_cycle  ...  LastCycle - 1
```

The checkpoint label is the cycle to resume:

- A periodic checkpoint triggered at loop cycle `N` is written before cycle `N` has completed. It is labeled `N`, so a restart begins at `first_cycle = N` and executes the `N -> N+1` update.
- The final checkpoint written during `Finalize()` is written after the loop has completed. If the last executed loop cycle was `L`, it is labeled `L + 1`, so a restart begins at `first_cycle = L + 1`.

**Example:** with `RestartOutputCycle = 100`, the checkpoint triggered while loop cycle `100` is running is labeled `100`. A restart from it begins at cycle `100`. If that restarted run uses `ncycles = 500`, it executes cycles `100-599`.

For a fresh start (no `restart` keyword), `first_cycle = 0`.

### What a restart checkpoint contains

For a periodic checkpoint triggered at loop cycle `N`, the payload is the state needed to execute cycle `N` again:

- **Particles** are the host SoA mirror staged by `outputCopyAsync(N - 1)`, after the previous mover/exchange completed. They represent the particle state at the cycle-`N` boundary.
- **B** is saved before `CalculateB(N)`, so it represents `B^N`.
- **E** is saved after `CalculateField(N)`, so it represents `E^{N+1}`.
- **rho, J, and pressure moments** are the moment arrays available at the cycle-`N` boundary, before the cycle-`N` particle mover/exchange has completed.

For a final checkpoint, the payload is written after the last cycle has completed and after a fresh particle copy. If the last executed loop cycle is `L`, the checkpoint contains particles, B, E, and moments at the boundary for cycle `L + 1`, and it is labeled `L + 1`.

### What happens during restart initialisation

1. **Input file is read** — all simulation parameters (grid, species, BCs, etc.) are taken from the input file, same as a fresh start.
2. **Fields** — the case-specific field initialiser is called, but on restart most cases (GEM, Dipole, etc.) detect the restart flag internally and delegate to `init()`, which calls `read_field_restart()` to load B, E, and rho from the checkpoint. Some cases (e.g. Dipole) still set up auxiliary data (external dipolar field) before loading the checkpoint.
3. **Particles** — instead of generating particles from a distribution function, `restartLoad()` reads positions, velocities, charges, and IDs from the per-rank restart file.
4. **Output directory** — the output folder is **not** cleared on restart. Output files are opened in append mode.

### Requirements

- Restart requires either **ADIOS2** (`USE_ADIOS2=ON`) or **HDF5** (`USE_HDF5=ON`) at compile time. If neither is available, restart will produce a fatal error. The HDF5 restart path is less tested than ADIOS2.
- The MPI topology (`XLEN × YLEN × ZLEN`) must match between the original and restarted runs.

## Simulation Cases

The simulation case is selected via the `Case` parameter in the input file (e.g. `Case = GEM`). Each case configures specific initial conditions for the electromagnetic fields and particle distributions.

### Supported cases

| `Case` | Description | Example input |
|--------|-------------|---------------|
| `GEM` | **GEM magnetic reconnection challenge.** Single Harris current sheet with $B_x = B_{0x} \tanh\!\bigl((y - L_y/2)/\delta\bigr)$ and a localized Gaussian flux perturbation. Density has a $1/\cosh^2$ profile plus a uniform background. Ions carry the initial drift current. | `share/inputfiles/magneticReconnection/testGEM3D*.inp` |
| `GEMnoPert` | Same Harris equilibrium as GEM but **without** the magnetic perturbation. Useful for stability studies or when perturbations are applied externally. | — |
| `GEMDoubleHarris` | **Double Harris sheet.** Two oppositely-directed current sheets centred at $L_y/4$ and $3L_y/4$, each with its own drift population. Creates two reconnection sites in a periodic domain. | — |
| `ForceFree` | **Force-free current sheet.** Harris $B_x$ profile with $B_z = B_0/\cosh\!\bigl((y - L_y/2)/\delta\bigr)$ so that $\mathbf{J}\times\mathbf{B}=0$. **Note:** field initialisation works, but particle initialisation (`force_free()`) is currently a stub that aborts. Fresh starts will fail; usable only via restart from pre-existing checkpoint data. | `share/inputfiles/magneticReconnection/testForceFree*.inp` |
| `Dipole` | **3-D planetary magnetosphere.** Magnetic dipole field ($B \propto 1/r^3$) centred in the domain with solar-wind inflow electric field. A spherical planet region is voided of particles. `ConstantChargePlanet` is enforced every cycle. | `share/inputfiles/magnetosphere/testMagnetosphere3Dsmall.inp` |
| `Dipole2D` | **2-D dipole** (in the $xz$-plane). Same physics as `Dipole` but configured for a 2-D simulation. `ConstantChargePlanet2DPlaneXZ` is enforced every cycle. | — |
| `NullPoints` | **Magnetic null-point topology.** Periodic field with components like $B_x \propto -\sin x\,\cos y\,\cos z$ creating a network of null points. Electron current is initialised from $\nabla\times\mathbf{B}$; ions are at rest. | — |
| `TaylorGreen` | **Taylor-Green vortex.** 3-D periodic velocity/magnetic field. Particle initialisation reuses `maxwellianNullPoints()` (drift from $\nabla\times\mathbf{B}$). Used for testing decaying MHD turbulence and energy transfer. | `share/inputfiles/turbulence/testTurbulence3D.inp` |
| `RandomCase` | **Random multi-mode magnetic field.** Divergence-free superposition of 49 Fourier modes (7×7, fixed pseudo-random phases) with a uniform $B_{0z}$ guide field. Uniform density; particles initialised with generic `maxwellian()`. No Harris sheet. | — |
| `GEMHarris` | **Generalised Harris sheet.** Combines a Harris $B_x$ profile with optional GEM perturbation (`pertGEM`) and/or hump perturbation (`pertHump`). Supports Ampere-consistent current initialisation (`currentFromAmpere = 1`) and spatially varying thermal velocity (`spatiallyVaryingThermal = 1`). When `currentFromAmpere = 1`, drift velocities are derived from $\nabla\times\mathbf{B}$; the per-species `u0`, `v0`, `w0` are used component-wise as charge-sign-weighted species fractions controlling current partition. | `share/inputfiles/magneticReconnection/inpLe2DGEMHarris.inp` |
| `HumpPert` | **Magnetic hump perturbation.** Uniform density with a localised $\operatorname{sech}^2$ magnetic-pressure hump centred at the domain midpoint superimposed on $\mathbf{B}_0$. Hump width is hard-coded as multiples of `delta` ($8\delta$ in x, $4\delta$ in y). The input parameters `pertHump`/`deltaxHump`/`deltayHump` are **not** used by this case (they apply to `GEMHarris`). | — |
| `BATSRUS` | **Coupling with BATS-R-US MHD code.** Reads fluid fields from an external MHD solution. Requires `USE_BATSRUS=ON` at compile time. **Note:** field initialisation works with the compile flag, but GPU particle initialisation aborts (`ParticleSoAHost` not supported). Without the flag, the case falls through to default initialisation. | — |

Any unrecognised `Case` string falls through to a **default** initialisation: uniform density, constant background $\mathbf{B}$, zero electric field. A warning is printed to standard output.

### Runtime boundary conditions

Some cases apply additional boundary-condition fixes during the time loop:

| `Case` | Runtime BC |
|--------|------------|
| `GEM`, `GEMnoPert`, `GEMDoubleHarris` | `fixBnGEM` / `fixBcGEM` — enforce Harris-sheet–consistent normal-B and density at $y$-boundaries |
| `ForceFree` | `fixBforcefree` — enforce force-free B profile at $y$-boundaries |
| `Dipole` | `ConstantChargePlanet` — maintain fixed charge inside the planet sphere (every cycle) |
| `Dipole2D` | `ConstantChargePlanet2DPlaneXZ` — 2-D variant of the above (every cycle) |
| `GEMHarris` | None |
| `HumpPert` | None |

### Setting the case

In the input file, add or modify:

```
Case = GEM
```

The `Case` string is **case-sensitive** and must match one of the names in the table above exactly.

### Case-specific input parameters

Some cases use additional input-file parameters beyond the common ones:

| Parameter | Default | Used by | Description |
|-----------|---------|---------|-------------|
| `delta` | `0.5` | GEM, GEMnoPert, GEMDoubleHarris, ForceFree, GEMHarris, HumpPert | Current sheet half-thickness |
| `pertGEM` | `0.0` | GEM, GEMHarris | GEM flux perturbation amplitude |
| `pertHump` | `0.0` | GEMHarris | Hump perturbation amplitude |
| `deltaxHump` | `8.0` | GEMHarris | Hump width in x (multiplied by `delta`) |
| `deltayHump` | `4.0` | GEMHarris | Hump width in y (multiplied by `delta`) |
| `kxHump` | `-1.0` | GEMHarris | Hump wave number x (if ≥ 0) |
| `kyHump` | `-1.0` | GEMHarris | Hump wave number y (if ≥ 0) |
| `currentFromAmpere` | `0` | GEMHarris | `1`: derive drift velocity from $\nabla\times\mathbf{B}$ instead of using `u0`/`v0`/`w0` directly |
| `spatiallyVaryingThermal` | `0` | GEMHarris | `1`: use spatially varying thermal velocity (requires `currentFromAmpere = 1`) |

## Output Control

Output is controlled through **cycle parameters** (how often to write) and **tag strings** (what to write).

### Output cycles

| Parameter | Default | Description |
|-----------|---------|-------------|
| `FieldOutputCycle` | `100` | Write field/moments data every N cycles. `0` disables field output. |
| `ParticlesOutputCycle` | `0` | Write particle data every N cycles. `0` disables particle output. |
| `RestartOutputCycle` | `5000` | Write restart checkpoint every N cycles. `0` disables periodic checkpoints. |
| `DiagnosticsOutputCycle` | `FieldOutputCycle` | Write scalar diagnostics (energy, etc.) every N cycles. Defaults to `FieldOutputCycle` if not set. |
| `TestPartOutputCycle` | `0` | Write test-particle trajectory data every N cycles. |
| `SortingCycle` | `0` | Re-sort particles by cell every N cycles for cache efficiency. `0` disables. |

### Field output tags (`FieldOutputTag`)

A `+`-separated string selecting which grid-level fields to write. Example: `FieldOutputTag = B+E+rho`

| Token | Data written |
|-------|-------------|
| `B` | Magnetic field ($B_x$, $B_y$, $B_z$) |
| `E` | Electric field ($E_x$, $E_y$, $E_z$) |
| `Je` | Current density for species 0 ($J_{x,0}$, $J_{y,0}$, $J_{z,0}$) |
| `Ji` | Current density for species 1 |
| `Je2` | Current density for species 2 |
| `Ji3` | Current density for species 3 |
| `rho` | Total charge density |

### Moments output tags (`MomentsOutputTag`)

A `+`-separated string selecting which per-species or total moments to write. Example: `MomentsOutputTag = rho+J+PXX+PYY+PZZ`

**All-species** (writes one file per species):

| Token | Data |
|-------|------|
| `rho` | Charge density |
| `J` | Current density |
| `P` | Full pressure tensor (all 6 components) |
| `PXX`, `PXY`, `PXZ`, `PYY`, `PYZ`, `PZZ` | Individual pressure tensor components |
| `Q` | Full heat-flux tensor (all 10 symmetric components) |
| `Qxxx`, `Qxxy`, `Qxxz`, `Qxyy`, `Qxyz`, `Qxzz`, `Qyyy`, `Qyyz`, `Qyzz`, `Qzzz` | Individual heat-flux tensor components |

**Species-indexed** (single species `s`):

| Pattern | Example | Data |
|---------|---------|------|
| `rho<s>` | `rho0` | Density of species 0 |
| `J<s>` | `J1` | Current of species 1 |
| `P<s>` | `P0` | Full pressure tensor of species 0 |
| `PXX<s>` ... `PZZ<s>` | `PXX2` | Single component for species 2 |
| `Q<s>` | `Q0` | Full heat-flux tensor of species 0 |
| `Qxxx<s>` ... `Qzzz<s>` | `Qxyz2` | Single heat-flux component for species 2 |

**Summed totals** (sum over all species):

| Token | Data |
|-------|------|
| `rho_tot` | Total charge density |
| `J_tot` | Total current density |
| `P_tot` | Total pressure tensor |
| `PXX_tot` ... `PZZ_tot` | Individual total pressure components |
| `Q_tot` | Total heat-flux tensor |
| `Qxxx_tot` ... `Qzzz_tot` | Individual total heat-flux components |

### Particle output tags (`ParticlesOutputTag`)

A `+`-separated string selecting which particle data to write. Example: `ParticlesOutputTag = position+velocity+q+ID`

| Token | Data |
|-------|------|
| `position` | Particle positions ($x$, $y$, $z$) |
| `velocity` | Particle velocities ($u$, $v$, $w$) |
| `q` | Particle charge |
| `ID` | Particle ID |

Additional ADIOS2-only tokens used for restart data: `proc_topology`, `E`, `B`, `Js`, `rhos`, `pressure`.

### Other output parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `WriteMethod` | — | Field output backend (see [I/O Backends](#io-backends) below) |
| `SaveDirName` | `data` | Output directory. **Warning:** existing contents are deleted on fresh start. |
| `RestartDirName` | `data` | Restart root containing `latest_restart.json` and `restart_A` / `restart_B` |
| `CallFinalize` | `1` | Write a final restart checkpoint when the simulation ends (requires `RestartOutputCycle > 0`) |
| `ParaviewScriptPath` | `""` | Path to ParaView Catalyst Python script (requires `USE_CATALYST`) |

## I/O Backends

iPIC3D-GPU writes three categories of data: **fields** (E, B, J, rho, moments), **particles** (position, velocity, charge, ID), and **restart checkpoints** (fields + particles). Each category can use a different I/O backend, selected through a combination of CMake options and the `WriteMethod` parameter in the input file.

### CMake options

| Option | Default | Effect |
|--------|---------|--------|
| `USE_HDF5` | `ON` | Compile HDF5-based backends (serial HDF5, parallel HDF5, H5hut) |
| `USE_PHDF5` | `ON` | Enable the parallel HDF5 field backend (requires `USE_HDF5=ON` and an MPI-enabled HDF5 library) |
| `USE_ADIOS2` | `ON` | Compile the ADIOS2 backend for particles and restarts |

Example:
```shell
cmake -DUSE_HDF5=ON -DUSE_PHDF5=ON -DUSE_ADIOS2=ON ..
```

### Input file: `WriteMethod`

The `WriteMethod` parameter in the input file controls the **field output** backend:

| `WriteMethod` | Field backend | Description |
|---------------|---------------|-------------|
| `shdf5` | Serial HDF5 | One HDF5 file per MPI rank (file-per-process) |
| `phdf5` | Parallel HDF5 | All ranks write collectively into a single HDF5 file (requires `USE_PHDF5=ON`) |
| `pvtk` | Blocking VTK | Collective MPI-IO into VTK files |
| `nbcvtk` | Non-blocking VTK | Non-blocking collective MPI-IO into VTK files; writes overlap with computation of the next cycle |
| `H5hut` | H5hut | Collective I/O through the H5hut library |
| `adios2` | ADIOS2 | **Not implemented yet** — will throw a runtime error |

### Particle and restart backend selection

Particle and restart backends are **not** controlled by `WriteMethod`. They are selected at **compile time** based on the CMake flags:

| | `USE_ADIOS2=ON` | `USE_ADIOS2=OFF`, `WriteMethod=H5hut` | `USE_ADIOS2=OFF` (other) |
|---|---|---|---|
| **Particles** | ADIOS2 (file-per-process, BP5) | H5hut (collective) | Serial HDF5 (file-per-process) |
| **Restarts** | ADIOS2 (file-per-process, BP5) | Serial HDF5 (file-per-process) | Serial HDF5 (file-per-process) |

When `USE_ADIOS2=ON`, ADIOS2 always takes priority for particles and restarts, regardless of `WriteMethod`.

### Available combinations summary

With the default CMake settings (`USE_HDF5=ON`, `USE_PHDF5=ON`, `USE_ADIOS2=ON`):

| `WriteMethod` | Fields | Particles | Restarts |
|---------------|--------|-----------|----------|
| `pvtk` | Blocking VTK (collective) | ADIOS2 (file-per-process) | ADIOS2 (file-per-process) |
| `nbcvtk` | Non-blocking VTK (collective, overlapped) | ADIOS2 (file-per-process) | ADIOS2 (file-per-process) |
| `shdf5` | Serial HDF5 (file-per-process) | ADIOS2 (file-per-process) | ADIOS2 (file-per-process) |
| `phdf5` | Parallel HDF5 (collective, single file) | ADIOS2 (file-per-process) | ADIOS2 (file-per-process) |
| `H5hut` | H5hut (collective) | ADIOS2 (file-per-process) | ADIOS2 (file-per-process) |

With `USE_ADIOS2=OFF` and `USE_HDF5=ON`:

| `WriteMethod` | Fields | Particles | Restarts |
|---------------|--------|-----------|----------|
| `pvtk` | Blocking VTK (collective) | Serial HDF5 (file-per-process) | Serial HDF5 (file-per-process) |
| `nbcvtk` | Non-blocking VTK (collective, overlapped) | Serial HDF5 (file-per-process) | Serial HDF5 (file-per-process) |
| `shdf5` | Serial HDF5 (file-per-process) | Serial HDF5 (file-per-process) | Serial HDF5 (file-per-process) |
| `phdf5` | Parallel HDF5 (collective, single file) | Serial HDF5 (file-per-process) | Serial HDF5 (file-per-process) |
| `H5hut` | H5hut (collective) | H5hut (collective) | Serial HDF5 (file-per-process) |

### Notes

- **`nbcvtk`** is the only field backend that overlaps I/O with computation. It starts a non-blocking MPI write and completes it at the next output cycle.
- **Parallel HDF5 (`phdf5`)** writes fields only — parallel particle output is not implemented for this backend.
- **ADIOS2** uses the BP5 engine with `MPI_COMM_SELF`, producing one `.bp` directory per MPI rank. ADIOS2 field output is declared in the code but not yet implemented.
- All backends perform **synchronous blocking** writes from the main thread, except `nbcvtk` for fields.

## Tool

### Benchmark
In [benchmark](./share/benchmark/) folder, we prepared some scripts for profiling, please read the [benchmark/readme](./share/benchmark/readme.md) for more infomation.

There's a performance baseline file for your reference, 2 threads per process is used:

![GH200](./Documentation/image/GH200_release_baseline.png)

<!-- ![dual-A100](./Documentation/image/dual_A100_release_baseline.png) -->

You can find the corresponding data at [./share/benchmark/GH200_release_baseline.csv](./share/benchmark/GH200_release_baseline.csv).
 <!-- and [./benchmark/Dual-A100_release_baseline.csv](./benchmark/Dual-A100_release_baseline.csv).  -->

<!-- Please note that the `Particle` and `Moments` parts are not exactly the time consumption of these two parts, as the kernels are interwaved in this version. The sum of the two parts are precise, though. -->


## Contact

Feel free to contact Professor Stefano Markidis at KTH for using iPIC3D. 
