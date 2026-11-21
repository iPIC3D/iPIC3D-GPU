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
- CUDA/HIP compatible hardware, CUDA capability 7.5 or higher
- CMake ≥ 3.21; a CUDA-aware (with HIP, ROCm-aware) MPI library, since the field solver passes device buffers to MPI (MPICH and OpenMPI are tested); ADIOS2 and HDF5 for the default build (`-DUSE_ADIOS2=OFF` / `-DUSE_HDF5=OFF` drop them; CMake finds ADIOS2 through `ADIOS2_DIR` or `ADIOS2_ROOT`); and a C/C++ compiler supporting the C++17 standard

**To meet the requirements of compatibility between CUDA and compiler, it's recommended to use a relatively new compiler version e.g. GCC 12**

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
| `CUDA_ARCH` | `75` | CUDA compute capability (minimum `75`). Example: `-DCUDA_ARCH=80` for A100 |
| `HIP_ON` | `OFF` | Set `ON` to compile with HIP instead of CUDA |
| `HIP_ARCH` | `gfx90a` | HIP GPU architecture (e.g. `gfx90a` for MI250X, `gfx942` for MI300A/MI300X). Example: `-DHIP_ARCH=gfx942` |
| `USE_HDF5` | `ON` | Compile HDF5-based I/O backends (serial HDF5, parallel HDF5) |
| `USE_PHDF5` | `ON` | Enable parallel HDF5 field output (requires `USE_HDF5=ON` and MPI-enabled HDF5). Auto-disabled when `USE_HDF5=OFF` |
| `USE_ADIOS2` | `ON` | Compile ADIOS2 backend for particles and restarts |
| `USE_CATALYST` | `OFF` | Enable ParaView Catalyst in-situ visualization (requires ParaView ≥ 5.7) |
| `USE_BATSRUS` | `OFF` | Enable BATS-R-US MHD coupling (adds `BATSRUS` compile definition) |
| `USE_OPENMP` | `ON` | Enable OpenMP in the host-side loops (see [OpenMP](#openmp)). **Delete CMake cache when changing this.** |
| `GPU_CYCLE_DIAGNOSTICS` | `OFF` | Instrument one GPU timestep and the Maxwell right-hand side that consumes it |
| `GPU_DIAGNOSTIC_CYCLE` | `1` | Cycle instrumented by `GPU_CYCLE_DIAGNOSTICS` (`-1`: every cycle) |
| `BUILD_UTESTS` | `OFF` | Build the unit tests (run with `ctest` in the build directory) |
| `TIME_TASKS` | `OFF` | Print per-cycle and cumulative task timings (`LOG_TASKS_TOTAL_TIME`) |
| `BENCHMARK_MODE` | `0` | Compile-time benchmark level: `0` normal; `1` disables disk output and output-related device-to-host copies; `2` additionally disables data analysis, macrocell spectra, and heat-flux calculations |
| `BUILD_SHARED_LIBS` | `ON` | Build shared libraries (`OFF` for static) |
| `SITE` | `default` | Select a predefined site configuration from `cmake/sites/` |

`BENCHMARK_MODE` removes output-related device-to-host transfers. Transfers
required by the simulation itself, such as the MPI/planet particle exchange and
the field solver's scalar reductions, remain enabled. At level 1,
calculation-essential transfers inside an enabled analysis algorithm also remain;
heat-flux kernels likewise remain on their configured field-output cadence, but
their D2H, host ghost preparation, and writing are removed. Level 2 removes
those optional diagnostic calculations entirely.

Example with explicit architecture and selected backends:
```shell
cmake -DCUDA_ARCH=80 -DUSE_ADIOS2=ON -DUSE_HDF5=ON -DUSE_PHDF5=ON ..
# or for HIP:
cmake -DHIP_ON=ON -DHIP_ARCH=gfx90a -DUSE_ADIOS2=ON ..
# core-solver benchmark with task timing, no disk output or data analysis:
cmake -DBENCHMARK_MODE=2 -DTIME_TASKS=ON ..
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

It's fine to use more than 1 MPI process per GPU. When a node runs more MPI ranks than it has visible GPUs, the **number of MPI ranks per node must be divisible by the number of GPUs**, and consecutive ranks share a GPU. The following example uses 4 nodes, each equipped with 4 GPUs:

``` shell
# 1 MPI process per GPU
srun --nodes=4 --ntasks=16 --ntasks-per-node=4 ./iPIC3D ../share/benchmark/GEM3Dsmall_4x2x2_100/testGEM3Dsmall.inp 

# 2 MPI processes per GPU
srun --nodes=4 --ntasks=32 --ntasks-per-node=8 ./iPIC3D ../share/benchmark/GEM3Dsmall_4x4x2_100/testGEM3Dsmall.inp  
```


### Result

This iPIC3D-GPU will create folder (usually named `data`) for the output results if it doesn't exist. On a **fresh start**, existing contents of the output folder, and of `RestartDirName` if it differs, are deleted. On a **restart**, the output folder is left intact and new data is appended.


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

The field solver, the particle mover and the moments run on the GPU. OpenMP parallelises the host-side loops: particle initialisation and the other host particle operations, and the vector and grid operations of the field initialisation. It is enabled by default:
``` shell
cmake .. # default

cmake -DUSE_OPENMP=OFF .. # if you'd like to disable OpenMP

# set OpenMP threads for each MPI process
export OMP_NUM_THREADS=4
```
Keep the number of threads per MPI process in a reasonable range, so that ranks × threads does not oversubscribe the cores of a node.

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

By default a restart reads the latest checkpoint. To read a given slot (see [Restart files](#restart-files)), name it:

```shell
mpirun -np 8 ./iPIC3D ../share/inputfiles/myInput.inp restart=B   # or: restart B
mpirun -np 8 ./iPIC3D ../share/inputfiles/myInput.inp restart=A   # or: restart A
mpirun -np 8 ./iPIC3D ../share/inputfiles/myInput.inp restart=latest  # same as restart
```

If the requested slot is not usable, the run falls back to the other slot and says so at startup. An unknown slot name, a repeated `restart` or a second input file stops the run with a usage message.

### Restart files

`RestartDirName` is the restart root directory. New checkpoints are written into two reusable slots under that root:

```text
RestartDirName/
  latest_restart.json
  restart_A/
  restart_B/
```

Each checkpoint goes to the slot not holding the run's previous checkpoint: a fresh run starts with `restart_A`, and the first checkpoint after a restart goes to the slot that was not read. Writing a checkpoint takes four steps:

1. Rank 0 removes the slot's `manifest.json`, marking the slot incomplete.
2. All ranks pass an MPI barrier, then replace their rank files.
3. All ranks pass a second barrier after closing their rank files.
4. Rank 0 writes `restart_A/manifest.json` or `restart_B/manifest.json` and atomically replaces `latest_restart.json`.

A write cut off before step 4 therefore leaves the slot without a manifest. Each checkpoint gets a `generation` one above the highest in the restart root, so the newest checkpoint always has the highest generation.

The rank-file format depends on the compile-time backend:

- **ADIOS2** (`USE_ADIOS2=ON`, default): `restart_A/restart_<rank>.bp` or `restart_B/restart_<rank>.bp`.
- **Serial HDF5** (`USE_ADIOS2=OFF`, `USE_HDF5=ON`): `restart_A/restart<rank>.hdf` or `restart_B/restart<rank>.hdf`.

On restart, rank 0 selects the slot and broadcasts the choice, so every rank reads the same checkpoint:

- With `restart=A` or `restart=B`, it tries that slot, then the other one.
- With `restart` or `restart=latest`, it tries the slot that `latest_restart.json` names as latest, then the other one. Without a usable `latest_restart.json`, it tries the slot whose manifest has the higher generation first.

A slot is usable when all of these hold:

- its `manifest.json` exists and names that slot;
- the manifest was written by the restart backend of this build;
- the manifest has a valid cycle, rank count and mesh;
- a rank file exists for each of the `nranks` ranks the manifest records.

Startup prints the slot read and the slot requested. It also prints notes: every slot passed over, with the reason, and, when the other slot holds a newer checkpoint, that the run's first checkpoint will overwrite it. The slot read and these notes are written at the top of `SimulationData.txt` too. If no slot is usable, the run stops and lists why. The slot manifest records the cycle label, backend, source rank count, and mesh metadata used for compatibility checks and remapping.

Restarting from the older slot starts a new branch of the run: its first checkpoint replaces the newer slot. Copy that slot elsewhere first if you need to keep it.

### Restart remapping

Restart files store only active data: E, B and the species density, current and pressure on active nodes (`NXN - 2`, `NYN - 2`, `NZN - 2`), and particles sorted by active cell with per-species offset/count metadata. A restart reads back E, B and the species densities; ghost nodes are rebuilt after loading.

The restart reader can load a checkpoint onto a different MPI topology when all of these conditions hold:

- The global mesh cell counts (`nxc`, `nyc`, `nzc`) are unchanged.
- The destination topology refines the source topology by powers of two in each direction; coarsening and arbitrary factors are not supported.
- The checkpoint contains active-node fields and active-cell particle metadata, as every checkpoint written by this code does.

`RestartRemapPlan` maps each destination rank to overlapping source active boxes. Fields are copied into guarded arrays at `[1][1][1]`; particles are read as the sorted spans belonging to the destination rank. The data-analysis records are per rank: a remapped restart keeps the records files of the analyses that run and the rank mappings of the old decomposition as `*_before_cycle<N>.*` and starts them again; those of the analyses switched off stay as they are (see Data analysis).

### Particle restart loading

Particle reads are span-based and chunked. Cell offset/count metadata is converted into merged spans, the destination host SoA is resized once, then spans are streamed through temporary buffers of at most `1 << 20` particles per component.

The host restart SoA is sized to the restarted count, padded to `DVECWIDTH`. The device SoA is allocated with `INITIAL_CAPACITY_FACTOR` times the particle count and grows on demand.

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
- **E** represents `E^N`: `CalculateField(N)` solves for `E^{N+θ}` into separate arrays, and `E` is advanced to `E^{N+1}` only at the end of `CalculateB(N)`. Checkpoints written by builds up to `a115269` hold `E^{N+1}` instead.
- **rho, J, and pressure moments** are the moment arrays available at the cycle-`N` boundary, before the cycle-`N` particle mover/exchange has completed.

For a final checkpoint, the payload is written after the last cycle has completed and after a fresh particle copy. If the last executed loop cycle is `L`, the checkpoint contains particles, B, E, and moments at the boundary for cycle `L + 1`, and it is labeled `L + 1`.

### What happens during restart initialisation

1. **Input file is read** — all simulation parameters (grid, species, BCs, etc.) are taken from the input file, same as a fresh start.
2. **Fields** — the case initialiser runs (for `Dipole`/`Dipole2D` it sets the external dipole field; the other cases only read the checkpoint), then `read_field_restart()` loads B, E and the species densities, remapping active-node boxes when needed. For `Dipole`/`Dipole2D` the planet charge is reset (`ConstantChargePlanet`) and for `TaylorGreen` `ConstantChargeOpenBC` is applied; ghost nodes and center fields are rebuilt.
3. **Particles** — `restartLoad()` reads positions, velocities, charges and, for species with `TrackParticleID`, IDs from the selected slot. With remapping, active-cell metadata selects the source spans for this destination rank.
4. **Output directory** — the output folder is **not** cleared on restart. Output files are opened in append mode.

### Requirements

- Restart requires either **ADIOS2** (`USE_ADIOS2=ON`) or **HDF5** (`USE_HDF5=ON`) at compile time. If neither is available, restarting aborts and writing checkpoints is disabled with a warning. The HDF5 restart path is less tested than ADIOS2.
- Restart requires the same global mesh dimensions and compatible boundary/input parameters. The MPI topology may either match the checkpoint topology or be refined by powers of two in each direction, as described above.
- A restart may use a different build than the one that wrote the checkpoint; this gives a warning, not an error (see [Build and run record](#build-and-run-record)).

## Build and run record

Every run records the code that produced it, so it can be reproduced later. At configure time and before every build, CMake (`cmake/GenerateBuildInfo.cmake`) records:

- the **version**, `git describe` of the commit, with `-dirty` if the working tree had uncommitted changes (modified, staged or untracked files that git does not ignore; files inside the build directory do not count, and an in-source build counts only tracked files);
- the **build signature**, one line with the full commit hash, a hash of the uncommitted changes (if any), the build type, the CMake options that change results (`CUDA_ARCH`/`HIP_ARCH`, `BENCHMARK_MODE`, `USE_*`, ...) and the compiler versions.

A build from a tree with uncommitted changes prints a CMake warning, at configure time and at every build, and the run prints the same warning: the commit alone does not reproduce such a binary. Builds without git (e.g. from a tarball) record `unknown`.

The record is written:

| Where | What |
|-------|------|
| Run log (rank 0, at startup) | "Build and run" block: version, commit, working-tree state, build type, options, compilers, start time, host, MPI ranks and library, OpenMP threads, GPUs, input file, fresh run or restart |
| `SaveDirName/SimulationData.txt` | the same block, above the simulation parameters |
| `SaveDirName/<input file>` | a copy of the input file; a restart adds `<name>.restart<cycle><ext>` |
| `latest_restart.json`, `restart_A/B/manifest.json` | `ipic3d_version` and `build_signature` of the build that wrote the checkpoint. A restart with a different signature, or from a checkpoint without one, prints a warning and continues |
| VTK field and moment files | dataset field data `ipic3d_version` and `ipic3d_build` (ParaView: Information panel, Field Data) |
| HDF5 files | string attributes `ipic3d_version`, `ipic3d_build` of the root group (`h5dump -A`) |
| ADIOS2 particle and restart files | attributes `ipic3d_version`, `ipic3d_build` (`bpls -a`) |
| Data-analysis records (the `.json` next to each `.bin`, of the run that last opened the file) | the same two entries |

## Simulation Cases

The simulation case is selected via the `Case` parameter in the input file (e.g. `Case = GEM`). Each case configures specific initial conditions for the electromagnetic fields and particle distributions.

### Supported cases

| `Case` | Description | Example input |
|--------|-------------|---------------|
| `GEM` | **GEM magnetic reconnection challenge.** Single Harris current sheet with $B_x = B_{0x} \tanh\!\bigl((y - L_y/2)/\delta\bigr)$ and a localized Gaussian flux perturbation of fixed amplitude $0.4\,B_{0x}$. Species with nonzero `u0` or `w0` form the sheet (density $\propto 1/\cosh^2$) and carry its current with their drift; the other species are a uniform background. | `share/inputfiles/magneticReconnection/testGEM3D*.inp` |
| `GEMnoPert` | Same Harris equilibrium as GEM but **without** the magnetic perturbation. Useful for stability studies or when perturbations are applied externally. | — |
| `GEMDoubleHarris` | **Double Harris sheet.** Two oppositely-directed current sheets centred at $L_y/4$ and $3L_y/4$, both carried by the drifting species (`u0` or `w0` ≠ 0): density $\operatorname{sech}^2$ of both sheets, drift $(u_0, v_0, w_0)$ below $L_y/2$ and $(u_0, v_0, -w_0)$ above; the other species are a uniform background. The X-point perturbation is applied to the lower sheet. Creates two reconnection sites in a periodic domain. | `share/inputfiles/turbulence/DebyeScaleTurbulence.inp` |
| `ForceFree` | **Force-free current sheet.** Uniform density, Harris $B_x$ profile and $B_z = B_{0z}/\cosh\!\bigl((y - L_y/2)/\delta\bigr)$, force-free ($\mathbf{J}\times\mathbf{B}=0$) when $B_{0z} = B_{0x}$ and $B_{0y} = 0$, plus a GEM-type perturbation whose flux function has amplitude $B_{0x}/10$. The current $\mathbf{J} = (c/4\pi)\nabla\times\mathbf{B}$ is split among the species by `u0`/`v0`/`w0` (`GEMHarris` with `currentFromAmpere = 1` splits its Harris current the same way), and the particles drift with $J_s/(\mathrm{sign}(q_s)\,\rho_s)$ (`maxwellianFromSpeciesCurrents()`). | `share/inputfiles/magneticReconnection/testForceFree2D.inp` |
| `Dipole` | **3-D planetary magnetosphere.** Magnetic dipole (strength `B1z`, $B \propto 1/r^3$) at (`x_center_dipole`, `y_center_dipole`, `z_center_dipole`), held as an external field that is zero inside the planet (`Planet_radius`), with the solar-wind inflow electric field. Particles that enter the planet are absorbed or reflected (see [Planet surface](#planet-surface-dipole-dipole2d)). The charge inside the planet is held constant every cycle (`gpuConstantChargePlanet`). | `share/inputfiles/magnetosphere/testMagnetosphere3Dsmall.inp` |
| `Dipole2D` | **2-D dipole** (in the $xz$-plane, $B \propto 1/r^2$). Same physics as `Dipole` but configured for a 2-D simulation. The charge inside the planet circle is held constant every cycle (`gpuConstantChargePlanet`). | `share/inputfiles/magnetosphere/Magnetosphere2D.inp`, `testMagnetosphere2Dsmall.inp` |
| `NullPoints` | **Magnetic null-point topology.** Periodic field $B_x = -B_{0x}\,(L_x/L_y)\cos(2\pi x/L_x)\sin(2\pi y/L_y)$, $B_y = B_{0x}\cos(2\pi y/L_y)\bigl[\sin(2\pi x/L_x) - 2\sin(2\pi z/L_z)\bigr]$, $B_z = 2B_{0x}\,(L_z/L_y)\sin(2\pi y/L_y)\cos(2\pi z/L_z)$ (divergence-free for any box), creating a network of null points. Electron current is initialised from $\nabla\times\mathbf{B}$; ions are at rest. | `share/inputfiles/NullPoints.inp` |
| `TaylorGreen` | **Taylor-Green vortex.** 3-D periodic velocity/magnetic field. Every species flows with the Taylor-Green velocity $\mathbf{u} = u_0\,(\sin x\cos y\cos z,\ -(L_y/L_x)\cos x\sin y\cos z,\ 0)$ in the field $\mathbf{B} = B_{0x}\,(\cos x\sin y\sin z,\ \sin x\cos y\sin z,\ -(L_z/L_x + L_z/L_y)\sin x\sin y\cos z)$ (`u0` of species 0, coordinates scaled by $2\pi/L$; both divergence-free for any box), so the flow carries no net current: the species currents are $J_s = \mathrm{sign}(q_s)\,\rho_s\,\mathbf{u}$, and particle initialisation reuses `maxwellianFromSpeciesCurrents()` (drift $J_s/(\mathrm{sign}(q_s)\,\rho_s) = \mathbf{u}$). Used for testing decaying MHD turbulence and energy transfer. | `share/inputfiles/turbulence/TaylorGreen.inp` |
| `RandomCase` | **Random multi-mode magnetic field.** Divergence-free superposition of 49 Fourier modes (7×7, fixed pseudo-random phases) with a uniform $B_{0z}$ guide field. Uniform density; particles initialised with generic `maxwellian()`. No Harris sheet. | — |
| `GEMHarris` | **Generalised Harris sheet.** Combines a Harris $B_x$ profile with optional GEM perturbation (`pertGEM`) and/or hump perturbation (`pertHump`). Supports Ampere-consistent current initialisation (`currentFromAmpere = 1`) and spatially varying thermal velocity (`spatiallyVaryingThermal = 1`). When `currentFromAmpere = 1`, the drifts follow Ampère's law: the Harris current $J_z = -(c/4\pi)(B_{0x}/\delta)\operatorname{sech}^2$ is split by the charge-sign-weighted `w0` fractions (a uniform drift per species), and the current of the perturbations, $(c/4\pi)\nabla\times\delta\mathbf{B}$, is carried by the drifting electron species in proportion to their density. The Harris equilibrium needs drifts in the ratio $u_e/u_i = -T_e/T_i$; the run warns when the `w0` split differs. The GEM perturbation fills the box, so `pertGEM` needs `currentFromAmpere = 0`; a hump wider than about the sheet would need drifts above $c$ where the sheet density vanishes, and the run stops at startup. | `share/inputfiles/magneticReconnection/inpLe2DGEMHarris.inp` |
| `HumpPert` | **Magnetic hump perturbation.** Uniform density; each component of $\mathbf{B}_0$ is multiplied by $1 + \tfrac12\operatorname{sech}^2\operatorname{sech}^2$ of the two transverse coordinates, measured from the domain midpoint, with width `delta`. The input parameters `pertHump`/`deltaxHump`/`deltayHump` are **not** used by this case (they apply to `GEMHarris`). | — |
| `BATSRUS` | **Coupling with BATS-R-US MHD code.** Reads fluid fields from an external MHD solution. Requires `USE_BATSRUS=ON` at compile time. **Note:** field initialisation works with the compile flag, but GPU particle initialisation aborts (`ParticleSoAHost` not supported), and restarts are not supported. Without the flag, the case falls through to default initialisation. | — |

Any unrecognised `Case` string falls through to a **default** initialisation: uniform density, constant background $\mathbf{B}$, zero electric field. A warning is printed to standard output.

### Runtime boundary conditions

Some cases apply additional boundary-condition fixes during the time loop. The GEM and ForceFree wall fixes force B and apply only with `ForceWallB = 1`. By default (`ForceWallB = 0`) the y walls keep the perfect-conductor conditions alone (below), and B there evolves by Faraday's law. The forcing is an external source of energy and reconnected flux: +13 % to +19 % total energy and flux 2.64 against 2.21 B0 d_i by tΩci = 19 in a 2D GEM.

| `Case` | Runtime BC |
|--------|------------|
| `GEM`, `GEMnoPert`, `GEMDoubleHarris` | with `ForceWallB = 1`: GEM case wall of `FieldBoundary` (the cell-B ghost rules, `LayerSet::CaseCells` and `CaseNodes`) — on the three cell and node planes next to each non-periodic $y$ wall, $B_x = B_{0x}\tanh\!\bigl((y_g - L_y/2)/\delta\bigr)$ with $y_g$ the centre of the ghost cell, and $B_z = B_{0z}$; on the ghost plane also $B_y = B_{0y}$, except on a perfect-conductor wall, whose rule for the normal $B_y$ (below) comes after it |
| `ForceFree` | with `ForceWallB = 1`: ForceFree case wall of `FieldBoundary` (the cell-B ghost rules and `LayerSet::CaseCells`) — on the three cell planes next to each non-periodic $y$ wall, $B_z = B_{0z}/\cosh\!\bigl((y - L_y/2)/\delta\bigr)$ at each plane's centre; on the ghost plane also $B_x = B_{0x}\tanh\!\bigl((y - L_y/2)/\delta\bigr)$ and $B_y = B_{0y}$, except on a perfect-conductor wall, whose rule for the normal $B_y$ (below) comes after it |
| `Dipole` | `gpuConstantChargePlanet` — maintain fixed charge inside the planet sphere (every cycle) |
| `Dipole2D` | `gpuConstantChargePlanet` — the same, on the planet circle of the x-z plane (every cycle) |
| `GEMHarris` | None |
| `HumpPert` | None |

### Field solver parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `SolverType` | `GMRES` | Maxwell solver: `GMRES` (GMRES(20)) or `FGMRESBlockJacobi` (FGMRES(20) with the communication-free point-block Jacobi preconditioner) |
| `GMREStol` | `1e-3` | Relative residual tolerance of the Maxwell solve: it stops when ‖r‖ < `GMREStol` ‖r₀‖, r₀ the residual of the initial guess Eⁿ; a warm start with ‖r₀‖ ≤ `GMREStol` ‖b‖ is kept without iterating |
| `DivCleaningTol` | `1e-6` | Relative residual tolerance of the solves of the ∇·E and ∇·B corrections (see [the cleaning solves](#solves-of-the-e-and-b-corrections)). It must lie in (0, 1) when a cleaning is on |
| `BlockJacobiSweeps`, `BlockJacobiOmega` | `1`, `1.0` | Richardson sweeps and relaxation factor of the block-Jacobi preconditioner (`SolverType = FGMRESBlockJacobi`) |
| `PoissonCorrection`, `PoissonCorrectionCycle` | `no`, `10` | ∇·E correction (`yes` enables) and its interval in cycles, which must be at least 1 when the correction is on (the run stops at startup otherwise) |
| `divBCorrection`, `divBCorrectionCycle` | `no`, `10` | ∇·B cleaning (`yes` enables) and its interval in cycles, which must be at least 1 when the cleaning is on (the run stops at startup otherwise) |
| `PoissonChebMaxIter` | `20` | The Chebyshev preconditioner of the ∇·E and ∇·B solves has degree `PoissonChebMaxIter` + 1 |
| `PoissonChebEigRatio` | `300` | Floor of its lower eigenvalue bound, λ_max / `PoissonChebEigRatio` (see [the cleaning solves](#solves-of-the-e-and-b-corrections)) |
| `PoissonChebRescaleEigMin`, `PoissonChebRescaleEigMax` | `1.0`, `1.0` | Factors applied to its lower and upper eigenvalue bounds |
| `Smooth`, `SmoothNiter` | `1.0`, `6` | Smoothing coefficient (`1` disables) and number of passes, applied to E after each solve and to the hat moments. `Smooth` must lie in [0, 1]. Below (2d − 3)/(2d), d being the number of directions that are not a single periodic cell (`0.5` in 3D, `0.25` in 2D), the filter's response is negative at grid scale and those modes flip sign every pass: with an odd `SmoothNiter` they come out inverted, and the run prints a warning; use an even `SmoothNiter` there. On the physical faces the E smoothing keeps the values the solve imposes (E_t = 0 on conductors, E0 on open inflow faces) and continues the other components evenly about the face node |
| `HaloOverlap` | `0` | `1` overlaps the halo exchanges of the field solver (Maxwell image, the ∇·E and ∇·B images, smoothing, B and hat-moment interpolation; a ∇·B cleaning cycle runs the B update blocking) with the computation that does not read the exchanged values; same results as `0`. Meant for multi-rank runs: on a single rank there is no MPI to hide and the extra boundary-layer launches can make it slower. The Maxwell image overlaps on ranks with at least 3 cells along every axis, except an axis of a single cell that is periodic onto the rank itself (one rank along it, as y of an x-z run; then only its node stage overlaps); other ranks use the blocking exchange |
| `SolverSourceGrowthLimit` | `100` | Stops the run when the 2-norm ‖b‖ of the Maxwell right-hand side is not finite or exceeds this factor times its value in the previous cycle; `0` disables the growth test. Between cycles ‖b‖ normally changes by tens of percent, and by up to a few times in the first cycles and on ∇·E correction cycles; a corrupted moment or field makes it jump by orders of magnitude. ‖b‖ is the norm the GMRES and FGMRES solvers compute for their status line, tested after the solve, so the check adds no work |
| `SolverStopOnNonConvergence` | `1` | `1` stops the run when a Krylov field solve (Maxwell GMRES or FGMRES, ∇·E or ∇·B correction) reaches its restart limit (200 for Maxwell, 66 for a correction) without meeting `GMREStol` or `DivCleaningTol`; `0` only prints the solver's warning and uses the unconverged solution |

Either guard prints the reason on rank 0 and stops all ranks with `MPI_Abort` before the result of the failed solve is used. No further output or checkpoint is written, the final one included, so the restart slots keep the checkpoints written before the failure.

### Boundary rows of the implicit field solve

The Maxwell solve for E(n+θ) (GMRES or FGMRES with block Jacobi, selected by `SolverType`) needs a linear operator: the image M·v that the solver evaluates on its own basis vectors must hold no constant term. Each boundary condition is therefore split into its linear part, which goes in the image row, and its constant part, which goes in the right-hand-side row.

| face | image row | right-hand-side row |
|---|---|---|
| open inflow face (`bcEMface* = 2`, `bcPface* = 2`, X-right excluded), with `ApplyInflowBcsEImage = 1` (default) | v | E0 = −(u0×B0)/c |
| perfect conductor (`bcEMface* = 0`), tangential components | v_t | 0 (a conductor at rest) |
| perfect conductor, normal component n | the interior row, read through the conductor image (below) | the interior right-hand side, read through the image |
| every other face, among them the open X-right face and open faces with `bcPface* ≠ 2` or `ApplyInflowBcsEImage = 0` | the operator's | the physical right-hand side |

With the constants on the right-hand side, the image is the linear map that the Krylov solvers assume.

### Fields on a perfect conductor

A perfect conductor (`bcEMface* = 0`) is at rest, so the tangential E on it is zero. The identity rows of the solve impose E_t = 0 on the face nodes. Where conductor faces meet (edges, corners), every component is tangential to one of them, so all three are 0. Since GMRES meets them only to `GMREStol`, the solved E(n+θ) is set to E_t = 0 there after the solve, and so are the smoothed E after every pass and the initial E. The wall E is not the drift field −(u0×B0)/c of the plasma: a wall at rest does not carry it.

Beyond the wall the fields are continued as their image in the conductor, the image that keeps E_t = 0 and that reverses the charges and the tangential currents:
- E_n and B_t are even about the wall;
- E_t and B_n are odd about their wall values (0 for E_t, the frozen W for B_n, below).

**Cell B.** E_t is fixed on the wall, so Faraday's law freezes the normal B there. The ghost cells of the cell B are its image, the one Faraday's law gives with the image E:
- tangential B: ghost = first cell;
- normal B: its change is odd, so the wall value (the mean of the ghost and the first cell) stays at its value W at the start: ghost = 2W − first cell.

W is the mean of the node B_n over the 4 face nodes of each cell face, taken from the initial field (or the restart file) when the solver starts. The face nodes then carry B_n = W for the whole run, and Faraday's law conserves divC2N(B) on them, as it does in the interior. The GEM and ForceFree wall fixes leave the normal component to this rule. A restart takes W from the checkpoint's node B. This averages W once more along the wall, which is exact for a wall field that is uniform or linear along the wall.

**Normal E on the wall.** The face node's normal row is the interior row of the implicit solve, Ampère's law with Faraday's and Gauss's law substituted:

(I+μ)E − δ²(∇²E + ∇∇·(μE)) = Eⁿ + δ(∇×Bⁿ − 4πĴ) − 4πδ²∇ρ̂,  δ = cθΔt,

read through the conductor image. Every quantity of the field solve takes the parity of the image, in which the image charge is opposite (`imageOdd`, `FieldBoundary.h`):
- **Ghost cells:** the gradients of E have the image of E, ∂_n E_n and ∂_t E_t odd. The charge-like ∇·(μE) and ρ̂ are odd, so the face row's Gauss term ∂_n(·) is twice the wall-cell value over Δ.
- **Cell charge:** the field solve's ρ_c reads the face-node charge as 0 (real plus image); the output densities keep the folded charge.
- **Wall-cell divergences of Ĵ and μE:** they read the face's tangential components as 0 and its normal component as stored. The face Ĵ_n is the image current, the change of the induced wall charge.
- **Ĵ on the face:** its pressure term −(Δt/2)∇·Π has the ghosts of a current's image (tangential components odd), so its tangential part interpolates to 0 on the face node. Ĵ_t is set to 0 after it is formed; the Ĵ smoothing keeps it 0 through its odd ghosts, which makes it the adjoint of the E smoothing.

With these, the Gauss term of the interior rows sums by parts up to the wall, which energy consistency needs. A face row without it, or with a copy ghost of the charge, leaves a wall term that drives a growing wall mode at θ = ½. The vacuum standing wave between the walls conserves energy at θ = ½ to round-off (unit test `conductor-modes`).

**Ghost nodes of E and B** (the mover reads them for a push that ends beyond the wall; on a reflecting face that push is only a trial, see below):
- E_n and B_t: even, ghost = the value of node 2 (the first node inside);
- E_t and B_n: odd about the face value, ghost = 2·face − node 2. That is −E_t(node 2) for E, and for B the eight-cell mean of the image cells.

### Particles on a reflecting face

On a non-periodic face with `bcPface* = 1` a particle is reflected specularly, on the face and inside its push:
- a push that would end beyond the face is shortened until it ends on it, to 10⁻⁶ of the smallest cell size;
- the normal velocity is reversed there; on an edge or a corner, for every reflecting face reached;
- the rest of the step is pushed from the face, and further bounces in the same step are treated alike.

Every part of the step is the mover's own implicit push, with the fields at its own mid position, so no force acts across the face. This holds for the standard push and for each subcycle of the relativistic one, on any face and in any decomposition.

Mirroring the particle after a free push would not be a reflection. The fields beyond a conductor are its image, not the mirror image of the fields inside, so each bounce would shift the velocity along the wall by −2(q/m)B_t d/c and the energy by −2(q/m)E_n d, with d the distance beyond the face: a current layer and a heating or cooling at the wall.

A step takes at most 10 pushes; after them the rest of the step is one free push, and a particle left beyond the face is mirrored back. Only walls closer than one step's flight, or a gyration the step does not resolve (ΩΔt well above 1), come to that.

### Boundary injection (reemission faces)

On a non-periodic face with `bcPface* = 2`, every particle of every species that ends a push in the last 3 cells before the face is deleted, so planetary species leave the domain there. For every species with `rhoINJECT > 0`, the GPU then refills those cells with a drifting Maxwellian (`rhoINJECT`, the species' drift and thermal speeds), every cycle and with the same number of particles per cell as the initial loading (`npcel*`). Each injected particle draws its numbers from its own Philox stream. The key is set by the cycle and the species, and the stream number is set by the particle's global cell and sub-cell. So ranks inject different plasma, the injected plasma does not depend on `XLEN`/`YLEN`/`ZLEN`, and a restart repeats the injection of every cycle.

### Open outflow boundary (X-right)

With `bcEMfaceXright = 2` and `bcPfaceXright = 3` (and `PERIODICX = 0`), the last L = max(3, `n_layers_sal`) cells before the X-right face form the outflow layer, beyond the reference plane x_s, L cells inside the face. The field solve sees the layer as neutral, with the current and pressure of x_s. This is always on:

- `OutflowLayerFields` sets what happens to the fields on the layer after each field solve:
  - `solved` (default): the solver's E and B are kept, so on the layer they satisfy Faraday's and Ampère's laws as inside, with the layer's neutral charge and copied current; only the ghost node and the ghost cell beyond the face are copied from the face, the zero-gradient continuation the solver assumes.
  - `copy`: E and B on the layer are copied from x_s, a flat continuation of x_s. This holds the field at the face at its value at x_s, which removes the outward push of a tangential field that weakens toward the face; where slow plasma drains through the face along field lines tied elsewhere, the flux then piles up behind x_s.
- Before each solve, the species densities on the layer are set to the x_s values, rescaled to equal positive and negative charge, (ρ₊ + |ρ₋|)/2 each; where a node of x_s holds charge of one sign only, the layer nodes in its line are emptied. J_s and the pressure tensors on the layer are copied from x_s.
- The predicted charge ρ̂ is zero on every cell beyond x_s, including the cell between x_s and the first layer node, so the solver's neutral exterior starts at x_s (where the field copy starts with `OutflowLayerFields = copy`).
- Inside the field solver, the face is closed with a zero-gradient ghost node (the Krylov vector is unchanged), and the E smoothing continues all components evenly about the face node, as on every physical face.
- With `divBCorrection`, the ∇·B cleaning corrects B on every cell, the outflow layer included (see below).

The written `rho`, `J` and pressure on the layer are these neutralized or copied values; the written E and B are the copied or the solved fields.

Particles, on every face with `bcPface* = 3`: a particle that ends a push beyond the face is deleted. The plasma beyond the face is the last 3 cells shifted 3 cells outward, moving as its sources, so a particle that crosses the plane 3 cells inside the face inward during the push re-enters. A copy with its new velocity and charge is appended at its end position shifted 3 cells outward, and pushed from the next cycle. Beyond an edge or a corner between such faces the plasma is shifted across each of them, so a particle that crosses these planes of two faces in one push gives three copies (one shifted across each face and one across both), and of three faces seven. A copy is kept if it lies inside the domain and outside the injection layers of the `bcPface* = 2` faces, which are refilled every cycle; a copy beyond a periodic boundary is wrapped by the particle exchange, like its source.

### Absorbing layer and ∇·B cleaning on open faces

On an open face (`bcEMface* = 2`) with `yes_sal = 1` (the default), every cycle blends the fields once toward the upstream values on the layers 0…`n_layers_sal`: E toward −(u0×B0)/c after the field solve, and B toward B0 after Faraday's law, before B is interpolated to the nodes. The weight s rises linearly from 0 on the ghost layer by 1/`n_layers_sal` per layer to 1 inside, so the face node of E and the first cell of B get 1/`n_layers_sal`. X-right is excluded: it always continues the interior (see above). Below, B_n is the component of B normal to the face and B_t the tangential ones.

- `sal_tangential_B = 1` (default): only B_t is blended; B_n is left to the field solve.
  - Blending B_n creates ∇·B of about (B_n − B0_n)/(`n_layers_sal` Δx) in the layer every cycle, however smooth the field is. Blending only B_t creates (1 − s) ∂_n B_n, which the cleaning drives to zero.
  - B_n across the layer then follows the interior field (zero normal gradient) instead of being clamped to B0_n.
- `sal_tangential_B = 0`: all three components are blended.

Without the SAL (`yes_sal = 0`), every open inflow face (`bcEMface* = 2` and `bcPface* = 2`, X-right excluded) holds E0 = −(u0×B0)/c on its node layers 0…`n_layers_sal` after the field solve, the value its rows impose on the face node, and B0 (all three components) on its cell layers 0…`n_layers_sal` after Faraday's law: the fields in which the plasma injected there drifts at u0. An open X-left face gets B0 whatever its `bcPface*`. On the other open faces (`bcPface* ≠ 2`; X-right, see above) E is left as solved, and B on the layers 0…`n_layers_sal` is copied from the next cell layer inward (zero-gradient continuation).

The B treatment of an open face (the blend; without the SAL, B0 or the zero-gradient continuation) runs only on wall ranks with at least 9 cells along the face normal (a gate meant for the flat direction of a 2D run, which is periodic and never reaches it). On a thinner wall rank of any other direction it is skipped, and the code and `scripts/ipic_preflight.py` print a warning: use fewer ranks along that direction (at least 9 cells per wall rank).

`divBCorrection = yes` cleans the cell-centred B every `divBCorrectionCycle` cycles, in the B update (`gpuCalculateB`) after Faraday's law and the face treatments, before B is interpolated to the nodes. It needs at least 2 cells along every non-periodic direction; otherwise the run stops at startup. Faraday's law keeps divC2N(B) at round-off (divC2N∘curlN2C = 0), so the cleaning removes what the face treatments create; where that is round-off the solve is skipped (see below).
1. It solves divC2N(gradN2C ψ) = divC2N(B) for ψ on the nodes (preconditioned FGMRES, see below). The nodes of a physical face are not unknowns: on a perfect conductor ψ there equals the next inner node (zero normal gradient), so B_n of the wall cells, and so at the wall, is left as Faraday's law made it; on the other faces ψ = 0 there, so the correction has no tangential part on the face.
2. It subtracts gradN2C ψ from B on the cells. Since the solve uses the composite operator divC2N∘gradN2C itself, this is the projection of B onto fields without divergence at the interior nodes (to `DivCleaningTol`): a field without divergence is left alone, and nothing filters B. Because it keeps B_n on a perfect conductor, the projection is orthogonal only where B does not cross the conductor: the B energy changes by −‖gradN2C ψ‖² − 2∮ψ B_n, so where the initial B crosses a wall it can rise (by a few percent in a cold test with y walls).

Use the cleaning with the tangential SAL. With `sal_tangential_B = 0`, the cleaning carries the SAL's B_n = B0_n at the faces into the whole box, and the code prints a warning. Without the SAL, an open X-left face and the open Y/Z inflow faces set the whole B, B_n included, to B0 every cycle, and the cleaning carries that face flux into the box; the code warns about this too. Without the cleaning, the tangential SAL lets ∇·B pile up in the layer.

### Solves of the ∇·E and ∇·B corrections

`PoissonCorrection` (∇·E) solves divN2C(gradC2N φ) = divN2C(E) − 4πρ for φ on the cells, and subtracts gradC2N φ from E on the nodes.

The ghost cells of φ:
- **On a perfect conductor (`bcEMface* = 0`):** the conductor image of φ, odd (`GhostSet::Phi`).
  - φ = 0 on the wall, which stays an equipotential.
  - The correction has no tangential part there.
  - The solve fills these ghosts after the halo exchange of every iteration.
- **On the other physical faces:** 0.

With these rules gradC2N is minus the transpose of divN2C, in the lumped inner product with E_t = 0 on the walls. The step is therefore the orthogonal projection onto the E of the discrete Gauss law.

After the correction:
- the values of the identity rows are restored: a conductor's E_t = 0 exactly, E0 on inflow faces;
- Eⁿ gets the boundary of a solved E (`EMfields3D::finishE`): its ghost nodes are exchanged and filled, then the layers of the open faces are set (inflow E0 or the absorbing blend, the X-right copy).

So for θ < 1 the advance E^{n+1} = (S(E^{n+θ}) − (1−θ)Eⁿ)/θ keeps the imposed face values. Both corrections solve with the operator of their correction, the 27-point Σ_a δ_a² μ_b² μ_c² (δ² the (1, −2, 1)/h² difference, μ² the (¼, ½, ¼) average), not with a 7-point Laplacian, which would remove only part of the divergence.

The solves use FGMRES(60), at most 66 restarts (about 4000 iterations), to `DivCleaningTol`: their vectors are at most a third of the Maxwell ones, so the GMRES workspace holds three times the Maxwell restart length, which the near-null modes of the averaged operators in 3D need. FGMRES is preconditioned on each rank by a Chebyshev polynomial of degree `PoissonChebMaxIter` + 1 in the rank's local operator (block Jacobi, zero values at rank cuts, the face rules on the physical faces). Along each axis μ² = 1 − T with T = −h²δ²/4 (the same end rules for both), so the eigenvalues of that operator are Σ_a (4/h_a²) t_a Π_{b≠a} (1 − t_b) over the eigenvalues t_a of the 1D T, and its exact bounds [λ_min, λ_max] come from the extreme 1D values.

On [λ_min, λ_max], the polynomial leaves a residual of at most 1/T_k(σ) ≈ 1/cosh(2k √(λ_min/λ_max)). With the exact minimum of a large block, this is close to 1: λ_max/λ_min ≈ 1.5·10⁵ for a 960 × 768 2D block. The preconditioner then does almost nothing, and the solves become slow or stall.

`PoissonChebEigRatio` (default 300) therefore raises the lower bound to max(λ_min, λ_max / `PoissonChebEigRatio`). The polynomial then maps the upper spectrum to within 1/T_k(σ) ≈ 0.18 of 1 (0.03 with 100), and FGMRES resolves the modes below the bound, which the polynomial still inverts with the right sign. `PoissonChebEigRatio = 0` restores the exact bounds. In 3D the averaged operators also have near-null checkerboard modes: near θ = π on all three axes every term of Σ_a δ_a² μ_b² μ_c² has two averages, each about ε²/4 with ε = π/(N + 1), so the faces lift these modes only to λ ≈ (3/4h²) ε⁴ for cubic cells, about 10⁻⁶ λ_max for 63 nodes per axis (with two axes near π, and in 2D, one average remains: λ ~ ε², like the smooth modes). No polynomial of this degree reaches them, so FGMRES resolves them. They hold little of the divergence, so they cost iterations only at tight tolerances.

The exact upper bound is the best one: `PoissonChebRescaleEigMax` below 1 leaves the top modes outside the interval, where the polynomial grows (0.95 multiplies the iterations by about 10), and above 1 it only spreads the upper cluster. A lower floor (a larger `PoissonChebEigRatio`, or `PoissonChebRescaleEigMin` below 1) trades iterations at loose tolerances for fewer at tight ones, and pays on one rank rather than on many.

A cleaning leaves the fraction `DivCleaningTol` (default 10⁻⁶) of the divergence it found. A 10⁻⁶ solve costs 2 to 11 times a 10⁻³ one.

The operators are singular in two cases, and the right-hand side then loses its components on the null space before the solve (the gradient does not depend on them):
- Constants: for ∇·E with all three directions periodic (removing the mean charge, a uniform neutralising background), and for ∇·B when every face is periodic or a perfect conductor (`bcEMface* = 0`). Between perfect conductors the mean of divC2N(B) is the net flux of the wall cells' B_n, which no correction can change; a uniform ∇·B of that size remains.
- Checkerboards (−1)^(i_a+i_b) f(i_c), for any profile f along the third direction c, of two directions a, b that each carry the checkerboard: the averaged gradients map them to zero. A direction carries it when it is:
  - periodic with an even cell count;
  - for ∇·E, between two perfect conductors, whatever its cell count (the odd ghosts of φ).

  The charge density can hold them, and no E produces them through divN2C (with E_t = 0 on the walls). That part of the charge stays in the Gauss residual. divC2N(B) holds them only at round-off.

A right-hand side at the round-off of the fields it comes from leaves nothing to clean, and the solve is skipped: ‖b‖ ≤ 10³ ε ‖B‖ (Σ_a h_a⁻²)^½ for ∇·B, ‖b‖ ≤ 10³ ε (‖E‖² Σ_a h_a⁻² + 16π² Σ_s ‖ρ_s‖²)^½ for ∇·E.

### Planet surface (`Dipole`, `Dipole2D`)

Particles that enter the planet (`Planet_radius`) are processed on the GPU every cycle, on each rank:

- All ions are absorbed.
- Electrons of all species are ranked together by the kinetic energy of one physical electron (relativistic, independent of the macroparticle weight) and absorbed from the most energetic down, until the absorbed charge best matches the charge to neutralise: the ions absorbed in this cycle plus the imbalance carried from earlier cycles. What is left, at most half an electron macroparticle unless too few electrons hit, is carried to the next cycle. The carried charge is not written to restart files; a restart starts it from zero.
- The other electrons are reflected, specularly or diffusely (`planetReflectionType = 0` or `1`), about the outward normal at the point where the push left them (radial; in 2D, the in-plane normal of the cylinder). An electron that ended at depth d below the surface is put on the same ray at R + d, the mirror image of that point, but at least at 1.05 R (R = `Planet_radius`; in 2D, y is kept). A push ends less than c·dt inside, so the mirror point lies beyond 1.05 R only when c·dt > 0.05 R; on larger planets every reflected electron starts at 1.05 R. The exception is the first cycle of a run: the electrons loaded inside the planet are processed then, and those not absorbed are placed at up to 2R. A specular reflection reverses only an inward normal velocity component, so an electron already moving outward keeps its velocity; a diffuse one keeps the speed and draws the direction by the Lambert cosine law. Either way the electron never leaves moving inward (v·n ≥ 0). It may land on another rank, which the particle exchange handles; one placed beyond a domain face (a planet centre within 2R of it) gets that face's particle boundary condition, and an exit or open face removes it.

Every `PlanetDiagnosticsCycle` cycles (default `DiagnosticsOutputCycle`; `<= 0` disables), rank 0 appends the surface balance to `PlanetBC.txt` in `SaveDirName`: per species, the macroparticles and charge that hit the planet and that were absorbed; the carried charge, summed over ranks and its extremes; the mean energy separating absorbed from reflected electrons; and the rank-cycles with hits and with too few electrons to balance the ions. A row labelled N covers the hits since the previous row, up to cycle N-1; a last row at the end of the run, labelled with the resume cycle like the final restart checkpoint, covers the rest.

### Particle count control (splitting and merging)

Flows move particles between MPI ranks, and open boundaries and the planet remove them, so the number of macroparticles of a rank drifts away from the number it was loaded with. Two options, both off by default, keep it in a band, after Lapenta and Brackbill, J. Comput. Phys. 115, 213 (1994):

| Parameter | Default | Description |
|-----------|---------|-------------|
| `ParticleSplitting` | `0` | Split particles of a species on a rank that holds fewer than `ParticleSplittingThreshold` times its reference count. |
| `ParticleMerging` | `0` | Merge particles of a species on a rank that holds more than `ParticleMergingThreshold` times its reference count. |
| `ParticleSplittingThreshold` | `0.95` | Lower edge of the band, as a fraction of the reference count. |
| `ParticleMergingThreshold` | `1.05` | Upper edge of the band, as a fraction of the reference count. |

The reference count of a species on a rank is the number of particles the input file loads there and the planet leaves: `npcelx * npcely * npcelz` times the cells of the rank whose centre is outside `Planet_radius`. It does not depend on the state of the run, so a restart continues with the same reference. A rank that loads another number of particles prints a warning at start. Planetary species (`ns_planetary`), which start empty, are not controlled. With only one of the two options on, the count is bounded on one side only; the run prints a warning and continues.

- **Splitting.** Before the mover, a rank below the band splits as many particles as it misses, each once per cycle at most, so a rank below half its reference at most doubles per cycle. A split also leaves the particle arrays the spare room the mover needs; they grow in the exchange that follows. A rank that has lost all particles of a species is not refilled. The parents are spread evenly over the particles of the rank, wherever they are, and change from cycle to cycle. A parent is replaced by two particles of half its charge and the same velocity, displaced by +δ and −δ along one grid axis (never an axis with a single cell in the domain), inside the cell of the parent. δ is half the loading interval of that axis (cell size / particles per cell), less for a parent near a cell face. Along one axis the weight of a node is linear, so the pair deposits exactly the charge, current and pressure of the parent on every node: the fields do not see the split. A species is not split in a cycle in which it is merged (possible only with `ParticleSplittingThreshold` above `ParticleMergingThreshold`).
- **Merging.** After the moments, the species above the band that was merged longest ago is chosen; one species is merged per cycle and rank. Before the next mover it is sorted by cell, and in every cell that holds more than `npcelx * npcely * npcelz` particles, each particle in turn looks for the nearest later particle in velocity. The two are merged if |v₁ − v₂|² is at most 0.9 % of the velocity variance of the cell (about the mean velocity of the cell, so the test does not depend on the frame). The merged particle has the total charge, the centre of charge and the mean velocity of the pair. A particle takes part in one merge at most, and a cell stops at `npcelx * npcely * npcelz` particles.
  - A merge keeps the charge, the momentum and the centre of charge of the pair exactly.
  - It loses the kinetic energy of the relative motion, m₁m₂|v₁ − v₂|² / (2(m₁ + m₂)): at most 0.23 % of the mean thermal energy of a pair of equal particles.
  - It keeps the ten moments on the nodes only for a pair along a grid axis with equal velocities. Otherwise it moves charge, current and pressure between the eight nodes of the cell; their total charge is kept.
  - The search compares all pairs of a cell: its cost grows with the square of the particles per cell.

The control acts on the particle count of a rank, not of a cell: it does not refill the cells of a depleted region of a rank whose total count is inside the band.

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
| `pertGEM` | `0.0` | GEMHarris | GEM flux perturbation (Birn et al. 2001): flux $\psi = $ `pertGEM` $B_{0x} L_x/(2\pi)\,\cos(2\pi x/L_x)\cos(\pi y/L_y)$ from the box centre, so $\delta B_y$ peaks at `pertGEM` $B_{0x}$ and vanishes on the $y$ walls |
| `pertHump` | `0.0` | GEMHarris | Hump perturbation amplitude |
| `deltaxHump` | `8.0` | GEMHarris | Hump width in x (multiplied by `delta`) |
| `deltayHump` | `4.0` | GEMHarris | Hump width in y (multiplied by `delta`) |
| `kxHump` | `-1.0` | GEMHarris | Hump wave number x (if ≥ 0) |
| `kyHump` | `-1.0` | GEMHarris | Hump wave number y (if ≥ 0) |
| `currentFromAmpere` | `0` | GEMHarris | `1`: derive drift velocity from $\nabla\times\mathbf{B}$ instead of using `u0`/`v0`/`w0` directly |
| `u0`, `v0`, `w0` | — | ForceFree, GEMHarris with `currentFromAmpere = 1` | Weights of the current split: species $s$ carries the share $\operatorname{sign}(q_s)\,u_{0,s} / \sum_r \operatorname{sign}(q_r)\,u_{0,r}$ of $J_x$ ($v_0$ for $J_y$, $w_0$ for $J_z$; GEMHarris: $w_0$ splits the Harris current $J_z$); a component whose weights sum to 0 gets no current. In every case the run stops at startup if a loaded drift reaches $c$ |
| `spatiallyVaryingThermal` | `0` | GEMHarris | `1`: the sheet (drifting) species get the temperature $T_s\,e^{-2A_p/(B_{0x}\delta)}$, $A_p$ the flux function of the perturbations ($B_x = \partial_y A$, $B_y = -\partial_x A$), so that their pressure follows the Harris law $p_0\,e^{-2A/(B_{0x}\delta)}$ along the perturbed field lines; the background stays isothermal. Needs `Case = GEMHarris`, `currentFromAmpere = 1`, `B0x` ≠ 0 and `B0y` = 0 (checked at startup) |

## Output Control

Output is controlled through **cycle parameters** (how often to write) and **tag strings** (what to write).

### Output cycles

| Parameter | Default | Description |
|-----------|---------|-------------|
| `FieldOutputCycle` | `100` | Write field/moments data every N cycles. `0` disables field output. |
| `ParticlesOutputCycle` | `0` | Write particle data every N cycles. `0` disables particle output. |
| `RestartOutputCycle` | `5000` | Write restart checkpoint every N cycles. `0` disables periodic checkpoints. |
| `DiagnosticsOutputCycle` | `FieldOutputCycle` | Write scalar diagnostics (energy, etc.) every N cycles. Defaults to `FieldOutputCycle` if not set. |
| `PlanetDiagnosticsCycle` | `DiagnosticsOutputCycle` | Append the planet surface balance to `PlanetBC.txt` every N cycles and at the end of the run (`Dipole`, `Dipole2D`). `<= 0` disables. |
| `TestPartOutputCycle` | `0` | Write test-particle trajectory data every N cycles. Test particles are written by the serial-HDF5 writer only (`WriteMethod = shdf5`, or `USE_ADIOS2=OFF`). |
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

**All-species** (one file or dataset per species):

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

Additional ADIOS2-only tokens used for restart data: `proc_topology`, `E`, `B`, `Js`, `rhos`, `pressure`, `particle_cell_metadata`.

### Other output parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `WriteMethod` | — | Field output backend (see [I/O Backends](#io-backends) below) |
| `SaveDirName` | `data` | Output directory. **Warning:** existing contents are deleted on fresh start. |
| `RestartDirName` | `data` | Restart root containing `latest_restart.json` and `restart_A` / `restart_B` |
| `CallFinalize` | `1` | Write a final restart checkpoint when the simulation ends (requires `RestartOutputCycle > 0`) |
| `ParaviewScriptPath` | `""` | Path to ParaView Catalyst Python script (requires `USE_CATALYST`) |

### Data analysis

Every `DataAnalysisCycle` cycles a host thread analyses the particles of every species while the field solver runs, on a GPU stream of the lowest priority: the field solver, of higher priority, gets the GPU first. It computes the 3D velocity histogram of each rank, or of each macrocell of cells, its fit by a Gaussian mixture (GMM), which compresses the velocity distribution into a few Gaussians, and the (v∥, v⊥) spectra relative to the local B in macrocells. The macrocell analyses sort the particles of their species by cell first; a macrocell takes the particles of its interior cells; a particle exactly on a face of the rank, which the sort can put in a ghost cell, is left out of the macrocell analyses (the histogram of a rank counts it). Each analysis writes, per rank, one file of binary records appended in cycle order, `<folder>/subDomain<rank>_<analysis>.bin`, and the `.json` of the same name that describes it: the rank, its regions (the rank, or the macrocell tiling), the bins and the layout of the records; the histograms and the spectra keep their non-empty bins only, and the records of a cycle are written at once. `postprocessing_tools/python/read_analysis_records.py` reads them; `plot_velocity_histograms.py` plots the histograms with their GMM fits (projections with the ellipses of the components, marginals with the weights of the mixture), or the fits alone from the `velocityGMM/` files (`--gmm-only`), and `plot_macrocell_spectra.py` the spectra. All analyses are off by default; with none on, no cycle analyses. `BENCHMARK_MODE = 1` runs them without writing, `2` leaves them out of the build. Per-species options take one value for every species or one per species; on/off options take `0` or `1`. On a planet run the first analysis also sees the particles loaded inside the planet, which the first mover removes.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `DataAnalysisCycle` | `50` | Analyse every N cycles; `0` disables all analyses |
| `DataAnalysisOutputDir` | `./` | Folder of the subfolders `velocityHistogram/`, `velocityGMM/`, `macrocellSpectra/`, each with, per rank, its records file (`subDomain<rank>_histograms`, `_gmm`, `_spectra`: `.bin` and `.json`) and its mapping `vctMapping_subDomain<rank>.txt`; a fresh start empties those of the analyses that are on and keeps the records and rank mappings of an earlier run (of all its ranks) of those that are off as `*_before_cycle0.*` (so that a restart neither cuts nor continues them), a restart cuts the records from its first cycle on (and a record cut short), which it writes again, and appends; it cuts those of an analysis switched off too, also with every analysis off (a file of another record format is kept aside instead). A restart onto another decomposition (restart remapping), or whose regions, macrocells or bins differ from those of the records, warns and keeps the records files (and on a remap the mappings, of every rank of the earlier run) as `*_before_cycle<N>.*` (`_2`, `_3`, ... added when such a file exists), then starts them again; on a remap, those of the analyses switched off are left as they are |
| `VelocityHistogram` | `0` | 3D (u, v, w) histogram of every species, weighted by \|q\| and summed exactly (fixed point), so the same in every run and for any order of the particles. Per rank, 100 bins per axis. The non-empty bins of every cycle and species are a record of `velocityHistogram/subDomain<rank>_histograms.bin`: the range of each region, its number of non-empty bins, their contents (float32) and their indices (u fastest; uint32 for a rank, uint16 for a macrocell) |
| `VelocityHistogramMacrocells` | `0` | `1`: a histogram (and with `VelocityGMM` a fit) per macrocell (`VelocityHistogramMacrocellNx`, `Ny`, `Nz`) instead of one per rank, 32 bins per axis (128 kB of GPU memory per macrocell, printed at the start, and up to 32 bytes per non-empty bin for the GMM points and the output). The records then hold one region per macrocell, the `.json` the tiling. Needs `VelocityHistogram = 1`. A restart that changes this option or the macrocell sizes of the histograms keeps the histogram and GMM records of the rank aside (`*_before_cycle<N>.*`) and starts them again |
| `VelocityHistogramMacrocellNx`, `VelocityHistogramMacrocellNy`, `VelocityHistogramMacrocellNz` | `MacrocellNx`, `MacrocellNy`, `MacrocellNz` | Macrocell size in cells of the macrocell histograms and their GMM fits, each axis by default the common size. With `VelocityHistogramMacrocells = 1` all three must be at least 1 and at most the cells of a rank along that axis (1 on the flat axis of a 2D run), and give a rank at most 65535 macrocells (an int indexes their bins, 8 GB of them) |
| `VelocityHistogramFixedRange` | `1` | Bin [`VelocityHistogramMin`, `VelocityHistogramMax`] on the three axes; `0`: the range of the finite velocities of the species at that cycle in the rank, or in the macrocell (on an axis where they have no spread, or bins not wider than 4 ε of the largest velocity (4 to 8 units in the last place), the width max − min with their middle at the centre of the middle bin; the fixed range on an axis without finite velocities) |
| `VelocityHistogramMax` | `0.2` if `qom < 0`, else `0.09` | Per species, positive, above `VelocityHistogramMin` by a finite width. The spectra bin v_par in [-max, max] and v_perp in [0, max] |
| `VelocityHistogramMin` | `-VelocityHistogramMax` | Per species |
| `VelocityGMM` | `0` | Fit every histogram (needs `VelocityHistogram = 1`) on its non-empty bins, weighted by their content. Each rank appends to one file, `velocityGMM/subDomain<rank>_gmm.bin`, a record per analysis cycle with the fits of every species and region (the rank, or each macrocell): the log-likelihood, the weights, means and covariances of the components (float64) and the EM steps (-1 for a region without particles in its range); the `.json` of the same name gives the regions (the macrocell tiling) and the layout, and `postprocessing_tools/python/read_analysis_records.py` reads them. The fits of the macrocells run together on the GPU. The log-likelihood, in the units of the velocities, is that of the parameters before the last EM step; pruned components stay in the record with weight 1e-15 |
| `GMMComponents` | `10` | Gaussians of each fit, 1 to 256. The GPU memory of a fit grows with them: 8 bytes per component and non-empty bin and 8 more per bin, up to 10^6 bins for a rank (8 MB per component and 8 MB) and 32768 for a macrocell, the macrocells fitted in batches of about 185 MB. The last fit of each species and region is kept on the host, 104 bytes per component. A restart may change it: each record gives its components |
| `GMMMaxIterations` | `100` | Expectation-maximisation iterations at most, plus one when the last one prunes a component, which gives its weight back to the others |
| `GMMConvergenceThreshold` | `1e-5` | Stop when the log-likelihood per unit weight of the histogram gains less, or decreases (from the 7th iteration on) |
| `GMMWarmStart` | `1` | Start from the previous fit of the species and region (of this run: not after a restart), drawing again the components that were pruned or are close to it; otherwise from equal weights, random means in the ball inscribed in the range of the bins and the diagonal covariance `uth²`, `vth²`, `wth²` |
| `GMMNormalize` | `1` | Fit the velocities divided by the largest velocity of the bins of the histogram (`VelocityHistogramMax` with the default range) |
| `GMMCheckCovariance` | `1` | Reset the non-finite elements of each covariance and raise its variances to `GMMMinVariance` |
| `GMMCovarianceTolerance` | `1e-9` | Smallest pivot of the Cholesky factorisation of a covariance, in the units of the fit: a nearly singular covariance is raised on its diagonal to stay invertible, and is written so |
| `GMMMinVariance` | `1e-4` | With `GMMCheckCovariance`, in the units of the fit: the smallest variance of each velocity given the previous ones (the Cholesky pivot, raised as with `GMMCovarianceTolerance`), so also on the diagonal, which bounds the density of a component by that of a round one of this variance; the default suits `GMMNormalize = 1` |
| `GMMPruneComponents` | `1` | Drop the components of weight below `GMMPruneThreshold`, the smallest first, one at a time |
| `GMMPruneThreshold` | `0.005` | Weight below which a component is pruned; with pruning, below 1 / `GMMComponents`, the weights of a cold start |
| `VelocitySpectra` | `0` | Spectra of the species in `VelocitySpectraSpecies`, 64 × 64 bins per macrocell: the non-empty bins of every cycle and species of the macrocells are a record of `macrocellSpectra/subDomain<rank>_spectra.bin` (bin index v_par fastest, uint16); a particle where \|B\| is at most 1e-12 is left out. A restart that changes the macrocell sizes of the spectra keeps the spectra records of the rank aside (`*_before_cycle<N>.*`) and starts them again |
| `VelocitySpectraMacrocellNx`, `VelocitySpectraMacrocellNy`, `VelocitySpectraMacrocellNz` | `MacrocellNx`, `MacrocellNy`, `MacrocellNz` | Macrocell size in cells of the spectra, each axis by default the common size. With `VelocitySpectra = 1` all three must be at least 1 and at most the cells of a rank along that axis, and give a rank at most 524287 macrocells (an int indexes their bins) |
| `MacrocellNx`, `MacrocellNy`, `MacrocellNz` | `0` | Common macrocell size in cells: the default of the sizes of the spectra and of the macrocell histograms, which are the same unless their own keys say otherwise. With two sizes, macrocell m of the spectra is not macrocell m of the histograms: each `.json` gives its own tiling. Along an axis where the cells of a rank are not a multiple of the size, the last macrocell has the remaining cells, fewer; on a rank with fewer cells than the size (a last rank when `nxc` is not a multiple of `XLEN`), the only one. When a rank has few macrocells (fewer GPU blocks than 4 per multiprocessor), several blocks share the cells of each: the same histograms bit for bit, the spectra summed in another order (float atomics). The start of the run prints the size of each analysis: a misspelt key is not an error and leaves the default |
| `VelocitySpectraSpecies` | all `0` | Per-species mask of the spectra |

The bins are compile-time constants in `src/include/dataAnalysis/dataAnalysisConfig.cuh`.

## I/O Backends

iPIC3D-GPU writes three categories of data: **fields** (E, B, J, rho, moments), **particles** (position, velocity, charge, ID), and **restart checkpoints** (fields + particles). Each category can use a different I/O backend, selected through a combination of CMake options and the `WriteMethod` parameter in the input file.

### CMake options

| Option | Default | Effect |
|--------|---------|--------|
| `USE_HDF5` | `ON` | Compile HDF5-based backends (serial HDF5, parallel HDF5) |
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
| `adios2` | ADIOS2 | **Not implemented yet** — will throw a runtime error |

Any other value stops the run at startup with an error.

### Particle and restart backend selection

Particle and restart backends are **not** controlled by `WriteMethod`. They are selected at **compile time** based on the CMake flags:

| | `USE_ADIOS2=ON` | `USE_ADIOS2=OFF` |
|---|---|---|
| **Particles** | ADIOS2 (file-per-process, BP5) | Serial HDF5 (file-per-process) |
| **Restarts** | ADIOS2 (file-per-process, BP5) | Serial HDF5 (file-per-process) |

When `USE_ADIOS2=ON`, ADIOS2 is used for particles and restarts, regardless of `WriteMethod`.

### Available combinations summary

With the default CMake settings (`USE_HDF5=ON`, `USE_PHDF5=ON`, `USE_ADIOS2=ON`):

| `WriteMethod` | Fields | Particles | Restarts |
|---------------|--------|-----------|----------|
| `pvtk` | Blocking VTK (collective) | ADIOS2 (file-per-process) | ADIOS2 (file-per-process) |
| `nbcvtk` | Non-blocking VTK (collective, overlapped) | ADIOS2 (file-per-process) | ADIOS2 (file-per-process) |
| `shdf5` | Serial HDF5 (file-per-process) | ADIOS2 (file-per-process) | ADIOS2 (file-per-process) |
| `phdf5` | Parallel HDF5 (collective, single file) | ADIOS2 (file-per-process) | ADIOS2 (file-per-process) |

With `USE_ADIOS2=OFF` and `USE_HDF5=ON`:

| `WriteMethod` | Fields | Particles | Restarts |
|---------------|--------|-----------|----------|
| `pvtk` | Blocking VTK (collective) | Serial HDF5 (file-per-process) | Serial HDF5 (file-per-process) |
| `nbcvtk` | Non-blocking VTK (collective, overlapped) | Serial HDF5 (file-per-process) | Serial HDF5 (file-per-process) |
| `shdf5` | Serial HDF5 (file-per-process) | Serial HDF5 (file-per-process) | Serial HDF5 (file-per-process) |
| `phdf5` | Parallel HDF5 (collective, single file) | Serial HDF5 (file-per-process) | Serial HDF5 (file-per-process) |

### Notes

- All writes run on the host while the GPU mover of the same cycle runs; **`nbcvtk`** additionally leaves its MPI-IO write in flight until the next output cycle.
- **Parallel HDF5 (`phdf5`)** writes fields only — parallel particle output is not implemented for this backend.
- **ADIOS2** uses the BP5 engine with `MPI_COMM_SELF`, producing one `.bp` directory per MPI rank. ADIOS2 field output is declared in the code but not yet implemented.
- Apart from that `nbcvtk` write, each backend's write call returns only when its data are written.

## Tool

### Benchmark
In [benchmark](./share/benchmark/) folder, we prepared some scripts for profiling, please read the [benchmark/readme](./share/benchmark/readme.md) for more information.


## Contact

Feel free to contact Professor Stefano Markidis at KTH for using iPIC3D. 
