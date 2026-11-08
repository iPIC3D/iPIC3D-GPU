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
| `USE_HDF5` | `ON` | Compile HDF5-based I/O backends (serial HDF5, parallel HDF5) |
| `USE_PHDF5` | `ON` | Enable parallel HDF5 field output (requires `USE_HDF5=ON` and MPI-enabled HDF5). Auto-disabled when `USE_HDF5=OFF` |
| `USE_ADIOS2` | `ON` | Compile ADIOS2 backend for particles and restarts |
| `USE_CATALYST` | `OFF` | Enable ParaView Catalyst in-situ visualization (requires ParaView ≥ 5.7) |
| `USE_BATSRUS` | `OFF` | Enable BATS-R-US MHD coupling (adds `BATSRUS` compile definition) |
| `USE_OPENMP` | `ON` | Enable OpenMP in the CPU solver. **Delete CMake cache when changing this.** |
| `GPU_SOLVER` | `OFF` | Run the whole field solver on the GPU (requires CUDA-aware MPI); `OFF` uses the CPU field solver |
| `HALO_OVERLAP` | `OFF` | Overlap halo exchanges with computation in the GPU field solver (requires `GPU_SOLVER=ON`) |
| `GPU_CYCLE_DIAGNOSTICS` | `OFF` | Instrument one GPU timestep and the Maxwell right-hand side that consumes it (requires `GPU_SOLVER=ON`) |
| `GPU_DIAGNOSTIC_CYCLE` | `1` | Cycle instrumented by `GPU_CYCLE_DIAGNOSTICS` (`-1`: every cycle) |
| `BUILD_UTESTS` | `OFF` | Build the unit tests (run with `ctest` in the build directory) |
| `TIME_TASKS` | `OFF` | Print per-cycle and cumulative task timings (`LOG_TASKS_TOTAL_TIME`) |
| `BENCHMARK_MODE` | `0` | Compile-time benchmark level: `0` normal; `1` disables disk output and output-related device-to-host copies; `2` additionally disables data analysis, macrocell spectra, and heat-flux calculations |
| `BUILD_SHARED_LIBS` | `ON` | Build shared libraries (`OFF` for static) |
| `SITE` | `default` | Select a predefined site configuration from `cmake/sites/` |

`BENCHMARK_MODE` removes output-related device-to-host transfers. Transfers
required by the simulation itself—such as CPU-solver moment copies, MPI/planet
particle exchange, and GPU-solver scalar reductions—remain enabled. At level 1,
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

On restart, the reader uses `latest_restart.json` to select the newest valid slot. If that slot is incomplete, it tries the previous slot recorded in the metadata. Legacy flat restart layouts directly inside `RestartDirName` are not supported. The slot manifest records the cycle label, backend, source rank count, and mesh metadata used for compatibility checks and remapping.

### Restart remapping

Restart files store only active data: fields and species density on active nodes (`NXN - 2`, `NYN - 2`, `NZN - 2`), and particles sorted by active cell with per-species offset/count metadata. Ghost nodes are rebuilt after loading.

The restart reader can load a checkpoint onto a different MPI topology when all of these conditions hold:

- The global mesh cell counts (`nxc`, `nyc`, `nzc`) are unchanged.
- The destination topology refines the source topology by powers of two in each direction; coarsening and arbitrary factors are not supported.
- The checkpoint contains active-node fields and active-cell particle metadata. Current restart files write this format.

`RestartRemapPlan` maps each destination rank to overlapping source active boxes. Fields are copied into guarded arrays at `[1][1][1]`; particles are read as the sorted spans belonging to the destination rank.

### Particle restart loading

Particle reads are span-based and chunked. Cell offset/count metadata is converted into merged spans, the destination host SoA is resized once, then spans are streamed through temporary buffers of at most `1 << 20` particles per component.

The host restart SoA is sized to the restarted count, padded to `DVECWIDTH`. CUDA device SoA allocation still uses `INITIAL_CAPACITY_FACTOR` and can expand later.

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
2. **Fields** — the case-specific field initialiser runs, then `read_field_restart()` loads B, E, and rho, remapping active-node boxes when needed. Ghost nodes and center fields are rebuilt after load.
3. **Particles** — `restartLoad()` reads positions, velocities, charges, and IDs from the selected slot. With remapping, active-cell metadata selects the source spans for this destination rank.
4. **Output directory** — the output folder is **not** cleared on restart. Output files are opened in append mode.

### Requirements

- Restart requires either **ADIOS2** (`USE_ADIOS2=ON`) or **HDF5** (`USE_HDF5=ON`) at compile time. If neither is available, restart will produce a fatal error. The HDF5 restart path is less tested than ADIOS2.
- Restart requires the same global mesh dimensions and compatible boundary/input parameters. The MPI topology may either match the checkpoint topology or be refined by powers of two in each direction, as described above.
- A restart may use a different build than the one that wrote the checkpoint; this gives a warning, not an error (see [Build and run record](#build-and-run-record)).

## Build and run record

Every run records the code that produced it, so it can be reproduced later. At configure time and before every build, CMake (`cmake/GenerateBuildInfo.cmake`) records:

- the **version**, `git describe` of the commit, with `-dirty` if the working tree had uncommitted changes (modified, staged or untracked files that git does not ignore; files inside the build directory do not count, and an in-source build counts only tracked files);
- the **build signature**, one line with the full commit hash, a hash of the uncommitted changes (if any), the build type, the CMake options that change results (`CUDA_ARCH`/`HIP_ARCH`, `GPU_SOLVER`, `HALO_OVERLAP`, `BENCHMARK_MODE`, `USE_*`, ...) and the compiler versions.

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
| Velocity histograms (VTK), macrocell spectra (`species_<s>.json` next to the `.bin` data) | the same two entries |

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

### Boundary rows of the implicit field solve

The Maxwell solve for E(n+θ) (GMRES, or with the GPU field solver also FGMRES with block Jacobi, or Chebyshev) needs a linear operator: the image M·v that the solver evaluates on its own basis vectors must hold no constant term. Each boundary condition is therefore split into its linear part, which goes in the image row, and its constant part, which goes in the right-hand-side row.

| face | image row | right-hand-side row |
|---|---|---|
| open inflow face (`bcEMface* = 2`, `bcPface* = 2`, X-right excluded), with `ApplyInflowBcsEImage = 1` (default) | v | E0 = −u0×B0 |
| perfect conductor (`bcEMface* = 0`), tangential components | v_t | E0_t |
| perfect conductor, normal component n (Ampère's law without curl B, χ_n·E = Eⁿ_n − 4πθΔt Ĵ_n, χ_n the row n of the implicit susceptibility that `MUdot` applies in the interior) | v_n + (χ_na v_a + χ_nb v_b)/χ_nn | (Eⁿ_n − 4πθΔt Ĵ_n)/χ_nn |
| every other face, among them the open X-right face and open faces with `bcPface* ≠ 2` or `ApplyInflowBcsEImage = 0` | the operator's | the physical right-hand side |

Earlier versions put the constants in the image (v − E0 on open faces; the Eⁿ and Ĵ terms on conductor faces). That is an affine map, so GMRES built a wrong Krylov space whenever the constants were nonzero. A 3D magnetosphere with E0 ≠ 0 took about 60 GMRES iterations per cycle instead of 11. The CPU GMRES, which stops on its own residual estimate, also accepted wrong fields: in a 2D case with E0 ≠ 0 the field after one solve was 2 % off at `GMREStol` = 1e-6. Two further defects are fixed. The conductor rows used the transpose of the interior susceptibility, which flipped the sign of the Hall coupling at the wall. The CPU GMRES stopped on the wrong Givens entry and dropped the last Krylov direction of each restart. With these fixes GMRES converges steadily down to 1e-15, and the CPU and GPU solvers take the same number of iterations and return the same field. Rows that kept the operator's image but had their right-hand side zeroed (the open X-right face with `bcPface = 2`, and every open face with `bcPface = 2` when `ApplyInflowBcsEImage = 0`) now keep their physical right-hand side.

### Boundary injection (reemission faces)

On a non-periodic face with `bcPface* = 2`, every particle that ends a push in the last 3 cells before the face is deleted. The GPU then refills those cells with a drifting Maxwellian (`rhoINJECT`, the species' drift and thermal speeds), every cycle and with the same number of particles per cell as the initial loading (`npcel*`). Each injected particle draws its numbers from its own Philox stream. The key is set by the cycle and the species, and the stream number is set by the particle's global cell and sub-cell. So ranks inject different plasma, the injected plasma does not depend on `XLEN`/`YLEN`/`ZLEN`, and a restart repeats the injection of every cycle.

### Open outflow boundary (X-right)

With `bcEMfaceXright = 2` and `bcPfaceXright = 3` (and `PERIODICX = 0`), the last L = max(3, `n_layers_sal`) cells before the X-right face form the outflow layer, beyond the reference plane x_s, L cells inside the face. The field solve sees the layer as neutral, with the current and pressure of x_s. This is always on, in the GPU and the CPU field solver (`GPU_SOLVER`):

- `OutflowLayerFields` sets what happens to the fields on the layer after each field solve:
  - `solved` (default): the solver's E and B are kept, so on the layer they satisfy Faraday's and Ampère's laws as inside, with the layer's neutral charge and copied current; only the ghost node and the ghost cell beyond the face are copied from the face, the zero-gradient continuation the solver assumes.
  - `copy`: E and B on the layer are copied from x_s, a flat continuation of x_s. This holds the field at the face at its value at x_s, which removes the outward push of a tangential field that weakens toward the face; where slow plasma drains through the face along field lines tied elsewhere, the flux then piles up behind x_s.
- Before each solve, the species densities on the layer are set to the x_s values, rescaled to equal positive and negative charge, (ρ₊ + |ρ₋|)/2 each; where a node of x_s holds charge of one sign only, the layer nodes in its line are emptied. J_s and the pressure tensors on the layer are copied from x_s.
- The predicted charge ρ̂ is zero on every cell beyond x_s, including the cell between x_s and the first layer node, so the solver's neutral exterior starts at x_s (where the field copy starts with `OutflowLayerFields = copy`).
- Inside the field solver, the face is closed with a zero-gradient ghost node (the Krylov vector is unchanged), and the E smoothing continues all components with zero gradient.
- With `divBCorrection`, the ∇·B cleaning corrects B on every node, the outflow layer included (see below).

The written `rho`, `J` and pressure on the layer are these neutralized or copied values; the written E and B are the copied or the solved fields.

### Absorbing layer and ∇·B cleaning on open faces

On an open face (`bcEMface* = 2`) with `yes_sal = 1`, every field solve blends the fields toward the upstream values on the cell layers 0…`n_layers_sal`: E toward −u0×B0 and B toward B0, with a weight s that is 0 at the face and rises linearly to 1 inside. X-right is excluded: it always continues the interior (see above). Below, B_n is the component of B normal to the face and B_t the tangential ones.

- `sal_tangential_B = 1` (default): only B_t is blended; B_n is left to the field solve.
  - Blending B_n creates ∇·B of about (B_n − B0_n)/(`n_layers_sal` Δx) in the layer every cycle, however smooth the field is. Blending only B_t creates (1 − s) ∂_n B_n, which the cleaning drives to zero.
  - B_n across the layer then follows the interior field (zero normal gradient) instead of being clamped to B0_n.
- `sal_tangential_B = 0`: all three components are blended (the behaviour before this option).

`divBCorrection = yes` cleans the node B every `divBCorrectionCycle` cycles, at the end of `calculateB`:
1. It solves ∇²ψ = ∇·B on the cells, with ψ = 0 beyond the physical faces (GMRES on the CPU, preconditioned FGMRES with the GPU field solver, see below).
2. It subtracts ∇ψ from B on every node, with the same ψ ghosts. With zero ghosts, `gradC2N` is minus the transpose of `divN2C`, so the step is the orthogonal projection of the node B onto divergence-free fields, apart from the difference between the 7-point Laplacian of the solve and divN2C∘gradC2N. That difference only ever reduces ∇·B, mode by mode.
3. It rebuilds the cell-centred B as interpN2C of the node B.

Use the cleaning with the tangential SAL. With `sal_tangential_B = 0`, the cleaning carries the SAL's B_n = B0_n at the faces into the whole box, and the code prints a warning. Without the cleaning, the tangential SAL lets ∇·B pile up in the layer.

### Poisson solves of the ∇·E and ∇·B corrections

`PoissonCorrection` (∇·E) and `divBCorrection` (∇·B) solve their Poisson problems with FGMRES(20), at most 200 restarts, to `GMREStol`. With the GPU field solver, FGMRES is preconditioned on each rank by a Chebyshev polynomial of degree `PoissonChebMaxIter` + 1 in the rank's local Laplacian (block Jacobi, zero values at rank cuts), using the analytic eigenvalue bounds [λ_min, λ_max] of that local operator.

On [λ_min, λ_max], the polynomial leaves a residual of at most 1/T_k(σ) ≈ 1/cosh(2k √(λ_min/λ_max)). With the exact minimum of a large block, this is close to 1: λ_max/λ_min ≈ 3·10⁵ for a 960 × 768 2D block. The preconditioner then does almost nothing, and the solves become slow or stall.

`PoissonChebEigRatio` (default 1000) therefore raises the lower bound to max(λ_min, λ_max / `PoissonChebEigRatio`). The polynomial then reduces the upper spectrum, and FGMRES resolves the few smooth modes below the bound, which the polynomial still inverts with the right sign. `PoissonChebEigRatio = 0` restores the exact bounds.

The CPU field solver uses unpreconditioned GMRES for these solves.

Both corrections subtract ∇ψ with the ghost rule of their solve (ψ = 0 beyond the physical faces), so the step is the discrete projection the solve inverted.

### Planet surface (`Dipole`, `Dipole2D`)

Particles that enter the planet (`Planet_radius`) are processed on the GPU every cycle, on each rank:

- All ions are absorbed.
- Electrons of all species are ranked together by the kinetic energy of one physical electron (relativistic, independent of the macroparticle weight) and absorbed from the most energetic down, until the absorbed charge best matches the charge to neutralise: the ions absorbed in this cycle plus the imbalance carried from earlier cycles. What is left, at most half an electron macroparticle unless too few electrons hit, is carried to the next cycle. The carried charge is not written to restart files; a restart starts it from zero.
- The other electrons are reflected, specularly or diffusely (`planetReflectionType = 0` or `1`), and placed on the surface.

Every `PlanetDiagnosticsCycle` cycles (default `DiagnosticsOutputCycle`; `<= 0` disables), rank 0 appends the surface balance to `PlanetBC.txt` in `SaveDirName`: per species, the macroparticles and charge that hit the planet and that were absorbed; the carried charge, summed over ranks and its extremes; the mean energy separating absorbed from reflected electrons; and the rank-cycles with hits and with too few electrons to balance the ions. A row labelled N covers the hits since the previous row, up to cycle N-1; a last row at the end of the run, labelled with the resume cycle like the final restart checkpoint, covers the rest.

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
| `PlanetDiagnosticsCycle` | `DiagnosticsOutputCycle` | Append the planet surface balance to `PlanetBC.txt` every N cycles and at the end of the run (`Dipole`, `Dipole2D`). `<= 0` disables. |
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
