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

If you's like to use HIP, notice the GPU architecture in [CMakeLists.txt](./CMakeLists.txt), change it according to your hardware to get bet performance:

``` cmake 
set_property(TARGET iPIC3Dlib PROPERTY HIP_ARCHITECTURES gfx90a) 
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

It's fine to use more than 1 MPI process per GPU. The following example uses 4 nodes, each equipped with 4 GPU:

``` shell
# 1 MPI process per GPU
srun --nodes=4 --ntasks=16 --ntasks-per-node=4 ./iPIC3D ../share/benchmark/GEM3Dsmall_4x2x2_100/testGEM3Dsmall.inp 

# 2 MPI processes per GPU
srun --nodes=4 --ntasks=32 --ntasks-per-node=8 ./iPIC3D ../share/benchmark/GEM3Dsmall_4x4x2_100/testGEM3Dsmall.inp  
```


### Result

This iPIC3D-GPU will create folder (usually named `data`) for the output results if it doesn't exist. However, **it will delete everything in the folder if it already exits**.


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

iPIC3D-GPU supports restarting a simulation from checkpoint files. Restart files store the full electromagnetic field state (E, B, rho) and all particle data so that a simulation can be resumed from the exact point where it stopped.

### Launching a restart

Add the `restart` keyword to the command line alongside the input file (order does not matter):

```shell
mpirun -np 8 ./iPIC3D restart ../share/inputfiles/myInput.inp
# or equivalently
mpirun -np 8 ./iPIC3D ../share/inputfiles/myInput.inp restart
```

Without `restart`, the simulation always starts fresh from the initial conditions.

### Restart files

Restart checkpoints are written as ADIOS2 BP5 files (one per MPI rank) into the `RestartDirName` directory:

```
data/restart_0.bp
data/restart_1.bp
...
data/restart_N.bp
```

Each file contains multiple ADIOS2 *steps*, one per checkpoint. On restart, the code reads the **last step** from `restart_0.bp` to determine the cycle number, then loads fields and particles from the corresponding per-rank file.

**Important:** you must restart with the **same number of MPI processes** as the original run, since each rank reads its own file.

### Input file parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `RestartDirName` | `data` | Directory containing the `restart_*.bp` files |
| `RestartOutputCycle` | `5000` | Write a restart checkpoint every N cycles. Set to `0` to disable periodic checkpoints. |
| `CallFinalize` | `1` | If `1`, write a final restart checkpoint when the simulation ends. Requires `RestartOutputCycle > 0` — if `RestartOutputCycle` is `0`, no final restart is written even with `CallFinalize = 1`. |
| `ncycles` | — | Number of **new** cycles to run from the restart point (not an absolute cycle number). |

### How the cycle counter works

The cycle variable `i` in the main loop is a **global/absolute counter** that continues from where the previous run stopped:

```
first_cycle = last_cycle_in_restart_file + 1
LastCycle    = first_cycle + ncycles
loop:  i = first_cycle  ...  LastCycle - 1
```

**Example:** original run completes 1000 cycles (0–999) and writes a restart at cycle 999. A restart run with `ncycles = 500` will execute cycles 1000–1499. Output file names, restart checkpoint labels, and all diagnostics use this absolute counter, so there is no ambiguity across runs.

For a fresh start (no `restart` keyword), `last_cycle = -1`, so `first_cycle = 0`.

### What happens during restart initialisation

1. **Input file is read** — all simulation parameters (grid, species, BCs, etc.) are taken from the input file, same as a fresh start.
2. **Fields** — the case-specific field initialiser (`initGEM`, `initDipole`, etc.) runs first, then `read_field_restart()` **overwrites** B, E, and rho with the data from the restart file.
3. **Particles** — instead of generating particles from a distribution function, `restartLoad()` reads positions, velocities, charges, and IDs from the per-rank restart file.
4. **Output directory** — the output folder is **not** cleared on restart. ADIOS2 output files are opened in `Append` mode so new data is added to existing files.

### Requirements

- Restart reading requires **ADIOS2** (`USE_ADIOS2=ON` at compile time). Without ADIOS2, attempting a restart will produce a fatal error.
- The MPI topology (`XLEN × YLEN × ZLEN`) must match between the original and restarted runs.

## Simulation Cases

The simulation case is selected via the `Case` parameter in the input file (e.g. `Case = GEM`). Each case configures specific initial conditions for the electromagnetic fields and particle distributions.

### Supported cases

| `Case` | Description | Example input |
|--------|-------------|---------------|
| `GEM` | **GEM magnetic reconnection challenge.** Single Harris current sheet with $B_x = B_{0x} \tanh\!\bigl((y - L_y/2)/\delta\bigr)$ and a localized Gaussian flux perturbation. Density has a $1/\cosh^2$ profile plus a uniform background. Ions carry the initial drift current. | `share/inputfiles/magneticReconnection/testGEM3D*.inp` |
| `GEMnoPert` | Same Harris equilibrium as GEM but **without** the magnetic perturbation. Useful for stability studies or when perturbations are applied externally. | — |
| `GEMDoubleHarris` | **Double Harris sheet.** Two oppositely-directed current sheets centred at $L_y/4$ and $3L_y/4$, each with its own drift population. Creates two reconnection sites in a periodic domain. | — |
| `ForceFree` | **Force-free current sheet.** Harris $B_x$ profile with $B_z = B_0/\cosh\!\bigl((y - L_y/2)/\delta\bigr)$ so that $\mathbf{J}\times\mathbf{B}=0$. Both electrons and ions share the current. Used for studying tearing instabilities without pressure gradients. | `share/inputfiles/magneticReconnection/testForceFree*.inp` |
| `Dipole` | **3-D planetary magnetosphere.** Magnetic dipole field ($B \propto 1/r^3$) centred in the domain with solar-wind inflow electric field. A spherical planet region is voided of particles. Boundary conditions apply `ConstantChargePlanet` on restarts. | `share/inputfiles/magnetosphere/testMagnetosphere3Dsmall.inp` |
| `Dipole2D` | **2-D dipole** (in the $xz$-plane). Same physics as `Dipole` but configured for a 2-D simulation with `ConstantChargePlanet2DPlaneXZ` boundary treatment. | — |
| `NullPoints` | **Magnetic null-point topology.** Periodic field with components like $B_x \propto -\sin x\,\cos y\,\cos z$ creating a network of null points. Electron current is initialised from $\nabla\times\mathbf{B}$; ions are at rest. | — |
| `TaylorGreen` | **Taylor-Green vortex.** 3-D periodic velocity field ($u_e \propto \sin x\,\cos y$, etc.) combined with a periodic magnetic field. Used for testing decaying MHD turbulence and energy transfer. | `share/inputfiles/turbulence/testTurbulence3D.inp` |
| `RandomCase` | **Random-perturbation reconnection.** Harris current sheet (like GEM) overlaid with a multi-mode random magnetic perturbation ($k_x$, $k_y$, $k_z$ harmonics with random phases and $1/k$ amplitude scaling). | — |
| `BATSRUS` | **Coupling with BATS-R-US MHD code.** Reads fluid fields (density, velocity, pressure, B) from an external MHD solution and initialises Maxwellian particle distributions cell-by-cell to match the MHD moments. Requires `#ifdef BATSRUS` at compile time. | — |

Any unrecognised `Case` string falls through to a **default** initialisation: uniform density, constant background $\mathbf{B}$, zero electric field. A warning is printed to standard output.

### Runtime boundary conditions

Some cases apply additional boundary-condition fixes during the time loop:

| `Case` | Runtime BC |
|--------|------------|
| `GEM`, `GEMnoPert`, `GEMDoubleHarris` | `fixBnGEM` / `fixBcGEM` — enforce Harris-sheet–consistent normal-B and density at $y$-boundaries |
| `ForceFree` | `fixBforcefree` — enforce force-free B profile at $y$-boundaries |
| `Dipole` | `ConstantChargePlanet` — maintain fixed charge inside the planet sphere (every cycle) |
| `Dipole2D` | `ConstantChargePlanet2DPlaneXZ` — 2-D variant of the above (every cycle) |

### Setting the case

In the input file, add or modify:

```
Case = GEM
```

The `Case` string is **case-sensitive** and must match one of the names in the table above exactly.

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



