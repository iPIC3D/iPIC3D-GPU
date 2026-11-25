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

iPIC3D-GPU is a three-dimensional implicit particle-in-cell code for kinetic plasma simulations, such as magnetic reconnection, turbulence and planetary magnetospheres. The field solver, particle mover and moment deposit run on NVIDIA (CUDA) or AMD (HIP) GPUs, on one workstation or across many nodes with MPI.

## Requirements

- A GPU with CUDA compute capability 7.5 or higher and the CUDA Toolkit, or an AMD GPU with a wave64 architecture (MI250X, MI300) and ROCm
- CMake 3.21 or newer and a C++17 compiler
- A CUDA-aware (with HIP, ROCm-aware) MPI library
- For the default build, an MPI-enabled HDF5 and ADIOS2 (`-DUSE_PHDF5=OFF` accepts a serial HDF5; `-DUSE_HDF5=OFF` and `-DUSE_ADIOS2=OFF` remove them)

See [Building](Documentation/building.md) for details and all options.

## Get the code

```shell
git clone https://github.com/iPIC3D/iPIC3D-GPU.git
cd iPIC3D-GPU
```

## Build and install

```shell
mkdir build && cd build
cmake -DCUDA_ARCH=80 ..    # your GPU's compute capability, here an A100
make -j"$(nproc)"
make install               # optional, honours -DCMAKE_INSTALL_PREFIX
```

For AMD GPUs, configure with ROCm's compiler:

```shell
CC=amdclang CXX=amdclang++ cmake -DHIP_ON=ON -DHIP_ARCH=gfx90a ..
```

## Run

Pass an input file to the `iPIC3D` executable. The number of MPI ranks must equal `XLEN × YLEN × ZLEN` of the input file, here 3 × 1 × 2:

```shell
export OMP_NUM_THREADS=2
mpirun -np 6 ./iPIC3D ../share/inputfiles/magnetosphere/testMagnetosphere2Dsmall.inp
```

On a cluster, launch with `srun`, as described in [Running](Documentation/running.md).

Results go to the folder `SaveDirName` of the input file (`data` here), relative to the current directory. **A fresh run deletes the contents of its output folders**, see [Output folder](Documentation/running.md#output-folder).

## Documentation

| Topic | Content |
|---|---|
| [Building](Documentation/building.md) | Requirements, CMake options, build types, OpenMP, benchmark modes |
| [Running](Documentation/running.md) | Command line, ranks and GPUs, Slurm, output folder, build and run record |
| [Input file](Documentation/input-file.md) | Syntax, value types, common keys, species, index of all keys |
| [Simulation cases](Documentation/cases.md) | Initial conditions, their keys, GEM and ForceFree wall forcing |
| [Code architecture](Documentation/architecture.md) | Source tree, data placement, time loop, GPU execution, memory |
| [Field solver](Documentation/field-solver.md) | Implicit scheme, solvers, smoothing, halo overlap, run guards, ∇·E and ∇·B corrections |
| [Boundary conditions](Documentation/boundary-conditions.md) | Boundary codes, inflow faces, conductors, reflection, injection, open outflow, absorbing layer, planet surface |
| [Particles](Documentation/particles.md) | Loading, mover, sorting, exchange, particle IDs, splitting and merging, exosphere, test particles |
| [Output](Documentation/output.md) | Output keys, tags, backends, file layouts, particle selection, deferred writing, diagnostics |
| [Restart](Documentation/restart.md) | Launching, checkpoint slots, cycle numbering, remapping |
| [Data analysis](Documentation/data-analysis.md) | Velocity histograms, Gaussian mixtures, spectra, record files |
| [Units](Documentation/units.md) | Normalisation, internal storage, written quantities, derived formulas |
| [Testing](Documentation/testing.md) | Unit tests, on a workstation and on Slurm |
| [Profiling](Documentation/profiling.md) | Cycle timing in the run log, profiler regions, Nsight, rocprofv3, CrayPat, stage timers |
| [Tools](Documentation/tools.md) | Input check, benchmarks, scaling, post-processing, code formatting |

## Citation

- Markidis, S., Hu, A., Peng, I., Pennati, L., Lumsden, I., Yokelson, D., Brink, S., Pearce, O., Scogland, T.R., de Supinski, B.R. and Delzanno, G.L., 2025. Exascale Implicit Kinetic Plasma Simulations on El Capitan for Solving the Micro-Macro Coupling in Magnetospheric Physics. arXiv preprint arXiv:2507.20719.
- Markidis, Stefano, Giovanni Lapenta, and Rizwan-uddin. "Multi-scale simulations of plasma with iPIC3D." Mathematics and Computers in Simulation 80.7 (2010): 1509-1519.

## Contact

Feel free to contact Professor Stefano Markidis at KTH for using iPIC3D.
