# Building

[Documentation index](../README.md#documentation)

## Requirements

| Component | Requirement |
|---|---|
| GPU | NVIDIA with CUDA compute capability 7.5 or higher, or AMD with HIP on a wave64 architecture (`gfx90a`, `gfx942`) |
| GPU toolkit | CUDA Toolkit, or ROCm (HIP) for AMD |
| CMake | 3.21 or newer |
| Compiler | C++17. A recent one, such as GCC 12, that your CUDA version supports is recommended |
| MPI | CUDA-aware (ROCm-aware with HIP): the field solver passes device buffers to MPI. MPICH and Open MPI are tested |
| HDF5 | C and high-level libraries. The default `USE_PHDF5=ON` needs an MPI-enabled HDF5; with a serial one, set `USE_PHDF5=OFF`. With both installed, add `-DHDF5_PREFER_PARALLEL=ON`. Not needed with `USE_HDF5=OFF` |
| ADIOS2 | Built with MPI (the build links `adios2::cxx_mpi`). Needed unless `USE_ADIOS2=OFF`. Found, in this order, through `-DADIOS2_DIR=<folder of adios2-config.cmake>`, `-DADIOS2_ROOT=<prefix>`, an install in `thirdparty/adios2`, then `CMAKE_PREFIX_PATH` (`-D` option or environment variable) |
| ParaView | 5.7 or newer with Catalyst and Python, only with `USE_CATALYST=ON` |

## Configure and compile

The basic commands are in the [README](../README.md#build-and-install). `make install` puts `iPIC3D` in `<prefix>/bin`, the shared library in `<prefix>/lib` (`lib64` on some systems), and the decks listed in `share/inputfiles/CMakeLists.txt`, without their subfolders, in `<prefix>/share/ipic3d/inputfiles`.

For AMD GPUs, configure with ROCm's clang as in the [README](../README.md#build-and-install): CMake passes the HIP flags to the plain C++ files too.

More examples:

```shell
cmake -DCUDA_ARCH=90 -DUSE_ADIOS2=OFF ..            # HDF5 and VTK output only
cmake -DBENCHMARK_MODE=2 -DTIME_TASKS=ON ..         # core solver timing, no output
```

## CMake options

| Option | Default | Description |
|---|---|---|
| `CUDA_ARCH` | `75` | CUDA compute capability, at least `75` (`80` for A100, `90` for H100/GH200) |
| `HIP_ON` | `OFF` | Compile with HIP instead of CUDA |
| `HIP_ARCH` | `gfx90a` | HIP architecture: `gfx90a` (MI250X), `gfx942` (MI300A/X) |
| `USE_HDF5` | `ON` | Serial-HDF5 backend for fields, particles and checkpoints |
| `USE_PHDF5` | `ON` | Parallel-HDF5 field backend. Needs an MPI-enabled HDF5. `USE_HDF5=OFF` sets it `OFF` in the cache: set it `ON` again after re-enabling HDF5 |
| `USE_ADIOS2` | `ON` | ADIOS2 backend for particles and checkpoints |
| `USE_CATALYST` | `OFF` | ParaView Catalyst in-situ visualisation |
| `USE_BATSRUS` | `OFF` | BATS-R-US coupling, see the [`BATSRUS` case](cases.md#supported-cases) |
| `USE_OPENMP` | `ON` | OpenMP in the host loops. **Delete the CMake cache when changing it** |
| `BUILD_UTESTS` | `OFF` | Build the [unit tests](testing.md) |
| `TIME_TASKS` | `OFF` | Print per-cycle and cumulative task timings |
| `BENCHMARK_MODE` | `0` | See [below](#benchmark-modes) |
| `GPU_CYCLE_DIAGNOSTICS` | `OFF` | Instrument one GPU timestep and the Maxwell right-hand side that consumes it |
| `GPU_DIAGNOSTIC_CYCLE` | `1` | Cycle instrumented by `GPU_CYCLE_DIAGNOSTICS`; `-1` for every cycle |
| `BUILD_SHARED_LIBS` | `ON` | Shared libraries; `OFF` for static |
| `SITE` | `default` | Include `cmake/sites/<SITE>.cmake` at the start of the configuration |
| `CMAKE_BUILD_TYPE` | `Release` | See [below](#build-types) |

## Build types

| `CMAKE_BUILD_TYPE` | Result |
|---|---|
| `Release` | Optimised, executable `iPIC3D` |
| `Debug` | Debug checks (`DEBUG_MODE`), executable `iPIC3D_d` |
| Any value outside the CMake build types, such as `Default` | No optimisation flags |

## OpenMP

The field solver, the mover and the moments run on the GPU. OpenMP parallelises the host loops: particle initialisation and other host particle operations, the field initialisation, and the packing of field output. The threads per rank are set at run time, see [Running](running.md#ranks-threads-and-gpus).

## Benchmark modes

`BENCHMARK_MODE` removes output work at compile time; the communication the simulation needs stays.

| Level | Effect |
|---|---|
| `0` | Normal build |
| `1` | No disk output and no output copies to the host. Data analysis and heat flux still compute |
| `2` | Level 1, and no data analysis, macrocell spectra or heat-flux calculation |
