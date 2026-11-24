# Running

[Documentation index](../README.md#documentation)

## Command line

```text
iPIC3D [input file] [restart | restart=latest | restart=A | restart=B | restart A | restart B]
```

The arguments may come in any order. Without an input file the code reads `inputfile`. A second input file, a repeated `restart` or an unknown slot stops the run with a usage message. See [Restart](restart.md). The README has an [example](../README.md#run), and an input file can be checked first with the [input check](tools.md#input-check).

The *Check Simulation Constraints* block of the startup report is advisory: the implicit scheme (`th` ≥ 0.5) does not need c·dt/dx < 1, and none of its lines stops the run, even those that say so.

## Ranks, threads and GPUs

- The number of MPI ranks must equal `XLEN × YLEN × ZLEN`. Otherwise the run stops at startup.
- Each rank owns one box of the grid. Along x, every rank gets ⌈`nxc`/`XLEN`⌉ cells and the last rank the rest, with a warning when `XLEN` does not divide `nxc`; likewise along y and z. The last rank must keep at least one cell, or the run stops at startup: `nxc = 9` with `XLEN = 4` would leave it none.
- `OMP_NUM_THREADS` sets the [OpenMP](building.md#openmp) threads per rank. Keep ranks × threads within the cores of a node.
- On each node, ranks are assigned to the visible GPUs in order. With more ranks than GPUs, **the ranks per node must be divisible by the GPUs per node**, and consecutive ranks share a GPU.
- Several ranks sharing one GPU time-slice it, so a single rank per GPU is usually faster.

With Open MPI on one node without UCX, select the CUDA-aware shared-memory transport:

```shell
mpirun -np 6 --mca pml ob1 --mca btl self,vader,smcuda ./iPIC3D input.inp
```

## Slurm

Example with 4 nodes of 4 GPUs each:

```shell
# 1 rank per GPU
srun --nodes=4 --ntasks=16 --ntasks-per-node=4 ./iPIC3D ../share/benchmark/GEM3Dsmall_4x2x2_100/testGEM3Dsmall.inp

# 2 ranks per GPU
srun --nodes=4 --ntasks=32 --ntasks-per-node=8 ./iPIC3D ../share/benchmark/GEM3Dsmall_4x4x2_100/testGEM3Dsmall.inp
```

Add the GPU request your site needs, such as `--gpus-per-node=4`. With Cray MPICH, also set `MPICH_GPU_SUPPORT_ENABLED=1`.

## Output folder

`SaveDirName` and `RestartDirName` are relative to the current directory and created if missing. A **fresh run deletes the contents** of `SaveDirName`, of `RestartDirName` if it differs, and of the folder of each [data analysis](data-analysis.md#restarts) that is on. A restart keeps them and appends, see [what a restart does to the output](output.md#restarts-and-existing-files). The outputs are described in [Output](output.md).

## Build and run record

Every build records its **version**, `git describe` plus `-dirty` when the tree has uncommitted or untracked non-ignored files, and its **build signature**: the full commit hash, a hash of the uncommitted changes, the build type, the GPU architecture, the options `BENCHMARK_MODE`, `USE_*`, `TIME_TASKS`, `GPU_CYCLE_DIAGNOSTICS` and `PARTICLE_ARRAY_*`, and the compiler versions. A build from a modified tree warns at configure time, at every build and in the run log. Builds outside git record `unknown`.

| Where | What |
|---|---|
| Run log (rank 0, at startup) | Version, commit, tree state, build type, options, compilers, start time, host, MPI ranks and library, OpenMP threads, GPUs, input file, fresh run or restart |
| `SaveDirName/SimulationData.txt` | The same block, above the simulation parameters |
| `SaveDirName/<input file>` | A copy of the input file; a restart adds `<name>.restart<cycle><ext>` |
| `RestartDirName/latest_restart.json`, `restart_A/manifest.json`, `restart_B/manifest.json` | `ipic3d_version` and `build_signature` |
| VTK files | Dataset field data `ipic3d_version`, `ipic3d_build` (ParaView: Information panel, Field Data) |
| HDF5 files | Root attributes `ipic3d_version`, `ipic3d_build` (`h5dump -A`). A file a restart appends to keeps those of the run that created it |
| ADIOS2 files | Attributes `ipic3d_version`, `ipic3d_build` (`bpls -a`) |
| Data-analysis `.json` files | The same two entries, of the run that last opened the file |

A restart from a checkpoint with another signature, or none, is described in [Restart](restart.md#requirements).
