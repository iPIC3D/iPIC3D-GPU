# Kernel profiles

Three executables, each timing one GPU kernel of the particle step on one GPU,
and the scripts that turn them into rooflines and speed-of-light tables:

| Executable | Kernel | Particle order | Before each launch | Check |
|---|---|---|---|---|
| `profileMover` | `moverSubcyclesKernel`, the relativistic mover with subcycles | sorted by cell | positions and velocities put back, departure flags cleared | no particle deleted |
| `profileCellAwareMoment` | `cellAwareMomentKernel`, the deposit of the particles sorted by cell, one warp per cell | sorted by cell | moments zeroed | deposited charge = particle charge |
| `profileMomentKernelNew` | `momentKernelNew`, the deposit of the unsorted tail, one thread per particle | random | moments zeroed | deposited charge = particle charge |
| `profileCeilings` | streaming and FP64 FMA kernels: the measured ceilings of the rooflines | | | |

All three kernels run every cycle in the 3D magnetosphere runs (`Case = Dipole`,
`SortingCycle = 1`). There `momentKernelNew` deposits only the tail of the
particles received from the neighbour ranks, about 0.1 % of a rank's particles
(an estimate): `--tail 0.001` profiles that size.

| File | Use |
|---|---|
| `profileKernels.inp` | Input file of the profiles |
| `build.sh` | Build of the profiles |
| `profile.sh` | Event timings, Nsight Compute counters and measured ceilings |
| `plot_roofline.py` | Tables, rooflines and speed of light of a `profile.sh` run, offline |
| `profileGH200/` | `build.sh` and `profile.sh` with the settings of NVIDIA GH200 |
| `profileRTX5080/` | `build.sh` and `profile.sh` with the settings of the workstation's RTX 5080 |

Each GPU system has a folder with `build.sh` and `profile.sh`, which set the
system's settings and call the scripts above; its build and results go in
that folder (`build/`, `results/`, ignored by git).

## What a profile runs

A profile builds one species of a periodic box of uniform plasma with the
solver's own code, in the solver's order:

- the input file `profileKernels.inp` (`Collective`), the topology and the grid;
- the fields (`EMfields3D::init`) and the Maxwellian particles
  (`ParticleSoAHost::maxwellian`), moved to the device arrays
  (`particleArrayCUDA`);
- the sort by cell (`CellSorter::sort`), or a random permutation;
- for the mover, the device fields packed as in the solver
  (`gpuSolverAllocate`, `gpuSolverSyncH2D`, `gpuPackFieldForPclsToCenter`), and
  the mover parameters with the boundary flags of the species;
- the launch geometry of the solver (`kMoverBlockSize`, `DEFAULT_BLOCK_SIZE`,
  `SMALL_BLOCK_SIZE`).

MPI is initialised on the one process, because the topology, the grid and the
fields are built on it. Nothing is communicated, and no `mpirun` is needed.

The input file has the species, time step, cell size and solar-wind B of the
production runs: electrons (species 0, q/m = −100) and ions (species 1). A
periodic box is a rank with neighbours on every face: no open, exit or wall
flags. Differences from a production rank:

- no planet test in the mover (a few operations per particle);
- uniform B, E = 0 (the arithmetic and the memory accesses do not depend on
  the values);
- `profileMover` replaces B by a uniform B along z that makes every particle
  take 4 subcycles of the push (`--subcycles`; the solar-wind B gives 1);
- the particles start on the regular sub-grid of each cell, so the first push
  moves fewer of them out of the box.

## Build

After loading the compilers, CUDA, MPI and CMake of the machine:

```shell
bash share/profileKernels/profileGH200/build.sh   # into profileGH200/build
```

It runs, with `CUDA_ARCH=90` (`profileRTX5080`: 120) and without HDF5 and
ADIOS2 (the profiles write no output):

```shell
cmake <repo>/iPIC3D-GPU -DBUILD_SHARED_LIBS=OFF -DBUILD_PROFILE_KERNELS=ON \
      -DCUDA_ARCH=90 <other options>
make profileMover profileCellAwareMoment profileMomentKernelNew profileCeilings
```

`BUILD`, `CUDA_ARCH`, `JOBS` and `CMAKE_OPTIONS` can be set in the
environment. The executables go to the build folder. They launch kernels of
the library themselves, which works only with the library linked into them,
so it must be static. With the shared library the profiles would not link on
CUDA 13: nvcc hides the launch functions of the kernels from other binaries.
On CUDA 12 they would link, but a launch would reach the library's own copy of
the static CUDA runtime and do nothing. The kernels are the objects of the
`iPIC3Dlib` target, compiled with the solver's flags.

HDF5 and ADIOS2 stay shared libraries (`HDF5_USE_STATIC_LIBRARIES` is `OFF` by
default). With `-DPROFILING=ON` the timed launches are inside the NVTX range
`Mover` or `MomentDeposit` and inside the Nsight capture window (see
[Profiler](../../Documentation/profiling.md#profiler)). On HIP add
`-DHIP_ON=ON -DHIP_ARCH=gfx90a`.

## Run one profile

```shell
<build>/profileMover share/profileKernels/profileKernels.inp --ppc 128
```

| Option | Default | Meaning |
|---|---|---|
| `--cells NX NY NZ` | `64 64 64` | Cells per axis. The cell size of the input file is kept |
| `--ppc N` | `64` | Particles per cell of every species, `npcelx ≥ npcely ≥ npcelz` as close to a cube as possible |
| `--species S` | `0` | Species profiled |
| `--warmup W` | `2` | Untimed launches before the timed ones |
| `--launches L` | `10` | Timed launches |
| `--subcycles N` | `4` | `profileMover`: subcycles of the push. Sets a uniform B along z; `0` keeps the input file's B, 1 subcycle for both species |
| `--tail F` | `1` | `profileMomentKernelNew`: deposit the last fraction `F` of the particles |

A profile prints the set-up, the device memory in use, the minimum, median,
mean and maximum launch time from CUDA events, and its check. Its last line is
the result as one JSON object: kernel, GPU, build signature, cells, particles
per cell, order, particles, parameter, times, nanoseconds per particle
(median) and check. It exits with 1 when the check fails.

## Profiling run

```shell
bash share/profileKernels/profileGH200/profile.sh
python3 share/profileKernels/plot_roofline.py \
    share/profileKernels/profileGH200/results/<date-time>
```

`profile.sh` runs the measured ceilings (`profileCeilings`: the device-memory
bandwidth of a streaming kernel and the FP64 rate of FMA chains, printed at
the start), then each profile at each particles-per-cell value: once timed
with CUDA events (`LAUNCHES` launches), then `REPETITIONS` times under Nsight
Compute. `REPETITIONS=0` gives the timings only, where the GPU counters are
not available. The settings are at the top of `profile.sh`, set by the
system's folder or in the environment:

| Setting | Default | Meaning |
|---|---|---|
| `BUILD`, `OUT` | the system's `build/`, `results/<date-time>` | Executables; a new folder for the results |
| `CELLS` | `64 64 64` | Cells per axis |
| `PPC` | `64 128 256 512` | Particles per cell |
| `PROFILES` | the three | Profiles run |
| `SPECIES` | `0` | Species (0 electrons, 1 ions) |
| `WARMUP`, `LAUNCHES` | `2`, `10` | Untimed launches before the timed or profiled one; timed launches |
| `REPETITIONS` | `2` | Nsight Compute runs of each point |
| `REPLAY` | `kernel` | Nsight Compute replay of the counter passes: `kernel` keeps a copy of the device memory the kernel reaches (twice the memory), `application` runs the whole profile again for each pass |
| `SUBCYCLES`, `TAIL` | `4`, `1` | `--subcycles` of `profileMover` (`0`: the B of the input file), `--tail` of `profileMomentKernelNew` |
| `LAUNCHER` | | The launch of one task on one GPU (`srun ...`) |

It refuses an existing `OUT` and stops at the first failing run. It writes:

| File | Content |
|---|---|
| `settings.txt` | The settings, the host and the Nsight Compute version |
| `ceilings.log` | `profileCeilings`: streaming bandwidth and FP64 FMA throughput of the GPU |
| `<profile>_ppc<N>/timed.log` | The timed run, with its result line |
| `<profile>_ppc<N>/profiled<R>.log`, `.ncu-rep`, `counters<R>.csv` | Nsight Compute run `R`: the output with its result line (one per pass with application replay), the report, its raw page (base units) |
| `points.tsv` | The points, each listed once all its runs are done |
| `COMPLETE` | Written at the end |

Each Nsight Compute run profiles the first launch after the warmup. Nsight
Compute replays that launch for its counter passes, with the clocks of a run
(`--clock-control none`) and the caches as the profile left them
(`--cache-control none`). It collects NVIDIA's sections `SpeedOfLight`,
`Occupancy` and `LaunchStats`, and, for the rooflines, metrics of the
conventional pipelines only (iPIC3D uses no tensor cores):

| Quantity | Metrics |
|---|---|
| FP64 FLOPs | `smsp__sass_thread_inst_executed_op_dadd_pred_on.sum` + `..._dmul_...` + 2 × `..._dfma_...`: operations of the active threads |
| Duration | `gpu__time_duration.sum` |
| Traffic | `dram__bytes.sum`, `lts__t_bytes.sum` (L2), `l1tex__t_bytes.sum` (L1) |
| Atomics | `lts__t_requests_op_atom.sum`, `lts__t_requests_op_red.sum`, and the L2 atomic unit `lts__d_atomic_input_cycles_active` |
| FP64 pipe | `sm__pipe_fp64_cycles_active` |
| MIO | `sm__mio_inst_issued`, `sm__mio_pq_write_cycles_active`, `sm__mio_pq_read_cycles_active`, `sm__mio2rf_writeback_active`: the four interfaces of MIO, the SM's way to its shared units (shared memory, shuffles, special math, dynamic branches) |
| Peaks | `.peak_sustained` of the FP64 FMAs (the DFMA pipe, not the FP64 tensor cores of GH200, twice as fast) and of the three traffics, times the clock (`*__cycles_elapsed.avg.per_second`) |

The FP64 adds of the deposits' atomics are done in L2, not in the FP64
pipeline, so they are not counted: 80 per particle in `momentKernelNew`. The
cell-aware deposit sums each warp's contributions in registers first: its
FLOPs include those adds (5 per moment and node), and it sends 80 atomics per
32 particles to L2.

`plot_roofline.py RUN` reads the run in place and launches nothing. It checks
that every run passed its check, comes from one build, grid and species, and
ran on the GPU of the ceilings. It prints the report of the run: the device's
rates, measured by `profileCeilings` and peak by Nsight Compute (FP64 of the
DFMA pipe, DRAM, L2, L1, at the SM clock of the profiled launches), and, per
kernel and particles per cell, the time per particle, GFLOP/s, DRAM GB/s,
their percent of the device's peaks and the busiest units. It writes to
`RUN/processed/`:

| File | Content |
|---|---|
| `report.txt` | The report |
| `times.csv` | The timed runs: minimum, median, mean, maximum launch time, nanoseconds per particle |
| `ceilings.csv` | The device's rates: measured, and the medians of the Nsight Compute peaks and SM clock |
| `points.csv` | Every profiled launch: GFLOP/s, GB/s at DRAM, L2 and L1 and their percent of the peaks, roofline coordinates, speed of light, occupancy, Nsight Compute peaks |
| `roofline_<gpu>.png/pdf` | FLOP per DRAM byte against GFLOP/s, with the device's peak and measured ceilings |
| `roofline_hierarchical_<gpu>.png/pdf` | One panel per kernel: FLOP per L1, L2 and DRAM byte, with the Nsight Compute peaks |
| `speed_of_light_<gpu>.png/pdf` | One panel per kernel: the percent of peak of SM, memory, L1, L2, DRAM, FP64 pipe and L2 atomic unit |
| `roofline_paper_<gpu>.png/pdf` | The DRAM roofline for papers, in the style of the article_veil rooflines: no title, large text, the FP64 roof of the Nsight Compute peaks and the kernels in one legend, the particles per cell in a legend of their own above the roof |
| `speed_of_light_paper_ppc<N>_<gpu>.png/pdf` | For papers, on the canvas and in the text sizes of `roofline_paper`: vertical bars, one group each for SM, memory, FP64 pipe and L2 atomic unit, one bar per kernel with its percent of peak, at 256 particles per cell (the largest of the run without 256); the legend in a box above the bars |
| `speed_of_light_paper_breakdown_ppc<N>_<gpu>.png/pdf` | The same, with MIO, L1, L2 and DRAM in place of SM and memory (FP64 pipe, MIO, L1, L2, DRAM, L2 atomic unit), the group names diagonal |

`<gpu>` is the GPU's short name: `rtx5080`, `gh200`, `a100`. A point is the
mean of the repetitions; markers or shades distinguish the particles per
cell. `--partial` uses an unfinished run, `--ppc 64,128` selects
particles per cell, `--no-plot` writes the tables only (no matplotlib needed).
Python 3.9 or newer. The script sets `LC_ALL=C`, the number format the
plotter reads.

### Reading the speed of light

Each column of `points.csv` ending in `_pct` is the share of a unit's peak
throughput used over the launch, as Nsight Compute reports it:

| Column | Unit |
|---|---|
| `sm_pct`, `memory_pct` | The busiest SM unit and the busiest memory unit (the `SpeedOfLight` headlines) |
| `l1_pct`, `l2_pct`, `dram_pct` | L1 (of its active cycles), L2, device memory |
| `fp64_pipe_pct` | FP64 pipeline |
| `mio_pct` | MIO: the busiest of its four interfaces, `mio_inst_issued_pct`, `mio_pq_write_pct`, `mio_pq_read_pct`, `mio_writeback_pct` |
| `l2_atomic_pct` | L2 atomic unit |
| `fp64_peak_pct`, `dram_peak_pct` | GFLOP/s and DRAM GB/s over the device's peaks at the SM clock of the launch (`sm_clock_ghz`) |
| `dram_roofline_pct` | GFLOP/s over the DRAM roofline bound at the point's FLOP per byte (measured ceilings) |

`busiest_units` names the three busiest units of the `SpeedOfLight`
breakdowns, the tensor pipes left out: the limiter of the kernel. A headline
near 80 % or more means the kernel is bound by that unit; both well below
means latency bound, and the occupancy columns (`achieved_occupancy_pct`,
`theoretical_occupancy_pct`, `registers_per_thread`) tell whether there are
enough warps to hide it. The reports (`.ncu-rep`) keep the breakdowns and
NVIDIA's advice for the Nsight Compute GUI.

The FP64 pipeline also runs FP64 instructions that are not FLOPs (comparisons,
conversions, min and max): `fp64_pipe_pct` can be well above `fp64_peak_pct`.
On the RTX 5080 (FP64 at 1/64 of FP32), at 64³ cells and 64 to 256 particles
per cell, the mover (1 subcycle, the solar-wind B) had the FP64 pipe at 89 % (64 % of the FP64 peak in
FLOPs), the cell-aware deposit at 84 % (its warp sums), and `momentKernelNew`
the L2 at 80 % (its atomics). The SM clock of `momentKernelNew` fell from 2.91
to 2.52 GHz as its particles grew, and its timed runs, ten launches in a row,
took 1.58 to 1.65 ns per particle against 1.30 to 1.45 for its profiled
launches; the other two kernels kept 2.9 GHz and the same time in both.

## GH200

- `profileGH200/build.sh` builds with `CUDA_ARCH=90`. The Grace host is
  aarch64; the build is the same.
- Memory: the 96 GB (HBM3) or 144 GB (HBM3e) of a GH200 hold every profile
  at 64³ cells up to 512 particles per cell (table below).
- `profileGH200/profile.sh` uses the defaults of `profile.sh`: 64³ cells, 64
  to 512 particles per cell, kernel replay.
- On Slurm, set `LAUNCHER` to the launch of one task on one GPU, with your
  account, partition and time:
  `LAUNCHER="srun --nodes=1 --ntasks=1 --gpus-per-task=1" bash profile.sh`.
  The script runs everything else locally.
- Nsight Compute needs access to the GPU performance counters. If it reports
  `ERR_NVGPUCTRPERM`, the site has to enable it, often through a Slurm option
  or a partition for profiling; `REPETITIONS=0` still gives the timings.

## RTX 5080

`profileRTX5080` runs on the workstation's GeForce RTX 5080 (16 GB, FP64 at
1/64 of FP32), which also drives the desktop: a full GPU takes the desktop
down. `profileRTX5080/profile.sh` sets:

- 64 to 256 particles per cell at 64³ cells: the GPU holds 8.4 GiB with the
  mover at 256 (desktop included); at 512 neither the mover (14 GiB) nor the
  cell-aware deposit fits;
- application replay: kernel replay would double the memory;
- a guard that stops the run when the GPU memory in use, all processes,
  passes `MAX_GPU_MIB` (10000). It samples `nvidia-smi` every 0.2 s, so it
  stops what stays high, not a short peak.

Run it on an otherwise idle GPU. A run takes about 15 minutes.

## Memory

Device bytes per particle: 56 for the particle arrays, plus 8 for the
departure flags and 48 for the saved input in `profileMover`, or 12 for the
kept cell sort in `profileCellAwareMoment`. The CUDA context and, for the
mover, the field-solver arrays add a few hundred MiB. At 64³ cells:

| ppc | Particles | `profileMover` | `profileCellAwareMoment` | `profileMomentKernelNew` |
|---|---|---|---|---|
| 64 | 16.8 M | 1.8 GiB | 1.1 GiB | 0.9 GiB |
| 128 | 33.6 M | 3.5 GiB | 2.1 GiB | 1.8 GiB |
| 256 | 67.1 M | 7.0 GiB | 4.3 GiB | 3.5 GiB |
| 512 | 134.2 M | 14.0 GiB | 8.5 GiB | 7.0 GiB |

The host holds the particles in page-locked memory (56 bytes per particle,
7.5 GiB at 512 ppc), plus 12 bytes per particle while shuffling. A profile
checks the device and host memory before it makes the particles, and stops
when either is short. With kernel replay, Nsight Compute keeps a copy of the
device memory the kernel can reach, on the device when there is room: twice
the memory of the table.

## AMD

The scripts collect Nsight Compute counters only. On AMD, rocprof-compute
profiles one kernel with its own measured ceilings:

```shell
rocprof-compute profile --name mover --roof-only -k moverSubcyclesKernel -- \
    ./profileMover share/profileKernels/profileKernels.inp --ppc 128 --launches 1
rocprof-compute analyze -p workloads/mover/<gpu model> -b 4   # roofline block
```

`-k` matches a substring of the kernel name. Without `--roof-only` it collects
all its counters, and `-b 2` shows its speed-of-light block.
