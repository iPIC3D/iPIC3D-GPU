# Profiling and timing

[Documentation index](../README.md#documentation)

## Timing in the run log

Every run times itself: host wall time on a monotonic clock, always on.

Each cycle, rank 0 prints

```text
Execution time cycle: 12.3148 ms  [ParticleCountLog=0.00514 AnalysisPrepare=0.000421 HeatFlux=0.00019 AnalysisStart=0.000201 FieldSolveE=5.65661 AnalysisWait=0.000541 MoverLaunch=0.334382 Output=0.098907 ParticleExchange=0.640091 FieldSolveB=0.776127 MomentsAwait=4.78559 OutputCopy=0.016611]
```

The bracket splits the cycle into its stages, the steps of the [time loop](architecture.md#time-loop), in milliseconds of the main thread. Each stage is a [profiler region](#regions) that the cycle timer opens at the start of the step and closes at the start of the next, so the stages add up to the cycle:

| Stage | Step | Work |
|---|---|---|
| `ParticleCountLog` | 1 | Particle counts to `particleNum<rank>.csv` |
| `AnalysisPrepare` | 2 | Sort and field pack for the data analysis |
| `HeatFlux` | 3 | Heat-flux kernels and copies, if an output needs them |
| `AnalysisStart` | 4 | Start of the data-analysis thread |
| `FieldSolveE` | 5 | E solve |
| `AnalysisWait` | 6 | Wait for the data analysis |
| `MoverLaunch` | 7 | Merging, launch of the movers |
| `Output` | 8 | Output and diagnostics, while the GPU mover runs |
| `ParticleExchange` | 9 | Mover wait, planet, injection, MPI exchange, sort, moments |
| `FieldSolveB` | 10 | B solve |
| `MomentsAwait` | 11 | Moment wait, moment halo, derived moments |
| `OutputCopy` | 12 | Copies for the next cycle's output |

A stage left out of the build (output and data analysis in [benchmark modes](building.md#benchmark-modes) 1 and 2) or of the cycle (`OutputCopy` of the last cycle) reads 0. The GPU runs asynchronously: a stage holds the GPU time of the work the main thread waits for in it, not that of the work it launches.

At the end, rank 0 prints the summary:

| Line | Interval |
|---|---|
| `Total execution` | Start of the program to the end of `Finalize` |
| `Initialization` | Start of the program to the end of `Init` |
| `Pre-cycle setup` | Data-analysis setup and initial moments |
| `Cycle loop` | All cycles; `avg` and `stddev` of the cycle lines |
| `Steady state` | All cycles but the first; `avg` per cycle, `stddev`, `min`, `max` of the cycle lines, and the mean (maximum) over the ranks of each stage of the cycle line |
| `Finalize` | Release of the data analysis, final checkpoint, output close, GPU release |
| `Simulation Time` | End of `Init` to the end of `Finalize` |

The **steady state** leaves out the first cycle, which loads the GPU code, opens the MPI connections and runs the first data analysis. It starts after the first cycle and ends after the last one, each time once every rank has drained its GPU and all ranks have met: its interval holds all of its GPU work and is the same on every rank. A run of one cycle has no steady state. The [profiler](#profiler) records this interval apart.

Durations are the maximum over the ranks; the cycle lines and their statistics are those of rank 0.

## Profiler

The code is divided into named [regions](#regions): the stages of the cycle and their parts, down to a Krylov iteration, a halo exchange or the mover of one species. Three independent CMake options each compile a backend in:

| Option | Backend |
|---|---|
| `PROFILING=ON` | Each region is an NVTX range on a CUDA build ([Nsight](#nsight-systems-and-nsight-compute-cuda)), a ROCTX range on a HIP build ([rocprofv3](#rocprofv3-hip)) |
| `CRAYPAT=ON` | [CrayPat](#craypat) records only the steady state |
| `TIME_TASKS=ON` | [Stage timers](#stage-timers): the host time of each region, printed at the end |

They can be combined. All three are `OFF` by default: the regions then compile to nothing and cost nothing. [Benchmark modes](building.md#benchmark-modes) 1 and 2 turn the stage timers on, and mode 2 turns the other two off. The [build signature](running.md#build-and-run-record) records them.

A region measures the host thread that opens it. The mover of each species runs on its own thread (`iPIC3D mover`), the data analysis on `iPIC3D analysis` and the deferred output on `iPIC3D output`; the tracing tools show these names.

### Nsight Systems and Nsight Compute (CUDA)

```shell
cmake -DPROFILING=ON <other options> ..
mpirun -np 4 nsys profile -t cuda,nvtx -o report.%q{OMPI_COMM_WORLD_RANK} ./iPIC3D input.inp
nsys stats --report nvtx_sum report.0.nsys-rep           # host time of each region
nsys stats --report nvtx_gpu_proj_sum report.0.nsys-rep  # GPU time of the work each region launched
```

With Slurm, name the reports with `%q{SLURM_PROCID}`.

- The ranges belong to the NVTX domain `iPIC3D`. An indexed region carries its index (the cycle, the species, the Krylov iteration) as the NVTX payload.
- The build calls `cudaProfilerStart` and `cudaProfilerStop` at the start and the end of the steady state. `nsys profile --capture-range=cudaProfilerApi --capture-range-end=stop` records only the steady state. Nsys's default end, `stop-shutdown`, ends the run there instead.
- `ncu --profile-from-start off` profiles only the kernels of the steady state. `ncu --nvtx --nvtx-include "iPIC3D@MaxwellImage/"` profiles only the kernels launched in a region. Add `-c <count>` to limit the number of kernels.

### rocprofv3 (HIP)

```shell
cmake -DHIP_ON=ON -DPROFILING=ON <other options> ..
srun ... rocprofv3 --marker-trace --kernel-trace --output-format pftrace -- ./iPIC3D input.inp
```

- The build takes `rocprofiler-sdk-roctx` from `ROCM_PATH` (or `/opt/rocm`). Without it, it falls back to the legacy `roctx64` of roctracer, read by `rocprof --roctx-trace`, which does not show the thread names.
- ROCTX ranges have no domain, colour or payload. An indexed region leaves its index out, so that it keeps one name the tools sum over its calls.
- The build sets no recording window on HIP. The `SteadyState` range marks the steady state.

### CrayPat

```shell
module load perftools-base
cmake -DCRAYPAT=ON <other options> ..
srun ... pat_run -w -g mpi,hip ./iPIC3D input.inp
pat_report <experiment folder>
```

`main` switches CrayPat's recording off, then on at the start of the steady state and off at its end. The switch is made on the main thread, so it covers every thread, and every table of `pat_report` covers the steady state. A run of one cycle records nothing. Without `pat_run` the executable runs normally. The executable needs the same perftools install at run time.

### Stage timers

With `TIME_TASKS=ON` the run ends with a table of every region that ran:

```text
Stage region               calls/cycle      min ms     mean ms      max ms      std ms   run max s
Stage FieldSolveE                 1.00       9.939       9.942       9.946       6.722       0.124
Stage MaxwellSolve                1.00       4.406       4.407       4.409       1.622       0.057
Stage KrylovIteration             5.00       3.301       3.302       3.304       1.318       0.042
Stage CleaningIteration           5.36       3.056       3.056       3.057       6.903       0.039
```

- The time of a region is its host wall time, the regions it contains included, summed over its calls and over the threads that ran it. The movers of the species run in parallel, so `Mover` can exceed the cycle.
- `calls/cycle`, `min ms`, `mean ms` and `max ms` cover the steady state: calls and milliseconds per cycle, minimum, mean and maximum over the ranks.
- `std ms` is the standard deviation of the milliseconds per cycle over the steady-state cycles, root mean square over the ranks. A region that does not run every cycle, such as the data analysis, an output or a divergence cleaning, has a large one; so does `SteadyState`, which ends once.
- `run max s` covers the whole run, in seconds, maximum over the ranks.
- A call counts when it ends: one that runs across an edge of the steady state, such as an output job on its writer thread, counts on the side where it ends.

A region costs two clock reads and two atomic additions per call, about 35 ns, and no synchronisation or MPI call. At the end of each steady-state cycle the main thread reads the totals of the regions once, in about 0.1 µs; the report at the end makes seven `MPI_Reduce` calls. A 2D GEM run of 150 regions per 16 ms cycle ran 0.07 % slower, within the run-to-run spread.

### Regions

`src/include/Profiler.h` defines the regions. An indexed region is followed by its index in italics.

| Group | Regions |
|---|---|
| Run | `Init` (`InitInput`, `InitSelectDevice`, `InitFields`, `InitParticles`, `InitOutput`, `InitDevice`), `InitialMoments`, `SteadyState`, `Cycle` *(cycle)*, `Finalize` (with `FinalizeOutput`, `FinalizeDevice`) |
| Cycle stages | `ParticleCountLog`, `AnalysisPrepare`, `HeatFlux`, `AnalysisStart`, `FieldSolveE`, `AnalysisWait`, `MoverLaunch`, `Output`, `ParticleExchange`, `FieldSolveB`, `MomentsAwait`, `OutputCopy` |
| Field solver | `PoissonCorrection`, `MaxwellSource`, `MaxwellSolve` with `EigenvalueEstimate` (Chebyshev bounds) and `KrylovIteration` *(iteration)*, `CleaningIteration` *(iteration)* of the ∇·E and ∇·B cleaning solves, `KrylovReduce` (stream wait and `MPI_Allreduce` of the dot products of every solve), `MaxwellImage`, `CleaningImage`, `Preconditioner`, `SmoothE`, `FinishE`, `Faraday`, `DivBCleaning`, `HaloExchange`, `HaloBegin`, `HaloEnd` |
| Moments | `MomentKernelsWait`, `MomentHalo`, `DerivedMoments`, `HatFunctions` |
| Particles | `FieldPack`, `ParticleMerge` *(species)*, `Mover` *(species)* with `ParticleSplit`, `MoverCounts` and `ParticleExtract` *(species)*, `MoverWait`, `PlanetAbsorption`, `ParticleInjection` with `CompactionWait` *(species)*, `SortStart`, `ParticleMPI` *(species)*, `Exosphere` with `ExosphereSample` *(species)* on the mover threads, `SortFinish`, `ParticleSync`, `ParticleAppend` *(species)* with `MomentDeposit` *(species)*, `MomentScatter`, `ParticleSort` *(species)*, `SortWait` |
| Output | `OutputPoll`, `FieldCopyD2H`, `HeatFluxHalo`, `Catalyst`, `Diagnostics`, `OutputCopyWait` (wait for the copies of `OutputCopy`), `WriteRestart`, `WriteFields`, `WriteParticles`, `WriteTestParticles`, `OutputWait`, `OutputJob` on the writer threads, `OutputFinish` (blocking completion, such as the wait for every rank's checkpoint), `ParticleCopyD2H` with `SampleCountWait` *(species)* |
| Data analysis | `AnalysisInit`; on the analysis thread `AnalysisRun` *(cycle)* with `VelocityHistogram` *(species)*, `GMMFit` *(species)* with `GMMIteration` *(step)*, `MacrocellSpectra` *(species)*, `AnalysisSync` *(species)* and `AnalysisWrite`; `AnalysisFinalize`. An analysis region includes the staging of its records, whose copy is the first wait for its kernels |

The cycle timer opens `Cycle`, the cycle stages in loop order inside it, and `SteadyState`. A new region is an entry of `profiling::Region` with its name at the same place in `kRegionNames`, opened where it starts by

```cpp
profiling::Range range(profiling::Region::MoverWait);
profiling::Range range(profiling::Region::Mover, species); // indexed
```

and closed when `range` goes out of scope. Where a scope does not fit, a `profiling::Phase` opens a region with `start` (which closes the one it had open) and closes it with `stop`, as the cycle timer does for the stages.
