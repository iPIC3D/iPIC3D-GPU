# Code architecture

[Documentation index](../README.md#documentation)

iPIC3D-GPU is an implicit particle-in-cell code. The field solver, particle mover, moment deposit, sorting and analyses run on the GPU; the host handles setup, the MPI particle exchange, the exosphere source and file output.

## Source tree

| Path | Content |
|---|---|
| `src/ipic3d/iPIC3D.cpp` | `main` and the time loop |
| `src/ipic3d/main/` | `c_Solver` (`iPIC3Dlib.cu`), input parsing (`Collective.cpp`), build record |
| `src/ipic3d/fields/` | `EMfields3D` (case initialisers, GPU solver in `EMfields3DGPU.cpp`), `FieldBoundary` |
| `src/ipic3d/cudaKernel/` | GPU kernels: mover, moments, sort, injection, planet, particle control, Krylov BLAS, stencils, halo exchange, cleaning |
| `src/ipic3d/particles/` | `ParticleSoAHost` (host particle arrays, loaders), `ParticleCommInjection` (MPI exchange, injection), `ExosphereIonization` |
| `src/ipic3d/dataAnalysis/` | Velocity histograms, GMM pipeline, macrocell spectra |
| `src/ipic3d/inputoutput/` | `IOManager`, writers, `AsyncOutput` (deferred output), restart reader and slots |
| `src/ipic3d/communication/`, `grids/` | Cartesian MPI topology and grid |
| `src/ipic3d/solvers/`, `mathlib/`, `bc/` | Host solvers, math library and field boundary routines, mostly unused by the GPU time loop |
| `src/ipic3d/utility/`, `performances/`, `ConfigFile/`, `PSKOutput3D/` | MPI wrapper and helpers, timing and [profiler](profiling.md), input parser, HDF5 adaptor |
| `src/include/` | Headers; `CUDA/` for device code, `dataAnalysis/` for the analyses, `hip/` for the CUDA-to-HIP mapping |
| `src/unitTest/` | [Unit tests](testing.md) |
| `share/`, `scripts/`, `postprocessing_tools/` | Decks, benchmarks and [tools](tools.md) |

## Grid and data placement

Each rank holds a box of the grid with one ghost layer per side.

| Quantity | Location |
|---|---|
| E, species moments (ρ, J, pressure) | Nodes |
| B | Cell centres (advanced by Faraday's law) and nodes (interpolated, read by the mover) |
| ρ_c, ∇·E potential | Cell centres |
| ∇·B potential | Nodes |
| Particles | Structure of arrays on the GPU, with a host mirror for I/O |

## Time loop

Before the loop, `Init` reads the input, builds the topology, selects the GPU, initialises fields and particles, and `CalculateMoments` deposits the first moments. Each cycle then runs:

| Step | Call | Work |
|---|---|---|
| 1 | `writeParticleNum` | Particle counts per species |
| 2 | `DA.prepareAnalysis` | Sort for the macrocell analyses |
| 3 | `ScheduleHeatFlux` | Heat-flux kernels, if an output needs them |
| 4 | `DA.startAnalysis` | Analysis thread, on a low-priority GPU stream |
| 5 | `CalculateField` | ∇·E correction, Maxwell source, Krylov solve for E(n+θ), smoothing, boundary layers |
| 6 | `DA.waitForAnalysis` | The mover changes the particles the analysis reads |
| 7 | `ParticlesMoverMomentAsync` | Merge if due; per species, split if due, launch the mover and, except on sorting cycles, the moments |
| 8 | `WriteOutput` | Field copies, then host output, while the GPU mover runs |
| 9 | `MoverAwaitAndPclExchange` | Planet, injection, MPI exchange, exosphere, sort, remaining moments |
| 10 | `CalculateB` | Faraday's law, open-face layers, ∇·B cleaning, interpolation to nodes, E(n) → E(n+1) |
| 11 | `MomentsAwait` | Merge scheduling, ghost moment exchange, planet charge, outflow layer, species sums, hat moments |
| 12 | `outputCopyAsync` | Particle copies and particle sums for the next cycle's output |

E(n) stays in its array through steps 5 to 9, so the output of cycle n holds E, B and particles at the same time level. The equations are in [Field solver](field-solver.md).

## GPU execution

- Each species has its own streams, so species advance concurrently. A thread pool launches them.
- Stream priorities: halo exchange ≥ field solver > particle streams ≥ diagnostics (analysis and heat flux). The levels depend on the device's priority range. The halo stream exists only with `HaloOverlap = 1`.
- Separate streams hold planet processing and output copies. Events order them against the solver and the mover.
- Halo exchanges pack up to 26 neighbours into one MPI round. `HaloOverlap` [overlaps them](field-solver.md#overlapped-halo-exchange) with computation.
- Kernels take the grid (`grid3DCUDA`) and, per species, the mover and injection parameters (`moverParameter`, `injectionParameter`) by value, built once at initialization: a warp reads a kernel parameter once for all its threads (`KERNEL_CONSTANT` in `src/include/CUDA/cudaTypeDef.cuh`). Kernel parameters are read-only, so the counters the movers update atomically (`MoverCounters`) live in device memory.

## Memory

- The host side of every GPU transfer is page-locked for the whole run: host field, moment and particle arrays count in the rank's physical memory.
- Device particle arrays start at `PARTICLE_ARRAY_HEADROOM` (default 1.3) times the particle count. When the count times `PARTICLE_ARRAY_RESIZE_THRESHOLD` (1.2) reaches the capacity, at the particle exchange, they grow to `PARTICLE_ARRAY_RESIZE_FACTOR` (1.3) times the count; the exchange buffers grow by that factor too. The factor must exceed the threshold, or the arrays would grow at every exchange; the headroom may be below it, and the arrays then grow at the first exchange. These are [CMake options](building.md#cmake-options).
- The spare room holds the open-outflow copies the mover appends; a mover that finds it full stops the run. A [split](particles.md#particle-count-control-splitting-and-merging) stops at the capacity divided by `PARTICLE_ARRAY_RESIZE_THRESHOLD`.
- The run prints its estimated device and host memory at startup.
