# Testing

[Documentation index](../README.md#documentation)

## Running the tests

```shell
cmake -DBUILD_UTESTS=ON ..
make -j"$(nproc)"
ctest --output-on-failure
```

Tests that use the GPU run one at a time, also with `ctest -j`, and need about 3 GB of GPU memory. To run a subset, use `ctest -R <regex>`.

## On Slurm

Run `ctest` inside an allocation. Each run of a test then starts as a job step (`src/unitTest/launch.sh` detects `SLURM_JOB_ID`) with its own number of ranks (1 to 8, 27 for `halo_distributed27`), sharing the CPUs and GPUs of the allocation, so one node with one GPU is enough:

```shell
salloc --nodes=1 --ntasks=8 --gpus=1 --time=00:30:00
ctest --output-on-failure
```

Options your site needs for `srun` go in `IPIC3D_SRUN_FLAGS`, for example `export IPIC3D_SRUN_FLAGS="--mpi=pmix"`. Outside an allocation, the tests use the MPI launcher found by CMake.

## Test suite

| Test | Ranks | Covers |
|---|---|---|
| `reduction` | 1 | GPU block reductions and scans against the CPU |
| `histogram` | 1 | 2D, rank and macrocell velocity histograms against a CPU reference |
| `gmm` | 1 | Gaussian-mixture fit of a synthetic histogram |
| `planet` | 1 | Planet boundary kernels |
| `mover_walls` | 1 | Particle push with reflecting walls |
| `particle_control` | 1 | Splitting and merging kernels |
| `particle_output_select` | 1 | Selection of the written particles |
| `field_kernels` | 1 | Boundary rows, susceptibility, launch sets and layer blends of the Maxwell solve |
| `cleaning` | 1 | ∇·E correction and ∇·B cleaning kernels |
| `maxwell_image` | 8 | The Maxwell operator against a reference |
| `maxwell_manufactured` | 8 | Manufactured solutions of the solve |
| `maxwell_wave` | 8 | Oblique wave, node B interpolation |
| `maxwell_inflow` | 8 | Inflow layers |
| `maxwell_conductor_wall` | 8 | Conductor wall rows |
| `maxwell_closed_box` | 8 | Closed box, conductor edges |
| `maxwell_conductor_modes` | 8 | Energy conservation of a vacuum standing wave between walls |
| `maxwell_ghost_fill` | 8 | Ghost fills |
| `halo_distributed` | 8 | CPU and GPU halo exchange, exact values |
| `halo_self` | 1 | Halo exchange of a rank with itself |
| `halo_distributed27` | 27 | The same on 27 ranks, registered only when `MPIEXEC_MAX_NUMPROCS` is at least 27 at configure time |
| `thread_errors` | 2 | An error on any thread ends every rank |
| `analysis_records` | 1 | Data-analysis record files on a restart |
| `async_output` | 1 | Deferred output ordering and error handling |
| `restart_output_files` | 1 | Serial-HDF5 files on a restart (only with HDF5) |
| `solver_runs` | 1, 4 | End-to-end runs of `iPIC3D` on reduced decks (only with `BENCHMARK_MODE=0`) |

`maxwell_image`, `maxwell_wave`, `maxwell_inflow` and `maxwell_conductor_wall` run with `HaloOverlap` 0 and 1.

`solver_runs` makes up to 27 short runs, depending on the backends built, and checks energy bounds, particle conservation, restart continuity, agreement between 1 and 4 ranks and between sorted and unsorted deposits, all available output backends, every initial condition, and the planet and injection paths.
