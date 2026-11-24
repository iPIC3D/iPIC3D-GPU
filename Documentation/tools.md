# Tools

[Documentation index](../README.md#documentation)

## Input check

`python3 scripts/ipic_preflight.py <input file>` prints most of the startup report of a run (parameters, memory estimate, warnings) without MPI or a GPU, and stops at the first error among the checks it shares with the run. It does not check the write-method, particle-selection, `AsyncOutput` and data-analysis keys, nor whether the build has the backend asked for.

`scripts/` also holds unmaintained helpers of the original iPIC3D (`ipic`, `ipic-setinp`, `ipic-setjob`, ...), written for its old build.

## Benchmarks and scaling

| Folder | Purpose |
|---|---|
| [`share/benchmark`](../share/benchmark/readme.md) | Local benchmark of the decks in its folders, with `BENCHMARK_MODE=2` and task timing. Its `build.sh` uses the default `CUDA_ARCH`; set yours |
| [`share/scaling`](../share/scaling/README.md) | Strong- and weak-scaling campaigns and plots, with launchers for LUMI-G |

## Post-processing

Scripts in `postprocessing_tools/python` (Python 3 with NumPy, and Matplotlib for the plots; `plot_total_particles.py` also needs pandas):

| Script | Purpose |
|---|---|
| `read_analysis_records.py` | Read the [data-analysis](data-analysis.md#files) records |
| `plot_velocity_histograms.py` | Plot the velocity histograms with their GMM fits, or the fits alone (`--gmm-only`) |
| `plot_macrocell_spectra.py` | Plot the macrocell spectra |
| `plot_total_particles.py` | Total particles against cycle, from `data/particleNum*.csv` |
| `magnetosphere_rho_vtk.py` | Density slices through the midpoint in y, from VTK files |
| `proc2vtr.py`, `proc2vtr_serial.py`, `h5tovtr.py`, `h5tovtr_v2.py` | Convert `shdf5` (`proc2vtr*`: SciPy, PyTables, VTK; `proc2vtr.py` also mpi4py, run with `mpirun`) or `phdf5` (`h5tovtr*`: h5py, evtk) fields to VTK. Python 2; set the data folder and cycles at the top of the script |

`postprocessing_tools` also holds older C++ HDF5-to-VTK converters and C++/Octave virtual-satellite and FFT tools.

To inspect HDF5 files, use `h5ls` or `h5dump`; for ADIOS2 files, `bpls`.

## Code formatting

`make format` runs `clang-format` (version 20 or newer, for `ReflowComments: Always`) on the C, C++ and CUDA sources, with the project's `.clang-format`. It takes the first `clang-format` on `PATH`; to use another, configure with `-DIPIC3D_CLANG_FORMAT=<path>`.
