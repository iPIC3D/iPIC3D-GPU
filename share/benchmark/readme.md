# Benchmark


## Usage

- Create a folder for each inputfile you'd like to benchmark, and copy the inputfile into the folders.
- Run the script `benchmark.sh`, 2 OpenMP threads per process by default, can be modified in [benchmark.sh](./benchmark.sh).
- Wait with a cup of coffee, check the output from time to time, they are executed in serial.
- Done.

The benchmark build uses `BENCHMARK_MODE=2`: simulation output and its
device-to-host copies are disabled, as are optional diagnostic calculations
(data analysis, macrocell spectra, and heat flux).
Task timing remains enabled and is printed to standard output. Use
`BENCHMARK_MODE=1` instead when those calculations should remain part of the
measured workload; heat-flux D2H and writing still remain disabled at level 1.

**NOTE**: The name of the folder must be in the format `name_XxYxZ_cycle`, as the script relies on the second segment to launch MPI processes.

## Baseline

There're few baseline files in this folder, which can be used as a reference for performance evaluation.
