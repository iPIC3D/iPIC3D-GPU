# Benchmark


## Usage

- Create a folder for each inputfile you'd like to benchmark, and copy the inputfile into the folders.
- Run the script `benchmark.sh`, 2 OpenMP threads per process by default, can be modified in [benchmark.sh](./benchmark.sh).
- Wait with a cup of coffee, check the output from time to time, they are executed in serial.
- Done.

**NOTE**: The name of the folder must be in the format `name_XxYxZ_cycle`, as the script relies on the second segment to launch MPI processes.

## Baseline

There're few baseline files in this folder, which can be used as a reference for performance evaluation.
