# iPIC3D GPU scaling tests

This directory runs and plots strong- and weak-scaling campaigns on LUMI-G.

- **Strong scaling:** fixed global mesh; inputs cover 8–512 nodes.
- **Weak scaling:** constant work per MPI rank/GPU; inputs cover 4–512 nodes.

The launchers use 8 MPI ranks/GPU GCDs per node and 6 OpenMP threads per rank.
Their Slurm resources and binding are LUMI-G-specific.

## Requirements

- A GPU build of iPIC3D accessible and executable on compute nodes.
- A readable shell profile that loads its MPI, ROCm, and runtime environment.
- A valid LUMI Slurm account and `sbatch`, `srun`, `squeue`, and `sacct`.
- Bash, `awk`, `sort`, Python 3, and Matplotlib.

## 1. Copy the workflow to scratch

Copy the complete `scaling` directory because the Python scripts are at its top
level. Use a fresh destination to avoid mixing campaigns.

```bash
cd /path/to/repository/share
export IPIC3D_SCALING_WORKDIR="${SCRATCH}/ipic3d-scaling"
cp -a scaling "$IPIC3D_SCALING_WORKDIR"
cd "$IPIC3D_SCALING_WORKDIR"
```

## 2. Configure the environment

Set these variables in the shell that will launch the campaigns:

```bash
export IPIC3D_SLURM_ACCOUNT='YOUR_PROJECT_ACCOUNT'
export IPIC3D_PROFILE='/absolute/path/to/ipic3d-runtime-profile.sh'
export IPIC3D_EXECUTABLE='/absolute/path/to/iPIC3D'
```

The profile must be readable and the executable must have execute permission.
Both paths must be visible from compute nodes; absolute paths are recommended.

## 3. Preview and submit

Run launchers **directly from the login shell**; do not put `sbatch` before them.
Each launcher validates the inputs and submits one job per node count.

Always preview first:

```bash
./strong/runIPIC3DStrong.slurm --dry-run
./weak/runIPIC3DWeak.slurm --dry-run
```

Dry runs validate configuration, paths, ranges, and MPI topology. They print the
exact `sbatch` commands without submitting jobs or creating results.

Submit every available input by omitting node arguments:

```bash
./strong/runIPIC3DStrong.slurm
./weak/runIPIC3DWeak.slurm
```

There is no literal `all` argument. One node argument selects exactly that size;
two select available inputs in an inclusive range:

```bash
./strong/runIPIC3DStrong.slurm 64
./strong/runIPIC3DStrong.slurm 16 128
./weak/runIPIC3DWeak.slurm 8 64
```

Ranges only select existing input files. Before submission, the launcher verifies that `XLEN * YLEN * ZLEN = nodes * 8`.

Use `--exclude` or `-x` to exclude Slurm nodes. Quote bracketed lists so the shell does not expand them:

```bash
./strong/runIPIC3DStrong.slurm --exclude 'nid[0010-0012],nid0020' 16 128
./weak/runIPIC3DWeak.slurm --dry-run -x 'nid[0010-0012]' 8 64
```

## 4. Monitor and inspect

The launcher prints every job ID and the new campaign directory. Monitor with:

```bash
squeue -u "$USER"
sacct -j JOB_ID_1,JOB_ID_2 --format=JobID,JobName%20,State,Elapsed,ExitCode
```

Plot only after every selected top-level job is `COMPLETED` with exit code `0:0`.

Campaigns are created beside their launcher as
`strong/results_YYYYMMDD_HHMMSS/` and `weak/results_YYYYMMDD_HHMMSS/`. Each has
one `N<NODES>/` working directory plus stdout and stderr files at its root.

Check nonempty error logs and final timings before plotting:

```bash
find strong/results_TIMESTAMP -name 'stderr_N*' -type f -size +0c -print
grep -H 'Cycle loop' strong/results_TIMESTAMP/stdout_N*
find weak/results_TIMESTAMP -name 'stderr_N*' -type f -size +0c -print
grep -H 'Cycle loop' weak/results_TIMESTAMP/stdout_N*
```

Use the timestamp reported by the launcher. Do not combine separate campaigns
or retries; duplicate node measurements are rejected.

## 5. Plot efficiency

From the copied `scaling` directory, run:

```bash
python3 scalingStrong.py strong/results_TIMESTAMP
python3 scalingWeak.py weak/results_TIMESTAMP
```

The scripts print timings and efficiencies and create `strong_scaling_efficiency.png`
or `weak_scaling_efficiency.png` inside the corresponding result directory.

Optional inclusive filters select a completed subset:

```bash
python3 scalingStrong.py strong/results_TIMESTAMP --min-nodes 16 --max-nodes 128
python3 scalingWeak.py weak/results_TIMESTAMP --min-nodes 8 --max-nodes 64
```

The smallest node count remaining after filtering becomes the 100% baseline.
For node count `n`, cycle-loop time `T(n)`, GPU count `G(n) = 8n`, and smallest
included node count `n0`:

```text
strong efficiency(n) = 100 * T(n0) * G(n0) / (T(n) * G(n))
weak efficiency(n)   = 100 * T(n0) / T(n)
```

All included stdout files must report the same cycle count. A missing `Cycle loop`
line usually means the job failed or did not finish; inspect stderr and `sacct`.