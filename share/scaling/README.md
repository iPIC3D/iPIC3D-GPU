# iPIC3D GPU scaling tests

Strong- and weak-scaling campaigns of the 3D GEM case, one folder per GPU system.

```text
share/scaling/
  submit.sh          submits a campaign: one Slurm job per node count
  job.sh             runs one job of a campaign (called through sbatch)
  plot_scaling.py    execution time and efficiency: figures (PNG, PDF), table, CSV
  mi250x/            LUMI-G: 8 MI250X GCDs per node, 64 GB each
    system.sh        ranks per node, Slurm options, srun line
    selectGPU.sh     shows each rank one GCD
    strong/          GEMstrongN<nodes>.inp
    weak/            GEMweakN<nodes>.inp
  a100/              MeluXina: 4 A100 per node, 40 GB each
    system.sh
    strong/
    weak/
```

Every job runs one MPI rank per GPU (per GCD on the MI250X).

## Test cases

At full load a rank owns 64 x 64 x 64 cells of a 10 x 10 x 10 domain, with four
species. The A100 decks keep the grid and reduce the particles per cell, so
both systems fill a similar fraction of their GPU memory. The memory is the
estimate of `scripts/ipic_preflight.py`.

| | `mi250x` | `a100` |
|---|---|---|
| `npcelx` x `npcely` x `npcelz` per species | 7 x 8 x 8 = 448 | 7 x 8 x 5 = 280 |
| Particles per rank | 469 762 048 | 293 601 280 |
| GPU memory per rank | 50.4 GB of 64 | 31.7 GB of 40 |
| Host memory per node | 248 GB | 78 GB |

Weak decks hold this load on every rank. Strong decks hold it at their
smallest node count and split the same grid further. Every deck runs 101
cycles: the first is a warm-up, the other 100 are measured. The decks write no
output and run no data analysis. They set `HaloOverlap = 1`, which overlaps the
field solver's halo exchanges with computation
([overlapped halo exchange](../../Documentation/field-solver.md#overlapped-halo-exchange)).

Weak scaling, `<system>/weak/GEMweakN<nodes>.inp`:

| GPUs | MPI topology | Grid cells | `mi250x` | `a100` |
|---|---|---|---|---|
| 4 | 2x2x1 | 128x128x64 | | N1 |
| 8 | 2x2x2 | 128x128x128 | | N2 |
| 16 | 4x2x2 | 256x128x128 | | N4 |
| 32 | 4x4x2 | 256x256x128 | N4 | N8 |
| 64 | 4x4x4 | 256x256x256 | N8 | N16 |
| 128 | 8x4x4 | 512x256x256 | N16 | N32 |
| 256 | 8x8x4 | 512x512x256 | N32 | N64 |
| 512 | 8x8x8 | 512x512x512 | N64 | N128 |
| 1024 | 16x8x8 | 1024x512x512 | N128 | |
| 2048 | 16x16x8 | 1024x1024x512 | N256 | |
| 4096 | 16x16x16 | 1024x1024x1024 | N512 | |

Strong scaling, `<system>/strong/GEMstrongN<nodes>.inp`:

| | `mi250x` | `a100` |
|---|---|---|
| Grid cells | 256x256x256 | 128x128x128 |
| Domain | 40x40x40 | 20x20x20 |
| Node counts | N8 to N512 | N2 to N128 |
| GPUs | 64 to 4096 | 8 to 512 |
| Cells per rank | 64x64x64 down to 16x16x16 | 64x64x64 down to 16x16x16 |

At equal GPU count both systems use the same topology.

## 1. Build

Build with `-DBENCHMARK_MODE=2` ([benchmark modes](../../Documentation/building.md#benchmark-modes)).
With the output compiled in, every rank still appends to its
`particleNum<rank>.csv` in each timed cycle, even with these decks.

```shell
# MI250X
CC=amdclang CXX=amdclang++ cmake -DHIP_ON=ON -DHIP_ARCH=gfx90a -DBENCHMARK_MODE=2 ..
# A100
cmake -DCUDA_ARCH=80 -DBENCHMARK_MODE=2 ..
```

The MPI library must be GPU-aware: the halo exchange hands GPU buffers to MPI.

## 2. Set the paths

```shell
export IPIC3D_SLURM_ACCOUNT='YOUR_PROJECT_ACCOUNT'
export IPIC3D_PROFILE='/absolute/path/ipic3d-profile.sh'
export IPIC3D_EXECUTABLE='/absolute/path/build/iPIC3D'
```

Each job sources `IPIC3D_PROFILE`: it loads the compiler, the GPU runtime and
the MPI library of the build. Both paths must be visible from the compute
nodes.

## 3. Submit

Run from this folder. A dry run checks the decks and prints the `sbatch`
commands without creating or submitting anything:

```shell
./submit.sh mi250x/strong --dry-run
```

Submit every deck of a folder, one node count, or a range of node counts:

```shell
./submit.sh mi250x/strong
./submit.sh a100/weak 1
./submit.sh a100/weak 2 32
```

Options after `--` go to `sbatch` after the defaults of `system.sh`, so they
override them. The time limit is 30 minutes for strong and 20 for weak jobs.

```shell
./submit.sh a100/weak 1 -- --qos=test --time=00:10:00
./submit.sh mi250x/strong 16 128 -- --exclude='nid[0010-0012],nid0020'
```

`submit.sh` checks that `XLEN*YLEN*ZLEN` of each deck equals the node count
times the ranks per node. Each submission prints its campaign folder:

```text
<system>/<kind>/results_<date>_<time>/
  campaign.txt              date, executable, profile, sbatch options, job IDs
  N<nodes>/                 run folder with a copy of the deck
  stdout_N<nodes>_<job>     output of the run
  stderr_N<nodes>_<job>
```

## 4. Check

```shell
campaign=a100/strong/results_YYYYMMDD_HHMMSS
sacct -j "$(awk '$1 == "job" { print $3 }' "$campaign/campaign.txt" | paste -sd,)" \
  --format=JobID,JobName%24,State,Elapsed,ExitCode
grep -H 'Cycle loop' "$campaign"/stdout_N*
```

A run has finished when its stdout has the `Cycle loop` line.

## 5. Plot

`plot_scaling.py` needs Python 3 with Matplotlib.

```shell
python3 plot_scaling.py a100/strong/results_YYYYMMDD_HHMMSS
```

It prints a table and writes into the campaign folder, each figure as PNG and
PDF (`<kind>` is `strong` or `weak`):

| File | Content |
|---|---|
| `<kind>_scaling_efficiency.png`, `.pdf` | Efficiency with its value at each point and the ideal line, with a title |
| `<kind>_scaling_time_efficiency.png`, `.pdf` | Execution time (left axis, log scale for strong) and efficiency (right axis), without a title |
| `<kind>_scaling.csv` | The numbers of the table |

Runs that did not finish are listed and left out. Font sizes are set by
`FONT_SIZES` at the top of the script.

The execution time is the sum of the 100 measured cycles, from the
`Execution time cycle` lines of rank 0. The efficiency of a run on G GPUs
with time T is measured against the smallest run of the plot, on G0 GPUs
with time T0:

```text
strong efficiency = 100 * T0 * G0 / (T * G)
weak efficiency   = 100 * T0 / T
```

More options:

```shell
# leave out node counts, for example a run slowed by a bad node
python3 plot_scaling.py a100/strong/results_A --min-nodes 4 --max-nodes 64 --skip-nodes 16

# combine campaigns of one system and kind, such as a resubmitted node count;
# the figures go to the first folder
python3 plot_scaling.py a100/weak/results_A a100/weak/results_B
```

A plot holds one system and one kind. A node count may appear only once
across its campaigns; a second finished run stops the script, so leave one
out with `--skip-nodes`.

## Adding a system

Add a folder with:

- `system.sh` that sets `TASKS_PER_NODE`, `GPU_LABEL` (the x-axis label of
  the plots, without spaces), the array `SBATCH_OPTIONS` and the function
  `system_srun`, which launches its arguments on every rank. See the two
  existing ones.
- `strong/` and `weak/` with the decks. A system with four GPUs per node can
  copy the `a100` decks, and one with eight the `mi250x` decks. Set `npcelz` so
  that `python3 ../../scripts/ipic_preflight.py <deck>` fits the GPU memory on
  the full-load decks.
