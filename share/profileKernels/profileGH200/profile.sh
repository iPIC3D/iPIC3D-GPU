#!/usr/bin/env bash
# Profile the kernels on NVIDIA GH200 (../README.md): ../profile.sh with the
# settings below and its defaults: 64³ cells, 64 to 512 particles per cell,
# kernel replay (the 96 or 144 GB hold every profile twice).  Build with
# build.sh first.  The other settings of ../profile.sh can be set in the
# environment; on Slurm, LAUNCHER is the launch of one task on one GPU:
#   LAUNCHER="srun --nodes=1 --ntasks=1 --gpus-per-task=1" bash profile.sh
set -euo pipefail
here=$(cd "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

# ---- settings
export BUILD=${BUILD:-"$here/build"}
export OUT=${OUT:-"$here/results/$(date +%Y%m%d-%H%M%S)"}
# ----

exec bash "$here/../profile.sh" "$@"
