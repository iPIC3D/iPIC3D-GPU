#!/usr/bin/env bash
# Profile the kernels on the NVIDIA GeForce RTX 5080 of the workstation
# (../README.md): ../profile.sh with the settings below.  Build with build.sh
# first, and run on an otherwise idle GPU.  The other settings of
# ../profile.sh can be set in the environment.
#
# The GPU has 16 GB and drives the desktop, which a full GPU takes down:
# - Nsight Compute runs the whole profile again for each counter pass
#   (application replay) instead of keeping a copy of the device memory;
# - up to 256 particles per cell at 64³ cells (8.4 GiB in use with the mover,
#   desktop included; 512 does not fit the mover and the cell-aware deposit);
# - the run stops when the GPU memory in use passes MAX_GPU_MIB.
set -euo pipefail
(($# == 0)) || { echo "no options: edit the settings at the top, or set them in the environment" >&2; exit 1; }
here=$(cd "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

# ---- settings
export BUILD=${BUILD:-"$here/build"}
export OUT=${OUT:-"$here/results/$(date +%Y%m%d-%H%M%S)"}
export PPC=${PPC:-"64 128 256"}
export REPLAY=${REPLAY:-application}
MAX_GPU_MIB=${MAX_GPU_MIB:-10000}     # GPU memory in use (all processes) that stops the run
# ----

# the run in a process group of its own, stopped as a whole when the GPU
# memory in use passes MAX_GPU_MIB
setsid bash "$here/../profile.sh" &
run=$!
trap 'kill -- -"$run" 2> /dev/null' INT TERM
while kill -0 "$run" 2> /dev/null; do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -n 1) || used=0
  if ((used > MAX_GPU_MIB)); then
    echo "GPU memory in use $used MiB > MAX_GPU_MIB $MAX_GPU_MIB: run stopped" >&2
    kill -9 -- -"$run"
  fi
  sleep 0.2
done
wait "$run"
