#!/bin/bash -l
# One run of a scaling campaign, submitted by submit.sh into its run folder.
# Arguments: system folder, kind (strong or weak), input deck.

set -euo pipefail

SYSTEM_DIR=$1
kind=$2
deck=$3

source "$SYSTEM_DIR/system.sh"
set +u  # site profiles may read unset variables
source "$IPIC3D_PROFILE"
set -u

# A crash at scale would otherwise write one large core file per rank.
ulimit -c 0

# plot_scaling.py reads this line.
echo "Scaling run: system=$(basename -- "$SYSTEM_DIR") kind=$kind" \
  "nodes=$SLURM_JOB_NUM_NODES ranks=$SLURM_NTASKS gpu_label=$GPU_LABEL" \
  "deck=$(basename -- "$deck")"
system_srun "$IPIC3D_EXECUTABLE" "$deck"
