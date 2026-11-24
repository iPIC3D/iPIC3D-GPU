# LUMI-G: eight AMD MI250X GCDs (64 GB each) per node, one MPI rank per GCD.
# Sourced by ../submit.sh and ../job.sh.

TASKS_PER_NODE=8
GPU_LABEL=GPUs/GCDs  # x-axis label of the plots, without spaces
SBATCH_OPTIONS=(--partition=standard-g --gpus-per-node=8 --cpus-per-task=6)

# Runs the command on every rank. Local rank i sees only GCD i (selectGPU.sh)
# and runs on six cores of the CCD attached to that GCD.
system_srun() {
  export MPICH_GPU_SUPPORT_ENABLED=1  # the halo exchange passes GPU buffers to MPI
  # Keeps every GPU exchange buffer mapping of the run: the default of 50
  # evicts mappings, and a peer GCD then faults on an unmapped address at scale.
  export MPICH_GPU_IPC_CACHE_MAX_SIZE=1024
  export OMP_NUM_THREADS=6
  local masks=7e000000000000,7e00000000000000,7e0000,7e000000
  masks=$masks,7e,7e00,7e00000000,7e0000000000
  srun --kill-on-bad-exit=1 --cpu-bind=mask_cpu:$masks \
    "$SYSTEM_DIR/selectGPU.sh" "$@"
}
