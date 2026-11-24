# MeluXina: four NVIDIA A100 (40 GB each) per node, one MPI rank per GPU.
# Sourced by ../submit.sh and ../job.sh.

TASKS_PER_NODE=4
GPU_LABEL=GPUs  # x-axis label of the plots, without spaces
SBATCH_OPTIONS=(--partition=gpu --qos=default --gpus-per-node=4 --cpus-per-task=32)

# Runs the command on every rank. iPIC3D gives each rank the GPU of its
# node-local rank, so no device wrapper is needed. srun does not inherit
# --cpus-per-task from sbatch, hence the repeat.
system_srun() {
  export OMP_NUM_THREADS=16  # 32 hardware threads are 16 cores
  srun --kill-on-bad-exit=1 --cpus-per-task=32 "$@"
}
