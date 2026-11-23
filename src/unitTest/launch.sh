#!/bin/sh
# Starts one unit test: launch.sh <ranks> <program> [arguments...]
# <ranks> is the number of MPI ranks, or "serial" for a program without MPI.
#
# Inside a Slurm allocation (sbatch, salloc) the test is a job step, so that it
# runs on the allocated nodes, where the GPUs are, also when ctest itself runs
# on a login node.  A test fixes its own rank count, whatever the job asked
# for: the per-task and per-node requests of the job are not inherited, and
# the step shares the CPUs, GPUs and memory of the allocation, with the ranks
# not bound to cores.  IPIC3D_SRUN_FLAGS adds options of the site, for example
# --mpi=pmix.
#
# Elsewhere an MPI test runs with the launcher CMake found (IPIC3D_MPIEXEC and
# its flags, set by unitTestLaunch.cmake), a serial one directly.

if [ "$#" -lt 2 ]; then
  echo "usage: launch.sh <ranks|serial> <program> [arguments...]" >&2
  exit 2
fi
ranks=$1
program=$2
shift 2

if [ -n "${SLURM_JOB_ID:-}" ]; then
  [ "$ranks" = serial ] && ranks=1
  unset SLURM_NTASKS_PER_NODE SLURM_CPUS_PER_TASK SRUN_CPUS_PER_TASK \
    SLURM_GPUS_PER_TASK SLURM_TRES_PER_TASK \
    SLURM_MEM_PER_CPU SLURM_MEM_PER_GPU SLURM_MEM_PER_NODE
  # --overlap (Slurm 20.11 and later): also inside a step that holds the
  # allocation, such as an interactive shell started with srun --pty
  overlap=
  if srun --help 2>&1 | grep -q -e '--overlap'; then
    overlap=--overlap
  fi
  # shellcheck disable=SC2086 # the flags are separate words
  exec srun --ntasks="$ranks" --overcommit --cpu-bind=none --mem=0 $overlap \
    ${IPIC3D_SRUN_FLAGS:-} "$program" "$@"
fi

if command -v srun >/dev/null 2>&1; then
  echo "note: no Slurm allocation, the test runs on this node" \
    "(use salloc or sbatch to reach a GPU node)" >&2
fi

if [ "$ranks" = serial ]; then
  exec "$program" "$@"
fi
if [ -z "${IPIC3D_MPIEXEC:-}" ]; then
  echo "launch.sh: CMake found no MPI launcher for a test of $ranks ranks" >&2
  exit 1
fi
# shellcheck disable=SC2086 # the flags are separate words
exec "$IPIC3D_MPIEXEC" ${IPIC3D_MPIEXEC_NUMPROC_FLAG:--n} "$ranks" \
  ${IPIC3D_MPIEXEC_PREFLAGS:-} "$program" ${IPIC3D_MPIEXEC_POSTFLAGS:-} "$@"
