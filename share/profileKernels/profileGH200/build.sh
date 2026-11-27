#!/usr/bin/env bash
# Build the kernel profiles for NVIDIA GH200 (../README.md): ../build.sh with
# the settings below; the others can be set in the environment.  Load the
# compilers, CUDA, MPI and CMake of the machine first.
set -euo pipefail
here=$(cd "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

# ---- settings
export BUILD=${BUILD:-"$here/build"}
export CUDA_ARCH=${CUDA_ARCH:-90}     # Hopper
# ----

exec bash "$here/../build.sh" "$@"
