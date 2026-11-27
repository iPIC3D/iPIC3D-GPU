#!/usr/bin/env bash
# Build the kernel profiles (README.md): the static iPIC3D library (the
# profiles launch its kernels) and the four executables, in BUILD, for the
# NVIDIA architecture CUDA_ARCH.  Load the compilers, CUDA, MPI and CMake of
# the machine first.
#
# The folder of each GPU system calls it with its settings
# (profileGH200/build.sh, profileRTX5080/build.sh).  Every setting below can
# be set in the environment; BUILD and CUDA_ARCH have no default.
set -euo pipefail
(($# == 0)) || { echo "no options: set the settings in the environment" >&2; exit 1; }
here=$(cd "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

# ---- settings
BUILD=${BUILD:?set BUILD, the build folder}
CUDA_ARCH=${CUDA_ARCH:?set CUDA_ARCH, the GPU architecture (90: GH200)}
JOBS=${JOBS:-16}
# the profiles write no output: no HDF5 or ADIOS2 needed
CMAKE_OPTIONS=(${CMAKE_OPTIONS:--DUSE_HDF5=OFF -DUSE_PHDF5=OFF -DUSE_ADIOS2=OFF})
# ----

cmake -S "$here/../.." -B "$BUILD" -DCMAKE_BUILD_TYPE=Release \
  -DCUDA_ARCH="$CUDA_ARCH" -DBUILD_SHARED_LIBS=OFF -DBUILD_PROFILE_KERNELS=ON \
  "${CMAKE_OPTIONS[@]}"
cmake --build "$BUILD" -j "$JOBS" --target profileMover profileCellAwareMoment \
  profileMomentKernelNew profileCeilings
echo "built in $BUILD"
