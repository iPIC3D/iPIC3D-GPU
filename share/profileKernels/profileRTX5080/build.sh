#!/usr/bin/env bash
# Build the kernel profiles for the NVIDIA GeForce RTX 5080 of the
# workstation (../README.md): ../build.sh with the settings below; the others
# can be set in the environment.
set -euo pipefail
here=$(cd "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

# ---- settings
export BUILD=${BUILD:-"$here/build"}
export CUDA_ARCH=${CUDA_ARCH:-120}    # Blackwell, GeForce
# ----

exec bash "$here/../build.sh" "$@"
