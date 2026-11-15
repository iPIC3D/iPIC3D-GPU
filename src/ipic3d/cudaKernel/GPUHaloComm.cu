/**
 * @file GPUHaloComm.cu
 * @brief Implementation of GPU-aware MPI halo communication and BC kernels.
 *
 * This file mirrors the logic of Com3DNonblk.cpp and BcFields3D.cpp, but
 * operates directly on device pointers (GPUFieldArray3).  The exchanges go
 * through the batched pack/unpack of GPUBatchedHaloComm.cu.
 */

#include "EMfields3D.h"
#include "GPUHaloComm.cuh"
#include "VCtopology3D.h"
#include <cassert>
#include <iostream>

// =========================================================================
//  Helper: linear index into a flat (nx, ny, nz) row-major array
// =========================================================================
__device__ __forceinline__ int idx3(int i, int j, int k, int ny, int nz) {
  return (i * ny + j) * nz + k;
}

// =========================================================================
//  BC kernel – face boundary condition application
//  bcType  0 = Dirichlet 0 second-order  ghost = -interior
//          1 = Dirichlet 0 first-order   ghost = 0
//          2 = Neumann   0 first-order   ghost = interior
//          3 = even node mirror          ghost = second interior layer
//              (BC_FACE_NODE_MIRROR: symmetric about the face node)
//         -1 = ghost layer left unchanged (BC_FACE_KEEP, not launched)
// =========================================================================

/** The rule bcType on the ghost plane of one face, one thread per point
 *  (1D launch): point t is (slow, fast) = (t / nFast, t % nFast), at index
 *  base + slow * strideSlow + fast * strideFast, fast being k (j on a z
 *  face); the first and second interior layers are step and 2 step away
 *  (step: the index stride toward the interior). */
__global__ void gpuBCfacePlane(cudaSolverType* __restrict__ arr, int base,
                               int nFast, int strideSlow, int strideFast,
                               int points, int step, int bcType) {
  const int t = blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= points)
    return;
  const int slow = t / nFast;
  const int dst = base + slow * strideSlow + (t - slow * nFast) * strideFast;
  switch (bcType) {
  case 0:
    arr[dst] = -arr[dst + step];
    break;
  case 1:
    arr[dst] = 0.0;
    break;
  case 2:
    arr[dst] = arr[dst + step];
    break;
  case BC_FACE_NODE_MIRROR:
    arr[dst] = arr[dst + 2 * step];
    break;
  }
}

// =========================================================================
//  Self-copy kernels – periodic face swap when rank == neighbour (not used,
//  see the batched kernels); contiguous index of the plane on threadIdx.x
// =========================================================================

__global__ void gpuSelfCopyFaceX(cudaSolverType* __restrict__ arr, int nx,
                                 int ny, int nz, int offset) {
  int j = blockIdx.y * blockDim.y + threadIdx.y + 1;
  int k = blockIdx.x * blockDim.x + threadIdx.x + 1;
  if (j >= ny - 1 || k >= nz - 1)
    return;
  const int xLoSrc = 1 + offset;
  const int xHiSrc = nx - 2 - offset;
  // left ghost = right interior, right ghost = left interior
  arr[idx3(0, j, k, ny, nz)] = arr[idx3(xHiSrc, j, k, ny, nz)];
  arr[idx3(nx - 1, j, k, ny, nz)] = arr[idx3(xLoSrc, j, k, ny, nz)];
}

__global__ void gpuSelfCopyFaceY(cudaSolverType* __restrict__ arr, int nx,
                                 int ny, int nz, int offset) {
  int i = blockIdx.y * blockDim.y + threadIdx.y + 1;
  int k = blockIdx.x * blockDim.x + threadIdx.x + 1;
  if (i >= nx - 1 || k >= nz - 1)
    return;
  const int yLoSrc = 1 + offset;
  const int yHiSrc = ny - 2 - offset;
  arr[idx3(i, 0, k, ny, nz)] = arr[idx3(i, yHiSrc, k, ny, nz)];
  arr[idx3(i, ny - 1, k, ny, nz)] = arr[idx3(i, yLoSrc, k, ny, nz)];
}

__global__ void gpuSelfCopyFaceZ(cudaSolverType* __restrict__ arr, int nx,
                                 int ny, int nz, int offset) {
  int i = blockIdx.y * blockDim.y + threadIdx.y + 1;
  int j = blockIdx.x * blockDim.x + threadIdx.x + 1;
  if (i >= nx - 1 || j >= ny - 1)
    return;
  const int zLoSrc = 1 + offset;
  const int zHiSrc = nz - 2 - offset;
  arr[idx3(i, j, 0, ny, nz)] = arr[idx3(i, j, zHiSrc, ny, nz)];
  arr[idx3(i, j, nz - 1, ny, nz)] = arr[idx3(i, j, zLoSrc, ny, nz)];
}

// =========================================================================
//  Self-copy kernels – periodic edge swap when rank == neighbour (not used)
// =========================================================================

__global__ void gpuSelfCopyEdgeX(cudaSolverType* __restrict__ arr, int nx,
                                 int ny, int nz, int offset, bool hasZright,
                                 bool hasZleft, bool hasYright, bool hasYleft) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  const int xLoSrc = 1 + offset;
  const int xHiSrc = nx - 2 - offset;
  // Along Y (1..ny-2) for Z-neighbour edges, along Z (1..nz-2) for Y-neighbour
  // edges
  if (hasZright) {
    int iy = tid + 1;
    if (iy < ny - 1) {
      arr[idx3(0, iy, nz - 1, ny, nz)] = arr[idx3(xHiSrc, iy, nz - 1, ny, nz)];
      arr[idx3(nx - 1, iy, nz - 1, ny, nz)] =
          arr[idx3(xLoSrc, iy, nz - 1, ny, nz)];
    }
  }
  if (hasZleft) {
    int iy = tid + 1;
    if (iy < ny - 1) {
      arr[idx3(0, iy, 0, ny, nz)] = arr[idx3(xHiSrc, iy, 0, ny, nz)];
      arr[idx3(nx - 1, iy, 0, ny, nz)] = arr[idx3(xLoSrc, iy, 0, ny, nz)];
    }
  }
  if (hasYright) {
    int iz = tid + 1;
    if (iz < nz - 1) {
      arr[idx3(0, ny - 1, iz, ny, nz)] = arr[idx3(xHiSrc, ny - 1, iz, ny, nz)];
      arr[idx3(nx - 1, ny - 1, iz, ny, nz)] =
          arr[idx3(xLoSrc, ny - 1, iz, ny, nz)];
    }
  }
  if (hasYleft) {
    int iz = tid + 1;
    if (iz < nz - 1) {
      arr[idx3(0, 0, iz, ny, nz)] = arr[idx3(xHiSrc, 0, iz, ny, nz)];
      arr[idx3(nx - 1, 0, iz, ny, nz)] = arr[idx3(xLoSrc, 0, iz, ny, nz)];
    }
  }
}

__global__ void gpuSelfCopyEdgeY(cudaSolverType* __restrict__ arr, int nx,
                                 int ny, int nz, int offset, bool hasXright,
                                 bool hasXleft, bool hasZright, bool hasZleft) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  const int yLoSrc = 1 + offset;
  const int yHiSrc = ny - 2 - offset;
  if (hasXright) {
    int iz = tid + 1;
    if (iz < nz - 1) {
      arr[idx3(nx - 1, 0, iz, ny, nz)] = arr[idx3(nx - 1, yHiSrc, iz, ny, nz)];
      arr[idx3(nx - 1, ny - 1, iz, ny, nz)] =
          arr[idx3(nx - 1, yLoSrc, iz, ny, nz)];
    }
  }
  if (hasXleft) {
    int iz = tid + 1;
    if (iz < nz - 1) {
      arr[idx3(0, 0, iz, ny, nz)] = arr[idx3(0, yHiSrc, iz, ny, nz)];
      arr[idx3(0, ny - 1, iz, ny, nz)] = arr[idx3(0, yLoSrc, iz, ny, nz)];
    }
  }
  if (hasZright) {
    int ix = tid + 1;
    if (ix < nx - 1) {
      arr[idx3(ix, 0, nz - 1, ny, nz)] = arr[idx3(ix, yHiSrc, nz - 1, ny, nz)];
      arr[idx3(ix, ny - 1, nz - 1, ny, nz)] =
          arr[idx3(ix, yLoSrc, nz - 1, ny, nz)];
    }
  }
  if (hasZleft) {
    int ix = tid + 1;
    if (ix < nx - 1) {
      arr[idx3(ix, 0, 0, ny, nz)] = arr[idx3(ix, yHiSrc, 0, ny, nz)];
      arr[idx3(ix, ny - 1, 0, ny, nz)] = arr[idx3(ix, yLoSrc, 0, ny, nz)];
    }
  }
}

__global__ void gpuSelfCopyEdgeZ(cudaSolverType* __restrict__ arr, int nx,
                                 int ny, int nz, int offset, bool hasYright,
                                 bool hasYleft, bool hasXright, bool hasXleft) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  const int zLoSrc = 1 + offset;
  const int zHiSrc = nz - 2 - offset;
  if (hasYright) {
    int ix = tid + 1;
    if (ix < nx - 1) {
      arr[idx3(ix, ny - 1, 0, ny, nz)] = arr[idx3(ix, ny - 1, zHiSrc, ny, nz)];
      arr[idx3(ix, ny - 1, nz - 1, ny, nz)] =
          arr[idx3(ix, ny - 1, zLoSrc, ny, nz)];
    }
  }
  if (hasYleft) {
    int ix = tid + 1;
    if (ix < nx - 1) {
      arr[idx3(ix, 0, 0, ny, nz)] = arr[idx3(ix, 0, zHiSrc, ny, nz)];
      arr[idx3(ix, 0, nz - 1, ny, nz)] = arr[idx3(ix, 0, zLoSrc, ny, nz)];
    }
  }
  if (hasXright) {
    int iy = tid + 1;
    if (iy < ny - 1) {
      arr[idx3(nx - 1, iy, 0, ny, nz)] = arr[idx3(nx - 1, iy, zHiSrc, ny, nz)];
      arr[idx3(nx - 1, iy, nz - 1, ny, nz)] =
          arr[idx3(nx - 1, iy, zLoSrc, ny, nz)];
    }
  }
  if (hasXleft) {
    int iy = tid + 1;
    if (iy < ny - 1) {
      arr[idx3(0, iy, 0, ny, nz)] = arr[idx3(0, iy, zHiSrc, ny, nz)];
      arr[idx3(0, iy, nz - 1, ny, nz)] = arr[idx3(0, iy, zLoSrc, ny, nz)];
    }
  }
}

// =========================================================================
//  Self-copy kernels – periodic corner swap (not used)
// =========================================================================

__global__ void gpuSelfCopyCornerX(cudaSolverType* __restrict__ arr, int nx,
                                   int ny, int nz, int offset, bool hasYleft,
                                   bool hasYright, bool hasZleft,
                                   bool hasZright) {
  // Single-thread kernel (only 4 elements at most)
  if (threadIdx.x != 0)
    return;
  const int xLoSrc = 1 + offset;
  const int xHiSrc = nx - 2 - offset;
  if (hasYleft && hasZleft) {
    arr[idx3(0, 0, 0, ny, nz)] = arr[idx3(xHiSrc, 0, 0, ny, nz)];
    arr[idx3(nx - 1, 0, 0, ny, nz)] = arr[idx3(xLoSrc, 0, 0, ny, nz)];
  }
  if (hasYleft && hasZright) {
    arr[idx3(0, 0, nz - 1, ny, nz)] = arr[idx3(xHiSrc, 0, nz - 1, ny, nz)];
    arr[idx3(nx - 1, 0, nz - 1, ny, nz)] = arr[idx3(xLoSrc, 0, nz - 1, ny, nz)];
  }
  if (hasYright && hasZleft) {
    arr[idx3(0, ny - 1, 0, ny, nz)] = arr[idx3(xHiSrc, ny - 1, 0, ny, nz)];
    arr[idx3(nx - 1, ny - 1, 0, ny, nz)] = arr[idx3(xLoSrc, ny - 1, 0, ny, nz)];
  }
  if (hasYright && hasZright) {
    arr[idx3(0, ny - 1, nz - 1, ny, nz)] =
        arr[idx3(xHiSrc, ny - 1, nz - 1, ny, nz)];
    arr[idx3(nx - 1, ny - 1, nz - 1, ny, nz)] =
        arr[idx3(xLoSrc, ny - 1, nz - 1, ny, nz)];
  }
}

__global__ void gpuSelfCopyCornerY(cudaSolverType* __restrict__ arr, int nx,
                                   int ny, int nz, int offset, bool hasXleft,
                                   bool hasXright, bool hasZleft,
                                   bool hasZright) {
  if (threadIdx.x != 0)
    return;
  const int yLoSrc = 1 + offset;
  const int yHiSrc = ny - 2 - offset;
  if (hasXleft && hasZleft) {
    arr[idx3(0, 0, 0, ny, nz)] = arr[idx3(0, yHiSrc, 0, ny, nz)];
    arr[idx3(0, ny - 1, 0, ny, nz)] = arr[idx3(0, yLoSrc, 0, ny, nz)];
  }
  if (hasXleft && hasZright) {
    arr[idx3(0, 0, nz - 1, ny, nz)] = arr[idx3(0, yHiSrc, nz - 1, ny, nz)];
    arr[idx3(0, ny - 1, nz - 1, ny, nz)] = arr[idx3(0, yLoSrc, nz - 1, ny, nz)];
  }
  if (hasXright && hasZleft) {
    arr[idx3(nx - 1, 0, 0, ny, nz)] = arr[idx3(nx - 1, yHiSrc, 0, ny, nz)];
    arr[idx3(nx - 1, ny - 1, 0, ny, nz)] = arr[idx3(nx - 1, yLoSrc, 0, ny, nz)];
  }
  if (hasXright && hasZright) {
    arr[idx3(nx - 1, 0, nz - 1, ny, nz)] =
        arr[idx3(nx - 1, yHiSrc, nz - 1, ny, nz)];
    arr[idx3(nx - 1, ny - 1, nz - 1, ny, nz)] =
        arr[idx3(nx - 1, yLoSrc, nz - 1, ny, nz)];
  }
}

__global__ void gpuSelfCopyCornerZ(cudaSolverType* __restrict__ arr, int nx,
                                   int ny, int nz, int offset, bool hasYleft,
                                   bool hasYright, bool hasXleft,
                                   bool hasXright) {
  if (threadIdx.x != 0)
    return;
  const int zLoSrc = 1 + offset;
  const int zHiSrc = nz - 2 - offset;
  if (hasYleft && hasXleft) {
    arr[idx3(0, 0, 0, ny, nz)] = arr[idx3(0, 0, zHiSrc, ny, nz)];
    arr[idx3(0, 0, nz - 1, ny, nz)] = arr[idx3(0, 0, zLoSrc, ny, nz)];
  }
  if (hasYleft && hasXright) {
    arr[idx3(nx - 1, 0, 0, ny, nz)] = arr[idx3(nx - 1, 0, zHiSrc, ny, nz)];
    arr[idx3(nx - 1, 0, nz - 1, ny, nz)] = arr[idx3(nx - 1, 0, zLoSrc, ny, nz)];
  }
  if (hasYright && hasXleft) {
    arr[idx3(0, ny - 1, 0, ny, nz)] = arr[idx3(0, ny - 1, zHiSrc, ny, nz)];
    arr[idx3(0, ny - 1, nz - 1, ny, nz)] = arr[idx3(0, ny - 1, zLoSrc, ny, nz)];
  }
  if (hasYright && hasXright) {
    arr[idx3(nx - 1, ny - 1, 0, ny, nz)] =
        arr[idx3(nx - 1, ny - 1, zHiSrc, ny, nz)];
    arr[idx3(nx - 1, ny - 1, nz - 1, ny, nz)] =
        arr[idx3(nx - 1, ny - 1, zLoSrc, ny, nz)];
  }
}

// =========================================================================
//  Batched self-copy kernels – face (uses blockIdx.z for field index; the
//  contiguous index of the plane on threadIdx.x)
// =========================================================================

__global__ void
gpuBatchSelfCopyFaceX(cudaSolverType* const* __restrict__ fields, int nx,
                      int ny, int nz, int offset) {
  int j = blockIdx.y * blockDim.y + threadIdx.y + 1;
  int k = blockIdx.x * blockDim.x + threadIdx.x + 1;
  if (j >= ny - 1 || k >= nz - 1)
    return;
  cudaSolverType* arr = fields[blockIdx.z];
  const int xLoSrc = 1 + offset;
  const int xHiSrc = nx - 2 - offset;
  arr[idx3(0, j, k, ny, nz)] = arr[idx3(xHiSrc, j, k, ny, nz)];
  arr[idx3(nx - 1, j, k, ny, nz)] = arr[idx3(xLoSrc, j, k, ny, nz)];
}

__global__ void
gpuBatchSelfCopyFaceY(cudaSolverType* const* __restrict__ fields, int nx,
                      int ny, int nz, int offset) {
  int i = blockIdx.y * blockDim.y + threadIdx.y + 1;
  int k = blockIdx.x * blockDim.x + threadIdx.x + 1;
  if (i >= nx - 1 || k >= nz - 1)
    return;
  cudaSolverType* arr = fields[blockIdx.z];
  const int yLoSrc = 1 + offset;
  const int yHiSrc = ny - 2 - offset;
  arr[idx3(i, 0, k, ny, nz)] = arr[idx3(i, yHiSrc, k, ny, nz)];
  arr[idx3(i, ny - 1, k, ny, nz)] = arr[idx3(i, yLoSrc, k, ny, nz)];
}

__global__ void
gpuBatchSelfCopyFaceZ(cudaSolverType* const* __restrict__ fields, int nx,
                      int ny, int nz, int offset) {
  int i = blockIdx.y * blockDim.y + threadIdx.y + 1;
  int j = blockIdx.x * blockDim.x + threadIdx.x + 1;
  if (i >= nx - 1 || j >= ny - 1)
    return;
  cudaSolverType* arr = fields[blockIdx.z];
  const int zLoSrc = 1 + offset;
  const int zHiSrc = nz - 2 - offset;
  arr[idx3(i, j, 0, ny, nz)] = arr[idx3(i, j, zHiSrc, ny, nz)];
  arr[idx3(i, j, nz - 1, ny, nz)] = arr[idx3(i, j, zLoSrc, ny, nz)];
}

// =========================================================================
//  Batched self-copy kernels – edge (uses blockIdx.y for field index)
// =========================================================================

__global__ void
gpuBatchSelfCopyEdgeX(cudaSolverType* const* __restrict__ fields, int nx,
                      int ny, int nz, int offset, bool hasZright, bool hasZleft,
                      bool hasYright, bool hasYleft) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  cudaSolverType* arr = fields[blockIdx.y];
  const int xLoSrc = 1 + offset;
  const int xHiSrc = nx - 2 - offset;
  if (hasZright) {
    int iy = tid + 1;
    if (iy < ny - 1) {
      arr[idx3(0, iy, nz - 1, ny, nz)] = arr[idx3(xHiSrc, iy, nz - 1, ny, nz)];
      arr[idx3(nx - 1, iy, nz - 1, ny, nz)] =
          arr[idx3(xLoSrc, iy, nz - 1, ny, nz)];
    }
  }
  if (hasZleft) {
    int iy = tid + 1;
    if (iy < ny - 1) {
      arr[idx3(0, iy, 0, ny, nz)] = arr[idx3(xHiSrc, iy, 0, ny, nz)];
      arr[idx3(nx - 1, iy, 0, ny, nz)] = arr[idx3(xLoSrc, iy, 0, ny, nz)];
    }
  }
  if (hasYright) {
    int iz = tid + 1;
    if (iz < nz - 1) {
      arr[idx3(0, ny - 1, iz, ny, nz)] = arr[idx3(xHiSrc, ny - 1, iz, ny, nz)];
      arr[idx3(nx - 1, ny - 1, iz, ny, nz)] =
          arr[idx3(xLoSrc, ny - 1, iz, ny, nz)];
    }
  }
  if (hasYleft) {
    int iz = tid + 1;
    if (iz < nz - 1) {
      arr[idx3(0, 0, iz, ny, nz)] = arr[idx3(xHiSrc, 0, iz, ny, nz)];
      arr[idx3(nx - 1, 0, iz, ny, nz)] = arr[idx3(xLoSrc, 0, iz, ny, nz)];
    }
  }
}

__global__ void
gpuBatchSelfCopyEdgeY(cudaSolverType* const* __restrict__ fields, int nx,
                      int ny, int nz, int offset, bool hasXright, bool hasXleft,
                      bool hasZright, bool hasZleft) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  cudaSolverType* arr = fields[blockIdx.y];
  const int yLoSrc = 1 + offset;
  const int yHiSrc = ny - 2 - offset;
  if (hasXright) {
    int iz = tid + 1;
    if (iz < nz - 1) {
      arr[idx3(nx - 1, 0, iz, ny, nz)] = arr[idx3(nx - 1, yHiSrc, iz, ny, nz)];
      arr[idx3(nx - 1, ny - 1, iz, ny, nz)] =
          arr[idx3(nx - 1, yLoSrc, iz, ny, nz)];
    }
  }
  if (hasXleft) {
    int iz = tid + 1;
    if (iz < nz - 1) {
      arr[idx3(0, 0, iz, ny, nz)] = arr[idx3(0, yHiSrc, iz, ny, nz)];
      arr[idx3(0, ny - 1, iz, ny, nz)] = arr[idx3(0, yLoSrc, iz, ny, nz)];
    }
  }
  if (hasZright) {
    int ix = tid + 1;
    if (ix < nx - 1) {
      arr[idx3(ix, 0, nz - 1, ny, nz)] = arr[idx3(ix, yHiSrc, nz - 1, ny, nz)];
      arr[idx3(ix, ny - 1, nz - 1, ny, nz)] =
          arr[idx3(ix, yLoSrc, nz - 1, ny, nz)];
    }
  }
  if (hasZleft) {
    int ix = tid + 1;
    if (ix < nx - 1) {
      arr[idx3(ix, 0, 0, ny, nz)] = arr[idx3(ix, yHiSrc, 0, ny, nz)];
      arr[idx3(ix, ny - 1, 0, ny, nz)] = arr[idx3(ix, yLoSrc, 0, ny, nz)];
    }
  }
}

__global__ void
gpuBatchSelfCopyEdgeZ(cudaSolverType* const* __restrict__ fields, int nx,
                      int ny, int nz, int offset, bool hasYright, bool hasYleft,
                      bool hasXright, bool hasXleft) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  cudaSolverType* arr = fields[blockIdx.y];
  const int zLoSrc = 1 + offset;
  const int zHiSrc = nz - 2 - offset;
  if (hasYright) {
    int ix = tid + 1;
    if (ix < nx - 1) {
      arr[idx3(ix, ny - 1, 0, ny, nz)] = arr[idx3(ix, ny - 1, zHiSrc, ny, nz)];
      arr[idx3(ix, ny - 1, nz - 1, ny, nz)] =
          arr[idx3(ix, ny - 1, zLoSrc, ny, nz)];
    }
  }
  if (hasYleft) {
    int ix = tid + 1;
    if (ix < nx - 1) {
      arr[idx3(ix, 0, 0, ny, nz)] = arr[idx3(ix, 0, zHiSrc, ny, nz)];
      arr[idx3(ix, 0, nz - 1, ny, nz)] = arr[idx3(ix, 0, zLoSrc, ny, nz)];
    }
  }
  if (hasXright) {
    int iy = tid + 1;
    if (iy < ny - 1) {
      arr[idx3(nx - 1, iy, 0, ny, nz)] = arr[idx3(nx - 1, iy, zHiSrc, ny, nz)];
      arr[idx3(nx - 1, iy, nz - 1, ny, nz)] =
          arr[idx3(nx - 1, iy, zLoSrc, ny, nz)];
    }
  }
  if (hasXleft) {
    int iy = tid + 1;
    if (iy < ny - 1) {
      arr[idx3(0, iy, 0, ny, nz)] = arr[idx3(0, iy, zHiSrc, ny, nz)];
      arr[idx3(0, iy, nz - 1, ny, nz)] = arr[idx3(0, iy, zLoSrc, ny, nz)];
    }
  }
}

// =========================================================================
//  Batched self-copy kernels – corner (uses blockIdx.x for field index)
// =========================================================================

__global__ void
gpuBatchSelfCopyCornerX(cudaSolverType* const* __restrict__ fields, int nx,
                        int ny, int nz, int offset, bool hasYleft,
                        bool hasYright, bool hasZleft, bool hasZright) {
  cudaSolverType* arr = fields[blockIdx.x];
  if (threadIdx.x != 0)
    return;
  const int xLoSrc = 1 + offset;
  const int xHiSrc = nx - 2 - offset;
  if (hasYleft && hasZleft) {
    arr[idx3(0, 0, 0, ny, nz)] = arr[idx3(xHiSrc, 0, 0, ny, nz)];
    arr[idx3(nx - 1, 0, 0, ny, nz)] = arr[idx3(xLoSrc, 0, 0, ny, nz)];
  }
  if (hasYleft && hasZright) {
    arr[idx3(0, 0, nz - 1, ny, nz)] = arr[idx3(xHiSrc, 0, nz - 1, ny, nz)];
    arr[idx3(nx - 1, 0, nz - 1, ny, nz)] = arr[idx3(xLoSrc, 0, nz - 1, ny, nz)];
  }
  if (hasYright && hasZleft) {
    arr[idx3(0, ny - 1, 0, ny, nz)] = arr[idx3(xHiSrc, ny - 1, 0, ny, nz)];
    arr[idx3(nx - 1, ny - 1, 0, ny, nz)] = arr[idx3(xLoSrc, ny - 1, 0, ny, nz)];
  }
  if (hasYright && hasZright) {
    arr[idx3(0, ny - 1, nz - 1, ny, nz)] =
        arr[idx3(xHiSrc, ny - 1, nz - 1, ny, nz)];
    arr[idx3(nx - 1, ny - 1, nz - 1, ny, nz)] =
        arr[idx3(xLoSrc, ny - 1, nz - 1, ny, nz)];
  }
}

__global__ void
gpuBatchSelfCopyCornerY(cudaSolverType* const* __restrict__ fields, int nx,
                        int ny, int nz, int offset, bool hasXleft,
                        bool hasXright, bool hasZleft, bool hasZright) {
  cudaSolverType* arr = fields[blockIdx.x];
  if (threadIdx.x != 0)
    return;
  const int yLoSrc = 1 + offset;
  const int yHiSrc = ny - 2 - offset;
  if (hasXleft && hasZleft) {
    arr[idx3(0, 0, 0, ny, nz)] = arr[idx3(0, yHiSrc, 0, ny, nz)];
    arr[idx3(0, ny - 1, 0, ny, nz)] = arr[idx3(0, yLoSrc, 0, ny, nz)];
  }
  if (hasXleft && hasZright) {
    arr[idx3(0, 0, nz - 1, ny, nz)] = arr[idx3(0, yHiSrc, nz - 1, ny, nz)];
    arr[idx3(0, ny - 1, nz - 1, ny, nz)] = arr[idx3(0, yLoSrc, nz - 1, ny, nz)];
  }
  if (hasXright && hasZleft) {
    arr[idx3(nx - 1, 0, 0, ny, nz)] = arr[idx3(nx - 1, yHiSrc, 0, ny, nz)];
    arr[idx3(nx - 1, ny - 1, 0, ny, nz)] = arr[idx3(nx - 1, yLoSrc, 0, ny, nz)];
  }
  if (hasXright && hasZright) {
    arr[idx3(nx - 1, 0, nz - 1, ny, nz)] =
        arr[idx3(nx - 1, yHiSrc, nz - 1, ny, nz)];
    arr[idx3(nx - 1, ny - 1, nz - 1, ny, nz)] =
        arr[idx3(nx - 1, yLoSrc, nz - 1, ny, nz)];
  }
}

__global__ void
gpuBatchSelfCopyCornerZ(cudaSolverType* const* __restrict__ fields, int nx,
                        int ny, int nz, int offset, bool hasYleft,
                        bool hasYright, bool hasXleft, bool hasXright) {
  cudaSolverType* arr = fields[blockIdx.x];
  if (threadIdx.x != 0)
    return;
  const int zLoSrc = 1 + offset;
  const int zHiSrc = nz - 2 - offset;
  if (hasYleft && hasXleft) {
    arr[idx3(0, 0, 0, ny, nz)] = arr[idx3(0, 0, zHiSrc, ny, nz)];
    arr[idx3(0, 0, nz - 1, ny, nz)] = arr[idx3(0, 0, zLoSrc, ny, nz)];
  }
  if (hasYleft && hasXright) {
    arr[idx3(nx - 1, 0, 0, ny, nz)] = arr[idx3(nx - 1, 0, zHiSrc, ny, nz)];
    arr[idx3(nx - 1, 0, nz - 1, ny, nz)] = arr[idx3(nx - 1, 0, zLoSrc, ny, nz)];
  }
  if (hasYright && hasXleft) {
    arr[idx3(0, ny - 1, 0, ny, nz)] = arr[idx3(0, ny - 1, zHiSrc, ny, nz)];
    arr[idx3(0, ny - 1, nz - 1, ny, nz)] = arr[idx3(0, ny - 1, zLoSrc, ny, nz)];
  }
  if (hasYright && hasXright) {
    arr[idx3(nx - 1, ny - 1, 0, ny, nz)] =
        arr[idx3(nx - 1, ny - 1, zHiSrc, ny, nz)];
    arr[idx3(nx - 1, ny - 1, nz - 1, ny, nz)] =
        arr[idx3(nx - 1, ny - 1, zLoSrc, ny, nz)];
  }
}

// =========================================================================
//  Additive interpolation kernels (for communicateInterp; not used)
// =========================================================================

__global__ void gpuAddFaceX(cudaSolverType* __restrict__ arr, int nx, int ny,
                            int nz, bool hasXright, bool hasXleft) {
  int j = blockIdx.y * blockDim.y + threadIdx.y + 1;
  int k = blockIdx.x * blockDim.x + threadIdx.x + 1;
  if (j > ny - 2 || k > nz - 2)
    return;
  if (hasXright)
    arr[idx3(nx - 2, j, k, ny, nz)] += arr[idx3(nx - 1, j, k, ny, nz)];
  if (hasXleft)
    arr[idx3(1, j, k, ny, nz)] += arr[idx3(0, j, k, ny, nz)];
}

__global__ void gpuAddFaceY(cudaSolverType* __restrict__ arr, int nx, int ny,
                            int nz, bool hasYright, bool hasYleft) {
  int i = blockIdx.y * blockDim.y + threadIdx.y + 1;
  int k = blockIdx.x * blockDim.x + threadIdx.x + 1;
  if (i > nx - 2 || k > nz - 2)
    return;
  if (hasYright)
    arr[idx3(i, ny - 2, k, ny, nz)] += arr[idx3(i, ny - 1, k, ny, nz)];
  if (hasYleft)
    arr[idx3(i, 1, k, ny, nz)] += arr[idx3(i, 0, k, ny, nz)];
}

__global__ void gpuAddFaceZ(cudaSolverType* __restrict__ arr, int nx, int ny,
                            int nz, bool hasZright, bool hasZleft) {
  int i = blockIdx.y * blockDim.y + threadIdx.y + 1;
  int j = blockIdx.x * blockDim.x + threadIdx.x + 1;
  if (i > nx - 2 || j > ny - 2)
    return;
  if (hasZright)
    arr[idx3(i, j, nz - 2, ny, nz)] += arr[idx3(i, j, nz - 1, ny, nz)];
  if (hasZleft)
    arr[idx3(i, j, 1, ny, nz)] += arr[idx3(i, j, 0, ny, nz)];
}

__global__ void gpuAddEdgeZ(cudaSolverType* __restrict__ arr, int nx, int ny,
                            int nz, bool hasXright, bool hasXleft,
                            bool hasYright, bool hasYleft) {
  int k = blockIdx.x * blockDim.x + threadIdx.x + 1;
  if (k > nz - 2)
    return;
  if (hasXright && hasYright)
    arr[idx3(nx - 2, ny - 2, k, ny, nz)] +=
        arr[idx3(nx - 1, ny - 1, k, ny, nz)];
  if (hasXleft && hasYleft)
    arr[idx3(1, 1, k, ny, nz)] += arr[idx3(0, 0, k, ny, nz)];
  if (hasXright && hasYleft)
    arr[idx3(nx - 2, 1, k, ny, nz)] += arr[idx3(nx - 1, 0, k, ny, nz)];
  if (hasXleft && hasYright)
    arr[idx3(1, ny - 2, k, ny, nz)] += arr[idx3(0, ny - 1, k, ny, nz)];
}

__global__ void gpuAddEdgeY(cudaSolverType* __restrict__ arr, int nx, int ny,
                            int nz, bool hasXright, bool hasXleft,
                            bool hasZright, bool hasZleft) {
  int j = blockIdx.x * blockDim.x + threadIdx.x + 1;
  if (j > ny - 2)
    return;
  if (hasXright && hasZright)
    arr[idx3(nx - 2, j, nz - 2, ny, nz)] +=
        arr[idx3(nx - 1, j, nz - 1, ny, nz)];
  if (hasXleft && hasZleft)
    arr[idx3(1, j, 1, ny, nz)] += arr[idx3(0, j, 0, ny, nz)];
  if (hasXleft && hasZright)
    arr[idx3(1, j, nz - 2, ny, nz)] += arr[idx3(0, j, nz - 1, ny, nz)];
  if (hasXright && hasZleft)
    arr[idx3(nx - 2, j, 1, ny, nz)] += arr[idx3(nx - 1, j, 0, ny, nz)];
}

__global__ void gpuAddEdgeX(cudaSolverType* __restrict__ arr, int nx, int ny,
                            int nz, bool hasYright, bool hasYleft,
                            bool hasZright, bool hasZleft) {
  int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
  if (i > nx - 2)
    return;
  if (hasYright && hasZright)
    arr[idx3(i, ny - 2, nz - 2, ny, nz)] +=
        arr[idx3(i, ny - 1, nz - 1, ny, nz)];
  if (hasYleft && hasZleft)
    arr[idx3(i, 1, 1, ny, nz)] += arr[idx3(i, 0, 0, ny, nz)];
  if (hasYleft && hasZright)
    arr[idx3(i, 1, nz - 2, ny, nz)] += arr[idx3(i, 0, nz - 1, ny, nz)];
  if (hasYright && hasZleft)
    arr[idx3(i, ny - 2, 1, ny, nz)] += arr[idx3(i, ny - 1, 0, ny, nz)];
}

__global__ void gpuAddCorner(cudaSolverType* __restrict__ arr, int nx, int ny,
                             int nz, bool hasXright, bool hasXleft,
                             bool hasYright, bool hasYleft, bool hasZright,
                             bool hasZleft) {
  if (threadIdx.x != 0)
    return;
  if (hasXright && hasYright && hasZright)
    arr[idx3(nx - 2, ny - 2, nz - 2, ny, nz)] +=
        arr[idx3(nx - 1, ny - 1, nz - 1, ny, nz)];
  if (hasXleft && hasYright && hasZright)
    arr[idx3(1, ny - 2, nz - 2, ny, nz)] +=
        arr[idx3(0, ny - 1, nz - 1, ny, nz)];
  if (hasXright && hasYleft && hasZright)
    arr[idx3(nx - 2, 1, nz - 2, ny, nz)] +=
        arr[idx3(nx - 1, 0, nz - 1, ny, nz)];
  if (hasXleft && hasYleft && hasZright)
    arr[idx3(1, 1, nz - 2, ny, nz)] += arr[idx3(0, 0, nz - 1, ny, nz)];
  if (hasXright && hasYright && hasZleft)
    arr[idx3(nx - 2, ny - 2, 1, ny, nz)] +=
        arr[idx3(nx - 1, ny - 1, 0, ny, nz)];
  if (hasXleft && hasYright && hasZleft)
    arr[idx3(1, ny - 2, 1, ny, nz)] += arr[idx3(0, ny - 1, 0, ny, nz)];
  if (hasXright && hasYleft && hasZleft)
    arr[idx3(nx - 2, 1, 1, ny, nz)] += arr[idx3(nx - 1, 0, 0, ny, nz)];
  if (hasXleft && hasYleft && hasZleft)
    arr[idx3(1, 1, 1, ny, nz)] += arr[idx3(0, 0, 0, ny, nz)];
}

// =========================================================================
//  gpuBCface – apply face boundary conditions on GPU array
// =========================================================================

/** Threads per block of gpuBCfacePlane.  A face kernel is latency-bound;
 *  small blocks spread a plane over more SMs.  On 2D and 3D faces this
 *  measured as fast as 16x16 2D blocks, and faster than 256-thread ones. */
static constexpr int kBCBlock = 64;

/** The face rules bcType on the faces with physical[f], f = Xleft, Xright,
 *  Yleft, Yright, Zleft, Zright: the launch order, which decides the edge
 *  and corner ghosts two faces share. */
static void launchBCfaces(cudaSolverType* d, int nx, int ny, int nz,
                          const bool physical[6], const int bcType[6],
                          cudaStream_t stream) {
  const int n[3] = {nx, ny, nz};
  const int stride[3] = {ny * nz, nz, 1};
  for (int f = 0; f < 6; f++) {
    if (!physical[f] || bcType[f] == BC_FACE_KEEP)
      continue;
    const int dir = f / 2;
    const bool right = f % 2 == 1;
    // in-plane axes: the slow one, then the fast one (k, or j on a z face)
    const int slow = dir == 0 ? 1 : 0, fast = dir == 2 ? 1 : 2;
    const int points = n[slow] * n[fast];
    gpuBCfacePlane<<<(points + kBCBlock - 1) / kBCBlock, kBCBlock, 0, stream>>>(
        d, right ? (n[dir] - 1) * stride[dir] : 0, n[fast], stride[slow],
        stride[fast], points, right ? -stride[dir] : stride[dir], bcType[f]);
  }
}

void gpuBCface(int nx, int ny, int nz, GPUFieldArray3& gpuArr, int bcFaceXright,
               int bcFaceXleft, int bcFaceYright, int bcFaceYleft,
               int bcFaceZright, int bcFaceZleft, const VirtualTopology3D* vct,
               cudaStream_t stream) {
  const bool physical[6] = {vct->getXleft_neighbor() == MPI_PROC_NULL,
                            vct->getXright_neighbor() == MPI_PROC_NULL,
                            vct->getYleft_neighbor() == MPI_PROC_NULL,
                            vct->getYright_neighbor() == MPI_PROC_NULL,
                            vct->getZleft_neighbor() == MPI_PROC_NULL,
                            vct->getZright_neighbor() == MPI_PROC_NULL};
  const int bcType[6] = {bcFaceXleft,  bcFaceXright, bcFaceYleft,
                         bcFaceYright, bcFaceZleft,  bcFaceZright};
  launchBCfaces(gpuArr.devPtr(), nx, ny, nz, physical, bcType, stream);
}

void gpuBCface_P(int nx, int ny, int nz, GPUFieldArray3& gpuArr,
                 int bcFaceXright, int bcFaceXleft, int bcFaceYright,
                 int bcFaceYleft, int bcFaceZright, int bcFaceZleft,
                 const VirtualTopology3D* vct, cudaStream_t stream) {
  // physical faces from the particle-topology neighbours
  const bool physical[6] = {vct->getXleft_neighbor_P() == MPI_PROC_NULL,
                            vct->getXright_neighbor_P() == MPI_PROC_NULL,
                            vct->getYleft_neighbor_P() == MPI_PROC_NULL,
                            vct->getYright_neighbor_P() == MPI_PROC_NULL,
                            vct->getZleft_neighbor_P() == MPI_PROC_NULL,
                            vct->getZright_neighbor_P() == MPI_PROC_NULL};
  const int bcType[6] = {bcFaceXleft,  bcFaceXright, bcFaceYleft,
                         bcFaceYright, bcFaceZleft,  bcFaceZright};
  launchBCfaces(gpuArr.devPtr(), nx, ny, nz, physical, bcType, stream);
}

// =========================================================================
//  EMfields3D GPU halo exchange wrapper methods
//
//  Single-field wrappers route through gpuBatchedHaloExchange (nFields=1).
//  Batched wrappers route through gpuBatchedHaloExchange (nFields=3 or 9).
// =========================================================================

void EMfields3D::gpuCommunicateCenterBC(int nx, int ny, int nz,
                                        GPUFieldArray3& gpuArr,
                                        int bcFaceXright, int bcFaceXleft,
                                        int bcFaceYright, int bcFaceYleft,
                                        int bcFaceZright, int bcFaceZleft) {
  cudaSolverType* ptr = gpuArr.devPtr();
  gpuBatchedHaloExchange(&ptr, 1, nx, ny, nz, true, false, false, false,
                         solverStream_);
  gpuBCface(nx, ny, nz, gpuArr, bcFaceXright, bcFaceXleft, bcFaceYright,
            bcFaceYleft, bcFaceZright, bcFaceZleft, &_vct, solverStream_);
}

void EMfields3D::gpuCommunicateCenterBC_P(int nx, int ny, int nz,
                                          GPUFieldArray3& gpuArr,
                                          int bcFaceXright, int bcFaceXleft,
                                          int bcFaceYright, int bcFaceYleft,
                                          int bcFaceZright, int bcFaceZleft) {
  cudaSolverType* ptr = gpuArr.devPtr();
  gpuBatchedHaloExchange(&ptr, 1, nx, ny, nz, true, false, false, true,
                         solverStream_);
  gpuBCface_P(nx, ny, nz, gpuArr, bcFaceXright, bcFaceXleft, bcFaceYright,
              bcFaceYleft, bcFaceZright, bcFaceZleft, &_vct, solverStream_);
}

void EMfields3D::gpuCommunicateNodeBoxStencilBC_P(
    int nx, int ny, int nz, GPUFieldArray3& gpuArr, int bcFaceXright,
    int bcFaceXleft, int bcFaceYright, int bcFaceYleft, int bcFaceZright,
    int bcFaceZleft) {
  cudaSolverType* ptr = gpuArr.devPtr();
  gpuBatchedHaloExchange(&ptr, 1, nx, ny, nz, false, true, false, true,
                         solverStream_);
  gpuBCface_P(nx, ny, nz, gpuArr, bcFaceXright, bcFaceXleft, bcFaceYright,
              bcFaceYleft, bcFaceZright, bcFaceZleft, &_vct, solverStream_);
}

void EMfields3D::gpuCommunicateNodeBoxStencilBC_P_3(
    int nx, int ny, int nz, GPUFieldArray3& a1, GPUFieldArray3& a2,
    GPUFieldArray3& a3, int bcXR, int bcXL, int bcYR, int bcYL, int bcZR,
    int bcZL) {
  cudaSolverType* ptrs[3] = {a1.devPtr(), a2.devPtr(), a3.devPtr()};
  gpuBatchedHaloExchange(ptrs, 3, nx, ny, nz, false, true, false, true,
                         solverStream_);
  gpuBCface_P(nx, ny, nz, a1, bcXR, bcXL, bcYR, bcYL, bcZR, bcZL, &_vct,
              solverStream_);
  gpuBCface_P(nx, ny, nz, a2, bcXR, bcXL, bcYR, bcYL, bcZR, bcZL, &_vct,
              solverStream_);
  gpuBCface_P(nx, ny, nz, a3, bcXR, bcXL, bcYR, bcYL, bcZR, bcZL, &_vct,
              solverStream_);
}

void EMfields3D::gpuCommunicateCenterBoxStencilBC_P(
    int nx, int ny, int nz, GPUFieldArray3& gpuArr, int bcFaceXright,
    int bcFaceXleft, int bcFaceYright, int bcFaceYleft, int bcFaceZright,
    int bcFaceZleft) {
  cudaSolverType* ptr = gpuArr.devPtr();
  gpuBatchedHaloExchange(&ptr, 1, nx, ny, nz, true, true, false, true,
                         solverStream_);
  gpuBCface_P(nx, ny, nz, gpuArr, bcFaceXright, bcFaceXleft, bcFaceYright,
              bcFaceYleft, bcFaceZright, bcFaceZleft, &_vct, solverStream_);
}

void EMfields3D::gpuCommunicateCenterBoxStencilBC_P_3(
    int nx, int ny, int nz, GPUFieldArray3& a1, GPUFieldArray3& a2,
    GPUFieldArray3& a3, int bcXR, int bcXL, int bcYR, int bcYL, int bcZR,
    int bcZL) {
  cudaSolverType* ptrs[3] = {a1.devPtr(), a2.devPtr(), a3.devPtr()};
  gpuBatchedHaloExchange(ptrs, 3, nx, ny, nz, true, true, false, true,
                         solverStream_);
  gpuBCface_P(nx, ny, nz, a1, bcXR, bcXL, bcYR, bcYL, bcZR, bcZL, &_vct,
              solverStream_);
  gpuBCface_P(nx, ny, nz, a2, bcXR, bcXL, bcYR, bcYL, bcZR, bcZL, &_vct,
              solverStream_);
  gpuBCface_P(nx, ny, nz, a3, bcXR, bcXL, bcYR, bcYL, bcZR, bcZL, &_vct,
              solverStream_);
}

void EMfields3D::gpuCommunicateCenterBoxStencilBC(
    int nx, int ny, int nz, GPUFieldArray3& gpuArr, int bcFaceXright,
    int bcFaceXleft, int bcFaceYright, int bcFaceYleft, int bcFaceZright,
    int bcFaceZleft) {
  cudaSolverType* ptr = gpuArr.devPtr();
  gpuBatchedHaloExchange(&ptr, 1, nx, ny, nz, true, true, false, false,
                         solverStream_);
  gpuBCface(nx, ny, nz, gpuArr, bcFaceXright, bcFaceXleft, bcFaceYright,
            bcFaceYleft, bcFaceZright, bcFaceZleft, &_vct, solverStream_);
}

// =========================================================================
//  Multi-field communicate wrappers: one batched exchange for 3 or 9
//  arrays, then the face BCs of each.
// =========================================================================

void EMfields3D::gpuCommunicateCenterBC_P_3(int nx, int ny, int nz,
                                            GPUFieldArray3& a1,
                                            GPUFieldArray3& a2,
                                            GPUFieldArray3& a3, int bcXR,
                                            int bcXL, int bcYR, int bcYL,
                                            int bcZR, int bcZL) {
  cudaSolverType* ptrs[3] = {a1.devPtr(), a2.devPtr(), a3.devPtr()};
  gpuBatchedHaloExchange(ptrs, 3, nx, ny, nz, true, false, false, true,
                         solverStream_);
  gpuBCface_P(nx, ny, nz, a1, bcXR, bcXL, bcYR, bcYL, bcZR, bcZL, &_vct,
              solverStream_);
  gpuBCface_P(nx, ny, nz, a2, bcXR, bcXL, bcYR, bcYL, bcZR, bcZL, &_vct,
              solverStream_);
  gpuBCface_P(nx, ny, nz, a3, bcXR, bcXL, bcYR, bcYL, bcZR, bcZL, &_vct,
              solverStream_);
}

void EMfields3D::gpuCommunicateCenterBC_3mixed(
    int nx, int ny, int nz, GPUFieldArray3& a1, const int* bc1,
    GPUFieldArray3& a2, const int* bc2, GPUFieldArray3& a3, const int* bc3) {
  cudaSolverType* ptrs[3] = {a1.devPtr(), a2.devPtr(), a3.devPtr()};
  gpuBatchedHaloExchange(ptrs, 3, nx, ny, nz, true, false, false, false,
                         solverStream_);
  gpuBCface(nx, ny, nz, a1, bc1[0], bc1[1], bc1[2], bc1[3], bc1[4], bc1[5],
            &_vct, solverStream_);
  gpuBCface(nx, ny, nz, a2, bc2[0], bc2[1], bc2[2], bc2[3], bc2[4], bc2[5],
            &_vct, solverStream_);
  gpuBCface(nx, ny, nz, a3, bc3[0], bc3[1], bc3[2], bc3[3], bc3[4], bc3[5],
            &_vct, solverStream_);
}

void EMfields3D::gpuCommunicateNodeBC_3mixed(int nx, int ny, int nz,
                                             GPUFieldArray3& a1, const int* bc1,
                                             GPUFieldArray3& a2, const int* bc2,
                                             GPUFieldArray3& a3,
                                             const int* bc3) {
  cudaSolverType* ptrs[3] = {a1.devPtr(), a2.devPtr(), a3.devPtr()};
  gpuBatchedHaloExchange(ptrs, 3, nx, ny, nz, false, false, false, false,
                         solverStream_);
  gpuBCface(nx, ny, nz, a1, bc1[0], bc1[1], bc1[2], bc1[3], bc1[4], bc1[5],
            &_vct, solverStream_);
  gpuBCface(nx, ny, nz, a2, bc2[0], bc2[1], bc2[2], bc2[3], bc2[4], bc2[5],
            &_vct, solverStream_);
  gpuBCface(nx, ny, nz, a3, bc3[0], bc3[1], bc3[2], bc3[3], bc3[4], bc3[5],
            &_vct, solverStream_);
}

void EMfields3D::gpuCommunicateNodeBC_6mixed(int nx, int ny, int nz,
                                             GPUFieldArray3* const a[6],
                                             const int* const bc[6]) {
  cudaSolverType* ptrs[6];
  for (int f = 0; f < 6; f++)
    ptrs[f] = a[f]->devPtr();
  gpuBatchedHaloExchange(ptrs, 6, nx, ny, nz, false, false, false, false,
                         solverStream_);
  for (int f = 0; f < 6; f++)
    gpuBCface(nx, ny, nz, *a[f], bc[f][0], bc[f][1], bc[f][2], bc[f][3],
              bc[f][4], bc[f][5], &_vct, solverStream_);
}

void EMfields3D::gpuCommunicateNodeBoxStencilBC_3mixed(
    int nx, int ny, int nz, GPUFieldArray3& a1, const int* bc1,
    GPUFieldArray3& a2, const int* bc2, GPUFieldArray3& a3, const int* bc3) {
  cudaSolverType* ptrs[3] = {a1.devPtr(), a2.devPtr(), a3.devPtr()};
  gpuBatchedHaloExchange(ptrs, 3, nx, ny, nz, false, true, false, false,
                         solverStream_);
  gpuBCface(nx, ny, nz, a1, bc1[0], bc1[1], bc1[2], bc1[3], bc1[4], bc1[5],
            &_vct, solverStream_);
  gpuBCface(nx, ny, nz, a2, bc2[0], bc2[1], bc2[2], bc2[3], bc2[4], bc2[5],
            &_vct, solverStream_);
  gpuBCface(nx, ny, nz, a3, bc3[0], bc3[1], bc3[2], bc3[3], bc3[4], bc3[5],
            &_vct, solverStream_);
}

void EMfields3D::gpuCommunicateCenterBC_9(
    int nx, int ny, int nz, GPUFieldArray3& a1, GPUFieldArray3& a2,
    GPUFieldArray3& a3, GPUFieldArray3& a4, GPUFieldArray3& a5,
    GPUFieldArray3& a6, GPUFieldArray3& a7, GPUFieldArray3& a8,
    GPUFieldArray3& a9, int bcXR, int bcXL, int bcYR, int bcYL, int bcZR,
    int bcZL) {
  cudaSolverType* ptrs[9] = {a1.devPtr(), a2.devPtr(), a3.devPtr(),
                             a4.devPtr(), a5.devPtr(), a6.devPtr(),
                             a7.devPtr(), a8.devPtr(), a9.devPtr()};
  gpuBatchedHaloExchange(ptrs, 9, nx, ny, nz, true, false, false, false,
                         solverStream_);
  gpuBCface(nx, ny, nz, a1, bcXR, bcXL, bcYR, bcYL, bcZR, bcZL, &_vct,
            solverStream_);
  gpuBCface(nx, ny, nz, a2, bcXR, bcXL, bcYR, bcYL, bcZR, bcZL, &_vct,
            solverStream_);
  gpuBCface(nx, ny, nz, a3, bcXR, bcXL, bcYR, bcYL, bcZR, bcZL, &_vct,
            solverStream_);
  gpuBCface(nx, ny, nz, a4, bcXR, bcXL, bcYR, bcYL, bcZR, bcZL, &_vct,
            solverStream_);
  gpuBCface(nx, ny, nz, a5, bcXR, bcXL, bcYR, bcYL, bcZR, bcZL, &_vct,
            solverStream_);
  gpuBCface(nx, ny, nz, a6, bcXR, bcXL, bcYR, bcYL, bcZR, bcZL, &_vct,
            solverStream_);
  gpuBCface(nx, ny, nz, a7, bcXR, bcXL, bcYR, bcYL, bcZR, bcZL, &_vct,
            solverStream_);
  gpuBCface(nx, ny, nz, a8, bcXR, bcXL, bcYR, bcYL, bcZR, bcZL, &_vct,
            solverStream_);
  gpuBCface(nx, ny, nz, a9, bcXR, bcXL, bcYR, bcYL, bcZR, bcZL, &_vct,
            solverStream_);
}
