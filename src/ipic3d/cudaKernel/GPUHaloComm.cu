/**
 * @file GPUHaloComm.cu
 * @brief Implementation of GPU-aware MPI halo communication and BC kernels.
 *
 * This file mirrors the logic of Com3DNonblk.cpp and BcFields3D.cpp, but
 * operates directly on device pointers (GPUFieldArray3).  MPI derived
 * datatypes from EMfields3D are reused because the contiguous row-major
 * layout on the GPU is identical to the host layout.
 *
 * Compile with -DGPU_SOLVER to activate.
 */

#ifdef GPU_SOLVER

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
//  BC kernels – face boundary condition application
//  bcType  0 = Dirichlet 0 second-order  ghost = -interior
//          1 = Dirichlet 0 first-order   ghost = 0
//          2 = Neumann   0 first-order   ghost = interior
// =========================================================================

__global__ void gpuBCfaceXleft(cudaSolverType* __restrict__ arr, int nx, int ny,
                               int nz, int bcType) {
  int j = blockIdx.x * blockDim.x + threadIdx.x;
  int k = blockIdx.y * blockDim.y + threadIdx.y;
  if (j >= ny || k >= nz)
    return;
  int dst = idx3(0, j, k, ny, nz);
  int src = idx3(1, j, k, ny, nz);
  switch (bcType) {
  case 0:
    arr[dst] = -arr[src];
    break;
  case 1:
    arr[dst] = 0.0;
    break;
  case 2:
    arr[dst] = arr[src];
    break;
  }
}

__global__ void gpuBCfaceXright(cudaSolverType* __restrict__ arr, int nx,
                                int ny, int nz, int bcType) {
  int j = blockIdx.x * blockDim.x + threadIdx.x;
  int k = blockIdx.y * blockDim.y + threadIdx.y;
  if (j >= ny || k >= nz)
    return;
  int dst = idx3(nx - 1, j, k, ny, nz);
  int src = idx3(nx - 2, j, k, ny, nz);
  switch (bcType) {
  case 0:
    arr[dst] = -arr[src];
    break;
  case 1:
    arr[dst] = 0.0;
    break;
  case 2:
    arr[dst] = arr[src];
    break;
  }
}

__global__ void gpuBCfaceYleft(cudaSolverType* __restrict__ arr, int nx, int ny,
                               int nz, int bcType) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int k = blockIdx.y * blockDim.y + threadIdx.y;
  if (i >= nx || k >= nz)
    return;
  int dst = idx3(i, 0, k, ny, nz);
  int src = idx3(i, 1, k, ny, nz);
  switch (bcType) {
  case 0:
    arr[dst] = -arr[src];
    break;
  case 1:
    arr[dst] = 0.0;
    break;
  case 2:
    arr[dst] = arr[src];
    break;
  }
}

__global__ void gpuBCfaceYright(cudaSolverType* __restrict__ arr, int nx,
                                int ny, int nz, int bcType) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int k = blockIdx.y * blockDim.y + threadIdx.y;
  if (i >= nx || k >= nz)
    return;
  int dst = idx3(i, ny - 1, k, ny, nz);
  int src = idx3(i, ny - 2, k, ny, nz);
  switch (bcType) {
  case 0:
    arr[dst] = -arr[src];
    break;
  case 1:
    arr[dst] = 0.0;
    break;
  case 2:
    arr[dst] = arr[src];
    break;
  }
}

__global__ void gpuBCfaceZleft(cudaSolverType* __restrict__ arr, int nx, int ny,
                               int nz, int bcType) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  if (i >= nx || j >= ny)
    return;
  int dst = idx3(i, j, 0, ny, nz);
  int src = idx3(i, j, 1, ny, nz);
  switch (bcType) {
  case 0:
    arr[dst] = -arr[src];
    break;
  case 1:
    arr[dst] = 0.0;
    break;
  case 2:
    arr[dst] = arr[src];
    break;
  }
}

__global__ void gpuBCfaceZright(cudaSolverType* __restrict__ arr, int nx,
                                int ny, int nz, int bcType) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  if (i >= nx || j >= ny)
    return;
  int dst = idx3(i, j, nz - 1, ny, nz);
  int src = idx3(i, j, nz - 2, ny, nz);
  switch (bcType) {
  case 0:
    arr[dst] = -arr[src];
    break;
  case 1:
    arr[dst] = 0.0;
    break;
  case 2:
    arr[dst] = arr[src];
    break;
  }
}

// =========================================================================
//  Self-copy kernels – periodic face swap when rank == neighbour
// =========================================================================

__global__ void gpuSelfCopyFaceX(cudaSolverType* __restrict__ arr, int nx,
                                 int ny, int nz, int offset) {
  int j = blockIdx.x * blockDim.x + threadIdx.x + 1;
  int k = blockIdx.y * blockDim.y + threadIdx.y + 1;
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
  int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
  int k = blockIdx.y * blockDim.y + threadIdx.y + 1;
  if (i >= nx - 1 || k >= nz - 1)
    return;
  const int yLoSrc = 1 + offset;
  const int yHiSrc = ny - 2 - offset;
  arr[idx3(i, 0, k, ny, nz)] = arr[idx3(i, yHiSrc, k, ny, nz)];
  arr[idx3(i, ny - 1, k, ny, nz)] = arr[idx3(i, yLoSrc, k, ny, nz)];
}

__global__ void gpuSelfCopyFaceZ(cudaSolverType* __restrict__ arr, int nx,
                                 int ny, int nz, int offset) {
  int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
  int j = blockIdx.y * blockDim.y + threadIdx.y + 1;
  if (i >= nx - 1 || j >= ny - 1)
    return;
  const int zLoSrc = 1 + offset;
  const int zHiSrc = nz - 2 - offset;
  arr[idx3(i, j, 0, ny, nz)] = arr[idx3(i, j, zHiSrc, ny, nz)];
  arr[idx3(i, j, nz - 1, ny, nz)] = arr[idx3(i, j, zLoSrc, ny, nz)];
}

// =========================================================================
//  Self-copy kernels – periodic edge swap when rank == neighbour
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
//  Self-copy kernels – periodic corner swap
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
//  Batched self-copy kernels – face (uses blockIdx.z for field index)
// =========================================================================

__global__ void
gpuBatchSelfCopyFaceX(cudaSolverType* const* __restrict__ fields, int nx,
                      int ny, int nz, int offset) {
  int j = blockIdx.x * blockDim.x + threadIdx.x + 1;
  int k = blockIdx.y * blockDim.y + threadIdx.y + 1;
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
  int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
  int k = blockIdx.y * blockDim.y + threadIdx.y + 1;
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
  int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
  int j = blockIdx.y * blockDim.y + threadIdx.y + 1;
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
//  Additive interpolation kernels (for communicateInterp)
// =========================================================================

__global__ void gpuAddFaceX(cudaSolverType* __restrict__ arr, int nx, int ny,
                            int nz, bool hasXright, bool hasXleft) {
  int j = blockIdx.x * blockDim.x + threadIdx.x + 1;
  int k = blockIdx.y * blockDim.y + threadIdx.y + 1;
  if (j > ny - 2 || k > nz - 2)
    return;
  if (hasXright)
    arr[idx3(nx - 2, j, k, ny, nz)] += arr[idx3(nx - 1, j, k, ny, nz)];
  if (hasXleft)
    arr[idx3(1, j, k, ny, nz)] += arr[idx3(0, j, k, ny, nz)];
}

__global__ void gpuAddFaceY(cudaSolverType* __restrict__ arr, int nx, int ny,
                            int nz, bool hasYright, bool hasYleft) {
  int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
  int k = blockIdx.y * blockDim.y + threadIdx.y + 1;
  if (i > nx - 2 || k > nz - 2)
    return;
  if (hasYright)
    arr[idx3(i, ny - 2, k, ny, nz)] += arr[idx3(i, ny - 1, k, ny, nz)];
  if (hasYleft)
    arr[idx3(i, 1, k, ny, nz)] += arr[idx3(i, 0, k, ny, nz)];
}

__global__ void gpuAddFaceZ(cudaSolverType* __restrict__ arr, int nx, int ny,
                            int nz, bool hasZright, bool hasZleft) {
  int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
  int j = blockIdx.y * blockDim.y + threadIdx.y + 1;
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

static constexpr int BC_BLOCK = 16;

void gpuBCface(int nx, int ny, int nz, GPUFieldArray3& gpuArr, int bcFaceXright,
               int bcFaceXleft, int bcFaceYright, int bcFaceYleft,
               int bcFaceZright, int bcFaceZleft, const VirtualTopology3D* vct,
               cudaStream_t stream) {
  cudaSolverType* d = gpuArr.devPtr();

  // X boundaries
  if (vct->getXleft_neighbor() == MPI_PROC_NULL) {
    dim3 block(BC_BLOCK, BC_BLOCK);
    dim3 grid((ny + BC_BLOCK - 1) / BC_BLOCK, (nz + BC_BLOCK - 1) / BC_BLOCK);
    gpuBCfaceXleft<<<grid, block, 0, stream>>>(d, nx, ny, nz, bcFaceXleft);
  }
  if (vct->getXright_neighbor() == MPI_PROC_NULL) {
    dim3 block(BC_BLOCK, BC_BLOCK);
    dim3 grid((ny + BC_BLOCK - 1) / BC_BLOCK, (nz + BC_BLOCK - 1) / BC_BLOCK);
    gpuBCfaceXright<<<grid, block, 0, stream>>>(d, nx, ny, nz, bcFaceXright);
  }

  // Y boundaries
  if (vct->getYleft_neighbor() == MPI_PROC_NULL) {
    dim3 block(BC_BLOCK, BC_BLOCK);
    dim3 grid((nx + BC_BLOCK - 1) / BC_BLOCK, (nz + BC_BLOCK - 1) / BC_BLOCK);
    gpuBCfaceYleft<<<grid, block, 0, stream>>>(d, nx, ny, nz, bcFaceYleft);
  }
  if (vct->getYright_neighbor() == MPI_PROC_NULL) {
    dim3 block(BC_BLOCK, BC_BLOCK);
    dim3 grid((nx + BC_BLOCK - 1) / BC_BLOCK, (nz + BC_BLOCK - 1) / BC_BLOCK);
    gpuBCfaceYright<<<grid, block, 0, stream>>>(d, nx, ny, nz, bcFaceYright);
  }

  // Z boundaries
  if (vct->getZleft_neighbor() == MPI_PROC_NULL) {
    dim3 block(BC_BLOCK, BC_BLOCK);
    dim3 grid((nx + BC_BLOCK - 1) / BC_BLOCK, (ny + BC_BLOCK - 1) / BC_BLOCK);
    gpuBCfaceZleft<<<grid, block, 0, stream>>>(d, nx, ny, nz, bcFaceZleft);
  }
  if (vct->getZright_neighbor() == MPI_PROC_NULL) {
    dim3 block(BC_BLOCK, BC_BLOCK);
    dim3 grid((nx + BC_BLOCK - 1) / BC_BLOCK, (ny + BC_BLOCK - 1) / BC_BLOCK);
    gpuBCfaceZright<<<grid, block, 0, stream>>>(d, nx, ny, nz, bcFaceZright);
  }
}

void gpuBCface_P(int nx, int ny, int nz, GPUFieldArray3& gpuArr,
                 int bcFaceXright, int bcFaceXleft, int bcFaceYright,
                 int bcFaceYleft, int bcFaceZright, int bcFaceZleft,
                 const VirtualTopology3D* vct, cudaStream_t stream) {
  // BCface_P uses particle-topology neighbours but same BC logic.
  // For field solver, particle topology is not used; delegate to gpuBCface
  // with the field topology neighbour checks already handled.
  cudaSolverType* d = gpuArr.devPtr();

  if (vct->getXleft_neighbor_P() == MPI_PROC_NULL) {
    dim3 block(BC_BLOCK, BC_BLOCK);
    dim3 grid((ny + BC_BLOCK - 1) / BC_BLOCK, (nz + BC_BLOCK - 1) / BC_BLOCK);
    gpuBCfaceXleft<<<grid, block, 0, stream>>>(d, nx, ny, nz, bcFaceXleft);
  }
  if (vct->getXright_neighbor_P() == MPI_PROC_NULL) {
    dim3 block(BC_BLOCK, BC_BLOCK);
    dim3 grid((ny + BC_BLOCK - 1) / BC_BLOCK, (nz + BC_BLOCK - 1) / BC_BLOCK);
    gpuBCfaceXright<<<grid, block, 0, stream>>>(d, nx, ny, nz, bcFaceXright);
  }
  if (vct->getYleft_neighbor_P() == MPI_PROC_NULL) {
    dim3 block(BC_BLOCK, BC_BLOCK);
    dim3 grid((nx + BC_BLOCK - 1) / BC_BLOCK, (nz + BC_BLOCK - 1) / BC_BLOCK);
    gpuBCfaceYleft<<<grid, block, 0, stream>>>(d, nx, ny, nz, bcFaceYleft);
  }
  if (vct->getYright_neighbor_P() == MPI_PROC_NULL) {
    dim3 block(BC_BLOCK, BC_BLOCK);
    dim3 grid((nx + BC_BLOCK - 1) / BC_BLOCK, (nz + BC_BLOCK - 1) / BC_BLOCK);
    gpuBCfaceYright<<<grid, block, 0, stream>>>(d, nx, ny, nz, bcFaceYright);
  }
  if (vct->getZleft_neighbor_P() == MPI_PROC_NULL) {
    dim3 block(BC_BLOCK, BC_BLOCK);
    dim3 grid((nx + BC_BLOCK - 1) / BC_BLOCK, (ny + BC_BLOCK - 1) / BC_BLOCK);
    gpuBCfaceZleft<<<grid, block, 0, stream>>>(d, nx, ny, nz, bcFaceZleft);
  }
  if (vct->getZright_neighbor_P() == MPI_PROC_NULL) {
    dim3 block(BC_BLOCK, BC_BLOCK);
    dim3 grid((nx + BC_BLOCK - 1) / BC_BLOCK, (ny + BC_BLOCK - 1) / BC_BLOCK);
    gpuBCfaceZright<<<grid, block, 0, stream>>>(d, nx, ny, nz, bcFaceZright);
  }
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
//  Batched 3-field communicate wrappers
//
//  These perform ONE batched halo exchange for 3 fields at once (nFields=3)
//  instead of 3 sequential exchanges, eliminating 2 complete rounds of
//  cudaStreamSynchronize + MPI_Waitall barriers per call.
// =========================================================================

void EMfields3D::gpuCommunicateCenterBC_3(int nx, int ny, int nz,
                                          GPUFieldArray3& a1,
                                          GPUFieldArray3& a2,
                                          GPUFieldArray3& a3, int bcXR,
                                          int bcXL, int bcYR, int bcYL,
                                          int bcZR, int bcZL) {
  cudaSolverType* ptrs[3] = {a1.devPtr(), a2.devPtr(), a3.devPtr()};
  gpuBatchedHaloExchange(ptrs, 3, nx, ny, nz, true, false, false, false,
                         solverStream_);
  gpuBCface(nx, ny, nz, a1, bcXR, bcXL, bcYR, bcYL, bcZR, bcZL, &_vct,
            solverStream_);
  gpuBCface(nx, ny, nz, a2, bcXR, bcXL, bcYR, bcYL, bcZR, bcZL, &_vct,
            solverStream_);
  gpuBCface(nx, ny, nz, a3, bcXR, bcXL, bcYR, bcYL, bcZR, bcZL, &_vct,
            solverStream_);
}

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

#endif // GPU_SOLVER
