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

#include "GPUHaloComm.cuh"
#include "EMfields3D.h"
#include "VCtopology3D.h"
#include <cassert>
#include <iostream>

// =========================================================================
//  Helper: linear index into a flat (nx, ny, nz) row-major array
// =========================================================================
__device__ __forceinline__
int idx3(int i, int j, int k, int ny, int nz)
{
    return (i * ny + j) * nz + k;
}

// =========================================================================
//  BC kernels – face boundary condition application
//  bcType  0 = Dirichlet 0 second-order  ghost = -interior
//          1 = Dirichlet 0 first-order   ghost = 0
//          2 = Neumann   0 first-order   ghost = interior
// =========================================================================

__global__ void gpuBCfaceXleft(double* __restrict__ arr,
                                int nx, int ny, int nz, int bcType)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int k = blockIdx.y * blockDim.y + threadIdx.y;
    if (j >= ny || k >= nz) return;
    int dst = idx3(0, j, k, ny, nz);
    int src = idx3(1, j, k, ny, nz);
    switch (bcType) {
        case 0: arr[dst] = -arr[src]; break;
        case 1: arr[dst] = 0.0;       break;
        case 2: arr[dst] =  arr[src]; break;
    }
}

__global__ void gpuBCfaceXright(double* __restrict__ arr,
                                 int nx, int ny, int nz, int bcType)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int k = blockIdx.y * blockDim.y + threadIdx.y;
    if (j >= ny || k >= nz) return;
    int dst = idx3(nx - 1, j, k, ny, nz);
    int src = idx3(nx - 2, j, k, ny, nz);
    switch (bcType) {
        case 0: arr[dst] = -arr[src]; break;
        case 1: arr[dst] = 0.0;       break;
        case 2: arr[dst] =  arr[src]; break;
    }
}

__global__ void gpuBCfaceYleft(double* __restrict__ arr,
                                int nx, int ny, int nz, int bcType)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int k = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= nx || k >= nz) return;
    int dst = idx3(i, 0, k, ny, nz);
    int src = idx3(i, 1, k, ny, nz);
    switch (bcType) {
        case 0: arr[dst] = -arr[src]; break;
        case 1: arr[dst] = 0.0;       break;
        case 2: arr[dst] =  arr[src]; break;
    }
}

__global__ void gpuBCfaceYright(double* __restrict__ arr,
                                 int nx, int ny, int nz, int bcType)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int k = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= nx || k >= nz) return;
    int dst = idx3(i, ny - 1, k, ny, nz);
    int src = idx3(i, ny - 2, k, ny, nz);
    switch (bcType) {
        case 0: arr[dst] = -arr[src]; break;
        case 1: arr[dst] = 0.0;       break;
        case 2: arr[dst] =  arr[src]; break;
    }
}

__global__ void gpuBCfaceZleft(double* __restrict__ arr,
                                int nx, int ny, int nz, int bcType)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= nx || j >= ny) return;
    int dst = idx3(i, j, 0, ny, nz);
    int src = idx3(i, j, 1, ny, nz);
    switch (bcType) {
        case 0: arr[dst] = -arr[src]; break;
        case 1: arr[dst] = 0.0;       break;
        case 2: arr[dst] =  arr[src]; break;
    }
}

__global__ void gpuBCfaceZright(double* __restrict__ arr,
                                 int nx, int ny, int nz, int bcType)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= nx || j >= ny) return;
    int dst = idx3(i, j, nz - 1, ny, nz);
    int src = idx3(i, j, nz - 2, ny, nz);
    switch (bcType) {
        case 0: arr[dst] = -arr[src]; break;
        case 1: arr[dst] = 0.0;       break;
        case 2: arr[dst] =  arr[src]; break;
    }
}

// =========================================================================
//  Self-copy kernels – periodic face swap when rank == neighbour
// =========================================================================

__global__ void gpuSelfCopyFaceX(double* __restrict__ arr,
                                  int nx, int ny, int nz)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int k = blockIdx.y * blockDim.y + threadIdx.y + 1;
    if (j >= ny - 1 || k >= nz - 1) return;
    // left ghost = right interior, right ghost = left interior
    arr[idx3(0,      j, k, ny, nz)] = arr[idx3(nx - 2, j, k, ny, nz)];
    arr[idx3(nx - 1, j, k, ny, nz)] = arr[idx3(1,      j, k, ny, nz)];
}

__global__ void gpuSelfCopyFaceY(double* __restrict__ arr,
                                  int nx, int ny, int nz)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int k = blockIdx.y * blockDim.y + threadIdx.y + 1;
    if (i >= nx - 1 || k >= nz - 1) return;
    arr[idx3(i, 0,      k, ny, nz)] = arr[idx3(i, ny - 2, k, ny, nz)];
    arr[idx3(i, ny - 1, k, ny, nz)] = arr[idx3(i, 1,      k, ny, nz)];
}

__global__ void gpuSelfCopyFaceZ(double* __restrict__ arr,
                                  int nx, int ny, int nz)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int j = blockIdx.y * blockDim.y + threadIdx.y + 1;
    if (i >= nx - 1 || j >= ny - 1) return;
    arr[idx3(i, j, 0,      ny, nz)] = arr[idx3(i, j, nz - 2, ny, nz)];
    arr[idx3(i, j, nz - 1, ny, nz)] = arr[idx3(i, j, 1,      ny, nz)];
}

// =========================================================================
//  Self-copy kernels – periodic edge swap when rank == neighbour
// =========================================================================

__global__ void gpuSelfCopyEdgeX(double* __restrict__ arr,
                                  int nx, int ny, int nz,
                                  bool hasZright, bool hasZleft,
                                  bool hasYright, bool hasYleft)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    // Along Y (1..ny-2) for Z-neighbour edges, along Z (1..nz-2) for Y-neighbour edges
    if (hasZright) {
        int iy = tid + 1;
        if (iy < ny - 1) {
            arr[idx3(0,      iy, nz - 1, ny, nz)] = arr[idx3(nx - 2, iy, nz - 1, ny, nz)];
            arr[idx3(nx - 1, iy, nz - 1, ny, nz)] = arr[idx3(1,      iy, nz - 1, ny, nz)];
        }
    }
    if (hasZleft) {
        int iy = tid + 1;
        if (iy < ny - 1) {
            arr[idx3(0,      iy, 0, ny, nz)] = arr[idx3(nx - 2, iy, 0, ny, nz)];
            arr[idx3(nx - 1, iy, 0, ny, nz)] = arr[idx3(1,      iy, 0, ny, nz)];
        }
    }
    if (hasYright) {
        int iz = tid + 1;
        if (iz < nz - 1) {
            arr[idx3(0,      ny - 1, iz, ny, nz)] = arr[idx3(nx - 2, ny - 1, iz, ny, nz)];
            arr[idx3(nx - 1, ny - 1, iz, ny, nz)] = arr[idx3(1,      ny - 1, iz, ny, nz)];
        }
    }
    if (hasYleft) {
        int iz = tid + 1;
        if (iz < nz - 1) {
            arr[idx3(0,      0, iz, ny, nz)] = arr[idx3(nx - 2, 0, iz, ny, nz)];
            arr[idx3(nx - 1, 0, iz, ny, nz)] = arr[idx3(1,      0, iz, ny, nz)];
        }
    }
}

__global__ void gpuSelfCopyEdgeY(double* __restrict__ arr,
                                  int nx, int ny, int nz,
                                  bool hasXright, bool hasXleft,
                                  bool hasZright, bool hasZleft)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (hasXright) {
        int iz = tid + 1;
        if (iz < nz - 1) {
            arr[idx3(nx - 1, 0,      iz, ny, nz)] = arr[idx3(nx - 1, ny - 2, iz, ny, nz)];
            arr[idx3(nx - 1, ny - 1, iz, ny, nz)] = arr[idx3(nx - 1, 1,      iz, ny, nz)];
        }
    }
    if (hasXleft) {
        int iz = tid + 1;
        if (iz < nz - 1) {
            arr[idx3(0, 0,      iz, ny, nz)] = arr[idx3(0, ny - 2, iz, ny, nz)];
            arr[idx3(0, ny - 1, iz, ny, nz)] = arr[idx3(0, 1,      iz, ny, nz)];
        }
    }
    if (hasZright) {
        int ix = tid + 1;
        if (ix < nx - 1) {
            arr[idx3(ix, 0,      nz - 1, ny, nz)] = arr[idx3(ix, ny - 2, nz - 1, ny, nz)];
            arr[idx3(ix, ny - 1, nz - 1, ny, nz)] = arr[idx3(ix, 1,      nz - 1, ny, nz)];
        }
    }
    if (hasZleft) {
        int ix = tid + 1;
        if (ix < nx - 1) {
            arr[idx3(ix, 0,      0, ny, nz)] = arr[idx3(ix, ny - 2, 0, ny, nz)];
            arr[idx3(ix, ny - 1, 0, ny, nz)] = arr[idx3(ix, 1,      0, ny, nz)];
        }
    }
}

__global__ void gpuSelfCopyEdgeZ(double* __restrict__ arr,
                                  int nx, int ny, int nz,
                                  bool hasYright, bool hasYleft,
                                  bool hasXright, bool hasXleft)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (hasYright) {
        int ix = tid + 1;
        if (ix < nx - 1) {
            arr[idx3(ix, ny - 1, 0,      ny, nz)] = arr[idx3(ix, ny - 1, nz - 2, ny, nz)];
            arr[idx3(ix, ny - 1, nz - 1, ny, nz)] = arr[idx3(ix, ny - 1, 1,      ny, nz)];
        }
    }
    if (hasYleft) {
        int ix = tid + 1;
        if (ix < nx - 1) {
            arr[idx3(ix, 0, 0,      ny, nz)] = arr[idx3(ix, 0, nz - 2, ny, nz)];
            arr[idx3(ix, 0, nz - 1, ny, nz)] = arr[idx3(ix, 0, 1,      ny, nz)];
        }
    }
    if (hasXright) {
        int iy = tid + 1;
        if (iy < ny - 1) {
            arr[idx3(nx - 1, iy, 0,      ny, nz)] = arr[idx3(nx - 1, iy, nz - 2, ny, nz)];
            arr[idx3(nx - 1, iy, nz - 1, ny, nz)] = arr[idx3(nx - 1, iy, 1,      ny, nz)];
        }
    }
    if (hasXleft) {
        int iy = tid + 1;
        if (iy < ny - 1) {
            arr[idx3(0, iy, 0,      ny, nz)] = arr[idx3(0, iy, nz - 2, ny, nz)];
            arr[idx3(0, iy, nz - 1, ny, nz)] = arr[idx3(0, iy, 1,      ny, nz)];
        }
    }
}

// =========================================================================
//  Self-copy kernels – periodic corner swap
// =========================================================================

__global__ void gpuSelfCopyCornerX(double* __restrict__ arr,
                                    int nx, int ny, int nz,
                                    bool hasYleft, bool hasYright,
                                    bool hasZleft, bool hasZright)
{
    // Single-thread kernel (only 4 elements at most)
    if (threadIdx.x != 0) return;
    if (hasYleft && hasZleft) {
        arr[idx3(0, 0, 0, ny, nz)]       = arr[idx3(nx - 2, 0, 0, ny, nz)];
        arr[idx3(nx - 1, 0, 0, ny, nz)]  = arr[idx3(1, 0, 0, ny, nz)];
    }
    if (hasYleft && hasZright) {
        arr[idx3(0, 0, nz - 1, ny, nz)]       = arr[idx3(nx - 2, 0, nz - 1, ny, nz)];
        arr[idx3(nx - 1, 0, nz - 1, ny, nz)]  = arr[idx3(1, 0, nz - 1, ny, nz)];
    }
    if (hasYright && hasZleft) {
        arr[idx3(0, ny - 1, 0, ny, nz)]       = arr[idx3(nx - 2, ny - 1, 0, ny, nz)];
        arr[idx3(nx - 1, ny - 1, 0, ny, nz)]  = arr[idx3(1, ny - 1, 0, ny, nz)];
    }
    if (hasYright && hasZright) {
        arr[idx3(0, ny - 1, nz - 1, ny, nz)]       = arr[idx3(nx - 2, ny - 1, nz - 1, ny, nz)];
        arr[idx3(nx - 1, ny - 1, nz - 1, ny, nz)]  = arr[idx3(1, ny - 1, nz - 1, ny, nz)];
    }
}

__global__ void gpuSelfCopyCornerY(double* __restrict__ arr,
                                    int nx, int ny, int nz,
                                    bool hasXleft, bool hasXright,
                                    bool hasZleft, bool hasZright)
{
    if (threadIdx.x != 0) return;
    if (hasXleft && hasZleft) {
        arr[idx3(0, 0, 0, ny, nz)]       = arr[idx3(0, ny - 2, 0, ny, nz)];
        arr[idx3(0, ny - 1, 0, ny, nz)]  = arr[idx3(0, 1, 0, ny, nz)];
    }
    if (hasXleft && hasZright) {
        arr[idx3(0, 0, nz - 1, ny, nz)]       = arr[idx3(0, ny - 2, nz - 1, ny, nz)];
        arr[idx3(0, ny - 1, nz - 1, ny, nz)]  = arr[idx3(0, 1, nz - 1, ny, nz)];
    }
    if (hasXright && hasZleft) {
        arr[idx3(nx - 1, 0, 0, ny, nz)]       = arr[idx3(nx - 1, ny - 2, 0, ny, nz)];
        arr[idx3(nx - 1, ny - 1, 0, ny, nz)]  = arr[idx3(nx - 1, 1, 0, ny, nz)];
    }
    if (hasXright && hasZright) {
        arr[idx3(nx - 1, 0, nz - 1, ny, nz)]       = arr[idx3(nx - 1, ny - 2, nz - 1, ny, nz)];
        arr[idx3(nx - 1, ny - 1, nz - 1, ny, nz)]  = arr[idx3(nx - 1, 1, nz - 1, ny, nz)];
    }
}

__global__ void gpuSelfCopyCornerZ(double* __restrict__ arr,
                                    int nx, int ny, int nz,
                                    bool hasYleft, bool hasYright,
                                    bool hasXleft, bool hasXright)
{
    if (threadIdx.x != 0) return;
    if (hasYleft && hasXleft) {
        arr[idx3(0, 0, 0, ny, nz)]       = arr[idx3(0, 0, nz - 2, ny, nz)];
        arr[idx3(0, 0, nz - 1, ny, nz)]  = arr[idx3(0, 0, 1, ny, nz)];
    }
    if (hasYleft && hasXright) {
        arr[idx3(nx - 1, 0, 0, ny, nz)]       = arr[idx3(nx - 1, 0, nz - 2, ny, nz)];
        arr[idx3(nx - 1, 0, nz - 1, ny, nz)]  = arr[idx3(nx - 1, 0, 1, ny, nz)];
    }
    if (hasYright && hasXleft) {
        arr[idx3(0, ny - 1, 0, ny, nz)]       = arr[idx3(0, ny - 1, nz - 2, ny, nz)];
        arr[idx3(0, ny - 1, nz - 1, ny, nz)]  = arr[idx3(0, ny - 1, 1, ny, nz)];
    }
    if (hasYright && hasXright) {
        arr[idx3(nx - 1, ny - 1, 0, ny, nz)]       = arr[idx3(nx - 1, ny - 1, nz - 2, ny, nz)];
        arr[idx3(nx - 1, ny - 1, nz - 1, ny, nz)]  = arr[idx3(nx - 1, ny - 1, 1, ny, nz)];
    }
}

// =========================================================================
//  Batched self-copy kernels – face (uses blockIdx.z for field index)
// =========================================================================

__global__ void gpuBatchSelfCopyFaceX(double* const* __restrict__ fields,
                                       int nx, int ny, int nz)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int k = blockIdx.y * blockDim.y + threadIdx.y + 1;
    if (j >= ny - 1 || k >= nz - 1) return;
    double* arr = fields[blockIdx.z];
    arr[idx3(0,      j, k, ny, nz)] = arr[idx3(nx - 2, j, k, ny, nz)];
    arr[idx3(nx - 1, j, k, ny, nz)] = arr[idx3(1,      j, k, ny, nz)];
}

__global__ void gpuBatchSelfCopyFaceY(double* const* __restrict__ fields,
                                       int nx, int ny, int nz)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int k = blockIdx.y * blockDim.y + threadIdx.y + 1;
    if (i >= nx - 1 || k >= nz - 1) return;
    double* arr = fields[blockIdx.z];
    arr[idx3(i, 0,      k, ny, nz)] = arr[idx3(i, ny - 2, k, ny, nz)];
    arr[idx3(i, ny - 1, k, ny, nz)] = arr[idx3(i, 1,      k, ny, nz)];
}

__global__ void gpuBatchSelfCopyFaceZ(double* const* __restrict__ fields,
                                       int nx, int ny, int nz)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int j = blockIdx.y * blockDim.y + threadIdx.y + 1;
    if (i >= nx - 1 || j >= ny - 1) return;
    double* arr = fields[blockIdx.z];
    arr[idx3(i, j, 0,      ny, nz)] = arr[idx3(i, j, nz - 2, ny, nz)];
    arr[idx3(i, j, nz - 1, ny, nz)] = arr[idx3(i, j, 1,      ny, nz)];
}

// =========================================================================
//  Batched self-copy kernels – edge (uses blockIdx.y for field index)
// =========================================================================

__global__ void gpuBatchSelfCopyEdgeX(double* const* __restrict__ fields,
                                       int nx, int ny, int nz,
                                       bool hasZright, bool hasZleft,
                                       bool hasYright, bool hasYleft)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    double* arr = fields[blockIdx.y];
    if (hasZright) {
        int iy = tid + 1;
        if (iy < ny - 1) {
            arr[idx3(0,      iy, nz - 1, ny, nz)] = arr[idx3(nx - 2, iy, nz - 1, ny, nz)];
            arr[idx3(nx - 1, iy, nz - 1, ny, nz)] = arr[idx3(1,      iy, nz - 1, ny, nz)];
        }
    }
    if (hasZleft) {
        int iy = tid + 1;
        if (iy < ny - 1) {
            arr[idx3(0,      iy, 0, ny, nz)] = arr[idx3(nx - 2, iy, 0, ny, nz)];
            arr[idx3(nx - 1, iy, 0, ny, nz)] = arr[idx3(1,      iy, 0, ny, nz)];
        }
    }
    if (hasYright) {
        int iz = tid + 1;
        if (iz < nz - 1) {
            arr[idx3(0,      ny - 1, iz, ny, nz)] = arr[idx3(nx - 2, ny - 1, iz, ny, nz)];
            arr[idx3(nx - 1, ny - 1, iz, ny, nz)] = arr[idx3(1,      ny - 1, iz, ny, nz)];
        }
    }
    if (hasYleft) {
        int iz = tid + 1;
        if (iz < nz - 1) {
            arr[idx3(0,      0, iz, ny, nz)] = arr[idx3(nx - 2, 0, iz, ny, nz)];
            arr[idx3(nx - 1, 0, iz, ny, nz)] = arr[idx3(1,      0, iz, ny, nz)];
        }
    }
}

__global__ void gpuBatchSelfCopyEdgeY(double* const* __restrict__ fields,
                                       int nx, int ny, int nz,
                                       bool hasXright, bool hasXleft,
                                       bool hasZright, bool hasZleft)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    double* arr = fields[blockIdx.y];
    if (hasXright) {
        int iz = tid + 1;
        if (iz < nz - 1) {
            arr[idx3(nx - 1, 0,      iz, ny, nz)] = arr[idx3(nx - 1, ny - 2, iz, ny, nz)];
            arr[idx3(nx - 1, ny - 1, iz, ny, nz)] = arr[idx3(nx - 1, 1,      iz, ny, nz)];
        }
    }
    if (hasXleft) {
        int iz = tid + 1;
        if (iz < nz - 1) {
            arr[idx3(0, 0,      iz, ny, nz)] = arr[idx3(0, ny - 2, iz, ny, nz)];
            arr[idx3(0, ny - 1, iz, ny, nz)] = arr[idx3(0, 1,      iz, ny, nz)];
        }
    }
    if (hasZright) {
        int ix = tid + 1;
        if (ix < nx - 1) {
            arr[idx3(ix, 0,      nz - 1, ny, nz)] = arr[idx3(ix, ny - 2, nz - 1, ny, nz)];
            arr[idx3(ix, ny - 1, nz - 1, ny, nz)] = arr[idx3(ix, 1,      nz - 1, ny, nz)];
        }
    }
    if (hasZleft) {
        int ix = tid + 1;
        if (ix < nx - 1) {
            arr[idx3(ix, 0,      0, ny, nz)] = arr[idx3(ix, ny - 2, 0, ny, nz)];
            arr[idx3(ix, ny - 1, 0, ny, nz)] = arr[idx3(ix, 1,      0, ny, nz)];
        }
    }
}

__global__ void gpuBatchSelfCopyEdgeZ(double* const* __restrict__ fields,
                                       int nx, int ny, int nz,
                                       bool hasYright, bool hasYleft,
                                       bool hasXright, bool hasXleft)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    double* arr = fields[blockIdx.y];
    if (hasYright) {
        int ix = tid + 1;
        if (ix < nx - 1) {
            arr[idx3(ix, ny - 1, 0,      ny, nz)] = arr[idx3(ix, ny - 1, nz - 2, ny, nz)];
            arr[idx3(ix, ny - 1, nz - 1, ny, nz)] = arr[idx3(ix, ny - 1, 1,      ny, nz)];
        }
    }
    if (hasYleft) {
        int ix = tid + 1;
        if (ix < nx - 1) {
            arr[idx3(ix, 0, 0,      ny, nz)] = arr[idx3(ix, 0, nz - 2, ny, nz)];
            arr[idx3(ix, 0, nz - 1, ny, nz)] = arr[idx3(ix, 0, 1,      ny, nz)];
        }
    }
    if (hasXright) {
        int iy = tid + 1;
        if (iy < ny - 1) {
            arr[idx3(nx - 1, iy, 0,      ny, nz)] = arr[idx3(nx - 1, iy, nz - 2, ny, nz)];
            arr[idx3(nx - 1, iy, nz - 1, ny, nz)] = arr[idx3(nx - 1, iy, 1,      ny, nz)];
        }
    }
    if (hasXleft) {
        int iy = tid + 1;
        if (iy < ny - 1) {
            arr[idx3(0, iy, 0,      ny, nz)] = arr[idx3(0, iy, nz - 2, ny, nz)];
            arr[idx3(0, iy, nz - 1, ny, nz)] = arr[idx3(0, iy, 1,      ny, nz)];
        }
    }
}

// =========================================================================
//  Batched self-copy kernels – corner (uses blockIdx.x for field index)
// =========================================================================

__global__ void gpuBatchSelfCopyCornerX(double* const* __restrict__ fields,
                                         int nx, int ny, int nz,
                                         bool hasYleft, bool hasYright,
                                         bool hasZleft, bool hasZright)
{
    double* arr = fields[blockIdx.x];
    if (threadIdx.x != 0) return;
    if (hasYleft && hasZleft) {
        arr[idx3(0, 0, 0, ny, nz)]       = arr[idx3(nx - 2, 0, 0, ny, nz)];
        arr[idx3(nx - 1, 0, 0, ny, nz)]  = arr[idx3(1, 0, 0, ny, nz)];
    }
    if (hasYleft && hasZright) {
        arr[idx3(0, 0, nz - 1, ny, nz)]       = arr[idx3(nx - 2, 0, nz - 1, ny, nz)];
        arr[idx3(nx - 1, 0, nz - 1, ny, nz)]  = arr[idx3(1, 0, nz - 1, ny, nz)];
    }
    if (hasYright && hasZleft) {
        arr[idx3(0, ny - 1, 0, ny, nz)]       = arr[idx3(nx - 2, ny - 1, 0, ny, nz)];
        arr[idx3(nx - 1, ny - 1, 0, ny, nz)]  = arr[idx3(1, ny - 1, 0, ny, nz)];
    }
    if (hasYright && hasZright) {
        arr[idx3(0, ny - 1, nz - 1, ny, nz)]       = arr[idx3(nx - 2, ny - 1, nz - 1, ny, nz)];
        arr[idx3(nx - 1, ny - 1, nz - 1, ny, nz)]  = arr[idx3(1, ny - 1, nz - 1, ny, nz)];
    }
}

__global__ void gpuBatchSelfCopyCornerY(double* const* __restrict__ fields,
                                         int nx, int ny, int nz,
                                         bool hasXleft, bool hasXright,
                                         bool hasZleft, bool hasZright)
{
    double* arr = fields[blockIdx.x];
    if (threadIdx.x != 0) return;
    if (hasXleft && hasZleft) {
        arr[idx3(0, 0, 0, ny, nz)]       = arr[idx3(0, ny - 2, 0, ny, nz)];
        arr[idx3(0, ny - 1, 0, ny, nz)]  = arr[idx3(0, 1, 0, ny, nz)];
    }
    if (hasXleft && hasZright) {
        arr[idx3(0, 0, nz - 1, ny, nz)]       = arr[idx3(0, ny - 2, nz - 1, ny, nz)];
        arr[idx3(0, ny - 1, nz - 1, ny, nz)]  = arr[idx3(0, 1, nz - 1, ny, nz)];
    }
    if (hasXright && hasZleft) {
        arr[idx3(nx - 1, 0, 0, ny, nz)]       = arr[idx3(nx - 1, ny - 2, 0, ny, nz)];
        arr[idx3(nx - 1, ny - 1, 0, ny, nz)]  = arr[idx3(nx - 1, 1, 0, ny, nz)];
    }
    if (hasXright && hasZright) {
        arr[idx3(nx - 1, 0, nz - 1, ny, nz)]       = arr[idx3(nx - 1, ny - 2, nz - 1, ny, nz)];
        arr[idx3(nx - 1, ny - 1, nz - 1, ny, nz)]  = arr[idx3(nx - 1, 1, nz - 1, ny, nz)];
    }
}

__global__ void gpuBatchSelfCopyCornerZ(double* const* __restrict__ fields,
                                         int nx, int ny, int nz,
                                         bool hasYleft, bool hasYright,
                                         bool hasXleft, bool hasXright)
{
    double* arr = fields[blockIdx.x];
    if (threadIdx.x != 0) return;
    if (hasYleft && hasXleft) {
        arr[idx3(0, 0, 0, ny, nz)]       = arr[idx3(0, 0, nz - 2, ny, nz)];
        arr[idx3(0, 0, nz - 1, ny, nz)]  = arr[idx3(0, 0, 1, ny, nz)];
    }
    if (hasYleft && hasXright) {
        arr[idx3(nx - 1, 0, 0, ny, nz)]       = arr[idx3(nx - 1, 0, nz - 2, ny, nz)];
        arr[idx3(nx - 1, 0, nz - 1, ny, nz)]  = arr[idx3(nx - 1, 0, 1, ny, nz)];
    }
    if (hasYright && hasXleft) {
        arr[idx3(0, ny - 1, 0, ny, nz)]       = arr[idx3(0, ny - 1, nz - 2, ny, nz)];
        arr[idx3(0, ny - 1, nz - 1, ny, nz)]  = arr[idx3(0, ny - 1, 1, ny, nz)];
    }
    if (hasYright && hasXright) {
        arr[idx3(nx - 1, ny - 1, 0, ny, nz)]       = arr[idx3(nx - 1, ny - 1, nz - 2, ny, nz)];
        arr[idx3(nx - 1, ny - 1, nz - 1, ny, nz)]  = arr[idx3(nx - 1, ny - 1, 1, ny, nz)];
    }
}

// =========================================================================
//  Additive interpolation kernels (for communicateInterp)
// =========================================================================

__global__ void gpuAddFaceX(double* __restrict__ arr, int nx, int ny, int nz,
                             bool hasXright, bool hasXleft)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int k = blockIdx.y * blockDim.y + threadIdx.y + 1;
    if (j > ny - 2 || k > nz - 2) return;
    if (hasXright)
        arr[idx3(nx - 2, j, k, ny, nz)] += arr[idx3(nx - 1, j, k, ny, nz)];
    if (hasXleft)
        arr[idx3(1, j, k, ny, nz)] += arr[idx3(0, j, k, ny, nz)];
}

__global__ void gpuAddFaceY(double* __restrict__ arr, int nx, int ny, int nz,
                             bool hasYright, bool hasYleft)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int k = blockIdx.y * blockDim.y + threadIdx.y + 1;
    if (i > nx - 2 || k > nz - 2) return;
    if (hasYright)
        arr[idx3(i, ny - 2, k, ny, nz)] += arr[idx3(i, ny - 1, k, ny, nz)];
    if (hasYleft)
        arr[idx3(i, 1, k, ny, nz)] += arr[idx3(i, 0, k, ny, nz)];
}

__global__ void gpuAddFaceZ(double* __restrict__ arr, int nx, int ny, int nz,
                             bool hasZright, bool hasZleft)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int j = blockIdx.y * blockDim.y + threadIdx.y + 1;
    if (i > nx - 2 || j > ny - 2) return;
    if (hasZright)
        arr[idx3(i, j, nz - 2, ny, nz)] += arr[idx3(i, j, nz - 1, ny, nz)];
    if (hasZleft)
        arr[idx3(i, j, 1, ny, nz)] += arr[idx3(i, j, 0, ny, nz)];
}

__global__ void gpuAddEdgeZ(double* __restrict__ arr, int nx, int ny, int nz,
                             bool hasXright, bool hasXleft,
                             bool hasYright, bool hasYleft)
{
    int k = blockIdx.x * blockDim.x + threadIdx.x + 1;
    if (k > nz - 2) return;
    if (hasXright && hasYright)
        arr[idx3(nx - 2, ny - 2, k, ny, nz)] += arr[idx3(nx - 1, ny - 1, k, ny, nz)];
    if (hasXleft && hasYleft)
        arr[idx3(1, 1, k, ny, nz)] += arr[idx3(0, 0, k, ny, nz)];
    if (hasXright && hasYleft)
        arr[idx3(nx - 2, 1, k, ny, nz)] += arr[idx3(nx - 1, 0, k, ny, nz)];
    if (hasXleft && hasYright)
        arr[idx3(1, ny - 2, k, ny, nz)] += arr[idx3(0, ny - 1, k, ny, nz)];
}

__global__ void gpuAddEdgeY(double* __restrict__ arr, int nx, int ny, int nz,
                             bool hasXright, bool hasXleft,
                             bool hasZright, bool hasZleft)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x + 1;
    if (j > ny - 2) return;
    if (hasXright && hasZright)
        arr[idx3(nx - 2, j, nz - 2, ny, nz)] += arr[idx3(nx - 1, j, nz - 1, ny, nz)];
    if (hasXleft && hasZleft)
        arr[idx3(1, j, 1, ny, nz)] += arr[idx3(0, j, 0, ny, nz)];
    if (hasXleft && hasZright)
        arr[idx3(1, j, nz - 2, ny, nz)] += arr[idx3(0, j, nz - 1, ny, nz)];
    if (hasXright && hasZleft)
        arr[idx3(nx - 2, j, 1, ny, nz)] += arr[idx3(nx - 1, j, 0, ny, nz)];
}

__global__ void gpuAddEdgeX(double* __restrict__ arr, int nx, int ny, int nz,
                             bool hasYright, bool hasYleft,
                             bool hasZright, bool hasZleft)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
    if (i > nx - 2) return;
    if (hasYright && hasZright)
        arr[idx3(i, ny - 2, nz - 2, ny, nz)] += arr[idx3(i, ny - 1, nz - 1, ny, nz)];
    if (hasYleft && hasZleft)
        arr[idx3(i, 1, 1, ny, nz)] += arr[idx3(i, 0, 0, ny, nz)];
    if (hasYleft && hasZright)
        arr[idx3(i, 1, nz - 2, ny, nz)] += arr[idx3(i, 0, nz - 1, ny, nz)];
    if (hasYright && hasZleft)
        arr[idx3(i, ny - 2, 1, ny, nz)] += arr[idx3(i, ny - 1, 0, ny, nz)];
}

__global__ void gpuAddCorner(double* __restrict__ arr, int nx, int ny, int nz,
                              bool hasXright, bool hasXleft,
                              bool hasYright, bool hasYleft,
                              bool hasZright, bool hasZleft)
{
    if (threadIdx.x != 0) return;
    if (hasXright && hasYright && hasZright)
        arr[idx3(nx - 2, ny - 2, nz - 2, ny, nz)] += arr[idx3(nx - 1, ny - 1, nz - 1, ny, nz)];
    if (hasXleft && hasYright && hasZright)
        arr[idx3(1, ny - 2, nz - 2, ny, nz)] += arr[idx3(0, ny - 1, nz - 1, ny, nz)];
    if (hasXright && hasYleft && hasZright)
        arr[idx3(nx - 2, 1, nz - 2, ny, nz)] += arr[idx3(nx - 1, 0, nz - 1, ny, nz)];
    if (hasXleft && hasYleft && hasZright)
        arr[idx3(1, 1, nz - 2, ny, nz)] += arr[idx3(0, 0, nz - 1, ny, nz)];
    if (hasXright && hasYright && hasZleft)
        arr[idx3(nx - 2, ny - 2, 1, ny, nz)] += arr[idx3(nx - 1, ny - 1, 0, ny, nz)];
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

void gpuBCface(int nx, int ny, int nz,
               GPUFieldArray3& gpuArr,
               int bcFaceXright, int bcFaceXleft,
               int bcFaceYright, int bcFaceYleft,
               int bcFaceZright, int bcFaceZleft,
               const VirtualTopology3D* vct,
               cudaStream_t stream)
{
    double* d = gpuArr.devPtr();

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

void gpuBCface_P(int nx, int ny, int nz,
                 GPUFieldArray3& gpuArr,
                 int bcFaceXright, int bcFaceXleft,
                 int bcFaceYright, int bcFaceYleft,
                 int bcFaceZright, int bcFaceZleft,
                 const VirtualTopology3D* vct,
                 cudaStream_t stream)
{
    // BCface_P uses particle-topology neighbours but same BC logic.
    // For field solver, particle topology is not used; delegate to gpuBCface
    // with the field topology neighbour checks already handled.
    double* d = gpuArr.devPtr();

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
//  GPU-aware version of NBDerivedHaloComm
//
//  Same 3-phase structure (faces → edges → corners) with MPI derived
//  datatypes, but all buffer pointers are device pointers from
//  GPUFieldArray3.  Local self-copies use CUDA kernels.
// =========================================================================

void gpuNBDerivedHaloComm(int nx, int ny, int nz,
                           GPUFieldArray3& gpuArr,
                           const VirtualTopology3D* vct,
                           EMfields3D* EMf,
                           bool isCenterFlag,
                           bool isFaceOnlyFlag,
                           bool needInterp,
                           bool isParticle,
                           cudaStream_t stream)
{
    static int callCount = 0;
    const int myrank = vct->getCartesian_rank();

    // Synchronise stream so all device writes are visible to MPI
    cudaStreamSynchronize(stream);

    double* d = gpuArr.devPtr();
    const MPI_Comm comm = isParticle ? vct->getParticleComm() : vct->getFieldComm();

    MPI_Status  stat[12];
    MPI_Request reqList[12];
    int communicationCnt[6] = {0, 0, 0, 0, 0, 0};
    int recvcnt = 0, sendcnt = 0;

    const int tag_XL = 1, tag_YL = 2, tag_ZL = 3, tag_XR = 4, tag_YR = 5, tag_ZR = 6;
    const int right_neighborX = isParticle ? vct->getXright_neighbor_P() : vct->getXright_neighbor();
    const int left_neighborX  = isParticle ? vct->getXleft_neighbor_P()  : vct->getXleft_neighbor();
    const int right_neighborY = isParticle ? vct->getYright_neighbor_P() : vct->getYright_neighbor();
    const int left_neighborY  = isParticle ? vct->getYleft_neighbor_P()  : vct->getYleft_neighbor();
    const int right_neighborZ = isParticle ? vct->getZright_neighbor_P() : vct->getZright_neighbor();
    const int left_neighborZ  = isParticle ? vct->getZleft_neighbor_P()  : vct->getZleft_neighbor();

    bool isCenterDim = (isCenterFlag && !needInterp);
    const MPI_Datatype yzFacetype  = EMf->getYZFacetype(isCenterDim);
    const MPI_Datatype xzFacetype  = EMf->getXZFacetype(isCenterDim);
    const MPI_Datatype xyFacetype  = EMf->getXYFacetype(isCenterDim);
    const MPI_Datatype xEdgetype   = EMf->getXEdgetype(isCenterDim);
    const MPI_Datatype yEdgetype   = EMf->getYEdgetype(isCenterDim);
    const MPI_Datatype zEdgetype   = EMf->getZEdgetype(isCenterDim);
    const MPI_Datatype xEdgetype2  = EMf->getXEdgetype2(isCenterDim);
    const MPI_Datatype yEdgetype2  = EMf->getYEdgetype2(isCenterDim);
    const MPI_Datatype zEdgetype2  = EMf->getZEdgetype2(isCenterDim);
    const MPI_Datatype cornertype  = EMf->getCornertype(isCenterDim);

    // Helper lambda: compute device pointer at [i][j][k]
    auto dptr = [&](int i, int j, int k) -> double* {
        return d + (i * ny + j) * nz + k;
    };

    // =====================================================================
    //  Phase 1: Face exchange
    // =====================================================================
    if (left_neighborX != MPI_PROC_NULL && left_neighborX != myrank) {
        MPI_Irecv(dptr(0, 1, 1), 1, yzFacetype, left_neighborX, tag_XR, comm, &reqList[recvcnt++]);
        communicationCnt[0] = 1;
    }
    if (right_neighborX != MPI_PROC_NULL && right_neighborX != myrank) {
        MPI_Irecv(dptr(nx - 1, 1, 1), 1, yzFacetype, right_neighborX, tag_XL, comm, &reqList[recvcnt++]);
        communicationCnt[1] = 1;
    }
    if (left_neighborY != MPI_PROC_NULL && left_neighborY != myrank) {
        MPI_Irecv(dptr(1, 0, 1), 1, xzFacetype, left_neighborY, tag_YR, comm, &reqList[recvcnt++]);
        communicationCnt[2] = 1;
    }
    if (right_neighborY != MPI_PROC_NULL && right_neighborY != myrank) {
        MPI_Irecv(dptr(1, ny - 1, 1), 1, xzFacetype, right_neighborY, tag_YL, comm, &reqList[recvcnt++]);
        communicationCnt[3] = 1;
    }
    if (left_neighborZ != MPI_PROC_NULL && left_neighborZ != myrank) {
        MPI_Irecv(dptr(1, 1, 0), 1, xyFacetype, left_neighborZ, tag_ZR, comm, &reqList[recvcnt++]);
        communicationCnt[4] = 1;
    }
    if (right_neighborZ != MPI_PROC_NULL && right_neighborZ != myrank) {
        MPI_Irecv(dptr(1, 1, nz - 1), 1, xyFacetype, right_neighborZ, tag_ZL, comm, &reqList[recvcnt++]);
        communicationCnt[5] = 1;
    }

    sendcnt = recvcnt;
    int offset = (isCenterFlag ? 0 : 1);

    if (communicationCnt[0] == 1)
        MPI_Isend(dptr(1 + offset, 1, 1),      1, yzFacetype, left_neighborX,  tag_XL, comm, &reqList[sendcnt++]);
    if (communicationCnt[1] == 1)
        MPI_Isend(dptr(nx - 2 - offset, 1, 1), 1, yzFacetype, right_neighborX, tag_XR, comm, &reqList[sendcnt++]);
    if (communicationCnt[2] == 1)
        MPI_Isend(dptr(1, 1 + offset, 1),      1, xzFacetype, left_neighborY,  tag_YL, comm, &reqList[sendcnt++]);
    if (communicationCnt[3] == 1)
        MPI_Isend(dptr(1, ny - 2 - offset, 1), 1, xzFacetype, right_neighborY, tag_YR, comm, &reqList[sendcnt++]);
    if (communicationCnt[4] == 1)
        MPI_Isend(dptr(1, 1, 1 + offset),      1, xyFacetype, left_neighborZ,  tag_ZL, comm, &reqList[sendcnt++]);
    if (communicationCnt[5] == 1)
        MPI_Isend(dptr(1, 1, nz - 2 - offset), 1, xyFacetype, right_neighborZ, tag_ZR, comm, &reqList[sendcnt++]);

    // Local self-copy for periodic self-neighbours (CUDA kernels)
    {
        dim3 block(BC_BLOCK, BC_BLOCK);
        if (right_neighborX == myrank && left_neighborX == myrank) {
            dim3 grid(((ny - 2) + BC_BLOCK - 1) / BC_BLOCK, ((nz - 2) + BC_BLOCK - 1) / BC_BLOCK);
            gpuSelfCopyFaceX<<<grid, block, 0, stream>>>(d, nx, ny, nz);
        }
        if (right_neighborY == myrank && left_neighborY == myrank) {
            dim3 grid(((nx - 2) + BC_BLOCK - 1) / BC_BLOCK, ((nz - 2) + BC_BLOCK - 1) / BC_BLOCK);
            gpuSelfCopyFaceY<<<grid, block, 0, stream>>>(d, nx, ny, nz);
        }
        if (right_neighborZ == myrank && left_neighborZ == myrank) {
            dim3 grid(((nx - 2) + BC_BLOCK - 1) / BC_BLOCK, ((ny - 2) + BC_BLOCK - 1) / BC_BLOCK);
            gpuSelfCopyFaceZ<<<grid, block, 0, stream>>>(d, nx, ny, nz);
        }
        cudaStreamSynchronize(stream);
    }

    if (sendcnt > 0) {
        MPI_Waitall(sendcnt, &reqList[0], &stat[0]);
    }

    // =====================================================================
    //  Phase 2: Edge exchange (skip if face-only)
    // =====================================================================
    if (!isFaceOnlyFlag) {
        recvcnt = 0; sendcnt = 0;

        // Y-edges (along XZ intersections)
        if (communicationCnt[0] == 1) {
            if (communicationCnt[4] == 1 && communicationCnt[5] == 1)
                MPI_Irecv(dptr(0, 1, 0), 1, yEdgetype2, left_neighborX, tag_XR, comm, &reqList[recvcnt++]);
            else if (communicationCnt[4] == 1)
                MPI_Irecv(dptr(0, 1, 0), 1, yEdgetype, left_neighborX, tag_XR, comm, &reqList[recvcnt++]);
            else if (communicationCnt[5] == 1)
                MPI_Irecv(dptr(0, 1, nz - 1), 1, yEdgetype, left_neighborX, tag_XR, comm, &reqList[recvcnt++]);
        }
        if (communicationCnt[1] == 1) {
            if (communicationCnt[4] == 1 && communicationCnt[5] == 1)
                MPI_Irecv(dptr(nx - 1, 1, 0), 1, yEdgetype2, right_neighborX, tag_XL, comm, &reqList[recvcnt++]);
            else if (communicationCnt[4] == 1)
                MPI_Irecv(dptr(nx - 1, 1, 0), 1, yEdgetype, right_neighborX, tag_XL, comm, &reqList[recvcnt++]);
            else if (communicationCnt[5] == 1)
                MPI_Irecv(dptr(nx - 1, 1, nz - 1), 1, yEdgetype, right_neighborX, tag_XL, comm, &reqList[recvcnt++]);
        }

        // Z-edges (along XY intersections)
        if (communicationCnt[2] == 1) {
            if (communicationCnt[0] == 1 && communicationCnt[1] == 1)
                MPI_Irecv(dptr(0, 0, 1), 1, zEdgetype2, left_neighborY, tag_YR, comm, &reqList[recvcnt++]);
            else if (communicationCnt[0] == 1)
                MPI_Irecv(dptr(0, 0, 1), 1, zEdgetype, left_neighborY, tag_YR, comm, &reqList[recvcnt++]);
            else if (communicationCnt[1] == 1)
                MPI_Irecv(dptr(nx - 1, 0, 1), 1, zEdgetype, left_neighborY, tag_YR, comm, &reqList[recvcnt++]);
        }
        if (communicationCnt[3] == 1) {
            if (communicationCnt[0] == 1 && communicationCnt[1] == 1)
                MPI_Irecv(dptr(0, ny - 1, 1), 1, zEdgetype2, right_neighborY, tag_YL, comm, &reqList[recvcnt++]);
            else if (communicationCnt[0] == 1)
                MPI_Irecv(dptr(0, ny - 1, 1), 1, zEdgetype, right_neighborY, tag_YL, comm, &reqList[recvcnt++]);
            else if (communicationCnt[1] == 1)
                MPI_Irecv(dptr(nx - 1, ny - 1, 1), 1, zEdgetype, right_neighborY, tag_YL, comm, &reqList[recvcnt++]);
        }

        // X-edges (along YZ intersections)
        if (communicationCnt[4] == 1) {
            if (communicationCnt[2] == 1 && communicationCnt[3] == 1)
                MPI_Irecv(dptr(1, 0, 0), 1, xEdgetype2, left_neighborZ, tag_ZR, comm, &reqList[recvcnt++]);
            else if (communicationCnt[2] == 1)
                MPI_Irecv(dptr(1, 0, 0), 1, xEdgetype, left_neighborZ, tag_ZR, comm, &reqList[recvcnt++]);
            else if (communicationCnt[3] == 1)
                MPI_Irecv(dptr(1, ny - 1, 0), 1, xEdgetype, left_neighborZ, tag_ZR, comm, &reqList[recvcnt++]);
        }
        if (communicationCnt[5] == 1) {
            if (communicationCnt[2] == 1 && communicationCnt[3] == 1)
                MPI_Irecv(dptr(1, 0, nz - 1), 1, xEdgetype2, right_neighborZ, tag_ZL, comm, &reqList[recvcnt++]);
            else if (communicationCnt[2] == 1)
                MPI_Irecv(dptr(1, 0, nz - 1), 1, xEdgetype, right_neighborZ, tag_ZL, comm, &reqList[recvcnt++]);
            else if (communicationCnt[3] == 1)
                MPI_Irecv(dptr(1, ny - 1, nz - 1), 1, xEdgetype, right_neighborZ, tag_ZL, comm, &reqList[recvcnt++]);
        }

        sendcnt = recvcnt;

        // Send edges
        if (communicationCnt[0] == 1) {
            if (communicationCnt[4] == 1 && communicationCnt[5] == 1)
                MPI_Isend(dptr(1, 1, 0), 1, yEdgetype2, left_neighborX, tag_XL, comm, &reqList[sendcnt++]);
            else if (communicationCnt[4] == 1)
                MPI_Isend(dptr(1, 1, 0), 1, yEdgetype, left_neighborX, tag_XL, comm, &reqList[sendcnt++]);
            else if (communicationCnt[5] == 1)
                MPI_Isend(dptr(1, 1, nz - 1), 1, yEdgetype, left_neighborX, tag_XL, comm, &reqList[sendcnt++]);
        }
        if (communicationCnt[1] == 1) {
            if (communicationCnt[4] == 1 && communicationCnt[5] == 1)
                MPI_Isend(dptr(nx - 2, 1, 0), 1, yEdgetype2, right_neighborX, tag_XR, comm, &reqList[sendcnt++]);
            else if (communicationCnt[4] == 1)
                MPI_Isend(dptr(nx - 2, 1, 0), 1, yEdgetype, right_neighborX, tag_XR, comm, &reqList[sendcnt++]);
            else if (communicationCnt[5] == 1)
                MPI_Isend(dptr(nx - 2, 1, nz - 1), 1, yEdgetype, right_neighborX, tag_XR, comm, &reqList[sendcnt++]);
        }
        if (communicationCnt[2] == 1) {
            if (communicationCnt[0] == 1 && communicationCnt[1] == 1)
                MPI_Isend(dptr(0, 1, 1), 1, zEdgetype2, left_neighborY, tag_YL, comm, &reqList[sendcnt++]);
            else if (communicationCnt[0] == 1)
                MPI_Isend(dptr(0, 1, 1), 1, zEdgetype, left_neighborY, tag_YL, comm, &reqList[sendcnt++]);
            else if (communicationCnt[1] == 1)
                MPI_Isend(dptr(nx - 1, 1, 1), 1, zEdgetype, left_neighborY, tag_YL, comm, &reqList[sendcnt++]);
        }
        if (communicationCnt[3] == 1) {
            if (communicationCnt[0] == 1 && communicationCnt[1] == 1)
                MPI_Isend(dptr(0, ny - 2, 1), 1, zEdgetype2, right_neighborY, tag_YR, comm, &reqList[sendcnt++]);
            else if (communicationCnt[0] == 1)
                MPI_Isend(dptr(0, ny - 2, 1), 1, zEdgetype, right_neighborY, tag_YR, comm, &reqList[sendcnt++]);
            else if (communicationCnt[1] == 1)
                MPI_Isend(dptr(nx - 1, ny - 2, 1), 1, zEdgetype, right_neighborY, tag_YR, comm, &reqList[sendcnt++]);
        }
        if (communicationCnt[4] == 1) {
            if (communicationCnt[2] == 1 && communicationCnt[3] == 1)
                MPI_Isend(dptr(1, 0, 1), 1, xEdgetype2, left_neighborZ, tag_ZL, comm, &reqList[sendcnt++]);
            else if (communicationCnt[2] == 1)
                MPI_Isend(dptr(1, 0, 1), 1, xEdgetype, left_neighborZ, tag_ZL, comm, &reqList[sendcnt++]);
            else if (communicationCnt[3] == 1)
                MPI_Isend(dptr(1, ny - 1, 1), 1, xEdgetype, left_neighborZ, tag_ZL, comm, &reqList[sendcnt++]);
        }
        if (communicationCnt[5] == 1) {
            if (communicationCnt[2] == 1 && communicationCnt[3] == 1)
                MPI_Isend(dptr(1, 0, nz - 2), 1, xEdgetype2, right_neighborZ, tag_ZR, comm, &reqList[sendcnt++]);
            else if (communicationCnt[2] == 1)
                MPI_Isend(dptr(1, 0, nz - 2), 1, xEdgetype, right_neighborZ, tag_ZR, comm, &reqList[sendcnt++]);
            else if (communicationCnt[3] == 1)
                MPI_Isend(dptr(1, ny - 1, nz - 2), 1, xEdgetype, right_neighborZ, tag_ZR, comm, &reqList[sendcnt++]);
        }

        // Local edge self-copy (CUDA kernels)
        {
            int maxDim = (nx > ny ? (nx > nz ? nx : nz) : (ny > nz ? ny : nz));
            int nblocks = (maxDim + 255) / 256;

            if (right_neighborX == myrank && left_neighborX == myrank) {
                gpuSelfCopyEdgeX<<<nblocks, 256, 0, stream>>>(d, nx, ny, nz,
                    right_neighborZ != MPI_PROC_NULL, left_neighborZ != MPI_PROC_NULL,
                    right_neighborY != MPI_PROC_NULL, left_neighborY != MPI_PROC_NULL);
            }
            if (right_neighborY == myrank && left_neighborY == myrank) {
                gpuSelfCopyEdgeY<<<nblocks, 256, 0, stream>>>(d, nx, ny, nz,
                    right_neighborX != MPI_PROC_NULL, left_neighborX != MPI_PROC_NULL,
                    right_neighborZ != MPI_PROC_NULL, left_neighborZ != MPI_PROC_NULL);
            }
            if (right_neighborZ == myrank && left_neighborZ == myrank) {
                gpuSelfCopyEdgeZ<<<nblocks, 256, 0, stream>>>(d, nx, ny, nz,
                    right_neighborY != MPI_PROC_NULL, left_neighborY != MPI_PROC_NULL,
                    right_neighborX != MPI_PROC_NULL, left_neighborX != MPI_PROC_NULL);
            }
            cudaStreamSynchronize(stream);
        }

        if (sendcnt > 0) {
            MPI_Waitall(sendcnt, &reqList[0], &stat[0]);
        }

        // =================================================================
        //  Phase 3: Corner exchange
        // =================================================================
        recvcnt = 0; sendcnt = 0;
        if ((communicationCnt[2] == 1 || communicationCnt[3] == 1) &&
            (communicationCnt[4] == 1 || communicationCnt[5] == 1))
        {
            if (communicationCnt[0] == 1)
                MPI_Irecv(dptr(0, 0, 0), 1, cornertype, left_neighborX, tag_XR, comm, &reqList[recvcnt++]);
            if (communicationCnt[1] == 1)
                MPI_Irecv(dptr(nx - 1, 0, 0), 1, cornertype, right_neighborX, tag_XL, comm, &reqList[recvcnt++]);

            sendcnt = recvcnt;

            if (communicationCnt[0] == 1)
                MPI_Isend(dptr(1, 0, 0), 1, cornertype, left_neighborX, tag_XL, comm, &reqList[sendcnt++]);
            if (communicationCnt[1] == 1)
                MPI_Isend(dptr(nx - 2, 0, 0), 1, cornertype, right_neighborX, tag_XR, comm, &reqList[sendcnt++]);
        }

        // Local corner self-copy
        {
            if (left_neighborX == myrank && right_neighborX == myrank) {
                gpuSelfCopyCornerX<<<1, 1, 0, stream>>>(d, nx, ny, nz,
                    left_neighborY != MPI_PROC_NULL, right_neighborY != MPI_PROC_NULL,
                    left_neighborZ != MPI_PROC_NULL, right_neighborZ != MPI_PROC_NULL);
            } else if (left_neighborY == myrank && right_neighborY == myrank) {
                gpuSelfCopyCornerY<<<1, 1, 0, stream>>>(d, nx, ny, nz,
                    left_neighborX != MPI_PROC_NULL, right_neighborX != MPI_PROC_NULL,
                    left_neighborZ != MPI_PROC_NULL, right_neighborZ != MPI_PROC_NULL);
            } else if (left_neighborZ == myrank && right_neighborZ == myrank) {
                gpuSelfCopyCornerZ<<<1, 1, 0, stream>>>(d, nx, ny, nz,
                    left_neighborY != MPI_PROC_NULL, right_neighborY != MPI_PROC_NULL,
                    left_neighborX != MPI_PROC_NULL, right_neighborX != MPI_PROC_NULL);
            }
            cudaStreamSynchronize(stream);
        }

        if (sendcnt > 0) {
            MPI_Waitall(sendcnt, &reqList[0], &stat[0]);
        }
    } // end !isFaceOnlyFlag

    // =====================================================================
    //  Additive interpolation (for moments)
    // =====================================================================
    if (needInterp) {
        dim3 block(BC_BLOCK, BC_BLOCK);
        int nxr = nx - 2, nyr = ny - 2, nzr = nz - 2;

        // addFace X,Y,Z
        {
            dim3 grid((nyr + BC_BLOCK - 1) / BC_BLOCK, (nzr + BC_BLOCK - 1) / BC_BLOCK);
            gpuAddFaceX<<<grid, block, 0, stream>>>(d, nx, ny, nz,
                vct->hasXrghtNeighbor_P(), vct->hasXleftNeighbor_P());
        }
        {
            dim3 grid((nxr + BC_BLOCK - 1) / BC_BLOCK, (nzr + BC_BLOCK - 1) / BC_BLOCK);
            gpuAddFaceY<<<grid, block, 0, stream>>>(d, nx, ny, nz,
                vct->hasYrghtNeighbor_P(), vct->hasYleftNeighbor_P());
        }
        {
            dim3 grid((nxr + BC_BLOCK - 1) / BC_BLOCK, (nyr + BC_BLOCK - 1) / BC_BLOCK);
            gpuAddFaceZ<<<grid, block, 0, stream>>>(d, nx, ny, nz,
                vct->hasZrghtNeighbor_P(), vct->hasZleftNeighbor_P());
        }

        // addEdge Z, Y, X
        {
            int nb = (nzr + 255) / 256;
            gpuAddEdgeZ<<<nb, 256, 0, stream>>>(d, nx, ny, nz,
                vct->hasXrghtNeighbor_P(), vct->hasXleftNeighbor_P(),
                vct->hasYrghtNeighbor_P(), vct->hasYleftNeighbor_P());
        }
        {
            int nb = (nyr + 255) / 256;
            gpuAddEdgeY<<<nb, 256, 0, stream>>>(d, nx, ny, nz,
                vct->hasXrghtNeighbor_P(), vct->hasXleftNeighbor_P(),
                vct->hasZrghtNeighbor_P(), vct->hasZleftNeighbor_P());
        }
        {
            int nb = (nxr + 255) / 256;
            gpuAddEdgeX<<<nb, 256, 0, stream>>>(d, nx, ny, nz,
                vct->hasYrghtNeighbor_P(), vct->hasYleftNeighbor_P(),
                vct->hasZrghtNeighbor_P(), vct->hasZleftNeighbor_P());
        }

        // addCorner
        gpuAddCorner<<<1, 1, 0, stream>>>(d, nx, ny, nz,
            vct->hasXrghtNeighbor_P(), vct->hasXleftNeighbor_P(),
            vct->hasYrghtNeighbor_P(), vct->hasYleftNeighbor_P(),
            vct->hasZrghtNeighbor_P(), vct->hasZleftNeighbor_P());

        cudaStreamSynchronize(stream);
    }

    callCount++;
}

// =========================================================================
//  EMfields3D GPU halo exchange wrapper methods
//
//  All wrappers now route through gpuBatchedHaloExchange (nFields=1)
//  which uses explicit CUDA pack/unpack + contiguous MPI instead of
//  MPI derived datatypes on device memory.
// =========================================================================

void EMfields3D::gpuCommunicateNodeBC(int nx, int ny, int nz, GPUFieldArray3& gpuArr,
                                       int bcFaceXright, int bcFaceXleft,
                                       int bcFaceYright, int bcFaceYleft,
                                       int bcFaceZright, int bcFaceZleft)
{
    double* ptr = gpuArr.devPtr();
    gpuBatchedHaloExchange(&ptr, 1, nx, ny, nz, false, false, false, false, solverStream_);
    gpuBCface(nx, ny, nz, gpuArr, bcFaceXright, bcFaceXleft, bcFaceYright, bcFaceYleft, bcFaceZright, bcFaceZleft, &_vct, solverStream_);
}

void EMfields3D::gpuCommunicateCenterBC(int nx, int ny, int nz, GPUFieldArray3& gpuArr,
                                         int bcFaceXright, int bcFaceXleft,
                                         int bcFaceYright, int bcFaceYleft,
                                         int bcFaceZright, int bcFaceZleft)
{
    double* ptr = gpuArr.devPtr();
    gpuBatchedHaloExchange(&ptr, 1, nx, ny, nz, true, false, false, false, solverStream_);
    gpuBCface(nx, ny, nz, gpuArr, bcFaceXright, bcFaceXleft, bcFaceYright, bcFaceYleft, bcFaceZright, bcFaceZleft, &_vct, solverStream_);
}

void EMfields3D::gpuCommunicateNodeBoxStencilBC(int nx, int ny, int nz, GPUFieldArray3& gpuArr,
                                                 int bcFaceXright, int bcFaceXleft,
                                                 int bcFaceYright, int bcFaceYleft,
                                                 int bcFaceZright, int bcFaceZleft)
{
    double* ptr = gpuArr.devPtr();
    gpuBatchedHaloExchange(&ptr, 1, nx, ny, nz, false, true, false, false, solverStream_);
    gpuBCface(nx, ny, nz, gpuArr, bcFaceXright, bcFaceXleft, bcFaceYright, bcFaceYleft, bcFaceZright, bcFaceZleft, &_vct, solverStream_);
}

void EMfields3D::gpuCommunicateCenterBC_P(int nx, int ny, int nz, GPUFieldArray3& gpuArr,
                                           int bcFaceXright, int bcFaceXleft,
                                           int bcFaceYright, int bcFaceYleft,
                                           int bcFaceZright, int bcFaceZleft)
{
    double* ptr = gpuArr.devPtr();
    gpuBatchedHaloExchange(&ptr, 1, nx, ny, nz, true, false, false, true, solverStream_);
    gpuBCface_P(nx, ny, nz, gpuArr, bcFaceXright, bcFaceXleft, bcFaceYright, bcFaceYleft, bcFaceZright, bcFaceZleft, &_vct, solverStream_);
}

void EMfields3D::gpuCommunicateInterp(int nx, int ny, int nz, GPUFieldArray3& gpuArr)
{
    double* ptr = gpuArr.devPtr();
    gpuBatchedHaloExchange(&ptr, 1, nx, ny, nz, true, false, true, true, solverStream_);
}

void EMfields3D::gpuCommunicateNode_P(int nx, int ny, int nz, GPUFieldArray3& gpuArr)
{
    double* ptr = gpuArr.devPtr();
    gpuBatchedHaloExchange(&ptr, 1, nx, ny, nz, false, false, false, true, solverStream_);
}

void EMfields3D::gpuCommunicateNodeBoxStencilBC_P(int nx, int ny, int nz, GPUFieldArray3& gpuArr,
                                                   int bcFaceXright, int bcFaceXleft,
                                                   int bcFaceYright, int bcFaceYleft,
                                                   int bcFaceZright, int bcFaceZleft)
{
    double* ptr = gpuArr.devPtr();
    gpuBatchedHaloExchange(&ptr, 1, nx, ny, nz, false, true, false, true, solverStream_);
    gpuBCface_P(nx, ny, nz, gpuArr, bcFaceXright, bcFaceXleft, bcFaceYright, bcFaceYleft, bcFaceZright, bcFaceZleft, &_vct, solverStream_);
}

void EMfields3D::gpuCommunicateNodeBoxStencilBC_P_3(int nx, int ny, int nz,
                                                    GPUFieldArray3& a1, GPUFieldArray3& a2, GPUFieldArray3& a3,
                                                    int bcXR, int bcXL, int bcYR, int bcYL, int bcZR, int bcZL)
{
    double* ptrs[3] = { a1.devPtr(), a2.devPtr(), a3.devPtr() };
    gpuBatchedHaloExchange(ptrs, 3, nx, ny, nz, false, true, false, true, solverStream_);
    gpuBCface_P(nx, ny, nz, a1, bcXR, bcXL, bcYR, bcYL, bcZR, bcZL, &_vct, solverStream_);
    gpuBCface_P(nx, ny, nz, a2, bcXR, bcXL, bcYR, bcYL, bcZR, bcZL, &_vct, solverStream_);
    gpuBCface_P(nx, ny, nz, a3, bcXR, bcXL, bcYR, bcYL, bcZR, bcZL, &_vct, solverStream_);
}

void EMfields3D::gpuCommunicateCenterBoxStencilBC_P(int nx, int ny, int nz, GPUFieldArray3& gpuArr,
                                                     int bcFaceXright, int bcFaceXleft,
                                                     int bcFaceYright, int bcFaceYleft,
                                                     int bcFaceZright, int bcFaceZleft)
{
    double* ptr = gpuArr.devPtr();
    gpuBatchedHaloExchange(&ptr, 1, nx, ny, nz, true, true, false, true, solverStream_);
    gpuBCface_P(nx, ny, nz, gpuArr, bcFaceXright, bcFaceXleft, bcFaceYright, bcFaceYleft, bcFaceZright, bcFaceZleft, &_vct, solverStream_);
}

void EMfields3D::gpuCommunicateCenterBoxStencilBC_P_3(int nx, int ny, int nz,
                                                      GPUFieldArray3& a1, GPUFieldArray3& a2, GPUFieldArray3& a3,
                                                      int bcXR, int bcXL, int bcYR, int bcYL, int bcZR, int bcZL)
{
    double* ptrs[3] = { a1.devPtr(), a2.devPtr(), a3.devPtr() };
    gpuBatchedHaloExchange(ptrs, 3, nx, ny, nz, true, true, false, true, solverStream_);
    gpuBCface_P(nx, ny, nz, a1, bcXR, bcXL, bcYR, bcYL, bcZR, bcZL, &_vct, solverStream_);
    gpuBCface_P(nx, ny, nz, a2, bcXR, bcXL, bcYR, bcYL, bcZR, bcZL, &_vct, solverStream_);
    gpuBCface_P(nx, ny, nz, a3, bcXR, bcXL, bcYR, bcYL, bcZR, bcZL, &_vct, solverStream_);
}

void EMfields3D::gpuCommunicateCenterBoxStencilBC(int nx, int ny, int nz, GPUFieldArray3& gpuArr,
                                                   int bcFaceXright, int bcFaceXleft,
                                                   int bcFaceYright, int bcFaceYleft,
                                                   int bcFaceZright, int bcFaceZleft)
{
    double* ptr = gpuArr.devPtr();
    gpuBatchedHaloExchange(&ptr, 1, nx, ny, nz, true, true, false, false, solverStream_);
    gpuBCface(nx, ny, nz, gpuArr, bcFaceXright, bcFaceXleft, bcFaceYright, bcFaceYleft, bcFaceZright, bcFaceZleft, &_vct, solverStream_);
}

// =========================================================================
//  Batched 3-field communicate wrappers
//
//  These perform ONE batched halo exchange for 3 fields at once (nFields=3)
//  instead of 3 sequential exchanges, eliminating 2 complete rounds of
//  cudaStreamSynchronize + MPI_Waitall barriers per call.
// =========================================================================

void EMfields3D::gpuCommunicateNodeBC_3(int nx, int ny, int nz,
                                         GPUFieldArray3& a1, GPUFieldArray3& a2, GPUFieldArray3& a3,
                                         int bcXR, int bcXL, int bcYR, int bcYL, int bcZR, int bcZL)
{
    double* ptrs[3] = { a1.devPtr(), a2.devPtr(), a3.devPtr() };
    gpuBatchedHaloExchange(ptrs, 3, nx, ny, nz, false, false, false, false, solverStream_);
    gpuBCface(nx, ny, nz, a1, bcXR, bcXL, bcYR, bcYL, bcZR, bcZL, &_vct, solverStream_);
    gpuBCface(nx, ny, nz, a2, bcXR, bcXL, bcYR, bcYL, bcZR, bcZL, &_vct, solverStream_);
    gpuBCface(nx, ny, nz, a3, bcXR, bcXL, bcYR, bcYL, bcZR, bcZL, &_vct, solverStream_);
}

void EMfields3D::gpuCommunicateCenterBC_3(int nx, int ny, int nz,
                                           GPUFieldArray3& a1, GPUFieldArray3& a2, GPUFieldArray3& a3,
                                           int bcXR, int bcXL, int bcYR, int bcYL, int bcZR, int bcZL)
{
    double* ptrs[3] = { a1.devPtr(), a2.devPtr(), a3.devPtr() };
    gpuBatchedHaloExchange(ptrs, 3, nx, ny, nz, true, false, false, false, solverStream_);
    gpuBCface(nx, ny, nz, a1, bcXR, bcXL, bcYR, bcYL, bcZR, bcZL, &_vct, solverStream_);
    gpuBCface(nx, ny, nz, a2, bcXR, bcXL, bcYR, bcYL, bcZR, bcZL, &_vct, solverStream_);
    gpuBCface(nx, ny, nz, a3, bcXR, bcXL, bcYR, bcYL, bcZR, bcZL, &_vct, solverStream_);
}

void EMfields3D::gpuCommunicateCenterBC_P_3(int nx, int ny, int nz,
                                             GPUFieldArray3& a1, GPUFieldArray3& a2, GPUFieldArray3& a3,
                                             int bcXR, int bcXL, int bcYR, int bcYL, int bcZR, int bcZL)
{
    double* ptrs[3] = { a1.devPtr(), a2.devPtr(), a3.devPtr() };
    gpuBatchedHaloExchange(ptrs, 3, nx, ny, nz, true, false, false, true, solverStream_);
    gpuBCface_P(nx, ny, nz, a1, bcXR, bcXL, bcYR, bcYL, bcZR, bcZL, &_vct, solverStream_);
    gpuBCface_P(nx, ny, nz, a2, bcXR, bcXL, bcYR, bcYL, bcZR, bcZL, &_vct, solverStream_);
    gpuBCface_P(nx, ny, nz, a3, bcXR, bcXL, bcYR, bcYL, bcZR, bcZL, &_vct, solverStream_);
}

void EMfields3D::gpuCommunicateNodeBoxStencilBC_3(int nx, int ny, int nz,
                                                   GPUFieldArray3& a1, GPUFieldArray3& a2, GPUFieldArray3& a3,
                                                   int bcXR, int bcXL, int bcYR, int bcYL, int bcZR, int bcZL)
{
    double* ptrs[3] = { a1.devPtr(), a2.devPtr(), a3.devPtr() };
    gpuBatchedHaloExchange(ptrs, 3, nx, ny, nz, false, true, false, false, solverStream_);
    gpuBCface(nx, ny, nz, a1, bcXR, bcXL, bcYR, bcYL, bcZR, bcZL, &_vct, solverStream_);
    gpuBCface(nx, ny, nz, a2, bcXR, bcXL, bcYR, bcYL, bcZR, bcZL, &_vct, solverStream_);
    gpuBCface(nx, ny, nz, a3, bcXR, bcXL, bcYR, bcYL, bcZR, bcZL, &_vct, solverStream_);
}

void EMfields3D::gpuCommunicateCenterBC_3mixed(int nx, int ny, int nz,
    GPUFieldArray3& a1, const int* bc1,
    GPUFieldArray3& a2, const int* bc2,
    GPUFieldArray3& a3, const int* bc3)
{
    double* ptrs[3] = { a1.devPtr(), a2.devPtr(), a3.devPtr() };
    gpuBatchedHaloExchange(ptrs, 3, nx, ny, nz, true, false, false, false, solverStream_);
    gpuBCface(nx, ny, nz, a1, bc1[0], bc1[1], bc1[2], bc1[3], bc1[4], bc1[5], &_vct, solverStream_);
    gpuBCface(nx, ny, nz, a2, bc2[0], bc2[1], bc2[2], bc2[3], bc2[4], bc2[5], &_vct, solverStream_);
    gpuBCface(nx, ny, nz, a3, bc3[0], bc3[1], bc3[2], bc3[3], bc3[4], bc3[5], &_vct, solverStream_);
}

void EMfields3D::gpuCommunicateNodeBC_3mixed(int nx, int ny, int nz,
    GPUFieldArray3& a1, const int* bc1,
    GPUFieldArray3& a2, const int* bc2,
    GPUFieldArray3& a3, const int* bc3)
{
    double* ptrs[3] = { a1.devPtr(), a2.devPtr(), a3.devPtr() };
    gpuBatchedHaloExchange(ptrs, 3, nx, ny, nz, false, false, false, false, solverStream_);
    gpuBCface(nx, ny, nz, a1, bc1[0], bc1[1], bc1[2], bc1[3], bc1[4], bc1[5], &_vct, solverStream_);
    gpuBCface(nx, ny, nz, a2, bc2[0], bc2[1], bc2[2], bc2[3], bc2[4], bc2[5], &_vct, solverStream_);
    gpuBCface(nx, ny, nz, a3, bc3[0], bc3[1], bc3[2], bc3[3], bc3[4], bc3[5], &_vct, solverStream_);
}

void EMfields3D::gpuCommunicateNodeBoxStencilBC_3mixed(int nx, int ny, int nz,
    GPUFieldArray3& a1, const int* bc1,
    GPUFieldArray3& a2, const int* bc2,
    GPUFieldArray3& a3, const int* bc3)
{
    double* ptrs[3] = { a1.devPtr(), a2.devPtr(), a3.devPtr() };
    gpuBatchedHaloExchange(ptrs, 3, nx, ny, nz, false, true, false, false, solverStream_);
    gpuBCface(nx, ny, nz, a1, bc1[0], bc1[1], bc1[2], bc1[3], bc1[4], bc1[5], &_vct, solverStream_);
    gpuBCface(nx, ny, nz, a2, bc2[0], bc2[1], bc2[2], bc2[3], bc2[4], bc2[5], &_vct, solverStream_);
    gpuBCface(nx, ny, nz, a3, bc3[0], bc3[1], bc3[2], bc3[3], bc3[4], bc3[5], &_vct, solverStream_);
}

void EMfields3D::gpuCommunicateCenterBC_9(int nx, int ny, int nz,
                                           GPUFieldArray3& a1, GPUFieldArray3& a2, GPUFieldArray3& a3,
                                           GPUFieldArray3& a4, GPUFieldArray3& a5, GPUFieldArray3& a6,
                                           GPUFieldArray3& a7, GPUFieldArray3& a8, GPUFieldArray3& a9,
                                           int bcXR, int bcXL, int bcYR, int bcYL, int bcZR, int bcZL)
{
    double* ptrs[9] = { a1.devPtr(), a2.devPtr(), a3.devPtr(),
                        a4.devPtr(), a5.devPtr(), a6.devPtr(),
                        a7.devPtr(), a8.devPtr(), a9.devPtr() };
    gpuBatchedHaloExchange(ptrs, 9, nx, ny, nz, true, false, false, false, solverStream_);
    gpuBCface(nx, ny, nz, a1, bcXR, bcXL, bcYR, bcYL, bcZR, bcZL, &_vct, solverStream_);
    gpuBCface(nx, ny, nz, a2, bcXR, bcXL, bcYR, bcYL, bcZR, bcZL, &_vct, solverStream_);
    gpuBCface(nx, ny, nz, a3, bcXR, bcXL, bcYR, bcYL, bcZR, bcZL, &_vct, solverStream_);
    gpuBCface(nx, ny, nz, a4, bcXR, bcXL, bcYR, bcYL, bcZR, bcZL, &_vct, solverStream_);
    gpuBCface(nx, ny, nz, a5, bcXR, bcXL, bcYR, bcYL, bcZR, bcZL, &_vct, solverStream_);
    gpuBCface(nx, ny, nz, a6, bcXR, bcXL, bcYR, bcYL, bcZR, bcZL, &_vct, solverStream_);
    gpuBCface(nx, ny, nz, a7, bcXR, bcXL, bcYR, bcYL, bcZR, bcZL, &_vct, solverStream_);
    gpuBCface(nx, ny, nz, a8, bcXR, bcXL, bcYR, bcYL, bcZR, bcZL, &_vct, solverStream_);
    gpuBCface(nx, ny, nz, a9, bcXR, bcXL, bcYR, bcYL, bcZR, bcZL, &_vct, solverStream_);
}

#endif // GPU_SOLVER
