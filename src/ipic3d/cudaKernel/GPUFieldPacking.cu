/**
 * @file GPUFieldPacking.cu
 * @brief CUDA kernel implementations for GPU field packing and moment
 *        unpacking.  See GPUFieldPacking.cuh for documentation.
 */

#include "GPUFieldPacking.cuh"

#ifdef GPU_SOLVER

// =========================================================================
//  gpuPackFieldForPclsToCenter
// =========================================================================
__global__ void gpuPackFieldForPclsToCenter(
    cudaFieldType* __restrict__ out,
    const double*  __restrict__ Ex,      const double* __restrict__ Ey,      const double* __restrict__ Ez,
    const double*  __restrict__ Bxn,     const double* __restrict__ Byn,     const double* __restrict__ Bzn,
    const double*  __restrict__ Bx_ext,  const double* __restrict__ By_ext,  const double* __restrict__ Bz_ext,
    int nxn, int nyn, int nzn)
{
    const int ncells = (nxn - 1) * (nyn - 1) * nzn;
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= ncells) return;

    // Decompose flat index into (i, j, k) cell coordinates
    const int k = idx % nzn;
    const int j = (idx / nzn) % (nyn - 1);
    const int i = idx / (nzn * (nyn - 1));

    // Source index helper: row-major [nxn][nyn][nzn]
    #define SRC(ii, jj, kk) ((ii) * nyn * nzn + (jj) * nzn + (kk))

    const int cellIndex = (i * (nyn - 1) + j) * nzn + k;
    cudaFieldType* dst = out + cellIndex * 24;

    // ---- Grid point 0: (i, j, k) ----
    {
        const int s = SRC(i, j, k);
        dst[ 0] = (cudaFieldType)(Bxn[s] + Bx_ext[s]);
        dst[ 1] = (cudaFieldType)(Byn[s] + By_ext[s]);
        dst[ 2] = (cudaFieldType)(Bzn[s] + Bz_ext[s]);
        dst[ 3] = (cudaFieldType)Ex[s];
        dst[ 4] = (cudaFieldType)Ey[s];
        dst[ 5] = (cudaFieldType)Ez[s];
    }

    // ---- Grid point 1: (i+1, j, k) ----
    {
        const int s = SRC(i + 1, j, k);
        dst[ 6] = (cudaFieldType)(Bxn[s] + Bx_ext[s]);
        dst[ 7] = (cudaFieldType)(Byn[s] + By_ext[s]);
        dst[ 8] = (cudaFieldType)(Bzn[s] + Bz_ext[s]);
        dst[ 9] = (cudaFieldType)Ex[s];
        dst[10] = (cudaFieldType)Ey[s];
        dst[11] = (cudaFieldType)Ez[s];
    }

    // ---- Grid point 2: (i+1, j+1, k) ----
    {
        const int s = SRC(i + 1, j + 1, k);
        dst[12] = (cudaFieldType)(Bxn[s] + Bx_ext[s]);
        dst[13] = (cudaFieldType)(Byn[s] + By_ext[s]);
        dst[14] = (cudaFieldType)(Bzn[s] + Bz_ext[s]);
        dst[15] = (cudaFieldType)Ex[s];
        dst[16] = (cudaFieldType)Ey[s];
        dst[17] = (cudaFieldType)Ez[s];
    }

    // ---- Grid point 3: (i, j+1, k) ----
    {
        const int s = SRC(i, j + 1, k);
        dst[18] = (cudaFieldType)(Bxn[s] + Bx_ext[s]);
        dst[19] = (cudaFieldType)(Byn[s] + By_ext[s]);
        dst[20] = (cudaFieldType)(Bzn[s] + Bz_ext[s]);
        dst[21] = (cudaFieldType)Ex[s];
        dst[22] = (cudaFieldType)Ey[s];
        dst[23] = (cudaFieldType)Ez[s];
    }

    #undef SRC
}

#endif // GPU_SOLVER
