/**
 * @file GPUFieldPacking.cu
 * @brief GPU field packing for the particle mover.  See
 *        GPUFieldPacking.cuh for documentation.
 */

#include "GPUFieldPacking.cuh"

// =========================================================================
//  gpuPackFieldForPclsToCenter
// =========================================================================
__global__ void gpuPackFieldForPclsToCenter(
    cudaFieldType* __restrict__ out, const cudaSolverType* __restrict__ Ex,
    const cudaSolverType* __restrict__ Ey,
    const cudaSolverType* __restrict__ Ez,
    const cudaSolverType* __restrict__ Bxn,
    const cudaSolverType* __restrict__ Byn,
    const cudaSolverType* __restrict__ Bzn,
    const cudaSolverType* __restrict__ Bx_ext,
    const cudaSolverType* __restrict__ By_ext,
    const cudaSolverType* __restrict__ Bz_ext, int nxn, int nyn, int nzn) {
  constexpr int W = 24;    // values per cell
  constexpr int P = W + 1; // padded row: conflict-free shared-memory writes
  __shared__ cudaFieldType tile[kPackFieldBlock * P];
  const int ncells = (nxn - 1) * (nyn - 1) * nzn;
  const int first = blockIdx.x * kPackFieldBlock; // first cell of the block
  const int cell = first + threadIdx.x;           // = cellIndex of the layout

  if (cell < ncells) {
    const int k = cell % nzn;
    const int j = (cell / nzn) % (nyn - 1);
    const int i = cell / (nzn * (nyn - 1));
    // grid points (i, j, k), (i+1, j, k), (i+1, j+1, k), (i, j+1, k)
    const int s0 = (i * nyn + j) * nzn + k;
    const int s[4] = {s0, s0 + nyn * nzn, s0 + (nyn + 1) * nzn, s0 + nzn};
    cudaFieldType* const row = tile + threadIdx.x * P;
#pragma unroll
    for (int p = 0; p < 4; p++) {
      row[6 * p + 0] = (cudaFieldType)(Bxn[s[p]] + Bx_ext[s[p]]);
      row[6 * p + 1] = (cudaFieldType)(Byn[s[p]] + By_ext[s[p]]);
      row[6 * p + 2] = (cudaFieldType)(Bzn[s[p]] + Bz_ext[s[p]]);
      row[6 * p + 3] = (cudaFieldType)Ex[s[p]];
      row[6 * p + 4] = (cudaFieldType)Ey[s[p]];
      row[6 * p + 5] = (cudaFieldType)Ez[s[p]];
    }
  }
  __syncthreads();

  // The cells of the block are consecutive in out: consecutive threads
  // write their values, coalesced.
  const int left = ncells - first;
  const int cells = left < kPackFieldBlock ? left : kPackFieldBlock;
  cudaFieldType* const dst = out + (size_t)first * W;
  for (int e = threadIdx.x; e < cells * W; e += kPackFieldBlock)
    dst[e] = tile[(e / W) * P + e % W];
}
