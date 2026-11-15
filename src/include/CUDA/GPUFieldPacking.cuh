/**
 * @file GPUFieldPacking.cuh
 * @brief GPU kernel packing E, B and B_ext into the cell-centered buffer read
 *        by the particle mover.
 *
 * Layout:
 *   out[cellIndex * 24 + nodeOffset * 6 + fieldOffset]
 * where cellIndex = (i * (nyn-1) + j) * nzn + k,
 *       nodeOffset ∈ {0,1,2,3}  for grid points (i,j,k), (i+1,j,k),
 *                                (i+1,j+1,k), (i,j+1,k),
 *       fieldOffset ∈ {0..5}   for Bx+Bx_ext, By+By_ext, Bz+Bz_ext,
 *                                Ex, Ey, Ez.
 */

#ifndef GPU_FIELD_PACKING_CUH
#define GPU_FIELD_PACKING_CUH

#include "cudaTypeDef.cuh"

/** Threads per block of gpuPackFieldForPclsToCenter (its shared tile holds
 *  kPackFieldBlock cells). */
constexpr int kPackFieldBlock = 128;

/**
 * @brief Pack node-based E, B, B_ext into the cell-centered mover buffer.
 *
 * One thread per cell, kPackFieldBlock threads per block.  Total cells =
 * (nxn-1) * (nyn-1) * nzn.  A block gathers its cells into shared memory
 * and writes their contiguous range of out with consecutive threads.
 *
 * @param out        Output packed buffer  [ncells * 24]
 * @param Ex,Ey,Ez   Node-based E fields   [nxn * nyn * nzn]
 * @param Bxn,Byn,Bzn Node-based B fields  [nxn * nyn * nzn]
 * @param Bx_ext,By_ext,Bz_ext  External B [nxn * nyn * nzn]
 * @param nxn,nyn,nzn  Node dimensions (including ghosts)
 */
__global__ void gpuPackFieldForPclsToCenter(
    cudaFieldType* __restrict__ out, const cudaSolverType* __restrict__ Ex,
    const cudaSolverType* __restrict__ Ey,
    const cudaSolverType* __restrict__ Ez,
    const cudaSolverType* __restrict__ Bxn,
    const cudaSolverType* __restrict__ Byn,
    const cudaSolverType* __restrict__ Bzn,
    const cudaSolverType* __restrict__ Bx_ext,
    const cudaSolverType* __restrict__ By_ext,
    const cudaSolverType* __restrict__ Bz_ext, int nxn, int nyn, int nzn);

#endif // GPU_FIELD_PACKING_CUH
