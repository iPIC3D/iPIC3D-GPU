/**
 * @file GPUFieldPacking.cuh
 * @brief GPU kernel for packing field arrays into the cell-centered layout
 *        used by the particle mover.
 *
 * When GPU_SOLVER is enabled, the E and B fields already reside on the GPU
 * in standard node-based GPUFieldArray3 arrays.  Instead of copying them
 * to the host, packing on the CPU (set_fieldForPclsToCenter), and
 * transferring the packed buffer back, this kernel performs the packing
 * entirely on the GPU — eliminating one D2H and one H2D transfer per cycle.
 *
 * The output layout matches exactly what set_fieldForPclsToCenter produces:
 *   out[cellIndex * 24 + nodeOffset * 6 + fieldOffset]
 * where cellIndex = (i * (nyn-1) + j) * nzn + k,
 *       nodeOffset ∈ {0,1,2,3}  for grid points (i,j,k), (i+1,j,k),
 *                                (i+1,j+1,k), (i,j+1,k),
 *       fieldOffset ∈ {0..5}   for Bx+Bx_ext, By+By_ext, Bz+Bz_ext,
 *                                Ex, Ey, Ez.
 *
 * Compile with -DGPU_SOLVER to activate.
 */

#ifndef GPU_FIELD_PACKING_CUH
#define GPU_FIELD_PACKING_CUH

#include "cudaTypeDef.cuh"

#ifdef GPU_SOLVER

/**
 * @brief Pack node-based E, B, B_ext into the cell-centered mover buffer.
 *
 * One thread per cell.  Total cells = (nxn-1) * (nyn-1) * nzn.
 *
 * @param out        Output packed buffer  [ncells * 24]
 * @param Ex,Ey,Ez   Node-based E fields   [nxn * nyn * nzn]
 * @param Bxn,Byn,Bzn Node-based B fields  [nxn * nyn * nzn]
 * @param Bx_ext,By_ext,Bz_ext  External B [nxn * nyn * nzn]
 * @param nxn,nyn,nzn  Node dimensions (including ghosts)
 */
__global__ void gpuPackFieldForPclsToCenter(
    cudaFieldType* __restrict__ out,
    const double*  __restrict__ Ex,      const double* __restrict__ Ey,      const double* __restrict__ Ez,
    const double*  __restrict__ Bxn,     const double* __restrict__ Byn,     const double* __restrict__ Bzn,
    const double*  __restrict__ Bx_ext,  const double* __restrict__ By_ext,  const double* __restrict__ Bz_ext,
    int nxn, int nyn, int nzn);

#endif // GPU_SOLVER

#endif // GPU_FIELD_PACKING_CUH
