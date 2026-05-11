#ifndef _PARTICLE_CONTROL_KERNEL_CUH_
#define _PARTICLE_CONTROL_KERNEL_CUH_


#include "cudaTypeDef.cuh"
#include "particleControlKernel.cuh"

#include "gridCUDA.cuh"
#include "particleArrayCUDA.cuh"
#include "particleExchange.cuh"

#include "gridCUDA.cuh"
#include "particleExchange.cuh"
#include "hashedSum.cuh"
#include "moverKernel.cuh"

/**
 * @brief Merge near-duplicate particles inside each cell of a sorted SoA array.
 *
 * The input particle array must already be cell-sorted. One warp processes one
 * cell and marks merged-away particles in the departure array.
 *
 * @param cellOffsetList Device array of per-cell particle offsets.
 * @param cellBinCountList Device array of per-cell particle counts.
 * @param grid Device-side grid descriptor.
 * @param pclArray Device-side particle SoA container.
 * @param departureArray Device-side departure metadata array.
 */
__global__ void mergingKernel(int* cellOffsetList, int* cellBinCountList, grid3DCUDA* grid, particleArrayCUDA* pclArray, departureArrayType* departureArray);

template <bool MULTIPLE>
/**
 * @brief Split particles to recover the target particle count for a species.
 *
 * `MULTIPLE == false` handles the case where fewer new particles than current
 * particles are needed. `MULTIPLE == true` handles repeated splitting when the
 * deficit exceeds the current particle count.
 *
 * @param moverParam Device-side mover parameter bundle for one species.
 * @param grid Device-side grid descriptor.
 */
__global__ void particleSplittingKernel(moverParameter* moverParam, grid3DCUDA* grid);

#endif // _PARTICLE_CONTROL_KERNEL_CUH_
