#ifndef MOMENT_KERNEL_CUH
#define MOMENT_KERNEL_CUH

#include "cudaTypeDef.cuh"
#include "gridCUDA.cuh"
#include "particleArrayCUDA.cuh"
#include "particleExchange.cuh"

/**
 * @brief Deposit moments for the stayed-particle prefix after the mover.
 *
 * Particles already marked as exiting/deleted are skipped. `appendCount`
 * extends the valid prefix when open-boundary duplication has appended extra
 * particles.
 *
 * @param pclsArray Device-side particle SoA container.
 * @param departureArray Device-side departure flags.
 * @param appendCount Device pointer to the number of appended particles.
 * @param grid The grid descriptor.
 * @param moments Packed moment output buffer for this species.
 */
__global__ void momentKernelStayed(particleArrayCUDA* pclsArray,
                                   departureArrayType* departureArray,
                                   const uint32_t* appendCount,
                                   const KERNEL_CONSTANT grid3DCUDA grid,
                                   cudaTypeArray1<cudaMomentType> moments);

/**
 * @brief Deposit moments for a contiguous particle tail `[stayedParticle,
 * NOP)`.
 *
 * This is used for full moment recomputation and for appending incoming
 * particles after MPI exchange and exosphere injection.
 *
 * @param pclsArray Device-side particle SoA container.
 * @param grid The grid descriptor.
 * @param moments Packed moment output buffer for this species.
 * @param stayedParticle Start index of the unsorted tail to deposit.
 */
__global__ void momentKernelNew(particleArrayCUDA* pclsArray,
                                const KERNEL_CONSTANT grid3DCUDA grid,
                                cudaTypeArray1<cudaMomentType> moments,
                                int stayedParticle);

/**
 * @brief Deposit moments for a cell-sorted particle prefix using one warp per
 * cell.
 *
 * All particles in `[cell_start_offsets[c], cell_start_offsets[c+1])` belong to
 * the same cell. The warp reduces the per-particle contributions in registers
 * and emits one atomic accumulation per `(moment,node)` pair for that cell.
 *
 * @param cell_start_offsets Device array of per-cell particle-start offsets.
 * @param num_cells Cells of the grid, ghosts included (nxc nyc nzc), one warp
 *        each.
 * @param num_to_sort Number of particles in the sorted prefix.
 * @param pclsArray Device-side particle SoA container.
 * @param grid The grid descriptor.
 * @param moments Packed moment output buffer for this species.
 */
__global__ void
cellAwareMomentKernel(const int* __restrict__ cell_start_offsets, int num_cells,
                      uint32_t num_to_sort, particleArrayCUDA* pclsArray,
                      const KERNEL_CONSTANT grid3DCUDA grid,
                      cudaTypeArray1<cudaMomentType> moments);

/**
 * @brief Deposit the third central velocity moment for an unsorted species.
 *
 * The bulk buffer is packed as [rho, Jx, Jy, Jz] on the nodal grid using
 * already-communicated lower moments. The output buffer stores the symmetric
 * heat-flux components in the order:
 * Qxxx,Qxxy,Qxxz,Qxyy,Qxyz,Qxzz,Qyyy,Qyyz,Qyzz,Qzzz.
 *
 * @param pclsArray Device-side particle SoA container.
 * @param grid The grid descriptor.
 * @param bulkMoments Packed lower moments [4][nxn][nyn][nzn].
 * @param heatFlux Packed heat-flux output [10][nxn][nyn][nzn].
 * @param qom Species charge-to-mass ratio.
 * @param rhoFloor Absolute charge-density floor for bulk-velocity division.
 */
__global__ void
heatFluxKernelUnsorted(particleArrayCUDA* pclsArray,
                       const KERNEL_CONSTANT grid3DCUDA grid,
                       const cudaTypeArray1<cudaMomentType> bulkMoments,
                       cudaTypeArray1<cudaMomentType> heatFlux,
                       cudaMomentType qom, cudaMomentType rhoFloor);

#endif // MOMENT_KERNEL_CUH
