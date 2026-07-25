#ifndef MOMENT_KERNEL_CUH
#define MOMENT_KERNEL_CUH

#include "cudaTypeDef.cuh"
#include "gridCUDA.cuh"
#include "particleArrayCUDA.cuh"
#include "particleExchange.cuh"

/**
 * @brief Lightweight parameter bundle passed to GPU moment kernels.
 *
 * Holds device pointers to the active particle SoA container and the departure
 * flags needed to skip particles that already left the local stayed prefix.
 */
class momentParameter {

public:
  particleArrayCUDA* pclsArray;       // main device particle array
  departureArrayType* departureArray; // device departure flags

  /**
   * @brief Construct from device-resident particle and departure buffers.
   *
   * @param pclsArrayCUDAPtr      Device pointer to the active particle SoA
   * container.
   * @param departureArrayCUDAPtr Device pointer to the departure-mark array.
   */
  __host__ momentParameter(particleArrayCUDA* pclsArrayCUDAPtr,
                           departureArrayType* departureArrayCUDAPtr) {
    pclsArray = pclsArrayCUDAPtr;
    departureArray = departureArrayCUDAPtr;
  }
};
/**
 * @brief Deposit moments for the stayed-particle prefix after the mover.
 *
 * Particles already marked as exiting/deleted are skipped. `appendCount`
 * extends the valid prefix when open-boundary duplication has appended extra
 * particles.
 *
 * @param appendCount Device pointer to the number of appended particles.
 * @param momentParam Device-side moment parameter bundle for one species.
 * @param grid Device-side grid descriptor.
 * @param moments Packed moment output buffer for this species.
 */
__global__ void momentKernelStayed(const uint32_t* appendCount,
                                   momentParameter* momentParam,
                                   grid3DCUDA* grid,
                                   cudaTypeArray1<cudaMomentType> moments);

/**
 * @brief Deposit moments for a contiguous particle tail `[stayedParticle,
 * NOP)`.
 *
 * This is used for full moment recomputation and for appending incoming
 * particles after MPI exchange and exosphere injection.
 *
 * @param momentParam Device-side moment parameter bundle for one species.
 * @param grid Device-side grid descriptor.
 * @param moments Packed moment output buffer for this species.
 * @param stayedParticle Start index of the unsorted tail to deposit.
 */
__global__ void momentKernelNew(momentParameter* momentParam, grid3DCUDA* grid,
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
 * @param num_cells Number of populated cells in the sorted prefix.
 * @param num_to_sort Number of particles in the sorted prefix.
 * @param pclsArray Device-side particle SoA container.
 * @param grid Device-side grid descriptor.
 * @param moments Packed moment output buffer for this species.
 */
__global__ void
cellAwareMomentKernel(const int* __restrict__ cell_start_offsets, int num_cells,
                      uint32_t num_to_sort, particleArrayCUDA* pclsArray,
                      grid3DCUDA* grid, cudaTypeArray1<cudaMomentType> moments);

/**
 * @brief Deposit the third central velocity moment for an unsorted species.
 *
 * The bulk buffer is packed as [rho, Jx, Jy, Jz] on the nodal grid using
 * already-communicated lower moments. The output buffer stores the symmetric
 * heat-flux components in the order:
 * Qxxx,Qxxy,Qxxz,Qxyy,Qxyz,Qxzz,Qyyy,Qyyz,Qyzz,Qzzz.
 *
 * @param momentParam Device-side moment parameter bundle for one species.
 * @param grid Device-side grid descriptor.
 * @param bulkMoments Packed lower moments [4][nxn][nyn][nzn].
 * @param heatFlux Packed heat-flux output [10][nxn][nyn][nzn].
 * @param qom Species charge-to-mass ratio.
 * @param rhoFloor Absolute charge-density floor for bulk-velocity division.
 */
__global__ void
heatFluxKernelUnsorted(momentParameter* momentParam, grid3DCUDA* grid,
                       const cudaTypeArray1<cudaMomentType> bulkMoments,
                       cudaTypeArray1<cudaMomentType> heatFlux,
                       cudaMomentType qom, cudaMomentType rhoFloor);

#endif // MOMENT_KERNEL_CUH
