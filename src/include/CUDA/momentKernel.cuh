#ifndef _MOMENTKERNEL_CUH_
#define _MOMENTKERNEL_CUH_


#include "cudaTypeDef.cuh"
#include "particleArrayCUDA.cuh"
#include "gridCUDA.cuh"
#include "particleExchange.cuh"





class momentParameter{

public:

    particleArrayCUDA* pclsArray; // default main array

    departureArrayType* departureArray; // a helper array for marking exiting particles


    //! @param pclsArrayCUDAPtr It should be a device pointer
    __host__ momentParameter(particleArrayCUDA* pclsArrayCUDAPtr, departureArrayType* departureArrayCUDAPtr){
        pclsArray = pclsArrayCUDAPtr;
        departureArray = departureArrayCUDAPtr;
    }

};

__global__ void momentKernelStayed(const uint32_t* appendCount, momentParameter* momentParam,
                                    grid3DCUDA* grid,
                                    cudaTypeArray1<cudaMomentType> moments);

__global__ void momentKernelNew(momentParameter* momentParam,
                                    grid3DCUDA* grid,
                                    cudaTypeArray1<cudaMomentType> moments,
                                    int stayedParticle);

// ── Cell-aware moment kernel for sorted particles ──
//
// One warp per cell.  All particles in [cell_start_offsets[c], cell_start_offsets[c+1])
// are guaranteed to belong to cell c (by the counting sort).  The warp reduces
// per-particle contributions via __shfl_down_sync and emits only 80 global
// atomicAdd per cell instead of 80 per particle.
//
// Parameters:
//   cell_start_offsets : [num_cells] exclusive prefix sum from the counting sort
//   num_cells          : total number of cells in the grid
//   num_to_sort        : number of sorted particles (acts as sentinel for last cell)
//   pclsArray          : device pointer to particleArrayCUDA (SoA fields)
//   grid               : device pointer to grid3DCUDA
//   moments            : [nxn*nyn*nzn*10] output moment array (must be zeroed before launch)
__global__ void cellAwareMomentKernel(
    const int*                    __restrict__ cell_start_offsets,
    int                           num_cells,
    uint32_t                      num_to_sort,
    particleArrayCUDA*            pclsArray,
    grid3DCUDA*                   grid,
    cudaTypeArray1<cudaMomentType> moments);


#endif