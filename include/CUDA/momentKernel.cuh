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

__global__ void momentKernelStayed(momentParameter* momentParam,
                                    grid3DCUDA* grid,
                                    cudaTypeArray1<cudaCommonType> moments);

__global__ void momentKernelNew(momentParameter* momentParam,
                                    grid3DCUDA* grid,
                                    cudaTypeArray1<cudaCommonType> moments,
                                    int stayedParticle);


#endif