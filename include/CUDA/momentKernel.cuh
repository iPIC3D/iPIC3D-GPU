#ifndef _MOMENTKERNEL_CUH_
#define _MOMENTKERNEL_CUH_


#include "cudaTypeDef.cuh"
#include "particleArrayCUDA.cuh"
#include "gridCUDA.cuh"





class momentParameter{

public:

    particleArrayCUDA* pclsArray; // default main array

    //! @param pclsArrayCUDAPtr It should be a device pointer
    __host__ momentParameter(particleArrayCUDA* pclsArrayCUDAPtr){
        pclsArray = pclsArrayCUDAPtr;
    }

};

__global__ void momentKernel(momentParameter* momentParam,
                            grid3DCUDA* grid,
                            cudaTypeArray1<cudaCommonType> moments);


#endif