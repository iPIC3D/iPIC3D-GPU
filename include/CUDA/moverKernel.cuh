#ifndef _MOVERKERNEL_CUH_
#define _MOVERKERNEL_CUH_

#include "Particles3D.h"
#include "cudaTypeDef.cuh"
#include "particleArrayCUDA.cuh"
#include "EMfields3D.h"
#include "gridCUDA.cuh"
#include "particleExchange.cuh"
#include "hashedSum.cuh"


class moverParameter
{

public: //particle arrays

    particleArrayCUDA* pclsArray; // default main array

    departureArrayType* departureArray; // a helper array for marking exiting particles

    hashedSum* hashedSumArray; // 8 hashed sum


public: // common parameter

    cudaCommonType dt;

    cudaCommonType qom;

    cudaCommonType c;

    int NiterMover;

    int DFIELD_3or4;

    cudaCommonType umax, umin, vmax, vmin, wmax, wmin;

    // moverOutflowParameter outflowParam;


public:


    __host__ moverParameter(Particles3D* p3D, VirtualTopology3D* vct)
        : dt(p3D->dt), qom(p3D->qom), c(p3D->c), NiterMover(p3D->NiterMover), DFIELD_3or4(::DFIELD_3or4),
        umax(p3D->umax), umin(p3D->umin), vmax(p3D->vmax), vmin(p3D->vmin), wmax(p3D->wmax), wmin(p3D->wmin)
    {
        // create the particle array, stream 0
        pclsArray = particleArrayCUDA(p3D).copyToDevice();
        departureArray = departureArrayType(p3D->getNOP() * 1.5).copyToDevice();

    }

    //! @param pclsArrayCUDAPtr It should be a device pointer
    __host__ moverParameter(Particles3D* p3D, particleArrayCUDA* pclsArrayCUDAPtr, 
                            departureArrayType* departureArrayCUDAPtr, hashedSum* hashedSumArrayCUDAPtr)
        : dt(p3D->dt), qom(p3D->qom), c(p3D->c), NiterMover(p3D->NiterMover), DFIELD_3or4(::DFIELD_3or4),
        umax(p3D->umax), umin(p3D->umin), vmax(p3D->vmax), vmin(p3D->vmin), wmax(p3D->wmax), wmin(p3D->wmin)
    {
        // create the particle array, stream 0
        pclsArray = pclsArrayCUDAPtr;
        departureArray = departureArrayCUDAPtr;
        hashedSumArray = hashedSumArrayCUDAPtr;

    }
};

__global__ void moverKernel(moverParameter *moverParam,
                            cudaTypeArray1<cudaCommonType> fieldForPcls,
                            grid3DCUDA *grid);

#endif