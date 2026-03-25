#ifndef _MOVERKERNEL_CUH_
#define _MOVERKERNEL_CUH_

#include "ParticleSoAHost.h"
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

    cudaParticleType dt;

    cudaParticleType qom;

    cudaParticleType c;

    int NiterMover;

    int DFIELD_3or4;

    cudaParticleType umax, umin, vmax, vmin, wmax, wmin;

    // moverOutflowParameter outflowParam;

    // For openBC, XLeft, XRight, YLeft, YRight, ZLeft, ZRight
    bool doOpenBC; // a OR of applyOpenBC
    bool applyOpenBC[6];
    cudaCommonType deleteBoundary[6];
    cudaCommonType openBoundary[6];
    uint32_t appendCountAtomic; // the number of duplicated particles to be appended to the array, just in time

    // For repopulate injection, XLeft, XRight, YLeft, YRight, ZLeft, ZRight
    bool doRepopulateInjection;
    bool doRepopulateInjectionSide[6];
    cudaCommonType repopulateBoundary[6];

    // For sphere
    int doSphere; // 0: no sphere, 1: sphere, 2: sphere2D(XZ)
    cudaCommonType sphereOrigin[3];
    cudaCommonType sphereRadius;

    // Per-face EXIT BC flags: if true, particles exiting this face are
    // deleted on the GPU (dest=DELETE) instead of being sent via MPI.
    // Order: XLeft, XRight, YLeft, YRight, ZLeft, ZRight
    bool isExitBC[6];

public:


    __host__ moverParameter(ParticleSoAHost* pclHost, VirtualTopology3D* vct)
        : dt(pclHost->timeStep_), qom(pclHost->chargeOverMass_), c(pclHost->speedOfLight_),
        NiterMover(pclHost->numMoverIterations_), DFIELD_3or4(::DFIELD_3or4),
        umax(pclHost->velocityCapMaxX_), umin(pclHost->velocityCapMinX_),
        vmax(pclHost->velocityCapMaxY_), vmin(pclHost->velocityCapMinY_),
        wmax(pclHost->velocityCapMaxZ_), wmin(pclHost->velocityCapMinZ_)
    {
        // create the particle array, stream 0
        pclsArray = particleArrayCUDA(pclHost).copyToDevice();
        departureArray = departureArrayType(pclHost->getNOP() * 1.5).copyToDevice();

    }

    //! @param pclsArrayCUDAPtr It should be a device pointer
    __host__ moverParameter(ParticleSoAHost* pclHost, particleArrayCUDA* pclsArrayCUDAPtr, 
                            departureArrayType* departureArrayCUDAPtr, hashedSum* hashedSumArrayCUDAPtr)
        : dt(pclHost->timeStep_), qom(pclHost->chargeOverMass_), c(pclHost->speedOfLight_),
        NiterMover(pclHost->numMoverIterations_), DFIELD_3or4(::DFIELD_3or4),
        umax(pclHost->velocityCapMaxX_), umin(pclHost->velocityCapMinX_),
        vmax(pclHost->velocityCapMaxY_), vmin(pclHost->velocityCapMinY_),
        wmax(pclHost->velocityCapMaxZ_), wmin(pclHost->velocityCapMinZ_)
    {
        // create the particle array, stream 0
        pclsArray = pclsArrayCUDAPtr;
        departureArray = departureArrayCUDAPtr;
        hashedSumArray = hashedSumArrayCUDAPtr;

    }
};

__global__ void moverKernel(moverParameter *moverParam,
                            cudaTypeArray1<cudaFieldType> fieldForPcls,
                            grid3DCUDA *grid);

// mover with adaptive subcycling --> divides dt by eight times the particle gyroperiod and performs a relativistic velocity update
__global__ void moverSubcyclesKernel(moverParameter *moverParam,
                            cudaTypeArray1<cudaFieldType> fieldForPcls,
                            grid3DCUDA *grid);

    
// __global__ void castingField(grid3DCUDA *grid, cudaTypeArray1<cudaCommonType> fieldForPcls);

#endif