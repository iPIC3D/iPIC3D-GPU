#ifndef _MOVERKERNEL_CUH_
#define _MOVERKERNEL_CUH_

#include "ParticleSoAHost.h"
#include "cudaTypeDef.cuh"
#include "particleArrayCUDA.cuh"
#include "EMfields3D.h"
#include "gridCUDA.cuh"
#include "particleExchange.cuh"
#include "hashedSum.cuh"

/**
 * @brief Device-side parameter bundle consumed by the GPU mover kernels.
 *
 * The solver populates one instance per species with particle-array pointers,
 * departure bookkeeping, scalar species constants, and boundary-condition flags.
 */
class moverParameter
{

public: // particle arrays

    particleArrayCUDA* pclsArray; // main device particle array

    departureArrayType* departureArray; // device departure flags

    hashedSum* hashedSumArray; // departure-direction hashed sums


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

    /**
     * @brief Construct a mover parameter bundle from existing device buffers.
     *
     * @param pclHost Host-side particle metadata source for scalar species parameters.
     * @param pclsArrayCUDAPtr      Device pointer to the species SoA container.
     * @param departureArrayCUDAPtr Device pointer to the departure array.
     * @param hashedSumArrayCUDAPtr Device pointer to the hashed-sum bucket array.
     */
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

/**
 * @brief Advance one species with the standard predictor-corrector mover.
 *
 * @param moverParam Device-side mover parameter bundle for one species.
 * @param fieldForPcls Packed mover field buffer sampled from the grid.
 * @param grid Device-side grid descriptor.
 */
__global__ void moverKernel(moverParameter *moverParam,
                            cudaTypeArray1<cudaFieldType> fieldForPcls,
                            grid3DCUDA *grid);

/**
 * @brief Advance one species with the adaptive-subcycling mover used for planet cases.
 *
 * The timestep is split into subcycles based on the local gyroperiod and the
 * velocity update is performed in relativistic form.
 *
 * @param moverParam Device-side mover parameter bundle for one species.
 * @param fieldForPcls Packed mover field buffer sampled from the grid.
 * @param grid Device-side grid descriptor.
 */
__global__ void moverSubcyclesKernel(moverParameter *moverParam,
                            cudaTypeArray1<cudaFieldType> fieldForPcls,
                            grid3DCUDA *grid);

    
// __global__ void castingField(grid3DCUDA *grid, cudaTypeArray1<cudaCommonType> fieldForPcls);

#endif
