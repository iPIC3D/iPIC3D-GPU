#ifndef _PARTICLE_EXCHANGE_H_
#define _PARTICLE_EXCHANGE_H_

#include "cudaTypeDef.cuh"
#include "arrayCUDA.cuh"
#include "hashedSum.cuh"
#include "ParticleSoADevice.cuh"


typedef struct departureArrayElement_s{
    enum departureDestination{
        STAY = 0,

        XLOW = 1,
        XHIGH,
        YLOW,
        YHIGH,
        ZLOW,
        ZHIGH,

        DELETE = 7,
        PLANET = 8
    };

    enum hashedSumIndex{
        XLOW_HASHEDSUM_INDEX = 0,
        XHIGH_HASHEDSUM_INDEX,
        YLOW_HASHEDSUM_INDEX,
        YHIGH_HASHEDSUM_INDEX,
        ZLOW_HASHEDSUM_INDEX,
        ZHIGH_HASHEDSUM_INDEX,
        DELETE_HASHEDSUM_INDEX = 6,
        PLANET_HASHEDSUM_INDEX = 7,
        HOLE_HASHEDSUM_INDEX,
        FILLER_HASHEDSUM_INDEX,

        HASHED_SUM_NUM = 10
    };

    uint32_t dest;          // destination of the particle exchange
    uint32_t hashedId;      // the id got from hashed sum
}departureArrayElement_t;

using departureArrayElementType = departureArrayElement_t;
using departureArrayType = arrayCUDA<departureArrayElementType>;

using fillerBuffer = arrayCUDA<int>;


/**
 * @brief Copy exiting particles from the main SoA arrays into a compact SoA
 *        exiting buffer, organised by direction using hashedSum offsets.
 *        Also prepares hashedSum data for the sorting kernels.
 */
__global__ void exitingKernel(particleArrayCUDA* pclsArray, departureArrayType* departureArray,
                                ParticleSoADevice* exitingSoA, hashedSum* hashedSumArray);

/**
 * @brief Compact particles flagged PLANET into a SoA planet buffer.
 *        Uses hashedSum[PLANET_HASHEDSUM_INDEX] for scatter indices.
 */
__global__ void planetExtractionKernel(particleArrayCUDA* pclsArray, departureArrayType* departureArray,
                                ParticleSoADevice* planetSoA, hashedSum* hashedSumArray);

__global__ void sortingKernel1(particleArrayCUDA* pclsArray, departureArrayType* departureArray, 
								fillerBuffer* fillerBuffer, hashedSum* hashedSumArray, int x);

__global__ void sortingKernel2(particleArrayCUDA* pclsArray, departureArrayType* departureArray, 
								fillerBuffer* fillerBuffer, hashedSum* hashedSumArray, int stayedParticle);


// ── Device SoA buffer management helpers (called from host) ──

/** Allocate 8 device arrays for an SoA buffer with the given capacity. */
inline void allocateDeviceSoA(ParticleSoADevice& soa, uint32_t capacity) {
    soa.capacity = capacity;
    soa.nop = 0;
    cudaErrChk(cudaMalloc(&soa.u, capacity * sizeof(cudaParticleType)));
    cudaErrChk(cudaMalloc(&soa.v, capacity * sizeof(cudaParticleType)));
    cudaErrChk(cudaMalloc(&soa.w, capacity * sizeof(cudaParticleType)));
    cudaErrChk(cudaMalloc(&soa.q, capacity * sizeof(cudaParticleType)));
    cudaErrChk(cudaMalloc(&soa.x, capacity * sizeof(cudaParticleType)));
    cudaErrChk(cudaMalloc(&soa.y, capacity * sizeof(cudaParticleType)));
    cudaErrChk(cudaMalloc(&soa.z, capacity * sizeof(cudaParticleType)));
    cudaErrChk(cudaMalloc(&soa.t, capacity * sizeof(cudaParticleType)));
}

/** Free all 8 device arrays of an SoA buffer. */
inline void freeDeviceSoA(ParticleSoADevice& soa) {
    cudaFree(soa.u); cudaFree(soa.v); cudaFree(soa.w); cudaFree(soa.q);
    cudaFree(soa.x); cudaFree(soa.y); cudaFree(soa.z); cudaFree(soa.t);
    soa = {};
}

/** Expand an SoA buffer (no data preservation — content is rebuilt each cycle).
 *  @param stream  Stream to synchronize before freeing (ensures in-flight kernels finish). */
inline void expandDeviceSoA(ParticleSoADevice& soa, uint32_t newCapacity, cudaStream_t stream) {
    if (newCapacity <= soa.capacity) return;
    cudaErrChk(cudaStreamSynchronize(stream));
    freeDeviceSoA(soa);
    allocateDeviceSoA(soa, newCapacity);
}


#endif