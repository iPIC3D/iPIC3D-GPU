#ifndef _PLANET_KERNEL_CUH_
#define _PLANET_KERNEL_CUH_

#include "cudaTypeDef.cuh"
#include "particleArrayCUDA.cuh"
#include "particleExchange.cuh"
#include "hashedSum.cuh"


/**
 * @brief Compact particles flagged PLANET into the planet SoA buffer (device-only).
 *        Uses hashedSum[PLANET_HASHEDSUM_INDEX] for scatter indices.
 */
__global__ void planetExtractionKernel(particleArrayCUDA* pclsArray,
    departureArrayType* departureArray, ParticleSoADevice* planetSoA,
    hashedSum* hashedSumArray);

/**
 * @brief Reduce the total |q| of all particles in a species' planet SoA buffer.
 *        Result is atomicAdd'd into *chargeOut.
 */
__global__ void planetChargeReductionKernel(ParticleSoADevice* planetSoA, int count,
    cudaParticleType* chargeOut);

/**
 * @brief Compute kinetic energy for each electron planet particle.
 *        Writes energy and a global index (speciesOffset + localIndex) into buffers.
 * @param qom charge-to-mass ratio of the species (negative for electrons)
 * @param speciesOffset offset into the merged energy/index buffers
 */
__global__ void planetEnergyKernel(ParticleSoADevice* planetSoA, int count,
    cudaParticleType qom,
    cudaParticleType* energyBuf, uint32_t* globalIdxBuf,
    int speciesOffset);

/**
 * @brief One step of bitonic sort for (key, value) pairs.
 *        Sorts keys in DESCENDING order, permuting values alongside.
 * @param n padded array length (must be power of 2)
 */
__global__ void bitonicSortStepKernel(
    cudaParticleType* __restrict__ keys,
    uint32_t* __restrict__ values,
    int j, int k, int n);

/**
 * @brief Pad tail of arrays with -inf/invalid for bitonic sort.
 */
__global__ void bitonicPadKernel(
    cudaParticleType* keys, uint32_t* values,
    int realN, int paddedN);

/**
 * @brief Sequential prefix-sum of |q| in sorted order to find the cutoff index.
 *        Electrons at indices [0..cutoff-1] are deleted (highest energy).
 *        Electrons at indices [cutoff..n-1] survive and are reflected.
 *
 * @param planetSoAArrs    device array of per-electron-species ParticleSoADevice pointers
 * @param nSpecies         number of electron species
 * @param speciesOffsets   per-electron-species offsets into the merged sorted buffers
 * @param sortedGlobalIdx  sorted global indices (speciesOffset + localIdx)
 * @param n                total number of electron planet particles
 * @param ionChargeTarget  device pointer to total |Q_ion| to match
 * @param cutoffIndex      output: first sorted index that SURVIVES (reflects)
 */
__global__ void chargeCutoffKernel(
    ParticleSoADevice** planetSoAArrs, int nSpecies, const int* speciesOffsets,
    const uint32_t* sortedGlobalIdx, int n,
    const cudaParticleType* ionChargeTarget,
    int* cutoffIndex);

/**
 * @brief Fused kernel: reflect surviving electrons and compact them into
 *        a contiguous SoA output buffer, one segment per electron species.
 *        Reads cutoff from device memory (no host sync needed).
 */
__global__ void planetReflectCompactKernel(
    ParticleSoADevice** planetSoAArrs, int nElecSpecies,
    const int* speciesOffsets,
    const uint32_t* sortedGlobalIdx,
    const int* cutoffDevice,
    int totalElecPlanet,
    ParticleSoADevice* outputSoA,
    int* survivorCounters,
    cudaCommonType originX, cudaCommonType originY, cudaCommonType originZ,
    cudaCommonType sphereRadius, int doSphere);

/**
 * @brief Fused kernel: DIFFUSELY scatter surviving electrons (isotropic
 *        velocity on outward hemisphere, preserving speed) and compact
 *        into a contiguous SoA output buffer.  Structure identical to
 *        planetReflectCompactKernel; only the velocity update differs.
 *
 * @param rngSeedBase  base seed for per-thread PRNG (e.g. cycle number)
 */
__global__ void planetDiffuseCompactKernel(
    ParticleSoADevice** planetSoAArrs, int nElecSpecies,
    const int* speciesOffsets,
    const uint32_t* sortedGlobalIdx,
    const int* cutoffDevice,
    int totalElecPlanet,
    ParticleSoADevice* outputSoA,
    int* survivorCounters,
    cudaCommonType originX, cudaCommonType originY, cudaCommonType originZ,
    cudaCommonType sphereRadius, int doSphere,
    uint32_t rngSeedBase);


#endif
