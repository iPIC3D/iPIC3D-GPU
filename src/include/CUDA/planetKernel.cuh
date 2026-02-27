#ifndef _PLANET_KERNEL_CUH_
#define _PLANET_KERNEL_CUH_

#include "cudaTypeDef.cuh"
#include "particleArrayCUDA.cuh"
#include "particleExchange.cuh"
#include "hashedSum.cuh"


/**
 * @brief Compact particles flagged PLANET into the planetArray (device-only).
 *        Uses hashedSum[PLANET_HASHEDSUM_INDEX] for scatter indices.
 */
__global__ void planetExtractionKernel(particleArrayCUDA* pclsArray,
    departureArrayType* departureArray, planetArray* planetArr,
    hashedSum* hashedSumArray);

/**
 * @brief Reduce the total |q| of all particles in a species' planetArray.
 *        Result is atomicAdd'd into *chargeOut.
 */
__global__ void planetChargeReductionKernel(planetArray* planetArr, int count,
    cudaParticleType* chargeOut);

/**
 * @brief Compute kinetic energy for each electron planet particle.
 *        Writes energy and a global index (speciesOffset + localIndex) into buffers.
 * @param qom charge-to-mass ratio of the species (negative for electrons)
 * @param speciesOffset offset into the merged energy/index buffers
 */
__global__ void planetEnergyKernel(planetArray* planetArr, int count,
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
 * @param planetArrs      device array of per-electron-species planetArray pointers
 * @param nSpecies         number of electron species
 * @param speciesOffsets   per-electron-species offsets into the merged sorted buffers
 * @param sortedGlobalIdx  sorted global indices (speciesOffset + localIdx)
 * @param n                total number of electron planet particles
 * @param ionChargeTarget  device pointer to total |Q_ion| to match
 * @param cutoffIndex      output: first sorted index that SURVIVES (reflects)
 */
__global__ void chargeCutoffKernel(
    planetArray** planetArrs, int nSpecies, const int* speciesOffsets,
    const uint32_t* sortedGlobalIdx, int n,
    const cudaParticleType* ionChargeTarget,
    int* cutoffIndex);

/**
 * @brief Fused kernel: reflect surviving electrons and compact them into
 *        a contiguous output buffer, one segment per electron species.
 *        Reads cutoff from device memory (no host sync needed).
 *        Each thread handles one entry in the sorted survivor range,
 *        decodes species, reflects in the source planetArray, and writes
 *        the reflected particle to outputBuf[speciesOffset + atomicSlot].
 *        Also increments per-species atomic counters.
 *
 * @param planetArrs       device array of per-electron-species planetArray pointers
 * @param nElecSpecies     number of electron species
 * @param speciesOffsets   per-electron-species offsets into the merged sorted buffers;
 *                         also used as write offsets into outputBuf
 * @param sortedGlobalIdx  sorted global indices from bitonic sort
 * @param cutoffDevice     device pointer written by chargeCutoffKernel
 * @param totalElecPlanet  total number of electron planet particles
 * @param outputBuf        compact output buffer (SpeciesParticle), laid out by species offsets
 * @param survivorCounters per-electron-species atomic counters (must be zeroed before launch)
 * @param originX/Y/Z      planet sphere center
 * @param sphereRadius      planet sphere radius
 * @param doSphere          1: 3D sphere, 2: 2D sphere (XZ plane)
 */
__global__ void planetReflectCompactKernel(
    planetArray** planetArrs, int nElecSpecies,
    const int* speciesOffsets,
    const uint32_t* sortedGlobalIdx,
    const int* cutoffDevice,
    int totalElecPlanet,
    SpeciesParticle* outputBuf,
    int* survivorCounters,
    cudaCommonType originX, cudaCommonType originY, cudaCommonType originZ,
    cudaCommonType sphereRadius, int doSphere);

/**
 * @brief Fused kernel: DIFFUSELY scatter surviving electrons (isotropic
 *        velocity on outward hemisphere, preserving speed) and compact
 *        into a contiguous output buffer.  Structure identical to
 *        planetReflectCompactKernel; only the velocity update differs.
 *
 * @param rngSeedBase  base seed for per-thread PRNG (e.g. cycle number)
 */
__global__ void planetDiffuseCompactKernel(
    planetArray** planetArrs, int nElecSpecies,
    const int* speciesOffsets,
    const uint32_t* sortedGlobalIdx,
    const int* cutoffDevice,
    int totalElecPlanet,
    SpeciesParticle* outputBuf,
    int* survivorCounters,
    cudaCommonType originX, cudaCommonType originY, cudaCommonType originZ,
    cudaCommonType sphereRadius, int doSphere,
    uint32_t rngSeedBase);


#endif
