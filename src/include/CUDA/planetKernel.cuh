#ifndef PLANET_KERNEL_CUH
#define PLANET_KERNEL_CUH

#include "cudaTypeDef.cuh"
#include "hashedSum.cuh"
#include "particleArrayCUDA.cuh"
#include "particleExchange.cuh"

/**
 * @brief Compact particles flagged PLANET into the planetArray (device-only).
 *        Uses hashedSum[PLANET_HASHEDSUM_INDEX] for scatter indices.
 *
 * @param pclsArray Device-side particle SoA container.
 * @param departureArray Device-side departure metadata array.
 * @param planetArr Device-side compact planet buffer.
 * @param hashedSumArray Device-side hashed-sum buckets used for scatter
 * indices.
 */
__global__ void planetExtractionKernel(particleArrayCUDA* pclsArray,
                                       departureArrayType* departureArray,
                                       planetArray* planetArr,
                                       hashedSum* hashedSumArray);

/**
 * @brief Reduce the total |q| of all particles in a species' planetArray.
 *        Result is atomicAdd'd into *chargeOut.
 *
 * @param planetArr Device-side planet buffer for one species.
 * @param count Number of valid planet particles in the buffer.
 * @param chargeOut Device pointer to the accumulated absolute charge.
 */
__global__ void planetChargeReductionKernel(planetArray* planetArr, int count,
                                            cudaParticleType* chargeOut);

/**
 * @brief Compute kinetic energy for each electron planet particle.
 *        Writes energy and a global index (speciesOffset + localIndex) into
 * buffers.
 *
 * @param planetArr Device-side planet buffer for one electron species.
 * @param count Number of valid planet particles in the buffer.
 * @param qom Charge-to-mass ratio of the species (negative for electrons).
 * @param energyBuf Output energy buffer shared across electron species.
 * @param globalIdxBuf Output global-index buffer shared across electron
 * species.
 * @param speciesOffset Offset into the merged energy/index buffers.
 */
__global__ void planetEnergyKernel(planetArray* planetArr, int count,
                                   cudaParticleType qom,
                                   cudaParticleType* energyBuf,
                                   uint32_t* globalIdxBuf, int speciesOffset);

/**
 * @brief One step of bitonic sort for (key, value) pairs.
 *        Sorts keys in DESCENDING order, permuting values alongside.
 *
 * @param keys Key buffer to sort in place.
 * @param values Value buffer permuted alongside @p keys.
 * @param j XOR distance for this bitonic step.
 * @param k Bitonic stage size.
 * @param n Padded array length; must be a power of two.
 */
__global__ void bitonicSortStepKernel(cudaParticleType* __restrict__ keys,
                                      uint32_t* __restrict__ values, int j,
                                      int k, int n);

/**
 * @brief Pad tail of arrays with -inf/invalid for bitonic sort.
 *
 * @param keys Key buffer to pad.
 * @param values Value buffer to pad with invalid sentinels.
 * @param realN Number of valid entries before padding.
 * @param paddedN Padded power-of-two buffer length.
 */
__global__ void bitonicPadKernel(cudaParticleType* keys, uint32_t* values,
                                 int realN, int paddedN);

/**
 * @brief Sequential prefix-sum of |q| in sorted order to find the cutoff index.
 *        Electrons at indices [0..cutoff-1] are deleted (highest energy).
 *        Electrons at indices [cutoff..n-1] survive and are reflected.
 *
 * @param planetArrs Device array of per-electron-species planetArray pointers.
 * @param nSpecies Number of electron species.
 * @param speciesOffsets Per-electron-species offsets into the merged sorted
 * buffers.
 * @param sortedGlobalIdx Sorted global indices encoded as `speciesOffset +
 * localIdx`.
 * @param n Total number of electron planet particles.
 * @param ionChargeTarget Device pointer to total ion absolute charge to match.
 * @param cutoffIndex Output pointer to the first sorted index that survives.
 */
__global__ void chargeCutoffKernel(planetArray** planetArrs, int nSpecies,
                                   const int* speciesOffsets,
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
 * @param planetArrs Device array of per-electron-species planetArray pointers.
 * @param nElecSpecies Number of electron species.
 * @param speciesOffsets Per-species offsets into the merged sorted buffers and
 * output buffer.
 * @param sortedGlobalIdx Sorted global indices from bitonic sort.
 * @param cutoffDevice Device pointer written by chargeCutoffKernel.
 * @param totalElecPlanet Total number of electron planet particles.
 * @param outputBuf Compact output buffer laid out by species offset.
 * @param survivorCounters Per-species atomic counters; must be zeroed before
 * launch.
 * @param originX X coordinate of the planet center.
 * @param originY Y coordinate of the planet center.
 * @param originZ Z coordinate of the planet center.
 * @param sphereRadius Planet radius.
 * @param doSphere Planet geometry selector: 1 for 3D sphere, 2 for 2D XZ-plane
 * circle.
 */
__global__ void planetReflectCompactKernel(
    planetArray** planetArrs, int nElecSpecies, const int* speciesOffsets,
    const uint32_t* sortedGlobalIdx, const int* cutoffDevice,
    int totalElecPlanet, SpeciesParticle* outputBuf, int* survivorCounters,
    cudaCommonType originX, cudaCommonType originY, cudaCommonType originZ,
    cudaCommonType sphereRadius, int doSphere);

/**
 * @brief Fused kernel: DIFFUSELY scatter surviving electrons (isotropic
 *        velocity on outward hemisphere, preserving speed) and compact
 *        into a contiguous output buffer.  Structure identical to
 *        planetReflectCompactKernel; only the velocity update differs.
 *
 * @param planetArrs Device array of per-electron-species planetArray pointers.
 * @param nElecSpecies Number of electron species.
 * @param speciesOffsets Per-species offsets into the merged sorted buffers and
 * output buffer.
 * @param sortedGlobalIdx Sorted global indices from bitonic sort.
 * @param cutoffDevice Device pointer written by chargeCutoffKernel.
 * @param totalElecPlanet Total number of electron planet particles.
 * @param outputBuf Compact output buffer laid out by species offset.
 * @param survivorCounters Per-species atomic counters; must be zeroed before
 * launch.
 * @param originX X coordinate of the planet center.
 * @param originY Y coordinate of the planet center.
 * @param originZ Z coordinate of the planet center.
 * @param sphereRadius Planet radius.
 * @param doSphere Planet geometry selector: 1 for 3D sphere, 2 for 2D XZ-plane
 * circle.
 * @param rngSeedBase Base seed for the per-thread PRNG (for example the cycle
 * number).
 */
__global__ void planetDiffuseCompactKernel(
    planetArray** planetArrs, int nElecSpecies, const int* speciesOffsets,
    const uint32_t* sortedGlobalIdx, const int* cutoffDevice,
    int totalElecPlanet, SpeciesParticle* outputBuf, int* survivorCounters,
    cudaCommonType originX, cudaCommonType originY, cudaCommonType originZ,
    cudaCommonType sphereRadius, int doSphere, uint32_t rngSeedBase);

#endif // PLANET_KERNEL_CUH
