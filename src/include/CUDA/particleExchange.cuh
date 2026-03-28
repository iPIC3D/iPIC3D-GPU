#ifndef _PARTICLE_EXCHANGE_H_
#define _PARTICLE_EXCHANGE_H_

#include "cudaTypeDef.cuh"
#include "arrayCUDA.cuh"
#include "hashedSum.cuh"


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

using exitingArray = arrayCUDA<SpeciesParticle>;

using planetArray = arrayCUDA<SpeciesParticle>;

using fillerBuffer = arrayCUDA<int>;


/**
 * @brief Copy exiting particles from the main SoA arrays into a compact AoS
 *        exiting buffer, organised by direction using hashedSum offsets.
 *        Gathers SoA fields into SpeciesParticle structs for single D→H memcpy.
 *        Also prepares hashedSum data for the sorting kernels.
 *
 * @param pclsArray Device-side particle SoA container.
 * @param departureArray Device-side departure metadata array.
 * @param exitingArray Device-side compact AoS exiting buffer.
 * @param hashedSumArray Device-side hashed-sum buckets used for scatter indices.
 */
__global__ void exitingKernel(particleArrayCUDA* pclsArray, departureArrayType* departureArray,
                                exitingArray* exitingArray, hashedSum* hashedSumArray);

/**
 * @brief Compact particles flagged PLANET into an AoS planet buffer.
 *        Uses hashedSum[PLANET_HASHEDSUM_INDEX] for scatter indices.
 *
 * @param pclsArray Device-side particle SoA container.
 * @param departureArray Device-side departure metadata array.
 * @param planetArr Device-side compact AoS planet buffer.
 * @param hashedSumArray Device-side hashed-sum buckets used for scatter indices.
 */
__global__ void planetExtractionKernel(particleArrayCUDA* pclsArray, departureArrayType* departureArray,
                                planetArray* planetArr, hashedSum* hashedSumArray);
/**
 * @brief Record filler-particle indices from the rear compact region of the SoA array.
 *
 * This kernel scans the back `x` entries of the particle array, identifies the
 * stayed particles that can be used to fill holes in the front region, and
 * writes their indices into the filler buffer.
 *
 * @param pclsArray Device-side particle SoA container.
 * @param departureArray Device-side departure metadata array.
 * @param fillerBuffer Device-side buffer that records rear filler indices.
 * @param hashedSumArray Device-side hashed-sum bucket for filler compaction.
 * @param x Number of rear entries to scan.
 */
__global__ void sortingKernel1(particleArrayCUDA* pclsArray, departureArrayType* departureArray, 
									fillerBuffer* fillerBuffer, hashedSum* hashedSumArray, int x);

/**
 * @brief Fill front-region holes in the SoA array using indices prepared by sortingKernel1.
 *
 * Each thread handles one particle in the stayed prefix and copies one rear
 * filler particle into the current hole when needed.
 *
 * @param pclsArray Device-side particle SoA container.
 * @param departureArray Device-side departure metadata array.
 * @param fillerBuffer Device-side buffer filled by sortingKernel1().
 * @param hashedSumArray Device-side hashed-sum bucket for front-hole compaction.
 * @param stayedParticle Number of particles in the compacted stayed prefix.
 */
__global__ void sortingKernel2(particleArrayCUDA* pclsArray, departureArrayType* departureArray, 
									fillerBuffer* fillerBuffer, hashedSum* hashedSumArray, int stayedParticle);


#endif
