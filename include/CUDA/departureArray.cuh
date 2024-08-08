#ifndef _DEPARTURE_ARRAY_H_
#define _DEPARTURE_ARRAY_H_

#include "cudaTypeDef.cuh"
#include "arrayCUDA.cuh"


typedef struct departureArrayElement_s{
    uint32_t dest;          // destination of the particle exchange
    uint32_t hashedId;      // the id got from hashed sum
}departureArrayElement_t;

using departureArrayElementType = departureArrayElement_t;
using departureArrayType = arrayCUDA<departureArrayElementType>;

using exitingArray = arrayCUDA<SpeciesParticle>;


#endif