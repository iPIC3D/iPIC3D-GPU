
#include "cudaTypeDef.cuh"
#include "arrayCUDA.cuh"
#include "hashedSum.cuh"
#include "particleArrayCUDA.cuh"
#include "particleExchange.cuh"

// ======= Particle compaction helpers =======
//
// Both kernels in this file only compact the SoA particle buffer by moving
// stayed-particles from the rear region into front-region holes left by
// exiting / deleted / planet-removed particles. No relative ordering is
// preserved or imposed.

/**
 * @brief Record the stayed particles from the rear filler region.
 *
 * The rear `x` entries of the SoA buffer may contain particles that can be
 * moved forward to fill front-side holes. This kernel collects those source
 * indices into the filler buffer. (Compaction stage 1, no sorting.)
 * @param pclsArray Particle SoA buffer being compacted.
 * @param departureArray Per-particle destination metadata.
 * @param fillerBuffer Buffer receiving source indices for reusable rear particles.
 * @param hashedSumArray Prefix-sum helpers for filler-buffer indexing.
 * @param x Number of rear entries to inspect.
 */
__global__ void compactParticles1(particleArrayCUDA* pclsArray, departureArrayType* departureArray, 
								fillerBuffer* fillerBuffer, hashedSum* hashedSumArray, int x){

	uint pidx = blockIdx.x * blockDim.x + threadIdx.x;

    if(pidx >= x)return; 		
	pidx += (pclsArray->getNOP() - x);			// rear part of pclArray

	auto departureElement = departureArray->getArray() + pidx;
    if(departureElement->dest != 0)return; 		// exiting particles, the holes in the rear part

	auto index = hashedSumArray->getIndex(pidx, departureElement->hashedId); // updated

	fillerBuffer->getArray()[index] = pidx;


}
/**
 * @brief Fill front-side holes in the SoA buffer using the recorded filler indices.
 *
 * This is the second stage of the GPU compaction path. It reads the source
 * indices produced by `compactParticles1()` and copies those particles into
 * the deleted or exiting slots in the front stayed-particle range.
 * (Compaction stage 2, no sorting.)
 * @param pclsArray Particle SoA buffer being compacted.
 * @param departureArray Per-particle destination metadata.
 * @param fillerBuffer Buffer containing source indices from the rear region.
 * @param hashedSumArray Prefix-sum helpers for hole indexing.
 * @param stayedParticle Number of front entries that remain in the compacted range.
 */
__global__ void compactParticles2(particleArrayCUDA* pclsArray, departureArrayType* departureArray, 
								fillerBuffer* fillerBuffer, hashedSum* hashedSumArray, int stayedParticle){

	uint pidx = blockIdx.x * blockDim.x + threadIdx.x;
    if(pidx >= stayedParticle)return; 		// front part of pclArray

	auto departureElement = departureArray->getArray() + pidx;
    if(departureElement->dest == 0)return; 				// exiting particles, the holes

	auto index = hashedSumArray->getIndex(pidx, departureElement->hashedId); // updated

	// Copy particle data from filler to hole via SoA
	const uint32_t srcIdx = fillerBuffer->getArray()[index];
	pclsArray->getX()[pidx] = pclsArray->getX()[srcIdx];
	pclsArray->getY()[pidx] = pclsArray->getY()[srcIdx];
	pclsArray->getZ()[pidx] = pclsArray->getZ()[srcIdx];
	pclsArray->getU()[pidx] = pclsArray->getU()[srcIdx];
	pclsArray->getV()[pidx] = pclsArray->getV()[srcIdx];
	pclsArray->getW()[pidx] = pclsArray->getW()[srcIdx];
	pclsArray->getQ()[pidx] = pclsArray->getQ()[srcIdx];
	pclsArray->getT()[pidx] = pclsArray->getT()[srcIdx];


}





