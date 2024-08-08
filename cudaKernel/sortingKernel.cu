
#include "cudaTypeDef.cuh"
#include "arrayCUDA.cuh"
#include "hashedSum.cuh"
#include "particleArrayCUDA.cuh"
#include "departureArray.cuh"


/**
 * @brief 	Fill the many holes in pclArray, making it compact. Launch (x) thread.
 * @details The exiting particles have been copied to exitingBuffer, they left the pclArray many holes.
 * 			
 * 			
 * @param fillerArray  Used for storing the index of the stayed particles in between (nop-x) to (nop-1)
 * @param hashedSumArray it contains 1 hashedSum instance, prepared in exitingKernel. 
 * 							For the filler particles in rear part.
 */
__global__ void sortingKernel1(particleArrayCUDA* pclsArray, departureArrayType* departureArray, 
								cudaTypeArray1<int> fillerArray, hashedSum* hashedSumArray){

	uint pidx = blockIdx.x * blockDim.x + threadIdx.x;
	uint x = hashedSumArray->getSum();
    if(pidx >= x)return; 		
	pidx += (pclsArray->getNOP() - x);			// rear part of pclArray

	auto departureElement = departureArray->getArray() + pidx;
    if(departureElement->dest != 0)return; 		// exiting particles, the holes in the rear part

	auto pcl = pclsArray->getpcls() + pidx;

	auto index = hashedSumArray->getIndex(pidx, departureElement->hashedId); // updated

	fillerArray[index] = pidx;


}





/**
 * @brief 	Fill the many holes in pclArray, making it compact. Launch (nop -x) thread.
 * @details Sorting2 has to be Launched after Moment kernel.
 * 			The indexes of the filler particles have been recorded into the fillerArray in the previous Sorting1Kernel
 * 			
 * @param fillerArray  Used for storing the index of the stayed particles in between (nop-x) to (nop-1)
 * @param hashedSumArray it contains 1 hashedSum instance, prepared in exitingKernel. 
 * 							For the exiting particles in front part.
 */
__global__ void sortingKernel2(particleArrayCUDA* pclsArray, departureArrayType* departureArray, 
								cudaTypeArray1<int> fillerArray, hashedSum* hashedSumArray){

	uint pidx = blockIdx.x * blockDim.x + threadIdx.x;
    if(pidx >= hashedSumArray->getSum())return; 		// front part of pclArray

	auto departureElement = departureArray->getArray() + pidx;
    if(departureElement->dest == 0)return; 				// exiting particles, the holes

	auto pcl = pclsArray->getpcls() + pidx;

	auto index = hashedSumArray->getIndex(pidx, departureElement->hashedId); // updated

	memcpy(pcl, pclsArray->getpcls() + fillerArray[index], sizeof(SpeciesParticle));


}







