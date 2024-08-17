
#include "cudaTypeDef.cuh"
#include "arrayCUDA.cuh"
#include "hashedSum.cuh"
#include "particleArrayCUDA.cuh"
#include "particleExchange.cuh"



/**
 * @brief   Copy the exiting particles of the species to ExitingBuffer. Launch (nop) threads.
 * @details By the end of Mover, the DepartureArray has been prepared, host allocates the ExitingBuffer according 
 *          to the Hashed SumUp value for each direction.
 *          This kernel is responsible for moving the exiting particles in the pclArray into the ExitingBuffer,
 *          according to the DepartureArray.  
 *          The exiting particles in ExitingBuffer are orginaized in their destinations, with random order adopted from hashedSum.
 *          
 *          This kernel is also responsible for preparing the 2 hashedSum for SortingKernel1 and SortingKernel2. It will modify the 
 *          elements of the departure array.
 *          
 * @param exitingArray The buffer used for exiting particles for 6 directions, the size and distriburtion are decided by the 6 hashedSum
 * @param hashedSumArray 8 hashedSum, 6 from the Mover, 2 for Sorting.
 * 
 */
__global__ void exitingKernel(particleArrayCUDA* pclsArray, departureArrayType* departureArray, 
                                exitingArray* exitingArray, hashedSum* hashedSumArray){
                                    
    uint pidx = blockIdx.x * blockDim.x + threadIdx.x;
    if(pidx >= pclsArray->getNOP())return;

    __shared__ int x; // x, the number of exiting particles  , can be a increasing array
    if(threadIdx.x == 0){ x = 0; for(int i=0; i < 6; i++)x += hashedSumArray[i].getSum(); }
    __syncthreads();
    
    auto departureElement = departureArray->getArray() + pidx;
    // the remained are 1. exiting particles in the front part 2. rear part
    if(pidx < (pclsArray->getNOP()-x) && departureElement->dest == 0)return; 
    
    if(departureElement->dest != 0){ // all exiting particles
        auto pcl = pclsArray->getpcls() + pidx;

        int index = 0;
        // get the index in exitingBuffer
        for(int i=0; i < departureElement->dest-1; i++){
            index += hashedSumArray[i].getSum(); // compact exiting buffer
        }
        // index in its direction
        index += hashedSumArray[departureElement->dest-1].getIndex(pidx, departureElement->hashedId);
        // copy the particle
        memcpy(exitingArray->getArray() + index, pcl, sizeof(SpeciesParticle));


        if(pidx >= (pclsArray->getNOP()-x))return;
        // holes
        departureElement->hashedId = hashedSumArray[6].add(pidx);

    }else{ // the fillers
        departureElement->hashedId = hashedSumArray[7].add(pidx);
    }

}







