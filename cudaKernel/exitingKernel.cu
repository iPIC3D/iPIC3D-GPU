
#include "cudaTypeDef.cuh"
#include "arrayCUDA.cuh"
#include "hashedSum.cuh"
#include "particleArrayCUDA.cuh"
#include "departureArray.cuh"



/**
 * @brief Copy the exiting particles of the species to ExitingBuffer
 * @details     By the end of Mover, the DepartureArray has been prepared, 
 *          host allocates the ExitingBuffer according to the Hashed SumUp value for each direction.
 *              This kernel is responsible for moving the exiting particles in the pclArray into the ExitingBuffer,
 *          according to the DepartureArray.  
 *              The exiting particles in ExitingBuffer are orginaized in their destinations, with random order adopted from hashedSum.
 * 
 * @param exitingArray The buffer used for exiting particles for 6 directions, the size and distriburtion are decided by the 6 hashedSum
 * 
 */
__global__ void exitingKernel(particleArrayCUDA* pclsArray, departureArrayType* departureArray, 
                                exitingArray* exitingArray, hashedSum* hashedSumArray){
                                    
    uint pidx = blockIdx.x * blockDim.x + threadIdx.x;
    if(pidx >= pclsArray->getNOP())return;

    // stayed particles
    auto departureElement = departureArray->getArray() + pidx;
    if(departureElement->dest == 0)return; 

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
    

}







