
#include "cudaTypeDef.cuh"
#include "arrayCUDA.cuh"
#include "hashedSum.cuh"
#include "particleArrayCUDA.cuh"
#include "particleExchange.cuh"



/**
 * @brief   Copy the exiting particles of the species to the SoA exiting buffer. Launch (nop) threads.
 * @details By the end of Mover, the DepartureArray has been prepared, host allocates the SoA exiting buffer
 *          according to the Hashed SumUp value for each direction.
 *          This kernel is responsible for moving the exiting particles from the main SoA arrays into the
 *          SoA exiting buffer, according to the DepartureArray.
 *          The exiting particles are organised by destination, with random order adopted from hashedSum.
 *
 *          This kernel is also responsible for preparing the 2 hashedSum for SortingKernel1 and SortingKernel2.
 *          It will modify the elements of the departure array.
 *
 * @param exitingSoA  SoA buffer for exiting particles (8 device arrays), sized by the 6 hashedSum
 * @param hashedSumArray 10 hashedSum, 6 from the Mover, 1 for deleted, 1 for planet, 2 for Sorting.
 */
__global__ void exitingKernel(particleArrayCUDA* pclsArray, departureArrayType* departureArray, 
                                ParticleSoADevice* exitingSoA, hashedSum* hashedSumArray){
                                    
    uint pidx = blockIdx.x * blockDim.x + threadIdx.x;
    if(pidx >= pclsArray->getNOP())return;

    __shared__ int x; // y, the number of holes (eixitng + deleted + planet)
    if(threadIdx.x == 0){ 
        x = 0; 
        for(int i=0; i <= departureArrayElementType::PLANET_HASHEDSUM_INDEX ; i++)x += hashedSumArray[i].getSum(); 
    }
    __syncthreads();
    
    auto departureElement = departureArray->getArray() + pidx;
    // return the stayed particles in the front part
    if(pidx < (pclsArray->getNOP()-x) && departureElement->dest == 0)return; 
    
    // Exiting particles — direct SoA-to-SoA copy (no AoS packing)
    if(departureElement->dest > 0 && departureElement->dest < departureArrayElementType::DELETE){ 

        int index = 0;
        // get the index in exitingBuffer
        for(int i=0; i < departureElement->dest-1; i++){
            index += hashedSumArray[i].getSum(); // compact exiting buffer
        }
        // index in its direction
        index += hashedSumArray[departureElement->dest-1].getIndex(pidx, departureElement->hashedId);
        // Direct SoA copy — no AoS intermediary
        exitingSoA->u[index] = pclsArray->getU()[pidx];
        exitingSoA->v[index] = pclsArray->getV()[pidx];
        exitingSoA->w[index] = pclsArray->getW()[pidx];
        exitingSoA->q[index] = pclsArray->getQ()[pidx];
        exitingSoA->x[index] = pclsArray->getX()[pidx];
        exitingSoA->y[index] = pclsArray->getY()[pidx];
        exitingSoA->z[index] = pclsArray->getZ()[pidx];
        exitingSoA->t[index] = pclsArray->getT()[pidx];
    }

    // holes
    if(departureElement->dest !=0){
        if(pidx >= (pclsArray->getNOP()-x))return; // return holes in the back part
        departureElement->hashedId = hashedSumArray[departureArrayElementType::HOLE_HASHEDSUM_INDEX].add(pidx);
        return; // return all holes
    }

    // Only fillers reach here
    if(pidx >= (pclsArray->getNOP()-x)){ 
        departureElement->hashedId = hashedSumArray[departureArrayElementType::FILLER_HASHEDSUM_INDEX].add(pidx);
    }

}






