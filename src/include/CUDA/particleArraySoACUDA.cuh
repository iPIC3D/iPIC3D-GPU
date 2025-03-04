#ifndef _PARTICLE_ARRAY_SOA_CUDA_H_
#define _PARTICLE_ARRAY_SOA_CUDA_H_

#include "Particle.h"
#include "particleArrayCUDA.cuh"
#include "arrayCUDA.cuh"
#include "cudaTypeDef.cuh"

namespace particleArraySoA{

enum particleArraySoAElement{
    U = 0,
    V = 1,
    W = 2,
    Q = 3,
    X = 4,
    Y = 5,
    Z = 6,
    T = 7
};


template<typename T, int startElement = 0, int stopElement = 6>
class particleArraySoACUDA
{

private:
    int nop;
    int size;

    // T* u;
    // T* v;
    // T* w;
    // T* q;
    // T* x;
    // T* y;
    // T* z;
    // T* t;
    
    T* elementPtr[8]; // array of pointers to the elements

    bool allocated = true;

private:
    __host__ void allocateMemory(){
        for (int i = startElement; i <= stopElement; i++){ // only allocate memory for u, v, w
            cudaErrChk(cudaMalloc(&(elementPtr[i]), size * sizeof(T)));
        }
    }

    __host__ void freeMemory(){
        for (int i = startElement; i <= stopElement; i++){
            cudaErrChk(cudaFree(elementPtr[i]));
        }
    }


    
public:
    /**
     * @brief Create SoA device particle array with AoS device pclArray
     */
    __host__ particleArraySoACUDA(particleArrayCUDA* pclArray, cudaStream_t stream = 0);

    __host__ particleArraySoACUDA(): allocated(false){}

    __host__ particleArraySoACUDA(int nop, cudaStream_t stream = 0): nop(nop), size(nop), allocated(true){
        allocateMemory();
    }

    __host__ void updateFromAoS(particleArrayCUDA* pclArray, cudaStream_t stream = 0);


    __host__ __device__ T* getElement(int i) const {
        return elementPtr[i];
    }

    __host__ __device__ int getNOP() const {
        return nop;
    }


    __host__ ~particleArraySoACUDA(){
        if (allocated)
        freeMemory();
    }
};






} // namespace particleArraySoA

#endif