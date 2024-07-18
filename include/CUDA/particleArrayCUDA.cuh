#ifndef _PARTICLE_ARRAY_CUDA_H_
#define _PARTICLE_ARRAY_CUDA_H_

#include <stdexcept>
#include <iostream>
#include "cudaTypeDef.cuh"
#include "Particle.h"
#include "Particles3D.h"
#include "cuda.h"
#include "arrayCUDA.cuh"

class particleArrayCUDA : public arrayCUDA<SpeciesParticle, 32>
{
private:
    ParticleType::Type type;

public:
    /**
     * @brief create and copy the array from host, 1.5 time size
     * 
     * @param stream the stream used for memory operations
     */ 
    __host__ particleArrayCUDA(Particles3D* p3D, cudaStream_t stream = 0): arrayCUDA(p3D->get_pclptr(0), p3D->getNOP(), 1.5), type(p3D->get_particleType()){
        assignStream(stream);
    }

    __host__ virtual particleArrayCUDA* copyToDevice(){
        particleArrayCUDA* ptr = nullptr;
        cudaErrChk(cudaMallocAsync((void**)&ptr, sizeof(particleArrayCUDA), stream));
        cudaErrChk(cudaMemcpyAsync(ptr, this, sizeof(particleArrayCUDA), cudaMemcpyDefault, stream));

        cudaErrChk(cudaStreamSynchronize(stream));
        return ptr;
    }


    __host__ __device__ uint32_t getNOP(){ return getNOE(); }
    __host__ __device__ SpeciesParticle* getpcls(){ return getArray(); }
    __host__ __device__ SpeciesParticle* getpcl(uint32_t index){ return getElement(index); }

    


};



#endif