#ifndef _PARTICLE_SOA_DEVICE_H_
#define _PARTICLE_SOA_DEVICE_H_

#include "cudaTypeDef.cuh"

/**
 * @brief Lightweight POD descriptor for SoA particle data on the GPU.
 *
 * Holds 8 device pointers (one per particle field) plus count and capacity.
 * Trivially copyable to device — pass by pointer to kernels.
 * No ownership semantics: the owning container (particleArrayCUDA) manages allocation.
 */
struct ParticleSoADevice {
    cudaParticleType* u;    // velocity x
    cudaParticleType* v;    // velocity y
    cudaParticleType* w;    // velocity z
    cudaParticleType* q;    // charge
    cudaParticleType* x;    // position x
    cudaParticleType* y;    // position y
    cudaParticleType* z;    // position z
    cudaParticleType* t;    // subcycle time / particle ID
    uint32_t nop;           // current number of particles
    uint32_t capacity;      // allocated capacity (elements per field)
};

static constexpr int PARTICLE_SOA_NUM_FIELDS = 8;

#endif
