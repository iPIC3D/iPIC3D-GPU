#ifndef _PARTICLE_SOA_DEVICE_H_
#define _PARTICLE_SOA_DEVICE_H_

#include "cudaTypeDef.cuh"

/**
 * @brief Lightweight POD descriptor for SoA particle data on the GPU.
 *
 * Holds device pointers for particle fields plus count and capacity.
 * The ID pointer is null when particle tracking is disabled for this species.
 * Trivially copyable to device — pass by pointer to kernels.
 * No ownership semantics: the owning container (particleArrayCUDA) manages allocation.
 */
struct ParticleSoADevice {
    cudaPclType_U* u;    // velocity x
    cudaPclType_V* v;    // velocity y
    cudaPclType_W* w;    // velocity z
    cudaPclType_Q* q;    // charge
    cudaPclType_X* x;    // position x
    cudaPclType_Y* y;    // position y
    cudaPclType_Z* z;    // position z
    cudaPclType_ID* id;   // particle ID, null when tracking is disabled
    uint32_t nop;        // current number of particles
    uint32_t capacity;   // allocated capacity (elements per field)
    bool trackParticleID;
};

#endif
