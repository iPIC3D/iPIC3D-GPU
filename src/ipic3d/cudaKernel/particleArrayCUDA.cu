#include "particleArrayCUDA.cuh"
#include "particleArraySoAView.cuh"
#include "cudaTypeDef.cuh"

// ======= AoS to SoA scatter =======

/**
 * @brief Scatter an external device-side AoS staging buffer into the solver SoA layout.
 *
 * The kernel writes `count` particles from `aosStagingBuf[0..count-1]` into
 * `pclsArray[destOffset..destOffset+count-1]`.
 * @param aosStagingBuf Device AoS input buffer.
 * @param pclsArray Destination particle SoA buffer.
 * @param destOffset First destination index in `pclsArray`.
 * @param count Number of particles to scatter.
 */

__global__ void scatterAoSToSoAKernel(const SpeciesParticle* __restrict__ aosStagingBuf,
                                       particleArrayCUDA* pclsArray,
                                       uint32_t destOffset, uint32_t count,
                                       ParticleIDGenerator particleIDGenerator)
{
    const uint32_t tidx = blockIdx.x * blockDim.x + threadIdx.x;
    if (tidx >= count) return;
    const uint32_t pidx = destOffset + tidx;
    const SpeciesParticle& pcl = aosStagingBuf[tidx];
    pclsArray->getU()[pidx] = pcl.get_u();
    pclsArray->getV()[pidx] = pcl.get_v();
    pclsArray->getW()[pidx] = pcl.get_w();
    pclsArray->getQ()[pidx] = pcl.get_q();
    pclsArray->getX()[pidx] = pcl.get_x();
    pclsArray->getY()[pidx] = pcl.get_y();
    pclsArray->getZ()[pidx] = pcl.get_z();
    if (pclsArray->tracksParticleID()) {
        const cudaPclType_ID particleID = pcl.get_id();
        pclsArray->getID()[pidx] = particleID != PARTICLE_ID_INVALID
            ? particleID
            : particleIDGenerator.generateID();
    }
}

// ======= SoA view borrowing =======

template<>
__host__ void particleArraySoAView<cudaParticleType, 4>::borrowFrom(particleArrayCUDA* pclArray) {
    if (owning) { freeMemory(); owning = false; }
    nop = pclArray->getNOP();
    ptrs[0] = pclArray->getU();
    ptrs[1] = pclArray->getV();
    ptrs[2] = pclArray->getW();
    ptrs[3] = pclArray->getQ();
}



