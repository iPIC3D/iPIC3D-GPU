#include "injectionKernel.cuh"

#ifndef HIPIFLY
#include <curand_kernel.h>
#else
#include <hiprand/hiprand_kernel.h>
#endif

/**
 * @brief GPU kernel: inject Maxwellian particles directly into the SoA tail.
 *
 * One thread per particle.  Thread mapping:
 *   globalIdx → face (via pclOffset[]) → cell within face → subcell
 *
 * RNG: Philox counter-based (stateless, no warm-up cost).
 * Rejection: regenerate if position outside domain or |v| > c.
 */
__global__ void injectionKernel(
    particleArrayCUDA*         pclsArray,
    const injectionParameter*  params,
    uint32_t                   soaWriteOffset,
    unsigned long long         rngSeed)
{
    const int globalIdx = blockIdx.x * blockDim.x + threadIdx.x;
    const int blockStart = blockIdx.x * blockDim.x;
    const int remaining = params->totalInjected - blockStart;
    const unsigned int blockCount =
        remaining > blockDim.x ? blockDim.x : (remaining > 0 ? (unsigned int)remaining : 0u);
    const bool trackParticleID = pclsArray->tracksParticleID();

    __shared__ ParticleIDGenerator::counter_type blockBaseSequence;
    if (threadIdx.x == 0 && blockCount > 0 && trackParticleID)
        blockBaseSequence = params->particleIDGenerator.reserveSequenceBlock(blockCount);
    __syncthreads();

    if (globalIdx >= params->totalInjected) return;

    // --- 1. Find which face this thread belongs to ---
    int face = 5;
    for (int faceIdx = 0; faceIdx < 5; faceIdx++) {
        if (globalIdx < params->pclOffset[faceIdx + 1]) { face = faceIdx; break; }
    }
    const int localIdx = globalIdx - params->pclOffset[face];
    const InjectionFaceRange& fr = params->faces[face];

    // --- 2. Cell and subcell indices ---
    const int cellIdx    = localIdx / params->numParticlesPerCell;
    const int subcellIdx = localIdx % params->numParticlesPerCell;

    // 3D cell within face range
    const int nYZ = fr.nY * fr.nZ;
    const int lix = cellIdx / nYZ;
    const int ljy = (cellIdx % nYZ) / fr.nZ;
    const int lkz = cellIdx % fr.nZ;

    const int ix = fr.ixBeg + lix;
    const int iy = fr.iyBeg + ljy;
    const int iz = fr.izBeg + lkz;

    // Subcell position within the cell
    const int npcy = params->numPclPerCellY;
    const int npcz = params->numPclPerCellZ;
    const int ii = subcellIdx / (npcy * npcz);
    const int jj = (subcellIdx % (npcy * npcz)) / npcz;
    const int kk = subcellIdx % npcz;

    // --- 3. Cell corner (uniform grid, arithmetic only) ---
    const double cellLowX = params->gridXstart + (ix - 1) * params->gridDx;
    const double cellLowY = params->gridYstart + (iy - 1) * params->gridDy;
    const double cellLowZ = params->gridZstart + (iz - 1) * params->gridDz;

    // --- 4. RNG init (Philox: zero warm-up, per-thread sequence) ---
    curandStatePhilox4_32_10_t rngState;
    curand_init(rngSeed, (unsigned long long)globalIdx, 0, &rngState);

    // --- 5. Generate position + velocity with rejection ---
    double posX, posY, posZ, velX, velY, velZ;
    do {
        velX = params->thermalVelX * curand_normal_double(&rngState) + params->driftVelX;
        velY = params->thermalVelY * curand_normal_double(&rngState) + params->driftVelY;
        velZ = params->thermalVelZ * curand_normal_double(&rngState) + params->driftVelZ;

        posX = (ii + curand_uniform_double(&rngState)) * params->dxPerPcl + cellLowX;
        posY = (jj + curand_uniform_double(&rngState)) * params->dyPerPcl + cellLowY;
        posZ = (kk + curand_uniform_double(&rngState)) * params->dzPerPcl + cellLowZ;
    } while (posX < 0.0 || posX > params->domainLengthX ||
             posY < 0.0 || posY > params->domainLengthY ||
             posZ < 0.0 || posZ > params->domainLengthZ ||
             (velX * velX + velY * velY + velZ * velZ) > params->speedOfLightSq);

    // --- 6. Write 8 SoA fields (coalesced, non-overlapping with stayed prefix) ---
    // The ID field stores the particle identifier.
    const uint32_t writeIdx = soaWriteOffset + (uint32_t)globalIdx;

    pclsArray->getU()[writeIdx] = velX;
    pclsArray->getV()[writeIdx] = velY;
    pclsArray->getW()[writeIdx] = velZ;
    pclsArray->getQ()[writeIdx] = params->chargePerParticle;
    pclsArray->getX()[writeIdx] = posX;
    pclsArray->getY()[writeIdx] = posY;
    pclsArray->getZ()[writeIdx] = posZ;
    if (trackParticleID) {
        pclsArray->getID()[writeIdx] =
            params->particleIDGenerator.idFromSequence(blockBaseSequence + threadIdx.x);
    }
}
