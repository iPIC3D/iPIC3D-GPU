#include "injectionKernel.cuh"

#ifndef HIPIFLY
#include <curand_kernel.h>
#else
#include <hiprand/hiprand_kernel.h>
#endif

/** The GPU injection (see injectionKernel.cuh). */
__global__ void injectionKernel(particleArrayCUDA* pclsArray,
                                const KERNEL_CONSTANT injectionParameter param,
                                uint32_t soaWriteOffset,
                                unsigned long long rngSeed) {
  const int globalIdx = blockIdx.x * blockDim.x + threadIdx.x;
  const int blockStart = blockIdx.x * blockDim.x;
  const int remaining = param.totalInjected - blockStart;
  const unsigned int blockCount =
      remaining > blockDim.x ? blockDim.x
                             : (remaining > 0 ? (unsigned int)remaining : 0u);
  const bool trackParticleID = pclsArray->tracksParticleID();

  __shared__ ParticleIDGenerator::counter_type blockBaseSequence;
  if (threadIdx.x == 0 && blockCount > 0 && trackParticleID)
    blockBaseSequence =
        param.particleIDGenerator.reserveSequenceBlock(blockCount);
  __syncthreads();

  if (globalIdx >= param.totalInjected)
    return;

  // --- 1. The face of this thread: the last one that starts at or before
  // it (pclOffset[0] = 0); unrolled, so that param is indexed by constants
  // only (see KERNEL_CONSTANT).
  int localIdx = globalIdx;
  InjectionFaceRange fr = param.faces[0];
#pragma unroll
  for (int f = 1; f < 6; f++) {
    if (globalIdx >= param.pclOffset[f]) {
      localIdx = globalIdx - param.pclOffset[f];
      fr = param.faces[f];
    }
  }

  // --- 2. Cell and subcell indices ---
  const int cellIdx = localIdx / param.numParticlesPerCell;
  const int subcellIdx = localIdx % param.numParticlesPerCell;

  // 3D cell within face range
  const int nYZ = fr.nY * fr.nZ;
  const int lix = cellIdx / nYZ;
  const int ljy = (cellIdx % nYZ) / fr.nZ;
  const int lkz = cellIdx % fr.nZ;

  const int ix = fr.ixBeg + lix;
  const int iy = fr.iyBeg + ljy;
  const int iz = fr.izBeg + lkz;

  // Subcell position within the cell
  const int npcy = param.numPclPerCellY;
  const int npcz = param.numPclPerCellZ;
  const int ii = subcellIdx / (npcy * npcz);
  const int jj = (subcellIdx % (npcy * npcz)) / npcz;
  const int kk = subcellIdx % npcz;

  // --- 3. Global cell index and cell corner ---
  // Local cell 1 is global cell globalCellOff*. The corner is computed from
  // the global index so that it is bitwise the same for any decomposition.
  const long long gx = param.globalCellOffX + (ix - 1);
  const long long gy = param.globalCellOffY + (iy - 1);
  const long long gz = param.globalCellOffZ + (iz - 1);
  const double cellLowX = (double)gx * param.gridDx;
  const double cellLowY = (double)gy * param.gridDy;
  const double cellLowZ = (double)gz * param.gridDz;

  // --- 4. RNG init (Philox: zero warm-up, O(1) in the stream number) ---
  // Stream = global particle slot (global cell, subcell): unique in the box
  // for a given (cycle, species) seed, independent of the rank layout.
  // Philox puts the stream in the counter's high 64 bits, so its size
  // does not change the cost; do not switch to XORWOW, whose curand_init
  // skips ahead by matrix powers.
  const unsigned long long stream =
      (unsigned long long)((gx * param.globalNumCellsY + gy) *
                               param.globalNumCellsZ +
                           gz) *
          (unsigned long long)param.numParticlesPerCell +
      (unsigned long long)subcellIdx;
  curandStatePhilox4_32_10_t rngState;
  curand_init(rngSeed, stream, 0, &rngState);

  // --- 5. Generate position + velocity with rejection ---
  double posX, posY, posZ, velX, velY, velZ;
  do {
    velX =
        param.thermalVelX * curand_normal_double(&rngState) + param.driftVelX;
    velY =
        param.thermalVelY * curand_normal_double(&rngState) + param.driftVelY;
    velZ =
        param.thermalVelZ * curand_normal_double(&rngState) + param.driftVelZ;

    posX = (ii + curand_uniform_double(&rngState)) * param.dxPerPcl + cellLowX;
    posY = (jj + curand_uniform_double(&rngState)) * param.dyPerPcl + cellLowY;
    posZ = (kk + curand_uniform_double(&rngState)) * param.dzPerPcl + cellLowZ;
  } while (posX < 0.0 || posX > param.domainLengthX || posY < 0.0 ||
           posY > param.domainLengthY || posZ < 0.0 ||
           posZ > param.domainLengthZ ||
           (velX * velX + velY * velY + velZ * velZ) > param.speedOfLightSq);

  // --- 6. Write 8 SoA fields (coalesced, non-overlapping with stayed prefix)
  // --- The ID field stores the particle identifier.
  const uint32_t writeIdx = soaWriteOffset + (uint32_t)globalIdx;

  pclsArray->getU()[writeIdx] = velX;
  pclsArray->getV()[writeIdx] = velY;
  pclsArray->getW()[writeIdx] = velZ;
  pclsArray->getQ()[writeIdx] = param.chargePerParticle;
  pclsArray->getX()[writeIdx] = posX;
  pclsArray->getY()[writeIdx] = posY;
  pclsArray->getZ()[writeIdx] = posZ;
  if (trackParticleID) {
    pclsArray->getID()[writeIdx] = param.particleIDGenerator.idFromSequence(
        blockBaseSequence + threadIdx.x);
  }
}
