#ifndef INJECTION_KERNEL_CUH
#define INJECTION_KERNEL_CUH

#include "ParticleIDGenerator.cuh"
#include "cudaTypeDef.cuh"
#include "particleArrayCUDA.cuh"

/**
 * @brief The cells of one of the 6 boundary faces
 * (ParticleCommInjection::injectionFaces): nCells cells from (ixBeg, iyBeg,
 * izBeg), z fastest.
 */
struct InjectionFaceRange {
  int ixBeg, iyBeg, izBeg; // the first cell
  int nY, nZ;              // cells along y and z
  int nCells;              // nX nY nZ, 0 if the face injects nothing
};

/**
 * @brief The parameters of the GPU injection kernel, which it takes by value
 * (KERNEL_CONSTANT).
 *
 * One instance per species, built once at initialization.
 */
struct injectionParameter {

  // --- Physics ---
  cudaParticleType thermalVelX, thermalVelY, thermalVelZ;
  cudaParticleType driftVelX, driftVelY, driftVelZ;
  cudaParticleType chargePerParticle;
  cudaParticleType speedOfLightSq; // c², for velocity rejection test

  // --- Subcell grid spacing ---
  cudaParticleType dxPerPcl, dyPerPcl, dzPerPcl;
  int numPclPerCellY, numPclPerCellZ;
  int numParticlesPerCell; // = npcelx * npcely * npcelz

  // --- Domain bounds (for position rejection) ---
  cudaParticleType domainLengthX, domainLengthY, domainLengthZ;

  // --- Cell size ---
  cudaParticleType gridDx, gridDy, gridDz;

  // --- Global cell indexing (RNG stream and cell corners) ---
  // Global index of local cell 1 per direction (Grid3DCU::getCellOffset*)
  // and the global cell counts in y and z. The cell corner is
  // globalCell * gridDx, and a particle's Philox stream is
  // (globalCell * numParticlesPerCell + subcell), so the injected plasma
  // depends only on (cycle, species, global cell, subcell): it differs from
  // rank to rank and does not depend on the decomposition.
  long long globalCellOffX, globalCellOffY, globalCellOffZ;
  long long globalNumCellsY, globalNumCellsZ;

  // --- 6 face ranges with cumulative particle offsets ---
  // Order: Xleft(0), Xright(1), Yleft(2), Yright(3), Zleft(4), Zright(5)
  InjectionFaceRange faces[6];
  int pclOffset[6];  // pclOffset[f] = sum of faces[0..f-1].nCells * nppc
  int totalInjected; // total particles across all 6 faces

  ParticleIDGenerator particleIDGenerator;
};
static_assert(std::is_trivially_copyable_v<injectionParameter>,
              "a kernel parameter (KERNEL_CONSTANT) is copied bytewise");

// ================================================================
// Kernel declaration
// ================================================================

/**
 * @brief GPU kernel: inject Maxwellian particles directly into SoA tail.
 *
 * One thread per particle.  Each thread:
 *  1. Determines its face, cell, and subcell from globalIdx
 *  2. Initializes a Philox RNG state (stateless, no warm-up)
 *  3. Generates position (uniform jitter) and velocity (Maxwellian)
 *  4. Applies rejection sampling (domain bounds + speed-of-light)
 *  5. Writes 8 SoA fields at soaWriteOffset + globalIdx
 *
 * @param pclsArray      Device SoA particle container
 * @param param          Injection parameters (faces, physics, grid)
 * @param soaWriteOffset First SoA index to write (= stayedParticle[i])
 * @param rngSeed        Per-cycle seed for Philox RNG
 */
__global__ void injectionKernel(particleArrayCUDA* pclsArray,
                                const KERNEL_CONSTANT injectionParameter param,
                                uint32_t soaWriteOffset,
                                unsigned long long rngSeed);

#endif // INJECTION_KERNEL_CUH
