#ifndef INJECTION_KERNEL_CUH
#define INJECTION_KERNEL_CUH

#include "ParticleIDGenerator.cuh"
#include "cudaTypeDef.cuh"
#include "gridCUDA.cuh"
#include "particleArrayCUDA.cuh"

/**
 * @brief Per-face injection cell range (pre-computed, constant across cycles).
 *
 * Encodes the 3D cell index range for one of the 6 boundary faces,
 * with the same narrowing logic used on the CPU side.
 */
struct InjectionFaceRange {
  int ixBeg, ixEnd;
  int iyBeg, iyEnd;
  int izBeg, izEnd;
  int nY, nZ; // iyEnd-iyBeg+1, izEnd-izBeg+1
  int nCells; // (ixEnd-ixBeg+1) * nY * nZ
  bool active;
};

/**
 * @brief Device-side parameter bundle for the GPU injection kernel.
 *
 * One instance per species.  All fields are constant across cycles
 * (set once at initialization, copied H→D, never updated).
 */
struct injectionParameter {

  // --- Physics ---
  cudaParticleType thermalVelX, thermalVelY, thermalVelZ;
  cudaParticleType driftVelX, driftVelY, driftVelZ;
  cudaParticleType chargePerParticle;
  cudaParticleType speedOfLightSq; // c², for velocity rejection test

  // --- Subcell grid spacing ---
  cudaParticleType dxPerPcl, dyPerPcl, dzPerPcl;
  int numPclPerCellX, numPclPerCellY, numPclPerCellZ;
  int numParticlesPerCell; // = npcelx * npcely * npcelz

  // --- Domain bounds (for position rejection) ---
  cudaParticleType domainLengthX, domainLengthY, domainLengthZ;

  // --- Grid origin (for computing cell-corner coordinates) ---
  // cellLowX = gridXstart + (ix - 1) * gridDx
  cudaParticleType gridXstart, gridYstart, gridZstart;
  cudaParticleType gridDx, gridDy, gridDz;

  // --- 6 face ranges with cumulative particle offsets ---
  // Order: Xleft(0), Xright(1), Yleft(2), Yright(3), Zleft(4), Zright(5)
  InjectionFaceRange faces[6];
  int pclOffset[6];  // pclOffset[f] = sum of faces[0..f-1].nCells * nppc
  int totalInjected; // total particles across all 6 faces

  ParticleIDGenerator particleIDGenerator;

  bool enabled; // master enable flag (false → kernel is a no-op)
};

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
 * @param params         Constant injection parameters (faces, physics, grid)
 * @param soaWriteOffset First SoA index to write (= stayedParticle[i])
 * @param rngSeed        Per-cycle seed for Philox RNG
 */
__global__ void injectionKernel(particleArrayCUDA* pclsArray,
                                const injectionParameter* params,
                                uint32_t soaWriteOffset,
                                unsigned long long rngSeed);

#endif // INJECTION_KERNEL_CUH
