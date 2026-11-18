#ifndef PLANET_KERNEL_CUH
#define PLANET_KERNEL_CUH

#include "cudaTypeDef.cuh"
#include "hashedSum.cuh"
#include "particleArrayCUDA.cuh"
#include "particleExchange.cuh"

/**
 * @brief Persistent device-side state of the planet quasi-neutral boundary
 *        condition (one instance per MPI rank).
 *
 * The surface balances the charge it absorbs over time rather than within a
 * single cycle: `residualCharge` carries the absorbed ion charge that absorbed
 * electrons have not matched yet (negative when more electron charge than ion
 * charge was absorbed). The residual is not written to restart files and
 * restarts from zero.
 *
 * The diagnostic fields only ever grow; the host reads them when it writes the
 * diagnostics and differences successive snapshots, so no device-side reset
 * (a memset kernel competing with the mover) is needed.
 */
struct PlanetBCState {
  cudaParticleType ionChargeCycle; ///< |Q| of ions absorbed this cycle;
                                   ///< consumed and zeroed by
                                   ///< chargeCutoffKernel
  cudaParticleType residualCharge; ///< absorbed |Q_ion| - |Q_elec| not yet
                                   ///< balanced, carried across cycles

  // ── Diagnostics, cumulative since the start of the run ──
  cudaParticleType cutoffEnergySum; ///< sum of per-electron cutoff energies
  uint32_t cutoffSamples;           ///< cycles contributing to cutoffEnergySum
  uint32_t electronLimitedCycles;   ///< cycles in which every electron that hit
                                    ///< was absorbed and positive charge was
                                    ///< still left
};

/**
 * @brief The electrons that hit the planet in this cycle, of all electron
 *        species, in one merged index space and ranked by energy.
 *
 * An electron has the merged index speciesOffsets[s] + its index in the
 * planet buffer of electron species s. sortedGlobalIdx lists the merged
 * indices from the most to the least energetic electron (planetEnergyKernel,
 * then the bitonic sort).
 */
struct PlanetElectrons {
  planetArray** planetArrs;        ///< planet buffer of each electron species
  const int* speciesOffsets;       ///< merged index of each species' first
                                   ///< electron
  const uint32_t* sortedGlobalIdx; ///< merged indices by descending energy
  int nSpecies;                    ///< electron species
  int total;                       ///< electrons of all species
};

/**
 * @brief Planet geometry and reflection rule of planetReflectCompactKernel.
 */
struct PlanetReflection {
  cudaCommonType originX; ///< planet centre
  cudaCommonType originY;
  cudaCommonType originZ;
  cudaCommonType radius; ///< planet radius R
  int doSphere;          ///< 1: 3D sphere, 2: circle in the x-z plane
  bool diffuse;          ///< Lambert (cosine-law) directions, else specular
  uint32_t rngSeed;      ///< seed of the diffuse directions, new every cycle
};

/**
 * @brief Compact particles flagged PLANET into the planetArray (device-only).
 *        Uses hashedSum[PLANET_HASHEDSUM_INDEX] for scatter indices.
 *
 * @param pclsArray Device-side particle SoA container.
 * @param departureArray Device-side departure metadata array.
 * @param planetArr Device-side compact planet buffer.
 * @param hashedSumArray Device-side hashed-sum buckets used for scatter
 * indices.
 */
__global__ void planetExtractionKernel(particleArrayCUDA* pclsArray,
                                       departureArrayType* departureArray,
                                       planetArray* planetArr,
                                       hashedSum* hashedSumArray);

/**
 * @brief Reduce the total |q| of all particles in a species' planetArray.
 *        Result is atomicAdd'd into *chargeOut and *chargeAccum.
 *
 * @param planetArr Device-side planet buffer for one species.
 * @param count Number of valid planet particles in the buffer.
 * @param chargeOut Device pointer to the accumulated absolute charge, or
 * nullptr.
 * @param chargeAccum Optional second accumulator (diagnostics), or nullptr.
 */
__global__ void planetChargeReductionKernel(planetArray* planetArr, int count,
                                            cudaParticleType* chargeOut,
                                            cudaParticleType* chargeAccum);

/**
 * @brief Compute the kinetic energy of ONE physical electron for each electron
 *        planet particle, (gamma - 1) m c^2 with m = 1/|qom|. The macroparticle
 *        weight |q| does not enter, so species with different weights are
 *        ranked consistently. Writes energy and a global index
 *        (speciesOffset + localIndex) into buffers.
 *
 * The block size must be a multiple of WARP_SIZE.
 *
 * @param planetArr Device-side planet buffer for one electron species.
 * @param count Number of valid planet particles in the buffer.
 * @param qom Charge-to-mass ratio of the species (negative for electrons).
 * @param c Speed of light in code units.
 * @param energyBuf Output energy buffer shared across electron species.
 * @param globalIdxBuf Output global-index buffer shared across electron
 * species.
 * @param speciesOffset Offset into the merged energy/index buffers.
 * @param hitChargeAccum Optional accumulator of the species' total |q|
 * (diagnostics), or nullptr.
 */
__global__ void planetEnergyKernel(planetArray* planetArr, int count,
                                   cudaParticleType qom, cudaParticleType c,
                                   cudaParticleType* energyBuf,
                                   uint32_t* globalIdxBuf, int speciesOffset,
                                   cudaParticleType* hitChargeAccum);

/**
 * @brief One step of bitonic sort for (key, value) pairs.
 *        Sorts keys in DESCENDING order, permuting values alongside.
 *        One thread per compare-and-swap pair: launch n / 2 threads.
 *
 * @param keys Key buffer to sort in place.
 * @param values Value buffer permuted alongside @p keys.
 * @param j Distance between the two entries of a pair (a power of two).
 * @param k Bitonic stage size.
 * @param n Padded array length; must be a power of two.
 */
__global__ void bitonicSortStepKernel(cudaParticleType* __restrict__ keys,
                                      uint32_t* __restrict__ values, int j,
                                      int k, int n);

/**
 * @brief Pad the tail of the arrays for the bitonic sort: keys of -1e30,
 *        below any energy, and invalid values.
 *
 * @param keys Key buffer to pad.
 * @param values Value buffer to pad with invalid sentinels.
 * @param realN Number of valid entries before padding.
 * @param paddedN Padded power-of-two buffer length.
 */
__global__ void bitonicPadKernel(cudaParticleType* keys, uint32_t* values,
                                 int realN, int paddedN);

/**
 * @brief Sum |q| over each warp of the energy-sorted electrons, per electron
 *        species: warp w covers the sorted positions
 *        [w WARP_SIZE, (w + 1) WARP_SIZE) and writes
 *        warpCharge[w * nSpecies + s]. chargeCutoffKernel steps over whole
 *        warps with these sums.
 *
 * The block size must be a multiple of WARP_SIZE; launch one thread per
 * electron.
 *
 * @param electrons Energy-sorted electrons that hit the planet.
 * @param warpCharge Output, planetWarpChargeCount(total, nSpecies) values.
 */
__global__ void planetWarpChargeKernel(PlanetElectrons electrons,
                                       cudaParticleType* warpCharge);

/**
 * @brief Number of sums planetWarpChargeKernel writes: one per warp of the
 *        sorted electrons and electron species.
 * @param total Electrons of all species.
 * @param nSpecies Electron species.
 */
inline size_t planetWarpChargeCount(int total, int nSpecies) {
  return (size_t)((total + WARP_SIZE - 1) / WARP_SIZE) * nSpecies;
}

/**
 * @brief Find the cutoff index that best matches the charge the surface must
 *        neutralise (state->residualCharge + state->ionChargeCycle), then
 *        carry the leftover to the next cycle.
 *        Electrons at sorted positions [0, cutoff) are deleted (highest
 *        energy); those at [cutoff, total) survive and are reflected.
 *        Must also be launched when no electron hit the planet (total = 0) so
 *        the absorbed ion charge is carried over.
 *
 * Launch with one thread.
 *
 * @param electrons Energy-sorted electrons that hit the planet.
 * @param sortedEnergy Per-electron energies sorted in descending order.
 * @param warpCharge Warp sums of planetWarpChargeKernel (not read when
 * total = 0).
 * @param state Persistent planet boundary state (charge carry-over and scalar
 * diagnostics).
 * @param absorbedChargeAccum Optional per-electron-species accumulator of the
 * absorbed |q| (diagnostics), or nullptr.
 * @param cutoffIndex Output pointer to the first sorted index that survives.
 */
__global__ void chargeCutoffKernel(PlanetElectrons electrons,
                                   const cudaParticleType* sortedEnergy,
                                   const cudaParticleType* warpCharge,
                                   PlanetBCState* state,
                                   cudaParticleType* absorbedChargeAccum,
                                   int* cutoffIndex);

/**
 * @brief Reflect the surviving electrons and compact them into a contiguous
 *        output buffer, one segment per electron species.
 *        Reads the cutoff from device memory (no host sync needed). Thread t
 *        takes the survivor at sorted position cutoff + t, reflects a copy of
 *        it, and writes it to outputBuf[speciesOffset + slot], the slot from
 *        the species' atomic counter. Launch one thread per electron.
 *
 *        Placement: on the ray from the planet centre through the end point
 *        of the push, at max(1.05 R, 2R - r), r being the distance of the end
 *        point from the centre (in 2D, in the x-z plane).
 *
 *        Velocity, specular: only an inward normal component is reversed,
 *        v' = v - 2 min(v.n, 0) n. Diffuse: the speed is kept and the
 *        direction drawn by the Lambert cosine law on the outward hemisphere.
 *        Either way the electron never leaves moving inward.
 *
 * @param electrons Energy-sorted electrons that hit the planet.
 * @param cutoffDevice Device pointer written by chargeCutoffKernel.
 * @param reflection Planet geometry and reflection rule.
 * @param outputBuf Compact output buffer laid out by species offset.
 * @param survivorCounters Per-species atomic counters; must be zeroed before
 * launch.
 */
__global__ void planetReflectCompactKernel(PlanetElectrons electrons,
                                           const int* cutoffDevice,
                                           PlanetReflection reflection,
                                           SpeciesParticle* outputBuf,
                                           int* survivorCounters);

#endif // PLANET_KERNEL_CUH
