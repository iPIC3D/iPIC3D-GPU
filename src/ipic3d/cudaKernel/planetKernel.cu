
#include "arrayCUDA.cuh"
#include "cudaTypeDef.cuh"
#include "hashedSum.cuh"
#include "particleArrayCUDA.cuh"
#include "particleExchange.cuh"
#include "planetKernel.cuh"

// ======= Constants =======

constexpr cudaCommonType TWO_PI = (cudaCommonType)6.28318530717958647692;

// Relative tolerance below which a positive residual counts as balanced when
// flagging electron-limited cycles (absorbs summation round-off).
constexpr cudaParticleType PLANET_RESIDUAL_REL_TOL = (cudaParticleType)1e-12;

// Smallest distance from the planet centre, in planet radii, at which a
// reflected electron is placed: an electron that barely entered starts its
// next push clear of the surface rather than on it.
constexpr cudaCommonType PLANET_REFLECT_MIN_RADIUS = (cudaCommonType)1.05;

// ======= Planet extraction =======

/**
 * @brief Compact all particles flagged PLANET into the planet buffer, an AoS
 * array that stays on the device. Analogous to exitingKernel. Uses
 * hashedSum[PLANET_HASHEDSUM_INDEX] for scatter indices.
 * @param pclsArray Source particle SoA buffer.
 * @param departureArray Per-particle destination metadata.
 * @param planetArr Destination device-side planet particle buffer.
 * @param hashedSumArray Prefix-sum helpers for destination indexing.
 */
__global__ void planetExtractionKernel(particleArrayCUDA* pclsArray,
                                       departureArrayType* departureArray,
                                       planetArray* planetArr,
                                       hashedSum* hashedSumArray) {
  uint pidx = blockIdx.x * blockDim.x + threadIdx.x;
  if (pidx >= pclsArray->getNOP())
    return;

  auto dep = departureArray->getArray()[pidx];
  if (dep.dest != departureArrayElementType::PLANET)
    return;

  int index = hashedSumArray[departureArrayElementType::PLANET_HASHEDSUM_INDEX]
                  .getIndex(pidx, dep.hashedId);

  // Gather SoA fields into AoS SpeciesParticle for planet buffer
  SpeciesParticle pcl;
  pcl.set_u(pclsArray->getU()[pidx]);
  pcl.set_v(pclsArray->getV()[pidx]);
  pcl.set_w(pclsArray->getW()[pidx]);
  pcl.set_q(pclsArray->getQ()[pidx]);
  pcl.set_x(pclsArray->getX()[pidx]);
  pcl.set_y(pclsArray->getY()[pidx]);
  pcl.set_z(pclsArray->getZ()[pidx]);
  if (pclsArray->tracksParticleID())
    pcl.set_id(pclsArray->getID()[pidx]);
  else
    pcl.set_id(PARTICLE_ID_INVALID);
  planetArr->getArray()[index] = pcl;
}

// ======= Ion charge reduction =======

/**
 * @brief Sum |q| of all particles in one species' planet buffer.
 *        Block-level reduction then atomicAdd into *chargeOut and
 *        *chargeAccum (each skipped when nullptr).
 * @param planetArr Device buffer holding the extracted planet particles.
 * @param count Number of valid entries in `planetArr`.
 * @param chargeOut Device scalar accumulating the total absolute charge.
 * @param chargeAccum Optional second device scalar (diagnostics).
 */
__global__ void planetChargeReductionKernel(planetArray* planetArr, int count,
                                            cudaParticleType* chargeOut,
                                            cudaParticleType* chargeAccum) {
  extern __shared__ cudaParticleType sdata[];

  uint tid = threadIdx.x;
  uint gid = blockIdx.x * blockDim.x + threadIdx.x;

  // load
  cudaParticleType val = 0;
  if (gid < (uint)count) {
    val = fabs(planetArr->getArray()[gid].get_q());
  }
  sdata[tid] = val;
  __syncthreads();

  // tree reduction in shared mem
  for (uint s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s)
      sdata[tid] += sdata[tid + s];
    __syncthreads();
  }

  if (tid == 0) {
    if (chargeOut != nullptr)
      atomicAdd(chargeOut, sdata[0]);
    if (chargeAccum != nullptr)
      atomicAdd(chargeAccum, sdata[0]);
  }
}

// ======= Electron energy staging =======

/**
 * @brief Relativistic kinetic energy of one physical particle of unit charge,
 *        (gamma - 1) m c^2 with m = 1/|qom|. Evaluated as
 *        beta^2 gamma^2 / (gamma + 1) c^2 to avoid cancellation at low speed;
 *        reduces to v^2 / (2 |qom|) for v << c.
 * @param u X velocity.
 * @param v Y velocity.
 * @param w Z velocity.
 * @param qom Charge-to-mass ratio of the species.
 * @param c Speed of light in code units.
 * @return Kinetic energy per physical particle in code units.
 */
__device__ inline cudaParticleType
planetParticleEnergy(cudaParticleType u, cudaParticleType v, cudaParticleType w,
                     cudaParticleType qom, cudaParticleType c) {
  const cudaParticleType c2 = c * c;
  const cudaParticleType beta2 = (u * u + v * v + w * w) / c2;
  // The subcycling mover deletes particles with v >= c; the floor only keeps
  // gamma finite should one slip through.
  const cudaParticleType oneMinusBeta2 =
      fmax((cudaParticleType)1.0 - beta2, (cudaParticleType)1e-12);
  const cudaParticleType gamma = (cudaParticleType)1.0 / sqrt(oneMinusBeta2);
  return beta2 / (oneMinusBeta2 * (gamma + (cudaParticleType)1.0)) * c2 /
         fabs(qom);
}

/**
 * @brief For each electron planet particle compute the kinetic energy of one
 *        physical electron and write it into merged buffers at position
 *        (speciesOffset + localIndex). The macroparticle weight |q| does not
 *        enter the energy, so electron species with different weights (solar
 *        wind vs exosphere, merged particles) are ranked by the energy that
 *        decides whether an electron reaches the surface.
 *        Optionally accumulates the species' total |q| for diagnostics with
 *        one atomic per warp.
 * @param planetArr Device buffer holding one electron species' planet
 * particles.
 * @param count Number of valid entries in `planetArr`.
 * @param qom Charge-to-mass ratio of the species.
 * @param c Speed of light in code units.
 * @param energyBuf Merged output buffer for kinetic-energy keys.
 * @param globalIdxBuf Merged output buffer for global particle indices.
 * @param speciesOffset Prefix offset of this species in the merged buffers.
 * @param hitChargeAccum Optional device scalar accumulating the total |q|.
 */
__global__ void planetEnergyKernel(planetArray* planetArr, int count,
                                   cudaParticleType qom, cudaParticleType c,
                                   cudaParticleType* energyBuf,
                                   uint32_t* globalIdxBuf, int speciesOffset,
                                   cudaParticleType* hitChargeAccum) {
  const uint gid = blockIdx.x * blockDim.x + threadIdx.x;

  // No early return: every lane must reach the warp reduction below.
  cudaParticleType absq = 0;
  if (gid < (uint)count) {
    const SpeciesParticle& pcl = planetArr->getArray()[gid];
    absq = fabs(pcl.get_q());

    const int idx = speciesOffset + gid;
    energyBuf[idx] =
        planetParticleEnergy(pcl.get_u(), pcl.get_v(), pcl.get_w(), qom, c);
    globalIdxBuf[idx] = (uint32_t)idx;
  }

  if (hitChargeAccum != nullptr) {
    absq = warp_reduce_sum(absq);
    if ((threadIdx.x & (WARP_SIZE - 1)) == 0 && absq != 0)
      atomicAdd(hitChargeAccum, absq);
  }
}

// ======= Bitonic energy sort =======

/**
 * @brief Pad tail of arrays beyond realN with keys of -1e30 so they
 *        sink to the end under descending sort.
 * @param keys Sort-key buffer to pad.
 * @param values Payload buffer aligned with `keys`.
 * @param realN Number of valid entries before padding.
 * @param paddedN Power-of-two length used by the bitonic sort.
 */
__global__ void bitonicPadKernel(cudaParticleType* keys, uint32_t* values,
                                 int realN, int paddedN) {
  int i = blockIdx.x * blockDim.x + threadIdx.x + realN;
  if (i >= paddedN)
    return;
  keys[i] = (cudaParticleType)(-1e30);
  values[i] = 0xFFFFFFFFu;
}

/**
 * @brief One compare-and-swap step of bitonic sort.
 *        Produces a descending sequence when the outer loop completes.
 *        Thread t handles the pair (i, i + j), i being t with a 0 bit
 *        inserted at the position of j, so no thread is idle.
 * @param keys Sort keys updated in place.
 * @param values Payload buffer permuted in lockstep with `keys`.
 * @param j Distance between the two entries of a pair (a power of two)
 * @param k block size for this stage
 * @param n padded array length (power of 2)
 */
__global__ void bitonicSortStepKernel(cudaParticleType* __restrict__ keys,
                                      uint32_t* __restrict__ values, int j,
                                      int k, int n) {
  const int t = blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= n / 2)
    return;

  const int low = t & (j - 1);
  const int i = ((t - low) << 1) | low;
  const int ixj = i + j;

  // ascending = true when (i & k) != 0, descending otherwise
  const bool ascending = (i & k) != 0;
  const cudaParticleType keyI = keys[i];
  const cudaParticleType keyJ = keys[ixj];
  if (ascending ? keyI > keyJ : keyI < keyJ) {
    keys[i] = keyJ;
    keys[ixj] = keyI;

    const uint32_t tmpV = values[i];
    values[i] = values[ixj];
    values[ixj] = tmpV;
  }
}

// ======= Electron lookup =======

/**
 * @brief The electron of merged index gidx: its particle in the planet buffer
 *        of its species, and the species (index among the electron species).
 *
 *        The species is the last one whose offset is not above gidx (an empty
 *        species shares the offset of the next one).
 * @param electrons Electrons that hit the planet.
 * @param gidx Merged index, an entry of sortedGlobalIdx.
 * @param species Output: electron species of the particle.
 * @return The particle, or nullptr when gidx is no electron: the padding of
 * the sort, which leaves the tail of the sorted order only if an energy is
 * NaN.
 */
__device__ inline const SpeciesParticle*
planetElectronAt(const PlanetElectrons& electrons, uint32_t gidx,
                 int& species) {
  if (gidx >= (uint32_t)electrons.total)
    return nullptr;
  int s = electrons.nSpecies - 1;
  while (s > 0 && (uint32_t)electrons.speciesOffsets[s] > gidx)
    s--;
  species = s;
  return electrons.planetArrs[s]->getArray() +
         (gidx - (uint32_t)electrons.speciesOffsets[s]);
}

// ======= Charge cutoff =======

/**
 * @brief Sum |q| over each warp of the energy-sorted electrons, per electron
 *        species: warpCharge[w * nSpecies + s] for the warp of sorted
 *        positions [w WARP_SIZE, (w + 1) WARP_SIZE).
 *
 *        The gather of the particles, random in memory, is done here by all
 *        threads, so that chargeCutoffKernel, a single thread, steps over
 *        whole warps and reads one sum per warp and species.
 * @param electrons Energy-sorted electrons that hit the planet.
 * @param warpCharge Output, planetWarpChargeCount(total, nSpecies) values.
 */
__global__ void planetWarpChargeKernel(PlanetElectrons electrons,
                                       cudaParticleType* warpCharge) {
  const uint gid = blockIdx.x * blockDim.x + threadIdx.x;

  // No early return: every lane must reach the warp reductions below.
  cudaParticleType absq = 0;
  int species = -1;
  if (gid < (uint)electrons.total) {
    const SpeciesParticle* pcl =
        planetElectronAt(electrons, electrons.sortedGlobalIdx[gid], species);
    if (pcl != nullptr)
      absq = fabs(pcl->get_q());
  }

  // The first lane writes the sums of its warp; a warp past the last electron
  // has no entry in warpCharge.
  const bool writes =
      (threadIdx.x & (WARP_SIZE - 1)) == 0 && gid < (uint)electrons.total;
  cudaParticleType* const sums =
      warpCharge + (size_t)(gid / WARP_SIZE) * electrons.nSpecies;
  for (int s = 0; s < electrons.nSpecies; s++) {
    const cudaParticleType sum =
        warp_reduce_sum(species == s ? absq : (cudaParticleType)0);
    if (writes)
      sums[s] = sum;
  }
}

/**
 * @brief Find the cutoff index of the electrons sorted by descending energy:
 *        electrons [0..cutoff-1] are deleted, electrons [cutoff..total-1]
 *        survive and will be reflected.
 *
 *        The target is the charge the surface still has to neutralise: the
 *        ions absorbed this cycle plus the residual carried from previous
 *        cycles. The next most energetic electron is absorbed only while it
 *        brings the absorbed charge closer to the target, so the per-cycle
 *        mismatch is at most half an electron macroparticle, whatever the
 *        weights of the species involved. The remaining mismatch, of either
 *        sign, is carried to the next cycle.
 *
 *        Single-thread kernel. It first steps over whole warps of the sorted
 *        order with the sums of planetWarpChargeKernel: while the charge up
 *        to the end of a warp stays below the target, every electron of that
 *        warp is absorbed by the rule above. From the first warp that reaches
 *        the target it applies the rule one electron at a time, which ends
 *        within one electron of the crossing.
 *
 * @param electrons Energy-sorted electrons that hit the planet (total may be
 * 0)
 * @param sortedEnergy  per-electron energies in descending order
 * @param warpCharge  warp sums of |q| per electron species
 * (planetWarpChargeKernel); not read when total = 0
 * @param state  persistent charge carry-over and scalar diagnostics
 * @param absorbedChargeAccum  optional per-electron-species absorbed |q|
 * accumulator (diagnostics)
 * @param cutoffIndex  output: first sorted index that SURVIVES (reflects)
 */
__global__ void chargeCutoffKernel(PlanetElectrons electrons,
                                   const cudaParticleType* sortedEnergy,
                                   const cudaParticleType* warpCharge,
                                   PlanetBCState* state,
                                   cudaParticleType* absorbedChargeAccum,
                                   int* cutoffIndex) {
  if (threadIdx.x != 0 || blockIdx.x != 0)
    return;

  const int n = electrons.total;
  const int nSpecies = electrons.nSpecies;
  const cudaParticleType target = state->residualCharge + state->ionChargeCycle;

  // Whole warps.
  cudaParticleType absorbed = 0;
  int warpsAbsorbed = 0;
  for (; warpsAbsorbed * WARP_SIZE < n; warpsAbsorbed++) {
    const cudaParticleType* sums =
        warpCharge + (size_t)warpsAbsorbed * nSpecies;
    cudaParticleType warpSum = 0;
    for (int s = 0; s < nSpecies; s++)
      warpSum += sums[s];
    if (absorbed + warpSum >= target)
      break;
    absorbed += warpSum;
  }
  if (absorbedChargeAccum != nullptr) {
    for (int s = 0; s < nSpecies; s++) {
      cudaParticleType speciesSum = 0;
      for (int w = 0; w < warpsAbsorbed; w++)
        speciesSum += warpCharge[(size_t)w * nSpecies + s];
      absorbedChargeAccum[s] += speciesSum;
    }
  }

  // One electron at a time.
  int cutoff = min(warpsAbsorbed * WARP_SIZE, n);
  for (; cutoff < n; cutoff++) {
    int species;
    const SpeciesParticle* pcl =
        planetElectronAt(electrons, electrons.sortedGlobalIdx[cutoff], species);
    if (pcl == nullptr)
      continue;
    const cudaParticleType absq = fabs(pcl->get_q());

    // Stop once absorbing this electron would move away from the target
    // (ties are reflected).
    if (absorbed + (cudaParticleType)0.5 * absq >= target)
      break;
    absorbed += absq;
    if (absorbedChargeAccum != nullptr)
      absorbedChargeAccum[species] += absq;
  }
  *cutoffIndex = cutoff; // delete [0..cutoff-1], reflect [cutoff..n-1]

  const cudaParticleType residual = target - absorbed;
  state->residualCharge = residual;
  state->ionChargeCycle = 0;

  // ======= Diagnostics =======
  if (cutoff == n && residual > PLANET_RESIDUAL_REL_TOL * fabs(target))
    state->electronLimitedCycles++;
  if (cutoff > 0 && cutoff < n) {
    // Energy separating the absorbed from the reflected electrons.
    state->cutoffEnergySum += (cudaParticleType)0.5 *
                              (sortedEnergy[cutoff - 1] + sortedEnergy[cutoff]);
    state->cutoffSamples++;
  }
}

// ======= Reflection =======

/// Outward unit normal of the planet surface.
struct PlanetNormal {
  cudaCommonType x, y, z;
};

/**
 * @brief Move an electron that ended its push inside the planet back out, and
 *        return the outward unit normal n at its end position.
 *
 *        n is radial from the planet centre; for the 2D circle in the x-z
 *        plane (a cylinder along y) it lies in that plane and y is kept. The
 *        electron ended at distance r < R from the centre, a depth R - r below
 *        the surface; it is put on the same ray at
 *        max(PLANET_REFLECT_MIN_RADIUS R, 2R - r): the mirror image of its end
 *        point through the surface, i.e. as far out as the push took it in,
 *        but never closer to the surface than the minimum. The new position
 *        may lie outside the local subdomain; the particle exchange that
 *        follows sends it to the rank that owns it.  An electron exactly at
 *        the centre leaves along +x (in the x-z plane, so valid in 2D too).
 * @param pcl Particle whose position is updated in place.
 * @param planet Planet centre, radius and geometry (doSphere).
 * @return Outward unit normal at the end position of the push.
 */
__device__ inline PlanetNormal
planetPlaceReflected(SpeciesParticle& pcl, const PlanetReflection& planet) {
  const bool sphere3D = planet.doSphere == 1;
  const cudaCommonType dx = pcl.get_x() - planet.originX;
  const cudaCommonType dy =
      sphere3D ? pcl.get_y() - planet.originY : (cudaCommonType)0.0;
  const cudaCommonType dz = pcl.get_z() - planet.originZ;
  const cudaCommonType r = sqrt(dx * dx + dy * dy + dz * dz);
  const PlanetNormal n =
      r > (cudaCommonType)0.0
          ? PlanetNormal{dx / r, dy / r, dz / r}
          : PlanetNormal{(cudaCommonType)1.0, (cudaCommonType)0.0,
                         (cudaCommonType)0.0};

  const cudaCommonType rNew = fmax(PLANET_REFLECT_MIN_RADIUS * planet.radius,
                                   (cudaCommonType)2.0 * planet.radius - r);
  pcl.set_x(0, planet.originX + rNew * n.x);
  if (sphere3D)
    pcl.set_x(1, planet.originY + rNew * n.y);
  pcl.set_x(2, planet.originZ + rNew * n.z);
  return n;
}

/**
 * @brief Specular reflection of the inward normal velocity component,
 *        v' = v - 2 min(v.n, 0) n.
 *
 *        An electron moving into the planet leaves with the mirrored velocity;
 *        one already moving outward at its end point (its path inside the
 *        planet went past the deepest point, or turned) keeps its velocity,
 *        which a full mirror would point back into the planet. Either way
 *        v'.n >= 0, and |v| and the tangential component are kept. In 2D n
 *        lies in the x-z plane, so the y component is unchanged.
 * @param pcl Particle whose velocity is updated in place.
 * @param n Outward unit normal.
 */
__device__ inline void planetSpecularVelocity(SpeciesParticle& pcl,
                                              const PlanetNormal& n) {
  const cudaCommonType u = pcl.get_u();
  const cudaCommonType v = pcl.get_v();
  const cudaCommonType w = pcl.get_w();
  const cudaCommonType vnInward =
      fmin(u * n.x + v * n.y + w * n.z, (cudaCommonType)0.0);
  pcl.set_u(0, u - (cudaCommonType)2.0 * vnInward * n.x);
  pcl.set_u(1, v - (cudaCommonType)2.0 * vnInward * n.y);
  pcl.set_u(2, w - (cudaCommonType)2.0 * vnInward * n.z);
}

/**
 * @brief Advance a simple xorshift32 PRNG state.
 * @param seed Current PRNG state.
 * @return Updated PRNG state after one xorshift step.
 */
__device__ inline uint32_t planetRngHash(uint32_t seed) {
  seed ^= seed << 13;
  seed ^= seed >> 17;
  seed ^= seed << 5;
  return seed;
}

/**
 * @brief Return a uniform random variate in the open interval (0, 1).
 *
 * The xorshift state is never 0, so the variate is at least 2^-32; it stays
 * below 1 in double precision, where every state converts exactly.
 * @param state PRNG state updated in place.
 * @return Pseudorandom variate in `(0, 1)`.
 */
__device__ inline cudaCommonType planetRngUniform(uint32_t& state) {
  state = planetRngHash(state);
  return (cudaCommonType)(state) * (cudaCommonType)(1.0 / 4294967296.0);
}

/**
 * @brief Diffuse reflection: a random direction on the outward hemisphere by
 *        the Lambert cosine law, keeping the speed |v|.
 *
 *        The emitted flux per solid angle is proportional to cos(theta), so
 *        cos^2(theta) is uniform (<cos(theta)> = 2/3) and the azimuth is
 *        uniform. The same law holds for the 2D circle, a cylinder along y,
 *        about its in-plane normal, since the particles have three velocity
 *        components. The direction is drawn in a local frame whose z axis is
 *        the normal and rotated to the global frame by an orthonormal basis
 *        (t1, t2, n): no rejection loop, so the lanes of a warp do not
 *        diverge.
 *
 *        The xorshift state is never 0, so u1 >= 2^-32 and
 *        cos(theta) >= 2^-16: v'.n > 0 for any nonzero speed, far above the
 *        round-off of the rotation.
 * @param pcl Particle whose velocity is updated in place.
 * @param n Outward unit normal.
 * @param seed Seed of the cycle.
 * @param sortedIdx Position of the particle in the sorted order, which makes
 * its random stream distinct.
 */
__device__ inline void planetDiffuseVelocity(SpeciesParticle& pcl,
                                             const PlanetNormal& n,
                                             uint32_t seed,
                                             uint32_t sortedIdx) {
  uint32_t rngState = planetRngHash(seed ^ (sortedIdx * 2654435761u + 1u));
  if (rngState == 0u)
    rngState = 1u; // xorshift32 absorbs at 0

  const cudaCommonType u = pcl.get_u();
  const cudaCommonType v = pcl.get_v();
  const cudaCommonType w = pcl.get_w();
  const cudaCommonType speed = sqrt(u * u + v * v + w * w);

  const cudaCommonType u1 = planetRngUniform(rngState); // in (0, 1)
  const cudaCommonType cosTheta = sqrt(u1);
  const cudaCommonType sinTheta = sqrt((cudaCommonType)1.0 - u1);
  const cudaCommonType phi = TWO_PI * planetRngUniform(rngState);

  // Velocity in the local frame where z_local aligns with the normal.
  const cudaCommonType vl_x = speed * sinTheta * cos(phi);
  const cudaCommonType vl_y = speed * sinTheta * sin(phi);
  const cudaCommonType vl_z = speed * cosTheta;

  // Build an orthonormal basis (t1, t2, n).
  cudaCommonType t1x, t1y, t1z;
  if (fabs(n.x) < (cudaCommonType)0.9) {
    const cudaCommonType len = sqrt(n.z * n.z + n.y * n.y);
    const cudaCommonType invLen = (cudaCommonType)1.0 / len;
    t1x = (cudaCommonType)0.0;
    t1y = n.z * invLen;
    t1z = -n.y * invLen;
  } else {
    const cudaCommonType len = sqrt(n.z * n.z + n.x * n.x);
    const cudaCommonType invLen = (cudaCommonType)1.0 / len;
    t1x = -n.z * invLen;
    t1y = (cudaCommonType)0.0;
    t1z = n.x * invLen;
  }
  // t2 = cross(n, t1).
  const cudaCommonType t2x = n.y * t1z - n.z * t1y;
  const cudaCommonType t2y = n.z * t1x - n.x * t1z;
  const cudaCommonType t2z = n.x * t1y - n.y * t1x;

  // Rotate back to the global frame: v = vl_x * t1 + vl_y * t2 + vl_z * n.
  pcl.set_u(0, vl_x * t1x + vl_y * t2x + vl_z * n.x);
  pcl.set_u(1, vl_x * t1y + vl_y * t2y + vl_z * n.y);
  pcl.set_u(2, vl_x * t1z + vl_y * t2z + vl_z * n.z);
}

/**
 * @brief Reflect the surviving electrons and compact them into a contiguous
 *        output buffer, one segment per electron species.
 *
 *        Reads the cutoff from device memory (written by chargeCutoffKernel
 *        on the same stream, no host sync needed). Thread t takes the
 *        survivor at sorted position cutoff + t; threads past the last
 *        electron return. The particle is moved out of the planet
 *        (planetPlaceReflected), gets its new velocity
 *        (planetSpecularVelocity or planetDiffuseVelocity) and is written to
 *        outputBuf[speciesOffset + slot], the slot from the species' atomic
 *        counter.
 * @param electrons Energy-sorted electrons that hit the planet.
 * @param cutoffDevice Device pointer to the first surviving sorted index.
 * @param reflection Planet geometry and reflection rule.
 * @param outputBuf Output AoS buffer receiving the reflected survivors.
 * @param survivorCounters Per-species device counters of the survivors,
 * zeroed before launch.
 */
__global__ void planetReflectCompactKernel(PlanetElectrons electrons,
                                           const int* cutoffDevice,
                                           PlanetReflection reflection,
                                           SpeciesParticle* outputBuf,
                                           int* survivorCounters) {
  const uint sortedIdx =
      (uint)*cutoffDevice + blockIdx.x * blockDim.x + threadIdx.x;
  if (sortedIdx >= (uint)electrons.total)
    return;

  int species;
  const SpeciesParticle* source = planetElectronAt(
      electrons, electrons.sortedGlobalIdx[sortedIdx], species);
  if (source == nullptr)
    return;
  SpeciesParticle pcl = *source;

  const PlanetNormal n = planetPlaceReflected(pcl, reflection);
  if (reflection.diffuse)
    planetDiffuseVelocity(pcl, n, reflection.rngSeed, sortedIdx);
  else
    planetSpecularVelocity(pcl, n);

  const int slot = atomicAdd(&survivorCounters[species], 1);
  outputBuf[electrons.speciesOffsets[species] + slot] = pcl;
}
