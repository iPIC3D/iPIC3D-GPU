#include "ExosphereIonization.h"

#include "Collective.h"
#include "Grid3DCU.h"
#include "VCtopology3D.h"

#include <cassert>
#include <cmath>
#include <numeric> // std::accumulate

// ─────────────────────────────────────────────────────────────────────
//  Constructor — allocates persistent per-species buffers and RNGs
// ─────────────────────────────────────────────────────────────────────

ExosphereIonization::ExosphereIonization(const Collective* collective,
                                         const Grid3DCU* grid,
                                         const VCtopology3D* topology)
    : collective(collective), grid(grid), topology(topology),
      numPlanetarySpecies(collective->getNumPlanetarySpecies()),
      firstPlanetarySpeciesIndex(collective->getNumSolarWindSpecies()),
      fourPi(16.0 * std::atan(1.0)),
      particlesPerCellSW(collective->getNpcel(0)) {
  const int n = std::max(numPlanetarySpecies, 1);

  // Allocate persistent particle buffers (one per planetary species).
  speciesParticleBuffers = std::make_unique<std::vector<SpeciesParticle>[]>(n);

  // Allocate per-species RNGs with deterministic, rank-dependent seeds.
  //
  // Quasi-neutrality guarantee: paired electron-ion species (sharing the same
  // neutral parent) are seeded with the SAME RNG state via pairIndex = i / 2.
  // Because both species in a pair have identical neutral parameters (surface
  // density, scale height, ionization frequency, weight factor), they:
  //   1. Compute the same expectedParticleCount for every cell
  //   2. Make the same stochastic rounding decision (same RNG state)
  //   3. Inject the same number of particles per cell (strict quasi-neutrality)
  //   4. Place particles at the same positions (co-located electron-ion pairs)
  //   5. Sample different velocities only because uth/vth/wth differ by species
  //
  // Convention: planetary species are ordered as (e0, i0, e1, i1, ...),
  // so neutralIndex 0 and 1 form pair 0, neutralIndex 2 and 3 form pair 1, etc.
  speciesRNGs = std::make_unique<std::mt19937_64[]>(n);
  const int mpiRank = topology->getCartesian_rank();
  for (int i = 0; i < n; i++) {
    const int pairIndex = i / 2; // electron-ion pair index
    std::seed_seq seed{mpiRank, pairIndex,
                       0x50494333}; // "PIC3" as extra entropy
    speciesRNGs[i].seed(seed);
  }

  // Allocate per-species charge accumulators.
  speciesInjectedCharge = std::make_unique<double[]>(n);
  for (int i = 0; i < n; i++)
    speciesInjectedCharge[i] = 0.0;

  // Validate: max injection radius must exceed planet radius
  assert(collective->getMaxInjectionRadius() > collective->getPlanet_radius() &&
         "RmaxExosphereInjection must be larger than Planet_radius");
}

// ─────────────────────────────────────────────────────────────────────
//  Chamberlain neutral density profile
// ─────────────────────────────────────────────────────────────────────

double ExosphereIonization::neutralDensity(double surfaceDensity,
                                           double distance, double planetRadius,
                                           double scaleHeight) {
  if (distance > planetRadius)
    return surfaceDensity * std::exp(-(distance - planetRadius) / scaleHeight);
  else
    return 0.0; // inside the planet
}

// ─────────────────────────────────────────────────────────────────────
//  Thread-safe Box-Muller Maxwellian velocity sampling
// ─────────────────────────────────────────────────────────────────────

void ExosphereIonization::sampleMaxwellianThreadSafe(
    double& u, double& v, double& w, double ut, double vt, double wt, double u0,
    double v0, double w0, std::mt19937_64& rng,
    std::uniform_real_distribution<double>& dist) {
  // Box-Muller pair 1 → (u, v)
  // dist(rng) generates [0, 1).  1.0 - dist(rng) gives (0, 1] to avoid log(0).
  const double r1 = std::sqrt(-2.0 * std::log(1.0 - dist(rng)));
  const double theta1 = 2.0 * M_PI * dist(rng);
  u = u0 + ut * r1 * std::cos(theta1);
  v = v0 + vt * r1 * std::sin(theta1);

  // Box-Muller pair 2 → w  (discard the second variate)
  const double r2 = std::sqrt(-2.0 * std::log(1.0 - dist(rng)));
  const double theta2 = 2.0 * M_PI * dist(rng);
  w = w0 + wt * r2 * std::cos(theta2);
}

// ─────────────────────────────────────────────────────────────────────
//  Diagnostics
// ─────────────────────────────────────────────────────────────────────

int ExosphereIonization::getLastInjectedCount(int speciesIndex) const {
  const int neutralIndex = speciesIndex - firstPlanetarySpeciesIndex;
  assert(neutralIndex >= 0 && neutralIndex < numPlanetarySpecies);
  return static_cast<int>(speciesParticleBuffers[neutralIndex].size());
}

double ExosphereIonization::getTotalInjectedCharge() const {
  double total = 0.0;
  for (int i = 0; i < numPlanetarySpecies; i++)
    total += speciesInjectedCharge[i];
  return total;
}

double ExosphereIonization::getSpeciesInjectedCharge(int speciesIndex) const {
  const int neutralIndex = speciesIndex - firstPlanetarySpeciesIndex;
  assert(neutralIndex >= 0 && neutralIndex < numPlanetarySpecies);
  return speciesInjectedCharge[neutralIndex];
}

/**
 * @brief Sample ionized exosphere particles for one planetary species on the
 * CPU.
 *
 * Thread safety:
 * Each species accesses only its own RNG, output buffer, and charge
 * accumulator. Grid and Collective accessors are read-only, and no global
 * mutable RNG state is used, so concurrent calls for different `speciesIndex`
 * values are safe.
 *
 * Performance notes:
 * Persistent per-species buffers avoid per-step allocation, the injection-rate
 * constant is precomputed once per call, squared-distance checks avoid extra
 * square roots, and the function exits early when injection parameters are
 * zero.
 *
 * @param speciesIndex Global species index of the planetary species to sample.
 * @param maxParticles Optional upper bound on the number of injected particles.
 * @return Reference to the persistent per-species output buffer for this call.
 */
const std::vector<SpeciesParticle>&
ExosphereIonization::sampleIonizedParticles(int speciesIndex,
                                            int maxParticles) {
  // ── Map global species index to neutral-species buffer index ──
  const int neutralIndex = speciesIndex - firstPlanetarySpeciesIndex;
  assert(neutralIndex >= 0 && neutralIndex < numPlanetarySpecies);

  // ── Get reference to this species' persistent buffer and reset particle
  // count ──
  std::vector<SpeciesParticle>& buffer = speciesParticleBuffers[neutralIndex];
  buffer.clear(); // resets size to 0 but preserves allocated capacity

  speciesInjectedCharge[neutralIndex] = 0.0;

  // ── Physical parameters from input file ──
  const double planetRadius = collective->getPlanet_radius();
  const double surfaceDensity =
      collective->getNeutralSurfaceDensity(neutralIndex);
  const double ionizationFreq =
      collective->getPhotoionizationFrequency(neutralIndex);
  const double exoScaleHeight =
      collective->getExosphericScaleHeight(neutralIndex);
  const double weightFactor =
      collective->getMacroParticleWeightRatio(neutralIndex);
  const double maxInjectionRadius = collective->getMaxInjectionRadius();

  // ── Early exit: nothing to inject if density or frequency is zero ──
  if (surfaceDensity == 0.0 || ionizationFreq == 0.0)
    return buffer;

  // ── Planet center coordinates ──
  const double planetCenterX = collective->getx_center_planet();
  const double planetCenterY = collective->gety_center_planet();
  const double planetCenterZ = collective->getz_center_planet();

  // ── Grid parameters (constant across all cells) ──
  const double cellSpacingX = grid->getDX();
  const double cellSpacingY = grid->getDY();
  const double cellSpacingZ = grid->getDZ();
  const double cellVolume = grid->getVOL();
  const double timeStep = collective->getDt();

  // ── Species parameters ──
  const double chargeToMassRatio = collective->getQOM(speciesIndex);
  const double thermalVelocityX = collective->getUth(speciesIndex);
  const double thermalVelocityY = collective->getVth(speciesIndex);
  const double thermalVelocityZ = collective->getWth(speciesIndex);
  const double driftVelocityX = collective->getU0(speciesIndex);
  const double driftVelocityY = collective->getV0(speciesIndex);
  const double driftVelocityZ = collective->getW0(speciesIndex);

  // ── Pre-computed loop-invariant quantities ──

  // Charge per injected macro-particle:
  //   q = sign(q/m) * VOL / (npcel_sw * 4π * weightFactor)
  const double chargeSign = (chargeToMassRatio > 0.0) ? 1.0 : -1.0;
  const double chargePerParticle =
      chargeSign * cellVolume / (particlesPerCellSW * fourPi * weightFactor);

  // Injection rate constant (everything in N_inject that doesn't depend on
  // position):
  //   injectionRateConstant = npcel_sw * weightFactor * dt * ionizationFreq
  // Then: N_inject(cell) = injectionRateConstant * neutralDensity(cell)
  const double injectionRateConstant =
      particlesPerCellSW * weightFactor * timeStep * ionizationFreq;

  // Squared radii for fast boundary checks (avoids sqrt)
  const double planetRadiusSquared = planetRadius * planetRadius;
  const double maxInjectionRadiusSquared =
      maxInjectionRadius * maxInjectionRadius;

  // ── Per-species RNG (thread-safe: each species uses its own) ──
  std::mt19937_64& rng = speciesRNGs[neutralIndex];
  std::uniform_real_distribution<double> uniformDist(0.0, 1.0); // [0, 1)

  // ── Grid loop dimensions ──
  const int numCellsX = grid->getNXC();
  const int numCellsY = grid->getNYC();
  const int numCellsZ = grid->getNZC();

  // ── Memory budget: 0 means unlimited ──
  const bool hasBudget = (maxParticles > 0);

  // ── Charge accumulator (local to this species) ──
  double chargeAccumulator = 0.0;

  // ── Loop over local grid cells (excluding ghost cells) ──
  for (int ix = 1; ix < numCellsX - 1; ix++) {
    for (int iy = 1; iy < numCellsY - 1; iy++) {
      for (int iz = 1; iz < numCellsZ - 1; iz++) {

        // Cell center position relative to planet center
        const double relativePosX = grid->getXC(ix, iy, iz) - planetCenterX;
        const double relativePosY = grid->getYC(ix, iy, iz) - planetCenterY;
        const double relativePosZ = grid->getZC(ix, iy, iz) - planetCenterZ;

        const double distanceSquared = relativePosX * relativePosX +
                                       relativePosY * relativePosY +
                                       relativePosZ * relativePosZ;

        // Fast boundary check using squared distances (no sqrt)
        if (distanceSquared <= planetRadiusSquared ||
            distanceSquared >= maxInjectionRadiusSquared)
          continue;

        // Only now compute sqrt (needed for exponential profile)
        const double distanceFromPlanet = std::sqrt(distanceSquared);

        // Neutral density via Chamberlain profile
        const double neutralDens = neutralDensity(
            surfaceDensity, distanceFromPlanet, planetRadius, exoScaleHeight);

        // Expected number of newly ionized macro-particles in this cell:
        //   N_inject = injectionRateConstant * neutralDensity
        const double expectedParticleCount =
            injectionRateConstant * neutralDens;
        if (expectedParticleCount <= 0.0)
          continue;

        // Stochastic rounding: integer part is always injected; the fractional
        // remainder is treated as a probability of injecting one additional
        // particle.  This correctly handles sub-unity injection rates (common
        // at low ionization frequencies) and preserves the expected injection
        // rate on average.
        const int baseCount = static_cast<int>(expectedParticleCount);
        const double fractional = expectedParticleCount - baseCount;
        int numParticlesToInject =
            baseCount + (uniformDist(rng) < fractional ? 1 : 0);
        if (numParticlesToInject <= 0)
          continue;

        // Enforce memory budget: clamp injection count.
        // When exhausted, continue skips the rest of this cell;
        // subsequent cells re-check and skip too (negligible cost:
        // only the distance check runs, no RNG or particle creation).
        if (hasBudget) {
          const int remaining = maxParticles - static_cast<int>(buffer.size());
          if (remaining <= 0)
            break; // budget exhausted, stop injecting more particles
          if (numParticlesToInject > remaining)
            numParticlesToInject = remaining;
        }

        // Cell center (cached for particle position sampling)
        const double cellCenterX = grid->getXC(ix, iy, iz);
        const double cellCenterY = grid->getYC(ix, iy, iz);
        const double cellCenterZ = grid->getZC(ix, iy, iz);

        // ── Sample each particle ──
        for (int ip = 0; ip < numParticlesToInject; ip++) {

          // Uniform random position within the cell (using thread-safe RNG)
          // Clamp to [node(1) + eps, node(nxn-2) - eps] to guarantee the
          // moment kernel's ix/iy/iz stay within [2, nxc-1] and ix-1 >= 1.
          const double eps = 1e-12;
          const double xLo = grid->getXstart() + eps;
          const double xHi = grid->getXend() - eps;
          const double yLo = grid->getYstart() + eps;
          const double yHi = grid->getYend() - eps;
          const double zLo = grid->getZstart() + eps;
          const double zHi = grid->getZend() - eps;

          double positionX =
              cellCenterX + (uniformDist(rng) - 0.5) * cellSpacingX;
          double positionY =
              cellCenterY + (uniformDist(rng) - 0.5) * cellSpacingY;
          double positionZ =
              cellCenterZ + (uniformDist(rng) - 0.5) * cellSpacingZ;

          // Safety: clamp to physical domain (guards against FP edge cases)
          if (positionX < xLo)
            positionX = xLo;
          if (positionX > xHi)
            positionX = xHi;
          if (positionY < yLo)
            positionY = yLo;
          if (positionY > yHi)
            positionY = yHi;
          if (positionZ < zLo)
            positionZ = zLo;
          if (positionZ > zHi)
            positionZ = zHi;

          // Maxwellian velocity sampling (thread-safe Box-Muller)
          double velocityX, velocityY, velocityZ;
          sampleMaxwellianThreadSafe(
              velocityX, velocityY, velocityZ, thermalVelocityX,
              thermalVelocityY, thermalVelocityZ, driftVelocityX,
              driftVelocityY, driftVelocityZ, rng, uniformDist);

          buffer
              .emplace_back(velocityX, velocityY, velocityZ, chargePerParticle,
                            positionX, positionY, positionZ, PARTICLE_ID_INVALID /* particle ID assigned during GPU scatter */);

          chargeAccumulator += chargePerParticle;
        }
      }
    }
  }

  speciesInjectedCharge[neutralIndex] = chargeAccumulator;
  return buffer;
}
