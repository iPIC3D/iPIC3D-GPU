#pragma once

#include <vector>
#include <memory>
#include <random>
#include "Particle.h" // SpeciesParticle

class Collective;
class Grid3DCU;
class VCtopology3D;

/**
 * @brief Exosphere ionization source for Mercury-like planetary simulations.
 *
 * Implements the Chamberlain neutral density profile and photoionization
 * injection of macro-particles. Sampling is done entirely on the CPU.
 * The caller is responsible for transferring the resulting particles to the GPU.
 *
 * Quasi-neutrality is maintained by input configuration: paired ion and
 * electron species share identical neutral parameters (neutralSurfaceDensity,
 * exosphericScaleHeight, photoionizationFrequency, macroParticleWeightRatio),
 * ensuring the same integer number of particles is injected per cell for each.
 *
 * Thread safety:
 *   - Each planetary species has its own std::mt19937_64 RNG and particle buffer.
 *   - sampleIonizedParticles() is safe to call concurrently for DIFFERENT speciesIndex
 *     values (each accesses only its own RNG, buffer, and charge accumulator).
 *   - Concurrent calls with the SAME speciesIndex are NOT safe.
 *
 * Performance notes:
 *   - Per-species particle buffers are allocated once and reused across timesteps
 *     (grow-only policy: capacity only increases, never shrinks).
 *   - The injection rate constant is pre-computed once per call (loop-invariant).
 *   - The radial boundary check uses squared distances to avoid sqrt when possible.
 *   - Per-species std::mt19937_64 RNGs eliminate global rand() contention.
 */
class ExosphereIonization {
public:
    /**
     * @brief Construct with references to simulation configuration, grid, and topology.
     *        Allocates persistent per-species particle buffers and RNGs for all planetary species.
     *
     * @param collective Simulation input/configuration object.
     * @param grid Local grid descriptor for the current MPI rank.
     * @param topology MPI topology descriptor for the current rank.
     */
    ExosphereIonization(const Collective* collective, const Grid3DCU* grid, const VCtopology3D* topology);

    /**
     * @brief Sample new ionized exosphere particles for one species on the CPU.
     *
     * Thread-safe for concurrent calls with different speciesIndex values. Each species
     * has its own RNG, particle buffer, and charge accumulator — no shared mutable state
     * is accessed. Uses std::mt19937_64 for reproducible, thread-safe random sampling.
     *
     * Particles are written into a persistent internal buffer (no heap allocation
     * unless the buffer must grow). The returned reference is valid until the next
     * call to sampleIonizedParticles for the *same* speciesIndex.
     *
     * @param speciesIndex  Global species index (must be >= numSolarWindSpecies,
     *                      i.e. a planetary species).
     *                      The neutral-parameter index is computed as
     *                      (speciesIndex - numSolarWindSpecies).
     * @param maxParticles  Maximum number of particles to inject for this species
     *                      in this call. 0 = unlimited (default). When the limit
     *                      is reached, sampling stops early. This allows the caller
     *                      to enforce memory budgets based on GPU/CPU availability.
     * @return Read-only reference to the internal particle buffer for this species.
     *         The caller must consume or copy the data before the next call for
     *         the same species.
     */
    const std::vector<SpeciesParticle>& sampleIonizedParticles(int speciesIndex, int maxParticles = 0);

    /**
     * @brief Get the total injected charge across all planetary species.
     *        This is a local (per-MPI-rank) value; use MPI_Allreduce for global diagnostics.
     */
    double getTotalInjectedCharge() const;

    /**
     * @brief Get the injected charge for a specific planetary species from the last call.
     *
     * @param speciesIndex Global species index for the requested planetary species.
     */
    double getSpeciesInjectedCharge(int speciesIndex) const;

    /**
     * @brief Get the number of particles produced by the last call for one species.
     *
     * @param speciesIndex Global species index for the requested planetary species.
     */
    int getLastInjectedCount(int speciesIndex) const;

private:
    /**
     * @brief Chamberlain neutral density profile.
     *
     * Returns Nexo * exp(-(distance - planetRadius) / scaleHeight) for distance > planetRadius,
     * and 0 otherwise (inside the planet).
     *
     * @param surfaceDensity  Neutral density at the planet surface (in n_sw units).
     * @param distance        Radial distance from planet center.
     * @param planetRadius    Planet radius.
     * @param scaleHeight     Exospheric scale height (exosphericScaleHeight).
     * @return Neutral density at the given distance (in n_sw units).
     */
    static double neutralDensity(double surfaceDensity, double distance,
                                 double planetRadius, double scaleHeight);

    /**
     * @brief Thread-safe Box-Muller Maxwellian velocity sampling.
     *
     * Replaces the global-rand()-based sample_maxwellian() with a per-species
     * RNG reference. Generates 3 normal variates (two via paired Box-Muller,
     * one via an independent Box-Muller pair, discarding the second).
     *
     * @param u Output x-velocity sample.
     * @param v Output y-velocity sample.
     * @param w Output z-velocity sample.
     * @param ut Thermal speed in x.
     * @param vt Thermal speed in y.
     * @param wt Thermal speed in z.
     * @param u0 Drift speed in x.
     * @param v0 Drift speed in y.
     * @param w0 Drift speed in z.
     * @param rng Per-species pseudo-random generator.
     * @param dist Uniform random distribution bound to @p rng.
     */
    static void sampleMaxwellianThreadSafe(
        double& u, double& v, double& w,
        double ut, double vt, double wt,
        double u0, double v0, double w0,
        std::mt19937_64& rng,
        std::uniform_real_distribution<double>& dist);

    const Collective*    collective;
    const Grid3DCU*      grid;
    const VCtopology3D*  topology;

    // ── Per-species state (indexed by neutral species index) ──
    int numPlanetarySpecies;             ///< number of planetary species (cached from Collective)
    int firstPlanetarySpeciesIndex;      ///< global index of first planetary species (= numSolarWindSpecies)

    /// Particle buffers — one per planetary neutral species.
    /// Capacity grows as needed but never shrinks. Size is reset to 0 each call.
    std::unique_ptr<std::vector<SpeciesParticle>[]> speciesParticleBuffers;

    /// Per-species RNGs — deterministic, thread-safe (each task uses its own).
    /// Seeded in constructor with (MPI_rank * numPlanetarySpecies + neutralIndex).
    std::unique_ptr<std::mt19937_64[]> speciesRNGs;

    /// Per-species injected charge accumulators (no cross-species contention).
    std::unique_ptr<double[]> speciesInjectedCharge;

    // ── Pre-computed constants (set once in constructor, read-only thereafter) ──
    double fourPi;                       ///< 4π  (computed once)
    int    particlesPerCellSW;           ///< npcel[0] for the reference solar wind species
};
