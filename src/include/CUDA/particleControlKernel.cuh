#ifndef PARTICLE_CONTROL_KERNEL_CUH
#define PARTICLE_CONTROL_KERNEL_CUH

#include "cudaTypeDef.cuh"

#include "ParticleIDGenerator.cuh"
#include "ParticleSoADevice.cuh"
#include "particleExchange.cuh"

/**
 * @file particleControlKernel.cuh
 * @brief Control of the number of particles of a species: merging two
 *        particles into one and splitting one particle into two, after
 *        Lapenta and Brackbill, J. Comput. Phys. 115, 213 (1994).
 *
 * Both operations act inside one grid cell, where the trilinear weight of a
 * node is linear along each axis.
 *
 *  - A split replaces a particle by two of half its charge and the same
 *    velocity, displaced by +delta and -delta along one grid axis. The ten
 *    node moments of the pair are those of the parent.
 *  - A merge replaces two particles by one with their total charge at their
 *    centre of charge, moving with their mean velocity. It keeps the charge,
 *    the momentum and the centre of charge. It loses the kinetic energy of
 *    the relative motion, m1 m2 |v1 - v2|^2 / (2 (m1 + m2)), and it changes
 *    the node moments unless the pair is aligned with a grid axis and has
 *    equal velocities. A pair is therefore merged only when its velocities
 *    are close (MERGE_VELOCITY_TOLERANCE).
 */

/** A pair is merged only when |v1 - v2|^2 is at most this fraction of the
 *  velocity variance of its cell (the trace, about the mean velocity of the
 *  cell). A merge then loses at most a quarter of this fraction of the mean
 *  thermal energy of a pair of equal particles. */
inline constexpr cudaParticleType MERGE_VELOCITY_TOLERANCE = 0.009;

/**
 * @brief Where the two particles of a split are placed.
 *
 * A particle is split along one of the listed axes. An axis with a single
 * cell in the domain is not listed: two particles apart along it have the
 * same orbit.
 */
struct SplitGeometry {
  cudaParticleType start[3]; ///< position of a cell face, per axis
  cudaParticleType d[3];     ///< cell size
  cudaParticleType invd[3];  ///< 1 / cell size
  /** Half the distance between the two particles; less for a parent close
   *  to a cell face. */
  cudaParticleType halfSpacing[3];
  int axis[3];   ///< the axes to split along
  int axisCount; ///< number of entries of axis; the kernel needs 1 at least
};

/**
 * @brief Merge pairs of particles with close velocities, in the cells that
 *        hold more than targetPerCell particles.
 *
 * One warp works on one cell: launch numCells * WARP_SIZE threads, in blocks
 * of a multiple of WARP_SIZE. The particles are sorted by cell. Each particle
 * of a cell in turn takes the nearest later particle in velocity (the first
 * of equals); if they are close enough the later one is merged into it. A
 * particle takes part in one merge at most, and the cell stops at
 * targetPerCell particles.
 *
 * The particle merged away is only flagged DELETE in the departure array (all
 * flags must be STAY on entry): the mover skips it and the compaction removes
 * it.
 *
 * @param cellStart Index of the first particle of each cell.
 * @param cellCount Number of particles of each cell.
 * @param numCells Number of cells.
 * @param targetPerCell Number of particles a cell is merged down to.
 * @param pcls The particles, sorted by cell.
 * @param departure Departure flag of each particle.
 */
__global__ void particleMergingKernel(const int* cellStart,
                                      const int* cellCount, int numCells,
                                      int targetPerCell, ParticleSoADevice pcls,
                                      departureArrayElementType* departure);

/**
 * @brief Split count particles of a species in two.
 *
 * Launch count threads. The parents are spread evenly over the pcls.nop
 * particles; shift rotates the choice, so that successive calls take
 * different parents. Twin t is written at index pcls.nop + t: the caller
 * provides count <= pcls.nop and pcls.nop + count <= pcls.capacity, and sets
 * the new particle count.
 *
 * @param pcls The particles.
 * @param count Number of particles to split.
 * @param shift Rotation of the choice of the parents.
 * @param geometry Cells and distance between the two particles.
 * @param idGenerator Source of the ID of the new particles.
 */
__global__ void particleSplittingKernel(ParticleSoADevice pcls, uint32_t count,
                                        uint32_t shift, SplitGeometry geometry,
                                        ParticleIDGenerator idGenerator);

#endif // PARTICLE_CONTROL_KERNEL_CUH
