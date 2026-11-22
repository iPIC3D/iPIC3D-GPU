#ifndef PARTICLE_OUTPUT_SELECT_CUH
#define PARTICLE_OUTPUT_SELECT_CUH

/**
 * @file particleOutputSelect.cuh
 * @brief The particles of a species that the particle output writes when
 * ParticlesOutputTrackedFraction or ParticlesOutputMaxPerCell selects them,
 * chosen on the GPU so that only these cross to the host.
 *
 * Every particle has a 64-bit key, u = key / 2^64 uniform in [0, 1) and
 * uncorrelated with the position and velocity of the particle: a hash of
 * its ID, which never changes (distinct IDs, distinct keys); for a species
 * without IDs a hash of its position and first velocity component, which
 * changes from cycle to cycle.
 *
 * A species selects its particles in one of two ways, never both:
 *
 * - The fraction f (species with IDs): the particles with u < f (key <=
 *   selectLimit(f)), about f of them, the same particles at every output,
 *   wherever they are; charge q / f. A compaction of the particles, no sort:
 *   launchFractionCount, launchFractionSelect, then launchGather and
 *   launchGatherScaled.
 * - The cap K: of the n particles of a cell, all when n <= K, else the K of
 *   smallest key, so every particle of the cell is written with the same
 *   probability, whatever the order of the particles (which the sort's
 *   atomics change from run to run); each written particle carries its
 *   charge times Q / Q_kept, so the cell's charge is exact (n / K for equal
 *   charges). On the species sorted by cell (CellSorter): launchCapCount,
 *   launchCapSelect (which writes the charges), then launchGather.
 *
 * Positions, velocities and IDs are not changed. A count phase gives the
 * number of particles written, which the host reads to size its copies.
 */

#include "cudaTypeDef.cuh"

#include <cmath>
#include <cstddef>
#include <cstdint>

/** Threads of a block of the scans (cells or particles per block). */
inline constexpr int kSelectScanBlock = 1024;

/** The arrays the keys of the particles are computed from: their IDs, else
 *  their coordinates (position and u). */
struct ParticleSelectKeys {
  const cudaPclType_ID* id; // null: the coordinates
  const cudaPclType_X* x;
  const cudaPclType_Y* y;
  const cudaPclType_Z* z;
  const cudaPclType_U* u;
  std::uint64_t salt; // per species, so species are chosen independently
};

/**
 * @brief The largest key of a fraction > 0, u = key / 2^64 < fraction: the
 * keys below ceil(fraction 2^64), at most 2^64 - 2048 for a fraction below 1
 * (the doubles from 2^52 on are integers); every key with fraction >= 1.
 */
inline std::uint64_t selectLimit(double fraction) {
  return fraction >= 1.0
             ? UINT64_MAX
             : (std::uint64_t)std::ceil(std::ldexp(fraction, 64)) - 1;
}

/** A device array of a selection that only grows, to a quarter more than
 *  asked; its content is lost when it does (cudaFree waits for the kernels
 *  that use it). */
template <typename T> struct GrowingDeviceArray {
  T* ptr = nullptr;
  std::size_t capacity = 0;

  void reserve(std::size_t n) {
    if (n <= capacity)
      return;
    cudaErrChk(cudaFree(ptr));
    capacity = n + n / 4;
    cudaErrChk(cudaMalloc(&ptr, capacity * sizeof(T)));
  }
  void release() {
    cudaErrChk(cudaFree(ptr));
    ptr = nullptr;
    capacity = 0;
  }
};

// ---- the fraction: a compaction of the nop particles, no sort

/**
 * @brief The selected particles (key <= limit) of each block of
 * kSelectScanBlock particles (blockSums: ceil(nop / kSelectScanBlock) device
 * ints), scanned into the first output slot of each block, and their total
 * into the device int *total.
 */
void launchFractionCount(const ParticleSelectKeys& keys, std::uint32_t nop,
                         std::uint64_t limit, int* blockSums, int* total,
                         cudaStream_t stream);

/** @brief Their indices, in the order of the particles. */
void launchFractionSelect(const ParticleSelectKeys& keys, std::uint32_t nop,
                          std::uint64_t limit, const int* blockSums,
                          std::uint32_t* idx, cudaStream_t stream);

// ---- the cap: the species sorted by cell

/**
 * @brief The particles written in each of the numCells cells, min(count[c],
 * maxPerCell), scanned into offset[c], the first output slot of cell c;
 * their total into the device int *total.
 * @param blockSums ceil(numCells / kSelectScanBlock) device ints of scratch.
 */
void launchCapCount(const int* count, int numCells, int maxPerCell, int* offset,
                    int* blockSums, int* total, cudaStream_t stream);

/**
 * @brief The indices of the written particles of every cell (the particles
 * of cell c are [start[c], start[c] + count[c])), cell after cell from
 * offset[c], and their charges into qOut (null: none), the same slots.
 */
void launchCapSelect(const int* count, const int* start, const int* offset,
                     int numCells, int maxPerCell,
                     const ParticleSelectKeys& keys, const cudaPclType_Q* q,
                     std::uint32_t* idx, cudaPclType_Q* qOut,
                     cudaStream_t stream);

// ---- the arrays of the n written particles

/** @brief dst[o] = src[idx[o]]. */
template <typename T>
void launchGather(T* dst, const T* src, const std::uint32_t* idx,
                  std::uint32_t n, cudaStream_t stream);

/** @brief The charges with the fraction, dst[o] = q[idx[o]] scale (1 / f). */
void launchGatherScaled(cudaPclType_Q* dst, const cudaPclType_Q* q,
                        const std::uint32_t* idx, std::uint32_t n, double scale,
                        cudaStream_t stream);

#endif // PARTICLE_OUTPUT_SELECT_CUH
