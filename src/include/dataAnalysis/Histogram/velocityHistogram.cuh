#ifndef VELOCITY_HISTOGRAM_CUH
#define VELOCITY_HISTOGRAM_CUH

#include "cudaTypeDef.cuh"
#include "dataAnalysisConfig.cuh"
#include "macrocellPartition.cuh"
#include "outputPrepare.h"
#include "particleArraySoAView.cuh"
#include "sparseBins.cuh"

#include <cstddef>
#include <cstdint>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

namespace velocityHistogram {
/** Bins of the histogram of a rank. */
inline constexpr int kNumBins = DAConfig::VELOCITY_HISTOGRAM3D_SIZE;

// ======= Accumulation (init) =======
/** Tile of bins summed by one block in shared memory, as 64-bit integers in
 *  two 32-bit words: 20 x 20 x 10 bins, 32 kB. */
inline constexpr int kAccumTileX = 20, kAccumTileY = 20, kAccumTileZ = 10;
inline constexpr int kAccumTileBins = kAccumTileX * kAccumTileY * kAccumTileZ;
inline constexpr int kAccumTilesX =
    DAConfig::VELOCITY_HISTOGRAM3D_RES_1 / kAccumTileX;
inline constexpr int kAccumTilesY =
    DAConfig::VELOCITY_HISTOGRAM3D_RES_2 / kAccumTileY;
inline constexpr int kAccumTilesZ =
    DAConfig::VELOCITY_HISTOGRAM3D_RES_3 / kAccumTileZ;
inline constexpr int kAccumTiles = kAccumTilesX * kAccumTilesY * kAccumTilesZ;
/** Particles of a batch (bounds the scratch memory), and of a chunk that one
 *  block accumulates. */
inline constexpr int kBatchParticles = 1 << 22;
inline constexpr int kChunkParticles = 1 << 15;
/** Blocks (also the threads of the scan per tile) and threads of the passes
 *  over the particles; the threads a power of 2 for the block reduction. */
inline constexpr int kHistogramBlocks = 1024;
inline constexpr int kHistogramThreads = 256;
/** Threads per block of the merge of the partial ranges of a region into its
 *  grid (regionGridKernel); a power of 2 for the block reduction. */
inline constexpr int kGridThreads = 256;
/** The range of the velocities (binGridOf) bins an axis when its bins are
 *  more than this share of the largest |velocity| wide, 4 to 8 units in the
 *  last place (their centres tell the velocities apart): otherwise the
 *  velocities have no spread. */
inline constexpr cudaParticleType kMinRelativeBinWidth =
    4 * std::numeric_limits<cudaParticleType>::epsilon();
/** Key of a particle outside the range (or with a non-finite velocity). */
inline constexpr unsigned kOutsideKey = 0xFFFFFFFFu;
/** A weight is held as the integer |q| * 2^62 / (n M), n the finite charges
 *  and M the largest |q|: exact sums, no bin above about 2^62 + n / 2 <
 *  2^63, and n M does not depend on the order of the particles either. The
 *  weights are rounded to n M / 2^62: a charge far below the largest (one
 *  wrong large finite charge) loses precision, below n M / 2^63 it weighs
 *  0. */
inline constexpr double kFixedPointOne = 4611686018427387904.0; // 2^62

// ======= Macrocell histograms (macrocellHistogram3D) =======
/** Bins of a macrocell histogram along u, v, w, and in all. */
inline constexpr int kMacrocellBinsU = DAConfig::MACROCELL_HISTOGRAM3D_RES_1;
inline constexpr int kMacrocellBinsV = DAConfig::MACROCELL_HISTOGRAM3D_RES_2;
inline constexpr int kMacrocellBinsW = DAConfig::MACROCELL_HISTOGRAM3D_RES_3;
inline constexpr int kMacrocellBins =
    kMacrocellBinsU * kMacrocellBinsV * kMacrocellBinsW;
/** Tile of a macrocell histogram that one block sums in shared memory, two
 *  32-bit words per bin: 16^3 bins, 32 kB; the tiles of a macrocell. */
inline constexpr int kMacrocellTileU = 16, kMacrocellTileV = 16,
                     kMacrocellTileW = 16;
inline constexpr int kMacrocellTileBins =
    kMacrocellTileU * kMacrocellTileV * kMacrocellTileW;
inline constexpr int kMacrocellTiles = kMacrocellBins / kMacrocellTileBins;
/** Threads per block of the macrocell kernels: a power of 2 for the block
 *  reductions. */
inline constexpr int kMacrocellThreads = 256;
/** Threads per block of the centres of the non-empty bins (GMM points). */
inline constexpr int kCenterThreads = 256;
/** The first word of a record of the velocity histograms, "HST2" in the
 *  little-endian file (sparseBins::SparseRecordHeader). */
inline constexpr std::uint32_t kHistogramRecordMagic = 0x32545348;

static_assert(DAConfig::VELOCITY_HISTOGRAM3D_RES_1 % kAccumTileX == 0 &&
                  DAConfig::VELOCITY_HISTOGRAM3D_RES_2 % kAccumTileY == 0 &&
                  DAConfig::VELOCITY_HISTOGRAM3D_RES_3 % kAccumTileZ == 0,
              "the accumulation tile must divide the histogram");
static_assert(kAccumTiles <= 1024, "one scan thread per tile");
static_assert(kAccumTileBins <= 65536, "a bin of a tile fits 16 bits");
static_assert(kHistogramBlocks <= 1024, "a scan thread per block");
static_assert(kHistogramThreads >= 2 * 64 && kMacrocellThreads >= 2 * 64 &&
                  kGridThreads >= 2 * 64,
              "two 64-wide wavefronts at least (blockReduce on AMD)");
static_assert(2 * sizeof(unsigned) * kAccumTileBins <= 48 * 1024,
              "the tile of the accumulation fits the default shared memory");
static_assert(kMacrocellBinsU % kMacrocellTileU == 0 &&
                  kMacrocellBinsV % kMacrocellTileV == 0 &&
                  kMacrocellBinsW % kMacrocellTileW == 0,
              "the tile of the macrocell kernel must divide its histogram");
static_assert(2 * sizeof(unsigned) * kMacrocellTileBins <= 48 * 1024,
              "the tile of the macrocell kernel fits the default shared "
              "memory");
} // namespace velocityHistogram

namespace histogram {

/** A histogram of dim dimensions with its bins on the host and on the
 *  device, filled by a kernel through getIndex. The analysis does not use it
 *  (histogramRegions holds its histograms): the 2D unit test does. */
template <typename U, int dim, typename T = int> class histogramCUDA {

private:
  T* hostPtr;
  T* cudaPtr;

  int bufferSize; // the physical size of current buffer, in elements
public:
  int size[dim]; // the logic size of each dimension, in elements
private:
  int logicSize; // the logic size of the whole histogram

  U min[dim], max[dim], resolution[dim];

public:
  /**
   * @param bufferSize the physical size of the buffer, in elements
   */
  __host__ histogramCUDA(int bufferSize) : bufferSize(bufferSize) {
    allocate();
  }

  /**
   * @param min the minimum value of each dimension
   * @param max the maximum value of each dimension
   * @param binThisDim the bins of each dimension
   */
  __host__ void setHistogram(const U* min, const U* max,
                             const int* binThisDim) {

    for (int i = 0; i < dim; i++) {
      if (min[i] >= max[i] || binThisDim[i] <= 0) {
        std::cerr << "[!]Invalid histogram range or binThisDim" << std::endl;
        std::cerr << "[!]min: " << min[i] << " max: " << max[i]
                  << " binThisDim: " << binThisDim[i] << std::endl;
        return;
      }
    }

    logicSize = 1;
    for (int i = 0; i < dim; i++) {
      this->min[i] = min[i];
      this->max[i] = max[i];

      size[i] = binThisDim[i];
      this->resolution[i] = (max[i] - min[i]) / size[i];
      logicSize *= size[i];
    }

    if (bufferSize < logicSize) {
      cudaErrChk(cudaFreeHost(hostPtr));
      cudaErrChk(cudaFree(cudaPtr));
      bufferSize = logicSize;
      allocate();
    }
  }

  __host__ void copyHistogramAsync(cudaStream_t stream) {
    cudaErrChk(cudaMemcpyAsync(hostPtr, cudaPtr, logicSize * sizeof(T),
                               cudaMemcpyDeviceToHost, stream));
  }

  __host__ T* getHistogram() { return hostPtr; }

  __host__ __device__ T* getHistogramCUDA() { return cudaPtr; }

  /** Linear bin of a point, -1 outside the range. */
  __device__ int getIndex(const U* data) {
    int index = 0;

    for (int i = dim - 1; i >= 0; i--) {
      // check the range
      if (data[i] < min[i] || data[i] > max[i]) {
        return -1;
      }

      auto tmp = (int)((data[i] - min[i]) / resolution[i]);
      if (tmp == size[i])
        tmp--; // the max value
      index += tmp;
      if (i != 0)
        index *= size[i - 1];
    }

    if (index >= logicSize)
      return -1;
    return index;
  }

private:
  __host__ void allocate() {
    cudaErrChk(cudaMallocHost((void**)&hostPtr, bufferSize * sizeof(T)));
    cudaErrChk(cudaMalloc((void**)&cudaPtr, bufferSize * sizeof(T)));
  }

public:
  __host__ ~histogramCUDA() {
    cudaErrChk(cudaFreeHost(hostPtr));
    cudaErrChk(cudaFree(cudaPtr));
  }
};

} // namespace histogram

namespace velocityHistogram {

using histogramTypeIn = cudaParticleType;
using histogramTypeOut = cudaTypeSingle;

using velocityHistogramCUDA2D =
    histogram::histogramCUDA<histogramTypeIn, 2, histogramTypeOut>;

using velocitySoA = particleArraySoAView<histogramTypeIn, 4>;

/** Bin geometry of a region in an init(), passed by value to the kernels:
 *  the bins along each axis, the range, the resolution (max - min) / bins and
 *  its inverse. */
struct BinGrid {
  int bins[3];
  histogramTypeIn min[3], max[3], resolution[3], invResolution[3];

  /** Centre of the bin of linear index `bin` (u fastest) along axis d. */
  __host__ __device__ histogramTypeIn binCenter(int bin, int d) const {
    for (int a = 0; a < d; a++)
      bin /= bins[a];
    return min[d] + (bin % bins[d] + 0.5) * resolution[d];
  }
};

/** What the grid and the weight scale of a region need of some of its
 *  particles: the range of their finite velocities (low > high without one;
 *  only without a fixed range), the largest finite |q| as the bits of the
 *  double (they order as |q|) and the number of finite charges. */
struct ParticleRange {
  histogramTypeIn low[3], high[3];
  unsigned long long largest;
  unsigned count;
};

/**
 * @brief The grid of the given bins: [vmin, vmax] on every axis with
 * fixedRange. Without, [low, high], the range of the finite velocities, on
 * each axis; on an axis where they have no spread (low == high, or bins
 * narrower than kMinRelativeBinWidth of the largest |velocity|), the
 * width vmax - vmin with their middle at the centre of its middle bin (bin
 * bins / 2, rounded down): one bin, whatever their rounding; [vmin, vmax] on
 * an axis where none is finite (low > high) or their width overflows.
 */
__host__ __device__ inline BinGrid
binGridOf(const int bins[3], const histogramTypeIn low[3],
          const histogramTypeIn high[3], histogramTypeIn vmin,
          histogramTypeIn vmax, bool fixedRange) {
  // a range that bins: a positive, finite width of a finite inverse
  // resolution (no isfinite, a device function only for HIP; NaN fails the
  // comparisons)
  auto binsOver = [](int b, histogramTypeIn lo, histogramTypeIn hi) {
    return hi - lo > 0 && hi - lo < INFINITY && b / (hi - lo) < INFINITY;
  };
  // a spread: bins of kMinRelativeBinWidth of the largest |velocity| at
  // least (lo <= hi)
  auto spreads = [](int b, histogramTypeIn lo, histogramTypeIn hi) {
    return hi - lo > b * kMinRelativeBinWidth * (hi > -lo ? hi : -lo);
  };
  const histogramTypeIn width = vmax - vmin;
  BinGrid grid;
  for (int d = 0; d < 3; d++) {
    histogramTypeIn lo = vmin, hi = vmax;
    if (!fixedRange && low[d] <= high[d]) {
      // the fixed width, middle in the centre of bin bins / 2: half a bin
      // lower for an even number of bins (no product, which would be fused
      // into an FMA on the device and not on the host)
      const histogramTypeIn middle = low[d] + (high[d] - low[d]) / 2;
      const histogramTypeIn below =
          middle - width / 2 - (bins[d] % 2 ? 0 : width / (2 * bins[d]));
      if (spreads(bins[d], low[d], high[d]) &&
          binsOver(bins[d], low[d], high[d])) {
        lo = low[d];
        hi = high[d];
      } else if (!spreads(bins[d], low[d], high[d]) &&
                 binsOver(bins[d], below, below + width)) {
        lo = below;
        hi = below + width;
      }
    }
    grid.bins[d] = bins[d];
    grid.min[d] = lo;
    grid.max[d] = hi;
    grid.resolution[d] = (hi - lo) / bins[d];
    grid.invResolution[d] = bins[d] / (hi - lo);
  }
  return grid;
}

/** Particles [begin, end) of the tile-ordered arrays, all in one tile, that
 *  one block accumulates. */
struct HistogramChunk {
  int begin, end;
};

/**
 * @brief 3D (u, v, w) velocity histograms of the regions of a rank, one
 * species at a time, weighted by |q|: the rank (velocityHistogram3D) or its
 * macrocells (macrocellHistogram3D). Holds the bins of the regions, region
 * after region (u fastest in a region), the bin grid of each region, their
 * non-empty bins (the records, and with their centres the points of the GMM
 * fits) and writes the records.
 */
class histogramRegions {
public:
  /** @param regions The regions; @param bins their bins along each axis;
   *  @param fixedRange bin the range given to init(), otherwise the range of
   *  the velocities of each region; @param regionsText The regions of the
   *  rank (macrocell::regionsDescription). */
  histogramRegions(int regions, const int bins[3], bool fixedRange,
                   const std::string& regionsText);
  virtual ~histogramRegions();
  histogramRegions(const histogramRegions&) = delete;
  histogramRegions& operator=(const histogramRegions&) = delete;

  int numRegions() const { return regions; }

  /** The bin grid of a region in the last init(). */
  const BinGrid& getGrid(int region) const { return gridHost[region]; }

  /** The bins of the last init() on the device, region after region (u
   *  fastest in a region), ready when its stream is. */
  const histogramTypeOut* deviceBins() const { return binsCUDA; }

  /**
   * @brief Gather the non-empty bins of the last init() (SparseBins): those
   * of region r are [getRegionOffsets()[r], getRegionOffsets()[r + 1]), their
   * contents getNonEmptyWeights(). Synchronises the stream; once per init().
   * @return the number of non-empty bins
   */
  int compactNonEmpty(cudaStream_t stream);
  /** The centres of the non-empty bins, u, v, w (device), on the grids of
   *  their regions, computed at each call (the GMM scales them in place);
   *  compacts if needed. Synchronises the stream. */
  histogramTypeIn* const* getNonEmptyCenters(cudaStream_t stream);
  histogramTypeOut* getNonEmptyWeights() { return sparse.weights(); }
  /** The offsets of the regions in the non-empty bins of the last
   *  compactNonEmpty(): numRegions() + 1 values. */
  const std::vector<int>& getRegionOffsets() const { return sparse.offsets(); }

  /** The description of the records file of the histograms
   *  (SparseBins::describe) after rank, the JSON members of the rank, and
   *  into layout its parts that its records need. */
  std::string describe(const std::string& rank,
                       std::vector<std::string>& layout) const;

  /** The regions of the rank (macrocell::regionsDescription). */
  const std::string& regionsDescription() const { return regionsText; }

  /** Add to file one record of the non-empty bins of the last init(), none
   *  without one: sparseBins::SparseRecordHeader, the range of every region
   *  (float64 [umin, vmin, wmin, umax, vmax, wmax]), then the arrays of
   *  SparseBins::writeRecord. */
  void writeRecord(RecordFile& file, int species, int cycle,
                   cudaStream_t stream);

protected:
  const int regions;
  const int regionBins;
  const int bins[3];
  const bool fixedRange;
  double* weightScaleCUDA = nullptr; // per region: to fixed point, to output

  // the bins of the last init(), region after region; and the grids
  histogramTypeOut* binsCUDA = nullptr;
  BinGrid* gridCUDA = nullptr;
  PinnedVector<BinGrid> gridHost; // page-locked: copied from the device

  /** The bins of all the regions. */
  size_t totalBins() const { return (size_t)regions * regionBins; }

  /** Called by init(): the bins change. */
  void binsChanged() { compacted = false; }

private:
  const std::string regionsText;
  sparseBins::SparseBins sparse;
  bool compacted = false; // since the last init()
  // the centres of the non-empty bins (GMM points), grown on demand
  histogramTypeIn* nonEmptyCenter[3] = {};
  int centerCapacity = 0;
};

/**
 * @brief The 3D (u, v, w) velocity histogram of all the particles of a rank,
 * one species at a time, weighted by |q|, over a fixed range or over the
 * range of the velocities: one region of kNumBins bins.
 */
class velocityHistogram3D : public histogramRegions {
private:
  ParticleRange* rangePartialCUDA = nullptr; // per block of rankRangeKernel

  // accumulation: the integer histogram and the buffers of fixed size
  unsigned long long* binSumCUDA = nullptr; // per bin, fixed point
  int* tileCountCUDA = nullptr;       // particles per (block, tile) of a batch
  int* tileBlockOffsetCUDA = nullptr; // their first slot within the tile
  int* tileTotalCUDA = nullptr;       // particles of each tile
  int* tileStartCUDA = nullptr;       // first slot of each tile
  int* tileChunkStartCUDA = nullptr;  // first chunk of each tile
  int* tileChunkCountCUDA = nullptr;  // chunks of each tile
  int* numChunksCUDA = nullptr;
  // scratch for a batch of scratchParticles, grown on demand
  int scratchParticles = 0;
  unsigned* keyCUDA = nullptr;                   // per particle: tile and bin
  unsigned short* tileBinCUDA = nullptr;         // per slot: bin in its tile
  unsigned long long* fixedWeightCUDA = nullptr; // per slot: fixed point
  HistogramChunk* chunksCUDA = nullptr;
  unsigned long long* partialTileCUDA = nullptr; // per chunk: its tile

  /** Chunks of a batch at most: a tile of c particles makes at most
   *  c / kChunkParticles + 1 chunks. */
  static int maxChunks(int particles) {
    return getGridSize(particles, kChunkParticles) + kAccumTiles;
  }

  /** Grow the scratch to a batch of particles, with a margin (no-op if large
   *  enough). */
  void reserveScratch(int particles);
  void freeScratch();

  /** The histogram of the particles over the grid into the float histogram:
   *  fixed-point weights, tiles summed in shared memory, no global
   *  atomics. */
  void accumulate(velocitySoA* pclArray, cudaStream_t stream);

public:
  /**
   * @brief Construct a velocity histogram accumulator.
   * @param fixedRange bin the range given to init(); otherwise the range of
   * the velocities
   */
  explicit velocityHistogram3D(bool fixedRange);
  ~velocityHistogram3D() override;

  /**
   * @brief Launch the histogram kernels for one species (synchronous, for
   * the grid).
   * @param pclArray Borrowed velocity SoA view for one species.
   * @param vmin, vmax The range of the three axes (binGridOf).
   * @param stream CUDA stream used by the histogram kernels.
   */
  void init(velocitySoA* pclArray, histogramTypeIn vmin, histogramTypeIn vmax,
            cudaStream_t stream);
};

/**
 * @brief The 3D (u, v, w) velocity histograms of the macrocells of a rank,
 * one species at a time, weighted by |q|: one region of kMacrocellBins bins
 * per macrocell. The particles of a macrocell are those of its cells, after
 * the cell sort; a particle in a ghost cell (exactly on a face of the rank)
 * is in none.
 * The range of a macrocell is the fixed one, or that of the velocities of its
 * particles (binGridOf); its weights are summed exactly in the fixed point of
 * its own charges. When the macrocells are few, several blocks walk each,
 * in slices of its cells (macrocell::slicesFor), and their ranges and integer
 * sums are merged: the same bits as one block per macrocell.
 */
class macrocellHistogram3D : public histogramRegions {
public:
  /**
   * @param part The macrocells of the rank.
   * @param fixedRange bin the range given to init(); otherwise the range of
   * the velocities of each macrocell
   */
  macrocellHistogram3D(const macrocell::MacrocellPartition& part,
                       bool fixedRange);

  /**
   * @brief Launch the histogram kernels of the macrocells for one species,
   * whose particles are sorted by cell (synchronous, for the grids); a
   * species without particles has empty bins and the fixed range, whatever
   * the cells say.
   * @param pclArray Borrowed velocity SoA view for the species.
   * @param cellStart, cellCount The first particle and the particles of each
   * cell of the guarded grid (the cell sort of the species).
   * @param vmin, vmax The range of the three axes (binGridOf).
   * @param stream CUDA stream used by the histogram kernels.
   */
  void init(velocitySoA* pclArray, const int* cellStart, const int* cellCount,
            histogramTypeIn vmin, histogramTypeIn vmax, cudaStream_t stream);

  ~macrocellHistogram3D() override;

private:
  const macrocell::MacrocellGeometry geometry;
  // the slices of each macrocell of the range and of the accumulation
  const int rangeSlices, accumulateSlices;
  // with several slices: the range of each slice of each macrocell, and the
  // integer bins of the macrocells
  ParticleRange* rangePartialCUDA = nullptr;
  unsigned long long* binSumCUDA = nullptr;
};

} // namespace velocityHistogram

#endif // VELOCITY_HISTOGRAM_CUH
