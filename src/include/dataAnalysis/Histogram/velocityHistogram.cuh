#ifndef VELOCITY_HISTOGRAM_CUH
#define VELOCITY_HISTOGRAM_CUH

#include "BuildInfo.h"
#include "cudaTypeDef.cuh"
#include "dataAnalysisConfig.cuh"
#include "particleArraySoAView.cuh"

#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <type_traits>
#include <vector>

namespace velocityHistogram {
/** The shared-memory tile of histogramKernel3D (not used currently): bins
 *  per axis, and in all. */
inline constexpr int kTile = DAConfig::VELOCITY_HISTOGRAM3D_TILE;
inline constexpr int kTileBins = kTile * kTile * kTile;

/** Bins of the histogram, and along axis d (a function: device code cannot
 *  index a constexpr array). */
inline constexpr int kNumBins = DAConfig::VELOCITY_HISTOGRAM3D_SIZE;
__host__ __device__ constexpr int binsAlong(int d) {
  return d == 0   ? DAConfig::VELOCITY_HISTOGRAM3D_RES_1
         : d == 1 ? DAConfig::VELOCITY_HISTOGRAM3D_RES_2
                  : DAConfig::VELOCITY_HISTOGRAM3D_RES_3;
}

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
/** Key of a particle outside the range (or with a non-finite velocity). */
inline constexpr unsigned kOutsideKey = 0xFFFFFFFFu;
/** A weight is held as the integer |q| * 2^62 / (n M), n the finite charges
 *  and M the largest |q|: exact sums, no bin above about 2^62 + n / 2 <
 *  2^63, and n M does not depend on the order of the particles either. The
 *  weights are rounded to n M / 2^62: a charge far below the largest (one
 *  wrong large finite charge) loses precision, below n M / 2^63 it weighs
 *  0. */
inline constexpr double kFixedPointOne = 4611686018427387904.0; // 2^62

/** Compaction of the non-empty bins: threads per block, rounds of that many
 *  consecutive bins per block, the bins of a block, and the blocks. */
inline constexpr int kCompactThreads = 256;
inline constexpr int kCompactRounds = 16;
inline constexpr int kCompactBlockBins = kCompactThreads * kCompactRounds;
inline constexpr int kCompactBlocks =
    (kNumBins + kCompactBlockBins - 1) / kCompactBlockBins;

/** Range of the velocities (without VelocityHistogramFixedRange): threads
 *  per block, particles per thread, and blocks at most. */
inline constexpr int kRangeThreads = 256;
inline constexpr int kRangeParticlesPerThread = 16;
inline constexpr int kRangeMaxBlocks = 1024;

static_assert(DAConfig::VELOCITY_HISTOGRAM3D_RES_1 % kTile == 0 &&
                  DAConfig::VELOCITY_HISTOGRAM3D_RES_2 % kTile == 0 &&
                  DAConfig::VELOCITY_HISTOGRAM3D_RES_3 % kTile == 0,
              "the tile of histogramKernel3D must divide the histogram");
static_assert(DAConfig::VELOCITY_HISTOGRAM3D_RES_1 % kAccumTileX == 0 &&
                  DAConfig::VELOCITY_HISTOGRAM3D_RES_2 % kAccumTileY == 0 &&
                  DAConfig::VELOCITY_HISTOGRAM3D_RES_3 % kAccumTileZ == 0,
              "the accumulation tile must divide the histogram");
static_assert(kAccumTiles <= 1024, "one scan thread per tile");
static_assert(kAccumTileBins <= 65536, "a bin of a tile fits 16 bits");
static_assert(kHistogramBlocks <= 1024, "a scan thread per block");
static_assert((kHistogramThreads & (kHistogramThreads - 1)) == 0 &&
                  kHistogramThreads >= 2 * 64,
              "two whole warps at least on NVIDIA and AMD (blockReduce)");
static_assert(kCompactThreads % 64 == 0, "whole warps on NVIDIA and AMD");
static_assert(kCompactBlocks <= 1024, "a scan thread per block of bins");
} // namespace velocityHistogram

namespace histogram {

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

  __host__ __device__ int getLogicSize() { return logicSize; }

  /** Not used currently. */
  __host__ void getSize(int* size) {
    for (int i = 0; i < dim; i++) {
      size[i] = this->size[i];
    }
  }

  __host__ __device__ int getSize(int index) { return size[index]; }

  /** Not used currently (nor getMax, getResolution). */
  __host__ U getMin(int index) { return min[index]; }

  __host__ U getMax(int index) { return max[index]; }

  __host__ U getResolution(int index) { return resolution[index]; }

  /** Linear bin of a point, -1 outside the range (used by the 2D unit test
   *  only). */
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

  /**
   * @brief get the index of the bin, in the buffer (used by histogramKernel3D,
   * not used currently)
   * @param data the data to be histogramed, dim elements
   * @param tile the start index of the tile, dim elements
   * @param tileSize the size of the tile, dim elements
   *
   * @return the index of the bin, in Tile
   */
  __device__ int getIndexTiled(const U* data, const int* tile,
                               const int* tileSize) {
    int index = 0;

    for (int i = dim - 1; i >= 0; i--) {
      // check the range
      if (data[i] < min[i] || data[i] > max[i]) {
        return -1;
      }

      auto tmp =
          (int)((data[i] - min[i]) /
                resolution[i]); // the index in the whole histogram dimension
      if (tmp == size[i])
        tmp--; // the max value

      if (tmp < tile[i] || tmp >= tile[i] + tileSize[i])
        return -1; // out of the tile range

      index += tmp - tile[i];
      if (i != 0)
        index *= tileSize[i - 1];
    }

    auto tileBufferSize = 1;
    for (int i = 0; i < dim; i++) {
      tileBufferSize *= tileSize[i];
    }

    if (index >= tileBufferSize)
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
/** An output value as an unsigned integer, for the byte swap of the VTK
 *  file. */
using histogramWordOut = std::conditional_t<sizeof(histogramTypeOut) == 8,
                                            std::uint64_t, std::uint32_t>;
static_assert(sizeof(histogramWordOut) == sizeof(histogramTypeOut),
              "4- or 8-byte output values");

using velocityHistogramCUDA2D =
    histogram::histogramCUDA<histogramTypeIn, 2, histogramTypeOut>;
using velocityHistogramCUDA3D =
    histogram::histogramCUDA<histogramTypeIn, 3, histogramTypeOut>;

using velocitySoA = particleArraySoAView<histogramTypeIn, 4>;

/** Bin geometry of an init(), passed by value to the kernels: the range of
 *  each axis, resolution (max - min) / binsAlong and its inverse. */
struct BinGrid {
  histogramTypeIn min[3], max[3], resolution[3], invResolution[3];

  /** Centre of the bin of linear index `bin` along axis d. */
  __host__ __device__ histogramTypeIn binCenter(int bin, int d) const {
    for (int a = 0; a < d; a++)
      bin /= binsAlong(a);
    return min[d] + (bin % binsAlong(d) + 0.5) * resolution[d];
  }
};

/** Particles [begin, end) of the tile-ordered arrays, all in one tile, that
 *  one block accumulates. */
struct HistogramChunk {
  int begin, end;
};

/**
 * @brief 3D (u, v, w) velocity histogram of one species at a time, weighted by
 * |q|, over a fixed range or over the range of the velocities.
 */
class velocityHistogram3D {
private:
  // the bins of the last init(), u fastest
  histogramTypeOut* binsCUDA = nullptr;
  histogramTypeOut* binsHost = nullptr; // pinned
  BinGrid grid;                         // of the last init()

  // bin the given range; otherwise the range of the velocities
  bool fixedRange;
  histogramTypeIn* reductionTempArrayCUDA = nullptr; // 6 x kRangeMaxBlocks
  histogramTypeIn* reductionResultCUDA = nullptr;    // min[3], then max[3]
  histogramTypeIn velocityRange[6] = {};             // min[3], then max[3]

  // the non-empty bins of the last init(), in increasing bin index
  histogramTypeIn* nonEmptyCenter[3] = {};
  histogramTypeOut* nonEmptyWeight = nullptr;
  int nonEmptyCapacity = 0;
  int* blockCountCUDA = nullptr;    // per block of bins, its offset; + total
  int* nonEmptyCountHost = nullptr; // pinned

  // accumulation: the integer histogram and the buffers of fixed size
  unsigned long long* binSumCUDA = nullptr; // per bin, fixed point
  double* weightPartialCUDA = nullptr; // per block: largest |q|, then count
  double* weightScaleCUDA = nullptr;   // to fixed point, and to the output
  int* tileCountCUDA = nullptr;        // particles per (block, tile) of a batch
  int* tileBlockOffsetCUDA = nullptr;  // their first slot within the tile
  int* tileTotalCUDA = nullptr;        // particles of each tile
  int* tileStartCUDA = nullptr;        // first slot of each tile
  int* tileChunkStartCUDA = nullptr;   // first chunk of each tile
  int* tileChunkCountCUDA = nullptr;   // chunks of each tile
  int* numChunksCUDA = nullptr;
  // scratch for a batch of scratchParticles, grown on demand
  int scratchParticles = 0;
  unsigned* keyCUDA = nullptr;                   // per particle: tile and bin
  unsigned short* tileBinCUDA = nullptr;         // per slot: bin in its tile
  unsigned long long* fixedWeightCUDA = nullptr; // per slot: fixed point
  HistogramChunk* chunksCUDA = nullptr;
  unsigned long long* partialTileCUDA = nullptr; // per chunk: its tile

  bool bigEndian;
  std::vector<histogramWordOut> vtkBuffer; // the bins of writeToFile

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

  /**
   * @brief The range of the finite velocities into velocityRange
   * (synchronous).
   * @param pclArray Borrowed velocity SoA view for one species.
   * @param stream CUDA stream used by the reduction kernels.
   */
  void dataRange(velocitySoA* pclArray, cudaStream_t stream);

public:
  /**
   * @brief Construct a velocity histogram accumulator.
   * @param fixedRange bin the range given to init(); otherwise the range of
   * the velocities
   */
  explicit velocityHistogram3D(bool fixedRange) : fixedRange(fixedRange) {
    cudaErrChk(cudaMalloc(&binsCUDA, sizeof(histogramTypeOut) * kNumBins));
    cudaErrChk(cudaMallocHost(&binsHost, sizeof(histogramTypeOut) * kNumBins));
    cudaErrChk(cudaMalloc(&blockCountCUDA, sizeof(int) * (kCompactBlocks + 1)));
    cudaErrChk(cudaMallocHost(&nonEmptyCountHost, sizeof(int)));
    cudaErrChk(cudaMalloc(&binSumCUDA, sizeof(unsigned long long) * kNumBins));
    cudaErrChk(
        cudaMalloc(&weightPartialCUDA, sizeof(double) * 2 * kHistogramBlocks));
    cudaErrChk(cudaMalloc(&weightScaleCUDA, sizeof(double) * 2));
    cudaErrChk(cudaMalloc(&tileCountCUDA,
                          sizeof(int) * kHistogramBlocks * kAccumTiles));
    cudaErrChk(cudaMalloc(&tileBlockOffsetCUDA,
                          sizeof(int) * kHistogramBlocks * kAccumTiles));
    cudaErrChk(cudaMalloc(&tileTotalCUDA, sizeof(int) * kAccumTiles));
    cudaErrChk(cudaMalloc(&tileStartCUDA, sizeof(int) * kAccumTiles));
    cudaErrChk(cudaMalloc(&tileChunkStartCUDA, sizeof(int) * kAccumTiles));
    cudaErrChk(cudaMalloc(&tileChunkCountCUDA, sizeof(int) * kAccumTiles));
    cudaErrChk(cudaMalloc(&numChunksCUDA, sizeof(int)));

    if (!fixedRange) {
      cudaErrChk(cudaMalloc(&reductionTempArrayCUDA,
                            sizeof(histogramTypeIn) * kRangeMaxBlocks * 6));
      cudaErrChk(cudaMalloc(&reductionResultCUDA, sizeof(histogramTypeIn) * 6));
    }

    { // check the endian
      int test = 1;
      char* ptr = reinterpret_cast<char*>(&test);
      if (*ptr == 1) {
        bigEndian = false;
      } else {
        bigEndian = true;
      }
    }
  }

  velocityHistogram3D(const velocityHistogram3D&) = delete;
  velocityHistogram3D& operator=(const velocityHistogram3D&) = delete;

  /**
   * @brief Launch the histogram kernels for one species (synchronous without
   * fixedRange, for the range of the velocities).
   * @param pclArray Borrowed velocity SoA view for one species.
   * @param vmin, vmax The range of the three axes with fixedRange. Without,
   * the range of the finite velocities of each axis; on an axis where they
   * have no spread, the width vmax - vmin about their value; vmin, vmax on
   * an axis where none is finite or their width overflows.
   * @param stream CUDA stream used by the histogram kernels.
   */
  void init(velocitySoA* pclArray, histogramTypeIn vmin, histogramTypeIn vmax,
            cudaStream_t stream);

  /**
   * @brief Wait for the histogram of the last init() and copy it to the host
   * (getVelocityHistogramHostPtr); writeToFile does it too.
   * @param stream CUDA stream of the init().
   */
  void copyHistogramToHost(cudaStream_t stream) {
    cudaErrChk(cudaMemcpyAsync(binsHost, binsCUDA,
                               sizeof(histogramTypeOut) * kNumBins,
                               cudaMemcpyDeviceToHost, stream));
    cudaErrChk(cudaStreamSynchronize(stream));
  }

  /**
   * @brief Write the histogram of the last init() to <filePath>3D_<cycle>.vtk:
   * legacy binary VTK (big-endian), a point per bin at its centre, u fastest;
   * a bin holds the sum of |q| VELOCITY_WEIGHT_SCALE of its particles.
   */
  void writeToFile(std::string filePath, int cycleNum, cudaStream_t stream) {
    copyHistogramToHost(stream);

    std::string vtkType;
    if constexpr (std::is_same_v<histogramTypeOut, float>) {
      vtkType = "float";
    } else if constexpr (std::is_same_v<histogramTypeOut, double>) {
      vtkType = "double";
    } else if constexpr (std::is_same_v<histogramTypeOut, int>) {
      vtkType = "int";
    } else {
      throw std::runtime_error("Unsupported histogramTypeOut");
    }

    std::ostringstream ossFileName;
    ossFileName << filePath << "3D_" << cycleNum << ".vtk";

    std::ofstream vtkFile(ossFileName.str(), std::ios::binary);

    vtkFile << "# vtk DataFile Version 3.0\n";
    vtkFile << "Velocity Histogram\n";
    vtkFile << "BINARY\n";
    vtkFile << "DATASET STRUCTURED_POINTS\n";
    vtkFile << std::setprecision(
        std::numeric_limits<histogramTypeIn>::max_digits10);
    vtkFile << "DIMENSIONS " << binsAlong(0) << " " << binsAlong(1) << " "
            << binsAlong(2) << "\n";
    // a point per bin, at its centre
    vtkFile << "ORIGIN " << grid.min[0] + 0.5 * grid.resolution[0] << " "
            << grid.min[1] + 0.5 * grid.resolution[1] << " "
            << grid.min[2] + 0.5 * grid.resolution[2] << "\n";
    vtkFile << "SPACING " << grid.resolution[0] << " " << grid.resolution[1]
            << " " << grid.resolution[2] << "\n";
    vtkFile << iPic3D::build::vtkFieldData(); // build identity (BuildInfo.h)
    vtkFile << "POINT_DATA " << kNumBins << "\n";
    vtkFile << "SCALARS scalars " << vtkType << " 1\n";
    vtkFile << "LOOKUP_TABLE default\n";

    // the bins big-endian, swapped in a buffer and written at once: a write
    // per bin costs more in the stream than in the disk
    vtkBuffer.resize(kNumBins);
    std::memcpy(vtkBuffer.data(), binsHost,
                kNumBins * sizeof(histogramWordOut));
    if (!bigEndian)
      for (histogramWordOut& word : vtkBuffer) {
        if constexpr (sizeof(histogramWordOut) == 8)
          word = __builtin_bswap64(word);
        else
          word = __builtin_bswap32(word);
      }
    vtkFile.write(reinterpret_cast<const char*>(vtkBuffer.data()),
                  kNumBins * sizeof(histogramWordOut));

    vtkFile.close();
    if (!vtkFile)
      throw std::runtime_error("velocityHistogram: cannot write " +
                               ossFileName.str());
  }

  /**
   * @brief Gather the non-empty bins of the last init(), in increasing bin
   * index: their centres into getNonEmptyCenters() and their contents into
   * getNonEmptyWeights(). Synchronises the stream.
   * @return the number of non-empty bins
   */
  int compactNonEmpty(cudaStream_t stream);
  histogramTypeIn** getNonEmptyCenters() { return nonEmptyCenter; }
  histogramTypeOut* getNonEmptyWeights() { return nonEmptyWeight; }

  /** The bins of the last copyHistogramToHost() or writeToFile(). */
  histogramTypeOut* getVelocityHistogramHostPtr() { return binsHost; }

  /** The bin geometry of the last init(). */
  const BinGrid& getGrid() const { return grid; }

  /** Not used currently. */
  histogramTypeOut* getVelocityHistogramCUDAArray() { return binsCUDA; }

  ~velocityHistogram3D() {
    for (void* buffer :
         {(void*)reductionTempArrayCUDA, (void*)reductionResultCUDA,
          (void*)binSumCUDA, (void*)weightPartialCUDA, (void*)weightScaleCUDA,
          (void*)tileCountCUDA, (void*)tileBlockOffsetCUDA,
          (void*)tileTotalCUDA, (void*)tileStartCUDA, (void*)tileChunkStartCUDA,
          (void*)tileChunkCountCUDA, (void*)numChunksCUDA,
          (void*)nonEmptyCenter[0], (void*)nonEmptyCenter[1],
          (void*)nonEmptyCenter[2], (void*)nonEmptyWeight,
          (void*)blockCountCUDA, (void*)binsCUDA})
      cudaErrChk(cudaFree(buffer)); // null for the unused ones
    freeScratch();
    cudaErrChk(cudaFreeHost(nonEmptyCountHost));
    cudaErrChk(cudaFreeHost(binsHost));
  }
};

} // namespace velocityHistogram

#endif // VELOCITY_HISTOGRAM_CUH
