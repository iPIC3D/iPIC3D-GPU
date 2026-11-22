#ifndef SPARSE_BINS_CUH
#define SPARSE_BINS_CUH

#include "cudaTypeDef.cuh"
#include "outputPrepare.h"
#include "sparseIndex.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

/**
 * @file sparseBins.cuh
 * @brief The non-empty bins of the dense histograms of R regions on the GPU,
 * gathered region after region: the records of the histograms and of the
 * spectra hold only them, and the GMM fits them.
 */
namespace sparseBins {

/** Compaction: threads per block, rounds of that many consecutive bins per
 *  block, the bins of a block (of one region), and the threads of the scan
 *  of the blocks. */
inline constexpr int kCompactThreads = 256;
inline constexpr int kCompactRounds = 16;
inline constexpr int kCompactBlockBins = kCompactThreads * kCompactRounds;
inline constexpr int kScanThreads = 1024;

static_assert(kCompactThreads % 64 == 0, "whole warps on NVIDIA and AMD");

/** The value of a bin. */
using binType = cudaTypeSingle;

/** The header of a record of sparse histograms (RecordFile), followed by the
 *  arrays of SparseBins::writeRecord. */
struct SparseRecordHeader {
  std::uint32_t magic; // of the file
  std::int32_t cycle;
  std::int32_t species;
  std::uint32_t regions;
  std::uint32_t nonEmpty;   // the non-empty bins of all the regions
  std::uint32_t indexBytes; // of their index (encodeBinIndex)
  std::uint64_t bytes;      // of the record, this header and padding included
};

/**
 * @brief The non-empty bins of the dense bins of R regions of B bins each,
 * on the GPU: their contents and their index in the region, region after
 * region and in increasing index, with the offsets of the regions. The
 * buffers grow only.
 */
class SparseBins {
public:
  SparseBins(int regions, int regionBins);
  ~SparseBins();
  SparseBins(const SparseBins&) = delete;
  SparseBins& operator=(const SparseBins&) = delete;

  /**
   * @brief Gather the bins above 0 of the dense bins (R B values on the
   * device, region after region). Synchronises the stream.
   * @return the number of non-empty bins
   */
  int compact(const binType* bins, cudaStream_t stream);

  /** The non-empty bins of the last compact(): their number; the offsets of
   *  the regions, R + 1 values on the host; their contents and indices on
   *  the device; on the device the offsets of the blocks of bins of the
   *  compaction, R blocksPerRegion() + 1 values (region r from block
   *  r blocksPerRegion() on). */
  int count() const { return nonEmpty; }
  const std::vector<int>& offsets() const { return regionOffsets; }
  binType* weights() { return weightCUDA; }
  const int* indices() const { return indexCUDA; }
  const int* deviceOffsets() const { return blockOffsetCUDA; }
  int blocksPerRegion() const { return regionBlocks; }

  /**
   * @brief The description of a records file of such bins (describeRecords):
   * the rank (JSON members), the regions, the bins and the weight scale
   * (JSON members that the records of the file need, into layout); the
   * records: SparseRecordHeader, the float64 range of the kind (rangeName,
   * of shape rangeShape), then counts (uint32 per region), weights (float32
   * per non-empty bin) and index (the bytes of encodeBinIndex, described by
   * the JSON member index_encoding); content (what the records hold), then
   * the description of index.
   */
  std::string describe(const std::string& rank, const std::string& regionsText,
                       const std::string& binsText,
                       const std::string& rangeName,
                       const std::string& rangeShape, std::uint32_t magic,
                       const std::string& content,
                       std::vector<std::string>& layout) const;

  /** Add to file one record of the non-empty bins of the last compact():
   *  SparseRecordHeader, the count values of range, then the arrays of
   *  describe(); none without a non-empty bin. */
  void writeRecord(RecordFile& file, int species, int cycle,
                   const double* range, std::size_t count, cudaStream_t stream);

private:
  const int regions, regionBins, regionBlocks;
  int nonEmpty = 0, capacity = 0;
  binType* weightCUDA = nullptr;
  int* indexCUDA = nullptr;
  int* blockOffsetCUDA = nullptr;  // per block of bins, its first; + total
  int* blockOffsetHost = nullptr;  // pinned copy
  std::vector<int> regionOffsets;  // regions + 1
  // the staging of writeRecord(), page-locked: copied from the device
  PinnedVector<binType> weightHost;
  PinnedVector<int> indexHost;
};

} // namespace sparseBins

#endif // SPARSE_BINS_CUH
