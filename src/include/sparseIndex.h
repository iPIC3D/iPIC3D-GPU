#ifndef SPARSE_INDEX_H
#define SPARSE_INDEX_H

#include <cstdint>
#include <vector>

/**
 * @file sparseIndex.h
 * @brief Which bins of a record of sparse bins (the histograms and the
 * spectra of the data analysis) are non-empty, region after region: the list
 * of their indices or, when smaller, the bitmap of the region. Host only.
 *
 * A region of B bins with n non-empty ones gives their n indices, increasing,
 * of binIndexBytes(B) bytes each in the byte order of the host, unless those
 * take more than its bitmap of bitmapBytes(B) bytes (isBitmap), whose bit
 * b % 8 of byte b / 8 is set for a non-empty bin b. Where a region starts
 * follows from the counts of the regions before it.
 */
namespace sparseBins {

/** A region of at most this many bins lists the index of a bin as 16 bits,
 *  as 32 bits otherwise. */
inline constexpr int kShortIndexBins = 65536;
/** Bins per byte of a bitmap. */
inline constexpr int kBinsPerBitmapByte = 8;

/** The bytes of the index of a bin in a region of regionBins bins. */
inline int binIndexBytes(int regionBins) {
  return regionBins <= kShortIndexBins ? 2 : 4;
}

/** The bytes of the bitmap of a region of regionBins bins. */
inline int bitmapBytes(int regionBins) {
  return (regionBins + kBinsPerBitmapByte - 1) / kBinsPerBitmapByte;
}

/** Whether a region of regionBins bins with count non-empty ones gives them
 *  as its bitmap. */
inline bool isBitmap(long long count, int regionBins) {
  return count * binIndexBytes(regionBins) > bitmapBytes(regionBins);
}

/** The index bytes of regions of regionBins bins (at least one), region r
 *  with the non-empty bins index[offsets[r]], ..., index[offsets[r + 1] - 1],
 *  increasing. */
std::vector<std::uint8_t> encodeBinIndex(const std::vector<int>& offsets,
                                         const int* index, int regionBins);

} // namespace sparseBins

#endif // SPARSE_INDEX_H
