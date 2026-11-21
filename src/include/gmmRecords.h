#ifndef GMM_RECORDS_H
#define GMM_RECORDS_H

#include <cstdint>
#include <limits>
#include <string>
#include <vector>

#include "outputPrepare.h"

/**
 * @file gmmRecords.h
 * @brief The records of the GMM fits of the velocity histograms
 * (velocityGMM/subDomain<rank>_gmm.bin, a RecordFile): one per analysis cycle
 * and species with a fit, holding only the fitted regions and their active
 * (not pruned) components. Host only.
 */

namespace dataAnalysis {

/** The types in a record: of the weights, means and covariances; of the
 *  region and of the number of active components of a fit; of the slot of a
 *  component. */
using GMMRecordValue = float;
using GMMRegionIndex = std::uint16_t;
using GMMComponentCount = std::uint16_t;
using GMMSlot = std::uint8_t;

/** The first word of a record of the GMM fits, "GMM2" in the little-endian
 *  file (GMMRecordHeader). */
inline constexpr std::uint32_t kGMMRecordMagic = 0x324d4d47;
/** Dimension of the fitted velocities (u, v, w). */
inline constexpr int kGMMRecordDim = 3;
/** Elements written of a covariance: its upper triangle, row after row. */
inline constexpr int kGMMPackedCovariance =
    kGMMRecordDim * (kGMMRecordDim + 1) / 2;
/** Regions of a rank and components of a fit at most, that a record can
 *  index. */
inline constexpr long long kGMMMaxRegions =
    (long long)std::numeric_limits<GMMRegionIndex>::max() + 1;
inline constexpr long long kGMMMaxSlots =
    (long long)std::numeric_limits<GMMSlot>::max() + 1;

/** The header of a record (RecordFile), followed by the arrays of
 *  writeGMMRecord. */
struct GMMRecordHeader {
  std::uint32_t magic; // kGMMRecordMagic
  std::int32_t cycle;
  std::int32_t species;
  std::uint32_t regions;    // of the layout: 1 (the rank), or the macrocells
  std::uint32_t fits;       // the regions with a fit, those of the record
  std::uint32_t components; // the active components of those fits
  std::uint64_t bytes;      // of the record, this header and padding included
};

/** A fit to write: its region, EM steps and log-likelihood, and its
 *  GMMComponents components (weights, means, row-major covariances), of
 *  which the active ones are written. */
struct GMMFitView {
  int region, steps;
  double logLikelihood;
  const bool* active;
  const double* weight;
  const double* mean;
  const double* coVariance;
};

/** The description of a GMM records file (describeRecords) after members,
 *  the JSON members of the rank and of its regions. */
std::string describeGMMRecords(const std::string& members);

/** Append to file the record of the fits of a species at a cycle: at least
 *  one, of increasing regions in [0, regions), each of `components`
 *  components (GMMComponents); another call throws invalid_argument. */
void writeGMMRecord(RecordFile& file, int cycle, int species, int regions,
                    int components, const std::vector<GMMFitView>& fits);

} // namespace dataAnalysis

#endif // GMM_RECORDS_H
