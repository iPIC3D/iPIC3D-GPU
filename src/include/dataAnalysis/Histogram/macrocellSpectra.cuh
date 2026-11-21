#ifndef MACROCELL_SPECTRA_CUH
#define MACROCELL_SPECTRA_CUH

#include <string>
#include <vector>

#include "cudaTypeDef.cuh"
#include "dataAnalysisConfig.cuh"
#include "macrocellPartition.cuh"
#include "outputPrepare.h"
#include "sparseBins.cuh"

// Forward declarations to keep this header lightweight.
class particleArrayCUDA;
class grid3DCUDA;

namespace macrocellSpectra {

/** The first word of a record of the spectra, "SPC1" in the little-endian
 *  file (sparseBins::SparseRecordHeader). */
inline constexpr std::uint32_t kSpectraRecordMagic = 0x31435053;
/** Bins of the spectrum of a macrocell. */
inline constexpr int kSpectraBins =
    DAConfig::MACROCELL_BINS_VPAR * DAConfig::MACROCELL_BINS_VPERP;
/** Values of B per cell that the kernel keeps in shared memory: the three
 *  components at the eight corners. */
inline constexpr int kCornerB = 24;

using macrocell::MacrocellPartition;

// Histogram element type for per-macrocell (v_par, v_perp) bins: the bins of
// the records (sparseBins::binType).
using macrocellHistType = sparseBins::binType;
static_assert(kSpectraBins * sizeof(macrocellHistType) +
                      kCornerB * sizeof(cudaCommonType) <=
                  48 * 1024,
              "the bins of a macrocell and the B of a cell fit the default "
              "shared memory");

// ======= Device-side flat 2D (v_par, v_perp) spectra accumulator =======

/**
 * @brief Per-species, per-macrocell (v_par, v_perp) histogram accumulator.
 * One block per macrocell, or per slice of its cells when the macrocells are
 * few (macrocell::slicesFor), whose bins are then added up with atomics.
 *
 * Layout: a single flat float buffer of size
 *     M * (MACROCELL_BINS_VPERP * MACROCELL_BINS_VPAR)
 * with vperp as the outer index and vpar as the inner index inside one
 * macrocell slice. Bin metadata (range and resolution) is uniform across
 * macrocells for the active species.
 *
 * The ranges of species s are
 *   - vpar  in [-vmax_s, +vmax_s]
 *   - vperp in [0,       +vmax_s]
 * with vmax_s its VelocityHistogramMax.
 *
 * Output: writeRecord adds the non-empty bins of the macrocells of a species
 * as one record to the records file of the rank (describe).
 */
class macrocellSpectra2D {
public:
  /**
   * @param part The macrocells of this rank.
   * @param vmax Velocity range of each species.
   */
  macrocellSpectra2D(const MacrocellPartition& part,
                     std::vector<cudaCommonType> vmax);

  /**
   * @brief Launch the fused interpolate-and-bin kernel for one species: it
   * writes every bin of every macrocell (all 0 for a species without
   * particles, whose cells the sort left as they were).
   *
   * The borrowed pointers must come from `c_Solver`:
   *   - `pclsHostPtr->getX()/Y()/Z()/U()/V()/W()/Q()` device pointers
   *   - `fieldForPclsCUDA`: packed mover field buffer (B in components 0..2)
   *   - `gridDevicePtr`:    device-resident `grid3DCUDA`
   *   - `cellStartOffsetsCUDA`, `cellCountsCUDA`: from the per-species
   *      `CellSorter`. Particles must already be cell-sorted.
   */
  void launch(particleArrayCUDA* pclsHostPtr, cudaFieldType* fieldForPclsCUDA,
              const grid3DCUDA* gridDevicePtr, const int* cellStartOffsetsCUDA,
              const int* cellCountsCUDA, int species, cudaStream_t stream);

  /** The description of the records file of the spectra
   *  (SparseBins::describe) after rank, the JSON members of the rank, and
   *  into layout its parts that its records need. */
  std::string describe(const std::string& rank,
                       std::vector<std::string>& layout) const;

  /** Add to file one record of the non-empty bins of the last launch():
   *  sparseBins::SparseRecordHeader, the range [vpar_min, vpar_max,
   *  vperp_min, vperp_max] of the species (float64), then
   *  the arrays of SparseBins::writeRecord. */
  void writeRecord(RecordFile& file, int species, int cycle,
                   cudaStream_t stream);

  ~macrocellSpectra2D();

  macrocellSpectra2D(const macrocellSpectra2D&) = delete;
  macrocellSpectra2D& operator=(const macrocellSpectra2D&) = delete;

private:
  MacrocellPartition part_;
  std::vector<cudaCommonType> vmax_; ///< velocity range of each species
  /// the slices of each macrocell, one block each (macrocell::slicesFor)
  const int slices_;

  // Flat histogram buffer on the device, M * kSpectraBins elements; shared
  // and reused across species sequentially.
  macrocellHistType* dHist_ = nullptr;
  size_t numFloats_ = 0;          ///< == part_.M * kSpectraBins
  sparseBins::SparseBins sparse_; // its non-empty bins
};

} // namespace macrocellSpectra

// ======= Device-side kernel declarations =======

__global__ void macrocellSpectraKernel(
    // Particle SoA
    const cudaCommonType* __restrict__ x, const cudaCommonType* __restrict__ y,
    const cudaCommonType* __restrict__ z, const cudaCommonType* __restrict__ u,
    const cudaCommonType* __restrict__ v, const cudaCommonType* __restrict__ w,
    const cudaCommonType* __restrict__ q,
    // Cell sort
    const int* __restrict__ cellStartOffsets,
    const int* __restrict__ cellCounts,
    // The macrocells (their tiling of the interior cells), and the slices of
    // the cells of each, one block per slice
    macrocell::MacrocellGeometry geometry, int slices,
    // Field & grid
    const cudaFieldType* __restrict__ fieldForPcls,
    const grid3DCUDA* __restrict__ grid,
    // Histogram parameters
    macrocellSpectra::macrocellHistType* __restrict__ histOut, // [M * bins]
    cudaCommonType vmax);

#endif // MACROCELL_SPECTRA_CUH
