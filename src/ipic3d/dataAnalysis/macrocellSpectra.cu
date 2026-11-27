#include "macrocellSpectra.cuh"

#include "cudaTypeDef.cuh"
#include "gridCUDA.cuh"
#include "particleArrayCUDA.cuh"

#include <sstream>
#include <stdexcept>
#include <utility>

namespace macrocellSpectra {

/** Threads per block of the spectra kernel. */
inline constexpr int kSpectraThreads = 128;
static_assert(kSpectraThreads >= kCornerB,
              "a thread per value of B of a cell in the kernel");

// ======= macrocellSpectra2D =======

macrocellSpectra2D::macrocellSpectra2D(const MacrocellPartition& part,
                                       std::vector<cudaCommonType> vmax)
    : part_(part), vmax_(std::move(vmax)),
      slices_(macrocell::slicesFor(part.M, 1, part.largestCells())),
      numFloats_(static_cast<size_t>(part.M) * kSpectraBins),
      sparse_(part.M, kSpectraBins) {
  cudaErrChk(cudaMalloc(&dHist_, numFloats_ * sizeof(macrocellHistType)));
}

macrocellSpectra2D::~macrocellSpectra2D() { cudaErrChk(cudaFree(dHist_)); }

void macrocellSpectra2D::launch(particleArrayCUDA* pclsHostPtr,
                                cudaFieldType* fieldForPclsCUDA,
                                const grid3DCUDA& grid,
                                const int* cellStartOffsetsCUDA,
                                const int* cellCountsCUDA, int species,
                                cudaStream_t stream) {
  // no particle (its cells are not sorted): every bin 0; several slices per
  // macrocell add up their bins
  const bool empty = pclsHostPtr->getNOP() == 0;
  if (empty || slices_ > 1)
    cudaErrChk(cudaMemsetAsync(dHist_, 0,
                               numFloats_ * sizeof(macrocellHistType), stream));
  if (empty)
    return;

  // One block per slice of a macrocell. Dynamic shared memory is only the
  // per-block mini-histogram; B corners use a small statically aligned shared
  // array in the kernel.
  const size_t shmemBytes = kSpectraBins * sizeof(macrocellHistType);

  macrocellSpectraKernel<<<part_.M * slices_, kSpectraThreads, shmemBytes,
                           stream>>>(
      pclsHostPtr->getX(), pclsHostPtr->getY(), pclsHostPtr->getZ(),
      pclsHostPtr->getU(), pclsHostPtr->getV(), pclsHostPtr->getW(),
      pclsHostPtr->getQ(), cellStartOffsetsCUDA, cellCountsCUDA,
      part_.geometry(), slices_, fieldForPclsCUDA, grid, dHist_,
      vmax_[species]);
  cudaErrChk(cudaGetLastError());
}

// ======= Output =======

std::string
macrocellSpectra2D::describe(const std::string& rank,
                             std::vector<std::string>& layout) const {
  std::ostringstream binsText, bMin;
  binsText << "  \"bins\": {\"vpar\": " << DAConfig::MACROCELL_BINS_VPAR
           << ", \"vperp\": " << DAConfig::MACROCELL_BINS_VPERP << "},\n";
  bMin << DAConfig::MACROCELL_BMIN;
  return sparse_.describe(
      rank, macrocell::regionsDescription(&part_), binsText.str(), "range", "4",
      kSpectraRecordMagic,
      "one record per analysis cycle and species of VelocitySpectraSpecies "
      "with a non-empty bin: range = [vpar_min, vpar_max, vperp_min, "
      "vperp_max] of the species; counts[r] the non-empty bins of region r, "
      "macrocell m = (mz*My + my)*Mx + mx; weights region after region, by "
      "increasing bin b = ipar + bins_vpar * iperp within one, bin i of an "
      "axis centred at min + (i + 0.5) * (max - min) / bins; v_par is along "
      "the local B, v_perp its magnitude across; a weight is the sum of |q| "
      "weight_scale of the particles of its bin, those where |B| is at most " +
          bMin.str() + " left out; the empty bins are not written",
      layout);
}

void macrocellSpectra2D::writeRecord(RecordFile& file, int species, int cycle,
                                     cudaStream_t stream) {
  sparse_.compact(dHist_, stream);
  const double vmax = vmax_[species];
  const double range[4] = {-vmax, vmax, 0, vmax};
  sparse_.writeRecord(file, species, cycle, range, 4, stream);
}

} // namespace macrocellSpectra
