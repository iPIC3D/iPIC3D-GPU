/**
 * @file profileCellAwareMoment.cu
 * @brief Profile of cellAwareMomentKernel, the moment deposit of the
 * particles sorted by cell (one warp per cell), on the particles sorted by the
 * solver's CellSorter, with its cell offsets: the deposit of the solver's
 * sorting cycles (SortingCycle).
 */

#include <cstdint>

#include "KernelProfile.cuh"
#include "momentKernel.cuh"

int main(int argc, char** argv) {
  using namespace profileKernels;
  KernelProfile profile(argc, argv, "cellAwareMomentKernel", Order::CellSorted,
                        nullptr, 0);
  const CellSorter& sorter = profile.cellSorter();
  const std::uint32_t n = profile.particleCount();
  const int numCells = sorter.getNumCells();
  const int threads = numCells * WARP_SIZE; // one warp per cell
  cudaMomentType* moments = profile.moments();

  profile.time(
      "-", n, profiling::Region::MomentDeposit, [&] { profile.clearMoments(); },
      [&] {
        cellAwareMomentKernel<<<getGridSize(threads, DEFAULT_BLOCK_SIZE),
                                DEFAULT_BLOCK_SIZE, 0, profile.stream()>>>(
            sorter.getCellStartOffsets(), numCells, n,
            profile.pclsArrayDevice(), profile.grid3DDevice(), moments);
      });
  return profile.finish(profile.checkDepositedCharge(0, n));
}
