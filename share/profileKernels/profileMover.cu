/**
 * @file profileMover.cu
 * @brief Profile of moverSubcyclesKernel, the relativistic mover with
 * subcycles that the solver runs in the planet cases (Dipole, Dipole2D), on
 * particles sorted by cell: the order in which the solver's mover reads them
 * after the sort of the previous cycle.
 *
 * The fields are uniform (EMfields3D::init): the E of the input file, and a B
 * along z whose |B| makes every particle take N subcycles (--subcycles N,
 * default 4; 0 keeps the B of the input file), moved to the device and packed
 * for the mover as the solver does (gpuSolverSyncH2D,
 * gpuPackFieldForPclsToCenter).  Every timed launch pushes the same
 * particles: their positions and velocities are put back, and the departure
 * flags cleared, before each launch.
 */

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <string>

#include "GPUFieldPacking.cuh"
#include "KernelProfile.cuh"
#include "moverKernel.cuh"

namespace {

/** Values per cell of the packed mover fields: B and E at the 8 nodes of the
 *  cell (gpuPackFieldForPclsToCenter). */
constexpr int kFieldValuesPerCell = 24;
/** The particle fields the mover writes: the position and the velocity. */
constexpr int kWrittenFields = 6;

/** The positions and velocities before the timed launches, put back before
 *  each launch so that every launch pushes the same particles. */
class MoverInput {
public:
  explicit MoverInput(profileKernels::KernelProfile& profile)
      : stream_(profile.stream()), count_(profile.particleCount()) {
    particleArrayCUDA* p = profile.pclsArrayHost();
    live_[0] = {p->getX(), sizeof(*p->getX())};
    live_[1] = {p->getY(), sizeof(*p->getY())};
    live_[2] = {p->getZ(), sizeof(*p->getZ())};
    live_[3] = {p->getU(), sizeof(*p->getU())};
    live_[4] = {p->getV(), sizeof(*p->getV())};
    live_[5] = {p->getW(), sizeof(*p->getW())};
    for (int f = 0; f < kWrittenFields; ++f) {
      cudaErrChk(cudaMalloc(&saved_[f], live_[f].bytesPerParticle * count_));
      copy(saved_[f], live_[f].data, live_[f].bytesPerParticle);
    }
    cudaErrChk(cudaStreamSynchronize(stream_));
  }
  ~MoverInput() {
    for (void* f : saved_)
      cudaFree(f);
  }
  MoverInput(const MoverInput&) = delete;
  MoverInput& operator=(const MoverInput&) = delete;

  /** Put the saved state back, on the stream. */
  void restore() {
    for (int f = 0; f < kWrittenFields; ++f)
      copy(live_[f].data, saved_[f], live_[f].bytesPerParticle);
  }

private:
  struct Field {
    void* data;
    std::size_t bytesPerParticle;
  };
  void copy(void* to, const void* from, std::size_t bytesPerParticle) {
    cudaErrChk(cudaMemcpyAsync(to, from, bytesPerParticle * count_,
                               cudaMemcpyDeviceToDevice, stream_));
  }

  cudaStream_t stream_;
  std::uint32_t count_;
  Field live_[kWrittenFields] = {};
  void* saved_[kWrittenFields] = {};
};

} // namespace

int main(int argc, char** argv) {
  using namespace profileKernels;
  KernelProfile profile(argc, argv, "moverSubcyclesKernel", Order::CellSorted,
                        "--subcycles",
                        sizeof(departureArrayElementType) +
                            kWrittenFields * sizeof(cudaParticleType));
  // the mover needs the sorted particles, not the sorter
  profile.releaseCellSorter();
  const cudaStream_t stream = profile.stream();
  const std::uint32_t n = profile.particleCount();

  // the fields of the mover, as the solver moves and packs them (initCUDA,
  // refreshFieldForPclsDeviceBuffer)
  EMfields3D& fields = profile.fields();
  fields.gpuSolverAllocate();
  fields.gpuSolverSyncH2D(stream);
  const Grid3DCU& grid = profile.grid();
  const int cells = (grid.getNXN() - 1) * (grid.getNYN() - 1) * grid.getNZN();
  const std::size_t fieldBytes =
      std::size_t(cells) * kFieldValuesPerCell * sizeof(cudaFieldType);
  profile.requireDeviceMemory(fieldBytes, "the mover fields");
  cudaFieldType* fieldForPcls = nullptr;
  cudaErrChk(cudaMalloc(&fieldForPcls, fieldBytes));
  gpuPackFieldForPclsToCenter<<<getGridSize(cells, kPackFieldBlock),
                                kPackFieldBlock, 0, stream>>>(
      fieldForPcls, fields.gpuExForPcls().devPtr(),
      fields.gpuEyForPcls().devPtr(), fields.gpuEzForPcls().devPtr(),
      fields.gpuBxn().devPtr(), fields.gpuByn().devPtr(),
      fields.gpuBzn().devPtr(), fields.gpuBx_ext().devPtr(),
      fields.gpuBy_ext().devPtr(), fields.gpuBz_ext().devPtr(), grid.getNXN(),
      grid.getNYN(), grid.getNZN());
  cudaErrChk(cudaGetLastError());

  // the mover parameters of the species (initCUDA), no planet in the box and
  // no wall list
  MoverCounters* counters = nullptr;
  cudaErrChk(cudaMalloc(&counters, sizeof(MoverCounters)));
  cudaErrChk(cudaMemsetAsync(counters, 0, sizeof(MoverCounters), stream));
  const moverParameter mover(&profile.particlesHost(),
                             profile.pclsArrayDevice(),
                             profile.departureArrayDevice(),
                             profile.hashedSumArrayDevice(), nullptr, counters);
  if (mover.doReflect)
    fail("the input file has reflecting walls, whose pass "
         "(moverSubcyclesWallsKernel) is not profiled: make the box periodic");

  MoverInput input(profile);
  const int subcycles =
      moverSubcycles(profile.collective(), profile.options().species);
  profile.time(
      "subcycles=" + std::to_string(subcycles), n, profiling::Region::Mover,
      [&] {
        input.restore();
        profile.resetDepartures();
      },
      [&] {
        moverSubcyclesKernel<<<getGridSize((int)n, kMoverBlockSize),
                               kMoverBlockSize, 0, stream>>>(
            mover, fieldForPcls, profile.gridCUDA());
      });

  // the departures of the last launch: a particle the mover deleted had a
  // speed at or beyond c, or not finite
  hashedSum* sums = profile.hashedSumArrayHost();
  cudaErrChk(cudaMemcpyAsync(sums, profile.hashedSumArrayDevice(),
                             departureArrayElementType::HASHED_SUM_NUM *
                                 sizeof(hashedSum),
                             cudaMemcpyDefault, stream));
  cudaErrChk(cudaStreamSynchronize(stream));
  int leaving = 0;
  for (int j = departureArrayElementType::XLOW_HASHEDSUM_INDEX;
       j <= departureArrayElementType::ZHIGH_HASHEDSUM_INDEX; j++)
    leaving += sums[j].getSum();
  const int deleted =
      sums[departureArrayElementType::DELETE_HASHEDSUM_INDEX].getSum();
  std::printf("  departure check: %d particles leave the box (%.3f %%), %d "
              "deleted: %s\n",
              leaving, 100.0 * leaving / n, deleted,
              deleted == 0 ? "pass" : "FAIL");

  cudaFree(counters);
  cudaFree(fieldForPcls);
  return profile.finish(deleted == 0);
}
