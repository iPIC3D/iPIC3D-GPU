#ifndef KERNEL_PROFILE_CUH
#define KERNEL_PROFILE_CUH

/**
 * @file KernelProfile.cuh
 * @brief Set-up and timing shared by the kernel profiles (profileMover.cu,
 * profileCellAwareMoment.cu, profileMomentKernelNew.cu).
 *
 * A profile builds one species of a periodic box of uniform plasma on one GPU
 * with the classes of the solver, in the solver's order: the input file
 * (Collective), the topology and the grid, the fields (EMfields3D::init), the
 * Maxwellian particles (ParticleSoAHost::maxwellian) and their device arrays
 * (particleArrayCUDA).  The particles are then sorted by cell with the
 * solver's CellSorter, or shuffled, and the profile times repeated launches of
 * one kernel.
 *
 * MPI is initialised on the one process because the topology, the grid and
 * the fields are built on it; nothing is communicated.
 */

#include <array>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>

#include "Collective.h"
#include "EMfields3D.h"
#include "Grid3DCU.h"
#include "ParticleSoAHost.h"
#include "Profiler.h"
#include "VCtopology3D.h"
#include "cellSortBuffers.cuh"
#include "cudaTypeDef.cuh"
#include "gridCUDA.cuh"
#include "hashedSum.cuh"
#include "particleArrayCUDA.cuh"
#include "particleExchange.cuh"

namespace profileKernels {

/** Untimed launches before the timed ones, by default. */
inline constexpr int kDefaultWarmupLaunches = 2;
/** Timed launches, by default. */
inline constexpr int kDefaultTimedLaunches = 10;
/** profileMover: subcycles of the relativistic push, by default. */
inline constexpr int kDefaultMoverSubcycles = 4;

/** The order of the particles in device memory. */
enum class Order {
  CellSorted, ///< sorted by cell with the solver's CellSorter
  Shuffled    ///< a random permutation of the particles
};

/** The command-line options of a profile (see the usage in KernelProfile.cu).
 */
struct Options {
  std::string deck;         ///< the input file
  int cells[3] = {0, 0, 0}; ///< cells per axis; 0 keeps the input file's
  int ppc = 0;              ///< particles per cell; 0 keeps the input file's
  int species = 0;          ///< the species profiled
  int warmup = kDefaultWarmupLaunches;
  int launches = kDefaultTimedLaunches;
  /// profileMover: subcycles of the push, from a uniform B along z
  /// (kDefaultMoverSubcycles by default); 0 keeps the B of the input file
  int subcycles = 0;
  double tail = 1.0; ///< profileMomentKernelNew: fraction deposited
};

/** The times of the timed launches, in milliseconds. */
struct LaunchTimes {
  double min;
  double median;
  double mean;
  double max;
};

/**
 * @brief One species of a periodic box on one GPU, and the timing of a kernel.
 *
 * The accessors follow the names of the solver (c_Solver): particlesHost is
 * the ParticleSoAHost, pclsArrayHost and pclsArrayDevice the host and device
 * copies of the particleArrayCUDA descriptor, and so on.  The departure
 * array, the hashed sums and the moments are allocated on first use.
 */
class KernelProfile {
public:
  /**
   * @param kernel Name of the profiled kernel, for the output.
   * @param order Order of the particles in device memory.
   * @param extraOption The option of this profile besides the common ones
   *        ("--subcycles", "--tail"), or nullptr.
   * @param extraBytesPerParticle Device bytes per particle that the profile
   *        allocates besides the particle arrays and the cell sort: the
   *        memory is checked before the particles are made.
   */
  KernelProfile(int argc, char** argv, const char* kernel, Order order,
                const char* extraOption, std::size_t extraBytesPerParticle);
  ~KernelProfile();
  KernelProfile(const KernelProfile&) = delete;
  KernelProfile& operator=(const KernelProfile&) = delete;

  const Options& options() const { return options_; }
  cudaStream_t stream() const { return stream_; }
  std::uint32_t particleCount() const { return pclsArrayHost_->getNOP(); }

  Collective& collective() { return *col_; }
  Grid3DCU& grid() { return *grid_; }
  EMfields3D& fields() { return *fields_; }
  ParticleSoAHost& particlesHost() { return *particlesHost_; }

  particleArrayCUDA* pclsArrayHost() { return pclsArrayHost_; }
  particleArrayCUDA* pclsArrayDevice() { return pclsArrayDevice_; }
  grid3DCUDA* grid3DDevice() { return grid3DDevice_; }
  departureArrayType* departureArrayDevice();
  hashedSum* hashedSumArrayHost();
  hashedSum* hashedSumArrayDevice();

  /** Clear the departure flags and the hashed sums, as the solver does at
   *  the end of each cycle. */
  void resetDepartures();

  /** The sorter of Order::CellSorted, with the cell offsets of the sort. */
  const CellSorter& cellSorter() const { return cellSorter_; }
  /** Free the sorter's buffers when only the sorted particles are needed. */
  void releaseCellSorter() { cellSorter_.free(); }

  /** The moments of the species (10 node arrays, as momentsCUDAPtr). */
  cudaMomentType* moments();
  /** Zero the moments on the stream, as the solver does before a deposit. */
  void clearMoments();
  /**
   * @brief Compare the charge in the deposited density with the charge of the
   * particles [first, first + count) of the device arrays.
   * @return True when they agree to the relative kChargeTolerance of
   *         KernelProfile.cu (the atomic additions only reorder the sum).
   */
  bool checkDepositedCharge(std::uint32_t first, std::uint32_t count);

  /** Stop the run when the device has not @p bytes free above a margin. */
  void requireDeviceMemory(std::size_t bytes, const char* what) const;

  /**
   * @brief Time the launches of the kernel: options().warmup untimed, then
   * options().launches timed, each after @p prepare.
   *
   * The timed launches run inside the profiler range @p region of the
   * species and inside the recording window of the profilers
   * (profiling::recordSteadyState), so that the Nsight and CrayPat options of
   * the solver's profiling work here too.  Prints the times; finish writes
   * the result line.
   *
   * @param parameter Kernel-specific parameter for the output ("tail=1").
   * @param particles Particles the kernel processes per launch.
   * @param prepare Work before each launch, on the stream (not timed).
   * @param launch The launch, on the stream.
   */
  void time(const std::string& parameter, std::uint32_t particles,
            profiling::Region region, const std::function<void()>& prepare,
            const std::function<void()>& launch);

  /**
   * @brief Print the result line that the scripts read (one JSON object:
   * workload, build, times of time(), @p checked) and return the exit code
   * of the profile.
   * @param checked The check of the profile passed.
   * @return EXIT_SUCCESS when @p checked, else EXIT_FAILURE.
   */
  int finish(bool checked);

private:
  void buildHost(int argc, char** argv, const char* extraOption,
                 std::size_t extraBytesPerParticle);
  void buildDevice();
  void printSetup() const;
  void requireHostMemory(std::size_t bytes, const char* what) const;
  /** Nodes of the grid, ghosts included: the size of one moment array. */
  std::size_t nodeCount() const;
  /** Median time of the last time() per particle, in nanoseconds. */
  double nsPerParticle() const;

  Options options_;
  std::string kernel_;
  Order order_;
  std::string gpu_;
  std::array<int, 3> cells_{};
  std::array<int, 3> npcel_{};
  std::string parameter_;             ///< of the last time()
  std::uint32_t launchParticles_ = 0; ///< of the last time()
  LaunchTimes times_{};               ///< of the last time()

  std::unique_ptr<Collective> col_;
  std::unique_ptr<VCtopology3D> vct_;
  std::unique_ptr<Grid3DCU> grid_;
  std::unique_ptr<EMfields3D> fields_;
  std::unique_ptr<ParticleSoAHost> particlesHost_;

  cudaStream_t stream_ = nullptr;
  cudaEvent_t start_ = nullptr;
  cudaEvent_t stop_ = nullptr;
  particleArrayCUDA* pclsArrayHost_ = nullptr;
  particleArrayCUDA* pclsArrayDevice_ = nullptr;
  grid3DCUDA* grid3DHost_ = nullptr;
  grid3DCUDA* grid3DDevice_ = nullptr;
  departureArrayType* departureArrayHost_ = nullptr;
  departureArrayType* departureArrayDevice_ = nullptr;
  hashedSum* hashedSumArrayHost_ = nullptr;
  hashedSum* hashedSumArrayDevice_ = nullptr;
  cudaMomentType* moments_ = nullptr;
  CellSorter cellSorter_;
};

/** The subcycles of the relativistic push (pushSubcycled in moverKernel.cu)
 *  of a particle of @p species in the uniform B of the input file. */
int moverSubcycles(const Collective& col, int species);

/** Print a message and end the run (with MPI_Finalize). */
[[noreturn]] void fail(const std::string& message);

/** The name of the current device, as the result lines and profileCeilings
 *  print it. */
std::string deviceName();

/** @p text as a JSON string. */
std::string jsonString(const std::string& text);

} // namespace profileKernels

#endif // KERNEL_PROFILE_CUH
