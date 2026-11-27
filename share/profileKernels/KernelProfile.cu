/**
 * @file KernelProfile.cu
 * @brief Set-up and timing shared by the kernel profiles (KernelProfile.cuh).
 */

#include "KernelProfile.cuh"

#include <algorithm>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <numeric>
#include <random>
#include <sstream>
#include <type_traits>
#include <vector>

#include "BuildInfo.h"
#include "ConfigFile.h"
#include "MPIdata.h"
#include "Parameters.h"

namespace profileKernels {

namespace {

/** Device memory that every allocation check leaves free, in bytes. */
constexpr std::size_t kMemoryMarginBytes = std::size_t(512) << 20;
/** Capacity of the particle arrays, times the particle count: the profiled
 *  kernels append no particle. */
constexpr float kParticleCapacityFactor = 1.0f;
/** Seed of the random particle order (Order::Shuffled). */
constexpr std::uint64_t kShuffleSeed = 0x5eed5eedULL;
/** Largest relative error of the deposited charge (checkDepositedCharge):
 *  the atomic additions only reorder the sum. */
constexpr double kChargeTolerance = 1.0e-10;
/** Moment arrays of a species (charge density, current, pressure), as the
 *  solver's momentsCUDAPtr. */
constexpr int kMomentCount = 10;
/** Buckets of each hashed sum, as the solver (initCUDA). */
constexpr int kHashedSumBuckets = 10;
/** Device bytes per particle of the particle arrays (no particle IDs). */
constexpr std::size_t kParticleBytes =
    sizeof(cudaPclType_U) + sizeof(cudaPclType_V) + sizeof(cudaPclType_W) +
    sizeof(cudaPclType_Q) + sizeof(cudaPclType_X) + sizeof(cudaPclType_Y) +
    sizeof(cudaPclType_Z);
/** Device bytes per particle of the cell sort: the permutation and the
 *  scratch field of its scatter (CellSorter::init). */
constexpr std::size_t kSortBytes = sizeof(unsigned int) + sizeof(double);
/** The input-file keys of the cells and of the length of each axis. */
constexpr const char* kAxisKeys[3][2] = {
    {"nxc", "Lx"}, {"nyc", "Ly"}, {"nzc", "Lz"}};
/** Host bytes per particle of the shuffle: the permutation and one field. */
constexpr std::size_t kShuffleBytes = sizeof(std::uint32_t) + sizeof(double);
/** Bytes per mebibyte, for the output. */
constexpr double kMiB = 1024.0 * 1024.0;
/** Nanoseconds per millisecond. */
constexpr double kNsPerMs = 1.0e6;
/** The options of every profile; the defaults follow. */
constexpr const char* kUsage =
    "usage: %s <input file> [options]\n"
    "  --cells NX NY NZ  cells per axis; the cell size of the input file is "
    "kept\n"
    "  --ppc N           particles per cell of every species\n"
    "  --species S       species profiled (default 0)\n"
    "  --warmup W        untimed launches before the timed ones (default %d)\n"
    "  --launches L      timed launches (default %d)\n";

/** The integer @p text of @p option, at least @p minimum. */
int intOption(const char* text, const std::string& option, int minimum) {
  char* end = nullptr;
  const long value = std::strtol(text, &end, 10);
  if (end == text || *end != '\0' || value < minimum || value > INT_MAX)
    fail(option + " takes an integer >= " + std::to_string(minimum) +
         ", not '" + text + "'");
  return static_cast<int>(value);
}

void printUsage(const char* program, const std::string& extraOption) {
  std::fprintf(stderr, kUsage, program, kDefaultWarmupLaunches,
               kDefaultTimedLaunches);
  if (extraOption == "--subcycles")
    std::fprintf(stderr,
                 "  --subcycles N     subcycles of the relativistic push: "
                 "sets a uniform B along z (default %d; 0 keeps the B of the "
                 "input file)\n",
                 kDefaultMoverSubcycles);
  if (extraOption == "--tail")
    std::fprintf(stderr, "  --tail F          fraction of the particles "
                         "deposited, the last ones (default 1)\n");
}

Options parseOptions(int argc, char** argv, const char* extraOptionName) {
  const std::string extraOption = extraOptionName ? extraOptionName : "";
  Options options;
  // only profileMover sets B for its subcycles: the other profiles keep the
  // B of the input file
  if (extraOption == "--subcycles")
    options.subcycles = kDefaultMoverSubcycles;
  // the next value of option, which i then points to
  const auto value = [&](int& i, const std::string& option) -> const char* {
    if (i + 1 >= argc)
      fail(option + " needs a value");
    return argv[++i];
  };
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "-h" || arg == "--help") {
      printUsage(argv[0], extraOption);
      MPIdata::exit(EXIT_SUCCESS);
    } else if (arg == "--cells") {
      for (int a = 0; a < 3; ++a)
        options.cells[a] = intOption(value(i, arg), arg, 1);
    } else if (arg == "--ppc") {
      options.ppc = intOption(value(i, arg), arg, 1);
    } else if (arg == "--species") {
      options.species = intOption(value(i, arg), arg, 0);
    } else if (arg == "--warmup") {
      options.warmup = intOption(value(i, arg), arg, 0);
    } else if (arg == "--launches") {
      options.launches = intOption(value(i, arg), arg, 1);
    } else if (arg == "--subcycles" && extraOption == arg) {
      options.subcycles = intOption(value(i, arg), arg, 0);
    } else if (arg == "--tail" && extraOption == arg) {
      const char* text = value(i, arg);
      char* end = nullptr;
      options.tail = std::strtod(text, &end);
      if (end == text || *end != '\0' ||
          !(options.tail > 0.0 && options.tail <= 1.0))
        fail(std::string("--tail takes a fraction in (0, 1], not '") + text +
             "'");
    } else if (!arg.empty() && arg[0] == '-') {
      printUsage(argv[0], extraOption);
      fail("unknown option " + arg);
    } else if (options.deck.empty()) {
      options.deck = arg;
    } else {
      fail("more than one input file: " + options.deck + ", " + arg);
    }
  }
  if (options.deck.empty()) {
    printUsage(argv[0], extraOption);
    fail("no input file");
  }
  return options;
}

/** npcelx >= npcely >= npcelz with product ppc, as close to a cube as
 *  possible (64: 4 4 4, 128: 8 4 4, 256: 8 8 4, 512: 8 8 8). */
std::array<int, 3> npcelFor(int ppc) {
  std::array<int, 3> best = {ppc, 1, 1};
  for (int x = 1; x <= ppc; ++x) {
    if (ppc % x)
      continue;
    for (int y = 1; y <= x; ++y) {
      if ((ppc / x) % y)
        continue;
      const int z = ppc / x / y;
      if (z <= y && x - z < best[0] - best[2])
        best = {x, y, z};
    }
  }
  return best;
}

/** The value of a per-species key of the input file, the same for all
 *  @p speciesCount species. */
std::string everySpecies(int speciesCount, int value) {
  std::ostringstream text;
  for (int s = 0; s < speciesCount; ++s)
    text << (s ? " " : "") << value;
  return text.str();
}

/** Put the particles of every field in one random order. */
void shuffleParticles(ParticleSoAHost& particles) {
  const std::size_t n = particles.getNOP();
  std::vector<std::uint32_t> order(n);
  std::iota(order.begin(), order.end(), 0u);
  std::shuffle(order.begin(), order.end(), std::mt19937_64(kShuffleSeed));
  const auto permute = [&](auto* field) {
    if (field == nullptr)
      return;
    std::vector<std::remove_pointer_t<decltype(field)>> moved(n);
#pragma omp parallel for
    for (std::size_t i = 0; i < n; ++i)
      moved[i] = field[order[i]];
    std::copy(moved.begin(), moved.end(), field);
  };
  permute(particles.getUallMut());
  permute(particles.getVallMut());
  permute(particles.getWallMut());
  permute(particles.getQallMut());
  permute(particles.getXallMut());
  permute(particles.getYallMut());
  permute(particles.getZallMut());
  permute(particles.getIDallMut());
}

/** The sum of the values, compensated (Neumaier): millions of equal charges
 *  added one by one would drift by about count x epsilon. */
template <typename T>
double compensatedSum(const T* values, std::size_t count) {
  double sum = 0.0, correction = 0.0;
  for (std::size_t i = 0; i < count; ++i) {
    const double value = values[i];
    const double next = sum + value;
    correction += std::fabs(sum) >= std::fabs(value) ? (sum - next) + value
                                                     : (value - next) + sum;
    sum = next;
  }
  return sum + correction;
}

} // namespace

void fail(const std::string& message) {
  std::fflush(stdout);
  std::fprintf(stderr, "[profileKernels] %s\n", message.c_str());
  MPIdata::finalize_mpi();
  std::exit(EXIT_FAILURE);
}

std::string deviceName() {
  int device = 0;
  cudaErrChk(cudaGetDevice(&device));
  cudaDeviceProp properties;
  cudaErrChk(cudaGetDeviceProperties(&properties, device));
  return properties.name;
}

std::string jsonString(const std::string& text) {
  std::string quoted = "\"";
  for (const char c : text) {
    if (c == '"' || c == '\\')
      quoted += '\\';
    quoted += static_cast<unsigned char>(c) < 0x20 ? ' ' : c;
  }
  return quoted + '"';
}

// ======= Construction =======

KernelProfile::KernelProfile(int argc, char** argv, const char* kernel,
                             Order order, const char* extraOption,
                             std::size_t extraBytesPerParticle)
    : kernel_(kernel), order_(order) {
  buildHost(argc, argv, extraOption, extraBytesPerParticle);
  buildDevice();
  printSetup();
}

void KernelProfile::buildHost(int argc, char** argv, const char* extraOption,
                              std::size_t extraBytesPerParticle) {
  MPIdata::init(&argc, &argv);
  if (MPIdata::get_nprocs() != 1)
    fail("a kernel profile runs on one process");
  // the device first: a page-locked allocation creates the context
  cudaErrChk(cudaSetDevice(0));
  gpu_ = deviceName();
  profiling::initRecording();
  Parameters::init_parameters();
  options_ = parseOptions(argc, argv, extraOption);

  if (!std::ifstream(options_.deck))
    fail("cannot read the input file " + options_.deck);
  ConfigFile config(options_.deck);
  config.add("XLEN", 1);
  config.add("YLEN", 1);
  config.add("ZLEN", 1);
  for (int a = 0; a < 3; ++a) {
    if (options_.cells[a] == 0)
      continue;
    const double cellSize = config.read<double>(kAxisKeys[a][1]) /
                            config.read<int>(kAxisKeys[a][0]);
    config.add(kAxisKeys[a][0], options_.cells[a]);
    config.add(kAxisKeys[a][1], options_.cells[a] * cellSize);
  }
  if (options_.ppc > 0) {
    const int ns = config.read<int>("ns");
    const std::array<int, 3> npcel = npcelFor(options_.ppc);
    config.add("npcelx", everySpecies(ns, npcel[0]));
    config.add("npcely", everySpecies(ns, npcel[1]));
    config.add("npcelz", everySpecies(ns, npcel[2]));
  }
  col_ = std::make_unique<Collective>(config, options_.deck, false);

  const int s = options_.species;
  if (s < 0 || s >= col_->getNs())
    fail("--species " + std::to_string(s) +
         " is not a species of the input file");
  if (options_.subcycles > 0) {
    // the push takes int((4 / pi) |q/m| dt |B| / c) + 1 subcycles
    // (moverSubcycles): B in the middle of the range that gives subcycles
    const double qom = std::fabs(col_->getQOM(s));
    if (qom == 0.0)
      fail("--subcycles needs a species with q/m != 0");
    const double b = (options_.subcycles - 0.5) * M_PI * col_->getC() /
                     (4.0 * qom * col_->getDt());
    config.add("B0x", 0.0);
    config.add("B0y", 0.0);
    config.add("B0z", b);
    col_ = std::make_unique<Collective>(config, options_.deck, false);
  }
  npcel_ = {col_->getNpcelx(s), col_->getNpcely(s), col_->getNpcelz(s)};

  // the memory of the whole profile, before the particles are made
  const std::uint64_t n = std::uint64_t(col_->getNxc()) * col_->getNyc() *
                          col_->getNzc() * npcel_[0] * npcel_[1] * npcel_[2];
  if (n > std::uint64_t(INT_MAX))
    fail(std::to_string(n) + " particles: ParticleSoAHost holds at most " +
         std::to_string(INT_MAX));
  const std::size_t sortBytes = order_ == Order::CellSorted ? kSortBytes : 0;
  requireDeviceMemory(n * (kParticleBytes + sortBytes + extraBytesPerParticle),
                      "the particles of this profile");
  const std::size_t shuffleBytes =
      order_ == Order::Shuffled ? kShuffleBytes : 0;
  requireHostMemory(n * (kParticleBytes + shuffleBytes),
                    "the host particle arrays");

  vct_ = std::make_unique<VCtopology3D>(*col_);
  vct_->setup_vctopology(MPIdata::get_PicGlobalComm());
  grid_ = std::make_unique<Grid3DCU>(col_.get(), vct_.get());
  cells_ = {grid_->getNXC() - 2, grid_->getNYC() - 2, grid_->getNZC() - 2};
  fields_ = std::make_unique<EMfields3D>(col_.get(), grid_.get(), vct_.get());
  fields_->init();
  particlesHost_ =
      std::make_unique<ParticleSoAHost>(s, col_.get(), vct_.get(), grid_.get());
  particlesHost_->maxwellian(fields_.get());
}

void KernelProfile::buildDevice() {
  cudaErrChk(cudaStreamCreate(&stream_));
  cudaErrChk(cudaEventCreate(&start_));
  cudaErrChk(cudaEventCreate(&stop_));

  // the device arrays are filled from the host ones
  if (order_ == Order::Shuffled)
    shuffleParticles(*particlesHost_);
  const std::size_t n = particlesHost_->getNOP();
  pclsArrayHost_ = newHostPinnedObject<particleArrayCUDA>(
      particlesHost_.get(), kParticleCapacityFactor, stream_);
  pclsArrayDevice_ = pclsArrayHost_->copyToDevice();
  gridCUDA_ = grid3DCUDA(grid_.get());

  if (order_ == Order::CellSorted) {
    cellSorter_.init(gridCUDA_, pclsArrayHost_->getCapacity(), stream_);
    cellSorter_.sort(pclsArrayHost_, gridCUDA_, n, stream_);
    // the sort swapped the SoA pointers on the host (sortSpeciesByCell)
    cudaErrChk(cudaMemcpyAsync(pclsArrayDevice_, pclsArrayHost_,
                               sizeof(particleArrayCUDA), cudaMemcpyDefault,
                               stream_));
  }
  cudaErrChk(cudaStreamSynchronize(stream_));
}

KernelProfile::~KernelProfile() {
  if (stream_)
    cudaErrChk(cudaStreamSynchronize(stream_));
  cellSorter_.free();
  cudaFree(moments_);
  if (hashedSumArrayHost_) {
    deleteHostPinnedObjectArray(hashedSumArrayHost_,
                                departureArrayElementType::HASHED_SUM_NUM);
    cudaFree(hashedSumArrayDevice_);
  }
  if (departureArrayHost_) {
    deleteHostPinnedObject(departureArrayHost_);
    cudaFree(departureArrayDevice_);
  }
  if (pclsArrayHost_) {
    deleteHostPinnedObject(pclsArrayHost_);
    cudaFree(pclsArrayDevice_);
  }
  if (start_)
    cudaEventDestroy(start_);
  if (stop_)
    cudaEventDestroy(stop_);
  if (stream_)
    cudaStreamDestroy(stream_);
  // the host objects use MPI (datatypes, communicators) up to their end
  particlesHost_.reset();
  fields_.reset();
  grid_.reset();
  vct_.reset();
  col_.reset();
  MPIdata::finalize_mpi();
}

void KernelProfile::printSetup() const {
  std::printf("%s on %s\n", kernel_.c_str(), gpu_.c_str());
  std::printf("  cells %d x %d x %d, %d particles per cell (%d x %d x %d), "
              "species %d (q/m %g), %u particles, %s\n",
              cells_[0], cells_[1], cells_[2],
              npcel_[0] * npcel_[1] * npcel_[2], npcel_[0], npcel_[1],
              npcel_[2], options_.species, col_->getQOM(options_.species),
              particleCount(),
              order_ == Order::CellSorted ? "sorted by cell (CellSorter)"
                                          : "in random order");
}

std::size_t KernelProfile::nodeCount() const {
  return std::size_t(grid_->getNXN()) * grid_->getNYN() * grid_->getNZN();
}

// ======= Buffers of the solver's particle step =======

departureArrayType* KernelProfile::departureArrayDevice() {
  if (departureArrayDevice_ == nullptr) {
    const std::size_t size = pclsArrayHost_->getSize();
    departureArrayHost_ = newHostPinnedObject<departureArrayType>(size);
    departureArrayDevice_ = departureArrayHost_->copyToDevice();
    cudaErrChk(cudaMemsetAsync(departureArrayHost_->getArray(), 0,
                               size * sizeof(departureArrayElementType),
                               stream_));
  }
  return departureArrayDevice_;
}

hashedSum* KernelProfile::hashedSumArrayHost() {
  if (hashedSumArrayHost_ == nullptr) {
    hashedSumArrayHost_ = newHostPinnedObjectArray<hashedSum>(
        departureArrayElementType::HASHED_SUM_NUM, kHashedSumBuckets);
    hashedSumArrayDevice_ =
        copyArrayToDevice(hashedSumArrayHost_,
                          departureArrayElementType::HASHED_SUM_NUM, stream_);
  }
  return hashedSumArrayHost_;
}

hashedSum* KernelProfile::hashedSumArrayDevice() {
  hashedSumArrayHost();
  return hashedSumArrayDevice_;
}

void KernelProfile::resetDepartures() {
  departureArrayDevice();
  hashedSum* sums = hashedSumArrayHost();
  for (int j = 0; j < departureArrayElementType::HASHED_SUM_NUM; j++)
    sums[j].resetBucket();
  cudaErrChk(cudaMemcpyAsync(hashedSumArrayDevice_, sums,
                             departureArrayElementType::HASHED_SUM_NUM *
                                 sizeof(hashedSum),
                             cudaMemcpyDefault, stream_));
  cudaErrChk(cudaMemsetAsync(departureArrayHost_->getArray(), 0,
                             departureArrayHost_->getSize() *
                                 sizeof(departureArrayElementType),
                             stream_));
}

cudaMomentType* KernelProfile::moments() {
  if (moments_ == nullptr) {
    const std::size_t bytes =
        nodeCount() * kMomentCount * sizeof(cudaMomentType);
    requireDeviceMemory(bytes, "the moments");
    cudaErrChk(cudaMalloc(&moments_, bytes));
  }
  return moments_;
}

void KernelProfile::clearMoments() {
  cudaErrChk(cudaMemsetAsync(
      moments(), 0, nodeCount() * kMomentCount * sizeof(cudaMomentType),
      stream_));
}

bool KernelProfile::checkDepositedCharge(std::uint32_t first,
                                         std::uint32_t count) {
  // the host arrays are in the device order unless the device sorted them
  if (order_ == Order::CellSorted && (first != 0 || count != particleCount()))
    fail("checkDepositedCharge: the sorted particles are checked as a whole");
  const std::size_t nodes = nodeCount();
  PinnedVector<cudaMomentType> rho(nodes); // moment 0, the charge density
  cudaErrChk(cudaMemcpyAsync(rho.data(), moments(),
                             nodes * sizeof(cudaMomentType), cudaMemcpyDefault,
                             stream_));
  cudaErrChk(cudaStreamSynchronize(stream_));
  const double deposited = compensatedSum(rho.data(), nodes) * grid_->getVOL();
  const double expected =
      compensatedSum(particlesHost_->getQall() + first, count);
  const double error = std::fabs(deposited - expected) / std::fabs(expected);
  const bool pass = error <= kChargeTolerance;
  std::printf("  charge check: deposited %.12e, particles %.12e, relative "
              "error %.1e: %s\n",
              deposited, expected, error, pass ? "pass" : "FAIL");
  return pass;
}

void KernelProfile::requireDeviceMemory(std::size_t bytes,
                                        const char* what) const {
  std::size_t freeBytes = 0, totalBytes = 0;
  cudaErrChk(cudaMemGetInfo(&freeBytes, &totalBytes));
  if (bytes + kMemoryMarginBytes > freeBytes) {
    std::ostringstream message;
    message << what << " need " << bytes / kMiB << " MiB, the device has "
            << freeBytes / kMiB << " MiB free (keeping "
            << kMemoryMarginBytes / kMiB
            << " MiB): use fewer cells or particles per cell";
    fail(message.str());
  }
}

void KernelProfile::requireHostMemory(std::size_t bytes,
                                      const char* what) const {
  std::ifstream meminfo("/proc/meminfo");
  std::string key;
  std::size_t kib = 0;
  std::string unit;
  while (meminfo >> key >> kib >> unit) {
    if (key != "MemAvailable:")
      continue;
    const std::size_t available = kib * 1024;
    if (bytes > available) {
      std::ostringstream message;
      message << what << " need " << bytes / kMiB << " MiB, the host has "
              << available / kMiB
              << " MiB available: use fewer cells or particles per cell";
      fail(message.str());
    }
    return;
  }
}

// ======= Timing =======

void KernelProfile::time(const std::string& parameter, std::uint32_t particles,
                         profiling::Region region,
                         const std::function<void()>& prepare,
                         const std::function<void()>& launch) {
  std::size_t freeBytes = 0, totalBytes = 0;
  cudaErrChk(cudaMemGetInfo(&freeBytes, &totalBytes));
  std::printf("  device memory in use (all processes): %.0f of %.0f MiB\n",
              (totalBytes - freeBytes) / kMiB, totalBytes / kMiB);
  for (int i = 0; i < options_.warmup; ++i) {
    prepare();
    launch();
    cudaErrChk(cudaGetLastError());
  }
  cudaErrChk(cudaStreamSynchronize(stream_));

  std::vector<double> ms;
  {
    profiling::Range range(region, options_.species);
    profiling::recordSteadyState(true);
    for (int i = 0; i < options_.launches; ++i) {
      prepare();
      cudaErrChk(cudaEventRecord(start_, stream_));
      launch();
      cudaErrChk(cudaGetLastError());
      cudaErrChk(cudaEventRecord(stop_, stream_));
      cudaErrChk(cudaEventSynchronize(stop_));
      float elapsed = 0.0f;
      cudaErrChk(cudaEventElapsedTime(&elapsed, start_, stop_));
      ms.push_back(elapsed);
    }
    profiling::recordSteadyState(false);
  }

  std::sort(ms.begin(), ms.end());
  const std::size_t n = ms.size();
  times_.min = ms.front();
  times_.max = ms.back();
  times_.median = n % 2 ? ms[n / 2] : 0.5 * (ms[n / 2 - 1] + ms[n / 2]);
  times_.mean = std::accumulate(ms.begin(), ms.end(), 0.0) / n;
  parameter_ = parameter;
  launchParticles_ = particles;

  std::printf("  %s, %u particles per launch\n", parameter.c_str(), particles);
  std::printf("  %d timed launches: min %.4f ms, median %.4f ms, mean %.4f "
              "ms, max %.4f ms; %.4f ns per particle (median)\n",
              options_.launches, times_.min, times_.median, times_.mean,
              times_.max, nsPerParticle());
}

double KernelProfile::nsPerParticle() const {
  return times_.median * kNsPerMs / launchParticles_;
}

int KernelProfile::finish(bool checked) {
  std::ostringstream json;
  json << std::setprecision(10) << "{\"kernel\":" << jsonString(kernel_)
       << ",\"gpu\":" << jsonString(gpu_)
       << ",\"build\":" << jsonString(iPic3D::build::signature())
       << ",\"cells\":[" << cells_[0] << ',' << cells_[1] << ',' << cells_[2]
       << "],\"ppc\":" << npcel_[0] * npcel_[1] * npcel_[2] << ",\"npcel\":["
       << npcel_[0] << ',' << npcel_[1] << ',' << npcel_[2]
       << "],\"species\":" << options_.species << ",\"order\":"
       << (order_ == Order::CellSorted ? "\"cell-sorted\"" : "\"random\"")
       << ",\"particles\":" << particleCount()
       << ",\"launch_particles\":" << launchParticles_
       << ",\"parameter\":" << jsonString(parameter_)
       << ",\"warmup\":" << options_.warmup
       << ",\"launches\":" << options_.launches << ",\"min_ms\":" << times_.min
       << ",\"median_ms\":" << times_.median << ",\"mean_ms\":" << times_.mean
       << ",\"max_ms\":" << times_.max
       << ",\"ns_per_particle\":" << nsPerParticle()
       << ",\"check\":" << (checked ? "true" : "false") << '}';
  std::printf("%s\n", json.str().c_str());
  return checked ? EXIT_SUCCESS : EXIT_FAILURE;
}

int moverSubcycles(const Collective& col, int species) {
  const double b =
      std::sqrt(col.getB0x() * col.getB0x() + col.getB0y() * col.getB0y() +
                col.getB0z() * col.getB0z());
  return int(4.0 / M_PI * std::fabs(col.getQOM(species)) * col.getDt() * b /
             col.getC()) +
         1;
}

} // namespace profileKernels
