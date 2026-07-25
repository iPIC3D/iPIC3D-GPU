/**
 * Exact-value tests for the CPU and batched-GPU halo implementations.
 *
 * Examples:
 *   mpirun -n 8 ./haloCommunicationTest --case all --suite distributed
 *   mpirun -n 1 ./haloCommunicationTest --case all --suite self
 *
 * The conventional test checks every semantically defined array entry,
 * including faces, edges, corners, and MPI_PROC_NULL face/edge ghosts.  The
 * eight corners outside a nonperiodic global domain are intentionally ignored:
 * the legacy CPU routine documents those as boundary-condition workspace and
 * may overwrite them before a caller applies its boundary conditions.  The
 * additive test builds an independent global-node sum with MPI_Allreduce and
 * checks every physical node.  Integer-valued inputs keep all expected sums
 * exactly representable in double precision.
 */

#include <mpi.h>

#include "CUDA/GPUFieldArray.cuh"
#include "Collective.h"
#include "Com3DNonblk.h"
#include "ConfigFile.h"
#include "EMfields3D.h"
#include "Grid3DCU.h"
#include "MPIdata.h"
#include "Parameters.h"
#include "VCtopology3D.h"
#include "cudaTypeDef.cuh"

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr int kFieldCount = 3;
constexpr int kSkipReturnCode = 77;
constexpr const char* kZeros12 = "0 0 0 0 0 0 0 0 0 0 0 0";
constexpr const char* kOnes12 = "1 1 1 1 1 1 1 1 1 1 1 1";

constexpr std::array<const char*, 8> kZeroSpeciesKeys = {
    "rhoINIT", "rhoINJECT", "uth", "vth", "wth", "u0", "v0", "w0"};
constexpr std::array<const char*, 3> kUnitParticleKeys = {"npcelx", "npcely",
                                                          "npcelz"};
constexpr std::array<const char*, 6> kPhiBoundaryKeys = {
    "bcPHIfaceXright", "bcPHIfaceXleft",  "bcPHIfaceYright",
    "bcPHIfaceYleft",  "bcPHIfaceZright", "bcPHIfaceZleft"};
constexpr std::array<const char*, 6> kEmBoundaryKeys = {
    "bcEMfaceXright", "bcEMfaceXleft",  "bcEMfaceYright",
    "bcEMfaceYleft",  "bcEMfaceZright", "bcEMfaceZleft"};
constexpr std::array<const char*, 6> kParticleBoundaryKeys = {
    "bcPfaceXright", "bcPfaceXleft",  "bcPfaceYright",
    "bcPfaceYleft",  "bcPfaceZright", "bcPfaceZleft"};
constexpr std::array<const char*, 4> kDisabledOutputCycleKeys = {
    "FieldOutputCycle", "ParticlesOutputCycle", "RestartOutputCycle",
    "DiagnosticsOutputCycle"};

enum class ExchangeCase { Conventional, Additive, All };
enum class Suite { Distributed, Self };
enum class Backend { Cpu, Gpu, Both };

struct Options {
  ExchangeCase exchangeCase = ExchangeCase::All;
  Suite suite = Suite::Distributed;
  Backend backend = Backend::Both;
  bool help = false;
};

struct Scenario {
  const char* name;
  std::array<int, 3> dims;
  std::array<int, 3> periodic;
  std::array<int, 3> localCells;
};

constexpr std::array<Scenario, 3> kDistributedScenarios = {{
    {"remote-periodic-2x2x2", {2, 2, 2}, {1, 1, 1}, {4, 5, 6}},
    {"remote-nonperiodic-2x2x2", {2, 2, 2}, {0, 0, 0}, {4, 5, 6}},
    {"mixed-remote-null-self-4x2x1", {4, 2, 1}, {1, 0, 1}, {4, 5, 6}},
}};

constexpr std::array<Scenario, 1> kSelfScenarios = {{
    {"self-periodic-1x1x1", {1, 1, 1}, {1, 1, 1}, {4, 5, 6}},
}};

template <size_t N, typename Value>
void addConfigValues(ConfigFile& cfg, const std::array<const char*, N>& keys,
                     const Value& value) {
  for (const char* key : keys)
    cfg.add(key, value);
}

ConfigFile makeConfig(const Scenario& scenario) {
  const int nxc = scenario.localCells[0] * scenario.dims[0];
  const int nyc = scenario.localCells[1] * scenario.dims[1];
  const int nzc = scenario.localCells[2] * scenario.dims[2];

  ConfigFile cfg;
  cfg.add("SaveDirName", ".");
  cfg.add("RestartDirName", ".");
  cfg.add("Case", "Default");
  cfg.add("PoissonCorrection", "no");
  cfg.add("divBCorrection", "no");
  cfg.add("WriteMethod", "default");
  cfg.add("SimulationName", "halo_communication");

  cfg.add("B0x", 0.0);
  cfg.add("B0y", 0.0);
  cfg.add("B0z", 0.0);
  cfg.add("delta", 0.5);
  cfg.add("dt", 0.05);
  cfg.add("ncycles", 1);
  cfg.add("th", 1.0);
  cfg.add("c", 1.0);
  cfg.add("Smooth", 1.0);
  cfg.add("SmoothNiter", 1);

  cfg.add("Lx", static_cast<double>(nxc));
  cfg.add("Ly", static_cast<double>(nyc));
  cfg.add("Lz", static_cast<double>(nzc));
  cfg.add("nxc", nxc);
  cfg.add("nyc", nyc);
  cfg.add("nzc", nzc);
  cfg.add("XLEN", scenario.dims[0]);
  cfg.add("YLEN", scenario.dims[1]);
  cfg.add("ZLEN", scenario.dims[2]);
  cfg.add("PERIODICX", scenario.periodic[0]);
  cfg.add("PERIODICY", scenario.periodic[1]);
  cfg.add("PERIODICZ", scenario.periodic[2]);
  cfg.add("PERIODICX_P", scenario.periodic[0]);
  cfg.add("PERIODICY_P", scenario.periodic[1]);
  cfg.add("PERIODICZ_P", scenario.periodic[2]);

  cfg.add("NiterMover", 1);
  cfg.add("ns", 1);
  cfg.add("qom", "-1 0 0 0 0 0 0 0 0 0 0 0");
  addConfigValues(cfg, kZeroSpeciesKeys, kZeros12);
  addConfigValues(cfg, kUnitParticleKeys, kOnes12);
  addConfigValues(cfg, kPhiBoundaryKeys, 1);
  addConfigValues(cfg, kEmBoundaryKeys, 0);
  addConfigValues(cfg, kParticleBoundaryKeys, 1);
  cfg.add("ApplyInflowBcsEImage", 0);

  cfg.add("CGtol", 1.0e-11);
  cfg.add("GMREStol", 1.0e-11);
  cfg.add("SolverType", "GMRES");
  addConfigValues(cfg, kDisabledOutputCycleKeys, 0);
  cfg.add("CallFinalize", 0);
  return cfg;
}

const char* caseName(ExchangeCase exchangeCase) {
  switch (exchangeCase) {
  case ExchangeCase::Conventional:
    return "conventional";
  case ExchangeCase::Additive:
    return "additive";
  case ExchangeCase::All:
    return "all";
  }
  return "unknown";
}

const char* backendName(Backend backend) {
  switch (backend) {
  case Backend::Cpu:
    return "cpu";
  case Backend::Gpu:
    return "gpu";
  case Backend::Both:
    return "both";
  }
  return "unknown";
}

void printUsage() {
  std::cout
      << "Usage: haloCommunicationTest [--case conventional|additive|all] "
         "[--suite distributed|self] [--backend cpu|gpu|both]\n";
}

Options parseOptions(int argc, char** argv) {
  Options options;
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--help" || arg == "-h") {
      options.help = true;
      continue;
    }
    if (arg != "--case" && arg != "--suite" && arg != "--backend") {
      throw std::runtime_error("unknown argument '" + arg + "'");
    }
    if (++i >= argc)
      throw std::runtime_error("missing value for " + arg);
    const std::string value = argv[i];
    if (arg == "--case") {
      if (value == "conventional")
        options.exchangeCase = ExchangeCase::Conventional;
      else if (value == "additive")
        options.exchangeCase = ExchangeCase::Additive;
      else if (value == "all")
        options.exchangeCase = ExchangeCase::All;
      else
        throw std::runtime_error("unknown --case value '" + value + "'");
    } else if (arg == "--suite") {
      if (value == "distributed")
        options.suite = Suite::Distributed;
      else if (value == "self")
        options.suite = Suite::Self;
      else
        throw std::runtime_error("unknown --suite value '" + value + "'");
    } else {
      if (value == "cpu")
        options.backend = Backend::Cpu;
      else if (value == "gpu")
        options.backend = Backend::Gpu;
      else if (value == "both")
        options.backend = Backend::Both;
      else
        throw std::runtime_error("unknown --backend value '" + value + "'");
    }
  }
  return options;
}

bool runsCpu(Backend backend) {
  return backend == Backend::Cpu || backend == Backend::Both;
}

bool runsGpu(Backend backend) {
  return backend == Backend::Gpu || backend == Backend::Both;
}

bool runsCase(ExchangeCase selection, ExchangeCase candidate) {
  return selection == ExchangeCase::All || selection == candidate;
}

class HostFields {
public:
  HostFields(int nx, int ny, int nz)
      : field0_(nx, ny, nz), field1_(nx, ny, nz), field2_(nx, ny, nz) {}

  arr3_double& operator[](int field) {
    if (field == 0)
      return field0_;
    if (field == 1)
      return field1_;
    return field2_;
  }

private:
  arr3_double field0_;
  arr3_double field1_;
  arr3_double field2_;
};

class DeviceFields {
public:
  DeviceFields(int nx, int ny, int nz)
      : field0_(nx, ny, nz), field1_(nx, ny, nz), field2_(nx, ny, nz) {}

  GPUFieldArray3& operator[](int field) {
    if (field == 0)
      return field0_;
    if (field == 1)
      return field1_;
    return field2_;
  }

  std::array<cudaSolverType*, kFieldCount> pointers() {
    return {field0_.devPtr(), field1_.devPtr(), field2_.devPtr()};
  }

private:
  GPUFieldArray3 field0_;
  GPUFieldArray3 field1_;
  GPUFieldArray3 field2_;
};

class StreamGuard {
public:
  StreamGuard() {
    cudaErrChk(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking));
  }
  ~StreamGuard() {
    if (stream_ != nullptr)
      cudaStreamDestroy(stream_);
  }
  cudaStream_t get() const { return stream_; }

private:
  cudaStream_t stream_ = nullptr;
};

class HaloBufferGuard {
public:
  explicit HaloBufferGuard(EMfields3D& fields) : fields_(fields) {
    fields_.gpuAllocateHaloBuffers();
  }
  ~HaloBufferGuard() { fields_.gpuFreeHaloBuffers(); }

private:
  EMfields3D& fields_;
};

struct Checker {
  int rank;
  const char* scenario;
  const char* exchange;
  const char* backend;
  int failures = 0;

  void exact(double actual, double expected, int field, int i, int j, int k) {
    if (std::isfinite(actual) && actual == expected)
      return;
    ++failures;
    if (failures <= 12) {
      std::cerr << "[rank " << rank << "] " << scenario << " " << exchange
                << " " << backend << " field " << field << " (" << i << "," << j
                << "," << k << ") = " << std::setprecision(17) << actual
                << ", expected " << expected << "\n";
    }
  }

  void fail(const std::string& message) {
    ++failures;
    if (failures <= 12) {
      std::cerr << "[rank " << rank << "] " << scenario << " " << exchange
                << " " << backend << ": " << message << "\n";
    }
  }
};

int positiveModulo(int value, int modulus) {
  const int remainder = value % modulus;
  return remainder < 0 ? remainder + modulus : remainder;
}

int canonicalCoordinate(int coordinate, int globalCells, bool periodic) {
  return periodic ? positiveModulo(coordinate, globalCells) : coordinate;
}

double ghostSentinel(int field, int i, int j, int k) {
  return -4000000000000.0 - field * 100000000.0 - i * 10000.0 - j * 100.0 - k;
}

double conventionalValue(int field, int gx, int gy, int gz) {
  return 1.0 + field * 1000000000.0 + gx * 1000000.0 + gy * 1000.0 + gz;
}

double localContribution(int field, int rank, int i, int j, int k) {
  return 1.0 + field * 10000000.0 + rank * 100000.0 + i * 1000.0 + j * 10.0 + k;
}

bool isPhysical(int index, int extent) {
  return index >= 1 && index <= extent - 2;
}

int rawGlobalCoordinate(int processCoordinate, int localCells, int index) {
  return processCoordinate * localCells + index - 1;
}

bool ghostHasNeighbor(int index, int extent, int processCoordinate,
                      int processCount, bool periodic) {
  if (index == 0)
    return periodic || processCoordinate > 0;
  if (index == extent - 1) {
    return periodic || processCoordinate + 1 < processCount;
  }
  return true;
}

void initializeConventional(HostFields& fields, const Scenario& scenario,
                            const VCtopology3D& vct, int nx, int ny, int nz) {
  for (int field = 0; field < kFieldCount; ++field) {
    for (int i = 0; i < nx; ++i) {
      for (int j = 0; j < ny; ++j) {
        for (int k = 0; k < nz; ++k) {
          if (!isPhysical(i, nx) || !isPhysical(j, ny) || !isPhysical(k, nz)) {
            fields[field].fetch(i, j, k) = ghostSentinel(field, i, j, k);
            continue;
          }
          const std::array<int, 3> index = {i, j, k};
          std::array<int, 3> global{};
          for (int axis = 0; axis < 3; ++axis) {
            const int raw =
                rawGlobalCoordinate(vct.getCoordinates(axis),
                                    scenario.localCells[axis], index[axis]);
            const int globalCells =
                scenario.localCells[axis] * scenario.dims[axis];
            global[axis] = canonicalCoordinate(raw, globalCells,
                                               scenario.periodic[axis] != 0);
          }
          fields[field].fetch(i, j, k) =
              conventionalValue(field, global[0], global[1], global[2]);
        }
      }
    }
  }
}

bool conventionalOutputIsDefined(const Scenario& scenario,
                                 const VCtopology3D& vct, int i, int j, int k,
                                 int nx, int ny, int nz) {
  const std::array<int, 3> index = {i, j, k};
  const std::array<int, 3> extent = {nx, ny, nz};
  int ghostAxes = 0;
  bool hasMissingNeighbor = false;
  for (int axis = 0; axis < 3; ++axis) {
    if (isPhysical(index[axis], extent[axis]))
      continue;
    ++ghostAxes;
    hasMissingNeighbor =
        hasMissingNeighbor ||
        !ghostHasNeighbor(index[axis], extent[axis], vct.getCoordinates(axis),
                          scenario.dims[axis], scenario.periodic[axis] != 0);
  }
  return ghostAxes != 3 || !hasMissingNeighbor;
}

double expectedConventional(const Scenario& scenario, const VCtopology3D& vct,
                            int field, int i, int j, int k, int nx, int ny,
                            int nz) {
  const std::array<int, 3> index = {i, j, k};
  const std::array<int, 3> extent = {nx, ny, nz};

  for (int axis = 0; axis < 3; ++axis) {
    if (!ghostHasNeighbor(index[axis], extent[axis], vct.getCoordinates(axis),
                          scenario.dims[axis], scenario.periodic[axis] != 0)) {
      return ghostSentinel(field, i, j, k);
    }
  }

  std::array<int, 3> global{};
  for (int axis = 0; axis < 3; ++axis) {
    const int raw = rawGlobalCoordinate(vct.getCoordinates(axis),
                                        scenario.localCells[axis], index[axis]);
    const int globalCells = scenario.localCells[axis] * scenario.dims[axis];
    global[axis] =
        canonicalCoordinate(raw, globalCells, scenario.periodic[axis] != 0);
  }
  return conventionalValue(field, global[0], global[1], global[2]);
}

void initializeAdditive(HostFields& fields, int rank, int nx, int ny, int nz) {
  for (int field = 0; field < kFieldCount; ++field) {
    for (int i = 0; i < nx; ++i) {
      for (int j = 0; j < ny; ++j) {
        for (int k = 0; k < nz; ++k) {
          fields[field].fetch(i, j, k) =
              isPhysical(i, nx) && isPhysical(j, ny) && isPhysical(k, nz)
                  ? localContribution(field, rank, i, j, k)
                  : ghostSentinel(field, i, j, k);
        }
      }
    }
  }
}

size_t globalNodeIndex(const std::array<int, 3>& node,
                       const std::array<int, 3>& nodeCounts) {
  return (static_cast<size_t>(node[0]) * nodeCounts[1] + node[1]) *
             nodeCounts[2] +
         node[2];
}

std::array<std::vector<double>, kFieldCount>
buildAdditiveOracle(const Scenario& scenario, const VCtopology3D& vct, int rank,
                    int nx, int ny, int nz) {
  std::array<int, 3> globalCells{};
  std::array<int, 3> nodeCounts{};
  for (int axis = 0; axis < 3; ++axis) {
    globalCells[axis] = scenario.localCells[axis] * scenario.dims[axis];
    nodeCounts[axis] = globalCells[axis] + (scenario.periodic[axis] ? 0 : 1);
  }
  const size_t totalNodes =
      static_cast<size_t>(nodeCounts[0]) * nodeCounts[1] * nodeCounts[2];
  std::array<std::vector<double>, kFieldCount> sums;
  for (auto& sum : sums)
    sum.assign(totalNodes, 0.0);

  for (int i = 1; i <= nx - 2; ++i) {
    for (int j = 1; j <= ny - 2; ++j) {
      for (int k = 1; k <= nz - 2; ++k) {
        const std::array<int, 3> index = {i, j, k};
        std::array<int, 3> node{};
        for (int axis = 0; axis < 3; ++axis) {
          const int raw = rawGlobalCoordinate(
              vct.getCoordinates(axis), scenario.localCells[axis], index[axis]);
          node[axis] = canonicalCoordinate(raw, globalCells[axis],
                                           scenario.periodic[axis] != 0);
        }
        const size_t flat = globalNodeIndex(node, nodeCounts);
        for (int field = 0; field < kFieldCount; ++field) {
          sums[field][flat] += localContribution(field, rank, i, j, k);
        }
      }
    }
  }

  for (auto& sum : sums) {
    MPI_Allreduce(MPI_IN_PLACE, sum.data(), static_cast<int>(sum.size()),
                  MPI_DOUBLE, MPI_SUM, vct.getParticleComm());
  }
  return sums;
}

int checkConventional(HostFields& fields, const Scenario& scenario,
                      const VCtopology3D& vct, int rank, const char* backend,
                      int nx, int ny, int nz) {
  Checker check{rank, scenario.name, "conventional", backend};
  for (int field = 0; field < kFieldCount; ++field) {
    for (int i = 0; i < nx; ++i) {
      for (int j = 0; j < ny; ++j) {
        for (int k = 0; k < nz; ++k) {
          if (!conventionalOutputIsDefined(scenario, vct, i, j, k, nx, ny,
                                           nz)) {
            continue;
          }
          check.exact(
              fields[field].get(i, j, k),
              expectedConventional(scenario, vct, field, i, j, k, nx, ny, nz),
              field, i, j, k);
        }
      }
    }
  }
  return check.failures;
}

int checkAdditive(HostFields& fields, const Scenario& scenario,
                  const VCtopology3D& vct, int rank, const char* backend,
                  int nx, int ny, int nz) {
  Checker check{rank, scenario.name, "additive", backend};
  const auto sums = buildAdditiveOracle(scenario, vct, rank, nx, ny, nz);

  std::array<int, 3> globalCells{};
  std::array<int, 3> nodeCounts{};
  for (int axis = 0; axis < 3; ++axis) {
    globalCells[axis] = scenario.localCells[axis] * scenario.dims[axis];
    nodeCounts[axis] = globalCells[axis] + (scenario.periodic[axis] ? 0 : 1);
  }

  for (int i = 1; i <= nx - 2; ++i) {
    for (int j = 1; j <= ny - 2; ++j) {
      for (int k = 1; k <= nz - 2; ++k) {
        const std::array<int, 3> index = {i, j, k};
        std::array<int, 3> node{};
        for (int axis = 0; axis < 3; ++axis) {
          const int raw = rawGlobalCoordinate(
              vct.getCoordinates(axis), scenario.localCells[axis], index[axis]);
          node[axis] = canonicalCoordinate(raw, globalCells[axis],
                                           scenario.periodic[axis] != 0);
        }
        const size_t flat = globalNodeIndex(node, nodeCounts);
        for (int field = 0; field < kFieldCount; ++field) {
          check.exact(fields[field].get(i, j, k), sums[field][flat], field, i,
                      j, k);
        }
      }
    }
  }
  return check.failures;
}

int runCpu(ExchangeCase exchangeCase, const Scenario& scenario,
           const VCtopology3D& vct, EMfields3D& emFields, int rank, int nx,
           int ny, int nz) {
  HostFields fields(nx, ny, nz);
  if (exchangeCase == ExchangeCase::Conventional) {
    initializeConventional(fields, scenario, vct, nx, ny, nz);
    for (int field = 0; field < kFieldCount; ++field) {
      communicateNode_P(nx, ny, nz, fields[field].fetch_arr3(), &vct,
                        &emFields);
    }
    return checkConventional(fields, scenario, vct, rank, "cpu", nx, ny, nz);
  }

  initializeAdditive(fields, rank, nx, ny, nz);
  for (int field = 0; field < kFieldCount; ++field) {
    communicateInterp(nx, ny, nz, fields[field].fetch_arr3(), &vct, &emFields);
  }
  return checkAdditive(fields, scenario, vct, rank, "cpu", nx, ny, nz);
}

int runGpu(ExchangeCase exchangeCase, const Scenario& scenario,
           const VCtopology3D& vct, EMfields3D& emFields, int rank, int nx,
           int ny, int nz) {
  HostFields host(nx, ny, nz);
  if (exchangeCase == ExchangeCase::Conventional) {
    initializeConventional(host, scenario, vct, nx, ny, nz);
  } else {
    initializeAdditive(host, rank, nx, ny, nz);
  }

  StreamGuard stream;
  DeviceFields device(nx, ny, nz);
  // GPUFieldArray3 initializes with cudaMemset on the legacy default stream.
  // A cudaStreamNonBlocking stream has no implicit ordering with that stream,
  // so finish constructor initialization before starting the test H2D copies.
  cudaErrChk(cudaDeviceSynchronize());
  for (int field = 0; field < kFieldCount; ++field) {
    device[field].copyFromHostAsync(host[field].fetch_arr(), stream.get());
  }
  auto pointers = device.pointers();
  HaloBufferGuard haloBuffers(emFields);
  emFields.gpuBatchedHaloExchange(pointers.data(), kFieldCount, nx, ny, nz,
                                  exchangeCase == ExchangeCase::Additive, false,
                                  exchangeCase == ExchangeCase::Additive, true,
                                  stream.get());
  cudaErrChk(cudaGetLastError());
  cudaErrChk(cudaStreamSynchronize(stream.get()));

  for (int field = 0; field < kFieldCount; ++field) {
    device[field].copyToHost(host[field].fetch_arr());
  }
  if (exchangeCase == ExchangeCase::Conventional) {
    return checkConventional(host, scenario, vct, rank, "gpu", nx, ny, nz);
  }
  return checkAdditive(host, scenario, vct, rank, "gpu", nx, ny, nz);
}

int runScenario(ExchangeCase exchangeCase, Backend backend,
                const Scenario& scenario, int rank) {
  MPI_Comm fieldComm = MPI_COMM_NULL;
  MPI_Comm particleComm = MPI_COMM_NULL;
  int globalFailures = 0;

  {
    ConfigFile config = makeConfig(scenario);
    Collective collective(config, "haloCommunicationTest");
    VCtopology3D vct(collective);
    vct.setup_vctopology(MPIdata::get_PicGlobalComm());
    fieldComm = vct.getFieldComm();
    particleComm = vct.getParticleComm();

    Grid3DCU grid(&collective, &vct);
    EMfields3D emFields(&collective, &grid, &vct);
    const int nx = grid.getNXN();
    const int ny = grid.getNYN();
    const int nz = grid.getNZN();

    int localFailures = 0;
    if (nx != scenario.localCells[0] + 3 || ny != scenario.localCells[1] + 3 ||
        nz != scenario.localCells[2] + 3) {
      Checker check{rank, scenario.name, caseName(exchangeCase),
                    backendName(backend)};
      check.fail("unexpected local node-grid dimensions");
      localFailures += check.failures;
    } else {
      if (runsCpu(backend)) {
        localFailures +=
            runCpu(exchangeCase, scenario, vct, emFields, rank, nx, ny, nz);
      }
      if (runsGpu(backend)) {
        localFailures +=
            runGpu(exchangeCase, scenario, vct, emFields, rank, nx, ny, nz);
      }
    }

    MPI_Allreduce(&localFailures, &globalFailures, 1, MPI_INT, MPI_SUM,
                  MPIdata::get_PicGlobalComm());
  }

  if (fieldComm != MPI_COMM_NULL)
    MPI_Comm_free(&fieldComm);
  if (particleComm != MPI_COMM_NULL)
    MPI_Comm_free(&particleComm);

  if (rank == 0) {
    if (globalFailures == 0) {
      std::cout << "haloCommunicationTest " << scenario.name << " "
                << caseName(exchangeCase) << " " << backendName(backend)
                << " PASS\n";
    } else {
      std::cerr << "haloCommunicationTest " << scenario.name << " "
                << caseName(exchangeCase) << " " << backendName(backend)
                << " FAIL: " << globalFailures << " mismatches\n";
    }
  }
  return globalFailures;
}

bool selectDeviceForRank() {
  MPI_Comm sharedComm = MPI_COMM_NULL;
  MPI_Comm_split_type(MPIdata::get_PicGlobalComm(), MPI_COMM_TYPE_SHARED, 0,
                      MPI_INFO_NULL, &sharedComm);
  int sharedRank = 0;
  MPI_Comm_rank(sharedComm, &sharedRank);

  int deviceCount = 0;
  const cudaError_t countError = cudaGetDeviceCount(&deviceCount);
  bool ready = countError == cudaSuccess && deviceCount > 0;
  if (ready)
    ready = cudaSetDevice(sharedRank % deviceCount) == cudaSuccess;
  MPI_Comm_free(&sharedComm);
  return ready;
}

template <size_t N>
int runScenarios(const std::array<Scenario, N>& scenarios,
                 const Options& options, int rank) {
  int failures = 0;
  for (const Scenario& scenario : scenarios) {
    if (runsCase(options.exchangeCase, ExchangeCase::Conventional)) {
      failures += runScenario(ExchangeCase::Conventional, options.backend,
                              scenario, rank);
    }
    if (runsCase(options.exchangeCase, ExchangeCase::Additive)) {
      failures +=
          runScenario(ExchangeCase::Additive, options.backend, scenario, rank);
    }
  }
  return failures;
}

} // namespace

int main(int argc, char** argv) {
  MPIdata::init(&argc, &argv);
  int exitCode = EXIT_SUCCESS;

  try {
    const int rank = MPIdata::get_rank();
    const int nprocs = MPIdata::get_nprocs();
    const Options options = parseOptions(argc, argv);
    if (options.help) {
      if (rank == 0)
        printUsage();
      MPIdata::finalize_mpi();
      return EXIT_SUCCESS;
    }

    const int requiredRanks = options.suite == Suite::Distributed ? 8 : 1;
    if (nprocs != requiredRanks) {
      if (rank == 0) {
        std::cerr << "haloCommunicationTest "
                  << (options.suite == Suite::Distributed ? "distributed"
                                                          : "self")
                  << " suite requires " << requiredRanks
                  << " MPI rank(s), but received " << nprocs << ".\n";
      }
      MPIdata::finalize_mpi();
      return EXIT_FAILURE;
    }

    if (runsGpu(options.backend)) {
      const int localReady = selectDeviceForRank() ? 1 : 0;
      int allReady = 0;
      MPI_Allreduce(&localReady, &allReady, 1, MPI_INT, MPI_MIN,
                    MPIdata::get_PicGlobalComm());
      if (!allReady) {
        if (rank == 0) {
          std::cerr << "haloCommunicationTest SKIP: every MPI rank needs a "
                       "visible CUDA/HIP device for backend "
                    << backendName(options.backend) << ".\n";
        }
        MPIdata::finalize_mpi();
        return kSkipReturnCode;
      }
    }

    Parameters::init_parameters();
    int failures = 0;
    if (options.suite == Suite::Distributed) {
      failures = runScenarios(kDistributedScenarios, options, rank);
    } else {
      failures = runScenarios(kSelfScenarios, options, rank);
    }
    if (rank == 0 && failures == 0) {
      std::cout << "haloCommunicationTest " << caseName(options.exchangeCase)
                << " " << backendName(options.backend) << " PASS\n";
    }
    exitCode = failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
  } catch (const std::exception& ex) {
    std::cerr << "[rank " << MPIdata::get_rank() << "] " << ex.what() << "\n";
    MPI_Abort(MPIdata::get_PicGlobalComm(), EXIT_FAILURE);
  } catch (...) {
    std::cerr << "[rank " << MPIdata::get_rank() << "] unknown exception\n";
    MPI_Abort(MPIdata::get_PicGlobalComm(), EXIT_FAILURE);
  }

  MPIdata::finalize_mpi();
  return exitCode;
}
