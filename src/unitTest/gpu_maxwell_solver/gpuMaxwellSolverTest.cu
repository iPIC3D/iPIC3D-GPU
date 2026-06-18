/**
 * Run with mpirun -n 8 ./gpuMaxwellSolverTest --case all
 * to execute all test cases, or with --case periodic, --case reflective, or
 * --case manufactured to run a single case.
 */

#include <mpi.h>

#include "Collective.h"
#include "ConfigFile.h"
#include "EMfields3D.h"
#include "GPUBlas.cuh"
#include "Grid3DCU.h"
#include "MPIdata.h"
#include "Parameters.h"
#include "VCtopology3D.h"
#include "cudaTypeDef.cuh"

#include <array>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>

namespace {

constexpr double kPi = 3.141592653589793238462643383279502884;
constexpr const char* kZeros12 = "0 0 0 0 0 0 0 0 0 0 0 0";
constexpr const char* kOnes12 = "1 1 1 1 1 1 1 1 1 1 1 1";

constexpr size_t kComponentCount = 3;

using Vec3 = std::array<double, kComponentCount>;
using FieldArray = std::array<arr3_double, kComponentCount>;
using FieldLabels = std::array<const char*, kComponentCount>;
using GpuFieldPtrs = std::array<GPUFieldArray3*, kComponentCount>;
using ConstGpuFieldPtrs = std::array<const GPUFieldArray3*, kComponentCount>;

constexpr FieldLabels kElectricLabels = {"Ex", "Ey", "Ez"};
constexpr FieldLabels kLeftGhostLabels = {
    "Ex left ghost", "Ey left ghost", "Ez left ghost"};
constexpr FieldLabels kRightGhostLabels = {
    "Ex right ghost", "Ey right ghost", "Ez right ghost"};
constexpr FieldLabels kManufacturedLabels = {
    "manufactured Ex", "manufactured Ey", "manufactured Ez"};

constexpr std::array<const char*, 4> kAlwaysPeriodicKeys = {
    "PERIODICY", "PERIODICZ", "PERIODICY_P", "PERIODICZ_P"};
constexpr std::array<const char*, 8> kZeroSpeciesKeys = {
    "rhoINIT", "rhoINJECT", "uth", "vth", "wth", "u0", "v0", "w0"};
constexpr std::array<const char*, 3> kUnitParticleKeys = {
    "npcelx", "npcely", "npcelz"};
constexpr std::array<const char*, 6> kPhiBoundaryKeys = {
    "bcPHIfaceXright", "bcPHIfaceXleft", "bcPHIfaceYright",
    "bcPHIfaceYleft", "bcPHIfaceZright", "bcPHIfaceZleft"};
constexpr std::array<const char*, 6> kEmBoundaryKeys = {
    "bcEMfaceXright", "bcEMfaceXleft", "bcEMfaceYright",
    "bcEMfaceYleft", "bcEMfaceZright", "bcEMfaceZleft"};
constexpr std::array<const char*, 6> kParticleBoundaryKeys = {
    "bcPfaceXright", "bcPfaceXleft", "bcPfaceYright",
    "bcPfaceYleft", "bcPfaceZright", "bcPfaceZleft"};
constexpr std::array<const char*, 4> kDisabledOutputCycleKeys = {
    "FieldOutputCycle", "ParticlesOutputCycle", "RestartOutputCycle",
    "DiagnosticsOutputCycle"};

enum class TestCase {
  Periodic,
  Reflective,
  Manufactured
};

constexpr std::array<TestCase, 3> kAllTestCases = {
    TestCase::Periodic, TestCase::Reflective, TestCase::Manufactured};

struct TestSelection {
  bool runAll = true;
  TestCase singleCase = TestCase::Periodic;
};

struct FieldBounds {
  size_t iBegin;
  size_t iEnd;
  size_t jBegin;
  size_t jEnd;
  size_t kBegin;
  size_t kEnd;
};

struct GridPoint {
  double x;
  double y;
  double z;
};

struct WaveNumbers {
  double x;
  double y;
  double z;
};

template <size_t N, typename Value>
void addConfigValues(ConfigFile& cfg, const std::array<const char*, N>& keys,
                     const Value& value) {
  for (const char* key : keys) {
    cfg.add(key, value);
  }
}

struct TestInput {
  double Lx = 16.0;
  double Ly = 4.0;
  double Lz = 4.0;
  int nxc = 128;
  int nyc = 164;
  int nzc = 128;
  int xlen = 2;
  int ylen = 2;
  int zlen = 2;

  double dt = 0.05;
  double c = 1.0;
  double th = 1.0;
  double delta = 0.5;
  double gmresTolerance = 1.0e-11;
  double comparisonTolerance = 1.0e-9;

  double exUniform = 1.25;
  double eyPeriodic = -0.5;
  double ezPeriodic = 0.75;
  double eyReflective = -0.25;
  double ezReflective = 0.5;

  double qom = -1.0;
  double density = 0.35;
  Vec3 backgroundB = {0.31, -0.17, 0.23};

  int mpiRanks() const { return xlen * ylen * zlen; }
  Vec3 periodicElectric() const { return {exUniform, eyPeriodic, ezPeriodic}; }
  Vec3 reflectiveElectric() const {
    return {exUniform, eyReflective, ezReflective};
  }

  ConfigFile config(TestCase testCase) const {
    const bool periodicX = (testCase != TestCase::Reflective);

    ConfigFile cfg;
    cfg.add("SaveDirName", ".");
    cfg.add("RestartDirName", ".");
    cfg.add("Case", "Default");
    cfg.add("PoissonCorrection", "no");
    cfg.add("divBCorrection", "no");
    cfg.add("WriteMethod", "default");
    cfg.add("SimulationName", "gpu_maxwell_solver");

    cfg.add("B0x", 0.0);
    cfg.add("B0y", 0.0);
    cfg.add("B0z", 0.0);
    cfg.add("delta", delta);
    cfg.add("dt", dt);
    cfg.add("ncycles", 1);
    cfg.add("th", th);
    cfg.add("c", c);
    cfg.add("Smooth", 1.0);
    cfg.add("SmoothNiter", 1);

    cfg.add("Lx", Lx);
    cfg.add("Ly", Ly);
    cfg.add("Lz", Lz);
    cfg.add("nxc", nxc);
    cfg.add("nyc", nyc);
    cfg.add("nzc", nzc);
    cfg.add("XLEN", xlen);
    cfg.add("YLEN", ylen);
    cfg.add("ZLEN", zlen);
    cfg.add("PERIODICX", periodicX ? 1 : 0);
    cfg.add("PERIODICX_P", periodicX ? 1 : 0);
    addConfigValues(cfg, kAlwaysPeriodicKeys, 1);

    cfg.add("NiterMover", 1);
    cfg.add("ns", 1);
    std::ostringstream qomValues;
    qomValues << qom << " 0 0 0 0 0 0 0 0 0 0 0";
    cfg.add("qom", qomValues.str());
    addConfigValues(cfg, kZeroSpeciesKeys, kZeros12);
    addConfigValues(cfg, kUnitParticleKeys, kOnes12);

    addConfigValues(cfg, kPhiBoundaryKeys, 1);
    addConfigValues(cfg, kEmBoundaryKeys, 0);
    addConfigValues(cfg, kParticleBoundaryKeys, 1);
    cfg.add("ApplyInflowBcsEImage", 0);

    cfg.add("CGtol", gmresTolerance);
    cfg.add("GMREStol", gmresTolerance);
    cfg.add("SolverType", "GMRES");
    addConfigValues(cfg, kDisabledOutputCycleKeys, 0);
    cfg.add("CallFinalize", 0);
    return cfg;
  }
};

struct Check {
  int rank = 0;
  int failures = 0;
  double tolerance = 0.0;

  void near(const char* label, double actual, double expected,
            size_t i, size_t j, size_t k) {
    if (std::fabs(actual - expected) <= tolerance) return;
    fail();
    if (failures <= 12) {
      std::cerr << "[rank " << rank << "] " << label
                << "(" << i << "," << j << "," << k << ") = "
                << actual << ", expected " << expected << "\n";
    }
  }

  void fail(const char* message = nullptr) {
    failures++;
    if (message != nullptr && failures <= 12) {
      std::cerr << "[rank " << rank << "] " << message << "\n";
    }
  }
};

const char* caseName(TestCase testCase) {
  switch (testCase) {
    case TestCase::Periodic:
      return "periodic";
    case TestCase::Reflective:
      return "reflective";
    case TestCase::Manufactured:
      return "manufactured";
  }
  return "unknown";
}

TestSelection parseSelection(int argc, char** argv) {
  for (int i = 1; i < argc; i++) {
    if (std::strcmp(argv[i], "--case") == 0) {
      if (i + 1 >= argc) {
        throw std::runtime_error(
            "missing value for --case; expected all, periodic, reflective, or "
            "manufactured");
      }
      const char* value = argv[i + 1];
      if (std::strcmp(value, "all") == 0) return {};
      if (std::strcmp(value, "periodic") == 0) {
        return {false, TestCase::Periodic};
      }
      if (std::strcmp(value, "reflective") == 0) {
        return {false, TestCase::Reflective};
      }
      if (std::strcmp(value, "manufactured") == 0) {
        return {false, TestCase::Manufactured};
      }
      throw std::runtime_error(std::string("unknown --case '") + value +
                               "'; expected all, periodic, reflective, or "
                               "manufactured");
    }
  }
  return {};
}

void reportCaseResult(TestCase testCase, int failures) {
  if (failures == 0) {
    std::cout << "gpuMaxwellSolverTest " << caseName(testCase) << " PASS\n";
  } else {
    std::cerr << "gpuMaxwellSolverTest " << caseName(testCase)
              << " FAIL: " << failures << " mismatches\n";
  }
}

bool selectDeviceForRank(int rank) {
  MPI_Comm sharedComm = MPI_COMM_NULL;
  MPI_Comm_split_type(MPIdata::get_PicGlobalComm(), MPI_COMM_TYPE_SHARED,
                      0, MPI_INFO_NULL, &sharedComm);

  int sharedRank = 0;
  MPI_Comm_rank(sharedComm, &sharedRank);

  int deviceCount = 0;
  cudaError_t err = cudaGetDeviceCount(&deviceCount);
  if (err != cudaSuccess || deviceCount == 0) {
    if (rank == 0) {
      std::cerr << "GPU Maxwell solver test requires a visible CUDA/HIP device.\n";
    }
    MPI_Comm_free(&sharedComm);
    return false;
  }

  cudaErrChk(cudaSetDevice(sharedRank % deviceCount));
  MPI_Comm_free(&sharedComm);
  return true;
}

template <size_t N>
void setAll(std::array<arr3_double, N> fields, double value) {
  for (arr3_double& field : fields) {
    field.setall(value);
  }
}

void zeroFields(EMfields3D& fields) {
  fields.setZeroDensities();
  setAll(std::array<arr3_double, 13>{
             fields.getEx(), fields.getEy(), fields.getEz(),
             fields.getBx(), fields.getBy(), fields.getBz(),
             fields.getBxc(), fields.getByc(), fields.getBzc(),
             fields.getBx_ext(), fields.getBy_ext(), fields.getBz_ext(),
             fields.getPHI()},
         0.0);
}

FieldArray electricFields(EMfields3D& fields) {
  return {fields.getEx(), fields.getEy(), fields.getEz()};
}

FieldArray magneticNodeFields(EMfields3D& fields) {
  return {fields.getBx(), fields.getBy(), fields.getBz()};
}

FieldArray magneticCenterFields(EMfields3D& fields) {
  return {fields.getBxc(), fields.getByc(), fields.getBzc()};
}

GpuFieldPtrs gpuElectricFields(EMfields3D& fields) {
  return {&fields.gpuEx(), &fields.gpuEy(), &fields.gpuEz()};
}

FieldBounds fullBounds(const arr3_double& field) {
  return {0, field.dim1(), 0, field.dim2(), 0, field.dim3()};
}

FieldBounds interiorBounds(const arr3_double& field) {
  return {1, field.dim1() - 1, 1, field.dim2() - 1, 1, field.dim3() - 1};
}

FieldBounds xFaceBounds(const arr3_double& field, size_t i) {
  return {i, i + 1, 1, field.dim2() - 1, 1, field.dim3() - 1};
}

template <typename Func>
void forEachIndex(const FieldBounds& bounds, Func func) {
  for (size_t i = bounds.iBegin; i < bounds.iEnd; i++) {
    for (size_t j = bounds.jBegin; j < bounds.jEnd; j++) {
      for (size_t k = bounds.kBegin; k < bounds.kEnd; k++) {
        func(i, j, k);
      }
    }
  }
}

void setFieldValues(FieldArray fields, const Vec3& values) {
  for (size_t component = 0; component < fields.size(); component++) {
    fields[component].setall(values[component]);
  }
}

void setFieldPoint(FieldArray& fields, size_t i, size_t j, size_t k,
                   const Vec3& values) {
  for (size_t component = 0; component < fields.size(); component++) {
    fields[component].fetch(i, j, k) = values[component];
  }
}

void setElectricFields(EMfields3D& fields, const Vec3& electric) {
  zeroFields(fields);
  setFieldValues(electricFields(fields), electric);
}

template <typename ExpectedValues>
void checkElectric(const FieldArray& fields, Check& check,
                   const FieldBounds& bounds, const FieldLabels& labels,
                   ExpectedValues expectedValues) {
  forEachIndex(bounds, [&](size_t i, size_t j, size_t k) {
    const Vec3 expected = expectedValues(i, j, k);
    for (size_t component = 0; component < fields.size(); component++) {
      check.near(labels[component], fields[component].get(i, j, k),
                 expected[component], i, j, k);
    }
  });
}

void checkConstantElectric(const FieldArray& fields, Check& check,
                           const Vec3& expected, const FieldBounds& bounds,
                           const FieldLabels& labels) {
  checkElectric(fields, check, bounds, labels,
                [&](size_t, size_t, size_t) { return expected; });
}

void checkReflective(EMfields3D& fields, Check& check, const TestInput& input) {
  const VirtualTopology3D& vct = fields.get_vct();
  const FieldArray electric = electricFields(fields);

  checkConstantElectric(electric, check, input.reflectiveElectric(),
                        interiorBounds(electric[0]), kElectricLabels);

  if (vct.getXleft_neighbor() == MPI_PROC_NULL) {
    checkConstantElectric(electric, check, {input.exUniform, 0.0, 0.0},
                          xFaceBounds(electric[0], 0), kLeftGhostLabels);
  }

  if (vct.getXright_neighbor() == MPI_PROC_NULL) {
    const size_t ghost = electric[0].dim1() - 1;
    checkConstantElectric(electric, check, {input.exUniform, 0.0, 0.0},
                          xFaceBounds(electric[0], ghost), kRightGhostLabels);
  }
}

Vec3 manufacturedE(const GridPoint& point, const WaveNumbers& wave) {
  const double x = wave.x * point.x;
  const double y = wave.y * point.y;
  const double z = wave.z * point.z;
  return {
      0.70 + 0.17 * std::sin(x) * std::cos(y) + 0.11 * std::cos(z)
           + 0.07 * std::sin(x + y + z),
     -0.35 + 0.13 * std::cos(x) * std::sin(y) + 0.09 * std::sin(z)
           + 0.05 * std::cos(x - 2.0 * z),
      0.22 + 0.15 * std::sin(z) * std::cos(x)
           + 0.06 * std::sin(y + 2.0 * x)
  };
}

Vec3 manufacturedB(const GridPoint& point, const WaveNumbers& wave,
                   const TestInput& input) {
  const double x = wave.x * point.x;
  const double y = wave.y * point.y;
  const double z = wave.z * point.z;
  return {
      input.backgroundB[0] + 0.08 * std::cos(x) * std::sin(y) * std::sin(z),
      input.backgroundB[1] + 0.06 * std::sin(x + y) * std::cos(z),
      input.backgroundB[2] + 0.07 * std::sin(x) * std::cos(y - z)
  };
}

double manufacturedDensity(const GridPoint& point, const WaveNumbers& wave,
                           const TestInput& input) {
  const double x = wave.x * point.x;
  const double y = wave.y * point.y;
  const double z = wave.z * point.z;
  return input.density *
         (1.0 + 0.12 * std::sin(x) * std::cos(y) + 0.07 * std::cos(z));
}

WaveNumbers waveNumbers(const TestInput& input) {
  return {2.0 * kPi / input.Lx, 2.0 * kPi / input.Ly,
          2.0 * kPi / input.Lz};
}

GridPoint nodePoint(const Grid3DCU& grid, size_t i, size_t j, size_t k) {
  return {grid.getXN(static_cast<int>(i)),
          grid.getYN(static_cast<int>(j)),
          grid.getZN(static_cast<int>(k))};
}

GridPoint centerPoint(const Grid3DCU& grid, size_t i, size_t j, size_t k) {
  return {grid.getXC(static_cast<int>(i)),
          grid.getYC(static_cast<int>(j)),
          grid.getZC(static_cast<int>(k))};
}

void initializeManufacturedFields(EMfields3D& fields, const Grid3DCU& grid,
                                  const TestInput& input) {
  zeroFields(fields);

  FieldArray electric = electricFields(fields);
  FieldArray magneticNode = magneticNodeFields(fields);
  arr3_double density = fields.getRHOns(0);

  const WaveNumbers wave = waveNumbers(input);
  forEachIndex(fullBounds(electric[0]), [&](size_t i, size_t j, size_t k) {
    const GridPoint point = nodePoint(grid, i, j, k);
    const Vec3 e = manufacturedE(point, wave);
    const Vec3 b = manufacturedB(point, wave, input);
    setFieldPoint(electric, i, j, k, e);
    setFieldPoint(magneticNode, i, j, k, b);
    density.fetch(i, j, k) = manufacturedDensity(point, wave, input);
  });

  FieldArray magneticCenter = magneticCenterFields(fields);
  forEachIndex(fullBounds(magneticCenter[0]), [&](size_t i, size_t j, size_t k) {
    setFieldPoint(magneticCenter, i, j, k,
                  manufacturedB(centerPoint(grid, i, j, k), wave, input));
  });
}

void checkManufactured(EMfields3D& fields, Check& check,
                       const Grid3DCU& grid, const TestInput& input) {
  const FieldArray electric = electricFields(fields);
  const WaveNumbers wave = waveNumbers(input);

  checkElectric(electric, check, interiorBounds(electric[0]), kManufacturedLabels,
                [&](size_t i, size_t j, size_t k) {
                  return manufacturedE(nodePoint(grid, i, j, k), wave);
                });
}

struct GpuReductionScratch {
  GpuReductionScratch() { gpuBlasAllocScratch(&ptr); }
  ~GpuReductionScratch() { gpuBlasFreeScratch(ptr); }

  GpuReductionScratch(const GpuReductionScratch&) = delete;
  GpuReductionScratch& operator=(const GpuReductionScratch&) = delete;

  cudaSolverType* ptr = nullptr;
};

double localNorm2(const GPUKrylovVector& values,
                  const GpuReductionScratch& scratch, cudaStream_t stream) {
  return gpuNorm2(values.devPtr(), values.size(), scratch.ptr, stream);
}

double localNorm2Sum(const ConstGpuFieldPtrs& fields,
                     const GpuReductionScratch& scratch, cudaStream_t stream) {
  double sum = 0.0;
  for (const GPUFieldArray3* field : fields) {
    sum += gpuNorm2(field->devPtr(), field->size(), scratch.ptr, stream);
  }
  return sum;
}

void checkNonZeroGlobalNorm(Check& check, int rank, double localSquaredNorm,
                            double tolerance, const char* message) {
  double globalNorm2 = 0.0;
  MPI_Allreduce(&localSquaredNorm, &globalNorm2, 1, MPI_DOUBLE, MPI_SUM,
                MPIdata::get_PicGlobalComm());
  if (rank == 0 && globalNorm2 <= tolerance * tolerance) {
    check.fail(message);
  }
}

void prepareManufacturedMoments(EMfields3D& fields) {
  fields.gpuCommunicateGhostP2G_AllSpecies();
  fields.gpuSetZeroDerivedMoments();
  fields.gpuSumOverSpecies();
  fields.gpuInterpDensitiesN2C();
  fields.gpuCalculateHatFunctions();
}

void communicateElectricBoundaries(EMfields3D& fields, Grid3DCU& grid) {
  fields.gpuCommunicateNodeBC_3mixed(grid.getNXN(), grid.getNYN(),
                                     grid.getNZN(),
                                     fields.gpuEx(), fields.get_col().bcEx,
                                     fields.gpuEy(), fields.get_col().bcEy,
                                     fields.gpuEz(), fields.get_col().bcEz);
}

void zeroGpuFields(GpuFieldPtrs fields, cudaStream_t stream) {
  for (GPUFieldArray3* field : fields) {
    field->setAll(0.0, stream);
  }
}

int maxwellVectorSize(const Grid3DCU& grid) {
  return static_cast<int>(kComponentCount) * (grid.getNXN() - 2) *
         (grid.getNYN() - 2) * (grid.getNZN() - 2);
}

void electricToSolver(GPUKrylovVector& target, EMfields3D& fields,
                      const Grid3DCU& grid, cudaStream_t stream) {
  gpuPhys2Solver3(target.devPtr(),
                  fields.gpuEx().devPtr(), fields.gpuEy().devPtr(),
                  fields.gpuEz().devPtr(),
                  grid.getNXN(), grid.getNYN(), grid.getNZN(), stream);
}

void solverToElectric(EMfields3D& fields, const GPUKrylovVector& source,
                      const Grid3DCU& grid, cudaStream_t stream) {
  gpuSolver2Phys3(fields.gpuEx().devPtr(), fields.gpuEy().devPtr(),
                  fields.gpuEz().devPtr(), source.devPtr(),
                  grid.getNXN(), grid.getNYN(), grid.getNZN(), stream);
}

void runPeriodicCase(EMfields3D& fields, const TestInput& input,
                     Check& check) {
  const cudaStream_t stream = fields.gpuSolverStream();
  setElectricFields(fields, input.periodicElectric());
  fields.gpuSolverSyncH2D(stream);
  fields.gpuCalculateE(0);
  fields.gpuSolverSyncD2H(stream);
  const FieldArray electric = electricFields(fields);
  checkConstantElectric(electric, check, input.periodicElectric(),
                        fullBounds(electric[0]), kElectricLabels);
}

void runReflectiveCase(EMfields3D& fields, Grid3DCU& grid,
                       const TestInput& input, Check& check) {
  const cudaStream_t stream = fields.gpuSolverStream();
  setElectricFields(fields, input.reflectiveElectric());
  fields.gpuSolverSyncH2D(stream);
  communicateElectricBoundaries(fields, grid);
  fields.gpuSolverSyncD2H(stream);
  checkReflective(fields, check, input);
}

void runManufacturedCase(EMfields3D& fields, Grid3DCU& grid,
                         const TestInput& input, int rank, Check& check) {
  const cudaStream_t stream = fields.gpuSolverStream();
  initializeManufacturedFields(fields, grid, input);
  fields.gpuSolverSyncH2D(stream);
  prepareManufacturedMoments(fields);

  const int nMaxwell = maxwellVectorSize(grid);
  GPUKrylovVector exact(nMaxwell);
  GPUKrylovVector rhs(nMaxwell);
  GPUKrylovVector sourceOnly(nMaxwell);
  GPUKrylovVector oldElectric(nMaxwell);

  electricToSolver(exact, fields, grid, stream);

  GPUFieldArray3 muX(grid.getNXN(), grid.getNYN(), grid.getNZN());
  GPUFieldArray3 muY(grid.getNXN(), grid.getNYN(), grid.getNZN());
  GPUFieldArray3 muZ(grid.getNXN(), grid.getNYN(), grid.getNZN());
  fields.gpuMUdot(muX, muY, muZ, fields.gpuEx(), fields.gpuEy(),
                  fields.gpuEz());

  GpuReductionScratch reductionScratch;
  checkNonZeroGlobalNorm(
      check, rank, localNorm2Sum({&muX, &muY, &muZ}, reductionScratch, stream),
      input.comparisonTolerance,
      "manufactured MUdot is zero; density/B setup is not exercising the "
      "full Maxwell operator");

  fields.gpuMaxwellImage(rhs.devPtr(), exact.devPtr());

  zeroGpuFields(gpuElectricFields(fields), stream);
  fields.gpuMaxwellSource(sourceOnly.devPtr());

  checkNonZeroGlobalNorm(
      check, rank, localNorm2(sourceOnly, reductionScratch, stream),
      input.comparisonTolerance,
      "manufactured Maxwell source contribution is zero; gpuMaxwellSource is "
      "not exercising the real RHS path");

  gpuSubRes(oldElectric.devPtr(), rhs.devPtr(), sourceOnly.devPtr(),
            oldElectric.size(), stream);

  solverToElectric(fields, oldElectric, grid, stream);
  fields.gpuCalculateE(0);
  fields.gpuSolverSyncD2H(stream);
  checkManufactured(fields, check, grid, input);
}

int runOnFields(TestCase testCase, EMfields3D& fields, Grid3DCU& grid,
                const TestInput& input, int rank) {
  Check check;
  check.rank = rank;
  check.tolerance = input.comparisonTolerance;

  fields.gpuSolverAllocate();

  switch (testCase) {
    case TestCase::Periodic:
      runPeriodicCase(fields, input, check);
      break;

    case TestCase::Reflective:
      runReflectiveCase(fields, grid, input, check);
      break;

    case TestCase::Manufactured:
      runManufacturedCase(fields, grid, input, rank, check);
      break;
  }

  return check.failures;
}

int runCase(TestCase testCase, int rank, int nprocs, const TestInput& input) {
  ConfigFile config = input.config(testCase);
  Collective col(config, "gpuMaxwellSolverTest");
  VCtopology3D vct(col);
  if (nprocs != vct.getNprocs()) {
    throw std::runtime_error("MPI size does not match generated topology");
  }
  vct.setup_vctopology(MPIdata::get_PicGlobalComm());

  Grid3DCU grid(&col, &vct);
  EMfields3D fields(&col, &grid, &vct);

  const int localFailures = runOnFields(testCase, fields, grid, input, rank);
  int globalFailures = 0;
  MPI_Allreduce(&localFailures, &globalFailures, 1, MPI_INT, MPI_SUM,
                MPIdata::get_PicGlobalComm());
  return globalFailures;
}

} // namespace

int main(int argc, char** argv) {
  MPIdata::init(&argc, &argv);

  int exitCode = 0;
  try {
    const int rank = MPIdata::get_rank();
    const int nprocs = MPIdata::get_nprocs();
    const TestInput input;

    if (nprocs != input.mpiRanks()) {
      if (rank == 0) {
        std::cerr << "GPU Maxwell solver test requires " << input.mpiRanks()
                  << " MPI ranks for the configured "
                  << input.xlen << "x" << input.ylen
                  << "x" << input.zlen
                  << " subdomains, but it was launched with " << nprocs
                  << ".\n";
      }
      MPIdata::finalize_mpi();
      return EXIT_FAILURE;
    }

    const int localDeviceReady = selectDeviceForRank(rank) ? 1 : 0;
    int allDeviceReady = 0;
    MPI_Allreduce(&localDeviceReady, &allDeviceReady, 1, MPI_INT, MPI_MIN,
                  MPIdata::get_PicGlobalComm());
    if (!allDeviceReady) {
      if (rank == 0) {
        std::cerr << "GPU Maxwell solver test requires every MPI rank to have "
                  << "a visible CUDA/HIP device.\n";
      }
      MPIdata::finalize_mpi();
      return EXIT_FAILURE;
    }

    Parameters::init_parameters();

    const TestSelection selection = parseSelection(argc, argv);
    int failures = 0;
    if (selection.runAll) {
      for (TestCase testCase : kAllTestCases) {
        const int caseFailures = runCase(testCase, rank, nprocs, input);
        failures += caseFailures;
        if (rank == 0) {
          reportCaseResult(testCase, caseFailures);
        }
      }
      if (rank == 0 && failures == 0) {
        std::cout << "gpuMaxwellSolverTest all PASS\n";
      }
    } else {
      failures = runCase(selection.singleCase, rank, nprocs, input);
      if (rank == 0) {
        reportCaseResult(selection.singleCase, failures);
      }
    }
    exitCode = failures == 0 ? 0 : 1;
  } catch (const std::exception& ex) {
    std::cerr << "[rank " << MPIdata::get_rank() << "] " << ex.what() << "\n";
    MPI_Abort(MPIdata::get_PicGlobalComm(), 1);
  } catch (...) {
    std::cerr << "[rank " << MPIdata::get_rank() << "] unknown exception\n";
    MPI_Abort(MPIdata::get_PicGlobalComm(), 1);
  }

  MPIdata::finalize_mpi();
  return exitCode;
}
