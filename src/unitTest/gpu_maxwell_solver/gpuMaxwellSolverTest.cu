/**
 * Run with mpirun -n 8 ./gpuMaxwellSolverTest --case all
 * to execute all test cases, or with --case periodic, --case reflective,
 * --case manufactured, --case oblique-wave, --case boundary-rows or
 * --case image-reference to run a single case.
 *
 * boundary-rows, and image-reference where its layout leaves x non-periodic,
 * use an open inflow face at x-left (bcEMface = bcPface = 2,
 * ApplyInflowBcsEImage = 1).
 *   boundary-rows (x-right perfect conductor): gpuMaxwellImage of a random
 *     vector must already hold the face rows, i.e. equal itself after
 *     gpuPerfectConductorImages and gpuOpenBoundaryInflowEImage are applied
 *     to it again.
 *   image-reference (x-right open, which keeps its operator rows, or an
 *     outflow face; conductor walls on the non-periodic y and z faces), on the
 *     layouts of kImageLayouts (3D and 2D, flat and self-periodic axes,
 *     odd and minimal subdomains): gpuMaxwellImage of a random vector against
 *     the image assembled step by step: the cell gradients exchanged,
 *     divC2N, MUdot, div D exchanged, gradC2N, delt^2 (-lap - grad div D) +
 *     D + E with the vector kernels, the packing and the face rows.  Each
 *     layout runs with HaloOverlap = 0 and 1; with 1 the image overlaps its
 *     exchange where the layout says so, and equals
 *     gpuMaxwellImage<HaloMode::Blocking> bitwise.
 *
 * --halo-overlap runs the selected cases with HaloOverlap = 1 (the solver
 * steps in HaloMode::Overlap); image-reference always runs both modes.
 */

#include <mpi.h>

#include "Collective.h"
#include "ConfigFile.h"
#include "EMfields3D.h"
#include "GPUBlas.cuh"
#include "GPUStencils.cuh"
#include "Grid3DCU.h"
#include "MPIdata.h"
#include "OutflowBoundary.h"
#include "Parameters.h"
#include "VCtopology3D.h"
#include "cudaTypeDef.cuh"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr double kPi = 3.141592653589793238462643383279502884;
constexpr const char* kZeros12 = "0 0 0 0 0 0 0 0 0 0 0 0";
constexpr const char* kOnes12 = "1 1 1 1 1 1 1 1 1 1 1 1";
// At th=1 the coupled Maxwell update is backward Euler.  The two references
// below separate its second-order spatial error from the first-order temporal
// error of a fixed-final-time dt ~ h refinement.
constexpr std::array<int, 3> kObliqueWaveCells = {16, 32, 64};
constexpr double kObliqueWaveLength = 1.0;
constexpr double kObliqueWaveFinalTime = 0.05;
constexpr double kObliqueWaveCourant = 0.2;
constexpr double kMinimumSpatialOrder = 1.8;
constexpr double kMaximumSpatialOrder = 2.2;
constexpr double kMaximumFineSpatialL2 = 0.005;
constexpr double kMaximumFineSpatialLinf = 0.005;
constexpr double kMinimumSpaceTimeOrder = 0.8;
constexpr double kMaximumSpaceTimeOrder = 1.2;
constexpr double kMaximumFineSpaceTimeL2 = 0.02;
constexpr double kMaximumFineSpaceTimeLinf = 0.02;

constexpr size_t kComponentCount = 3;

using Vec3 = std::array<double, kComponentCount>;
using FieldArray = std::array<arr3_double, kComponentCount>;
using FieldLabels = std::array<const char*, kComponentCount>;
using GpuFieldPtrs = std::array<GPUFieldArray3*, kComponentCount>;
using ConstGpuFieldPtrs = std::array<const GPUFieldArray3*, kComponentCount>;

constexpr FieldLabels kElectricLabels = {"Ex", "Ey", "Ez"};
constexpr FieldLabels kLeftGhostLabels = {"Ex left ghost", "Ey left ghost",
                                          "Ez left ghost"};
constexpr FieldLabels kRightGhostLabels = {"Ex right ghost", "Ey right ghost",
                                           "Ez right ghost"};
constexpr FieldLabels kManufacturedLabels = {
    "manufactured Ex", "manufactured Ey", "manufactured Ez"};

constexpr std::array<const char*, 4> kAlwaysPeriodicKeys = {
    "PERIODICY", "PERIODICZ", "PERIODICY_P", "PERIODICZ_P"};
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

enum class TestCase {
  Periodic,
  Reflective,
  Manufactured,
  ObliqueWave,
  BoundaryRows,
  ImageReference
};

constexpr std::array<TestCase, 6> kAllTestCases = {
    TestCase::Periodic,    TestCase::Reflective,   TestCase::Manufactured,
    TestCase::ObliqueWave, TestCase::BoundaryRows, TestCase::ImageReference};

struct TestSelection {
  bool runAll = true;
  TestCase singleCase = TestCase::Periodic;
  bool haloOverlap = false; // --halo-overlap
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

struct ObliqueWave {
  WaveNumbers waveVector;
  Vec3 electricPolarization;
  Vec3 magneticPolarization;
  double angularFrequency;
};

struct WaveState {
  double amplitude;
  double phaseAdvance;
};

struct WaveError {
  double electricL2;
  double magneticL2;
  double combinedL2;
  double electricLinf;
  double magneticLinf;
  unsigned long long nodeCount;
  unsigned long long centerCount;
};

struct WaveLevelResult {
  int cells = 0;
  int steps = 0;
  double dt = 0.0;
  WaveError spatialError = {};
  WaveError spaceTimeError = {};
};

struct WaveStudyCriteria {
  const char* name;
  const char* reference;
  double expectedOrder;
  double minimumOrder;
  double maximumOrder;
  double maximumFineL2;
  double maximumFineLinf;
  WaveError WaveLevelResult::* errorMember;
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
  int cycles = 1;
  int haloOverlap = 0; // input HaloOverlap

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
    const bool openInflowXleft = testCase == TestCase::BoundaryRows ||
                                 testCase == TestCase::ImageReference;
    const bool periodicX = testCase != TestCase::Reflective && !openInflowXleft;

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
    cfg.add("ncycles", cycles);
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
    if (openInflowXleft) {
      // open inflow face at x-left
      cfg.add("bcEMfaceXleft", 2);
      cfg.add("bcPfaceXleft", 2);
      cfg.add("ApplyInflowBcsEImage", 1);
    }
    if (testCase == TestCase::ImageReference) {
      // x-right open but not an inflow face: it keeps its operator rows, so
      // the image reads the face-BC ghost cells there; conductor walls in y
      cfg.add("bcEMfaceXright", 2);
      cfg.add("bcPfaceXright", 2);
      cfg.add("PERIODICY", 0);
      cfg.add("PERIODICY_P", 0);
    }

    cfg.add("CGtol", gmresTolerance);
    cfg.add("GMREStol", gmresTolerance);
    cfg.add("SolverType", "GMRES");
    addConfigValues(cfg, kDisabledOutputCycleKeys, 0);
    cfg.add("CallFinalize", 0);
    cfg.add("HaloOverlap", haloOverlap);
    return cfg;
  }
};

/** A layout of the image-reference case (8 ranks).  Non-periodic x: open
 *  inflow face at x-left, open x-right face (outflowXright: the outflow
 *  closure, bcPfaceXright = 3); non-periodic y, z: conductor walls. */
struct ImageLayout {
  const char* name;
  int cells[3];         // global cells
  int ranks[3];         // ranks per axis
  const char* periodic; // periodic axes: letters of "xyz"
  bool outflowXright;
  bool split; // with HaloOverlap = 1 the image overlaps its exchange
};

constexpr int kImageLayoutRanks = 8;
constexpr ImageLayout kImageLayouts[] = {
    {"3D, conductor y", {128, 164, 128}, {2, 2, 2}, "z", false, true},
    {"3D, outflow", {64, 40, 48}, {2, 2, 2}, "z", true, true},
    // y flat (one periodic cell)
    {"2D x-z, outflow", {96, 1, 64}, {4, 1, 2}, "yz", true, true},
    // z flat
    {"2D x-y, conductor y", {64, 48, 1}, {2, 4, 1}, "xz", false, true},
    // y and z periodic onto the rank itself
    {"3D, y z on the rank", {64, 20, 24}, {8, 1, 1}, "xyz", false, true},
    {"3D, 37x23x29 per rank", {74, 46, 58}, {2, 2, 2}, "xyz", false, true},
    // the fewest cells that split
    {"3D, 3 cells in x", {6, 26, 14}, {2, 2, 2}, "y", true, true},
    // too few: the blocking exchange
    {"3D, 2 cells in z", {32, 20, 4}, {2, 2, 2}, "z", false, false},
};

struct Check {
  int rank = 0;
  int failures = 0;
  double tolerance = 0.0;

  void near(const char* label, double actual, double expected, size_t i,
            size_t j, size_t k) {
    if (std::fabs(actual - expected) <= tolerance)
      return;
    fail();
    if (failures <= 12) {
      std::cerr << "[rank " << rank << "] " << label << "(" << i << "," << j
                << "," << k << ") = " << actual << ", expected " << expected
                << "\n";
    }
  }

  void fail(const char* message = nullptr) {
    failures++;
    if (message != nullptr && failures <= 12) {
      std::cerr << "[rank " << rank << "] " << message << "\n";
    }
  }
};

struct InterpolationBox {
  const char* name;
  int iLo;
  int iHi;
  int jLo;
  int jHi;
  int kLo;
  int kHi;
};

int runInterpolationRangeComplementRegression(int rank) {
  constexpr int nxn = 7;
  constexpr int nyn = 8;
  constexpr int nzn = 9;
  constexpr int nxc = nxn - 1;
  constexpr int nyc = nyn - 1;
  constexpr int nzc = nzn - 1;
  constexpr cudaSolverType sentinel = -1234567.0;
  constexpr std::array<InterpolationBox, 4> boxes = {{
      {"default", 2, nxn - 3, 2, nyn - 3, 2, nzn - 3},
      {"asymmetric", 2, 3, 4, 5, 3, 6},
      {"full", 1, nxn - 2, 1, nyn - 2, 1, nzn - 2},
      {"empty", 4, 3, 2, nyn - 3, 2, nzn - 3},
  }};

  const size_t centerSize = static_cast<size_t>(nxc) * nyc * nzc;
  const size_t nodeSize = static_cast<size_t>(nxn) * nyn * nzn;
  std::vector<cudaSolverType> oldCenterHost(centerSize);
  std::vector<cudaSolverType> newCenterHost(centerSize);
  std::vector<cudaSolverType> oldReference(nodeSize);
  std::vector<cudaSolverType> newReference(nodeSize);
  std::vector<cudaSolverType> resultHost(nodeSize);

  for (int i = 0; i < nxc; ++i) {
    for (int j = 0; j < nyc; ++j) {
      for (int k = 0; k < nzc; ++k) {
        const size_t idx = (static_cast<size_t>(i) * nyc + j) * nzc + k;
        oldCenterHost[idx] = 1000.0 + 100.0 * i + 10.0 * j + k;
        newCenterHost[idx] = -2000.0 - 80.0 * i - 7.0 * j - 0.5 * k;
      }
    }
  }

  GPUFieldArray3 oldCenter(nxc, nyc, nzc);
  GPUFieldArray3 newCenter(nxc, nyc, nzc);
  GPUFieldArray3 reference(nxn, nyn, nzn);
  GPUFieldArray3 result(nxn, nyn, nzn);
  oldCenter.copyFromHost(oldCenterHost.data());
  newCenter.copyFromHost(newCenterHost.data());

  gpuInterpC2N(reference.devPtr(), oldCenter.devPtr(), nxn, nyn, nzn);
  reference.copyToHost(oldReference.data());
  gpuInterpC2N(reference.devPtr(), newCenter.devPtr(), nxn, nyn, nzn);
  reference.copyToHost(newReference.data());

  Check check;
  check.rank = rank;
  check.tolerance = 0.0;
  for (const InterpolationBox& box : boxes) {
    result.setAll(sentinel);
    gpuInterpC2N_range(result.devPtr(), oldCenter.devPtr(), nxn, nyn, nzn,
                       box.iLo, box.iHi, box.jLo, box.jHi, box.kLo, box.kHi);
    gpuInterpC2N_complement(result.devPtr(), newCenter.devPtr(), nxn, nyn, nzn,
                            box.iLo, box.iHi, box.jLo, box.jHi, box.kLo,
                            box.kHi);
    result.copyToHost(resultHost.data());

    const int iLo = std::max(1, box.iLo);
    const int iHi = std::min(nxn - 2, box.iHi);
    const int jLo = std::max(1, box.jLo);
    const int jHi = std::min(nyn - 2, box.jHi);
    const int kLo = std::max(1, box.kLo);
    const int kHi = std::min(nzn - 2, box.kHi);
    const bool boxIsEmpty = iLo > iHi || jLo > jHi || kLo > kHi;
    const std::string label = std::string("interpC2N ") + box.name;

    for (int i = 0; i < nxn; ++i) {
      for (int j = 0; j < nyn; ++j) {
        for (int k = 0; k < nzn; ++k) {
          const size_t idx = (static_cast<size_t>(i) * nyn + j) * nzn + k;
          const bool active = i >= 1 && i <= nxn - 2 && j >= 1 &&
                              j <= nyn - 2 && k >= 1 && k <= nzn - 2;
          const bool inSafeBox = !boxIsEmpty && i >= iLo && i <= iHi &&
                                 j >= jLo && j <= jHi && k >= kLo && k <= kHi;
          const cudaSolverType expected =
              !active ? sentinel
                      : (inSafeBox ? oldReference[idx] : newReference[idx]);
          check.near(label.c_str(), resultHost[idx], expected, i, j, k);
        }
      }
    }
  }
  return check.failures;
}

void reportInterpolationResult(int failures) {
  if (failures == 0) {
    std::cout << "gpuMaxwellSolverTest interpolation PASS\n";
  } else {
    std::cerr << "gpuMaxwellSolverTest interpolation FAIL: " << failures
              << " mismatches\n";
  }
}

const char* caseName(TestCase testCase) {
  switch (testCase) {
  case TestCase::Periodic:
    return "periodic";
  case TestCase::Reflective:
    return "reflective";
  case TestCase::Manufactured:
    return "manufactured";
  case TestCase::ObliqueWave:
    return "oblique-wave";
  case TestCase::BoundaryRows:
    return "boundary-rows";
  case TestCase::ImageReference:
    return "image-reference";
  }
  return "unknown";
}

TestSelection parseCase(int argc, char** argv);

TestSelection parseSelection(int argc, char** argv) {
  bool haloOverlap = false;
  for (int i = 1; i < argc; i++)
    haloOverlap = haloOverlap || std::strcmp(argv[i], "--halo-overlap") == 0;
  TestSelection selection = parseCase(argc, argv);
  selection.haloOverlap = haloOverlap;
  return selection;
}

TestSelection parseCase(int argc, char** argv) {
  for (int i = 1; i < argc; i++) {
    if (std::strcmp(argv[i], "--case") == 0) {
      if (i + 1 >= argc) {
        throw std::runtime_error(
            "missing value for --case; expected all, periodic, reflective, "
            "manufactured, oblique-wave, boundary-rows or image-reference");
      }
      const char* value = argv[i + 1];
      if (std::strcmp(value, "all") == 0)
        return {};
      if (std::strcmp(value, "periodic") == 0) {
        return {false, TestCase::Periodic};
      }
      if (std::strcmp(value, "reflective") == 0) {
        return {false, TestCase::Reflective};
      }
      if (std::strcmp(value, "manufactured") == 0) {
        return {false, TestCase::Manufactured};
      }
      if (std::strcmp(value, "oblique-wave") == 0) {
        return {false, TestCase::ObliqueWave};
      }
      if (std::strcmp(value, "boundary-rows") == 0) {
        return {false, TestCase::BoundaryRows};
      }
      if (std::strcmp(value, "image-reference") == 0) {
        return {false, TestCase::ImageReference};
      }
      throw std::runtime_error(std::string("unknown --case '") + value +
                               "'; expected all, periodic, reflective, "
                               "manufactured, oblique-wave, boundary-rows or "
                               "image-reference");
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
  MPI_Comm_split_type(MPIdata::get_PicGlobalComm(), MPI_COMM_TYPE_SHARED, 0,
                      MPI_INFO_NULL, &sharedComm);

  int sharedRank = 0;
  MPI_Comm_rank(sharedComm, &sharedRank);

  int deviceCount = 0;
  cudaError_t err = cudaGetDeviceCount(&deviceCount);
  if (err != cudaSuccess || deviceCount == 0) {
    if (rank == 0) {
      std::cerr
          << "GPU Maxwell solver test requires a visible CUDA/HIP device.\n";
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
  setAll(
      std::array<arr3_double, 13>{
          fields.getEx(), fields.getEy(), fields.getEz(), fields.getBx(),
          fields.getBy(), fields.getBz(), fields.getBxc(), fields.getByc(),
          fields.getBzc(), fields.getBx_ext(), fields.getBy_ext(),
          fields.getBz_ext(), fields.getPHI()},
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

FieldBounds ownedPeriodicNodeBounds(const arr3_double& field) {
  // The high proper node of each subdomain is the low proper node of its
  // periodic neighbor.  Excluding all local high faces counts each global node
  // exactly once, including edges and corners.
  return {1, field.dim1() - 2, 1, field.dim2() - 2, 1, field.dim3() - 2};
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
  return {0.70 + 0.17 * std::sin(x) * std::cos(y) + 0.11 * std::cos(z) +
              0.07 * std::sin(x + y + z),
          -0.35 + 0.13 * std::cos(x) * std::sin(y) + 0.09 * std::sin(z) +
              0.05 * std::cos(x - 2.0 * z),
          0.22 + 0.15 * std::sin(z) * std::cos(x) +
              0.06 * std::sin(y + 2.0 * x)};
}

Vec3 manufacturedB(const GridPoint& point, const WaveNumbers& wave,
                   const TestInput& input) {
  const double x = wave.x * point.x;
  const double y = wave.y * point.y;
  const double z = wave.z * point.z;
  return {input.backgroundB[0] + 0.08 * std::cos(x) * std::sin(y) * std::sin(z),
          input.backgroundB[1] + 0.06 * std::sin(x + y) * std::cos(z),
          input.backgroundB[2] + 0.07 * std::sin(x) * std::cos(y - z)};
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
  return {2.0 * kPi / input.Lx, 2.0 * kPi / input.Ly, 2.0 * kPi / input.Lz};
}

GridPoint nodePoint(const Grid3DCU& grid, size_t i, size_t j, size_t k) {
  return {grid.getXN(static_cast<int>(i)), grid.getYN(static_cast<int>(j)),
          grid.getZN(static_cast<int>(k))};
}

GridPoint centerPoint(const Grid3DCU& grid, size_t i, size_t j, size_t k) {
  return {grid.getXC(static_cast<int>(i)), grid.getYC(static_cast<int>(j)),
          grid.getZC(static_cast<int>(k))};
}

ObliqueWave makeObliqueWave(const TestInput& input) {
  const WaveNumbers waveVector = {2.0 * kPi / input.Lx, 2.0 * kPi / input.Ly,
                                  2.0 * kPi / input.Lz};
  const double magnitude =
      std::sqrt(waveVector.x * waveVector.x + waveVector.y * waveVector.y +
                waveVector.z * waveVector.z);

  // For a cubic domain k is parallel to (1,1,1).  Both polarization vectors
  // are unit length, perpendicular to k, and have no zero component.  The
  // magnetic polarization is khat x electricPolarization, matching the
  // production signs dE/dt = c curl(B), dB/dt = -c curl(E).
  return {
      waveVector,
      {1.0 / std::sqrt(14.0), 2.0 / std::sqrt(14.0), -3.0 / std::sqrt(14.0)},
      {-5.0 / std::sqrt(42.0), 4.0 / std::sqrt(42.0), 1.0 / std::sqrt(42.0)},
      input.c * magnitude};
}

WaveState continuumWaveState(const ObliqueWave& wave, double time) {
  return {1.0, wave.angularFrequency * time};
}

WaveState backwardEulerWaveState(const ObliqueWave& wave, double dt,
                                 int steps) {
  // Exact continuous-space backward-Euler amplification
  // (1 + i*omega*dt)^(-steps), expressed as decay and phase advance.
  const double scaledFrequency = wave.angularFrequency * dt;
  return {
      std::exp(-0.5 * steps * std::log1p(scaledFrequency * scaledFrequency)),
      steps * std::atan(scaledFrequency)};
}

Vec3 obliqueWaveField(const GridPoint& point, const ObliqueWave& wave,
                      const WaveState& state, const Vec3& polarization) {
  const double phase = wave.waveVector.x * point.x +
                       wave.waveVector.y * point.y +
                       wave.waveVector.z * point.z - state.phaseAdvance;
  const double value = state.amplitude * std::cos(phase);
  return {polarization[0] * value, polarization[1] * value,
          polarization[2] * value};
}

Vec3 obliqueElectric(const GridPoint& point, const ObliqueWave& wave,
                     const WaveState& state) {
  return obliqueWaveField(point, wave, state, wave.electricPolarization);
}

Vec3 obliqueMagnetic(const GridPoint& point, const ObliqueWave& wave,
                     const WaveState& state) {
  return obliqueWaveField(point, wave, state, wave.magneticPolarization);
}

void initializeObliqueWave(EMfields3D& fields, const Grid3DCU& grid,
                           const ObliqueWave& wave) {
  zeroFields(fields);

  constexpr WaveState initialState = {1.0, 0.0};
  FieldArray electric = electricFields(fields);
  FieldArray magneticNode = magneticNodeFields(fields);
  forEachIndex(fullBounds(electric[0]), [&](size_t i, size_t j, size_t k) {
    const GridPoint point = nodePoint(grid, i, j, k);
    setFieldPoint(electric, i, j, k,
                  obliqueElectric(point, wave, initialState));
    setFieldPoint(magneticNode, i, j, k,
                  obliqueMagnetic(point, wave, initialState));
  });

  FieldArray magneticCenter = magneticCenterFields(fields);
  forEachIndex(fullBounds(magneticCenter[0]),
               [&](size_t i, size_t j, size_t k) {
                 setFieldPoint(magneticCenter, i, j, k,
                               obliqueMagnetic(centerPoint(grid, i, j, k), wave,
                                               initialState));
               });
}

void accumulatePointError(const FieldArray& numerical, const Vec3& exact,
                          size_t i, size_t j, size_t k, double& error2,
                          double& reference2, double& maxError) {
  for (size_t component = 0; component < kComponentCount; component++) {
    const double difference =
        numerical[component].get(i, j, k) - exact[component];
    error2 += difference * difference;
    reference2 += exact[component] * exact[component];
    maxError = std::max(maxError, std::fabs(difference));
  }
}

WaveError measureObliqueWaveError(EMfields3D& fields, const Grid3DCU& grid,
                                  const ObliqueWave& wave,
                                  const WaveState& reference) {
  const FieldArray electric = electricFields(fields);
  const FieldArray magneticCenter = magneticCenterFields(fields);

  std::array<double, 4> localSums = {0.0, 0.0, 0.0, 0.0};
  std::array<double, 2> localMaxima = {0.0, 0.0};
  std::array<unsigned long long, 2> localCounts = {0, 0};

  forEachIndex(ownedPeriodicNodeBounds(electric[0]), [&](size_t i, size_t j,
                                                         size_t k) {
    accumulatePointError(
        electric, obliqueElectric(nodePoint(grid, i, j, k), wave, reference), i,
        j, k, localSums[0], localSums[1], localMaxima[0]);
    localCounts[0]++;
  });

  forEachIndex(
      interiorBounds(magneticCenter[0]), [&](size_t i, size_t j, size_t k) {
        accumulatePointError(
            magneticCenter,
            obliqueMagnetic(centerPoint(grid, i, j, k), wave, reference), i, j,
            k, localSums[2], localSums[3], localMaxima[1]);
        localCounts[1]++;
      });

  std::array<double, 4> globalSums = {};
  std::array<double, 2> globalMaxima = {};
  std::array<unsigned long long, 2> globalCounts = {};
  MPI_Allreduce(localSums.data(), globalSums.data(),
                static_cast<int>(localSums.size()), MPI_DOUBLE, MPI_SUM,
                MPIdata::get_PicGlobalComm());
  MPI_Allreduce(localMaxima.data(), globalMaxima.data(),
                static_cast<int>(localMaxima.size()), MPI_DOUBLE, MPI_MAX,
                MPIdata::get_PicGlobalComm());
  MPI_Allreduce(localCounts.data(), globalCounts.data(),
                static_cast<int>(localCounts.size()), MPI_UNSIGNED_LONG_LONG,
                MPI_SUM, MPIdata::get_PicGlobalComm());

  const double electricL2 = std::sqrt(globalSums[0] / globalSums[1]);
  const double magneticL2 = std::sqrt(globalSums[2] / globalSums[3]);
  const double combinedL2 = std::sqrt((globalSums[0] + globalSums[2]) /
                                      (globalSums[1] + globalSums[3]));
  return {electricL2,      magneticL2,      combinedL2,     globalMaxima[0],
          globalMaxima[1], globalCounts[0], globalCounts[1]};
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
  forEachIndex(
      fullBounds(magneticCenter[0]), [&](size_t i, size_t j, size_t k) {
        setFieldPoint(magneticCenter, i, j, k,
                      manufacturedB(centerPoint(grid, i, j, k), wave, input));
      });
}

void checkManufactured(EMfields3D& fields, Check& check, const Grid3DCU& grid,
                       const TestInput& input) {
  const FieldArray electric = electricFields(fields);
  const WaveNumbers wave = waveNumbers(input);

  checkElectric(electric, check, interiorBounds(electric[0]),
                kManufacturedLabels, [&](size_t i, size_t j, size_t k) {
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
  fields.gpuCommunicateNodeBC_3mixed(
      grid.getNXN(), grid.getNYN(), grid.getNZN(), fields.gpuEx(),
      fields.get_col().bcEx, fields.gpuEy(), fields.get_col().bcEy,
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
  gpuPhys2Solver3(target.devPtr(), fields.gpuEx().devPtr(),
                  fields.gpuEy().devPtr(), fields.gpuEz().devPtr(),
                  grid.getNXN(), grid.getNYN(), grid.getNZN(), stream);
}

void solverToElectric(EMfields3D& fields, const GPUKrylovVector& source,
                      const Grid3DCU& grid, cudaStream_t stream) {
  gpuSolver2Phys3(fields.gpuEx().devPtr(), fields.gpuEy().devPtr(),
                  fields.gpuEz().devPtr(), source.devPtr(), grid.getNXN(),
                  grid.getNYN(), grid.getNZN(), stream);
}

void runPeriodicCase(EMfields3D& fields, const TestInput& input, Check& check) {
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

void runBoundaryRowsCase(EMfields3D& fields, Grid3DCU& grid,
                         const TestInput& input, int rank, Check& check) {
  const cudaStream_t stream = fields.gpuSolverStream();
  // non-uniform B and density, so the conductor rows are not trivial
  initializeManufacturedFields(fields, grid, input);
  fields.gpuSolverSyncH2D(stream);
  prepareManufacturedMoments(fields);

  const int n = maxwellVectorSize(grid);
  const int nxn = grid.getNXN(), nyn = grid.getNYN(), nzn = grid.getNZN();
  std::vector<cudaSolverType> hx(n);
  std::mt19937_64 gen(1234 + rank);
  std::uniform_real_distribution<double> uniform(-1.0, 1.0);
  for (auto& v : hx)
    v = uniform(gen);
  GPUKrylovVector x(n), image(n), reapplied(n), rowsOnly(n);
  GPUFieldArray3 vX(nxn, nyn, nzn), vY(nxn, nyn, nzn), vZ(nxn, nyn, nzn);
  // the allocations zero the arrays outside the solver stream
  cudaErrChk(cudaDeviceSynchronize());
  x.copyFromHost(hx.data(), stream);
  fields.gpuMaxwellImage(image.devPtr(), x.devPtr());

  // the face rows applied to a copy of the image, and to a zero vector
  gpuSolver2Phys3(vX.devPtr(), vY.devPtr(), vZ.devPtr(), x.devPtr(), nxn, nyn,
                  nzn, stream);
  gpuEq(reapplied.devPtr(), image.devPtr(), n, stream);
  rowsOnly.setZero(stream);
  for (GPUKrylovVector* target : {&reapplied, &rowsOnly}) {
    fields.gpuPerfectConductorImages(target->devPtr(), vX, vY, vZ);
    fields.gpuOpenBoundaryInflowEImage(target->devPtr(), vX.devPtr(),
                                       vY.devPtr(), vZ.devPtr(), nxn, nyn, nzn);
  }
  std::vector<cudaSolverType> hImage(n), hReapplied(n), hRows(n);
  image.copyToHost(hImage.data(), stream);
  reapplied.copyToHost(hReapplied.data(), stream);
  rowsOnly.copyToHost(hRows.data(), stream);
  cudaErrChk(cudaStreamSynchronize(stream));

  double local[2] = {0.0, 0.0}; // max |reapplied - image|, rows written
  for (int q = 0; q < n; q++) {
    local[0] = std::max(local[0], std::fabs(hReapplied[q] - hImage[q]));
    local[1] = std::max(local[1], std::fabs(hRows[q]));
  }
  double global[2];
  MPI_Allreduce(local, global, 2, MPI_DOUBLE, MPI_MAX,
                MPIdata::get_PicGlobalComm());
  if (rank == 0) {
    std::cout << "boundary-rows: max |rows reapplied - image| = " << global[0]
              << ", max |rows of a zero image| = " << global[1] << "\n";
    if (global[1] == 0.0)
      check.fail("no boundary row was written: the case has no face rows");
    if (global[0] != 0.0)
      check.fail("gpuMaxwellImage does not hold the boundary rows");
  }
}

void runImageReferenceCase(EMfields3D& fields, Grid3DCU& grid,
                           const TestInput& input, const ImageLayout& layout,
                           int rank, Check& check) {
  const cudaStream_t stream = fields.gpuSolverStream();
  initializeManufacturedFields(fields, grid, input);
  fields.gpuSolverSyncH2D(stream);
  prepareManufacturedMoments(fields);

  const int n = maxwellVectorSize(grid);
  const int nxn = grid.getNXN(), nyn = grid.getNYN(), nzn = grid.getNZN();
  const int nxc = grid.getNXC(), nyc = grid.getNYC(), nzc = grid.getNZC();
  const double ix = grid.get_invdx(), iy = grid.get_invdy(),
               iz = grid.get_invdz();
  const double delt = input.c * input.th * input.dt;
  const size_t nodes = (size_t)nxn * nyn * nzn;
  // X-right outflow face: the cell stencils fill the ghost cell plane, whose
  // face BC keeps it
  const bool outflow = layout.outflowXright &&
                       fields.get_vct().getXright_neighbor() == MPI_PROC_NULL;
  const int iHi = outflow ? nxc - 1 : nxc - 2;
  const int bcGradXright = outflow ? BC_FACE_KEEP : 1;
  const int bcDivXright = outflow ? BC_FACE_KEEP : 2;

  GPUKrylovVector x(n), image(n), ref(n);
  std::vector<GPUFieldArray3> v, D, lap, gd, g;
  for (int c = 0; c < 3; c++) {
    v.emplace_back(nxn, nyn, nzn);
    D.emplace_back(nxn, nyn, nzn);
    lap.emplace_back(nxn, nyn, nzn);
    gd.emplace_back(nxn, nyn, nzn);
  }
  for (int a = 0; a < 9; a++)
    g.emplace_back(nxc, nyc, nzc);
  GPUFieldArray3 divD(nxc, nyc, nzc);
  // the allocations zero the arrays outside the solver stream
  cudaErrChk(cudaDeviceSynchronize());

  std::vector<cudaSolverType> hx(n);
  std::mt19937_64 gen(4321 + rank);
  std::uniform_real_distribution<double> uniform(-1.0, 1.0);
  for (auto& value : hx)
    value = uniform(gen);
  x.copyFromHost(hx.data(), stream);
  fields.gpuMaxwellImage(image.devPtr(), x.devPtr());

  // reference: lap E from the exchanged cell gradients, grad div(μ·E) from the
  // exchanged divergence, then delt² (-lap - grad div D) + D + E
  gpuSolver2Phys3(v[0].devPtr(), v[1].devPtr(), v[2].devPtr(), x.devPtr(), nxn,
                  nyn, nzn, stream);
  GPUFieldArray3* vn[3] = {&v[0], &v[1], &v[2]};
  GPUFieldArray3* Dn[3] = {&D[0], &D[1], &D[2]};
  fields.gpuOutflowGhostNodes(vn);
  for (int c = 0; c < 3; c++)
    gpuGradN2C(g[3 * c].devPtr(), g[3 * c + 1].devPtr(), g[3 * c + 2].devPtr(),
               v[c].devPtr(), nxc, nyc, nzc, ix, iy, iz, stream, 1, iHi);
  fields.gpuMUdot(D[0], D[1], D[2], v[0], v[1], v[2]);
  fields.gpuOutflowGhostNodes(Dn);
  gpuDivN2C(divD.devPtr(), D[0].devPtr(), D[1].devPtr(), D[2].devPtr(), nxc,
            nyc, nzc, ix, iy, iz, stream, 1, iHi);
  cudaSolverType* cellPtr[10];
  for (int a = 0; a < 9; a++)
    cellPtr[a] = g[a].devPtr();
  cellPtr[9] = divD.devPtr();
  fields.gpuOutflowGhostRows(cellPtr, 10);
  fields.gpuCommunicateCenterBC_9(nxc, nyc, nzc, g[0], g[1], g[2], g[3], g[4],
                                  g[5], g[6], g[7], g[8], bcGradXright, 1, 1, 1,
                                  1, 1);
  for (int c = 0; c < 3; c++)
    gpuDivC2N(lap[c].devPtr(), g[3 * c].devPtr(), g[3 * c + 1].devPtr(),
              g[3 * c + 2].devPtr(), nxn, nyn, nzn, ix, iy, iz, stream);
  fields.gpuCommunicateCenterBC(nxc, nyc, nzc, divD, bcDivXright, 2, 2, 2, 2,
                                2);
  gpuGradC2N(gd[0].devPtr(), gd[1].devPtr(), gd[2].devPtr(), divD.devPtr(), nxn,
             nyn, nzn, ix, iy, iz, stream);
  gpuNeg3(lap[0].devPtr(), lap[1].devPtr(), lap[2].devPtr(), nodes, stream);
  gpuSub3(lap[0].devPtr(), gd[0].devPtr(), lap[1].devPtr(), gd[1].devPtr(),
          lap[2].devPtr(), gd[2].devPtr(), nodes, stream);
  gpuScale3(lap[0].devPtr(), lap[1].devPtr(), lap[2].devPtr(), delt * delt,
            nodes, stream);
  gpuSumAddTwo3(lap[0].devPtr(), D[0].devPtr(), v[0].devPtr(), lap[1].devPtr(),
                D[1].devPtr(), v[1].devPtr(), lap[2].devPtr(), D[2].devPtr(),
                v[2].devPtr(), nodes, stream);
  gpuPhys2Solver3(ref.devPtr(), lap[0].devPtr(), lap[1].devPtr(),
                  lap[2].devPtr(), nxn, nyn, nzn, stream);
  fields.gpuPerfectConductorImages(ref.devPtr(), v[0], v[1], v[2]);
  fields.gpuOpenBoundaryInflowEImage(ref.devPtr(), v[0].devPtr(), v[1].devPtr(),
                                     v[2].devPtr(), nxn, nyn, nzn);

  std::vector<cudaSolverType> hImage(n), hRef(n);
  image.copyToHost(hImage.data(), stream);
  ref.copyToHost(hRef.data(), stream);
  cudaErrChk(cudaStreamSynchronize(stream));
  // max |image - ref|, max |ref|, max |image - blocking image|, split ranks
  double local[4] = {0.0, 0.0, 0.0, 0.0};
  for (int q = 0; q < n; q++) {
    local[0] = std::max(local[0], std::fabs(hImage[q] - hRef[q]));
    local[1] = std::max(local[1], std::fabs(hRef[q]));
  }
  GPUKrylovVector blocking(n);
  fields.gpuMaxwellImage<EMfields3D::HaloMode::Blocking>(blocking.devPtr(),
                                                         x.devPtr());
  std::vector<cudaSolverType> hBlocking(n);
  blocking.copyToHost(hBlocking.data(), stream);
  cudaErrChk(cudaStreamSynchronize(stream));
  for (int q = 0; q < n; q++)
    local[2] = std::max(local[2], std::fabs(hImage[q] - hBlocking[q]));
  local[3] = fields.gpuMaxwellImageBoxes().split ? 1.0 : 0.0;
  double globalMax[3], splitRanks = 0.0;
  MPI_Allreduce(local, globalMax, 3, MPI_DOUBLE, MPI_MAX,
                MPIdata::get_PicGlobalComm());
  MPI_Allreduce(&local[3], &splitRanks, 1, MPI_DOUBLE, MPI_SUM,
                MPIdata::get_PicGlobalComm());
  if (rank == 0) {
    const double rel = globalMax[0] / globalMax[1];
    std::cout << "image-reference [" << layout.name << ", HaloOverlap "
              << input.haloOverlap
              << "]: max |image - reference| / max |reference| = " << rel
              << "; split on " << splitRanks
              << " ranks, max |image - blocking image| = " << globalMax[2]
              << "\n";
    if (!(rel < 1e-13))
      check.fail("gpuMaxwellImage differs from the step-by-step image");
    const double expected =
        input.haloOverlap && layout.split ? kImageLayoutRanks : 0.0;
    if (splitRanks != expected)
      check.fail("the image does not overlap its exchange as expected");
    if (globalMax[2] != 0.0)
      check.fail("the image differs from the blocking image");
  }
}

/** The image-reference case on one layout and halo mode; failures of this
 *  rank. */
int runImageLayout(const ImageLayout& layout, const TestInput& baseInput,
                   int haloOverlap, int rank, int nprocs) {
  if (layout.ranks[0] * layout.ranks[1] * layout.ranks[2] != kImageLayoutRanks)
    throw std::logic_error("an image layout must have 8 ranks");
  TestInput input = baseInput;
  input.nxc = layout.cells[0];
  input.nyc = layout.cells[1];
  input.nzc = layout.cells[2];
  input.xlen = layout.ranks[0];
  input.ylen = layout.ranks[1];
  input.zlen = layout.ranks[2];
  input.haloOverlap = haloOverlap;
  ConfigFile config = input.config(TestCase::ImageReference);
  const char* periodicKeys[3][2] = {{"PERIODICX", "PERIODICX_P"},
                                    {"PERIODICY", "PERIODICY_P"},
                                    {"PERIODICZ", "PERIODICZ_P"}};
  for (int a = 0; a < 3; a++)
    for (const char* key : periodicKeys[a])
      config.add(key, std::strchr(layout.periodic, "xyz"[a]) ? 1 : 0);
  if (layout.outflowXright)
    config.add("bcPfaceXright", 3);

  Collective col(config, "gpuMaxwellSolverTest");
  VCtopology3D vct(col);
  if (nprocs != vct.getNprocs())
    throw std::runtime_error("MPI size does not match generated topology");
  vct.setup_vctopology(MPIdata::get_PicGlobalComm());
  Grid3DCU grid(&col, &vct);
  EMfields3D fields(&col, &grid, &vct);
  fields.gpuSolverAllocate();

  Check check;
  check.rank = rank;
  check.tolerance = input.comparisonTolerance;
  runImageReferenceCase(fields, grid, input, layout, rank, check);
  return check.failures;
}

WaveLevelResult runObliqueWaveLevel(int cells, const TestInput& baseInput,
                                    int nprocs) {
  TestInput input = baseInput;
  input.Lx = kObliqueWaveLength;
  input.Ly = kObliqueWaveLength;
  input.Lz = kObliqueWaveLength;
  input.nxc = cells;
  input.nyc = cells;
  input.nzc = cells;
  input.th = 1.0;
  input.gmresTolerance = 1.0e-12;

  const double spacing = kObliqueWaveLength / cells;
  const int steps = static_cast<int>(std::lround(
      kObliqueWaveFinalTime * input.c / (kObliqueWaveCourant * spacing)));
  if (steps <= 0) {
    throw std::runtime_error("oblique-wave refinement produced no time steps");
  }
  input.dt = kObliqueWaveFinalTime / steps;
  input.cycles = steps;

  ConfigFile config = input.config(TestCase::ObliqueWave);
  Collective col(config, "gpuMaxwellSolverTest");
  VCtopology3D vct(col);
  if (nprocs != vct.getNprocs()) {
    throw std::runtime_error("MPI size does not match generated topology");
  }
  vct.setup_vctopology(MPIdata::get_PicGlobalComm());

  Grid3DCU grid(&col, &vct);
  EMfields3D fields(&col, &grid, &vct);
  fields.gpuSolverAllocate();

  const ObliqueWave wave = makeObliqueWave(input);
  initializeObliqueWave(fields, grid, wave);
  const cudaStream_t stream = fields.gpuSolverStream();
  fields.gpuSolverSyncH2D(stream);

  for (int cycle = 0; cycle < steps; cycle++) {
    fields.gpuCalculateE(cycle);
    fields.gpuCalculateB(cycle);
  }

  fields.gpuSolverSyncD2H(stream);
  const WaveState backwardEulerReference =
      backwardEulerWaveState(wave, input.dt, steps);
  const WaveState continuumReference =
      continuumWaveState(wave, kObliqueWaveFinalTime);
  return {cells, steps, input.dt,
          measureObliqueWaveError(fields, grid, wave, backwardEulerReference),
          measureObliqueWaveError(fields, grid, wave, continuumReference)};
}

double observedOrder(double coarseError, double fineError, int coarseCells,
                     int fineCells) {
  return std::log(coarseError / fineError) /
         std::log(static_cast<double>(fineCells) / coarseCells);
}

void requireWaveCondition(Check& check, bool condition,
                          const std::string& message) {
  if (!condition) {
    check.fail(message.c_str());
  }
}

int assessWaveStudy(
    int rank,
    const std::array<WaveLevelResult, kObliqueWaveCells.size()>& levels,
    const WaveStudyCriteria& study) {
  Check check;
  check.rank = rank;
  std::array<double, kObliqueWaveCells.size() - 1> electricOrders = {};
  std::array<double, kObliqueWaveCells.size() - 1> magneticOrders = {};
  std::array<double, kObliqueWaveCells.size() - 1> combinedOrders = {};

  for (size_t level = 0; level < levels.size(); level++) {
    const WaveLevelResult& result = levels[level];
    const WaveError& error = result.*study.errorMember;
    const unsigned long long expectedCount =
        static_cast<unsigned long long>(result.cells) * result.cells *
        result.cells;

    std::ostringstream countMessage;
    countMessage << "oblique-wave " << study.name << " N=" << result.cells
                 << " ownership count mismatch: nodes=" << error.nodeCount
                 << ", centers=" << error.centerCount
                 << ", expected=" << expectedCount;
    requireWaveCondition(check,
                         error.nodeCount == expectedCount &&
                             error.centerCount == expectedCount,
                         countMessage.str());

    const bool validErrors =
        std::isfinite(error.electricL2) && error.electricL2 > 0.0 &&
        std::isfinite(error.magneticL2) && error.magneticL2 > 0.0 &&
        std::isfinite(error.combinedL2) && error.combinedL2 > 0.0 &&
        std::isfinite(error.electricLinf) && error.electricLinf > 0.0 &&
        std::isfinite(error.magneticLinf) && error.magneticLinf > 0.0;
    std::ostringstream validityMessage;
    validityMessage << "oblique-wave " << study.name << " N=" << result.cells
                    << " produced a non-finite or non-positive error";
    requireWaveCondition(check, validErrors, validityMessage.str());

    if (level == 0) {
      continue;
    }

    const WaveLevelResult& coarse = levels[level - 1];
    const WaveError& coarseError = coarse.*study.errorMember;
    electricOrders[level - 1] = observedOrder(
        coarseError.electricL2, error.electricL2, coarse.cells, result.cells);
    magneticOrders[level - 1] = observedOrder(
        coarseError.magneticL2, error.magneticL2, coarse.cells, result.cells);
    combinedOrders[level - 1] = observedOrder(
        coarseError.combinedL2, error.combinedL2, coarse.cells, result.cells);

    std::ostringstream monotonicityMessage;
    monotonicityMessage << "oblique-wave " << study.name
                        << " errors did not all decrease from N="
                        << coarse.cells << " to N=" << result.cells;
    requireWaveCondition(check,
                         error.electricL2 < coarseError.electricL2 &&
                             error.magneticL2 < coarseError.magneticL2 &&
                             error.combinedL2 < coarseError.combinedL2 &&
                             error.electricLinf < coarseError.electricLinf &&
                             error.magneticLinf < coarseError.magneticLinf,
                         monotonicityMessage.str());

    std::ostringstream orderMessage;
    orderMessage << "oblique-wave " << study.name
                 << " orders from N=" << coarse.cells
                 << " to N=" << result.cells
                 << " are E=" << electricOrders[level - 1]
                 << ", B=" << magneticOrders[level - 1]
                 << ", combined=" << combinedOrders[level - 1]
                 << "; expected each in [" << study.minimumOrder << ", "
                 << study.maximumOrder << "] for th=1";
    requireWaveCondition(check,
                         std::isfinite(electricOrders[level - 1]) &&
                             std::isfinite(magneticOrders[level - 1]) &&
                             std::isfinite(combinedOrders[level - 1]) &&
                             electricOrders[level - 1] >= study.minimumOrder &&
                             electricOrders[level - 1] <= study.maximumOrder &&
                             magneticOrders[level - 1] >= study.minimumOrder &&
                             magneticOrders[level - 1] <= study.maximumOrder &&
                             combinedOrders[level - 1] >= study.minimumOrder &&
                             combinedOrders[level - 1] <= study.maximumOrder,
                         orderMessage.str());
  }

  const WaveError& fineError = levels.back().*study.errorMember;
  std::ostringstream fineMessage;
  fineMessage << "oblique-wave " << study.name << " finest combined error "
              << fineError.combinedL2 << " exceeds " << study.maximumFineL2;
  requireWaveCondition(check, fineError.combinedL2 < study.maximumFineL2,
                       fineMessage.str());

  std::ostringstream fineMaxMessage;
  fineMaxMessage << "oblique-wave " << study.name
                 << " finest Linf errors E=" << fineError.electricLinf
                 << ", B=" << fineError.magneticLinf << " exceed "
                 << study.maximumFineLinf;
  requireWaveCondition(check,
                       fineError.electricLinf < study.maximumFineLinf &&
                           fineError.magneticLinf < study.maximumFineLinf,
                       fineMaxMessage.str());

  std::cout << "Oblique-wave " << study.name << " convergence ("
            << study.reference << ", th=1, expected order "
            << study.expectedOrder << "):\n"
            << "  N      dt  steps       relL2(E)       relL2(B)"
               "    relL2(E,B)      p(E)      p(B)    p(E,B)"
               "     Linf(E)     Linf(B)\n";
  for (size_t level = 0; level < levels.size(); level++) {
    const WaveLevelResult& result = levels[level];
    const WaveError& error = result.*study.errorMember;
    std::cout << std::scientific << std::setprecision(6) << std::setw(4)
              << result.cells << "  " << std::setw(10) << result.dt << "  "
              << std::setw(5) << result.steps << "  " << std::setw(13)
              << error.electricL2 << "  " << std::setw(13) << error.magneticL2
              << "  " << std::setw(13) << error.combinedL2 << "  ";
    if (level == 0) {
      std::cout << "         -         -         -";
    } else {
      std::cout << std::setw(10) << electricOrders[level - 1] << "  "
                << std::setw(8) << magneticOrders[level - 1] << "  "
                << std::setw(8) << combinedOrders[level - 1];
    }
    std::cout << "  " << std::setw(11) << error.electricLinf << "  "
              << std::setw(11) << error.magneticLinf << "\n";
  }
  std::cout << std::defaultfloat;

  return check.failures;
}

int runObliqueWaveConvergence(int rank, int nprocs,
                              const TestInput& baseInput) {
  std::array<WaveLevelResult, kObliqueWaveCells.size()> levels;
  for (size_t level = 0; level < levels.size(); level++) {
    levels[level] =
        runObliqueWaveLevel(kObliqueWaveCells[level], baseInput, nprocs);
  }

  if (rank != 0) {
    return 0;
  }

  const WaveStudyCriteria spatialStudy = {
      "spatial-only",
      "continuous-space backward-Euler reference",
      2.0,
      kMinimumSpatialOrder,
      kMaximumSpatialOrder,
      kMaximumFineSpatialL2,
      kMaximumFineSpatialLinf,
      &WaveLevelResult::spatialError};
  const WaveStudyCriteria spaceTimeStudy = {"space-time",
                                            "continuous Maxwell reference",
                                            1.0,
                                            kMinimumSpaceTimeOrder,
                                            kMaximumSpaceTimeOrder,
                                            kMaximumFineSpaceTimeL2,
                                            kMaximumFineSpaceTimeLinf,
                                            &WaveLevelResult::spaceTimeError};

  return assessWaveStudy(rank, levels, spatialStudy) +
         assessWaveStudy(rank, levels, spaceTimeStudy);
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

  case TestCase::BoundaryRows:
    runBoundaryRowsCase(fields, grid, input, rank, check);
    break;

  case TestCase::ObliqueWave:
    throw std::logic_error(
        "oblique-wave refinement must run through its multi-grid driver");

  case TestCase::ImageReference:
    throw std::logic_error("image-reference must run through its layouts");
  }

  return check.failures;
}

int runCase(TestCase testCase, int rank, int nprocs, const TestInput& input) {
  if (testCase == TestCase::ObliqueWave ||
      testCase == TestCase::ImageReference) {
    int localFailures = 0;
    if (testCase == TestCase::ObliqueWave)
      localFailures = runObliqueWaveConvergence(rank, nprocs, input);
    else
      for (int haloOverlap = 0; haloOverlap < 2; haloOverlap++)
        for (const ImageLayout& layout : kImageLayouts)
          localFailures +=
              runImageLayout(layout, input, haloOverlap, rank, nprocs);
    int globalFailures = 0;
    MPI_Allreduce(&localFailures, &globalFailures, 1, MPI_INT, MPI_SUM,
                  MPIdata::get_PicGlobalComm());
    return globalFailures;
  }

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
    TestInput input;

    if (nprocs != input.mpiRanks()) {
      if (rank == 0) {
        std::cerr << "GPU Maxwell solver test requires " << input.mpiRanks()
                  << " MPI ranks for the configured " << input.xlen << "x"
                  << input.ylen << "x" << input.zlen
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
    input.haloOverlap = selection.haloOverlap ? 1 : 0;
    int failures = 0;
    const int localInterpolationFailures =
        runInterpolationRangeComplementRegression(rank);
    int interpolationFailures = 0;
    MPI_Allreduce(&localInterpolationFailures, &interpolationFailures, 1,
                  MPI_INT, MPI_SUM, MPIdata::get_PicGlobalComm());
    failures += interpolationFailures;
    if (rank == 0) {
      reportInterpolationResult(interpolationFailures);
    }
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
      const int caseFailures =
          runCase(selection.singleCase, rank, nprocs, input);
      failures += caseFailures;
      if (rank == 0) {
        reportCaseResult(selection.singleCase, caseFailures);
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
