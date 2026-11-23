/**
 * Run with mpirun -n 8 ./gpuMaxwellSolverTest --case all
 * to execute all test cases, or with --case periodic, --case reflective,
 * --case manufactured, --case oblique-wave, --case boundary-rows,
 * --case smoothed-faces, --case inflow-layers, --case conductor-wall,
 * --case conductor-edges, --case conductor-modes, --case image-reference,
 * --case local-image or --case ghost-fill to run a single case.
 *
 * boundary-rows, smoothed-faces, and image-reference where its layout leaves
 * x non-periodic, use an open inflow face at x-left (bcEMface = bcPface = 2,
 * ApplyInflowBcsEImage = 1).
 *   boundary-rows (x-right perfect conductor): gpuMaxwellImage of a random
 *     vector must already hold every boundary row (the identity rows of the
 *     inflow face and of the conductor's E_t), i.e. equal itself after the
 *     row pass is applied to it again.
 *   smoothed-faces (x-right perfect conductor, Smooth < 1, a drift u0 so that
 *     E0 = -(u0 x B0) / c is not zero): after gpuCalculateE the smoothed E
 *     the mover samples must hold the values the face rows impose, E0 on the
 *     inflow face and E_t = 0 on the conductor face, bitwise.
 *   inflow-layers (the smoothed-faces setup, plus open inflow faces at y-left
 *     and y-right, no SAL): every inflow face must hold E0 on its node layers
 *     0..n_layers_sal after gpuCalculateE, in the E Faraday's law uses and in
 *     the smoothed E, and B0 on its cell layers 0..n_layers_sal after
 *     gpuCalculateB, bitwise.
 *   conductor-wall (conductor walls on the non-periodic x, the manufactured
 *     fields, whose B_x varies along the walls, Smooth < 1 and a drift u0
 *     whose E0 is tangential to the walls): over three E and B updates
 *     - the face nodes keep the wall value of B_x, the initial node B_x
 *       averaged to the cell faces and back, and divC2N(B) on the face nodes
 *       its initial value, while B_x of the first cells changes;
 *     - E_t = 0 on the face nodes, in the E Faraday's law uses and in the
 *       smoothed E the mover samples;
 *     - the ghost nodes hold the image of the fields in the wall: E_n and
 *       B_t even (the value of node 2), E_t and B_n odd about the face
 *       value (2 face - node 2);
 *     - with the div(E) cleaning every cycle (odd phi ghosts on the
 *       walls): the stored E^n keeps E_t = 0 on the face
 *       nodes and the image in its ghost nodes, and its Gauss residual
 *       divN2C(E^n) - 4 pi rho_c drops to the cleaning tolerance.
 *   conductor-edges (a closed box, all six faces perfect conductors, the
 *     manufactured fields and moments): every component tangential to a wall
 *     has an identity row with source 0, in the image and in the row pass.
 *     On the nodes of two or three walls (edges, corners) every component is
 *     tangential to a wall, so all three are; on the nodes of one wall the
 *     normal row is the operator's (the interior row of the conductor
 *     image): the row pass does not write it, nor its source.
 *   conductor-modes (vacuum, theta = 1/2, conductor walls at x = 0 and Lx,
 *     y and z periodic, 32 x 32 x 8 cells, two periods): a TM standing wave,
 *     B_z = cos(pi x) cos(2 pi y) cos(w t), whose E_x on the walls is the
 *     largest E_x of the mode (the wall charge, from c curl(B)_n in the
 *     normal row), and a TE one (E_z, no normal E).  The wall E_n must follow
 *     the mode, div E stay at the solver's tolerance and the energy be
 *     conserved, as the theta = 1/2 scheme does in vacuum.
 *   image-reference (x-right open, which keeps its operator rows, or an
 *     outflow face; conductor walls on the non-periodic y and z faces), on the
 *     layouts of kImageLayouts (3D and 2D, flat and self-periodic axes,
 *     odd and minimal subdomains): gpuMaxwellImage of a random vector against
 *     the image assembled step by step: the cell gradients exchanged,
 *     divC2N, MUdot, div D exchanged, gradC2N, delt^2 (-lap - grad div D) +
 *     D + E with the vector kernels, the packing and the identity rows, with
 *     the face rules and rows of the test (not the solver's plan): on a
 *     conductor the image of E and of div D (odd), and D through the image
 *     (D_t = 0 on the wall) before its divergence.  The conductor's normal
 *     row is also checked against an independent form, the interior row
 *     E_n + D_n - delt^2 (lap E_n + d_n div D) from the node values of the
 *     image of E and of D in the wall, on a vector with E_t = 0 on the
 *     walls.  Each
 *     layout runs with HaloOverlap = 0 and 1; with 1 the image overlaps its
 *     exchange where the layout says so, and equals
 *     gpuMaxwellImage<HaloMode::Blocking> bitwise.
 *   local-image (the closed box of conductor-edges): the local operator of
 *     the block-Jacobi preconditioner equals the full image bitwise on every
 *     row whose stencils read no neighbour rank's ghost cell, the conductor
 *     rows included.
 *   image-reference and local-image also check that the operator, boundary
 *     rows included, is linear and not affine: for the full and the local
 *     image A 0 = 0 bitwise and A(a u + b v) = a A u + b A v to round-off,
 *     each image applied right after one of other data (a stale value read
 *     before it is written would show).
 *   ghost-fill (layouts: a closed GEM box, open faces with and without the
 *     SAL, a ForceFree box with walls and open faces; the two cases with
 *     ForceWallB = 1, so that their wall layers are in the plan): every ghost
 *     set of the solver's plan, filled in one launch, equals the face-by-face
 *     passes of its rules on random arrays, bitwise, edges and corners
 *     included; the moment fold equals the face-by-face doubling.  The rules
 *     themselves are checked against Collective's face codes and the
 *     conductor image for the sets without open-face or case-wall passes
 *     (checkRuleTable).
 *
 * --halo-overlap runs the selected cases with HaloOverlap = 1 (the solver
 * steps in HaloMode::Overlap), and checks that the Maxwell image then
 * overlaps its exchange; image-reference always runs both modes.
 */

#include <mpi.h>

#include "Collective.h"
#include "ConfigFile.h"
#include "EMfields3D.h"
#include "GPUBlas.cuh"
#include "GPUStencils.cuh"
#include "Grid3DCU.h"
#include "MPIdata.h"
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
constexpr FieldLabels kLeftWallLabels = {"Ex left wall", "Ey left wall",
                                         "Ez left wall"};
constexpr FieldLabels kRightWallLabels = {"Ex right wall", "Ey right wall",
                                          "Ez right wall"};
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
  SmoothedFaces,
  InflowLayers,
  ConductorWall,
  ConductorEdges,
  ConductorModes,
  ImageReference,
  LocalImage,
  GhostFill
};

constexpr std::array<TestCase, 13> kAllTestCases = {
    TestCase::Periodic,       TestCase::Reflective,
    TestCase::Manufactured,   TestCase::ObliqueWave,
    TestCase::BoundaryRows,   TestCase::SmoothedFaces,
    TestCase::InflowLayers,   TestCase::ConductorWall,
    TestCase::ConductorEdges, TestCase::ConductorModes,
    TestCase::ImageReference, TestCase::LocalImage,
    TestCase::GhostFill};

// conductor-modes acceptance.  Measured (2026-11-16): TM wall E_n / exact
// 1.0005 (the discrete mode), div E 5e-14 and energy 6e-15 relative for TM
// and TE.  The normal row without curl B kept the wall E_n at 0, and lost
// 54 % of the energy with div E of 10-22 in the wall cells.
constexpr double kModeWallMin = 0.98;
constexpr double kModeWallMax = 1.02;
constexpr double kModeDivMax = 1.0e-9;
constexpr double kModeEnergyDev = 1.0e-9;

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
  // smoothed-faces and inflow-layers: drift of species 0 along x, and the
  // smoothing
  double driftX = 0.1;
  double smooth = 0.5;
  int smoothPasses = 3;
  // inflow-layers: n_layers_sal
  int inflowLayers = 3;

  /** E0 = -(u0 x B0) / c with u0 = (driftX, 0, 0) (smoothed-faces). */
  Vec3 driftElectric() const {
    return {0.0, driftX * backgroundB[2] / c, -driftX * backgroundB[1] / c};
  }

  int mpiRanks() const { return xlen * ylen * zlen; }
  Vec3 periodicElectric() const { return {exUniform, eyPeriodic, ezPeriodic}; }
  Vec3 reflectiveElectric() const {
    return {exUniform, eyReflective, ezReflective};
  }

  ConfigFile config(TestCase testCase) const {
    const bool openInflowXleft = testCase == TestCase::BoundaryRows ||
                                 testCase == TestCase::SmoothedFaces ||
                                 testCase == TestCase::InflowLayers ||
                                 testCase == TestCase::ImageReference;
    const bool periodicX = testCase != TestCase::Reflective &&
                           testCase != TestCase::ConductorWall &&
                           testCase != TestCase::ConductorEdges &&
                           testCase != TestCase::LocalImage &&
                           testCase != TestCase::ConductorModes &&
                           !openInflowXleft;
    const bool driftAndSmoothing = testCase == TestCase::SmoothedFaces ||
                                   testCase == TestCase::InflowLayers ||
                                   testCase == TestCase::ConductorWall;

    ConfigFile cfg;
    cfg.add("SaveDirName", ".");
    cfg.add("RestartDirName", ".");
    cfg.add("Case", "Default");
    // conductor-wall: the div(E) cleaning on every cycle
    cfg.add("PoissonCorrection",
            testCase == TestCase::ConductorWall ? "yes" : "no");
    cfg.add("PoissonCorrectionCycle", 1);
    cfg.add("divBCorrection", "no");
    cfg.add("FieldWriteMethod", "pvtk");
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
    cfg.add("qom", qom);
    // one species: one value in each per-species list
    addConfigValues(cfg, kZeroSpeciesKeys, 0);
    addConfigValues(cfg, kUnitParticleKeys, 1);

    addConfigValues(cfg, kPhiBoundaryKeys, 1);
    addConfigValues(cfg, kEmBoundaryKeys, 0);
    addConfigValues(cfg, kParticleBoundaryKeys, 1);
    // the open faces of these cases without the SAL (on by default)
    cfg.add("yes_sal", 0);
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
    if (testCase == TestCase::InflowLayers) {
      // open inflow faces at y-left and y-right too, without the SAL
      cfg.add("bcEMfaceYleft", 2);
      cfg.add("bcEMfaceYright", 2);
      cfg.add("bcPfaceYleft", 2);
      cfg.add("bcPfaceYright", 2);
      cfg.add("PERIODICY", 0);
      cfg.add("PERIODICY_P", 0);
      cfg.add("n_layers_sal", inflowLayers);
    }
    if (testCase == TestCase::ConductorEdges ||
        testCase == TestCase::LocalImage) {
      // a closed box: conductors on every face (x from periodicX)
      cfg.add("PERIODICY", 0);
      cfg.add("PERIODICY_P", 0);
      cfg.add("PERIODICZ", 0);
      cfg.add("PERIODICZ_P", 0);
    }
    if (driftAndSmoothing) {
      cfg.add("Smooth", smooth);
      cfg.add("SmoothNiter", smoothPasses);
      cfg.add("B0x", backgroundB[0]);
      cfg.add("B0y", backgroundB[1]);
      cfg.add("B0z", backgroundB[2]);
      cfg.add("u0", driftX);
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
  PinnedVector<cudaSolverType> oldCenterHost(centerSize);
  PinnedVector<cudaSolverType> newCenterHost(centerSize);
  PinnedVector<cudaSolverType> oldReference(nodeSize);
  PinnedVector<cudaSolverType> newReference(nodeSize);
  PinnedVector<cudaSolverType> resultHost(nodeSize);

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
  case TestCase::SmoothedFaces:
    return "smoothed-faces";
  case TestCase::InflowLayers:
    return "inflow-layers";
  case TestCase::ConductorWall:
    return "conductor-wall";
  case TestCase::ConductorEdges:
    return "conductor-edges";
  case TestCase::ConductorModes:
    return "conductor-modes";
  case TestCase::ImageReference:
    return "image-reference";
  case TestCase::LocalImage:
    return "local-image";
  case TestCase::GhostFill:
    return "ghost-fill";
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
            "manufactured, oblique-wave, boundary-rows, smoothed-faces, "
            "inflow-layers, conductor-wall, conductor-edges, "
            "conductor-modes, image-reference, local-image or ghost-fill");
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
      if (std::strcmp(value, "smoothed-faces") == 0) {
        return {false, TestCase::SmoothedFaces};
      }
      if (std::strcmp(value, "inflow-layers") == 0) {
        return {false, TestCase::InflowLayers};
      }
      if (std::strcmp(value, "conductor-wall") == 0) {
        return {false, TestCase::ConductorWall};
      }
      if (std::strcmp(value, "conductor-edges") == 0) {
        return {false, TestCase::ConductorEdges};
      }
      if (std::strcmp(value, "conductor-modes") == 0) {
        return {false, TestCase::ConductorModes};
      }
      if (std::strcmp(value, "image-reference") == 0) {
        return {false, TestCase::ImageReference};
      }
      if (std::strcmp(value, "local-image") == 0) {
        return {false, TestCase::LocalImage};
      }
      if (std::strcmp(value, "ghost-fill") == 0) {
        return {false, TestCase::GhostFill};
      }
      throw std::runtime_error(std::string("unknown --case '") + value +
                               "'; expected all, periodic, reflective, "
                               "manufactured, oblique-wave, boundary-rows, "
                               "smoothed-faces, inflow-layers, "
                               "conductor-wall, conductor-edges, "
                               "conductor-modes, image-reference, "
                               "local-image or ghost-fill");
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

// The x walls are perfect conductors: a uniform E keeps its value inside,
// gets E_t = 0 on the face nodes, and the ghost nodes hold its image in the
// wall: E_n even (the value of node 2), E_t odd (minus the value of node 2).
void checkReflective(EMfields3D& fields, Check& check, const TestInput& input) {
  const VirtualTopology3D& vct = fields.get_vct();
  const FieldArray electric = electricFields(fields);
  const bool left = vct.getXleft_neighbor() == MPI_PROC_NULL;
  const bool right = vct.getXright_neighbor() == MPI_PROC_NULL;
  const size_t n = electric[0].dim1();
  const Vec3 e = input.reflectiveElectric();
  const Vec3 wall = {e[0], 0.0, 0.0};
  const Vec3 image = {e[0], -e[1], -e[2]};

  FieldBounds inner = interiorBounds(electric[0]);
  if (left)
    inner.iBegin = 2;
  if (right)
    inner.iEnd = n - 2;
  checkConstantElectric(electric, check, e, inner, kElectricLabels);

  if (left) {
    checkConstantElectric(electric, check, wall, xFaceBounds(electric[0], 1),
                          kLeftWallLabels);
    checkConstantElectric(electric, check, image, xFaceBounds(electric[0], 0),
                          kLeftGhostLabels);
  }
  if (right) {
    checkConstantElectric(electric, check, wall,
                          xFaceBounds(electric[0], n - 2), kRightWallLabels);
    checkConstantElectric(electric, check, image,
                          xFaceBounds(electric[0], n - 1), kRightGhostLabels);
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
  // the page-locked host slot of the result
  PinnedVector<cudaSolverType> result = PinnedVector<cudaSolverType>(1);
};

double localNorm2(const GPUKrylovVector& values, GpuReductionScratch& scratch,
                  cudaStream_t stream) {
  return gpuNorm2(values.devPtr(), values.size(), scratch.ptr,
                  scratch.result.data(), stream);
}

double localNorm2Sum(const ConstGpuFieldPtrs& fields,
                     GpuReductionScratch& scratch, cudaStream_t stream) {
  double sum = 0.0;
  for (const GPUFieldArray3* field : fields) {
    sum += gpuNorm2(field->devPtr(), field->size(), scratch.ptr,
                    scratch.result.data(), stream);
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

void communicateElectricBoundaries(EMfields3D& fields, Grid3DCU&) {
  cudaSolverType* const E[3] = {fields.gpuEx().devPtr(),
                                fields.gpuEy().devPtr(),
                                fields.gpuEz().devPtr()};
  fields.exchangeAndFill(GhostSet::ENode, E, 3);
}

void zeroGpuFields(GpuFieldPtrs fields, cudaStream_t stream) {
  for (GPUFieldArray3* field : fields) {
    field->setAll(0.0, stream);
  }
}

/** The size values of the device array d, once the work queued on stream
 *  is done. */
PinnedVector<cudaSolverType> toHost(const cudaSolverType* d, size_t size,
                                    cudaStream_t stream) {
  PinnedVector<cudaSolverType> h(size);
  cudaErrChk(cudaStreamSynchronize(stream));
  cudaErrChk(cudaMemcpy(h.data(), d, size * sizeof(cudaSolverType),
                        cudaMemcpyDeviceToHost));
  return h;
}

/** n values uniform in [-1, 1), from a generator seeded with seed. */
PinnedVector<cudaSolverType> randomValues(size_t n, unsigned seed) {
  std::mt19937_64 gen(seed);
  std::uniform_real_distribution<double> uniform(-1.0, 1.0);
  PinnedVector<cudaSolverType> v(n);
  for (auto& value : v)
    value = uniform(gen);
  return v;
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
  fields.gpuAdvanceE(); // E^n -> E(n+1), normally done by gpuCalculateB
  fields.gpuSolverSyncD2H(FieldOutputArrays::all(), stream);
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
  fields.gpuSolverSyncD2H(FieldOutputArrays::all(), stream);
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
  fields.gpuAdvanceE(); // E^n -> E(n+1), normally done by gpuCalculateB
  fields.gpuSolverSyncD2H(FieldOutputArrays::all(), stream);
  checkManufactured(fields, check, grid, input);
}

void runBoundaryRowsCase(EMfields3D& fields, Grid3DCU& grid,
                         const TestInput& input, int rank, Check& check) {
  const cudaStream_t stream = fields.gpuSolverStream();
  // non-uniform B and density, so the operator (mu) is not trivial
  initializeManufacturedFields(fields, grid, input);
  fields.gpuSolverSyncH2D(stream);
  prepareManufacturedMoments(fields);

  const int n = maxwellVectorSize(grid);
  const PinnedVector<cudaSolverType> hx = randomValues(n, 1234 + rank);
  GPUKrylovVector x(n), image(n), reapplied(n), rowsOnly(n);
  // the allocations zero the arrays outside the solver stream
  cudaErrChk(cudaDeviceSynchronize());
  x.copyFromHost(hx.data(), stream);
  fields.gpuMaxwellImage(image.devPtr(), x.devPtr());

  // every boundary row (identity rows of the inflow face and of the
  // conductor's E_t) applied again to a copy of the image, and to a zero
  // vector
  gpuEq(reapplied.devPtr(), image.devPtr(), n, stream);
  rowsOnly.setZero(stream);
  for (GPUKrylovVector* target : {&reapplied, &rowsOnly})
    fields.gpuBoundaryRows(target->devPtr());
  PinnedVector<cudaSolverType> hImage(n), hReapplied(n), hRows(n);
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

void runSmoothedFacesCase(EMfields3D& fields, Grid3DCU& grid,
                          const TestInput& input, int rank, Check& check) {
  const cudaStream_t stream = fields.gpuSolverStream();
  initializeManufacturedFields(fields, grid, input);
  fields.gpuSolverSyncH2D(stream);
  prepareManufacturedMoments(fields);
  fields.gpuCalculateE(0);

  const int nxn = grid.getNXN(), nyn = grid.getNYN(), nzn = grid.getNZN();
  const size_t size = (size_t)nxn * nyn * nzn;
  const std::array<PinnedVector<cudaSolverType>, 3> e = {
      toHost(fields.gpuExForPcls().devPtr(), size, stream),
      toHost(fields.gpuEyForPcls().devPtr(), size, stream),
      toHost(fields.gpuEzForPcls().devPtr(), size, stream)};

  // max |E - value| over the face plane i of the components [c0, 3)
  const Vec3 e0 = input.driftElectric();
  auto faceDeviation = [&](int i, int c0, const Vec3& value) {
    double dev = 0.0;
    for (int j = 1; j < nyn - 1; j++)
      for (int k = 1; k < nzn - 1; k++)
        for (int c = c0; c < 3; c++)
          dev = std::max(
              dev, std::fabs(e[c][((size_t)i * nyn + j) * nzn + k] - value[c]));
    return dev;
  };
  const VirtualTopology3D& vct = fields.get_vct();
  // the inflow face holds E0; the conductor face E_t = 0 (a wall at rest,
  // whatever the plasma drift)
  double local[2] = {0.0, 0.0}; // inflow face, conductor face (tangential)
  if (vct.getXleft_neighbor() == MPI_PROC_NULL)
    local[0] = faceDeviation(1, 0, e0);
  if (vct.getXright_neighbor() == MPI_PROC_NULL)
    local[1] = faceDeviation(nxn - 2, 1, {0.0, 0.0, 0.0});
  double global[2];
  MPI_Allreduce(local, global, 2, MPI_DOUBLE, MPI_MAX,
                MPIdata::get_PicGlobalComm());
  if (rank == 0) {
    std::cout << "smoothed-faces: max |E - E0| on the inflow face = "
              << global[0]
              << ", max |E_t| on the conductor face = " << global[1] << "\n";
    if (!fields.smoothingOn())
      check.fail("the case runs without smoothing");
    if (global[0] != 0.0 || global[1] != 0.0)
      check.fail("the smoothed E does not hold the face values of the rows");
  }
}

void runInflowLayersCase(EMfields3D& fields, Grid3DCU& grid,
                         const TestInput& input, int rank, Check& check) {
  const cudaStream_t stream = fields.gpuSolverStream();
  initializeManufacturedFields(fields, grid, input);
  fields.gpuSolverSyncH2D(stream);
  prepareManufacturedMoments(fields);
  fields.gpuCalculateE(0);

  // max |a - value| over the layers 0..L of face f (0 x-left, 2 y-left,
  // 3 y-right) of an array with extents n, on the interior of the other axes.
  // For the node E (onNodes) also away from the face node planes of the
  // other physical faces: where the y inflow layers cross the x-right
  // conductor, they overwrite the wall's E_t = 0 after the solve, which the
  // test does not assert as correct (kept as it is).
  const int L = input.inflowLayers;
  const VirtualTopology3D& topo = fields.get_vct();
  const bool physicalLo[3] = {topo.getXleft_neighbor() == MPI_PROC_NULL,
                              topo.getYleft_neighbor() == MPI_PROC_NULL,
                              topo.getZleft_neighbor() == MPI_PROC_NULL};
  const bool physicalHi[3] = {topo.getXright_neighbor() == MPI_PROC_NULL,
                              topo.getYright_neighbor() == MPI_PROC_NULL,
                              topo.getZright_neighbor() == MPI_PROC_NULL};
  auto layerDeviation = [&](const PinnedVector<cudaSolverType>& a,
                            const int n[3], int f, double value, bool onNodes) {
    const int dir = f / 2;
    const int first = f % 2 ? n[dir] - L - 1 : 0;
    const int last = f % 2 ? n[dir] - 1 : L;
    int lo[3] = {1, 1, 1}, hi[3] = {n[0] - 2, n[1] - 2, n[2] - 2};
    for (int a = 0; a < 3 && onNodes; a++)
      if (a != dir) {
        lo[a] += physicalLo[a] ? 1 : 0;
        hi[a] -= physicalHi[a] ? 1 : 0;
      }
    lo[dir] = first;
    hi[dir] = last;
    double dev = 0.0;
    for (int i = lo[0]; i <= hi[0]; i++)
      for (int j = lo[1]; j <= hi[1]; j++)
        for (int k = lo[2]; k <= hi[2]; k++)
          dev = std::max(
              dev, std::fabs(a[((size_t)i * n[1] + j) * n[2] + k] - value));
    return dev;
  };
  const VirtualTopology3D& vct = fields.get_vct();
  const bool owns[6] = {vct.getXleft_neighbor() == MPI_PROC_NULL,
                        false,
                        vct.getYleft_neighbor() == MPI_PROC_NULL,
                        vct.getYright_neighbor() == MPI_PROC_NULL,
                        false,
                        false};
  // the deviation of three field components from a uniform value over the
  // layers of every inflow face this rank owns
  auto inflowDeviation = [&](GPUFieldArray3* const a[3], const int n[3],
                             const Vec3& value, bool onNodes) {
    const size_t size = (size_t)n[0] * n[1] * n[2];
    double dev = 0.0;
    for (int c = 0; c < 3; c++) {
      const PinnedVector<cudaSolverType> h =
          toHost(a[c]->devPtr(), size, stream);
      for (int f = 0; f < 6; f++)
        if (owns[f])
          dev = std::max(dev, layerDeviation(h, n, f, value[c], onNodes));
    }
    return dev;
  };

  const int nodes[3] = {grid.getNXN(), grid.getNYN(), grid.getNZN()};
  const int cells[3] = {grid.getNXC(), grid.getNYC(), grid.getNZC()};
  const Vec3 e0 = input.driftElectric();
  GPUFieldArray3* const solved[3] = {&fields.gpuExth(), &fields.gpuEyth(),
                                     &fields.gpuEzth()};
  GPUFieldArray3* const smoothed[3] = {
      &fields.gpuExForPcls(), &fields.gpuEyForPcls(), &fields.gpuEzForPcls()};
  double local[3];
  local[0] = inflowDeviation(solved, nodes, e0, true);
  local[1] = inflowDeviation(smoothed, nodes, e0, true);

  fields.gpuCalculateB(0);
  GPUFieldArray3* const bc[3] = {&fields.gpuBxc(), &fields.gpuByc(),
                                 &fields.gpuBzc()};
  local[2] = inflowDeviation(bc, cells, input.backgroundB, false);

  double global[3];
  MPI_Allreduce(local, global, 3, MPI_DOUBLE, MPI_MAX,
                MPIdata::get_PicGlobalComm());
  if (rank == 0) {
    std::cout << "inflow-layers: on the layers 0.." << L
              << " of the inflow faces, max |E - E0| = " << global[0]
              << " (solved), " << global[1]
              << " (smoothed), max |B - B0| = " << global[2] << "\n";
    if (!fields.smoothingOn())
      check.fail("the case runs without smoothing");
    if (global[0] != 0.0 || global[1] != 0.0)
      check.fail("E on the inflow layers is not E0");
    if (global[2] != 0.0)
      check.fail("B on the inflow layers is not B0");
  }
}

void runConductorWallCase(EMfields3D& fields, Grid3DCU& grid,
                          const TestInput& input, int rank, Check& check) {
  const cudaStream_t stream = fields.gpuSolverStream();
  initializeManufacturedFields(fields, grid, input);
  fields.gpuSolverSyncH2D(stream);
  prepareManufacturedMoments(fields);

  const int nxn = grid.getNXN(), nyn = grid.getNYN(), nzn = grid.getNZN();
  const int nxc = grid.getNXC(), nyc = grid.getNYC(), nzc = grid.getNZC();
  const size_t nodes = (size_t)nxn * nyn * nzn;
  const size_t cells = (size_t)nxc * nyc * nzc;
  auto node = [&](int i, int j, int k) {
    return ((size_t)i * nyn + j) * nzn + k;
  };
  auto cell = [&](int i, int j, int k) {
    return ((size_t)i * nyc + j) * nzc + k;
  };
  GPUFieldArray3 div(nxn, nyn, nzn);
  cudaErrChk(cudaDeviceSynchronize());
  auto divergence = [&]() {
    gpuDivC2N(div.devPtr(), fields.gpuBxc().devPtr(), fields.gpuByc().devPtr(),
              fields.gpuBzc().devPtr(), nxn, nyn, nzn, grid.get_invdx(),
              grid.get_invdy(), grid.get_invdz(), stream);
    return toHost(div.devPtr(), nodes, stream);
  };

  // the wall faces of this rank: (face node plane, first cell plane)
  const VirtualTopology3D& vct = fields.get_vct();
  std::vector<std::array<int, 2>> walls;
  if (vct.getXleft_neighbor() == MPI_PROC_NULL)
    walls.push_back({1, 1});
  if (vct.getXright_neighbor() == MPI_PROC_NULL)
    walls.push_back({nxn - 2, nxc - 2});

  // the wall value on the face nodes: W, the mean of the 4 face nodes of a
  // cell face, from the initial node B_x, averaged over the 4 cells of the
  // node
  const PinnedVector<cudaSolverType> bx0 =
      toHost(fields.gpuBxn().devPtr(), nodes, stream);
  auto wallValue = [&](int i, int j, int k) { // W of the cell face (j, k)
    return 0.25 * (bx0[node(i, j, k)] + bx0[node(i, j + 1, k)] +
                   bx0[node(i, j, k + 1)] + bx0[node(i, j + 1, k + 1)]);
  };
  std::vector<double> expected(nodes, 0.0);
  for (const auto& w : walls)
    for (int j = 1; j < nyn - 1; j++)
      for (int k = 1; k < nzn - 1; k++)
        expected[node(w[0], j, k)] =
            0.25 * (wallValue(w[0], j - 1, k - 1) + wallValue(w[0], j, k - 1) +
                    wallValue(w[0], j - 1, k) + wallValue(w[0], j, k));
  const PinnedVector<cudaSolverType> div0 = divergence();
  const PinnedVector<cudaSolverType> bxc0 =
      toHost(fields.gpuBxc().devPtr(), cells, stream);

  // the deviation of the ghost nodes of a node field from its image in the
  // wall: odd about the face value (2 face - node 2) or even (node 2)
  auto imageDeviation = [&](const PinnedVector<cudaSolverType>& a, bool odd) {
    double dev = 0.0;
    for (const auto& w : walls) {
      const int face = w[0];
      const int ghost = face == 1 ? 0 : nxn - 1;
      const int second = face == 1 ? 2 : nxn - 3;
      for (int j = 1; j < nyn - 1; j++)
        for (int k = 1; k < nzn - 1; k++) {
          const double image =
              odd ? 2.0 * a[node(face, j, k)] - a[node(second, j, k)]
                  : a[node(second, j, k)];
          dev = std::max(dev, std::fabs(a[node(ghost, j, k)] - image));
        }
    }
    return dev;
  };
  // max |value| over the face nodes of the walls
  auto wallMax = [&](const PinnedVector<cudaSolverType>& a) {
    double m = 0.0;
    for (const auto& w : walls)
      for (int j = 1; j < nyn - 1; j++)
        for (int k = 1; k < nzn - 1; k++)
          m = std::max(m, std::fabs(a[node(w[0], j, k)]));
    return m;
  };
  if (!fields.smoothingOn())
    check.fail("the case runs without smoothing");
  GPUFieldArray3* const solvedE[3] = {&fields.gpuExth(), &fields.gpuEyth(),
                                      &fields.gpuEzth()};
  GPUFieldArray3* const smoothedE[3] = {
      &fields.gpuExForPcls(), &fields.gpuEyForPcls(), &fields.gpuEzForPcls()};
  GPUFieldArray3* const nodeB[3] = {&fields.gpuBxn(), &fields.gpuByn(),
                                    &fields.gpuBzn()};

  // the Gauss residual divN2C(E^n) - 4 pi rho_c on the interior cells
  GPUFieldArray3 gauss(nxc, nyc, nzc);
  cudaErrChk(cudaDeviceSynchronize());
  auto gaussResidual = [&]() {
    gpuDivN2C(gauss.devPtr(), fields.gpuEx().devPtr(), fields.gpuEy().devPtr(),
              fields.gpuEz().devPtr(), nxc, nyc, nzc, grid.get_invdx(),
              grid.get_invdy(), grid.get_invdz(), stream);
    const PinnedVector<cudaSolverType> d =
        toHost(gauss.devPtr(), cells, stream);
    const PinnedVector<cudaSolverType> rho =
        toHost(fields.gpuRhoc().devPtr(), cells, stream);
    double r = 0.0;
    for (int i = 1; i < nxc - 1; i++)
      for (int j = 1; j < nyc - 1; j++)
        for (int k = 1; k < nzc - 1; k++)
          r = std::max(
              r, std::fabs(d[cell(i, j, k)] - 4.0 * M_PI * rho[cell(i, j, k)]));
    return r;
  };
  GPUFieldArray3* const storedE[3] = {&fields.gpuEx(), &fields.gpuEy(),
                                      &fields.gpuEz()};

  // max over the cycles of |B_x - wall value| and |div - div0| on the face
  // nodes, of the change of B_x in the first cells, of |E_t| on the face
  // nodes, of the ghost nodes' deviation from the image (x is normal); of
  // E^n after its cleaning: |E_t| on the face nodes and the deviation of its
  // ghost nodes; the Gauss residual of E^n before the first cleaning and
  // after each
  double local[9] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
  local[7] = gaussResidual();
  for (int cycle = 0; cycle < 3; cycle++) {
    fields.gpuCalculateE(cycle);
    for (int c = 0; c < 3; c++) {
      const PinnedVector<cudaSolverType> h =
          toHost(storedE[c]->devPtr(), nodes, stream);
      if (c != 0)
        local[5] = std::max(local[5], wallMax(h));
      local[6] = std::max(local[6], imageDeviation(h, c != 0));
    }
    local[8] = std::max(local[8], gaussResidual());
    for (GPUFieldArray3* const* E : {solvedE, smoothedE})
      for (int c = 0; c < 3; c++) {
        const PinnedVector<cudaSolverType> h =
            toHost(E[c]->devPtr(), nodes, stream);
        if (c != 0)
          local[3] = std::max(local[3], wallMax(h));
        local[4] = std::max(local[4], imageDeviation(h, c != 0));
      }
    fields.gpuCalculateB(cycle);
    for (int c = 0; c < 3; c++)
      local[4] = std::max(
          local[4],
          imageDeviation(toHost(nodeB[c]->devPtr(), nodes, stream), c == 0));
    const PinnedVector<cudaSolverType> bx =
        toHost(fields.gpuBxn().devPtr(), nodes, stream);
    const PinnedVector<cudaSolverType> d = divergence();
    const PinnedVector<cudaSolverType> bxc =
        toHost(fields.gpuBxc().devPtr(), cells, stream);
    for (const auto& w : walls)
      for (int j = 1; j < nyn - 1; j++)
        for (int k = 1; k < nzn - 1; k++) {
          const size_t n = node(w[0], j, k);
          local[0] = std::max(local[0], std::fabs(bx[n] - expected[n]));
          local[1] = std::max(local[1], std::fabs(d[n] - div0[n]));
          if (j < nyc - 1 && k < nzc - 1)
            local[2] = std::max(local[2], std::fabs(bxc[cell(w[1], j, k)] -
                                                    bxc0[cell(w[1], j, k)]));
        }
  }
  double global[9];
  MPI_Allreduce(local, global, 9, MPI_DOUBLE, MPI_MAX,
                MPIdata::get_PicGlobalComm());
  if (rank == 0) {
    std::cout << "conductor-wall: over 3 cycles, on the face nodes max |B_x - "
                 "wall value| = "
              << global[0] << ", max |div(B) - initial| = " << global[1]
              << ", max |E_t| = " << global[3]
              << "; max change of B_x in the first cells = " << global[2]
              << "; ghost nodes of E and B, max deviation from the image = "
              << global[4] << "\n";
    if (global[3] != 0.0)
      check.fail("E_t on the conductor face nodes is not 0");
    if (global[4] != 0.0)
      check.fail("the ghost nodes on the conductor faces are not the image "
                 "of the fields");
    if (global[0] > 1.0e-12)
      check.fail("B_x on the conductor face nodes left its wall value");
    if (global[1] > 1.0e-7)
      check.fail("div(B) on the conductor face nodes changed");
    if (global[2] < 1.0e-6)
      check.fail("B_x of the first cells did not change: the case tests "
                 "nothing");
    std::cout << "conductor-wall: div(E) cleaning of E^n, max |E_t| on the "
                 "face nodes = "
              << global[5]
              << ", ghost nodes' max deviation from the image = " << global[6]
              << "; max |div E - 4 pi rho_c| before = " << global[7]
              << ", after = " << global[8] << "\n";
    if (global[5] != 0.0)
      check.fail("the cleaning left E_t on the conductor face nodes of E^n");
    if (global[6] != 0.0)
      check.fail("the ghost nodes of the cleaned E^n are not its image");
    if (!(global[8] < 1.0e-5 * global[7]))
      check.fail("the cleaning did not remove the Gauss residual of E^n");
  }
}

void runConductorEdgesCase(EMfields3D& fields, Grid3DCU& grid,
                           const TestInput& input, int rank, Check& check) {
  const cudaStream_t stream = fields.gpuSolverStream();
  initializeManufacturedFields(fields, grid, input);
  fields.gpuSolverSyncH2D(stream);
  prepareManufacturedMoments(fields);

  const int n[3] = {grid.getNXN(), grid.getNYN(), grid.getNZN()};
  const size_t nodes = (size_t)n[0] * n[1] * n[2];
  auto node = [&](int i, int j, int k) {
    return ((size_t)i * n[1] + j) * n[2] + k;
  };
  const VirtualTopology3D& vct = fields.get_vct();
  const bool lo[3] = {vct.getXleft_neighbor() == MPI_PROC_NULL,
                      vct.getYleft_neighbor() == MPI_PROC_NULL,
                      vct.getZleft_neighbor() == MPI_PROC_NULL};
  const bool hi[3] = {vct.getXright_neighbor() == MPI_PROC_NULL,
                      vct.getYright_neighbor() == MPI_PROC_NULL,
                      vct.getZright_neighbor() == MPI_PROC_NULL};

  // the image of a random vector, the boundary rows of that image on a zero
  // vector, and the values of the identity rows written over a sentinel,
  // all back on the nodes
  const int nK = maxwellVectorSize(grid);
  const PinnedVector<cudaSolverType> hx = randomValues(nK, 777 + rank);
  std::vector<GPUFieldArray3> v, im, rows, src;
  for (int c = 0; c < 3; c++) {
    v.emplace_back(n[0], n[1], n[2]);
    im.emplace_back(n[0], n[1], n[2]);
    rows.emplace_back(n[0], n[1], n[2]);
    src.emplace_back(n[0], n[1], n[2]);
  }
  GPUKrylovVector x(nK), image(nK), rowsOnly(nK);
  // the allocations zero the arrays outside the solver stream
  cudaErrChk(cudaDeviceSynchronize());
  x.copyFromHost(hx.data(), stream);
  PinnedVector<cudaSolverType> sentinel(nodes, 7.0);
  for (int c = 0; c < 3; c++)
    src[c].copyFromHost(sentinel.data());
  // the uploads ran on the default stream, which the non-blocking solver
  // stream does not wait for
  cudaErrChk(cudaDeviceSynchronize());
  fields.gpuMaxwellImage(image.devPtr(), x.devPtr());
  // zeroed on the solver stream, which the row kernels use
  cudaErrChk(cudaMemsetAsync(rowsOnly.devPtr(), 0,
                             rowsOnly.size() * sizeof(cudaSolverType), stream));
  fields.gpuBoundaryRows(rowsOnly.devPtr());
  gpuSolver2Phys3(v[0].devPtr(), v[1].devPtr(), v[2].devPtr(), x.devPtr(), n[0],
                  n[1], n[2], stream);
  gpuSolver2Phys3(im[0].devPtr(), im[1].devPtr(), im[2].devPtr(),
                  image.devPtr(), n[0], n[1], n[2], stream);
  gpuSolver2Phys3(rows[0].devPtr(), rows[1].devPtr(), rows[2].devPtr(),
                  rowsOnly.devPtr(), n[0], n[1], n[2], stream);
  cudaSolverType* const srcPtr[3] = {src[0].devPtr(), src[1].devPtr(),
                                     src[2].devPtr()};
  fields.gpuBoundary().rowValues(srcPtr, stream);
  cudaErrChk(cudaStreamSynchronize(stream));
  std::array<PinnedVector<cudaSolverType>, 3> hv, him, hrows, hs;
  for (int c = 0; c < 3; c++) {
    hv[c] = toHost(v[c].devPtr(), nodes, stream);
    him[c] = toHost(im[c].devPtr(), nodes, stream);
    hrows[c] = toHost(rows[c].devPtr(), nodes, stream);
    hs[c] = toHost(src[c].devPtr(), nodes, stream);
  }

  // [0] tangential rows that are not identity rows with source 0, in the
  // image or in the row pass (edges, corners, faces); [1] edge and corner
  // nodes; [2] normal rows of single walls that the row pass or the identity
  // rows' values wrote (the interior row of the image is the operator's);
  // [3] normal rows of single walls
  long local[4] = {0, 0, 0, 0};
  for (int i = 1; i < n[0] - 1; i++)
    for (int j = 1; j < n[1] - 1; j++)
      for (int k = 1; k < n[2] - 1; k++) {
        const int idx[3] = {i, j, k};
        bool on[3];
        int walls = 0;
        for (int a = 0; a < 3; a++) {
          on[a] = (lo[a] && idx[a] == 1) || (hi[a] && idx[a] == n[a] - 2);
          walls += on[a];
        }
        if (walls == 0)
          continue;
        if (walls >= 2)
          local[1]++;
        const size_t q = node(i, j, k);
        for (int c = 0; c < 3; c++) {
          bool tangential = false;
          for (int a = 0; a < 3; a++)
            tangential = tangential || (on[a] && a != c);
          if (tangential) {
            if (him[c][q] != hv[c][q] || hrows[c][q] != hv[c][q] ||
                hs[c][q] != 0.0)
              local[0]++;
          } else {
            local[2] += hrows[c][q] != 0.0 || hs[c][q] != 7.0;
            local[3]++;
          }
        }
      }
  long global[4];
  MPI_Allreduce(local, global, 4, MPI_LONG, MPI_SUM,
                MPIdata::get_PicGlobalComm());
  if (rank == 0) {
    std::cout << "conductor-edges: " << global[1]
              << " edge and corner nodes; tangential rows that are not "
                 "identity rows with source 0: "
              << global[0] << "; normal rows on single walls: " << global[3]
              << ", written by the row pass: " << global[2] << "\n";
    if (global[0] != 0)
      check.fail("a component tangential to a wall has a non-identity row");
    if (global[1] == 0 || global[3] == 0)
      check.fail("the box has no edge, corner or single-wall node");
    if (global[2] != 0)
      check.fail("the row pass wrote a single wall's normal row");
  }
}

// One vacuum standing wave between the conductor walls at x = 0 and Lx (see
// conductor-modes), two periods.  Returns, over the run: max over time of the
// wall E_n relative to its exact amplitude (TM; 0 for TE), max |div E| on the
// cells relative to max |E| / dx, and max |energy / initial - 1|.
std::array<double, 3> runConductorMode(EMfields3D& fields, Grid3DCU& grid,
                                       const TestInput& input, int rank,
                                       bool tm) {
  const cudaStream_t stream = fields.gpuSolverStream();
  zeroFields(fields);
  const double kx = M_PI / input.Lx, ky = 2.0 * M_PI / input.Ly, c = input.c;
  const double w = c * std::sqrt(kx * kx + ky * ky);
  // t = 0: E = 0; TM B_z = cos(kx x) cos(ky y), TE the B of E_z = sin(kx x)
  // cos(ky y) sin(w t)
  auto mode = [&](const GridPoint& p) -> Vec3 {
    if (tm)
      return {0.0, 0.0, std::cos(kx * p.x) * std::cos(ky * p.y)};
    return {-(c * ky / w) * std::sin(kx * p.x) * std::sin(ky * p.y),
            -(c * kx / w) * std::cos(kx * p.x) * std::cos(ky * p.y), 0.0};
  };
  FieldArray Bn = magneticNodeFields(fields), Bc = magneticCenterFields(fields);
  forEachIndex(fullBounds(Bn[0]), [&](size_t i, size_t j, size_t k) {
    setFieldPoint(Bn, i, j, k, mode(nodePoint(grid, i, j, k)));
  });
  forEachIndex(fullBounds(Bc[0]), [&](size_t i, size_t j, size_t k) {
    setFieldPoint(Bc, i, j, k, mode(centerPoint(grid, i, j, k)));
  });
  fields.gpuSolverSyncH2D(stream);
  prepareManufacturedMoments(fields);

  const int nn[3] = {grid.getNXN(), grid.getNYN(), grid.getNZN()};
  const int nc[3] = {grid.getNXC(), grid.getNYC(), grid.getNZC()};
  const size_t nodes = (size_t)nn[0] * nn[1] * nn[2];
  const size_t cells = (size_t)nc[0] * nc[1] * nc[2];
  auto N = [&](int i, int j, int k) {
    return ((size_t)i * nn[1] + j) * nn[2] + k;
  };
  auto C = [&](int i, int j, int k) {
    return ((size_t)i * nc[1] + j) * nc[2] + k;
  };
  const VirtualTopology3D& vct = fields.get_vct();
  const bool lo = vct.getXleft_neighbor() == MPI_PROC_NULL;
  const bool hi = vct.getXright_neighbor() == MPI_PROC_NULL;
  GPUFieldArray3 div(nc[0], nc[1], nc[2]);
  cudaErrChk(cudaDeviceSynchronize());

  double energy0 = -1.0, wallMax = 0.0, divMax = 0.0, energyDev = 0.0;
  const int steps = (int)std::lround(2.0 * 2.0 * M_PI / w / input.dt);
  for (int cycle = 0; cycle <= steps; cycle++) {
    if (cycle > 0) {
      fields.gpuCalculateE(cycle);
      fields.gpuCalculateB(cycle);
    }
    std::array<PinnedVector<cudaSolverType>, 3> e, b;
    GPUFieldArray3* const E[3] = {&fields.gpuEx(), &fields.gpuEy(),
                                  &fields.gpuEz()};
    GPUFieldArray3* const B[3] = {&fields.gpuBxc(), &fields.gpuByc(),
                                  &fields.gpuBzc()};
    for (int a = 0; a < 3; a++) {
      e[a] = toHost(E[a]->devPtr(), nodes, stream);
      b[a] = toHost(B[a]->devPtr(), cells, stream);
    }
    gpuDivN2C(div.devPtr(), E[0]->devPtr(), E[1]->devPtr(), E[2]->devPtr(),
              nc[0], nc[1], nc[2], grid.get_invdx(), grid.get_invdy(),
              grid.get_invdz(), stream);
    const PinnedVector<cudaSolverType> d = toHost(div.devPtr(), cells, stream);
    // [0] max |E_x| on the walls, [1] max |div E|, [2] max |E|; energy:
    // the cells' B^2 and the owned nodes' E^2 (trapezoidal on the walls)
    double mx[3] = {0.0, 0.0, 0.0}, sum = 0.0;
    for (int j = 1; j < nn[1] - 1; j++)
      for (int k = 1; k < nn[2] - 1; k++) {
        if (lo)
          mx[0] = std::max(mx[0], (double)std::fabs(e[0][N(1, j, k)]));
        if (hi)
          mx[0] = std::max(mx[0], (double)std::fabs(e[0][N(nn[0] - 2, j, k)]));
      }
    for (int i = 1; i < nc[0] - 1; i++)
      for (int j = 1; j < nc[1] - 1; j++)
        for (int k = 1; k < nc[2] - 1; k++) {
          mx[1] = std::max(mx[1], (double)std::fabs(d[C(i, j, k)]));
          for (int a = 0; a < 3; a++)
            sum += b[a][C(i, j, k)] * b[a][C(i, j, k)];
        }
    for (int i = 1; i <= (hi ? nn[0] - 2 : nn[0] - 3); i++)
      for (int j = 1; j <= nn[1] - 3; j++)
        for (int k = 1; k <= nn[2] - 3; k++) {
          const double wgt =
              ((lo && i == 1) || (hi && i == nn[0] - 2)) ? 0.5 : 1.0;
          for (int a = 0; a < 3; a++) {
            sum += wgt * e[a][N(i, j, k)] * e[a][N(i, j, k)];
            mx[2] = std::max(mx[2], (double)std::fabs(e[a][N(i, j, k)]));
          }
        }
    double gmx[3], gsum;
    MPI_Allreduce(mx, gmx, 3, MPI_DOUBLE, MPI_MAX,
                  MPIdata::get_PicGlobalComm());
    MPI_Allreduce(&sum, &gsum, 1, MPI_DOUBLE, MPI_SUM,
                  MPIdata::get_PicGlobalComm());
    if (energy0 < 0.0)
      energy0 = gsum;
    if (tm)
      wallMax = std::max(wallMax, gmx[0] / (c * ky / w));
    if (gmx[2] > 0.0)
      divMax = std::max(divMax, gmx[1] / (gmx[2] * grid.get_invdx()));
    energyDev = std::max(energyDev, std::fabs(gsum / energy0 - 1.0));
  }
  if (rank == 0)
    std::cout << "conductor-modes " << (tm ? "TM" : "TE")
              << ": max wall E_n / exact amplitude = " << wallMax
              << ", max |div E| / (max |E| / dx) = " << divMax
              << ", max |energy / initial - 1| = " << energyDev << "\n";
  return {wallMax, divMax, energyDev};
}

void runConductorModesCase(EMfields3D& fields, Grid3DCU& grid,
                           const TestInput& input, int rank, Check& check) {
  const std::array<double, 3> tm =
      runConductorMode(fields, grid, input, rank, true);
  const std::array<double, 3> te =
      runConductorMode(fields, grid, input, rank, false);
  if (rank == 0) {
    if (tm[0] < kModeWallMin || tm[0] > kModeWallMax)
      check.fail("TM: the wall E_n does not follow the mode");
    if (tm[1] > kModeDivMax || te[1] > kModeDivMax)
      check.fail("div E left the solver's tolerance");
    if (tm[2] > kModeEnergyDev || te[2] > kModeEnergyDev)
      check.fail("the vacuum energy is not conserved");
  }
}

/** The Maxwell operator is linear, boundary rows included: for the full and
 *  the local image, A 0 = 0 bitwise and A(a u + b v) = a A u + b A v to
 *  round-off.  Each image follows one of different data, so a value the
 *  image read before writing it (a stale ghost of its scratch arrays, shared
 *  with the cleaning solves) would make it affine and show here.  Prints
 *  the global maxima and fails the check above round-off. */
void checkImageLinearity(EMfields3D& fields, const Grid3DCU& grid, int rank,
                         const char* label, Check& check) {
  const cudaStream_t stream = fields.gpuSolverStream();
  const int n = maxwellVectorSize(grid);
  const double a = 0.7, b = -1.3;
  const PinnedVector<cudaSolverType> hu = randomValues(n, 5151 + rank);
  const PinnedVector<cudaSolverType> hv = randomValues(n, 6262 + rank);
  PinnedVector<cudaSolverType> hw(n);
  for (int q = 0; q < n; q++)
    hw[q] = a * hu[q] + b * hv[q];
  GPUKrylovVector u(n), v(n), w(n), zero(n), image(n);
  // the allocations zero the vectors outside the solver stream
  cudaErrChk(cudaDeviceSynchronize());
  u.copyFromHost(hu.data(), stream);
  v.copyFromHost(hv.data(), stream);
  w.copyFromHost(hw.data(), stream);
  zero.setZero(stream);
  // [image][0]: max |A 0|, [image][1]: max |A w - (a A u + b A v)| relative
  // to max |a A u + b A v|; image 0 the full, 1 the local operator
  double local[4] = {0.0, 0.0, 0.0, 0.0};
  for (int op = 0; op < 2; op++) {
    auto apply = [&](GPUKrylovVector& x) {
      if (op == 0)
        fields.gpuMaxwellImage(image.devPtr(), x.devPtr());
      else
        fields.gpuMaxwellImageLocal(image.devPtr(), x.devPtr());
      return toHost(image.devPtr(), n, stream);
    };
    const PinnedVector<cudaSolverType> Au = apply(u), A0 = apply(zero),
                                       Av = apply(v), Aw = apply(w);
    double err = 0.0, scale = 0.0;
    for (int q = 0; q < n; q++) {
      const double expected = a * Au[q] + b * Av[q];
      scale = std::max(scale, std::fabs(expected));
      err = std::max(err, std::fabs(Aw[q] - expected));
      local[2 * op] = std::max(local[2 * op], (double)std::fabs(A0[q]));
    }
    local[2 * op + 1] = scale > 0.0 ? err / scale : 0.0;
  }
  double global[4];
  MPI_Allreduce(local, global, 4, MPI_DOUBLE, MPI_MAX,
                MPIdata::get_PicGlobalComm());
  if (rank == 0) {
    std::cout << label << " linearity: max |A 0| = " << global[0] << " (full), "
              << global[2]
              << " (local); max |A(au+bv) - (aAu+bAv)| / max |aAu+bAv| = "
              << global[1] << " (full), " << global[3] << " (local)\n";
    if (global[0] != 0.0 || global[2] != 0.0)
      check.fail("the image of the zero vector is not 0: the operator is "
                 "affine");
    if (global[1] > 1.0e-12 || global[3] > 1.0e-12)
      check.fail("the image is not linear to round-off");
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
  // The reference is assembled from the stencils, the halo exchange and
  // the boundary rules of this test, written from the physics here (not
  // read from the solver's plan).  Faces in Collective's order (XR, XL, YR,
  // YL, ZR, ZL).
  const VirtualTopology3D& vct = fields.get_vct();
  const Collective& col = fields.get_col();
  const int neighbour[6] = {vct.getXright_neighbor(), vct.getXleft_neighbor(),
                            vct.getYright_neighbor(), vct.getYleft_neighbor(),
                            vct.getZright_neighbor(), vct.getZleft_neighbor()};
  const int bcEM[6] = {col.getBcEMfaceXright(), col.getBcEMfaceXleft(),
                       col.getBcEMfaceYright(), col.getBcEMfaceYleft(),
                       col.getBcEMfaceZright(), col.getBcEMfaceZleft()};
  const int bcP[6] = {col.getBcPfaceXright(), col.getBcPfaceXleft(),
                      col.getBcPfaceYright(), col.getBcPfaceYleft(),
                      col.getBcPfaceZright(), col.getBcPfaceZleft()};
  auto physical = [&](int fc) { return neighbour[fc] == MPI_PROC_NULL; };
  auto conductor = [&](int fc) { return physical(fc) && bcEM[fc] == 0; };
  // inflow rows: open, bcPface = 2, not X-right, with ApplyInflowBcsEImage
  auto inflow = [&](int fc) {
    return physical(fc) && fc != 0 && bcEM[fc] == 2 && bcP[fc] == 2 &&
           col.getApplyInflowBcsEImage();
  };
  // X-right outflow face: the cell stencils fill the ghost cell plane, whose
  // face rule keeps it
  const bool outflow = layout.outflowXright && physical(0);
  const int iHi = outflow ? nxc - 1 : nxc - 2;
  // The face rules of the operator's cell arrays (Collective's codes: 0
  // odd, 1 zero, 2 even; kKeep leaves the plane): zero gradients and copied
  // div D on the open faces, the outflow plane kept, and on a conductor the
  // image in the wall (E_n even, E_t odd about the face node; the image
  // charge opposite): grad_d E_c odd when (c == n) == (d == n), even
  // otherwise, and div D odd.
  constexpr int kKeep = -1;
  auto opCode = [&](int a, int fc) {
    if (fc == 0 && outflow)
      return kKeep;
    if (a == 9)
      return bcEM[fc] == 0 ? 0 : 2;
    if (bcEM[fc] != 0)
      return 1;
    const int c = a / 3, d = a % 3, axis = fc / 2;
    return (c == axis) == (d == axis) ? 0 : 2;
  };
  // applied on the host, face by face over the whole ghost planes in the
  // order XL, XR, YL, YR, ZL, ZR (a later face overwrites the edges)
  auto faceRules = [&](GPUFieldArray3& a, int array) {
    const int n[3] = {nxc, nyc, nzc};
    PinnedVector<cudaSolverType> h =
        toHost(a.devPtr(), (size_t)nxc * nyc * nzc, stream);
    auto at = [&](const int c[3]) {
      return ((size_t)c[0] * nyc + c[1]) * nzc + c[2];
    };
    for (int f = 0; f < 6; f++) {
      const int fc = 2 * (f / 2) + (f % 2 ? 0 : 1); // Collective's order
      const int code = opCode(array, fc);
      if (!physical(fc) || code == kKeep)
        continue;
      const int axis = f / 2, u = (axis + 1) % 3, w = (axis + 2) % 3;
      int c[3], first[3];
      c[axis] = f % 2 ? n[axis] - 1 : 0;
      for (c[u] = 0; c[u] < n[u]; c[u]++)
        for (c[w] = 0; c[w] < n[w]; c[w]++) {
          for (int e = 0; e < 3; e++)
            first[e] = c[e];
          first[axis] += f % 2 ? -1 : 1;
          h[at(c)] = code == 0 ? -h[at(first)] : code == 1 ? 0.0 : h[at(first)];
        }
    }
    cudaErrChk(cudaMemcpy(a.devPtr(), h.data(),
                          h.size() * sizeof(cudaSolverType),
                          cudaMemcpyHostToDevice));
    // before the solver stream reads it
    cudaErrChk(cudaDeviceSynchronize());
  };
  // the outflow face's ghost node plane: a copy of the face plane
  auto outflowGhostNodes = [&](std::vector<GPUFieldArray3>& a) {
    if (!outflow)
      return;
    const size_t plane = (size_t)nyn * nzn;
    for (GPUFieldArray3& f : a)
      cudaErrChk(cudaMemcpyAsync(
          f.devPtr() + (nxn - 1) * plane, f.devPtr() + (nxn - 2) * plane,
          plane * sizeof(cudaSolverType), cudaMemcpyDeviceToDevice, stream));
  };
  // the ghost rows of the outflow face's ghost cell plane: a physical face
  // keeps its rule, a periodic image on this rank gives the partner row, a
  // neighbour rank's row is stood in for by the adjacent row; y rows first
  auto outflowGhostRows = [&](cudaSolverType* const* arrays, int count) {
    if (!outflow)
      return;
    const size_t plane = (size_t)nyc * nzc;
    const int self = vct.getCartesian_rank();
    const int side[4] = {vct.getYleft_neighbor(), vct.getYright_neighbor(),
                         vct.getZleft_neighbor(), vct.getZright_neighbor()};
    auto source = [&](int s, int ghost, int extent) {
      if (side[s] == self)
        return ghost == 0 ? extent - 2 : 1;
      return ghost == 0 ? 1 : extent - 2;
    };
    PinnedVector<cudaSolverType> h(plane);
    for (int a = 0; a < count; a++) {
      cudaSolverType* p = arrays[a] + (nxc - 1) * plane;
      cudaErrChk(cudaStreamSynchronize(stream));
      cudaErrChk(cudaMemcpy(h.data(), p, plane * sizeof(cudaSolverType),
                            cudaMemcpyDeviceToHost));
      for (int s = 0; s < 2; s++)
        if (side[s] != MPI_PROC_NULL)
          for (int k = 1; k < nzc - 1; k++) {
            const int jg = s == 0 ? 0 : nyc - 1;
            h[(size_t)jg * nzc + k] = h[(size_t)source(s, jg, nyc) * nzc + k];
          }
      for (int s = 2; s < 4; s++)
        if (side[s] != MPI_PROC_NULL)
          for (int j = 0; j < nyc; j++) {
            const int kg = s == 2 ? 0 : nzc - 1;
            h[(size_t)j * nzc + kg] = h[(size_t)j * nzc + source(s, kg, nzc)];
          }
      cudaErrChk(cudaMemcpy(p, h.data(), plane * sizeof(cudaSolverType),
                            cudaMemcpyHostToDevice));
    }
    // before the solver stream reads them
    cudaErrChk(cudaDeviceSynchronize());
  };

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

  // a random vector with E_t = 0 on the conductor walls, as the solution
  // has: then the image of E in the wall is odd about 0 for E_t, and the
  // normal row can be checked against the image row (below)
  auto krylov = [&](int i, int j, int k) {
    return ((i - 1) * (nyn - 2) + (j - 1)) * (nzn - 2) + (k - 1);
  };
  auto onWall = [&](int axis, int i, int j, int k) {
    const int c[3] = {i, j, k}, nn[3] = {nxn, nyn, nzn};
    return (c[axis] == 1 && conductor(2 * axis + 1)) ||
           (c[axis] == nn[axis] - 2 && conductor(2 * axis));
  };
  PinnedVector<cudaSolverType> hx = randomValues(n, 4321 + rank);
  for (int i = 1; i < nxn - 1; i++)
    for (int j = 1; j < nyn - 1; j++)
      for (int k = 1; k < nzn - 1; k++)
        for (int axis = 0; axis < 3; axis++)
          if (onWall(axis, i, j, k))
            for (int c = 0; c < 3; c++)
              if (c != axis)
                hx[3 * krylov(i, j, k) + c] = 0.0;
  x.copyFromHost(hx.data(), stream);
  fields.gpuMaxwellImage(image.devPtr(), x.devPtr());

  // reference: lap E from the exchanged cell gradients, grad div(μ·E) from the
  // exchanged divergence, then delt² (-lap - grad div D) + D + E
  gpuSolver2Phys3(v[0].devPtr(), v[1].devPtr(), v[2].devPtr(), x.devPtr(), nxn,
                  nyn, nzn, stream);
  outflowGhostNodes(v);
  for (int c = 0; c < 3; c++)
    gpuGradN2C(g[3 * c].devPtr(), g[3 * c + 1].devPtr(), g[3 * c + 2].devPtr(),
               v[c].devPtr(), nxc, nyc, nzc, ix, iy, iz, stream, 1, iHi);
  fields.gpuMUdot(D[0], D[1], D[2], v[0], v[1], v[2]);
  // D through the conductor image, independently of the operator's selects
  // (ImageWalls): its components tangential to a wall written 0 on the face
  // plane (before the outflow copy, which then carries them), then the plain
  // divergence
  {
    cudaSolverType* const Dp[3] = {D[0].devPtr(), D[1].devPtr(), D[2].devPtr()};
    fields.gpuBoundary().zeroTangential(Dp, stream);
  }
  outflowGhostNodes(D);
  gpuDivN2C(divD.devPtr(), D[0].devPtr(), D[1].devPtr(), D[2].devPtr(), nxc,
            nyc, nzc, ix, iy, iz, stream, 1, iHi);
  cudaSolverType* cellPtr[10];
  for (int a = 0; a < 9; a++)
    cellPtr[a] = g[a].devPtr();
  cellPtr[9] = divD.devPtr();
  outflowGhostRows(cellPtr, 10);
  fields.gpuBatchedHaloExchange(cellPtr, 9, nxc, nyc, nzc, true, false, false,
                                false, stream);
  for (int a = 0; a < 9; a++)
    faceRules(g[a], a);
  for (int c = 0; c < 3; c++)
    gpuDivC2N(lap[c].devPtr(), g[3 * c].devPtr(), g[3 * c + 1].devPtr(),
              g[3 * c + 2].devPtr(), nxn, nyn, nzn, ix, iy, iz, stream);
  fields.gpuBatchedHaloExchange(&cellPtr[9], 1, nxc, nyc, nzc, true, false,
                                false, false, stream);
  faceRules(divD, 9);
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

  // the operator data on the host, for the face rows
  std::array<PinnedVector<cudaSolverType>, 3> hv, hD;
  for (int c = 0; c < 3; c++) {
    hv[c] = toHost(v[c].devPtr(), nodes, stream);
    hD[c] = toHost(D[c].devPtr(), nodes, stream);
  }
  auto nodeAt = [&](int i, int j, int k) {
    return ((size_t)i * nyn + j) * nzn + k;
  };
  const double q[3] = {0.25 * ix, 0.25 * iy, 0.25 * iz};
  PinnedVector<cudaSolverType> hImage(n), hRef(n);
  image.copyToHost(hImage.data(), stream);
  ref.copyToHost(hRef.data(), stream);
  cudaErrChk(cudaStreamSynchronize(stream));

  // The identity rows of the reference, node by node, where an inflow face,
  // or a conductor tangential to the component, holds the node.  A
  // conductor's normal row keeps the row assembled above.  Faces as (axis,
  // side): Collective index 2 axis + (low ? 1 : 0).
  const int nn[3] = {nxn, nyn, nzn};
  for (int i = 1; i < nxn - 1; i++)
    for (int j = 1; j < nyn - 1; j++)
      for (int k = 1; k < nzn - 1; k++) {
        const int c3[3] = {i, j, k};
        bool inflowNode = false;
        int cond = 0;
        for (int axis = 0; axis < 3; axis++) {
          const int fc = c3[axis] == 1              ? 2 * axis + 1
                         : c3[axis] == nn[axis] - 2 ? 2 * axis
                                                    : -1;
          if (fc < 0)
            continue;
          inflowNode = inflowNode || inflow(fc);
          if (conductor(fc))
            cond |= 1 << axis;
        }
        for (int c = 0; c < 3; c++)
          if (inflowNode || (cond & ~(1 << c)) != 0)
            hRef[3 * krylov(i, j, k) + c] = hv[c][nodeAt(i, j, k)];
      }

  // The conductor's normal row against an independent form, the interior
  // row of the conductor image, E_n + D_n - delt^2 (lap E_n + d_n div D),
  // with every difference taken on the host from the node values of the
  // image of E and of D = mu E in the wall (ghost node: the normal component
  // even, the tangential ones odd; on the wall E_t = 0 and D_t read as 0),
  // on the wall nodes whose stencils read no other face or rank's ghost
  // (in-plane coordinates in [2, n-3]).
  double independent[2] = {0.0, 0.0}; // max |image - image row|, max |row|
  long wallNodes = 0;
  for (int axis = 0; axis < 3; axis++)
    for (int low = 0; low < 2; low++) {
      const int fc = 2 * axis + low;
      if (!conductor(fc))
        continue;
      const int plane = low ? 1 : nn[axis] - 2;
      const int out = low ? -1 : 1; // the ghost side
      // component c of the node field F at (i, j, k) through the image
      auto img = [&](const std::array<PinnedVector<cudaSolverType>, 3>& F,
                     int c, int i, int j, int k) {
        int p3[3] = {i, j, k};
        if (p3[axis] == plane && c != axis)
          return 0.0;
        double sign = 1.0;
        if (p3[axis] == plane + out) {
          p3[axis] = plane - out;
          sign = c == axis ? 1.0 : -1.0;
        }
        return sign * F[c][nodeAt(p3[0], p3[1], p3[2])];
      };
      // d/d(e) of component c of F at the cell (ci, cj, ck): nodes ci..ci+1
      auto dN2C = [&](const std::array<PinnedVector<cudaSolverType>, 3>& F,
                      int c, int e, int ci, int cj, int ck) {
        double sum = 0.0;
        for (int a = 0; a < 2; a++)
          for (int b = 0; b < 2; b++) {
            int lo3[3] = {ci, cj, ck}, hi3[3] = {ci, cj, ck};
            const int u = (e + 1) % 3, w = (e + 2) % 3;
            lo3[u] += a;
            hi3[u] += a;
            lo3[w] += b;
            hi3[w] += b;
            hi3[e] += 1;
            sum += img(F, c, hi3[0], hi3[1], hi3[2]) -
                   img(F, c, lo3[0], lo3[1], lo3[2]);
          }
        return q[e] * sum;
      };
      // d/d(e) at the node (i, j, k) of the cell function f: cells i-1..i
      auto dC2N = [&](const auto& f, int e, int i, int j, int k) {
        double sum = 0.0;
        for (int a = 0; a < 2; a++)
          for (int b = 0; b < 2; b++) {
            int hi3[3] = {i, j, k}, lo3[3] = {i, j, k};
            const int s1 = (e + 1) % 3, s2 = (e + 2) % 3;
            hi3[s1] -= a;
            lo3[s1] -= a;
            hi3[s2] -= b;
            lo3[s2] -= b;
            lo3[e] -= 1;
            sum += f(hi3[0], hi3[1], hi3[2]) - f(lo3[0], lo3[1], lo3[2]);
          }
        return q[e] * sum;
      };
      const auto divD = [&](int ci, int cj, int ck) {
        return dN2C(hD, 0, 0, ci, cj, ck) + dN2C(hD, 1, 1, ci, cj, ck) +
               dN2C(hD, 2, 2, ci, cj, ck);
      };
      for (int i = 1; i < nxn - 1; i++)
        for (int j = 1; j < nyn - 1; j++)
          for (int k = 1; k < nzn - 1; k++) {
            const int p3[3] = {i, j, k};
            bool inside = p3[axis] == plane;
            for (int e = 0; e < 3; e++)
              if (e != axis)
                inside = inside && p3[e] >= 2 && p3[e] <= nn[e] - 3;
            if (!inside)
              continue;
            double lapN = 0.0;
            for (int e = 0; e < 3; e++)
              lapN +=
                  dC2N([&](int ci, int cj,
                           int ck) { return dN2C(hv, axis, e, ci, cj, ck); },
                       e, i, j, k);
            const size_t m = nodeAt(i, j, k);
            const double row = hv[axis][m] + hD[axis][m] -
                               delt * delt * (lapN + dC2N(divD, axis, i, j, k));
            const double got = hImage[3 * krylov(i, j, k) + axis];
            independent[0] = std::max(independent[0], std::fabs(got - row));
            independent[1] = std::max(independent[1], std::fabs(row));
            wallNodes++;
          }
    }

  // max |image - ref|, max |ref|, max |image - blocking image|, split ranks
  double local[4] = {0.0, 0.0, 0.0, 0.0};
  for (int q = 0; q < n; q++) {
    local[0] = std::max(local[0], std::fabs(hImage[q] - hRef[q]));
    local[1] = std::max(local[1], std::fabs(hRef[q]));
  }
  GPUKrylovVector blocking(n);
  fields.gpuMaxwellImage<EMfields3D::HaloMode::Blocking>(blocking.devPtr(),
                                                         x.devPtr());
  PinnedVector<cudaSolverType> hBlocking(n);
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
  double independentMax[2];
  MPI_Allreduce(independent, independentMax, 2, MPI_DOUBLE, MPI_MAX,
                MPIdata::get_PicGlobalComm());
  long wallNodesAll = 0;
  MPI_Allreduce(&wallNodes, &wallNodesAll, 1, MPI_LONG, MPI_SUM,
                MPIdata::get_PicGlobalComm());
  // a conductor wall with room for the check: 5 nodes or more along both of
  // its in-plane axes (not a flat axis)
  int localWalls = 0, walls = 0;
  for (int fc = 0; fc < 6; fc++) {
    bool room = conductor(fc);
    for (int e = 0; e < 3; e++)
      room = room && (e == fc / 2 || nn[e] >= 5);
    localWalls = localWalls || room;
  }
  MPI_Allreduce(&localWalls, &walls, 1, MPI_INT, MPI_MAX,
                MPIdata::get_PicGlobalComm());
  if (rank == 0) {
    const double rel = globalMax[0] / globalMax[1];
    const double relWall =
        independentMax[1] > 0.0 ? independentMax[0] / independentMax[1] : 0.0;
    std::cout << "image-reference [" << layout.name << ", HaloOverlap "
              << input.haloOverlap
              << "]: max |image - reference| / max |reference| = " << rel
              << "; split on " << splitRanks
              << " ranks, max |image - blocking image| = " << globalMax[2]
              << "; conductor normal rows against the interior row of the "
                 "conductor image: "
              << relWall << " on " << wallNodesAll << " nodes\n";
    if (!(rel < 1e-13))
      check.fail("gpuMaxwellImage differs from the step-by-step image");
    if (walls && wallNodesAll == 0)
      check.fail("no conductor wall node was checked against the image row");
    if (!(relWall < 1e-11))
      check.fail("the conductor's normal row is not the interior row of the "
                 "conductor image of E and mu E");
    const double expected =
        input.haloOverlap && layout.split ? kImageLayoutRanks : 0.0;
    if (splitRanks != expected)
      check.fail("the image does not overlap its exchange as expected");
    if (globalMax[2] != 0.0)
      check.fail("the image differs from the blocking image");
  }
  const std::string label = std::string("image-reference [") + layout.name +
                            ", HaloOverlap " +
                            std::to_string(input.haloOverlap) + "]";
  checkImageLinearity(fields, grid, rank, label.c_str(), check);
}

// local-image (a closed box on 8 ranks, the manufactured fields and
// moments): the local operator of the block-Jacobi preconditioner against the
// full image of a random vector.  The local operator has no exchange: its
// ghost cells are 0 except on the conductor walls, where they are the image
// of E as in the full image.  So the two images are equal bitwise on every
// row whose stencils read no ghost cell of a neighbour rank, the conductor
// rows (identity and normal) included.
void runLocalImageCase(EMfields3D& fields, Grid3DCU& grid,
                       const TestInput& input, int rank, Check& check) {
  const cudaStream_t stream = fields.gpuSolverStream();
  initializeManufacturedFields(fields, grid, input);
  fields.gpuSolverSyncH2D(stream);
  prepareManufacturedMoments(fields);

  const int n = maxwellVectorSize(grid);
  const int nn[3] = {grid.getNXN(), grid.getNYN(), grid.getNZN()};
  const PinnedVector<cudaSolverType> hx = randomValues(n, 2468 + rank);
  GPUKrylovVector x(n), full(n), local(n);
  cudaErrChk(cudaDeviceSynchronize());
  x.copyFromHost(hx.data(), stream);
  fields.gpuMaxwellImage(full.devPtr(), x.devPtr());
  fields.gpuMaxwellImageLocal(local.devPtr(), x.devPtr());
  PinnedVector<cudaSolverType> hFull(n), hLocal(n);
  full.copyToHost(hFull.data(), stream);
  local.copyToHost(hLocal.data(), stream);
  cudaErrChk(cudaStreamSynchronize(stream));

  const VirtualTopology3D& vct = fields.get_vct();
  const bool neighbourLo[3] = {vct.getXleft_neighbor() != MPI_PROC_NULL,
                               vct.getYleft_neighbor() != MPI_PROC_NULL,
                               vct.getZleft_neighbor() != MPI_PROC_NULL};
  const bool neighbourHi[3] = {vct.getXright_neighbor() != MPI_PROC_NULL,
                               vct.getYright_neighbor() != MPI_PROC_NULL,
                               vct.getZright_neighbor() != MPI_PROC_NULL};
  // [0] rows compared, [1] of them on a wall, [2] rows that differ
  long counts[3] = {0, 0, 0};
  for (int i = 1; i < nn[0] - 1; i++)
    for (int j = 1; j < nn[1] - 1; j++)
      for (int k = 1; k < nn[2] - 1; k++) {
        const int c3[3] = {i, j, k};
        bool interface = false, wall = false;
        for (int a = 0; a < 3; a++) {
          interface = interface || (neighbourLo[a] && c3[a] == 1) ||
                      (neighbourHi[a] && c3[a] == nn[a] - 2);
          wall = wall || (!neighbourLo[a] && c3[a] == 1) ||
                 (!neighbourHi[a] && c3[a] == nn[a] - 2);
        }
        if (interface)
          continue;
        const int q =
            3 * (((i - 1) * (nn[1] - 2) + (j - 1)) * (nn[2] - 2) + (k - 1));
        counts[0]++;
        counts[1] += wall;
        for (int c = 0; c < 3; c++)
          counts[2] += hFull[q + c] != hLocal[q + c];
      }
  long global[3];
  MPI_Allreduce(counts, global, 3, MPI_LONG, MPI_SUM,
                MPIdata::get_PicGlobalComm());
  if (rank == 0) {
    std::cout << "local-image: " << global[0] << " rows compared (" << global[1]
              << " on the walls), " << global[2]
              << " differ from the full image\n";
    if (global[0] == 0 || global[1] == 0)
      check.fail("no interior or wall row was compared");
    if (global[2] != 0)
      check.fail("the local operator differs from the full image away from "
                 "the rank interfaces");
  }
  checkImageLinearity(fields, grid, rank, "local-image", check);
}

// ghost-fill: the one-launch ghost fill (FieldBoundary::fill) against the
// sequence of whole-plane passes it replaces, on random arrays, for every
// ghost set of several boundary layouts.  The reference applies, pass by
// pass (the rank of the rules) and face by face (XL ... ZR), each face's
// rule over its whole ghost plane, with the expressions of the former face
// kernels; the fill composes the rules of a point's faces in one launch.
// Also the moment fold against face-by-face doubling.
struct FillLayout {
  const char* name;
  int bcEM[6]; // XR, XL, YR, YL, ZR, ZL (Collective's order)
  int bcP[6];
  int sal;
  const char* simCase;
  int forceWallB; // the case's B forced on the wall layers (case-wall passes)
};
constexpr FillLayout kFillLayouts[] = {
    {"closed box, GEM", {0, 0, 0, 0, 0, 0}, {1, 1, 1, 1, 1, 1}, 0, "GEM", 1},
    {"open, no SAL", {2, 2, 2, 2, 2, 2}, {3, 2, 3, 3, 3, 3}, 0, "Default", 0},
    {"open, SAL, y inflow",
     {2, 2, 2, 2, 2, 2},
     {3, 2, 2, 2, 3, 3},
     1,
     "Default",
     0},
    {"ForceFree, x y walls, z open",
     {0, 0, 0, 0, 2, 2},
     {1, 1, 1, 1, 3, 3},
     0,
     "ForceFree",
     1},
};

/** The rules of the plan against a reference written here from Collective's
 *  face codes (bcEx ... bcBz, which the solver applied before the plan) and
 *  the physics of the faces, not from the plan: the codes on the physical
 *  faces, the image of the fields on a conductor (E_n and B_t even, E_t and
 *  B_n odd about the face node; the cell B_n odd about the wall value),
 *  even moments on the particle-physical faces, the charge and Jhat of the
 *  field solve as the image on a conductor, phi the image (odd) on a
 *  conductor and 0 beyond the other faces, psi no rule.  The sets whose
 *  rules also hold open-face and case-wall passes are checked by the decks
 *  and image-reference (OpCells).  Mismatches of this rank. */
long checkRuleTable(const EMfields3D& fields, const char* layout, int rank) {
  const FieldBoundary& bc = fields.gpuBoundary();
  const Collective& col = fields.get_col();
  const VirtualTopology3D& vct = fields.get_vct();
  const int nb[6] = {vct.getXleft_neighbor(), vct.getXright_neighbor(),
                     vct.getYleft_neighbor(), vct.getYright_neighbor(),
                     vct.getZleft_neighbor(), vct.getZright_neighbor()};
  const int nbP[6] = {vct.getXleft_neighbor_P(), vct.getXright_neighbor_P(),
                      vct.getYleft_neighbor_P(), vct.getYright_neighbor_P(),
                      vct.getZleft_neighbor_P(), vct.getZright_neighbor_P()};
  const int bcEM[6] = {col.getBcEMfaceXleft(), col.getBcEMfaceXright(),
                       col.getBcEMfaceYleft(), col.getBcEMfaceYright(),
                       col.getBcEMfaceZleft(), col.getBcEMfaceZright()};
  const int* const codeE[3] = {col.bcEx, col.bcEy, col.bcEz};
  const int* const codeB[3] = {col.bcBx, col.bcBy, col.bcBz};
  auto code = [](int c) {
    return c == 0   ? GHOST_ODD_CELL
           : c == 1 ? GHOST_ZERO
           : c == 2 ? GHOST_EVEN_CELL
                    : -1;
  };
  long bad = 0;
  auto expect = [&](GhostSet set, const char* name, auto rule) {
    for (int a = 0; a < bc.slots(set); a++)
      for (int f = 0; f < 6; f++) {
        // Collective's per-face arrays list XR, XL, YR, YL, ...
        const int axis = f / 2, cf = 2 * axis + (f % 2 ? 0 : 1);
        const bool physical = nb[f] == MPI_PROC_NULL;
        const bool conductor = physical && bcEM[f] == 0;
        const int want =
            rule(a, axis, cf, physical, conductor, nbP[f] == MPI_PROC_NULL);
        const GhostRuleDev& r = bc.rule(set, a, f);
        const bool ok =
            r.kind == want && (r.kind != GHOST_WALL || r.wall != nullptr);
        if (!ok && bad++ < 8)
          std::cerr << "ghost-fill [" << layout << "] rank " << rank
                    << " rule table " << name << " slot " << a << " face " << f
                    << ": " << r.kind << ", expected " << want << "\n";
      }
  };
  const auto eImage = [](int c, int axis) {
    return c == axis ? GHOST_EVEN_NODE : GHOST_ODD_NODE;
  };
  const auto bImage = [](int c, int axis) {
    return c == axis ? GHOST_ODD_NODE : GHOST_EVEN_NODE;
  };
  expect(
      GhostSet::ENode, "ENode",
      [&](int c, int axis, int cf, bool phys, bool cond, bool) {
        return !phys ? GHOST_KEEP : cond ? eImage(c, axis) : code(codeE[c][cf]);
      });
  expect(GhostSet::ENodeSmooth, "ENodeSmooth",
         [&](int, int, int, bool phys, bool, bool) {
           return phys ? GHOST_EVEN_NODE : GHOST_KEEP;
         });
  expect(
      GhostSet::BNode, "BNode",
      [&](int c, int axis, int cf, bool phys, bool cond, bool) {
        return !phys ? GHOST_KEEP : cond ? bImage(c, axis) : code(codeB[c][cf]);
      });
  expect(GhostSet::BNodeConductor, "BNodeConductor",
         [&](int c, int axis, int, bool, bool cond, bool) {
           return cond ? bImage(c, axis) : GHOST_KEEP;
         });
  expect(GhostSet::BCellStart, "BCellStart",
         [&](int c, int axis, int cf, bool phys, bool cond, bool) {
           return !phys               ? GHOST_KEEP
                  : cond && c == axis ? GHOST_WALL
                                      : code(codeB[c][cf]);
         });
  // the charge of the field solve: the image charge is opposite
  expect(GhostSet::CellChargeP, "CellChargeP",
         [&](int, int, int, bool, bool cond, bool physP) {
           return cond ? GHOST_ODD_CELL : physP ? GHOST_EVEN_CELL : GHOST_KEEP;
         });
  // div(Pi), a current: the image of E on a conductor, even elsewhere
  expect(GhostSet::CellCurrentP, "CellCurrentP",
         [&](int c, int axis, int, bool, bool cond, bool physP) {
           return cond    ? (c == axis ? GHOST_EVEN_CELL : GHOST_ODD_CELL)
                  : physP ? GHOST_EVEN_CELL
                          : GHOST_KEEP;
         });
  // Jhat smoothing: the image of E on a conductor, elsewhere the node mirror
  // of the E smoothing
  expect(GhostSet::NodeCurrentP, "NodeCurrentP",
         [&](int c, int axis, int, bool, bool cond, bool physP) {
           return cond ? eImage(c, axis) : physP ? GHOST_EVEN_NODE : GHOST_KEEP;
         });
  // the div(E) cleaning potential: the conductor image (odd), 0 beyond the
  // other faces; the div(B) one: no rule
  expect(GhostSet::Phi, "Phi", [&](int, int, int, bool phys, bool cond, bool) {
    return cond ? GHOST_ODD_CELL : phys ? GHOST_ZERO : GHOST_KEEP;
  });
  expect(GhostSet::Psi, "Psi",
         [&](int, int, int, bool, bool, bool) { return GHOST_KEEP; });
  return bad;
}

/** Failures of this rank on one layout. */
long runGhostFillLayout(const FillLayout& layout, const TestInput& baseInput,
                        int rank, int nprocs) {
  TestInput input = baseInput;
  input.nxc = 36;
  input.nyc = 32;
  input.nzc = 28;
  ConfigFile config = input.config(TestCase::GhostFill);
  for (const char* key : {"PERIODICX", "PERIODICX_P", "PERIODICY",
                          "PERIODICY_P", "PERIODICZ", "PERIODICZ_P"})
    config.add(key, 0);
  for (int f = 0; f < 6; f++) {
    config.add(kEmBoundaryKeys[f], layout.bcEM[f]);
    config.add(kParticleBoundaryKeys[f], layout.bcP[f]);
  }
  config.add("yes_sal", layout.sal);
  config.add("n_layers_sal", 3);
  config.add("ApplyInflowBcsEImage", 1);
  config.add("Case", layout.simCase);
  config.add("ForceWallB", layout.forceWallB);
  config.add("B0x", 0.3);
  config.add("B0y", -0.2);
  config.add("B0z", 0.25);

  Collective col(config, "gpuMaxwellSolverTest");
  VCtopology3D vct(col);
  if (nprocs != vct.getNprocs())
    throw std::runtime_error("MPI size does not match generated topology");
  vct.setup_vctopology(MPIdata::get_PicGlobalComm());
  Grid3DCU grid(&col, &vct);
  EMfields3D fields(&col, &grid, &vct);
  fields.gpuSolverAllocate();
  // the conductor wall values from a non-uniform node B
  initializeManufacturedFields(fields, grid, input);
  const cudaStream_t stream = fields.gpuSolverStream();
  fields.gpuSolverSyncH2D(stream);
  const FieldBoundary& bc = fields.gpuBoundary();

  std::mt19937_64 gen(97 + rank);
  std::uniform_real_distribution<double> uniform(-1.0, 1.0);
  long filled = 0;
  long failures = checkRuleTable(fields, layout.name, rank);
  for (int id = 0; id < static_cast<int>(GhostSet::Count); id++) {
    const GhostSet set = static_cast<GhostSet>(id);
    const FaceSlabs& shell = bc.shell(set);
    const int* e = shell.n;
    const size_t size = (size_t)e[0] * e[1] * e[2];
    const int slots = bc.slots(set);
    std::vector<GPUFieldArray3> arrays;
    std::vector<PinnedVector<cudaSolverType>> host(slots);
    for (int a = 0; a < slots; a++)
      arrays.emplace_back(e[0], e[1], e[2]);
    cudaErrChk(cudaDeviceSynchronize());
    cudaSolverType* p[kGhostBatchMax];
    for (int a = 0; a < slots; a++) {
      host[a].resize(size);
      for (auto& value : host[a])
        value = uniform(gen);
      arrays[a].copyFromHost(host[a].data());
      p[a] = arrays[a].devPtr();
    }
    // the uploads ran on the default stream, which the non-blocking solver
    // stream does not wait for
    cudaErrChk(cudaDeviceSynchronize());
    bc.fill(set, p, slots, stream);
    cudaErrChk(cudaStreamSynchronize(stream));

    auto at = [&](int i, int j, int k) {
      return ((size_t)i * e[1] + j) * e[2] + k;
    };
    int maxRank = 0;
    for (int a = 0; a < slots; a++)
      for (int f = 0; f < 6; f++)
        maxRank = std::max(maxRank, bc.rule(set, a, f).rank);
    for (int a = 0; a < slots; a++) {
      PinnedVector<cudaSolverType>& h = host[a];
      for (int pass = 0; pass <= maxRank; pass++)
        for (int f = 0; f < 6; f++) {
          const GhostRuleDev& r = bc.rule(set, a, f);
          if (!shell.active[f] || r.kind == GHOST_KEEP || r.rank != pass)
            continue;
          const int axis = f / 2, ghost = f % 2 ? e[axis] - 1 : 0;
          const int inward = f % 2 ? -1 : 1;
          const int slow = axis == 0 ? 1 : 0, fast = axis == 2 ? 1 : 2;
          const PinnedVector<cudaSolverType> wall =
              r.kind == GHOST_WALL
                  ? toHost(r.wall, (size_t)e[slow] * e[fast], stream)
                  : PinnedVector<cudaSolverType>();
          for (int u = 0; u < e[slow]; u++)
            for (int w = 0; w < e[fast]; w++) {
              int c3[3];
              c3[axis] = ghost;
              c3[slow] = u;
              c3[fast] = w;
              auto in = [&](int m) {
                int d3[3] = {c3[0], c3[1], c3[2]};
                d3[axis] += inward * m;
                return h[at(d3[0], d3[1], d3[2])];
              };
              double value = 0.0;
              switch (r.kind) {
              case GHOST_ZERO:
                value = 0.0;
                break;
              case GHOST_EVEN_CELL:
                value = in(1);
                break;
              case GHOST_ODD_CELL:
                value = -in(1);
                break;
              case GHOST_EVEN_NODE:
                value = in(2);
                break;
              case GHOST_ODD_NODE:
                value = 2.0 * in(1) - in(2);
                break;
              case GHOST_WALL:
                value = wall[(size_t)u * e[fast] + w] - in(1);
                break;
              case GHOST_CONST:
                value = r.value;
                break;
              case GHOST_COPY:
                value = in(r.copy);
                break;
              }
              h[at(c3[0], c3[1], c3[2])] = value;
              filled++;
            }
        }
      const PinnedVector<cudaSolverType> device =
          toHost(arrays[a].devPtr(), size, stream);
      long setFailures = 0;
      for (size_t m = 0; m < size; m++)
        if (device[m] != h[m] && setFailures++ == 0) {
          const int i = (int)(m / ((size_t)e[1] * e[2]));
          const int j = (int)((m / e[2]) % e[1]), k = (int)(m % e[2]);
          std::cerr << "ghost-fill [" << layout.name << "] rank " << rank
                    << " set " << id << " array " << a << " at (" << i << ", "
                    << j << ", " << k << ") of (" << e[0] << ", " << e[1]
                    << ", " << e[2] << "): fill " << device[m] << ", passes "
                    << h[m] << "; rules (kind/rank)";
          for (int f = 0; f < 6; f++)
            std::cerr << " " << bc.rule(set, a, f).kind << "/"
                      << bc.rule(set, a, f).rank
                      << (shell.active[f] ? "" : "(inactive)");
          std::cerr << "\n";
        }
      failures += setFailures;
    }
  }

  // the moment fold: 2 per particle-physical face plane holding the node
  {
    const FaceSlabs& planes = bc.momentPlanes();
    const int* e = planes.n;
    const size_t size = (size_t)e[0] * e[1] * e[2];
    constexpr int kArrays = 4;
    std::vector<GPUFieldArray3> arrays;
    for (int a = 0; a < kArrays; a++)
      arrays.emplace_back(e[0], e[1], e[2]);
    cudaErrChk(cudaDeviceSynchronize());
    std::vector<PinnedVector<cudaSolverType>> host(kArrays);
    PinnedVector<cudaSolverType*> p(kArrays); // copied to the device
    for (int a = 0; a < kArrays; a++) {
      host[a].resize(size);
      for (auto& value : host[a])
        value = uniform(gen);
      arrays[a].copyFromHost(host[a].data());
      p[a] = arrays[a].devPtr();
    }
    cudaSolverType** d_p = nullptr;
    cudaErrChk(cudaMalloc(&d_p, kArrays * sizeof(cudaSolverType*)));
    cudaErrChk(cudaMemcpy(d_p, p.data(), kArrays * sizeof(cudaSolverType*),
                          cudaMemcpyHostToDevice));
    cudaErrChk(cudaDeviceSynchronize()); // as above
    bc.foldMoments(d_p, kArrays, stream);
    cudaErrChk(cudaStreamSynchronize(stream));
    cudaErrChk(cudaFree(d_p));
    for (int a = 0; a < kArrays; a++) {
      PinnedVector<cudaSolverType>& h = host[a];
      for (int f = 0; f < 6; f++) {
        if (!planes.active[f])
          continue;
        const int axis = f / 2, plane = f % 2 ? e[axis] - 2 : 1;
        for (int i = 1; i < e[0] - 1; i++)
          for (int j = 1; j < e[1] - 1; j++)
            for (int k = 1; k < e[2] - 1; k++) {
              const int c3[3] = {i, j, k};
              if (c3[axis] == plane)
                h[((size_t)i * e[1] + j) * e[2] + k] *= 2.0;
            }
      }
      const PinnedVector<cudaSolverType> device =
          toHost(arrays[a].devPtr(), size, stream);
      for (size_t m = 0; m < size; m++)
        failures += device[m] != h[m];
    }
  }

  long global[2], local[2] = {failures, filled};
  MPI_Allreduce(local, global, 2, MPI_LONG, MPI_SUM,
                MPIdata::get_PicGlobalComm());
  if (rank == 0)
    std::cout << "ghost-fill [" << layout.name << "]: " << global[1]
              << " ghost points of the passes, " << global[0]
              << " values differ from the passes\n";
  if (global[1] == 0)
    return rank == 0 ? 1 : 0;
  return rank == 0 ? (global[0] != 0) : 0;
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

bool imageSplitsEverywhere(const EMfields3D& fields);

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
  if (input.haloOverlap && !imageSplitsEverywhere(fields))
    throw std::runtime_error("oblique-wave: HaloOverlap = 1, but the Maxwell "
                             "image does not split");

  const ObliqueWave wave = makeObliqueWave(input);
  initializeObliqueWave(fields, grid, wave);
  const cudaStream_t stream = fields.gpuSolverStream();
  fields.gpuSolverSyncH2D(stream);

  for (int cycle = 0; cycle < steps; cycle++) {
    fields.gpuCalculateE(cycle);
    fields.gpuCalculateB(cycle);
  }

  fields.gpuSolverSyncD2H(FieldOutputArrays::all(), stream);
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

/** With HaloOverlap = 1 the Maxwell image has to overlap its exchange on
 *  every rank (MaxwellImageBoxes::split): an overlap variant that fell back
 *  to the blocking image would test nothing more than the blocking one. */
bool imageSplitsEverywhere(const EMfields3D& fields) {
  int local = fields.gpuMaxwellImageBoxes().split ? 1 : 0, global = 0;
  MPI_Allreduce(&local, &global, 1, MPI_INT, MPI_MIN,
                MPIdata::get_PicGlobalComm());
  return global == 1;
}

void checkOverlapSplit(const EMfields3D& fields, const TestInput& input,
                       int rank, Check& check) {
  if (input.haloOverlap && !imageSplitsEverywhere(fields) && rank == 0)
    check.fail("HaloOverlap = 1, but the Maxwell image does not split: the "
               "overlap variant would run the blocking image");
}

int runOnFields(TestCase testCase, EMfields3D& fields, Grid3DCU& grid,
                const TestInput& input, int rank) {
  Check check;
  check.rank = rank;
  check.tolerance = input.comparisonTolerance;

  fields.gpuSolverAllocate();
  checkOverlapSplit(fields, input, rank, check);

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

  case TestCase::SmoothedFaces:
    runSmoothedFacesCase(fields, grid, input, rank, check);
    break;

  case TestCase::InflowLayers:
    runInflowLayersCase(fields, grid, input, rank, check);
    break;

  case TestCase::ConductorWall:
    runConductorWallCase(fields, grid, input, rank, check);
    break;

  case TestCase::ConductorEdges:
    runConductorEdgesCase(fields, grid, input, rank, check);
    break;

  case TestCase::ConductorModes:
    runConductorModesCase(fields, grid, input, rank, check);
    break;

  case TestCase::ObliqueWave:
    throw std::logic_error(
        "oblique-wave refinement must run through its multi-grid driver");

  case TestCase::LocalImage:
    runLocalImageCase(fields, grid, input, rank, check);
    break;

  case TestCase::ImageReference:
    throw std::logic_error("image-reference must run through its layouts");

  case TestCase::GhostFill:
    throw std::logic_error("ghost-fill must run through its layouts");
  }

  return check.failures;
}

int runCase(TestCase testCase, int rank, int nprocs, const TestInput& input) {
  if (testCase == TestCase::ObliqueWave ||
      testCase == TestCase::ImageReference || testCase == TestCase::GhostFill) {
    int localFailures = 0;
    if (testCase == TestCase::ObliqueWave)
      localFailures = runObliqueWaveConvergence(rank, nprocs, input);
    else if (testCase == TestCase::GhostFill)
      for (const FillLayout& layout : kFillLayouts)
        localFailures +=
            static_cast<int>(runGhostFillLayout(layout, input, rank, nprocs));
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

  // conductor-modes: a small vacuum box, theta = 1/2
  TestInput caseInput = input;
  if (testCase == TestCase::ConductorModes) {
    caseInput.Lx = 1.0;
    caseInput.Ly = 1.0;
    caseInput.Lz = 0.25;
    caseInput.nxc = 32;
    caseInput.nyc = 32;
    caseInput.nzc = 8; // 4 cells per rank: the overlap variant splits
    caseInput.dt = 0.01;
    caseInput.th = 0.5;
  }
  ConfigFile config = caseInput.config(testCase);
  Collective col(config, "gpuMaxwellSolverTest");
  VCtopology3D vct(col);
  if (nprocs != vct.getNprocs()) {
    throw std::runtime_error("MPI size does not match generated topology");
  }
  vct.setup_vctopology(MPIdata::get_PicGlobalComm());

  Grid3DCU grid(&col, &vct);
  EMfields3D fields(&col, &grid, &vct);

  const int localFailures =
      runOnFields(testCase, fields, grid, caseInput, rank);
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
