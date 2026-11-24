/* iPIC3D was originally developed by Stefano Markidis and Giovanni Lapenta.
 * This release was contributed by Alec Johnson and Ivy Bo Peng.
 * Publications that use results from iPIC3D need to properly cite
 * 'S. Markidis, G. Lapenta, and Rizwan-uddin. "Multi-scale simulations of
 * plasma with iPIC3D." Mathematics and Computers in Simulation 80.7 (2010):
 * 1509-1519.'
 *
 *        Copyright 2015 KTH Royal Institute of Technology
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *         http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "Collective.h"
#include "BenchmarkMode.h"
#include "ConfigFile.h"
#include "HeatFluxComponents.h"
#include "MPIdata.h"
#include "RestartReader.h"
#include "asserts.h" // for assert_ge
#include "dataAnalysisConfig.cuh"
#include "debug.h"
#include "limits.h" // for INT_MAX
#include "string.h"
#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdarg>
#include <math.h>
#include <mpi.h>
#include <numeric>
#include <sstream>
#include <stdio.h>
#include <stdlib.h>
#include <vector>

// The HDF5 field backends of this build (FieldWriteMethod): shdf5 needs
// HDF5, phdf5 parallel HDF5.
#ifdef NO_HDF5
constexpr bool kBuildHasHDF5 = false;
#else
constexpr bool kBuildHasHDF5 = true;
#endif
#ifdef PHDF5
constexpr bool kBuildHasParallelHDF5 = true;
#else
constexpr bool kBuildHasParallelHDF5 = false;
#endif
// The GPU field solver's sizes in the memory estimate (Print): the GMRES
// restart length and the fields of a halo batch per direction.
constexpr double kGmresM = 20.0;      // EMfields3D::GMRES_M
constexpr double kHaloMaxBatch = 64.0; // EMfields3D::HALO_MAX_BATCH

// order must agree with Enum in Collective.h
static const char* enumNames[] = {
    "default", "initial", "final",
    // used by ImplSusceptMode
    "explPredict", "implPredict",
    // marker for last enumerated symbol of this class
    "NUMBER_OF_ENUMS", "INVALID_ENUM"};

namespace {

bool parseTrackParticleIDToken(const std::string& token) {
  std::string lower;
  lower.reserve(token.size());
  for (char ch : token)
    lower.push_back(
        static_cast<char>(std::tolower(static_cast<unsigned char>(ch))));

  if (lower == "0" || lower == "false" || lower == "f" || lower == "no" ||
      lower == "n" || lower == "none")
    return false;
  return true;
}

std::vector<unsigned char> parseTrackParticleIDMask(const ConfigFile& config,
                                                    int count) {
  std::vector<unsigned char> mask(static_cast<size_t>(count), 0);
  const std::string raw =
      config.read<std::string>("TrackParticleID", std::string());
  std::istringstream iss(raw);
  std::vector<unsigned char> values;
  std::string token;
  while (iss >> token)
    values.push_back(parseTrackParticleIDToken(token) ? 1 : 0);

  if (values.empty())
    return mask;
  if (values.size() == 1) {
    std::fill(mask.begin(), mask.end(), values[0]);
    return mask;
  }

  const size_t ncopy = std::min(mask.size(), values.size());
  std::copy(values.begin(), values.begin() + ncopy, mask.begin());
  return mask;
}

/** The values of the input list text, "a b c ..." (none if empty), into
 *  read. @return false for a value that is not a number of the type (out of
 *  its range too), or not followed by a space or the end ("1.0.5" and "3+4"
 *  are no lists of two). */
template <typename T>
bool readList(const std::string& text, std::vector<T>& read) {
  std::istringstream iss(text);
  read.clear();
  while (!(iss >> std::ws).eof()) {
    T value;
    if (!(iss >> value) || !(iss.eof() || std::isspace(iss.peek())))
      return false;
    read.push_back(value);
  }
  return true;
}

/** Per-species values of an optional input list "a b c ...": absent, it
 *  keeps `values`; one value goes to every species, otherwise one value per
 *  species. @return false for another number of values or a value that is
 *  not a number of the type (out of its range too). */
template <typename T>
bool readSpeciesList(const ConfigFile& config, const char* key,
                     std::vector<T>& values) {
  std::vector<T> read;
  if (!readList(config.read<std::string>(key, std::string()), read))
    return false;
  if (read.size() > 1 && read.size() != values.size())
    return false;
  for (size_t s = 0; s < values.size() && !read.empty(); s++)
    values[s] = read[read.size() == 1 ? 0 : s];
  return true;
}

// The cells of every rank along an axis but the last: Grid3DCU's
// ceiling_of_ratio (ipicmath.h, which needs the CUDA types)
int ceilDiv(int cells, int ranks) { return (cells + ranks - 1) / ranks; }

} // namespace

int Collective::read_enum_parameter(const char* option_name,
                                    const char* default_value,
                                    const ConfigFile& config) {
  string enum_name = config.read<string>(option_name, default_value);
  // search the list (could use std::map)
  //
  for (int i = 0; i < NUMBER_OF_ENUMS; i++) {
    if (!strcmp(enum_name.c_str(), enumNames[i]))
      return i;
  }
  requireInput(false, "there is an invalid option %s", enum_name.c_str());
  return INVALID_ENUM;
}

const char* Collective::get_name_of_enum(int in) {
  assert_ge(in, 0);
  assert_lt(in, NUMBER_OF_ENUMS);
  return enumNames[in];
}

// The Smooth filter, a f + (1-a)/6 (sum of the 6 neighbours), has the
// response a + (1-a)/3 (cos kx + cos ky + cos kz).  Along a single periodic
// cell the cos stays 1, so with d other directions the lowest response is
// a - (1-a)(2d-3)/3, negative below a = (2d-3)/(2d): 1/2 in 3D, 1/4 in 2D,
// never in 1D.
double Collective::smoothFlipBound() const {
  const int d = !(nxc == 1 && PERIODICX) + !(nyc == 1 && PERIODICY) +
                !(nzc == 1 && PERIODICZ);
  return d > 1 ? (2.0 * d - 3.0) / (2.0 * d) : 0.0;
}

// Every rank reads the same input and stops together (eprintf would abort
// rank 0 and leave the others in MPI_Finalize).
void Collective::requireInput(bool ok, const char* fmt, ...) const {
  if (ok)
    return;
  if (!MPIdata::get_rank()) {
    char msg[512];
    va_list args;
    va_start(args, fmt);
    vsnprintf(msg, sizeof(msg), fmt, args);
    va_end(args);
    printf("ERROR: in input file %s %s\n", inputfile.c_str(), msg);
    fflush(stdout);
  }
  MPIdata::exit(1);
}

void Collective::readSwitch(const ConfigFile& config, const char* key,
                            bool& value) const {
  const string text = config.read<string>(key, value ? "1" : "0");
  requireInput(text == "0" || text == "1", "%s must be 0 or 1, not \"%s\"", key,
               text.c_str());
  value = text == "1";
}

/*! Read the input file; on a restart also select the checkpoint to load (the
 * fields and particles are loaded later) */
void Collective::ReadInput(string inputfile) {
  ConfigFile config(inputfile);
  ReadInput(config);
}

void Collective::ReadInput(const ConfigFile& config) {
  using namespace std;
  int test_verbose;
  // A list of a required key of exactly count values, one per species in
  // order (of `whose`: species, test-particle species, ...), into values; a
  // missing key, another number of values, or a value that is not a number
  // of the type stops the run.
  const auto readExactList = [&](const char* key, int count, const char* whose,
                                 auto& values) {
    using T = typename std::decay_t<decltype(values)>::element_type;
    // a missing key reads as an empty list (count is at least 1)
    const string text = config.read<string>(key, "");
    std::vector<T> read;
    requireInput(readList(text, read) && (int)read.size() == count,
                 "%s needs %d value%s, one per %s, not \"%s\"", key, count,
                 count == 1 ? "" : "s", whose, text.c_str());
    values = std::make_unique<T[]>(count);
    std::copy(read.begin(), read.end(), values.get());
  };
  // the following variables are ALWAYS taken from inputfile, even if restarting
  {

#ifdef BATSRUS
    if (RESTART1) {
      cout << " The fluid interface can not handle RESTART yet, aborting!\n"
           << flush;
      abort();
    }
#endif

    dt = config.read<double>("dt");
    ncycles = config.read<int>("ncycles");
    th = config.read<double>("th", 1.0);
    // the theta scheme is unconditionally stable only for th >= 1/2
    requireInput(th >= 0.5, "th must be at least 0.5, not %g", th);

    Smooth = config.read<double>("Smooth", 1.0);
    SmoothNiter = config.read<int>("SmoothNiter", 6);
    haloOverlap = config.read<int>("HaloOverlap", 0) != 0;
    solverSourceGrowthLimit =
        config.read<double>("SolverSourceGrowthLimit", 100.0);
    solverStopOnNonConvergence =
        config.read<int>("SolverStopOnNonConvergence", 1) != 0;

    SaveDirName = config.read<string>("SaveDirName", "data");
    RestartDirName = config.read<string>("RestartDirName", "data");
    RestartReadDirName = RestartDirName;
    ns = config.read<int>("ns");
    nstestpart = config.read<int>("nsTestPart", 0);
    requireInput(ns >= 1 && nstestpart >= 0,
                 "ns must be at least 1 and nsTestPart at least 0, not %d "
                 "and %d",
                 ns, nstestpart);
    TrackParticleID = parseTrackParticleIDMask(config, ns + nstestpart);
    NpMaxNpRatio = config.read<double>("NpMaxNpRatio", 1.5);
    assert_ge(NpMaxNpRatio, 1.);
    // mode parameters for second order in time
    PushWithBatTime = config.read<double>("PushWithBatTime", 0);
    PushWithEatTime = config.read<double>("PushWithEatTime", 1);
    ImplSusceptTime = config.read<double>("ImplSusceptTime", 0);
    ImplSusceptMode = read_enum_parameter("ImplSusceptMode", "initial", config);
    switch (ImplSusceptMode) {
    // values not yet supported:
    case explPredict:
    case implPredict:
    default:
      unsupported_value_error(ImplSusceptMode);
    // supported values:
    case initial:;
    }
    // GEM Challenge
    B0x = config.read<double>("B0x", 0.0);
    B0y = config.read<double>("B0y", 0.0);
    B0z = config.read<double>("B0z", 0.0);

    // Earth parameters
    B1x = 0.0;
    B1y = 0.0;
    B1z = 0.0;
    B1x = config.read<double>("B1x", 0.0);
    B1y = config.read<double>("B1y", 0.0);
    B1z = config.read<double>("B1z", 0.0);

    delta = config.read<double>("delta", 0.5);

    pertGEM = config.read<double>("pertGEM", 0.0);
    pertHump = config.read<double>("pertHump", 0.0);
    deltaxHump = config.read<double>("deltaxHump", 8.0);
    deltayHump = config.read<double>("deltayHump", 4.0);
    kxHump = config.read<double>("kxHump", -1.0);
    kyHump = config.read<double>("kyHump", -1.0);
    currentFromAmpere = config.read<int>("currentFromAmpere", 0);
    spatiallyVaryingThermal = config.read<int>("spatiallyVaryingThermal", 0);

    Case = config.read<string>("Case");
    // the Harris pressure law on the perturbed flux function needs the Harris
    // sheet, a periodic flux function (no uniform B_y) and the Ampere loader
    requireInput(!spatiallyVaryingThermal ||
                     (Case == "GEMHarris" && currentFromAmpere && B0x != 0.0 &&
                      B0y == 0.0),
                 "spatiallyVaryingThermal = 1 needs Case = GEMHarris, "
                 "currentFromAmpere = 1, B0x != 0 and B0y = 0");
    // the GEM perturbation current fills the box, where the sheet electrons
    // that carry the perturbation currents have no density
    requireInput(Case != "GEMHarris" || !currentFromAmpere || pertGEM == 0.0,
                 "pertGEM needs currentFromAmpere = 0: its current fills the "
                 "box, where the sheet electrons that would carry it have no "
                 "density");
    // the field output backend. WriteMethod, the old single key, stops
    // every rank before anything runs: it names none of the three now.
    requireInput(!config.keyExists("WriteMethod"),
                 "WriteMethod is no longer read: remove it and set the three "
                 "write methods, FieldWriteMethod (fields: shdf5, phdf5, pvtk "
                 "or nbcvtk; required), ParticlesWriteMethod (particles: "
                 "adios2 or shdf5) and RestartWriteMethod (checkpoints: adios2 "
                 "or shdf5)");
    fieldWriteMethod = config.read<string>("FieldWriteMethod");
    // one of this build, also with the field output off (the key is
    // required): an output would otherwise stop the run at its first cycle
    const bool hdf5Fields =
        fieldWriteMethod == "shdf5" || fieldWriteMethod == "phdf5";
    requireInput(hdf5Fields || fieldWriteMethod == "pvtk" ||
                     fieldWriteMethod == "nbcvtk",
                 "FieldWriteMethod must be shdf5, phdf5, pvtk or nbcvtk, not "
                 "%s",
                 fieldWriteMethod.c_str());
    requireInput(!hdf5Fields || kBuildHasHDF5,
                 "FieldWriteMethod = %s needs a build with HDF5 "
                 "(USE_HDF5=ON); pvtk and nbcvtk write VTK files",
                 fieldWriteMethod.c_str());
    requireInput(fieldWriteMethod != "phdf5" || kBuildHasParallelHDF5,
                 "FieldWriteMethod = phdf5 needs a build with parallel HDF5 "
                 "(USE_PHDF5=ON); shdf5 writes the fields with serial HDF5, "
                 "one file per rank");
    SimName = config.read<string>("SimulationName");
    PoissonCorrection = config.read<string>("PoissonCorrection", "no");
    PoissonCorrectionCycle = config.read<int>("PoissonCorrectionCycle", 10);
    divBCorrection = config.read<string>("divBCorrection", "no");
    divBCorrectionCycle = config.read<int>("divBCorrectionCycle", 10);
    // the field solver takes cycle % <name>Cycle
    requireInput(PoissonCorrection != "yes" || PoissonCorrectionCycle >= 1,
                 "PoissonCorrectionCycle must be at least 1, not %d",
                 PoissonCorrectionCycle);
    requireInput(divBCorrection != "yes" || divBCorrectionCycle >= 1,
                 "divBCorrectionCycle must be at least 1, not %d",
                 divBCorrectionCycle);

    readExactList("rhoINIT", ns, "species", rhoINIT);
    readExactList("rhoINJECT", ns, "species", rhoINJECT);

    // solver tolerances (CGtol: host CG, not used currently)
    CGtol = config.read<double>("CGtol", 1e-3);
    GMREStol = config.read<double>("GMREStol", 1e-3);
    DivCleaningTol = config.read<double>("DivCleaningTol", 1e-6);
    requireInput((PoissonCorrection != "yes" && divBCorrection != "yes") ||
                     (DivCleaningTol > 0.0 && DivCleaningTol < 1.0),
                 "DivCleaningTol must be in (0, 1), not %g", DivCleaningTol);
    NiterMover = config.read<int>("NiterMover", 3);
    // the movers take the average velocity from the first iteration
    requireInput(NiterMover >= 1, "NiterMover must be at least 1, not %d",
                 NiterMover);
    // Maxwell solver: "GMRES" (default), "FGMRESBlockJacobi" or "Chebyshev"
    // (the other two experimental, warned in Print); the solve compares the
    // names exactly, so a misspelt one would run GMRES unnoticed
    SolverType = config.read<string>("SolverType", "GMRES");
    requireInput(SolverType == "GMRES" || SolverType == "FGMRESBlockJacobi" ||
                     SolverType == "Chebyshev",
                 "SolverType must be GMRES, FGMRESBlockJacobi or Chebyshev, "
                 "not %s",
                 SolverType.c_str());
    ChebyshevMaxIter = config.read<int>("ChebyshevMaxIter", 20);
    ChebyshevEigMin = config.read<double>("ChebyshevEigMin", 0.0);
    ChebyshevEigMax = config.read<double>("ChebyshevEigMax", 0.0);
    // Block-Jacobi preconditioner parameters
    BlockJacobiSweeps = config.read<int>("BlockJacobiSweeps", 1);
    BlockJacobiOmega = config.read<double>("BlockJacobiOmega", 1.0);
    // Poisson Chebyshev preconditioner parameters (divergence cleaning)
    PoissonChebMaxIter = config.read<int>("PoissonChebMaxIter", 20);
    PoissonChebRescaleEigMin =
        config.read<double>("PoissonChebRescaleEigMin", 1.0);
    PoissonChebRescaleEigMax =
        config.read<double>("PoissonChebRescaleEigMax", 1.0);
    PoissonChebEigRatio = config.read<double>("PoissonChebEigRatio", 300.0);
    // take the injection of the particless
    Vinj = config.read<double>("Vinj", 0.0);

    // take the output cycles
    FieldOutputCycle = config.read<int>("FieldOutputCycle", 100);
    ParticlesOutputCycle = config.read<int>("ParticlesOutputCycle", 0);
    FieldOutputTag = config.read<string>("FieldOutputTag", "");
    ParticlesOutputTag = config.read<string>("ParticlesOutputTag", "");
    // an unknown token: refused here, as the HDF5 writer would ignore it.
    // Parsed before the call that prints the token: the arguments of a call
    // are evaluated in no set order.
    string unknownToken;
    const bool particleTagValid = parseParticleTag(
        ParticlesOutputTag, ParticlesOutputArrays, unknownToken);
    requireInput(particleTagValid,
                 "ParticlesOutputTag: %s is not position, velocity, q or ID",
                 unknownToken.c_str());
    // the backends of the particle output and of the checkpoints; another
    // one of the build if it lacks the one asked for (IOManager)
    auto readFileBackend = [&](const char* key) {
      const string method = config.read<string>(key, "");
      requireInput(method.empty() || method == "adios2" || method == "shdf5",
                   "%s must be adios2 or shdf5, not %s", key, method.c_str());
      return method;
    };
    ParticlesWriteMethod = readFileBackend("ParticlesWriteMethod");
    RestartWriteMethod = readFileBackend("RestartWriteMethod");
    // outputs written on writer threads while the run goes on
    // (AsyncOutput.h), or at their cycle on the main thread
    readSwitch(config, "AsyncOutput", asyncOutput);
    // the species whose particles are written, nonzero = on, every species
    // by default
    std::vector<int> particleSpecies(ns, 1);
    requireInput(
        readSpeciesList(config, "ParticlesOutputSpecies", particleSpecies),
        "ParticlesOutputSpecies needs one number, or one per species (%d)", ns);
    AllSpecies.resize(ns);
    std::iota(AllSpecies.begin(), AllSpecies.end(), 0);
    ParticlesOutputSpecies.clear();
    for (int s = 0; s < ns; s++)
      if (particleSpecies[s] != 0)
        ParticlesOutputSpecies.push_back(s);
    requireInput(!particle_output_requested() ||
                     !ParticlesOutputSpecies.empty(),
                 "ParticlesOutputSpecies has no species: ParticlesOutputCycle "
                 "= 0 writes no particles");
    // the particle output writes at most this many particles of each
    // species in a cell, 0: all of them (particleOutputSelect.cuh)
    ParticlesOutputMaxPerCell.assign(ns, 0);
    requireInput(readSpeciesList(config, "ParticlesOutputMaxPerCell",
                                 ParticlesOutputMaxPerCell) &&
                     std::all_of(ParticlesOutputMaxPerCell.begin(),
                                 ParticlesOutputMaxPerCell.end(),
                                 [](int k) { return k >= 0; }),
                 "ParticlesOutputMaxPerCell needs one number, or one per "
                 "species (%d), at least 0",
                 ns);
    // of a species with IDs, the particle output writes the particles of
    // u < this fraction (particleOutputSelect.cuh), ParticlesOutputMaxPerCell
    // then ignored: 1 all, 0 none; -1 off
    ParticlesOutputTrackedFraction.assign(ns, -1.0);
    requireInput(readSpeciesList(config, "ParticlesOutputTrackedFraction",
                                 ParticlesOutputTrackedFraction) &&
                     std::all_of(ParticlesOutputTrackedFraction.begin(),
                                 ParticlesOutputTrackedFraction.end(),
                                 [](double f) {
                                   return f == -1.0 || (f >= 0.0 && f <= 1.0);
                                 }),
                 "ParticlesOutputTrackedFraction needs one number, or one per "
                 "species (%d), each -1 (off) or from 0 to 1",
                 ns);
    MomentsOutputTag = config.read<string>("MomentsOutputTag", "");
    outputConfig_ = parseOutputTags(FieldOutputTag, MomentsOutputTag, ns);
    TestParticlesOutputCycle = config.read<int>("TestPartOutputCycle", 0);
    testPartFlushCycle = config.read<int>("TestParticlesOutputCycle", 10);
    RestartOutputCycle = config.read<int>("RestartOutputCycle", 5000);
    DiagnosticsOutputCycle =
        config.read<int>("DiagnosticsOutputCycle", FieldOutputCycle);
    PlanetDiagnosticsCycle =
        config.read<int>("PlanetDiagnosticsCycle", DiagnosticsOutputCycle);
    SortingCycle = config.read<int>("SortingCycle", 0);
    ParticleSplitting = config.read<bool>("ParticleSplitting", false);
    ParticleMerging = config.read<bool>("ParticleMerging", false);
    ParticleSplittingThreshold =
        config.read<double>("ParticleSplittingThreshold", 0.95);
    ParticleMergingThreshold =
        config.read<double>("ParticleMergingThreshold", 1.05);
    ParaviewScriptPath = config.read<string>("ParaviewScriptPath", "");
    CallFinalize = config.read<bool>("CallFinalize", true);
  }

  // read everything from input file, if restart is true, overwrite the setting

  restart_status = 0;
  last_cycle = -1;
  c = config.read<double>("c", 1.0);
  // the Gaussian equations divide by c
  requireInput(c > 0.0, "the speed of light c must be positive, not %g", c);

#ifdef BATSRUS
  // set grid size and resolution based on the initial file from fluid code
  Lx = getFluidLx();
  Ly = getFluidLy();
  Lz = getFluidLz();
  nxc = getFluidNxc();
  nyc = getFluidNyc();
  nzc = getFluidNzc();
#else
  Lx = config.read<double>("Lx", 10.0);
  Ly = config.read<double>("Ly", 10.0);
  Lz = config.read<double>("Lz", 10.0);
  nxc = config.read<int>("nxc", 64);
  nyc = config.read<int>("nyc", 64);
  nzc = config.read<int>("nzc", 64);
#endif
  XLEN = config.read<int>("XLEN", 1);
  YLEN = config.read<int>("YLEN", 1);
  ZLEN = config.read<int>("ZLEN", 1);
  // Grid3DCU gives every rank along an axis ceil(cells / ranks) cells and the
  // last one the rest, which must be at least one cell
  auto requireSplit = [&](const char* cellKey, int cells, const char* rankKey,
                          int ranks) {
    requireInput(cells >= 1 && ranks >= 1,
                 "%s and %s must be positive, not %d and %d", cellKey, rankKey,
                 cells, ranks);
    const int regular = ceilDiv(cells, ranks);
    requireInput(cells - regular * (ranks - 1) >= 1,
                 "%s = %d over %s = %d ranks of %d cells each leaves the last "
                 "rank no cells; choose values that leave it at least one, "
                 "for example with %s dividing %s",
                 cellKey, cells, rankKey, ranks, regular, rankKey, cellKey);
  };
  requireSplit("nxc", nxc, "XLEN", XLEN);
  requireSplit("nyc", nyc, "YLEN", YLEN);
  requireSplit("nzc", nzc, "ZLEN", ZLEN);
  PERIODICX = config.read<bool>("PERIODICX", true);
  PERIODICY = config.read<bool>("PERIODICY", true);
  PERIODICZ = config.read<bool>("PERIODICZ", true);

  PERIODICX_P = config.read<bool>("PERIODICX_P", PERIODICX);
  PERIODICY_P = config.read<bool>("PERIODICY_P", PERIODICY);
  PERIODICZ_P = config.read<bool>("PERIODICZ_P", PERIODICZ);

  // The Smooth filter damps every mode but k = 0 only for Smooth in [0, 1]:
  // above 1 every other mode grows, below 0 the shortest ones do.  Below
  // smoothFlipBound() it flips the shortest modes (a warning in Print).
  requireInput(Smooth >= 0.0 && Smooth <= 1.0,
               "Smooth must be in [0, 1], not %g", Smooth);
  // the div(B) face rules take the next inner node: a non-periodic direction
  // needs 2 cells
  requireInput(divBCorrection != "yes" ||
                   ((nxc > 1 || PERIODICX) && (nyc > 1 || PERIODICY) &&
                    (nzc > 1 || PERIODICZ)),
               "divBCorrection needs at least 2 cells along every non-periodic "
               "direction (a single-cell direction must be periodic)");

  x_center_dipole = config.read<double>("x_center_dipole", 5.0);
  y_center_dipole = config.read<double>("y_center_dipole", 5.0);
  z_center_dipole = config.read<double>("z_center_dipole", 5.0);
  x_center_planet = config.read<double>("x_center_planet", 5.0);
  y_center_planet = config.read<double>("y_center_planet", 5.0);
  z_center_planet = config.read<double>("z_center_planet", 5.0);
  Planet_radius = config.read<double>("Planet_radius", 5.0);
  if (config.keyExists("L_square")) {
    const bool hasPlanetRadius = config.keyExists("Planet_radius");
    if (!hasPlanetRadius)
      Planet_radius = config.read<double>("L_square");

    if (MPIdata::get_rank() == 0) {
      cerr << "WARNING: input parameter 'L_square' is deprecated; use "
           << "'Planet_radius' instead";
      if (hasPlanetRadius)
        cerr << ". The 'L_square' value is ignored";
      cerr << "." << endl;
    }
  }

  // ── Planet reflection model: 0=specular, 1=diffuse (Lambert, default) ──
  planetReflectionType = config.read<int>("planetReflectionType", 1);
  requireInput(planetReflectionType == 0 || planetReflectionType == 1,
               "planetReflectionType must be 0 (specular) or 1 (diffuse), "
               "not %d",
               planetReflectionType);

  // ── Exosphere / planet species parameters ──
  numSolarWindSpecies =
      config.read<int>("ns_solar_wind", ns); // default: all species are SW
  numPlanetarySpecies =
      config.read<int>("ns_planetary", 0); // default: no planetary species
  enableExosphereInjection =
      config.read<int>("AddExosphereInjection", 0); // 0=off, 1=on
  maxInjectionRadius = config.read<double>("RmaxExosphereInjection", 3.0);
  maxExosphereParticlesPerSpecies =
      config.read<int>("MaxExosphereParticlesPerSpecies", 0); // 0 = unlimited
  // the solver injects into the species from ns_solar_wind on, whose
  // neutral parameters (below) are ns_planetary
  requireInput(numSolarWindSpecies >= 0 && numSolarWindSpecies <= ns &&
                   numPlanetarySpecies >= 0,
               "ns_solar_wind must be from 0 to ns (%d) and ns_planetary at "
               "least 0, not %d and %d",
               ns, numSolarWindSpecies, numPlanetarySpecies);
  requireInput(!enableExosphereInjection ||
                   numPlanetarySpecies == ns - numSolarWindSpecies,
               "AddExosphereInjection = 1 needs ns_planetary = ns - "
               "ns_solar_wind = %d, not %d",
               ns - numSolarWindSpecies, numPlanetarySpecies);

  // the neutral parameters of the planetary species, one value each; without
  // planetary species one entry of defaults (the arrays are never null)
  if (numPlanetarySpecies == 0) {
    neutralSurfaceDensity = std::make_unique<double[]>(1);
    exosphericScaleHeight = std::make_unique<double[]>(1);
    photoionizationFrequency = std::make_unique<double[]>(1);
    macroParticleWeightRatio = std::make_unique<double[]>(1);
    exosphericScaleHeight[0] = 1.0;
    macroParticleWeightRatio[0] = 1.0;
  } else {
    const char* planetary = "planetary species";
    readExactList("NeutralSurfaceDensity", numPlanetarySpecies, planetary,
                  neutralSurfaceDensity);
    readExactList("ExosphericScaleHeight", numPlanetarySpecies, planetary,
                  exosphericScaleHeight);
    readExactList("PhotoionizationFrequency", numPlanetarySpecies, planetary,
                  photoionizationFrequency);
    readExactList("MacroParticleWeightRatio", numPlanetarySpecies, planetary,
                  macroParticleWeightRatio);
  }

  readExactList("uth", ns, "species", uth);
  readExactList("vth", ns, "species", vth);
  readExactList("wth", ns, "species", wth);
  readExactList("u0", ns, "species", u0);
  readExactList("v0", ns, "species", v0);
  readExactList("w0", ns, "species", w0);

  if (nstestpart > 0) {
    readExactList("pitch_angle", nstestpart, "test-particle species",
                  pitch_angle);
    readExactList("energy", nstestpart, "test-particle species", energy);
  }

  const char* all = "species and test-particle species";
  readExactList("npcelx", ns + nstestpart, all, npcelx);
  readExactList("npcely", ns + nstestpart, all, npcely);
  readExactList("npcelz", ns + nstestpart, all, npcelz);
  readExactList("qom", ns + nstestpart, all, qom);
  // a species with density loads and injects npcelx npcely npcelz particles
  // per cell (planetary species start empty: no density, no particles); a
  // qom of 0 is warned of (Print)
  for (int is = 0; is < ns + nstestpart; is++) {
    requireInput(npcelx[is] >= 0 && npcely[is] >= 0 && npcelz[is] >= 0,
                 "npcelx, npcely and npcelz of species %d must be at least 0, "
                 "not %d %d %d",
                 is, npcelx[is], npcely[is], npcelz[is]);
  }
  for (int is = 0; is < ns; is++) {
    requireInput(rhoINIT[is] >= 0.0 && rhoINJECT[is] >= 0.0,
                 "rhoINIT and rhoINJECT of species %d must be at least 0, "
                 "not %g and %g",
                 is, rhoINIT[is], rhoINJECT[is]);
    // each factor, not the product, which may overflow
    requireInput((rhoINIT[is] == 0.0 && rhoINJECT[is] == 0.0) ||
                     (npcelx[is] > 0 && npcely[is] > 0 && npcelz[is] > 0),
                 "species %d has rhoINIT = %g and rhoINJECT = %g but "
                 "npcelx * npcely * npcelz = 0: it would load or inject no "
                 "particles",
                 is, rhoINIT[is], rhoINJECT[is]);
  }
  // the exosphere sizes its particles by those per cell of species 0
  // (ExosphereIonization)
  requireInput(!enableExosphereInjection || numPlanetarySpecies == 0 ||
                   (npcelx[0] > 0 && npcely[0] > 0 && npcelz[0] > 0),
               "AddExosphereInjection = 1 sizes the injected particles by "
               "npcelx * npcely * npcelz of species 0, which is 0: it would "
               "inject none");

  // verbose = config.read < bool > ("verbose",false);

  // PHI Electrostatic Potential
  bcPHIfaceXright = config.read<int>("bcPHIfaceXright", 1);
  bcPHIfaceXleft = config.read<int>("bcPHIfaceXleft", 1);
  bcPHIfaceYright = config.read<int>("bcPHIfaceYright", 1);
  bcPHIfaceYleft = config.read<int>("bcPHIfaceYleft", 1);
  bcPHIfaceZright = config.read<int>("bcPHIfaceZright", 1);
  bcPHIfaceZleft = config.read<int>("bcPHIfaceZleft", 1);

  // EM field boundary condition (required keys)
  auto readBcEM = [&](const char* key) {
    const int code = config.read<int>(key);
    requireInput(code >= 0 && code <= 2,
                 "%s must be 0 (perfect conductor), 1 (perfect magnetic "
                 "conductor) or 2 (open), not %d",
                 key, code);
    return code;
  };
  bcEMfaceXright = readBcEM("bcEMfaceXright");
  bcEMfaceXleft = readBcEM("bcEMfaceXleft");
  bcEMfaceYright = readBcEM("bcEMfaceYright");
  bcEMfaceYleft = readBcEM("bcEMfaceYleft");
  bcEMfaceZright = readBcEM("bcEMfaceZright");
  bcEMfaceZleft = readBcEM("bcEMfaceZleft");

  // EM field absorbing boundary condition parameters (SAL on by default; it
  // acts only on open faces, bcEMface = 2)
  yes_sal = config.read<int>("yes_sal", 1);
  n_layers_sal = config.read<int>("n_layers_sal", 3);
  // B SAL on the components tangential to the face only: the global div(B)
  // correction is consistent only with this choice (see
  // Documentation/boundary-conditions.md)
  sal_tangential_B = config.read<int>("sal_tangential_B", 1);

  /*  ------------------------------------------------------------------- */
  /*  Electric and Magnetic field boundary conditions for BCface          */
  /*  ------------------------------------------------------------------- */
  /*  bcEM* == 0 : Perfect Electric Conductor (PEC)                       */
  /*    - E_tangential = 0   (Dirichlet, bc=1)                            */
  /*    - E_normal     = free (Neumann,  bc=2)                            */
  /*    - B_tangential = free (Neumann,  bc=2)                            */
  /*    - B_normal     = bc=1 here; the GPU solver then sets the ghost    */
  /*      cells of the cell B to 2 W - (first cell), W the initial wall   */
  /*      value: a conductor freezes B_n on the wall                      */
  /*      (GHOST_WALL rule of FieldBoundary, GPUFieldBoundary.cuh)       */
  /*    These are the codes of the cell B, of the host fields, and of     */
  /*    the GPU node fields on other faces.  On a conductor the GPU node  */
  /*    fields (E, B) instead get the image of the field in the wall,     */
  /*    E_n and B_t even, E_t and B_n odd about the face value            */
  /*    (GhostSet::ENode, BNode), and E_t = 0 on the wall                 */
  /*    (FieldBoundary::zeroTangential).  The GPU rules:                  */
  /*    FieldBoundary.cpp.                                                */
  /*  bcEM* != 0 : Perfect Magnetic Conductor (PMC) codes, for code 1     */
  /*    and for the open faces (2), whose GPU rules are their own         */
  /*    - E_tangential = free (Neumann,  bc=2)                            */
  /*    - E_normal     = 0   (Dirichlet, bc=1)                            */
  /*    - B_tangential = 0   (Dirichlet, bc=1)                            */
  /*    - B_normal     = free (Neumann,  bc=2)                            */
  /*                                                                      */
  /*  Convention: bc=1 -> Dirichlet (value fixed to 0)                    */
  /*             bc=2 -> Neumann   (derivative fixed, value free)         */
  /*                                                                      */
  /*  Note: E and B always get complementary bc values on every face.     */
  /*  Face indices: 0=Xright, 1=Xleft, 2=Yright, 3=Yleft, 4=Zright, 5=Zleft */
  /*  ------------------------------------------------------------------- */

  /* Ex Bx component: normal on X-faces (bc=2 for PEC), tangential on Y/Z-faces
   * (bc=1 for PEC) */
  bcEx[0] = bcEMfaceXright == 0 ? 2 : 1;
  bcBx[0] = bcEMfaceXright == 0 ? 1 : 2;
  bcEx[1] = bcEMfaceXleft == 0 ? 2 : 1;
  bcBx[1] = bcEMfaceXleft == 0 ? 1 : 2;
  bcEx[2] = bcEMfaceYright == 0 ? 1 : 2;
  bcBx[2] = bcEMfaceYright == 0 ? 2 : 1;
  bcEx[3] = bcEMfaceYleft == 0 ? 1 : 2;
  bcBx[3] = bcEMfaceYleft == 0 ? 2 : 1;
  bcEx[4] = bcEMfaceZright == 0 ? 1 : 2;
  bcBx[4] = bcEMfaceZright == 0 ? 2 : 1;
  bcEx[5] = bcEMfaceZleft == 0 ? 1 : 2;
  bcBx[5] = bcEMfaceZleft == 0 ? 2 : 1;
  /* Ey By component: tangential on X-faces (bc=1 for PEC), normal on Y-faces
   * (bc=2 for PEC), tangential on Z-faces */
  bcEy[0] = bcEMfaceXright == 0 ? 1 : 2;
  bcBy[0] = bcEMfaceXright == 0 ? 2 : 1;
  bcEy[1] = bcEMfaceXleft == 0 ? 1 : 2;
  bcBy[1] = bcEMfaceXleft == 0 ? 2 : 1;
  bcEy[2] = bcEMfaceYright == 0 ? 2 : 1;
  bcBy[2] = bcEMfaceYright == 0 ? 1 : 2;
  bcEy[3] = bcEMfaceYleft == 0 ? 2 : 1;
  bcBy[3] = bcEMfaceYleft == 0 ? 1 : 2;
  bcEy[4] = bcEMfaceZright == 0 ? 1 : 2;
  bcBy[4] = bcEMfaceZright == 0 ? 2 : 1;
  bcEy[5] = bcEMfaceZleft == 0 ? 1 : 2;
  bcBy[5] = bcEMfaceZleft == 0 ? 2 : 1;
  /* Ez Bz component: tangential on X/Y-faces (bc=1 for PEC), normal on Z-faces
   * (bc=2 for PEC) */
  bcEz[0] = bcEMfaceXright == 0 ? 1 : 2;
  bcBz[0] = bcEMfaceXright == 0 ? 2 : 1;
  bcEz[1] = bcEMfaceXleft == 0 ? 1 : 2;
  bcBz[1] = bcEMfaceXleft == 0 ? 2 : 1;
  bcEz[2] = bcEMfaceYright == 0 ? 1 : 2;
  bcBz[2] = bcEMfaceYright == 0 ? 2 : 1;
  bcEz[3] = bcEMfaceYleft == 0 ? 1 : 2;
  bcBz[3] = bcEMfaceYleft == 0 ? 2 : 1;
  bcEz[4] = bcEMfaceZright == 0 ? 2 : 1;
  bcBz[4] = bcEMfaceZright == 0 ? 1 : 2;
  bcEz[5] = bcEMfaceZleft == 0 ? 2 : 1;
  bcBz[5] = bcEMfaceZleft == 0 ? 1 : 2;

  // Particles Boundary condition: a code of BCparticles (ParticleSoAHost.h)
  auto readBcP = [&](const char* key) {
    const int code = config.read<int>(key, 1);
    requireInput(code >= 0 && code <= 3,
                 "%s must be 0 (exit), 1 (mirror), 2 (open inflow) or 3 "
                 "(open outflow), not %d",
                 key, code);
    return code;
  };
  bcPfaceXright = readBcP("bcPfaceXright");
  bcPfaceXleft = readBcP("bcPfaceXleft");
  bcPfaceYright = readBcP("bcPfaceYright");
  bcPfaceYleft = readBcP("bcPfaceYleft");
  bcPfaceZright = readBcP("bcPfaceZright");
  bcPfaceZleft = readBcP("bcPfaceZleft");

  // E field inflow BC: master switch for the inflow rows of the Maxwell image
  applyInflowBcsEImage = config.read<int>("ApplyInflowBcsEImage", 1);

  // The equilibrium B forced on the y-wall layers of the GEM and ForceFree
  // cases every cycle: off by default (only the conductor walls' BCs)
  forceWallB = config.read<int>("ForceWallB", 0);

  // Fields on the X-right outflow layer: kept as solved (default), or copied
  // from the reference plane
  const string outflowLayerFields =
      config.read<string>("OutflowLayerFields", "solved");
  requireInput(outflowLayerFields == "copy" || outflowLayerFields == "solved",
               "OutflowLayerFields must be copy or solved, not %s",
               outflowLayerFields.c_str());
  outflowLayerSolved = outflowLayerFields == "solved";

  readDataAnalysisInput(config);

  if (RESTART1) { // you are restarting
    RestartDirName = config.read<string>("RestartDirName", "data");
    RestartReadDirName = RestartDirName;
    restart_status = 1;

    // Resolve the A/B restart checkpoint: the requested slot, or the latest,
    // falling back to the other slot when it is not usable.
    RestartCheckpoint checkpoint = RestartReader::resolveCheckpoint(
        RestartDirName, RestartSlotRequest, RestartSelectionReport);
    RestartReadDirName = checkpoint.dataDir;
    RestartReadSlot = checkpoint.slot;
    last_cycle = checkpoint.cycle;
  }
}

void Collective::readDataAnalysisInput(const ConfigFile& config) {
  DataAnalysisInput& in = dataAnalysis_;
  in.cycle = config.read<int>("DataAnalysisCycle", in.cycle);
  requireInput(in.cycle >= 0, "DataAnalysisCycle must be >= 0, not %d",
               in.cycle);
  in.outputDir = config.read<string>("DataAnalysisOutputDir", in.outputDir);
  if (in.outputDir.empty())
    in.outputDir = "./";
  else if (in.outputDir.back() != '/')
    in.outputDir += '/';
  auto requireSpeciesList = [&](bool ok, const char* key) {
    requireInput(ok, "%s needs one number, or one per species (%d)", key, ns);
  };
  // a per-species mask, nonzero = on, of every species `all` by default: the
  // analysis of switchKey, when on, needs a species in it
  auto readSpeciesMask = [&](const char* key, int all, const char* switchKey,
                             bool on, std::vector<int>& mask) {
    mask.assign(ns, all);
    requireSpeciesList(readSpeciesList(config, key, mask), key);
    requireInput(!on || std::any_of(mask.begin(), mask.end(),
                                    [](int s) { return s != 0; }),
                 "%s = 1 needs a species in %s", switchKey, key);
  };

  readSwitch(config, "VelocityHistogram", in.velocityHistogram);
  readSpeciesMask("VelocityHistGMMSpecies", 1, "VelocityHistogram",
                  in.velocityHistogram, in.histogramSpecies);
  readSwitch(config, "VelocityHistogramMacrocells", in.histogramMacrocells);
  requireInput(!in.histogramMacrocells || in.velocityHistogram,
               "VelocityHistogramMacrocells = 1 needs VelocityHistogram = 1");
  readSwitch(config, "VelocityHistogramFixedRange", in.histogramFixedRange);
  // the default range depends on the sign of the charge; Min defaults to -Max
  in.histogramMax.resize(ns);
  for (int s = 0; s < ns; s++)
    in.histogramMax[s] =
        qom[s] < 0 ? kHistogramVmaxElectrons : kHistogramVmaxIons;
  requireSpeciesList(
      readSpeciesList(config, "VelocityHistogramMax", in.histogramMax),
      "VelocityHistogramMax");
  in.histogramMin.resize(ns);
  for (int s = 0; s < ns; s++)
    in.histogramMin[s] = -in.histogramMax[s];
  requireSpeciesList(
      readSpeciesList(config, "VelocityHistogramMin", in.histogramMin),
      "VelocityHistogramMin");
  for (int s = 0; s < ns; s++)
    requireInput(in.histogramMax[s] > 0 &&
                     in.histogramMin[s] < in.histogramMax[s] &&
                     std::isfinite(in.histogramMax[s] - in.histogramMin[s]),
                 "VelocityHistogramMax of species %d (%g) must be positive "
                 "and above its VelocityHistogramMin (%g), by a finite width",
                 s, in.histogramMax[s], in.histogramMin[s]);

  readSwitch(config, "VelocityGMM", in.gmm);
  requireInput(!in.gmm || in.velocityHistogram,
               "VelocityGMM = 1 needs VelocityHistogram = 1");
  in.gmmComponents = config.read<int>("GMMComponents", in.gmmComponents);
  in.gmmMaxIterations =
      config.read<int>("GMMMaxIterations", in.gmmMaxIterations);
  in.gmmConvergenceThreshold = config.read<double>("GMMConvergenceThreshold",
                                                   in.gmmConvergenceThreshold);
  readSwitch(config, "GMMWarmStart", in.gmmWarmStart);
  readSwitch(config, "GMMNormalize", in.gmmNormalize);
  readSwitch(config, "GMMCheckCovariance", in.gmmCheckCovariance);
  in.gmmCovarianceTolerance =
      config.read<double>("GMMCovarianceTolerance", in.gmmCovarianceTolerance);
  in.gmmMinVariance = config.read<double>("GMMMinVariance", in.gmmMinVariance);
  readSwitch(config, "GMMPruneComponents", in.gmmPruneComponents);
  in.gmmPruneThreshold =
      config.read<double>("GMMPruneThreshold", in.gmmPruneThreshold);
  requireInput(in.gmmComponents >= 1 && in.gmmComponents <= kGMMMaxComponents,
               "GMMComponents must be in [1, %d], not %d", kGMMMaxComponents,
               in.gmmComponents);
  requireInput(in.gmmMaxIterations >= 1,
               "GMMMaxIterations must be >= 1, not %d", in.gmmMaxIterations);
  requireInput(in.gmmConvergenceThreshold > 0 &&
                   in.gmmCovarianceTolerance > 0 && in.gmmMinVariance > 0,
               "GMMConvergenceThreshold, GMMCovarianceTolerance and "
               "GMMMinVariance must be positive");
  requireInput(in.gmmPruneThreshold > 0 && in.gmmPruneThreshold < 1,
               "GMMPruneThreshold must be in (0, 1), not %g",
               in.gmmPruneThreshold);
  requireInput(!in.gmm || !in.gmmPruneComponents ||
                   in.gmmPruneThreshold * in.gmmComponents < 1,
               "GMMPruneThreshold (%g) must be below 1 / GMMComponents (%g) "
               "with GMMPruneComponents = 1: the cold start gives each "
               "component the weight 1 / GMMComponents",
               in.gmmPruneThreshold, 1.0 / in.gmmComponents);

  // the macrocell sizes of the spectra and of the macrocell histograms: each
  // has its own keys, which default, axis by axis, to the common MacrocellNx,
  // Ny, Nz; an analysis that is on needs its sizes, at most the cells of a
  // rank (the regular ones; along an axis where a truncated last rank has
  // fewer cells, its one macrocell has all of them); the spectra need a
  // species too
  readSwitch(config, "VelocitySpectra", in.velocitySpectra);
  const char* axes[3] = {"Nx", "Ny", "Nz"};
  const int cells[3] = {ceilDiv(nxc, XLEN), ceilDiv(nyc, YLEN),
                        ceilDiv(nzc, ZLEN)};
  for (int a = 0; a < 3; a++) {
    const std::string commonKey = std::string("Macrocell") + axes[a];
    const int common = config.read<int>(commonKey, 0);
    // the size of the analysis of this switch, given by <analysis>Macrocell
    auto readSize = [&](const char* analysis, const char* switchKey, bool on,
                        int& size) {
      const std::string key = std::string(analysis) + commonKey;
      size = config.read<int>(key, common);
      requireInput(!on || (size > 0 && size <= cells[a]),
                   "%s = 1 needs %s (default: %s) in [1, %d] (the cells of a "
                   "rank), not %d",
                   switchKey, key.c_str(), commonKey.c_str(), cells[a], size);
    };
    readSize("VelocitySpectra", "VelocitySpectra", in.velocitySpectra,
             in.spectraMacrocellSize[a]);
    readSize("VelocityHistogram", "VelocityHistogramMacrocells",
             in.histogramMacrocells, in.histogramMacrocellSize[a]);
  }
  // and at most `most` macrocells on a rank (a regular one has the most)
  auto requireMacrocells = [&](const char* analysis, const char* switchKey,
                               bool on, const int size[3], int most) {
    if (!on)
      return;
    long long macrocells = 1;
    for (int a = 0; a < 3; a++)
      macrocells *= (cells[a] + size[a] - 1) / size[a];
    requireInput(macrocells <= most,
                 "%s = 1 with macrocells of %dx%dx%d cells gives a rank of "
                 "%dx%dx%d cells %lld macrocells, more than the %d it can "
                 "hold: give larger %sMacrocellNx, Ny, Nz (default: "
                 "MacrocellNx, Ny, Nz)",
                 switchKey, size[0], size[1], size[2], cells[0], cells[1],
                 cells[2], macrocells, most, analysis);
  };
  requireMacrocells("VelocitySpectra", "VelocitySpectra", in.velocitySpectra,
                    in.spectraMacrocellSize, DAConfig::MAX_SPECTRA_MACROCELLS);
  requireMacrocells("VelocityHistogram", "VelocityHistogramMacrocells",
                    in.histogramMacrocells, in.histogramMacrocellSize,
                    DAConfig::MAX_HISTOGRAM_MACROCELLS);
  readSpeciesMask("VelocitySpectraSpecies", 0, "VelocitySpectra",
                  in.velocitySpectra, in.spectraSpecies);
}

bool Collective::field_output_is_off() const { return (FieldOutputCycle <= 0); }

bool Collective::particle_output_is_off() const {
  return getParticlesOutputCycle() <= 0;
}
bool Collective::particle_output_requested() const {
  return !particle_output_is_off() && ParticlesOutputArrays.any();
}
bool Collective::diagnostics_due(int cycle) const {
  return DiagnosticsOutputCycle > 0 && cycle % DiagnosticsOutputCycle == 0;
}
bool Collective::testparticle_output_is_off() const {
  return getTestParticlesOutputCycle() <= 0;
}

void Collective::read_field_restart(const VCtopology3D* vct, const Grid* grid,
                                    arr3_double Bxn, arr3_double Byn,
                                    arr3_double Bzn, arr3_double Ex,
                                    arr3_double Ey, arr3_double Ez,
                                    arr4_double* rhons_, int ns) const {
  // Delegate to RestartReader (implementation in inputoutput/RestartReader.cpp)
  RestartReader::readFields(vct, grid, Bxn, Byn, Bzn, Ex, Ey, Ez, rhons_, ns,
                            getRestartReadDirName(), last_cycle);
}

void Collective::read_particles_restart(const VCtopology3D* vct,
                                        int species_number, vector_double& u,
                                        vector_double& v, vector_double& w,
                                        vector_double& q, vector_double& x,
                                        vector_double& y, vector_double& z,
                                        vector_cudaPclType_ID& id) const {
  // Delegate to RestartReader (implementation in inputoutput/RestartReader.cpp)
  RestartReader::readParticles(vct, species_number, u, v, w, q, x, y, z, id,
                               getTrackParticleID(species_number),
                               getRestartReadDirName(), last_cycle);
}

/*! constructor: parse the command line, then read the input file.
 *
 * Arguments, in any order: the input file (default "inputfile") and, for a
 * restart, one of
 *   restart | restart=latest | restart=A | restart=B | restart A | restart B
 * where A and B are the checkpoint slots restart_A and restart_B. */
Collective::Collective(int argc, char** argv) {
  RESTART1 = false;
  std::vector<std::string> inputs;
  std::string error;
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    const bool bare = arg == "restart";
    const bool withValue = arg.rfind("restart=", 0) == 0;
    if (!bare && !withValue) {
      inputs.push_back(arg);
      continue;
    }
    if (RESTART1) {
      error = "the restart option is given more than once";
      break;
    }
    RESTART1 = true;
    if (withValue) {
      const std::string value = arg.substr(std::string("restart=").size());
      if (!RestartSlotManager::parseSlotName(value, RestartSlotRequest)) {
        error = "unknown restart slot '" + value + "' (use A, B or latest)";
        break;
      }
    } else if (i + 1 < argc) {
      // "restart A": take the next argument when it names a slot
      std::string slot;
      const std::string next = argv[i + 1];
      if (!next.empty() && RestartSlotManager::parseSlotName(next, slot)) {
        RestartSlotRequest = slot;
        ++i;
      }
    }
  }
  if (error.empty() && inputs.size() > 1)
    error = "more than one input file given ('" + inputs[0] + "', '" +
            inputs[1] + "')";
  if (!error.empty()) {
    if (MPIdata::get_rank() == 0)
      std::cout << "Error in the command line: " << error << ".\n"
                << "Usage: iPIC3D <input file> [restart | restart=A | "
                   "restart=B | restart=latest]"
                << std::endl;
    MPIdata::exit(1);
  }
  inputfile = inputs.empty() ? std::string("inputfile") : inputs[0];

  if (RESTART1 && MPIdata::get_rank() == 0)
    std::cout << "Restarting..." << endl;
  ReadInput(inputfile);
  init_derived_parameters();
}

Collective::Collective(const ConfigFile& config, const std::string& input_name,
                       bool restart) {
  inputfile = input_name;
  RESTART1 = restart;
  ReadInput(config);
  init_derived_parameters();
}

void Collective::init_derived_parameters() {
  /*! fourpi = 4 greek pi */
  fourpi = 16.0 * atan(1.0);
  /*! dx = space step - X direction */
  dx = Lx / (double)nxc;
  /*! dy = space step - Y direction */
  dy = Ly / (double)nyc;
  /*! dz = space step - Z direction */
  dz = Lz / (double)nzc;
  /*! npcel = number of particles per cell */
  npcel = std::make_unique<int[]>(ns + nstestpart);
  /*! np = number of particles of different species */
  // np = new int[ns];
  /*! npMax = maximum number of particles of different species */
  // npMax = new int[ns];

  /* quantities per process */

  // check that procs divides grid
  // (this restriction should be removed).
  //
  if (0 == MPIdata::get_rank()) {
    fflush(stdout);
    bool xerror = false;
    bool yerror = false;
    bool zerror = false;
    if (nxc % XLEN)
      xerror = true;
    if (nyc % YLEN)
      yerror = true;
    if (nzc % ZLEN)
      zerror = true;
    if (xerror)
      warning_printf("XLEN=%d does not divide nxc=%d\n", XLEN, nxc);
    if (yerror)
      warning_printf("YLEN=%d does not divide nyc=%d\n", YLEN, nyc);
    if (zerror)
      warning_printf("ZLEN=%d does not divide nzc=%d\n", ZLEN, nzc);
    fflush(stdout);
    bool error = (xerror || yerror || zerror);
    // Comment out this check if your postprocessing code does not
    // require the field output subarrays to be the same size.
    // Alternatively, you could modify the output routine to pad
    // with zeros...
    // if(error)
    //{
    //  eprintf("processor dimensions must divide mesh cell dimensions");
    //}
  }

  int num_cells_r = nxc * nyc * nzc;
  // num_procs = XLEN*YLEN*ZLEN;
  // ncells_rs = nxc_rs*nyc_rs*nzc_rs;

  for (int i = 0; i < (ns + nstestpart); i++) {
    npcel[i] = npcelx[i] * npcely[i] * npcelz[i];
    // np[i] = npcel[i] * num_cells;
    // nop_rs[i] = npcel[i] * ncells_rs;
    // maxnop_rs[i] = NpMaxNpRatio * nop_rs[i];
    //  INT_MAX is about 2 billion, surely enough
    //  to index the particles in a single MPI process:
    // assert_le(NpMaxNpRatio * npcel[i] * ncells_proper_per_proc ,
    // double(INT_MAX)); double npMaxi = (NpMaxNpRatio * np[i]); npMax[i] =
    // (int) npMaxi;
  }
}

/*! Print Simulation Parameters */
void Collective::Print() {
  cout << endl;
  cout << "Simulation Parameters" << endl;
  cout << "---------------------" << endl;
  cout << "Number of species    = " << ns << endl;
  for (int i = 0; i < ns; i++)
    cout << "qom[" << i << "] = " << qom[i] << endl;
  cout << "x-Length                 = " << Lx << endl;
  cout << "y-Length                 = " << Ly << endl;
  cout << "z-Length                 = " << Lz << endl;
  cout << "Number of cells (x)      = " << nxc << endl;
  cout << "Number of cells (y)      = " << nyc << endl;
  cout << "Number of cells (z)      = " << nzc << endl;
  cout << "Time step                = " << dt << endl;
  cout << "Number of cycles         = " << ncycles << endl;
  cout << "Results saved in  : " << SaveDirName << endl;
  cout << "Case type         : " << Case << endl;
  cout << "Simulation name   : " << SimName << endl;
  cout << "Maxwell solver    : " << SolverType << endl;
  if (SolverType != "GMRES")
    cout << "WARNING: SolverType = " << SolverType
         << " is experimental; GMRES is the tested Maxwell solver" << endl;
  cout << "Smoothing         : " << (Smooth == 1.0 ? "off" : "on")
       << " (alpha=" << Smooth << ", Niter=" << SmoothNiter << ")" << endl;
  if (Smooth != 1.0 && SmoothNiter % 2 == 1 && Smooth < smoothFlipBound())
    cout << "WARNING: Smooth = " << Smooth << " is below " << smoothFlipBound()
         << " on this grid: the filter's response to the shortest grid modes "
            "is negative, so every pass flips their sign, and the odd "
            "SmoothNiter = "
         << SmoothNiter
         << " leaves them inverted in the smoothed E and Jhat; use an even "
            "SmoothNiter"
         << endl;
  cout << "Halo overlap      : " << (haloOverlap ? "on" : "off")
       << " (field solver)" << endl;
  cout << "Solver guard      : ||b|| growth limit " << solverSourceGrowthLimit
       << " per cycle (0 = off), non-converged solve "
       << (solverStopOnNonConvergence ? "stops" : "warns") << endl;
  cout << "Particle control  : splitting "
       << (ParticleSplitting ? "on" : "off");
  if (ParticleSplitting)
    cout << " (below " << ParticleSplittingThreshold
         << " of the reference count)";
  cout << ", merging " << (ParticleMerging ? "on" : "off");
  if (ParticleMerging)
    cout << " (above " << ParticleMergingThreshold << ")";
  cout << endl;
  if (ParticleSplitting != ParticleMerging)
    cout << "WARNING: particle "
         << (ParticleSplitting ? "splitting" : "merging") << " is on without "
         << (ParticleSplitting ? "merging" : "splitting")
         << ": the number of particles of a rank is bounded only "
         << (ParticleSplitting ? "from below" : "from above") << endl;
  if (ParticleSplitting && ParticleMerging &&
      ParticleSplittingThreshold >= ParticleMergingThreshold)
    cout << "WARNING: ParticleSplittingThreshold = "
         << ParticleSplittingThreshold
         << " is not below ParticleMergingThreshold = "
         << ParticleMergingThreshold
         << ": splitting and merging undo each other" << endl;
  // a qom of 0: allowed. The charge sign qom/|qom| and the mass q/qom divide
  // by it: the charge of the particles, the charge held fixed (the planet,
  // the open faces of a TaylorGreen restart) and the bulk-flow energy
  for (int is = 0; is < ns + nstestpart; is++) {
    if (qom[is] != 0.0)
      continue;
    if (is < ns)
      cout << "WARNING: qom of species " << is
           << " is 0: its charge sign and mass (q/qom) are not defined: its "
              "particles, or a charge held fixed for it (a Dipole or Dipole2D "
              "planet, the open faces of a TaylorGreen restart), make the "
              "field solve non-finite; with neither, its bulk-flow energy in "
              "ConservedQuantities.txt is nan"
           << endl;
    else
      cout << "WARNING: qom of species " << is
           << " (a test-particle species) is 0: the charge of its particles "
              "is not defined (nan in the serial-HDF5 checkpoints)"
           << endl;
  }
  // the particles written per species (ParticlesOutputTrackedFraction,
  // ParticlesOutputMaxPerCell)
  bool anySet = false;
  for (const int s : ParticlesOutputSpecies)
    anySet = anySet || ParticlesOutputTrackedFraction[s] >= 0.0 ||
             ParticlesOutputMaxPerCell[s] > 0;
  if (particle_output_requested() && anySet) {
    cout << "Particle output   :";
    for (const int s : ParticlesOutputSpecies) {
      cout << " species " << s << ":";
      if (particleOutputFractionOn(s))
        cout << " fraction " << particleOutputFraction(s);
      else if (particleOutputMaxPerCell(s) > 0)
        cout << " at most " << particleOutputMaxPerCell(s) << " per cell";
      else
        cout << " all";
      cout << ";";
    }
    cout << endl;
    for (const int s : ParticlesOutputSpecies) {
      if (ParticlesOutputTrackedFraction[s] >= 0.0 && !getTrackParticleID(s))
        cout << "WARNING: ParticlesOutputTrackedFraction = "
             << ParticlesOutputTrackedFraction[s] << " of species " << s
             << " has no effect: the species has no particle IDs "
                "(TrackParticleID)"
             << endl;
      if (particleOutputFractionOn(s) && ParticlesOutputMaxPerCell[s] > 0)
        cout << "WARNING: ParticlesOutputMaxPerCell = "
             << ParticlesOutputMaxPerCell[s] << " of species " << s
             << " has no effect: ParticlesOutputTrackedFraction = "
             << ParticlesOutputTrackedFraction[s] << " selects its particles"
             << endl;
    }
  }
  const DataAnalysisInput& da = dataAnalysis_;
  // the macrocells of an analysis: " of Nx x Ny x Nz cells"
  auto macrocellsOf = [](const int size[3]) {
    return " of " + std::to_string(size[0]) + " x " + std::to_string(size[1]) +
           " x " + std::to_string(size[2]) + " cells";
  };
  cout << "Data analysis     : ";
  if (!BenchmarkConfig::DIAGNOSTIC_CALCULATIONS_ENABLED)
    cout << "not in this build (BENCHMARK_MODE = 2)" << endl;
  else if (da.any())
    cout << "every " << da.cycle << " cycles in " << da.outputDir
         << ", histograms "
         << (!da.velocityHistogram ? std::string("off")
             : da.histogramMacrocells
                 ? "per macrocell" + macrocellsOf(da.histogramMacrocellSize)
                 : std::string("per rank"))
         << ", GMM "
         << (da.gmm ? "on (" + std::to_string(da.gmmComponents) + " components)"
                    : std::string("off"))
         << ", macrocell spectra "
         << (da.velocitySpectra
                 ? "on (macrocells" + macrocellsOf(da.spectraMacrocellSize) +
                       ")"
                 : std::string("off"))
         << (BenchmarkConfig::DISK_OUTPUT_ENABLED
                 ? ""
                 : " (not written, BENCHMARK_MODE = 1)")
         << endl;
  else
    cout << "off" << endl;
  cout << "---------------------" << endl;
  cout << "EM Field Boundary Conditions" << endl;
  cout << "---------------------" << endl;
  // helper lambda to decode bcEMface codes
  auto bcName = [](int code) -> const char* {
    switch (code) {
    case 0:
      return "perfect conductor";
    case 1:
      return "perfect magnetic conductor (first order)";
    case 2:
      return "open";
    default:
      return "unknown";
    }
  };
  cout << "Xleft  : " << bcEMfaceXleft << " (" << bcName(bcEMfaceXleft) << ")"
       << endl;
  cout << "Xright : " << bcEMfaceXright << " (" << bcName(bcEMfaceXright) << ")"
       << endl;
  cout << "Yleft  : " << bcEMfaceYleft << " (" << bcName(bcEMfaceYleft) << ")"
       << endl;
  cout << "Yright : " << bcEMfaceYright << " (" << bcName(bcEMfaceYright) << ")"
       << endl;
  cout << "Zleft  : " << bcEMfaceZleft << " (" << bcName(bcEMfaceZleft) << ")"
       << endl;
  cout << "Zright : " << bcEMfaceZright << " (" << bcName(bcEMfaceZright) << ")"
       << endl;
  cout << "SAL (absorbing layer): " << (yes_sal ? "yes" : "no");
  if (yes_sal)
    cout << ", n_layers=" << n_layers_sal
         << (sal_tangential_B ? ", B tangential only" : ", all B components");
  cout << endl;
  // The open-face B layers (FieldBoundarySpec::openBLayers) run only on wall
  // ranks with more than 8 cells along the face normal, a gate meant for the
  // flat direction of a 2D run: flag a thinner split of any other direction.
  // Cells per rank as in Grid3DCU: the last rank takes the remainder.
  auto warnThinOpenFace = [&](const char* face, int bc, bool periodic,
                              int cells, int len, bool right, char axis) {
    const int first = ceilDiv(cells, len);
    const int local = right ? cells - first * (len - 1) : first;
    if (bc == 2 && !periodic && cells > 1 && local <= 8)
      cout << "WARNING: open face " << face << " has " << local
           << " cells per rank along " << axis
           << ": the open-face B treatment needs at least 9 and is skipped; "
              "use fewer ranks along "
           << axis << " (at least 9 cells per rank on that face)" << endl;
  };
  warnThinOpenFace("Xleft", bcEMfaceXleft, PERIODICX, nxc, XLEN, false, 'x');
  warnThinOpenFace("Xright", bcEMfaceXright, PERIODICX, nxc, XLEN, true, 'x');
  warnThinOpenFace("Yleft", bcEMfaceYleft, PERIODICY, nyc, YLEN, false, 'y');
  warnThinOpenFace("Yright", bcEMfaceYright, PERIODICY, nyc, YLEN, true, 'y');
  warnThinOpenFace("Zleft", bcEMfaceZleft, PERIODICZ, nzc, ZLEN, false, 'z');
  warnThinOpenFace("Zright", bcEMfaceZright, PERIODICZ, nzc, ZLEN, true, 'z');
  cout << "---------------------" << endl;
  cout << "Particle Boundary Conditions" << endl;
  cout << "---------------------" << endl;
  auto bcPName = [](int code) -> const char* {
    switch (code) {
    case 0:
      return "exit";
    case 1:
      return "perfect mirror";
    case 2:
      return "open inflow";
    case 3:
      return "open outflow";
    default:
      return "unknown";
    }
  };
  cout << "Xleft  : " << bcPfaceXleft << " (" << bcPName(bcPfaceXleft) << ")"
       << endl;
  cout << "Xright : " << bcPfaceXright << " (" << bcPName(bcPfaceXright) << ")"
       << endl;
  cout << "Yleft  : " << bcPfaceYleft << " (" << bcPName(bcPfaceYleft) << ")"
       << endl;
  cout << "Yright : " << bcPfaceYright << " (" << bcPName(bcPfaceYright) << ")"
       << endl;
  cout << "Zleft  : " << bcPfaceZleft << " (" << bcPName(bcPfaceZleft) << ")"
       << endl;
  cout << "Zright : " << bcPfaceZright << " (" << bcPName(bcPfaceZright) << ")"
       << endl;
  cout << "E inflow BCs in Maxwell image : "
       << (applyInflowBcsEImage ? "yes" : "no")
       << " (on faces with bcEMface = 2 and bcPface = 2, not X-right)" << endl;
  cout << "B forced on the y walls (ForceWallB) : "
       << (forceWallB ? "yes (GEM/ForceFree equilibrium layers)" : "no")
       << endl;
  if (bcEMfaceXright == 2 && bcPfaceXright == 3)
    cout << "X-right outflow layer fields : "
         << (outflowLayerSolved
                 ? "solved (only the ghost node and ghost cell copied)"
                 : "copy (E and B copied from the reference plane)")
         << endl;
  cout << "---------------------" << endl;
  cout << "Field Corrections" << endl;
  cout << "---------------------" << endl;
  cout << "Poisson div(E) correction  : " << PoissonCorrection;
  if (PoissonCorrection == "yes")
    cout << ", every " << PoissonCorrectionCycle
         << " cycles (in gpuCalculateE)";
  cout << endl;
  cout << "div(B) cleaning            : " << divBCorrection;
  if (divBCorrection == "yes")
    cout << ", every " << divBCorrectionCycle << " cycles (in gpuCalculateB)";
  cout << endl;
  if (divBCorrection == "yes" && yes_sal && !sal_tangential_B)
    cout << "WARNING: full B SAL with the global div(B) cleaning: the cleaning "
            "carries the SAL's B_n = B0_n at the faces into the whole box; use "
            "sal_tangential_B = 1"
         << endl;
  if (divBCorrection == "yes" && !yes_sal) {
    // Without the SAL these open faces set the whole B to B0: X-left, and the
    // Y/Z inflow faces (bcPface = 2)
    string faces;
    int count = 0;
    auto addFace = [&](const char* name, bool set) {
      if (set)
        faces += count++ ? string(", ") + name : string(name);
    };
    addFace("Xleft", !PERIODICX && bcEMfaceXleft == 2);
    addFace("Yleft", !PERIODICY && bcEMfaceYleft == 2 && bcPfaceYleft == 2);
    addFace("Yright", !PERIODICY && bcEMfaceYright == 2 && bcPfaceYright == 2);
    addFace("Zleft", !PERIODICZ && bcEMfaceZleft == 2 && bcPfaceZleft == 2);
    addFace("Zright", !PERIODICZ && bcEMfaceZright == 2 && bcPfaceZright == 2);
    if (!faces.empty())
      cout << "WARNING: open " << faces << (count > 1 ? " faces" : " face")
           << " without SAL and the global div(B) cleaning: "
           << (count > 1 ? "these faces set" : "the face sets")
           << " the whole B, B_n included, to B0 every cycle and the cleaning "
              "carries that face flux into the whole box; use yes_sal = 1 with "
              "sal_tangential_B = 1"
           << endl;
  }
  cout << "---------------------" << endl;
  cout << "Planet Boundary" << endl;
  cout << "---------------------" << endl;
  cout << "Reflection type            : " << planetReflectionType
       << (planetReflectionType == 0 ? " (specular)"
                                     : " (diffuse, Lambert cosine law)")
       << endl;
  cout << "Diagnostics cycle          : " << PlanetDiagnosticsCycle
       << (PlanetDiagnosticsCycle > 0 ? " (PlanetBC.txt)" : " (disabled)")
       << endl;
  cout << "---------------------" << endl;
  cout << "Exosphere Ionization" << endl;
  cout << "---------------------" << endl;
  if (enableExosphereInjection && numPlanetarySpecies > 0) {
    cout << "Status                     : enabled" << endl;
    cout << "Solar wind species         : " << numSolarWindSpecies
         << " (indices 0.." << numSolarWindSpecies - 1 << ")" << endl;
    cout << "Planetary species          : " << numPlanetarySpecies
         << " (indices " << numSolarWindSpecies << ".." << ns - 1 << ")"
         << endl;
    cout << "Max injection radius       : " << maxInjectionRadius << " d_i"
         << endl;
    cout << "Planet radius              : " << Planet_radius << " d_i" << endl;
    for (int i = 0; i < numPlanetarySpecies; i++) {
      int globalIdx = numSolarWindSpecies + i;
      cout << "  Species " << globalIdx << " (neutral " << i << "):" << endl;
      cout << "    NeutralSurfaceDensity    = " << neutralSurfaceDensity[i]
           << " n_sw" << endl;
      cout << "    ExosphericScaleHeight    = " << exosphericScaleHeight[i]
           << " d_i" << endl;
      cout << "    PhotoionizationFrequency = " << photoionizationFrequency[i]
           << " wci" << endl;
      cout << "    MacroParticleWeightRatio = " << macroParticleWeightRatio[i]
           << endl;
      cout << "    qom                      = " << qom[globalIdx] << endl;
      cout << "    uth/vth/wth              = " << uth[globalIdx] << " / "
           << vth[globalIdx] << " / " << wth[globalIdx] << endl;
    }
  } else {
    cout << "Status                     : disabled" << endl;
  }
  cout << "---------------------" << endl;
  cout << "Check Simulation Constraints" << endl;
  cout << "---------------------" << endl;
  cout << "Accuracy Constraint:  " << endl;
  for (int i = 0; i < ns; i++) {
    cout << "u_th < dx/dt species " << i << ".....";
    if (uth[i] < (dx / dt))
      cout << "OK" << endl;
    else
      cout << "NOT SATISFIED. STOP THE SIMULATION." << endl;

    cout << "v_th < dy/dt species " << i << "......";
    if (vth[i] < (dy / dt))
      cout << "OK" << endl;
    else
      cout << "NOT SATISFIED. STOP THE SIMULATION." << endl;
  }
  cout << endl;
  cout << "Finite Grid Stability Constraint:  ";
  cout << endl;
  for (int is = 0; is < ns; is++) {
    if (uth[is] * dt / dx > .1)
      cout << "OK u_th*dt/dx (species " << is << ") = " << uth[is] * dt / dx
           << " > .1" << endl;
    else
      cout << "WARNING. u_th*dt/dx (species " << is
           << ") = " << uth[is] * dt / dx << " < .1" << endl;

    if (vth[is] * dt / dy > .1)
      cout << "OK v_th*dt/dy (species " << is << ") = " << vth[is] * dt / dy
           << " > .1" << endl;
    else
      cout << "WARNING. v_th*dt/dy (species " << is
           << ") = " << vth[is] * dt / dy << " < .1" << endl;
  }

  cout << endl;
  cout << "CFL Condition (c*dt/dx < 1):  " << endl;
  cout << "---------------------" << endl;
  cout << "Speed of light c     = " << c << endl;
  cout << "Time step dt         = " << dt << endl;
  cout << "Grid spacing dx      = " << dx << endl;
  cout << "Grid spacing dy      = " << dy << endl;
  cout << "Grid spacing dz      = " << dz << endl;
  double cfl_x = c * dt / dx;
  double cfl_y = c * dt / dy;
  double cfl_z = c * dt / dz;
  double cfl_max = cfl_x;
  if (cfl_y > cfl_max)
    cfl_max = cfl_y;
  if (nzc > 1 && cfl_z > cfl_max)
    cfl_max = cfl_z;
  cout << "c*dt/dx              = " << cfl_x;
  if (cfl_x < 1.0)
    cout << "  OK" << endl;
  else
    cout << "  WARNING: CFL VIOLATED!" << endl;
  cout << "c*dt/dy              = " << cfl_y;
  if (cfl_y < 1.0)
    cout << "  OK" << endl;
  else
    cout << "  WARNING: CFL VIOLATED!" << endl;
  if (nzc > 1) {
    cout << "c*dt/dz              = " << cfl_z;
    if (cfl_z < 1.0)
      cout << "  OK" << endl;
    else
      cout << "  WARNING: CFL VIOLATED!" << endl;
  }
  // Multi-dimensional CFL: c*dt * sqrt(1/dx^2 + 1/dy^2 + 1/dz^2) < 1
  double cfl_multi = c * dt *
                     sqrt(1.0 / (dx * dx) + 1.0 / (dy * dy) +
                          (nzc > 1 ? 1.0 / (dz * dz) : 0.0));
  cout << "c*dt*|1/dx|          = " << cfl_multi;
  if (cfl_multi < 1.0)
    cout << "  OK" << endl;
  else
    cout << "  WARNING: multi-dim CFL VIOLATED!" << endl;

  cout << endl;
  cout << "Numerical Resolution Parameters:  " << endl;
  cout << "---------------------" << endl;
  // Larmor radius / grid spacing (using thermal velocity and B0)
  // In Gaussian CGS: Omega_s = |q/m|_s * B0 / c, so r_L = v_th * c / (|q/m| *
  // B0)
  double B0 = sqrt(B0x * B0x + B0y * B0y + B0z * B0z);
  if (B0 > 0.0) {
    for (int is = 0; is < ns; is++) {
      // Use single-component thermal speed as proxy for perpendicular v_th
      double rL = uth[is] * c / (fabs(qom[is]) * B0);
      cout << "Larmor radius / dx (species " << is << ") = " << rL / dx
           << "  (rL = " << rL
           << ", Omega_c*dt = " << fabs(qom[is]) * B0 / c * dt << ")" << endl;
    }
  }
  // Plasma frequency: omega_ps^2 = 4*pi * n_s * |q/m|_s  (CGS)
  // Since rhoINIT = 4*pi * n_s: omega_ps = sqrt(|qom| * rhoINIT)
  for (int is = 0; is < ns; is++) {
    double omega_p = sqrt(fabs(qom[is]) * rhoINIT[is]);
    double skin_depth = c / omega_p;
    double omega_p_dt = omega_p * dt;
    cout << "Species " << is << " (qom=" << qom[is]
         << "): omega_p = " << omega_p << ", d_s/dx = " << skin_depth / dx
         << ", omega_p*dt = " << omega_p_dt;
    if (omega_p_dt < 2.0)
      cout << "  OK" << endl;
    else
      cout << "  WARNING: plasma oscillations under-resolved!" << endl;
  }

  // ======= Estimated Memory Usage Per Rank =======
  cout << endl;
  cout << "Estimated Memory Per Rank" << endl;
  cout << "---------------------" << endl;

  const int nxc_loc = ceilDiv(nxc, XLEN) + 2; // +2 ghost cells
  const int nyc_loc = ceilDiv(nyc, YLEN) + 2;
  const int nzc_loc = ceilDiv(nzc, ZLEN) + 2;
  const int nxn_loc = nxc_loc + 1;
  const int nyn_loc = nyc_loc + 1;
  const int nzn_loc = nzc_loc + 1;
  const long long gridN = (long long)nxn_loc * nyn_loc * nzn_loc;
  const long long gridC = (long long)nxc_loc * nyc_loc * nzc_loc;
  const long long fieldSize =
      (long long)nzn_loc * (nyn_loc - 1) * (nxn_loc - 1);
  const double MB = 1024.0 * 1024.0;

  cout << "Local grid (with ghosts): " << nxc_loc << "x" << nyc_loc << "x"
       << nzc_loc << " cells, " << nxn_loc << "x" << nyn_loc << "x" << nzn_loc
       << " nodes" << endl;

  // Interior cell counts (no ghost layers) — used for particle estimates
  const int nxc_r = nxc_loc - 2, nyc_r = nyc_loc - 2, nzc_r = nzc_loc - 2;
  const double intC = (double)nxc_r * nyc_r * nzc_r;

  // Particle SoA arrays (host and device): x, y, z, u, v, w, q, and the ID of
  // the species with TrackParticleID, 8 bytes each
  auto soaArrays = [&](int species) {
    return 7.0 + (getTrackParticleID(species) ? 1.0 : 0.0);
  };

  // Heat flux is computed only when field output writes it (as in
  // c_Solver::initCUDA)
  const bool heatFlux =
      !field_output_is_off() && outputConfig_.needsAnyHeatFlux();

  // --- HOST ---
  // EMfields3D (its constructor): 19 node arrays (E, B, rhon, J, the 3
  // initialization temporaries, B_ext, B_tot) and 4 cell arrays (PHI, Bc);
  // per species the 10 node moments and the cell density rhocs.
  double hostEMf =
      (19.0 * gridN + 4.0 * gridC + ns * (10.0 * gridN + gridC)) * 8;
  // The HeatFlux::ComponentCount node heat-flux arrays of each species are
  // allocated only when the output writes the heat flux
  const double hostHeatFlux =
      heatFlux ? ns * HeatFlux::ComponentCount * gridN * 8.0 : 0.0;
  // Restart cell metadata when checkpoints are written: offsets and counts
  // (int) of the guarded and of the active cells, per species
  const double hostRestart =
      RestartOutputCycle > 0 ? ns * 2.0 * (gridC + intC) * 4 : 0.0;
  double hostPcl = 0, hostComm = 0;
  for (int i = 0; i < ns; i++) {
    long long nop_i = (long long)npcel[i] * nxc_r * nyc_r *
                      nzc_r; // interior cells (same as device)
    hostPcl += nop_i * soaArrays(i) * 8;
    hostComm += 0.1 * nop_i * 64.0; // MPI exchange buffer (64 B AoS)
  }
  double hostTotal =
      (hostEMf + hostHeatFlux + hostRestart + hostPcl + hostComm) / MB;

  cout << "HOST:   EMfields = " << hostEMf / MB << " MB"
       << ", particles = " << hostPcl / MB << " MB"
       << ", comm = " << hostComm / MB << " MB";
  if (hostHeatFlux > 0)
    cout << ", heat flux = " << hostHeatFlux / MB << " MB";
  if (hostRestart > 0)
    cout << ", restart metadata = " << hostRestart / MB << " MB";
  cout << endl;
  cout << "        TOTAL = " << hostTotal << " MB (" << hostTotal / 1024
       << " GB)  (+ ~250 MB runtime and libraries)" << endl;

  // --- DEVICE ---
  // GPU allocations use actual NOP from maxwellian (interior cells only, no
  // ghosts)
  const double CAP_FACTOR = 1.4, AUX_FRAC = 0.1, PLANET_FRAC = 0.05;
  bool isDipole = (Case == "Dipole" || Case == "Dipole2D");

  double devFieldBuf = fieldSize * 24.0 * 8;
  double devMoments = ns * gridN * 10.0 * 8;
  double devPcl = 0, devBuf = 0, devSort = 0, devPlanet = 0;

  for (int i = 0; i < ns; i++) {
    long long nop_i = (long long)npcel[i] * nxc_r * nyc_r *
                      nzc_r; // actual NOP (interior cells)
    long long cap_i = (long long)(nop_i * CAP_FACTOR);

    devPcl += cap_i * soaArrays(i) * 8;
    devBuf +=
        cap_i * 8.0                               // departure
        + AUX_FRAC * nop_i * (64.0 + 4.0 + 64.0); // exiting + filler + staging

    if (SortingCycle > 0 || ParticleMerging) // merging sorts by cell
      devSort += 3.0 * gridC * 4 +
                 cap_i * (4.0 + 8.0); // histograms + indices + scratch

    if (isDipole)
      devPlanet += PLANET_FRAC * nop_i * 64.0;
  }
  // GPU field solver (EMfields3D::gpuSolverAllocate): 35 node arrays (and PSI
  // with the div(B) cleaning), plus the smoothed E(n+theta) the mover samples
  // when E is smoothed, the 10 node moments of each species, 16 cell arrays,
  // the Krylov vectors (Maxwell x and b, x and b of each cleaning that is
  // on), the GMRES basis V (kGmresM + 1 vectors) and w, the lazily allocated
  // FGMRES basis Z and Chebyshev-solver / block-Jacobi vectors, the two
  // Chebyshev iterates of the cleaning preconditioner, and the halo send and
  // receive buffers (kHaloMaxBatch fields per direction).
  const bool smoothE = Smooth != 1.0 && SmoothNiter > 0; // smoothingOn()
  const double intN = (double)(nxn_loc - 2) * (nyn_loc - 2) * (nzn_loc - 2);
  const double nMaxwell = 3.0 * intN;
  const double nKrylov = nMaxwell; // the longest Krylov vectors
  const bool divE = PoissonCorrection == "yes", divB = divBCorrection == "yes";
  const bool cleaning = divE || divB;
  // the div(B) rows: the interior nodes, one row plane along a flat axis
  // (EMfields3D::divBExtent)
  const double intB = (double)(nxc == 1 && PERIODICX ? nxc_r : nxn_loc - 2) *
                      (nyc == 1 && PERIODICY ? nyc_r : nyn_loc - 2) *
                      (nzc == 1 && PERIODICZ ? nzc_r : nzn_loc - 2);
  double devSolver =
      (35.0 + (divB ? 1.0 : 0.0) + (smoothE ? 3.0 : 0.0) + 10.0 * ns) * gridN +
      16.0 * gridC + 2.0 * nMaxwell + (divE ? 2.0 * intC : 0.0) +
      (divB ? 2.0 * intB : 0.0) + (kGmresM + 2.0) * nKrylov;
  if (cleaning || SolverType == "FGMRESBlockJacobi")
    devSolver += kGmresM * nKrylov;
  if (SolverType == "Chebyshev")
    devSolver += 4.0 * nMaxwell;
  if (cleaning)
    devSolver += 2.0 * std::max(divE ? intC : 0.0, divB ? intB : 0.0);
  if (SolverType == "FGMRESBlockJacobi")
    devSolver += 2.0 * nMaxwell;
  devSolver += 2.0 * kHaloMaxBatch * 2.0 *
               ((double)nyn_loc * nzn_loc + (double)nxn_loc * nzn_loc +
                (double)nxn_loc * nyn_loc);
  devSolver *= 8;

  // Heat flux (c_Solver::initCUDA): the components and the 4 bulk moments of
  // each species, when computed
  const double devHeatFlux =
      heatFlux ? ns * (HeatFlux::ComponentCount + 4.0) * gridN * 8 : 0.0;

  double devTotal = (devFieldBuf + devMoments + devPcl + devBuf + devSort +
                     devPlanet + devSolver + devHeatFlux) /
                    MB;

  cout << "DEVICE: particles = " << devPcl / MB << " MB"
       << ", buffers = " << devBuf / MB << " MB"
       << ", moments = " << devMoments / MB << " MB"
       << ", field = " << devFieldBuf / MB << " MB"
       << ", field solver = " << devSolver / MB << " MB";
  if (devSort > 0)
    cout << ", sort = " << devSort / MB << " MB";
  if (devPlanet > 0)
    cout << ", planet = " << devPlanet / MB << " MB";
  if (devHeatFlux > 0)
    cout << ", heat flux = " << devHeatFlux / MB << " MB";
  cout << endl;
  cout << "        TOTAL = " << devTotal << " MB (" << devTotal / 1024 << " GB)"
       << "  (+ 500 MB CUDA context)" << endl;

  cout << "COMBINED = " << (hostTotal + devTotal) << " MB ("
       << (hostTotal + devTotal) / 1024 << " GB)" << endl;
  cout << "---------------------" << endl;
}
/*! Write the build record and simulation parameters to SimulationData.txt */
void Collective::save(const std::string& header) {
  string temp;
  temp = SaveDirName + "/SimulationData.txt";
  ofstream my_file(temp.c_str());
  my_file << header;
  my_file << "---------------------------" << endl;
  my_file << "-  Simulation Parameters  -" << endl;
  my_file << "---------------------------" << endl;

  my_file << "Number of species    = " << ns << endl;
  for (int i = 0; i < ns; i++)
    my_file << "qom[" << i << "] = " << qom[i] << endl;
  my_file << "---------------------------" << endl;
  my_file << "x-Length                 = " << Lx << endl;
  my_file << "y-Length                 = " << Ly << endl;
  my_file << "z-Length                 = " << Lz << endl;
  my_file << "Number of cells (x)      = " << nxc << endl;
  my_file << "Number of cells (y)      = " << nyc << endl;
  my_file << "Number of cells (z)      = " << nzc << endl;
  my_file << "---------------------------" << endl;
  my_file << "Time step                = " << dt << endl;
  my_file << "Number of cycles         = " << ncycles << endl;
  my_file << "---------------------------" << endl;
  for (int is = 0; is < ns; is++) {
    my_file << "rho init species   " << is << " = " << rhoINIT[is] << endl;
    my_file << "rho inject species " << is << " = " << rhoINJECT[is] << endl;
  }
  my_file << "current sheet thickness  = " << delta << endl;
  my_file << "B0x                      = " << B0x << endl;
  my_file << "BOy                      = " << B0y << endl;
  my_file << "B0z                      = " << B0z << endl;
  my_file << "---------------------------" << endl;
  my_file << "Smooth                   = " << Smooth << endl;
  my_file << "SmoothNiter              = " << SmoothNiter << endl;
  my_file << "HaloOverlap              = " << haloOverlap << endl;
  my_file << "SolverSourceGrowthLimit  = " << solverSourceGrowthLimit << endl;
  my_file << "SolverStopOnNonConvergence = " << solverStopOnNonConvergence
          << endl;
  my_file << "GMRES error tolerance    = " << GMREStol << endl;
  my_file << "DivCleaningTol           = " << DivCleaningTol << endl;
  my_file << "CG error tolerance       = " << CGtol << endl;
  my_file << "Mover iterations         = " << NiterMover << endl;
  // the particles the particle output writes, as applied: a fraction only of
  // a species with IDs, no cap with the fraction on
  my_file << "ParticlesOutputTrackedFraction =";
  for (int s = 0; s < ns; s++)
    my_file << " "
            << (particleOutputFractionOn(s) ? particleOutputFraction(s) : -1.0);
  my_file << " (-1: off)" << endl;
  my_file << "ParticlesOutputMaxPerCell =";
  for (int s = 0; s < ns; s++)
    my_file << " " << particleOutputMaxPerCell(s);
  my_file << " (0: every particle)" << endl;
  my_file << "---------------------------" << endl;
  my_file << "Results saved in: " << SaveDirName << endl;
  my_file << "Restart saved in: " << RestartDirName << endl;
  my_file << "---------------------" << endl;
  my_file.close();
}
