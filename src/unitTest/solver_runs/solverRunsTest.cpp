/**
 * @file solverRunsTest.cpp
 * @brief The whole solver: short runs of the iPIC3D executable on the shipped
 *        decks, reduced, each started through the launcher of the unit tests.
 *
 *   reconnection    GEM with every output, the data analysis, both cleanings
 *                   and particle control: the energy stays bounded, every
 *                   output is written; a restart continues the diagnostics.
 *   restart         a run cut in two by a restart against the same run
 *                   without one, at the cycle of the restart: the same E and
 *                   particle energies, B within what its rebuild changes.
 *   deposit         a cold drifting plasma, sorted by cell and not: the bulk
 *                   energy of the deposited moments is the kinetic energy of
 *                   the particles, and the two deposits give the same run.
 *   decomposition   the same run on 1 and on 4 ranks, in a closed box, which
 *                   keeps every particle, and with injection on open faces.
 *   magnetosphere   a planet, open faces, injection and the exosphere, on the
 *                   other particle and restart backend; a restart in place.
 *   field writers   each field backend of the build writes its files.
 *   cases           every initial condition runs and stays finite.
 *   test particles  the test species of a deck are written.
 *
 * Usage: solverRunsTest <iPIC3D> <launch.sh> <share/inputfiles> <HDF5 0|1>
 *                       <parallel HDF5 0|1> <ADIOS2 0|1>
 * The runs are in ./solver_runs_work, removed when every check passes.
 */

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

namespace fs = std::filesystem;

constexpr int kRanks = 4;
/** Host threads of a rank: the ranks share the node. */
constexpr const char* kThreads = "2";
/** Largest change of the total energy of the closed reconnection runs, over
 *  their few cycles. */
constexpr double kMaxEnergyDrift = 0.05;
/** At the cycle of the restart, against the same run without it: the largest
 *  relative difference of the electric and kinetic energies (the digits of the
 *  diagnostics file), and of the magnetic one: a checkpoint holds the node B,
 *  from which the restart rebuilds the cell B. */
constexpr double kMaxRestartDifference = 1e-6;
constexpr double kMaxRestartBDifference = 0.05;
/** The same energies between 1 and 4 ranks: a few units of the seven digits
 *  the diagnostics file keeps. */
constexpr double kMaxDecompositionDifference = 1e-5;
/** Bulk over kinetic energy of a cold plasma, minus 1: the digits of the
 *  diagnostics file. */
constexpr double kMaxDepositDifference = 1e-5;
/** Lines of a failed run's log that are printed. */
constexpr size_t kLogTail = 25;

using Keys = std::vector<std::pair<std::string, std::string>>;

struct Build {
  std::string binary, launch;
  fs::path decks, work;
  bool hdf5, phdf5, adios2;
};

int failures = 0;

void check(bool ok, const std::string& what) {
  std::cout << "  [" << (ok ? "ok" : "FAIL") << "] " << what << std::endl;
  failures += !ok;
}

Keys operator+(Keys a, const Keys& b) {
  a.insert(a.end(), b.begin(), b.end());
  return a;
}

std::string quoted(const std::string& text) {
  std::string out = "'";
  for (const char c : text)
    out += c == '\'' ? std::string("'\\''") : std::string(1, c);
  return out + "'";
}

/** The shipped deck with the keys replaced, or added. */
std::string deckText(const fs::path& deck, const Keys& keys) {
  std::ifstream in(deck);
  if (!in)
    throw std::runtime_error("cannot read " + deck.string());
  std::vector<std::string> lines;
  for (std::string line; std::getline(in, line);)
    lines.push_back(line);
  for (const auto& [key, value] : keys) {
    const std::string entry = key + " = " + value;
    const auto holds = [&, &name = key](const std::string& line) {
      if (line.compare(0, name.size(), name) != 0)
        return false;
      const size_t next = line.find_first_not_of(" \t", name.size());
      return next != std::string::npos && line[next] == '=';
    };
    const auto it = std::find_if(lines.begin(), lines.end(), holds);
    if (it != lines.end())
      *it = entry;
    else
      lines.push_back(entry);
  }
  std::string text;
  for (const std::string& line : lines)
    text += line + "\n";
  return text;
}

/** Runs the deck in dir on the ranks; true if the run ended normally. */
bool runSolver(const Build& build, const fs::path& dir, const std::string& deck,
               const Keys& keys, int ranks, bool restart = false) {
  fs::create_directories(dir);
  std::ofstream(dir / "deck.inp") << deckText(build.decks / deck, keys);
  const std::string log = restart ? "restart.log" : "run.log";
  const std::string command =
      "cd " + quoted(dir.string()) + " && sh " + quoted(build.launch) + " " +
      std::to_string(ranks) + " " + quoted(build.binary) + " deck.inp" +
      (restart ? " restart" : "") + " > " + log + " 2>&1";
  const bool ok = std::system(command.c_str()) == 0;
  check(ok, dir.filename().string() + (restart ? ": restart ends" : ": run ends"));
  if (!ok) {
    std::ifstream in(dir / log);
    std::vector<std::string> lines;
    for (std::string line; std::getline(in, line);)
      lines.push_back(line);
    const size_t first = lines.size() > kLogTail ? lines.size() - kLogTail : 0;
    for (size_t i = first; i < lines.size(); i++)
      std::cout << "    " << lines[i] << "\n";
  }
  return ok;
}

/** A table of the diagnostics: a header, then one row per cycle. */
struct Table {
  std::vector<std::string> names;
  std::vector<std::vector<double>> rows;

  size_t column(const std::string& name) const {
    const auto it = std::find(names.begin(), names.end(), name);
    if (it == names.end())
      throw std::runtime_error("no column " + name);
    return it - names.begin();
  }
  const std::vector<double>* rowAt(int cycle) const {
    const std::vector<double>* found = nullptr;
    for (const auto& row : rows)
      if (static_cast<int>(row[0]) == cycle)
        found = &row; // the last one written for that cycle
    return found;
  }
  int lastCycle() const {
    return rows.empty() ? -1 : static_cast<int>(rows.back()[0]);
  }
  bool finite() const {
    for (const auto& row : rows)
      for (const double value : row)
        if (!std::isfinite(value))
          return false;
    return !rows.empty();
  }
};

Table readTable(const fs::path& file) {
  Table table;
  std::ifstream in(file);
  std::string line;
  if (!std::getline(in, line))
    return table;
  std::istringstream header(line);
  for (std::string name; header >> name;)
    table.names.push_back(name);
  while (std::getline(in, line)) {
    std::istringstream row(line);
    std::vector<double> values;
    for (std::string token; row >> token;)
      values.push_back(std::strtod(token.c_str(), nullptr));
    if (values.size() == table.names.size())
      table.rows.push_back(std::move(values));
  }
  return table;
}

Table readConserved(const fs::path& dir) {
  return readTable(dir / "data" / "ConservedQuantities.txt");
}

/** The largest difference at the cycle of the named energies, each relative
 *  to its value in a. */
double energyDifference(const Table& a, const Table& b, int cycle,
                        const std::vector<std::string>& names) {
  const std::vector<double>* const ra = a.rowAt(cycle);
  const std::vector<double>* const rb = b.rowAt(cycle);
  if (!ra || !rb)
    return HUGE_VAL;
  double worst = 0;
  for (const std::string& name : names) {
    const double value = (*ra)[a.column(name)];
    worst = std::max(worst, std::fabs(value - (*rb)[b.column(name)]) /
                                std::fabs(value));
  }
  return worst;
}

/** A non-empty file of the folder whose name holds the text. */
bool holdsFile(const fs::path& dir, const std::string& text) {
  std::error_code error;
  for (const auto& entry : fs::directory_iterator(dir, error))
    if (entry.path().filename().string().find(text) != std::string::npos &&
        (entry.is_directory() || entry.file_size() > 0))
      return true;
  return false;
}

std::string sci(double value) {
  char text[32];
  std::snprintf(text, sizeof(text), "%.2e", value);
  return text;
}

// The reduced decks
const char* const kGem2D = "magneticReconnection/testGEM2Dsmall.inp";
const char* const kExosphere =
    "magnetosphere/testMagnetosphere3DsmallExosphereIonization.inp";

/** GEM on 32 x 16 cells, two particles per cell and axis. */
Keys gemKeys(int xlen, int ylen, int cycles) {
  return {{"nxc", "32"},
          {"nyc", "16"},
          {"XLEN", std::to_string(xlen)},
          {"YLEN", std::to_string(ylen)},
          {"ZLEN", "1"},
          {"ncycles", std::to_string(cycles)},
          {"npcelx", "2 2 2 2"},
          {"npcely", "2 2 2 2"},
          {"npcelz", "2 2 2 2"}};
}

/** No output but the diagnostics of every other cycle. */
const Keys kQuiet = {{"FieldWriteMethod", "pvtk"},   {"FieldOutputCycle", "0"},
                     {"DiagnosticsOutputCycle", "2"}, {"ParticlesOutputCycle", "0"},
                     {"RestartOutputCycle", "0"},    {"DataAnalysisCycle", "0"},
                     {"CallFinalize", "0"}};

/** The particle and restart backend: the first the build has. */
std::string storeBackend(const Build& build) {
  return build.adios2 ? "adios2" : build.hdf5 ? "shdf5" : "";
}

void testReconnection(const Build& build) {
  std::cout << "reconnection\n";
  const fs::path dir = build.work / "reconnection";
  const std::string store = storeBackend(build);
  const Keys keys =
      gemKeys(2, 2, 12) +
      Keys{{"PoissonCorrection", "yes"},
           {"PoissonCorrectionCycle", "3"},
           {"divBCorrection", "yes"},
           {"divBCorrectionCycle", "3"},
           {"FieldWriteMethod", build.hdf5 ? "shdf5" : "pvtk"},
           {"FieldOutputCycle", "4"},
           {"FieldOutputTag", "B+E+Je+Ji+rho"},
           {"MomentsOutputTag", "rho+J+P+Q+rho_tot+J_tot+P_tot+Q_tot"},
           {"ParticlesOutputCycle", store.empty() ? "0" : "4"},
           {"ParticlesOutputTag", "position+velocity+q+ID"},
           {"RestartOutputCycle", store.empty() ? "0" : "6"},
           {"DiagnosticsOutputCycle", "2"},
           {"DataAnalysisCycle", "4"},
           {"VelocityHistogram", "1"},
           {"VelocityGMM", "1"},
           {"GMMComponents", "3"},
           {"VelocitySpectra", "1"},
           {"MacrocellNx", "4"},
           {"MacrocellNy", "4"},
           {"MacrocellNz", "1"},
           {"VelocitySpectraSpecies", "1 1 0 0"},
           {"ParticleSplitting", "1"},
           {"ParticleSplittingThreshold", "1.5"},
           {"ParticleMerging", "1"},
           {"ParticleMergingThreshold", "1.2"}} +
      (store.empty() ? Keys{{"CallFinalize", "0"}}
                     : Keys{{"ParticlesWriteMethod", store},
                            {"RestartWriteMethod", store}});
  if (!runSolver(build, dir, kGem2D, keys, kRanks))
    return;
  const Table first = readConserved(dir);
  check(first.finite() && first.lastCycle() == 10,
        "diagnostics finite up to cycle 10, last " +
            std::to_string(first.lastCycle()));
  if (first.finite()) {
    const size_t total = first.column("Total_Energy");
    const double drift = std::fabs(first.rows.back()[total] / first.rows[0][total] - 1);
    check(drift < kMaxEnergyDrift, "total energy bounded: change " + sci(drift));
  }
  const fs::path data = dir / "data";
  if (build.hdf5)
    check(holdsFile(data, "proc0.hdf"), "serial-HDF5 field file");
  if (build.adios2)
    check(holdsFile(data, "particle_0.bp"), "ADIOS2 particle file");
  for (const char* folder : {"velocityHistogram", "velocityGMM", "macrocellSpectra"})
    check(holdsFile(dir / folder, "subDomain0"), std::string(folder) + " records");
  if (store.empty())
    return;
  check(holdsFile(data, "latest_restart.json"), "checkpoint manifest");
  if (!runSolver(build, dir, kGem2D, keys + Keys{{"ncycles", "6"}}, kRanks, true))
    return;
  const Table next = readConserved(dir);
  check(next.finite() && next.lastCycle() == 16,
        "the restart continues the diagnostics to cycle 16, last " +
            std::to_string(next.lastCycle()));
}

void testRestart(const Build& build) {
  const std::string store = storeBackend(build);
  if (store.empty())
    return;
  std::cout << "restart\n";
  const Keys common = kQuiet + Keys{{"RestartWriteMethod", store},
                                    {"RestartOutputCycle", "4"},
                                    {"CallFinalize", "1"}};
  const fs::path whole = build.work / "restart_whole";
  const fs::path cut = build.work / "restart_cut";
  if (!runSolver(build, whole, kGem2D, gemKeys(2, 2, 8) + common, kRanks) ||
      !runSolver(build, cut, kGem2D, gemKeys(2, 2, 4) + common, kRanks) ||
      !runSolver(build, cut, kGem2D, gemKeys(2, 2, 4) + common, kRanks, true))
    return;
  const Table a = readConserved(whole), b = readConserved(cut);
  const double exact = energyDifference(a, b, 4, {"Eenergy", "Kenergy"});
  check(exact < kMaxRestartDifference,
        "E and particle energies at the restart cycle: " + sci(exact));
  const double magnetic = energyDifference(a, b, 4, {"Benergy"});
  check(magnetic < kMaxRestartBDifference,
        "B energy at the restart cycle, rebuilt from node B: " + sci(magnetic));
}

void testDeposit(const Build& build) {
  std::cout << "deposit\n";
  const Keys cold = gemKeys(2, 2, 5) + kQuiet +
                    Keys{{"uth", "0 0 0 0"},
                         {"vth", "0 0 0 0"},
                         {"wth", "0 0 0 0"},
                         {"u0", "0.02 0.01 0.03 0.015"},
                         {"v0", "-0.01 0.02 0.01 -0.02"},
                         {"w0", "0.015 -0.01 -0.02 0.01"}};
  // the deposit of the particles sorted by cell, and of the unsorted ones
  std::vector<Table> tables;
  for (const char* sorting : {"1", "0"}) {
    const std::string label = sorting[0] == '1' ? "sorted" : "unsorted";
    const fs::path dir = build.work / ("deposit_" + label);
    if (!runSolver(build, dir, kGem2D, cold + Keys{{"SortingCycle", sorting}},
                   kRanks))
      continue;
    tables.push_back(readConserved(dir));
    const Table& table = tables.back();
    const std::vector<double>* const row = table.rowAt(0);
    if (!row) {
      check(false, label + ": no diagnostics at cycle 0");
      continue;
    }
    double worst = 0;
    for (int is = 0; is < 4; is++) {
      const std::string index = "[" + std::to_string(is) + "]";
      const double kinetic = (*row)[table.column("Ke" + index)];
      const double bulk = (*row)[table.column("BulkE" + index)];
      worst = std::max(worst, std::fabs(bulk / kinetic - 1));
    }
    check(worst < kMaxDepositDifference,
          label + ": bulk energy of the moments = kinetic energy: " + sci(worst));
  }
  if (tables.size() == 2) {
    const double paths = energyDifference(tables[0], tables[1], 4,
                                          {"Total_Energy", "Benergy", "Kenergy"});
    check(paths < kMaxDecompositionDifference,
          "the two deposits give the same run at cycle 4: " + sci(paths));
  }
}

/** Per cycle, the particles of every species summed over the rank files
 *  data/particleNum<rank>.csv. */
std::map<int, std::vector<long long>> particleCounts(const fs::path& dir) {
  std::map<int, std::vector<long long>> counts;
  for (const auto& entry : fs::directory_iterator(dir / "data")) {
    const std::string name = entry.path().filename().string();
    if (name.rfind("particleNum", 0) != 0)
      continue;
    std::ifstream in(entry.path());
    std::string line;
    std::getline(in, line); // the header
    while (std::getline(in, line)) {
      std::replace(line.begin(), line.end(), ',', ' ');
      std::istringstream row(line);
      int cycle;
      row >> cycle;
      std::vector<long long>& total = counts[cycle];
      size_t species = 0;
      for (long long value; row >> value; species++) {
        if (total.size() <= species)
          total.push_back(0);
        total[species] += value;
      }
    }
  }
  return counts;
}

void testDecomposition(const Build& build) {
  std::cout << "decomposition\n";
  const fs::path one = build.work / "decomposition_1";
  const fs::path four = build.work / "decomposition_4";
  // a small hot box: many particles cross the rank faces every cycle
  const Keys crossing = kQuiet + Keys{{"Lx", "4"},
                                      {"Ly", "2"},
                                      {"uth", "0.1 0.05 0.1 0.05"},
                                      {"vth", "0.1 0.05 0.1 0.05"},
                                      {"wth", "0.1 0.05 0.1 0.05"}};
  if (!runSolver(build, one, kGem2D, gemKeys(1, 1, 7) + crossing, 1) ||
      !runSolver(build, four, kGem2D, gemKeys(2, 2, 7) + crossing, kRanks))
    return;
  const double closed =
      energyDifference(readConserved(one), readConserved(four), 6,
                       {"Total_Energy", "Benergy", "Kenergy"});
  check(closed < kMaxDecompositionDifference,
        "closed box, 1 and 4 ranks agree at cycle 6: " + sci(closed));
  const auto counts = particleCounts(four);
  bool kept = counts.size() == 7;
  for (const auto& [cycle, total] : counts)
    kept = kept && total == counts.begin()->second;
  check(kept, "closed box: every particle kept over the " +
                  std::to_string(counts.size()) + " cycles");

  // open faces: the injected plasma does not depend on the decomposition
  const Keys open = kQuiet + Keys{{"nxc", "24"},
                                  {"nyc", "24"},
                                  {"nzc", "24"},
                                  {"ZLEN", "1"},
                                  {"ncycles", "5"},
                                  {"npcelx", "2 2 0 0 0 0"},
                                  {"npcely", "2 2 0 0 0 0"},
                                  {"npcelz", "2 2 0 0 0 0"},
                                  {"AddExosphereInjection", "0"},
                                  {"VelocitySpectra", "0"}};
  const fs::path openOne = build.work / "open_1";
  const fs::path openFour = build.work / "open_4";
  if (!runSolver(build, openOne, kExosphere,
                 open + Keys{{"XLEN", "1"}, {"YLEN", "1"}}, 1) ||
      !runSolver(build, openFour, kExosphere,
                 open + Keys{{"XLEN", "2"}, {"YLEN", "2"}}, kRanks))
    return;
  const double injected =
      energyDifference(readConserved(openOne), readConserved(openFour), 4,
                       {"Total_Energy", "Benergy", "Kenergy"});
  check(injected < kMaxDecompositionDifference,
        "open faces, 1 and 4 ranks agree at cycle 4: " + sci(injected));
}

void testMagnetosphere(const Build& build) {
  std::cout << "magnetosphere\n";
  const fs::path dir = build.work / "magnetosphere";
  // the other backend than the reconnection run where the build has it
  const std::string store = build.hdf5 ? "shdf5" : storeBackend(build);
  const Keys keys =
      Keys{{"nxc", "24"},
           {"nyc", "24"},
           {"nzc", "24"},
           {"XLEN", "1"},
           {"YLEN", "2"},
           {"ZLEN", "2"},
           {"ncycles", "6"},
           {"npcelx", "2 2 0 0 0 0"},
           {"npcely", "2 2 0 0 0 0"},
           {"npcelz", "2 2 0 0 0 0"},
           {"TrackParticleID", "1 1 1 1 1 1"},
           {"FieldWriteMethod", build.hdf5 ? "shdf5" : "pvtk"},
           {"FieldOutputCycle", "2"},
           {"DiagnosticsOutputCycle", "2"},
           {"PlanetDiagnosticsCycle", "2"},
           {"divBCorrectionCycle", "2"},
           {"DataAnalysisCycle", "2"},
           {"MacrocellNx", "4"},
           {"MacrocellNy", "4"},
           {"MacrocellNz", "4"},
           {"ParticlesOutputCycle", store.empty() ? "0" : "2"},
           {"RestartOutputCycle", store.empty() ? "0" : "3"}} +
      (store.empty() ? Keys{{"CallFinalize", "0"}}
                     : Keys{{"ParticlesWriteMethod", store},
                            {"RestartWriteMethod", store}});
  if (!runSolver(build, dir, kExosphere, keys, kRanks))
    return;
  const Table table = readConserved(dir);
  check(table.finite() && table.lastCycle() == 4,
        "diagnostics finite up to cycle 4, last " +
            std::to_string(table.lastCycle()));
  const Table planet = readTable(dir / "data" / "PlanetBC.txt");
  double hits = 0;
  for (const auto& row : planet.rows)
    for (size_t c = 0; c < planet.names.size(); c++)
      if (planet.names[c].rfind("Nhit", 0) == 0)
        hits += row[c];
  check(hits > 0, "particles reach the planet: " + sci(hits));
  if (store.empty())
    return;
  if (!runSolver(build, dir, kExosphere, keys + Keys{{"ncycles", "3"}}, kRanks,
                 true))
    return;
  const Table next = readConserved(dir);
  check(next.finite() && next.lastCycle() == 8,
        "the restart in place continues to cycle 8, last " +
            std::to_string(next.lastCycle()));
}

void testFieldWriters(const Build& build) {
  std::cout << "field writers\n";
  std::vector<std::pair<std::string, std::string>> writers = {{"pvtk", ".vtk"},
                                                              {"nbcvtk", ".vtk"}};
  if (build.phdf5)
    writers.push_back({"phdf5", ".h5"});
  for (const auto& [method, suffix] : writers) {
    const fs::path dir = build.work / ("writer_" + method);
    const Keys keys = gemKeys(2, 2, 3) + kQuiet +
                      Keys{{"FieldWriteMethod", method},
                           {"FieldOutputCycle", "2"},
                           {"FieldOutputTag", "B+E"},
                           {"MomentsOutputTag", "rho+P_tot"}};
    if (runSolver(build, dir, kGem2D, keys, kRanks))
      check(holdsFile(dir / "data", suffix), method + " field files");
  }
}

void testCases(const Build& build) {
  std::cout << "cases\n";
  const Keys tiny2D = gemKeys(2, 2, 3) + kQuiet;
  const Keys tiny3D = kQuiet + Keys{{"nxc", "16"},
                                    {"nyc", "16"},
                                    {"nzc", "16"},
                                    {"XLEN", "2"},
                                    {"YLEN", "2"},
                                    {"ZLEN", "1"},
                                    {"ncycles", "3"}};
  const std::vector<std::pair<std::string, std::pair<std::string, Keys>>> runs = {
      {"GEMnoPert", {kGem2D, tiny2D + Keys{{"Case", "GEMnoPert"}}}},
      {"GEMDoubleHarris", {kGem2D, tiny2D + Keys{{"Case", "GEMDoubleHarris"}}}},
      {"GEMHarris", {kGem2D, tiny2D + Keys{{"Case", "GEMHarris"}}}},
      {"HumpPert", {kGem2D, tiny2D + Keys{{"Case", "HumpPert"}}}},
      {"RandomCase", {kGem2D, tiny2D + Keys{{"Case", "RandomCase"}}}},
      {"Default", {kGem2D, tiny2D + Keys{{"Case", "NoSuchCase"}}}},
      {"ForceFree",
       {"magneticReconnection/testForceFree2D.inp",
        tiny2D + Keys{{"npcelx", "2 2"}, {"npcely", "2 2"}, {"npcelz", "2 2"}}}},
      {"NullPoints", {"NullPoints.inp", tiny3D}},
      {"TaylorGreen", {"turbulence/TaylorGreen.inp", tiny3D}},
      {"Dipole2D",
       {"magnetosphere/testMagnetosphere2Dsmall.inp",
        kQuiet + Keys{{"nxc", "32"},
                      {"nyc", "32"},
                      {"XLEN", "2"},
                      {"YLEN", "2"},
                      {"ZLEN", "1"},
                      {"ncycles", "3"}}}}};
  for (const auto& [name, run] : runs) {
    const fs::path dir = build.work / ("case_" + name);
    if (runSolver(build, dir, run.first, run.second, kRanks))
      check(readConserved(dir).finite(), name + ": diagnostics finite");
  }
}

void testParticlesOfTestSpecies(const Build& build) {
  if (!build.hdf5)
    return; // written with serial HDF5 only
  std::cout << "test particles\n";
  const fs::path dir = build.work / "test_particles";
  const Keys keys = kQuiet + Keys{{"nxc", "16"},
                                  {"nyc", "16"},
                                  {"nzc", "8"},
                                  {"XLEN", "2"},
                                  {"YLEN", "2"},
                                  {"ZLEN", "1"},
                                  {"ncycles", "3"},
                                  {"TestPartOutputCycle", "1"}};
  if (runSolver(build, dir, "Ram1TestParticle.inp", keys, kRanks))
    check(holdsFile(dir / "data", "proc0.hdf"), "test particles written");
}

} // namespace

int main(int argc, char** argv) {
  if (argc != 7) {
    std::cerr << "usage: solverRunsTest <iPIC3D> <launch.sh> <share/inputfiles>"
                 " <HDF5 0|1> <parallel HDF5 0|1> <ADIOS2 0|1>\n";
    return 2;
  }
  try {
    const Build build = {argv[1],
                         argv[2],
                         argv[3],
                         fs::absolute("solver_runs_work"),
                         std::string(argv[4]) == "1",
                         std::string(argv[5]) == "1",
                         std::string(argv[6]) == "1"};
    setenv("OMP_NUM_THREADS", kThreads, 0);
    fs::remove_all(build.work);
    testReconnection(build);
    testRestart(build);
    testDeposit(build);
    testDecomposition(build);
    testMagnetosphere(build);
    testFieldWriters(build);
    testCases(build);
    testParticlesOfTestSpecies(build);
    if (failures == 0)
      fs::remove_all(build.work);
  } catch (const std::exception& error) {
    std::cerr << "solverRunsTest: " << error.what() << "\n";
    return 1;
  }
  std::cout << (failures == 0 ? "PASSED" : "FAILED") << " (" << failures
            << " failure" << (failures == 1 ? "" : "s") << ")\n";
  return failures == 0 ? 0 : 1;
}
