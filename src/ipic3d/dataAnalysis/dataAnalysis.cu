#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <future>
#include <iostream>
#include <memory>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

#include "BuildInfo.h"
#include "Collective.h"
#include "Grid3DCU.h"
#include "MPIdata.h"
#include "VCtopology3D.h"
#include "iPic3D.h"
#include "outputPrepare.h"
#include "threadPool.hpp"

#include "GMM/cudaGMM.cuh"
#include "cellSortBuffers.cuh"
#include "dataAnalysis.cuh"
#include "dataAnalysisConfig.cuh"
#include "macrocellPartition.cuh"
#include "macrocellSpectra.cuh"
#include "velocityHistogram.cuh"

#include "mpi.h"

namespace dataAnalysis {

/** Dimension of the Gaussian mixture; its components are GMMComponents. */
inline constexpr int kDim = DAConfig::DATA_DIM_GMM;
static_assert(kDim == 3, "the GMM fits (u, v, w) histograms");
static_assert(kGMMMaxComponents <= cudaGMMWeight::kGMMThreads,
              "the components a GMM fits");
/** A warm start keeps a component of the previous fit of weight at least
 *  this times GMMPruneThreshold, or this share of the equal weight
 *  1 / GMMComponents if smaller (never below GMMPruneThreshold), and draws
 *  the others again. */
inline constexpr double kWarmStartWeightFactor = 10;
inline constexpr double kWarmStartEqualShare = 0.5;
/** Doubles per point of the GMM (GMMComponents responsibilities and the
 *  weighted log-likelihood) of the regions fitted at once at most: 2^21
 *  points with 10 components, ~185 MB (~230 MB allocated, with the margin of
 *  the GMM buffers). A larger region is fitted alone: the histogram of a
 *  rank, up to kNumBins points, takes 8 MB per component and 8 MB. */
inline constexpr int kGMMBatchDoubles = 11 * (1 << 21);
/** The device memory of a fit beyond its points, per component, counted in
 *  the batch as that of this many points: its log weight and normalisation,
 *  mean, covariance and Cholesky factor, and one block of M-step
 *  statistics. */
inline constexpr int kGMMFitPoints =
    2 + kDim + 2 * kDim * kDim + cudaGMMWeightKernel::kMStepStatistics<kDim>;
/** The first word of a record of the GMM fits, "GMM1" in the little-endian
 *  file (GMMRecordHeader). */
inline constexpr std::uint32_t kGMMRecordMagic = 0x314d4d47;

using namespace iPic3D;
using velocityHistogram::velocitySoA;
using GMMType = cudaParticleType;
using weightType = velocityHistogram::histogramTypeOut;
using GMMModel = cudaGMMWeight::GMM<GMMType, kDim, weightType>;
using GMMFit = cudaGMMWeight::GMMResult<GMMType, kDim>;

namespace {

/** The header of a record of the GMM fits of a cycle (RecordFile), followed
 *  by the arrays of writeGMMRecord. */
struct GMMRecordHeader {
  std::uint32_t magic; // kGMMRecordMagic
  std::int32_t cycle;
  std::uint32_t species;
  std::uint32_t regions; // per species: 1 (the rank), or the macrocells
  std::uint32_t components;
  std::uint32_t reserved;
  std::uint64_t bytes; // of the record, this header and padding included
};

/** Destroys an owned stream. */
struct StreamDeleter {
  void operator()(cudaStream_t stream) const {
    cudaErrChk(cudaStreamDestroy(stream));
  }
};
using OwnedStream =
    std::unique_ptr<std::remove_pointer_t<cudaStream_t>, StreamDeleter>;

/** Rank 0 prepares the folder shared by the ranks before they write in it
 *  (an error of rank 0 ends every rank, see MPIdata): emptied on a fresh
 *  run, kept on a restart. */
void prepareSharedFolder(const std::string& path, int rank, bool isRestart) {
  if (rank == 0) {
    if (isRestart)
      ensureOutputFolder(path);
    else
      checkOutputFolder(path);
  }
  MPI_Barrier(MPIdata::get_PicGlobalComm());
}

/** Rank 0 warns of the records files that the ranks kept aside, those of
 *  `kept` ranks by this one, which are why (collective). */
void warnKeptAside(int kept, const std::string& folder, int firstCycle,
                   const char* why) {
  const int mine = kept;
  int total = 0;
  MPI_Reduce(&mine, &total, 1, MPI_INT, MPI_SUM, 0,
             MPIdata::get_PicGlobalComm());
  if (MPIdata::get_rank() == 0 && total > 0)
    std::fprintf(stderr,
                 "[!] Warning: the records files of %d ranks in %s %s: they "
                 "are kept as *_before_cycle%d.*\n",
                 total, folder.c_str(), why, firstCycle);
}

/** The place of this rank in the MPI topology, next to its analysis files. */
void writeVctMapping(const std::string& filePath, VirtualTopology3D* vct) {
  std::ofstream out(filePath);
  if (!out.is_open())
    throw std::runtime_error("[!]Error: Can not create VCT mapping file " +
                             filePath);
  out << "Cartesian rank: " << vct->getCartesian_rank() << "\n"
      << "Number of processes: " << vct->getNprocs() << "\n"
      << "XLEN: " << vct->getXLEN() << "\n"
      << "YLEN: " << vct->getYLEN() << "\n"
      << "ZLEN: " << vct->getZLEN() << "\n"
      << "X: " << vct->getCoordinates(0) << "\n"
      << "Y: " << vct->getCoordinates(1) << "\n"
      << "Z: " << vct->getCoordinates(2) << "\n"
      << "PERIODICX: " << vct->getPERIODICX() << "\n"
      << "PERIODICY: " << vct->getPERIODICY() << "\n"
      << "PERIODICZ: " << vct->getPERIODICZ() << "\n"
      << "Neighbor X left: " << vct->getXleft_neighbor() << "\n"
      << "Neighbor X right: " << vct->getXright_neighbor() << "\n"
      << "Neighbor Y left: " << vct->getYleft_neighbor() << "\n"
      << "Neighbor Y right: " << vct->getYright_neighbor() << "\n"
      << "Neighbor Z left: " << vct->getZleft_neighbor() << "\n"
      << "Neighbor Z right: " << vct->getZright_neighbor() << "\n";
  out.close();
  if (!out)
    throw std::runtime_error("[!]Error: Can not write the VCT mapping file " +
                             filePath);
}

/** The JSON members of the records files of a rank on its place in the MPI
 *  topology and on the build that wrote them (describeRecords). */
std::string rankDescription(int rank, const VCtopology3D* vct) {
  std::ostringstream out;
  out << "  \"subdomain_rank\": " << rank << ",\n"
      << "  \"cartesian_rank\": " << vct->getCartesian_rank() << ",\n"
      << "  \"cartesian_coords\": [" << vct->getCoordinates(0) << ","
      << vct->getCoordinates(1) << "," << vct->getCoordinates(2) << "],\n"
      << "  \"topology\": [" << vct->getXLEN() << "," << vct->getYLEN() << ","
      << vct->getZLEN() << "],\n"
      << "  \"ipic3d_version\": \"" << iPic3D::build::version() << "\",\n"
      << "  \"ipic3d_build\": \"" << iPic3D::build::signature() << "\",\n";
  return out.str();
}

/** The decomposition "XLEN YLEN ZLEN" of a writeVctMapping file, "" without
 *  one. */
std::string mappingDecomposition(const std::string& filePath) {
  std::ifstream in(filePath);
  std::string line, decomposition;
  while (std::getline(in, line))
    for (const std::string key : {"XLEN: ", "YLEN: ", "ZLEN: "})
      if (line.rfind(key, 0) == 0)
        decomposition +=
            (decomposition.empty() ? "" : " ") + line.substr(key.size());
  return decomposition;
}

/** Per analysis folder, given by the mapping file of rank 0 in it ("" for an
 *  analysis that is off), whether its records come from another
 *  decomposition (a remapped restart), whose records of each rank cannot
 *  continue: rank 0 reads the mappings, warns, and tells the ranks. */
std::vector<int> decompositionChanged(const std::vector<std::string>& mappings,
                                      VirtualTopology3D* vct, int firstCycle) {
  const std::string current = std::to_string(vct->getXLEN()) + " " +
                              std::to_string(vct->getYLEN()) + " " +
                              std::to_string(vct->getZLEN());
  std::vector<int> differs(mappings.size(), 0);
  if (MPIdata::get_rank() == 0) {
    std::string others; // the mappings that differ: their decomposition
    for (size_t f = 0; f < mappings.size(); f++) {
      const std::string written =
          mappings[f].empty() ? "" : mappingDecomposition(mappings[f]);
      if (written.empty() || written == current)
        continue;
      differs[f] = 1;
      others += (others.empty() ? "" : ", ") + mappings[f] + ": " + written;
    }
    if (!others.empty())
      std::fprintf(stderr,
                   "[!] Warning: this restart runs on XLEN YLEN ZLEN = %s, "
                   "the analysis records on another decomposition (%s): in "
                   "those folders the records files and the rank mappings of "
                   "the analyses that run are kept as *_before_cycle%d.* and "
                   "start again, those of the others are left as they are.\n",
                   current.c_str(), others.c_str(), firstCycle);
  }
  MPI_Bcast(differs.data(), (int)differs.size(), MPI_INT, 0,
            MPIdata::get_PicGlobalComm());
  return differs;
}

} // namespace

class dataAnalysisPipelineImpl {
public:
  explicit dataAnalysisPipelineImpl(c_Solver& KCode);

  void prepareAnalysis(int cycle);
  void startAnalysis(int cycle);
  void waitForAnalysis();

private:
  c_Solver& solver;
  const DataAnalysisInput input;
  // the GPU work of the analyses, below the field solve it overlaps; null
  // when no analysis is on
  OwnedStream stream;

  // velocity histograms, of the rank or of its macrocells, and their GMM
  // fits; null when off
  std::unique_ptr<velocitySoA> velocitySoACUDA;
  std::unique_ptr<velocityHistogram::velocityHistogram3D> rankHistogram;
  std::unique_ptr<velocityHistogram::macrocellHistogram3D> macrocellHistogram;
  velocityHistogram::histogramRegions* histograms = nullptr; // the one on
  std::unique_ptr<GMMModel> gmm;
  // the last fit per species and region (the rank, or a macrocell), on the
  // host: 13 GMMComponents doubles each
  std::vector<std::vector<std::unique_ptr<GMMFit>>> lastFit;

  // macrocell spectra; null when off
  std::unique_ptr<macrocellSpectra::macrocellSpectra2D> macrocellHist;

  // the records files of the rank, one per analysis
  RecordFile histogramRecords{velocityHistogram::kHistogramRecordMagic};
  RecordFile gmmRecords{kGMMRecordMagic};
  RecordFile spectraRecords{macrocellSpectra::kSpectraRecordMagic};

  std::future<void> analysisFuture;
  int analysisCycle = 0; // of analysisFuture
  // last member: destroyed first, so its thread stops before the buffers go;
  // null when no analysis is on
  std::unique_ptr<ThreadPool> threadPool;

  void openRecords(RecordFile& file, const char* folder,
                   const std::string& name, const std::string& json,
                   const std::vector<std::string>& layout, bool on,
                   bool remapped);
  macrocell::MacrocellPartition macrocellsOf(const int size[3]) const;
  void analysisEntry(int cycle);
  void initialComponents(int species, int region,
                         const velocityHistogram::BinGrid& bins,
                         std::mt19937& gen, GMMType* weight, GMMType* mean,
                         GMMType* coVariance) const;
  void fitGMM(int cycle, int species);
  void initAnalyses();
  std::string gmmDescription(const std::string& rank,
                             std::vector<std::string>& layout) const;
  void writeGMMRecord(int cycle);
};

dataAnalysisPipelineImpl::dataAnalysisPipelineImpl(c_Solver& KCode)
    : solver(KCode), input(KCode.col->getDataAnalysis()) {
  const bool isRestart = solver.restart_status != 0;
  const int firstCycle = solver.first_cycle;
  // per folder (histograms, GMM, spectra): whether its analysis runs and, on
  // a restart, whether its records come from another decomposition, which
  // then start again, the old ones kept aside
  const char* folders[3] = {DAConfig::HISTOGRAM_SUBDIR, DAConfig::GMM_SUBDIR,
                            DAConfig::MACROCELL_SPECTRA_SUBDIR};
  const bool any = input.any();
  const bool on[3] = {any && input.velocityHistogram, any && input.gmm,
                      any && input.velocitySpectra};
  std::vector<int> remapped(3, 0);
  if (DAConfig::DATA_ANALYSIS_OUTPUT && isRestart) {
    std::vector<std::string> mappings(3); // of an analysis off too
    for (int f = 0; f < 3; f++)
      mappings[f] = input.outputDir + folders[f] + "vctMapping_subDomain0.txt";
    remapped = decompositionChanged(mappings, solver.vct, firstCycle);
  }
  if (any)
    initAnalyses();

  // the records files (of an analysis off: those of an earlier run kept
  // aside or cut, openRecords)
  if constexpr (DAConfig::DATA_ANALYSIS_OUTPUT) {
    const std::string rank = rankDescription(solver.myrank, solver.vct);
    RecordFile* files[3] = {&histogramRecords, &gmmRecords, &spectraRecords};
    const char* names[3] = {"histograms", "gmm", "spectra"};
    for (int f = 0; f < 3; f++) {
      std::vector<std::string> layout;
      std::string json;
      if (f == 0 && histograms)
        json = histograms->describe(rank, layout);
      else if (f == 1 && gmm)
        json = gmmDescription(rank, layout);
      else if (f == 2 && macrocellHist)
        json = macrocellHist->describe(rank, layout);
      openRecords(*files[f], folders[f], names[f], json, layout, on[f],
                  remapped[f]);
    }
  }
}

/** The macrocells of this rank of the given size in cells (an analysis):
 *  the tiling of its interior cells (one ghost layer per side), with their
 *  global offset. */
macrocell::MacrocellPartition
dataAnalysisPipelineImpl::macrocellsOf(const int size[3]) const {
  macrocell::MacrocellPartition partition;
  const Grid3DCU& grid = *solver.grid;
  if (!partition.build(grid.getNXC() - 2, grid.getNYC() - 2, grid.getNZC() - 2,
                       size[0], size[1], size[2], grid.getCellOffsetX(),
                       grid.getCellOffsetY(), grid.getCellOffsetZ()))
    throw std::runtime_error(
        "data analysis: macrocells of " + std::to_string(size[0]) + " x " +
        std::to_string(size[1]) + " x " + std::to_string(size[2]) +
        " cells on a rank without cells");
  return partition;
}

/** The stream, the thread and the analyses of the rank: the histograms (of
 *  the rank, or of its macrocells) and their GMM, the spectra, each of the
 *  macrocell analyses on its own macrocells (macrocellsOf). Rank 0 prints
 *  the GPU memory of the dense bins of its macrocells. */
void dataAnalysisPipelineImpl::initAnalyses() {
  stream.reset(createPriorityStream(StreamPriority::Diagnostic));
  threadPool = std::make_unique<ThreadPool>(1);
  // the dense bins of the macrocells of an analysis on rank 0
  auto printBins = [&](const char* analysis, int macrocells, size_t bytes) {
    if (solver.myrank == 0)
      std::cout << "[macrocell] " << analysis << ": " << macrocells
                << " macrocells on rank 0, "
                << (double)macrocells * bytes / (1 << 20)
                << " MiB of GPU memory for their bins\n";
  };

  if (input.velocityHistogram) {
    velocitySoACUDA = std::make_unique<velocitySoA>();
    if (!input.histogramMacrocells) {
      rankHistogram = std::make_unique<velocityHistogram::velocityHistogram3D>(
          input.histogramFixedRange);
      histograms = rankHistogram.get();
    } else {
      const macrocell::MacrocellPartition partition =
          macrocellsOf(input.histogramMacrocellSize);
      macrocellHistogram =
          std::make_unique<velocityHistogram::macrocellHistogram3D>(
              partition, input.histogramFixedRange);
      histograms = macrocellHistogram.get();
      printBins("histograms", partition.M,
                velocityHistogram::kMacrocellBins *
                    sizeof(velocityHistogram::histogramTypeOut));
    }
    if (input.gmm) {
      gmm = std::make_unique<GMMModel>(stream.get());
      lastFit.resize(solver.ns);
      for (auto& fits : lastFit)
        fits.resize(histograms->numRegions());
    }
  }
  if (input.velocitySpectra) {
    const macrocell::MacrocellPartition partition =
        macrocellsOf(input.spectraMacrocellSize);
    macrocellHist = std::make_unique<macrocellSpectra::macrocellSpectra2D>(
        partition, input.histogramMax);
    printBins("spectra", partition.M,
              macrocellSpectra::kSpectraBins *
                  sizeof(macrocellSpectra::macrocellHistType));
  }
}

/**
 * @brief The records file of an analysis of this rank,
 * <folder>subDomain<rank>_<name>.bin, described by its .json, next to the
 * mapping of the rank: rank 0 prepares the folder, and on a restart onto
 * another decomposition (remapped) keeps aside the records and mappings of
 * every rank of the earlier run first; the file is opened
 * (RecordFile::open), described by json. Of an analysis that is off
 * (on false), the records and mappings of an earlier run, of all its ranks,
 * are kept aside by rank 0 on a fresh start (a restart of this run would cut
 * or continue them), and cut from the
 * first cycle on by a restart on the same decomposition (another format
 * kept aside). Rank 0 warns of the files kept aside. Collective.
 */
void dataAnalysisPipelineImpl::openRecords(
    RecordFile& file, const char* folder, const std::string& name,
    const std::string& json, const std::vector<std::string>& layout, bool on,
    bool remapped) {
  const bool isRestart = solver.restart_status != 0;
  const int firstCycle = solver.first_cycle;
  const std::string path = input.outputDir + folder;
  const std::string rank = std::to_string(solver.myrank);
  const std::string stem = path + "subDomain" + rank + "_" + name;
  if (!on) {
    if (!isRestart) // rank 0, for every rank of the earlier run
      warnKeptAside(
          solver.myrank == 0 ? keepAsideRecords(path, name, firstCycle) : 0,
          path, firstCycle,
          "are of an earlier run, whose analysis this fresh start "
          "does not run (with their rank mappings)");
    else if (!remapped)
      warnKeptAside(file.cut(stem, firstCycle), path, firstCycle,
                    "have another record format than this build");
    return;
  }
  if (remapped && solver.myrank == 0) // before any rank writes
    keepAsideRecords(path, name, firstCycle);
  prepareSharedFolder(path, solver.myrank, isRestart);
  const std::string mapping = path + "vctMapping_subDomain" + rank + ".txt";
  writeVctMapping(mapping, solver.vct);
  const bool keptAside =
      file.open(stem, json, layout, isRestart && !remapped, firstCycle);
  if (isRestart)
    warnKeptAside(keptAside, path, firstCycle,
                  "have another layout (regions, macrocells or bins) or "
                  "record format than this run, and start again");
}

/**
 * @brief The initial components of the fit of a region of the given bins:
 * from the previous fit of the region with GMMWarmStart, otherwise from
 * equal weights, means drawn uniformly in the ball inscribed in the box of
 * the bins and the diagonal covariance of the thermal speeds of the species.
 * A component that was pruned, is close to it (weight below the smaller of
 * kWarmStartWeightFactor GMMPruneThreshold and kWarmStartEqualShare /
 * GMMComponents, GMMPruneThreshold at least) or has a non-finite mean or
 * covariance is drawn again, and the weights then restart equal.
 */
void dataAnalysisPipelineImpl::initialComponents(
    int species, int region, const velocityHistogram::BinGrid& bins,
    std::mt19937& gen, GMMType* weight, GMMType* mean,
    GMMType* coVariance) const {
  const GMMType vth[kDim] = {(GMMType)solver.col->getUth(species),
                             (GMMType)solver.col->getVth(species),
                             (GMMType)solver.col->getWth(species)};
  // the ball inscribed in the box of the bins
  GMMType center[kDim], radius = INFINITY;
  for (int d = 0; d < kDim; d++) {
    center[d] = ((GMMType)bins.min[d] + (GMMType)bins.max[d]) / 2;
    radius =
        std::min(radius, ((GMMType)bins.max[d] - (GMMType)bins.min[d]) / 2);
  }
  std::uniform_real_distribution<GMMType> unif01(0.0, 1.0);
  std::normal_distribution<GMMType> normal(0.0, 1.0);
  auto sampleMean = [&](GMMType* m) {
    const GMMType n[kDim] = {normal(gen), normal(gen), normal(gen)};
    const GMMType r = radius * std::cbrt(unif01(gen)) /
                      std::sqrt(n[0] * n[0] + n[1] * n[1] + n[2] * n[2]);
    for (int d = 0; d < kDim; d++)
      m[d] = center[d] + r * n[d];
  };

  const GMMFit* last =
      input.gmmWarmStart ? lastFit[species][region].get() : nullptr;
  const int components = input.gmmComponents;
  const double keepWeight =
      std::max(input.gmmPruneThreshold,
               std::min(kWarmStartWeightFactor * input.gmmPruneThreshold,
                        kWarmStartEqualShare / components));
  bool equalWeights = last == nullptr;
  for (int j = 0; j < components; j++) {
    GMMType* m = &mean[j * kDim];
    GMMType* c = &coVariance[j * kDim * kDim];
    bool keep = last && last->weight[j] >= keepWeight;
    for (int d = 0; d < kDim; d++)
      keep = keep && std::isfinite(last->mean[j * kDim + d]);
    for (int e = 0; e < kDim * kDim; e++)
      keep = keep && std::isfinite(last->coVariance[j * kDim * kDim + e]);
    if (keep) {
      for (int d = 0; d < kDim; d++)
        m[d] = last->mean[j * kDim + d];
      for (int e = 0; e < kDim * kDim; e++)
        c[e] = last->coVariance[j * kDim * kDim + e];
    } else {
      sampleMean(m);
      for (int e = 0; e < kDim * kDim; e++)
        c[e] = 0;
      for (int d = 0; d < kDim; d++)
        c[d * (kDim + 1)] = vth[d] * vth[d];
      equalWeights = true;
    }
  }
  for (int j = 0; j < components; j++)
    weight[j] = equalWeights ? GMMType(1) / components : last->weight[j];
}

/**
 * @brief Fit a Gaussian mixture to the non-empty bins of the velocity
 * histogram of each region (the rank, or each macrocell) of a species, which
 * init() has just computed on the stream; a region without such bins keeps
 * its previous fit and has none at this cycle.
 *
 * The fit of a region runs on the velocities divided by the largest one of
 * its bins (with GMMNormalize), from initialComponents; the draws are
 * reproducible, different for every cycle, species and rank, region after
 * region. The regions with bins are fitted together, in batches of at most
 * kGMMBatchDoubles doubles of the GMM per point (kGMMFitPoints more points
 * per fit; a larger region alone).
 */
void dataAnalysisPipelineImpl::fitGMM(int cycle, int species) {
  // the points are the centres of the non-empty bins, weighted by their
  // content, region after region
  const int numPoints = histograms->compactNonEmpty(stream.get());
  if (numPoints == 0) // no particle in the range: nothing to fit
    return;
  GMMType* const* allCenters = histograms->getNonEmptyCenters(stream.get());
  const std::vector<int>& offset = histograms->getRegionOffsets();
  auto pointsOf = [&](int r) { return offset[r + 1] - offset[r]; };

  // the fits, of the regions with points: their initial components and the
  // largest velocity of their bins (the scale of the fit)
  std::vector<int> fitted;
  for (int r = 0; r < histograms->numRegions(); r++)
    if (pointsOf(r) > 0)
      fitted.push_back(r);
  const int fits = (int)fitted.size(), components = input.gmmComponents;
  std::seed_seq seed{cycle, species, solver.myrank};
  std::mt19937 gen(seed);
  std::vector<GMMType> weight(fits * components),
      mean(fits * components * kDim),
      coVariance(fits * components * kDim * kDim), scale(fits, 0);
  for (int f = 0; f < fits; f++) {
    const velocityHistogram::BinGrid& bins = histograms->getGrid(fitted[f]);
    for (int d = 0; d < kDim; d++)
      scale[f] = std::max({scale[f], (GMMType)std::fabs(bins.min[d]),
                           (GMMType)std::fabs(bins.max[d])});
    initialComponents(species, fitted[f], bins, gen, &weight[f * components],
                      &mean[f * components * kDim],
                      &coVariance[f * components * kDim * kDim]);
  }

  cudaGMMWeight::GMMParam_t<GMMType> param = {
      .numComponents = components,
      .maxIteration = input.gmmMaxIterations,
      .threshold = (GMMType)input.gmmConvergenceThreshold,
      .normalize = input.gmmNormalize,
      .checkCovariance = input.gmmCheckCovariance,
      .covarianceTolerance = (GMMType)input.gmmCovarianceTolerance,
      .minVariance = (GMMType)input.gmmMinVariance,
      .prune = input.gmmPruneComponents,
      .pruneThreshold = (GMMType)input.gmmPruneThreshold};
  // batches of the fits [first, last): at most batchPoints points with
  // kGMMFitPoints per fit, or one fit
  const long batchPoints = kGMMBatchDoubles / (components + 1);
  for (int first = 0, last = 0; first < fits; first = last) {
    long cost = 0;
    do
      cost += pointsOf(fitted[last++]) + kGMMFitPoints;
    while (last < fits &&
           cost + pointsOf(fitted[last]) + kGMMFitPoints <= batchPoints);
    // the points of the batch are those of its fits, consecutive
    const int base = offset[fitted[first]];
    std::vector<int> fitOffset;
    for (int f = first; f < last; f++)
      fitOffset.push_back(offset[fitted[f]] - base);
    fitOffset.push_back(offset[fitted[last - 1] + 1] - base);
    param.weightInit = &weight[first * components];
    param.meanInit = &mean[first * components * kDim];
    param.coVarianceInit = &coVariance[first * components * kDim * kDim];
    param.scale = &scale[first];
    GMMType* centers[kDim];
    for (int d = 0; d < kDim; d++)
      centers[d] = allCenters[d] + base;
    cudaGMMWeight::GMMDataMultiDim<GMMType, kDim, weightType> data(
        fitOffset.back(), centers, histograms->getNonEmptyWeights() + base);

    gmm->config(&param, &data, last - first, fitOffset.data());
    gmm->initGMM();
    for (int f = first; f < last; f++)
      lastFit[species][fitted[f]] =
          std::make_unique<GMMFit>(gmm->getGMMResult(cycle, f - first));
  }
}

/**
 * @brief The description of the GMM records file of the rank
 * (describeRecords) after rank, the JSON members of the rank, and into
 * layout its part that the records of the file need: the regions of the
 * histograms (the rank, or the macrocell tiling).
 */
std::string dataAnalysisPipelineImpl::gmmDescription(
    const std::string& rank, std::vector<std::string>& layout) const {
  const std::string& regions = histograms->regionsDescription();
  layout = {regions};
  return describeRecords(
      rank + regions, kGMMRecordMagic,
      {{"magic", dtypeName<decltype(GMMRecordHeader::magic)>(), ""},
       {"cycle", dtypeName<decltype(GMMRecordHeader::cycle)>(), ""},
       {"species", dtypeName<decltype(GMMRecordHeader::species)>(), ""},
       {"regions", dtypeName<decltype(GMMRecordHeader::regions)>(), ""},
       {"components", dtypeName<decltype(GMMRecordHeader::components)>(), ""},
       {"reserved", dtypeName<decltype(GMMRecordHeader::reserved)>(), ""},
       {"record_bytes", dtypeName<decltype(GMMRecordHeader::bytes)>(), ""}},
      {{"log_likelihood", dtypeName<double>(), "\"species\", \"regions\""},
       {"weights", dtypeName<double>(),
        "\"species\", \"regions\", \"components\""},
       {"means", dtypeName<double>(),
        "\"species\", \"regions\", \"components\", 3"},
       {"covariances", dtypeName<double>(),
        "\"species\", \"regions\", \"components\", 3, 3"},
       {"steps", dtypeName<std::int32_t>(), "\"species\", \"regions\""}},
      "one record per analysis cycle with the fits of every species (the "
      "header's species: their number) and "
      "region (the rank, or macrocell m = (mz*My + my)*Mx + mx): the means "
      "(u, v, w) and the covariances (row major) in the units of the "
      "velocities; steps -1 for a region without a fit at the cycle (no "
      "particle in its range), its values 0; a pruned component has weight "
      "1e-15; the log-likelihood is that of the parameters before the last "
      "EM step");
}

/** Append the fits of a cycle, after the fitGMM of every species, to the
 *  GMM records file: GMMRecordHeader, then per fit (species after species,
 *  region after region) its log-likelihood, the weights, means and
 *  covariances of its components (float64), and its EM steps (int32, -1 for
 *  a region without a fit at this cycle). */
void dataAnalysisPipelineImpl::writeGMMRecord(int cycle) {
  const int regions = histograms->numRegions(), K = input.gmmComponents;
  const size_t fits = (size_t)solver.ns * regions;
  // the fit of this cycle, or none
  auto fitOf = [&](size_t f) -> const GMMFit* {
    const GMMFit* fit = lastFit[f / regions][f % regions].get();
    return fit && fit->simulationStep == cycle ? fit : nullptr;
  };
  gmmRecords.begin(GMMRecordHeader{0, cycle, (std::uint32_t)solver.ns,
                                   (std::uint32_t)regions, (std::uint32_t)K, 0,
                                   0});
  // per fit count values of value(fit, i), 0 without a fit
  std::vector<double> values;
  auto put = [&](int count, auto value) {
    values.assign(fits * count, 0.0);
    for (size_t f = 0; f < fits; f++)
      if (const GMMFit* fit = fitOf(f))
        for (int i = 0; i < count; i++)
          values[f * count + i] = (double)value(*fit, i);
    gmmRecords.put(values.data(), values.size());
  };
  put(1, [](const GMMFit& fit, int) { return fit.logLikelihoodFinal; });
  put(K, [](const GMMFit& fit, int i) { return fit.weight[i]; });
  put(K * kDim, [](const GMMFit& fit, int i) { return fit.mean[i]; });
  put(K * kDim * kDim,
      [](const GMMFit& fit, int i) { return fit.coVariance[i]; });
  std::vector<std::int32_t> steps(fits);
  for (size_t f = 0; f < fits; f++) {
    const GMMFit* fit = fitOf(f);
    steps[f] = fit ? fit->convergeStep : -1;
  }
  gmmRecords.put(steps.data(), steps.size());
  gmmRecords.end();
}

/**
 * @brief The analyses of one cycle, species by species to save GPU memory.
 * @details Runs on the pipeline's thread and on its stream, while the main
 * thread solves the field on the solver stream (of higher priority). It reads
 * the particles (the stream waits for their last sort, see startAnalysis),
 * and for the macrocell analyses the cell offsets of prepareAnalysis (and
 * for the spectra its field), synchronised before; the main thread changes none
 * of them until waitForAnalysis() returns. Every species ends with the stream
 * drained: the next one rewrites the host buffers of the copies, and the
 * mover changes the particles after waitForAnalysis().
 * @param cycle Simulation cycle being analysed.
 */
void dataAnalysisPipelineImpl::analysisEntry(int cycle) {
  cudaErrChk(cudaSetDevice(solver.cudaDeviceOnNode));

  for (int i = 0; i < solver.ns; i++) {
    if (histograms) {
      // borrow the SoA pointers of the species (no copy)
      velocitySoACUDA->borrowFrom(solver.pclsArrayHostPtr[i]);
      if (macrocellHistogram) {
        const CellSorter& sorter = solver.cellSorters[i];
        macrocellHistogram->init(velocitySoACUDA.get(),
                                 sorter.getCellStartOffsets(),
                                 sorter.getCellCounts(), input.histogramMin[i],
                                 input.histogramMax[i], stream.get());
      } else {
        rankHistogram->init(velocitySoACUDA.get(), input.histogramMin[i],
                            input.histogramMax[i], stream.get());
      }
      if constexpr (DAConfig::DATA_ANALYSIS_OUTPUT)
        histograms->writeRecord(histogramRecords, i, cycle, stream.get());
      if (gmm)
        fitGMM(cycle, i);
    }

    if (macrocellHist && input.spectraSpecies[i]) {
      const CellSorter& sorter = solver.cellSorters[i];
      macrocellHist->launch(solver.pclsArrayHostPtr[i],
                            solver.fieldForPclCUDAPtr, solver.grid3DCUDACUDAPtr,
                            sorter.getCellStartOffsets(),
                            sorter.getCellCounts(), i, stream.get());
      if constexpr (DAConfig::DATA_ANALYSIS_OUTPUT)
        macrocellHist->writeRecord(spectraRecords, i, cycle, stream.get());
    }
    cudaErrChk(cudaStreamSynchronize(stream.get()));
  }
  // the records of the cycle, one write per file
  if constexpr (DAConfig::DATA_ANALYSIS_OUTPUT) {
    if (gmm)
      writeGMMRecord(cycle);
    for (RecordFile* file : {&histogramRecords, &gmmRecords, &spectraRecords})
      file->flush();
  }
}

/** The macrocell analyses read the particles sorted by cell: sort the
 *  species of the spectra, every species with the macrocell histograms. The
 *  spectra read B from the mover's field buffer: pack the current field (the
 *  buffer holds the B of before the last CalculateB). */
void dataAnalysisPipelineImpl::prepareAnalysis(int cycle) {
  if (!input.runsAt(cycle) || !(macrocellHist || macrocellHistogram))
    return;
  std::vector<int> sort(solver.ns, macrocellHistogram ? 1 : 0);
  if (macrocellHist)
    for (int s = 0; s < solver.ns; s++)
      sort[s] = sort[s] || input.spectraSpecies[s];
  solver.sortSpecies(sort);
  if (macrocellHist)
    solver.refreshFieldForPclsDeviceBuffer(true);
}

void dataAnalysisPipelineImpl::startAnalysis(int cycle) {
  if (analysisFuture.valid()) // its error would be lost, its reads go on
    throw std::logic_error("startAnalysis: the analysis of cycle " +
                           std::to_string(analysisCycle) +
                           " was not waited for");
  if (!input.runsAt(cycle))
    return;

  if (MPIdata::get_rank() == 0)
    printf("  [Cycle %d] Data analysis: ON [%s%s%s]\n", cycle,
           input.velocityHistogram ? "histogram " : "", input.gmm ? "GMM " : "",
           input.velocitySpectra ? "macrocellSpectra" : "");

  // the particles are final after the last work on their streams: the sort
  // of a checkpoint (outputCopyAsync) runs there with no host wait
  for (int i = 0; i < solver.ns; i++)
    cudaErrChk(cudaStreamWaitEvent(stream.get(), solver.cycleEndEvent[i], 0));
  analysisCycle = cycle;
  analysisFuture = threadPool->enqueue(&dataAnalysisPipelineImpl::analysisEntry,
                                       this, cycle);
}

/** Wait for the running analysis, if any; an error of the analysis is
 *  thrown again here, on the main thread. */
void dataAnalysisPipelineImpl::waitForAnalysis() {
  if (analysisFuture.valid())
    awaitTask(analysisFuture,
              "data analysis of cycle " + std::to_string(analysisCycle));
}

dataAnalysisPipeline::dataAnalysisPipeline(iPic3D::c_Solver& KCode)
    : impl(std::make_unique<dataAnalysisPipelineImpl>(KCode)) {}

void dataAnalysisPipeline::prepareAnalysis(int cycle) {
  impl->prepareAnalysis(cycle);
}

void dataAnalysisPipeline::startAnalysis(int cycle) {
  impl->startAnalysis(cycle);
}

void dataAnalysisPipeline::waitForAnalysis() { impl->waitForAnalysis(); }

dataAnalysisPipeline::~dataAnalysisPipeline() = default;

} // namespace dataAnalysis
