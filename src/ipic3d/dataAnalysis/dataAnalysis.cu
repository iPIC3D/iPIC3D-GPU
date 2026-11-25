#include <algorithm>
#include <cmath>
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
#include "Profiler.h"
#include "VCtopology3D.h"
#include "iPic3D.h"
#include "outputPrepare.h"
#include "threadPool.hpp"

#include "GMM/cudaGMM.cuh"
#include "cellSortBuffers.cuh"
#include "dataAnalysis.cuh"
#include "dataAnalysisConfig.cuh"
#include "gmmRecords.h"
#include "macrocellPartition.cuh"
#include "macrocellSpectra.cuh"
#include "velocityHistogram.cuh"

#include "mpi.h"

namespace dataAnalysis {

/** Dimension of the Gaussian mixture; its components are GMMComponents. */
inline constexpr int kDim = DAConfig::DATA_DIM_GMM;
static_assert(kDim == 3 && kDim == kGMMRecordDim,
              "the GMM fits (u, v, w) histograms");
static_assert(DAConfig::MAX_HISTOGRAM_MACROCELLS <= kGMMMaxRegions,
              "the regions of a GMM record");
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

using namespace iPic3D;
using velocityHistogram::velocitySoA;
using GMMType = cudaParticleType;
using weightType = velocityHistogram::histogramTypeOut;
using GMMModel = cudaGMMWeight::GMM<GMMType, kDim, weightType>;
using GMMFit = cudaGMMWeight::GMMResult<GMMType, kDim>;
static_assert(std::is_same_v<GMMType, double>, "a GMMFitView is of doubles");

namespace {

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
      << "  \"periodic\": [" << vct->getPERIODICX() << ","
      << vct->getPERIODICY() << "," << vct->getPERIODICZ() << "],\n"
      << "  \"ipic3d_version\": \"" << iPic3D::build::version() << "\",\n"
      << "  \"ipic3d_build\": \"" << iPic3D::build::signature() << "\",\n";
  return out.str();
}

/** The decomposition "XLEN YLEN ZLEN" of the records file described by the
 *  JSON at path, its "topology" (rankDescription); "" without one. */
std::string writtenDecomposition(const std::string& path) {
  const std::string key = "\"topology\": [";
  std::ifstream in(path);
  for (std::string line; std::getline(in, line);) {
    const size_t at = line.find(key);
    if (at == std::string::npos)
      continue;
    const size_t first = at + key.size();
    std::string decomposition =
        line.substr(first, line.find(']', first) - first);
    std::replace(decomposition.begin(), decomposition.end(), ',', ' ');
    return decomposition;
  }
  return "";
}

/** Per analysis folder, given by the description of the records of rank 0 in
 *  it, whether its records come from another decomposition (a remapped
 *  restart), whose records of each rank cannot continue: rank 0 reads the
 *  descriptions, warns, and tells the ranks. */
std::vector<int>
decompositionChanged(const std::vector<std::string>& descriptions,
                     VirtualTopology3D* vct, int firstCycle) {
  const std::string current = std::to_string(vct->getXLEN()) + " " +
                              std::to_string(vct->getYLEN()) + " " +
                              std::to_string(vct->getZLEN());
  std::vector<int> differs(descriptions.size(), 0);
  if (MPIdata::get_rank() == 0) {
    std::string others; // the descriptions that differ: their decomposition
    for (size_t f = 0; f < descriptions.size(); f++) {
      const std::string written = writtenDecomposition(descriptions[f]);
      if (written.empty() || written == current)
        continue;
      differs[f] = 1;
      others += (others.empty() ? "" : ", ") + descriptions[f] + ": " + written;
    }
    if (!others.empty())
      std::fprintf(stderr,
                   "[!] Warning: this restart runs on XLEN YLEN ZLEN = %s, "
                   "the analysis records on another decomposition (%s): in "
                   "those folders the records files of the analyses that run "
                   "are kept as *_before_cycle%d.* and start again, those of "
                   "the others are left as they are.\n",
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
  // host: 13 doubles and a flag per component
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
  void writeFits(int cycle, int species);
  void initAnalyses();
};

dataAnalysisPipelineImpl::dataAnalysisPipelineImpl(c_Solver& KCode)
    : solver(KCode), input(KCode.col->getDataAnalysis()) {
  profiling::Range range(profiling::Region::AnalysisInit);
  const bool isRestart = solver.restart_status != 0;
  const int firstCycle = solver.first_cycle;
  // per folder (histograms, GMM, spectra): whether its analysis runs and, on
  // a restart, whether its records come from another decomposition, which
  // then start again, the old ones kept aside
  const char* folders[3] = {DAConfig::HISTOGRAM_SUBDIR, DAConfig::GMM_SUBDIR,
                            DAConfig::MACROCELL_SPECTRA_SUBDIR};
  const char* names[3] = {"histograms", "gmm", "spectra"};
  const bool any = input.any();
  const bool on[3] = {any && input.velocityHistogram, any && input.gmm,
                      any && input.velocitySpectra};
  std::vector<int> remapped(3, 0);
  if (DAConfig::DATA_ANALYSIS_OUTPUT && isRestart) {
    std::vector<std::string> descriptions(3); // of an analysis off too
    for (int f = 0; f < 3; f++)
      descriptions[f] =
          recordsStem(input.outputDir + folders[f], names[f], 0) + ".json";
    remapped = decompositionChanged(descriptions, solver.vct, firstCycle);
  }
  if (any)
    initAnalyses();

  // the records files (of an analysis off: those of an earlier run kept
  // aside or cut, openRecords)
  if constexpr (DAConfig::DATA_ANALYSIS_OUTPUT) {
    const std::string rank = rankDescription(solver.myrank, solver.vct);
    RecordFile* files[3] = {&histogramRecords, &gmmRecords, &spectraRecords};
    for (int f = 0; f < 3; f++) {
      std::vector<std::string> layout;
      std::string json;
      if (f == 0 && histograms)
        json = histograms->describe(rank, layout);
      else if (f == 1 && gmm) { // the fits of the regions of the histograms
        layout = {histograms->regionsDescription()};
        json = describeGMMRecords(rank + layout[0]);
      } else if (f == 2 && macrocellHist)
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
  threadPool = std::make_unique<ThreadPool>(1, "iPIC3D analysis");
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
 * <folder>subDomain<rank>_<name>.bin, described by its .json: rank 0
 * prepares the folder, and on a restart onto another decomposition
 * (remapped) keeps aside the records of every rank of the earlier run first;
 * the file is opened (RecordFile::open), described by json. Of an analysis
 * that is off (on false), the records of an earlier run, of all its ranks,
 * are kept aside by rank 0 on a fresh start (a restart of this run would cut
 * or continue them), and cut from the first cycle on by a restart on the
 * same decomposition (another format kept aside). Rank 0 warns of the files
 * kept aside. Collective.
 */
void dataAnalysisPipelineImpl::openRecords(
    RecordFile& file, const char* folder, const std::string& name,
    const std::string& json, const std::vector<std::string>& layout, bool on,
    bool remapped) {
  const bool isRestart = solver.restart_status != 0;
  const int firstCycle = solver.first_cycle;
  const std::string path = input.outputDir + folder;
  const std::string stem = recordsStem(path, name, solver.myrank);
  if (!on) {
    if (!isRestart) // rank 0, for every rank of the earlier run
      warnKeptAside(
          solver.myrank == 0 ? keepAsideRecords(path, name, firstCycle) : 0,
          path, firstCycle,
          "are of an earlier run, whose analysis this fresh start "
          "does not run");
    else if (!remapped)
      warnKeptAside(file.cut(stem, firstCycle), path, firstCycle,
                    "have another record format than this build");
    return;
  }
  if (remapped && solver.myrank == 0) // before any rank writes
    keepAsideRecords(path, name, firstCycle);
  prepareSharedFolder(path, solver.myrank, isRestart);
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
    // the description of the points, page-locked: config copies it to the
    // device
    using Points = cudaGMMWeight::GMMDataMultiDim<GMMType, kDim, weightType>;
    const PinnedVector<Points> data(
        1, Points(fitOffset.back(), centers,
                  histograms->getNonEmptyWeights() + base));

    gmm->config(&param, data.data(), last - first, fitOffset.data());
    gmm->initGMM();
    for (int f = first; f < last; f++)
      lastFit[species][fitted[f]] =
          std::make_unique<GMMFit>(gmm->getGMMResult(cycle, f - first));
  }
}

/** Append the fits of a species at a cycle, of its fitGMM just done, to the
 *  GMM records file: a record of the regions with a fit at this cycle, none
 *  without one. */
void dataAnalysisPipelineImpl::writeFits(int cycle, int species) {
  std::vector<GMMFitView> fits;
  for (int r = 0; r < histograms->numRegions(); r++) {
    const GMMFit* fit = lastFit[species][r].get();
    if (fit && fit->simulationStep == cycle)
      fits.push_back({r, fit->convergeStep, fit->logLikelihoodFinal,
                      fit->active.get(), fit->weight.get(), fit->mean.get(),
                      fit->coVariance.get()});
  }
  if (!fits.empty())
    writeGMMRecord(gmmRecords, cycle, species, histograms->numRegions(),
                   input.gmmComponents, fits);
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
  profiling::Range range(profiling::Region::AnalysisRun, cycle);

  for (int i = 0; i < solver.ns; i++) {
    if (histograms && input.histogramSpecies[i]) {
      {
        profiling::Range histogram(profiling::Region::VelocityHistogram, i);
        // borrow the SoA pointers of the species (no copy)
        velocitySoACUDA->borrowFrom(solver.pclsArrayHostPtr[i]);
        if (macrocellHistogram) {
          const CellSorter& sorter = solver.cellSorters[i];
          macrocellHistogram->init(
              velocitySoACUDA.get(), sorter.getCellStartOffsets(),
              sorter.getCellCounts(), input.histogramMin[i],
              input.histogramMax[i], stream.get());
        } else {
          rankHistogram->init(velocitySoACUDA.get(), input.histogramMin[i],
                              input.histogramMax[i], stream.get());
        }
        if constexpr (DAConfig::DATA_ANALYSIS_OUTPUT)
          histograms->writeRecord(histogramRecords, i, cycle, stream.get());
      }
      if (gmm) {
        profiling::Range fit(profiling::Region::GMMFit, i);
        fitGMM(cycle, i);
        if constexpr (DAConfig::DATA_ANALYSIS_OUTPUT)
          writeFits(cycle, i);
      }
    }

    if (macrocellHist && input.spectraSpecies[i]) {
      profiling::Range spectra(profiling::Region::MacrocellSpectra, i);
      const CellSorter& sorter = solver.cellSorters[i];
      macrocellHist->launch(solver.pclsArrayHostPtr[i],
                            solver.fieldForPclCUDAPtr, solver.grid3DCUDACUDAPtr,
                            sorter.getCellStartOffsets(),
                            sorter.getCellCounts(), i, stream.get());
      if constexpr (DAConfig::DATA_ANALYSIS_OUTPUT)
        macrocellHist->writeRecord(spectraRecords, i, cycle, stream.get());
    }
    {
      profiling::Range sync(profiling::Region::AnalysisSync, i);
      cudaErrChk(cudaStreamSynchronize(stream.get()));
    }
  }
  // the records of the cycle, one write per file
  if constexpr (DAConfig::DATA_ANALYSIS_OUTPUT) {
    profiling::Range write(profiling::Region::AnalysisWrite);
    for (RecordFile* file : {&histogramRecords, &gmmRecords, &spectraRecords})
      file->flush();
  }
}

/** The macrocell analyses read the particles sorted by cell: sort the
 *  species of the spectra and of the macrocell histograms. The spectra read B
 *  from the mover's field buffer: pack the current field (the buffer holds
 *  the B of before the last CalculateB). */
void dataAnalysisPipelineImpl::prepareAnalysis(int cycle) {
  if (!input.runsAt(cycle) || !(macrocellHist || macrocellHistogram))
    return;
  std::vector<int> sort(solver.ns);
  for (int s = 0; s < solver.ns; s++)
    sort[s] = (macrocellHistogram && input.histogramSpecies[s]) ||
              (macrocellHist && input.spectraSpecies[s]);
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
  // of a checkpoint or of a capped particle output (outputCopyAsync) runs
  // there with no host wait
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

dataAnalysisPipeline::~dataAnalysisPipeline() {
  profiling::Range range(profiling::Region::AnalysisFinalize);
  impl.reset();
}

} // namespace dataAnalysis
