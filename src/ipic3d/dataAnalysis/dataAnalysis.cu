#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <future>
#include <iostream>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

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
#include "macrocellSpectra.cuh"
#include "velocityHistogram.cuh"

#include "mpi.h"

namespace dataAnalysis {

/** Dimension and components of the Gaussian mixture. */
inline constexpr int kDim = DAConfig::DATA_DIM_GMM;
inline constexpr int kComponents = DAConfig::NUM_COMPONENT_GMM;
static_assert(kDim == 3, "the GMM fits (u, v, w) histograms");
static_assert(kComponents >= 1 && kComponents <= cudaGMMWeight::kGMMThreads,
              "the components a GMM fits");
/** A warm start keeps a component of the previous fit of weight at least
 *  this times GMMPruneThreshold, and draws the others again. */
inline constexpr double kWarmStartWeightFactor = 10;

using namespace iPic3D;
using velocityHistogram::velocitySoA;
using GMMType = cudaParticleType;
using weightType = velocityHistogram::histogramTypeOut;
using GMMModel = cudaGMMWeight::GMM<GMMType, kDim, weightType>;
using GMMFit = cudaGMMWeight::GMMResult<GMMType, kDim>;

namespace {

/** Destroys an owned stream. */
struct StreamDeleter {
  void operator()(cudaStream_t stream) const {
    cudaErrChk(cudaStreamDestroy(stream));
  }
};
using OwnedStream =
    std::unique_ptr<std::remove_pointer_t<cudaStream_t>, StreamDeleter>;

/** Create the folder: emptied on a fresh run, kept on a restart. */
void prepareFolder(const std::string& path, bool isRestart) {
  if (isRestart)
    ensureOutputFolder(path);
  else
    checkOutputFolder(path);
}

/** Rank 0 prepares the folder shared by the ranks before they write in it
 *  (an error of rank 0 ends every rank, see MPIdata); on a restart it removes
 *  the files of the cycles the restart writes again (the histograms). */
void prepareSharedFolder(const std::string& path, int rank, bool isRestart,
                         int firstCycle) {
  if (rank == 0) {
    prepareFolder(path, isRestart);
    if (isRestart)
      dropVtkFrom(path, firstCycle);
  }
  MPI_Barrier(MPIdata::get_PicGlobalComm());
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
                   "those folders the GMM and spectra records and the rank "
                   "mappings are kept as *_before_cycle%d.* and start again, "
                   "the histograms of the earlier cycles keep their names.\n",
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
  int checkAnalysis();
  int waitForAnalysis();

private:
  c_Solver& solver;
  const DataAnalysisInput input;
  // the GPU work of the analyses, below the field solve it overlaps; null
  // when no analysis is on
  OwnedStream stream;

  // velocity histograms and their GMM fit; null when off
  std::unique_ptr<velocitySoA> velocitySoACUDA;
  std::unique_ptr<velocityHistogram::velocityHistogram3D> velocityHistogram;
  std::string histogramPath; // prefix of this rank's histogram files
  std::unique_ptr<GMMModel> gmm;
  std::string gmmPath; // prefix of this rank's GMM files
  std::vector<std::unique_ptr<GMMFit>> lastFit; // per species

  // macrocell spectra; null when off on this rank
  std::unique_ptr<macrocellSpectra::macrocellSpectra2D> macrocellHist;

  std::future<int> analysisFuture;
  int analysisCycle = 0; // of analysisFuture
  // last member: destroyed first, so its thread stops before the buffers go;
  // null when no analysis is on
  std::unique_ptr<ThreadPool> threadPool;

  void initMacrocellSpectra(bool isRestart, bool remapped);
  int analysisEntry(int cycle);
  void fitGMM(int cycle, int species);
};

dataAnalysisPipelineImpl::dataAnalysisPipelineImpl(c_Solver& KCode)
    : solver(KCode), input(KCode.col->getDataAnalysis()) {
  if (!input.any())
    return;
  stream.reset(createPriorityStream(StreamPriority::Diagnostic));
  threadPool = std::make_unique<ThreadPool>(1);
  const bool isRestart = solver.restart_status != 0;
  const int firstCycle = solver.first_cycle;
  const std::string rankTag = std::to_string(solver.myrank);
  // per folder (histograms, GMM, spectra): whether this restart runs on
  // another decomposition than its records, which then start again, the old
  // ones kept aside
  std::vector<int> remapped(3, 0);
  if (DAConfig::DATA_ANALYSIS_OUTPUT && isRestart) {
    const std::string mapping = "vctMapping_subDomain0.txt";
    remapped = decompositionChanged(
        {input.velocityHistogram
             ? input.outputDir + DAConfig::HISTOGRAM_SUBDIR + mapping
             : "",
         input.gmm ? input.outputDir + DAConfig::GMM_SUBDIR + mapping : "",
         input.velocitySpectra
             ? input.outputDir + DAConfig::MACROCELL_SPECTRA_SUBDIR +
                   "subDomain_0/vctMapping.txt"
             : ""},
        solver.vct, firstCycle);
  }

  if (input.velocityHistogram) {
    velocitySoACUDA = std::make_unique<velocitySoA>();
    velocityHistogram =
        std::make_unique<velocityHistogram::velocityHistogram3D>(
            input.histogramFixedRange);
    histogramPath = input.outputDir + DAConfig::HISTOGRAM_SUBDIR;
    if constexpr (DAConfig::DATA_ANALYSIS_OUTPUT) {
      prepareSharedFolder(histogramPath, solver.myrank, isRestart, firstCycle);
      const std::string mapping =
          histogramPath + "vctMapping_subDomain" + rankTag + ".txt";
      if (remapped[0])
        keepAside({mapping}, firstCycle);
      writeVctMapping(mapping, solver.vct);
    }
    histogramPath += "subDomain" + rankTag + "_";

    if (input.gmm) {
      gmm = std::make_unique<GMMModel>(stream.get());
      lastFit.resize(solver.ns);
      gmmPath = input.outputDir + DAConfig::GMM_SUBDIR;
      if constexpr (DAConfig::DATA_ANALYSIS_OUTPUT) {
        prepareSharedFolder(gmmPath, solver.myrank, isRestart, firstCycle);
        // the mapping and the records of this rank, kept aside together
        std::vector<std::string> files = {gmmPath + "vctMapping_subDomain" +
                                          rankTag + ".txt"};
        for (int s = 0; s < solver.ns; s++)
          files.push_back(gmmPath + "subDomain" + rankTag + "_species" +
                          std::to_string(s) + ".json");
        if (remapped[1])
          keepAside(files, firstCycle);
        else if (isRestart)
          for (size_t f = 1; f < files.size(); f++)
            dropJsonRecordsFrom(files[f], firstCycle);
        writeVctMapping(files[0], solver.vct);
      }
      gmmPath += "subDomain" + rankTag + "_";
    }
  }

  if (input.velocitySpectra)
    initMacrocellSpectra(isRestart, remapped[2]);
}

/**
 * Tile the interior cells of this rank in macrocells and register the output
 * files of the species; the spectra stay off on a rank whose interior is
 * smaller than a macrocell. On a restart onto another decomposition
 * (remapped) the files of the rank are kept aside and start again, as those
 * of other macrocells, bins or velocity range, of which rank 0 warns.
 */
void dataAnalysisPipelineImpl::initMacrocellSpectra(bool isRestart,
                                                    bool remapped) {
  const int firstCycle = solver.first_cycle;
  const std::string root = input.outputDir + DAConfig::MACROCELL_SPECTRA_SUBDIR;
  if constexpr (DAConfig::DATA_ANALYSIS_OUTPUT)
    prepareSharedFolder(root, solver.myrank, isRestart, firstCycle);

  // interior cells (one ghost layer per side) and their global offset
  const Grid3DCU& grid = *solver.grid;
  const int interior[3] = {grid.getNXC() - 2, grid.getNYC() - 2,
                           grid.getNZC() - 2};
  const int offset[3] = {grid.getCellOffsetX(), grid.getCellOffsetY(),
                         grid.getCellOffsetZ()};
  const int* size = input.macrocell;

  macrocellSpectra::MacrocellPartition part;
  if (part.build(interior[0], interior[1], interior[2], size[0], size[1],
                 size[2], offset[0], offset[1], offset[2]))
    macrocellHist = std::make_unique<macrocellSpectra::macrocellSpectra2D>(
        part, std::vector<cudaCommonType>(input.histogramMax.begin(),
                                          input.histogramMax.end()));
  else
    std::cerr << "[macrocellSpectra] rank " << solver.myrank
              << ": macrocell sizes (" << size[0] << "," << size[1] << ","
              << size[2] << ") above its cells (" << interior[0] << ","
              << interior[1] << "," << interior[2]
              << "); no spectra on this rank.\n";

  if constexpr (DAConfig::DATA_ANALYSIS_OUTPUT) {
    int keptAside = 0; // species files of another layout
    if (macrocellHist) {
      const std::string subDir =
          root + "subDomain_" + std::to_string(solver.myrank) + "/";
      prepareFolder(subDir, isRestart);
      if (remapped) { // the mapping and the records, kept aside together
        std::vector<std::string> files = {subDir + "vctMapping.txt"};
        for (int s = 0; s < solver.ns; s++)
          for (const char* extension : {".json", ".bin"})
            files.push_back(subDir + "species_" + std::to_string(s) +
                            extension);
        keepAside(files, firstCycle);
      }
      writeVctMapping(subDir + "vctMapping.txt", solver.vct);
      // on a restart the records are read back from the JSON files
      for (int s = 0; s < solver.ns; ++s)
        if (input.spectraSpecies[s])
          keptAside += macrocellHist->initSpeciesFile(subDir, s, solver.myrank,
                                                      isRestart && !remapped,
                                                      firstCycle, solver.vct);
    }
    if (isRestart) { // every rank, with spectra or not
      int total = 0;
      MPI_Reduce(&keptAside, &total, 1, MPI_INT, MPI_SUM, 0,
                 MPIdata::get_PicGlobalComm());
      if (solver.myrank == 0 && total > 0)
        std::fprintf(stderr,
                     "[!] Warning: %d species_<s>.json spectra records in "
                     "%s have other macrocells, bins or velocity range "
                     "than this run: they are kept as *_before_cycle%d.*, "
                     "with their .bin, and start again.\n",
                     total, root.c_str(), firstCycle);
    }
  }
}

/**
 * @brief Fit the Gaussian mixture to the non-empty bins of the velocity
 * histogram of a species, which init() has just computed on the stream.
 *
 * The fit runs on the velocities divided by the largest one of the bins
 * (with GMMNormalize). The first fit, or every fit without GMMWarmStart,
 * starts from equal weights, means drawn uniformly in the ball of that radius
 * and the diagonal covariance of the thermal speeds of the species. A warm
 * start takes the previous fit. A component that was pruned, is close to it
 * (weight below kWarmStartWeightFactor GMMPruneThreshold) or has a non-finite
 * mean or covariance is drawn again, and the weights then restart equal.
 */
void dataAnalysisPipelineImpl::fitGMM(int cycle, int species) {
  // the points are the centres of the non-empty bins, weighted by their
  // content
  const int numPoints = velocityHistogram->compactNonEmpty(stream.get());
  if (numPoints == 0) // no particle in the range: nothing to fit
    return;

  // the largest velocity of the bins: the scale of the fit
  const velocityHistogram::BinGrid& bins = velocityHistogram->getGrid();
  GMMType scale = 0;
  for (int d = 0; d < kDim; d++)
    scale = std::max({scale, (GMMType)std::fabs(bins.min[d]),
                      (GMMType)std::fabs(bins.max[d])});
  const GMMType vth[kDim] = {(GMMType)solver.col->getUth(species),
                             (GMMType)solver.col->getVth(species),
                             (GMMType)solver.col->getWth(species)};

  // reproducible draws, different for every fit
  std::seed_seq seed{cycle, species, solver.myrank};
  std::mt19937 gen(seed);
  std::uniform_real_distribution<GMMType> unif01(0.0, 1.0);
  std::normal_distribution<GMMType> normal(0.0, 1.0);
  auto sampleMean = [&](GMMType* mean) {
    const GMMType n[kDim] = {normal(gen), normal(gen), normal(gen)};
    const GMMType r = scale * std::cbrt(unif01(gen)) /
                      std::sqrt(n[0] * n[0] + n[1] * n[1] + n[2] * n[2]);
    for (int d = 0; d < kDim; d++)
      mean[d] = r * n[d];
  };

  GMMType weight[kComponents];
  GMMType mean[kComponents * kDim];
  GMMType coVariance[kComponents * kDim * kDim] = {};
  const GMMFit* last = input.gmmWarmStart ? lastFit[species].get() : nullptr;
  bool equalWeights = last == nullptr;
  for (int j = 0; j < kComponents; j++) {
    GMMType* m = &mean[j * kDim];
    GMMType* c = &coVariance[j * kDim * kDim];
    bool keep = last && last->weight[j] >=
                            kWarmStartWeightFactor * input.gmmPruneThreshold;
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
      for (int d = 0; d < kDim; d++)
        c[d * (kDim + 1)] = vth[d] * vth[d];
      equalWeights = true;
    }
  }
  for (int j = 0; j < kComponents; j++)
    weight[j] = equalWeights ? GMMType(1) / kComponents : last->weight[j];

  cudaGMMWeight::GMMParam_t<GMMType> param = {
      .numComponents = kComponents,
      .maxIteration = input.gmmMaxIterations,
      .threshold = (GMMType)input.gmmConvergenceThreshold,
      .weightInit = weight,
      .meanInit = mean,
      .coVarianceInit = coVariance,
      .normalize = input.gmmNormalize,
      .scale = scale,
      .checkCovariance = input.gmmCheckCovariance,
      .covarianceTolerance = (GMMType)input.gmmCovarianceTolerance,
      .minVariance = (GMMType)input.gmmMinVariance,
      .prune = input.gmmPruneComponents,
      .pruneThreshold = (GMMType)input.gmmPruneThreshold};

  cudaGMMWeight::GMMDataMultiDim<GMMType, kDim, weightType> data(
      numPoints, velocityHistogram->getNonEmptyCenters(),
      velocityHistogram->getNonEmptyWeights());

  gmm->config(&param, &data);
  const int steps = gmm->initGMM();
  lastFit[species] = std::make_unique<GMMFit>(gmm->getGMMResult(cycle, steps));

  if constexpr (DAConfig::DATA_ANALYSIS_OUTPUT)
    appendJsonRecord(gmmPath + "species" + std::to_string(species) + ".json",
                     cycle, lastFit[species]->outputString());
}

/**
 * @brief The analyses of one cycle, species by species to save GPU memory.
 * @details Runs on the pipeline's thread and on its stream, while the main
 * thread solves the field on the solver stream (of higher priority). It reads
 * the particles (the stream waits for their last sort, see startAnalysis),
 * and for the spectra the cell offsets and the field of prepareAnalysis,
 * synchronised before; the main thread changes none of them until
 * waitForAnalysis() returns. Every species ends with the stream
 * drained: the next one rewrites the host buffers of the copies, and the
 * mover changes the particles after waitForAnalysis().
 * @param cycle Simulation cycle being analysed.
 * @return `0` on success.
 */
int dataAnalysisPipelineImpl::analysisEntry(int cycle) {
  cudaErrChk(cudaSetDevice(solver.cudaDeviceOnNode));

  for (int i = 0; i < solver.ns; i++) {
    if (velocityHistogram) {
      // borrow the SoA pointers of the species (no copy)
      velocitySoACUDA->borrowFrom(solver.pclsArrayHostPtr[i]);
      velocityHistogram->init(velocitySoACUDA.get(), input.histogramMin[i],
                              input.histogramMax[i], stream.get());
      if constexpr (DAConfig::DATA_ANALYSIS_OUTPUT)
        velocityHistogram->writeToFile(histogramPath + "species" +
                                           std::to_string(i) + "_",
                                       cycle, stream.get());
      if (gmm)
        fitGMM(cycle, i);
    }

    if (macrocellHist && input.spectraSpecies[i]) {
      const CellSorter& sorter = solver.cellSorters[i];
      macrocellHist->reset(stream.get());
      macrocellHist->launch(solver.pclsArrayHostPtr[i],
                            solver.fieldForPclCUDAPtr, solver.grid3DCUDACUDAPtr,
                            sorter.getCellStartOffsets(),
                            sorter.getCellCounts(), i, stream.get());
      if constexpr (DAConfig::DATA_ANALYSIS_OUTPUT)
        macrocellHist->writeToFile(i, cycle, stream.get());
    }
    cudaErrChk(cudaStreamSynchronize(stream.get()));
  }

  return 0;
}

/** The spectra read the particles sorted by cell, and B from the mover's
 *  field buffer: sort their species and pack the current field (the buffer
 *  holds the B of before the last CalculateB). */
void dataAnalysisPipelineImpl::prepareAnalysis(int cycle) {
  if (!input.runsAt(cycle) || !macrocellHist)
    return;
  solver.sortSpecies(input.spectraSpecies);
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

/** @return 1 while an analysis is running, 0 otherwise (non-blocking; not
 *  used currently). */
int dataAnalysisPipelineImpl::checkAnalysis() {
  return analysisFuture.valid() &&
         analysisFuture.wait_for(std::chrono::seconds(0)) !=
             std::future_status::ready;
}

/** @return the result of the running analysis, 0 if none (blocking); an
 *  error of the analysis is thrown again here, on the main thread. */
int dataAnalysisPipelineImpl::waitForAnalysis() {
  return analysisFuture.valid()
             ? awaitTask(analysisFuture, "data analysis of cycle " +
                                             std::to_string(analysisCycle))
             : 0;
}

dataAnalysisPipeline::dataAnalysisPipeline(iPic3D::c_Solver& KCode)
    : impl(std::make_unique<dataAnalysisPipelineImpl>(KCode)) {}

void dataAnalysisPipeline::prepareAnalysis(int cycle) {
  impl->prepareAnalysis(cycle);
}

void dataAnalysisPipeline::startAnalysis(int cycle) {
  impl->startAnalysis(cycle);
}

int dataAnalysisPipeline::checkAnalysis() { return impl->checkAnalysis(); }

int dataAnalysisPipeline::waitForAnalysis() { return impl->waitForAnalysis(); }

dataAnalysisPipeline::~dataAnalysisPipeline() = default;

} // namespace dataAnalysis
