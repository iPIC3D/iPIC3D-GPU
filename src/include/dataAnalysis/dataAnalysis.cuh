#ifndef DATA_ANALYSIS_CUH
#define DATA_ANALYSIS_CUH

#include <memory>

#include "iPic3D.h"

namespace dataAnalysis {
class dataAnalysisPipelineImpl;

/**
 * @brief The data analyses of the input file (DataAnalysisInput), run every
 * DataAnalysisCycle cycles during the field solve, on a host thread and a GPU
 * stream of the lowest priority: velocity histograms, their GMM fit and the
 * macrocell spectra.
 */
class dataAnalysisPipeline {

private:
  std::unique_ptr<dataAnalysisPipelineImpl> impl;

public:
  /** Sets up the enabled analyses and opens their records files, one per
   *  analysis and rank (on a restart: continued, cut at the first cycle,
   *  or kept aside); collective (MPI), every rank constructs it. */
  dataAnalysisPipeline(iPic3D::c_Solver& KCode);

  /** On an analysis cycle with the macrocell spectra or histograms, sort by
   *  cell the particles of their species and, for the spectra, pack B
   *  (synchronous); called before startAnalysis. */
  void prepareAnalysis(int cycle);

  /** Start the analyses of the cycle, if it is an analysis cycle, on the
   *  pipeline's thread; the mover must wait for waitForAnalysis. */
  void startAnalysis(int cycle);

  /** Blocking wait for the analysis, if one runs. An error of the analysis
   *  thread is thrown again here, on the calling thread. */
  void waitForAnalysis();

  ~dataAnalysisPipeline();
};

} // namespace dataAnalysis

#endif // DATA_ANALYSIS_CUH
