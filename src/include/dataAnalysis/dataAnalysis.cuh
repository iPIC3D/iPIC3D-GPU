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
  /** Creates the output folders of the enabled analyses; collective (MPI),
   *  every rank constructs it. */
  dataAnalysisPipeline(iPic3D::c_Solver& KCode);

  /** On an analysis cycle with the macrocell spectra, sort by cell the
   *  particles of their species and pack B for them (synchronous); called
   *  before startAnalysis. */
  void prepareAnalysis(int cycle);

  /** Start the analyses of the cycle, if it is an analysis cycle, on the
   *  pipeline's thread; the mover must wait for waitForAnalysis. */
  void startAnalysis(int cycle);

  /** Non-blocking: 1 while an analysis runs (not used currently). */
  int checkAnalysis();

  /** Blocking wait for the analysis: its result, 0 if none. An error of
   *  the analysis thread is thrown again here, on the calling thread. */
  int waitForAnalysis();

  ~dataAnalysisPipeline();
};

} // namespace dataAnalysis

#endif // DATA_ANALYSIS_CUH
