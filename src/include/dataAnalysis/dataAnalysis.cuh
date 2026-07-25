#ifndef DATA_ANALYSIS_CUH
#define DATA_ANALYSIS_CUH

#include <memory>
#include <thread>

#include "VCtopology3D.h"
#include "dataAnalysisConfig.cuh"
#include "iPic3D.h"

namespace dataAnalysis {
class dataAnalysisPipelineImpl;

class dataAnalysisPipeline {

private:
  std::unique_ptr<dataAnalysisPipelineImpl> impl;

public:
  dataAnalysisPipeline(iPic3D::c_Solver& KCode);

  // returns true if this cycle triggers data analysis
  static bool isAnalysisCycle(int cycle) {
    return DAConfig::DATA_ANALYSIS_ENABLED &&
           DAConfig::DATA_ANALYSIS_EVERY_CYCLE > 0 &&
           (cycle % DAConfig::DATA_ANALYSIS_EVERY_CYCLE == 0);
  }

  // create the output directory
  static void createOutputDirectory(int myrank, int ns, VirtualTopology3D* vct,
                                    bool isRestart,
                                    bool velocitySpectraEnabled = false);

  // called in the main loop
  void startAnalysis(int cycle);

  // non-blocking check if the analysis is done
  int checkAnalysis();

  // blocking wait for the analysis to finish
  int waitForAnalysis();

  ~dataAnalysisPipeline();
};

} // namespace dataAnalysis

#endif // DATA_ANALYSIS_CUH
