#ifndef _DATA_ANALYSIS_CUH_
#define _DATA_ANALYSIS_CUH_

#include <thread>
#include <memory>

#include "iPic3D.h"
#include "VCtopology3D.h"



namespace dataAnalysis
{
class dataAnalysisPipelineImpl;

class dataAnalysisPipeline {

private:
    std::unique_ptr<dataAnalysisPipelineImpl> impl;

public:

    dataAnalysisPipeline(iPic3D::c_Solver& KCode);

    // create the output directory
    static void createOutputDirectory(int myrank, int ns, VirtualTopology3D* vct);

    // called in the main loop
    void startAnalysis(int cycle);

    // non-blocking check if the analysis is done
    int checkAnalysis();

    // blocking wait for the analysis to finish
    int waitForAnalysis();

    // write the results of GMM, the aggregated method
    void writeGMMResults();

    ~dataAnalysisPipeline();

};


    
} // namespace dataAnalysis






#endif





