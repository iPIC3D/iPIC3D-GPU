
#include <thread>
#include <future>
#include <vector>
#include <string>
#include <memory>
#include <random>
#include <type_traits>
#include <cstring>

#include "iPic3D.h"
#include "VCtopology3D.h"
#include "outputPrepare.h"
#include "threadPool.hpp"

#include "dataAnalysis.cuh"
#include "dataAnalysisConfig.cuh"
#include "GMM/cudaGMMUtility.cuh"
#include "GMM/cudaGMM.cuh"
#include "particleArraySoACUDA.cuh"
#include "velocityHistogram.cuh"

#ifdef USE_ADIOS2
#include "adios2.h"
#endif

namespace dataAnalysis
{

using namespace iPic3D;
using velocitySoA = particleArraySoA::particleArraySoACUDA<cudaParticleType, 0, 3>;
using GMMType = cudaParticleType;
using weightType = velocityHistogram::histogramTypeOut;

class dataAnalysisPipelineImpl {
private:
    int ns;
    int deviceOnNode;
    // pointers to objects in KCode
    cudaStream_t* streams;
    particleArrayCUDA** pclsArrayHostPtr = nullptr;

    std::future<int> analysisFuture;

    ThreadPool* DAthreadPool = nullptr;
    velocitySoA* velocitySoACUDA = nullptr;
    // histogram
    string HistogramSubDomainOutputPath;
    velocityHistogram::velocityHistogram* velocityHistogram = nullptr;

    // GMM
    string GMMSubDomainOutputPath;
    cudaGMMWeight::GMM<GMMType, DATA_DIM_GMM, weightType>* gmmArray = nullptr;
    cudaGMMWeight::GMMParam_output_store<GMMType> paramHost_last_array[12]; // object on host, GMM output parameters at the last cycle - numSpecies * numPlanes(uvw) (4*3 in this case)
    // dimensions: numSpecies, numPlanes (uvw), numGMMDA
    std::vector<std::array<std::vector<cudaGMMWeight::GMMResult<GMMType, DATA_DIM_GMM>>, 3>> gmmResults;

#ifdef USE_ADIOS2
    adios2::ADIOS adios;
    adios2::IO ioGMM;
    adios2::Engine engineGMM;
#endif

public:

    dataAnalysisPipelineImpl(c_Solver& KCode) {
        ns = KCode.ns;
        deviceOnNode = KCode.cudaDeviceOnNode;
        streams = KCode.streams;
        pclsArrayHostPtr = KCode.pclsArrayHostPtr;

        DAthreadPool = new ThreadPool(4);

        if constexpr (VELOCITY_HISTOGRAM_ENABLE) { // velocity histogram
            velocitySoACUDA = new velocitySoA();

            HistogramSubDomainOutputPath = HISTOGRAM_OUTPUT_DIR + "subDomain" + std::to_string(KCode.myrank) + "/";
            velocityHistogram = new velocityHistogram::velocityHistogram(VELOCITY_HISTOGRAM_RES*VELOCITY_HISTOGRAM_RES);

            if constexpr (GMM_ENABLE) { // GMM
                GMMSubDomainOutputPath = GMM_OUTPUT_DIR + "subDomain" + std::to_string(KCode.myrank) + "/";
                gmmArray = new cudaGMMWeight::GMM<GMMType, DATA_DIM_GMM, weightType>[3];
                if constexpr (GMM_OUTPUT) {
                    gmmResults.resize(ns);

#ifdef USE_ADIOS2
                    if constexpr (GMM_ENABLE && GMM_OUTPUT && OUTPUT_ALL_END_GMM){
                        adios = adios2::ADIOS(MPIdata::get_PicGlobalComm());
                        ioGMM = adios.DeclareIO("GMM");
                        engineGMM = ioGMM.Open(GMM_OUTPUT_DIR + "subDomain" + std::to_string(KCode.myrank) + "/" + "GMMResult.bp", adios2::Mode::Write, MPI_COMM_SELF);
                    }
#endif

                }
            }
        }
    }

    void startAnalysis(int cycle);

    int checkAnalysis();

    int waitForAnalysis();

#ifdef USE_ADIOS2
    void outputGMMADIOS2();
#endif

    void writeGMMResults() {
        if constexpr (GMM_OUTPUT && OUTPUT_ALL_END_GMM) {
            std::string uvw[3] = {"uv", "vw", "uw"};

            int i = 0; // species index
            for (auto& speciesResArray : gmmResults) {
                int j = 0; // uvw index
                for (auto& plane : speciesResArray) {
                    string planePath = GMMSubDomainOutputPath + "species" + std::to_string(i) + "_" + uvw[j] + ".json";
                    cudaGMMWeight::GMMResult<GMMType, DATA_DIM_GMM>::outputResultArray(plane, planePath, uvw[j]); 
                    j++;  
                }
                i++;
            }
        }
    }


    ~dataAnalysisPipelineImpl() {     

        if (DAthreadPool != nullptr) delete DAthreadPool;
        if (velocitySoACUDA != nullptr) delete velocitySoACUDA;
        if (velocityHistogram != nullptr) delete velocityHistogram;
        if (gmmArray != nullptr) delete[] gmmArray;
    }

private:

    int analysisEntre(int cycle);

    int GMMAnalysisSpecies(const int cycle, const int species, const std::string outputPath);
    
};



/**
 * @brief analysis function for each species, uv, uw, vw
 * @details It launches 3 threads for uv uw vw analysis in parallel
 * 
 */
int dataAnalysisPipelineImpl::GMMAnalysisSpecies(const int cycle, const int species, const std::string outputPath){

    using weightType = cudaTypeSingle;

    std::future<int> future[3];

    auto GMMLambda = [=](int i) mutable {

        using namespace cudaGMMWeight;

        cudaErrChk(cudaSetDevice(deviceOnNode));

        // GMM config
        auto& paramHost_last = paramHost_last_array[species * 3 + i];

        // set the random number generator to sample velocity from circle of radius max velocity
        std::random_device rd;  // True random seed
        std::mt19937 gen(rd()); // Mersenne Twister PRN
        std::uniform_real_distribution<GMMType>  unif01(0.0, 1.0);
        std::uniform_real_distribution<GMMType> distTheta(0, 2*M_PI);

        const GMMType maxVelocity = (species == 0 || species == 2) ? MAX_VELOCITY_HIST_E : MAX_VELOCITY_HIST_I;
        // it is assumed that DATA_DIM_GMM == 2 and that the velocity range is homogenues in all dimensions
        const GMMType maxVelocityArray[DATA_DIM_GMM] = {maxVelocity,maxVelocity};
        // right now it is not used (fixed to zero) --> mean is not subtracted to data, but it might be useful for future developments
        const GMMType meanArray[DATA_DIM_GMM] = {0.0,0.0};

        const GMMType uth = species == 0 || species == 2 ? 0.045 : 0.0126;
        const GMMType vth = species == 0 || species == 2 ? 0.045 : 0.0126;
        const GMMType wth = species == 0 || species == 2 ? 0.045 : 0.0126;
        
        GMMType var1 = 0.01;
        GMMType var2 = 0.01; 

        GMMType weightVector[NUM_COMPONENT_GMM];
        GMMType meanVector[NUM_COMPONENT_GMM * DATA_DIM_GMM];
        GMMType coVarianceMatrix[NUM_COMPONENT_GMM * DATA_DIM_GMM * DATA_DIM_GMM ];
        
        if (i==0){
            var1 = uth*uth; 
            var2 = vth*vth;
        }
        else if(i==1){
            var1 = vth*vth;
            var2 = wth*wth;
        }
        else if(i==2){
            var1 = uth*uth;
            var2 = wth*wth;
        }
        
        // normalize initial parameters if NORMALIZE_DATA_FOR_GMM==true
        GMMType normalization = 1.0; 
        if constexpr(NORMALIZE_DATA_FOR_GMM) normalization = maxVelocity;

        if constexpr (START_WITH_LAST_PARAMETERS_GMM) // start GMM with output GMM parameters from last cycle as initial parameters
        {
            // if cycle == 0 initialize GMM with the usual fixed parameters
            if(cycle == 0)
            {
                paramHost_last.numComponents = NUM_COMPONENT_GMM;
                paramHost_last.maxIteration = MAX_ITERATION_GMM;
                paramHost_last.threshold = THRESHOLD_CONVERGENCE_GMM;
                
                for(int j = 0; j < NUM_COMPONENT_GMM; j++){
                    weightVector[j] = 1.0/NUM_COMPONENT_GMM;
                    GMMType radius = maxVelocity * sqrt(unif01(gen));
                    GMMType theta = distTheta(gen);
                    meanVector[j * 2] =  radius*cos(theta);
                    meanVector[j * 2 + 1] = radius*sin(theta);
                    coVarianceMatrix[j * 4] = var1;
                    coVarianceMatrix[j * 4 + 1] = 0.0;
                    coVarianceMatrix[j * 4 + 2] = 0.0;
                    coVarianceMatrix[j * 4 + 3] = var2;
                }
            }
            else // if cycle > 0 initialize GMM with previous output parameters
            {   
                bool reset = false;
                
                // safety checks on components weights and mean since after pruning some components have zero weight
                // check if meanVector is NaN or component weight is too small
                // if meanVector is NaN sample new mean vector
                for(int j = 0; j < NUM_COMPONENT_GMM; j++){ 
                    if( std::isnan(paramHost_last.meanVector[j * 2]) || std::isnan(paramHost_last.meanVector[j * 2 + 1]) || 
                        std::isinf(paramHost_last.meanVector[j * 2]) || std::isinf(paramHost_last.meanVector[j * 2 + 1]) ){
                        reset = true;
                        GMMType radius = maxVelocity * sqrt(unif01(gen));
                        GMMType theta = distTheta(gen);
                        meanVector[j * 2] =  radius*cos(theta) / normalization;
                        meanVector[j * 2 + 1] = radius*sin(theta) / normalization;
                    }
                    else{
                        meanVector[j * 2] = paramHost_last.meanVector[j * 2] / normalization;
                        meanVector[j * 2 + 1] = paramHost_last.meanVector[j * 2 + 1] / normalization;
                    }

                    if( paramHost_last.weightVector[j] < PRUNE_THRESHOLD_GMM*10 ) reset = true;
                }

                // if meanVector is NaN or weigth is too small reset components weights
                // adjust cov if it is too small
                for(int j = 0; j < NUM_COMPONENT_GMM; j++){
                    if(reset){
                        weightVector[j] = 1.0/NUM_COMPONENT_GMM;
                    }
                    else{
                        weightVector[j] = paramHost_last.weightVector[j];
                    }

                    coVarianceMatrix[j * 4] = paramHost_last.coVarianceMatrix[j * 4] > (10 * EPS_COVMATRIX_GMM * (normalization*normalization)) ? 
                                                        paramHost_last.coVarianceMatrix[j * 4] : var1;
                    coVarianceMatrix[j * 4] /= (normalization*normalization);                               
                    coVarianceMatrix[j * 4 + 1] = 0.0;
                    coVarianceMatrix[j * 4 + 2] = 0.0;
                    coVarianceMatrix[j * 4 + 3] = paramHost_last.coVarianceMatrix[j * 4 + 3] > (10 * EPS_COVMATRIX_GMM * (normalization*normalization)) ? 
                                                            paramHost_last.coVarianceMatrix[j * 4 + 3] : var2;
                    coVarianceMatrix[j * 4 + 3] /= (normalization*normalization);
                }
            }
        }
        else // start GMM with fixed initial parameters at any cycle
        {
            for(int j = 0; j < NUM_COMPONENT_GMM; j++){
                weightVector[j] = 1.0/NUM_COMPONENT_GMM;
                GMMType radius = maxVelocity * sqrt(unif01(gen));
                GMMType theta = distTheta(gen);
                meanVector[j * 2] =  radius*cos(theta) / normalization;
                meanVector[j * 2 + 1] = radius*sin(theta) / normalization;
                coVarianceMatrix[j * 4] = var1 / (normalization*normalization);
                coVarianceMatrix[j * 4 + 1] = 0.0;
                coVarianceMatrix[j * 4 + 2] = 0.0;
                coVarianceMatrix[j * 4 + 3] = var2 / (normalization*normalization);
            }
        }

        GMMParam_t<GMMType> GMMParam = {
            .numComponents = NUM_COMPONENT_GMM,
            .maxIteration = MAX_ITERATION_GMM,
            .threshold = THRESHOLD_CONVERGENCE_GMM,
            .weightInit = weightVector,
            .meanInit = meanVector,
            .coVarianceInit = coVarianceMatrix
        };  


        // data
        GMMDataMultiDim<GMMType, DATA_DIM_GMM, weightType> GMMData
            (VELOCITY_HISTOGRAM_RES*VELOCITY_HISTOGRAM_RES, velocityHistogram->getHistogramScaleMark(i), velocityHistogram->getVelocityHistogramCUDAArray(i), 
            {maxVelocityArray[0], maxVelocityArray[1]});

        cudaErrChk(cudaHostRegister(&GMMData, sizeof(GMMData), cudaHostRegisterDefault));
        
        // generate exact output file path
        std::string uvw[3] = {"uv_", "vw_", "uw_"};
        auto fileOutputPath = outputPath + uvw[i] + std::to_string(cycle) + ".json";
        
        auto& gmm = gmmArray[i];

        gmm.config(&GMMParam, &GMMData);
        // preprocess the data --> normalize data
        gmm.preProcessDataGMM(meanArray);
        // run GMM
        auto convergStep = gmm.initGMM(fileOutputPath); // the exact output file name
        // postporcess data --> normalize data back
        gmm.postProcessDataGMM();
        // move GMM output to host
        int moveBackToHost = gmm.moveBackToHostGMM(paramHost_last);
        int ret = 0;
        
        if constexpr (GMM_OUTPUT) {
            ret = gmm.outputGMM(convergStep, fileOutputPath, moveBackToHost); // immediate output
            // results vector
            if constexpr (OUTPUT_ALL_END_GMM) gmmResults[species][i].push_back(gmm.getGMMResult(cycle, convergStep));
        }

        cudaErrChk(cudaHostUnregister(&GMMData));

        return ret;
    };

    for(int i = 0; i < 3; i++){
        // launch 3 async threads for uv, uw, vw
        future[i] = DAthreadPool->enqueue(GMMLambda, i); 
    }

    for(int i = 0; i < 3; i++){
        future[i].wait();
    }

    return 0;
}

/**
 * @brief analysis function, called by startAnalysis
 * @details procesures in this function should be executed in sequence, the order of the analysis should be defined here
 *          But the procedures can launch other threads to do the analysis
 *          Also this function is a friend function of c_Solver, resources in the c_Slover should be dispatched here
 */
int dataAnalysisPipelineImpl::analysisEntre(int cycle){
    cudaErrChk(cudaSetDevice(deviceOnNode));

    // species by species to save VRAM
    for(int i = 0; i < ns; i++){
        if constexpr (VELOCITY_HISTOGRAM_ENABLE) {
            // to SoA
            velocitySoACUDA->updateFromAoS(pclsArrayHostPtr[i], streams[i]);

            // histogram
            auto histogramSpeciesOutputPath = HistogramSubDomainOutputPath + "species" + std::to_string(i) + "/";
            velocityHistogram->init(velocitySoACUDA, cycle, i, streams[i]);
            if constexpr (HISTOGRAM_OUTPUT)
            velocityHistogram->writeToFile(histogramSpeciesOutputPath, streams[i]); // TODO
            else cudaErrChk(cudaStreamSynchronize(streams[i]));

            if constexpr (GMM_ENABLE) { // GMM
                    auto GMMSpeciesOutputPath = GMMSubDomainOutputPath + "species" + std::to_string(i) + "/";
                    GMMAnalysisSpecies(cycle, i, GMMSpeciesOutputPath);
            }
        }
    }

    
    return 0;
}


/**
 * @brief start all the analysis registered here
 */
void dataAnalysisPipelineImpl::startAnalysis(int cycle){

    if(DATA_ANALYSIS_EVERY_CYCLE == 0 || (cycle % DATA_ANALYSIS_EVERY_CYCLE != 0)){
        analysisFuture = std::future<int>();
    } else {
        analysisFuture = DAthreadPool->enqueue(&dataAnalysisPipelineImpl::analysisEntre, this, cycle); 

        if(analysisFuture.valid() == false){
            throw std::runtime_error("[!]Error: Can not start data analysis");
        }
    }

}

/**
 * @brief check if the analysis is done, non-blocking
 * 
 * @return 0 if the analysis is done, 1 if it is not done
 */
int dataAnalysisPipelineImpl::checkAnalysis(){

    if(analysisFuture.valid() == false){
        return 0;
    }

    if(analysisFuture.wait_for(std::chrono::seconds(0)) == std::future_status::ready){
        return 0;
    }else{
        return 1;
    }

    return 0;
}

/**
 * @brief wait for the analysis to be done, blocking
 */
int dataAnalysisPipelineImpl::waitForAnalysis(){

    if(analysisFuture.valid() == false){
        return 0;
    }

    analysisFuture.wait();

    return 0;
}

#ifdef USE_ADIOS2

void dataAnalysisPipelineImpl::outputGMMADIOS2() {
    if constexpr (!GMM_OUTPUT) return;

    constexpr int GMMDim = 2;

    // variable array
    adios2::Variable<GMMType> varReal[ns][3][4];
    adios2::Variable<int> varInt[ns][3][3];

    // register the variables
    for(int i = 0; i < ns; i++){
        for(int j = 0; j < 3; j++){
            std::string uvw[3] = {"uv", "vw", "uw"};
            std::string varName = "species" + std::to_string(i) + "_" + uvw[j];

            adios2::Dims shape = {1}; // {component}
            varReal[i][j][0] = ioGMM.DefineVariable<GMMType>(varName+"_weight", shape, {0}, shape, false);
            varReal[i][j][1] = ioGMM.DefineVariable<GMMType>(varName+"_mean", shape, {0}, shape, false);
            varReal[i][j][2] = ioGMM.DefineVariable<GMMType>(varName+"_coVariance", shape, {0}, shape, false);


            // loglikelihood
            std::string logLikelihoodVarName = "species" + std::to_string(i) + "_" + uvw[j] + "_logLikelihood";
            varReal[i][j][3] = ioGMM.DefineVariable<GMMType>(logLikelihoodVarName);

            // convergence step
            std::string convergeStepVarName = "species" + std::to_string(i) + "_" + uvw[j] + "_convergeStep";
            varInt[i][j][0] = ioGMM.DefineVariable<int>(convergeStepVarName);

            // simulation step
            std::string simuStepVarName = "species" + std::to_string(i) + "_" + uvw[j] + "_simulationStep";
            varInt[i][j][1] = ioGMM.DefineVariable<int>(simuStepVarName);

            // number of components
            std::string numComponentsVarName = "species" + std::to_string(i) + "_" + uvw[j] + "_numComponents";
            varInt[i][j][2] = ioGMM.DefineVariable<int>(numComponentsVarName);

        }
    }


    int cycleCount = gmmResults[0][0].size();
    for(int i = 0; i < cycleCount; i++){
        engineGMM.BeginStep();
        for(int j=0; j < ns; j++){ 
            for(int k=0; k < 3; k++){

                // write data
                engineGMM.Put(varInt[j][k][0], gmmResults[j][k][i].convergeStep);
                engineGMM.Put(varInt[j][k][1], gmmResults[j][k][i].simulationStep);
                engineGMM.Put(varInt[j][k][2], gmmResults[j][k][i].numComponents);

                engineGMM.Put(varReal[j][k][3], gmmResults[j][k][i].logLikelihoodFinal);

                // write the GMM components, adjust the shape first
                const size_t componentNum = (size_t)gmmResults[j][k][i].numComponents;
                varReal[j][k][0].SetShape({componentNum}); 
                varReal[j][k][1].SetShape({componentNum * GMMDim});
                varReal[j][k][2].SetShape({componentNum * GMMDim * GMMDim});

                varReal[j][k][0].SetSelection({{0}, {componentNum}});
                varReal[j][k][1].SetSelection({{0}, {componentNum * GMMDim}});
                varReal[j][k][2].SetSelection({{0}, {componentNum * GMMDim * GMMDim}});

                engineGMM.Put<GMMType>(varReal[j][k][0], gmmResults[j][k][i].weight.get());
                engineGMM.Put<GMMType>(varReal[j][k][1], gmmResults[j][k][i].mean.get());
                engineGMM.Put<GMMType>(varReal[j][k][2], gmmResults[j][k][i].coVariance.get());
                
            }
        }
        engineGMM.EndStep();
    }

    engineGMM.Close();
}


#endif



/**
 * @brief create output directory for the data analysis, controlled by dataAnalysisConfig.cuh
 */
void dataAnalysisPipeline::createOutputDirectory(int myrank, int ns, VirtualTopology3D* vct){ // output path for data analysis
    if constexpr (DATA_ANALYSIS_ENABLED == false){
        return;
    }

    // VCT mapping for this subdomain
    auto writeVctMapping = [&](const std::string& filePath) {
        std::ofstream vctMapping(filePath);
        if(vctMapping.is_open()){
        vctMapping << "Cartesian rank: " << vct->getCartesian_rank() << std::endl;
        vctMapping << "Number of processes: " << vct->getNprocs() << std::endl;
        vctMapping << "XLEN: " << vct->getXLEN() << std::endl;
        vctMapping << "YLEN: " << vct->getYLEN() << std::endl;
        vctMapping << "ZLEN: " << vct->getZLEN() << std::endl;
        vctMapping << "X: " << vct->getCoordinates(0) << std::endl;
        vctMapping << "Y: " << vct->getCoordinates(1) << std::endl;
        vctMapping << "Z: " << vct->getCoordinates(2) << std::endl;
        vctMapping << "PERIODICX: " << vct->getPERIODICX() << std::endl;
        vctMapping << "PERIODICY: " << vct->getPERIODICY() << std::endl;
        vctMapping << "PERIODICZ: " << vct->getPERIODICZ() << std::endl;

        vctMapping << "Neighbor X left: " << vct->getXleft_neighbor() << std::endl;
        vctMapping << "Neighbor X right: " << vct->getXright_neighbor() << std::endl;
        vctMapping << "Neighbor Y left: " << vct->getYleft_neighbor() << std::endl;
        vctMapping << "Neighbor Y right: " << vct->getYright_neighbor() << std::endl;
        vctMapping << "Neighbor Z left: " << vct->getZleft_neighbor() << std::endl;
        vctMapping << "Neighbor Z right: " << vct->getZright_neighbor() << std::endl;

        vctMapping.close();
        } else {
        throw std::runtime_error("[!]Error: Can not create VCT mapping for velocity GMM species");
        }
    };

    if constexpr (VELOCITY_HISTOGRAM_ENABLE && HISTOGRAM_OUTPUT) {
        auto histogramSubDomainOutputPath = HISTOGRAM_OUTPUT_DIR + "subDomain" + std::to_string(myrank) + "/";
        if(0 != checkOutputFolder(histogramSubDomainOutputPath)){
            throw std::runtime_error("[!]Error: Can not create output folder for velocity histogram");
        }

        for(int i = 0; i < ns; i++){
            auto histogramSpeciesOutputPath = histogramSubDomainOutputPath + "species" + std::to_string(i);
            if(0 != checkOutputFolder(histogramSpeciesOutputPath)){
            throw std::runtime_error("[!]Error: Can not create output folder for velocity histogram species");
            }
        }
        writeVctMapping(histogramSubDomainOutputPath + "vctMapping.txt");
    }

    if constexpr (GMM_ENABLE && GMM_OUTPUT) {
        auto GMMSubDomainOutputPath = GMM_OUTPUT_DIR + "subDomain" + std::to_string(myrank) + "/";
        if(0 != checkOutputFolder(GMMSubDomainOutputPath)){
            throw std::runtime_error("[!]Error: Can not create output folder for velocity GMM");
        }
        for(int i = 0; i < ns; i++){
            auto GMMSpeciesOutputPath = GMMSubDomainOutputPath + "species" + std::to_string(i) + "/";
            if(0 != checkOutputFolder(GMMSpeciesOutputPath)){
            throw std::runtime_error("[!]Error: Can not create output folder for velocity GMM species");
            }
        }
        writeVctMapping(GMMSubDomainOutputPath + "vctMapping.txt");
    }

}



dataAnalysisPipeline::dataAnalysisPipeline(iPic3D::c_Solver& KCode) {
    if constexpr (DATA_ANALYSIS_ENABLED == false){
        impl = nullptr;
        return;
    }
    impl = std::make_unique<dataAnalysisPipelineImpl>(std::ref(KCode));
}

void dataAnalysisPipeline::startAnalysis(int cycle) {
    if constexpr (DATA_ANALYSIS_ENABLED == false){
        return;
    }
    impl->startAnalysis(cycle);
}

int dataAnalysisPipeline::checkAnalysis() {
    if constexpr (DATA_ANALYSIS_ENABLED == false){
        return 0;
    }
    return impl->checkAnalysis();
}

int dataAnalysisPipeline::waitForAnalysis() {
    if constexpr (DATA_ANALYSIS_ENABLED == false){
        return 0;
    }
    return impl->waitForAnalysis();
}

void dataAnalysisPipeline::writeGMMResults() {
    if constexpr ((GMM_ENABLE && GMM_OUTPUT && OUTPUT_ALL_END_GMM) == false){
        return;
    }
    impl->writeGMMResults();
    impl->outputGMMADIOS2();
}

dataAnalysisPipeline::~dataAnalysisPipeline() {
    
}
    
} // namespace dataAnalysis







