
#include <thread>
#include <vector>
#include <future>
#include <string>
#include <memory>
#include <random>

#include "iPic3D.h"
#include "VCtopology3D.h"
#include "outputPrepare.h"
#include "threadPool.hpp"

#include "dataAnalysis.cuh"
#include "dataAnalysisConfig.cuh"
#include "GMM/cudaGMM.cuh"
#include "particleArraySoACUDA.cuh"
#include "velocityHistogram.cuh"



namespace dataAnalysis
{

using namespace iPic3D;
using velocitySoA = particleArraySoA::particleArraySoACUDA<cudaParticleType, 0, 3>;
using namespace std;
using namespace cudaGMMWeight;

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
    cudaGMMWeight::GMM<GMMType, GMM_DATA_DIM, weightType>* gmmArray = nullptr;

    vector<array<vector<GMMResult<GMMType, GMM_DATA_DIM>>, 3>> gmmResults;

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
            velocityHistogram = new velocityHistogram::velocityHistogram(VELOCITY_HISTOGRAM_RES * VELOCITY_HISTOGRAM_RES);

            if constexpr (GMM_ENABLE) { // GMM
                GMMSubDomainOutputPath = GMM_OUTPUT_DIR + "subDomain" + std::to_string(KCode.myrank) + "/";
                gmmArray = new cudaGMMWeight::GMM<GMMType, GMM_DATA_DIM, weightType>[3];

                if constexpr (GMM_OUTPUT) gmmResults.resize(ns);

            }
        }
    }

    void startAnalysis(int cycle);

    int checkAnalysis();

    int waitForAnalysis();

    void writeGMMResults() {

        if constexpr (GMM_OUTPUT) {
            std::string uvw[3] = {"uv", "vw", "uw"};

            int i = 0; // species index
            for (auto& speciesResArray : gmmResults) {
                int j = 0; // uvw index
                for (auto& plane : speciesResArray) {
                    string planePath = GMMSubDomainOutputPath + "species" + std::to_string(i) + "_" + uvw[j] + ".json";
                    GMMResult<GMMType, GMM_DATA_DIM>::outputResultArray(plane, planePath, uvw[j]); 
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

    std::future<int> future[3];

    auto GMMLambda = [=](int i) mutable {

        using namespace cudaGMMWeight;

        // GMM config
        // set the random number generator
        std::random_device rd;  // True random seed
        std::mt19937 gen(rd()); // Mersenne Twister PRN
        const GMMType maxVelocity = species == 0 || species == 2 ? MAX_VELOCITY_HIST_E : MAX_VELOCITY_HIST_I;

        // it is assumed that GMM_DATA_DIM == 2 and thta the velocity range is homogenues in all dimensions
        const GMMType maxVelocityArray[GMM_DATA_DIM] = {maxVelocity,maxVelocity};

        std::uniform_real_distribution<GMMType> distR(1e-8, maxVelocity);
        std::uniform_real_distribution<GMMType> distTheta(0, 2*M_PI);
        

        cudaErrChk(cudaSetDevice(deviceOnNode));

        GMMType weightVector[NUM_COMPONENT_GMM];
        GMMType meanVector[NUM_COMPONENT_GMM * GMM_DATA_DIM];
        GMMType coVarianceMatrix[NUM_COMPONENT_GMM * GMM_DATA_DIM * GMM_DATA_DIM ];

        const GMMType uth = species == 0 || species == 2 ? 0.045 : 0.0126;
        const GMMType vth = species == 0 || species == 2 ? 0.045 : 0.0126;
        const GMMType wth = species == 0 || species == 2 ? 0.045 : 0.0126;
        
        GMMType var1 = 0.01;
        GMMType var2 = 0.01; 
        
        if (i==0)
        {
            var1 = uth;
            var2 = vth;
        }
        else if(i==1)
        {
            var1 = uth;
            var2 = wth;
        }
        else if(i==2)
        {
            var1 = vth;
            var2 = wth;
        }
        
        
        GMMType normalization = 1.0;
        // normalize initial parameters if NORMALIZE_DATA_FOR_GMM==true
        if constexpr(NORMALIZE_DATA_FOR_GMM)
        {
            normalization = maxVelocity;
        }
        for(int j = 0; j < NUM_COMPONENT_GMM; j++){
            weightVector[j] = 1.0/NUM_COMPONENT_GMM;
            GMMType radius = distR(gen);
            GMMType theta = distTheta(gen);
            meanVector[j * 2] =  radius*cos(theta)/normalization;
            meanVector[j * 2 + 1] = radius*sin(theta)/normalization;
            coVarianceMatrix[j * 4] = var1/(normalization*normalization);
            coVarianceMatrix[j * 4 + 1] = 0.0;
            coVarianceMatrix[j * 4 + 2] = 0.0;
            coVarianceMatrix[j * 4 + 3] = var2/(normalization*normalization);
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
        GMMDataMultiDim<GMMType, GMM_DATA_DIM, weightType> GMMData
            (VELOCITY_HISTOGRAM_RES*VELOCITY_HISTOGRAM_RES, 
                velocityHistogram->getHistogramScaleMark(i), 
                velocityHistogram->getVelocityHistogramCUDAArray(i), 
                {maxVelocityArray[0], maxVelocityArray[1]});

        cudaErrChk(cudaHostRegister(&GMMData, sizeof(GMMData), cudaHostRegisterDefault));
        
        // generate exact output file path
        std::string uvw[3] = {"/uv_", "/vw_", "/uw_"};
        auto fileOutputPath = outputPath + uvw[i] + std::to_string(cycle) + ".json";

        auto& gmm = gmmArray[i];
        gmm.config(&GMMParam, &GMMData);
        auto convergStep = gmm.initGMM(); // the exact output file name
        int ret = 0;
        if constexpr (GMM_OUTPUT) {
            ret = gmm.outputGMM(convergStep, fileOutputPath); // immediate output

            // results vector
            gmmResults[species][i].push_back(gmm.getGMMResult(cycle, convergStep));
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
    if constexpr (DATA_ANALYSIS_ENABLED == false){
        return;
    }
    impl->writeGMMResults();
}

dataAnalysisPipeline::~dataAnalysisPipeline() {
    
}
    
} // namespace dataAnalysis







