
#include <thread>
#include <future>
#include <vector>
#include <string>
#include <memory>
#include <random>
#include <type_traits>
#include <cstring>
#include <iostream>
#include <fstream>
#include <filesystem>

#include "iPic3D.h"
#include "VCtopology3D.h"
#include "Collective.h"
#include "Grid3DCU.h"
#include "outputPrepare.h"
#include "threadPool.hpp"

#include "dataAnalysis.cuh"
#include "dataAnalysisConfig.cuh"
#include "GMM/cudaGMMUtility.cuh"
#include "GMM/cudaGMM.cuh"
#include "particleArraySoAView.cuh"
#include "velocityHistogram.cuh"
#include "macrocellSpectra.cuh"
#include "cellSortBuffers.cuh"


#include "mpi.h"

namespace dataAnalysis
{

using namespace iPic3D;
using velocitySoA = particleArraySoAView<cudaParticleType, 4>;
using GMMType = cudaParticleType;
using weightType = velocityHistogram::histogramTypeOut;

using namespace DAConfig;

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
    velocityHistogram::velocityHistogram3D* velocityHistogram = nullptr;

    // GMM
    string GMMSubDomainOutputPath;
    cudaGMMWeight::GMM<GMMType, DATA_DIM_GMM, weightType>* gmmArray = nullptr;
    // dimensions: numSpecies, cycles
    std::vector<std::vector<cudaGMMWeight::GMMResult<GMMType, DATA_DIM_GMM>>> gmmResults;

    // Macrocell (v_par, v_perp) spectra
    std::unique_ptr<macrocellSpectra::macrocellSpectra2D> macrocellHist;
    std::string macrocellSubdomainDir;
    int  macrocellMyRank = 0;
    bool macrocellEnabled = false;
    std::vector<char> macrocellSpeciesMask; // size ns, 1=on, 0=off
    // Borrowed from c_Solver for the macrocell module:
    cudaCommonType*    macrocellFieldForPcls = nullptr; // shared by all species
    const grid3DCUDA*  macrocellGridDevicePtr = nullptr;
    CellSorter*        macrocellCellSorters   = nullptr; // per-species sorter array
    const VCtopology3D*      macrocellVct      = nullptr;


public:

    dataAnalysisPipelineImpl(c_Solver& KCode) {
        ns = KCode.ns;
        deviceOnNode = KCode.cudaDeviceOnNode;
        streams = KCode.streams;
        pclsArrayHostPtr = KCode.pclsArrayHostPtr;

        DAthreadPool = new ThreadPool(4);

        if constexpr (VELOCITY_HISTOGRAM_ENABLE) { // velocity histogram
            velocitySoACUDA = new velocitySoA();

            HistogramSubDomainOutputPath = HISTOGRAM_OUTPUT_DIR + "subDomain" + std::to_string(KCode.myrank) + "_";
            velocityHistogram = new velocityHistogram::velocityHistogram3D(VELOCITY_HISTOGRAM3D_SIZE);

            if constexpr (GMM_ENABLE) { // GMM
                GMMSubDomainOutputPath = GMM_OUTPUT_DIR + "subDomain" + std::to_string(KCode.myrank) + "_";
                gmmArray = new cudaGMMWeight::GMM<GMMType, DATA_DIM_GMM, weightType>[1];
                gmmResults.resize(ns);
            }
        }

        if constexpr (MACROCELL_SPECTRA_ENABLE) {
            const int Cx = KCode.col->getMacrocellNx();
            const int Cy = KCode.col->getMacrocellNy();
            const int Cz = KCode.col->getMacrocellNz();
            // Per-species enable mask from input file (default off).
            macrocellSpeciesMask.assign(ns, 0);
            int anySpecies = 0;
            for (int s = 0; s < ns; ++s) {
                if (KCode.col->getVelocitySpectraSpecies(s)) {
                    macrocellSpeciesMask[s] = 1;
                    ++anySpecies;
                }
            }
            if (Cx > 0 && Cy > 0 && Cz > 0 && anySpecies > 0) {
                // Interior cell extents per subdomain (exclude one ghost layer per side).
                const int Nx_int = KCode.grid->getNXC() - 2;
                const int Ny_int = KCode.grid->getNYC() - 2;
                const int Nz_int = KCode.grid->getNZC() - 2;
                // Subdomain origin in the global interior cell index space.
                const int gOffX = KCode.vct->getCoordinates(0) * Nx_int;
                const int gOffY = KCode.vct->getCoordinates(1) * Ny_int;
                const int gOffZ = KCode.vct->getCoordinates(2) * Nz_int;

                macrocellSpectra::MacrocellPartition part;
                if (!part.build(Nx_int, Ny_int, Nz_int, Cx, Cy, Cz,
                                gOffX, gOffY, gOffZ)) {
                    std::cerr << "[macrocellSpectra] invalid macrocell sizes ("
                              << Cx << "," << Cy << "," << Cz
                              << ") vs interior (" << Nx_int << "," << Ny_int
                              << "," << Nz_int << "); feature disabled.\n";
                } else {
                    macrocellHist = std::make_unique<macrocellSpectra::macrocellSpectra2D>(part);
                    macrocellSubdomainDir = MACROCELL_SPECTRA_OUTPUT_DIR
                                          + "subDomain_" + std::to_string(KCode.myrank) + "/";
                    macrocellMyRank        = KCode.myrank;
                    macrocellFieldForPcls  = KCode.fieldForPclCUDAPtr;
                    macrocellGridDevicePtr = KCode.grid3DCUDACUDAPtr;
                    macrocellCellSorters   = KCode.cellSorters;
                    macrocellVct           = KCode.vct;
                    macrocellEnabled       = true;
                    if constexpr (MACROCELL_SPECTRA_OUTPUT) {
                        macrocellHist->writePartitionMetadata(macrocellSubdomainDir,
                                                              macrocellMyRank,
                                                              macrocellVct);
                    }
                }
            }
        }
    }

    void startAnalysis(int cycle);

    int checkAnalysis();

    int waitForAnalysis();



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


int dataAnalysisPipelineImpl::GMMAnalysisSpecies(const int cycle, const int species, const std::string outputPath){

    using weightType = cudaTypeSingle;

    std::future<int> future;

    auto GMMLambda = [=]() mutable {

        using namespace cudaGMMWeight;

        cudaErrChk(cudaSetDevice(deviceOnNode));

        // GMM config

        // set the random number generator to sample velocity from circle of radius max velocity
        std::random_device rd;  // True random seed
        std::mt19937 gen(rd()); // Mersenne Twister PRN
        std::uniform_real_distribution<GMMType>  unif01(0.0, 1.0);
        std::uniform_real_distribution<GMMType> distTheta(0, 2*M_PI);
        std::normal_distribution<GMMType> norm_dist(0.0, 1.0); // mean=0, stddev=1


        const GMMType maxVelocity = (species == 0 || species == 2) ? MAX_VELOCITY_HIST_E : MAX_VELOCITY_HIST_I;
        // it is assumed that DATA_DIM_GMM == 3 and that the velocity range is homogenues in all dimensions
        const GMMType maxVelocityArray[DATA_DIM_GMM] = {maxVelocity,maxVelocity,maxVelocity};
        // right now it is not used (fixed to zero) --> mean is not subtracted to data, but it might be useful for future developments
        const GMMType meanArray[DATA_DIM_GMM] = {0.0,0.0,0.0};

        const GMMType uth = species == 0 || species == 2 ? 0.045 : 0.0126;
        const GMMType vth = species == 0 || species == 2 ? 0.045 : 0.0126;
        const GMMType wth = species == 0 || species == 2 ? 0.045 : 0.0126;
        const GMMType varArray[3] = {uth,vth,wth};
    
        GMMType weightVector[NUM_COMPONENT_GMM];
        GMMType meanVector[NUM_COMPONENT_GMM * DATA_DIM_GMM];
        GMMType coVarianceMatrix[NUM_COMPONENT_GMM * DATA_DIM_GMM * DATA_DIM_GMM ];
        
        // normalize initial parameters if NORMALIZE_DATA_FOR_GMM==true
        GMMType normalization = 1.0; 
        if constexpr(NORMALIZE_DATA_FOR_GMM) normalization = maxVelocity;

        if constexpr (START_WITH_LAST_PARAMETERS_GMM) // start GMM with output GMM parameters from last cycle as initial parameters
        {
            // if first time initialize GMM with the usual fixed parameters
            if(gmmResults[species].size() == 0)
            {                
                for(int j = 0; j < NUM_COMPONENT_GMM; j++){
                    weightVector[j] = 1.0/NUM_COMPONENT_GMM;
                    const GMMType n1 = norm_dist(gen);
                    const GMMType n2 = norm_dist(gen);
                    const GMMType n3 = norm_dist(gen);
                    const GMMType u = unif01(gen);
                    const GMMType llsqrt = sqrt( n1*n1 + n2*n2 + n3*n3); 
                    meanVector[j * DATA_DIM_GMM] = maxVelocity * std::cbrt(u) * n1 / llsqrt / normalization; 
                    meanVector[j * DATA_DIM_GMM + 1] = maxVelocity * std::cbrt(u) * n2 / llsqrt / normalization;
                    meanVector[j * DATA_DIM_GMM + 2] = maxVelocity * std::cbrt(u) * n3 / llsqrt / normalization;
    
                    for (int k = 0; k < DATA_DIM_GMM * DATA_DIM_GMM; k++){
                        const int i = k / DATA_DIM_GMM; // row index
                        const int l = k % DATA_DIM_GMM; // column index
                        coVarianceMatrix[j * DATA_DIM_GMM * DATA_DIM_GMM + k] = varArray[i] * varArray[l]  / ( normalization * normalization);
                    }
                }
            }
            else // Initialize GMM with previous output parameters
            {   
                auto& lastResult = gmmResults[species].back();

                bool reset = false;
                
                // safety checks on components weights and mean since after pruning some components have zero weight
                // check if meanVector is NaN or component weight is too small
                // if meanVector is NaN sample new mean vector
                for(int j = 0; j < NUM_COMPONENT_GMM; j++){ 
                    if( std::isnan(lastResult.mean[j * DATA_DIM_GMM]) || std::isinf(lastResult.mean[j * DATA_DIM_GMM]) || 
                        std::isnan(lastResult.mean[j * DATA_DIM_GMM + 1]) || std::isinf(lastResult.mean[j * DATA_DIM_GMM + 1]) ||
                        std::isnan(lastResult.mean[j * DATA_DIM_GMM + 2]) || std::isinf(lastResult.mean[j * DATA_DIM_GMM + 2]) )
                    {
                        reset = true;
                        const GMMType n1 = norm_dist(gen);
                        const GMMType n2 = norm_dist(gen);
                        const GMMType n3 = norm_dist(gen);
                        const GMMType u = unif01(gen);
                        const GMMType llsqrt = sqrt( n1*n1 + n2*n2 + n3*n3); 
                        meanVector[j * DATA_DIM_GMM] = maxVelocity * std::cbrt(u) * n1 / llsqrt / normalization; 
                        meanVector[j * DATA_DIM_GMM + 1] = maxVelocity * std::cbrt(u) * n2 / llsqrt / normalization;
                        meanVector[j * DATA_DIM_GMM + 2] = maxVelocity * std::cbrt(u) * n3 / llsqrt / normalization;
                    }
                    else{
                        meanVector[j * DATA_DIM_GMM] = lastResult.mean[j * DATA_DIM_GMM] / normalization;
                        meanVector[j * DATA_DIM_GMM + 1] = lastResult.mean[j * DATA_DIM_GMM + 1] / normalization;
                        meanVector[j * DATA_DIM_GMM + 2] =  lastResult.mean[j * DATA_DIM_GMM + 2] / normalization;
                    }

                    if( lastResult.weight[j] < PRUNE_THRESHOLD_GMM*10 ) reset = true;
                }

                // if meanVector is NaN or weigth is too small reset components weights
                // adjust cov if it is too small
                for(int j = 0; j < NUM_COMPONENT_GMM; j++){
                    if(reset){
                        weightVector[j] = 1.0/NUM_COMPONENT_GMM;
                    }
                    else{
                        weightVector[j] = lastResult.weight[j];
                    }

                    for (int k = 0; k < DATA_DIM_GMM * DATA_DIM_GMM; k++){
                        const int i = k / DATA_DIM_GMM; // row index
                        const int l = k % DATA_DIM_GMM; // column index
                        if( i == l){
                            coVarianceMatrix[j * DATA_DIM_GMM * DATA_DIM_GMM + k] = lastResult.coVariance[j * DATA_DIM_GMM * DATA_DIM_GMM + k]  > (10 * EPS_COVMATRIX_GMM * (normalization*normalization)) ? 
                                                    lastResult.coVariance[j * DATA_DIM_GMM * DATA_DIM_GMM + k] : varArray[i] * varArray[l];
                            coVarianceMatrix[j * DATA_DIM_GMM * DATA_DIM_GMM + k] /= (normalization * normalization);
                        }
                        else
                        {
                            coVarianceMatrix[j * DATA_DIM_GMM * DATA_DIM_GMM + k] = varArray[i] * varArray[l]  / ( normalization * normalization);
                        }   
                    }
                }
            }
        }
        else // start GMM with fixed initial parameters at any cycle
        {
            for(int j = 0; j < NUM_COMPONENT_GMM; j++){
                weightVector[j] = 1.0/NUM_COMPONENT_GMM;
                // GMMType radius = maxVelocity * sqrt(unif01(gen));
                // GMMType theta = distTheta(gen);
                const GMMType n1 = norm_dist(gen);
                const GMMType n2 = norm_dist(gen);
                const GMMType n3 = norm_dist(gen);
                const GMMType u = unif01(gen);
                const GMMType llsqrt = sqrt( n1*n1 + n2*n2 + n3*n3); 
                meanVector[j * DATA_DIM_GMM] = maxVelocity * std::cbrt(u) * n1 / llsqrt / normalization; 
                meanVector[j * DATA_DIM_GMM + 1] = maxVelocity * std::cbrt(u) * n2 / llsqrt / normalization;
                meanVector[j * DATA_DIM_GMM + 2] = maxVelocity * std::cbrt(u) * n3 / llsqrt / normalization;

                for (int k = 0; k < DATA_DIM_GMM * DATA_DIM_GMM; k++){
                    const int i = k / DATA_DIM_GMM; // row index
                    const int l = k % DATA_DIM_GMM; // column index
                    coVarianceMatrix[j * DATA_DIM_GMM * DATA_DIM_GMM + k] = varArray[i] * varArray[l]  / ( normalization * normalization);
                }
                // for (int k = 0; k < DATA_DIM_GMM; k++){
                //     meanVector[j * DATA_DIM_GMM + k] =  0.01 * j + 0.01 * k;
                // }

                // for (int k = 0; k < DATA_DIM_GMM * DATA_DIM_GMM; k++){
                //     coVarianceMatrix[j * DATA_DIM_GMM * DATA_DIM_GMM + k] = (k % (DATA_DIM_GMM + 1)) == 0 ? 0.0001 : 0.0;
                // }
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
            (VELOCITY_HISTOGRAM3D_SIZE, velocityHistogram->getHistogramScaleMark(), velocityHistogram->getVelocityHistogramCUDAArray());

        cudaErrChk(cudaHostRegister(&GMMData, sizeof(GMMData), cudaHostRegisterDefault));
        
        // generate exact output file path        
        auto& gmm = gmmArray[0];
        gmm.config(&GMMParam, &GMMData);
        // preprocess the data --> normalize data
        gmm.preProcessDataGMM(meanArray, maxVelocityArray);
        // run GMM
        auto convergStep = gmm.initGMM(); // the exact output file name
        // postporcess data --> normalize data back
        gmm.postProcessDataGMM(maxVelocityArray);
        // move GMM output to host
        gmmResults[species].push_back(gmm.getGMMResult(cycle, convergStep));

        cudaErrChk(cudaHostUnregister(&GMMData));

        return 0;
    };

    future = DAthreadPool->enqueue(GMMLambda); 
    
    if(future.valid() == false){
        throw std::runtime_error("[!]Error: Can not start GMM analysis");
    }

    future.wait();
    
    

    // GMM result output, in the gmmResult
    if constexpr (GMM_OUTPUT) {

        if (!std::filesystem::exists(outputPath)){ 
            std::ofstream output(outputPath);
            if(output.is_open()){
                output.close();
            } else {
                throw std::runtime_error("[!]Error: Can not open output file for velocity GMM species");
            }
        }

        if (std::isnan(gmmResults[species].back().getlogLikelihoodFinal()))
        std::cerr << "[!]GMM: LogLikelihood is NaN!" << " path "<< outputPath << " time step " << gmmResults[species].back().simulationStep << std::endl;

        std::fstream output(outputPath, std::ios::in | std::ios::out | std::ios::ate);
        if(output.is_open()){
            std::ostringstream buffer;

            if(output.tellg() == 0) buffer << "{\n"; 
            else { 
                output.seekp(-2, std::ios_base::end);
                buffer << ",\n";
            }

            buffer << "\"" << std::to_string(cycle) << "\": " << gmmResults[species].back().outputString() << "\n";

            buffer << "}";
            
            output << buffer.str();
            output.close();
        } else {
            throw std::runtime_error("[!]Error: Can not open output file for velocity GMM species");
        }

    }

    return 0;
}

/**
 * @brief analysis function, called by startAnalysis
 * @details procesures in this function should be executed in sequence, the order of the analysis should be defined here
 *          But the procedures can launch other threads to do the analysis
 *          Also this function is a friend function of c_Solver, resources in the c_Slover should be dispatched here
 * @param cycle Simulation cycle being analyzed.
 * @return `0` on success.
 */
int dataAnalysisPipelineImpl::analysisEntre(int cycle){
    // Stream ownership invariant: this function is enqueued on DAthreadPool and
    // mutates streams[i]; the caller (iPIC3D.cpp) must keep streams[i] idle on
    // entry (sortAllSpecies syncs) and must not touch streams[i] again until
    // DA.waitForAnalysis() returns (which only happens once each per-species
    // path here ends with a cudaStreamSynchronize on streams[i]).
    cudaErrChk(cudaSetDevice(deviceOnNode));

    // species by species to save VRAM
    for(int i = 0; i < ns; i++){
        if constexpr (VELOCITY_HISTOGRAM_ENABLE) {
            // Borrow SoA pointers from particleArrayCUDA (zero-copy, no kernel)
            velocitySoACUDA->borrowFrom(pclsArrayHostPtr[i]);

            // histogram
            auto histogramSpeciesOutputPath = HistogramSubDomainOutputPath + "species" + std::to_string(i) + "_";
            velocityHistogram->init(velocitySoACUDA, i, streams[i]);
            if constexpr (HISTOGRAM_OUTPUT)
            velocityHistogram->writeToFile(histogramSpeciesOutputPath, cycle, streams[i]); 
            else cudaErrChk(cudaStreamSynchronize(streams[i]));

            if constexpr (GMM_ENABLE) { // GMM
                auto GMMSpeciesOutputPath = GMMSubDomainOutputPath + "species" + std::to_string(i) + ".json";
                GMMAnalysisSpecies(cycle, i, GMMSpeciesOutputPath);
            }
        }

        if constexpr (MACROCELL_SPECTRA_ENABLE) {
            if (macrocellEnabled && i < (int)macrocellSpeciesMask.size()
                && macrocellSpeciesMask[i]) {
                macrocellHist->reset(streams[i]);
                macrocellHist->launch(pclsArrayHostPtr[i],
                                      macrocellFieldForPcls,
                                      macrocellGridDevicePtr,
                                      macrocellCellSorters[i].getCellStartOffsets(),
                                      macrocellCellSorters[i].getCellCounts(),
                                      i, streams[i]);
                if constexpr (MACROCELL_SPECTRA_OUTPUT) {
                    macrocellHist->writeToFile(macrocellSubdomainDir,
                                               macrocellMyRank, macrocellVct,
                                               i, cycle, streams[i]);
                } else {
                    // Single shared dHist_ is reused across species: drain
                    // streams[i] so the next species' reset/launch on
                    // streams[i+1] cannot race with this species' kernel.
                    cudaErrChk(cudaStreamSynchronize(streams[i]));
                }
            }
        }
    }

    
    return 0;
}


/**
 * @brief start all the analysis registered here
 * @param cycle Simulation cycle to analyze.
 */
void dataAnalysisPipelineImpl::startAnalysis(int cycle){

    if(DATA_ANALYSIS_EVERY_CYCLE == 0 || (cycle % DATA_ANALYSIS_EVERY_CYCLE != 0)){
        analysisFuture = std::future<int>();
    } else {
        if (MPIdata::get_rank() == 0) {
            const bool histOn = VELOCITY_HISTOGRAM_ENABLE;
            const bool gmmOn  = VELOCITY_HISTOGRAM_ENABLE && GMM_ENABLE;
            const bool mcOn   = MACROCELL_SPECTRA_ENABLE && macrocellEnabled;
            if (histOn || gmmOn || mcOn) {
                printf("  [Cycle %d] Data analysis: ON [%s%s%s]\n", cycle,
                       histOn ? "histogram " : "",
                       gmmOn  ? "GMM "       : "",
                       mcOn   ? "macrocellSpectra" : "");
            }
        }
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
 * @return `0` after all queued analysis work has completed.
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
 * @param myrank MPI rank used to create rank-local paths.
 * @param ns Number of species included in the analysis output.
 * @param vct MPI topology used for subdomain-coordinate metadata.
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
        const auto histogramSubDomainOutputPath = HISTOGRAM_OUTPUT_DIR;

        if(myrank == 0 && 0 != checkOutputFolder(histogramSubDomainOutputPath)){
            throw std::runtime_error("[!]Error: Can not create output folder for velocity histogram");
        }

        MPI_Barrier(MPIdata::get_PicGlobalComm());

        writeVctMapping(histogramSubDomainOutputPath + "vctMapping_subDomain" + std::to_string(myrank) + ".txt");
    }

    if constexpr (GMM_ENABLE && GMM_OUTPUT) {
        const auto GMMSubDomainOutputPath = GMM_OUTPUT_DIR;

        if(myrank == 0 && 0 != checkOutputFolder(GMMSubDomainOutputPath)){
            throw std::runtime_error("[!]Error: Can not create output folder for velocity GMM");
        }

        MPI_Barrier(MPIdata::get_PicGlobalComm());

        writeVctMapping(GMMSubDomainOutputPath + "vctMapping_subDomain" + std::to_string(myrank) + ".txt");
    }

    if constexpr (MACROCELL_SPECTRA_ENABLE && MACROCELL_SPECTRA_OUTPUT) {
        const auto macrocellRoot = MACROCELL_SPECTRA_OUTPUT_DIR;

        if (myrank == 0 && 0 != checkOutputFolder(macrocellRoot)) {
            throw std::runtime_error("[!]Error: Can not create output folder for macrocell spectra");
        }

        MPI_Barrier(MPIdata::get_PicGlobalComm());

        // Per-subdomain folder.
        const auto subDir = macrocellRoot + "subDomain_" + std::to_string(myrank) + "/";
        if (0 != checkOutputFolder(subDir)) {
            throw std::runtime_error("[!]Error: Can not create subdomain folder for macrocell spectra");
        }
        writeVctMapping(subDir + "vctMapping.txt");
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



dataAnalysisPipeline::~dataAnalysisPipeline() {
    
}
    
} // namespace dataAnalysis






