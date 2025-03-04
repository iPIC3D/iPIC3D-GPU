#ifndef _CUDA_GMM_H_
#define _CUDA_GMM_H_

#include "cudaTypeDef.cuh"
#include "cudaGMMUtility.cuh"
#include "cudaGMMkernel.cuh"
#include "cudaReduction.cuh"

#include <stdio.h>
#include <string>
#include <fstream>
#include <iostream>


namespace cudaGMMWeight
{



template <typename T, int dataDim, typename U = int>
class GMM{

private:
    cudaStream_t GMMStream;

    GMMDataMultiDim<T, dataDim, U>* dataHostPtr; // ogject on host, data pointers pointing to the data on device
    GMMDataMultiDim<T, dataDim, U>* dataDevicePtr = nullptr; // object on device, data pointers pointing to the data on device

    GMMParam_t<T>* paramHostPtr; // object on host, the parameters for the GMM
    // GMMParam_t<T>* paramDevicePtr; // object on device, the parameters for the GMM

    int sizeNumComponents = 0; // the size of the buffers below, for space checking
    int sizeNumData = 0; // the size of the buffers below, for space checking, numComponents * numData

    // the arrays on device
    T* weightCUDA;                  // numComponents, it will be log(weight) during the iteration
    T* meanCUDA;                    // numComponents * dataDim
    T* coVarianceCUDA;              // numComponents * dataDim * dataDim
    T* coVarianceDecomposedCUDA;    // numComponents * dataDim * dataDim, only the lower triangle
    T* normalizerCUDA;              // numComponents
    T* PosteriorCUDA;               // numComponents, for each component, the sum of the posterior of all data points

    T* posteriorCUDA;               // numComponents * numData
    T* tempArrayCUDA;               // numComponents * numData * dataDim * dataDim, for temporary storage

    int reductionTempArraySize = 0;
    T* reductionTempArrayCUDA;      // dynamic size, for reduction, between reduction and redunctionWarp


    // the arrays on host, results and init values
    T* weight;                  // numComponents
    T* mean;                    // numComponents * dataDim
    T* coVariance;              // numComponents * dataDim * dataDim
    T* coVarianceDecomposed;    // numComponents * dataDim * dataDim, only the lower triangle
    T* normalizer;              // numComponents

    T* logResult = nullptr;

    // runtime variables
    T* logLikelihoodCUDA = nullptr;
    T logLikelihood = - INFINITY;
    T logLikelihoodOld = - INFINITY;

    

public:

    //allocate or check if the arrays are allocated adequately, then replace the param
    __host__ int config(GMMParam_t<T>* GMMParam, GMMDataMultiDim<T, dataDim, U>* data){
        auto oldSizeComponents = sizeNumComponents;
        auto oldSizeData = sizeNumData;

        // check if the arrays are allocated adequately
        if(GMMParam->numComponents > oldSizeComponents){
            // deallocate the old arrays
            if(oldSizeComponents > 0){
                // device
                cudaErrChk(cudaFree(weightCUDA));
                cudaErrChk(cudaFree(meanCUDA));
                cudaErrChk(cudaFree(coVarianceCUDA));
                cudaErrChk(cudaFree(coVarianceDecomposedCUDA));
                cudaErrChk(cudaFree(normalizerCUDA));
                cudaErrChk(cudaFree(PosteriorCUDA));

                // host
                cudaErrChk(cudaFreeHost(weight));
                cudaErrChk(cudaFreeHost(mean));
                cudaErrChk(cudaFreeHost(coVariance));
                cudaErrChk(cudaFreeHost(coVarianceDecomposed));
                cudaErrChk(cudaFreeHost(normalizer));

            }

            sizeNumComponents = GMMParam->numComponents; // the new size
            auto& numCompo = sizeNumComponents;

            // allocate the new arrays
            cudaErrChk(cudaMalloc(&weightCUDA, sizeof(T)*numCompo));
            cudaErrChk(cudaMalloc(&meanCUDA, sizeof(T)*numCompo*dataDim));
            cudaErrChk(cudaMalloc(&coVarianceCUDA, sizeof(T)*numCompo*dataDim*dataDim));
            cudaErrChk(cudaMalloc(&coVarianceDecomposedCUDA, sizeof(T)*numCompo*dataDim*dataDim));
            cudaErrChk(cudaMalloc(&normalizerCUDA, sizeof(T)*numCompo));
            cudaErrChk(cudaMalloc(&PosteriorCUDA, sizeof(T)*numCompo));

            cudaErrChk(cudaMallocHost(&weight, sizeof(T)*numCompo));
            cudaErrChk(cudaMallocHost(&mean, sizeof(T)*numCompo*dataDim));
            cudaErrChk(cudaMallocHost(&coVariance, sizeof(T)*numCompo*dataDim*dataDim));
            cudaErrChk(cudaMallocHost(&coVarianceDecomposed, sizeof(T)*numCompo*dataDim*dataDim));
            cudaErrChk(cudaMallocHost(&normalizer, sizeof(T)*numCompo));
        }

        // check the numPoint related arrays
        if(GMMParam->numComponents * data->getNumData() > oldSizeData){
            // deallocate the old arrays
            if(oldSizeData > 0){
                // device
                cudaErrChk(cudaFree(posteriorCUDA));
                cudaErrChk(cudaFree(tempArrayCUDA));
            }

            sizeNumData = GMMParam->numComponents * data->getNumData(); // the new size

            // allocate the new arrays
            cudaErrChk(cudaMalloc(&posteriorCUDA, sizeof(T)*sizeNumData));
            cudaErrChk(cudaMalloc(&tempArrayCUDA, sizeof(T)*sizeNumData*dataDim*dataDim));
        }


        // load the init values, if needed
        if(GMMParam->weightInit != nullptr && GMMParam->meanInit != nullptr && GMMParam->coVarianceInit != nullptr){
            memcpy(weight, GMMParam->weightInit, sizeof(T)*GMMParam->numComponents);
            memcpy(mean, GMMParam->meanInit, sizeof(T)*GMMParam->numComponents*dataDim);
            memcpy(coVariance, GMMParam->coVarianceInit, sizeof(T)*GMMParam->numComponents*dataDim*dataDim);
        }else{ // init with internal initiator
            // TODO
        }

        // precompute log of the mean vector
        for(int i = 0; i < GMMParam->numComponents; i++){
            weight[i] = log(weight[i]);
        }

        // copy to device
        cudaErrChk(cudaMemcpyAsync(weightCUDA, weight, sizeof(T)*GMMParam->numComponents, cudaMemcpyHostToDevice, GMMStream));
        cudaErrChk(cudaMemcpyAsync(meanCUDA, mean, sizeof(T)*GMMParam->numComponents*dataDim, cudaMemcpyHostToDevice, GMMStream));
        cudaErrChk(cudaMemcpyAsync(coVarianceCUDA, coVariance, sizeof(T)*GMMParam->numComponents*dataDim*dataDim, cudaMemcpyHostToDevice, GMMStream));
        cudaErrChk(cudaMemcpyAsync(coVarianceDecomposedCUDA, coVariance, sizeof(T)*GMMParam->numComponents*dataDim*dataDim,cudaMemcpyHostToDevice, GMMStream));
        // cudaErrChk(cudaMemcpy(normalizerCUDA, normalizer, sizeof(T)*GMMParam->numComponents, cudaMemcpyHostToDevice));

        // replace the param
        paramHostPtr = GMMParam;
        dataHostPtr = data;

        cudaErrChk(cudaMemcpyAsync(dataDevicePtr, dataHostPtr, sizeof(GMMDataMultiDim<T, dataDim, U>), cudaMemcpyDefault, GMMStream));

        { // reset the value for reuse
            logLikelihood = - INFINITY;
            logLikelihoodOld = - INFINITY;
        }
        return 0; // some cuda async operation are not finished yet, but we are using the same stream
    }

    __host__ GMM(){
        cudaErrChk(cudaStreamCreate(&GMMStream));

        cudaErrChk(cudaMalloc(&dataDevicePtr, sizeof(GMMDataMultiDim<T, dataDim, U>)));

        reductionTempArraySize = 1024;
        cudaErrChk(cudaMalloc(&reductionTempArrayCUDA, sizeof(T)*reductionTempArraySize));

        cudaErrChk(cudaMalloc(&logLikelihoodCUDA, sizeof(T)));
        cudaErrChk(cudaMallocHost(&logResult, sizeof(T)));
    }

    __host__ int initGMM(){

        // do the GMR
        int step = 0;
        if constexpr(NORMALIZE_DATA_FOR_GMM)
        {
            if(paramHostPtr->maxIteration>0)
            {
                normalizePoints();
            }
        }
        while(step < paramHostPtr->maxIteration){
            
            // E
            calcPxAtMeanAndCoVariance();
            calcLogLikelihoodPxAndposterior();
            logLikelihood = sumLogLikelihood();

            // compare the log likelihood increament with the threshold, if the increament is smaller than the threshold, or the log likelihood is smaller than the previous one, output the GMM
            if( std::isnan(logLikelihood) || fabs(logLikelihood - logLikelihoodOld) < paramHostPtr->threshold || logLikelihood < logLikelihoodOld){
                // std::cout << "Converged at step " << step << std::endl;
                break;
            }
            // std::cout << "Step " << step << " log likelihood: " << logLikelihood << std::endl;
            logLikelihoodOld = logLikelihood;

            // M
            calcPosterior();
            updateMean();
            updateWeight();
            updateCoVarianceAndDecomposition();

            step++;
        }

        if constexpr(NORMALIZE_DATA_FOR_GMM)
        {
            if(step>0)
            {
                normalizeDataBack();
            }
        }
        return step;

    }

    __host__ GMMResult<T, dataDim> getGMMResult(int simuStep, int convergeStep){
        GMMResult<T, dataDim> result(simuStep, paramHostPtr->numComponents);

        result.convergeStep = convergeStep;

        if (std::isnan(logLikelihood))
        std::cerr << "[!]GMM: LogLikelihood is NaN!" << " GMM step: "<< convergeStep << std::endl;

        result.logLikelihoodFinal = logLikelihood;

        // copy from device array
        cudaErrChk(cudaMemcpyAsync(result.weight.get(), weightCUDA, sizeof(T)*paramHostPtr->numComponents, cudaMemcpyDefault, GMMStream));
        cudaErrChk(cudaMemcpyAsync(result.mean.get(), meanCUDA, sizeof(T)*paramHostPtr->numComponents*dataDim, cudaMemcpyDefault, GMMStream));
        cudaErrChk(cudaMemcpyAsync(result.coVariance.get(), coVarianceCUDA, sizeof(T)*paramHostPtr->numComponents*dataDim*dataDim, cudaMemcpyDefault, GMMStream));
        
        cudaErrChk(cudaStreamSynchronize(GMMStream));

        for(int i = 0; i < paramHostPtr->numComponents; i++){
            result.weight[i] = exp(result.weight[i]);
        }

        return result;
    }

    __host__ int outputGMM(int convergeStep, std::string outputPath){
        auto numComponents = paramHostPtr->numComponents;
        auto numData = dataHostPtr->getNumData();
        // copy the results to host and post process
        cudaErrChk(cudaMemcpyAsync(weight, weightCUDA, sizeof(T)*numComponents, cudaMemcpyDefault, GMMStream));
        cudaErrChk(cudaMemcpyAsync(mean, meanCUDA, sizeof(T)*numComponents*dataDim, cudaMemcpyDefault, GMMStream));
        cudaErrChk(cudaMemcpyAsync(coVariance, coVarianceCUDA, sizeof(T)*numComponents*dataDim*dataDim, cudaMemcpyDefault, GMMStream));
        cudaErrChk(cudaMemcpyAsync(coVarianceDecomposed, coVarianceDecomposedCUDA, sizeof(T)*numComponents*dataDim*dataDim, cudaMemcpyDefault, GMMStream));
        cudaErrChk(cudaMemcpyAsync(normalizer, normalizerCUDA, sizeof(T)*numComponents, cudaMemcpyDefault, GMMStream));

        cudaErrChk(cudaStreamSynchronize(GMMStream));

        for(int i = 0; i < numComponents; i++){
            weight[i] = exp(weight[i]);
        }

        {  // output the GMM to json
            std::ofstream file(outputPath);
            if (!file.is_open()) {
                std::cerr << "[!]Could not open file " << outputPath << std::endl;
                return -1;
            }

            if (std::isnan(logLikelihood))
                std::cerr << "[!]LogLikelihood " <<outputPath<< " is NaN!" << " GMM step: "<< convergeStep << std::endl;


            file << "{\n";
            file << "\"convergeStep\": " << convergeStep << ",\n";
            file << "\"logLikelyHood\": " << logLikelihood << ",\n";
            file << "\"model\": {\n";
            file << "\"dataDim\": " << dataDim << ",\n";
            file << "\"numComponent\": " << numComponents << ",\n";
            file << "\"components\": [\n";

            for (size_t k = 0; k < numComponents; ++k) {
                file << "{\n";
                file << "\"weight\": " << weight[k] << ",\n";

                file << "\"mean\": [ ";
                for (size_t dim = 0; dim < dataDim; ++dim) {
                    file << mean[k * dataDim + dim];
                    if (dim + 1 < dataDim) {
                        file << ", ";
                    }
                }
                file << " ],\n";

                file << "\"coVariance\": [ ";
                for (size_t dim = 0; dim < dataDim * dataDim; ++dim) {
                    file << coVariance[k * dataDim * dataDim + dim];
                    if (dim + 1 < dataDim * dataDim) {
                        file << ", ";
                    }
                }
                file << " ]\n";
                file << "}";

                if (k + 1 < numComponents) {
                    file << ", ";
                }
            }

            file << "]\n";
            file << "}\n";
            file << "}\n";
        }

        return 0;

    }

    __host__ ~GMM(){
        // created in constructor
        cudaErrChk(cudaStreamDestroy(GMMStream));

        cudaErrChk(cudaFree(dataDevicePtr));
        cudaErrChk(cudaFree(reductionTempArrayCUDA));
        cudaErrChk(cudaFree(logLikelihoodCUDA));
        cudaErrChk(cudaFreeHost(logResult));

        // allocated in config

        // deallocate the old arrays
        if(sizeNumComponents > 0){
            // device
            cudaErrChk(cudaFree(weightCUDA));
            cudaErrChk(cudaFree(meanCUDA));
            cudaErrChk(cudaFree(coVarianceCUDA));
            cudaErrChk(cudaFree(coVarianceDecomposedCUDA));
            cudaErrChk(cudaFree(normalizerCUDA));
            cudaErrChk(cudaFree(PosteriorCUDA));

            if(sizeNumData > 0){
                // device
                cudaErrChk(cudaFree(posteriorCUDA));
                cudaErrChk(cudaFree(tempArrayCUDA));
            }

            // host
            cudaErrChk(cudaFreeHost(weight));
            cudaErrChk(cudaFreeHost(mean));
            cudaErrChk(cudaFreeHost(coVariance));
            cudaErrChk(cudaFreeHost(coVarianceDecomposed));
            cudaErrChk(cudaFreeHost(normalizer));

        }
    }

private:

    int reduceBlockNum(int dataSize, int blockSize){
        constexpr int elementPerThread = 8; // 8 elements per thread, as the input is 10000
        if(dataSize < elementPerThread)dataSize = elementPerThread;
        auto blockNum = getGridSize(dataSize / elementPerThread, blockSize); 
        blockNum = blockNum > 1024 ? 1024 : blockNum;

        if(reductionTempArraySize < blockNum){
            cudaErrChk(cudaFree(reductionTempArrayCUDA));
            cudaErrChk(cudaMalloc(&reductionTempArrayCUDA, sizeof(T)*blockNum));
            reductionTempArraySize = blockNum;
        }

        return blockNum;
    }


    void normalizePoints(){
        
        // normalize data such that velocities are in range -1;1
        // launch kernel
        cudaGMMWeightKernel::normalizePointsKernel<T,dataDim,U,false><<<getGridSize(dataHostPtr->getNumData(), 256), 256, 0, GMMStream>>>
                                (dataDevicePtr, dataHostPtr->maxVelocityArray);
    }

    void normalizeDataBack(){
        
        // normalize data back to the original data range
        // launch kernel
        cudaGMMWeightKernel::normalizePointsKernel<T,dataDim,U,true><<<getGridSize(dataHostPtr->getNumData(), 256), 256, 0, GMMStream>>>
                                (dataDevicePtr, dataHostPtr->maxVelocityArray );

        cudaGMMWeightKernel::normalizeMeanAndCovBack<T, dataDim><<<1, paramHostPtr->numComponents, 0, GMMStream>>>
                            (meanCUDA, coVarianceCUDA, paramHostPtr->numComponents, dataHostPtr->maxVelocityArray);
    }
    
    void calcPxAtMeanAndCoVariance(){

        // launch kernel
        cudaGMMWeightKernel::calcLogLikelihoodForPointsKernel<<<getGridSize(dataHostPtr->getNumData(), 256), 256, 0, GMMStream>>>
                                (dataDevicePtr, meanCUDA, coVarianceDecomposedCUDA, posteriorCUDA, paramHostPtr->numComponents);
        
        // posterior_nk holds log p(x_i|mean,coVariance) for each data point i and each component k, temporary storage
    }




    void calcLogLikelihoodPxAndposterior(){
        // launch kernel, the first posterior_nk is the log p(x_i|mean,coVariance)
        cudaGMMWeightKernel::calcLogLikelihoodPxAndposteriorKernel<<<getGridSize(dataHostPtr->getNumData(), 256), 256, 0, GMMStream>>>
                                (dataDevicePtr, weightCUDA, posteriorCUDA, tempArrayCUDA, posteriorCUDA, paramHostPtr->numComponents);
        
        // now the posterior_nk is the log posterior_nk
        // the tempArrayCUDA is the log Px for each data point 

    }

    T sumLogLikelihood(){ // its sync now, but can be async with the M step
        // sum the log likelihood Px with reduction, then return the sum
        constexpr int blockSize = 256;
        auto blockNum = reduceBlockNum(dataHostPtr->getNumData(), blockSize);

        cudaReduction::reduceSumPreProcess<T, blockSize, cudaReduction::PreProcessType::none, void, U, true> // weighted sum
            <<<blockNum, blockSize, blockSize*sizeof(T), GMMStream>>>
            (tempArrayCUDA, reductionTempArrayCUDA, dataHostPtr->getNumData(), nullptr, dataHostPtr->getWeight());
        cudaReduction::reduceSumWarp<T><<<1, WARP_SIZE, 0, GMMStream>>>(reductionTempArrayCUDA, logLikelihoodCUDA, blockNum);

        cudaErrChk(cudaMemcpyAsync(logResult, logLikelihoodCUDA, sizeof(T), cudaMemcpyDefault, GMMStream));

        cudaErrChk(cudaStreamSynchronize(GMMStream)); // The M step can be preLaunched for most of the time

        return *logResult;

    }

    void calcPosterior(){ // the Big Gamma, it can be direct sum then log

        constexpr int blockSize = 256;
        auto blockNum = reduceBlockNum(dataHostPtr->getNumData(), blockSize);

        auto maxValueArray = tempArrayCUDA; // maxValues of posterior_nk for each component
        for(int component = 0; component < paramHostPtr->numComponents; component++){
            // get the max value of the posterior_nk(little gamma), with reduction
            auto posteriorComponent = posteriorCUDA + component*dataHostPtr->getNumData();
            
            cudaReduction::reduceMax<T, blockSize>
                <<<blockNum, blockSize, blockSize*sizeof(T), GMMStream>>>
                (posteriorComponent, reductionTempArrayCUDA, dataHostPtr->getNumData());

            cudaReduction::reduceMaxWarp<T>
                <<<1, WARP_SIZE, 0, GMMStream>>>
                (reductionTempArrayCUDA, maxValueArray + component, blockNum);

            // reduction sum with pre-process and post-process to get the log Posterior_k 
            cudaReduction::reduceSumPreProcess<T, blockSize, cudaReduction::PreProcessType::minusConstThenEXP, T, U, true>
                <<<blockNum, blockSize, blockSize*sizeof(T), GMMStream>>>
                (posteriorComponent, reductionTempArrayCUDA, dataHostPtr->getNumData(), maxValueArray + component, dataHostPtr->getWeight());
            
            cudaReduction::reduceSumWarpPostProcess<T, cudaReduction::PostProcessType::logAdd, T>
                <<<1, WARP_SIZE, 0, GMMStream>>>
                (reductionTempArrayCUDA, PosteriorCUDA + component, blockNum, maxValueArray + component);

        }

        // now(after kernel execution) we have the log Posterior_k for each component

    }

    void updateMean(){
        constexpr int blockSize = 256;
        auto blockNum = reduceBlockNum(dataHostPtr->getNumData(), blockSize);

        // for each component, for each dimension
        for(int component = 0; component < paramHostPtr->numComponents; component++){
            for(int dim = 0; dim < dataHostPtr->getDim(); dim++){
                // calc x_i * posterior_nk, could be merged with the reduction sum
                // sum the x_i * posterior_nk with reduction
                cudaReduction::reduceSumPreProcess<T, blockSize, cudaReduction::PreProcessType::multiplyEXP, T, U, true>
                    <<<blockNum, blockSize, blockSize*sizeof(T), GMMStream>>>
                    (dataHostPtr->getDim(dim), reductionTempArrayCUDA, dataHostPtr->getNumData(), posteriorCUDA + component*dataHostPtr->getNumData(), dataHostPtr->getWeight());

                // divide by the Posterior_k, can be post processed with the reduction sum warp
                cudaReduction::reduceSumWarpPostProcess<T, cudaReduction::PostProcessType::divideEXP, T>
                    <<<1, WARP_SIZE, 0, GMMStream>>>
                    (reductionTempArrayCUDA, meanCUDA + component*dataHostPtr->getDim() + dim, blockNum, PosteriorCUDA + component);
            }
        }

    }

    void updateWeight(){
        // calc the new weight for components
        cudaGMMWeightKernel::updateWeightKernel
            <<<1, paramHostPtr->numComponents, paramHostPtr->numComponents * sizeof(T), GMMStream>>>
            (weightCUDA, PosteriorCUDA, paramHostPtr->numComponents);
    }

    void updateCoVarianceAndDecomposition(){
        constexpr int blockSize = 256;
        auto blockNum = reduceBlockNum(dataHostPtr->getNumData(), blockSize);

        // calc the new coVariance for components
        cudaGMMWeightKernel::updateCoVarianceKernel
            <<<getGridSize(dataHostPtr->getNumData(), 256), 256, 0, GMMStream>>>
            (dataDevicePtr, posteriorCUDA, PosteriorCUDA, meanCUDA, tempArrayCUDA, paramHostPtr->numComponents);

        // sum the coVariance with reduction, then divide by the Posterior_k
        for(int component = 0; component < paramHostPtr->numComponents; component++){
            auto coVarianceComponent = tempArrayCUDA + component*dataHostPtr->getNumData()*dataDim*dataDim;
            for(int element = 0; element < dataDim * dataDim; element++){
                cudaReduction::reduceSumPreProcess<T, blockSize, cudaReduction::PreProcessType::none, void, U, true>
                    <<<blockNum, blockSize, blockSize*sizeof(T), GMMStream>>>
                    (coVarianceComponent + element*dataHostPtr->getNumData(), reductionTempArrayCUDA, dataHostPtr->getNumData(), nullptr, dataHostPtr->getWeight());
                    
                cudaReduction::reduceSumWarpPostProcess<T, cudaReduction::PostProcessType::divideEXP, T>
                    <<<1, WARP_SIZE, 0, GMMStream>>>
                    (reductionTempArrayCUDA, coVarianceCUDA + component*dataDim*dataDim + element, blockNum, PosteriorCUDA + component);
            }
        }

        // check cov-matrix and adjust main diagonal to ensure determinate>0 and cholesky decomposition
        if constexpr(CHECK_COVMATRIX_GMM)
        {
            cudaGMMWeightKernel::checkAdjustCoVarianceKernel<T, dataDim><<<1, paramHostPtr->numComponents, 0, GMMStream>>>(coVarianceCUDA, paramHostPtr->numComponents);
        }

        // decompose the coVariance with cholesky decomposition -> A = LL^T
        cudaGMMWeightKernel::decomposeCoVarianceKernel<T, dataDim>
            <<<1, paramHostPtr->numComponents, 0, GMMStream>>>
            (coVarianceCUDA, coVarianceDecomposedCUDA, normalizerCUDA, paramHostPtr->numComponents);
    }

};

}

#endif // _CUDA_GMM_H_



