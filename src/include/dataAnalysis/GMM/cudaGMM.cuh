#ifndef CUDA_GMM_CUH
#define CUDA_GMM_CUH

#include "cudaGMMUtility.cuh"
#include "cudaGMMkernel.cuh"
#include "cudaReduction.cuh"
#include "cudaTypeDef.cuh"

#include "dataAnalysisConfig.cuh"

#include <algorithm> // required for std::copy
#include <fstream>
#include <iostream>
#include <stdio.h>
#include <string>

namespace cudaGMMWeight {

using namespace DAConfig;

template <typename T, int dataDim, typename U = int> class GMM {

private:
  cudaStream_t GMMStream;

  GMMDataMultiDim<T, dataDim, U>* dataHostPtr; // ogject on host, data pointers
                                               // pointing to the data on device
  GMMDataMultiDim<T, dataDim, U>* dataDevicePtr =
      nullptr; // object on device, data pointers pointing to the data on device

  GMMParam_t<T>* paramHostPtr; // object on host, the parameters for the GMM
  // GMMParam_t<T>* paramDevicePtr; // object on device, the parameters for the
  // GMM

  int sizeNumComponents =
      0;                   // the size of the buffers below, for space checking
  int sizeNumData = 0;     // the size of the buffers below, for space checking,
                           // numComponents * numData
  int numActiveComponents; // number of active GMM components

  // the arrays on device
  bool* flagActiveComponentsCUDA; // array with flags true/false to identify
                                  // active components, last element is a flag
                                  // that specify if at that cycle one component
                                  // has been pruned, numComponents + 1
  bool* flagResetMeanCUDA;
  T* meanDataInitCUDA; // initial mean of the data, used to normalize data -
                       // size dataDim --> right now is unused (=0) but it might
                       // be useful in the future, dataDim
  T* weightCUDA; // numComponents, it will be log(weight) during the iteration
  T* meanCUDA;   // numComponents * dataDim
  T* coVarianceCUDA;           // numComponents * dataDim * dataDim
  T* coVarianceDecomposedCUDA; // numComponents * dataDim * dataDim, only the
                               // lower triangle
  T* normalizerCUDA;           // numComponents
  T* PosteriorCUDA; // numComponents, for each component, the sum of the
                    // posterior of all data points

  T* posteriorCUDA; // numComponents * numData
  T* tempArrayCUDA; // numComponents * numData * dataDim * dataDim, for
                    // temporary storage

  int reductionTempArraySize = 0;
  T* reductionTempArrayCUDA; // dynamic size, for reduction, between reduction
                             // and redunctionWarp

  // the arrays on host, results and init values
  bool* flagActiveComponents; // array with flags true/false to identify active
                              // components, numComponents
  bool* flagResetMean;
  T* meanDataInit;         // dataDim
  T* weight;               // numComponents
  T* mean;                 // numComponents * dataDim
  T* coVariance;           // numComponents * dataDim * dataDim
  T* coVarianceDecomposed; // numComponents * dataDim * dataDim, only the lower
                           // triangle
  T* normalizer;           // numComponents

  T* logResult = nullptr;

  // runtime variables
  T* logLikelihoodCUDA = nullptr;
  T logLikelihood = -INFINITY;
  T logLikelihoodOld = -INFINITY;

public:
  // allocate or check if the arrays are allocated adequately, then replace the
  // param
  __host__ int config(GMMParam_t<T>* GMMParam,
                      GMMDataMultiDim<T, dataDim, U>* data) {
    auto oldSizeComponents = sizeNumComponents;
    auto oldSizeData = sizeNumData;

    // check if the arrays are allocated adequately
    if (GMMParam->numComponents > oldSizeComponents) {
      // deallocate the old arrays
      if (oldSizeComponents > 0) {
        // device
        cudaErrChk(cudaFree(flagActiveComponentsCUDA));
        cudaErrChk(cudaFree(flagResetMeanCUDA));
        cudaErrChk(cudaFree(weightCUDA));
        cudaErrChk(cudaFree(meanCUDA));
        cudaErrChk(cudaFree(meanDataInitCUDA));
        cudaErrChk(cudaFree(coVarianceCUDA));
        cudaErrChk(cudaFree(coVarianceDecomposedCUDA));
        cudaErrChk(cudaFree(normalizerCUDA));
        cudaErrChk(cudaFree(PosteriorCUDA));

        // host
        cudaErrChk(cudaFreeHost(flagActiveComponents));
        cudaErrChk(cudaFreeHost(flagResetMean));
        cudaErrChk(cudaFreeHost(meanDataInit));
        cudaErrChk(cudaFreeHost(weight));
        cudaErrChk(cudaFreeHost(mean));
        cudaErrChk(cudaFreeHost(coVariance));
        cudaErrChk(cudaFreeHost(coVarianceDecomposed));
        cudaErrChk(cudaFreeHost(normalizer));
      }

      sizeNumComponents = GMMParam->numComponents; // the new size
      auto& numCompo = sizeNumComponents;

      // allocate the new arrays
      cudaErrChk(
          cudaMalloc(&flagActiveComponentsCUDA, sizeof(bool) * (numCompo + 1)));
      cudaErrChk(cudaMalloc(&flagResetMeanCUDA, sizeof(bool)));
      cudaErrChk(cudaMalloc(&weightCUDA, sizeof(T) * numCompo));
      cudaErrChk(cudaMalloc(&meanCUDA, sizeof(T) * numCompo * dataDim));
      cudaErrChk(cudaMalloc(&meanDataInitCUDA, sizeof(T) * dataDim));
      cudaErrChk(cudaMalloc(&coVarianceCUDA,
                            sizeof(T) * numCompo * dataDim * dataDim));
      cudaErrChk(cudaMalloc(&coVarianceDecomposedCUDA,
                            sizeof(T) * numCompo * dataDim * dataDim));
      cudaErrChk(cudaMalloc(&normalizerCUDA, sizeof(T) * numCompo));
      cudaErrChk(cudaMalloc(&PosteriorCUDA, sizeof(T) * numCompo));

      cudaErrChk(
          cudaMallocHost(&flagActiveComponents, sizeof(bool) * (numCompo + 1)));
      cudaErrChk(cudaMallocHost(&flagResetMean, sizeof(bool)));
      cudaErrChk(cudaMallocHost(&meanDataInit, sizeof(T) * dataDim));
      cudaErrChk(cudaMallocHost(&weight, sizeof(T) * numCompo));
      cudaErrChk(cudaMallocHost(&mean, sizeof(T) * numCompo * dataDim));
      cudaErrChk(cudaMallocHost(&coVariance,
                                sizeof(T) * numCompo * dataDim * dataDim));
      cudaErrChk(cudaMallocHost(&coVarianceDecomposed,
                                sizeof(T) * numCompo * dataDim * dataDim));
      cudaErrChk(cudaMallocHost(&normalizer, sizeof(T) * numCompo));
    }

    // check the numPoint related arrays
    if (GMMParam->numComponents * data->getNumData() > oldSizeData) {
      // deallocate the old arrays
      if (oldSizeData > 0) {
        // device
        cudaErrChk(cudaFree(posteriorCUDA));
        cudaErrChk(cudaFree(tempArrayCUDA));
      }

      sizeNumData =
          GMMParam->numComponents * data->getNumData(); // the new size

      // allocate the new arrays
      cudaErrChk(cudaMalloc(&posteriorCUDA, sizeof(T) * sizeNumData));
      cudaErrChk(cudaMalloc(&tempArrayCUDA,
                            sizeof(T) * sizeNumData * dataDim * dataDim));
    }

    // load the init values, if needed
    if (GMMParam->weightInit != nullptr && GMMParam->meanInit != nullptr &&
        GMMParam->coVarianceInit != nullptr) {
      memcpy(weight, GMMParam->weightInit, sizeof(T) * GMMParam->numComponents);
      memcpy(mean, GMMParam->meanInit,
             sizeof(T) * GMMParam->numComponents * dataDim);
      memcpy(coVariance, GMMParam->coVarianceInit,
             sizeof(T) * GMMParam->numComponents * dataDim * dataDim);
    } else { // init with internal initiator
             // TODO
    }

    // precompute log of the mean vector
    // set all component to active
    for (int i = 0; i < GMMParam->numComponents; i++) {
      weight[i] = log(weight[i]);
      flagActiveComponents[i] = true;
    }
    flagActiveComponents[GMMParam->numComponents] = false;
    numActiveComponents = GMMParam->numComponents;
    *flagResetMean = false;

    // copy to device
    cudaErrChk(cudaMemcpyAsync(flagActiveComponentsCUDA, flagActiveComponents,
                               sizeof(bool) * (GMMParam->numComponents + 1),
                               cudaMemcpyHostToDevice, GMMStream));
    cudaErrChk(cudaMemcpyAsync(flagResetMeanCUDA, flagResetMean, sizeof(bool),
                               cudaMemcpyHostToDevice, GMMStream));
    cudaErrChk(cudaMemcpyAsync(weightCUDA, weight,
                               sizeof(T) * GMMParam->numComponents,
                               cudaMemcpyHostToDevice, GMMStream));
    cudaErrChk(cudaMemcpyAsync(meanCUDA, mean,
                               sizeof(T) * GMMParam->numComponents * dataDim,
                               cudaMemcpyHostToDevice, GMMStream));
    cudaErrChk(
        cudaMemcpyAsync(coVarianceCUDA, coVariance,
                        sizeof(T) * GMMParam->numComponents * dataDim * dataDim,
                        cudaMemcpyHostToDevice, GMMStream));
    cudaErrChk(
        cudaMemcpyAsync(coVarianceDecomposedCUDA, coVariance,
                        sizeof(T) * GMMParam->numComponents * dataDim * dataDim,
                        cudaMemcpyHostToDevice, GMMStream));
    // cudaErrChk(cudaMemcpy(normalizerCUDA, normalizer,
    // sizeof(T)*GMMParam->numComponents, cudaMemcpyHostToDevice));

    // replace the param
    paramHostPtr = GMMParam;
    dataHostPtr = data;
    cudaErrChk(cudaMemcpyAsync(dataDevicePtr, dataHostPtr,
                               sizeof(GMMDataMultiDim<T, dataDim, U>),
                               cudaMemcpyDefault, GMMStream));

    cudaErrChk(cudaStreamSynchronize(GMMStream));
    { // reset the value for reuse
      logLikelihood = -INFINITY;
      logLikelihoodOld = -INFINITY;
    }
    return 0; // some cuda async operation are not finished yet, but we are
              // using the same stream
  }

  __host__ GMM() {
    cudaErrChk(cudaStreamCreate(&GMMStream));

    cudaErrChk(
        cudaMalloc(&dataDevicePtr, sizeof(GMMDataMultiDim<T, dataDim, U>)));

    reductionTempArraySize = 1024;
    cudaErrChk(cudaMalloc(&reductionTempArrayCUDA,
                          sizeof(T) * reductionTempArraySize));

    cudaErrChk(cudaMalloc(&logLikelihoodCUDA, sizeof(T)));
    cudaErrChk(cudaMallocHost(&logResult, sizeof(T)));
  }

  __host__ void preProcessDataGMM(const T* meanArray,
                                  const T* maxVelocityArray) {

    memcpy(meanDataInit, meanArray, sizeof(T) * dataDim);
    cudaErrChk(cudaMemcpyAsync(meanDataInitCUDA, meanDataInit,
                               sizeof(T) * dataDim, cudaMemcpyHostToDevice,
                               GMMStream));
    cudaErrChk(cudaStreamSynchronize(GMMStream));

    if constexpr (NORMALIZE_DATA_FOR_GMM)
      normalizePoints(maxVelocityArray);
  }

  __host__ int initGMM() {
    // do the GMR
    int step = 0;
    bool pruned = false;

    while (step < paramHostPtr->maxIteration) {
      pruned = false;
      bool thresholdLH = false;
      int internalStep = 0;
      while (internalStep < 10) {
        // E
        calcPxAtMeanAndCoVariance();
        calcLogLikelihoodPxAndposterior();
        logLikelihood = sumLogLikelihood();

        // compare the log likelihood increament with the threshold, if the
        // increament is smaller than the threshold, or the log likelihood is
        // smaller than the previous one, output the GMM
        /*
        step > 5 is required to reach convergence for simulation at time step 0,
        when particle distribution functions are exactly gaussians, otherwise
        GMM exits after 2-3 iterations without reaching correct convergence
        (probably there's local minimum)
        */
        if (std::isnan(logLikelihood) ||
            ((fabs(logLikelihood - logLikelihoodOld) <
                  paramHostPtr->threshold ||
              logLikelihood < logLikelihoodOld) &&
             step > 5)) {
          // std::cout << "Converged at step " << step << std::endl;
          thresholdLH = true;
          break;
        }
        // std::cout << "Step " << step << " log likelihood: " << logLikelihood
        // << std::endl;
        logLikelihoodOld = logLikelihood;

        // M
        calcPosterior();
        updateMean();
        updateWeight();
        updateCoVarianceAndDecomposition();
        internalStep++;
        step++;
      }
      if constexpr (PRUNE_COMPONENTS_GMM) {
        pruned = pruneOneComponent(step);
      }
      if (thresholdLH && !pruned)
        break;
    }

    // If we exit the EM cycle just after pruning one component, we do one more
    // EM iteration to ensure conservation of mean value and cov-matrix
    if (pruned) {
      // E
      calcPxAtMeanAndCoVariance();
      calcLogLikelihoodPxAndposterior();
      logLikelihood = sumLogLikelihood();
      // M
      calcPosterior();
      updateMean();
      updateWeight();
      updateCoVarianceAndDecomposition();
    }

    return step;
  }

  __host__ void postProcessDataGMM(const T* maxVelocityArray) {

    if constexpr (NORMALIZE_DATA_FOR_GMM)
      normalizeDataBack(maxVelocityArray);
  }

  __host__ GMMResult<T, dataDim> getGMMResult(int simuStep, int convergeStep) {
    GMMResult<T, dataDim> result(simuStep, paramHostPtr->numComponents);

    result.convergeStep = convergeStep;

    if (std::isnan(logLikelihood))
      std::cerr << "[!]GMM: LogLikelihood is NaN!"
                << " GMM step: " << convergeStep << std::endl;

    result.logLikelihoodFinal = logLikelihood;

    // copy from device array
    cudaErrChk(cudaMemcpyAsync(result.weight.get(), weightCUDA,
                               sizeof(T) * paramHostPtr->numComponents,
                               cudaMemcpyDefault, GMMStream));
    cudaErrChk(
        cudaMemcpyAsync(result.mean.get(), meanCUDA,
                        sizeof(T) * paramHostPtr->numComponents * dataDim,
                        cudaMemcpyDefault, GMMStream));
    cudaErrChk(cudaMemcpyAsync(result.coVariance.get(), coVarianceCUDA,
                               sizeof(T) * paramHostPtr->numComponents *
                                   dataDim * dataDim,
                               cudaMemcpyDefault, GMMStream));

    cudaErrChk(cudaStreamSynchronize(GMMStream));

    for (int i = 0; i < paramHostPtr->numComponents; i++) {
      result.weight[i] = exp(result.weight[i]);
    }

    return result;
  }

  __host__ ~GMM() {
    // created in constructor
    cudaErrChk(cudaStreamDestroy(GMMStream));

    cudaErrChk(cudaFree(dataDevicePtr));
    cudaErrChk(cudaFree(reductionTempArrayCUDA));
    cudaErrChk(cudaFree(logLikelihoodCUDA));
    cudaErrChk(cudaFreeHost(logResult));

    // allocated in config

    // deallocate the old arrays
    if (sizeNumComponents > 0) {
      // device
      cudaErrChk(cudaFree(flagActiveComponentsCUDA));
      cudaErrChk(cudaFree(weightCUDA));
      cudaErrChk(cudaFree(meanCUDA));
      cudaErrChk(cudaFree(meanDataInitCUDA));
      cudaErrChk(cudaFree(coVarianceCUDA));
      cudaErrChk(cudaFree(coVarianceDecomposedCUDA));
      cudaErrChk(cudaFree(normalizerCUDA));
      cudaErrChk(cudaFree(PosteriorCUDA));

      if (sizeNumData > 0) {
        // device
        cudaErrChk(cudaFree(posteriorCUDA));
        cudaErrChk(cudaFree(tempArrayCUDA));
      }

      // host
      cudaErrChk(cudaFreeHost(flagActiveComponents));
      cudaErrChk(cudaFreeHost(weight));
      cudaErrChk(cudaFreeHost(mean));
      cudaErrChk(cudaFreeHost(meanDataInit));
      cudaErrChk(cudaFreeHost(coVariance));
      cudaErrChk(cudaFreeHost(coVarianceDecomposed));
      cudaErrChk(cudaFreeHost(normalizer));
    }
  }

private:
  int reduceBlockNum(int dataSize, int blockSize) {
    constexpr int elementPerThread =
        8; // 8 elements per thread, as the input is 10000
    if (dataSize < elementPerThread)
      dataSize = elementPerThread;
    auto blockNum = getGridSize(dataSize / elementPerThread, blockSize);
    blockNum = blockNum > 1024 ? 1024 : blockNum;

    if (reductionTempArraySize < blockNum) {
      cudaErrChk(cudaFree(reductionTempArrayCUDA));
      cudaErrChk(cudaMalloc(&reductionTempArrayCUDA, sizeof(T) * blockNum));
      reductionTempArraySize = blockNum;
    }

    return blockNum;
  }

  void normalizePoints(const T* maxVelocityArray) {

    // normalize data such that velocities are in range -1;1
    // launch kernel
    cudaGMMWeightKernel::normalizePointsKernel<T, dataDim, U, false>
        <<<getGridSize(dataHostPtr->getNumData(), 256), 256, 0, GMMStream>>>(
            dataDevicePtr, meanDataInitCUDA, maxVelocityArray);
  }

  void normalizeDataBack(const T* maxVelocityArray) {

    // normalize data back to the original data range
    // launch kernel
    cudaGMMWeightKernel::normalizePointsKernel<T, dataDim, U, true>
        <<<getGridSize(dataHostPtr->getNumData(), 256), 256, 0, GMMStream>>>(
            dataDevicePtr, meanDataInitCUDA, maxVelocityArray);

    cudaGMMWeightKernel::normalizeMeanAndCovBack<T, dataDim>
        <<<1, paramHostPtr->numComponents, 0, GMMStream>>>(
            meanCUDA, coVarianceCUDA, meanDataInitCUDA, maxVelocityArray,
            paramHostPtr->numComponents, flagActiveComponentsCUDA);
  }

  void calcPxAtMeanAndCoVariance() {

    // launch kernel
    cudaGMMWeightKernel::calcLogLikelihoodForPointsKernel<<<
        getGridSize(dataHostPtr->getNumData(), 256), 256, 0, GMMStream>>>(
        dataDevicePtr, meanCUDA, coVarianceDecomposedCUDA, posteriorCUDA,
        paramHostPtr->numComponents, flagActiveComponentsCUDA);

    // posterior_nk holds log p(x_i|mean,coVariance) for each data point i and
    // each component k, temporary storage
  }

  void calcLogLikelihoodPxAndposterior() {
    // launch kernel, the first posterior_nk is the log p(x_i|mean,coVariance)
    cudaGMMWeightKernel::calcLogLikelihoodPxAndposteriorKernel<<<
        getGridSize(dataHostPtr->getNumData(), 256), 256, 0, GMMStream>>>(
        dataDevicePtr, weightCUDA, posteriorCUDA, tempArrayCUDA, posteriorCUDA,
        paramHostPtr->numComponents, flagActiveComponentsCUDA);

    // now the posterior_nk is the log posterior_nk
    // the tempArrayCUDA is the log Px for each data point
  }

  T sumLogLikelihood() { // its sync now, but can be async with the M step
    // sum the log likelihood Px with reduction, then return the sum
    constexpr int blockSize = 256;
    auto blockNum = reduceBlockNum(dataHostPtr->getNumData(), blockSize);

    cudaReduction::reduceSumPreProcess<
        T, blockSize, cudaReduction::PreProcessType::none, void, U,
        true> // weighted sum
        <<<blockNum, blockSize, blockSize * sizeof(T), GMMStream>>>(
            tempArrayCUDA, reductionTempArrayCUDA, dataHostPtr->getNumData(),
            nullptr, dataHostPtr->getWeight());
    cudaReduction::reduceSumWarp<T><<<1, WARP_SIZE, 0, GMMStream>>>(
        reductionTempArrayCUDA, logLikelihoodCUDA, blockNum);

    cudaErrChk(cudaMemcpyAsync(logResult, logLikelihoodCUDA, sizeof(T),
                               cudaMemcpyDefault, GMMStream));

    cudaErrChk(cudaStreamSynchronize(
        GMMStream)); // The M step can be preLaunched for most of the time

    return *logResult;
  }

  void calcPosterior() { // the Big Gamma, it can be direct sum then log

    constexpr int blockSize = 256;
    auto blockNum = reduceBlockNum(dataHostPtr->getNumData(), blockSize);

    auto maxValueArray =
        tempArrayCUDA; // maxValues of posterior_nk for each component
    for (int component = 0; component < paramHostPtr->numComponents;
         component++) {

      if (!flagActiveComponents[component])
        continue;

      // get the max value of the posterior_nk(little gamma), with reduction
      auto posteriorComponent =
          posteriorCUDA + component * dataHostPtr->getNumData();

      cudaReduction::reduceMax<T, blockSize>
          <<<blockNum, blockSize, blockSize * sizeof(T), GMMStream>>>(
              posteriorComponent, reductionTempArrayCUDA,
              dataHostPtr->getNumData());

      cudaReduction::reduceMaxWarp<T><<<1, WARP_SIZE, 0, GMMStream>>>(
          reductionTempArrayCUDA, maxValueArray + component, blockNum);

      // reduction sum with pre-process and post-process to get the log
      // Posterior_k
      cudaReduction::reduceSumPreProcess<
          T, blockSize, cudaReduction::PreProcessType::minusConstThenEXP, T, U,
          true><<<blockNum, blockSize, blockSize * sizeof(T), GMMStream>>>(
          posteriorComponent, reductionTempArrayCUDA, dataHostPtr->getNumData(),
          maxValueArray + component, dataHostPtr->getWeight());

      cudaReduction::reduceSumWarpPostProcess<
          T, cudaReduction::PostProcessType::logAdd, T>
          <<<1, WARP_SIZE, 0, GMMStream>>>(reductionTempArrayCUDA,
                                           PosteriorCUDA + component, blockNum,
                                           maxValueArray + component);
    }

    // now(after kernel execution) we have the log Posterior_k for each
    // component
  }

  void updateMean() {
    constexpr int blockSize = 256;
    auto blockNum = reduceBlockNum(dataHostPtr->getNumData(), blockSize);

    // for each component, for each dimension
    for (int component = 0; component < paramHostPtr->numComponents;
         component++) {

      if (!flagActiveComponents[component])
        continue;

      for (int dim = 0; dim < dataHostPtr->getDim(); dim++) {
        // calc x_i * posterior_nk, could be merged with the reduction sum
        // sum the x_i * posterior_nk with reduction
        cudaReduction::reduceSumPreProcess<
            T, blockSize, cudaReduction::PreProcessType::multiplyEXP, T, U,
            true><<<blockNum, blockSize, blockSize * sizeof(T), GMMStream>>>(
            dataHostPtr->getDim(dim), reductionTempArrayCUDA,
            dataHostPtr->getNumData(),
            posteriorCUDA + component * dataHostPtr->getNumData(),
            dataHostPtr->getWeight());

        // divide by the Posterior_k, can be post processed with the reduction
        // sum warp
        cudaReduction::reduceSumWarpPostProcess<
            T, cudaReduction::PostProcessType::divideEXP, T>
            <<<1, WARP_SIZE, 0, GMMStream>>>(
                reductionTempArrayCUDA,
                meanCUDA + component * dataHostPtr->getDim() + dim, blockNum,
                PosteriorCUDA + component);
      }
    }
  }

  void updateWeight() {
    // calc the new weight for components
    cudaGMMWeightKernel::updateWeightKernel<<<
        1, paramHostPtr->numComponents, paramHostPtr->numComponents * sizeof(T),
        GMMStream>>>(weightCUDA, PosteriorCUDA, paramHostPtr->numComponents,
                     flagActiveComponentsCUDA);
  }

  void updateCoVarianceAndDecomposition() {
    constexpr int blockSize = 256;
    auto blockNum = reduceBlockNum(dataHostPtr->getNumData(), blockSize);

    // calc the new coVariance for components
    cudaGMMWeightKernel::updateCoVarianceKernel<<<
        getGridSize(dataHostPtr->getNumData(), 256), 256, 0, GMMStream>>>(
        dataDevicePtr, posteriorCUDA, PosteriorCUDA, meanCUDA, tempArrayCUDA,
        paramHostPtr->numComponents, flagActiveComponentsCUDA);

    // sum the coVariance with reduction, then divide by the Posterior_k
    for (int component = 0; component < paramHostPtr->numComponents;
         component++) {

      if (!flagActiveComponents[component])
        continue;

      auto coVarianceComponent = tempArrayCUDA + component *
                                                     dataHostPtr->getNumData() *
                                                     dataDim * dataDim;
      for (int element = 0; element < dataDim * dataDim; element++) {
        cudaReduction::reduceSumPreProcess<
            T, blockSize, cudaReduction::PreProcessType::none, void, U, true>
            <<<blockNum, blockSize, blockSize * sizeof(T), GMMStream>>>(
                coVarianceComponent + element * dataHostPtr->getNumData(),
                reductionTempArrayCUDA, dataHostPtr->getNumData(), nullptr,
                dataHostPtr->getWeight());

        cudaReduction::reduceSumWarpPostProcess<
            T, cudaReduction::PostProcessType::divideEXP, T>
            <<<1, WARP_SIZE, 0, GMMStream>>>(
                reductionTempArrayCUDA,
                coVarianceCUDA + component * dataDim * dataDim + element,
                blockNum, PosteriorCUDA + component);
      }
    }

    // check cov-matrix and adjust main diagonal to ensure determinate>0 and
    // cholesky decomposition
    if constexpr (CHECK_COVMATRIX_GMM) {
      cudaGMMWeightKernel::checkAdjustCoVarianceKernel<T, dataDim>
          <<<1, paramHostPtr->numComponents, 0, GMMStream>>>(
              coVarianceCUDA, paramHostPtr->numComponents,
              flagActiveComponentsCUDA);
    }

    // decompose the coVariance with cholesky decomposition -> A = LL^T
    cudaGMMWeightKernel::decomposeCoVarianceKernel<T, dataDim>
        <<<1, paramHostPtr->numComponents, 0, GMMStream>>>(
            coVarianceCUDA, coVarianceDecomposedCUDA, normalizerCUDA,
            paramHostPtr->numComponents, flagActiveComponentsCUDA);
  }

  // prune 1 GMM component if its weight is below a threshold, reset the other
  // low weight components
  bool pruneOneComponent(const int step) {

    bool pruned = false;
    *flagResetMean = false;
    // set low weight component to inactive and update the other component
    // weights launch kernel
    cudaGMMWeightKernel::pruneOneComponentKernel<<<
        1, paramHostPtr->numComponents,
        paramHostPtr->numComponents * (sizeof(T) + sizeof(bool)) + sizeof(bool),
        GMMStream>>>(weightCUDA, PRUNE_THRESHOLD_GMM,
                     paramHostPtr->numComponents, flagActiveComponentsCUDA);

    cudaErrChk(cudaMemcpyAsync(flagActiveComponents, flagActiveComponentsCUDA,
                               sizeof(bool) * (paramHostPtr->numComponents + 1),
                               cudaMemcpyDefault, GMMStream));

    // check if any GMM component has NaN mean --> reset mean
    // launch kernel
    cudaGMMWeightKernel::checkMeanValueComponents<T, dataDim>
        <<<1, paramHostPtr->numComponents,
           paramHostPtr->numComponents * sizeof(bool), GMMStream>>>(
            meanCUDA, paramHostPtr->numComponents, flagActiveComponentsCUDA,
            flagResetMeanCUDA);

    cudaErrChk(cudaMemcpyAsync(flagResetMean, flagResetMeanCUDA, sizeof(bool),
                               cudaMemcpyDefault, GMMStream));
    cudaErrChk(cudaStreamSynchronize(GMMStream));

    pruned = flagActiveComponents[paramHostPtr->numComponents];

    if (pruned) {
      // reset logLikeLihood to restart GMM
      this->logLikelihoodOld = -INFINITY;
      numActiveComponents -= 1;
      if (numActiveComponents < 1) {
        std::cerr << "[!]Error PruneComponents " << " numActiveComponents < 1 "
                  << numActiveComponents << " - " << " GMM step: " << step
                  << std::endl;
      }
    } else if (*flagResetMean) {
      this->logLikelihoodOld = -INFINITY;
      pruned = true;
    }

    return pruned;
  }
};

} // namespace cudaGMMWeight

#endif // CUDA_GMM_CUH
