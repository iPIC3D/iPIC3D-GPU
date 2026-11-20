#ifndef CUDA_GMM_CUH
#define CUDA_GMM_CUH

#include "cudaGMMUtility.cuh"
#include "cudaGMMkernel.cuh"
#include "cudaTypeDef.cuh"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>

namespace cudaGMMWeight {

/** Threads per block of the EM kernels: also the most components. */
inline constexpr int kGMMThreads = 256;
/** Points per thread of the M-step statistics. */
inline constexpr int kStatisticsPointsPerThread = 4;
/** Most blocks of the M-step statistics (a grid stride beyond): enough to
 *  fill a large GPU on the largest histograms. */
inline constexpr int kMaxStatisticsBlocks = 256;
/** EM steps at most between two prunings (and resets of the non-finite
 *  means); a converged fit is checked at once. */
inline constexpr int kPruneInterval = 10;
/** EM steps before the fit may converge: at time step 0, when the
 *  distributions are exactly Gaussian, the fit otherwise stops after 2-3
 *  iterations, far from the maximum of the likelihood. */
inline constexpr int kMinStepsToConverge = 6;

static_assert(kGMMThreads % 64 == 0 &&
                  kGMMThreads <= cudaGMMWeightKernel::kMaxStatisticsThreads,
              "whole warps on NVIDIA and AMD, within the statistics kernel");

/**
 * @brief Gaussian mixture fitted by expectation-maximisation to weighted
 * points, on the GPU. One EM iteration is three kernels (E step, M-step
 * statistics, M-step finalize) and one host synchronisation; the buffers are
 * sized by the points and components and grow only.
 */
template <typename T, int dataDim, typename U = int> class GMM {

private:
  cudaStream_t GMMStream; // the caller's: all the work of the fit runs on it

  // object on host, its data pointers to the data on device
  GMMDataMultiDim<T, dataDim, U>* dataHostPtr = nullptr;
  // object on device, its data pointers to the data on device
  GMMDataMultiDim<T, dataDim, U>* dataDevicePtr = nullptr;

  GMMParam_t<T>* paramHostPtr = nullptr; // the parameters of the fit

  int sizeNumComponents = 0; // components the buffers below hold
  int sizeNumPoints = 0;     // points the point buffers below hold

  // the arrays on device
  bool* flagActiveComponentsCUDA = nullptr; // numComponents + 2: the active
                                            // components, then whether one
                                            // was pruned and whether a mean
                                            // was reset
  T* weightCUDA = nullptr;                  // numComponents, log(weight)
  T* meanCUDA = nullptr;                    // numComponents * dataDim
  T* coVarianceCUDA = nullptr;              // numComponents * dataDim * dataDim
  T* coVarianceDecomposedCUDA = nullptr; // numComponents * dataDim * dataDim,
                                         // Cholesky factor (choleskyFactor)
  T* normalizerCUDA = nullptr;           // numComponents, log normalisation
  T* partialCUDA = nullptr; // M-step block sums, kMaxStatisticsBlocks *
                            // (numComponents * kMStepStatistics + 1)
  T* resultCUDA = nullptr;  // log-likelihood and total weight of an EM step

  T* gammaCUDA = nullptr; // responsibilities, numComponents * numData
  T* weightedLogLikelihoodCUDA = nullptr; // h log p(x), numData

  // the arrays on host, results and init values
  bool* flagActiveComponents = nullptr; // numComponents + 2, as on device
  T* weight = nullptr;                  // numComponents
  T* mean = nullptr;                    // numComponents * dataDim
  T* coVariance = nullptr;              // numComponents * dataDim * dataDim
  T* resultHost = nullptr;              // pinned copy of resultCUDA

  // runtime variables
  T logLikelihood = -INFINITY;
  T logLikelihoodOld = -INFINITY;
  T totalWeight = 0; // sum of the weights of the data

public:
  /** Allocate the buffers if they are too small, then set the parameters
   *  and the initial components (required, in the units of the points),
   *  normalised with the points when the parameters ask to. The fit keeps
   *  the two pointers: both must live until getGMMResult. */
  __host__ void config(GMMParam_t<T>* GMMParam,
                       GMMDataMultiDim<T, dataDim, U>* data) {
    const int numComponents = GMMParam->numComponents;
    if (numComponents < 1 || numComponents > kGMMThreads)
      throw std::invalid_argument("GMM: 1 to " + std::to_string(kGMMThreads) +
                                  " components, not " +
                                  std::to_string(numComponents));
    if (data->getNumData() < 1)
      throw std::invalid_argument("GMM: no point to fit");
    if (!GMMParam->weightInit || !GMMParam->meanInit ||
        !GMMParam->coVarianceInit)
      throw std::invalid_argument("GMM: the initial components are required");

    const int oldSizeComponents = sizeNumComponents;
    if (numComponents > oldSizeComponents) {
      freeComponentBuffers();
      const int K = numComponents;
      cudaErrChk(cudaMalloc(&flagActiveComponentsCUDA, sizeof(bool) * (K + 2)));
      cudaErrChk(cudaMalloc(&weightCUDA, sizeof(T) * K));
      cudaErrChk(cudaMalloc(&meanCUDA, sizeof(T) * K * dataDim));
      cudaErrChk(
          cudaMalloc(&coVarianceCUDA, sizeof(T) * K * dataDim * dataDim));
      cudaErrChk(cudaMalloc(&coVarianceDecomposedCUDA,
                            sizeof(T) * K * dataDim * dataDim));
      cudaErrChk(cudaMalloc(&normalizerCUDA, sizeof(T) * K));
      cudaErrChk(cudaMalloc(
          &partialCUDA,
          sizeof(T) * kMaxStatisticsBlocks *
              (K * cudaGMMWeightKernel::kMStepStatistics<dataDim> + 1)));

      cudaErrChk(cudaMallocHost(&flagActiveComponents, sizeof(bool) * (K + 2)));
      cudaErrChk(cudaMallocHost(&weight, sizeof(T) * K));
      cudaErrChk(cudaMallocHost(&mean, sizeof(T) * K * dataDim));
      cudaErrChk(
          cudaMallocHost(&coVariance, sizeof(T) * K * dataDim * dataDim));
      sizeNumComponents = K; // allocated
    }

    // the arrays sized by the points (and the components), with a margin:
    // the number of points drifts up between fits
    if (data->getNumData() > sizeNumPoints ||
        sizeNumComponents > oldSizeComponents) {
      const int numPoints =
          std::max(sizeNumPoints, data->getNumData() + data->getNumData() / 4);
      freePointBuffers();
      cudaErrChk(
          cudaMalloc(&gammaCUDA, sizeof(T) * sizeNumComponents * numPoints));
      cudaErrChk(cudaMalloc(&weightedLogLikelihoodCUDA, sizeof(T) * numPoints));
      sizeNumPoints = numPoints; // allocated
    }

    // the initial components in the units of the fit, the weights as logs,
    // all active
    const T scale = GMMParam->normalize ? GMMParam->scale : T(1);
    for (int i = 0; i < numComponents * dataDim; i++)
      mean[i] = GMMParam->meanInit[i] / scale;
    for (int i = 0; i < numComponents * dataDim * dataDim; i++)
      coVariance[i] = GMMParam->coVarianceInit[i] / (scale * scale);
    for (int i = 0; i < numComponents; i++) {
      weight[i] = log(GMMParam->weightInit[i]);
      flagActiveComponents[i] = true;
    }
    flagActiveComponents[numComponents] = false;
    flagActiveComponents[numComponents + 1] = false;

    cudaErrChk(cudaMemcpyAsync(flagActiveComponentsCUDA, flagActiveComponents,
                               sizeof(bool) * (numComponents + 2),
                               cudaMemcpyHostToDevice, GMMStream));
    cudaErrChk(cudaMemcpyAsync(weightCUDA, weight, sizeof(T) * numComponents,
                               cudaMemcpyHostToDevice, GMMStream));
    cudaErrChk(cudaMemcpyAsync(meanCUDA, mean,
                               sizeof(T) * numComponents * dataDim,
                               cudaMemcpyHostToDevice, GMMStream));
    cudaErrChk(cudaMemcpyAsync(coVarianceCUDA, coVariance,
                               sizeof(T) * numComponents * dataDim * dataDim,
                               cudaMemcpyHostToDevice, GMMStream));

    paramHostPtr = GMMParam;
    dataHostPtr = data;
    cudaErrChk(cudaMemcpyAsync(dataDevicePtr, dataHostPtr,
                               sizeof(GMMDataMultiDim<T, dataDim, U>),
                               cudaMemcpyDefault, GMMStream));
    if (paramHostPtr->normalize)
      cudaGMMWeightKernel::normalizePointsKernel<T, dataDim, U, false>
          <<<getGridSize(data->getNumData(), kGMMThreads), kGMMThreads, 0,
             GMMStream>>>(dataDevicePtr, paramHostPtr->scale);

    // the first E step needs the factor of the initial coVariance
    checkAndDecomposeCoVariance();
    logLikelihood = -INFINITY;
    logLikelihoodOld = -INFINITY;
    totalWeight = 0;
  }

  /** @param stream the stream of the fit, ordered after the work that
   *  produced the points */
  __host__ explicit GMM(cudaStream_t stream) : GMMStream(stream) {
    cudaErrChk(
        cudaMalloc(&dataDevicePtr, sizeof(GMMDataMultiDim<T, dataDim, U>)));

    cudaErrChk(cudaMalloc(&resultCUDA, 2 * sizeof(T)));
    cudaErrChk(cudaMallocHost(&resultHost, 2 * sizeof(T)));
  }

  GMM(const GMM&) = delete;
  GMM& operator=(const GMM&) = delete;

  /** Run the EM, then give the fit and the points back in the units of the
   *  points; returns the number of EM steps, the last one after a pruning
   *  included. A step that finds no weight (points of no weight, or no
   *  finite component) ends the EM, unless a pruning or a reset of a
   *  non-finite mean follows: the EM then starts again. */
  __host__ int initGMM() {
    int step = 0;
    bool pruned = false;

    while (step < paramHostPtr->maxIteration) {
      pruned = false;
      bool converged = false;
      for (int internalStep = 0;
           internalStep < kPruneInterval && step < paramHostPtr->maxIteration;
           internalStep++) {
        logLikelihood = emStep(); // of the parameters before its M step
        // converged when the log-likelihood per unit weight gains less than
        // the threshold, or decreases
        const bool done = std::isnan(logLikelihood) ||
                          ((fabs(logLikelihood - logLikelihoodOld) <
                                paramHostPtr->threshold * totalWeight ||
                            logLikelihood < logLikelihoodOld) &&
                           step >= kMinStepsToConverge);
        logLikelihoodOld = logLikelihood;
        step++;
        if (done || !(totalWeight > 0)) {
          converged = true;
          break;
        }
      }
      pruned = pruneAndResetComponents();
      if (converged && !pruned)
        break;
    }

    // one more EM step after a pruning, so that the mean value and the
    // coVariance are consistent with the remaining components
    if (pruned) {
      logLikelihood = emStep();
      step++;
    }

    // back to the units of the points (multiplied back, up to rounding); the
    // density of the normalised points is scale^dataDim times larger
    if (paramHostPtr->normalize) {
      const T scale = paramHostPtr->scale;
      cudaGMMWeightKernel::normalizePointsKernel<T, dataDim, U, true>
          <<<getGridSize(dataHostPtr->getNumData(), kGMMThreads), kGMMThreads,
             0, GMMStream>>>(dataDevicePtr, scale);
      cudaGMMWeightKernel::normalizeMeanAndCovBack<T, dataDim>
          <<<1, paramHostPtr->numComponents, 0, GMMStream>>>(
              meanCUDA, coVarianceCUDA, scale, paramHostPtr->numComponents);
      logLikelihood -= totalWeight * dataDim * log(scale);
    }
    cudaErrChk(cudaGetLastError());
    return step;
  }

  /** The fit: its components (weights, not their logs) and the
   *  log-likelihood of the parameters before the last M step. */
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
    // allocated in the constructor
    cudaErrChk(cudaFree(dataDevicePtr));
    cudaErrChk(cudaFree(resultCUDA));
    cudaErrChk(cudaFreeHost(resultHost));

    // allocated in config
    freeComponentBuffers();
    freePointBuffers();
  }

private:
  /** Free a device or pinned buffer, and null it. */
  template <typename V> static void freeDevice(V*& buffer) {
    cudaErrChk(cudaFree(buffer));
    buffer = nullptr;
  }
  template <typename V> static void freePinned(V*& buffer) {
    cudaErrChk(cudaFreeHost(buffer));
    buffer = nullptr;
  }

  /** Free the buffers sized by the components (allocated in config; null
   *  when not, also after a failed allocation). */
  void freeComponentBuffers() {
    sizeNumComponents = 0;
    freeDevice(flagActiveComponentsCUDA);
    freeDevice(weightCUDA);
    freeDevice(meanCUDA);
    freeDevice(coVarianceCUDA);
    freeDevice(coVarianceDecomposedCUDA);
    freeDevice(normalizerCUDA);
    freeDevice(partialCUDA);
    freePinned(flagActiveComponents);
    freePinned(weight);
    freePinned(mean);
    freePinned(coVariance);
  }

  /** Free the buffers sized by the points. */
  void freePointBuffers() {
    sizeNumPoints = 0;
    freeDevice(gammaCUDA);
    freeDevice(weightedLogLikelihoodCUDA);
  }

  /**
   * One EM iteration: responsibilities (E step), the moments of the points
   * per component and the new parameters (M step). Synchronous.
   * @return the log-likelihood of the parameters before the M step; the total
   * weight of the data goes to totalWeight
   */
  T emStep() {
    const int numData = dataHostPtr->getNumData();
    const int numComponents = paramHostPtr->numComponents;
    cudaGMMWeightKernel::eStepKernel<<<getGridSize(numData, kGMMThreads),
                                       kGMMThreads, 0, GMMStream>>>(
        dataDevicePtr, weightCUDA, meanCUDA, coVarianceDecomposedCUDA,
        normalizerCUDA, numComponents, flagActiveComponentsCUDA, gammaCUDA,
        weightedLogLikelihoodCUDA);

    const int blocks =
        std::min(getGridSize(numData, kStatisticsPointsPerThread * kGMMThreads),
                 kMaxStatisticsBlocks);
    cudaGMMWeightKernel::
        mStepStatisticsKernel<<<blocks, kGMMThreads, 0, GMMStream>>>(
            dataDevicePtr, gammaCUDA, weightedLogLikelihoodCUDA, meanCUDA,
            numComponents, flagActiveComponentsCUDA, partialCUDA);

    const int width =
        numComponents * cudaGMMWeightKernel::kMStepStatistics<dataDim> + 1;
    cudaGMMWeightKernel::mStepFinalizeKernel<T, dataDim>
        <<<1, kGMMThreads, width * sizeof(T), GMMStream>>>(
            partialCUDA, blocks, weightCUDA, meanCUDA, coVarianceCUDA,
            coVarianceDecomposedCUDA, normalizerCUDA, numComponents,
            flagActiveComponentsCUDA, paramHostPtr->checkCovariance,
            paramHostPtr->minVariance, paramHostPtr->minPivot(), resultCUDA);
    cudaErrChk(cudaGetLastError());

    cudaErrChk(cudaMemcpyAsync(resultHost, resultCUDA, 2 * sizeof(T),
                               cudaMemcpyDeviceToHost, GMMStream));
    cudaErrChk(cudaStreamSynchronize(GMMStream));
    totalWeight = resultHost[1];
    return resultHost[0];
  }

  /**
   * Check the coVariance (non-finite elements, smallest variance) if the
   * parameters ask to, and factor it, A = LL^T, with the log normalisation
   * of each component; a pivot below minPivot() is raised to it.
   */
  void checkAndDecomposeCoVariance() {
    if (paramHostPtr->checkCovariance)
      cudaGMMWeightKernel::checkAdjustCoVarianceKernel<T, dataDim>
          <<<1, paramHostPtr->numComponents, 0, GMMStream>>>(
              coVarianceCUDA, paramHostPtr->numComponents,
              flagActiveComponentsCUDA, paramHostPtr->minVariance);

    cudaGMMWeightKernel::decomposeCoVarianceKernel<T, dataDim>
        <<<1, paramHostPtr->numComponents, 0, GMMStream>>>(
            coVarianceCUDA, coVarianceDecomposedCUDA, normalizerCUDA,
            paramHostPtr->numComponents, flagActiveComponentsCUDA,
            paramHostPtr->minPivot());
  }

  /** Prune the component of smallest weight below the threshold (with the
   *  parameter prune) and reset the non-finite means; true if the EM
   *  restarts. */
  bool pruneAndResetComponents() {
    const int numComponents = paramHostPtr->numComponents;
    if (paramHostPtr->prune)
      cudaGMMWeightKernel::pruneOneComponentKernel<<<
          1, numComponents,
          numComponents * (sizeof(T) + sizeof(bool)) + sizeof(bool),
          GMMStream>>>(weightCUDA, paramHostPtr->pruneThreshold, numComponents,
                       flagActiveComponentsCUDA);

    cudaGMMWeightKernel::checkMeanValueComponents<T, dataDim>
        <<<1, numComponents, numComponents * sizeof(bool), GMMStream>>>(
            meanCUDA, numComponents, flagActiveComponentsCUDA,
            flagActiveComponentsCUDA + numComponents + 1,
            paramHostPtr->normalize ? T(1) : paramHostPtr->scale);

    cudaErrChk(cudaMemcpyAsync(flagActiveComponents, flagActiveComponentsCUDA,
                               sizeof(bool) * (numComponents + 2),
                               cudaMemcpyDefault, GMMStream));
    cudaErrChk(cudaStreamSynchronize(GMMStream));

    // the EM restarts: the log-likelihood of the fewer or moved components
    // is not comparable with the last one
    const bool restart = flagActiveComponents[numComponents] ||
                         flagActiveComponents[numComponents + 1];
    if (restart)
      logLikelihoodOld = -INFINITY;
    return restart;
  }
};

} // namespace cudaGMMWeight

#endif // CUDA_GMM_CUH
