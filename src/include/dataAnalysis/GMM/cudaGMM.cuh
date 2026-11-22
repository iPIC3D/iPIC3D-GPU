#ifndef CUDA_GMM_CUH
#define CUDA_GMM_CUH

#include "cudaGMMUtility.cuh"
#include "cudaGMMkernel.cuh"
#include "cudaTypeDef.cuh"

#include <algorithm>
#include <climits>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace cudaGMMWeight {

/** Points per thread of the M-step statistics. */
inline constexpr int kStatisticsPointsPerThread = 4;
/** Most blocks of the M-step statistics of a fit (a grid stride beyond):
 *  enough to fill a large GPU on the largest histograms. */
inline constexpr int kMaxStatisticsBlocks = 256;

using cudaGMMWeightKernel::kGMMThreads;

/** A device (pinned with `pinned`) array that only grows, by a quarter more
 *  than asked; its content is lost when it grows. */
template <typename V, bool pinned = false> class GrowingBuffer {
public:
  GrowingBuffer() = default;
  GrowingBuffer(const GrowingBuffer&) = delete;
  GrowingBuffer& operator=(const GrowingBuffer&) = delete;
  ~GrowingBuffer() { release(); }

  void reserve(size_t count) {
    if (count <= capacity)
      return;
    release();
    const size_t size = count + count / 4;
    if constexpr (pinned)
      cudaErrChk(cudaMallocHost(&data, sizeof(V) * size));
    else
      cudaErrChk(cudaMalloc(&data, sizeof(V) * size));
    capacity = size;
  }
  V* get() const { return data; }
  V& operator[](size_t i) const {
    static_assert(pinned, "host access to a pinned buffer only");
    return data[i];
  }

private:
  void release() {
    if constexpr (pinned)
      cudaErrChk(cudaFreeHost(data));
    else
      cudaErrChk(cudaFree(data));
    data = nullptr;
    capacity = 0;
  }
  V* data = nullptr;
  size_t capacity = 0;
};

/**
 * @brief Gaussian mixtures fitted by expectation-maximisation to weighted
 * points, on the GPU: one or several fits at once, each on its own range of
 * the points, with the same settings and number of components. One EM
 * iteration of all the fits is three kernels (E step, M-step statistics,
 * M-step finalize, which also decides the convergence and the pruning of
 * each fit) and one host synchronisation; the buffers grow only.
 */
template <typename T, int dataDim, typename U = int> class GMM {

private:
  cudaStream_t GMMStream; // the caller's: all the work of the fit runs on it

  // the points: a copy on the device of the caller's object, whose pointers
  // are to device arrays
  GMMDataMultiDim<T, dataDim, U>* dataDevicePtr = nullptr;

  // the fits of the last config: fit f has the points [fitOffset[f],
  // fitOffset[f + 1]) and the components [f K, (f + 1) K) of the arrays
  int numFits = 0, numComponents = 0; // K
  bool normalize = false; // the points divided by the scale of their fit
  cudaGMMWeightKernel::GMMControl<T> control{};
  int numEStepBlocks = 0, numStatisticsBlocks = 0;

  // on device, per fit and component
  GrowingBuffer<bool> activeCUDA;            // the active components
  GrowingBuffer<T> logWeightCUDA;            // log(weight)
  GrowingBuffer<T> meanCUDA;                 // dataDim per component
  GrowingBuffer<T> coVarianceCUDA;           // dataDim^2 per component
  GrowingBuffer<T> coVarianceDecomposedCUDA; // Cholesky factors
  GrowingBuffer<T> normalizerCUDA;           // log normalisation
  // on device, per fit: its phase (kFitRunning, ...), its EM state, its
  // scale, its first point and its first statistics block (both + 1, the
  // ends)
  GrowingBuffer<int> phaseCUDA;
  GrowingBuffer<cudaGMMWeightKernel::GMMFitState<T>> stateCUDA;
  GrowingBuffer<T> scaleCUDA;
  GrowingBuffer<int> fitOffsetCUDA;
  GrowingBuffer<int> fitStatisticsBlockCUDA;
  // the blocks of the E step (fit and first point) and of the statistics
  GrowingBuffer<int> eStepBlockFitCUDA, eStepBlockFirstCUDA;
  GrowingBuffer<int> statisticsBlockFitCUDA;
  // M-step block sums, (numComponents * kMStepStatistics + 1) per block
  GrowingBuffer<T> partialCUDA;
  // per point: responsibilities (numComponents * numData), h log p(x)
  GrowingBuffer<T> gammaCUDA;
  GrowingBuffer<T> weightedLogLikelihoodCUDA;

  // pinned, the initial values and the results of the fits
  GrowingBuffer<bool, true> active;
  GrowingBuffer<T, true> logWeight;
  GrowingBuffer<T, true> mean;
  GrowingBuffer<T, true> coVariance;
  GrowingBuffer<int, true> phase;
  GrowingBuffer<cudaGMMWeightKernel::GMMFitState<T>, true> state;
  GrowingBuffer<T, true> scale;
  // the block tables of config(), page-locked: copied to the device
  PinnedVector<int> fitOffset, fitStatisticsBlock;
  PinnedVector<int> eStepFit, eStepFirst, statisticsFit;

public:
  /**
   * @brief Set the parameters, the points and the initial components of
   * numFits fits, allocating the buffers if they are too small.
   * @param GMMParam the settings, and the initial components (required, in
   * the units of the points) and the scale of each fit, fit after fit; the
   * components are normalised with the points when the settings ask to
   * @param data the points of all the fits; the object in page-locked
   * memory (it is copied to the device)
   * @param fits the fits
   * @param offsets the points of fit f are [offsets[f], offsets[f + 1]), from
   * 0 to all the points (a fit may have none); nullptr for one fit of all
   * the points
   * data and the device arrays of the points must live until initGMM
   * returns; the fit keeps no pointer to the two objects.
   */
  __host__ void config(const GMMParam_t<T>* GMMParam,
                       const GMMDataMultiDim<T, dataDim, U>* data, int fits = 1,
                       const int* offsets = nullptr) {
    const int K = GMMParam->numComponents;
    const int numData = data->getNumData();
    if (K < 1 || K > kGMMThreads)
      throw std::invalid_argument("GMM: 1 to " + std::to_string(kGMMThreads) +
                                  " components, not " + std::to_string(K));
    if (numData < 1)
      throw std::invalid_argument("GMM: no point to fit");
    // the int indices of the responsibilities and of the components
    if ((size_t)K * numData > INT_MAX ||
        (size_t)fits * K * dataDim * dataDim > INT_MAX)
      throw std::invalid_argument("GMM: more than INT_MAX responsibilities or "
                                  "coVariance elements");
    if (!GMMParam->weightInit || !GMMParam->meanInit ||
        !GMMParam->coVarianceInit || !GMMParam->scale)
      throw std::invalid_argument(
          "GMM: the initial components and the scales are required");
    if (fits < 1 || (!offsets && fits != 1))
      throw std::invalid_argument("GMM: no fit, or several without offsets");
    if (offsets)
      fitOffset.assign(offsets, offsets + fits + 1);
    else
      fitOffset.assign({0, numData});
    if (fitOffset.front() != 0 || fitOffset.back() != numData ||
        !std::is_sorted(fitOffset.begin(), fitOffset.end()))
      throw std::invalid_argument("GMM: the fits must cover the points, in "
                                  "order");
    if (GMMParam->maxIteration < 1)
      throw std::invalid_argument("GMM: 1 EM iteration at least");
    // finite means stay finite: weighted means of the points
    auto finite = [](const T* values, size_t count) {
      return std::all_of(values, values + count,
                         [](T x) { return std::isfinite(x); });
    };
    if (!finite(GMMParam->meanInit, (size_t)fits * K * dataDim) ||
        !finite(GMMParam->coVarianceInit, (size_t)fits * K * dataDim * dataDim))
      throw std::invalid_argument("GMM: non-finite initial means or "
                                  "coVariances");
    for (int f = 0; f < fits && GMMParam->normalize; f++)
      if (fitOffset[f + 1] > fitOffset[f] &&
          !(GMMParam->scale[f] > 0 && GMMParam->scale[f] < INFINITY))
        throw std::invalid_argument("GMM: a positive finite scale per fit "
                                    "with points");
    numFits = fits;
    numComponents = K;
    normalize = GMMParam->normalize;
    control = {.maxIteration = GMMParam->maxIteration,
               .threshold = GMMParam->threshold,
               .checkCovariance = GMMParam->checkCovariance,
               .minVariance = GMMParam->minVariance,
               .minPivot = GMMParam->minPivot(),
               .prune = GMMParam->prune,
               .pruneThreshold = GMMParam->pruneThreshold};

    // the blocks: per fit, of the E step one per kGMMThreads points, of the
    // statistics those of the launch of a fit alone
    eStepFit.clear();
    eStepFirst.clear();
    statisticsFit.clear();
    fitStatisticsBlock.assign(numFits + 1, 0);
    for (int f = 0; f < numFits; f++) {
      for (int n = fitOffset[f]; n < fitOffset[f + 1]; n += kGMMThreads) {
        eStepFit.push_back(f);
        eStepFirst.push_back(n);
      }
      fitStatisticsBlock[f] = (int)statisticsFit.size();
      const int points = fitOffset[f + 1] - fitOffset[f];
      const int blocks =
          points > 0 ? std::min(getGridSize(points, kStatisticsPointsPerThread *
                                                        kGMMThreads),
                                kMaxStatisticsBlocks)
                     : 0;
      statisticsFit.insert(statisticsFit.end(), blocks, f);
    }
    fitStatisticsBlock[numFits] = (int)statisticsFit.size();
    numEStepBlocks = (int)eStepFit.size();
    numStatisticsBlocks = (int)statisticsFit.size();

    const size_t slots = (size_t)numFits * K;
    activeCUDA.reserve(slots);
    logWeightCUDA.reserve(slots);
    normalizerCUDA.reserve(slots);
    meanCUDA.reserve(slots * dataDim);
    coVarianceCUDA.reserve(slots * dataDim * dataDim);
    coVarianceDecomposedCUDA.reserve(slots * dataDim * dataDim);
    phaseCUDA.reserve(numFits);
    stateCUDA.reserve(numFits);
    scaleCUDA.reserve(numFits);
    fitOffsetCUDA.reserve(numFits + 1);
    fitStatisticsBlockCUDA.reserve(numFits + 1);
    eStepBlockFitCUDA.reserve(numEStepBlocks);
    eStepBlockFirstCUDA.reserve(numEStepBlocks);
    statisticsBlockFitCUDA.reserve(numStatisticsBlocks);
    partialCUDA.reserve(
        (size_t)numStatisticsBlocks *
        (K * cudaGMMWeightKernel::kMStepStatistics<dataDim> + 1));
    gammaCUDA.reserve((size_t)K * numData);
    weightedLogLikelihoodCUDA.reserve(numData);
    active.reserve(slots);
    logWeight.reserve(slots);
    mean.reserve(slots * dataDim);
    coVariance.reserve(slots * dataDim * dataDim);
    phase.reserve(numFits);
    state.reserve(numFits);
    scale.reserve(numFits);

    // the initial components in the units of the fit, the weights as logs;
    // the scale 1 without normalize and for a fit without points (done at
    // once: it gives its initial components back)
    for (int f = 0; f < numFits; f++) {
      const bool hasPoints = fitOffset[f + 1] > fitOffset[f];
      scale[f] = hasPoints && normalize ? GMMParam->scale[f] : T(1);
      const T s = scale[f];
      for (int i = f * K * dataDim; i < (f + 1) * K * dataDim; i++)
        mean[i] = GMMParam->meanInit[i] / s;
      for (int i = f * K * dataDim * dataDim;
           i < (f + 1) * K * dataDim * dataDim; i++)
        coVariance[i] = GMMParam->coVarianceInit[i] / (s * s);
      for (int i = f * K; i < (f + 1) * K; i++)
        logWeight[i] = log(GMMParam->weightInit[i]);
      phase[f] = hasPoints ? cudaGMMWeightKernel::kFitRunning
                           : cudaGMMWeightKernel::kFitDone;
      state[f] = {-INFINITY, -INFINITY, 0, 0, 0};
    }

    auto upload = [&](auto* device, const auto* host, size_t count) {
      cudaErrChk(cudaMemcpyAsync(device, host, sizeof(*host) * count,
                                 cudaMemcpyHostToDevice, GMMStream));
    };
    cudaErrChk(cudaMemsetAsync(activeCUDA.get(), true, sizeof(bool) * slots,
                               GMMStream)); // all active
    upload(logWeightCUDA.get(), logWeight.get(), slots);
    upload(meanCUDA.get(), mean.get(), slots * dataDim);
    upload(coVarianceCUDA.get(), coVariance.get(), slots * dataDim * dataDim);
    upload(phaseCUDA.get(), phase.get(), numFits);
    upload(stateCUDA.get(), state.get(), numFits);
    upload(scaleCUDA.get(), scale.get(), numFits);
    upload(fitOffsetCUDA.get(), fitOffset.data(), numFits + 1);
    upload(fitStatisticsBlockCUDA.get(), fitStatisticsBlock.data(),
           numFits + 1);
    upload(eStepBlockFitCUDA.get(), eStepFit.data(), numEStepBlocks);
    upload(eStepBlockFirstCUDA.get(), eStepFirst.data(), numEStepBlocks);
    upload(statisticsBlockFitCUDA.get(), statisticsFit.data(),
           numStatisticsBlocks);

    upload(dataDevicePtr, data, 1);
    if (normalize)
      cudaGMMWeightKernel::normalizePointsKernel<T, dataDim, U, false>
          <<<numEStepBlocks, kGMMThreads, 0, GMMStream>>>(
              dataDevicePtr, fitOffsetCUDA.get(), eStepBlockFitCUDA.get(),
              eStepBlockFirstCUDA.get(), scaleCUDA.get());

    // the first E step needs the factor of the initial coVariance
    cudaGMMWeightKernel::factorCoVarianceKernel<T, dataDim>
        <<<numFits, K, 0, GMMStream>>>(
            coVarianceCUDA.get(), coVarianceDecomposedCUDA.get(),
            normalizerCUDA.get(), K, activeCUDA.get(), phaseCUDA.get(),
            control);
    cudaErrChk(cudaGetLastError());
  }

  /** @param stream the stream of the fit, ordered after the work that
   *  produced the points */
  __host__ explicit GMM(cudaStream_t stream) : GMMStream(stream) {
    cudaErrChk(
        cudaMalloc(&dataDevicePtr, sizeof(GMMDataMultiDim<T, dataDim, U>)));
  }

  GMM(const GMM&) = delete;
  GMM& operator=(const GMM&) = delete;

  /** Run the EM of all the fits until each is done (mStepFinalizeKernel:
   *  at most maxIteration + 1 steps), then give the points back in their
   *  units (getGMMResult gives the fits in them). */
  __host__ void initGMM() {
    for (int steps = 0; steps <= control.maxIteration; steps++) {
      if (std::all_of(phase.get(), phase.get() + numFits,
                      [](int p) { return p == cudaGMMWeightKernel::kFitDone; }))
        break;
      emStep();
      cudaErrChk(cudaMemcpyAsync(phase.get(), phaseCUDA.get(),
                                 sizeof(int) * numFits, cudaMemcpyDeviceToHost,
                                 GMMStream));
      cudaErrChk(cudaStreamSynchronize(GMMStream));
    }

    // the points back in their units (multiplied back, up to rounding)
    if (normalize) {
      cudaGMMWeightKernel::normalizePointsKernel<T, dataDim, U, true>
          <<<numEStepBlocks, kGMMThreads, 0, GMMStream>>>(
              dataDevicePtr, fitOffsetCUDA.get(), eStepBlockFitCUDA.get(),
              eStepBlockFirstCUDA.get(), scaleCUDA.get());
      cudaErrChk(cudaGetLastError());
    }

    // the results of all the fits, in the units of the fits
    const size_t slots = (size_t)numFits * numComponents;
    auto download = [&](auto* host, const auto* device, size_t count) {
      cudaErrChk(cudaMemcpyAsync(host, device, sizeof(*host) * count,
                                 cudaMemcpyDeviceToHost, GMMStream));
    };
    download(active.get(), activeCUDA.get(), slots);
    download(logWeight.get(), logWeightCUDA.get(), slots);
    download(mean.get(), meanCUDA.get(), slots * dataDim);
    download(coVariance.get(), coVarianceCUDA.get(), slots * dataDim * dataDim);
    download(state.get(), stateCUDA.get(), numFits);
    cudaErrChk(cudaStreamSynchronize(GMMStream));
  }

  /** A fit of the last initGMM: its components (whether each is active, not
   *  pruned, and their weights, not their logs), its EM steps, and the
   *  log-likelihood of the parameters before its last M step, in the units
   *  of the points (the means and the coVariances of the normalised points
   *  multiplied by scale and scale^2, their density scale^dataDim times
   *  larger). */
  __host__ GMMResult<T, dataDim> getGMMResult(int simuStep, int fit = 0) {
    const int K = numComponents;
    const cudaGMMWeightKernel::GMMFitState<T>& fitState = state[fit];
    GMMResult<T, dataDim> result(simuStep, K);
    result.convergeStep = fitState.step;
    const T s = scale[fit];
    T logLikelihood = fitState.logLikelihood;
    if (normalize)
      logLikelihood -= fitState.totalWeight * dataDim * log(s);
    if (std::isnan(logLikelihood))
      std::cerr << "[!]GMM: LogLikelihood is NaN!"
                << " GMM step: " << fitState.step << std::endl;
    result.logLikelihoodFinal = logLikelihood;

    for (int k = 0; k < K; k++) {
      result.active[k] = active[fit * K + k];
      result.weight[k] = exp(logWeight[fit * K + k]);
    }
    for (int i = 0; i < K * dataDim; i++)
      result.mean[i] = mean[fit * K * dataDim + i] * s;
    for (int i = 0; i < K * dataDim * dataDim; i++)
      result.coVariance[i] =
          coVariance[fit * K * dataDim * dataDim + i] * (s * s);
    return result;
  }

  __host__ ~GMM() { cudaErrChk(cudaFree(dataDevicePtr)); }

private:
  /**
   * One EM iteration of the fits that run: responsibilities (E step), the
   * moments of the points per component, the new parameters and the
   * convergence (M step); asynchronous.
   */
  void emStep() {
    const int K = numComponents;
    cudaGMMWeightKernel::
        eStepKernel<<<numEStepBlocks, kGMMThreads, 0, GMMStream>>>(
            dataDevicePtr, fitOffsetCUDA.get(), eStepBlockFitCUDA.get(),
            eStepBlockFirstCUDA.get(), phaseCUDA.get(), logWeightCUDA.get(),
            meanCUDA.get(), coVarianceDecomposedCUDA.get(),
            normalizerCUDA.get(), K, activeCUDA.get(), gammaCUDA.get(),
            weightedLogLikelihoodCUDA.get());

    cudaGMMWeightKernel::mStepStatisticsKernel<<<numStatisticsBlocks,
                                                 kGMMThreads, 0, GMMStream>>>(
        dataDevicePtr, fitOffsetCUDA.get(), statisticsBlockFitCUDA.get(),
        fitStatisticsBlockCUDA.get(), phaseCUDA.get(), gammaCUDA.get(),
        weightedLogLikelihoodCUDA.get(), meanCUDA.get(), K, activeCUDA.get(),
        partialCUDA.get());

    const int width = K * cudaGMMWeightKernel::kMStepStatistics<dataDim> + 1;
    cudaGMMWeightKernel::mStepFinalizeKernel<T, dataDim>
        <<<numFits, kGMMThreads, width * sizeof(T), GMMStream>>>(
            partialCUDA.get(), fitStatisticsBlockCUDA.get(), phaseCUDA.get(),
            stateCUDA.get(), logWeightCUDA.get(), meanCUDA.get(),
            coVarianceCUDA.get(), coVarianceDecomposedCUDA.get(),
            normalizerCUDA.get(), K, activeCUDA.get(), control);
    cudaErrChk(cudaGetLastError());
  }
};

} // namespace cudaGMMWeight

#endif // CUDA_GMM_CUH
