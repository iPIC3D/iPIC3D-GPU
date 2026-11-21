#ifndef CUDA_GMM_KERNEL_CUH
#define CUDA_GMM_KERNEL_CUH

#include "cudaTypeDef.cuh"

namespace cudaGMMWeight {
template <typename T, int dataDim, typename U> class GMMDataMultiDim;
}

namespace cudaGMMWeightKernel {

/** Number of M-step statistics of a component: the sum of the weights, and
 *  the dataDim first and dataDim (dataDim + 1) / 2 distinct second moments
 *  about its mean. */
template <int dataDim>
inline constexpr int kMStepStatistics =
    1 + dataDim + dataDim * (dataDim + 1) / 2;
/** Smallest component weight after an M step. */
inline constexpr double kMinComponentWeight = 1e-8;
/** Threads per block of the EM kernels: also the most components. */
inline constexpr int kGMMThreads = 256;
static_assert(kGMMThreads % 64 == 0, "whole warps on NVIDIA and AMD");
/** Pruning: the weight of a pruned component, and the share of the equal
 *  weight 1 / numComponents given to the other components below the
 *  threshold (the threshold at least): 0.05 with 10 components; not at the
 *  iteration limit, where one EM step follows only. */
inline constexpr double kPrunedWeight = 1e-15;
inline constexpr double kRaisedShare = 0.5;
/** EM steps at most between two prunings; a converged fit is checked at
 *  once. */
inline constexpr int kPruneInterval = 10;
/** EM steps before the fit may converge: at time step 0, when the
 *  distributions are exactly Gaussian, the fit otherwise stops after 2-3
 *  iterations, far from the maximum of the likelihood. */
inline constexpr int kMinStepsToConverge = 6;
/** The phase of a fit: running, one last EM step after a pruning at the
 *  iteration limit, done. */
inline constexpr int kFitRunning = 0;
inline constexpr int kFitFinalStep = 1;
inline constexpr int kFitDone = 2;

/** Where a fit is in its EM iterations (mStepFinalizeKernel). */
template <typename T> struct GMMFitState {
  T logLikelihood;    // of the last EM step, before its M step
  T logLikelihoodOld; // of the step before; -inf at first and after a restart
  T totalWeight;      // of the points, in the last EM step
  int step;           // EM steps run
  int sinceCheck;     // EM steps since the last pruning check
};

/** The settings of the fits that the EM iterations use, by value. */
template <typename T> struct GMMControl {
  int maxIteration;
  T threshold; // of the log-likelihood gain per unit weight
  bool checkCovariance;
  T minVariance;
  T minPivot;
  bool prune;
  T pruneThreshold;
};

/** |y|^2 with L y = b: b^T (L L^T)^-1 b, with L the factor of
 *  choleskyFactor (its diagonal inverted). */
template <typename T, int dataDim>
__device__ __forceinline__ T mahalanobis2(const T* L, const T* b) {
  T y[dataDim];
  T norm2 = 0;
  for (int i = 0; i < dataDim; i++) {
    T sum = 0;
    for (int j = 0; j < i; j++)
      sum += L[i * dataDim + j] * y[j];
    y[i] = (b[i] - sum) * L[i * dataDim + i];
    norm2 += y[i] * y[i];
  }
  return norm2;
}

/** Reset the non-finite elements of a coVariance matrix (to 0 off the
 *  diagonal, to minVariance on it) and copy its upper triangle to the lower
 *  one; choleskyFactor then raises its variances (pivots) to minPivot, at
 *  least minVariance. */
template <typename T, int dataDim>
__device__ __forceinline__ void checkCovariance(T* coVariance,
                                                const T minVariance) {
  for (int i = 0; i < dataDim; i++) {
    for (int j = i + 1; j < dataDim; j++) {
      T& upper = coVariance[i * dataDim + j];
      if (!isfinite(upper))
        upper = 0;
      coVariance[j * dataDim + i] = upper;
    }
    T& variance = coVariance[i * dataDim + i];
    if (!isfinite(variance))
      variance = minVariance;
  }
}

/**
 * Cholesky factor L (lower triangle, the rest 0) of a coVariance matrix,
 * whose lower triangle is read, stored with its diagonal inverted (1 / L_ii:
 * mahalanobis2 multiplies). A pivot below minPivot (a nearly singular or
 * non-positive matrix, or a NaN variance) is raised to it, so that L stays
 * invertible, and the variance of its row with it: the coVariance stays
 * L L^T, the matrix of the density. A non-finite element off the diagonal
 * stays (checkCovariance resets it).
 * @return the log normalisation of the density,
 * -0.5 * (dataDim * log(2 pi) + log(det coVariance))
 */
template <typename T, int dataDim>
__device__ __forceinline__ T choleskyFactor(T* coVariance, T* L,
                                            const T minPivot) {
  for (int i = 0; i < dataDim * dataDim; i++)
    L[i] = 0;
  T logDeterminant = 0;
  for (int row = 0; row < dataDim; ++row) {
    T squares = 0; // left of the diagonal
    for (int j = 0; j < row; j++)
      squares += L[row * dataDim + j] * L[row * dataDim + j];
    T pivot = coVariance[row * dataDim + row] - squares;
    if (!(pivot > minPivot)) { // also NaN: the factored variance is written
      pivot = minPivot;
      coVariance[row * dataDim + row] = minPivot + squares;
    }
    const T diagonal = sqrt(pivot);
    const T inverse = 1 / diagonal;
    L[row * dataDim + row] = inverse;
    logDeterminant += 2 * log(diagonal);
    for (int i = row + 1; i < dataDim; ++i) { // below the diagonal element
      T lowerElementSum = 0;
      for (int column = 0; column < row; column++)
        lowerElementSum += L[i * dataDim + column] * L[row * dataDim + column];
      L[i * dataDim + row] =
          (coVariance[i * dataDim + row] - lowerElementSum) * inverse;
    }
  }
  return -0.5 * (dataDim * log(2.0 * M_PI) + logDeterminant);
}

/** The coVariance of a component checked (checkCovariance, with
 *  control.checkCovariance) and factored (choleskyFactor, pivots raised to
 *  control.minPivot) into coVarianceDecomp. @return its log normalisation. */
template <typename T, int dataDim>
__device__ __forceinline__ T factorCoVariance(T* coVariance,
                                              T* coVarianceDecomp,
                                              const GMMControl<T>& control) {
  if (control.checkCovariance)
    checkCovariance<T, dataDim>(coVariance, control.minVariance);
  return choleskyFactor<T, dataDim>(coVariance, coVarianceDecomp,
                                    control.minPivot);
}

/**
 * @brief The initial coVariance of every active component of the fits that
 * run, checked and factored (factorCoVariance), with its log normalisation;
 * a fit already done (without points) keeps its components as they are.
 * @details one block of numComponents threads per fit, component k of fit f
 * at f * numComponents + k; the arrays as those of mStepFinalizeKernel
 */
template <typename T, int dataDim>
__global__ void factorCoVarianceKernel(T* coVariance, T* coVarianceDecomp,
                                       T* normalizer, const int numComponents,
                                       const bool* flagActiveComponents,
                                       const int* fitPhase,
                                       const GMMControl<T> control) {
  const int idx = blockIdx.x * numComponents + threadIdx.x;
  if (fitPhase[blockIdx.x] == kFitDone || !flagActiveComponents[idx])
    return;
  normalizer[idx] = factorCoVariance<T, dataDim>(
      coVariance + idx * dataDim * dataDim,
      coVarianceDecomp + idx * dataDim * dataDim, control);
}

/**
 * @brief Divide the points of each fit by its scale (normalizeBack ==
 * false), or multiply them by it (normalizeBack == true).
 * @details the blocks of the E step: block b on the points blockFirst[b] +
 * threadIdx.x of fit blockFit[b], below fitOffset[fit + 1]
 *
 * @param dataCUDAPtr pointer to the data, including numData
 * @param scale the scale of each fit
 */
template <typename T, int dataDim, typename U, bool normalizeBack>
__global__ void normalizePointsKernel(
    cudaGMMWeight::GMMDataMultiDim<T, dataDim, U>* dataCUDAPtr,
    const int* fitOffset, const int* blockFit, const int* blockFirst,
    const T* scale) {
  const int fit = blockFit[blockIdx.x];
  const int idx = blockFirst[blockIdx.x] + threadIdx.x;
  if (idx >= fitOffset[fit + 1])
    return;

  const T rescaleFactor = scale[fit];
  for (int dim = 0; dim < dataDim; dim++) {
    if constexpr (normalizeBack)
      dataCUDAPtr->getDim(dim)[idx] *= rescaleFactor;
    else
      dataCUDAPtr->getDim(dim)[idx] /= rescaleFactor;
  }
}

/**
 * @brief prune the active component of smallest weight below the threshold,
 * if any and if it is not the last active one: with raise, the other
 * active components below it get kRaisedShare / numComponents (the
 * threshold at least); the pruned one gets kPrunedWeight, and the weights of
 * the active ones are normalised
 * @details called by every thread of a block of numComponents threads at
 * least, the components of one fit; scratch: numComponents * sizeof(T) +
 * numComponents + 1 bytes of shared memory, free again on return
 *
 * @param logWeightVector the weights of the components, log(weight)
 * @param weightThreshold the threshold below which the components are pruned
 * @param numComponents number of components
 * @param flagActiveComponents the active components
 * @param raise raise the other components below the threshold: not at the
 * iteration limit, where one EM step follows only
 * @return whether a component was pruned, in every thread
 */
template <typename T>
__device__ bool pruneOneComponent(T* logWeightVector, const T weightThreshold,
                                  const int numComponents,
                                  bool* flagActiveComponents, const bool raise,
                                  unsigned char* scratch) {
  T* sharedWeights = reinterpret_cast<T*>(scratch);
  bool* sharedFlags = reinterpret_cast<bool*>(sharedWeights + numComponents);
  const int idx = threadIdx.x;
  if (idx < numComponents) {
    sharedWeights[idx] = logWeightVector[idx];
    sharedFlags[idx] = flagActiveComponents[idx];
  }
  __syncthreads();

  if (idx == 0) {
    int smallest = -1, active = 0;
    for (int i = 0; i < numComponents; i++) {
      active += sharedFlags[i];
      if (sharedFlags[i] && exp(sharedWeights[i]) < weightThreshold &&
          (smallest < 0 || sharedWeights[i] < sharedWeights[smallest]))
        smallest = i;
    }
    const bool prune = smallest >= 0 && active > 1;
    if (prune)
      sharedFlags[smallest] = false;
    sharedFlags[numComponents] = prune;
  }
  __syncthreads();

  const bool prune = sharedFlags[numComponents];
  if (prune) {
    // an active component of low weight gets a raised weight (with raise),
    // at least the threshold before the normalisation below, an inactive one
    // kPrunedWeight
    if (idx < numComponents) {
      flagActiveComponents[idx] = sharedFlags[idx];
      if (raise && sharedFlags[idx] &&
          exp(logWeightVector[idx]) < weightThreshold)
        logWeightVector[idx] =
            log(fmax(T(kRaisedShare) / numComponents, weightThreshold));
      if (!sharedFlags[idx])
        logWeightVector[idx] = log(T(kPrunedWeight));
      sharedWeights[idx] = logWeightVector[idx];
    }
    __syncthreads();

    // normalize weights of active components
    if (idx < numComponents && sharedFlags[idx]) {
      T sum = 0.0;
      for (int i = 0; i < numComponents; i++)
        if (sharedFlags[i])
          sum += exp(sharedWeights[i]);
      logWeightVector[idx] -= log(sum);
    }
  }
  __syncthreads(); // the scratch is free again
  return prune;
}

// ======= EM step on the points of several fits: E step, M-step statistics,
// M-step finalize (three kernels per iteration) =======
// Fit f has the points [fitOffset[f], fitOffset[f + 1]) and the components
// [f K, (f + 1) K) of every array of components (K = numComponents). The
// kernels skip the fits whose phase is kFitDone.

/**
 * @brief E step: the responsibility gamma_kn = w_k p_k(x_n) / p(x_n) of every
 * active component for every point (those of an inactive one are not
 * written), and h_n log p(x_n) for the log-likelihood.
 * @details one thread per point: block b on the points blockFirst[b] +
 * threadIdx.x of fit blockFit[b]
 *
 * @param data the points and their weights h
 * @param logWeight log of the component weights
 * @param mean component means, K * dataDim per fit
 * @param coVarianceDecomp Cholesky factors (choleskyFactor: diagonal
 * inverted), K * dataDim * dataDim per fit
 * @param normalizer log normalisation of the components
 * @param numComponents number of components of a fit
 * @param flagActiveComponents active components
 * @param gamma responsibilities, numComponents * numData (component major),
 * written for the active components
 * @param weightedLogLikelihood h_n log p(x_n), numData
 */
template <typename T, int dataDim, typename U>
__global__ void
eStepKernel(const cudaGMMWeight::GMMDataMultiDim<T, dataDim, U>* data,
            const int* fitOffset, const int* blockFit, const int* blockFirst,
            const int* fitPhase, const T* logWeight, const T* mean,
            const T* coVarianceDecomp, const T* normalizer,
            const int numComponents, const bool* flagActiveComponents, T* gamma,
            T* weightedLogLikelihood) {
  const int fit = blockFit[blockIdx.x];
  const int n = blockFirst[blockIdx.x] + threadIdx.x;
  if (fitPhase[fit] == kFitDone || n >= fitOffset[fit + 1])
    return;
  const int numData = data->getNumData();
  logWeight += fit * numComponents;
  mean += fit * numComponents * dataDim;
  coVarianceDecomp += fit * numComponents * dataDim * dataDim;
  normalizer += fit * numComponents;
  flagActiveComponents += fit * numComponents;

  T x[dataDim];
  for (int d = 0; d < dataDim; d++)
    x[d] = data->getDim(d)[n];

  // a_k = log w_k + log p_k(x), kept in gamma (-inf if not finite: a broken
  // component takes no point); log p(x) by log-sum-exp
  T maxA = -INFINITY;
  for (int k = 0; k < numComponents; k++) {
    if (!flagActiveComponents[k])
      continue;
    T xMinusMean[dataDim];
    for (int d = 0; d < dataDim; d++)
      xMinusMean[d] = x[d] - mean[k * dataDim + d];
    T a = logWeight[k] + normalizer[k] -
          0.5 * mahalanobis2<T, dataDim>(
                    coVarianceDecomp + k * dataDim * dataDim, xMinusMean);
    if (!isfinite_fast(a))
      a = -INFINITY;
    gamma[k * numData + n] = a;
    if (a > maxA)
      maxA = a;
  }
  // exp(a_k - maxA) in gamma, then divided by their sum
  T sum = 0;
  if (maxA > -INFINITY)
    for (int k = 0; k < numComponents; k++)
      if (flagActiveComponents[k]) {
        const T e = exp(gamma[k * numData + n] - maxA);
        gamma[k * numData + n] = e;
        sum += e;
      }
  const T logP = maxA + log(sum); // -inf if no component has the point
  const T inverseSum = 1 / sum;

  for (int k = 0; k < numComponents; k++)
    if (flagActiveComponents[k])
      gamma[k * numData + n] =
          maxA > -INFINITY ? gamma[k * numData + n] * inverseSum : T(0);
  weightedLogLikelihood[n] = data->getWeight()[n] * logP;
}

/**
 * @brief M step, first part: per block, the sums over its points of
 * h gamma_kn (1, x - mean_k, upper triangle of (x - mean_k)(x - mean_k)^T)
 * for every active component, about its current mean (0 for an inactive
 * one), and of h log p(x) (the log-likelihood).
 * @details blocks of kGMMThreads threads (at least kMStepStatistics); fit f
 * has the blocks [fitStatBlock[f], fitStatBlock[f + 1]) (blockFit gives the
 * fit of a block), which walk its points with their own grid stride; a fixed
 * summation order (no atomics). Block b writes to partial[b * (numComponents
 * * S + 1) + k * S + s], S = kMStepStatistics<dataDim>, and the
 * log-likelihood at numComponents * S.
 */
template <typename T, int dataDim, typename U>
__global__ void
mStepStatisticsKernel(const cudaGMMWeight::GMMDataMultiDim<T, dataDim, U>* data,
                      const int* fitOffset, const int* blockFit,
                      const int* fitStatBlock, const int* fitPhase,
                      const T* gamma, const T* weightedLogLikelihood,
                      const T* mean, const int numComponents,
                      const bool* flagActiveComponents, T* partial) {
  static_assert(kMStepStatistics<dataDim> <= kGMMThreads,
                "a thread per statistic in the block sums");
  __shared__ T warpSum[kGMMThreads / WARP_SIZE][kMStepStatistics<dataDim>];
  const int fit = blockFit[blockIdx.x];
  if (fitPhase[fit] == kFitDone) // the same for the whole block
    return;
  const int numData = data->getNumData();
  const int firstBlock = fitStatBlock[fit];
  const int stride = (fitStatBlock[fit + 1] - firstBlock) * blockDim.x;
  const int first =
      fitOffset[fit] + (blockIdx.x - firstBlock) * blockDim.x + threadIdx.x;
  const int end = fitOffset[fit + 1];
  const int lane = threadIdx.x % WARP_SIZE, warp = threadIdx.x / WARP_SIZE;
  const U* h = data->getWeight();
  mean += fit * numComponents * dataDim;
  flagActiveComponents += fit * numComponents;
  T* blockPartial =
      partial +
      (size_t)blockIdx.x * (numComponents * kMStepStatistics<dataDim> + 1);

  // the block sums of the count values of every thread into out
  auto blockSum = [&](const T* value, const int count, T* out) {
    for (int s = 0; s < count; s++) {
      const T v = warp_reduce_sum(value[s]); // in lane 0
      if (lane == 0)
        warpSum[warp][s] = v;
    }
    __syncthreads();
    if (threadIdx.x < count) {
      T sum = 0;
      for (int w = 0; w < blockDim.x / WARP_SIZE; w++)
        sum += warpSum[w][threadIdx.x];
      out[threadIdx.x] = sum;
    }
    __syncthreads();
  };

  T logLikelihood = 0;
  for (int n = first; n < end; n += stride)
    logLikelihood += weightedLogLikelihood[n];
  blockSum(&logLikelihood, 1,
           blockPartial + numComponents * kMStepStatistics<dataDim>);

  for (int k = 0; k < numComponents; k++) {
    if (!flagActiveComponents[k]) { // the same for the whole block
      for (int s = threadIdx.x; s < kMStepStatistics<dataDim>; s += blockDim.x)
        blockPartial[k * kMStepStatistics<dataDim> + s] = 0;
      continue;
    }
    T sums[kMStepStatistics<dataDim>] = {};
    for (int n = first; n < end; n += stride) {
      const T g = h[n] * gamma[k * numData + n];
      T diff[dataDim];
      for (int d = 0; d < dataDim; d++)
        diff[d] = data->getDim(d)[n] - mean[k * dataDim + d];
      sums[0] += g;
      int s = 1;
      for (int i = 0; i < dataDim; i++)
        sums[s++] += g * diff[i];
      for (int i = 0; i < dataDim; i++)
        for (int j = i; j < dataDim; j++)
          sums[s++] += g * diff[i] * diff[j];
    }
    blockSum(sums, kMStepStatistics<dataDim>,
             blockPartial + k * kMStepStatistics<dataDim>);
  }
}

/**
 * @brief M step, second part, one block per fit: the partials of its blocks
 * summed in order, then thread k updates its active component; then the
 * convergence of the fit and its pruning.
 *
 * With N_k = sum h gamma_k and N the sum over the active components: weight
 * N_k / N (as a log, floored at kMinComponentWeight); mean += S1_k / N_k;
 * coVariance = S2_k / N_k - (S1_k / N_k)(S1_k / N_k)^T (moments about the old
 * mean). A component with N_k = 0 keeps its mean and coVariance and gets the
 * smallest weight, for the pruning; with N = 0 or not finite every component
 * is kept. Then
 * the coVariance check (with checkCovariance) and the Cholesky factor.
 *
 * The fit converges when the log-likelihood (of the parameters before this M
 * step) per unit weight gains less than the threshold, or decreases, from
 * kMinStepsToConverge steps on, or at once when it is NaN or when N is 0 or
 * not finite. At convergence, every
 * kPruneInterval steps and at the iteration limit, the smallest component
 * below the threshold is pruned (with prune), which restarts the EM (the
 * log-likelihoods are not comparable) and, but at the limit, raises the other
 * components below the threshold (pruneOneComponent). The fit is done when
 * it converged without a restart, or at the iteration limit, after one more
 * step if it restarted. The means stay finite: each is a weighted mean of the
 * points, or the one before (the fit stops when N is not finite).
 * @details one block of numComponents threads at least per fit; dynamic
 * shared memory (numComponents * S + 1) * sizeof(T)
 */
template <typename T, int dataDim>
__global__ void
mStepFinalizeKernel(const T* partial, const int* fitStatBlock, int* fitPhase,
                    GMMFitState<T>* fitState, T* logWeight, T* mean,
                    T* coVariance, T* coVarianceDecomp, T* normalizer,
                    const int numComponents, bool* flagActiveComponents,
                    const GMMControl<T> control) {
  extern __shared__ __align__(sizeof(double)) unsigned char sharedBytes[];
  __shared__ bool check, atLimit;
  bool converged = false; // in thread 0
  const int fit = blockIdx.x;
  const int phase = fitPhase[fit];
  if (phase == kFitDone) // the same for the whole block
    return;
  T* total = reinterpret_cast<T*>(sharedBytes);
  const int width = numComponents * kMStepStatistics<dataDim> + 1;
  const int firstBlock = fitStatBlock[fit];
  const int numBlocks = fitStatBlock[fit + 1] - firstBlock;
  for (int i = threadIdx.x; i < width; i += blockDim.x) {
    T sum = 0;
    for (int b = 0; b < numBlocks; b++)
      sum += partial[(size_t)(firstBlock + b) * width + i];
    total[i] = sum;
  }
  __syncthreads();

  logWeight += fit * numComponents;
  mean += fit * numComponents * dataDim;
  coVariance += fit * numComponents * dataDim * dataDim;
  coVarianceDecomp += fit * numComponents * dataDim * dataDim;
  normalizer += fit * numComponents;
  flagActiveComponents += fit * numComponents;

  T weightSum = 0; // N
  for (int j = 0; j < numComponents; j++)
    if (flagActiveComponents[j])
      weightSum += total[j * kMStepStatistics<dataDim>];
  // a finite positive N (a bin weight that overflowed to infinity gives an
  // infinite or NaN one: the fit stops as without weight)
  const bool weighted = weightSum > 0 && weightSum < INFINITY;
  const T logLikelihood = total[width - 1];

  const int k = threadIdx.x;
  if (k < numComponents && flagActiveComponents[k]) {
    const T* stat = total + k * kMStepStatistics<dataDim>;
    T* m = mean + k * dataDim;
    T* c = coVariance + k * dataDim * dataDim;
    const T weightK = stat[0]; // N_k
    if (weightK > 0 && weighted) {
      T shift[dataDim];
      for (int d = 0; d < dataDim; d++)
        shift[d] = stat[1 + d] / weightK;
      int s = 1 + dataDim;
      for (int i = 0; i < dataDim; i++)
        for (int j = i; j < dataDim; j++, s++) {
          c[i * dataDim + j] = stat[s] / weightK - shift[i] * shift[j];
          c[j * dataDim + i] = c[i * dataDim + j];
        }
      for (int d = 0; d < dataDim; d++)
        m[d] += shift[d];
      logWeight[k] = log(fmax(weightK / weightSum, T(kMinComponentWeight)));
    } else if (weighted) {
      logWeight[k] = log(T(kMinComponentWeight));
    }
    normalizer[k] = factorCoVariance<T, dataDim>(
        c, coVarianceDecomp + k * dataDim * dataDim, control);
  }
  __syncthreads(); // the components updated, total free

  if (threadIdx.x == 0) {
    GMMFitState<T> state = fitState[fit];
    state.logLikelihood = logLikelihood;
    state.totalWeight = weightSum;
    check = atLimit = false;
    if (phase == kFitFinalStep) {
      state.step++;
      fitPhase[fit] = kFitDone;
    } else {
      const bool done = isnan(logLikelihood) ||
                        ((fabs(logLikelihood - state.logLikelihoodOld) <
                              control.threshold * weightSum ||
                          logLikelihood < state.logLikelihoodOld) &&
                         state.step >= kMinStepsToConverge);
      state.logLikelihoodOld = logLikelihood;
      state.step++;
      state.sinceCheck++;
      converged = done || !weighted;
      atLimit = state.step >= control.maxIteration;
      check = converged || state.sinceCheck == kPruneInterval || atLimit;
    }
    fitState[fit] = state;
  }
  __syncthreads();
  if (!check) // the same for the whole block
    return;

  const bool restart =
      control.prune &&
      pruneOneComponent(logWeight, control.pruneThreshold, numComponents,
                        flagActiveComponents, !atLimit, sharedBytes);
  if (threadIdx.x == 0) {
    GMMFitState<T>& state = fitState[fit];
    // a restart: the log-likelihood of the fewer components is not
    // comparable with the last one
    if (restart)
      state.logLikelihoodOld = -INFINITY;
    state.sinceCheck = 0;
    fitPhase[fit] = converged && !restart ? kFitDone
                    : state.step >= control.maxIteration
                        ? (restart ? kFitFinalStep : kFitDone)
                        : kFitRunning;
  }
}

} // namespace cudaGMMWeightKernel

#endif // CUDA_GMM_KERNEL_CUH
