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
/** Most threads per block of mStepStatisticsKernel. */
inline constexpr int kMaxStatisticsThreads = 1024;
/** Pruning: the weight of a pruned component, and the weight given to the
 *  other components below the threshold. */
inline constexpr double kPrunedWeight = 1e-15;
inline constexpr double kRaisedWeight = 0.05;
/** A non-finite mean is reset to distinct values in [-kResetMeanOffset,
 *  1 - kResetMeanOffset) times the half width of the data range. */
inline constexpr double kResetMeanOffset = 0.3;

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
 *  diagonal, to minVariance on it), raise its variances to minVariance, copy
 *  its upper triangle to the lower one. */
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
    if (!(variance >= minVariance && isfinite(variance)))
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

/**
 * @brief calculate the log likelihood of the data points for all components, to
 * be summed up to get the total log likelihood
 * @details this cuda kernel will be launched once for all data points
 * @param dataCUDAPtr pointer to the data, including numData
 * @param meanVector pointer to the mean vector, number of components * dataDim
 * @param coVarianceDecomp Cholesky factors (choleskyFactor: lower triangle,
 * diagonal inverted), number of components * dataDim * dataDim
 * @param normalizer pointer to the log normalisation of each component,
 * -0.5 * (dataDim * log(2 pi) + log(det coVariance)), number of components
 * @param logLikelihoodForPoints pointer to the logLikelihoodForPoints array, to
 * store the log p(x_i|mean,coVariance) of each data point, used later, number
 * of components * numData
 * @param numComponents number of components
 * @param flagActiveComponents boolean array with flags that indicate which
 * components are active, number of components
 *
 * Not used currently: the EM runs eStepKernel, mStepStatisticsKernel and
 * mStepFinalizeKernel.
 */
template <typename T, int dataDim, typename U>
__global__ void calcLogLikelihoodForPointsKernel(
    const cudaGMMWeight::GMMDataMultiDim<T, dataDim, U>* dataCUDAPtr,
    const T* meanVector, const T* coVarianceDecomp, const T* normalizer,
    T* logLikelihoodForPoints, const int numComponents,
    const bool* flagActiveComponents) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  auto numData = dataCUDAPtr->getNumData();
  if (idx >= numData)
    return;

  for (int component = 0; component < numComponents; component++) {
    auto logLikelihoods = logLikelihoodForPoints + component * numData;
    if (!flagActiveComponents[component]) {
      logLikelihoods[idx] = 0.0;
      continue;
    }
    T xMinusMean[dataDim];
    for (int dim = 0; dim < dataDim; dim++)
      xMinusMean[dim] =
          dataCUDAPtr->getDim(dim)[idx] - meanVector[component * dataDim + dim];
    logLikelihoods[idx] =
        normalizer[component] -
        0.5 * mahalanobis2<T, dataDim>(
                  coVarianceDecomp + component * dataDim * dataDim, xMinusMean);
  }
}

/**
 * @brief calculate the log likelihood of the data points for all components, to
 * be summed up to get the total log likelihood, and the posterior
 * @details this cuda kernel will be launched once for all data points
 *
 * @param dataCUDAPtr pointer to the data, including numData
 * @param logWeightVector pointer to the weight vector, log(weight), number of
 * components
 * @param logLikelihoodForPoints pointer to the log likelihood of the data
 * points for all components, p(x_i|mean,coVariance), number of components *
 * numData
 * @param logLikelihood pointer to the log likelihood(log p(x_i)) of the data
 * points, to be summed up to get the total log likelihood(L or log p(x)),
 * numData
 * @param posterior pointer to the posterior_nk(gamma) of the data points for
 * all components, to be summed up to get the total Posterior(Gamma), number of
 * components * numData
 * @param flagActiveComponents boolean array with flags that indicate which
 * components are active, number of components
 *
 * Not used currently: the EM runs eStepKernel, mStepStatisticsKernel and
 * mStepFinalizeKernel.
 */
template <typename T, int dataDim, typename U>
__global__ void calcLogLikelihoodPxAndposteriorKernel(
    const cudaGMMWeight::GMMDataMultiDim<T, dataDim, U>* dataCUDAPtr,
    const T* logWeightVector, const T* logLikelihoodForPoints, T* logLikelihood,
    T* posterior, const int numComponents, const bool* flagActiveComponents) {

  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  auto numData = dataCUDAPtr->getNumData();
  if (idx >= numData)
    return;

  T maxValue = -INFINITY;
  T sum = 0;

  for (int component = 0; component < numComponents; component++) {
    if (!flagActiveComponents[component])
      continue;
    T logPxComponent =
        logWeightVector[component] +
        logLikelihoodForPoints[component * numData +
                               idx]; // log(weight) +
                                     // log(p(x_i|mean,coVariance))
    if (logPxComponent > maxValue)
      maxValue = logPxComponent;
  }

  for (int component = 0; component < numComponents; component++) {
    if (!flagActiveComponents[component])
      continue;
    T logPxComponent =
        logWeightVector[component] +
        logLikelihoodForPoints[component * numData +
                               idx]; // log(weight) +
                                     // log(p(x_i|mean,coVariance))
    sum += exp(logPxComponent - maxValue);
  }

  logLikelihood[idx] = maxValue + log(sum);

  for (int component = 0; component < numComponents; component++) {
    if (!flagActiveComponents[component]) {
      posterior[component * numData + idx] = 0.0;
      continue;
    }
    posterior[component * numData + idx] -= logLikelihood[idx];
  }
}

/**
 * @brief update the weight of each component
 * @details this cuda kernel will be launched once for all components, In one
 * block. The shared memory should be sizeof(T) * numComponents. The threads
 * of inactive components return before __syncthreads: fix before using it.
 *
 * @param logWeightVector pointer to the old weight vector, log(weight), number
 * of components
 * @param logPosterior pointer to the posterior_k(Gamma), number of components
 * @param flagActiveComponents boolean array with flags that indicate which
 * components are active, number of components
 *
 * Not used currently: the EM runs eStepKernel, mStepStatisticsKernel and
 * mStepFinalizeKernel.
 */
template <typename T>
__global__ void updateWeightKernel(T* logWeightVector, const T* logPosterior,
                                   const int numComponents,
                                   const bool* flagActiveComponents) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= numComponents)
    return;
  if (!flagActiveComponents[idx])
    return;

  extern __shared__ T
      sharedLogMeanTimesPosterior[]; // for each component, the sum
  sharedLogMeanTimesPosterior[threadIdx.x] =
      logWeightVector[idx] +
      logPosterior[idx]; // log(weight_k) + log(Posterior_k)
  __syncthreads();

  T sum = 0;
  for (int i = 0; i < numComponents; i++) {
    if (!flagActiveComponents[i])
      continue;
    sum += exp(sharedLogMeanTimesPosterior[i]);
  }

  logWeightVector[idx] = sharedLogMeanTimesPosterior[idx] - log(sum);

  if (logWeightVector[idx] < log(1e-8))
    logWeightVector[idx] = log(1e-8);
}

/**
 * @brief calculate the new coVariance matrix for each component, to be summed
 * up and divided to get the total coVariance matrix
 * @details this cuda kernel will be launched once for all data points
 *
 * @param dataCUDAPtr pointer to the data, including numData
 * @param logPosterior_nk pointer to the posterior_nk(Gamma), number of
 * components * numData
 * @param logPosterior_k pointer to the posterior_k(Gamma), number of components
 * @param meanVector pointer to the mean vector, number of components * dataDim,
 * just updated
 * @param tempCoVarianceForDataPoints pointer to the coVariance matrix for each
 * data point, number of components * dataNum * dataDim * dataDim
 * @param flagActiveComponents boolean array with flags that indicate which
 * components are active, number of components
 *
 * Not used currently: the EM runs eStepKernel, mStepStatisticsKernel and
 * mStepFinalizeKernel.
 */
template <typename T, int dataDim, typename U>
__global__ void updateCoVarianceKernel(
    const cudaGMMWeight::GMMDataMultiDim<T, dataDim, U>* dataCUDAPtr,
    const T* logPosterior_nk, const T* logPosterior_k, const T* meanVector,
    T* tempCoVarianceForDataPoints, const int numComponents,
    const bool* flagActiveComponents) {

  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  auto numData = dataCUDAPtr->getNumData();
  if (idx >= numData)
    return;

  // for each component
  for (int component = 0; component < numComponents; component++) {
    if (!flagActiveComponents[component])
      continue;
    auto logPosterior_nkComponent = logPosterior_nk + component * numData;
    auto meanComponent = meanVector + component * dataDim;
    auto coVarianceComponent =
        tempCoVarianceForDataPoints + component * numData * dataDim * dataDim;

    // update the coVariance matrix
    T xMinusMean[dataDim]; // (x - mean^(t+1)) vector
    for (int dim = 0; dim < dataDim; dim++) {
      xMinusMean[dim] = dataCUDAPtr->getDim(dim)[idx] - meanComponent[dim];
    }

    for (int i = 0; i < dataDim; i++) {
      for (int j = 0; j < dataDim; j++) {
        const auto elementInMatrix = i * dataDim + j;
        coVarianceComponent[elementInMatrix * numData + idx] =
            exp(logPosterior_nkComponent[idx]) * xMinusMean[i] * xMinusMean[j];
      }
    }
  }
}

/**
 * @brief decompose the coVariance matrix for each component
 * @details this cuda kernel will be launched once for all components
 *
 * @param coVariance pointer to the coVariance matrix, number of components *
 * dataDim * dataDim
 * @param coVarianceDecomp Cholesky factors (choleskyFactor: lower triangle,
 * diagonal inverted), number of components * dataDim * dataDim
 * @param normalizer pointer to the log normalisation of the density,
 * number of components
 * @param flagActiveComponents boolean array with flags that indicate which
 * components are active, number of components
 * @param minPivot smallest square of a diagonal element of the factor (the
 * variance of an axis given the previous ones): a smaller pivot (a nearly
 * singular or non-positive coVariance) is raised to it, and the coVariance
 * with it, so that the factor stays invertible
 */
template <typename T, int dataDim>
__global__ void
decomposeCoVarianceKernel(T* coVariance, T* coVarianceDecomp, T* normalizer,
                          const int numComponents,
                          const bool* flagActiveComponents, const T minPivot) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= numComponents)
    return;
  if (!flagActiveComponents[idx])
    return;

  normalizer[idx] = choleskyFactor<T, dataDim>(
      coVariance + idx * dataDim * dataDim,
      coVarianceDecomp + idx * dataDim * dataDim, minPivot);
}

/**
 * @brief check the coVariance matrix of each component: a non-finite element
 * is reset (to 0 off the diagonal), a variance is raised to minVariance at
 * least, and the matrix is made symmetric. A matrix that is still nearly
 * singular is handled by the pivot floor of decomposeCoVarianceKernel.
 * @details this cuda kernel will be launched once for all components
 *
 * @param coVariance pointer to the coVariance matrix, number of components *
 * dataDim * dataDim
 * @param numComponents number of GMM components
 * @param flagActiveComponents boolean array with flags that indicate which
 * components are active, number of components
 * @param minVariance smallest value of the main diagonal
 */
template <typename T, int dataDim>
__global__ void checkAdjustCoVarianceKernel(T* coVariance,
                                            const int numComponents,
                                            const bool* flagActiveComponents,
                                            const T minVariance) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= numComponents)
    return;
  if (!flagActiveComponents[idx])
    return;

  checkCovariance<T, dataDim>(coVariance + idx * dataDim * dataDim,
                              minVariance);
}

/**
 * @brief Divide the data points by rescaleFactor (normalizeBack == false), or
 * multiply them by it (normalizeBack == true); one thread per point.
 *
 * @param dataCUDAPtr pointer to the data, including numData
 * @param rescaleFactor the rescale factor of all dimensions
 */
template <typename T, int dataDim, typename U, bool normalizeBack>
__global__ void normalizePointsKernel(
    cudaGMMWeight::GMMDataMultiDim<T, dataDim, U>* dataCUDAPtr,
    const T rescaleFactor) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  auto numData = dataCUDAPtr->getNumData();
  if (idx >= numData)
    return;

  for (int dim = 0; dim < dataDim; dim++) {
    if constexpr (normalizeBack)
      dataCUDAPtr->getDim(dim)[idx] *= rescaleFactor;
    else
      dataCUDAPtr->getDim(dim)[idx] /= rescaleFactor;
  }
}

/**
 * @brief normalize mean vector and cov-matrix back such that the range is the
 * original one
 * @details this cuda kernel will be launched once for all components
 *
 * @param meanVector pointer to the mean vector, number of components * dataDim
 * @param coVariance pointer to the coVariance matrix, number of components *
 * dataDim * dataDim
 * @param rescaleFactor the rescale factor of all dimensions
 * @param numComponents number of components, all rescaled (a pruned one keeps
 * the parameters it had, in the same units as the others)
 */
template <typename T, int dataDim>
__global__ void normalizeMeanAndCovBack(T* meanVector, T* coVariance,
                                        const T rescaleFactor,
                                        const int numComponents) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= numComponents)
    return;

  auto meanComponent = meanVector + idx * dataDim;
  auto coVarianceComponent = coVariance + idx * dataDim * dataDim;
  for (int i = 0; i < dataDim; i++)
    meanComponent[i] *= rescaleFactor;
  const T rescaleFactor2 = rescaleFactor * rescaleFactor;
  for (int i = 0; i < dataDim * dataDim; i++)
    coVarianceComponent[i] *= rescaleFactor2;
}

/**
 * @brief prune the active component of smallest weight below the threshold,
 * if any and if it is not the last active one: the other active components
 * below it get kRaisedWeight, the pruned one kPrunedWeight, and the weights of
 * the active ones are normalised
 * @details one block, one thread per component; shared memory sizeof(T) *
 * numComponents + sizeof(bool) * (numComponents + 1)
 *
 * @param logWeightVector pointer to the old weight vector, log(weight), number
 * of components
 * @param weightThreshold the threshold below which the components are pruned
 * @param numComponents num components
 * @param flagActiveComponents boolean array with flags that indicate which
 * components are active plus one element for pruning flag, number of components
 * +1
 */
template <typename T>
__global__ void
pruneOneComponentKernel(T* logWeightVector, const T weightThreshold,
                        const int numComponents, bool* flagActiveComponents) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= numComponents)
    return;

  extern __shared__ char
      sharedMemory[]; // for each component, weight and flagActive

  T* sharedWeights = (T*)sharedMemory; // First segment for weight values
  bool* sharedFlags =
      (bool*)&sharedWeights[numComponents]; // Second segment for boolean flags
  sharedWeights[idx] = logWeightVector[idx];
  sharedFlags[idx] = flagActiveComponents[idx];
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
    flagActiveComponents[numComponents] = sharedFlags[numComponents];
  }
  __syncthreads();

  flagActiveComponents[idx] = sharedFlags[idx];

  if (sharedFlags[numComponents]) {

    // an active component of low weight gets kRaisedWeight, an inactive one
    // kPrunedWeight
    if (flagActiveComponents[idx] &&
        (exp(logWeightVector[idx]) < weightThreshold)) {
      logWeightVector[idx] = log(T(kRaisedWeight));
    }
    if (!flagActiveComponents[idx]) {
      logWeightVector[idx] = log(T(kPrunedWeight));
    }

    sharedWeights[idx] = logWeightVector[idx];
    __syncthreads();

    if (!flagActiveComponents[idx])
      return;

    // normalize weights of active components
    T sum = 0.0;
    for (int i = 0; i < numComponents; i++)
      if (sharedFlags[i])
        sum += exp(sharedWeights[i]);

    logWeightVector[idx] -= log(sum);
  }
}

/**
 * @brief safety check on the GMM components mean vector: reset the mean of
 * every component (active or not) with a non-finite element
 * @details one block, one thread per component; shared memory numComponents
 * booleans
 *
 * @param meanVector pointer to the mean vector, number of components * dataDim
 * @param numComponents num components
 * @param flagActiveComponents boolean array with flags that indicate which
 * components are active, number of components
 * @param flagResetMean set to whether the mean of an active component was
 * reset
 * @param range half width of the range of the data, in the units of the fit
 */
template <typename T, int dataDim>
__global__ void checkMeanValueComponents(T* meanVector, const int numComponents,
                                         const bool* flagActiveComponents,
                                         bool* flagResetMean, const T range) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= numComponents)
    return;

  extern __shared__ char sharedMemory[];
  bool* sharedFlags = (bool*)&sharedMemory;

  auto meanComponent = meanVector + idx * dataDim;
  bool flagNaN = false;
  for (int i = 0; i < dataDim; i++) {
    if (!isfinite(meanComponent[i])) {
      flagNaN = true;
      break;
    }
  }
  sharedFlags[idx] = flagNaN;
  // a distinct mean for each component, inside the range of the data
  if (flagNaN) {
    for (int i = 0; i < dataDim; i++) {
      meanComponent[i] =
          ((T)(i * dataDim + idx) / (T)(dataDim * dataDim + numComponents) -
           T(kResetMeanOffset)) *
          range;
    }
  }

  __syncthreads();

  if (idx == 0) {
    bool reset = false;
    for (int i = 0; i < numComponents; i++)
      reset = reset || (sharedFlags[i] && flagActiveComponents[i]);
    *flagResetMean = reset;
  }
}

// ======= EM step on the data points: E step, M-step statistics, M-step
// finalize (three kernels per iteration) =======

/**
 * @brief E step: the responsibility gamma_kn = w_k p_k(x_n) / p(x_n) of every
 * active component for every point (those of an inactive one are not
 * written), and h_n log p(x_n) for the log-likelihood.
 * @details one thread per point
 *
 * @param data the points and their weights h
 * @param logWeight log of the component weights, numComponents
 * @param mean component means, numComponents * dataDim
 * @param coVarianceDecomp Cholesky factors (choleskyFactor: diagonal
 * inverted), numComponents * dataDim * dataDim
 * @param normalizer log normalisation of the components, numComponents
 * @param numComponents number of components
 * @param flagActiveComponents active components, numComponents
 * @param gamma responsibilities, numComponents * numData (component major),
 * written for the active components
 * @param weightedLogLikelihood h_n log p(x_n), numData
 */
template <typename T, int dataDim, typename U>
__global__ void
eStepKernel(const cudaGMMWeight::GMMDataMultiDim<T, dataDim, U>* data,
            const T* logWeight, const T* mean, const T* coVarianceDecomp,
            const T* normalizer, const int numComponents,
            const bool* flagActiveComponents, T* gamma,
            T* weightedLogLikelihood) {
  const int n = blockIdx.x * blockDim.x + threadIdx.x;
  const int numData = data->getNumData();
  if (n >= numData)
    return;

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
 * @details any number of blocks of a multiple of WARP_SIZE threads, at least
 * kMStepStatistics and at most kMaxStatisticsThreads; a fixed summation order
 * (no atomics). Block b writes
 * to partial[b * (numComponents * S + 1) + k * S + s], S =
 * kMStepStatistics<dataDim>, and the log-likelihood at numComponents * S.
 */
template <typename T, int dataDim, typename U>
__global__ void
mStepStatisticsKernel(const cudaGMMWeight::GMMDataMultiDim<T, dataDim, U>* data,
                      const T* gamma, const T* weightedLogLikelihood,
                      const T* mean, const int numComponents,
                      const bool* flagActiveComponents, T* partial) {
  __shared__ T
      warpSum[kMaxStatisticsThreads / WARP_SIZE][kMStepStatistics<dataDim>];
  const int numData = data->getNumData();
  const int stride = gridDim.x * blockDim.x;
  const int first = blockIdx.x * blockDim.x + threadIdx.x;
  const int lane = threadIdx.x % WARP_SIZE, warp = threadIdx.x / WARP_SIZE;
  const U* h = data->getWeight();
  T* blockPartial =
      partial + blockIdx.x * (numComponents * kMStepStatistics<dataDim> + 1);

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
  for (int n = first; n < numData; n += stride)
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
    for (int n = first; n < numData; n += stride) {
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
 * @brief M step, second part, one block: the block partials summed in order,
 * then thread k updates its active component.
 *
 * With N_k = sum h gamma_k and N the sum over the active components: weight
 * N_k / N (as a log, floored at kMinComponentWeight); mean += S1_k / N_k;
 * coVariance = S2_k / N_k - (S1_k / N_k)(S1_k / N_k)^T (moments about the old
 * mean). A component with N_k = 0 keeps its mean and coVariance and gets the
 * smallest weight, for the pruning; with N = 0 every component is kept. Then
 * the coVariance check (with applyCovarianceCheck) and the Cholesky factor.
 * Thread 0 writes the log-likelihood and N to result.
 * @details one block of numComponents threads at least; dynamic shared memory
 * (numComponents * S + 1) * sizeof(T)
 */
template <typename T, int dataDim>
__global__ void
mStepFinalizeKernel(const T* partial, const int numBlocks, T* logWeight,
                    T* mean, T* coVariance, T* coVarianceDecomp, T* normalizer,
                    const int numComponents, const bool* flagActiveComponents,
                    const bool applyCovarianceCheck, const T minVariance,
                    const T minPivot, T* result) {
  extern __shared__ __align__(sizeof(double)) unsigned char sharedBytes[];
  T* total = reinterpret_cast<T*>(sharedBytes);
  const int width = numComponents * kMStepStatistics<dataDim> + 1;
  for (int i = threadIdx.x; i < width; i += blockDim.x) {
    T sum = 0;
    for (int b = 0; b < numBlocks; b++)
      sum += partial[b * width + i];
    total[i] = sum;
  }
  __syncthreads();

  T weightSum = 0; // N
  for (int j = 0; j < numComponents; j++)
    if (flagActiveComponents[j])
      weightSum += total[j * kMStepStatistics<dataDim>];
  if (threadIdx.x == 0) {
    result[0] = total[width - 1];
    result[1] = weightSum;
  }

  const int k = threadIdx.x;
  if (k >= numComponents || !flagActiveComponents[k])
    return;
  const T* stat = total + k * kMStepStatistics<dataDim>;
  T* m = mean + k * dataDim;
  T* c = coVariance + k * dataDim * dataDim;
  const T weightK = stat[0]; // N_k
  if (weightK > 0 && weightSum > 0) {
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
  } else if (weightSum > 0) {
    logWeight[k] = log(T(kMinComponentWeight));
  }
  if (applyCovarianceCheck)
    checkCovariance<T, dataDim>(c, minVariance);
  normalizer[k] = choleskyFactor<T, dataDim>(
      c, coVarianceDecomp + k * dataDim * dataDim, minPivot);
}

} // namespace cudaGMMWeightKernel

#endif // CUDA_GMM_KERNEL_CUH
