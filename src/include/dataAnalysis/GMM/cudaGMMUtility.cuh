#ifndef CUDA_GMM_UTILITY_CUH
#define CUDA_GMM_UTILITY_CUH

#include <memory>

#include "cudaTypeDef.cuh"

namespace cudaGMMWeight {

template <typename T, int dataDim, typename U = int> class GMMDataMultiDim {

private:
  int numData;
  T* data[dataDim]; // pointers to the dimensions of the data points
  U* weight;

public:
  // all dim in separate arrays
  __host__ GMMDataMultiDim(int numData, T** data, U* weight) {
    this->numData = numData;
    for (int i = 0; i < dataDim; i++) {
      this->data[i] = data[i];
    }
    this->weight = weight;
  }

  __device__ __host__ T* getDim(int dim) const { return data[dim]; }

  __device__ __host__ int getNumData() const { return numData; }

  __device__ __host__ U* getWeight() const { return weight; }
};

template <typename T> struct GMMParam_t {
  int numComponents;
  int maxIteration;
  T threshold; // the threshold for the log likelihood per unit weight
  // the initial components of each fit, fit after fit (required):
  // numComponents weights, numComponents * dataDim means, numComponents *
  // dataDim^2 coVariances per fit
  const T* weightInit;
  const T* meanInit;
  const T* coVarianceInit;
  // the points of fit f lie in [-scale[f], scale[f]]; with normalize the fit
  // runs on the points, the means and the coVariances divided by scale (its
  // square) and gives them back in the units of the points
  bool normalize;
  const T* scale;
  bool checkCovariance;  // reset non-finite elements, floor the variances
  T covarianceTolerance; // smallest pivot of the Cholesky factorisation
  // with checkCovariance, the smallest variance of each axis given the
  // previous ones (a Cholesky pivot), so on the diagonal too: the density of
  // a component stays below that of a round one of this variance
  T minVariance;
  bool prune; // drop components of weight below pruneThreshold
  T pruneThreshold;

  /** The smallest pivot of the Cholesky factorisation, in the units of the
   *  fit. */
  T minPivot() const {
    return checkCovariance && minVariance > covarianceTolerance
               ? minVariance
               : covarianceTolerance;
  }
};

// the result of a fit (T: the type of its parameters); move-only (its
// arrays)
template <typename T, int dataDim> class GMMResult {

public:
  int simulationStep;

  std::unique_ptr<T[]> weight; // the weight of each component
  std::unique_ptr<T[]> mean;   // the mean of each component
  std::unique_ptr<T[]>
      coVariance; // the coVariance of each component, full, row major

  int convergeStep;
  T logLikelihoodFinal; // the final log likelihood

public:
  GMMResult(const int simuStep, const int numComponents)
      : simulationStep(simuStep) {
    weight = std::make_unique<T[]>(numComponents);
    mean = std::make_unique<T[]>(numComponents * dataDim);
    coVariance = std::make_unique<T[]>(numComponents * dataDim * dataDim);
  }
};

} // namespace cudaGMMWeight

#endif // CUDA_GMM_UTILITY_CUH
