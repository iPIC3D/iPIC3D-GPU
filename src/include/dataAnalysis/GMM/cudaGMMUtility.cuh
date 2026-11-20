#ifndef CUDA_GMM_UTILITY_CUH
#define CUDA_GMM_UTILITY_CUH

#include <cmath>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <limits>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "cudaTypeDef.cuh"

namespace cudaGMMWeight {

template <typename T, int dataDim, typename U = int> class GMMDataMultiDim {

private:
  int dim = dataDim;
  int numData;
  T* data[dataDim]; // pointers to the dimensions of the data points
  U* weight;

public:
  // all dim in one array; not used currently
  __host__ GMMDataMultiDim(int numData, T* data, U* weight) {
    this->numData = numData;
    for (int i = 0; i < dataDim; i++) {
      this->data[i] = data + i * numData;
    }
    this->weight = weight;
  }

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

  /** Not used currently. */
  __device__ __host__ int getDim() const { return dim; }

  __device__ __host__ U* getWeight() const { return weight; }
};

template <typename T> struct GMMParam_s {
  int numComponents;
  int maxIteration;
  T threshold; // the threshold for the log likelihood per unit weight
  // the initial components (required)
  T* weightInit;
  T* meanInit;
  T* coVarianceInit;
  // the points lie in [-scale, scale]; with normalize the fit runs on the
  // points, the means and the coVariances divided by scale (its square) and
  // gives them back in the units of the points
  bool normalize;
  T scale;
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

template <typename T> using GMMParam_t = GMMParam_s<T>;

// the result of a fit (T: the type of its parameters); move-only
template <typename T, int dataDim> class GMMResult {

public:
  int simulationStep;
  int numComponents; // number of components

  std::unique_ptr<T[]> weight; // the weight of each component
  std::unique_ptr<T[]> mean;   // the mean of each component
  std::unique_ptr<T[]>
      coVariance; // the coVariance of each component, full, row major

  int convergeStep;
  T logLikelihoodFinal; // the final log likelihood

public:
  /** The record as JSON: a non-finite value (a failed fit) as null. */
  std::string outputString() const {
    std::ostringstream outputStream;
    outputStream << std::scientific // max_digits10 significant digits
                 << std::setprecision(std::numeric_limits<T>::max_digits10 - 1);
    auto number = [&](T value) -> std::ostream& {
      return std::isfinite(value) ? outputStream << value
                                  : outputStream << "null";
    };

    outputStream << "{\n";
    outputStream << "\"simulationStep\": " << simulationStep << ",\n";
    outputStream << "\"numComponent\": " << numComponents << ",\n";
    outputStream << "\"dataDim\": " << dataDim << ",\n";
    outputStream << "\"convergeStep\": " << convergeStep << ",\n";
    outputStream << "\"logLikeliHood\": ";
    number(logLikelihoodFinal) << ",\n";

    outputStream << "\"components\": [\n";
    for (int i = 0; i < numComponents; i++) {
      outputStream << "  {\n";
      outputStream << "    \"weight\": ";
      number(weight[i]) << ",\n";
      outputStream << "    \"mean\": [";
      for (int j = 0; j < dataDim; j++) {
        number(mean[i * dataDim + j]);
        if (j != dataDim - 1) {
          outputStream << ", ";
        }
      }
      outputStream << "],\n";
      outputStream << "    \"coVariance\": [";
      for (int k = 0; k < dataDim * dataDim; k++) {
        number(coVariance[i * dataDim * dataDim + k]);
        if (k != dataDim * dataDim - 1) {
          outputStream << ", ";
        }
      }
      outputStream << "]\n";
      outputStream << "  }";
      if (i != numComponents - 1) {
        outputStream << ",\n";
      } else {
        outputStream << "\n";
      }
    }
    outputStream << "]\n";
    outputStream << "}";

    return outputStream.str();
  }

public:
  GMMResult(const int simuStep, const int numComponents)
      : simulationStep(simuStep), numComponents(numComponents) {

    weight = std::make_unique<T[]>(numComponents);
    mean = std::make_unique<T[]>(numComponents * dataDim);
    coVariance = std::make_unique<T[]>(numComponents * dataDim * dataDim);
  }

  GMMResult(GMMResult&&) noexcept = default;
  GMMResult& operator=(GMMResult&&) noexcept = default;

  GMMResult(const GMMResult&) = delete;
  GMMResult& operator=(const GMMResult&) = delete;

  /** Not used currently. */
  static void
  outputResultArray(const std::vector<GMMResult<T, dataDim>>& resultArray,
                    const std::string outputPath,
                    const std::string metaData = "") {

    // open file or create file
    std::ofstream file(outputPath);
    if (!file.is_open()) {
      throw std::runtime_error("GMMResult: Error: can not open file " +
                               outputPath);
    }

    std::string outputString = "{\n";

    // date and time
    time_t now = time(0);
    tm* ltm = localtime(&now);
    outputString += "\"date\": \"" + std::to_string(1900 + ltm->tm_year) + "-" +
                    std::to_string(1 + ltm->tm_mon) + "-" +
                    std::to_string(ltm->tm_mday) + "\",\n";
    outputString += "\"time\": \"" + std::to_string(ltm->tm_hour) + ":" +
                    std::to_string(ltm->tm_min) + ":" +
                    std::to_string(ltm->tm_sec) + "\",\n";

    // meta data
    outputString += "\"metaData\": \"" + metaData + "\",\n";

    // out put to json file, create the string first, then write to file
    outputString += "\"results\": [\n";
    for (int i = 0; i < resultArray.size(); i++) { // for each result
      outputString += resultArray[i].outputString();
      if (i != resultArray.size() - 1) {
        outputString += ",";
      }
    }
    outputString += "]";

    outputString += "}\n";

    file << outputString;
    file.close();
  }

  /** Not used currently. */
  T getlogLikelihoodFinal() const { return logLikelihoodFinal; }

  ~GMMResult() = default;
};

} // namespace cudaGMMWeight

#endif // CUDA_GMM_UTILITY_CUH
