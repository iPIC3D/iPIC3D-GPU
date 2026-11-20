/**
 * @file reductionTest.cu
 * @brief Unit test of the block reductions of cudaReduction.cuh against the
 *        CPU: the minimum and maximum of the finite values (float and
 *        double) and an integer sum.
 *
 * Block sizes: two warps (the smallest), 256 (the velocity histogram's) and
 * 512 (the largest). Sizes: 0, 1, fewer than a block, several strides of the
 * grid with a remainder, and 2^24 (the most blocks). Values: random finite
 * ones; the same with NaN and +-inf among them (left out); all non-finite
 * (the result is then -inf for the maximum, +inf for the minimum). The
 * integer sum checks blockReduce with SumOp, as the charge count of the
 * velocity histogram uses it.
 */

#include "cudaReduction.cuh"
#include "cudaTypeDef.cuh"

#include <cmath>
#include <cstdlib>
#include <exception>
#include <iostream>
#include <limits>
#include <random>
#include <string>
#include <vector>

namespace {

/** Blocks at most and values per thread, as the data range of the velocity
 *  histogram launches them. */
constexpr int kMaxBlocks = 1024;
constexpr int kValuesPerThread = 16;
/** Sizes of the cases: 0, 1, below a block, a prime (several strides of
 *  the grid of blocksFor and a remainder), the most blocks. */
const size_t kSizes[] = {0, 1, 100, 1000003, (size_t)1 << 24};

int failures = 0;

void report(bool ok, const std::string& what) {
  std::cout << "Test " << (ok ? "passed" : "failed") << ": " << what << "\n";
  failures += !ok;
}

int blocksFor(size_t n, int blockSize) {
  const int perBlock = blockSize * kValuesPerThread;
  return std::min(std::max((int)((n + perBlock - 1) / perBlock), 1),
                  kMaxBlocks);
}

/** GPU minimum and maximum of the finite values of data. */
template <typename T, unsigned blockSize>
void gpuMinMax(const std::vector<T>& data, T& minimum, T& maximum) {
  const size_t n = data.size();
  const int blocks = blocksFor(n, blockSize);
  T *input, *partial, *result;
  cudaErrChk(cudaMalloc(&input, std::max(n, (size_t)1) * sizeof(T)));
  cudaErrChk(cudaMalloc(&partial, blocks * sizeof(T)));
  cudaErrChk(cudaMalloc(&result, 2 * sizeof(T)));
  if (n > 0)
    cudaErrChk(
        cudaMemcpy(input, data.data(), n * sizeof(T), cudaMemcpyHostToDevice));
  cudaReduction::reduceMin<T, blockSize>
      <<<blocks, blockSize, blockSize * sizeof(T)>>>(input, partial, n);
  cudaReduction::reduceMinWarp<T><<<1, WARP_SIZE>>>(partial, result, blocks);
  cudaReduction::reduceMax<T, blockSize>
      <<<blocks, blockSize, blockSize * sizeof(T)>>>(input, partial, n);
  cudaReduction::reduceMaxWarp<T>
      <<<1, WARP_SIZE>>>(partial, result + 1, blocks);
  cudaErrChk(cudaGetLastError());
  T host[2];
  cudaErrChk(cudaMemcpy(host, result, 2 * sizeof(T), cudaMemcpyDeviceToHost));
  minimum = host[0];
  maximum = host[1];
  for (T* buffer : {input, partial, result})
    cudaErrChk(cudaFree(buffer));
}

template <typename T, unsigned blockSize>
void checkMinMax(const std::vector<T>& data, const std::string& what) {
  T cpuMin = std::numeric_limits<T>::infinity(), cpuMax = -cpuMin;
  for (T x : data)
    if (std::isfinite(x)) {
      cpuMin = std::min(cpuMin, x);
      cpuMax = std::max(cpuMax, x);
    }
  T gpuMin, gpuMax;
  gpuMinMax<T, blockSize>(data, gpuMin, gpuMax);
  report(gpuMin == cpuMin && gpuMax == cpuMax,
         what + " (" + std::to_string(data.size()) + " values, " +
             std::to_string(blockSize) + " threads)");
}

template <typename T, unsigned blockSize>
void testType(const std::string& name) {
  std::mt19937 gen(12345);
  std::uniform_real_distribution<T> uniform(-100, 100);
  const T nan = std::numeric_limits<T>::quiet_NaN();
  const T inf = std::numeric_limits<T>::infinity();
  for (size_t n : kSizes) {
    std::vector<T> data(n);
    for (T& x : data)
      x = uniform(gen);
    checkMinMax<T, blockSize>(data, name + " finite");
    for (size_t i = 0; i < n; i += 7)
      data[i] = i % 3 == 0 ? nan : i % 3 == 1 ? inf : -inf;
    checkMinMax<T, blockSize>(data, name + " with NaN and inf");
    for (size_t i = 0; i < n; i++)
      data[i] = i % 2 ? nan : inf;
    checkMinMax<T, blockSize>(data, name + " none finite");
  }
}

/** The integer sum of reduceSum (blockReduce with SumOp), exact. */
template <unsigned blockSize> void testIntegerSum() {
  for (size_t n : kSizes) {
    std::vector<int> data(n);
    long long cpu = 0;
    for (size_t i = 0; i < n; i++)
      cpu += data[i] = (int)(i % 5);
    const int blocks = blocksFor(n, blockSize);
    int *input, *partial, *result;
    cudaErrChk(cudaMalloc(&input, std::max(n, (size_t)1) * sizeof(int)));
    cudaErrChk(cudaMalloc(&partial, blocks * sizeof(int)));
    cudaErrChk(cudaMalloc(&result, sizeof(int)));
    if (n > 0)
      cudaErrChk(cudaMemcpy(input, data.data(), n * sizeof(int),
                            cudaMemcpyHostToDevice));
    cudaReduction::reduceSum<int, blockSize>
        <<<blocks, blockSize, blockSize * sizeof(int)>>>(input, partial, n);
    cudaReduction::reduceSumWarp<int>
        <<<1, WARP_SIZE>>>(partial, result, blocks);
    cudaErrChk(cudaGetLastError());
    int gpu = 0;
    cudaErrChk(cudaMemcpy(&gpu, result, sizeof(int), cudaMemcpyDeviceToHost));
    report(gpu == cpu, "int sum (" + std::to_string(n) + " values, " +
                           std::to_string(blockSize) + " threads)");
    for (int* buffer : {input, partial, result})
      cudaErrChk(cudaFree(buffer));
  }
}

template <unsigned blockSize> void testBlockSize() {
  testType<float, blockSize>("float");
  testType<double, blockSize>("double");
  testIntegerSum<blockSize>();
}

} // namespace

int main() {
  try {
    testBlockSize<2 * WARP_SIZE>();
    testBlockSize<256>();
    testBlockSize<512>();
  } catch (const std::exception& error) {
    std::cerr << "reductionTest failed: " << error.what() << std::endl;
    return EXIT_FAILURE;
  }
  return failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
