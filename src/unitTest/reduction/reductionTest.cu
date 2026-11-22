/**
 * @file reductionTest.cu
 * @brief Unit test of the block reductions of cudaReduction.cuh against the
 *        CPU: blockReduce (sum, minimum, maximum) and blockExclusiveScan.
 *
 * blockReduce, in blocks of two warps (the smallest), 256 (the histograms')
 * and 512 (the largest): the sums of doubles of integer values and of 32- and
 * 64-bit integers beyond 32 bits (exact whatever the order, so bit for bit),
 * the maximum of 64-bit integers (as the largest |q| of the histograms, the
 * bits of a double), the minimum and the maximum of doubles with +-inf among
 * them. blockExclusiveScan, in
 * blocks of two warps, 256, 1000 (not a power of 2) and 1024 (the scans of
 * the histograms): the exclusive sums and the total of every block. The
 * particle sums of the energy diagnostics (particleSums.cuh) against the CPU.
 */

#include "cudaReduction.cuh"
#include "cudaTypeDef.cuh"
#include "particleSums.cuh"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <iostream>
#include <limits>
#include <random>
#include <string>
#include <vector>

namespace {

/** Blocks of every case, and the most threads of a block of the scan. */
constexpr int kBlocks = 37;
constexpr int kMaxScanThreads = 1024;

int failures = 0;

void report(bool ok, const std::string& what) {
  std::cout << "Test " << (ok ? "passed" : "failed") << ": " << what << "\n";
  failures += !ok;
}

template <unsigned blockSize, typename T, typename Op>
__global__ void blockReduceKernel(const T* in, T* out, Op op) {
  __shared__ T sdata[blockSize];
  const T value = cudaReduction::blockReduce<blockSize>(
      in[blockIdx.x * blockSize + threadIdx.x], sdata, op);
  if (threadIdx.x == 0)
    out[blockIdx.x] = value;
}

__global__ void blockScanKernel(const int* in, int* out, int* total) {
  __shared__ int buffer[kMaxScanThreads];
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  int sum;
  out[i] = cudaReduction::blockExclusiveScan(in[i], buffer, sum);
  if (threadIdx.x == 0)
    total[blockIdx.x] = sum;
}

/** blockReduce with op of every block of blockSize values of data on the
 *  GPU, against cpuOp from identity on the CPU, bit for bit. */
template <unsigned blockSize, typename T, typename Op, typename CpuOp>
void checkReduce(const PinnedVector<T>& data, Op op, CpuOp cpuOp, T identity,
                 const std::string& what) {
  T *in, *out;
  cudaErrChk(cudaMalloc(&in, data.size() * sizeof(T)));
  cudaErrChk(cudaMalloc(&out, kBlocks * sizeof(T)));
  cudaErrChk(cudaMemcpy(in, data.data(), data.size() * sizeof(T),
                        cudaMemcpyHostToDevice));
  blockReduceKernel<blockSize><<<kBlocks, blockSize>>>(in, out, op);
  cudaErrChk(cudaGetLastError());
  PinnedVector<T> gpu(kBlocks);
  cudaErrChk(
      cudaMemcpy(gpu.data(), out, kBlocks * sizeof(T), cudaMemcpyDeviceToHost));
  bool ok = true;
  for (int b = 0; b < kBlocks; b++) {
    T cpu = identity;
    for (unsigned t = 0; t < blockSize; t++)
      cpu = cpuOp(cpu, data[b * blockSize + t]);
    ok = ok && gpu[b] == cpu;
  }
  report(ok, what + " (" + std::to_string(blockSize) + " threads)");
  for (T* buffer : {in, out})
    cudaErrChk(cudaFree(buffer));
}

template <unsigned blockSize> void testReduce() {
  using namespace cudaReduction;
  const size_t n = (size_t)kBlocks * blockSize;
  std::mt19937_64 gen(12345 + blockSize);
  std::uniform_int_distribution<int> small(-1000, 1000);
  std::uniform_int_distribution<unsigned> word(0, 1u << 20);
  std::uniform_int_distribution<unsigned long long> wide(0, 1ull << 52);
  std::uniform_real_distribution<double> uniform(-100, 100);
  const double inf = std::numeric_limits<double>::infinity();
  auto sum = [](auto a, auto b) { return a + b; };

  PinnedVector<double> integers(n);
  for (double& x : integers)
    x = small(gen);
  checkReduce<blockSize>(integers, SumOp{}, sum, 0.0,
                         "sum of doubles of integer values");
  PinnedVector<unsigned> words(n);
  for (unsigned& x : words)
    x = word(gen);
  checkReduce<blockSize>(words, SumOp{}, sum, 0u, "sum of 32-bit integers");
  PinnedVector<unsigned long long> wides(n);
  for (unsigned long long& x : wides)
    x = wide(gen);
  checkReduce<blockSize>(wides, SumOp{}, sum, 0ull,
                         "sum of 64-bit integers beyond 32 bits");
  checkReduce<blockSize>(wides, MaxU64Op{}, MaxU64Op{}, 0ull,
                         "maximum of 64-bit integers");

  PinnedVector<double> values(n);
  for (double& x : values)
    x = uniform(gen);
  for (size_t i = 0; i < n; i += 97)
    values[i] = i % 2 ? inf : -inf;
  checkReduce<blockSize>(
      values, MinOp{}, [](double a, double b) { return b < a ? b : a; }, inf,
      "minimum of doubles with +-inf");
  checkReduce<blockSize>(
      values, MaxOp{}, [](double a, double b) { return b > a ? b : a; }, -inf,
      "maximum of doubles with +-inf");
}

/** blockExclusiveScan in kBlocks blocks of threads. */
void testScan(int threads) {
  const size_t n = (size_t)kBlocks * threads;
  std::mt19937 gen(54321 + threads);
  std::uniform_int_distribution<int> count(0, 100);
  PinnedVector<int> data(n);
  for (int& x : data)
    x = count(gen);
  int *in, *out, *total;
  cudaErrChk(cudaMalloc(&in, n * sizeof(int)));
  cudaErrChk(cudaMalloc(&out, n * sizeof(int)));
  cudaErrChk(cudaMalloc(&total, kBlocks * sizeof(int)));
  cudaErrChk(
      cudaMemcpy(in, data.data(), n * sizeof(int), cudaMemcpyHostToDevice));
  blockScanKernel<<<kBlocks, threads>>>(in, out, total);
  cudaErrChk(cudaGetLastError());
  PinnedVector<int> gpu(n), gpuTotal(kBlocks);
  cudaErrChk(
      cudaMemcpy(gpu.data(), out, n * sizeof(int), cudaMemcpyDeviceToHost));
  cudaErrChk(cudaMemcpy(gpuTotal.data(), total, kBlocks * sizeof(int),
                        cudaMemcpyDeviceToHost));
  bool ok = true;
  for (int b = 0; b < kBlocks; b++) {
    int sum = 0;
    for (int t = 0; t < threads; t++) {
      ok = ok && gpu[b * threads + t] == sum;
      sum += data[b * threads + t];
    }
    ok = ok && gpuTotal[b] == sum;
  }
  report(ok,
         "exclusive scan and total (" + std::to_string(threads) + " threads)");
  for (int* buffer : {in, out, total})
    cudaErrChk(cudaFree(buffer));
}

/** The particle sums of the energy diagnostics (launchParticleSums) of n
 *  random particles, against the CPU: the same values to 1e-12 of the sum of
 *  the absolute terms (another order of the additions). Relativistic, the
 *  speeds reach beyond c, and those particles are left out of the energy and
 *  momentum, not of the charge. */
void testParticleSums(uint32_t n, bool relativistic) {
  const double c = 2.0, qom = -64.0, invC2 = 1.0 / (c * c);
  std::mt19937 gen(777 + n + relativistic);
  std::uniform_real_distribution<double> velocity(-1.3, 1.3);
  std::uniform_real_distribution<double> weight(-2e-3, -1e-3);
  PinnedVector<double> u(n), v(n), w(n), q(n);
  ParticleSums cpu{}, absolute{};
  uint32_t leftOut = 0;
  for (uint32_t p = 0; p < n; p++) {
    u[p] = velocity(gen);
    v[p] = velocity(gen);
    w[p] = velocity(gen);
    q[p] = weight(gen);
    cpu.charge += q[p];
    absolute.charge += std::abs(q[p]);
    const double mass = q[p] / qom;
    const double v2 = u[p] * u[p] + v[p] * v[p] + w[p] * w[p];
    double gamma = 1.0, keFactor = 0.5;
    if (relativistic) {
      if (!(v2 * invC2 < 1.0)) {
        leftOut++;
        continue;
      }
      gamma = 1.0 / std::sqrt(1.0 - v2 * invC2);
      keFactor = gamma * gamma / (gamma + 1.0);
    }
    const double terms[4] = {keFactor * mass * v2, gamma * mass * u[p],
                             gamma * mass * v[p], gamma * mass * w[p]};
    double* const sums[4] = {&cpu.energy, &cpu.momentum[0], &cpu.momentum[1],
                             &cpu.momentum[2]};
    double* const abss[4] = {&absolute.energy, &absolute.momentum[0],
                             &absolute.momentum[1], &absolute.momentum[2]};
    for (int k = 0; k < 4; k++) {
      *sums[k] += terms[k];
      *abss[k] += std::abs(terms[k]);
    }
  }

  std::vector<double*> arrays(4, nullptr);
  const PinnedVector<double>* host[4] = {&u, &v, &w, &q};
  for (int k = 0; k < 4; k++) {
    cudaErrChk(cudaMalloc(&arrays[k], std::max<size_t>(n, 1) * sizeof(double)));
    if (n > 0)
      cudaErrChk(cudaMemcpy(arrays[k], host[k]->data(), n * sizeof(double),
                            cudaMemcpyHostToDevice));
  }
  ParticleSums* device;
  cudaErrChk(cudaMalloc(&device, sizeof(ParticleSums)));
  cudaErrChk(cudaMemset(device, 0, sizeof(ParticleSums)));
  launchParticleSums(arrays[0], arrays[1], arrays[2], arrays[3], n, qom, c,
                     relativistic, device, 0);
  PinnedVector<ParticleSums> gpuSums(1);
  cudaErrChk(cudaMemcpy(gpuSums.data(), device, sizeof(ParticleSums),
                        cudaMemcpyDeviceToHost));
  const ParticleSums& gpu = gpuSums[0];

  const double gpuValues[kParticleSumsValues] = {gpu.energy, gpu.momentum[0],
                                                 gpu.momentum[1],
                                                 gpu.momentum[2], gpu.charge};
  const double cpuValues[kParticleSumsValues] = {cpu.energy, cpu.momentum[0],
                                                 cpu.momentum[1],
                                                 cpu.momentum[2], cpu.charge};
  const double absValues[kParticleSumsValues] = {
      absolute.energy, absolute.momentum[0], absolute.momentum[1],
      absolute.momentum[2], absolute.charge};
  bool ok = !relativistic || n < 1000 || leftOut > 0;
  for (int k = 0; k < kParticleSumsValues; k++)
    ok = ok && std::abs(gpuValues[k] - cpuValues[k]) <= 1e-12 * absValues[k];
  report(ok, "particle sums of " + std::to_string(n) + " particles (" +
                 (relativistic ? "relativistic, " + std::to_string(leftOut) +
                                     " beyond c left out"
                               : std::string("classical")) +
                 ")");
  for (double* array : arrays)
    cudaErrChk(cudaFree(array));
  cudaErrChk(cudaFree(device));
}

} // namespace

int main() {
  try {
    testReduce<2 * WARP_SIZE>();
    testReduce<256>();
    testReduce<512>();
    for (int threads : {2 * WARP_SIZE, 256, 1000, kMaxScanThreads})
      testScan(threads);
    // more particles than one grid of kParticleSumsMaxBlocks blocks covers
    for (bool relativistic : {false, true})
      for (uint32_t n : {0u, 1u, 300007u})
        testParticleSums(n, relativistic);
  } catch (const std::exception& error) {
    std::cerr << "reductionTest failed: " << error.what() << std::endl;
    return EXIT_FAILURE;
  }
  return failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
