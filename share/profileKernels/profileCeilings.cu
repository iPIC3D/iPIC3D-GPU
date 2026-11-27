/**
 * @file profileCeilings.cu
 * @brief The measured ceilings of the rooflines (plot_roofline.py): the
 * device-memory bandwidth of a streaming kernel and the FP64 FMA throughput
 * of the GPU, printed as a line of text and one JSON object.
 *
 * Empirical reference rates, not specification peaks: what simple kernels
 * reach on the GPU at its current clocks.  The streaming arrays are at least
 * twice the L2 cache, so that they come from device memory.
 */

#include <algorithm>
#include <cstddef>
#include <cstdio>
#include <limits>
#include <string>

#include "KernelProfile.cuh"

namespace {

/** Bytes of each of the three streaming arrays (c = a + b). */
constexpr std::size_t kStreamBytes = std::size_t(256) << 20;
/** Threads per block of the measurement kernels. */
constexpr int kThreads = 256;
/** Blocks per multiprocessor of the measurement kernels. */
constexpr int kBlocksPerSM = 8;
/** Iterations of the FMA kernel. */
constexpr int kFmaIterations = 4096;
/** Independent FMA chains per thread, enough to fill the FP64 pipeline. */
constexpr int kFmaChains = 8;
/** Timed launches of each measurement; the fastest counts. */
constexpr int kTimedLaunches = 5;

__global__ void streamAdd(const double* a, const double* b, double* c,
                          std::size_t n) {
  const std::size_t stride = std::size_t(gridDim.x) * blockDim.x;
  for (std::size_t i = blockIdx.x * std::size_t(blockDim.x) + threadIdx.x;
       i < n; i += stride)
    c[i] = a[i] + b[i];
}

__global__ void fmaChains(double* out, int iterations, double a, double b) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  double x[kFmaChains];
#pragma unroll
  for (int j = 0; j < kFmaChains; ++j)
    x[j] = (i + j) * 1e-8;
#pragma unroll 1
  for (int k = 0; k < iterations; ++k) {
#pragma unroll
    for (int j = 0; j < kFmaChains; ++j)
      x[j] = ::fma(a, x[j], b);
  }
  double sum = 0.0;
#pragma unroll
  for (int j = 0; j < kFmaChains; ++j)
    sum += x[j];
  out[i] = sum; // keeps the chains
}

/** The fastest of kTimedLaunches launches, after one untimed, in seconds. */
template <typename Launch>
double fastestSeconds(cudaStream_t stream, const Launch& launch) {
  cudaEvent_t start, stop;
  cudaErrChk(cudaEventCreate(&start));
  cudaErrChk(cudaEventCreate(&stop));
  launch();
  cudaErrChk(cudaGetLastError());
  float fastest = std::numeric_limits<float>::infinity();
  for (int r = 0; r < kTimedLaunches; ++r) {
    cudaErrChk(cudaEventRecord(start, stream));
    launch();
    cudaErrChk(cudaEventRecord(stop, stream));
    cudaErrChk(cudaEventSynchronize(stop));
    float ms = 0.0f;
    cudaErrChk(cudaEventElapsedTime(&ms, start, stop));
    fastest = std::min(fastest, ms);
  }
  cudaErrChk(cudaEventDestroy(start));
  cudaErrChk(cudaEventDestroy(stop));
  return fastest * 1e-3;
}

} // namespace

int main() {
  int device = 0;
  cudaErrChk(cudaSetDevice(device));
  cudaDeviceProp properties;
  cudaErrChk(cudaGetDeviceProperties(&properties, device));
  if (properties.l2CacheSize > 0 &&
      kStreamBytes < 2 * std::size_t(properties.l2CacheSize)) {
    std::fprintf(stderr, "[profileCeilings] the streaming arrays are smaller "
                         "than twice the L2 cache\n");
    return 1;
  }
  const int blocks = std::max(1, properties.multiProcessorCount) * kBlocksPerSM;
  const std::size_t n = kStreamBytes / sizeof(double);

  cudaStream_t stream;
  cudaErrChk(cudaStreamCreate(&stream));
  double *a, *b, *c, *sink;
  cudaErrChk(cudaMalloc(&a, kStreamBytes));
  cudaErrChk(cudaMalloc(&b, kStreamBytes));
  cudaErrChk(cudaMalloc(&c, kStreamBytes));
  cudaErrChk(
      cudaMalloc(&sink, std::size_t(blocks) * kThreads * sizeof(double)));
  // nonzero inputs: no zero-only memory traffic
  cudaErrChk(cudaMemsetAsync(a, 0x3f, kStreamBytes, stream));
  cudaErrChk(cudaMemsetAsync(b, 0x3e, kStreamBytes, stream));

  const double streamSeconds = fastestSeconds(
      stream, [&] { streamAdd<<<blocks, kThreads, 0, stream>>>(a, b, c, n); });
  const double fmaSeconds = fastestSeconds(stream, [&] {
    fmaChains<<<blocks, kThreads, 0, stream>>>(sink, kFmaIterations, 1.0000001,
                                               1e-7);
  });
  const double bandwidth = 3 * kStreamBytes / streamSeconds;
  const double fp64 = 2.0 * blocks * kThreads * double(kFmaIterations) *
                      kFmaChains / fmaSeconds;

  const std::string name = profileKernels::deviceName();
  std::printf("%s: %d SMs, L2 %d MiB; measured: device memory %.0f GB/s "
              "(c = a + b), FP64 %.0f GFLOP/s (FMA chains)\n",
              name.c_str(), properties.multiProcessorCount,
              properties.l2CacheSize >> 20, bandwidth * 1e-9, fp64 * 1e-9);
  std::printf("{\"device\":%s,\"compute_capability\":\"%d.%d\","
              "\"multiprocessors\":%d,\"l2_bytes\":%d,"
              "\"stream_bytes_per_launch\":%zu,"
              "\"bandwidth_bytes_per_second\":%.10g,"
              "\"fp64_flops_per_second\":%.10g}\n",
              profileKernels::jsonString(name).c_str(), properties.major,
              properties.minor, properties.multiProcessorCount,
              properties.l2CacheSize, 3 * kStreamBytes, bandwidth, fp64);

  cudaErrChk(cudaFree(a));
  cudaErrChk(cudaFree(b));
  cudaErrChk(cudaFree(c));
  cudaErrChk(cudaFree(sink));
  cudaErrChk(cudaStreamDestroy(stream));
  return 0;
}
