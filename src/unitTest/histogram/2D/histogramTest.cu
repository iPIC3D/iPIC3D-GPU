#include <cmath>
#include <cstdlib>
#include <exception>
#include <iostream>
#include <vector>

#include "velocityHistogram.cuh"

namespace {

/** Bins per axis and range of the 2D test histogram. */
constexpr int kBins = 200;
constexpr velocityHistogram::histogramTypeIn kVmin = -0.2, kVmax = 0.2;
constexpr int kHistogramSize = kBins * kBins;
/** One sample on the upper corner and two out of range. */
constexpr int kExtraSamples = 3;
constexpr int kSampleCount = kHistogramSize + kExtraSamples;
constexpr int kBlockSize = 256;
constexpr double kTolerance = 1e-6;
/** Charge of a sample, and the weight of a charge: one sample weighs 1. */
constexpr velocityHistogram::histogramTypeIn kCharge = 1e-6;
constexpr velocityHistogram::histogramTypeIn kWeightScale = 1 / kCharge;

__global__ void
fillHistogram2D(const int sampleCount,
                const velocityHistogram::histogramTypeIn* u,
                const velocityHistogram::histogramTypeIn* v,
                const velocityHistogram::histogramTypeIn* q,
                velocityHistogram::velocityHistogramCUDA2D* histogram) {
  const int tid = threadIdx.x + blockIdx.x * blockDim.x;
  const int stride = gridDim.x * blockDim.x;
  auto* histogramBuffer = histogram->getHistogramCUDA();

  for (int sample = tid; sample < sampleCount; sample += stride) {
    velocityHistogram::histogramTypeIn data[2] = {u[sample], v[sample]};
    const int bin = histogram->getIndex(data);
    if (bin >= 0) {
      atomicAdd(&histogramBuffer[bin],
                static_cast<velocityHistogram::histogramTypeOut>(
                    fabs(q[sample] * kWeightScale)));
    }
  }
}

} // namespace

static int runTest() {
  using namespace velocityHistogram;

  histogramTypeIn minRange[2] = {kVmin, kVmin};
  histogramTypeIn maxRange[2] = {kVmax, kVmax};
  int binsPerDim[2] = {kBins, kBins};

  // the histogram object and the particles in page-locked memory: they are
  // copied to the device
  velocityHistogramCUDA2D* const histogram =
      newHostPinnedObject<velocityHistogramCUDA2D>(kHistogramSize);
  histogram->setHistogram(minRange, maxRange, binsPerDim);
  cudaErrChk(cudaMemset(histogram->getHistogramCUDA(), 0,
                        kHistogramSize * sizeof(histogramTypeOut)));

  PinnedVector<histogramTypeIn> u(kSampleCount);
  PinnedVector<histogramTypeIn> v(kSampleCount);
  PinnedVector<histogramTypeIn> q(kSampleCount, kCharge);
  std::vector<histogramTypeOut> expected(kHistogramSize, 1.0);

  const histogramTypeIn resolution = (maxRange[0] - minRange[0]) / kBins;
  for (int iy = 0; iy < kBins; ++iy) {
    for (int ix = 0; ix < kBins; ++ix) {
      const int sample = iy * kBins + ix;
      u[sample] = minRange[0] + (ix + 0.5) * resolution;
      v[sample] = minRange[1] + (iy + 0.5) * resolution;
    }
  }

  u[kHistogramSize] = maxRange[0];
  v[kHistogramSize] = maxRange[1];
  q[kHistogramSize] = 2e-6;
  expected.back() += 2.0;

  u[kHistogramSize + 1] = minRange[0] - resolution;
  v[kHistogramSize + 1] = minRange[1];
  q[kHistogramSize + 1] = 100e-6;

  u[kHistogramSize + 2] = minRange[0];
  v[kHistogramSize + 2] = maxRange[1] + resolution;
  q[kHistogramSize + 2] = 100e-6;

  histogramTypeIn* uDevice = nullptr;
  histogramTypeIn* vDevice = nullptr;
  histogramTypeIn* qDevice = nullptr;
  velocityHistogramCUDA2D* histogramDevice = nullptr;

  cudaErrChk(cudaMalloc(&uDevice, kSampleCount * sizeof(histogramTypeIn)));
  cudaErrChk(cudaMalloc(&vDevice, kSampleCount * sizeof(histogramTypeIn)));
  cudaErrChk(cudaMalloc(&qDevice, kSampleCount * sizeof(histogramTypeIn)));
  cudaErrChk(cudaMalloc(&histogramDevice, sizeof(velocityHistogramCUDA2D)));

  cudaErrChk(cudaMemcpy(uDevice, u.data(),
                        kSampleCount * sizeof(histogramTypeIn),
                        cudaMemcpyHostToDevice));
  cudaErrChk(cudaMemcpy(vDevice, v.data(),
                        kSampleCount * sizeof(histogramTypeIn),
                        cudaMemcpyHostToDevice));
  cudaErrChk(cudaMemcpy(qDevice, q.data(),
                        kSampleCount * sizeof(histogramTypeIn),
                        cudaMemcpyHostToDevice));
  cudaErrChk(cudaMemcpy(histogramDevice, histogram,
                        sizeof(velocityHistogramCUDA2D),
                        cudaMemcpyHostToDevice));

  fillHistogram2D<<<getGridSize(kSampleCount, kBlockSize), kBlockSize>>>(
      kSampleCount, uDevice, vDevice, qDevice, histogramDevice);
  cudaErrChk(cudaGetLastError());
  cudaErrChk(cudaDeviceSynchronize());

  histogram->copyHistogramAsync(0);
  cudaErrChk(cudaDeviceSynchronize());

  const auto* actual = histogram->getHistogram();
  bool pass = true;
  for (int i = 0; i < kHistogramSize; ++i) {
    if (std::fabs(static_cast<double>(actual[i] - expected[i])) > kTolerance) {
      std::cout << "Mismatch in 2D histogram at bin " << i
                << ": GPU = " << actual[i] << ", CPU = " << expected[i] << "\n";
      pass = false;
      break;
    }
  }

  deleteHostPinnedObject(histogram);
  cudaErrChk(cudaFree(histogramDevice));
  cudaErrChk(cudaFree(qDevice));
  cudaErrChk(cudaFree(vDevice));
  cudaErrChk(cudaFree(uDevice));

  if (pass) {
    std::cout << "Test passed: 2D histogram bins match expected counts.\n";
  } else {
    std::cout
        << "Test failed: 2D histogram bins do not match expected counts.\n";
  }

  return pass ? 0 : 1;
}

int main() {
  try {
    return runTest();
  } catch (const std::exception& error) {
    std::cerr << "histogram2DTest failed: " << error.what() << "\n";
    return EXIT_FAILURE;
  }
}
