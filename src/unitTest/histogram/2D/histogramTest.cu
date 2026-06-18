#include <cmath>
#include <cstdlib>
#include <exception>
#include <iostream>
#include <vector>

#include "dataAnalysisConfig.cuh"
#include "velocityHistogram.cuh"

namespace {

__global__ void fillHistogram2D(const int sampleCount,
                                const velocityHistogram::histogramTypeIn* u,
                                const velocityHistogram::histogramTypeIn* v,
                                const velocityHistogram::histogramTypeIn* q,
                                velocityHistogram::velocityHistogramCUDA2D* histogram)
{
    const int tid = threadIdx.x + blockIdx.x * blockDim.x;
    const int stride = gridDim.x * blockDim.x;
    auto* histogramBuffer = histogram->getHistogramCUDA();

    for (int sample = tid; sample < sampleCount; sample += stride) {
        velocityHistogram::histogramTypeIn data[2] = {u[sample], v[sample]};
        const int bin = histogram->getIndex(data);
        if (bin >= 0) {
            atomicAdd(&histogramBuffer[bin], static_cast<velocityHistogram::histogramTypeOut>(fabs(q[sample] * 1e6)));
        }
    }
}

} // namespace

static int runTest()
{
    using namespace DAConfig;
    using namespace velocityHistogram;

    constexpr int bins = VELOCITY_HISTOGRAM2D_RES;
    constexpr int histogramSize = bins * bins;
    constexpr int boundaryAndOutOfRangeSamples = 3;
    constexpr int sampleCount = histogramSize + boundaryAndOutOfRangeSamples;

    histogramTypeIn minRange[2] = {MIN_VELOCITY_HIST_E, MIN_VELOCITY_HIST_E};
    histogramTypeIn maxRange[2] = {MAX_VELOCITY_HIST_E, MAX_VELOCITY_HIST_E};
    int binsPerDim[2] = {bins, bins};

    velocityHistogramCUDA2D histogram(histogramSize);
    histogram.setHistogram(minRange, maxRange, binsPerDim);
    cudaErrChk(cudaMemset(histogram.getHistogramCUDA(), 0, histogramSize * sizeof(histogramTypeOut)));

    std::vector<histogramTypeIn> u(sampleCount);
    std::vector<histogramTypeIn> v(sampleCount);
    std::vector<histogramTypeIn> q(sampleCount, 1e-6);
    std::vector<histogramTypeOut> expected(histogramSize, 1.0);

    const histogramTypeIn resolution = (maxRange[0] - minRange[0]) / bins;
    for (int iy = 0; iy < bins; ++iy) {
        for (int ix = 0; ix < bins; ++ix) {
            const int sample = iy * bins + ix;
            u[sample] = minRange[0] + (ix + 0.5) * resolution;
            v[sample] = minRange[1] + (iy + 0.5) * resolution;
        }
    }

    u[histogramSize] = maxRange[0];
    v[histogramSize] = maxRange[1];
    q[histogramSize] = 2e-6;
    expected.back() += 2.0;

    u[histogramSize + 1] = minRange[0] - resolution;
    v[histogramSize + 1] = minRange[1];
    q[histogramSize + 1] = 100e-6;

    u[histogramSize + 2] = minRange[0];
    v[histogramSize + 2] = maxRange[1] + resolution;
    q[histogramSize + 2] = 100e-6;

    histogramTypeIn* uDevice = nullptr;
    histogramTypeIn* vDevice = nullptr;
    histogramTypeIn* qDevice = nullptr;
    velocityHistogramCUDA2D* histogramDevice = nullptr;

    cudaErrChk(cudaMalloc(&uDevice, sampleCount * sizeof(histogramTypeIn)));
    cudaErrChk(cudaMalloc(&vDevice, sampleCount * sizeof(histogramTypeIn)));
    cudaErrChk(cudaMalloc(&qDevice, sampleCount * sizeof(histogramTypeIn)));
    cudaErrChk(cudaMalloc(&histogramDevice, sizeof(velocityHistogramCUDA2D)));

    cudaErrChk(cudaMemcpy(uDevice, u.data(), sampleCount * sizeof(histogramTypeIn), cudaMemcpyHostToDevice));
    cudaErrChk(cudaMemcpy(vDevice, v.data(), sampleCount * sizeof(histogramTypeIn), cudaMemcpyHostToDevice));
    cudaErrChk(cudaMemcpy(qDevice, q.data(), sampleCount * sizeof(histogramTypeIn), cudaMemcpyHostToDevice));
    cudaErrChk(cudaMemcpy(histogramDevice, &histogram, sizeof(velocityHistogramCUDA2D), cudaMemcpyHostToDevice));

    constexpr int blockSize = 256;
    fillHistogram2D<<<getGridSize(sampleCount, blockSize), blockSize>>>(sampleCount, uDevice, vDevice, qDevice, histogramDevice);
    cudaErrChk(cudaGetLastError());
    cudaErrChk(cudaDeviceSynchronize());

    histogram.copyHistogramAsync();
    cudaErrChk(cudaDeviceSynchronize());

    const auto* actual = histogram.getHistogram();
    constexpr double tolerance = 1e-6;
    bool pass = true;
    for (int i = 0; i < histogramSize; ++i) {
        if (std::fabs(static_cast<double>(actual[i] - expected[i])) > tolerance) {
            std::cout << "Mismatch in 2D histogram at bin " << i
                      << ": GPU = " << actual[i]
                      << ", CPU = " << expected[i] << "\n";
            pass = false;
            break;
        }
    }

    cudaErrChk(cudaFree(histogramDevice));
    cudaErrChk(cudaFree(qDevice));
    cudaErrChk(cudaFree(vDevice));
    cudaErrChk(cudaFree(uDevice));

    if (pass) {
        std::cout << "Test passed: 2D histogram bins match expected counts.\n";
    } else {
        std::cout << "Test failed: 2D histogram bins do not match expected counts.\n";
    }

    return pass ? 0 : 1;
}

int main()
{
    try {
        return runTest();
    } catch (const std::exception& error) {
        std::cerr << "histogram2DTest failed: " << error.what() << "\n";
        return EXIT_FAILURE;
    }
}
