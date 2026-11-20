#include "velocityHistogram.cuh"

#include "cudaReduction.cuh"

#include <algorithm>
#include <cmath>

namespace velocityHistogram {

__global__ void weightRangeKernel(const histogramTypeIn* q, int n,
                                  double* partial);
__global__ void weightScaleKernel(const double* partial, double* scale);
__global__ void binKernel(const histogramTypeIn* u, const histogramTypeIn* v,
                          const histogramTypeIn* w, int n, BinGrid grid,
                          unsigned* key, int* tileCount);
__global__ void tileBlockScanKernel(const int* tileCount, int* tileBlockOffset,
                                    int* tileTotal);
__global__ void tileScanKernel(const int* tileTotal, int* tileStart,
                               int* tileChunkStart, int* tileChunkCount,
                               HistogramChunk* chunks, int* numChunks);
__global__ void scatterKernel(const unsigned* key, const histogramTypeIn* q,
                              int n, const int* tileStart,
                              const int* tileBlockOffset, const double* scale,
                              unsigned short* tileBin,
                              unsigned long long* fixedWeight);
__global__ void accumulateKernel(const HistogramChunk* chunks,
                                 const int* numChunks,
                                 const unsigned short* tileBin,
                                 const unsigned long long* fixedWeight,
                                 unsigned long long* partialTile);
__global__ void mergeTilesKernel(const int* tileChunkStart,
                                 const int* tileChunkCount,
                                 const unsigned long long* partialTile,
                                 unsigned long long* binSum);
__global__ void binSumToHistogramKernel(const unsigned long long* binSum,
                                        const double* scale,
                                        histogramTypeOut* histogram);

__global__ void countNonEmptyKernel(const histogramTypeOut* bins,
                                    int* blockCount);
__global__ void scanBinBlocksKernel(int* blockCount);
__global__ void
scatterNonEmptyKernel(BinGrid grid, const histogramTypeOut* bins,
                      const int* blockOffset, histogramTypeIn* centerU,
                      histogramTypeIn* centerV, histogramTypeIn* centerW,
                      histogramTypeOut* weight);

__host__ void velocityHistogram3D::init(velocitySoA* pclArray,
                                        histogramTypeIn vmin,
                                        histogramTypeIn vmax,
                                        cudaStream_t stream) {
  if (!fixedRange)
    dataRange(pclArray, stream);
  // a range that bins: a positive, finite width
  auto bins = [](histogramTypeIn lo, histogramTypeIn hi) {
    return hi - lo > 0 && std::isfinite(hi - lo);
  };
  const histogramTypeIn half = (vmax - vmin) / 2;
  for (int d = 0; d < 3; d++) {
    // without fixedRange: the range of the finite velocities, or the fixed
    // width about their value if they have no spread
    histogramTypeIn lo = vmin, hi = vmax;
    if (!fixedRange) {
      const histogramTypeIn low = velocityRange[d], high = velocityRange[3 + d];
      if (bins(low, high)) {
        lo = low;
        hi = high;
      } else if (low == high && bins(low - half, low + half)) {
        lo = low - half;
        hi = low + half;
      }
    }
    grid.min[d] = lo;
    grid.max[d] = hi;
    grid.resolution[d] = (hi - lo) / binsAlong(d);
    grid.invResolution[d] = binsAlong(d) / (hi - lo);
  }
  accumulate(pclArray, stream);
}

__host__ void velocityHistogram3D::freeScratch() {
  for (void* buffer :
       {(void*)keyCUDA, (void*)tileBinCUDA, (void*)fixedWeightCUDA,
        (void*)chunksCUDA, (void*)partialTileCUDA})
    cudaErrChk(cudaFree(buffer));
  keyCUDA = nullptr;
  tileBinCUDA = nullptr;
  fixedWeightCUDA = nullptr;
  chunksCUDA = nullptr;
  partialTileCUDA = nullptr;
  scratchParticles = 0;
}

__host__ void velocityHistogram3D::reserveScratch(int particles) {
  if (particles <= scratchParticles)
    return;
  freeScratch();
  // a margin: the particle count drifts up between analyses
  const int size = std::min(particles + particles / 4, kBatchParticles);
  cudaErrChk(cudaMalloc(&keyCUDA, sizeof(unsigned) * size));
  cudaErrChk(cudaMalloc(&tileBinCUDA, sizeof(unsigned short) * size));
  cudaErrChk(cudaMalloc(&fixedWeightCUDA, sizeof(unsigned long long) * size));
  cudaErrChk(cudaMalloc(&chunksCUDA, sizeof(HistogramChunk) * maxChunks(size)));
  cudaErrChk(cudaMalloc(&partialTileCUDA, sizeof(unsigned long long) *
                                              maxChunks(size) *
                                              kAccumTileBins));
  scratchParticles = size;
}

__host__ void velocityHistogram3D::accumulate(velocitySoA* pclArray,
                                              cudaStream_t stream) {
  const int n = pclArray->getNOP();
  const histogramTypeIn* u = pclArray->getElement(0);
  const histogramTypeIn* v = pclArray->getElement(1);
  const histogramTypeIn* w = pclArray->getElement(2);
  const histogramTypeIn* q = pclArray->getElement(3);
  const int batch = std::min(n, kBatchParticles);
  reserveScratch(batch);

  cudaErrChk(cudaMemsetAsync(binSumCUDA, 0,
                             sizeof(unsigned long long) * kNumBins, stream));
  weightRangeKernel<<<kHistogramBlocks, kHistogramThreads, 0, stream>>>(
      q, n, weightPartialCUDA);
  weightScaleKernel<<<1, WARP_SIZE, 0, stream>>>(weightPartialCUDA,
                                                 weightScaleCUDA);

  for (int first = 0; first < n; first += kBatchParticles) {
    const int count = std::min(kBatchParticles, n - first);
    binKernel<<<kHistogramBlocks, kHistogramThreads, 0, stream>>>(
        u + first, v + first, w + first, count, grid, keyCUDA, tileCountCUDA);
    tileBlockScanKernel<<<kAccumTiles, kHistogramBlocks, 0, stream>>>(
        tileCountCUDA, tileBlockOffsetCUDA, tileTotalCUDA);
    tileScanKernel<<<1, kAccumTiles, 0, stream>>>(
        tileTotalCUDA, tileStartCUDA, tileChunkStartCUDA, tileChunkCountCUDA,
        chunksCUDA, numChunksCUDA);
    scatterKernel<<<kHistogramBlocks, kHistogramThreads, 0, stream>>>(
        keyCUDA, q + first, count, tileStartCUDA, tileBlockOffsetCUDA,
        weightScaleCUDA, tileBinCUDA, fixedWeightCUDA);
    accumulateKernel<<<maxChunks(batch), kHistogramThreads, 0, stream>>>(
        chunksCUDA, numChunksCUDA, tileBinCUDA, fixedWeightCUDA,
        partialTileCUDA);
    mergeTilesKernel<<<getGridSize(kAccumTiles * kAccumTileBins,
                                   kHistogramThreads),
                       kHistogramThreads, 0, stream>>>(
        tileChunkStartCUDA, tileChunkCountCUDA, partialTileCUDA, binSumCUDA);
  }
  binSumToHistogramKernel<<<getGridSize(kNumBins, kHistogramThreads),
                            kHistogramThreads, 0, stream>>>(
      binSumCUDA, weightScaleCUDA, binsCUDA);
  cudaErrChk(cudaGetLastError()); // a launch error of the passes above
}

__host__ int velocityHistogram3D::compactNonEmpty(cudaStream_t stream) {
  countNonEmptyKernel<<<kCompactBlocks, kCompactThreads, 0, stream>>>(
      binsCUDA, blockCountCUDA);
  scanBinBlocksKernel<<<1, kCompactBlocks, 0, stream>>>(blockCountCUDA);
  cudaErrChk(cudaGetLastError());
  cudaErrChk(cudaMemcpyAsync(nonEmptyCountHost, blockCountCUDA + kCompactBlocks,
                             sizeof(int), cudaMemcpyDeviceToHost, stream));
  cudaErrChk(cudaStreamSynchronize(stream));
  const int count = *nonEmptyCountHost;

  if (count > nonEmptyCapacity) { // grow only, with a margin
    const int capacity = std::min(count + count / 4, kNumBins);
    nonEmptyCapacity = 0; // until the buffers are there
    for (auto*& center : nonEmptyCenter) {
      cudaErrChk(cudaFree(center));
      center = nullptr;
      cudaErrChk(cudaMalloc(&center, sizeof(histogramTypeIn) * capacity));
    }
    cudaErrChk(cudaFree(nonEmptyWeight));
    nonEmptyWeight = nullptr;
    cudaErrChk(
        cudaMalloc(&nonEmptyWeight, sizeof(histogramTypeOut) * capacity));
    nonEmptyCapacity = capacity;
  }
  if (count > 0) {
    scatterNonEmptyKernel<<<kCompactBlocks, kCompactThreads, 0, stream>>>(
        grid, binsCUDA, blockCountCUDA, nonEmptyCenter[0], nonEmptyCenter[1],
        nonEmptyCenter[2], nonEmptyWeight);
    cudaErrChk(cudaGetLastError());
    cudaErrChk(cudaStreamSynchronize(stream));
  }
  return count;
}

__host__ void velocityHistogram3D::dataRange(velocitySoA* pclArray,
                                             cudaStream_t stream) {
  using namespace cudaReduction;
  const int n = pclArray->getNOP();
  const int blocks = std::min(
      getGridSize(std::max(n / kRangeParticlesPerThread, 1), kRangeThreads),
      kRangeMaxBlocks);
  histogramTypeIn* minPartial = reductionTempArrayCUDA;
  histogramTypeIn* maxPartial = reductionTempArrayCUDA + 3 * kRangeMaxBlocks;
  const size_t shared = kRangeThreads * sizeof(histogramTypeIn);

  for (int d = 0; d < 3; d++) { // UVW
    const histogramTypeIn* velocity = pclArray->getElement(d);
    reduceMin<histogramTypeIn, kRangeThreads>
        <<<blocks, kRangeThreads, shared, stream>>>(
            velocity, minPartial + d * kRangeMaxBlocks, n);
    reduceMinWarp<histogramTypeIn><<<1, WARP_SIZE, 0, stream>>>(
        minPartial + d * kRangeMaxBlocks, reductionResultCUDA + d, blocks);
    reduceMax<histogramTypeIn, kRangeThreads>
        <<<blocks, kRangeThreads, shared, stream>>>(
            velocity, maxPartial + d * kRangeMaxBlocks, n);
    reduceMaxWarp<histogramTypeIn><<<1, WARP_SIZE, 0, stream>>>(
        maxPartial + d * kRangeMaxBlocks, reductionResultCUDA + 3 + d, blocks);
  }
  cudaErrChk(cudaGetLastError());
  cudaErrChk(cudaMemcpyAsync(velocityRange, reductionResultCUDA,
                             sizeof(velocityRange), cudaMemcpyDeviceToHost,
                             stream));
  cudaErrChk(cudaStreamSynchronize(stream));
}

} // namespace velocityHistogram
