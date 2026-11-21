#include "velocityHistogram.cuh"

#include <algorithm>
#include <sstream>
#include <stdexcept>

namespace velocityHistogram {

/** The bins along u, v, w of the histogram of a rank and of a macrocell. */
inline constexpr int kRankBins[3] = {DAConfig::VELOCITY_HISTOGRAM3D_RES_1,
                                     DAConfig::VELOCITY_HISTOGRAM3D_RES_2,
                                     DAConfig::VELOCITY_HISTOGRAM3D_RES_3};
inline constexpr int kMacrocellBinsPerAxis[3] = {
    kMacrocellBinsU, kMacrocellBinsV, kMacrocellBinsW};

__global__ void rankRangeKernel(const histogramTypeIn* u,
                                const histogramTypeIn* v,
                                const histogramTypeIn* w,
                                const histogramTypeIn* q, int n,
                                bool fixedRange, ParticleRange* partial);
__global__ void regionGridKernel(const ParticleRange* partial, int partials,
                                 int binsU, int binsV, int binsW,
                                 histogramTypeIn vmin, histogramTypeIn vmax,
                                 bool fixedRange, BinGrid* grids,
                                 double* scale);
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
                                        const double* scale, int regionBins,
                                        int n, histogramTypeOut* histogram);

__global__ void macrocellRangeKernel(
    const histogramTypeIn* u, const histogramTypeIn* v,
    const histogramTypeIn* w, const histogramTypeIn* q, const int* cellStart,
    const int* cellCount, macrocell::MacrocellGeometry geometry, int slices,
    histogramTypeIn vmin, histogramTypeIn vmax, bool fixedRange, bool walk,
    ParticleRange* partial, BinGrid* grids, double* scale);
__global__ void
macrocellAccumulateKernel(const histogramTypeIn* u, const histogramTypeIn* v,
                          const histogramTypeIn* w, const histogramTypeIn* q,
                          const int* cellStart, const int* cellCount,
                          macrocell::MacrocellGeometry geometry, int slices,
                          const BinGrid* grids, const double* scale,
                          histogramTypeOut* bins, unsigned long long* binSum);

__global__ void binCentersKernel(const BinGrid* grids, const int* index,
                                 const int* offsets, int blocksPerRegion,
                                 histogramTypeIn* centerU,
                                 histogramTypeIn* centerV,
                                 histogramTypeIn* centerW);

// ======= histogramRegions =======

histogramRegions::histogramRegions(int regions, const int bins[3],
                                   bool fixedRange,
                                   const std::string& regionsText)
    : regions(regions), regionBins(bins[0] * bins[1] * bins[2]),
      bins{bins[0], bins[1], bins[2]}, fixedRange(fixedRange),
      gridHost(regions), regionsText(regionsText), sparse(regions, regionBins) {
  cudaErrChk(cudaMalloc(&weightScaleCUDA, sizeof(double) * 2 * regions));
  cudaErrChk(cudaMalloc(&binsCUDA, sizeof(histogramTypeOut) * totalBins()));
  cudaErrChk(cudaMalloc(&gridCUDA, sizeof(BinGrid) * regions));
}

histogramRegions::~histogramRegions() {
  for (void* buffer : {(void*)nonEmptyCenter[0], (void*)nonEmptyCenter[1],
                       (void*)nonEmptyCenter[2], (void*)gridCUDA,
                       (void*)binsCUDA, (void*)weightScaleCUDA})
    cudaErrChk(cudaFree(buffer)); // null for the unused ones
}

int histogramRegions::compactNonEmpty(cudaStream_t stream) {
  if (!compacted)
    sparse.compact(binsCUDA, stream);
  compacted = true;
  return sparse.count();
}

histogramTypeIn* const*
histogramRegions::getNonEmptyCenters(cudaStream_t stream) {
  const int count = compactNonEmpty(stream);
  if (count > centerCapacity) { // grow only, with a margin
    centerCapacity = (int)std::min((size_t)count + count / 4, totalBins());
    for (auto*& center : nonEmptyCenter) {
      cudaErrChk(cudaFree(center));
      center = nullptr;
      cudaErrChk(cudaMalloc(&center, sizeof(histogramTypeIn) * centerCapacity));
    }
  }
  if (count > 0) {
    binCentersKernel<<<regions * sparse.blocksPerRegion(), kCenterThreads, 0,
                       stream>>>(gridCUDA, sparse.indices(),
                                 sparse.deviceOffsets(),
                                 sparse.blocksPerRegion(), nonEmptyCenter[0],
                                 nonEmptyCenter[1], nonEmptyCenter[2]);
    cudaErrChk(cudaGetLastError());
    cudaErrChk(cudaStreamSynchronize(stream));
  }
  return nonEmptyCenter;
}

std::string histogramRegions::describe(const std::string& rank,
                                       std::vector<std::string>& layout) const {
  std::ostringstream binsText;
  binsText << "  \"bins\": {\"u\": " << bins[0] << ", \"v\": " << bins[1]
           << ", \"w\": " << bins[2] << "},\n";
  return sparse.describe(
      rank, regionsText, binsText.str(), "ranges", "\"regions\", 6",
      kHistogramRecordMagic,
      "one record per analysis cycle and species of VelocityHistGMMSpecies "
      "with a non-empty bin: ranges[r] = [umin, vmin, wmin, umax, vmax, wmax] "
      "of region r (the rank, or macrocell m = (mz*My + my)*Mx + mx) and "
      "counts[r] its non-empty bins; weights region after region, by "
      "increasing bin b = iu + bins_u * (iv + bins_v * iw) within one, bin i "
      "of an axis centred at min + (i + 0.5) * (max - min) / bins; a weight "
      "is the sum of |q| weight_scale of the particles of its bin; the empty "
      "bins are not written",
      layout);
}

void histogramRegions::writeRecord(RecordFile& file, int species, int cycle,
                                   cudaStream_t stream) {
  compactNonEmpty(stream);
  std::vector<double> ranges;
  for (const BinGrid& g : gridHost)
    ranges.insert(ranges.end(),
                  {g.min[0], g.min[1], g.min[2], g.max[0], g.max[1], g.max[2]});
  sparse.writeRecord(file, species, cycle, ranges.data(), ranges.size(),
                     stream);
}

// ======= velocityHistogram3D =======

velocityHistogram3D::velocityHistogram3D(bool fixedRange)
    : histogramRegions(1, kRankBins, fixedRange,
                       macrocell::regionsDescription(nullptr)) {
  cudaErrChk(cudaMalloc(&binSumCUDA, sizeof(unsigned long long) * kNumBins));
  cudaErrChk(
      cudaMalloc(&rangePartialCUDA, sizeof(ParticleRange) * kHistogramBlocks));
  cudaErrChk(
      cudaMalloc(&tileCountCUDA, sizeof(int) * kHistogramBlocks * kAccumTiles));
  cudaErrChk(cudaMalloc(&tileBlockOffsetCUDA,
                        sizeof(int) * kHistogramBlocks * kAccumTiles));
  cudaErrChk(cudaMalloc(&tileTotalCUDA, sizeof(int) * kAccumTiles));
  cudaErrChk(cudaMalloc(&tileStartCUDA, sizeof(int) * kAccumTiles));
  cudaErrChk(cudaMalloc(&tileChunkStartCUDA, sizeof(int) * kAccumTiles));
  cudaErrChk(cudaMalloc(&tileChunkCountCUDA, sizeof(int) * kAccumTiles));
  cudaErrChk(cudaMalloc(&numChunksCUDA, sizeof(int)));
}

velocityHistogram3D::~velocityHistogram3D() {
  for (void* buffer :
       {(void*)rangePartialCUDA, (void*)binSumCUDA, (void*)tileCountCUDA,
        (void*)tileBlockOffsetCUDA, (void*)tileTotalCUDA, (void*)tileStartCUDA,
        (void*)tileChunkStartCUDA, (void*)tileChunkCountCUDA,
        (void*)numChunksCUDA})
    cudaErrChk(cudaFree(buffer));
  freeScratch();
}

__host__ void velocityHistogram3D::init(velocitySoA* pclArray,
                                        histogramTypeIn vmin,
                                        histogramTypeIn vmax,
                                        cudaStream_t stream) {
  binsChanged();
  // the grid and the weight scale of the rank, the grid on the host too (the
  // binning, the files, the GMM): one pass over the particles, then the
  // blocks merged
  rankRangeKernel<<<kHistogramBlocks, kHistogramThreads, 0, stream>>>(
      pclArray->getElement(0), pclArray->getElement(1), pclArray->getElement(2),
      pclArray->getElement(3), pclArray->getNOP(), fixedRange,
      rangePartialCUDA);
  regionGridKernel<<<1, kGridThreads, 0, stream>>>(
      rangePartialCUDA, kHistogramBlocks, kRankBins[0], kRankBins[1],
      kRankBins[2], vmin, vmax, fixedRange, gridCUDA, weightScaleCUDA);
  cudaErrChk(cudaGetLastError());
  cudaErrChk(cudaMemcpyAsync(gridHost.data(), gridCUDA, sizeof(BinGrid),
                             cudaMemcpyDeviceToHost, stream));
  cudaErrChk(cudaStreamSynchronize(stream));
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

  for (int first = 0; first < n; first += kBatchParticles) {
    const int count = std::min(kBatchParticles, n - first);
    binKernel<<<kHistogramBlocks, kHistogramThreads, 0, stream>>>(
        u + first, v + first, w + first, count, gridHost[0], keyCUDA,
        tileCountCUDA);
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
      binSumCUDA, weightScaleCUDA, kNumBins, kNumBins, binsCUDA);
  cudaErrChk(cudaGetLastError()); // a launch error of the passes above
}

// ======= macrocellHistogram3D =======

macrocellHistogram3D::macrocellHistogram3D(
    const macrocell::MacrocellPartition& part, bool fixedRange)
    : histogramRegions(part.M, kMacrocellBinsPerAxis, fixedRange,
                       macrocell::regionsDescription(&part)),
      geometry(part.geometry()),
      rangeSlices(macrocell::slicesFor(part.M, 1, part.largestCells())),
      accumulateSlices(
          macrocell::slicesFor(part.M, kMacrocellTiles, part.largestCells())) {
  if (rangeSlices > 1)
    cudaErrChk(cudaMalloc(&rangePartialCUDA,
                          sizeof(ParticleRange) * regions * rangeSlices));
  if (accumulateSlices > 1)
    cudaErrChk(
        cudaMalloc(&binSumCUDA, sizeof(unsigned long long) * totalBins()));
}

macrocellHistogram3D::~macrocellHistogram3D() {
  cudaErrChk(cudaFree(rangePartialCUDA)); // null with one slice
  cudaErrChk(cudaFree(binSumCUDA));       // null with one slice
}

void macrocellHistogram3D::init(velocitySoA* pclArray, const int* cellStart,
                                const int* cellCount, histogramTypeIn vmin,
                                histogramTypeIn vmax, cudaStream_t stream) {
  binsChanged();
  const int n = pclArray->getNOP();
  const histogramTypeIn* u = pclArray->getElement(0);
  const histogramTypeIn* v = pclArray->getElement(1);
  const histogramTypeIn* w = pclArray->getElement(2);
  const histogramTypeIn* q = pclArray->getElement(3);

  // the grid and the weight scale of every macrocell, from the ranges of its
  // slices with several, the grids on the host too (the files, the GMM); the
  // cell sort of a species without particles leaves its cells as they were:
  // no walk, no particle to bin
  macrocellRangeKernel<<<regions * rangeSlices, kMacrocellThreads, 0, stream>>>(
      u, v, w, q, cellStart, cellCount, geometry, rangeSlices, vmin, vmax,
      fixedRange, n > 0, rangePartialCUDA, gridCUDA, weightScaleCUDA);
  if (rangeSlices > 1)
    regionGridKernel<<<regions, kGridThreads, 0, stream>>>(
        rangePartialCUDA, rangeSlices, kMacrocellBinsU, kMacrocellBinsV,
        kMacrocellBinsW, vmin, vmax, fixedRange, gridCUDA, weightScaleCUDA);
  cudaErrChk(cudaGetLastError());
  cudaErrChk(cudaMemcpyAsync(gridHost.data(), gridCUDA,
                             sizeof(BinGrid) * regions, cudaMemcpyDeviceToHost,
                             stream));
  cudaErrChk(cudaStreamSynchronize(stream));
  if (n == 0) {
    cudaErrChk(cudaMemsetAsync(binsCUDA, 0,
                               sizeof(histogramTypeOut) * totalBins(), stream));
    return;
  }

  // one block per tile and slice of each macrocell: with one slice it writes
  // the bins of its tile, with several they add up the integer bins, then
  // written
  const bool merge = accumulateSlices > 1;
  if (merge)
    cudaErrChk(cudaMemsetAsync(
        binSumCUDA, 0, sizeof(unsigned long long) * totalBins(), stream));
  macrocellAccumulateKernel<<<regions * kMacrocellTiles * accumulateSlices,
                              kMacrocellThreads, 0, stream>>>(
      u, v, w, q, cellStart, cellCount, geometry, accumulateSlices, gridCUDA,
      weightScaleCUDA, binsCUDA, merge ? binSumCUDA : nullptr);
  if (merge)
    binSumToHistogramKernel<<<getGridSize((int)totalBins(), kMacrocellThreads),
                              kMacrocellThreads, 0, stream>>>(
        binSumCUDA, weightScaleCUDA, kMacrocellBins, (int)totalBins(),
        binsCUDA);
  cudaErrChk(cudaGetLastError());
}

} // namespace velocityHistogram
