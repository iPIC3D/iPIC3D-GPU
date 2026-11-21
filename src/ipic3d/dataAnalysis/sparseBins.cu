#include "sparseBins.cuh"

#include "cudaReduction.cuh"
#include "dataAnalysisConfig.cuh"

#include <algorithm>
#include <climits>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>

namespace sparseBins {

// ======= Kernels =======
// The bins of each region in blocks of kCompactBlockBins consecutive bins,
// none across two regions: CUDA block blk handles block blk %
// blocksPerRegion of region blk / blocksPerRegion, in kCompactRounds rounds
// of kCompactThreads bins; launch kCompactThreads threads per block.

/** Number of non-empty bins of each block of bins. */
__global__ void countNonEmptyKernel(const binType* bins, int regionBins,
                                    int blocksPerRegion, int* blockCount) {
  const binType* region =
      bins + (size_t)(blockIdx.x / blocksPerRegion) * regionBins;
  const int firstBin = blockIdx.x % blocksPerRegion * kCompactBlockBins;
  int count = 0;
  for (int round = 0; round < kCompactRounds; round++) {
    const int bin = firstBin + round * kCompactThreads + threadIdx.x;
    count += __syncthreads_count(bin < regionBins && region[bin] > 0);
  }
  if (threadIdx.x == 0)
    blockCount[blockIdx.x] = count;
}

/** Exclusive prefix sum of the counts of the numBlocks blocks of bins in
 *  place, and the total into blockCount[numBlocks]; one block of
 *  kScanThreads threads, each on a strip of consecutive blocks. */
__global__ void scanBinBlocksKernel(int* blockCount, int numBlocks) {
  __shared__ int buffer[kScanThreads];
  const int strip = (numBlocks + blockDim.x - 1) / blockDim.x;
  const int begin = min((int)threadIdx.x * strip, numBlocks),
            end = min(begin + strip, numBlocks);
  int sum = 0;
  for (int b = begin; b < end; b++)
    sum += blockCount[b];
  int total;
  int offset = cudaReduction::blockExclusiveScan(sum, buffer, total);
  for (int b = begin; b < end; b++) {
    const int count = blockCount[b];
    blockCount[b] = offset;
    offset += count;
  }
  if (threadIdx.x == 0)
    blockCount[numBlocks] = total;
}

/** The content and the index in its region of every non-empty bin of a
 *  block of bins at blockOffset + its rank in the block: the bins keep
 *  their order. */
__global__ void scatterNonEmptyKernel(const binType* bins, int regionBins,
                                      int blocksPerRegion,
                                      const int* blockOffset, binType* weight,
                                      int* binIndex) {
  __shared__ int warpTotal[kCompactThreads / WARP_SIZE];
  const int lane = threadIdx.x % WARP_SIZE, warp = threadIdx.x / WARP_SIZE;
  const unsigned long long lanesBelow = (1ull << lane) - 1;
  const binType* region =
      bins + (size_t)(blockIdx.x / blocksPerRegion) * regionBins;
  const int firstBin = blockIdx.x % blocksPerRegion * kCompactBlockBins;

  int next = blockOffset[blockIdx.x];
  for (int round = 0; round < kCompactRounds; round++) {
    const int bin = firstBin + round * kCompactThreads + threadIdx.x;
    const bool keep = bin < regionBins && region[bin] > 0;
    const unsigned long long ballot = __ballot_sync(WARP_FULL_MASK, keep);
    if (lane == 0)
      warpTotal[warp] = __popcll(ballot);
    __syncthreads();

    int rank = __popcll(ballot & lanesBelow), roundTotal = 0;
    for (int w = 0; w < kCompactThreads / WARP_SIZE; w++) {
      rank += w < warp ? warpTotal[w] : 0;
      roundTotal += warpTotal[w];
    }
    if (keep) {
      weight[next + rank] = region[bin];
      binIndex[next + rank] = bin;
    }
    next += roundTotal;
    __syncthreads(); // warpTotal is rewritten by the next round
  }
}

// ======= SparseBins =======

SparseBins::SparseBins(int regions, int regionBins)
    : regions(regions), regionBins(regionBins),
      regionBlocks(getGridSize(regionBins, kCompactBlockBins)),
      regionOffsets(regions + 1, 0) {
  if ((size_t)regions * regionBins > INT_MAX) // the int offsets and counts
    throw std::invalid_argument("SparseBins: " + std::to_string(regions) +
                                " regions of " + std::to_string(regionBins) +
                                " bins, more than INT_MAX bins");
  const int numBlocks = regions * regionBlocks;
  cudaErrChk(cudaMalloc(&blockOffsetCUDA, sizeof(int) * (numBlocks + 1)));
  cudaErrChk(cudaMallocHost(&blockOffsetHost, sizeof(int) * (numBlocks + 1)));
}

SparseBins::~SparseBins() {
  for (void* buffer :
       {(void*)weightCUDA, (void*)indexCUDA, (void*)blockOffsetCUDA})
    cudaErrChk(cudaFree(buffer)); // null for the unused ones
  cudaErrChk(cudaFreeHost(blockOffsetHost));
}

int SparseBins::compact(const binType* bins, cudaStream_t stream) {
  const int numBlocks = regions * regionBlocks;
  countNonEmptyKernel<<<numBlocks, kCompactThreads, 0, stream>>>(
      bins, regionBins, regionBlocks, blockOffsetCUDA);
  scanBinBlocksKernel<<<1, kScanThreads, 0, stream>>>(blockOffsetCUDA,
                                                      numBlocks);
  cudaErrChk(cudaGetLastError());
  cudaErrChk(cudaMemcpyAsync(blockOffsetHost, blockOffsetCUDA,
                             sizeof(int) * (numBlocks + 1),
                             cudaMemcpyDeviceToHost, stream));
  cudaErrChk(cudaStreamSynchronize(stream));
  nonEmpty = blockOffsetHost[numBlocks];
  for (int r = 0; r <= regions; r++)
    regionOffsets[r] = blockOffsetHost[r * regionBlocks];

  if (nonEmpty > capacity) { // grow only, with a margin
    capacity = (int)std::min((size_t)nonEmpty + nonEmpty / 4,
                             (size_t)regions * regionBins);
    cudaErrChk(cudaFree(weightCUDA));
    cudaErrChk(cudaFree(indexCUDA));
    weightCUDA = nullptr;
    indexCUDA = nullptr;
    cudaErrChk(cudaMalloc(&weightCUDA, sizeof(binType) * capacity));
    cudaErrChk(cudaMalloc(&indexCUDA, sizeof(int) * capacity));
  }
  if (nonEmpty > 0) {
    scatterNonEmptyKernel<<<numBlocks, kCompactThreads, 0, stream>>>(
        bins, regionBins, regionBlocks, blockOffsetCUDA, weightCUDA, indexCUDA);
    cudaErrChk(cudaGetLastError());
    cudaErrChk(cudaStreamSynchronize(stream));
  }
  return nonEmpty;
}

std::string
SparseBins::describe(const std::string& rank, const std::string& regionsText,
                     const std::string& binsText, const std::string& rangeName,
                     const std::string& rangeShape, std::uint32_t magic,
                     const std::string& content,
                     std::vector<std::string>& layout) const {
  std::ostringstream scale;
  scale << std::setprecision(std::numeric_limits<double>::max_digits10)
        << "  \"weight_scale\": " << DAConfig::VELOCITY_WEIGHT_SCALE << ",\n";
  layout = {regionsText, binsText, scale.str()};
  const std::string index = regionBins <= kShortIndexBins
                                ? dtypeName<std::uint16_t>()
                                : dtypeName<std::uint32_t>();
  return describeRecords(rank + regionsText + binsText + scale.str(), magic,
                         {{"magic", dtypeName<std::uint32_t>(), ""},
                          {"cycle", dtypeName<std::int32_t>(), ""},
                          {"species", dtypeName<std::int32_t>(), ""},
                          {"regions", dtypeName<std::uint32_t>(), ""},
                          {"non_empty", dtypeName<std::uint64_t>(), ""},
                          {"record_bytes", dtypeName<std::uint64_t>(), ""}},
                         {{rangeName, dtypeName<double>(), rangeShape},
                          {"counts", dtypeName<std::uint32_t>(), "\"regions\""},
                          {"weights", dtypeName<binType>(), "\"non_empty\""},
                          {"bins", index, "\"non_empty\""}},
                         content);
}

void SparseBins::writeRecord(RecordFile& file, int species, int cycle,
                             const double* range, std::size_t count,
                             cudaStream_t stream) {
  weightHost.resize(nonEmpty);
  indexHost.resize(nonEmpty);
  if (nonEmpty > 0) {
    cudaErrChk(cudaMemcpyAsync(weightHost.data(), weightCUDA,
                               sizeof(binType) * nonEmpty,
                               cudaMemcpyDeviceToHost, stream));
    cudaErrChk(cudaMemcpyAsync(indexHost.data(), indexCUDA,
                               sizeof(int) * nonEmpty, cudaMemcpyDeviceToHost,
                               stream));
    cudaErrChk(cudaStreamSynchronize(stream));
  }
  file.begin(SparseRecordHeader{0, cycle, species, (std::uint32_t)regions,
                                (std::uint64_t)nonEmpty, 0});
  file.put(range, count);
  std::vector<std::uint32_t> counts(regions);
  for (int r = 0; r < regions; r++)
    counts[r] = regionOffsets[r + 1] - regionOffsets[r];
  file.put(counts.data(), counts.size());
  file.put(weightHost.data(), weightHost.size());
  if (regionBins <= kShortIndexBins) {
    const std::vector<std::uint16_t> index(indexHost.begin(), indexHost.end());
    file.put(index.data(), index.size());
  } else {
    const std::vector<std::uint32_t> index(indexHost.begin(), indexHost.end());
    file.put(index.data(), index.size());
  }
  file.end();
}

} // namespace sparseBins
