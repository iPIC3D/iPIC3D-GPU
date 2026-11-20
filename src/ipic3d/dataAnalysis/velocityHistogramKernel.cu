#include "velocityHistogram.cuh"

#include "cudaReduction.cuh"

namespace velocityHistogram {

/**
 * @brief The histogram in one pass over the particles per tile of kTile^3
 * bins, summed in shared memory with float atomics, then added to the global
 * histogram with float atomics.
 * @details Not used currently: velocityHistogram3D::accumulate reads the
 * velocities once per batch, sums in fixed point and uses no global atomics.
 */
__global__ void histogramKernel3D(const int nop, const histogramTypeIn* d1,
                                  const histogramTypeIn* d2,
                                  const histogramTypeIn* d3,
                                  const histogramTypeIn* q,
                                  velocityHistogramCUDA3D* histogramCUDAPtr) {

  int pidx = threadIdx.x + blockIdx.x * blockDim.x;
  int gridSize = gridDim.x * blockDim.x;

  extern __shared__ histogramTypeOut sHistogram[];
  auto gHistogram = histogramCUDAPtr[0].getHistogramCUDA();

  histogramTypeIn data[3];
  int dim[3];
  int dimSize[3] = {kTile, kTile, kTile};
  histogramTypeOut qAbs;

  // size of this histogram
  const auto dim0Size = histogramCUDAPtr[0].getSize(0);
  const auto dim1Size = histogramCUDAPtr[0].getSize(1);
  const auto dim2Size = histogramCUDAPtr[0].getSize(2);

  // the dim sizes are multiply of tile
  for (int dim0 = 0; dim0 < dim0Size; dim0 += kTile) // unroll if const size
    for (int dim1 = 0; dim1 < dim1Size; dim1 += kTile)
      for (int dim2 = 0; dim2 < dim2Size; dim2 += kTile) {
        dim[0] = dim0;
        dim[1] = dim1;
        dim[2] = dim2;

        // Initialize shared memory to zero
        for (int i = threadIdx.x; i < kTileBins; i += blockDim.x) {
          sHistogram[i] = 0.0;
        }
        __syncthreads();

        for (int i = pidx; i < nop; i += gridSize) {
          data[0] = d1[i];
          data[1] = d2[i];
          data[2] = d3[i];
          const auto sIndex =
              histogramCUDAPtr[0].getIndexTiled(data, dim, dimSize);
          if (sIndex < 0)
            continue; // out of tile range

          qAbs = abs(q[i] * DAConfig::VELOCITY_WEIGHT_SCALE);
          atomicAdd(&sHistogram[sIndex], qAbs);
        }

        __syncthreads();

        for (int i = threadIdx.x; i < kTileBins; i += blockDim.x) {
          atomicAdd(
              &gHistogram[dim0 + i % kTile +
                          (dim1 + i / kTile % kTile) * dim0Size +
                          (dim2 + i / kTile / kTile) * dim0Size * dim1Size],
              sHistogram[i]);
        }
      }
}

/**
 * @brief zero the histogram bins
 * @details Not used currently: the accumulation writes every bin.
 * @param histogramCUDAPtr Device histogram descriptor to reset.
 */
__global__ void resetBin(velocityHistogramCUDA3D* histogramCUDAPtr) {
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  int gridSize = gridDim.x * blockDim.x;

  auto histogram = histogramCUDAPtr->getHistogramCUDA();
  const auto histogramSize = histogramCUDAPtr->getLogicSize();

  for (int i = idx; i < histogramSize; i += gridSize)
    histogram[i] = 0.0;
}

// ======= Accumulation: fixed-point weights, tiles in shared memory =======
// The weight scale once (weightRangeKernel, weightScaleKernel), then per
// batch of particles: binKernel (key, count per block and tile),
// tileBlockScanKernel and tileScanKernel (slots and chunks), scatterKernel
// (tile order, fixed point), accumulateKernel (one chunk per block) and
// mergeTilesKernel (into the integer histogram). Only shared-memory atomics,
// on integers.

/** Exclusive prefix sum over the block (any block size); total gets the sum.
 *  buffer holds blockDim entries. */
static __device__ int blockExclusiveScan(int value, int* buffer, int& total) {
  buffer[threadIdx.x] = value;
  __syncthreads();
  for (int offset = 1; offset < (int)blockDim.x; offset *= 2) {
    const int add =
        (int)threadIdx.x >= offset ? buffer[threadIdx.x - offset] : 0;
    __syncthreads();
    buffer[threadIdx.x] += add;
    __syncthreads();
  }
  const int inclusive = buffer[threadIdx.x];
  total = buffer[blockDim.x - 1];
  __syncthreads();
  return inclusive - value;
}

/** Tile * kAccumTileBins + bin in the tile, or kOutsideKey for a velocity
 *  outside [min, max] or not finite; max is in the last bin. */
__device__ __forceinline__ unsigned binKey(const histogramTypeIn* v,
                                           const BinGrid& grid) {
  int bin[3];
  for (int d = 0; d < 3; d++) {
    if (!(v[d] >= grid.min[d] && v[d] <= grid.max[d])) // also NaN
      return kOutsideKey;
    const int b = (int)((v[d] - grid.min[d]) * grid.invResolution[d]);
    bin[d] = b < binsAlong(d) ? b : binsAlong(d) - 1;
  }
  const unsigned tile = bin[0] / kAccumTileX +
                        kAccumTilesX * (bin[1] / kAccumTileY +
                                        kAccumTilesY * (bin[2] / kAccumTileZ));
  const unsigned inTile = bin[0] % kAccumTileX +
                          kAccumTileX * (bin[1] % kAccumTileY +
                                         kAccumTileY * (bin[2] % kAccumTileZ));
  return tile * kAccumTileBins + inTile;
}

/** The largest |q| of the finite charges and their number, per block:
 *  partial[block] and partial[kHistogramBlocks + block]. Integer operations
 *  (double ones are slow on consumer GPUs): the bits of |q| order as |q|. */
__global__ void weightRangeKernel(const histogramTypeIn* q, int n,
                                  double* partial) {
  __shared__ unsigned long long largest[kHistogramThreads];
  __shared__ unsigned count[kHistogramThreads];
  // a plain comparison: MaxOp would need a max() of 64-bit integers, which
  // HIP does not overload
  auto maxOp = [](unsigned long long a, unsigned long long b) {
    return a > b ? a : b;
  };
  unsigned long long m = 0;
  unsigned c = 0;
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
       i += gridDim.x * blockDim.x) {
    const double charge = q[i];
    const unsigned long long bits = // |q|: the sign bit cleared
        (unsigned long long)(unsigned)(__double2hiint(charge) & 0x7fffffff)
            << 32 |
        (unsigned)__double2loint(charge);
    const bool finite = isfinite_fast(charge);
    m = finite && bits > m ? bits : m;
    c += finite;
  }
  m = cudaReduction::blockReduce<kHistogramThreads>(m, largest, maxOp);
  c = cudaReduction::blockReduce<kHistogramThreads>(c, count,
                                                    cudaReduction::SumOp{});
  if (threadIdx.x == 0) {
    partial[blockIdx.x] = __hiloint2double((int)(m >> 32), (int)(unsigned)m);
    partial[kHistogramBlocks + blockIdx.x] = c;
  }
}

/** From n, the finite charges, and M, the largest |q| (both exact, whatever
 *  the order of the particles: the counts are integers): scale[0] = 2^62 /
 *  (n M) (to fixed point) and scale[1] = n M / 2^62 * VELOCITY_WEIGHT_SCALE
 *  (to the output), without forming n M, which overflows for a huge finite
 *  M; one block of WARP_SIZE threads. */
__global__ void weightScaleKernel(const double* partial, double* scale) {
  const double largest = cudaReduction::reduceInWarp(
      partial, kHistogramBlocks, cudaReduction::MaxOp{}, 0.0);
  const double count =
      cudaReduction::reduceInWarp(partial + kHistogramBlocks, kHistogramBlocks,
                                  cudaReduction::SumOp{}, 0.0);
  if (threadIdx.x == 0) {
    const double toFixed = kFixedPointOne / largest / count; // inf for n M = 0
    scale[0] = isfinite(toFixed) ? toFixed : 0; // then every weight is 0
    scale[1] =
        largest / kFixedPointOne * DAConfig::VELOCITY_WEIGHT_SCALE * count;
  }
}

/** Key of every particle, and the particles of each tile in each block
 *  (tileCount[block * kAccumTiles + tile]). */
__global__ void binKernel(const histogramTypeIn* u, const histogramTypeIn* v,
                          const histogramTypeIn* w, int n, BinGrid grid,
                          unsigned* key, int* tileCount) {
  __shared__ int count[kAccumTiles];
  for (int t = threadIdx.x; t < kAccumTiles; t += blockDim.x)
    count[t] = 0;
  __syncthreads();
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
       i += gridDim.x * blockDim.x) {
    const histogramTypeIn velocity[3] = {u[i], v[i], w[i]};
    const unsigned k = binKey(velocity, grid);
    key[i] = k;
    if (k != kOutsideKey)
      atomicAdd(&count[k / kAccumTileBins], 1);
  }
  __syncthreads();
  for (int t = threadIdx.x; t < kAccumTiles; t += blockDim.x)
    tileCount[blockIdx.x * kAccumTiles + t] = count[t];
}

/** One block per tile, one thread per particle block: the first slot of each
 *  particle block within the tile, and the particles of the tile. */
__global__ void tileBlockScanKernel(const int* tileCount, int* tileBlockOffset,
                                    int* tileTotal) {
  __shared__ int buffer[kHistogramBlocks];
  const int t = blockIdx.x, b = threadIdx.x;
  int total;
  tileBlockOffset[b * kAccumTiles + t] =
      blockExclusiveScan(tileCount[b * kAccumTiles + t], buffer, total);
  if (b == 0)
    tileTotal[t] = total;
}

/**
 * @brief The first slot of each tile in tile order, and the chunks: one block
 * of kAccumTiles threads, thread t for tile t. Each tile is cut into chunks
 * of at most kChunkParticles.
 */
__global__ void tileScanKernel(const int* tileTotal, int* tileStart,
                               int* tileChunkStart, int* tileChunkCount,
                               HistogramChunk* chunks, int* numChunks) {
  __shared__ int buffer[kAccumTiles];
  const int t = threadIdx.x;
  const int count = tileTotal[t];
  int total;
  const int start = blockExclusiveScan(count, buffer, total);
  const int tileChunks = (count + kChunkParticles - 1) / kChunkParticles;
  int allChunks;
  const int firstChunk = blockExclusiveScan(tileChunks, buffer, allChunks);
  tileStart[t] = start;
  tileChunkStart[t] = firstChunk;
  tileChunkCount[t] = tileChunks;
  for (int c = 0; c < tileChunks; c++)
    chunks[firstChunk + c] = {start + c * kChunkParticles,
                              start + min((c + 1) * kChunkParticles, count)};
  if (t == 0)
    *numChunks = allChunks;
}

/** Each particle in range to its slot (the grid and grid stride of
 *  binKernel, whose per-block counts give the slots): its bin in the tile and
 *  its weight in fixed point (0 for a non-finite charge). */
__global__ void scatterKernel(const unsigned* key, const histogramTypeIn* q,
                              int n, const int* tileStart,
                              const int* tileBlockOffset, const double* scale,
                              unsigned short* tileBin,
                              unsigned long long* fixedWeight) {
  __shared__ int cursor[kAccumTiles];
  for (int t = threadIdx.x; t < kAccumTiles; t += blockDim.x)
    cursor[t] = tileStart[t] + tileBlockOffset[blockIdx.x * kAccumTiles + t];
  __syncthreads();
  const double toFixed = scale[0];
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
       i += gridDim.x * blockDim.x) {
    const unsigned k = key[i];
    if (k == kOutsideKey)
      continue;
    // the order within a slot range varies; integer sums do not mind
    const int slot = atomicAdd(&cursor[k / kAccumTileBins], 1);
    tileBin[slot] = (unsigned short)(k % kAccumTileBins);
    const histogramTypeIn charge = q[i];
    fixedWeight[slot] =
        isfinite_fast(charge) ? __double2ull_rn(fabs(charge) * toFixed) : 0;
  }
}

/** One chunk per block: its particles summed into the bins of its tile in
 *  shared memory, written to partialTile[chunk]. A bin is two 32-bit words:
 *  NVIDIA has no 64-bit shared-memory add (a compare-and-swap loop, slow on
 *  a narrow distribution), the carry of the low word goes to the high one. */
__global__ void accumulateKernel(const HistogramChunk* chunks,
                                 const int* numChunks,
                                 const unsigned short* tileBin,
                                 const unsigned long long* fixedWeight,
                                 unsigned long long* partialTile) {
  if ((int)blockIdx.x >= *numChunks) // launched for the most chunks
    return;
  __shared__ unsigned tileLow[kAccumTileBins], tileHigh[kAccumTileBins];
  for (int j = threadIdx.x; j < kAccumTileBins; j += blockDim.x) {
    tileLow[j] = 0;
    tileHigh[j] = 0;
  }
  __syncthreads();
  auto add = [&](int bin, unsigned long long weight) {
    const unsigned low = (unsigned)weight;
    const unsigned old = atomicAdd(&tileLow[bin], low);
    const unsigned high = (unsigned)(weight >> 32) + (old + low < old);
    if (high)
      atomicAdd(&tileHigh[bin], high);
  };
  const HistogramChunk chunk = chunks[blockIdx.x];
  const int lane = threadIdx.x % WARP_SIZE;
  for (int base = chunk.begin; base < chunk.end; base += blockDim.x) {
    const int i = base + threadIdx.x;
    const bool valid = i < chunk.end;
    const int bin = valid ? tileBin[i] : -1;
    unsigned long long weight = valid ? fixedWeight[i] : 0;
    // a warp whose particles share one bin (a cold beam) adds once
    const int firstBin = __shfl_sync(WARP_FULL_MASK, bin, 0);
    if (__ballot_sync(WARP_FULL_MASK, bin == firstBin) == WARP_FULL_MASK) {
      weight = warp_reduce_sum(weight);
      if (lane == 0 && firstBin >= 0)
        add(firstBin, weight);
    } else if (valid) {
      add(bin, weight);
    }
  }
  __syncthreads();
  for (int j = threadIdx.x; j < kAccumTileBins; j += blockDim.x)
    partialTile[(size_t)blockIdx.x * kAccumTileBins + j] =
        (unsigned long long)tileHigh[j] << 32 | tileLow[j];
}

/** One thread per (tile, bin): the partial tiles of the tile's chunks added
 *  to the integer histogram. */
__global__ void mergeTilesKernel(const int* tileChunkStart,
                                 const int* tileChunkCount,
                                 const unsigned long long* partialTile,
                                 unsigned long long* binSum) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= kAccumTiles * kAccumTileBins)
    return;
  const int t = idx / kAccumTileBins, j = idx % kAccumTileBins;
  const int first = tileChunkStart[t], count = tileChunkCount[t];
  if (count == 0)
    return;
  unsigned long long sum = 0;
  for (int c = first; c < first + count; c++)
    sum += partialTile[(size_t)c * kAccumTileBins + j];
  const int tx = t % kAccumTilesX, ty = t / kAccumTilesX % kAccumTilesY,
            tz = t / (kAccumTilesX * kAccumTilesY);
  const int jx = j % kAccumTileX, jy = j / kAccumTileX % kAccumTileY,
            jz = j / (kAccumTileX * kAccumTileY);
  binSum[(tx * kAccumTileX + jx) +
         DAConfig::VELOCITY_HISTOGRAM3D_RES_1 *
             ((ty * kAccumTileY + jy) + DAConfig::VELOCITY_HISTOGRAM3D_RES_2 *
                                            (tz * kAccumTileZ + jz))] += sum;
}

/** The integer histogram to the output histogram. */
__global__ void binSumToHistogramKernel(const unsigned long long* binSum,
                                        const double* scale,
                                        histogramTypeOut* histogram) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < kNumBins)
    histogram[i] = (histogramTypeOut)((double)binSum[i] * scale[1]);
}

// ======= Compaction of the non-empty bins =======
// A CUDA block handles a block of kCompactBlockBins consecutive bins in
// kCompactRounds rounds of kCompactThreads bins; launch kCompactThreads
// threads per block.

/** Number of non-empty bins of each block of bins. */
__global__ void countNonEmptyKernel(const histogramTypeOut* bins,
                                    int* blockCount) {
  const int firstBin = blockIdx.x * kCompactBlockBins;
  int count = 0;
  for (int round = 0; round < kCompactRounds; round++) {
    const int bin = firstBin + round * kCompactThreads + threadIdx.x;
    count += __syncthreads_count(bin < kNumBins && bins[bin] > 0);
  }
  if (threadIdx.x == 0)
    blockCount[blockIdx.x] = count;
}

/** Exclusive prefix sum of the counts of the blocks of bins in place, and the
 *  total into blockCount[kCompactBlocks]; one block of kCompactBlocks
 *  threads. */
__global__ void scanBinBlocksKernel(int* blockCount) {
  __shared__ int buffer[kCompactBlocks];
  int total;
  const int offset = blockExclusiveScan(blockCount[threadIdx.x], buffer, total);
  blockCount[threadIdx.x] = offset;
  if (threadIdx.x == 0)
    blockCount[kCompactBlocks] = total;
}

/**
 * @brief Write the centre and the content of every non-empty bin of a block
 * of bins at blockOffset + its rank in the block, so that the bins keep
 * their order.
 */
__global__ void
scatterNonEmptyKernel(BinGrid grid, const histogramTypeOut* bins,
                      const int* blockOffset, histogramTypeIn* centerU,
                      histogramTypeIn* centerV, histogramTypeIn* centerW,
                      histogramTypeOut* weight) {
  __shared__ int warpTotal[kCompactThreads / WARP_SIZE];
  const int lane = threadIdx.x % WARP_SIZE, warp = threadIdx.x / WARP_SIZE;
  const unsigned long long lanesBelow = (1ull << lane) - 1;

  int next = blockOffset[blockIdx.x];
  for (int round = 0; round < kCompactRounds; round++) {
    const int bin =
        blockIdx.x * kCompactBlockBins + round * kCompactThreads + threadIdx.x;
    const bool keep = bin < kNumBins && bins[bin] > 0;
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
      const int out = next + rank;
      centerU[out] = grid.binCenter(bin, 0);
      centerV[out] = grid.binCenter(bin, 1);
      centerW[out] = grid.binCenter(bin, 2);
      weight[out] = bins[bin];
    }
    next += roundTotal;
    __syncthreads(); // warpTotal is rewritten by the next round
  }
}

} // namespace velocityHistogram
