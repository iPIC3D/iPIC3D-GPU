#include "velocityHistogram.cuh"

#include "cudaReduction.cuh"

namespace velocityHistogram {

// ======= Accumulation: fixed-point weights, tiles in shared memory =======
// The grid and the weight scale once (rankRangeKernel, regionGridKernel), then
// per batch of particles: binKernel (key, count per block and tile),
// tileBlockScanKernel and tileScanKernel (slots and chunks), scatterKernel
// (tile order, fixed point), accumulateKernel (one chunk per block) and
// mergeTilesKernel (into the integer histogram). Only shared-memory atomics,
// on integers.

/** A grid of BU x BV x BW bins (u fastest) in tiles of TU x TV x TW bins,
 *  tile after tile (u fastest) in the shared memory of the accumulation. */
template <int BU, int BV, int BW, int TU, int TV, int TW> struct TiledBins {
  /** Tile * tile bins + bin in the tile of the velocity v on the grid, or
   *  kOutsideKey for a velocity outside [min, max] or not finite; max is in
   *  the last bin. */
  static __device__ __forceinline__ unsigned key(const histogramTypeIn* v,
                                                 const BinGrid& grid) {
    int bin[3];
    for (int d = 0; d < 3; d++) {
      if (!(v[d] >= grid.min[d] && v[d] <= grid.max[d])) // also NaN
        return kOutsideKey;
      const int bins = d == 0 ? BU : d == 1 ? BV : BW;
      const int b = (int)((v[d] - grid.min[d]) * grid.invResolution[d]);
      bin[d] = b < bins ? b : bins - 1;
    }
    const unsigned tile =
        bin[0] / TU + (BU / TU) * (bin[1] / TV + (BV / TV) * (bin[2] / TW));
    const unsigned inTile =
        bin[0] % TU + TU * (bin[1] % TV + TV * (bin[2] % TW));
    return tile * (TU * TV * TW) + inTile;
  }

  /** The bin of the histogram of bin inTile of tile tile (the inverse of
   *  key). */
  static __device__ __forceinline__ int bin(int tile, int inTile) {
    const int tu = tile % (BU / TU), tv = tile / (BU / TU) % (BV / TV),
              tw = tile / ((BU / TU) * (BV / TV));
    const int ju = inTile % TU, jv = inTile / TU % TV, jw = inTile / (TU * TV);
    return (tu * TU + ju) + BU * ((tv * TV + jv) + BV * (tw * TW + jw));
  }
};

/** The tiles of the histogram of a rank and of a macrocell. */
using RankTiles = TiledBins<DAConfig::VELOCITY_HISTOGRAM3D_RES_1,
                            DAConfig::VELOCITY_HISTOGRAM3D_RES_2,
                            DAConfig::VELOCITY_HISTOGRAM3D_RES_3, kAccumTileX,
                            kAccumTileY, kAccumTileZ>;
using MacrocellTiles =
    TiledBins<kMacrocellBinsU, kMacrocellBinsV, kMacrocellBinsW,
              kMacrocellTileU, kMacrocellTileV, kMacrocellTileW>;

/** Add a fixed-point weight to bin of a tile in shared memory, kept as two
 *  32-bit words: NVIDIA has no 64-bit shared-memory add (a compare-and-swap
 *  loop, slow on a narrow distribution), the carry of the low word goes to the
 *  high one. Every lane of the warp calls it, a lane with no weight with bin
 *  -1; a warp whose lanes share one bin (a cold beam) adds once. */
__device__ __forceinline__ void addToTile(unsigned* tileLow, unsigned* tileHigh,
                                          int bin, unsigned long long weight) {
  auto add = [&](int b, unsigned long long w) {
    const unsigned low = (unsigned)w;
    const unsigned old = atomicAdd(&tileLow[b], low);
    const unsigned high = (unsigned)(w >> 32) + (old + low < old);
    if (high)
      atomicAdd(&tileHigh[b], high);
  };
  const int firstBin = __shfl_sync(WARP_FULL_MASK, bin, 0);
  if (__ballot_sync(WARP_FULL_MASK, bin == firstBin) == WARP_FULL_MASK) {
    if (firstBin >= 0) { // the same for the warp
      weight = warp_reduce_sum(weight);
      if (threadIdx.x % WARP_SIZE == 0)
        add(firstBin, weight);
    }
  } else if (bin >= 0) {
    add(bin, weight);
  }
}

/** |q| as the bits of a double with the sign bit cleared, which order as
 *  |q|: integer operations (double ones are slow on consumer GPUs). */
__device__ __forceinline__ unsigned long long absBits(double charge) {
  return (unsigned long long)(unsigned)(__double2hiint(charge) & 0x7fffffff)
             << 32 |
         (unsigned)__double2loint(charge);
}

/** The double of the bits of absBits. */
__device__ __forceinline__ double fromBits(unsigned long long bits) {
  return __hiloint2double((int)(bits >> 32), (int)(unsigned)bits);
}

/** A charge into the largest |q| (as absBits) and the number of the finite
 *  charges. */
__device__ __forceinline__ void
addCharge(double charge, unsigned long long& largest, unsigned& count) {
  const unsigned long long bits = absBits(charge);
  const bool finite = isfinite_fast(charge);
  largest = finite && bits > largest ? bits : largest;
  count += finite;
}

/** The fixed-point weight of a charge, scale[0] of fixedPointScale; 0 for a
 *  non-finite one. */
__device__ __forceinline__ unsigned long long fixedPointWeight(double charge,
                                                               double toFixed) {
  return isfinite_fast(charge) ? __double2ull_rn(fabs(charge) * toFixed) : 0;
}

/** From n, the finite charges, and M, the largest |q|: scale[0] = 2^62 /
 *  (n M) (to fixed point; 0 without charges) and scale[1] = n M / 2^62 *
 *  VELOCITY_WEIGHT_SCALE (to the output), without forming n M, which
 *  overflows for a huge finite M. */
__device__ __forceinline__ void fixedPointScale(double largest, double count,
                                                double* scale) {
  const double toFixed = kFixedPointOne / largest / count; // inf for n M = 0
  scale[0] = isfinite(toFixed) ? toFixed : 0; // then every weight is 0
  scale[1] = largest / kFixedPointOne * DAConfig::VELOCITY_WEIGHT_SCALE * count;
}

/** The range of no particle. */
__device__ __forceinline__ ParticleRange emptyRange() {
  return {
      {INFINITY, INFINITY, INFINITY}, {-INFINITY, -INFINITY, -INFINITY}, 0, 0};
}

/** Particle i into range: its charge, and its finite velocities without
 *  fixedRange. */
__device__ __forceinline__ void
addParticle(ParticleRange& range, const histogramTypeIn* u,
            const histogramTypeIn* v, const histogramTypeIn* w,
            const histogramTypeIn* q, int i, bool fixedRange) {
  addCharge(q[i], range.largest, range.count);
  if (fixedRange)
    return;
  const histogramTypeIn velocity[3] = {u[i], v[i], w[i]};
  for (int d = 0; d < 3; d++)
    if (isfinite_fast(velocity[d])) {
      range.low[d] = velocity[d] < range.low[d] ? velocity[d] : range.low[d];
      range.high[d] = velocity[d] > range.high[d] ? velocity[d] : range.high[d];
    }
}

/** Shared memory of blockRange for a block of `threads`. */
template <int threads> struct RangeScratch {
  histogramTypeIn velocity[threads];
  unsigned long long largest[threads];
  unsigned count[threads];
};

/** from merged into range: the minima, maxima and sums of both. */
__device__ __forceinline__ void mergeRange(ParticleRange& range,
                                           const ParticleRange& from) {
  range.largest = from.largest > range.largest ? from.largest : range.largest;
  range.count += from.count;
  for (int d = 0; d < 3; d++) {
    range.low[d] = from.low[d] < range.low[d] ? from.low[d] : range.low[d];
    range.high[d] = from.high[d] > range.high[d] ? from.high[d] : range.high[d];
  }
}

/** The range of the particles of the threads of the block, in thread 0
 *  (blockReduce): exact, whatever their order (minima, maxima and integer
 *  sums). */
template <int threads>
__device__ ParticleRange blockRange(ParticleRange range, bool fixedRange,
                                    RangeScratch<threads>& scratch) {
  using namespace cudaReduction;
  range.largest =
      blockReduce<threads>(range.largest, scratch.largest, MaxU64Op{});
  range.count = blockReduce<threads>(range.count, scratch.count, SumOp{});
  if (!fixedRange) // the same for the whole block
    for (int d = 0; d < 3; d++) {
      range.low[d] =
          blockReduce<threads>(range.low[d], scratch.velocity, MinOp{});
      __syncthreads();
      range.high[d] =
          blockReduce<threads>(range.high[d], scratch.velocity, MaxOp{});
      __syncthreads();
    }
  return range;
}

/** The range of the particles of each block of kHistogramThreads (a grid
 *  stride over the n particles): partial[block]. */
__global__ void rankRangeKernel(const histogramTypeIn* u,
                                const histogramTypeIn* v,
                                const histogramTypeIn* w,
                                const histogramTypeIn* q, int n,
                                bool fixedRange, ParticleRange* partial) {
  __shared__ RangeScratch<kHistogramThreads> scratch;
  ParticleRange range = emptyRange();
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
       i += gridDim.x * blockDim.x)
    addParticle(range, u, v, w, q, i, fixedRange);
  range = blockRange<kHistogramThreads>(range, fixedRange, scratch);
  if (threadIdx.x == 0)
    partial[blockIdx.x] = range;
}

/** The grid of a region of the given bins from the range of its particles
 *  (binGridOf) and the fixed-point scale of its charges (fixedPointScale),
 *  scale[0] and scale[1]. */
__device__ __forceinline__ void
regionGrid(const ParticleRange& range, const int bins[3], histogramTypeIn vmin,
           histogramTypeIn vmax, bool fixedRange, BinGrid* grid,
           double* scale) {
  *grid = binGridOf(bins, range.low, range.high, vmin, vmax, fixedRange);
  fixedPointScale(fromBits(range.largest), range.count, scale);
}

/** One block of kGridThreads per region: its `partials` partial ranges
 *  (partial[region * partials + p]) merged, its grid of binsU x binsV x
 *  binsW bins and the scale of its charges (regionGrid) in grids[region],
 *  scale[2 region], scale[2 region + 1]. */
__global__ void regionGridKernel(const ParticleRange* partial, int partials,
                                 int binsU, int binsV, int binsW,
                                 histogramTypeIn vmin, histogramTypeIn vmax,
                                 bool fixedRange, BinGrid* grids,
                                 double* scale) {
  __shared__ RangeScratch<kGridThreads> scratch;
  ParticleRange range = emptyRange();
  for (int p = threadIdx.x; p < partials; p += blockDim.x)
    mergeRange(range, partial[(size_t)blockIdx.x * partials + p]);
  range = blockRange<kGridThreads>(range, fixedRange, scratch);
  if (threadIdx.x == 0) {
    const int bins[3] = {binsU, binsV, binsW};
    regionGrid(range, bins, vmin, vmax, fixedRange, grids + blockIdx.x,
               scale + 2 * blockIdx.x);
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
    const unsigned k = RankTiles::key(velocity, grid);
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
  tileBlockOffset[b * kAccumTiles + t] = cudaReduction::blockExclusiveScan(
      tileCount[b * kAccumTiles + t], buffer, total);
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
  const int start = cudaReduction::blockExclusiveScan(count, buffer, total);
  const int tileChunks = (count + kChunkParticles - 1) / kChunkParticles;
  int allChunks;
  const int firstChunk =
      cudaReduction::blockExclusiveScan(tileChunks, buffer, allChunks);
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
    fixedWeight[slot] = fixedPointWeight(q[i], toFixed);
  }
}

/** One chunk per block: its particles summed into the bins of its tile in
 *  shared memory (addToTile), written to partialTile[chunk]. */
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
  const HistogramChunk chunk = chunks[blockIdx.x];
  for (int base = chunk.begin; base < chunk.end; base += blockDim.x) {
    const int i = base + threadIdx.x;
    const bool valid = i < chunk.end;
    addToTile(tileLow, tileHigh, valid ? tileBin[i] : -1,
              valid ? fixedWeight[i] : 0);
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
  binSum[RankTiles::bin(t, j)] += sum;
}

/** The integer bins of the regions of regionBins bins (n bins in all) to
 *  the output bins, with the scale of each region (scale[2 region + 1]). */
__global__ void binSumToHistogramKernel(const unsigned long long* binSum,
                                        const double* scale, int regionBins,
                                        int n, histogramTypeOut* histogram) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n)
    histogram[i] =
        (histogramTypeOut)((double)binSum[i] * scale[2 * (i / regionBins) + 1]);
}

// ======= Macrocell histograms =======
// One block per slice of the cells of a macrocell (forEachMacrocellRow: one
// slice per macrocell unless the macrocells are few) walks the rows of its
// cells: macrocellRangeKernel gives the range of each slice (with one slice,
// the grid of its macrocell), regionGridKernel the grid of each macrocell
// from its slices, then one block per slice and tile sums the particles of
// the tile in shared memory and writes the bins of the tile, or adds them to
// the integer bins of the macrocell (macrocellAccumulateKernel).

/** One block of kMacrocellThreads per slice of `slices` of each macrocell:
 *  the range of its particles (their finite velocities without fixedRange),
 *  partial[m * slices + slice] (regionGridKernel merges them); with one
 *  slice, that of the macrocell: its grid and scale (regionGrid) in
 *  grids[m], scale[2 m], scale[2 m + 1]. Without walk (a species without
 *  particles, whose cells the sort left as they were) the range of no
 *  particle. */
__global__ void macrocellRangeKernel(
    const histogramTypeIn* u, const histogramTypeIn* v,
    const histogramTypeIn* w, const histogramTypeIn* q, const int* cellStart,
    const int* cellCount, macrocell::MacrocellGeometry geometry, int slices,
    histogramTypeIn vmin, histogramTypeIn vmax, bool fixedRange, bool walk,
    ParticleRange* partial, BinGrid* grids, double* scale) {
  __shared__ RangeScratch<kMacrocellThreads> scratch;
  ParticleRange range = emptyRange();
  if (walk) // the same for the whole block
    macrocell::forEachMacrocellRow(
        geometry, blockIdx.x / slices, blockIdx.x % slices, slices, cellStart,
        cellCount, [&](int begin, int end) {
          for (int i = begin + threadIdx.x; i < end; i += blockDim.x)
            addParticle(range, u, v, w, q, i, fixedRange);
        });
  range = blockRange<kMacrocellThreads>(range, fixedRange, scratch);
  if (threadIdx.x != 0)
    return;
  if (slices > 1) {
    partial[blockIdx.x] = range;
  } else {
    const int bins[3] = {kMacrocellBinsU, kMacrocellBinsV, kMacrocellBinsW};
    regionGrid(range, bins, vmin, vmax, fixedRange, grids + blockIdx.x,
               scale + 2 * blockIdx.x);
  }
}

/** One block of kMacrocellThreads per tile and slice of `slices` of each
 *  macrocell: the fixed-point weights of the particles of the slice in the
 *  tile summed in shared memory (addToTile, 0 for a non-finite charge), then
 *  the bins of the tile written (one slice) or their non-zero sums added to
 *  binSum, the integer bins of the macrocells (several: exact, whatever the
 *  order); the scale of the macrocell of regionGridKernel. */
__global__ void
macrocellAccumulateKernel(const histogramTypeIn* u, const histogramTypeIn* v,
                          const histogramTypeIn* w, const histogramTypeIn* q,
                          const int* cellStart, const int* cellCount,
                          macrocell::MacrocellGeometry geometry, int slices,
                          const BinGrid* grids, const double* scale,
                          histogramTypeOut* bins, unsigned long long* binSum) {
  const int slice = blockIdx.x % slices, tileOf = blockIdx.x / slices;
  const int m = tileOf / kMacrocellTiles, t = tileOf % kMacrocellTiles;
  __shared__ unsigned tileLow[kMacrocellTileBins], tileHigh[kMacrocellTileBins];
  for (int j = threadIdx.x; j < kMacrocellTileBins; j += blockDim.x) {
    tileLow[j] = 0;
    tileHigh[j] = 0;
  }
  __syncthreads();
  const BinGrid grid = grids[m];
  const double toFixed = scale[2 * m];
  macrocell::forEachMacrocellRow(
      geometry, m, slice, slices, cellStart, cellCount,
      [&](int begin, int end) {
        for (int base = begin; base < end; base += blockDim.x) {
          const int i = base + threadIdx.x;
          int bin = -1;
          unsigned long long weight = 0;
          if (i < end) {
            const histogramTypeIn velocity[3] = {u[i], v[i], w[i]};
            const unsigned key = MacrocellTiles::key(velocity, grid);
            if (key != kOutsideKey && key / kMacrocellTileBins == (unsigned)t) {
              bin = key % kMacrocellTileBins;
              weight = fixedPointWeight(q[i], toFixed);
            }
          }
          addToTile(tileLow, tileHigh, bin, weight);
        }
      });
  __syncthreads();
  const double toOutput = scale[2 * m + 1];
  const size_t first = (size_t)m * kMacrocellBins;
  for (int j = threadIdx.x; j < kMacrocellTileBins; j += blockDim.x) {
    const unsigned long long sum =
        (unsigned long long)tileHigh[j] << 32 | tileLow[j];
    const size_t bin = first + MacrocellTiles::bin(t, j);
    if (!binSum) // the same for the whole grid
      bins[bin] = (histogramTypeOut)((double)sum * toOutput);
    else if (sum)
      atomicAdd(&binSum[bin], sum);
  }
}

// ======= The GMM points =======

/** One block per block of bins of the compaction (sparseBins::SparseBins:
 *  its non-empty bins [offsets[b], offsets[b + 1]), their indices, and its
 *  region b / blocksPerRegion): the centre of every non-empty bin on the
 *  grid of its region. */
__global__ void binCentersKernel(const BinGrid* grids, const int* index,
                                 const int* offsets, int blocksPerRegion,
                                 histogramTypeIn* centerU,
                                 histogramTypeIn* centerV,
                                 histogramTypeIn* centerW) {
  const BinGrid grid = grids[blockIdx.x / blocksPerRegion];
  const int begin = offsets[blockIdx.x], end = offsets[blockIdx.x + 1];
  for (int i = begin + threadIdx.x; i < end; i += blockDim.x) {
    const int bin = index[i];
    centerU[i] = grid.binCenter(bin, 0);
    centerV[i] = grid.binCenter(bin, 1);
    centerW[i] = grid.binCenter(bin, 2);
  }
}

} // namespace velocityHistogram
