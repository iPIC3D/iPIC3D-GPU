/**
 * @file particleOutputSelect.cu
 * @brief The selection of the particles that the particle output writes,
 * ParticlesOutputTrackedFraction and ParticlesOutputMaxPerCell
 * (particleOutputSelect.cuh).
 */

#include "cudaReduction.cuh"
#include "particleOutputSelect.cuh"

#include <cstdint>

namespace {

/** Threads of the block that selects the particles of one cell (the cap). */
constexpr int kCapBlock = 128;
/** Bits of a digit of the radix selection of the keys, and its values. */
constexpr int kDigitBits = 8;
constexpr int kDigits = 1 << kDigitBits;
/** Threads of a block of the fraction kernels; kSelectScanBlock particles
 *  per block, in kFractionRounds rounds. */
constexpr int kFractionThreads = 256;
constexpr int kFractionRounds = kSelectScanBlock / kFractionThreads;
/** Threads of a block of the gathers. */
constexpr int kGatherBlock = 256;
/** Most blocks of a gather launch; a grid-stride loop covers the rest. */
constexpr unsigned int kGatherMaxBlocks = 4096;

static_assert(kDigits == 2 * kCapBlock,
              "the digit search scans two bins per thread");
static_assert(kSelectScanBlock % kFractionThreads == 0 &&
                  kFractionThreads % WARP_SIZE == 0,
              "whole rounds of whole warps");

/** The splitmix64 finalizer: a bijection of 64-bit values that spreads
 *  every input bit over the output. */
__device__ __forceinline__ std::uint64_t mix64(std::uint64_t z) {
  z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
  z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
  return z ^ (z >> 31);
}

/** The key of particle i: of its ID (distinct IDs, distinct keys), else of
 *  the bits of its coordinates. */
__device__ __forceinline__ std::uint64_t keyOf(const ParticleSelectKeys& k,
                                               std::uint32_t i) {
  if (k.id)
    return mix64(k.id[i] ^ k.salt);
  std::uint64_t h = mix64(k.salt ^ (std::uint64_t)__double_as_longlong(k.x[i]));
  h = mix64(h ^ (std::uint64_t)__double_as_longlong(k.y[i]));
  h = mix64(h ^ (std::uint64_t)__double_as_longlong(k.z[i]));
  return mix64(h ^ (std::uint64_t)__double_as_longlong(k.u[i]));
}

// ---- scans of block totals

/** The block totals scanned in place by one block, chunk after chunk; the
 *  sum of all into *total. */
__global__ void scanTotalsKernel(int* __restrict__ blockSums, int nBlocks,
                                 int* __restrict__ total) {
  __shared__ int buffer[kSelectScanBlock];
  int carry = 0;
  for (int base = 0; base < nBlocks; base += kSelectScanBlock) {
    const int b = base + threadIdx.x;
    const int value = b < nBlocks ? blockSums[b] : 0;
    int chunk;
    const int before = cudaReduction::blockExclusiveScan(value, buffer, chunk);
    if (b < nBlocks)
      blockSums[b] = carry + before;
    carry += chunk;
  }
  if (threadIdx.x == 0)
    *total = carry;
}

// ---- the fraction

/** Particle `round` of this thread in its block. */
__device__ __forceinline__ std::uint32_t fractionParticle(int round) {
  return blockIdx.x * kSelectScanBlock + round * kFractionThreads + threadIdx.x;
}

/** The selected particles of a block of particles into blockSums. */
__global__ void __launch_bounds__(kFractionThreads)
    fractionCountKernel(ParticleSelectKeys keys, std::uint32_t nop,
                        std::uint64_t limit, int* __restrict__ blockSums) {
  int count = 0;
  for (int round = 0; round < kFractionRounds; round++) {
    const std::uint32_t i = fractionParticle(round);
    count += __syncthreads_count(i < nop && keyOf(keys, i) <= limit);
  }
  if (threadIdx.x == 0)
    blockSums[blockIdx.x] = count;
}

/** The indices of the selected particles of a block of particles, in their
 *  order from the first output slot of the block: each at its rank among
 *  the selected ones of its warp (a ballot) and of the warps before. */
__global__ void __launch_bounds__(kFractionThreads)
    fractionSelectKernel(ParticleSelectKeys keys, std::uint32_t nop,
                         std::uint64_t limit, const int* __restrict__ blockSums,
                         std::uint32_t* __restrict__ idx) {
  __shared__ int warpTotal[kFractionThreads / WARP_SIZE];
  const int lane = threadIdx.x % WARP_SIZE, warp = threadIdx.x / WARP_SIZE;
  const unsigned long long lanesBelow = (1ull << lane) - 1;
  int next = blockSums[blockIdx.x];
  for (int round = 0; round < kFractionRounds; round++) {
    const std::uint32_t i = fractionParticle(round);
    const bool selected = i < nop && keyOf(keys, i) <= limit;
    const unsigned long long ballot = __ballot_sync(WARP_FULL_MASK, selected);
    if (lane == 0)
      warpTotal[warp] = __popcll(ballot);
    __syncthreads();
    int rank = __popcll(ballot & lanesBelow), roundTotal = 0;
    for (int w = 0; w < kFractionThreads / WARP_SIZE; w++) {
      rank += w < warp ? warpTotal[w] : 0;
      roundTotal += warpTotal[w];
    }
    if (selected)
      idx[next + rank] = i;
    next += roundTotal;
    __syncthreads(); // warpTotal is rewritten by the next round
  }
}

// ---- the cap

/** The particles written in each cell of a block of cells, min(count, K),
 *  scanned into offset (exclusive); the block's total into blockSums. */
__global__ void capScanKernel(const int* __restrict__ count, int numCells,
                              int K, int* __restrict__ offset,
                              int* __restrict__ blockSums) {
  __shared__ int buffer[kSelectScanBlock];
  const int c = blockIdx.x * kSelectScanBlock + threadIdx.x;
  const int kept = c < numCells ? min(count[c], K) : 0;
  int total;
  const int before = cudaReduction::blockExclusiveScan(kept, buffer, total);
  if (c < numCells)
    offset[c] = before;
  if (threadIdx.x == 0)
    blockSums[blockIdx.x] = total;
}

/** The scanned total of the blocks before it (0 for the first), added to a
 *  block of cells. */
__global__ void scanAddKernel(int* __restrict__ offset, int numCells,
                              const int* __restrict__ blockSums) {
  const int c = blockIdx.x * kSelectScanBlock + threadIdx.x;
  if (c < numCells)
    offset[c] += blockSums[blockIdx.x];
}

/**
 * One block per cell: of its n particles, all when n <= K (their charges
 * unchanged), else the K of smallest key. A radix selection of the K-th
 * smallest key t, digit after digit (8 bits) from the highest, until the
 * keys with the digits found are exactly the ones still to keep (usually
 * after one or two digits); then the keys below and, of those with t's
 * digits (equal keys after all 8 only with equal coordinates without IDs),
 * as many as complete K. Their charges into qOut (if not null) times
 * Q / Q_kept. The order of the written particles within the cell is not
 * fixed.
 */
__global__ void __launch_bounds__(kCapBlock) capSelectKernel(
    const int* __restrict__ count, const int* __restrict__ start,
    const int* __restrict__ offset, int K, ParticleSelectKeys keys,
    const cudaPclType_Q* __restrict__ q, std::uint32_t* __restrict__ idx,
    cudaPclType_Q* __restrict__ qOut) {
  const int c = blockIdx.x;
  const int n = count[c];
  if (n == 0)
    return;
  const std::uint32_t first = start[c];
  const int out = offset[c];
  if (n <= K) {
    for (int j = threadIdx.x; j < n; j += blockDim.x) {
      idx[out + j] = first + j;
      if (qOut)
        qOut[out + j] = q[first + j];
    }
    return;
  }

  __shared__ unsigned int histogram[kDigits];
  __shared__ int scanBuffer[kCapBlock];
  __shared__ double partial[kCapBlock];
  __shared__ unsigned long long prefix; // the digits of t found so far
  __shared__ int rank;  // the rank of t among the keys with that prefix
  __shared__ bool done; // the keys with that prefix: exactly rank
  __shared__ int kept, keptEqual;
  __shared__ double factor;
  if (threadIdx.x == 0) {
    prefix = 0;
    rank = K;
    done = false;
    kept = 0;
    keptEqual = 0;
  }
  __syncthreads();

  unsigned long long found = 0; // mask of the digits found
  for (int shift = 64 - kDigitBits; shift >= 0 && !done; shift -= kDigitBits) {
    for (int d = threadIdx.x; d < kDigits; d += blockDim.x)
      histogram[d] = 0;
    __syncthreads();
    const unsigned long long p = prefix;
    const int r = rank;
    for (int j = threadIdx.x; j < n; j += blockDim.x) {
      const unsigned long long key = keyOf(keys, first + j);
      if ((key & found) == p)
        atomicAdd(&histogram[(key >> shift) & (kDigits - 1)], 1u);
    }
    __syncthreads();
    // the digit of the r-th of these keys: a scan of the bins, two per
    // thread; the one thread whose bins hold it records it
    const int d0 = 2 * threadIdx.x;
    const int b0 = histogram[d0], b1 = histogram[d0 + 1];
    int total;
    const int below =
        cudaReduction::blockExclusiveScan(b0 + b1, scanBuffer, total);
    if (below < r && r <= below + b0 + b1) {
      const bool low = r <= below + b0;
      const int rest = r - below - (low ? 0 : b0);
      prefix = p | ((unsigned long long)(low ? d0 : d0 + 1) << shift);
      rank = rest;
      done = rest == (low ? b0 : b1);
    }
    found |= (unsigned long long)(kDigits - 1) << shift;
    __syncthreads();
  }

  // K - rank keys below t's digits, rank with them
  const unsigned long long t = prefix;
  const int equalToKeep = rank;
  double cellCharge = 0.0, keptCharge = 0.0;
  for (int j = threadIdx.x; j < n; j += blockDim.x) {
    const unsigned long long masked = keyOf(keys, first + j) & found;
    const bool keep =
        masked < t || (masked == t && atomicAdd(&keptEqual, 1) < equalToKeep);
    const double qp = qOut ? q[first + j] : 0.0;
    cellCharge += qp;
    if (keep) {
      const int slot = out + atomicAdd(&kept, 1);
      idx[slot] = first + j;
      if (qOut)
        qOut[slot] = qp;
      keptCharge += qp;
    }
  }
  if (!qOut)
    return;
  const double all = cudaReduction::blockReduce<kCapBlock>(
      cellCharge, partial, cudaReduction::SumOp());
  __syncthreads(); // blockReduce reuses partial
  const double some = cudaReduction::blockReduce<kCapBlock>(
      keptCharge, partial, cudaReduction::SumOp());
  if (threadIdx.x == 0) // equal charges: n / K
    factor = some != 0.0 ? all / some : (double)n / K;
  __syncthreads(); // factor, and the charges of the cell in qOut
  for (int o = threadIdx.x; o < K; o += blockDim.x)
    qOut[out + o] *= factor;
}

// ---- the arrays

template <typename T>
__global__ void gatherKernel(T* __restrict__ dst, const T* __restrict__ src,
                             const std::uint32_t* __restrict__ idx,
                             std::uint32_t n) {
  for (std::uint32_t o = blockIdx.x * blockDim.x + threadIdx.x; o < n;
       o += blockDim.x * gridDim.x)
    dst[o] = src[idx[o]];
}

__global__ void gatherScaledKernel(cudaPclType_Q* __restrict__ dst,
                                   const cudaPclType_Q* __restrict__ q,
                                   const std::uint32_t* __restrict__ idx,
                                   std::uint32_t n, double scale) {
  for (std::uint32_t o = blockIdx.x * blockDim.x + threadIdx.x; o < n;
       o += blockDim.x * gridDim.x)
    dst[o] = q[idx[o]] * scale;
}

unsigned int gatherBlocks(std::uint32_t n) {
  const unsigned int blocks = (n + kGatherBlock - 1) / kGatherBlock;
  return blocks < kGatherMaxBlocks ? blocks : kGatherMaxBlocks;
}

} // namespace

void launchFractionCount(const ParticleSelectKeys& keys, std::uint32_t nop,
                         std::uint64_t limit, int* blockSums, int* total,
                         cudaStream_t stream) {
  if (nop == 0) {
    cudaErrChk(cudaMemsetAsync(total, 0, sizeof(int), stream));
    return;
  }
  const int nBlocks = (int)((nop + kSelectScanBlock - 1) / kSelectScanBlock);
  fractionCountKernel<<<nBlocks, kFractionThreads, 0, stream>>>(
      keys, nop, limit, blockSums);
  scanTotalsKernel<<<1, kSelectScanBlock, 0, stream>>>(blockSums, nBlocks,
                                                       total);
  cudaErrChk(cudaGetLastError());
}

void launchFractionSelect(const ParticleSelectKeys& keys, std::uint32_t nop,
                          std::uint64_t limit, const int* blockSums,
                          std::uint32_t* idx, cudaStream_t stream) {
  if (nop == 0)
    return;
  const int nBlocks = (int)((nop + kSelectScanBlock - 1) / kSelectScanBlock);
  fractionSelectKernel<<<nBlocks, kFractionThreads, 0, stream>>>(
      keys, nop, limit, blockSums, idx);
  cudaErrChk(cudaGetLastError());
}

void launchCapCount(const int* count, int numCells, int maxPerCell, int* offset,
                    int* blockSums, int* total, cudaStream_t stream) {
  if (numCells == 0) {
    cudaErrChk(cudaMemsetAsync(total, 0, sizeof(int), stream));
    return;
  }
  const int nBlocks = (numCells + kSelectScanBlock - 1) / kSelectScanBlock;
  capScanKernel<<<nBlocks, kSelectScanBlock, 0, stream>>>(
      count, numCells, maxPerCell, offset, blockSums);
  scanTotalsKernel<<<1, kSelectScanBlock, 0, stream>>>(blockSums, nBlocks,
                                                       total);
  if (nBlocks > 1)
    scanAddKernel<<<nBlocks, kSelectScanBlock, 0, stream>>>(offset, numCells,
                                                            blockSums);
  cudaErrChk(cudaGetLastError());
}

void launchCapSelect(const int* count, const int* start, const int* offset,
                     int numCells, int maxPerCell,
                     const ParticleSelectKeys& keys, const cudaPclType_Q* q,
                     std::uint32_t* idx, cudaPclType_Q* qOut,
                     cudaStream_t stream) {
  if (numCells == 0)
    return;
  capSelectKernel<<<numCells, kCapBlock, 0, stream>>>(
      count, start, offset, maxPerCell, keys, q, idx, qOut);
  cudaErrChk(cudaGetLastError());
}

template <typename T>
void launchGather(T* dst, const T* src, const std::uint32_t* idx,
                  std::uint32_t n, cudaStream_t stream) {
  if (n == 0)
    return;
  gatherKernel<T>
      <<<gatherBlocks(n), kGatherBlock, 0, stream>>>(dst, src, idx, n);
  cudaErrChk(cudaGetLastError());
}

template void launchGather<double>(double*, const double*, const std::uint32_t*,
                                   std::uint32_t, cudaStream_t);
template void launchGather<cudaPclType_ID>(cudaPclType_ID*,
                                           const cudaPclType_ID*,
                                           const std::uint32_t*, std::uint32_t,
                                           cudaStream_t);

void launchGatherScaled(cudaPclType_Q* dst, const cudaPclType_Q* q,
                        const std::uint32_t* idx, std::uint32_t n, double scale,
                        cudaStream_t stream) {
  if (n == 0)
    return;
  gatherScaledKernel<<<gatherBlocks(n), kGatherBlock, 0, stream>>>(dst, q, idx,
                                                                   n, scale);
  cudaErrChk(cudaGetLastError());
}
