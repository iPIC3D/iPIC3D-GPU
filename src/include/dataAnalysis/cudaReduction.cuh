#ifndef CUDA_REDUCTION_CUH
#define CUDA_REDUCTION_CUH

#include "cudaTypeDef.cuh"
#include <assert.h>

namespace cudaReduction {

/** A block size of the block reductions: a power of 2 from 2 warps to 512. */
template <unsigned int blockSize>
inline constexpr bool isReductionBlockSize =
    blockSize >= 2 * WARP_SIZE && blockSize <= 512 &&
    (blockSize & (blockSize - 1)) == 0;

/** Exclusive prefix sum over the block (any block size); total gets the sum.
 *  buffer holds blockDim entries. */
__device__ inline int blockExclusiveScan(int value, int* buffer, int& total) {
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

struct MaxOp {
  template <typename T> __device__ T operator()(T a, T b) const {
    return max(a, b);
  }
};
struct MinOp {
  template <typename T> __device__ T operator()(T a, T b) const {
    return min(a, b);
  }
};
struct SumOp {
  template <typename T> __device__ T operator()(T a, T b) const {
    return a + b;
  }
};
/** The larger of two 64-bit integers: HIP has no max() of them for MaxOp. */
struct MaxU64Op {
  __host__ __device__ unsigned long long
  operator()(unsigned long long a, unsigned long long b) const {
    return a > b ? a : b;
  }
};

/**
 * @brief The reduction with op of one value per thread of a block of
 * blockSize threads: a tree in sdata (blockSize entries) down to 2 WARP_SIZE
 * values, then warp_reduce by the first warp. No barrier at the end: another
 * call on the same sdata needs a __syncthreads() first.
 * @return the result, in thread 0
 */
template <unsigned int blockSize, typename T, typename Op>
__device__ T blockReduce(T value, T* sdata, Op op) {
  static_assert(isReductionBlockSize<blockSize>, "block size");
  assert(blockDim.x == blockSize);
  const unsigned int tid = threadIdx.x;
  sdata[tid] = value;
  __syncthreads();
  if constexpr (blockSize >= 512) {
    if (tid < 256)
      sdata[tid] = op(sdata[tid], sdata[tid + 256]);
    __syncthreads();
  }
  if constexpr (blockSize >= 256) {
    if (tid < 128)
      sdata[tid] = op(sdata[tid], sdata[tid + 128]);
    __syncthreads();
  }
  if constexpr (blockSize >= 128 && WARP_SIZE < 64) {
    if (tid < 64)
      sdata[tid] = op(sdata[tid], sdata[tid + 64]);
    __syncthreads();
  }
  return tid < WARP_SIZE
             ? warp_reduce(op(sdata[tid], sdata[tid + WARP_SIZE]), op)
             : value;
}

} // namespace cudaReduction

#endif // CUDA_REDUCTION_CUH
