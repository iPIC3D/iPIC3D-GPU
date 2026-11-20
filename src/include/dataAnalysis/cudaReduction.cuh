#ifndef CUDA_REDUCTION_CUH
#define CUDA_REDUCTION_CUH

#include "cudaTypeDef.cuh"
#include <assert.h>
#include <type_traits>

namespace cudaReduction {

/** Loads in flight per thread of the block reductions. */
inline constexpr int kReductionLoads = 4;

/** A block size of the block reductions: a power of 2 from 2 warps to 512. */
template <unsigned int blockSize>
inline constexpr bool isReductionBlockSize =
    blockSize >= 2 * WARP_SIZE && blockSize <= 512 &&
    (blockSize & (blockSize - 1)) == 0;

/** The dynamic shared memory of a kernel as an array of T: one untyped
 *  declaration, so that the kernels of several T can share a translation
 *  unit. */
template <typename T> __device__ __forceinline__ T* sharedArray() {
  extern __shared__ __align__(sizeof(double)) unsigned char sharedBytes[];
  return reinterpret_cast<T*>(sharedBytes);
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

/**
 * @brief The reduction with op of one value per thread of the block: a tree
 * in sdata (blockSize entries) down to 2 WARP_SIZE values, then warp_reduce
 * by the first warp. No barrier at the end: another call on the same sdata
 * needs a __syncthreads() first.
 * @return the result, in thread 0
 */
template <unsigned int blockSize, typename T, typename Op>
__device__ T blockReduce(T value, T* sdata, Op op) {
  static_assert(isReductionBlockSize<blockSize>, "block size");
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

/**
 * @brief One value per block: op over the input values from identity (the
 * finite ones only with finiteOnly), with blockSize threads per block and
 * blockSize values of dynamic shared memory; each thread has kReductionLoads
 * loads in flight.
 */
template <unsigned int blockSize, bool finiteOnly, typename T, typename Op>
__device__ void reduceToBlocks(const T* g_idata, T* g_odata, unsigned int n,
                               Op op, T identity) {
  const unsigned int tid = threadIdx.x;
  T value = identity;
  for (unsigned int base = blockIdx.x * blockSize * kReductionLoads + tid;
       base < n; base += blockSize * kReductionLoads * gridDim.x) {
    T x[kReductionLoads];
#pragma unroll
    for (int k = 0; k < kReductionLoads; k++) {
      const unsigned int i = base + k * blockSize;
      x[k] = i < n ? g_idata[i] : identity;
    }
#pragma unroll
    for (int k = 0; k < kReductionLoads; k++)
      if constexpr (finiteOnly) {
        if (isfinite_fast(x[k]))
          value = op(value, x[k]);
      } else {
        value = op(value, x[k]);
      }
  }
  value = blockReduce<blockSize>(value, sharedArray<T>(), op);
  if (tid == 0)
    g_odata[blockIdx.x] = value;
}

/**
 * @brief op over the n input values from identity (the values of the blocks
 * of the block kernel); one block of WARP_SIZE threads.
 * @return the result, in thread 0
 */
template <typename T, typename Op>
__device__ T reduceInWarp(const T* g_idata, unsigned int n, Op op, T identity) {
  assert(blockDim.x == WARP_SIZE);
  assert(gridDim.x == 1);
  T value = identity;
  for (int i = threadIdx.x; i < (int)n; i += WARP_SIZE)
    value = op(value, g_idata[i]);
  return warp_reduce(value, op);
}

// ********************* max reduction *********************

/**
 * @brief The maximum of the finite values of the input, one value per block
 * (-INFINITY if none): postprocess with reduceMaxWarp.
 * @param g_idata input array, n is the size of the array
 * @param g_odata output array, the size of the array is GridSize, which means
 * the number of blocks
 * @param n the size of the input array
 */
template <typename T, unsigned int blockSize>
__global__ void reduceMax(const T* g_idata, T* g_odata, unsigned int n) {
  static_assert(std::is_floating_point_v<T>, "the start value is -INFINITY");
  reduceToBlocks<blockSize, true>(g_idata, g_odata, n, MaxOp{}, (T)-INFINITY);
}

/**
 * @brief Reduce the input array to one maximum value; one block of WARP_SIZE
 * threads.
 *
 * @param g_idata input array, n is the size of the array
 * @param g_odata output value, the size is 1
 */
template <typename T>
__global__ void reduceMaxWarp(const T* g_idata, T* g_odata, unsigned int n) {
  const T value = reduceInWarp(g_idata, n, MaxOp{}, (T)-INFINITY);
  if (threadIdx.x == 0)
    g_odata[0] = value;
}

// ********************* min reduction *********************

/** The minimum of the finite values, one value per block (INFINITY if none):
 *  postprocess with reduceMinWarp. */
template <typename T, unsigned int blockSize>
__global__ void reduceMin(const T* g_idata, T* g_odata, unsigned int n) {
  static_assert(std::is_floating_point_v<T>, "the start value is INFINITY");
  reduceToBlocks<blockSize, true>(g_idata, g_odata, n, MinOp{}, (T)INFINITY);
}

/** Reduce the input array to one minimum value; one block of WARP_SIZE
 *  threads. */
template <typename T>
__global__ void reduceMinWarp(const T* g_idata, T* g_odata, unsigned int n) {
  const T value = reduceInWarp(g_idata, n, MinOp{}, (T)INFINITY);
  if (threadIdx.x == 0)
    g_odata[0] = value;
}

// ********************* sum reduction *********************

/**
 * @brief Reduce the input array to the sum, one value per block:
 * postprocess with reduceSumWarp. Not used currently.
 *
 * @param g_idata input array, n is the size of the array
 * @param g_odata output array, the size of the array is GridSize, which means
 * the number of blocks
 * @param n the size of the input array
 */
template <typename T, unsigned int blockSize>
__global__ void reduceSum(const T* g_idata, T* g_odata, unsigned int n) {
  reduceToBlocks<blockSize, false>(g_idata, g_odata, n, SumOp{}, T(0));
}

/**
 * @brief Reduce the input array to one sum value; one block of WARP_SIZE
 * threads. Not used currently.
 *
 * @param g_idata input array, n is the size of the array
 * @param g_odata output value, the size is 1
 */
template <typename T>
__global__ void reduceSumWarp(const T* g_idata, T* g_odata, unsigned int n) {
  const T value = reduceInWarp(g_idata, n, SumOp{}, T(0));
  if (threadIdx.x == 0)
    g_odata[0] = value;
}

// ********************* pre and post process reduction *********************

enum class PreProcessType {
  none,
  minusConstThenEXP,
  multiply,
  multiplyEXP,
};

enum class PostProcessType { logAdd, divide, divideEXP };

// pre and post process reduction, it can be more elegent and flexible with cuda
// extended lambda

template <typename T, PreProcessType preProc, typename U>
__device__ __inline__ T preProcess(T value, U* preProcOprand, int eid) {
  if constexpr (preProc == PreProcessType::minusConstThenEXP) {
    return exp(value - *preProcOprand);
  } else if constexpr (preProc == PreProcessType::multiply) {
    return value * preProcOprand[eid];
  } else if constexpr (preProc == PreProcessType::multiplyEXP) {
    return value * exp(preProcOprand[eid]);
  } else {
    return value;
  }
}

template <typename T, PostProcessType postProc, typename U>
__device__ __inline__ T postProcess(T value, U* postProcOprand) {
  if constexpr (postProc == PostProcessType::logAdd) {
    return log(value) + *postProcOprand;
  } else if constexpr (postProc == PostProcessType::divide) {
    return value / *postProcOprand;
  } else if constexpr (postProc == PostProcessType::divideEXP) {
    return value / exp(*postProcOprand);
  }
}

/**
 * @brief Reduce an input array after applying an optional preprocessing step.
 * Not used currently.
 * @param g_idata Input array.
 * @param g_odata Per-block output partial sums.
 * @param n Number of valid input elements.
 * @param preProcOprand Auxiliary operand consumed by the selected preprocessing
 * rule.
 * @param weight Optional per-element weights, used only when `ifWeight` is
 * true.
 */
template <typename T, unsigned int blockSize, PreProcessType preProc,
          typename U, typename V = int, bool ifWeight = false>
__global__ void reduceSumPreProcess(T* g_idata, T* g_odata, unsigned int n,
                                    U* preProcOprand, V* weight = nullptr) {
  const unsigned int tid = threadIdx.x;
  T value = 0;
  for (unsigned int i = blockIdx.x * blockSize + tid; i < n;
       i += blockSize * gridDim.x) {
    const T term = preProcess<T, preProc, U>(g_idata[i], preProcOprand, i);
    if constexpr (ifWeight)
      value += term * weight[i];
    else
      value += term;
  }
  value = blockReduce<blockSize>(value, sharedArray<T>(), SumOp{});
  if (tid == 0)
    g_odata[blockIdx.x] = value;
}

/** One block of WARP_SIZE threads: the sum of the block sums, then the
 *  postprocessing. Not used currently. */
template <typename T, PostProcessType postProc, typename U>
__global__ void reduceSumWarpPostProcess(T* g_idata, T* g_odata, unsigned int n,
                                         U* postProcOprand) {
  const T value = reduceInWarp<T>(g_idata, n, SumOp{}, T(0));
  if (threadIdx.x == 0)
    g_odata[0] = postProcess<T, postProc, U>(value, postProcOprand);
}

} // namespace cudaReduction

#endif // CUDA_REDUCTION_CUH
