/**
 * @file GPUBlas.cu
 * @brief CUDA kernel implementations for GPU BLAS-like operations.
 *        See GPUBlas.cuh for API documentation.
 */

#include "GPUBlas.cuh"
#include "GPUFieldArray.cuh" // for fillKernelLaunch
#include <cstdio>
#include <cstdlib>

// =========================================================================
//  Kernel configuration
// =========================================================================
static constexpr int BLAS_BLOCK = 256;
static constexpr int MAX_WARPS = BLAS_BLOCK / WARP_SIZE;
static inline size_t divCeil(size_t n, size_t d) { return (n + d - 1) / d; }

// =========================================================================
//  Element-wise CUDA kernels
// =========================================================================

template <typename T>
__global__ void k_scale(T* __restrict__ d, T alfa, size_t n) {
  size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
  if (i < n)
    d[i] *= alfa;
}

template <typename T>
__global__ void k_scaleCopy(T* __restrict__ dst, const T* __restrict__ src,
                            T alfa, size_t n) {
  size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
  if (i < n)
    dst[i] = src[i] * alfa;
}

template <typename T>
__global__ void k_sum(T* __restrict__ dst, const T* __restrict__ src,
                      size_t n) {
  size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
  if (i < n)
    dst[i] += src[i];
}

template <typename T>
__global__ void k_sub(T* __restrict__ dst, const T* __restrict__ src,
                      size_t n) {
  size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
  if (i < n)
    dst[i] -= src[i];
}

template <typename T>
__global__ void k_subRes(T* __restrict__ res, const T* __restrict__ a,
                         const T* __restrict__ b, size_t n) {
  size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
  if (i < n)
    res[i] = a[i] - b[i];
}

template <typename T>
__global__ void k_addscale(T alfa, T* __restrict__ dst,
                           const T* __restrict__ src, size_t n) {
  size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
  if (i < n)
    dst[i] += alfa * src[i];
}

/** Not used (gpuAddscale2). */
template <typename T>
__global__ void k_addscale2(T alfa, T beta, T* __restrict__ dst,
                            const T* __restrict__ src, size_t n) {
  size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
  if (i < n)
    dst[i] = beta * dst[i] + alfa * src[i];
}

template <typename T>
__global__ void k_addConst(T* __restrict__ d, T c, size_t n) {
  size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
  if (i < n)
    d[i] += c;
}

/** Not used (gpuNeg). */
template <typename T> __global__ void k_neg(T* __restrict__ d, size_t n) {
  size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
  if (i < n)
    d[i] = -d[i];
}

// =========================================================================
//  Element-wise host wrappers
// =========================================================================

void gpuEqValue(cudaSolverType* d, cudaSolverType val, size_t n,
                cudaStream_t stream) {
  if (n == 0)
    return;
  if (val == cudaSolverType{}) {
    cudaErrChk(cudaMemsetAsync(d, 0, n * sizeof(cudaSolverType), stream));
  } else {
    GPUFieldArray3::fillKernelLaunch(d, val, n, stream);
  }
}

void gpuScale(cudaSolverType* d, cudaSolverType alfa, size_t n,
              cudaStream_t stream) {
  if (n == 0)
    return;
  k_scale<<<divCeil(n, BLAS_BLOCK), BLAS_BLOCK, 0, stream>>>(d, alfa, n);
  cudaErrChk(cudaGetLastError());
}

void gpuScaleCopy(cudaSolverType* dst, const cudaSolverType* src,
                  cudaSolverType alfa, size_t n, cudaStream_t stream) {
  if (n == 0)
    return;
  k_scaleCopy<<<divCeil(n, BLAS_BLOCK), BLAS_BLOCK, 0, stream>>>(dst, src, alfa,
                                                                 n);
  cudaErrChk(cudaGetLastError());
}

void gpuSum(cudaSolverType* dst, const cudaSolverType* src, size_t n,
            cudaStream_t stream) {
  if (n == 0)
    return;
  k_sum<<<divCeil(n, BLAS_BLOCK), BLAS_BLOCK, 0, stream>>>(dst, src, n);
  cudaErrChk(cudaGetLastError());
}

void gpuSub(cudaSolverType* dst, const cudaSolverType* src, size_t n,
            cudaStream_t stream) {
  if (n == 0)
    return;
  k_sub<<<divCeil(n, BLAS_BLOCK), BLAS_BLOCK, 0, stream>>>(dst, src, n);
  cudaErrChk(cudaGetLastError());
}

void gpuSubRes(cudaSolverType* res, const cudaSolverType* a,
               const cudaSolverType* b, size_t n, cudaStream_t stream) {
  if (n == 0)
    return;
  k_subRes<<<divCeil(n, BLAS_BLOCK), BLAS_BLOCK, 0, stream>>>(res, a, b, n);
  cudaErrChk(cudaGetLastError());
}

void gpuAddscale(cudaSolverType alfa, cudaSolverType* dst,
                 const cudaSolverType* src, size_t n, cudaStream_t stream) {
  if (n == 0)
    return;
  k_addscale<<<divCeil(n, BLAS_BLOCK), BLAS_BLOCK, 0, stream>>>(alfa, dst, src,
                                                                n);
  cudaErrChk(cudaGetLastError());
}

void gpuAddscale2(cudaSolverType alfa, cudaSolverType beta, cudaSolverType* dst,
                  const cudaSolverType* src, size_t n, cudaStream_t stream) {
  if (n == 0)
    return;
  k_addscale2<<<divCeil(n, BLAS_BLOCK), BLAS_BLOCK, 0, stream>>>(alfa, beta,
                                                                 dst, src, n);
  cudaErrChk(cudaGetLastError());
}

void gpuAddConst(cudaSolverType* d, cudaSolverType c, size_t n,
                 cudaStream_t stream) {
  if (n == 0)
    return;
  k_addConst<<<divCeil(n, BLAS_BLOCK), BLAS_BLOCK, 0, stream>>>(d, c, n);
  cudaErrChk(cudaGetLastError());
}

void gpuNeg(cudaSolverType* d, size_t n, cudaStream_t stream) {
  if (n == 0)
    return;
  k_neg<<<divCeil(n, BLAS_BLOCK), BLAS_BLOCK, 0, stream>>>(d, n);
  cudaErrChk(cudaGetLastError());
}

void gpuEq(cudaSolverType* dst, const cudaSolverType* src, size_t n,
           cudaStream_t stream) {
  if (n == 0)
    return;
  cudaErrChk(cudaMemcpyAsync(dst, src, n * sizeof(cudaSolverType),
                             cudaMemcpyDeviceToDevice, stream));
}

// =========================================================================
//  Fused triple kernels – 3 arrays in one launch
// =========================================================================

/** Not used (gpuNeg3, a unit-test reference). */
template <typename T>
__global__ void k_neg3(T* __restrict__ dX, T* __restrict__ dY,
                       T* __restrict__ dZ, size_t n) {
  size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
  if (i < n) {
    dX[i] = -dX[i];
    dY[i] = -dY[i];
    dZ[i] = -dZ[i];
  }
}

/** Not used (gpuSub3, a unit-test reference). */
template <typename T>
__global__ void k_sub3(T* __restrict__ dstX, const T* __restrict__ srcX,
                       T* __restrict__ dstY, const T* __restrict__ srcY,
                       T* __restrict__ dstZ, const T* __restrict__ srcZ,
                       size_t n) {
  size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
  if (i < n) {
    dstX[i] -= srcX[i];
    dstY[i] -= srcY[i];
    dstZ[i] -= srcZ[i];
  }
}

template <typename T>
__global__ void k_scale3(T* __restrict__ dX, T* __restrict__ dY,
                         T* __restrict__ dZ, T alfa, size_t n) {
  size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
  if (i < n) {
    dX[i] *= alfa;
    dY[i] *= alfa;
    dZ[i] *= alfa;
  }
}

template <typename T>
__global__ void k_sumAddTwo3(T* __restrict__ dstX, const T* __restrict__ srcAX,
                             const T* __restrict__ srcBX, T* __restrict__ dstY,
                             const T* __restrict__ srcAY,
                             const T* __restrict__ srcBY, T* __restrict__ dstZ,
                             const T* __restrict__ srcAZ,
                             const T* __restrict__ srcBZ, size_t n) {
  size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
  if (i < n) {
    dstX[i] += srcAX[i] + srcBX[i];
    dstY[i] += srcAY[i] + srcBY[i];
    dstZ[i] += srcAZ[i] + srcBZ[i];
  }
}

template <typename T>
__global__ void k_scaleCopy3(T* __restrict__ dstX, const T* __restrict__ srcX,
                             T* __restrict__ dstY, const T* __restrict__ srcY,
                             T* __restrict__ dstZ, const T* __restrict__ srcZ,
                             T alfa, size_t n) {
  size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
  if (i < n) {
    dstX[i] = srcX[i] * alfa;
    dstY[i] = srcY[i] * alfa;
    dstZ[i] = srcZ[i] * alfa;
  }
}

template <typename T>
__global__ void k_sum3(T* __restrict__ dstX, const T* __restrict__ srcX,
                       T* __restrict__ dstY, const T* __restrict__ srcY,
                       T* __restrict__ dstZ, const T* __restrict__ srcZ,
                       size_t n) {
  size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
  if (i < n) {
    dstX[i] += srcX[i];
    dstY[i] += srcY[i];
    dstZ[i] += srcZ[i];
  }
}

template <typename T>
__global__ void k_addscale3(T alfa, T* __restrict__ dstX,
                            const T* __restrict__ srcX, T* __restrict__ dstY,
                            const T* __restrict__ srcY, T* __restrict__ dstZ,
                            const T* __restrict__ srcZ, size_t n) {
  size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
  if (i < n) {
    dstX[i] += alfa * srcX[i];
    dstY[i] += alfa * srcY[i];
    dstZ[i] += alfa * srcZ[i];
  }
}

template <typename T>
__global__ void k_addscale2_3(T alfa, T beta, T* __restrict__ dstX,
                              const T* __restrict__ srcX, T* __restrict__ dstY,
                              const T* __restrict__ srcY, T* __restrict__ dstZ,
                              const T* __restrict__ srcZ, size_t n) {
  size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
  if (i < n) {
    dstX[i] = alfa * srcX[i] + beta * dstX[i];
    dstY[i] = alfa * srcY[i] + beta * dstY[i];
    dstZ[i] = alfa * srcZ[i] + beta * dstZ[i];
  }
}

// ---- Fused triple host wrappers ----

void gpuNeg3(cudaSolverType* dX, cudaSolverType* dY, cudaSolverType* dZ,
             size_t n, cudaStream_t stream) {
  if (n == 0)
    return;
  k_neg3<<<divCeil(n, BLAS_BLOCK), BLAS_BLOCK, 0, stream>>>(dX, dY, dZ, n);
  cudaErrChk(cudaGetLastError());
}

void gpuSub3(cudaSolverType* dstX, const cudaSolverType* srcX,
             cudaSolverType* dstY, const cudaSolverType* srcY,
             cudaSolverType* dstZ, const cudaSolverType* srcZ, size_t n,
             cudaStream_t stream) {
  if (n == 0)
    return;
  k_sub3<<<divCeil(n, BLAS_BLOCK), BLAS_BLOCK, 0, stream>>>(
      dstX, srcX, dstY, srcY, dstZ, srcZ, n);
  cudaErrChk(cudaGetLastError());
}

void gpuScale3(cudaSolverType* dX, cudaSolverType* dY, cudaSolverType* dZ,
               cudaSolverType alfa, size_t n, cudaStream_t stream) {
  if (n == 0)
    return;
  k_scale3<<<divCeil(n, BLAS_BLOCK), BLAS_BLOCK, 0, stream>>>(dX, dY, dZ, alfa,
                                                              n);
  cudaErrChk(cudaGetLastError());
}

void gpuSumAddTwo3(cudaSolverType* dstX, const cudaSolverType* srcAX,
                   const cudaSolverType* srcBX, cudaSolverType* dstY,
                   const cudaSolverType* srcAY, const cudaSolverType* srcBY,
                   cudaSolverType* dstZ, const cudaSolverType* srcAZ,
                   const cudaSolverType* srcBZ, size_t n, cudaStream_t stream) {
  if (n == 0)
    return;
  k_sumAddTwo3<<<divCeil(n, BLAS_BLOCK), BLAS_BLOCK, 0, stream>>>(
      dstX, srcAX, srcBX, dstY, srcAY, srcBY, dstZ, srcAZ, srcBZ, n);
  cudaErrChk(cudaGetLastError());
}

void gpuSetAll0_N(cudaSolverType** d_ptrs, int nFields, size_t n,
                  cudaStream_t stream) {
  for (int f = 0; f < nFields; ++f)
    cudaErrChk(
        cudaMemsetAsync(d_ptrs[f], 0, n * sizeof(cudaSolverType), stream));
}

void gpuScaleCopy3(cudaSolverType* dstX, const cudaSolverType* srcX,
                   cudaSolverType* dstY, const cudaSolverType* srcY,
                   cudaSolverType* dstZ, const cudaSolverType* srcZ,
                   cudaSolverType alfa, size_t n, cudaStream_t stream) {
  if (n == 0)
    return;
  k_scaleCopy3<<<divCeil(n, BLAS_BLOCK), BLAS_BLOCK, 0, stream>>>(
      dstX, srcX, dstY, srcY, dstZ, srcZ, alfa, n);
  cudaErrChk(cudaGetLastError());
}

void gpuSum3(cudaSolverType* dstX, const cudaSolverType* srcX,
             cudaSolverType* dstY, const cudaSolverType* srcY,
             cudaSolverType* dstZ, const cudaSolverType* srcZ, size_t n,
             cudaStream_t stream) {
  if (n == 0)
    return;
  k_sum3<<<divCeil(n, BLAS_BLOCK), BLAS_BLOCK, 0, stream>>>(
      dstX, srcX, dstY, srcY, dstZ, srcZ, n);
  cudaErrChk(cudaGetLastError());
}

void gpuAddscale3(cudaSolverType alfa, cudaSolverType* dstX,
                  const cudaSolverType* srcX, cudaSolverType* dstY,
                  const cudaSolverType* srcY, cudaSolverType* dstZ,
                  const cudaSolverType* srcZ, size_t n, cudaStream_t stream) {
  if (n == 0)
    return;
  k_addscale3<<<divCeil(n, BLAS_BLOCK), BLAS_BLOCK, 0, stream>>>(
      alfa, dstX, srcX, dstY, srcY, dstZ, srcZ, n);
  cudaErrChk(cudaGetLastError());
}

void gpuAddscale2_3(cudaSolverType alfa, cudaSolverType beta,
                    cudaSolverType* dstX, const cudaSolverType* srcX,
                    cudaSolverType* dstY, const cudaSolverType* srcY,
                    cudaSolverType* dstZ, const cudaSolverType* srcZ, size_t n,
                    cudaStream_t stream) {
  if (n == 0)
    return;
  k_addscale2_3<<<divCeil(n, BLAS_BLOCK), BLAS_BLOCK, 0, stream>>>(
      alfa, beta, dstX, srcX, dstY, srcY, dstZ, srcZ, n);
  cudaErrChk(cudaGetLastError());
}

// =========================================================================
//  Reduction CUDA kernel – warp-shuffle + atomicAdd
// =========================================================================

/**
 * Adds the sum of the block's per-thread values to *d_result: warp shuffle,
 * the warp sums through shared memory, then one atomicAdd per block.  Every
 * thread of the block must call it.
 */
template <typename T>
__device__ __forceinline__ void blockSumAtomicAdd(T sum, T* d_result) {
  sum = warp_reduce_sum(sum);

  __shared__ T warpSums[MAX_WARPS];
  const int lane = threadIdx.x % WARP_SIZE;
  const int warpId = threadIdx.x / WARP_SIZE;
  if (lane == 0)
    warpSums[warpId] = sum;
  __syncthreads();

  if (warpId == 0) {
    sum = (lane < MAX_WARPS) ? warpSums[lane] : T(0);
    sum = warp_reduce_sum(sum);
    if (lane == 0)
      atomicAdd(d_result, sum);
  }
}

/**
 * Generic dot-product reduction kernel.
 * Each block reduces a chunk using grid-stride loop + warp shuffle.
 * The final per-block scalar is atomically added to *d_result.
 * d_result MUST be zeroed before launch.
 */
template <typename T>
__global__ void k_dotReduce(const T* __restrict__ a, const T* __restrict__ b,
                            size_t n, T* __restrict__ d_result) {
  T sum = 0.0;
  for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n;
       i += (size_t)blockDim.x * gridDim.x) {
    sum += a[i] * b[i];
  }
  blockSumAtomicAdd(sum, d_result);
}

/** Same structure but computes norm2 (a[i]*a[i]). */
template <typename T>
__global__ void k_norm2Reduce(const T* __restrict__ a, size_t n,
                              T* __restrict__ d_result) {
  T sum = 0.0;
  for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n;
       i += (size_t)blockDim.x * gridDim.x) {
    T v = a[i];
    sum += v * v;
  }
  blockSumAtomicAdd(sum, d_result);
}

/** Same structure, sum of a[i]. */
template <typename T>
__global__ void k_sumReduce(const T* __restrict__ a, size_t n,
                            T* __restrict__ d_result) {
  T sum = 0.0;
  for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n;
       i += (size_t)blockDim.x * gridDim.x)
    sum += a[i];
  blockSumAtomicAdd(sum, d_result);
}

// ---- grid size limiter for reductions (avoid excessive blocks) ----
static inline int reduceGridSize(size_t n) {
  int blocks = (int)divCeil(n, BLAS_BLOCK);
  // Cap at 1024 blocks – grid-stride loop handles the rest
  return (blocks > 1024) ? 1024 : blocks;
}

// =========================================================================
//  Reduction host wrappers
// =========================================================================

cudaSolverType gpuDot(const cudaSolverType* a, const cudaSolverType* b,
                      size_t n, cudaSolverType* d_scratch,
                      cudaStream_t stream) {
  cudaErrChk(cudaMemsetAsync(d_scratch, 0, sizeof(cudaSolverType), stream));
  if (n > 0) {
    k_dotReduce<<<reduceGridSize(n), BLAS_BLOCK, 0, stream>>>(a, b, n,
                                                              d_scratch);
    cudaErrChk(cudaGetLastError());
  }
  cudaSolverType result;
  cudaErrChk(cudaMemcpyAsync(&result, d_scratch, sizeof(cudaSolverType),
                             cudaMemcpyDeviceToHost, stream));
  cudaErrChk(cudaStreamSynchronize(stream));
  return result;
}

cudaSolverType gpuNorm2(const cudaSolverType* a, size_t n,
                        cudaSolverType* d_scratch, cudaStream_t stream) {
  cudaErrChk(cudaMemsetAsync(d_scratch, 0, sizeof(cudaSolverType), stream));
  if (n > 0) {
    k_norm2Reduce<<<reduceGridSize(n), BLAS_BLOCK, 0, stream>>>(a, n,
                                                                d_scratch);
    cudaErrChk(cudaGetLastError());
  }
  cudaSolverType result;
  cudaErrChk(cudaMemcpyAsync(&result, d_scratch, sizeof(cudaSolverType),
                             cudaMemcpyDeviceToHost, stream));
  cudaErrChk(cudaStreamSynchronize(stream));
  return result;
}

cudaSolverType gpuReduceSum(const cudaSolverType* a, size_t n,
                            cudaSolverType* d_scratch, cudaStream_t stream) {
  cudaErrChk(cudaMemsetAsync(d_scratch, 0, sizeof(cudaSolverType), stream));
  if (n > 0) {
    k_sumReduce<<<reduceGridSize(n), BLAS_BLOCK, 0, stream>>>(a, n, d_scratch);
    cudaErrChk(cudaGetLastError());
  }
  cudaSolverType result;
  cudaErrChk(cudaMemcpyAsync(&result, d_scratch, sizeof(cudaSolverType),
                             cudaMemcpyDeviceToHost, stream));
  cudaErrChk(cudaStreamSynchronize(stream));
  return result;
}

// ---- Async variants (no stream sync — caller must sync before reading) ----

void gpuNorm2_async(const cudaSolverType* a, size_t n,
                    cudaSolverType* d_scratch, cudaSolverType* h_result,
                    cudaStream_t stream) {
  cudaErrChk(cudaMemsetAsync(d_scratch, 0, sizeof(cudaSolverType), stream));
  if (n > 0) {
    k_norm2Reduce<<<reduceGridSize(n), BLAS_BLOCK, 0, stream>>>(a, n,
                                                                d_scratch);
    cudaErrChk(cudaGetLastError());
  }
  cudaErrChk(cudaMemcpyAsync(h_result, d_scratch, sizeof(cudaSolverType),
                             cudaMemcpyDeviceToHost, stream));
}

void gpuDot_async(const cudaSolverType* a, const cudaSolverType* b, size_t n,
                  cudaSolverType* d_scratch, cudaSolverType* h_result,
                  cudaStream_t stream) {
  cudaErrChk(cudaMemsetAsync(d_scratch, 0, sizeof(cudaSolverType), stream));
  if (n > 0) {
    k_dotReduce<<<reduceGridSize(n), BLAS_BLOCK, 0, stream>>>(a, b, n,
                                                              d_scratch);
    cudaErrChk(cudaGetLastError());
  }
  cudaErrChk(cudaMemcpyAsync(h_result, d_scratch, sizeof(cudaSolverType),
                             cudaMemcpyDeviceToHost, stream));
}

// =========================================================================
//  Batched dot+norm kernel for GMRES Arnoldi
// =========================================================================

// Most dot products per call of gpuBatchedDotNorm (k + 1, at most the GMRES
// restart length): the instantiations of k_batchedDotNorm
constexpr int kBatchedDotMax = 32;

/**
 * Fused kernel: the KP1 dot products dot(w, V[j]), j < KP1, and norm2(w) in
 * one grid-stride pass over w.  KP1 is a template parameter so that the
 * KP1 + 1 per-thread sums stay in registers (indexed with a runtime count
 * they would live in local memory).
 *
 * d_out must have KP1 + 1 doubles, zeroed before launch: d_out[j] =
 * dot(w, V[j]), d_out[KP1] = norm2(w).  V[j] starts at V_base + j*stride.
 */
template <typename T, int KP1>
__global__ void k_batchedDotNorm(const T* __restrict__ w,
                                 const T* __restrict__ V_base, size_t stride,
                                 size_t n, T* __restrict__ d_out) {
  constexpr int total = KP1 + 1;
  T sums[total];
#pragma unroll
  for (int j = 0; j < total; j++)
    sums[j] = 0.0;

  for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n;
       i += (size_t)blockDim.x * gridDim.x) {
    const T wi = w[i];
#pragma unroll
    for (int j = 0; j < KP1; j++)
      sums[j] += wi * V_base[j * stride + i];
    sums[KP1] += wi * wi;
  }

  // Warp-level reduction for all accumulators
#pragma unroll
  for (int j = 0; j < total; j++)
    sums[j] = warp_reduce_sum(sums[j]);

  // Per-warp partial sums to shared memory
  __shared__ T smem[MAX_WARPS * total];
  const int lane = threadIdx.x % WARP_SIZE;
  const int warpId = threadIdx.x / WARP_SIZE;
  if (lane == 0) {
#pragma unroll
    for (int j = 0; j < total; j++)
      smem[warpId * total + j] = sums[j];
  }
  __syncthreads();

  // First warp reduces across warps
  if (warpId == 0) {
    for (int j = 0; j < total; j++) {
      T val = (lane < MAX_WARPS) ? smem[lane * total + j] : 0.0;
      val = warp_reduce_sum(val);
      if (lane == 0)
        atomicAdd(&d_out[j], val);
    }
  }
}

/** Launch k_batchedDotNorm<KP1> for the runtime count kp1, KP1 <= kp1 <=
 *  kBatchedDotMax (host-side dispatch; kp1 = 0 computes norm2(w) only). */
template <int KP1>
static void launchBatchedDotNorm(int kp1, int grid, cudaStream_t stream,
                                 const cudaSolverType* w,
                                 const cudaSolverType* V_base, size_t stride,
                                 size_t n, cudaSolverType* d_out) {
  if constexpr (KP1 < kBatchedDotMax) {
    if (kp1 > KP1)
      return launchBatchedDotNorm<KP1 + 1>(kp1, grid, stream, w, V_base, stride,
                                           n, d_out);
  }
  k_batchedDotNorm<cudaSolverType, KP1>
      <<<grid, BLAS_BLOCK, 0, stream>>>(w, V_base, stride, n, d_out);
}

void gpuBatchedDotNorm(const cudaSolverType* w, const cudaSolverType* V_base,
                       size_t stride, int k, size_t n, cudaSolverType* d_out,
                       cudaStream_t stream) {
  const int kp1 = k + 1; // number of dot products
  // more vectors than the instantiated kernels
  if (kp1 > kBatchedDotMax) {
    fprintf(stderr,
            "gpuBatchedDotNorm: %d vectors exceed the %d the kernel "
            "supports (GMRES restart length too large)\n",
            kp1, kBatchedDotMax);
    abort();
  }

  // Zero the output array (kp1 dot products + norm2)
  cudaErrChk(
      cudaMemsetAsync(d_out, 0, (kp1 + 1) * sizeof(cudaSolverType), stream));

  if (n == 0)
    return;

  launchBatchedDotNorm<0>(kp1, reduceGridSize(n), stream, w, V_base, stride, n,
                          d_out);
  cudaErrChk(cudaGetLastError());
}

// =========================================================================
//  Phys2Solver / Solver2Phys CUDA kernels
// =========================================================================

/**
 * Pack 3 node-based fields into interleaved Krylov vector.
 * Thread per interior point: tid ∈ [0, (nx-2)*(ny-2)*(nz-2)).
 */
template <typename T>
__global__ void
k_phys2solver3(T* __restrict__ d_solver, const T* __restrict__ d_physX,
               const T* __restrict__ d_physY, const T* __restrict__ d_physZ,
               int nx, int ny, int nz) {
  int numInterior = (nx - 2) * (ny - 2) * (nz - 2);
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= numInterior)
    return;

  int nz2 = nz - 2;
  int ny2 = ny - 2;
  int kk = tid % nz2;
  int jj = (tid / nz2) % ny2;
  int ii = tid / (nz2 * ny2);

  int i = ii + 1, j = jj + 1, k = kk + 1;
  int phys_idx = i * ny * nz + j * nz + k;
  int sol_idx = tid * 3;

  d_solver[sol_idx] = d_physX[phys_idx];
  d_solver[sol_idx + 1] = d_physY[phys_idx];
  d_solver[sol_idx + 2] = d_physZ[phys_idx];
}

/** Pack single scalar field into Krylov vector (Poisson). */
template <typename T>
__global__ void k_phys2solver1(T* __restrict__ d_solver,
                               const T* __restrict__ d_phys, int nx, int ny,
                               int nz) {
  int numInterior = (nx - 2) * (ny - 2) * (nz - 2);
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= numInterior)
    return;

  int nz2 = nz - 2;
  int ny2 = ny - 2;
  int kk = tid % nz2;
  int jj = (tid / nz2) % ny2;
  int ii = tid / (nz2 * ny2);

  int i = ii + 1, j = jj + 1, k = kk + 1;
  int phys_idx = i * ny * nz + j * nz + k;

  d_solver[tid] = d_phys[phys_idx];
}

/** Unpack interleaved Krylov vector into 3 node-based fields. */
template <typename T>
__global__ void k_solver2phys3(T* __restrict__ d_physX, T* __restrict__ d_physY,
                               T* __restrict__ d_physZ,
                               const T* __restrict__ d_solver, int nx, int ny,
                               int nz) {
  int numInterior = (nx - 2) * (ny - 2) * (nz - 2);
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= numInterior)
    return;

  int nz2 = nz - 2;
  int ny2 = ny - 2;
  int kk = tid % nz2;
  int jj = (tid / nz2) % ny2;
  int ii = tid / (nz2 * ny2);

  int i = ii + 1, j = jj + 1, k = kk + 1;
  int phys_idx = i * ny * nz + j * nz + k;
  int sol_idx = tid * 3;

  d_physX[phys_idx] = d_solver[sol_idx];
  d_physY[phys_idx] = d_solver[sol_idx + 1];
  d_physZ[phys_idx] = d_solver[sol_idx + 2];
}

/** Unpack Krylov vector into single scalar field (Poisson). */
template <typename T>
__global__ void k_solver2phys1(T* __restrict__ d_phys,
                               const T* __restrict__ d_solver, int nx, int ny,
                               int nz) {
  int numInterior = (nx - 2) * (ny - 2) * (nz - 2);
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= numInterior)
    return;

  int nz2 = nz - 2;
  int ny2 = ny - 2;
  int kk = tid % nz2;
  int jj = (tid / nz2) % ny2;
  int ii = tid / (nz2 * ny2);

  int i = ii + 1, j = jj + 1, k = kk + 1;
  int phys_idx = i * ny * nz + j * nz + k;

  d_phys[phys_idx] = d_solver[tid];
}

// =========================================================================
//  Phys2Solver / Solver2Phys host wrappers
// =========================================================================

void gpuPhys2Solver3(cudaSolverType* d_solver, const cudaSolverType* d_physX,
                     const cudaSolverType* d_physY,
                     const cudaSolverType* d_physZ, int nx, int ny, int nz,
                     cudaStream_t stream) {
  int numInterior = (nx - 2) * (ny - 2) * (nz - 2);
  if (numInterior <= 0)
    return;
  int grid = (int)divCeil((size_t)numInterior, BLAS_BLOCK);
  k_phys2solver3<<<grid, BLAS_BLOCK, 0, stream>>>(d_solver, d_physX, d_physY,
                                                  d_physZ, nx, ny, nz);
  cudaErrChk(cudaGetLastError());
}

void gpuPhys2Solver1(cudaSolverType* d_solver, const cudaSolverType* d_phys,
                     int nx, int ny, int nz, cudaStream_t stream) {
  int numInterior = (nx - 2) * (ny - 2) * (nz - 2);
  if (numInterior <= 0)
    return;
  int grid = (int)divCeil((size_t)numInterior, BLAS_BLOCK);
  k_phys2solver1<<<grid, BLAS_BLOCK, 0, stream>>>(d_solver, d_phys, nx, ny, nz);
  cudaErrChk(cudaGetLastError());
}

void gpuSolver2Phys3(cudaSolverType* d_physX, cudaSolverType* d_physY,
                     cudaSolverType* d_physZ, const cudaSolverType* d_solver,
                     int nx, int ny, int nz, cudaStream_t stream) {
  int numInterior = (nx - 2) * (ny - 2) * (nz - 2);
  if (numInterior <= 0)
    return;
  int grid = (int)divCeil((size_t)numInterior, BLAS_BLOCK);
  k_solver2phys3<<<grid, BLAS_BLOCK, 0, stream>>>(d_physX, d_physY, d_physZ,
                                                  d_solver, nx, ny, nz);
  cudaErrChk(cudaGetLastError());
}

void gpuSolver2Phys1(cudaSolverType* d_phys, const cudaSolverType* d_solver,
                     int nx, int ny, int nz, cudaStream_t stream) {
  int numInterior = (nx - 2) * (ny - 2) * (nz - 2);
  if (numInterior <= 0)
    return;
  int grid = (int)divCeil((size_t)numInterior, BLAS_BLOCK);
  k_solver2phys1<<<grid, BLAS_BLOCK, 0, stream>>>(d_phys, d_solver, nx, ny, nz);
  cudaErrChk(cudaGetLastError());
}

// =========================================================================
//  Scratch buffer helpers
// =========================================================================

void gpuBlasAllocScratch(cudaSolverType** d_scratch) {
  cudaErrChk(cudaMalloc(d_scratch, sizeof(cudaSolverType)));
}

void gpuBlasFreeScratch(cudaSolverType* d_scratch) {
  if (d_scratch)
    cudaFree(d_scratch);
}
