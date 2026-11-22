/**
 * @file GPUBlas.cu
 * @brief CUDA kernel implementations for GPU BLAS-like operations.
 *        See GPUBlas.cuh for API documentation.
 */

#include "GPUBlas.cuh"
#include "GPUFieldArray.cuh" // for fillKernelLaunch
#include <algorithm>
#include <climits>
#include <cstdio>
#include <cstdlib>
#include <utility>

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

struct Combination {
  cudaSolverType c[kMaxCombination];
};

template <typename T>
__global__ void k_subCombination(T* __restrict__ w, const T* __restrict__ V,
                                 size_t stride, int count, size_t n,
                                 Combination coef) {
  const size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
  if (i >= n)
    return;
  T acc = w[i];
  for (int j = 0; j < count; j++)
    acc += -coef.c[j] * V[j * stride + i];
  w[i] = acc;
}

/** Not used (gpuAddscale2). */
template <typename T>
__global__ void k_addscale2(T alfa, T beta, T* __restrict__ dst,
                            const T* __restrict__ src, size_t n) {
  size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
  if (i < n)
    dst[i] = beta * dst[i] + alfa * src[i];
}

/** Not used (gpuAddConst). */
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

void gpuSubCombination(cudaSolverType* w, const cudaSolverType* V,
                       size_t stride, int count, size_t n,
                       const cudaSolverType* coef, cudaStream_t stream) {
  if (n == 0 || count <= 0)
    return;
  if (count > kMaxCombination) {
    fprintf(stderr, "gpuSubCombination takes at most %d terms, not %d\n",
            kMaxCombination, count);
    abort();
  }
  Combination c;
  for (int j = 0; j < count; j++)
    c.c[j] = coef[j];
  k_subCombination<<<divCeil(n, BLAS_BLOCK), BLAS_BLOCK, 0, stream>>>(
      w, V, stride, count, n, c);
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

/** Same structure, sum of a[i].  Not used (gpuReduceSum). */
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

// The blocking ones wait for the result in the page-locked slot h_result;
// gpuDot and gpuNorm2 launch and copy as their _async variants below.

cudaSolverType gpuDot(const cudaSolverType* a, const cudaSolverType* b,
                      size_t n, cudaSolverType* d_scratch,
                      cudaSolverType* h_result, cudaStream_t stream) {
  gpuDot_async(a, b, n, d_scratch, h_result, stream);
  cudaErrChk(cudaStreamSynchronize(stream));
  return *h_result;
}

cudaSolverType gpuNorm2(const cudaSolverType* a, size_t n,
                        cudaSolverType* d_scratch, cudaSolverType* h_result,
                        cudaStream_t stream) {
  gpuNorm2_async(a, n, d_scratch, h_result, stream);
  cudaErrChk(cudaStreamSynchronize(stream));
  return *h_result;
}

cudaSolverType gpuReduceSum(const cudaSolverType* a, size_t n,
                            cudaSolverType* d_scratch, cudaSolverType* h_result,
                            cudaStream_t stream) {
  cudaErrChk(cudaMemsetAsync(d_scratch, 0, sizeof(cudaSolverType), stream));
  if (n > 0) {
    k_sumReduce<<<reduceGridSize(n), BLAS_BLOCK, 0, stream>>>(a, n, d_scratch);
    cudaErrChk(cudaGetLastError());
  }
  cudaErrChk(cudaMemcpyAsync(h_result, d_scratch, sizeof(cudaSolverType),
                             cudaMemcpyDeviceToHost, stream));
  cudaErrChk(cudaStreamSynchronize(stream));
  return *h_result;
}

// ---- Signed box sums and updates (the null-space components of the
//      cleaning solves: means, checkerboards) ----

/** Parity of sum_a sgn[a] (idx[a] + par[a]): the sign (-1)^parity. */
__device__ __forceinline__ bool oddParity(const int sgn[3], int i, int j, int k,
                                          const int par[3]) {
  return (sgn[0] * (i + par[0]) + sgn[1] * (j + par[1]) +
          sgn[2] * (k + par[2])) &
         1;
}

/** The point t of the box with its points along axis c (c >= 0) collapsed
 *  to the line l: its indices and packed index. */
__device__ __forceinline__ int signedBoxPoint(const SignedBox& b, int c, int l,
                                              int t, int& i, int& j, int& k) {
  const int dk = c == 2 ? 1 : b.hi[2] - b.lo[2];
  const int dj = c == 1 ? 1 : b.hi[1] - b.lo[1];
  const int r = t / dk;
  i = b.lo[0] + (c == 0 ? l : r / dj);
  j = b.lo[1] + (c == 1 ? l : r - (r / dj) * dj);
  k = b.lo[2] + (c == 2 ? l : t - r * dk);
  return (i * b.n[1] + j) * b.n[2] + k;
}

template <typename T>
__global__ void k_signedBoxSums(const T* __restrict__ v, SignedBox b, int c,
                                int points, T* __restrict__ out) {
  T sum = T(0);
  for (int t = blockIdx.x * blockDim.x + threadIdx.x; t < points;
       t += gridDim.x * blockDim.x) {
    int i, j, k;
    const int p = signedBoxPoint(b, c, blockIdx.y, t, i, j, k);
    sum += oddParity(b.sgn, i, j, k, b.par) ? -v[p] : v[p];
  }
  blockSumAtomicAdd(sum, &out[blockIdx.y]);
}

struct SignedTerms {
  SignedTerm t[kMaxSignedTerms];
  int n;
};

template <typename T>
__global__ void k_subSignedTerms(T* __restrict__ v, SignedBox b,
                                 SignedTerms terms, int points) {
  const int t = blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= points)
    return;
  int i, j, k;
  const int p = signedBoxPoint(b, -1, 0, t, i, j, k);
  T sub = T(0);
  // a compile-time index into the terms: a runtime one can make the compiler
  // copy the kernel argument to per-thread memory
#pragma unroll
  for (int m = 0; m < kMaxSignedTerms; m++) {
    if (m >= terms.n)
      break;
    const SignedTerm& s = terms.t[m];
    const int l = s.c == 0 ? i - b.lo[0] : s.c == 1 ? j - b.lo[1] : k - b.lo[2];
    const T value = s.c >= 0 ? s.line[l] : s.value;
    sub += oddParity(s.sgn, i, j, k, b.par) ? -value : value;
  }
  v[p] -= sub;
}

static int boxPoints(const SignedBox& b, int c) {
  long long points = 1;
  for (int a = 0; a < 3; a++)
    points *= a == c ? 1 : std::max(b.hi[a] - b.lo[a], 0);
  if (points > INT_MAX) {
    fprintf(stderr, "signed box of %lld points exceeds INT_MAX\n", points);
    abort();
  }
  return (int)points;
}

void gpuSignedBoxSums(const cudaSolverType* v, const SignedBox& box, int c,
                      cudaSolverType* d_out, cudaStream_t stream) {
  const int lines = c >= 0 ? box.hi[c] - box.lo[c] : 1;
  if (lines <= 0)
    return;
  if (lines > 65535) { // gridDim.y
    fprintf(stderr, "gpuSignedBoxSums: %d lines exceed the grid limit\n",
            lines);
    abort();
  }
  cudaErrChk(cudaMemsetAsync(d_out, 0, lines * sizeof(cudaSolverType), stream));
  const int points = boxPoints(box, c);
  if (points == 0)
    return;
  // a few blocks per line, as many as reduceGridSize for one total
  const int blocks =
      c >= 0 ? std::min(reduceGridSize(points), 64) : reduceGridSize(points);
  k_signedBoxSums<<<dim3(blocks, lines), BLAS_BLOCK, 0, stream>>>(
      v, box, c, points, d_out);
  cudaErrChk(cudaGetLastError());
}

void gpuSubSignedTerms(cudaSolverType* v, const SignedBox& box,
                       const SignedTerm* terms, int nTerms,
                       cudaStream_t stream) {
  if (nTerms == 0)
    return;
  if (nTerms < 0 || nTerms > kMaxSignedTerms) {
    fprintf(stderr, "gpuSubSignedTerms takes 0 to %d terms, not %d\n",
            kMaxSignedTerms, nTerms);
    abort();
  }
  SignedTerms set{};
  for (int m = 0; m < nTerms; m++)
    set.t[m] = terms[m];
  set.n = nTerms;
  const int points = boxPoints(box, -1);
  if (points == 0)
    return;
  k_subSignedTerms<<<(unsigned)divCeil(points, BLAS_BLOCK), BLAS_BLOCK, 0,
                     stream>>>(v, box, set, points);
  cudaErrChk(cudaGetLastError());
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

// Most dot products per launch of k_batchedDotNorm (its instantiations);
// gpuBatchedDotNorm launches more in chunks of this size
constexpr int kBatchedDotMax = 32;

/** Smallest power of two >= c. */
__host__ __device__ constexpr int pow2Ceil(int c) {
  int p = 1;
  while (p < c)
    p *= 2;
  return p;
}

/** Largest power of two <= c (c >= 1). */
__host__ __device__ constexpr int pow2Floor(int c) {
  int p = 1;
  while (2 * p <= c)
    p *= 2;
  return p;
}

/** Shuffles (and additions) per lane of warpSumTransposed for C values
 *  padded to P = pow2Ceil(C) <= WARP_SIZE: P - 1 + log2(WARP_SIZE / P). */
__host__ __device__ constexpr int warpSumPadCost(int c) {
  int cost = pow2Ceil(c) - 1;
  for (int p = pow2Ceil(c); p < WARP_SIZE; p *= 2)
    cost++;
  return cost;
}

/** Shuffles per lane of warpSumsToShared for C values: those of padding
 *  them (warpSumPadCost) or, above WARP_SIZE values or when cheaper, of two
 *  groups, the largest power of two below C and the rest (e.g. 17 = 16 + 1
 *  values: 16 + 5 instead of 31). */
__host__ __device__ constexpr int warpSumCost(int c) {
  if (c <= 0)
    return 0;
  const int split =
      warpSumPadCost(pow2Floor(c)) + warpSumCost(c - pow2Floor(c));
  return c > WARP_SIZE || split < warpSumPadCost(c) ? split : warpSumPadCost(c);
}

/** Whether warpSumsToShared sums C >= 1 values in two groups (warpSumCost). */
__host__ __device__ constexpr bool warpSumSplit(int c) {
  return c > WARP_SIZE || warpSumCost(c) < warpSumPadCost(c);
}

/** f(std::integral_constant<int, J>()) for J = 0 .. N-1: the indices are
 *  constants in the source (with a loop, even unrolled, the compiler can turn
 *  a select between two array elements into a runtime index and keep the
 *  array in local memory, as nvcc does for sm_80). */
template <typename F, int... J>
__device__ __forceinline__ void
staticForImpl(F& f, std::integer_sequence<int, J...>) {
  (f(std::integral_constant<int, J>()), ...);
}
template <int N, typename F> __device__ __forceinline__ void staticFor(F&& f) {
  staticForImpl(f, std::make_integer_sequence<int, N>());
}

/** The halving steps of warpSumTransposed: H pairs (b[j], b[j + H]) at lane
 *  offset O, then H / 2 pairs at O / 2, ... */
template <int H, int O, typename T, int P>
__device__ __forceinline__ void warpHalvings(T (&b)[P], int lane) {
  if constexpr (H >= 1) {
    const bool upper = lane & O;
    staticFor<H>([&](auto J) {
      constexpr int j = decltype(J)::value;
      const T lo = b[j], hi = b[j + H];
      b[j] = (upper ? hi : lo) +
             __shfl_sync(WARP_FULL_MASK, upper ? lo : hi, lane ^ O);
    });
    warpHalvings<H / 2, O / 2>(b, lane);
  }
}

/**
 * Warp sums of the C values a[OFF + j], j < C <= WARP_SIZE, of every lane:
 * returns the sum of value lane / (WARP_SIZE / P) (P = pow2Ceil(C); a lane
 * past C - 1 returns a dummy).  Each step pairs lane l with lane l ^ o
 * (o = WARP_SIZE/2 .. 1): the first log2(P) steps halve the values a lane
 * holds (it keeps one half and adds the partner's copy of it), the others
 * add one value.  Every value is so summed by the tree of warp_reduce_sum,
 * which pairs l with l + o = l ^ o (l < o), with at most the two operands of
 * an addition swapped: the same sum bit for bit, with warpSumPadCost(C)
 * additions and shuffles per lane instead of C log2(WARP_SIZE).
 */
template <int C, int OFF, typename T, int N>
__device__ __forceinline__ T warpSumTransposed(const T (&a)[N], int lane) {
  constexpr int P = pow2Ceil(C);
  static_assert(C >= 1 && P <= WARP_SIZE && OFF + C <= N, "value range");
  T b[P];
  staticFor<P>([&](auto J) {
    constexpr int j = decltype(J)::value;
    b[j] = j < C ? a[OFF + j] : T(0);
  });
  warpHalvings<P / 2, WARP_SIZE / 2>(b, lane);
#pragma unroll
  for (int o = WARP_SIZE / (2 * P); o > 0; o /= 2)
    b[0] += __shfl_sync(WARP_FULL_MASK, b[0], lane ^ o);
  return b[0];
}

/** The warp sums of a[OFF + j], j < C, into dst[OFF + j]: warpSumTransposed
 *  on groups of values (warpSumSplit). */
template <int C, int OFF, typename T, int N>
__device__ __forceinline__ void warpSumsToShared(const T (&a)[N], int lane,
                                                 T* dst) {
  if constexpr (warpSumSplit(C)) {
    constexpr int H = pow2Floor(C);
    warpSumsToShared<H, OFF>(a, lane, dst);
    warpSumsToShared<C - H, OFF + H>(a, lane, dst);
  } else {
    constexpr int laneStep = WARP_SIZE / pow2Ceil(C);
    const T s = warpSumTransposed<C, OFF>(a, lane);
    const int j = lane / laneStep;
    if (lane % laneStep == 0 && j < C)
      dst[OFF + j] = s;
  }
}

/**
 * Fused kernel: the KP1 dot products dot(w, V[j]), j < KP1, and norm2(w) in
 * one grid-stride pass over w.  KP1 is a template parameter so that the
 * KP1 + 1 per-thread sums stay in registers (indexed with a runtime count
 * they would live in local memory).
 *
 * d_out[j] = dot(w, V[j]) and *d_norm = norm2(w) (unless d_norm is null)
 * are added to, so they are zeroed before launch.  V[j] starts at V_base +
 * j*stride.
 *
 * The block sums are those of warp_reduce_sum per value, bit for bit: the
 * warp sums of all values in transposed passes (warpSumsToShared), then the
 * sums over the warps with the values spread over the warps.
 */
template <typename T, int KP1>
__global__ void k_batchedDotNorm(const T* __restrict__ w,
                                 const T* __restrict__ V_base, size_t stride,
                                 size_t n, T* __restrict__ d_out,
                                 T* __restrict__ d_norm) {
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

  // Per-warp sums to shared memory
  __shared__ T smem[MAX_WARPS * total];
  const int lane = threadIdx.x % WARP_SIZE;
  const int warpId = threadIdx.x / WARP_SIZE;
  warpSumsToShared<total, 0>(sums, lane, &smem[warpId * total]);
  __syncthreads();

  // Sums over the warps, value j by warp j % MAX_WARPS
  for (int j = warpId; j < total; j += MAX_WARPS) {
    T val = (lane < MAX_WARPS) ? smem[lane * total + j] : 0.0;
    val = warp_reduce_sum(val);
    T* out = j < KP1 ? &d_out[j] : d_norm;
    if (lane == 0 && out)
      atomicAdd(out, val);
  }
}

/** Launch k_batchedDotNorm<KP1> for the runtime count kp1, KP1 <= kp1 <=
 *  kBatchedDotMax (host-side dispatch; kp1 = 0 computes norm2(w) only). */
template <int KP1>
static void launchBatchedDotNorm(int kp1, int grid, cudaStream_t stream,
                                 const cudaSolverType* w,
                                 const cudaSolverType* V_base, size_t stride,
                                 size_t n, cudaSolverType* d_out,
                                 cudaSolverType* d_norm) {
  if constexpr (KP1 < kBatchedDotMax) {
    if (kp1 > KP1)
      return launchBatchedDotNorm<KP1 + 1>(kp1, grid, stream, w, V_base, stride,
                                           n, d_out, d_norm);
  }
  k_batchedDotNorm<cudaSolverType, KP1>
      <<<grid, BLAS_BLOCK, 0, stream>>>(w, V_base, stride, n, d_out, d_norm);
}

void gpuBatchedDotNorm(const cudaSolverType* w, const cudaSolverType* V_base,
                       size_t stride, int k, size_t n, cudaSolverType* d_out,
                       cudaStream_t stream) {
  const int kp1 = k + 1; // number of dot products

  // Zero the output array (kp1 dot products + norm2)
  cudaErrChk(
      cudaMemsetAsync(d_out, 0, (kp1 + 1) * sizeof(cudaSolverType), stream));

  if (n == 0)
    return;

  // chunks of at most kBatchedDotMax vectors, norm2(w) with the last one
  for (int j0 = 0; j0 == 0 || j0 < kp1; j0 += kBatchedDotMax) {
    const int count = std::min(kBatchedDotMax, kp1 - j0);
    const bool last = j0 + count == kp1;
    launchBatchedDotNorm<0>(count, reduceGridSize(n), stream, w,
                            V_base + j0 * stride, stride, n, d_out + j0,
                            last ? d_out + kp1 : nullptr);
  }
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

/** Pack single scalar field into Krylov vector (unit tests and the unused
 *  7-point images). */
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

/** Unpack Krylov vector into single scalar field (unit tests and the
 *  unused 7-point images). */
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
