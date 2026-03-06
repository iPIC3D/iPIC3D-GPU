/**
 * @file GPUBlas.cu
 * @brief CUDA kernel implementations for GPU BLAS-like operations.
 *        See GPUBlas.cuh for API documentation.
 */

#include "GPUBlas.cuh"
#include "GPUFieldArray.cuh"  // for fillKernelLaunch

#ifdef GPU_SOLVER

// =========================================================================
//  Kernel configuration
// =========================================================================
static constexpr int BLAS_BLOCK = 256;
static  constexpr int MAX_WARPS = BLAS_BLOCK / WARP_SIZE;
static inline size_t divCeil(size_t n, size_t d) { return (n + d - 1) / d; }

// =========================================================================
//  Element-wise CUDA kernels
// =========================================================================

__global__ void k_scale(double* __restrict__ d, double alfa, size_t n)
{
    size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (i < n) d[i] *= alfa;
}

__global__ void k_scaleCopy(double* __restrict__ dst,
                            const double* __restrict__ src,
                            double alfa, size_t n)
{
    size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (i < n) dst[i] = src[i] * alfa;
}

__global__ void k_sum(double* __restrict__ dst,
                      const double* __restrict__ src, size_t n)
{
    size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (i < n) dst[i] += src[i];
}

__global__ void k_sub(double* __restrict__ dst,
                      const double* __restrict__ src, size_t n)
{
    size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (i < n) dst[i] -= src[i];
}

__global__ void k_subRes(double* __restrict__ res,
                         const double* __restrict__ a,
                         const double* __restrict__ b, size_t n)
{
    size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (i < n) res[i] = a[i] - b[i];
}

__global__ void k_addscale(double alfa,
                           double* __restrict__ dst,
                           const double* __restrict__ src, size_t n)
{
    size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (i < n) dst[i] += alfa * src[i];
}

__global__ void k_addscale2(double alfa, double beta,
                            double* __restrict__ dst,
                            const double* __restrict__ src, size_t n)
{
    size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (i < n) dst[i] = beta * dst[i] + alfa * src[i];
}

__global__ void k_neg(double* __restrict__ d, size_t n)
{
    size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (i < n) d[i] = -d[i];
}

// =========================================================================
//  Element-wise host wrappers
// =========================================================================

void gpuEqValue(double* d, double val, size_t n, cudaStream_t stream)
{
    if (n == 0) return;
    if (val == 0.0) {
        cudaErrChk(cudaMemsetAsync(d, 0, n * sizeof(double), stream));
    } else {
        GPUFieldArray3::fillKernelLaunch(d, val, n, stream);
    }
}

void gpuScale(double* d, double alfa, size_t n, cudaStream_t stream)
{
    if (n == 0) return;
    k_scale<<<divCeil(n, BLAS_BLOCK), BLAS_BLOCK, 0, stream>>>(d, alfa, n);
    cudaErrChk(cudaGetLastError());
}

void gpuScaleCopy(double* dst, const double* src,
                  double alfa, size_t n, cudaStream_t stream)
{
    if (n == 0) return;
    k_scaleCopy<<<divCeil(n, BLAS_BLOCK), BLAS_BLOCK, 0, stream>>>(dst, src, alfa, n);
    cudaErrChk(cudaGetLastError());
}

void gpuSum(double* dst, const double* src, size_t n, cudaStream_t stream)
{
    if (n == 0) return;
    k_sum<<<divCeil(n, BLAS_BLOCK), BLAS_BLOCK, 0, stream>>>(dst, src, n);
    cudaErrChk(cudaGetLastError());
}

void gpuSub(double* dst, const double* src, size_t n, cudaStream_t stream)
{
    if (n == 0) return;
    k_sub<<<divCeil(n, BLAS_BLOCK), BLAS_BLOCK, 0, stream>>>(dst, src, n);
    cudaErrChk(cudaGetLastError());
}

void gpuSubRes(double* res, const double* a, const double* b,
               size_t n, cudaStream_t stream)
{
    if (n == 0) return;
    k_subRes<<<divCeil(n, BLAS_BLOCK), BLAS_BLOCK, 0, stream>>>(res, a, b, n);
    cudaErrChk(cudaGetLastError());
}

void gpuAddscale(double alfa, double* dst, const double* src,
                 size_t n, cudaStream_t stream)
{
    if (n == 0) return;
    k_addscale<<<divCeil(n, BLAS_BLOCK), BLAS_BLOCK, 0, stream>>>(alfa, dst, src, n);
    cudaErrChk(cudaGetLastError());
}

void gpuAddscale2(double alfa, double beta, double* dst, const double* src,
                  size_t n, cudaStream_t stream)
{
    if (n == 0) return;
    k_addscale2<<<divCeil(n, BLAS_BLOCK), BLAS_BLOCK, 0, stream>>>(alfa, beta, dst, src, n);
    cudaErrChk(cudaGetLastError());
}

void gpuNeg(double* d, size_t n, cudaStream_t stream)
{
    if (n == 0) return;
    k_neg<<<divCeil(n, BLAS_BLOCK), BLAS_BLOCK, 0, stream>>>(d, n);
    cudaErrChk(cudaGetLastError());
}

void gpuEq(double* dst, const double* src, size_t n, cudaStream_t stream)
{
    if (n == 0) return;
    cudaErrChk(cudaMemcpyAsync(dst, src, n * sizeof(double),
                                cudaMemcpyDeviceToDevice, stream));
}

// =========================================================================
//  Fused triple kernels – 3 arrays in one launch
// =========================================================================

__global__ void k_neg3(double* __restrict__ dX,
                       double* __restrict__ dY,
                       double* __restrict__ dZ, size_t n)
{
    size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (i < n) { dX[i] = -dX[i]; dY[i] = -dY[i]; dZ[i] = -dZ[i]; }
}

__global__ void k_sub3(double* __restrict__ dstX, const double* __restrict__ srcX,
                       double* __restrict__ dstY, const double* __restrict__ srcY,
                       double* __restrict__ dstZ, const double* __restrict__ srcZ,
                       size_t n)
{
    size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (i < n) { dstX[i] -= srcX[i]; dstY[i] -= srcY[i]; dstZ[i] -= srcZ[i]; }
}

__global__ void k_scale3(double* __restrict__ dX,
                         double* __restrict__ dY,
                         double* __restrict__ dZ,
                         double alfa, size_t n)
{
    size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (i < n) { dX[i] *= alfa; dY[i] *= alfa; dZ[i] *= alfa; }
}

__global__ void k_sumAddTwo3(
    double* __restrict__ dstX, const double* __restrict__ srcAX, const double* __restrict__ srcBX,
    double* __restrict__ dstY, const double* __restrict__ srcAY, const double* __restrict__ srcBY,
    double* __restrict__ dstZ, const double* __restrict__ srcAZ, const double* __restrict__ srcBZ,
    size_t n)
{
    size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (i < n) {
        dstX[i] += srcAX[i] + srcBX[i];
        dstY[i] += srcAY[i] + srcBY[i];
        dstZ[i] += srcAZ[i] + srcBZ[i];
    }
}

__global__ void k_scaleCopy3(
    double* __restrict__ dstX, const double* __restrict__ srcX,
    double* __restrict__ dstY, const double* __restrict__ srcY,
    double* __restrict__ dstZ, const double* __restrict__ srcZ,
    double alfa, size_t n)
{
    size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (i < n) {
        dstX[i] = srcX[i] * alfa;
        dstY[i] = srcY[i] * alfa;
        dstZ[i] = srcZ[i] * alfa;
    }
}

__global__ void k_sum3(
    double* __restrict__ dstX, const double* __restrict__ srcX,
    double* __restrict__ dstY, const double* __restrict__ srcY,
    double* __restrict__ dstZ, const double* __restrict__ srcZ,
    size_t n)
{
    size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (i < n) { dstX[i] += srcX[i]; dstY[i] += srcY[i]; dstZ[i] += srcZ[i]; }
}

__global__ void k_addscale3(
    double alfa,
    double* __restrict__ dstX, const double* __restrict__ srcX,
    double* __restrict__ dstY, const double* __restrict__ srcY,
    double* __restrict__ dstZ, const double* __restrict__ srcZ,
    size_t n)
{
    size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (i < n) {
        dstX[i] += alfa * srcX[i];
        dstY[i] += alfa * srcY[i];
        dstZ[i] += alfa * srcZ[i];
    }
}

__global__ void k_addscale2_3(
    double alfa, double beta,
    double* __restrict__ dstX, const double* __restrict__ srcX,
    double* __restrict__ dstY, const double* __restrict__ srcY,
    double* __restrict__ dstZ, const double* __restrict__ srcZ,
    size_t n)
{
    size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (i < n) {
        dstX[i] = alfa * srcX[i] + beta * dstX[i];
        dstY[i] = alfa * srcY[i] + beta * dstY[i];
        dstZ[i] = alfa * srcZ[i] + beta * dstZ[i];
    }
}

// ---- Fused triple host wrappers ----

void gpuNeg3(double* dX, double* dY, double* dZ, size_t n, cudaStream_t stream)
{
    if (n == 0) return;
    k_neg3<<<divCeil(n, BLAS_BLOCK), BLAS_BLOCK, 0, stream>>>(dX, dY, dZ, n);
    cudaErrChk(cudaGetLastError());
}

void gpuSub3(double* dstX, const double* srcX,
             double* dstY, const double* srcY,
             double* dstZ, const double* srcZ,
             size_t n, cudaStream_t stream)
{
    if (n == 0) return;
    k_sub3<<<divCeil(n, BLAS_BLOCK), BLAS_BLOCK, 0, stream>>>(dstX, srcX, dstY, srcY, dstZ, srcZ, n);
    cudaErrChk(cudaGetLastError());
}

void gpuScale3(double* dX, double* dY, double* dZ, double alfa, size_t n, cudaStream_t stream)
{
    if (n == 0) return;
    k_scale3<<<divCeil(n, BLAS_BLOCK), BLAS_BLOCK, 0, stream>>>(dX, dY, dZ, alfa, n);
    cudaErrChk(cudaGetLastError());
}

void gpuSumAddTwo3(double* dstX, const double* srcAX, const double* srcBX,
                   double* dstY, const double* srcAY, const double* srcBY,
                   double* dstZ, const double* srcAZ, const double* srcBZ,
                   size_t n, cudaStream_t stream)
{
    if (n == 0) return;
    k_sumAddTwo3<<<divCeil(n, BLAS_BLOCK), BLAS_BLOCK, 0, stream>>>(
        dstX, srcAX, srcBX, dstY, srcAY, srcBY, dstZ, srcAZ, srcBZ, n);
    cudaErrChk(cudaGetLastError());
}

void gpuSetAll0_N(double** d_ptrs, int nFields, size_t n, cudaStream_t stream)
{
    for (int f = 0; f < nFields; ++f)
        cudaErrChk(cudaMemsetAsync(d_ptrs[f], 0, n * sizeof(double), stream));
}

void gpuScaleCopy3(double* dstX, const double* srcX,
                   double* dstY, const double* srcY,
                   double* dstZ, const double* srcZ,
                   double alfa, size_t n, cudaStream_t stream)
{
    if (n == 0) return;
    k_scaleCopy3<<<divCeil(n, BLAS_BLOCK), BLAS_BLOCK, 0, stream>>>(
        dstX, srcX, dstY, srcY, dstZ, srcZ, alfa, n);
    cudaErrChk(cudaGetLastError());
}

void gpuSum3(double* dstX, const double* srcX,
             double* dstY, const double* srcY,
             double* dstZ, const double* srcZ,
             size_t n, cudaStream_t stream)
{
    if (n == 0) return;
    k_sum3<<<divCeil(n, BLAS_BLOCK), BLAS_BLOCK, 0, stream>>>(
        dstX, srcX, dstY, srcY, dstZ, srcZ, n);
    cudaErrChk(cudaGetLastError());
}

void gpuAddscale3(double alfa,
                  double* dstX, const double* srcX,
                  double* dstY, const double* srcY,
                  double* dstZ, const double* srcZ,
                  size_t n, cudaStream_t stream)
{
    if (n == 0) return;
    k_addscale3<<<divCeil(n, BLAS_BLOCK), BLAS_BLOCK, 0, stream>>>(
        alfa, dstX, srcX, dstY, srcY, dstZ, srcZ, n);
    cudaErrChk(cudaGetLastError());
}

void gpuAddscale2_3(double alfa, double beta,
                    double* dstX, const double* srcX,
                    double* dstY, const double* srcY,
                    double* dstZ, const double* srcZ,
                    size_t n, cudaStream_t stream)
{
    if (n == 0) return;
    k_addscale2_3<<<divCeil(n, BLAS_BLOCK), BLAS_BLOCK, 0, stream>>>(
        alfa, beta, dstX, srcX, dstY, srcY, dstZ, srcZ, n);
    cudaErrChk(cudaGetLastError());
}

// =========================================================================
//  Reduction CUDA kernel – warp-shuffle + atomicAdd
// =========================================================================

/**
 * Generic dot-product reduction kernel.
 * Each block reduces a chunk using grid-stride loop + warp shuffle.
 * The final per-block scalar is atomically added to *d_result.
 * d_result MUST be zeroed before launch.
 */
__global__ void k_dotReduce(const double* __restrict__ a,
                            const double* __restrict__ b,
                            size_t n,
                            double* __restrict__ d_result)
{
    double sum = 0.0;
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
         i < n;
         i += (size_t)blockDim.x * gridDim.x)
    {
        sum += a[i] * b[i];
    }

    // Warp-level reduction
    for (int offset = warpSize / 2; offset > 0; offset >>= 1)
        sum += __shfl_down_sync(0xFFFFFFFF, sum, offset);

    // First lane of each warp writes to shared memory
    __shared__ double warpSums[MAX_WARPS];
    int lane   = threadIdx.x % WARP_SIZE;
    int warpId = threadIdx.x / WARP_SIZE;
    if (lane == 0) warpSums[warpId] = sum;
    __syncthreads();

    // First warp reduces warp sums
    if (warpId == 0) {
        sum = (lane < MAX_WARPS) ? warpSums[lane] : 0.0;
        for (int offset = warpSize / 2; offset > 0; offset >>= 1)
            sum += __shfl_down_sync(0xFFFFFFFF, sum, offset);
        if (lane == 0) atomicAdd(d_result, sum);
    }
}

/** Same structure but computes norm2 (a[i]*a[i]). */
__global__ void k_norm2Reduce(const double* __restrict__ a,
                              size_t n,
                              double* __restrict__ d_result)
{
    double sum = 0.0;
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
         i < n;
         i += (size_t)blockDim.x * gridDim.x)
    {
        double v = a[i];
        sum += v * v;
    }

    for (int offset = warpSize / 2; offset > 0; offset >>= 1)
        sum += __shfl_down_sync(0xFFFFFFFF, sum, offset);

    __shared__ double warpSums[MAX_WARPS];
    int lane   = threadIdx.x % WARP_SIZE;
    int warpId = threadIdx.x / WARP_SIZE;
    if (lane == 0) warpSums[warpId] = sum;
    __syncthreads();

    if (warpId == 0) {
        sum = (lane < MAX_WARPS) ? warpSums[lane] : 0.0;
        for (int offset = warpSize / 2; offset > 0; offset >>= 1)
            sum += __shfl_down_sync(0xFFFFFFFF, sum, offset);
        if (lane == 0) atomicAdd(d_result, sum);
    }
}

// ---- grid size limiter for reductions (avoid excessive blocks) ----
static inline int reduceGridSize(size_t n)
{
    int blocks = (int)divCeil(n, BLAS_BLOCK);
    // Cap at 1024 blocks – grid-stride loop handles the rest
    return (blocks > 1024) ? 1024 : blocks;
}

// =========================================================================
//  Reduction host wrappers
// =========================================================================

double gpuDot(const double* a, const double* b, size_t n,
              double* d_scratch, cudaStream_t stream)
{
    cudaErrChk(cudaMemsetAsync(d_scratch, 0, sizeof(double), stream));
    if (n > 0) {
        k_dotReduce<<<reduceGridSize(n), BLAS_BLOCK, 0, stream>>>(a, b, n, d_scratch);
        cudaErrChk(cudaGetLastError());
    }
    double result;
    cudaErrChk(cudaMemcpyAsync(&result, d_scratch, sizeof(double),
                                cudaMemcpyDeviceToHost, stream));
    cudaErrChk(cudaStreamSynchronize(stream));
    return result;
}

double gpuNorm2(const double* a, size_t n,
                double* d_scratch, cudaStream_t stream)
{
    cudaErrChk(cudaMemsetAsync(d_scratch, 0, sizeof(double), stream));
    if (n > 0) {
        k_norm2Reduce<<<reduceGridSize(n), BLAS_BLOCK, 0, stream>>>(a, n, d_scratch);
        cudaErrChk(cudaGetLastError());
    }
    double result;
    cudaErrChk(cudaMemcpyAsync(&result, d_scratch, sizeof(double),
                                cudaMemcpyDeviceToHost, stream));
    cudaErrChk(cudaStreamSynchronize(stream));
    return result;
}

// ---- Async variants (no stream sync — caller must sync before reading) ----

void gpuNorm2_async(const double* a, size_t n,
                    double* d_scratch, double* h_result,
                    cudaStream_t stream)
{
    cudaErrChk(cudaMemsetAsync(d_scratch, 0, sizeof(double), stream));
    if (n > 0) {
        k_norm2Reduce<<<reduceGridSize(n), BLAS_BLOCK, 0, stream>>>(a, n, d_scratch);
        cudaErrChk(cudaGetLastError());
    }
    cudaErrChk(cudaMemcpyAsync(h_result, d_scratch, sizeof(double),
                                cudaMemcpyDeviceToHost, stream));
}

void gpuDot_async(const double* a, const double* b, size_t n,
                  double* d_scratch, double* h_result,
                  cudaStream_t stream)
{
    cudaErrChk(cudaMemsetAsync(d_scratch, 0, sizeof(double), stream));
    if (n > 0) {
        k_dotReduce<<<reduceGridSize(n), BLAS_BLOCK, 0, stream>>>(a, b, n, d_scratch);
        cudaErrChk(cudaGetLastError());
    }
    cudaErrChk(cudaMemcpyAsync(h_result, d_scratch, sizeof(double),
                                cudaMemcpyDeviceToHost, stream));
}

// =========================================================================
//  Batched dot+norm kernel for GMRES Arnoldi
// =========================================================================

/**
 * Fused kernel: compute (k+1) dot products dot(w, V[j]) for j=0..k
 * AND norm2(w) simultaneously.  Uses grid-stride loop.
 *
 * d_out must have k+2 doubles, zeroed before launch.
 * V[j] starts at V_base + j*stride.
 *
 * Template-free: runtime loop over j values.  k is small (~20) so the
 * overhead of the inner loop is negligible.
 */
__global__ void k_batchedDotNorm(const double* __restrict__ w,
                                 const double* __restrict__ V_base,
                                 size_t stride, int kp1, size_t n,
                                 double* __restrict__ d_out)
{
    // kp1 = k+1 (number of dot products). d_out has kp1+1 entries.
    // d_out[0..kp1-1] = dot(w, V[j]), d_out[kp1] = norm2(w).

    // Each thread accumulates kp1+1 partial sums.
    // Use stack-allocated array up to a compile-time max.
    constexpr int MAX_BATCH = 32;
    double sums[MAX_BATCH + 1]; // +1 for norm2
    int total = kp1 + 1;
    if (total > MAX_BATCH + 1) total = MAX_BATCH + 1; // safety clamp
    for (int j = 0; j < total; j++) sums[j] = 0.0;

    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
         i < n;
         i += (size_t)blockDim.x * gridDim.x)
    {
        double wi = w[i];
        for (int j = 0; j < kp1; j++)
            sums[j] += wi * V_base[j * stride + i];
        sums[kp1] += wi * wi;
    }

    // Warp-level reduction for ALL accumulators
    for (int j = 0; j < total; j++) {
        for (int offset = warpSize / 2; offset > 0; offset >>= 1)
            sums[j] += __shfl_down_sync(0xFFFFFFFF, sums[j], offset);
    }

    // Per-warp partial sums to shared memory
    extern __shared__ double smem[]; // MAX_WARPS * total
    int lane   = threadIdx.x % WARP_SIZE;
    int warpId = threadIdx.x / WARP_SIZE;

    if (lane == 0) {
        for (int j = 0; j < total; j++)
            smem[warpId * total + j] = sums[j];
    }
    __syncthreads();

    // First warp reduces across warps
    if (warpId == 0) {
        for (int j = 0; j < total; j++) {
            double val = (lane < MAX_WARPS) ? smem[lane * total + j] : 0.0;
            for (int offset = warpSize / 2; offset > 0; offset >>= 1)
                val += __shfl_down_sync(0xFFFFFFFF, val, offset);
            if (lane == 0) atomicAdd(&d_out[j], val);
        }
    }
}

void gpuBatchedDotNorm(const double* w, const double* V_base,
                       size_t stride, int k, size_t n,
                       double* d_out, cudaStream_t stream)
{
    int kp1 = k + 1; // number of dot products
    int total = kp1 + 1; // +1 for norm2

    // Zero the output array
    cudaErrChk(cudaMemsetAsync(d_out, 0, total * sizeof(double), stream));

    if (n == 0) return;

    int grid = reduceGridSize(n);
    // Shared memory: MAX_WARPS * total doubles
    size_t smemBytes = MAX_WARPS * total * sizeof(double);

    k_batchedDotNorm<<<grid, BLAS_BLOCK, smemBytes, stream>>>(
        w, V_base, stride, kp1, n, d_out);
    cudaErrChk(cudaGetLastError());
}

// =========================================================================
//  Phys2Solver / Solver2Phys CUDA kernels
// =========================================================================

/**
 * Pack 3 node-based fields into interleaved Krylov vector.
 * Thread per interior point: tid ∈ [0, (nx-2)*(ny-2)*(nz-2)).
 */
__global__ void k_phys2solver3(double* __restrict__ d_solver,
                               const double* __restrict__ d_physX,
                               const double* __restrict__ d_physY,
                               const double* __restrict__ d_physZ,
                               int nx, int ny, int nz)
{
    int numInterior = (nx - 2) * (ny - 2) * (nz - 2);
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= numInterior) return;

    int nz2 = nz - 2;
    int ny2 = ny - 2;
    int kk = tid % nz2;
    int jj = (tid / nz2) % ny2;
    int ii = tid / (nz2 * ny2);

    int i = ii + 1, j = jj + 1, k = kk + 1;
    int phys_idx = i * ny * nz + j * nz + k;
    int sol_idx  = tid * 3;

    d_solver[sol_idx]     = d_physX[phys_idx];
    d_solver[sol_idx + 1] = d_physY[phys_idx];
    d_solver[sol_idx + 2] = d_physZ[phys_idx];
}

/** Pack single scalar field into Krylov vector (Poisson). */
__global__ void k_phys2solver1(double* __restrict__ d_solver,
                               const double* __restrict__ d_phys,
                               int nx, int ny, int nz)
{
    int numInterior = (nx - 2) * (ny - 2) * (nz - 2);
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= numInterior) return;

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
__global__ void k_solver2phys3(double* __restrict__ d_physX,
                               double* __restrict__ d_physY,
                               double* __restrict__ d_physZ,
                               const double* __restrict__ d_solver,
                               int nx, int ny, int nz)
{
    int numInterior = (nx - 2) * (ny - 2) * (nz - 2);
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= numInterior) return;

    int nz2 = nz - 2;
    int ny2 = ny - 2;
    int kk = tid % nz2;
    int jj = (tid / nz2) % ny2;
    int ii = tid / (nz2 * ny2);

    int i = ii + 1, j = jj + 1, k = kk + 1;
    int phys_idx = i * ny * nz + j * nz + k;
    int sol_idx  = tid * 3;

    d_physX[phys_idx] = d_solver[sol_idx];
    d_physY[phys_idx] = d_solver[sol_idx + 1];
    d_physZ[phys_idx] = d_solver[sol_idx + 2];
}

/** Unpack Krylov vector into single scalar field (Poisson). */
__global__ void k_solver2phys1(double* __restrict__ d_phys,
                               const double* __restrict__ d_solver,
                               int nx, int ny, int nz)
{
    int numInterior = (nx - 2) * (ny - 2) * (nz - 2);
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= numInterior) return;

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

void gpuPhys2Solver3(double* d_solver,
                     const double* d_physX,
                     const double* d_physY,
                     const double* d_physZ,
                     int nx, int ny, int nz,
                     cudaStream_t stream)
{
    int numInterior = (nx - 2) * (ny - 2) * (nz - 2);
    if (numInterior <= 0) return;
    int grid = (int)divCeil((size_t)numInterior, BLAS_BLOCK);
    k_phys2solver3<<<grid, BLAS_BLOCK, 0, stream>>>(
        d_solver, d_physX, d_physY, d_physZ, nx, ny, nz);
    cudaErrChk(cudaGetLastError());
}

void gpuPhys2Solver1(double* d_solver,
                     const double* d_phys,
                     int nx, int ny, int nz,
                     cudaStream_t stream)
{
    int numInterior = (nx - 2) * (ny - 2) * (nz - 2);
    if (numInterior <= 0) return;
    int grid = (int)divCeil((size_t)numInterior, BLAS_BLOCK);
    k_phys2solver1<<<grid, BLAS_BLOCK, 0, stream>>>(d_solver, d_phys, nx, ny, nz);
    cudaErrChk(cudaGetLastError());
}

void gpuSolver2Phys3(double* d_physX,
                     double* d_physY,
                     double* d_physZ,
                     const double* d_solver,
                     int nx, int ny, int nz,
                     cudaStream_t stream)
{
    int numInterior = (nx - 2) * (ny - 2) * (nz - 2);
    if (numInterior <= 0) return;
    int grid = (int)divCeil((size_t)numInterior, BLAS_BLOCK);
    k_solver2phys3<<<grid, BLAS_BLOCK, 0, stream>>>(
        d_physX, d_physY, d_physZ, d_solver, nx, ny, nz);
    cudaErrChk(cudaGetLastError());
}

void gpuSolver2Phys1(double* d_phys,
                     const double* d_solver,
                     int nx, int ny, int nz,
                     cudaStream_t stream)
{
    int numInterior = (nx - 2) * (ny - 2) * (nz - 2);
    if (numInterior <= 0) return;
    int grid = (int)divCeil((size_t)numInterior, BLAS_BLOCK);
    k_solver2phys1<<<grid, BLAS_BLOCK, 0, stream>>>(d_phys, d_solver, nx, ny, nz);
    cudaErrChk(cudaGetLastError());
}

// =========================================================================
//  Scratch buffer helpers
// =========================================================================

void gpuBlasAllocScratch(double** d_scratch)
{
    cudaErrChk(cudaMalloc(d_scratch, sizeof(double)));
}

void gpuBlasFreeScratch(double* d_scratch)
{
    if (d_scratch) cudaFree(d_scratch);
}

#endif // GPU_SOLVER
