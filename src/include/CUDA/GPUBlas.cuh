/**
 * @file GPUBlas.cuh
 * @brief GPU BLAS-like operations for the iPIC3D GPU field solver.
 *
 * Element-wise vector operations (fill, scale, sum, sub, addscale, neg)
 * and reduction operations (dot product, norm) on flat device arrays.
 * Also provides GPU kernels for phys2solver / solver2phys conversion.
 *
 * All operations work on raw device pointers (cudaSolverType*) and can be
 * used with GPUFieldArray3::devPtr(), GPUFieldArray4::speciesPtr(), and
 * GPUKrylovVector::devPtr().
 *
 * Compile with -DGPU_SOLVER to activate.
 */

#ifndef GPU_BLAS_CUH
#define GPU_BLAS_CUH

#include "cudaTypeDef.cuh"
#include <cstddef>

#ifdef GPU_SOLVER

// =========================================================================
//  Element-wise operations (flat device pointers, size n)
// =========================================================================

/** d[i] = val  for i=0..n-1.  Uses cudaMemset for val==0. */
void gpuEqValue(cudaSolverType* d, cudaSolverType val, size_t n,
                cudaStream_t stream = 0);

/** d[i] *= alfa */
void gpuScale(cudaSolverType* d, cudaSolverType alfa, size_t n,
              cudaStream_t stream = 0);

/** dst[i] = src[i] * alfa */
void gpuScaleCopy(cudaSolverType* dst, const cudaSolverType* src,
                  cudaSolverType alfa, size_t n, cudaStream_t stream = 0);

/** dst[i] += src[i] */
void gpuSum(cudaSolverType* dst, const cudaSolverType* src, size_t n,
            cudaStream_t stream = 0);

/** dst[i] -= src[i] */
void gpuSub(cudaSolverType* dst, const cudaSolverType* src, size_t n,
            cudaStream_t stream = 0);

/** res[i] = a[i] - b[i] */
void gpuSubRes(cudaSolverType* res, const cudaSolverType* a,
               const cudaSolverType* b, size_t n, cudaStream_t stream = 0);

/** dst[i] += alfa * src[i] */
void gpuAddscale(cudaSolverType alfa, cudaSolverType* dst,
                 const cudaSolverType* src, size_t n, cudaStream_t stream = 0);

/** dst[i] = beta*dst[i] + alfa*src[i] */
void gpuAddscale2(cudaSolverType alfa, cudaSolverType beta, cudaSolverType* dst,
                  const cudaSolverType* src, size_t n, cudaStream_t stream = 0);

/** d[i] = -d[i] */
void gpuNeg(cudaSolverType* d, size_t n, cudaStream_t stream = 0);

/** dst = src  (device-to-device memcpy) */
void gpuEq(cudaSolverType* dst, const cudaSolverType* src, size_t n,
           cudaStream_t stream = 0);

// =========================================================================
//  Fused triple operations (3 arrays in one kernel launch)
// =========================================================================

/** dX[i] = -dX[i];  dY[i] = -dY[i];  dZ[i] = -dZ[i]; */
void gpuNeg3(cudaSolverType* dX, cudaSolverType* dY, cudaSolverType* dZ,
             size_t n, cudaStream_t stream = 0);

/** dstX[i] -= srcX[i]; dstY -= srcY; dstZ -= srcZ; */
void gpuSub3(cudaSolverType* dstX, const cudaSolverType* srcX,
             cudaSolverType* dstY, const cudaSolverType* srcY,
             cudaSolverType* dstZ, const cudaSolverType* srcZ, size_t n,
             cudaStream_t stream = 0);

/** dX[i] *= alfa;  dY[i] *= alfa;  dZ[i] *= alfa; */
void gpuScale3(cudaSolverType* dX, cudaSolverType* dY, cudaSolverType* dZ,
               cudaSolverType alfa, size_t n, cudaStream_t stream = 0);

/** dstX += srcAX + srcBX; dstY += srcAY + srcBY; dstZ += srcAZ + srcBZ; */
void gpuSumAddTwo3(cudaSolverType* dstX, const cudaSolverType* srcAX,
                   const cudaSolverType* srcBX, cudaSolverType* dstY,
                   const cudaSolverType* srcAY, const cudaSolverType* srcBY,
                   cudaSolverType* dstZ, const cudaSolverType* srcAZ,
                   const cudaSolverType* srcBZ, size_t n,
                   cudaStream_t stream = 0);

/** Zero N arrays at once:  d_ptrs[f][i] = 0  for f=0..nFields-1 */
void gpuSetAll0_N(cudaSolverType** d_ptrs, int nFields, size_t n,
                  cudaStream_t stream = 0);

/** dstX[i] = srcX[i]*alfa; dstY = srcY*alfa; dstZ = srcZ*alfa; */
void gpuScaleCopy3(cudaSolverType* dstX, const cudaSolverType* srcX,
                   cudaSolverType* dstY, const cudaSolverType* srcY,
                   cudaSolverType* dstZ, const cudaSolverType* srcZ,
                   cudaSolverType alfa, size_t n, cudaStream_t stream = 0);

/** dstX[i] += srcX[i]; dstY += srcY; dstZ += srcZ; */
void gpuSum3(cudaSolverType* dstX, const cudaSolverType* srcX,
             cudaSolverType* dstY, const cudaSolverType* srcY,
             cudaSolverType* dstZ, const cudaSolverType* srcZ, size_t n,
             cudaStream_t stream = 0);

/** dstX[i] += alfa*srcX[i]; dstY += alfa*srcY; dstZ += alfa*srcZ; */
void gpuAddscale3(cudaSolverType alfa, cudaSolverType* dstX,
                  const cudaSolverType* srcX, cudaSolverType* dstY,
                  const cudaSolverType* srcY, cudaSolverType* dstZ,
                  const cudaSolverType* srcZ, size_t n,
                  cudaStream_t stream = 0);

/** dstX[i] = alfa*srcX[i]+beta*dstX[i];  (triple) */
void gpuAddscale2_3(cudaSolverType alfa, cudaSolverType beta,
                    cudaSolverType* dstX, const cudaSolverType* srcX,
                    cudaSolverType* dstY, const cudaSolverType* srcY,
                    cudaSolverType* dstZ, const cudaSolverType* srcZ, size_t n,
                    cudaStream_t stream = 0);

// =========================================================================
//  Reduction operations
// =========================================================================

/**
 * Local dot product: sum_i a[i]*b[i].
 * d_scratch must point to a device cudaSolverType.
 * Returns the result synchronously (D2H copy after kernel).
 */
cudaSolverType gpuDot(const cudaSolverType* a, const cudaSolverType* b,
                      size_t n, cudaSolverType* d_scratch,
                      cudaStream_t stream = 0);

/**
 * Local squared norm: sum_i a[i]*a[i].
 * d_scratch must point to a device cudaSolverType.
 * Returns the result synchronously.
 */
cudaSolverType gpuNorm2(const cudaSolverType* a, size_t n,
                        cudaSolverType* d_scratch, cudaStream_t stream = 0);

/**
 * Async variants: launch reduction kernel + D→H copy into PINNED host
 * memory.  No stream synchronisation — caller must sync before reading.
 */
void gpuNorm2_async(const cudaSolverType* a, size_t n,
                    cudaSolverType* d_scratch, cudaSolverType* h_result,
                    cudaStream_t stream = 0);
void gpuDot_async(const cudaSolverType* a, const cudaSolverType* b, size_t n,
                  cudaSolverType* d_scratch, cudaSolverType* h_result,
                  cudaStream_t stream = 0);

/**
 * Batched dot products + norm² for GMRES Arnoldi step.
 * Computes:
 *   d_out[j] = dot(w, V_base + j*stride, n)  for j=0..k
 *   d_out[k+1] = norm2(w, n)
 *
 * d_out is a device array of >= (k+2) cudaSolverType elements (zeroed
 * internally). After call, d_out contains the LOCAL partial sums (caller must
 * MPI_Allreduce).
 */
void gpuBatchedDotNorm(const cudaSolverType* w, const cudaSolverType* V_base,
                       size_t stride, int k, size_t n, cudaSolverType* d_out,
                       cudaStream_t stream = 0);

// =========================================================================
//  Krylov space  <-->  physical space conversion
// =========================================================================

/**
 * GPU phys2solver: pack 3 node-based 3D fields [nx][ny][nz] into a single
 * interleaved 1D Krylov vector of length 3*(nx-2)*(ny-2)*(nz-2).
 *
 * Krylov index: ((i-1)*(ny-2)*(nz-2) + (j-1)*(nz-2) + (k-1))*3 + comp
 * Interior indices: i=1..nx-2, j=1..ny-2, k=1..nz-2.
 */
void gpuPhys2Solver3(cudaSolverType* d_solver, const cudaSolverType* d_physX,
                     const cudaSolverType* d_physY,
                     const cudaSolverType* d_physZ, int nx, int ny, int nz,
                     cudaStream_t stream = 0);

/**
 * GPU phys2solver for a single scalar field (Poisson).
 * Krylov index: (i-1)*(ny-2)*(nz-2) + (j-1)*(nz-2) + (k-1)
 */
void gpuPhys2Solver1(cudaSolverType* d_solver, const cudaSolverType* d_phys,
                     int nx, int ny, int nz, cudaStream_t stream = 0);

/**
 * GPU solver2phys: unpack 1D Krylov vector into 3 node-based fields.
 * Populates interior points [1..nx-2][1..ny-2][1..nz-2], ghosts untouched.
 */
void gpuSolver2Phys3(cudaSolverType* d_physX, cudaSolverType* d_physY,
                     cudaSolverType* d_physZ, const cudaSolverType* d_solver,
                     int nx, int ny, int nz, cudaStream_t stream = 0);

/**
 * GPU solver2phys for a single scalar field (Poisson).
 */
void gpuSolver2Phys1(cudaSolverType* d_phys, const cudaSolverType* d_solver,
                     int nx, int ny, int nz, cudaStream_t stream = 0);

// =========================================================================
//  Scratch buffer helpers
// =========================================================================

/** Allocate a small device scratch buffer (one cudaSolverType) for reductions.
 */
void gpuBlasAllocScratch(cudaSolverType** d_scratch);

/** Free the scratch buffer. */
void gpuBlasFreeScratch(cudaSolverType* d_scratch);

#endif // GPU_SOLVER

#endif // GPU_BLAS_CUH
