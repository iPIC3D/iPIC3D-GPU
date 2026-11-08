// =========================================================================
//  GPUChebyshev.cu
//
//  CUDA kernels for the Chebyshev semi-iterative solver.  The generic step
//  kernels operate on flat Krylov vectors (the Maxwell one interleaves
//  Ex,Ey,Ez at interior nodes); the Poisson step kernels evaluate the
//  communication-free -∇² on the packed interior cells in the same pass.
// =========================================================================

#include "GPUChebyshev.cuh"
#include "GPUPointLaunch.cuh"

// =========================================================================
//  Step 1 kernel:  z = b/theta,  y = coeff_b*b + coeff_Ab*Ab
// =========================================================================
template <typename T>
__global__ void k_chebyshevStep1(T* __restrict__ d_y, T* __restrict__ d_z,
                                 const T* __restrict__ d_b,
                                 const T* __restrict__ d_Ab, T invTheta,
                                 T coeff_b, T coeff_Ab, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n)
    return;
  T bv = d_b[idx];
  T abv = d_Ab[idx];
  d_z[idx] = bv * invTheta;
  d_y[idx] = coeff_b * bv + coeff_Ab * abv;
}

void gpuChebyshevStep1(cudaSolverType* d_y, cudaSolverType* d_z,
                       const cudaSolverType* d_b, const cudaSolverType* d_Ab,
                       cudaSolverType theta, cudaSolverType delta,
                       cudaSolverType rho, int n, cudaStream_t stream) {
  constexpr int BLK = 256;
  int nblk = (n + BLK - 1) / BLK;
  cudaSolverType invTheta = 1.0 / theta;
  cudaSolverType coeff_b = 4.0 * rho / delta; // 2*rho/delta * 2
  cudaSolverType coeff_Ab =
      -2.0 * rho / (delta * theta); // 2*rho/delta * (-1/theta)
  k_chebyshevStep1<<<nblk, BLK, 0, stream>>>(d_y, d_z, d_b, d_Ab, invTheta,
                                             coeff_b, coeff_Ab, n);
  cudaErrChk(cudaGetLastError());
}

// =========================================================================
//  Step N kernel:  w = fY*y + fB*b + fAy*Ay + fZ*z
// =========================================================================
template <typename T>
__global__ void
k_chebyshevStepN(T* __restrict__ d_w, const T* __restrict__ d_y,
                 const T* __restrict__ d_z, const T* __restrict__ d_b,
                 const T* __restrict__ d_Ay, T fY, T fB, T fAy, T fZ, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n)
    return;
  d_w[idx] = fY * d_y[idx] + fB * d_b[idx] + fAy * d_Ay[idx] + fZ * d_z[idx];
}

void gpuChebyshevStepN(cudaSolverType* d_w, const cudaSolverType* d_y,
                       const cudaSolverType* d_z, const cudaSolverType* d_b,
                       const cudaSolverType* d_Ay, cudaSolverType delta,
                       cudaSolverType sigma, cudaSolverType rho,
                       cudaSolverType rhoOld, int n, cudaStream_t stream) {
  constexpr int BLK = 256;
  int nblk = (n + BLK - 1) / BLK;
  cudaSolverType fY = rho * 2.0 * sigma;
  cudaSolverType fB = rho * 2.0 / delta;
  cudaSolverType fAy = -rho * 2.0 / delta; // from  b - A(y)
  cudaSolverType fZ = -rho * rhoOld;
  k_chebyshevStepN<<<nblk, BLK, 0, stream>>>(d_w, d_y, d_z, d_b, d_Ay, fY, fB,
                                             fAy, fZ, n);
  cudaErrChk(cudaGetLastError());
}

// =========================================================================
//  Poisson preconditioner steps: A = -∇² of PoissonLocalOperator, evaluated
//  in the step kernel on the packed interior cells (one kernel per step, no
//  unpacked array, no ghost layer).  Launch shape of GPUPointLaunch.cuh over
//  ni x nj x nk, k on threadIdx.x.
// =========================================================================

/** -∇² y at the packed cell p = (i nj + j) nk + k.  Branch-free: every thread
 *  loads the six neighbours; outside the interior the neighbour index is the
 *  cell itself and its weight the face factor g, inside the weight is 1. */
template <typename T>
__device__ __forceinline__ T negLapLocal(const T* __restrict__ y, int i, int j,
                                         int k, int p,
                                         const PoissonLocalOperator& op) {
  const int sx = op.nj * op.nk, sy = op.nk;
  const bool inXm = i > 0, inXp = i < op.ni - 1;
  const bool inYm = j > 0, inYp = j < op.nj - 1;
  const bool inZm = k > 0, inZp = k < op.nk - 1;
  const T f = y[p];
  const T xm = (inXm ? T(1) : op.g[0]) * y[inXm ? p - sx : p];
  const T xp = (inXp ? T(1) : op.g[1]) * y[inXp ? p + sx : p];
  const T ym = (inYm ? T(1) : op.g[2]) * y[inYm ? p - sy : p];
  const T yp = (inYp ? T(1) : op.g[3]) * y[inYp ? p + sy : p];
  const T zm = (inZm ? T(1) : op.g[4]) * y[inZm ? p - 1 : p];
  const T zp = (inZp ? T(1) : op.g[5]) * y[inZp ? p + 1 : p];
  return -((xm - T(2) * f + xp) * op.invdx2 + (ym - T(2) * f + yp) * op.invdy2 +
           (zm - T(2) * f + zp) * op.invdz2);
}

template <typename T>
__global__ void k_chebyshevPoissonStep1(T* __restrict__ out, T* __restrict__ z,
                                        const T* __restrict__ b,
                                        PoissonLocalOperator op, T invTheta,
                                        T coeffB, T coeffAb, T outScale) {
  const int i = blockIdx.z * blockDim.z + threadIdx.z;
  const int j = blockIdx.y * blockDim.y + threadIdx.y;
  const int k = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= op.ni || j >= op.nj || k >= op.nk)
    return;
  const int p = (i * op.nj + j) * op.nk + k;
  const T bv = b[p];
  z[p] = bv * invTheta;
  out[p] = outScale * (coeffB * bv + coeffAb * negLapLocal(b, i, j, k, p, op));
}

// out and z may be the same array (in-place update), so neither is
// __restrict__; y and b never alias them.
template <typename T>
__global__ void k_chebyshevPoissonStepN(T* out, const T* __restrict__ y,
                                        const T* z, const T* __restrict__ b,
                                        PoissonLocalOperator op, T fY, T fB,
                                        T fAy, T fZ, T outScale) {
  const int i = blockIdx.z * blockDim.z + threadIdx.z;
  const int j = blockIdx.y * blockDim.y + threadIdx.y;
  const int k = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= op.ni || j >= op.nj || k >= op.nk)
    return;
  const int p = (i * op.nj + j) * op.nk + k;
  out[p] = outScale * (fY * y[p] + fB * b[p] +
                       fAy * negLapLocal(y, i, j, k, p, op) + fZ * z[p]);
}

void gpuChebyshevPoissonStep1(cudaSolverType* out, cudaSolverType* z,
                              const cudaSolverType* b,
                              const PoissonLocalOperator& op,
                              cudaSolverType theta, cudaSolverType delta,
                              cudaSolverType rho, cudaSolverType outScale,
                              cudaStream_t stream) {
  const PointLaunch L = pointLaunch(op.ni, op.nj, op.nk);
  k_chebyshevPoissonStep1<<<L.grid, L.block, 0, stream>>>(
      out, z, b, op, 1.0 / theta, 4.0 * rho / delta,
      -2.0 * rho / (delta * theta), outScale);
  cudaErrChk(cudaGetLastError());
}

void gpuChebyshevPoissonStepN(cudaSolverType* out, const cudaSolverType* y,
                              const cudaSolverType* z, const cudaSolverType* b,
                              const PoissonLocalOperator& op,
                              cudaSolverType delta, cudaSolverType sigma,
                              cudaSolverType rho, cudaSolverType rhoOld,
                              cudaSolverType outScale, cudaStream_t stream) {
  const PointLaunch L = pointLaunch(op.ni, op.nj, op.nk);
  k_chebyshevPoissonStepN<<<L.grid, L.block, 0, stream>>>(
      out, y, z, b, op, rho * 2.0 * sigma, rho * 2.0 / delta,
      -rho * 2.0 / delta, -rho * rhoOld, outScale);
  cudaErrChk(cudaGetLastError());
}
