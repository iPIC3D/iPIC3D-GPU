// =========================================================================
//  GPUChebyshev.cu
//
//  CUDA kernels for the Chebyshev semi-iterative solver.
//  All kernels operate on flat Krylov vectors (interleaved Ex,Ey,Ez at
//  interior nodes).
// =========================================================================
#ifdef GPU_SOLVER

#include "GPUChebyshev.cuh"

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
}

#endif // GPU_SOLVER
