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
__global__ void k_chebyshevStep1(double* __restrict__ d_y,
                                 double* __restrict__ d_z,
                                 const double* __restrict__ d_b,
                                 const double* __restrict__ d_Ab,
                                 double invTheta,
                                 double coeff_b,
                                 double coeff_Ab,
                                 int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    double bv  = d_b[idx];
    double abv = d_Ab[idx];
    d_z[idx] = bv * invTheta;
    d_y[idx] = coeff_b * bv + coeff_Ab * abv;
}

void gpuChebyshevStep1(double* d_y, double* d_z,
                       const double* d_b, const double* d_Ab,
                       double theta, double delta, double rho,
                       int n, cudaStream_t stream)
{
    constexpr int BLK = 256;
    int nblk = (n + BLK - 1) / BLK;
    double invTheta = 1.0 / theta;
    double coeff_b  = 4.0 * rho / delta;            // 2*rho/delta * 2
    double coeff_Ab = -2.0 * rho / (delta * theta);  // 2*rho/delta * (-1/theta)
    k_chebyshevStep1<<<nblk, BLK, 0, stream>>>(
        d_y, d_z, d_b, d_Ab, invTheta, coeff_b, coeff_Ab, n);
}

// =========================================================================
//  Step N kernel:  w = fY*y + fB*b + fAy*Ay + fZ*z
// =========================================================================
__global__ void k_chebyshevStepN(double* __restrict__ d_w,
                                 const double* __restrict__ d_y,
                                 const double* __restrict__ d_z,
                                 const double* __restrict__ d_b,
                                 const double* __restrict__ d_Ay,
                                 double fY, double fB, double fAy, double fZ,
                                 int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    d_w[idx] = fY * d_y[idx] + fB * d_b[idx]
             + fAy * d_Ay[idx] + fZ * d_z[idx];
}

void gpuChebyshevStepN(double* d_w,
                       const double* d_y, const double* d_z,
                       const double* d_b, const double* d_Ay,
                       double delta, double sigma,
                       double rho, double rhoOld,
                       int n, cudaStream_t stream)
{
    constexpr int BLK = 256;
    int nblk = (n + BLK - 1) / BLK;
    double fY  =  rho * 2.0 * sigma;
    double fB  =  rho * 2.0 / delta;
    double fAy = -rho * 2.0 / delta;   // from  b - A(y)
    double fZ  = -rho * rhoOld;
    k_chebyshevStepN<<<nblk, BLK, 0, stream>>>(
        d_w, d_y, d_z, d_b, d_Ay, fY, fB, fAy, fZ, n);
}

#endif // GPU_SOLVER
