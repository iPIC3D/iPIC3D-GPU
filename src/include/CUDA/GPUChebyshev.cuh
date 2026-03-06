// =========================================================================
//  GPUChebyshev.cuh
//
//  GPU Chebyshev semi-iterative solver for the iPIC3D Maxwell system.
//
//  Provides two operational modes:
//    1. Full solver      — replaces GMRES,  uses gpuMaxwellImage  (with MPI)
//    2. Preconditioner   — communication-free, uses gpuMaxwellImageLocal
//
//  The Chebyshev iteration approximates  A^{-1} b  via a polynomial in A.
//  It requires eigenvalue bounds [eigMin, eigMax] of the operator A.
//  No inner products are needed → no MPI_Allreduce per iteration → ideal
//  for GPU acceleration.
//
//  Algorithm (applied to the NEGATED positive-definite operator -A):
//    z = b / theta                                                [step 0]
//    y = (2*rho/delta) * (2*b - A(b)/theta)                      [step 1]
//    for n = 2..N:
//      w = rho_n * (2*sigma*y + (2/delta)*(b - A(y)) - rho_{n-1}*z)
//      z <- y,  y <- w                                           [rotate]
//    x += -y                                                     [output]
//
//  Parameters:
//    theta = (eigMin + eigMax) / 2
//    delta = (eigMax - eigMin) / 2
//    sigma = theta / delta
//    rho_0 = 1/sigma,  rho_n = 1 / (2*sigma - rho_{n-1})
// =========================================================================
#pragma once
#ifdef GPU_SOLVER

#include "cudaTypeDef.cuh"

// ---- Element-wise CUDA kernel wrappers --------------------------------

// Step 1: compute  z = b/theta  and  y = coeff_b*b + coeff_Ab*A(b)
//   coeff_b  = 4*rho / delta
//   coeff_Ab = -2*rho / (delta*theta)
void gpuChebyshevStep1(double* d_y, double* d_z,
                       const double* d_b, const double* d_Ab,
                       double theta, double delta, double rho,
                       int n, cudaStream_t stream);

// Step N: compute  w = fY*y + fB*b + fAy*A(y) + fZ*z
//   fY  = rho * 2 * sigma
//   fB  = rho * 2 / delta
//   fAy = -rho * 2 / delta          (sign encodes b - A(y))
//   fZ  = -rho * rhoOld
void gpuChebyshevStepN(double* d_w,
                       const double* d_y, const double* d_z,
                       const double* d_b, const double* d_Ay,
                       double delta, double sigma,
                       double rho, double rhoOld,
                       int n, cudaStream_t stream);

#endif // GPU_SOLVER
