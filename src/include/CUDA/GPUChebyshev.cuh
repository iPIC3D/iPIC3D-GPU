#ifndef GPU_CHEBYSHEV_CUH
#define GPU_CHEBYSHEV_CUH

// =========================================================================
//  GPUChebyshev.cuh
//
//  GPU Chebyshev semi-iterative solver for the iPIC3D Maxwell system.
//
//  Provides two operational modes:
//    1. Full solver      — SolverType = Chebyshev, uses gpuMaxwellImage
//                          (with MPI)
//    2. Preconditioner   — communication-free Chebyshev polynomial in the
//                          rank-local Laplacian for the div(E) / div(B)
//                          Poisson solves (gpuChebyshevPrecondPoisson;
//                          the step kernels apply the operator,
//                          PoissonLocalOperator)
//
//  The Chebyshev iteration approximates  A^{-1} b  via a polynomial in A.
//  It requires eigenvalue bounds [eigMin, eigMax] of the operator A.
//  No inner products are needed → no MPI_Allreduce per iteration → ideal
//  for GPU acceleration.
//
//  Algorithm, for an operator A with spectrum in [eigMin, eigMax] (the
//  Poisson preconditioner applies it to -∇² and negates the result):
//    z = b / theta                                                [step 0]
//    y = (2*rho/delta) * (2*b - A(b)/theta)                      [step 1]
//    for n = 2..N:
//      w = rho_n * (2*sigma*y + (2/delta)*(b - A(y)) - rho_{n-1}*z)
//      z <- y,  y <- w                                           [rotate]
//    x += y                                                      [output]
//
//  Parameters:
//    theta = (eigMin + eigMax) / 2
//    delta = (eigMax - eigMin) / 2
//    sigma = theta / delta
//    rho_0 = 1/sigma,  rho_n = 1 / (2*sigma - rho_{n-1})
// =========================================================================

#include "cudaTypeDef.cuh"

// ---- Element-wise CUDA kernel wrappers --------------------------------

// Step 1: compute  z = b/theta  and  y = coeff_b*b + coeff_Ab*A(b)
//   coeff_b  = 4*rho / delta
//   coeff_Ab = -2*rho / (delta*theta)
void gpuChebyshevStep1(cudaSolverType* d_y, cudaSolverType* d_z,
                       const cudaSolverType* d_b, const cudaSolverType* d_Ab,
                       cudaSolverType theta, cudaSolverType delta,
                       cudaSolverType rho, int n, cudaStream_t stream);

// Step N: compute  w = fY*y + fB*b + fAy*A(y) + fZ*z
//   fY  = rho * 2 * sigma
//   fB  = rho * 2 / delta
//   fAy = -rho * 2 / delta          (sign encodes b - A(y))
//   fZ  = -rho * rhoOld
void gpuChebyshevStepN(cudaSolverType* d_w, const cudaSolverType* d_y,
                       const cudaSolverType* d_z, const cudaSolverType* d_b,
                       const cudaSolverType* d_Ay, cudaSolverType delta,
                       cudaSolverType sigma, cudaSolverType rho,
                       cudaSolverType rhoOld, int n, cudaStream_t stream);

// ---- Poisson preconditioner: steps with the operator in the same pass --

/** The communication-free -∇² of the Poisson preconditioner on the packed
 *  interior cells p = (i nj + j) nk + k, 0 <= i < ni, ... (the Poisson Krylov
 *  layout of gpuPhys2Solver1): 7-point stencil, and a neighbour outside the
 *  rank's interior is g[f] times the cell, f = Xleft, Xright, Yleft, Yright,
 *  Zleft, Zright, g in {-1, 0, +1} (poissonLocalGhostRules). */
struct PoissonLocalOperator {
  int ni, nj, nk;
  cudaSolverType invdx2, invdy2, invdz2;
  cudaSolverType g[6];
};

// Step 1 with A(b) = PoissonLocalOperator applied to b in the same pass:
//   z = b/theta,  out = outScale * (coeff_b*b + coeff_Ab*A(b))
// (coefficients of gpuChebyshevStep1); outScale = -1 writes -y.
void gpuChebyshevPoissonStep1(cudaSolverType* out, cudaSolverType* z,
                              const cudaSolverType* b,
                              const PoissonLocalOperator& op,
                              cudaSolverType theta, cudaSolverType delta,
                              cudaSolverType rho, cudaSolverType outScale,
                              cudaStream_t stream);

// Step N with A(y) = PoissonLocalOperator applied to y in the same pass:
//   out = outScale * (fY*y + fB*b + fAy*A(y) + fZ*z)
// (coefficients of gpuChebyshevStepN).  out may be z (in-place update); it
// must not alias y or b.
void gpuChebyshevPoissonStepN(cudaSolverType* out, const cudaSolverType* y,
                              const cudaSolverType* z, const cudaSolverType* b,
                              const PoissonLocalOperator& op,
                              cudaSolverType delta, cudaSolverType sigma,
                              cudaSolverType rho, cudaSolverType rhoOld,
                              cudaSolverType outScale, cudaStream_t stream);

#endif // GPU_CHEBYSHEV_CUH
