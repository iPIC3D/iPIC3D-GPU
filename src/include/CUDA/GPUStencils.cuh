/**
 * @file GPUStencils.cuh
 * @brief GPU stencil kernels for the iPIC3D GPU field solver.
 *
 * Provides GPU counterparts of all Grid3DCU stencil operations:
 *   gradC2N, gradN2C, divN2C, divC2N, curlC2N, curlN2C,
 *   interpC2N, interpN2C, divSymmTensorN2C, smooth (box stencil).
 *
 * All kernels operate on flat device pointers (row-major [nx][ny][nz]).
 * Grid dimensions include ghost cells.
 *
 * Compile with -DGPU_SOLVER to activate.
 */

#ifndef GPU_STENCILS_CUH
#define GPU_STENCILS_CUH

#include "cudaTypeDef.cuh"

#ifdef GPU_SOLVER

// =========================================================================
//  Gradient operations
// =========================================================================

/**
 * Gradient center→node:  gradX/Y/Z_N = ∇(scFieldC).
 * Output on node grid [nxn][nyn][nzn], interior i=1..nxn-2.
 * Input on center grid [nxc][nyc][nzc] (nxc=nxn-1, etc.).
 * Stencil: 8-point finite difference.
 */
void gpuGradC2N(double* gradXN, double* gradYN, double* gradZN,
                const double* scFieldC,
                int nxn, int nyn, int nzn,
                double invdx, double invdy, double invdz,
                cudaStream_t stream = 0);

/**
 * Gradient node→center:  gradX/Y/Z_C = ∇(scFieldN).
 * Output on center grid [nxc][nyc][nzc], interior i=1..nxc-2.
 * Input on node grid [nxn][nyn][nzn] (nxn=nxc+1, etc.).
 */
void gpuGradN2C(double* gradXC, double* gradYC, double* gradZC,
                const double* scFieldN,
                int nxc, int nyc, int nzc,
                double invdx, double invdy, double invdz,
                cudaStream_t stream = 0);

// =========================================================================
//  Divergence operations
// =========================================================================

/**
 * Divergence node→center:  divC = ∇·(vecX_N, vecY_N, vecZ_N).
 * Output on center grid [nxc][nyc][nzc], interior i=1..nxc-2.
 * Input on node grid [nxn][nyn][nzn].
 */
void gpuDivN2C(double* divC,
               const double* vecXN, const double* vecYN, const double* vecZN,
               int nxc, int nyc, int nzc,
               double invdx, double invdy, double invdz,
               cudaStream_t stream = 0);

/**
 * Divergence center→node:  divN = ∇·(vecX_C, vecY_C, vecZ_C).
 * Output on node grid [nxn][nyn][nzn], interior i=1..nxn-2.
 * Input on center grid [nxc][nyc][nzc].
 */
void gpuDivC2N(double* divN,
               const double* vecXC, const double* vecYC, const double* vecZC,
               int nxn, int nyn, int nzn,
               double invdx, double invdy, double invdz,
               cudaStream_t stream = 0);

// =========================================================================
//  Curl operations
// =========================================================================

/**
 * Curl center→node:  curl_N = ∇×(vec_C).
 * Output on node grid [nxn][nyn][nzn], interior i=1..nxn-2.
 * Input on center grid [nxc][nyc][nzc].
 */
void gpuCurlC2N(double* curlXN, double* curlYN, double* curlZN,
                const double* vecXC, const double* vecYC, const double* vecZC,
                int nxn, int nyn, int nzn,
                double invdx, double invdy, double invdz,
                cudaStream_t stream = 0);

/**
 * Curl node→center:  curl_C = ∇×(vec_N).
 * Output on center grid [nxc][nyc][nzc], interior i=1..nxc-2.
 * Input on node grid [nxn][nyn][nzn].
 */
void gpuCurlN2C(double* curlXC, double* curlYC, double* curlZC,
                const double* vecXN, const double* vecYN, const double* vecZN,
                int nxc, int nyc, int nzc,
                double invdx, double invdy, double invdz,
                cudaStream_t stream = 0);

// =========================================================================
//  Interpolation operations
// =========================================================================

/**
 * Interpolate center→node:  fieldN = interp(fieldC).
 * 8-point average from surrounding center cells.
 * Output on node grid, interior i=1..nxn-2.
 */
void gpuInterpC2N(double* fieldN, const double* fieldC,
                  int nxn, int nyn, int nzn,
                  cudaStream_t stream = 0);

/**
 * Interpolate node→center:  fieldC = interp(fieldN).
 * 8-point average from surrounding nodes.
 * Output on center grid, interior i=1..nxc-2.
 */
void gpuInterpN2C(double* fieldC, const double* fieldN,
                  int nxc, int nyc, int nzc,
                  cudaStream_t stream = 0);

// =========================================================================
//  Symmetric tensor divergence
// =========================================================================

/**
 * Divergence of a symmetric tensor, node→center.
 *   divCX = d(pXX)/dx + d(pXY)/dy + d(pXZ)/dz
 *   divCY = d(pXY)/dx + d(pYY)/dy + d(pYZ)/dz
 *   divCZ = d(pXZ)/dx + d(pYZ)/dy + d(pZZ)/dz
 *
 * Tensor components are species-sliced: pXX[is] starts at pXX_base + is*nxn*nyn*nzn.
 * Output on center grid, interior i=1..nxc-2.
 * is = species offset into the 4D arrays (pXX = pXX_4D + is*nxn*nyn*nzn).
 */
void gpuDivSymmTensorN2C(double* divCX, double* divCY, double* divCZ,
                         const double* pXX, const double* pXY, const double* pXZ,
                         const double* pYY, const double* pYZ, const double* pZZ,
                         int nxc, int nyc, int nzc,
                         double invdx, double invdy, double invdz,
                         cudaStream_t stream = 0);

// =========================================================================
//  Smoothing (box stencil)
// =========================================================================

/**
 * Single iteration of 6-point box smooth:
 *   out[i][j][k] = alpha * in[i][j][k] + beta3D * (6 neighbours)
 *
 * Both 'in' and 'out' must be on the same grid (node- or center-based).
 * Interior: i=1..nx-2, j=1..ny-2, k=1..nz-2.
 * Ghost cells must be valid (from prior halo exchange).
 */
void gpuSmoothStep(double* out, const double* in,
                   int nx, int ny, int nz,
                   double alpha, double beta3D,
                   cudaStream_t stream = 0);

// =========================================================================
//  Interior / boundary stencil variants for halo-computation overlap.
//  Interior covers [2..n-3] (no ghost dependency).
//  Boundary covers [1..n-2] minus the interior block.
// =========================================================================
#ifdef HALO_OVERLAP

void gpuDivC2N_interior(double* divN,
                        const double* vecXC, const double* vecYC, const double* vecZC,
                        int nxn, int nyn, int nzn,
                        double invdx, double invdy, double invdz,
                        cudaStream_t stream = 0);
void gpuDivC2N_boundary(double* divN,
                        const double* vecXC, const double* vecYC, const double* vecZC,
                        int nxn, int nyn, int nzn,
                        double invdx, double invdy, double invdz,
                        cudaStream_t stream = 0);

void gpuGradC2N_interior(double* gradXN, double* gradYN, double* gradZN,
                         const double* scFieldC,
                         int nxn, int nyn, int nzn,
                         double invdx, double invdy, double invdz,
                         cudaStream_t stream = 0);
void gpuGradC2N_boundary(double* gradXN, double* gradYN, double* gradZN,
                         const double* scFieldC,
                         int nxn, int nyn, int nzn,
                         double invdx, double invdy, double invdz,
                         cudaStream_t stream = 0);

void gpuInterpC2N_interior(double* fieldN, const double* fieldC,
                           int nxn, int nyn, int nzn,
                           cudaStream_t stream = 0);
void gpuInterpC2N_boundary(double* fieldN, const double* fieldC,
                           int nxn, int nyn, int nzn,
                           cudaStream_t stream = 0);

void gpuSmoothStep_interior(double* out, const double* in,
                            int nx, int ny, int nz,
                            double alpha, double beta3D,
                            cudaStream_t stream = 0);
void gpuSmoothStep_boundary(double* out, const double* in,
                            int nx, int ny, int nz,
                            double alpha, double beta3D,
                            cudaStream_t stream = 0);

#endif // HALO_OVERLAP

#endif // GPU_SOLVER
#endif // GPU_STENCILS_CUH
