/**
 * @file GPUMaxwellLocal.cu
 * @brief Communication-free (local) Maxwell image operator — fused CUDA kernels.
 *
 * Two kernels that together compute  A·E  without any MPI communication:
 *
 *   k_maxwellLocalCenterOps  — fuses 3×gradN2C + divN2C  (center-level)
 *   k_maxwellLocalNodeFused  — fuses 3×divC2N + gradC2N + arithmetic
 *                               + phys2solver packing     (node-level)
 *
 * Ghost cells are assumed to be zero (set by caller).
 * Boundary image corrections are not applied in these kernels; callers can
 * enforce them after unpacking if they need the local operator to match the
 * full Maxwell image more closely.
 *
 * See GPUMaxwellLocal.cuh for API documentation.
 */

#include "GPUMaxwellLocal.cuh"

#ifdef GPU_SOLVER

// =========================================================================
//  Thread-block configuration (matches GPUStencils.cu)
// =========================================================================
static constexpr int BX = 8;
static constexpr int BY = 8;
static constexpr int BZ = 4;  // 8×8×4 = 256 threads

// =========================================================================
//  Helpers
// =========================================================================

/** Compute 3D grid covering interior cells: i ∈ [1, n-2]. */
static inline dim3 interiorGrid(int nx, int ny, int nz)
{
    return dim3(((nx - 2) + BX - 1) / BX,
                ((ny - 2) + BY - 1) / BY,
                ((nz - 2) + BZ - 1) / BZ);
}

// =========================================================================
//  Kernel 1:  Fused center-level operations
//
//  Combines 3× gradN2C(Ex,Ey,Ez) + divN2C(Dx,Dy,Dz) into ONE kernel.
//  Each thread processes one interior center cell (i,j,k).
//
//  Stencil: reads 8 corner nodes  (i..i+1) × (j..j+1) × (k..k+1)
//           for each of 6 input node arrays.
//  Output:  10 center-sized arrays.
// =========================================================================

__global__ void k_maxwellLocalCenterOps(
    double* __restrict__ gExX, double* __restrict__ gExY, double* __restrict__ gExZ,
    double* __restrict__ gEyX, double* __restrict__ gEyY, double* __restrict__ gEyZ,
    double* __restrict__ gEzX, double* __restrict__ gEzY, double* __restrict__ gEzZ,
    double* __restrict__ divD,
    const double* __restrict__ vX, const double* __restrict__ vY, const double* __restrict__ vZ,
    const double* __restrict__ dX, const double* __restrict__ dY, const double* __restrict__ dZ,
    int nxc, int nyc, int nzc,
    int nyn, int nzn,
    double invdx, double invdy, double invdz)
{
    int i = blockIdx.x * BX + threadIdx.x + 1;
    int j = blockIdx.y * BY + threadIdx.y + 1;
    int k = blockIdx.z * BZ + threadIdx.z + 1;
    if (i > nxc - 2 || j > nyc - 2 || k > nzc - 2) return;

    // The 8 corner node positions for this center cell (i,j,k):
    //   node indices: (i, i+1) × (j, j+1) × (k, k+1)

    // Flat index helper for node arrays  [nxn][nyn][nzn]
    #define NI(ii, jj, kk) ((ii) * nyn * nzn + (jj) * nzn + (kk))

    // Load 8 corner positions — process one input field at a time
    // to limit register pressure.  Compiler may still keep them
    // in registers across the block if beneficial.

    const int ci = i * nyc * nzc + j * nzc + k;   // flat center index
    const double Q = 0.25;

    // ---- gradN2C(Ex) → gExX, gExY, gExZ ----
    {
        double n000 = vX[NI(i  , j  , k  )];
        double n001 = vX[NI(i  , j  , k+1)];
        double n010 = vX[NI(i  , j+1, k  )];
        double n011 = vX[NI(i  , j+1, k+1)];
        double n100 = vX[NI(i+1, j  , k  )];
        double n101 = vX[NI(i+1, j  , k+1)];
        double n110 = vX[NI(i+1, j+1, k  )];
        double n111 = vX[NI(i+1, j+1, k+1)];

        gExX[ci] = Q * invdx * ((n100-n000) + (n101-n001) + (n110-n010) + (n111-n011));
        gExY[ci] = Q * invdy * ((n010-n000) + (n011-n001) + (n110-n100) + (n111-n101));
        gExZ[ci] = Q * invdz * ((n001-n000) + (n101-n100) + (n011-n010) + (n111-n110));
    }

    // ---- gradN2C(Ey) → gEyX, gEyY, gEyZ ----
    {
        double n000 = vY[NI(i  , j  , k  )];
        double n001 = vY[NI(i  , j  , k+1)];
        double n010 = vY[NI(i  , j+1, k  )];
        double n011 = vY[NI(i  , j+1, k+1)];
        double n100 = vY[NI(i+1, j  , k  )];
        double n101 = vY[NI(i+1, j  , k+1)];
        double n110 = vY[NI(i+1, j+1, k  )];
        double n111 = vY[NI(i+1, j+1, k+1)];

        gEyX[ci] = Q * invdx * ((n100-n000) + (n101-n001) + (n110-n010) + (n111-n011));
        gEyY[ci] = Q * invdy * ((n010-n000) + (n011-n001) + (n110-n100) + (n111-n101));
        gEyZ[ci] = Q * invdz * ((n001-n000) + (n101-n100) + (n011-n010) + (n111-n110));
    }

    // ---- gradN2C(Ez) → gEzX, gEzY, gEzZ ----
    {
        double n000 = vZ[NI(i  , j  , k  )];
        double n001 = vZ[NI(i  , j  , k+1)];
        double n010 = vZ[NI(i  , j+1, k  )];
        double n011 = vZ[NI(i  , j+1, k+1)];
        double n100 = vZ[NI(i+1, j  , k  )];
        double n101 = vZ[NI(i+1, j  , k+1)];
        double n110 = vZ[NI(i+1, j+1, k  )];
        double n111 = vZ[NI(i+1, j+1, k+1)];

        gEzX[ci] = Q * invdx * ((n100-n000) + (n101-n001) + (n110-n010) + (n111-n011));
        gEzY[ci] = Q * invdy * ((n010-n000) + (n011-n001) + (n110-n100) + (n111-n101));
        gEzZ[ci] = Q * invdz * ((n001-n000) + (n101-n100) + (n011-n010) + (n111-n110));
    }

    // ---- divN2C(D) → divD ----
    //   ∇·D = d(Dx)/dx + d(Dy)/dy + d(Dz)/dz
    {
        // d(Dx)/dx
        double compX = Q * invdx * (
            (dX[NI(i+1,j  ,k  )] - dX[NI(i,j  ,k  )]) +
            (dX[NI(i+1,j  ,k+1)] - dX[NI(i,j  ,k+1)]) +
            (dX[NI(i+1,j+1,k  )] - dX[NI(i,j+1,k  )]) +
            (dX[NI(i+1,j+1,k+1)] - dX[NI(i,j+1,k+1)]));
        // d(Dy)/dy
        double compY = Q * invdy * (
            (dY[NI(i  ,j+1,k  )] - dY[NI(i  ,j,k  )]) +
            (dY[NI(i  ,j+1,k+1)] - dY[NI(i  ,j,k+1)]) +
            (dY[NI(i+1,j+1,k  )] - dY[NI(i+1,j,k  )]) +
            (dY[NI(i+1,j+1,k+1)] - dY[NI(i+1,j,k+1)]));
        // d(Dz)/dz
        double compZ = Q * invdz * (
            (dZ[NI(i  ,j  ,k+1)] - dZ[NI(i  ,j  ,k)]) +
            (dZ[NI(i+1,j  ,k+1)] - dZ[NI(i+1,j  ,k)]) +
            (dZ[NI(i  ,j+1,k+1)] - dZ[NI(i  ,j+1,k)]) +
            (dZ[NI(i+1,j+1,k+1)] - dZ[NI(i+1,j+1,k)]));

        divD[ci] = compX + compY + compZ;
    }
    #undef NI
}

// =========================================================================
//  Kernel 2:  Fused node-level operations + Krylov packing
//
//  Combines:
//    3× divC2N  (Laplacians from 9 center gradient arrays)
//    1× gradC2N (gradient of divD)
//    negate, subtract, scale(dt²), add D, add E
//    phys2solver packing (directly into Krylov vector)
//
//  Each thread processes one interior node (i,j,k).
//
//  Stencil: reads 8 surrounding center cells (i-1..i) × (j-1..j) × (k-1..k)
//           for each of 10 center arrays, plus 6 node values at (i,j,k).
//  Output:  3 interleaved doubles in the Krylov vector d_im.
// =========================================================================

__global__ void k_maxwellLocalNodeFused(
    double* __restrict__ d_im,
    // 9 center arrays: gradients of E (for Laplacian via divC2N)
    const double* __restrict__ gExX, const double* __restrict__ gExY, const double* __restrict__ gExZ,
    const double* __restrict__ gEyX, const double* __restrict__ gEyY, const double* __restrict__ gEyZ,
    const double* __restrict__ gEzX, const double* __restrict__ gEzY, const double* __restrict__ gEzZ,
    // 1 center array: divergence of D (for gradient via gradC2N)
    const double* __restrict__ divD,
    // 6 node arrays
    const double* __restrict__ vX, const double* __restrict__ vY, const double* __restrict__ vZ,
    const double* __restrict__ dX, const double* __restrict__ dY, const double* __restrict__ dZ,
    int nxn, int nyn, int nzn,
    double invdx, double invdy, double invdz,
    double dt2)
{
    int i = blockIdx.x * BX + threadIdx.x + 1;
    int j = blockIdx.y * BY + threadIdx.y + 1;
    int k = blockIdx.z * BZ + threadIdx.z + 1;
    if (i > nxn - 2 || j > nyn - 2 || k > nzn - 2) return;

    // Center grid dimensions
    int nyc = nyn - 1;
    int nzc = nzn - 1;

    // Pre-compute the 8 flat center indices for the surrounding cells:
    //   (i-1..i) × (j-1..j) × (k-1..k)
    // Binary encoding: bit2=i-dir, bit1=j-dir, bit0=k-dir
    //   0→(dim-1), 1→(dim)
    #define CI(ii, jj, kk) ((ii) * nyc * nzc + (jj) * nzc + (kk))
    int c000 = CI(i-1, j-1, k-1);
    int c001 = CI(i-1, j-1, k  );
    int c010 = CI(i-1, j  , k-1);
    int c011 = CI(i-1, j  , k  );
    int c100 = CI(i  , j-1, k-1);
    int c101 = CI(i  , j-1, k  );
    int c110 = CI(i  , j  , k-1);
    int c111 = CI(i  , j  , k  );
    #undef CI

    const double Q = 0.25;

    // ================================================================
    //  Compute lapX = divC2N( gExX, gExY, gExZ )
    //
    //  divC2N stencil (from k_divC2N in GPUStencils.cu):
    //    compX = 0.25*invdx * Σ (vecXC[i,*,*] - vecXC[i-1,*,*])
    //    compY = 0.25*invdy * Σ (vecYC[*,j,*] - vecYC[*,j-1,*])
    //    compZ = 0.25*invdz * Σ (vecZC[*,*,k] - vecZC[*,*,k-1])
    //    div = compX + compY + compZ
    // ================================================================

    // Helper macro: divC2N for a triplet of center arrays (vXC,vYC,vZC)
    // Returns the scalar divergence at node (i,j,k).
    #define DIVC2N(arrX, arrY, arrZ)                                     \
        (Q * invdx * ((arrX[c111] - arrX[c011]) + (arrX[c110] - arrX[c010]) \
                    + (arrX[c101] - arrX[c001]) + (arrX[c100] - arrX[c000]))  \
       + Q * invdy * ((arrY[c111] - arrY[c101]) + (arrY[c110] - arrY[c100]) \
                    + (arrY[c011] - arrY[c001]) + (arrY[c010] - arrY[c000]))  \
       + Q * invdz * ((arrZ[c111] - arrZ[c110]) + (arrZ[c101] - arrZ[c100]) \
                    + (arrZ[c011] - arrZ[c010]) + (arrZ[c001] - arrZ[c000])))

    double lapX = DIVC2N(gExX, gExY, gExZ);
    double lapY = DIVC2N(gEyX, gEyY, gEyZ);
    double lapZ = DIVC2N(gEzX, gEzY, gEzZ);

    // ================================================================
    //  Compute gdivX/Y/Z = gradC2N( divD )
    //
    //  gradC2N stencil (from k_gradC2N in GPUStencils.cu):
    //    gradX = 0.25*invdx * [(c111-c011)+(c110-c010)+(c101-c001)+(c100-c000)]
    //    gradY = 0.25*invdy * [(c111-c101)+(c110-c100)+(c011-c001)+(c010-c000)]
    //    gradZ = 0.25*invdz * [(c111-c110)+(c101-c100)+(c011-c010)+(c001-c000)]
    // ================================================================

    double dd000 = divD[c000], dd001 = divD[c001];
    double dd010 = divD[c010], dd011 = divD[c011];
    double dd100 = divD[c100], dd101 = divD[c101];
    double dd110 = divD[c110], dd111 = divD[c111];

    double gdivX = Q * invdx * ((dd111 - dd011) + (dd110 - dd010) + (dd101 - dd001) + (dd100 - dd000));
    double gdivY = Q * invdy * ((dd111 - dd101) + (dd110 - dd100) + (dd011 - dd001) + (dd010 - dd000));
    double gdivZ = Q * invdz * ((dd111 - dd110) + (dd101 - dd100) + (dd011 - dd010) + (dd001 - dd000));

    #undef DIVC2N

    // ================================================================
    //  Arithmetic:  im = dt² * (-lap - gdiv) + D + E
    //
    //  Matches the CPU MaxwellImage computation:
    //    image = -lap(E)                         [negate]
    //    image -= grad(div(μ·E))                 [subtract]
    //    image *= dt²                            [scale]
    //    image += D + E                          [add mass tensor + identity]
    // ================================================================

    int nidx = i * nyn * nzn + j * nzn + k;

    double ex = vX[nidx];
    double ey = vY[nidx];
    double ez = vZ[nidx];
    double dxv = dX[nidx];
    double dyv = dY[nidx];
    double dzv = dZ[nidx];

    double imX = dt2 * (-lapX - gdivX) + dxv + ex;
    double imY = dt2 * (-lapY - gdivY) + dyv + ey;
    double imZ = dt2 * (-lapZ - gdivZ) + dzv + ez;

    // ================================================================
    //  Pack to Krylov solver vector (phys2solver interleaved layout)
    //    sol_idx = ((i-1)*(ny2) + (j-1)) * nz2 + (k-1)
    //    d_im[sol_idx*3 + 0/1/2] = imX/imY/imZ
    // ================================================================

    int ny2 = nyn - 2;
    int nz2 = nzn - 2;
    int sol_idx = ((i - 1) * ny2 + (j - 1)) * nz2 + (k - 1);

    d_im[sol_idx * 3    ] = imX;
    d_im[sol_idx * 3 + 1] = imY;
    d_im[sol_idx * 3 + 2] = imZ;
}

// =========================================================================
//  Host wrappers
// =========================================================================

void gpuMaxwellLocalCenterOps(
    double* gExX, double* gExY, double* gExZ,
    double* gEyX, double* gEyY, double* gEyZ,
    double* gEzX, double* gEzY, double* gEzZ,
    double* divD,
    const double* vX, const double* vY, const double* vZ,
    const double* dX, const double* dY, const double* dZ,
    int nxc, int nyc, int nzc,
    double invdx, double invdy, double invdz,
    cudaStream_t stream)
{
    int nyn = nyc + 1, nzn = nzc + 1;
    dim3 grid = interiorGrid(nxc, nyc, nzc);
    dim3 block(BX, BY, BZ);
    k_maxwellLocalCenterOps<<<grid, block, 0, stream>>>(
        gExX, gExY, gExZ,
        gEyX, gEyY, gEyZ,
        gEzX, gEzY, gEzZ,
        divD,
        vX, vY, vZ,
        dX, dY, dZ,
        nxc, nyc, nzc, nyn, nzn,
        invdx, invdy, invdz);
    cudaErrChk(cudaGetLastError());
}

void gpuMaxwellLocalNodeFused(
    double* d_im,
    const double* gExX, const double* gExY, const double* gExZ,
    const double* gEyX, const double* gEyY, const double* gEyZ,
    const double* gEzX, const double* gEzY, const double* gEzZ,
    const double* divD,
    const double* vX, const double* vY, const double* vZ,
    const double* dX, const double* dY, const double* dZ,
    int nxn, int nyn, int nzn,
    double invdx, double invdy, double invdz,
    double dt2,
    cudaStream_t stream)
{
    dim3 grid = interiorGrid(nxn, nyn, nzn);
    dim3 block(BX, BY, BZ);
    k_maxwellLocalNodeFused<<<grid, block, 0, stream>>>(
        d_im,
        gExX, gExY, gExZ,
        gEyX, gEyY, gEyZ,
        gEzX, gEzY, gEzZ,
        divD,
        vX, vY, vZ,
        dX, dY, dZ,
        nxn, nyn, nzn,
        invdx, invdy, invdz,
        dt2);
    cudaErrChk(cudaGetLastError());
}

#endif // GPU_SOLVER
