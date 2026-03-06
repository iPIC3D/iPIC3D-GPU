/**
 * @file GPUStencils.cu
 * @brief CUDA kernel implementations for all grid stencil operations.
 *        See GPUStencils.cuh for API documentation.
 *
 * All kernels use a 3D thread-block layout (8×8×4 = 256 threads) mapped
 * to the interior grid points.  Each thread computes one output point.
 * The flat row-major index [i][j][k] = i*ny*nz + j*nz + k is used for
 * both node-based and center-based grids.
 */

#include "GPUStencils.cuh"

#ifdef GPU_SOLVER

// =========================================================================
//  Thread-block configuration for 3D stencils
// =========================================================================
static constexpr int BX = 8;
static constexpr int BY = 8;
static constexpr int BZ = 4;  // 8*8*4 = 256

/** Indexing macro for row-major [d1][d2][d3] layout. */
#define IDX(i, j, k, d2, d3) ((i) * (d2) * (d3) + (j) * (d3) + (k))

/**
 * Compute 3D grid dimensions for interior points.
 * Interior range: i ∈ [1, nx-2], j ∈ [1, ny-2], k ∈ [1, nz-2].
 * Number of interior points along each axis: nx-2, ny-2, nz-2.
 */
static inline dim3 stencilGrid(int nx, int ny, int nz)
{
    return dim3(((nx - 2) + BX - 1) / BX,
                ((ny - 2) + BY - 1) / BY,
                ((nz - 2) + BZ - 1) / BZ);
}

// =========================================================================
//  Gradient kernels
// =========================================================================

/**
 * gradC2N: gradient center→node.
 * Interior output on node grid: i=1..nxn-2 → reads center[i-1..i].
 * nxc = nxn-1, nyc = nyn-1, nzc = nzn-1.
 */
__global__ void k_gradC2N(double* __restrict__ gradXN,
                          double* __restrict__ gradYN,
                          double* __restrict__ gradZN,
                          const double* __restrict__ C,
                          int nxn, int nyn, int nzn,
                          int nyc, int nzc,
                          double invdx, double invdy, double invdz)
{
    int i = blockIdx.x * BX + threadIdx.x + 1;
    int j = blockIdx.y * BY + threadIdx.y + 1;
    int k = blockIdx.z * BZ + threadIdx.z + 1;
    if (i > nxn - 2 || j > nyn - 2 || k > nzn - 2) return;

    // Center indices (i-1, i) × (j-1, j) × (k-1, k)
    #define C_(ii, jj, kk) C[IDX(ii, jj, kk, nyc, nzc)]
    double c000 = C_(i-1, j-1, k-1);
    double c001 = C_(i-1, j-1, k  );
    double c010 = C_(i-1, j  , k-1);
    double c011 = C_(i-1, j  , k  );
    double c100 = C_(i  , j-1, k-1);
    double c101 = C_(i  , j-1, k  );
    double c110 = C_(i  , j  , k-1);
    double c111 = C_(i  , j  , k  );

    int nidx = IDX(i, j, k, nyn, nzn);
    gradXN[nidx] = 0.25 * invdx * ((c111 - c011) + (c110 - c010) + (c101 - c001) + (c100 - c000));
    gradYN[nidx] = 0.25 * invdy * ((c111 - c101) + (c110 - c100) + (c011 - c001) + (c010 - c000));
    gradZN[nidx] = 0.25 * invdz * ((c111 - c110) + (c101 - c100) + (c011 - c010) + (c001 - c000));
    #undef C_
}

/**
 * gradN2C: gradient node→center.
 * Interior output on center grid: i=1..nxc-2 → reads node[i..i+1].
 */
__global__ void k_gradN2C(double* __restrict__ gradXC,
                          double* __restrict__ gradYC,
                          double* __restrict__ gradZC,
                          const double* __restrict__ N,
                          int nxc, int nyc, int nzc,
                          int nyn, int nzn,
                          double invdx, double invdy, double invdz)
{
    int i = blockIdx.x * BX + threadIdx.x + 1;
    int j = blockIdx.y * BY + threadIdx.y + 1;
    int k = blockIdx.z * BZ + threadIdx.z + 1;
    if (i > nxc - 2 || j > nyc - 2 || k > nzc - 2) return;

    // Node indices (i, i+1) × (j, j+1) × (k, k+1)
    #define N_(ii, jj, kk) N[IDX(ii, jj, kk, nyn, nzn)]
    double n000 = N_(i  , j  , k  );
    double n001 = N_(i  , j  , k+1);
    double n010 = N_(i  , j+1, k  );
    double n011 = N_(i  , j+1, k+1);
    double n100 = N_(i+1, j  , k  );
    double n101 = N_(i+1, j  , k+1);
    double n110 = N_(i+1, j+1, k  );
    double n111 = N_(i+1, j+1, k+1);

    int cidx = IDX(i, j, k, nyc, nzc);
    gradXC[cidx] = 0.25 * invdx * ((n100 - n000) + (n101 - n001) + (n110 - n010) + (n111 - n011));
    gradYC[cidx] = 0.25 * invdy * ((n010 - n000) + (n011 - n001) + (n110 - n100) + (n111 - n101));
    gradZC[cidx] = 0.25 * invdz * ((n001 - n000) + (n101 - n100) + (n011 - n010) + (n111 - n110));
    #undef N_
}

// =========================================================================
//  Divergence kernels
// =========================================================================

/** divN2C: divergence node→center.  Same stencil as gradN2C but applied to 3 components. */
__global__ void k_divN2C(double* __restrict__ divC,
                         const double* __restrict__ vecXN,
                         const double* __restrict__ vecYN,
                         const double* __restrict__ vecZN,
                         int nxc, int nyc, int nzc,
                         int nyn, int nzn,
                         double invdx, double invdy, double invdz)
{
    int i = blockIdx.x * BX + threadIdx.x + 1;
    int j = blockIdx.y * BY + threadIdx.y + 1;
    int k = blockIdx.z * BZ + threadIdx.z + 1;
    if (i > nxc - 2 || j > nyc - 2 || k > nzc - 2) return;

    #define N_(arr, ii, jj, kk) arr[IDX(ii, jj, kk, nyn, nzn)]
    // X-component derivative (d/dx)
    double compX = 0.25 * invdx * (
        (N_(vecXN,i+1,j  ,k  ) - N_(vecXN,i,j  ,k  )) +
        (N_(vecXN,i+1,j  ,k+1) - N_(vecXN,i,j  ,k+1)) +
        (N_(vecXN,i+1,j+1,k  ) - N_(vecXN,i,j+1,k  )) +
        (N_(vecXN,i+1,j+1,k+1) - N_(vecXN,i,j+1,k+1)));
    // Y-component derivative (d/dy)
    double compY = 0.25 * invdy * (
        (N_(vecYN,i  ,j+1,k  ) - N_(vecYN,i  ,j,k  )) +
        (N_(vecYN,i  ,j+1,k+1) - N_(vecYN,i  ,j,k+1)) +
        (N_(vecYN,i+1,j+1,k  ) - N_(vecYN,i+1,j,k  )) +
        (N_(vecYN,i+1,j+1,k+1) - N_(vecYN,i+1,j,k+1)));
    // Z-component derivative (d/dz)
    double compZ = 0.25 * invdz * (
        (N_(vecZN,i  ,j  ,k+1) - N_(vecZN,i  ,j  ,k)) +
        (N_(vecZN,i+1,j  ,k+1) - N_(vecZN,i+1,j  ,k)) +
        (N_(vecZN,i  ,j+1,k+1) - N_(vecZN,i  ,j+1,k)) +
        (N_(vecZN,i+1,j+1,k+1) - N_(vecZN,i+1,j+1,k)));

    divC[IDX(i, j, k, nyc, nzc)] = compX + compY + compZ;
    #undef N_
}

/** divC2N: divergence center→node.  Reads center[i-1..i] for node output. */
__global__ void k_divC2N(double* __restrict__ divN,
                         const double* __restrict__ vecXC,
                         const double* __restrict__ vecYC,
                         const double* __restrict__ vecZC,
                         int nxn, int nyn, int nzn,
                         int nyc, int nzc,
                         double invdx, double invdy, double invdz)
{
    int i = blockIdx.x * BX + threadIdx.x + 1;
    int j = blockIdx.y * BY + threadIdx.y + 1;
    int k = blockIdx.z * BZ + threadIdx.z + 1;
    if (i > nxn - 2 || j > nyn - 2 || k > nzn - 2) return;

    #define C_(arr, ii, jj, kk) arr[IDX(ii, jj, kk, nyc, nzc)]
    double compX = 0.25 * invdx * (
        (C_(vecXC,i  ,j  ,k  ) - C_(vecXC,i-1,j  ,k  )) +
        (C_(vecXC,i  ,j  ,k-1) - C_(vecXC,i-1,j  ,k-1)) +
        (C_(vecXC,i  ,j-1,k  ) - C_(vecXC,i-1,j-1,k  )) +
        (C_(vecXC,i  ,j-1,k-1) - C_(vecXC,i-1,j-1,k-1)));
    double compY = 0.25 * invdy * (
        (C_(vecYC,i  ,j  ,k  ) - C_(vecYC,i  ,j-1,k  )) +
        (C_(vecYC,i  ,j  ,k-1) - C_(vecYC,i  ,j-1,k-1)) +
        (C_(vecYC,i-1,j  ,k  ) - C_(vecYC,i-1,j-1,k  )) +
        (C_(vecYC,i-1,j  ,k-1) - C_(vecYC,i-1,j-1,k-1)));
    double compZ = 0.25 * invdz * (
        (C_(vecZC,i  ,j  ,k  ) - C_(vecZC,i  ,j  ,k-1)) +
        (C_(vecZC,i-1,j  ,k  ) - C_(vecZC,i-1,j  ,k-1)) +
        (C_(vecZC,i  ,j-1,k  ) - C_(vecZC,i  ,j-1,k-1)) +
        (C_(vecZC,i-1,j-1,k  ) - C_(vecZC,i-1,j-1,k-1)));

    divN[IDX(i, j, k, nyn, nzn)] = compX + compY + compZ;
    #undef C_
}

// =========================================================================
//  Curl kernels
// =========================================================================

__global__ void k_curlC2N(double* __restrict__ curlXN,
                          double* __restrict__ curlYN,
                          double* __restrict__ curlZN,
                          const double* __restrict__ vecXC,
                          const double* __restrict__ vecYC,
                          const double* __restrict__ vecZC,
                          int nxn, int nyn, int nzn,
                          int nyc, int nzc,
                          double invdx, double invdy, double invdz)
{
    int i = blockIdx.x * BX + threadIdx.x + 1;
    int j = blockIdx.y * BY + threadIdx.y + 1;
    int k = blockIdx.z * BZ + threadIdx.z + 1;
    if (i > nxn - 2 || j > nyn - 2 || k > nzn - 2) return;

    #define C_(arr, ii, jj, kk) arr[IDX(ii, jj, kk, nyc, nzc)]
    // curl_X = dZdy - dYdz
    double compZDY = 0.25 * invdy * (
        (C_(vecZC,i  ,j  ,k  ) - C_(vecZC,i  ,j-1,k  )) +
        (C_(vecZC,i  ,j  ,k-1) - C_(vecZC,i  ,j-1,k-1)) +
        (C_(vecZC,i-1,j  ,k  ) - C_(vecZC,i-1,j-1,k  )) +
        (C_(vecZC,i-1,j  ,k-1) - C_(vecZC,i-1,j-1,k-1)));
    double compYDZ = 0.25 * invdz * (
        (C_(vecYC,i  ,j  ,k  ) - C_(vecYC,i  ,j  ,k-1)) +
        (C_(vecYC,i-1,j  ,k  ) - C_(vecYC,i-1,j  ,k-1)) +
        (C_(vecYC,i  ,j-1,k  ) - C_(vecYC,i  ,j-1,k-1)) +
        (C_(vecYC,i-1,j-1,k  ) - C_(vecYC,i-1,j-1,k-1)));

    // curl_Y = dXdz - dZdx
    double compXDZ = 0.25 * invdz * (
        (C_(vecXC,i  ,j  ,k  ) - C_(vecXC,i  ,j  ,k-1)) +
        (C_(vecXC,i-1,j  ,k  ) - C_(vecXC,i-1,j  ,k-1)) +
        (C_(vecXC,i  ,j-1,k  ) - C_(vecXC,i  ,j-1,k-1)) +
        (C_(vecXC,i-1,j-1,k  ) - C_(vecXC,i-1,j-1,k-1)));
    double compZDX = 0.25 * invdx * (
        (C_(vecZC,i  ,j  ,k  ) - C_(vecZC,i-1,j  ,k  )) +
        (C_(vecZC,i  ,j  ,k-1) - C_(vecZC,i-1,j  ,k-1)) +
        (C_(vecZC,i  ,j-1,k  ) - C_(vecZC,i-1,j-1,k  )) +
        (C_(vecZC,i  ,j-1,k-1) - C_(vecZC,i-1,j-1,k-1)));

    // curl_Z = dYdx - dXdy
    double compYDX = 0.25 * invdx * (
        (C_(vecYC,i  ,j  ,k  ) - C_(vecYC,i-1,j  ,k  )) +
        (C_(vecYC,i  ,j  ,k-1) - C_(vecYC,i-1,j  ,k-1)) +
        (C_(vecYC,i  ,j-1,k  ) - C_(vecYC,i-1,j-1,k  )) +
        (C_(vecYC,i  ,j-1,k-1) - C_(vecYC,i-1,j-1,k-1)));
    double compXDY = 0.25 * invdy * (
        (C_(vecXC,i  ,j  ,k  ) - C_(vecXC,i  ,j-1,k  )) +
        (C_(vecXC,i  ,j  ,k-1) - C_(vecXC,i  ,j-1,k-1)) +
        (C_(vecXC,i-1,j  ,k  ) - C_(vecXC,i-1,j-1,k  )) +
        (C_(vecXC,i-1,j  ,k-1) - C_(vecXC,i-1,j-1,k-1)));

    int nidx = IDX(i, j, k, nyn, nzn);
    curlXN[nidx] = compZDY - compYDZ;
    curlYN[nidx] = compXDZ - compZDX;
    curlZN[nidx] = compYDX - compXDY;
    #undef C_
}

__global__ void k_curlN2C(double* __restrict__ curlXC,
                          double* __restrict__ curlYC,
                          double* __restrict__ curlZC,
                          const double* __restrict__ vecXN,
                          const double* __restrict__ vecYN,
                          const double* __restrict__ vecZN,
                          int nxc, int nyc, int nzc,
                          int nyn, int nzn,
                          double invdx, double invdy, double invdz)
{
    int i = blockIdx.x * BX + threadIdx.x + 1;
    int j = blockIdx.y * BY + threadIdx.y + 1;
    int k = blockIdx.z * BZ + threadIdx.z + 1;
    if (i > nxc - 2 || j > nyc - 2 || k > nzc - 2) return;

    #define N_(arr, ii, jj, kk) arr[IDX(ii, jj, kk, nyn, nzn)]
    // curl_X = dZdy - dYdz
    double compZDY = 0.25 * invdy * (
        (N_(vecZN,i  ,j+1,k  ) - N_(vecZN,i  ,j,k  )) +
        (N_(vecZN,i  ,j+1,k+1) - N_(vecZN,i  ,j,k+1)) +
        (N_(vecZN,i+1,j+1,k  ) - N_(vecZN,i+1,j,k  )) +
        (N_(vecZN,i+1,j+1,k+1) - N_(vecZN,i+1,j,k+1)));
    double compYDZ = 0.25 * invdz * (
        (N_(vecYN,i  ,j  ,k+1) - N_(vecYN,i  ,j  ,k)) +
        (N_(vecYN,i+1,j  ,k+1) - N_(vecYN,i+1,j  ,k)) +
        (N_(vecYN,i  ,j+1,k+1) - N_(vecYN,i  ,j+1,k)) +
        (N_(vecYN,i+1,j+1,k+1) - N_(vecYN,i+1,j+1,k)));

    // curl_Y = dXdz - dZdx
    double compXDZ = 0.25 * invdz * (
        (N_(vecXN,i  ,j  ,k+1) - N_(vecXN,i  ,j  ,k)) +
        (N_(vecXN,i+1,j  ,k+1) - N_(vecXN,i+1,j  ,k)) +
        (N_(vecXN,i  ,j+1,k+1) - N_(vecXN,i  ,j+1,k)) +
        (N_(vecXN,i+1,j+1,k+1) - N_(vecXN,i+1,j+1,k)));
    double compZDX = 0.25 * invdx * (
        (N_(vecZN,i+1,j  ,k  ) - N_(vecZN,i,j  ,k  )) +
        (N_(vecZN,i+1,j  ,k+1) - N_(vecZN,i,j  ,k+1)) +
        (N_(vecZN,i+1,j+1,k  ) - N_(vecZN,i,j+1,k  )) +
        (N_(vecZN,i+1,j+1,k+1) - N_(vecZN,i,j+1,k+1)));

    // curl_Z = dYdx - dXdy
    double compYDX = 0.25 * invdx * (
        (N_(vecYN,i+1,j  ,k  ) - N_(vecYN,i,j  ,k  )) +
        (N_(vecYN,i+1,j  ,k+1) - N_(vecYN,i,j  ,k+1)) +
        (N_(vecYN,i+1,j+1,k  ) - N_(vecYN,i,j+1,k  )) +
        (N_(vecYN,i+1,j+1,k+1) - N_(vecYN,i,j+1,k+1)));
    double compXDY = 0.25 * invdy * (
        (N_(vecXN,i  ,j+1,k  ) - N_(vecXN,i  ,j,k  )) +
        (N_(vecXN,i  ,j+1,k+1) - N_(vecXN,i  ,j,k+1)) +
        (N_(vecXN,i+1,j+1,k  ) - N_(vecXN,i+1,j,k  )) +
        (N_(vecXN,i+1,j+1,k+1) - N_(vecXN,i+1,j,k+1)));

    int cidx = IDX(i, j, k, nyc, nzc);
    curlXC[cidx] = compZDY - compYDZ;
    curlYC[cidx] = compXDZ - compZDX;
    curlZC[cidx] = compYDX - compXDY;
    #undef N_
}

// =========================================================================
//  Interpolation kernels
// =========================================================================

__global__ void k_interpC2N(double* __restrict__ fieldN,
                            const double* __restrict__ fieldC,
                            int nxn, int nyn, int nzn,
                            int nyc, int nzc)
{
    int i = blockIdx.x * BX + threadIdx.x + 1;
    int j = blockIdx.y * BY + threadIdx.y + 1;
    int k = blockIdx.z * BZ + threadIdx.z + 1;
    if (i > nxn - 2 || j > nyn - 2 || k > nzn - 2) return;

    #define C_(ii, jj, kk) fieldC[IDX(ii, jj, kk, nyc, nzc)]
    fieldN[IDX(i, j, k, nyn, nzn)] = 0.125 * (
        C_(i,j,k) + C_(i-1,j,k) + C_(i,j-1,k) + C_(i,j,k-1) +
        C_(i-1,j-1,k) + C_(i-1,j,k-1) + C_(i,j-1,k-1) + C_(i-1,j-1,k-1));
    #undef C_
}

__global__ void k_interpN2C(double* __restrict__ fieldC,
                            const double* __restrict__ fieldN,
                            int nxc, int nyc, int nzc,
                            int nyn, int nzn)
{
    int i = blockIdx.x * BX + threadIdx.x + 1;
    int j = blockIdx.y * BY + threadIdx.y + 1;
    int k = blockIdx.z * BZ + threadIdx.z + 1;
    if (i > nxc - 2 || j > nyc - 2 || k > nzc - 2) return;

    #define N_(ii, jj, kk) fieldN[IDX(ii, jj, kk, nyn, nzn)]
    fieldC[IDX(i, j, k, nyc, nzc)] = 0.125 * (
        N_(i,j,k) + N_(i+1,j,k) + N_(i,j+1,k) + N_(i,j,k+1) +
        N_(i+1,j+1,k) + N_(i+1,j,k+1) + N_(i,j+1,k+1) + N_(i+1,j+1,k+1));
    #undef N_
}

// =========================================================================
//  Symmetric tensor divergence kernel
// =========================================================================

/**
 * divSymmTensorN2C:
 *   divCX = d(pXX)/dx + d(pXY)/dy + d(pXZ)/dz
 *   divCY = d(pXY)/dx + d(pYY)/dy + d(pYZ)/dz
 *   divCZ = d(pXZ)/dx + d(pYZ)/dy + d(pZZ)/dz
 *
 * All tensor components are node-based [nxn][nyn][nzn] (for a given species).
 * Output on center grid [nxc][nyc][nzc].
 * Uses gradN2C stencil: reads node [i..i+1]×[j..j+1]×[k..k+1].
 */
__global__ void k_divSymmTensorN2C(
    double* __restrict__ divCX,
    double* __restrict__ divCY,
    double* __restrict__ divCZ,
    const double* __restrict__ pXX, const double* __restrict__ pXY,
    const double* __restrict__ pXZ, const double* __restrict__ pYY,
    const double* __restrict__ pYZ, const double* __restrict__ pZZ,
    int nxc, int nyc, int nzc,
    int nyn, int nzn,
    double invdx, double invdy, double invdz)
{
    int i = blockIdx.x * BX + threadIdx.x + 1;
    int j = blockIdx.y * BY + threadIdx.y + 1;
    int k = blockIdx.z * BZ + threadIdx.z + 1;
    if (i > nxc - 2 || j > nyc - 2 || k > nzc - 2) return;

    // Helper: derivative by x (gradN2C-x stencil)
    #define N_(arr, ii, jj, kk) arr[IDX(ii, jj, kk, nyn, nzn)]
    #define DX(arr) (0.25 * invdx * (\
        (N_(arr,i+1,j,k) - N_(arr,i,j,k)) + (N_(arr,i+1,j,k+1) - N_(arr,i,j,k+1)) +\
        (N_(arr,i+1,j+1,k) - N_(arr,i,j+1,k)) + (N_(arr,i+1,j+1,k+1) - N_(arr,i,j+1,k+1))))
    #define DY(arr) (0.25 * invdy * (\
        (N_(arr,i,j+1,k) - N_(arr,i,j,k)) + (N_(arr,i,j+1,k+1) - N_(arr,i,j,k+1)) +\
        (N_(arr,i+1,j+1,k) - N_(arr,i+1,j,k)) + (N_(arr,i+1,j+1,k+1) - N_(arr,i+1,j,k+1))))
    #define DZ(arr) (0.25 * invdz * (\
        (N_(arr,i,j,k+1) - N_(arr,i,j,k)) + (N_(arr,i+1,j,k+1) - N_(arr,i+1,j,k)) +\
        (N_(arr,i,j+1,k+1) - N_(arr,i,j+1,k)) + (N_(arr,i+1,j+1,k+1) - N_(arr,i+1,j+1,k))))

    int cidx = IDX(i, j, k, nyc, nzc);
    divCX[cidx] = DX(pXX) + DY(pXY) + DZ(pXZ);
    divCY[cidx] = DX(pXY) + DY(pYY) + DZ(pYZ);
    divCZ[cidx] = DX(pXZ) + DY(pYZ) + DZ(pZZ);

    #undef DZ
    #undef DY
    #undef DX
    #undef N_
}

// =========================================================================
//  Smooth (box stencil) kernel
// =========================================================================

__global__ void k_smoothStep(double* __restrict__ out,
                             const double* __restrict__ in,
                             int nx, int ny, int nz,
                             double alpha, double beta3D)
{
    int i = blockIdx.x * BX + threadIdx.x + 1;
    int j = blockIdx.y * BY + threadIdx.y + 1;
    int k = blockIdx.z * BZ + threadIdx.z + 1;
    if (i > nx - 2 || j > ny - 2 || k > nz - 2) return;

    #define IN(ii, jj, kk) in[IDX(ii, jj, kk, ny, nz)]
    out[IDX(i, j, k, ny, nz)] = alpha * IN(i, j, k) + beta3D * (
        IN(i-1,j,k) + IN(i+1,j,k) +
        IN(i,j-1,k) + IN(i,j+1,k) +
        IN(i,j,k-1) + IN(i,j,k+1));
    #undef IN
}

// =========================================================================
//  Host wrappers
// =========================================================================

void gpuGradC2N(double* gradXN, double* gradYN, double* gradZN,
                const double* scFieldC,
                int nxn, int nyn, int nzn,
                double invdx, double invdy, double invdz,
                cudaStream_t stream)
{
    int nyc = nyn - 1, nzc = nzn - 1;
    dim3 grid = stencilGrid(nxn, nyn, nzn);
    dim3 block(BX, BY, BZ);
    k_gradC2N<<<grid, block, 0, stream>>>(
        gradXN, gradYN, gradZN, scFieldC,
        nxn, nyn, nzn, nyc, nzc, invdx, invdy, invdz);
    cudaErrChk(cudaGetLastError());
}

void gpuGradN2C(double* gradXC, double* gradYC, double* gradZC,
                const double* scFieldN,
                int nxc, int nyc, int nzc,
                double invdx, double invdy, double invdz,
                cudaStream_t stream)
{
    int nyn = nyc + 1, nzn = nzc + 1;
    dim3 grid = stencilGrid(nxc, nyc, nzc);
    dim3 block(BX, BY, BZ);
    k_gradN2C<<<grid, block, 0, stream>>>(
        gradXC, gradYC, gradZC, scFieldN,
        nxc, nyc, nzc, nyn, nzn, invdx, invdy, invdz);
    cudaErrChk(cudaGetLastError());
}

void gpuDivN2C(double* divC,
               const double* vecXN, const double* vecYN, const double* vecZN,
               int nxc, int nyc, int nzc,
               double invdx, double invdy, double invdz,
               cudaStream_t stream)
{
    int nyn = nyc + 1, nzn = nzc + 1;
    dim3 grid = stencilGrid(nxc, nyc, nzc);
    dim3 block(BX, BY, BZ);
    k_divN2C<<<grid, block, 0, stream>>>(
        divC, vecXN, vecYN, vecZN,
        nxc, nyc, nzc, nyn, nzn, invdx, invdy, invdz);
    cudaErrChk(cudaGetLastError());
}

void gpuDivC2N(double* divN,
               const double* vecXC, const double* vecYC, const double* vecZC,
               int nxn, int nyn, int nzn,
               double invdx, double invdy, double invdz,
               cudaStream_t stream)
{
    int nyc = nyn - 1, nzc = nzn - 1;
    dim3 grid = stencilGrid(nxn, nyn, nzn);
    dim3 block(BX, BY, BZ);
    k_divC2N<<<grid, block, 0, stream>>>(
        divN, vecXC, vecYC, vecZC,
        nxn, nyn, nzn, nyc, nzc, invdx, invdy, invdz);
    cudaErrChk(cudaGetLastError());
}

void gpuCurlC2N(double* curlXN, double* curlYN, double* curlZN,
                const double* vecXC, const double* vecYC, const double* vecZC,
                int nxn, int nyn, int nzn,
                double invdx, double invdy, double invdz,
                cudaStream_t stream)
{
    int nyc = nyn - 1, nzc = nzn - 1;
    dim3 grid = stencilGrid(nxn, nyn, nzn);
    dim3 block(BX, BY, BZ);
    k_curlC2N<<<grid, block, 0, stream>>>(
        curlXN, curlYN, curlZN, vecXC, vecYC, vecZC,
        nxn, nyn, nzn, nyc, nzc, invdx, invdy, invdz);
    cudaErrChk(cudaGetLastError());
}

void gpuCurlN2C(double* curlXC, double* curlYC, double* curlZC,
                const double* vecXN, const double* vecYN, const double* vecZN,
                int nxc, int nyc, int nzc,
                double invdx, double invdy, double invdz,
                cudaStream_t stream)
{
    int nyn = nyc + 1, nzn = nzc + 1;
    dim3 grid = stencilGrid(nxc, nyc, nzc);
    dim3 block(BX, BY, BZ);
    k_curlN2C<<<grid, block, 0, stream>>>(
        curlXC, curlYC, curlZC, vecXN, vecYN, vecZN,
        nxc, nyc, nzc, nyn, nzn, invdx, invdy, invdz);
    cudaErrChk(cudaGetLastError());
}

void gpuInterpC2N(double* fieldN, const double* fieldC,
                  int nxn, int nyn, int nzn,
                  cudaStream_t stream)
{
    int nyc = nyn - 1, nzc = nzn - 1;
    dim3 grid = stencilGrid(nxn, nyn, nzn);
    dim3 block(BX, BY, BZ);
    k_interpC2N<<<grid, block, 0, stream>>>(
        fieldN, fieldC, nxn, nyn, nzn, nyc, nzc);
    cudaErrChk(cudaGetLastError());
}

void gpuInterpN2C(double* fieldC, const double* fieldN,
                  int nxc, int nyc, int nzc,
                  cudaStream_t stream)
{
    int nyn = nyc + 1, nzn = nzc + 1;
    dim3 grid = stencilGrid(nxc, nyc, nzc);
    dim3 block(BX, BY, BZ);
    k_interpN2C<<<grid, block, 0, stream>>>(
        fieldC, fieldN, nxc, nyc, nzc, nyn, nzn);
    cudaErrChk(cudaGetLastError());
}

void gpuDivSymmTensorN2C(double* divCX, double* divCY, double* divCZ,
                         const double* pXX, const double* pXY, const double* pXZ,
                         const double* pYY, const double* pYZ, const double* pZZ,
                         int nxc, int nyc, int nzc,
                         double invdx, double invdy, double invdz,
                         cudaStream_t stream)
{
    int nyn = nyc + 1, nzn = nzc + 1;
    dim3 grid = stencilGrid(nxc, nyc, nzc);
    dim3 block(BX, BY, BZ);
    k_divSymmTensorN2C<<<grid, block, 0, stream>>>(
        divCX, divCY, divCZ, pXX, pXY, pXZ, pYY, pYZ, pZZ,
        nxc, nyc, nzc, nyn, nzn, invdx, invdy, invdz);
    cudaErrChk(cudaGetLastError());
}

void gpuSmoothStep(double* out, const double* in,
                   int nx, int ny, int nz,
                   double alpha, double beta3D,
                   cudaStream_t stream)
{
    dim3 grid = stencilGrid(nx, ny, nz);
    dim3 block(BX, BY, BZ);
    k_smoothStep<<<grid, block, 0, stream>>>(out, in, nx, ny, nz, alpha, beta3D);
    cudaErrChk(cudaGetLastError());
}

// =========================================================================
//  HALO_OVERLAP: interior / boundary stencil variants
//
//  Each centre→node stencil (divC2N, gradC2N, interpC2N) reads
//  centre[i-1..i][j-1..j][k-1..k] for node output at [i][j][k].
//    Interior [2..n-3]: all reads fall in centre interior — no ghost deps.
//    Boundary [1, n-2]: at least one read hits a ghost centre cell.
//
//  smoothStep reads [i±1][j±1][k±1]: same interior range [2..n-3].
//
//  Device helper functions are used to avoid duplicating computation code
//  across the original, interior, and boundary kernel variants.
// =========================================================================
#ifdef HALO_OVERLAP

// Helper: grid dims covering a ranged sub-volume [lo..hi] per axis.
static inline dim3 rangedGrid(int iLo, int iHi, int jLo, int jHi, int kLo, int kHi)
{
    return dim3(((iHi - iLo + 1) + BX - 1) / BX,
                ((jHi - jLo + 1) + BY - 1) / BY,
                ((kHi - kLo + 1) + BZ - 1) / BZ);
}

// -----------------------------------------------------------------
//  divC2N  (centre → node divergence)
// -----------------------------------------------------------------
__device__ __forceinline__
void d_divC2N(double* __restrict__ divN,
              const double* __restrict__ vXC,
              const double* __restrict__ vYC,
              const double* __restrict__ vZC,
              int i, int j, int k,
              int nyn, int nzn, int nyc, int nzc,
              double invdx, double invdy, double invdz)
{
    #define C_D(arr, ii, jj, kk) arr[IDX(ii, jj, kk, nyc, nzc)]
    double cX = 0.25 * invdx * (
        (C_D(vXC,i  ,j  ,k  ) - C_D(vXC,i-1,j  ,k  )) +
        (C_D(vXC,i  ,j  ,k-1) - C_D(vXC,i-1,j  ,k-1)) +
        (C_D(vXC,i  ,j-1,k  ) - C_D(vXC,i-1,j-1,k  )) +
        (C_D(vXC,i  ,j-1,k-1) - C_D(vXC,i-1,j-1,k-1)));
    double cY = 0.25 * invdy * (
        (C_D(vYC,i  ,j  ,k  ) - C_D(vYC,i  ,j-1,k  )) +
        (C_D(vYC,i  ,j  ,k-1) - C_D(vYC,i  ,j-1,k-1)) +
        (C_D(vYC,i-1,j  ,k  ) - C_D(vYC,i-1,j-1,k  )) +
        (C_D(vYC,i-1,j  ,k-1) - C_D(vYC,i-1,j-1,k-1)));
    double cZ = 0.25 * invdz * (
        (C_D(vZC,i  ,j  ,k  ) - C_D(vZC,i  ,j  ,k-1)) +
        (C_D(vZC,i-1,j  ,k  ) - C_D(vZC,i-1,j  ,k-1)) +
        (C_D(vZC,i  ,j-1,k  ) - C_D(vZC,i  ,j-1,k-1)) +
        (C_D(vZC,i-1,j-1,k  ) - C_D(vZC,i-1,j-1,k-1)));
    divN[IDX(i, j, k, nyn, nzn)] = cX + cY + cZ;
    #undef C_D
}

__global__ void k_divC2N_interior(
    double* __restrict__ divN,
    const double* __restrict__ vXC, const double* __restrict__ vYC, const double* __restrict__ vZC,
    int nxn, int nyn, int nzn, int nyc, int nzc,
    double invdx, double invdy, double invdz,
    int iLo, int iHi, int jLo, int jHi, int kLo, int kHi)
{
    int i = blockIdx.x * BX + threadIdx.x + iLo;
    int j = blockIdx.y * BY + threadIdx.y + jLo;
    int k = blockIdx.z * BZ + threadIdx.z + kLo;
    if (i > iHi || j > jHi || k > kHi) return;
    d_divC2N(divN, vXC, vYC, vZC, i, j, k, nyn, nzn, nyc, nzc, invdx, invdy, invdz);
}

__global__ void k_divC2N_boundary(
    double* __restrict__ divN,
    const double* __restrict__ vXC, const double* __restrict__ vYC, const double* __restrict__ vZC,
    int nxn, int nyn, int nzn, int nyc, int nzc,
    double invdx, double invdy, double invdz,
    int sILo, int sIHi, int sJLo, int sJHi, int sKLo, int sKHi)
{
    int i = blockIdx.x * BX + threadIdx.x + 1;
    int j = blockIdx.y * BY + threadIdx.y + 1;
    int k = blockIdx.z * BZ + threadIdx.z + 1;
    if (i > nxn - 2 || j > nyn - 2 || k > nzn - 2) return;
    if (i >= sILo && i <= sIHi && j >= sJLo && j <= sJHi && k >= sKLo && k <= sKHi) return;
    d_divC2N(divN, vXC, vYC, vZC, i, j, k, nyn, nzn, nyc, nzc, invdx, invdy, invdz);
}

void gpuDivC2N_interior(double* divN,
                        const double* vXC, const double* vYC, const double* vZC,
                        int nxn, int nyn, int nzn,
                        double invdx, double invdy, double invdz,
                        cudaStream_t stream)
{
    int iLo=2, iHi=nxn-3, jLo=2, jHi=nyn-3, kLo=2, kHi=nzn-3;
    if (iLo>iHi || jLo>jHi || kLo>kHi) return;
    int nyc=nyn-1, nzc=nzn-1;
    dim3 grid = rangedGrid(iLo,iHi,jLo,jHi,kLo,kHi);
    dim3 block(BX,BY,BZ);
    k_divC2N_interior<<<grid,block,0,stream>>>(divN,vXC,vYC,vZC,nxn,nyn,nzn,nyc,nzc,invdx,invdy,invdz,iLo,iHi,jLo,jHi,kLo,kHi);
    cudaErrChk(cudaGetLastError());
}

void gpuDivC2N_boundary(double* divN,
                        const double* vXC, const double* vYC, const double* vZC,
                        int nxn, int nyn, int nzn,
                        double invdx, double invdy, double invdz,
                        cudaStream_t stream)
{
    int nyc=nyn-1, nzc=nzn-1;
    dim3 grid = stencilGrid(nxn,nyn,nzn);
    dim3 block(BX,BY,BZ);
    k_divC2N_boundary<<<grid,block,0,stream>>>(divN,vXC,vYC,vZC,nxn,nyn,nzn,nyc,nzc,invdx,invdy,invdz,2,nxn-3,2,nyn-3,2,nzn-3);
    cudaErrChk(cudaGetLastError());
}

// -----------------------------------------------------------------
//  gradC2N  (centre → node gradient)
// -----------------------------------------------------------------
__device__ __forceinline__
void d_gradC2N(double* __restrict__ gXN, double* __restrict__ gYN, double* __restrict__ gZN,
               const double* __restrict__ C,
               int i, int j, int k,
               int nyn, int nzn, int nyc, int nzc,
               double invdx, double invdy, double invdz)
{
    #define CG(ii,jj,kk) C[IDX(ii,jj,kk,nyc,nzc)]
    double c000=CG(i-1,j-1,k-1), c001=CG(i-1,j-1,k), c010=CG(i-1,j,k-1), c011=CG(i-1,j,k);
    double c100=CG(i,  j-1,k-1), c101=CG(i,  j-1,k), c110=CG(i,  j,k-1), c111=CG(i,  j,k);
    int nidx = IDX(i,j,k,nyn,nzn);
    gXN[nidx] = 0.25*invdx*((c111-c011)+(c110-c010)+(c101-c001)+(c100-c000));
    gYN[nidx] = 0.25*invdy*((c111-c101)+(c110-c100)+(c011-c001)+(c010-c000));
    gZN[nidx] = 0.25*invdz*((c111-c110)+(c101-c100)+(c011-c010)+(c001-c000));
    #undef CG
}

__global__ void k_gradC2N_interior(
    double* __restrict__ gXN, double* __restrict__ gYN, double* __restrict__ gZN,
    const double* __restrict__ C,
    int nxn, int nyn, int nzn, int nyc, int nzc,
    double invdx, double invdy, double invdz,
    int iLo, int iHi, int jLo, int jHi, int kLo, int kHi)
{
    int i=blockIdx.x*BX+threadIdx.x+iLo, j=blockIdx.y*BY+threadIdx.y+jLo, k=blockIdx.z*BZ+threadIdx.z+kLo;
    if (i>iHi||j>jHi||k>kHi) return;
    d_gradC2N(gXN,gYN,gZN,C,i,j,k,nyn,nzn,nyc,nzc,invdx,invdy,invdz);
}

__global__ void k_gradC2N_boundary(
    double* __restrict__ gXN, double* __restrict__ gYN, double* __restrict__ gZN,
    const double* __restrict__ C,
    int nxn, int nyn, int nzn, int nyc, int nzc,
    double invdx, double invdy, double invdz,
    int sILo, int sIHi, int sJLo, int sJHi, int sKLo, int sKHi)
{
    int i=blockIdx.x*BX+threadIdx.x+1, j=blockIdx.y*BY+threadIdx.y+1, k=blockIdx.z*BZ+threadIdx.z+1;
    if (i>nxn-2||j>nyn-2||k>nzn-2) return;
    if (i>=sILo&&i<=sIHi&&j>=sJLo&&j<=sJHi&&k>=sKLo&&k<=sKHi) return;
    d_gradC2N(gXN,gYN,gZN,C,i,j,k,nyn,nzn,nyc,nzc,invdx,invdy,invdz);
}

void gpuGradC2N_interior(double* gXN, double* gYN, double* gZN,
                         const double* C,
                         int nxn, int nyn, int nzn,
                         double invdx, double invdy, double invdz,
                         cudaStream_t stream)
{
    int iLo=2,iHi=nxn-3,jLo=2,jHi=nyn-3,kLo=2,kHi=nzn-3;
    if (iLo>iHi||jLo>jHi||kLo>kHi) return;
    int nyc=nyn-1,nzc=nzn-1;
    dim3 grid=rangedGrid(iLo,iHi,jLo,jHi,kLo,kHi); dim3 block(BX,BY,BZ);
    k_gradC2N_interior<<<grid,block,0,stream>>>(gXN,gYN,gZN,C,nxn,nyn,nzn,nyc,nzc,invdx,invdy,invdz,iLo,iHi,jLo,jHi,kLo,kHi);
    cudaErrChk(cudaGetLastError());
}

void gpuGradC2N_boundary(double* gXN, double* gYN, double* gZN,
                         const double* C,
                         int nxn, int nyn, int nzn,
                         double invdx, double invdy, double invdz,
                         cudaStream_t stream)
{
    int nyc=nyn-1,nzc=nzn-1;
    dim3 grid=stencilGrid(nxn,nyn,nzn); dim3 block(BX,BY,BZ);
    k_gradC2N_boundary<<<grid,block,0,stream>>>(gXN,gYN,gZN,C,nxn,nyn,nzn,nyc,nzc,invdx,invdy,invdz,2,nxn-3,2,nyn-3,2,nzn-3);
    cudaErrChk(cudaGetLastError());
}

// -----------------------------------------------------------------
//  interpC2N  (centre → node interpolation)
// -----------------------------------------------------------------
__device__ __forceinline__
void d_interpC2N(double* __restrict__ fN, const double* __restrict__ fC,
                 int i, int j, int k, int nyn, int nzn, int nyc, int nzc)
{
    #define CI(ii,jj,kk) fC[IDX(ii,jj,kk,nyc,nzc)]
    fN[IDX(i,j,k,nyn,nzn)] = 0.125*(CI(i,j,k)+CI(i-1,j,k)+CI(i,j-1,k)+CI(i,j,k-1)
                                     +CI(i-1,j-1,k)+CI(i-1,j,k-1)+CI(i,j-1,k-1)+CI(i-1,j-1,k-1));
    #undef CI
}

__global__ void k_interpC2N_interior(
    double* __restrict__ fN, const double* __restrict__ fC,
    int nxn, int nyn, int nzn, int nyc, int nzc,
    int iLo, int iHi, int jLo, int jHi, int kLo, int kHi)
{
    int i=blockIdx.x*BX+threadIdx.x+iLo, j=blockIdx.y*BY+threadIdx.y+jLo, k=blockIdx.z*BZ+threadIdx.z+kLo;
    if (i>iHi||j>jHi||k>kHi) return;
    d_interpC2N(fN,fC,i,j,k,nyn,nzn,nyc,nzc);
}

__global__ void k_interpC2N_boundary(
    double* __restrict__ fN, const double* __restrict__ fC,
    int nxn, int nyn, int nzn, int nyc, int nzc,
    int sILo, int sIHi, int sJLo, int sJHi, int sKLo, int sKHi)
{
    int i=blockIdx.x*BX+threadIdx.x+1, j=blockIdx.y*BY+threadIdx.y+1, k=blockIdx.z*BZ+threadIdx.z+1;
    if (i>nxn-2||j>nyn-2||k>nzn-2) return;
    if (i>=sILo&&i<=sIHi&&j>=sJLo&&j<=sJHi&&k>=sKLo&&k<=sKHi) return;
    d_interpC2N(fN,fC,i,j,k,nyn,nzn,nyc,nzc);
}

void gpuInterpC2N_interior(double* fN, const double* fC,
                           int nxn, int nyn, int nzn,
                           cudaStream_t stream)
{
    int iLo=2,iHi=nxn-3,jLo=2,jHi=nyn-3,kLo=2,kHi=nzn-3;
    if (iLo>iHi||jLo>jHi||kLo>kHi) return;
    int nyc=nyn-1,nzc=nzn-1;
    dim3 grid=rangedGrid(iLo,iHi,jLo,jHi,kLo,kHi); dim3 block(BX,BY,BZ);
    k_interpC2N_interior<<<grid,block,0,stream>>>(fN,fC,nxn,nyn,nzn,nyc,nzc,iLo,iHi,jLo,jHi,kLo,kHi);
    cudaErrChk(cudaGetLastError());
}

void gpuInterpC2N_boundary(double* fN, const double* fC,
                           int nxn, int nyn, int nzn,
                           cudaStream_t stream)
{
    int nyc=nyn-1,nzc=nzn-1;
    dim3 grid=stencilGrid(nxn,nyn,nzn); dim3 block(BX,BY,BZ);
    k_interpC2N_boundary<<<grid,block,0,stream>>>(fN,fC,nxn,nyn,nzn,nyc,nzc,2,nxn-3,2,nyn-3,2,nzn-3);
    cudaErrChk(cudaGetLastError());
}

// -----------------------------------------------------------------
//  smoothStep  (6-point box smooth)
// -----------------------------------------------------------------
__device__ __forceinline__
void d_smoothStep(double* __restrict__ out, const double* __restrict__ in,
                  int i, int j, int k, int ny, int nz,
                  double alpha, double beta3D)
{
    #define INS(ii,jj,kk) in[IDX(ii,jj,kk,ny,nz)]
    out[IDX(i,j,k,ny,nz)] = alpha*INS(i,j,k) + beta3D*(
        INS(i-1,j,k)+INS(i+1,j,k)+INS(i,j-1,k)+INS(i,j+1,k)+INS(i,j,k-1)+INS(i,j,k+1));
    #undef INS
}

__global__ void k_smoothStep_interior(
    double* __restrict__ out, const double* __restrict__ in,
    int nx, int ny, int nz, double alpha, double beta3D,
    int iLo, int iHi, int jLo, int jHi, int kLo, int kHi)
{
    int i=blockIdx.x*BX+threadIdx.x+iLo, j=blockIdx.y*BY+threadIdx.y+jLo, k=blockIdx.z*BZ+threadIdx.z+kLo;
    if (i>iHi||j>jHi||k>kHi) return;
    d_smoothStep(out,in,i,j,k,ny,nz,alpha,beta3D);
}

__global__ void k_smoothStep_boundary(
    double* __restrict__ out, const double* __restrict__ in,
    int nx, int ny, int nz, double alpha, double beta3D,
    int sILo, int sIHi, int sJLo, int sJHi, int sKLo, int sKHi)
{
    int i=blockIdx.x*BX+threadIdx.x+1, j=blockIdx.y*BY+threadIdx.y+1, k=blockIdx.z*BZ+threadIdx.z+1;
    if (i>nx-2||j>ny-2||k>nz-2) return;
    if (i>=sILo&&i<=sIHi&&j>=sJLo&&j<=sJHi&&k>=sKLo&&k<=sKHi) return;
    d_smoothStep(out,in,i,j,k,ny,nz,alpha,beta3D);
}

void gpuSmoothStep_interior(double* out, const double* in,
                            int nx, int ny, int nz,
                            double alpha, double beta3D,
                            cudaStream_t stream)
{
    int iLo=2,iHi=nx-3,jLo=2,jHi=ny-3,kLo=2,kHi=nz-3;
    if (iLo>iHi||jLo>jHi||kLo>kHi) return;
    dim3 grid=rangedGrid(iLo,iHi,jLo,jHi,kLo,kHi); dim3 block(BX,BY,BZ);
    k_smoothStep_interior<<<grid,block,0,stream>>>(out,in,nx,ny,nz,alpha,beta3D,iLo,iHi,jLo,jHi,kLo,kHi);
    cudaErrChk(cudaGetLastError());
}

void gpuSmoothStep_boundary(double* out, const double* in,
                            int nx, int ny, int nz,
                            double alpha, double beta3D,
                            cudaStream_t stream)
{
    dim3 grid=stencilGrid(nx,ny,nz); dim3 block(BX,BY,BZ);
    k_smoothStep_boundary<<<grid,block,0,stream>>>(out,in,nx,ny,nz,alpha,beta3D,2,nx-3,2,ny-3,2,nz-3);
    cudaErrChk(cudaGetLastError());
}

#endif // HALO_OVERLAP

#undef IDX

#endif // GPU_SOLVER
