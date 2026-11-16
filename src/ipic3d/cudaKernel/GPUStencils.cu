/**
 * @file GPUStencils.cu
 * @brief CUDA kernel implementations for all grid stencil operations.
 *        See GPUStencils.cuh for API documentation.
 *
 * Each thread computes one output point, with the launch shape of
 * GPUPointLaunch.cuh (k, the contiguous index, on threadIdx.x).
 * The flat row-major index [i][j][k] = (i*ny + j)*nz + k (stencilIdx) is
 * used for both node-based and center-based grids.
 */

#include "GPUPointLaunch.cuh"
#include "GPUStencils.cuh"

// =========================================================================
//  Gradient kernels
// =========================================================================

/**
 * gradC2N: gradient center→node.
 * Interior output on node grid: i=1..nxn-2 → reads center[i-1..i].
 * nxc = nxn-1, nyc = nyn-1, nzc = nzn-1.
 */
template <typename T>
__global__ void k_gradC2N(T* __restrict__ gradXN, T* __restrict__ gradYN,
                          T* __restrict__ gradZN, const T* __restrict__ C,
                          int nxn, int nyn, int nzn, int nyc, int nzc,
                          DiffScale<T> q) {
  int i = blockIdx.z * blockDim.z + threadIdx.z + 1;
  int j = blockIdx.y * blockDim.y + threadIdx.y + 1;
  int k = blockIdx.x * blockDim.x + threadIdx.x + 1;
  if (i > nxn - 2 || j > nyn - 2 || k > nzn - 2)
    return;

  const int nidx = stencilIdx(i, j, k, nyn, nzn);
  gradC2NAt(C, i, j, k, nyc, nzc, q, gradXN[nidx], gradYN[nidx], gradZN[nidx]);
}

/**
 * gradN2C: gradient node→center.
 * Output on center grid: i=iLo..iHi (interior 1..nxc-2 by default), j, k
 * interior → reads node[i..i+1].
 * Not used (gpuGradN2C, a unit-test reference).
 */
template <typename T>
__global__ void k_gradN2C(T* __restrict__ gradXC, T* __restrict__ gradYC,
                          T* __restrict__ gradZC, const T* __restrict__ N,
                          int iLo, int iHi, int nyc, int nzc, int nyn, int nzn,
                          DiffScale<T> q) {
  int i = blockIdx.z * blockDim.z + threadIdx.z + iLo;
  int j = blockIdx.y * blockDim.y + threadIdx.y + 1;
  int k = blockIdx.x * blockDim.x + threadIdx.x + 1;
  if (i > iHi || j > nyc - 2 || k > nzc - 2)
    return;

  const int cidx = stencilIdx(i, j, k, nyc, nzc);
  gradN2CAt(N, i, j, k, nyn, nzn, q, gradXC[cidx], gradYC[cidx], gradZC[cidx]);
}

// =========================================================================
//  Divergence kernels
// =========================================================================

/** divN2C: divergence node→center.  Same stencil as gradN2C but applied to 3
 * components. */
template <typename T>
__global__ void k_divN2C(T* __restrict__ divC, const T* __restrict__ vecXN,
                         const T* __restrict__ vecYN,
                         const T* __restrict__ vecZN, int iLo, int iHi, int nyc,
                         int nzc, int nyn, int nzn, DiffScale<T> q) {
  int i = blockIdx.z * blockDim.z + threadIdx.z + iLo;
  int j = blockIdx.y * blockDim.y + threadIdx.y + 1;
  int k = blockIdx.x * blockDim.x + threadIdx.x + 1;
  if (i > iHi || j > nyc - 2 || k > nzc - 2)
    return;

  divC[stencilIdx(i, j, k, nyc, nzc)] =
      divN2CAt(vecXN, vecYN, vecZN, i, j, k, nyn, nzn, q);
}

/** divC2N: divergence center→node.  Reads center[i-1..i] for node output.
 *  Not used (gpuDivC2N, a unit-test reference). */
template <typename T>
__global__ void k_divC2N(T* __restrict__ divN, const T* __restrict__ vecXC,
                         const T* __restrict__ vecYC,
                         const T* __restrict__ vecZC, int nxn, int nyn, int nzn,
                         int nyc, int nzc, DiffScale<T> q) {
  int i = blockIdx.z * blockDim.z + threadIdx.z + 1;
  int j = blockIdx.y * blockDim.y + threadIdx.y + 1;
  int k = blockIdx.x * blockDim.x + threadIdx.x + 1;
  if (i > nxn - 2 || j > nyn - 2 || k > nzn - 2)
    return;

  divN[stencilIdx(i, j, k, nyn, nzn)] =
      divC2NAt(vecXC, vecYC, vecZC, i, j, k, nyc, nzc, q);
}

// =========================================================================
//  Curl kernels
// =========================================================================

template <typename T>
__global__ void k_curlC2N(T* __restrict__ curlXN, T* __restrict__ curlYN,
                          T* __restrict__ curlZN, const T* __restrict__ vecXC,
                          const T* __restrict__ vecYC,
                          const T* __restrict__ vecZC, int nxn, int nyn,
                          int nzn, int nyc, int nzc, DiffScale<T> q) {
  int i = blockIdx.z * blockDim.z + threadIdx.z + 1;
  int j = blockIdx.y * blockDim.y + threadIdx.y + 1;
  int k = blockIdx.x * blockDim.x + threadIdx.x + 1;
  if (i > nxn - 2 || j > nyn - 2 || k > nzn - 2)
    return;

  const auto C_ = [&](const T* a, int ii, int jj, int kk) {
    return a[stencilIdx(ii, jj, kk, nyc, nzc)];
  };
  // curl_X = dZdy - dYdz
  T compZDY =
      q.y * ((C_(vecZC, i, j, k) - C_(vecZC, i, j - 1, k)) +
             (C_(vecZC, i, j, k - 1) - C_(vecZC, i, j - 1, k - 1)) +
             (C_(vecZC, i - 1, j, k) - C_(vecZC, i - 1, j - 1, k)) +
             (C_(vecZC, i - 1, j, k - 1) - C_(vecZC, i - 1, j - 1, k - 1)));
  T compYDZ =
      q.z * ((C_(vecYC, i, j, k) - C_(vecYC, i, j, k - 1)) +
             (C_(vecYC, i - 1, j, k) - C_(vecYC, i - 1, j, k - 1)) +
             (C_(vecYC, i, j - 1, k) - C_(vecYC, i, j - 1, k - 1)) +
             (C_(vecYC, i - 1, j - 1, k) - C_(vecYC, i - 1, j - 1, k - 1)));

  // curl_Y = dXdz - dZdx
  T compXDZ =
      q.z * ((C_(vecXC, i, j, k) - C_(vecXC, i, j, k - 1)) +
             (C_(vecXC, i - 1, j, k) - C_(vecXC, i - 1, j, k - 1)) +
             (C_(vecXC, i, j - 1, k) - C_(vecXC, i, j - 1, k - 1)) +
             (C_(vecXC, i - 1, j - 1, k) - C_(vecXC, i - 1, j - 1, k - 1)));
  T compZDX =
      q.x * ((C_(vecZC, i, j, k) - C_(vecZC, i - 1, j, k)) +
             (C_(vecZC, i, j, k - 1) - C_(vecZC, i - 1, j, k - 1)) +
             (C_(vecZC, i, j - 1, k) - C_(vecZC, i - 1, j - 1, k)) +
             (C_(vecZC, i, j - 1, k - 1) - C_(vecZC, i - 1, j - 1, k - 1)));

  // curl_Z = dYdx - dXdy
  T compYDX =
      q.x * ((C_(vecYC, i, j, k) - C_(vecYC, i - 1, j, k)) +
             (C_(vecYC, i, j, k - 1) - C_(vecYC, i - 1, j, k - 1)) +
             (C_(vecYC, i, j - 1, k) - C_(vecYC, i - 1, j - 1, k)) +
             (C_(vecYC, i, j - 1, k - 1) - C_(vecYC, i - 1, j - 1, k - 1)));
  T compXDY =
      q.y * ((C_(vecXC, i, j, k) - C_(vecXC, i, j - 1, k)) +
             (C_(vecXC, i, j, k - 1) - C_(vecXC, i, j - 1, k - 1)) +
             (C_(vecXC, i - 1, j, k) - C_(vecXC, i - 1, j - 1, k)) +
             (C_(vecXC, i - 1, j, k - 1) - C_(vecXC, i - 1, j - 1, k - 1)));

  int nidx = stencilIdx(i, j, k, nyn, nzn);
  curlXN[nidx] = compZDY - compYDZ;
  curlYN[nidx] = compXDZ - compZDX;
  curlZN[nidx] = compYDX - compXDY;
}

template <typename T>
__global__ void k_curlN2C(T* __restrict__ curlXC, T* __restrict__ curlYC,
                          T* __restrict__ curlZC, const T* __restrict__ vecXN,
                          const T* __restrict__ vecYN,
                          const T* __restrict__ vecZN, int nxc, int nyc,
                          int nzc, int nyn, int nzn, DiffScale<T> q) {
  int i = blockIdx.z * blockDim.z + threadIdx.z + 1;
  int j = blockIdx.y * blockDim.y + threadIdx.y + 1;
  int k = blockIdx.x * blockDim.x + threadIdx.x + 1;
  if (i > nxc - 2 || j > nyc - 2 || k > nzc - 2)
    return;

  T xx, xy, xz, yx, yy, yz, zx, zy, zz; // d(component)/d(axis)
  gradN2CAt(vecXN, i, j, k, nyn, nzn, q, xx, xy, xz);
  gradN2CAt(vecYN, i, j, k, nyn, nzn, q, yx, yy, yz);
  gradN2CAt(vecZN, i, j, k, nyn, nzn, q, zx, zy, zz);
  const int cidx = stencilIdx(i, j, k, nyc, nzc);
  curlXC[cidx] = zy - yz;
  curlYC[cidx] = xz - zx;
  curlZC[cidx] = yx - xy;
}

// =========================================================================
//  Interpolation kernels
// =========================================================================

/** interpC2N at node (i, j, k): the centres (i-1..i, j-1..j, k-1..k). */
template <typename T>
__device__ __forceinline__ void
d_interpC2N(T* __restrict__ fN, const T* __restrict__ fC, int i, int j, int k,
            int nyn, int nzn, int nyc, int nzc) {
  const auto CI = [&](int ii, int jj, int kk) {
    return fC[stencilIdx(ii, jj, kk, nyc, nzc)];
  };
  fN[stencilIdx(i, j, k, nyn, nzn)] =
      0.125 * (CI(i, j, k) + CI(i - 1, j, k) + CI(i, j - 1, k) +
               CI(i, j, k - 1) + CI(i - 1, j - 1, k) + CI(i - 1, j, k - 1) +
               CI(i, j - 1, k - 1) + CI(i - 1, j - 1, k - 1));
}

/** interpN2C, the face planes of the conductor walls w read as 0 (the
 *  image of a charge-like scalar; without walls the plain mean). */
template <typename T>
__global__ void k_interpN2C(T* __restrict__ fieldC,
                            const T* __restrict__ fieldN, int nxc, int nyc,
                            int nzc, int nyn, int nzn, ImageWalls w) {
  int i = blockIdx.z * blockDim.z + threadIdx.z + 1;
  int j = blockIdx.y * blockDim.y + threadIdx.y + 1;
  int k = blockIdx.x * blockDim.x + threadIdx.x + 1;
  if (i > nxc - 2 || j > nyc - 2 || k > nzc - 2)
    return;

  fieldC[stencilIdx(i, j, k, nyc, nzc)] =
      interpN2CAt(fieldN, i, j, k, nyn, nzn, imagePlanes(w, i, j, k));
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
 * divN2CAt per row: reads node [i..i+1]×[j..j+1]×[k..k+1].
 */
template <typename T>
__global__ void
k_divSymmTensorN2C(T* __restrict__ divCX, T* __restrict__ divCY,
                   T* __restrict__ divCZ, const T* __restrict__ pXX,
                   const T* __restrict__ pXY, const T* __restrict__ pXZ,
                   const T* __restrict__ pYY, const T* __restrict__ pYZ,
                   const T* __restrict__ pZZ, int nxc, int nyc, int nzc,
                   int nyn, int nzn, DiffScale<T> q) {
  int i = blockIdx.z * blockDim.z + threadIdx.z + 1;
  int j = blockIdx.y * blockDim.y + threadIdx.y + 1;
  int k = blockIdx.x * blockDim.x + threadIdx.x + 1;
  if (i > nxc - 2 || j > nyc - 2 || k > nzc - 2)
    return;

  const int cidx = stencilIdx(i, j, k, nyc, nzc);
  divCX[cidx] = divN2CAt(pXX, pXY, pXZ, i, j, k, nyn, nzn, q);
  divCY[cidx] = divN2CAt(pXY, pYY, pYZ, i, j, k, nyn, nzn, q);
  divCZ[cidx] = divN2CAt(pXZ, pYZ, pZZ, i, j, k, nyn, nzn, q);
}

// =========================================================================
//  Host wrappers
// =========================================================================

void gpuGradC2N(cudaSolverType* gradXN, cudaSolverType* gradYN,
                cudaSolverType* gradZN, const cudaSolverType* scFieldC, int nxn,
                int nyn, int nzn, cudaSolverType invdx, cudaSolverType invdy,
                cudaSolverType invdz, cudaStream_t stream) {
  int nyc = nyn - 1, nzc = nzn - 1;
  const PointLaunch L = pointLaunch(nxn - 2, nyn - 2, nzn - 2);
  k_gradC2N<<<L.grid, L.block, 0, stream>>>(gradXN, gradYN, gradZN, scFieldC,
                                            nxn, nyn, nzn, nyc, nzc,
                                            diffScale(invdx, invdy, invdz));
  cudaErrChk(cudaGetLastError());
}

void gpuGradN2C(cudaSolverType* gradXC, cudaSolverType* gradYC,
                cudaSolverType* gradZC, const cudaSolverType* scFieldN, int nxc,
                int nyc, int nzc, cudaSolverType invdx, cudaSolverType invdy,
                cudaSolverType invdz, cudaStream_t stream, int iLo, int iHi) {
  if (iHi < 0)
    iHi = nxc - 2;
  if (iHi < iLo)
    return;
  int nyn = nyc + 1, nzn = nzc + 1;
  const PointLaunch L = pointLaunch(iHi - iLo + 1, nyc - 2, nzc - 2);
  k_gradN2C<<<L.grid, L.block, 0, stream>>>(gradXC, gradYC, gradZC, scFieldN,
                                            iLo, iHi, nyc, nzc, nyn, nzn,
                                            diffScale(invdx, invdy, invdz));
  cudaErrChk(cudaGetLastError());
}

void gpuDivN2C(cudaSolverType* divC, const cudaSolverType* vecXN,
               const cudaSolverType* vecYN, const cudaSolverType* vecZN,
               int nxc, int nyc, int nzc, cudaSolverType invdx,
               cudaSolverType invdy, cudaSolverType invdz, cudaStream_t stream,
               int iLo, int iHi) {
  if (iHi < 0)
    iHi = nxc - 2;
  if (iHi < iLo)
    return;
  int nyn = nyc + 1, nzn = nzc + 1;
  const PointLaunch L = pointLaunch(iHi - iLo + 1, nyc - 2, nzc - 2);
  k_divN2C<<<L.grid, L.block, 0, stream>>>(divC, vecXN, vecYN, vecZN, iLo, iHi,
                                           nyc, nzc, nyn, nzn,
                                           diffScale(invdx, invdy, invdz));
  cudaErrChk(cudaGetLastError());
}

void gpuDivC2N(cudaSolverType* divN, const cudaSolverType* vecXC,
               const cudaSolverType* vecYC, const cudaSolverType* vecZC,
               int nxn, int nyn, int nzn, cudaSolverType invdx,
               cudaSolverType invdy, cudaSolverType invdz,
               cudaStream_t stream) {
  int nyc = nyn - 1, nzc = nzn - 1;
  const PointLaunch L = pointLaunch(nxn - 2, nyn - 2, nzn - 2);
  k_divC2N<<<L.grid, L.block, 0, stream>>>(divN, vecXC, vecYC, vecZC, nxn, nyn,
                                           nzn, nyc, nzc,
                                           diffScale(invdx, invdy, invdz));
  cudaErrChk(cudaGetLastError());
}

void gpuCurlC2N(cudaSolverType* curlXN, cudaSolverType* curlYN,
                cudaSolverType* curlZN, const cudaSolverType* vecXC,
                const cudaSolverType* vecYC, const cudaSolverType* vecZC,
                int nxn, int nyn, int nzn, cudaSolverType invdx,
                cudaSolverType invdy, cudaSolverType invdz,
                cudaStream_t stream) {
  int nyc = nyn - 1, nzc = nzn - 1;
  const PointLaunch L = pointLaunch(nxn - 2, nyn - 2, nzn - 2);
  k_curlC2N<<<L.grid, L.block, 0, stream>>>(
      curlXN, curlYN, curlZN, vecXC, vecYC, vecZC, nxn, nyn, nzn, nyc, nzc,
      diffScale(invdx, invdy, invdz));
  cudaErrChk(cudaGetLastError());
}

void gpuCurlN2C(cudaSolverType* curlXC, cudaSolverType* curlYC,
                cudaSolverType* curlZC, const cudaSolverType* vecXN,
                const cudaSolverType* vecYN, const cudaSolverType* vecZN,
                int nxc, int nyc, int nzc, cudaSolverType invdx,
                cudaSolverType invdy, cudaSolverType invdz,
                cudaStream_t stream) {
  int nyn = nyc + 1, nzn = nzc + 1;
  const PointLaunch L = pointLaunch(nxc - 2, nyc - 2, nzc - 2);
  k_curlN2C<<<L.grid, L.block, 0, stream>>>(
      curlXC, curlYC, curlZC, vecXN, vecYN, vecZN, nxc, nyc, nzc, nyn, nzn,
      diffScale(invdx, invdy, invdz));
  cudaErrChk(cudaGetLastError());
}

void gpuInterpC2N(cudaSolverType* fieldN, const cudaSolverType* fieldC, int nxn,
                  int nyn, int nzn, cudaStream_t stream) {
  gpuInterpC2N_range(fieldN, fieldC, nxn, nyn, nzn, 1, nxn - 2, 1, nyn - 2, 1,
                     nzn - 2, stream);
}

void gpuInterpN2C(cudaSolverType* fieldC, const cudaSolverType* fieldN, int nxc,
                  int nyc, int nzc, cudaStream_t stream,
                  const ImageWalls& walls) {
  int nyn = nyc + 1, nzn = nzc + 1;
  const PointLaunch L = pointLaunch(nxc - 2, nyc - 2, nzc - 2);
  k_interpN2C<<<L.grid, L.block, 0, stream>>>(fieldC, fieldN, nxc, nyc, nzc,
                                              nyn, nzn, walls);
  cudaErrChk(cudaGetLastError());
}

void gpuDivSymmTensorN2C(cudaSolverType* divCX, cudaSolverType* divCY,
                         cudaSolverType* divCZ, const cudaSolverType* pXX,
                         const cudaSolverType* pXY, const cudaSolverType* pXZ,
                         const cudaSolverType* pYY, const cudaSolverType* pYZ,
                         const cudaSolverType* pZZ, int nxc, int nyc, int nzc,
                         cudaSolverType invdx, cudaSolverType invdy,
                         cudaSolverType invdz, cudaStream_t stream) {
  int nyn = nyc + 1, nzn = nzc + 1;
  const PointLaunch L = pointLaunch(nxc - 2, nyc - 2, nzc - 2);
  k_divSymmTensorN2C<<<L.grid, L.block, 0, stream>>>(
      divCX, divCY, divCZ, pXX, pXY, pXZ, pYY, pYZ, pZZ, nxc, nyc, nzc, nyn,
      nzn, diffScale(invdx, invdy, invdz));
  cudaErrChk(cudaGetLastError());
}

// =========================================================================
//  Interior / boundary stencil variants (HaloMode::Overlap)
//
//  Each centre→node stencil (divC2N, gradC2N, interpC2N) reads
//  centre[i-1..i][j-1..j][k-1..k] for node output at [i][j][k].
//    Interior [2..n-3]: all reads fall in centre interior — no ghost deps.
//    Boundary [1, n-2]: at least one read hits a ghost centre cell.
//
//  The divC2N and gradC2N variants share the device stencils of
//  GPUStencils.cuh, the interpC2N ones d_interpC2N.
// =========================================================================

// -----------------------------------------------------------------
//  divC2N  (centre → node divergence; these variants are not used)
// -----------------------------------------------------------------
template <typename T>
__global__ void
k_divC2N_interior(T* __restrict__ divN, const T* __restrict__ vXC,
                  const T* __restrict__ vYC, const T* __restrict__ vZC, int nxn,
                  int nyn, int nzn, int nyc, int nzc, DiffScale<T> q, int iLo,
                  int iHi, int jLo, int jHi, int kLo, int kHi) {
  int i = blockIdx.z * blockDim.z + threadIdx.z + iLo;
  int j = blockIdx.y * blockDim.y + threadIdx.y + jLo;
  int k = blockIdx.x * blockDim.x + threadIdx.x + kLo;
  if (i > iHi || j > jHi || k > kHi)
    return;
  divN[stencilIdx(i, j, k, nyn, nzn)] =
      divC2NAt(vXC, vYC, vZC, i, j, k, nyc, nzc, q);
}

template <typename T>
__global__ void
k_divC2N_boundary(T* __restrict__ divN, const T* __restrict__ vXC,
                  const T* __restrict__ vYC, const T* __restrict__ vZC, int nxn,
                  int nyn, int nzn, int nyc, int nzc, DiffScale<T> q, int sILo,
                  int sIHi, int sJLo, int sJHi, int sKLo, int sKHi) {
  int i = blockIdx.z * blockDim.z + threadIdx.z + 1;
  int j = blockIdx.y * blockDim.y + threadIdx.y + 1;
  int k = blockIdx.x * blockDim.x + threadIdx.x + 1;
  if (i > nxn - 2 || j > nyn - 2 || k > nzn - 2)
    return;
  if (i >= sILo && i <= sIHi && j >= sJLo && j <= sJHi && k >= sKLo &&
      k <= sKHi)
    return;
  divN[stencilIdx(i, j, k, nyn, nzn)] =
      divC2NAt(vXC, vYC, vZC, i, j, k, nyc, nzc, q);
}

void gpuDivC2N_interior(cudaSolverType* divN, const cudaSolverType* vXC,
                        const cudaSolverType* vYC, const cudaSolverType* vZC,
                        int nxn, int nyn, int nzn, cudaSolverType invdx,
                        cudaSolverType invdy, cudaSolverType invdz,
                        cudaStream_t stream) {
  int iLo = 2, iHi = nxn - 3, jLo = 2, jHi = nyn - 3, kLo = 2, kHi = nzn - 3;
  if (iLo > iHi || jLo > jHi || kLo > kHi)
    return;
  int nyc = nyn - 1, nzc = nzn - 1;
  const PointLaunch L =
      pointLaunch(iHi - iLo + 1, jHi - jLo + 1, kHi - kLo + 1);
  k_divC2N_interior<<<L.grid, L.block, 0, stream>>>(
      divN, vXC, vYC, vZC, nxn, nyn, nzn, nyc, nzc,
      diffScale(invdx, invdy, invdz), iLo, iHi, jLo, jHi, kLo, kHi);
  cudaErrChk(cudaGetLastError());
}

void gpuDivC2N_boundary(cudaSolverType* divN, const cudaSolverType* vXC,
                        const cudaSolverType* vYC, const cudaSolverType* vZC,
                        int nxn, int nyn, int nzn, cudaSolverType invdx,
                        cudaSolverType invdy, cudaSolverType invdz,
                        cudaStream_t stream) {
  int nyc = nyn - 1, nzc = nzn - 1;
  const PointLaunch L = pointLaunch(nxn - 2, nyn - 2, nzn - 2);
  k_divC2N_boundary<<<L.grid, L.block, 0, stream>>>(
      divN, vXC, vYC, vZC, nxn, nyn, nzn, nyc, nzc,
      diffScale(invdx, invdy, invdz), 2, nxn - 3, 2, nyn - 3, 2, nzn - 3);
  cudaErrChk(cudaGetLastError());
}

// -----------------------------------------------------------------
//  gradC2N  (centre → node gradient; these variants are not used)
// -----------------------------------------------------------------
template <typename T>
__global__ void k_gradC2N_interior(T* __restrict__ gXN, T* __restrict__ gYN,
                                   T* __restrict__ gZN, const T* __restrict__ C,
                                   int nxn, int nyn, int nzn, int nyc, int nzc,
                                   DiffScale<T> q, int iLo, int iHi, int jLo,
                                   int jHi, int kLo, int kHi) {
  int i = blockIdx.z * blockDim.z + threadIdx.z + iLo,
      j = blockIdx.y * blockDim.y + threadIdx.y + jLo,
      k = blockIdx.x * blockDim.x + threadIdx.x + kLo;
  if (i > iHi || j > jHi || k > kHi)
    return;
  const int nidx = stencilIdx(i, j, k, nyn, nzn);
  gradC2NAt(C, i, j, k, nyc, nzc, q, gXN[nidx], gYN[nidx], gZN[nidx]);
}

template <typename T>
__global__ void k_gradC2N_boundary(T* __restrict__ gXN, T* __restrict__ gYN,
                                   T* __restrict__ gZN, const T* __restrict__ C,
                                   int nxn, int nyn, int nzn, int nyc, int nzc,
                                   DiffScale<T> q, int sILo, int sIHi, int sJLo,
                                   int sJHi, int sKLo, int sKHi) {
  int i = blockIdx.z * blockDim.z + threadIdx.z + 1,
      j = blockIdx.y * blockDim.y + threadIdx.y + 1,
      k = blockIdx.x * blockDim.x + threadIdx.x + 1;
  if (i > nxn - 2 || j > nyn - 2 || k > nzn - 2)
    return;
  if (i >= sILo && i <= sIHi && j >= sJLo && j <= sJHi && k >= sKLo &&
      k <= sKHi)
    return;
  const int nidx = stencilIdx(i, j, k, nyn, nzn);
  gradC2NAt(C, i, j, k, nyc, nzc, q, gXN[nidx], gYN[nidx], gZN[nidx]);
}

void gpuGradC2N_interior(cudaSolverType* gXN, cudaSolverType* gYN,
                         cudaSolverType* gZN, const cudaSolverType* C, int nxn,
                         int nyn, int nzn, cudaSolverType invdx,
                         cudaSolverType invdy, cudaSolverType invdz,
                         cudaStream_t stream) {
  int iLo = 2, iHi = nxn - 3, jLo = 2, jHi = nyn - 3, kLo = 2, kHi = nzn - 3;
  if (iLo > iHi || jLo > jHi || kLo > kHi)
    return;
  int nyc = nyn - 1, nzc = nzn - 1;
  const PointLaunch L =
      pointLaunch(iHi - iLo + 1, jHi - jLo + 1, kHi - kLo + 1);
  k_gradC2N_interior<<<L.grid, L.block, 0, stream>>>(
      gXN, gYN, gZN, C, nxn, nyn, nzn, nyc, nzc, diffScale(invdx, invdy, invdz),
      iLo, iHi, jLo, jHi, kLo, kHi);
  cudaErrChk(cudaGetLastError());
}

void gpuGradC2N_boundary(cudaSolverType* gXN, cudaSolverType* gYN,
                         cudaSolverType* gZN, const cudaSolverType* C, int nxn,
                         int nyn, int nzn, cudaSolverType invdx,
                         cudaSolverType invdy, cudaSolverType invdz,
                         cudaStream_t stream) {
  int nyc = nyn - 1, nzc = nzn - 1;
  const PointLaunch L = pointLaunch(nxn - 2, nyn - 2, nzn - 2);
  k_gradC2N_boundary<<<L.grid, L.block, 0, stream>>>(
      gXN, gYN, gZN, C, nxn, nyn, nzn, nyc, nzc, diffScale(invdx, invdy, invdz),
      2, nxn - 3, 2, nyn - 3, 2, nzn - 3);
  cudaErrChk(cudaGetLastError());
}

template <typename T>
__global__ void
k_interpC2N_interior(T* __restrict__ fN, const T* __restrict__ fC, int nxn,
                     int nyn, int nzn, int nyc, int nzc, int iLo, int iHi,
                     int jLo, int jHi, int kLo, int kHi) {
  int i = blockIdx.z * blockDim.z + threadIdx.z + iLo,
      j = blockIdx.y * blockDim.y + threadIdx.y + jLo,
      k = blockIdx.x * blockDim.x + threadIdx.x + kLo;
  if (i > iHi || j > jHi || k > kHi)
    return;
  d_interpC2N(fN, fC, i, j, k, nyn, nzn, nyc, nzc);
}

template <typename T>
__device__ __forceinline__ void
d_interpC2N_box(T* __restrict__ fN, const T* __restrict__ fC,
                unsigned int linear, int iLo, int jLo, int kLo, int jCount,
                int kCount, int nyn, int nzn, int nyc, int nzc) {
  const unsigned int jkCount = jCount * kCount;
  const int i = iLo + static_cast<int>(linear / jkCount);
  const unsigned int rem = linear % jkCount;
  const int j = jLo + static_cast<int>(rem / kCount);
  const int k = kLo + static_cast<int>(rem % kCount);
  d_interpC2N(fN, fC, i, j, k, nyn, nzn, nyc, nzc);
}

// Map one linear launch over the six disjoint slabs surrounding the safe box,
// with one thread for each node in the complement.
template <typename T>
__global__ void k_interpC2N_complement(T* __restrict__ fN,
                                       const T* __restrict__ fC, int nxn,
                                       int nyn, int nzn, int nyc, int nzc,
                                       int sILo, int sIHi, int sJLo, int sJHi,
                                       int sKLo, int sKHi, unsigned int total) {
  unsigned int linear = blockIdx.x * blockDim.x + threadIdx.x;
  if (linear >= total)
    return;

  const int activeI = nxn - 2;
  const int activeJ = nyn - 2;
  const int activeK = nzn - 2;
  const int safeI = sIHi - sILo + 1;
  const int safeJ = sJHi - sJLo + 1;

  // Select slab metadata first, then execute one shared interpolation tail.
  // Keeping d_interpC2N outside the branch tree prevents the compiler from
  // cloning the eight-load stencil for every slab.
  int iLo, jLo, kLo, jCount, kCount;
  unsigned int slab = (sILo - 1) * activeJ * activeK;
  if (linear < slab) {
    // Low-X slab spans the full active Y-Z plane.
    iLo = jLo = kLo = 1;
    jCount = activeJ;
    kCount = activeK;
  } else {
    linear -= slab;
    slab = (activeI - sIHi) * activeJ * activeK;
    if (linear < slab) {
      // High-X slab spans the full active Y-Z plane.
      iLo = sIHi + 1;
      jLo = kLo = 1;
      jCount = activeJ;
      kCount = activeK;
    } else {
      linear -= slab;
      slab = safeI * (sJLo - 1) * activeK;
      if (linear < slab) {
        // Low-Y slab is restricted to the safe X interval.
        iLo = sILo;
        jLo = kLo = 1;
        jCount = sJLo - 1;
        kCount = activeK;
      } else {
        linear -= slab;
        slab = safeI * (activeJ - sJHi) * activeK;
        if (linear < slab) {
          // High-Y slab is restricted to the safe X interval.
          iLo = sILo;
          jLo = sJHi + 1;
          kLo = 1;
          jCount = activeJ - sJHi;
          kCount = activeK;
        } else {
          linear -= slab;
          slab = safeI * safeJ * (sKLo - 1);
          iLo = sILo;
          jLo = sJLo;
          jCount = safeJ;
          if (linear < slab) {
            // Low-Z slab is restricted to the safe X-Y rectangle.
            kLo = 1;
            kCount = sKLo - 1;
          } else {
            // High-Z slab is the remaining safe-X-Y complement.
            linear -= slab;
            kLo = sKHi + 1;
            kCount = activeK - sKHi;
          }
        }
      }
    }
  }

  d_interpC2N_box(fN, fC, linear, iLo, jLo, kLo, jCount, kCount, nyn, nzn, nyc,
                  nzc);
}

void gpuInterpC2N_range(cudaSolverType* fN, const cudaSolverType* fC, int nxn,
                        int nyn, int nzn, int iLo, int iHi, int jLo, int jHi,
                        int kLo, int kHi, cudaStream_t stream) {
  // Only active node values are defined by interpC2N.  Clamp here so callers
  // can conservatively shrink a safe box without manufacturing launch bounds.
  iLo = (iLo < 1) ? 1 : iLo;
  jLo = (jLo < 1) ? 1 : jLo;
  kLo = (kLo < 1) ? 1 : kLo;
  iHi = (iHi > nxn - 2) ? nxn - 2 : iHi;
  jHi = (jHi > nyn - 2) ? nyn - 2 : jHi;
  kHi = (kHi > nzn - 2) ? nzn - 2 : kHi;
  if (iLo > iHi || jLo > jHi || kLo > kHi)
    return;
  int nyc = nyn - 1, nzc = nzn - 1;
  const PointLaunch L =
      pointLaunch(iHi - iLo + 1, jHi - jLo + 1, kHi - kLo + 1);
  k_interpC2N_interior<<<L.grid, L.block, 0, stream>>>(
      fN, fC, nxn, nyn, nzn, nyc, nzc, iLo, iHi, jLo, jHi, kLo, kHi);
  cudaErrChk(cudaGetLastError());
}

void gpuInterpC2N_complement(cudaSolverType* fN, const cudaSolverType* fC,
                             int nxn, int nyn, int nzn, int safeILo,
                             int safeIHi, int safeJLo, int safeJHi, int safeKLo,
                             int safeKHi, cudaStream_t stream) {
  safeILo = (safeILo < 1) ? 1 : safeILo;
  safeJLo = (safeJLo < 1) ? 1 : safeJLo;
  safeKLo = (safeKLo < 1) ? 1 : safeKLo;
  safeIHi = (safeIHi > nxn - 2) ? nxn - 2 : safeIHi;
  safeJHi = (safeJHi > nyn - 2) ? nyn - 2 : safeJHi;
  safeKHi = (safeKHi > nzn - 2) ? nzn - 2 : safeKHi;

  // No early work was possible: overwrite every active node after the halo
  // and all center-field fixups have completed.
  if (safeILo > safeIHi || safeJLo > safeJHi || safeKLo > safeKHi) {
    gpuInterpC2N(fN, fC, nxn, nyn, nzn, stream);
    return;
  }

  const int activeI = nxn - 2, activeJ = nyn - 2, activeK = nzn - 2;
  const int safeI = safeIHi - safeILo + 1;
  const int safeJ = safeJHi - safeJLo + 1;
  const size_t totalSize =
      (size_t)(safeILo - 1 + activeI - safeIHi) * activeJ * activeK +
      (size_t)safeI * (safeJLo - 1 + activeJ - safeJHi) * activeK +
      (size_t)safeI * safeJ * (safeKLo - 1 + activeK - safeKHi);
  if (totalSize == 0)
    return;
  // EMfields3D construction validates the signed-int Krylov/indexing contract;
  // the compact decoder therefore uses faster 32-bit div/rem safely.
  const unsigned int total = static_cast<unsigned int>(totalSize);

  constexpr int block = 256;
  const int nyc = nyn - 1, nzc = nzn - 1;
  k_interpC2N_complement<<<(total - 1) / block + 1, block, 0, stream>>>(
      fN, fC, nxn, nyn, nzn, nyc, nzc, safeILo, safeIHi, safeJLo, safeJHi,
      safeKLo, safeKHi, total);
  cudaErrChk(cudaGetLastError());
}

void gpuInterpC2N_interior(cudaSolverType* fN, const cudaSolverType* fC,
                           int nxn, int nyn, int nzn, cudaStream_t stream) {
  gpuInterpC2N_range(fN, fC, nxn, nyn, nzn, 2, nxn - 3, 2, nyn - 3, 2, nzn - 3,
                     stream);
}

void gpuInterpC2N_boundary(cudaSolverType* fN, const cudaSolverType* fC,
                           int nxn, int nyn, int nzn, cudaStream_t stream) {
  gpuInterpC2N_complement(fN, fC, nxn, nyn, nzn, 2, nxn - 3, 2, nyn - 3, 2,
                          nzn - 3, stream);
}

// =========================================================================
//  Smoothing pass (gpuSmoothFields): the 7-point smooth of 1 or 3 fields in
//  one launch over all points, the interior or the shell
// =========================================================================

/** The 7-point smooth of one field at one point. */
template <typename T>
__device__ __forceinline__ void
d_smoothStep(T* __restrict__ out, const T* __restrict__ in, int i, int j, int k,
             int ny, int nz, T alpha, T beta3D) {
  const auto INS = [&](int ii, int jj, int kk) {
    return in[stencilIdx(ii, jj, kk, ny, nz)];
  };
  out[stencilIdx(i, j, k, ny, nz)] =
      alpha * INS(i, j, k) +
      beta3D * (INS(i - 1, j, k) + INS(i + 1, j, k) + INS(i, j - 1, k) +
                INS(i, j + 1, k) + INS(i, j, k - 1) + INS(i, j, k + 1));
}

/** One smoothing pass of the N fields at the points of `points` (a PointBox
 *  for All and Interior, the PointList of the three shell slabs for Shell),
 *  with the arithmetic of d_smoothStep. */
template <int N, typename Points>
__global__ void k_smoothFields(SmoothFields<N> f, int ny, int nz,
                               cudaSolverType alpha, cudaSolverType beta3D,
                               Points points) {
  int i, j, k;
  if (!pointOf(points, i, j, k))
    return;
#pragma unroll
  for (int c = 0; c < N; c++)
    d_smoothStep(f.out[c], f.in[c], i, j, k, ny, nz, alpha, beta3D);
}

template <int N>
void gpuSmoothFields(const SmoothFields<N>& fields, int nx, int ny, int nz,
                     cudaSolverType alpha, cudaSolverType beta3D,
                     SmoothRegion region, cudaStream_t stream) {
  const PointBox all = {pointRange(1, nx - 2), pointRange(1, ny - 2),
                        pointRange(1, nz - 2)};
  const PointBox interior = {pointRange(2, nx - 3), pointRange(2, ny - 3),
                             pointRange(2, nz - 3)};
  auto launchBox = [&](const PointBox& box) {
    if (box.empty())
      return;
    const PointLaunch L = pointLaunch(box);
    k_smoothFields<N, PointBox>
        <<<L.grid, L.block, 0, stream>>>(fields, ny, nz, alpha, beta3D, box);
    cudaErrChk(cudaGetLastError());
  };
  if (region == SmoothRegion::All ||
      (region == SmoothRegion::Shell && interior.empty())) {
    launchBox(all);
  } else if (region == SmoothRegion::Interior) {
    launchBox(interior);
  } else {
    // All minus Interior: the two x layers whole, the two y layers inside
    // the x range of Interior, the two z layers inside its x and y ranges.
    // Interior is not empty, so each axis has at least 3 points and the two
    // layers of a run are distinct.
    const PointBox slabs[3] = {
        {pointRuns(1, 1, nx - 2, nx - 2), all.j, all.k},
        {interior.i, pointRuns(1, 1, ny - 2, ny - 2), all.k},
        {interior.i, interior.j, pointRuns(1, 1, nz - 2, nz - 2)}};
    const PointList shell = pointList(slabs, 3);
    const PointLaunch L = pointLaunch(shell);
    k_smoothFields<N, PointList>
        <<<L.grid, L.block, 0, stream>>>(fields, ny, nz, alpha, beta3D, shell);
    cudaErrChk(cudaGetLastError());
  }
}

template void gpuSmoothFields<1>(const SmoothFields<1>&, int, int, int,
                                 cudaSolverType, cudaSolverType, SmoothRegion,
                                 cudaStream_t);
template void gpuSmoothFields<3>(const SmoothFields<3>&, int, int, int,
                                 cudaSolverType, cudaSolverType, SmoothRegion,
                                 cudaStream_t);
