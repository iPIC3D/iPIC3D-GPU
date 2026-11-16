#ifndef GPU_STENCILS_CUH
#define GPU_STENCILS_CUH

/**
 * @file GPUStencils.cuh
 * @brief GPU stencil kernels for the iPIC3D GPU field solver.
 *
 * Node and cell stencils: gradC2N, gradN2C, divN2C, divC2N, curlC2N,
 *   curlN2C, interpC2N, interpN2C, divSymmTensorN2C, and the 7-point
 *   smoothing pass of 1 or 3 fields (gpuSmoothFields).
 *
 * All kernels operate on flat device pointers (row-major [nx][ny][nz]).
 * Grid dimensions include ghost cells.
 */

#include "FieldBoundary.h" // ImageWalls
#include "cudaTypeDef.cuh"

// =========================================================================
//  Gradient operations
// =========================================================================

/**
 * Gradient center→node:  gradX/Y/Z_N = ∇(scFieldC).
 * Output on node grid [nxn][nyn][nzn], interior i=1..nxn-2.
 * Input on center grid [nxc][nyc][nzc] (nxc=nxn-1, etc.).
 * Stencil: 8-point finite difference.
 */
void gpuGradC2N(cudaSolverType* gradXN, cudaSolverType* gradYN,
                cudaSolverType* gradZN, const cudaSolverType* scFieldC, int nxn,
                int nyn, int nzn, cudaSolverType invdx, cudaSolverType invdy,
                cudaSolverType invdz, cudaStream_t stream = 0);

/**
 * Gradient node→center:  gradX/Y/Z_C = ∇(scFieldN).
 * Output on center grid [nxc][nyc][nzc]: i=iLo..iHi (iHi < 0: nxc-2, i.e.
 * the interior), j and k interior. iHi = nxc-1 also evaluates the X-right
 * ghost cell plane from the ghost node plane.
 * Input on node grid [nxn][nyn][nzn] (nxn=nxc+1, etc.).
 * Not used currently: the Maxwell images evaluate it inside
 * gpuMaxwellLocalCenterOps (the unit tests use it as reference).
 */
void gpuGradN2C(cudaSolverType* gradXC, cudaSolverType* gradYC,
                cudaSolverType* gradZC, const cudaSolverType* scFieldN, int nxc,
                int nyc, int nzc, cudaSolverType invdx, cudaSolverType invdy,
                cudaSolverType invdz, cudaStream_t stream = 0, int iLo = 1,
                int iHi = -1);

// =========================================================================
//  Divergence operations
// =========================================================================

/**
 * Divergence node→center:  divC = ∇·(vecX_N, vecY_N, vecZ_N).
 * Output on center grid [nxc][nyc][nzc]: i=iLo..iHi as for gpuGradN2C.
 * Input on node grid [nxn][nyn][nzn].
 */
void gpuDivN2C(cudaSolverType* divC, const cudaSolverType* vecXN,
               const cudaSolverType* vecYN, const cudaSolverType* vecZN,
               int nxc, int nyc, int nzc, cudaSolverType invdx,
               cudaSolverType invdy, cudaSolverType invdz,
               cudaStream_t stream = 0, int iLo = 1, int iHi = -1);

/**
 * Divergence center→node:  divN = ∇·(vecX_C, vecY_C, vecZ_C).
 * Output on node grid [nxn][nyn][nzn], interior i=1..nxn-2.
 * Input on center grid [nxc][nyc][nzc].
 * Not used currently: the Maxwell images evaluate it inside
 * gpuMaxwellLocalNodeFused (the unit tests use it as reference).
 */
void gpuDivC2N(cudaSolverType* divN, const cudaSolverType* vecXC,
               const cudaSolverType* vecYC, const cudaSolverType* vecZC,
               int nxn, int nyn, int nzn, cudaSolverType invdx,
               cudaSolverType invdy, cudaSolverType invdz,
               cudaStream_t stream = 0);

// =========================================================================
//  Curl operations
// =========================================================================

/**
 * Curl center→node:  curl_N = ∇×(vec_C).
 * Output on node grid [nxn][nyn][nzn], interior i=1..nxn-2.
 * Input on center grid [nxc][nyc][nzc].
 */
void gpuCurlC2N(cudaSolverType* curlXN, cudaSolverType* curlYN,
                cudaSolverType* curlZN, const cudaSolverType* vecXC,
                const cudaSolverType* vecYC, const cudaSolverType* vecZC,
                int nxn, int nyn, int nzn, cudaSolverType invdx,
                cudaSolverType invdy, cudaSolverType invdz,
                cudaStream_t stream = 0);

/**
 * Curl node→center:  curl_C = ∇×(vec_N).
 * Output on center grid [nxc][nyc][nzc], interior i=1..nxc-2.
 * Input on node grid [nxn][nyn][nzn].
 */
void gpuCurlN2C(cudaSolverType* curlXC, cudaSolverType* curlYC,
                cudaSolverType* curlZC, const cudaSolverType* vecXN,
                const cudaSolverType* vecYN, const cudaSolverType* vecZN,
                int nxc, int nyc, int nzc, cudaSolverType invdx,
                cudaSolverType invdy, cudaSolverType invdz,
                cudaStream_t stream = 0);

// =========================================================================
//  Interpolation operations
// =========================================================================

/**
 * Interpolate center→node:  fieldN = interp(fieldC).
 * 8-point average from surrounding center cells.
 * Output on node grid, interior i=1..nxn-2.
 */
void gpuInterpC2N(cudaSolverType* fieldN, const cudaSolverType* fieldC, int nxn,
                  int nyn, int nzn, cudaStream_t stream = 0);

/**
 * Interpolate node→center:  fieldC = interp(fieldN).
 * 8-point average from surrounding nodes.
 * Output on center grid, interior i=1..nxc-2.  walls: the face nodes of
 * these conductor walls read as 0, the image of a charge-like scalar.
 */
void gpuInterpN2C(cudaSolverType* fieldC, const cudaSolverType* fieldN, int nxc,
                  int nyc, int nzc, cudaStream_t stream = 0,
                  const ImageWalls& walls = ImageWalls());

// =========================================================================
//  Symmetric tensor divergence
// =========================================================================

/**
 * Divergence of a symmetric tensor, node→center.
 *   divCX = d(pXX)/dx + d(pXY)/dy + d(pXZ)/dz
 *   divCY = d(pXY)/dx + d(pYY)/dy + d(pYZ)/dz
 *   divCZ = d(pXZ)/dx + d(pYZ)/dy + d(pZZ)/dz
 *
 * The caller passes the species slices of the tensor components. Output on
 * center grid, interior i=1..nxc-2.
 */
void gpuDivSymmTensorN2C(cudaSolverType* divCX, cudaSolverType* divCY,
                         cudaSolverType* divCZ, const cudaSolverType* pXX,
                         const cudaSolverType* pXY, const cudaSolverType* pXZ,
                         const cudaSolverType* pYY, const cudaSolverType* pYZ,
                         const cudaSolverType* pZZ, int nxc, int nyc, int nzc,
                         cudaSolverType invdx, cudaSolverType invdy,
                         cudaSolverType invdz, cudaStream_t stream = 0);

// =========================================================================
//  Interior / boundary stencil variants of the overlapped halo exchanges
//  (HaloMode::Overlap).
//  Interior covers [2..n-3] (no ghost dependency).
//  Boundary covers [1..n-2] minus the interior block.
//  The divC2N and gradC2N variants are not used currently (the Maxwell
//  image evaluates both in gpuMaxwellLocalNodeFused).
// =========================================================================

void gpuDivC2N_interior(cudaSolverType* divN, const cudaSolverType* vecXC,
                        const cudaSolverType* vecYC,
                        const cudaSolverType* vecZC, int nxn, int nyn, int nzn,
                        cudaSolverType invdx, cudaSolverType invdy,
                        cudaSolverType invdz, cudaStream_t stream = 0);
void gpuDivC2N_boundary(cudaSolverType* divN, const cudaSolverType* vecXC,
                        const cudaSolverType* vecYC,
                        const cudaSolverType* vecZC, int nxn, int nyn, int nzn,
                        cudaSolverType invdx, cudaSolverType invdy,
                        cudaSolverType invdz, cudaStream_t stream = 0);

void gpuGradC2N_interior(cudaSolverType* gradXN, cudaSolverType* gradYN,
                         cudaSolverType* gradZN, const cudaSolverType* scFieldC,
                         int nxn, int nyn, int nzn, cudaSolverType invdx,
                         cudaSolverType invdy, cudaSolverType invdz,
                         cudaStream_t stream = 0);
void gpuGradC2N_boundary(cudaSolverType* gradXN, cudaSolverType* gradYN,
                         cudaSolverType* gradZN, const cudaSolverType* scFieldC,
                         int nxn, int nyn, int nzn, cudaSolverType invdx,
                         cudaSolverType invdy, cudaSolverType invdz,
                         cudaStream_t stream = 0);

void gpuInterpC2N_interior(cudaSolverType* fieldN, const cudaSolverType* fieldC,
                           int nxn, int nyn, int nzn, cudaStream_t stream = 0);
void gpuInterpC2N_boundary(cudaSolverType* fieldN, const cudaSolverType* fieldC,
                           int nxn, int nyn, int nzn, cudaStream_t stream = 0);

/** Interpolate a caller-supplied inclusive node box.  Bounds are clamped to
 *  the active node domain [1..n-2]; an empty box is a no-op. */
void gpuInterpC2N_range(cudaSolverType* fieldN, const cudaSolverType* fieldC,
                        int nxn, int nyn, int nzn, int iLo, int iHi, int jLo,
                        int jHi, int kLo, int kHi, cudaStream_t stream = 0);

/** Interpolate the active node domain except for a caller-supplied inclusive
 *  safe box.  The complement is mapped as six disjoint boundary slabs.  If the
 *  clamped safe box is empty, interpolates the full active domain
 *  (gpuInterpC2N_boundary: the complement of the interior box). */
void gpuInterpC2N_complement(cudaSolverType* fieldN,
                             const cudaSolverType* fieldC, int nxn, int nyn,
                             int nzn, int safeILo, int safeIHi, int safeJLo,
                             int safeJHi, int safeKLo, int safeKHi,
                             cudaStream_t stream = 0);

// =========================================================================
//  Smoothing pass of N fields in one launch (EMfields3D::gpuSmoothPasses)
// =========================================================================

/** The points of a smoothing pass: All = [1..n-2]^3 (blocking halo
 *  exchange); Interior = [2..n-3]^3, which reads no ghost (while the exchange
 *  is in flight); Shell = All minus Interior (after it), launched as its
 *  three slabs, or as All when Interior is empty (a flat or thin axis). */
enum class SmoothRegion { All, Interior, Shell };

/** The output and input fields of one smoothing pass (N = 1 or 3). */
template <int N> struct SmoothFields {
  cudaSolverType* out[N];
  const cudaSolverType* in[N];
};

/** One 7-point smoothing pass of the N fields over region, in one launch:
 *    out = alpha * in + beta3D * (6 neighbours of in).
 *  The ghosts of in must be valid for the points that read them; out and in
 *  must not alias. */
template <int N>
void gpuSmoothFields(const SmoothFields<N>& fields, int nx, int ny, int nz,
                     cudaSolverType alpha, cudaSolverType beta3D,
                     SmoothRegion region, cudaStream_t stream = 0);

// =========================================================================
//  Device stencils at one point, shared by the kernels of GPUStencils.cu and
//  the fused kernels of the divergence cleaning (GPUCleaning.cu).  Each
//  difference along an axis is averaged over the 4 pairs of the other two.
// =========================================================================

/** The factors q = 0.25 / h of the three axes, computed once on the host
 *  (0.25 * invd is exact): an averaged difference is q * (sum of the 4). */
template <typename T> struct DiffScale {
  T x, y, z;
};
template <typename T> inline DiffScale<T> diffScale(T invdx, T invdy, T invdz) {
  return {0.25 * invdx, 0.25 * invdy, 0.25 * invdz};
}

#if defined(__CUDACC__) || defined(__HIPCC__)
/** Row-major [i][j][k] index of an array with extents (., ny, nz). */
__host__ __device__ __forceinline__ int stencilIdx(int i, int j, int k, int ny,
                                                   int nz) {
  return (i * ny + j) * nz + k;
}

/** gradC2N at node (i, j, k) of the cell array C: the centres (i-1..i,
 *  j-1..j, k-1..k). */
template <typename T>
__device__ __forceinline__ void
gradC2NAt(const T* __restrict__ C, int i, int j, int k, int nyc, int nzc,
          const DiffScale<T>& q, T& gx, T& gy, T& gz) {
  const auto c = [&](int ii, int jj, int kk) {
    return C[stencilIdx(ii, jj, kk, nyc, nzc)];
  };
  const T c000 = c(i - 1, j - 1, k - 1), c001 = c(i - 1, j - 1, k);
  const T c010 = c(i - 1, j, k - 1), c011 = c(i - 1, j, k);
  const T c100 = c(i, j - 1, k - 1), c101 = c(i, j - 1, k);
  const T c110 = c(i, j, k - 1), c111 = c(i, j, k);
  gx = q.x * ((c111 - c011) + (c110 - c010) + (c101 - c001) + (c100 - c000));
  gy = q.y * ((c111 - c101) + (c110 - c100) + (c011 - c001) + (c010 - c000));
  gz = q.z * ((c111 - c110) + (c101 - c100) + (c011 - c010) + (c001 - c000));
}

/** gradN2C at centre (i, j, k): the nodes (i..i+1, j..j+1, k..k+1). */
template <typename T>
__device__ __forceinline__ void
gradN2CAt(const T* __restrict__ N, int i, int j, int k, int nyn, int nzn,
          const DiffScale<T>& q, T& gx, T& gy, T& gz) {
  const auto n = [&](int ii, int jj, int kk) {
    return N[stencilIdx(ii, jj, kk, nyn, nzn)];
  };
  const T n000 = n(i, j, k), n001 = n(i, j, k + 1);
  const T n010 = n(i, j + 1, k), n011 = n(i, j + 1, k + 1);
  const T n100 = n(i + 1, j, k), n101 = n(i + 1, j, k + 1);
  const T n110 = n(i + 1, j + 1, k), n111 = n(i + 1, j + 1, k + 1);
  gx = q.x * ((n100 - n000) + (n101 - n001) + (n110 - n010) + (n111 - n011));
  gy = q.y * ((n010 - n000) + (n011 - n001) + (n110 - n100) + (n111 - n101));
  gz = q.z * ((n001 - n000) + (n101 - n100) + (n011 - n010) + (n111 - n110));
}

/** The conductor face planes among the node planes of cell (i, j, k):
 *  on[e][s] for its low (s = 0) and high (s = 1) node plane along axis e.
 *  The default has none: the stencils below are then the plain ones (the
 *  compiler folds the constant selects). */
struct ImagePlanes {
  bool on[3][2] = {};
};
__device__ __forceinline__ ImagePlanes imagePlanes(const ImageWalls& w, int i,
                                                   int j, int k) {
  ImagePlanes p;
  p.on[0][0] = i == w.lo[0];
  p.on[0][1] = i == w.hi[0];
  p.on[1][0] = j == w.lo[1];
  p.on[1][1] = j == w.hi[1];
  p.on[2][0] = k == w.lo[2];
  p.on[2][1] = k == w.hi[2];
  return p;
}

/** divN2C at centre (i, j, k): the nodes (i..i+1, j..j+1, k..k+1), each
 *  difference along an axis averaged over the 4 edges of the cell.  On the
 *  conductor planes p a polar vector is read through the image: a component
 *  is 0 on a face plane tangential to it, so a difference along an edge that
 *  lies in such a plane is 0. */
template <typename T>
__device__ __forceinline__ T divN2CAt(const T* __restrict__ X,
                                      const T* __restrict__ Y,
                                      const T* __restrict__ Z, int i, int j,
                                      int k, int nyn, int nzn,
                                      const DiffScale<T>& q,
                                      const ImagePlanes& p = ImagePlanes()) {
  const bool (&x)[2] = p.on[0];
  const bool (&y)[2] = p.on[1];
  const bool (&z)[2] = p.on[2];
  const auto n = [&](const T* a, int ii, int jj, int kk) {
    return a[stencilIdx(ii, jj, kk, nyn, nzn)];
  };
  const auto d = [](bool inWall, T hi, T lo) {
    return inWall ? T(0) : hi - lo;
  };
  const T cx =
      q.x * (d(y[0] || z[0], n(X, i + 1, j, k), n(X, i, j, k)) +
             d(y[0] || z[1], n(X, i + 1, j, k + 1), n(X, i, j, k + 1)) +
             d(y[1] || z[0], n(X, i + 1, j + 1, k), n(X, i, j + 1, k)) +
             d(y[1] || z[1], n(X, i + 1, j + 1, k + 1), n(X, i, j + 1, k + 1)));
  const T cy =
      q.y * (d(x[0] || z[0], n(Y, i, j + 1, k), n(Y, i, j, k)) +
             d(x[0] || z[1], n(Y, i, j + 1, k + 1), n(Y, i, j, k + 1)) +
             d(x[1] || z[0], n(Y, i + 1, j + 1, k), n(Y, i + 1, j, k)) +
             d(x[1] || z[1], n(Y, i + 1, j + 1, k + 1), n(Y, i + 1, j, k + 1)));
  const T cz =
      q.z * (d(x[0] || y[0], n(Z, i, j, k + 1), n(Z, i, j, k)) +
             d(x[1] || y[0], n(Z, i + 1, j, k + 1), n(Z, i + 1, j, k)) +
             d(x[0] || y[1], n(Z, i, j + 1, k + 1), n(Z, i, j + 1, k)) +
             d(x[1] || y[1], n(Z, i + 1, j + 1, k + 1), n(Z, i + 1, j + 1, k)));
  return cx + cy + cz;
}

/** interpN2C, the mean of the 8 nodes, at centre (i, j, k); on the
 *  conductor planes p a charge-like scalar is read through the image, as
 *  0. */
template <typename T>
__device__ __forceinline__ T interpN2CAt(const T* __restrict__ N, int i, int j,
                                         int k, int nyn, int nzn,
                                         const ImagePlanes& p = ImagePlanes()) {
  const auto n = [&](int a, int b, int c) {
    return p.on[0][a] || p.on[1][b] || p.on[2][c]
               ? T(0)
               : N[stencilIdx(i + a, j + b, k + c, nyn, nzn)];
  };
  return 0.125 * (n(0, 0, 0) + n(1, 0, 0) + n(0, 1, 0) + n(0, 0, 1) +
                  n(1, 1, 0) + n(1, 0, 1) + n(0, 1, 1) + n(1, 1, 1));
}

/** divC2N at node (i, j, k): the centres (i-1..i, j-1..j, k-1..k).  Not
 *  built on gradC2NAt: its z sum is in another order. */
template <typename T>
__device__ __forceinline__ T divC2NAt(const T* __restrict__ X,
                                      const T* __restrict__ Y,
                                      const T* __restrict__ Z, int i, int j,
                                      int k, int nyc, int nzc,
                                      const DiffScale<T>& q) {
  const auto c = [&](const T* a, int ii, int jj, int kk) {
    return a[stencilIdx(ii, jj, kk, nyc, nzc)];
  };
  const T cx = q.x * ((c(X, i, j, k) - c(X, i - 1, j, k)) +
                      (c(X, i, j, k - 1) - c(X, i - 1, j, k - 1)) +
                      (c(X, i, j - 1, k) - c(X, i - 1, j - 1, k)) +
                      (c(X, i, j - 1, k - 1) - c(X, i - 1, j - 1, k - 1)));
  const T cy = q.y * ((c(Y, i, j, k) - c(Y, i, j - 1, k)) +
                      (c(Y, i, j, k - 1) - c(Y, i, j - 1, k - 1)) +
                      (c(Y, i - 1, j, k) - c(Y, i - 1, j - 1, k)) +
                      (c(Y, i - 1, j, k - 1) - c(Y, i - 1, j - 1, k - 1)));
  const T cz = q.z * ((c(Z, i, j, k) - c(Z, i, j, k - 1)) +
                      (c(Z, i - 1, j, k) - c(Z, i - 1, j, k - 1)) +
                      (c(Z, i, j - 1, k) - c(Z, i, j - 1, k - 1)) +
                      (c(Z, i - 1, j - 1, k) - c(Z, i - 1, j - 1, k - 1)));
  return cx + cy + cz;
}
#endif

#endif // GPU_STENCILS_CUH
