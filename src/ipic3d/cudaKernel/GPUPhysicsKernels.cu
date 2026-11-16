/**
 * @file GPUPhysicsKernels.cu
 * @brief CUDA kernel implementations for iPIC3D physics operations:
 *        MUdot, PIdot, planet-charge, Laplacian and block-Jacobi kernels.
 *
 * The MUdot, PIdot, block-Jacobi and Poisson-image Laplacian kernels use the
 * launch shape of GPUPointLaunch.cuh; the others flat 1D launches.
 */

#include "GPUPhysicsKernels.cuh"
#include "GPUPointLaunch.cuh"
#include "GPUStencils.cuh"

// =========================================================================
//  MUdot kernel
// =========================================================================

/** One species of MUdot.  Not used (gpuMUdotSpecies, a unit-test reference:
 *  gpuMUdot treats all species in one kernel). */
template <typename T>
__global__ void
k_MUdotSpecies(T* __restrict__ MUdotX, T* __restrict__ MUdotY,
               T* __restrict__ MUdotZ, const T* __restrict__ vX,
               const T* __restrict__ vY, const T* __restrict__ vZ,
               const T* __restrict__ Bxn, const T* __restrict__ Byn,
               const T* __restrict__ Bzn, const T* __restrict__ Bx_ext,
               const T* __restrict__ By_ext, const T* __restrict__ Bz_ext,
               const T* __restrict__ rhons_is, T beta, T prefactor, int nxn,
               int nyn, int nzn, bool firstSpecies) {
  int i = blockIdx.z * blockDim.z + threadIdx.z + 1;
  int j = blockIdx.y * blockDim.y + threadIdx.y + 1;
  int k = blockIdx.x * blockDim.x + threadIdx.x + 1;
  if (i > nxn - 2 || j > nyn - 2 || k > nzn - 2)
    return;

  int idx = stencilIdx(i, j, k, nyn, nzn);

  T omcx = beta * (Bxn[idx] + Bx_ext[idx]);
  T omcy = beta * (Byn[idx] + By_ext[idx]);
  T omcz = beta * (Bzn[idx] + Bz_ext[idx]);

  T vx = vX[idx], vy = vY[idx], vz = vZ[idx];
  T edotb = vx * omcx + vy * omcy + vz * omcz;
  T denom = prefactor * rhons_is[idx] /
            (1.0 + omcx * omcx + omcy * omcy + omcz * omcz);

  T dx = (vx + (vy * omcz - vz * omcy + edotb * omcx)) * denom;
  T dy = (vy + (vz * omcx - vx * omcz + edotb * omcy)) * denom;
  T dz = (vz + (vx * omcy - vy * omcx + edotb * omcz)) * denom;

  if (firstSpecies) {
    MUdotX[idx] = dx;
    MUdotY[idx] = dy;
    MUdotZ[idx] = dz;
  } else {
    MUdotX[idx] += dx;
    MUdotY[idx] += dy;
    MUdotZ[idx] += dz;
  }
}

void gpuMUdotSpecies(cudaSolverType* MUdotX, cudaSolverType* MUdotY,
                     cudaSolverType* MUdotZ, const cudaSolverType* vX,
                     const cudaSolverType* vY, const cudaSolverType* vZ,
                     const cudaSolverType* Bxn, const cudaSolverType* Byn,
                     const cudaSolverType* Bzn, const cudaSolverType* Bx_ext,
                     const cudaSolverType* By_ext, const cudaSolverType* Bz_ext,
                     const cudaSolverType* rhons_is, cudaSolverType beta,
                     cudaSolverType prefactor, int nxn, int nyn, int nzn,
                     bool firstSpecies, cudaStream_t stream) {
  const PointLaunch L = pointLaunch(nxn - 2, nyn - 2, nzn - 2);
  k_MUdotSpecies<<<L.grid, L.block, 0, stream>>>(
      MUdotX, MUdotY, MUdotZ, vX, vY, vZ, Bxn, Byn, Bzn, Bx_ext, By_ext, Bz_ext,
      rhons_is, beta, prefactor, nxn, nyn, nzn, firstSpecies);
  cudaErrChk(cudaGetLastError());
}

// =========================================================================
//  MUdot of all species in one kernel
// =========================================================================

/** Per-species constants of one MUdot launch, passed by value (constant
 *  bank). */
template <typename T> struct MUdotSpeciesSet {
  int count;                 // species in this set
  T scale[kMUdotMaxSpecies]; // (4π/2) delt dt/c (q/m)_s
  T beta[kMUdotMaxSpecies];  // (q/m)_s dt / (2c)
  T beta2[kMUdotMaxSpecies]; // beta_s^2
};

/** μ·v = A v + C1 v×b + C2 (v·b) b over the species of set (gpuMUdot).
 *  NS > 0 is the species count at compile time (loop fully unrolled); NS = 0
 *  reads it from set.count.  kAccumulate adds to the output (species sets
 *  after the first). */
template <typename T, int NS, bool kAccumulate, typename Points>
__global__ void
k_MUdot(T* __restrict__ MUdotX, T* __restrict__ MUdotY, T* __restrict__ MUdotZ,
        const T* __restrict__ vX, const T* __restrict__ vY,
        const T* __restrict__ vZ, const T* __restrict__ Bxn,
        const T* __restrict__ Byn, const T* __restrict__ Bzn,
        const T* __restrict__ Bx_ext, const T* __restrict__ By_ext,
        const T* __restrict__ Bz_ext, const T* __restrict__ rhons,
        size_t speciesStride, MUdotSpeciesSet<T> set, int nyn, int nzn,
        Points points) {
  int i, j, k;
  if (!pointOf(points, i, j, k))
    return;

  const int idx = stencilIdx(i, j, k, nyn, nzn);
  const T bx = Bxn[idx] + Bx_ext[idx];
  const T by = Byn[idx] + By_ext[idx];
  const T bz = Bzn[idx] + Bz_ext[idx];
  const T b2 = bx * bx + by * by + bz * bz;

  // species sums A, C1, C2 (they depend on b and rho only)
  T a = 0, c1 = 0, c2 = 0;
  const int count = NS > 0 ? NS : set.count;
#pragma unroll
  for (int s = 0; s < count; s++) {
    const T d = set.scale[s] * rhons[s * speciesStride + idx] /
                (T(1) + set.beta2[s] * b2);
    a += d;
    c1 += d * set.beta[s];
    c2 += d * set.beta2[s];
  }

  const T vx = vX[idx], vy = vY[idx], vz = vZ[idx];
  const T c2vb = c2 * (vx * bx + vy * by + vz * bz);
  const T dx = a * vx + c1 * (vy * bz - vz * by) + c2vb * bx;
  const T dy = a * vy + c1 * (vz * bx - vx * bz) + c2vb * by;
  const T dz = a * vz + c1 * (vx * by - vy * bx) + c2vb * bz;
  if constexpr (kAccumulate) {
    MUdotX[idx] += dx;
    MUdotY[idx] += dy;
    MUdotZ[idx] += dz;
  } else {
    MUdotX[idx] = dx;
    MUdotY[idx] = dy;
    MUdotZ[idx] = dz;
  }
}

/** The pointer and size arguments shared by every MUdot launch. */
struct MUdotArgs {
  cudaSolverType *MUdotX, *MUdotY, *MUdotZ;
  const cudaSolverType *vX, *vY, *vZ, *Bxn, *Byn, *Bzn, *Bx_ext, *By_ext,
      *Bz_ext, *rhons;
  size_t speciesStride;
  int nyn, nzn;
};

template <int NS, bool kAccumulate, typename Points>
static void
launchMUdot(const PointLaunch& L, const MUdotArgs& a, const Points& points,
            const MUdotSpeciesSet<cudaSolverType>& set, cudaStream_t stream) {
  k_MUdot<cudaSolverType, NS, kAccumulate, Points>
      <<<L.grid, L.block, 0, stream>>>(
          a.MUdotX, a.MUdotY, a.MUdotZ, a.vX, a.vY, a.vZ, a.Bxn, a.Byn, a.Bzn,
          a.Bx_ext, a.By_ext, a.Bz_ext, a.rhons, a.speciesStride, set, a.nyn,
          a.nzn, points);
  cudaErrChk(cudaGetLastError());
}

template <typename Points>
void gpuMUdot(cudaSolverType* MUdotX, cudaSolverType* MUdotY,
              cudaSolverType* MUdotZ, const cudaSolverType* vX,
              const cudaSolverType* vY, const cudaSolverType* vZ,
              const cudaSolverType* Bxn, const cudaSolverType* Byn,
              const cudaSolverType* Bzn, const cudaSolverType* Bx_ext,
              const cudaSolverType* By_ext, const cudaSolverType* Bz_ext,
              const cudaSolverType* rhons, const double* qom, int ns, double dt,
              double c, double FourPI, double delt, int nxn, int nyn, int nzn,
              const Points& points, cudaStream_t stream) {
  if (points.empty())
    return;
  const PointLaunch L = pointLaunch(points);
  const size_t stride = (size_t)nxn * nyn * nzn;
  MUdotArgs a{MUdotX, MUdotY, MUdotZ, vX,     vY,    vZ,     Bxn, Byn,
              Bzn,    Bx_ext, By_ext, Bz_ext, rhons, stride, nyn, nzn};
  // Species sets of up to kMUdotMaxSpecies: the first writes, the others add.
  // With ns = 0 one launch with an empty set writes zero.
  for (int first = 0; first == 0 || first < ns; first += kMUdotMaxSpecies) {
    MUdotSpeciesSet<cudaSolverType> set{};
    set.count = ns - first < kMUdotMaxSpecies ? ns - first : kMUdotMaxSpecies;
    for (int s = 0; s < set.count; s++) {
      const double q = qom[first + s];
      const double beta = 0.5 * q * dt / c;
      set.scale[s] = FourPI / 2.0 * delt * dt / c * q;
      set.beta[s] = beta;
      set.beta2[s] = beta * beta;
    }
    a.rhons = rhons + first * stride;
    if (first > 0)
      launchMUdot<0, true>(L, a, points, set, stream);
    else if (set.count == 1)
      launchMUdot<1, false>(L, a, points, set, stream);
    else if (set.count == 2)
      launchMUdot<2, false>(L, a, points, set, stream);
    else if (set.count == 3)
      launchMUdot<3, false>(L, a, points, set, stream);
    else if (set.count == 4)
      launchMUdot<4, false>(L, a, points, set, stream);
    else
      launchMUdot<0, false>(L, a, points, set, stream);
  }
}

// The two point sets: PointList (1D launch, the images) and PointBox (3D
// launch, the unit tests)
#define MUDOT_INSTANTIATE(Points)                                              \
  template void gpuMUdot<Points>(                                              \
      cudaSolverType*, cudaSolverType*, cudaSolverType*,                       \
      const cudaSolverType*, const cudaSolverType*, const cudaSolverType*,     \
      const cudaSolverType*, const cudaSolverType*, const cudaSolverType*,     \
      const cudaSolverType*, const cudaSolverType*, const cudaSolverType*,     \
      const cudaSolverType*, const double*, int, double, double, double,       \
      double, int, int, int, const Points&, cudaStream_t);
MUDOT_INSTANTIATE(PointBox)
MUDOT_INSTANTIATE(PointList)
#undef MUDOT_INSTANTIATE

// =========================================================================
//  PIdot kernel
// =========================================================================

template <typename T>
__global__ void
k_PIdotSpecies(T* __restrict__ PIX, T* __restrict__ PIY, T* __restrict__ PIZ,
               const T* __restrict__ vX, const T* __restrict__ vY,
               const T* __restrict__ vZ, const T* __restrict__ Bxn,
               const T* __restrict__ Byn, const T* __restrict__ Bzn,
               const T* __restrict__ Bx_ext, const T* __restrict__ By_ext,
               const T* __restrict__ Bz_ext, T beta, int nxn, int nyn,
               int nzn) {
  int i = blockIdx.z * blockDim.z + threadIdx.z + 1;
  int j = blockIdx.y * blockDim.y + threadIdx.y + 1;
  int k = blockIdx.x * blockDim.x + threadIdx.x + 1;
  if (i > nxn - 2 || j > nyn - 2 || k > nzn - 2)
    return;

  int idx = stencilIdx(i, j, k, nyn, nzn);

  T omcx = beta * (Bxn[idx] + Bx_ext[idx]);
  T omcy = beta * (Byn[idx] + By_ext[idx]);
  T omcz = beta * (Bzn[idx] + Bz_ext[idx]);

  T vx = vX[idx], vy = vY[idx], vz = vZ[idx];
  T edotb = vx * omcx + vy * omcy + vz * omcz;
  T denom = 1.0 / (1.0 + omcx * omcx + omcy * omcy + omcz * omcz);

  PIX[idx] += (vx + (vy * omcz - vz * omcy + edotb * omcx)) * denom;
  PIY[idx] += (vy + (vz * omcx - vx * omcz + edotb * omcy)) * denom;
  PIZ[idx] += (vz + (vx * omcy - vy * omcx + edotb * omcz)) * denom;
}

void gpuPIdotSpecies(cudaSolverType* PIX, cudaSolverType* PIY,
                     cudaSolverType* PIZ, const cudaSolverType* vX,
                     const cudaSolverType* vY, const cudaSolverType* vZ,
                     const cudaSolverType* Bxn, const cudaSolverType* Byn,
                     const cudaSolverType* Bzn, const cudaSolverType* Bx_ext,
                     const cudaSolverType* By_ext, const cudaSolverType* Bz_ext,
                     cudaSolverType beta, int nxn, int nyn, int nzn,
                     cudaStream_t stream) {
  const PointLaunch L = pointLaunch(nxn - 2, nyn - 2, nzn - 2);
  k_PIdotSpecies<<<L.grid, L.block, 0, stream>>>(PIX, PIY, PIZ, vX, vY, vZ, Bxn,
                                                 Byn, Bzn, Bx_ext, By_ext,
                                                 Bz_ext, beta, nxn, nyn, nzn);
  cudaErrChk(cudaGetLastError());
}

// =========================================================================
//  ConstantChargePlanet kernels
// =========================================================================

/** Set rhons = val inside sphere of radius R centred at (xc,yc,zc). */
template <typename T>
__global__ void k_constantChargePlanet(T* rhons, T val, T R2, T xc, T yc, T zc,
                                       T xStart, T yStart, T zStart, T dx, T dy,
                                       T dz, int nxn, int nyn, int nzn) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int ni = nxn - 1, nj = nyn - 1, nk = nzn - 1;
  if (tid >= ni * nj * nk)
    return;
  int i = tid / (nj * nk) + 1;
  int rem = tid % (nj * nk);
  int j = rem / nk + 1;
  int k = rem % nk + 1;
  T xd = xStart + (T)(i - 1) * dx - xc;
  T yd = yStart + (T)(j - 1) * dy - yc;
  T zd = zStart + (T)(k - 1) * dz - zc;
  if (xd * xd + yd * yd + zd * zd <= R2)
    rhons[(i * nyn + j) * nzn + k] = val;
}

void gpuConstantChargePlanetKernel(cudaSolverType* rhons, cudaSolverType val,
                                   cudaSolverType R, cudaSolverType xc,
                                   cudaSolverType yc, cudaSolverType zc,
                                   cudaSolverType xStart, cudaSolverType yStart,
                                   cudaSolverType zStart, cudaSolverType dx,
                                   cudaSolverType dy, cudaSolverType dz,
                                   int nxn, int nyn, int nzn,
                                   cudaStream_t stream) {
  int total = (nxn - 1) * (nyn - 1) * (nzn - 1);
  if (total <= 0)
    return;
  int blk = 256;
  k_constantChargePlanet<<<(total + blk - 1) / blk, blk, 0, stream>>>(
      rhons, val, R * R, xc, yc, zc, xStart, yStart, zStart, dx, dy, dz, nxn,
      nyn, nzn);
  cudaErrChk(cudaGetLastError());
}

/** 2D version: set rhons inside circle in XZ plane (nyn==4). */
template <typename T>
__global__ void k_constantChargePlanet2D(T* rhons, T val, T R2, T xc, T zc,
                                         T xStart, T zStart, T dx, T dz,
                                         int nxn, int nyn, int nzn) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int ni = nxn - 1, nk = nzn - 1;
  if (tid >= ni * nk)
    return;
  int i = tid / nk + 1;
  int k = tid % nk + 1;
  T xd = xStart + (T)(i - 1) * dx - xc;
  T zd = zStart + (T)(k - 1) * dz - zc;
  if (xd * xd + zd * zd <= R2) {
    rhons[(i * nyn + 1) * nzn + k] = val;
    rhons[(i * nyn + 2) * nzn + k] = val;
  }
}

void gpuConstantChargePlanet2DKernel(cudaSolverType* rhons, cudaSolverType val,
                                     cudaSolverType R, cudaSolverType xc,
                                     cudaSolverType zc, cudaSolverType xStart,
                                     cudaSolverType zStart, cudaSolverType dx,
                                     cudaSolverType dz, int nxn, int nyn,
                                     int nzn, cudaStream_t stream) {
  int total = (nxn - 1) * (nzn - 1);
  if (total <= 0)
    return;
  int blk = 256;
  k_constantChargePlanet2D<<<(total + blk - 1) / blk, blk, 0, stream>>>(
      rhons, val, R * R, xc, zc, xStart, zStart, dx, dz, nxn, nyn, nzn);
  cudaErrChk(cudaGetLastError());
}

// =========================================================================
//  Center-to-center 7-point Laplacian (not used currently: the cleaning
//  solves use the 27-point operators of GPUCleaning.cu)
// =========================================================================

/** 7-point center Laplacian of the interior cells, written packed:
 *  lapK[(i nj + j) nk + k] for the interior cell (i+1, j+1, k+1).  Reads the
 *  face ghosts of fC, never the edge or corner ghosts.  Not used
 *  (gpuLapC2CToKrylov). */
template <typename T>
__global__ void k_lapC2CToKrylov(T* __restrict__ lapK, const T* __restrict__ fC,
                                 int nxc, int nyc, int nzc, T invdx2, T invdy2,
                                 T invdz2) {
  const int i = blockIdx.z * blockDim.z + threadIdx.z;
  const int j = blockIdx.y * blockDim.y + threadIdx.y;
  const int k = blockIdx.x * blockDim.x + threadIdx.x;
  const int ni = nxc - 2, nj = nyc - 2, nk = nzc - 2;
  if (i >= ni || j >= nj || k >= nk)
    return;
  const int sX = nyc * nzc; // stride in X direction
  const int sY = nzc;       // stride in Y direction
  const int idx = ((i + 1) * nyc + (j + 1)) * nzc + (k + 1);
  const T f = fC[idx];
  lapK[(i * nj + j) * nk + k] =
      (fC[idx - sX] - 2.0 * f + fC[idx + sX]) * invdx2 +
      (fC[idx - sY] - 2.0 * f + fC[idx + sY]) * invdy2 +
      (fC[idx - 1] - 2.0 * f + fC[idx + 1]) * invdz2;
}

void gpuLapC2CToKrylov(cudaSolverType* lapK, const cudaSolverType* fC, int nxc,
                       int nyc, int nzc, cudaSolverType invdx2,
                       cudaSolverType invdy2, cudaSolverType invdz2,
                       cudaStream_t stream) {
  const PointLaunch L = pointLaunch(nxc - 2, nyc - 2, nzc - 2);
  k_lapC2CToKrylov<<<L.grid, L.block, 0, stream>>>(lapK, fC, nxc, nyc, nzc,
                                                   invdx2, invdy2, invdz2);
  cudaErrChk(cudaGetLastError());
}

/** 7-point center Laplacian: lapC = d²f/dx² + d²f/dy² + d²f/dz².  Used
 *  only by the unit tests (not used). */
template <typename T>
__global__ void k_lapC2C(T* lapC, const T* fC, int nxc, int nyc, int nzc,
                         T invdx2, T invdy2, T invdz2) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int ni = nxc - 2, nj = nyc - 2, nk = nzc - 2;
  if (tid >= ni * nj * nk)
    return;
  int i = tid / (nj * nk) + 1;
  int rem = tid % (nj * nk);
  int j = rem / nk + 1;
  int k = rem % nk + 1;
  int idx = (i * nyc + j) * nzc + k;
  int sX = nyc * nzc; // stride in X direction
  int sY = nzc;       // stride in Y direction
  lapC[idx] = (fC[idx - sX] - 2.0 * fC[idx] + fC[idx + sX]) * invdx2 +
              (fC[idx - sY] - 2.0 * fC[idx] + fC[idx + sY]) * invdy2 +
              (fC[idx - 1] - 2.0 * fC[idx] + fC[idx + 1]) * invdz2;
}

void gpuLapC2CKernel(cudaSolverType* lapC, const cudaSolverType* fC, int nxc,
                     int nyc, int nzc, cudaSolverType invdx2,
                     cudaSolverType invdy2, cudaSolverType invdz2,
                     cudaStream_t stream) {
  int total = (nxc - 2) * (nyc - 2) * (nzc - 2);
  if (total <= 0)
    return;
  int blk = 256;
  k_lapC2C<<<(total + blk - 1) / blk, blk, 0, stream>>>(lapC, fC, nxc, nyc, nzc,
                                                        invdx2, invdy2, invdz2);
  cudaErrChk(cudaGetLastError());
}

/** 7-point center Laplacian with per-face ghost rules (no ghost reads).
 *  A neighbour outside the interior is replaced by g * f(cell), g in
 *  {0, +1, -1}.  With one interior cell both faces act on the same cell.
 *  Not used currently. */
template <typename T>
__global__ void k_lapC2CLocal(T* lapC, const T* fC, int nxc, int nyc, int nzc,
                              T invdx2, T invdy2, T invdz2, T gXl, T gXr, T gYl,
                              T gYr, T gZl, T gZr) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int ni = nxc - 2, nj = nyc - 2, nk = nzc - 2;
  if (tid >= ni * nj * nk)
    return;
  int i = tid / (nj * nk) + 1;
  int rem = tid % (nj * nk);
  int j = rem / nk + 1;
  int k = rem % nk + 1;
  int idx = (i * nyc + j) * nzc + k;
  int sX = nyc * nzc; // stride in X direction
  int sY = nzc;       // stride in Y direction
  T f = fC[idx];
  T xm = (i > 1) ? fC[idx - sX] : gXl * f;
  T xp = (i < ni) ? fC[idx + sX] : gXr * f;
  T ym = (j > 1) ? fC[idx - sY] : gYl * f;
  T yp = (j < nj) ? fC[idx + sY] : gYr * f;
  T zm = (k > 1) ? fC[idx - 1] : gZl * f;
  T zp = (k < nk) ? fC[idx + 1] : gZr * f;
  lapC[idx] = (xm - 2.0 * f + xp) * invdx2 + (ym - 2.0 * f + yp) * invdy2 +
              (zm - 2.0 * f + zp) * invdz2;
}

void gpuLapC2CLocalKernel(cudaSolverType* lapC, const cudaSolverType* fC,
                          int nxc, int nyc, int nzc, cudaSolverType invdx2,
                          cudaSolverType invdy2, cudaSolverType invdz2,
                          const int ghostFactor[6], cudaStream_t stream) {
  int total = (nxc - 2) * (nyc - 2) * (nzc - 2);
  if (total <= 0)
    return;
  int blk = 256;
  k_lapC2CLocal<<<(total + blk - 1) / blk, blk, 0, stream>>>(
      lapC, fC, nxc, nyc, nzc, invdx2, invdy2, invdz2,
      (cudaSolverType)ghostFactor[0], (cudaSolverType)ghostFactor[1],
      (cudaSolverType)ghostFactor[2], (cudaSolverType)ghostFactor[3],
      (cudaSolverType)ghostFactor[4], (cudaSolverType)ghostFactor[5]);
  cudaErrChk(cudaGetLastError());
}

// =========================================================================
//  Block-Jacobi preconditioner kernel (not used: FGMRESBlockJacobi uses
//  gpuPrecomputeBlockJacobiInv + gpuApplyBlockJacobiInvKrylov)
//  Solves D_i * z_i = r_i at each interior node, where
//    D_i = (1 + δ²/2 * cΣ) I  +  (I + δ²/2 * C) * μ_i
//  μ_i is the full MUdot susceptibility tensor summed over species.
//  3×3 inversion via Cramer's rule.
// =========================================================================

template <typename T>
__global__ void
k_BlockJacobiPrecond(T* __restrict__ zX, T* __restrict__ zY, T* __restrict__ zZ,
                     const T* __restrict__ rX, const T* __restrict__ rY,
                     const T* __restrict__ rZ, const T* __restrict__ Bxn,
                     const T* __restrict__ Byn, const T* __restrict__ Bzn,
                     const T* __restrict__ Bx_ext, const T* __restrict__ By_ext,
                     const T* __restrict__ Bz_ext,
                     const T* __restrict__ rhons, // flat [ns][nxn][nyn][nzn]
                     const T* __restrict__ d_qom, // [ns]
                     int ns, T dt, T c_val, T delt, T FourPI,
                     T diagScalar, // 1 + δ²/2 * cΣ
                     T wx,         // 1 + δ²/2 / hx²
                     T wy,         // 1 + δ²/2 / hy²
                     T wz,         // 1 + δ²/2 / hz²
                     int nxn, int nyn, int nzn) {
  int i = blockIdx.z * blockDim.z + threadIdx.z + 1;
  int j = blockIdx.y * blockDim.y + threadIdx.y + 1;
  int k = blockIdx.x * blockDim.x + threadIdx.x + 1;
  if (i > nxn - 2 || j > nyn - 2 || k > nzn - 2)
    return;

  int idx = stencilIdx(i, j, k, nyn, nzn);
  size_t nodeSlice = (size_t)nxn * nyn * nzn;

  // ---- Build μ tensor at this node (sum over species) ----
  T mu00 = 0.0, mu01 = 0.0, mu02 = 0.0;
  T mu10 = 0.0, mu11 = 0.0, mu12 = 0.0;
  T mu20 = 0.0, mu21 = 0.0, mu22 = 0.0;

  T bx = Bxn[idx] + Bx_ext[idx];
  T by = Byn[idx] + By_ext[idx];
  T bz = Bzn[idx] + Bz_ext[idx];

  for (int is = 0; is < ns; is++) {
    T beta = 0.5 * d_qom[is] * dt / c_val;
    T omcx = beta * bx;
    T omcy = beta * by;
    T omcz = beta * bz;
    T omc2 = omcx * omcx + omcy * omcy + omcz * omcz;
    T prefactor = FourPI / 2.0 * delt * dt / c_val * d_qom[is];
    T denom = prefactor * rhons[is * nodeSlice + idx] / (1.0 + omc2);

    // μ_s = denom * (I + ω×  + ωωᵀ)
    // Row 0: E_x + (E_y*ωz - E_z*ωy) + (E·ω)*ωx
    //   → coeffs: E_x*(1 + ωx²), E_y*(ωz + ωx*ωy), E_z*(-ωy + ωx*ωz)
    mu00 += denom * (1.0 + omcx * omcx);
    mu01 += denom * (omcz + omcx * omcy);
    mu02 += denom * (-omcy + omcx * omcz);
    // Row 1: E_y + (E_z*ωx - E_x*ωz) + (E·ω)*ωy
    //   → coeffs: E_x*(-ωz + ωy*ωx), E_y*(1 + ωy²), E_z*(ωx + ωy*ωz)
    mu10 += denom * (-omcz + omcy * omcx);
    mu11 += denom * (1.0 + omcy * omcy);
    mu12 += denom * (omcx + omcy * omcz);
    // Row 2: E_z + (E_x*ωy - E_y*ωx) + (E·ω)*ωz
    //   → coeffs: E_x*(ωy + ωz*ωx), E_y*(-ωx + ωz*ωy), E_z*(1 + ωz²)
    mu20 += denom * (omcy + omcz * omcx);
    mu21 += denom * (-omcx + omcz * omcy);
    mu22 += denom * (1.0 + omcz * omcz);
  }

  // ---- Build D_i = diagScalar*I + diag(wx,wy,wz) * μ ----
  // D[p][s] = diagScalar * δ_{ps} + w_p * μ_{ps}
  T D00 = diagScalar + wx * mu00;
  T D01 = wx * mu01;
  T D02 = wx * mu02;
  T D10 = wy * mu10;
  T D11 = diagScalar + wy * mu11;
  T D12 = wy * mu12;
  T D20 = wz * mu20;
  T D21 = wz * mu21;
  T D22 = diagScalar + wz * mu22;

  // ---- Solve D * z = r via Cramer's rule ----
  T rx = rX[idx], ry = rY[idx], rz = rZ[idx];

  // det(D)
  T det = D00 * (D11 * D22 - D12 * D21) - D01 * (D10 * D22 - D12 * D20) +
          D02 * (D10 * D21 - D11 * D20);

  T invDet = 1.0 / det;

  // Adjugate (cofactor transpose) applied to r
  zX[idx] =
      invDet * ((D11 * D22 - D12 * D21) * rx + (D02 * D21 - D01 * D22) * ry +
                (D01 * D12 - D02 * D11) * rz);

  zY[idx] =
      invDet * ((D12 * D20 - D10 * D22) * rx + (D00 * D22 - D02 * D20) * ry +
                (D02 * D10 - D00 * D12) * rz);

  zZ[idx] =
      invDet * ((D10 * D21 - D11 * D20) * rx + (D01 * D20 - D00 * D21) * ry +
                (D00 * D11 - D01 * D10) * rz);
}

void gpuBlockJacobiPrecondKernel(
    cudaSolverType* zX, cudaSolverType* zY, cudaSolverType* zZ,
    const cudaSolverType* rX, const cudaSolverType* rY,
    const cudaSolverType* rZ, const cudaSolverType* Bxn,
    const cudaSolverType* Byn, const cudaSolverType* Bzn,
    const cudaSolverType* Bx_ext, const cudaSolverType* By_ext,
    const cudaSolverType* Bz_ext, const cudaSolverType* rhons,
    const cudaSolverType* d_qom, int ns, cudaSolverType dt,
    cudaSolverType c_val, cudaSolverType delt, cudaSolverType FourPI,
    cudaSolverType diagScalar, cudaSolverType wx, cudaSolverType wy,
    cudaSolverType wz, int nxn, int nyn, int nzn, cudaStream_t stream) {
  const PointLaunch L = pointLaunch(nxn - 2, nyn - 2, nzn - 2);
  k_BlockJacobiPrecond<<<L.grid, L.block, 0, stream>>>(
      zX, zY, zZ, rX, rY, rZ, Bxn, Byn, Bzn, Bx_ext, By_ext, Bz_ext, rhons,
      d_qom, ns, dt, c_val, delt, FourPI, diagScalar, wx, wy, wz, nxn, nyn,
      nzn);
  cudaErrChk(cudaGetLastError());
}

// =========================================================================
//  Precompute D^{-1} at each interior node.
//  Stores 9 entries per node in row-major order:
//    Dinv[(row*3 + col) * nodeSlice + nodeIdx]
// =========================================================================
template <typename T>
__global__ void k_PrecomputeBlockJacobiInv(
    T* __restrict__ Dinv, const T* __restrict__ Bxn, const T* __restrict__ Byn,
    const T* __restrict__ Bzn, const T* __restrict__ Bx_ext,
    const T* __restrict__ By_ext, const T* __restrict__ Bz_ext,
    const T* __restrict__ rhons, const T* __restrict__ d_qom, int ns, T dt,
    T c_val, T delt, T FourPI, T diagScalar, T wx, T wy, T wz, int nxn, int nyn,
    int nzn) {
  int i = blockIdx.z * blockDim.z + threadIdx.z + 1;
  int j = blockIdx.y * blockDim.y + threadIdx.y + 1;
  int k = blockIdx.x * blockDim.x + threadIdx.x + 1;
  if (i > nxn - 2 || j > nyn - 2 || k > nzn - 2)
    return;

  int idx = stencilIdx(i, j, k, nyn, nzn);
  size_t nodeSlice = (size_t)nxn * nyn * nzn;

  T mu00 = 0.0, mu01 = 0.0, mu02 = 0.0;
  T mu10 = 0.0, mu11 = 0.0, mu12 = 0.0;
  T mu20 = 0.0, mu21 = 0.0, mu22 = 0.0;

  T bx = Bxn[idx] + Bx_ext[idx];
  T by = Byn[idx] + By_ext[idx];
  T bz = Bzn[idx] + Bz_ext[idx];

  for (int is = 0; is < ns; is++) {
    T beta = 0.5 * d_qom[is] * dt / c_val;
    T omcx = beta * bx, omcy = beta * by, omcz = beta * bz;
    T omc2 = omcx * omcx + omcy * omcy + omcz * omcz;
    T prefactor = FourPI / 2.0 * delt * dt / c_val * d_qom[is];
    T denom = prefactor * rhons[is * nodeSlice + idx] / (1.0 + omc2);

    mu00 += denom * (1.0 + omcx * omcx);
    mu01 += denom * (omcz + omcx * omcy);
    mu02 += denom * (-omcy + omcx * omcz);
    mu10 += denom * (-omcz + omcy * omcx);
    mu11 += denom * (1.0 + omcy * omcy);
    mu12 += denom * (omcx + omcy * omcz);
    mu20 += denom * (omcy + omcz * omcx);
    mu21 += denom * (-omcx + omcz * omcy);
    mu22 += denom * (1.0 + omcz * omcz);
  }

  T D00 = diagScalar + wx * mu00, D01 = wx * mu01, D02 = wx * mu02;
  T D10 = wy * mu10, D11 = diagScalar + wy * mu11, D12 = wy * mu12;
  T D20 = wz * mu20, D21 = wz * mu21, D22 = diagScalar + wz * mu22;

  T det = D00 * (D11 * D22 - D12 * D21) - D01 * (D10 * D22 - D12 * D20) +
          D02 * (D10 * D21 - D11 * D20);
  T invDet = 1.0 / det;

  Dinv[0 * nodeSlice + idx] = invDet * (D11 * D22 - D12 * D21); // (0,0)
  Dinv[1 * nodeSlice + idx] = invDet * (D02 * D21 - D01 * D22); // (0,1)
  Dinv[2 * nodeSlice + idx] = invDet * (D01 * D12 - D02 * D11); // (0,2)
  Dinv[3 * nodeSlice + idx] = invDet * (D12 * D20 - D10 * D22); // (1,0)
  Dinv[4 * nodeSlice + idx] = invDet * (D00 * D22 - D02 * D20); // (1,1)
  Dinv[5 * nodeSlice + idx] = invDet * (D02 * D10 - D00 * D12); // (1,2)
  Dinv[6 * nodeSlice + idx] = invDet * (D10 * D21 - D11 * D20); // (2,0)
  Dinv[7 * nodeSlice + idx] = invDet * (D01 * D20 - D00 * D21); // (2,1)
  Dinv[8 * nodeSlice + idx] = invDet * (D00 * D11 - D01 * D10); // (2,2)
}

void gpuPrecomputeBlockJacobiInv(
    cudaSolverType* Dinv, const cudaSolverType* Bxn, const cudaSolverType* Byn,
    const cudaSolverType* Bzn, const cudaSolverType* Bx_ext,
    const cudaSolverType* By_ext, const cudaSolverType* Bz_ext,
    const cudaSolverType* rhons, const cudaSolverType* d_qom, int ns,
    cudaSolverType dt, cudaSolverType c_val, cudaSolverType delt,
    cudaSolverType FourPI, cudaSolverType diagScalar, cudaSolverType wx,
    cudaSolverType wy, cudaSolverType wz, int nxn, int nyn, int nzn,
    cudaStream_t stream) {
  const PointLaunch L = pointLaunch(nxn - 2, nyn - 2, nzn - 2);
  k_PrecomputeBlockJacobiInv<<<L.grid, L.block, 0, stream>>>(
      Dinv, Bxn, Byn, Bzn, Bx_ext, By_ext, Bz_ext, rhons, d_qom, ns, dt, c_val,
      delt, FourPI, diagScalar, wx, wy, wz, nxn, nyn, nzn);
  cudaErrChk(cudaGetLastError());
}

// =========================================================================
//  Fast D^{-1} application operating directly on Krylov vectors.
//  No pack/unpack needed — reads and writes Krylov layout directly.
//  Krylov layout: INTERLEAVED [Ex0,Ey0,Ez0, Ex1,Ey1,Ez1, ...]
//  tid = (i-1)*ny2*nz2 + (j-1)*nz2 + (k-1), sol_idx = tid*3
// =========================================================================
template <typename T>
__global__ void k_ApplyBlockJacobiInvKrylov(T* __restrict__ zKrylov,
                                            const T* __restrict__ rKrylov,
                                            const T* __restrict__ Dinv, int nxn,
                                            int nyn, int nzn) {
  int i = blockIdx.z * blockDim.z + threadIdx.z + 1;
  int j = blockIdx.y * blockDim.y + threadIdx.y + 1;
  int k = blockIdx.x * blockDim.x + threadIdx.x + 1;
  if (i > nxn - 2 || j > nyn - 2 || k > nzn - 2)
    return;

  int nodeIdx = stencilIdx(i, j, k, nyn, nzn);
  size_t nodeSlice = (size_t)nxn * nyn * nzn;

  // Interior linearisation (matches gpuSolver2Phys3/gpuPhys2Solver3)
  int nz2 = nzn - 2, ny2 = nyn - 2;
  int tid = ((i - 1) * ny2 + (j - 1)) * nz2 + (k - 1);
  int sol_idx = tid * 3;

  // Read r from interleaved Krylov vector
  T rx = rKrylov[sol_idx];
  T ry = rKrylov[sol_idx + 1];
  T rz = rKrylov[sol_idx + 2];

  // Read D^{-1} entries
  T A00 = Dinv[0 * nodeSlice + nodeIdx];
  T A01 = Dinv[1 * nodeSlice + nodeIdx];
  T A02 = Dinv[2 * nodeSlice + nodeIdx];
  T A10 = Dinv[3 * nodeSlice + nodeIdx];
  T A11 = Dinv[4 * nodeSlice + nodeIdx];
  T A12 = Dinv[5 * nodeSlice + nodeIdx];
  T A20 = Dinv[6 * nodeSlice + nodeIdx];
  T A21 = Dinv[7 * nodeSlice + nodeIdx];
  T A22 = Dinv[8 * nodeSlice + nodeIdx];

  // z = D^{-1} r
  zKrylov[sol_idx] = A00 * rx + A01 * ry + A02 * rz;
  zKrylov[sol_idx + 1] = A10 * rx + A11 * ry + A12 * rz;
  zKrylov[sol_idx + 2] = A20 * rx + A21 * ry + A22 * rz;
}

void gpuApplyBlockJacobiInvKrylov(cudaSolverType* zKrylov,
                                  const cudaSolverType* rKrylov,
                                  const cudaSolverType* Dinv, int nxn, int nyn,
                                  int nzn, cudaStream_t stream) {
  const PointLaunch L = pointLaunch(nxn - 2, nyn - 2, nzn - 2);
  k_ApplyBlockJacobiInvKrylov<<<L.grid, L.block, 0, stream>>>(
      zKrylov, rKrylov, Dinv, nxn, nyn, nzn);
  cudaErrChk(cudaGetLastError());
}
