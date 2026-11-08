/**
 * @file GPUPhysicsKernels.cu
 * @brief CUDA kernel implementations for iPIC3D physics operations:
 *        MUdot, PIdot, perfect-conductor rows, open-boundary, outflow,
 *        case-B-fix, planet-charge, Laplacian and block-Jacobi kernels.
 *
 * The MUdot, PIdot, block-Jacobi and Poisson-image Laplacian kernels use the
 * launch shape of GPUPointLaunch.cuh; the others use flat 1D or 2D face
 * launches.
 */

#include "GPUPhysicsKernels.cuh"
#include "GPUPointLaunch.cuh"

#define IDX3(i, j, k, ny, nz) ((i) * (ny) * (nz) + (j) * (nz) + (k))

// =========================================================================
//  MUdot kernel
// =========================================================================

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

  int idx = IDX3(i, j, k, nyn, nzn);

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

  const int idx = IDX3(i, j, k, nyn, nzn);
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

  int idx = IDX3(i, j, k, nyn, nzn);

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
//  Perfect conductor boundary condition kernels
//
//  On a conductor face the tangential E rows are E_t = ebc_t (the RHS) and
//  the normal row is Ampere's law without curl B, chi_n . E = E^n_n -
//  4 pi theta dt Jhat_n, with chi the implicit susceptibility at the node.
//  The image (A x) carries only the part linear in x, the RHS the rest, so
//  the operator the Krylov solvers see is linear:
//    image_n  = x_n + (chi_na x_a + chi_nb x_b) / chi_nn
//    source_n = (E^n_n - 4 pi theta dt Jhat_n) / chi_nn
// =========================================================================

/** Index of point (a, b) of the face plane `face` normal to DIR: compile-time
 *  layout, no branch at run time.  Shared by the conductor and open-face
 *  kernels. */
template <int DIR>
__device__ __forceinline__ int faceIdx3(int face, int a, int b, int ny,
                                        int nz) {
  if constexpr (DIR == 0)
    return IDX3(face, a, b, ny, nz);
  else if constexpr (DIR == 1)
    return IDX3(a, face, b, ny, nz);
  else
    return IDX3(a, b, face, ny, nz);
}

/** The point (a, b) of this thread on the interior [1..n-2]^2 of a face plane
 *  normal to DIR (1D launch, facePoints threads), b fastest: k on x and y
 *  faces, so a warp reads and writes along the contiguous axis; false past
 *  the plane. */
template <int DIR>
__device__ __forceinline__ bool facePoint(int& a, int& b, int nx, int ny,
                                          int nz) {
  const int d1 = (DIR == 0 ? ny : nx) - 2, d2 = (DIR == 2 ? ny : nz) - 2;
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= d1 * d2)
    return false;
  a = tid / d2 + 1;
  b = tid % d2 + 1;
  return true;
}

/** Packed interior index of point (a, b) of the face plane `face` normal to
 *  DIR in the Krylov layout of gpuPhys2Solver3: node (i, j, k) is entry
 *  ((i-1)(ny-2) + j-1)(nz-2) + k-1, with 3 components per entry. */
template <int DIR>
__device__ __forceinline__ int faceKrylovNode(int face, int a, int b, int ny,
                                              int nz) {
  const int i = DIR == 0 ? face : a;
  const int j = DIR == 0 ? a : (DIR == 1 ? face : b);
  const int k = DIR == 2 ? face : b;
  return ((i - 1) * (ny - 2) + (j - 1)) * (nz - 2) + (k - 1);
}

/** Row DIR of the implicit susceptibility at node idx, as (chi_nn, chi_na,
 *  chi_nb) with a = DIR+1, b = DIR+2 (mod 3):
 *  chi_nj = delta_nj + sum_s d_s (delta_nj + w_n w_j + eps_njk w_k), the
 *  row n of the operator MUdot applies in the interior,
 *  w = (q/m)_s dt B / 2c, d_s = 2 pi delt dt (q/m)_s rho_s / (c (1 + w^2)). */
template <typename T, int DIR>
__device__ __forceinline__ void
pecSusceptibilityRow(T& chiN, T& chiA, T& chiB, int idx,
                     const T* __restrict__ Bxn, const T* __restrict__ Byn,
                     const T* __restrict__ Bzn, const T* __restrict__ Bx_ext,
                     const T* __restrict__ By_ext, const T* __restrict__ Bz_ext,
                     const T* __restrict__ rhons, const T* __restrict__ qom_d,
                     int ns, int speciesStride, T dt, T c, T FourPI, T delt) {
  const T bx = Bxn[idx] + Bx_ext[idx], by = Byn[idx] + By_ext[idx],
          bz = Bzn[idx] + Bz_ext[idx];
  // field components in the (n, a, b) frame of the face, at compile time
  const T bn = DIR == 0 ? bx : (DIR == 1 ? by : bz);
  const T ba = DIR == 0 ? by : (DIR == 1 ? bz : bx);
  const T bb = DIR == 0 ? bz : (DIR == 1 ? bx : by);
  const T b2 = bx * bx + by * by + bz * bz;
  chiN = 1.0;
  chiA = 0.0;
  chiB = 0.0;
  for (int is = 0; is < ns; is++) {
    const T beta = 0.5 * qom_d[is] * dt / c;
    const T wn = beta * bn, wa = beta * ba, wb = beta * bb;
    const T d = FourPI / 2.0 * delt * dt / c * qom_d[is] *
                rhons[is * speciesStride + idx] / (1.0 + beta * beta * b2);
    chiN += (1.0 + wn * wn) * d;
    // the rows of MUdot / k_PIdotSpecies: (v + v x w + (v.w) w) d
    chiA += (wn * wa + wb) * d; // (v x w)_n from v_a: +w_b
    chiB += (wn * wb - wa) * d; // (v x w)_n from v_b: -w_a
  }
}

/** Conductor image rows on face node plane `face` normal to DIR (tangential
 *  interior), linear in the Krylov vector (see the banner above), written
 *  into the Krylov image. */
template <typename T, int DIR>
__global__ void k_perfectConductorImage(
    T* __restrict__ im, const T* __restrict__ vX, const T* __restrict__ vY,
    const T* __restrict__ vZ, const T* __restrict__ Bxn,
    const T* __restrict__ Byn, const T* __restrict__ Bzn,
    const T* __restrict__ Bx_ext, const T* __restrict__ By_ext,
    const T* __restrict__ Bz_ext, const T* __restrict__ rhons,
    const T* __restrict__ qom_d, int ns, T dt, T c, T FourPI, T delt, int face,
    int nxn, int nyn, int nzn) {
  constexpr int A = (DIR + 1) % 3, B = (DIR + 2) % 3;
  int a, b;
  if (!facePoint<DIR>(a, b, nxn, nyn, nzn))
    return;
  const int idx = faceIdx3<DIR>(face, a, b, nyn, nzn);
  T chiN, chiA, chiB;
  pecSusceptibilityRow<T, DIR>(chiN, chiA, chiB, idx, Bxn, Byn, Bzn, Bx_ext,
                               By_ext, Bz_ext, rhons, qom_d, ns,
                               nxn * nyn * nzn, dt, c, FourPI, delt);
  T* const row = im + 3 * faceKrylovNode<DIR>(face, a, b, nyn, nzn);
  const T v[3] = {vX[idx], vY[idx], vZ[idx]};
  row[DIR] = v[DIR] + (chiA * v[A] + chiB * v[B]) / chiN;
  row[A] = v[A];
  row[B] = v[B];
}

/** Conductor RHS rows on the same plane: normal (E^n_n - 4 pi theta dt
 *  Jhat_n) / chi_nn, tangential ebc. */
template <typename T, int DIR>
__global__ void k_perfectConductorSource(
    T* __restrict__ sX, T* __restrict__ sY, T* __restrict__ sZ,
    const T* __restrict__ En, const T* __restrict__ Jn,
    const T* __restrict__ Bxn, const T* __restrict__ Byn,
    const T* __restrict__ Bzn, const T* __restrict__ Bx_ext,
    const T* __restrict__ By_ext, const T* __restrict__ Bz_ext,
    const T* __restrict__ rhons, const T* __restrict__ qom_d, int ns, T dt, T c,
    T th, T FourPI, T delt, T ebc0, T ebc1, T ebc2, int face, int nxn, int nyn,
    int nzn) {
  constexpr int A = (DIR + 1) % 3, B = (DIR + 2) % 3;
  int a, b;
  if (!facePoint<DIR>(a, b, nxn, nyn, nzn))
    return;
  const int idx = faceIdx3<DIR>(face, a, b, nyn, nzn);
  T chiN, chiA, chiB;
  pecSusceptibilityRow<T, DIR>(chiN, chiA, chiB, idx, Bxn, Byn, Bzn, Bx_ext,
                               By_ext, Bz_ext, rhons, qom_d, ns,
                               nxn * nyn * nzn, dt, c, FourPI, delt);
  T* const src[3] = {sX, sY, sZ};
  const T ebc[3] = {ebc0, ebc1, ebc2};
  src[DIR][idx] = (En[idx] - Jn[idx] * dt * th * FourPI) / chiN;
  src[A][idx] = ebc[A];
  src[B][idx] = ebc[B];
}

// =========================================================================
//  Host wrappers
// =========================================================================

/** Points of the interior [1..n-2]^2 of a face plane normal to dir. */
static inline int facePoints(int dir, int nx, int ny, int nz) {
  return ((dir == 0 ? ny : nx) - 2) * ((dir == 2 ? ny : nz) - 2);
}
static constexpr int kFaceBlock = 256;

/** Launch `KERNEL<cudaSolverType, DIR>` for the runtime face direction dir:
 *  the only direction dispatch, on the host, once per launch. */
#define FACE_DIR_DISPATCH(dir, KERNEL, grid, block, stream, ...)               \
  switch (dir) {                                                               \
  case 0:                                                                      \
    KERNEL<cudaSolverType, 0><<<grid, block, 0, stream>>>(__VA_ARGS__);        \
    break;                                                                     \
  case 1:                                                                      \
    KERNEL<cudaSolverType, 1><<<grid, block, 0, stream>>>(__VA_ARGS__);        \
    break;                                                                     \
  default:                                                                     \
    KERNEL<cudaSolverType, 2><<<grid, block, 0, stream>>>(__VA_ARGS__);        \
    break;                                                                     \
  }

void gpuPerfectConductorImage(
    cudaSolverType* image, const cudaSolverType* vectX,
    const cudaSolverType* vectY, const cudaSolverType* vectZ,
    const cudaSolverType* Bxn, const cudaSolverType* Byn,
    const cudaSolverType* Bzn, const cudaSolverType* Bx_ext,
    const cudaSolverType* By_ext, const cudaSolverType* Bz_ext,
    const cudaSolverType* rhons, const cudaSolverType* d_qom, int ns,
    cudaSolverType dt, cudaSolverType c, cudaSolverType FourPI,
    cudaSolverType delt, int nxn, int nyn, int nzn, int dir, bool high,
    cudaStream_t stream) {
  const int n[3] = {nxn, nyn, nzn};
  const int face = high ? n[dir] - 2 : 1;
  const int total = facePoints(dir, nxn, nyn, nzn);
  if (total <= 0)
    return;
  const int grid = (total + kFaceBlock - 1) / kFaceBlock;
  FACE_DIR_DISPATCH(dir, k_perfectConductorImage, grid, kFaceBlock, stream,
                    image, vectX, vectY, vectZ, Bxn, Byn, Bzn, Bx_ext, By_ext,
                    Bz_ext, rhons, d_qom, ns, dt, c, FourPI, delt, face, nxn,
                    nyn, nzn)
  cudaErrChk(cudaGetLastError());
}

void gpuPerfectConductorSource(
    cudaSolverType* vectorX, cudaSolverType* vectorY, cudaSolverType* vectorZ,
    const cudaSolverType* En, const cudaSolverType* Jn,
    const cudaSolverType* Bxn, const cudaSolverType* Byn,
    const cudaSolverType* Bzn, const cudaSolverType* Bx_ext,
    const cudaSolverType* By_ext, const cudaSolverType* Bz_ext,
    const cudaSolverType* rhons, const cudaSolverType* d_qom, int ns,
    cudaSolverType dt, cudaSolverType c, cudaSolverType th,
    cudaSolverType FourPI, cudaSolverType delt, cudaSolverType ebc0,
    cudaSolverType ebc1, cudaSolverType ebc2, int nxn, int nyn, int nzn,
    int dir, bool high, cudaStream_t stream) {
  const int n[3] = {nxn, nyn, nzn};
  const int face = high ? n[dir] - 2 : 1;
  const int total = facePoints(dir, nxn, nyn, nzn);
  if (total <= 0)
    return;
  const int grid = (total + kFaceBlock - 1) / kFaceBlock;
  FACE_DIR_DISPATCH(dir, k_perfectConductorSource, grid, kFaceBlock, stream,
                    vectorX, vectorY, vectorZ, En, Jn, Bxn, Byn, Bzn, Bx_ext,
                    By_ext, Bz_ext, rhons, d_qom, ns, dt, c, th, FourPI, delt,
                    ebc0, ebc1, ebc2, face, nxn, nyn, nzn)
  cudaErrChk(cudaGetLastError());
}

#undef IDX3

// =========================================================================
//  Non-periodic particle faces: double the node moments on the face plane
// =========================================================================

// Kernel: double the values on the X face plane i = fixedI
template <typename T>
__global__ void k_doubleFaceX(T* __restrict__ arr, int fixedI, int nyn,
                              int nzn) {
  int j = blockIdx.x * blockDim.x + threadIdx.x + 1;
  int k = blockIdx.y * blockDim.y + threadIdx.y + 1;
  if (j >= nyn - 1 || k >= nzn - 1)
    return;
  arr[(fixedI * nyn + j) * nzn + k] *= 2.0;
}

// Kernel: double the values on the Y face plane j = fixedJ
template <typename T>
__global__ void k_doubleFaceY(T* __restrict__ arr, int fixedJ, int nxn, int nyn,
                              int nzn) {
  int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
  int k = blockIdx.y * blockDim.y + threadIdx.y + 1;
  if (i >= nxn - 1 || k >= nzn - 1)
    return;
  arr[(i * nyn + fixedJ) * nzn + k] *= 2.0;
}

// Kernel: double the values on the Z face plane k = fixedK
template <typename T>
__global__ void k_doubleFaceZ(T* __restrict__ arr, int fixedK, int nxn, int nyn,
                              int nzn) {
  int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
  int j = blockIdx.y * blockDim.y + threadIdx.y + 1;
  if (i >= nxn - 1 || j >= nyn - 1)
    return;
  arr[(i * nyn + j) * nzn + fixedK] *= 2.0;
}

// Batched versions: blockIdx.z selects the field from the pointer array
template <typename T>
__global__ void k_batchDoubleFaceX(T* const* __restrict__ ptrs, int fixedI,
                                   int nyn, int nzn) {
  int j = blockIdx.x * blockDim.x + threadIdx.x + 1;
  int k = blockIdx.y * blockDim.y + threadIdx.y + 1;
  if (j >= nyn - 1 || k >= nzn - 1)
    return;
  ptrs[blockIdx.z][(fixedI * nyn + j) * nzn + k] *= 2.0;
}

template <typename T>
__global__ void k_batchDoubleFaceY(T* const* __restrict__ ptrs, int fixedJ,
                                   int nxn, int nyn, int nzn) {
  int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
  int k = blockIdx.y * blockDim.y + threadIdx.y + 1;
  if (i >= nxn - 1 || k >= nzn - 1)
    return;
  ptrs[blockIdx.z][(i * nyn + fixedJ) * nzn + k] *= 2.0;
}

template <typename T>
__global__ void k_batchDoubleFaceZ(T* const* __restrict__ ptrs, int fixedK,
                                   int nxn, int nyn, int nzn) {
  int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
  int j = blockIdx.y * blockDim.y + threadIdx.y + 1;
  if (i >= nxn - 1 || j >= nyn - 1)
    return;
  ptrs[blockIdx.z][(i * nyn + j) * nzn + fixedK] *= 2.0;
}

void gpuAdjustNonPeriodicDensities(int nptrs, cudaSolverType* const* d_devPtrs,
                                   int nxn, int nyn, int nzn, bool xLeftNull,
                                   bool xRightNull, bool yLeftNull,
                                   bool yRightNull, bool zLeftNull,
                                   bool zRightNull, cudaStream_t stream) {
  static constexpr int BLK = 16;

  // X faces (fixedI = 1 or nxn-2)
  if (xLeftNull) {
    dim3 block(BLK, BLK);
    dim3 grid(((nyn - 2) + BLK - 1) / BLK, ((nzn - 2) + BLK - 1) / BLK, nptrs);
    k_batchDoubleFaceX<<<grid, block, 0, stream>>>(d_devPtrs, 1, nyn, nzn);
  }
  if (xRightNull) {
    dim3 block(BLK, BLK);
    dim3 grid(((nyn - 2) + BLK - 1) / BLK, ((nzn - 2) + BLK - 1) / BLK, nptrs);
    k_batchDoubleFaceX<<<grid, block, 0, stream>>>(d_devPtrs, nxn - 2, nyn,
                                                   nzn);
  }

  // Y faces (fixedJ = 1 or nyn-2)
  if (yLeftNull) {
    dim3 block(BLK, BLK);
    dim3 grid(((nxn - 2) + BLK - 1) / BLK, ((nzn - 2) + BLK - 1) / BLK, nptrs);
    k_batchDoubleFaceY<<<grid, block, 0, stream>>>(d_devPtrs, 1, nxn, nyn, nzn);
  }
  if (yRightNull) {
    dim3 block(BLK, BLK);
    dim3 grid(((nxn - 2) + BLK - 1) / BLK, ((nzn - 2) + BLK - 1) / BLK, nptrs);
    k_batchDoubleFaceY<<<grid, block, 0, stream>>>(d_devPtrs, nyn - 2, nxn, nyn,
                                                   nzn);
  }

  // Z faces (fixedK = 1 or nzn-2)
  if (zLeftNull) {
    dim3 block(BLK, BLK);
    dim3 grid(((nxn - 2) + BLK - 1) / BLK, ((nyn - 2) + BLK - 1) / BLK, nptrs);
    k_batchDoubleFaceZ<<<grid, block, 0, stream>>>(d_devPtrs, 1, nxn, nyn, nzn);
  }
  if (zRightNull) {
    dim3 block(BLK, BLK);
    dim3 grid(((nxn - 2) + BLK - 1) / BLK, ((nyn - 2) + BLK - 1) / BLK, nptrs);
    k_batchDoubleFaceZ<<<grid, block, 0, stream>>>(d_devPtrs, nzn - 2, nxn, nyn,
                                                   nzn);
  }

  cudaErrChk(cudaGetLastError());
}

// =========================================================================
//  Open Boundary kernels
// =========================================================================

/** Set 3 arrays to (cx, cy, cz) on a face plane interior [1..n-2]. */
template <typename T, int DIR>
__global__ void k_setFacePlane3(T* __restrict__ X, T* __restrict__ Y,
                                T* __restrict__ Z, T cx, T cy, T cz, int face,
                                int nx, int ny, int nz) {
  int a, b;
  if (!facePoint<DIR>(a, b, nx, ny, nz))
    return;
  const int idx = faceIdx3<DIR>(face, a, b, ny, nz);
  X[idx] = cx;
  Y[idx] = cy;
  Z[idx] = cz;
}

/** Identity rows: image = vect on a face plane interior [1..n-2], written
 *  into the Krylov image. */
template <typename T, int DIR>
__global__ void k_copyFacePlane3(T* __restrict__ im, const T* __restrict__ vX,
                                 const T* __restrict__ vY,
                                 const T* __restrict__ vZ, int face, int nx,
                                 int ny, int nz) {
  int a, b;
  if (!facePoint<DIR>(a, b, nx, ny, nz))
    return;
  const int idx = faceIdx3<DIR>(face, a, b, ny, nz);
  T* const row = im + 3 * faceKrylovNode<DIR>(face, a, b, ny, nz);
  row[0] = vX[idx];
  row[1] = vY[idx];
  row[2] = vZ[idx];
}

void gpuOpenBCSetFace3(cudaSolverType* X, cudaSolverType* Y, cudaSolverType* Z,
                       cudaSolverType cx, cudaSolverType cy, cudaSolverType cz,
                       int dir, int faceIdx, int nx, int ny, int nz,
                       cudaStream_t stream) {
  const int total = facePoints(dir, nx, ny, nz);
  if (total <= 0)
    return;
  const int grid = (total + kFaceBlock - 1) / kFaceBlock;
  FACE_DIR_DISPATCH(dir, k_setFacePlane3, grid, kFaceBlock, stream, X, Y, Z, cx,
                    cy, cz, faceIdx, nx, ny, nz)
  cudaErrChk(cudaGetLastError());
}

void gpuOpenBCCopyFace3(cudaSolverType* image, const cudaSolverType* vX,
                        const cudaSolverType* vY, const cudaSolverType* vZ,
                        int dir, int faceIdx, int nx, int ny, int nz,
                        cudaStream_t stream) {
  const int total = facePoints(dir, nx, ny, nz);
  if (total <= 0)
    return;
  const int grid = (total + kFaceBlock - 1) / kFaceBlock;
  FACE_DIR_DISPATCH(dir, k_copyFacePlane3, grid, kFaceBlock, stream, image, vX,
                    vY, vZ, faceIdx, nx, ny, nz)
  cudaErrChk(cudaGetLastError());
}

#undef FACE_DIR_DISPATCH

/**
 * SAL blend: v[idx] = v[idx]*sal + target*(1-sal) on boundary layers.
 * dir: 0=X, 1=Y, 2=Z.
 * ascending: true if sal increases from layerStart to layerEnd.
 * skipComp: component (0=X, 1=Y, 2=Z) left unchanged, or -1 for none.
 */
template <typename T>
__global__ void k_salBlend3(T* X, T* Y, T* Z, T tgtX, T tgtY, T tgtZ, int dir,
                            int layerStart, int layerEnd, T invNLayers,
                            int ascending, int skipComp, int nx, int ny,
                            int nz) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int nLayers = layerEnd - layerStart + 1;
  int d1, d2;
  if (dir == 0) {
    d1 = ny;
    d2 = nz;
  } else if (dir == 1) {
    d1 = nx;
    d2 = nz;
  } else {
    d1 = nx;
    d2 = ny;
  }
  int total = nLayers * d1 * d2;
  if (tid >= total)
    return;
  int layerOff = tid / (d1 * d2);
  int rem = tid % (d1 * d2);
  int a = rem / d2;
  int b = rem % d2;
  int layer = layerStart + layerOff;
  T sal = ascending ? (T)(layer - layerStart) * invNLayers
                    : (T)(layerEnd - layer) * invNLayers;
  int i, j, k;
  if (dir == 0) {
    i = layer;
    j = a;
    k = b;
  } else if (dir == 1) {
    i = a;
    j = layer;
    k = b;
  } else {
    i = a;
    j = b;
    k = layer;
  }
  int idx = (i * ny + j) * nz + k;
  if (skipComp != 0)
    X[idx] = X[idx] * sal + tgtX * (1.0 - sal);
  if (skipComp != 1)
    Y[idx] = Y[idx] * sal + tgtY * (1.0 - sal);
  if (skipComp != 2)
    Z[idx] = Z[idx] * sal + tgtZ * (1.0 - sal);
}

void gpuSALBlendLayers3(cudaSolverType* X, cudaSolverType* Y, cudaSolverType* Z,
                        cudaSolverType tgtX, cudaSolverType tgtY,
                        cudaSolverType tgtZ, int dir, int layerStart,
                        int layerEnd, cudaSolverType invNLayers, int ascending,
                        int nx, int ny, int nz, cudaStream_t stream,
                        int skipComp) {
  int nLayers = layerEnd - layerStart + 1;
  int d1, d2;
  if (dir == 0) {
    d1 = ny;
    d2 = nz;
  } else if (dir == 1) {
    d1 = nx;
    d2 = nz;
  } else {
    d1 = nx;
    d2 = ny;
  }
  int total = nLayers * d1 * d2;
  if (total <= 0)
    return;
  int blk = 256;
  k_salBlend3<<<(total + blk - 1) / blk, blk, 0, stream>>>(
      X, Y, Z, tgtX, tgtY, tgtZ, dir, layerStart, layerEnd, invNLayers,
      ascending, skipComp, nx, ny, nz);
  cudaErrChk(cudaGetLastError());
}

/** Set 3 arrays to constant values on boundary layers. */
template <typename T>
__global__ void k_setConstLayers3(T* X, T* Y, T* Z, T cx, T cy, T cz, int dir,
                                  int layerStart, int layerEnd, int nx, int ny,
                                  int nz) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int nLayers = layerEnd - layerStart + 1;
  int d1, d2;
  if (dir == 0) {
    d1 = ny;
    d2 = nz;
  } else if (dir == 1) {
    d1 = nx;
    d2 = nz;
  } else {
    d1 = nx;
    d2 = ny;
  }
  int total = nLayers * d1 * d2;
  if (tid >= total)
    return;
  int layerOff = tid / (d1 * d2);
  int rem = tid % (d1 * d2);
  int a = rem / d2;
  int b = rem % d2;
  int layer = layerStart + layerOff;
  int i, j, k;
  if (dir == 0) {
    i = layer;
    j = a;
    k = b;
  } else if (dir == 1) {
    i = a;
    j = layer;
    k = b;
  } else {
    i = a;
    j = b;
    k = layer;
  }
  int idx = (i * ny + j) * nz + k;
  X[idx] = cx;
  Y[idx] = cy;
  Z[idx] = cz;
}

void gpuSetConstLayers3(cudaSolverType* X, cudaSolverType* Y, cudaSolverType* Z,
                        cudaSolverType cx, cudaSolverType cy, cudaSolverType cz,
                        int dir, int layerStart, int layerEnd, int nx, int ny,
                        int nz, cudaStream_t stream) {
  int nLayers = layerEnd - layerStart + 1;
  int d1, d2;
  if (dir == 0) {
    d1 = ny;
    d2 = nz;
  } else if (dir == 1) {
    d1 = nx;
    d2 = nz;
  } else {
    d1 = nx;
    d2 = ny;
  }
  int total = nLayers * d1 * d2;
  if (total <= 0)
    return;
  int blk = 256;
  k_setConstLayers3<<<(total + blk - 1) / blk, blk, 0, stream>>>(
      X, Y, Z, cx, cy, cz, dir, layerStart, layerEnd, nx, ny, nz);
  cudaErrChk(cudaGetLastError());
}

/** Copy from a constant reference plane to boundary layers. */
template <typename T>
__global__ void k_extrapolateLayers3(T* X, T* Y, T* Z, int dir, int layerStart,
                                     int layerEnd, int refLayer, int nx, int ny,
                                     int nz) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int nLayers = layerEnd - layerStart + 1;
  int d1, d2;
  if (dir == 0) {
    d1 = ny;
    d2 = nz;
  } else if (dir == 1) {
    d1 = nx;
    d2 = nz;
  } else {
    d1 = nx;
    d2 = ny;
  }
  int total = nLayers * d1 * d2;
  if (tid >= total)
    return;
  int layerOff = tid / (d1 * d2);
  int rem = tid % (d1 * d2);
  int a = rem / d2;
  int b = rem % d2;
  int layer = layerStart + layerOff;
  int i, j, k, ri, rj, rk;
  if (dir == 0) {
    i = layer;
    j = a;
    k = b;
    ri = refLayer;
    rj = a;
    rk = b;
  } else if (dir == 1) {
    i = a;
    j = layer;
    k = b;
    ri = a;
    rj = refLayer;
    rk = b;
  } else {
    i = a;
    j = b;
    k = layer;
    ri = a;
    rj = b;
    rk = refLayer;
  }
  int idx = (i * ny + j) * nz + k;
  int ridx = (ri * ny + rj) * nz + rk;
  X[idx] = X[ridx];
  Y[idx] = Y[ridx];
  Z[idx] = Z[ridx];
}

void gpuExtrapolateLayers3(cudaSolverType* X, cudaSolverType* Y,
                           cudaSolverType* Z, int dir, int layerStart,
                           int layerEnd, int refLayer, int nx, int ny, int nz,
                           cudaStream_t stream) {
  int nLayers = layerEnd - layerStart + 1;
  int d1, d2;
  if (dir == 0) {
    d1 = ny;
    d2 = nz;
  } else if (dir == 1) {
    d1 = nx;
    d2 = nz;
  } else {
    d1 = nx;
    d2 = ny;
  }
  int total = nLayers * d1 * d2;
  if (total <= 0)
    return;
  int blk = 256;
  k_extrapolateLayers3<<<(total + blk - 1) / blk, blk, 0, stream>>>(
      X, Y, Z, dir, layerStart, layerEnd, refLayer, nx, ny, nz);
  cudaErrChk(cudaGetLastError());
}

/** One thread per (j,k) line: rescale the reference-plane species densities
 *  to a neutral pair and write them on every layer. A line where one sign
 *  is missing is emptied, which is the only neutral choice left. */
template <typename T>
__global__ void k_neutralizeLayersX(T* rhons, const T* qom, int ns,
                                    int layerStart, int layerEnd, int refLayer,
                                    int nx, int ny, int nz) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= ny * nz)
    return;
  int j = tid / nz;
  int k = tid % nz;
  size_t nodeSize = (size_t)nx * ny * nz;
  size_t ref = ((size_t)refLayer * ny + j) * nz + k;
  T pos = 0, neg = 0;
  for (int is = 0; is < ns; is++) {
    T r = rhons[is * nodeSize + ref];
    if (qom[is] > 0)
      pos += r;
    else
      neg -= r;
  }
  T target = (T)0.5 * (pos + neg);
  bool both = pos > 0 && neg > 0;
  T scalePos = both ? target / pos : (T)0;
  T scaleNeg = both ? target / neg : (T)0;
  for (int is = 0; is < ns; is++) {
    T r = rhons[is * nodeSize + ref] * (qom[is] > 0 ? scalePos : scaleNeg);
    for (int i = layerStart; i <= layerEnd; i++)
      rhons[is * nodeSize + ((size_t)i * ny + j) * nz + k] = r;
  }
}

void gpuNeutralizeLayersX(cudaSolverType* rhons, const cudaSolverType* qom,
                          int ns, int layerStart, int layerEnd, int refLayer,
                          int nx, int ny, int nz, cudaStream_t stream) {
  int total = ny * nz;
  if (total <= 0 || layerEnd < layerStart)
    return;
  int blk = 256;
  k_neutralizeLayersX<<<(total + blk - 1) / blk, blk, 0, stream>>>(
      rhons, qom, ns, layerStart, layerEnd, refLayer, nx, ny, nz);
  cudaErrChk(cudaGetLastError());
}

// =========================================================================
//  Ghost rows of the X-right ghost cell plane (outflow face)
// =========================================================================

__device__ __forceinline__ size_t ghostIdx(int i, int j, int k, int ny,
                                           int nz) {
  return ((size_t)i * ny + j) * nz + k;
}

/** Source row of a ghost row under rule r: partner (periodic) or adjacent. */
__device__ __forceinline__ int ghostRowSource(int r, int ghost, int n) {
  if (r == GHOST_ROW_PERIODIC)
    return ghost == 0 ? n - 2 : 1;
  return ghost == 0 ? 1 : n - 2;
}

/** axis 1: rows j = 0, nyc-1 over k in [1, nzc-2]; axis 2: rows k = 0, nzc-1
 *  over all j. blockIdx.y selects the array. */
__global__ void k_ghostRowsXright(CellArrayBatch batch, int nxc, int nyc,
                                  int nzc, int axis, int ruleLo, int ruleHi) {
  cudaSolverType* a = batch.p[blockIdx.y];
  const int t = blockIdx.x * blockDim.x + threadIdx.x;
  const int i = nxc - 1;
  if (axis == 1) {
    const int k = t + 1;
    if (k > nzc - 2)
      return;
    if (ruleLo != GHOST_ROW_KEEP)
      a[ghostIdx(i, 0, k, nyc, nzc)] =
          a[ghostIdx(i, ghostRowSource(ruleLo, 0, nyc), k, nyc, nzc)];
    if (ruleHi != GHOST_ROW_KEEP)
      a[ghostIdx(i, nyc - 1, k, nyc, nzc)] =
          a[ghostIdx(i, ghostRowSource(ruleHi, nyc - 1, nyc), k, nyc, nzc)];
  } else {
    const int j = t;
    if (j > nyc - 1)
      return;
    if (ruleLo != GHOST_ROW_KEEP)
      a[ghostIdx(i, j, 0, nyc, nzc)] =
          a[ghostIdx(i, j, ghostRowSource(ruleLo, 0, nzc), nyc, nzc)];
    if (ruleHi != GHOST_ROW_KEEP)
      a[ghostIdx(i, j, nzc - 1, nyc, nzc)] =
          a[ghostIdx(i, j, ghostRowSource(ruleHi, nzc - 1, nzc), nyc, nzc)];
  }
}

void gpuGhostRowsXright(const CellArrayBatch& batch, int nxc, int nyc, int nzc,
                        const int rule[4], cudaStream_t stream) {
  if (batch.n <= 0)
    return;
  const int blk = 128;
  if (rule[0] != GHOST_ROW_KEEP || rule[1] != GHOST_ROW_KEEP) {
    dim3 grid((nzc - 2 + blk - 1) / blk, batch.n);
    k_ghostRowsXright<<<grid, blk, 0, stream>>>(batch, nxc, nyc, nzc, 1,
                                                rule[0], rule[1]);
    cudaErrChk(cudaGetLastError());
  }
  if (rule[2] != GHOST_ROW_KEEP || rule[3] != GHOST_ROW_KEEP) {
    dim3 grid((nyc + blk - 1) / blk, batch.n);
    k_ghostRowsXright<<<grid, blk, 0, stream>>>(batch, nxc, nyc, nzc, 2,
                                                rule[2], rule[3]);
    cudaErrChk(cudaGetLastError());
  }
}

// =========================================================================
//  Fix B kernels for GEM / ForceFree
// =========================================================================

/** Fix center-based B for GEM on 3 Y-boundary layers.
 *  side: 0=left (j=0,1,2), 1=right (j=nyc-1, nyc-2, nyc-3). */
template <typename T>
__global__ void k_fixBcGEM(T* Bxc, T* Byc, T* Bzc, T B0x, T B0y, T B0z,
                           T yStart, T dy, T LyH, T delta, int side, int nxc,
                           int nyc, int nzc) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= nxc * nzc)
    return;
  int i = tid / nzc;
  int k = tid % nzc;
  int j0 = (side == 0) ? 0 : nyc - 1;
  int j1 = (side == 0) ? 1 : nyc - 2;
  int j2 = (side == 0) ? 2 : nyc - 3;
  T yc0 = yStart + ((T)j0 - 0.5) * dy;
  T bx_val = B0x * tanh((yc0 - LyH) / delta);
  int idx0 = (i * nyc + j0) * nzc + k;
  int idx1 = (i * nyc + j1) * nzc + k;
  int idx2 = (i * nyc + j2) * nzc + k;
  Bxc[idx0] = bx_val;
  Bxc[idx1] = bx_val;
  Bxc[idx2] = bx_val;
  Byc[idx0] = B0y;
  Bzc[idx0] = B0z;
  Bzc[idx1] = B0z;
  Bzc[idx2] = B0z;
}

void gpuFixBcGEMKernel(cudaSolverType* Bxc, cudaSolverType* Byc,
                       cudaSolverType* Bzc, cudaSolverType B0x,
                       cudaSolverType B0y, cudaSolverType B0z,
                       cudaSolverType yStart, cudaSolverType dy,
                       cudaSolverType LyH, cudaSolverType delta, int side,
                       int nxc, int nyc, int nzc, cudaStream_t stream) {
  int total = nxc * nzc;
  if (total <= 0)
    return;
  int blk = 256;
  k_fixBcGEM<<<(total + blk - 1) / blk, blk, 0, stream>>>(
      Bxc, Byc, Bzc, B0x, B0y, B0z, yStart, dy, LyH, delta, side, nxc, nyc,
      nzc);
  cudaErrChk(cudaGetLastError());
}

/** Fix node-based B for GEM on 3 Y-boundary layers.
 *  Uses the CENTER Y coordinate of the ghost cell for the tanh profile, as
 *  k_fixBcGEM does.
 *  side: 0=left (j=0,1,2), 1=right (j=nyn-1, nyn-2, nyn-3). */
template <typename T>
__global__ void k_fixBnGEM(T* Bxn, T* Byn, T* Bzn, T B0x, T B0y, T B0z,
                           T yStart, T dy, T LyH, T delta, int side, int nxn,
                           int nyn, int nzn, int nyc) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= nxn * nzn)
    return;
  int i = tid / nzn;
  int k = tid % nzn;
  // Center Y index for tanh profile: 0 for left, nyc-1 for right
  int jc = (side == 0) ? 0 : nyc - 1;
  T yc = yStart + ((T)jc - 0.5) * dy;
  T bx_val = B0x * tanh((yc - LyH) / delta);
  int j0 = (side == 0) ? 0 : nyn - 1;
  int j1 = (side == 0) ? 1 : nyn - 2;
  int j2 = (side == 0) ? 2 : nyn - 3;
  int idx0 = (i * nyn + j0) * nzn + k;
  int idx1 = (i * nyn + j1) * nzn + k;
  int idx2 = (i * nyn + j2) * nzn + k;
  Bxn[idx0] = bx_val;
  Bxn[idx1] = bx_val;
  Bxn[idx2] = bx_val;
  Byn[idx0] = B0y;
  Bzn[idx0] = B0z;
  Bzn[idx1] = B0z;
  Bzn[idx2] = B0z;
}

void gpuFixBnGEMKernel(cudaSolverType* Bxn, cudaSolverType* Byn,
                       cudaSolverType* Bzn, cudaSolverType B0x,
                       cudaSolverType B0y, cudaSolverType B0z,
                       cudaSolverType yStart, cudaSolverType dy,
                       cudaSolverType LyH, cudaSolverType delta, int side,
                       int nxn, int nyn, int nzn, int nyc,
                       cudaStream_t stream) {
  int total = nxn * nzn;
  if (total <= 0)
    return;
  int blk = 256;
  k_fixBnGEM<<<(total + blk - 1) / blk, blk, 0, stream>>>(
      Bxn, Byn, Bzn, B0x, B0y, B0z, yStart, dy, LyH, delta, side, nxn, nyn, nzn,
      nyc);
  cudaErrChk(cudaGetLastError());
}

/** Fix center-based B for ForceFree on a Y boundary: Bz = B0z/cosh on the
 *  three cell planes next to the wall; Bx = B0x tanh and By = B0y on the
 *  ghost plane. */
template <typename T>
__global__ void k_fixBforcefree(T* Bxc, T* Byc, T* Bzc, T B0x, T B0y, T B0z,
                                T yStart, T dy, T LyH, T delta, int side,
                                int nxc, int nyc, int nzc) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= nxc * nzc)
    return;
  int i = tid / nzc;
  int k = tid % nzc;
  int j0, j1, j2;
  if (side == 0) {
    j0 = 0;
    j1 = 1;
    j2 = 2;
  } else {
    j0 = nyc - 1;
    j1 = nyc - 2;
    j2 = nyc - 3;
  }
  T yc0 = yStart + ((T)j0 - 0.5) * dy;
  T yc1 = yStart + ((T)j1 - 0.5) * dy;
  T yc2 = yStart + ((T)j2 - 0.5) * dy;
  T arg0 = (yc0 - LyH) / delta;
  T arg1 = (yc1 - LyH) / delta;
  T arg2 = (yc2 - LyH) / delta;
  int idx0 = (i * nyc + j0) * nzc + k;
  int idx1 = (i * nyc + j1) * nzc + k;
  int idx2 = (i * nyc + j2) * nzc + k;
  Bxc[idx0] = B0x * tanh(arg0);
  Byc[idx0] = B0y;
  Bzc[idx0] = B0z / cosh(arg0);
  Bzc[idx1] = B0z / cosh(arg1);
  Bzc[idx2] = B0z / cosh(arg2);
}

void gpuFixBforcefreeKernel(cudaSolverType* Bxc, cudaSolverType* Byc,
                            cudaSolverType* Bzc, cudaSolverType B0x,
                            cudaSolverType B0y, cudaSolverType B0z,
                            cudaSolverType yStart, cudaSolverType dy,
                            cudaSolverType LyH, cudaSolverType delta, int side,
                            int nxc, int nyc, int nzc, cudaStream_t stream) {
  int total = nxc * nzc;
  if (total <= 0)
    return;
  int blk = 256;
  k_fixBforcefree<<<(total + blk - 1) / blk, blk, 0, stream>>>(
      Bxc, Byc, Bzc, B0x, B0y, B0z, yStart, dy, LyH, delta, side, nxc, nyc,
      nzc);
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
//  Center-to-center Laplacian (for Poisson solver)
// =========================================================================

/** 7-point center Laplacian of the interior cells, written packed:
 *  lapK[(i nj + j) nk + k] for the interior cell (i+1, j+1, k+1).  Reads the
 *  face ghosts of fC, never the edge or corner ghosts. */
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

/** 7-point center Laplacian: lapC = d²f/dx² + d²f/dy² + d²f/dz².  Not used
 *  currently (gpuLapC2CToKrylov). */
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
 *  Not used currently (gpuChebyshevPoissonStep1/N). */
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

#define BJK_IDX3(i, j, k, ny, nz) ((i) * (ny) * (nz) + (j) * (nz) + (k))

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

  int idx = BJK_IDX3(i, j, k, nyn, nzn);
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

  int idx = BJK_IDX3(i, j, k, nyn, nzn);
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

  int nodeIdx = BJK_IDX3(i, j, k, nyn, nzn);
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
