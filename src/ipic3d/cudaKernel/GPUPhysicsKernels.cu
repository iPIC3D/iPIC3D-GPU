/**
 * @file GPUPhysicsKernels.cu
 * @brief CUDA kernel implementations for iPIC3D physics operations:
 *        MUdot, PIdot, perfect conductor BCs (fused sustensor).
 *
 * All kernels use a 3D thread-block layout (8×8×4 = 256) for 3D grids,
 * or a 2D layout for boundary face operations.
 */

#include "GPUPhysicsKernels.cuh"

#ifdef GPU_SOLVER

// =========================================================================
//  Thread-block configuration
// =========================================================================
static constexpr int BX = 8;
static constexpr int BY = 8;
static constexpr int BZ = 4;

#define IDX3(i, j, k, ny, nz) ((i) * (ny) * (nz) + (j) * (nz) + (k))

static inline dim3 interiorGrid3D(int nxn, int nyn, int nzn) {
  return dim3(((nxn - 2) + BX - 1) / BX, ((nyn - 2) + BY - 1) / BY,
              ((nzn - 2) + BZ - 1) / BZ);
}

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
  int i = blockIdx.x * BX + threadIdx.x + 1;
  int j = blockIdx.y * BY + threadIdx.y + 1;
  int k = blockIdx.z * BZ + threadIdx.z + 1;
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
  dim3 grid = interiorGrid3D(nxn, nyn, nzn);
  dim3 block(BX, BY, BZ);
  k_MUdotSpecies<<<grid, block, 0, stream>>>(
      MUdotX, MUdotY, MUdotZ, vX, vY, vZ, Bxn, Byn, Bzn, Bx_ext, By_ext, Bz_ext,
      rhons_is, beta, prefactor, nxn, nyn, nzn, firstSpecies);
  cudaErrChk(cudaGetLastError());
}

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
  int i = blockIdx.x * BX + threadIdx.x + 1;
  int j = blockIdx.y * BY + threadIdx.y + 1;
  int k = blockIdx.z * BZ + threadIdx.z + 1;
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
  dim3 grid = interiorGrid3D(nxn, nyn, nzn);
  dim3 block(BX, BY, BZ);
  k_PIdotSpecies<<<grid, block, 0, stream>>>(PIX, PIY, PIZ, vX, vY, vZ, Bxn,
                                             Byn, Bzn, Bx_ext, By_ext, Bz_ext,
                                             beta, nxn, nyn, nzn);
  cudaErrChk(cudaGetLastError());
}

// =========================================================================
//  Perfect conductor boundary condition kernels
//  Each kernel operates on one 2D face and fuses the sustensor computation.
// =========================================================================

// Maximum number of species for stack allocation in kernels
#define MAX_SPECIES 8

/**
 * Perfect conductor LEFT boundary, direction X (face i=1).
 * 2D grid over (j,k) ∈ [1, nyn-2] × [1, nzn-2].
 * Thread block: 16×16.
 */
template <typename T>
__global__ void k_perfectConductorLeftX(
    T* __restrict__ imageX, T* __restrict__ imageY, T* __restrict__ imageZ,
    const T* __restrict__ vX, const T* __restrict__ vY,
    const T* __restrict__ vZ, const T* __restrict__ Ex,
    const T* __restrict__ Jxh, const T* __restrict__ Bxn,
    const T* __restrict__ Byn, const T* __restrict__ Bzn,
    const T* __restrict__ Bx_ext, const T* __restrict__ By_ext,
    const T* __restrict__ Bz_ext, const T* __restrict__ rhons,
    const T* __restrict__ qom_d, int ns, T dt, T c, T th, T FourPI, T delt,
    int nxn, int nyn, int nzn) {
  int j = blockIdx.x * blockDim.x + threadIdx.x + 1;
  int k = blockIdx.y * blockDim.y + threadIdx.y + 1;
  if (j > nyn - 2 || k > nzn - 2)
    return;

  int bnd = IDX3(1, j, k, nyn, nzn);
  int speciesStride = nxn * nyn * nzn;

  // Compute sustensor at boundary
  T susxx = 1.0, susyx = 0.0, suszx = 0.0;
  for (int is = 0; is < ns; is++) {
    T beta = 0.5 * qom_d[is] * dt / c;
    T omcx = beta * (Bxn[bnd] + Bx_ext[bnd]);
    T omcy = beta * (Byn[bnd] + By_ext[bnd]);
    T omcz = beta * (Bzn[bnd] + Bz_ext[bnd]);
    T rho = rhons[is * speciesStride + bnd];
    T denom = FourPI / 2.0 * delt * dt / c * qom_d[is] * rho /
              (1.0 + omcx * omcx + omcy * omcy + omcz * omcz);
    susxx += (1.0 + omcx * omcx) * denom;
    susyx += (-omcz + omcx * omcy) * denom;
    suszx += (omcy + omcx * omcz) * denom;
  }

  // Apply perfect conductor BC
  imageX[bnd] = vX[bnd] - (Ex[bnd] - susyx * vY[bnd] - suszx * vZ[bnd] -
                           Jxh[bnd] * dt * th * FourPI) /
                              susxx;
  imageY[bnd] = vY[bnd];
  imageZ[bnd] = vZ[bnd];
}

template <typename T>
__global__ void k_perfectConductorLeftY(
    T* __restrict__ imageX, T* __restrict__ imageY, T* __restrict__ imageZ,
    const T* __restrict__ vX, const T* __restrict__ vY,
    const T* __restrict__ vZ, const T* __restrict__ Ey,
    const T* __restrict__ Jyh, const T* __restrict__ Bxn,
    const T* __restrict__ Byn, const T* __restrict__ Bzn,
    const T* __restrict__ Bx_ext, const T* __restrict__ By_ext,
    const T* __restrict__ Bz_ext, const T* __restrict__ rhons,
    const T* __restrict__ qom_d, int ns, T dt, T c, T th, T FourPI, T delt,
    int nxn, int nyn, int nzn) {
  int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
  int k = blockIdx.y * blockDim.y + threadIdx.y + 1;
  if (i > nxn - 2 || k > nzn - 2)
    return;

  int bnd = IDX3(i, 1, k, nyn, nzn);
  int speciesStride = nxn * nyn * nzn;

  T susxy = 0.0, susyy = 1.0, suszy = 0.0;
  for (int is = 0; is < ns; is++) {
    T beta = 0.5 * qom_d[is] * dt / c;
    T omcx = beta * (Bxn[bnd] + Bx_ext[bnd]);
    T omcy = beta * (Byn[bnd] + By_ext[bnd]);
    T omcz = beta * (Bzn[bnd] + Bz_ext[bnd]);
    T rho = rhons[is * speciesStride + bnd];
    T denom = FourPI / 2.0 * delt * dt / c * qom_d[is] * rho /
              (1.0 + omcx * omcx + omcy * omcy + omcz * omcz);
    susxy += (omcz + omcx * omcy) * denom;
    susyy += (1.0 + omcy * omcy) * denom;
    suszy += (-omcx + omcy * omcz) * denom;
  }

  imageX[bnd] = vX[bnd];
  imageY[bnd] = vY[bnd] - (Ey[bnd] - susxy * vX[bnd] - suszy * vZ[bnd] -
                           Jyh[bnd] * dt * th * FourPI) /
                              susyy;
  imageZ[bnd] = vZ[bnd];
}

template <typename T>
__global__ void k_perfectConductorLeftZ(
    T* __restrict__ imageX, T* __restrict__ imageY, T* __restrict__ imageZ,
    const T* __restrict__ vX, const T* __restrict__ vY,
    const T* __restrict__ vZ, const T* __restrict__ Ez,
    const T* __restrict__ Jzh, const T* __restrict__ Bxn,
    const T* __restrict__ Byn, const T* __restrict__ Bzn,
    const T* __restrict__ Bx_ext, const T* __restrict__ By_ext,
    const T* __restrict__ Bz_ext, const T* __restrict__ rhons,
    const T* __restrict__ qom_d, int ns, T dt, T c, T th, T FourPI, T delt,
    int nxn, int nyn, int nzn) {
  int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
  int j = blockIdx.y * blockDim.y + threadIdx.y + 1;
  if (i > nxn - 2 || j > nyn - 2)
    return;

  int bnd = IDX3(i, j, 1, nyn, nzn);
  int speciesStride = nxn * nyn * nzn;

  T susxz = 0.0, susyz = 0.0, suszz = 1.0;
  for (int is = 0; is < ns; is++) {
    T beta = 0.5 * qom_d[is] * dt / c;
    T omcx = beta * (Bxn[bnd] + Bx_ext[bnd]);
    T omcy = beta * (Byn[bnd] + By_ext[bnd]);
    T omcz = beta * (Bzn[bnd] + Bz_ext[bnd]);
    T rho = rhons[is * speciesStride + bnd];
    T denom = FourPI / 2.0 * delt * dt / c * qom_d[is] * rho /
              (1.0 + omcx * omcx + omcy * omcy + omcz * omcz);
    susxz += (-omcy + omcx * omcz) * denom;
    susyz += (omcx + omcy * omcz) * denom;
    suszz += (1.0 + omcz * omcz) * denom;
  }

  imageX[bnd] = vX[bnd];
  imageY[bnd] = vY[bnd];
  imageZ[bnd] = vZ[bnd] - (Ez[bnd] - susxz * vX[bnd] - susyz * vY[bnd] -
                           Jzh[bnd] * dt * th * FourPI) /
                              suszz;
}

// ---- RIGHT boundary kernels ----

template <typename T>
__global__ void k_perfectConductorRightX(
    T* __restrict__ imageX, T* __restrict__ imageY, T* __restrict__ imageZ,
    const T* __restrict__ vX, const T* __restrict__ vY,
    const T* __restrict__ vZ, const T* __restrict__ Ex,
    const T* __restrict__ Jxh, const T* __restrict__ Bxn,
    const T* __restrict__ Byn, const T* __restrict__ Bzn,
    const T* __restrict__ Bx_ext, const T* __restrict__ By_ext,
    const T* __restrict__ Bz_ext, const T* __restrict__ rhons,
    const T* __restrict__ qom_d, int ns, T dt, T c, T th, T FourPI, T delt,
    int nxn, int nyn, int nzn) {
  int j = blockIdx.x * blockDim.x + threadIdx.x + 1;
  int k = blockIdx.y * blockDim.y + threadIdx.y + 1;
  if (j > nyn - 2 || k > nzn - 2)
    return;

  int bnd = IDX3(nxn - 2, j, k, nyn, nzn);
  int speciesStride = nxn * nyn * nzn;

  T susxx = 1.0, susyx = 0.0, suszx = 0.0;
  for (int is = 0; is < ns; is++) {
    T beta = 0.5 * qom_d[is] * dt / c;
    T omcx = beta * (Bxn[bnd] + Bx_ext[bnd]);
    T omcy = beta * (Byn[bnd] + By_ext[bnd]);
    T omcz = beta * (Bzn[bnd] + Bz_ext[bnd]);
    T rho = rhons[is * speciesStride + bnd];
    T denom = FourPI / 2.0 * delt * dt / c * qom_d[is] * rho /
              (1.0 + omcx * omcx + omcy * omcy + omcz * omcz);
    susxx += (1.0 + omcx * omcx) * denom;
    susyx += (-omcz + omcx * omcy) * denom;
    suszx += (omcy + omcx * omcz) * denom;
  }

  imageX[bnd] = vX[bnd] - (Ex[bnd] - susyx * vY[bnd] - suszx * vZ[bnd] -
                           Jxh[bnd] * dt * th * FourPI) /
                              susxx;
  imageY[bnd] = vY[bnd];
  imageZ[bnd] = vZ[bnd];
}

template <typename T>
__global__ void k_perfectConductorRightY(
    T* __restrict__ imageX, T* __restrict__ imageY, T* __restrict__ imageZ,
    const T* __restrict__ vX, const T* __restrict__ vY,
    const T* __restrict__ vZ, const T* __restrict__ Ey,
    const T* __restrict__ Jyh, const T* __restrict__ Bxn,
    const T* __restrict__ Byn, const T* __restrict__ Bzn,
    const T* __restrict__ Bx_ext, const T* __restrict__ By_ext,
    const T* __restrict__ Bz_ext, const T* __restrict__ rhons,
    const T* __restrict__ qom_d, int ns, T dt, T c, T th, T FourPI, T delt,
    int nxn, int nyn, int nzn) {
  int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
  int k = blockIdx.y * blockDim.y + threadIdx.y + 1;
  if (i > nxn - 2 || k > nzn - 2)
    return;

  int bnd = IDX3(i, nyn - 2, k, nyn, nzn);
  int speciesStride = nxn * nyn * nzn;

  T susxy = 0.0, susyy = 1.0, suszy = 0.0;
  for (int is = 0; is < ns; is++) {
    T beta = 0.5 * qom_d[is] * dt / c;
    T omcx = beta * (Bxn[bnd] + Bx_ext[bnd]);
    T omcy = beta * (Byn[bnd] + By_ext[bnd]);
    T omcz = beta * (Bzn[bnd] + Bz_ext[bnd]);
    T rho = rhons[is * speciesStride + bnd];
    T denom = FourPI / 2.0 * delt * dt / c * qom_d[is] * rho /
              (1.0 + omcx * omcx + omcy * omcy + omcz * omcz);
    susxy += (omcz + omcx * omcy) * denom;
    susyy += (1.0 + omcy * omcy) * denom;
    suszy += (-omcx + omcy * omcz) * denom;
  }

  imageX[bnd] = vX[bnd];
  imageY[bnd] = vY[bnd] - (Ey[bnd] - susxy * vX[bnd] - suszy * vZ[bnd] -
                           Jyh[bnd] * dt * th * FourPI) /
                              susyy;
  imageZ[bnd] = vZ[bnd];
}

template <typename T>
__global__ void k_perfectConductorRightZ(
    T* __restrict__ imageX, T* __restrict__ imageY, T* __restrict__ imageZ,
    const T* __restrict__ vX, const T* __restrict__ vY,
    const T* __restrict__ vZ, const T* __restrict__ Ez,
    const T* __restrict__ Jzh, const T* __restrict__ Bxn,
    const T* __restrict__ Byn, const T* __restrict__ Bzn,
    const T* __restrict__ Bx_ext, const T* __restrict__ By_ext,
    const T* __restrict__ Bz_ext, const T* __restrict__ rhons,
    const T* __restrict__ qom_d, int ns, T dt, T c, T th, T FourPI, T delt,
    int nxn, int nyn, int nzn) {
  int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
  int j = blockIdx.y * blockDim.y + threadIdx.y + 1;
  if (i > nxn - 2 || j > nyn - 2)
    return;

  int bnd = IDX3(i, j, nzn - 2, nyn, nzn);
  int speciesStride = nxn * nyn * nzn;

  T susxz = 0.0, susyz = 0.0, suszz = 1.0;
  for (int is = 0; is < ns; is++) {
    T beta = 0.5 * qom_d[is] * dt / c;
    T omcx = beta * (Bxn[bnd] + Bx_ext[bnd]);
    T omcy = beta * (Byn[bnd] + By_ext[bnd]);
    T omcz = beta * (Bzn[bnd] + Bz_ext[bnd]);
    T rho = rhons[is * speciesStride + bnd];
    T denom = FourPI / 2.0 * delt * dt / c * qom_d[is] * rho /
              (1.0 + omcx * omcx + omcy * omcy + omcz * omcz);
    susxz += (-omcy + omcx * omcz) * denom;
    susyz += (omcx + omcy * omcz) * denom;
    suszz += (1.0 + omcz * omcz) * denom;
  }

  imageX[bnd] = vX[bnd];
  imageY[bnd] = vY[bnd];
  imageZ[bnd] = vZ[bnd] - (Ez[bnd] - susxz * vX[bnd] - susyz * vY[bnd] -
                           Jzh[bnd] * dt * th * FourPI) /
                              suszz;
}

// =========================================================================
//  Perfect conductor source kernels
// =========================================================================

template <typename T>
__global__ void k_perfectConductorLeftSX(T* __restrict__ vX, T* __restrict__ vY,
                                         T* __restrict__ vZ, T ebc1, T ebc2,
                                         int nyn, int nzn) {
  int j = blockIdx.x * blockDim.x + threadIdx.x + 1;
  int k = blockIdx.y * blockDim.y + threadIdx.y + 1;
  if (j > nyn - 2 || k > nzn - 2)
    return;
  int idx = 1 * nyn * nzn + j * nzn + k;
  vX[idx] = 0.0;
  vY[idx] = ebc1;
  vZ[idx] = ebc2;
}

template <typename T>
__global__ void k_perfectConductorLeftSY(T* __restrict__ vX, T* __restrict__ vY,
                                         T* __restrict__ vZ, T ebc0, T ebc2,
                                         int nxn, int nyn, int nzn) {
  int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
  int k = blockIdx.y * blockDim.y + threadIdx.y + 1;
  if (i > nxn - 2 || k > nzn - 2)
    return;
  int idx = i * nyn * nzn + 1 * nzn + k;
  vX[idx] = ebc0;
  vY[idx] = 0.0;
  vZ[idx] = ebc2;
}

template <typename T>
__global__ void k_perfectConductorLeftSZ(T* __restrict__ vX, T* __restrict__ vY,
                                         T* __restrict__ vZ, T ebc0, T ebc1,
                                         int nxn, int nyn, int nzn) {
  int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
  int j = blockIdx.y * blockDim.y + threadIdx.y + 1;
  if (i > nxn - 2 || j > nyn - 2)
    return;
  int idx = i * nyn * nzn + j * nzn + 1;
  vX[idx] = ebc0;
  vY[idx] = ebc1;
  vZ[idx] = 0.0;
}

template <typename T>
__global__ void k_perfectConductorRightSX(T* __restrict__ vX,
                                          T* __restrict__ vY,
                                          T* __restrict__ vZ, T ebc1, T ebc2,
                                          int nxn, int nyn, int nzn) {
  int j = blockIdx.x * blockDim.x + threadIdx.x + 1;
  int k = blockIdx.y * blockDim.y + threadIdx.y + 1;
  if (j > nyn - 2 || k > nzn - 2)
    return;
  int idx = (nxn - 2) * nyn * nzn + j * nzn + k;
  vX[idx] = 0.0;
  vY[idx] = ebc1;
  vZ[idx] = ebc2;
}

template <typename T>
__global__ void k_perfectConductorRightSY(T* __restrict__ vX,
                                          T* __restrict__ vY,
                                          T* __restrict__ vZ, T ebc0, T ebc2,
                                          int nxn, int nyn, int nzn) {
  int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
  int k = blockIdx.y * blockDim.y + threadIdx.y + 1;
  if (i > nxn - 2 || k > nzn - 2)
    return;
  int idx = i * nyn * nzn + (nyn - 2) * nzn + k;
  vX[idx] = ebc0;
  vY[idx] = 0.0;
  vZ[idx] = ebc2;
}

template <typename T>
__global__ void k_perfectConductorRightSZ(T* __restrict__ vX,
                                          T* __restrict__ vY,
                                          T* __restrict__ vZ, T ebc0, T ebc1,
                                          int nxn, int nyn, int nzn) {
  int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
  int j = blockIdx.y * blockDim.y + threadIdx.y + 1;
  if (i > nxn - 2 || j > nyn - 2)
    return;
  int idx = i * nyn * nzn + j * nzn + (nzn - 2);
  vX[idx] = ebc0;
  vY[idx] = ebc1;
  vZ[idx] = 0.0;
}

// =========================================================================
//  Host wrappers
// =========================================================================

static inline dim3 faceGrid2D(int d1, int d2) {
  return dim3(((d1 - 2) + 15) / 16, ((d2 - 2) + 15) / 16);
}
static const dim3 faceBlock2D(16, 16);

void gpuPerfectConductorLeft(
    cudaSolverType* imageX, cudaSolverType* imageY, cudaSolverType* imageZ,
    const cudaSolverType* vectX, const cudaSolverType* vectY,
    const cudaSolverType* vectZ, const cudaSolverType* Ex,
    const cudaSolverType* Ey, const cudaSolverType* Ez,
    const cudaSolverType* Jxh, const cudaSolverType* Jyh,
    const cudaSolverType* Jzh, const cudaSolverType* Bxn,
    const cudaSolverType* Byn, const cudaSolverType* Bzn,
    const cudaSolverType* Bx_ext, const cudaSolverType* By_ext,
    const cudaSolverType* Bz_ext, const cudaSolverType* rhons,
    const cudaSolverType* d_qom, int ns, cudaSolverType dt, cudaSolverType c,
    cudaSolverType th, cudaSolverType FourPI, cudaSolverType delt, int nxn,
    int nyn, int nzn, int dir, cudaStream_t stream) {
  switch (dir) {
  case 0: {
    dim3 grid = faceGrid2D(nyn, nzn);
    k_perfectConductorLeftX<<<grid, faceBlock2D, 0, stream>>>(
        imageX, imageY, imageZ, vectX, vectY, vectZ, Ex, Jxh, Bxn, Byn, Bzn,
        Bx_ext, By_ext, Bz_ext, rhons, d_qom, ns, dt, c, th, FourPI, delt, nxn,
        nyn, nzn);
  } break;
  case 1: {
    dim3 grid = faceGrid2D(nxn, nzn);
    k_perfectConductorLeftY<<<grid, faceBlock2D, 0, stream>>>(
        imageX, imageY, imageZ, vectX, vectY, vectZ, Ey, Jyh, Bxn, Byn, Bzn,
        Bx_ext, By_ext, Bz_ext, rhons, d_qom, ns, dt, c, th, FourPI, delt, nxn,
        nyn, nzn);
  } break;
  case 2: {
    dim3 grid = faceGrid2D(nxn, nyn);
    k_perfectConductorLeftZ<<<grid, faceBlock2D, 0, stream>>>(
        imageX, imageY, imageZ, vectX, vectY, vectZ, Ez, Jzh, Bxn, Byn, Bzn,
        Bx_ext, By_ext, Bz_ext, rhons, d_qom, ns, dt, c, th, FourPI, delt, nxn,
        nyn, nzn);
  } break;
  }
  cudaErrChk(cudaGetLastError());
}

void gpuPerfectConductorRight(
    cudaSolverType* imageX, cudaSolverType* imageY, cudaSolverType* imageZ,
    const cudaSolverType* vectX, const cudaSolverType* vectY,
    const cudaSolverType* vectZ, const cudaSolverType* Ex,
    const cudaSolverType* Ey, const cudaSolverType* Ez,
    const cudaSolverType* Jxh, const cudaSolverType* Jyh,
    const cudaSolverType* Jzh, const cudaSolverType* Bxn,
    const cudaSolverType* Byn, const cudaSolverType* Bzn,
    const cudaSolverType* Bx_ext, const cudaSolverType* By_ext,
    const cudaSolverType* Bz_ext, const cudaSolverType* rhons,
    const cudaSolverType* d_qom, int ns, cudaSolverType dt, cudaSolverType c,
    cudaSolverType th, cudaSolverType FourPI, cudaSolverType delt, int nxn,
    int nyn, int nzn, int dir, cudaStream_t stream) {
  switch (dir) {
  case 0: {
    dim3 grid = faceGrid2D(nyn, nzn);
    k_perfectConductorRightX<<<grid, faceBlock2D, 0, stream>>>(
        imageX, imageY, imageZ, vectX, vectY, vectZ, Ex, Jxh, Bxn, Byn, Bzn,
        Bx_ext, By_ext, Bz_ext, rhons, d_qom, ns, dt, c, th, FourPI, delt, nxn,
        nyn, nzn);
  } break;
  case 1: {
    dim3 grid = faceGrid2D(nxn, nzn);
    k_perfectConductorRightY<<<grid, faceBlock2D, 0, stream>>>(
        imageX, imageY, imageZ, vectX, vectY, vectZ, Ey, Jyh, Bxn, Byn, Bzn,
        Bx_ext, By_ext, Bz_ext, rhons, d_qom, ns, dt, c, th, FourPI, delt, nxn,
        nyn, nzn);
  } break;
  case 2: {
    dim3 grid = faceGrid2D(nxn, nyn);
    k_perfectConductorRightZ<<<grid, faceBlock2D, 0, stream>>>(
        imageX, imageY, imageZ, vectX, vectY, vectZ, Ez, Jzh, Bxn, Byn, Bzn,
        Bx_ext, By_ext, Bz_ext, rhons, d_qom, ns, dt, c, th, FourPI, delt, nxn,
        nyn, nzn);
  } break;
  }
  cudaErrChk(cudaGetLastError());
}

void gpuPerfectConductorLeftS(cudaSolverType* vectorX, cudaSolverType* vectorY,
                              cudaSolverType* vectorZ, cudaSolverType ebc0,
                              cudaSolverType ebc1, cudaSolverType ebc2, int nxn,
                              int nyn, int nzn, int dir, cudaStream_t stream) {
  switch (dir) {
  case 0: {
    dim3 grid = faceGrid2D(nyn, nzn);
    k_perfectConductorLeftSX<<<grid, faceBlock2D, 0, stream>>>(
        vectorX, vectorY, vectorZ, ebc1, ebc2, nyn, nzn);
  } break;
  case 1: {
    dim3 grid = faceGrid2D(nxn, nzn);
    k_perfectConductorLeftSY<<<grid, faceBlock2D, 0, stream>>>(
        vectorX, vectorY, vectorZ, ebc0, ebc2, nxn, nyn, nzn);
  } break;
  case 2: {
    dim3 grid = faceGrid2D(nxn, nyn);
    k_perfectConductorLeftSZ<<<grid, faceBlock2D, 0, stream>>>(
        vectorX, vectorY, vectorZ, ebc0, ebc1, nxn, nyn, nzn);
  } break;
  }
  cudaErrChk(cudaGetLastError());
}

void gpuPerfectConductorRightS(cudaSolverType* vectorX, cudaSolverType* vectorY,
                               cudaSolverType* vectorZ, cudaSolverType ebc0,
                               cudaSolverType ebc1, cudaSolverType ebc2,
                               int nxn, int nyn, int nzn, int dir,
                               cudaStream_t stream) {
  switch (dir) {
  case 0: {
    dim3 grid = faceGrid2D(nyn, nzn);
    k_perfectConductorRightSX<<<grid, faceBlock2D, 0, stream>>>(
        vectorX, vectorY, vectorZ, ebc1, ebc2, nxn, nyn, nzn);
  } break;
  case 1: {
    dim3 grid = faceGrid2D(nxn, nzn);
    k_perfectConductorRightSY<<<grid, faceBlock2D, 0, stream>>>(
        vectorX, vectorY, vectorZ, ebc0, ebc2, nxn, nyn, nzn);
  } break;
  case 2: {
    dim3 grid = faceGrid2D(nxn, nyn);
    k_perfectConductorRightSZ<<<grid, faceBlock2D, 0, stream>>>(
        vectorX, vectorY, vectorZ, ebc0, ebc1, nxn, nyn, nzn);
  } break;
  }
  cudaErrChk(cudaGetLastError());
}

#undef IDX3
#undef MAX_SPECIES

// =========================================================================
//  adjustNonPeriodicDensities: cudaSolverType boundary-face values on GPU
// =========================================================================

// Kernel: cudaSolverType values on X-face (i = fixedI)
template <typename T>
__global__ void k_doubleFaceX(T* __restrict__ arr, int fixedI, int nyn,
                              int nzn) {
  int j = blockIdx.x * blockDim.x + threadIdx.x + 1;
  int k = blockIdx.y * blockDim.y + threadIdx.y + 1;
  if (j >= nyn - 1 || k >= nzn - 1)
    return;
  arr[(fixedI * nyn + j) * nzn + k] *= 2.0;
}

// Kernel: cudaSolverType values on Y-face (j = fixedJ)
template <typename T>
__global__ void k_doubleFaceY(T* __restrict__ arr, int fixedJ, int nxn, int nyn,
                              int nzn) {
  int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
  int k = blockIdx.y * blockDim.y + threadIdx.y + 1;
  if (i >= nxn - 1 || k >= nzn - 1)
    return;
  arr[(i * nyn + fixedJ) * nzn + k] *= 2.0;
}

// Kernel: cudaSolverType values on Z-face (k = fixedK)
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

/** Zero 3 arrays on a single face plane (interior [1..n-2]). */
template <typename T>
__global__ void k_zeroFacePlane3(T* X, T* Y, T* Z, int dir, int faceIdx, int nx,
                                 int ny, int nz) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int d1, d2;
  if (dir == 0) {
    d1 = ny - 2;
    d2 = nz - 2;
  } else if (dir == 1) {
    d1 = nx - 2;
    d2 = nz - 2;
  } else {
    d1 = nx - 2;
    d2 = ny - 2;
  }
  if (tid >= d1 * d2)
    return;
  int a = tid / d2 + 1;
  int b = tid % d2 + 1;
  int i, j, k;
  if (dir == 0) {
    i = faceIdx;
    j = a;
    k = b;
  } else if (dir == 1) {
    i = a;
    j = faceIdx;
    k = b;
  } else {
    i = a;
    j = b;
    k = faceIdx;
  }
  int idx = (i * ny + j) * nz + k;
  X[idx] = 0.0;
  Y[idx] = 0.0;
  Z[idx] = 0.0;
}

void gpuOpenBCZeroFace3(cudaSolverType* X, cudaSolverType* Y, cudaSolverType* Z,
                        int dir, int faceIdx, int nx, int ny, int nz,
                        cudaStream_t stream) {
  int d1, d2;
  if (dir == 0) {
    d1 = ny - 2;
    d2 = nz - 2;
  } else if (dir == 1) {
    d1 = nx - 2;
    d2 = nz - 2;
  } else {
    d1 = nx - 2;
    d2 = ny - 2;
  }
  int total = d1 * d2;
  if (total <= 0)
    return;
  int blk = 256;
  k_zeroFacePlane3<<<(total + blk - 1) / blk, blk, 0, stream>>>(
      X, Y, Z, dir, faceIdx, nx, ny, nz);
  cudaErrChk(cudaGetLastError());
}

/** Set image = vect - injE on a single face plane (interior [1..n-2]). */
template <typename T>
__global__ void k_imageDiffFacePlane3(T* imX, T* imY, T* imZ, const T* vX,
                                      const T* vY, const T* vZ, T injE0,
                                      T injE1, T injE2, int dir, int faceIdx,
                                      int nx, int ny, int nz) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int d1, d2;
  if (dir == 0) {
    d1 = ny - 2;
    d2 = nz - 2;
  } else if (dir == 1) {
    d1 = nx - 2;
    d2 = nz - 2;
  } else {
    d1 = nx - 2;
    d2 = ny - 2;
  }
  if (tid >= d1 * d2)
    return;
  int a = tid / d2 + 1;
  int b = tid % d2 + 1;
  int i, j, k;
  if (dir == 0) {
    i = faceIdx;
    j = a;
    k = b;
  } else if (dir == 1) {
    i = a;
    j = faceIdx;
    k = b;
  } else {
    i = a;
    j = b;
    k = faceIdx;
  }
  int idx = (i * ny + j) * nz + k;
  imX[idx] = vX[idx] - injE0;
  imY[idx] = vY[idx] - injE1;
  imZ[idx] = vZ[idx] - injE2;
}

void gpuOpenBCImageDiffFace3(cudaSolverType* imX, cudaSolverType* imY,
                             cudaSolverType* imZ, const cudaSolverType* vX,
                             const cudaSolverType* vY, const cudaSolverType* vZ,
                             cudaSolverType injE0, cudaSolverType injE1,
                             cudaSolverType injE2, int dir, int faceIdx, int nx,
                             int ny, int nz, cudaStream_t stream) {
  int d1, d2;
  if (dir == 0) {
    d1 = ny - 2;
    d2 = nz - 2;
  } else if (dir == 1) {
    d1 = nx - 2;
    d2 = nz - 2;
  } else {
    d1 = nx - 2;
    d2 = ny - 2;
  }
  int total = d1 * d2;
  if (total <= 0)
    return;
  int blk = 256;
  k_imageDiffFacePlane3<<<(total + blk - 1) / blk, blk, 0, stream>>>(
      imX, imY, imZ, vX, vY, vZ, injE0, injE1, injE2, dir, faceIdx, nx, ny, nz);
  cudaErrChk(cudaGetLastError());
}

/**
 * SAL blend: v[idx] = v[idx]*sal + target*(1-sal) on boundary layers.
 * dir: 0=X, 1=Y, 2=Z.
 * ascending: true if sal increases from layerStart to layerEnd.
 */
template <typename T>
__global__ void k_salBlend3(T* X, T* Y, T* Z, T tgtX, T tgtY, T tgtZ, int dir,
                            int layerStart, int layerEnd, T invNLayers,
                            int ascending, int nx, int ny, int nz) {
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
  X[idx] = X[idx] * sal + tgtX * (1.0 - sal);
  Y[idx] = Y[idx] * sal + tgtY * (1.0 - sal);
  Z[idx] = Z[idx] * sal + tgtZ * (1.0 - sal);
}

void gpuSALBlendLayers3(cudaSolverType* X, cudaSolverType* Y, cudaSolverType* Z,
                        cudaSolverType tgtX, cudaSolverType tgtY,
                        cudaSolverType tgtZ, int dir, int layerStart,
                        int layerEnd, cudaSolverType invNLayers, int ascending,
                        int nx, int ny, int nz, cudaStream_t stream) {
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
      ascending, nx, ny, nz);
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
 *  Uses CENTER Y coordinate for the tanh profile (matches CPU).
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

/** Fix center-based B for ForceFree on Y-boundary layers.
 *  Bx = B0x*tanh, Bz = B0z/cosh profile on up to 3 layers. */
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
//  Subtract grad(PSI) from B on boundary layers (divB cleaning)
// =========================================================================

/** B -= gradPSI on boundary layers [layerStart..layerEnd] in direction dir. */
template <typename T>
__global__ void k_subLayers3(T* BxN, T* ByN, T* BzN, const T* gX, const T* gY,
                             const T* gZ, int dir, int layerStart, int layerEnd,
                             int nxn, int nyn, int nzn) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int nLayers = layerEnd - layerStart + 1;
  int d1, d2;
  if (dir == 0) {
    d1 = nyn;
    d2 = nzn;
  } else if (dir == 1) {
    d1 = nxn;
    d2 = nzn;
  } else {
    d1 = nxn;
    d2 = nyn;
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
  int idx = (i * nyn + j) * nzn + k;
  BxN[idx] -= gX[idx];
  ByN[idx] -= gY[idx];
  BzN[idx] -= gZ[idx];
}

void gpuSubBoundaryLayers3(cudaSolverType* BxN, cudaSolverType* ByN,
                           cudaSolverType* BzN, const cudaSolverType* gX,
                           const cudaSolverType* gY, const cudaSolverType* gZ,
                           int dir, int layerStart, int layerEnd, int nxn,
                           int nyn, int nzn, cudaStream_t stream) {
  int nLayers = layerEnd - layerStart + 1;
  int d1, d2;
  if (dir == 0) {
    d1 = nyn;
    d2 = nzn;
  } else if (dir == 1) {
    d1 = nxn;
    d2 = nzn;
  } else {
    d1 = nxn;
    d2 = nyn;
  }
  int total = nLayers * d1 * d2;
  if (total <= 0)
    return;
  int blk = 256;
  k_subLayers3<<<(total + blk - 1) / blk, blk, 0, stream>>>(
      BxN, ByN, BzN, gX, gY, gZ, dir, layerStart, layerEnd, nxn, nyn, nzn);
  cudaErrChk(cudaGetLastError());
}

// =========================================================================
//  Center-to-center Laplacian (for Poisson solver)
// =========================================================================

/** 7-point center Laplacian: lapC = d²f/dx² + d²f/dy² + d²f/dz². */
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

// =========================================================================
//  Block-Jacobi preconditioner kernel
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
  int i = blockIdx.x * BX + threadIdx.x + 1;
  int j = blockIdx.y * BY + threadIdx.y + 1;
  int k = blockIdx.z * BZ + threadIdx.z + 1;
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
  dim3 grid = interiorGrid3D(nxn, nyn, nzn);
  dim3 block(BX, BY, BZ);
  k_BlockJacobiPrecond<<<grid, block, 0, stream>>>(
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
  int i = blockIdx.x * BX + threadIdx.x + 1;
  int j = blockIdx.y * BY + threadIdx.y + 1;
  int k = blockIdx.z * BZ + threadIdx.z + 1;
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
  dim3 grid = interiorGrid3D(nxn, nyn, nzn);
  dim3 block(BX, BY, BZ);
  k_PrecomputeBlockJacobiInv<<<grid, block, 0, stream>>>(
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
  int i = blockIdx.x * BX + threadIdx.x + 1;
  int j = blockIdx.y * BY + threadIdx.y + 1;
  int k = blockIdx.z * BZ + threadIdx.z + 1;
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
  dim3 grid = interiorGrid3D(nxn, nyn, nzn);
  dim3 block(BX, BY, BZ);
  k_ApplyBlockJacobiInvKrylov<<<grid, block, 0, stream>>>(zKrylov, rKrylov,
                                                          Dinv, nxn, nyn, nzn);
  cudaErrChk(cudaGetLastError());
}

#endif // GPU_SOLVER
