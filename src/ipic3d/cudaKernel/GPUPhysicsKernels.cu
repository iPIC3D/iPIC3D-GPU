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

static inline dim3 interiorGrid3D(int nxn, int nyn, int nzn)
{
    return dim3(((nxn - 2) + BX - 1) / BX,
                ((nyn - 2) + BY - 1) / BY,
                ((nzn - 2) + BZ - 1) / BZ);
}

// =========================================================================
//  MUdot kernel
// =========================================================================

__global__ void k_MUdotSpecies(
    double* __restrict__ MUdotX,
    double* __restrict__ MUdotY,
    double* __restrict__ MUdotZ,
    const double* __restrict__ vX,
    const double* __restrict__ vY,
    const double* __restrict__ vZ,
    const double* __restrict__ Bxn,
    const double* __restrict__ Byn,
    const double* __restrict__ Bzn,
    const double* __restrict__ Bx_ext,
    const double* __restrict__ By_ext,
    const double* __restrict__ Bz_ext,
    const double* __restrict__ rhons_is,
    double beta, double prefactor,
    int nxn, int nyn, int nzn,
    bool firstSpecies)
{
    int i = blockIdx.x * BX + threadIdx.x + 1;
    int j = blockIdx.y * BY + threadIdx.y + 1;
    int k = blockIdx.z * BZ + threadIdx.z + 1;
    if (i > nxn - 2 || j > nyn - 2 || k > nzn - 2) return;

    int idx = IDX3(i, j, k, nyn, nzn);

    double omcx = beta * (Bxn[idx] + Bx_ext[idx]);
    double omcy = beta * (Byn[idx] + By_ext[idx]);
    double omcz = beta * (Bzn[idx] + Bz_ext[idx]);

    double vx = vX[idx], vy = vY[idx], vz = vZ[idx];
    double edotb = vx * omcx + vy * omcy + vz * omcz;
    double denom = prefactor * rhons_is[idx] /
                   (1.0 + omcx * omcx + omcy * omcy + omcz * omcz);

    double dx = (vx + (vy * omcz - vz * omcy + edotb * omcx)) * denom;
    double dy = (vy + (vz * omcx - vx * omcz + edotb * omcy)) * denom;
    double dz = (vz + (vx * omcy - vy * omcx + edotb * omcz)) * denom;

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

void gpuMUdotSpecies(double* MUdotX, double* MUdotY, double* MUdotZ,
                     const double* vX, const double* vY, const double* vZ,
                     const double* Bxn, const double* Byn, const double* Bzn,
                     const double* Bx_ext, const double* By_ext, const double* Bz_ext,
                     const double* rhons_is,
                     double beta, double prefactor,
                     int nxn, int nyn, int nzn,
                     bool firstSpecies,
                     cudaStream_t stream)
{
    dim3 grid = interiorGrid3D(nxn, nyn, nzn);
    dim3 block(BX, BY, BZ);
    k_MUdotSpecies<<<grid, block, 0, stream>>>(
        MUdotX, MUdotY, MUdotZ,
        vX, vY, vZ,
        Bxn, Byn, Bzn,
        Bx_ext, By_ext, Bz_ext,
        rhons_is, beta, prefactor,
        nxn, nyn, nzn, firstSpecies);
    cudaErrChk(cudaGetLastError());
}

// =========================================================================
//  PIdot kernel
// =========================================================================

__global__ void k_PIdotSpecies(
    double* __restrict__ PIX,
    double* __restrict__ PIY,
    double* __restrict__ PIZ,
    const double* __restrict__ vX,
    const double* __restrict__ vY,
    const double* __restrict__ vZ,
    const double* __restrict__ Bxn,
    const double* __restrict__ Byn,
    const double* __restrict__ Bzn,
    const double* __restrict__ Bx_ext,
    const double* __restrict__ By_ext,
    const double* __restrict__ Bz_ext,
    double beta,
    int nxn, int nyn, int nzn)
{
    int i = blockIdx.x * BX + threadIdx.x + 1;
    int j = blockIdx.y * BY + threadIdx.y + 1;
    int k = blockIdx.z * BZ + threadIdx.z + 1;
    if (i > nxn - 2 || j > nyn - 2 || k > nzn - 2) return;

    int idx = IDX3(i, j, k, nyn, nzn);

    double omcx = beta * (Bxn[idx] + Bx_ext[idx]);
    double omcy = beta * (Byn[idx] + By_ext[idx]);
    double omcz = beta * (Bzn[idx] + Bz_ext[idx]);

    double vx = vX[idx], vy = vY[idx], vz = vZ[idx];
    double edotb = vx * omcx + vy * omcy + vz * omcz;
    double denom = 1.0 / (1.0 + omcx * omcx + omcy * omcy + omcz * omcz);

    PIX[idx] += (vx + (vy * omcz - vz * omcy + edotb * omcx)) * denom;
    PIY[idx] += (vy + (vz * omcx - vx * omcz + edotb * omcy)) * denom;
    PIZ[idx] += (vz + (vx * omcy - vy * omcx + edotb * omcz)) * denom;
}

void gpuPIdotSpecies(double* PIX, double* PIY, double* PIZ,
                     const double* vX, const double* vY, const double* vZ,
                     const double* Bxn, const double* Byn, const double* Bzn,
                     const double* Bx_ext, const double* By_ext, const double* Bz_ext,
                     double beta,
                     int nxn, int nyn, int nzn,
                     cudaStream_t stream)
{
    dim3 grid = interiorGrid3D(nxn, nyn, nzn);
    dim3 block(BX, BY, BZ);
    k_PIdotSpecies<<<grid, block, 0, stream>>>(
        PIX, PIY, PIZ,
        vX, vY, vZ,
        Bxn, Byn, Bzn,
        Bx_ext, By_ext, Bz_ext,
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
__global__ void k_perfectConductorLeftX(
    double* __restrict__ imageX, double* __restrict__ imageY, double* __restrict__ imageZ,
    const double* __restrict__ vX, const double* __restrict__ vY, const double* __restrict__ vZ,
    const double* __restrict__ Ex,
    const double* __restrict__ Jxh,
    const double* __restrict__ Bxn, const double* __restrict__ Byn, const double* __restrict__ Bzn,
    const double* __restrict__ Bx_ext, const double* __restrict__ By_ext, const double* __restrict__ Bz_ext,
    const double* __restrict__ rhons,
    const double* __restrict__ qom_d,
    int ns,
    double dt, double c, double th, double FourPI, double delt,
    int nxn, int nyn, int nzn)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int k = blockIdx.y * blockDim.y + threadIdx.y + 1;
    if (j > nyn - 2 || k > nzn - 2) return;

    int bnd = IDX3(1, j, k, nyn, nzn);
    int speciesStride = nxn * nyn * nzn;

    // Compute sustensor at boundary
    double susxx = 1.0, susyx = 0.0, suszx = 0.0;
    for (int is = 0; is < ns; is++) {
        double beta = 0.5 * qom_d[is] * dt / c;
        double omcx = beta * (Bxn[bnd] + Bx_ext[bnd]);
        double omcy = beta * (Byn[bnd] + By_ext[bnd]);
        double omcz = beta * (Bzn[bnd] + Bz_ext[bnd]);
        double rho = rhons[is * speciesStride + bnd];
        double denom = FourPI / 2.0 * delt * dt / c * qom_d[is] * rho /
                       (1.0 + omcx * omcx + omcy * omcy + omcz * omcz);
        susxx += (1.0 + omcx * omcx) * denom;
        susyx += (-omcz + omcx * omcy) * denom;
        suszx += (omcy + omcx * omcz) * denom;
    }

    // Apply perfect conductor BC
    imageX[bnd] = vX[bnd] - (Ex[bnd] - susyx * vY[bnd] - suszx * vZ[bnd]
                              - Jxh[bnd] * dt * th * FourPI) / susxx;
    imageY[bnd] = vY[bnd];
    imageZ[bnd] = vZ[bnd];
}

__global__ void k_perfectConductorLeftY(
    double* __restrict__ imageX, double* __restrict__ imageY, double* __restrict__ imageZ,
    const double* __restrict__ vX, const double* __restrict__ vY, const double* __restrict__ vZ,
    const double* __restrict__ Ey,
    const double* __restrict__ Jyh,
    const double* __restrict__ Bxn, const double* __restrict__ Byn, const double* __restrict__ Bzn,
    const double* __restrict__ Bx_ext, const double* __restrict__ By_ext, const double* __restrict__ Bz_ext,
    const double* __restrict__ rhons,
    const double* __restrict__ qom_d,
    int ns,
    double dt, double c, double th, double FourPI, double delt,
    int nxn, int nyn, int nzn)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int k = blockIdx.y * blockDim.y + threadIdx.y + 1;
    if (i > nxn - 2 || k > nzn - 2) return;

    int bnd = IDX3(i, 1, k, nyn, nzn);
    int speciesStride = nxn * nyn * nzn;

    double susxy = 0.0, susyy = 1.0, suszy = 0.0;
    for (int is = 0; is < ns; is++) {
        double beta = 0.5 * qom_d[is] * dt / c;
        double omcx = beta * (Bxn[bnd] + Bx_ext[bnd]);
        double omcy = beta * (Byn[bnd] + By_ext[bnd]);
        double omcz = beta * (Bzn[bnd] + Bz_ext[bnd]);
        double rho = rhons[is * speciesStride + bnd];
        double denom = FourPI / 2.0 * delt * dt / c * qom_d[is] * rho /
                       (1.0 + omcx * omcx + omcy * omcy + omcz * omcz);
        susxy += (omcz + omcx * omcy) * denom;
        susyy += (1.0 + omcy * omcy) * denom;
        suszy += (-omcx + omcy * omcz) * denom;
    }

    imageX[bnd] = vX[bnd];
    imageY[bnd] = vY[bnd] - (Ey[bnd] - susxy * vX[bnd] - suszy * vZ[bnd]
                              - Jyh[bnd] * dt * th * FourPI) / susyy;
    imageZ[bnd] = vZ[bnd];
}

__global__ void k_perfectConductorLeftZ(
    double* __restrict__ imageX, double* __restrict__ imageY, double* __restrict__ imageZ,
    const double* __restrict__ vX, const double* __restrict__ vY, const double* __restrict__ vZ,
    const double* __restrict__ Ez,
    const double* __restrict__ Jzh,
    const double* __restrict__ Bxn, const double* __restrict__ Byn, const double* __restrict__ Bzn,
    const double* __restrict__ Bx_ext, const double* __restrict__ By_ext, const double* __restrict__ Bz_ext,
    const double* __restrict__ rhons,
    const double* __restrict__ qom_d,
    int ns,
    double dt, double c, double th, double FourPI, double delt,
    int nxn, int nyn, int nzn)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int j = blockIdx.y * blockDim.y + threadIdx.y + 1;
    if (i > nxn - 2 || j > nyn - 2) return;

    int bnd = IDX3(i, j, 1, nyn, nzn);
    int speciesStride = nxn * nyn * nzn;

    double susxz = 0.0, susyz = 0.0, suszz = 1.0;
    for (int is = 0; is < ns; is++) {
        double beta = 0.5 * qom_d[is] * dt / c;
        double omcx = beta * (Bxn[bnd] + Bx_ext[bnd]);
        double omcy = beta * (Byn[bnd] + By_ext[bnd]);
        double omcz = beta * (Bzn[bnd] + Bz_ext[bnd]);
        double rho = rhons[is * speciesStride + bnd];
        double denom = FourPI / 2.0 * delt * dt / c * qom_d[is] * rho /
                       (1.0 + omcx * omcx + omcy * omcy + omcz * omcz);
        susxz += (-omcy + omcx * omcz) * denom;
        susyz += (omcx + omcy * omcz) * denom;
        suszz += (1.0 + omcz * omcz) * denom;
    }

    imageX[bnd] = vX[bnd];
    imageY[bnd] = vY[bnd];
    imageZ[bnd] = vZ[bnd] - (Ez[bnd] - susxz * vX[bnd] - susyz * vY[bnd]
                              - Jzh[bnd] * dt * th * FourPI) / suszz;
}

// ---- RIGHT boundary kernels ----

__global__ void k_perfectConductorRightX(
    double* __restrict__ imageX, double* __restrict__ imageY, double* __restrict__ imageZ,
    const double* __restrict__ vX, const double* __restrict__ vY, const double* __restrict__ vZ,
    const double* __restrict__ Ex,
    const double* __restrict__ Jxh,
    const double* __restrict__ Bxn, const double* __restrict__ Byn, const double* __restrict__ Bzn,
    const double* __restrict__ Bx_ext, const double* __restrict__ By_ext, const double* __restrict__ Bz_ext,
    const double* __restrict__ rhons,
    const double* __restrict__ qom_d,
    int ns,
    double dt, double c, double th, double FourPI, double delt,
    int nxn, int nyn, int nzn)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int k = blockIdx.y * blockDim.y + threadIdx.y + 1;
    if (j > nyn - 2 || k > nzn - 2) return;

    int bnd = IDX3(nxn - 2, j, k, nyn, nzn);
    int speciesStride = nxn * nyn * nzn;

    double susxx = 1.0, susyx = 0.0, suszx = 0.0;
    for (int is = 0; is < ns; is++) {
        double beta = 0.5 * qom_d[is] * dt / c;
        double omcx = beta * (Bxn[bnd] + Bx_ext[bnd]);
        double omcy = beta * (Byn[bnd] + By_ext[bnd]);
        double omcz = beta * (Bzn[bnd] + Bz_ext[bnd]);
        double rho = rhons[is * speciesStride + bnd];
        double denom = FourPI / 2.0 * delt * dt / c * qom_d[is] * rho /
                       (1.0 + omcx * omcx + omcy * omcy + omcz * omcz);
        susxx += (1.0 + omcx * omcx) * denom;
        susyx += (-omcz + omcx * omcy) * denom;
        suszx += (omcy + omcx * omcz) * denom;
    }

    imageX[bnd] = vX[bnd] - (Ex[bnd] - susyx * vY[bnd] - suszx * vZ[bnd]
                              - Jxh[bnd] * dt * th * FourPI) / susxx;
    imageY[bnd] = vY[bnd];
    imageZ[bnd] = vZ[bnd];
}

__global__ void k_perfectConductorRightY(
    double* __restrict__ imageX, double* __restrict__ imageY, double* __restrict__ imageZ,
    const double* __restrict__ vX, const double* __restrict__ vY, const double* __restrict__ vZ,
    const double* __restrict__ Ey,
    const double* __restrict__ Jyh,
    const double* __restrict__ Bxn, const double* __restrict__ Byn, const double* __restrict__ Bzn,
    const double* __restrict__ Bx_ext, const double* __restrict__ By_ext, const double* __restrict__ Bz_ext,
    const double* __restrict__ rhons,
    const double* __restrict__ qom_d,
    int ns,
    double dt, double c, double th, double FourPI, double delt,
    int nxn, int nyn, int nzn)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int k = blockIdx.y * blockDim.y + threadIdx.y + 1;
    if (i > nxn - 2 || k > nzn - 2) return;

    int bnd = IDX3(i, nyn - 2, k, nyn, nzn);
    int speciesStride = nxn * nyn * nzn;

    double susxy = 0.0, susyy = 1.0, suszy = 0.0;
    for (int is = 0; is < ns; is++) {
        double beta = 0.5 * qom_d[is] * dt / c;
        double omcx = beta * (Bxn[bnd] + Bx_ext[bnd]);
        double omcy = beta * (Byn[bnd] + By_ext[bnd]);
        double omcz = beta * (Bzn[bnd] + Bz_ext[bnd]);
        double rho = rhons[is * speciesStride + bnd];
        double denom = FourPI / 2.0 * delt * dt / c * qom_d[is] * rho /
                       (1.0 + omcx * omcx + omcy * omcy + omcz * omcz);
        susxy += (omcz + omcx * omcy) * denom;
        susyy += (1.0 + omcy * omcy) * denom;
        suszy += (-omcx + omcy * omcz) * denom;
    }

    imageX[bnd] = vX[bnd];
    imageY[bnd] = vY[bnd] - (Ey[bnd] - susxy * vX[bnd] - suszy * vZ[bnd]
                              - Jyh[bnd] * dt * th * FourPI) / susyy;
    imageZ[bnd] = vZ[bnd];
}

__global__ void k_perfectConductorRightZ(
    double* __restrict__ imageX, double* __restrict__ imageY, double* __restrict__ imageZ,
    const double* __restrict__ vX, const double* __restrict__ vY, const double* __restrict__ vZ,
    const double* __restrict__ Ez,
    const double* __restrict__ Jzh,
    const double* __restrict__ Bxn, const double* __restrict__ Byn, const double* __restrict__ Bzn,
    const double* __restrict__ Bx_ext, const double* __restrict__ By_ext, const double* __restrict__ Bz_ext,
    const double* __restrict__ rhons,
    const double* __restrict__ qom_d,
    int ns,
    double dt, double c, double th, double FourPI, double delt,
    int nxn, int nyn, int nzn)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int j = blockIdx.y * blockDim.y + threadIdx.y + 1;
    if (i > nxn - 2 || j > nyn - 2) return;

    int bnd = IDX3(i, j, nzn - 2, nyn, nzn);
    int speciesStride = nxn * nyn * nzn;

    double susxz = 0.0, susyz = 0.0, suszz = 1.0;
    for (int is = 0; is < ns; is++) {
        double beta = 0.5 * qom_d[is] * dt / c;
        double omcx = beta * (Bxn[bnd] + Bx_ext[bnd]);
        double omcy = beta * (Byn[bnd] + By_ext[bnd]);
        double omcz = beta * (Bzn[bnd] + Bz_ext[bnd]);
        double rho = rhons[is * speciesStride + bnd];
        double denom = FourPI / 2.0 * delt * dt / c * qom_d[is] * rho /
                       (1.0 + omcx * omcx + omcy * omcy + omcz * omcz);
        susxz += (-omcy + omcx * omcz) * denom;
        susyz += (omcx + omcy * omcz) * denom;
        suszz += (1.0 + omcz * omcz) * denom;
    }

    imageX[bnd] = vX[bnd];
    imageY[bnd] = vY[bnd];
    imageZ[bnd] = vZ[bnd] - (Ez[bnd] - susxz * vX[bnd] - susyz * vY[bnd]
                              - Jzh[bnd] * dt * th * FourPI) / suszz;
}

// =========================================================================
//  Perfect conductor source kernels (zero tangential components)
// =========================================================================

__global__ void k_perfectConductorLeftSX(
    double* __restrict__ vX, double* __restrict__ vY, double* __restrict__ vZ,
    int nyn, int nzn)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int k = blockIdx.y * blockDim.y + threadIdx.y + 1;
    if (j > nyn - 2 || k > nzn - 2) return;
    int idx = 1 * nyn * nzn + j * nzn + k;
    vY[idx] = 0.0;
    vZ[idx] = 0.0;
}

__global__ void k_perfectConductorLeftSY(
    double* __restrict__ vX, double* __restrict__ vY, double* __restrict__ vZ,
    int nxn, int nyn, int nzn)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int k = blockIdx.y * blockDim.y + threadIdx.y + 1;
    if (i > nxn - 2 || k > nzn - 2) return;
    int idx = i * nyn * nzn + 1 * nzn + k;
    vX[idx] = 0.0;
    vZ[idx] = 0.0;
}

__global__ void k_perfectConductorLeftSZ(
    double* __restrict__ vX, double* __restrict__ vY, double* __restrict__ vZ,
    int nxn, int nyn, int nzn)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int j = blockIdx.y * blockDim.y + threadIdx.y + 1;
    if (i > nxn - 2 || j > nyn - 2) return;
    int idx = i * nyn * nzn + j * nzn + 1;
    vX[idx] = 0.0;
    vY[idx] = 0.0;
}

__global__ void k_perfectConductorRightSX(
    double* __restrict__ vX, double* __restrict__ vY, double* __restrict__ vZ,
    int nxn, int nyn, int nzn)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int k = blockIdx.y * blockDim.y + threadIdx.y + 1;
    if (j > nyn - 2 || k > nzn - 2) return;
    int idx = (nxn - 2) * nyn * nzn + j * nzn + k;
    vY[idx] = 0.0;
    vZ[idx] = 0.0;
}

__global__ void k_perfectConductorRightSY(
    double* __restrict__ vX, double* __restrict__ vY, double* __restrict__ vZ,
    int nxn, int nyn, int nzn)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int k = blockIdx.y * blockDim.y + threadIdx.y + 1;
    if (i > nxn - 2 || k > nzn - 2) return;
    int idx = i * nyn * nzn + (nyn - 2) * nzn + k;
    vX[idx] = 0.0;
    vZ[idx] = 0.0;
}

__global__ void k_perfectConductorRightSZ(
    double* __restrict__ vX, double* __restrict__ vY, double* __restrict__ vZ,
    int nxn, int nyn, int nzn)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int j = blockIdx.y * blockDim.y + threadIdx.y + 1;
    if (i > nxn - 2 || j > nyn - 2) return;
    int idx = i * nyn * nzn + j * nzn + (nzn - 2);
    vX[idx] = 0.0;
    vY[idx] = 0.0;
}

// =========================================================================
//  Host wrappers
// =========================================================================

static inline dim3 faceGrid2D(int d1, int d2)
{
    return dim3(((d1 - 2) + 15) / 16,
                ((d2 - 2) + 15) / 16);
}
static const dim3 faceBlock2D(16, 16);

void gpuPerfectConductorLeft(
    double* imageX, double* imageY, double* imageZ,
    const double* vectX, const double* vectY, const double* vectZ,
    const double* Ex, const double* Ey, const double* Ez,
    const double* Jxh, const double* Jyh, const double* Jzh,
    const double* Bxn, const double* Byn, const double* Bzn,
    const double* Bx_ext, const double* By_ext, const double* Bz_ext,
    const double* rhons,
    const double* d_qom, int ns,
    double dt, double c, double th, double FourPI, double delt,
    int nxn, int nyn, int nzn,
    int dir,
    cudaStream_t stream)
{
    switch (dir) {
    case 0: {
        dim3 grid = faceGrid2D(nyn, nzn);
        k_perfectConductorLeftX<<<grid, faceBlock2D, 0, stream>>>(
            imageX, imageY, imageZ, vectX, vectY, vectZ,
            Ex, Jxh, Bxn, Byn, Bzn, Bx_ext, By_ext, Bz_ext,
            rhons, d_qom, ns, dt, c, th, FourPI, delt, nxn, nyn, nzn);
    } break;
    case 1: {
        dim3 grid = faceGrid2D(nxn, nzn);
        k_perfectConductorLeftY<<<grid, faceBlock2D, 0, stream>>>(
            imageX, imageY, imageZ, vectX, vectY, vectZ,
            Ey, Jyh, Bxn, Byn, Bzn, Bx_ext, By_ext, Bz_ext,
            rhons, d_qom, ns, dt, c, th, FourPI, delt, nxn, nyn, nzn);
    } break;
    case 2: {
        dim3 grid = faceGrid2D(nxn, nyn);
        k_perfectConductorLeftZ<<<grid, faceBlock2D, 0, stream>>>(
            imageX, imageY, imageZ, vectX, vectY, vectZ,
            Ez, Jzh, Bxn, Byn, Bzn, Bx_ext, By_ext, Bz_ext,
            rhons, d_qom, ns, dt, c, th, FourPI, delt, nxn, nyn, nzn);
    } break;
    }
    cudaErrChk(cudaGetLastError());
}

void gpuPerfectConductorRight(
    double* imageX, double* imageY, double* imageZ,
    const double* vectX, const double* vectY, const double* vectZ,
    const double* Ex, const double* Ey, const double* Ez,
    const double* Jxh, const double* Jyh, const double* Jzh,
    const double* Bxn, const double* Byn, const double* Bzn,
    const double* Bx_ext, const double* By_ext, const double* Bz_ext,
    const double* rhons,
    const double* d_qom, int ns,
    double dt, double c, double th, double FourPI, double delt,
    int nxn, int nyn, int nzn,
    int dir,
    cudaStream_t stream)
{
    switch (dir) {
    case 0: {
        dim3 grid = faceGrid2D(nyn, nzn);
        k_perfectConductorRightX<<<grid, faceBlock2D, 0, stream>>>(
            imageX, imageY, imageZ, vectX, vectY, vectZ,
            Ex, Jxh, Bxn, Byn, Bzn, Bx_ext, By_ext, Bz_ext,
            rhons, d_qom, ns, dt, c, th, FourPI, delt, nxn, nyn, nzn);
    } break;
    case 1: {
        dim3 grid = faceGrid2D(nxn, nzn);
        k_perfectConductorRightY<<<grid, faceBlock2D, 0, stream>>>(
            imageX, imageY, imageZ, vectX, vectY, vectZ,
            Ey, Jyh, Bxn, Byn, Bzn, Bx_ext, By_ext, Bz_ext,
            rhons, d_qom, ns, dt, c, th, FourPI, delt, nxn, nyn, nzn);
    } break;
    case 2: {
        dim3 grid = faceGrid2D(nxn, nyn);
        k_perfectConductorRightZ<<<grid, faceBlock2D, 0, stream>>>(
            imageX, imageY, imageZ, vectX, vectY, vectZ,
            Ez, Jzh, Bxn, Byn, Bzn, Bx_ext, By_ext, Bz_ext,
            rhons, d_qom, ns, dt, c, th, FourPI, delt, nxn, nyn, nzn);
    } break;
    }
    cudaErrChk(cudaGetLastError());
}

void gpuPerfectConductorLeftS(
    double* vectorX, double* vectorY, double* vectorZ,
    int nxn, int nyn, int nzn,
    int dir,
    cudaStream_t stream)
{
    switch (dir) {
    case 0: {
        dim3 grid = faceGrid2D(nyn, nzn);
        k_perfectConductorLeftSX<<<grid, faceBlock2D, 0, stream>>>(vectorX, vectorY, vectorZ, nyn, nzn);
    } break;
    case 1: {
        dim3 grid = faceGrid2D(nxn, nzn);
        k_perfectConductorLeftSY<<<grid, faceBlock2D, 0, stream>>>(vectorX, vectorY, vectorZ, nxn, nyn, nzn);
    } break;
    case 2: {
        dim3 grid = faceGrid2D(nxn, nyn);
        k_perfectConductorLeftSZ<<<grid, faceBlock2D, 0, stream>>>(vectorX, vectorY, vectorZ, nxn, nyn, nzn);
    } break;
    }
    cudaErrChk(cudaGetLastError());
}

void gpuPerfectConductorRightS(
    double* vectorX, double* vectorY, double* vectorZ,
    int nxn, int nyn, int nzn,
    int dir,
    cudaStream_t stream)
{
    switch (dir) {
    case 0: {
        dim3 grid = faceGrid2D(nyn, nzn);
        k_perfectConductorRightSX<<<grid, faceBlock2D, 0, stream>>>(vectorX, vectorY, vectorZ, nxn, nyn, nzn);
    } break;
    case 1: {
        dim3 grid = faceGrid2D(nxn, nzn);
        k_perfectConductorRightSY<<<grid, faceBlock2D, 0, stream>>>(vectorX, vectorY, vectorZ, nxn, nyn, nzn);
    } break;
    case 2: {
        dim3 grid = faceGrid2D(nxn, nyn);
        k_perfectConductorRightSZ<<<grid, faceBlock2D, 0, stream>>>(vectorX, vectorY, vectorZ, nxn, nyn, nzn);
    } break;
    }
    cudaErrChk(cudaGetLastError());
}

#undef IDX3
#undef MAX_SPECIES

// =========================================================================
//  adjustNonPeriodicDensities: double boundary-face values on GPU
// =========================================================================

// Kernel: double values on X-face (i = fixedI)
__global__ void k_doubleFaceX(double* __restrict__ arr,
                               int fixedI, int nyn, int nzn)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int k = blockIdx.y * blockDim.y + threadIdx.y + 1;
    if (j >= nyn - 1 || k >= nzn - 1) return;
    arr[(fixedI * nyn + j) * nzn + k] *= 2.0;
}

// Kernel: double values on Y-face (j = fixedJ)
__global__ void k_doubleFaceY(double* __restrict__ arr,
                               int fixedJ, int nxn, int nyn, int nzn)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int k = blockIdx.y * blockDim.y + threadIdx.y + 1;
    if (i >= nxn - 1 || k >= nzn - 1) return;
    arr[(i * nyn + fixedJ) * nzn + k] *= 2.0;
}

// Kernel: double values on Z-face (k = fixedK)
__global__ void k_doubleFaceZ(double* __restrict__ arr,
                               int fixedK, int nxn, int nyn, int nzn)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int j = blockIdx.y * blockDim.y + threadIdx.y + 1;
    if (i >= nxn - 1 || j >= nyn - 1) return;
    arr[(i * nyn + j) * nzn + fixedK] *= 2.0;
}

// Batched versions: blockIdx.z selects the field from the pointer array
__global__ void k_batchDoubleFaceX(double* const* __restrict__ ptrs,
                                    int fixedI, int nyn, int nzn)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int k = blockIdx.y * blockDim.y + threadIdx.y + 1;
    if (j >= nyn - 1 || k >= nzn - 1) return;
    ptrs[blockIdx.z][(fixedI * nyn + j) * nzn + k] *= 2.0;
}

__global__ void k_batchDoubleFaceY(double* const* __restrict__ ptrs,
                                    int fixedJ, int nxn, int nyn, int nzn)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int k = blockIdx.y * blockDim.y + threadIdx.y + 1;
    if (i >= nxn - 1 || k >= nzn - 1) return;
    ptrs[blockIdx.z][(i * nyn + fixedJ) * nzn + k] *= 2.0;
}

__global__ void k_batchDoubleFaceZ(double* const* __restrict__ ptrs,
                                    int fixedK, int nxn, int nyn, int nzn)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int j = blockIdx.y * blockDim.y + threadIdx.y + 1;
    if (i >= nxn - 1 || j >= nyn - 1) return;
    ptrs[blockIdx.z][(i * nyn + j) * nzn + fixedK] *= 2.0;
}

void gpuAdjustNonPeriodicDensities(
    int nptrs,
    double* const* d_devPtrs,
    int nxn, int nyn, int nzn,
    bool xLeftNull, bool xRightNull,
    bool yLeftNull, bool yRightNull,
    bool zLeftNull, bool zRightNull,
    cudaStream_t stream)
{
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
        k_batchDoubleFaceX<<<grid, block, 0, stream>>>(d_devPtrs, nxn - 2, nyn, nzn);
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
        k_batchDoubleFaceY<<<grid, block, 0, stream>>>(d_devPtrs, nyn - 2, nxn, nyn, nzn);
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
        k_batchDoubleFaceZ<<<grid, block, 0, stream>>>(d_devPtrs, nzn - 2, nxn, nyn, nzn);
    }

    cudaErrChk(cudaGetLastError());
}

// =========================================================================
//  Open Boundary kernels
// =========================================================================

/** Zero 3 arrays on a single face plane (interior [1..n-2]). */
__global__ void k_zeroFacePlane3(double* X, double* Y, double* Z,
                                 int dir, int faceIdx,
                                 int nx, int ny, int nz)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int d1, d2;
    if      (dir == 0) { d1 = ny - 2; d2 = nz - 2; }
    else if (dir == 1) { d1 = nx - 2; d2 = nz - 2; }
    else               { d1 = nx - 2; d2 = ny - 2; }
    if (tid >= d1 * d2) return;
    int a = tid / d2 + 1;
    int b = tid % d2 + 1;
    int i, j, k;
    if      (dir == 0) { i = faceIdx; j = a; k = b; }
    else if (dir == 1) { i = a; j = faceIdx; k = b; }
    else               { i = a; j = b; k = faceIdx; }
    int idx = (i * ny + j) * nz + k;
    X[idx] = 0.0; Y[idx] = 0.0; Z[idx] = 0.0;
}

void gpuOpenBCZeroFace3(double* X, double* Y, double* Z,
                        int dir, int faceIdx,
                        int nx, int ny, int nz, cudaStream_t stream)
{
    int d1, d2;
    if      (dir == 0) { d1 = ny - 2; d2 = nz - 2; }
    else if (dir == 1) { d1 = nx - 2; d2 = nz - 2; }
    else               { d1 = nx - 2; d2 = ny - 2; }
    int total = d1 * d2;
    if (total <= 0) return;
    int blk = 256;
    k_zeroFacePlane3<<<(total + blk - 1) / blk, blk, 0, stream>>>(
        X, Y, Z, dir, faceIdx, nx, ny, nz);
    cudaErrChk(cudaGetLastError());
}

/** Set image = vect - injE on a single face plane (interior [1..n-2]). */
__global__ void k_imageDiffFacePlane3(double* imX, double* imY, double* imZ,
                                      const double* vX, const double* vY, const double* vZ,
                                      double injE0, double injE1, double injE2,
                                      int dir, int faceIdx,
                                      int nx, int ny, int nz)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int d1, d2;
    if      (dir == 0) { d1 = ny - 2; d2 = nz - 2; }
    else if (dir == 1) { d1 = nx - 2; d2 = nz - 2; }
    else               { d1 = nx - 2; d2 = ny - 2; }
    if (tid >= d1 * d2) return;
    int a = tid / d2 + 1;
    int b = tid % d2 + 1;
    int i, j, k;
    if      (dir == 0) { i = faceIdx; j = a; k = b; }
    else if (dir == 1) { i = a; j = faceIdx; k = b; }
    else               { i = a; j = b; k = faceIdx; }
    int idx = (i * ny + j) * nz + k;
    imX[idx] = vX[idx] - injE0;
    imY[idx] = vY[idx] - injE1;
    imZ[idx] = vZ[idx] - injE2;
}

void gpuOpenBCImageDiffFace3(double* imX, double* imY, double* imZ,
                             const double* vX, const double* vY, const double* vZ,
                             double injE0, double injE1, double injE2,
                             int dir, int faceIdx,
                             int nx, int ny, int nz, cudaStream_t stream)
{
    int d1, d2;
    if      (dir == 0) { d1 = ny - 2; d2 = nz - 2; }
    else if (dir == 1) { d1 = nx - 2; d2 = nz - 2; }
    else               { d1 = nx - 2; d2 = ny - 2; }
    int total = d1 * d2;
    if (total <= 0) return;
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
__global__ void k_salBlend3(double* X, double* Y, double* Z,
                            double tgtX, double tgtY, double tgtZ,
                            int dir, int layerStart, int layerEnd,
                            double invNLayers, int ascending,
                            int nx, int ny, int nz)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int nLayers = layerEnd - layerStart + 1;
    int d1, d2;
    if      (dir == 0) { d1 = ny; d2 = nz; }
    else if (dir == 1) { d1 = nx; d2 = nz; }
    else               { d1 = nx; d2 = ny; }
    int total = nLayers * d1 * d2;
    if (tid >= total) return;
    int layerOff = tid / (d1 * d2);
    int rem = tid % (d1 * d2);
    int a = rem / d2;
    int b = rem % d2;
    int layer = layerStart + layerOff;
    double sal = ascending ? (double)(layer - layerStart) * invNLayers
                           : (double)(layerEnd - layer) * invNLayers;
    int i, j, k;
    if      (dir == 0) { i = layer; j = a; k = b; }
    else if (dir == 1) { i = a; j = layer; k = b; }
    else               { i = a; j = b; k = layer; }
    int idx = (i * ny + j) * nz + k;
    X[idx] = X[idx] * sal + tgtX * (1.0 - sal);
    Y[idx] = Y[idx] * sal + tgtY * (1.0 - sal);
    Z[idx] = Z[idx] * sal + tgtZ * (1.0 - sal);
}

void gpuSALBlendLayers3(double* X, double* Y, double* Z,
                        double tgtX, double tgtY, double tgtZ,
                        int dir, int layerStart, int layerEnd,
                        double invNLayers, int ascending,
                        int nx, int ny, int nz, cudaStream_t stream)
{
    int nLayers = layerEnd - layerStart + 1;
    int d1, d2;
    if      (dir == 0) { d1 = ny; d2 = nz; }
    else if (dir == 1) { d1 = nx; d2 = nz; }
    else               { d1 = nx; d2 = ny; }
    int total = nLayers * d1 * d2;
    if (total <= 0) return;
    int blk = 256;
    k_salBlend3<<<(total + blk - 1) / blk, blk, 0, stream>>>(
        X, Y, Z, tgtX, tgtY, tgtZ, dir, layerStart, layerEnd, invNLayers, ascending, nx, ny, nz);
    cudaErrChk(cudaGetLastError());
}

/** Set 3 arrays to constant values on boundary layers. */
__global__ void k_setConstLayers3(double* X, double* Y, double* Z,
                                   double cx, double cy, double cz,
                                   int dir, int layerStart, int layerEnd,
                                   int nx, int ny, int nz)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int nLayers = layerEnd - layerStart + 1;
    int d1, d2;
    if      (dir == 0) { d1 = ny; d2 = nz; }
    else if (dir == 1) { d1 = nx; d2 = nz; }
    else               { d1 = nx; d2 = ny; }
    int total = nLayers * d1 * d2;
    if (tid >= total) return;
    int layerOff = tid / (d1 * d2);
    int rem = tid % (d1 * d2);
    int a = rem / d2;
    int b = rem % d2;
    int layer = layerStart + layerOff;
    int i, j, k;
    if      (dir == 0) { i = layer; j = a; k = b; }
    else if (dir == 1) { i = a; j = layer; k = b; }
    else               { i = a; j = b; k = layer; }
    int idx = (i * ny + j) * nz + k;
    X[idx] = cx; Y[idx] = cy; Z[idx] = cz;
}

void gpuSetConstLayers3(double* X, double* Y, double* Z,
                        double cx, double cy, double cz,
                        int dir, int layerStart, int layerEnd,
                        int nx, int ny, int nz, cudaStream_t stream)
{
    int nLayers = layerEnd - layerStart + 1;
    int d1, d2;
    if      (dir == 0) { d1 = ny; d2 = nz; }
    else if (dir == 1) { d1 = nx; d2 = nz; }
    else               { d1 = nx; d2 = ny; }
    int total = nLayers * d1 * d2;
    if (total <= 0) return;
    int blk = 256;
    k_setConstLayers3<<<(total + blk - 1) / blk, blk, 0, stream>>>(
        X, Y, Z, cx, cy, cz, dir, layerStart, layerEnd, nx, ny, nz);
    cudaErrChk(cudaGetLastError());
}

/** Copy from a constant reference plane to boundary layers. */
__global__ void k_extrapolateLayers3(double* X, double* Y, double* Z,
                                      int dir, int layerStart, int layerEnd,
                                      int refLayer,
                                      int nx, int ny, int nz)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int nLayers = layerEnd - layerStart + 1;
    int d1, d2;
    if      (dir == 0) { d1 = ny; d2 = nz; }
    else if (dir == 1) { d1 = nx; d2 = nz; }
    else               { d1 = nx; d2 = ny; }
    int total = nLayers * d1 * d2;
    if (tid >= total) return;
    int layerOff = tid / (d1 * d2);
    int rem = tid % (d1 * d2);
    int a = rem / d2;
    int b = rem % d2;
    int layer = layerStart + layerOff;
    int i, j, k, ri, rj, rk;
    if      (dir == 0) { i = layer; j = a; k = b; ri = refLayer; rj = a; rk = b; }
    else if (dir == 1) { i = a; j = layer; k = b; ri = a; rj = refLayer; rk = b; }
    else               { i = a; j = b; k = layer; ri = a; rj = b; rk = refLayer; }
    int idx  = (i  * ny + j)  * nz + k;
    int ridx = (ri * ny + rj) * nz + rk;
    X[idx] = X[ridx];
    Y[idx] = Y[ridx];
    Z[idx] = Z[ridx];
}

void gpuExtrapolateLayers3(double* X, double* Y, double* Z,
                           int dir, int layerStart, int layerEnd,
                           int refLayer,
                           int nx, int ny, int nz, cudaStream_t stream)
{
    int nLayers = layerEnd - layerStart + 1;
    int d1, d2;
    if      (dir == 0) { d1 = ny; d2 = nz; }
    else if (dir == 1) { d1 = nx; d2 = nz; }
    else               { d1 = nx; d2 = ny; }
    int total = nLayers * d1 * d2;
    if (total <= 0) return;
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
__global__ void k_fixBcGEM(double* Bxc, double* Byc, double* Bzc,
                           double B0x, double B0y, double B0z,
                           double yStart, double dy, double LyH, double delta,
                           int side, int nxc, int nyc, int nzc)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= nxc * nzc) return;
    int i = tid / nzc;
    int k = tid % nzc;
    int j0 = (side == 0) ? 0 : nyc - 1;
    int j1 = (side == 0) ? 1 : nyc - 2;
    int j2 = (side == 0) ? 2 : nyc - 3;
    double yc0 = yStart + ((double)j0 - 0.5) * dy;
    double bx_val = B0x * tanh((yc0 - LyH) / delta);
    int idx0 = (i * nyc + j0) * nzc + k;
    int idx1 = (i * nyc + j1) * nzc + k;
    int idx2 = (i * nyc + j2) * nzc + k;
    Bxc[idx0] = bx_val; Bxc[idx1] = bx_val; Bxc[idx2] = bx_val;
    Byc[idx0] = B0y;
    Bzc[idx0] = B0z; Bzc[idx1] = B0z; Bzc[idx2] = B0z;
}

void gpuFixBcGEMKernel(double* Bxc, double* Byc, double* Bzc,
                       double B0x, double B0y, double B0z,
                       double yStart, double dy, double LyH, double delta,
                       int side, int nxc, int nyc, int nzc, cudaStream_t stream)
{
    int total = nxc * nzc;
    if (total <= 0) return;
    int blk = 256;
    k_fixBcGEM<<<(total + blk - 1) / blk, blk, 0, stream>>>(
        Bxc, Byc, Bzc, B0x, B0y, B0z, yStart, dy, LyH, delta, side, nxc, nyc, nzc);
    cudaErrChk(cudaGetLastError());
}

/** Fix node-based B for GEM on 3 Y-boundary layers.
 *  Uses CENTER Y coordinate for the tanh profile (matches CPU).
 *  side: 0=left (j=0,1,2), 1=right (j=nyn-1, nyn-2, nyn-3). */
__global__ void k_fixBnGEM(double* Bxn, double* Byn, double* Bzn,
                           double B0x, double B0y, double B0z,
                           double yStart, double dy, double LyH, double delta,
                           int side, int nxn, int nyn, int nzn, int nyc)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= nxn * nzn) return;
    int i = tid / nzn;
    int k = tid % nzn;
    // Center Y index for tanh profile: 0 for left, nyc-1 for right
    int jc = (side == 0) ? 0 : nyc - 1;
    double yc = yStart + ((double)jc - 0.5) * dy;
    double bx_val = B0x * tanh((yc - LyH) / delta);
    int j0 = (side == 0) ? 0 : nyn - 1;
    int j1 = (side == 0) ? 1 : nyn - 2;
    int j2 = (side == 0) ? 2 : nyn - 3;
    int idx0 = (i * nyn + j0) * nzn + k;
    int idx1 = (i * nyn + j1) * nzn + k;
    int idx2 = (i * nyn + j2) * nzn + k;
    Bxn[idx0] = bx_val; Bxn[idx1] = bx_val; Bxn[idx2] = bx_val;
    Byn[idx0] = B0y;
    Bzn[idx0] = B0z; Bzn[idx1] = B0z; Bzn[idx2] = B0z;
}

void gpuFixBnGEMKernel(double* Bxn, double* Byn, double* Bzn,
                       double B0x, double B0y, double B0z,
                       double yStart, double dy, double LyH, double delta,
                       int side, int nxn, int nyn, int nzn, int nyc,
                       cudaStream_t stream)
{
    int total = nxn * nzn;
    if (total <= 0) return;
    int blk = 256;
    k_fixBnGEM<<<(total + blk - 1) / blk, blk, 0, stream>>>(
        Bxn, Byn, Bzn, B0x, B0y, B0z, yStart, dy, LyH, delta, side, nxn, nyn, nzn, nyc);
    cudaErrChk(cudaGetLastError());
}

/** Fix center-based B for ForceFree on Y-boundary layers.
 *  Bx = B0x*tanh, Bz = B0z/cosh profile on up to 3 layers. */
__global__ void k_fixBforcefree(double* Bxc, double* Byc, double* Bzc,
                                double B0x, double B0y, double B0z,
                                double yStart, double dy, double LyH, double delta,
                                int side, int nxc, int nyc, int nzc)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= nxc * nzc) return;
    int i = tid / nzc;
    int k = tid % nzc;
    int j0, j1, j2;
    if (side == 0) { j0 = 0; j1 = 1; j2 = 2; }
    else           { j0 = nyc - 1; j1 = nyc - 2; j2 = nyc - 3; }
    double yc0 = yStart + ((double)j0 - 0.5) * dy;
    double yc1 = yStart + ((double)j1 - 0.5) * dy;
    double yc2 = yStart + ((double)j2 - 0.5) * dy;
    double arg0 = (yc0 - LyH) / delta;
    double arg1 = (yc1 - LyH) / delta;
    double arg2 = (yc2 - LyH) / delta;
    int idx0 = (i * nyc + j0) * nzc + k;
    int idx1 = (i * nyc + j1) * nzc + k;
    int idx2 = (i * nyc + j2) * nzc + k;
    Bxc[idx0] = B0x * tanh(arg0);
    Byc[idx0] = B0y;
    Bzc[idx0] = B0z / cosh(arg0);
    Bzc[idx1] = B0z / cosh(arg1);
    Bzc[idx2] = B0z / cosh(arg2);
}

void gpuFixBforcefreeKernel(double* Bxc, double* Byc, double* Bzc,
                            double B0x, double B0y, double B0z,
                            double yStart, double dy, double LyH, double delta,
                            int side, int nxc, int nyc, int nzc,
                            cudaStream_t stream)
{
    int total = nxc * nzc;
    if (total <= 0) return;
    int blk = 256;
    k_fixBforcefree<<<(total + blk - 1) / blk, blk, 0, stream>>>(
        Bxc, Byc, Bzc, B0x, B0y, B0z, yStart, dy, LyH, delta, side, nxc, nyc, nzc);
    cudaErrChk(cudaGetLastError());
}

// =========================================================================
//  ConstantChargePlanet kernels
// =========================================================================

/** Set rhons = val inside sphere of radius R centred at (xc,yc,zc). */
__global__ void k_constantChargePlanet(double* rhons, double val,
                                       double R2,
                                       double xc, double yc, double zc,
                                       double xStart, double yStart, double zStart,
                                       double dx, double dy, double dz,
                                       int nxn, int nyn, int nzn)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int ni = nxn - 1, nj = nyn - 1, nk = nzn - 1;
    if (tid >= ni * nj * nk) return;
    int i = tid / (nj * nk) + 1;
    int rem = tid % (nj * nk);
    int j = rem / nk + 1;
    int k = rem % nk + 1;
    double xd = xStart + (double)(i - 1) * dx - xc;
    double yd = yStart + (double)(j - 1) * dy - yc;
    double zd = zStart + (double)(k - 1) * dz - zc;
    if (xd * xd + yd * yd + zd * zd <= R2)
        rhons[(i * nyn + j) * nzn + k] = val;
}

void gpuConstantChargePlanetKernel(double* rhons, double val,
                                   double R, double xc, double yc, double zc,
                                   double xStart, double yStart, double zStart,
                                   double dx, double dy, double dz,
                                   int nxn, int nyn, int nzn,
                                   cudaStream_t stream)
{
    int total = (nxn - 1) * (nyn - 1) * (nzn - 1);
    if (total <= 0) return;
    int blk = 256;
    k_constantChargePlanet<<<(total + blk - 1) / blk, blk, 0, stream>>>(
        rhons, val, R * R, xc, yc, zc, xStart, yStart, zStart, dx, dy, dz, nxn, nyn, nzn);
    cudaErrChk(cudaGetLastError());
}

/** 2D version: set rhons inside circle in XZ plane (nyn==4). */
__global__ void k_constantChargePlanet2D(double* rhons, double val,
                                          double R2,
                                          double xc, double zc,
                                          double xStart, double zStart,
                                          double dx, double dz,
                                          int nxn, int nyn, int nzn)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int ni = nxn - 1, nk = nzn - 1;
    if (tid >= ni * nk) return;
    int i = tid / nk + 1;
    int k = tid % nk + 1;
    double xd = xStart + (double)(i - 1) * dx - xc;
    double zd = zStart + (double)(k - 1) * dz - zc;
    if (xd * xd + zd * zd <= R2) {
        rhons[(i * nyn + 1) * nzn + k] = val;
        rhons[(i * nyn + 2) * nzn + k] = val;
    }
}

void gpuConstantChargePlanet2DKernel(double* rhons, double val,
                                     double R, double xc, double zc,
                                     double xStart, double zStart,
                                     double dx, double dz,
                                     int nxn, int nyn, int nzn,
                                     cudaStream_t stream)
{
    int total = (nxn - 1) * (nzn - 1);
    if (total <= 0) return;
    int blk = 256;
    k_constantChargePlanet2D<<<(total + blk - 1) / blk, blk, 0, stream>>>(
        rhons, val, R * R, xc, zc, xStart, zStart, dx, dz, nxn, nyn, nzn);
    cudaErrChk(cudaGetLastError());
}

// =========================================================================
//  Subtract grad(PSI) from B on boundary layers (divB cleaning)
// =========================================================================

/** B -= gradPSI on boundary layers [layerStart..layerEnd] in direction dir. */
__global__ void k_subLayers3(double* BxN, double* ByN, double* BzN,
                             const double* gX, const double* gY, const double* gZ,
                             int dir, int layerStart, int layerEnd,
                             int nxn, int nyn, int nzn)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int nLayers = layerEnd - layerStart + 1;
    int d1, d2;
    if      (dir == 0) { d1 = nyn; d2 = nzn; }
    else if (dir == 1) { d1 = nxn; d2 = nzn; }
    else               { d1 = nxn; d2 = nyn; }
    int total = nLayers * d1 * d2;
    if (tid >= total) return;
    int layerOff = tid / (d1 * d2);
    int rem = tid % (d1 * d2);
    int a = rem / d2;
    int b = rem % d2;
    int layer = layerStart + layerOff;
    int i, j, k;
    if      (dir == 0) { i = layer; j = a; k = b; }
    else if (dir == 1) { i = a; j = layer; k = b; }
    else               { i = a; j = b; k = layer; }
    int idx = (i * nyn + j) * nzn + k;
    BxN[idx] -= gX[idx];
    ByN[idx] -= gY[idx];
    BzN[idx] -= gZ[idx];
}

void gpuSubBoundaryLayers3(double* BxN, double* ByN, double* BzN,
                           const double* gX, const double* gY, const double* gZ,
                           int dir, int layerStart, int layerEnd,
                           int nxn, int nyn, int nzn, cudaStream_t stream)
{
    int nLayers = layerEnd - layerStart + 1;
    int d1, d2;
    if      (dir == 0) { d1 = nyn; d2 = nzn; }
    else if (dir == 1) { d1 = nxn; d2 = nzn; }
    else               { d1 = nxn; d2 = nyn; }
    int total = nLayers * d1 * d2;
    if (total <= 0) return;
    int blk = 256;
    k_subLayers3<<<(total + blk - 1) / blk, blk, 0, stream>>>(
        BxN, ByN, BzN, gX, gY, gZ, dir, layerStart, layerEnd, nxn, nyn, nzn);
    cudaErrChk(cudaGetLastError());
}

// =========================================================================
//  Center-to-center Laplacian (for Poisson solver)
// =========================================================================

/** 7-point center Laplacian: lapC = d²f/dx² + d²f/dy² + d²f/dz². */
__global__ void k_lapC2C(double* lapC, const double* fC,
                         int nxc, int nyc, int nzc,
                         double invdx2, double invdy2, double invdz2)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int ni = nxc - 2, nj = nyc - 2, nk = nzc - 2;
    if (tid >= ni * nj * nk) return;
    int i = tid / (nj * nk) + 1;
    int rem = tid % (nj * nk);
    int j = rem / nk + 1;
    int k = rem % nk + 1;
    int idx = (i * nyc + j) * nzc + k;
    int sX = nyc * nzc;  // stride in X direction
    int sY = nzc;         // stride in Y direction
    lapC[idx] = (fC[idx - sX] - 2.0 * fC[idx] + fC[idx + sX]) * invdx2
              + (fC[idx - sY] - 2.0 * fC[idx] + fC[idx + sY]) * invdy2
              + (fC[idx - 1]  - 2.0 * fC[idx] + fC[idx + 1])  * invdz2;
}

void gpuLapC2CKernel(double* lapC, const double* fC,
                     int nxc, int nyc, int nzc,
                     double invdx2, double invdy2, double invdz2,
                     cudaStream_t stream)
{
    int total = (nxc - 2) * (nyc - 2) * (nzc - 2);
    if (total <= 0) return;
    int blk = 256;
    k_lapC2C<<<(total + blk - 1) / blk, blk, 0, stream>>>(
        lapC, fC, nxc, nyc, nzc, invdx2, invdy2, invdz2);
    cudaErrChk(cudaGetLastError());
}

#endif // GPU_SOLVER
