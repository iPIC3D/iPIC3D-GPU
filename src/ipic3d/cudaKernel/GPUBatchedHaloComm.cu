/**
 * @file GPUBatchedHaloComm.cu
 * @brief Batched GPU halo exchange with explicit pack/unpack.
 *
 * Replaces MPI derived datatypes on device memory with explicit CUDA
 * pack/unpack kernels followed by contiguous MPI transfers.  Multiple
 * field arrays are batched into single MPI messages to minimise the
 * number of MPI calls per exchange.
 *
 * Compile with -DGPU_SOLVER to activate.
 */

#ifdef GPU_SOLVER

#include "GPUHaloComm.cuh"   // single-array self-copy kernels, BC kernels
#include "EMfields3D.h"
#include "VCtopology3D.h"
#include <cassert>
#include <cstring>

// =========================================================================
//  Helper: linear index into a flat (nx, ny, nz) row-major array
// =========================================================================
__device__ __forceinline__
int bidx3(int i, int j, int k, int ny, int nz)
{
    return (i * ny + j) * nz + k;
}

// =========================================================================
//  BATCHED PACK / UNPACK KERNELS
//
//  k_batchPack2D / k_batchUnpack2D handle any rectangular 2D region
//  described by (base, outerStride, innerStride, outerCount, innerCount).
//  This covers faces, edges (innerCount=1), and single elements.
// =========================================================================

__global__ void k_batchPack2D(
    double* __restrict__ buf,
    double* const* __restrict__ fields,
    int nFields,
    int base,
    int outerStride,
    int innerStride,
    int outerCount,
    int innerCount)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int elemsPerField = outerCount * innerCount;
    int total = elemsPerField * nFields;
    if (tid >= total) return;

    int f   = tid / elemsPerField;
    int rem = tid % elemsPerField;
    int oi  = rem / innerCount;
    int ii  = rem % innerCount;

    buf[tid] = fields[f][base + oi * outerStride + ii * innerStride];
}

__global__ void k_batchUnpack2D(
    const double* __restrict__ buf,
    double* const* __restrict__ fields,
    int nFields,
    int base,
    int outerStride,
    int innerStride,
    int outerCount,
    int innerCount)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int elemsPerField = outerCount * innerCount;
    int total = elemsPerField * nFields;
    if (tid >= total) return;

    int f   = tid / elemsPerField;
    int rem = tid % elemsPerField;
    int oi  = rem / innerCount;
    int ii  = rem % innerCount;

    fields[f][base + oi * outerStride + ii * innerStride] = buf[tid];
}

// =========================================================================
//  CORNER PACK / UNPACK  (4 scattered elements per field per direction)
// =========================================================================

__global__ void k_batchPackCorners4(
    double* __restrict__ buf,
    double* const* __restrict__ fields,
    int nFields,
    int base,       // ix * ny * nz
    int ny, int nz)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= nFields * 4) return;

    int f = tid / 4;
    int c = tid % 4;

    // 4 corners of the YZ face: (iy,iz) ∈ {(0,0),(0,nz-1),(ny-1,0),(ny-1,nz-1)}
    int offsets[4] = {0, nz - 1, (ny - 1) * nz, ny * nz - 1};
    buf[tid] = fields[f][base + offsets[c]];
}

__global__ void k_batchUnpackCorners4(
    const double* __restrict__ buf,
    double* const* __restrict__ fields,
    int nFields,
    int base,
    int ny, int nz)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= nFields * 4) return;

    int f = tid / 4;
    int c = tid % 4;

    int offsets[4] = {0, nz - 1, (ny - 1) * nz, ny * nz - 1};
    fields[f][base + offsets[c]] = buf[tid];
}

// =========================================================================
//  BATCHED ADDITIVE ACCUMULATION KERNELS  (for interp / moments)
//
//  After the halo exchange populates ghost cells, these kernels add ghost
//  values into the adjacent interior boundary for nFields arrays at once.
// =========================================================================

__global__ void k_batchAddFaceX(
    double* const* __restrict__ fields,
    int nFields, int nx, int ny, int nz,
    bool hasRight, bool hasLeft)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int faceSize = (ny - 2) * (nz - 2);
    int total = faceSize * nFields;
    if (tid >= total) return;

    int f   = tid / faceSize;
    int rem = tid % faceSize;
    int j   = rem / (nz - 2) + 1;
    int k   = rem % (nz - 2) + 1;

    if (hasRight)
        fields[f][bidx3(nx - 2, j, k, ny, nz)] += fields[f][bidx3(nx - 1, j, k, ny, nz)];
    if (hasLeft)
        fields[f][bidx3(1, j, k, ny, nz)] += fields[f][bidx3(0, j, k, ny, nz)];
}

__global__ void k_batchAddFaceY(
    double* const* __restrict__ fields,
    int nFields, int nx, int ny, int nz,
    bool hasRight, bool hasLeft)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int faceSize = (nx - 2) * (nz - 2);
    int total = faceSize * nFields;
    if (tid >= total) return;

    int f   = tid / faceSize;
    int rem = tid % faceSize;
    int i   = rem / (nz - 2) + 1;
    int k   = rem % (nz - 2) + 1;

    if (hasRight)
        fields[f][bidx3(i, ny - 2, k, ny, nz)] += fields[f][bidx3(i, ny - 1, k, ny, nz)];
    if (hasLeft)
        fields[f][bidx3(i, 1, k, ny, nz)] += fields[f][bidx3(i, 0, k, ny, nz)];
}

__global__ void k_batchAddFaceZ(
    double* const* __restrict__ fields,
    int nFields, int nx, int ny, int nz,
    bool hasRight, bool hasLeft)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int faceSize = (nx - 2) * (ny - 2);
    int total = faceSize * nFields;
    if (tid >= total) return;

    int f   = tid / faceSize;
    int rem = tid % faceSize;
    int i   = rem / (ny - 2) + 1;
    int j   = rem % (ny - 2) + 1;

    if (hasRight)
        fields[f][bidx3(i, j, nz - 2, ny, nz)] += fields[f][bidx3(i, j, nz - 1, ny, nz)];
    if (hasLeft)
        fields[f][bidx3(i, j, 1, ny, nz)] += fields[f][bidx3(i, j, 0, ny, nz)];
}

__global__ void k_batchAddEdgeZ(
    double* const* __restrict__ fields,
    int nFields, int nx, int ny, int nz,
    bool hasXR, bool hasXL, bool hasYR, bool hasYL)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int edgeLen = nz - 2;
    int total = edgeLen * nFields;
    if (tid >= total) return;

    int f = tid / edgeLen;
    int k = tid % edgeLen + 1;

    if (hasXR && hasYR)
        fields[f][bidx3(nx-2, ny-2, k, ny, nz)] += fields[f][bidx3(nx-1, ny-1, k, ny, nz)];
    if (hasXL && hasYL)
        fields[f][bidx3(1, 1, k, ny, nz)] += fields[f][bidx3(0, 0, k, ny, nz)];
    if (hasXR && hasYL)
        fields[f][bidx3(nx-2, 1, k, ny, nz)] += fields[f][bidx3(nx-1, 0, k, ny, nz)];
    if (hasXL && hasYR)
        fields[f][bidx3(1, ny-2, k, ny, nz)] += fields[f][bidx3(0, ny-1, k, ny, nz)];
}

__global__ void k_batchAddEdgeY(
    double* const* __restrict__ fields,
    int nFields, int nx, int ny, int nz,
    bool hasXR, bool hasXL, bool hasZR, bool hasZL)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int edgeLen = ny - 2;
    int total = edgeLen * nFields;
    if (tid >= total) return;

    int f = tid / edgeLen;
    int j = tid % edgeLen + 1;

    if (hasXR && hasZR)
        fields[f][bidx3(nx-2, j, nz-2, ny, nz)] += fields[f][bidx3(nx-1, j, nz-1, ny, nz)];
    if (hasXL && hasZL)
        fields[f][bidx3(1, j, 1, ny, nz)] += fields[f][bidx3(0, j, 0, ny, nz)];
    if (hasXL && hasZR)
        fields[f][bidx3(1, j, nz-2, ny, nz)] += fields[f][bidx3(0, j, nz-1, ny, nz)];
    if (hasXR && hasZL)
        fields[f][bidx3(nx-2, j, 1, ny, nz)] += fields[f][bidx3(nx-1, j, 0, ny, nz)];
}

__global__ void k_batchAddEdgeX(
    double* const* __restrict__ fields,
    int nFields, int nx, int ny, int nz,
    bool hasYR, bool hasYL, bool hasZR, bool hasZL)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int edgeLen = nx - 2;
    int total = edgeLen * nFields;
    if (tid >= total) return;

    int f = tid / edgeLen;
    int i = tid % edgeLen + 1;

    if (hasYR && hasZR)
        fields[f][bidx3(i, ny-2, nz-2, ny, nz)] += fields[f][bidx3(i, ny-1, nz-1, ny, nz)];
    if (hasYL && hasZL)
        fields[f][bidx3(i, 1, 1, ny, nz)] += fields[f][bidx3(i, 0, 0, ny, nz)];
    if (hasYL && hasZR)
        fields[f][bidx3(i, 1, nz-2, ny, nz)] += fields[f][bidx3(i, 0, nz-1, ny, nz)];
    if (hasYR && hasZL)
        fields[f][bidx3(i, ny-2, 1, ny, nz)] += fields[f][bidx3(i, ny-1, 0, ny, nz)];
}

__global__ void k_batchAddCorner(
    double* const* __restrict__ fields,
    int nFields, int nx, int ny, int nz,
    bool hasXR, bool hasXL, bool hasYR, bool hasYL,
    bool hasZR, bool hasZL)
{
    int f = blockIdx.x * blockDim.x + threadIdx.x;
    if (f >= nFields) return;

    if (hasXR && hasYR && hasZR)
        fields[f][bidx3(nx-2,ny-2,nz-2,ny,nz)] += fields[f][bidx3(nx-1,ny-1,nz-1,ny,nz)];
    if (hasXL && hasYR && hasZR)
        fields[f][bidx3(1,ny-2,nz-2,ny,nz)] += fields[f][bidx3(0,ny-1,nz-1,ny,nz)];
    if (hasXR && hasYL && hasZR)
        fields[f][bidx3(nx-2,1,nz-2,ny,nz)] += fields[f][bidx3(nx-1,0,nz-1,ny,nz)];
    if (hasXL && hasYL && hasZR)
        fields[f][bidx3(1,1,nz-2,ny,nz)] += fields[f][bidx3(0,0,nz-1,ny,nz)];
    if (hasXR && hasYR && hasZL)
        fields[f][bidx3(nx-2,ny-2,1,ny,nz)] += fields[f][bidx3(nx-1,ny-1,0,ny,nz)];
    if (hasXL && hasYR && hasZL)
        fields[f][bidx3(1,ny-2,1,ny,nz)] += fields[f][bidx3(0,ny-1,0,ny,nz)];
    if (hasXR && hasYL && hasZL)
        fields[f][bidx3(nx-2,1,1,ny,nz)] += fields[f][bidx3(nx-1,0,0,ny,nz)];
    if (hasXL && hasYL && hasZL)
        fields[f][bidx3(1,1,1,ny,nz)] += fields[f][bidx3(0,0,0,ny,nz)];
}

// =========================================================================
//  BATCHED HALO EXCHANGE
//
//  Replaces gpuNBDerivedHaloComm for multiple arrays at once.
//  Uses explicit CUDA pack/unpack into persistent contiguous GPU buffers
//  followed by MPI_Isend/Irecv of contiguous data.
// =========================================================================

static constexpr int BATCH_BLK = 256;

// Helper: launch k_batchPack2D with auto grid sizing
static inline void launchPack2D(
    double* buf, double* const* fields, int nFields,
    int base, int outerStride, int innerStride,
    int outerCount, int innerCount, cudaStream_t s)
{
    int total = outerCount * innerCount * nFields;
    if (total <= 0) return;
    k_batchPack2D<<<(total + BATCH_BLK - 1) / BATCH_BLK, BATCH_BLK, 0, s>>>(
        buf, fields, nFields, base, outerStride, innerStride, outerCount, innerCount);
}

static inline void launchUnpack2D(
    const double* buf, double* const* fields, int nFields,
    int base, int outerStride, int innerStride,
    int outerCount, int innerCount, cudaStream_t s)
{
    int total = outerCount * innerCount * nFields;
    if (total <= 0) return;
    k_batchUnpack2D<<<(total + BATCH_BLK - 1) / BATCH_BLK, BATCH_BLK, 0, s>>>(
        buf, fields, nFields, base, outerStride, innerStride, outerCount, innerCount);
}

void EMfields3D::gpuBatchedHaloExchange(
    double** h_fieldPtrs,
    int nFields,
    int nx, int ny, int nz,
    bool isCenterFlag,
    bool isFaceOnlyFlag,
    bool needInterp,
    bool isParticle,
    cudaStream_t stream)
{
    const VirtualTopology3D* vct = &get_vct();
    const int myrank = vct->getCartesian_rank();

    // ---- Neighbour ranks ----
    const int xlN = isParticle ? vct->getXleft_neighbor_P()  : vct->getXleft_neighbor();
    const int xrN = isParticle ? vct->getXright_neighbor_P() : vct->getXright_neighbor();
    const int ylN = isParticle ? vct->getYleft_neighbor_P()  : vct->getYleft_neighbor();
    const int yrN = isParticle ? vct->getYright_neighbor_P() : vct->getYright_neighbor();
    const int zlN = isParticle ? vct->getZleft_neighbor_P()  : vct->getZleft_neighbor();
    const int zrN = isParticle ? vct->getZright_neighbor_P() : vct->getZright_neighbor();

    const MPI_Comm comm = isParticle ? vct->getParticleComm() : vct->getFieldComm();

    // ---- Copy field pointers via pinned staging → device ----
    memcpy(h_ptrArray_, h_fieldPtrs, nFields * sizeof(double*));
    cudaMemcpyAsync(d_ptrArray_, h_ptrArray_, nFields * sizeof(double*),
                    cudaMemcpyHostToDevice, stream);

    // ---- Active-direction flags (not NULL and not self) ----
    //   0=XL  1=XR  2=YL  3=YR  4=ZL  5=ZR
    int cc[6];
    cc[0] = (xlN != MPI_PROC_NULL && xlN != myrank) ? 1 : 0;
    cc[1] = (xrN != MPI_PROC_NULL && xrN != myrank) ? 1 : 0;
    cc[2] = (ylN != MPI_PROC_NULL && ylN != myrank) ? 1 : 0;
    cc[3] = (yrN != MPI_PROC_NULL && yrN != myrank) ? 1 : 0;
    cc[4] = (zlN != MPI_PROC_NULL && zlN != myrank) ? 1 : 0;
    cc[5] = (zrN != MPI_PROC_NULL && zrN != myrank) ? 1 : 0;

    const int offset = isCenterFlag ? 0 : 1;

    const int tag_XL = 1, tag_XR = 4;
    const int tag_YL = 2, tag_YR = 5;
    const int tag_ZL = 3, tag_ZR = 6;

    MPI_Status  mpiStat[12];
    MPI_Request mpiReq[12];

    double* const* d_ptrs = (double* const*) d_ptrArray_;

    // ---- Stencil sizes ----
    const int nyzF = (ny - 2) * (nz - 2);   // YZ face element count per field
    const int nxzF = (nx - 2) * (nz - 2);   // XZ face
    const int nxyF = (nx - 2) * (ny - 2);   // XY face

    // =====================================================================
    //  PHASE 1 : Face exchange
    // =====================================================================

    // ---- Pack faces into contiguous send buffers ----
    if (cc[0]) {
        int ix = 1 + offset;
        launchPack2D(d_haloBuf_send_[0], d_ptrs, nFields,
                     ix*ny*nz + 1*nz + 1, nz, 1, ny-2, nz-2, stream);
    }
    if (cc[1]) {
        int ix = nx - 2 - offset;
        launchPack2D(d_haloBuf_send_[1], d_ptrs, nFields,
                     ix*ny*nz + 1*nz + 1, nz, 1, ny-2, nz-2, stream);
    }
    if (cc[2]) {
        int iy = 1 + offset;
        launchPack2D(d_haloBuf_send_[2], d_ptrs, nFields,
                     1*ny*nz + iy*nz + 1, ny*nz, 1, nx-2, nz-2, stream);
    }
    if (cc[3]) {
        int iy = ny - 2 - offset;
        launchPack2D(d_haloBuf_send_[3], d_ptrs, nFields,
                     1*ny*nz + iy*nz + 1, ny*nz, 1, nx-2, nz-2, stream);
    }
    if (cc[4]) {
        int iz = 1 + offset;
        launchPack2D(d_haloBuf_send_[4], d_ptrs, nFields,
                     1*ny*nz + 1*nz + iz, ny*nz, nz, nx-2, ny-2, stream);
    }
    if (cc[5]) {
        int iz = nz - 2 - offset;
        launchPack2D(d_haloBuf_send_[5], d_ptrs, nFields,
                     1*ny*nz + 1*nz + iz, ny*nz, nz, nx-2, ny-2, stream);
    }

    // Wait for pack to finish before MPI reads the buffers
    cudaStreamSynchronize(stream);

    // ---- Post face Irecv / Isend ----
    int rcnt = 0, scnt;
    if (cc[0]) MPI_Irecv(d_haloBuf_recv_[0], nyzF*nFields, MPI_DOUBLE, xlN, tag_XR, comm, &mpiReq[rcnt++]);
    if (cc[1]) MPI_Irecv(d_haloBuf_recv_[1], nyzF*nFields, MPI_DOUBLE, xrN, tag_XL, comm, &mpiReq[rcnt++]);
    if (cc[2]) MPI_Irecv(d_haloBuf_recv_[2], nxzF*nFields, MPI_DOUBLE, ylN, tag_YR, comm, &mpiReq[rcnt++]);
    if (cc[3]) MPI_Irecv(d_haloBuf_recv_[3], nxzF*nFields, MPI_DOUBLE, yrN, tag_YL, comm, &mpiReq[rcnt++]);
    if (cc[4]) MPI_Irecv(d_haloBuf_recv_[4], nxyF*nFields, MPI_DOUBLE, zlN, tag_ZR, comm, &mpiReq[rcnt++]);
    if (cc[5]) MPI_Irecv(d_haloBuf_recv_[5], nxyF*nFields, MPI_DOUBLE, zrN, tag_ZL, comm, &mpiReq[rcnt++]);

    scnt = rcnt;
    if (cc[0]) MPI_Isend(d_haloBuf_send_[0], nyzF*nFields, MPI_DOUBLE, xlN, tag_XL, comm, &mpiReq[scnt++]);
    if (cc[1]) MPI_Isend(d_haloBuf_send_[1], nyzF*nFields, MPI_DOUBLE, xrN, tag_XR, comm, &mpiReq[scnt++]);
    if (cc[2]) MPI_Isend(d_haloBuf_send_[2], nxzF*nFields, MPI_DOUBLE, ylN, tag_YL, comm, &mpiReq[scnt++]);
    if (cc[3]) MPI_Isend(d_haloBuf_send_[3], nxzF*nFields, MPI_DOUBLE, yrN, tag_YR, comm, &mpiReq[scnt++]);
    if (cc[4]) MPI_Isend(d_haloBuf_send_[4], nxyF*nFields, MPI_DOUBLE, zlN, tag_ZL, comm, &mpiReq[scnt++]);
    if (cc[5]) MPI_Isend(d_haloBuf_send_[5], nxyF*nFields, MPI_DOUBLE, zrN, tag_ZR, comm, &mpiReq[scnt++]);

    // ---- Self-copy faces (periodic self-neighbour, batched) ----
    {
        constexpr int BLK = 16;
        dim3 block(BLK, BLK);
        if (xlN == myrank && xrN == myrank) {
            dim3 grid(((ny-2)+BLK-1)/BLK, ((nz-2)+BLK-1)/BLK, nFields);
            gpuBatchSelfCopyFaceX<<<grid, block, 0, stream>>>(d_ptrs, nx, ny, nz);
        }
        if (ylN == myrank && yrN == myrank) {
            dim3 grid(((nx-2)+BLK-1)/BLK, ((nz-2)+BLK-1)/BLK, nFields);
            gpuBatchSelfCopyFaceY<<<grid, block, 0, stream>>>(d_ptrs, nx, ny, nz);
        }
        if (zlN == myrank && zrN == myrank) {
            dim3 grid(((nx-2)+BLK-1)/BLK, ((ny-2)+BLK-1)/BLK, nFields);
            gpuBatchSelfCopyFaceZ<<<grid, block, 0, stream>>>(d_ptrs, nx, ny, nz);
        }
        // No sync needed: self-copy and unpack share the same stream,
        // so CUDA stream ordering guarantees self-copy completes first.
    }

    if (scnt > 0) MPI_Waitall(scnt, mpiReq, mpiStat);

    // ---- Unpack face recv buffers into ghost faces ----
    if (cc[0])
        launchUnpack2D(d_haloBuf_recv_[0], d_ptrs, nFields,
                       0*ny*nz + 1*nz + 1, nz, 1, ny-2, nz-2, stream);
    if (cc[1])
        launchUnpack2D(d_haloBuf_recv_[1], d_ptrs, nFields,
                       (nx-1)*ny*nz + 1*nz + 1, nz, 1, ny-2, nz-2, stream);
    if (cc[2])
        launchUnpack2D(d_haloBuf_recv_[2], d_ptrs, nFields,
                       1*ny*nz + 0*nz + 1, ny*nz, 1, nx-2, nz-2, stream);
    if (cc[3])
        launchUnpack2D(d_haloBuf_recv_[3], d_ptrs, nFields,
                       1*ny*nz + (ny-1)*nz + 1, ny*nz, 1, nx-2, nz-2, stream);
    if (cc[4])
        launchUnpack2D(d_haloBuf_recv_[4], d_ptrs, nFields,
                       1*ny*nz + 1*nz + 0, ny*nz, nz, nx-2, ny-2, stream);
    if (cc[5])
        launchUnpack2D(d_haloBuf_recv_[5], d_ptrs, nFields,
                       1*ny*nz + 1*nz + (nz-1), ny*nz, nz, nx-2, ny-2, stream);

    // =====================================================================
    //  PHASE 2 : Edge exchange  (skip if face-only)
    // =====================================================================
    if (!isFaceOnlyFlag) {

        // Face unpack and edge pack share the same stream — CUDA
        // stream ordering guarantees face data is visible to edge packing.

        // ---- Pack edges ----
        // Y-edges to X neighbours:  (ix_send, jy=1..ny-2, jz=0 and/or nz-1)
        //   Edge at jz=0 exists if cc[4] (ZL active)
        //   Edge at jz=nz-1 exists if cc[5] (ZR active)
        int edgeYlen = ny - 2, edgeZlen = nz - 2, edgeXlen = nx - 2;

        // --- X-direction edges (Y-edges sent to X neighbours) ---
        for (int dir = 0; dir < 2; ++dir) {   // dir 0=XL, 1=XR
            if (!cc[dir]) continue;
            int ix_send = (dir == 0) ? 1 : (nx - 2);
            double* buf = d_haloBuf_send_[dir];
            int off = 0;
            if (cc[4]) {  // ZL edge at jz=0
                launchPack2D(buf + off, d_ptrs, nFields,
                             ix_send*ny*nz + 1*nz + 0, nz, 1, edgeYlen, 1, stream);
                off += edgeYlen * nFields;
            }
            if (cc[5]) {  // ZR edge at jz=nz-1
                launchPack2D(buf + off, d_ptrs, nFields,
                             ix_send*ny*nz + 1*nz + (nz-1), nz, 1, edgeYlen, 1, stream);
                off += edgeYlen * nFields;
            }
        }

        // --- Y-direction edges (Z-edges sent to Y neighbours) ---
        for (int dir = 2; dir < 4; ++dir) {   // dir 2=YL, 3=YR
            if (!cc[dir]) continue;
            int iy_send = (dir == 2) ? 1 : (ny - 2);
            double* buf = d_haloBuf_send_[dir];
            int off = 0;
            if (cc[0]) {  // XL edge at ix=0
                launchPack2D(buf + off, d_ptrs, nFields,
                             0*ny*nz + iy_send*nz + 1, 1, 1, edgeZlen, 1, stream);
                off += edgeZlen * nFields;
            }
            if (cc[1]) {  // XR edge at ix=nx-1
                launchPack2D(buf + off, d_ptrs, nFields,
                             (nx-1)*ny*nz + iy_send*nz + 1, 1, 1, edgeZlen, 1, stream);
                off += edgeZlen * nFields;
            }
        }

        // --- Z-direction edges (X-edges sent to Z neighbours) ---
        for (int dir = 4; dir < 6; ++dir) {   // dir 4=ZL, 5=ZR
            if (!cc[dir]) continue;
            int iz_send = (dir == 4) ? 1 : (nz - 2);
            double* buf = d_haloBuf_send_[dir];
            int off = 0;
            if (cc[2]) {  // YL edge at iy=0
                launchPack2D(buf + off, d_ptrs, nFields,
                             1*ny*nz + 0*nz + iz_send, ny*nz, 1, edgeXlen, 1, stream);
                off += edgeXlen * nFields;
            }
            if (cc[3]) {  // YR edge at iy=ny-1
                launchPack2D(buf + off, d_ptrs, nFields,
                             1*ny*nz + (ny-1)*nz + iz_send, ny*nz, 1, edgeXlen, 1, stream);
                off += edgeXlen * nFields;
            }
        }

        cudaStreamSynchronize(stream);

        // ---- Post edge Irecv / Isend ----
        rcnt = 0;

        // Recv Y-edges from X neighbours
        for (int dir = 0; dir < 2; ++dir) {
            if (!cc[dir]) continue;
            int nEdges = (cc[4] ? 1 : 0) + (cc[5] ? 1 : 0);
            if (nEdges == 0) continue;
            int neighbor = (dir == 0) ? xlN : xrN;
            int tag      = (dir == 0) ? tag_XR : tag_XL;
            MPI_Irecv(d_haloBuf_recv_[dir], nEdges*edgeYlen*nFields, MPI_DOUBLE,
                      neighbor, tag, comm, &mpiReq[rcnt++]);
        }
        // Recv Z-edges from Y neighbours
        for (int dir = 2; dir < 4; ++dir) {
            if (!cc[dir]) continue;
            int nEdges = (cc[0] ? 1 : 0) + (cc[1] ? 1 : 0);
            if (nEdges == 0) continue;
            int neighbor = (dir == 2) ? ylN : yrN;
            int tag      = (dir == 2) ? tag_YR : tag_YL;
            MPI_Irecv(d_haloBuf_recv_[dir], nEdges*edgeZlen*nFields, MPI_DOUBLE,
                      neighbor, tag, comm, &mpiReq[rcnt++]);
        }
        // Recv X-edges from Z neighbours
        for (int dir = 4; dir < 6; ++dir) {
            if (!cc[dir]) continue;
            int nEdges = (cc[2] ? 1 : 0) + (cc[3] ? 1 : 0);
            if (nEdges == 0) continue;
            int neighbor = (dir == 4) ? zlN : zrN;
            int tag      = (dir == 4) ? tag_ZR : tag_ZL;
            MPI_Irecv(d_haloBuf_recv_[dir], nEdges*edgeXlen*nFields, MPI_DOUBLE,
                      neighbor, tag, comm, &mpiReq[rcnt++]);
        }

        scnt = rcnt;

        // Send Y-edges to X neighbours
        for (int dir = 0; dir < 2; ++dir) {
            if (!cc[dir]) continue;
            int nEdges = (cc[4] ? 1 : 0) + (cc[5] ? 1 : 0);
            if (nEdges == 0) continue;
            int neighbor = (dir == 0) ? xlN : xrN;
            int tag      = (dir == 0) ? tag_XL : tag_XR;
            MPI_Isend(d_haloBuf_send_[dir], nEdges*edgeYlen*nFields, MPI_DOUBLE,
                      neighbor, tag, comm, &mpiReq[scnt++]);
        }
        // Send Z-edges to Y neighbours
        for (int dir = 2; dir < 4; ++dir) {
            if (!cc[dir]) continue;
            int nEdges = (cc[0] ? 1 : 0) + (cc[1] ? 1 : 0);
            if (nEdges == 0) continue;
            int neighbor = (dir == 2) ? ylN : yrN;
            int tag      = (dir == 2) ? tag_YL : tag_YR;
            MPI_Isend(d_haloBuf_send_[dir], nEdges*edgeZlen*nFields, MPI_DOUBLE,
                      neighbor, tag, comm, &mpiReq[scnt++]);
        }
        // Send X-edges to Z neighbours
        for (int dir = 4; dir < 6; ++dir) {
            if (!cc[dir]) continue;
            int nEdges = (cc[2] ? 1 : 0) + (cc[3] ? 1 : 0);
            if (nEdges == 0) continue;
            int neighbor = (dir == 4) ? zlN : zrN;
            int tag      = (dir == 4) ? tag_ZL : tag_ZR;
            MPI_Isend(d_haloBuf_send_[dir], nEdges*edgeXlen*nFields, MPI_DOUBLE,
                      neighbor, tag, comm, &mpiReq[scnt++]);
        }

        // ---- Self-copy edges (batched) ----
        {
            int maxDim = (nx > ny ? (nx > nz ? nx : nz) : (ny > nz ? ny : nz));
            int nblks = (maxDim + 255) / 256;
            if (xlN == myrank && xrN == myrank) {
                dim3 grid(nblks, nFields);
                gpuBatchSelfCopyEdgeX<<<grid, 256, 0, stream>>>(d_ptrs, nx, ny, nz,
                    zrN != MPI_PROC_NULL, zlN != MPI_PROC_NULL,
                    yrN != MPI_PROC_NULL, ylN != MPI_PROC_NULL);
            }
            if (ylN == myrank && yrN == myrank) {
                dim3 grid(nblks, nFields);
                gpuBatchSelfCopyEdgeY<<<grid, 256, 0, stream>>>(d_ptrs, nx, ny, nz,
                    xrN != MPI_PROC_NULL, xlN != MPI_PROC_NULL,
                    zrN != MPI_PROC_NULL, zlN != MPI_PROC_NULL);
            }
            if (zlN == myrank && zrN == myrank) {
                dim3 grid(nblks, nFields);
                gpuBatchSelfCopyEdgeZ<<<grid, 256, 0, stream>>>(d_ptrs, nx, ny, nz,
                    yrN != MPI_PROC_NULL, ylN != MPI_PROC_NULL,
                    xrN != MPI_PROC_NULL, xlN != MPI_PROC_NULL);
            }
            // No sync needed: same stream as unpack.
        }

        if (scnt > 0) MPI_Waitall(scnt, mpiReq, mpiStat);

        // ---- Unpack edge recv buffers ----
        // Y-edges from X neighbours → into ghost ix=0 or nx-1
        for (int dir = 0; dir < 2; ++dir) {
            if (!cc[dir]) continue;
            int ix_recv = (dir == 0) ? 0 : (nx - 1);
            const double* buf = d_haloBuf_recv_[dir];
            int off = 0;
            if (cc[4]) {
                launchUnpack2D(buf + off, d_ptrs, nFields,
                               ix_recv*ny*nz + 1*nz + 0, nz, 1, edgeYlen, 1, stream);
                off += edgeYlen * nFields;
            }
            if (cc[5]) {
                launchUnpack2D(buf + off, d_ptrs, nFields,
                               ix_recv*ny*nz + 1*nz + (nz-1), nz, 1, edgeYlen, 1, stream);
                off += edgeYlen * nFields;
            }
        }
        // Z-edges from Y neighbours → into ghost iy=0 or ny-1
        for (int dir = 2; dir < 4; ++dir) {
            if (!cc[dir]) continue;
            int iy_recv = (dir == 2) ? 0 : (ny - 1);
            const double* buf = d_haloBuf_recv_[dir];
            int off = 0;
            if (cc[0]) {
                launchUnpack2D(buf + off, d_ptrs, nFields,
                               0*ny*nz + iy_recv*nz + 1, 1, 1, edgeZlen, 1, stream);
                off += edgeZlen * nFields;
            }
            if (cc[1]) {
                launchUnpack2D(buf + off, d_ptrs, nFields,
                               (nx-1)*ny*nz + iy_recv*nz + 1, 1, 1, edgeZlen, 1, stream);
                off += edgeZlen * nFields;
            }
        }
        // X-edges from Z neighbours → into ghost iz=0 or nz-1
        for (int dir = 4; dir < 6; ++dir) {
            if (!cc[dir]) continue;
            int iz_recv = (dir == 4) ? 0 : (nz - 1);
            const double* buf = d_haloBuf_recv_[dir];
            int off = 0;
            if (cc[2]) {
                launchUnpack2D(buf + off, d_ptrs, nFields,
                               1*ny*nz + 0*nz + iz_recv, ny*nz, 1, edgeXlen, 1, stream);
                off += edgeXlen * nFields;
            }
            if (cc[3]) {
                launchUnpack2D(buf + off, d_ptrs, nFields,
                               1*ny*nz + (ny-1)*nz + iz_recv, ny*nz, 1, edgeXlen, 1, stream);
                off += edgeXlen * nFields;
            }
        }

        // =================================================================
        //  PHASE 3 : Corner exchange
        // =================================================================

        // Corners are only exchanged if at least one Y and one Z neighbour is
        // active, and only with X neighbours.
        if ((cc[2] || cc[3]) && (cc[4] || cc[5])) {
            // Edge unpack and corner pack share the same stream — CUDA
            // stream ordering guarantees visibility.

            // Pack corners: 4 elements at (ix, 0, 0/nz-1, ny-1, 0/nz-1)
            if (cc[0]) {
                int total = nFields * 4;
                k_batchPackCorners4<<<(total+BATCH_BLK-1)/BATCH_BLK, BATCH_BLK, 0, stream>>>(
                    d_haloBuf_send_[0], d_ptrs, nFields, 1*ny*nz, ny, nz);
            }
            if (cc[1]) {
                int total = nFields * 4;
                k_batchPackCorners4<<<(total+BATCH_BLK-1)/BATCH_BLK, BATCH_BLK, 0, stream>>>(
                    d_haloBuf_send_[1], d_ptrs, nFields, (nx-2)*ny*nz, ny, nz);
            }

            cudaStreamSynchronize(stream);

            // Post corner Irecv / Isend
            rcnt = 0;
            if (cc[0]) MPI_Irecv(d_haloBuf_recv_[0], 4*nFields, MPI_DOUBLE, xlN, tag_XR, comm, &mpiReq[rcnt++]);
            if (cc[1]) MPI_Irecv(d_haloBuf_recv_[1], 4*nFields, MPI_DOUBLE, xrN, tag_XL, comm, &mpiReq[rcnt++]);
            scnt = rcnt;
            if (cc[0]) MPI_Isend(d_haloBuf_send_[0], 4*nFields, MPI_DOUBLE, xlN, tag_XL, comm, &mpiReq[scnt++]);
            if (cc[1]) MPI_Isend(d_haloBuf_send_[1], 4*nFields, MPI_DOUBLE, xrN, tag_XR, comm, &mpiReq[scnt++]);

            // Corner self-copy (batched)
            {
                if (xlN == myrank && xrN == myrank) {
                    gpuBatchSelfCopyCornerX<<<nFields, 1, 0, stream>>>(d_ptrs, nx, ny, nz,
                        ylN != MPI_PROC_NULL, yrN != MPI_PROC_NULL,
                        zlN != MPI_PROC_NULL, zrN != MPI_PROC_NULL);
                } else if (ylN == myrank && yrN == myrank) {
                    gpuBatchSelfCopyCornerY<<<nFields, 1, 0, stream>>>(d_ptrs, nx, ny, nz,
                        xlN != MPI_PROC_NULL, xrN != MPI_PROC_NULL,
                        zlN != MPI_PROC_NULL, zrN != MPI_PROC_NULL);
                } else if (zlN == myrank && zrN == myrank) {
                    gpuBatchSelfCopyCornerZ<<<nFields, 1, 0, stream>>>(d_ptrs, nx, ny, nz,
                        ylN != MPI_PROC_NULL, yrN != MPI_PROC_NULL,
                        xlN != MPI_PROC_NULL, xrN != MPI_PROC_NULL);
                }
                // No sync needed: same stream as unpack.
            }

            if (scnt > 0) MPI_Waitall(scnt, mpiReq, mpiStat);

            // Unpack corners
            if (cc[0]) {
                int total = nFields * 4;
                k_batchUnpackCorners4<<<(total+BATCH_BLK-1)/BATCH_BLK, BATCH_BLK, 0, stream>>>(
                    d_haloBuf_recv_[0], d_ptrs, nFields, 0*ny*nz, ny, nz);
            }
            if (cc[1]) {
                int total = nFields * 4;
                k_batchUnpackCorners4<<<(total+BATCH_BLK-1)/BATCH_BLK, BATCH_BLK, 0, stream>>>(
                    d_haloBuf_recv_[1], d_ptrs, nFields, (nx-1)*ny*nz, ny, nz);
            }
        }
    } // end !isFaceOnlyFlag

    // =====================================================================
    //  Additive interpolation  (for moments: ghost → interior boundary)
    // =====================================================================
    if (needInterp) {
        // Unpack and additive kernels share the same stream — CUDA
        // stream ordering guarantees all ghost data is visible.

        // The hasNeighbor flags for additive kernels use particle topology
        bool hasXR = vct->hasXrghtNeighbor_P();
        bool hasXL = vct->hasXleftNeighbor_P();
        bool hasYR = vct->hasYrghtNeighbor_P();
        bool hasYL = vct->hasYleftNeighbor_P();
        bool hasZR = vct->hasZrghtNeighbor_P();
        bool hasZL = vct->hasZleftNeighbor_P();

        int nxr = nx - 2, nyr = ny - 2, nzr = nz - 2;

        // Face add
        {
            int total = nyr * nzr * nFields;
            k_batchAddFaceX<<<(total+BATCH_BLK-1)/BATCH_BLK, BATCH_BLK, 0, stream>>>(
                d_ptrs, nFields, nx, ny, nz, hasXR, hasXL);
        }
        {
            int total = nxr * nzr * nFields;
            k_batchAddFaceY<<<(total+BATCH_BLK-1)/BATCH_BLK, BATCH_BLK, 0, stream>>>(
                d_ptrs, nFields, nx, ny, nz, hasYR, hasYL);
        }
        {
            int total = nxr * nyr * nFields;
            k_batchAddFaceZ<<<(total+BATCH_BLK-1)/BATCH_BLK, BATCH_BLK, 0, stream>>>(
                d_ptrs, nFields, nx, ny, nz, hasZR, hasZL);
        }

        // Edge add
        {
            int total = nzr * nFields;
            k_batchAddEdgeZ<<<(total+BATCH_BLK-1)/BATCH_BLK, BATCH_BLK, 0, stream>>>(
                d_ptrs, nFields, nx, ny, nz, hasXR, hasXL, hasYR, hasYL);
        }
        {
            int total = nyr * nFields;
            k_batchAddEdgeY<<<(total+BATCH_BLK-1)/BATCH_BLK, BATCH_BLK, 0, stream>>>(
                d_ptrs, nFields, nx, ny, nz, hasXR, hasXL, hasZR, hasZL);
        }
        {
            int total = nxr * nFields;
            k_batchAddEdgeX<<<(total+BATCH_BLK-1)/BATCH_BLK, BATCH_BLK, 0, stream>>>(
                d_ptrs, nFields, nx, ny, nz, hasYR, hasYL, hasZR, hasZL);
        }

        // Corner add
        k_batchAddCorner<<<(nFields+BATCH_BLK-1)/BATCH_BLK, BATCH_BLK, 0, stream>>>(
            d_ptrs, nFields, nx, ny, nz, hasXR, hasXL, hasYR, hasYL, hasZR, hasZL);

        cudaStreamSynchronize(stream);
    }
}

#endif // GPU_SOLVER
