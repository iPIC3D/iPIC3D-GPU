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

#include "EMfields3D.h"
#include "GPUHaloComm.cuh" // single-array self-copy kernels, BC kernels
#include "GPUSolverMPITypes.h"
#include "VCtopology3D.h"
#include <climits>
#include <cstdio>
#include <cstdlib>
#include <cstring>

// =========================================================================
//  Helper: linear index into a flat (nx, ny, nz) row-major array
// =========================================================================
__device__ __forceinline__ int bidx3(int i, int j, int k, int ny, int nz) {
  return (i * ny + j) * nz + k;
}

// =========================================================================
//  BATCHED PACK / UNPACK KERNELS
//
//  k_batchPack2D / k_batchUnpack2D handle any rectangular 2D region
//  described by (base, outerStride, innerStride, outerCount, innerCount).
//  This covers faces, edges (innerCount=1), and single elements.
// =========================================================================

__global__ void k_batchPack2D(cudaSolverType* __restrict__ buf,
                              cudaSolverType* const* __restrict__ fields,
                              int nFields, int base, int outerStride,
                              int innerStride, int outerCount, int innerCount) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int elemsPerField = outerCount * innerCount;
  int total = elemsPerField * nFields;
  if (tid >= total)
    return;

  int f = tid / elemsPerField;
  int rem = tid % elemsPerField;
  int oi = rem / innerCount;
  int ii = rem % innerCount;

  buf[tid] = fields[f][base + oi * outerStride + ii * innerStride];
}

__global__ void k_batchUnpack2D(const cudaSolverType* __restrict__ buf,
                                cudaSolverType* const* __restrict__ fields,
                                int nFields, int base, int outerStride,
                                int innerStride, int outerCount,
                                int innerCount) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int elemsPerField = outerCount * innerCount;
  int total = elemsPerField * nFields;
  if (tid >= total)
    return;

  int f = tid / elemsPerField;
  int rem = tid % elemsPerField;
  int oi = rem / innerCount;
  int ii = rem % innerCount;

  fields[f][base + oi * outerStride + ii * innerStride] = buf[tid];
}

// =========================================================================
//  CORNER PACK / UNPACK  (4 scattered elements per field per direction)
// =========================================================================

__global__ void k_batchPackCorners4(cudaSolverType* __restrict__ buf,
                                    cudaSolverType* const* __restrict__ fields,
                                    int nFields,
                                    int base, // ix * ny * nz
                                    int ny, int nz) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= nFields * 4)
    return;

  int f = tid / 4;
  int c = tid % 4;

  // 4 corners of the YZ face: (iy,iz) ∈ {(0,0),(0,nz-1),(ny-1,0),(ny-1,nz-1)}
  int offsets[4] = {0, nz - 1, (ny - 1) * nz, ny * nz - 1};
  buf[tid] = fields[f][base + offsets[c]];
}

__global__ void
k_batchUnpackCorners4(const cudaSolverType* __restrict__ buf,
                      cudaSolverType* const* __restrict__ fields, int nFields,
                      int base, int ny, int nz) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= nFields * 4)
    return;

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

__global__ void k_batchAddFaceX(cudaSolverType* const* __restrict__ fields,
                                int nFields, int nx, int ny, int nz,
                                bool hasRight, bool hasLeft) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int faceSize = (ny - 2) * (nz - 2);
  int total = faceSize * nFields;
  if (tid >= total)
    return;

  int f = tid / faceSize;
  int rem = tid % faceSize;
  int j = rem / (nz - 2) + 1;
  int k = rem % (nz - 2) + 1;

  if (hasRight)
    fields[f][bidx3(nx - 2, j, k, ny, nz)] +=
        fields[f][bidx3(nx - 1, j, k, ny, nz)];
  if (hasLeft)
    fields[f][bidx3(1, j, k, ny, nz)] += fields[f][bidx3(0, j, k, ny, nz)];
}

__global__ void k_batchAddFaceY(cudaSolverType* const* __restrict__ fields,
                                int nFields, int nx, int ny, int nz,
                                bool hasRight, bool hasLeft) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int faceSize = (nx - 2) * (nz - 2);
  int total = faceSize * nFields;
  if (tid >= total)
    return;

  int f = tid / faceSize;
  int rem = tid % faceSize;
  int i = rem / (nz - 2) + 1;
  int k = rem % (nz - 2) + 1;

  if (hasRight)
    fields[f][bidx3(i, ny - 2, k, ny, nz)] +=
        fields[f][bidx3(i, ny - 1, k, ny, nz)];
  if (hasLeft)
    fields[f][bidx3(i, 1, k, ny, nz)] += fields[f][bidx3(i, 0, k, ny, nz)];
}

__global__ void k_batchAddFaceZ(cudaSolverType* const* __restrict__ fields,
                                int nFields, int nx, int ny, int nz,
                                bool hasRight, bool hasLeft) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int faceSize = (nx - 2) * (ny - 2);
  int total = faceSize * nFields;
  if (tid >= total)
    return;

  int f = tid / faceSize;
  int rem = tid % faceSize;
  int i = rem / (ny - 2) + 1;
  int j = rem % (ny - 2) + 1;

  if (hasRight)
    fields[f][bidx3(i, j, nz - 2, ny, nz)] +=
        fields[f][bidx3(i, j, nz - 1, ny, nz)];
  if (hasLeft)
    fields[f][bidx3(i, j, 1, ny, nz)] += fields[f][bidx3(i, j, 0, ny, nz)];
}

__global__ void k_batchAddEdgeZ(cudaSolverType* const* __restrict__ fields,
                                int nFields, int nx, int ny, int nz, bool hasXR,
                                bool hasXL, bool hasYR, bool hasYL) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int edgeLen = nz - 2;
  int total = edgeLen * nFields;
  if (tid >= total)
    return;

  int f = tid / edgeLen;
  int k = tid % edgeLen + 1;

  if (hasXR && hasYR)
    fields[f][bidx3(nx - 2, ny - 2, k, ny, nz)] +=
        fields[f][bidx3(nx - 1, ny - 1, k, ny, nz)];
  if (hasXL && hasYL)
    fields[f][bidx3(1, 1, k, ny, nz)] += fields[f][bidx3(0, 0, k, ny, nz)];
  if (hasXR && hasYL)
    fields[f][bidx3(nx - 2, 1, k, ny, nz)] +=
        fields[f][bidx3(nx - 1, 0, k, ny, nz)];
  if (hasXL && hasYR)
    fields[f][bidx3(1, ny - 2, k, ny, nz)] +=
        fields[f][bidx3(0, ny - 1, k, ny, nz)];
}

__global__ void k_batchAddEdgeY(cudaSolverType* const* __restrict__ fields,
                                int nFields, int nx, int ny, int nz, bool hasXR,
                                bool hasXL, bool hasZR, bool hasZL) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int edgeLen = ny - 2;
  int total = edgeLen * nFields;
  if (tid >= total)
    return;

  int f = tid / edgeLen;
  int j = tid % edgeLen + 1;

  if (hasXR && hasZR)
    fields[f][bidx3(nx - 2, j, nz - 2, ny, nz)] +=
        fields[f][bidx3(nx - 1, j, nz - 1, ny, nz)];
  if (hasXL && hasZL)
    fields[f][bidx3(1, j, 1, ny, nz)] += fields[f][bidx3(0, j, 0, ny, nz)];
  if (hasXL && hasZR)
    fields[f][bidx3(1, j, nz - 2, ny, nz)] +=
        fields[f][bidx3(0, j, nz - 1, ny, nz)];
  if (hasXR && hasZL)
    fields[f][bidx3(nx - 2, j, 1, ny, nz)] +=
        fields[f][bidx3(nx - 1, j, 0, ny, nz)];
}

__global__ void k_batchAddEdgeX(cudaSolverType* const* __restrict__ fields,
                                int nFields, int nx, int ny, int nz, bool hasYR,
                                bool hasYL, bool hasZR, bool hasZL) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int edgeLen = nx - 2;
  int total = edgeLen * nFields;
  if (tid >= total)
    return;

  int f = tid / edgeLen;
  int i = tid % edgeLen + 1;

  if (hasYR && hasZR)
    fields[f][bidx3(i, ny - 2, nz - 2, ny, nz)] +=
        fields[f][bidx3(i, ny - 1, nz - 1, ny, nz)];
  if (hasYL && hasZL)
    fields[f][bidx3(i, 1, 1, ny, nz)] += fields[f][bidx3(i, 0, 0, ny, nz)];
  if (hasYL && hasZR)
    fields[f][bidx3(i, 1, nz - 2, ny, nz)] +=
        fields[f][bidx3(i, 0, nz - 1, ny, nz)];
  if (hasYR && hasZL)
    fields[f][bidx3(i, ny - 2, 1, ny, nz)] +=
        fields[f][bidx3(i, ny - 1, 0, ny, nz)];
}

__global__ void k_batchAddCorner(cudaSolverType* const* __restrict__ fields,
                                 int nFields, int nx, int ny, int nz,
                                 bool hasXR, bool hasXL, bool hasYR, bool hasYL,
                                 bool hasZR, bool hasZL) {
  int f = blockIdx.x * blockDim.x + threadIdx.x;
  if (f >= nFields)
    return;

  if (hasXR && hasYR && hasZR)
    fields[f][bidx3(nx - 2, ny - 2, nz - 2, ny, nz)] +=
        fields[f][bidx3(nx - 1, ny - 1, nz - 1, ny, nz)];
  if (hasXL && hasYR && hasZR)
    fields[f][bidx3(1, ny - 2, nz - 2, ny, nz)] +=
        fields[f][bidx3(0, ny - 1, nz - 1, ny, nz)];
  if (hasXR && hasYL && hasZR)
    fields[f][bidx3(nx - 2, 1, nz - 2, ny, nz)] +=
        fields[f][bidx3(nx - 1, 0, nz - 1, ny, nz)];
  if (hasXL && hasYL && hasZR)
    fields[f][bidx3(1, 1, nz - 2, ny, nz)] +=
        fields[f][bidx3(0, 0, nz - 1, ny, nz)];
  if (hasXR && hasYR && hasZL)
    fields[f][bidx3(nx - 2, ny - 2, 1, ny, nz)] +=
        fields[f][bidx3(nx - 1, ny - 1, 0, ny, nz)];
  if (hasXL && hasYR && hasZL)
    fields[f][bidx3(1, ny - 2, 1, ny, nz)] +=
        fields[f][bidx3(0, ny - 1, 0, ny, nz)];
  if (hasXR && hasYL && hasZL)
    fields[f][bidx3(nx - 2, 1, 1, ny, nz)] +=
        fields[f][bidx3(nx - 1, 0, 0, ny, nz)];
  if (hasXL && hasYL && hasZL)
    fields[f][bidx3(1, 1, 1, ny, nz)] += fields[f][bidx3(0, 0, 0, ny, nz)];
}

// =========================================================================
//  BATCHED HALO EXCHANGE
//
//  Replaces gpuNBDerivedHaloComm for multiple arrays at once.
//  Uses explicit CUDA pack/unpack into persistent contiguous GPU buffers
//  followed by MPI_Isend/Irecv of contiguous data.
// =========================================================================

static constexpr int BATCH_BLK = 256;
static constexpr int HALO_SEND_TAG[6] = {1, 4, 2, 5, 3, 6};
static constexpr int HALO_RECV_TAG[6] = {4, 1, 5, 2, 6, 3};

static inline int haloBatchBlocks(int total) {
  return 1 + (total - 1) / BATCH_BLK;
}

[[noreturn]] static void batchedHaloContractFailure(const char* reason) {
  std::fprintf(stderr, "GPU batched-halo contract failure: %s\n", reason);
  MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
  std::abort();
}

[[noreturn]] static void batchedHaloMpiFailure(int error, MPI_Comm comm,
                                               const char* operation) {
  char message[MPI_MAX_ERROR_STRING] = {};
  int length = 0;
  MPI_Error_string(error, message, &length);
  std::fprintf(stderr, "GPU batched-halo MPI failure in %s: %.*s\n", operation,
               length, message);
  MPI_Abort(comm, error);
  std::abort();
}

static void haloIrecv(void* buffer, int count, MPI_Datatype datatype,
                      int source, int tag, MPI_Comm comm,
                      MPI_Request* request) {
  const int error =
      MPI_Irecv(buffer, count, datatype, source, tag, comm, request);
  if (error != MPI_SUCCESS)
    batchedHaloMpiFailure(error, comm, "MPI_Irecv");
}

static void haloIsend(const void* buffer, int count, MPI_Datatype datatype,
                      int destination, int tag, MPI_Comm comm,
                      MPI_Request* request) {
  const int error =
      MPI_Isend(buffer, count, datatype, destination, tag, comm, request);
  if (error != MPI_SUCCESS)
    batchedHaloMpiFailure(error, comm, "MPI_Isend");
}

static void haloWaitall(int count, MPI_Request* requests, MPI_Status* statuses,
                        MPI_Comm comm) {
  const int error = MPI_Waitall(count, requests, statuses);
  if (error != MPI_SUCCESS)
    batchedHaloMpiFailure(error, comm, "MPI_Waitall");
}

void EMfields3D::gpuInitializeHaloPlans() {
  if (haloPlansInitialized_)
    return;

  const VirtualTopology3D& vct = get_vct();
  for (int kind = HALO_FIELD; kind <= HALO_PARTICLE; ++kind) {
    const bool particle = kind == HALO_PARTICLE;
    HaloTopologyPlan& plan = haloTopologyPlans_[kind];
    plan.comm = particle ? vct.getParticleComm() : vct.getFieldComm();
    const int rank =
        particle ? vct.getParticleCartesian_rank() : vct.getCartesian_rank();
    plan.neighbors[0] =
        particle ? vct.getXleft_neighbor_P() : vct.getXleft_neighbor();
    plan.neighbors[1] =
        particle ? vct.getXright_neighbor_P() : vct.getXright_neighbor();
    plan.neighbors[2] =
        particle ? vct.getYleft_neighbor_P() : vct.getYleft_neighbor();
    plan.neighbors[3] =
        particle ? vct.getYright_neighbor_P() : vct.getYright_neighbor();
    plan.neighbors[4] =
        particle ? vct.getZleft_neighbor_P() : vct.getZleft_neighbor();
    plan.neighbors[5] =
        particle ? vct.getZright_neighbor_P() : vct.getZright_neighbor();
    if (plan.comm == MPI_COMM_NULL || rank == MPI_PROC_NULL)
      batchedHaloContractFailure("halo topology is not initialized");

    bool remote[6] = {};
    for (int direction = 0; direction < 6; ++direction) {
      plan.present[direction] = plan.neighbors[direction] != MPI_PROC_NULL;
      remote[direction] =
          plan.present[direction] && plan.neighbors[direction] != rank;
      if (remote[direction])
        plan.remoteDirections[plan.remoteDirectionCount++] = direction;
    }
    for (int axis = 0; axis < 3; ++axis)
      plan.selfAxis[axis] = plan.neighbors[2 * axis] == rank &&
                            plan.neighbors[2 * axis + 1] == rank;

    static constexpr int edgeDependencies[3][2] = {{4, 5}, {0, 1}, {2, 3}};
    for (int direction = 0; direction < 6; ++direction) {
      const int edgeAxis = direction / 2;
      for (int segment = 0; segment < 2; ++segment) {
        const int dependency = edgeDependencies[edgeAxis][segment];
        if (remote[dependency])
          plan.edgeSegments[direction][plan.edgeMultiplicity[direction]++] =
              segment;
      }
      if (remote[direction] && plan.edgeMultiplicity[direction] > 0)
        plan.edgeDirections[plan.edgeDirectionCount++] = direction;
    }
    for (int direction = 0; direction < 2; ++direction) {
      if (remote[direction])
        plan.cornerDirections[plan.cornerDirectionCount++] = direction;
    }
    const bool remoteY = remote[2] || remote[3];
    const bool remoteZ = remote[4] || remote[5];
    plan.hasRemoteYZ = remoteY && remoteZ;
  }

  auto buildGeometry = [&](HaloGeometryPlan& plan, int nx, int ny, int nz,
                           int offset) {
    const int minExtent = offset == 0 ? 3 : 4;
    if (nx < minExtent || ny < minExtent || nz < minExtent)
      batchedHaloContractFailure("invalid canonical halo geometry");
    const size_t volume = (size_t)nx * ny * nz;
    if (volume > INT_MAX)
      batchedHaloContractFailure("invalid canonical halo geometry");

    plan.nx = nx;
    plan.ny = ny;
    plan.nz = nz;
    plan.offset = offset;
    plan.active[0] = nx - 2;
    plan.active[1] = ny - 2;
    plan.active[2] = nz - 2;

    auto setFace = [&](int direction, int sendBase, int recvBase,
                       int outerStride, int innerStride, int outerCount,
                       int innerCount) {
      const size_t elements = (size_t)outerCount * innerCount;
      if (elements > (size_t)INT_MAX / HALO_MAX_BATCH)
        batchedHaloContractFailure("halo face MPI count exceeds INT_MAX");
      HaloFacePlan& face = plan.faces[direction];
      face.sendBase = sendBase;
      face.recvBase = recvBase;
      face.outerStride = outerStride;
      face.innerStride = innerStride;
      face.outerCount = outerCount;
      face.innerCount = innerCount;
      face.elements = static_cast<int>(elements);
    };

    const int yz = ny * nz;
    setFace(0, (1 + offset) * yz + nz + 1, nz + 1, nz, 1, ny - 2, nz - 2);
    setFace(1, (nx - 2 - offset) * yz + nz + 1, (nx - 1) * yz + nz + 1, nz, 1,
            ny - 2, nz - 2);
    setFace(2, yz + (1 + offset) * nz + 1, yz + 1, yz, 1, nx - 2, nz - 2);
    setFace(3, yz + (ny - 2 - offset) * nz + 1, yz + (ny - 1) * nz + 1, yz, 1,
            nx - 2, nz - 2);
    setFace(4, yz + nz + 1 + offset, yz + nz, yz, nz, nx - 2, ny - 2);
    setFace(5, yz + nz + nz - 2 - offset, yz + nz + nz - 1, yz, nz, nx - 2,
            ny - 2);

    auto setEdge = [&](int direction, int segment, int sendBase, int recvBase,
                       int stride, int elements) {
      HaloEdgePlan& edge = plan.edges[direction][segment];
      edge.sendBase = sendBase;
      edge.recvBase = recvBase;
      edge.stride = stride;
      edge.elements = elements;
    };
    for (int direction = 0; direction < 2; ++direction) {
      const int sendX = direction == 0 ? 1 + offset : nx - 2 - offset;
      const int recvX = direction == 0 ? 0 : nx - 1;
      setEdge(direction, 0, sendX * yz + nz, recvX * yz + nz, nz, ny - 2);
      setEdge(direction, 1, sendX * yz + nz + nz - 1, recvX * yz + nz + nz - 1,
              nz, ny - 2);
    }
    for (int direction = 2; direction < 4; ++direction) {
      const int sendY = direction == 2 ? 1 + offset : ny - 2 - offset;
      const int recvY = direction == 2 ? 0 : ny - 1;
      setEdge(direction, 0, sendY * nz + 1, recvY * nz + 1, 1, nz - 2);
      setEdge(direction, 1, (nx - 1) * yz + sendY * nz + 1,
              (nx - 1) * yz + recvY * nz + 1, 1, nz - 2);
    }
    for (int direction = 4; direction < 6; ++direction) {
      const int sendZ = direction == 4 ? 1 + offset : nz - 2 - offset;
      const int recvZ = direction == 4 ? 0 : nz - 1;
      setEdge(direction, 0, yz + sendZ, yz + recvZ, yz, nx - 2);
      setEdge(direction, 1, yz + (ny - 1) * nz + sendZ,
              yz + (ny - 1) * nz + recvZ, yz, nx - 2);
    }

    plan.edgeLength[0] = ny - 2;
    plan.edgeLength[1] = nz - 2;
    plan.edgeLength[2] = nx - 2;
    for (int direction = 0; direction < 6; ++direction) {
      const int worstEdgeElements = 2 * plan.edgeLength[direction / 2];
      int maximum = plan.faces[direction].elements;
      if (worstEdgeElements > maximum)
        maximum = worstEdgeElements;
      if (maximum < 4)
        maximum = 4;
      if (maximum > INT_MAX / HALO_MAX_BATCH)
        batchedHaloContractFailure("halo buffer or MPI count exceeds INT_MAX");
      plan.maxBufferElements[direction] = maximum;
    }

    constexpr int faceBlock = 16;
    plan.selfFaceBlocks[0][0] = (ny - 2 + faceBlock - 1) / faceBlock;
    plan.selfFaceBlocks[0][1] = (nz - 2 + faceBlock - 1) / faceBlock;
    plan.selfFaceBlocks[1][0] = (nx - 2 + faceBlock - 1) / faceBlock;
    plan.selfFaceBlocks[1][1] = (nz - 2 + faceBlock - 1) / faceBlock;
    plan.selfFaceBlocks[2][0] = (nx - 2 + faceBlock - 1) / faceBlock;
    plan.selfFaceBlocks[2][1] = (ny - 2 + faceBlock - 1) / faceBlock;
    int maximumExtent = nx > ny ? nx : ny;
    if (nz > maximumExtent)
      maximumExtent = nz;
    plan.edgeKernelBlocks = (maximumExtent + 255) / 256;
    plan.cornerSendBase[0] = (1 + offset) * yz;
    plan.cornerSendBase[1] = (nx - 2 - offset) * yz;
    plan.cornerRecvBase[0] = 0;
    plan.cornerRecvBase[1] = (nx - 1) * yz;
  };

  buildGeometry(haloGeometryPlans_[HALO_CENTER_OFFSET0], nxc, nyc, nzc, 0);
  buildGeometry(haloGeometryPlans_[HALO_NODE_OFFSET1], nxn, nyn, nzn, 1);
  buildGeometry(haloGeometryPlans_[HALO_NODE_OFFSET0], nxn, nyn, nzn, 0);
  haloPlansInitialized_ = true;
}

const EMfields3D::HaloGeometryPlan&
EMfields3D::gpuSelectHaloGeometry(int nx, int ny, int nz,
                                  bool offsetZero) const {
  if (!haloPlansInitialized_)
    batchedHaloContractFailure("halo plans are not initialized");
  if (nx == nxc && ny == nyc && nz == nzc && offsetZero)
    return haloGeometryPlans_[HALO_CENTER_OFFSET0];
  if (nx == nxn && ny == nyn && nz == nzn)
    return haloGeometryPlans_[offsetZero ? HALO_NODE_OFFSET0
                                         : HALO_NODE_OFFSET1];
  batchedHaloContractFailure("exchange does not match a cached halo geometry");
}

// Helper: launch k_batchPack2D with auto grid sizing
static inline void launchPack2D(cudaSolverType* buf,
                                cudaSolverType* const* fields, int nFields,
                                int base, int outerStride, int innerStride,
                                int outerCount, int innerCount,
                                cudaStream_t s) {
  int total = outerCount * innerCount * nFields;
  if (total <= 0)
    return;
  k_batchPack2D<<<haloBatchBlocks(total), BATCH_BLK, 0, s>>>(
      buf, fields, nFields, base, outerStride, innerStride, outerCount,
      innerCount);
}

static inline void launchUnpack2D(const cudaSolverType* buf,
                                  cudaSolverType* const* fields, int nFields,
                                  int base, int outerStride, int innerStride,
                                  int outerCount, int innerCount,
                                  cudaStream_t s) {
  int total = outerCount * innerCount * nFields;
  if (total <= 0)
    return;
  k_batchUnpack2D<<<haloBatchBlocks(total), BATCH_BLK, 0, s>>>(
      buf, fields, nFields, base, outerStride, innerStride, outerCount,
      innerCount);
}

void EMfields3D::gpuQueueHaloFacePack(cudaSolverType** h_fieldPtrs, int nFields,
                                      const HaloTopologyPlan& topology,
                                      const HaloGeometryPlan& geometry,
                                      cudaStream_t stream) {
  memcpy(h_ptrArray_, h_fieldPtrs, nFields * sizeof(cudaSolverType*));
  cudaErrChk(cudaMemcpyAsync(d_ptrArray_, h_ptrArray_,
                             nFields * sizeof(cudaSolverType*),
                             cudaMemcpyHostToDevice, stream));

  auto* const* d_ptrs = reinterpret_cast<cudaSolverType* const*>(d_ptrArray_);
  for (int item = 0; item < topology.remoteDirectionCount; ++item) {
    const int direction = topology.remoteDirections[item];
    const HaloFacePlan& face = geometry.faces[direction];
    launchPack2D(d_haloBuf_send_[direction], d_ptrs, nFields, face.sendBase,
                 face.outerStride, face.innerStride, face.outerCount,
                 face.innerCount, stream);
  }
  cudaErrChk(cudaGetLastError());
}

void EMfields3D::gpuCompleteHaloPhases(const HaloTopologyPlan& topology,
                                       const HaloGeometryPlan& geometry,
                                       int nFields, bool faceOnly,
                                       cudaStream_t stream) {
  const int* neighbors = topology.neighbors;
  const MPI_Comm comm = topology.comm;
  const int nx = geometry.nx;
  const int ny = geometry.ny;
  const int nz = geometry.nz;
  const int offset = geometry.offset;
  auto* const* d_ptrs = reinterpret_cast<cudaSolverType* const*>(d_ptrArray_);
  MPI_Status statuses[12];
  MPI_Request requests[12];

  int requestCount = 0;
  for (int item = 0; item < topology.remoteDirectionCount; ++item) {
    const int direction = topology.remoteDirections[item];
    haloIrecv(d_haloBuf_recv_[direction],
              geometry.faces[direction].elements * nFields,
              mpiTypeOf<cudaSolverType>(), neighbors[direction],
              HALO_RECV_TAG[direction], comm, &requests[requestCount++]);
  }
  for (int item = 0; item < topology.remoteDirectionCount; ++item) {
    const int direction = topology.remoteDirections[item];
    haloIsend(d_haloBuf_send_[direction],
              geometry.faces[direction].elements * nFields,
              mpiTypeOf<cudaSolverType>(), neighbors[direction],
              HALO_SEND_TAG[direction], comm, &requests[requestCount++]);
  }

  constexpr int faceBlock = 16;
  const dim3 block(faceBlock, faceBlock);
  if (topology.selfAxis[0]) {
    const dim3 grid(geometry.selfFaceBlocks[0][0],
                    geometry.selfFaceBlocks[0][1], nFields);
    gpuBatchSelfCopyFaceX<<<grid, block, 0, stream>>>(d_ptrs, nx, ny, nz,
                                                      offset);
  }
  if (topology.selfAxis[1]) {
    const dim3 grid(geometry.selfFaceBlocks[1][0],
                    geometry.selfFaceBlocks[1][1], nFields);
    gpuBatchSelfCopyFaceY<<<grid, block, 0, stream>>>(d_ptrs, nx, ny, nz,
                                                      offset);
  }
  if (topology.selfAxis[2]) {
    const dim3 grid(geometry.selfFaceBlocks[2][0],
                    geometry.selfFaceBlocks[2][1], nFields);
    gpuBatchSelfCopyFaceZ<<<grid, block, 0, stream>>>(d_ptrs, nx, ny, nz,
                                                      offset);
  }

  if (requestCount > 0)
    haloWaitall(requestCount, requests, statuses, comm);
  for (int item = 0; item < topology.remoteDirectionCount; ++item) {
    const int direction = topology.remoteDirections[item];
    const HaloFacePlan& face = geometry.faces[direction];
    launchUnpack2D(d_haloBuf_recv_[direction], d_ptrs, nFields, face.recvBase,
                   face.outerStride, face.innerStride, face.outerCount,
                   face.innerCount, stream);
  }

  if (faceOnly)
    return;

  for (int item = 0; item < topology.edgeDirectionCount; ++item) {
    const int direction = topology.edgeDirections[item];
    int bufferOffset = 0;
    for (int edgeItem = 0; edgeItem < topology.edgeMultiplicity[direction];
         ++edgeItem) {
      const int segment = topology.edgeSegments[direction][edgeItem];
      const HaloEdgePlan& edge = geometry.edges[direction][segment];
      launchPack2D(d_haloBuf_send_[direction] + bufferOffset, d_ptrs, nFields,
                   edge.sendBase, edge.stride, 1, edge.elements, 1, stream);
      bufferOffset += edge.elements * nFields;
    }
  }
  cudaErrChk(cudaGetLastError());
  if (topology.edgeDirectionCount > 0)
    cudaErrChk(cudaStreamSynchronize(stream));

  requestCount = 0;
  for (int item = 0; item < topology.edgeDirectionCount; ++item) {
    const int direction = topology.edgeDirections[item];
    const int count = topology.edgeMultiplicity[direction] *
                      geometry.edgeLength[direction / 2] * nFields;
    haloIrecv(d_haloBuf_recv_[direction], count, mpiTypeOf<cudaSolverType>(),
              neighbors[direction], HALO_RECV_TAG[direction], comm,
              &requests[requestCount++]);
  }
  for (int item = 0; item < topology.edgeDirectionCount; ++item) {
    const int direction = topology.edgeDirections[item];
    const int count = topology.edgeMultiplicity[direction] *
                      geometry.edgeLength[direction / 2] * nFields;
    haloIsend(d_haloBuf_send_[direction], count, mpiTypeOf<cudaSolverType>(),
              neighbors[direction], HALO_SEND_TAG[direction], comm,
              &requests[requestCount++]);
  }

  const int edgeBlocks = geometry.edgeKernelBlocks;
  if (topology.selfAxis[0]) {
    const dim3 grid(edgeBlocks, nFields);
    gpuBatchSelfCopyEdgeX<<<grid, 256, 0, stream>>>(
        d_ptrs, nx, ny, nz, offset, topology.present[5], topology.present[4],
        topology.present[3], topology.present[2]);
  }
  if (topology.selfAxis[1]) {
    const dim3 grid(edgeBlocks, nFields);
    gpuBatchSelfCopyEdgeY<<<grid, 256, 0, stream>>>(
        d_ptrs, nx, ny, nz, offset, topology.present[1], topology.present[0],
        topology.present[5], topology.present[4]);
  }
  if (topology.selfAxis[2]) {
    const dim3 grid(edgeBlocks, nFields);
    gpuBatchSelfCopyEdgeZ<<<grid, 256, 0, stream>>>(
        d_ptrs, nx, ny, nz, offset, topology.present[3], topology.present[2],
        topology.present[1], topology.present[0]);
  }

  if (requestCount > 0)
    haloWaitall(requestCount, requests, statuses, comm);
  for (int item = 0; item < topology.edgeDirectionCount; ++item) {
    const int direction = topology.edgeDirections[item];
    int bufferOffset = 0;
    for (int edgeItem = 0; edgeItem < topology.edgeMultiplicity[direction];
         ++edgeItem) {
      const int segment = topology.edgeSegments[direction][edgeItem];
      const HaloEdgePlan& edge = geometry.edges[direction][segment];
      launchUnpack2D(d_haloBuf_recv_[direction] + bufferOffset, d_ptrs, nFields,
                     edge.recvBase, edge.stride, 1, edge.elements, 1, stream);
      bufferOffset += edge.elements * nFields;
    }
  }

  requestCount = 0;
  if (topology.hasRemoteYZ) {
    for (int item = 0; item < topology.cornerDirectionCount; ++item) {
      const int direction = topology.cornerDirections[item];
      const int total = 4 * nFields;
      k_batchPackCorners4<<<haloBatchBlocks(total), BATCH_BLK, 0, stream>>>(
          d_haloBuf_send_[direction], d_ptrs, nFields,
          geometry.cornerSendBase[direction], ny, nz);
    }
    cudaErrChk(cudaGetLastError());
    if (topology.cornerDirectionCount > 0)
      cudaErrChk(cudaStreamSynchronize(stream));

    for (int item = 0; item < topology.cornerDirectionCount; ++item) {
      const int direction = topology.cornerDirections[item];
      haloIrecv(d_haloBuf_recv_[direction], 4 * nFields,
                mpiTypeOf<cudaSolverType>(), neighbors[direction],
                HALO_RECV_TAG[direction], comm, &requests[requestCount++]);
    }
    for (int item = 0; item < topology.cornerDirectionCount; ++item) {
      const int direction = topology.cornerDirections[item];
      haloIsend(d_haloBuf_send_[direction], 4 * nFields,
                mpiTypeOf<cudaSolverType>(), neighbors[direction],
                HALO_SEND_TAG[direction], comm, &requests[requestCount++]);
    }
  }

  if (topology.selfAxis[0])
    gpuBatchSelfCopyCornerX<<<nFields, 1, 0, stream>>>(
        d_ptrs, nx, ny, nz, offset, topology.present[2], topology.present[3],
        topology.present[4], topology.present[5]);
  else if (topology.selfAxis[1])
    gpuBatchSelfCopyCornerY<<<nFields, 1, 0, stream>>>(
        d_ptrs, nx, ny, nz, offset, topology.present[0], topology.present[1],
        topology.present[4], topology.present[5]);
  else if (topology.selfAxis[2])
    gpuBatchSelfCopyCornerZ<<<nFields, 1, 0, stream>>>(
        d_ptrs, nx, ny, nz, offset, topology.present[2], topology.present[3],
        topology.present[0], topology.present[1]);

  if (requestCount > 0)
    haloWaitall(requestCount, requests, statuses, comm);
  if (topology.hasRemoteYZ) {
    for (int item = 0; item < topology.cornerDirectionCount; ++item) {
      const int direction = topology.cornerDirections[item];
      const int total = 4 * nFields;
      k_batchUnpackCorners4<<<haloBatchBlocks(total), BATCH_BLK, 0, stream>>>(
          d_haloBuf_recv_[direction], d_ptrs, nFields,
          geometry.cornerRecvBase[direction], ny, nz);
    }
  }
}

void EMfields3D::gpuBatchedHaloExchange(cudaSolverType** h_fieldPtrs,
                                        int nFields, int nx, int ny, int nz,
                                        bool offsetZero, bool isFaceOnlyFlag,
                                        bool needInterp, bool isParticle,
                                        cudaStream_t stream) {
  if (!h_fieldPtrs)
    batchedHaloContractFailure("null blocking-exchange field table");
  if (!haloBufsAllocated_)
    batchedHaloContractFailure("blocking halo buffers are not initialized");
  if (nFields <= 0 || nFields > HALO_MAX_BATCH)
    batchedHaloContractFailure("invalid blocking-exchange field count");
  const HaloGeometryPlan& geometry =
      gpuSelectHaloGeometry(nx, ny, nz, offsetZero);
  const HaloTopologyPlan& topology =
      haloTopologyPlans_[isParticle ? HALO_PARTICLE : HALO_FIELD];
#ifdef HALO_OVERLAP
  if (haloExchange_.active)
    batchedHaloContractFailure("overlapping use of shared GPU halo buffers");
  // This also orders a blocking exchange on an arbitrary CUDA stream after
  // the final buffer consumer of the preceding blocking or split exchange.
  if (haloBuffersReady_)
    cudaErrChk(cudaStreamWaitEvent(stream, haloBuffersReady_, 0));
#endif
  gpuQueueHaloFacePack(h_fieldPtrs, nFields, topology, geometry, stream);
  // CUDA-aware MPI may consume the send buffers only after face packing.
  // This also completes the pinned pointer-table upload before reuse.
  cudaErrChk(cudaStreamSynchronize(stream));
  gpuCompleteHaloPhases(topology, geometry, nFields, isFaceOnlyFlag, stream);

  auto* const* d_ptrs = reinterpret_cast<cudaSolverType* const*>(d_ptrArray_);

  // =====================================================================
  //  Additive interpolation  (for moments: ghost → interior boundary)
  // =====================================================================
  if (needInterp) {
    // Unpack and additive kernels share the same stream — CUDA
    // stream ordering guarantees all ghost data is visible.

    // Additive kernels use neighbor presence from the topology selected for
    // this exchange.
    const bool hasXR = topology.present[1];
    const bool hasXL = topology.present[0];
    const bool hasYR = topology.present[3];
    const bool hasYL = topology.present[2];
    const bool hasZR = topology.present[5];
    const bool hasZL = topology.present[4];

    const int nxr = geometry.active[0];
    const int nyr = geometry.active[1];
    const int nzr = geometry.active[2];

    // Face add
    {
      int total = nyr * nzr * nFields;
      k_batchAddFaceX<<<haloBatchBlocks(total), BATCH_BLK, 0, stream>>>(
          d_ptrs, nFields, nx, ny, nz, hasXR, hasXL);
    }
    {
      int total = nxr * nzr * nFields;
      k_batchAddFaceY<<<haloBatchBlocks(total), BATCH_BLK, 0, stream>>>(
          d_ptrs, nFields, nx, ny, nz, hasYR, hasYL);
    }
    {
      int total = nxr * nyr * nFields;
      k_batchAddFaceZ<<<haloBatchBlocks(total), BATCH_BLK, 0, stream>>>(
          d_ptrs, nFields, nx, ny, nz, hasZR, hasZL);
    }

    // Edge add
    {
      int total = nzr * nFields;
      k_batchAddEdgeZ<<<haloBatchBlocks(total), BATCH_BLK, 0, stream>>>(
          d_ptrs, nFields, nx, ny, nz, hasXR, hasXL, hasYR, hasYL);
    }
    {
      int total = nyr * nFields;
      k_batchAddEdgeY<<<haloBatchBlocks(total), BATCH_BLK, 0, stream>>>(
          d_ptrs, nFields, nx, ny, nz, hasXR, hasXL, hasZR, hasZL);
    }
    {
      int total = nxr * nFields;
      k_batchAddEdgeX<<<haloBatchBlocks(total), BATCH_BLK, 0, stream>>>(
          d_ptrs, nFields, nx, ny, nz, hasYR, hasYL, hasZR, hasZL);
    }

    // Corner add
    k_batchAddCorner<<<haloBatchBlocks(nFields), BATCH_BLK, 0, stream>>>(
        d_ptrs, nFields, nx, ny, nz, hasXR, hasXL, hasYR, hasYL, hasZR, hasZL);
  }

  cudaErrChk(cudaGetLastError());

#ifdef HALO_OVERLAP
  if (haloBuffersReady_)
    cudaErrChk(cudaEventRecord(haloBuffersReady_, stream));
#endif
}

// =========================================================================
//  HALO_OVERLAP: asynchronous face packing plus shared halo completion
// =========================================================================
#ifdef HALO_OVERLAP

void EMfields3D::gpuBatchedHaloBeginExchange(cudaSolverType** h_fieldPtrs,
                                             int nFields, int nx, int ny,
                                             int nz, bool offsetZero,
                                             bool isFaceOnlyFlag,
                                             bool isParticle,
                                             cudaStream_t computeStream) {
  if (!h_fieldPtrs)
    batchedHaloContractFailure("null split-exchange field table");
  if (haloExchange_.active)
    batchedHaloContractFailure("overlapping use of shared GPU halo buffers");
  if (nFields <= 0 || nFields > HALO_MAX_BATCH)
    batchedHaloContractFailure("invalid split halo field count");
  if (!haloBufsAllocated_)
    batchedHaloContractFailure("split halo buffers are not initialized");
  if (!haloStream_ || !haloInputReady_ || !haloPhaseReady_ ||
      !haloBuffersReady_)
    batchedHaloContractFailure("split halo stream/events are not initialized");

  const HaloGeometryPlan& geometry =
      gpuSelectHaloGeometry(nx, ny, nz, offsetZero);
  const HaloTopologyPlan& topology =
      haloTopologyPlans_[isParticle ? HALO_PARTICLE : HALO_FIELD];
  haloExchange_.active = true;
  haloExchange_.topology = &topology;
  haloExchange_.geometry = &geometry;
  haloExchange_.nFields = nFields;
  haloExchange_.faceOnly = isFaceOnlyFlag;
  haloExchange_.computeStream = computeStream;

  // Capture the producer tail before the caller queues dependency-free work.
  cudaErrChk(cudaEventRecord(haloInputReady_, computeStream));
  cudaErrChk(cudaStreamWaitEvent(haloStream_, haloInputReady_, 0));
  cudaErrChk(cudaStreamWaitEvent(haloStream_, haloBuffersReady_, 0));
  gpuQueueHaloFacePack(h_fieldPtrs, nFields, topology, geometry, haloStream_);

  // End waits at this exact point before CUDA-aware MPI reads send buffers.
  cudaErrChk(cudaEventRecord(haloPhaseReady_, haloStream_));
}

void EMfields3D::gpuBatchedHaloEndExchange() {
  if (!haloExchange_.active)
    batchedHaloContractFailure("EndExchange without a matching BeginExchange");

  const HaloTopologyPlan& topology = *haloExchange_.topology;
  const HaloGeometryPlan& geometry = *haloExchange_.geometry;
  const int nFields = haloExchange_.nFields;
  const bool faceOnly = haloExchange_.faceOnly;
  const cudaStream_t computeStream = haloExchange_.computeStream;

  // This fence also protects the pinned pointer staging table when a topology
  // has no remote faces.
  cudaErrChk(cudaEventSynchronize(haloPhaseReady_));
  gpuCompleteHaloPhases(topology, geometry, nFields, faceOnly, haloStream_);
  cudaErrChk(cudaGetLastError());
  cudaErrChk(cudaEventRecord(haloBuffersReady_, haloStream_));
  cudaErrChk(cudaStreamWaitEvent(computeStream, haloBuffersReady_, 0));
  haloExchange_ = HaloExchangeState{};
}

#endif // HALO_OVERLAP
#endif // GPU_SOLVER
