/**
 * @file GPUBatchedHaloComm.cu
 * @brief Batched GPU halo exchange: the 26 neighbour directions in one round.
 *
 * Every ghost box (6 faces, 12 edges, 8 corners) comes straight from the
 * neighbour across it.  The boxes for one neighbour rank travel as one
 * contiguous message for all batched fields; the receives are posted before
 * packing, then all sends, and one MPI_Waitall completes the exchange.
 * Along an axis that is periodic on one process the ghost boxes are copies
 * within the array.  The plans (HaloPlan) are built once, when the halo
 * buffers are allocated; an exchange selects one, and one pack, one unpack
 * and one self-copy launch move the boxes of all its fields.
 */

#include "EMfields3D.h"
#include "GPUHaloComm.cuh" // HaloPlan
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

static constexpr int HALO_BLOCK = 256; // threads per block of every launch

static inline int haloBlocks(int threads) {
  return 1 + (threads - 1) / HALO_BLOCK;
}

// =========================================================================
//  HALO BOX COPY: one launch packs (or unpacks) every box of every field
// =========================================================================

/** PACK: the send boxes into the send buffer, and the self copies; otherwise
 *  the receive buffer into the ghost boxes.  A block reads one box, the last
 *  one starting at or before blockIdx.x; blockIdx.y selects the field. */
template <bool PACK>
__global__ void
k_haloCopy(const HaloCopyTable t, cudaSolverType* __restrict__ buf,
           cudaSolverType* const* __restrict__ fields, int nFields) {
  int lo = 0, hi = t.boxes - 1;
  while (lo < hi) {
    const int mid = (lo + hi + 1) >> 1;
    if (t.box[mid].block <= (int)blockIdx.x)
      lo = mid;
    else
      hi = mid - 1;
  }
  const HaloBox& b = t.box[lo];
  const int e = ((int)blockIdx.x - b.block) * HALO_BLOCK + (int)threadIdx.x;
  if (e >= b.points)
    return;
  const int line = e / b.lk; // (i, j) line of the box, k = e - line lk
  const int i = line / b.lj;
  const int at = (i * t.ny + line - i * b.lj) * t.nz + e - line * b.lk;
  cudaSolverType* const f = fields[blockIdx.y];
  if constexpr (PACK) {
    if (b.self >= 0) {
      f[b.self + at] = f[b.field + at];
      return;
    }
  }
  const int m = b.message * nFields + (int)blockIdx.y * b.length + b.offset + e;
  if constexpr (PACK)
    buf[m] = f[b.field + at];
  else
    f[b.field + at] = buf[m];
}

template <bool PACK>
static void launchHaloCopy(const HaloCopyTable& t, cudaSolverType* buf,
                           cudaSolverType* const* fields, int nFields,
                           cudaStream_t stream) {
  if (t.boxes == 0)
    return;
  k_haloCopy<PACK><<<dim3(t.blocks, nFields), HALO_BLOCK, 0, stream>>>(
      t, buf, fields, nFields);
}

/** The device pointers of the batched fields, passed by value: the launch
 *  copies them, so the host table is free at once. */
struct HaloFieldPointers {
  cudaSolverType* p[64];
};

/** One thread per field. */
__global__ void k_haloSetPointers(const HaloFieldPointers h,
                                  cudaSolverType** __restrict__ table) {
  table[threadIdx.x] = h.p[threadIdx.x];
}

// =========================================================================
//  BATCHED PACK / UNPACK KERNELS (not used: k_haloCopy moves all boxes)
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
//  CORNER PACK / UNPACK  (4 scattered elements per field per direction; not
//  used)
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

/** Add the ghosts of the faces, edges and corners with a neighbour (present:
 *  XL XR YL YR ZL ZR) into the nodes they share with it. */
static void launchHaloAdd(const bool present[6], cudaSolverType* const* fields,
                          int nFields, int nx, int ny, int nz,
                          cudaStream_t stream) {
  const bool xl = present[0], xr = present[1], yl = present[2], yr = present[3],
             zl = present[4], zr = present[5];
  const int nxr = nx - 2, nyr = ny - 2, nzr = nz - 2;
  k_batchAddFaceX<<<haloBlocks(nyr * nzr * nFields), HALO_BLOCK, 0, stream>>>(
      fields, nFields, nx, ny, nz, xr, xl);
  k_batchAddFaceY<<<haloBlocks(nxr * nzr * nFields), HALO_BLOCK, 0, stream>>>(
      fields, nFields, nx, ny, nz, yr, yl);
  k_batchAddFaceZ<<<haloBlocks(nxr * nyr * nFields), HALO_BLOCK, 0, stream>>>(
      fields, nFields, nx, ny, nz, zr, zl);
  k_batchAddEdgeZ<<<haloBlocks(nzr * nFields), HALO_BLOCK, 0, stream>>>(
      fields, nFields, nx, ny, nz, xr, xl, yr, yl);
  k_batchAddEdgeY<<<haloBlocks(nyr * nFields), HALO_BLOCK, 0, stream>>>(
      fields, nFields, nx, ny, nz, xr, xl, zr, zl);
  k_batchAddEdgeX<<<haloBlocks(nxr * nFields), HALO_BLOCK, 0, stream>>>(
      fields, nFields, nx, ny, nz, yr, yl, zr, zl);
  k_batchAddCorner<<<haloBlocks(nFields), HALO_BLOCK, 0, stream>>>(
      fields, nFields, nx, ny, nz, xr, xl, yr, yl, zr, zl);
  cudaErrChk(cudaGetLastError());
}

// =========================================================================
//  EXCHANGE PLANS
//
//  Direction d = 9 (dx + 1) + 3 (dy + 1) + (dz + 1) holds the neighbour at
//  offset (dx, dy, dz) in the process grid, 13 = HALO_SELF this rank, and
//  26 - d is the opposite direction.  Along an axis with step s, the send box
//  of d is the point 1 + offset (s = -1), the points 1..n-2 (s = 0) or the
//  point n-2-offset (s = +1), and its ghost box the point 0, the points
//  1..n-2 or the point n-1.  The ghost boxes tile the ghost shell, and every
//  send box lies in the points 1..n-2, so the sends read no ghost and all
//  boxes travel in one round.  Along an axis with s = 0 the two ranks share
//  the coordinate, hence the extent n-2, also when the cells do not divide
//  evenly among the processes.
// =========================================================================

static constexpr int HALO_SELF = 13;
static constexpr int HALO_STRIDE[3] = {9, 3, 1}; // of a step along x, y, z
// XL XR YL YR ZL ZR, the order of HaloPlan::present
static constexpr int HALO_FACE_DIRECTION[6] = {4, 22, 10, 16, 12, 14};
// One tag: an exchange sends one message per neighbour pair and completes
// before the next one starts, and MPI keeps the order of the messages
// between two ranks
static constexpr int HALO_TAG = 2626;
static constexpr bool HALO_SEND[3] = {true, true, true};
static constexpr bool HALO_GHOST[3] = {false, false, false};

static inline int haloStep(int d, int axis) {
  return d / HALO_STRIDE[axis] % 3 - 1;
}

static inline bool haloFaceDirection(int d) {
  return std::abs(haloStep(d, 0)) + std::abs(haloStep(d, 1)) +
             std::abs(haloStep(d, 2)) ==
         1;
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

/** The neighbour rank of every direction on a Cartesian communicator:
 *  periodic axes wrap, a step beyond a non-periodic edge is MPI_PROC_NULL. */
static void haloNeighbors(MPI_Comm comm, int neighbor[27]) {
  int dims[3], periods[3], coords[3];
  int error = MPI_Cart_get(comm, 3, dims, periods, coords);
  if (error != MPI_SUCCESS)
    batchedHaloMpiFailure(error, comm, "MPI_Cart_get");
  for (int d = 0; d < 27; ++d) {
    int c[3];
    bool outside = false;
    for (int a = 0; a < 3; ++a) {
      c[a] = coords[a] + haloStep(d, a);
      if (c[a] < 0 || c[a] >= dims[a]) {
        if (periods[a])
          c[a] = (c[a] + dims[a]) % dims[a];
        else
          outside = true;
      }
    }
    neighbor[d] = MPI_PROC_NULL;
    if (!outside) {
      error = MPI_Cart_rank(comm, c, &neighbor[d]);
      if (error != MPI_SUCCESS)
        batchedHaloMpiFailure(error, comm, "MPI_Cart_rank");
    }
  }
}

/** Append a box of extents ext from the array index origin to t. */
static HaloBox& haloAddBox(HaloCopyTable& t, const int ext[3], int origin) {
  if (t.boxes == 26)
    batchedHaloContractFailure("more than 26 halo boxes");
  HaloBox& b = t.box[t.boxes++];
  b = HaloBox{};
  b.block = t.blocks;
  b.points = ext[0] * ext[1] * ext[2];
  b.lj = ext[1];
  b.lk = ext[2];
  b.field = origin;
  b.self = -1;
  t.blocks += haloBlocks(b.points);
  return b;
}

/** Builds the plan of one communicator, array geometry and extent. */
class HaloPlanBuilder {
public:
  HaloPlanBuilder(const int neighbor[27], int rank, const int n[3], int offset,
                  bool faceOnly)
      : neighbor_(neighbor), offset_(offset), faceOnly_(faceOnly) {
    const int minExtent = offset == 0 ? 3 : 4;
    for (int a = 0; a < 3; ++a) {
      if (n[a] < minExtent)
        batchedHaloContractFailure("invalid canonical halo geometry");
      n_[a] = n[a];
      // periodic on one process: this rank on both sides
      selfAxis_[a] = neighbor[HALO_SELF - HALO_STRIDE[a]] == rank &&
                     neighbor[HALO_SELF + HALO_STRIDE[a]] == rank;
    }
    if ((size_t)n[0] * n[1] * n[2] > INT_MAX)
      batchedHaloContractFailure("halo array exceeds INT_MAX points");
  }

  HaloPlan build(MPI_Comm comm) const {
    HaloPlan plan;
    plan.comm = comm;
    for (int f = 0; f < 6; ++f)
      plan.present[f] = neighbor_[HALO_FACE_DIRECTION[f]] != MPI_PROC_NULL;
    for (HaloCopyTable* t : {&plan.pack, &plan.unpack, &plan.fill}) {
      t->ny = n_[1];
      t->nz = n_[2];
    }
    // one message per distinct remote neighbour, in the order of its first
    // direction
    for (int d = 0; d < 27; ++d) {
      if (!sent(d) || hasPeer(plan, neighbor_[d]))
        continue;
      HaloPeer& peer = plan.peer[plan.peers++];
      peer.rank = neighbor_[d];
      peer.offset = plan.bufferPoints;
      peer.length = addMessage(plan.pack, peer, true);
      addMessage(plan.unpack, peer, false); // the same boxes and length
      if ((size_t)plan.bufferPoints + peer.length > INT_MAX)
        batchedHaloContractFailure("halo buffer exceeds INT_MAX");
      plan.bufferPoints += peer.length;
    }
    // A box with a step along a self axis is a copy within the array: along
    // the self axes from the send point of the opposite step, along the
    // other axes from the same ghost box.  With steps along self axes only
    // it reads interior points (pack launch); otherwise it reads the ghost
    // box of the direction without the self steps, which the unpack writes
    // (fill launch, after it).  No copy reads a ghost box that another copy
    // of its launch writes.
    for (int d = 0; d < 27; ++d) {
      if (!exchanged(d) || !stepsAlong(d, true))
        continue;
      int ext[3];
      const int ghost = boxOrigin(d, HALO_GHOST, ext);
      const int source = boxOrigin(selfMirror(d), selfAxis_, ext);
      haloAddBox(stepsAlong(d, false) ? plan.fill : plan.pack, ext, source)
          .self = ghost;
    }
    return plan;
  }

private:
  const int* neighbor_; // [27], by direction
  int n_[3];
  int offset_;
  bool faceOnly_;
  bool selfAxis_[3];

  /** d steps along a self axis (self) or along another axis (!self). */
  bool stepsAlong(int d, bool self) const {
    for (int a = 0; a < 3; ++a)
      if (selfAxis_[a] == self && haloStep(d, a) != 0)
        return true;
    return false;
  }
  /** The ghost box of d has a neighbour and belongs to the exchange. */
  bool exchanged(int d) const {
    return d != HALO_SELF && neighbor_[d] != MPI_PROC_NULL &&
           (!faceOnly_ || haloFaceDirection(d));
  }
  /** It travels over MPI: with no step along a self axis the neighbour is
   *  remote. */
  bool sent(int d) const { return exchanged(d) && !stepsAlong(d, true); }

  /** d with its steps along the self axes reversed. */
  int selfMirror(int d) const {
    for (int a = 0; a < 3; ++a)
      if (selfAxis_[a])
        d -= 2 * haloStep(d, a) * HALO_STRIDE[a];
    return d;
  }

  /** The box of d: along an axis with step -1 (+1) the send point 1 + offset
   *  (n-2-offset) where send[a], else the ghost point 0 (n-1); with step 0
   *  the points 1..n-2.  Sets its extents and returns its origin. */
  int boxOrigin(int d, const bool send[3], int ext[3]) const {
    int o[3];
    for (int a = 0; a < 3; ++a) {
      const int s = haloStep(d, a);
      ext[a] = s == 0 ? n_[a] - 2 : 1;
      o[a] = s == 0  ? 1
             : s < 0 ? (send[a] ? 1 + offset_ : 0)
                     : (send[a] ? n_[a] - 2 - offset_ : n_[a] - 1);
    }
    return (o[0] * n_[1] + o[1]) * n_[2] + o[2];
  }

  static bool hasPeer(const HaloPlan& plan, int rank) {
    for (int p = 0; p < plan.peers; ++p)
      if (plan.peer[p].rank == rank)
        return true;
    return false;
  }

  /** Append the boxes of the message of peer to t, the send boxes (send) or
   *  the ghost boxes, at consecutive offsets; returns its length.  Both
   *  sides list the boxes in ascending direction of the sender: d on the
   *  send side, 26 - d on the receive side. */
  int addMessage(HaloCopyTable& t, const HaloPeer& peer, bool send) const {
    const int first = t.boxes;
    int length = 0;
    for (int i = 0; i < 27; ++i) {
      const int d = send ? i : 26 - i;
      if (!sent(d) || neighbor_[d] != peer.rank)
        continue;
      int ext[3];
      const int origin = boxOrigin(d, send ? HALO_SEND : HALO_GHOST, ext);
      HaloBox& b = haloAddBox(t, ext, origin);
      b.message = peer.offset;
      b.offset = length;
      length += b.points;
    }
    for (int b = first; b < t.boxes; ++b)
      t.box[b].length = length;
    return length;
  }
};

void EMfields3D::gpuInitializeHaloPlans() {
  const VirtualTopology3D& vct = get_vct();
  // extents and send offsets of HALO_CENTER_OFFSET0, _NODE_OFFSET1, _OFFSET0
  const int extents[3][3] = {{nxc, nyc, nzc}, {nxn, nyn, nzn}, {nxn, nyn, nzn}};
  const int offsets[3] = {0, 1, 0};
  for (int kind = HALO_FIELD; kind <= HALO_PARTICLE; ++kind) {
    const bool particle = kind == HALO_PARTICLE;
    const MPI_Comm comm = particle ? vct.getParticleComm() : vct.getFieldComm();
    if (comm == MPI_COMM_NULL)
      batchedHaloContractFailure("halo topology is not initialized");
    int rank = MPI_PROC_NULL;
    const int error = MPI_Comm_rank(comm, &rank);
    if (error != MPI_SUCCESS)
      batchedHaloMpiFailure(error, comm, "MPI_Comm_rank");
    int neighbor[27];
    haloNeighbors(comm, neighbor);
    const int faceNeighbor[6] = {
        particle ? vct.getXleft_neighbor_P() : vct.getXleft_neighbor(),
        particle ? vct.getXright_neighbor_P() : vct.getXright_neighbor(),
        particle ? vct.getYleft_neighbor_P() : vct.getYleft_neighbor(),
        particle ? vct.getYright_neighbor_P() : vct.getYright_neighbor(),
        particle ? vct.getZleft_neighbor_P() : vct.getZleft_neighbor(),
        particle ? vct.getZright_neighbor_P() : vct.getZright_neighbor()};
    for (int f = 0; f < 6; ++f)
      if (neighbor[HALO_FACE_DIRECTION[f]] != faceNeighbor[f])
        batchedHaloContractFailure("halo neighbours disagree with topology");
    for (int g = 0; g < 3; ++g)
      for (int faceOnly = 0; faceOnly < 2; ++faceOnly)
        haloPlans_[kind][g][faceOnly] =
            HaloPlanBuilder(neighbor, rank, extents[g], offsets[g],
                            faceOnly == 1)
                .build(comm);
  }
}

const HaloPlan& EMfields3D::gpuSelectHaloPlan(cudaSolverType** h_fieldPtrs,
                                              int nFields, int nx, int ny,
                                              int nz, bool offsetZero,
                                              bool faceOnly,
                                              bool isParticle) const {
  if (!haloBufsAllocated_)
    batchedHaloContractFailure("halo buffers and plans are not initialized");
  if (haloExchange_.active)
    batchedHaloContractFailure("overlapping use of shared GPU halo buffers");
  if (!h_fieldPtrs)
    batchedHaloContractFailure("null halo field table");
  if (nFields <= 0 || nFields > HALO_MAX_BATCH)
    batchedHaloContractFailure("invalid halo field count");
  int geometry;
  if (nx == nxc && ny == nyc && nz == nzc && offsetZero)
    geometry = HALO_CENTER_OFFSET0;
  else if (nx == nxn && ny == nyn && nz == nzn)
    geometry = offsetZero ? HALO_NODE_OFFSET0 : HALO_NODE_OFFSET1;
  else
    batchedHaloContractFailure(
        "exchange does not match a cached halo geometry");
  return haloPlans_[isParticle ? HALO_PARTICLE : HALO_FIELD][geometry]
                   [faceOnly ? 1 : 0];
}

// =========================================================================
//  ONE EXCHANGE ROUND: receives, pack, sends, one MPI_Waitall, unpack
// =========================================================================

cudaSolverType* EMfields3D::gpuHaloPostReceives(const HaloPlan& plan,
                                                int nFields,
                                                MPI_Request* requests) {
  if (plan.peers == 0)
    return nullptr;
  cudaSolverType* const recv = d_haloRecv_[haloRecvSlot_];
  haloRecvSlot_ ^= 1;
  for (int p = 0; p < plan.peers; ++p) {
    const HaloPeer& peer = plan.peer[p];
    const int error =
        MPI_Irecv(recv + (size_t)peer.offset * nFields, peer.length * nFields,
                  mpiTypeOf<cudaSolverType>(), peer.rank, HALO_TAG, plan.comm,
                  &requests[p]);
    if (error != MPI_SUCCESS)
      batchedHaloMpiFailure(error, plan.comm, "MPI_Irecv");
  }
  return recv;
}

void EMfields3D::gpuHaloPack(const HaloPlan& plan, cudaSolverType** h_fieldPtrs,
                             int nFields, cudaStream_t stream) {
  static_assert(HALO_MAX_BATCH <= 64, "HaloFieldPointers holds 64 pointers");
  HaloFieldPointers h;
  std::memcpy(h.p, h_fieldPtrs, nFields * sizeof(cudaSolverType*));
  k_haloSetPointers<<<1, nFields, 0, stream>>>(h, d_ptrArray_);
  launchHaloCopy<true>(plan.pack, d_haloSend_, d_ptrArray_, nFields, stream);
  cudaErrChk(cudaGetLastError());
}

void EMfields3D::gpuHaloSendAndWait(const HaloPlan& plan, int nFields,
                                    MPI_Request* requests) {
  if (plan.peers == 0)
    return;
  for (int p = 0; p < plan.peers; ++p) {
    const HaloPeer& peer = plan.peer[p];
    const int error =
        MPI_Isend(d_haloSend_ + (size_t)peer.offset * nFields,
                  peer.length * nFields, mpiTypeOf<cudaSolverType>(), peer.rank,
                  HALO_TAG, plan.comm, &requests[plan.peers + p]);
    if (error != MPI_SUCCESS)
      batchedHaloMpiFailure(error, plan.comm, "MPI_Isend");
  }
  MPI_Status statuses[2 * 26];
  const int error = MPI_Waitall(2 * plan.peers, requests, statuses);
  if (error != MPI_SUCCESS)
    batchedHaloMpiFailure(error, plan.comm, "MPI_Waitall");
  // a longer message fails in MPI; a shorter one would leave stale ghosts
  for (int p = 0; p < plan.peers; ++p) {
    int count = -1;
    MPI_Get_count(&statuses[p], mpiTypeOf<cudaSolverType>(), &count);
    if (count != plan.peer[p].length * nFields)
      batchedHaloContractFailure("halo message shorter than its plan");
  }
}

void EMfields3D::gpuHaloUnpack(const HaloPlan& plan, cudaSolverType* recv,
                               int nFields, cudaStream_t stream) {
  launchHaloCopy<false>(plan.unpack, recv, d_ptrArray_, nFields, stream);
  // the fill holds self copies only, which the PACK variant runs
  launchHaloCopy<true>(plan.fill, nullptr, d_ptrArray_, nFields, stream);
  cudaErrChk(cudaGetLastError());
}

void EMfields3D::gpuBatchedHaloExchange(cudaSolverType** h_fieldPtrs,
                                        int nFields, int nx, int ny, int nz,
                                        bool offsetZero, bool isFaceOnlyFlag,
                                        bool needInterp, bool isParticle,
                                        cudaStream_t stream) {
  const HaloPlan& plan = gpuSelectHaloPlan(
      h_fieldPtrs, nFields, nx, ny, nz, offsetZero, isFaceOnlyFlag, isParticle);
  if (needInterp && (!offsetZero || isFaceOnlyFlag))
    batchedHaloContractFailure("additive exchange needs every box, offset 0");

  MPI_Request requests[2 * 26];
  cudaSolverType* const recv = gpuHaloPostReceives(plan, nFields, requests);
  // With the split exchange resources (HaloOverlap), this orders a blocking
  // exchange on an arbitrary CUDA stream after the final buffer consumer of
  // the preceding blocking or split exchange.
  if (haloBuffersReady_)
    cudaErrChk(cudaStreamWaitEvent(stream, haloBuffersReady_, 0));
  gpuHaloPack(plan, h_fieldPtrs, nFields, stream);
  // CUDA-aware MPI may read the send buffer only once packed; without a
  // remote neighbour the exchange stays asynchronous
  if (plan.peers > 0)
    cudaErrChk(cudaStreamSynchronize(stream));
  gpuHaloSendAndWait(plan, nFields, requests);
  gpuHaloUnpack(plan, recv, nFields, stream);
  // moments: the ghosts now hold the neighbours' values of the shared nodes
  if (needInterp)
    launchHaloAdd(plan.present, d_ptrArray_, nFields, nx, ny, nz, stream);

  if (haloBuffersReady_)
    cudaErrChk(cudaEventRecord(haloBuffersReady_, stream));
}

// =========================================================================
//  Split exchange (HaloOverlap): receives and packing in Begin, sends,
//  completion and unpacking in End
// =========================================================================

void EMfields3D::gpuBatchedHaloBeginExchange(cudaSolverType** h_fieldPtrs,
                                             int nFields, int nx, int ny,
                                             int nz, bool offsetZero,
                                             bool isFaceOnlyFlag,
                                             bool isParticle,
                                             cudaStream_t computeStream) {
  const HaloPlan& plan = gpuSelectHaloPlan(
      h_fieldPtrs, nFields, nx, ny, nz, offsetZero, isFaceOnlyFlag, isParticle);
  if (!haloStream_ || !haloInputReady_ || !haloPacked_ || !haloBuffersReady_)
    batchedHaloContractFailure("split halo stream/events are not initialized");

  haloExchange_.active = true;
  haloExchange_.plan = &plan;
  haloExchange_.nFields = nFields;
  haloExchange_.computeStream = computeStream;
  // receives first: a neighbour that packs sooner can deliver at once
  haloExchange_.recv =
      gpuHaloPostReceives(plan, nFields, haloExchange_.requests);

  // Capture the producer tail before the caller queues dependency-free work.
  cudaErrChk(cudaEventRecord(haloInputReady_, computeStream));
  cudaErrChk(cudaStreamWaitEvent(haloStream_, haloInputReady_, 0));
  cudaErrChk(cudaStreamWaitEvent(haloStream_, haloBuffersReady_, 0));
  gpuHaloPack(plan, h_fieldPtrs, nFields, haloStream_);
  // End waits at this exact point before CUDA-aware MPI reads send buffers.
  cudaErrChk(cudaEventRecord(haloPacked_, haloStream_));
}

void EMfields3D::gpuBatchedHaloEndExchange() {
  if (!haloExchange_.active)
    batchedHaloContractFailure("EndExchange without a matching BeginExchange");

  const HaloPlan& plan = *haloExchange_.plan;
  const int nFields = haloExchange_.nFields;
  if (plan.peers > 0)
    cudaErrChk(cudaEventSynchronize(haloPacked_));
  gpuHaloSendAndWait(plan, nFields, haloExchange_.requests);
  gpuHaloUnpack(plan, haloExchange_.recv, nFields, haloStream_);
  cudaErrChk(cudaEventRecord(haloBuffersReady_, haloStream_));
  cudaErrChk(
      cudaStreamWaitEvent(haloExchange_.computeStream, haloBuffersReady_, 0));
  haloExchange_ = HaloExchangeState{};
}
