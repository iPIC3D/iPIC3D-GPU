/**
 * @file GPUFieldBoundary.cu
 * @brief Kernels of the field boundary conditions (GPUFieldBoundary.cuh):
 *        ghost fill, boundary rows, layer operations, moment fold, the
 *        conductor wall values and the X-right outflow face.
 *
 * The face kernels are launch-bound (a face is a small fraction of a
 * subdomain), so each mechanism is one launch for all faces and all arrays
 * of a call.  Threads run along k on x and y faces (FaceSlabs), the
 * contiguous axis; on z faces the k-fastest layout makes the access strided.
 * No kernel indexes a kernel argument by a run-time value: per-array rules
 * are in device memory, and the slab tables are read through unrolled
 * selects.
 */

#include "GPUFieldBoundary.cuh"
#include "GPUPointLaunch.cuh"
#include "GPUStencils.cuh"

#include <algorithm>

using T = cudaSolverType;

namespace {

/** Threads per block of the face kernels: small blocks spread a plane over
 *  more SMs (a face-plane kernel measured as fast with 64 as with 2D
 *  blocks). */
constexpr int kFaceBlock = 64;

int blocks(int points) { return (points + kFaceBlock - 1) / kFaceBlock; }

/** Array a (blockIdx.y) of a batch: an unrolled select, so that no kernel
 *  argument is indexed by a run-time value (no stack frame). */
__device__ __forceinline__ T* batchArray(const GhostBatch& b, int a) {
  T* p = nullptr;
#pragma unroll
  for (int x = 0; x < kGhostBatchMax; x++)
    if (x == a)
      p = b.p[x];
  return p;
}

/** Point t of the slabs: the coordinates (i, j, k); false past the last
 *  slab. */
__device__ __forceinline__ bool slabPoint(const FaceSlabs& s, int t, int& i,
                                          int& j, int& k) {
  if (t >= s.start[kFaces])
    return false;
  int f = 0, plane = 0, slowLo = 0, fastLo = 0, fastN = 1, start = 0;
#pragma unroll
  for (int g = 0; g < kFaces; g++)
    if (t >= s.start[g] && t < s.start[g + 1]) {
      f = g;
      plane = s.plane[g];
      slowLo = s.slowLo[g];
      fastLo = s.fastLo[g];
      fastN = s.fastN[g];
      start = s.start[g];
    }
  const int q = t - start, qs = q / fastN, qf = q - qs * fastN;
  const int a = f / 2;
  i = a == 0 ? plane : slowLo + qs;
  j = a == 1 ? plane : (a == 0 ? slowLo + qs : fastLo + qf);
  k = a == 2 ? plane : fastLo + qf;
  return true;
}

// -------------------------------------------------------------------------
//  Ghost fill
// -------------------------------------------------------------------------

/** The rule of one ghost axis of a point. */
struct AxisRule {
  int axis = 0, inward = 1; // inward: +1 on a low face, -1 on a high face
  GhostRuleDev r;
};

/** Index of the point (i, j, k) of a ghost plane normal to axis e in the
 *  wall array of that face: slow * (extent of fast) + fast (cell extents). */
__device__ __forceinline__ int wallIndex(int e, int i, int j, int k, int ny,
                                         int nz) {
  const int slow = e == 0 ? j : i;
  const int fast = e == 2 ? j : k;
  return slow * (e == 2 ? ny : nz) + fast;
}

/** Rule R at (i, j, k), g(m) the value m layers inward (GhostRule). */
template <typename G>
__device__ __forceinline__ T applyRule(const AxisRule& R, int i, int j, int k,
                                       int ny, int nz, G&& g) {
  switch (R.r.kind) {
  case GHOST_ZERO:
    return 0.0;
  case GHOST_EVEN_CELL:
    return g(1);
  case GHOST_ODD_CELL:
    return -g(1);
  case GHOST_EVEN_NODE:
    return g(2);
  case GHOST_ODD_NODE:
    return 2.0 * g(1) - g(2);
  case GHOST_WALL:
    return R.r.wall[wallIndex(R.axis, i, j, k, ny, nz)] - g(1);
  case GHOST_CONST:
    return R.r.value;
  case GHOST_COPY:
    return g(R.r.copy);
  default:
    return g(0);
  }
}

/** The value at (i, j, k) after the rules r0..r(D-1), applied in that order
 *  (the last one outermost): the earlier rules give the values the later
 *  ones read. */
template <int D>
__device__ __forceinline__ T compose(const T* __restrict__ a, int i, int j,
                                     int k, const AxisRule& r0,
                                     const AxisRule& r1, const AxisRule& r2,
                                     int ny, int nz) {
  if constexpr (D == 0) {
    return a[stencilIdx(i, j, k, ny, nz)];
  } else {
    const AxisRule& R = D == 3 ? r2 : (D == 2 ? r1 : r0);
    const auto g = [&](int m) {
      const int d = R.inward * m;
      return compose<D - 1>(a, i + (R.axis == 0 ? d : 0),
                            j + (R.axis == 1 ? d : 0),
                            k + (R.axis == 2 ? d : 0), r0, r1, r2, ny, nz);
    };
    return applyRule(R, i, j, k, ny, nz, g);
  }
}

/** One ghost point of one array (blockIdx.y) per thread. */
__global__ void k_fillGhosts(GhostBatch b, FaceSlabs s) {
  const int a = blockIdx.y;
  int i, j, k;
  if (!slabPoint(s, blockIdx.x * blockDim.x + threadIdx.x, i, j, k))
    return;
  T* const arr = batchArray(b, a);
  const GhostRuleDev* R = b.rules + (a % b.slots) * kFaces;

  // the ghost axes of the point that have a rule, sorted by (rank, axis)
  AxisRule r0, r1, r2;
  int count = 0;
#pragma unroll
  for (int e = 0; e < 3; e++) {
    const int c = e == 0 ? i : (e == 1 ? j : k);
    const bool low = c == 0, high = c == s.n[e] - 1;
    if (!(low && s.active[2 * e]) && !(high && s.active[2 * e + 1]))
      continue;
    AxisRule x;
    x.axis = e;
    x.inward = high ? -1 : 1;
    x.r = R[2 * e + (high ? 1 : 0)];
    if (x.r.kind == GHOST_KEEP)
      continue;
    if (count == 0) {
      r0 = x;
    } else if (count == 1) {
      if (x.r.rank < r0.r.rank) {
        r1 = r0;
        r0 = x;
      } else {
        r1 = x;
      }
    } else {
      if (x.r.rank < r0.r.rank) {
        r2 = r1;
        r1 = r0;
        r0 = x;
      } else if (x.r.rank < r1.r.rank) {
        r2 = r1;
        r1 = x;
      } else {
        r2 = x;
      }
    }
    count++;
  }
  if (count == 0)
    return;
  const int ny = s.n[1], nz = s.n[2];
  T v;
  if (count == 1)
    v = compose<1>(arr, i, j, k, r0, r1, r2, ny, nz);
  else if (count == 2)
    v = compose<2>(arr, i, j, k, r0, r1, r2, ny, nz);
  else
    v = compose<3>(arr, i, j, k, r0, r1, r2, ny, nz);
  arr[stencilIdx(i, j, k, ny, nz)] = v;
}

// -------------------------------------------------------------------------
//  Boundary rows
// -------------------------------------------------------------------------

/** The special faces holding node (i, j, k): inflow, and the axes of the
 *  conductor faces (bit e). */
__device__ __forceinline__ void rowFacesAt(const RowFaces& rf, int i, int j,
                                           int k, bool& inflow, int& cond) {
  inflow = false;
  cond = 0;
#pragma unroll
  for (int e = 0; e < 3; e++) {
    const int c = e == 0 ? i : (e == 1 ? j : k);
    const int kind = c == 1                    ? rf.kind[2 * e]
                     : c == rf.planes.n[e] - 2 ? rf.kind[2 * e + 1]
                                               : ROW_NONE;
    inflow = inflow || kind == ROW_INFLOW;
    if (kind == ROW_CONDUCTOR)
      cond |= 1 << e;
  }
}

/** Packed Krylov index of node (i, j, k) (layout of gpuPhys2Solver3). */
__device__ __forceinline__ int krylovNode(int i, int j, int k, int nyn,
                                          int nzn) {
  return ((i - 1) * (nyn - 2) + (j - 1)) * (nzn - 2) + (k - 1);
}

/** The identity rows (image = v) where an inflow face or a conductor face
 *  tangential to the component holds the node.  A conductor's normal row is
 *  the interior row the node kernel wrote, read through the conductor image
 *  (its ghosts): it is not touched. */
__global__ void k_boundaryImageRows(RowFaces rf, T* __restrict__ im,
                                    const T* __restrict__ vx,
                                    const T* __restrict__ vy,
                                    const T* __restrict__ vz) {
  int i, j, k;
  if (!slabPoint(rf.planes, blockIdx.x * blockDim.x + threadIdx.x, i, j, k))
    return;
  bool inflow;
  int cond;
  rowFacesAt(rf, i, j, k, inflow, cond);
  const int nyn = rf.planes.n[1], nzn = rf.planes.n[2];
  const int idx = stencilIdx(i, j, k, nyn, nzn);
  T* const row = im + 3 * krylovNode(i, j, k, nyn, nzn);
  if (inflow || (cond & ~1) != 0)
    row[0] = vx[idx];
  if (inflow || (cond & ~2) != 0)
    row[1] = vy[idx];
  if (inflow || (cond & ~4) != 0)
    row[2] = vz[idx];
}

/** Face values of the node vector (ex, ey, ez): E0 on inflow faces (with
 *  kInflow), then 0 for the components tangential to a conductor. */
template <bool kInflow>
__global__ void k_boundaryFaceValues(RowFaces rf, T* __restrict__ ex,
                                     T* __restrict__ ey, T* __restrict__ ez) {
  int i, j, k;
  if (!slabPoint(rf.planes, blockIdx.x * blockDim.x + threadIdx.x, i, j, k))
    return;
  bool inflow;
  int cond;
  rowFacesAt(rf, i, j, k, inflow, cond);
  const int idx = stencilIdx(i, j, k, rf.planes.n[1], rf.planes.n[2]);
  T* const E[3] = {ex, ey, ez};
#pragma unroll
  for (int c = 0; c < 3; c++) {
    if (kInflow && inflow)
      E[c][idx] = rf.E0[c];
    else if ((cond & ~(1 << c)) != 0)
      E[c][idx] = 0.0;
  }
}

// -------------------------------------------------------------------------
//  Layer operations
// -------------------------------------------------------------------------

/** Op on the point (i, j, k) of its box; K the operation. */
template <int K>
__global__ void k_layerOp(T* X, T* Y, T* Z, LayerOp op, PointBox box, int ny,
                          int nz, int stride) {
  int i, j, k;
  if (!pointOf(box, i, j, k))
    return;
  const int idx = stencilIdx(i, j, k, ny, nz);
  const int layer = op.axis == 0 ? i : (op.axis == 1 ? j : k);
  if constexpr (K == LAYER_BLEND) {
    const T sal = op.ascending ? (T)(layer - op.first) * op.invLayers
                               : (T)(op.last - layer) * op.invLayers;
    if (op.mask & 1)
      X[idx] = X[idx] * sal + op.target[0] * (1.0 - sal);
    if (op.mask & 2)
      Y[idx] = Y[idx] * sal + op.target[1] * (1.0 - sal);
    if (op.mask & 4)
      Z[idx] = Z[idx] * sal + op.target[2] * (1.0 - sal);
  } else if constexpr (K == LAYER_SET) {
    if (op.mask & 1)
      X[idx] = op.target[0];
    if (op.mask & 2)
      Y[idx] = op.target[1];
    if (op.mask & 4)
      Z[idx] = op.target[2];
  } else {
    const int ref = idx + (op.ref - layer) * stride;
    X[idx] = X[ref];
    Y[idx] = Y[ref];
    Z[idx] = Z[ref];
  }
}

// -------------------------------------------------------------------------
//  Moment fold, wall values, outflow face
// -------------------------------------------------------------------------

/** 2 per active face whose node plane holds the point. */
__global__ void k_foldMoments(FaceSlabs s, T* const* __restrict__ ptrs) {
  int i, j, k;
  if (!slabPoint(s, blockIdx.x * blockDim.x + threadIdx.x, i, j, k))
    return;
  T factor = 1.0;
#pragma unroll
  for (int e = 0; e < 3; e++) {
    const int c = e == 0 ? i : (e == 1 ? j : k);
    if ((c == 1 && s.active[2 * e]) || (c == s.n[e] - 2 && s.active[2 * e + 1]))
      factor *= 2.0;
  }
  ptrs[blockIdx.y][stencilIdx(i, j, k, s.n[1], s.n[2])] *= factor;
}

/** One cell (s, f) of the face plane per thread: the sum of the 4 face
 *  nodes of the cell face, halved (2 W). */
__global__ void k_conductorWallSum(T* __restrict__ sum,
                                   const T* __restrict__ Bn, int base,
                                   int strideSlow, int strideFast, int nFast,
                                   int points) {
  const int t = blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= points)
    return;
  const int s = t / nFast;
  const int n = base + s * strideSlow + (t - s * nFast) * strideFast;
  sum[t] = 0.5 * ((Bn[n] + Bn[n + strideSlow]) +
                  (Bn[n + strideFast] + Bn[n + strideSlow + strideFast]));
}

/** One thread per (j,k) line: rescale the reference-plane species densities
 *  to a neutral pair and write them on every layer.  A line where one sign
 *  is missing is emptied, which is the only neutral choice left. */
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

/** Source row of a ghost row under rule r: partner (periodic) or adjacent. */
__device__ __forceinline__ int ghostRowSource(int r, int ghost, int n) {
  if (r == GHOST_ROW_PERIODIC)
    return ghost == 0 ? n - 2 : 1;
  return ghost == 0 ? 1 : n - 2;
}

/** The ghost rows of the plane i = nxc-1 of array blockIdx.y: axis 1, the
 *  rows j = 0, nyc-1 over k in [1, nzc-2]; axis 2, the rows k = 0, nzc-1
 *  over all j. */
__global__ void k_ghostRowsXright(GhostBatch b, int nxc, int nyc, int nzc,
                                  int axis, int ruleLo, int ruleHi) {
  T* const a = batchArray(b, blockIdx.y);
  const int t = blockIdx.x * blockDim.x + threadIdx.x;
  const int i = nxc - 1;
  if (axis == 1) {
    const int k = t + 1;
    if (k > nzc - 2)
      return;
    if (ruleLo != GHOST_ROW_KEEP)
      a[stencilIdx(i, 0, k, nyc, nzc)] =
          a[stencilIdx(i, ghostRowSource(ruleLo, 0, nyc), k, nyc, nzc)];
    if (ruleHi != GHOST_ROW_KEEP)
      a[stencilIdx(i, nyc - 1, k, nyc, nzc)] =
          a[stencilIdx(i, ghostRowSource(ruleHi, nyc - 1, nyc), k, nyc, nzc)];
  } else {
    const int j = t;
    if (j > nyc - 1)
      return;
    if (ruleLo != GHOST_ROW_KEEP)
      a[stencilIdx(i, j, 0, nyc, nzc)] =
          a[stencilIdx(i, j, ghostRowSource(ruleLo, 0, nzc), nyc, nzc)];
    if (ruleHi != GHOST_ROW_KEEP)
      a[stencilIdx(i, j, nzc - 1, nyc, nzc)] =
          a[stencilIdx(i, j, ghostRowSource(ruleHi, nzc - 1, nzc), nyc, nzc)];
  }
}

/** Slabs of the planes of the active faces: ghost planes (whole) or face
 *  node planes (interior). */
FaceSlabs makeSlabs(const int n[3], const bool active[kFaces], bool ghost) {
  FaceSlabs s;
  int lo[3], hi[3];
  for (int e = 0; e < 3; e++) {
    s.n[e] = n[e];
    lo[e] = ghost ? 0 : 1;
    hi[e] = ghost ? n[e] - 1 : n[e] - 2;
  }
  int count = 0;
  for (int f = 0; f < kFaces; f++) {
    const int a = faceAxis(f);
    s.active[f] = active[f];
    s.plane[f] =
        ghost ? (faceHigh(f) ? n[a] - 1 : 0) : (faceHigh(f) ? n[a] - 2 : 1);
    s.start[f] = count;
    s.fastN[f] = 1;
    if (active[f]) {
      const int slow = a == 0 ? 1 : 0, fast = a == 2 ? 1 : 2;
      s.slowLo[f] = lo[slow];
      s.fastLo[f] = lo[fast];
      const int ns = std::max(0, hi[slow] - lo[slow] + 1);
      const int nf = std::max(0, hi[fast] - lo[fast] + 1);
      s.fastN[f] = std::max(1, nf);
      count += ns * nf;
    }
    // the later axes leave out the planes of this axis's active faces
    if (faceHigh(f)) {
      if (active[f - 1])
        lo[a] = s.plane[f - 1] + 1;
      if (active[f])
        hi[a] = s.plane[f] - 1;
    }
  }
  s.start[kFaces] = count;
  return s;
}

} // namespace

// =========================================================================
//  Launchers
// =========================================================================

FaceSlabs ghostShell(const int n[3], const bool active[kFaces]) {
  return makeSlabs(n, active, true);
}

FaceSlabs nodePlanes(const int n[3], const bool active[kFaces]) {
  return makeSlabs(n, active, false);
}

void gpuFillGhosts(const GhostBatch& batch, const FaceSlabs& shell,
                   cudaStream_t stream) {
  if (shell.count() == 0 || batch.n == 0)
    return;
  k_fillGhosts<<<dim3(blocks(shell.count()), batch.n), kFaceBlock, 0, stream>>>(
      batch, shell);
  cudaErrChk(cudaGetLastError());
}

void gpuBoundaryImageRows(const RowFaces& rows, cudaSolverType* im,
                          const cudaSolverType* const v[3],
                          cudaStream_t stream) {
  if (rows.planes.count() == 0)
    return;
  k_boundaryImageRows<<<blocks(rows.planes.count()), kFaceBlock, 0, stream>>>(
      rows, im, v[0], v[1], v[2]);
  cudaErrChk(cudaGetLastError());
}

void gpuBoundaryFaceValues(const RowFaces& rows, cudaSolverType* const v[3],
                           bool inflow, cudaStream_t stream) {
  if (rows.planes.count() == 0)
    return;
  const auto launch = [&](auto kernel) {
    kernel<<<blocks(rows.planes.count()), kFaceBlock, 0, stream>>>(rows, v[0],
                                                                   v[1], v[2]);
  };
  if (inflow)
    launch(k_boundaryFaceValues<true>);
  else
    launch(k_boundaryFaceValues<false>);
  cudaErrChk(cudaGetLastError());
}

void gpuLayerOp(const LayerOp& op, cudaSolverType* X, cudaSolverType* Y,
                cudaSolverType* Z, const int n[3], cudaStream_t stream) {
  PointSpan span[3] = {pointRange(0, n[0] - 1), pointRange(0, n[1] - 1),
                       pointRange(0, n[2] - 1)};
  span[op.axis] = pointRange(op.first, op.last);
  const PointBox box = {span[0], span[1], span[2]};
  if (box.empty())
    return;
  const int stride[3] = {n[1] * n[2], n[2], 1};
  const PointLaunch L = pointLaunch(box);
  const auto launch = [&](auto kernel) {
    kernel<<<L.grid, L.block, 0, stream>>>(X, Y, Z, op, box, n[1], n[2],
                                           stride[op.axis]);
  };
  switch (op.kind) {
  case LAYER_BLEND:
    launch(k_layerOp<LAYER_BLEND>);
    break;
  case LAYER_SET:
    launch(k_layerOp<LAYER_SET>);
    break;
  default:
    launch(k_layerOp<LAYER_COPY>);
    break;
  }
  cudaErrChk(cudaGetLastError());
}

void gpuFoldMoments(const FaceSlabs& planes, cudaSolverType* const* d_ptrs,
                    int nFields, cudaStream_t stream) {
  if (planes.count() == 0 || nFields == 0)
    return;
  k_foldMoments<<<dim3(blocks(planes.count()), nFields), kFaceBlock, 0,
                  stream>>>(planes, d_ptrs);
  cudaErrChk(cudaGetLastError());
}

void gpuConductorWallSum(cudaSolverType* sum, const cudaSolverType* Bn,
                         const int nodes[3], int f, cudaStream_t stream) {
  const int dir = faceAxis(f);
  const int stride[3] = {nodes[1] * nodes[2], nodes[2], 1};
  const int slow = dir == 0 ? 1 : 0, fast = dir == 2 ? 1 : 2;
  const int nFast = nodes[fast] - 1; // cells along the fast axis
  const int points = (nodes[slow] - 1) * nFast;
  if (points <= 0)
    return;
  const int face = faceHigh(f) ? nodes[dir] - 2 : 1; // the face node plane
  k_conductorWallSum<<<blocks(points), kFaceBlock, 0, stream>>>(
      sum, Bn, face * stride[dir], stride[slow], stride[fast], nFast, points);
  cudaErrChk(cudaGetLastError());
}

void gpuNeutralizeLayersX(cudaSolverType* rhons, const cudaSolverType* qom,
                          int ns, int first, int last, int ref, int nx, int ny,
                          int nz, cudaStream_t stream) {
  const int total = ny * nz;
  if (total <= 0 || last < first)
    return;
  k_neutralizeLayersX<<<blocks(total), kFaceBlock, 0, stream>>>(
      rhons, qom, ns, first, last, ref, nx, ny, nz);
  cudaErrChk(cudaGetLastError());
}

void gpuGhostRowsXright(const GhostBatch& batch, int nxc, int nyc, int nzc,
                        const int rule[4], cudaStream_t stream) {
  if (batch.n <= 0)
    return;
  if (rule[0] != GHOST_ROW_KEEP || rule[1] != GHOST_ROW_KEEP) {
    k_ghostRowsXright<<<dim3(blocks(nzc - 2), batch.n), kFaceBlock, 0,
                        stream>>>(batch, nxc, nyc, nzc, 1, rule[0], rule[1]);
    cudaErrChk(cudaGetLastError());
  }
  if (rule[2] != GHOST_ROW_KEEP || rule[3] != GHOST_ROW_KEEP) {
    k_ghostRowsXright<<<dim3(blocks(nyc), batch.n), kFaceBlock, 0, stream>>>(
        batch, nxc, nyc, nzc, 2, rule[2], rule[3]);
    cudaErrChk(cudaGetLastError());
  }
}
