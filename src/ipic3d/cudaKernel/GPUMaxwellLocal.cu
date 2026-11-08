/**
 * @file GPUMaxwellLocal.cu
 * @brief Fused CUDA kernels of the Maxwell image operator.
 *
 * Two kernels that together compute  A·E  on the interior nodes, both for
 * the full and the communication-free image:
 *
 *   k_maxwellLocalCenterOps  — fuses 3×gradN2C + divN2C  (center-level)
 *   k_maxwellLocalNodeFused  — fuses 3×divC2N + gradC2N + arithmetic
 *                               + Krylov packing          (node-level)
 *
 * Each launch covers a PointBox (3D launch) or a PointList (1D launch); the
 * images launch the PointLists of maxwellImageBoxes, for the whole and the
 * split operator.  The ghost cells the node kernel reads
 * are set by the caller (zero, or the halo exchange and face BCs).  Boundary
 * rows are written by the caller into the Krylov image afterwards.
 *
 * See GPUMaxwellLocal.cuh for API documentation.
 */

#include "GPUMaxwellLocal.cuh"
#include "GPUPointLaunch.cuh"

// =========================================================================
//  Kernel 1:  Fused center-level operations
//
//  Combines 3× gradN2C(Ex,Ey,Ez) + divN2C(Dx,Dy,Dz) into ONE kernel,
//  with the expressions of k_gradN2C and k_divN2C.
//  Each thread processes one center cell (i,j,k) of points.
//
//  Stencil: reads 8 corner nodes  (i..i+1) × (j..j+1) × (k..k+1)
//           for each of 6 input node arrays.
//  Output:  10 center-sized arrays.
// =========================================================================

template <typename T, typename Points>
__global__ void k_maxwellLocalCenterOps(
    T* __restrict__ gExX, T* __restrict__ gExY, T* __restrict__ gExZ,
    T* __restrict__ gEyX, T* __restrict__ gEyY, T* __restrict__ gEyZ,
    T* __restrict__ gEzX, T* __restrict__ gEzY, T* __restrict__ gEzZ,
    T* __restrict__ divD, const T* __restrict__ vX, const T* __restrict__ vY,
    const T* __restrict__ vZ, const T* __restrict__ dX,
    const T* __restrict__ dY, const T* __restrict__ dZ, int nyc, int nzc,
    int nyn, int nzn, T invdx, T invdy, T invdz, Points points) {
  int i, j, k;
  if (!pointOf(points, i, j, k))
    return;

// The 8 corner node positions for this center cell (i,j,k):
//   node indices: (i, i+1) × (j, j+1) × (k, k+1)

// Flat index helper for node arrays  [nxn][nyn][nzn]
#define NI(ii, jj, kk) ((ii) * nyn * nzn + (jj) * nzn + (kk))

  // Load 8 corner positions — process one input field at a time
  // to limit register pressure.  Compiler may still keep them
  // in registers across the block if beneficial.

  const int ci = i * nyc * nzc + j * nzc + k; // flat center index
  const T Q = 0.25;

  // ---- gradN2C(Ex) → gExX, gExY, gExZ ----
  {
    T n000 = vX[NI(i, j, k)];
    T n001 = vX[NI(i, j, k + 1)];
    T n010 = vX[NI(i, j + 1, k)];
    T n011 = vX[NI(i, j + 1, k + 1)];
    T n100 = vX[NI(i + 1, j, k)];
    T n101 = vX[NI(i + 1, j, k + 1)];
    T n110 = vX[NI(i + 1, j + 1, k)];
    T n111 = vX[NI(i + 1, j + 1, k + 1)];

    gExX[ci] = Q * invdx *
               ((n100 - n000) + (n101 - n001) + (n110 - n010) + (n111 - n011));
    gExY[ci] = Q * invdy *
               ((n010 - n000) + (n011 - n001) + (n110 - n100) + (n111 - n101));
    gExZ[ci] = Q * invdz *
               ((n001 - n000) + (n101 - n100) + (n011 - n010) + (n111 - n110));
  }

  // ---- gradN2C(Ey) → gEyX, gEyY, gEyZ ----
  {
    T n000 = vY[NI(i, j, k)];
    T n001 = vY[NI(i, j, k + 1)];
    T n010 = vY[NI(i, j + 1, k)];
    T n011 = vY[NI(i, j + 1, k + 1)];
    T n100 = vY[NI(i + 1, j, k)];
    T n101 = vY[NI(i + 1, j, k + 1)];
    T n110 = vY[NI(i + 1, j + 1, k)];
    T n111 = vY[NI(i + 1, j + 1, k + 1)];

    gEyX[ci] = Q * invdx *
               ((n100 - n000) + (n101 - n001) + (n110 - n010) + (n111 - n011));
    gEyY[ci] = Q * invdy *
               ((n010 - n000) + (n011 - n001) + (n110 - n100) + (n111 - n101));
    gEyZ[ci] = Q * invdz *
               ((n001 - n000) + (n101 - n100) + (n011 - n010) + (n111 - n110));
  }

  // ---- gradN2C(Ez) → gEzX, gEzY, gEzZ ----
  {
    T n000 = vZ[NI(i, j, k)];
    T n001 = vZ[NI(i, j, k + 1)];
    T n010 = vZ[NI(i, j + 1, k)];
    T n011 = vZ[NI(i, j + 1, k + 1)];
    T n100 = vZ[NI(i + 1, j, k)];
    T n101 = vZ[NI(i + 1, j, k + 1)];
    T n110 = vZ[NI(i + 1, j + 1, k)];
    T n111 = vZ[NI(i + 1, j + 1, k + 1)];

    gEzX[ci] = Q * invdx *
               ((n100 - n000) + (n101 - n001) + (n110 - n010) + (n111 - n011));
    gEzY[ci] = Q * invdy *
               ((n010 - n000) + (n011 - n001) + (n110 - n100) + (n111 - n101));
    gEzZ[ci] = Q * invdz *
               ((n001 - n000) + (n101 - n100) + (n011 - n010) + (n111 - n110));
  }

  // ---- divN2C(D) → divD ----
  //   ∇·D = d(Dx)/dx + d(Dy)/dy + d(Dz)/dz
  {
    // d(Dx)/dx
    T compX = Q * invdx *
              ((dX[NI(i + 1, j, k)] - dX[NI(i, j, k)]) +
               (dX[NI(i + 1, j, k + 1)] - dX[NI(i, j, k + 1)]) +
               (dX[NI(i + 1, j + 1, k)] - dX[NI(i, j + 1, k)]) +
               (dX[NI(i + 1, j + 1, k + 1)] - dX[NI(i, j + 1, k + 1)]));
    // d(Dy)/dy
    T compY = Q * invdy *
              ((dY[NI(i, j + 1, k)] - dY[NI(i, j, k)]) +
               (dY[NI(i, j + 1, k + 1)] - dY[NI(i, j, k + 1)]) +
               (dY[NI(i + 1, j + 1, k)] - dY[NI(i + 1, j, k)]) +
               (dY[NI(i + 1, j + 1, k + 1)] - dY[NI(i + 1, j, k + 1)]));
    // d(Dz)/dz
    T compZ = Q * invdz *
              ((dZ[NI(i, j, k + 1)] - dZ[NI(i, j, k)]) +
               (dZ[NI(i + 1, j, k + 1)] - dZ[NI(i + 1, j, k)]) +
               (dZ[NI(i, j + 1, k + 1)] - dZ[NI(i, j + 1, k)]) +
               (dZ[NI(i + 1, j + 1, k + 1)] - dZ[NI(i + 1, j + 1, k)]));

    divD[ci] = compX + compY + compZ;
  }
#undef NI
}

// =========================================================================
//  Kernel 2:  Fused node-level operations + Krylov packing
//
//  Combines:
//    3× divC2N  (Laplacians from 9 center gradient arrays)
//    1× gradC2N (gradient of divD)
//    negate, subtract, scale(dt²), add D, add E
//    Krylov packing (directly into the Krylov vector)
//
//  Each thread processes one interior node (i,j,k) of points.
//
//  Stencil: reads 8 surrounding center cells (i-1..i) × (j-1..j) × (k-1..k)
//           for each of 10 center arrays (indexed by cell, see CellIndex),
//           plus 6 node values at (i,j,k).
//  Output:  3 interleaved doubles in the Krylov vector d_im.
//  kUnitK: cell.sk == 1 at compile time, so the k-1 cells are immediate
//  offsets of the k cells (cell.sk == 0 only along a flat z axis).
// =========================================================================

template <typename T, typename Points, bool kUnitK>
__global__ void k_maxwellLocalNodeFused(
    T* __restrict__ d_im,
    // 9 center arrays: gradients of E (for Laplacian via divC2N)
    const T* __restrict__ gExX, const T* __restrict__ gExY,
    const T* __restrict__ gExZ, const T* __restrict__ gEyX,
    const T* __restrict__ gEyY, const T* __restrict__ gEyZ,
    const T* __restrict__ gEzX, const T* __restrict__ gEzY,
    const T* __restrict__ gEzZ,
    // 1 center array: divergence of D (for gradient via gradC2N)
    const T* __restrict__ divD,
    // 6 node arrays
    const T* __restrict__ vX, const T* __restrict__ vY,
    const T* __restrict__ vZ, const T* __restrict__ dX,
    const T* __restrict__ dY, const T* __restrict__ dZ, int nyn, int nzn,
    T invdx, T invdy, T invdz, T dt2, Points points, CellIndex cell) {
  int i, j, k;
  if (!pointOf(points, i, j, k))
    return;

  // Pre-compute the 8 flat center indices for the surrounding cells:
  //   (i-1..i) × (j-1..j) × (k-1..k)
  // Binary encoding: bit2=i-dir, bit1=j-dir, bit0=k-dir
  //   0→(dim-1), 1→(dim)
  const int sk = kUnitK ? 1 : cell.sk;
  const int c111 = cell.base + i * cell.si + j * cell.sj + k * sk;
  const int c110 = c111 - sk;
  const int c101 = c111 - cell.sj;
  const int c100 = c101 - sk;
  const int c011 = c111 - cell.si;
  const int c010 = c011 - sk;
  const int c001 = c011 - cell.sj;
  const int c000 = c001 - sk;

  const T Q = 0.25;

// ================================================================
//  Compute lapX = divC2N( gExX, gExY, gExZ )
//
//  divC2N stencil (from k_divC2N in GPUStencils.cu):
//    compX = 0.25*invdx * Σ (vecXC[i,*,*] - vecXC[i-1,*,*])
//    compY = 0.25*invdy * Σ (vecYC[*,j,*] - vecYC[*,j-1,*])
//    compZ = 0.25*invdz * Σ (vecZC[*,*,k] - vecZC[*,*,k-1])
//    div = compX + compY + compZ
// ================================================================

// Helper macro: divC2N for a triplet of center arrays (vXC,vYC,vZC)
// Returns the scalar divergence at node (i,j,k).
#define DIVC2N(arrX, arrY, arrZ)                                               \
  (Q * invdx *                                                                 \
       ((arrX[c111] - arrX[c011]) + (arrX[c110] - arrX[c010]) +                \
        (arrX[c101] - arrX[c001]) + (arrX[c100] - arrX[c000])) +               \
   Q * invdy *                                                                 \
       ((arrY[c111] - arrY[c101]) + (arrY[c110] - arrY[c100]) +                \
        (arrY[c011] - arrY[c001]) + (arrY[c010] - arrY[c000])) +               \
   Q * invdz *                                                                 \
       ((arrZ[c111] - arrZ[c110]) + (arrZ[c101] - arrZ[c100]) +                \
        (arrZ[c011] - arrZ[c010]) + (arrZ[c001] - arrZ[c000])))

  T lapX = DIVC2N(gExX, gExY, gExZ);
  T lapY = DIVC2N(gEyX, gEyY, gEyZ);
  T lapZ = DIVC2N(gEzX, gEzY, gEzZ);

  // ================================================================
  //  Compute gdivX/Y/Z = gradC2N( divD )
  //
  //  gradC2N stencil (from k_gradC2N in GPUStencils.cu):
  //    gradX = 0.25*invdx * [(c111-c011)+(c110-c010)+(c101-c001)+(c100-c000)]
  //    gradY = 0.25*invdy * [(c111-c101)+(c110-c100)+(c011-c001)+(c010-c000)]
  //    gradZ = 0.25*invdz * [(c111-c110)+(c101-c100)+(c011-c010)+(c001-c000)]
  // ================================================================

  T dd000 = divD[c000], dd001 = divD[c001];
  T dd010 = divD[c010], dd011 = divD[c011];
  T dd100 = divD[c100], dd101 = divD[c101];
  T dd110 = divD[c110], dd111 = divD[c111];

  T gdivX =
      Q * invdx *
      ((dd111 - dd011) + (dd110 - dd010) + (dd101 - dd001) + (dd100 - dd000));
  T gdivY =
      Q * invdy *
      ((dd111 - dd101) + (dd110 - dd100) + (dd011 - dd001) + (dd010 - dd000));
  T gdivZ =
      Q * invdz *
      ((dd111 - dd110) + (dd101 - dd100) + (dd011 - dd010) + (dd001 - dd000));

#undef DIVC2N

  // ================================================================
  //  Arithmetic:  im = dt² * (-lap - gdiv) + D + E
  // ================================================================

  int nidx = i * nyn * nzn + j * nzn + k;

  T ex = vX[nidx];
  T ey = vY[nidx];
  T ez = vZ[nidx];
  T dxv = dX[nidx];
  T dyv = dY[nidx];
  T dzv = dZ[nidx];

  T imX = dt2 * (-lapX - gdivX) + dxv + ex;
  T imY = dt2 * (-lapY - gdivY) + dyv + ey;
  T imZ = dt2 * (-lapZ - gdivZ) + dzv + ez;

  // ================================================================
  //  Pack to the Krylov vector (interleaved layout of gpuPhys2Solver3)
  //    sol_idx = ((i-1)*(ny2) + (j-1)) * nz2 + (k-1)
  //    d_im[sol_idx*3 + 0/1/2] = imX/imY/imZ
  // ================================================================

  int ny2 = nyn - 2;
  int nz2 = nzn - 2;
  int sol_idx = ((i - 1) * ny2 + (j - 1)) * nz2 + (k - 1);

  d_im[sol_idx * 3] = imX;
  d_im[sol_idx * 3 + 1] = imY;
  d_im[sol_idx * 3 + 2] = imZ;
}

// =========================================================================
//  Host wrappers
// =========================================================================

template <typename Points>
void gpuMaxwellLocalCenterOps(
    cudaSolverType* gExX, cudaSolverType* gExY, cudaSolverType* gExZ,
    cudaSolverType* gEyX, cudaSolverType* gEyY, cudaSolverType* gEyZ,
    cudaSolverType* gEzX, cudaSolverType* gEzY, cudaSolverType* gEzZ,
    cudaSolverType* divD, const cudaSolverType* vX, const cudaSolverType* vY,
    const cudaSolverType* vZ, const cudaSolverType* dX,
    const cudaSolverType* dY, const cudaSolverType* dZ, int nxc, int nyc,
    int nzc, cudaSolverType invdx, cudaSolverType invdy, cudaSolverType invdz,
    const Points& points, cudaStream_t stream) {
  if (points.empty())
    return;
  const PointLaunch L = pointLaunch(points);
  k_maxwellLocalCenterOps<cudaSolverType, Points>
      <<<L.grid, L.block, 0, stream>>>(gExX, gExY, gExZ, gEyX, gEyY, gEyZ, gEzX,
                                       gEzY, gEzZ, divD, vX, vY, vZ, dX, dY, dZ,
                                       nyc, nzc, nyc + 1, nzc + 1, invdx, invdy,
                                       invdz, points);
  cudaErrChk(cudaGetLastError());
}

template <typename Points>
void gpuMaxwellLocalNodeFused(
    cudaSolverType* d_im, const cudaSolverType* gExX,
    const cudaSolverType* gExY, const cudaSolverType* gExZ,
    const cudaSolverType* gEyX, const cudaSolverType* gEyY,
    const cudaSolverType* gEyZ, const cudaSolverType* gEzX,
    const cudaSolverType* gEzY, const cudaSolverType* gEzZ,
    const cudaSolverType* divD, const cudaSolverType* vX,
    const cudaSolverType* vY, const cudaSolverType* vZ,
    const cudaSolverType* dX, const cudaSolverType* dY,
    const cudaSolverType* dZ, int nxn, int nyn, int nzn, cudaSolverType invdx,
    cudaSolverType invdy, cudaSolverType invdz, cudaSolverType dt2,
    const Points& points, const CellIndex& cell, cudaStream_t stream) {
  if (points.empty())
    return;
  const PointLaunch L = pointLaunch(points);
  if (cell.sk == 1)
    k_maxwellLocalNodeFused<cudaSolverType, Points, true>
        <<<L.grid, L.block, 0, stream>>>(d_im, gExX, gExY, gExZ, gEyX, gEyY,
                                         gEyZ, gEzX, gEzY, gEzZ, divD, vX, vY,
                                         vZ, dX, dY, dZ, nyn, nzn, invdx, invdy,
                                         invdz, dt2, points, cell);
  else
    k_maxwellLocalNodeFused<cudaSolverType, Points, false>
        <<<L.grid, L.block, 0, stream>>>(d_im, gExX, gExY, gExZ, gEyX, gEyY,
                                         gEyZ, gEzX, gEzY, gEzZ, divD, vX, vY,
                                         vZ, dX, dY, dZ, nyn, nzn, invdx, invdy,
                                         invdz, dt2, points, cell);
  cudaErrChk(cudaGetLastError());
}

// The two point sets: PointList (1D launch, the images) and PointBox (3D
// launch, the unit tests)
#define MAXWELL_LOCAL_INSTANTIATE(Points)                                      \
  template void gpuMaxwellLocalCenterOps<Points>(                              \
      cudaSolverType*, cudaSolverType*, cudaSolverType*, cudaSolverType*,      \
      cudaSolverType*, cudaSolverType*, cudaSolverType*, cudaSolverType*,      \
      cudaSolverType*, cudaSolverType*, const cudaSolverType*,                 \
      const cudaSolverType*, const cudaSolverType*, const cudaSolverType*,     \
      const cudaSolverType*, const cudaSolverType*, int, int, int,             \
      cudaSolverType, cudaSolverType, cudaSolverType, const Points&,           \
      cudaStream_t);                                                           \
  template void gpuMaxwellLocalNodeFused<Points>(                              \
      cudaSolverType*, const cudaSolverType*, const cudaSolverType*,           \
      const cudaSolverType*, const cudaSolverType*, const cudaSolverType*,     \
      const cudaSolverType*, const cudaSolverType*, const cudaSolverType*,     \
      const cudaSolverType*, const cudaSolverType*, const cudaSolverType*,     \
      const cudaSolverType*, const cudaSolverType*, const cudaSolverType*,     \
      const cudaSolverType*, const cudaSolverType*, int, int, int,             \
      cudaSolverType, cudaSolverType, cudaSolverType, cudaSolverType,          \
      const Points&, const CellIndex&, cudaStream_t);
MAXWELL_LOCAL_INSTANTIATE(PointBox)
MAXWELL_LOCAL_INSTANTIATE(PointList)
#undef MAXWELL_LOCAL_INSTANTIATE

// =========================================================================
//  Launch sets
// =========================================================================

MaxwellImageBoxes maxwellImageBoxes(const int nCell[3], int cellEndX,
                                    const bool flat[3], bool overlap) {
  MaxwellImageBoxes b{};
  int nc[3], nn[3], cellEnd[3];
  PointSpan mudot[3], cells[3], nodes[3];
  for (int a = 0; a < 3; a++) {
    nc[a] = nCell[a];
    nn[a] = nCell[a] + 1;
    cellEnd[a] = a == 0 ? cellEndX : nc[a] - 2;
    mudot[a] = pointRange(1, nn[a] - 2);
    cells[a] = pointRange(1, cellEnd[a]);
    nodes[a] = pointRange(1, nn[a] - 2);
  }
  const PointBox mudotBox = {mudot[0], mudot[1], mudot[2]};
  const PointBox cellBox = {cells[0], cells[1], cells[2]};
  const PointBox nodeBox = {nodes[0], nodes[1], nodes[2]};
  b.mudot = pointList(&mudotBox, 1);
  b.cells = pointList(&cellBox, 1);
  b.nodes = pointList(&nodeBox, 1);
  const int stride[3] = {nc[1] * nc[2], nc[2], 1};
  b.cellIndex = {0, stride[0], stride[1], stride[2]};
  if (!overlap)
    return b;

  int nSplit = 0;
  for (int a = 0; a < 3; a++) {
    if (flat[a])
      continue;
    if (nc[a] - 2 < 3)
      return b;
    nSplit++;
  }
  if (nSplit == 0)
    return b;
  b.split = true;

  // a flat axis reads its single cell (index 1) for both neighbours
  int base = 0, s[3];
  for (int a = 0; a < 3; a++) {
    s[a] = flat[a] ? 0 : stride[a];
    base += flat[a] ? stride[a] : 0;
  }
  b.splitCellIndex = {base, s[0], s[1], s[2]};

  PointSpan mudotIn[3], cellIn[3], nodeIn[3], mudotSh[3], cellSh[3], nodeSh[3];
  for (int a = 0; a < 3; a++) {
    if (flat[a]) {
      mudotIn[a] = mudot[a];
      cellIn[a] = cells[a];
      nodeIn[a] = nodes[a];
      continue;
    }
    mudotIn[a] = pointRange(3, nn[a] - 4);
    cellIn[a] = pointRange(2, nc[a] - 3);
    nodeIn[a] = pointRange(2, nn[a] - 3);
    mudotSh[a] = pointRuns(1, 2, nn[a] - 3, nn[a] - 2);
    cellSh[a] = pointRuns(1, 1, nc[a] - 2, cellEnd[a]);
    nodeSh[a] = pointRuns(1, 1, nn[a] - 2, nn[a] - 2);
  }
  const PointBox mudotInBox = {mudotIn[0], mudotIn[1], mudotIn[2]};
  const PointBox cellInBox = {cellIn[0], cellIn[1], cellIn[2]};
  const PointBox nodeInBox = {nodeIn[0], nodeIn[1], nodeIn[2]};
  b.nodeInterior = pointList(&nodeInBox, 1);

  // shell of split axis a: interior along the split axes before it, the
  // whole range along the others; the shells of a stage form one list
  PointBox mudotShell[3], cellShell[3], nodeShell[3];
  int nShell = 0;
  for (int a = 0; a < 3; a++) {
    if (flat[a])
      continue;
    PointSpan m[3], c[3], n[3];
    for (int d = 0; d < 3; d++) {
      const bool before = d < a && !flat[d];
      m[d] = d == a ? mudotSh[d] : before ? mudotIn[d] : mudot[d];
      c[d] = d == a ? cellSh[d] : before ? cellIn[d] : cells[d];
      n[d] = d == a ? nodeSh[d] : before ? nodeIn[d] : nodes[d];
    }
    mudotShell[nShell] = {m[0], m[1], m[2]};
    cellShell[nShell] = {c[0], c[1], c[2]};
    nodeShell[nShell] = {n[0], n[1], n[2]};
    nShell++;
  }
  b.nodeShells = pointList(nodeShell, nShell);

  // The exchange copies the cell of a flat axis into its ghosts for every
  // cell of the other axes: D and the cell arrays are then computed whole
  // before it starts, and only the image is split.
  if (flat[0] || flat[1] || flat[2]) {
    b.mudotShells = pointList(&mudotBox, 1);
    b.cellShells = pointList(&cellBox, 1);
  } else {
    b.mudotShells = pointList(mudotShell, nShell);
    b.cellShells = pointList(cellShell, nShell);
    b.mudotInterior = pointList(&mudotInBox, 1);
    b.cellInterior = pointList(&cellInBox, 1);
  }
  return b;
}
