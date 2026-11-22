/**
 * @file GPUCleaning.cu
 * @brief Operators and fused kernels of the divergence cleanings; see
 *        GPUCleaning.cuh.
 */

#include "GPUCleaning.cuh"
#include "GPUStencils.cuh"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>

// =========================================================================
//  Host: operator, launch rows, eigenvalue bounds, null space
// =========================================================================

Lap27Op lap27Op(const int nPacked[3], const double invh2[3], const bool flat[3],
                const bool faceRow[6], const bool faceCopy[6], bool nodes) {
  Lap27Op op{};
  op.ni = nPacked[0];
  op.nj = nPacked[1];
  op.nk = nPacked[2];
  op.divNk = RowDiv(op.nk);
  op.divNj = RowDiv(op.nj);
  for (int a = 0; a < 3; a++) {
    op.invh2[a] = invh2[a];
    op.flat[a] = flat[a];
    op.twin[a] = nodes && flat[a];
  }
  for (int f = 0; f < 6; f++) {
    if (flat[f / 2] && faceRow[f]) {
      fprintf(stderr, "Lap27Op: face rows on the flat axis %d\n", f / 2);
      abort();
    }
    op.faceRow[f] = faceRow[f];
    op.faceCopy[f] = faceRow[f] && faceCopy[f];
  }
  // w(a, b, c) = sum_d invh2_d δ²(o_d) μ²(o_e) μ²(o_f) by o != 0; a flat axis
  // has δ² = 0 and μ² = 1 at offset 0, 0 at ±1
  for (int x = 0; x < 2; x++)
    for (int y = 0; y < 2; y++)
      for (int z = 0; z < 2; z++) {
        const int o[3] = {x, y, z};
        double w = 0.0;
        for (int d = 0; d < 3; d++) {
          double term = op.flat[d] ? 0.0 : invh2[d] * (o[d] ? 1.0 : -2.0);
          for (int e = 0; e < 3; e++)
            if (e != d)
              term *= op.flat[e] ? (o[e] ? 0.0 : 1.0) : (o[e] ? 0.25 : 0.5);
          w += term;
        }
        op.w[x][y][z] = w;
      }
  const Lap27Eig e = lap27EigBounds(op);
  op.diag = e.max > 0.0 ? 0.5 * (e.min + e.max) : 1.0;
  return op;
}

Lap27Rows lap27Rows(const int n[3], const bool flat[3], bool overlap) {
  Lap27Rows r{};
  PointSpan all[3], in[3];
  int nSplit = 0;
  bool thin = false;
  for (int a = 0; a < 3; a++) {
    all[a] = pointRange(1, n[a] - 2);
    in[a] = flat[a] ? all[a] : pointRange(2, n[a] - 3);
    if (!flat[a]) {
      nSplit++;
      thin = thin || n[a] - 2 < 3;
    }
  }
  const PointBox allBox = {all[0], all[1], all[2]};
  r.all = pointList(&allBox, 1);
  r.split = overlap && nSplit > 0 && !thin;
  if (!r.split)
    return r;
  const PointBox inBox = {in[0], in[1], in[2]};
  r.interior = pointList(&inBox, 1);
  // shell of split axis a: interior along the split axes before it, all rows
  // along the others
  PointBox shells[3];
  int nShell = 0;
  for (int a = 0; a < 3; a++) {
    if (flat[a])
      continue;
    PointSpan s[3];
    for (int d = 0; d < 3; d++)
      s[d] = d == a  ? pointRuns(1, 1, n[d] - 2, n[d] - 2)
             : d < a ? in[d]
                     : all[d];
    shells[nShell++] = {s[0], s[1], s[2]};
  }
  r.shells = pointList(shells, nShell);
  return r;
}

namespace {
/** Spectrum of T u = -(u[j-1] - 2 u[j] + u[j+1]) / 4 over N points beyond
 *  which the value is 0 (copy = false) or the end point itself (copy =
 *  true): t_m = sin²(θ_m / 2).  The eigenvectors are sin(θ j + φ): a zero end
 *  puts a node one point out, a copy end an antinode half a point out, so
 *  θ_m = (m - C/2) π / (N + 1 - C/2), C the number of copy ends. */
struct Spectrum1D {
  double lo, hi, lo2; // smallest, largest, second smallest
};
Spectrum1D spectrum1D(int N, bool copyL, bool copyR) {
  const int C = copyL + copyR;
  const double L = N + 1 - 0.5 * C;
  const auto t = [&](int m) {
    const double s = std::sin(0.5 * (m - 0.5 * C) * M_PI / L);
    return s * s;
  };
  return {t(1), t(N), N > 1 ? t(2) : t(1)};
}
} // namespace

Lap27Eig lap27EigBounds(const Lap27Op& op) {
  const int n[3] = {op.ni, op.nj, op.nk};
  // candidate values of t per axis: the extremes, and the second smallest
  // where the smallest is 0 (the smallest non-zero eigenvalue may sit there)
  double cand[3][3];
  int nCand[3];
  for (int a = 0; a < 3; a++) {
    nCand[a] = 0;
    if (op.flat[a]) {
      cand[a][nCand[a]++] = 0.0;
      continue;
    }
    const bool rowL = op.faceRow[2 * a], rowR = op.faceRow[2 * a + 1];
    const int N = n[a] - rowL - rowR; // rows that are not identity rows
    const Spectrum1D s =
        spectrum1D(std::max(N, 1), op.faceCopy[2 * a], op.faceCopy[2 * a + 1]);
    cand[a][nCand[a]++] = s.lo;
    cand[a][nCand[a]++] = s.hi;
    if (s.lo == 0.0)
      cand[a][nCand[a]++] = s.lo2;
  }
  Lap27Eig e{0.0, 0.0, false};
  for (int x = 0; x < nCand[0]; x++)
    for (int y = 0; y < nCand[1]; y++)
      for (int z = 0; z < nCand[2]; z++) {
        const double t[3] = {cand[0][x], cand[1][y], cand[2][z]};
        double lam = 0.0;
        for (int a = 0; a < 3; a++) {
          double prod = 4.0 * op.invh2[a] * t[a];
          for (int b = 0; b < 3; b++)
            if (b != a)
              prod *= 1.0 - t[b];
          lam += prod;
        }
        e.max = std::max(e.max, lam);
        if (lam > 0.0 && (e.min == 0.0 || lam < e.min))
          e.min = lam;
        e.singular = e.singular || lam == 0.0;
      }
  return e;
}

CleaningNullSpace cleaningNullSpace(const Lap27Op& op, bool nodes,
                                    const int N[3], const int offset[3],
                                    const bool periodic[3],
                                    const bool sharedRight[3],
                                    const bool oddWalls[3], bool mean) {
  CleaningNullSpace ns{};
  const int n[3] = {op.ni, op.nj, op.nk};
  for (int a = 0; a < 3; a++) {
    ns.N[a] = N[a];
    ns.periodic[a] = periodic[a];
    ns.box.n[a] = n[a];
    ns.box.lo[a] = op.faceRow[2 * a] ? 1 : 0;
    ns.box.hi[a] = n[a] - (op.faceRow[2 * a + 1] ? 1 : 0);
    ns.box.par[a] = offset[a];
  }
  SignedBox own = ns.box;
  for (int a = 0; a < 3; a++)
    if (sharedRight[a])
      own.hi[a] = n[a] - 1;
  auto add = [&](int axis, int sa, int sb, int globalLines, double count) {
    CleaningNullSpace::Sum& s = ns.sums[ns.nSums++];
    s.box = own;
    if (axis >= 0)
      s.box.sgn[sa] = s.box.sgn[sb] = 1;
    s.axis = axis;
    s.lines = axis >= 0 ? std::max(own.hi[axis] - own.lo[axis], 0) : 1;
    s.local = ns.localSize;
    s.global = ns.globalSize;
    s.first = axis >= 0 ? offset[axis] + own.lo[axis] : 0;
    s.count = count;
    ns.localSize += s.lines;
    ns.globalSize += globalLines;
  };
  if (mean) {
    double unique = 1.0; // non-periodic node axis: N + 1 less 2 face rows
    for (int a = 0; a < 3; a++)
      unique *= nodes && !periodic[a] ? N[a] - 1 : N[a];
    add(-1, 0, 0, 1, unique);
  }
  bool even[3]; // the checkerboard axes
  for (int a = 0; a < 3; a++)
    even[a] = !op.flat[a] && ((periodic[a] && N[a] % 2 == 0) || oddWalls[a]);
  for (int c = 2; c >= 0; c--) { // the families (a, b), lines along c
    const int a = c == 0 ? 1 : 0, b = c == 2 ? 1 : 2;
    if (even[a] && even[b])
      add(c, a, b, N[c], (double)N[a] * N[b]);
  }
  return ns;
}

/** The terms of gpuSubSignedTerms from the global sums: coef receives the
 *  nCoef line coefficients of the rows of box (to be copied to d_coef, which
 *  the terms point into); returns the number of terms. */
static int cleaningNullTerms(const CleaningNullSpace& ns,
                             const std::vector<cudaSolverType>& global,
                             cudaSolverType* coef, int& nCoef,
                             const cudaSolverType* d_coef,
                             SignedTerm terms[kMaxSignedTerms]) {
  int nTerms = 0, nFamilies = 0;
  nCoef = 0;
  const cudaSolverType* xyLines = nullptr; // global line sums of family (x, y)
  for (int s = 0; s < ns.nSums; s++) {
    const CleaningNullSpace::Sum& sum = ns.sums[s];
    SignedTerm& t = terms[nTerms++];
    t = SignedTerm{
        sum.axis, {sum.box.sgn[0], sum.box.sgn[1], sum.box.sgn[2]}, nullptr, 0};
    if (sum.axis < 0) {
      t.value = global[sum.global] / sum.count;
      continue;
    }
    // a coefficient for every row of box along c, shared rows included
    const int c = sum.axis;
    t.line = d_coef + nCoef;
    for (int row = ns.box.lo[c]; row < ns.box.hi[c]; row++) {
      int g = ns.box.par[c] + row;
      if (ns.periodic[c])
        g %= ns.N[c];
      coef[nCoef++] = global[sum.global + g] / sum.count;
    }
    if (c == 2)
      xyLines = &global[sum.global];
    nFamilies++;
  }
  if (nFamilies == 3) { // the triple checkerboard, from the (x, y) family
    double triple = 0.0;
    for (int g = 0; g < ns.N[2]; g++)
      triple += (g % 2 ? -1.0 : 1.0) * xyLines[g];
    triple /= (double)ns.N[0] * ns.N[1] * ns.N[2];
    terms[nTerms++] =
        SignedTerm{-1, {1, 1, 1}, nullptr, (cudaSolverType)(-2.0 * triple)};
  }
  return nTerms;
}

void gpuCleaningNullSums(const cudaSolverType* d_b, const CleaningNullSpace& ns,
                         cudaSolverType* d_sums, cudaSolverType* h_sums,
                         std::vector<cudaSolverType>& global,
                         cudaStream_t stream) {
  for (int s = 0; s < ns.nSums; s++)
    gpuSignedBoxSums(d_b, ns.sums[s].box, ns.sums[s].axis,
                     d_sums + ns.sums[s].local, stream);
  cudaErrChk(cudaMemcpyAsync(h_sums, d_sums,
                             ns.localSize * sizeof(cudaSolverType),
                             cudaMemcpyDeviceToHost, stream));
  cudaErrChk(cudaStreamSynchronize(stream));
  global.assign(ns.globalSize, 0.0);
  for (int s = 0; s < ns.nSums; s++)
    for (int l = 0; l < ns.sums[s].lines; l++)
      global[ns.sums[s].global + ns.sums[s].first + l] =
          h_sums[ns.sums[s].local + l];
}

void gpuCleaningNullSubtract(cudaSolverType* d_b, const CleaningNullSpace& ns,
                             const std::vector<cudaSolverType>& global,
                             cudaSolverType* d_coef, cudaSolverType* h_coef,
                             cudaStream_t stream) {
  SignedTerm terms[kMaxSignedTerms];
  int nCoef = 0;
  const int nTerms =
      cleaningNullTerms(ns, global, h_coef, nCoef, d_coef, terms);
  if (nCoef > 0)
    cudaErrChk(cudaMemcpyAsync(d_coef, h_coef, nCoef * sizeof(cudaSolverType),
                               cudaMemcpyHostToDevice, stream));
  gpuSubSignedTerms(d_b, ns.box, terms, nTerms, stream);
  cudaErrChk(cudaStreamSynchronize(stream));
}

// =========================================================================
//  Kernels (the packed-row ones in 1D launches, see kRowBlock)
// =========================================================================

namespace {

// ---- the unpack and the image of a cleaning solve, once per FGMRES
// iteration: integer work per row fixed on the host (Lap27Unpack,
// Lap27ImageBox), the flat axes at compile time (no division, no offset, no
// face rule along them; a flat axis has the single row 0)

/** Whether the image reads its deep rows from the packed vector, and the
 *  unpack for the image leaves the inner rows out (Lap27Unpack::innerLo):
 *  with a flat axis (2D, 1D), where the unpack costs as much as the image
 *  and, with z flat, every point of f is a sector of its own.  In 3D the
 *  per-row source select costs the image about what the unpack saves, and
 *  f is read everywhere. */
__host__ __device__ constexpr bool lap27DeepRows(int FLAT) {
  return FLAT != 0 && FLAT != 7;
}

/** The unpack of the packed rows (i, j, k) = p, r = p / nk:
 *    q   = p + r qR + i qI + q0, the point (i+1, j+1, k+1) of f (the
 *          kernel takes f + q0);
 *    0   iff (unsigned)(x_a - zeroLo[a]) >= zeroN[a] on some axis (a zero
 *          face plane);
 *    src = p + s_a [x_a < copyLo[a]] - s_a [x_a >= copyHi[a]] (a copy face
 *          plane reads the next inner row);
 *    the rows with (unsigned)(x_a - innerLo[a]) < innerN[a] on every split
 *          axis are left out (innerN = 0: none);
 *    a twin axis: the same value in both node planes. */
struct Lap27Unpack {
  int count; // ni nj nk
  RowDiv nk, nj;
  int qR, qI, q0;
  int stride[3]; // of the packed rows: nj nk, nk, 1
  int zeroLo[3], copyLo[3], copyHi[3], innerLo[3];
  unsigned zeroN[3], innerN[3];
  int twin[2]; // offset of the second node plane: ny nz, nz (1 along z)
  bool face;   // some face plane
  bool nodes;  // the flat axes are twin axes
};

Lap27Unpack lap27Unpack(const Lap27Op& op, int ny, int nz, bool imageOnly) {
  Lap27Unpack u{};
  const int n[3] = {op.ni, op.nj, op.nk};
  u.count = op.ni * op.nj * op.nk;
  u.nk = op.divNk;
  u.nj = op.divNj;
  u.qR = nz - op.nk;
  u.qI = (ny - op.nj) * nz;
  u.q0 = ny * nz + nz + 1;
  u.stride[0] = op.nj * op.nk;
  u.stride[1] = op.nk;
  u.stride[2] = 1;
  u.twin[0] = ny * nz;
  u.twin[1] = nz;
  const bool inner = imageOnly && lap27DeepRows(lap27FlatMask(op));
  for (int a = 0; a < 3; a++) {
    // a face plane: 0 (zero) or the next inner row (faceCopy)
    const bool lo = op.faceRow[2 * a], hi = op.faceRow[2 * a + 1];
    const bool zLo = lo && !op.faceCopy[2 * a];
    const bool zHi = hi && !op.faceCopy[2 * a + 1];
    u.face = u.face || lo || hi;
    u.zeroLo[a] = zLo;
    u.zeroN[a] = (unsigned)(n[a] - zLo - zHi);
    u.copyLo[a] = op.faceCopy[2 * a] ? 1 : 0;
    u.copyHi[a] = op.faceCopy[2 * a + 1] ? n[a] - 1 : n[a];
    // the rows the image reads from f, or sends in the exchange, are the
    // ones within 2 rows (3 beyond a face plane) of the boundary along some
    // split axis (Lap27ImageBox::deepOff)
    u.innerLo[a] = 2 + lo;
    u.innerN[a] =
        inner ? (unsigned)std::max(n[a] - 2 - hi - u.innerLo[a], 0) : 0u;
  }
  u.nodes = op.twin[0] || op.twin[1] || op.twin[2];
  return u;
}

template <int FLAT, bool FACE, typename T>
__global__ void k_solver2PhysLap27(T* __restrict__ f, const T* __restrict__ vec,
                                   Lap27Unpack u) {
  const int p = blockIdx.x * blockDim.x + threadIdx.x;
  if (p >= u.count)
    return;
  int r, x[3];
  lap27RowOf<FLAT>(p, u.nk, u.nj, r, x);
  if constexpr (lap27DeepRows(FLAT)) {
    bool inner = true;
#pragma unroll
    for (int a = 0; a < 3; a++)
      if (!lap27Flat(FLAT, a))
        inner &= (unsigned)(x[a] - u.innerLo[a]) < u.innerN[a];
    if (inner)
      return;
  }
  T v;
  if constexpr (FACE) {
    bool zero = false;
    int src = p;
#pragma unroll
    for (int a = 0; a < 3; a++) {
      if (lap27Flat(FLAT, a))
        continue;
      zero = zero || (unsigned)(x[a] - u.zeroLo[a]) >= u.zeroN[a];
      if (x[a] < u.copyLo[a])
        src += u.stride[a];
      if (x[a] >= u.copyHi[a])
        src -= u.stride[a];
    }
    v = zero ? T(0) : vec[src];
  } else {
    v = vec[p];
  }
  const int q = p + r * u.qR + x[0] * u.qI; // f + q0
  f[q] = v;
  // the second node plane of every twin (flat node) axis
  if constexpr (FLAT != 0) {
    if (u.nodes) {
#pragma unroll
      for (int m = 1; m < 8; m++)
        if ((m & ~FLAT) == 0)
          f[q + (m & 1 ? u.twin[0] : 0) + (m & 2 ? u.twin[1] : 0) +
            (m & 4 ? 1 : 0)] = v;
    }
  }
}

/** A box of image rows without gaps (Lap27Rows::all, ::interior) in a 1D
 *  launch: thread t is the point (ti, tj, tk) of the box (k fastest),
 *  r = t / nk, ti = r / nj, and
 *    p = t + r pR + ti pI  (packed row less p0, the first one),
 *    q = t + r qR + ti qI  (point of f less q0, the first one);
 *  a face row:  (unsigned)(t_a + faceOff[a]) >= faceN[a] on some axis;
 *  a deep row:  (unsigned)(t_a + deepOff[a]) < deepN[a] on every split axis
 *  (lap27DeepRows): its stencil reads neither a ghost nor a face plane, so
 *  it reads vec (strides vI, vJ) instead of f (strides fI, fJ). */
struct Lap27ImageBox {
  int count;
  RowDiv nk, nj;
  int pR, pI, p0, qR, qI, q0;
  int faceOff[3], deepOff[3];
  unsigned faceN[3], deepN[3];
  int vI, vJ, fI, fJ;
};

/** The arrays of a box image from its first row: im + p0, vec + p0, and the
 *  9 lines (a, b) of the stencil in f, f + q0 + (a ny + b) nz.  On sm_80 a
 *  line is one IMAD.WIDE of q with a constant-bank address, and an offset c
 *  is an immediate of the load. */
template <typename T> struct Lap27ImageLines {
  T* im;
  const T* vec;
  const T* f[3][3];
};

/** faceOff/faceN for the packed row x_a = t_a + off[a] (face row: x_a below
 *  faceRow[2a] or above n - 1 - faceRow[2a+1]). */
void lap27FaceBounds(const Lap27Op& op, const int off[3], int faceOff[3],
                     unsigned faceN[3]) {
  const int n[3] = {op.ni, op.nj, op.nk};
  for (int a = 0; a < 3; a++) {
    const int lo = op.faceRow[2 * a], hi = op.faceRow[2 * a + 1];
    faceOff[a] = off[a] - lo;
    faceN[a] = (unsigned)(n[a] - lo - hi);
  }
}

/** The box of a list that is one box without gaps (false otherwise); face:
 *  whether a row of it can be a face row. */
bool lap27ImageBox(const Lap27Op& op, int ny, int nz, const PointList& L,
                   Lap27ImageBox& B, bool& face) {
  const PointBox& b = L.box[0];
  if (L.first[1] != L.count() || b.i.n1 != b.i.n || b.j.n1 != b.j.n ||
      b.k.n1 != b.k.n)
    return false;
  const int n[3] = {op.ni, op.nj, op.nk}, N[3] = {b.i.n, b.j.n, b.k.n};
  const int off[3] = {b.i.lo - 1, b.j.lo - 1, b.k.lo - 1};
  B.count = L.count();
  B.nk = RowDiv(N[2]);
  B.nj = RowDiv(N[1]);
  B.pR = op.nk - N[2];
  B.pI = (op.nj - N[1]) * op.nk;
  B.p0 = (off[0] * op.nj + off[1]) * op.nk + off[2];
  B.qR = nz - N[2];
  B.qI = (ny - N[1]) * nz;
  B.q0 = (b.i.lo * ny + b.j.lo) * nz + b.k.lo;
  B.vI = op.nj * op.nk;
  B.vJ = op.nk;
  B.fI = ny * nz;
  B.fJ = nz;
  lap27FaceBounds(op, off, B.faceOff, B.faceN);
  face = false;
  for (int a = 0; a < 3; a++) {
    const int lo = op.faceRow[2 * a], hi = op.faceRow[2 * a + 1];
    face = face || (lo && off[a] <= 0) || (hi && off[a] + N[a] - 1 >= n[a] - 1);
    // deep: x_a in [1 + lo, n - 2 - hi], the reads in [lo, n - 1 - hi]
    B.deepOff[a] = off[a] - 1 - lo;
    B.deepN[a] = (unsigned)std::max(n[a] - 2 - hi - lo, 0);
  }
  return true;
}

template <int FLAT, bool FACE, typename T>
__global__ void k_lap27BoxToKrylov(Lap27Op op, Lap27ImageBox B,
                                   Lap27ImageLines<T> L) {
  const int t = blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= B.count)
    return;
  int r, ta[3];
  lap27RowOf<FLAT>(t, B.nk, B.nj, r, ta);
  const int p = t + r * B.pR + ta[0] * B.pI;
  if constexpr (FACE) {
    bool face = false;
#pragma unroll
    for (int a = 0; a < 3; a++)
      if (!lap27Flat(FLAT, a))
        face = face || (unsigned)(ta[a] + B.faceOff[a]) >= B.faceN[a];
    if (face) {
      // ::__ldg, here and below: ROCm 6.4 declares its half-float __ldg in
      // an unnamed namespace, which hides the global overloads from the
      // unnamed namespace of this file
      L.im[p] = -op.diag * ::__ldg(L.vec + p);
      return;
    }
  }
  // FLAT = lap27FlatMask(op): lap27 reads no offset along a flat axis
  const int q = t + r * B.qR + ta[0] * B.qI;
  if constexpr (lap27DeepRows(FLAT)) {
    bool deep = true;
#pragma unroll
    for (int a = 0; a < 3; a++)
      if (!lap27Flat(FLAT, a))
        deep &= (unsigned)(ta[a] + B.deepOff[a]) < B.deepN[a];
    const T* src = deep ? L.vec : L.f[1][1];
    const int c = deep ? p : q, sx = deep ? B.vI : B.fI,
              sy = deep ? B.vJ : B.fJ;
    const auto v = [&](int a, int b, int d) {
      return ::__ldg(src + (c + a * sx + b * sy + d));
    };
    L.im[p] = lap27<FLAT, T>(v, op);
  } else {
    const auto v = [&](int a, int b, int c) {
      return ::__ldg(L.f[a + 1][b + 1] + (q + c));
    };
    L.im[p] = lap27<FLAT, T>(v, op);
  }
}

/** The image on a list of several boxes (Lap27Rows::shells): pointOf, the
 *  face test of Lap27ImageBox on the index of f (faceOff for off = -1). */
struct Lap27Faces {
  int off[3];
  unsigned n[3];
};

template <int FLAT, typename T>
__global__ void k_lap27ToKrylov(T* __restrict__ im, const T* __restrict__ f,
                                const T* __restrict__ vec, int ny, int nz,
                                Lap27Op op, PointList rows, Lap27Faces F) {
  int x[3];
  if (!pointOf(rows, x[0], x[1], x[2]))
    return;
#pragma unroll
  for (int a = 0; a < 3; a++)
    if (lap27Flat(FLAT, a))
      x[a] = 1; // the single row
  const int p = stencilIdx(x[0] - 1, x[1] - 1, x[2] - 1, op.nj, op.nk);
  bool face = false;
#pragma unroll
  for (int a = 0; a < 3; a++)
    if (!lap27Flat(FLAT, a))
      face = face || (unsigned)(x[a] + F.off[a]) >= F.n[a];
  if (face) {
    im[p] = -op.diag * vec[p];
    return;
  }
  // FLAT = lap27FlatMask(op): lap27 reads no offset along a flat axis
  const int q = stencilIdx(x[0], x[1], x[2], ny, nz);
  const auto v = [&](int a, int b, int c) {
    return f[q + (a * ny + b) * nz + c];
  };
  im[p] = lap27<FLAT, T>(v, op);
}

/** Whether the packed row (i, j, k) lies on a face plane of identity rows. */
__device__ __forceinline__ bool lap27FaceRow(const Lap27Op& op, int i, int j,
                                             int k) {
  return (op.faceRow[0] && i == 0) || (op.faceRow[1] && i == op.ni - 1) ||
         (op.faceRow[2] && j == 0) || (op.faceRow[3] && j == op.nj - 1) ||
         (op.faceRow[4] && k == 0) || (op.faceRow[5] && k == op.nk - 1);
}

template <typename T>
__global__ void k_divC2NToKrylov(T* __restrict__ b, const T* __restrict__ bx,
                                 const T* __restrict__ by,
                                 const T* __restrict__ bz, int nyc, int nzc,
                                 DiffScale<T> q, Lap27Op op) {
  const int p = blockIdx.x * blockDim.x + threadIdx.x;
  if (p >= op.ni * op.nj * op.nk)
    return;
  int r, x[3];
  lap27RowOf<0>(p, op.divNk, op.divNj, r, x);
  b[p] = lap27FaceRow(op, x[0], x[1], x[2])
             ? T(0)
             : divC2NAt(bx, by, bz, x[0] + 1, x[1] + 1, x[2] + 1, nyc, nzc, q);
}

template <typename T>
__global__ void
k_divN2CRhoToKrylov(T* __restrict__ b, const T* __restrict__ ex,
                    const T* __restrict__ ey, const T* __restrict__ ez,
                    const T* __restrict__ rho, T rhoScale, int nyn, int nzn,
                    DiffScale<T> q, Lap27Op op) {
  const int p = blockIdx.x * blockDim.x + threadIdx.x;
  if (p >= op.ni * op.nj * op.nk)
    return;
  int r, x[3];
  lap27RowOf<0>(p, op.divNk, op.divNj, r, x);
  const T div = divN2CAt(ex, ey, ez, x[0] + 1, x[1] + 1, x[2] + 1, nyn, nzn, q);
  b[p] =
      div + rhoScale *
                rho[stencilIdx(x[0] + 1, x[1] + 1, x[2] + 1, nyn - 1, nzn - 1)];
}

template <typename T>
__global__ void k_subGradN2C(T* __restrict__ bx, T* __restrict__ by,
                             T* __restrict__ bz, const T* __restrict__ psi,
                             int nxc, int nyc, int nzc, DiffScale<T> q) {
  const int i = blockIdx.z * blockDim.z + threadIdx.z + 1;
  const int j = blockIdx.y * blockDim.y + threadIdx.y + 1;
  const int k = blockIdx.x * blockDim.x + threadIdx.x + 1;
  if (i > nxc - 2 || j > nyc - 2 || k > nzc - 2)
    return;
  T gx, gy, gz;
  gradN2CAt(psi, i, j, k, nyc + 1, nzc + 1, q, gx, gy, gz);
  const int c = stencilIdx(i, j, k, nyc, nzc);
  bx[c] -= gx;
  by[c] -= gy;
  bz[c] -= gz;
}

template <typename T>
__global__ void k_subGradC2N(T* __restrict__ ex, T* __restrict__ ey,
                             T* __restrict__ ez, const T* __restrict__ phi,
                             int nxn, int nyn, int nzn, DiffScale<T> q) {
  const int i = blockIdx.z * blockDim.z + threadIdx.z + 1;
  const int j = blockIdx.y * blockDim.y + threadIdx.y + 1;
  const int k = blockIdx.x * blockDim.x + threadIdx.x + 1;
  if (i > nxn - 2 || j > nyn - 2 || k > nzn - 2)
    return;
  T gx, gy, gz;
  gradC2NAt(phi, i, j, k, nyn - 1, nzn - 1, q, gx, gy, gz);
  const int n = stencilIdx(i, j, k, nyn, nzn);
  ex[n] -= gx;
  ey[n] -= gy;
  ez[n] -= gz;
}

} // namespace

// =========================================================================
//  Host wrappers
// =========================================================================

void gpuSolver2PhysLap27(cudaSolverType* f, const cudaSolverType* vec, int ny,
                         int nz, const Lap27Op& op, cudaStream_t stream,
                         bool imageOnly) {
  const Lap27Unpack u = lap27Unpack(op, ny, nz, imageOnly);
  lap27FlatDispatch(op, [&](auto flat) {
    constexpr int F = decltype(flat)::value;
    if (u.face)
      k_solver2PhysLap27<F, true>
          <<<rowBlocks(op), kRowBlock, 0, stream>>>(f + u.q0, vec, u);
    else
      k_solver2PhysLap27<F, false>
          <<<rowBlocks(op), kRowBlock, 0, stream>>>(f + u.q0, vec, u);
  });
  cudaErrChk(cudaGetLastError());
}

void gpuLap27ToKrylov(cudaSolverType* im, const cudaSolverType* f,
                      const cudaSolverType* vec, int ny, int nz,
                      const Lap27Op& op, const PointList& rows,
                      cudaStream_t stream) {
  if (rows.empty())
    return;
  Lap27ImageBox B;
  bool face;
  if (lap27ImageBox(op, ny, nz, rows, B, face)) {
    Lap27ImageLines<cudaSolverType> lines;
    lines.im = im + B.p0;
    lines.vec = vec + B.p0;
    for (int a = -1; a <= 1; a++)
      for (int b = -1; b <= 1; b++) // from the first row: in the array
        lines.f[a + 1][b + 1] = f + (B.q0 + (a * ny + b) * nz);
    const int blocks = (B.count + kRowBlock - 1) / kRowBlock;
    lap27FlatDispatch(op, [&](auto flat) {
      constexpr int F = decltype(flat)::value;
      if (face)
        k_lap27BoxToKrylov<F, true>
            <<<blocks, kRowBlock, 0, stream>>>(op, B, lines);
      else
        k_lap27BoxToKrylov<F, false>
            <<<blocks, kRowBlock, 0, stream>>>(op, B, lines);
    });
  } else {
    Lap27Faces F;
    const int off[3] = {-1, -1, -1};
    lap27FaceBounds(op, off, F.off, F.n);
    const PointLaunch L = pointLaunch(rows);
    lap27FlatDispatch(op, [&](auto flat) {
      k_lap27ToKrylov<decltype(flat)::value>
          <<<L.grid, L.block, 0, stream>>>(im, f, vec, ny, nz, op, rows, F);
    });
  }
  cudaErrChk(cudaGetLastError());
}

void gpuDivC2NToKrylov(cudaSolverType* b, const cudaSolverType* bx,
                       const cudaSolverType* by, const cudaSolverType* bz,
                       int nyc, int nzc, const cudaSolverType invd[3],
                       const Lap27Op& op, cudaStream_t stream) {
  k_divC2NToKrylov<<<rowBlocks(op), kRowBlock, 0, stream>>>(
      b, bx, by, bz, nyc, nzc, diffScale(invd[0], invd[1], invd[2]), op);
  cudaErrChk(cudaGetLastError());
}

void gpuDivN2CRhoToKrylov(cudaSolverType* b, const cudaSolverType* ex,
                          const cudaSolverType* ey, const cudaSolverType* ez,
                          const cudaSolverType* rho, cudaSolverType rhoScale,
                          int nyn, int nzn, const cudaSolverType invd[3],
                          const Lap27Op& op, cudaStream_t stream) {
  k_divN2CRhoToKrylov<<<rowBlocks(op), kRowBlock, 0, stream>>>(
      b, ex, ey, ez, rho, rhoScale, nyn, nzn,
      diffScale(invd[0], invd[1], invd[2]), op);
  cudaErrChk(cudaGetLastError());
}

void gpuSubGradN2C(cudaSolverType* bx, cudaSolverType* by, cudaSolverType* bz,
                   const cudaSolverType* psi, int nxc, int nyc, int nzc,
                   const cudaSolverType invd[3], cudaStream_t stream) {
  const PointLaunch L = pointLaunch(nxc - 2, nyc - 2, nzc - 2);
  k_subGradN2C<<<L.grid, L.block, 0, stream>>>(
      bx, by, bz, psi, nxc, nyc, nzc, diffScale(invd[0], invd[1], invd[2]));
  cudaErrChk(cudaGetLastError());
}

void gpuSubGradC2N(cudaSolverType* ex, cudaSolverType* ey, cudaSolverType* ez,
                   const cudaSolverType* phi, int nxn, int nyn, int nzn,
                   const cudaSolverType invd[3], cudaStream_t stream) {
  const PointLaunch L = pointLaunch(nxn - 2, nyn - 2, nzn - 2);
  k_subGradC2N<<<L.grid, L.block, 0, stream>>>(
      ex, ey, ez, phi, nxn, nyn, nzn, diffScale(invd[0], invd[1], invd[2]));
  cudaErrChk(cudaGetLastError());
}
