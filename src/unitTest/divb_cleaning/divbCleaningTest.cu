/* Unit tests of the divergence cleanings and of the B absorbing layer on the
 * GPU kernels, one --case each:
 *   sal-tangential      gpuSALBlendLayers3: the normal component bit-identical,
 *                       the tangential ones blended, nothing outside touched.
 *   adjoint             gradC2N = -divN2Cᵀ with zero ghost centres, not with
 *                       copies.
 *   lap27-image         gpuSolver2PhysLap27 + gpuLap27ToKrylov vs the stencils
 *                       and hostLap27 (cut, zero, copy faces), overlap split,
 *                       twin unpack.
 *   lap27-local         fused Chebyshev steps of Lap27Op; symmetry and
 *                       lap27EigBounds vs dense eigenvalues.
 *   chebyshev-cleaning  gpuChebyshevCleaning vs a host recurrence, within the
 *                       Chebyshev residual bound; its degenerate branches.
 *   divb-projection     one div(B) cleaning: divC2N(Bc) removed, conductor
 *                       B_n kept, curlN2C(E) unchanged.
 *   dive-projection     the div(E) right-hand side; divN2C(E) = 4 pi rho after
 *                       one cleaning.
 *   null-space          gpuCleaningNullSums/Subtract on emulated ranks: no
 *                       null component left, only null-space parts removed.
 *   krylov-kernels      gpuBatchedDotNorm beyond 32 vectors; gpuSubCombination
 *                       vs gpuAddscale.
 *   poisson-local       7-point PoissonLocalOperator steps (not used) vs the
 *                       composed reference, every face factor.
 *   poisson-image       gpuLapC2CToKrylov vs host; no edge ghost read.
 * Layout (i ny + j) nz + k, ghosts at 0 and n-1; NaN marks values never to
 * be read or left unwritten, and fails every comparison. */
#include "GPUBlas.cuh"
#include "GPUChebyshev.cuh"
#include "GPUCleaning.cuh"
#include "GPUPhysicsKernels.cuh"
#include "GPUStencils.cuh"
#include "cudaTypeDef.cuh"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <random>
#include <string>
#include <utility>
#include <vector>

using T = cudaSolverType;
using Vec = std::vector<T>;

constexpr double kH[3] = {0.1, 0.13, 0.07}; // grid spacings
constexpr double kInf = std::numeric_limits<double>::infinity();
constexpr T kNaN = std::numeric_limits<T>::quiet_NaN();

static int failures = 0;
static void check(bool ok, const std::string& what, double value) {
  std::printf("  [%s] %s: %.3e\n", ok ? "ok" : "FAIL", what.c_str(), value);
  if (!ok)
    failures++;
}

struct Dev {
  T* p = nullptr;
  size_t n = 0;
  explicit Dev(const Vec& h) : n(h.size()) {
    cudaErrChk(cudaMalloc(&p, n * sizeof(T)));
    cudaErrChk(cudaMemcpy(p, h.data(), n * sizeof(T), cudaMemcpyHostToDevice));
  }
  Dev(const Dev&) = delete;
  Dev& operator=(const Dev&) = delete;
  Vec get() const {
    Vec h(n);
    cudaErrChk(cudaDeviceSynchronize());
    cudaErrChk(cudaMemcpy(h.data(), p, n * sizeof(T), cudaMemcpyDeviceToHost));
    return h;
  }
  ~Dev() { cudaFree(p); }
};

static Vec randomVec(size_t n, std::mt19937_64& g) {
  std::normal_distribution<double> d(0.0, 1.0);
  Vec v(n);
  for (auto& x : v)
    x = d(g);
  return v;
}

/** m = max(m, x); a NaN x makes m +infinity (std::max would drop it). */
static void keepMax(double& m, double x) {
  m = std::isnan(x) ? kInf : std::max(m, x);
}

/** max |v|; +infinity if an entry is NaN or infinite. */
static double maxAbs(const Vec& v) {
  double m = 0;
  for (const T x : v) {
    if (!std::isfinite(x))
      return kInf;
    m = std::max(m, (double)std::fabs(x));
  }
  return m;
}

static Vec minus(const Vec& a, const Vec& b) {
  Vec d(a.size());
  for (size_t q = 0; q < a.size(); q++)
    d[q] = a[q] - b[q];
  return d;
}

static Vec negated(Vec v) {
  for (auto& q : v)
    q = -q;
  return v;
}

/** max |a - b| / max |b|, or max |a - b| when b is zero; +infinity if an
 *  entry of a or b is NaN or infinite. */
static double relDiff(const Vec& a, const Vec& b) {
  const double d = maxAbs(minus(a, b)), m = maxAbs(b);
  return m > 0 && m < kInf ? d / m : d;
}

static double dot(const Vec& u, const Vec& v) {
  double s = 0;
  for (size_t q = 0; q < u.size(); q++)
    s += (double)u[q] * v[q];
  return s;
}

static double norm2(const Vec& v) { return std::sqrt(dot(v, v)); }

// ---------------------------------------------------------------------------
static void testSalTangential() {
  std::printf("sal-tangential\n");
  const int n[3] = {14, 13, 12}, nsal = 3;
  const T tgt[3] = {0.3, -0.7, 1.1};
  const T invNL = T(1) / nsal;
  std::mt19937_64 gen(1);
  const size_t N = (size_t)n[0] * n[1] * n[2];
  for (int skipMode = 0; skipMode < 2; skipMode++)
    for (int dir = 0; dir < 3; dir++)
      for (int asc = 0; asc < 2; asc++) {
        Vec h[3] = {randomVec(N, gen), randomVec(N, gen), randomVec(N, gen)};
        Dev d[3] = {Dev(h[0]), Dev(h[1]), Dev(h[2])};
        // the calls of gpuOpenBoundaryInflowB: left face layers 0..nsal
        // ascending, right face layers n-nsal-1..n-1 descending
        const int lo = asc ? 0 : n[dir] - nsal - 1;
        const int hi = asc ? nsal : n[dir] - 1;
        const int skip = skipMode ? dir : -1;
        gpuSALBlendLayers3(d[0].p, d[1].p, d[2].p, tgt[0], tgt[1], tgt[2], dir,
                           lo, hi, invNL, asc, n[0], n[1], n[2], 0, skip);
        Vec blend, normal, outside; // result minus expected value
        Vec r[3] = {d[0].get(), d[1].get(), d[2].get()};
        for (int i = 0; i < n[0]; i++)
          for (int j = 0; j < n[1]; j++)
            for (int k = 0; k < n[2]; k++) {
              const int idx[3] = {i, j, k};
              const int layer = idx[dir];
              const size_t a = ((size_t)i * n[1] + j) * n[2] + k;
              const bool in = layer >= lo && layer <= hi;
              const T sal =
                  asc ? (T)(layer - lo) * invNL : (T)(hi - layer) * invNL;
              for (int c = 0; c < 3; c++) {
                if (!in)
                  outside.push_back(r[c][a] - h[c][a]);
                else if (c == skip)
                  normal.push_back(r[c][a] - h[c][a]);
                else
                  blend.push_back(r[c][a] -
                                  (h[c][a] * sal + tgt[c] * (T(1) - sal)));
              }
            }
        const double errBlend = maxAbs(blend), errNormal = maxAbs(normal),
                     errOutside = maxAbs(outside);
        char pre[64];
        std::snprintf(pre, sizeof(pre), "dir %d %s face, skipComp %+d: ", dir,
                      asc ? "left" : "right", skip);
        const std::string p = pre;
        check(errBlend < 1e-12, p + "blended components", errBlend);
        if (skipMode)
          check(errNormal == 0, p + "normal component bit-identical",
                errNormal);
        check(errOutside == 0, p + "outside the layers untouched", errOutside);
      }
}

// ---------------------------------------------------------------------------
struct Grid {
  int nc[3], nn[3];
  T h[3];
  size_t C() const { return (size_t)nc[0] * nc[1] * nc[2]; }
  size_t N() const { return (size_t)nn[0] * nn[1] * nn[2]; }
  size_t c(int i, int j, int k) const {
    return ((size_t)i * nc[1] + j) * nc[2] + k;
  }
  size_t n(int i, int j, int k) const {
    return ((size_t)i * nn[1] + j) * nn[2] + k;
  }
};

static Grid makeGrid(int mx, int my, int mz) { // m = physical cells
  Grid g;
  const int m[3] = {mx, my, mz};
  for (int d = 0; d < 3; d++) {
    g.nc[d] = m[d] + 2;
    g.nn[d] = m[d] + 3;
    g.h[d] = kH[d];
  }
  return g;
}

/** Ghost centres of v: the nearest interior cell, or 0 if the ghost lies
 *  beyond an axis without copy. */
static void setGhosts(const Grid& g, Vec& v, const bool copy[3]) {
  for (int i = 0; i < g.nc[0]; i++)
    for (int j = 0; j < g.nc[1]; j++)
      for (int k = 0; k < g.nc[2]; k++) {
        const int t[3] = {i, j, k};
        int s[3];
        bool ghost = false, zero = false;
        for (int a = 0; a < 3; a++) {
          s[a] = std::min(std::max(t[a], 1), g.nc[a] - 2);
          ghost = ghost || s[a] != t[a];
          zero = zero || (s[a] != t[a] && !copy[a]);
        }
        if (ghost)
          v[g.c(i, j, k)] = zero ? T(0) : v[g.c(s[0], s[1], s[2])];
      }
}

static Vec divN2C(const Grid& g, const Vec v[3]) {
  Dev vx(v[0]), vy(v[1]), vz(v[2]), d(Vec(g.C(), 0));
  gpuDivN2C(d.p, vx.p, vy.p, vz.p, g.nc[0], g.nc[1], g.nc[2], 1 / g.h[0],
            1 / g.h[1], 1 / g.h[2]);
  return d.get();
}

static void gradC2N(const Grid& g, const Vec& psi, Vec out[3]) {
  Dev p(psi), gx(Vec(g.N(), 0)), gy(Vec(g.N(), 0)), gz(Vec(g.N(), 0));
  gpuGradC2N(gx.p, gy.p, gz.p, p.p, g.nn[0], g.nn[1], g.nn[2], 1 / g.h[0],
             1 / g.h[1], 1 / g.h[2]);
  out[0] = gx.get();
  out[1] = gy.get();
  out[2] = gz.get();
}

static Vec divC2N(const Grid& g, const Vec v[3]) {
  Dev vx(v[0]), vy(v[1]), vz(v[2]), d(Vec(g.N(), 0));
  gpuDivC2N(d.p, vx.p, vy.p, vz.p, g.nn[0], g.nn[1], g.nn[2], 1 / g.h[0],
            1 / g.h[1], 1 / g.h[2]);
  return d.get();
}

static void gradN2C(const Grid& g, const Vec& psi, Vec out[3]) {
  Dev p(psi), gx(Vec(g.C(), 0)), gy(Vec(g.C(), 0)), gz(Vec(g.C(), 0));
  gpuGradN2C(gx.p, gy.p, gz.p, p.p, g.nc[0], g.nc[1], g.nc[2], 1 / g.h[0],
             1 / g.h[1], 1 / g.h[2]);
  out[0] = gx.get();
  out[1] = gy.get();
  out[2] = gz.get();
}

static void testAdjoint() {
  std::printf("adjoint\n");
  const Grid g = makeGrid(10, 9, 8);
  std::mt19937_64 gen(2);
  Vec v[3] = {randomVec(g.N(), gen), randomVec(g.N(), gen),
              randomVec(g.N(), gen)};
  Vec psi = randomVec(g.C(), gen);
  const Vec d = divN2C(g, v);
  // sum over comps components of <a, b> on the interior points of extents n
  auto dotInterior = [](const Vec* a, const Vec* b, int comps, const int n[3]) {
    double s = 0;
    for (int c = 0; c < comps; c++)
      for (int i = 1; i < n[0] - 1; i++)
        for (int j = 1; j < n[1] - 1; j++)
          for (int k = 1; k < n[2] - 1; k++) {
            const size_t q = ((size_t)i * n[1] + j) * n[2] + k;
            s += (double)a[c][q] * b[c][q];
          }
    return s;
  };
  for (const bool copy : {false, true}) {
    const bool copy3[3] = {copy, copy, copy};
    setGhosts(g, psi, copy3);
    Vec gr[3];
    gradC2N(g, psi, gr);
    const double s1 = dotInterior(&d, &psi, 1, g.nc),
                 s2 = dotInterior(v, gr, 3, g.nn);
    const double rel = std::fabs(s1 + s2) / (std::fabs(s1) + std::fabs(s2));
    if (!copy)
      check(rel < 1e-12, "zero ghosts: |<D v, psi> + <v, G psi>| / scale", rel);
    else
      check(rel > 1e-3, "copy ghosts: not an adjoint pair", rel);
  }
}

// ---------------------------------------------------------------------------
// The 27-point operators of the cleaning solves (GPUCleaning.cuh)

/** Lap27Op for the packed rows n with the given faces. */
static Lap27Op makeOp(const int n[3], const bool flat[3], const bool row[6],
                      const bool copy[6], bool nodes) {
  const double invh2[3] = {1 / (kH[0] * kH[0]), 1 / (kH[1] * kH[1]),
                           1 / (kH[2] * kH[2])};
  return lap27Op(n, invh2, flat, row, copy, nodes);
}

/** The faces of a base-3 code, digit f for face f: 0 a cut (zero beyond), 1
 *  a zero face, 2 a copy face (identity rows); false if a flat axis gets a
 *  face or an axis keeps no row that is not a face row. */
static bool faceCode(int code, const int n[3], const bool flat[3], bool row[6],
                     bool copy[6]) {
  for (int f = 0; f < 6; f++, code /= 3) {
    row[f] = code % 3 > 0;
    copy[f] = code % 3 == 2;
    if (flat[f / 2] && row[f])
      return false;
  }
  for (int a = 0; a < 3; a++)
    if (n[a] - row[2 * a] - row[2 * a + 1] < 1)
      return false;
  return true;
}

/** Whether the packed row (i, j, k) lies on a face plane of identity rows. */
static bool hostFaceRow(const Lap27Op& op, int i, int j, int k) {
  return (op.faceRow[0] && i == 0) || (op.faceRow[1] && i == op.ni - 1) ||
         (op.faceRow[2] && j == 0) || (op.faceRow[3] && j == op.nj - 1) ||
         (op.faceRow[4] && k == 0) || (op.faceRow[5] && k == op.nk - 1);
}

/** +L x by the 27 coefficients, independent of the GPU code: beyond the
 *  rows 0 (a cut, or the zero ghost of a cell face) or, along an axis of
 *  wrap, the periodic image; a face-plane neighbour 0 or the reading row; a
 *  flat axis the row; -diag x on a face row. */
static Vec hostLap27(const Vec& x, const Lap27Op& op,
                     const bool* wrap = nullptr) {
  const int n[3] = {op.ni, op.nj, op.nk};
  const double m[3] = {0.25, 0.5, 0.25}, d2[3] = {1, -2, 1};
  Vec r(x.size());
  auto P = [&](int i, int j, int k) {
    return ((size_t)i * op.nj + j) * op.nk + k;
  };
  for (int i = 0; i < n[0]; i++)
    for (int j = 0; j < n[1]; j++)
      for (int k = 0; k < n[2]; k++) {
        if (hostFaceRow(op, i, j, k)) {
          r[P(i, j, k)] = -op.diag * x[P(i, j, k)];
          continue;
        }
        const int idx[3] = {i, j, k};
        double sum = 0;
        for (int oa = -1; oa <= 1; oa++)
          for (int ob = -1; ob <= 1; ob++)
            for (int oc = -1; oc <= 1; oc++) {
              const int o[3] = {oa, ob, oc};
              int t[3];
              bool zero = false;
              for (int a = 0; a < 3; a++) {
                t[a] = idx[a] + o[a];
                if (op.flat[a] || o[a] == 0) {
                  t[a] = idx[a];
                } else if (t[a] < 0 || t[a] > n[a] - 1) {
                  if (wrap && wrap[a])
                    t[a] = (t[a] + n[a]) % n[a];
                  else
                    zero = true;
                } else {
                  const int face = 2 * a + (o[a] > 0);
                  const bool plane =
                      op.faceRow[face] && t[a] == (o[a] > 0 ? n[a] - 1 : 0);
                  if (plane && op.faceCopy[face])
                    t[a] = idx[a];
                  else if (plane)
                    zero = true;
                }
              }
              if (zero)
                continue;
              double c = 0;
              for (int a = 0; a < 3; a++)
                c += op.invh2[a] * d2[o[a] + 1] * m[o[(a + 1) % 3] + 1] *
                     m[o[(a + 2) % 3] + 1];
              sum += c * x[P(t[0], t[1], t[2])];
            }
        r[P(i, j, k)] = sum;
      }
  return r;
}

/** The packed rows of op from the points 1.. of an array of extents ext;
 *  -diag x on a face row. */
static Vec packRows(const Vec& u, const int ext[3], const Lap27Op& op,
                    const Vec& x) {
  Vec r(x.size());
  for (int i = 0; i < op.ni; i++)
    for (int j = 0; j < op.nj; j++)
      for (int k = 0; k < op.nk; k++) {
        const size_t p = ((size_t)i * op.nj + j) * op.nk + k;
        r[p] = hostFaceRow(op, i, j, k)
                   ? -op.diag * x[p]
                   : u[((size_t)(i + 1) * ext[1] + j + 1) * ext[2] + k + 1];
      }
  return r;
}

/** The rows of the image of op as points of the unpacked array, extents
 *  n + 2 without the twin plane (lap27Rows). */
static Lap27Rows imageRows(const Lap27Op& op, bool overlap) {
  const int ext[3] = {op.ni + 2, op.nj + 2, op.nk + 2};
  return lap27Rows(ext, op.flat, overlap);
}

/** The rows gpuImage computes: all of them; the interior, then the shells
 *  of the overlapped exchange (all when lap27Rows does not split them); the
 *  interior only. */
enum class ImageRows { all, split, interior };

/** The GPU image: unpack into an array of extents n + 2 (+ 1 along a twin
 *  axis) of NaN, but 0 beyond a cut as in hostLap27 (NaN there too for the
 *  interior rows, which read no ghost), then gpuLap27ToKrylov over the rows
 *  into an image of NaN. */
static Vec gpuImage(const Vec& x, const Lap27Op& op, ImageRows which) {
  const int ext[3] = {op.ni + 2 + op.twin[0], op.nj + 2 + op.twin[1],
                      op.nk + 2 + op.twin[2]};
  Vec u((size_t)ext[0] * ext[1] * ext[2], kNaN);
  for (int i = 0; i < ext[0]; i++)
    for (int j = 0; j < ext[1]; j++)
      for (int k = 0; k < ext[2]; k++) {
        const int t[3] = {i, j, k};
        bool ghost = false, cut = which != ImageRows::interior;
        for (int a = 0; a < 3; a++) {
          const int side = t[a] == ext[a] - 1;
          if (t[a] == 0 || side) {
            ghost = true;
            cut = cut && !op.flat[a] && !op.faceRow[2 * a + side];
          }
        }
        if (ghost && cut)
          u[((size_t)i * ext[1] + j) * ext[2] + k] = 0;
      }
  Dev f(u), dx(x), im(Vec(x.size(), kNaN));
  gpuSolver2PhysLap27(f.p, dx.p, ext[1], ext[2], op, 0);
  const Lap27Rows rows = imageRows(op, which != ImageRows::all);
  if (!rows.split) {
    gpuLap27ToKrylov(im.p, f.p, dx.p, ext[1], ext[2], op, rows.all, 0);
  } else {
    gpuLap27ToKrylov(im.p, f.p, dx.p, ext[1], ext[2], op, rows.interior, 0);
    if (which == ImageRows::split)
      gpuLap27ToKrylov(im.p, f.p, dx.p, ext[1], ext[2], op, rows.shells, 0);
  }
  return im.get();
}

static void testLap27Image() {
  std::printf("lap27-image\n");
  std::mt19937_64 gen(7);
  // the overlap split against the image im over all rows: the same image, a
  // partition of the rows, and interior rows that read no ghost
  double errSplit = 0;
  int splits = 0, badCount = 0, badInterior = 0;
  auto checkSplit = [&](const Vec& x, const Lap27Op& op, const Vec& im) {
    keepMax(errSplit, relDiff(gpuImage(x, op, ImageRows::split), im));
    const Lap27Rows rows = imageRows(op, true);
    if (!rows.split)
      return;
    splits++;
    badCount += rows.interior.count() + rows.shells.count() != rows.all.count();
    const Vec in = gpuImage(x, op, ImageRows::interior);
    for (int p = 0; p < rows.interior.count(); p++) {
      int i, j, k;
      rows.interior.at(p, i, j, k);
      const size_t q = ((size_t)(i - 1) * op.nj + j - 1) * op.nk + k - 1;
      badInterior += !(in[q] == im[q]);
    }
  };

  // cells (div(E)): divN2C(gradC2N phi), zero ghosts (copies along a flat
  // axis), against the stencils
  double errCell = 0, errCellHost = 0;
  const int cellShapes[][3] = {{6, 5, 4}, {3, 7, 1}, {1, 4, 6},
                               {9, 3, 5}, {5, 1, 4}, {7, 1, 1}};
  for (const auto& sh : cellShapes) {
    const bool flat[3] = {sh[0] == 1, sh[1] == 1, sh[2] == 1}, none[6] = {};
    const Lap27Op op = makeOp(sh, flat, none, none, false);
    const Grid g = makeGrid(sh[0], sh[1], sh[2]);
    const Vec x = randomVec((size_t)sh[0] * sh[1] * sh[2], gen);
    Vec phi(g.C(), 0);
    {
      Dev dx(x), dp(phi);
      gpuSolver2Phys1(dp.p, dx.p, g.nc[0], g.nc[1], g.nc[2], 0);
      phi = dp.get();
    }
    setGhosts(g, phi, flat);
    Vec gr[3];
    gradC2N(g, phi, gr);
    const Vec im = gpuImage(x, op, ImageRows::all);
    keepMax(errCell, relDiff(im, packRows(divN2C(g, gr), g.nc, op, x)));
    keepMax(errCellHost, relDiff(im, hostLap27(x, op)));
    checkSplit(x, op, im);
  }
  check(errCell < 1e-12, "div(E) image vs divN2C(gradC2N phi) (stencils)",
        errCell);
  check(errCellHost < 1e-12,
        "div(E) image vs host 27 coefficients (NaN flat ghosts: none read)",
        errCellHost);

  // nodes (div(B)): every face cut, zero or copy (codes in steps of 4),
  // against hostLap27; without a cut also against divC2N(gradN2C psi) with
  // identity face rows
  double errNode = 0, errNodeHost = 0;
  int combos = 0, physical = 0;
  const int nodeShapes[][3] = {{9, 8, 7}, {5, 4, 1}, {4, 8, 3}};
  for (const auto& sh : nodeShapes) {
    const bool flat[3] = {false, false, sh[2] == 1};
    for (int code = 0; code < 729; code += 4) {
      bool row[6], copy[6], cut = false;
      if (!faceCode(code, sh, flat, row, copy))
        continue;
      combos++;
      const Lap27Op op = makeOp(sh, flat, row, copy, true);
      const Vec x = randomVec((size_t)sh[0] * sh[1] * sh[2], gen);
      const Vec im = gpuImage(x, op, ImageRows::all);
      keepMax(errNodeHost, relDiff(im, hostLap27(x, op)));
      checkSplit(x, op, im);
      for (int f = 0; f < 6; f++)
        cut = cut || (!row[f] && !flat[f / 2]);
      if (cut)
        continue;
      physical++;
      // the extended psi (face planes by the rules, x then y then z; both
      // node planes of the flat z), gradN2C on the cells (ghost cells of a
      // flat axis copied), divC2N on the nodes
      const int nn[3] = {sh[0] + 2, sh[1] + 2, sh[2] + 2 + flat[2]};
      const Grid g = makeGrid(nn[0] - 3, nn[1] - 3, nn[2] - 3);
      Vec psi(g.N(), 0);
      for (int i = 0; i < sh[0]; i++)
        for (int j = 0; j < sh[1]; j++)
          for (int k = 0; k < nn[2] - 2; k++)
            psi[g.n(i + 1, j + 1, k + 1)] =
                x[((size_t)i * sh[1] + j) * sh[2] + std::min(k, sh[2] - 1)];
      for (int a = 0; a < 3; a++)
        for (int side = 0; side < 2; side++) {
          if (!row[2 * a + side])
            continue;
          const int plane = side ? nn[a] - 2 : 1, inner = side ? nn[a] - 3 : 2;
          for (int i = 0; i < nn[0]; i++)
            for (int j = 0; j < nn[1]; j++)
              for (int k = 0; k < nn[2]; k++) {
                int t[3] = {i, j, k};
                if (t[a] != plane)
                  continue;
                int s[3] = {i, j, k};
                s[a] = inner;
                psi[g.n(t[0], t[1], t[2])] =
                    copy[2 * a + side] ? psi[g.n(s[0], s[1], s[2])] : T(0);
              }
        }
      Vec gc[3];
      gradN2C(g, psi, gc);
      for (int c = 0; c < 3; c++)
        setGhosts(g, gc[c], flat);
      keepMax(errNode, relDiff(im, packRows(divC2N(g, gc), g.nn, op, x)));
    }
  }
  std::printf("  %d node face codes (%d without a cut), %d split images\n",
              combos, physical, splits);
  check(errNode < 1e-12, "div(B) image vs divC2N(gradN2C psi) (stencils)",
        errNode);
  check(errNodeHost < 1e-12,
        "div(B) image vs host 27 coefficients (NaN ghosts: none read)",
        errNodeHost);
  check(errSplit == 0, "overlap split (interior, then shells) vs all rows",
        errSplit);
  check(badCount == 0, "lap27Rows: interior + shells count = all rows",
        badCount);
  check(badInterior == 0,
        "interior rows with NaN ghosts vs the full image (rows off)",
        badInterior);

  // the twin unpack: the row plane of a flat node axis a into both of its
  // node planes, one stride of a apart
  int unwritten = 0;
  for (int a = 0; a < 3; a++) {
    int sh[3] = {4, 3, 5}, ext[3];
    bool flat[3] = {};
    sh[a] = 1;
    flat[a] = true;
    const bool none[6] = {};
    const Lap27Op op = makeOp(sh, flat, none, none, true);
    for (int d = 0; d < 3; d++)
      ext[d] = sh[d] + 2 + (d == a);
    const size_t stride[3] = {(size_t)ext[1] * ext[2], (size_t)ext[2], 1};
    const Vec x = randomVec((size_t)sh[0] * sh[1] * sh[2], gen);
    Dev f(Vec((size_t)ext[0] * ext[1] * ext[2], kNaN)), dx(x);
    gpuSolver2PhysLap27(f.p, dx.p, ext[1], ext[2], op, 0);
    const Vec u = f.get();
    for (int i = 0; i < sh[0]; i++)
      for (int j = 0; j < sh[1]; j++)
        for (int k = 0; k < sh[2]; k++) {
          const T v = x[((size_t)i * sh[1] + j) * sh[2] + k];
          const size_t q = ((size_t)(i + 1) * ext[1] + j + 1) * ext[2] + k + 1;
          unwritten += !(u[q] == v && u[q + stride[a]] == v);
        }
  }
  check(unwritten == 0,
        "twin unpack: rows not in both node planes of a flat axis", unwritten);
}

/** Errors of the fused Chebyshev steps (gpuChebyshevPoissonStep1/N with
 *  the local operator op) against the generic steps fed by the reference
 *  -A b and -A y: step 1, its z = b/theta, step N in place, and the negated
 *  output of a last step. */
struct StepErrors {
  double step1 = 0, z = 0, stepN = 0, neg = 0;
};
constexpr T kTheta = 2.5, kDelta = 1.5, kRho = 0.4, kSigma = kTheta / kDelta,
            kRhoOld = 0.3;

template <class Op>
static void checkFusedSteps(const Op& op, const Vec& b, const Vec& y,
                            const Vec& z, const Vec& Ab, const Vec& Ay,
                            StepErrors& e) {
  const int n = (int)b.size();
  Dev db(b), dy(y), dz(z), dAb(Ab), dAy(Ay);
  Dev yRef(Vec(n, 0)), zRef(Vec(n, 0)), wRef(Vec(n, 0));
  gpuChebyshevStep1(yRef.p, zRef.p, db.p, dAb.p, kTheta, kDelta, kRho, n, 0);
  gpuChebyshevStepN(wRef.p, dy.p, dz.p, db.p, dAy.p, kDelta, kSigma, kRho,
                    kRhoOld, n, 0);
  Dev out1(Vec(n, 0)), z1(Vec(n, 0)), outNeg(Vec(n, 0)), z1b(Vec(n, 0));
  gpuChebyshevPoissonStep1(out1.p, z1.p, db.p, op, kTheta, kDelta, kRho, 1.0,
                           0);
  gpuChebyshevPoissonStep1(outNeg.p, z1b.p, db.p, op, kTheta, kDelta, kRho,
                           -1.0, 0);
  Dev zInPlace(z), outN(Vec(n, 0));
  gpuChebyshevPoissonStepN(zInPlace.p, dy.p, zInPlace.p, db.p, op, kDelta,
                           kSigma, kRho, kRhoOld, 1.0, 0);
  gpuChebyshevPoissonStepN(outN.p, dy.p, dz.p, db.p, op, kDelta, kSigma, kRho,
                           kRhoOld, -1.0, 0);
  const Vec y1 = yRef.get(), w = wRef.get();
  keepMax(e.step1, relDiff(out1.get(), y1));
  keepMax(e.z, relDiff(z1.get(), zRef.get()));
  keepMax(e.stepN, relDiff(zInPlace.get(), w));
  keepMax(e.neg, relDiff(outNeg.get(), negated(y1)));
  keepMax(e.neg, relDiff(outN.get(), negated(w)));
}

static void reportSteps(const StepErrors& e, const std::string& what) {
  const struct {
    const char* name;
    double err, tol;
  } rows[] = {{"step 1 vs reference", e.step1, 1e-14},
              {"step 1 z = b/theta", e.z, 0},
              {"step N in place vs reference", e.stepN, 1e-14},
              {"negated output vs -reference", e.neg, 1e-14}};
  for (const auto& r : rows)
    check(r.err <= r.tol, "fused " + what + " " + r.name, r.err);
}

/** Eigenvalues of a symmetric matrix (cyclic Jacobi). */
static std::vector<double> symEig(std::vector<double> A, int n) {
  for (int sweep = 0; sweep < 100; sweep++) {
    double off = 0;
    for (int p = 0; p < n; p++)
      for (int q = p + 1; q < n; q++)
        off += A[p * n + q] * A[p * n + q];
    if (off < 1e-26)
      break;
    for (int p = 0; p < n; p++)
      for (int q = p + 1; q < n; q++) {
        const double apq = A[p * n + q];
        if (std::fabs(apq) < 1e-300)
          continue;
        const double th = 0.5 * (A[q * n + q] - A[p * n + p]) / apq;
        const double t =
            (th >= 0 ? 1 : -1) / (std::fabs(th) + std::sqrt(th * th + 1));
        const double c = 1 / std::sqrt(t * t + 1), s = t * c;
        for (int k = 0; k < n; k++) {
          const double akp = A[k * n + p], akq = A[k * n + q];
          A[k * n + p] = c * akp - s * akq;
          A[k * n + q] = s * akp + c * akq;
        }
        for (int k = 0; k < n; k++) {
          const double apk = A[p * n + k], aqk = A[q * n + k];
          A[p * n + k] = c * apk - s * aqk;
          A[q * n + k] = s * apk + c * aqk;
        }
      }
  }
  std::vector<double> e(n);
  for (int p = 0; p < n; p++)
    e[p] = A[p * n + p];
  std::sort(e.begin(), e.end());
  return e;
}

static void testLap27Local() {
  std::printf("lap27-local\n");
  std::mt19937_64 gen(13);
  StepErrors steps;
  double errSym = 0;
  double errMin = 0, errMax = 0, errDiag = 0;
  int combos = 0, singular = 0, errSingular = 0;
  // packed rows of a local block, every face code; a flat axis has 1 row
  // (flat masks 0, 1, 2, 4, 6 of the step kernels)
  const int shapes[][3] = {{5, 4, 3}, {4, 5, 1}, {3, 3, 6},
                           {1, 4, 5}, {4, 1, 5}, {5, 1, 1}};
  for (const auto& sh : shapes) {
    const bool flat[3] = {sh[0] == 1, sh[1] == 1, sh[2] == 1};
    for (int code = 0; code < 729; code++) {
      bool row[6], copy[6];
      if (!faceCode(code, sh, flat, row, copy))
        continue;
      combos++;
      const Lap27Op op = makeOp(sh, flat, row, copy, true);
      const int N = sh[0] * sh[1] * sh[2];
      const Vec b = randomVec(N, gen), y = randomVec(N, gen),
                z = randomVec(N, gen);
      auto negHost = [&](const Vec& v) { return negated(hostLap27(v, op)); };
      checkFusedSteps(op, b, y, z, negHost(b), negHost(y), steps);

      // symmetry and exact bounds of -L without the identity rows, whose
      // value is diag
      std::vector<int> basis; // packed index of the representative rows
      for (int i = 0; i < sh[0]; i++)
        for (int j = 0; j < sh[1]; j++)
          for (int k = 0; k < sh[2]; k++)
            if (!hostFaceRow(op, i, j, k))
              basis.push_back((i * sh[1] + j) * sh[2] + k);
      const int m = basis.size();
      std::vector<double> A((size_t)m * m);
      for (int col = 0; col < m; col++) {
        Vec e(N, 0);
        e[basis[col]] = 1;
        const Vec Ae = negHost(e);
        for (int r = 0; r < m; r++)
          A[(size_t)r * m + col] = Ae[basis[r]];
      }
      double asym = 0, scale = 0;
      for (int r = 0; r < m; r++)
        for (int q = 0; q < m; q++) {
          keepMax(asym, std::fabs(A[r * m + q] - A[q * m + r]));
          keepMax(scale, std::fabs(A[r * m + q]));
        }
      keepMax(errSym, asym / scale);
      const std::vector<double> ev = symEig(A, m);
      double lo = 0, hi = 0;
      for (double v : ev) {
        hi = std::max(hi, v);
        if (v > 1e-9 * ev.back() && (lo == 0 || v < lo))
          lo = v;
      }
      const bool denseSingular = ev.front() < 1e-9 * ev.back();
      const Lap27Eig e = lap27EigBounds(op);
      keepMax(errMax, std::fabs(e.max - hi) / hi);
      keepMax(errMin, std::fabs(e.min - lo) / lo);
      keepMax(errDiag, std::fabs(op.diag - 0.5 * (lo + hi)) / hi);
      singular += denseSingular;
      errSingular += e.singular != denseSingular;
    }
  }
  std::printf("  %d operator/box combinations (%d singular)\n", combos,
              singular);
  reportSteps(steps, "27-point");
  check(errSym < 1e-14, "local operator symmetric (face rules, cuts)", errSym);
  check(errMax < 1e-12, "lap27EigBounds max vs dense eigenvalues", errMax);
  check(errMin < 1e-9, "lap27EigBounds smallest non-zero vs dense", errMin);
  check(errSingular == 0,
        "lap27EigBounds singular vs dense (smallest < 1e-9 largest), "
        "mismatches",
        errSingular);
  check(errDiag < 1e-9,
        "identity rows: |diag - (lo + hi)/2| / hi (dense lo, hi)", errDiag);
}

/** Solve L x = b by CG on -L x = -b (-L symmetric positive definite on the
 *  given operator, which has a zero-psi face or zero ghosts). */
static Vec cgLap27(const Vec& b, const Lap27Op& op) {
  Vec x(b.size(), 0), r = negated(b), p = r;
  double rr = dot(r, r);
  const double r0 = std::sqrt(rr);
  for (int it = 0; it < 20000 && std::sqrt(rr) > 1e-14 * r0; it++) {
    const Vec Ap = negated(hostLap27(p, op));
    const double a = rr / dot(p, Ap);
    for (size_t q = 0; q < x.size(); q++) {
      x[q] += a * p[q];
      r[q] -= a * Ap[q];
    }
    const double rn = dot(r, r);
    for (size_t q = 0; q < x.size(); q++)
      p[q] = r[q] + (rn / rr) * p[q];
    rr = rn;
  }
  return x;
}

// ---------------------------------------------------------------------------
// The cleaning preconditioner (gpuChebyshevCleaning)

/** x = gpuChebyshevCleaning(b) on [eigMin, eigMax]; x and the scratch
 *  vectors start as NaN. */
static Vec chebCleaning(const Vec& b, const Lap27Op& op, double eigMin,
                        double eigMax, int steps) {
  const Vec nan(b.size(), kNaN);
  Dev db(b), x(nan), y(nan), z(nan);
  gpuChebyshevCleaning(x.p, db.p, op, eigMin, eigMax, steps, y.p, z.p, 0);
  return x.get();
}

/** The same polynomial on the host: the recurrence of GPUChebyshev.cuh with
 *  A = -L (hostLap27), x = -y. */
static Vec hostChebyshev(const Vec& b, const Lap27Op& op, double eigMin,
                         double eigMax, int steps) {
  const double theta = 0.5 * (eigMin + eigMax), delta = 0.5 * (eigMax - eigMin);
  const double sigma = theta / delta;
  double rho = 1 / (2 * sigma - 1 / sigma);
  const Vec Ab = negated(hostLap27(b, op));
  Vec y(b.size()), z(b.size());
  for (size_t q = 0; q < b.size(); q++) {
    z[q] = b[q] / theta;
    y[q] = 4 * rho / delta * b[q] - 2 * rho / (delta * theta) * Ab[q];
  }
  for (int step = 2; step <= steps; step++) {
    const double rhoOld = rho;
    rho = 1 / (2 * sigma - rhoOld);
    const Vec Ay = negated(hostLap27(y, op));
    for (size_t q = 0; q < b.size(); q++)
      z[q] =
          rho * (2 * sigma * y[q] + 2 / delta * (b[q] - Ay[q]) - rhoOld * z[q]);
    std::swap(y, z);
  }
  return negated(y);
}

static void testChebyshevCleaning() {
  std::printf("chebyshev-cleaning\n");
  std::mt19937_64 gen(31);
  const bool flat[3] = {}, none[6] = {};
  // node rows with zero and copy faces and cuts: the polynomial against the
  // host recurrence, odd and even last steps
  {
    const int n[3] = {6, 5, 4};
    const bool row[6] = {true, false, true, true, false, true};
    const bool copy[6] = {false, false, true, false, false, true};
    const Lap27Op op = makeOp(n, flat, row, copy, true);
    const Lap27Eig e = lap27EigBounds(op);
    const Vec b = randomVec((size_t)n[0] * n[1] * n[2], gen);
    for (int steps : {1, 2, 3, 40}) {
      const double err = relDiff(chebCleaning(b, op, e.min, e.max, steps),
                                 hostChebyshev(b, op, e.min, e.max, steps));
      check(err < 1e-12, std::to_string(steps) + " step(s) vs host recurrence",
            err);
    }
    const double errZero = relDiff(chebCleaning(b, op, 0, 0, 5), negated(b));
    check(errZero == 0, "eigMax = 0: x = -b", errZero);
    const double errCopy = relDiff(chebCleaning(b, op, e.min, e.max, 0), b);
    check(errCopy == 0, "0 steps: x = b", errCopy);
  }
  // one row, no flat axis: a single eigenvalue (min = max), x = L⁻¹ b
  {
    const int n[3] = {1, 1, 1};
    const Lap27Op op = makeOp(n, flat, none, none, false);
    const Lap27Eig e = lap27EigBounds(op);
    const Vec b = randomVec(1, gen);
    const double err =
        relDiff(hostLap27(chebCleaning(b, op, e.min, e.max, 5), op), b);
    check(err < 1e-14, "1 row (min = max), 5 steps: L x vs b", err);
  }
  // 40 steps on the exact bounds of zero-face rows (-L symmetric, its
  // spectrum in [min, max], diag in the middle): the residual polynomial
  // has degree 41, at most 1/T_41(sigma) <= 2 q^41 on [min, max] in the
  // 2-norm, q = (sqrt(kappa) - 1) / (sqrt(kappa) + 1); 1e-12 for rounding.
  // The second block, kappa ~ 600, makes the bound bite.  The residual with
  // +L fails on a sign error ((-L)⁻¹ b gives 2).
  const int shapes[][3] = {{5, 4, 3}, {12, 10, 8}};
  for (const auto& sh : shapes) {
    const bool row[6] = {true, true, true, true, true, true};
    const Lap27Op op = makeOp(sh, flat, row, none, true);
    const Lap27Eig e = lap27EigBounds(op);
    const Vec b = randomVec((size_t)sh[0] * sh[1] * sh[2], gen);
    const Vec x = chebCleaning(b, op, e.min, e.max, 40);
    const double res = norm2(minus(hostLap27(x, op), b)) / norm2(b);
    const double kappa = e.max / e.min;
    const double q = (std::sqrt(kappa) - 1) / (std::sqrt(kappa) + 1);
    const double bound = 2 * std::pow(q, 41) + 1e-12;
    char what[160];
    std::snprintf(what, sizeof(what),
                  "rows %dx%dx%d, kappa %.3g, 40 steps: |L x - b| / |b| <= "
                  "2 q^41 + 1e-12 = %.3e",
                  sh[0], sh[1], sh[2], kappa, bound);
    check(res <= bound, what, res);
  }
}

// ---------------------------------------------------------------------------
static void testDivBProjection() {
  std::printf("divb-projection\n");
  std::mt19937_64 gen(17);
  const Grid g = makeGrid(9, 8, 7);
  const T invd[3] = {1 / g.h[0], 1 / g.h[1], 1 / g.h[2]};
  const int n[3] = {g.nn[0] - 2, g.nn[1] - 2, g.nn[2] - 2};
  const bool flat[3] = {false, false, false};
  // x: two conductors, y: open (psi = 0) and conductor, z: two open faces
  const bool row[6] = {true, true, true, true, true, true};
  const bool copy[6] = {true, true, false, true, false, false};
  const Lap27Op op = makeOp(n, flat, row, copy, true);
  auto project = [&](Vec B[3]) {
    Dev bx(B[0]), by(B[1]), bz(B[2]), rhs(Vec(op.ni * op.nj * op.nk, 0));
    gpuDivC2NToKrylov(rhs.p, bx.p, by.p, bz.p, g.nc[1], g.nc[2], invd, op, 0);
    const Vec b = rhs.get();
    const Vec x = cgLap27(b, op);
    Dev dx(x), psi(Vec(g.N(), kNaN));
    gpuSolver2PhysLap27(psi.p, dx.p, g.nn[1], g.nn[2], op, 0);
    gpuSubGradN2C(bx.p, by.p, bz.p, psi.p, g.nc[0], g.nc[1], g.nc[2], invd, 0);
    B[0] = bx.get();
    B[1] = by.get();
    B[2] = bz.get();
    Dev after(Vec(b.size(), 0));
    gpuDivC2NToKrylov(after.p, bx.p, by.p, bz.p, g.nc[1], g.nc[2], invd, op, 0);
    return std::make_pair(maxAbs(b), maxAbs(after.get()));
  };
  // a random B (ghost cells random too: the right-hand side never reads
  // them, the face rows being 0)
  Vec B[3] = {randomVec(g.C(), gen), randomVec(g.C(), gen),
              randomVec(g.C(), gen)};
  const Vec B0[3] = {B[0], B[1], B[2]};
  const auto r = project(B);
  check(r.second < 1e-11 * r.first,
        "divC2N(Bc) after / before at the constraint nodes",
        r.second / r.first);
  // the change of B_n in the wall cells: conductors (x faces, y right face)
  // and the open y-left face
  Vec dn, dopen;
  for (int j = 1; j < g.nc[1] - 1; j++)
    for (int k = 1; k < g.nc[2] - 1; k++)
      for (int i : {1, g.nc[0] - 2})
        dn.push_back(B[0][g.c(i, j, k)] - B0[0][g.c(i, j, k)]);
  for (int i = 1; i < g.nc[0] - 1; i++)
    for (int k = 1; k < g.nc[2] - 1; k++) {
      dn.push_back(B[1][g.c(i, g.nc[1] - 2, k)] -
                   B0[1][g.c(i, g.nc[1] - 2, k)]);
      dopen.push_back(B[1][g.c(i, 1, k)] - B0[1][g.c(i, 1, k)]);
    }
  check(maxAbs(dn) == 0, "conductor faces: B_n of the wall cells bit-identical",
        maxAbs(dn));
  const double dOpen = maxAbs(dopen);
  check(dOpen > 1e-6 && dOpen < kInf,
        "open face: B_n of the wall cells corrected", dOpen);
  // B = curlN2C E has no divergence: unchanged
  Vec E[3] = {randomVec(g.N(), gen), randomVec(g.N(), gen),
              randomVec(g.N(), gen)};
  Vec C[3];
  {
    Dev ex(E[0]), ey(E[1]), ez(E[2]), cx(Vec(g.C(), 0)), cy(Vec(g.C(), 0)),
        cz(Vec(g.C(), 0));
    gpuCurlN2C(cx.p, cy.p, cz.p, ex.p, ey.p, ez.p, g.nc[0], g.nc[1], g.nc[2],
               invd[0], invd[1], invd[2]);
    C[0] = cx.get();
    C[1] = cy.get();
    C[2] = cz.get();
  }
  const Vec C0[3] = {C[0], C[1], C[2]};
  const auto rc = project(C);
  double dc = 0, sc = 0;
  for (int c = 0; c < 3; c++) {
    dc = std::max(dc, maxAbs(minus(C[c], C0[c])));
    sc = std::max(sc, maxAbs(C0[c]));
  }
  std::printf("  curl(E): |divC2N| %.2e before\n", rc.first);
  check(dc < 1e-12 * sc, "curlN2C(E) left unchanged / |B|", dc / sc);
}

static void testDivEProjection() {
  std::printf("dive-projection\n");
  std::mt19937_64 gen(19);
  const Grid g = makeGrid(9, 8, 7);
  const T invd[3] = {1 / g.h[0], 1 / g.h[1], 1 / g.h[2]};
  const int n[3] = {g.nc[0] - 2, g.nc[1] - 2, g.nc[2] - 2};
  const size_t rows = (size_t)n[0] * n[1] * n[2];
  const bool flat[3] = {false, false, false}, none[6] = {};
  const Lap27Op op = makeOp(n, flat, none, none, false);
  const Vec E[3] = {randomVec(g.N(), gen), randomVec(g.N(), gen),
                    randomVec(g.N(), gen)};
  const Vec rho = randomVec(g.C(), gen);
  const T scale = -4 * M_PI;
  // divN2C(E) + scale rho on the cell rows, by the stencil kernel
  auto residual = [&](const Vec Ev[3]) {
    const Vec d = divN2C(g, Ev);
    Vec r(rows);
    for (int i = 0; i < n[0]; i++)
      for (int j = 0; j < n[1]; j++)
        for (int k = 0; k < n[2]; k++) {
          const size_t c = g.c(i + 1, j + 1, k + 1);
          r[((size_t)i * n[1] + j) * n[2] + k] = d[c] + scale * rho[c];
        }
    return r;
  };
  Dev ex(E[0]), ey(E[1]), ez(E[2]), dr(rho), db(Vec(rows, 0));
  gpuDivN2CRhoToKrylov(db.p, ex.p, ey.p, ez.p, dr.p, scale, g.nn[1], g.nn[2],
                       invd, op, 0);
  const Vec b = db.get();
  const double errRhs = relDiff(b, residual(E));
  check(errRhs < 1e-14,
        "right-hand side vs divN2C(E) - 4 pi rho (stencil kernel)", errRhs);
  const Vec x = cgLap27(b, op);
  Dev dx(x), phi(Vec(g.C(), 0));
  gpuSolver2PhysLap27(phi.p, dx.p, g.nc[1], g.nc[2], op, 0); // ghosts stay 0
  gpuSubGradC2N(ex.p, ey.p, ez.p, phi.p, g.nn[0], g.nn[1], g.nn[2], invd, 0);
  const Vec En[3] = {ex.get(), ey.get(), ez.get()};
  const double n0 = maxAbs(b), n1 = maxAbs(residual(En));
  check(n1 < 1e-11 * n0, "divN2C(E) - 4 pi rho after / before (every cell)",
        n1 / n0);
}

// ---------------------------------------------------------------------------
// The null space of the cleaning operators

/** One pass of gpuRemoveCleaningNullSpace over emulated ranks: per rank the
 *  signed sums of its vector v[r] (plan ns[r]), their sum over the ranks,
 *  then per rank the terms. */
static void projectNullSpace(std::vector<Vec>& v,
                             const std::vector<CleaningNullSpace>& ns) {
  const int ranks = (int)v.size();
  std::vector<T> global(ns[0].globalSize, 0.0), part;
  for (int r = 0; r < ranks; r++) {
    Dev d(v[r]), sums(Vec(std::max(ns[r].localSize, 1), 0));
    gpuCleaningNullSums(d.p, ns[r], sums.p, part, 0);
    for (size_t q = 0; q < global.size(); q++)
      global[q] += part[q];
  }
  for (int r = 0; r < ranks; r++) {
    const SignedBox& box = ns[r].box;
    Dev d(v[r]), coef(Vec(box.n[0] + box.n[1] + box.n[2], 0));
    gpuCleaningNullSubtract(d.p, ns[r], global, coef.p, 0);
    v[r] = d.get();
  }
}

/** Largest null-space component of the global g (extents G) on the box
 *  [lo, hi), over max |g|: the sum (if mean), and the checkerboard families
 *  (-1)^(i_a + i_b) f(i_c) of every two even axes a, b. */
static double nullComponent(const Vec& g, const int G[3], const int lo[3],
                            const int hi[3], const bool even[3], bool mean) {
  auto e = [](int i) { return i % 2 ? -1.0 : 1.0; };
  double sum = 0, left = 0;
  std::vector<double> fam[3] = {std::vector<double>(G[0], 0), // lines along c
                                std::vector<double>(G[1], 0),
                                std::vector<double>(G[2], 0)};
  for (int i = lo[0]; i < hi[0]; i++)
    for (int j = lo[1]; j < hi[1]; j++)
      for (int k = lo[2]; k < hi[2]; k++) {
        const double x = g[((size_t)i * G[1] + j) * G[2] + k];
        sum += x;
        fam[0][i] += e(j) * e(k) * x;
        fam[1][j] += e(i) * e(k) * x;
        fam[2][k] += e(i) * e(j) * x;
      }
  if (mean)
    keepMax(left, std::fabs(sum));
  for (int c = 0; c < 3; c++)
    if (even[(c + 1) % 3] && even[(c + 2) % 3])
      for (const double q : fam[c])
        keepMax(left, std::fabs(q));
  const double scale = maxAbs(g);
  return scale < kInf ? left / scale : kInf;
}

/** One projection on `ranks` emulated ranks along the periodic x, N global
 *  cells (1 along a flat axis): cells, or nodes with shared rows along the
 *  periodic axes and conductor face rows (faceCopy) on the others, as in
 *  testGEM3Dsmall.  The plans hold the mean and one family per two even
 *  axes, no null component is left, the removed part d is in the null space
 *  of the global operator, and the shared copies and face rows agree. */
static void nullSpaceCase(const int N[3], int ranks, bool nodes,
                          const bool periodic[3], bool mean,
                          std::mt19937_64& gen) {
  int G[3], lo[3], hi[3], families = 0; // unique global rows, non-face box
  bool flat[3], shared[3], even[3], row[6];
  for (int a = 0; a < 3; a++) {
    flat[a] = N[a] == 1;
    shared[a] = nodes && periodic[a] && !flat[a];
    even[a] = periodic[a] && !flat[a] && N[a] % 2 == 0;
    row[2 * a] = row[2 * a + 1] = nodes && !periodic[a];
    G[a] = N[a] + row[2 * a];
    lo[a] = row[2 * a];
    hi[a] = G[a] - row[2 * a];
  }
  for (int c = 0; c < 3; c++)
    families += even[(c + 1) % 3] && even[(c + 2) % 3];
  const Vec g0 = randomVec((size_t)G[0] * G[1] * G[2], gen);
  std::vector<Lap27Op> op(ranks);
  std::vector<CleaningNullSpace> ns(ranks);
  std::vector<Vec> v(ranks);
  int badSums = 0;
  for (int r = 0; r < ranks; r++) {
    const int x0 = r * N[0] / ranks, offset[3] = {x0, 0, 0};
    const int n[3] = {(r + 1) * N[0] / ranks - x0 + shared[0], G[1] + shared[1],
                      G[2] + shared[2]};
    op[r] = makeOp(n, flat, row, row, nodes);
    ns[r] = cleaningNullSpace(op[r], nodes, N, offset, periodic, shared, mean);
    badSums += ns[r].nSums != mean + families;
  }
  auto at = [&](int i, int j, int k) { // global row, periodic wrap
    return ((size_t)(i % G[0]) * G[1] + j % G[1]) * G[2] + k % G[2];
  };
  // f(local row, global row) over the rows of rank r
  auto forRows = [&](int r, auto&& f) {
    size_t p = 0;
    for (int i = 0; i < op[r].ni; i++)
      for (int j = 0; j < op[r].nj; j++)
        for (int k = 0; k < op[r].nk; k++)
          f(p++, at(r * N[0] / ranks + i, j, k));
  };
  for (int r = 0; r < ranks; r++) {
    v[r].resize((size_t)op[r].ni * op[r].nj * op[r].nk);
    forRows(r, [&](size_t p, size_t q) { v[r][p] = g0[q]; });
  }
  projectNullSpace(v, ns);
  Vec g(g0.size(), kNaN);
  double mismatch = 0;
  for (int r = 0; r < ranks; r++)
    forRows(r, [&](size_t p, size_t q) {
      if (std::isnan(g[q]))
        g[q] = v[r][p];
      else
        keepMax(mismatch, std::fabs(v[r][p] - g[q]));
    });
  // L d = 0 for the global operator, defined on every row: periodic wrap,
  // and along a wall its zero ghost cells or conductor face rows; d = 0 on
  // the face rows
  const Lap27Op global = makeOp(G, flat, row, row, nodes);
  const Vec d = minus(g0, g);
  double wsum = 0, faceRows = 0;
  for (int a = 0; a < 2; a++)
    for (int b = 0; b < 2; b++)
      for (int c = 0; c < 2; c++)
        wsum += std::fabs(global.w[a][b][c]) * (1 << (a + b + c));
  for (int i = 0; i < G[0]; i++)
    for (int j = 0; j < G[1]; j++)
      for (int k = 0; k < G[2]; k++) {
        const int t[3] = {i, j, k};
        for (int a = 0; a < 3; a++)
          if (t[a] < lo[a] || t[a] >= hi[a])
            keepMax(faceRows, std::fabs(d[at(i, j, k)]));
      }
  const double left = nullComponent(g, G, lo, hi, even, mean);
  const double Ld = maxAbs(hostLap27(d, global, periodic)) / (wsum * maxAbs(d));
  std::string walls;
  for (int a = 0; a < 3; a++)
    if (!periodic[a])
      walls += "xyz"[a];
  char label[96];
  std::snprintf(label, sizeof(label),
                "%s %dx%dx%d%s%s, %d rank(s): ", nodes ? "nodes" : "cells",
                N[0], N[1], N[2], walls.empty() ? "" : ", walls ",
                walls.c_str(), ranks);
  const std::string l = label;
  check(badSums == 0,
        l + "nSums = " + std::to_string(mean + families) +
            " (mean, families), ranks off",
        badSums);
  check(left < 1e-12, l + "no null-space component left", left);
  check(Ld <= 1e-13, l + "removed d: |L d| / (sum |w| |d|)", Ld);
  if (nodes)
    check(mismatch == 0, l + "the copies of a shared row agree", mismatch);
  if (nodes && !walls.empty())
    check(faceRows == 0, l + "face rows bit-identical", faceRows);
}

static void testNullSpace() {
  std::printf("null-space\n");
  std::mt19937_64 gen(23);
  const bool all[3] = {true, true, true}, yWalls[3] = {true, false, true};
  // three even axes, a flat z, an odd y (no family through y); 4 ranks: odd
  // x offsets
  const int even3[3] = {12, 24, 32}, flatZ[3] = {12, 24, 1},
            oddY[3] = {12, 5, 32};
  for (int ranks : {1, 3, 4}) {
    for (const int* N : {even3, flatZ, oddY})
      for (const bool nodes : {false, true})
        nullSpaceCase(N, ranks, nodes, all, true, gen);
    // y walls: zero ghost cells (no mean), conductor node faces
    nullSpaceCase(even3, ranks, false, yWalls, false, gen);
    nullSpaceCase(even3, ranks, true, yWalls, true, gen);
  }
}

// ---------------------------------------------------------------------------
// Krylov kernels of the FGMRES(60) cleaning solves (GPUBlas)

static void testKrylovKernels() {
  std::printf("krylov-kernels\n");
  std::mt19937_64 gen(29);
  const size_t n = 3001; // not a multiple of the block
  const int count = 61;  // KRYLOV_M_MAX + 1 vectors
  const Vec V = randomVec(n * count, gen), w = randomVec(n, gen);
  Dev dV(V), dw(w), out(Vec(count + 2, 0));
  // batched dot products in chunks of 32: k + 1 = 1, 32, 33, 41 and 60
  // vectors
  double errDot = 0, errNorm = 0;
  for (int k : {0, 31, 32, 40, 59}) {
    gpuBatchedDotNorm(dw.p, dV.p, n, k, n, out.p, 0);
    const Vec o = out.get();
    for (int j = 0; j <= k; j++) {
      double ref = 0;
      for (size_t i = 0; i < n; i++)
        ref += (double)w[i] * V[j * n + i];
      keepMax(errDot, std::fabs(o[j] - ref) / (1 + std::fabs(ref)));
    }
    double ref = 0;
    for (size_t i = 0; i < n; i++)
      ref += (double)w[i] * w[i];
    keepMax(errNorm, std::fabs(o[k + 1] - ref) / ref);
  }
  check(errDot < 1e-12, "gpuBatchedDotNorm (up to 60 vectors) vs host dots",
        errDot);
  check(errNorm < 1e-12, "gpuBatchedDotNorm norm2 with the last chunk",
        errNorm);
  // the fused Arnoldi update against count sequential gpuAddscale calls
  const Vec h = randomVec(count, gen);
  Dev seq(w), fused(w);
  for (int j = 0; j < count; j++)
    gpuAddscale(-h[j], seq.p, dV.p + j * n, n, 0);
  gpuSubCombination(fused.p, dV.p, n, count, n, h.data(), 0);
  const double diff = relDiff(fused.get(), seq.get());
  check(diff == 0, "gpuSubCombination bit-identical to gpuAddscale calls",
        diff);
}

// ---------------------------------------------------------------------------
// 7-point operators (not used currently)

/** Host -∇² of PoissonLocalOperator on the packed cells. */
static Vec hostNegLapLocal(const Vec& y, const PoissonLocalOperator& op) {
  Vec r(y.size());
  for (int i = 0; i < op.ni; i++)
    for (int j = 0; j < op.nj; j++)
      for (int k = 0; k < op.nk; k++) {
        auto at = [&](int a, int b, int c) {
          return y[((size_t)a * op.nj + b) * op.nk + c];
        };
        const double f = at(i, j, k);
        const double xm = i > 0 ? at(i - 1, j, k) : op.g[0] * f;
        const double xp = i < op.ni - 1 ? at(i + 1, j, k) : op.g[1] * f;
        const double ym = j > 0 ? at(i, j - 1, k) : op.g[2] * f;
        const double yp = j < op.nj - 1 ? at(i, j + 1, k) : op.g[3] * f;
        const double zm = k > 0 ? at(i, j, k - 1) : op.g[4] * f;
        const double zp = k < op.nk - 1 ? at(i, j, k + 1) : op.g[5] * f;
        r[((size_t)i * op.nj + j) * op.nk + k] =
            -((xm - 2 * f + xp) * op.invdx2 + (ym - 2 * f + yp) * op.invdy2 +
              (zm - 2 * f + zp) * op.invdz2);
      }
  return r;
}

static void testPoissonLocal() {
  std::printf("poisson-local\n");
  const int shapes[][3] = {{5, 4, 3}, {1, 6, 5}, {6, 1, 5}, {4, 5, 1},
                           {1, 1, 7}, {1, 1, 1}, {2, 2, 2}, {17, 9, 33}};
  std::mt19937_64 gen(11);
  StepErrors steps;
  double errHost = 0;
  int combos = 0;
  for (const auto& sh : shapes) {
    PoissonLocalOperator op;
    op.ni = sh[0];
    op.nj = sh[1];
    op.nk = sh[2];
    op.invdx2 = 1.3;
    op.invdy2 = 0.7;
    op.invdz2 = 2.1;
    const int n = op.ni * op.nj * op.nk;
    const int nxc = op.ni + 2, nyc = op.nj + 2, nzc = op.nk + 2;
    const size_t nCell = (size_t)nxc * nyc * nzc;
    // every combination on the small boxes, a random sample on the large one
    const bool all = n <= 64;
    std::uniform_int_distribution<int> pick(0, 728);
    for (int c = 0; c < (all ? 729 : 40); c++, combos++) {
      int code = all ? c : pick(gen);
      int factor[6];
      for (int f = 0; f < 6; f++, code /= 3) {
        factor[f] = code % 3 - 1;
        op.g[f] = factor[f];
      }
      const Vec b = randomVec(n, gen), y = randomVec(n, gen),
                z = randomVec(n, gen);
      Dev db(b), dy(y);
      const Vec nanCells(nCell, kNaN);

      // composed reference: Ay = -(pack(lapLocal(unpack(v))))
      auto refNegLap = [&](const Dev& v) {
        Dev phys(nanCells), lap(nanCells), out(Vec(n, 0));
        gpuSolver2Phys1(phys.p, v.p, nxc, nyc, nzc, 0);
        gpuLapC2CLocalKernel(lap.p, phys.p, nxc, nyc, nzc, op.invdx2, op.invdy2,
                             op.invdz2, factor, 0);
        gpuPhys2Solver1(out.p, lap.p, nxc, nyc, nzc, 0);
        gpuScale(out.p, -1.0, n, 0);
        return out.get();
      };
      const Vec Ab = refNegLap(db), Ay = refNegLap(dy);
      keepMax(errHost, relDiff(Ay, hostNegLapLocal(y, op)));

      checkFusedSteps(op, b, y, z, Ab, Ay, steps);
    }
  }
  std::printf("  %d operator/box combinations\n", combos);
  check(errHost < 1e-13,
        "reference -∇² (unpack, gpuLapC2CLocalKernel, pack) "
        "vs host",
        errHost);
  reportSteps(steps, "7-point");
}

static void testPoissonImage() {
  std::printf("poisson-image\n");
  const int shapes[][3] = {{7, 6, 5}, {3, 8, 6}, {8, 3, 6},
                           {6, 7, 3}, {3, 3, 3}, {19, 11, 35}};
  const T ix2 = 1.7, iy2 = 0.9, iz2 = 2.3;
  std::mt19937_64 gen(5);
  double errHost = 0, errOld = 0;
  bool finite = true;
  for (const auto& sh : shapes) {
    const int nxc = sh[0], nyc = sh[1], nzc = sh[2];
    const int ni = nxc - 2, nj = nyc - 2, nk = nzc - 2, n = ni * nj * nk;
    // interior and face ghosts random, edge and corner ghosts NaN
    Vec f = randomVec((size_t)nxc * nyc * nzc, gen);
    auto idx = [&](int i, int j, int k) {
      return ((size_t)i * nyc + j) * nzc + k;
    };
    for (int i = 0; i < nxc; i++)
      for (int j = 0; j < nyc; j++)
        for (int k = 0; k < nzc; k++) {
          const int ghosts = (i == 0 || i == nxc - 1) +
                             (j == 0 || j == nyc - 1) +
                             (k == 0 || k == nzc - 1);
          if (ghosts >= 2)
            f[idx(i, j, k)] = kNaN;
        }
    Vec host(n);
    for (int i = 1; i <= ni; i++)
      for (int j = 1; j <= nj; j++)
        for (int k = 1; k <= nk; k++) {
          const double c = f[idx(i, j, k)];
          host[((size_t)(i - 1) * nj + j - 1) * nk + k - 1] =
              (f[idx(i - 1, j, k)] - 2 * c + f[idx(i + 1, j, k)]) * ix2 +
              (f[idx(i, j - 1, k)] - 2 * c + f[idx(i, j + 1, k)]) * iy2 +
              (f[idx(i, j, k - 1)] - 2 * c + f[idx(i, j, k + 1)]) * iz2;
        }
    Dev df(f), dk(Vec(n, 0)), dl(Vec(f.size(), 0)), dk2(Vec(n, 0));
    gpuLapC2CToKrylov(dk.p, df.p, nxc, nyc, nzc, ix2, iy2, iz2, 0);
    gpuLapC2CKernel(dl.p, df.p, nxc, nyc, nzc, ix2, iy2, iz2, 0);
    gpuPhys2Solver1(dk2.p, dl.p, nxc, nyc, nzc, 0);
    const Vec k = dk.get();
    for (const T x : k)
      finite = finite && std::isfinite(x);
    keepMax(errHost, relDiff(k, host));
    keepMax(errOld, relDiff(k, dk2.get()));
  }
  check(finite, "no edge or corner ghost read (result finite)", finite ? 0 : 1);
  check(errHost < 1e-14, "gpuLapC2CToKrylov vs host", errHost);
  check(errOld < 1e-15, "gpuLapC2CToKrylov vs gpuLapC2CKernel + pack", errOld);
}

int main(int argc, char** argv) {
  std::string which = "all";
  for (int a = 1; a + 1 < argc; a++)
    if (std::strcmp(argv[a], "--case") == 0)
      which = argv[a + 1];
  const std::pair<const char*, void (*)()> cases[] = {
      {"sal-tangential", testSalTangential},
      {"adjoint", testAdjoint},
      {"lap27-image", testLap27Image},
      {"lap27-local", testLap27Local},
      {"chebyshev-cleaning", testChebyshevCleaning},
      {"divb-projection", testDivBProjection},
      {"dive-projection", testDivEProjection},
      {"null-space", testNullSpace},
      {"krylov-kernels", testKrylovKernels},
      {"poisson-local", testPoissonLocal},
      {"poisson-image", testPoissonImage}};
  int ran = 0;
  for (const auto& c : cases)
    if (which == "all" || which == c.first) {
      c.second();
      ran++;
    }
  if (ran == 0) {
    std::printf("unknown case %s\n", which.c_str());
    return 2;
  }
  std::printf("%s (%d failure%s)\n", failures ? "FAILED" : "PASSED", failures,
              failures == 1 ? "" : "s");
  return failures ? 1 : 0;
}
