/* Unit tests of the div(B) cleaning and of the B absorbing layer (SAL) on the
 * GPU kernels used by EMfields3D::gpuApplyDivBCleaning and
 * EMfields3D::gpuOpenBoundaryInflowB.
 *
 *   --case sal-tangential  gpuSALBlendLayers3 with skipComp = face direction
 *                          leaves the normal component bit-identical, blends
 *                          the tangential ones, touches nothing outside the
 *                          layers; skipComp = -1 blends all three.
 *   --case adjoint         with zero psi ghosts (kPoissonBCface), gradC2N is
 *                          minus the transpose of divN2C:
 *                          <divN2C v, psi> + <v, gradC2N psi> = 0; with copy
 *                          ghosts it is not.
 *   --case contraction     psi solved exactly with the 7-point zero-ghost
 *                          Laplacian, B -= gradC2N psi on every node: |div B|
 *                          decreases at least by the mode bound 1 - l27/l7.
 *
 * Array layout as in the code: (i * ny + j) * nz + k, centres 0..nc-1 with
 * ghosts 0 and nc-1, nodes 0..nn-1 = nc with ghost nodes 0 and nn-1. */
#include "GPUBlas.cuh"
#include "GPUPhysicsKernels.cuh"
#include "GPUStencils.cuh"
#include "cudaTypeDef.cuh"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>

using T = cudaSolverType;
using Vec = std::vector<T>;

static int failures = 0;
static void check(bool ok, const char* what, double value) {
  std::printf("  [%s] %s: %.3e\n", ok ? "ok" : "FAIL", what, value);
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
        double errBlend = 0, errNormal = 0, errOutside = 0;
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
                if (!in) {
                  errOutside = std::max(errOutside,
                                        (double)std::fabs(r[c][a] - h[c][a]));
                } else if (c == skip) {
                  errNormal =
                      std::max(errNormal, (double)std::fabs(r[c][a] - h[c][a]));
                } else {
                  const T e = h[c][a] * sal + tgt[c] * (T(1) - sal);
                  errBlend = std::max(errBlend, (double)std::fabs(r[c][a] - e));
                }
              }
            }
        char what[160];
        std::snprintf(what, sizeof(what),
                      "dir %d %s face, skipComp %+d: blended components", dir,
                      asc ? "left" : "right", skip);
        check(errBlend < 1e-12, what, errBlend);
        if (skipMode) {
          std::snprintf(what, sizeof(what),
                        "dir %d %s face: normal component bit-identical", dir,
                        asc ? "left" : "right");
          check(errNormal == 0.0, what, errNormal);
        }
        std::snprintf(
            what, sizeof(what),
            "dir %d %s face, skipComp %+d: outside the layers untouched", dir,
            asc ? "left" : "right", skip);
        check(errOutside == 0.0, what, errOutside);
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
  bool ghostC(int i, int j, int k) const {
    return i == 0 || j == 0 || k == 0 || i == nc[0] - 1 || j == nc[1] - 1 ||
           k == nc[2] - 1;
  }
};

static Grid makeGrid(int mx, int my, int mz) { // m = physical cells
  Grid g;
  const int m[3] = {mx, my, mz};
  const T h[3] = {0.1, 0.13, 0.07};
  for (int d = 0; d < 3; d++) {
    g.nc[d] = m[d] + 2;
    g.nn[d] = m[d] + 3;
    g.h[d] = h[d];
  }
  return g;
}

// ghost centres: factor 0 (zero) or +1 (copy of the adjacent cell)
static void setGhosts(const Grid& g, Vec& psi, int factor) {
  for (int i = 0; i < g.nc[0]; i++)
    for (int j = 0; j < g.nc[1]; j++)
      for (int k = 0; k < g.nc[2]; k++)
        if (g.ghostC(i, j, k)) {
          const int ii = std::min(std::max(i, 1), g.nc[0] - 2);
          const int jj = std::min(std::max(j, 1), g.nc[1] - 2);
          const int kk = std::min(std::max(k, 1), g.nc[2] - 2);
          psi[g.c(i, j, k)] = factor == 0 ? T(0) : psi[g.c(ii, jj, kk)];
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

static double dotCells(const Grid& g, const Vec& a, const Vec& b) {
  double s = 0;
  for (int i = 1; i < g.nc[0] - 1; i++)
    for (int j = 1; j < g.nc[1] - 1; j++)
      for (int k = 1; k < g.nc[2] - 1; k++)
        s += (double)a[g.c(i, j, k)] * b[g.c(i, j, k)];
  return s;
}

static double dotNodes(const Grid& g, const Vec a[3], const Vec b[3]) {
  double s = 0;
  for (int c = 0; c < 3; c++)
    for (int i = 1; i < g.nn[0] - 1; i++)
      for (int j = 1; j < g.nn[1] - 1; j++)
        for (int k = 1; k < g.nn[2] - 1; k++)
          s += (double)a[c][g.n(i, j, k)] * b[c][g.n(i, j, k)];
  return s;
}

static void testAdjoint() {
  std::printf("adjoint\n");
  const Grid g = makeGrid(10, 9, 8);
  std::mt19937_64 gen(2);
  Vec v[3] = {randomVec(g.N(), gen), randomVec(g.N(), gen),
              randomVec(g.N(), gen)};
  Vec psi = randomVec(g.C(), gen);
  const Vec d = divN2C(g, v);
  for (int factor : {0, 1}) {
    setGhosts(g, psi, factor);
    Vec gr[3];
    gradC2N(g, psi, gr);
    const double s1 = dotCells(g, d, psi), s2 = dotNodes(g, v, gr);
    const double rel = std::fabs(s1 + s2) / (std::fabs(s1) + std::fabs(s2));
    if (factor == 0)
      check(rel < 1e-12, "zero ghosts: |<D v, psi> + <v, G psi>| / scale", rel);
    else
      check(rel > 1e-3, "copy ghosts (old correction): not an adjoint pair",
            rel);
  }
}

// ---------------------------------------------------------------------------
// -(7-point Laplacian) on the physical cells, zero ghosts
static void negLap7(const Grid& g, const Vec& x, Vec& y) {
  const T w[3] = {1 / (g.h[0] * g.h[0]), 1 / (g.h[1] * g.h[1]),
                  1 / (g.h[2] * g.h[2])};
  auto at = [&](int i, int j, int k) -> T {
    return (i < 1 || j < 1 || k < 1 || i > g.nc[0] - 2 || j > g.nc[1] - 2 ||
            k > g.nc[2] - 2)
               ? T(0)
               : x[g.c(i, j, k)];
  };
  for (int i = 1; i < g.nc[0] - 1; i++)
    for (int j = 1; j < g.nc[1] - 1; j++)
      for (int k = 1; k < g.nc[2] - 1; k++) {
        const T c = x[g.c(i, j, k)];
        y[g.c(i, j, k)] = -(w[0] * (at(i - 1, j, k) - 2 * c + at(i + 1, j, k)) +
                            w[1] * (at(i, j - 1, k) - 2 * c + at(i, j + 1, k)) +
                            w[2] * (at(i, j, k - 1) - 2 * c + at(i, j, k + 1)));
      }
}

static void cgSolve(const Grid& g, const Vec& b, Vec& x) { // -L7 x = b
  Vec r = b, p = b, Ap(g.C(), 0);
  std::fill(x.begin(), x.end(), T(0));
  double rr = dotCells(g, r, r), r0 = std::sqrt(rr);
  for (int it = 0; it < 20000 && std::sqrt(rr) > 1e-13 * r0; it++) {
    negLap7(g, p, Ap);
    const double a = rr / dotCells(g, p, Ap);
    for (size_t q = 0; q < g.C(); q++) {
      x[q] += a * p[q];
      r[q] -= a * Ap[q];
    }
    const double rn = dotCells(g, r, r);
    for (size_t q = 0; q < g.C(); q++)
      p[q] = r[q] + (rn / rr) * p[q];
    rr = rn;
  }
}

static void testContraction() {
  std::printf("contraction\n");
  const Grid g = makeGrid(16, 14, 12);
  std::mt19937_64 gen(3);
  Vec v[3] = {randomVec(g.N(), gen), randomVec(g.N(), gen),
              randomVec(g.N(), gen)};
  const Vec d = divN2C(g, v);
  // solve L7 psi = d, i.e. -L7 psi = -d, psi = 0 on the ghost cells
  Vec rhs(g.C(), 0), psi(g.C(), 0);
  for (size_t q = 0; q < g.C(); q++)
    rhs[q] = -d[q];
  for (int i = 0; i < g.nc[0]; i++)
    for (int j = 0; j < g.nc[1]; j++)
      for (int k = 0; k < g.nc[2]; k++)
        if (g.ghostC(i, j, k))
          rhs[g.c(i, j, k)] = 0;
  cgSolve(g, rhs, psi);
  setGhosts(g, psi, 0);
  // B -= gradC2N psi on every node (gpuSub over the whole node array, as in
  // gpuApplyDivBCleaning; the ghost nodes of the gradient are 0)
  Vec gr[3];
  gradC2N(g, psi, gr);
  Vec vn[3];
  for (int c = 0; c < 3; c++) {
    Dev dv(v[c]), dg(gr[c]);
    gpuSub(dv.p, dg.p, g.N());
    vn[c] = dv.get();
  }
  const Vec d2 = divN2C(g, vn);
  const double n0 = std::sqrt(dotCells(g, d, d)),
               n1 = std::sqrt(dotCells(g, d2, d2));
  // mode bound: max over the box of 1 - l27/l7 (zero ghosts: theta_m = m
  // pi/(N+1))
  double bound = 0;
  const int m[3] = {g.nc[0] - 2, g.nc[1] - 2, g.nc[2] - 2};
  for (int a = 1; a <= m[0]; a++)
    for (int b = 1; b <= m[1]; b++)
      for (int c = 1; c <= m[2]; c++) {
        const double s[3] = {
            std::pow(std::sin(0.5 * a * M_PI / (m[0] + 1)), 2),
            std::pow(std::sin(0.5 * b * M_PI / (m[1] + 1)), 2),
            std::pow(std::sin(0.5 * c * M_PI / (m[2] + 1)), 2)};
        double l7 = 0, l27 = 0;
        for (int dd = 0; dd < 3; dd++) {
          const double w = 4 / (g.h[dd] * g.h[dd]);
          l7 += w * s[dd];
          l27 += w * s[dd] * (1 - s[(dd + 1) % 3]) * (1 - s[(dd + 2) % 3]);
        }
        bound = std::max(bound, 1 - l27 / l7);
      }
  std::printf(
      "  |div B| before %.4e, after %.4e, ratio %.4f, mode bound %.4f\n", n0,
      n1, n1 / n0, bound);
  check(n1 < n0, "global correction reduces |div B|", n1 / n0);
  check(n1 <= bound * n0 * (1 + 1e-10),
        "|div B| after <= (max_k 1 - l27/l7) |div B| before", n1 / n0 - bound);
}

int main(int argc, char** argv) {
  std::string which = "all";
  for (int a = 1; a + 1 < argc; a++)
    if (std::strcmp(argv[a], "--case") == 0)
      which = argv[a + 1];
  if (which == "all" || which == "sal-tangential")
    testSalTangential();
  if (which == "all" || which == "adjoint")
    testAdjoint();
  if (which == "all" || which == "contraction")
    testContraction();
  std::printf("%s (%d failure%s)\n", failures ? "FAILED" : "PASSED", failures,
              failures == 1 ? "" : "s");
  return failures ? 1 : 0;
}
