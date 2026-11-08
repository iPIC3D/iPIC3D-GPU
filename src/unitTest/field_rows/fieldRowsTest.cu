/* Unit tests of the boundary rows of the implicit Maxwell solve (GPU kernels
 * used by EMfields3D::gpuMaxwellImage / gpuMaxwellSource):
 *
 *   --case pec-linear      gpuPerfectConductorImage is linear in the Krylov
 *                          vector: image(0) = 0 on the face rows and image(a u
 *                          + b v) = a image(u) + b image(v), every face.
 *   --case pec-equivalent  image(x) - source equals the conductor row
 *                          x_n - (E_n - chi_na x_a - chi_nb x_b - 4 pi theta
 *                          dt J_n) / chi_nn with chi the rows of MUdot (host),
 *                          and the tangential rows are x_t - ebc_t.
 *   --case open-rows       gpuOpenBCSetFace3 / gpuOpenBCCopyFace3 write the
 *                          constants / identity on the face rows only.
 *   --case mudot           gpuMUdot (all species in one kernel) against a host
 *                          per-species MUdot and the per-species kernel
 *                          gpuMUdotSpecies, for 1..6, 16, 17 and 33 species
 *                          (several species sets) on 3D and 2D boxes; ghost
 *                          nodes are not written.
 *
 * Array layout as in the code: (i * ny + j) * nz + k. */
#include "GPUPhysicsKernels.cuh"
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

static Vec randomVec(size_t n, std::mt19937_64& g, double lo = -1,
                     double hi = 1) {
  std::uniform_real_distribution<double> d(lo, hi);
  Vec v(n);
  for (auto& x : v)
    x = d(g);
  return v;
}

// grid, species and constants of a small plasma
constexpr int NX = 9, NY = 8, NZ = 7, NS = 2;
constexpr size_t N = (size_t)NX * NY * NZ;
const T kDt = 0.3, kC = 1.0, kTh = 1.0, kFourPI = 16.0 * std::atan(1.0),
        kDelt = kC * kTh * kDt;
const T kQom[NS] = {-25.0, 1.0};
const T kEbc[3] = {0.11, -0.07, 0.05};

struct Plasma {
  Vec B[3], Be[3], rho;
  Dev *dB[3], *dBe[3], *drho, *dqom;
  explicit Plasma(std::mt19937_64& g) {
    for (int c = 0; c < 3; c++) {
      B[c] = randomVec(N, g, -0.5, 0.5);
      Be[c] = randomVec(N, g, -0.2, 0.2);
      dB[c] = new Dev(B[c]);
      dBe[c] = new Dev(Be[c]);
    }
    rho = randomVec(N * NS, g, 0.05, 0.2);
    for (size_t q = 0; q < N; q++)
      rho[q] = -rho[q]; // species 0 negative (electrons)
    drho = new Dev(rho);
    dqom = new Dev(Vec(kQom, kQom + NS));
  }
  ~Plasma() {
    for (int c = 0; c < 3; c++) {
      delete dB[c];
      delete dBe[c];
    }
    delete drho;
    delete dqom;
  }
  // host susceptibility row n at node q: the rows of the interior operator
  // MUdot, (v + v x w + (v.w) w) d per species
  void chiRow(int n, size_t q, double row[3]) const {
    row[0] = row[1] = row[2] = 0;
    row[n] = 1;
    for (int s = 0; s < NS; s++) {
      const double beta = 0.5 * kQom[s] * kDt / kC;
      const double w[3] = {beta * (B[0][q] + Be[0][q]),
                           beta * (B[1][q] + Be[1][q]),
                           beta * (B[2][q] + Be[2][q])};
      const double d = kFourPI / 2.0 * kDelt * kDt / kC * kQom[s] *
                       rho[s * N + q] /
                       (1.0 + w[0] * w[0] + w[1] * w[1] + w[2] * w[2]);
      // MUdot row n applied to the unit vector e_j
      for (int j = 0; j < 3; j++) {
        double e[3] = {0, 0, 0};
        e[j] = 1;
        const double edotw = e[0] * w[0] + e[1] * w[1] + e[2] * w[2];
        const double cross[3] = {e[1] * w[2] - e[2] * w[1],
                                 e[2] * w[0] - e[0] * w[2],
                                 e[0] * w[1] - e[1] * w[0]};
        row[j] += (e[n] + cross[n] + edotw * w[n]) * d;
      }
    }
  }
};

static size_t at(int i, int j, int k) { return ((size_t)i * NY + j) * NZ + k; }

// face rows of face (dir, high): node plane 1 or n-2, tangential interior
template <typename F> static void forFaceRows(int dir, bool high, F fn) {
  const int n[3] = {NX, NY, NZ};
  const int plane = high ? n[dir] - 2 : 1;
  for (int a = 1; a < (dir == 0 ? NY : NX) - 1; a++)
    for (int b = 1; b < (dir == 2 ? NY : NZ) - 1; b++) {
      const int i = dir == 0 ? plane : a;
      const int j = dir == 0 ? a : (dir == 1 ? plane : b);
      const int k = dir == 2 ? plane : b;
      fn(at(i, j, k));
    }
}

static void image(const Plasma& p, const Vec v[3], Vec out[3], int dir,
                  bool high) {
  Dev vx(v[0]), vy(v[1]), vz(v[2]);
  Vec zero(N, 0.0);
  Dev ix(zero), iy(zero), iz(zero);
  gpuPerfectConductorImage(ix.p, iy.p, iz.p, vx.p, vy.p, vz.p, p.dB[0]->p,
                           p.dB[1]->p, p.dB[2]->p, p.dBe[0]->p, p.dBe[1]->p,
                           p.dBe[2]->p, p.drho->p, p.dqom->p, NS, kDt, kC,
                           kFourPI, kDelt, NX, NY, NZ, dir, high);
  out[0] = ix.get();
  out[1] = iy.get();
  out[2] = iz.get();
}

static const char* faceName(int dir, bool high) {
  static const char* const names[6] = {"x-left",  "x-right", "y-left",
                                       "y-right", "z-left",  "z-right"};
  return names[2 * dir + (high ? 1 : 0)];
}

// ---------------------------------------------------------------------------
static void testPecLinear() {
  std::printf("pec-linear\n");
  std::mt19937_64 gen(21);
  Plasma p(gen);
  const double a = 0.7, b = -1.3;
  for (int dir = 0; dir < 3; dir++)
    for (bool high : {false, true}) {
      Vec u[3] = {randomVec(N, gen), randomVec(N, gen), randomVec(N, gen)};
      Vec v[3] = {randomVec(N, gen), randomVec(N, gen), randomVec(N, gen)};
      Vec w[3], z[3];
      for (int c = 0; c < 3; c++) {
        w[c] = Vec(N);
        z[c] = Vec(N, 0.0);
        for (size_t q = 0; q < N; q++)
          w[c][q] = a * u[c][q] + b * v[c][q];
      }
      Vec iu[3], iv[3], iw[3], iz[3];
      image(p, u, iu, dir, high);
      image(p, v, iv, dir, high);
      image(p, w, iw, dir, high);
      image(p, z, iz, dir, high);
      double errLin = 0, errZero = 0;
      forFaceRows(dir, high, [&](size_t q) {
        for (int c = 0; c < 3; c++) {
          errLin = std::max(
              errLin, std::fabs(iw[c][q] - (a * iu[c][q] + b * iv[c][q])));
          errZero = std::max(errZero, (double)std::fabs(iz[c][q]));
        }
      });
      char what[96];
      std::snprintf(what, sizeof(what), "%s: image(0) = 0",
                    faceName(dir, high));
      check(errZero == 0.0, what, errZero);
      std::snprintf(what, sizeof(what), "%s: image(a u + b v) - (a Au + b Av)",
                    faceName(dir, high));
      check(errLin < 1e-13, what, errLin);
    }
}

static void testPecEquivalent() {
  std::printf("pec-equivalent\n");
  std::mt19937_64 gen(22);
  Plasma p(gen);
  for (int dir = 0; dir < 3; dir++)
    for (bool high : {false, true}) {
      Vec x[3] = {randomVec(N, gen), randomVec(N, gen), randomVec(N, gen)};
      const Vec En = randomVec(N, gen), Jn = randomVec(N, gen);
      Vec im[3];
      image(p, x, im, dir, high);
      Dev sx(Vec(N, 0.0)), sy(Vec(N, 0.0)), sz(Vec(N, 0.0)), dE(En), dJ(Jn);
      gpuPerfectConductorSource(sx.p, sy.p, sz.p, dE.p, dJ.p, p.dB[0]->p,
                                p.dB[1]->p, p.dB[2]->p, p.dBe[0]->p,
                                p.dBe[1]->p, p.dBe[2]->p, p.drho->p, p.dqom->p,
                                NS, kDt, kC, kTh, kFourPI, kDelt, kEbc[0],
                                kEbc[1], kEbc[2], NX, NY, NZ, dir, high);
      const Vec src[3] = {sx.get(), sy.get(), sz.get()};
      double err = 0, scale = 0;
      forFaceRows(dir, high, [&](size_t q) {
        double chi[3];
        p.chiRow(dir, q, chi);
        const int na = (dir + 1) % 3, nb = (dir + 2) % 3;
        // the same rows in affine form (constants in the image, RHS 0; the
        // tangential RHS is ebc)
        const double oldN =
            x[dir][q] - (En[q] - chi[na] * x[na][q] - chi[nb] * x[nb][q] -
                         Jn[q] * kDt * kTh * kFourPI) /
                            chi[dir];
        err = std::max(err, std::fabs((im[dir][q] - src[dir][q]) - oldN));
        scale = std::max(scale, std::fabs(oldN));
        for (int t : {na, nb})
          err = std::max(
              err, std::fabs((im[t][q] - src[t][q]) - (x[t][q] - kEbc[t])));
      });
      char what[96];
      std::snprintf(what, sizeof(what),
                    "%s: image - source == affine row (rel.)",
                    faceName(dir, high));
      check(err < 1e-12 * std::max(scale, 1.0), what, err / scale);
    }
}

static void testOpenRows() {
  std::printf("open-rows\n");
  std::mt19937_64 gen(23);
  for (int dir = 0; dir < 3; dir++)
    for (bool high : {false, true}) {
      const int n[3] = {NX, NY, NZ};
      const int plane = high ? n[dir] - 2 : 1;
      Vec d0[3] = {randomVec(N, gen), randomVec(N, gen), randomVec(N, gen)};
      Vec v[3] = {randomVec(N, gen), randomVec(N, gen), randomVec(N, gen)};
      Dev sx(d0[0]), sy(d0[1]), sz(d0[2]);
      Dev ix(d0[0]), iy(d0[1]), iz(d0[2]);
      Dev vx(v[0]), vy(v[1]), vz(v[2]);
      gpuOpenBCSetFace3(sx.p, sy.p, sz.p, kEbc[0], kEbc[1], kEbc[2], dir, plane,
                        NX, NY, NZ);
      gpuOpenBCCopyFace3(ix.p, iy.p, iz.p, vx.p, vy.p, vz.p, dir, plane, NX, NY,
                         NZ);
      const Vec s[3] = {sx.get(), sy.get(), sz.get()};
      const Vec im[3] = {ix.get(), iy.get(), iz.get()};
      std::vector<char> row(N, 0);
      forFaceRows(dir, high, [&](size_t q) { row[q] = 1; });
      double errRow = 0, errOut = 0;
      for (size_t q = 0; q < N; q++)
        for (int c = 0; c < 3; c++) {
          if (row[q]) {
            errRow = std::max(errRow, (double)std::fabs(s[c][q] - kEbc[c]));
            errRow = std::max(errRow, (double)std::fabs(im[c][q] - v[c][q]));
          } else {
            errOut = std::max(errOut, (double)std::fabs(s[c][q] - d0[c][q]));
            errOut = std::max(errOut, (double)std::fabs(im[c][q] - d0[c][q]));
          }
        }
      char what[96];
      std::snprintf(what, sizeof(what), "%s: RHS = E0, image = v on face rows",
                    faceName(dir, high));
      check(errRow == 0.0, what, errRow);
      std::snprintf(what, sizeof(what), "%s: elsewhere untouched",
                    faceName(dir, high));
      check(errOut == 0.0, what, errOut);
    }
}

// ---------------------------------------------------------------------------
static void testMUdot() {
  std::printf("mudot\n");
  const int shapes[][3] = {{9, 8, 7}, {12, 4, 10}, {6, 9, 4}, {4, 4, 4}};
  const int counts[] = {1, 2, 3, 4, 5, 6, 16, 17, 33};
  const T sentinel = 1234.5;
  std::mt19937_64 g(17);
  std::uniform_real_distribution<double> mag(0.5, 100.0);
  double errHost = 0, errKernel = 0;
  bool ghostsKept = true;
  int cases = 0;
  for (const auto& sh : shapes)
    for (const int ns : counts) {
      const int nx = sh[0], ny = sh[1], nz = sh[2];
      const size_t n = (size_t)nx * ny * nz;
      std::vector<double> qom(ns);
      for (int s = 0; s < ns; s++)
        qom[s] = (s % 2 ? 1.0 : -1.0) * mag(g);
      Vec v[3], b[3], be[3];
      for (int c = 0; c < 3; c++) {
        v[c] = randomVec(n, g);
        b[c] = randomVec(n, g, -0.5, 0.5);
        be[c] = randomVec(n, g, -0.2, 0.2);
      }
      const Vec rho = randomVec(n * ns, g, -0.3, 0.3);
      Dev dv[3] = {Dev(v[0]), Dev(v[1]), Dev(v[2])};
      Dev db[3] = {Dev(b[0]), Dev(b[1]), Dev(b[2])};
      Dev dbe[3] = {Dev(be[0]), Dev(be[1]), Dev(be[2])};
      Dev drho(rho);
      const Vec fill(n, sentinel);
      Dev out[3] = {Dev(fill), Dev(fill), Dev(fill)};
      Dev ref[3] = {Dev(fill), Dev(fill), Dev(fill)};

      gpuMUdot(out[0].p, out[1].p, out[2].p, dv[0].p, dv[1].p, dv[2].p, db[0].p,
               db[1].p, db[2].p, dbe[0].p, dbe[1].p, dbe[2].p, drho.p,
               qom.data(), ns, kDt, kC, kFourPI, kDelt, nx, ny, nz);
      for (int s = 0; s < ns; s++)
        gpuMUdotSpecies(ref[0].p, ref[1].p, ref[2].p, dv[0].p, dv[1].p, dv[2].p,
                        db[0].p, db[1].p, db[2].p, dbe[0].p, dbe[1].p, dbe[2].p,
                        drho.p + s * n, 0.5 * qom[s] * kDt / kC,
                        kFourPI / 2.0 * kDelt * kDt / kC * qom[s], nx, ny, nz,
                        s == 0);
      const Vec o[3] = {out[0].get(), out[1].get(), out[2].get()};
      const Vec r[3] = {ref[0].get(), ref[1].get(), ref[2].get()};

      // host MUdot, species by species: (v + v x w + (v.w) w) d_s, w = beta b
      double scale = 0, dh = 0, dk = 0;
      for (int i = 0; i < nx; i++)
        for (int j = 0; j < ny; j++)
          for (int k = 0; k < nz; k++) {
            const size_t q = ((size_t)i * ny + j) * nz + k;
            const bool ghost = i == 0 || i == nx - 1 || j == 0 || j == ny - 1 ||
                               k == 0 || k == nz - 1;
            if (ghost) {
              for (int c = 0; c < 3; c++)
                ghostsKept = ghostsKept && o[c][q] == sentinel;
              continue;
            }
            double h[3] = {0, 0, 0};
            for (int s = 0; s < ns; s++) {
              const double beta = 0.5 * qom[s] * kDt / kC;
              const double w[3] = {beta * (b[0][q] + be[0][q]),
                                   beta * (b[1][q] + be[1][q]),
                                   beta * (b[2][q] + be[2][q])};
              const double vw =
                  v[0][q] * w[0] + v[1][q] * w[1] + v[2][q] * w[2];
              const double d = kFourPI / 2.0 * kDelt * kDt / kC * qom[s] *
                               rho[s * n + q] /
                               (1.0 + w[0] * w[0] + w[1] * w[1] + w[2] * w[2]);
              const double cr[3] = {v[1][q] * w[2] - v[2][q] * w[1],
                                    v[2][q] * w[0] - v[0][q] * w[2],
                                    v[0][q] * w[1] - v[1][q] * w[0]};
              for (int c = 0; c < 3; c++)
                h[c] += (v[c][q] + cr[c] + vw * w[c]) * d;
            }
            for (int c = 0; c < 3; c++) {
              scale = std::max(scale, std::fabs(h[c]));
              dh = std::max(dh, (double)std::fabs(o[c][q] - h[c]));
              dk = std::max(dk, (double)std::fabs(o[c][q] - r[c][q]));
            }
          }
      errHost = std::max(errHost, dh / scale);
      errKernel = std::max(errKernel, dk / scale);
      cases++;
    }
  std::printf("  %d box/species-count cases\n", cases);
  check(errHost < 1e-12, "gpuMUdot vs host per-species MUdot", errHost);
  check(errKernel < 1e-12, "gpuMUdot vs gpuMUdotSpecies summed", errKernel);
  check(ghostsKept, "ghost nodes not written", ghostsKept ? 0 : 1);
}

int main(int argc, char** argv) {
  std::string which = "all";
  for (int a = 1; a + 1 < argc; a++)
    if (std::strcmp(argv[a], "--case") == 0)
      which = argv[a + 1];
  if (which == "all" || which == "pec-linear")
    testPecLinear();
  if (which == "all" || which == "pec-equivalent")
    testPecEquivalent();
  if (which == "all" || which == "open-rows")
    testOpenRows();
  if (which == "all" || which == "mudot")
    testMUdot();
  std::printf("%s (%d failure%s)\n", failures ? "FAILED" : "PASSED", failures,
              failures == 1 ? "" : "s");
  return failures ? 1 : 0;
}
