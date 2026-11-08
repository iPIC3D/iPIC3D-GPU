/* Unit tests of GPU kernels of the implicit Maxwell solve (used by
 * EMfields3D::gpuMaxwellImage / gpuMaxwellSource): the boundary rows,
 * MUdot, PointList/FastDiv and the launch sets and kernels of the image.
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
 *                          nodes are not written.  On a box of two runs
 *                          along x (PointBox), and on a list of two boxes
 *                          (PointList), it writes those nodes only, with the
 *                          same values.
 *   --case point-list      FastDiv against / (every divisor up to 4096, edge
 *                          and random numerators below 2^31); PointList
 *                          enumeration on the device equal to PointList::at
 *                          on the host, empty boxes left out.
 *   --case image-boxes     maxwellImageBoxes on subdomains of 1-5 and 9
 *                          interior cells per axis, flat axes and an outflow
 *                          face: the split condition and one node shell box
 *                          per split axis; shells and interior cover the
 *                          whole operator, each point once; the interiors
 *                          read no ghost cell and write no cell the exchange
 *                          reads; the cell shells read only D of the MUdot
 *                          shells; the whole and interior sets are lists of
 *                          one box (the D and cell interiors empty with a
 *                          flat axis);
 *                          cellIndex natural, splitCellIndex stride 0 along
 *                          flat axes.
 *   --case image-kernels   gpuMUdot, gpuMaxwellLocalCenterOps and
 *                          gpuMaxwellLocalNodeFused launched on the whole list
 *                          and on the shells plus the interior equal one 3D
 *                          launch on the whole box, bitwise, the node kernel
 *                          of a flat axis in the split launches reading the
 *                          cell itself (splitCellIndex) where the whole launch
 *                          reads the ghost cells equal to it.
 *
 * Array layout as in the code: (i * ny + j) * nz + k; the image rows are
 * written into the interleaved Krylov image of gpuPhys2Solver3. */
#include "GPUMaxwellLocal.cuh"
#include "GPUPhysicsKernels.cuh"
#include "cudaTypeDef.cuh"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <deque>
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

// Krylov layout of the image rows (gpuPhys2Solver3): interior node (i, j, k)
// is entry ((i-1)(NY-2) + j-1)(NZ-2) + k-1, 3 interleaved components
constexpr size_t NK = (size_t)(NX - 2) * (NY - 2) * (NZ - 2);
static size_t kAt(int i, int j, int k) {
  return ((size_t)(i - 1) * (NY - 2) + j - 1) * (NZ - 2) + k - 1;
}
template <typename F> static void forInterior(F fn) {
  for (int i = 1; i < NX - 1; i++)
    for (int j = 1; j < NY - 1; j++)
      for (int k = 1; k < NZ - 1; k++)
        fn(at(i, j, k), kAt(i, j, k));
}
// node arrays -> Krylov vector (interior), and back (ghost nodes 0)
static Vec pack(const Vec v[3]) {
  Vec K(3 * NK);
  forInterior([&](size_t q, size_t p) {
    for (int c = 0; c < 3; c++)
      K[3 * p + c] = v[c][q];
  });
  return K;
}
static void unpack(const Vec& K, Vec out[3]) {
  for (int c = 0; c < 3; c++)
    out[c] = Vec(N, 0.0);
  forInterior([&](size_t q, size_t p) {
    for (int c = 0; c < 3; c++)
      out[c][q] = K[3 * p + c];
  });
}

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

// conductor image rows into a zero Krylov image, returned as node arrays
static void image(const Plasma& p, const Vec v[3], Vec out[3], int dir,
                  bool high) {
  Dev vx(v[0]), vy(v[1]), vz(v[2]);
  Dev im(Vec(3 * NK, 0.0));
  gpuPerfectConductorImage(im.p, vx.p, vy.p, vz.p, p.dB[0]->p, p.dB[1]->p,
                           p.dB[2]->p, p.dBe[0]->p, p.dBe[1]->p, p.dBe[2]->p,
                           p.drho->p, p.dqom->p, NS, kDt, kC, kFourPI, kDelt,
                           NX, NY, NZ, dir, high);
  unpack(im.get(), out);
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
      Dev im(pack(d0)); // the image rows go into the Krylov image
      Dev vx(v[0]), vy(v[1]), vz(v[2]);
      gpuOpenBCSetFace3(sx.p, sy.p, sz.p, kEbc[0], kEbc[1], kEbc[2], dir, plane,
                        NX, NY, NZ);
      gpuOpenBCCopyFace3(im.p, vx.p, vy.p, vz.p, dir, plane, NX, NY, NZ);
      const Vec s[3] = {sx.get(), sy.get(), sz.get()};
      const Vec K = im.get();
      std::vector<char> row(N, 0);
      forFaceRows(dir, high, [&](size_t q) { row[q] = 1; });
      double errRow = 0, errOut = 0;
      for (size_t q = 0; q < N; q++)
        for (int c = 0; c < 3; c++) {
          if (row[q])
            errRow = std::max(errRow, (double)std::fabs(s[c][q] - kEbc[c]));
          else
            errOut = std::max(errOut, (double)std::fabs(s[c][q] - d0[c][q]));
        }
      forInterior([&](size_t q, size_t p) {
        for (int c = 0; c < 3; c++) {
          const double e =
              std::fabs(K[3 * p + c] - (row[q] ? v[c][q] : d0[c][q]));
          if (row[q])
            errRow = std::max(errRow, e);
          else
            errOut = std::max(errOut, e);
        }
      });
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
  bool ghostsKept = true, boxExact = true, listExact = true;
  int cases = 0, boxCases = 0;
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

      const PointBox interior = {pointRange(1, nx - 2), pointRange(1, ny - 2),
                                 pointRange(1, nz - 2)};
      gpuMUdot(out[0].p, out[1].p, out[2].p, dv[0].p, dv[1].p, dv[2].p, db[0].p,
               db[1].p, db[2].p, dbe[0].p, dbe[1].p, dbe[2].p, drho.p,
               qom.data(), ns, kDt, kC, kFourPI, kDelt, nx, ny, nz, interior);
      // the boundary layers {1, 2, nx-3, nx-2} along x (as a split image
      // launches them), one run along y and z
      const PointBox layers = {pointRuns(1, 2, nx - 3, nx - 2),
                               pointRange(2, ny - 2), pointRange(1, nz - 3)};
      // and a list of the layers and the row j = 1 between them (empty when
      // nx < 7, then left out of the list)
      const PointBox row = {pointRange(3, nx - 4), pointRange(1, 1),
                            pointRange(1, nz - 2)};
      const PointBox both[2] = {layers, row};
      const PointList list = pointList(both, 2);
      Dev part[3] = {Dev(fill), Dev(fill), Dev(fill)};
      Dev listed[3] = {Dev(fill), Dev(fill), Dev(fill)};
      if (nx >= 6) {
        gpuMUdot(part[0].p, part[1].p, part[2].p, dv[0].p, dv[1].p, dv[2].p,
                 db[0].p, db[1].p, db[2].p, dbe[0].p, dbe[1].p, dbe[2].p,
                 drho.p, qom.data(), ns, kDt, kC, kFourPI, kDelt, nx, ny, nz,
                 layers);
        gpuMUdot(listed[0].p, listed[1].p, listed[2].p, dv[0].p, dv[1].p,
                 dv[2].p, db[0].p, db[1].p, db[2].p, dbe[0].p, dbe[1].p,
                 dbe[2].p, drho.p, qom.data(), ns, kDt, kC, kFourPI, kDelt, nx,
                 ny, nz, list);
      }
      for (int s = 0; s < ns; s++)
        gpuMUdotSpecies(ref[0].p, ref[1].p, ref[2].p, dv[0].p, dv[1].p, dv[2].p,
                        db[0].p, db[1].p, db[2].p, dbe[0].p, dbe[1].p, dbe[2].p,
                        drho.p + s * n, 0.5 * qom[s] * kDt / kC,
                        kFourPI / 2.0 * kDelt * kDt / kC * qom[s], nx, ny, nz,
                        s == 0);
      const Vec o[3] = {out[0].get(), out[1].get(), out[2].get()};
      const Vec r[3] = {ref[0].get(), ref[1].get(), ref[2].get()};
      if (nx >= 6) {
        const Vec pt[3] = {part[0].get(), part[1].get(), part[2].get()};
        const Vec ls[3] = {listed[0].get(), listed[1].get(), listed[2].get()};
        for (int i = 0; i < nx; i++)
          for (int j = 0; j < ny; j++)
            for (int k = 0; k < nz; k++) {
              const size_t q = ((size_t)i * ny + j) * nz + k;
              const bool in =
                  (i == 1 || i == 2 || i == nx - 3 || i == nx - 2) && j >= 2 &&
                  j <= ny - 2 && k >= 1 && k <= nz - 3;
              const bool inRow =
                  i >= 3 && i <= nx - 4 && j == 1 && k >= 1 && k <= nz - 2;
              for (int c = 0; c < 3; c++) {
                boxExact = boxExact && pt[c][q] == (in ? o[c][q] : sentinel);
                listExact =
                    listExact && ls[c][q] == (in || inRow ? o[c][q] : sentinel);
              }
            }
        boxCases++;
      }

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
  std::printf("  %d two-run box cases\n", boxCases);
  check(boxCases > 0 && boxExact, "box: its nodes only, same values",
        boxExact ? 0 : 1);
  check(boxCases > 0 && listExact, "list: its nodes only, same values",
        listExact ? 0 : 1);
}

// ---------------------------------------------------------------------------
// Launch boxes of the Maxwell image

/** Calls f(i, j, k) for each point of box b. */
template <typename F> static void forBox(const PointBox& b, F f) {
  for (int ti = 0; ti < b.i.n; ti++)
    for (int tj = 0; tj < b.j.n; tj++)
      for (int tk = 0; tk < b.k.n; tk++)
        f(b.i.at(ti), b.j.at(tj), b.k.at(tk));
}

static bool inSpan(const PointSpan& s, int x) {
  for (int t = 0; t < s.n; t++)
    if (s.at(t) == x)
      return true;
  return false;
}

static bool inBox(const PointBox& b, int i, int j, int k) {
  return inSpan(b.i, i) && inSpan(b.j, j) && inSpan(b.k, k);
}

/** Calls f(i, j, k) for each point of list L, in its order (PointList::at). */
template <typename F> static void forList(const PointList& L, F f) {
  for (int p = 0; p < L.count(); p++) {
    int i, j, k;
    L.at(p, i, j, k);
    f(i, j, k);
  }
}

/** All points of L in its first box. */
static bool oneBox(const PointList& L) {
  return !L.empty() && L.first[1] == L.count();
}

static bool inList(const PointList& L, int i, int j, int k) {
  for (int b = 0; b < PointList::kMaxBoxes; b++)
    if (L.first[b + 1] > L.first[b] && inBox(L.box[b], i, j, k))
      return true;
  return false;
}

/** The list holds the non-empty boxes of boxes[0..n-1] in order, and
 *  enumerates each box with k fastest, then j, then i. */
static bool listMatches(const PointList& L, const PointBox* boxes, int n) {
  std::vector<std::array<int, 3>> expected, got;
  for (int s = 0; s < n; s++)
    forBox(boxes[s],
           [&](int i, int j, int k) { expected.push_back({i, j, k}); });
  forList(L, [&](int i, int j, int k) { got.push_back({i, j, k}); });
  bool ok = L.first[0] == 0 && expected == got;
  for (int b = 0; b < PointList::kMaxBoxes; b++)
    ok = ok && L.first[b + 1] >= L.first[b];
  return ok;
}

__global__ void k_listPoints(int* out, PointList L) {
  int i, j, k;
  if (!pointOf(L, i, j, k))
    return;
  const int p = blockIdx.x * blockDim.x + threadIdx.x;
  out[3 * p] = i;
  out[3 * p + 1] = j;
  out[3 * p + 2] = k;
}

static void testPointList() {
  std::printf("point-list\n");
  // FastDiv: every divisor up to 4096 and some large ones; all numerators
  // below 4096, the multiples of d and their neighbours near 0 and 2^31, and
  // random ones
  std::mt19937_64 g(5);
  std::uniform_int_distribution<int> anyInt(0, 2147483647);
  std::vector<int> divisors;
  for (int d = 1; d <= 4096; d++)
    divisors.push_back(d);
  for (int d : {65535, 65536, 65537, 1000003, 1 << 30, 2147483647})
    divisors.push_back(d);
  long long tried = 0, wrong = 0;
  for (const int d : divisors) {
    const FastDiv f(d);
    auto test = [&](long long n) {
      if (n < 0 || n > 2147483647)
        return;
      tried++;
      wrong += f.div(static_cast<int>(n)) != static_cast<int>(n / d);
    };
    for (int n = 0; n < 4096; n++)
      test(n);
    const long long top = 2147483647LL / d * d;
    for (long long m : {(long long)d, 2LL * d, 3LL * d, top - d, top})
      for (int e = -1; e <= 1; e++)
        test(m + e);
    for (int r = 0; r < 64; r++)
      test(anyInt(g));
  }
  std::printf("  %lld quotients checked\n", tried);
  check(wrong == 0, "FastDiv == integer division", (double)wrong);

  // device enumeration == host enumeration, for random lists of 1..3 boxes,
  // empty boxes included
  std::uniform_int_distribution<int> lo(0, 5), len(0, 9), gap(0, 4);
  auto span = [&]() {
    const int a = lo(g), n1 = len(g);
    return gap(g) % 2 ? pointRuns(a, a + n1, a + n1 + 2 + gap(g),
                                  a + n1 + 2 + gap(g) + len(g))
                      : pointRange(a, a + n1 - 1);
  };
  bool same = true, structure = true;
  for (int t = 0; t < 200; t++) {
    const int n = 1 + t % 3;
    PointBox boxes[3];
    for (int s = 0; s < n; s++)
      boxes[s] = {span(), span(), span()};
    const PointList L = pointList(boxes, n);
    structure = structure && listMatches(L, boxes, n);
    if (L.empty())
      continue;
    std::vector<int> host(3 * L.count());
    for (int p = 0; p < L.count(); p++)
      L.at(p, host[3 * p], host[3 * p + 1], host[3 * p + 2]);
    int* d = nullptr;
    cudaErrChk(cudaMalloc(&d, host.size() * sizeof(int)));
    const PointLaunch launch = pointLaunch(L);
    k_listPoints<<<launch.grid, launch.block>>>(d, L);
    cudaErrChk(cudaGetLastError());
    std::vector<int> dev(host.size());
    cudaErrChk(cudaMemcpy(dev.data(), d, dev.size() * sizeof(int),
                          cudaMemcpyDeviceToHost));
    cudaErrChk(cudaFree(d));
    same = same && dev == host;
  }
  check(structure, "lists: boxes in order, k fastest, empty boxes out",
        structure ? 0 : 1);
  check(same, "device enumeration == PointList::at", same ? 0 : 1);
}

/** The split operator covers each point of whole once: shells plus interior,
 *  nothing outside whole (dims: the array extents). */
static bool coversOnce(const PointBox& whole, const PointList& shells,
                       const PointList& interior, const int dims[3]) {
  std::vector<int> count((size_t)dims[0] * dims[1] * dims[2], 0);
  auto add = [&](int i, int j, int k) {
    count[((size_t)i * dims[1] + j) * dims[2] + k]++;
  };
  forList(shells, add);
  forList(interior, add);
  bool ok = true;
  for (int i = 0; i < dims[0]; i++)
    for (int j = 0; j < dims[1]; j++)
      for (int k = 0; k < dims[2]; k++)
        ok = ok && count[((size_t)i * dims[1] + j) * dims[2] + k] ==
                       (inBox(whole, i, j, k) ? 1 : 0);
  return ok;
}

static void testImageBoxes() {
  std::printf("image-boxes\n");
  const int sizes[] = {3, 4, 5, 6, 7, 11}; // cells per axis, ghosts included
  int configs = 0, splits = 0;
  bool okSplit = true, okWhole = true, okCover = true, okNodeRead = true,
       okSend = true, okDep = true, okIndex = true;
  for (const int nx : sizes)
    for (const int ny : sizes)
      for (const int nz : sizes)
        for (int outflow = 0; outflow < 2; outflow++)
          for (int flatMask = 0; flatMask < 8; flatMask++) {
            const int nc[3] = {nx, ny, nz};
            const int nn[3] = {nx + 1, ny + 1, nz + 1};
            bool flat[3], skip = false;
            for (int a = 0; a < 3; a++) {
              flat[a] = (flatMask >> a) & 1;
              skip = skip || (flat[a] && nc[a] != 3); // single cell only
            }
            if (skip || (outflow && flat[0]))
              continue;
            const int cellEnd[3] = {outflow ? nx - 1 : nx - 2, ny - 2, nz - 2};
            const MaxwellImageBoxes b =
                maxwellImageBoxes(nc, cellEnd[0], flat, true);
            configs++;

            int nSplit = 0;
            bool small = false;
            for (int a = 0; a < 3; a++)
              if (!flat[a]) {
                nSplit++;
                small = small || nc[a] - 2 < 3;
              }
            const bool expect = nSplit > 0 && !small;
            int shellBoxes = 0; // one node shell box per split axis
            for (int s = 0; s < PointList::kMaxBoxes; s++)
              shellBoxes += b.nodeShells.first[s + 1] > b.nodeShells.first[s];
            okSplit = okSplit && b.split == expect &&
                      shellBoxes == (expect ? nSplit : 0);

            // whole operator: lists of one box
            const PointBox& M = b.mudot.box[0];
            const PointBox& C = b.cells.box[0];
            const PointBox& N = b.nodes.box[0];
            okWhole = okWhole && oneBox(b.mudot) && oneBox(b.cells) &&
                      oneBox(b.nodes);
            for (int a = 0; a < 3; a++) {
              const PointSpan m[3] = {M.i, M.j, M.k};
              const PointSpan c[3] = {C.i, C.j, C.k};
              const PointSpan n[3] = {N.i, N.j, N.k};
              okWhole = okWhole && m[a].n == nn[a] - 2 && m[a].at(0) == 1 &&
                        m[a].at(m[a].n - 1) == nn[a] - 2 &&
                        c[a].n == cellEnd[a] && c[a].at(0) == 1 &&
                        c[a].at(c[a].n - 1) == cellEnd[a] &&
                        n[a].n == nn[a] - 2 && n[a].at(0) == 1 &&
                        n[a].at(n[a].n - 1) == nn[a] - 2;
            }
            const int natural[3] = {ny * nz, nz, 1};
            okIndex = okIndex && b.cellIndex.base == 0 &&
                      b.cellIndex.si == natural[0] &&
                      b.cellIndex.sj == natural[1] &&
                      b.cellIndex.sk == natural[2];
            if (!b.split)
              continue;
            splits++;

            okCover = okCover &&
                      coversOnce(M, b.mudotShells, b.mudotInterior, nn) &&
                      coversOnce(C, b.cellShells, b.cellInterior, nc) &&
                      coversOnce(N, b.nodeShells, b.nodeInterior, nn);
            // the interiors are lists of one box; the D and cell interiors
            // are empty with a flat axis
            for (const PointList* in :
                 {&b.mudotInterior, &b.cellInterior, &b.nodeInterior})
              okWhole = okWhole && (in->empty() || oneBox(*in));

            // the interior nodes read cells (i-1..i) inside [1..nc-2] along
            // the split axes: no ghost cell
            forList(b.nodeInterior, [&](int i, int j, int k) {
              const int p[3] = {i, j, k};
              for (int a = 0; a < 3; a++)
                if (!flat[a])
                  okNodeRead = okNodeRead && p[a] - 1 >= 1 && p[a] <= nc[a] - 2;
            });
            // the interior cells hold no cell the exchange reads: no split
            // coordinate 1 or nc-2 (the faces it sends and copies), no
            // outflow ghost plane cell, and no flat axis (it copies the cell
            // of one for every cell of the others)
            forList(b.cellInterior, [&](int i, int j, int k) {
              const int p[3] = {i, j, k};
              for (int a = 0; a < 3; a++)
                okSend = okSend && !flat[a] && p[a] != 1 && p[a] != nc[a] - 2 &&
                         p[a] <= nc[a] - 2;
            });
            // a shell cell reads the nodes (i..i+1)x(j..j+1)x(k..k+1): D of
            // a MUdot shell, or the outflow ghost node plane nxn-1
            forList(b.cellShells, [&](int i, int j, int k) {
              for (int di = 0; di < 2; di++)
                for (int dj = 0; dj < 2; dj++)
                  for (int dk = 0; dk < 2; dk++) {
                    const int q[3] = {i + di, j + dj, k + dk};
                    if (outflow && q[0] == nn[0] - 1)
                      continue;
                    okDep = okDep && inList(b.mudotShells, q[0], q[1], q[2]);
                  }
            });
            // a flat axis reads its single cell
            const int si[3] = {b.splitCellIndex.si, b.splitCellIndex.sj,
                               b.splitCellIndex.sk};
            int base = 0;
            for (int a = 0; a < 3; a++) {
              okIndex = okIndex && si[a] == (flat[a] ? 0 : natural[a]);
              base += flat[a] ? natural[a] : 0;
            }
            okIndex = okIndex && b.splitCellIndex.base == base;
          }
  std::printf("  %d subdomains, %d split\n", configs, splits);
  check(okSplit, "split when every split axis has at least 3 interior cells",
        okSplit ? 0 : 1);
  check(okWhole, "whole-operator boxes", okWhole ? 0 : 1);
  check(okCover, "shells + interior cover each point once", okCover ? 0 : 1);
  check(okNodeRead, "interior nodes read no ghost cell", okNodeRead ? 0 : 1);
  check(okSend, "interior cells hold no cell the exchange reads",
        okSend ? 0 : 1);
  check(okDep, "cell shells read D of the MUdot shells only", okDep ? 0 : 1);
  check(okIndex, "cell indexing (flat axes: stride 0)", okIndex ? 0 : 1);
}

static void testImageKernels() {
  std::printf("image-kernels\n");
  // cells per axis (ghosts included), flat axes, X-right outflow face
  struct Shape {
    int nc[3];
    bool flat[3];
    bool outflow;
  };
  const Shape shapes[] = {
      {{12, 9, 7}, {false, false, false}, false},
      {{12, 9, 7}, {false, false, false}, true},
      {{10, 3, 8}, {false, true, false}, true},  // x-z run
      {{9, 7, 3}, {false, false, true}, false},  // x-y run
      {{5, 5, 5}, {false, false, false}, false}, // 3 interior cells
      {{8, 3, 3}, {false, true, true}, true}};   // x only
  const T sentinel = 1234.5;
  const T ix = 1.7, iy = 0.9, iz = 1.3, dt2 = 0.37;
  std::mt19937_64 g(99);
  bool okMudot = true, okCells = true, okNodes = true;
  int shapesRun = 0;
  for (const Shape& sh : shapes) {
    const int nxc = sh.nc[0], nyc = sh.nc[1], nzc = sh.nc[2];
    const int nxn = nxc + 1, nyn = nyc + 1, nzn = nzc + 1;
    const size_t nC = (size_t)nxc * nyc * nzc, nN = (size_t)nxn * nyn * nzn;
    const size_t nK = 3 * (size_t)(nxn - 2) * (nyn - 2) * (nzn - 2);
    const MaxwellImageBoxes b =
        maxwellImageBoxes(sh.nc, sh.outflow ? nxc - 1 : nxc - 2, sh.flat, true);
    if (!b.split) {
      check(false, "shape expected to split", 1);
      continue;
    }
    shapesRun++;

    // MUdot: whole box vs shells + interior
    const int ns = 2;
    const std::vector<double> qom = {-25.0, 1.0};
    Vec v[3], bb[3], be[3];
    for (int c = 0; c < 3; c++) {
      v[c] = randomVec(nN, g);
      bb[c] = randomVec(nN, g, -0.5, 0.5);
      be[c] = randomVec(nN, g, -0.2, 0.2);
    }
    const Vec rho = randomVec(nN * ns, g, -0.3, 0.3);
    Dev dv[3] = {Dev(v[0]), Dev(v[1]), Dev(v[2])};
    Dev db[3] = {Dev(bb[0]), Dev(bb[1]), Dev(bb[2])};
    Dev dbe[3] = {Dev(be[0]), Dev(be[1]), Dev(be[2])};
    Dev drho(rho);
    const Vec fillN(nN, sentinel);
    Dev mWhole[3] = {Dev(fillN), Dev(fillN), Dev(fillN)};
    Dev mList[3] = {Dev(fillN), Dev(fillN), Dev(fillN)};
    Dev mSplit[3] = {Dev(fillN), Dev(fillN), Dev(fillN)};
    auto mudot = [&](Dev* out, const auto& box) {
      gpuMUdot(out[0].p, out[1].p, out[2].p, dv[0].p, dv[1].p, dv[2].p, db[0].p,
               db[1].p, db[2].p, dbe[0].p, dbe[1].p, dbe[2].p, drho.p,
               qom.data(), ns, kDt, kC, kFourPI, kDelt, nxn, nyn, nzn, box);
    };
    mudot(mWhole, b.mudot.box[0]); // reference: 3D launch of the whole box
    mudot(mList, b.mudot);
    mudot(mSplit, b.mudotShells);
    mudot(mSplit, b.mudotInterior);
    for (int c = 0; c < 3; c++) {
      const Vec w = mWhole[c].get();
      okMudot = okMudot && w == mList[c].get() && w == mSplit[c].get();
    }

    // cell arrays from E (v) and D (the MUdot output)
    const Vec fillC(nC, sentinel);
    std::deque<Dev> cWhole, cList, cSplit; // a deque never moves a Dev
    for (int a = 0; a < 10; a++) {
      cWhole.emplace_back(fillC);
      cList.emplace_back(fillC);
      cSplit.emplace_back(fillC);
    }
    auto cellOps = [&](std::deque<Dev>& o, const auto& box) {
      gpuMaxwellLocalCenterOps(o[0].p, o[1].p, o[2].p, o[3].p, o[4].p, o[5].p,
                               o[6].p, o[7].p, o[8].p, o[9].p, dv[0].p, dv[1].p,
                               dv[2].p, mWhole[0].p, mWhole[1].p, mWhole[2].p,
                               nxc, nyc, nzc, ix, iy, iz, box);
    };
    cellOps(cWhole, b.cells.box[0]);
    cellOps(cList, b.cells);
    cellOps(cSplit, b.cellShells);
    cellOps(cSplit, b.cellInterior);
    for (int a = 0; a < 10; a++) {
      const Vec w = cWhole[a].get();
      okCells = okCells && w == cList[a].get() && w == cSplit[a].get();
    }

    // node kernel: random cell arrays whose ghosts along a flat axis equal
    // the cell (as the periodic exchange leaves them)
    std::vector<Vec> cell(10);
    std::deque<Dev> dcell;
    for (int a = 0; a < 10; a++) {
      cell[a] = randomVec(nC, g);
      for (int i = 0; i < nxc; i++)
        for (int j = 0; j < nyc; j++)
          for (int k = 0; k < nzc; k++) {
            int q[3] = {i, j, k};
            for (int d = 0; d < 3; d++)
              if (sh.flat[d])
                q[d] = 1;
            cell[a][((size_t)i * nyc + j) * nzc + k] =
                cell[a][((size_t)q[0] * nyc + q[1]) * nzc + q[2]];
          }
      dcell.emplace_back(cell[a]);
    }
    const Vec fillK(nK, sentinel);
    Dev kWhole(fillK), kList(fillK), kSplit(fillK);
    auto nodeOps = [&](Dev& out, const auto& box, const CellIndex& ci) {
      gpuMaxwellLocalNodeFused(
          out.p, dcell[0].p, dcell[1].p, dcell[2].p, dcell[3].p, dcell[4].p,
          dcell[5].p, dcell[6].p, dcell[7].p, dcell[8].p, dcell[9].p, dv[0].p,
          dv[1].p, dv[2].p, mWhole[0].p, mWhole[1].p, mWhole[2].p, nxn, nyn,
          nzn, ix, iy, iz, dt2, box, ci);
    };
    nodeOps(kWhole, b.nodes.box[0], b.cellIndex);
    nodeOps(kList, b.nodes, b.cellIndex);
    nodeOps(kSplit, b.nodeShells, b.splitCellIndex);
    nodeOps(kSplit, b.nodeInterior, b.splitCellIndex);
    const Vec kw = kWhole.get();
    okNodes = okNodes && kw == kList.get() && kw == kSplit.get() &&
              std::find(kw.begin(), kw.end(), sentinel) == kw.end();
  }
  std::printf("  %d subdomains\n", shapesRun);
  check(okMudot, "MUdot: whole list, shells + interior == 3D whole box",
        okMudot ? 0 : 1);
  check(okCells, "cell arrays: whole list, shells + interior == 3D whole box",
        okCells ? 0 : 1);
  check(okNodes,
        "image: whole list, shells + interior == 3D whole box, all rows",
        okNodes ? 0 : 1);
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
  if (which == "all" || which == "point-list")
    testPointList();
  if (which == "all" || which == "image-boxes")
    testImageBoxes();
  if (which == "all" || which == "image-kernels")
    testImageKernels();
  std::printf("%s (%d failure%s)\n", failures ? "FAILED" : "PASSED", failures,
              failures == 1 ? "" : "s");
  return failures ? 1 : 0;
}
