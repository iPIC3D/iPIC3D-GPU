/**
 * @file planetTest.cu
 * @brief Unit test for the planet boundary-condition kernels defined in
 *        planetKernel.cu.
 *
 * What is tested
 * ──────────────
 *  1. Ion charge reduction (planetChargeReductionKernel)
 *     → per-cycle ion charge and diagnostic accumulator
 *  2. Energy computation  (planetEnergyKernel)
 *     → energy of ONE physical electron, (gamma-1) c^2 / |qom|, independent of
 *       the macroparticle weight |q|; diagnostic hit-charge accumulation
 *       (including a multi-block launch)
 *  3. Bitonic sort         (bitonicPadKernel + bitonicSortStepKernel)
 *     → descending energy order; a light fast electron outranks a heavy slow
 *       one
 *  4. Charge cutoff        (planetWarpChargeKernel + chargeCutoffKernel) over
 *     four cycles
 *     → nearest-match cutoff, charge carried across cycles (both signs),
 *       ions-only cycle, conservation ∑|q_ion| - ∑|q_absorbed| == residual,
 *       diagnostics; then 5077 electrons of two species against the rule
 *       applied one electron at a time on the host
 *  5. Reflect + compact    (planetReflectCompactKernel, specular)
 *     → the survivors, compacted per species, are moved out to 2R - r and
 *       keep their outward velocity
 *  6. Diffuse reflection   (planetReflectCompactKernel, diffuse), 3D sphere
 *     and 2D circle
 *     → speed (all three components) kept, placed at max(1.05 R, 2R - r),
 *       outward, Lambert law: <cos θ> = 2/3, cos²θ uniform
 *       (Kolmogorov-Smirnov), isotropic tangential directions
 *  7. Specular geometry    (planetReflectCompactKernel, specular), 3D sphere
 *     and 2D circle
 *     → placement at max(1.05 R, 2R - r), only an inward normal velocity
 *       component reversed; an electron at the exact centre leaves along +x
 *       with a finite velocity (specular and diffuse)
 *
 * Test data (c = 1)
 * ─────────────────
 *  • ion species (global 0, qom = +1): 5 particles, q = +2.0 → |Q_ion| = 10
 *  • electron species A (global 1, qom = -1): 4 particles
 *  • electron species B (global 2, qom = -2): 3 particles
 *
 *    pcl   gidx   q      u      per-electron energy
 *    A0     0   -0.25   0.40   0.0910895   (light and fast)
 *    A1     1   -4.00   0.30   0.0482848   (heavy and slower)
 *    A2     2   -1.00   0.20   0.0206207
 *    A3     3   -0.50   0.10   0.0050378
 *    B0     4   -2.50   0.50   0.0773503
 *    B1     5   -1.50   0.35   0.0337606
 *    B2     6   -1.00   0.05   0.0006262
 *
 *  Sorted (descending energy): A0 B0 A1 B1 A2 A3 B2
 *  |q| in that order:          0.25 2.5 4 1.5 1 0.5 1   (prefix 0.25 2.75 6.75
 *                                                         8.25 9.25 9.75 10.75)
 *  Weighting the energy by |q| would rank A1 above A0.
 *
 *  Cycles (an electron is absorbed while it brings the absorbed charge closer
 *  to target = residual + ion charge):
 *    1: ions 10, target 10    → absorb 6 (9.75),  residual +0.25
 *       (a ">= target" rule would absorb all 7: 10.75)
 *    2: ions 3, no electrons  → absorb 0,         residual +3.25 (limited)
 *    3: no ions, target 3.25  → absorb 2 (2.75),  residual +0.50
 *    4: ions 2,  target 2.5   → absorb 2 (2.75),  residual -0.25
 */

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <initializer_list>
#include <iomanip>
#include <iostream>
#include <memory>
#include <numeric>
#include <random>
#include <vector>

// planetKernel.cuh pulls in cudaTypeDef, particleArrayCUDA, particleExchange,
// hashedSum, arrayCUDA — in the correct order.
#include "planetKernel.cuh"

// ─── helpers ───────────────────────────────────────────────────

static constexpr int BLOCK = 256;

/** Next power of two >= n */
static int nextPow2(int n) {
  int p = 1;
  while (p < n)
    p <<= 1;
  return p;
}

/** Device buffer of n values, not initialised; release with cudaFree. */
template <typename T> static T* deviceBuffer(size_t n) {
  T* device;
  cudaErrChk(cudaMalloc(&device, n * sizeof(T)));
  return device;
}

// The host buffers of the test that are copied to or from the device are
// page-locked: PinnedVector for the values, PinnedObject for the objects.

/** Device copy of the host values; release with cudaFree. */
template <typename T> static T* toDevice(const PinnedVector<T>& host) {
  T* device = deviceBuffer<T>(host.size());
  cudaErrChk(cudaMemcpy(device, host.data(), host.size() * sizeof(T),
                        cudaMemcpyHostToDevice));
  return device;
}
template <typename T> static T* toDevice(std::initializer_list<T> values) {
  return toDevice(PinnedVector<T>(values));
}

/** One value to the device. */
template <typename T> static void setDevice(T* device, const T& value) {
  const PinnedVector<T> host(1, value);
  cudaErrChk(
      cudaMemcpy(device, host.data(), sizeof(T), cudaMemcpyHostToDevice));
}

/** Host copy of n device values. */
template <typename T>
static PinnedVector<T> fromDevice(const T* device, size_t n) {
  PinnedVector<T> host(n);
  cudaErrChk(
      cudaMemcpy(host.data(), device, n * sizeof(T), cudaMemcpyDeviceToHost));
  return host;
}

/** An object in page-locked host memory (copyToDevice copies it). */
struct PinnedDelete {
  template <typename T> void operator()(T* object) const {
    deleteHostPinnedObject(object);
  }
};
template <typename T> using PinnedObject = std::unique_ptr<T, PinnedDelete>;
template <typename T, typename... Args>
static PinnedObject<T> pinnedObject(Args... args) {
  return PinnedObject<T>(newHostPinnedObject<T>(args...));
}

/** Sort the n staged (energy, merged index) pairs by descending energy; the
 *  buffers hold nextPow2(n) entries. */
static void sortByEnergy(cudaParticleType* d_energy, uint32_t* d_idx, int n) {
  const int paddedN = nextPow2(n);
  if (paddedN > n)
    bitonicPadKernel<<<(paddedN - n + BLOCK - 1) / BLOCK, BLOCK>>>(
        d_energy, d_idx, n, paddedN);
  // one thread per compare-and-swap pair
  for (int k = 2; k <= paddedN; k <<= 1)
    for (int j = k >> 1; j > 0; j >>= 1)
      bitonicSortStepKernel<<<(paddedN / 2 + BLOCK - 1) / BLOCK, BLOCK>>>(
          d_energy, d_idx, j, k, paddedN);
  cudaErrChk(cudaDeviceSynchronize());
}

/** The cutoff of one cycle: the warp sums of the sorted electrons, then
 *  chargeCutoffKernel.  d_warpCharge holds
 *  planetWarpChargeCount(e.total, e.nSpecies) values. */
static void runCutoff(const PlanetElectrons& e,
                      const cudaParticleType* d_energy,
                      cudaParticleType* d_warpCharge, PlanetBCState* d_state,
                      cudaParticleType* d_absorbedCharge, int* d_cutoff) {
  if (e.total > 0)
    planetWarpChargeKernel<<<(e.total + BLOCK - 1) / BLOCK, BLOCK>>>(
        e, d_warpCharge);
  chargeCutoffKernel<<<1, 1>>>(e, d_energy, d_warpCharge, d_state,
                               d_absorbedCharge, d_cutoff);
  cudaErrChk(cudaDeviceSynchronize());
}

/** Reference per-electron kinetic energy (gamma - 1) c^2 / |qom|, computed
 *  directly in long double. */
static cudaParticleType hostEnergy(const SpeciesParticle& p,
                                   cudaParticleType qom, cudaParticleType c) {
  const long double v2 = (long double)p.get_u() * p.get_u() +
                         (long double)p.get_v() * p.get_v() +
                         (long double)p.get_w() * p.get_w();
  const long double c2 = (long double)c * c;
  const long double gamma = 1.0L / sqrtl(1.0L - v2 / c2);
  return (cudaParticleType)((gamma - 1.0L) * c2 / fabsl((long double)qom));
}

static bool closeRel(cudaParticleType a, cudaParticleType b,
                     cudaParticleType tol = 1e-12) {
  return fabs(a - b) <= tol * std::max((cudaParticleType)1.0, fabs(b));
}

// ─── printing helpers ──────────────────────────────────────────

static void printParticle(const char* tag, int idx, const SpeciesParticle& p,
                          cudaParticleType energy = -1.0) {
  std::cout << "  " << std::left << std::setw(12) << tag << " [" << std::setw(2)
            << idx << "]"
            << "  pos=(" << std::setw(8) << p.get_x() << ", " << std::setw(8)
            << p.get_y() << ", " << std::setw(8) << p.get_z() << ")"
            << "  vel=(" << std::setw(8) << p.get_u() << ", " << std::setw(8)
            << p.get_v() << ", " << std::setw(8) << p.get_w() << ")"
            << "  q=" << std::setw(8) << p.get_q();
  if (energy >= 0.0)
    std::cout << "  E=" << energy;
  std::cout << "\n";
}

static void printSeparator(const char* title) {
  std::cout << "\n═══════════════════════════════════════════════════\n"
            << " " << title
            << "\n═══════════════════════════════════════════════════\n";
}

static void check(bool ok, const std::string& what, bool& allPassed) {
  if (ok) {
    std::cout << "  ok    " << what << "\n";
  } else {
    std::cerr << "  *** FAIL: " << what << " ***\n";
    allPassed = false;
  }
}

static PlanetBCState readState(const PlanetBCState* d_state) {
  return fromDevice(d_state, 1)[0];
}

static void setIonChargeCycle(PlanetBCState* d_state, cudaParticleType q) {
  setDevice(&d_state->ionChargeCycle, q);
}

/** Radius at which a reflected particle that ended its push at distance r from
 *  the centre is placed: max(1.05 R, 2R - r). */
static double expectedReflectedRadius(double r, double R) {
  return std::max(1.05 * R, 2.0 * R - r);
}

/** Distance from the planet centre and outward unit normal at a position; for
 *  the 2D circle (doSphere == 2) both in the x-z plane. */
static double planetNormalAt(const SpeciesParticle& p, int doSphere, double ox,
                             double oy, double oz, double nv[3]) {
  const double d[3] = {p.get_x() - ox, doSphere == 1 ? p.get_y() - oy : 0.0,
                       p.get_z() - oz};
  const double r = std::sqrt(d[0] * d[0] + d[1] * d[1] + d[2] * d[2]);
  for (int k = 0; k < 3; k++)
    nv[k] = d[k] / r;
  return r;
}

/** Orthonormal tangent basis (t1, t2) of the unit normal nv, built as in the
 *  diffuse kernel; for an x-z plane normal t1 or t2 is the y axis. */
static void tangentBasis(const double nv[3], double t1[3], double t2[3]) {
  if (std::fabs(nv[0]) < 0.9) {
    const double l = std::sqrt(nv[1] * nv[1] + nv[2] * nv[2]);
    t1[0] = 0, t1[1] = nv[2] / l, t1[2] = -nv[1] / l;
  } else {
    const double l = std::sqrt(nv[0] * nv[0] + nv[2] * nv[2]);
    t1[0] = -nv[2] / l, t1[1] = 0, t1[2] = nv[0] / l;
  }
  t2[0] = nv[1] * t1[2] - nv[2] * t1[1];
  t2[1] = nv[2] * t1[0] - nv[0] * t1[2];
  t2[2] = nv[0] * t1[1] - nv[1] * t1[0];
}

/** Run the reflection kernel, specular or diffuse, on one electron species
 *  with every particle surviving (cutoff 0).  Returns the compacted output,
 *  in the kernel's order. */
static PinnedVector<SpeciesParticle>
reflectAll(const PinnedVector<SpeciesParticle>& in, bool diffuse, int doSphere,
           cudaCommonType ox, cudaCommonType oy, cudaCommonType oz,
           cudaCommonType R) {
  const int n = (int)in.size();
  const auto pa = pinnedObject<planetArray>(in.data(), n);
  planetArray* d_pa = pa->copyToDevice();
  PinnedVector<uint32_t> idx(n);
  std::iota(idx.begin(), idx.end(), 0u);
  planetArray** d_ptrs = toDevice({d_pa});
  int* d_offsets = toDevice({0});
  int* d_cutoff = toDevice({0});
  int* d_count = toDevice({0});
  uint32_t* d_idx = toDevice(idx);
  SpeciesParticle* d_out = deviceBuffer<SpeciesParticle>(n);
  const PlanetElectrons electrons = {d_ptrs, d_offsets, d_idx, 1, n};
  const PlanetReflection reflection = {ox,       oy,      oz,    R,
                                       doSphere, diffuse, 12345u};
  planetReflectCompactKernel<<<(n + BLOCK - 1) / BLOCK, BLOCK>>>(
      electrons, d_cutoff, reflection, d_out, d_count);
  cudaErrChk(cudaDeviceSynchronize());
  PinnedVector<SpeciesParticle> out = fromDevice(d_out, n);
  out.resize(fromDevice(d_count, 1)[0]);
  cudaFree(d_ptrs);
  cudaFree(d_offsets);
  cudaFree(d_cutoff);
  cudaFree(d_count);
  cudaFree(d_idx);
  cudaFree(d_out);
  cudaFree(d_pa);
  return out;
}

/** Diffuse reflection of n electrons that ended their push inside the planet
 *  at distances uniform in [0, R) from the centre, with random velocities,
 *  all survivors (cutoff 0): the speed and, for the 2D circle in the x-z
 *  plane, y are kept; the particles sit at max(1.05 R, 2R - r) moving
 *  outward; mu = cos θ about the outward normal follows the Lambert law,
 *  <mu> = 2/3 and mu² uniform; the two tangential direction cosines have mean
 *  0 and mean square 1/4.  The bounds are 5 standard deviations, and
 *  1.95 / sqrt(n) (0.1 %) for the KS distance. */
static bool diffuseReflectionCase(int doSphere, cudaCommonType ox,
                                  cudaCommonType oy, cudaCommonType oz,
                                  cudaCommonType R) {
  const int n = 200000;
  std::mt19937_64 gen(17 + doSphere);
  std::normal_distribution<double> normal;
  PinnedVector<SpeciesParticle> in(n);
  for (int i = 0; i < n; i++) {
    double d[3] = {normal(gen), doSphere == 1 ? normal(gen) : 0.0, normal(gen)};
    const double len = std::sqrt(d[0] * d[0] + d[1] * d[1] + d[2] * d[2]);
    const double r0 = R * std::uniform_real_distribution<double>()(gen);
    in[i].set(0.1 * normal(gen), 0.1 * normal(gen), 0.1 * normal(gen), -1.0,
              ox + r0 * d[0] / len,
              doSphere == 1 ? oy + r0 * d[1] / len : oy + 0.3,
              oz + r0 * d[2] / len, (cudaPclType_ID)i);
  }
  const PinnedVector<SpeciesParticle> out =
      reflectAll(in, true, doSphere, ox, oy, oz, R);
  const int count = (int)out.size();

  double errSpeed = 0, errRadius = 0, errRay = 0, errY = 0, minMu = 1;
  double sMu = 0, sT1 = 0, sT2 = 0, sT1q = 0, sT2q = 0;
  std::vector<double> mu2;
  mu2.reserve(n);
  for (int i = 0; i < count; i++) {
    const SpeciesParticle& p = out[i];
    const SpeciesParticle& q = in[p.get_id()];
    double nv[3], nq[3];
    const double r = planetNormalAt(p, doSphere, ox, oy, oz, nv);
    const double rq = planetNormalAt(q, doSphere, ox, oy, oz, nq);
    const double v[3] = {p.get_u(), p.get_v(), p.get_w()};
    const double s = std::sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
    const double s0 = std::sqrt(q.get_u() * q.get_u() + q.get_v() * q.get_v() +
                                q.get_w() * q.get_w());
    errSpeed = std::max(errSpeed, std::fabs(s - s0) / s0);
    errRadius =
        std::max(errRadius, std::fabs(r - expectedReflectedRadius(rq, R)) / R);
    for (int k = 0; k < 3; k++)
      errRay = std::max(errRay, std::fabs(nv[k] - nq[k]));
    if (doSphere == 2)
      errY = std::max(errY, std::fabs(p.get_y() - q.get_y()));
    double t1[3], t2[3];
    tangentBasis(nv, t1, t2);
    const double mu = (v[0] * nv[0] + v[1] * nv[1] + v[2] * nv[2]) / s;
    const double e1 = (v[0] * t1[0] + v[1] * t1[1] + v[2] * t1[2]) / s;
    const double e2 = (v[0] * t2[0] + v[1] * t2[1] + v[2] * t2[2]) / s;
    minMu = std::min(minMu, mu);
    sMu += mu;
    sT1 += e1;
    sT2 += e2;
    sT1q += e1 * e1;
    sT2q += e2 * e2;
    mu2.push_back(mu * mu);
  }
  std::sort(mu2.begin(), mu2.end());
  double ks = 0;
  for (size_t i = 0; i < mu2.size(); i++)
    ks = std::max({ks, std::fabs((i + 1.0) / mu2.size() - mu2[i]),
                   std::fabs((double)i / mu2.size() - mu2[i])});
  const double N = count, meanMu = sMu / N;
  const double sigMu = std::sqrt(1.0 / 18.0 / N), sigT = std::sqrt(0.25 / N),
               sigTq = std::sqrt(1.0 / 16.0 / N);
  const char* geom = doSphere == 1 ? "3D sphere" : "2D circle";
  std::cout << "  " << geom << ": <mu> = " << meanMu << " (2/3), KS " << ks
            << ", tangential means " << sT1 / N << " " << sT2 / N
            << ", mean squares " << sT1q / N << " " << sT2q / N
            << " (1/4), speed error " << errSpeed << ", radius error "
            << errRadius << ", ray error " << errRay << ", min mu " << minMu
            << "\n";
  bool ok = count == n;
  ok &= errSpeed < 1e-12 && errRadius < 1e-12 && errRay < 1e-12 &&
        errY == 0.0 && minMu > 0.0;
  ok &= std::fabs(meanMu - 2.0 / 3.0) < 5 * sigMu;
  ok &= ks < 1.95 / std::sqrt(N);
  ok &= std::fabs(sT1 / N) < 5 * sigT && std::fabs(sT2 / N) < 5 * sigT;
  ok &= std::fabs(sT1q / N - 0.25) < 5 * sigTq &&
        std::fabs(sT2q / N - 0.25) < 5 * sigTq;
  return ok;
}

/** Specular reflection geometry for one geometry.  Electrons end their push
 *  on five rays (in 2D in the x-z plane, y offsets kept) at r = 0.999, 0.97 R
 *  (placed at 1.05 R) and 0.9, 0.5, 0.05 R (placed at 2R - r), each with an
 *  inward, an outward and a tangential velocity.  Checks: the particle is on
 *  the ray of its end point at max(1.05 R, 2R - r), 2D keeps y, |v| and the
 *  tangential component are kept, v'·n = |v·n|, and an outward velocity is
 *  unchanged bit for bit. */
static bool specularGeometryCase(int doSphere, cudaCommonType ox,
                                 cudaCommonType oy, cudaCommonType oz,
                                 cudaCommonType R) {
  const double rays3D[][3] = {
      {1, 0, 0}, {0, 0, -1}, {0.6, -0.8, 0}, {1, 2, -2}, {-2, 3, 6}};
  const double rays2D[][3] = {
      {1, 0, 0}, {0, 0, -1}, {0.6, 0, 0.8}, {-0.8, 0, -0.6}, {-1, 0, 3}};
  const double radii[] = {0.999, 0.97, 0.9, 0.5, 0.05}; // r / R
  // velocity components along (n, t1, t2)
  enum { INWARD, OUTWARD, TANGENTIAL, NUM_KINDS };
  const double velNT[NUM_KINDS][3] = {
      {-0.3, 0.2, -0.1}, {0.25, -0.15, 0.05}, {0, 0.3, 0.1}};

  PinnedVector<SpeciesParticle> in;
  std::vector<int> kind;
  for (int a = 0; a < 5; a++) {
    const double* ray = doSphere == 1 ? rays3D[a] : rays2D[a];
    const double len =
        std::sqrt(ray[0] * ray[0] + ray[1] * ray[1] + ray[2] * ray[2]);
    const double nv[3] = {ray[0] / len, ray[1] / len, ray[2] / len};
    double t1[3], t2[3];
    tangentBasis(nv, t1, t2);
    for (double rr : radii) {
      for (int k = 0; k < NUM_KINDS; k++) {
        double v[3];
        for (int c = 0; c < 3; c++)
          v[c] =
              velNT[k][0] * nv[c] + velNT[k][1] * t1[c] + velNT[k][2] * t2[c];
        const int id = (int)in.size();
        const double y =
            doSphere == 1 ? oy + rr * R * nv[1] : oy + 0.3 + 0.01 * id;
        SpeciesParticle p;
        p.set(v[0], v[1], v[2], -1.0, ox + rr * R * nv[0], y,
              oz + rr * R * nv[2], (cudaPclType_ID)id);
        in.push_back(p);
        kind.push_back(k);
      }
    }
  }
  const PinnedVector<SpeciesParticle> out =
      reflectAll(in, false, doSphere, ox, oy, oz, R);

  int nOutwardChanged = 0;
  double errRadius = 0, errRay = 0, errY = 0, errSpeed = 0;
  double errTangential = 0, errNormal = 0;
  for (const SpeciesParticle& p : out) {
    const SpeciesParticle& q = in[p.get_id()];
    double nv[3], nq[3];
    const double r = planetNormalAt(p, doSphere, ox, oy, oz, nv);
    const double rq = planetNormalAt(q, doSphere, ox, oy, oz, nq);
    errRadius =
        std::max(errRadius, std::fabs(r - expectedReflectedRadius(rq, R)) / R);
    for (int c = 0; c < 3; c++)
      errRay = std::max(errRay, std::fabs(nv[c] - nq[c]));
    if (doSphere == 2)
      errY = std::max(errY, std::fabs(p.get_y() - q.get_y()));
    const double v[3] = {p.get_u(), p.get_v(), p.get_w()};
    const double v0[3] = {q.get_u(), q.get_v(), q.get_w()};
    const double s = std::sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
    const double s0 = std::sqrt(v0[0] * v0[0] + v0[1] * v0[1] + v0[2] * v0[2]);
    errSpeed = std::max(errSpeed, std::fabs(s - s0) / s0);
    const double vn = v[0] * nq[0] + v[1] * nq[1] + v[2] * nq[2];
    const double vn0 = v0[0] * nq[0] + v0[1] * nq[1] + v0[2] * nq[2];
    errNormal = std::max(errNormal, std::fabs(vn - std::fabs(vn0)) / s0);
    for (int c = 0; c < 3; c++)
      errTangential =
          std::max(errTangential,
                   std::fabs((v[c] - vn * nq[c]) - (v0[c] - vn0 * nq[c])) / s0);
    if (kind[p.get_id()] == OUTWARD &&
        !(v[0] == v0[0] && v[1] == v0[1] && v[2] == v0[2]))
      nOutwardChanged++;
  }

  std::cout << "  " << (doSphere == 1 ? "3D sphere" : "2D circle") << ": "
            << out.size() << " of " << in.size() << " reflected, radius error "
            << errRadius << ", ray error " << errRay << ", speed error "
            << errSpeed << ", normal error " << errNormal
            << ", tangential error " << errTangential
            << ", outward velocities changed " << nOutwardChanged << "\n";
  return out.size() == in.size() && errRadius < 1e-12 && errRay < 1e-12 &&
         errY == 0.0 && errSpeed < 1e-12 && errNormal < 1e-12 &&
         errTangential < 1e-12 && nOutwardChanged == 0;
}

/** An electron exactly at the planet centre: either rule places it at
 *  origin + 2R along +x (y kept in 2D) with a finite velocity of the same
 *  speed. */
static bool centreCase(bool diffuse, int doSphere, cudaCommonType ox,
                       cudaCommonType oy, cudaCommonType oz, cudaCommonType R) {
  SpeciesParticle p;
  p.set(0.1, -0.2, 0.3, -1.0, ox, oy, oz, (cudaPclType_ID)0);
  const PinnedVector<SpeciesParticle> out =
      reflectAll({p}, diffuse, doSphere, ox, oy, oz, R);
  if (out.size() != 1)
    return false;
  const SpeciesParticle& q = out[0];
  const double s = std::sqrt(q.get_u() * q.get_u() + q.get_v() * q.get_v() +
                             q.get_w() * q.get_w());
  return q.get_x() == ox + 2 * R && q.get_y() == oy && q.get_z() == oz &&
         std::isfinite(s) && std::fabs(s - std::sqrt(0.14)) < 1e-12;
}

/** The sort and the cutoff over many electrons: 5077 electrons of two
 *  species behind an empty leading one, with random energies and weights, so
 *  the sort spans several blocks and the cutoff steps over many warps and
 *  ends in a partial one.  The sorted energies must be descending and hold
 *  every electron once.  For targets from below the first electron to beyond
 *  all of them, and a negative one, the cutoff index must equal that of the
 *  rule applied one electron at a time on the host, and the carried charge
 *  and the absorbed charge per species must match it to round-off. */
static bool manyElectronCutoffCase() {
  const int nSpecies = 3; // species 0 has no hits
  const int counts[nSpecies] = {0, 3000, 2077};
  const PinnedVector<int> offsets = {0, 0, 3000};
  const int n = counts[1] + counts[2];
  std::mt19937_64 gen(41);
  std::uniform_real_distribution<double> uniform;
  std::normal_distribution<double> normal;

  PinnedVector<SpeciesParticle> pcls[nSpecies];
  PinnedVector<planetArray*> h_ptrs(nSpecies);
  std::vector<PinnedObject<planetArray>> arrays;
  for (int s = 0; s < nSpecies; s++) {
    pcls[s].resize(std::max(counts[s], 1));
    for (int i = 0; i < counts[s]; i++)
      pcls[s][i].set(0.1 * normal(gen), 0.1 * normal(gen), 0.1 * normal(gen),
                     -s * (0.5 + uniform(gen)), 5.3, 5.0, 5.0,
                     (cudaPclType_ID)i);
    arrays.push_back(pinnedObject<planetArray>(pcls[s].data(), counts[s]));
    h_ptrs[s] = arrays.back()->copyToDevice();
  }
  planetArray** d_ptrs = toDevice(h_ptrs);
  int* d_offsets = toDevice(offsets);
  int* d_cutoff = deviceBuffer<int>(1);
  cudaParticleType* d_energy = deviceBuffer<cudaParticleType>(nextPow2(n));
  uint32_t* d_idx = deviceBuffer<uint32_t>(nextPow2(n));
  cudaParticleType* d_absorbed = deviceBuffer<cudaParticleType>(nSpecies);
  PlanetBCState* d_state = deviceBuffer<PlanetBCState>(1);
  const PlanetElectrons electrons = {d_ptrs, d_offsets, d_idx, nSpecies, n};
  cudaParticleType* d_warpCharge = deviceBuffer<cudaParticleType>(
      planetWarpChargeCount(electrons.total, electrons.nSpecies));

  for (int s = 1; s < nSpecies; s++)
    planetEnergyKernel<<<(counts[s] + BLOCK - 1) / BLOCK, BLOCK>>>(
        h_ptrs[s], counts[s], -1.0, 1.0, d_energy, d_idx, offsets[s], nullptr);
  sortByEnergy(d_energy, d_idx, n);
  const PinnedVector<uint32_t> sorted = fromDevice(d_idx, n);
  const PinnedVector<cudaParticleType> sortedEnergy = fromDevice(d_energy, n);
  // the sort over several blocks: descending energies, every electron once
  PinnedVector<uint32_t> seen(sorted);
  std::sort(seen.begin(), seen.end());
  bool ok = true;
  for (int i = 0; i < n; i++)
    ok &= seen[i] == (uint32_t)i &&
          (i == 0 || sortedEnergy[i] <= sortedEnergy[i - 1]);
  std::cout << "  sorted order of " << n << " electrons "
            << (ok ? "is" : "is NOT")
            << " descending and holds every electron once\n";
  auto chargeOf = [&](uint32_t gidx, int& species) {
    species = gidx < (uint32_t)offsets[2] ? 1 : 2;
    return std::fabs(pcls[species][gidx - offsets[species]].get_q());
  };
  double total = 0;
  int species;
  for (int i = 0; i < n; i++)
    total += chargeOf(sorted[i], species);

  const double first = chargeOf(sorted[0], species);
  const double targets[] = {-1.0,        0.0,           0.4 * first,
                            0.6 * first, 0.01 * total,  0.37 * total,
                            0.5 * total, 0.999 * total, 2.0 * total};
  for (const double target : targets) {
    PlanetBCState st = {};
    st.ionChargeCycle = target;
    setDevice(d_state, st);
    cudaErrChk(cudaMemset(d_absorbed, 0, nSpecies * sizeof(cudaParticleType)));
    runCutoff(electrons, d_energy, d_warpCharge, d_state, d_absorbed, d_cutoff);

    // the rule, one electron at a time
    int expectedCutoff = 0;
    double absorbed = 0, absorbedSpecies[nSpecies] = {0, 0, 0};
    for (; expectedCutoff < n; expectedCutoff++) {
      const double absq = chargeOf(sorted[expectedCutoff], species);
      if (absorbed + 0.5 * absq >= target)
        break;
      absorbed += absq;
      absorbedSpecies[species] += absq;
    }

    const int cutoff = fromDevice(d_cutoff, 1)[0];
    const PinnedVector<cudaParticleType> h_absorbed =
        fromDevice(d_absorbed, nSpecies);
    st = readState(d_state);
    // the sums differ from the host's by their order: round-off of the total
    const double tol = 1e-12 * total;
    const bool pass =
        cutoff == expectedCutoff &&
        std::fabs(st.residualCharge - (target - absorbed)) <= tol &&
        h_absorbed[0] == 0 &&
        std::fabs(h_absorbed[1] - absorbedSpecies[1]) <= tol &&
        std::fabs(h_absorbed[2] - absorbedSpecies[2]) <= tol &&
        st.electronLimitedCycles == (target > total ? 1u : 0u);
    std::cout << "  target " << std::setw(12) << target << " → cutoff "
              << std::setw(4) << cutoff << " (expected " << expectedCutoff
              << "), residual " << st.residualCharge << (pass ? "" : "  ***")
              << "\n";
    ok &= pass;
  }

  cudaFree(d_ptrs);
  cudaFree(d_offsets);
  cudaFree(d_cutoff);
  cudaFree(d_energy);
  cudaFree(d_idx);
  cudaFree(d_warpCharge);
  cudaFree(d_absorbed);
  cudaFree(d_state);
  for (int s = 0; s < nSpecies; s++)
    cudaFree(h_ptrs[s]);
  return ok;
}

// ═══════════════════════════════════════════════════════════════
//  MAIN
// ═══════════════════════════════════════════════════════════════

static int runTest() {
  bool allPassed = true;

  // ─── Planet sphere parameters ──────────────────────────────
  const cudaCommonType originX = 5.0;
  const cudaCommonType originY = 5.0;
  const cudaCommonType originZ = 5.0;
  const cudaCommonType sphereRadius = 1.0;
  const int doSphere = 1; // 3-D

  // ─── Species parameters ────────────────────────────────────
  const cudaParticleType c = 1.0;
  const cudaParticleType qomElec[2] = {-1.0, -2.0};
  const int ns = 3; // ion, electron A, electron B
  const int elecGlobal[2] = {1, 2};
  const int nElecSpecies = 2;

  // ────────────────────────────────────────────────────────────
  //  Particles (all inside the sphere at (5.3, 5, 5), moving along +x)
  // ────────────────────────────────────────────────────────────
  const int nIonPlanet = 5;
  PinnedVector<SpeciesParticle> ionPcls(nIonPlanet);
  for (int i = 0; i < nIonPlanet; i++)
    ionPcls[i].set(/*u*/ 0.01, /*v*/ 0.0, /*w*/ 0.0, /*q*/ 2.0,
                   /*x*/ 5.3, /*y*/ 5.0, /*z*/ 5.0, (cudaParticleType)i);

  const int nElec0 = 4;
  PinnedVector<SpeciesParticle> elec0Pcls(nElec0);
  {
    const cudaParticleType qs[] = {-0.25, -4.0, -1.0, -0.5};
    const cudaParticleType us[] = {0.40, 0.30, 0.20, 0.10};
    for (int i = 0; i < nElec0; i++)
      elec0Pcls[i].set(us[i], 0.0, 0.0, qs[i], 5.3, 5.0, 5.0,
                       (cudaParticleType)i);
  }
  const int nElec1 = 3;
  PinnedVector<SpeciesParticle> elec1Pcls(nElec1);
  {
    const cudaParticleType qs[] = {-2.5, -1.5, -1.0};
    const cudaParticleType us[] = {0.50, 0.35, 0.05};
    for (int i = 0; i < nElec1; i++)
      elec1Pcls[i].set(us[i], 0.0, 0.0, qs[i], 5.3, 5.0, 5.0,
                       (cudaParticleType)i);
  }
  const int totalElecPlanet = nElec0 + nElec1; // 7

  auto pclOf = [&](int gidx) -> const SpeciesParticle& {
    return gidx < nElec0 ? elec0Pcls[gidx] : elec1Pcls[gidx - nElec0];
  };

  // ─── Expected values ───────────────────────────────────────
  const cudaParticleType expectedIonCharge = 10.0;
  const uint32_t expectedSortedGidx[] = {0, 4, 1, 5, 2, 3, 6};
  PinnedVector<cudaParticleType> expectedEnergy(totalElecPlanet);
  for (int g = 0; g < totalElecPlanet; g++)
    expectedEnergy[g] = hostEnergy(pclOf(g), qomElec[g < nElec0 ? 0 : 1], c);

  // ════════════════════════════════════════════════════════════
  //  Device buffers
  // ════════════════════════════════════════════════════════════
  const auto ionPA = pinnedObject<planetArray>(ionPcls.data(), nIonPlanet);
  planetArray* d_ionPA = ionPA->copyToDevice();
  const auto elec0PA = pinnedObject<planetArray>(elec0Pcls.data(), nElec0);
  planetArray* d_elec0PA = elec0PA->copyToDevice();
  const auto elec1PA = pinnedObject<planetArray>(elec1Pcls.data(), nElec1);
  planetArray* d_elec1PA = elec1PA->copyToDevice();

  planetArray** d_elecPtrs = toDevice({d_elec0PA, d_elec1PA});

  const PinnedVector<int> h_speciesOffsets = {0, nElec0};
  int* d_speciesOffsets = toDevice(h_speciesOffsets);

  PlanetBCState* d_state;
  cudaErrChk(cudaMalloc(&d_state, sizeof(PlanetBCState)));
  cudaErrChk(cudaMemset(d_state, 0, sizeof(PlanetBCState)));

  // Diagnostics layout as in the solver: hit |Q| per global species, then
  // absorbed |Q| per electron species.
  cudaParticleType* d_diag;
  cudaErrChk(
      cudaMalloc(&d_diag, (ns + nElecSpecies) * sizeof(cudaParticleType)));
  cudaErrChk(
      cudaMemset(d_diag, 0, (ns + nElecSpecies) * sizeof(cudaParticleType)));

  // ════════════════════════════════════════════════════════════
  //  TEST 1: Ion charge reduction
  // ════════════════════════════════════════════════════════════
  printSeparator("TEST 1 — Ion charge reduction");
  planetChargeReductionKernel<<<(nIonPlanet + BLOCK - 1) / BLOCK, BLOCK,
                                BLOCK * sizeof(cudaParticleType)>>>(
      d_ionPA, nIonPlanet, &d_state->ionChargeCycle, d_diag + 0);
  cudaErrChk(cudaDeviceSynchronize());
  {
    const PlanetBCState st = readState(d_state);
    const cudaParticleType diagIon = fromDevice(d_diag, 1)[0];
    std::cout << "  ion |Q| this cycle = " << st.ionChargeCycle
              << "  diagnostic = " << diagIon << "  (expected "
              << expectedIonCharge << ")\n";
    check(closeRel(st.ionChargeCycle, expectedIonCharge),
          "ion charge reduced into state", allPassed);
    check(closeRel(diagIon, expectedIonCharge),
          "ion charge reduced into diagnostic accumulator", allPassed);
  }

  // ════════════════════════════════════════════════════════════
  //  TEST 2: Per-electron energies + hit-charge accumulation
  // ════════════════════════════════════════════════════════════
  printSeparator("TEST 2 — Per-electron energy (weight independent)");

  const int paddedN = nextPow2(totalElecPlanet);
  cudaParticleType* d_energyBuf;
  uint32_t* d_globalIdxBuf;
  cudaErrChk(cudaMalloc(&d_energyBuf, paddedN * sizeof(cudaParticleType)));
  cudaErrChk(cudaMalloc(&d_globalIdxBuf, paddedN * sizeof(uint32_t)));

  planetEnergyKernel<<<(nElec0 + BLOCK - 1) / BLOCK, BLOCK>>>(
      d_elec0PA, nElec0, qomElec[0], c, d_energyBuf, d_globalIdxBuf,
      /*speciesOffset=*/0, d_diag + elecGlobal[0]);
  planetEnergyKernel<<<(nElec1 + BLOCK - 1) / BLOCK, BLOCK>>>(
      d_elec1PA, nElec1, qomElec[1], c, d_energyBuf, d_globalIdxBuf,
      /*speciesOffset=*/nElec0, d_diag + elecGlobal[1]);
  cudaErrChk(cudaDeviceSynchronize());

  PinnedVector<cudaParticleType> h_energy =
      fromDevice(d_energyBuf, totalElecPlanet);
  PinnedVector<uint32_t> h_gidx = fromDevice(d_globalIdxBuf, totalElecPlanet);
  {
    bool energyPass = true;
    for (int i = 0; i < totalElecPlanet; i++) {
      std::cout << "    gidx=" << h_gidx[i] << "  q=" << std::setw(6)
                << pclOf(i).get_q() << "  E=" << std::setw(12) << h_energy[i]
                << "  (expected " << expectedEnergy[i] << ")\n";
      energyPass &=
          closeRel(h_energy[i], expectedEnergy[i]) && h_gidx[i] == (uint32_t)i;
    }
    check(energyPass, "energies match (gamma-1) c^2 / |qom|", allPassed);

    const PinnedVector<cudaParticleType> diagElec =
        fromDevice(d_diag + elecGlobal[0], 2);
    std::cout << "  hit |Q|: A = " << diagElec[0]
              << " (expected 5.75), B = " << diagElec[1] << " (expected 5)\n";
    check(closeRel(diagElec[0], 5.75) && closeRel(diagElec[1], 5.0),
          "electron hit charge accumulated per species", allPassed);
  }

  // Same velocity, weights 40x apart → identical energy keys.
  {
    PinnedVector<SpeciesParticle> twin(2);
    twin[0].set(0.3, 0.1, -0.2, -0.1, 5.3, 5.0, 5.0, 0.0);
    twin[1].set(0.3, 0.1, -0.2, -4.0, 5.3, 5.0, 5.0, 1.0);
    const auto twinPA = pinnedObject<planetArray>(twin.data(), 2);
    planetArray* d_twinPA = twinPA->copyToDevice();
    cudaParticleType* d_twinE;
    uint32_t* d_twinIdx;
    cudaErrChk(cudaMalloc(&d_twinE, 2 * sizeof(cudaParticleType)));
    cudaErrChk(cudaMalloc(&d_twinIdx, 2 * sizeof(uint32_t)));
    planetEnergyKernel<<<1, BLOCK>>>(d_twinPA, 2, -1.0, c, d_twinE, d_twinIdx,
                                     0, nullptr);
    cudaErrChk(cudaDeviceSynchronize());
    const PinnedVector<cudaParticleType> twinE = fromDevice(d_twinE, 2);
    check(twinE[0] == twinE[1],
          "energy independent of macroparticle weight (|q| 0.1 vs 4)",
          allPassed);
    cudaFree(d_twinE);
    cudaFree(d_twinIdx);
    cudaFree(d_twinPA);
  }

  // Multi-block launch: hit charge summed over several blocks and warps.
  {
    const int nBig = 2000; // 8 blocks of 256, last one partial
    PinnedVector<SpeciesParticle> big(nBig);
    for (int i = 0; i < nBig; i++)
      big[i].set(0.01 * (i % 7), 0.0, 0.0, -0.5, 5.3, 5.0, 5.0,
                 (cudaParticleType)i);
    const auto bigPA = pinnedObject<planetArray>(big.data(), nBig);
    planetArray* d_bigPA = bigPA->copyToDevice();
    cudaParticleType *d_bigE, *d_bigQ;
    uint32_t* d_bigIdx;
    cudaErrChk(cudaMalloc(&d_bigE, nBig * sizeof(cudaParticleType)));
    cudaErrChk(cudaMalloc(&d_bigIdx, nBig * sizeof(uint32_t)));
    cudaErrChk(cudaMalloc(&d_bigQ, sizeof(cudaParticleType)));
    cudaErrChk(cudaMemset(d_bigQ, 0, sizeof(cudaParticleType)));
    planetEnergyKernel<<<(nBig + BLOCK - 1) / BLOCK, BLOCK>>>(
        d_bigPA, nBig, -1.0, c, d_bigE, d_bigIdx, 0, d_bigQ);
    cudaErrChk(cudaDeviceSynchronize());
    const cudaParticleType bigQ = fromDevice(d_bigQ, 1)[0];
    std::cout << "  multi-block hit |Q| = " << bigQ << "  (expected 1000)\n";
    check(bigQ == 1000.0, "multi-block hit-charge accumulation", allPassed);
    cudaFree(d_bigE);
    cudaFree(d_bigIdx);
    cudaFree(d_bigQ);
    cudaFree(d_bigPA);
  }

  // ════════════════════════════════════════════════════════════
  //  TEST 3: Bitonic sort (descending by energy)
  // ════════════════════════════════════════════════════════════
  printSeparator("TEST 3 — Bitonic sort (descending energy)");
  sortByEnergy(d_energyBuf, d_globalIdxBuf, totalElecPlanet);

  h_energy = fromDevice(d_energyBuf, totalElecPlanet);
  h_gidx = fromDevice(d_globalIdxBuf, totalElecPlanet);
  {
    bool sortPass = true;
    for (int i = 0; i < totalElecPlanet; i++) {
      std::cout << "    rank " << i << "  gidx=" << h_gidx[i]
                << "  |q|=" << std::setw(5) << fabs(pclOf(h_gidx[i]).get_q())
                << "  E=" << h_energy[i] << "\n";
      sortPass &= h_gidx[i] == expectedSortedGidx[i];
      if (i > 0)
        sortPass &= h_energy[i] <= h_energy[i - 1];
    }
    check(sortPass, "descending order A0 B0 A1 B1 A2 A3 B2", allPassed);
    check(h_gidx[0] == 0 && h_gidx[2] == 1,
          "light fast electron (A0) outranks heavy slow one (A1)", allPassed);
  }

  // ════════════════════════════════════════════════════════════
  //  TEST 4: Charge cutoff with carry-over across cycles
  // ════════════════════════════════════════════════════════════
  printSeparator("TEST 4 — Charge cutoff & carry-over");

  int* d_cutoff;
  cudaErrChk(cudaMalloc(&d_cutoff, sizeof(int)));
  PlanetElectrons electrons = {d_elecPtrs, d_speciesOffsets, d_globalIdxBuf,
                               nElecSpecies, totalElecPlanet};
  cudaParticleType* d_warpCharge;
  cudaErrChk(
      cudaMalloc(&d_warpCharge,
                 planetWarpChargeCount(electrons.total, electrons.nSpecies) *
                     sizeof(cudaParticleType)));

  struct CycleCase {
    const char* label;
    cudaParticleType ionCharge; // set before the cycle (cycle 1: from TEST 1)
    int nElec;                  // electrons that hit this cycle
    int expectedCutoff;
    cudaParticleType expectedResidual;
  };
  const CycleCase cycles[] = {
      {"cycle 1: ions 10, 7 electrons", -1.0, totalElecPlanet, 6, 0.25},
      {"cycle 2: ions 3, no electrons", 3.0, 0, 0, 3.25},
      {"cycle 3: no ions, 7 electrons", 0.0, totalElecPlanet, 2, 0.5},
      {"cycle 4: ions 2, 7 electrons", 2.0, totalElecPlanet, 2, -0.25},
  };

  cudaParticleType totalIon = 0, totalAbsorbed = 0;
  int h_cutoff = 0;
  for (const CycleCase& cc : cycles) {
    if (cc.ionCharge >= 0)
      setIonChargeCycle(d_state, cc.ionCharge);
    totalIon += readState(d_state).ionChargeCycle;

    electrons.total = cc.nElec;
    runCutoff(electrons, d_energyBuf, d_warpCharge, d_state, d_diag + ns,
              d_cutoff);
    h_cutoff = fromDevice(d_cutoff, 1)[0];
    const PlanetBCState st = readState(d_state);

    cudaParticleType absorbed = 0;
    for (int i = 0; i < h_cutoff; i++)
      absorbed += fabs(pclOf(h_gidx[i]).get_q());
    totalAbsorbed += absorbed;

    std::cout << "  " << cc.label << " → cutoff " << h_cutoff
              << ", absorbed |q| " << absorbed << ", residual "
              << st.residualCharge << "\n";
    check(h_cutoff == cc.expectedCutoff &&
              closeRel(st.residualCharge, cc.expectedResidual) &&
              st.ionChargeCycle == 0,
          std::string(cc.label) + ": cutoff, residual, ion charge cleared",
          allPassed);
  }

  {
    const PlanetBCState st = readState(d_state);
    std::cout << "  ∑|q_ion| = " << totalIon
              << "  ∑|q_absorbed electrons| = " << totalAbsorbed
              << "  residual = " << st.residualCharge << "\n";
    check(closeRel(totalIon - totalAbsorbed, st.residualCharge),
          "charge conserved: ∑ ion - ∑ absorbed electron == residual",
          allPassed);

    const PinnedVector<cudaParticleType> absorbedBySpecies =
        fromDevice(d_diag + ns, 2);
    std::cout << "  absorbed |Q|: A = " << absorbedBySpecies[0]
              << " (expected 6.25), B = " << absorbedBySpecies[1]
              << " (expected 9)\n";
    check(closeRel(absorbedBySpecies[0], 6.25) &&
              closeRel(absorbedBySpecies[1], 9.0),
          "absorbed charge per electron species", allPassed);

    // Cutoff energies: between A3/B2 (cycle 1) and B0/A1 (cycles 3 and 4).
    const cudaParticleType expectedCutSum =
        0.5 * (expectedEnergy[3] + expectedEnergy[6]) +
        2 * 0.5 * (expectedEnergy[4] + expectedEnergy[1]);
    std::cout << "  cutoff samples = " << st.cutoffSamples
              << "  energy sum = " << st.cutoffEnergySum << " (expected "
              << expectedCutSum
              << ")  electron-limited cycles = " << st.electronLimitedCycles
              << "\n";
    check(st.cutoffSamples == 3 &&
              closeRel(st.cutoffEnergySum, expectedCutSum) &&
              st.electronLimitedCycles == 1,
          "cutoff-energy and electron-limited diagnostics", allPassed);
  }

  std::cout << "  5077 electrons of two species, " << WARP_SIZE
            << " per warp:\n";
  check(manyElectronCutoffCase(),
        "many electrons: sorted by descending energy; cutoff, carried and "
        "absorbed charge match the rule applied one electron at a time",
        allPassed);

  // ════════════════════════════════════════════════════════════
  //  TEST 5: Reflect + compact (cutoff from cycle 4: 2)
  // ════════════════════════════════════════════════════════════
  printSeparator("TEST 5 — Reflect & compact surviving electrons");

  SpeciesParticle* d_outputBuf;
  cudaErrChk(
      cudaMalloc(&d_outputBuf, totalElecPlanet * sizeof(SpeciesParticle)));
  // Unused tail slots of each species segment are copied back below too.
  cudaErrChk(
      cudaMemset(d_outputBuf, 0, totalElecPlanet * sizeof(SpeciesParticle)));
  int* d_survivorCounters;
  cudaErrChk(cudaMalloc(&d_survivorCounters, nElecSpecies * sizeof(int)));
  cudaErrChk(cudaMemset(d_survivorCounters, 0, nElecSpecies * sizeof(int)));

  electrons.total = totalElecPlanet;
  const PlanetReflection specular = {originX,  originY, originZ, sphereRadius,
                                     doSphere, false,   0u};
  planetReflectCompactKernel<<<(totalElecPlanet + BLOCK - 1) / BLOCK, BLOCK>>>(
      electrons, d_cutoff, specular, d_outputBuf, d_survivorCounters);
  cudaErrChk(cudaDeviceSynchronize());

  const PinnedVector<int> h_survivorCounters =
      fromDevice(d_survivorCounters, nElecSpecies);
  const PinnedVector<SpeciesParticle> h_outputBuf =
      fromDevice(d_outputBuf, totalElecPlanet);

  std::cout << "  Survivors per species: A=" << h_survivorCounters[0]
            << "  B=" << h_survivorCounters[1] << "\n";
  check(h_survivorCounters[0] == 3 && h_survivorCounters[1] == 2,
        "survivors A1 A2 A3 and B1 B2", allPassed);

  bool reflectPass = true;
  for (int s = 0; s < nElecSpecies; s++) {
    for (int i = 0; i < h_survivorCounters[s]; i++) {
      const SpeciesParticle& p = h_outputBuf[h_speciesOffsets[s] + i];
      const cudaCommonType dx = p.get_x() - originX;
      const cudaCommonType dy = p.get_y() - originY;
      const cudaCommonType dz = p.get_z() - originZ;
      const cudaCommonType r = sqrt(dx * dx + dy * dy + dz * dz);
      printParticle(s == 0 ? "reflected A" : "reflected B", i, p);
      // Ended at r = 0.3 R moving along the outward normal (1,0,0): placed at
      // 2R - r = 1.7 R on that ray, velocity unchanged.
      const SpeciesParticle& q =
          s == 0 ? elec0Pcls[p.get_id()] : elec1Pcls[p.get_id()];
      reflectPass &= std::fabs(r - 1.7 * sphereRadius) < 1e-12 && dx > 0.0 &&
                     p.get_u() == q.get_u() && p.get_v() == 0.0 &&
                     p.get_w() == 0.0;
    }
  }
  check(reflectPass,
        "survivors at 2R - r on their ray, outward velocity unchanged",
        allPassed);

  // ════════════════════════════════════════════════════════════
  //  TEST 6: Diffuse reflection (Lambert law), 3D sphere and 2D circle
  // ════════════════════════════════════════════════════════════
  printSeparator("TEST 6 — Diffuse reflection");
  check(diffuseReflectionCase(1, originX, originY, originZ, sphereRadius),
        "3D sphere: speed kept, at max(1.05 R, 2R - r), outward, Lambert law",
        allPassed);
  check(diffuseReflectionCase(2, originX, originY, originZ, sphereRadius),
        "2D circle: speed and y kept, at max(1.05 R, 2R - r), outward, "
        "Lambert law",
        allPassed);

  // ════════════════════════════════════════════════════════════
  //  TEST 7: Specular reflection geometry; an electron at the centre
  // ════════════════════════════════════════════════════════════
  printSeparator("TEST 7 — Specular geometry, electron at the centre");
  for (int geometry = 1; geometry <= 2; geometry++) {
    const std::string geom = geometry == 1 ? "3D sphere" : "2D circle";
    check(
        specularGeometryCase(geometry, originX, originY, originZ, sphereRadius),
        geom + ": at max(1.05 R, 2R - r), v'.n = |v.n|, |v| and tangential "
               "kept, outward velocity unchanged",
        allPassed);
    for (const bool diffuse : {false, true})
      check(centreCase(diffuse, geometry, originX, originY, originZ,
                       sphereRadius),
            geom + (diffuse ? ", diffuse" : ", specular") +
                ": an electron at the centre leaves along +x, finite velocity",
            allPassed);
  }

  // ════════════════════════════════════════════════════════════
  //  Summary
  // ════════════════════════════════════════════════════════════
  printSeparator("SUMMARY");
  std::cout << (allPassed ? "  ALL TESTS PASSED\n\n"
                          : "  SOME TESTS FAILED\n\n");

  // ─── Cleanup ───────────────────────────────────────────────
  cudaFree(d_state);
  cudaFree(d_diag);
  cudaFree(d_energyBuf);
  cudaFree(d_globalIdxBuf);
  cudaFree(d_cutoff);
  cudaFree(d_warpCharge);
  cudaFree(d_outputBuf);
  cudaFree(d_survivorCounters);
  cudaFree(d_speciesOffsets);
  cudaFree(d_elecPtrs);
  // d_ionPA, d_elec0PA, d_elec1PA freed by cudaFree (raw device structs)
  cudaFree(d_ionPA);
  cudaFree(d_elec0PA);
  cudaFree(d_elec1PA);
  // Host arrayCUDA destructors free internal device arrays automatically

  return allPassed ? 0 : 1;
}

int main() {
  try {
    return runTest();
  } catch (const std::exception& error) {
    std::cerr << "planetTest failed: " << error.what() << "\n";
    return EXIT_FAILURE;
  }
}
