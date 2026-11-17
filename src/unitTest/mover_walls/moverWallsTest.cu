/**
 * @file moverWallsTest.cu
 * @brief Unit test of the particle push with reflecting walls
 *        (moverPush.cuh): the pushes of the mover kernels and pushWithWalls,
 *        with uniform fields, on the host and on the device.
 *
 * In uniform fields a specular wall conserves, for every particle, what the
 * implicit push conserves exactly:
 *   - with B tangential to the wall (along z, wall normal y),
 *     P = v_x - (q/m) (B/c) y;
 *   - with E normal to the wall, H = v^2 / 2 - (q/m) E y.
 * pushWithWalls changes them only by the landing tolerance; mirroring the end
 * of a free push would move them by about 0.3 v_th and 0.1 v_th^2 per bounce
 * here.
 *
 * Cases (--case <name>):
 *   invariants   P and H of a thermal population between two walls
 *   box          six walls, oblique B: inside, finite, energy
 *   corner       free flight into an edge and a corner: the exact mirror
 *   turned-back  a particle on the wall turned back to it: the skipping orbit
 *   narrow       walls closer than a step: the push budget ends the step
 *   wall-time    wallTime against the roots of exact quadratics
 *   no-wall      pushWithWalls without walls is the free push, bitwise
 *   relativistic the relativistic push with walls: inside, gamma kept
 *   stopped      a push that deletes the particle stops the step
 */

#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include "cudaTypeDef.cuh"
#include "moverPush.cuh"

using real = cudaParticleType;

namespace {

constexpr real kInf = std::numeric_limits<real>::infinity();

/** Predictor-corrector iterations of a push. */
constexpr int kPushIterations = 3;

/** Threads per block of the device runs. */
constexpr int kBlockSize = 256;

// The uniform decks: cell 0.078, dt 0.15, electrons q/m = -64 with thermal
// speed 0.045, |B| = 0.0276 (Omega dt = 0.265, v_th dt = 0.087 cells), walls
// 4 cells apart
constexpr real kCell = 0.078, kDt = 0.15, kQom = -64.0, kVth = 0.045,
               kB = 0.0276;
constexpr int kCells = 4;

/** Uniform fields as the gather of a push. */
struct UniformFields {
  real B[3], E[3];
  __host__ __device__ void operator()(real, real, real, real f[6]) const {
    for (int i = 0; i < 3; i++) {
      f[i] = B[i];
      f[3 + i] = E[i];
    }
  }
};

/** One run: the fields, the walls and the speed of light. */
struct Setup {
  UniformFields fields;
  real lo[3], hi[3];
  real c;
};

/** One step from s with the walls, as a mover kernel makes it.  pushes
 *  counts the pushes when not null; push number stopAt reports a deleted
 *  particle, as a kernel's push can.  False when the step was stopped. */
template <bool kRelativistic>
__host__ __device__ bool step(MoverState& s, const Setup& su,
                              int* pushes = nullptr, int stopAt = 0) {
  const int niter = kPushIterations; // the pushes take it by reference
  const real invc = 1.0 / su.c;
  return pushWithWalls(
      s, kDt, su.lo, su.hi, kWallTolerance * kCell, kRelativistic ? su.c : 1.0,
      [&](const MoverState& from, real h, MoverState& to) {
        if constexpr (kRelativistic)
          relativisticPush(from, h, kQom, su.c, invc, niter, su.fields, to);
        else
          implicitPush(from, h, kQom, invc, niter, su.fields, to);
        if (pushes)
          ++*pushes;
        return !pushes || *pushes != stopAt;
      });
}

template <bool kRelativistic>
__global__ void k_steps(MoverState* p, int n, int steps, Setup su) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n)
    return;
  MoverState s = p[i];
  for (int k = 0; k < steps; k++)
    step<kRelativistic>(s, su);
  p[i] = s;
}

/** The particles after `steps` steps on the device. */
template <bool kRelativistic>
std::vector<MoverState> onDevice(const std::vector<MoverState>& p, int steps,
                                 const Setup& su) {
  const size_t bytes = p.size() * sizeof(MoverState);
  MoverState* d = nullptr;
  cudaErrChk(cudaMalloc(&d, bytes));
  cudaErrChk(cudaMemcpy(d, p.data(), bytes, cudaMemcpyHostToDevice));
  k_steps<kRelativistic>
      <<<getGridSize((int)p.size(), kBlockSize), kBlockSize>>>(d, (int)p.size(),
                                                               steps, su);
  cudaErrChk(cudaGetLastError());
  std::vector<MoverState> out(p.size());
  cudaErrChk(cudaMemcpy(out.data(), d, bytes, cudaMemcpyDeviceToHost));
  cudaErrChk(cudaFree(d));
  return out;
}

/** The particles after `steps` steps on the host. */
std::vector<MoverState> onHost(std::vector<MoverState> p, int steps,
                               const Setup& su) {
  for (MoverState& s : p)
    for (int k = 0; k < steps; k++)
      step<false>(s, su);
  return p;
}

Setup setup(const real B[3], const real E[3], const bool wall[3],
            real c = 1.0) {
  Setup su;
  su.c = c;
  for (int a = 0; a < 3; a++) {
    su.fields.B[a] = B[a];
    su.fields.E[a] = E[a];
    su.lo[a] = wall[a] ? 0.0 : -kInf;
    su.hi[a] = wall[a] ? kCells * kCell : kInf;
  }
  return su;
}

/** n thermal particles, uniform in the walled directions. */
std::vector<MoverState> thermal(int n, const Setup& su, unsigned seed,
                                real vth = kVth) {
  std::mt19937_64 gen(seed);
  std::normal_distribution<real> maxwell(0.0, vth);
  std::uniform_real_distribution<real> unit(0.0, 1.0);
  std::vector<MoverState> p(n);
  for (MoverState& s : p)
    for (int a = 0; a < 3; a++) {
      s.x[a] = std::isfinite(su.hi[a]) ? unit(gen) * su.hi[a] : 0.0;
      s.v[a] = maxwell(gen);
    }
  return p;
}

/** The particles beyond a wall or with a velocity that is not finite. */
int outside(const std::vector<MoverState>& p, const Setup& su) {
  int n = 0;
  for (const MoverState& s : p)
    for (int a = 0; a < 3; a++)
      if (!(s.x[a] >= su.lo[a] && s.x[a] <= su.hi[a]) ||
          !std::isfinite(s.v[a])) {
        n++;
        break;
      }
  return n;
}

int check(bool ok, const char* what) {
  if (!ok)
    std::printf("  FAILED: %s\n", what);
  return ok ? 0 : 1;
}

// -------------------------------------------------------------------------

int testInvariants() {
  const real B[3] = {0, 0, kB}, E[3] = {0, 2e-4, 0};
  const bool wall[3] = {false, true, false};
  const Setup su = setup(B, E, wall);
  const int steps = 400;
  const std::vector<MoverState> start = thermal(20000, su, 1);
  const auto P = [&](const MoverState& s) {
    return s.v[0] - kQom * B[2] * s.x[1];
  };
  const auto H = [&](const MoverState& s) {
    return 0.5 * (s.v[0] * s.v[0] + s.v[1] * s.v[1] + s.v[2] * s.v[2]) -
           kQom * E[1] * s.x[1];
  };
  int bad = 0;
  for (const bool device : {false, true}) {
    const std::vector<MoverState> end =
        device ? onDevice<false>(start, steps, su) : onHost(start, steps, su);
    real dP = 0, dH = 0;
    for (size_t i = 0; i < start.size(); i++) {
      dP = std::max(dP, std::fabs(P(end[i]) - P(start[i])) / kVth);
      dH = std::max(dH, std::fabs(H(end[i]) - H(start[i])) / (kVth * kVth));
    }
    const int out = outside(end, su);
    std::printf("invariants (%s): max |dP| = %.2e v_th, max |dH| = %.2e "
                "v_th^2, outside %d\n",
                device ? "device" : "host", dP, dH, out);
    // at most `steps` bounces, each within the landing tolerance
    bad += check(dP < 1e-3 && dH < 1e-4, "an invariant moved") +
           check(out == 0, "a particle beyond a wall");
  }
  return bad;
}

int testBox() {
  const real s3 = kB / std::sqrt(1.0 + 0.49 + 0.16);
  const real B[3] = {s3, 0.7 * s3, -0.4 * s3}, E[3] = {0, 0, 0};
  const bool wall[3] = {true, true, true};
  const Setup su = setup(B, E, wall);
  const int steps = 400;
  const std::vector<MoverState> start = thermal(20000, su, 2);
  const auto v2 = [](const MoverState& s) {
    return s.v[0] * s.v[0] + s.v[1] * s.v[1] + s.v[2] * s.v[2];
  };
  int bad = 0;
  for (const bool device : {false, true}) {
    const std::vector<MoverState> end =
        device ? onDevice<false>(start, steps, su) : onHost(start, steps, su);
    real dE = 0;
    for (size_t i = 0; i < start.size(); i++)
      dE = std::max(dE, std::fabs(v2(end[i]) / v2(start[i]) - 1.0));
    const int out = outside(end, su);
    std::printf("box (%s): max relative change of v^2 = %.2e, outside %d\n",
                device ? "device" : "host", dE, out);
    bad += check(dE < 1e-11, "the energy changed") +
           check(out == 0, "a particle beyond a wall");
  }
  return bad;
}

int testCorner() {
  // no field: straight lines, and the mirror in every wall crossed is exact
  const real B[3] = {0, 0, 0}, E[3] = {0, 0, 0};
  const bool wall[3] = {true, true, true};
  const Setup su = setup(B, E, wall);
  int bad = 0;
  const real v = 0.1 * kCell / kDt;
  for (const int walls : {2, 3}) {
    // 0.03, 0.05, 0.08 cells from the walls, 0.1 cells of flight
    MoverState s = {{0.03 * kCell, 0.05 * kCell, 0.08 * kCell}, {-v, -v, -v}};
    if (walls == 2) {
      s.x[2] = 2 * kCell;
      s.v[2] = 0.3 * v;
    }
    const MoverState s0 = s;
    step<false>(s, su);
    real err = 0;
    for (int a = 0; a < 3; a++) {
      const bool hit = a < walls;
      const real x = s0.x[a] + s0.v[a] * kDt;
      err = std::max(err, std::fabs(s.x[a] - (hit ? -x : x)) / kCell);
      err = std::max(err, std::fabs(s.v[a] - (hit ? -s0.v[a] : s0.v[a])) / v);
    }
    std::printf("corner (%d walls): max error %.2e\n", walls, err);
    bad += check(err < 1e-5, "not the mirror in the walls");
  }
  return bad;
}

int testTurnedBack() {
  // A particle on the wall, leaving it with the normal speed vn and the
  // tangential speed vt, is turned back by a tangential B: its push of
  // duration h ends on the wall again for Omega h / 2 = vn / vt, with the
  // normal velocity reversed and the tangential one unchanged.  Reflected,
  // it leaves as it started: it skips along the wall with the period
  // h* = 2 vn / (Omega vt).  With h* = 0.6 dt the step is one skip and a
  // free push of 0.4 dt (the second form of wallTime, a start on the wall).
  // One bounce takes at most kMaxWallRefinements + 2 pushes.
  const real omega = 1.0 / kDt, v0 = kVth;
  const real B[3] = {0, 0, omega / std::fabs(kQom)}, E[3] = {0, 0, 0};
  const bool wall[3] = {false, true, false};
  const Setup su = setup(B, E, wall);
  MoverState s = {{0, 0, 0}, {-v0, 0.3 * v0, 0.2 * v0}};
  MoverState ref;
  implicitPush(s, 0.4 * kDt, kQom, 1.0 / su.c, kPushIterations, su.fields, ref);
  int pushes = 0;
  step<false>(s, su, &pushes);
  real err = std::fabs(s.x[1] - ref.x[1]) / kCell;
  for (int a = 0; a < 3; a++)
    err = std::max(err, std::fabs(s.v[a] - ref.v[a]) / v0);
  std::printf("turned-back: max error %.2e against the free push of the "
              "remainder, %d pushes\n",
              err, pushes);
  return check(err < 1e-4, "not one skip and the remainder") +
         check(pushes <= kMaxWallRefinements + 2, "more pushes than a bounce");
}

int testNarrow() {
  // two walls 0.01 cells apart, 0.09 cells of flight per step: about 9
  // bounces of two pushes each, more than the push budget, which ends the
  // step with a free push
  const real B[3] = {0, 0, 0}, E[3] = {0, 0, 0};
  const bool wall[3] = {false, true, false};
  Setup su = setup(B, E, wall);
  su.hi[1] = 0.01 * kCell;
  MoverState s = {{0, 0.004 * kCell, 0}, {0, 0.09 * kCell / kDt, 0}};
  int pushes = 0;
  step<false>(s, su, &pushes);
  std::printf("narrow: %d pushes, y = %.3f cells\n", pushes, s.x[1] / kCell);
  return check(pushes == kMaxWallPushes + 1, "not the push budget") +
         check(std::isfinite(s.x[1]) && std::isfinite(s.v[1]), "not finite");
}

int testWallTime() {
  // F(t) = F0 + g0 t + a t^2 / 2 with the root at t = 1 or t = 0.4
  real worst = 0;
  for (const real root : {1.0, 0.4})
    for (const real g0 : {2.0, 0.5, 0.0, -0.7})
      for (const real F0 : {-1.0, -0.2, 0.0}) {
        if (F0 == 0.0 && g0 >= 0.0)
          continue; // on the wall, not leaving it
        const real a = -2.0 * (F0 + g0 * root) / (root * root);
        const real h = 1.5;
        const real F1 = F0 + g0 * h + 0.5 * a * h * h;
        if (!(F1 > 0))
          continue;
        worst = std::max(worst, std::fabs(wallTime(F0, g0, h, F1) - root));
      }
  // a = 0: the straight line
  worst = std::max(worst, std::fabs(wallTime(-1.0, 2.0, 1.0, 1.0) - 0.5));
  std::printf("wall-time: max error %.2e\n", worst);
  return check(worst < 1e-12, "not the root of the quadratic");
}

int testNoWall() {
  const real B[3] = {0.3 * kB, -0.5 * kB, kB}, E[3] = {1e-4, -2e-4, 3e-4};
  const bool wall[3] = {false, false, false};
  const Setup su = setup(B, E, wall);
  const std::vector<MoverState> start = thermal(1000, su, 4);
  int differ = 0;
  for (MoverState s : start) {
    MoverState free;
    implicitPush(s, kDt, kQom, 1.0 / su.c, kPushIterations, su.fields, free);
    step<false>(s, su);
    differ += std::memcmp(&s, &free, sizeof(MoverState)) != 0;
  }
  std::printf("no-wall: %d of %zu differ from the free push\n", differ,
              start.size());
  return check(differ == 0, "not the free push");
}

int testRelativistic() {
  // beta ~ 0.2, and c = 0.5 so that the units of c matter: the relativistic
  // push between two walls keeps gamma (E = 0)
  const real B[3] = {0, 0, kB}, E[3] = {0, 0, 0};
  const bool wall[3] = {false, true, false};
  const Setup su = setup(B, E, wall, 0.5);
  const std::vector<MoverState> start = thermal(20000, su, 5, 0.1);
  const std::vector<MoverState> end = onDevice<true>(start, 400, su);
  const auto gamma = [](const MoverState& s) {
    return 1.0 / std::sqrt(1.0 - (s.v[0] * s.v[0] + s.v[1] * s.v[1] +
                                  s.v[2] * s.v[2]));
  };
  real dg = 0;
  for (size_t i = 0; i < start.size(); i++)
    dg = std::max(dg, std::fabs(gamma(end[i]) / gamma(start[i]) - 1.0));
  const int out = outside(end, su);
  std::printf("relativistic (device): max relative change of gamma = %.2e, "
              "outside %d\n",
              dg, out);
  return check(dg < 1e-10, "gamma changed") +
         check(out == 0, "a particle beyond a wall");
}

int testStopped() {
  // a push that reports a deleted particle ends the step at once: here the
  // second push, the first landing attempt
  const real B[3] = {0, 0, 0}, E[3] = {0, 0, 0};
  const bool wall[3] = {false, true, false};
  const Setup su = setup(B, E, wall);
  MoverState s = {{0, 0.01 * kCell, 0}, {0, -kVth, 0}};
  int pushes = 0;
  const bool done = step<false>(s, su, &pushes, 2);
  std::printf("stopped: %d pushes, step %s\n", pushes,
              done ? "completed" : "stopped");
  return check(!done && pushes == 2, "the step went on");
}

} // namespace

int main(int argc, char** argv) {
  const std::pair<const char*, int (*)()> cases[] = {
      {"invariants", testInvariants}, {"box", testBox},
      {"corner", testCorner},         {"turned-back", testTurnedBack},
      {"narrow", testNarrow},         {"wall-time", testWallTime},
      {"no-wall", testNoWall},        {"relativistic", testRelativistic},
      {"stopped", testStopped}};
  std::string only;
  for (int i = 1; i + 1 < argc; i++)
    if (std::string(argv[i]) == "--case")
      only = argv[i + 1];
  int bad = 0, ran = 0;
  for (const auto& c : cases)
    if (only.empty() || only == c.first) {
      bad += c.second();
      ran++;
    }
  if (ran == 0) {
    std::printf("unknown case %s\n", only.c_str());
    return 2;
  }
  std::printf("%s\n", bad == 0 ? "PASSED" : "FAILED");
  return bad == 0 ? 0 : 1;
}
