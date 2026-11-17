#ifndef MOVER_PUSH_CUH
#define MOVER_PUSH_CUH

/**
 * @file moverPush.cuh
 * @brief The particle push of the mover kernels: the implicit push of one
 *        step, non-relativistic and relativistic, and the push of a step
 *        with specular reflection at the reflecting walls.
 *
 * Reflecting walls.  A push that ends beyond a reflecting wall is shortened
 * until it ends on the wall, the normal velocity is reversed there, and the
 * rest of the step is pushed from the wall (pushWithWalls).  Every part is
 * the kernel's own push, with its own mid-point fields, so no force acts
 * across the wall.  Mirroring the end of the free push would be a reflection
 * only if the fields beyond the wall were the mirror image of those inside;
 * the ghost nodes hold the conductor image, the opposite, and every bounce
 * would shift the velocity along the wall and the energy, always in the same
 * sense.
 *
 * The pushes are templated on the field gather and the wall loop on the
 * push: the unit test runs them with analytic fields.
 */

#include <cmath>

#include "cudaTypeDef.cuh"

/** Square of the tolerance of the predictor-corrector iteration of a push,
 *  on the relative change of the average velocity. */
inline constexpr cudaParticleType kPushTolerance2 = 1E-12;
/** Landing tolerance of a push on a reflecting wall, in units of the
 *  smallest cell size. */
inline constexpr cudaParticleType kWallTolerance = 1e-6;
/** Most refinements of the duration of a push towards one wall. */
inline constexpr int kMaxWallRefinements = 5;
/** Most pushes of one step with reflecting walls, the free push included;
 *  beyond them the rest of the step is one free push. */
inline constexpr int kMaxWallPushes = 10;

/** Position and velocity of a particle in a mover kernel.  The velocity is
 *  in the kernel's unit: physical, or in units of c (relativistic push). */
struct MoverState {
  cudaParticleType x[3], v[3];
};

// =========================================================================
//  The implicit push of one step
// =========================================================================

/**
 * @brief Predictor-corrector convergence test.
 *
 * The relative squared change between two successive average-velocity
 * estimates, |v - v_old|^2 / |v_old|^2, is compared with kPushTolerance2
 * without the division.  As with the division, a NaN stops the iteration, and
 * so does a zero change of a zero velocity.
 */
__host__ __device__ __forceinline__ bool
pcNotConverged(cudaParticleType uavg, cudaParticleType vavg,
               cudaParticleType wavg, cudaParticleType uavg_old,
               cudaParticleType vavg_old, cudaParticleType wavg_old) {
  const cudaParticleType change2 = (uavg_old - uavg) * (uavg_old - uavg) +
                                   (vavg_old - vavg) * (vavg_old - vavg) +
                                   (wavg_old - wavg) * (wavg_old - wavg);
  return change2 >
         kPushTolerance2 *
             (uavg_old * uavg_old + vavg_old * vavg_old + wavg_old * wavg_old);
}

/**
 * @brief One implicit push of duration h from s (non-relativistic).
 *
 * The average velocity of the step, from the fields at the mid position
 * x + vavg h / 2, is iterated at most niter times; then e.x = x + vavg h and
 * e.v = 2 vavg - v.  e may be s: the start is read before the end is
 * written.  Gaussian Lorentz force q (E + v x B / c): the half-step
 * E kick is q h / (2 m) E, the rotation vector q h / (2 m c) B.
 *
 * @param niter Most iterations.  A reference: a kernel passes the value in
 *        its parameter block, which the loop then reads at every iteration.
 *        Held in a register it makes the compiler hoist the gather's
 *        loop-invariant grid loads and keep them alive across the loop, 18
 *        registers beyond the bound of the mover kernels.
 * @param fields The gather: fields(x, y, z, f) fills f = {Bx, By, Bz, Ex,
 *        Ey, Ez}.
 */
template <typename Fields>
__host__ __device__ __forceinline__ void
implicitPush(const MoverState& s, cudaParticleType h, cudaParticleType qom,
             cudaParticleType invc, const int& niter, const Fields& fields,
             MoverState& e) {
  using real = cudaParticleType;
  const real dto2 = .5 * h;
  const real qdto2m = qom * dto2;
  const real qdto2mc = qdto2m * invc;
  const real xorig = s.x[0], yorig = s.x[1], zorig = s.x[2];
  const real uorig = s.v[0], vorig = s.v[1], worig = s.v[2];
  real xavg = xorig, yavg = yorig, zavg = zorig;
  real uavg, vavg, wavg;
  real uavg_old = uorig, vavg_old = vorig, wavg_old = worig;

  int innter = 0;
  bool notConverged = true;

  // calculate the average velocity iteratively
  while (notConverged && innter < niter) {
    // E and B at the current average position
    real f[6];
    fields(xavg, yavg, zavg, f);
    const real Omx = qdto2mc * f[0];
    const real Omy = qdto2mc * f[1];
    const real Omz = qdto2mc * f[2];
    const real omsq = (Omx * Omx + Omy * Omy + Omz * Omz);
    const real denom = 1.0 / (1.0 + omsq);
    // half electric kick
    const real ut = uorig + qdto2m * f[3];
    const real vt = vorig + qdto2m * f[4];
    const real wt = worig + qdto2m * f[5];
    const real udotOm = ut * Omx + vt * Omy + wt * Omz;
    // solve the velocity equation
    uavg = (ut + (vt * Omz - wt * Omy + udotOm * Omx)) * denom;
    vavg = (vt + (wt * Omx - ut * Omz + udotOm * Omy)) * denom;
    wavg = (wt + (ut * Omy - vt * Omx + udotOm * Omz)) * denom;
    // update average position
    xavg = xorig + uavg * dto2;
    yavg = yorig + vavg * dto2;
    zavg = zorig + wavg * dto2;

    innter++;
    notConverged =
        pcNotConverged(uavg, vavg, wavg, uavg_old, vavg_old, wavg_old);
    // capture the new velocity for the next iteration
    uavg_old = uavg;
    vavg_old = vavg;
    wavg_old = wavg;
  }

  e.x[0] = xorig + uavg * h;
  e.x[1] = yorig + vavg * h;
  e.x[2] = zorig + wavg * h;
  e.v[0] = 2.0 * uavg - uorig;
  e.v[1] = 2.0 * vavg - vorig;
  e.v[2] = 2.0 * wavg - worig;
}

/**
 * @brief One relativistic implicit push of duration h from s.
 *
 * The velocities are beta = v / c (gamma = 1 / sqrt(1 - beta^2), momentum
 * u = gamma beta), and the positions advance with c beta.  The caller keeps
 * |beta| < 1.  e may be s.
 *
 * @param niter Most iterations (a reference, as in implicitPush).
 * @param fields The gather, as in implicitPush; its first call is at s.x.
 */
template <typename Fields>
__device__ __forceinline__ void
relativisticPush(const MoverState& s, cudaParticleType h, cudaParticleType qom,
                 cudaParticleType c, cudaParticleType invc, const int& niter,
                 const Fields& fields, MoverState& e) {
  using real = cudaParticleType;
  const real dto2 = .5 * h;
  const real qdto2mc = qom * dto2 * invc;
  // position steps for a velocity in units of c
  const real cdto2 = c * dto2;
  const real cdt = c * h;
  const real xorig = s.x[0], yorig = s.x[1], zorig = s.x[2];
  const real uorig = s.v[0], vorig = s.v[1], worig = s.v[2];
  real xavg = xorig, yavg = yorig, zavg = zorig;
  real uavg, vavg, wavg;
  real uavg_old = uorig, vavg_old = vorig, wavg_old = worig;

  const real vorig_sq = uorig * uorig + vorig * vorig + worig * worig;
  const real gamma0 = rsqrt(1.0 - vorig_sq);
  real invGamma1; // 1 / gamma~

  int innter = 0;
  bool notConverged = true;

  // calculate the average velocity iteratively - predictor corrector
  while (notConverged && innter < niter) {
    // E and B at the current average position
    real f[6];
    fields(xavg, yavg, zavg, f);

    // half electric kick: u~ = gamma(m) beta(m) + (q h / 2 m c) E
    const real ut = uorig * gamma0 + qdto2mc * f[3];
    const real vt = vorig * gamma0 + qdto2mc * f[4];
    const real wt = worig * gamma0 + qdto2mc * f[5];
    invGamma1 = rsqrt(1.0 + ut * ut + vt * vt + wt * wt);

    // rotation vector w / gamma~ with w = (q h / 2 m c) B (Peng et al.
    // 2015, Eq. 2.8): the relativistic gyrofrequency is Omega / gamma
    const real qdto2mcg = qdto2mc * invGamma1;
    const real Omx = qdto2mcg * f[0];
    const real Omy = qdto2mcg * f[1];
    const real Omz = qdto2mcg * f[2];
    const real omsq = (Omx * Omx + Omy * Omy + Omz * Omz);
    // beta_bar = alpha(w / gamma~) u~ / gamma~
    const real denom = invGamma1 / (1.0 + omsq);

    const real udotOm = ut * Omx + vt * Omy + wt * Omz;
    // solve the velocity equation
    uavg = (ut + (vt * Omz - wt * Omy + udotOm * Omx)) * denom;
    vavg = (vt + (wt * Omx - ut * Omz + udotOm * Omy)) * denom;
    wavg = (wt + (ut * Omy - vt * Omx + udotOm * Omz)) * denom;
    // update average position
    xavg = xorig + uavg * cdto2;
    yavg = yorig + vavg * cdto2;
    zavg = zorig + wavg * cdto2;

    notConverged =
        pcNotConverged(uavg, vavg, wavg, uavg_old, vavg_old, wavg_old);
    // capture the new velocity for the next iteration
    uavg_old = uavg;
    vavg_old = vavg;
    wavg_old = wavg;

    innter++;
  }

  // relativistic velocity update
  const real ut = uorig * gamma0;
  const real vt = vorig * gamma0;
  const real wt = worig * gamma0;

  const real velt_sq = ut * ut + vt * vt + wt * wt;
  const real velavg_sq = uavg * uavg + vavg * vavg + wavg * wavg;
  const real velt_velavg = ut * uavg + vt * vavg + wt * wavg;

  const real cfa = 1.0 - velavg_sq;
  const real cfb = -2.0 * (-velt_velavg + gamma0 * velavg_sq);
  const real cfc =
      -1.0 - gamma0 * gamma0 * velavg_sq + 2.0 * gamma0 * velt_velavg - velt_sq;

  const real delta_rel = cfb * cfb - 4.0 * cfa * cfc;

  e.x[0] = xorig + uavg * cdt;
  e.x[1] = yorig + vavg * cdt;
  e.x[2] = zorig + wavg * cdt;
  if (delta_rel < 0.0) {
    // Unreachable in exact arithmetic (cfa > 0 because |beta_bar| < 1, and
    // cfc < 0); only round-off with |beta_bar| -> 1 lands here.  Estimate
    // the momentum u(m+1) = 2 gamma~ beta_bar - u(m) and store it as a
    // velocity, beta = u / sqrt(1 + u^2), which keeps |beta| < 1.
    const real twoGamma1 = 2.0 / invGamma1;
    const real un = twoGamma1 * uavg - ut;
    const real vn = twoGamma1 * vavg - vt;
    const real wn = twoGamma1 * wavg - wt;
    const real invGammaNew = rsqrt(1.0 + un * un + vn * vn + wn * wn);
    e.v[0] = un * invGammaNew;
    e.v[1] = vn * invGammaNew;
    e.v[2] = wn * invGammaNew;
  } else {
    // 1 / gamma(m+1), gamma(m+1) = (-cfb + sqrt(delta_rel)) / (2 cfa) the
    // positive root of cfa g^2 + cfb g + cfc = 0.  Then beta(m+1) =
    // u(m+1) / gamma(m+1) with u(m+1) = (gamma(m) + gamma(m+1)) beta_bar -
    // u(m).
    const real invGammaNew = (2.0 * cfa) / (sqrt(delta_rel) - cfb);
    const real coef = 1.0 + gamma0 * invGammaNew;
    e.v[0] = coef * uavg - ut * invGammaNew;
    e.v[1] = coef * vavg - vt * invGammaNew;
    e.v[2] = coef * wavg - wt * invGammaNew;
  }
}

// =========================================================================
//  Reflecting walls
// =========================================================================

/** The end of a push lies beyond a reflecting wall (lo, hi: the coordinates
 *  of the low and the high wall per axis, -/+ infinity where the face does
 *  not reflect). */
__host__ __device__ __forceinline__ bool
beyondWalls(const MoverState& e, const cudaParticleType lo[3],
            const cudaParticleType hi[3]) {
  return e.x[0] < lo[0] || e.x[0] > hi[0] || e.x[1] < lo[1] || e.x[1] > hi[1] ||
         e.x[2] < lo[2] || e.x[2] > hi[2];
}

/**
 * @brief The duration of the push that reaches a wall.
 *
 * A particle starts at the signed distance F0 <= 0 from the wall (positive
 * beyond it) with the outward speed g0, and its push of duration h ends at
 * F1 > 0.  The distance of the end of a push of duration t is F0 + g0 t +
 * a t^2 / 2 + O(t^3): the smallest positive root of the quadratic through
 * the three data.  The second form is the particle that starts inwards (or
 * on the wall, after a bounce) and is turned back to the wall.
 */
__host__ __device__ __forceinline__ cudaParticleType
wallTime(cudaParticleType F0, cudaParticleType g0, cudaParticleType h,
         cudaParticleType F1) {
  const cudaParticleType a = 2.0 * (F1 - F0 - g0 * h) / (h * h);
  const cudaParticleType D = g0 * g0 - 2.0 * a * F0;
  const cudaParticleType root = sqrt(D > 0.0 ? D : 0.0);
  return g0 > 0.0 ? -2.0 * F0 / (g0 + root) : (root - g0) / a;
}

/**
 * @brief The push of duration T from s with specular reflection at the
 *        reflecting walls.
 *
 * The step is a sequence of pushes, the first over the whole of it.  A push
 * that ends beyond a wall is a landing attempt on the wall the particle
 * reaches first: its duration is refined (wallTime, then the secant on the
 * distance of the end to the wall, kept inside a bracket) until the end is
 * within tol of the wall.  There the normal velocity is reversed, on every
 * wall the end is beyond (an edge or a corner), and the rest of the step is
 * pushed from the wall, with further bounces the same way.  After
 * kMaxWallPushes pushes the rest is one free push, which may end beyond a
 * wall (the host mirrors it).
 *
 * @param s In: the start of the step.  Out: its end.
 * @param lo, hi The low and the high wall per axis (see beyondWalls).
 * @param tol Landing tolerance, a distance.
 * @param speed Physical speed per unit of the state's velocity.
 * @param push push(from, h, to): the kernel's push; false stops the step
 *        (the particle was deleted).
 * @return False when a push stopped the step.
 */
template <typename Push>
__host__ __device__ __forceinline__ bool
pushWithWalls(MoverState& s, cudaParticleType T, const cudaParticleType lo[3],
              const cudaParticleType hi[3], cudaParticleType tol,
              cudaParticleType speed, const Push& push) {
  using real = cudaParticleType;
  MoverState e;        // end of the trial push from s
  real h = T;          // its duration
  int face = -1;       // wall of the landing attempt (2 axis + side)
  int refinements = 0; // of this attempt
  // its bracket (a duration that ends inside the wall, one beyond, with the
  // distances F of their ends) and its previous trial
  real hLo = 0, FLo = 0, hHi = 0, FHi = 0, hPrev = 0, FPrev = 0;

  for (int pushes = 0;; pushes++) {
    // pushes exhausted: the rest of the step as one free push
    const bool rest = pushes == kMaxWallPushes;
    if (rest)
      h = T;
    if (!push(s, h, e))
      return false;
    if (rest) {
      s = e;
      return true;
    }

    // The wall of this trial, with the distance F of the end beyond it
    // (F0 at the start, g0 the outward speed there): of the walls the end
    // is beyond, the one reached first along the straight line (the
    // smallest -F0 / (F - F0), compared without the divisions); else the
    // wall of the attempt
    int f = -1;
    real F = 0, F0 = 0, g0 = 0;
    real Fatt = 0;
#pragma unroll
    for (int a = 0; a < 3; a++) {
#pragma unroll
      for (int side = 0; side < 2; side++) {
        const real wall = side ? hi[a] : lo[a];
        const real sign = side ? 1.0 : -1.0;
        const real Fe = sign * (e.x[a] - wall);
        if (Fe > 0) {
          const real Fs = sign * (s.x[a] - wall);
          if (f < 0 || -Fs * (F - F0) < -F0 * (Fe - Fs)) {
            f = 2 * a + side;
            F = Fe;
            F0 = Fs;
            g0 = sign * speed * s.v[a];
          }
        } else if (2 * a + side == face) {
          Fatt = Fe;
        }
      }
    }
    if (f < 0 && face >= 0) {
      f = face;
      F = Fatt;
    }

    // no wall: the push was over the rest of the step, which is complete
    if (f < 0) {
      s = e;
      return true;
    }

    if (fabs(F) <= tol || (f == face && refinements == kMaxWallRefinements)) {
      // Landed.  On every wall the end is beyond: the position on the wall,
      // the normal velocity reversed.  An end inside all of them is at the
      // wall of the attempt: the velocity normal to that one.  (The axes
      // are compared, not indexed: an index would put the state in memory.)
      const int axis = f >> 1;
#pragma unroll
      for (int a = 0; a < 3; a++) {
        const bool beyond = e.x[a] < lo[a] || e.x[a] > hi[a];
        if (beyond)
          e.x[a] = e.x[a] < lo[a] ? lo[a] : hi[a];
        if (beyond || a == axis)
          e.v[a] = -e.v[a];
      }
      s = e;
      T -= h;
      h = T;
      face = -1;
    } else {
      // Refine the duration.  A new attempt starts from the start of the
      // segment and this trial, with the estimate of wallTime.  A later
      // trial replaces its end of the bracket, and the next duration is the
      // secant through the last two trials.
      real hn;
      if (f != face) {
        face = f;
        refinements = 0;
        hLo = 0;
        FLo = F0;
        hHi = h;
        FHi = F;
        hn = wallTime(F0, g0, h, F);
      } else {
        if (F > 0) {
          hHi = h;
          FHi = F;
        } else {
          hLo = h;
          FLo = F;
        }
        hn = h - F * (h - hPrev) / (F - FPrev);
      }
      hPrev = h;
      FPrev = F;
      // strictly inside the bracket: else the secant through its ends, else
      // (a NaN, a flat secant) its middle
      if (!(hn > hLo && hn < hHi))
        hn = hLo - FLo * (hHi - hLo) / (FHi - FLo);
      if (!(hn > hLo && hn < hHi))
        hn = 0.5 * (hLo + hHi);
      h = hn;
      refinements++;
    }

    // a landing may have used up the step
    if (!(T > 0))
      return true;
  }
}

#endif // MOVER_PUSH_CUH
