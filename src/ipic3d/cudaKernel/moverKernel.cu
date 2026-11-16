#include <assert.h>
#include <cstdint>
#include <math.h>

#include "cudaTypeDef.cuh"
#include "gridCUDA.cuh"
#include "hashedSum.cuh"
#include "moverKernel.cuh"
#include "particleArrayCUDA.cuh"
#include "particleExchange.cuh"

using commonType = cudaParticleType;

constexpr cudaTypeDouble PC_err_2 = 1E-12; // square of error tolerance

__device__ constexpr bool cap_velocity() { return false; }

/** Safety cap of the particle speed, |v| < kVelocityCapOverC c. */
constexpr commonType kVelocityCapOverC = 0.9999;

/** Most subcycles of one relativistic push.  The implicit rotation is stable
 *  at any angle; the subcycles keep it accurate (at most pi/4 of gyration
 *  each), and the shipped dipole decks need about 50 at the planet surface.
 *  The cap bounds the cost of a corrupted, huge or infinite, |B|. */
constexpr int kMaxSubcycles = 1024;

// ======= Inline mover helpers =======

/**
 * @brief Mark a particle for deletion and register it in the delete hashed sum.
 *
 * The particle state itself is left untouched; only the departure metadata is
 * updated so later compaction and communication stages can remove it safely.
 *
 * @param moverParam Device-side mover parameter bundle.
 * @param pidx Particle index in the species SoA buffer.
 */
__device__ __forceinline__ void markForDeletion(moverParameter* moverParam,
                                                uint32_t pidx) {
  moverParam->departureArray->getArray()[pidx].dest =
      departureArrayElementType::DELETE;
  moverParam->departureArray->getArray()[pidx].hashedId =
      moverParam
          ->hashedSumArray[departureArrayElementType::DELETE_HASHEDSUM_INDEX]
          .add(pidx);
}

/**
 * @brief Clamp finite superluminal velocities and delete non-finite particles.
 *
 * Operates on register-resident velocity components. Finite velocities with
 * |v| at or above the safety cap are rescaled back to it; NaN and Inf values
 * are marked for deletion because they would poison later gamma and moment
 * calculations.
 *
 * @param ufinal In-out x-velocity component.
 * @param vfinal In-out y-velocity component.
 * @param wfinal In-out z-velocity component.
 * @param v2max Square of the cap, in the units of the components
 *        ((kVelocityCapOverC c)^2 for physical velocities).
 * @param moverParam Device-side mover parameter bundle.
 * @param pidx Particle index in the species SoA buffer.
 * @return True if the particle was deleted and the caller should return.
 */
__device__ __forceinline__ bool
capVelocityOrDelete(commonType& ufinal, commonType& vfinal, commonType& wfinal,
                    commonType v2max, moverParameter* moverParam,
                    uint32_t pidx) {
  const commonType v2 = ufinal * ufinal + vfinal * vfinal + wfinal * wfinal;
  if (!(v2 < v2max)) {
    if (isfinite(v2)) {
      const commonType scale = sqrt(v2max / v2);
      ufinal *= scale;
      vfinal *= scale;
      wfinal *= scale;
    } else {
      markForDeletion(moverParam, pidx);
      return true;
    }
  }
  return false;
}

/**
 * @brief Interpolate particle fields from the packed node-centered field
 * buffer.
 *
 * Updates the owning cell indices and trilinear weights, then accumulates the
 * first @p nFields components into @p sampled_field.
 *
 * @param x Particle x position.
 * @param y Particle y position.
 * @param z Particle z position.
 * @param grid Device-side grid descriptor.
 * @param fieldForPcls Packed field buffer consumed by the mover.
 * @param weights Output trilinear weights for the enclosing cell.
 * @param cx Output cell index in x.
 * @param cy Output cell index in y.
 * @param cz Output cell index in z.
 * @param sampled_field Output sampled field values.
 */
template <int nFields>
__device__ __forceinline__ void sampleFieldsAtPosition(
    commonType x, commonType y, commonType z, grid3DCUDA* grid,
    cudaTypeArray1<cudaFieldType> fieldForPcls, commonType weights[8], int& cx,
    int& cy, int& cz, commonType sampled_field[6]) {
  grid->get_safe_cell_and_weights(x, y, z, cx, cy, cz, weights);
  const int previousIndex = (cx * (grid->nyn - 1) + cy) * grid->nzn + cz;
  assert(previousIndex >= 0 &&
         previousIndex < (grid->nxn - 1) * (grid->nyn - 1) * grid->nzn);
  for (int i = 0; i < nFields; i++)
    sampled_field[i] = 0;
  for (int c = 0; c < 8; c++)
    for (int i = 0; i < nFields; i++)
      sampled_field[i] +=
          weights[c] * fieldForPcls[previousIndex * 24 + c * 6 + i];
}

/**
 * @brief Predictor-corrector convergence test.
 *
 * The relative squared change between two successive average-velocity
 * estimates, |v - v_old|^2 / |v_old|^2, is compared with PC_err_2 without the
 * division.  As with the division, a NaN stops the iteration, and so does a
 * zero change of a zero velocity.
 *
 * @param uavg Current averaged x velocity.
 * @param vavg Current averaged y velocity.
 * @param wavg Current averaged z velocity.
 * @param uavg_old Previous averaged x velocity.
 * @param vavg_old Previous averaged y velocity.
 * @param wavg_old Previous averaged z velocity.
 * @return True while the relative squared change exceeds PC_err_2.
 */
__device__ __forceinline__ bool
pcNotConverged(commonType uavg, commonType vavg, commonType wavg,
               commonType uavg_old, commonType vavg_old, commonType wavg_old) {
  const commonType change2 = (uavg_old - uavg) * (uavg_old - uavg) +
                             (vavg_old - vavg) * (vavg_old - vavg) +
                             (wavg_old - wavg) * (wavg_old - wavg);
  return change2 > PC_err_2 * (uavg_old * uavg_old + vavg_old * vavg_old +
                               wavg_old * wavg_old);
}

// ======= End inline mover helpers =======

/**
 * @brief Classify a moved particle and update its departure metadata.
 *
 * @param xpcl Particle x position after the mover.
 * @param ypcl Particle y position after the mover.
 * @param zpcl Particle z position after the mover.
 * @param xstart Particle x position before the mover.
 * @param ystart Particle y position before the mover.
 * @param zstart Particle z position before the mover.
 * @param pclsArray Device-side particle SoA container.
 * @param pidx Particle index in the species SoA buffer.
 * @param moverParam Device-side mover parameter bundle.
 * @param departureArray Device-side departure metadata array.
 * @param grid Device-side grid descriptor.
 * @param hashedSumArray Device-side hashed-sum buckets for departure
 * compaction.
 */
__device__ void prepareDepartureArray(commonType xpcl, commonType ypcl,
                                      commonType zpcl, commonType xstart,
                                      commonType ystart, commonType zstart,
                                      particleArrayCUDA* pclsArray,
                                      uint32_t pidx, moverParameter* moverParam,
                                      departureArrayType* departureArray,
                                      grid3DCUDA* grid,
                                      hashedSum* hashedSumArray);

/**
 * @brief Advance one particle over the full solver time step with the standard
 * mover.
 *
 * The kernel performs the predictor-corrector push, clamps or deletes invalid
 * velocities, writes the final SoA state, and classifies the particle for
 * staying, exchange, deletion, or planet handling.
 *
 * @param moverParam Device-side mover parameter bundle for one species.
 * @param fieldForPcls Packed mover field buffer.
 * @param grid Device-side grid descriptor.
 */
__global__ void MOVER_LAUNCH_BOUNDS
moverKernel(moverParameter* moverParam,
            cudaTypeArray1<cudaFieldType> fieldForPcls, grid3DCUDA* grid) {
  uint pidx = blockIdx.x * blockDim.x + threadIdx.x;

  auto pclsArray = moverParam->pclsArray;
  if (pidx >= pclsArray->getNOP())
    return;

  // Gaussian Lorentz force q (E + v x B / c): the half-step E kick is
  // q dt / (2 m) E, the rotation vector q dt / (2 m c) B
  const commonType dto2 = .5 * moverParam->dt;
  const commonType qdto2m = moverParam->qom * dto2;
  const commonType qdto2mc = qdto2m * moverParam->invc;

  // Early exit for particles already deleted (e.g. during merging)
  if (moverParam->departureArray->getArray()[pidx].dest != 0) {
    // not moved: the position is both start and end
    const commonType x = pclsArray->getX()[pidx];
    const commonType y = pclsArray->getY()[pidx];
    const commonType z = pclsArray->getZ()[pidx];
    prepareDepartureArray(x, y, z, x, y, z, pclsArray, pidx, moverParam,
                          moverParam->departureArray, grid,
                          moverParam->hashedSumArray);
    return;
  }

  // Load particle state from SoA into registers
  const commonType xorig = pclsArray->getX()[pidx];
  const commonType yorig = pclsArray->getY()[pidx];
  const commonType zorig = pclsArray->getZ()[pidx];
  const commonType uorig = pclsArray->getU()[pidx];
  const commonType vorig = pclsArray->getV()[pidx];
  const commonType worig = pclsArray->getW()[pidx];
  commonType xavg = xorig;
  commonType yavg = yorig;
  commonType zavg = zorig;
  commonType uavg, vavg, wavg;
  commonType uavg_old = uorig;
  commonType vavg_old = vorig;
  commonType wavg_old = worig;

  int innter = 0;
  bool notConverged = true;

  // calculate the average velocity iteratively
  while (notConverged && innter < moverParam->NiterMover) {

    // sample E and B field components at current average position
    commonType weights[8];
    int cx, cy, cz;
    commonType sampled_field[6];
    sampleFieldsAtPosition<6>(xavg, yavg, zavg, grid, fieldForPcls, weights, cx,
                              cy, cz, sampled_field);

    commonType& Bxl = sampled_field[0];
    commonType& Byl = sampled_field[1];
    commonType& Bzl = sampled_field[2];
    commonType& Exl = sampled_field[3];
    commonType& Eyl = sampled_field[4];
    commonType& Ezl = sampled_field[5];
    const commonType Omx = qdto2mc * Bxl;
    const commonType Omy = qdto2mc * Byl;
    const commonType Omz = qdto2mc * Bzl;

    // end interpolation
    const commonType omsq = (Omx * Omx + Omy * Omy + Omz * Omz);
    const commonType denom = 1.0 / (1.0 + omsq);
    // solve the position equation
    const commonType ut = uorig + qdto2m * Exl;
    const commonType vt = vorig + qdto2m * Eyl;
    const commonType wt = worig + qdto2m * Ezl;
    const commonType udotOm = ut * Omx + vt * Omy + wt * Omz;
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

  } // end of iteration

  // Compute final position and velocity in registers
  commonType xfinal = xorig + uavg * moverParam->dt;
  commonType yfinal = yorig + vavg * moverParam->dt;
  commonType zfinal = zorig + wavg * moverParam->dt;
  commonType ufinal = 2.0 * uavg - uorig;
  commonType vfinal = 2.0 * vavg - vorig;
  commonType wfinal = 2.0 * wavg - worig;

  // Cap velocity to prevent superluminal particles.
  // The non-relativistic implicit mover has no intrinsic speed-of-light limit,
  // so extreme E fields can produce |v| >= c.
  const commonType vmax = kVelocityCapOverC * moverParam->c;
  if (capVelocityOrDelete(ufinal, vfinal, wfinal, vmax * vmax, moverParam,
                          pidx))
    return;

  // Write final state to SoA
  pclsArray->getX()[pidx] = xfinal;
  pclsArray->getY()[pidx] = yfinal;
  pclsArray->getZ()[pidx] = zfinal;
  pclsArray->getU()[pidx] = ufinal;
  pclsArray->getV()[pidx] = vfinal;
  pclsArray->getW()[pidx] = wfinal;

  // prepare the departure array
  prepareDepartureArray(xfinal, yfinal, zfinal, xorig, yorig, zorig, pclsArray,
                        pidx, moverParam, moverParam->departureArray, grid,
                        moverParam->hashedSumArray);
}

/**
 * @brief Advance one particle with adaptive subcycling based on local magnetic
 * field strength.
 *
 * The full step is split into smaller substeps when the local gyrofrequency is
 * large. State is kept in registers across the subcycle loop and written back
 * only once at the end.
 *
 * @param moverParam Device-side mover parameter bundle for one species.
 * @param fieldForPcls Packed mover field buffer.
 * @param grid Device-side grid descriptor.
 */
__global__ void MOVER_LAUNCH_BOUNDS moverSubcyclesKernel(
    moverParameter* moverParam, cudaTypeArray1<cudaFieldType> fieldForPcls,
    grid3DCUDA* grid) {
  uint pidx = blockIdx.x * blockDim.x + threadIdx.x;

  auto pclsArray = moverParam->pclsArray;
  if (pidx >= pclsArray->getNOP())
    return;

  // Early exit for particles already deleted (e.g. during merging)
  if (moverParam->departureArray->getArray()[pidx].dest != 0) {
    // not moved: the position is both start and end
    const commonType x = pclsArray->getX()[pidx];
    const commonType y = pclsArray->getY()[pidx];
    const commonType z = pclsArray->getZ()[pidx];
    prepareDepartureArray(x, y, z, x, y, z, pclsArray, pidx, moverParam,
                          moverParam->departureArray, grid,
                          moverParam->hashedSumArray);
    return;
  }

  // Load particle state from SoA and keep it in registers across all
  // subcycles.  The push works in units of c: cur_u/v/w hold beta = v / c
  // (gamma = 1 / sqrt(1 - beta^2), momentum u = gamma beta), and positions
  // advance with c beta.
  const commonType c = moverParam->c;
  const commonType invc = moverParam->invc;
  commonType cur_x = pclsArray->getX()[pidx];
  commonType cur_y = pclsArray->getY()[pidx];
  commonType cur_z = pclsArray->getZ()[pidx];
  commonType cur_u = pclsArray->getU()[pidx] * invc;
  commonType cur_v = pclsArray->getV()[pidx] * invc;
  commonType cur_w = pclsArray->getW()[pidx] * invc;

  // The fields at the start position: |B| there sets the number of
  // subcycles, and the first iteration of the first subcycle samples there
  commonType weights[8];
  int cx, cy, cz;
  commonType sampled_field[6];
  sampleFieldsAtPosition<6>(cur_x, cur_y, cur_z, grid, fieldForPcls, weights,
                            cx, cy, cz, sampled_field);
  bool fieldsAtAvg = true; // sampled_field holds the fields at x/y/zavg

  // evaluate local B field magnitude
  const commonType B_mag = sqrt(sampled_field[0] * sampled_field[0] +
                                sampled_field[1] * sampled_field[1] +
                                sampled_field[2] * sampled_field[2]);

  // number of sub cycles: at most pi/4 of gyration per subcycle, so
  // nsub = dt |q/m| |B| / (c pi/4).  int() of a NaN or of a value past
  // INT_MAX is undefined: a NaN |B| takes one subcycle (its NaN result
  // deletes the particle after the push), a huge one kMaxSubcycles.
  const commonType nsub =
      (4 / M_PI) * fabs(moverParam->qom) * moverParam->dt * invc * B_mag;
  const int sub_cycles = !(nsub >= 0)               ? 1
                         : nsub < kMaxSubcycles - 1 ? (int)nsub + 1
                                                    : kMaxSubcycles;
  const commonType dt_sub = moverParam->dt / (commonType)(sub_cycles);

  const commonType dto2_sub = .5 * dt_sub;
  const commonType qdto2mc_sub = moverParam->qom * dto2_sub * invc;
  // position steps for a velocity in units of c
  const commonType cdto2_sub = c * dto2_sub;
  const commonType cdt_sub = c * dt_sub;

  // Safety: check initial velocity before entering subcycle loop.
  // If beta² >= 1 or NaN (e.g. from MPI injection or previous-cycle noise),
  // gamma0 = 1/sqrt(1-beta²) would produce NaN, cascading through the mover.
  // After subcycle 0, the velocity cap at the end of each subcycle keeps
  // beta² at kVelocityCapOverC² or below (to round-off), well below 1, so
  // this check is only needed once.
  {
    const commonType v2_init = cur_u * cur_u + cur_v * cur_v + cur_w * cur_w;
    if (!(v2_init < 1.0)) {
      markForDeletion(moverParam, pidx);
      return;
    }
  }

  // Start subcycling; keep the state in registers across the loop.
  for (int cyc_cnt = 0; cyc_cnt < sub_cycles; cyc_cnt++) {
    const commonType xorig = cur_x;
    const commonType yorig = cur_y;
    const commonType zorig = cur_z;
    const commonType uorig = cur_u;
    const commonType vorig = cur_v;
    const commonType worig = cur_w;
    commonType xavg = xorig;
    commonType yavg = yorig;
    commonType zavg = zorig;
    commonType uavg, vavg, wavg;
    commonType uavg_old = uorig;
    commonType vavg_old = vorig;
    commonType wavg_old = worig;

    const commonType vorig_sq = uorig * uorig + vorig * vorig + worig * worig;
    const commonType gamma0 = rsqrt(1.0 - vorig_sq);
    commonType invGamma1; // 1 / gamma~

    int innter = 0;
    bool notConverged = true;

    // calculate the average velocity iteratively - predictor corrector
    // uses dt_subcycle
    while (notConverged && innter < moverParam->NiterMover) {

      // sample E and B field components at current average position
      if (!fieldsAtAvg)
        sampleFieldsAtPosition<6>(xavg, yavg, zavg, grid, fieldForPcls, weights,
                                  cx, cy, cz, sampled_field);
      fieldsAtAvg = false;

      const commonType Bxl = sampled_field[0];
      const commonType Byl = sampled_field[1];
      const commonType Bzl = sampled_field[2];
      const commonType Exl = sampled_field[3];
      const commonType Eyl = sampled_field[4];
      const commonType Ezl = sampled_field[5];

      // half electric kick: u~ = gamma(m) beta(m) + (q h / 2 m c) E
      const commonType ut = uorig * gamma0 + qdto2mc_sub * Exl;
      const commonType vt = vorig * gamma0 + qdto2mc_sub * Eyl;
      const commonType wt = worig * gamma0 + qdto2mc_sub * Ezl;
      invGamma1 = rsqrt(1.0 + ut * ut + vt * vt + wt * wt);

      // rotation vector w / gamma~ with w = (q h / 2 m c) B (Peng et al.
      // 2015, Eq. 2.8): the relativistic gyrofrequency is Omega / gamma
      const commonType qdto2mcg = qdto2mc_sub * invGamma1;
      const commonType Omx = qdto2mcg * Bxl;
      const commonType Omy = qdto2mcg * Byl;
      const commonType Omz = qdto2mcg * Bzl;
      const commonType omsq = (Omx * Omx + Omy * Omy + Omz * Omz);
      // beta_bar = alpha(w / gamma~) u~ / gamma~
      const commonType denom = invGamma1 / (1.0 + omsq);

      const commonType udotOm = ut * Omx + vt * Omy + wt * Omz;
      // solve the velocity equation
      uavg = (ut + (vt * Omz - wt * Omy + udotOm * Omx)) * denom;
      vavg = (vt + (wt * Omx - ut * Omz + udotOm * Omy)) * denom;
      wavg = (wt + (ut * Omy - vt * Omx + udotOm * Omz)) * denom;
      // update average position
      xavg = xorig + uavg * cdto2_sub;
      yavg = yorig + vavg * cdto2_sub;
      zavg = zorig + wavg * cdto2_sub;

      notConverged =
          pcNotConverged(uavg, vavg, wavg, uavg_old, vavg_old, wavg_old);
      // capture the new velocity for the next iteration
      uavg_old = uavg;
      vavg_old = vavg;
      wavg_old = wavg;

      innter++;

    } // end of iteration

    // relativistic velocity update
    const commonType ut = uorig * gamma0;
    const commonType vt = vorig * gamma0;
    const commonType wt = worig * gamma0;

    const commonType velt_sq = ut * ut + vt * vt + wt * wt;
    const commonType velavg_sq = uavg * uavg + vavg * vavg + wavg * wavg;
    const commonType velt_velavg = ut * uavg + vt * vavg + wt * wavg;

    const commonType cfa = 1.0 - velavg_sq;
    const commonType cfb = -2.0 * (-velt_velavg + gamma0 * velavg_sq);
    const commonType cfc = -1.0 - gamma0 * gamma0 * velavg_sq +
                           2.0 * gamma0 * velt_velavg - velt_sq;

    const commonType delta_rel = cfb * cfb - 4.0 * cfa * cfc;

    // update position and velocity in registers
    cur_x = xorig + uavg * cdt_sub;
    cur_y = yorig + vavg * cdt_sub;
    cur_z = zorig + wavg * cdt_sub;
    if (delta_rel < 0.0) {
      // Unreachable in exact arithmetic (cfa > 0 because |beta_bar| < 1, and
      // cfc < 0); only round-off with |beta_bar| -> 1 lands here.  Estimate
      // the momentum u(m+1) = 2 gamma~ beta_bar - u(m) and store it as a
      // velocity, beta = u / sqrt(1 + u^2), which keeps |beta| < 1.
      const commonType twoGamma1 = 2.0 / invGamma1;
      const commonType un = twoGamma1 * uavg - ut;
      const commonType vn = twoGamma1 * vavg - vt;
      const commonType wn = twoGamma1 * wavg - wt;
      const commonType invGammaNew = rsqrt(1.0 + un * un + vn * vn + wn * wn);
      cur_u = un * invGammaNew;
      cur_v = vn * invGammaNew;
      cur_w = wn * invGammaNew;
    } else {
      // 1 / gamma(m+1), gamma(m+1) = (-cfb + sqrt(delta_rel)) / (2 cfa) the
      // positive root of cfa g^2 + cfb g + cfc = 0.  Then beta(m+1) =
      // u(m+1) / gamma(m+1) with u(m+1) = (gamma(m) + gamma(m+1)) beta_bar -
      // u(m).
      const commonType invGammaNew = (2.0 * cfa) / (sqrt(delta_rel) - cfb);
      const commonType coef = 1.0 + gamma0 * invGammaNew;
      cur_u = coef * uavg - ut * invGammaNew;
      cur_v = coef * vavg - vt * invGammaNew;
      cur_w = coef * wavg - wt * invGammaNew;
    }

    // Cap velocity to prevent superluminal particles.
    // If beta^2 >= 1, the next subcycle's gamma = 1/sqrt(1-beta^2) -> NaN.
    if (capVelocityOrDelete(cur_u, cur_v, cur_w,
                            kVelocityCapOverC * kVelocityCapOverC, moverParam,
                            pidx))
      return;

  } // end iteration over subcycles

  // The open outflow faces need the position before the push, which the
  // subcycles overwrote in registers; the SoA holds it until the write below.
  commonType xstart = cur_x, ystart = cur_y, zstart = cur_z;
  if (moverParam->doOpenBC) {
    xstart = pclsArray->getX()[pidx];
    ystart = pclsArray->getY()[pidx];
    zstart = pclsArray->getZ()[pidx];
  }

  // Write final state to SoA once (velocities back from units of c)
  pclsArray->getX()[pidx] = cur_x;
  pclsArray->getY()[pidx] = cur_y;
  pclsArray->getZ()[pidx] = cur_z;
  pclsArray->getU()[pidx] = cur_u * c;
  pclsArray->getV()[pidx] = cur_v * c;
  pclsArray->getW()[pidx] = cur_w * c;

  // prepare the departure array
  prepareDepartureArray(cur_x, cur_y, cur_z, xstart, ystart, zstart, pclsArray,
                        pidx, moverParam, moverParam->departureArray, grid,
                        moverParam->hashedSumArray);
}

// ======= Boundary classification helpers =======

/**
 * @brief Delete particles that end the push in the repopulation layers, the 3
 * cells before a REEMISSION face.
 *
 * Applies to every species: the particles leave the domain there, and
 * injectionKernel refills the layers with the species that have rhoINJECT > 0.
 *
 * @param xpcl Particle x position after the mover.
 * @param ypcl Particle y position after the mover.
 * @param zpcl Particle z position after the mover.
 * @param moverParam Device-side mover parameter bundle.
 * @param grid Device-side grid descriptor.
 * @return `DELETE` when the particle should be removed, otherwise 0.
 */
__device__ uint32_t deleteRepopulateInjection(commonType xpcl, commonType ypcl,
                                              commonType zpcl,
                                              moverParameter* moverParam,
                                              grid3DCUDA* grid) {
  if (!moverParam->doRepopulateInjection)
    return 0;

  auto& doRepopulateInjectionSide = moverParam->doRepopulateInjectionSide;
  auto& repopulateBoundary = moverParam->repopulateBoundary;

  if ((doRepopulateInjectionSide[0] && xpcl < repopulateBoundary[0]) ||
      (doRepopulateInjectionSide[1] && xpcl > repopulateBoundary[1]) ||
      (doRepopulateInjectionSide[2] && ypcl < repopulateBoundary[2]) ||
      (doRepopulateInjectionSide[3] && ypcl > repopulateBoundary[3]) ||
      (doRepopulateInjectionSide[4] && zpcl < repopulateBoundary[4]) ||
      (doRepopulateInjectionSide[5] &&
       zpcl > repopulateBoundary[5])) { // In the repopulate layers
    return departureArrayElementType::DELETE;
  } else {
    return 0;
  }
}

/**
 * @brief Append a particle that enters through the open outflow faces, and
 * classify it for the exchange.
 *
 * @param pos Position of the new particle.
 * @param vel Velocity of the new particle.
 * @param charge Charge of the new particle.
 * @param pclsArray Device-side particle SoA container.
 * @param moverParam Device-side mover parameter bundle.
 * @param departureArray Device-side departure metadata array.
 * @param grid Device-side grid descriptor.
 * @param hashedSumArray Device-side hashed-sum buckets for departure
 * compaction.
 * @return False when the append tail is full (the run then aborts).
 */
__device__ bool appendOpenBCCopy(const commonType pos[3],
                                 const commonType vel[3], commonType charge,
                                 particleArrayCUDA* pclsArray,
                                 moverParameter* moverParam,
                                 departureArrayType* departureArray,
                                 grid3DCUDA* grid, hashedSum* hashedSumArray) {
  const uint32_t appendBase = pclsArray->getNOP();
  const uint32_t capacity = pclsArray->getCapacity();
  const uint32_t appendCapacity =
      appendBase < capacity ? capacity - appendBase : 0u;
  const uint32_t appendOffset =
      atomicAdd(&moverParam->openBCAppendCounters.accepted, 1u);

  if (appendOffset >= appendCapacity) {
    // Keep the final accepted count equal to the dense initialized
    // append prefix. The rejected ticket is necessarily outside that
    // prefix, so rollback cannot make a valid slot available twice.
    atomicSub(&moverParam->openBCAppendCounters.accepted, 1u);
    atomicAdd(&moverParam->openBCAppendCounters.rejected, 1u);
    return false;
  }

  const uint32_t index = appendBase + appendOffset;
  departureArrayElementType element;
  // Write new particle to SoA arrays
  pclsArray->getX()[index] = pos[0];
  pclsArray->getY()[index] = pos[1];
  pclsArray->getZ()[index] = pos[2];
  pclsArray->getU()[index] = vel[0];
  pclsArray->getV()[index] = vel[1];
  pclsArray->getW()[index] = vel[2];
  pclsArray->getQ()[index] = charge;
  if (pclsArray->tracksParticleID())
    pclsArray->getID()[index] = moverParam->particleIDGenerator.generateID();

  if (pos[0] < grid->xStart) {
    element.dest = departureArrayElementType::XLOW;
  } else if (pos[0] > grid->xEnd) {
    element.dest = departureArrayElementType::XHIGH;
  } else if (pos[1] < grid->yStart) {
    element.dest = departureArrayElementType::YLOW;
  } else if (pos[1] > grid->yEnd) {
    element.dest = departureArrayElementType::YHIGH;
  } else if (pos[2] < grid->zStart) {
    element.dest = departureArrayElementType::ZLOW;
  } else if (pos[2] > grid->zEnd) {
    element.dest = departureArrayElementType::ZHIGH;
  } else
    element.dest = departureArrayElementType::STAY;

  if (element.dest != 0) {
    element.hashedId = hashedSumArray[element.dest - 1].add(index);
  } else {
    element.hashedId = 0;
  }

  departureArray->getArray()[index] = element;
  return true;
}

/**
 * @brief Delete particles beyond an open outflow face, and append the
 * particles that enter through it.
 *
 * Beyond each open outflow face the plasma is the band of the last 3 cells
 * shifted 3 cells outward (zero-gradient continuation), and its particles move
 * as their sources in the band.  A particle that crosses the inner edge of the
 * band inward during the push therefore re-enters through the face: a copy
 * with its velocity and charge is appended at its end position shifted 3
 * cells outward, and pushed from the next cycle.  Beyond an edge or a corner
 * of the domain the plasma is the band shifted across each of its faces, so a
 * particle that crosses the band edges of several faces gives one copy for
 * each set of them.  A copy is kept if it lies inside the domain and outside
 * the repopulation layers, which injectionKernel refills.
 *
 * @param xpcl Particle x position after the mover.
 * @param ypcl Particle y position after the mover.
 * @param zpcl Particle z position after the mover.
 * @param xstart Particle x position before the mover.
 * @param ystart Particle y position before the mover.
 * @param zstart Particle z position before the mover.
 * @param pclsArray Device-side particle SoA container.
 * @param pidx Particle index in the species SoA buffer.
 * @param moverParam Device-side mover parameter bundle.
 * @param departureArray Device-side departure metadata array.
 * @param grid Device-side grid descriptor.
 * @param hashedSumArray Device-side hashed-sum buckets for departure
 * compaction.
 * @return `DELETE` for a particle beyond an open outflow face, otherwise 0.
 */
__device__ uint32_t deleteAppendOpenBCOutflow(
    commonType xpcl, commonType ypcl, commonType zpcl, commonType xstart,
    commonType ystart, commonType zstart, particleArrayCUDA* pclsArray,
    uint32_t pidx, moverParameter* moverParam,
    departureArrayType* departureArray, grid3DCUDA* grid,
    hashedSum* hashedSumArray) {

  if (!moverParam->doOpenBC)
    return 0;

  auto& delBdry = moverParam->deleteBoundary;
  auto& openBdry = moverParam->openBoundary;

  const commonType pos[3] = {xpcl, ypcl, zpcl};
  const commonType start[3] = {xstart, ystart, zstart};

  // The axes whose band edge the particle crossed inward (one bit each), and
  // the outward shift of the copies across them
  int crossed = 0;
  commonType shift[3] = {0, 0, 0};

  for (int side = 0; side < 6; side++) {

    if (!moverParam->applyOpenBC[side])
      continue;

    const int direction = side / 2; // x,y,z
    const bool right = side % 2;
    const commonType location = pos[direction];
    const commonType from = start[direction];

    // Beyond the face: deleted.  Its copies would lie beyond it too.
    if (right ? location > delBdry[side] : location < delBdry[side])
      return departureArrayElementType::DELETE;

    // inner edge of the band crossed inward: the particle re-enters (the
    // band includes its edge, so a particle that ends a push on it is
    // counted when it leaves)
    if (right ? from >= openBdry[side] && location < openBdry[side]
              : from <= openBdry[side] && location > openBdry[side]) {
      crossed |= 1 << direction;
      shift[direction] =
          right ? openBdry[direction * 2] : -openBdry[direction * 2];
    }
  }

  if (crossed == 0)
    return 0;

  // Velocity (already the new one) and charge from SoA for the copies
  const commonType vel[3] = {pclsArray->getU()[pidx], pclsArray->getV()[pidx],
                             pclsArray->getW()[pidx]};
  const commonType charge = pclsArray->getQ()[pidx];

  // One copy for each nonempty set of the crossed faces
  for (int set = crossed; set != 0; set = (set - 1) & crossed) {
    commonType newPos[3];
    for (int d = 0; d < 3; d++)
      newPos[d] = (set >> d) & 1 ? pos[d] + shift[d] : pos[d];

    // kept inside the domain and outside the repopulation layers
    if (!(newPos[0] > delBdry[0] && newPos[0] < delBdry[1] &&
          newPos[1] > delBdry[2] && newPos[1] < delBdry[3] &&
          newPos[2] > delBdry[4] && newPos[2] < delBdry[5]) ||
        deleteRepopulateInjection(newPos[0], newPos[1], newPos[2], moverParam,
                                  grid))
      continue;

    if (!appendOpenBCCopy(newPos, vel, charge, pclsArray, moverParam,
                          departureArray, grid, hashedSumArray))
      return departureArrayElementType::DELETE;
  }

  return 0;
}

/**
 * @brief Tag particles that crossed into the absorbing planet region.
 *
 * Supports both the full 3D sphere and the 2D XZ-plane circle used by the
 * dipole-2D configuration.
 *
 * @param xpcl Particle x position after the mover.
 * @param ypcl Particle y position after the mover.
 * @param zpcl Particle z position after the mover.
 * @param moverParam Device-side mover parameter bundle.
 * @param grid Device-side grid descriptor.
 * @return `PLANET` when the particle is inside the body, otherwise 0.
 */
__device__ uint32_t deleteInsideSphere(commonType xpcl, commonType ypcl,
                                       commonType zpcl,
                                       moverParameter* moverParam,
                                       grid3DCUDA* grid) {

  if (moverParam->doSphere == 0)
    return 0;

  if (moverParam->doSphere == 1) { // 3D sphere
    const auto& sphereOrigin = moverParam->sphereOrigin;
    const auto& sphereRadius = moverParam->sphereRadius;

    const auto dx = xpcl - sphereOrigin[0];
    const auto dy = ypcl - sphereOrigin[1];
    const auto dz = zpcl - sphereOrigin[2];

    if (dx * dx + dy * dy + dz * dz < sphereRadius * sphereRadius) {
      return departureArrayElementType::PLANET;
    }
  } else if (moverParam->doSphere == 2) { // 2D sphere
    const auto& sphereOrigin = moverParam->sphereOrigin;
    const auto& sphereRadius = moverParam->sphereRadius;

    const auto dx = xpcl - sphereOrigin[0];
    const auto dz = zpcl - sphereOrigin[2];

    if (dx * dx + dz * dz < sphereRadius * sphereRadius) {
      return departureArrayElementType::PLANET;
    }
  }

  return 0;
}
// ======= Departure classification =======

/**
 * @brief Finalize departure metadata for one moved particle.
 *
 * The routine applies, in order, open-boundary outflow handling, repopulation
 * injection deletion, planet absorption, and regular domain-exit checks. Any
 * non-staying destination is registered in the matching hashed-sum bucket.
 *
 * @param xpcl Particle x position after the mover.
 * @param ypcl Particle y position after the mover.
 * @param zpcl Particle z position after the mover.
 * @param xstart Particle x position before the mover.
 * @param ystart Particle y position before the mover.
 * @param zstart Particle z position before the mover.
 * @param pclsArray Device-side particle SoA container.
 * @param pidx Particle index in the species SoA buffer.
 * @param moverParam Device-side mover parameter bundle.
 * @param departureArray Device-side departure metadata array.
 * @param grid Device-side grid descriptor.
 * @param hashedSumArray Device-side hashed-sum buckets for departure
 * compaction.
 */
__device__ void prepareDepartureArray(commonType xpcl, commonType ypcl,
                                      commonType zpcl, commonType xstart,
                                      commonType ystart, commonType zstart,
                                      particleArrayCUDA* pclsArray,
                                      uint32_t pidx, moverParameter* moverParam,
                                      departureArrayType* departureArray,
                                      grid3DCUDA* grid,
                                      hashedSum* hashedSumArray) {

  if (departureArray->getArray()[pidx].dest != 0) {
    departureArray->getArray()[pidx].hashedId =
        hashedSumArray[departureArray->getArray()[pidx].dest - 1].add(pidx);

    return;
  }

  // Safety: NaN positions bypass all comparison-based boundary checks
  // (IEEE 754: NaN < x and NaN > x are both false), so a NaN particle
  // would fall through to STAY and deposit NaN into the moments array.
  if (!isfinite(xpcl) || !isfinite(ypcl) || !isfinite(zpcl)) {
    departureArrayElementType element;
    element.dest = departureArrayElementType::DELETE;
    element.hashedId =
        hashedSumArray[departureArrayElementType::DELETE_HASHEDSUM_INDEX].add(
            pidx);
    departureArray->getArray()[pidx] = element;
    return;
  }

  departureArrayElementType element;

  do {

    // OpenBC_outflow
    element.dest = deleteAppendOpenBCOutflow(
        xpcl, ypcl, zpcl, xstart, ystart, zstart, pclsArray, pidx, moverParam,
        departureArray, grid, hashedSumArray);
    if (element.dest != 0)
      break;

    // INJECT
    element.dest =
        deleteRepopulateInjection(xpcl, ypcl, zpcl, moverParam, grid);
    if (element.dest != 0)
      break;

    // sphere
    element.dest = deleteInsideSphere(xpcl, ypcl, zpcl, moverParam, grid);
    if (element.dest != 0)
      break;

    // Exiting

    if (xpcl < grid->xStart) {
      element.dest = moverParam->isExitBC[0] ? departureArrayElementType::DELETE
                                             : departureArrayElementType::XLOW;
    } else if (xpcl > grid->xEnd) {
      element.dest = moverParam->isExitBC[1] ? departureArrayElementType::DELETE
                                             : departureArrayElementType::XHIGH;
    } else if (ypcl < grid->yStart) {
      element.dest = moverParam->isExitBC[2] ? departureArrayElementType::DELETE
                                             : departureArrayElementType::YLOW;
    } else if (ypcl > grid->yEnd) {
      element.dest = moverParam->isExitBC[3] ? departureArrayElementType::DELETE
                                             : departureArrayElementType::YHIGH;
    } else if (zpcl < grid->zStart) {
      element.dest = moverParam->isExitBC[4] ? departureArrayElementType::DELETE
                                             : departureArrayElementType::ZLOW;
    } else if (zpcl > grid->zEnd) {
      element.dest = moverParam->isExitBC[5] ? departureArrayElementType::DELETE
                                             : departureArrayElementType::ZHIGH;
    } else
      element.dest = departureArrayElementType::STAY;

  } while (0);

  if (element.dest != 0) {
    element.hashedId = hashedSumArray[element.dest - 1].add(pidx);
  } else {
    element.hashedId = 0;
  }

  departureArray->getArray()[pidx] = element;
}
