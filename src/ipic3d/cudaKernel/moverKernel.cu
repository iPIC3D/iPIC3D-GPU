#include <assert.h>
#include <cstdint>
#include <math.h>

#include "cudaTypeDef.cuh"
#include "gridCUDA.cuh"
#include "hashedSum.cuh"
#include "moverKernel.cuh"
#include "moverPush.cuh"
#include "particleArrayCUDA.cuh"
#include "particleExchange.cuh"

using commonType = cudaParticleType;

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
 * @brief The field gather of the mover kernels: B and E at a position, from
 * the packed node-centered field buffer (per cell, its 8 corner nodes with 6
 * components each).
 */
struct GridFields {
  grid3DCUDA* grid;
  cudaTypeArray1<cudaFieldType> buffer;

  /** f = {Bx, By, Bz, Ex, Ey, Ez} at (x, y, z). */
  __device__ __forceinline__ void
  operator()(commonType x, commonType y, commonType z, commonType f[6]) const {
    commonType weights[8];
    int cx, cy, cz;
    grid->get_safe_cell_and_weights(x, y, z, cx, cy, cz, weights);
    const int cell = (cx * (grid->nyn - 1) + cy) * grid->nzn + cz;
    assert(cell >= 0 && cell < (grid->nxn - 1) * (grid->nyn - 1) * grid->nzn);
    for (int i = 0; i < 6; i++)
      f[i] = 0;
    for (int c = 0; c < 8; c++)
      for (int i = 0; i < 6; i++)
        f[i] += weights[c] * buffer[cell * 24 + c * 6 + i];
  }
};

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

/** How the pushes of one particle over the step ended. */
enum class PushEnd : int {
  Pushed,  ///< the particle is at its end state
  Deleted, ///< a velocity at or beyond c, or not finite: marked for deletion
  Wall,    ///< a push ended beyond a reflecting wall: the particle is left
};

/**
 * @brief One push of duration h from s, in place (a push reads its start
 * before it writes its end).
 *
 * With kWalls the push reflects at the reflecting walls (pushWithWalls).
 * Without it, a push that ends beyond a reflecting wall is reported
 * (PushEnd::Wall), for the caller to abandon the particle.
 *
 * @param speed Physical speed per unit of the state's velocity.
 * @param push push(from, h, to): the push; false when it deleted the
 *        particle.
 */
template <bool kWalls, typename Push>
__device__ __forceinline__ PushEnd pushToWalls(moverParameter* moverParam,
                                               MoverState& s, commonType h,
                                               commonType speed,
                                               const Push& push) {
  if constexpr (kWalls) {
    if (!pushWithWalls(s, h, moverParam->reflectLo, moverParam->reflectHi,
                       moverParam->reflectTol, speed, push))
      return PushEnd::Deleted;
  } else {
    if (!push(s, h, s))
      return PushEnd::Deleted;
    if (moverParam->doReflect &&
        beyondWalls(s, moverParam->reflectLo, moverParam->reflectHi))
      return PushEnd::Wall;
  }
  return PushEnd::Pushed;
}

/**
 * @brief The standard push of one particle over the full solver time step:
 * the predictor-corrector push (implicitPush) from its SoA state, with or
 * without the walls (pushToWalls).
 *
 * @param end Output: the state after the push.
 */
template <bool kWalls>
__device__ __forceinline__ PushEnd pushStandard(moverParameter* moverParam,
                                                const GridFields& fields,
                                                uint32_t pidx,
                                                MoverState& end) {
  auto pclsArray = moverParam->pclsArray;
  end = {{pclsArray->getX()[pidx], pclsArray->getY()[pidx],
          pclsArray->getZ()[pidx]},
         {pclsArray->getU()[pidx], pclsArray->getV()[pidx],
          pclsArray->getW()[pidx]}};
  const auto push = [&](const MoverState& from, commonType h, MoverState& to) {
    implicitPush(from, h, moverParam->qom, moverParam->invc,
                 moverParam->NiterMover, fields, to);
    return true;
  };
  const PushEnd how =
      pushToWalls<kWalls>(moverParam, end, moverParam->dt, 1.0, push);
  if (how != PushEnd::Pushed)
    return how;

  // Cap velocity to prevent superluminal particles.
  // The non-relativistic implicit mover has no intrinsic speed-of-light limit,
  // so extreme E fields can produce |v| >= c.
  const commonType vmax = kVelocityCapOverC * moverParam->c;
  return capVelocityOrDelete(end.v[0], end.v[1], end.v[2], vmax * vmax,
                             moverParam, pidx)
             ? PushEnd::Deleted
             : PushEnd::Pushed;
}

/**
 * @brief The push of one particle over the step with adaptive subcycling
 * based on local magnetic field strength.
 *
 * The full step is split into smaller substeps when the local gyrofrequency
 * is large, each a relativistic push (relativisticPush) followed by the
 * velocity cap, with or without the walls (pushToWalls).
 *
 * @param cur Output: the state after the push.
 */
template <bool kWalls>
__device__ __forceinline__ PushEnd pushSubcycled(moverParameter* moverParam,
                                                 const GridFields& fields,
                                                 uint32_t pidx,
                                                 MoverState& cur) {
  // Load particle state from SoA and keep it in registers across all
  // subcycles.  The push works in units of c: cur.v holds beta = v / c
  // (gamma = 1 / sqrt(1 - beta^2), momentum u = gamma beta), and positions
  // advance with c beta.
  auto pclsArray = moverParam->pclsArray;
  const commonType c = moverParam->c;
  const commonType invc = moverParam->invc;
  cur = {{pclsArray->getX()[pidx], pclsArray->getY()[pidx],
          pclsArray->getZ()[pidx]},
         {pclsArray->getU()[pidx] * invc, pclsArray->getV()[pidx] * invc,
          pclsArray->getW()[pidx] * invc}};

  // The fields at the start position: |B| there sets the number of
  // subcycles, and the first gather of the first subcycle is there.  Every
  // gather goes through the one array, so that the start fields occupy no
  // registers of their own during the later gathers.
  commonType sampled[6];
  fields(cur.x[0], cur.x[1], cur.x[2], sampled);
  bool atStart = true;
  const auto gather = [&](commonType x, commonType y, commonType z,
                          commonType f[6]) {
    if (!atStart)
      fields(x, y, z, sampled);
    atStart = false;
#pragma unroll
    for (int i = 0; i < 6; i++)
      f[i] = sampled[i];
  };

  // evaluate local B field magnitude
  const commonType B_mag =
      sqrt(sampled[0] * sampled[0] + sampled[1] * sampled[1] +
           sampled[2] * sampled[2]);

  // number of sub cycles: at most pi/4 of gyration per subcycle, so
  // nsub = dt |q/m| |B| / (c pi/4).  int() of a NaN or of a value past
  // INT_MAX is undefined: a NaN |B| takes one subcycle (its NaN result
  // deletes the particle after the push), a huge one kMaxSubcycles.
  const commonType qom = moverParam->qom;
  const commonType nsub =
      (4 / M_PI) * fabs(qom) * moverParam->dt * invc * B_mag;
  const int sub_cycles = !(nsub >= 0)               ? 1
                         : nsub < kMaxSubcycles - 1 ? (int)nsub + 1
                                                    : kMaxSubcycles;
  const commonType dt_sub = moverParam->dt / (commonType)(sub_cycles);

  // Safety: check initial velocity before entering subcycle loop.
  // If beta² >= 1 or NaN (e.g. from MPI injection or previous-cycle noise),
  // gamma0 = 1/sqrt(1-beta²) would produce NaN, cascading through the mover.
  // After subcycle 0, the velocity cap at the end of each push keeps beta²
  // at kVelocityCapOverC² or below (to round-off), well below 1, so this
  // check is only needed once.
  {
    const commonType v2_init =
        cur.v[0] * cur.v[0] + cur.v[1] * cur.v[1] + cur.v[2] * cur.v[2];
    if (!(v2_init < 1.0)) {
      markForDeletion(moverParam, pidx);
      return PushEnd::Deleted;
    }
  }

  // One push, with the cap that keeps beta^2 < 1 for the next push's
  // gamma = 1 / sqrt(1 - beta^2); false when the particle was deleted
  const auto push = [&](const MoverState& from, commonType h, MoverState& to) {
    relativisticPush(from, h, qom, c, invc, moverParam->NiterMover, gather, to);
    return !capVelocityOrDelete(to.v[0], to.v[1], to.v[2],
                                kVelocityCapOverC * kVelocityCapOverC,
                                moverParam, pidx);
  };

  for (int cyc_cnt = 0; cyc_cnt < sub_cycles; cyc_cnt++) {
    const PushEnd how = pushToWalls<kWalls>(moverParam, cur, dt_sub, c, push);
    if (how != PushEnd::Pushed)
      return how;
  }

  // back to physical velocities
#pragma unroll
  for (int i = 0; i < 3; i++)
    cur.v[i] *= c;
  return PushEnd::Pushed;
}

/** What pushAgainWithWalls returns: the end state and how the push ended. */
struct WallPush {
  MoverState end;
  PushEnd how;
};

/**
 * @brief The push of a particle whose push ended beyond a reflecting wall and
 * that the wall list has no room for: the whole push again, from its SoA
 * state, with the walls.
 *
 * Not inlined and without particle state in its arguments: a mover kernel is
 * at its register bound, and the particles that reach a wall must keep
 * nothing alive, in registers or on the stack, during the push of all the
 * others.  Internal linkage: the function is compiled with each kernel that
 * calls it, under that kernel's register bound.
 */
template <bool kRelativistic>
static __device__ __noinline__ WallPush pushAgainWithWalls(
    moverParameter* moverParam, GridFields fields, uint32_t pidx) {
  WallPush w;
  if constexpr (kRelativistic)
    w.how = pushSubcycled<true>(moverParam, fields, pidx, w.end);
  else
    w.how = pushStandard<true>(moverParam, fields, pidx, w.end);
  return w;
}

/**
 * @brief One particle of a mover kernel: the push over the step, standard or
 * subcycled relativistic, then the final state to the SoA and the departure
 * classification.
 *
 * A particle whose push ends beyond a reflecting wall is left untouched to
 * the wall pass (kWallPass), which pushes the particles of
 * moverParameter::wallList with the walls.  Few particles reach a wall in a
 * step: gathered, they fill warps of their own instead of holding the warps
 * of their neighbours through the wall loop.  A particle the list has no
 * room for is pushed again here.
 */
template <bool kRelativistic, bool kWallPass>
__device__ __forceinline__ void
moveParticle(moverParameter* moverParam,
             cudaTypeArray1<cudaFieldType> fieldForPcls, grid3DCUDA* grid) {
  const uint32_t thread = blockIdx.x * blockDim.x + threadIdx.x;
  auto pclsArray = moverParam->pclsArray;
  uint32_t pidx = thread;
  if constexpr (kWallPass) {
    // the thread's entry of the wall list
    const auto wallList = moverParam->wallList;
    if (thread >= moverParam->wallCount || thread >= wallList->getSize())
      return;
    pidx = wallList->getArray()[thread];
  } else {
    if (pidx >= pclsArray->getNOP())
      return;

    // Particles already deleted (e.g. during merging) are not moved: the
    // position is both start and end
    if (moverParam->departureArray->getArray()[pidx].dest != 0) {
      const commonType x = pclsArray->getX()[pidx];
      const commonType y = pclsArray->getY()[pidx];
      const commonType z = pclsArray->getZ()[pidx];
      prepareDepartureArray(x, y, z, x, y, z, pclsArray, pidx, moverParam,
                            moverParam->departureArray, grid,
                            moverParam->hashedSumArray);
      return;
    }
  }

  const GridFields fields = {grid, fieldForPcls};
  MoverState end;
  PushEnd how;
  if constexpr (kRelativistic)
    how = pushSubcycled<kWallPass>(moverParam, fields, pidx, end);
  else
    how = pushStandard<kWallPass>(moverParam, fields, pidx, end);
  if constexpr (!kWallPass) {
    if (how == PushEnd::Wall) {
      const auto wallList = moverParam->wallList;
      const uint32_t slot = atomicAdd(&moverParam->wallCount, 1u);
      if (slot < wallList->getSize()) {
        wallList->getArray()[slot] = pidx;
        return;
      }
      const WallPush w =
          pushAgainWithWalls<kRelativistic>(moverParam, fields, pidx);
      end = w.end;
      how = w.how;
    }
  }
  if (how == PushEnd::Deleted)
    return;

  // The open outflow faces need the position before the push: the SoA holds
  // it until the write below.
  commonType xstart = end.x[0], ystart = end.x[1], zstart = end.x[2];
  if (moverParam->doOpenBC) {
    xstart = pclsArray->getX()[pidx];
    ystart = pclsArray->getY()[pidx];
    zstart = pclsArray->getZ()[pidx];
  }

  // Write final state to SoA once
  pclsArray->getX()[pidx] = end.x[0];
  pclsArray->getY()[pidx] = end.x[1];
  pclsArray->getZ()[pidx] = end.x[2];
  pclsArray->getU()[pidx] = end.v[0];
  pclsArray->getV()[pidx] = end.v[1];
  pclsArray->getW()[pidx] = end.v[2];

  // prepare the departure array
  prepareDepartureArray(end.x[0], end.x[1], end.x[2], xstart, ystart, zstart,
                        pclsArray, pidx, moverParam, moverParam->departureArray,
                        grid, moverParam->hashedSumArray);
}

/** The standard mover (see moverKernel.cuh). */
__global__ void MOVER_LAUNCH_BOUNDS
moverKernel(moverParameter* moverParam,
            cudaTypeArray1<cudaFieldType> fieldForPcls, grid3DCUDA* grid) {
  moveParticle<false, false>(moverParam, fieldForPcls, grid);
}

/** The subcycled relativistic mover (see moverKernel.cuh). */
__global__ void MOVER_LAUNCH_BOUNDS moverSubcyclesKernel(
    moverParameter* moverParam, cudaTypeArray1<cudaFieldType> fieldForPcls,
    grid3DCUDA* grid) {
  moveParticle<true, false>(moverParam, fieldForPcls, grid);
}

/** The wall pass of moverKernel. */
__global__ void moverWallsKernel(moverParameter* moverParam,
                                 cudaTypeArray1<cudaFieldType> fieldForPcls,
                                 grid3DCUDA* grid) {
  moveParticle<false, true>(moverParam, fieldForPcls, grid);
}

/** The wall pass of moverSubcyclesKernel. */
__global__ void
moverSubcyclesWallsKernel(moverParameter* moverParam,
                          cudaTypeArray1<cudaFieldType> fieldForPcls,
                          grid3DCUDA* grid) {
  moveParticle<true, true>(moverParam, fieldForPcls, grid);
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
