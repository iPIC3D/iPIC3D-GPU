#ifndef MOVER_KERNEL_CUH
#define MOVER_KERNEL_CUH

#include "ParticleIDGenerator.cuh"
#include "ParticleSoAHost.h"
#include "cudaTypeDef.cuh"
#include "gridCUDA.cuh"
#include "hashedSum.cuh"
#include "particleArrayCUDA.cuh"
#include "particleExchange.cuh"

/** Threads per block of the mover kernels. */
constexpr int kMoverBlockSize = 256;

/** Per-cycle OpenBC append accounting shared by the mover and host pipeline. */
struct OpenBCAppendCounters {
  uint32_t accepted = 0; ///< Dense initialized prefix at mover completion.
  uint32_t rejected = 0; ///< Reservations rejected because the tail was full.
};

/** What the mover kernels of one species write (atomics), in device memory
 *  (moverParameter::counters). */
struct MoverCounters {
  /** The particles that reached a reflecting wall in this cycle: the first
   *  wallList->getSize() wait in moverParameter::wallList for the wall pass,
   *  the mover kernel pushed the others itself.  The host zeroes it after the
   *  wall pass. */
  uint32_t wallCount = 0;
  OpenBCAppendCounters openBCAppendCounters;
};

/**
 * @brief The parameters of the GPU mover kernels for one species, which they
 * take by value (KERNEL_CONSTANT): device buffers, scalar species constants
 * and boundary-condition flags.
 */
class moverParameter {

public:
  // device buffers
  particleArrayCUDA* pclsArray = nullptr;       // main device particle array
  departureArrayType* departureArray = nullptr; // device departure flags
  hashedSum* hashedSumArray = nullptr; // departure-direction hashed sums
  MoverCounters* counters = nullptr;   // what the kernels write
  // The indices of the particles a mover kernel left to its wall pass (see
  // MoverCounters::wallCount), held in the filler buffer of the compaction,
  // which is idle during the mover
  fillerBuffer* wallList = nullptr;

  // species constants
  cudaParticleType dt;
  cudaParticleType qom;
  cudaParticleType c;
  cudaParticleType invc; // 1 / c
  int NiterMover;

  // The per-face arrays [6] below are in the order XLeft, XRight, YLeft,
  // YRight, ZLeft, ZRight.

  // open boundaries (OpenBC)
  bool doOpenBC = false; // a OR of applyOpenBC
  bool applyOpenBC[6] = {};
  cudaCommonType deleteBoundary[6] = {};
  cudaCommonType openBoundary[6] = {};
  ParticleIDGenerator particleIDGenerator;

  // Repopulation layers of the open-inflow faces (deleteRepopulateInjection)
  bool doRepopulateInjection = false;
  bool doRepopulateInjectionSide[6] = {};
  cudaCommonType repopulateBoundary[6] = {};

  // For sphere
  int doSphere = 0; // 0: no sphere, 1: sphere, 2: sphere2D(XZ)
  cudaCommonType sphereOrigin[3] = {};
  cudaCommonType sphereRadius = 0;

  // Per-face EXIT BC flags (boundary ranks only): if true, particles exiting
  // this face are deleted on the GPU (dest=DELETE) instead of being sent via
  // MPI
  bool isExitBC[6] = {};

  // Reflecting walls (bcPface = 1) of this rank: per axis its low and high
  // boundary where the face reflects, -/+ infinity where it does not, and
  // the landing tolerance of a push on a wall (pushWithWalls)
  bool doReflect = false; // any wall
  cudaCommonType reflectLo[3] = {}, reflectHi[3] = {};
  cudaCommonType reflectTol = 0;

  /**
   * @brief The parameters of species pclHost on its device buffers, with its
   * scalar constants and boundary flags; countersCUDAPtr must hold zeros.
   * The caller sets reflectTol and the planet (doSphere, sphereOrigin,
   * sphereRadius).
   */
  __host__ moverParameter(const ParticleSoAHost* pclHost,
                          particleArrayCUDA* pclsArrayCUDAPtr,
                          departureArrayType* departureArrayCUDAPtr,
                          hashedSum* hashedSumArrayCUDAPtr,
                          fillerBuffer* wallListCUDAPtr,
                          MoverCounters* countersCUDAPtr)
      : pclsArray(pclsArrayCUDAPtr), departureArray(departureArrayCUDAPtr),
        hashedSumArray(hashedSumArrayCUDAPtr), counters(countersCUDAPtr),
        wallList(wallListCUDAPtr), dt(pclHost->timeStep_),
        qom(pclHost->chargeOverMass_), c(pclHost->speedOfLight_),
        invc(1.0 / pclHost->speedOfLight_),
        NiterMover(pclHost->numMoverIterations_) {
    if (pclHost->tracksParticleID())
      particleIDGenerator = pclHost->getParticleIDGenerator();
    pclHost->openbc_particles_outflowInfo(&doOpenBC, applyOpenBC,
                                          deleteBoundary, openBoundary);
    pclHost->fillExitBCFlags(isExitBC);
    pclHost->reflectingWalls(&doReflect, reflectLo, reflectHi);
    pclHost->repopulate_particlesInfo(
        &doRepopulateInjection, doRepopulateInjectionSide, repopulateBoundary);
  }
};
static_assert(std::is_trivially_copyable_v<moverParameter>,
              "a kernel parameter (KERNEL_CONSTANT) is copied bytewise");

// CUDA: at least 3 blocks of kMoverBlockSize threads per SM, which bounds the
// registers to 80 per thread (65536 per SM); the block size alone would let
// ptxas take about 160.  HIP reads the second argument as waves per
// execution unit and is not tuned here, so it gets no bound, nor do host
// compilers.
#if defined(__CUDACC__) && !defined(__HIPCC__)
#define MOVER_LAUNCH_BOUNDS __launch_bounds__(kMoverBlockSize, 3)
#else
#define MOVER_LAUNCH_BOUNDS
#endif

/**
 * @brief Advance one species with the standard predictor-corrector mover.
 *
 * @param moverParam The mover parameters of the species.
 * @param fieldForPcls Packed mover field buffer sampled from the grid.
 * @param grid The grid descriptor.
 */
__global__ void MOVER_LAUNCH_BOUNDS
moverKernel(const KERNEL_CONSTANT moverParameter moverParam,
            cudaTypeArray1<cudaFieldType> fieldForPcls,
            const KERNEL_CONSTANT grid3DCUDA grid);

/**
 * @brief Advance one species with the adaptive-subcycling mover used for planet
 * cases.
 *
 * The timestep is split into subcycles based on the local gyroperiod and the
 * velocity update is performed in relativistic form.  Parameters as
 * moverKernel.
 */
__global__ void MOVER_LAUNCH_BOUNDS
moverSubcyclesKernel(const KERNEL_CONSTANT moverParameter moverParam,
                     cudaTypeArray1<cudaFieldType> fieldForPcls,
                     const KERNEL_CONSTANT grid3DCUDA grid);

/**
 * @brief The wall pass of moverKernel: the particles it left at a reflecting
 * wall (moverParameter::wallList), pushed with the walls.  Launched over the
 * capacity of the list, after moverKernel on its stream.  Parameters as
 * moverKernel.
 */
__global__ void
moverWallsKernel(const KERNEL_CONSTANT moverParameter moverParam,
                 cudaTypeArray1<cudaFieldType> fieldForPcls,
                 const KERNEL_CONSTANT grid3DCUDA grid);

/** The wall pass of moverSubcyclesKernel (see moverWallsKernel). */
__global__ void
moverSubcyclesWallsKernel(const KERNEL_CONSTANT moverParameter moverParam,
                          cudaTypeArray1<cudaFieldType> fieldForPcls,
                          const KERNEL_CONSTANT grid3DCUDA grid);

#endif // MOVER_KERNEL_CUH
