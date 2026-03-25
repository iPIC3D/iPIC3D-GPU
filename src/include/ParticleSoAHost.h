/**
 * ParticleSoAHost — Unified SoA host container for particle data.
 *
 * This is the single authoritative host-side particle storage class.
 * Holds 8 pinned-memory SoA arrays (pclVelX, pclVelY, pclVelZ, pclCharge,
 * pclPosX, pclPosY, pclPosZ, pclID) plus all species metadata extracted
 * from Collective, Grid, and VCtopology3D.
 *
 * Provides:
 *  - Particle generation (maxwellian, restart, pitch_angle_energy, etc.)
 *  - Diagnostics (getKe, getP, getTotalQ, getMaxVelocity, getVelocityDistribution)
 *  - Cell-sorted reorder (serial + parallel)
 *  - BC configuration queries (for GPU kernel setup)
 *  - Accessor interface for IO backends (PSKOutput, ParallelIO, ADIOS2IO)
 *  - Direct target for cudaMemcpyAsync D→H / H→D (pinned memory)
 *
 * No AoS storage, no BlockCommunicator, no MPI_Comm_dup.
 * Communication is handled by the companion ParticleCommInjection class.
 *
 * Memory saved per species vs. old Particles3Dcomm:
 *   12 BlockCommunicators × 4 blocks × 8192 × 64 B  ≈ 24 MiB
 *   + MPI_Comm_dup + 6 persistent MPI_Recv_init
 *   + AoS duplicate _pcls buffer (sizeof(NOP) × 64 B)
 */

#ifndef PARTICLE_SOA_HOST_H
#define PARTICLE_SOA_HOST_H

#include <cmath>
#include <cstring>
#include <cassert>
#include <vector>
#include <mpi.h>

#include "aligned_vector.h"  // vector_cudaParticleType_registered (pinned Larray)
#include "ipicdefs.h"        // DVECWIDTH
#include "ipicmath.h"        // roundup_to_multiple, sample_maxwellian
#include "IDgenerator.h"     // doubleIDgenerator
#include "ipicfwd.h"         // Grid, Field, CollectiveIO, VirtualTopology3D
#include "cudaTypeDef.cuh"   // cudaParticleType, cudaCommonType
#include "Alloc.h"           // array3_int

// Forward declarations
class moverParameter;

namespace BCparticles
{
    enum Enum
    {
        EXIT = 0,
        PERFECT_MIRROR = 1,
        REEMISSION = 2,
        OPENBCOut = 3,
        OPENBCIn = 4
    };
}

/**
 * @brief Pinned-memory SoA host container for all particle data on a single
 *        MPI rank, for one species.
 *
 * Stores 8 SoA arrays in CUDA-registered (pinned) host memory.
 * Contains all species parameters needed for particle generation,
 * GPU mover configuration, diagnostics, and IO.
 *
 * The borrowed MPI communicator is only used for MPI_Allreduce diagnostics.
 */
class ParticleSoAHost
{
  friend class moverParameter;

public:

  // ===== Construction / Destruction =====

  ParticleSoAHost() = default;

  /**
   * @brief Construct a container for the given species, extracting all
   *        parameters from Collective, Grid and VCtopology3D.
   *
   * @param speciesNum  Species index (0-based, includes test particles).
   * @param col         Collective input parameters (borrowed, NOT owned).
   * @param vct         MPI topology (borrowed, NOT owned).
   * @param grid        Grid pointer (borrowed, NOT owned).
   */
  ParticleSoAHost(int speciesNum, CollectiveIO* col,
                  VirtualTopology3D* vct, Grid* grid);

  ~ParticleSoAHost();

  /** Non-copyable (pinned memory). */
  ParticleSoAHost(const ParticleSoAHost&) = delete;
  ParticleSoAHost& operator=(const ParticleSoAHost&) = delete;

  /** Movable. */
  ParticleSoAHost(ParticleSoAHost&&) = default;
  ParticleSoAHost& operator=(ParticleSoAHost&&) = default;

  // ===== Particle count =====

  int getNOP() const { return static_cast<int>(u.size()); }

  // ===== Read-only SoA bulk accessors =====

  const double* getUall()          const { return &u[0]; }
  const double* getVall()          const { return &v[0]; }
  const double* getWall()          const { return &w[0]; }
  const double* getQall()          const { return &q[0]; }
  const double* getXall()          const { return &x[0]; }
  const double* getYall()          const { return &y[0]; }
  const double* getZall()          const { return &z[0]; }
  const double* getParticleIDall() const { return &t[0]; }

  // ===== Mutable SoA bulk pointers (targets for cudaMemcpyAsync D->H) =====

  double* getUallMut() { return &u[0]; }
  double* getVallMut() { return &v[0]; }
  double* getWallMut() { return &w[0]; }
  double* getQallMut() { return &q[0]; }
  double* getXallMut() { return &x[0]; }
  double* getYallMut() { return &y[0]; }
  double* getZallMut() { return &z[0]; }
  double* getTallMut() { return &t[0]; }

  // ===== Per-element accessors (used by PSKOutput / IO backends) =====

  double getU(int index) const { return u[index]; }
  double getV(int index) const { return v[index]; }
  double getW(int index) const { return w[index]; }
  double getQ(int index) const { return q[index]; }
  double getX(int index) const { return x[index]; }
  double getY(int index) const { return y[index]; }
  double getZ(int index) const { return z[index]; }
  double getT(int index) const { return t[index]; }

  // ===== Species metadata =====

  int    get_species_num() const { return speciesNumber_; }
  double getQOM()          const { return chargeOverMass_; }

  // ===== Grid / domain geometry accessors =====

  double get_dx()     const { return gridSpacingX_; }
  double get_dy()     const { return gridSpacingY_; }
  double get_dz()     const { return gridSpacingZ_; }
  double get_invdx()  const { return invGridSpacingX_; }
  double get_invdy()  const { return invGridSpacingY_; }
  double get_invdz()  const { return invGridSpacingZ_; }
  double get_xstart() const { return subdomainXstart_; }
  double get_ystart() const { return subdomainYstart_; }
  double get_zstart() const { return subdomainZstart_; }

  // ===== Resize / Prepare =====

  /** Resize all SoA vectors to hold exactly @p numParticles.
   *  Capacity is padded to DVECWIDTH alignment. */
  void prepareSoAForNOP(int numParticles) {
    const int padded = roundup_to_multiple(numParticles, DVECWIDTH);
    u.reserve(padded); v.reserve(padded); w.reserve(padded); q.reserve(padded);
    x.reserve(padded); y.reserve(padded); z.reserve(padded); t.reserve(padded);
    u.resize(numParticles); v.resize(numParticles); w.resize(numParticles); q.resize(numParticles);
    x.resize(numParticles); y.resize(numParticles); z.resize(numParticles); t.resize(numParticles);
  }

  void reserveSpace(int numParticles) {
    const int padded = roundup_to_multiple(numParticles, DVECWIDTH);
    u.reserve(padded); v.reserve(padded); w.reserve(padded); q.reserve(padded);
    x.reserve(padded); y.reserve(padded); z.reserve(padded); t.reserve(padded);
  }

  void clearParticles() {
    u.resize(0); v.resize(0); w.resize(0); q.resize(0);
    x.resize(0); y.resize(0); z.resize(0); t.resize(0);
  }

  /** Pad capacities to DVECWIDTH for vectorized access. */
  void padCapacities() {
    u.reserve(roundup_to_multiple(u.size(), DVECWIDTH));
    v.reserve(roundup_to_multiple(v.size(), DVECWIDTH));
    w.reserve(roundup_to_multiple(w.size(), DVECWIDTH));
    q.reserve(roundup_to_multiple(q.size(), DVECWIDTH));
    x.reserve(roundup_to_multiple(x.size(), DVECWIDTH));
    y.reserve(roundup_to_multiple(y.size(), DVECWIDTH));
    z.reserve(roundup_to_multiple(z.size(), DVECWIDTH));
    t.reserve(roundup_to_multiple(t.size(), DVECWIDTH));
  }

  // ===== Particle creation =====

  /** Push a single particle into SoA storage with auto-generated ID. */
  void create_new_particle(
    double velocityX, double velocityY, double velocityZ, double charge,
    double positionX, double positionY, double positionZ)
  {
    const double particleID = particleIDGenerator_.generateID();
    u.push_back(velocityX); v.push_back(velocityY); w.push_back(velocityZ);
    q.push_back(charge);
    x.push_back(positionX); y.push_back(positionY); z.push_back(positionZ);
    t.push_back(particleID);
  }

  /** Push a single particle with an explicit ID. */
  void add_new_particle(
    double velocityX, double velocityY, double velocityZ, double charge,
    double positionX, double positionY, double positionZ, double particleID)
  {
    u.push_back(velocityX); v.push_back(velocityY); w.push_back(velocityZ);
    q.push_back(charge);
    x.push_back(positionX); y.push_back(positionY); z.push_back(positionZ);
    t.push_back(particleID);
  }

  /** Swap-remove particle at index. O(1). */
  void delete_particle(int particleIndex)
  {
    const int lastIndex = getNOP() - 1;
    if (particleIndex != lastIndex) {
      u[particleIndex] = u[lastIndex]; v[particleIndex] = v[lastIndex];
      w[particleIndex] = w[lastIndex]; q[particleIndex] = q[lastIndex];
      x[particleIndex] = x[lastIndex]; y[particleIndex] = y[lastIndex];
      z[particleIndex] = z[lastIndex]; t[particleIndex] = t[lastIndex];
    }
    u.pop_back(); v.pop_back(); w.pop_back(); q.pop_back();
    x.pop_back(); y.pop_back(); z.pop_back(); t.pop_back();
  }

  /** Reserve remaining particle-ID space (call after initial fill). */
  void reserve_remaining_particle_IDs() {
    particleIDGenerator_.reserve_particles_in_range(getNOP());
  }

  // ===== Particle initialisation (implemented in ParticleSoAHost.cpp) =====

  void maxwellian(Field* EMf);
  void maxwellianNullPoints(Field* EMf);
  void maxwellianDoubleHarris(Field* EMf);
  void maxwellianHumpPerturbation(Field* EMf);
  void pitch_angle_energy(Field* EMf);
  void force_free(Field* EMf);
  void restartLoad();

  // ===== Diagnostics (MPI reductions via borrowed communicator) =====

  /** Total kinetic energy (MPI-reduced across all ranks). */
  double getKe() const {
    double localKe = 0.0;
    const int numParticles = getNOP();
    #pragma omp parallel for reduction(+:localKe)
    for (int idx = 0; idx < numParticles; idx++) {
      const double velX = u[idx], velY = v[idx], velZ = w[idx];
      const double charge = q[idx];
      localKe += 0.5 * (charge / chargeOverMass_) * (velX*velX + velY*velY + velZ*velZ);
    }
    double totalKe = 0.0;
    MPI_Allreduce(&localKe, &totalKe, 1, MPI_DOUBLE, MPI_SUM, mpiComm_);
    return totalKe;
  }

  /** Total momentum magnitude (MPI-reduced across all ranks). */
  double getP() const {
    double localP = 0.0;
    const int numParticles = getNOP();
    #pragma omp parallel for reduction(+:localP)
    for (int idx = 0; idx < numParticles; idx++) {
      const double velX = u[idx], velY = v[idx], velZ = w[idx];
      const double charge = q[idx];
      localP += (charge / chargeOverMass_) * std::sqrt(velX*velX + velY*velY + velZ*velZ);
    }
    double totalP = 0.0;
    MPI_Allreduce(&localP, &totalP, 1, MPI_DOUBLE, MPI_SUM, mpiComm_);
    return totalP;
  }

  /** Total charge (MPI-reduced across all ranks). */
  double getTotalQ() const {
    double localQ = 0.0;
    const int numParticles = getNOP();
    #pragma omp parallel for reduction(+:localQ)
    for (int idx = 0; idx < numParticles; idx++)
      localQ += q[idx];
    double totalQ = 0.0;
    MPI_Allreduce(&localQ, &totalQ, 1, MPI_DOUBLE, MPI_SUM, mpiComm_);
    return totalQ;
  }

  /** Maximum velocity magnitude (MPI-reduced across all ranks). */
  double getMaxVelocity() const;

  /** Velocity distribution histogram (MPI-reduced). Caller owns result. */
  long long* getVelocityDistribution(int numBins, double maxVelocity) const;

  // ===== Cell-sorted reorder (implemented in ParticleSoAHost.cpp) =====

  void sort_particles_serial();
  void sort_particles_parallel(int* cellCount, int* cellOffset);

  // ===== BC configuration queries (for GPU kernel setup) =====

  /** Fill repopulate-injection info for GPU kernel configuration. */
  void repopulate_particlesInfo(bool* doRepopulateInjection,
                                bool* doRepopulateInjectionSide,
                                cudaCommonType* repopulateBoundary) const;

  /** Fill open-BC outflow info for GPU kernel configuration. */
  void openbc_particles_outflowInfo(bool* doOpenBC, bool* applyOpenBC,
                                    cudaCommonType* deleteBoundary,
                                    cudaCommonType* openBoundary) const;

  /** Fill per-face EXIT BC flags for GPU kernel. */
  void fillExitBCFlags(bool* isExitBC) const;

  // ===== BC face config (read-only) =====

  int getBcPfaceXleft()  const { return bcPfaceXleft_; }
  int getBcPfaceXright() const { return bcPfaceXright_; }
  int getBcPfaceYleft()  const { return bcPfaceYleft_; }
  int getBcPfaceYright() const { return bcPfaceYright_; }
  int getBcPfaceZleft()  const { return bcPfaceZleft_; }
  int getBcPfaceZright() const { return bcPfaceZright_; }

  // ===== Append from external AoS (for exosphere / planet injection) =====

  /** Scatter AoS particles into SoA arrays. Expandable buffer. */
  void appendFromAoS(const SpeciesParticle* buffer, int count);

  // ===== Raw borrowed-pointer accessors (for ParticleCommInjection etc.) =====

  const CollectiveIO*       getCollective()     const { return col_; }
  const VirtualTopology3D*  getVirtualTopology() const { return vct_; }
  const Grid*               getGrid()           const { return grid_; }
  MPI_Comm                  getMpiComm()        const { return mpiComm_; }

  // ===== SoA data arrays (pinned host memory) — public for direct D→H memcpy =====

  vector_cudaParticleType_registered u, v, w, q, x, y, z, t;

private:

  // ===== Helper: fill one cell with Maxwellian particles =====
  void populateCellWithParticles(int cellIndexX, int cellIndexY, int cellIndexZ,
                                 double chargePerParticle,
                                 double dxPerPcl, double dyPerPcl, double dzPerPcl);

  // ===== Species identity =====
  int        speciesNumber_     = 0;
  double     chargeOverMass_    = 0.0;     // qom
  bool       isTestParticle_    = false;

  // ===== Borrowed references (NOT owned, NOT freed) =====
  const CollectiveIO*       col_  = nullptr;
  const VirtualTopology3D*  vct_  = nullptr;
  const Grid*               grid_ = nullptr;
  MPI_Comm                  mpiComm_ = MPI_COMM_NULL;

  // ===== Species parameters (extracted from Collective in constructor) =====
  int    numParticlesPerCell_  = 0;   // npcel
  int    numPclPerCellX_       = 0;   // npcelx
  int    numPclPerCellY_       = 0;   // npcely
  int    numPclPerCellZ_       = 0;   // npcelz
  double thermalVelocityX_     = 0.0; // uth
  double thermalVelocityY_     = 0.0; // vth
  double thermalVelocityZ_     = 0.0; // wth
  double driftVelocityX_       = 0.0; // u0
  double driftVelocityY_       = 0.0; // v0
  double driftVelocityZ_       = 0.0; // w0

  // ===== Test particle parameters =====
  double pitchAngle_           = 0.0;
  double energy_               = 0.0;

  // ===== Grid / domain geometry =====
  double gridSpacingX_         = 0.0; // dx
  double gridSpacingY_         = 0.0; // dy
  double gridSpacingZ_         = 0.0; // dz
  double invGridSpacingX_      = 0.0; // 1/dx
  double invGridSpacingY_      = 0.0; // 1/dy
  double invGridSpacingZ_      = 0.0; // 1/dz
  double domainLengthX_        = 0.0; // Lx
  double domainLengthY_        = 0.0; // Ly
  double domainLengthZ_        = 0.0; // Lz
  double subdomainXstart_      = 0.0;
  double subdomainXend_        = 0.0;
  double subdomainYstart_      = 0.0;
  double subdomainYend_        = 0.0;
  double subdomainZstart_      = 0.0;
  double subdomainZend_        = 0.0;
  double inverseVolume_        = 0.0; // invVOL
  int    numCellsX_            = 0;   // nxc
  int    numCellsY_            = 0;   // nyc
  int    numCellsZ_            = 0;   // nzc
  int    numNodesX_            = 0;   // nxn
  int    numNodesY_            = 0;   // nyn
  int    numNodesZ_            = 0;   // nzn

  // ===== Physics / mover parameters (read by moverParameter via friend) =====
  double timeStep_             = 0.0; // dt
  double speedOfLight_         = 0.0; // c
  int    numMoverIterations_   = 0;   // NiterMover
  double reconThickness_       = 0.0; // delta

  // ===== Velocity caps =====
  double velocityCapMaxX_      = 0.0; // umax
  double velocityCapMaxY_      = 0.0; // vmax
  double velocityCapMaxZ_      = 0.0; // wmax
  double velocityCapMinX_      = 0.0; // umin
  double velocityCapMinY_      = 0.0; // vmin
  double velocityCapMinZ_      = 0.0; // wmin

  // ===== Injection / open BC parameters =====
  double injectionVelocity_    = 0.0; // Vinj
  double injectionDensity_     = 0.0; // Ninj

  // ===== Boundary condition face types =====
  int bcPfaceXleft_            = 0;
  int bcPfaceXright_           = 0;
  int bcPfaceYleft_            = 0;
  int bcPfaceYright_           = 0;
  int bcPfaceZleft_            = 0;
  int bcPfaceZright_           = 0;

  // ===== Sorting arrays =====
  array3_int* numParticlesInBucket_    = nullptr;
  array3_int* numParticlesInBucketNow_ = nullptr;
  array3_int* bucketOffset_            = nullptr;

  // ===== Unique particle-ID generator =====
  doubleIDgenerator particleIDGenerator_;
};

// Convenience typedef used by PSKOutput and legacy IO
typedef ParticleSoAHost Particles;

#endif // PARTICLE_SOA_HOST_H
