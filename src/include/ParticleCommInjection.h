/**
 * ParticleCommInjection — MPI particle exchange and boundary injection engine.
 *
 * Owns:
 *  - 12 BlockCommunicators (6 send + 6 recv) for MPI AoS block exchange
 *  - AoS comm buffer (expandable pinned vector of SpeciesParticle) for
 *    particles that arrive from neighbour processes or are injected at boundaries
 *
 * Data flow each cycle:
 *  1. GPU compacts exiting particles into AoS → single D→H memcpy into commPcls
 *  2. separateAndSendParticles(): iterates AoS comm buffer, sends via BlockCommunicator
 *  3. recommunicateParticlesUntilDone(): iterative flush/recv/Allreduce loop
 *  4. handleReceivedParticles(): receives AoS blocks, applies BCs, appends
 *     surviving particles to AoS comm buffer
 *  5. repopulateParticlesOnlyInjection() / openBCParticlesOutflow():
 *     inject new particles into AoS comm buffer
 *  6. Caller reads commPcls for single H→D upload + scatterAoSToSoAKernel
 *
 * No persistent storage of "all" particles — only transient exchange buffer.
 */

#ifndef PARTICLE_COMM_INJECTION_H
#define PARTICLE_COMM_INJECTION_H

#include "ipicfwd.h"
#include "Particle.h"
#include "BlockCommunicator.h"
#include "aligned_vector.h"
#include "IDgenerator.h"
#include "cudaTypeDef.cuh"
#include "ipicdefs.h"
#include "ipicmath.h"
#include "ParticleSoAHost.h"

/**
 * @brief MPI particle exchange engine with expandable AoS comm buffer.
 *
 * This class does NOT own the main particle data. It owns a temporary
 * AoS buffer for particles entering/leaving the subdomain and the
 * BlockCommunicators that handle MPI messaging.
 */
class ParticleCommInjection
{
public:

  // ===== Construction / Destruction =====

  /**
   * @brief Construct from a ParticleSoAHost, borrowing its species params.
   *
   * @param hostParticles  The main particle container (borrowed reference).
   */
  ParticleCommInjection(ParticleSoAHost& hostParticles);

  ~ParticleCommInjection();

  /** Non-copyable (MPI state). */
  ParticleCommInjection(const ParticleCommInjection&) = delete;
  ParticleCommInjection& operator=(const ParticleCommInjection&) = delete;

  // ===== AoS comm buffer access =====

  /** Number of particles currently in the comm buffer. */
  int getCommNOP() const { return static_cast<int>(commPcls.size()); }

  /** Set logical size of comm buffer; only grows capacity, never shrinks
   *  (avoids expensive pinned-memory realloc every cycle). */
  void prepareCommBufferForNOP(int numParticles) {
    if (numParticles > commPcls.capacity()) {
      const int padded = roundup_to_multiple(
          static_cast<int>(numParticles * 1.5), DVECWIDTH);
      commPcls.reserve(padded);
    }
    commPcls.setSize(numParticles);
  }

  /** Clear the comm buffer (before a new cycle). Capacity is retained. */
  void clearCommBuffer() {
    commPcls.resize(0);
  }

  /** Reserve space in the comm buffer (grow-only). */
  void reserveCommBuffer(int capacity) {
    const int padded = roundup_to_multiple(capacity, DVECWIDTH);
    commPcls.reserve(padded);
  }

  // ===== AoS comm buffer pointers (for H↔D transfer) =====

  /** Read-only AoS data pointer (for H→D upload). */
  const SpeciesParticle* getCommPclsData() const { return const_cast<vector_SpeciesParticle_registered&>(commPcls).getList(); }

  /** Mutable AoS data pointer (for D→H download of exiting particles). */
  SpeciesParticle* getCommPclsDataMut() { return commPcls.getList(); }

  /** Direct access to the AoS comm buffer vector. */
  const vector_SpeciesParticle_registered& getCommPclsVec() const { return commPcls; }
  vector_SpeciesParticle_registered& getCommPclsVec() { return commPcls; }

  // ===== MPI exchange engine =====

  /**
   * @brief Iterate the comm buffer, send exiting particles to neighbours.
   * Removes sent particles from the buffer (swap-remove) and returns the
   * number of particles sent.
   *
   * @return Number of particles sent out of the local comm buffer.
   */
  int separateAndSendParticles();

  /**
   * @brief Iterative MPI exchange until all particles are in correct subdomain.
   * @param minNumIterations  Minimum number of flush/recv iterations.
   */
  void recommunicateParticlesUntilDone(int minNumIterations = 3);

  // ===== Boundary injection =====

  /** Inject new Maxwellian particles at REEMISSION boundaries (into comm buffer). */
  void repopulateParticlesOnlyInjection();

  /** Open BC: duplicate boundary particles, delete exiting ones (into comm buffer). */
  void openBCParticlesOutflow();

  // ===== Append from external AoS (CPU-side: exosphere injection) =====

  /**
   * @brief Append AoS particles to the AoS comm buffer.
   *
   * @param buffer Pointer to the input AoS particle array.
   * @param count Number of particles to append from @p buffer.
   */
  void appendFromAoS(const SpeciesParticle* buffer, int count);

  // ===== AoS comm buffer (pinned host memory) — public for direct D→H memcpy =====

  vector_SpeciesParticle_registered commPcls;

public: // BC methods (virtual for user override)
  /**
   * @brief Apply the left-X boundary condition to particles in place.
   *
   * @param pcls Particle list to modify.
   * @param start Start index of the subrange to process.
   */
  virtual void apply_Xleft_BC(vector_SpeciesParticle& pcls, int start = 0);
  /**
   * @brief Apply the left-Y boundary condition to particles in place.
   *
   * @param pcls Particle list to modify.
   * @param start Start index of the subrange to process.
   */
  virtual void apply_Yleft_BC(vector_SpeciesParticle& pcls, int start = 0);
  /**
   * @brief Apply the left-Z boundary condition to particles in place.
   *
   * @param pcls Particle list to modify.
   * @param start Start index of the subrange to process.
   */
  virtual void apply_Zleft_BC(vector_SpeciesParticle& pcls, int start = 0);
  /**
   * @brief Apply the right-X boundary condition to particles in place.
   *
   * @param pcls Particle list to modify.
   * @param start Start index of the subrange to process.
   */
  virtual void apply_Xrght_BC(vector_SpeciesParticle& pcls, int start = 0);
  /**
   * @brief Apply the right-Y boundary condition to particles in place.
   *
   * @param pcls Particle list to modify.
   * @param start Start index of the subrange to process.
   */
  virtual void apply_Yrght_BC(vector_SpeciesParticle& pcls, int start = 0);
  /**
   * @brief Apply the right-Z boundary condition to particles in place.
   *
   * @param pcls Particle list to modify.
   * @param start Start index of the subrange to process.
   */
  virtual void apply_Zrght_BC(vector_SpeciesParticle& pcls, int start = 0);

private:

  // --- Internal helpers ---
  void flushSend();
  bool sendParticleToAppropriateBuffer(SpeciesParticle& pcl, int count[6]);
  int handleReceivedParticles(int pclCommMode = 0);

  void applyPeriodicBCGlobal(vector_SpeciesParticle& pclList, int startIndex);
  void applyNonperiodicBCsGlobal(vector_SpeciesParticle& pclList, int startIndex);
  void applyBCsGlobally(vector_SpeciesParticle& pclList);
  void applyBCsLocally(vector_SpeciesParticle& pclList,
                        int direction, bool applyShift, bool doApplyBCs);

  bool testOutsideDomain(const SpeciesParticle& pcl) const;
  bool testOutsideNonperiodicDomain(const SpeciesParticle& pcl) const;
  bool testPclsAreInDomain(const vector_SpeciesParticle& pcls) const;
  bool testPclsAreInNonperiodicDomain(const vector_SpeciesParticle& pcls) const;

  bool testXleftOfDomain(const SpeciesParticle& pcl) const { return pcl.get_x() < 0.; }
  bool testXrghtOfDomain(const SpeciesParticle& pcl) const { return pcl.get_x() > domainLengthX_; }
  bool testYleftOfDomain(const SpeciesParticle& pcl) const { return pcl.get_y() < 0.; }
  bool testYrghtOfDomain(const SpeciesParticle& pcl) const { return pcl.get_y() > domainLengthY_; }
  bool testZleftOfDomain(const SpeciesParticle& pcl) const { return pcl.get_z() < 0.; }
  bool testZrghtOfDomain(const SpeciesParticle& pcl) const { return pcl.get_z() > domainLengthZ_; }

  /** Helper: push a single AoS particle into the AoS comm buffer. */
  void appendSingleParticleToComm(const SpeciesParticle& pcl) {
    commPcls.push_back(pcl);
  }

  /** Helper: populate one cell with Maxwellian particles into comm buffer. */
  void populateCellWithParticles(int cellIndexX, int cellIndexY, int cellIndexZ,
                                 double chargePerParticle,
                                 double dxPerPcl, double dyPerPcl, double dzPerPcl);

  /** Swap-remove particle at index from comm buffer. */
  void deleteCommParticle(int particleIndex) {
    const int lastIndex = getCommNOP() - 1;
    if (particleIndex != lastIndex) {
      commPcls[particleIndex] = commPcls[lastIndex];
    }
    commPcls.pop_back();
  }

  // --- Borrowed references from ParticleSoAHost ---
  ParticleSoAHost& hostParticles_;
  const CollectiveIO*      col_;
  const VirtualTopology3D* vct_;
  const Grid*              grid_;

  // --- Species params (copied from hostParticles_ for fast access) ---
  int    speciesNumber_;
  double chargeOverMass_;  // qom
  double thermalVelocityX_, thermalVelocityY_, thermalVelocityZ_;
  double driftVelocityX_, driftVelocityY_, driftVelocityZ_;

  // --- Domain geometry (copied for fast access) ---
  double domainLengthX_, domainLengthY_, domainLengthZ_;
  double gridSpacingX_, gridSpacingY_, gridSpacingZ_;
  double subdomainXstart_, subdomainXend_;
  double subdomainYstart_, subdomainYend_;
  double subdomainZstart_, subdomainZend_;
  int    numCellsX_, numCellsY_, numCellsZ_;
  double speedOfLight_;
  double timeStep_;
  double injectionDensity_;
  int    numPclPerCellX_, numPclPerCellY_, numPclPerCellZ_;
  int    numParticlesPerCell_;

  // --- BC face types ---
  int bcPfaceXleft_, bcPfaceXright_;
  int bcPfaceYleft_, bcPfaceYright_;
  int bcPfaceZleft_, bcPfaceZright_;

  // --- MPI communicator (own copy via MPI_Comm_dup) ---
  MPI_Comm mpiComm_;

  // --- BlockCommunicators for particle exchange ---
  BlockCommunicator<SpeciesParticle> sendXleft_, sendXrght_;
  BlockCommunicator<SpeciesParticle> sendYleft_, sendYrght_;
  BlockCommunicator<SpeciesParticle> sendZleft_, sendZrght_;
  BlockCommunicator<SpeciesParticle> recvXleft_, recvXrght_;
  BlockCommunicator<SpeciesParticle> recvYleft_, recvYrght_;
  BlockCommunicator<SpeciesParticle> recvZleft_, recvZrght_;

  // --- Particle-ID generator (for injected particles) ---
  doubleIDgenerator particleIDGenerator_;

  bool cVERBOSE_;
};

#endif // PARTICLE_COMM_INJECTION_H
