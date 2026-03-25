/**
 * ParticleCommInjection — MPI particle exchange and boundary injection engine.
 *
 * Owns:
 *  - 12 BlockCommunicators (6 send + 6 recv) for MPI AoS block exchange
 *  - SoA comm buffer (expandable pinned vectors) for particles that arrive
 *    from neighbour processes or are injected at boundaries
 *
 * Data flow each cycle:
 *  1. GPU compacts exiting particles into SoA → D→H into commU/V/W/Q/X/Y/Z/T
 *  2. separateAndSendParticles(): iterates SoA comm buffer, converts each
 *     particle on-the-fly to AoS SpeciesParticle, sends via BlockCommunicator
 *  3. recommunicateParticlesUntilDone(): iterative flush/recv/Allreduce loop
 *  4. handleReceivedParticles(): receives AoS blocks, applies BCs, scatters
 *     surviving particles into SoA comm buffer
 *  5. repopulateParticlesOnlyInjection() / openBCParticlesOutflow():
 *     inject new particles into SoA comm buffer
 *  6. Caller reads commU/V/W/Q/X/Y/Z/T for H→D upload
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
 * @brief MPI particle exchange engine with expandable SoA comm buffer.
 *
 * This class does NOT own the main particle data. It owns a temporary
 * SoA buffer for particles entering/leaving the subdomain and the
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

  // ===== SoA comm buffer access =====

  /** Number of particles currently in the comm buffer. */
  int getCommNOP() const { return static_cast<int>(commU.size()); }

  /** Resize all comm SoA vectors to hold exactly @p numParticles. */
  void prepareCommBufferForNOP(int numParticles) {
    const int padded = roundup_to_multiple(numParticles, DVECWIDTH);
    commU.reserve(padded); commV.reserve(padded); commW.reserve(padded);
    commQ.reserve(padded); commX.reserve(padded); commY.reserve(padded);
    commZ.reserve(padded); commT.reserve(padded);
    commU.resize(numParticles); commV.resize(numParticles); commW.resize(numParticles);
    commQ.resize(numParticles); commX.resize(numParticles); commY.resize(numParticles);
    commZ.resize(numParticles); commT.resize(numParticles);
  }

  /** Clear the comm buffer (before a new cycle). */
  void clearCommBuffer() {
    commU.resize(0); commV.resize(0); commW.resize(0); commQ.resize(0);
    commX.resize(0); commY.resize(0); commZ.resize(0); commT.resize(0);
  }

  /** Reserve space in the comm buffer. */
  void reserveCommBuffer(int capacity) {
    const int padded = roundup_to_multiple(capacity, DVECWIDTH);
    commU.reserve(padded); commV.reserve(padded); commW.reserve(padded);
    commQ.reserve(padded); commX.reserve(padded); commY.reserve(padded);
    commZ.reserve(padded); commT.reserve(padded);
  }

  // ===== Read-only SoA comm buffer pointers (for H→D upload) =====

  const double* getCommUall() const { return &commU[0]; }
  const double* getCommVall() const { return &commV[0]; }
  const double* getCommWall() const { return &commW[0]; }
  const double* getCommQall() const { return &commQ[0]; }
  const double* getCommXall() const { return &commX[0]; }
  const double* getCommYall() const { return &commY[0]; }
  const double* getCommZall() const { return &commZ[0]; }
  const double* getCommTall() const { return &commT[0]; }

  // ===== Mutable SoA comm buffer pointers (for D→H of exiting particles) =====

  double* getCommUallMut() { return &commU[0]; }
  double* getCommVallMut() { return &commV[0]; }
  double* getCommWallMut() { return &commW[0]; }
  double* getCommQallMut() { return &commQ[0]; }
  double* getCommXallMut() { return &commX[0]; }
  double* getCommYallMut() { return &commY[0]; }
  double* getCommZallMut() { return &commZ[0]; }
  double* getCommTallMut() { return &commT[0]; }

  // ===== MPI exchange engine =====

  /**
   * @brief Iterate the comm buffer, send exiting particles to neighbours.
   * Removes sent particles from the buffer (swap-remove) and returns the
   * number of particles sent.
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

  /** Scatter AoS particles into the SoA comm buffer. */
  void appendFromAoS(const SpeciesParticle* buffer, int count);

  // ===== SoA comm buffer arrays (pinned host memory) — public for direct D→H memcpy =====

  vector_cudaParticleType_registered commU, commV, commW, commQ;
  vector_cudaParticleType_registered commX, commY, commZ, commT;

public: // BC methods (virtual for user override)
  virtual void apply_Xleft_BC(vector_SpeciesParticle& pcls, int start = 0);
  virtual void apply_Yleft_BC(vector_SpeciesParticle& pcls, int start = 0);
  virtual void apply_Zleft_BC(vector_SpeciesParticle& pcls, int start = 0);
  virtual void apply_Xrght_BC(vector_SpeciesParticle& pcls, int start = 0);
  virtual void apply_Yrght_BC(vector_SpeciesParticle& pcls, int start = 0);
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

  /** Helper: push a single particle from AoS into the SoA comm buffer. */
  void appendSingleParticleToComm(const SpeciesParticle& pcl) {
    commU.push_back(pcl.get_u()); commV.push_back(pcl.get_v());
    commW.push_back(pcl.get_w()); commQ.push_back(pcl.get_q());
    commX.push_back(pcl.get_x()); commY.push_back(pcl.get_y());
    commZ.push_back(pcl.get_z()); commT.push_back(pcl.get_t());
  }

  /** Helper: populate one cell with Maxwellian particles into comm buffer. */
  void populateCellWithParticles(int cellIndexX, int cellIndexY, int cellIndexZ,
                                 double chargePerParticle,
                                 double dxPerPcl, double dyPerPcl, double dzPerPcl);

  /** Swap-remove particle at index from comm buffer. */
  void deleteCommParticle(int particleIndex) {
    const int lastIndex = getCommNOP() - 1;
    if (particleIndex != lastIndex) {
      commU[particleIndex] = commU[lastIndex]; commV[particleIndex] = commV[lastIndex];
      commW[particleIndex] = commW[lastIndex]; commQ[particleIndex] = commQ[lastIndex];
      commX[particleIndex] = commX[lastIndex]; commY[particleIndex] = commY[lastIndex];
      commZ[particleIndex] = commZ[lastIndex]; commT[particleIndex] = commT[lastIndex];
    }
    commU.pop_back(); commV.pop_back(); commW.pop_back(); commQ.pop_back();
    commX.pop_back(); commY.pop_back(); commZ.pop_back(); commT.pop_back();
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
