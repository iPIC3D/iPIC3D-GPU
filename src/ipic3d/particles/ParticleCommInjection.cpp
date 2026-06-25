/**
 * ParticleCommInjection.cpp — MPI particle exchange engine implementation.
 *
 * Adapts the communication, BC, and injection methods from the former
 * Particles3Dcomm + Particles3D into a standalone comm engine that
 * operates on an AoS comm buffer and AoS BlockCommunicator blocks.
 */

#include "ParticleCommInjection.h"
#include "ParticleSoAHost.h"
#include "Collective.h"
#include "VCtopology3D.h"
#include "Grid3DCU.h"
#include "MPIdata.h"
#include "asserts.h"
#include "Particle.h"
#include "Parameters.h"
#include "parallel.h"
#include "debug.h"

#include <mpi.h>
#include <iostream>
#include <cmath>
#include <algorithm>
#include <cstring>
#include <random>

using std::cout;
using std::endl;

static bool print_pcl_comm_counts = false;

// sort_pcls: macro to put all particles that satisfy a condition
// at the end of an array of given size.
#define sort_pcls(pcls, start_in, start_out, condition) \
{ \
  int start = (start_in); \
  assert(0<=start); \
  start_out = pcls.size(); \
  for(int pidx=pcls.size()-1;pidx>=start;pidx--) \
  { \
    assert(pidx<start_out); \
    if(condition(pcls[pidx])) \
    { \
      --start_out; \
      SpeciesParticle tmp_pcl = pcls[pidx]; \
      pcls[pidx] = pcls[start_out]; \
      pcls[start_out] = tmp_pcl; \
    } \
  } \
}

// ========================================================================
// Constructor / Destructor
// ========================================================================

ParticleCommInjection::ParticleCommInjection(ParticleSoAHost& hostParticles)
  : hostParticles_(hostParticles),
    col_(hostParticles.getCollective()),
    vct_(hostParticles.getVirtualTopology()),
    grid_(hostParticles.getGrid())
{
  // --- MPI communicator (own copy) ---
  MPI_Comm_dup(vct_->getParticleComm(), &mpiComm_);

  // --- BlockCommunicator setup ---
  using namespace Direction;
  sendXleft_.init(Connection::null2self(vct_->getXleft_neighbor_P(), XDN, XDN, mpiComm_));
  sendXrght_.init(Connection::null2self(vct_->getXright_neighbor_P(), XUP, XUP, mpiComm_));
  recvXleft_.init(Connection::null2self(vct_->getXleft_neighbor_P(), XUP, XDN, mpiComm_));
  recvXrght_.init(Connection::null2self(vct_->getXright_neighbor_P(), XDN, XUP, mpiComm_));

  sendYleft_.init(Connection::null2self(vct_->getYleft_neighbor_P(), YDN, YDN, mpiComm_));
  sendYrght_.init(Connection::null2self(vct_->getYright_neighbor_P(), YUP, YUP, mpiComm_));
  recvYleft_.init(Connection::null2self(vct_->getYleft_neighbor_P(), YUP, YDN, mpiComm_));
  recvYrght_.init(Connection::null2self(vct_->getYright_neighbor_P(), YDN, YUP, mpiComm_));

  sendZleft_.init(Connection::null2self(vct_->getZleft_neighbor_P(), ZDN, ZDN, mpiComm_));
  sendZrght_.init(Connection::null2self(vct_->getZright_neighbor_P(), ZUP, ZUP, mpiComm_));
  recvZleft_.init(Connection::null2self(vct_->getZleft_neighbor_P(), ZUP, ZDN, mpiComm_));
  recvZrght_.init(Connection::null2self(vct_->getZright_neighbor_P(), ZDN, ZUP, mpiComm_));

  recvXleft_.post_recvs();
  recvXrght_.post_recvs();
  recvYleft_.post_recvs();
  recvYrght_.post_recvs();
  recvZleft_.post_recvs();
  recvZrght_.post_recvs();

  // --- Copy species params from hostParticles for fast access ---
  speciesNumber_     = hostParticles.get_species_num();
  chargeOverMass_    = hostParticles.getQOM();

  thermalVelocityX_  = col_->getUth(speciesNumber_);
  thermalVelocityY_  = col_->getVth(speciesNumber_);
  thermalVelocityZ_  = col_->getWth(speciesNumber_);
  driftVelocityX_    = col_->getU0(speciesNumber_);
  driftVelocityY_    = col_->getV0(speciesNumber_);
  driftVelocityZ_    = col_->getW0(speciesNumber_);

  // --- Domain geometry ---
  domainLengthX_ = col_->getLx();
  domainLengthY_ = col_->getLy();
  domainLengthZ_ = col_->getLz();
  gridSpacingX_  = grid_->getDX();
  gridSpacingY_  = grid_->getDY();
  gridSpacingZ_  = grid_->getDZ();
  subdomainXstart_ = grid_->getXstart();
  subdomainXend_   = grid_->getXend();
  subdomainYstart_ = grid_->getYstart();
  subdomainYend_   = grid_->getYend();
  subdomainZstart_ = grid_->getZstart();
  subdomainZend_   = grid_->getZend();
  numCellsX_ = grid_->getNXC();
  numCellsY_ = grid_->getNYC();
  numCellsZ_ = grid_->getNZC();
  speedOfLight_ = col_->getC();

  // --- Injection density ---
  if (speciesNumber_ < col_->getNs()) {
    injectionDensity_ = col_->getRHOinject(speciesNumber_);
  } else {
    injectionDensity_ = 0.0;
  }
  numPclPerCellX_    = col_->getNpcelx(speciesNumber_);
  numPclPerCellY_    = col_->getNpcely(speciesNumber_);
  numPclPerCellZ_    = col_->getNpcelz(speciesNumber_);
  numParticlesPerCell_ = col_->getNpcel(speciesNumber_);

  // --- BC face types ---
  bcPfaceXleft_  = col_->getBcPfaceXleft();
  bcPfaceXright_ = col_->getBcPfaceXright();
  bcPfaceYleft_  = col_->getBcPfaceYleft();
  bcPfaceYright_ = col_->getBcPfaceYright();
  bcPfaceZleft_  = col_->getBcPfaceZleft();
  bcPfaceZright_ = col_->getBcPfaceZright();

  cVERBOSE_ = vct_->getcVERBOSE();

  // Seed BC reemission RNG (single-thread, few particles per cycle)
  bcRng_.seed(static_cast<uint64_t>(MPIdata::get_rank()) * 31 +
              static_cast<uint64_t>(speciesNumber_) * 127 + 9973);

  // Compute injection count once (depends only on grid, BC, topology — all constant).
  cachedInjectionCount_ = computeInjectionCountImpl();
}

ParticleCommInjection::~ParticleCommInjection()
{
  MPI_Comm_free(&mpiComm_);
}

// ========================================================================
// Internal helpers
// ========================================================================

inline bool ParticleCommInjection::sendParticleToAppropriateBuffer(
  SpeciesParticle& pcl, int count[6])
{
  bool wasSent = true;
  if      (pcl.get_x() < subdomainXstart_) { sendXleft_.send(pcl); count[0]++; }
  else if (pcl.get_x() > subdomainXend_)   { sendXrght_.send(pcl); count[1]++; }
  else if (pcl.get_y() < subdomainYstart_) { sendYleft_.send(pcl); count[2]++; }
  else if (pcl.get_y() > subdomainYend_)   { sendYrght_.send(pcl); count[3]++; }
  else if (pcl.get_z() < subdomainZstart_) { sendZleft_.send(pcl); count[4]++; }
  else if (pcl.get_z() > subdomainZend_)   { sendZrght_.send(pcl); count[5]++; }
  else wasSent = false;
  return wasSent;
}

void ParticleCommInjection::flushSend()
{
  sendXleft_.send_complete();
  sendXrght_.send_complete();
  sendYleft_.send_complete();
  sendYrght_.send_complete();
  sendZleft_.send_complete();
  sendZrght_.send_complete();
}

// ========================================================================
// Boundary condition test helpers
// ========================================================================

bool ParticleCommInjection::testOutsideDomain(const SpeciesParticle& pcl) const
{
  return pcl.get_x() < 0. || pcl.get_y() < 0. || pcl.get_z() < 0.
      || pcl.get_x() > domainLengthX_ || pcl.get_y() > domainLengthY_ || pcl.get_z() > domainLengthZ_;
}

bool ParticleCommInjection::testOutsideNonperiodicDomain(const SpeciesParticle& pcl) const
{
  return (!vct_->getPERIODICX_P() && (pcl.get_x() < 0. || pcl.get_x() > domainLengthX_))
      || (!vct_->getPERIODICY_P() && (pcl.get_y() < 0. || pcl.get_y() > domainLengthY_))
      || (!vct_->getPERIODICZ_P() && (pcl.get_z() < 0. || pcl.get_z() > domainLengthZ_));
}

bool ParticleCommInjection::testPclsAreInDomain(const vector_SpeciesParticle& pcls) const
{
  for (int pidx = 0; pidx < (int)pcls.size(); pidx++) {
    if (__builtin_expect(testOutsideDomain(pcls[pidx]), false)) return false;
  }
  return true;
}

bool ParticleCommInjection::testPclsAreInNonperiodicDomain(const vector_SpeciesParticle& pcls) const
{
  for (int pidx = 0; pidx < (int)pcls.size(); pidx++) {
    if (__builtin_expect(testOutsideNonperiodicDomain(pcls[pidx]), false)) return false;
  }
  return true;
}

// ========================================================================
// Periodic BC global: apply modulo to coordinates
// ========================================================================

void ParticleCommInjection::applyPeriodicBCGlobal(
  vector_SpeciesParticle& pclList, int startIndex)
{
  const double Lxinv = 1.0 / domainLengthX_;
  const double Lyinv = 1.0 / domainLengthY_;
  const double Lzinv = 1.0 / domainLengthZ_;
  for (int pidx = startIndex; pidx < (int)pclList.size(); pidx++) {
    SpeciesParticle& pcl = pclList[pidx];
    if (vct_->getPERIODICX_P()) {
      cudaParticleType& posX = pcl.fetch_x();
      posX = modulo(posX, domainLengthX_, Lxinv);
    }
    if (vct_->getPERIODICY_P()) {
      cudaParticleType& posY = pcl.fetch_y();
      posY = modulo(posY, domainLengthY_, Lyinv);
    }
    if (vct_->getPERIODICZ_P()) {
      cudaParticleType& posZ = pcl.fetch_z();
      posZ = modulo(posZ, domainLengthZ_, Lzinv);
    }
  }
}

// ========================================================================
// Non-periodic BC global: apply face BCs to particles outside domain
// ========================================================================

void ParticleCommInjection::applyNonperiodicBCsGlobal(
  vector_SpeciesParticle& pclList, int startIndex)
{
  int lstart;
  if (!vct_->getPERIODICX_P()) {
    sort_pcls(pclList, startIndex, lstart, testXleftOfDomain);
    apply_Xleft_BC(pclList, lstart);
    sort_pcls(pclList, startIndex, lstart, testXrghtOfDomain);
    apply_Xrght_BC(pclList, lstart);
  }
  if (!vct_->getPERIODICY_P()) {
    sort_pcls(pclList, startIndex, lstart, testYleftOfDomain);
    apply_Yleft_BC(pclList, lstart);
    sort_pcls(pclList, startIndex, lstart, testYrghtOfDomain);
    apply_Yrght_BC(pclList, lstart);
  }
  if (!vct_->getPERIODICZ_P()) {
    sort_pcls(pclList, startIndex, lstart, testZleftOfDomain);
    apply_Zleft_BC(pclList, lstart);
    sort_pcls(pclList, startIndex, lstart, testZrghtOfDomain);
    apply_Zrght_BC(pclList, lstart);
  }
}

// ========================================================================
// apply_BCs_globally: iteratively apply BCs until all particles inside domain
// ========================================================================

static bool do_apply_periodic_BC_global = false;

void ParticleCommInjection::applyBCsGlobally(vector_SpeciesParticle& pclList)
{
  int pstart = 0;
  sort_pcls(pclList, 0, pstart, testOutsideDomain);

  for (int iteration = 0; pstart < (int)pclList.size(); iteration++) {
    if (do_apply_periodic_BC_global) {
      applyPeriodicBCGlobal(pclList, pstart);
      applyNonperiodicBCsGlobal(pclList, pstart);
      sort_pcls(pclList, pstart, pstart, testOutsideDomain);
    } else {
      applyNonperiodicBCsGlobal(pclList, pstart);
      sort_pcls(pclList, pstart, pstart, testOutsideNonperiodicDomain);
    }
    if (iteration >= 100) {
      dprintf("WARNING: applyBCsGlobally removing %d unrecoverable particles (species %d)",
              (int)pclList.size() - pstart, speciesNumber_);
      pclList.resize(pstart);
      break;
    }
  }
  if (do_apply_periodic_BC_global) {
    assert(testPclsAreInDomain(pclList));
  } else {
    assert(testPclsAreInNonperiodicDomain(pclList));
  }
}

// ========================================================================
// apply_BCs_locally: apply periodic shift or face BCs to an incoming block
// ========================================================================

void ParticleCommInjection::applyBCsLocally(
  vector_SpeciesParticle& pclList, int direction, bool applyShift, bool doApplyBCs)
{
  using namespace Direction;
  const double Lxinv = 1.0 / domainLengthX_;
  const double Lyinv = 1.0 / domainLengthY_;
  const double Lzinv = 1.0 / domainLengthZ_;

  if (applyShift) {
    switch (direction) {
      default: invalid_value_error(direction);
      case XDN: case XUP:
        for (int pidx = 0; pidx < (int)pclList.size(); pidx++) {
          cudaParticleType& posX = pclList[pidx].fetch_x();
          posX = modulo(posX, domainLengthX_, Lxinv);
        }
        break;
      case YDN: case YUP:
        for (int pidx = 0; pidx < (int)pclList.size(); pidx++) {
          cudaParticleType& posY = pclList[pidx].fetch_y();
          posY = modulo(posY, domainLengthY_, Lyinv);
        }
        break;
      case ZDN: case ZUP:
        for (int pidx = 0; pidx < (int)pclList.size(); pidx++) {
          cudaParticleType& posZ = pclList[pidx].fetch_z();
          posZ = modulo(posZ, domainLengthZ_, Lzinv);
        }
        break;
    }
  } else if (doApplyBCs) {
    switch (direction) {
      default: invalid_value_error(direction);
      case XDN: assert(vct_->noXleftNeighbor_P()); apply_Xleft_BC(pclList); break;
      case XUP: assert(vct_->noXrghtNeighbor_P()); apply_Xrght_BC(pclList); break;
      case YDN: assert(vct_->noYleftNeighbor_P()); apply_Yleft_BC(pclList); break;
      case YUP: assert(vct_->noYrghtNeighbor_P()); apply_Yrght_BC(pclList); break;
      case ZDN: assert(vct_->noZleftNeighbor_P()); apply_Zleft_BC(pclList); break;
      case ZUP: assert(vct_->noZrghtNeighbor_P()); apply_Zrght_BC(pclList); break;
    }
  }
}

// ========================================================================
// 6 face BC methods (virtual, operating on AoS blocks from BlockCommunicator)
// ========================================================================

void ParticleCommInjection::apply_Xleft_BC(vector_SpeciesParticle& pcls, int start)
{
  const int size = pcls.size();
  assert_le(0, start);
  switch (bcPfaceXleft_) {
    default: unsupported_value_error(bcPfaceXleft_);
    case BCparticles::PERFECT_MIRROR:
      for (int pidx = start; pidx < size; pidx++) {
        pcls[pidx].fetch_x() *= -1;
        pcls[pidx].fetch_u() *= -1;
      }
      break;
    case BCparticles::REEMISSION:
      for (int pidx = start; pidx < size; pidx++) {
        SpeciesParticle& pcl = pcls[pidx];
        pcl.fetch_x() *= -1;
        double vel[3];
        sample_maxwellian(vel[0], vel[1], vel[2],
                          thermalVelocityX_, thermalVelocityY_, thermalVelocityZ_, bcRng_);
        vel[0] = fabs(vel[0]);
        pcl.set_u(vel);
      }
      break;
    case BCparticles::EXIT:
      pcls.resize(start);
      break;
    case BCparticles::OPENBCIn:  break;
    case BCparticles::OPENBCOut: break;
  }
}

void ParticleCommInjection::apply_Yleft_BC(vector_SpeciesParticle& pcls, int start)
{
  const int size = pcls.size();
  assert_le(0, start);
  switch (bcPfaceYleft_) {
    default: unsupported_value_error(bcPfaceYleft_);
    case BCparticles::PERFECT_MIRROR:
      for (int pidx = start; pidx < size; pidx++) {
        pcls[pidx].fetch_y() *= -1;
        pcls[pidx].fetch_v() *= -1;
      }
      break;
    case BCparticles::REEMISSION:
      for (int pidx = start; pidx < size; pidx++) {
        SpeciesParticle& pcl = pcls[pidx];
        pcl.fetch_y() *= -1;
        double vel[3];
        sample_maxwellian(vel[0], vel[1], vel[2],
                          thermalVelocityX_, thermalVelocityY_, thermalVelocityZ_, bcRng_);
        vel[1] = fabs(vel[1]);
        pcl.set_u(vel);
      }
      break;
    case BCparticles::EXIT:
      pcls.resize(start);
      break;
    case BCparticles::OPENBCIn:  break;
    case BCparticles::OPENBCOut: break;
  }
}

void ParticleCommInjection::apply_Zleft_BC(vector_SpeciesParticle& pcls, int start)
{
  const int size = pcls.size();
  assert_le(0, start);
  switch (bcPfaceZleft_) {
    default: unsupported_value_error(bcPfaceZleft_);
    case BCparticles::PERFECT_MIRROR:
      for (int pidx = start; pidx < size; pidx++) {
        pcls[pidx].fetch_z() *= -1;
        pcls[pidx].fetch_w() *= -1;
      }
      break;
    case BCparticles::REEMISSION:
      for (int pidx = start; pidx < size; pidx++) {
        SpeciesParticle& pcl = pcls[pidx];
        pcl.fetch_z() *= -1;
        double vel[3];
        sample_maxwellian(vel[0], vel[1], vel[2],
                          thermalVelocityX_, thermalVelocityY_, thermalVelocityZ_, bcRng_);
        vel[2] = fabs(vel[2]);
        pcl.set_u(vel);
      }
      break;
    case BCparticles::EXIT:
      pcls.resize(start);
      break;
    case BCparticles::OPENBCIn:  break;
    case BCparticles::OPENBCOut: break;
  }
}

void ParticleCommInjection::apply_Xrght_BC(vector_SpeciesParticle& pcls, int start)
{
  const int size = pcls.size();
  assert_le(0, start);
  switch (bcPfaceXright_) {
    default: unsupported_value_error(bcPfaceXright_);
    case BCparticles::PERFECT_MIRROR:
      for (int pidx = start; pidx < size; pidx++) {
        cudaParticleType& posX = pcls[pidx].fetch_x();
        posX = 2 * domainLengthX_ - posX;
        pcls[pidx].fetch_u() *= -1;
      }
      break;
    case BCparticles::REEMISSION:
      for (int pidx = start; pidx < size; pidx++) {
        SpeciesParticle& pcl = pcls[pidx];
        cudaParticleType& posX = pcl.fetch_x();
        posX = 2 * domainLengthX_ - posX;
        double vel[3];
        sample_maxwellian(vel[0], vel[1], vel[2],
                          thermalVelocityX_, thermalVelocityY_, thermalVelocityZ_, bcRng_);
        vel[0] = -fabs(vel[0]);
        pcl.set_u(vel);
      }
      break;
    case BCparticles::EXIT:
      pcls.resize(start);
      break;
    case BCparticles::OPENBCIn:  break;
    case BCparticles::OPENBCOut: break;
  }
}

void ParticleCommInjection::apply_Yrght_BC(vector_SpeciesParticle& pcls, int start)
{
  const int size = pcls.size();
  assert_le(0, start);
  switch (bcPfaceYright_) {
    default: unsupported_value_error(bcPfaceYright_);
    case BCparticles::PERFECT_MIRROR:
      for (int pidx = start; pidx < size; pidx++) {
        cudaParticleType& posY = pcls[pidx].fetch_y();
        posY = 2 * domainLengthY_ - posY;
        pcls[pidx].fetch_v() *= -1;
      }
      break;
    case BCparticles::REEMISSION:
      for (int pidx = start; pidx < size; pidx++) {
        SpeciesParticle& pcl = pcls[pidx];
        cudaParticleType& posY = pcl.fetch_y();
        posY = 2 * domainLengthY_ - posY;
        double vel[3];
        sample_maxwellian(vel[0], vel[1], vel[2],
                          thermalVelocityX_, thermalVelocityY_, thermalVelocityZ_, bcRng_);
        vel[1] = -fabs(vel[1]);
        pcl.set_u(vel);
      }
      break;
    case BCparticles::EXIT:
      pcls.resize(start);
      break;
    case BCparticles::OPENBCIn:  break;
    case BCparticles::OPENBCOut: break;
  }
}

void ParticleCommInjection::apply_Zrght_BC(vector_SpeciesParticle& pcls, int start)
{
  const int size = pcls.size();
  assert_le(0, start);
  switch (bcPfaceZright_) {
    default: unsupported_value_error(bcPfaceZright_);
    case BCparticles::PERFECT_MIRROR:
      for (int pidx = start; pidx < size; pidx++) {
        cudaParticleType& posZ = pcls[pidx].fetch_z();
        posZ = 2 * domainLengthZ_ - posZ;
        pcls[pidx].fetch_w() *= -1;
      }
      break;
    case BCparticles::REEMISSION:
      for (int pidx = start; pidx < size; pidx++) {
        SpeciesParticle& pcl = pcls[pidx];
        cudaParticleType& posZ = pcl.fetch_z();
        posZ = 2 * domainLengthZ_ - posZ;
        double vel[3];
        sample_maxwellian(vel[0], vel[1], vel[2],
                          thermalVelocityX_, thermalVelocityY_, thermalVelocityZ_, bcRng_);
        vel[2] = -fabs(vel[2]);
        pcl.set_u(vel);
      }
      break;
    case BCparticles::EXIT:
      pcls.resize(start);
      break;
    case BCparticles::OPENBCIn:  break;
    case BCparticles::OPENBCOut: break;
  }
}

/**
 * @brief Send exiting particles from the AoS communication buffer to neighbours.
 *
 * The method iterates the current communication buffer, forwards particles to
 * the appropriate `BlockCommunicator`, and swap-removes the particles that
 * were successfully queued for sending.
 *
 * @return Number of particles removed from the local communication buffer.
 */
int ParticleCommInjection::separateAndSendParticles()
{
  // activate receiving
  recvXleft_.recv_start(); recvXrght_.recv_start();
  recvYleft_.recv_start(); recvYrght_.recv_start();
  recvZleft_.recv_start(); recvZrght_.recv_start();

  // prepare send blocks
  sendXleft_.send_start(); sendXrght_.send_start();
  sendYleft_.send_start(); sendYrght_.send_start();
  sendZleft_.send_start(); sendZrght_.send_start();

  int sendCount[6] = {0, 0, 0, 0, 0, 0};
  const int numPclsInitially = getCommNOP();
  int currentIndex = 0;

  while (currentIndex < getCommNOP()) {
    SpeciesParticle& pcl = commPcls[currentIndex];

    bool wasSent = sendParticleToAppropriateBuffer(pcl, sendCount);

    if (__builtin_expect(wasSent, true)) {
      // In the GPU version, all particles in the comm buffer are exiting
      deleteCommParticle(currentIndex);
    } else {
      currentIndex++;
    }
  }

  const int numPclsSent = numPclsInitially - getCommNOP();
  if (print_pcl_comm_counts) {
    dprintf("spec %d send_count: %d+%d+%d+%d+%d+%d=%d", speciesNumber_,
            sendCount[0], sendCount[1], sendCount[2],
            sendCount[3], sendCount[4], sendCount[5], numPclsSent);
  }
  return numPclsSent;
}

// ======= Receive and post-process MPI particle blocks =======

namespace PclCommMode
{
  enum Enum
  {
    do_apply_BCs_globally = 1,
    print_sent_pcls = 2,
  };
}

/**
 * @brief Receive incoming particle blocks, apply boundary conditions, and keep survivors.
 *
 * Surviving particles are appended to the AoS communication buffer; particles
 * that still belong to other ranks are re-sent through the communicator layer.
 *
 * @param pclCommMode Bitmask controlling BC handling and debug printing.
 * @return Number of particles re-sent to a different rank.
 */
int ParticleCommInjection::handleReceivedParticles(int pclCommMode)
{
  using namespace PclCommMode;

  recvXleft_.recv_start(); recvXrght_.recv_start();
  recvYleft_.recv_start(); recvYrght_.recv_start();
  recvZleft_.recv_start(); recvZrght_.recv_start();

  sendXleft_.send_start(); sendXrght_.send_start();
  sendYleft_.send_start(); sendYrght_.send_start();
  sendZleft_.send_start(); sendZrght_.send_start();

  const int numRecvBuffers = 6;
  int recvCount[6] = {0, 0, 0, 0, 0, 0};
  int sendCount[6] = {0, 0, 0, 0, 0, 0};
  int numPclsRecved = 0;
  int numPclsResent = 0;

  MPI_Request recvRequests[numRecvBuffers] = {
    recvXleft_.get_curr_request(), recvXrght_.get_curr_request(),
    recvYleft_.get_curr_request(), recvYrght_.get_curr_request(),
    recvZleft_.get_curr_request(), recvZrght_.get_curr_request()
  };
  BlockCommunicator<SpeciesParticle>* recvBuffArr[numRecvBuffers] = {
    &recvXleft_, &recvXrght_,
    &recvYleft_, &recvYrght_,
    &recvZleft_, &recvZrght_
  };

  assert(!recvXleft_.comm_finished());
  assert(!recvXrght_.comm_finished());
  assert(!recvYleft_.comm_finished());
  assert(!recvYrght_.comm_finished());
  assert(!recvZleft_.comm_finished());
  assert(!recvZrght_.comm_finished());

  const bool applyShift[numRecvBuffers] = {
    vct_->isPeriodicXlower_P(), vct_->isPeriodicXupper_P(),
    vct_->isPeriodicYlower_P(), vct_->isPeriodicYupper_P(),
    vct_->isPeriodicZlower_P(), vct_->isPeriodicZupper_P()
  };
  const bool doApplyBCs[numRecvBuffers] = {
    vct_->noXleftNeighbor_P(), vct_->noXrghtNeighbor_P(),
    vct_->noYleftNeighbor_P(), vct_->noYrghtNeighbor_P(),
    vct_->noZleftNeighbor_P(), vct_->noZrghtNeighbor_P()
  };
  const int directionMap[numRecvBuffers] = {
    Direction::XDN, Direction::XUP,
    Direction::YDN, Direction::YUP,
    Direction::ZDN, Direction::ZUP
  };

  while (!(recvXleft_.comm_finished() && recvXrght_.comm_finished() &&
           recvYleft_.comm_finished() && recvYrght_.comm_finished() &&
           recvZleft_.comm_finished() && recvZrght_.comm_finished()))
  {
    int recvIndex;
    MPI_Status recvStatus;
    MPI_Waitany(numRecvBuffers, recvRequests, &recvIndex, &recvStatus);
    if (recvIndex == MPI_UNDEFINED)
      eprintf("recvRequests contains no active handles");
    assert_ge(recvIndex, 0);
    assert_lt(recvIndex, numRecvBuffers);

    BlockCommunicator<SpeciesParticle>* recvBuff = recvBuffArr[recvIndex];
    Block<SpeciesParticle>& recvBlock = recvBuff->fetch_received_block(recvStatus);
    vector_SpeciesParticle& pclList = recvBlock.fetch_block();

    if (pclCommMode & do_apply_BCs_globally) {
      applyBCsGlobally(pclList);
    } else {
      applyBCsLocally(pclList, directionMap[recvIndex],
                       applyShift[recvIndex], doApplyBCs[recvIndex]);
    }

    recvCount[recvIndex] += recvBlock.size();
    numPclsRecved += recvBlock.size();

    // Process each particle in the received block
    for (int pidx = 0; pidx < recvBlock.size(); pidx++) {
      SpeciesParticle& pcl = recvBlock[pidx];
      bool wasSent = sendParticleToAppropriateBuffer(pcl, sendCount);

      if (__builtin_expect(wasSent, false)) {
        numPclsResent++;
      } else {
        // Particle belongs here → append to AoS comm buffer
        appendSingleParticleToComm(pcl);
      }
    }

    recvBuff->release_received_block();
    recvRequests[recvIndex] = recvBuff->get_curr_request();
  }

  if (print_pcl_comm_counts) {
    dprintf("spec %d recved_count: %d+%d+%d+%d+%d+%d=%d", speciesNumber_,
            recvCount[0], recvCount[1], recvCount[2],
            recvCount[3], recvCount[4], recvCount[5], numPclsRecved);
    dprintf("spec %d resent_count: %d+%d+%d+%d+%d+%d=%d", speciesNumber_,
            sendCount[0], sendCount[1], sendCount[2],
            sendCount[3], sendCount[4], sendCount[5], numPclsResent);
  }

  return numPclsResent;
}

/**
 * @brief Iterate particle exchange until no particles remain in transit.
 *
 * The first `minNumIterations` iterations run unconditionally. After that, the
 * method switches to a global-allreduce termination check and stops once every
 * rank reports zero forwarded particles.
 *
 * @param minNumIterations Minimum number of exchange iterations before convergence checks.
 */
void ParticleCommInjection::recommunicateParticlesUntilDone(int minNumIterations)
{
  assert_gt(minNumIterations, 0);

  long long numPclsSent;
  for (int iteration = 0; iteration < minNumIterations; iteration++) {
    flushSend();
    numPclsSent = handleReceivedParticles();
  }

  // Apply BCs globally to remaining incoming particles
  flushSend();
  numPclsSent = handleReceivedParticles(PclCommMode::do_apply_BCs_globally);

  // Continue until global all-reduce confirms no more particles in transit
  long long totalNumPclsSent;
  MPI_Allreduce(&numPclsSent, &totalNumPclsSent, 1, MPI_LONG_LONG, MPI_SUM, mpiComm_);

  int commMaxTimes = vct_->getXLEN() + vct_->getYLEN() + vct_->getZLEN();
  if (!do_apply_periodic_BC_global) commMaxTimes *= 2;
  int commCount = 0;

  while (totalNumPclsSent) {
    if (commCount >= commMaxTimes) {
      dprintf("spec %d particles still uncommunicated:", speciesNumber_);
      flushSend();
      numPclsSent = handleReceivedParticles(PclCommMode::print_sent_pcls);
      eprintf("failed to finish up particle communication"
              " within %d communications", commMaxTimes);
    }
    flushSend();
    numPclsSent = handleReceivedParticles();
    MPI_Allreduce(&numPclsSent, &totalNumPclsSent, 1, MPI_LONG_LONG, MPI_SUM, mpiComm_);
    if (print_pcl_comm_counts) {
      dprint(totalNumPclsSent);
    }
    commCount++;
  }
}

/**
 * @brief Append externally generated AoS particles to the communication buffer.
 *
 * @param buffer Pointer to the input AoS particle buffer.
 * @param count Number of particles to append from @p buffer.
 */
void ParticleCommInjection::appendFromAoS(const SpeciesParticle* buffer, int count)
{
  if (count <= 0) return;
  const int newTotalNOP = getCommNOP() + count;
  const int padded = roundup_to_multiple(newTotalNOP, DVECWIDTH);
  commPcls.reserve(padded);
  const int oldSize = commPcls.size();
  commPcls.resize(oldSize + count);
  memcpy(commPcls.getList() + oldSize, buffer, count * sizeof(SpeciesParticle));
}

// ======= Injection count (read-only) =======

int ParticleCommInjection::computeInjectionCount() const
{
  return cachedInjectionCount_;
}

int ParticleCommInjection::computeInjectionCountImpl() const
{
  using namespace BCparticles;

  if (!vct_->isBoundaryProcess_P()) return 0;

  const bool repopXleft = (vct_->noXleftNeighbor_P() && bcPfaceXleft_ == REEMISSION);
  const bool repopYleft = (vct_->noYleftNeighbor_P() && bcPfaceYleft_ == REEMISSION);
  const bool repopZleft = (vct_->noZleftNeighbor_P() && bcPfaceZleft_ == REEMISSION);
  const bool repopXrght = (vct_->noXrghtNeighbor_P() && bcPfaceXright_ == REEMISSION);
  const bool repopYrght = (vct_->noYrghtNeighbor_P() && bcPfaceYright_ == REEMISSION);
  const bool repopZrght = (vct_->noZrghtNeighbor_P() && bcPfaceZright_ == REEMISSION);

  if (!(repopXleft || repopYleft || repopZleft || repopXrght || repopYrght || repopZrght))
    return 0;

  const int nxc = numCellsX_;
  const int nyc = numCellsY_;
  const int nzc = numCellsZ_;
  const int numLayers = 3;

  const int upXstart = nxc - 1 - numLayers;
  const int upYstart = nyc - 1 - numLayers;
  const int upZstart = nzc - 1 - numLayers;

  int xbeg = 1, xend = nxc - 2;
  int ybeg = 1, yend = nyc - 2;
  int zbeg = 1, zend = nzc - 2;

  auto faceCells = [&](int ixB, int ixE, int iyB, int iyE, int izB, int izE) {
    return (ixE - ixB + 1) * (iyE - iyB + 1) * (izE - izB + 1);
  };

  int total = 0;
  // Xleft
  if (repopXleft) { total += faceCells(1, numLayers, ybeg, yend, zbeg, zend); xbeg += numLayers; }
  // Xrght
  if (repopXrght) { total += faceCells(upXstart, xend, ybeg, yend, zbeg, zend); xend -= numLayers; }
  // Yleft
  if (repopYleft) { total += faceCells(xbeg, xend, 1, numLayers, zbeg, zend); ybeg += numLayers; }
  // Yrght
  if (repopYrght) { total += faceCells(xbeg, xend, upYstart, yend, zbeg, zend); yend -= numLayers; }
  // Zleft
  if (repopZleft)  total += faceCells(xbeg, xend, ybeg, yend, 1, numLayers);
  // Zrght
  if (repopZrght)  total += faceCells(xbeg, xend, ybeg, yend, upZstart, zend);

  return total * numParticlesPerCell_;
}

// ======= Fill GPU injection parameter struct (called once at init) =======

void ParticleCommInjection::fillInjectionParameter(injectionParameter* param) const
{
  using namespace BCparticles;

  // Zero-initialize the whole struct
  memset(param, 0, sizeof(injectionParameter));

  // Default: disabled
  param->enabled = false;
  param->totalInjected = 0;
  if (hostParticles_.tracksParticleID()) {
    param->particleIDGenerator = hostParticles_.getParticleIDGenerator();
  }

  if (!vct_->isBoundaryProcess_P()) return;

  const bool repopXleft = (vct_->noXleftNeighbor_P() && bcPfaceXleft_ == REEMISSION);
  const bool repopYleft = (vct_->noYleftNeighbor_P() && bcPfaceYleft_ == REEMISSION);
  const bool repopZleft = (vct_->noZleftNeighbor_P() && bcPfaceZleft_ == REEMISSION);
  const bool repopXrght = (vct_->noXrghtNeighbor_P() && bcPfaceXright_ == REEMISSION);
  const bool repopYrght = (vct_->noYrghtNeighbor_P() && bcPfaceYright_ == REEMISSION);
  const bool repopZrght = (vct_->noZrghtNeighbor_P() && bcPfaceZright_ == REEMISSION);

  if (!(repopXleft || repopYleft || repopZleft || repopXrght || repopYrght || repopZrght))
    return;

  // --- Physics ---
  param->thermalVelX = thermalVelocityX_;
  param->thermalVelY = thermalVelocityY_;
  param->thermalVelZ = thermalVelocityZ_;
  param->driftVelX   = driftVelocityX_;
  param->driftVelY   = driftVelocityY_;
  param->driftVelZ   = driftVelocityZ_;
  param->speedOfLightSq = speedOfLight_ * speedOfLight_;

  const double FourPI = 16.0 * atan(1.0);
  param->chargePerParticle =
      (chargeOverMass_ / fabs(chargeOverMass_)) *
      (injectionDensity_ / FourPI / numParticlesPerCell_) *
      (1.0 / grid_->getInvVOL());

  // --- Subcell grid ---
  param->dxPerPcl = gridSpacingX_ / numPclPerCellX_;
  param->dyPerPcl = gridSpacingY_ / numPclPerCellY_;
  param->dzPerPcl = gridSpacingZ_ / numPclPerCellZ_;
  param->numPclPerCellX = numPclPerCellX_;
  param->numPclPerCellY = numPclPerCellY_;
  param->numPclPerCellZ = numPclPerCellZ_;
  param->numParticlesPerCell = numParticlesPerCell_;

  // --- Domain bounds ---
  param->domainLengthX = domainLengthX_;
  param->domainLengthY = domainLengthY_;
  param->domainLengthZ = domainLengthZ_;

  // --- Grid origin (for cell-corner formula: xStart + (ix-1)*dx) ---
  param->gridXstart = grid_->getXstart();
  param->gridYstart = grid_->getYstart();
  param->gridZstart = grid_->getZstart();
  param->gridDx = gridSpacingX_;
  param->gridDy = gridSpacingY_;
  param->gridDz = gridSpacingZ_;

  // --- Face ranges (same narrowing logic as computeInjectionCountImpl) ---
  const int nxc = numCellsX_;
  const int nyc = numCellsY_;
  const int nzc = numCellsZ_;
  const int numLayers = 3;

  const int upXstart = nxc - 1 - numLayers;
  const int upYstart = nyc - 1 - numLayers;
  const int upZstart = nzc - 1 - numLayers;

  int xbeg = 1, xend = nxc - 2;
  int ybeg = 1, yend = nyc - 2;
  int zbeg = 1, zend = nzc - 2;

  auto setFace = [&](int faceIdx, int ixB, int ixE, int iyB, int iyE, int izB, int izE, bool active) {
    param->faces[faceIdx].ixBeg = ixB;  param->faces[faceIdx].ixEnd = ixE;
    param->faces[faceIdx].iyBeg = iyB;  param->faces[faceIdx].iyEnd = iyE;
    param->faces[faceIdx].izBeg = izB;  param->faces[faceIdx].izEnd = izE;
    param->faces[faceIdx].nY = iyE - iyB + 1;
    param->faces[faceIdx].nZ = izE - izB + 1;
    param->faces[faceIdx].nCells = active ? (ixE - ixB + 1) * param->faces[faceIdx].nY * param->faces[faceIdx].nZ : 0;
    param->faces[faceIdx].active = active;
  };

  // Xleft (face 0)
  setFace(0, 1, numLayers, ybeg, yend, zbeg, zend, repopXleft);
  if (repopXleft) xbeg += numLayers;
  // Xrght (face 1)
  setFace(1, upXstart, xend, ybeg, yend, zbeg, zend, repopXrght);
  if (repopXrght) xend -= numLayers;
  // Yleft (face 2)
  setFace(2, xbeg, xend, 1, numLayers, zbeg, zend, repopYleft);
  if (repopYleft) ybeg += numLayers;
  // Yrght (face 3)
  setFace(3, xbeg, xend, upYstart, yend, zbeg, zend, repopYrght);
  if (repopYrght) yend -= numLayers;
  // Zleft (face 4)
  setFace(4, xbeg, xend, ybeg, yend, 1, numLayers, repopZleft);
  // Zrght (face 5)
  setFace(5, xbeg, xend, ybeg, yend, upZstart, zend, repopZrght);

  // Cumulative particle offsets
  int total = 0;
  for (int faceIdx = 0; faceIdx < 6; faceIdx++) {
    param->pclOffset[faceIdx] = total;
    total += param->faces[faceIdx].nCells * numParticlesPerCell_;
  }
  param->totalInjected = total;
  param->enabled = (total > 0);
}
