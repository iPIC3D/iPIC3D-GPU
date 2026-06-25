/**
 * ParticleSoAHost.cpp — Unified SoA host container implementation.
 *
 * Merges all init, diagnostics, sort, BC config, and injection methods
 * from the former Particles3D, Particles3Dcomm, and ParticleSoAHost.
 */

#include "ParticleSoAHost.h"
#include "ompdefs.h"
#include "Collective.h"
#include "VCtopology3D.h"
#include "Grid3DCU.h"
#include "EMfields3D.h"
#include "MPIdata.h"
#include "asserts.h"
#include "errors.h"
#include "Particle.h"
#include "Larray.h"

#include <iostream>
#include <cstdlib>
#include <algorithm>
#include <random>

using std::cout;
using std::endl;

// ========================================================================
// Constructor
// ========================================================================

ParticleSoAHost::ParticleSoAHost(int speciesNum, CollectiveIO* col,
                                 VirtualTopology3D* vct, Grid* grid)
  : speciesNumber_(speciesNum),
    col_(col),
    vct_(vct),
    grid_(grid),
    mpiComm_(vct->getParticleComm())
{
  // --- Species identity ---
  isTestParticle_ = (speciesNum >= col->getNs());
  trackParticleID_ = col->getTrackParticleID(speciesNum);
  chargeOverMass_ = col->getQOM(speciesNum);
  numParticlesPerCell_ = col->getNpcel(speciesNum);
  numPclPerCellX_ = col->getNpcelx(speciesNum);
  numPclPerCellY_ = col->getNpcely(speciesNum);
  numPclPerCellZ_ = col->getNpcelz(speciesNum);

  if (!isTestParticle_) {
    thermalVelocityX_ = col->getUth(speciesNum);
    thermalVelocityY_ = col->getVth(speciesNum);
    thermalVelocityZ_ = col->getWth(speciesNum);
    driftVelocityX_   = col->getU0(speciesNum);
    driftVelocityY_   = col->getV0(speciesNum);
    driftVelocityZ_   = col->getW0(speciesNum);
    injectionDensity_ = col->getRHOinject(speciesNum);
  } else {
    pitchAngle_ = col->getPitchAngle(speciesNum - col->getNs());
    energy_     = col->getEnergy(speciesNum - col->getNs());
  }

  // --- Time step and physics ---
  timeStep_           = col->getDt();
  speedOfLight_       = col->getC();
  numMoverIterations_ = col->getNiterMover();
  injectionVelocity_  = col->getVinj();
  reconThickness_     = col->getDelta();

  // --- Domain geometry ---
  domainLengthX_ = col->getLx();
  domainLengthY_ = col->getLy();
  domainLengthZ_ = col->getLz();

  gridSpacingX_    = grid->getDX();
  gridSpacingY_    = grid->getDY();
  gridSpacingZ_    = grid->getDZ();
  invGridSpacingX_ = 1.0 / gridSpacingX_;
  invGridSpacingY_ = 1.0 / gridSpacingY_;
  invGridSpacingZ_ = 1.0 / gridSpacingZ_;

  subdomainXstart_ = grid->getXstart();
  subdomainXend_   = grid->getXend();
  subdomainYstart_ = grid->getYstart();
  subdomainYend_   = grid->getYend();
  subdomainZstart_ = grid->getZstart();
  subdomainZend_   = grid->getZend();

  numNodesX_ = grid->getNXN();
  numNodesY_ = grid->getNYN();
  numNodesZ_ = grid->getNZN();
  numCellsX_ = grid->getNXC();
  numCellsY_ = grid->getNYC();
  numCellsZ_ = grid->getNZC();
  inverseVolume_ = grid->getInvVOL();

  // --- Boundary conditions ---
  bcPfaceXleft_  = col->getBcPfaceXleft();
  bcPfaceXright_ = col->getBcPfaceXright();
  bcPfaceYleft_  = col->getBcPfaceYleft();
  bcPfaceYright_ = col->getBcPfaceYright();
  bcPfaceZleft_  = col->getBcPfaceZleft();
  bcPfaceZright_ = col->getBcPfaceZright();

  // --- Velocity caps ---
  velocityCapMaxX_ = 0.95 * domainLengthX_ / timeStep_;
  velocityCapMaxY_ = 0.95 * domainLengthY_ / timeStep_;
  velocityCapMaxZ_ = 0.95 * domainLengthZ_ / timeStep_;
  velocityCapMinX_ = -velocityCapMaxX_;
  velocityCapMinY_ = -velocityCapMaxY_;
  velocityCapMinZ_ = -velocityCapMaxZ_;

  // --- Particle ID generator ---
  if (trackParticleID_) {
    particleIDGenerator_.initialize(vct_->getCartesian_rank(), vct_->getNprocs(),
                                    speciesNumber_,
                                    col_->getNs() + col_->getNsTestPart());
  }

  // --- Sorting arrays ---
  numParticlesInBucket_    = new array3_int(numCellsX_, numCellsY_, numCellsZ_);
  numParticlesInBucketNow_ = new array3_int(numCellsX_, numCellsY_, numCellsZ_);
  bucketOffset_            = new array3_int(numCellsX_, numCellsY_, numCellsZ_);
}

// ======= Destruction =======

ParticleSoAHost::~ParticleSoAHost()
{
  delete numParticlesInBucket_;
  delete numParticlesInBucketNow_;
  delete bucketOffset_;
}

// ======= Particle generation =======

/**
 * @brief Populate the species with the default Maxwellian initialization.
 *
 * @param EMf Field object used to sample equilibrium density.
 */
void ParticleSoAHost::maxwellian(Field* EMf)
{
  assert_eq(getNOP(), 0);

  const int nxr = numCellsX_ - 2;
  const int nyr = numCellsY_ - 2;
  const int nzr = numCellsZ_ - 2;
  const int nop = nxr * nyr * nzr * numParticlesPerCell_;
  const auto baseSequence = trackParticleID_
      ? particleIDGenerator_.reserveHostSequenceBlock(nop)
      : ParticleIDGenerator::counter_type{0};
  const double chargeFactor = (chargeOverMass_ / fabs(chargeOverMass_)) * grid_->getVOL() / numParticlesPerCell_;

  prepareSoAForNOP(nop);

  #pragma omp parallel
  {
    const int tid = omp_get_thread_num();
    std::mt19937_64 rng(vct_->getCartesian_rank() * 31 + speciesNumber_ * 127 + tid * 1049);

    #pragma omp for collapse(3) schedule(static)
    for (int i = 1; i < numCellsX_ - 1; ++i)
    for (int j = 1; j < numCellsY_ - 1; ++j)
    for (int k = 1; k < numCellsZ_ - 1; ++k)
    {
      const int cellIdx = ((i - 1) * nyr + (j - 1)) * nzr + (k - 1);
      const int idxBase = cellIdx * numParticlesPerCell_;
      const double chargePerParticle = chargeFactor * EMf->getRHOcs(i, j, k, speciesNumber_);

      for (int ii = 0; ii < numPclPerCellX_; ++ii)
      for (int jj = 0; jj < numPclPerCellY_; ++jj)
      for (int kk = 0; kk < numPclPerCellZ_; ++kk)
      {
        const int subIdx = (ii * numPclPerCellY_ + jj) * numPclPerCellZ_ + kk;
        const int idx = idxBase + subIdx;

        double velX, velY, velZ;
        sample_maxwellian(velX, velY, velZ,
                          thermalVelocityX_, thermalVelocityY_, thermalVelocityZ_,
                          driftVelocityX_, driftVelocityY_, driftVelocityZ_, rng);

        u[idx] = velX;
        v[idx] = velY;
        w[idx] = velZ;
        q[idx] = chargePerParticle;
        x[idx] = (ii + .5) * (gridSpacingX_ / numPclPerCellX_) + grid_->getXN(i, j, k);
        y[idx] = (jj + .5) * (gridSpacingY_ / numPclPerCellY_) + grid_->getYN(i, j, k);
        z[idx] = (kk + .5) * (gridSpacingZ_ / numPclPerCellZ_) + grid_->getZN(i, j, k);
        if (trackParticleID_)
          id[idx] = particleIDGenerator_.idFromSequence(baseSequence + idx);
      }
    }
  }
}

/**
 * @brief Populate the null-points/Taylor-Green cases with local current-driven drift.
 *
 * @param EMf Field object used to sample density and local current.
 */
void ParticleSoAHost::maxwellianNullPoints(Field* EMf)
{
  const int nxr = numCellsX_ - 2;
  const int nyr = numCellsY_ - 2;
  const int nzr = numCellsZ_ - 2;
  const int nop = nxr * nyr * nzr * numParticlesPerCell_;
  const auto baseSequence = trackParticleID_
      ? particleIDGenerator_.reserveHostSequenceBlock(nop)
      : ParticleIDGenerator::counter_type{0};
  const double chargeSign = chargeOverMass_ / fabs(chargeOverMass_);
  const double chargeFactor = chargeSign * grid_->getVOL() / numParticlesPerCell_;

  prepareSoAForNOP(nop);

  #pragma omp parallel
  {
    const int tid = omp_get_thread_num();
    std::mt19937_64 rng(vct_->getCartesian_rank() * 31 + speciesNumber_ * 127 + tid * 1049);

    #pragma omp for collapse(3) schedule(static)
    for (int i = 1; i < numCellsX_ - 1; i++)
    for (int j = 1; j < numCellsY_ - 1; j++)
    for (int k = 1; k < numCellsZ_ - 1; k++)
    {
      const int cellIdx = ((i - 1) * nyr + (j - 1)) * nzr + (k - 1);
      const int idxBase = cellIdx * numParticlesPerCell_;
      const double chargePerParticle = chargeFactor * EMf->getRHOcs(i, j, k, speciesNumber_);

      // Jxs from initGEMHarris/initNullPoints stores the Ampere current in
      // the field convention (unsigned rhons).  The moment kernel deposits
      // Jzs_moment = sign(q)*rhons*v, so drift = Jzs / (sign(q)*rhons).
      const double signedRho = chargeSign * EMf->getRHOns(i, j, k, speciesNumber_);
      double localDriftX = EMf->getJxs(i, j, k, speciesNumber_) / signedRho;
      double localDriftY = EMf->getJys(i, j, k, speciesNumber_) / signedRho;
      double localDriftZ = EMf->getJzs(i, j, k, speciesNumber_) / signedRho;

      for (int ii = 0; ii < numPclPerCellX_; ++ii)
      for (int jj = 0; jj < numPclPerCellY_; ++jj)
      for (int kk = 0; kk < numPclPerCellZ_; ++kk)
      {
        const int subIdx = (ii * numPclPerCellY_ + jj) * numPclPerCellZ_ + kk;
        const int idx = idxBase + subIdx;

        double velX, velY, velZ;
        sample_maxwellian(velX, velY, velZ,
                          thermalVelocityX_, thermalVelocityY_, thermalVelocityZ_,
                          localDriftX, localDriftY, localDriftZ, rng);

        u[idx] = velX;
        v[idx] = velY;
        w[idx] = velZ;
        q[idx] = chargePerParticle;
        x[idx] = (ii + .5) * (gridSpacingX_ / numPclPerCellX_) + grid_->getXN(i, j, k);
        y[idx] = (jj + .5) * (gridSpacingY_ / numPclPerCellY_) + grid_->getYN(i, j, k);
        z[idx] = (kk + .5) * (gridSpacingZ_ / numPclPerCellZ_) + grid_->getZN(i, j, k);
        if (trackParticleID_)
          id[idx] = particleIDGenerator_.idFromSequence(baseSequence + idx);
      }
    }
  }
}

/**
 * @brief Populate using local Ampere-driven drift and reference-state thermal velocity.
 *
 * Drift: v_drift = Jxs/rhons per cell (from currentFromAmpere).
 * Thermal: vth = sqrt(pXXsn/rhons - (Jxs/rhons)^2) per cell.
 * Matches the ECsim reference-state initialization (at id=0 this gives global uth
 * algebraically, but preserves the mechanism for discrete numerical effects and
 * future extension to non-trivial pressure profiles).
 *
 * @param EMf Field object with Ampere currents and reference pressure state.
 */
void ParticleSoAHost::maxwellianAmpereVaryingThermal(Field* EMf)
{
  assert_eq(getNOP(), 0);

  const int nxr = numCellsX_ - 2;
  const int nyr = numCellsY_ - 2;
  const int nzr = numCellsZ_ - 2;
  const int nop = nxr * nyr * nzr * numParticlesPerCell_;
  const auto baseSequence = trackParticleID_
      ? particleIDGenerator_.reserveHostSequenceBlock(nop)
      : ParticleIDGenerator::counter_type{0};
  const double chargeSign = chargeOverMass_ / fabs(chargeOverMass_);  // +1 ions, -1 electrons
  const double chargeFactor = chargeSign * grid_->getVOL() / numParticlesPerCell_;

  prepareSoAForNOP(nop);

  #pragma omp parallel
  {
    const int tid = omp_get_thread_num();
    std::mt19937_64 rng(vct_->getCartesian_rank() * 31 + speciesNumber_ * 127 + tid * 1049);

    #pragma omp for collapse(3) schedule(static)
    for (int i = 1; i < numCellsX_ - 1; i++)
    for (int j = 1; j < numCellsY_ - 1; j++)
    for (int k = 1; k < numCellsZ_ - 1; k++)
    {
      const int cellIdx = ((i - 1) * nyr + (j - 1)) * nzr + (k - 1);
      const int idxBase = cellIdx * numParticlesPerCell_;
      const double chargePerParticle = chargeFactor * EMf->getRHOcs(i, j, k, speciesNumber_);

      // Local drift velocity from Ampere current.
      // Jxs is stored in the Ampere/field convention (no charge sign),
      // but rhons during init is always positive (unsigned).  The moment
      // kernel deposits Jzs_moment = sign(q)*rho*v, so to recover the
      // physical drift velocity we must divide by sign(q)*rho.
      const double rho = EMf->getRHOns(i, j, k, speciesNumber_);
      const double signedRho = chargeSign * rho;
      const double localDriftX = EMf->getJxs(i, j, k, speciesNumber_) / signedRho;
      const double localDriftY = EMf->getJys(i, j, k, speciesNumber_) / signedRho;
      const double localDriftZ = EMf->getJzs(i, j, k, speciesNumber_) / signedRho;

      // Local thermal velocity from reference pressure state:
      // vth_i = sqrt(p_ii / rho - (J_i / rho)^2)
      const double pxx = EMf->getpXXsn(i, j, k, speciesNumber_);
      const double pyy = EMf->getpYYsn(i, j, k, speciesNumber_);
      const double pzz = EMf->getpZZsn(i, j, k, speciesNumber_);
      const double vthX2 = pxx / rho - localDriftX * localDriftX;
      const double vthY2 = pyy / rho - localDriftY * localDriftY;
      const double vthZ2 = pzz / rho - localDriftZ * localDriftZ;
      // Guard against negative values from floating-point round-off
      const double localVthX = (vthX2 > 0.0) ? sqrt(vthX2) : thermalVelocityX_;
      const double localVthY = (vthY2 > 0.0) ? sqrt(vthY2) : thermalVelocityY_;
      const double localVthZ = (vthZ2 > 0.0) ? sqrt(vthZ2) : thermalVelocityZ_;

      for (int ii = 0; ii < numPclPerCellX_; ++ii)
      for (int jj = 0; jj < numPclPerCellY_; ++jj)
      for (int kk = 0; kk < numPclPerCellZ_; ++kk)
      {
        const int subIdx = (ii * numPclPerCellY_ + jj) * numPclPerCellZ_ + kk;
        const int idx = idxBase + subIdx;

        double velX, velY, velZ;
        sample_maxwellian(velX, velY, velZ,
                          localVthX, localVthY, localVthZ,
                          localDriftX, localDriftY, localDriftZ, rng);

        u[idx] = velX;
        v[idx] = velY;
        w[idx] = velZ;
        q[idx] = chargePerParticle;
        x[idx] = (ii + .5) * (gridSpacingX_ / numPclPerCellX_) + grid_->getXN(i, j, k);
        y[idx] = (jj + .5) * (gridSpacingY_ / numPclPerCellY_) + grid_->getYN(i, j, k);
        z[idx] = (kk + .5) * (gridSpacingZ_ / numPclPerCellZ_) + grid_->getZN(i, j, k);
        if (trackParticleID_)
          id[idx] = particleIDGenerator_.idFromSequence(baseSequence + idx);
      }
    }
  }
}

/**
 * @brief Populate the double-Harris configuration.
 *
 * @param EMf Field object used to sample equilibrium density.
 */
void ParticleSoAHost::maxwellianDoubleHarris(Field* EMf)
{
  assert_eq(getNOP(), 0);

  const int nxr = numCellsX_ - 2;
  const int nyr = numCellsY_ - 2;
  const int nzr = numCellsZ_ - 2;
  const int nop = nxr * nyr * nzr * numParticlesPerCell_;
  const auto baseSequence = trackParticleID_
      ? particleIDGenerator_.reserveHostSequenceBlock(nop)
      : ParticleIDGenerator::counter_type{0};
  const double chargeFactor = (chargeOverMass_ / fabs(chargeOverMass_)) * grid_->getVOL() / numParticlesPerCell_;
  const double domainYUpper = domainLengthY_ / 2.0;

  prepareSoAForNOP(nop);

  #pragma omp parallel
  {
    const int tid = omp_get_thread_num();
    std::mt19937_64 rng(vct_->getCartesian_rank() * 31 + speciesNumber_ * 127 + tid * 1049);

    #pragma omp for collapse(3) schedule(static)
    for (int i = 1; i < numCellsX_ - 1; i++)
    for (int j = 1; j < numCellsY_ - 1; j++)
    for (int k = 1; k < numCellsZ_ - 1; k++)
    {
      const int cellIdx = ((i - 1) * nyr + (j - 1)) * nzr + (k - 1);
      const int idxBase = cellIdx * numParticlesPerCell_;
      const double chargePerParticle = chargeFactor * EMf->getRHOcs(i, j, k, speciesNumber_);

      for (int ii = 0; ii < numPclPerCellX_; ++ii)
      for (int jj = 0; jj < numPclPerCellY_; ++jj)
      for (int kk = 0; kk < numPclPerCellZ_; ++kk)
      {
        const int subIdx = (ii * numPclPerCellY_ + jj) * numPclPerCellZ_ + kk;
        const int idx = idxBase + subIdx;

        const double posX = (ii + .5) * (gridSpacingX_ / numPclPerCellX_) + grid_->getXN(i, j, k);
        const double posY = (jj + .5) * (gridSpacingY_ / numPclPerCellY_) + grid_->getYN(i, j, k);
        const double posZ = (kk + .5) * (gridSpacingZ_ / numPclPerCellZ_) + grid_->getZN(i, j, k);

        double velX, velY, velZ;
        if (posY > domainYUpper)
          sample_maxwellian(velX, velY, velZ, thermalVelocityX_, thermalVelocityY_, thermalVelocityZ_,
                            driftVelocityX_, driftVelocityY_, driftVelocityZ_, rng);
        else
          sample_maxwellian(velX, velY, velZ, thermalVelocityX_, thermalVelocityY_, thermalVelocityZ_,
                            driftVelocityX_, driftVelocityY_, driftVelocityZ_, rng);

        u[idx] = velX;
        v[idx] = velY;
        w[idx] = velZ;
        q[idx] = chargePerParticle;
        x[idx] = posX;
        y[idx] = posY;
        z[idx] = posZ;
        if (trackParticleID_)
          id[idx] = particleIDGenerator_.idFromSequence(baseSequence + idx);
      }
    }
  }
}

/**
 * @brief Populate the hump-perturbation configuration.
 *
 * @param EMf Field object used to sample equilibrium density.
 */
void ParticleSoAHost::maxwellianHumpPerturbation(Field* EMf)
{
  assert_eq(getNOP(), 0);

  const int nxr = numCellsX_ - 2;
  const int nyr = numCellsY_ - 2;
  const int nzr = numCellsZ_ - 2;
  const int nop = nxr * nyr * nzr * numParticlesPerCell_;
  const auto baseSequence = trackParticleID_
      ? particleIDGenerator_.reserveHostSequenceBlock(nop)
      : ParticleIDGenerator::counter_type{0};
  const double chargeFactor = (chargeOverMass_ / fabs(chargeOverMass_)) * grid_->getVOL() / numParticlesPerCell_;

  prepareSoAForNOP(nop);

  #pragma omp parallel
  {
    const int tid = omp_get_thread_num();
    std::mt19937_64 rng(vct_->getCartesian_rank() * 31 + speciesNumber_ * 127 + tid * 1049);

    #pragma omp for collapse(3) schedule(static)
    for (int i = 1; i < numCellsX_ - 1; i++)
    for (int j = 1; j < numCellsY_ - 1; j++)
    for (int k = 1; k < numCellsZ_ - 1; k++)
    {
      const int cellIdx = ((i - 1) * nyr + (j - 1)) * nzr + (k - 1);
      const int idxBase = cellIdx * numParticlesPerCell_;
      const double chargePerParticle = chargeFactor * EMf->getRHOcs(i, j, k, speciesNumber_);

      for (int ii = 0; ii < numPclPerCellX_; ++ii)
      for (int jj = 0; jj < numPclPerCellY_; ++jj)
      for (int kk = 0; kk < numPclPerCellZ_; ++kk)
      {
        const int subIdx = (ii * numPclPerCellY_ + jj) * numPclPerCellZ_ + kk;
        const int idx = idxBase + subIdx;

        const double posX = (ii + .5) * (gridSpacingX_ / numPclPerCellX_) + grid_->getXN(i, j, k);
        const double posY = (jj + .5) * (gridSpacingY_ / numPclPerCellY_) + grid_->getYN(i, j, k);
        const double posZ = (kk + .5) * (gridSpacingZ_ / numPclPerCellZ_) + grid_->getZN(i, j, k);

        double velX, velY, velZ;
        sample_maxwellian(velX, velY, velZ,
                          thermalVelocityX_, thermalVelocityY_, thermalVelocityZ_,
                          driftVelocityX_, driftVelocityY_, driftVelocityZ_, rng);

        u[idx] = velX;
        v[idx] = velY;
        w[idx] = velZ;
        q[idx] = chargePerParticle;
        x[idx] = posX;
        y[idx] = posY;
        z[idx] = posZ;
        if (trackParticleID_)
          id[idx] = particleIDGenerator_.idFromSequence(baseSequence + idx);
      }
    }
  }
}

/**
 * @brief Initialize test particles from pitch angle and energy.
 *
 * @param EMf Field object used to sample reference density.
 */
void ParticleSoAHost::pitch_angle_energy(Field* EMf)
{
  assert_eq(getNOP(), 0);

  const int nxr = numCellsX_ - 2;
  const int nyr = numCellsY_ - 2;
  const int nzr = numCellsZ_ - 2;
  const int nop = nxr * nyr * nzr * numParticlesPerCell_;
  const auto baseSequence = trackParticleID_
      ? particleIDGenerator_.reserveHostSequenceBlock(nop)
      : ParticleIDGenerator::counter_type{0};
  const double chargeFactor = (chargeOverMass_ / fabs(chargeOverMass_)) * grid_->getVOL() / numParticlesPerCell_;

  prepareSoAForNOP(nop);

  #pragma omp parallel
  {
    const int tid = omp_get_thread_num();
    std::mt19937_64 rng(vct_->getCartesian_rank() * 31 + speciesNumber_ * 127 + tid * 1049);

    #pragma omp for collapse(3) schedule(static)
    for (int i = 1; i < numCellsX_ - 1; i++)
    for (int j = 1; j < numCellsY_ - 1; j++)
    for (int k = 1; k < numCellsZ_ - 1; k++)
    {
      const int cellIdx = ((i - 1) * nyr + (j - 1)) * nzr + (k - 1);
      const int idxBase = cellIdx * numParticlesPerCell_;
      // charge following electron (species 0)
      const double chargePerParticle = chargeFactor * EMf->getRHOcs(i, j, k, 0);

      for (int ii = 0; ii < numPclPerCellX_; ii++)
      for (int jj = 0; jj < numPclPerCellY_; jj++)
      for (int kk = 0; kk < numPclPerCellZ_; kk++)
      {
        const int subIdx = (ii * numPclPerCellY_ + jj) * numPclPerCellZ_ + kk;
        const int idx = idxBase + subIdx;

        // velocity — assumes B is along z
        const double totalMomentum    = sqrt((energy_ + 1) * (energy_ + 1) - 1);
        const double parallelVelocity = totalMomentum * cos(pitchAngle_);
        const double perpMomentum     = totalMomentum * sin(pitchAngle_);
        const double gyroPhase        = 2 * M_PI * sample_u_double(rng);
        const double velX             = perpMomentum * cos(gyroPhase);
        const double velY             = perpMomentum * sin(gyroPhase);

        u[idx] = velX;
        v[idx] = velY;
        w[idx] = parallelVelocity;
        q[idx] = chargePerParticle;
        x[idx] = (ii + .5) * (gridSpacingX_ / numPclPerCellX_) + grid_->getXN(i, j, k);
        y[idx] = (jj + .5) * (gridSpacingY_ / numPclPerCellY_) + grid_->getYN(i, j, k);
        z[idx] = (kk + .5) * (gridSpacingZ_ / numPclPerCellZ_) + grid_->getZN(i, j, k);
        if (trackParticleID_)
          id[idx] = particleIDGenerator_.idFromSequence(baseSequence + idx);
      }
    }
  }

  if (vct_->getCartesian_rank() == 0) {
    cout << "------------------------------------------" << endl;
    cout << "Initialize Test Particle " << speciesNumber_
         << " with pitch angle " << pitchAngle_
         << ", energy " << energy_
         << ", qom " << chargeOverMass_
         << ", npcel " << nop << endl;
    cout << "------------------------------------------" << endl;
  }
}

/**
 * @brief Force-free particle initialization stub.
 *
 * The solver can select this path, but the actual particle initialization logic
 * is intentionally left unimplemented in the current code.
 *
 * @param EMf Field object passed through from the solver.
 */
void ParticleSoAHost::force_free(Field* EMf)
{
  eprintf("force_free was not properly implemented and needs to be revised.");
}

/**
 * @brief Load this species from the restart file into the host SoA arrays.
 */
void ParticleSoAHost::restartLoad()
{
  col_->read_particles_restart(vct_, speciesNumber_, u, v, w, q, x, y, z, id);
  if (!trackParticleID_) {
    id.clear();
    return;
  }

  const int nop = getNOP();
  if (static_cast<int>(id.size()) != nop) {
    const int oldSize = id.size();
    id.resize(nop);
    for (int i = oldSize; i < nop; ++i) {
      id[i] = PARTICLE_ID_INVALID;
    }
  }

  particleIDGenerator_.seedFromExistingIDs(nop > 0 ? id.getList() : nullptr,
                                           nop, mpiComm_);
  for (int i = 0; i < nop; ++i) {
    if (id[i] == PARTICLE_ID_INVALID) {
      id[i] = particleIDGenerator_.generateHostID();
    }
  }
}

// ======= Diagnostics =======

/**
 * @brief Compute the MPI-reduced maximum particle speed for this species.
 */
double ParticleSoAHost::getMaxVelocity() const
{
  double localMaxVel = 0.0;
  const int numParticles = getNOP();
  #pragma omp parallel for reduction(max:localMaxVel)
  for (int idx = 0; idx < numParticles; idx++) {
    const double velX = u[idx], velY = v[idx], velZ = w[idx];
    localMaxVel = std::max(localMaxVel, sqrt(velX*velX + velY*velY + velZ*velZ));
  }
  double globalMaxVel = 0.0;
  MPI_Allreduce(&localMaxVel, &globalMaxVel, 1, MPI_DOUBLE, MPI_MAX, mpiComm_);
  return globalMaxVel;
}

/**
 * @brief Compute an MPI-reduced speed histogram for this species.
 *
 * The caller owns the returned histogram buffer.
 *
 * @param numBins Number of histogram bins.
 * @param maxVelocity Maximum speed represented by the histogram.
 * @return Newly allocated histogram buffer owned by the caller.
 */
long long* ParticleSoAHost::getVelocityDistribution(int numBins, double maxVelocity) const
{
  long long* histogram = new long long[numBins];
  for (int bin = 0; bin < numBins; bin++)
    histogram[bin] = 0;

  const double binWidth = maxVelocity / numBins;
  const int numParticles = getNOP();

  #pragma omp parallel
  {
    long long* localHistogram = new long long[numBins]();
    #pragma omp for nowait
    for (int idx = 0; idx < numParticles; idx++) {
      const double velX = u[idx], velY = v[idx], velZ = w[idx];
      const double speed = sqrt(velX*velX + velY*velY + velZ*velZ);
      int bin = static_cast<int>(floor(speed / binWidth));
      if (bin >= numBins)
        localHistogram[numBins - 1] += 1;
      else
        localHistogram[bin] += 1;
    }
    #pragma omp critical
    {
      for (int bin = 0; bin < numBins; bin++)
        histogram[bin] += localHistogram[bin];
    }
    delete[] localHistogram;
  }
  MPI_Allreduce(MPI_IN_PLACE, histogram, numBins, MPI_LONG_LONG, MPI_SUM, mpiComm_);
  return histogram;
}

// ======= Cell-sorted reorder =======

/**
 * @brief Reorder particles by cell index using a serial counting sort.
 */
void ParticleSoAHost::sort_particles_serial()
{
  const int numParticles = getNOP();
  if (numParticles == 0) return;

  Larray<double> uSorted(numParticles), vSorted(numParticles), wSorted(numParticles), qSorted(numParticles);
  Larray<double> xSorted(numParticles), ySorted(numParticles), zSorted(numParticles);
  Larray<cudaPclType_ID> idSorted;
  uSorted.resize(numParticles); vSorted.resize(numParticles);
  wSorted.resize(numParticles); qSorted.resize(numParticles);
  xSorted.resize(numParticles); ySorted.resize(numParticles);
  zSorted.resize(numParticles);
  if (trackParticleID_) idSorted.resize(numParticles);

  numParticlesInBucket_->setall(0);

  // Pass 1: count particles per cell
  for (int pidx = 0; pidx < numParticles; pidx++) {
    int cellX, cellY, cellZ;
    grid_->get_safe_cell_coordinates(cellX, cellY, cellZ, x[pidx], y[pidx], z[pidx]);
    (*numParticlesInBucket_)[cellX][cellY][cellZ]++;
  }

  // Prefix sum → bucket offsets
  int accumulator = 0;
  for (int cellX = 0; cellX < numCellsX_; cellX++)
  for (int cellY = 0; cellY < numCellsY_; cellY++)
  for (int cellZ = 0; cellZ < numCellsZ_; cellZ++) {
    (*bucketOffset_)[cellX][cellY][cellZ] = accumulator;
    accumulator += (*numParticlesInBucket_)[cellX][cellY][cellZ];
  }
  assert(accumulator == numParticles);

  numParticlesInBucketNow_->setall(0);

  // Pass 2: scatter into sorted order
  for (int pidx = 0; pidx < numParticles; pidx++) {
    int cellX, cellY, cellZ;
    grid_->get_safe_cell_coordinates(cellX, cellY, cellZ, x[pidx], y[pidx], z[pidx]);
    const int destIndex = (*bucketOffset_)[cellX][cellY][cellZ]
                        + (*numParticlesInBucketNow_)[cellX][cellY][cellZ]++;
    uSorted[destIndex] = u[pidx]; vSorted[destIndex] = v[pidx];
    wSorted[destIndex] = w[pidx]; qSorted[destIndex] = q[pidx];
    xSorted[destIndex] = x[pidx]; ySorted[destIndex] = y[pidx];
    zSorted[destIndex] = z[pidx];
    if (trackParticleID_) idSorted[destIndex] = id[pidx];
  }

  u.swap(uSorted); v.swap(vSorted); w.swap(wSorted); q.swap(qSorted);
  x.swap(xSorted); y.swap(ySorted); z.swap(zSorted);
  if (trackParticleID_) id.swap(idSorted);
}

void ParticleSoAHost::sort_particles_parallel(int* cellCount, int* cellOffset)
{
  const int numParticles = getNOP();
  if (numParticles == 0) return;
  assert(grid_ && "sort_particles_parallel requires a grid pointer");

  const int totalCells = numCellsX_ * numCellsY_ * numCellsZ_;
  const int numThreads = omp_get_max_threads();

  Larray<double> uSorted(numParticles), vSorted(numParticles), wSorted(numParticles), qSorted(numParticles);
  Larray<double> xSorted(numParticles), ySorted(numParticles), zSorted(numParticles);
  Larray<cudaPclType_ID> idSorted;
  uSorted.resize(numParticles); vSorted.resize(numParticles);
  wSorted.resize(numParticles); qSorted.resize(numParticles);
  xSorted.resize(numParticles); ySorted.resize(numParticles);
  zSorted.resize(numParticles);
  if (trackParticleID_) idSorted.resize(numParticles);

  std::vector<std::vector<int>> threadLocalCounts(numThreads, std::vector<int>(totalCells, 0));
  std::vector<std::vector<int>> threadLocalOffsets(numThreads, std::vector<int>(totalCells, 0));

  std::fill(cellCount, cellCount + totalCells, 0);
  std::fill(cellOffset, cellOffset + totalCells, 0);

  #pragma omp parallel
  {
    const int threadID = omp_get_thread_num();
    const int blockSize = numParticles / numThreads;
    const int remainder = numParticles % numThreads;
    const int rangeStart = threadID * blockSize + (threadID < remainder ? threadID : remainder);
    const int rangeEnd   = rangeStart + blockSize + (threadID < remainder ? 1 : 0);

    for (int pidx = rangeStart; pidx < rangeEnd; pidx++) {
      int cellX, cellY, cellZ;
      grid_->get_safe_cell_coordinates(cellX, cellY, cellZ, x[pidx], y[pidx], z[pidx]);
      int cellIndex = cellX * (numCellsY_ * numCellsZ_) + cellY * numCellsZ_ + cellZ;
      threadLocalCounts[threadID][cellIndex]++;
    }
    #pragma omp barrier
    #pragma omp single
    {
      int accumulator = 0;
      for (int cell = 0; cell < totalCells; cell++) {
        int localSum = 0;
        for (int tid = 0; tid < numThreads; tid++) {
          threadLocalOffsets[tid][cell] = accumulator + localSum;
          localSum += threadLocalCounts[tid][cell];
        }
        cellCount[cell] = localSum;
        cellOffset[cell] = accumulator;
        accumulator += localSum;
      }
      assert(accumulator == numParticles);
    }
    #pragma omp barrier
    for (int pidx = rangeStart; pidx < rangeEnd; pidx++) {
      int cellX, cellY, cellZ;
      grid_->get_safe_cell_coordinates(cellX, cellY, cellZ, x[pidx], y[pidx], z[pidx]);
      int cellIndex = cellX * (numCellsY_ * numCellsZ_) + cellY * numCellsZ_ + cellZ;
      int destIndex = threadLocalOffsets[threadID][cellIndex]++;
      uSorted[destIndex] = u[pidx]; vSorted[destIndex] = v[pidx];
      wSorted[destIndex] = w[pidx]; qSorted[destIndex] = q[pidx];
      xSorted[destIndex] = x[pidx]; ySorted[destIndex] = y[pidx];
      zSorted[destIndex] = z[pidx];
      if (trackParticleID_) idSorted[destIndex] = id[pidx];
    }
  }

  u.swap(uSorted); v.swap(vSorted); w.swap(wSorted); q.swap(qSorted);
  x.swap(xSorted); y.swap(ySorted); z.swap(zSorted);
  if (trackParticleID_) id.swap(idSorted);
}

// ======= Boundary-condition configuration queries =======

/**
 * @brief Build reemission-boundary configuration for the GPU mover.
 *
 * @param doRepopulateInjection Output global flag enabling repopulation logic.
 * @param doRepopulateInjectionSide Output per-face repopulation flags.
 * @param repopulateBoundary Output per-face repopulation boundary positions.
 */
void ParticleSoAHost::repopulate_particlesInfo(
    bool* doRepopulateInjection,
    bool* doRepopulateInjectionSide,
    cudaCommonType* repopulateBoundary) const
{
  using namespace BCparticles;

  *doRepopulateInjection = false;

  if (!vct_->isBoundaryProcess_P()) return;

  const bool repopBndryX = !vct_->getPERIODICX_P() &&
        (bcPfaceXleft_ == REEMISSION || bcPfaceXright_ == REEMISSION);
  const bool repopBndryY = !vct_->getPERIODICY_P() &&
        (bcPfaceYleft_ == REEMISSION || bcPfaceYright_ == REEMISSION);
  const bool repopBndryZ = !vct_->getPERIODICZ_P() &&
        (bcPfaceZleft_ == REEMISSION || bcPfaceZright_ == REEMISSION);

  if (!(repopBndryX || repopBndryY || repopBndryZ)) return;

  const bool repopXleft = (vct_->noXleftNeighbor_P() && bcPfaceXleft_ == REEMISSION);
  const bool repopYleft = (vct_->noYleftNeighbor_P() && bcPfaceYleft_ == REEMISSION);
  const bool repopZleft = (vct_->noZleftNeighbor_P() && bcPfaceZleft_ == REEMISSION);
  const bool repopXrght = (vct_->noXrghtNeighbor_P() && bcPfaceXright_ == REEMISSION);
  const bool repopYrght = (vct_->noYrghtNeighbor_P() && bcPfaceYright_ == REEMISSION);
  const bool repopZrght = (vct_->noZrghtNeighbor_P() && bcPfaceZright_ == REEMISSION);

  if (!(repopXleft || repopYleft || repopZleft || repopXrght || repopYrght || repopZrght))
    return;

  *doRepopulateInjection = true;

  const int numLayers = 3;
  const double xLow = numLayers * gridSpacingX_;
  const double yLow = numLayers * gridSpacingY_;
  const double zLow = numLayers * gridSpacingZ_;
  const double xHgh = domainLengthX_ - xLow;
  const double yHgh = domainLengthY_ - yLow;
  const double zHgh = domainLengthZ_ - zLow;

  doRepopulateInjectionSide[0] = repopXleft;
  doRepopulateInjectionSide[1] = repopXrght;
  doRepopulateInjectionSide[2] = repopYleft;
  doRepopulateInjectionSide[3] = repopYrght;
  doRepopulateInjectionSide[4] = repopZleft;
  doRepopulateInjectionSide[5] = repopZrght;

  repopulateBoundary[0] = xLow;
  repopulateBoundary[1] = xHgh;
  repopulateBoundary[2] = yLow;
  repopulateBoundary[3] = yHgh;
  repopulateBoundary[4] = zLow;
  repopulateBoundary[5] = zHgh;
}

/**
 * @brief Build open-boundary outflow configuration for the GPU mover.
 *
 * @param doOpenBC Output global flag enabling open-boundary logic.
 * @param applyOpenBC Output per-face open-boundary enable flags.
 * @param deleteBoundary Output per-face delete-boundary positions.
 * @param openBoundary Output per-face open-boundary positions.
 */
void ParticleSoAHost::openbc_particles_outflowInfo(
    bool* doOpenBC, bool* applyOpenBC,
    cudaCommonType* deleteBoundary, cudaCommonType* openBoundary) const
{
  using namespace BCparticles;

  *doOpenBC = false;
  if (!vct_->isBoundaryProcess_P()) return;

  const bool openXleft  = !vct_->getPERIODICX_P() && vct_->noXleftNeighbor_P() && bcPfaceXleft_  == OPENBCOut;
  const bool openYleft  = !vct_->getPERIODICY_P() && vct_->noYleftNeighbor_P() && bcPfaceYleft_  == OPENBCOut;
  const bool openZleft  = !vct_->getPERIODICZ_P() && vct_->noZleftNeighbor_P() && bcPfaceZleft_  == OPENBCOut;
  const bool openXright = !vct_->getPERIODICX_P() && vct_->noXrghtNeighbor_P() && bcPfaceXright_ == OPENBCOut;
  const bool openYright = !vct_->getPERIODICY_P() && vct_->noYrghtNeighbor_P() && bcPfaceYright_ == OPENBCOut;
  const bool openZright = !vct_->getPERIODICZ_P() && vct_->noZrghtNeighbor_P() && bcPfaceZright_ == OPENBCOut;

  if (!(openXleft || openYleft || openZleft || openXright || openYright || openZright))
    return;

  *doOpenBC = true;

  applyOpenBC[0] = openXleft;
  applyOpenBC[1] = openXright;
  applyOpenBC[2] = openYleft;
  applyOpenBC[3] = openYright;
  applyOpenBC[4] = openZleft;
  applyOpenBC[5] = openZright;

  const int numLayers = 3;
  const double xLow = numLayers * gridSpacingX_;
  const double yLow = numLayers * gridSpacingY_;
  const double zLow = numLayers * gridSpacingZ_;
  const double xHgh = domainLengthX_ - xLow;
  const double yHgh = domainLengthY_ - yLow;
  const double zHgh = domainLengthZ_ - zLow;

  deleteBoundary[0] = 0;               deleteBoundary[1] = domainLengthX_;
  deleteBoundary[2] = 0;               deleteBoundary[3] = domainLengthY_;
  deleteBoundary[4] = 0;               deleteBoundary[5] = domainLengthZ_;
  openBoundary[0] = xLow;              openBoundary[1] = xHgh;
  openBoundary[2] = yLow;              openBoundary[3] = yHgh;
  openBoundary[4] = zLow;              openBoundary[5] = zHgh;
}

/**
 * @brief Fill per-face EXIT-boundary flags for the GPU mover.
 *
 * @param isExitBC Output per-face flags; true means exiting particles are deleted locally.
 */
void ParticleSoAHost::fillExitBCFlags(bool* isExitBC) const
{
  using namespace BCparticles;
  isExitBC[0] = (!vct_->getPERIODICX_P() && vct_->noXleftNeighbor_P() && bcPfaceXleft_  == EXIT);
  isExitBC[1] = (!vct_->getPERIODICX_P() && vct_->noXrghtNeighbor_P() && bcPfaceXright_ == EXIT);
  isExitBC[2] = (!vct_->getPERIODICY_P() && vct_->noYleftNeighbor_P() && bcPfaceYleft_  == EXIT);
  isExitBC[3] = (!vct_->getPERIODICY_P() && vct_->noYrghtNeighbor_P() && bcPfaceYright_ == EXIT);
  isExitBC[4] = (!vct_->getPERIODICZ_P() && vct_->noZleftNeighbor_P() && bcPfaceZleft_  == EXIT);
  isExitBC[5] = (!vct_->getPERIODICZ_P() && vct_->noZrghtNeighbor_P() && bcPfaceZright_ == EXIT);
}

// ======= AoS append path =======

/**
 * @brief Append externally produced AoS particles into the host SoA arrays.
 *
 * @param buffer Input AoS particle buffer.
 * @param count Number of particles to append from @p buffer.
 */
void ParticleSoAHost::appendFromAoS(const SpeciesParticle* buffer, int count)
{
  if (count <= 0) return;
  const int oldNOP = getNOP();
  const int newNOP = oldNOP + count;
  const int padded = roundup_to_multiple(newNOP, DVECWIDTH);
  u.reserve(padded); v.reserve(padded); w.reserve(padded); q.reserve(padded);
  x.reserve(padded); y.reserve(padded); z.reserve(padded);
  if (trackParticleID_) id.reserve(padded);
  for (int idx = 0; idx < count; idx++) {
    u.push_back(buffer[idx].get_u());
    v.push_back(buffer[idx].get_v());
    w.push_back(buffer[idx].get_w());
    q.push_back(buffer[idx].get_q());
    x.push_back(buffer[idx].get_x());
    y.push_back(buffer[idx].get_y());
    z.push_back(buffer[idx].get_z());
    if (trackParticleID_) id.push_back(buffer[idx].get_id());
  }
}
