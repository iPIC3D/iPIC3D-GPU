#ifndef EM_FIELDS_3D_H
#define EM_FIELDS_3D_H

/* iPIC3D was originally developed by Stefano Markidis and Giovanni Lapenta.
 * This release was contributed by Alec Johnson and Ivy Bo Peng.
 * Publications that use results from iPIC3D need to properly cite
 * 'S. Markidis, G. Lapenta, and Rizwan-uddin. "Multi-scale simulations of
 * plasma with iPIC3D." Mathematics and Computers in Simulation 80.7 (2010):
 * 1509-1519.'
 *
 *        Copyright 2015 KTH Royal Institute of Technology
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *         http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/**
 * @file EMfields3D.h
 * @brief Electromagnetic field storage and solver interface for one local
 * domain.
 */

#include "Alloc.h"
#include "Basic.h"
#include "asserts.h"
#include "ipicfwd.h"
#include "mpi.h"

#include <cstddef>
#include <memory>
#include <vector>

#include "HeatFluxComponents.h"
#include "cudaTypeDef.cuh"

#include "GPUCleaning.cuh"
#include "GPUFieldArray.cuh"
#include "GPUFieldBoundary.cuh"
#include "GPUHaloComm.cuh"
#include "GPUMaxwellLocal.cuh"
#if defined(IPIC3D_GPU_CYCLE_DIAGNOSTICS)
#include "GPUCycleDiagnostics.cuh"
#endif

struct FieldOutputArrays; // OutputTagConfig.h

/**
 * @brief Electromagnetic fields, sources, and implicit-solver work arrays for
 * one MPI rank.
 *
 * The solver uses this class to initialize case-dependent fields, advance the
 * implicit Maxwell system, reduce the particle moments deposited by the moment
 * kernels (ghost exchange, species sums, hat functions), and hold the device
 * field arrays that the particle mover's field packing reads (gpuExForPcls(),
 * gpuBxn(), ...).
 */
class EMfields3D // :public Field
{
public:
  /** Halo exchanges of the field solver (input HaloOverlap): Blocking
   *  exchanges, then computes; Overlap computes what does not read the
   *  exchanged values while the exchange is in flight.  gpuCalculateE/B/
   *  HatFunctions, the smoothing routines and gpuMaxwellImage are compiled
   *  for both modes (template parameter): the public gpuCalculate* run the
   *  mode of the run, the public gpuMaxwellImage the one of
   *  gpuMaxwellImageFn(), the cleaning images (gpuCleaningImage) Overlap
   *  where their rows split (Lap27Rows::split).
   *  The other exchanges (Maxwell source, the rest of the cleanings) are
   *  blocking, and a div(B) cleaning cycle runs the B update blocking. */
  enum class HaloMode { Blocking, Overlap };

  /** Outcome of a Krylov solve, read by the field-solver guard after the
   *  solve: whether the true residual met the tolerance, and the ||b|| the
   *  solver computes anyway for its status line (-1 when not computed). */
  struct KrylovSolveStatus {
    bool converged = true;
    double normB = -1.0;
  };

  // ======= Construction and lifetime =======

  /**
   * @brief Construct the host field storage for the local domain; the GPU
   * solver arrays are allocated by gpuSolverAllocate().
   *
   * @param col Collective input/configuration object.
   * @param grid Local grid descriptor.
   * @param vct MPI topology descriptor.
   */
  EMfields3D(Collective* col, Grid* grid, VirtualTopology3D* vct);
  /** @brief Destroy the field container and its owned work buffers. */
  ~EMfields3D();

  // ======= Solver-facing initialization routines =======

  /** @brief Initialize the default field state or restart-provided state. */
  void init();
  /**
   * @brief Initialize the beam test configuration (not used: no Case selects
   * it).
   *
   * @param x_center Beam-center x coordinate.
   * @param y_center Beam-center y coordinate.
   * @param z_center Beam-center z coordinate.
   * @param radius Beam radius.
   */
  void initBEAM(double x_center, double y_center, double z_center,
                double radius);
  /** @brief Initialize the standard GEM challenge configuration. */
  void initGEM();
  /** @brief Single Harris sheet with optional GEM/hump perturbations and
   * optional Ampère current. */
  void initGEMHarris();
  /** Not used: no Case selects it. */
  void initOriginalGEM();
  /** @brief Initialize the double-Harris-sheet GEM configuration. */
  void initGEMDoubleHarris();
  /** Not used: no Case selects it. */
  void initDoublePeriodicHarrisWithGaussianHumpPerturbation();
  /** @brief Initialize the hump-perturbation configuration. */
  void initHumpPerturbation();
  /** @brief Initialize GEM with a dipole-like tail and no perturbation (not
   * used: no Case selects it). */
  void initGEMDipoleLikeTailNoPert();
  /** @brief Initialize GEM without the perturbation term. */
  void initGEMnoPert();
#ifdef BATSRUS
  /** @brief Initialize fields from BATSRUS input data. */
  void initBATSRUS();
#endif
  /** @brief Initialize the random-field test case. */
  void initRandomField();
  /** @brief Initialize the force-free equilibrium case. */
  void initForceFree();
  /**
   * @brief Initialize a uniform rotated magnetic field (not used).
   *
   * @param B Magnetic-field magnitude.
   * @param theta Rotation angle.
   */
  void initEM_rotate(double B, double theta);
  /**
   * @brief Add a perturbation cos(kx x + ky y + phase) to the densities of
   * species 0 and 1 (not used).
   *
   * @param deltaBoB Magnetic perturbation amplitude normalized by the
   * background field.
   * @param kx Perturbation wave number in x.
   * @param ky Perturbation wave number in y.
   * @param Bx_mod, By_mod, Bz_mod Set only the normalisation |B_mod|.
   * @param ne_mod Electron-density perturbation amplitude.
   * @param ne_phase Electron-density perturbation phase.
   * @param ni_mod Ion-density perturbation amplitude.
   * @param ni_phase Ion-density perturbation phase.
   * @param B0 Background magnetic-field magnitude.
   * @param grid Local grid descriptor.
   */
  void AddPerturbationRho(double deltaBoB, double kx, double ky, double Bx_mod,
                          double By_mod, double Bz_mod, double ne_mod,
                          double ne_phase, double ni_mod, double ni_phase,
                          double B0, Grid* grid);
  /**
   * @brief Add a perturbation to the electromagnetic field (not used).
   *
   * @param deltaBoB Magnetic perturbation amplitude normalized by the
   * background field.
   * @param kx Perturbation wave number in x.
   * @param ky Perturbation wave number in y.
   * @param Ex_mod Ex perturbation amplitude.
   * @param Ex_phase Ex perturbation phase.
   * @param Ey_mod Ey perturbation amplitude.
   * @param Ey_phase Ey perturbation phase.
   * @param Ez_mod Ez perturbation amplitude.
   * @param Ez_phase Ez perturbation phase.
   * @param Bx_mod Bx perturbation amplitude.
   * @param Bx_phase Bx perturbation phase.
   * @param By_mod By perturbation amplitude.
   * @param By_phase By perturbation phase.
   * @param Bz_mod Bz perturbation amplitude.
   * @param Bz_phase Bz perturbation phase.
   * @param B0 Background magnetic-field magnitude.
   * @param grid Local grid descriptor.
   */
  void AddPerturbation(double deltaBoB, double kx, double ky, double Ex_mod,
                       double Ex_phase, double Ey_mod, double Ey_phase,
                       double Ez_mod, double Ez_phase, double Bx_mod,
                       double Bx_phase, double By_mod, double By_phase,
                       double Bz_mod, double Bz_phase, double B0, Grid* grid);
  /** @brief Initialize the 3D magnetic-dipole planetary case. */
  void initDipole();
  /** @brief Initialize the 2D magnetic-dipole planetary case. */
  void initDipole2D();
  /** @brief Initialize the magnetic-null-points configuration. */
  void initNullPoints();
  /** @brief Initialize the Taylor-Green configuration. */
  void initTaylorGreen();

  // ======= Host-side moment helpers (initialization, restart, output) =======

  /**
   * @brief Constant-charge planet (Dipole, Dipole2D): the species densities
   *        set to planetChargeDensity on the nodes of planetNodes.
   */
  void ConstantChargePlanet();
  /** @brief Impose a constant charge in the open-boundary layers. */
  void ConstantChargeOpenBC();
  /** @brief Zero all density-related fields (used by the unit tests only). */
  void setZeroDensities();
  /** @brief Zero the per-species primary moments. */
  void setZeroPrimaryMoments();
  /** @brief Zero the aggregate moments derived from the primary moments. */
  void setZeroDerivedMoments();
  /** @brief Sum nodal current density over all species. */
  void sumOverSpeciesJ();
  /**
   * @brief Communicate per-species heat-flux tensor components for output.
   *
   * Sums the shared nodes across ranks, doubles the nodes on non-periodic
   * particle faces (adjustNonPeriodicHeatFlux), then fills the ghost nodes.
   *
   * @param is Species index.
   */
  void communicateGhostHeatFlux(int is);

  /** @brief Double the heat flux on the nodes of non-periodic particle faces,
   * which collect deposits from one side only. */
  void adjustNonPeriodicHeatFlux(int is);

  /*** accessor methods ***/

  /*! get Potential array */
  arr3_double getPHI() { return PHI; }

  // field components defined on nodes
  //
  double getEx(int X, int Y, int Z) const { return Ex.get(X, Y, Z); }
  double getEy(int X, int Y, int Z) const { return Ey.get(X, Y, Z); }
  double getEz(int X, int Y, int Z) const { return Ez.get(X, Y, Z); }
  double getBx(int X, int Y, int Z) const { return Bxn.get(X, Y, Z); }
  double getBy(int X, int Y, int Z) const { return Byn.get(X, Y, Z); }
  double getBz(int X, int Y, int Z) const { return Bzn.get(X, Y, Z); }
  //
  arr3_double getEx() { return Ex; }
  arr3_double getEy() { return Ey; }
  arr3_double getEz() { return Ez; }
  arr3_double getBx() { return Bxn; }
  arr3_double getBy() { return Byn; }
  arr3_double getBz() { return Bzn; }

  // cell-centred B (used by the unit tests only)
  arr3_double getBxc() { return Bxc; };
  arr3_double getByc() { return Byc; };
  arr3_double getBzc() { return Bzc; };

  arr3_double getRHOn() { return rhon; }
  double getRHOn(int X, int Y, int Z) const { return rhon.get(X, Y, Z); }

  // densities per species:
  //
  double getRHOcs(int X, int Y, int Z, int is) const {
    return rhocs.get(is, X, Y, Z);
  }
  double getRHOns(int X, int Y, int Z, int is) const {
    return rhons.get(is, X, Y, Z);
  }
  arr4_double getRHOns() { return rhons; }
  arr4_double getRHOcs() { return rhocs; }
  // per-species 3D slice accessors
  arr3_double getRHOcs(int is) {
    return arr3_double(rhocs.fetch_arr4()[is], nxc, nyc, nzc);
  }
  arr3_double getRHOns(int is) {
    return arr3_double(rhons.fetch_arr4()[is], nxn, nyn, nzn);
  }

  // B_ext accessors (used by the unit tests only)
  double getBx_ext(int X, int Y, int Z) const { return Bx_ext.get(X, Y, Z); }
  double getBy_ext(int X, int Y, int Z) const { return By_ext.get(X, Y, Z); }
  double getBz_ext(int X, int Y, int Z) const { return Bz_ext.get(X, Y, Z); }

  arr3_double getBx_ext() { return Bx_ext; }
  arr3_double getBy_ext() { return By_ext; }
  arr3_double getBz_ext() { return Bz_ext; }

  // B_tot = B + B_ext
  arr3_double getBxTot() {
    addscale(1.0, Bxn, Bx_ext, Bx_tot, nxn, nyn, nzn);
    return Bx_tot;
  }
  arr3_double getByTot() {
    addscale(1.0, Byn, By_ext, By_tot, nxn, nyn, nzn);
    return By_tot;
  }
  arr3_double getBzTot() {
    addscale(1.0, Bzn, Bz_ext, Bz_tot, nxn, nyn, nzn);
    return Bz_tot;
  }
  double getBxTot(int X, int Y, int Z) const {
    return Bxn.get(X, Y, Z) + Bx_ext.get(X, Y, Z);
    ;
  }
  double getByTot(int X, int Y, int Z) const {
    return Byn.get(X, Y, Z) + By_ext.get(X, Y, Z);
  }
  double getBzTot(int X, int Y, int Z) const {
    return Bzn.get(X, Y, Z) + Bz_ext.get(X, Y, Z);
  }

  arr4_double getpXXsn() { return pXXsn; }
  double getpXXsn(int X, int Y, int Z, int is) const {
    return pXXsn.get(is, X, Y, Z);
  }

  arr4_double getpXYsn() { return pXYsn; }
  double getpXYsn(int X, int Y, int Z, int is) const {
    return pXYsn.get(is, X, Y, Z);
  }

  arr4_double getpXZsn() { return pXZsn; }
  double getpXZsn(int X, int Y, int Z, int is) const {
    return pXZsn.get(is, X, Y, Z);
  }

  arr4_double getpYYsn() { return pYYsn; }
  double getpYYsn(int X, int Y, int Z, int is) const {
    return pYYsn.get(is, X, Y, Z);
  }

  arr4_double getpYZsn() { return pYZsn; }
  double getpYZsn(int X, int Y, int Z, int is) const {
    return pYZsn.get(is, X, Y, Z);
  }

  arr4_double getpZZsn() { return pZZsn; }
  double getpZZsn(int X, int Y, int Z, int is) const {
    return pZZsn.get(is, X, Y, Z);
  }

  /** Allocates the host heat flux, page-locked; called when the output
   *  writes it (the accessors below are for that case only). */
  void allocateHeatFlux();
  arr4_double getHeatFlux() { return *heatFlux; }
  arr3_double getHeatFluxComponent(int is, int component) {
    return arr3_double(
        heatFlux->fetch_arr4()[HeatFlux::componentIndex(is, component)], nxn,
        nyn, nzn);
  }
  double getHeatFlux(int X, int Y, int Z, int is, int component) const {
    return heatFlux->get(HeatFlux::componentIndex(is, component), X, Y, Z);
  }
  double* getHeatFluxSpeciesPtr(int is) {
    return heatFlux->fetch_arr() +
           HeatFlux::componentIndex(is, 0) * nxn * nyn * nzn;
  }

  // per-species 3D slice accessors (node grid)
  arr3_double getpXXsn(int is) {
    return arr3_double(pXXsn.fetch_arr4()[is], nxn, nyn, nzn);
  }
  arr3_double getpXYsn(int is) {
    return arr3_double(pXYsn.fetch_arr4()[is], nxn, nyn, nzn);
  }
  arr3_double getpXZsn(int is) {
    return arr3_double(pXZsn.fetch_arr4()[is], nxn, nyn, nzn);
  }
  arr3_double getpYYsn(int is) {
    return arr3_double(pYYsn.fetch_arr4()[is], nxn, nyn, nzn);
  }
  arr3_double getpYZsn(int is) {
    return arr3_double(pYZsn.fetch_arr4()[is], nxn, nyn, nzn);
  }
  arr3_double getpZZsn(int is) {
    return arr3_double(pZZsn.fetch_arr4()[is], nxn, nyn, nzn);
  }

  double getJx(int X, int Y, int Z) const { return Jx.get(X, Y, Z); }
  double getJy(int X, int Y, int Z) const { return Jy.get(X, Y, Z); }
  double getJz(int X, int Y, int Z) const { return Jz.get(X, Y, Z); }
  arr3_double getJx() { return Jx; }
  arr3_double getJy() { return Jy; }
  arr3_double getJz() { return Jz; }
  arr4_double getJxs() { return Jxs; }
  arr4_double getJys() { return Jys; }
  arr4_double getJzs() { return Jzs; }
  // per-species 3D slice accessors (node grid)
  arr3_double getJxs(int is) {
    return arr3_double(Jxs.fetch_arr4()[is], nxn, nyn, nzn);
  }
  arr3_double getJys(int is) {
    return arr3_double(Jys.fetch_arr4()[is], nxn, nyn, nzn);
  }
  arr3_double getJzs(int is) {
    return arr3_double(Jzs.fetch_arr4()[is], nxn, nyn, nzn);
  }

  double getJxs(int X, int Y, int Z, int is) const {
    return Jxs.get(is, X, Y, Z);
  }
  double getJys(int X, int Y, int Z, int is) const {
    return Jys.get(is, X, Y, Z);
  }
  double getJzs(int X, int Y, int Z, int is) const {
    return Jzs.get(is, X, Y, Z);
  }

  /*! electric field energy of this rank: sum over its owned nodes of
   *  w dV |E|^2 / (8 pi), w the trapezoidal weight (ownedNodeWeights) */
  double getLocalEenergy();
  /*! magnetic field energy of this rank: sum over its cells of
   *  dV |B_c + B_ext|^2 / (8 pi), with the cell-centred B that Faraday's law
   *  advances (the node B is its interpolation, which filters the
   *  grid-scale part) and B_ext averaged from the 8 corner nodes */
  double getLocalBenergy();
  /*! electric field energy, summed over the ranks */
  double getEenergy();
  /*! magnetic field energy, summed over the ranks */
  double getBenergy();
  /*! bulk kinetic energy of species is, summed over the ranks (owned nodes
   *  with the weights of getLocalEenergy) */
  double getBulkEnergy(int is);
  /*! Last node this rank owns along x, y, z; the owned nodes are [1, end].
   *  end = n-3 where the next rank or the periodic image holds node n-2,
   *  end = n-2 on a non-periodic last rank (node n-2 is the boundary node),
   *  so that every global node is counted once. */
  void ownedNodeEnds(int& iEnd, int& jEnd, int& kEnd) const;
  /*! Volume weights of the owned nodes along x, y, z (w[d][1..end]): 1, or
   *  1/2 on a node of a non-periodic domain face, which holds half a cell
   *  of the domain (trapezoidal rule). */
  void ownedNodeWeights(std::vector<double> w[3]) const;

  /*! print electromagnetic fields info (not used; empty) */
  void print(void) const;

  // get MPI Derived Datatype
  MPI_Datatype getYZFacetype(bool isCenterFlag) {
    return isCenterFlag ? yzFacetypeC : yzFacetypeN;
  }
  MPI_Datatype getXZFacetype(bool isCenterFlag) {
    return isCenterFlag ? xzFacetypeC : xzFacetypeN;
  }
  MPI_Datatype getXYFacetype(bool isCenterFlag) {
    return isCenterFlag ? xyFacetypeC : xyFacetypeN;
  }
  MPI_Datatype getXEdgetype(bool isCenterFlag) {
    return isCenterFlag ? xEdgetypeC : xEdgetypeN;
  }
  MPI_Datatype getYEdgetype(bool isCenterFlag) {
    return isCenterFlag ? yEdgetypeC : yEdgetypeN;
  }
  MPI_Datatype getZEdgetype(bool isCenterFlag) {
    return isCenterFlag ? zEdgetypeC : zEdgetypeN;
  }
  MPI_Datatype getXEdgetype2(bool isCenterFlag) {
    return isCenterFlag ? xEdgetypeC2 : xEdgetypeN2;
  }
  MPI_Datatype getYEdgetype2(bool isCenterFlag) {
    return isCenterFlag ? yEdgetypeC2 : yEdgetypeN2;
  }
  MPI_Datatype getZEdgetype2(bool isCenterFlag) {
    return isCenterFlag ? zEdgetypeC2 : zEdgetypeN2;
  }
  MPI_Datatype getCornertype(bool isCenterFlag) {
    return isCenterFlag ? cornertypeC : cornertypeN;
  }

  MPI_Datatype getProcview() { return procview; }
  MPI_Datatype getXYZeType() { return xyzcomp; }
  MPI_Datatype getProcviewXYZ() { return procviewXYZ; }
  MPI_Datatype getGhostType() { return ghosttype; } // not used

  void freeDataType();
  bool isLittleEndian() { return lEndFlag; };

  // ---- GPU Solver: lifecycle ----
  /** Allocate all GPU-resident field arrays (called once after construction).
   */
  void gpuSolverAllocate();
  /** Free all GPU-resident field arrays. */
  void gpuSolverFree();

  /** Return the dedicated solver CUDA stream. */
  cudaStream_t gpuSolverStream() const { return solverStream_; }

  // ---- GPU Solver: host ↔ device synchronisation ----
  /** Copy all primary field arrays from host to device (for initialisation /
   * restart), then the boundary of the initial fields, whatever the case
   * left in the ghosts: E_t = 0 on a conductor, the ghosts of E and the node
   * B (ENode, BNode), the conductor wall values W from that node B
   * (FieldBoundary::initWall), the ghosts of the cell B (BCellStart; the
   * first gpuMaxwellSource reads them without an exchange of its own). */
  void gpuSolverSyncH2D(cudaStream_t stream = 0);
  /** Copy the given groups of the fields and moments to the host arrays
   * (what the output, the diagnostics, a checkpoint or Catalyst read at a
   * cycle), and wait for them. */
  void gpuSolverSyncD2H(const FieldOutputArrays& arrays,
                        cudaStream_t stream = 0);
  // ---- GPU Solver: halo exchange and boundary of the fields ----
  /** The field boundary conditions (GPUFieldBoundary.cuh): the spec, built
   *  at construction, and the GPU plan, built in gpuSolverAllocate. */
  const FieldBoundary& gpuBoundary() const { return bc_; }
  /** The one halo entry of the field arrays: exchange the n arrays p with
   *  the neighbours (facesOnly: the 6 face boxes, for box stencils), then
   *  fill the ghosts of the physical faces with the rules of set s (one
   *  launch).  The set gives the geometry (cells or nodes) and the
   *  topology (field or particle communicator). */
  void exchangeAndFill(GhostSet s, cudaSolverType* const* p, int n,
                       bool facesOnly = false);
  /** The same, split: begin queues the exchange (HaloOverlap), end
   *  completes it and fills the ghosts. */
  void beginExchange(GhostSet s, cudaSolverType* const* p, int n,
                     bool facesOnly = false);
  void endExchangeAndFill(GhostSet s, cudaSolverType* const* p, int n);
  /** The boundary of a solved node E (n arrays, triples of components): the
   *  ghost nodes (exchangeAndFill of GhostSet::ENode), then the layers of
   *  the open faces (LayerSet::E: inflow E0 or the absorbing blend, the
   *  X-right copy).  E(n+theta) after the solve, E^n after the div(E)
   *  correction. */
  void finishE(cudaSolverType* const* E, int n);

  // ---- GPU Solver: field-solver methods ----
  // gpuCalculateE, gpuCalculateB and gpuCalculateHatFunctions run the
  // instantiation of the run's HaloMode (input HaloOverlap), except
  // gpuCalculateB on a div(B) cleaning cycle, which runs Blocking.
  /** Solve for E(n+theta): div(E) correction of E^n (PoissonCorrection),
   * implicit Maxwell solve into Exth, and, when E is smoothed, the smoothed
   * E(n+theta) in ExthSmooth; halo exchange and post-solve open-boundary
   * layers (LayerSet::E) of both.  The mover samples
   * gpuExForPcls().  E (d_Ex) still holds E^n afterwards: gpuCalculateB
   * advances it. */
  void gpuCalculateE(int cycle);
  /** E^n -> E(n+1) = (S(E(n+theta)) - (1-theta) E^n) / theta, S the smoothing
   * (identity when off).  theta = 1: E(n+1) = S(E(n+theta)), a buffer swap
   * that leaves E^n's storage as the next solve's work array.  Called at the
   * end of gpuCalculateB, after Faraday's law has used Exth. */
  void gpuAdvanceE();
  /** Complete the step: B by Faraday's law from the unsmoothed Exth, its
   * boundary conditions and case fixes, div(B) cleaning of the cell B
   * (divBCorrection), interpolation to the nodes, then E to n+1
   * (gpuAdvanceE). */
  void gpuCalculateB(int cycle);
  /** Build the hat quantities (rhohat, Jhat) of the implicit field solve;
   * cycle=-1 denotes initialization. */
  void gpuCalculateHatFunctions(int cycle = -1);
  /** Maxwell operator A*x on device Krylov vectors (runs
   * gpuMaxwellImageFn(), the Krylov callback the field solvers take):
   *   A*x = delt^2 (-lap x - grad div(mu.x)) + mu.x + x
   * on every node, the normal rows of the conductor faces included (through
   * the conductor image of x and mu.x), then the identity rows of the
   * boundary (gpuBoundaryRows).  The cell gradients of x
   * and div(mu.x) are exchanged between ranks in one halo exchange;
   * gpuMaxwellLocalNodeFused writes the Krylov image. */
  void gpuMaxwellImage(cudaSolverType* d_im, cudaSolverType* d_vector);
  /** The boundary rows (identity rows) of the Maxwell image whose vector
   *  d_vectX/Y/Z holds written into d_im: the last step of the full and the
   *  local image (and of the unit tests' reapplication). */
  void gpuBoundaryRows(cudaSolverType* d_im);
  /** The Maxwell image in halo mode M.  Overlap (the split operator) computes
   *  the boundary layers first (with a flat axis: D and the cell arrays
   *  whole) and the interior while the exchange is in flight; it needs
   *  MaxwellImageBoxes::split (aborts otherwise).  Both give the same
   *  values. */
  template <HaloMode M>
  void gpuMaxwellImage(cudaSolverType* d_im, cudaSolverType* d_vector);
  /** A Krylov operator: im = A * vector on device Krylov vectors. */
  using GpuImageFn = void (EMfields3D::*)(cudaSolverType*, cudaSolverType*);
  /** The Maxwell image of this run: the split operator where HaloOverlap is
   *  on and the subdomain allows it (MaxwellImageBoxes::split), otherwise
   *  the blocking one. */
  GpuImageFn gpuMaxwellImageFn() const;
  /** Launch sets of the Maxwell images and gpuMUdot (set by
   *  gpuSolverAllocate); this accessor is used by the unit tests only. */
  const MaxwellImageBoxes& gpuMaxwellImageBoxes() const { return imageBoxes_; }
  /** Maxwell operator, communication-free: A*x for use as
   * preconditioner (the Richardson sweeps of gpuBlockJacobiPrecond, run when
   * BlockJacobiSweeps > 1). Ghost cells are treated as zero, apart from the
   * X-right ghost cell plane on an outflow face and the image ghost cells of
   * the conductor faces (GhostSet::OpCellsLocal), and the boundary rows are
   * those of the full image, so the operator better matches the full
   * Maxwell image. */
  void gpuMaxwellImageLocal(cudaSolverType* d_im, cudaSolverType* d_vector);
  /** Build the RHS of the Maxwell system (result in device Krylov vector);
   * cycle=-1 is used by direct/unit-test callers. */
  void gpuMaxwellSource(cudaSolverType* d_bkrylov, int cycle = -1);

  // ---- GPU Chebyshev Semi-Iterative Solver ----
  /** Full Chebyshev solver (with MPI communication).
   *  Solves  A·x = b  in Krylov space using Chebyshev polynomial acceleration.
   *  @param d_x      [in/out] initial guess → solution (Krylov vector, device)
   *  @param n         Krylov vector length = 3*(nxn-2)*(nyn-2)*(nzn-2)
   *  @param d_b       [in] right-hand side (Krylov vector, device)
   *  @param GpuImage  operator callback A·v  (e.g. gpuMaxwellImageFn())
   *  @param maxIter   number of Chebyshev polynomial steps
   *  @param eigMin    lower bound on eigenvalues of A  (>0)
   *  @param eigMax    upper bound on eigenvalues of A  (>eigMin)
   *  @param fieldcomm MPI communicator for residual norms
   */
  void gpuChebyshevSolve(cudaSolverType* d_x, int n, cudaSolverType* d_b,
                         void (EMfields3D::*GpuImage)(cudaSolverType*,
                                                      cudaSolverType*),
                         int maxIter, cudaSolverType eigMin,
                         cudaSolverType eigMax, MPI_Comm fieldcomm);

  /** GPU point-block Jacobi preconditioner (communication-free).
   *  D is the 3×3 diagonal block of the Maxwell operator at each node; D⁻¹
   *  is computed once per solve.  With one sweep x = ω D⁻¹ b; otherwise
   *  x = D⁻¹ b followed by BlockJacobiSweeps − 1 Richardson sweeps
   *  x += ω D⁻¹ (b − A_local x).
   *  @param d_x  [out] approximate solution (Krylov vector, device)
   *  @param d_b  [in]  right-hand side       (Krylov vector, device)
   */
  void gpuBlockJacobiPrecond(cudaSolverType* d_x, cudaSolverType* d_b);

  /** Allocate the FGMRES Z workspace on first actual FGMRES use. */
  void gpuEnsureFGMRESWorkspace(int m, int n);

  /** GPU FGMRES(m) with communication-free block-Jacobi preconditioner.
   *  Right-preconditioned flexible GMRES: Z[k] = M⁻¹ V[k], w = A Z[k].
   *  Uses gpuBlockJacobiPrecond as the preconditioner.
   */
  KrylovSolveStatus gpuFGMRES_BlockJacobiPrecond(cudaSolverType* d_x, int n,
                                                 cudaSolverType* d_b, int m,
                                                 int max_iter,
                                                 cudaSolverType tol,
                                                 MPI_Comm fieldcomm);

  /** Power iteration to estimate the largest eigenvalue of A.
   *  @param GpuImage  operator callback A·v
   *  @param n         Krylov vector length
   *  @param nIter     number of power iterations (10–20 typical)
   *  @param fieldcomm MPI communicator for reductions
   *  @return          estimate of lambda_max
   */
  cudaSolverType gpuEstimateMaxEigenvalue(
      void (EMfields3D::*GpuImage)(cudaSolverType*, cudaSolverType*), int n,
      int nIter, MPI_Comm fieldcomm);
  /** GPU MUdot: compute μ·E for all species on the interior nodes (after
   *  gpuSolverAllocate). */
  void gpuMUdot(GPUFieldArray3& MUdotX, GPUFieldArray3& MUdotY,
                GPUFieldArray3& MUdotZ, GPUFieldArray3& vX, GPUFieldArray3& vY,
                GPUFieldArray3& vZ);
  /** GPU MUdot on the nodes of points (a PointBox or a PointList). */
  template <typename Points>
  void gpuMUdot(GPUFieldArray3& MUdotX, GPUFieldArray3& MUdotY,
                GPUFieldArray3& MUdotZ, GPUFieldArray3& vX, GPUFieldArray3& vY,
                GPUFieldArray3& vZ, const Points& points);
  /** GPU PIdot: compute π·v for one species (accumulative). */
  void gpuPIdot(GPUFieldArray3& PIX, GPUFieldArray3& PIY, GPUFieldArray3& PIZ,
                GPUFieldArray3& vX, GPUFieldArray3& vY, GPUFieldArray3& vZ,
                int is);

  // ---- GPU Solver: moment post-processing on GPU ----
  /** D2D scatter: copy 10 packed moment arrays from the moment-kernel buffer
   *  (momentsSrc, layout [10][gridSize]) into the per-species slices of
   *  d_rhons, d_Jxs, d_Jys, d_Jzs, d_pXXsn, d_pXYsn, d_pXZsn,
   *  d_pYYsn, d_pYZsn, d_pZZsn.  Pure device-to-device, no host touch. */
  void gpuScatterMomentsD2D(cudaMomentType* momentsSrc, int species,
                            cudaStream_t stream = 0);

#if defined(IPIC3D_GPU_CYCLE_DIAGNOSTICS)
  /** Report the ten resident per-species moment arrays at one checkpoint. */
  void gpuDiagnosticReportSpeciesMoments(const char* stage, int cycle,
                                         int species, cudaStream_t stream);
#endif

  /** Batched ghost exchange of the species moments, HALO_MAX_BATCH/10
   *  species per pass. */
  void gpuCommunicateGhostP2G_AllSpecies(int cycle = -1);

  /** Batched GPU halo exchange of nFields 3D arrays: every ghost box comes
   *  from the neighbour across it (faces, edges, corners), one message per
   *  distinct neighbour rank for all fields, all in flight at once; a ghost
   *  box with no neighbour (MPI_PROC_NULL) is left unchanged.
   *  @param h_fieldPtrs  Host array of nFields device pointers.
   *  @param nFields      Number of field arrays to batch.
   *  @param nx,ny,nz     Canonical local dimensions including ghosts: center
   *                      with offset 0, or node with offset 0/1. Arbitrary
   *                      subarray geometries are not accepted.
   *  @param offsetZero   true = send offset 0, false = send offset 1.  P2G
   *                      accumulation intentionally uses node dimensions with
   *                      offset 0.
   *  @param isFaceOnlyFlag true = the 6 face boxes only, edges and corners
   *                      unchanged.
   *  @param needInterp   true = additive accumulation after exchange: the
   *                      ghosts, which hold the neighbours' values of the
   *                      shared nodes, are added into those nodes.  Needs
   *                      offsetZero and all boxes (not isFaceOnlyFlag).
   *  @param isParticle   true = use particle communicator.
   *  @param stream       CUDA stream for kernels.  With HaloOverlap off,
   *                      successive calls sharing the persistent buffers must
   *                      use this same stream or be externally synchronized. */
  void gpuBatchedHaloExchange(cudaSolverType* const* h_fieldPtrs, int nFields,
                              int nx, int ny, int nz, bool offsetZero,
                              bool isFaceOnlyFlag, bool needInterp,
                              bool isParticle, cudaStream_t stream);

  /** Post the receives, then queue pointer upload and packing on the halo
   *  stream.  The caller then queues dependency-free work on
   *  @p computeStream.  Needs the split exchange resources (HaloOverlap on).
   */
  void gpuBatchedHaloBeginExchange(cudaSolverType* const* h_fieldPtrs,
                                   int nFields, int nx, int ny, int nz,
                                   bool offsetZero, bool isFaceOnlyFlag,
                                   bool isParticle, cudaStream_t computeStream);
  /** Post the sends as soon as packing completes, wait for all messages,
   *  queue the unpack, then make the caller-provided compute stream wait for
   *  completion. */
  void gpuBatchedHaloEndExchange();

  /** Allocate persistent GPU halo exchange buffers. */
  void gpuAllocateHaloBuffers();
  /** Free persistent GPU halo exchange buffers. */
  void gpuFreeHaloBuffers();

  /** Zero d_rhon, the one derived moment that is accumulated; the others are
   * overwritten before they are read (see the definition). */
  void gpuSetZeroDerivedMoments();

  /**
   * @brief Neutralize the species charge on the X-right open outflow layer.
   *
   * The field solve sees the outflow layer (FieldBoundarySpec::outflowLayers)
   * as neutral. With
   * OutflowLayerFields = copy, E and B are copied over it from a reference
   * plane after every solve, so it carries no field gradient and, by Gauss's
   * law, must carry no net charge; with OutflowLayerFields = solved the
   * solver's field, that of a neutral layer, is kept there.  The species
   * densities on the layer's node planes are replaced by those of the
   * reference plane, rescaled to a neutral pair of magnitude
   * (rho_+ + |rho_-|)/2.  Call after the planet charge fix and before
   * gpuSumOverSpecies().  Does nothing on ranks that do not own an outflow
   * face.
   */
  void gpuNeutralizeOutflowLayer();

  /**
   * @brief Copy J_s and the second moments P_s over the X-right outflow layer
   * from the reference plane (zero-gradient continuation, in both
   * OutflowLayerFields modes), so the solver's current is flat across the
   * layer and the smoothing of J-hat does not carry the currents of the layer
   * particles into the kept nodes. Call next to gpuNeutralizeOutflowLayer();
   * output moments on the layer are these copies.
   */
  void gpuCopyOutflowLayerMoments();

  /** rhon += sum_s(rhons_s) on device. */
  void gpuSumOverSpecies();

  /** Jx/Jy/Jz = sum_s(Jxs/Jys/Jzs)_s on device. */
  void gpuSumOverSpeciesJ();

  /** rhoc = interpN2C(rhon) on device, through the conductor image (the
   *  face-node charge read as 0; FieldBoundary::imageWalls). */
  void gpuInterpDensitiesN2C();

  // ---- GPU Solver: ConstantChargePlanet ----
  /** GPU ConstantChargePlanet (Dipole, Dipole2D): rhons = planetChargeDensity
   *  on the planet nodes (d_planetNodes_, the host planetNodes list). */
  void gpuConstantChargePlanet();

  // ---- GPU Solver: div(E) and div(B) cleaning (GPUCleaning.cuh) ----
  /** 7-point Laplacian image on the centres (with communication).  Not used
   *  currently: the div(E) solve uses gpuCleaningImage. */
  void gpuPoissonImage(cudaSolverType* d_im, cudaSolverType* d_vec);
  /** Communication-free 7-point -∇² (positive eigenvalues): physical faces
   *  use the ghost rule of gpuPoissonImage, a periodic direction with one
   *  cell is flat, rank interfaces are cut with zero ghosts.  Not used
   *  currently. */
  void gpuPoissonImageLocal(cudaSolverType* d_im, cudaSolverType* d_vec);

private:
  /** div(E) cleaning (PoissonCorrection, on its cycles; gpuCalculateE):
   *  solve divN2C(gradC2N phi) = divN2C(E) - 4 pi rho with the ghosts of
   *  GhostSet::Phi (odd on a perfect conductor: phi = 0 on the wall), then
   *  E -= gradC2N phi on the nodes, the values of the identity rows (the
   *  conductor's E_t, inflow E0) restored and the ghost nodes of E
   *  exchanged and filled. */
  void gpuPoissonCorrection(int cycle);
  /** div(B) cleaning (gpuCalculateB on a cleaning cycle, divBCorrection on):
   *  solve divC2N(gradN2C psi) = divC2N(Bc) on the nodes (zero normal
   *  gradient of psi on perfect conductors, psi = 0 on the other physical
   *  faces), then Bc -= gradN2C psi on the cells; restores the ghost cells of
   *  Bc. */
  void gpuApplyDivBCleaning(int cycle);
  /** Chebyshev interval of a local cleaning operator (gpuSolverAllocate). */
  struct ChebBounds {
    double min = 0.0, max = 0.0;
  };
  /** A cleaning solve, set up by gpuSolverAllocate when that cleaning is on;
   *  buf is the array its image unpacks into (d_PHI on the cells, d_PSI on
   *  the nodes), ghosts the set of its exchange (the geometry and the ghost
   *  rules: GhostSet::Phi, GhostSet::Psi). */
  struct CleaningSolve {
    Lap27Op op{};
    Lap27Rows rows{};
    ChebBounds cheb;
    CleaningNullSpace null{};
    GPUFieldArray3* buf = nullptr;
    GhostSet ghosts = GhostSet::Phi;
    const char* name = "";  // "div(E)", "div(B)"
    const char* label = ""; // of the FGMRES output
    GPUKrylovVector x, b;
  };
  bool isDivBCleaningCycle(int cycle) const {
    return divBCorrection && cycle % divBCorrectionCycle == 0;
  }
  /** A flat axis: one periodic cell on one rank (the flat direction of a 2D
   *  run; FieldBoundarySpec::flat). */
  bool flatAxis(int a) const { return bc_.spec().flat[a]; }
  /** Extent with ghosts of the div(B) rows along axis a: the nodes, or the
   *  cells on a flat axis, whose two interior node planes are one periodic
   *  node (Lap27Op::twin). */
  int divBExtent(int a) const {
    const int cells[3] = {nxc, nyc, nzc}, nodes[3] = {nxn, nyn, nzn};
    return flatAxis(a) ? cells[a] : nodes[a];
  }
  /** Slots of d_cleanLines_ for the null-space sums, then as many for the
   *  line coefficients (cleaningNullSpace). */
  int cleanLinesSlots() const { return 1 + nxn + nyn + nzn; }
  /** The Chebyshev interval of a local cleaning operator (printed). */
  ChebBounds cleaningChebBounds(const Lap27Op& op, const char* label);
  /** A field the right-hand side of a cleaning was computed from, weight
   *  times its squared norm adding to the round-off scale. */
  struct RoundOffTerm {
    const cudaSolverType* data;
    size_t n;
    double weight;
  };
  /** Image of the solve cleaning_ points to, in halo mode M: unpack into its
   *  buf (face rules of the rows), exchange the halo and fill the ghosts of
   *  its set (the conductor ghosts of phi follow the iterate), 27-point
   *  image. */
  template <HaloMode M>
  void gpuCleaningImage(cudaSolverType* d_im, cudaSolverType* d_vec);
  /** Its preconditioner: x ≈ L^-1 b (gpuChebyshevCleaning). */
  void gpuCleaningPrecond(cudaSolverType* d_x, cudaSolverType* d_b);
  /** FGMRES(CLEANING_M) of s from x = 0 to divCleaningTol, after removing
   *  the null space from s.b; false (nothing solved) when s.b is at the
   *  round-off of the fields in scale. */
  bool gpuCleaningSolve(CleaningSolve& s,
                        const std::vector<RoundOffTerm>& scale, int cycle);
  /** Subtract from the right-hand side of a cleaning solve its components on
   *  the null space of the operator (ns, CleaningNullSpace). */
  void gpuRemoveCleaningNullSpace(cudaSolverType* d_b,
                                  const CleaningNullSpace& ns);
  /** Whether the right-hand side b is at the round-off of the fields it was
   *  computed from, ||b||² <= (1e3 eps)² sum weight ||field||² over the
   *  ranks: then there is nothing to clean. */
  bool cleaningAtRoundOff(const cudaSolverType* d_b, int n,
                          const std::vector<RoundOffTerm>& fields);

public:
  // ---- Field-solver guard (SolverSourceGrowthLimit,
  //      SolverStopOnNonConvergence) ----
  /** Stop the run if the Maxwell right-hand side norm is not finite or grew
   *  by more than SolverSourceGrowthLimit since the previous E solve.
   *  normB < 0 (not computed by the solver) is not checked. */
  void checkMaxwellSource(double normB, int cycle);
  /** Stop the run if a field solve did not converge and
   *  SolverStopOnNonConvergence is set (the solver has already warned). */
  void checkSolverConverged(bool converged, const char* solve, int cycle);
  /** Print why on rank 0 and abort every rank, before the result of the
   *  failed solve is used and before any further output or checkpoint. */
  [[noreturn]] void stopForSolverFailure(const std::string& reason) const;

  // ---- GPU Solver: accessor helpers ----
  GPUFieldArray3& gpuEx() { return d_Ex; } // unit tests only
  GPUFieldArray3& gpuEy() { return d_Ey; } // unit tests only
  GPUFieldArray3& gpuEz() { return d_Ez; } // unit tests only
  /** The E the mover samples: E(n+theta), smoothed when smoothing is on. */
  GPUFieldArray3& gpuExForPcls() {
    return smoothingOn() ? d_ExthSmooth : d_Exth;
  }
  GPUFieldArray3& gpuEyForPcls() {
    return smoothingOn() ? d_EythSmooth : d_Eyth;
  }
  GPUFieldArray3& gpuEzForPcls() {
    return smoothingOn() ? d_EzthSmooth : d_Ezth;
  }
  /** True when the smoothing of E, rhoc and Jhat changes the fields
   *  (Smooth != 1 and SmoothNiter > 0); otherwise ExthSmooth is neither
   *  allocated nor used. */
  bool smoothingOn() const { return Smooth != 1.0 && SmoothNiter > 0; }
  GPUFieldArray3& gpuExth() { return d_Exth; } // unit tests only
  GPUFieldArray3& gpuEyth() { return d_Eyth; } // unit tests only
  GPUFieldArray3& gpuEzth() { return d_Ezth; } // unit tests only
  GPUFieldArray3& gpuBxc() { return d_Bxc; }   // unit tests only
  GPUFieldArray3& gpuByc() { return d_Byc; }   // unit tests only
  GPUFieldArray3& gpuBzc() { return d_Bzc; }   // unit tests only
  GPUFieldArray3& gpuRhoc() { return d_rhoc; } // unit tests only
  GPUFieldArray3& gpuBxn() { return d_Bxn; }
  GPUFieldArray3& gpuByn() { return d_Byn; }
  GPUFieldArray3& gpuBzn() { return d_Bzn; }
  GPUFieldArray3& gpuBx_ext() { return d_Bx_ext; }
  GPUFieldArray3& gpuBy_ext() { return d_By_ext; }
  GPUFieldArray3& gpuBz_ext() { return d_Bz_ext; }
  GPUFieldArray4& gpuRhons() { return d_rhons; }
  GPUFieldArray4& gpuJxs() { return d_Jxs; }
  GPUFieldArray4& gpuJys() { return d_Jys; }
  GPUFieldArray4& gpuJzs() { return d_Jzs; }

public: // accessors
  const Collective& get_col() const { return _col; }
  const Grid& get_grid() const { return _grid; };
  const VirtualTopology3D& get_vct() const { return _vct; }

private: // initialization helpers
  /** Whether the case has a planet (Dipole, Dipole2D). */
  bool hasPlanet() const;
  /** Whether node (i, j, k) is in the planet: within Planet_radius of the
   *  planet centre, the surface included; planeXZ (Dipole2D) measures the
   *  distance in the x-z plane.  The node position comes from its global
   *  index (Grid3DCU::getGlobalXN), so every decomposition gives the same
   *  answer.  The one mask of the external dipole field (zero in the planet)
   *  and of the planet charge (planetNodes). */
  bool isPlanetNode(int i, int j, int k, bool planeXZ) const;
  /** Flat indices (i * nyn + j) * nzn + k of the local planet nodes
   *  (isPlanetNode), ghosts included.  A ghost gets the answer of the
   *  neighbour's node it copies, except across a periodic face, whose ghost
   *  coordinates are not wrapped.  The host and the GPU constant-charge
   *  models both use this list, so they set the same nodes whatever the
   *  device compiler contracts into multiply-adds.  Page-locked: copied to
   *  the device. */
  PinnedVector<int> planetNodes() const;
  /** sign(q) rhoINIT / 4 pi of species is: its planet charge density. */
  double planetChargeDensity(int is) const;
  /** Zero the species currents Jxs/Jys/Jzs and pressures p**sn. With
   *  fromAmpere, set the species currents to shares of J = (c/4pi) curl(B),
   *  B the cell-centred field: J_s,x = f_s,x J_x with
   *  f_s,x = sign(q_s) u0_s / sum_r sign(q_r) u0_r (v0 for y, w0 for z), so
   *  the drift J_s / (sign(q_s) rho_s) of maxwellianFromSpeciesCurrents is
   *  u0_s J_x / (N_u rho_s), N_u = sum_r sign(q_r) u0_r; a component whose
   *  weights sum to 0 gets no current (ForceFree). */
  void initSpeciesCurrents(bool fromAmpere);

private: // ideal drift field
  /** E0 = -(u0 x B0) / c: the ideal electric field of the plasma drifting at
   *  u0 = (ue0, ve0, we0) (species 0) in B0 = (B0x, B0y, B0z), Gaussian units.
   *  The inflow faces, the conductor walls and the Dipole initial state use
   *  it. */
  void driftElectricField(double E0[3]) const;

private: // X-right open boundary layer
  // Maxwell-operator closure of the outflow face: the ghost node plane of
  // the operator's node arrays copies the face (LayerSet::OutflowNodes), the
  // node-to-cell stencils evaluate the ghost cell plane (kept by the face's
  // rule, GHOST_KEEP), and its ghost rows follow
  // FieldBoundarySpec::outflowRows (FieldBoundary::outflowGhostRows).
  /** Last cell plane of the operator's node-to-cell stencils: the ghost
   *  plane nxc-1 on an outflow face, the interior nxc-2 elsewhere. */
  int operatorCellEnd() const;
  /** X-right outflow face: the ghost node plane of the operator's node
   *  arrays a[0..2] := their face plane (LayerSet::OutflowNodes); no-op
   *  elsewhere. */
  void outflowGhostNodes(GPUFieldArray3* const a[3]);
  /** The 10 cell arrays of the Maxwell operator (full and local image):
   *  the gradients grad_d E_c in cp[3 c + d] and div(mu E) in cp[9]. */
  void operatorCells(cudaSolverType* cp[10]);

private: // solver steps in halo mode M (see HaloMode)
  template <HaloMode M> void gpuCalculateE(int cycle);
  template <HaloMode M> void gpuCalculateB(int cycle);
  template <HaloMode M> void gpuCalculateHatFunctions(int cycle);
  /** SmoothNiter smoothing passes of the N fields src into dst (dst == src:
   *  in place).  The outputs alternate between dst and scratch so that the
   *  last pass writes dst, with no copies (in place with an odd SmoothNiter:
   *  one copy at the end).  Each pass first refreshes the ghosts of its
   *  input: the exchange of the 6 face boxes (the box stencil reads no edge)
   *  and the ghost rules of set ghosts; with HaloMode::Overlap the interior
   *  is smoothed while the exchange is in flight.  fix(out), on the device
   *  pointers of the output, then corrects it before the next pass reads
   *  it. */
  template <HaloMode M, int N, typename Fix>
  void gpuSmoothPasses(GPUFieldArray3* const src[N],
                       GPUFieldArray3* const dst[N],
                       GPUFieldArray3* const scratch[N], int nx, int ny, int nz,
                       GhostSet ghosts, Fix&& fix);
  /** Smoothing in place of the N arrays a (rhoc: N = 1, Jhat: N = 3) with
   *  the rules of ghosts, which also fixes cells or nodes: gpuSmoothPasses
   *  with the scratch d_temp2X/Y/Z viewed with their shape. */
  template <HaloMode M, int N>
  void gpuSmoothInPlace(GPUFieldArray3* const a[N], GhostSet ghosts);
  /** Smoothing of the electric field src (3 components) into dst,
   *  gpuSmoothPasses with the scratch d_tempX/Y/Z.  On physical faces the
   *  ghost nodes are even about the face node (GhostSet::ENodeSmooth), and
   *  every pass restores the values of the identity rows (conductor E_t = 0,
   *  inflow E0: FieldBoundary::rowValues), so those rows keep the values the
   *  solve imposed.  Needs smoothingOn(); src keeps its interior, its ghosts
   *  are overwritten. */
  template <HaloMode M>
  void gpuSmoothE(GPUFieldArray3* const src[3], GPUFieldArray3* const dst[3]);

  /* ********************************* // VARIABLES
   * ********************************* */

private:
  // access to global data
  const Collective& _col;
  const Grid& _grid;
  const VirtualTopology3D& _vct;
  /*! light speed */
  double c;
  /* 4*PI for normalization */
  double FourPI;
  /*! time step */
  double dt;
  /*! decentering parameter */
  double th;
  /*! Smoothing value */
  double Smooth;
  int SmoothNiter;
  /*! Halo mode of the field solver (input HaloOverlap) */
  bool haloOverlap_;
  /*! ||b|| of the previous Maxwell solve (field-solver guard); 0 before the
   *  first one */
  double previousMaxwellSourceNorm_ = 0.0;
  /*! delt = c*th*dt */
  double delt;
  /*! number of particles species */
  int ns;
  /*! GEM challenge parameters */
  double B0x, B0y, B0z, delta;
  /** Earth Model parameters */
  double B1x, B1y, B1z;
  /*! charge to mass ratio array for different species (page-locked: copied
   * to the device) */
  double* qom;
  /*! u0, v0, w0 of species 0 (E0 = -(u0 x B0) / c, TaylorGreen flow) */
  double ue0, ve0, we0;

  // KEEP IN MEMORY GUARD CELLS ARE INCLUDED
  /*! number of cells - X direction, including + 2 (guard cells) */
  int nxc;
  /*! number of nodes - X direction, including + 2 extra nodes for guard cells
   */
  int nxn;
  /*! number of cell - Y direction, including + 2 (guard cells) */
  int nyc;
  /*! number of nodes - Y direction, including + 2 extra nodes for guard cells
   */
  int nyn;
  /*! number of cell - Z direction, including + 2 (guard cells) */
  int nzc;
  /*! number of nodes - Z direction, including + 2 extra nodes for guard cells
   */
  int nzn;
  /*! local grid boundaries coordinate */
  double xStart, xEnd, yStart, yEnd, zStart, zEnd;
  /*! grid spacing */
  double dx, dy, dz, invVOL;
  /*! simulation box length - X direction */
  double Lx;
  /*! simulation box length - Y direction */
  double Ly;
  /*! simulation box length - Z direction */
  double Lz;
  /** source center - X direction   */
  double x_center_dipole;
  /** source center - Y direction   */
  double y_center_dipole;
  /** source center - Z direction   */
  double z_center_dipole;
  /** planet center - X direction   */
  double x_center_planet;
  /** planet center - Y direction   */
  double y_center_planet;
  /** planet center - Z direction   */
  double z_center_planet;
  /** Planet radius */
  double Planet_radius;

  // The host arrays below that are copied to or from the device
  // (gpuSolverSyncH2D, gpuSolverSyncD2H) are page-locked (pinned_array).

  /*! PHI: electric potential (indexX, indexY, indexZ), defined on central
   * points between nodes */
  pinned_array3_double PHI;

  // Electric field components defined on nodes
  //
  pinned_array3_double Ex;
  pinned_array3_double Ey;
  pinned_array3_double Ez;

  // magnetic field components defined on central points between nodes
  //
  pinned_array3_double Bxc;
  pinned_array3_double Byc;
  pinned_array3_double Bzc;

  // magnetic field components defined on nodes
  //
  pinned_array3_double Bxn;
  pinned_array3_double Byn;
  pinned_array3_double Bzn;

  /*! temporary node arrays of the initializations (initSpeciesCurrents,
   * initGEMHarris, initNullPoints) */
  array3_double tempXN;
  array3_double tempYN;
  array3_double tempZN;

  // *******************************************************************************
  // *********** SOURCES **
  // *******************************************************************************

  /*! Charge density, defined on nodes */
  pinned_array3_double rhon;
  /*! SPECIES: charge density for each species, defined on nodes */
  pinned_array4_double rhons;
  /*! SPECIES: charge density for each species, defined on central points of the
   * cell */
  array4_double rhocs;

  // current density defined on nodes
  //
  pinned_array3_double Jx;
  pinned_array3_double Jy;
  pinned_array3_double Jz;

  // species-specific current densities defined on nodes
  //
  pinned_array4_double Jxs;
  pinned_array4_double Jys;
  pinned_array4_double Jzs;

  // external magnetic field on nodes
  pinned_array3_double Bx_ext;
  pinned_array3_double By_ext;
  pinned_array3_double Bz_ext;

  array3_double Bx_tot;
  array3_double By_tot;
  array3_double Bz_tot;

  // pressure tensor components, defined on nodes
  pinned_array4_double pXXsn;
  pinned_array4_double pXYsn;
  pinned_array4_double pXZsn;
  pinned_array4_double pYYsn;
  pinned_array4_double pYZsn;
  pinned_array4_double pZZsn;
  // heat flux of the output, [ns * HeatFlux::ComponentCount][nxn][nyn][nzn],
  // copied from the device: it exists only when the output writes it
  // (allocateHeatFlux)
  std::unique_ptr<pinned_array4_double> heatFlux;

  // boundary conditions for EM field: 0 = perfect conductor, 1 = magnetic
  // mirror, 2 = open
  int bcEMfaceXright;
  int bcEMfaceXleft;
  int bcEMfaceYright;
  int bcEMfaceYleft;
  int bcEMfaceZright;
  int bcEMfaceZleft;
  // The field boundary conditions: the spec (constructor) and the GPU plan
  // (gpuSolverAllocate), the one source of every boundary rule
  FieldBoundary bc_;

  /*! per-species initial density (4 pi n_s) */
  double* rhoINIT;
  /*! Drift of the species */
  bool* DriftSpecies;

  /*! boolean for divergence cleaning */
  bool PoissonCorrection;
  int PoissonCorrectionCycle;
  bool divBCorrection;
  int divBCorrectionCycle;

  // Krylov vector lengths: 3 x interior nodes (Maxwell), interior cells
  // (div(E) cleaning), interior nodes with one row plane along a flat axis
  // (div(B) cleaning)
  int maxwellKrylovSize_ = 0;
  int divEKrylovSize_ = 0;
  int divBKrylovSize_ = 0;

  /*! RESTART BOOLEAN */
  int restart1;

  /*! tolerance of the Maxwell solve (GMRES, FGMRES) */
  double GMREStol;
  /*! tolerance of the div(E) and div(B) cleaning solves (FGMRES) */
  double divCleaningTol;
  /*! Maxwell solver: "GMRES" (default, also any unknown value),
   * "FGMRESBlockJacobi" or "Chebyshev" */
  std::string SolverType;

  // MPI Derived Datatype for Center Halo Exchange
  MPI_Datatype yzFacetypeC;
  MPI_Datatype xzFacetypeC;
  MPI_Datatype xyFacetypeC;
  MPI_Datatype xEdgetypeC;
  MPI_Datatype yEdgetypeC;
  MPI_Datatype zEdgetypeC;
  MPI_Datatype xEdgetypeC2;
  MPI_Datatype yEdgetypeC2;
  MPI_Datatype zEdgetypeC2;
  MPI_Datatype cornertypeC;

  // MPI Derived Datatype for Node Halo Exchange
  MPI_Datatype yzFacetypeN;
  MPI_Datatype xzFacetypeN;
  MPI_Datatype xyFacetypeN;
  MPI_Datatype xEdgetypeN;
  MPI_Datatype yEdgetypeN;
  MPI_Datatype zEdgetypeN;
  MPI_Datatype xEdgetypeN2;
  MPI_Datatype yEdgetypeN2;
  MPI_Datatype zEdgetypeN2;
  MPI_Datatype cornertypeN;

  // for VTK output
  MPI_Datatype procviewXYZ, xyzcomp, procview, ghosttype;
  bool lEndFlag;

  // =========================================================================
  //  Device field arrays of the field solver
  //  -----------------------------------------------------------------------
  //  Row-major and contiguous; the arrays that also exist on the host have
  //  the host layout, so gpuSolverSyncH2D / gpuSolverSyncD2H copy them
  //  whole.
  // =========================================================================

  // Electric field (node-based): E^n, advanced to E(n+1) by gpuAdvanceE
  GPUFieldArray3 d_Ex, d_Ey, d_Ez;
  // E(n+theta) from the solve, unsmoothed (Faraday's law)
  GPUFieldArray3 d_Exth, d_Eyth, d_Ezth;
  // E(n+theta) smoothed: the field the mover samples, and the one E(n+1) is
  // built from; allocated only when smoothingOn()
  GPUFieldArray3 d_ExthSmooth, d_EythSmooth, d_EzthSmooth;

  // Magnetic field (center-based)
  GPUFieldArray3 d_Bxc, d_Byc, d_Bzc;
  // Magnetic field (node-based)
  GPUFieldArray3 d_Bxn, d_Byn, d_Bzn;

  // Charge / current densities summed over species (d_rhoc, d_rhoh:
  // cell-centred; the others node-based)
  GPUFieldArray3 d_rhon, d_rhoc, d_rhoh;
  GPUFieldArray3 d_Jx, d_Jy, d_Jz;
  GPUFieldArray3 d_Jxh, d_Jyh, d_Jzh;

  // Per-species densities and currents (node-based, species-indexed)
  GPUFieldArray4 d_rhons;
  GPUFieldArray4 d_Jxs, d_Jys, d_Jzs;

  // Pressure tensor (node-based, species-indexed)
  GPUFieldArray4 d_pXXsn, d_pXYsn, d_pXZsn;
  GPUFieldArray4 d_pYYsn, d_pYZsn, d_pZZsn;

  // Potentials of the cleanings: PHI of div(E) on the centres, PSI of div(B)
  // on the nodes; both also unpack the Krylov vectors of their image
  GPUFieldArray3 d_PHI, d_PSI;

  // External B (node-based)
  GPUFieldArray3 d_Bx_ext, d_By_ext, d_Bz_ext;

  // Temporary / work arrays (*C: cell-centred; the others node-based)
  GPUFieldArray3 d_tempXC, d_tempYC, d_tempZC;
  GPUFieldArray3 d_tempXN, d_tempYN, d_tempZN;
  GPUFieldArray3 d_tempC;
  GPUFieldArray3 d_tempX, d_tempY, d_tempZ;
  GPUFieldArray3 d_temp2X, d_temp2Y, d_temp2Z;
  GPUFieldArray3 d_divD; // div(μ·E) of the Maxwell images (cell-centred)
  GPUFieldArray3 d_Dx, d_Dy, d_Dz;
  GPUFieldArray3 d_vectX, d_vectY, d_vectZ;
  GPUFieldArray3 d_divC;
  // cell scratch of the Maxwell images (poissonTemp/Im also of the 7-point
  // images, not used currently)
  GPUFieldArray3 d_divBwork, d_divE_work, d_poissonTemp, d_poissonIm;

  // Krylov vectors of the Maxwell solve (the cleanings: CleaningSolve)
  GPUKrylovVector d_xkrylovMaxwell, d_bkrylovMaxwell;

  // Device copy of species q/m
  cudaSolverType* d_qom = nullptr;

  // Planet nodes of the constant-charge model (Dipole, Dipole2D): the host
  // planetNodes list, uploaded once
  int* d_planetNodes_ = nullptr;
  int planetNodeCount_ = 0;

  // Reduction scratch buffer for GPU BLAS dot/norm operations
  cudaSolverType* d_blasScratch = nullptr;

  // GPU GMRES workspace, persistently allocated for the Maxwell length (the
  // longest Krylov vectors).
  cudaSolverType* d_gmresV = nullptr; // [GMRES_MP1][max krylov length]
  cudaSolverType* d_gmresW = nullptr; // [max krylov length]
  std::size_t gmresVAlloc = 0;        // allocated d_gmresV element count

  // GPU FGMRES workspace (Z basis = preconditioned vectors), allocated on first
  // FGMRES use.
  cudaSolverType* d_fgmresZ = nullptr; // [GMRES_M][max krylov length]
  std::size_t fgmresZAlloc = 0;        // allocated d_fgmresZ element count

  // ---- Chebyshev solver workspace (4 Maxwell vectors, lazy) ----
  cudaSolverType* d_chebY = nullptr;   // current iterate
  cudaSolverType* d_chebW = nullptr;   // new iterate
  cudaSolverType* d_chebZ = nullptr;   // previous iterate
  cudaSolverType* d_chebTmp = nullptr; // operator output scratch
  int chebAlloc = 0;                   // allocated length (0 = not yet)
  // Chebyshev parameters (configurable, estimated if <= 0)
  int chebMaxIter = 20;    // default polynomial degree
  double chebEigMin = 0.0; // 0 → will be set to 1.0
  double chebEigMax = 0.0; // 0 → estimated via power iteration

  // The div(E) (cells) and div(B) (nodes) cleaning solves, and the one
  // FGMRES runs (its image and preconditioner read it)
  CleaningSolve divE_, divB_;
  const CleaningSolve* cleaning_ = nullptr;
  // the null-space sums, then the line coefficients (cleanLinesSlots() each)
  cudaSolverType* d_cleanLines_ = nullptr;
  // their page-locked host slots, for the copies of the sums and coefficients
  PinnedVector<cudaSolverType> h_cleanLines_;
  // the two Chebyshev iterates of the cleaning preconditioner, of
  // cleanChebLength_ each
  cudaSolverType* d_cleanCheb_ = nullptr;
  int cleanChebLength_ = 0;
  int poissonChebMaxIter =
      20; // Chebyshev steps of the cleaning preconditioners
  double poissonChebRescaleEigMin = 1.0; // rescale factor for min eigenvalue
  double poissonChebRescaleEigMax = 1.0; // rescale factor for max eigenvalue
  double poissonChebEigRatio = 300.0;    // eigMin >= eigMax / this (<= 0: off)

  // Block-Jacobi preconditioner parameters
  int blockJacobiSweeps =
      1; // number of Richardson sweeps (1 = single application)
  double blockJacobiOmega =
      1.0; // damping factor (1.0 = no damping, 2/3 typical for 3D)

  // Precomputed D^{-1} (9 entries per node, allocated lazily)
  cudaSolverType* d_blockJacobiDinv = nullptr;
  int blockJacobiDinvAlloc = 0; // allocated nodeSlice (0 = not yet)
  bool blockJacobiDinvStale = true;

  // Block-Jacobi Richardson sweep scratch (separate from Chebyshev workspace)
  double* d_bjScratch1 = nullptr; // residual vector
  double* d_bjScratch2 = nullptr; // D^{-1} residual
  int bjScratchAlloc = 0;         // allocated length (0 = not yet)

  // ---- Persistent PINNED host buffers for GMRES reductions ----
  // Avoids per-call heap allocation and enables true async D→H DMA.
  static constexpr int GMRES_M = 20;
  static constexpr int GMRES_MP1 = GMRES_M + 1;
  // Restart length of the cleaning solves.  Their vectors (interior cells or
  // nodes) are at most a third of the Maxwell ones (3 x interior nodes), so
  // the device workspace sized for GMRES_M Maxwell vectors holds the longer
  // basis; the host buffers and the reduction scratch are sized for it.
  static constexpr int CLEANING_M = 3 * GMRES_M;
  static constexpr int KRYLOV_M_MAX =
      CLEANING_M > GMRES_M ? CLEANING_M : GMRES_M;
  static_assert(KRYLOV_M_MAX <= kMaxCombination,
                "an Arnoldi step passes up to KRYLOV_M_MAX coefficients");
  // pinned host buffers of the Arnoldi process, for up to KRYLOV_M_MAX
  // vectors: reductions (local, global), Hessenberg matrix, residual,
  // Givens cosines and sines, back substitution
  double* h_gmresReduceLocal = nullptr;  // [KRYLOV_M_MAX + 1]
  double* h_gmresReduceGlobal = nullptr; // [KRYLOV_M_MAX + 1]
  double* h_gmresH = nullptr;            // [(KRYLOV_M_MAX + 1) * KRYLOV_M_MAX]
  double* h_gmresG = nullptr;            // [KRYLOV_M_MAX + 1]
  double* h_gmresCS = nullptr;           // [KRYLOV_M_MAX]
  double* h_gmresSN = nullptr;           // [KRYLOV_M_MAX]
  double* h_gmresY = nullptr;            // [KRYLOV_M_MAX + 1]

  // Dedicated non-blocking CUDA stream for the GPU field solver.
  // Avoids implicit serialisation with particle streams via the legacy default
  // stream.
  cudaStream_t solverStream_ = 0;

  // Launch sets of the Maxwell images and gpuMUdot (whole and split
  // operator), built once by gpuSolverAllocate
  MaxwellImageBoxes imageBoxes_{};

  // ---- Halo exchange (GPUBatchedHaloComm.cu) ----
  // An exchange fills every ghost box from the neighbour across it, the 26
  // directions (6 faces, 12 edges, 8 corners), with one message per
  // distinct neighbour rank: the receives are posted first, then all sends,
  // and one MPI_Waitall completes them.  Along a self axis (periodic on one
  // process) the ghost boxes are copies within the array, so only the boxes
  // with no step along a self axis travel: in 2D (one cell along z) the
  // z-ghosts of the received boxes are copied after the unpack, not sent.
  // HALO_MAX_BATCH is the max number of fields per single batched exchange.
  // Callers with more fields split them themselves (e.g.
  // gpuCommunicateGhostP2G_AllSpecies, HALO_MAX_BATCH/10 species per pass).
  static constexpr int HALO_MAX_BATCH = 64;
  cudaSolverType* d_haloSend_ = nullptr; // the messages one after another
  // Receive buffers, used by alternate exchanges with a remote neighbour: an
  // exchange posts its receives before packing, into the buffer of the
  // last-but-one such exchange.  Its unpack has completed: the host waited
  // for the pack of the last such exchange, which the GPU queues after it
  // (haloBuffersReady_, or the one stream of the blocking exchanges).
  cudaSolverType* d_haloRecv_[2] = {};
  int haloRecvSlot_ = 0;
  cudaSolverType** d_ptrArray_ = nullptr; // device table of the field pointers
  bool haloBufsAllocated_ = false;

  // Topology and local mesh dimensions do not change while an EMfields3D
  // instance is alive: gpuAllocateHaloBuffers builds the plans once, for
  // each topology (field, particle communicator), geometry (centres, nodes
  // with send offset 1 or 0) and extent (all boxes, faces only); an
  // exchange selects one.
  enum HaloTopologyKind { HALO_FIELD = 0, HALO_PARTICLE = 1 };
  enum HaloGeometryKind {
    HALO_CENTER_OFFSET0 = 0,
    HALO_NODE_OFFSET1 = 1,
    HALO_NODE_OFFSET0 = 2
  };
  HaloPlan haloPlans_[2][3][2] = {}; // [topology][geometry][faces only]

  void gpuInitializeHaloPlans();
  /** The plan of an exchange of nFields arrays of (nx, ny, nz); aborts on
   *  arguments outside the contract of the exchanges. */
  const HaloPlan& gpuSelectHaloPlan(cudaSolverType* const* h_fieldPtrs,
                                    int nFields, int nx, int ny, int nz,
                                    bool offsetZero, bool faceOnly,
                                    bool isParticle) const;
  /** Post the receives of plan into the next receive buffer; returns it and
   *  sets requests[0, plan.peers). */
  cudaSolverType* gpuHaloPostReceives(const HaloPlan& plan, int nFields,
                                      MPI_Request* requests);
  /** Upload the field pointers and queue the pack launch on stream. */
  void gpuHaloPack(const HaloPlan& plan, cudaSolverType* const* h_fieldPtrs,
                   int nFields, cudaStream_t stream);
  /** Post the sends of plan (the send buffer is packed) into
   *  requests[plan.peers, 2 plan.peers) and complete all of them. */
  void gpuHaloSendAndWait(const HaloPlan& plan, int nFields,
                          MPI_Request* requests);
  /** Queue the unpack of recv and the self copies after it on stream. */
  void gpuHaloUnpack(const HaloPlan& plan, cudaSolverType* recv, int nFields,
                     cudaStream_t stream);

  // ---- Split halo exchange state ----
  // Communication kernels use a separate stream so packing, MPI and
  // unpacking can overlap the dependency-free stencil core on solverStream_.
  // Created only with HaloOverlap on (gpuAllocateHaloBuffers).
  cudaStream_t haloStream_ = 0;
  cudaEvent_t haloInputReady_ = nullptr;
  cudaEvent_t haloPacked_ = nullptr;
  cudaEvent_t haloBuffersReady_ = nullptr;

  struct HaloExchangeState {
    bool active = false;
    const HaloPlan* plan = nullptr;
    int nFields = 0;
    cudaStream_t computeStream = 0;
    cudaSolverType* recv = nullptr;    // receive buffer of this exchange
    MPI_Request requests[2 * 26] = {}; // receives, then sends
  } haloExchange_;

  // Flag tracking whether GPU solver arrays have been allocated
  bool gpuSolverAllocated_ = false;
#if defined(IPIC3D_GPU_CYCLE_DIAGNOSTICS)
  // Persistent bounded reductions; allocated only if an enabled cycle runs.
  GPUCycleDiagnostics gpuCycleDiagnostics_;
#endif
};

typedef EMfields3D Field;

#endif // EM_FIELDS_3D_H
