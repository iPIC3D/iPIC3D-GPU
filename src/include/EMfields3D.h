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

#ifndef EM_FIELDS_3D_H
#define EM_FIELDS_3D_H

#include "Alloc.h"
#include "Basic.h"
#include "asserts.h"
#include "ipicfwd.h"
#include "mpi.h"

#include <cstddef>

#include "HeatFluxComponents.h"
#include "OutflowBoundary.h"
#include "cudaTypeDef.cuh"

#include "GPUFieldArray.cuh"
#include "GPUHaloComm.cuh"
#include "GPUMaxwellLocal.cuh"
#if defined(IPIC3D_GPU_CYCLE_DIAGNOSTICS)
#include "GPUCycleDiagnostics.cuh"
#endif

/**
 * @brief Electromagnetic fields, sources, and implicit-solver work arrays for
 * one MPI rank.
 *
 * The solver uses this class to initialize case-dependent fields, advance the
 * implicit Maxwell system, reduce the particle moments deposited by the moment
 * kernels (ghost exchange, species sums, hat functions), and hold the device
 * field arrays that the particle mover's field packing reads (gpuEx(), ...).
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
   *  gpuMaxwellImageFn().  The other exchanges (Maxwell source, Poisson
   *  solves) are blocking. */
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
   * @brief Initialize the beam test configuration.
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
  void initOriginalGEM();
  /** @brief Initialize the double-Harris-sheet GEM configuration. */
  void initGEMDoubleHarris();
  void initDoublePeriodicHarrisWithGaussianHumpPerturbation();
  /** @brief Initialize the hump-perturbation configuration. */
  void initHumpPerturbation();
  /** @brief Initialize GEM with a dipole-like tail and no perturbation. */
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
   * @brief Initialize a uniform rotated magnetic field.
   *
   * @param B Magnetic-field magnitude.
   * @param theta Rotation angle.
   */
  void initEM_rotate(double B, double theta);
  /**
   * @brief Add a perturbation cos(kx x + ky y + phase) to the densities of
   * species 0 and 1.
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
   * @brief Add a perturbation to the electromagnetic field.
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
   * @brief Impose a constant charge inside the 3D planet sphere.
   *
   * @param R Planet radius.
   * @param x_center Planet-center x coordinate.
   * @param y_center Planet-center y coordinate.
   * @param z_center Planet-center z coordinate.
   */
  void ConstantChargePlanet(double R, double x_center, double y_center,
                            double z_center);
  /**
   * @brief Impose a constant charge inside the 2D XZ-plane planet mask.
   *
   * @param R Planet radius.
   * @param x_center Planet-center x coordinate.
   * @param z_center Planet-center z coordinate.
   */
  void ConstantChargePlanet2DPlaneXZ(double R, double x_center,
                                     double z_center);
  /** @brief Impose a constant charge in the open-boundary layers. */
  void ConstantChargeOpenBC();
  /** @brief Zero all density-related fields. */
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

  // cell-centred B
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

  arr4_double getHeatFlux() { return heatFlux; }
  arr3_double getHeatFluxComponent(int is, int component) {
    return arr3_double(
        heatFlux.fetch_arr4()[HeatFlux::componentIndex(is, component)], nxn,
        nyn, nzn);
  }
  double getHeatFlux(int X, int Y, int Z, int is, int component) const {
    return heatFlux.get(HeatFlux::componentIndex(is, component), X, Y, Z);
  }
  double* getHeatFluxRaw() { return heatFlux.fetch_arr(); }
  double* getHeatFluxSpeciesPtr(int is) {
    return heatFlux.fetch_arr() +
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

  /*! electric field energy of this rank: sum over its owned nodes
   *  (ownedNodeEnds) of dV |E|^2 / (8 pi) */
  double getLocalEenergy();
  /*! magnetic field energy of this rank, with B_ext (owned nodes, as
   *  getLocalEenergy) */
  double getLocalBenergy();
  /*! electric field energy, summed over the ranks */
  double getEenergy();
  /*! magnetic field energy, summed over the ranks */
  double getBenergy();
  /*! bulk kinetic energy of species is, summed over the ranks (owned nodes,
   *  as getLocalEenergy) */
  double getBulkEnergy(int is);
  /*! Last node this rank owns along x, y, z; the owned nodes are [1, end].
   *  end = n-3 where the next rank or the periodic image holds node n-2,
   *  end = n-2 on a non-periodic last rank (node n-2 is the boundary node),
   *  so that every global node is counted once. */
  void ownedNodeEnds(int& iEnd, int& jEnd, int& kEnd) const;

  /*! print electromagnetic fields info */
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
  MPI_Datatype getGhostType() { return ghosttype; }

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
   * restart). */
  void gpuSolverSyncH2D(cudaStream_t stream = 0);
  /** Copy the fields and moments to the host (output, diagnostics, restart,
   * Catalyst). */
  void gpuSolverSyncD2H(cudaStream_t stream = 0);
  // ---- GPU Solver: GPU-aware MPI halo exchange ----
  /** Center-based halo exchange using GPU-aware MPI (operates on device
   * pointer). */
  void gpuCommunicateCenterBC(int nx, int ny, int nz, GPUFieldArray3& gpuArr,
                              int bcFaceXright, int bcFaceXleft,
                              int bcFaceYright, int bcFaceYleft,
                              int bcFaceZright, int bcFaceZleft);
  /** Batched 9-field center-based halo exchange + BC in one MPI round.  Not
   * used currently: the Maxwell image exchanges its 10 cell arrays with
   * per-array face BCs (the image-reference test uses it as reference). */
  void gpuCommunicateCenterBC_9(int nx, int ny, int nz, GPUFieldArray3& a1,
                                GPUFieldArray3& a2, GPUFieldArray3& a3,
                                GPUFieldArray3& a4, GPUFieldArray3& a5,
                                GPUFieldArray3& a6, GPUFieldArray3& a7,
                                GPUFieldArray3& a8, GPUFieldArray3& a9,
                                int bcXR, int bcXL, int bcYR, int bcYL,
                                int bcZR, int bcZL);
  /** Batched 3-field center-based halo exchange + BC_P in one MPI round. */
  void gpuCommunicateCenterBC_P_3(int nx, int ny, int nz, GPUFieldArray3& a1,
                                  GPUFieldArray3& a2, GPUFieldArray3& a3,
                                  int bcXR, int bcXL, int bcYR, int bcYL,
                                  int bcZR, int bcZL);
  /** Batched 3-field center-based halo exchange with per-field BCs (bc is
   * int[6]). */
  void gpuCommunicateCenterBC_3mixed(int nx, int ny, int nz, GPUFieldArray3& a1,
                                     const int* bc1, GPUFieldArray3& a2,
                                     const int* bc2, GPUFieldArray3& a3,
                                     const int* bc3);
  /** Batched 3-field node-based halo exchange with per-field BCs (bc is
   * int[6]). */
  void gpuCommunicateNodeBC_3mixed(int nx, int ny, int nz, GPUFieldArray3& a1,
                                   const int* bc1, GPUFieldArray3& a2,
                                   const int* bc2, GPUFieldArray3& a3,
                                   const int* bc3);
  /** Batched 3-field node box-stencil halo exchange with per-field BCs. */
  void gpuCommunicateNodeBoxStencilBC_3mixed(int nx, int ny, int nz,
                                             GPUFieldArray3& a1, const int* bc1,
                                             GPUFieldArray3& a2, const int* bc2,
                                             GPUFieldArray3& a3,
                                             const int* bc3);
  /** Particle-communicator centre halo exchange (for moments). */
  void gpuCommunicateCenterBC_P(int nx, int ny, int nz, GPUFieldArray3& gpuArr,
                                int bcFaceXright, int bcFaceXleft,
                                int bcFaceYright, int bcFaceYleft,
                                int bcFaceZright, int bcFaceZleft);
  /** Node-based box-stencil halo exchange using particle communicator (for
   * smooth). */
  void gpuCommunicateNodeBoxStencilBC_P(int nx, int ny, int nz,
                                        GPUFieldArray3& gpuArr,
                                        int bcFaceXright, int bcFaceXleft,
                                        int bcFaceYright, int bcFaceYleft,
                                        int bcFaceZright, int bcFaceZleft);
  /** Batched 3-field node-based box-stencil halo exchange using particle
   * communicator. */
  void gpuCommunicateNodeBoxStencilBC_P_3(int nx, int ny, int nz,
                                          GPUFieldArray3& a1,
                                          GPUFieldArray3& a2,
                                          GPUFieldArray3& a3, int bcXR,
                                          int bcXL, int bcYR, int bcYL,
                                          int bcZR, int bcZL);
  /** Center-based box-stencil halo exchange using particle communicator (for
   * smooth). */
  void gpuCommunicateCenterBoxStencilBC_P(int nx, int ny, int nz,
                                          GPUFieldArray3& gpuArr,
                                          int bcFaceXright, int bcFaceXleft,
                                          int bcFaceYright, int bcFaceYleft,
                                          int bcFaceZright, int bcFaceZleft);
  /** Batched 3-field center-based box-stencil halo exchange using particle
   * communicator. */
  void gpuCommunicateCenterBoxStencilBC_P_3(int nx, int ny, int nz,
                                            GPUFieldArray3& a1,
                                            GPUFieldArray3& a2,
                                            GPUFieldArray3& a3, int bcXR,
                                            int bcXL, int bcYR, int bcYL,
                                            int bcZR, int bcZL);

  // ---- GPU Solver: field-solver methods ----
  // gpuCalculateE, gpuCalculateB and gpuCalculateHatFunctions run the
  // instantiation of the run's HaloMode (input HaloOverlap).
  /** Advance E: div(E) correction (PoissonCorrection), implicit Maxwell
   * solve, smoothing, halo exchange and the post-solve open-boundary update
   * (gpuOpenBoundaryInflowE). */
  void gpuCalculateE(int cycle);
  /** Advance B by Faraday's law from Eth, apply its boundary conditions and
   * case fixes, interpolate to the nodes and clean div(B) (divBCorrection). */
  void gpuCalculateB(int cycle);
  /** Build the hat quantities (rhohat, Jhat) of the implicit field solve;
   * cycle=-1 denotes initialization. */
  void gpuCalculateHatFunctions(int cycle = -1);
  /** Maxwell operator A*x on device Krylov vectors (runs
   * gpuMaxwellImageFn(), the Krylov callback the field solvers take):
   *   A*x = delt^2 (-lap x - grad div(mu.x)) + mu.x + x
   * on the interior nodes, then the boundary rows.  The cell gradients of x
   * and div(mu.x) are exchanged between ranks in one halo exchange;
   * gpuMaxwellLocalNodeFused writes the Krylov image. */
  void gpuMaxwellImage(cudaSolverType* d_im, cudaSolverType* d_vector);
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
   *  gpuSolverAllocate). */
  const MaxwellImageBoxes& gpuMaxwellImageBoxes() const { return imageBoxes_; }
  /** Maxwell operator, communication-free: A*x for use as
   * preconditioner (the Richardson sweeps of gpuBlockJacobiPrecond, run when
   * BlockJacobiSweeps > 1). Ghost cells are treated as zero, apart from the
   * X-right ghost cell plane on an outflow face, and physical boundary image
   * corrections are enforced locally so the operator better matches the full
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
  /** GPU perfect conductor BC, image rows (linear) on every conductor face
   *  of this rank (bcEMface = 0), written into the Krylov image d_im:
   *  gpuPerfectConductorImage. */
  void gpuPerfectConductorImages(cudaSolverType* d_im, GPUFieldArray3& vX,
                                 GPUFieldArray3& vY, GPUFieldArray3& vZ);
  /** GPU perfect conductor BC, RHS rows on every conductor face of this rank
   *  (bcEMface = 0): tangential -(u0 x B0), normal the constant part of the
   *  normal row (gpuPerfectConductorSource). */
  void gpuPerfectConductorSources(cudaSolverType* dX, cudaSolverType* dY,
                                  cudaSolverType* dZ);

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

  /** Batched GPU halo exchange: exchanges nFields 3D arrays in a single set
   *  of MPI messages per direction, using explicit CUDA pack/unpack into
   *  persistent contiguous GPU buffers.
   *  @param h_fieldPtrs  Host array of nFields device pointers.
   *  @param nFields      Number of field arrays to batch.
   *  @param nx,ny,nz     Canonical local dimensions including ghosts: center
   *                      with offset 0, or node with offset 0/1. Arbitrary
   *                      subarray geometries are not accepted.
   *  @param offsetZero   true = send offset 0, false = send offset 1.  P2G
   *                      accumulation intentionally uses node dimensions with
   *                      offset 0.
   *  @param isFaceOnlyFlag true = skip edges/corners.
   *  @param needInterp   true = additive accumulation after exchange.
   *  @param isParticle   true = use particle communicator.
   *  @param stream       CUDA stream for kernels.  With HaloOverlap off,
   *                      successive calls sharing the persistent buffers must
   *                      use this same stream or be externally synchronized. */
  void gpuBatchedHaloExchange(cudaSolverType** h_fieldPtrs, int nFields, int nx,
                              int ny, int nz, bool offsetZero,
                              bool isFaceOnlyFlag, bool needInterp,
                              bool isParticle, cudaStream_t stream);

  /** Queue pointer upload and face packing on the halo stream.  The caller
   *  then queues dependency-free work on @p computeStream.  Needs the split
   *  exchange resources (HaloOverlap on). */
  void gpuBatchedHaloBeginExchange(cudaSolverType** h_fieldPtrs, int nFields,
                                   int nx, int ny, int nz, bool offsetZero,
                                   bool isFaceOnlyFlag, bool isParticle,
                                   cudaStream_t computeStream);
  /** Post face MPI as soon as packing completes, progress faces, edges and
   *  corners, then make the caller-provided compute stream wait for
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
   * The field solve sees the outflow layer (layersXright()) as neutral. With
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

  /**
   * @brief rhohat = 0 on every cell beyond the reference node x_s of the
   * X-right outflow face (the cell straddling x_s, the layer cells and the
   * ghost cell), so the solver's charge exterior starts at x_s like the
   * post-solve field copy. Without it the solve builds the field of the
   * straddling half cell (half the charge of node x_s), the copy removes it
   * every cycle, and the Gauss residual left behind is spread inward by the
   * solver's cleaning. With OutflowLayerFields = solved nothing is copied,
   * and the kept layer field is that of a neutral layer. Called in
   * gpuCalculateHatFunctions().
   */
  void gpuZeroOutflowRhohat();

  /** rhon += sum_s(rhons_s) on device. */
  void gpuSumOverSpecies();

  /** Jx/Jy/Jz = sum_s(Jxs/Jys/Jzs)_s on device. */
  void gpuSumOverSpeciesJ();

  /** rhoc = interpN2C(rhon) on device. */
  void gpuInterpDensitiesN2C();

  // ---- GPU Solver: open boundary conditions ----
  /** GPU open BC: RHS rows (E0 = -u0 x B0) of the open inflow faces (only
   *  with ApplyInflowBcsEImage). */
  void gpuOpenBoundaryInflowESource(cudaSolverType* dX, cudaSolverType* dY,
                                    cudaSolverType* dZ, int nx, int ny, int nz);
  /** GPU open BC: identity image rows of the open inflow faces, written into
   *  the Krylov image d_im. */
  void gpuOpenBoundaryInflowEImage(cudaSolverType* d_im,
                                   const cudaSolverType* vX,
                                   const cudaSolverType* vY,
                                   const cudaSolverType* vZ, int nx, int ny,
                                   int nz);
  /** GPU open BC (post-solve): on E inflow faces SAL blend, or without SAL
   * Dirichlet (X-left) / extrapolation (Y, Z); and
   * on an X-right outflow face the copy over fieldCopyLayersXright() node
   * planes (gpuOutflowCopyE). */
  void gpuOpenBoundaryInflowE(cudaSolverType* dX, cudaSolverType* dY,
                              cudaSolverType* dZ, int nx, int ny, int nz);
  /** GPU open BC: SAL blend / extrapolation on center B inflow faces; an
   * open X-right face copies the plane before them over
   * fieldCopyLayersXright() cell planes. */
  void gpuOpenBoundaryInflowB(cudaSolverType* dX, cudaSolverType* dY,
                              cudaSolverType* dZ, int nx, int ny, int nz);

  // ---- GPU Solver: ghost node on the X-right outflow face ----
  // (see the X-right open boundary layer section below)
  /** Ghost node plane of the node arrays a[0..2] := their face plane on an
   * outflow face; no-op elsewhere. */
  void gpuOutflowGhostNodes(GPUFieldArray3* const a[3]);
  /** Fill the y/z ghost rows of the X-right ghost cell plane of the cell
   * arrays a[0..n-1] on an outflow face; no-op elsewhere. */
  void gpuOutflowGhostRows(cudaSolverType* const a[], int n);
  /** Copy E (dX, dY, dZ, node arrays) over fieldCopyLayersXright() node
   * planes of the X-right outflow face and its ghost node from the plane
   * before them: the reference plane, or with OutflowLayerFields = solved
   * the face; no-op elsewhere. */
  void gpuOutflowCopyE(cudaSolverType* dX, cudaSolverType* dY,
                       cudaSolverType* dZ);

  // ---- GPU Solver: case-specific B fixes ----
  /** GPU fix center B for GEM case. */
  void gpuFixBcGEM();
  /** GPU fix node B for GEM case. */
  void gpuFixBnGEM();
  /** GPU fix center B for ForceFree case. */
  void gpuFixBforcefree();

  // ---- GPU Solver: ConstantChargePlanet ----
  /** GPU ConstantChargePlanet: set rhons inside sphere. */
  void gpuConstantChargePlanet(double R, double x_center, double y_center,
                               double z_center);
  /** GPU ConstantChargePlanet 2D: set rhons inside circle in XZ plane. */
  void gpuConstantChargePlanet2DPlaneXZ(double R, double x_center,
                                        double z_center);

  // ---- GPU Solver: Poisson/divB correction ----
  /** GPU Poisson image operator (laplacian on centers, with communication). */
  void gpuPoissonImage(cudaSolverType* d_im, cudaSolverType* d_vec);
  /** GPU Poisson image operator, communication-free (computes -∇², positive
   * eigenvalues).  Physical faces use the ghost rule of gpuPoissonImage, a
   * periodic direction with one cell is flat, rank interfaces are cut with
   * zero ghosts (block Jacobi).  Not used currently: the Chebyshev
   * preconditioner applies the same operator inside its step kernels. */
  void gpuPoissonImageLocal(cudaSolverType* d_im, cudaSolverType* d_vec);
  /** Analytic eigenvalue bounds of the local -∇², eigMin floored at
   * eigMax / PoissonChebEigRatio and both rescaled (once per simulation). */
  void computePoissonChebyshevEigenvalues();
  /** Chebyshev preconditioner for Poisson (communication-free, analytic
   * eigenvalues). */
  void gpuChebyshevPrecondPoisson(cudaSolverType* d_x, cudaSolverType* d_b);
  /** GPU FGMRES(m) with Chebyshev preconditioner for Poisson (divergence
   * cleaning). */
  KrylovSolveStatus gpuFGMRES_PoissonChebyshev(cudaSolverType* d_x, int n,
                                               cudaSolverType* d_b, int m,
                                               int max_iter, cudaSolverType tol,
                                               MPI_Comm fieldcomm);
  /** GPU Poisson correction for div(E) cleaning. */
  void gpuPoissonCorrection(int cycle);
  /** Divergence cleaning of B: solve lap(PSI) = div(B) with PSI = 0 beyond
   * the physical faces and subtract grad(PSI) from the node B on every
   * node, as the div(E) correction does. */
  void gpuApplyDivBCleaning(int cycle);

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

  // ---- GPU Solver: additional halo exchange ----
  /** Center box-stencil halo exchange using field communicator. */
  void gpuCommunicateCenterBoxStencilBC(int nx, int ny, int nz,
                                        GPUFieldArray3& gpuArr,
                                        int bcFaceXright, int bcFaceXleft,
                                        int bcFaceYright, int bcFaceYleft,
                                        int bcFaceZright, int bcFaceZleft);

  // ---- GPU Solver: accessor helpers ----
  GPUFieldArray3& gpuEx() { return d_Ex; }
  GPUFieldArray3& gpuEy() { return d_Ey; }
  GPUFieldArray3& gpuEz() { return d_Ez; }
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
  /** Zero the species currents Jxs/Jys/Jzs and pressures p**sn. With
   *  fromAmpere, set the species currents to shares of J = (c/4pi) curl(B),
   *  B the cell-centred field: J_s,x = f_s,x J_x with
   *  f_s,x = sign(q_s) u0_s / sum_r sign(q_r) u0_r (v0 for y, w0 for z), so
   *  the drift J_s / (sign(q_s) rho_s) of maxwellianFromSpeciesCurrents is
   *  u0_s J_x / (N_u rho_s), N_u = sum_r sign(q_r) u0_r; a component whose
   *  weights sum to 0 gets no current. With varyingThermal, also set the
   *  diagonal pressures p_s,xx = J_s,x^2 / rho_s + uth_s^2 rho_s (yy, zz
   *  alike), read by maxwellianFromSpeciesCurrents for the thermal speed. */
  void initSpeciesCurrents(bool fromAmpere, bool varyingThermal);

private: // X-right open boundary layer
  /** True on ranks owning an X-right open outflow face (bcEMfaceXright = 2,
   *  bcPfaceXright = 3). */
  bool ownsOutflowFaceXright() const;
  /** Layers L of the X-right open face: node planes nxn-1-L..nxn-1
   *  (centers nxc-1-L..nxc-1) take the values of plane nxn-2-L (nxc-2-L).
   *  On an outflow face the neutralized charge, the copied moments and, with
   *  OutflowLayerFields = copy, E and B share L = max(3, n_layers_sal), so
   *  the L cells between the reference plane and the face contain the 3-cell
   *  particle open layer; otherwise L = n_layers_sal. */
  int layersXright() const;
  /** Layers over which E and B are copied after a solve on the X-right open
   *  face: layersXright(), except on an outflow face with
   *  OutflowLayerFields = solved, where it is 0: only the ghost node plane
   *  nxn-1 and the ghost cell plane nxc-1 take the face values, the
   *  zero-gradient continuation the Maxwell operator assumes. */
  int fieldCopyLayersXright() const;

  // Maxwell-operator closure of the outflow face: see OutflowBoundary.h.
  /** Last cell plane of the operator's node-to-cell stencils: the ghost
   *  plane nxc-1 on an outflow face, the interior nxc-2 elsewhere. */
  int operatorCellEnd() const;
  /** Face code for the X-right face of the operator's cell arrays:
   *  BC_FACE_KEEP on an outflow face, where the stencils fill the ghost
   *  plane, otherwise code. */
  int operatorBCXright(int code) const;
  /** Ghost-row rules of the X-right ghost cell plane, rule[0..3] = y-left,
   *  y-right, z-left, z-right: a physical face keeps its own BC, a periodic
   *  image on this rank gives the exact partner row, and a neighbour rank's
   *  row, whose face-plane nodes the operator does not exchange, is stood in
   *  for by the adjacent row. */
  void outflowGhostRowRules(int rule[4]) const;

private: // solver steps in halo mode M (see HaloMode)
  template <HaloMode M> void gpuCalculateE(int cycle);
  template <HaloMode M> void gpuCalculateB(int cycle);
  template <HaloMode M> void gpuCalculateHatFunctions(int cycle);
  /** SmoothNiter iterations of box smoothing of arr (type 0: cells). */
  template <HaloMode M> void gpuSmooth(GPUFieldArray3& arr, int type);
  /** Smoothing of E, the 3 components together. */
  template <HaloMode M> void gpuSmoothE();
  /** Smoothing of 3 arrays together (type 0: cells; Jhat). */
  template <HaloMode M>
  void gpuSmooth3(GPUFieldArray3& a1, GPUFieldArray3& a2, GPUFieldArray3& a3,
                  int type);

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
  /*! charge to mass ratio array for different species */
  double* qom;
  /*! u0, v0, w0 of species 0 (E0 = -u0 x B0, TaylorGreen flow) */
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

  /*! PHI: electric potential (indexX, indexY, indexZ), defined on central
   * points between nodes */
  array3_double PHI;

  // Electric field components defined on nodes
  //
  array3_double Ex;
  array3_double Ey;
  array3_double Ez;

  // magnetic field components defined on central points between nodes
  //
  array3_double Bxc;
  array3_double Byc;
  array3_double Bzc;

  // magnetic field components defined on nodes
  //
  array3_double Bxn;
  array3_double Byn;
  array3_double Bzn;

  /*! temporary node arrays of the initializations (initSpeciesCurrents for
   * GEMHarris and ForceFree, initNullPoints) */
  array3_double tempXN;
  array3_double tempYN;
  array3_double tempZN;

  // *******************************************************************************
  // *********** SOURCES **
  // *******************************************************************************

  /*! Charge density, defined on nodes */
  array3_double rhon;
  /*! SPECIES: charge density for each species, defined on nodes */
  array4_double rhons;
  /*! SPECIES: charge density for each species, defined on central points of the
   * cell */
  array4_double rhocs;

  // current density defined on nodes
  //
  array3_double Jx;
  array3_double Jy;
  array3_double Jz;

  // species-specific current densities defined on nodes
  //
  array4_double Jxs;
  array4_double Jys;
  array4_double Jzs;

  // external magnetic field on nodes
  array3_double Bx_ext;
  array3_double By_ext;
  array3_double Bz_ext;

  array3_double Bx_tot;
  array3_double By_tot;
  array3_double Bz_tot;

  // pressure tensor components, defined on nodes
  array4_double pXXsn;
  array4_double pXYsn;
  array4_double pXZsn;
  array4_double pYYsn;
  array4_double pYZsn;
  array4_double pZZsn;
  array4_double heatFlux;

  // boundary conditions for EM field: 0 = perfect conductor, 1 = magnetic
  // mirror, 2 = open
  int bcEMfaceXright;
  int bcEMfaceXleft;
  int bcEMfaceYright;
  int bcEMfaceYleft;
  int bcEMfaceZright;
  int bcEMfaceZleft;
  // Faces of this rank (0 Xleft, 1 Xright, 2 Yleft, 3 Yright, 4 Zleft,
  // 5 Zright), fixed at construction so the solver loops do not re-evaluate
  // the topology and boundary codes:
  // - openInflowFace_: bcEMface = 2 with bcPface = 2, never X-right (an open
  //   X-right face keeps its operator rows).  With ApplyInflowBcsEImage its
  //   Maxwell rows are the identity (image) and E0 = -u0 x B0 (RHS);
  // - conductorFace_: perfect conductor, bcEMface = 0.
  bool openInflowFace_[6];
  bool conductorFace_[6];

  // Absorbing boundary parameters
  int yes_sal;
  int n_layers_sal;
  // SAL on B leaves the face-normal component alone (sal_tangential_B)
  int sal_tangential_B;

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
  // (Poisson)
  int maxwellKrylovSize_ = 0;
  int poissonKrylovSize_ = 0;

  /*! RESTART BOOLEAN */
  int restart1;

  /*! tolerance of the Maxwell (GMRES, FGMRES) and Poisson solves */
  double GMREStol;
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

  // Electric field (node-based)
  GPUFieldArray3 d_Ex, d_Ey, d_Ez;
  GPUFieldArray3 d_Exth, d_Eyth, d_Ezth;

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

  // Potentials (center-based)
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

  // divB cleaning work arrays
  GPUFieldArray3 d_divBwork;
  GPUFieldArray3 d_gradPSIX, d_gradPSIY, d_gradPSIZ;

  // Poisson / Maxwell Krylov vectors
  GPUKrylovVector d_xkrylovMaxwell, d_bkrylovMaxwell;
  GPUKrylovVector d_xkrylovPoisson_B, d_bkrylovPoisson_B;
  GPUKrylovVector d_xkrylovPoisson_E, d_bkrylovPoisson_E;

  // div(E) correction work arrays
  GPUFieldArray3 d_divE_work;
  GPUFieldArray3 d_gradPHIX_work, d_gradPHIY_work, d_gradPHIZ_work;

  // Poisson image work arrays
  GPUFieldArray3 d_poissonTemp, d_poissonIm;

  // Smooth temp buffer
  GPUFieldArray3 d_smoothTemp;

  // Device copy of species q/m
  cudaSolverType* d_qom = nullptr;

  // Reduction scratch buffer for GPU BLAS dot/norm operations
  cudaSolverType* d_blasScratch = nullptr;

  // GPU GMRES workspace, persistently allocated for max(Maxwell, Poisson).
  cudaSolverType* d_gmresV = nullptr; // [GMRES_MP1][max krylov length]
  cudaSolverType* d_gmresW = nullptr; // [max krylov length]
  std::size_t gmresVAlloc = 0;        // allocated d_gmresV element count

  // GPU FGMRES workspace (Z basis = preconditioned vectors), allocated on first
  // FGMRES use.
  cudaSolverType* d_fgmresZ = nullptr; // [GMRES_M][max krylov length]
  std::size_t fgmresZAlloc = 0;        // allocated d_fgmresZ element count

  // ---- GPU Chebyshev workspace (4 Krylov-sized vectors) ----
  cudaSolverType* d_chebY = nullptr;   // current iterate
  cudaSolverType* d_chebW = nullptr;   // new iterate
  cudaSolverType* d_chebZ = nullptr;   // previous iterate
  cudaSolverType* d_chebTmp = nullptr; // operator output scratch
  int chebAlloc = 0;                   // allocated length (0 = not yet)
  // Chebyshev parameters (configurable, estimated if <= 0)
  int chebMaxIter = 20;    // default polynomial degree
  double chebEigMin = 0.0; // 0 → will be set to 1.0
  double chebEigMax = 0.0; // 0 → estimated via power iteration

  // Poisson Chebyshev preconditioner (analytic eigenvalues of local -∇²)
  double poissonChebEigMin = 0.0;
  double poissonChebEigMax = 0.0;
  bool poissonChebComputed = false;
  int poissonChebMaxIter =
      10; // Chebyshev polynomial degree for Poisson preconditioner
  double poissonChebRescaleEigMin = 1.0; // rescale factor for min eigenvalue
  double poissonChebRescaleEigMax = 1.0; // rescale factor for max eigenvalue
  double poissonChebEigRatio = 1000.0;   // eigMin >= eigMax / this (<= 0: off)

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
  double* h_gmresReduceLocal =
      nullptr; // pinned, [GMRES_MP1] for Arnoldi reductions
  double* h_gmresReduceGlobal =
      nullptr;                // pinned, [GMRES_MP1] for MPI_Allreduce output
  double* h_gmresH = nullptr; // pinned, [GMRES_MP1 * GMRES_M] Hessenberg matrix
  double* h_gmresG = nullptr; // pinned, [GMRES_MP1] residual vector
  double* h_gmresCS = nullptr; // pinned, [GMRES_M] Givens cosines
  double* h_gmresSN = nullptr; // pinned, [GMRES_M] Givens sines
  double* h_gmresY = nullptr;  // pinned, [GMRES_MP1] back-substitution work

  // Dedicated non-blocking CUDA stream for the GPU field solver.
  // Avoids implicit serialisation with particle streams via the legacy default
  // stream.
  cudaStream_t solverStream_ = 0;

  // Launch sets of the Maxwell images and gpuMUdot (whole and split
  // operator), built once by gpuSolverAllocate
  MaxwellImageBoxes imageBoxes_{};

  // ---- Persistent batched halo-exchange buffers ----
  // 6 directions: 0=XL 1=XR 2=YL 3=YR 4=ZL 5=ZR
  // HALO_MAX_BATCH is the max number of fields per single batched exchange.
  // Callers with more fields split them themselves (e.g.
  // gpuCommunicateGhostP2G_AllSpecies, HALO_MAX_BATCH/10 species per pass).
  static constexpr int HALO_MAX_BATCH = 64;
  double* d_haloBuf_send_[6] = {}; // contiguous GPU send buffers
  double* d_haloBuf_recv_[6] = {}; // contiguous GPU recv buffers
  double** d_ptrArray_ = nullptr;  // device array of field pointers for kernels
  double** h_ptrArray_ =
      nullptr; // pinned host staging for d_ptrArray_ H→D copies
  bool haloBufsAllocated_ = false;

  // ---- Immutable halo topology and geometry plans ----
  // Topology and local mesh dimensions do not change while an EMfields3D
  // instance is alive.  Resolve them once when halo storage is initialized;
  // exchanges only select one of these fixed plans.
  struct HaloTopologyPlan {
    MPI_Comm comm = MPI_COMM_NULL;
    int neighbors[6] = {};
    int present[6] = {};
    int selfAxis[3] = {};
    int remoteDirections[6] = {};
    int remoteDirectionCount = 0;
    int edgeDirections[6] = {};
    int edgeDirectionCount = 0;
    int edgeSegments[6][2] = {};
    int edgeMultiplicity[6] = {};
    int cornerDirections[2] = {};
    int cornerDirectionCount = 0;
    bool hasRemoteYZ = false;
  };

  struct HaloFacePlan {
    int sendBase = 0;
    int recvBase = 0;
    int outerStride = 0;
    int innerStride = 0;
    int outerCount = 0;
    int innerCount = 0;
    int elements = 0;
  };

  struct HaloEdgePlan {
    int sendBase = 0;
    int recvBase = 0;
    int stride = 0;
    int elements = 0;
  };

  struct HaloGeometryPlan {
    int nx = 0;
    int ny = 0;
    int nz = 0;
    int offset = 0;
    int active[3] = {};
    HaloFacePlan faces[6] = {};
    HaloEdgePlan edges[6][2] = {};
    int edgeLength[3] = {};
    int selfFaceBlocks[3][2] = {};
    int edgeKernelBlocks = 0;
    int cornerSendBase[2] = {};
    int cornerRecvBase[2] = {};
    int maxBufferElements[6] = {};
  };

  enum HaloTopologyKind { HALO_FIELD = 0, HALO_PARTICLE = 1 };
  enum HaloGeometryKind {
    HALO_CENTER_OFFSET0 = 0,
    HALO_NODE_OFFSET1 = 1,
    HALO_NODE_OFFSET0 = 2
  };
  HaloTopologyPlan haloTopologyPlans_[2] = {};
  HaloGeometryPlan haloGeometryPlans_[3] = {};
  bool haloPlansInitialized_ = false;

  void gpuInitializeHaloPlans();
  const HaloGeometryPlan& gpuSelectHaloGeometry(int nx, int ny, int nz,
                                                bool offsetZero) const;
  void gpuQueueHaloFacePack(cudaSolverType** h_fieldPtrs, int nFields,
                            const HaloTopologyPlan& topology,
                            const HaloGeometryPlan& geometry,
                            cudaStream_t stream);
  void gpuCompleteHaloPhases(const HaloTopologyPlan& topology,
                             const HaloGeometryPlan& geometry, int nFields,
                             bool faceOnly, cudaStream_t stream);

  // ---- Split halo exchange state ----
  // Communication kernels use a separate stream so face, edge and corner
  // progress can overlap the dependency-free stencil core on solverStream_.
  // Created only with HaloOverlap on (gpuAllocateHaloBuffers).
  cudaStream_t haloStream_ = 0;
  cudaEvent_t haloInputReady_ = nullptr;
  cudaEvent_t haloPhaseReady_ = nullptr;
  cudaEvent_t haloBuffersReady_ = nullptr;

  struct HaloExchangeState {
    bool active = false;
    const HaloTopologyPlan* topology = nullptr;
    const HaloGeometryPlan* geometry = nullptr;
    int nFields = 0;
    bool faceOnly = false;
    cudaStream_t computeStream = 0;
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
