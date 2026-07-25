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

#include "HeatFluxComponents.h"
#include "cudaTypeDef.cuh"

#ifdef GPU_SOLVER
#include "GPUFieldArray.cuh"
#include "GPUHaloComm.cuh"
#endif

// dimension of vectors used in fieldForPcls
const int DFIELD_3or4 = 4; // 4 pads with garbage but is needed for alignment

/**
 * @brief Electromagnetic fields, sources, and implicit-solver work arrays for
 * one MPI rank.
 *
 * The solver uses this class to initialize case-dependent fields, advance the
 * implicit Maxwell system, accumulate and reduce particle moments, and prepare
 * field buffers consumed by the GPU particle mover.
 */
class EMfields3D // :public Field
{
public:
  // ======= Construction and lifetime =======

  /**
   * @brief Construct field storage and solver work arrays for the local domain.
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
   * @brief Add a perturbation to the charge density.
   *
   * @param deltaBoB Magnetic perturbation amplitude normalized by the
   * background field.
   * @param kx Perturbation wave number in x.
   * @param ky Perturbation wave number in y.
   * @param Bx_mod Bx perturbation amplitude.
   * @param By_mod By perturbation amplitude.
   * @param Bz_mod Bz perturbation amplitude.
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

  // ======= Solver-facing field updates =======

  /**
   * @brief Advance the electric field with the implicit Maxwell solver.
   *
   * @param cycle Current simulation cycle.
   */
  void calculateE(int cycle);
  /**
   * @brief Apply the Poisson image operator used by the linear solver.
   *
   * @param image Output image vector.
   * @param vector Input Krylov vector.
   */
  void PoissonImage(double* image, double* vector);
  /**
   * @brief Apply the Maxwell image operator used by the linear solver.
   *
   * @param im Output image vector.
   * @param vector Input Krylov vector.
   */
  void MaxwellImage(double* im, double* vector);
  /**
   * @brief Apply the local Maxwell image operator used by the preconditioner.
   *
   * @param im Output image vector.
   * @param vector Input Krylov vector.
   */
  void MaxwellImageLocal(double* im, double* vector);
  /**
   * @brief Build the Maxwell right-hand-side source term.
   *
   * @param bkrylov Output right-hand-side vector.
   */
  void MaxwellSource(double* bkrylov);
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
  /** @brief Alternate constant-charge treatment for open boundaries. */
  void ConstantChargeOpenBCv2();
  /**
   * @brief Advance the magnetic field after the electric-field solve.
   *
   * @param cycle Current simulation cycle.
   */
  void calculateB(int cycle);
  /** @brief Apply divergence cleaning to the magnetic field. */
  void applyDivBCleaning();
  /** @brief Fix magnetic-field boundary values for the GEM challenge. */
  void fixBcGEM();
  void fixBnGEM();
  /** @brief Fix magnetic-field boundary values for the force-free case. */
  void fixBforcefree();

  // ======= Solver-facing moment post-processing =======

  /**
   * @brief Apply the implicit pressure tensor to a vector field.
   *
   * @param PIdotX Output x component of the tensor product.
   * @param PIdotY Output y component of the tensor product.
   * @param PIdotZ Output z component of the tensor product.
   * @param vectX Input x component.
   * @param vectY Input y component.
   * @param vectZ Input z component.
   * @param ns Species index.
   */
  void PIdot(arr3_double PIdotX, arr3_double PIdotY, arr3_double PIdotZ,
             const_arr3_double vectX, const_arr3_double vectY,
             const_arr3_double vectZ, int ns);
  /**
   * @brief Apply the implicit permeability tensor to a vector field.
   *
   * @param MUdotX Output x component of the tensor product.
   * @param MUdotY Output y component of the tensor product.
   * @param MUdotZ Output z component of the tensor product.
   * @param vectX Input x component.
   * @param vectY Input y component.
   * @param vectZ Input z component.
   */
  void MUdot(arr3_double MUdotX, arr3_double MUdotY, arr3_double MUdotZ,
             const_arr3_double vectX, const_arr3_double vectY,
             const_arr3_double vectZ);
  /** @brief Build the hat quantities used by the implicit field solve. */
  void calculateHatFunctions();

  /** @brief Interpolate nodal densities to cell centers. */
  void interpDensitiesN2C();
  /** @brief Zero all density-related fields. */
  void setZeroDensities();
  /** @brief Zero the per-species primary moments. */
  void setZeroPrimaryMoments();
  /** @brief Zero the aggregate moments derived from the primary moments. */
  void setZeroDerivedMoments();
  /** @brief Sum nodal charge density over all species. */
  void sumOverSpecies();
  /** @brief Sum nodal current density over all species. */
  void sumOverSpeciesJ();
  /**
   * @brief Smooth a cell-centered scalar field after interpolation.
   *
   * @param vector Field to smooth in place.
   * @param type Smoothing mode selector.
   */
  void smooth(arr3_double vector, int type);
  /**
   * @brief Smooth a per-species field after interpolation.
   *
   * @param value Constant fill value used by the smoother.
   * @param vector Per-species field to smooth in place.
   * @param is Species index.
   * @param type Smoothing mode selector.
   */
  void smooth(double value, arr4_double vector, int is, int type);
  /** @brief Smooth the electric field components. */
  void smoothE();

  /** @brief Populate the legacy nodal field buffer used by the particle mover.
   */
  void set_fieldForPcls();

  /**
   * @brief Pack field data into the cell-centered GPU mover buffer.
   *
   * The packed layout stores the four XY-plane corner nodes needed by the GPU
   * mover for each cell slab in Z.
   *
   * @param fieldForPclsOnCenter Output packed field buffer.
   */
  void set_fieldForPclsToCenter(cudaFieldType* fieldForPclsOnCenter);

  /**
   * @brief Communicate per-species moments before particle-to-grid reductions.
   *
   * @param ns Number of particle species.
   */
  void communicateGhostP2G(int ns);

  /**
   * @brief Communicate per-species heat-flux tensor components for output.
   *
   * Uses the same shared-node summation, non-periodic boundary scaling, and
   * ghost-node population pattern as communicateGhostP2G().
   *
   * @param is Species index.
   */
  void communicateGhostHeatFlux(int is);

  /**
   * @brief Adjust densities on non-periodic boundaries.
   *
   * @param is Species index.
   */
  void adjustNonPeriodicDensities(int is);
  /** @brief Apply the current moment boundary scaling to heat flux. */
  void adjustNonPeriodicHeatFlux(int is);

  /*! Perfect conductor boundary conditions LEFT wall */
  void perfectConductorLeft(arr3_double imageX, arr3_double imageY,
                            arr3_double imageZ, const_arr3_double vectorX,
                            const_arr3_double vectorY,
                            const_arr3_double vectorZ, int dir);
  /*! Perfect conductor boundary conditions RIGHT wall */
  void perfectConductorRight(arr3_double imageX, arr3_double imageY,
                             arr3_double imageZ, const_arr3_double vectorX,
                             const_arr3_double vectorY,
                             const_arr3_double vectorZ, int dir);
  /*! Perfect conductor boundary conditions for source LEFT wall */
  void perfectConductorLeftS(arr3_double vectorX, arr3_double vectorY,
                             arr3_double vectorZ, int dir);
  /*! Perfect conductor boundary conditions for source RIGHT wall */
  void perfectConductorRightS(arr3_double vectorX, arr3_double vectorY,
                              arr3_double vectorZ, int dir);

  /*! Calculate the sysceptibility tensor on the boundary */
  void sustensorRightX(double** susxx, double** susyx, double** suszx);
  void sustensorLeftX(double** susxx, double** susyx, double** suszx);
  void sustensorRightY(double** susxy, double** susyy, double** suszy);
  void sustensorLeftY(double** susxy, double** susyy, double** suszy);
  void sustensorRightZ(double** susxz, double** susyz, double** suszz);
  void sustensorLeftZ(double** susxz, double** susyz, double** suszz);

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
  const_arr4_pfloat get_fieldForPcls() { return fieldForPcls; }
  arr3_double getEx() { return Ex; }
  arr3_double getEy() { return Ey; }
  arr3_double getEz() { return Ez; }
  arr3_double getBx() { return Bxn; }
  arr3_double getBy() { return Byn; }
  arr3_double getBz() { return Bzn; }

  // for parallel vtk
  arr3_double getBxc() { return Bxc; };
  arr3_double getByc() { return Byc; };
  arr3_double getBzc() { return Bzc; };

  arr3_double getRHOc() { return rhoc; }
  arr3_double getRHOn() { return rhon; }
  double getRHOc(int X, int Y, int Z) const { return rhoc.get(X, Y, Z); }
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
  // per-species 3D slice accessors for parallel HDF5 output
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

  // per-species 3D slice accessors (node grid) for parallel HDF5 / H5hut output
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
  // per-species 3D slice accessors (node grid) for parallel HDF5 output
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

  /*! get the electric field energy */
  double getEenergy();
  /*! get the magnetic field energy */
  double getBenergy();
  /*! get bulk kinetic energy */
  double getBulkEnergy(int is);

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

#ifdef GPU_SOLVER
  // ---- GPU Solver: lifecycle ----
  /** Allocate all GPU-resident field arrays (called once after construction).
   */
  void gpuSolverAllocate();
  /** Free all GPU-resident field arrays. */
  void gpuSolverFree();

  /** Return the dedicated solver CUDA stream. */
  cudaStream_t gpuSolverStream() const { return solverStream_; }
  /** Synchronise the solver stream (block host until all solver work is done).
   */
  void gpuSolverStreamSync() { cudaStreamSynchronize(solverStream_); }

  // ---- GPU Solver: host ↔ device synchronisation ----
  /** Copy all primary field arrays from host to device (for initialisation /
   * restart). */
  void gpuSolverSyncH2D(cudaStream_t stream = 0);
  /** Copy field arrays from device to host (for I/O output). */
  void gpuSolverSyncD2H(cudaStream_t stream = 0);
  // ---- GPU Solver: GPU-aware MPI halo exchange ----
  /** Center-based halo exchange using GPU-aware MPI (operates on device
   * pointer). */
  void gpuCommunicateCenterBC(int nx, int ny, int nz, GPUFieldArray3& gpuArr,
                              int bcFaceXright, int bcFaceXleft,
                              int bcFaceYright, int bcFaceYleft,
                              int bcFaceZright, int bcFaceZleft);
  /** Batched 3-field center-based halo exchange + BC in one MPI round. */
  void gpuCommunicateCenterBC_3(int nx, int ny, int nz, GPUFieldArray3& a1,
                                GPUFieldArray3& a2, GPUFieldArray3& a3,
                                int bcXR, int bcXL, int bcYR, int bcYL,
                                int bcZR, int bcZL);
  /** Batched 9-field center-based halo exchange + BC in one MPI round (for
   * fused triple Laplacian). */
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
  /** GPU version of calculateE. Falls back to CPU if GPU_SOLVER off. */
  void gpuCalculateE(int cycle);
  /** GPU version of calculateB. */
  void gpuCalculateB(int cycle);
  /** GPU version of calculateHatFunctions. */
  void gpuCalculateHatFunctions();
  /** GPU MaxwellImage: A*x callback for GMRES (operates on device Krylov
   * vectors). */
  void gpuMaxwellImage(cudaSolverType* d_im, cudaSolverType* d_vector);
  /** GPU MaxwellImage (local): communication-free A*x for use as
   * preconditioner. Ghost cells are treated as zero and physical boundary image
   * corrections are enforced locally so the operator better matches the full
   * Maxwell image. */
  void gpuMaxwellImageLocal(cudaSolverType* d_im, cudaSolverType* d_vector);
  /** GPU MaxwellSource: build RHS of Maxwell system (result in device Krylov
   * vector). */
  void gpuMaxwellSource(cudaSolverType* d_bkrylov);

  // ---- GPU Chebyshev Semi-Iterative Solver ----
  /** Full Chebyshev solver (with MPI communication).
   *  Solves  A·x = b  in Krylov space using Chebyshev polynomial acceleration.
   *  @param d_x      [in/out] initial guess → solution (Krylov vector, device)
   *  @param n         Krylov vector length = 3*(nxn-2)*(nyn-2)*(nzn-2)
   *  @param d_b       [in] right-hand side (Krylov vector, device)
   *  @param GpuImage  operator callback A·v  (e.g.
   * &EMfields3D::gpuMaxwellImage)
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
   *  Builds the 3×3 diagonal block D_i of the Maxwell operator at each
   *  node and solves D_i z_i = r_i via Cramer's rule.
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
  void gpuFGMRES_BlockJacobiPrecond(cudaSolverType* d_x, int n,
                                    cudaSolverType* d_b, int m, int max_iter,
                                    cudaSolverType tol, MPI_Comm fieldcomm);

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
  /** GPU MUdot: compute μ·E for all species. */
  void gpuMUdot(GPUFieldArray3& MUdotX, GPUFieldArray3& MUdotY,
                GPUFieldArray3& MUdotZ, GPUFieldArray3& vX, GPUFieldArray3& vY,
                GPUFieldArray3& vZ);
  /** GPU PIdot: compute π·v for one species (accumulative). */
  void gpuPIdot(GPUFieldArray3& PIX, GPUFieldArray3& PIY, GPUFieldArray3& PIZ,
                GPUFieldArray3& vX, GPUFieldArray3& vY, GPUFieldArray3& vZ,
                int is);
  /** GPU smooth: SmoothNiter iterations of box smoothing. */
  void gpuSmooth(GPUFieldArray3& arr, int type);
  /** GPU smoothE: fused 3-component E-field smoothing. */
  void gpuSmoothE();
  /** GPU smooth3: fused 3-component node-field smoothing (for Jhat). */
  void gpuSmooth3(GPUFieldArray3& a1, GPUFieldArray3& a2, GPUFieldArray3& a3,
                  int type);
  /** GPU perfect conductor BC: left boundary. */
  void gpuPerfectConductorLeft(GPUFieldArray3& imX, GPUFieldArray3& imY,
                               GPUFieldArray3& imZ, GPUFieldArray3& vX,
                               GPUFieldArray3& vY, GPUFieldArray3& vZ, int dir);
  /** GPU perfect conductor BC: right boundary. */
  void gpuPerfectConductorRight(GPUFieldArray3& imX, GPUFieldArray3& imY,
                                GPUFieldArray3& imZ, GPUFieldArray3& vX,
                                GPUFieldArray3& vY, GPUFieldArray3& vZ,
                                int dir);
  /** Fused triple Laplacian: 3× lapN2N with ONE batched halo exchange. */
  void gpuLapN2N_3(GPUFieldArray3& lapA, GPUFieldArray3& fieldA,
                   GPUFieldArray3& lapB, GPUFieldArray3& fieldB,
                   GPUFieldArray3& lapC, GPUFieldArray3& fieldC);

  // ---- GPU Solver: moment post-processing on GPU ----
  /** D2D scatter: copy 10 packed moment arrays from the moment-kernel buffer
   *  (momentsSrc, layout [10][gridSize]) into the per-species slices of
   *  d_rhons, d_Jxs, d_Jys, d_Jzs, d_pXXsn, d_pXYsn, d_pXZsn,
   *  d_pYYsn, d_pYZsn, d_pZZsn.  Pure device-to-device, no host touch. */
  void gpuScatterMomentsD2D(cudaMomentType* momentsSrc, int species,
                            cudaStream_t stream = 0);

  /** Batched ghost exchange: all species at once (reduces MPI barriers). */
  void gpuCommunicateGhostP2G_AllSpecies();

  /** Batched GPU halo exchange: exchanges nFields 3D arrays in a single set
   *  of MPI messages per direction, using explicit CUDA pack/unpack into
   *  persistent contiguous GPU buffers.
   *  @param h_fieldPtrs  Host array of nFields device pointers.
   *  @param nFields      Number of field arrays to batch.
   *  @param nx,ny,nz     Grid dimensions (including ghosts).
   *  @param isCenterFlag true = center-grid offsets, false = node-grid.
   *  @param isFaceOnlyFlag true = skip edges/corners.
   *  @param needInterp   true = additive accumulation after exchange.
   *  @param isParticle   true = use particle communicator.
   *  @param stream       CUDA stream for kernels. */
  void gpuBatchedHaloExchange(cudaSolverType** h_fieldPtrs, int nFields, int nx,
                              int ny, int nz, bool isCenterFlag,
                              bool isFaceOnlyFlag, bool needInterp,
                              bool isParticle, cudaStream_t stream);

#ifdef HALO_OVERLAP
  /** Phase 1 of split halo exchange: pack boundary faces, post
   *  non-blocking MPI sends/receives, self-copy periodic faces.
   *  Returns the number of MPI requests stored in haloFaceRequests_. */
  int gpuBatchedHaloBeginExchange(cudaSolverType** h_fieldPtrs, int nFields,
                                  int nx, int ny, int nz, bool isCenterFlag,
                                  bool isFaceOnlyFlag, bool needInterp,
                                  bool isParticle, cudaStream_t stream);
  /** Phase 2 of split halo exchange: MPI_Waitall on face requests,
   *  unpack face buffers, then run edge + corner phases as usual. */
  void gpuBatchedHaloEndExchange(cudaSolverType** h_fieldPtrs, int nFields,
                                 int nx, int ny, int nz, bool isCenterFlag,
                                 bool isFaceOnlyFlag, bool needInterp,
                                 bool isParticle, cudaStream_t stream);
#endif

  /** Allocate persistent GPU halo exchange buffers. */
  void gpuAllocateHaloBuffers();
  /** Free persistent GPU halo exchange buffers. */
  void gpuFreeHaloBuffers();

  /** GPU setZeroDerivedMoments: zero Jx/Jy/Jz, Jxh/Jyh/Jzh, rhon, rhoc, rhoh on
   * device. */
  void gpuSetZeroDerivedMoments();

  /** GPU sumOverSpecies: rhon += sum_s(rhons_s) on device. */
  void gpuSumOverSpecies();

  /** GPU sumOverSpeciesJ: Jx/Jy/Jz = sum_s(Jxs/Jys/Jzs)_s on device. */
  void gpuSumOverSpeciesJ();

  /** GPU interpDensitiesN2C: rhoc = interpN2C(rhon) on device. */
  void gpuInterpDensitiesN2C();

  // ---- GPU Solver: open boundary conditions ----
  /** GPU open BC: zero source term on inflow faces. */
  void gpuOpenBoundaryInflowESource(cudaSolverType* dX, cudaSolverType* dY,
                                    cudaSolverType* dZ, int nx, int ny, int nz);
  /** GPU open BC: set image = vect - E_inj on inflow faces. */
  void gpuOpenBoundaryInflowEImage(cudaSolverType* imX, cudaSolverType* imY,
                                   cudaSolverType* imZ,
                                   const cudaSolverType* vX,
                                   const cudaSolverType* vY,
                                   const cudaSolverType* vZ, int nx, int ny,
                                   int nz);
  /** GPU open BC: SAL blend / Dirichlet on E inflow faces (post-solve). */
  void gpuOpenBoundaryInflowE(cudaSolverType* dX, cudaSolverType* dY,
                              cudaSolverType* dZ, int nx, int ny, int nz);
  /** GPU open BC: SAL blend / extrapolation on center B inflow faces. */
  void gpuOpenBoundaryInflowB(cudaSolverType* dX, cudaSolverType* dY,
                              cudaSolverType* dZ, int nx, int ny, int nz);

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
   * eigenvalues). */
  void gpuPoissonImageLocal(cudaSolverType* d_im, cudaSolverType* d_vec);
  /** Compute analytic eigenvalue bounds of the local -∇² operator (once per
   * simulation). */
  void computePoissonChebyshevEigenvalues();
  /** Chebyshev preconditioner for Poisson (communication-free, analytic
   * eigenvalues). */
  void gpuChebyshevPrecondPoisson(cudaSolverType* d_x, cudaSolverType* d_b);
  /** GPU FGMRES(m) with Chebyshev preconditioner for Poisson (divergence
   * cleaning). */
  void gpuFGMRES_PoissonChebyshev(cudaSolverType* d_x, int n,
                                  cudaSolverType* d_b, int m, int max_iter,
                                  cudaSolverType tol, MPI_Comm fieldcomm);
  /** GPU Poisson correction for div(E) cleaning. */
  void gpuPoissonCorrection(int cycle);
  /** GPU div(B) cleaning: solve lap(PSI)=div(B), correct B on boundary layers.
   */
  void gpuApplyDivBCleaning();

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
  GPUFieldArray3& gpuExth() { return d_Exth; }
  GPUFieldArray3& gpuEyth() { return d_Eyth; }
  GPUFieldArray3& gpuEzth() { return d_Ezth; }
  GPUFieldArray3& gpuBxn() { return d_Bxn; }
  GPUFieldArray3& gpuByn() { return d_Byn; }
  GPUFieldArray3& gpuBzn() { return d_Bzn; }
  GPUFieldArray3& gpuBxc() { return d_Bxc; }
  GPUFieldArray3& gpuByc() { return d_Byc; }
  GPUFieldArray3& gpuBzc() { return d_Bzc; }
  GPUFieldArray3& gpuBx_ext() { return d_Bx_ext; }
  GPUFieldArray3& gpuBy_ext() { return d_By_ext; }
  GPUFieldArray3& gpuBz_ext() { return d_Bz_ext; }
  GPUFieldArray4& gpuRhons() { return d_rhons; }
  GPUFieldArray4& gpuJxs() { return d_Jxs; }
  GPUFieldArray4& gpuJys() { return d_Jys; }
  GPUFieldArray4& gpuJzs() { return d_Jzs; }
  GPUFieldArray4& gpupXXsn() { return d_pXXsn; }
  GPUFieldArray4& gpupXYsn() { return d_pXYsn; }
  GPUFieldArray4& gpupXZsn() { return d_pXZsn; }
  GPUFieldArray4& gpupYYsn() { return d_pYYsn; }
  GPUFieldArray4& gpupYZsn() { return d_pYZsn; }
  GPUFieldArray4& gpupZZsn() { return d_pZZsn; }
  bool isGpuSolverAllocated() const { return gpuSolverAllocated_; }
#endif // GPU_SOLVER

public: // accessors
  const Collective& get_col() const { return _col; }
  const Grid& get_grid() const { return _grid; };
  const VirtualTopology3D& get_vct() const { return _vct; }
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
  /*! Boundary electron speed */
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

  /*! PSI: magnetic potential (indexX, indexY, indexZ), defined on central
   * points between nodes */
  array3_double PSI;

  /*! PHI: electric potential (indexX, indexY, indexZ), defined on central
   * points between nodes */
  array3_double PHI;

  // Electric field component used to move particles
  // organized for rapid access in mover_PC()
  // [This is the information transferred from cluster to booster].
  array4_pfloat fieldForPcls;

  // Electric field components defined on nodes
  //
  array3_double Ex;
  array3_double Ey;
  array3_double Ez;

  // implicit electric field components defined on nodes
  //
  array3_double Exth;
  array3_double Eyth;
  array3_double Ezth;

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

  // *************************************
  // TEMPORARY ARRAY
  // ************************************
  /*!some temporary arrays (for calculate hat functions) */
  array3_double tempXC;
  array3_double tempYC;
  array3_double tempZC;
  array3_double tempXN;
  array3_double tempYN;
  array3_double tempZN;
  /*! other temporary arrays (in MaxwellSource) */
  array3_double tempC;
  array3_double tempX;
  array3_double tempY;
  array3_double tempZ;
  array3_double temp2X;
  array3_double temp2Y;
  array3_double temp2Z;
  /*! and some for MaxwellImage */
  array3_double imageX;
  array3_double imageY;
  array3_double imageZ;
  array3_double Dx;
  array3_double Dy;
  array3_double Dz;
  array3_double vectX;
  array3_double vectY;
  array3_double vectZ;
  array3_double divC;
  // array3_double arr;

  // *******************************************************************************
  // *********** SOURCES **
  // *******************************************************************************

  /*! Charge density, defined on central points of the cell */
  array3_double rhoc;
  /*! Charge density, defined on nodes */
  array3_double rhon;
  /*! Implicit charge density, defined on central points of the cell */
  array3_double rhoh;
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

  // implicit current density defined on nodes
  //
  array3_double Jxh;
  array3_double Jyh;
  array3_double Jzh;

  // species-specific current densities defined on nodes
  //
  array4_double Jxs;
  array4_double Jys;
  array4_double Jzs;

  // magnetic field components defined on nodes
  //
  array3_double Bx_ext;
  array3_double By_ext;
  array3_double Bz_ext;

  array3_double Bx_tot;
  array3_double By_tot;
  array3_double Bz_tot;

  // external current, defined on nodes
  array3_double Jx_ext;
  array3_double Jy_ext;
  array3_double Jz_ext;

  // pressure tensor components, defined on nodes
  array4_double pXXsn;
  array4_double pXYsn;
  array4_double pXZsn;
  array4_double pYYsn;
  array4_double pYZsn;
  array4_double pZZsn;
  array4_double heatFlux;

  /*! Field Boundary Condition
    0 = Dirichlet Boundary Condition: specifies the
        value on the boundary of the domain
    1 = Neumann Boundary Condition: specifies the value of
        derivative on the boundary of the domain
    2 = Periodic boundary condition */

  // boundary conditions for electrostatic potential
  //
  int bcPHIfaceXright;
  int bcPHIfaceXleft;
  int bcPHIfaceYright;
  int bcPHIfaceYleft;
  int bcPHIfaceZright;
  int bcPHIfaceZleft;

  /*! Boundary condition for electric field 0 = perfect conductor 1 = magnetic
   * mirror */
  //
  // boundary conditions for EM field
  //
  int bcEMfaceXright;
  int bcEMfaceXleft;
  int bcEMfaceYright;
  int bcEMfaceYleft;
  int bcEMfaceZright;
  int bcEMfaceZleft;

  // Absorbing boundary parameters
  int yes_sal;
  int n_layers_sal;

  /*! GEM Challenge background ion */
  double* rhoINIT;
  /*! Drift of the species */
  bool* DriftSpecies;

  /*! boolean for divergence cleaning */
  bool PoissonCorrection;
  int PoissonCorrectionCycle;
  bool divBCorrection;
  int divBCorrectionCycle;

  // persistent arrays for divB cleaning (avoid per-cycle allocation)
  array3_double divBwork;
  array3_double gradPSIX;
  array3_double gradPSIY;
  array3_double gradPSIZ;
  double* xkrylovPoisson_B;
  double* bkrylovPoisson_B;

  // Persistent arrays for calculateE (avoid per-cycle allocation)
  double* xkrylovMaxwell;
  double* bkrylovMaxwell;
  double* xkrylovPoisson_E;
  double* bkrylovPoisson_E;
  array3_double divE_work;
  array3_double gradPHIX_work;
  array3_double gradPHIY_work;
  array3_double gradPHIZ_work;

  // Persistent arrays for PoissonImage (avoid per-GMRES-iteration allocation)
  array3_double poissonTemp;
  array3_double poissonIm;

  // Persistent temp buffer for smooth (avoid per-call allocation)
  array3_double smoothTemp;

  /*! RESTART BOOLEAN */
  int restart1;

  /*! CG tolerance criterium for stopping iterations */
  double CGtol;
  /*! GMRES tolerance criterium for stopping iterations */
  double GMREStol;
  /*! Solver type for Maxwell: "GMRES" or "Chebyshev" */
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

  void OpenBoundaryInflowB(arr3_double vectorX, arr3_double vectorY,
                           arr3_double vectorZ,

                           int nx, int ny, int nz);
  void OpenBoundaryInflowE(arr3_double vectorX, arr3_double vectorY,
                           arr3_double vectorZ, int nx, int ny, int nz);
  void OpenBoundaryInflowEImage(arr3_double imageX, arr3_double imageY,
                                arr3_double imageZ, const_arr3_double vectorX,
                                const_arr3_double vectorY,
                                const_arr3_double vectorZ, int nx, int ny,
                                int nz);
  void OpenBoundaryInflowESource(arr3_double vectorX, arr3_double vectorY,
                                 arr3_double vectorZ, int nx, int ny, int nz);

#ifdef GPU_SOLVER
  // =========================================================================
  //  GPU-resident copies of all field arrays
  //  -----------------------------------------------------------------------
  //  Layout is identical to the host arrays (row-major, contiguous), so the
  //  same MPI derived datatypes can be used with GPU-aware MPI by passing
  //  the device pointer instead of the host pointer.
  // =========================================================================

  // Electric field (node-based)
  GPUFieldArray3 d_Ex, d_Ey, d_Ez;
  GPUFieldArray3 d_Exth, d_Eyth, d_Ezth;

  // Magnetic field (center-based)
  GPUFieldArray3 d_Bxc, d_Byc, d_Bzc;
  // Magnetic field (node-based)
  GPUFieldArray3 d_Bxn, d_Byn, d_Bzn;

  // Charge / current densities (node-based, summed over species)
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

  // External / total B (node-based)
  GPUFieldArray3 d_Bx_ext, d_By_ext, d_Bz_ext;

  // Temporary / work arrays (node-based)
  GPUFieldArray3 d_tempXC, d_tempYC, d_tempZC;
  GPUFieldArray3 d_tempXN, d_tempYN, d_tempZN;
  GPUFieldArray3 d_tempC;
  GPUFieldArray3 d_tempX, d_tempY, d_tempZ;
  GPUFieldArray3 d_temp2X, d_temp2Y, d_temp2Z;
  GPUFieldArray3 d_imageX, d_imageY, d_imageZ;
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

  // calculateE work arrays
  GPUFieldArray3 d_divE_work;
  GPUFieldArray3 d_gradPHIX_work, d_gradPHIY_work, d_gradPHIZ_work;

  // Poisson image work arrays
  GPUFieldArray3 d_poissonTemp, d_poissonIm;

  // Smooth temp buffer
  GPUFieldArray3 d_smoothTemp;

  // Device copy of species q/m for perfectConductor kernels
  cudaSolverType* d_qom = nullptr;

  // Reduction scratch buffer for GPU BLAS dot/norm operations
  cudaSolverType* d_blasScratch = nullptr;

  // GPU GMRES workspace, persistently allocated for max(Maxwell, Poisson).
  cudaSolverType* d_gmresV = nullptr; // [GMRES_MP1][max krylov length]
  cudaSolverType* d_gmresW = nullptr; // [max krylov length]
  int gmresVAlloc = 0;                // allocated d_gmresV element count

  // GPU FGMRES workspace (Z basis = preconditioned vectors), allocated on first
  // FGMRES use.
  cudaSolverType* d_fgmresZ = nullptr; // [GMRES_M][max krylov length]
  int fgmresZAlloc = 0;                // allocated d_fgmresZ element count

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

  // ---- Persistent batched halo-exchange buffers ----
  // 6 directions: 0=XL 1=XR 2=YL 3=YR 4=ZL 5=ZR
  // HALO_MAX_BATCH is the max number of fields per single batched exchange.
  // Functions with more fields (e.g. gpuCommunicateGhostP2G_AllSpecies)
  // automatically split into multiple passes of this size.
  static constexpr int HALO_MAX_BATCH = 64;
  double* d_haloBuf_send_[6] = {}; // contiguous GPU send buffers
  double* d_haloBuf_recv_[6] = {}; // contiguous GPU recv buffers
  double** d_ptrArray_ = nullptr;  // device array of field pointers for kernels
  double** h_ptrArray_ =
      nullptr; // pinned host staging for d_ptrArray_ H→D copies
  bool haloBufsAllocated_ = false;

#ifdef HALO_OVERLAP
  // ---- Split halo exchange state ----
  MPI_Request haloFaceRequests_[12] = {};
  int haloFaceReqCount_ = 0;
#endif // HALO_OVERLAP

  // Flag tracking whether GPU solver arrays have been allocated
  bool gpuSolverAllocated_ = false;
#endif // GPU_SOLVER
};

typedef EMfields3D Field;

#endif // EM_FIELDS_3D_H
