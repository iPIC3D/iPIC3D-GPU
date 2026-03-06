/* iPIC3D was originally developed by Stefano Markidis and Giovanni Lapenta.
 * This release was contributed by Alec Johnson and Ivy Bo Peng.
 * Publications that use results from iPIC3D need to properly cite
 * 'S. Markidis, G. Lapenta, and Rizwan-uddin. "Multi-scale simulations of
 * plasma with iPIC3D." Mathematics and Computers in Simulation 80.7 (2010): 1509-1519.'
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

#include <mpi.h>
#include "ipichdf5.h"
#include "EMfields3D.h"
#include "Collective.h"
#include "Basic.h"
#include "Com3DNonblk.h"
#include "VCtopology3D.h"
#include "Grid3DCU.h"
#include "CG.h"
#include "GMRES.h"
#include "Particles3Dcomm.h"
#include "Moments.h"
#include "Parameters.h"
#include "ompdefs.h"
#include "debug.h"
#include "string.h"
#include "mic_particles.h"
#include "TimeTasks.h"
#include "ipicmath.h" // for roundup_to_multiple
#include "Alloc.h"
#include "asserts.h"
#ifndef NO_HDF5
#endif

#include "cudaTypeDef.cuh"

#ifdef GPU_SOLVER
#include "GPUBlas.cuh"
#include "GPUStencils.cuh"
#include "GPUPhysicsKernels.cuh"

#ifdef HALO_OVERLAP
// Forward declarations for BC face functions (defined in GPUHaloComm.cu).
// Cannot include GPUHaloComm.cuh here because it contains __global__ decls
// and this file is compiled by the host compiler.
void gpuBCface(int nx, int ny, int nz, GPUFieldArray3& gpuArr,
               int bcFaceXright, int bcFaceXleft,
               int bcFaceYright, int bcFaceYleft,
               int bcFaceZright, int bcFaceZleft,
               const VirtualTopology3D* vct,
               cudaStream_t stream);
void gpuBCface_P(int nx, int ny, int nz, GPUFieldArray3& gpuArr,
                 int bcFaceXright, int bcFaceXleft,
                 int bcFaceYright, int bcFaceYleft,
                 int bcFaceZright, int bcFaceZleft,
                 const VirtualTopology3D* vct,
                 cudaStream_t stream);
#endif // HALO_OVERLAP
#endif // GPU_SOLVER

#include <algorithm>
#include <iostream>
#include <chrono>
#include <vector>
#include <cmath>
// #include <sstream>
using std::cout;
using std::endl;
using namespace iPic3D;

// Note:
// Open BCS inflow on E field are applied only on domain boundary faces
// where particle BCS are set to reemission (bcPface == 2).
// The master switch ApplyInflowBcsEImage (read from input file, default true)
// controls whether inflow BCs are applied in the GMRes image iterations.

/*! constructor */
//
// We rely on the following rule from the C++ standard, section 12.6.2.5:
//
//   nonstatic data members shall be initialized in the order
//   they were declared in the class definition
//
// in particular, nxc, nyc, nzc and nxn, nyn, nzn are assumed
// initialized when subsequently used.
//
EMfields3D::EMfields3D(Collective *col, Grid *grid, VirtualTopology3D *vct) : 
  _col(*col),
  _grid(*grid),
  _vct(*vct),
  nxc(grid->getNXC()),
  nxn(grid->getNXN()),
  nyc(grid->getNYC()),
  nyn(grid->getNYN()),
  nzc(grid->getNZC()),
  nzn(grid->getNZN()),
  dx(grid->getDX()),
  dy(grid->getDY()),
  dz(grid->getDZ()),
  invVOL(grid->getInvVOL()),
  xStart(grid->getXstart()),
  xEnd(grid->getXend()),
  yStart(grid->getYstart()),
  yEnd(grid->getYend()),
  zStart(grid->getZstart()),
  zEnd(grid->getZend()),
  Lx(col->getLx()),
  Ly(col->getLy()),
  Lz(col->getLz()),
  ns(col->getNs()),
  c(col->getC()),
  dt(col->getDt()),
  th(col->getTh()),
  ue0(col->getU0(0)),
  ve0(col->getV0(0)),
  we0(col->getW0(0)),
  x_center_dipole(col->getx_center_dipole()),
  y_center_dipole(col->gety_center_dipole()),
  z_center_dipole(col->getz_center_dipole()),
  x_center_planet(col->getx_center_planet()),
  y_center_planet(col->gety_center_planet()),
  z_center_planet(col->getz_center_planet()),
  L_square(col->getL_square()),
  delt(c * th * dt), // declared after these
  //
  // array allocation: nodes
  //
  fieldForPcls(nxn, nyn, nzn, 2 * DFIELD_3or4),
  Ex(nxn, nyn, nzn),
  Ey(nxn, nyn, nzn),
  Ez(nxn, nyn, nzn),
  Exth(nxn, nyn, nzn),
  Eyth(nxn, nyn, nzn),
  Ezth(nxn, nyn, nzn),
  Bxn(nxn, nyn, nzn),
  Byn(nxn, nyn, nzn),
  Bzn(nxn, nyn, nzn),
  rhon(nxn, nyn, nzn),
  Jx(nxn, nyn, nzn),
  Jy(nxn, nyn, nzn),
  Jz(nxn, nyn, nzn),
  Jxh(nxn, nyn, nzn),
  Jyh(nxn, nyn, nzn),
  Jzh(nxn, nyn, nzn),
  //
  // species-specific quantities
  //
  rhons(ns, nxn, nyn, nzn),
  rhocs(ns, nxc, nyc, nzc),
  Jxs(ns, nxn, nyn, nzn),
  Jys(ns, nxn, nyn, nzn),
  Jzs(ns, nxn, nyn, nzn),
  pXXsn(ns, nxn, nyn, nzn),
  pXYsn(ns, nxn, nyn, nzn),
  pXZsn(ns, nxn, nyn, nzn),
  pYYsn(ns, nxn, nyn, nzn),
  pYZsn(ns, nxn, nyn, nzn),
  pZZsn(ns, nxn, nyn, nzn),

  // array allocation: central points
  //
  PSI(nxc, nyc, nzc),
  PHI(nxc, nyc, nzc),
  Bxc(nxc, nyc, nzc),
  Byc(nxc, nyc, nzc),
  Bzc(nxc, nyc, nzc),
  rhoc(nxc, nyc, nzc),
  rhoh(nxc, nyc, nzc),

  // temporary arrays
  //
  tempXC(nxc, nyc, nzc),
  tempYC(nxc, nyc, nzc),
  tempZC(nxc, nyc, nzc),
  //
  tempXN(nxn, nyn, nzn),
  tempYN(nxn, nyn, nzn),
  tempZN(nxn, nyn, nzn),
  tempC(nxc, nyc, nzc),
  tempX(nxn, nyn, nzn),
  tempY(nxn, nyn, nzn),
  tempZ(nxn, nyn, nzn),
  temp2X(nxn, nyn, nzn),
  temp2Y(nxn, nyn, nzn),
  temp2Z(nxn, nyn, nzn),
  imageX(nxn, nyn, nzn),
  imageY(nxn, nyn, nzn),
  imageZ(nxn, nyn, nzn),
  Dx(nxn, nyn, nzn),
  Dy(nxn, nyn, nzn),
  Dz(nxn, nyn, nzn),
  vectX(nxn, nyn, nzn),
  vectY(nxn, nyn, nzn),
  vectZ(nxn, nyn, nzn),
  divC(nxc, nyc, nzc),
  // arr (nxc-2,nyc-2,nzc-2),
  //  B_ext and J_ext should not be allocated unless used.
  Bx_ext(nxn, nyn, nzn),
  By_ext(nxn, nyn, nzn),
  Bz_ext(nxn, nyn, nzn),
  Bx_tot(nxn, nyn, nzn),
  By_tot(nxn, nyn, nzn),
  Bz_tot(nxn, nyn, nzn),
  Jx_ext(nxn, nyn, nzn),
  Jy_ext(nxn, nyn, nzn),
  Jz_ext(nxn, nyn, nzn),
  // persistent arrays for divB cleaning
  divBwork(nxc, nyc, nzc),
  gradPSIX(nxn, nyn, nzn),
  gradPSIY(nxn, nyn, nzn),
  gradPSIZ(nxn, nyn, nzn),
  // persistent arrays for calculateE
  divE_work(nxc, nyc, nzc),
  gradPHIX_work(nxn, nyn, nzn),
  gradPHIY_work(nxn, nyn, nzn),
  gradPHIZ_work(nxn, nyn, nzn),
  // persistent arrays for PoissonImage
  poissonTemp(nxc, nyc, nzc),
  poissonIm(nxc, nyc, nzc),
  // persistent temp buffer for smooth
  smoothTemp(nxn, nyn, nzn)
{
  // allocate persistent Krylov vectors for divB cleaning
  const int nPoissonKrylov = (nxc - 2) * (nyc - 2) * (nzc - 2);
  xkrylovPoisson_B = new double[nPoissonKrylov];
  bkrylovPoisson_B = new double[nPoissonKrylov];

  // allocate persistent Krylov vectors for calculateE
  const int nMaxwellKrylov = 3 * (nxn - 2) * (nyn - 2) * (nzn - 2);
  xkrylovMaxwell = new double[nMaxwellKrylov];
  bkrylovMaxwell = new double[nMaxwellKrylov];
  const int nPoissonKrylov_E = (nxc - 2) * (nyc - 2) * (nzc - 2);
  xkrylovPoisson_E = new double[nPoissonKrylov_E];
  bkrylovPoisson_E = new double[nPoissonKrylov_E];

  // External imposed fields
  //
  B1x = col->getB1x();
  B1y = col->getB1y();
  B1z = col->getB1z();
  // if(B1x!=0. || B1y !=0. || B1z!=0.)
  //{
  //   eprintf("This functionality has not yet been implemented");
  // }
  Bx_ext.setall(0.);
  By_ext.setall(0.);
  Bz_ext.setall(0.);
  Bx_tot.setall(0.);
  By_tot.setall(0.);
  Bz_tot.setall(0.);
  //
  PoissonCorrection = false;
  if (col->getPoissonCorrection() == "yes")
  {
    PoissonCorrection = true;
    PoissonCorrectionCycle = col->getPoissonCorrectionCycle();
  }
  divBCorrection = false;
  if (col->getdivBCorrection() == "yes")
  {
    divBCorrection = true;
    divBCorrectionCycle = col->getdivBCorrectionCycle();
  }
  CGtol = col->getCGtol();
  GMREStol = col->getGMREStol();
  qom = new double[ns];
  for (int i = 0; i < ns; i++)
    qom[i] = col->getQOM(i);
  // boundary conditions: PHI and EM fields
  bcPHIfaceXright = col->getBcPHIfaceXright();
  bcPHIfaceXleft = col->getBcPHIfaceXleft();
  bcPHIfaceYright = col->getBcPHIfaceYright();
  bcPHIfaceYleft = col->getBcPHIfaceYleft();
  bcPHIfaceZright = col->getBcPHIfaceZright();
  bcPHIfaceZleft = col->getBcPHIfaceZleft();

  bcEMfaceXright = col->getBcEMfaceXright();
  bcEMfaceXleft = col->getBcEMfaceXleft();
  bcEMfaceYright = col->getBcEMfaceYright();
  bcEMfaceYleft = col->getBcEMfaceYleft();
  bcEMfaceZright = col->getBcEMfaceZright();
  bcEMfaceZleft = col->getBcEMfaceZleft();
  // absorbing boundary
  yes_sal = col->getYes_sal();
  n_layers_sal = col->getN_layers_sal();
  // GEM challenge parameters
  B0x = col->getB0x();
  B0y = col->getB0y();
  B0z = col->getB0z();
  delta = col->getDelta();
  Smooth = col->getSmooth();
  SmoothNiter = col->getSmoothNiter();
  // get the density background for the gem Challange
  rhoINIT = new double[ns];
  DriftSpecies = new bool[ns];
  for (int i = 0; i < ns; i++)
  {
    rhoINIT[i] = col->getRHOinit(i);
    if ((fabs(col->getW0(i)) != 0) || (fabs(col->getU0(i)) != 0)) // GEM and LHDI
      DriftSpecies[i] = true;
    else
      DriftSpecies[i] = false;
  }
  /*! parameters for GEM challenge */
  FourPI = 16 * atan(1.0);
  /*! Restart */
  restart1 = col->getRestart_status();

  // Define MPI Derived Data types for Center Halo Exchange
  // For face exchange on X dir
  MPI_Type_vector((nyc - 2), (nzc - 2), nzc, MPI_DOUBLE, &yzFacetypeC);
  MPI_Type_commit(&yzFacetypeC);

  // For face exchange on Y dir
  MPI_Type_create_hvector((nxc - 2), (nzc - 2), (nzc * nyc * sizeof(double)), MPI_DOUBLE, &xzFacetypeC);
  MPI_Type_commit(&xzFacetypeC);

  MPI_Type_vector((nyc - 2), 1, nzc, MPI_DOUBLE, &yEdgetypeC);
  MPI_Type_commit(&yEdgetypeC);

  // For face exchangeg on Z dir
  MPI_Type_create_hvector((nxc - 2), 1, (nzc * nyc * sizeof(double)), yEdgetypeC, &xyFacetypeC);
  MPI_Type_commit(&xyFacetypeC);

  // 2 yEdgeType can be merged into one message
  MPI_Type_create_hvector(2, 1, (nzc - 1) * sizeof(double), yEdgetypeC, &yEdgetypeC2);
  MPI_Type_commit(&yEdgetypeC2);

  MPI_Type_contiguous((nzc - 2), MPI_DOUBLE, &zEdgetypeC);
  MPI_Type_commit(&zEdgetypeC);

  MPI_Type_create_hvector(2, (nzc - 2), (nxc - 1) * (nyc * nzc) * sizeof(double), MPI_DOUBLE, &zEdgetypeC2);
  MPI_Type_commit(&zEdgetypeC2);

  MPI_Type_vector((nxc - 2), 1, nyc * nzc, MPI_DOUBLE, &xEdgetypeC);
  MPI_Type_commit(&xEdgetypeC);
  MPI_Type_create_hvector(2, 1, (nyc - 1) * nzc * sizeof(double), xEdgetypeC, &xEdgetypeC2);
  MPI_Type_commit(&xEdgetypeC2);

  // corner used to communicate in x direction
  int blocklengthC[] = {1, 1, 1, 1};
  int displacementsC[] = {0, nzc - 1, (nyc - 1) * nzc, nyc * nzc - 1};
  MPI_Type_indexed(4, blocklengthC, displacementsC, MPI_DOUBLE, &cornertypeC);
  MPI_Type_commit(&cornertypeC);

  // Define MPI Derived Data types for Node Halo Exchange
  // For face exchange on X dir
  MPI_Type_vector((nyn - 2), (nzn - 2), nzn, MPI_DOUBLE, &yzFacetypeN);
  MPI_Type_commit(&yzFacetypeN);

  // For face exchange on Y dir
  MPI_Type_create_hvector((nxn - 2), (nzn - 2), (nzn * nyn * sizeof(double)), MPI_DOUBLE, &xzFacetypeN);
  MPI_Type_commit(&xzFacetypeN);

  MPI_Type_vector((nyn - 2), 1, nzn, MPI_DOUBLE, &yEdgetypeN);
  MPI_Type_commit(&yEdgetypeN);

  // For face exchangeg on Z dir
  MPI_Type_create_hvector((nxn - 2), 1, (nzn * nyn * sizeof(double)), yEdgetypeN, &xyFacetypeN);
  MPI_Type_commit(&xyFacetypeN);

  // 2 yEdgeType can be merged into one message
  MPI_Type_create_hvector(2, 1, (nzn - 1) * sizeof(double), yEdgetypeN, &yEdgetypeN2);
  MPI_Type_commit(&yEdgetypeN2);

  MPI_Type_contiguous((nzn - 2), MPI_DOUBLE, &zEdgetypeN);
  MPI_Type_commit(&zEdgetypeN);

  MPI_Type_create_hvector(2, (nzn - 2), (nxn - 1) * (nyn * nzn) * sizeof(double), MPI_DOUBLE, &zEdgetypeN2);
  MPI_Type_commit(&zEdgetypeN2);

  MPI_Type_vector((nxn - 2), 1, nyn * nzn, MPI_DOUBLE, &xEdgetypeN);
  MPI_Type_commit(&xEdgetypeN);
  MPI_Type_create_hvector(2, 1, (nyn - 1) * nzn * sizeof(double), xEdgetypeN, &xEdgetypeN2);
  MPI_Type_commit(&xEdgetypeN2);

  // corner used to communicate in x direction
  int blocklengthN[] = {1, 1, 1, 1};
  int displacementsN[] = {0, nzn - 1, (nyn - 1) * nzn, nyn * nzn - 1};
  MPI_Type_indexed(4, blocklengthN, displacementsN, MPI_DOUBLE, &cornertypeN);
  MPI_Type_commit(&cornertypeN);

  if (col->getWriteMethod() == "pvtk" || col->getWriteMethod() == "nbcvtk")
  {
    // test Endian
    int TestEndian = 1;
    lEndFlag = *(char *)&TestEndian;

    // create process file view
    int size[3], subsize[3], start[3];

    // 3D subarray - reverse X, Z
    //  Each process writes its interior nodes; upper-boundary processes
    //  also write the last physical boundary node (+1 in that direction).
    subsize[0] = (nzc - 2) + (vct->isZupper() ? 1 : 0);
    subsize[1] = (nyc - 2) + (vct->isYupper() ? 1 : 0);
    subsize[2] = (nxc - 2) + (vct->isXupper() ? 1 : 0);
    // Global node count = global cell count + 1
    size[0] = col->getNzc() + 1;
    size[1] = col->getNyc() + 1;
    size[2] = col->getNxc() + 1;
    // Start offset: use the regular (untruncated) block size
    const int nxc_rr = (col->getNxc() + col->getXLEN() - 1) / col->getXLEN();
    const int nyc_rr = (col->getNyc() + col->getYLEN() - 1) / col->getYLEN();
    const int nzc_rr = (col->getNzc() + col->getZLEN() - 1) / col->getZLEN();
    start[0] = vct->getCoordinates(2) * nzc_rr;
    start[1] = vct->getCoordinates(1) * nyc_rr;
    start[2] = vct->getCoordinates(0) * nxc_rr;

    MPI_Type_contiguous(3, MPI_FLOAT, &xyzcomp);
    MPI_Type_commit(&xyzcomp);

    MPI_Type_create_subarray(3, size, subsize, start, MPI_ORDER_C, xyzcomp, &procviewXYZ);
    MPI_Type_commit(&procviewXYZ);

    MPI_Type_create_subarray(3, size, subsize, start, MPI_ORDER_C, MPI_FLOAT, &procview);
    MPI_Type_commit(&procview);

    subsize[0] = nxc - 2;
    subsize[1] = nyc - 2;
    subsize[2] = nzc - 2;
    size[0] = nxc;
    size[1] = nyc;
    size[2] = nzc;
    start[0] = 1;
    start[1] = 1;
    start[2] = 1;
    MPI_Type_create_subarray(3, size, subsize, start, MPI_ORDER_C, MPI_FLOAT, &ghosttype);
    MPI_Type_commit(&ghosttype);
  }
}
void EMfields3D::freeDataType()
{
  MPI_Type_free(&yzFacetypeC);
  MPI_Type_free(&xzFacetypeC);
  MPI_Type_free(&xyFacetypeC);
  MPI_Type_free(&xEdgetypeC);
  MPI_Type_free(&yEdgetypeC);
  MPI_Type_free(&zEdgetypeC);
  MPI_Type_free(&xEdgetypeC2);
  MPI_Type_free(&yEdgetypeC2);
  MPI_Type_free(&zEdgetypeC2);
  MPI_Type_free(&cornertypeC);

  MPI_Type_free(&yzFacetypeN);
  MPI_Type_free(&xzFacetypeN);
  MPI_Type_free(&xyFacetypeN);
  MPI_Type_free(&xEdgetypeN);
  MPI_Type_free(&yEdgetypeN);
  MPI_Type_free(&zEdgetypeN);
  MPI_Type_free(&xEdgetypeN2);
  MPI_Type_free(&yEdgetypeN2);
  MPI_Type_free(&zEdgetypeN2);
  MPI_Type_free(&cornertypeN);

  if (_col.getWriteMethod() == "pvtk" || _col.getWriteMethod() == "nbcvtk")
  {
    MPI_Type_free(&procview);
    MPI_Type_free(&xyzcomp);
    MPI_Type_free(&procviewXYZ);
    MPI_Type_free(&ghosttype);
  }

  // free persistent Krylov vectors for divB cleaning
  delete[] xkrylovPoisson_B;
  delete[] bkrylovPoisson_B;

  // free persistent Krylov vectors for calculateE
  delete[] xkrylovMaxwell;
  delete[] bkrylovMaxwell;
  delete[] xkrylovPoisson_E;
  delete[] bkrylovPoisson_E;

#ifdef GPU_SOLVER
  gpuSolverFree();
#endif
}

// =========================================================================
//  GPU Solver: allocation / deallocation / synchronisation
// =========================================================================
#ifdef GPU_SOLVER

void EMfields3D::gpuSolverAllocate()
{
  if (gpuSolverAllocated_) return;

  // ---- Electric field (node-based) ----
  d_Ex   = GPUFieldArray3(nxn, nyn, nzn);
  d_Ey   = GPUFieldArray3(nxn, nyn, nzn);
  d_Ez   = GPUFieldArray3(nxn, nyn, nzn);
  d_Exth = GPUFieldArray3(nxn, nyn, nzn);
  d_Eyth = GPUFieldArray3(nxn, nyn, nzn);
  d_Ezth = GPUFieldArray3(nxn, nyn, nzn);

  // ---- Magnetic field ----
  d_Bxc = GPUFieldArray3(nxc, nyc, nzc);
  d_Byc = GPUFieldArray3(nxc, nyc, nzc);
  d_Bzc = GPUFieldArray3(nxc, nyc, nzc);
  d_Bxn = GPUFieldArray3(nxn, nyn, nzn);
  d_Byn = GPUFieldArray3(nxn, nyn, nzn);
  d_Bzn = GPUFieldArray3(nxn, nyn, nzn);

  // ---- Charge / current densities (node-based, summed over species) ----
  d_rhon = GPUFieldArray3(nxn, nyn, nzn);
  d_rhoc = GPUFieldArray3(nxc, nyc, nzc);
  d_rhoh = GPUFieldArray3(nxc, nyc, nzc);
  d_Jx   = GPUFieldArray3(nxn, nyn, nzn);
  d_Jy   = GPUFieldArray3(nxn, nyn, nzn);
  d_Jz   = GPUFieldArray3(nxn, nyn, nzn);
  d_Jxh  = GPUFieldArray3(nxn, nyn, nzn);
  d_Jyh  = GPUFieldArray3(nxn, nyn, nzn);
  d_Jzh  = GPUFieldArray3(nxn, nyn, nzn);

  // ---- Per-species densities and currents (node-based) ----
  d_rhons = GPUFieldArray4(ns, nxn, nyn, nzn);
  d_Jxs   = GPUFieldArray4(ns, nxn, nyn, nzn);
  d_Jys   = GPUFieldArray4(ns, nxn, nyn, nzn);
  d_Jzs   = GPUFieldArray4(ns, nxn, nyn, nzn);

  // ---- Pressure tensor (node-based, species-indexed) ----
  d_pXXsn = GPUFieldArray4(ns, nxn, nyn, nzn);
  d_pXYsn = GPUFieldArray4(ns, nxn, nyn, nzn);
  d_pXZsn = GPUFieldArray4(ns, nxn, nyn, nzn);
  d_pYYsn = GPUFieldArray4(ns, nxn, nyn, nzn);
  d_pYZsn = GPUFieldArray4(ns, nxn, nyn, nzn);
  d_pZZsn = GPUFieldArray4(ns, nxn, nyn, nzn);

  // ---- Potentials (center-based) ----
  d_PHI = GPUFieldArray3(nxc, nyc, nzc);
  d_PSI = GPUFieldArray3(nxc, nyc, nzc);

  // ---- External B (node-based) ----
  d_Bx_ext = GPUFieldArray3(nxn, nyn, nzn);
  d_By_ext = GPUFieldArray3(nxn, nyn, nzn);
  d_Bz_ext = GPUFieldArray3(nxn, nyn, nzn);

  // ---- Temporary / work arrays ----
  d_tempXC = GPUFieldArray3(nxc, nyc, nzc);
  d_tempYC = GPUFieldArray3(nxc, nyc, nzc);
  d_tempZC = GPUFieldArray3(nxc, nyc, nzc);
  d_tempXN = GPUFieldArray3(nxn, nyn, nzn);
  d_tempYN = GPUFieldArray3(nxn, nyn, nzn);
  d_tempZN = GPUFieldArray3(nxn, nyn, nzn);
  d_tempC  = GPUFieldArray3(nxc, nyc, nzc);
  d_tempX  = GPUFieldArray3(nxn, nyn, nzn);
  d_tempY  = GPUFieldArray3(nxn, nyn, nzn);
  d_tempZ  = GPUFieldArray3(nxn, nyn, nzn);
  d_temp2X = GPUFieldArray3(nxn, nyn, nzn);
  d_temp2Y = GPUFieldArray3(nxn, nyn, nzn);
  d_temp2Z = GPUFieldArray3(nxn, nyn, nzn);
  d_imageX = GPUFieldArray3(nxn, nyn, nzn);
  d_imageY = GPUFieldArray3(nxn, nyn, nzn);
  d_imageZ = GPUFieldArray3(nxn, nyn, nzn);
  d_Dx     = GPUFieldArray3(nxn, nyn, nzn);
  d_Dy     = GPUFieldArray3(nxn, nyn, nzn);
  d_Dz     = GPUFieldArray3(nxn, nyn, nzn);
  d_vectX  = GPUFieldArray3(nxn, nyn, nzn);
  d_vectY  = GPUFieldArray3(nxn, nyn, nzn);
  d_vectZ  = GPUFieldArray3(nxn, nyn, nzn);
  d_divC   = GPUFieldArray3(nxc, nyc, nzc);

  // ---- divB cleaning work arrays ----
  d_divBwork = GPUFieldArray3(nxc, nyc, nzc);
  d_gradPSIX = GPUFieldArray3(nxn, nyn, nzn);
  d_gradPSIY = GPUFieldArray3(nxn, nyn, nzn);
  d_gradPSIZ = GPUFieldArray3(nxn, nyn, nzn);

  // ---- Krylov vectors ----
  const int nMaxwellKrylov  = 3 * (nxn - 2) * (nyn - 2) * (nzn - 2);
  const int nPoissonKrylov  = (nxc - 2) * (nyc - 2) * (nzc - 2);
  d_xkrylovMaxwell   = GPUKrylovVector(nMaxwellKrylov);
  d_bkrylovMaxwell   = GPUKrylovVector(nMaxwellKrylov);
  d_xkrylovPoisson_B = GPUKrylovVector(nPoissonKrylov);
  d_bkrylovPoisson_B = GPUKrylovVector(nPoissonKrylov);
  d_xkrylovPoisson_E = GPUKrylovVector(nPoissonKrylov);
  d_bkrylovPoisson_E = GPUKrylovVector(nPoissonKrylov);

  // ---- calculateE work arrays ----
  d_divE_work     = GPUFieldArray3(nxc, nyc, nzc);
  d_gradPHIX_work = GPUFieldArray3(nxn, nyn, nzn);
  d_gradPHIY_work = GPUFieldArray3(nxn, nyn, nzn);
  d_gradPHIZ_work = GPUFieldArray3(nxn, nyn, nzn);

  // ---- Poisson image work arrays ----
  d_poissonTemp = GPUFieldArray3(nxc, nyc, nzc);
  d_poissonIm   = GPUFieldArray3(nxc, nyc, nzc);

  // ---- Smooth temp buffer ----
  d_smoothTemp = GPUFieldArray3(nxn, nyn, nzn);

  // ---- Device copy of qom ----
  cudaErrChk(cudaMalloc(&d_qom, ns * sizeof(double)));
  cudaErrChk(cudaMemcpy(d_qom, qom, ns * sizeof(double), cudaMemcpyHostToDevice));

  // ---- BLAS reduction scratch ----
  // Need (m+2) doubles for gpuBatchedDotNorm in GMRES (m=20 → 22 doubles).
  // Also used as 1-double scratch for individual gpuDot/gpuNorm2 calls.
  cudaErrChk(cudaMalloc(&d_blasScratch, (GMRES_M + 2) * sizeof(double)));

  // ---- Pinned host buffers for GMRES reductions ----
  cudaErrChk(cudaHostAlloc(&h_gmresReduceLocal,  GMRES_MP1 * sizeof(double), cudaHostAllocDefault));
  cudaErrChk(cudaHostAlloc(&h_gmresReduceGlobal, GMRES_MP1 * sizeof(double), cudaHostAllocDefault));
  cudaErrChk(cudaHostAlloc(&h_gmresH,  (size_t)GMRES_MP1 * GMRES_M * sizeof(double), cudaHostAllocDefault));
  cudaErrChk(cudaHostAlloc(&h_gmresG,  GMRES_MP1 * sizeof(double), cudaHostAllocDefault));
  cudaErrChk(cudaHostAlloc(&h_gmresCS, GMRES_M * sizeof(double), cudaHostAllocDefault));
  cudaErrChk(cudaHostAlloc(&h_gmresSN, GMRES_M * sizeof(double), cudaHostAllocDefault));
  cudaErrChk(cudaHostAlloc(&h_gmresY,  GMRES_MP1 * sizeof(double), cudaHostAllocDefault));

  // ---- GMRES workspace (lazy allocation in gpuCalculateE) ----
  d_gmresV    = nullptr;
  d_gmresW    = nullptr;
  gmresVAlloc = 0;

  // ---- Dedicated non-blocking solver stream ----
  cudaErrChk(cudaStreamCreateWithFlags(&solverStream_, cudaStreamNonBlocking));

  // ---- Persistent batched halo-exchange buffers ----
  gpuAllocateHaloBuffers();

  gpuSolverAllocated_ = true;
}

void EMfields3D::gpuSolverFree()
{
  if (!gpuSolverAllocated_) return;

  // Electric field
  d_Ex.free();   d_Ey.free();   d_Ez.free();
  d_Exth.free(); d_Eyth.free(); d_Ezth.free();

  // Magnetic field
  d_Bxc.free(); d_Byc.free(); d_Bzc.free();
  d_Bxn.free(); d_Byn.free(); d_Bzn.free();

  // Charge / current densities
  d_rhon.free(); d_rhoc.free(); d_rhoh.free();
  d_Jx.free();   d_Jy.free();   d_Jz.free();
  d_Jxh.free();  d_Jyh.free();  d_Jzh.free();

  // Per-species
  d_rhons.free();
  d_Jxs.free(); d_Jys.free(); d_Jzs.free();
  d_pXXsn.free(); d_pXYsn.free(); d_pXZsn.free();
  d_pYYsn.free(); d_pYZsn.free(); d_pZZsn.free();

  // Potentials
  d_PHI.free(); d_PSI.free();

  // External B
  d_Bx_ext.free(); d_By_ext.free(); d_Bz_ext.free();

  // Temporary arrays
  d_tempXC.free(); d_tempYC.free(); d_tempZC.free();
  d_tempXN.free(); d_tempYN.free(); d_tempZN.free();
  d_tempC.free();
  d_tempX.free();  d_tempY.free();  d_tempZ.free();
  d_temp2X.free(); d_temp2Y.free(); d_temp2Z.free();
  d_imageX.free(); d_imageY.free(); d_imageZ.free();
  d_Dx.free();     d_Dy.free();     d_Dz.free();
  d_vectX.free();  d_vectY.free();  d_vectZ.free();
  d_divC.free();

  // divB cleaning
  d_divBwork.free();
  d_gradPSIX.free(); d_gradPSIY.free(); d_gradPSIZ.free();

  // Krylov
  d_xkrylovMaxwell.free();   d_bkrylovMaxwell.free();
  d_xkrylovPoisson_B.free(); d_bkrylovPoisson_B.free();
  d_xkrylovPoisson_E.free(); d_bkrylovPoisson_E.free();

  // calculateE work
  d_divE_work.free();
  d_gradPHIX_work.free(); d_gradPHIY_work.free(); d_gradPHIZ_work.free();

  // Poisson image
  d_poissonTemp.free(); d_poissonIm.free();

  // Smooth temp
  d_smoothTemp.free();

  // qom device copy
  if (d_qom) { cudaFree(d_qom); d_qom = nullptr; }
  if (d_blasScratch) { cudaFree(d_blasScratch); d_blasScratch = nullptr; }
  if (d_gmresV) { cudaFree(d_gmresV); d_gmresV = nullptr; }
  if (d_gmresW) { cudaFree(d_gmresW); d_gmresW = nullptr; }
  gmresVAlloc = 0;

  // Free pinned GMRES host buffers
  if (h_gmresReduceLocal)  { cudaFreeHost(h_gmresReduceLocal);  h_gmresReduceLocal  = nullptr; }
  if (h_gmresReduceGlobal) { cudaFreeHost(h_gmresReduceGlobal); h_gmresReduceGlobal = nullptr; }
  if (h_gmresH)  { cudaFreeHost(h_gmresH);  h_gmresH  = nullptr; }
  if (h_gmresG)  { cudaFreeHost(h_gmresG);  h_gmresG  = nullptr; }
  if (h_gmresCS) { cudaFreeHost(h_gmresCS); h_gmresCS = nullptr; }
  if (h_gmresSN) { cudaFreeHost(h_gmresSN); h_gmresSN = nullptr; }
  if (h_gmresY)  { cudaFreeHost(h_gmresY);  h_gmresY  = nullptr; }

  // Free batched halo buffers
  gpuFreeHaloBuffers();

  // Destroy solver stream
  if (solverStream_) { cudaStreamDestroy(solverStream_); solverStream_ = 0; }

  gpuSolverAllocated_ = false;
}

// =========================================================================
//  Persistent batched halo-exchange buffer management
// =========================================================================

void EMfields3D::gpuAllocateHaloBuffers()
{
  if (haloBufsAllocated_) return;

  // Compute max per-field element count per direction across ALL phases
  // (face, edge, corner) to avoid overflow for skinny local domains.
  //
  // Face phase (per field):
  //   Dir 0,1 (XL,XR): (nyn-2)*(nzn-2)
  //   Dir 2,3 (YL,YR): (nxn-2)*(nzn-2)
  //   Dir 4,5 (ZL,ZR): (nxn-2)*(nyn-2)
  // Edge phase (per field, worst case both cross-edges active):
  //   Dir 0,1: 2*(nyn-2)   [Y-edges to X neighbours]
  //   Dir 2,3: 2*(nzn-2)   [Z-edges to Y neighbours]
  //   Dir 4,5: 2*(nxn-2)   [X-edges to Z neighbours]
  // Corner phase (per field): 4 per direction

  size_t faceSz[6], edgeSz[6];
  faceSz[0] = faceSz[1] = (size_t)(nyn - 2) * (nzn - 2);
  faceSz[2] = faceSz[3] = (size_t)(nxn - 2) * (nzn - 2);
  faceSz[4] = faceSz[5] = (size_t)(nxn - 2) * (nyn - 2);

  edgeSz[0] = edgeSz[1] = 2 * (size_t)(nyn - 2);
  edgeSz[2] = edgeSz[3] = 2 * (size_t)(nzn - 2);
  edgeSz[4] = edgeSz[5] = 2 * (size_t)(nxn - 2);

  constexpr size_t cornerSz = 4;  // 4 corners per direction

  for (int d = 0; d < 6; ++d) {
    size_t maxPerField = faceSz[d];
    if (edgeSz[d]  > maxPerField) maxPerField = edgeSz[d];
    if (cornerSz   > maxPerField) maxPerField = cornerSz;
    size_t bytes = maxPerField * HALO_MAX_BATCH * sizeof(double);
    cudaErrChk(cudaMalloc(&d_haloBuf_send_[d], bytes));
    cudaErrChk(cudaMalloc(&d_haloBuf_recv_[d], bytes));
  }
  cudaErrChk(cudaMalloc(&d_ptrArray_, HALO_MAX_BATCH * sizeof(double*)));
  cudaErrChk(cudaHostAlloc(&h_ptrArray_, HALO_MAX_BATCH * sizeof(double*), cudaHostAllocDefault));

  haloBufsAllocated_ = true;
}

void EMfields3D::gpuFreeHaloBuffers()
{
  if (!haloBufsAllocated_) return;

  for (int d = 0; d < 6; ++d) {
    if (d_haloBuf_send_[d]) { cudaFree(d_haloBuf_send_[d]); d_haloBuf_send_[d] = nullptr; }
    if (d_haloBuf_recv_[d]) { cudaFree(d_haloBuf_recv_[d]); d_haloBuf_recv_[d] = nullptr; }
  }
  if (d_ptrArray_) { cudaFree(d_ptrArray_); d_ptrArray_ = nullptr; }
  if (h_ptrArray_) { cudaFreeHost(h_ptrArray_); h_ptrArray_ = nullptr; }

  haloBufsAllocated_ = false;
}

void EMfields3D::gpuSolverSyncH2D(cudaStream_t stream)
{
  // Electric field
  d_Ex.copyFromHostAsync(Ex.fetch_arr(), stream);
  d_Ey.copyFromHostAsync(Ey.fetch_arr(), stream);
  d_Ez.copyFromHostAsync(Ez.fetch_arr(), stream);
  d_Exth.copyFromHostAsync(Exth.fetch_arr(), stream);
  d_Eyth.copyFromHostAsync(Eyth.fetch_arr(), stream);
  d_Ezth.copyFromHostAsync(Ezth.fetch_arr(), stream);

  // Magnetic field
  d_Bxc.copyFromHostAsync(Bxc.fetch_arr(), stream);
  d_Byc.copyFromHostAsync(Byc.fetch_arr(), stream);
  d_Bzc.copyFromHostAsync(Bzc.fetch_arr(), stream);
  d_Bxn.copyFromHostAsync(Bxn.fetch_arr(), stream);
  d_Byn.copyFromHostAsync(Byn.fetch_arr(), stream);
  d_Bzn.copyFromHostAsync(Bzn.fetch_arr(), stream);

  // Charge / current densities
  d_rhon.copyFromHostAsync(rhon.fetch_arr(), stream);
  d_rhoc.copyFromHostAsync(rhoc.fetch_arr(), stream);
  d_rhoh.copyFromHostAsync(rhoh.fetch_arr(), stream);
  d_Jx.copyFromHostAsync(Jx.fetch_arr(), stream);
  d_Jy.copyFromHostAsync(Jy.fetch_arr(), stream);
  d_Jz.copyFromHostAsync(Jz.fetch_arr(), stream);
  d_Jxh.copyFromHostAsync(Jxh.fetch_arr(), stream);
  d_Jyh.copyFromHostAsync(Jyh.fetch_arr(), stream);
  d_Jzh.copyFromHostAsync(Jzh.fetch_arr(), stream);

  // Per-species densities
  d_rhons.copyFromHostAsync(rhons.fetch_arr(), stream);
  d_Jxs.copyFromHostAsync(Jxs.fetch_arr(), stream);
  d_Jys.copyFromHostAsync(Jys.fetch_arr(), stream);
  d_Jzs.copyFromHostAsync(Jzs.fetch_arr(), stream);

  // Pressure tensor
  d_pXXsn.copyFromHostAsync(pXXsn.fetch_arr(), stream);
  d_pXYsn.copyFromHostAsync(pXYsn.fetch_arr(), stream);
  d_pXZsn.copyFromHostAsync(pXZsn.fetch_arr(), stream);
  d_pYYsn.copyFromHostAsync(pYYsn.fetch_arr(), stream);
  d_pYZsn.copyFromHostAsync(pYZsn.fetch_arr(), stream);
  d_pZZsn.copyFromHostAsync(pZZsn.fetch_arr(), stream);

  // Potentials
  d_PHI.copyFromHostAsync(PHI.fetch_arr(), stream);
  d_PSI.copyFromHostAsync(PSI.fetch_arr(), stream);

  // External B
  d_Bx_ext.copyFromHostAsync(Bx_ext.fetch_arr(), stream);
  d_By_ext.copyFromHostAsync(By_ext.fetch_arr(), stream);
  d_Bz_ext.copyFromHostAsync(Bz_ext.fetch_arr(), stream);

  cudaStreamSynchronize(stream);
}

void EMfields3D::gpuSolverSyncD2H(cudaStream_t stream)
{
  // Only copy field data needed for I/O or particle mover
  d_Ex.copyToHostAsync(Ex.fetch_arr(), stream);
  d_Ey.copyToHostAsync(Ey.fetch_arr(), stream);
  d_Ez.copyToHostAsync(Ez.fetch_arr(), stream);
  d_Exth.copyToHostAsync(Exth.fetch_arr(), stream);
  d_Eyth.copyToHostAsync(Eyth.fetch_arr(), stream);
  d_Ezth.copyToHostAsync(Ezth.fetch_arr(), stream);

  d_Bxc.copyToHostAsync(Bxc.fetch_arr(), stream);
  d_Byc.copyToHostAsync(Byc.fetch_arr(), stream);
  d_Bzc.copyToHostAsync(Bzc.fetch_arr(), stream);
  d_Bxn.copyToHostAsync(Bxn.fetch_arr(), stream);
  d_Byn.copyToHostAsync(Byn.fetch_arr(), stream);
  d_Bzn.copyToHostAsync(Bzn.fetch_arr(), stream);

  d_rhon.copyToHostAsync(rhon.fetch_arr(), stream);
  d_rhoc.copyToHostAsync(rhoc.fetch_arr(), stream);
  d_Jx.copyToHostAsync(Jx.fetch_arr(), stream);
  d_Jy.copyToHostAsync(Jy.fetch_arr(), stream);
  d_Jz.copyToHostAsync(Jz.fetch_arr(), stream);

  // Per-species 4D arrays: copy species-by-species to match the
  // per-species cudaHostRegister granularity.  Bulk cudaMemcpyAsync
  // on a host buffer whose sub-regions are separately pinned can
  // cause "invalid argument" errors.
  {
    const size_t speciesSlice = (size_t)nxn * nyn * nzn;
    const size_t sliceBytes   = speciesSlice * sizeof(double);
    for (int is = 0; is < ns; is++) {
      cudaErrChk(cudaMemcpyAsync(rhons.fetch_arr() + is * speciesSlice,
                                  d_rhons.speciesPtr(is), sliceBytes,
                                  cudaMemcpyDeviceToHost, stream));
      cudaErrChk(cudaMemcpyAsync(Jxs.fetch_arr()   + is * speciesSlice,
                                  d_Jxs.speciesPtr(is), sliceBytes,
                                  cudaMemcpyDeviceToHost, stream));
      cudaErrChk(cudaMemcpyAsync(Jys.fetch_arr()   + is * speciesSlice,
                                  d_Jys.speciesPtr(is), sliceBytes,
                                  cudaMemcpyDeviceToHost, stream));
      cudaErrChk(cudaMemcpyAsync(Jzs.fetch_arr()   + is * speciesSlice,
                                  d_Jzs.speciesPtr(is), sliceBytes,
                                  cudaMemcpyDeviceToHost, stream));
      cudaErrChk(cudaMemcpyAsync(pXXsn.fetch_arr() + is * speciesSlice,
                                  d_pXXsn.speciesPtr(is), sliceBytes,
                                  cudaMemcpyDeviceToHost, stream));
      cudaErrChk(cudaMemcpyAsync(pXYsn.fetch_arr() + is * speciesSlice,
                                  d_pXYsn.speciesPtr(is), sliceBytes,
                                  cudaMemcpyDeviceToHost, stream));
      cudaErrChk(cudaMemcpyAsync(pXZsn.fetch_arr() + is * speciesSlice,
                                  d_pXZsn.speciesPtr(is), sliceBytes,
                                  cudaMemcpyDeviceToHost, stream));
      cudaErrChk(cudaMemcpyAsync(pYYsn.fetch_arr() + is * speciesSlice,
                                  d_pYYsn.speciesPtr(is), sliceBytes,
                                  cudaMemcpyDeviceToHost, stream));
      cudaErrChk(cudaMemcpyAsync(pYZsn.fetch_arr() + is * speciesSlice,
                                  d_pYZsn.speciesPtr(is), sliceBytes,
                                  cudaMemcpyDeviceToHost, stream));
      cudaErrChk(cudaMemcpyAsync(pZZsn.fetch_arr() + is * speciesSlice,
                                  d_pZZsn.speciesPtr(is), sliceBytes,
                                  cudaMemcpyDeviceToHost, stream));
    }
  }

  d_PHI.copyToHostAsync(PHI.fetch_arr(), stream);
  d_PSI.copyToHostAsync(PSI.fetch_arr(), stream);

  cudaStreamSynchronize(stream);
}

// =========================================================================
//  GPU Solver: physics method implementations
// =========================================================================

void EMfields3D::gpuMUdot(GPUFieldArray3& MUdotX, GPUFieldArray3& MUdotY, GPUFieldArray3& MUdotZ,
                           GPUFieldArray3& vX, GPUFieldArray3& vY, GPUFieldArray3& vZ)
{
  for (int is = 0; is < ns; is++) {
    double beta = 0.5 * qom[is] * dt / c;
    double prefactor = FourPI / 2.0 * delt * dt / c * qom[is];
    gpuMUdotSpecies(MUdotX.devPtr(), MUdotY.devPtr(), MUdotZ.devPtr(),
                    vX.devPtr(), vY.devPtr(), vZ.devPtr(),
                    d_Bxn.devPtr(), d_Byn.devPtr(), d_Bzn.devPtr(),
                    d_Bx_ext.devPtr(), d_By_ext.devPtr(), d_Bz_ext.devPtr(),
                    d_rhons.speciesPtr(is),
                    beta, prefactor, nxn, nyn, nzn,
                    /*firstSpecies=*/(is == 0), solverStream_);
  }
}

void EMfields3D::gpuPIdot(GPUFieldArray3& PIX, GPUFieldArray3& PIY, GPUFieldArray3& PIZ,
                           GPUFieldArray3& vX, GPUFieldArray3& vY, GPUFieldArray3& vZ, int is)
{
  double beta = 0.5 * qom[is] * dt / c;
  gpuPIdotSpecies(PIX.devPtr(), PIY.devPtr(), PIZ.devPtr(),
                  vX.devPtr(), vY.devPtr(), vZ.devPtr(),
                  d_Bxn.devPtr(), d_Byn.devPtr(), d_Bzn.devPtr(),
                  d_Bx_ext.devPtr(), d_By_ext.devPtr(), d_Bz_ext.devPtr(),
                  beta, nxn, nyn, nzn, solverStream_);
}

void EMfields3D::gpuSmooth(GPUFieldArray3& arr, int type)
{
  if (Smooth == 1.0) return;

  const double alpha  = Smooth;
  const double beta3D = (1.0 - alpha) / 6.0;

  int nx, ny, nz;
  if (type == 0) { nx = nxc; ny = nyc; nz = nzc; }
  else           { nx = nxn; ny = nyn; nz = nzn; }

  size_t fieldSize = (size_t)nx * ny * nz;
  bool isCenter = (type == 0);

  for (int icount = 1; icount < SmoothNiter + 1; icount++) {
#ifdef HALO_OVERLAP
    double* ptr1[1] = { arr.devPtr() };
    gpuBatchedHaloBeginExchange(ptr1, 1, nx, ny, nz,
                                isCenter, true, false, true, solverStream_);
    gpuSmoothStep_interior(d_smoothTemp.devPtr(), arr.devPtr(), nx, ny, nz, alpha, beta3D, solverStream_);
    gpuBatchedHaloEndExchange(ptr1, 1, nx, ny, nz,
                              isCenter, true, false, true, solverStream_);
    gpuBCface_P(nx, ny, nz, arr, 2,2,2,2,2,2, &_vct, solverStream_);
    gpuSmoothStep_boundary(d_smoothTemp.devPtr(), arr.devPtr(), nx, ny, nz, alpha, beta3D, solverStream_);
#else
    if (type == 0)
      gpuCommunicateCenterBoxStencilBC_P(nx, ny, nz, arr, 2, 2, 2, 2, 2, 2);
    else
      gpuCommunicateNodeBoxStencilBC_P(nx, ny, nz, arr, 2, 2, 2, 2, 2, 2);
    gpuSmoothStep(d_smoothTemp.devPtr(), arr.devPtr(), nx, ny, nz, alpha, beta3D, solverStream_);
#endif
    gpuEq(arr.devPtr(), d_smoothTemp.devPtr(), fieldSize, solverStream_);
  }
}

void EMfields3D::gpuSmoothE()
{
  if (Smooth == 1.0) return;

  const Collective* col = &get_col();
  const double alpha  = Smooth;
  const double beta3D = (1.0 - alpha) / 6.0;
  size_t nodeSize = (size_t)nxn * nyn * nzn;

  for (int icount = 1; icount < SmoothNiter + 1; icount++) {
#ifdef HALO_OVERLAP
    double* eptrs[3] = { d_Ex.devPtr(), d_Ey.devPtr(), d_Ez.devPtr() };
    gpuBatchedHaloBeginExchange(eptrs, 3, nxn, nyn, nzn,
                                false, true, false, false, solverStream_);
    gpuSmoothStep_interior(d_tempX.devPtr(), d_Ex.devPtr(), nxn, nyn, nzn, alpha, beta3D, solverStream_);
    gpuSmoothStep_interior(d_tempY.devPtr(), d_Ey.devPtr(), nxn, nyn, nzn, alpha, beta3D, solverStream_);
    gpuSmoothStep_interior(d_tempZ.devPtr(), d_Ez.devPtr(), nxn, nyn, nzn, alpha, beta3D, solverStream_);
    gpuBatchedHaloEndExchange(eptrs, 3, nxn, nyn, nzn,
                              false, true, false, false, solverStream_);
    gpuBCface(nxn, nyn, nzn, d_Ex, col->bcEx[0], col->bcEx[1], col->bcEx[2], col->bcEx[3], col->bcEx[4], col->bcEx[5], &_vct, solverStream_);
    gpuBCface(nxn, nyn, nzn, d_Ey, col->bcEy[0], col->bcEy[1], col->bcEy[2], col->bcEy[3], col->bcEy[4], col->bcEy[5], &_vct, solverStream_);
    gpuBCface(nxn, nyn, nzn, d_Ez, col->bcEz[0], col->bcEz[1], col->bcEz[2], col->bcEz[3], col->bcEz[4], col->bcEz[5], &_vct, solverStream_);
    gpuSmoothStep_boundary(d_tempX.devPtr(), d_Ex.devPtr(), nxn, nyn, nzn, alpha, beta3D, solverStream_);
    gpuSmoothStep_boundary(d_tempY.devPtr(), d_Ey.devPtr(), nxn, nyn, nzn, alpha, beta3D, solverStream_);
    gpuSmoothStep_boundary(d_tempZ.devPtr(), d_Ez.devPtr(), nxn, nyn, nzn, alpha, beta3D, solverStream_);
#else
    // Batched: 3 fields in 1 MPI round instead of 3 sequential exchanges
    gpuCommunicateNodeBoxStencilBC_3mixed(nxn, nyn, nzn,
        d_Ex, col->bcEx, d_Ey, col->bcEy, d_Ez, col->bcEz);
    gpuSmoothStep(d_tempX.devPtr(),  d_Ex.devPtr(), nxn, nyn, nzn, alpha, beta3D, solverStream_);
    gpuSmoothStep(d_tempY.devPtr(),  d_Ey.devPtr(), nxn, nyn, nzn, alpha, beta3D, solverStream_);
    gpuSmoothStep(d_tempZ.devPtr(),  d_Ez.devPtr(), nxn, nyn, nzn, alpha, beta3D, solverStream_);
#endif

    gpuEq(d_Ex.devPtr(), d_tempX.devPtr(), nodeSize, solverStream_);
    gpuEq(d_Ey.devPtr(), d_tempY.devPtr(), nodeSize, solverStream_);
    gpuEq(d_Ez.devPtr(), d_tempZ.devPtr(), nodeSize, solverStream_);
  }
}

void EMfields3D::gpuSmooth3(GPUFieldArray3& a1, GPUFieldArray3& a2, GPUFieldArray3& a3, int type)
{
  if (Smooth == 1.0) return;

  const double alpha  = Smooth;
  const double beta3D = (1.0 - alpha) / 6.0;

  int nx, ny, nz;
  if (type == 0) { nx = nxc; ny = nyc; nz = nzc; }
  else           { nx = nxn; ny = nyn; nz = nzn; }

  size_t fieldSize = (size_t)nx * ny * nz;
  bool isCenter = (type == 0);

  for (int icount = 1; icount < SmoothNiter + 1; icount++) {
#ifdef HALO_OVERLAP
    double* s3ptrs[3] = { a1.devPtr(), a2.devPtr(), a3.devPtr() };
    gpuBatchedHaloBeginExchange(s3ptrs, 3, nx, ny, nz,
                                isCenter, true, false, true, solverStream_);
    gpuSmoothStep_interior(d_temp2X.devPtr(), a1.devPtr(), nx, ny, nz, alpha, beta3D, solverStream_);
    gpuSmoothStep_interior(d_temp2Y.devPtr(), a2.devPtr(), nx, ny, nz, alpha, beta3D, solverStream_);
    gpuSmoothStep_interior(d_temp2Z.devPtr(), a3.devPtr(), nx, ny, nz, alpha, beta3D, solverStream_);
    gpuBatchedHaloEndExchange(s3ptrs, 3, nx, ny, nz,
                              isCenter, true, false, true, solverStream_);
    gpuBCface_P(nx, ny, nz, a1, 2,2,2,2,2,2, &_vct, solverStream_);
    gpuBCface_P(nx, ny, nz, a2, 2,2,2,2,2,2, &_vct, solverStream_);
    gpuBCface_P(nx, ny, nz, a3, 2,2,2,2,2,2, &_vct, solverStream_);
    gpuSmoothStep_boundary(d_temp2X.devPtr(), a1.devPtr(), nx, ny, nz, alpha, beta3D, solverStream_);
    gpuSmoothStep_boundary(d_temp2Y.devPtr(), a2.devPtr(), nx, ny, nz, alpha, beta3D, solverStream_);
    gpuSmoothStep_boundary(d_temp2Z.devPtr(), a3.devPtr(), nx, ny, nz, alpha, beta3D, solverStream_);
#else
    // Batched: 3 fields in 1 MPI round
    if (type == 0)
      gpuCommunicateCenterBoxStencilBC_P_3(nx, ny, nz, a1, a2, a3, 2, 2, 2, 2, 2, 2);
    else
      gpuCommunicateNodeBoxStencilBC_P_3(nx, ny, nz, a1, a2, a3, 2, 2, 2, 2, 2, 2);
    gpuSmoothStep(d_temp2X.devPtr(), a1.devPtr(), nx, ny, nz, alpha, beta3D, solverStream_);
    gpuSmoothStep(d_temp2Y.devPtr(), a2.devPtr(), nx, ny, nz, alpha, beta3D, solverStream_);
    gpuSmoothStep(d_temp2Z.devPtr(), a3.devPtr(), nx, ny, nz, alpha, beta3D, solverStream_);
#endif

    gpuEq(a1.devPtr(), d_temp2X.devPtr(), fieldSize, solverStream_);
    gpuEq(a2.devPtr(), d_temp2Y.devPtr(), fieldSize, solverStream_);
    gpuEq(a3.devPtr(), d_temp2Z.devPtr(), fieldSize, solverStream_);
  }
}

void EMfields3D::gpuPerfectConductorLeft(
    GPUFieldArray3& imX, GPUFieldArray3& imY, GPUFieldArray3& imZ,
    GPUFieldArray3& vX, GPUFieldArray3& vY, GPUFieldArray3& vZ, int dir)
{
  ::gpuPerfectConductorLeft(
      imX.devPtr(), imY.devPtr(), imZ.devPtr(),
      vX.devPtr(), vY.devPtr(), vZ.devPtr(),
      d_Ex.devPtr(), d_Ey.devPtr(), d_Ez.devPtr(),
      d_Jxh.devPtr(), d_Jyh.devPtr(), d_Jzh.devPtr(),
      d_Bxn.devPtr(), d_Byn.devPtr(), d_Bzn.devPtr(),
      d_Bx_ext.devPtr(), d_By_ext.devPtr(), d_Bz_ext.devPtr(),
      d_rhons.devPtr(), d_qom, ns,
      dt, c, th, FourPI, delt,
      nxn, nyn, nzn, dir, solverStream_);
}

void EMfields3D::gpuPerfectConductorRight(
    GPUFieldArray3& imX, GPUFieldArray3& imY, GPUFieldArray3& imZ,
    GPUFieldArray3& vX, GPUFieldArray3& vY, GPUFieldArray3& vZ, int dir)
{
  ::gpuPerfectConductorRight(
      imX.devPtr(), imY.devPtr(), imZ.devPtr(),
      vX.devPtr(), vY.devPtr(), vZ.devPtr(),
      d_Ex.devPtr(), d_Ey.devPtr(), d_Ez.devPtr(),
      d_Jxh.devPtr(), d_Jyh.devPtr(), d_Jzh.devPtr(),
      d_Bxn.devPtr(), d_Byn.devPtr(), d_Bzn.devPtr(),
      d_Bx_ext.devPtr(), d_By_ext.devPtr(), d_Bz_ext.devPtr(),
      d_rhons.devPtr(), d_qom, ns,
      dt, c, th, FourPI, delt,
      nxn, nyn, nzn, dir, solverStream_);
}

void EMfields3D::gpuLapN2N(GPUFieldArray3& lapN, GPUFieldArray3& fieldN)
{
  const Grid* grid = &get_grid();
  double _invdx = grid->get_invdx();
  double _invdy = grid->get_invdy();
  double _invdz = grid->get_invdz();

  // gradN2C: node→center gradient
  gpuGradN2C(d_tempXC.devPtr(), d_tempYC.devPtr(), d_tempZC.devPtr(),
             fieldN.devPtr(), nxc, nyc, nzc, _invdx, _invdy, _invdz, solverStream_);

#ifdef HALO_OVERLAP
  double* ptrs3[3] = { d_tempXC.devPtr(), d_tempYC.devPtr(), d_tempZC.devPtr() };
  gpuBatchedHaloBeginExchange(ptrs3, 3, nxc, nyc, nzc,
                              true, false, false, false, solverStream_);

  gpuDivC2N_interior(lapN.devPtr(),
            d_tempXC.devPtr(), d_tempYC.devPtr(), d_tempZC.devPtr(),
            nxn, nyn, nzn, _invdx, _invdy, _invdz, solverStream_);

  gpuBatchedHaloEndExchange(ptrs3, 3, nxc, nyc, nzc,
                            true, false, false, false, solverStream_);

  gpuBCface(nxc, nyc, nzc, d_tempXC, 1,1,1,1,1,1, &_vct, solverStream_);
  gpuBCface(nxc, nyc, nzc, d_tempYC, 1,1,1,1,1,1, &_vct, solverStream_);
  gpuBCface(nxc, nyc, nzc, d_tempZC, 1,1,1,1,1,1, &_vct, solverStream_);

  gpuDivC2N_boundary(lapN.devPtr(),
            d_tempXC.devPtr(), d_tempYC.devPtr(), d_tempZC.devPtr(),
            nxn, nyn, nzn, _invdx, _invdy, _invdz, solverStream_);
#else
  // Communicate gradient ghost cells (batched: 3 fields in 1 MPI round)
  gpuCommunicateCenterBC_3(nxc, nyc, nzc, d_tempXC, d_tempYC, d_tempZC, 1, 1, 1, 1, 1, 1);

  // divC2N: divergence of gradient → Laplacian on nodes
  gpuDivC2N(lapN.devPtr(),
            d_tempXC.devPtr(), d_tempYC.devPtr(), d_tempZC.devPtr(),
            nxn, nyn, nzn, _invdx, _invdy, _invdz, solverStream_);
#endif
}

// =========================================================================
//  Fused triple Laplacian: 3 independent lap(fieldN) with ONE halo exchange
//  Uses 9 center-sized scratch arrays:
//    fieldA → d_tempXC / d_tempYC / d_tempZC
//    fieldB → d_divC   / d_poissonTemp / d_poissonIm
//    fieldC → d_divBwork / d_divE_work / d_tempC
// =========================================================================

void EMfields3D::gpuLapN2N_3(
    GPUFieldArray3& lapA, GPUFieldArray3& fieldA,
    GPUFieldArray3& lapB, GPUFieldArray3& fieldB,
    GPUFieldArray3& lapC, GPUFieldArray3& fieldC)
{
  const Grid* grid = &get_grid();
  double _invdx = grid->get_invdx();
  double _invdy = grid->get_invdy();
  double _invdz = grid->get_invdz();

  // Gradient A → d_tempXC, d_tempYC, d_tempZC
  gpuGradN2C(d_tempXC.devPtr(), d_tempYC.devPtr(), d_tempZC.devPtr(),
             fieldA.devPtr(), nxc, nyc, nzc, _invdx, _invdy, _invdz, solverStream_);

  // Gradient B → d_divC, d_poissonTemp, d_poissonIm  (center-sized scratch)
  gpuGradN2C(d_divC.devPtr(), d_poissonTemp.devPtr(), d_poissonIm.devPtr(),
             fieldB.devPtr(), nxc, nyc, nzc, _invdx, _invdy, _invdz, solverStream_);

  // Gradient C → d_divBwork, d_divE_work, d_tempC  (center-sized scratch)
  gpuGradN2C(d_divBwork.devPtr(), d_divE_work.devPtr(), d_tempC.devPtr(),
             fieldC.devPtr(), nxc, nyc, nzc, _invdx, _invdy, _invdz, solverStream_);

#ifdef HALO_OVERLAP
  // ---- Begin halo exchange: pack faces + post MPI ----
  double* ptrs9[9] = { d_tempXC.devPtr(), d_tempYC.devPtr(), d_tempZC.devPtr(),
                       d_divC.devPtr(), d_poissonTemp.devPtr(), d_poissonIm.devPtr(),
                       d_divBwork.devPtr(), d_divE_work.devPtr(), d_tempC.devPtr() };
  gpuBatchedHaloBeginExchange(ptrs9, 9, nxc, nyc, nzc,
                              true, false, false, false, solverStream_);

  // ---- Interior divC2N while MPI is in flight ----
  gpuDivC2N_interior(lapA.devPtr(),
            d_tempXC.devPtr(), d_tempYC.devPtr(), d_tempZC.devPtr(),
            nxn, nyn, nzn, _invdx, _invdy, _invdz, solverStream_);
  gpuDivC2N_interior(lapB.devPtr(),
            d_divC.devPtr(), d_poissonTemp.devPtr(), d_poissonIm.devPtr(),
            nxn, nyn, nzn, _invdx, _invdy, _invdz, solverStream_);
  gpuDivC2N_interior(lapC.devPtr(),
            d_divBwork.devPtr(), d_divE_work.devPtr(), d_tempC.devPtr(),
            nxn, nyn, nzn, _invdx, _invdy, _invdz, solverStream_);

  // ---- End halo exchange: MPI_Waitall + unpack + edges/corners ----
  gpuBatchedHaloEndExchange(ptrs9, 9, nxc, nyc, nzc,
                            true, false, false, false, solverStream_);

  // ---- BC face application (type 1 on all faces) ----
  gpuBCface(nxc, nyc, nzc, d_tempXC,      1,1,1,1,1,1, &_vct, solverStream_);
  gpuBCface(nxc, nyc, nzc, d_tempYC,      1,1,1,1,1,1, &_vct, solverStream_);
  gpuBCface(nxc, nyc, nzc, d_tempZC,      1,1,1,1,1,1, &_vct, solverStream_);
  gpuBCface(nxc, nyc, nzc, d_divC,        1,1,1,1,1,1, &_vct, solverStream_);
  gpuBCface(nxc, nyc, nzc, d_poissonTemp, 1,1,1,1,1,1, &_vct, solverStream_);
  gpuBCface(nxc, nyc, nzc, d_poissonIm,   1,1,1,1,1,1, &_vct, solverStream_);
  gpuBCface(nxc, nyc, nzc, d_divBwork,    1,1,1,1,1,1, &_vct, solverStream_);
  gpuBCface(nxc, nyc, nzc, d_divE_work,   1,1,1,1,1,1, &_vct, solverStream_);
  gpuBCface(nxc, nyc, nzc, d_tempC,       1,1,1,1,1,1, &_vct, solverStream_);

  // ---- Boundary divC2N (ghost + BC data now available) ----
  gpuDivC2N_boundary(lapA.devPtr(),
            d_tempXC.devPtr(), d_tempYC.devPtr(), d_tempZC.devPtr(),
            nxn, nyn, nzn, _invdx, _invdy, _invdz, solverStream_);
  gpuDivC2N_boundary(lapB.devPtr(),
            d_divC.devPtr(), d_poissonTemp.devPtr(), d_poissonIm.devPtr(),
            nxn, nyn, nzn, _invdx, _invdy, _invdz, solverStream_);
  gpuDivC2N_boundary(lapC.devPtr(),
            d_divBwork.devPtr(), d_divE_work.devPtr(), d_tempC.devPtr(),
            nxn, nyn, nzn, _invdx, _invdy, _invdz, solverStream_);

#else
  // ---- Original blocking path ----
  gpuCommunicateCenterBC_9(nxc, nyc, nzc,
      d_tempXC, d_tempYC, d_tempZC,
      d_divC, d_poissonTemp, d_poissonIm,
      d_divBwork, d_divE_work, d_tempC,
      1, 1, 1, 1, 1, 1);

  // Divergence A
  gpuDivC2N(lapA.devPtr(),
            d_tempXC.devPtr(), d_tempYC.devPtr(), d_tempZC.devPtr(),
            nxn, nyn, nzn, _invdx, _invdy, _invdz, solverStream_);

  // Divergence B
  gpuDivC2N(lapB.devPtr(),
            d_divC.devPtr(), d_poissonTemp.devPtr(), d_poissonIm.devPtr(),
            nxn, nyn, nzn, _invdx, _invdy, _invdz, solverStream_);

  // Divergence C
  gpuDivC2N(lapC.devPtr(),
            d_divBwork.devPtr(), d_divE_work.devPtr(), d_tempC.devPtr(),
            nxn, nyn, nzn, _invdx, _invdy, _invdz, solverStream_);
#endif
}

// =========================================================================
//  GPU MaxwellImage:  im = A * vector  (Krylov ↔ Krylov)
// =========================================================================

void EMfields3D::gpuMaxwellImage(double* d_im, double* d_vector)
{
  const VirtualTopology3D* vct = &get_vct();
  const Grid* grid = &get_grid();
  double _invdx = grid->get_invdx();
  double _invdy = grid->get_invdy();
  double _invdz = grid->get_invdz();

  size_t nodeSize = (size_t)nxn * nyn * nzn;

  // Zero work arrays (9 memsets batched)
  double* zptrs[9] = { d_imageX.devPtr(), d_imageY.devPtr(), d_imageZ.devPtr(),
                       d_tempX.devPtr(),  d_tempY.devPtr(),  d_tempZ.devPtr(),
                       d_Dx.devPtr(),     d_Dy.devPtr(),     d_Dz.devPtr() };
  gpuSetAll0_N(zptrs, 9, nodeSize, solverStream_);

  // Krylov → physical space
  gpuSolver2Phys3(d_vectX.devPtr(), d_vectY.devPtr(), d_vectZ.devPtr(),
                  d_vector, nxn, nyn, nzn, solverStream_);

  // Laplacian: image = -lap(vect)  (fused: 3 Laps with 1 halo exchange)
  gpuLapN2N_3(d_imageX, d_vectX,
              d_imageY, d_vectY,
              d_imageZ, d_vectZ);
  gpuNeg3(d_imageX.devPtr(), d_imageY.devPtr(), d_imageZ.devPtr(), nodeSize, solverStream_);

  // MUdot: D = μ·vect
  gpuMUdot(d_Dx, d_Dy, d_Dz, d_vectX, d_vectY, d_vectZ);

  // div(D) on centers
  gpuDivN2C(d_divC.devPtr(),
            d_Dx.devPtr(), d_Dy.devPtr(), d_Dz.devPtr(),
            nxc, nyc, nzc, _invdx, _invdy, _invdz, solverStream_);

#ifdef HALO_OVERLAP
  // ---- Begin halo on divC ----
  double* ptr1[1] = { d_divC.devPtr() };
  gpuBatchedHaloBeginExchange(ptr1, 1, nxc, nyc, nzc,
                              true, false, false, false, solverStream_);

  // ---- Interior gradC2N while MPI is in flight ----
  gpuGradC2N_interior(d_tempX.devPtr(), d_tempY.devPtr(), d_tempZ.devPtr(),
                      d_divC.devPtr(), nxn, nyn, nzn, _invdx, _invdy, _invdz, solverStream_);

  // ---- End halo ----
  gpuBatchedHaloEndExchange(ptr1, 1, nxc, nyc, nzc,
                            true, false, false, false, solverStream_);
  gpuBCface(nxc, nyc, nzc, d_divC, 2,2,2,2,2,2, &_vct, solverStream_);

  // ---- Boundary gradC2N ----
  gpuGradC2N_boundary(d_tempX.devPtr(), d_tempY.devPtr(), d_tempZ.devPtr(),
                      d_divC.devPtr(), nxn, nyn, nzn, _invdx, _invdy, _invdz, solverStream_);
#else
  // Communicate divC
  gpuCommunicateCenterBC(nxc, nyc, nzc, d_divC, 2, 2, 2, 2, 2, 2);

  // grad(divC) on nodes
  gpuGradC2N(d_tempX.devPtr(), d_tempY.devPtr(), d_tempZ.devPtr(),
             d_divC.devPtr(), nxn, nyn, nzn, _invdx, _invdy, _invdz, solverStream_);
#endif

  // image -= temp  (fused triple)
  gpuSub3(d_imageX.devPtr(), d_tempX.devPtr(),
          d_imageY.devPtr(), d_tempY.devPtr(),
          d_imageZ.devPtr(), d_tempZ.devPtr(), nodeSize, solverStream_);

  // Scale by delt²  (fused triple)
  gpuScale3(d_imageX.devPtr(), d_imageY.devPtr(), d_imageZ.devPtr(),
            delt * delt, nodeSize, solverStream_);

  // Add ε·E: image += D + vect  (fused: 6 gpuSum → 1 gpuSumAddTwo3)
  gpuSumAddTwo3(d_imageX.devPtr(), d_Dx.devPtr(), d_vectX.devPtr(),
                d_imageY.devPtr(), d_Dy.devPtr(), d_vectY.devPtr(),
                d_imageZ.devPtr(), d_Dz.devPtr(), d_vectZ.devPtr(),
                nodeSize, solverStream_);

  // Perfect conductor BCs
  if (vct->getXleft_neighbor() == MPI_PROC_NULL && bcEMfaceXleft == 0)
    gpuPerfectConductorLeft(d_imageX, d_imageY, d_imageZ, d_vectX, d_vectY, d_vectZ, 0);
  if (vct->getXright_neighbor() == MPI_PROC_NULL && bcEMfaceXright == 0)
    gpuPerfectConductorRight(d_imageX, d_imageY, d_imageZ, d_vectX, d_vectY, d_vectZ, 0);
  if (vct->getYleft_neighbor() == MPI_PROC_NULL && bcEMfaceYleft == 0)
    gpuPerfectConductorLeft(d_imageX, d_imageY, d_imageZ, d_vectX, d_vectY, d_vectZ, 1);
  if (vct->getYright_neighbor() == MPI_PROC_NULL && bcEMfaceYright == 0)
    gpuPerfectConductorRight(d_imageX, d_imageY, d_imageZ, d_vectX, d_vectY, d_vectZ, 1);
  if (vct->getZleft_neighbor() == MPI_PROC_NULL && bcEMfaceZleft == 0)
    gpuPerfectConductorLeft(d_imageX, d_imageY, d_imageZ, d_vectX, d_vectY, d_vectZ, 2);
  if (vct->getZright_neighbor() == MPI_PROC_NULL && bcEMfaceZright == 0)
    gpuPerfectConductorRight(d_imageX, d_imageY, d_imageZ, d_vectX, d_vectY, d_vectZ, 2);

  // OpenBC: apply inflow BCs to GMRES image if enabled
  if (get_col().getApplyInflowBcsEImage())
    gpuOpenBoundaryInflowEImage(d_imageX.devPtr(), d_imageY.devPtr(), d_imageZ.devPtr(),
                                d_vectX.devPtr(), d_vectY.devPtr(), d_vectZ.devPtr(),
                                nxn, nyn, nzn);

  // Physical → Krylov space
  gpuPhys2Solver3(d_im,
                  d_imageX.devPtr(), d_imageY.devPtr(), d_imageZ.devPtr(),
                  nxn, nyn, nzn, solverStream_);
}

// =========================================================================
//  GPU MaxwellSource:  build RHS of Maxwell system
// =========================================================================

void EMfields3D::gpuMaxwellSource(double* d_bkrylov)
{
  const Collective* col = &get_col();
  const VirtualTopology3D* vct = &get_vct();
  const Grid* grid = &get_grid();
  double _invdx = grid->get_invdx();
  double _invdy = grid->get_invdy();
  double _invdz = grid->get_invdz();

  size_t nodeSize = (size_t)nxn * nyn * nzn;
  size_t centSize = (size_t)nxc * nyc * nzc;

  // Zero work arrays (batched) — tempC is centSize, others are nodeSize
  d_tempC.setAll(0.0, solverStream_);
  double* zptrs2[9] = { d_tempX.devPtr(), d_tempY.devPtr(), d_tempZ.devPtr(),
                         d_tempXN.devPtr(), d_tempYN.devPtr(), d_tempZN.devPtr(),
                         d_temp2X.devPtr(), d_temp2Y.devPtr(), d_temp2Z.devPtr() };
  gpuSetAll0_N(zptrs2, 9, nodeSize, solverStream_);

  // Communicate Bc ghost cells (batched 3-field, mixed BCs)
  gpuCommunicateCenterBC_3mixed(nxc, nyc, nzc,
      d_Bxc, col->bcBx,
      d_Byc, col->bcBy,
      d_Bzc, col->bcBz);

  // Case-specific B fixes (before curl, matching CPU MaxwellSource order)
  {
    const string& simCase = col->getCase();
    if (simCase == "ForceFree")
      gpuFixBforcefree();
    if (simCase == "GEM" || simCase == "GEMnoPert" || simCase == "GEMDoubleHarris")
      gpuFixBnGEM();
  }

  // OpenBC: apply inflow BCs on center B
  gpuOpenBoundaryInflowB(d_Bxc.devPtr(), d_Byc.devPtr(), d_Bzc.devPtr(), nxc, nyc, nzc);

  // Case-specific center B fix after OpenBC
  {
    const string& simCase = col->getCase();
    if (simCase == "GEM" || simCase == "GEMnoPert" || simCase == "GEMDoubleHarris")
      gpuFixBcGEM();
  }

  // curl(Bc) → tempXN/YN/ZN
  gpuCurlC2N(d_tempXN.devPtr(), d_tempYN.devPtr(), d_tempZN.devPtr(),
             d_Bxc.devPtr(), d_Byc.devPtr(), d_Bzc.devPtr(),
             nxn, nyn, nzn, _invdx, _invdy, _invdz, solverStream_);

  // temp2 = -4π/c * Jhat  (fused triple)
  gpuScaleCopy3(d_temp2X.devPtr(), d_Jxh.devPtr(),
                d_temp2Y.devPtr(), d_Jyh.devPtr(),
                d_temp2Z.devPtr(), d_Jzh.devPtr(),
                -FourPI / c, nodeSize, solverStream_);

  // temp2 += curl(B)  (fused triple)
  gpuSum3(d_temp2X.devPtr(), d_tempXN.devPtr(),
          d_temp2Y.devPtr(), d_tempYN.devPtr(),
          d_temp2Z.devPtr(), d_tempZN.devPtr(), nodeSize, solverStream_);

  // temp2 *= delt  (fused triple)
  gpuScale3(d_temp2X.devPtr(), d_temp2Y.devPtr(), d_temp2Z.devPtr(),
            delt, nodeSize, solverStream_);

  // Communicate rhoh
  gpuCommunicateCenterBC_P(nxc, nyc, nzc, d_rhoh, 2, 2, 2, 2, 2, 2);

  // grad(rhoh) → tempX/Y/Z
  gpuGradC2N(d_tempX.devPtr(), d_tempY.devPtr(), d_tempZ.devPtr(),
             d_rhoh.devPtr(), nxn, nyn, nzn, _invdx, _invdy, _invdz, solverStream_);

  // temp *= -delt² * 4π  (fused triple)
  gpuScale3(d_tempX.devPtr(), d_tempY.devPtr(), d_tempZ.devPtr(),
            -delt * delt * FourPI, nodeSize, solverStream_);

  // Add E and curl+current parts  (fused: 6 gpuSum → 1 gpuSumAddTwo3)
  gpuSumAddTwo3(d_tempX.devPtr(), d_Ex.devPtr(), d_temp2X.devPtr(),
                d_tempY.devPtr(), d_Ey.devPtr(), d_temp2Y.devPtr(),
                d_tempZ.devPtr(), d_Ez.devPtr(), d_temp2Z.devPtr(),
                nodeSize, solverStream_);

  // Perfect conductor BCs for source
  if (vct->getXleft_neighbor() == MPI_PROC_NULL && bcEMfaceXleft == 0)
    gpuPerfectConductorLeftS(d_tempX.devPtr(), d_tempY.devPtr(), d_tempZ.devPtr(), nxn, nyn, nzn, 0, solverStream_);
  if (vct->getXright_neighbor() == MPI_PROC_NULL && bcEMfaceXright == 0)
    gpuPerfectConductorRightS(d_tempX.devPtr(), d_tempY.devPtr(), d_tempZ.devPtr(), nxn, nyn, nzn, 0, solverStream_);
  if (vct->getYleft_neighbor() == MPI_PROC_NULL && bcEMfaceYleft == 0)
    gpuPerfectConductorLeftS(d_tempX.devPtr(), d_tempY.devPtr(), d_tempZ.devPtr(), nxn, nyn, nzn, 1, solverStream_);
  if (vct->getYright_neighbor() == MPI_PROC_NULL && bcEMfaceYright == 0)
    gpuPerfectConductorRightS(d_tempX.devPtr(), d_tempY.devPtr(), d_tempZ.devPtr(), nxn, nyn, nzn, 1, solverStream_);
  if (vct->getZleft_neighbor() == MPI_PROC_NULL && bcEMfaceZleft == 0)
    gpuPerfectConductorLeftS(d_tempX.devPtr(), d_tempY.devPtr(), d_tempZ.devPtr(), nxn, nyn, nzn, 2, solverStream_);
  if (vct->getZright_neighbor() == MPI_PROC_NULL && bcEMfaceZright == 0)
    gpuPerfectConductorRightS(d_tempX.devPtr(), d_tempY.devPtr(), d_tempZ.devPtr(), nxn, nyn, nzn, 2, solverStream_);

  // OpenBC: zero source on inflow boundary nodes
  gpuOpenBoundaryInflowESource(d_tempX.devPtr(), d_tempY.devPtr(), d_tempZ.devPtr(), nxn, nyn, nzn);

  // Physical → Krylov space
  gpuPhys2Solver3(d_bkrylov,
                  d_tempX.devPtr(), d_tempY.devPtr(), d_tempZ.devPtr(),
                  nxn, nyn, nzn, solverStream_);
}

// =========================================================================
//  GPU GMRES:  restarted GMRES(m) on device Krylov vectors
//
//  Performance-critical design choices:
//    • All host-side small arrays (H, g, cs, sn, y, reduction buffers)
//      are PINNED and PERSISTENT — allocated once in gpuSolverInit().
//    • D→H reduction copies use async memcpy into pinned memory.
//    • Stream syncs are batched: ONE sync per Arnoldi step (before
//      MPI_Allreduce) plus ONE for the post-ortho norm.
//    • Device workspace (V, w) is allocated lazily and cached.
// =========================================================================

static void gpuGMRES_impl(EMfields3D* field,
                          void (EMfields3D::*GpuImage)(double*, double*),
                          double* d_x, int n, double* d_b,
                          int m, int max_iter, double tol,
                          double* d_scratch,
                          double*& d_gmresV, double*& d_gmresW, int& gmresVAlloc,
                          MPI_Comm fieldcomm,
                          cudaStream_t stream,
                          // Persistent PINNED host buffers (from EMfields3D members)
                          double* h_reduceLocal,
                          double* h_reduceGlobal,
                          double* H,    // [mp1 * m]
                          double* g,    // [mp1]
                          double* cs,   // [m]
                          double* sn,   // [m]
                          double* y)    // [mp1]
{
  const int mp1 = m + 1;

  // Allocate GMRES workspace on device (lazy, cached across calls)
  if (gmresVAlloc < mp1 * n) {
    if (d_gmresV) cudaFree(d_gmresV);
    if (d_gmresW) cudaFree(d_gmresW);
    cudaErrChk(cudaMalloc(&d_gmresV, (size_t)mp1 * n * sizeof(double)));
    cudaErrChk(cudaMalloc(&d_gmresW, (size_t)n * sizeof(double)));
    gmresVAlloc = mp1 * n;
  }
  double* d_V = d_gmresV;
  double* d_w = d_gmresW;

  // r = b - A*x  →  V[0]
  (field->*GpuImage)(d_w, d_x);          // w = A*x
  gpuEq(d_V, d_b, n, stream);            // V[0] = b
  gpuSub(d_V, d_w, n, stream);           // V[0] = b - A*x

  // ||r||₂  (async D→H into pinned buffer, one sync)
  gpuNorm2_async(d_V, n, d_scratch, &h_reduceLocal[0], stream);
  cudaErrChk(cudaStreamSynchronize(stream));
  double initial_error;
  MPI_Allreduce(&h_reduceLocal[0], &initial_error, 1, MPI_DOUBLE, MPI_SUM, fieldcomm);
  initial_error = sqrt(initial_error);
  if (initial_error < 1e-30) return;

  int gmresRank;
  MPI_Comm_rank(fieldcomm, &gmresRank);

  // Compute ||b|| for status print (async, combined sync)
  gpuNorm2_async(d_b, n, d_scratch, &h_reduceLocal[0], stream);
  cudaErrChk(cudaStreamSynchronize(stream));
  double normb;
  MPI_Allreduce(&h_reduceLocal[0], &normb, 1, MPI_DOUBLE, MPI_SUM, fieldcomm);
  normb = sqrt(normb);
  if (normb == 0.0) normb = 1.0;
  if (gmresRank == 0)
    printf("Initial residual: %g norm b vector (source) = %g\n", initial_error, normb);

  gpuScale(d_V, 1.0 / initial_error, n, stream);
  double error = initial_error;

  for (int restart = 0; restart < max_iter; restart++) {
    // Zero persistent host arrays
    memset(H,  0, (size_t)mp1 * m * sizeof(double));
    memset(g,  0, mp1 * sizeof(double));
    memset(cs, 0, m * sizeof(double));
    memset(sn, 0, m * sizeof(double));
    g[0] = error;

    for (int k = 0; k < m; k++) {
      // w = A * V[k]
      (field->*GpuImage)(d_w, d_V + (size_t)k * n);

      // ---- Batched Arnoldi: fuse all k+1 dot products + norm² ----
      // ONE kernel launch → ONE async D→H copy → ONE sync → ONE MPI_Allreduce
      gpuBatchedDotNorm(d_w, d_V, (size_t)n, k, (size_t)n, d_scratch, stream);
      cudaErrChk(cudaMemcpyAsync(h_reduceLocal, d_scratch,
                                  (k + 2) * sizeof(double),
                                  cudaMemcpyDeviceToHost, stream));
      cudaErrChk(cudaStreamSynchronize(stream));
      MPI_Allreduce(h_reduceLocal, h_reduceGlobal,
                    k + 2, MPI_DOUBLE, MPI_SUM, fieldcomm);

      // Store H[j][k] and apply orthogonalisation updates
      for (int j = 0; j <= k; j++) {
        double h_jk = h_reduceGlobal[j];
        H[j * m + k] = h_jk;
        gpuAddscale(-h_jk, d_w, d_V + (size_t)j * n, n, stream);
      }

      // H[k+1][k] = ||w_perp||  (post-orthogonalisation norm)
      // Async D→H, single sync
      gpuNorm2_async(d_w, n, d_scratch, &h_reduceLocal[0], stream);
      cudaErrChk(cudaStreamSynchronize(stream));
      double global_wNorm;
      MPI_Allreduce(&h_reduceLocal[0], &global_wNorm, 1, MPI_DOUBLE, MPI_SUM, fieldcomm);
      H[(k + 1) * m + k] = sqrt(global_wNorm);

      // Re-orthogonalise if needed (matches CPU GMRES)
      double av = sqrt(h_reduceGlobal[k + 1]); // pre-ortho ||w||
      const double delta = 0.001;
      if (av + delta * H[(k + 1) * m + k] == av) {
        for (int j = 0; j <= k; j++) {
          gpuDot_async(d_w, d_V + (size_t)j * n, n, d_scratch, &h_reduceLocal[0], stream);
          cudaErrChk(cudaStreamSynchronize(stream));
          double htmp;
          MPI_Allreduce(&h_reduceLocal[0], &htmp, 1, MPI_DOUBLE, MPI_SUM, fieldcomm);
          H[j * m + k] += htmp;
          gpuAddscale(-htmp, d_w, d_V + (size_t)j * n, n, stream);
        }
        gpuNorm2_async(d_w, n, d_scratch, &h_reduceLocal[0], stream);
        cudaErrChk(cudaStreamSynchronize(stream));
        MPI_Allreduce(&h_reduceLocal[0], &global_wNorm, 1, MPI_DOUBLE, MPI_SUM, fieldcomm);
        H[(k + 1) * m + k] = sqrt(global_wNorm);
      }

      // V[k+1] = w / H[k+1][k]
      if (H[(k + 1) * m + k] > 1e-30)
        gpuScaleCopy(d_V + (size_t)(k + 1) * n, d_w, 1.0 / H[(k + 1) * m + k], n, stream);
      else
        gpuEq(d_V + (size_t)(k + 1) * n, d_w, n, stream);

      // Apply previous Givens rotations
      for (int j = 0; j < k; j++) {
        double h0 = H[j * m + k];
        double h1 = H[(j + 1) * m + k];
        H[j * m + k]       =  cs[j] * h0 + sn[j] * h1;
        H[(j + 1) * m + k] = -sn[j] * h0 + cs[j] * h1;
      }

      // Compute new Givens rotation
      double h_kk  = H[k * m + k];
      double h_k1k = H[(k + 1) * m + k];
      double r_val = sqrt(h_kk * h_kk + h_k1k * h_k1k);
      cs[k] = h_kk  / r_val;
      sn[k] = h_k1k / r_val;
      H[k * m + k]       = r_val;
      H[(k + 1) * m + k] = 0.0;

      double g_k = g[k];
      g[k]     =  cs[k] * g_k;
      g[k + 1] = -sn[k] * g_k;

      error = fabs(g[k + 1]);

      if (error / initial_error < tol) {
        // Early exit: solve and update
        for (int i = k; i >= 0; i--) {
          y[i] = g[i];
          for (int j = i + 1; j <= k; j++)
            y[i] -= H[i * m + j] * y[j];
          y[i] /= H[i * m + i];
        }
        for (int j = 0; j <= k; j++)
          gpuAddscale(y[j], d_x, d_V + (size_t)j * n, n, stream);
        if (gmresRank == 0)
          printf("GMRES converged at restart # %d; iteration #%d with error: %g\n",
                 restart, k, error / initial_error);
        return;
      }
    } // end inner loop

    // Full m iterations: solve and update
    for (int i = m - 1; i >= 0; i--) {
      y[i] = g[i];
      for (int j = i + 1; j < m; j++)
        y[i] -= H[i * m + j] * y[j];
      y[i] /= H[i * m + i];
    }
    for (int j = 0; j < m; j++)
      gpuAddscale(y[j], d_x, d_V + (size_t)j * n, n, stream);

    // Restart: new residual
    (field->*GpuImage)(d_w, d_x);
    gpuEq(d_V, d_b, n, stream);
    gpuSub(d_V, d_w, n, stream);

    gpuNorm2_async(d_V, n, d_scratch, &h_reduceLocal[0], stream);
    cudaErrChk(cudaStreamSynchronize(stream));
    MPI_Allreduce(&h_reduceLocal[0], &error, 1, MPI_DOUBLE, MPI_SUM, fieldcomm);
    error = sqrt(error);

    if (error / initial_error < tol) {
      if (gmresRank == 0)
        printf("GMRES converged at restart # %d; iteration #%d with error: %g\n",
               restart, m - 1, error / initial_error);
      return;
    }
    gpuScale(d_V, 1.0 / error, n, stream);
  }
  if (gmresRank == 0)
    std::cout << "  [GMRES] WARNING: did not converge after " << max_iter << " restarts" << std::endl;
}

// =========================================================================
//  GPU calculateE: full E-field solver
// =========================================================================

void EMfields3D::gpuCalculateE(int cycle)
{
  const Collective* col = &get_col();
  const VirtualTopology3D* vct = &get_vct();

  if (vct->getCartesian_rank() == 0)
    cout << "*** E CALCULATION [GPU] ***" << endl;

  const int nMaxwell = 3 * (nxn - 2) * (nyn - 2) * (nzn - 2);
  size_t nodeSize = (size_t)nxn * nyn * nzn;

  // Divergence cleaning on E (Poisson correction)
  gpuPoissonCorrection(cycle);

  if (vct->getCartesian_rank() == 0)
    cout << "*** MAXWELL SOLVER [GPU] ***" << endl;

  // Build RHS
  //if (vct->getCartesian_rank() == 0)
  //  cout << "  [gpuCalculateE] Building Maxwell source..." << std::flush;
  gpuMaxwellSource(d_bkrylovMaxwell.devPtr());
  //if (vct->getCartesian_rank() == 0)
  //  cout << " done" << endl;

  // Initial guess: pack current E into x
  gpuPhys2Solver3(d_xkrylovMaxwell.devPtr(),
                  d_Ex.devPtr(), d_Ey.devPtr(), d_Ez.devPtr(),
                  nxn, nyn, nzn, solverStream_);

  // GMRES solve on device
  //if (vct->getCartesian_rank() == 0)
  //  cout << "  [gpuCalculateE] Starting GMRES (n=" << nMaxwell << ", m=20, maxIter=200, tol=" << GMREStol << ")..." << endl;
  MPI_Comm fieldcomm = vct->getFieldComm();
  gpuGMRES_impl(this, &EMfields3D::gpuMaxwellImage,
                d_xkrylovMaxwell.devPtr(), nMaxwell,
                d_bkrylovMaxwell.devPtr(),
                20, 200, GMREStol,
                d_blasScratch,
                d_gmresV, d_gmresW, gmresVAlloc,
                fieldcomm, solverStream_,
                h_gmresReduceLocal, h_gmresReduceGlobal,
                h_gmresH, h_gmresG, h_gmresCS, h_gmresSN, h_gmresY);
  //if (vct->getCartesian_rank() == 0)
  //  cout << "  [gpuCalculateE] GMRES done" << endl;

  // Krylov → physical: Exth, Eyth, Ezth
  gpuSolver2Phys3(d_Exth.devPtr(), d_Eyth.devPtr(), d_Ezth.devPtr(),
                  d_xkrylovMaxwell.devPtr(), nxn, nyn, nzn, solverStream_);

  // E^{n+1} = beta*E + alfa*Eth  where alfa=1/th, beta=-(1-th)/th
  gpuAddscale2_3(1.0 / th, -(1.0 - th) / th,
                 d_Ex.devPtr(), d_Exth.devPtr(),
                 d_Ey.devPtr(), d_Eyth.devPtr(),
                 d_Ez.devPtr(), d_Ezth.devPtr(), nodeSize, solverStream_);

  // Smooth E
  gpuSmoothE();

  // Communicate final E fields (batched: 2 × 3 fields in 1 MPI round each)
  gpuCommunicateNodeBC_3mixed(nxn, nyn, nzn,
      d_Exth, col->bcEx, d_Eyth, col->bcEy, d_Ezth, col->bcEz);
  gpuCommunicateNodeBC_3mixed(nxn, nyn, nzn,
      d_Ex, col->bcEx, d_Ey, col->bcEy, d_Ez, col->bcEz);

  // OpenBC Inflow on solved E fields
  gpuOpenBoundaryInflowE(d_Exth.devPtr(), d_Eyth.devPtr(), d_Ezth.devPtr(), nxn, nyn, nzn);
  gpuOpenBoundaryInflowE(d_Ex.devPtr(), d_Ey.devPtr(), d_Ez.devPtr(), nxn, nyn, nzn);
}

// =========================================================================
//  GPU calculateB: Faraday update
// =========================================================================

void EMfields3D::gpuCalculateB(int cycle)
{
  const Collective* col = &get_col();
  const VirtualTopology3D* vct = &get_vct();
  const Grid* grid = &get_grid();
  double _invdx = grid->get_invdx();
  double _invdy = grid->get_invdy();
  double _invdz = grid->get_invdz();

  if (vct->getCartesian_rank() == 0)
    cout << "*** B CALCULATION [GPU] ***" << endl;

  size_t centSize = (size_t)nxc * nyc * nzc;

  // curl(Eth) → tempXC/YC/ZC
  gpuCurlN2C(d_tempXC.devPtr(), d_tempYC.devPtr(), d_tempZC.devPtr(),
             d_Exth.devPtr(), d_Eyth.devPtr(), d_Ezth.devPtr(),
             nxc, nyc, nzc, _invdx, _invdy, _invdz, solverStream_);

  // B^{n+1} = B^n - c*dt * curl(Eth)
  gpuAddscale3(-c * dt,
               d_Bxc.devPtr(), d_tempXC.devPtr(),
               d_Byc.devPtr(), d_tempYC.devPtr(),
               d_Bzc.devPtr(), d_tempZC.devPtr(), centSize, solverStream_);

  // Communicate center B ghost cells (batched: 3 fields in 1 MPI round)
#ifdef HALO_OVERLAP
  {
    double* bptrs[3] = { d_Bxc.devPtr(), d_Byc.devPtr(), d_Bzc.devPtr() };
    gpuBatchedHaloBeginExchange(bptrs, 3, nxc, nyc, nzc,
                                true, false, false, false, solverStream_);

    // Interior interpC2N while MPI is in flight
    gpuInterpC2N_interior(d_Bxn.devPtr(), d_Bxc.devPtr(), nxn, nyn, nzn, solverStream_);
    gpuInterpC2N_interior(d_Byn.devPtr(), d_Byc.devPtr(), nxn, nyn, nzn, solverStream_);
    gpuInterpC2N_interior(d_Bzn.devPtr(), d_Bzc.devPtr(), nxn, nyn, nzn, solverStream_);

    gpuBatchedHaloEndExchange(bptrs, 3, nxc, nyc, nzc,
                              true, false, false, false, solverStream_);

    // Mixed BC face application
    gpuBCface(nxc, nyc, nzc, d_Bxc, col->bcBx[0], col->bcBx[1], col->bcBx[2], col->bcBx[3], col->bcBx[4], col->bcBx[5], &_vct, solverStream_);
    gpuBCface(nxc, nyc, nzc, d_Byc, col->bcBy[0], col->bcBy[1], col->bcBy[2], col->bcBy[3], col->bcBy[4], col->bcBy[5], &_vct, solverStream_);
    gpuBCface(nxc, nyc, nzc, d_Bzc, col->bcBz[0], col->bcBz[1], col->bcBz[2], col->bcBz[3], col->bcBz[4], col->bcBz[5], &_vct, solverStream_);
  }

  // Open boundary conditions on center-based B
  gpuOpenBoundaryInflowB(d_Bxc.devPtr(), d_Byc.devPtr(), d_Bzc.devPtr(), nxc, nyc, nzc);

  // Case-specific fixes on center-based B
  {
    const string& simCase = col->getCase();
    if (simCase == "GEM" || simCase == "GEMnoPert" || simCase == "GEMDoubleHarris")
      gpuFixBcGEM();
  }

  // Boundary interpC2N (ghost + BC + fixup data now available)
  gpuInterpC2N_boundary(d_Bxn.devPtr(), d_Bxc.devPtr(), nxn, nyn, nzn, solverStream_);
  gpuInterpC2N_boundary(d_Byn.devPtr(), d_Byc.devPtr(), nxn, nyn, nzn, solverStream_);
  gpuInterpC2N_boundary(d_Bzn.devPtr(), d_Bzc.devPtr(), nxn, nyn, nzn, solverStream_);
#else
  gpuCommunicateCenterBC_3mixed(nxc, nyc, nzc,
      d_Bxc, col->bcBx, d_Byc, col->bcBy, d_Bzc, col->bcBz);

  // Open boundary conditions on center-based B
  gpuOpenBoundaryInflowB(d_Bxc.devPtr(), d_Byc.devPtr(), d_Bzc.devPtr(), nxc, nyc, nzc);

  // Case-specific fixes on center-based B
  {
    const string& simCase = col->getCase();
    if (simCase == "GEM" || simCase == "GEMnoPert" || simCase == "GEMDoubleHarris")
      gpuFixBcGEM();
  }

  // Interpolate center → node
  gpuInterpC2N(d_Bxn.devPtr(), d_Bxc.devPtr(), nxn, nyn, nzn, solverStream_);
  gpuInterpC2N(d_Byn.devPtr(), d_Byc.devPtr(), nxn, nyn, nzn, solverStream_);
  gpuInterpC2N(d_Bzn.devPtr(), d_Bzc.devPtr(), nxn, nyn, nzn, solverStream_);
#endif

  // Communicate node B ghost cells (batched: 3 fields in 1 MPI round)
  gpuCommunicateNodeBC_3mixed(nxn, nyn, nzn,
      d_Bxn, col->bcBx, d_Byn, col->bcBy, d_Bzn, col->bcBz);

  // Case-specific fixes on node-based B
  {
    const string& simCase = col->getCase();
    if (simCase == "ForceFree")
      gpuFixBforcefree();
    if (simCase == "GEM" || simCase == "GEMnoPert" || simCase == "GEMDoubleHarris")
      gpuFixBnGEM();
  }

  // Divergence cleaning: lap(PSI) = div(B), B = B - grad(PSI)
  if (divBCorrection && cycle % divBCorrectionCycle == 0)
    gpuApplyDivBCleaning();
}

// =========================================================================
//  GPU calculateHatFunctions: compute Jhat and rhohat
// =========================================================================

void EMfields3D::gpuCalculateHatFunctions()
{
  const Grid* grid = &get_grid();
  double _invdx = grid->get_invdx();
  double _invdy = grid->get_invdy();
  double _invdz = grid->get_invdz();

  size_t nodeSize = (size_t)nxn * nyn * nzn;
  size_t centSize = (size_t)nxc * nyc * nzc;

  // Smooth rhoc
  gpuSmooth(d_rhoc, 0);

  // Initialise Jxh/Jyh/Jzh = 0
  double* jptrs[3] = { d_Jxh.devPtr(), d_Jyh.devPtr(), d_Jzh.devPtr() };
  gpuSetAll0_N(jptrs, 3, nodeSize, solverStream_);

  for (int is = 0; is < ns; is++) {
    // divSymmTensorN2C for this species
    gpuDivSymmTensorN2C(d_tempXC.devPtr(), d_tempYC.devPtr(), d_tempZC.devPtr(),
                        d_pXXsn.speciesPtr(is), d_pXYsn.speciesPtr(is), d_pXZsn.speciesPtr(is),
                        d_pYYsn.speciesPtr(is), d_pYZsn.speciesPtr(is), d_pZZsn.speciesPtr(is),
                        nxc, nyc, nzc, _invdx, _invdy, _invdz, solverStream_);

    // Scale by -dt/2 (fused triple)
    gpuScale3(d_tempXC.devPtr(), d_tempYC.devPtr(), d_tempZC.devPtr(),
              -dt / 2.0, centSize, solverStream_);

#ifdef HALO_OVERLAP
    // --- overlap: begin face exchange for 3 centre fields ---
    double* hatPtrs[3] = { d_tempXC.devPtr(), d_tempYC.devPtr(), d_tempZC.devPtr() };
    int hatReqs = gpuBatchedHaloBeginExchange(hatPtrs, 3, nxc, nyc, nzc,
                    /*isCenter=*/true, /*faceOnly=*/false,
                    /*needInterp=*/false, /*isParticle=*/true, solverStream_);
    // --- interior interpC2N while faces in flight ---
    gpuInterpC2N_interior(d_tempXN.devPtr(), d_tempXC.devPtr(), nxn, nyn, nzn, solverStream_);
    gpuInterpC2N_interior(d_tempYN.devPtr(), d_tempYC.devPtr(), nxn, nyn, nzn, solverStream_);
    gpuInterpC2N_interior(d_tempZN.devPtr(), d_tempZC.devPtr(), nxn, nyn, nzn, solverStream_);
    // --- end exchange: wait + unpack + edges + corners ---
    gpuBatchedHaloEndExchange(hatPtrs, 3, nxc, nyc, nzc,
                    /*isCenter=*/true, /*faceOnly=*/false,
                    /*needInterp=*/false, /*isParticle=*/true, solverStream_);
    // BC
    gpuBCface_P(nxc, nyc, nzc, d_tempXC, 2, 2, 2, 2, 2, 2, &get_vct(), solverStream_);
    gpuBCface_P(nxc, nyc, nzc, d_tempYC, 2, 2, 2, 2, 2, 2, &get_vct(), solverStream_);
    gpuBCface_P(nxc, nyc, nzc, d_tempZC, 2, 2, 2, 2, 2, 2, &get_vct(), solverStream_);
    // --- boundary interpC2N ---
    gpuInterpC2N_boundary(d_tempXN.devPtr(), d_tempXC.devPtr(), nxn, nyn, nzn, solverStream_);
    gpuInterpC2N_boundary(d_tempYN.devPtr(), d_tempYC.devPtr(), nxn, nyn, nzn, solverStream_);
    gpuInterpC2N_boundary(d_tempZN.devPtr(), d_tempZC.devPtr(), nxn, nyn, nzn, solverStream_);
#else
    // Communicate (batched: 3 fields in 1 MPI round)
    gpuCommunicateCenterBC_P_3(nxc, nyc, nzc, d_tempXC, d_tempYC, d_tempZC, 2, 2, 2, 2, 2, 2);

    // Interpolate C → N
    gpuInterpC2N(d_tempXN.devPtr(), d_tempXC.devPtr(), nxn, nyn, nzn, solverStream_);
    gpuInterpC2N(d_tempYN.devPtr(), d_tempYC.devPtr(), nxn, nyn, nzn, solverStream_);
    gpuInterpC2N(d_tempZN.devPtr(), d_tempZC.devPtr(), nxn, nyn, nzn, solverStream_);
#endif

    // Add species current: tempN += Jxs[is] (fused triple)
    gpuSum3(d_tempXN.devPtr(), d_Jxs.speciesPtr(is),
            d_tempYN.devPtr(), d_Jys.speciesPtr(is),
            d_tempZN.devPtr(), d_Jzs.speciesPtr(is), nodeSize, solverStream_);

    // PIdot: Jhat += π(tempN)
    gpuPIdot(d_Jxh, d_Jyh, d_Jzh, d_tempXN, d_tempYN, d_tempZN, is);
  }

  // Smooth Jhat (batched: 3 fields in 1 MPI round per iteration)
  gpuSmooth3(d_Jxh, d_Jyh, d_Jzh, 1);

  // rhohat = rhoc - dt*theta*div(Jhat)
  gpuDivN2C(d_tempXC.devPtr(),
            d_Jxh.devPtr(), d_Jyh.devPtr(), d_Jzh.devPtr(),
            nxc, nyc, nzc, _invdx, _invdy, _invdz, solverStream_);
  gpuScale(d_tempXC.devPtr(), -dt * th, centSize, solverStream_);
  gpuSum(d_tempXC.devPtr(), d_rhoc.devPtr(), centSize, solverStream_);
  gpuEq(d_rhoh.devPtr(), d_tempXC.devPtr(), centSize, solverStream_);

  // Communicate rhoh
  gpuCommunicateCenterBC_P(nxc, nyc, nzc, d_rhoh, 2, 2, 2, 2, 2, 2);
}

// =========================================================================
//  GPU moment-processing: D2D scatter from packed moment buffer
// =========================================================================

void EMfields3D::gpuScatterMomentsD2D(double* momentsSrc, int species, cudaStream_t stream)
{
  const size_t gridSize = (size_t)nxn * nyn * nzn;
  // momentsSrc layout: [rhons | Jxs | Jys | Jzs | pXXsn | pXYsn | pXZsn | pYYsn | pYZsn | pZZsn]
  // Each block is gridSize doubles, contiguous.
  cudaErrChk(cudaMemcpyAsync(d_rhons.speciesPtr(species), momentsSrc + 0*gridSize, gridSize*sizeof(double), cudaMemcpyDeviceToDevice, stream));
  cudaErrChk(cudaMemcpyAsync(d_Jxs.speciesPtr(species),   momentsSrc + 1*gridSize, gridSize*sizeof(double), cudaMemcpyDeviceToDevice, stream));
  cudaErrChk(cudaMemcpyAsync(d_Jys.speciesPtr(species),   momentsSrc + 2*gridSize, gridSize*sizeof(double), cudaMemcpyDeviceToDevice, stream));
  cudaErrChk(cudaMemcpyAsync(d_Jzs.speciesPtr(species),   momentsSrc + 3*gridSize, gridSize*sizeof(double), cudaMemcpyDeviceToDevice, stream));
  cudaErrChk(cudaMemcpyAsync(d_pXXsn.speciesPtr(species), momentsSrc + 4*gridSize, gridSize*sizeof(double), cudaMemcpyDeviceToDevice, stream));
  cudaErrChk(cudaMemcpyAsync(d_pXYsn.speciesPtr(species), momentsSrc + 5*gridSize, gridSize*sizeof(double), cudaMemcpyDeviceToDevice, stream));
  cudaErrChk(cudaMemcpyAsync(d_pXZsn.speciesPtr(species), momentsSrc + 6*gridSize, gridSize*sizeof(double), cudaMemcpyDeviceToDevice, stream));
  cudaErrChk(cudaMemcpyAsync(d_pYYsn.speciesPtr(species), momentsSrc + 7*gridSize, gridSize*sizeof(double), cudaMemcpyDeviceToDevice, stream));
  cudaErrChk(cudaMemcpyAsync(d_pYZsn.speciesPtr(species), momentsSrc + 8*gridSize, gridSize*sizeof(double), cudaMemcpyDeviceToDevice, stream));
  cudaErrChk(cudaMemcpyAsync(d_pZZsn.speciesPtr(species), momentsSrc + 9*gridSize, gridSize*sizeof(double), cudaMemcpyDeviceToDevice, stream));
}

// =========================================================================
//  GPU communicateGhostP2G: full ghost exchange for one species
// =========================================================================

void EMfields3D::gpuCommunicateGhostP2G(int species)
{
  const VirtualTopology3D* vct = &get_vct();

  // Gather the 10 per-species device pointers into a host array
  double* ptrs[10] = {
    d_rhons.speciesPtr(species),
    d_Jxs.speciesPtr(species),
    d_Jys.speciesPtr(species),
    d_Jzs.speciesPtr(species),
    d_pXXsn.speciesPtr(species),
    d_pXYsn.speciesPtr(species),
    d_pXZsn.speciesPtr(species),
    d_pYYsn.speciesPtr(species),
    d_pYZsn.speciesPtr(species),
    d_pZZsn.speciesPtr(species)
  };

  // Phase 1: Batched additive (interpolating) halo exchange — ghost → shared nodes
  //   isCenterFlag=true (offset=0), needInterp=true, isParticle=true
  gpuBatchedHaloExchange(ptrs, 10, nxn, nyn, nzn,
                         /*isCenterFlag=*/true, /*isFaceOnly=*/false,
                         /*needInterp=*/true, /*isParticle=*/true, solverStream_);

  // Phase 2: adjust non-periodic boundary densities (×2 on domain boundaries)
  gpuAdjustNonPeriodicDensities(
      10, d_ptrArray_, nxn, nyn, nzn,
      vct->getXleft_neighbor_P()  == MPI_PROC_NULL,
      vct->getXright_neighbor_P() == MPI_PROC_NULL,
      vct->getYleft_neighbor_P()  == MPI_PROC_NULL,
      vct->getYright_neighbor_P() == MPI_PROC_NULL,
      vct->getZleft_neighbor_P()  == MPI_PROC_NULL,
      vct->getZright_neighbor_P() == MPI_PROC_NULL, solverStream_);

  // Phase 3: Batched copy-style node halo exchange — shared → ghost nodes
  //   isCenterFlag=false (offset=1), needInterp=false, isParticle=true
  gpuBatchedHaloExchange(ptrs, 10, nxn, nyn, nzn,
                         /*isCenterFlag=*/false, /*isFaceOnly=*/false,
                         /*needInterp=*/false, /*isParticle=*/true, solverStream_);
}

// =========================================================================
//  GPU CommunicateGhostP2G – ALL species batched
//  Processes species in chunks of HALO_MAX_BATCH / 10 . Each chunk runs the full 3-phase
//  sequence (additive halo → boundary adjust → copy halo) independently.
// =========================================================================

void EMfields3D::gpuCommunicateGhostP2G_AllSpecies()
{
  const VirtualTopology3D* vct = &get_vct();
  const int nFieldsPerSpecies = 10;
  const int speciesPerBatch = HALO_MAX_BATCH / nFieldsPerSpecies;  // floor

  for (int isStart = 0; isStart < ns; isStart += speciesPerBatch) {
    const int isEnd   = std::min(isStart + speciesPerBatch, ns);
    const int nsBatch = isEnd - isStart;
    const int nFields = nsBatch * nFieldsPerSpecies;

    // Gather device pointers for this chunk
    double* ptrs[HALO_MAX_BATCH];
    for (int is = 0; is < nsBatch; is++) {
      int off = is * nFieldsPerSpecies;
      int src = isStart + is;
      ptrs[off + 0] = d_rhons.speciesPtr(src);
      ptrs[off + 1] = d_Jxs.speciesPtr(src);
      ptrs[off + 2] = d_Jys.speciesPtr(src);
      ptrs[off + 3] = d_Jzs.speciesPtr(src);
      ptrs[off + 4] = d_pXXsn.speciesPtr(src);
      ptrs[off + 5] = d_pXYsn.speciesPtr(src);
      ptrs[off + 6] = d_pXZsn.speciesPtr(src);
      ptrs[off + 7] = d_pYYsn.speciesPtr(src);
      ptrs[off + 8] = d_pYZsn.speciesPtr(src);
      ptrs[off + 9] = d_pZZsn.speciesPtr(src);
    }

    // Phase 1: Batched additive (interpolating) halo exchange — ghost → shared nodes
    gpuBatchedHaloExchange(ptrs, nFields, nxn, nyn, nzn,
                           /*isCenterFlag=*/true, /*isFaceOnly=*/false,
                           /*needInterp=*/true, /*isParticle=*/true, solverStream_);

    // Phase 2: adjust non-periodic boundary densities
    gpuAdjustNonPeriodicDensities(
        nFields, d_ptrArray_, nxn, nyn, nzn,
        vct->getXleft_neighbor_P()  == MPI_PROC_NULL,
        vct->getXright_neighbor_P() == MPI_PROC_NULL,
        vct->getYleft_neighbor_P()  == MPI_PROC_NULL,
        vct->getYright_neighbor_P() == MPI_PROC_NULL,
        vct->getZleft_neighbor_P()  == MPI_PROC_NULL,
        vct->getZright_neighbor_P() == MPI_PROC_NULL, solverStream_);

    // Phase 3: Batched copy-style node halo exchange — shared → ghost nodes
    gpuBatchedHaloExchange(ptrs, nFields, nxn, nyn, nzn,
                           /*isCenterFlag=*/false, /*isFaceOnly=*/false,
                           /*needInterp=*/false, /*isParticle=*/true, solverStream_);
  }
}

// =========================================================================
//  GPU setZeroDerivedMoments
// =========================================================================

void EMfields3D::gpuSetZeroDerivedMoments()
{
  d_Jx.setAll(0.0, solverStream_);
  d_Jy.setAll(0.0, solverStream_);
  d_Jz.setAll(0.0, solverStream_);
  d_Jxh.setAll(0.0, solverStream_);
  d_Jyh.setAll(0.0, solverStream_);
  d_Jzh.setAll(0.0, solverStream_);
  d_rhon.setAll(0.0, solverStream_);
  d_rhoc.setAll(0.0, solverStream_);
  d_rhoh.setAll(0.0, solverStream_);
}

// =========================================================================
//  GPU sumOverSpecies: rhon = Σ_s rhons_s
// =========================================================================

void EMfields3D::gpuSumOverSpecies()
{
  size_t nodeSize = (size_t)nxn * nyn * nzn;
  for (int is = 0; is < ns; is++)
    gpuSum(d_rhon.devPtr(), d_rhons.speciesPtr(is), nodeSize, solverStream_);
}

// =========================================================================
//  GPU interpDensitiesN2C: rhoc = interp(rhon)
// =========================================================================

void EMfields3D::gpuInterpDensitiesN2C()
{
  gpuInterpN2C(d_rhoc.devPtr(), d_rhon.devPtr(), nxc, nyc, nzc, solverStream_);
}

// =========================================================================
//  GPU OpenBoundaryInflowESource: zero source RHS on open inflow faces
// =========================================================================

void EMfields3D::gpuOpenBoundaryInflowESource(double* dX, double* dY, double* dZ,
                                               int nx, int ny, int nz)
{
  const VirtualTopology3D* vct = &get_vct();
  const Collective* col = &get_col();

  if (vct->getXleft_neighbor() == MPI_PROC_NULL && bcEMfaceXleft == 2 && col->getBcPfaceXleft() == 2)
    gpuOpenBCZeroFace3(dX, dY, dZ, 0, 1, nx, ny, nz, solverStream_);
  if (vct->getXright_neighbor() == MPI_PROC_NULL && bcEMfaceXright == 2 && col->getBcPfaceXright() == 2)
    gpuOpenBCZeroFace3(dX, dY, dZ, 0, nx - 2, nx, ny, nz, solverStream_);
  if (vct->getYleft_neighbor() == MPI_PROC_NULL && bcEMfaceYleft == 2 && col->getBcPfaceYleft() == 2)
    gpuOpenBCZeroFace3(dX, dY, dZ, 1, 1, nx, ny, nz, solverStream_);
  if (vct->getYright_neighbor() == MPI_PROC_NULL && bcEMfaceYright == 2 && col->getBcPfaceYright() == 2)
    gpuOpenBCZeroFace3(dX, dY, dZ, 1, ny - 2, nx, ny, nz, solverStream_);
  if (vct->getZleft_neighbor() == MPI_PROC_NULL && bcEMfaceZleft == 2 && col->getBcPfaceZleft() == 2)
    gpuOpenBCZeroFace3(dX, dY, dZ, 2, 1, nx, ny, nz, solverStream_);
  if (vct->getZright_neighbor() == MPI_PROC_NULL && bcEMfaceZright == 2 && col->getBcPfaceZright() == 2)
    gpuOpenBCZeroFace3(dX, dY, dZ, 2, nz - 2, nx, ny, nz, solverStream_);
}

// =========================================================================
//  GPU OpenBoundaryInflowEImage: image = vect - injE on open inflow faces
// =========================================================================

void EMfields3D::gpuOpenBoundaryInflowEImage(double* imX, double* imY, double* imZ,
                                              const double* vX, const double* vY, const double* vZ,
                                              int nx, int ny, int nz)
{
  const VirtualTopology3D* vct = &get_vct();
  const Collective* col = &get_col();

  // injE = -(ue0,ve0,we0) × (B0x,B0y,B0z)
  double injE[3];
  injE[0] = -(ve0 * B0z - we0 * B0y);
  injE[1] = -(we0 * B0x - ue0 * B0z);
  injE[2] = -(ue0 * B0y - ve0 * B0x);

  if (vct->getXleft_neighbor() == MPI_PROC_NULL && bcEMfaceXleft == 2 && col->getBcPfaceXleft() == 2)
    gpuOpenBCImageDiffFace3(imX, imY, imZ, vX, vY, vZ, injE[0], injE[1], injE[2], 0, 1, nx, ny, nz, solverStream_);
  // Xright image disabled (matches CPU)
  if (vct->getYleft_neighbor() == MPI_PROC_NULL && bcEMfaceYleft == 2 && col->getBcPfaceYleft() == 2)
    gpuOpenBCImageDiffFace3(imX, imY, imZ, vX, vY, vZ, injE[0], injE[1], injE[2], 1, 1, nx, ny, nz, solverStream_);
  if (vct->getYright_neighbor() == MPI_PROC_NULL && bcEMfaceYright == 2 && col->getBcPfaceYright() == 2)
    gpuOpenBCImageDiffFace3(imX, imY, imZ, vX, vY, vZ, injE[0], injE[1], injE[2], 1, ny - 2, nx, ny, nz, solverStream_);
  if (vct->getZleft_neighbor() == MPI_PROC_NULL && bcEMfaceZleft == 2 && col->getBcPfaceZleft() == 2)
    gpuOpenBCImageDiffFace3(imX, imY, imZ, vX, vY, vZ, injE[0], injE[1], injE[2], 2, 1, nx, ny, nz, solverStream_);
  if (vct->getZright_neighbor() == MPI_PROC_NULL && bcEMfaceZright == 2 && col->getBcPfaceZright() == 2)
    gpuOpenBCImageDiffFace3(imX, imY, imZ, vX, vY, vZ, injE[0], injE[1], injE[2], 2, nz - 2, nx, ny, nz, solverStream_);
}

// =========================================================================
//  GPU OpenBoundaryInflowE: SAL blend / Dirichlet on E inflow faces
// =========================================================================

void EMfields3D::gpuOpenBoundaryInflowE(double* dX, double* dY, double* dZ,
                                         int nx, int ny, int nz)
{
  const VirtualTopology3D* vct = &get_vct();
  const Collective* col = &get_col();

  double injE[3];
  injE[0] = -(ve0 * B0z - we0 * B0y);
  injE[1] = -(we0 * B0x - ue0 * B0z);
  injE[2] = -(ue0 * B0y - ve0 * B0x);

  double invNL = (n_layers_sal > 0) ? 1.0 / (double)n_layers_sal : 1.0;

  if (yes_sal) {
    // SAL mode: blend E toward injE over n_layers_sal layers
    if (vct->getXleft_neighbor() == MPI_PROC_NULL && bcEMfaceXleft == 2 && col->getBcPfaceXleft() == 2)
      gpuSALBlendLayers3(dX, dY, dZ, injE[0], injE[1], injE[2],
                         0, 0, n_layers_sal, invNL, 1, nx, ny, nz, solverStream_);
    // Xright: outflow — copy from interior reference plane (bcPface==3)
    if (vct->getXright_neighbor() == MPI_PROC_NULL && bcEMfaceXright == 2 && col->getBcPfaceXright() == 3)
      gpuExtrapolateLayers3(dX, dY, dZ, 0, nx - n_layers_sal - 1, nx - 1,
                            nx - 2 - n_layers_sal, nx, ny, nz, solverStream_);
    if (vct->getYleft_neighbor() == MPI_PROC_NULL && bcEMfaceYleft == 2 && col->getBcPfaceYleft() == 2)
      gpuSALBlendLayers3(dX, dY, dZ, injE[0], injE[1], injE[2],
                         1, 0, n_layers_sal, invNL, 1, nx, ny, nz, solverStream_);
    if (vct->getYright_neighbor() == MPI_PROC_NULL && bcEMfaceYright == 2 && col->getBcPfaceYright() == 2)
      gpuSALBlendLayers3(dX, dY, dZ, injE[0], injE[1], injE[2],
                         1, ny - n_layers_sal - 1, ny - 1, invNL, 0, nx, ny, nz, solverStream_);
    if (vct->getZleft_neighbor() == MPI_PROC_NULL && bcEMfaceZleft == 2 && col->getBcPfaceZleft() == 2)
      gpuSALBlendLayers3(dX, dY, dZ, injE[0], injE[1], injE[2],
                         2, 0, n_layers_sal, invNL, 1, nx, ny, nz, solverStream_);
    if (vct->getZright_neighbor() == MPI_PROC_NULL && bcEMfaceZright == 2 && col->getBcPfaceZright() == 2)
      gpuSALBlendLayers3(dX, dY, dZ, injE[0], injE[1], injE[2],
                         2, nz - n_layers_sal - 1, nz - 1, invNL, 0, nx, ny, nz, solverStream_);
  } else {
    // No SAL: Dirichlet inflow or extrapolation from interior
    if (vct->getXleft_neighbor() == MPI_PROC_NULL && bcEMfaceXleft == 2 && col->getBcPfaceXleft() == 2)
      gpuSetConstLayers3(dX, dY, dZ, injE[0], injE[1], injE[2],
                         0, 0, n_layers_sal, nx, ny, nz, solverStream_);
    if (vct->getXright_neighbor() == MPI_PROC_NULL && bcEMfaceXright == 2 && col->getBcPfaceXright() == 3)
      gpuExtrapolateLayers3(dX, dY, dZ, 0, nx - n_layers_sal - 1, nx - 1,
                            nx - 2 - n_layers_sal, nx, ny, nz, solverStream_);
    if (vct->getYleft_neighbor() == MPI_PROC_NULL && bcEMfaceYleft == 2 && col->getBcPfaceYleft() == 2)
      gpuExtrapolateLayers3(dX, dY, dZ, 1, 0, n_layers_sal,
                            n_layers_sal + 1, nx, ny, nz, solverStream_);
    if (vct->getYright_neighbor() == MPI_PROC_NULL && bcEMfaceYright == 2 && col->getBcPfaceYright() == 2)
      gpuExtrapolateLayers3(dX, dY, dZ, 1, ny - n_layers_sal - 1, ny - 1,
                            ny - 2 - n_layers_sal, nx, ny, nz, solverStream_);
    if (vct->getZleft_neighbor() == MPI_PROC_NULL && bcEMfaceZleft == 2 && col->getBcPfaceZleft() == 2)
      gpuExtrapolateLayers3(dX, dY, dZ, 2, 0, n_layers_sal,
                            n_layers_sal + 1, nx, ny, nz, solverStream_);
    if (vct->getZright_neighbor() == MPI_PROC_NULL && bcEMfaceZright == 2 && col->getBcPfaceZright() == 2)
      gpuExtrapolateLayers3(dX, dY, dZ, 2, nz - n_layers_sal - 1, nz - 1,
                            nz - 2 - n_layers_sal, nx, ny, nz, solverStream_);
  }
}

// =========================================================================
//  GPU OpenBoundaryInflowB: SAL blend / extrapolation on center B
// =========================================================================

void EMfields3D::gpuOpenBoundaryInflowB(double* dX, double* dY, double* dZ,
                                         int nx, int ny, int nz)
{
  const VirtualTopology3D* vct = &get_vct();
  double invNL = (n_layers_sal > 0) ? 1.0 / (double)n_layers_sal : 1.0;

  // Xleft
  if (vct->getXleft_neighbor() == MPI_PROC_NULL && bcEMfaceXleft == 2 && nx > 10) {
    if (yes_sal)
      gpuSALBlendLayers3(dX, dY, dZ, B0x, B0y, B0z,
                         0, 0, n_layers_sal, invNL, 1, nx, ny, nz, solverStream_);
    else
      gpuSetConstLayers3(dX, dY, dZ, B0x, B0y, B0z,
                         0, 0, n_layers_sal, nx, ny, nz, solverStream_);
  }
  // Xright: always extrapolation from interior
  if (vct->getXright_neighbor() == MPI_PROC_NULL && bcEMfaceXright == 2 && nx > 10)
    gpuExtrapolateLayers3(dX, dY, dZ, 0, nx - n_layers_sal - 1, nx - 1,
                          nx - 2 - n_layers_sal, nx, ny, nz, solverStream_);
  // Yleft
  if (vct->getYleft_neighbor() == MPI_PROC_NULL && bcEMfaceYleft == 2 && ny > 10) {
    if (yes_sal)
      gpuSALBlendLayers3(dX, dY, dZ, B0x, B0y, B0z,
                         1, 0, n_layers_sal, invNL, 1, nx, ny, nz, solverStream_);
    else
      gpuExtrapolateLayers3(dX, dY, dZ, 1, 0, n_layers_sal,
                            n_layers_sal + 1, nx, ny, nz, solverStream_);
  }
  // Yright
  if (vct->getYright_neighbor() == MPI_PROC_NULL && bcEMfaceYright == 2 && ny > 10) {
    if (yes_sal)
      gpuSALBlendLayers3(dX, dY, dZ, B0x, B0y, B0z,
                         1, ny - n_layers_sal - 1, ny - 1, invNL, 0, nx, ny, nz, solverStream_);
    else
      gpuExtrapolateLayers3(dX, dY, dZ, 1, ny - n_layers_sal - 1, ny - 1,
                            ny - 2 - n_layers_sal, nx, ny, nz, solverStream_);
  }
  // Zleft
  if (vct->getZleft_neighbor() == MPI_PROC_NULL && bcEMfaceZleft == 2 && nz > 10) {
    if (yes_sal)
      gpuSALBlendLayers3(dX, dY, dZ, B0x, B0y, B0z,
                         2, 0, n_layers_sal, invNL, 1, nx, ny, nz, solverStream_);
    else
      gpuExtrapolateLayers3(dX, dY, dZ, 2, 0, n_layers_sal,
                            n_layers_sal + 1, nx, ny, nz, solverStream_);
  }
  // Zright
  if (vct->getZright_neighbor() == MPI_PROC_NULL && bcEMfaceZright == 2 && nz > 10) {
    if (yes_sal)
      gpuSALBlendLayers3(dX, dY, dZ, B0x, B0y, B0z,
                         2, nz - n_layers_sal - 1, nz - 1, invNL, 0, nx, ny, nz, solverStream_);
    else
      gpuExtrapolateLayers3(dX, dY, dZ, 2, nz - n_layers_sal - 1, nz - 1,
                            nz - 2 - n_layers_sal, nx, ny, nz, solverStream_);
  }
}

// =========================================================================
//  GPU fixBcGEM / fixBnGEM / fixBforcefree
// =========================================================================

void EMfields3D::gpuFixBcGEM()
{
  const VirtualTopology3D* vct = &get_vct();
  double LyH = Ly / 2.0;
  if (vct->getYright_neighbor() == MPI_PROC_NULL)
    gpuFixBcGEMKernel(d_Bxc.devPtr(), d_Byc.devPtr(), d_Bzc.devPtr(),
                      B0x, B0y, B0z, yStart, dy, LyH, delta, 1, nxc, nyc, nzc, solverStream_);
  if (vct->getYleft_neighbor() == MPI_PROC_NULL)
    gpuFixBcGEMKernel(d_Bxc.devPtr(), d_Byc.devPtr(), d_Bzc.devPtr(),
                      B0x, B0y, B0z, yStart, dy, LyH, delta, 0, nxc, nyc, nzc, solverStream_);
}

void EMfields3D::gpuFixBnGEM()
{
  const VirtualTopology3D* vct = &get_vct();
  double LyH = Ly / 2.0;
  if (vct->getYright_neighbor() == MPI_PROC_NULL)
    gpuFixBnGEMKernel(d_Bxn.devPtr(), d_Byn.devPtr(), d_Bzn.devPtr(),
                      B0x, B0y, B0z, yStart, dy, LyH, delta, 1, nxn, nyn, nzn, nyc, solverStream_);
  if (vct->getYleft_neighbor() == MPI_PROC_NULL)
    gpuFixBnGEMKernel(d_Bxn.devPtr(), d_Byn.devPtr(), d_Bzn.devPtr(),
                      B0x, B0y, B0z, yStart, dy, LyH, delta, 0, nxn, nyn, nzn, nyc, solverStream_);
}

void EMfields3D::gpuFixBforcefree()
{
  const VirtualTopology3D* vct = &get_vct();
  double LyH = Ly / 2.0;
  if (vct->getYright_neighbor() == MPI_PROC_NULL)
    gpuFixBforcefreeKernel(d_Bxc.devPtr(), d_Byc.devPtr(), d_Bzc.devPtr(),
                           B0x, B0y, B0z, yStart, dy, LyH, delta, 1, nxc, nyc, nzc, solverStream_);
  if (vct->getYleft_neighbor() == MPI_PROC_NULL)
    gpuFixBforcefreeKernel(d_Bxc.devPtr(), d_Byc.devPtr(), d_Bzc.devPtr(),
                           B0x, B0y, B0z, yStart, dy, LyH, delta, 0, nxc, nyc, nzc, solverStream_);
}

// =========================================================================
//  GPU ConstantChargePlanet
// =========================================================================

void EMfields3D::gpuConstantChargePlanet(double R, double x_center, double y_center, double z_center)
{
  for (int is = 0; is < ns; is++) {
    double ff = qom[is] / fabs(qom[is]);
    double val = ff * rhoINIT[is] / FourPI;
    gpuConstantChargePlanetKernel(d_rhons.speciesPtr(is), val,
                                  R, x_center, y_center, z_center,
                                  xStart, yStart, zStart, dx, dy, dz,
                                  nxn, nyn, nzn, solverStream_);
  }
}

void EMfields3D::gpuConstantChargePlanet2DPlaneXZ(double R, double x_center, double z_center)
{
  for (int is = 0; is < ns; is++) {
    double sign_q = qom[is] / fabs(qom[is]);
    double val = sign_q * rhoINIT[is] / FourPI;
    gpuConstantChargePlanet2DKernel(d_rhons.speciesPtr(is), val,
                                    R, x_center, z_center,
                                    xStart, zStart, dx, dz,
                                    nxn, nyn, nzn, solverStream_);
  }
}

// =========================================================================
//  GPU PoissonImage: A*x for Poisson solver (center Laplacian)
// =========================================================================

void EMfields3D::gpuPoissonImage(double* d_im, double* d_vec)
{
  const Grid* grid = &get_grid();
  double _invdx = grid->get_invdx();
  double _invdy = grid->get_invdy();
  double _invdz = grid->get_invdz();

  const int nPoisson = (nxc - 2) * (nyc - 2) * (nzc - 2);
  size_t centSize = (size_t)nxc * nyc * nzc;

  d_poissonTemp.setAll(0.0, solverStream_);
  d_poissonIm.setAll(0.0, solverStream_);

  // Krylov → physical
  gpuSolver2Phys1(d_poissonTemp.devPtr(), d_vec, nxc, nyc, nzc, solverStream_);

  // Communicate ghost cells (center box stencil)
  gpuCommunicateCenterBoxStencilBC(nxc, nyc, nzc, d_poissonTemp, 1, 1, 1, 1, 1, 1);

  // Laplacian
  gpuLapC2CKernel(d_poissonIm.devPtr(), d_poissonTemp.devPtr(),
                  nxc, nyc, nzc, _invdx * _invdx, _invdy * _invdy, _invdz * _invdz, solverStream_);

  // Physical → Krylov
  gpuPhys2Solver1(d_im, d_poissonIm.devPtr(), nxc, nyc, nzc, solverStream_);
}

// =========================================================================
//  GPU PoissonCorrection (div(E) cleaning)
// =========================================================================

void EMfields3D::gpuPoissonCorrection(int cycle)
{
  if (!PoissonCorrection || cycle % PoissonCorrectionCycle != 0)
    return;

  const Collective* col = &get_col();
  const VirtualTopology3D* vct = &get_vct();
  const Grid* grid = &get_grid();
  double _invdx = grid->get_invdx();
  double _invdy = grid->get_invdy();
  double _invdz = grid->get_invdz();

  const int nPoisson = (nxc - 2) * (nyc - 2) * (nzc - 2);
  size_t nodeSize = (size_t)nxn * nyn * nzn;
  size_t centSize = (size_t)nxc * nyc * nzc;

  // Zero work arrays
  d_xkrylovPoisson_E.setZero(solverStream_);
  d_divE_work.setAll(0.0, solverStream_);
  d_tempC.setAll(0.0, solverStream_);
  d_gradPHIX_work.setAll(0.0, solverStream_);
  d_gradPHIY_work.setAll(0.0, solverStream_);
  d_gradPHIZ_work.setAll(0.0, solverStream_);

  // div(E) on centers
  gpuDivN2C(d_divE_work.devPtr(),
            d_Ex.devPtr(), d_Ey.devPtr(), d_Ez.devPtr(),
            nxc, nyc, nzc, _invdx, _invdy, _invdz, solverStream_);

  // tempC = -4π * rhoc
  gpuScaleCopy(d_tempC.devPtr(), d_rhoc.devPtr(), -FourPI, centSize, solverStream_);

  // divE_work += tempC
  gpuSum(d_divE_work.devPtr(), d_tempC.devPtr(), centSize, solverStream_);

  // RHS → Krylov
  gpuPhys2Solver1(d_bkrylovPoisson_E.devPtr(), d_divE_work.devPtr(), nxc, nyc, nzc, solverStream_);

  if (vct->getCartesian_rank() == 0)
    cout << "*** DIVERGENCE CLEANING div(E) using GMRes [GPU] ***" << endl;

  // Solve
  MPI_Comm fieldcomm = vct->getFieldComm();
  gpuGMRES_impl(this, &EMfields3D::gpuPoissonImage,
                d_xkrylovPoisson_E.devPtr(), nPoisson,
                d_bkrylovPoisson_E.devPtr(),
                20, 200, GMREStol,
                d_blasScratch,
                d_gmresV, d_gmresW, gmresVAlloc,
                fieldcomm, solverStream_,
                h_gmresReduceLocal, h_gmresReduceGlobal,
                h_gmresH, h_gmresG, h_gmresCS, h_gmresSN, h_gmresY);

  // Solution → physical
  gpuSolver2Phys1(d_PHI.devPtr(), d_xkrylovPoisson_E.devPtr(), nxc, nyc, nzc, solverStream_);
  gpuCommunicateCenterBC(nxc, nyc, nzc, d_PHI, 2, 2, 2, 2, 2, 2);

  // grad(PHI)
  gpuGradC2N(d_gradPHIX_work.devPtr(), d_gradPHIY_work.devPtr(), d_gradPHIZ_work.devPtr(),
             d_PHI.devPtr(), nxn, nyn, nzn, _invdx, _invdy, _invdz, solverStream_);

  // E -= grad(PHI)
  gpuSub(d_Ex.devPtr(), d_gradPHIX_work.devPtr(), nodeSize, solverStream_);
  gpuSub(d_Ey.devPtr(), d_gradPHIY_work.devPtr(), nodeSize, solverStream_);
  gpuSub(d_Ez.devPtr(), d_gradPHIZ_work.devPtr(), nodeSize, solverStream_);
}

// =========================================================================
//  GPU applyDivBCleaning
// =========================================================================

void EMfields3D::gpuApplyDivBCleaning()
{
  const Collective* col = &get_col();
  const VirtualTopology3D* vct = &get_vct();
  const Grid* grid = &get_grid();
  double _invdx = grid->get_invdx();
  double _invdy = grid->get_invdy();
  double _invdz = grid->get_invdz();

  const int nPoisson = (nxc - 2) * (nyc - 2) * (nzc - 2);

  // Zero work arrays
  d_divBwork.setAll(0.0, solverStream_);
  d_gradPSIX.setAll(0.0, solverStream_);
  d_gradPSIY.setAll(0.0, solverStream_);
  d_gradPSIZ.setAll(0.0, solverStream_);
  d_xkrylovPoisson_B.setZero(solverStream_);

  // div(Bn) on centers
  gpuDivN2C(d_divBwork.devPtr(),
            d_Bxn.devPtr(), d_Byn.devPtr(), d_Bzn.devPtr(),
            nxc, nyc, nzc, _invdx, _invdy, _invdz, solverStream_);

  // RHS → Krylov
  gpuPhys2Solver1(d_bkrylovPoisson_B.devPtr(), d_divBwork.devPtr(), nxc, nyc, nzc, solverStream_);

  if (vct->getCartesian_rank() == 0)
    cout << "*** DIVERGENCE CLEANING div(B)=0 using GMRes [GPU] ***" << endl;

  // Solve
  MPI_Comm fieldcomm = vct->getFieldComm();
  gpuGMRES_impl(this, &EMfields3D::gpuPoissonImage,
                d_xkrylovPoisson_B.devPtr(), nPoisson,
                d_bkrylovPoisson_B.devPtr(),
                20, 200, GMREStol,
                d_blasScratch,
                d_gmresV, d_gmresW, gmresVAlloc,
                fieldcomm, solverStream_,
                h_gmresReduceLocal, h_gmresReduceGlobal,
                h_gmresH, h_gmresG, h_gmresCS, h_gmresSN, h_gmresY);

  // Solution → physical
  gpuSolver2Phys1(d_PSI.devPtr(), d_xkrylovPoisson_B.devPtr(), nxc, nyc, nzc, solverStream_);
  gpuCommunicateCenterBC(nxc, nyc, nzc, d_PSI, 2, 2, 2, 2, 2, 2);

  // grad(PSI)
  gpuGradC2N(d_gradPSIX.devPtr(), d_gradPSIY.devPtr(), d_gradPSIZ.devPtr(),
             d_PSI.devPtr(), nxn, nyn, nzn, _invdx, _invdy, _invdz, solverStream_);

  // Subtract grad(PSI) from Bn on boundary layers
  if (vct->getXleft_neighbor() == MPI_PROC_NULL && bcEMfaceXleft == 2)
    gpuSubBoundaryLayers3(d_Bxn.devPtr(), d_Byn.devPtr(), d_Bzn.devPtr(),
                          d_gradPSIX.devPtr(), d_gradPSIY.devPtr(), d_gradPSIZ.devPtr(),
                          0, 0, n_layers_sal - 1, nxn, nyn, nzn, solverStream_);
  if (vct->getXright_neighbor() == MPI_PROC_NULL && bcEMfaceXright == 2)
    gpuSubBoundaryLayers3(d_Bxn.devPtr(), d_Byn.devPtr(), d_Bzn.devPtr(),
                          d_gradPSIX.devPtr(), d_gradPSIY.devPtr(), d_gradPSIZ.devPtr(),
                          0, nxn - n_layers_sal, nxn - 1, nxn, nyn, nzn, solverStream_);
  if (vct->getYleft_neighbor() == MPI_PROC_NULL && bcEMfaceYleft == 2)
    gpuSubBoundaryLayers3(d_Bxn.devPtr(), d_Byn.devPtr(), d_Bzn.devPtr(),
                          d_gradPSIX.devPtr(), d_gradPSIY.devPtr(), d_gradPSIZ.devPtr(),
                          1, 0, n_layers_sal - 1, nxn, nyn, nzn, solverStream_);
  if (vct->getYright_neighbor() == MPI_PROC_NULL && bcEMfaceYright == 2)
    gpuSubBoundaryLayers3(d_Bxn.devPtr(), d_Byn.devPtr(), d_Bzn.devPtr(),
                          d_gradPSIX.devPtr(), d_gradPSIY.devPtr(), d_gradPSIZ.devPtr(),
                          1, nyn - n_layers_sal, nyn - 1, nxn, nyn, nzn, solverStream_);
  if (vct->getZleft_neighbor() == MPI_PROC_NULL && bcEMfaceZleft == 2)
    gpuSubBoundaryLayers3(d_Bxn.devPtr(), d_Byn.devPtr(), d_Bzn.devPtr(),
                          d_gradPSIX.devPtr(), d_gradPSIY.devPtr(), d_gradPSIZ.devPtr(),
                          2, 0, n_layers_sal - 1, nxn, nyn, nzn, solverStream_);
  if (vct->getZright_neighbor() == MPI_PROC_NULL && bcEMfaceZright == 2)
    gpuSubBoundaryLayers3(d_Bxn.devPtr(), d_Byn.devPtr(), d_Bzn.devPtr(),
                          d_gradPSIX.devPtr(), d_gradPSIY.devPtr(), d_gradPSIZ.devPtr(),
                          2, nzn - n_layers_sal, nzn - 1, nxn, nyn, nzn, solverStream_);

  // Communicate corrected Bn (batched: 3 fields in 1 MPI round)
  gpuCommunicateNodeBC_3mixed(nxn, nyn, nzn,
      d_Bxn, col->bcBx, d_Byn, col->bcBy, d_Bzn, col->bcBz);

  // Recompute center B from corrected node B
  gpuInterpN2C(d_Bxc.devPtr(), d_Bxn.devPtr(), nxc, nyc, nzc, solverStream_);
  gpuInterpN2C(d_Byc.devPtr(), d_Byn.devPtr(), nxc, nyc, nzc, solverStream_);
  gpuInterpN2C(d_Bzc.devPtr(), d_Bzn.devPtr(), nxc, nyc, nzc, solverStream_);

  // Communicate corrected center B (batched: 3 fields in 1 MPI round)
  gpuCommunicateCenterBC_3mixed(nxc, nyc, nzc,
      d_Bxc, col->bcBx, d_Byc, col->bcBy, d_Bzc, col->bcBz);
}

#endif // GPU_SOLVER

/** method to convert a 1D field in a 3D field not considering guard cells*/
void solver2phys(arr3_double vectPhys, double *vectSolver, int nx, int ny, int nz)
{
#pragma omp parallel for collapse(2)
  for (int i = 1; i < nx - 1; i++)
    for (int j = 1; j < ny - 1; j++)
    {
      int idx = (i - 1) * (ny - 2) + (j - 1);
      for (int k = 1; k < nz - 1; k++)
        vectPhys[i][j][k] = vectSolver[idx * (nz - 2) + (k - 1)];
    }
}
/** method to convert a 1D field in a 3D field not considering guard cells*/
void solver2phys(arr3_double vectPhys1, arr3_double vectPhys2, arr3_double vectPhys3, double *vectSolver, int nx, int ny, int nz)
{
#pragma omp parallel for collapse(2)
  for (int i = 1; i < nx - 1; i++)
    for (int j = 1; j < ny - 1; j++)
    {
      int idx = (i - 1) * (ny - 2) + (j - 1);
      for (int k = 1; k < nz - 1; k++)
      {
        int idy = (idx * (nz - 2) + (k - 1)) * 3;
        vectPhys1[i][j][k] = vectSolver[idy];
        vectPhys2[i][j][k] = vectSolver[idy + 1];
        vectPhys3[i][j][k] = vectSolver[idy + 2];
      }
    }
}
/** method to convert a 3D field in a 1D field not considering guard cells*/
void phys2solver(double *vectSolver, const arr3_double vectPhys, int nx, int ny, int nz)
{
#pragma omp parallel for collapse(2)
  for (int i = 1; i < nx - 1; i++)
    for (int j = 1; j < ny - 1; j++)
    {
      int idx = (i - 1) * (ny - 2) + (j - 1);
      for (int k = 1; k < nz - 1; k++)
        vectSolver[idx * (nz - 2) + (k - 1)] = vectPhys.get(i, j, k);
    }
}
/** method to convert a 3D field in a 1D field not considering guard cells*/
void phys2solver(double *vectSolver, const arr3_double vectPhys1, const arr3_double vectPhys2, const arr3_double vectPhys3, int nx, int ny, int nz)
{
#pragma omp parallel for collapse(2)
  for (int i = 1; i < nx - 1; i++)
    for (int j = 1; j < ny - 1; j++)
    {
      int idx = (i - 1) * (ny - 2) + (j - 1);
      for (int k = 1; k < nz - 1; k++)
      {
        int idy = (idx * (nz - 2) + (k - 1)) * 3;
        vectSolver[idy] = vectPhys1.get(i, j, k);
        vectSolver[idy + 1] = vectPhys2.get(i, j, k);
        vectSolver[idy + 2] = vectPhys3.get(i, j, k);
      }
    }
}
/*! Calculate Electric field with the implicit solver: the Maxwell solver method is called here */
void EMfields3D::calculateE(int cycle)
{
  const Collective *col = &get_col();
  const VirtualTopology3D *vct = &get_vct();
  const Grid *grid = &get_grid();

  if (vct->getCartesian_rank() == 0)
    cout << "*** E CALCULATION [CPU] ***" << endl;

  const int nMaxwellKrylov = 3 * (nxn - 2) * (nyn - 2) * (nzn - 2);
  const int nPoissonKrylov_E = (nxc - 2) * (nyc - 2) * (nzc - 2);

  // set to zero persistent Krylov vectors and work arrays
  eqValue(0.0, xkrylovMaxwell, nMaxwellKrylov);
  eqValue(0.0, bkrylovMaxwell, nMaxwellKrylov);
  eqValue(0.0, divE_work, nxc, nyc, nzc);
  eqValue(0.0, tempC, nxc, nyc, nzc);
  eqValue(0.0, gradPHIX_work, nxn, nyn, nzn);
  eqValue(0.0, gradPHIY_work, nxn, nyn, nzn);
  eqValue(0.0, gradPHIZ_work, nxn, nyn, nzn);
  // Adjust E calculating laplacian(PHI) = div(E) -4*PI*rho DIVERGENCE CLEANING
  // correct the e field regularly, to fulfill the Gussian law
  if (PoissonCorrection && cycle % PoissonCorrectionCycle == 0)
  {
    eqValue(0.0, xkrylovPoisson_E, nPoissonKrylov_E);

    grid->divN2C(divE_work, Ex, Ey, Ez);
    scale(tempC, rhoc, -FourPI, nxc, nyc, nzc);
    sum(divE_work, tempC, nxc, nyc, nzc);
    // move to krylov space
    phys2solver(bkrylovPoisson_E, divE_work, nxc, nyc, nzc);
    // use conjugate gradient first
    // if (!CG(xkrylovPoisson, (nxc - 2) * (nyc - 2) * (nzc - 2), bkrylovPoisson, 3000, CGtol, &Field::PoissonImage, this)) {
    // if (vct->getCartesian_rank() == 0)
    // cout << "CG not Converged. Trying with GMRes. Consider to increase the number of the CG iterations" << endl;
    // eqValue(0.0, xkrylovPoisson, (nxc - 2) * (nyc - 2) * (nzc - 2));
    if (vct->getCartesian_rank() == 0)
      cout << "*** DIVERGENCE CLEANING using GMRes [CPU] ***" << endl;
    GMRES(&Field::PoissonImage, xkrylovPoisson_E, nPoissonKrylov_E, bkrylovPoisson_E, 20, 200, GMREStol, this);

    //}
    solver2phys(PHI, xkrylovPoisson_E, nxc, nyc, nzc);
    communicateCenterBC(nxc, nyc, nzc, PHI, 2, 2, 2, 2, 2, 2, vct, this);
    // calculate the gradient
    grid->gradC2N(gradPHIX_work, gradPHIY_work, gradPHIZ_work, PHI);
    // sub
    sub(Ex, gradPHIX_work, nxn, nyn, nzn);
    sub(Ey, gradPHIY_work, nxn, nyn, nzn);
    sub(Ez, gradPHIZ_work, nxn, nyn, nzn);
  } // end of divergence cleaning

  if (vct->getCartesian_rank() == 0)
    cout << "*** MAXWELL SOLVER [CPU] ***" << endl;
  // prepare the source
  MaxwellSource(bkrylovMaxwell);
  phys2solver(xkrylovMaxwell, Ex, Ey, Ez, nxn, nyn, nzn);

  // solver
  GMRES(&Field::MaxwellImage, xkrylovMaxwell, nMaxwellKrylov, bkrylovMaxwell, 20, 200, GMREStol, this);
  //FGMRES(&Field::MaxwellImage, &Field::MaxwellImageLocal, xkrylov, 3 * (nxn - 2) * (nyn - 2) * (nzn - 2), bkrylov, 20, 200, GMREStol, this);
  
  // move from krylov space to physical space
  solver2phys(Exth, Eyth, Ezth, xkrylovMaxwell, nxn, nyn, nzn);

  addscale(1 / th, -(1.0 - th) / th, Ex, Exth, nxn, nyn, nzn);
  addscale(1 / th, -(1.0 - th) / th, Ey, Eyth, nxn, nyn, nzn);
  addscale(1 / th, -(1.0 - th) / th, Ez, Ezth, nxn, nyn, nzn);

  // apply to smooth to electric field 3 times
  smoothE();

  // communicate so the interpolation can have good values
  communicateNodeBC(nxn, nyn, nzn, Exth, col->bcEx[0], col->bcEx[1], col->bcEx[2], col->bcEx[3], col->bcEx[4], col->bcEx[5], vct, this);
  communicateNodeBC(nxn, nyn, nzn, Eyth, col->bcEy[0], col->bcEy[1], col->bcEy[2], col->bcEy[3], col->bcEy[4], col->bcEy[5], vct, this);
  communicateNodeBC(nxn, nyn, nzn, Ezth, col->bcEz[0], col->bcEz[1], col->bcEz[2], col->bcEz[3], col->bcEz[4], col->bcEz[5], vct, this);
  communicateNodeBC(nxn, nyn, nzn, Ex, col->bcEx[0], col->bcEx[1], col->bcEx[2], col->bcEx[3], col->bcEx[4], col->bcEx[5], vct, this);
  communicateNodeBC(nxn, nyn, nzn, Ey, col->bcEy[0], col->bcEy[1], col->bcEy[2], col->bcEy[3], col->bcEy[4], col->bcEy[5], vct, this);
  communicateNodeBC(nxn, nyn, nzn, Ez, col->bcEz[0], col->bcEz[1], col->bcEz[2], col->bcEz[3], col->bcEz[4], col->bcEz[5], vct, this);

  // OpenBC Inflow: this needs to be integrate to Halo Exchange BC
  OpenBoundaryInflowE(Exth, Eyth, Ezth, nxn, nyn, nzn);
  OpenBoundaryInflowE(Ex, Ey, Ez, nxn, nyn, nzn);
}

/*! Calculate sorgent for Maxwell solver */
void EMfields3D::MaxwellSource(double *bkrylov)
{
  const Collective *col = &get_col();
  const VirtualTopology3D *vct = &get_vct();
  const Grid *grid = &get_grid();

  eqValue(0.0, tempC, nxc, nyc, nzc);
  eqValue(0.0, tempX, nxn, nyn, nzn);
  eqValue(0.0, tempY, nxn, nyn, nzn);
  eqValue(0.0, tempZ, nxn, nyn, nzn);
  eqValue(0.0, tempXN, nxn, nyn, nzn);
  eqValue(0.0, tempYN, nxn, nyn, nzn);
  eqValue(0.0, tempZN, nxn, nyn, nzn);
  eqValue(0.0, temp2X, nxn, nyn, nzn);
  eqValue(0.0, temp2Y, nxn, nyn, nzn);
  eqValue(0.0, temp2Z, nxn, nyn, nzn);

  communicateCenterBC(nxc, nyc, nzc, Bxc, col->bcBx[0], col->bcBx[1], col->bcBx[2], col->bcBx[3], col->bcBx[4], col->bcBx[5], vct, this);
  communicateCenterBC(nxc, nyc, nzc, Byc, col->bcBy[0], col->bcBy[1], col->bcBy[2], col->bcBy[3], col->bcBy[4], col->bcBy[5], vct, this);
  communicateCenterBC(nxc, nyc, nzc, Bzc, col->bcBz[0], col->bcBz[1], col->bcBz[2], col->bcBz[3], col->bcBz[4], col->bcBz[5], vct, this);

  if (get_col().getCase() == "ForceFree")
    fixBforcefree();
  if (get_col().getCase() == "GEM")
    fixBnGEM();
  if (get_col().getCase() == "GEMnoPert")
    fixBnGEM();
  if (get_col().getCase() == "GEMDoubleHarris")
    fixBnGEM();

  // OpenBC:
  OpenBoundaryInflowB(Bxc, Byc, Bzc, nxc, nyc, nzc);

  if (get_col().getCase() == "GEM")
    fixBcGEM();
  if (get_col().getCase() == "GEMnoPert")
    fixBcGEM();
  if (get_col().getCase() == "GEMDoubleHarris")
    fixBcGEM();

  // prepare curl of B for known term of Maxwell solver: for the source term
  grid->curlC2N(tempXN, tempYN, tempZN, Bxc, Byc, Bzc);
  scale(temp2X, Jxh, -FourPI / c, nxn, nyn, nzn);
  scale(temp2Y, Jyh, -FourPI / c, nxn, nyn, nzn);
  scale(temp2Z, Jzh, -FourPI / c, nxn, nyn, nzn);

  /* -- dipole SOURCE version using J_ext,This is not initialized, causing program crash over 2048 processes
  addscale(-FourPI/c,temp2X,Jx_ext,nxn,nyn,nzn);
  addscale(-FourPI/c,temp2Y,Jy_ext,nxn,nyn,nzn);
  addscale(-FourPI/c,temp2Z,Jz_ext,nxn,nyn,nzn);
  // -- end of dipole SOURCE version using J_ext*/

  sum(temp2X, tempXN, nxn, nyn, nzn);
  sum(temp2Y, tempYN, nxn, nyn, nzn);
  sum(temp2Z, tempZN, nxn, nyn, nzn);
  scale(temp2X, delt, nxn, nyn, nzn);
  scale(temp2Y, delt, nxn, nyn, nzn);
  scale(temp2Z, delt, nxn, nyn, nzn);

  communicateCenterBC_P(nxc, nyc, nzc, rhoh, 2, 2, 2, 2, 2, 2, vct, this);
  grid->gradC2N(tempX, tempY, tempZ, rhoh);

  scale(tempX, -delt * delt * FourPI, nxn, nyn, nzn);
  scale(tempY, -delt * delt * FourPI, nxn, nyn, nzn);
  scale(tempZ, -delt * delt * FourPI, nxn, nyn, nzn);
  // sum E, past values
  sum(tempX, Ex, nxn, nyn, nzn);
  sum(tempY, Ey, nxn, nyn, nzn);
  sum(tempZ, Ez, nxn, nyn, nzn);
  // sum curl(B) + jhat part
  sum(tempX, temp2X, nxn, nyn, nzn);
  sum(tempY, temp2Y, nxn, nyn, nzn);
  sum(tempZ, temp2Z, nxn, nyn, nzn);

  // Boundary condition in the known term
  // boundary condition: Xleft
  if (vct->getXleft_neighbor() == MPI_PROC_NULL && bcEMfaceXleft == 0) // perfect conductor
    perfectConductorLeftS(tempX, tempY, tempZ, 0);
  // boundary condition: Xright
  if (vct->getXright_neighbor() == MPI_PROC_NULL && bcEMfaceXright == 0) // perfect conductor
    perfectConductorRightS(tempX, tempY, tempZ, 0);
  // boundary condition: Yleft
  if (vct->getYleft_neighbor() == MPI_PROC_NULL && bcEMfaceYleft == 0) // perfect conductor
    perfectConductorLeftS(tempX, tempY, tempZ, 1);
  // boundary condition: Yright
  if (vct->getYright_neighbor() == MPI_PROC_NULL && bcEMfaceYright == 0) // perfect conductor
    perfectConductorRightS(tempX, tempY, tempZ, 1);
  // boundary condition: Zleft
  if (vct->getZleft_neighbor() == MPI_PROC_NULL && bcEMfaceZleft == 0) // perfect conductor
    perfectConductorLeftS(tempX, tempY, tempZ, 2);
  // boundary condition: Zright
  if (vct->getZright_neighbor() == MPI_PROC_NULL && bcEMfaceZright == 0) // perfect conductor
    perfectConductorRightS(tempX, tempY, tempZ, 2);

  // OpenBC: set RHS to zero at inflow boundary nodes
  OpenBoundaryInflowESource(tempX, tempY, tempZ, nxn, nyn, nzn);

  // physical space -> Krylov space
  phys2solver(bkrylov, tempX, tempY, tempZ, nxn, nyn, nzn);
}

/*! Mapping of Maxwell image to give to solver */
//
// In the field solver, there is one layer of ghost cells.  The
// nodes on the ghost cells define two outer layers of nodes: the
// outermost nodes are clearly in the interior of the neighboring
// subdomain and can naturally be referred to as "ghost nodes",
// but the second-outermost layer is on the boundary between
// subdomains and thus does not clearly belong to any one process.
// Refer to these shared nodes as "boundary nodes".
//
// To compute the laplacian, we first compute the gradient
// at the center of each cell by differencing the values at
// the corners of the cell.  We then compute the laplacian
// (i.e. the divergence of the gradient) at each node by
// differencing the cell-center values in the cells sharing
// the node.
//
// The laplacian is required to be defined on all boundary
// and interior nodes.
//
// In the krylov solver, we make no attempt to use or to
// update the (outer) ghost nodes, and we assume (presumably
// correctly) that the boundary nodes are updated identically
// by all processes that share them.  Therefore, we must
// communicate gradient values in the ghost cells.  The
// subsequent computation of the divergence requires that
// this boundary communication first complete.
//
// An alternative way would be to communicate outer ghost node
// values after each update of Eth.  In this case, there would
// be no need for the 10=3*3+1 boundary communications in the body
// of MaxwellImage() entailed in the calls to lapN2N plus the
// call needed prior to the call to gradC2N.  Of course,
// we would then need to communicate the 3 components of the
// electric field for the outer ghost nodes prior to each call
// to MaxwellImage().  This second alternative would thus reduce
// communication by over a factor of 3.  Essentially, we would
// replace the cost of communicating cell-centered differences
// for ghost cell values with the cost of directly computing them.
//
// Also, while this second method does not increase the potential
// to avoid exposing latency, it can make it easier to do so.
//
// Another change that I would propose: define:
//
//   array4_double physical_vector(3,nxn,nyn,nzn);
//   arr3_double vectX = physical_vector[0];
//   arr3_double vectY = physical_vector[1];
//   arr3_double vectZ = physical_vector[2];
//   vector = &physical_vector[0][0][0][0];
//
// It is currently the case that boundary nodes are
// duplicated in "vector" and therefore receive a weight
// that is twice, four times, or eight times as much as
// other nodes in the Krylov inner product.  The definitions
// above would imply that ghost nodes also appear in the
// inner product.  To avoid this issue, we could simply zero
// ghost nodes before returning to the Krylov solver.  With
// the definitions above, phys2solver() would simply zero
// the ghost nodes and solver2phys() would populate them via
// communication.  Note that it would also be possible, if
// desired, to give duplicated nodes equal weight by
// rescaling their values in these two methods.
//
void EMfields3D::MaxwellImage(double *im, double *vector)
{
  const Collective *col = &get_col();
  const VirtualTopology3D *vct = &get_vct();
  const Grid *grid = &get_grid();

  eqValue(0.0, im, 3 * (nxn - 2) * (nyn - 2) * (nzn - 2));
  eqValue(0.0, imageX, nxn, nyn, nzn);
  eqValue(0.0, imageY, nxn, nyn, nzn);
  eqValue(0.0, imageZ, nxn, nyn, nzn);
  eqValue(0.0, tempX, nxn, nyn, nzn);
  eqValue(0.0, tempY, nxn, nyn, nzn);
  eqValue(0.0, tempZ, nxn, nyn, nzn);
  eqValue(0.0, Dx, nxn, nyn, nzn);
  eqValue(0.0, Dy, nxn, nyn, nzn);
  eqValue(0.0, Dz, nxn, nyn, nzn);
  // move from krylov space to physical space
  solver2phys(vectX, vectY, vectZ, vector, nxn, nyn, nzn);
  grid->lapN2N(imageX, vectX, this);
  grid->lapN2N(imageY, vectY, this);
  grid->lapN2N(imageZ, vectZ, this);
  neg(imageX, nxn, nyn, nzn);
  neg(imageY, nxn, nyn, nzn);
  neg(imageZ, nxn, nyn, nzn);
  // grad(div(mu dot E(n + theta)) mu dot E(n + theta) = D
  MUdot(Dx, Dy, Dz, vectX, vectY, vectZ);
  grid->divN2C(divC, Dx, Dy, Dz);
  // communicate you should put BC
  // think about the Physics
  // communicateCenterBC(nxc,nyc,nzc,divC,1,1,1,1,1,1,vct);

  communicateCenterBC(nxc, nyc, nzc, divC, 2, 2, 2, 2, 2, 2, vct, this);

  grid->gradC2N(tempX, tempY, tempZ, divC);

  // -lap(E(n +theta)) - grad(div(mu dot E(n + theta))
  sub(imageX, tempX, nxn, nyn, nzn);
  sub(imageY, tempY, nxn, nyn, nzn);
  sub(imageZ, tempZ, nxn, nyn, nzn);

  // scale delt*delt
  scale(imageX, delt * delt, nxn, nyn, nzn);
  scale(imageY, delt * delt, nxn, nyn, nzn);
  scale(imageZ, delt * delt, nxn, nyn, nzn);

  // -lap(E(n +theta)) - grad(div(mu dot E(n + theta)) + eps dot E(n + theta)
  sum(imageX, Dx, nxn, nyn, nzn);
  sum(imageY, Dy, nxn, nyn, nzn);
  sum(imageZ, Dz, nxn, nyn, nzn);
  sum(imageX, vectX, nxn, nyn, nzn);
  sum(imageY, vectY, nxn, nyn, nzn);
  sum(imageZ, vectZ, nxn, nyn, nzn);

  // boundary condition: Xleft
  if (vct->getXleft_neighbor() == MPI_PROC_NULL && bcEMfaceXleft == 0) // perfect conductor
    perfectConductorLeft(imageX, imageY, imageZ, vectX, vectY, vectZ, 0);
  // boundary condition: Xright
  if (vct->getXright_neighbor() == MPI_PROC_NULL && bcEMfaceXright == 0) // perfect conductor
    perfectConductorRight(imageX, imageY, imageZ, vectX, vectY, vectZ, 0);
  // boundary condition: Yleft
  if (vct->getYleft_neighbor() == MPI_PROC_NULL && bcEMfaceYleft == 0) // perfect conductor
    perfectConductorLeft(imageX, imageY, imageZ, vectX, vectY, vectZ, 1);
  // boundary condition: Yright
  if (vct->getYright_neighbor() == MPI_PROC_NULL && bcEMfaceYright == 0) // perfect conductor
    perfectConductorRight(imageX, imageY, imageZ, vectX, vectY, vectZ, 1);
  // boundary condition: Zleft
  if (vct->getZleft_neighbor() == MPI_PROC_NULL && bcEMfaceZleft == 0) // perfect conductor
    perfectConductorLeft(imageX, imageY, imageZ, vectX, vectY, vectZ, 2);
  // boundary condition: Zright
  if (vct->getZright_neighbor() == MPI_PROC_NULL && bcEMfaceZright == 0) // perfect conductor
    perfectConductorRight(imageX, imageY, imageZ, vectX, vectY, vectZ, 2);

  // OpenBC: apply inflow BCs to GMRes image if enabled
  if (get_col().getApplyInflowBcsEImage()){
    OpenBoundaryInflowEImage(imageX, imageY, imageZ, vectX, vectY, vectZ, nxn, nyn, nzn);
  }
  // move from physical space to krylov space
  phys2solver(im, imageX, imageY, imageZ, nxn, nyn, nzn);
}

void EMfields3D::MaxwellImageLocal(double *im, double *vector)
{
  const Collective *col = &get_col();
  const VirtualTopology3D *vct = &get_vct();
  const Grid *grid = &get_grid();

  eqValue(0.0, im, 3 * (nxn - 2) * (nyn - 2) * (nzn - 2));
  eqValue(0.0, imageX, nxn, nyn, nzn);
  eqValue(0.0, imageY, nxn, nyn, nzn);
  eqValue(0.0, imageZ, nxn, nyn, nzn);
  eqValue(0.0, tempX, nxn, nyn, nzn);
  eqValue(0.0, tempY, nxn, nyn, nzn);
  eqValue(0.0, tempZ, nxn, nyn, nzn);
  eqValue(0.0, Dx, nxn, nyn, nzn);
  eqValue(0.0, Dy, nxn, nyn, nzn);
  eqValue(0.0, Dz, nxn, nyn, nzn);
  // move from krylov space to physical space
  solver2phys(vectX, vectY, vectZ, vector, nxn, nyn, nzn);
  grid->lapN2NLocal(imageX, vectX, this);
  grid->lapN2NLocal(imageY, vectY, this);
  grid->lapN2NLocal(imageZ, vectZ, this);
  neg(imageX, nxn, nyn, nzn);
  neg(imageY, nxn, nyn, nzn);
  neg(imageZ, nxn, nyn, nzn);
  // grad(div(mu dot E(n + theta)) mu dot E(n + theta) = D
  MUdot(Dx, Dy, Dz, vectX, vectY, vectZ);
  grid->divN2C(divC, Dx, Dy, Dz);
  // communicate you should put BC
  // think about the Physics
  // communicateCenterBC(nxc,nyc,nzc,divC,1,1,1,1,1,1,vct);

  // communicateCenterBC(nxc, nyc, nzc, divC, 2, 2, 2, 2, 2, 2, vct, this);

  grid->gradC2N(tempX, tempY, tempZ, divC);

  // -lap(E(n +theta)) - grad(div(mu dot E(n + theta))
  sub(imageX, tempX, nxn, nyn, nzn);
  sub(imageY, tempY, nxn, nyn, nzn);
  sub(imageZ, tempZ, nxn, nyn, nzn);

  // scale delt*delt
  scale(imageX, delt * delt, nxn, nyn, nzn);
  scale(imageY, delt * delt, nxn, nyn, nzn);
  scale(imageZ, delt * delt, nxn, nyn, nzn);

  // -lap(E(n +theta)) - grad(div(mu dot E(n + theta)) + eps dot E(n + theta)
  sum(imageX, Dx, nxn, nyn, nzn);
  sum(imageY, Dy, nxn, nyn, nzn);
  sum(imageZ, Dz, nxn, nyn, nzn);
  sum(imageX, vectX, nxn, nyn, nzn);
  sum(imageY, vectY, nxn, nyn, nzn);
  sum(imageZ, vectZ, nxn, nyn, nzn);

  // boundary condition: Xleft
  if (vct->getXleft_neighbor() == MPI_PROC_NULL && bcEMfaceXleft == 0) // perfect conductor
    perfectConductorLeft(imageX, imageY, imageZ, vectX, vectY, vectZ, 0);
  // boundary condition: Xright
  if (vct->getXright_neighbor() == MPI_PROC_NULL && bcEMfaceXright == 0) // perfect conductor
    perfectConductorRight(imageX, imageY, imageZ, vectX, vectY, vectZ, 0);
  // boundary condition: Yleft
  if (vct->getYleft_neighbor() == MPI_PROC_NULL && bcEMfaceYleft == 0) // perfect conductor
    perfectConductorLeft(imageX, imageY, imageZ, vectX, vectY, vectZ, 1);
  // boundary condition: Yright
  if (vct->getYright_neighbor() == MPI_PROC_NULL && bcEMfaceYright == 0) // perfect conductor
    perfectConductorRight(imageX, imageY, imageZ, vectX, vectY, vectZ, 1);
  // boundary condition: Zleft
  if (vct->getZleft_neighbor() == MPI_PROC_NULL && bcEMfaceZleft == 0) // perfect conductor
    perfectConductorLeft(imageX, imageY, imageZ, vectX, vectY, vectZ, 2);
  // boundary condition: Zright
  if (vct->getZright_neighbor() == MPI_PROC_NULL && bcEMfaceZright == 0) // perfect conductor
    perfectConductorRight(imageX, imageY, imageZ, vectX, vectY, vectZ, 2);

  // OpenBC: apply inflow BCs to GMRes image if enabled
  if (get_col().getApplyInflowBcsEImage()){
    OpenBoundaryInflowEImage(imageX, imageY, imageZ, vectX, vectY, vectZ, nxn, nyn, nzn);
  }
  // move from physical space to krylov space
  phys2solver(im, imageX, imageY, imageZ, nxn, nyn, nzn);
}

/*! Calculate PI dot (vectX, vectY, vectZ) */
void EMfields3D::PIdot(arr3_double PIdotX, arr3_double PIdotY, arr3_double PIdotZ, const_arr3_double vectX, const_arr3_double vectY, const_arr3_double vectZ, int ns)
{
  const Grid *grid = &get_grid();
  double beta, edotb, omcx, omcy, omcz, denom;
  beta = .5 * qom[ns] * dt / c;
#pragma omp parallel for collapse(2) private(edotb, omcx, omcy, omcz, denom)
  for (int i = 1; i < nxn - 1; i++)
    for (int j = 1; j < nyn - 1; j++)
      for (int k = 1; k < nzn - 1; k++)
      {
        omcx = beta * (Bxn[i][j][k] + Bx_ext[i][j][k]);
        omcy = beta * (Byn[i][j][k] + By_ext[i][j][k]);
        omcz = beta * (Bzn[i][j][k] + Bz_ext[i][j][k]);
        edotb = vectX.get(i, j, k) * omcx + vectY.get(i, j, k) * omcy + vectZ.get(i, j, k) * omcz;
        denom = 1 / (1.0 + omcx * omcx + omcy * omcy + omcz * omcz);
        PIdotX.fetch(i, j, k) += (vectX.get(i, j, k) + (vectY.get(i, j, k) * omcz - vectZ.get(i, j, k) * omcy + edotb * omcx)) * denom;
        PIdotY.fetch(i, j, k) += (vectY.get(i, j, k) + (vectZ.get(i, j, k) * omcx - vectX.get(i, j, k) * omcz + edotb * omcy)) * denom;
        PIdotZ.fetch(i, j, k) += (vectZ.get(i, j, k) + (vectX.get(i, j, k) * omcy - vectY.get(i, j, k) * omcx + edotb * omcz)) * denom;
      }
}
/*! Calculate MU dot (vectX, vectY, vectZ) */
void EMfields3D::MUdot(arr3_double MUdotX, arr3_double MUdotY, arr3_double MUdotZ,
                       const_arr3_double vectX, const_arr3_double vectY, const_arr3_double vectZ)
{
  const Grid *grid = &get_grid();
  double beta, edotb, omcx, omcy, omcz, denom;
#pragma omp parallel for collapse(2)
  for (int i = 1; i < nxn - 1; i++)
    for (int j = 1; j < nyn - 1; j++)
      for (int k = 1; k < nzn - 1; k++)
      {
        MUdotX[i][j][k] = 0.0;
        MUdotY[i][j][k] = 0.0;
        MUdotZ[i][j][k] = 0.0;
      }
  for (int is = 0; is < ns; is++)
  {
    beta = .5 * qom[is] * dt / c;
#pragma omp parallel for collapse(2) private(edotb, omcx, omcy, omcz, denom)
    for (int i = 1; i < nxn - 1; i++)
      for (int j = 1; j < nyn - 1; j++)
        for (int k = 1; k < nzn - 1; k++)
        {
          omcx = beta * (Bxn[i][j][k] + Bx_ext[i][j][k]);
          omcy = beta * (Byn[i][j][k] + By_ext[i][j][k]);
          omcz = beta * (Bzn[i][j][k] + Bz_ext[i][j][k]);
          edotb = vectX.get(i, j, k) * omcx + vectY.get(i, j, k) * omcy + vectZ.get(i, j, k) * omcz;
          denom = FourPI / 2 * delt * dt / c * qom[is] * rhons[is][i][j][k] / (1.0 + omcx * omcx + omcy * omcy + omcz * omcz);
          MUdotX.fetch(i, j, k) += (vectX.get(i, j, k) + (vectY.get(i, j, k) * omcz - vectZ.get(i, j, k) * omcy + edotb * omcx)) * denom;
          MUdotY.fetch(i, j, k) += (vectY.get(i, j, k) + (vectZ.get(i, j, k) * omcx - vectX.get(i, j, k) * omcz + edotb * omcy)) * denom;
          MUdotZ.fetch(i, j, k) += (vectZ.get(i, j, k) + (vectX.get(i, j, k) * omcy - vectY.get(i, j, k) * omcx + edotb * omcz)) * denom;
        }
  }
}
/* Interpolation smoothing: Smoothing (vector must already have ghost cells) TO MAKE SMOOTH value as to be different from 1.0 type = 0 --> center based vector ; type = 1 --> node based vector ; */
void EMfields3D::smooth(arr3_double vector, int type)
{
  if (Smooth == 1.0)
    return;
  const VirtualTopology3D *vct = &get_vct();
  const Grid *grid = &get_grid();

  const double alpha = Smooth;
  const double beta3D = (1 - alpha) / 6.0;
  int nx, ny, nz;
  switch (type)
  {
  case (0):
    nx = grid->getNXC();
    ny = grid->getNYC();
    nz = grid->getNZC();
    break;
  case (1):
    nx = grid->getNXN();
    ny = grid->getNYN();
    nz = grid->getNZN();
    break;
  }
  // Use persistent smoothTemp buffer (node-sized, safe for both center and node)
  for (int icount = 1; icount < SmoothNiter + 1; icount++)
  {
    switch (type)
    {
    case (0):
      communicateCenterBoxStencilBC_P(nx, ny, nz, vector, 2, 2, 2, 2, 2, 2, vct, this);
      break;
    case (1):
      communicateNodeBoxStencilBC_P(nx, ny, nz, vector, 2, 2, 2, 2, 2, 2, vct, this);
      break;
    }

#pragma omp parallel for collapse(2)
    for (int i = 1; i < nx - 1; i++)
      for (int j = 1; j < ny - 1; j++)
        for (int k = 1; k < nz - 1; k++)
          smoothTemp[i][j][k] = alpha * vector[i][j][k] + beta3D * (vector[i - 1][j][k] + vector[i + 1][j][k] + vector[i][j - 1][k] + vector[i][j + 1][k] + vector[i][j][k - 1] + vector[i][j][k + 1]);

#pragma omp parallel for collapse(2)
    for (int i = 1; i < nx - 1; i++)
      for (int j = 1; j < ny - 1; j++)
        for (int k = 1; k < nz - 1; k++)
          vector[i][j][k] = smoothTemp[i][j][k];
  }
}

/* Interpolation smoothing: Smoothing (vector must already have ghost cells)
 * TO MAKE SMOOTH value as to be different from 1.0 type = 0 --> center based vector ; type = 1 --> node based vector ; */
void EMfields3D::smoothE()
{
  if (Smooth == 1.0)
    return;
  const Collective *col = &get_col();
  const VirtualTopology3D *vct = &get_vct();

  const double alpha = Smooth;
  const double beta3D = (1 - alpha) / 6.0;

  // Uses existing persistent member arrays tempX/Y/Z as temp buffers.
  // These are free at this point: smoothE is called after the Maxwell solve.
  for (int icount = 1; icount < SmoothNiter + 1; icount++)
  {
    communicateNodeBoxStencilBC(nxn, nyn, nzn, Ex, col->bcEx[0], col->bcEx[1], col->bcEx[2], col->bcEx[3], col->bcEx[4], col->bcEx[5], vct, this);
    communicateNodeBoxStencilBC(nxn, nyn, nzn, Ey, col->bcEy[0], col->bcEy[1], col->bcEy[2], col->bcEy[3], col->bcEy[4], col->bcEy[5], vct, this);
    communicateNodeBoxStencilBC(nxn, nyn, nzn, Ez, col->bcEz[0], col->bcEz[1], col->bcEz[2], col->bcEz[3], col->bcEz[4], col->bcEz[5], vct, this);

// Fused smooth: compute all 3 components into temp buffers in a single pass
#pragma omp parallel for collapse(2)
    for (int i = 1; i < nxn - 1; i++)
      for (int j = 1; j < nyn - 1; j++)
        for (int k = 1; k < nzn - 1; k++)
        {
          tempX[i][j][k] = alpha * Ex[i][j][k] + beta3D * (Ex[i - 1][j][k] + Ex[i + 1][j][k] + Ex[i][j - 1][k] + Ex[i][j + 1][k] + Ex[i][j][k - 1] + Ex[i][j][k + 1]);
          tempY[i][j][k] = alpha * Ey[i][j][k] + beta3D * (Ey[i - 1][j][k] + Ey[i + 1][j][k] + Ey[i][j - 1][k] + Ey[i][j + 1][k] + Ey[i][j][k - 1] + Ey[i][j][k + 1]);
          tempZ[i][j][k] = alpha * Ez[i][j][k] + beta3D * (Ez[i - 1][j][k] + Ez[i + 1][j][k] + Ez[i][j - 1][k] + Ez[i][j + 1][k] + Ez[i][j][k - 1] + Ez[i][j][k + 1]);
        }

// Copy all 3 components back in a single pass
#pragma omp parallel for collapse(2)
    for (int i = 1; i < nxn - 1; i++)
      for (int j = 1; j < nyn - 1; j++)
        for (int k = 1; k < nzn - 1; k++)
        {
          Ex[i][j][k] = tempX[i][j][k];
          Ey[i][j][k] = tempY[i][j][k];
          Ez[i][j][k] = tempZ[i][j][k];
        }
  }
}

/* SPECIES: Interpolation smoothing TO MAKE SMOOTH value as to be different from 1.0 type = 0 --> center based vector type = 1 --> node based vector */
void EMfields3D::smooth(double value, arr4_double vector, int is, int type)
{
  eprintf("Smoothing for Species not implemented in 3D");
}

/*! fix the B boundary when running gem , This assume non-periodic condition on Y dimension*/
void EMfields3D::fixBcGEM()
{
  const VirtualTopology3D *vct = &get_vct();
  const Grid *grid = &get_grid();

  if (vct->getYright_neighbor() == MPI_PROC_NULL)
  {
    for (int i = 0; i < nxc; i++)
      for (int k = 0; k < nzc; k++)
      {
        Bxc[i][nyc - 1][k] = B0x * tanh((grid->getYC(i, nyc - 1, k) - Ly / 2) / delta);
        Bxc[i][nyc - 2][k] = Bxc[i][nyc - 1][k];
        Bxc[i][nyc - 3][k] = Bxc[i][nyc - 1][k];
        Byc[i][nyc - 1][k] = B0y;
        Bzc[i][nyc - 1][k] = B0z;
        Bzc[i][nyc - 2][k] = B0z;
        Bzc[i][nyc - 3][k] = B0z;
      }
  }
  if (vct->getYleft_neighbor() == MPI_PROC_NULL)
  {
    for (int i = 0; i < nxc; i++)
      for (int k = 0; k < nzc; k++)
      {
        Bxc[i][0][k] = B0x * tanh((grid->getYC(i, 0, k) - Ly / 2) / delta);
        Bxc[i][1][k] = Bxc[i][0][k];
        Bxc[i][2][k] = Bxc[i][0][k];
        Byc[i][0][k] = B0y;
        Bzc[i][0][k] = B0z;
        Bzc[i][1][k] = B0z;
        Bzc[i][2][k] = B0z;
      }
  }
}

void EMfields3D::fixBnGEM()
{
  const VirtualTopology3D *vct = &get_vct();
  const Grid *grid = &get_grid();

  if (vct->getYright_neighbor() == MPI_PROC_NULL)
  {
    for (int i = 0; i < nxn; i++)
      for (int k = 0; k < nzn; k++)
      {
        Bxn[i][nyn - 1][k] = B0x * tanh((grid->getYC(i, nyc - 1, k) - Ly / 2) / delta);
        Bxn[i][nyn - 2][k] = Bxn[i][nyn - 1][k];
        Bxn[i][nyn - 3][k] = Bxn[i][nyn - 1][k];
        Byn[i][nyn - 1][k] = B0y;
        Bzn[i][nyn - 1][k] = B0z;
        Bzn[i][nyn - 2][k] = B0z;
        Bzn[i][nyn - 3][k] = B0z;
      }
  }
  if (vct->getYleft_neighbor() == MPI_PROC_NULL)
  {
    for (int i = 0; i < nxn; i++)
      for (int k = 0; k < nzn; k++)
      {
        Bxn[i][0][k] = B0x * tanh((grid->getYC(i, 0, k) - Ly / 2) / delta);
        Bxn[i][1][k] = Bxn[i][0][k];
        Bxn[i][2][k] = Bxn[i][0][k];
        Byn[i][0][k] = B0y;
        Bzn[i][0][k] = B0z;
        Bzn[i][1][k] = B0z;
        Bzn[i][2][k] = B0z;
      }
  }
}

/*! fix the B boundary when running forcefree */
void EMfields3D::fixBforcefree()
{
  const VirtualTopology3D *vct = &get_vct();
  const Grid *grid = &get_grid();

  if (vct->getYright_neighbor() == MPI_PROC_NULL)
  {
    for (int i = 0; i < nxc; i++)
      for (int k = 0; k < nzc; k++)
      {
        Bxc[i][nyc - 1][k] = B0x * tanh((grid->getYC(i, nyc - 1, k) - Ly / 2) / delta);
        Byc[i][nyc - 1][k] = B0y;
        Bzc[i][nyc - 1][k] = B0z / cosh((grid->getYC(i, nyc - 1, k) - Ly / 2) / delta);
        ;
        Bzc[i][nyc - 2][k] = B0z / cosh((grid->getYC(i, nyc - 2, k) - Ly / 2) / delta);
        ;
        Bzc[i][nyc - 3][k] = B0z / cosh((grid->getYC(i, nyc - 3, k) - Ly / 2) / delta);
      }
  }
  if (vct->getYleft_neighbor() == MPI_PROC_NULL)
  {
    for (int i = 0; i < nxc; i++)
      for (int k = 0; k < nzc; k++)
      {
        Bxc[i][0][k] = B0x * tanh((grid->getYC(i, 0, k) - Ly / 2) / delta);
        Byc[i][0][k] = B0y;
        Bzc[i][0][k] = B0z / cosh((grid->getYC(i, 0, k) - Ly / 2) / delta);
        Bzc[i][1][k] = B0z / cosh((grid->getYC(i, 1, k) - Ly / 2) / delta);
        Bzc[i][2][k] = B0z / cosh((grid->getYC(i, 2, k) - Ly / 2) / delta);
      }
  }
}

// This method assumes mirror boundary conditions;
// we therefore need to double the density on the boundary
// nodes to incorporate the mirror particles from the mirror
// cell just outside the domain.
//
/*! adjust densities on boundaries that are not periodic */
void EMfields3D::adjustNonPeriodicDensities(int is)
{
  const VirtualTopology3D *vct = &get_vct();
  if (vct->getXleft_neighbor_P() == MPI_PROC_NULL)
  {
    for (int i = 1; i < nyn - 1; i++)
      for (int k = 1; k < nzn - 1; k++)
      {
        rhons[is][1][i][k] *= 2;
        Jxs[is][1][i][k] *= 2;
        Jys[is][1][i][k] *= 2;
        Jzs[is][1][i][k] *= 2;
        pXXsn[is][1][i][k] *= 2;
        pXYsn[is][1][i][k] *= 2;
        pXZsn[is][1][i][k] *= 2;
        pYYsn[is][1][i][k] *= 2;
        pYZsn[is][1][i][k] *= 2;
        pZZsn[is][1][i][k] *= 2;
      }
  }
  if (vct->getYleft_neighbor_P() == MPI_PROC_NULL)
  {
    for (int i = 1; i < nxn - 1; i++)
      for (int k = 1; k < nzn - 1; k++)
      {
        rhons[is][i][1][k] *= 2;
        Jxs[is][i][1][k] *= 2;
        Jys[is][i][1][k] *= 2;
        Jzs[is][i][1][k] *= 2;
        pXXsn[is][i][1][k] *= 2;
        pXYsn[is][i][1][k] *= 2;
        pXZsn[is][i][1][k] *= 2;
        pYYsn[is][i][1][k] *= 2;
        pYZsn[is][i][1][k] *= 2;
        pZZsn[is][i][1][k] *= 2;
      }
  }
  if (vct->getZleft_neighbor_P() == MPI_PROC_NULL)
  {
    for (int i = 1; i < nxn - 1; i++)
      for (int j = 1; j < nyn - 1; j++)
      {
        rhons[is][i][j][1] *= 2;
        Jxs[is][i][j][1] *= 2;
        Jys[is][i][j][1] *= 2;
        Jzs[is][i][j][1] *= 2;
        pXXsn[is][i][j][1] *= 2;
        pXYsn[is][i][j][1] *= 2;
        pXZsn[is][i][j][1] *= 2;
        pYYsn[is][i][j][1] *= 2;
        pYZsn[is][i][j][1] *= 2;
        pZZsn[is][i][j][1] *= 2;
      }
  }
  if (vct->getXright_neighbor_P() == MPI_PROC_NULL)
  {
    for (int i = 1; i < nyn - 1; i++)
      for (int k = 1; k < nzn - 1; k++)
      {
        rhons[is][nxn - 2][i][k] *= 2;
        Jxs[is][nxn - 2][i][k] *= 2;
        Jys[is][nxn - 2][i][k] *= 2;
        Jzs[is][nxn - 2][i][k] *= 2;
        pXXsn[is][nxn - 2][i][k] *= 2;
        pXYsn[is][nxn - 2][i][k] *= 2;
        pXZsn[is][nxn - 2][i][k] *= 2;
        pYYsn[is][nxn - 2][i][k] *= 2;
        pYZsn[is][nxn - 2][i][k] *= 2;
        pZZsn[is][nxn - 2][i][k] *= 2;
      }
  }
  if (vct->getYright_neighbor_P() == MPI_PROC_NULL)
  {
    for (int i = 1; i < nxn - 1; i++)
      for (int k = 1; k < nzn - 1; k++)
      {
        rhons[is][i][nyn - 2][k] *= 2;
        Jxs[is][i][nyn - 2][k] *= 2;
        Jys[is][i][nyn - 2][k] *= 2;
        Jzs[is][i][nyn - 2][k] *= 2;
        pXXsn[is][i][nyn - 2][k] *= 2;
        pXYsn[is][i][nyn - 2][k] *= 2;
        pXZsn[is][i][nyn - 2][k] *= 2;
        pYYsn[is][i][nyn - 2][k] *= 2;
        pYZsn[is][i][nyn - 2][k] *= 2;
        pZZsn[is][i][nyn - 2][k] *= 2;
      }
  }
  if (vct->getZright_neighbor_P() == MPI_PROC_NULL)
  {
    for (int i = 1; i < nxn - 1; i++)
      for (int j = 1; j < nyn - 1; j++)
      {
        rhons[is][i][j][nzn - 2] *= 2;
        Jxs[is][i][j][nzn - 2] *= 2;
        Jys[is][i][j][nzn - 2] *= 2;
        Jzs[is][i][j][nzn - 2] *= 2;
        pXXsn[is][i][j][nzn - 2] *= 2;
        pXYsn[is][i][j][nzn - 2] *= 2;
        pXZsn[is][i][j][nzn - 2] *= 2;
        pYYsn[is][i][j][nzn - 2] *= 2;
        pYZsn[is][i][j][nzn - 2] *= 2;
        pZZsn[is][i][j][nzn - 2] *= 2;
      }
  }
}

void EMfields3D::ConstantChargeOpenBCv2()
{
  const VirtualTopology3D *vct = &get_vct();
  const Grid *grid = &get_grid();

  double ff;

  int nx = grid->getNXN();
  int ny = grid->getNYN();
  int nz = grid->getNZN();

  for (int is = 0; is < ns; is++)
  {

    ff = qom[is] / fabs(qom[is]);

    if (vct->getXleft_neighbor() == MPI_PROC_NULL && bcEMfaceXleft == 2)
    {
      for (int j = 0; j < ny; j++)
        for (int k = 0; k < nz; k++)
        {
          rhons[is][0][j][k] = rhons[is][4][j][k];
          rhons[is][1][j][k] = rhons[is][4][j][k];
          rhons[is][2][j][k] = rhons[is][4][j][k];
          rhons[is][3][j][k] = rhons[is][4][j][k];
        }
    }

    if (vct->getXright_neighbor() == MPI_PROC_NULL && bcEMfaceXright == 2)
    {
      for (int j = 0; j < ny; j++)
        for (int k = 0; k < nz; k++)
        {
          rhons[is][nx - 4][j][k] = rhons[is][nx - 5][j][k];
          rhons[is][nx - 3][j][k] = rhons[is][nx - 5][j][k];
          rhons[is][nx - 2][j][k] = rhons[is][nx - 5][j][k];
          rhons[is][nx - 1][j][k] = rhons[is][nx - 5][j][k];
        }
    }

    if (vct->getYleft_neighbor() == MPI_PROC_NULL && bcEMfaceYleft == 2)
    {
      for (int i = 0; i < nx; i++)
        for (int k = 0; k < nz; k++)
        {
          rhons[is][i][0][k] = rhons[is][i][4][k];
          rhons[is][i][1][k] = rhons[is][i][4][k];
          rhons[is][i][2][k] = rhons[is][i][4][k];
          rhons[is][i][3][k] = rhons[is][i][4][k];
        }
    }

    if (vct->getYright_neighbor() == MPI_PROC_NULL && bcEMfaceYright == 2)
    {
      for (int i = 0; i < nx; i++)
        for (int k = 0; k < nz; k++)
        {
          rhons[is][i][ny - 4][k] = rhons[is][i][ny - 5][k];
          rhons[is][i][ny - 3][k] = rhons[is][i][ny - 5][k];
          rhons[is][i][ny - 2][k] = rhons[is][i][ny - 5][k];
          rhons[is][i][ny - 1][k] = rhons[is][i][ny - 5][k];
        }
    }

    if (vct->getZleft_neighbor() == MPI_PROC_NULL && bcEMfaceZleft == 2)
    {
      for (int i = 0; i < nx; i++)
        for (int j = 0; j < ny; j++)
        {
          rhons[is][i][j][0] = rhons[is][i][j][4];
          rhons[is][i][j][1] = rhons[is][i][j][4];
          rhons[is][i][j][2] = rhons[is][i][j][4];
          rhons[is][i][j][3] = rhons[is][i][j][4];
        }
    }

    if (vct->getZright_neighbor() == MPI_PROC_NULL && bcEMfaceZright == 2)
    {
      for (int i = 0; i < nx; i++)
        for (int j = 0; j < ny; j++)
        {
          rhons[is][i][j][nz - 4] = rhons[is][i][j][nz - 5];
          rhons[is][i][j][nz - 3] = rhons[is][i][j][nz - 5];
          rhons[is][i][j][nz - 2] = rhons[is][i][j][nz - 5];
          rhons[is][i][j][nz - 1] = rhons[is][i][j][nz - 5];
        }
    }
  }
}

void EMfields3D::ConstantChargeOpenBC()
{
  const VirtualTopology3D *vct = &get_vct();
  const Grid *grid = &get_grid();

  double ff;

  int nx = grid->getNXN();
  int ny = grid->getNYN();
  int nz = grid->getNZN();

  for (int is = 0; is < ns; is++)
  {

    ff = qom[is] / fabs(qom[is]);

    if (vct->getXleft_neighbor() == MPI_PROC_NULL && (bcEMfaceXleft == 2))
    {
      for (int j = 0; j < ny; j++)
        for (int k = 0; k < nz; k++)
        {
          rhons[is][0][j][k] = ff * rhoINIT[is] / FourPI;
          rhons[is][1][j][k] = ff * rhoINIT[is] / FourPI;
          rhons[is][2][j][k] = ff * rhoINIT[is] / FourPI;
          rhons[is][3][j][k] = ff * rhoINIT[is] / FourPI;
        }
    }

    if (vct->getXright_neighbor() == MPI_PROC_NULL && (bcEMfaceXright == 2))
    {
      for (int j = 0; j < ny; j++)
        for (int k = 0; k < nz; k++)
        {
          rhons[is][nx - 4][j][k] = ff * rhoINIT[is] / FourPI;
          rhons[is][nx - 3][j][k] = ff * rhoINIT[is] / FourPI;
          rhons[is][nx - 2][j][k] = ff * rhoINIT[is] / FourPI;
          rhons[is][nx - 1][j][k] = ff * rhoINIT[is] / FourPI;
        }
    }

    if (vct->getYleft_neighbor() == MPI_PROC_NULL && (bcEMfaceYleft == 2))
    {
      for (int i = 0; i < nx; i++)
        for (int k = 0; k < nz; k++)
        {
          rhons[is][i][0][k] = ff * rhoINIT[is] / FourPI;
          rhons[is][i][1][k] = ff * rhoINIT[is] / FourPI;
          rhons[is][i][2][k] = ff * rhoINIT[is] / FourPI;
          rhons[is][i][3][k] = ff * rhoINIT[is] / FourPI;
        }
    }

    if (vct->getYright_neighbor() == MPI_PROC_NULL && (bcEMfaceYright == 2))
    {
      for (int i = 0; i < nx; i++)
        for (int k = 0; k < nz; k++)
        {
          rhons[is][i][ny - 4][k] = ff * rhoINIT[is] / FourPI;
          rhons[is][i][ny - 3][k] = ff * rhoINIT[is] / FourPI;
          rhons[is][i][ny - 2][k] = ff * rhoINIT[is] / FourPI;
          rhons[is][i][ny - 1][k] = ff * rhoINIT[is] / FourPI;
        }
    }

    if (vct->getZleft_neighbor() == MPI_PROC_NULL && (bcEMfaceZleft == 2))
    {
      for (int i = 0; i < nx; i++)
        for (int j = 0; j < ny; j++)
        {
          rhons[is][i][j][0] = ff * rhoINIT[is] / FourPI;
          rhons[is][i][j][1] = ff * rhoINIT[is] / FourPI;
          rhons[is][i][j][2] = ff * rhoINIT[is] / FourPI;
          rhons[is][i][j][3] = ff * rhoINIT[is] / FourPI;
        }
    }

    if (vct->getZright_neighbor() == MPI_PROC_NULL && (bcEMfaceZright == 2))
    {
      for (int i = 0; i < nx; i++)
        for (int j = 0; j < ny; j++)
        {
          rhons[is][i][j][nz - 4] = ff * rhoINIT[is] / FourPI;
          rhons[is][i][j][nz - 3] = ff * rhoINIT[is] / FourPI;
          rhons[is][i][j][nz - 2] = ff * rhoINIT[is] / FourPI;
          rhons[is][i][j][nz - 1] = ff * rhoINIT[is] / FourPI;
        }
    }
  }
}

void EMfields3D::ConstantChargePlanet(double R,
                                      double x_center, double y_center, double z_center)
{
  const Grid *grid = &get_grid();

  double xd;
  double yd;
  double zd;
  double ff;

  for (int is = 0; is < ns; is++)
  {

    ff = qom[is] / fabs(qom[is]);

    for (int i = 1; i < nxn; i++)
    {
      for (int j = 1; j < nyn; j++)
      {
        for (int k = 1; k < nzn; k++)
        {

          xd = grid->getXN(i, j, k) - x_center;
          yd = grid->getYN(i, j, k) - y_center;
          zd = grid->getZN(i, j, k) - z_center;

          if ((xd * xd + yd * yd + zd * zd) <= R * R)
          {
            rhons[is][i][j][k] = ff * rhoINIT[is] / FourPI;
          }
        }
      }
    }
  }
}

void EMfields3D::ConstantChargePlanet2DPlaneXZ(double R, double x_center, double z_center)
{
  const Grid *grid = &get_grid();
  // if (get_vct().getCartesian_rank() == 0)
  // cout << "*** Constant Charge 2D Planet ***" << endl;

  assert_eq(nyn, 4);
  double xd;
  double zd;

  for (int is = 0; is < ns; is++)
  {
    const double sign_q = qom[is] / (fabs(qom[is]));
    for (int i = 1; i < nxn; i++)
      for (int k = 1; k < nzn; k++)
      {

        xd = grid->getXN(i, 1, k) - x_center;
        zd = grid->getZN(i, 1, k) - z_center;

        if ((xd * xd + zd * zd) <= R * R)
        {
          rhons[is][i][1][k] = sign_q * rhoINIT[is] / FourPI;
          rhons[is][i][2][k] = sign_q * rhoINIT[is] / FourPI;
        }
      }
  }
}

/*! Populate the field data used to push particles */
//
//
//
void EMfields3D::set_fieldForPcls()
{
#pragma omp parallel for collapse(3)
  for (int i = 0; i < nxn; i++)
    for (int j = 0; j < nyn; j++)
      for (int k = 0; k < nzn; k++)
      {
        fieldForPcls[i][j][k][0] = (pfloat)(Bxn[i][j][k] + Bx_ext[i][j][k]);
        fieldForPcls[i][j][k][1] = (pfloat)(Byn[i][j][k] + By_ext[i][j][k]);
        fieldForPcls[i][j][k][2] = (pfloat)(Bzn[i][j][k] + Bz_ext[i][j][k]);
        fieldForPcls[i][j][k][0 + DFIELD_3or4] = (pfloat)Ex[i][j][k];
        fieldForPcls[i][j][k][1 + DFIELD_3or4] = (pfloat)Ey[i][j][k];
        fieldForPcls[i][j][k][2 + DFIELD_3or4] = (pfloat)Ez[i][j][k];
      }
}

/**
 * @brief field for a cell, optimized for GPU memory access
 * @details each cell has 6 fields on 8 grid points
 *        for this GPU optimized buffer, store data from 4 grid points in this cell
 *        which can be used by it self and the next cell
 *        the overhead is smaller than 3 times of the original buffer
 *
 * @param fieldForPclsOnCenter field buffer for particles, (nxn-1)*(nyn-1)*(nzn)*4*6
 */
void EMfields3D::set_fieldForPclsToCenter(cudaFieldType *fieldForPclsOnCenter)
{
#pragma omp parallel for collapse(3)
  for (int i = 0; i < nxn - 1; i++)
    for (int j = 0; j < nyn - 1; j++)
      for (int k = 0; k < nzn; k++) // additional cell for the head
      {
        const auto cellIndex = (i * (nyn - 1) + j) * nzn + k;
        // grid point (i, j, k)
        fieldForPclsOnCenter[cellIndex * 4 * 6 + 0 * 6 + 0] = (cudaFieldType)(Bxn[i][j][k] + Bx_ext[i][j][k]);
        fieldForPclsOnCenter[cellIndex * 4 * 6 + 0 * 6 + 1] = (cudaFieldType)(Byn[i][j][k] + By_ext[i][j][k]);
        fieldForPclsOnCenter[cellIndex * 4 * 6 + 0 * 6 + 2] = (cudaFieldType)(Bzn[i][j][k] + Bz_ext[i][j][k]);
        fieldForPclsOnCenter[cellIndex * 4 * 6 + 0 * 6 + 3] = (cudaFieldType)Ex[i][j][k];
        fieldForPclsOnCenter[cellIndex * 4 * 6 + 0 * 6 + 4] = (cudaFieldType)Ey[i][j][k];
        fieldForPclsOnCenter[cellIndex * 4 * 6 + 0 * 6 + 5] = (cudaFieldType)Ez[i][j][k];

        // grid point (i+1, j, k)
        fieldForPclsOnCenter[cellIndex * 4 * 6 + 1 * 6 + 0] = (cudaFieldType)(Bxn[i + 1][j][k] + Bx_ext[i + 1][j][k]);
        fieldForPclsOnCenter[cellIndex * 4 * 6 + 1 * 6 + 1] = (cudaFieldType)(Byn[i + 1][j][k] + By_ext[i + 1][j][k]);
        fieldForPclsOnCenter[cellIndex * 4 * 6 + 1 * 6 + 2] = (cudaFieldType)(Bzn[i + 1][j][k] + Bz_ext[i + 1][j][k]);
        fieldForPclsOnCenter[cellIndex * 4 * 6 + 1 * 6 + 3] = (cudaFieldType)Ex[i + 1][j][k];
        fieldForPclsOnCenter[cellIndex * 4 * 6 + 1 * 6 + 4] = (cudaFieldType)Ey[i + 1][j][k];
        fieldForPclsOnCenter[cellIndex * 4 * 6 + 1 * 6 + 5] = (cudaFieldType)Ez[i + 1][j][k];

        // grid point (i+1, j+1, k)
        fieldForPclsOnCenter[cellIndex * 4 * 6 + 2 * 6 + 0] = (cudaFieldType)(Bxn[i + 1][j + 1][k] + Bx_ext[i + 1][j + 1][k]);
        fieldForPclsOnCenter[cellIndex * 4 * 6 + 2 * 6 + 1] = (cudaFieldType)(Byn[i + 1][j + 1][k] + By_ext[i + 1][j + 1][k]);
        fieldForPclsOnCenter[cellIndex * 4 * 6 + 2 * 6 + 2] = (cudaFieldType)(Bzn[i + 1][j + 1][k] + Bz_ext[i + 1][j + 1][k]);
        fieldForPclsOnCenter[cellIndex * 4 * 6 + 2 * 6 + 3] = (cudaFieldType)Ex[i + 1][j + 1][k];
        fieldForPclsOnCenter[cellIndex * 4 * 6 + 2 * 6 + 4] = (cudaFieldType)Ey[i + 1][j + 1][k];
        fieldForPclsOnCenter[cellIndex * 4 * 6 + 2 * 6 + 5] = (cudaFieldType)Ez[i + 1][j + 1][k];

        // grid point (i, j+1, k)
        fieldForPclsOnCenter[cellIndex * 4 * 6 + 3 * 6 + 0] = (cudaFieldType)(Bxn[i][j + 1][k] + Bx_ext[i][j + 1][k]);
        fieldForPclsOnCenter[cellIndex * 4 * 6 + 3 * 6 + 1] = (cudaFieldType)(Byn[i][j + 1][k] + By_ext[i][j + 1][k]);
        fieldForPclsOnCenter[cellIndex * 4 * 6 + 3 * 6 + 2] = (cudaFieldType)(Bzn[i][j + 1][k] + Bz_ext[i][j + 1][k]);
        fieldForPclsOnCenter[cellIndex * 4 * 6 + 3 * 6 + 3] = (cudaFieldType)Ex[i][j + 1][k];
        fieldForPclsOnCenter[cellIndex * 4 * 6 + 3 * 6 + 4] = (cudaFieldType)Ey[i][j + 1][k];
        fieldForPclsOnCenter[cellIndex * 4 * 6 + 3 * 6 + 5] = (cudaFieldType)Ez[i][j + 1][k];
      }
}

/*! Calculate Magnetic field with the implicit solver: calculate B defined on nodes With E(n+ theta) computed, the magnetic field is evaluated from Faraday's law */
void EMfields3D::calculateB(int cycle)
{
  const Collective *col = &get_col();
  const VirtualTopology3D *vct = &get_vct();
  const Grid *grid = &get_grid();

  if (vct->getCartesian_rank() == 0)
    cout << "*** B CALCULATION [CPU] ***" << endl;

  // calculate the curl of Eth
  grid->curlN2C(tempXC, tempYC, tempZC, Exth, Eyth, Ezth);

  // update the magnetic field: B^{n+1} = B^n - c*dt * curl(E^{n+theta})
  addscale(-c * dt, 1, Bxc, tempXC, nxc, nyc, nzc);
  addscale(-c * dt, 1, Byc, tempYC, nxc, nyc, nzc);
  addscale(-c * dt, 1, Bzc, tempZC, nxc, nyc, nzc);

  // communicate ghost cells for center-based B
  communicateCenterBC(nxc, nyc, nzc, Bxc, col->bcBx[0], col->bcBx[1], col->bcBx[2], col->bcBx[3], col->bcBx[4], col->bcBx[5], vct, this);
  communicateCenterBC(nxc, nyc, nzc, Byc, col->bcBy[0], col->bcBy[1], col->bcBy[2], col->bcBy[3], col->bcBy[4], col->bcBy[5], vct, this);
  communicateCenterBC(nxc, nyc, nzc, Bzc, col->bcBz[0], col->bcBz[1], col->bcBz[2], col->bcBz[3], col->bcBz[4], col->bcBz[5], vct, this);

  // open boundary conditions on center-based B
  OpenBoundaryInflowB(Bxc, Byc, Bzc, nxc, nyc, nzc);

  // case-specific fixes on center-based B
  const string &simCase = get_col().getCase();
  if (simCase == "GEM" || simCase == "GEMnoPert" || simCase == "GEMDoubleHarris")
    fixBcGEM();

  // interpolate center-to-node
  grid->interpC2N(Bxn, Bxc);
  grid->interpC2N(Byn, Byc);
  grid->interpC2N(Bzn, Bzc);

  // communicate ghost cells for node-based B
  communicateNodeBC(nxn, nyn, nzn, Bxn, col->bcBx[0], col->bcBx[1], col->bcBx[2], col->bcBx[3], col->bcBx[4], col->bcBx[5], vct, this);
  communicateNodeBC(nxn, nyn, nzn, Byn, col->bcBy[0], col->bcBy[1], col->bcBy[2], col->bcBy[3], col->bcBy[4], col->bcBy[5], vct, this);
  communicateNodeBC(nxn, nyn, nzn, Bzn, col->bcBz[0], col->bcBz[1], col->bcBz[2], col->bcBz[3], col->bcBz[4], col->bcBz[5], vct, this);

  // case-specific fixes on node-based B
  if (simCase == "ForceFree")
    fixBforcefree();
  if (simCase == "GEM" || simCase == "GEMnoPert" || simCase == "GEMDoubleHarris")
    fixBnGEM();

  // divergence cleaning: laplacian(PSI) = div(B), B = B - grad(PSI)
  if (divBCorrection && cycle % divBCorrectionCycle == 0)
    applyDivBCleaning();
}

/*! Apply divergence cleaning on B: solve laplacian(PSI) = div(B), then correct B = B - grad(PSI) on boundary layers */
void EMfields3D::applyDivBCleaning()
{
  const Collective *col = &get_col();
  const VirtualTopology3D *vct = &get_vct();
  const Grid *grid = &get_grid();

  const int nPoissonKrylov = (nxc - 2) * (nyc - 2) * (nzc - 2);

  // zero work arrays
  eqValue(0.0, divBwork, nxc, nyc, nzc);
  eqValue(0.0, gradPSIX, nxn, nyn, nzn);
  eqValue(0.0, gradPSIY, nxn, nyn, nzn);
  eqValue(0.0, gradPSIZ, nxn, nyn, nzn);
  eqValue(0.0, xkrylovPoisson_B, nPoissonKrylov);

  // compute div(B) on centers
  grid->divN2C(divBwork, Bxn, Byn, Bzn);

  // move RHS to Krylov space
  phys2solver(bkrylovPoisson_B, divBwork, nxc, nyc, nzc);

  // solve laplacian(PSI) = div(B) using GMRES
  if (vct->getCartesian_rank() == 0)
    cout << "*** DIVERGENCE CLEANING div(B)=0 using GMRes [CPU] ***" << endl;
  GMRES(&Field::PoissonImage, xkrylovPoisson_B, nPoissonKrylov, bkrylovPoisson_B, 20, 200, GMREStol, this);

  // solution back to physical space
  solver2phys(PSI, xkrylovPoisson_B, nxc, nyc, nzc);
  communicateCenterBC(nxc, nyc, nzc, PSI, 2, 2, 2, 2, 2, 2, vct, this);

  // compute gradient of PSI
  grid->gradC2N(gradPSIX, gradPSIY, gradPSIZ, PSI);

  // correct B on nodes in the boundary layers
  // Xleft
  if (vct->getXleft_neighbor() == MPI_PROC_NULL && bcEMfaceXleft == 2)
  {
    for (int i = 0; i < n_layers_sal; i++)
      for (int j = 0; j < nyn; j++)
        for (int k = 0; k < nzn; k++)
        {
          Bxn.fetch(i, j, k) -= gradPSIX.get(i, j, k);
          Byn.fetch(i, j, k) -= gradPSIY.get(i, j, k);
          Bzn.fetch(i, j, k) -= gradPSIZ.get(i, j, k);
        }
  }
  // Xright
  if (vct->getXright_neighbor() == MPI_PROC_NULL && bcEMfaceXright == 2)
  {
    for (int i = nxn - n_layers_sal; i < nxn; i++)
      for (int j = 0; j < nyn; j++)
        for (int k = 0; k < nzn; k++)
        {
          Bxn.fetch(i, j, k) -= gradPSIX.get(i, j, k);
          Byn.fetch(i, j, k) -= gradPSIY.get(i, j, k);
          Bzn.fetch(i, j, k) -= gradPSIZ.get(i, j, k);
        }
  }
  // Yleft
  if (vct->getYleft_neighbor() == MPI_PROC_NULL && bcEMfaceYleft == 2)
  {
    for (int i = 0; i < nxn; i++)
      for (int j = 0; j < n_layers_sal; j++)
        for (int k = 0; k < nzn; k++)
        {
          Bxn.fetch(i, j, k) -= gradPSIX.get(i, j, k);
          Byn.fetch(i, j, k) -= gradPSIY.get(i, j, k);
          Bzn.fetch(i, j, k) -= gradPSIZ.get(i, j, k);
        }
  }
  // Yright
  if (vct->getYright_neighbor() == MPI_PROC_NULL && bcEMfaceYright == 2)
  {
    for (int i = 0; i < nxn; i++)
      for (int j = nyn - n_layers_sal; j < nyn; j++)
        for (int k = 0; k < nzn; k++)
        {
          Bxn.fetch(i, j, k) -= gradPSIX.get(i, j, k);
          Byn.fetch(i, j, k) -= gradPSIY.get(i, j, k);
          Bzn.fetch(i, j, k) -= gradPSIZ.get(i, j, k);
        }
  }
  // Zleft
  if (vct->getZleft_neighbor() == MPI_PROC_NULL && bcEMfaceZleft == 2)
  {
    for (int i = 0; i < nxn; i++)
      for (int j = 0; j < nyn; j++)
        for (int k = 0; k < n_layers_sal; k++)
        {
          Bxn.fetch(i, j, k) -= gradPSIX.get(i, j, k);
          Byn.fetch(i, j, k) -= gradPSIY.get(i, j, k);
          Bzn.fetch(i, j, k) -= gradPSIZ.get(i, j, k);
        }
  }
  // Zright
  if (vct->getZright_neighbor() == MPI_PROC_NULL && bcEMfaceZright == 2)
  {
    for (int i = 0; i < nxn; i++)
      for (int j = 0; j < nyn; j++)
        for (int k = nzn - n_layers_sal; k < nzn; k++)
        {
          Bxn.fetch(i, j, k) -= gradPSIX.get(i, j, k);
          Byn.fetch(i, j, k) -= gradPSIY.get(i, j, k);
          Bzn.fetch(i, j, k) -= gradPSIZ.get(i, j, k);
        }
  }

  // communicate ghost cells for corrected node-based B
  communicateNodeBC(nxn, nyn, nzn, Bxn, col->bcBx[0], col->bcBx[1], col->bcBx[2], col->bcBx[3], col->bcBx[4], col->bcBx[5], vct, this);
  communicateNodeBC(nxn, nyn, nzn, Byn, col->bcBy[0], col->bcBy[1], col->bcBy[2], col->bcBy[3], col->bcBy[4], col->bcBy[5], vct, this);
  communicateNodeBC(nxn, nyn, nzn, Bzn, col->bcBz[0], col->bcBz[1], col->bcBz[2], col->bcBz[3], col->bcBz[4], col->bcBz[5], vct, this);

  // recompute center-based B from corrected node-based B
  grid->interpN2C(Bxc, Bxn);
  grid->interpN2C(Byc, Byn);
  grid->interpN2C(Bzc, Bzn);

  // communicate ghost cells for corrected center-based B
  communicateCenterBC(nxc, nyc, nzc, Bxc, col->bcBx[0], col->bcBx[1], col->bcBx[2], col->bcBx[3], col->bcBx[4], col->bcBx[5], vct, this);
  communicateCenterBC(nxc, nyc, nzc, Byc, col->bcBy[0], col->bcBy[1], col->bcBy[2], col->bcBy[3], col->bcBy[4], col->bcBy[5], vct, this);
  communicateCenterBC(nxc, nyc, nzc, Bzc, col->bcBz[0], col->bcBz[1], col->bcBz[2], col->bcBz[3], col->bcBz[4], col->bcBz[5], vct, this);
}

/*!Add a periodic perturbation in rho exp i(kx - \omega t); deltaBoB is the ratio (Delta B / B0) * */
void EMfields3D::AddPerturbationRho(double deltaBoB, double kx, double ky, double Bx_mod, double By_mod, double Bz_mod, double ne_mod, double ne_phase, double ni_mod, double ni_phase, double B0, Grid *grid)
{

  double alpha;
  alpha = deltaBoB * B0 / sqrt(Bx_mod * Bx_mod + By_mod * By_mod + Bz_mod * Bz_mod);

  ne_mod *= alpha;
  ni_mod *= alpha;
  // cout<<" ne="<<ne_mod<<" ni="<<ni_mod<<" alpha="<<alpha<<endl;
  for (int i = 0; i < nxn; i++)
    for (int j = 0; j < nyn; j++)
    {
      rhons[0][i][j][0] += ne_mod * cos(kx * grid->getXN(i, j, 0) + ky * grid->getYN(i, j, 0) + ne_phase);
      rhons[1][i][j][0] += ni_mod * cos(kx * grid->getXN(i, j, 0) + ky * grid->getYN(i, j, 0) + ni_phase);
    }

  for (int is = 0; is < ns; is++)
    grid->interpN2C(rhocs, is, rhons);
}

/*!Add a periodic perturbation exp i(kx - \omega t); deltaBoB is the ratio (Delta B / B0) * */
void EMfields3D::AddPerturbation(double deltaBoB, double kx, double ky, double Ex_mod, double Ex_phase, double Ey_mod, double Ey_phase, double Ez_mod, double Ez_phase, double Bx_mod, double Bx_phase, double By_mod, double By_phase, double Bz_mod, double Bz_phase, double B0, Grid *grid)
{

  double alpha;

  alpha = deltaBoB * B0 / sqrt(Bx_mod * Bx_mod + By_mod * By_mod + Bz_mod * Bz_mod);

  Ex_mod *= alpha;
  Ey_mod *= alpha;
  Ez_mod *= alpha;
  Bx_mod *= alpha;
  By_mod *= alpha;
  Bz_mod *= alpha;

  for (int i = 0; i < nxn; i++)
    for (int j = 0; j < nyn; j++)
    {
      Ex[i][j][0] += Ex_mod * cos(kx * grid->getXN(i, j, 0) + ky * grid->getYN(i, j, 0) + Ex_phase);
      Ey[i][j][0] += Ey_mod * cos(kx * grid->getXN(i, j, 0) + ky * grid->getYN(i, j, 0) + Ey_phase);
      Ez[i][j][0] += Ez_mod * cos(kx * grid->getXN(i, j, 0) + ky * grid->getYN(i, j, 0) + Ez_phase);
      Bxn[i][j][0] += Bx_mod * cos(kx * grid->getXN(i, j, 0) + ky * grid->getYN(i, j, 0) + Bx_phase);
      Byn[i][j][0] += By_mod * cos(kx * grid->getXN(i, j, 0) + ky * grid->getYN(i, j, 0) + By_phase);
      Bzn[i][j][0] += Bz_mod * cos(kx * grid->getXN(i, j, 0) + ky * grid->getYN(i, j, 0) + Bz_phase);
    }

  // initialize B on centers
  grid->interpN2C(Bxc, Bxn);
  grid->interpN2C(Byc, Byn);
  grid->interpN2C(Bzc, Bzn);
}

/*! Calculate hat rho hat, Jx hat, Jy hat, Jz hat */
void EMfields3D::calculateHatFunctions()
{
  const VirtualTopology3D *vct = &get_vct();
  const Grid *grid = &get_grid();
  // smoothing
  smooth(rhoc, 0);
  // calculate j hat

  for (int is = 0; is < ns; is++)
  {
    grid->divSymmTensorN2C(tempXC, tempYC, tempZC, pXXsn, pXYsn, pXZsn, pYYsn, pYZsn, pZZsn, is);

    scale(tempXC, -dt / 2.0, nxc, nyc, nzc);
    scale(tempYC, -dt / 2.0, nxc, nyc, nzc);
    scale(tempZC, -dt / 2.0, nxc, nyc, nzc);
    // communicate before interpolating
    communicateCenterBC_P(nxc, nyc, nzc, tempXC, 2, 2, 2, 2, 2, 2, vct, this);
    communicateCenterBC_P(nxc, nyc, nzc, tempYC, 2, 2, 2, 2, 2, 2, vct, this);
    communicateCenterBC_P(nxc, nyc, nzc, tempZC, 2, 2, 2, 2, 2, 2, vct, this);

    grid->interpC2N(tempXN, tempXC);
    grid->interpC2N(tempYN, tempYC);
    grid->interpC2N(tempZN, tempZC);
    sum(tempXN, Jxs, nxn, nyn, nzn, is);
    sum(tempYN, Jys, nxn, nyn, nzn, is);
    sum(tempZN, Jzs, nxn, nyn, nzn, is);
    // PIDOT
    PIdot(Jxh, Jyh, Jzh, tempXN, tempYN, tempZN, is);
  }
  // smooth j
  smooth(Jxh, 1);
  smooth(Jyh, 1);
  smooth(Jzh, 1);

  // calculate rho hat = rho - (dt*theta)div(jhat)
  grid->divN2C(tempXC, Jxh, Jyh, Jzh);
  scale(tempXC, -dt * th, nxc, nyc, nzc);
  sum(tempXC, rhoc, nxc, nyc, nzc);
  eq(rhoh, tempXC, nxc, nyc, nzc);
  // communicate rhoh
  communicateCenterBC_P(nxc, nyc, nzc, rhoh, 2, 2, 2, 2, 2, 2, vct, this);
}

/*! Image of Poisson Solver (uses persistent poissonTemp/poissonIm arrays) */
void EMfields3D::PoissonImage(double *image, double *vector)
{
  const VirtualTopology3D *vct = &get_vct();
  const Grid *grid = &get_grid();

  eqValue(0.0, image, (nxc - 2) * (nyc - 2) * (nzc - 2));
  eqValue(0.0, poissonTemp, nxc, nyc, nzc);
  eqValue(0.0, poissonIm, nxc, nyc, nzc);
  // move from krylov space to physical space and communicate ghost cells
  solver2phys(poissonTemp, vector, nxc, nyc, nzc);
  // calculate the laplacian
  grid->lapC2Cpoisson(poissonIm, poissonTemp, this);
  // move from physical space to krylov space
  phys2solver(image, poissonIm, nxc, nyc, nzc);
}
/*! interpolate charge density and pressure density from node to center */
void EMfields3D::interpDensitiesN2C()
{
  // do we need communication or not really?
  get_grid().interpN2C(rhoc, rhon);
}
/*! communicate ghost for grid -> Particles interpolation */
void EMfields3D::communicateGhostP2G(int ns)
{
  // interpolate adding common nodes among processors
  timeTasks_set_communicating();

  const VirtualTopology3D *vct = &get_vct();

  double ***moment0 = convert_to_arr3(rhons[ns]);
  double ***moment1 = convert_to_arr3(Jxs[ns]);
  double ***moment2 = convert_to_arr3(Jys[ns]);
  double ***moment3 = convert_to_arr3(Jzs[ns]);
  double ***moment4 = convert_to_arr3(pXXsn[ns]);
  double ***moment5 = convert_to_arr3(pXYsn[ns]);
  double ***moment6 = convert_to_arr3(pXZsn[ns]);
  double ***moment7 = convert_to_arr3(pYYsn[ns]);
  double ***moment8 = convert_to_arr3(pYZsn[ns]);
  double ***moment9 = convert_to_arr3(pZZsn[ns]);
  // add the values for the shared nodes

  // Call NonBlocking Halo Exchange + Interpolation
  communicateInterp(nxn, nyn, nzn, moment0, vct, this);
  communicateInterp(nxn, nyn, nzn, moment1, vct, this);
  communicateInterp(nxn, nyn, nzn, moment2, vct, this);
  communicateInterp(nxn, nyn, nzn, moment3, vct, this);
  communicateInterp(nxn, nyn, nzn, moment4, vct, this);
  communicateInterp(nxn, nyn, nzn, moment5, vct, this);
  communicateInterp(nxn, nyn, nzn, moment6, vct, this);
  communicateInterp(nxn, nyn, nzn, moment7, vct, this);
  communicateInterp(nxn, nyn, nzn, moment8, vct, this);
  communicateInterp(nxn, nyn, nzn, moment9, vct, this);
  // calculate the correct densities on the boundaries
  adjustNonPeriodicDensities(ns);

  // populate the ghost nodes

  // Call Nonblocking Halo Exchange
  communicateNode_P(nxn, nyn, nzn, moment0, vct, this);
  communicateNode_P(nxn, nyn, nzn, moment1, vct, this);
  communicateNode_P(nxn, nyn, nzn, moment2, vct, this);
  communicateNode_P(nxn, nyn, nzn, moment3, vct, this);
  communicateNode_P(nxn, nyn, nzn, moment4, vct, this);
  communicateNode_P(nxn, nyn, nzn, moment5, vct, this);
  communicateNode_P(nxn, nyn, nzn, moment6, vct, this);
  communicateNode_P(nxn, nyn, nzn, moment7, vct, this);
  communicateNode_P(nxn, nyn, nzn, moment8, vct, this);
  communicateNode_P(nxn, nyn, nzn, moment9, vct, this);
}

/*! communicate ghost for grid -> Particles interpolation */
// void EMfields3D::communicateGhostMomentsX()
//{
//   const VirtualTopology3D * vct = &get_vct();
//   timeTasks_set_communicating();
//
//   // start communication of moments in the X direction
//   for(int is=0; is<ns; i++)
//   {
//     for(int im=0; im<10; im++)
//     {
//       // copy data from faces into buffers
//       // send buffer data right and left
//     }
//   }
//
//   // receive and parse communication
// }

void EMfields3D::setZeroDerivedMoments()
{
  for (int i = 0; i < nxn; i++)
    for (int j = 0; j < nyn; j++)
      for (int k = 0; k < nzn; k++)
      {
        Jx[i][j][k] = 0.0;
        Jxh[i][j][k] = 0.0;
        Jy[i][j][k] = 0.0;
        Jyh[i][j][k] = 0.0;
        Jz[i][j][k] = 0.0;
        Jzh[i][j][k] = 0.0;
        rhon[i][j][k] = 0.0;
      }
  for (int i = 0; i < nxc; i++)
    for (int j = 0; j < nyc; j++)
      for (int k = 0; k < nzc; k++)
      {
        rhoc[i][j][k] = 0.0;
        rhoh[i][j][k] = 0.0;
      }
}

void EMfields3D::setZeroPrimaryMoments()
{

  // set primary moments to zero
  //
  for (int kk = 0; kk < ns; kk++)
    for (int i = 0; i < nxn; i++)
      for (int j = 0; j < nyn; j++)
        for (int k = 0; k < nzn; k++)
        {
          rhons[kk][i][j][k] = 0.0;
          Jxs[kk][i][j][k] = 0.0;
          Jys[kk][i][j][k] = 0.0;
          Jzs[kk][i][j][k] = 0.0;
          pXXsn[kk][i][j][k] = 0.0;
          pXYsn[kk][i][j][k] = 0.0;
          pXZsn[kk][i][j][k] = 0.0;
          pYYsn[kk][i][j][k] = 0.0;
          pYZsn[kk][i][j][k] = 0.0;
          pZZsn[kk][i][j][k] = 0.0;
        }
}
/*! set to 0 all the densities fields */
void EMfields3D::setZeroDensities()
{
  setZeroDerivedMoments();
  setZeroPrimaryMoments();
}

/*!SPECIES: Sum the charge density of different species on NODES */
void EMfields3D::sumOverSpecies()
{
  for (int is = 0; is < ns; is++)
    for (int i = 0; i < nxn; i++)
      for (int j = 0; j < nyn; j++)
        for (int k = 0; k < nzn; k++)
          rhon[i][j][k] += rhons[is][i][j][k];
}

/*!SPECIES: Sum current density for different species */
void EMfields3D::sumOverSpeciesJ()
{
  for (int is = 0; is < ns; is++)
    for (int i = 0; i < nxn; i++)
      for (int j = 0; j < nyn; j++)
        for (int k = 0; k < nzn; k++)
        {
          Jx[i][j][k] += Jxs[is][i][j][k];
          Jy[i][j][k] += Jys[is][i][j][k];
          Jz[i][j][k] += Jzs[is][i][j][k];
        }
}

/*! Calculate the susceptibility on the boundary leftX */
void EMfields3D::sustensorLeftX(double **susxx, double **susyx, double **suszx)
{
  double beta, omcx, omcy, omcz, denom;
  for (int j = 0; j < nyn; j++)
    for (int k = 0; k < nzn; k++)
    {
      susxx[j][k] = 1.0;
      susyx[j][k] = 0.0;
      suszx[j][k] = 0.0;
    }
  for (int is = 0; is < ns; is++)
  {
    beta = .5 * qom[is] * dt / c;
    for (int j = 0; j < nyn; j++)
      for (int k = 0; k < nzn; k++)
      {
        omcx = beta * (Bxn[1][j][k] + Bx_ext[1][j][k]);
        omcy = beta * (Byn[1][j][k] + By_ext[1][j][k]);
        omcz = beta * (Bzn[1][j][k] + Bz_ext[1][j][k]);
        denom = FourPI / 2 * delt * dt / c * qom[is] * rhons[is][1][j][k] / (1.0 + omcx * omcx + omcy * omcy + omcz * omcz);
        susxx[j][k] += (1.0 + omcx * omcx) * denom;
        susyx[j][k] += (-omcz + omcx * omcy) * denom;
        suszx[j][k] += (omcy + omcx * omcz) * denom;
      }
  }
}
/*! Calculate the susceptibility on the boundary rightX */
void EMfields3D::sustensorRightX(double **susxx, double **susyx, double **suszx)
{
  double beta, omcx, omcy, omcz, denom;
  for (int j = 0; j < nyn; j++)
    for (int k = 0; k < nzn; k++)
    {
      susxx[j][k] = 1.0;
      susyx[j][k] = 0.0;
      suszx[j][k] = 0.0;
    }
  for (int is = 0; is < ns; is++)
  {
    beta = .5 * qom[is] * dt / c;
    for (int j = 0; j < nyn; j++)
      for (int k = 0; k < nzn; k++)
      {
        omcx = beta * (Bxn[nxn - 2][j][k] + Bx_ext[nxn - 2][j][k]);
        omcy = beta * (Byn[nxn - 2][j][k] + By_ext[nxn - 2][j][k]);
        omcz = beta * (Bzn[nxn - 2][j][k] + Bz_ext[nxn - 2][j][k]);
        denom = FourPI / 2 * delt * dt / c * qom[is] * rhons[is][nxn - 2][j][k] / (1.0 + omcx * omcx + omcy * omcy + omcz * omcz);
        susxx[j][k] += (1.0 + omcx * omcx) * denom;
        susyx[j][k] += (-omcz + omcx * omcy) * denom;
        suszx[j][k] += (omcy + omcx * omcz) * denom;
      }
  }
}

/*! Calculate the susceptibility on the boundary left */
void EMfields3D::sustensorLeftY(double **susxy, double **susyy, double **suszy)
{
  double beta, omcx, omcy, omcz, denom;
  for (int i = 0; i < nxn; i++)
    for (int k = 0; k < nzn; k++)
    {
      susxy[i][k] = 0.0;
      susyy[i][k] = 1.0;
      suszy[i][k] = 0.0;
    }
  for (int is = 0; is < ns; is++)
  {
    beta = .5 * qom[is] * dt / c;
    for (int i = 0; i < nxn; i++)
      for (int k = 0; k < nzn; k++)
      {
        omcx = beta * (Bxn[i][1][k] + Bx_ext[i][1][k]);
        omcy = beta * (Byn[i][1][k] + By_ext[i][1][k]);
        omcz = beta * (Bzn[i][1][k] + Bz_ext[i][1][k]);
        denom = FourPI / 2 * delt * dt / c * qom[is] * rhons[is][i][1][k] / (1.0 + omcx * omcx + omcy * omcy + omcz * omcz);
        susxy[i][k] += (omcz + omcx * omcy) * denom;
        susyy[i][k] += (1.0 + omcy * omcy) * denom;
        suszy[i][k] += (-omcx + omcy * omcz) * denom;
      }
  }
}
/*! Calculate the susceptibility on the boundary right */
void EMfields3D::sustensorRightY(double **susxy, double **susyy, double **suszy)
{
  double beta, omcx, omcy, omcz, denom;
  for (int i = 0; i < nxn; i++)
    for (int k = 0; k < nzn; k++)
    {
      susxy[i][k] = 0.0;
      susyy[i][k] = 1.0;
      suszy[i][k] = 0.0;
    }
  for (int is = 0; is < ns; is++)
  {
    beta = .5 * qom[is] * dt / c;
    for (int i = 0; i < nxn; i++)
      for (int k = 0; k < nzn; k++)
      {
        omcx = beta * (Bxn[i][nyn - 2][k] + Bx_ext[i][nyn - 2][k]);
        omcy = beta * (Byn[i][nyn - 2][k] + By_ext[i][nyn - 2][k]);
        omcz = beta * (Bzn[i][nyn - 2][k] + Bz_ext[i][nyn - 2][k]);
        denom = FourPI / 2 * delt * dt / c * qom[is] * rhons[is][i][nyn - 2][k] / (1.0 + omcx * omcx + omcy * omcy + omcz * omcz);
        susxy[i][k] += (omcz + omcx * omcy) * denom;
        susyy[i][k] += (1.0 + omcy * omcy) * denom;
        suszy[i][k] += (-omcx + omcy * omcz) * denom;
      }
  }
}

/*! Calculate the susceptibility on the boundary left */
void EMfields3D::sustensorLeftZ(double **susxz, double **susyz, double **suszz)
{
  double beta, omcx, omcy, omcz, denom;
  for (int i = 0; i < nxn; i++)
    for (int j = 0; j < nyn; j++)
    {
      susxz[i][j] = 0.0;
      susyz[i][j] = 0.0;
      suszz[i][j] = 1.0;
    }
  for (int is = 0; is < ns; is++)
  {
    beta = .5 * qom[is] * dt / c;
    for (int i = 0; i < nxn; i++)
      for (int j = 0; j < nyn; j++)
      {
        omcx = beta * (Bxn[i][j][1] + Bx_ext[i][j][1]);
        omcy = beta * (Byn[i][j][1] + By_ext[i][j][1]);
        omcz = beta * (Bzn[i][j][1] + Bz_ext[i][j][1]);
        denom = FourPI / 2 * delt * dt / c * qom[is] * rhons[is][i][j][1] / (1.0 + omcx * omcx + omcy * omcy + omcz * omcz);
        susxz[i][j] += (-omcy + omcx * omcz) * denom;
        susyz[i][j] += (omcx + omcy * omcz) * denom;
        suszz[i][j] += (1.0 + omcz * omcz) * denom;
      }
  }
}
/*! Calculate the susceptibility on the boundary right */
void EMfields3D::sustensorRightZ(double **susxz, double **susyz, double **suszz)
{
  double beta, omcx, omcy, omcz, denom;
  for (int i = 0; i < nxn; i++)
    for (int j = 0; j < nyn; j++)
    {
      susxz[i][j] = 0.0;
      susyz[i][j] = 0.0;
      suszz[i][j] = 1.0;
    }
  for (int is = 0; is < ns; is++)
  {
    beta = .5 * qom[is] * dt / c;
    for (int i = 0; i < nxn; i++)
      for (int j = 0; j < nyn; j++)
      {
        omcx = beta * (Bxn[i][j][nzn - 2] + Bx_ext[i][j][nzn - 2]);
        omcy = beta * (Byn[i][j][nzn - 2] + By_ext[i][j][nzn - 2]);
        omcz = beta * (Bzn[i][j][nzn - 2] + Bz_ext[i][j][nzn - 2]);
        denom = FourPI / 2 * delt * dt / c * qom[is] * rhons[is][i][j][nzn - 2] / (1.0 + omcx * omcx + omcy * omcy + omcz * omcz);
        susxz[i][j] += (-omcy + omcx * omcz) * denom;
        susyz[i][j] += (omcx + omcy * omcz) * denom;
        suszz[i][j] += (1.0 + omcz * omcz) * denom;
      }
  }
}

/*! Perfect conductor boundary conditions: LEFT wall */
void EMfields3D::perfectConductorLeft(arr3_double imageX, arr3_double imageY, arr3_double imageZ,
                                      const_arr3_double vectorX, const_arr3_double vectorY, const_arr3_double vectorZ,
                                      int dir)
{
  double **susxy;
  double **susyy;
  double **suszy;
  double **susxx;
  double **susyx;
  double **suszx;
  double **susxz;
  double **susyz;
  double **suszz;
  switch (dir)
  {
  case 0: // boundary condition on X-DIRECTION
    susxx = newArr2(double, nyn, nzn);
    susyx = newArr2(double, nyn, nzn);
    suszx = newArr2(double, nyn, nzn);
    sustensorLeftX(susxx, susyx, suszx);
    for (int i = 1; i < nyn - 1; i++)
      for (int j = 1; j < nzn - 1; j++)
      {
        imageX[1][i][j] = vectorX.get(1, i, j) - (Ex[1][i][j] - susyx[i][j] * vectorY.get(1, i, j) - suszx[i][j] * vectorZ.get(1, i, j) - Jxh[1][i][j] * dt * th * FourPI) / susxx[i][j];
        imageY[1][i][j] = vectorY.get(1, i, j) - 0.0 * vectorY.get(2, i, j);
        imageZ[1][i][j] = vectorZ.get(1, i, j) - 0.0 * vectorZ.get(2, i, j);
      }
    delArr2(susxx, nyn);
    delArr2(susyx, nyn);
    delArr2(suszx, nyn);
    break;
  case 1: // boundary condition on Y-DIRECTION
    susxy = newArr2(double, nxn, nzn);
    susyy = newArr2(double, nxn, nzn);
    suszy = newArr2(double, nxn, nzn);
    sustensorLeftY(susxy, susyy, suszy);
    for (int i = 1; i < nxn - 1; i++)
      for (int j = 1; j < nzn - 1; j++)
      {
        imageX[i][1][j] = vectorX.get(i, 1, j) - 0.0 * vectorX.get(i, 2, j);
        imageY[i][1][j] = vectorY.get(i, 1, j) - (Ey[i][1][j] - susxy[i][j] * vectorX.get(i, 1, j) - suszy[i][j] * vectorZ.get(i, 1, j) - Jyh[i][1][j] * dt * th * FourPI) / susyy[i][j];
        imageZ[i][1][j] = vectorZ.get(i, 1, j) - 0.0 * vectorZ.get(i, 2, j);
      }
    delArr2(susxy, nxn);
    delArr2(susyy, nxn);
    delArr2(suszy, nxn);
    break;
  case 2: // boundary condition on Z-DIRECTION
    susxz = newArr2(double, nxn, nyn);
    susyz = newArr2(double, nxn, nyn);
    suszz = newArr2(double, nxn, nyn);
    sustensorLeftZ(susxz, susyz, suszz);
    for (int i = 1; i < nxn - 1; i++)
      for (int j = 1; j < nyn - 1; j++)
      {
        imageX[i][j][1] = vectorX.get(i, j, 1);
        imageY[i][j][1] = vectorY.get(i, j, 1);
        imageZ[i][j][1] = vectorZ.get(i, j, 1) - (Ez[i][j][1] - susxz[i][j] * vectorX.get(i, j, 1) - susyz[i][j] * vectorY.get(i, j, 1) - Jzh[i][j][1] * dt * th * FourPI) / suszz[i][j];
      }
    delArr2(susxz, nxn);
    delArr2(susyz, nxn);
    delArr2(suszz, nxn);
    break;
  }
}

/*! Perfect conductor boundary conditions: RIGHT wall */
void EMfields3D::perfectConductorRight(
    arr3_double imageX, arr3_double imageY, arr3_double imageZ,
    const_arr3_double vectorX,
    const_arr3_double vectorY,
    const_arr3_double vectorZ,
    int dir)
{
  double beta, omcx, omcy, omcz, denom;
  double **susxy;
  double **susyy;
  double **suszy;
  double **susxx;
  double **susyx;
  double **suszx;
  double **susxz;
  double **susyz;
  double **suszz;
  switch (dir)
  {
  case 0: // boundary condition on X-DIRECTION RIGHT
    susxx = newArr2(double, nyn, nzn);
    susyx = newArr2(double, nyn, nzn);
    suszx = newArr2(double, nyn, nzn);
    sustensorRightX(susxx, susyx, suszx);
    for (int i = 1; i < nyn - 1; i++)
      for (int j = 1; j < nzn - 1; j++)
      {
        imageX[nxn - 2][i][j] = vectorX.get(nxn - 2, i, j) - (Ex[nxn - 2][i][j] - susyx[i][j] * vectorY.get(nxn - 2, i, j) - suszx[i][j] * vectorZ.get(nxn - 2, i, j) - Jxh[nxn - 2][i][j] * dt * th * FourPI) / susxx[i][j];
        imageY[nxn - 2][i][j] = vectorY.get(nxn - 2, i, j) - 0.0 * vectorY.get(nxn - 3, i, j);
        imageZ[nxn - 2][i][j] = vectorZ.get(nxn - 2, i, j) - 0.0 * vectorZ.get(nxn - 3, i, j);
      }
    delArr2(susxx, nyn);
    delArr2(susyx, nyn);
    delArr2(suszx, nyn);
    break;
  case 1: // boundary condition on Y-DIRECTION RIGHT
    susxy = newArr2(double, nxn, nzn);
    susyy = newArr2(double, nxn, nzn);
    suszy = newArr2(double, nxn, nzn);
    sustensorRightY(susxy, susyy, suszy);
    for (int i = 1; i < nxn - 1; i++)
      for (int j = 1; j < nzn - 1; j++)
      {
        imageX[i][nyn - 2][j] = vectorX.get(i, nyn - 2, j) - 0.0 * vectorX.get(i, nyn - 3, j);
        imageY[i][nyn - 2][j] = vectorY.get(i, nyn - 2, j) - (Ey[i][nyn - 2][j] - susxy[i][j] * vectorX.get(i, nyn - 2, j) - suszy[i][j] * vectorZ.get(i, nyn - 2, j) - Jyh[i][nyn - 2][j] * dt * th * FourPI) / susyy[i][j];
        imageZ[i][nyn - 2][j] = vectorZ.get(i, nyn - 2, j) - 0.0 * vectorZ.get(i, nyn - 3, j);
      }
    delArr2(susxy, nxn);
    delArr2(susyy, nxn);
    delArr2(suszy, nxn);
    break;
  case 2: // boundary condition on Z-DIRECTION RIGHT
    susxz = newArr2(double, nxn, nyn);
    susyz = newArr2(double, nxn, nyn);
    suszz = newArr2(double, nxn, nyn);
    sustensorRightZ(susxz, susyz, suszz);
    for (int i = 1; i < nxn - 1; i++)
      for (int j = 1; j < nyn - 1; j++)
      {
        imageX[i][j][nzn - 2] = vectorX.get(i, j, nzn - 2);
        imageY[i][j][nzn - 2] = vectorY.get(i, j, nzn - 2);
        imageZ[i][j][nzn - 2] = vectorZ.get(i, j, nzn - 2) - (Ez[i][j][nzn - 2] - susxz[i][j] * vectorX.get(i, j, nzn - 2) - susyz[i][j] * vectorY.get(i, j, nzn - 2) - Jzh[i][j][nzn - 2] * dt * th * FourPI) / suszz[i][j];
      }
    delArr2(susxz, nxn);
    delArr2(susyz, nxn);
    delArr2(suszz, nxn);
    break;
  }
}

/*! Perfect conductor boundary conditions for source: LEFT WALL */
void EMfields3D::perfectConductorLeftS(arr3_double vectorX, arr3_double vectorY, arr3_double vectorZ, int dir)
{

  double ebc[3];

  // Assuming E = - ve x B
  cross_product(ue0, ve0, we0, B0x, B0y, B0z, ebc);
  scale(ebc, -1.0, 3);

  switch (dir)
  {
  case 0: // boundary condition on X-DIRECTION LEFT
    for (int i = 1; i < nyn - 1; i++)
      for (int j = 1; j < nzn - 1; j++)
      {
        vectorX[1][i][j] = 0.0;
        vectorY[1][i][j] = ebc[1];
        vectorZ[1][i][j] = ebc[2];
        //+//          vectorX[1][i][j] = 0.0;
        //+//          vectorY[1][i][j] = 0.0;
        //+//          vectorZ[1][i][j] = 0.0;
      }
    break;
  case 1: // boundary condition on Y-DIRECTION LEFT
    for (int i = 1; i < nxn - 1; i++)
      for (int j = 1; j < nzn - 1; j++)
      {
        vectorX[i][1][j] = ebc[0];
        vectorY[i][1][j] = 0.0;
        vectorZ[i][1][j] = ebc[2];
        //+//          vectorX[i][1][j] = 0.0;
        //+//          vectorY[i][1][j] = 0.0;
        //+//          vectorZ[i][1][j] = 0.0;
      }
    break;
  case 2: // boundary condition on Z-DIRECTION LEFT
    for (int i = 1; i < nxn - 1; i++)
      for (int j = 1; j < nyn - 1; j++)
      {
        vectorX[i][j][1] = ebc[0];
        vectorY[i][j][1] = ebc[1];
        vectorZ[i][j][1] = 0.0;
        //+//          vectorX[i][j][1] = 0.0;
        //+//          vectorY[i][j][1] = 0.0;
        //+//          vectorZ[i][j][1] = 0.0;
      }
    break;
  }
}

/*! Perfect conductor boundary conditions for source: RIGHT WALL */
void EMfields3D::perfectConductorRightS(arr3_double vectorX, arr3_double vectorY, arr3_double vectorZ, int dir)
{

  double ebc[3];

  // Assuming E = - ve x B
  cross_product(ue0, ve0, we0, B0x, B0y, B0z, ebc);
  scale(ebc, -1.0, 3);

  switch (dir)
  {
  case 0: // boundary condition on X-DIRECTION RIGHT
    for (int i = 1; i < nyn - 1; i++)
      for (int j = 1; j < nzn - 1; j++)
      {
        vectorX[nxn - 2][i][j] = 0.0;
        vectorY[nxn - 2][i][j] = ebc[1];
        vectorZ[nxn - 2][i][j] = ebc[2];
        //+//          vectorX[nxn-2][i][j] = 0.0;
        //+//          vectorY[nxn-2][i][j] = 0.0;
        //+//          vectorZ[nxn-2][i][j] = 0.0;
      }
    break;
  case 1: // boundary condition on Y-DIRECTION RIGHT
    for (int i = 1; i < nxn - 1; i++)
      for (int j = 1; j < nzn - 1; j++)
      {
        vectorX[i][nyn - 2][j] = ebc[0];
        vectorY[i][nyn - 2][j] = 0.0;
        vectorZ[i][nyn - 2][j] = ebc[2];
        //+//          vectorX[i][nyn-2][j] = 0.0;
        //+//          vectorY[i][nyn-2][j] = 0.0;
        //+//          vectorZ[i][nyn-2][j] = 0.0;
      }
    break;
  case 2:
    for (int i = 1; i < nxn - 1; i++)
      for (int j = 1; j < nyn - 1; j++)
      {
        vectorX[i][j][nzn - 2] = ebc[0];
        vectorY[i][j][nzn - 2] = ebc[1];
        vectorZ[i][j][nzn - 2] = 0.0;
        //+//          vectorX[i][j][nzn-2] = 0.0;
        //+//          vectorY[i][j][nzn-2] = 0.0;
        //+//          vectorZ[i][j][nzn-2] = 0.0;
      }
    break;
  }
}

/*! Open boundary inflow source: set RHS to zero at boundary nodes
 *  where the operator row has been replaced by E - E_inflow.
 *  This ensures GMRES enforces E = E_inflow at the boundary. */
void EMfields3D::OpenBoundaryInflowESource(arr3_double vectorX, arr3_double vectorY, arr3_double vectorZ,
                                           int nx, int ny, int nz)
{
  const VirtualTopology3D *vct = &get_vct();
  const Collective *col = &get_col();

  // Zero RHS on each face where both EM BC is open (bcEMface==2)
  // and particle BC is reemission (bcPface==2)
  if (vct->getXleft_neighbor() == MPI_PROC_NULL && bcEMfaceXleft == 2 && col->getBcPfaceXleft() == 2)
  {
    for (int j = 1; j < ny - 1; j++)
      for (int k = 1; k < nz - 1; k++)
      {
        vectorX[1][j][k] = 0.0;
        vectorY[1][j][k] = 0.0;
        vectorZ[1][j][k] = 0.0;
      }
  }
  if (vct->getXright_neighbor() == MPI_PROC_NULL && bcEMfaceXright == 2 && col->getBcPfaceXright() == 2)
  {
    for (int j = 1; j < ny - 1; j++)
      for (int k = 1; k < nz - 1; k++)
      {
        vectorX[nx-2][j][k] = 0.0;
        vectorY[nx-2][j][k] = 0.0;
        vectorZ[nx-2][j][k] = 0.0;
      }
  }
  if (vct->getYleft_neighbor() == MPI_PROC_NULL && bcEMfaceYleft == 2 && col->getBcPfaceYleft() == 2)
  {
    for (int i = 1; i < nx - 1; i++)
      for (int k = 1; k < nz - 1; k++)
      {
        vectorX[i][1][k] = 0.0;
        vectorY[i][1][k] = 0.0;
        vectorZ[i][1][k] = 0.0;
      }
  }
  if (vct->getYright_neighbor() == MPI_PROC_NULL && bcEMfaceYright == 2 && col->getBcPfaceYright() == 2)
  {
    for (int i = 1; i < nx - 1; i++)
      for (int k = 1; k < nz - 1; k++)
      {
        vectorX[i][ny-2][k] = 0.0;
        vectorY[i][ny-2][k] = 0.0;
        vectorZ[i][ny-2][k] = 0.0;
      }
  }
  if (vct->getZleft_neighbor() == MPI_PROC_NULL && bcEMfaceZleft == 2 && col->getBcPfaceZleft() == 2)
  {
    for (int i = 1; i < nx - 1; i++)
      for (int j = 1; j < ny - 1; j++)
      {
        vectorX[i][j][1] = 0.0;
        vectorY[i][j][1] = 0.0;
        vectorZ[i][j][1] = 0.0;
      }
  }
  if (vct->getZright_neighbor() == MPI_PROC_NULL && bcEMfaceZright == 2 && col->getBcPfaceZright() == 2)
  {
    for (int i = 1; i < nx - 1; i++)
      for (int j = 1; j < ny - 1; j++)
      {
        vectorX[i][j][nz-2] = 0.0;
        vectorY[i][j][nz-2] = 0.0;
        vectorZ[i][j][nz-2] = 0.0;
      }
  }
}

void EMfields3D::OpenBoundaryInflowEImage(arr3_double imageX, arr3_double imageY, arr3_double imageZ,
                                          const_arr3_double vectorX, const_arr3_double vectorY, const_arr3_double vectorZ,
                                          int nx, int ny, int nz)
{
  const VirtualTopology3D *vct = &get_vct();
  const Collective *col = &get_col();
  // Assuming E = - ve x B
  double injE[3];
  cross_product(ue0, ve0, we0, B0x, B0y, B0z, injE);
  scale(injE, -1.0, 3);

  // Apply image = E - E_inflow on each face where both EM BC is open (bcEMface==2)
  // and particle BC is reemission (bcPface==2)
  if (vct->getXleft_neighbor() == MPI_PROC_NULL && bcEMfaceXleft == 2 && col->getBcPfaceXleft() == 2)
  {
    for (int j = 1; j < ny - 1; j++)
      for (int k = 1; k < nz - 1; k++)
      {
        imageX[1][j][k] = vectorX[1][j][k] - injE[0];
        imageY[1][j][k] = vectorY[1][j][k] - injE[1];
        imageZ[1][j][k] = vectorZ[1][j][k] - injE[2];
      }
  }
  /* Xright inflow image currently disabled (outflow face uses extrapolation in post-solve).
  if (vct->getXright_neighbor() == MPI_PROC_NULL && bcEMfaceXright == 2 && col->getBcPfaceXright() == 2)
  {
    for (int j = 1; j < ny - 1; j++)
      for (int k = 1; k < nz - 1; k++)
      {
        imageX[nx-2][j][k] = vectorX[nx-2][j][k] - injE[0];
        imageY[nx-2][j][k] = vectorY[nx-2][j][k] - injE[1];
        imageZ[nx-2][j][k] = vectorZ[nx-2][j][k] - injE[2];
      }
  }
  */
  if (vct->getYleft_neighbor() == MPI_PROC_NULL && bcEMfaceYleft == 2 && col->getBcPfaceYleft() == 2)
  {
    for (int i = 1; i < nx - 1; i++)
      for (int k = 1; k < nz - 1; k++)
      {
        imageX[i][1][k] = vectorX[i][1][k] - injE[0];
        imageY[i][1][k] = vectorY[i][1][k] - injE[1];
        imageZ[i][1][k] = vectorZ[i][1][k] - injE[2];
      }
  }
  if (vct->getYright_neighbor() == MPI_PROC_NULL && bcEMfaceYright == 2 && col->getBcPfaceYright() == 2)
  {
    for (int i = 1; i < nx - 1; i++)
      for (int k = 1; k < nz - 1; k++)
      {
        imageX[i][ny-2][k] = vectorX[i][ny-2][k] - injE[0];
        imageY[i][ny-2][k] = vectorY[i][ny-2][k] - injE[1];
        imageZ[i][ny-2][k] = vectorZ[i][ny-2][k] - injE[2];
      }
  }
  if (vct->getZleft_neighbor() == MPI_PROC_NULL && bcEMfaceZleft == 2 && col->getBcPfaceZleft() == 2)
  {
    for (int i = 1; i < nx - 1; i++)
      for (int j = 1; j < ny - 1; j++)
      {
        imageX[i][j][1] = vectorX[i][j][1] - injE[0];
        imageY[i][j][1] = vectorY[i][j][1] - injE[1];
        imageZ[i][j][1] = vectorZ[i][j][1] - injE[2];
      }
  }
  if (vct->getZright_neighbor() == MPI_PROC_NULL && bcEMfaceZright == 2 && col->getBcPfaceZright() == 2)
  {
    for (int i = 1; i < nx - 1; i++)
      for (int j = 1; j < ny - 1; j++)
      {
        imageX[i][j][nz-2] = vectorX[i][j][nz-2] - injE[0];
        imageY[i][j][nz-2] = vectorY[i][j][nz-2] - injE[1];
        imageZ[i][j][nz-2] = vectorZ[i][j][nz-2] - injE[2];
      }
  }
}

void EMfields3D::OpenBoundaryInflowB(arr3_double vectorX, arr3_double vectorY, arr3_double vectorZ,
                                     int nx, int ny, int nz)
{
  const VirtualTopology3D *vct = &get_vct();

  double sal;

  if (vct->getXleft_neighbor() == MPI_PROC_NULL && bcEMfaceXleft == 2 && nx > 10)
  {
    if (yes_sal)
    {
      for (int i = 0; i <= n_layers_sal; i++)
      {
        sal = (double)i / n_layers_sal;
        for (int j = 0; j < ny; j++)
          for (int k = 0; k < nz; k++)
          {
            vectorX[i][j][k] = vectorX[i][j][k] * sal + B0x * (1. - sal);
            vectorY[i][j][k] = vectorY[i][j][k] * sal + B0y * (1. - sal);
            vectorZ[i][j][k] = vectorZ[i][j][k] * sal + B0z * (1. - sal);
          }
      }
    }
    else
    {
      for (int i = 0; i <= n_layers_sal; i++)
        for (int j = 0; j < ny; j++)
          for (int k = 0; k < nz; k++)
          {
            vectorX[i][j][k] = B0x;
            vectorY[i][j][k] = B0y;
            vectorZ[i][j][k] = B0z;
          }
    }
  }

  if (vct->getXright_neighbor() == MPI_PROC_NULL && bcEMfaceXright == 2 && nx > 10)
  {
    // force not to apply sal in Xright
    for (int i = nx - n_layers_sal - 1; i < nx; i++)
      for (int j = 0; j < ny; j++)
        for (int k = 0; k < nz; k++)
        {
          vectorX[i][j][k] = vectorX[nx - 2 - n_layers_sal][j][k];
          vectorY[i][j][k] = vectorY[nx - 2 - n_layers_sal][j][k];
          vectorZ[i][j][k] = vectorZ[nx - 2 - n_layers_sal][j][k];
        }
  }

  if (vct->getYleft_neighbor() == MPI_PROC_NULL && bcEMfaceYleft == 2 && ny > 10)
  {
    if (yes_sal)
    {
      for (int i = 0; i < nx; i++)
        for (int j = 0; j <= n_layers_sal; j++)
        {
          sal = (double)j / n_layers_sal;
          for (int k = 0; k < nz; k++)
          {
            vectorX[i][j][k] = vectorX[i][j][k] * sal + B0x * (1. - sal);
            vectorY[i][j][k] = vectorY[i][j][k] * sal + B0y * (1. - sal);
            vectorZ[i][j][k] = vectorZ[i][j][k] * sal + B0z * (1. - sal);
          }
        }
    }
    else
    {
      for (int i = 0; i < nx; i++)
        for (int j = 0; j <= n_layers_sal; j++)
          for (int k = 0; k < nz; k++)
          {
            vectorX[i][j][k] = vectorX[i][n_layers_sal + 1][k];
            vectorY[i][j][k] = vectorY[i][n_layers_sal + 1][k];
            vectorZ[i][j][k] = vectorZ[i][n_layers_sal + 1][k];
          }
    }
  }

  if (vct->getYright_neighbor() == MPI_PROC_NULL && bcEMfaceYright == 2 && ny > 10)
  {
    if (yes_sal)
    {
      for (int i = 0; i < nx; i++)
        for (int j = ny - n_layers_sal - 1; j < ny; j++)
        {
          sal = (double)(ny - 1 - j) / n_layers_sal;
          for (int k = 0; k < nz; k++)
          {
            vectorX[i][j][k] = vectorX[i][j][k] * sal + B0x * (1. - sal);
            vectorY[i][j][k] = vectorY[i][j][k] * sal + B0y * (1. - sal);
            vectorZ[i][j][k] = vectorZ[i][j][k] * sal + B0z * (1. - sal);
          }
        }
    }
    else
    {
      for (int i = 0; i < nx; i++)
        for (int j = ny - n_layers_sal - 1; j < ny; j++)
          for (int k = 0; k < nz; k++)
          {
            vectorX[i][j][k] = vectorX[i][ny - 2 - n_layers_sal][k];
            vectorY[i][j][k] = vectorY[i][ny - 2 - n_layers_sal][k];
            vectorZ[i][j][k] = vectorZ[i][ny - 2 - n_layers_sal][k];
          }
    }
  }

  if (vct->getZleft_neighbor() == MPI_PROC_NULL && bcEMfaceZleft == 2 && nz > 10)
  {
    if (yes_sal)
    {
      for (int i = 0; i < nx; i++)
        for (int j = 0; j < ny; j++)
          for (int k = 0; k <= n_layers_sal; k++)
          {
            sal = (double)k / n_layers_sal;
            vectorX[i][j][k] = vectorX[i][j][k] * sal + B0x * (1. - sal);
            vectorY[i][j][k] = vectorY[i][j][k] * sal + B0y * (1. - sal);
            vectorZ[i][j][k] = vectorZ[i][j][k] * sal + B0z * (1. - sal);
          }
    }
    else
    {
      for (int i = 0; i < nx; i++)
        for (int j = 0; j < ny; j++)
          for (int k = 0; k <= n_layers_sal; k++)
          {
            vectorX[i][j][k] = vectorX[i][j][n_layers_sal + 1];
            vectorY[i][j][k] = vectorY[i][j][n_layers_sal + 1];
            vectorZ[i][j][k] = vectorZ[i][j][n_layers_sal + 1];
          }
    }
  }

  if (vct->getZright_neighbor() == MPI_PROC_NULL && bcEMfaceZright == 2 && nz > 10)
  {
    if (yes_sal)
    {
      for (int i = 0; i < nx; i++)
        for (int j = 0; j < ny; j++)
          for (int k = nz - n_layers_sal - 1; k < nz; k++)
          {
            sal = (double)(nz - 1 - k) / n_layers_sal;
            vectorX[i][j][k] = vectorX[i][j][k] * sal + B0x * (1. - sal);
            vectorY[i][j][k] = vectorY[i][j][k] * sal + B0y * (1. - sal);
            vectorZ[i][j][k] = vectorZ[i][j][k] * sal + B0z * (1. - sal);
          }
    }
    else
    {
      for (int i = 0; i < nx; i++)
        for (int j = 0; j < ny; j++)
          for (int k = nz - n_layers_sal - 1; k < nz; k++)
          {
            vectorX[i][j][k] = vectorX[i][j][nz - 2 - n_layers_sal];
            vectorY[i][j][k] = vectorY[i][j][nz - 2 - n_layers_sal];
            vectorZ[i][j][k] = vectorZ[i][j][nz - 2 - n_layers_sal];
          }
    }
  }
}

void EMfields3D::OpenBoundaryInflowE(arr3_double vectorX, arr3_double vectorY, arr3_double vectorZ,
                                     int nx, int ny, int nz)
{
  const VirtualTopology3D *vct = &get_vct();
  const Collective *col = &get_col();
  // Assuming E = - ve x B
  double injE[3];
  cross_product(ue0, ve0, we0, B0x, B0y, B0z, injE);
  scale(injE, -1.0, 3);

  if (yes_sal)
  {
    // SAL (simple absorbing layer): blend solved E toward injE over n_layers_sal nodes
    // Applied per face where bcEMface==2 and bcPface==2 (reemission)
    double sal;

    if (vct->getXleft_neighbor() == MPI_PROC_NULL && bcEMfaceXleft == 2 && col->getBcPfaceXleft() == 2)
    {
      for (int i = 0; i <= n_layers_sal; i++)
      {
        sal = (double)i / n_layers_sal;
        for (int j = 0; j < ny; j++)
          for (int k = 0; k < nz; k++)
          {
            vectorX[i][j][k] = vectorX[i][j][k] * sal + injE[0] * (1. - sal);
            vectorY[i][j][k] = vectorY[i][j][k] * sal + injE[1] * (1. - sal);
            vectorZ[i][j][k] = vectorZ[i][j][k] * sal + injE[2] * (1. - sal);
          }
      }
    }
    if (vct->getXright_neighbor() == MPI_PROC_NULL && bcEMfaceXright == 2 && col->getBcPfaceXright() == 3)
    {
      for (int i = nx - n_layers_sal - 1; i < nx; i++)
      {
        sal = (double)(nx - 1. - i) / n_layers_sal;
        for (int j = 0; j < ny; j++)
          for (int k = 0; k < nz; k++)
          {
            vectorX[i][j][k] = vectorX[nx - 2 - n_layers_sal][j][k];
            vectorY[i][j][k] = vectorY[nx - 2 - n_layers_sal][j][k];
            vectorZ[i][j][k] = vectorZ[nx - 2 - n_layers_sal][j][k];
          }
      }
    }

    if (vct->getYleft_neighbor() == MPI_PROC_NULL && bcEMfaceYleft == 2 && col->getBcPfaceYleft() == 2)
    {
      for (int i = 0; i < nx; i++)
        for (int j = 0; j <= n_layers_sal; j++)
        {
          sal = (double)j / n_layers_sal;
          for (int k = 0; k < nz; k++)
          {
            vectorX[i][j][k] = vectorX[i][j][k] * sal + injE[0] * (1. - sal);
            vectorY[i][j][k] = vectorY[i][j][k] * sal + injE[1] * (1. - sal);
            vectorZ[i][j][k] = vectorZ[i][j][k] * sal + injE[2] * (1. - sal);
          }
        }
    }

    if (vct->getYright_neighbor() == MPI_PROC_NULL && bcEMfaceYright == 2 && col->getBcPfaceYright() == 2)
    {
      for (int i = 0; i < nx; i++)
        for (int j = ny - n_layers_sal - 1; j < ny; j++)
        {
          sal = (double)(ny - 1 - j) / n_layers_sal;
          for (int k = 0; k < nz; k++)
          {
            vectorX[i][j][k] = vectorX[i][j][k] * sal + injE[0] * (1. - sal);
            vectorY[i][j][k] = vectorY[i][j][k] * sal + injE[1] * (1. - sal);
            vectorZ[i][j][k] = vectorZ[i][j][k] * sal + injE[2] * (1. - sal);
          }
        }
    }

    if (vct->getZleft_neighbor() == MPI_PROC_NULL && bcEMfaceZleft == 2 && col->getBcPfaceZleft() == 2)
    {
      for (int i = 0; i < nx; i++)
        for (int j = 0; j < ny; j++)
          for (int k = 0; k <= n_layers_sal; k++)
          {
            sal = (double)k / n_layers_sal;
            vectorX[i][j][k] = vectorX[i][j][k] * sal + injE[0] * (1. - sal);
            vectorY[i][j][k] = vectorY[i][j][k] * sal + injE[1] * (1. - sal);
            vectorZ[i][j][k] = vectorZ[i][j][k] * sal + injE[2] * (1. - sal);
          }
    }

    if (vct->getZright_neighbor() == MPI_PROC_NULL && bcEMfaceZright == 2 && col->getBcPfaceZright() == 2)
    {
      for (int i = 0; i < nx; i++)
        for (int j = 0; j < ny; j++)
          for (int k = nz - n_layers_sal - 1; k < nz; k++)
          {
            sal = (double)(nz - 1 - k) / n_layers_sal;
            vectorX[i][j][k] = vectorX[i][j][k] * sal + injE[0] * (1. - sal);
            vectorY[i][j][k] = vectorY[i][j][k] * sal + injE[1] * (1. - sal);
            vectorZ[i][j][k] = vectorZ[i][j][k] * sal + injE[2] * (1. - sal);
          }
    }
  }
  else
  {
    // No SAL: Dirichlet inflow or extrapolation from interior
    // Applied per face where bcEMface==2 and bcPface==2 (reemission)

    if (vct->getXleft_neighbor() == MPI_PROC_NULL && bcEMfaceXleft == 2 && col->getBcPfaceXleft() == 2)
    {
      for (int i = 0; i <= n_layers_sal; i++)
        for (int j = 0; j < ny; j++)
          for (int k = 0; k < nz; k++)
          {
            vectorX[i][j][k] = injE[0];
            vectorY[i][j][k] = injE[1];
            vectorZ[i][j][k] = injE[2];
          }
    }
    // outflow face 
    if (vct->getXright_neighbor() == MPI_PROC_NULL && bcEMfaceXright == 2 && col->getBcPfaceXright() == 3)
    {
      for (int i = nx - n_layers_sal - 1; i < nx; i++)
        for (int j = 0; j < ny; j++)
          for (int k = 0; k < nz; k++)
          {
            vectorX[i][j][k] = vectorX[nx - 2 - n_layers_sal][j][k];
            vectorY[i][j][k] = vectorY[nx - 2 - n_layers_sal][j][k];
            vectorZ[i][j][k] = vectorZ[nx - 2 - n_layers_sal][j][k];
          }
    }

    if (vct->getYleft_neighbor() == MPI_PROC_NULL && bcEMfaceYleft == 2 && col->getBcPfaceYleft() == 2)
    {
      for (int i = 0; i < nx; i++)
        for (int j = 0; j <= n_layers_sal; j++)
          for (int k = 0; k < nz; k++)
          {
            vectorX[i][j][k] = vectorX[i][n_layers_sal + 1][k];
            vectorY[i][j][k] = vectorY[i][n_layers_sal + 1][k];
            vectorZ[i][j][k] = vectorZ[i][n_layers_sal + 1][k];
          }
    }

    if (vct->getYright_neighbor() == MPI_PROC_NULL && bcEMfaceYright == 2 && col->getBcPfaceYright() == 2)
    {
      for (int i = 0; i < nx; i++)
        for (int j = ny - n_layers_sal - 1; j < ny; j++)
          for (int k = 0; k < nz; k++)
          {
            vectorX[i][j][k] = vectorX[i][ny - 2 - n_layers_sal][k];
            vectorY[i][j][k] = vectorY[i][ny - 2 - n_layers_sal][k];
            vectorZ[i][j][k] = vectorZ[i][ny - 2 - n_layers_sal][k];
          }
    }

    if (vct->getZleft_neighbor() == MPI_PROC_NULL && bcEMfaceZleft == 2 && col->getBcPfaceZleft() == 2)
    {
      for (int i = 0; i < nx; i++)
        for (int j = 0; j < ny; j++)
          for (int k = 0; k <= n_layers_sal; k++)
          {
            vectorX[i][j][k] = vectorX[i][j][n_layers_sal + 1];
            vectorY[i][j][k] = vectorY[i][j][n_layers_sal + 1];
            vectorZ[i][j][k] = vectorZ[i][j][n_layers_sal + 1];
          }
    }

    if (vct->getZright_neighbor() == MPI_PROC_NULL && bcEMfaceZright == 2 && col->getBcPfaceZright() == 2)
    {
      for (int i = 0; i < nx; i++)
        for (int j = 0; j < ny; j++)
          for (int k = nz - n_layers_sal - 1; k < nz; k++)
          {
            vectorX[i][j][k] = vectorX[i][j][nz - 2 - n_layers_sal];
            vectorY[i][j][k] = vectorY[i][j][nz - 2 - n_layers_sal];
            vectorZ[i][j][k] = vectorZ[i][j][nz - 2 - n_layers_sal];
          }
    }
  }
}

/*! get Electric Field component X array cell without the ghost cells */
// arr3_double EMfields3D::getExc()
//{
//   array3_double tmp(nxc,nyc,nzc);
//   get_grid().interpN2C(tmp, Ex);
//
//   for (int i = 1; i < nxc-1; i++)
//     for (int j = 1; j < nyc-1; j++)
//       for (int k = 1; k < nzc-1; k++)
//         arr[i-1][j-1][k-1]=tmp[i][j][k];
//   return arr;
// }
/*! get Electric Field component Y array cell without the ghost cells */
// arr3_double EMfields3D::getEyc()
//{
//   array3_double tmp(nxc,nyc,nzc);
//   get_grid().interpN2C(tmp, Ey);
//
//   for (int i = 1; i < nxc-1; i++)
//     for (int j = 1; j < nyc-1; j++)
//       for (int k = 1; k < nzc-1; k++)
//         arr[i-1][j-1][k-1]=tmp[i][j][k];
//   return arr;
// }
/*! get Electric Field component Z array cell without the ghost cells */
// arr3_double EMfields3D::getEzc()
//{
//   array3_double tmp(nxc,nyc,nzc);
//   get_grid().interpN2C(tmp, Ez);
//
//   for (int i = 1; i < nxc-1; i++)
//     for (int j = 1; j < nyc-1; j++)
//       for (int k = 1; k < nzc-1; k++)
//         arr[i-1][j-1][k-1]=tmp[i][j][k];
//   return arr;
// }
/*! get Magnetic Field component X array cell without the ghost cells */
// arr3_double EMfields3D::getBxc() {
//   for (int i = 1; i < nxc-1; i++)
//     for (int j = 1; j < nyc-1; j++)
//       for (int k = 1; k < nzc-1; k++)
//         arr[i-1][j-1][k-1]=Bxc[i][j][k];
//   return arr;
// }
/*! get Magnetic Field component Y array cell without the ghost cells */
// arr3_double EMfields3D::getByc() {
//   for (int i = 1; i < nxc-1; i++)
//     for (int j = 1; j < nyc-1; j++)
//       for (int k = 1; k < nzc-1; k++)
//         arr[i-1][j-1][k-1]=Byc[i][j][k];
//   return arr;
// }
/*! get Magnetic Field component Z array cell without the ghost cells */
// arr3_double EMfields3D::getBzc() {
//   for (int i = 1; i < nxc-1; i++)
//     for (int j = 1; j < nyc-1; j++)
//       for (int k = 1; k < nzc-1; k++)
//         arr[i-1][j-1][k-1]=Bzc[i][j][k];
//   return arr;
// }
/*! get species density component X array cell without the ghost cells */
// arr3_double EMfields3D::getRHOcs(int is)
//{
//   array4_double tmp(ns,nxc,nyc,nzc);
//   get_grid().interpN2C(tmp, is, rhons);
//
//   for (int i = 1; i < nxc-1; i++)
//     for (int j = 1; j < nyc-1; j++)
//       for (int k = 1; k < nzc-1; k++)
//         arr[i-1][j-1][k-1]=tmp[is][i][j][k];
//   return arr;
// }

/*! get Magnetic Field component X array species is cell without the ghost cells */
// arr3_double EMfields3D::getJxsc(int is)
//{
//   array4_double tmp(ns,nxc,nyc,nzc);
//   get_grid().interpN2C(tmp, is, Jxs);
//
//   for (int i = 1; i < nxc-1; i++)
//     for (int j = 1; j < nyc-1; j++)
//       for (int k = 1; k < nzc-1; k++)
//         arr[i-1][j-1][k-1]=tmp[is][i][j][k];
//   return arr;
// }

/*! get current component Y array species is cell without the ghost cells */
// arr3_double EMfields3D::getJysc(int is)
//{
//   array4_double tmp(ns,nxc,nyc,nzc);
//   get_grid().interpN2C(tmp, is, Jys);
//
//   for (int i = 1; i < nxc-1; i++)
//     for (int j = 1; j < nyc-1; j++)
//       for (int k = 1; k < nzc-1; k++)
//         arr[i-1][j-1][k-1]=tmp[is][i][j][k];
//   return arr;
// }
/*! get current component Z array species is cell without the ghost cells */
// arr3_double EMfields3D::getJzsc(int is)
//{
//   array4_double tmp(ns,nxc,nyc,nzc);
//   get_grid().interpN2C(tmp, is, Jzs);
//
//   for (int i = 1; i < nxc-1; i++)
//     for (int j = 1; j < nyc-1; j++)
//       for (int k = 1; k < nzc-1; k++)
//         arr[i-1][j-1][k-1]=tmp[is][i][j][k];
//   return arr;
// }

/*! get the electric field energy */
double EMfields3D::getEenergy(void)
{
  double localEenergy = 0.0;
  double totalEenergy = 0.0;
  for (int i = 1; i < nxn - 2; i++)
    for (int j = 1; j < nyn - 2; j++)
      for (int k = 1; k < nzn - 2; k++)
        localEenergy += .5 * dx * dy * dz * (Ex[i][j][k] * Ex[i][j][k] + Ey[i][j][k] * Ey[i][j][k] + Ez[i][j][k] * Ez[i][j][k]) / (FourPI);

  MPI_Allreduce(&localEenergy, &totalEenergy, 1, MPI_DOUBLE, MPI_SUM, (&get_vct())->getFieldComm());
  return (totalEenergy);
}
/*! get the magnetic field energy */
double EMfields3D::getBenergy(void)
{
  double localBenergy = 0.0;
  double totalBenergy = 0.0;
  double Bxt = 0.0;
  double Byt = 0.0;
  double Bzt = 0.0;
  for (int i = 1; i < nxn - 2; i++)
    for (int j = 1; j < nyn - 2; j++)
      for (int k = 1; k < nzn - 2; k++)
      {
        Bxt = Bxn[i][j][k] + Bx_ext[i][j][k];
        Byt = Byn[i][j][k] + By_ext[i][j][k];
        Bzt = Bzn[i][j][k] + Bz_ext[i][j][k];
        localBenergy += .5 * dx * dy * dz * (Bxt * Bxt + Byt * Byt + Bzt * Bzt) / (FourPI);
      }

  MPI_Allreduce(&localBenergy, &totalBenergy, 1, MPI_DOUBLE, MPI_SUM, (&get_vct())->getFieldComm());
  return (totalBenergy);
}

/*! get bulk kinetic energy*/
double EMfields3D::getBulkEnergy(int is)
{
  double localBenergy = 0.0;
  double totalBenergy = 0.0;
  for (int i = 1; i < nxn - 2; i++)
    for (int j = 1; j < nyn - 2; j++)
      for (int k = 1; k < nzn - 2; k++)
        // Trying to avoid division by zero. Where rho iz 0, current must be 0.
        localBenergy += (fabs(rhons[is][i][j][k]) > 1.e-20) ? (0.5 * dx * dy * dz * (Jxs[is][i][j][k] * Jxs[is][i][j][k] + Jys[is][i][j][k] * Jys[is][i][j][k] + Jzs[is][i][j][k] * Jzs[is][i][j][k]) / rhons[is][i][j][k]) : 0.0;

  MPI_Allreduce(&localBenergy, &totalBenergy, 1, MPI_DOUBLE, MPI_SUM, (&get_vct())->getFieldComm());
  return (totalBenergy / qom[is]);
}

/*! Print info about electromagnetic field */
void EMfields3D::print(void) const
{
}

/*! destructor*/
EMfields3D::~EMfields3D()
{
  delete[] qom;
  delete[] rhoINIT;
  delete[] DriftSpecies;
  freeDataType();
}

//********************************** */
//*** Initialization Methods *******/
//***********************************  **/

/*! initialize Magnetic and Electric Field with initial configuration */
void EMfields3D::init()
{
  const Collective *col = &get_col();
  const VirtualTopology3D *vct = &get_vct();
  const Grid *grid = &get_grid();

  if (restart1 == 0)
  {
    for (int i = 0; i < nxn; i++)
    {
      for (int j = 0; j < nyn; j++)
      {
        for (int k = 0; k < nzn; k++)
        {
          for (int is = 0; is < ns; is++)
          {
            rhons[is][i][j][k] = rhoINIT[is] / FourPI;
          }
          Ex[i][j][k] = 0.0;
          Ey[i][j][k] = 0.0;
          Ez[i][j][k] = 0.0;
          Bxn[i][j][k] = B0x;
          Byn[i][j][k] = B0y;
          Bzn[i][j][k] = B0z;
        }
      }
    }

    // initialize B on centers
    grid->interpN2C(Bxc, Bxn);
    grid->interpN2C(Byc, Byn);
    grid->interpN2C(Bzc, Bzn);

    for (int is = 0; is < ns; is++)
      grid->interpN2C(rhocs, is, rhons);
  }
  else
  { // READING FROM RESTART
    col->read_field_restart(vct, grid, Bxn, Byn, Bzn, Ex, Ey, Ez, &rhons, ns);

    // communicate species densities to ghost nodes
    for (int is = 0; is < ns; is++)
    {
      double ***moment0 = convert_to_arr3(rhons[is]);
      communicateNode_P(nxn, nyn, nzn, moment0, vct, this);
    }

    if (col->getCase() == "Dipole")
    {
      ConstantChargePlanet(col->getL_square(), col->getx_center_planet(), col->gety_center_planet(), col->getz_center_planet());
    }
    else if (col->getCase() == "Dipole2D")
    {
      ConstantChargePlanet2DPlaneXZ(col->getL_square(), col->getx_center_planet(), col->getz_center_planet());
    }
    // I am not sure what this open BC does, but perhaps it is responsible for energy losses in the restart? Jan 2017, Slavik.
    else if ((col->getCase().find("TaylorGreen") != std::string::npos) && (col->getCase() != "NullPoints"))
    {
      ConstantChargeOpenBC();
    }

    // communicate ghost
    communicateNodeBC(nxn, nyn, nzn, Bxn, col->bcBx[0], col->bcBx[1], col->bcBx[2], col->bcBx[3], col->bcBx[4], col->bcBx[5], vct, this);
    communicateNodeBC(nxn, nyn, nzn, Byn, col->bcBy[0], col->bcBy[1], col->bcBy[2], col->bcBy[3], col->bcBy[4], col->bcBy[5], vct, this);
    communicateNodeBC(nxn, nyn, nzn, Bzn, col->bcBz[0], col->bcBz[1], col->bcBz[2], col->bcBz[3], col->bcBz[4], col->bcBz[5], vct, this);

    // initialize B on centers
    grid->interpN2C(Bxc, Bxn);
    grid->interpN2C(Byc, Byn);
    grid->interpN2C(Bzc, Bzn);

    // communicate ghost
    communicateCenterBC(nxc, nyc, nzc, Bxc, col->bcBx[0], col->bcBx[1], col->bcBx[2], col->bcBx[3], col->bcBx[4], col->bcBx[5], vct, this);
    communicateCenterBC(nxc, nyc, nzc, Byc, col->bcBy[0], col->bcBy[1], col->bcBy[2], col->bcBy[3], col->bcBy[4], col->bcBy[5], vct, this);
    communicateCenterBC(nxc, nyc, nzc, Bzc, col->bcBz[0], col->bcBz[1], col->bcBz[2], col->bcBz[3], col->bcBz[4], col->bcBz[5], vct, this);

    // communicate E
    communicateNodeBC(nxn, nyn, nzn, Ex, col->bcEx[0], col->bcEx[1], col->bcEx[2], col->bcEx[3], col->bcEx[4], col->bcEx[5], vct, this);
    communicateNodeBC(nxn, nyn, nzn, Ey, col->bcEy[0], col->bcEy[1], col->bcEy[2], col->bcEy[3], col->bcEy[4], col->bcEy[5], vct, this);
    communicateNodeBC(nxn, nyn, nzn, Ez, col->bcEz[0], col->bcEz[1], col->bcEz[2], col->bcEz[3], col->bcEz[4], col->bcEz[5], vct, this);

    for (int is = 0; is < ns; is++)
      grid->interpN2C(rhocs, is, rhons);
  }
}

/*! initialize EM field with transverse electric waves 1D and rotate anticlockwise (theta degrees) */
void EMfields3D::initEM_rotate(double B, double theta)
{
  const Grid *grid = &get_grid();

  // initialize E and rhos on nodes
  for (int i = 0; i < nxn; i++)
    for (int j = 0; j < nyn; j++)
    {
      Ex[i][j][0] = 0.0;
      Ey[i][j][0] = 0.0;
      Ez[i][j][0] = 0.0;
      Bxn[i][j][0] = B * cos(theta * M_PI / 180);
      Byn[i][j][0] = B * sin(theta * M_PI / 180);
      Bzn[i][j][0] = 0.0;
      rhons[0][i][j][0] = 0.07957747154595; // electrons: species is now first index
      rhons[1][i][j][0] = 0.07957747154595; // protons: species is now first index
    }
  // initialize B on centers
  grid->interpN2C(Bxc, Bxn);
  grid->interpN2C(Byc, Byn);
  grid->interpN2C(Bzc, Bzn);

  for (int is = 0; is < ns; is++)
    grid->interpN2C(rhocs, is, rhons);
}

/*! initiliaze EM for GEM challange */
void EMfields3D::initGEM()
{
  const VirtualTopology3D *vct = &get_vct();
  const Grid *grid = &get_grid();
  // perturbation localized in X
  double pertX = 0.4;
  double xpert, ypert, exp_pert;
  if (restart1 == 0)
  {
    // initialize
    if (get_vct().getCartesian_rank() == 0)
    {
      cout << "------------------------------------------" << endl;
      cout << "Initialize GEM Challenge with Perturbation" << endl;
      cout << "------------------------------------------" << endl;
      cout << "B0x                              = " << B0x << endl;
      cout << "B0y                              = " << B0y << endl;
      cout << "B0z                              = " << B0z << endl;
      cout << "Delta (current sheet thickness) = " << delta << endl;
      for (int i = 0; i < ns; i++)
      {
        cout << "rho species " << i << " = " << rhoINIT[i];
        if (DriftSpecies[i])
          cout << " DRIFTING " << endl;
        else
          cout << " BACKGROUND " << endl;
      }
      cout << "-------------------------" << endl;
    }
    for (int i = 0; i < nxn; i++)
      for (int j = 0; j < nyn; j++)
        for (int k = 0; k < nzn; k++)
        {
          // initialize the density for species
          for (int is = 0; is < ns; is++)
          {
            if (DriftSpecies[is])
              rhons[is][i][j][k] = ((rhoINIT[is] / (cosh((grid->getYN(i, j, k) - Ly / 2) / delta) * cosh((grid->getYN(i, j, k) - Ly / 2) / delta)))) / FourPI;
            else
              rhons[is][i][j][k] = rhoINIT[is] / FourPI;
          }
          // electric field
          Ex[i][j][k] = 0.0;
          Ey[i][j][k] = 0.0;
          Ez[i][j][k] = 0.0;
          // Magnetic field
          Bxn[i][j][k] = B0x * tanh((grid->getYN(i, j, k) - Ly / 2) / delta);
          // add the initial GEM perturbation
          // Bxn[i][j][k] += (B0x/10.0)*(M_PI/Ly)*cos(2*M_PI*grid->getXN(i,j,k)/Lx)*sin(M_PI*(grid->getYN(i,j,k)- Ly/2)/Ly );
          Byn[i][j][k] = B0y; // - (B0x/10.0)*(2*M_PI/Lx)*sin(2*M_PI*grid->getXN(i,j,k)/Lx)*cos(M_PI*(grid->getYN(i,j,k)- Ly/2)/Ly);
          // add the initial X perturbation
          xpert = grid->getXN(i, j, k) - Lx / 2;
          ypert = grid->getYN(i, j, k) - Ly / 2;
          exp_pert = exp(-(xpert / delta) * (xpert / delta) - (ypert / delta) * (ypert / delta));
          Bxn[i][j][k] += (B0x * pertX) * exp_pert * (-cos(M_PI * xpert / 10.0 / delta) * cos(M_PI * ypert / 10.0 / delta) * 2.0 * ypert / delta - cos(M_PI * xpert / 10.0 / delta) * sin(M_PI * ypert / 10.0 / delta) * M_PI / 10.0);
          Byn[i][j][k] += (B0x * pertX) * exp_pert * (cos(M_PI * xpert / 10.0 / delta) * cos(M_PI * ypert / 10.0 / delta) * 2.0 * xpert / delta + sin(M_PI * xpert / 10.0 / delta) * cos(M_PI * ypert / 10.0 / delta) * M_PI / 10.0);
          // guide field
          Bzn[i][j][k] = B0z;
        }
    // initialize B on centers
    for (int i = 0; i < nxc; i++)
      for (int j = 0; j < nyc; j++)
        for (int k = 0; k < nzc; k++)
        {
          // Magnetic field
          Bxc[i][j][k] = B0x * tanh((grid->getYC(i, j, k) - Ly / 2) / delta);
          // add the initial GEM perturbation
          // Bxc[i][j][k] += (B0x/10.0)*(M_PI/Ly)*cos(2*M_PI*grid->getXC(i,j,k)/Lx)*sin(M_PI*(grid->getYC(i,j,k)- Ly/2)/Ly );
          Byc[i][j][k] = B0y; // - (B0x/10.0)*(2*M_PI/Lx)*sin(2*M_PI*grid->getXC(i,j,k)/Lx)*cos(M_PI*(grid->getYC(i,j,k)- Ly/2)/Ly);
          // add the initial X perturbation
          xpert = grid->getXC(i, j, k) - Lx / 2;
          ypert = grid->getYC(i, j, k) - Ly / 2;
          exp_pert = exp(-(xpert / delta) * (xpert / delta) - (ypert / delta) * (ypert / delta));
          Bxc[i][j][k] += (B0x * pertX) * exp_pert * (-cos(M_PI * xpert / 10.0 / delta) * cos(M_PI * ypert / 10.0 / delta) * 2.0 * ypert / delta - cos(M_PI * xpert / 10.0 / delta) * sin(M_PI * ypert / 10.0 / delta) * M_PI / 10.0);
          Byc[i][j][k] += (B0x * pertX) * exp_pert * (cos(M_PI * xpert / 10.0 / delta) * cos(M_PI * ypert / 10.0 / delta) * 2.0 * xpert / delta + sin(M_PI * xpert / 10.0 / delta) * cos(M_PI * ypert / 10.0 / delta) * M_PI / 10.0);
          // guide field
          Bzc[i][j][k] = B0z;
        }
    for (int is = 0; is < ns; is++)
      grid->interpN2C(rhocs, is, rhons);
  }
  else
  {
    init(); // use the fields from restart file
  }
}

void EMfields3D::initNullPoints()
{
  const VirtualTopology3D *vct = &get_vct();
  const Grid *grid = &get_grid();
  if (restart1 == 0)
  {
    if (vct->getCartesian_rank() == 0)
    {
      cout << "----------------------------------------" << endl;
      cout << "       Initialize 3D null point(s)" << endl;
      cout << "----------------------------------------" << endl;
      cout << "B0x                              = " << B0x << endl;
      cout << "B0y                              = " << B0y << endl;
      cout << "B0z                              = " << B0z << endl;
      for (int i = 0; i < ns; i++)
      {
        cout << "rho species " << i << " = " << rhoINIT[i] << endl;
      }
      cout << "Smoothing Factor = " << Smooth << endl;
      cout << "-------------------------" << endl;
    }

    for (int i = 0; i < nxn; i++)
      for (int j = 0; j < nyn; j++)
        for (int k = 0; k < nzn; k++)
        {
          // initialize the density for species
          for (int is = 0; is < ns; is++)
            rhons[is][i][j][k] = rhoINIT[is] / FourPI;

          // electric field
          Ex[i][j][k] = 0.0;
          Ey[i][j][k] = 0.0;
          Ez[i][j][k] = 0.0;
          // Magnetic field
          Bxn[i][j][k] = -B0x * cos(2. * M_PI * grid->getXN(i, j, k) / Lx) * sin(2. * M_PI * grid->getYN(i, j, k) / Ly);
          Byn[i][j][k] = B0x * cos(2. * M_PI * grid->getYN(i, j, k) / Ly) * (-2. * sin(2. * M_PI * grid->getZN(i, j, k) / Lz) + sin(2. * M_PI * grid->getXN(i, j, k) / Lx));
          Bzn[i][j][k] = 2. * B0x * cos(2. * M_PI * grid->getZN(i, j, k) / Lz) * sin(2. * M_PI * grid->getYN(i, j, k) / Ly);
        }

    for (int i = 0; i < nxc; i++)
      for (int j = 0; j < nyc; j++)
        for (int k = 0; k < nzc; k++)
        {
          Bxc[i][j][k] = -B0x * cos(2. * M_PI * grid->getXC(i, j, k) / Lx) * sin(2. * M_PI * grid->getYC(i, j, k) / Ly);
          Byc[i][j][k] = B0x * cos(2. * M_PI * grid->getYC(i, j, k) / Ly) * (-2. * sin(2. * M_PI * grid->getZC(i, j, k) / Lz) + sin(2. * M_PI * grid->getXC(i, j, k) / Lx));
          Bzc[i][j][k] = 2. * B0x * cos(2. * M_PI * grid->getZC(i, j, k) / Lz) * sin(2. * M_PI * grid->getYC(i, j, k) / Ly);
        }

    // currents are used to calculate in the Maxwell's solver
    // The ion current is equal to 0 (all current is on electrons)
    for (int i = 0; i < nxn; i++)
      for (int j = 0; j < nyn; j++)
        for (int k = 0; k < nzn; k++)
        {
          Jxs[1][i][j][k] = 0.0; // ion species is species 1
          Jys[1][i][j][k] = 0.0; // ion species is species 1
          Jzs[1][i][j][k] = 0.0; // ion species is species 1
        }

    // calculate the electron current from
    eqValue(0.0, tempXN, nxn, nyn, nzn);
    eqValue(0.0, tempYN, nxn, nyn, nzn);
    eqValue(0.0, tempZN, nxn, nyn, nzn);
    grid->curlC2N(tempXN, tempYN, tempZN, Bxc, Byc, Bzc); // here you calculate curl(B)
    // all current is on electrons, calculated from Ampere's law
    for (int i = 0; i < nxn; i++)
      for (int j = 0; j < nyn; j++)
        for (int k = 0; k < nzn; k++)
        {                                                 // electrons are species 0
          Jxs[0][i][j][k] = c * tempXN[i][j][k] / FourPI; // ion species is species 1
          Jys[0][i][j][k] = c * tempYN[i][j][k] / FourPI; // ion species is species 1
          Jzs[0][i][j][k] = c * tempZN[i][j][k] / FourPI; // ion species is species 1
        }

    for (int is = 0; is < ns; is++)
      grid->interpN2C(rhocs, is, rhons);
  }
  else
  {
    init(); // use the fields from restart file
  }
}

void EMfields3D::initTaylorGreen()
{
  const VirtualTopology3D *vct = &get_vct();
  const Grid *grid = &get_grid();
  if (restart1 == 0)
  {
    if (vct->getCartesian_rank() == 0)
    {
      cout << "----------------------------------------" << endl;
      cout << "       Initialize Taylor-Green flow     " << endl;
      cout << "----------------------------------------" << endl;
      cout << "B0                               = " << B0x << endl;
      cout << "u0                               = " << ue0 << endl;
      for (int i = 0; i < ns; i++)
      {
        cout << "rho species " << i << " = " << rhoINIT[i] << endl;
      }
      cout << "Smoothing Factor = " << Smooth << endl;
      cout << "-------------------------" << endl;
    }

    for (int i = 0; i < nxn; i++)
      for (int j = 0; j < nyn; j++)
        for (int k = 0; k < nzn; k++)
        {
          // initialize the density for species
          for (int is = 0; is < ns; is++)
          {
            rhons[is][i][j][k] = rhoINIT[is] / FourPI;

            // The flow will be initialized from currents
            Jxs[is][i][j][k] = ue0 * rhons[is][i][j][k] * sin(2. * M_PI * grid->getXC(i, j, k) / Lx) * cos(2. * M_PI * grid->getYC(i, j, k) / Ly) * cos(2. * M_PI * grid->getZC(i, j, k) / Lz);
            Jys[is][i][j][k] = -ue0 * rhons[is][i][j][k] * cos(2. * M_PI * grid->getXC(i, j, k) / Lx) * sin(2. * M_PI * grid->getYC(i, j, k) / Ly) * cos(2. * M_PI * grid->getZC(i, j, k) / Lz);
            Jzs[is][i][j][k] = 0.; // Z velocity is zero
          }

          // electric field
          Ex[i][j][k] = 0.0;
          Ey[i][j][k] = 0.0;
          Ez[i][j][k] = 0.0;
          // Magnetic field
          Bxn[i][j][k] = B0x * cos(2. * M_PI * grid->getXN(i, j, k) / Lx) * sin(2. * M_PI * grid->getYN(i, j, k) / Ly) * sin(2. * M_PI * grid->getZN(i, j, k) / Lz);
          Byn[i][j][k] = B0x * sin(2. * M_PI * grid->getXN(i, j, k) / Lx) * cos(2. * M_PI * grid->getYN(i, j, k) / Ly) * sin(2. * M_PI * grid->getZN(i, j, k) / Lz);
          Bzn[i][j][k] = -2. * B0x * sin(2. * M_PI * grid->getXN(i, j, k) / Lx) * sin(2. * M_PI * grid->getYN(i, j, k) / Ly) * cos(2. * M_PI * grid->getZN(i, j, k) / Lz);
        }

    for (int i = 0; i < nxc; i++)
      for (int j = 0; j < nyc; j++)
        for (int k = 0; k < nzc; k++)
        {
          Bxc[i][j][k] = B0x * cos(2. * M_PI * grid->getXC(i, j, k) / Lx) * sin(2. * M_PI * grid->getYC(i, j, k) / Ly) * sin(2. * M_PI * grid->getZC(i, j, k) / Lz);
          Byc[i][j][k] = B0x * sin(2. * M_PI * grid->getXC(i, j, k) / Lx) * cos(2. * M_PI * grid->getYC(i, j, k) / Ly) * sin(2. * M_PI * grid->getZC(i, j, k) / Lz);
          Bzc[i][j][k] = -2. * B0x * sin(2. * M_PI * grid->getXC(i, j, k) / Lx) * sin(2. * M_PI * grid->getYC(i, j, k) / Ly) * cos(2. * M_PI * grid->getZC(i, j, k) / Lz);
        }

    for (int is = 0; is < ns; is++)
      grid->interpN2C(rhocs, is, rhons);
  }
  else
  {
    init(); // use the fields from restart file
  }
}

void EMfields3D::initOriginalGEM()
{
  const Grid *grid = &get_grid();
  // perturbation localized in X
  if (restart1 == 0)
  {
    // initialize
    if (get_vct().getCartesian_rank() == 0)
    {
      cout << "------------------------------------------" << endl;
      cout << "Initialize GEM Challenge with Pertubation" << endl;
      cout << "------------------------------------------" << endl;
      cout << "B0x                              = " << B0x << endl;
      cout << "B0y                              = " << B0y << endl;
      cout << "B0z                              = " << B0z << endl;
      cout << "Delta (current sheet thickness) = " << delta << endl;
      for (int i = 0; i < ns; i++)
      {
        cout << "rho species " << i << " = " << rhoINIT[i];
        if (DriftSpecies[i])
          cout << " DRIFTING " << endl;
        else
          cout << " BACKGROUND " << endl;
      }
      cout << "-------------------------" << endl;
    }
    for (int i = 0; i < nxn; i++)
      for (int j = 0; j < nyn; j++)
        for (int k = 0; k < nzn; k++)
        {
          // initialize the density for species
          for (int is = 0; is < ns; is++)
          {
            if (DriftSpecies[is])
              rhons[is][i][j][k] = ((rhoINIT[is] / (cosh((grid->getYN(i, j, k) - Ly / 2) / delta) * cosh((grid->getYN(i, j, k) - Ly / 2) / delta)))) / FourPI;
            else
              rhons[is][i][j][k] = rhoINIT[is] / FourPI;
          }
          // electric field
          Ex[i][j][k] = 0.0;
          Ey[i][j][k] = 0.0;
          Ez[i][j][k] = 0.0;
          // Magnetic field
          const double yM = grid->getYN(i, j, k) - .5 * Ly;
          Bxn[i][j][k] = B0x * tanh(yM / delta);
          // add the initial GEM perturbation
          const double xM = grid->getXN(i, j, k) - .5 * Lx;
          Bxn[i][j][k] -= (B0x / 10.0) * (M_PI / Ly) * cos(2 * M_PI * xM / Lx) * sin(M_PI * yM / Ly);
          Byn[i][j][k] = B0y + (B0x / 10.0) * (2 * M_PI / Lx) * sin(2 * M_PI * xM / Lx) * cos(M_PI * yM / Ly);
          Bzn[i][j][k] = B0z;
        }
    // initialize B on centers
    for (int i = 0; i < nxc; i++)
      for (int j = 0; j < nyc; j++)
        for (int k = 0; k < nzc; k++)
        {
          // Magnetic field
          const double yM = grid->getYC(i, j, k) - .5 * Ly;
          Bxc[i][j][k] = B0x * tanh(yM / delta);
          // add the initial GEM perturbation
          const double xM = grid->getXC(i, j, k) - .5 * Lx;
          Bxc[i][j][k] -= (B0x / 10.0) * (M_PI / Ly) * cos(2 * M_PI * xM / Lx) * sin(M_PI * yM / Ly);
          Byc[i][j][k] = B0y + (B0x / 10.0) * (2 * M_PI / Lx) * sin(2 * M_PI * xM / Lx) * cos(M_PI * yM / Ly);
          Bzc[i][j][k] = B0z;
        }
    for (int is = 0; is < ns; is++)
      grid->interpN2C(rhocs, is, rhons);
  }
  else
  {
    init(); // use the fields from restart file
  }
}

void EMfields3D::initGEMDoubleHarris()
{
  const Collective *col = &get_col();
  const VirtualTopology3D *vct = &get_vct();
  const Grid *grid = &get_grid();

  double pertX = 0.4;
  double xpert, ypert, exp_pert;
  if (restart1 == 0)
  {

    if (vct->getCartesian_rank() == 0)
    {
      cout << "------------------------------------------" << endl;
      cout << "Initialize Double Harris Sheet with Perturbation" << endl;
      cout << "------------------------------------------" << endl;
      cout << "B0x                              = " << B0x << endl;
      cout << "B0y                              = " << B0y << endl;
      cout << "B0z                              = " << B0z << endl;
      cout << "Delta (current sheet thickness)  = " << delta << endl;
      for (int i = 0; i < ns; i++)
      {
        cout << "rho species " << i << " = " << rhoINIT[i];
        if (DriftSpecies[i])
          cout << " DRIFTING " << endl;
        else
          cout << " BACKGROUND " << endl;
      }
      cout << "-------------------------" << endl;
    }
    for (int i = 0; i < nxn; i++)
      for (int j = 0; j < nyn; j++)
        for (int k = 0; k < nzn; k++)
        {
          const double xM = grid->getXN(i, j, k) - 0.5 * Lx;
          const double yB = grid->getYN(i, j, k) - 0.25 * Ly;
          const double yT = grid->getYN(i, j, k) - 0.75 * Ly;
          const double yBd = yB / delta;
          const double yTd = yT / delta;
          // initialize the density for species
          for (int is = 0; is < ns; is++)
          {
            // dprintf("is=%d, DriftSpecies[is]=%d",is,DriftSpecies[is]);
            if (DriftSpecies[is])
            {
              const double sech_yBd = 1. / cosh(yBd); //+1e-5;
              const double sech_yTd = 1. / cosh(yTd); //+1e-5;
              if (is == 0 || is == 1)
                rhons[is][i][j][k] = rhoINIT[is] * sech_yBd * sech_yBd / FourPI;
              else if (is == 2 || is == 3)
                rhons[is][i][j][k] = rhoINIT[is] * sech_yTd * sech_yTd / FourPI;
            }
            else
              rhons[is][i][j][k] = rhoINIT[is] / FourPI;
          }
          // electric field
          Ex[i][j][k] = 0.0;
          Ey[i][j][k] = 0.0;
          Ez[i][j][k] = 0.0;
          // Magnetic field
          Bxn[i][j][k] = B0x * (-1.0 + tanh(yBd) - tanh(yTd));
          Byn[i][j][k] = B0y;
          Bzn[i][j][k] = B0z;
          // add the initial X perturbation
          xpert = grid->getXN(i, j, k) - Lx / 2;
          ypert = yB;
          exp_pert = exp(-(xpert / delta) * (xpert / delta) - (ypert / delta) * (ypert / delta));
          Bxn[i][j][k] += (B0x * pertX) * exp_pert * (-cos(M_PI * xpert / 10.0 / delta) * cos(M_PI * ypert / 10.0 / delta) * 2.0 * ypert / delta - cos(M_PI * xpert / 10.0 / delta) * sin(M_PI * ypert / 10.0 / delta) * M_PI / 10.0);
          Byn[i][j][k] += (B0x * pertX) * exp_pert * (cos(M_PI * xpert / 10.0 / delta) * cos(M_PI * ypert / 10.0 / delta) * 2.0 * xpert / delta + sin(M_PI * xpert / 10.0 / delta) * cos(M_PI * ypert / 10.0 / delta) * M_PI / 10.0);
        }

    // communicate ghost
    communicateNodeBC(nxn, nyn, nzn, Bxn, 1, 1, 2, 2, 1, 1, vct, this);
    communicateNodeBC(nxn, nyn, nzn, Byn, 1, 1, 1, 1, 1, 1, vct, this);
    communicateNodeBC(nxn, nyn, nzn, Bzn, 1, 1, 2, 2, 1, 1, vct, this);
    // initialize B on centers
    grid->interpN2C(Bxc, Bxn);
    grid->interpN2C(Byc, Byn);
    grid->interpN2C(Bzc, Bzn);
    // communicate ghost
    communicateCenterBC(nxc, nyc, nzc, Bxc, 2, 2, 2, 2, 2, 2, vct, this);
    communicateCenterBC(nxc, nyc, nzc, Byc, 1, 1, 1, 1, 1, 1, vct, this);
    communicateCenterBC(nxc, nyc, nzc, Bzc, 2, 2, 2, 2, 2, 2, vct, this);
    for (int is = 0; is < ns; is++)
      grid->interpN2C(rhocs, is, rhons);
  }
  else
  {
    init(); // use the fields from restart file
  }
}

void EMfields3D::initDoublePeriodicHarrisWithGaussianHumpPerturbation()
{
  const Collective *col = &get_col();
  const VirtualTopology3D *vct = &get_vct();
  const Grid *grid = &get_grid();
  // perturbation localized in X
  const double pertX = 0.4;
  const double deltax = 8. * delta;
  const double deltay = 4. * delta;
  if (restart1 == 0)
  {
    // initialize
    if (get_vct().getCartesian_rank() == 0)
    {
      cout << "------------------------------------------" << endl;
      cout << "Initialize GEM Challenge with Pertubation" << endl;
      cout << "------------------------------------------" << endl;
      cout << "B0x                              = " << B0x << endl;
      cout << "B0y                              = " << B0y << endl;
      cout << "B0z                              = " << B0z << endl;
      cout << "Delta (current sheet thickness) = " << delta << endl;
      for (int i = 0; i < ns; i++)
      {
        cout << "rho species " << i << " = " << rhoINIT[i];
        if (DriftSpecies[i])
          cout << " DRIFTING " << endl;
        else
          cout << " BACKGROUND " << endl;
      }
      cout << "-------------------------" << endl;
    }
    for (int i = 0; i < nxn; i++)
      for (int j = 0; j < nyn; j++)
        for (int k = 0; k < nzn; k++)
        {
          const double xM = grid->getXN(i, j, k) - .5 * Lx;
          const double yB = grid->getYN(i, j, k) - .25 * Ly;
          const double yT = grid->getYN(i, j, k) - .75 * Ly;
          const double yBd = yB / delta;
          const double yTd = yT / delta;
          // initialize the density for species
          for (int is = 0; is < ns; is++)
          {
            if (DriftSpecies[is])
            {
              const double sech_yBd = 1. / cosh(yBd);
              const double sech_yTd = 1. / cosh(yTd);
              rhons[is][i][j][k] = rhoINIT[is] * sech_yBd * sech_yBd / FourPI;
              rhons[is][i][j][k] += rhoINIT[is] * sech_yTd * sech_yTd / FourPI;
            }
            else
              rhons[is][i][j][k] = rhoINIT[is] / FourPI;
          }
          // electric field
          Ex[i][j][k] = 0.0;
          Ey[i][j][k] = 0.0;
          Ez[i][j][k] = 0.0;
          // Magnetic field
          Bxn[i][j][k] = B0x * (-1.0 + tanh(yBd) - tanh(yTd));
          // add the initial GEM perturbation
          Bxn[i][j][k] += 0.;
          Byn[i][j][k] = B0y;
          // add the initial X perturbation
          const double xMdx = xM / deltax;
          const double yBdy = yB / deltay;
          const double yTdy = yT / deltay;
          const double humpB = exp(-xMdx * xMdx - yBdy * yBdy);
          Bxn[i][j][k] -= (B0x * pertX) * humpB * (2.0 * yBdy);
          Byn[i][j][k] += (B0x * pertX) * humpB * (2.0 * xMdx);
          // add the second initial X perturbation
          const double humpT = exp(-xMdx * xMdx - yTdy * yTdy);
          Bxn[i][j][k] += (B0x * pertX) * humpT * (2.0 * yTdy);
          Byn[i][j][k] -= (B0x * pertX) * humpT * (2.0 * xMdx);

          // guide field
          Bzn[i][j][k] = B0z;
        }
    // communicate ghost
    communicateNodeBC(nxn, nyn, nzn, Bxn, col->bcBx[0], col->bcBx[1], col->bcBx[2], col->bcBx[3], col->bcBx[4], col->bcBx[5], vct, this);
    communicateNodeBC(nxn, nyn, nzn, Byn, col->bcBy[0], col->bcBy[1], col->bcBy[2], col->bcBy[3], col->bcBy[4], col->bcBy[5], vct, this);
    communicateNodeBC(nxn, nyn, nzn, Bzn, col->bcBz[0], col->bcBz[1], col->bcBz[2], col->bcBz[3], col->bcBz[4], col->bcBz[5], vct, this);

    // initialize B on centers
    for (int i = 0; i < nxc; i++)
      for (int j = 0; j < nyc; j++)
        for (int k = 0; k < nzc; k++)
        {
          const double xM = grid->getXN(i, j, k) - .5 * Lx;
          const double yB = grid->getYN(i, j, k) - .25 * Ly;
          const double yT = grid->getYN(i, j, k) - .75 * Ly;
          const double yBd = yB / delta;
          const double yTd = yT / delta;
          Bxc[i][j][k] = B0x * (-1.0 + tanh(yBd) - tanh(yTd));
          // add the initial GEM perturbation
          Bxc[i][j][k] += 0.;
          Byc[i][j][k] = B0y;
          // add the initial X perturbation
          const double xMdx = xM / deltax;
          const double yBdy = yB / deltay;
          const double yTdy = yT / deltay;
          const double humpB = exp(-xMdx * xMdx - yBdy * yBdy);
          Bxc[i][j][k] -= (B0x * pertX) * humpB * (2.0 * yBdy);
          Byc[i][j][k] += (B0x * pertX) * humpB * (2.0 * xMdx);
          // add the second initial X perturbation
          const double humpT = exp(-xMdx * xMdx - yTdy * yTdy);
          Bxc[i][j][k] += (B0x * pertX) * humpT * (2.0 * yTdy);
          Byc[i][j][k] -= (B0x * pertX) * humpT * (2.0 * xMdx);
          // guide field
          Bzc[i][j][k] = B0z;
        }
    // communicate ghost
    communicateCenterBC(nxc, nyc, nzc, Bxc, col->bcBx[0], col->bcBx[1], col->bcBx[2], col->bcBx[3], col->bcBx[4], col->bcBx[5], vct, this);
    communicateCenterBC(nxc, nyc, nzc, Byc, col->bcBy[0], col->bcBy[1], col->bcBy[2], col->bcBy[3], col->bcBy[4], col->bcBy[5], vct, this);
    communicateCenterBC(nxc, nyc, nzc, Bzc, col->bcBz[0], col->bcBz[1], col->bcBz[2], col->bcBz[3], col->bcBz[4], col->bcBz[5], vct, this);
    for (int is = 0; is < ns; is++)
      grid->interpN2C(rhocs, is, rhons);
  }
  else
  {
    init(); // use the fields from restart file
  }
}

void EMfields3D::initHumpPerturbation()
{
  const Collective *col = &get_col();
  const VirtualTopology3D *vct = &get_vct();
  const Grid *grid = &get_grid();
  // perturbation localized in X
  const double pertX = 0.4;
  const double deltax = 8. * delta;
  const double deltay = 4. * delta;
  if (restart1 == 0)
  {
    // initialize
    if (get_vct().getCartesian_rank() == 0)
    {
      cout << "------------------------------------------" << endl;
      cout << "Initialize with Hump Pertubation" << endl;
      cout << "------------------------------------------" << endl;
      cout << "B0x                              = " << B0x << endl;
      cout << "B0y                              = " << B0y << endl;
      cout << "B0z                              = " << B0z << endl;
      cout << "Delta                            = " << delta << endl;
      for (int i = 0; i < ns; i++)
      {
        cout << "rho species " << i << " = " << rhoINIT[i];
        if (DriftSpecies[i])
          cout << " DRIFTING " << endl;
        else
          cout << " BACKGROUND " << endl;
      }
      cout << "-------------------------" << endl;
    }
    for (int i = 0; i < nxn; i++)
      for (int j = 0; j < nyn; j++)
        for (int k = 0; k < nzn; k++)
        {
          const double xM = grid->getXN(i, j, k) - .5 * Lx;
          const double yM = grid->getYN(i, j, k) - .5 * Ly;
          const double zM = grid->getZN(i, j, k) - .5 * Lz;
          const double xMd = xM / delta;
          const double yMd = yM / delta;
          const double zMd = zM / delta;
          // initialize the density for species
          for (int is = 0; is < ns; is++)
          {
            rhons[is][i][j][k] = rhoINIT[is] / FourPI;
          }
          const double sech_xMd = 1. / cosh(xMd);
          const double sech_yMd = 1. / cosh(yMd);
          const double sech_zMd = 1. / cosh(zMd);
          // electric field
          Ex[i][j][k] = 0.0;
          Ey[i][j][k] = 0.0;
          Ez[i][j][k] = 0.0;
          // Magnetic field
          Bxn[i][j][k] = B0x * (0.5 * sech_yMd * sech_yMd * sech_zMd * sech_zMd + 1.0);
          Byn[i][j][k] = B0y * (0.5 * sech_xMd * sech_xMd * sech_zMd * sech_zMd + 1.0);
          Bzn[i][j][k] = B0z * (0.5 * sech_xMd * sech_xMd * sech_yMd * sech_yMd + 1.0);
        }
    // communicate ghost
    communicateNodeBC(nxn, nyn, nzn, Bxn, col->bcBx[0], col->bcBx[1], col->bcBx[2], col->bcBx[3], col->bcBx[4], col->bcBx[5], vct, this);
    communicateNodeBC(nxn, nyn, nzn, Byn, col->bcBy[0], col->bcBy[1], col->bcBy[2], col->bcBy[3], col->bcBy[4], col->bcBy[5], vct, this);
    communicateNodeBC(nxn, nyn, nzn, Bzn, col->bcBz[0], col->bcBz[1], col->bcBz[2], col->bcBz[3], col->bcBz[4], col->bcBz[5], vct, this);

    // initialize B on centers
    for (int i = 0; i < nxc; i++)
      for (int j = 0; j < nyc; j++)
        for (int k = 0; k < nzc; k++)
        {
          const double xM = grid->getXC(i, j, k) - .5 * Lx;
          const double yM = grid->getYC(i, j, k) - .5 * Ly;
          const double zM = grid->getZC(i, j, k) - .5 * Lz;
          const double xMd = xM / delta;
          const double yMd = yM / delta;
          const double zMd = zM / delta;
          const double sech_xMd = 1. / cosh(xMd);
          const double sech_yMd = 1. / cosh(yMd);
          const double sech_zMd = 1. / cosh(zMd);
          Bxc[i][j][k] = B0x * (0.5 * sech_yMd * sech_yMd * sech_zMd * sech_zMd + 1.0);
          Byc[i][j][k] = B0y * (0.5 * sech_xMd * sech_xMd * sech_zMd * sech_zMd + 1.0);
          Bzc[i][j][k] = B0z * (0.5 * sech_xMd * sech_xMd * sech_yMd * sech_yMd + 1.0);
        }
    // communicate ghost
    communicateCenterBC(nxc, nyc, nzc, Bxc, col->bcBx[0], col->bcBx[1], col->bcBx[2], col->bcBx[3], col->bcBx[4], col->bcBx[5], vct, this);
    communicateCenterBC(nxc, nyc, nzc, Byc, col->bcBy[0], col->bcBy[1], col->bcBy[2], col->bcBy[3], col->bcBy[4], col->bcBy[5], vct, this);
    communicateCenterBC(nxc, nyc, nzc, Bzc, col->bcBz[0], col->bcBz[1], col->bcBz[2], col->bcBz[3], col->bcBz[4], col->bcBz[5], vct, this);
    for (int is = 0; is < ns; is++)
      grid->interpN2C(rhocs, is, rhons);
  }
  else
  {
    init(); // use the fields from restart file
  }
}

/*! initialize GEM challenge with no Perturbation with dipole-like tail topology */
void EMfields3D::initGEMDipoleLikeTailNoPert()
{
  const Grid *grid = &get_grid();
  // parameters controling the field topology
  // e.g., x1=Lx/5,x2=Lx/4 give 'separated' fields, x1=Lx/4,x2=Lx/3 give 'reconnected' topology

  double x1 = Lx / 6.0;         // minimal position of the gaussian peak
  double x2 = Lx / 4.0;         // maximal position of the gaussian peak (the one closer to the center)
  double sigma = Lx / 15;       // base sigma of the gaussian - later it changes with the grid
  double stretch_curve = 2.0;   // stretch the sin^2 function over the x dimension - also can regulate the number of 'knots/reconnecitons points' if less than 1
  double skew_parameter = 0.50; // skew of the shape of the gaussian
  double pi = 3.1415927;
  double r1, r2, delta_x1x2;

  if (restart1 == 0)
  {

    // initialize
    if (get_vct().getCartesian_rank() == 0)
    {
      cout << "----------------------------------------------" << endl;
      cout << "Initialize GEM Challenge without Perturbation" << endl;
      cout << "----------------------------------------------" << endl;
      cout << "B0x                              = " << B0x << endl;
      cout << "B0y                              = " << B0y << endl;
      cout << "B0z                              = " << B0z << endl;
      cout << "Delta (current sheet thickness) = " << delta << endl;
      for (int i = 0; i < ns; i++)
      {
        cout << "rho species " << i << " = " << rhoINIT[i];
        if (DriftSpecies[i])
          cout << " DRIFTING " << endl;
        else
          cout << " BACKGROUND " << endl;
      }
      cout << "-------------------------" << endl;
    }

    for (int i = 0; i < nxn; i++)
      for (int j = 0; j < nyn; j++)
        for (int k = 0; k < nzn; k++)
        {
          // initialize the density for species
          for (int is = 0; is < ns; is++)
          {
            if (DriftSpecies[is])
              rhons[is][i][j][k] = ((rhoINIT[is] / (cosh((grid->getYN(i, j, k) - Ly / 2) / delta) * cosh((grid->getYN(i, j, k) - Ly / 2) / delta)))) / FourPI;
            else
              rhons[is][i][j][k] = rhoINIT[is] / FourPI;
          }
          // electric field
          Ex[i][j][k] = 0.0;
          Ey[i][j][k] = 0.0;
          Ez[i][j][k] = 0.0;
          // Magnetic field

          delta_x1x2 = x1 - x2 * (sin(((grid->getXN(i, j, k) - Lx / 2) / Lx * 180.0 / stretch_curve) * (0.25 * FourPI) / 180.0)) * (sin(((grid->getXN(i, j, k) - Lx / 2) / Lx * 180.0 / stretch_curve) * (0.25 * FourPI) / 180.0));

          r1 = (grid->getYN(i, j, k) - (x1 + delta_x1x2)) * (1.0 - skew_parameter * (sin(((grid->getXN(i, j, k) - Lx / 2) / Lx * 180.0) * (0.25 * FourPI) / 180.0)) * (sin(((grid->getXN(i, j, k) - Lx / 2) / Lx * 180.0) * (0.25 * FourPI) / 180.0)));
          r2 = (grid->getYN(i, j, k) - ((Lx - x1) - delta_x1x2)) * (1.0 - skew_parameter * (sin(((grid->getXN(i, j, k) - Lx / 2) / Lx * 180.0) * (0.25 * FourPI) / 180.0)) * (sin(((grid->getXN(i, j, k) - Lx / 2) / Lx * 180.0) * (0.25 * FourPI) / 180.0)));

          // tail-like field topology
          Bxn[i][j][k] = B0x * 0.5 * (-exp(-((r1) * (r1)) / (sigma * sigma)) + exp(-((r2) * (r2)) / (sigma * sigma)));

          Byn[i][j][k] = B0y;
          // guide field
          Bzn[i][j][k] = B0z;
        }
    // initialize B on centers
    for (int i = 0; i < nxc; i++)
      for (int j = 0; j < nyc; j++)
        for (int k = 0; k < nzc; k++)
        {
          // Magnetic field

          delta_x1x2 = x1 - x2 * (sin(((grid->getXC(i, j, k) - Lx / 2) / Lx * 180.0 / stretch_curve) * (0.25 * FourPI) / 180.0)) * (sin(((grid->getXC(i, j, k) - Lx / 2) / Lx * 180.0 / stretch_curve) * (0.25 * FourPI) / 180.0));

          r1 = (grid->getYC(i, j, k) - (x1 + delta_x1x2)) * (1.0 - skew_parameter * (sin(((grid->getXC(i, j, k) - Lx / 2) / Lx * 180.0) * (0.25 * FourPI) / 180.0)) * (sin(((grid->getXC(i, j, k) - Lx / 2) / Lx * 180.0) * (0.25 * FourPI) / 180.0)));
          r2 = (grid->getYC(i, j, k) - ((Lx - x1) - delta_x1x2)) * (1.0 - skew_parameter * (sin(((grid->getXC(i, j, k) - Lx / 2) / Lx * 180.0) * (0.25 * FourPI) / 180.0)) * (sin(((grid->getXC(i, j, k) - Lx / 2) / Lx * 180.0) * (0.25 * FourPI) / 180.0)));

          // tail-like field topology
          Bxc[i][j][k] = B0x * 0.5 * (-exp(-((r1) * (r1)) / (sigma * sigma)) + exp(-((r2) * (r2)) / (sigma * sigma)));

          Byc[i][j][k] = B0y;
          // guide field
          Bzc[i][j][k] = B0z;
        }
    for (int is = 0; is < ns; is++)
      grid->interpN2C(rhocs, is, rhons);
  }
  else
  {
    init(); // use the fields from restart file
  }
}

/*! initialize GEM challenge with no Perturbation */
void EMfields3D::initGEMnoPert()
{
  const Collective *col = &get_col();
  const VirtualTopology3D *vct = &get_vct();
  const Grid *grid = &get_grid();
  if (restart1 == 0)
  {

    // initialize
    if (get_vct().getCartesian_rank() == 0)
    {
      cout << "----------------------------------------------" << endl;
      cout << "Initialize GEM Challenge without Perturbation" << endl;
      cout << "----------------------------------------------" << endl;
      cout << "B0x                              = " << B0x << endl;
      cout << "B0y                              = " << B0y << endl;
      cout << "B0z                              = " << B0z << endl;
      cout << "Delta (current sheet thickness) = " << delta << endl;
      for (int i = 0; i < ns; i++)
      {
        cout << "rho species " << i << " = " << rhoINIT[i];
        if (DriftSpecies[i])
          cout << " DRIFTING " << endl;
        else
          cout << " BACKGROUND " << endl;
      }
      cout << "-------------------------" << endl;
    }
    for (int i = 0; i < nxn; i++)
      for (int j = 0; j < nyn; j++)
        for (int k = 0; k < nzn; k++)
        {
          // initialize the density for species
          for (int is = 0; is < ns; is++)
          {
            if (DriftSpecies[is])
              rhons[is][i][j][k] = ((rhoINIT[is] / (cosh((grid->getYN(i, j, k) - Ly / 2) / delta) * cosh((grid->getYN(i, j, k) - Ly / 2) / delta)))) / FourPI;
            else
              rhons[is][i][j][k] = rhoINIT[is] / FourPI;
          }
          // electric field
          Ex[i][j][k] = 0.0;
          Ey[i][j][k] = 0.0;
          Ez[i][j][k] = 0.0;
          // Magnetic field
          Bxn[i][j][k] = B0x * tanh((grid->getYN(i, j, k) - Ly / 2) / delta);
          Byn[i][j][k] = B0y;
          // guide field
          Bzn[i][j][k] = B0z;
        }
    // initialize B on centers
    for (int i = 0; i < nxc; i++)
      for (int j = 0; j < nyc; j++)
        for (int k = 0; k < nzc; k++)
        {
          // Magnetic field
          Bxc[i][j][k] = B0x * tanh((grid->getYC(i, j, k) - Ly / 2) / delta);
          Byc[i][j][k] = B0y;
          // guide field
          Bzc[i][j][k] = B0z;
        }
    for (int is = 0; is < ns; is++)
      grid->interpN2C(rhocs, is, rhons);
  }
  else
  {
    init(); // use the fields from restart file
  }
}

// new init, random problem
void EMfields3D::initRandomField()
{
  const VirtualTopology3D *vct = &get_vct();
  const Grid *grid = &get_grid();
  double **modes_seed = newArr2(double, 7, 7);
  if (restart1 == 0)
  {
    // initialize
    if (get_vct().getCartesian_rank() == 0)
    {
      cout << "------------------------------------------" << endl;
      cout << "Initialize GEM Challenge with Pertubation" << endl;
      cout << "------------------------------------------" << endl;
      cout << "B0x                              = " << B0x << endl;
      cout << "B0y                              = " << B0y << endl;
      cout << "B0z                              = " << B0z << endl;
      cout << "Delta (current sheet thickness) = " << delta << endl;
      for (int i = 0; i < ns; i++)
      {
        cout << "rho species " << i << " = " << rhoINIT[i];
        if (DriftSpecies[i])
          cout << " DRIFTING " << endl;
        else
          cout << " BACKGROUND " << endl;
      }
      cout << "-------------------------" << endl;
    }
    double kx;
    double ky;

    /*       stringstream num_proc;
       num_proc << vct->getCartesian_rank() ;
       string cqsat = SaveDirName + "/RandomNumbers" + num_proc.str() + ".txt";
        ofstream my_file(cqsat.c_str(), fstream::binary);
  for (int m=-3; m < 4; m++)
            for (int n=-3; n < 4; n++){
            modes_seed[m+3][n+3] = rand() / (double) RAND_MAX;
            my_file <<"modes_seed["<< m+3<<"][" << "\t" << n+3 << "] = " << modes_seed[m+3][n+3] << endl;
            }
              my_file.close();
    */
    modes_seed[0][0] = 0.532767;
    modes_seed[0][1] = 0.218959;
    modes_seed[0][2] = 0.0470446;
    modes_seed[0][3] = 0.678865;
    modes_seed[0][4] = 0.679296;
    modes_seed[0][5] = 0.934693;
    modes_seed[0][6] = 0.383502;
    modes_seed[1][0] = 0.519416;
    modes_seed[1][1] = 0.830965;
    modes_seed[1][2] = 0.0345721;
    modes_seed[1][3] = 0.0534616;
    modes_seed[1][4] = 0.5297;
    modes_seed[1][5] = 0.671149;
    modes_seed[1][6] = 0.00769819;
    modes_seed[2][0] = 0.383416;
    modes_seed[2][1] = 0.0668422;
    modes_seed[2][2] = 0.417486;
    modes_seed[2][3] = 0.686773;
    modes_seed[2][4] = 0.588977;
    modes_seed[2][5] = 0.930436;
    modes_seed[2][6] = 0.846167;
    modes_seed[3][0] = 0.526929;
    modes_seed[3][1] = 0.0919649;
    modes_seed[3][2] = 0.653919;
    modes_seed[3][3] = 0.415999;
    modes_seed[3][4] = 0.701191;
    modes_seed[3][5] = 0.910321;
    modes_seed[3][6] = 0.762198;
    modes_seed[4][0] = 0.262453;
    modes_seed[4][1] = 0.0474645;
    modes_seed[4][2] = 0.736082;
    modes_seed[4][3] = 0.328234;
    modes_seed[4][4] = 0.632639;
    modes_seed[4][5] = 0.75641;
    modes_seed[4][6] = 0.991037;
    modes_seed[5][0] = 0.365339;
    modes_seed[5][1] = 0.247039;
    modes_seed[5][2] = 0.98255;
    modes_seed[5][3] = 0.72266;
    modes_seed[5][4] = 0.753356;
    modes_seed[5][5] = 0.651519;
    modes_seed[5][6] = 0.0726859;
    modes_seed[6][0] = 0.631635;
    modes_seed[6][1] = 0.884707;
    modes_seed[6][2] = 0.27271;
    modes_seed[6][3] = 0.436411;
    modes_seed[6][4] = 0.766495;
    modes_seed[6][5] = 0.477732;
    modes_seed[6][6] = 0.237774;

    for (int i = 0; i < nxn; i++)
      for (int j = 0; j < nyn; j++)
        for (int k = 0; k < nzn; k++)
        {
          // initialize the density for species
          for (int is = 0; is < ns; is++)
          {
            rhons[is][i][j][k] = rhoINIT[is] / FourPI;
          }
          // electric field
          Ex[i][j][k] = 0.0;
          Ey[i][j][k] = 0.0;
          Ez[i][j][k] = 0.0;
          // Magnetic field
          Bxn[i][j][k] = 0.0;
          Byn[i][j][k] = 0.0;
          Bzn[i][j][k] = B0z;
          for (int m = -3; m < 4; m++)
            for (int n = -3; n < 4; n++)
            {

              kx = 2.0 * M_PI * m / Lx;
              ky = 2.0 * M_PI * n / Ly;
              Bxn[i][j][k] += -B0x * ky * cos(grid->getXN(i, j, k) * kx + grid->getYN(i, j, k) * ky + 2.0 * M_PI * modes_seed[m + 3][n + 3]);
              Byn[i][j][k] += B0x * kx * cos(grid->getXN(i, j, k) * kx + grid->getYN(i, j, k) * ky + 2.0 * M_PI * modes_seed[m + 3][n + 3]);
              // Bzn[i][j][k] += B0x*cos(grid->getXN(i,j,k)*kx+grid->getYN(i,j,k)*ky+2.0*M_PI*modes_seed[m+3][n+3]);
            }
        }
    // communicate ghost
    communicateNodeBC(nxn, nyn, nzn, Bxn, 1, 1, 2, 2, 1, 1, vct, this);
    communicateNodeBC(nxn, nyn, nzn, Byn, 1, 1, 1, 1, 1, 1, vct, this);
    communicateNodeBC(nxn, nyn, nzn, Bzn, 1, 1, 2, 2, 1, 1, vct, this);

    // initialize B on centers
    grid->interpN2C(Bxc, Bxn);
    grid->interpN2C(Byc, Byn);
    grid->interpN2C(Bzc, Bzn);
    // communicate ghost
    communicateCenterBC(nxc, nyc, nzc, Bxc, 2, 2, 2, 2, 2, 2, vct, this);
    communicateCenterBC(nxc, nyc, nzc, Byc, 1, 1, 1, 1, 1, 1, vct, this);
    communicateCenterBC(nxc, nyc, nzc, Bzc, 2, 2, 2, 2, 2, 2, vct, this);
    for (int is = 0; is < ns; is++)
      grid->interpN2C(rhocs, is, rhons);
  }
  else
  {
    init(); // use the fields from restart file
  }
  delArr2(modes_seed, 7);
}

/*! Init Force Free (JxB=0) */
void EMfields3D::initForceFree()
{
  const VirtualTopology3D *vct = &get_vct();
  const Grid *grid = &get_grid();
  if (restart1 == 0)
  {

    // initialize
    if (get_vct().getCartesian_rank() == 0)
    {
      cout << "----------------------------------------" << endl;
      cout << "Initialize Force Free with Perturbation" << endl;
      cout << "----------------------------------------" << endl;
      cout << "B0x                              = " << B0x << endl;
      cout << "B0y                              = " << B0y << endl;
      cout << "B0z                              = " << B0z << endl;
      cout << "Delta (current sheet thickness) = " << delta << endl;
      for (int i = 0; i < ns; i++)
      {
        cout << "rho species " << i << " = " << rhoINIT[i];
      }
      cout << "Smoothing Factor = " << Smooth << endl;
      cout << "-------------------------" << endl;
    }
    for (int i = 0; i < nxn; i++)
      for (int j = 0; j < nyn; j++)
        for (int k = 0; k < nzn; k++)
        {
          // initialize the density for species
          for (int is = 0; is < ns; is++)
          {
            rhons[is][i][j][k] = rhoINIT[is] / FourPI;
          }
          // electric field
          Ex[i][j][k] = 0.0;
          Ey[i][j][k] = 0.0;
          Ez[i][j][k] = 0.0;
          // Magnetic field
          Bxn[i][j][k] = B0x * tanh((grid->getYN(i, j, k) - Ly / 2) / delta);
          // add the initial GEM perturbation
          Bxn[i][j][k] += (B0x / 10.0) * (M_PI / Ly) * cos(2 * M_PI * grid->getXN(i, j, k) / Lx) * sin(M_PI * (grid->getYN(i, j, k) - Ly / 2) / Ly);
          Byn[i][j][k] = B0y - (B0x / 10.0) * (2 * M_PI / Lx) * sin(2 * M_PI * grid->getXN(i, j, k) / Lx) * cos(M_PI * (grid->getYN(i, j, k) - Ly / 2) / Ly);
          // guide field
          Bzn[i][j][k] = B0z / cosh((grid->getYN(i, j, k) - Ly / 2) / delta);
        }
    for (int i = 0; i < nxc; i++)
      for (int j = 0; j < nyc; j++)
        for (int k = 0; k < nzc; k++)
        {
          Bxc[i][j][k] = B0x * tanh((grid->getYC(i, j, k) - Ly / 2) / delta);
          // add the perturbation
          Bxc[i][j][k] += (B0x / 10.0) * (M_PI / Ly) * cos(2 * M_PI * grid->getXC(i, j, k) / Lx) * sin(M_PI * (grid->getYC(i, j, k) - Ly / 2) / Ly);
          Byc[i][j][k] = B0y - (B0x / 10.0) * (2 * M_PI / Lx) * sin(2 * M_PI * grid->getXC(i, j, k) / Lx) * cos(M_PI * (grid->getYC(i, j, k) - Ly / 2) / Ly);
          // guide field
          Bzc[i][j][k] = B0z / cosh((grid->getYC(i, j, k) - Ly / 2) / delta);
        }

    for (int is = 0; is < ns; is++)
      grid->interpN2C(rhocs, is, rhons);
  }
  else
  {
    init(); // use the fields from restart file
  }
}
/*! Initialize the EM field with constants values or from restart */
void EMfields3D::initBEAM(double x_center, double y_center, double z_center,
                          double radius)
{
  const Grid *grid = &get_grid();

  double distance;
  // initialize E and rhos on nodes
  if (restart1 == 0)
  {
    for (int i = 0; i < nxn; i++)
      for (int j = 0; j < nyn; j++)
        for (int k = 0; k < nzn; k++)
        {
          Ex[i][j][k] = 0.0;
          Ey[i][j][k] = 0.0;
          Ez[i][j][k] = 0.0;
          Bxn[i][j][k] = 0.0;
          Byn[i][j][k] = 0.0;
          Bzn[i][j][k] = 0.0;
          distance = (grid->getXN(i, j, k) - x_center) * (grid->getXN(i, j, k) - x_center) / (radius * radius) + (grid->getYN(i, j, k) - y_center) * (grid->getYN(i, j, k) - y_center) / (radius * radius) + (grid->getZN(i, j, k) - z_center) * (grid->getZN(i, j, k) - z_center) / (4 * radius * radius);
          // plasma
          rhons[0][i][j][k] = rhoINIT[0] / FourPI; // initialize with constant density
          // electrons
          rhons[1][i][j][k] = rhoINIT[1] / FourPI;
          // beam
          if (distance < 1.0)
            rhons[2][i][j][k] = rhoINIT[2] / FourPI;
          else
            rhons[2][i][j][k] = 0.0;
        }
    // initialize B on centers
    for (int i = 0; i < nxc; i++)
      for (int j = 0; j < nyc; j++)
        for (int k = 0; k < nzc; k++)
        {
          // Magnetic field
          Bxc[i][j][k] = 0.0;
          Byc[i][j][k] = 0.0;
          Bzc[i][j][k] = 0.0;
        }
    for (int is = 0; is < ns; is++)
      grid->interpN2C(rhocs, is, rhons);
  }
  else
  {
    init(); // use the fields from restart file
  }
}

/*! Initialise a combination of magnetic dipoles */
void EMfields3D::initDipole()
{
  const Collective *col = &get_col();
  const VirtualTopology3D *vct = &get_vct();
  const Grid *grid = &get_grid();

  // initialize
  if (vct->getCartesian_rank() == 0)
  {
    cout << "------------------------------------------" << endl;
    cout << "Initialise a Magnetic Dipole " << endl;
    cout << "------------------------------------------" << endl;
    cout << "B0x                              = " << B0x << endl;
    cout << "B0y                              = " << B0y << endl;
    cout << "B0z                              = " << B0z << endl;
    cout << "B1x   (external dipole field) - X  = " << B1x << endl;
    cout << "B1y                              = " << B1y << endl;
    cout << "B1z                              = " << B1z << endl;
    cout << "L_square - no magnetic field inside a sphere with radius L_square  = " << L_square << endl;
    cout << "Center dipole - X                = " << x_center_dipole << endl;
    cout << "Center dipole - Y                = " << y_center_dipole << endl;
    cout << "Center dipole - Z                = " << z_center_dipole << endl;
    cout << "Center planet - X                = " << x_center_planet << endl;
    cout << "Center planet - Y                = " << y_center_planet << endl;
    cout << "Center planet - Z                = " << z_center_planet << endl;
    cout << "Solar Wind drift velocity        = " << ue0 << endl;
  }

  double distance;
  double x_displ, y_displ, z_displ, fac1;

  double ebc[3];
  cross_product(ue0, ve0, we0, B0x, B0y, B0z, ebc);
  scale(ebc, -1.0, 3);

  for (int i = 0; i < nxn; i++)
  {
    for (int j = 0; j < nyn; j++)
    {
      for (int k = 0; k < nzn; k++)
      {
        for (int is = 0; is < ns; is++)
        {
          rhons[is][i][j][k] = rhoINIT[is] / FourPI;
        }
        Ex[i][j][k] = ebc[0];
        Ey[i][j][k] = ebc[1];
        Ez[i][j][k] = ebc[2];

        double blp[3];
        // radius of the planet
        double a = L_square;

        double x = grid->getXN(i, j, k);
        double y = grid->getYN(i, j, k);
        double z = grid->getZN(i, j, k);

        // Distance from planet center
        double r2 = ((x - x_center_planet) * (x - x_center_planet)) + ((y - y_center_planet) * (y - y_center_planet)) + ((z - z_center_planet) * (z - z_center_planet));
        // Distance from dipole center
        double r2d = ((x - x_center_dipole) * (x - x_center_dipole)) + ((y - y_center_dipole) * (y - y_center_dipole)) + ((z - z_center_dipole) * (z - z_center_dipole));

        // Compute dipolar field B_ext

        if (r2 > a * a)
        {
          x_displ = x - x_center_dipole; // position from the dipole center
          y_displ = y - y_center_dipole;
          z_displ = z - z_center_dipole;
          fac1 = -B1z * a * a * a / pow(r2d, 2.5);
          Bx_ext[i][j][k] = 3 * x_displ * z_displ * fac1;
          By_ext[i][j][k] = 3 * y_displ * z_displ * fac1;
          Bz_ext[i][j][k] = (2 * z_displ * z_displ - x_displ * x_displ - y_displ * y_displ) * fac1;
        }
        else
        { // no field inside the planet
          Bx_ext[i][j][k] = 0.0;
          By_ext[i][j][k] = 0.0;
          Bz_ext[i][j][k] = 0.0;
        }
        Bxn[i][j][k] = B0x; // + Bx_ext[i][j][k]
        Byn[i][j][k] = B0y; // + By_ext[i][j][k]
        Bzn[i][j][k] = B0z; // + Bz_ext[i][j][k]
      }
    }
  }

  grid->interpN2C(Bxc, Bxn);
  grid->interpN2C(Byc, Byn);
  grid->interpN2C(Bzc, Bzn);
  dprintf("1 Bzc[1][15][0]=%f", Bzc[1][15][0]);

  communicateCenterBC_P(nxc, nyc, nzc, Bxc, col->bcBx[0], col->bcBx[1], col->bcBx[2], col->bcBx[3], col->bcBx[4], col->bcBx[5], vct, this);
  communicateCenterBC_P(nxc, nyc, nzc, Byc, col->bcBy[0], col->bcBy[1], col->bcBy[2], col->bcBy[3], col->bcBy[4], col->bcBy[5], vct, this);
  communicateCenterBC_P(nxc, nyc, nzc, Bzc, col->bcBz[0], col->bcBz[1], col->bcBz[2], col->bcBz[3], col->bcBz[4], col->bcBz[5], vct, this);

  for (int is = 0; is < ns; is++)
    grid->interpN2C(rhocs, is, rhons);

  if (restart1 != 0)
  {         // EM initialization from RESTART
    init(); // use the fields from restart file
  }
}

/*! Initialise a 2D magnetic dipoles according to paper L.K.S Two-way coupling of a global Hall ....*/
void EMfields3D::initDipole2D()
{
  const Collective *col = &get_col();
  const VirtualTopology3D *vct = &get_vct();
  const Grid *grid = &get_grid();

  // initialize
  if (vct->getCartesian_rank() == 0)
  {
    cout << "------------------------------------------" << endl;
    cout << "Initialise a 2D Magnetic Dipole on XY Plane" << endl;
    cout << "------------------------------------------" << endl;
    cout << "B0x                              = " << B0x << endl;
    cout << "B0y                              = " << B0y << endl;
    cout << "B0z                              = " << B0z << endl;
    cout << "B1x   (external dipole field)    = " << B1x << endl;
    cout << "B1y                              = " << B1y << endl;
    cout << "B1z                              = " << B1z << endl;
    cout << "L_square - no magnetic field inside a sphere with radius L_square  = " << L_square << endl;
    cout << "Center dipole - X                = " << x_center_dipole << endl;
    cout << "Center dipole - Y                = " << y_center_dipole << endl;
    cout << "Center dipole - Z                = " << z_center_dipole << endl;
    cout << "Center planet - X                = " << x_center_planet << endl;
    cout << "Center planet - Y                = " << y_center_planet << endl;
    cout << "Center planet - Z                = " << z_center_planet << endl;
    cout << "Solar Wind drift velocity        = " << ue0 << endl;
    cout << "2D Smoothing Factor              = " << Smooth << endl;
    cout << "Smooth Iteration                 = " << SmoothNiter << endl;
  }

  double distance;
  double x_displ, z_displ, fac1;

  double ebc[3];
  cross_product(ue0, ve0, we0, B0x, B0y, B0z, ebc);
  scale(ebc, -1.0, 3);

  for (int i = 0; i < nxn; i++)
  {
    for (int j = 0; j < nyn; j++)
    {
      for (int k = 0; k < nzn; k++)
      {
        for (int is = 0; is < ns; is++)
        {
          rhons[is][i][j][k] = rhoINIT[is] / FourPI;
        }
        Ex[i][j][k] = ebc[0];
        Ey[i][j][k] = ebc[1];
        Ez[i][j][k] = ebc[2];

        double blp[3];
        double a = L_square;

        double xc = x_center_dipole;
        double zc = z_center_dipole;

        double x = grid->getXN(i, j, k);
        double z = grid->getZN(i, j, k);

        // Distance from planet center (for interior mask)
        double r2 = ((x - x_center_planet) * (x - x_center_planet)) + ((z - z_center_planet) * (z - z_center_planet));
        // Distance from dipole center (for field formula)
        double r2d = ((x - xc) * (x - xc)) + ((z - zc) * (z - zc));

        // Compute dipolar field B_ext

        if (r2 > a * a)
        {
          x_displ = x - xc;
          z_displ = z - zc;

          fac1 = -B1z * a * a / (r2d * r2d); // fac1 = D/4?

          Bx_ext[i][j][k] = 2 * x_displ * z_displ * fac1;
          By_ext[i][j][k] = 0.0;
          Bz_ext[i][j][k] = (z_displ * z_displ - x_displ * x_displ) * fac1;
        }
        else
        { // no field inside the planet
          Bx_ext[i][j][k] = 0.0;
          By_ext[i][j][k] = 0.0;
          Bz_ext[i][j][k] = 0.0;
        }

        Bxn[i][j][k] = B0x; // + Bx_ext[i][j][k]
        Byn[i][j][k] = B0y; // + By_ext[i][j][k]
        Bzn[i][j][k] = B0z; // + Bz_ext[i][j][k]
      }
    }
  }

  grid->interpN2C(Bxc, Bxn);
  grid->interpN2C(Byc, Byn);
  grid->interpN2C(Bzc, Bzn);

  communicateCenterBC_P(nxc, nyc, nzc, Bxc, col->bcBx[0], col->bcBx[1], col->bcBx[2], col->bcBx[3], col->bcBx[4], col->bcBx[5], vct, this);
  communicateCenterBC_P(nxc, nyc, nzc, Byc, col->bcBy[0], col->bcBy[1], col->bcBy[2], col->bcBy[3], col->bcBy[4], col->bcBy[5], vct, this);
  communicateCenterBC_P(nxc, nyc, nzc, Bzc, col->bcBz[0], col->bcBz[1], col->bcBz[2], col->bcBz[3], col->bcBz[4], col->bcBz[5], vct, this);

  for (int is = 0; is < ns; is++)
    grid->interpN2C(rhocs, is, rhons);

  if (restart1 != 0)
  {         // EM initialization from RESTART
    init(); // use the fields from restart file
  }
}

#ifdef BATSRUS
/*! initiliaze EM for GEM challange */
void EMfields3D::initBATSRUS()
{
  const Collective *col = &get_col();
  const Grid *grid = &get_grid();
  cout << "------------------------------------------" << endl;
  cout << "         Initialize from BATSRUS          " << endl;
  cout << "------------------------------------------" << endl;

  // loop over species and cell centers: fill in charge density
  for (int is = 0; is < ns; is++)
    for (int i = 0; i < nxc; i++)
      for (int j = 0; j < nyc; j++)
        for (int k = 0; k < nzc; k++)
        {
          // WARNING getFluidRhoCenter contains "case" statment
          rhocs[is][i][j][k] = col->getFluidRhoCenter(i, j, k, is);
        }

  // loop over cell centers and fill in magnetic and electric fields
  for (int i = 0; i < nxc; i++)
    for (int j = 0; j < nyc; j++)
      for (int k = 0; k < nzc; k++)
      {
        // WARNING getFluidRhoCenter contains "case" statment
        col->setFluidFieldsCenter(&Ex[i][j][k], &Ey[i][j][k], &Ez[i][j][k],
                                  &Bxc[i][j][k], &Byc[i][j][k], &Bzc[i][j][k], i, j, k);
      }

  // interpolate from cell centers to nodes (corners of cells)
  for (int is = 0; is < ns; is++)
    grid->interpC2N(rhons[is], rhocs[is]);
  grid->interpC2N(Bxn, Bxc);
  grid->interpC2N(Byn, Byc);
  grid->interpC2N(Bzn, Bzc);
}
#endif