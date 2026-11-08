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

#include "EMfields3D.h"
#include "Alloc.h"
#include "Basic.h"
#include "Collective.h"
#include "Com3DNonblk.h"
#include "Grid3DCU.h"
#include "Moments.h"
#include "Parameters.h"
#include "ParticleSoAHost.h"
#include "TimeTasks.h"
#include "VCtopology3D.h"
#include "asserts.h"
#include "debug.h"
#include "ompdefs.h"
#include "string.h"
#include <algorithm>
#include <climits>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <mpi.h>
using std::cout;
using std::endl;
using namespace iPic3D;

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
EMfields3D::EMfields3D(Collective* col, Grid* grid, VirtualTopology3D* vct)
    : _col(*col), _grid(*grid), _vct(*vct), nxc(grid->getNXC()),
      nxn(grid->getNXN()), nyc(grid->getNYC()), nyn(grid->getNYN()),
      nzc(grid->getNZC()), nzn(grid->getNZN()), dx(grid->getDX()),
      dy(grid->getDY()), dz(grid->getDZ()), invVOL(grid->getInvVOL()),
      xStart(grid->getXstart()), xEnd(grid->getXend()),
      yStart(grid->getYstart()), yEnd(grid->getYend()),
      zStart(grid->getZstart()), zEnd(grid->getZend()), Lx(col->getLx()),
      Ly(col->getLy()), Lz(col->getLz()), ns(col->getNs()), c(col->getC()),
      dt(col->getDt()), th(col->getTh()), ue0(col->getU0(0)),
      ve0(col->getV0(0)), we0(col->getW0(0)),
      x_center_dipole(col->getx_center_dipole()),
      y_center_dipole(col->gety_center_dipole()),
      z_center_dipole(col->getz_center_dipole()),
      x_center_planet(col->getx_center_planet()),
      y_center_planet(col->gety_center_planet()),
      z_center_planet(col->getz_center_planet()),
      Planet_radius(col->getPlanet_radius()),
      delt(c * th * dt), // declared after these
      //
      // array allocation: nodes
      //
      Ex(nxn, nyn, nzn), Ey(nxn, nyn, nzn), Ez(nxn, nyn, nzn),
      Bxn(nxn, nyn, nzn), Byn(nxn, nyn, nzn), Bzn(nxn, nyn, nzn),
      rhon(nxn, nyn, nzn), Jx(nxn, nyn, nzn), Jy(nxn, nyn, nzn),
      Jz(nxn, nyn, nzn),
      //
      // species-specific quantities
      //
      rhons(ns, nxn, nyn, nzn), rhocs(ns, nxc, nyc, nzc),
      Jxs(ns, nxn, nyn, nzn), Jys(ns, nxn, nyn, nzn), Jzs(ns, nxn, nyn, nzn),
      pXXsn(ns, nxn, nyn, nzn), pXYsn(ns, nxn, nyn, nzn),
      pXZsn(ns, nxn, nyn, nzn), pYYsn(ns, nxn, nyn, nzn),
      pYZsn(ns, nxn, nyn, nzn), pZZsn(ns, nxn, nyn, nzn),
      heatFlux(ns * HeatFlux::ComponentCount, nxn, nyn, nzn),

      // array allocation: central points
      //
      PHI(nxc, nyc, nzc), Bxc(nxc, nyc, nzc), Byc(nxc, nyc, nzc),
      Bzc(nxc, nyc, nzc),

      // temporary arrays of the initializations
      //
      tempXN(nxn, nyn, nzn), tempYN(nxn, nyn, nzn), tempZN(nxn, nyn, nzn),
      //
      // external and total magnetic field
      //
      Bx_ext(nxn, nyn, nzn), By_ext(nxn, nyn, nzn), Bz_ext(nxn, nyn, nzn),
      Bx_tot(nxn, nyn, nzn), By_tot(nxn, nyn, nzn), Bz_tot(nxn, nyn, nzn) {
  auto checkedKrylovLength = [&](const char* name, int components, int nx,
                                 int ny, int nz) {
    const int extents[3] = {nx - 2, ny - 2, nz - 2};
    size_t count = components;
    for (const int extent : extents) {
      if (extent <= 0 || count > static_cast<size_t>(INT_MAX) / extent) {
        std::cerr << "ERROR: invalid " << name
                  << " Krylov length for local grid " << nx << "x" << ny << "x"
                  << nz << " on rank " << _vct.getCartesian_rank() << std::endl;
        MPI_Abort(_vct.getFieldComm(), EXIT_FAILURE);
        std::abort();
      }
      count *= extent;
    }
    return static_cast<int>(count);
  };
  poissonKrylovSize_ = checkedKrylovLength("Poisson", 1, nxc, nyc, nzc);
  maxwellKrylovSize_ = checkedKrylovLength("Maxwell", 3, nxn, nyn, nzn);

  // External imposed fields
  //
  B1x = col->getB1x();
  B1y = col->getB1y();
  B1z = col->getB1z();
  Bx_ext.setall(0.);
  By_ext.setall(0.);
  Bz_ext.setall(0.);
  Bx_tot.setall(0.);
  By_tot.setall(0.);
  Bz_tot.setall(0.);
  //
  PoissonCorrection = false;
  if (col->getPoissonCorrection() == "yes") {
    PoissonCorrection = true;
    PoissonCorrectionCycle = col->getPoissonCorrectionCycle();
  }
  divBCorrection = false;
  if (col->getdivBCorrection() == "yes") {
    divBCorrection = true;
    divBCorrectionCycle = col->getdivBCorrectionCycle();
  }
  GMREStol = col->getGMREStol();
  SolverType = col->getSolverType();
  chebMaxIter = col->getChebyshevMaxIter();
  chebEigMin = col->getChebyshevEigMin();
  chebEigMax = col->getChebyshevEigMax();
  blockJacobiSweeps = col->getBlockJacobiSweeps();
  blockJacobiOmega = col->getBlockJacobiOmega();
  poissonChebMaxIter = col->getPoissonChebMaxIter();
  poissonChebRescaleEigMin = col->getPoissonChebRescaleEigMin();
  poissonChebRescaleEigMax = col->getPoissonChebRescaleEigMax();
  poissonChebEigRatio = col->getPoissonChebEigRatio();
  qom = new double[ns];
  for (int i = 0; i < ns; i++)
    qom[i] = col->getQOM(i);
  // boundary conditions: EM fields
  bcEMfaceXright = col->getBcEMfaceXright();
  bcEMfaceXleft = col->getBcEMfaceXleft();
  bcEMfaceYright = col->getBcEMfaceYright();
  bcEMfaceYleft = col->getBcEMfaceYleft();
  bcEMfaceZright = col->getBcEMfaceZright();
  bcEMfaceZleft = col->getBcEMfaceZleft();
  {
    const bool own[6] = {vct->getXleft_neighbor() == MPI_PROC_NULL,
                         vct->getXright_neighbor() == MPI_PROC_NULL,
                         vct->getYleft_neighbor() == MPI_PROC_NULL,
                         vct->getYright_neighbor() == MPI_PROC_NULL,
                         vct->getZleft_neighbor() == MPI_PROC_NULL,
                         vct->getZright_neighbor() == MPI_PROC_NULL};
    const int bcEM[6] = {bcEMfaceXleft,  bcEMfaceXright, bcEMfaceYleft,
                         bcEMfaceYright, bcEMfaceZleft,  bcEMfaceZright};
    const int bcP[6] = {col->getBcPfaceXleft(), col->getBcPfaceXright(),
                        col->getBcPfaceYleft(), col->getBcPfaceYright(),
                        col->getBcPfaceZleft(), col->getBcPfaceZright()};
    for (int f = 0; f < 6; f++) {
      conductorFace_[f] = own[f] && bcEM[f] == 0;
      // X-right (f = 1) is never an inflow face: an open X-right face keeps
      // its operator rows
      openInflowFace_[f] = own[f] && f != 1 && bcEM[f] == 2 && bcP[f] == 2;
    }
  }
  // absorbing boundary
  yes_sal = col->getYes_sal();
  n_layers_sal = col->getN_layers_sal();
  sal_tangential_B = col->getSal_tangential_B();
  // GEM challenge parameters
  B0x = col->getB0x();
  B0y = col->getB0y();
  B0z = col->getB0z();
  delta = col->getDelta();
  Smooth = col->getSmooth();
  SmoothNiter = col->getSmoothNiter();
  haloOverlap_ = col->getHaloOverlap();
  // per-species initial density and drifting (u0 or w0 != 0) species
  rhoINIT = new double[ns];
  DriftSpecies = new bool[ns];
  for (int i = 0; i < ns; i++) {
    rhoINIT[i] = col->getRHOinit(i);
    if ((fabs(col->getW0(i)) != 0) ||
        (fabs(col->getU0(i)) != 0)) // GEM and LHDI
      DriftSpecies[i] = true;
    else
      DriftSpecies[i] = false;
  }
  FourPI = 16 * atan(1.0);
  /*! Restart */
  restart1 = col->getRestart_status();

  // Define MPI Derived Data types for Center Halo Exchange
  // For face exchange on X dir
  MPI_Type_vector((nyc - 2), (nzc - 2), nzc, MPI_DOUBLE, &yzFacetypeC);
  MPI_Type_commit(&yzFacetypeC);

  // For face exchange on Y dir
  MPI_Type_create_hvector((nxc - 2), (nzc - 2), (nzc * nyc * sizeof(double)),
                          MPI_DOUBLE, &xzFacetypeC);
  MPI_Type_commit(&xzFacetypeC);

  MPI_Type_vector((nyc - 2), 1, nzc, MPI_DOUBLE, &yEdgetypeC);
  MPI_Type_commit(&yEdgetypeC);

  // For face exchangeg on Z dir
  MPI_Type_create_hvector((nxc - 2), 1, (nzc * nyc * sizeof(double)),
                          yEdgetypeC, &xyFacetypeC);
  MPI_Type_commit(&xyFacetypeC);

  // 2 yEdgeType can be merged into one message
  MPI_Type_create_hvector(2, 1, (nzc - 1) * sizeof(double), yEdgetypeC,
                          &yEdgetypeC2);
  MPI_Type_commit(&yEdgetypeC2);

  MPI_Type_contiguous((nzc - 2), MPI_DOUBLE, &zEdgetypeC);
  MPI_Type_commit(&zEdgetypeC);

  MPI_Type_create_hvector(2, (nzc - 2),
                          (nxc - 1) * (nyc * nzc) * sizeof(double), MPI_DOUBLE,
                          &zEdgetypeC2);
  MPI_Type_commit(&zEdgetypeC2);

  MPI_Type_vector((nxc - 2), 1, nyc * nzc, MPI_DOUBLE, &xEdgetypeC);
  MPI_Type_commit(&xEdgetypeC);
  MPI_Type_create_hvector(2, 1, (nyc - 1) * nzc * sizeof(double), xEdgetypeC,
                          &xEdgetypeC2);
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
  MPI_Type_create_hvector((nxn - 2), (nzn - 2), (nzn * nyn * sizeof(double)),
                          MPI_DOUBLE, &xzFacetypeN);
  MPI_Type_commit(&xzFacetypeN);

  MPI_Type_vector((nyn - 2), 1, nzn, MPI_DOUBLE, &yEdgetypeN);
  MPI_Type_commit(&yEdgetypeN);

  // For face exchangeg on Z dir
  MPI_Type_create_hvector((nxn - 2), 1, (nzn * nyn * sizeof(double)),
                          yEdgetypeN, &xyFacetypeN);
  MPI_Type_commit(&xyFacetypeN);

  // 2 yEdgeType can be merged into one message
  MPI_Type_create_hvector(2, 1, (nzn - 1) * sizeof(double), yEdgetypeN,
                          &yEdgetypeN2);
  MPI_Type_commit(&yEdgetypeN2);

  MPI_Type_contiguous((nzn - 2), MPI_DOUBLE, &zEdgetypeN);
  MPI_Type_commit(&zEdgetypeN);

  MPI_Type_create_hvector(2, (nzn - 2),
                          (nxn - 1) * (nyn * nzn) * sizeof(double), MPI_DOUBLE,
                          &zEdgetypeN2);
  MPI_Type_commit(&zEdgetypeN2);

  MPI_Type_vector((nxn - 2), 1, nyn * nzn, MPI_DOUBLE, &xEdgetypeN);
  MPI_Type_commit(&xEdgetypeN);
  MPI_Type_create_hvector(2, 1, (nyn - 1) * nzn * sizeof(double), xEdgetypeN,
                          &xEdgetypeN2);
  MPI_Type_commit(&xEdgetypeN2);

  // corner used to communicate in x direction
  int blocklengthN[] = {1, 1, 1, 1};
  int displacementsN[] = {0, nzn - 1, (nyn - 1) * nzn, nyn * nzn - 1};
  MPI_Type_indexed(4, blocklengthN, displacementsN, MPI_DOUBLE, &cornertypeN);
  MPI_Type_commit(&cornertypeN);

  if (col->getWriteMethod() == "pvtk" || col->getWriteMethod() == "nbcvtk") {
    // test Endian
    int TestEndian = 1;
    lEndFlag = *(char*)&TestEndian;

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

    MPI_Type_create_subarray(3, size, subsize, start, MPI_ORDER_C, xyzcomp,
                             &procviewXYZ);
    MPI_Type_commit(&procviewXYZ);

    MPI_Type_create_subarray(3, size, subsize, start, MPI_ORDER_C, MPI_FLOAT,
                             &procview);
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
    MPI_Type_create_subarray(3, size, subsize, start, MPI_ORDER_C, MPI_FLOAT,
                             &ghosttype);
    MPI_Type_commit(&ghosttype);
  }
}
void EMfields3D::freeDataType() {
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

  if (_col.getWriteMethod() == "pvtk" || _col.getWriteMethod() == "nbcvtk") {
    MPI_Type_free(&procview);
    MPI_Type_free(&xyzcomp);
    MPI_Type_free(&procviewXYZ);
    MPI_Type_free(&ghosttype);
  }

  gpuSolverFree();
}

void EMfields3D::adjustNonPeriodicHeatFlux(int is) {
  const VirtualTopology3D* vct = &get_vct();

  auto scaleNode = [&](int i, int j, int k) {
    for (int c = 0; c < HeatFlux::ComponentCount; ++c)
      heatFlux[HeatFlux::componentIndex(is, c)][i][j][k] *= 2;
  };

  if (vct->getXleft_neighbor_P() == MPI_PROC_NULL)
    for (int j = 1; j < nyn - 1; j++)
      for (int k = 1; k < nzn - 1; k++)
        scaleNode(1, j, k);

  if (vct->getYleft_neighbor_P() == MPI_PROC_NULL)
    for (int i = 1; i < nxn - 1; i++)
      for (int k = 1; k < nzn - 1; k++)
        scaleNode(i, 1, k);

  if (vct->getZleft_neighbor_P() == MPI_PROC_NULL)
    for (int i = 1; i < nxn - 1; i++)
      for (int j = 1; j < nyn - 1; j++)
        scaleNode(i, j, 1);

  if (vct->getXright_neighbor_P() == MPI_PROC_NULL)
    for (int j = 1; j < nyn - 1; j++)
      for (int k = 1; k < nzn - 1; k++)
        scaleNode(nxn - 2, j, k);

  if (vct->getYright_neighbor_P() == MPI_PROC_NULL)
    for (int i = 1; i < nxn - 1; i++)
      for (int k = 1; k < nzn - 1; k++)
        scaleNode(i, nyn - 2, k);

  if (vct->getZright_neighbor_P() == MPI_PROC_NULL)
    for (int i = 1; i < nxn - 1; i++)
      for (int j = 1; j < nyn - 1; j++)
        scaleNode(i, j, nzn - 2);
}

void EMfields3D::ConstantChargeOpenBC() {
  const VirtualTopology3D* vct = &get_vct();
  const Grid* grid = &get_grid();

  double ff;

  int nx = grid->getNXN();
  int ny = grid->getNYN();
  int nz = grid->getNZN();

  for (int is = 0; is < ns; is++) {

    ff = qom[is] / fabs(qom[is]);

    if (vct->getXleft_neighbor() == MPI_PROC_NULL && (bcEMfaceXleft == 2)) {
      for (int j = 0; j < ny; j++)
        for (int k = 0; k < nz; k++) {
          rhons[is][0][j][k] = ff * rhoINIT[is] / FourPI;
          rhons[is][1][j][k] = ff * rhoINIT[is] / FourPI;
          rhons[is][2][j][k] = ff * rhoINIT[is] / FourPI;
          rhons[is][3][j][k] = ff * rhoINIT[is] / FourPI;
        }
    }

    if (vct->getXright_neighbor() == MPI_PROC_NULL && (bcEMfaceXright == 2)) {
      for (int j = 0; j < ny; j++)
        for (int k = 0; k < nz; k++) {
          rhons[is][nx - 4][j][k] = ff * rhoINIT[is] / FourPI;
          rhons[is][nx - 3][j][k] = ff * rhoINIT[is] / FourPI;
          rhons[is][nx - 2][j][k] = ff * rhoINIT[is] / FourPI;
          rhons[is][nx - 1][j][k] = ff * rhoINIT[is] / FourPI;
        }
    }

    if (vct->getYleft_neighbor() == MPI_PROC_NULL && (bcEMfaceYleft == 2)) {
      for (int i = 0; i < nx; i++)
        for (int k = 0; k < nz; k++) {
          rhons[is][i][0][k] = ff * rhoINIT[is] / FourPI;
          rhons[is][i][1][k] = ff * rhoINIT[is] / FourPI;
          rhons[is][i][2][k] = ff * rhoINIT[is] / FourPI;
          rhons[is][i][3][k] = ff * rhoINIT[is] / FourPI;
        }
    }

    if (vct->getYright_neighbor() == MPI_PROC_NULL && (bcEMfaceYright == 2)) {
      for (int i = 0; i < nx; i++)
        for (int k = 0; k < nz; k++) {
          rhons[is][i][ny - 4][k] = ff * rhoINIT[is] / FourPI;
          rhons[is][i][ny - 3][k] = ff * rhoINIT[is] / FourPI;
          rhons[is][i][ny - 2][k] = ff * rhoINIT[is] / FourPI;
          rhons[is][i][ny - 1][k] = ff * rhoINIT[is] / FourPI;
        }
    }

    if (vct->getZleft_neighbor() == MPI_PROC_NULL && (bcEMfaceZleft == 2)) {
      for (int i = 0; i < nx; i++)
        for (int j = 0; j < ny; j++) {
          rhons[is][i][j][0] = ff * rhoINIT[is] / FourPI;
          rhons[is][i][j][1] = ff * rhoINIT[is] / FourPI;
          rhons[is][i][j][2] = ff * rhoINIT[is] / FourPI;
          rhons[is][i][j][3] = ff * rhoINIT[is] / FourPI;
        }
    }

    if (vct->getZright_neighbor() == MPI_PROC_NULL && (bcEMfaceZright == 2)) {
      for (int i = 0; i < nx; i++)
        for (int j = 0; j < ny; j++) {
          rhons[is][i][j][nz - 4] = ff * rhoINIT[is] / FourPI;
          rhons[is][i][j][nz - 3] = ff * rhoINIT[is] / FourPI;
          rhons[is][i][j][nz - 2] = ff * rhoINIT[is] / FourPI;
          rhons[is][i][j][nz - 1] = ff * rhoINIT[is] / FourPI;
        }
    }
  }
}

/**
 * @brief Impose the constant-charge planet model inside the 3D spherical mask.
 *
 * @param R Planet radius.
 * @param x_center Planet-center x coordinate.
 * @param y_center Planet-center y coordinate.
 * @param z_center Planet-center z coordinate.
 */
void EMfields3D::ConstantChargePlanet(double R, double x_center,
                                      double y_center, double z_center) {
  const Grid* grid = &get_grid();

  double xd;
  double yd;
  double zd;
  double ff;

  for (int is = 0; is < ns; is++) {

    ff = qom[is] / fabs(qom[is]);

#pragma omp parallel for collapse(3) private(xd, yd, zd)
    for (int i = 1; i < nxn; i++) {
      for (int j = 1; j < nyn; j++) {
        for (int k = 1; k < nzn; k++) {

          xd = grid->getXN(i, j, k) - x_center;
          yd = grid->getYN(i, j, k) - y_center;
          zd = grid->getZN(i, j, k) - z_center;

          if ((xd * xd + yd * yd + zd * zd) <= R * R) {
            rhons[is][i][j][k] = ff * rhoINIT[is] / FourPI;
          }
        }
      }
    }
  }
}

/**
 * @brief Impose the constant-charge planet model inside the 2D XZ-plane mask.
 *
 * @param R Planet radius.
 * @param x_center Planet-center x coordinate.
 * @param z_center Planet-center z coordinate.
 */
void EMfields3D::ConstantChargePlanet2DPlaneXZ(double R, double x_center,
                                               double z_center) {
  const Grid* grid = &get_grid();
  // if (get_vct().getCartesian_rank() == 0)
  // cout << "*** Constant Charge 2D Planet ***" << endl;

  assert_eq(nyn, 4);
  double xd;
  double zd;

  for (int is = 0; is < ns; is++) {
    const double sign_q = qom[is] / (fabs(qom[is]));
#pragma omp parallel for collapse(2) private(xd, zd)
    for (int i = 1; i < nxn; i++)
      for (int k = 1; k < nzn; k++) {

        xd = grid->getXN(i, 1, k) - x_center;
        zd = grid->getZN(i, 1, k) - z_center;

        if ((xd * xd + zd * zd) <= R * R) {
          rhons[is][i][1][k] = sign_q * rhoINIT[is] / FourPI;
          rhons[is][i][2][k] = sign_q * rhoINIT[is] / FourPI;
        }
      }
  }
}

void EMfields3D::communicateGhostHeatFlux(int is) {
  timeTasks_set_communicating();

  const VirtualTopology3D* vct = &get_vct();

  for (int c = 0; c < HeatFlux::ComponentCount; ++c) {
    double*** component =
        convert_to_arr3(heatFlux[HeatFlux::componentIndex(is, c)]);
    communicateInterp(nxn, nyn, nzn, component, vct, this);
  }

  adjustNonPeriodicHeatFlux(is);

  for (int c = 0; c < HeatFlux::ComponentCount; ++c) {
    double*** component =
        convert_to_arr3(heatFlux[HeatFlux::componentIndex(is, c)]);
    communicateNode_P(nxn, nyn, nzn, component, vct, this);
  }
}

/**
 * @brief Zero the aggregate moments derived from the per-species primary
 * moments.
 */
void EMfields3D::setZeroDerivedMoments() {
  const size_t nNode = (size_t)nxn * nyn * nzn * sizeof(double);
  memset(Jx.fetch_arr(), 0, nNode);
  memset(Jy.fetch_arr(), 0, nNode);
  memset(Jz.fetch_arr(), 0, nNode);
  memset(rhon.fetch_arr(), 0, nNode);
}

void EMfields3D::setZeroPrimaryMoments() {
  // All per-species nodal arrays are laid out as [ns][nxn][nyn][nzn] in one
  // contiguous allocation, so a single memset covers every species at once.
  const size_t nSpeciesNodes = (size_t)ns * nxn * nyn * nzn * sizeof(double);
  memset(rhons.fetch_arr(), 0, nSpeciesNodes);
  memset(Jxs.fetch_arr(), 0, nSpeciesNodes);
  memset(Jys.fetch_arr(), 0, nSpeciesNodes);
  memset(Jzs.fetch_arr(), 0, nSpeciesNodes);
  memset(pXXsn.fetch_arr(), 0, nSpeciesNodes);
  memset(pXYsn.fetch_arr(), 0, nSpeciesNodes);
  memset(pXZsn.fetch_arr(), 0, nSpeciesNodes);
  memset(pYYsn.fetch_arr(), 0, nSpeciesNodes);
  memset(pYZsn.fetch_arr(), 0, nSpeciesNodes);
  memset(pZZsn.fetch_arr(), 0, nSpeciesNodes);
}
/*! set to 0 all the densities fields */
void EMfields3D::setZeroDensities() {
  setZeroDerivedMoments();
  setZeroPrimaryMoments();
}

bool EMfields3D::ownsOutflowFaceXright() const {
  return get_vct().getXright_neighbor() == MPI_PROC_NULL &&
         bcEMfaceXright == 2 && get_col().getBcPfaceXright() == 3;
}

int EMfields3D::layersXright() const {
  return ownsOutflowFaceXright() ? std::max(3, n_layers_sal) : n_layers_sal;
}

int EMfields3D::fieldCopyLayersXright() const {
  return ownsOutflowFaceXright() && get_col().getOutflowLayerSolved()
             ? 0
             : layersXright();
}

// =========================================================================
//  Zero-gradient ghost node on the X-right outflow face (Maxwell operator)
// =========================================================================

int EMfields3D::operatorCellEnd() const {
  return ownsOutflowFaceXright() ? nxc - 1 : nxc - 2;
}

int EMfields3D::operatorBCXright(int code) const {
  return ownsOutflowFaceXright() ? BC_FACE_KEEP : code;
}

void EMfields3D::outflowGhostRowRules(int rule[4]) const {
  const VirtualTopology3D& vct = get_vct();
  const int self = vct.getCartesian_rank();
  const int nb[4] = {vct.getYleft_neighbor(), vct.getYright_neighbor(),
                     vct.getZleft_neighbor(), vct.getZright_neighbor()};
  for (int s = 0; s < 4; s++)
    rule[s] = nb[s] == MPI_PROC_NULL ? GHOST_ROW_KEEP
              : nb[s] == self        ? GHOST_ROW_PERIODIC
                                     : GHOST_ROW_ADJACENT;
}

/**
 * @brief Sum nodal current density over all species.
 *
 * Zeroes Jx/Jy/Jz before accumulating so the function is safe to call
 * from any context (output writers, main loop) without requiring the caller
 * to zero the buffers beforehand.
 */
void EMfields3D::sumOverSpeciesJ() {
  const int nNodes = nxn * nyn * nzn;
  double* jx = Jx.fetch_arr();
  double* jy = Jy.fetch_arr();
  double* jz = Jz.fetch_arr();

  memset(jx, 0, nNodes * sizeof(double));
  memset(jy, 0, nNodes * sizeof(double));
  memset(jz, 0, nNodes * sizeof(double));

  for (int is = 0; is < ns; is++) {
    const double* jxs = Jxs.get_arr() + is * nNodes;
    const double* jys = Jys.get_arr() + is * nNodes;
    const double* jzs = Jzs.get_arr() + is * nNodes;
    for (int n = 0; n < nNodes; n++) {
      jx[n] += jxs[n];
      jy[n] += jys[n];
      jz[n] += jzs[n];
    }
  }
}

void EMfields3D::ownedNodeEnds(int& iEnd, int& jEnd, int& kEnd) const {
  const VirtualTopology3D& vct = get_vct();
  iEnd = vct.getXright_neighbor() == MPI_PROC_NULL ? nxn - 2 : nxn - 3;
  jEnd = vct.getYright_neighbor() == MPI_PROC_NULL ? nyn - 2 : nyn - 3;
  kEnd = vct.getZright_neighbor() == MPI_PROC_NULL ? nzn - 2 : nzn - 3;
}

double EMfields3D::getLocalEenergy() {
  int iEnd, jEnd, kEnd;
  ownedNodeEnds(iEnd, jEnd, kEnd);
  double localEenergy = 0.0;
  for (int i = 1; i <= iEnd; i++)
    for (int j = 1; j <= jEnd; j++)
      for (int k = 1; k <= kEnd; k++)
        localEenergy += .5 * dx * dy * dz *
                        (Ex[i][j][k] * Ex[i][j][k] + Ey[i][j][k] * Ey[i][j][k] +
                         Ez[i][j][k] * Ez[i][j][k]) /
                        (FourPI);
  return localEenergy;
}

double EMfields3D::getLocalBenergy() {
  int iEnd, jEnd, kEnd;
  ownedNodeEnds(iEnd, jEnd, kEnd);
  double localBenergy = 0.0;
  for (int i = 1; i <= iEnd; i++)
    for (int j = 1; j <= jEnd; j++)
      for (int k = 1; k <= kEnd; k++) {
        const double Bxt = Bxn[i][j][k] + Bx_ext[i][j][k];
        const double Byt = Byn[i][j][k] + By_ext[i][j][k];
        const double Bzt = Bzn[i][j][k] + Bz_ext[i][j][k];
        localBenergy +=
            .5 * dx * dy * dz * (Bxt * Bxt + Byt * Byt + Bzt * Bzt) / (FourPI);
      }
  return localBenergy;
}

double EMfields3D::getEenergy() {
  double localEenergy = getLocalEenergy();
  double totalEenergy = 0.0;
  MPI_Allreduce(&localEenergy, &totalEenergy, 1, MPI_DOUBLE, MPI_SUM,
                (&get_vct())->getFieldComm());
  return totalEenergy;
}

double EMfields3D::getBenergy() {
  double localBenergy = getLocalBenergy();
  double totalBenergy = 0.0;
  MPI_Allreduce(&localBenergy, &totalBenergy, 1, MPI_DOUBLE, MPI_SUM,
                (&get_vct())->getFieldComm());
  return totalBenergy;
}

/*! get bulk kinetic energy*/
double EMfields3D::getBulkEnergy(int is) {
  int iEnd, jEnd, kEnd;
  ownedNodeEnds(iEnd, jEnd, kEnd);
  double localBenergy = 0.0;
  double totalBenergy = 0.0;
  for (int i = 1; i <= iEnd; i++)
    for (int j = 1; j <= jEnd; j++)
      for (int k = 1; k <= kEnd; k++)
        // Trying to avoid division by zero. Where rho iz 0, current must be 0.
        localBenergy += (fabs(rhons[is][i][j][k]) > 1.e-20)
                            ? (0.5 * dx * dy * dz *
                               (Jxs[is][i][j][k] * Jxs[is][i][j][k] +
                                Jys[is][i][j][k] * Jys[is][i][j][k] +
                                Jzs[is][i][j][k] * Jzs[is][i][j][k]) /
                               rhons[is][i][j][k])
                            : 0.0;

  MPI_Allreduce(&localBenergy, &totalBenergy, 1, MPI_DOUBLE, MPI_SUM,
                (&get_vct())->getFieldComm());
  return (totalBenergy / qom[is]);
}

/*! Print info about electromagnetic field */
void EMfields3D::print(void) const {}

/*! destructor*/
EMfields3D::~EMfields3D() {
  delete[] qom;
  delete[] rhoINIT;
  delete[] DriftSpecies;
  freeDataType();
}

//********************************** */
//*** Initialization Methods *******/
//***********************************  **/

/*! initialize Magnetic and Electric Field with initial configuration */
void EMfields3D::init() {
  const Collective* col = &get_col();
  const VirtualTopology3D* vct = &get_vct();
  const Grid* grid = &get_grid();

  if (restart1 == 0) {
#pragma omp parallel for collapse(3)
    for (int i = 0; i < nxn; i++) {
      for (int j = 0; j < nyn; j++) {
        for (int k = 0; k < nzn; k++) {
          for (int is = 0; is < ns; is++) {
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
  } else { // READING FROM RESTART
    col->read_field_restart(vct, grid, Bxn, Byn, Bzn, Ex, Ey, Ez, &rhons, ns);

    // communicate species densities to ghost nodes
    for (int is = 0; is < ns; is++) {
      double*** moment0 = convert_to_arr3(rhons[is]);
      communicateNode_P(nxn, nyn, nzn, moment0, vct, this);
    }

    if (col->getCase() == "Dipole") {
      ConstantChargePlanet(col->getPlanet_radius(), col->getx_center_planet(),
                           col->gety_center_planet(),
                           col->getz_center_planet());
    } else if (col->getCase() == "Dipole2D") {
      ConstantChargePlanet2DPlaneXZ(col->getPlanet_radius(),
                                    col->getx_center_planet(),
                                    col->getz_center_planet());
    }
    // TaylorGreen: reset the species densities on the node planes of each
    // open face (bcEMface = 2) to sign(q) rhoINIT / 4pi
    else if (col->getCase().find("TaylorGreen") != std::string::npos) {
      ConstantChargeOpenBC();
    }

    // communicate ghost
    communicateNodeBC(nxn, nyn, nzn, Bxn, col->bcBx[0], col->bcBx[1],
                      col->bcBx[2], col->bcBx[3], col->bcBx[4], col->bcBx[5],
                      vct, this);
    communicateNodeBC(nxn, nyn, nzn, Byn, col->bcBy[0], col->bcBy[1],
                      col->bcBy[2], col->bcBy[3], col->bcBy[4], col->bcBy[5],
                      vct, this);
    communicateNodeBC(nxn, nyn, nzn, Bzn, col->bcBz[0], col->bcBz[1],
                      col->bcBz[2], col->bcBz[3], col->bcBz[4], col->bcBz[5],
                      vct, this);

    // initialize B on centers
    grid->interpN2C(Bxc, Bxn);
    grid->interpN2C(Byc, Byn);
    grid->interpN2C(Bzc, Bzn);

    // communicate ghost
    communicateCenterBC(nxc, nyc, nzc, Bxc, col->bcBx[0], col->bcBx[1],
                        col->bcBx[2], col->bcBx[3], col->bcBx[4], col->bcBx[5],
                        vct, this);
    communicateCenterBC(nxc, nyc, nzc, Byc, col->bcBy[0], col->bcBy[1],
                        col->bcBy[2], col->bcBy[3], col->bcBy[4], col->bcBy[5],
                        vct, this);
    communicateCenterBC(nxc, nyc, nzc, Bzc, col->bcBz[0], col->bcBz[1],
                        col->bcBz[2], col->bcBz[3], col->bcBz[4], col->bcBz[5],
                        vct, this);

    // communicate E
    communicateNodeBC(nxn, nyn, nzn, Ex, col->bcEx[0], col->bcEx[1],
                      col->bcEx[2], col->bcEx[3], col->bcEx[4], col->bcEx[5],
                      vct, this);
    communicateNodeBC(nxn, nyn, nzn, Ey, col->bcEy[0], col->bcEy[1],
                      col->bcEy[2], col->bcEy[3], col->bcEy[4], col->bcEy[5],
                      vct, this);
    communicateNodeBC(nxn, nyn, nzn, Ez, col->bcEz[0], col->bcEz[1],
                      col->bcEz[2], col->bcEz[3], col->bcEz[4], col->bcEz[5],
                      vct, this);

    for (int is = 0; is < ns; is++) {
      grid->interpN2C(rhocs, is, rhons);
      communicateCenterBC_P(nxc, nyc, nzc, getRHOcs(is), 2, 2, 2, 2, 2, 2, vct,
                            this);
    }
  }
}

/*! initialize EM field with transverse electric waves 1D and rotate
 * anticlockwise (theta degrees) */
void EMfields3D::initEM_rotate(double B, double theta) {
  const Grid* grid = &get_grid();

  // initialize E and rhos on nodes
  for (int i = 0; i < nxn; i++)
    for (int j = 0; j < nyn; j++) {
      Ex[i][j][0] = 0.0;
      Ey[i][j][0] = 0.0;
      Ez[i][j][0] = 0.0;
      Bxn[i][j][0] = B * cos(theta * M_PI / 180);
      Byn[i][j][0] = B * sin(theta * M_PI / 180);
      Bzn[i][j][0] = 0.0;
      rhons[0][i][j][0] = 0.07957747154595; // species 0 (electrons)
      rhons[1][i][j][0] = 0.07957747154595; // species 1 (protons)
    }
  // initialize B on centers
  grid->interpN2C(Bxc, Bxn);
  grid->interpN2C(Byc, Byn);
  grid->interpN2C(Bzc, Bzn);

  for (int is = 0; is < ns; is++)
    grid->interpN2C(rhocs, is, rhons);
}

/*!Add a periodic perturbation cos(kx x + ky y + phase) to the densities of
 * species 0 and 1; deltaBoB is the ratio (Delta B / B0), and Bx_mod, By_mod,
 * Bz_mod set only the normalisation |B_mod| */
void EMfields3D::AddPerturbationRho(double deltaBoB, double kx, double ky,
                                    double Bx_mod, double By_mod, double Bz_mod,
                                    double ne_mod, double ne_phase,
                                    double ni_mod, double ni_phase, double B0,
                                    Grid* grid) {

  double alpha;
  alpha =
      deltaBoB * B0 / sqrt(Bx_mod * Bx_mod + By_mod * By_mod + Bz_mod * Bz_mod);

  ne_mod *= alpha;
  ni_mod *= alpha;
  // cout<<" ne="<<ne_mod<<" ni="<<ni_mod<<" alpha="<<alpha<<endl;
  for (int i = 0; i < nxn; i++)
    for (int j = 0; j < nyn; j++) {
      rhons[0][i][j][0] += ne_mod * cos(kx * grid->getXN(i, j, 0) +
                                        ky * grid->getYN(i, j, 0) + ne_phase);
      rhons[1][i][j][0] += ni_mod * cos(kx * grid->getXN(i, j, 0) +
                                        ky * grid->getYN(i, j, 0) + ni_phase);
    }

  for (int is = 0; is < ns; is++)
    grid->interpN2C(rhocs, is, rhons);
}

/*!Add a periodic perturbation cos(kx x + ky y + phase) to E and B; deltaBoB
 * is the ratio (Delta B / B0) */
void EMfields3D::AddPerturbation(double deltaBoB, double kx, double ky,
                                 double Ex_mod, double Ex_phase, double Ey_mod,
                                 double Ey_phase, double Ez_mod,
                                 double Ez_phase, double Bx_mod,
                                 double Bx_phase, double By_mod,
                                 double By_phase, double Bz_mod,
                                 double Bz_phase, double B0, Grid* grid) {

  double alpha;

  alpha =
      deltaBoB * B0 / sqrt(Bx_mod * Bx_mod + By_mod * By_mod + Bz_mod * Bz_mod);

  Ex_mod *= alpha;
  Ey_mod *= alpha;
  Ez_mod *= alpha;
  Bx_mod *= alpha;
  By_mod *= alpha;
  Bz_mod *= alpha;

  for (int i = 0; i < nxn; i++)
    for (int j = 0; j < nyn; j++) {
      Ex[i][j][0] += Ex_mod * cos(kx * grid->getXN(i, j, 0) +
                                  ky * grid->getYN(i, j, 0) + Ex_phase);
      Ey[i][j][0] += Ey_mod * cos(kx * grid->getXN(i, j, 0) +
                                  ky * grid->getYN(i, j, 0) + Ey_phase);
      Ez[i][j][0] += Ez_mod * cos(kx * grid->getXN(i, j, 0) +
                                  ky * grid->getYN(i, j, 0) + Ez_phase);
      Bxn[i][j][0] += Bx_mod * cos(kx * grid->getXN(i, j, 0) +
                                   ky * grid->getYN(i, j, 0) + Bx_phase);
      Byn[i][j][0] += By_mod * cos(kx * grid->getXN(i, j, 0) +
                                   ky * grid->getYN(i, j, 0) + By_phase);
      Bzn[i][j][0] += Bz_mod * cos(kx * grid->getXN(i, j, 0) +
                                   ky * grid->getYN(i, j, 0) + Bz_phase);
    }

  // initialize B on centers
  grid->interpN2C(Bxc, Bxn);
  grid->interpN2C(Byc, Byn);
  grid->interpN2C(Bzc, Bzn);
}

/**
 * @brief Initialize the standard GEM challenge field configuration.
 */
void EMfields3D::initGEM() {
  const VirtualTopology3D* vct = &get_vct();
  const Grid* grid = &get_grid();
  // perturbation localized in X
  double pertX = 0.4;
  double xpert, ypert, exp_pert;
  if (restart1 == 0) {
    // initialize
    if (get_vct().getCartesian_rank() == 0) {
      cout << "------------------------------------------" << endl;
      cout << "Initialize GEM Challenge with Perturbation" << endl;
      cout << "------------------------------------------" << endl;
      cout << "B0x                              = " << B0x << endl;
      cout << "B0y                              = " << B0y << endl;
      cout << "B0z                              = " << B0z << endl;
      cout << "Delta (current sheet thickness) = " << delta << endl;
      for (int i = 0; i < ns; i++) {
        cout << "rho species " << i << " = " << rhoINIT[i];
        if (DriftSpecies[i])
          cout << " DRIFTING " << endl;
        else
          cout << " BACKGROUND " << endl;
      }
      cout << "-------------------------" << endl;
    }
#pragma omp parallel for collapse(3) private(exp_pert, xpert, ypert)
    for (int i = 0; i < nxn; i++)
      for (int j = 0; j < nyn; j++)
        for (int k = 0; k < nzn; k++) {
          // initialize the density for species
          for (int is = 0; is < ns; is++) {
            if (DriftSpecies[is])
              rhons[is][i][j][k] =
                  ((rhoINIT[is] /
                    (cosh((grid->getYN(i, j, k) - Ly / 2) / delta) *
                     cosh((grid->getYN(i, j, k) - Ly / 2) / delta)))) /
                  FourPI;
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
          // add the initial X perturbation
          xpert = grid->getXN(i, j, k) - Lx / 2;
          ypert = grid->getYN(i, j, k) - Ly / 2;
          exp_pert = exp(-(xpert / delta) * (xpert / delta) -
                         (ypert / delta) * (ypert / delta));
          Bxn[i][j][k] +=
              (B0x * pertX) * exp_pert *
              (-cos(M_PI * xpert / 10.0 / delta) *
                   cos(M_PI * ypert / 10.0 / delta) * 2.0 * ypert / delta -
               cos(M_PI * xpert / 10.0 / delta) *
                   sin(M_PI * ypert / 10.0 / delta) * M_PI / 10.0);
          Byn[i][j][k] +=
              (B0x * pertX) * exp_pert *
              (cos(M_PI * xpert / 10.0 / delta) *
                   cos(M_PI * ypert / 10.0 / delta) * 2.0 * xpert / delta +
               sin(M_PI * xpert / 10.0 / delta) *
                   cos(M_PI * ypert / 10.0 / delta) * M_PI / 10.0);
          // guide field
          Bzn[i][j][k] = B0z;
        }
    // initialize B on centers
#pragma omp parallel for collapse(3) private(exp_pert, xpert, ypert)
    for (int i = 0; i < nxc; i++)
      for (int j = 0; j < nyc; j++)
        for (int k = 0; k < nzc; k++) {
          // Magnetic field
          Bxc[i][j][k] = B0x * tanh((grid->getYC(i, j, k) - Ly / 2) / delta);
          Byc[i][j][k] = B0y;
          // add the initial X perturbation
          xpert = grid->getXC(i, j, k) - Lx / 2;
          ypert = grid->getYC(i, j, k) - Ly / 2;
          exp_pert = exp(-(xpert / delta) * (xpert / delta) -
                         (ypert / delta) * (ypert / delta));
          Bxc[i][j][k] +=
              (B0x * pertX) * exp_pert *
              (-cos(M_PI * xpert / 10.0 / delta) *
                   cos(M_PI * ypert / 10.0 / delta) * 2.0 * ypert / delta -
               cos(M_PI * xpert / 10.0 / delta) *
                   sin(M_PI * ypert / 10.0 / delta) * M_PI / 10.0);
          Byc[i][j][k] +=
              (B0x * pertX) * exp_pert *
              (cos(M_PI * xpert / 10.0 / delta) *
                   cos(M_PI * ypert / 10.0 / delta) * 2.0 * xpert / delta +
               sin(M_PI * xpert / 10.0 / delta) *
                   cos(M_PI * ypert / 10.0 / delta) * M_PI / 10.0);
          // guide field
          Bzc[i][j][k] = B0z;
        }
    for (int is = 0; is < ns; is++)
      grid->interpN2C(rhocs, is, rhons);
  } else {
    init(); // use the fields from restart file
  }
}

void EMfields3D::initSpeciesCurrents(bool fromAmpere, bool varyingThermal) {
  const Collective* col = &get_col();
  const VirtualTopology3D* vct = &get_vct();
  const Grid* grid = &get_grid();

  // Zero moment arrays so particle init never reads uninitialized memory.
  // Each per-species array is one contiguous [ns][nxn][nyn][nzn] allocation.
  const size_t nSpeciesNodes = (size_t)ns * nxn * nyn * nzn * sizeof(double);
  for (double* a : {Jxs.fetch_arr(), Jys.fetch_arr(), Jzs.fetch_arr(),
                    pXXsn.fetch_arr(), pXYsn.fetch_arr(), pXZsn.fetch_arr(),
                    pYYsn.fetch_arr(), pYZsn.fetch_arr(), pZZsn.fetch_arr()})
    memset(a, 0, nSpeciesNodes);

  if (!fromAmpere)
    return;

  // J = (c/4pi) curl(B) distributed by u0/v0/w0 weights
  eqValue(0.0, tempXN, nxn, nyn, nzn);
  eqValue(0.0, tempYN, nxn, nyn, nzn);
  eqValue(0.0, tempZN, nxn, nyn, nzn);
  grid->curlC2N(tempXN, tempYN, tempZN, Bxc, Byc, Bzc);

  // --- Charge-weighted normalization: sign(q)*u0 sums to N_u (v0, w0
  //     alike), so f_s = sign(q_s)*u0_s / N_u gives sum(f_s) = 1 and the
  //     drift u0_s J_x / (N_u rho_s). ---
  double sumU = 0.0, sumV = 0.0, sumW = 0.0;
  for (int is = 0; is < ns; is++) {
    const double qs = (col->getQOM(is) > 0.0) ? 1.0 : -1.0;
    sumU += qs * col->getU0(is);
    sumV += qs * col->getV0(is);
    sumW += qs * col->getW0(is);
  }
  if (vct->getCartesian_rank() == 0) {
    cout << "Species currents from Ampere, charge-weighted sums: "
         << "N_u=" << sumU << "  N_v=" << sumV << "  N_w=" << sumW << endl;
  }

  // --- Reference pressure state for spatially varying thermal velocity ---
  if (varyingThermal && vct->getCartesian_rank() == 0)
    cout << "Building reference pressure state (spatiallyVaryingThermal=1)"
         << endl;

  for (int is = 0; is < ns; is++) {
    const double qs = (col->getQOM(is) > 0.0) ? 1.0 : -1.0;
    const double wU = (sumU != 0.0) ? qs * col->getU0(is) / sumU : 0.0;
    const double wV = (sumV != 0.0) ? qs * col->getV0(is) / sumV : 0.0;
    const double wW = (sumW != 0.0) ? qs * col->getW0(is) / sumW : 0.0;
    const double factU = wU * c / FourPI;
    const double factV = wV * c / FourPI;
    const double factW = wW * c / FourPI;
    const double uthS2 = col->getUth(is) * col->getUth(is);
    const double vthS2 = col->getVth(is) * col->getVth(is);
    const double wthS2 = col->getWth(is) * col->getWth(is);

#pragma omp parallel for collapse(3)
    for (int i = 0; i < nxn; i++)
      for (int j = 0; j < nyn; j++)
        for (int k = 0; k < nzn; k++) {
          Jxs[is][i][j][k] = factU * tempXN[i][j][k];
          Jys[is][i][j][k] = factV * tempYN[i][j][k];
          Jzs[is][i][j][k] = factW * tempZN[i][j][k];
        }

    if (varyingThermal) {
#pragma omp parallel for collapse(3)
      for (int i = 0; i < nxn; i++)
        for (int j = 0; j < nyn; j++)
          for (int k = 0; k < nzn; k++) {
            const double rho = rhons[is][i][j][k];
            const double Jx = Jxs[is][i][j][k];
            const double Jy = Jys[is][i][j][k];
            const double Jz = Jzs[is][i][j][k];
            pXXsn[is][i][j][k] = Jx * Jx / rho + uthS2 * rho;
            pYYsn[is][i][j][k] = Jy * Jy / rho + vthS2 * rho;
            pZZsn[is][i][j][k] = Jz * Jz / rho + wthS2 * rho;
          }
    }
  }
}

void EMfields3D::initGEMHarris() {
  const Collective* col = &get_col();
  const VirtualTopology3D* vct = &get_vct();
  const Grid* grid = &get_grid();

  // --- configurable perturbation parameters ---
  const double pertGEM = col->getPertGEM();
  const double pertHump = col->getPertHump();
  const double deltax = col->getDeltaxHump() * delta;
  const double deltay = col->getDeltayHump() * delta;
  const int ampere = col->getCurrentFromAmpere();

  // --- wavenumbers ---
  const double kxG = 2.0 * M_PI / Lx; // pertGEM tearing-mode kx
  const double kyG = 2.0 * M_PI / Ly; // pertGEM tearing-mode ky
  const double kx =
      (col->getKxHump() < 0.0) ? 2.0 * M_PI / Lx : col->getKxHump();
  const double kyH = (col->getKyHump() < 0.0) ? M_PI / Ly : col->getKyHump();
  const double A0 = (pertHump != 0.0) ? pertHump * B0x / kyH : 0.0;

  if (restart1 == 0) {
    if (vct->getCartesian_rank() == 0) {
      cout << "---------------------------------------------------" << endl;
      cout << "       Initialize GEM Harris                       " << endl;
      cout << "---------------------------------------------------" << endl;
      cout << "B0x                = " << B0x << endl;
      cout << "B0y                = " << B0y << endl;
      cout << "B0z                = " << B0z << endl;
      cout << "delta              = " << delta << endl;
      cout << "pertGEM            = " << pertGEM << endl;
      cout << "pertHump           = " << pertHump << endl;
      if (pertHump != 0.0) {
        cout << "deltaxHump         = " << col->getDeltaxHump() << " (physical "
             << deltax << ")" << endl;
        cout << "deltayHump         = " << col->getDeltayHump() << " (physical "
             << deltay << ")" << endl;
        cout << "kxHump             = " << kx << endl;
        cout << "kyHump             = " << kyH << endl;
      }
      cout << "currentFromAmpere  = " << ampere << endl;
      for (int i = 0; i < ns; i++) {
        cout << "rho species " << i << " = " << rhoINIT[i];
        if (DriftSpecies[i])
          cout << "  DRIFTING" << endl;
        else
          cout << "  BACKGROUND" << endl;
      }
      cout << "---------------------------------------------------" << endl;
    }

    // === B and density on nodes ===
#pragma omp parallel for collapse(3)
    for (int i = 0; i < nxn; i++)
      for (int j = 0; j < nyn; j++)
        for (int k = 0; k < nzn; k++) {
          const double xM = grid->getXN(i, j, k) - 0.5 * Lx;
          const double yM = grid->getYN(i, j, k) - 0.5 * Ly;

          // --- density ---
          for (int is = 0; is < ns; is++) {
            if (DriftSpecies[is]) {
              const double sech = 1.0 / cosh(yM / delta);
              rhons[is][i][j][k] = rhoINIT[is] * sech * sech / FourPI;
            } else
              rhons[is][i][j][k] = rhoINIT[is] / FourPI;
          }

          // --- E = 0 ---
          Ex[i][j][k] = 0.0;
          Ey[i][j][k] = 0.0;
          Ez[i][j][k] = 0.0;

          // === Harris equilibrium ===
          Bxn[i][j][k] = B0x * tanh(yM / delta);
          Byn[i][j][k] = B0y;
          Bzn[i][j][k] = B0z;

          // === GEM perturbation (from vector potential, div-free) ===
          if (pertGEM != 0.0) {
            Bxn[i][j][k] +=
                -pertGEM * B0x * (Lx / Ly) * cos(kxG * xM) * sin(kyG * yM);
            Byn[i][j][k] += pertGEM * B0x * sin(kxG * xM) * cos(kyG * yM);
          }

          // === Hump perturbation (from vector potential, div-free) ===
          if (pertHump != 0.0) {
            const double g =
                exp(-xM * xM / (deltax * deltax) - yM * yM / (deltay * deltay));
            const double Cx = cos(kx * xM);
            const double Sx = sin(kx * xM);
            const double Cy = cos(kyH * yM);
            const double Sy = sin(kyH * yM);

            // dAz/dy  ->  delta Bx
            Bxn[i][j][k] +=
                A0 * g * Cx * (-2.0 * yM / (deltay * deltay) * Cy - kyH * Sy);
            // -dAz/dx ->  delta By
            Byn[i][j][k] +=
                A0 * g * Cy * (2.0 * xM / (deltax * deltax) * Cx + kx * Sx);
          }
        }

    // --- ghost communication on nodes ---
    communicateNodeBC(nxn, nyn, nzn, Bxn, col->bcBx[0], col->bcBx[1],
                      col->bcBx[2], col->bcBx[3], col->bcBx[4], col->bcBx[5],
                      vct, this);
    communicateNodeBC(nxn, nyn, nzn, Byn, col->bcBy[0], col->bcBy[1],
                      col->bcBy[2], col->bcBy[3], col->bcBy[4], col->bcBy[5],
                      vct, this);
    communicateNodeBC(nxn, nyn, nzn, Bzn, col->bcBz[0], col->bcBz[1],
                      col->bcBz[2], col->bcBz[3], col->bcBz[4], col->bcBz[5],
                      vct, this);

    // --- B on centers via interpolation ---
    grid->interpN2C(Bxc, Bxn);
    grid->interpN2C(Byc, Byn);
    grid->interpN2C(Bzc, Bzn);

    // --- ghost communication on centers ---
    communicateCenterBC(nxc, nyc, nzc, Bxc, col->bcBx[0], col->bcBx[1],
                        col->bcBx[2], col->bcBx[3], col->bcBx[4], col->bcBx[5],
                        vct, this);
    communicateCenterBC(nxc, nyc, nzc, Byc, col->bcBy[0], col->bcBy[1],
                        col->bcBy[2], col->bcBy[3], col->bcBy[4], col->bcBy[5],
                        vct, this);
    communicateCenterBC(nxc, nyc, nzc, Bzc, col->bcBz[0], col->bcBz[1],
                        col->bcBz[2], col->bcBz[3], col->bcBz[4], col->bcBz[5],
                        vct, this);

    // --- density on centers ---
    for (int is = 0; is < ns; is++)
      grid->interpN2C(rhocs, is, rhons);

    // --- current initialization ---
    initSpeciesCurrents(ampere, col->getSpatiallyVaryingThermal());
    // without Ampere: Jxs/Jys/Jzs stay at zero and particle init uses each
    // species' uniform u0/v0/w0 as drift velocities
  } else {
    init(); // restart
  }
}

void EMfields3D::initNullPoints() {
  const VirtualTopology3D* vct = &get_vct();
  const Grid* grid = &get_grid();
  if (restart1 == 0) {
    if (vct->getCartesian_rank() == 0) {
      cout << "----------------------------------------" << endl;
      cout << "       Initialize 3D null point(s)" << endl;
      cout << "----------------------------------------" << endl;
      cout << "B0x                              = " << B0x << endl;
      cout << "B0y                              = " << B0y << endl;
      cout << "B0z                              = " << B0z << endl;
      for (int i = 0; i < ns; i++) {
        cout << "rho species " << i << " = " << rhoINIT[i] << endl;
      }
      cout << "Smoothing Factor = " << Smooth << endl;
      cout << "-------------------------" << endl;
    }

#pragma omp parallel for collapse(3)
    for (int i = 0; i < nxn; i++)
      for (int j = 0; j < nyn; j++)
        for (int k = 0; k < nzn; k++) {
          // initialize the density for species
          for (int is = 0; is < ns; is++)
            rhons[is][i][j][k] = rhoINIT[is] / FourPI;

          // electric field
          Ex[i][j][k] = 0.0;
          Ey[i][j][k] = 0.0;
          Ez[i][j][k] = 0.0;
          // Magnetic field
          Bxn[i][j][k] = -B0x * cos(2. * M_PI * grid->getXN(i, j, k) / Lx) *
                         sin(2. * M_PI * grid->getYN(i, j, k) / Ly);
          Byn[i][j][k] = B0x * cos(2. * M_PI * grid->getYN(i, j, k) / Ly) *
                         (-2. * sin(2. * M_PI * grid->getZN(i, j, k) / Lz) +
                          sin(2. * M_PI * grid->getXN(i, j, k) / Lx));
          Bzn[i][j][k] = 2. * B0x * cos(2. * M_PI * grid->getZN(i, j, k) / Lz) *
                         sin(2. * M_PI * grid->getYN(i, j, k) / Ly);
        }

#pragma omp parallel for collapse(3)
    for (int i = 0; i < nxc; i++)
      for (int j = 0; j < nyc; j++)
        for (int k = 0; k < nzc; k++) {
          Bxc[i][j][k] = -B0x * cos(2. * M_PI * grid->getXC(i, j, k) / Lx) *
                         sin(2. * M_PI * grid->getYC(i, j, k) / Ly);
          Byc[i][j][k] = B0x * cos(2. * M_PI * grid->getYC(i, j, k) / Ly) *
                         (-2. * sin(2. * M_PI * grid->getZC(i, j, k) / Lz) +
                          sin(2. * M_PI * grid->getXC(i, j, k) / Lx));
          Bzc[i][j][k] = 2. * B0x * cos(2. * M_PI * grid->getZC(i, j, k) / Lz) *
                         sin(2. * M_PI * grid->getYC(i, j, k) / Ly);
        }

    // Species currents read by maxwellianFromSpeciesCurrents for the particle
    // drifts; the ion current (species 1) is 0, all current is on electrons
#pragma omp parallel for collapse(3)
    for (int i = 0; i < nxn; i++)
      for (int j = 0; j < nyn; j++)
        for (int k = 0; k < nzn; k++) {
          Jxs[1][i][j][k] = 0.0; // ion species is species 1
          Jys[1][i][j][k] = 0.0; // ion species is species 1
          Jzs[1][i][j][k] = 0.0; // ion species is species 1
        }

    // electron current from Ampere's law, J = (c/4pi) curl(B)
    eqValue(0.0, tempXN, nxn, nyn, nzn);
    eqValue(0.0, tempYN, nxn, nyn, nzn);
    eqValue(0.0, tempZN, nxn, nyn, nzn);
    grid->curlC2N(tempXN, tempYN, tempZN, Bxc, Byc,
                  Bzc); // here you calculate curl(B)
    // all current is on electrons, calculated from Ampere's law
#pragma omp parallel for collapse(3)
    for (int i = 0; i < nxn; i++)
      for (int j = 0; j < nyn; j++)
        for (int k = 0; k < nzn; k++) { // electrons are species 0
          Jxs[0][i][j][k] = c * tempXN[i][j][k] / FourPI;
          Jys[0][i][j][k] = c * tempYN[i][j][k] / FourPI;
          Jzs[0][i][j][k] = c * tempZN[i][j][k] / FourPI;
        }

    for (int is = 0; is < ns; is++)
      grid->interpN2C(rhocs, is, rhons);
  } else {
    init(); // use the fields from restart file
  }
}

void EMfields3D::initTaylorGreen() {
  const VirtualTopology3D* vct = &get_vct();
  const Grid* grid = &get_grid();
  if (restart1 == 0) {
    if (vct->getCartesian_rank() == 0) {
      cout << "----------------------------------------" << endl;
      cout << "       Initialize Taylor-Green flow     " << endl;
      cout << "----------------------------------------" << endl;
      cout << "B0                               = " << B0x << endl;
      cout << "u0                               = " << ue0 << endl;
      for (int i = 0; i < ns; i++) {
        cout << "rho species " << i << " = " << rhoINIT[i] << endl;
      }
      cout << "Smoothing Factor = " << Smooth << endl;
      cout << "-------------------------" << endl;
    }

#pragma omp parallel for collapse(3)
    for (int i = 0; i < nxn; i++)
      for (int j = 0; j < nyn; j++)
        for (int k = 0; k < nzn; k++) {
          // Densities, and the species currents of the Taylor-Green flow u
          // that every species shares (no net current): J_s = sign(q_s)
          // rhons u, the moment convention, so maxwellianFromSpeciesCurrents
          // gives each species the drift u.
          for (int is = 0; is < ns; is++) {
            rhons[is][i][j][k] = rhoINIT[is] / FourPI;
            const double flowAmp =
                qom[is] / fabs(qom[is]) * ue0 * rhons[is][i][j][k];
            Jxs[is][i][j][k] = flowAmp *
                               sin(2. * M_PI * grid->getXN(i, j, k) / Lx) *
                               cos(2. * M_PI * grid->getYN(i, j, k) / Ly) *
                               cos(2. * M_PI * grid->getZN(i, j, k) / Lz);
            Jys[is][i][j][k] = -flowAmp *
                               cos(2. * M_PI * grid->getXN(i, j, k) / Lx) *
                               sin(2. * M_PI * grid->getYN(i, j, k) / Ly) *
                               cos(2. * M_PI * grid->getZN(i, j, k) / Lz);
            Jzs[is][i][j][k] = 0.; // Z velocity is zero
          }

          // electric field
          Ex[i][j][k] = 0.0;
          Ey[i][j][k] = 0.0;
          Ez[i][j][k] = 0.0;
          // Magnetic field
          Bxn[i][j][k] = B0x * cos(2. * M_PI * grid->getXN(i, j, k) / Lx) *
                         sin(2. * M_PI * grid->getYN(i, j, k) / Ly) *
                         sin(2. * M_PI * grid->getZN(i, j, k) / Lz);
          Byn[i][j][k] = B0x * sin(2. * M_PI * grid->getXN(i, j, k) / Lx) *
                         cos(2. * M_PI * grid->getYN(i, j, k) / Ly) *
                         sin(2. * M_PI * grid->getZN(i, j, k) / Lz);
          Bzn[i][j][k] = -2. * B0x *
                         sin(2. * M_PI * grid->getXN(i, j, k) / Lx) *
                         sin(2. * M_PI * grid->getYN(i, j, k) / Ly) *
                         cos(2. * M_PI * grid->getZN(i, j, k) / Lz);
        }

#pragma omp parallel for collapse(3)
    for (int i = 0; i < nxc; i++)
      for (int j = 0; j < nyc; j++)
        for (int k = 0; k < nzc; k++) {
          Bxc[i][j][k] = B0x * cos(2. * M_PI * grid->getXC(i, j, k) / Lx) *
                         sin(2. * M_PI * grid->getYC(i, j, k) / Ly) *
                         sin(2. * M_PI * grid->getZC(i, j, k) / Lz);
          Byc[i][j][k] = B0x * sin(2. * M_PI * grid->getXC(i, j, k) / Lx) *
                         cos(2. * M_PI * grid->getYC(i, j, k) / Ly) *
                         sin(2. * M_PI * grid->getZC(i, j, k) / Lz);
          Bzc[i][j][k] = -2. * B0x *
                         sin(2. * M_PI * grid->getXC(i, j, k) / Lx) *
                         sin(2. * M_PI * grid->getYC(i, j, k) / Ly) *
                         cos(2. * M_PI * grid->getZC(i, j, k) / Lz);
        }

    for (int is = 0; is < ns; is++)
      grid->interpN2C(rhocs, is, rhons);
  } else {
    init(); // use the fields from restart file
  }
}

void EMfields3D::initOriginalGEM() {
  const Grid* grid = &get_grid();
  if (restart1 == 0) {
    // initialize
    if (get_vct().getCartesian_rank() == 0) {
      cout << "------------------------------------------" << endl;
      cout << "Initialize GEM Challenge with Pertubation" << endl;
      cout << "------------------------------------------" << endl;
      cout << "B0x                              = " << B0x << endl;
      cout << "B0y                              = " << B0y << endl;
      cout << "B0z                              = " << B0z << endl;
      cout << "Delta (current sheet thickness) = " << delta << endl;
      for (int i = 0; i < ns; i++) {
        cout << "rho species " << i << " = " << rhoINIT[i];
        if (DriftSpecies[i])
          cout << " DRIFTING " << endl;
        else
          cout << " BACKGROUND " << endl;
      }
      cout << "-------------------------" << endl;
    }
#pragma omp parallel for collapse(3)
    for (int i = 0; i < nxn; i++)
      for (int j = 0; j < nyn; j++)
        for (int k = 0; k < nzn; k++) {
          // initialize the density for species
          for (int is = 0; is < ns; is++) {
            if (DriftSpecies[is])
              rhons[is][i][j][k] =
                  ((rhoINIT[is] /
                    (cosh((grid->getYN(i, j, k) - Ly / 2) / delta) *
                     cosh((grid->getYN(i, j, k) - Ly / 2) / delta)))) /
                  FourPI;
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
          Bxn[i][j][k] -= (B0x / 10.0) * (M_PI / Ly) * cos(2 * M_PI * xM / Lx) *
                          sin(M_PI * yM / Ly);
          Byn[i][j][k] = B0y + (B0x / 10.0) * (2 * M_PI / Lx) *
                                   sin(2 * M_PI * xM / Lx) *
                                   cos(M_PI * yM / Ly);
          Bzn[i][j][k] = B0z;
        }
    // initialize B on centers
#pragma omp parallel for collapse(3)
    for (int i = 0; i < nxc; i++)
      for (int j = 0; j < nyc; j++)
        for (int k = 0; k < nzc; k++) {
          // Magnetic field
          const double yM = grid->getYC(i, j, k) - .5 * Ly;
          Bxc[i][j][k] = B0x * tanh(yM / delta);
          // add the initial GEM perturbation
          const double xM = grid->getXC(i, j, k) - .5 * Lx;
          Bxc[i][j][k] -= (B0x / 10.0) * (M_PI / Ly) * cos(2 * M_PI * xM / Lx) *
                          sin(M_PI * yM / Ly);
          Byc[i][j][k] = B0y + (B0x / 10.0) * (2 * M_PI / Lx) *
                                   sin(2 * M_PI * xM / Lx) *
                                   cos(M_PI * yM / Ly);
          Bzc[i][j][k] = B0z;
        }
    for (int is = 0; is < ns; is++)
      grid->interpN2C(rhocs, is, rhons);
  } else {
    init(); // use the fields from restart file
  }
}

void EMfields3D::initGEMDoubleHarris() {
  const Collective* col = &get_col();
  const VirtualTopology3D* vct = &get_vct();
  const Grid* grid = &get_grid();

  double pertX = 0.4;
  double xpert, ypert, exp_pert;
  if (restart1 == 0) {

    if (vct->getCartesian_rank() == 0) {
      cout << "------------------------------------------" << endl;
      cout << "Initialize Double Harris Sheet with Perturbation" << endl;
      cout << "------------------------------------------" << endl;
      cout << "B0x                              = " << B0x << endl;
      cout << "B0y                              = " << B0y << endl;
      cout << "B0z                              = " << B0z << endl;
      cout << "Delta (current sheet thickness)  = " << delta << endl;
      for (int i = 0; i < ns; i++) {
        cout << "rho species " << i << " = " << rhoINIT[i];
        if (DriftSpecies[i])
          cout << " DRIFTING " << endl;
        else
          cout << " BACKGROUND " << endl;
      }
      cout << "-------------------------" << endl;
    }
#pragma omp parallel for collapse(3) private(exp_pert, xpert, ypert)
    for (int i = 0; i < nxn; i++)
      for (int j = 0; j < nyn; j++)
        for (int k = 0; k < nzn; k++) {
          const double xM = grid->getXN(i, j, k) - 0.5 * Lx;
          const double yB = grid->getYN(i, j, k) - 0.25 * Ly;
          const double yT = grid->getYN(i, j, k) - 0.75 * Ly;
          const double yBd = yB / delta;
          const double yTd = yT / delta;
          // initialize the density for species
          for (int is = 0; is < ns; is++) {
            // dprintf("is=%d, DriftSpecies[is]=%d",is,DriftSpecies[is]);
            if (DriftSpecies[is]) {
              const double sech_yBd = 1. / cosh(yBd); //+1e-5;
              const double sech_yTd = 1. / cosh(yTd); //+1e-5;
              if (is == 0 || is == 1)
                rhons[is][i][j][k] = rhoINIT[is] * sech_yBd * sech_yBd / FourPI;
              else if (is == 2 || is == 3)
                rhons[is][i][j][k] = rhoINIT[is] * sech_yTd * sech_yTd / FourPI;
            } else
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
          exp_pert = exp(-(xpert / delta) * (xpert / delta) -
                         (ypert / delta) * (ypert / delta));
          Bxn[i][j][k] +=
              (B0x * pertX) * exp_pert *
              (-cos(M_PI * xpert / 10.0 / delta) *
                   cos(M_PI * ypert / 10.0 / delta) * 2.0 * ypert / delta -
               cos(M_PI * xpert / 10.0 / delta) *
                   sin(M_PI * ypert / 10.0 / delta) * M_PI / 10.0);
          Byn[i][j][k] +=
              (B0x * pertX) * exp_pert *
              (cos(M_PI * xpert / 10.0 / delta) *
                   cos(M_PI * ypert / 10.0 / delta) * 2.0 * xpert / delta +
               sin(M_PI * xpert / 10.0 / delta) *
                   cos(M_PI * ypert / 10.0 / delta) * M_PI / 10.0);
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
  } else {
    init(); // use the fields from restart file
  }
}

void EMfields3D::initDoublePeriodicHarrisWithGaussianHumpPerturbation() {
  const Collective* col = &get_col();
  const VirtualTopology3D* vct = &get_vct();
  const Grid* grid = &get_grid();
  // perturbation localized in X
  const double pertX = 0.4;
  const double deltax = 8. * delta;
  const double deltay = 4. * delta;
  if (restart1 == 0) {
    // initialize
    if (get_vct().getCartesian_rank() == 0) {
      cout << "------------------------------------------" << endl;
      cout << "Initialize GEM Challenge with Pertubation" << endl;
      cout << "------------------------------------------" << endl;
      cout << "B0x                              = " << B0x << endl;
      cout << "B0y                              = " << B0y << endl;
      cout << "B0z                              = " << B0z << endl;
      cout << "Delta (current sheet thickness) = " << delta << endl;
      for (int i = 0; i < ns; i++) {
        cout << "rho species " << i << " = " << rhoINIT[i];
        if (DriftSpecies[i])
          cout << " DRIFTING " << endl;
        else
          cout << " BACKGROUND " << endl;
      }
      cout << "-------------------------" << endl;
    }
#pragma omp parallel for collapse(3)
    for (int i = 0; i < nxn; i++)
      for (int j = 0; j < nyn; j++)
        for (int k = 0; k < nzn; k++) {
          const double xM = grid->getXN(i, j, k) - .5 * Lx;
          const double yB = grid->getYN(i, j, k) - .25 * Ly;
          const double yT = grid->getYN(i, j, k) - .75 * Ly;
          const double yBd = yB / delta;
          const double yTd = yT / delta;
          // initialize the density for species
          for (int is = 0; is < ns; is++) {
            if (DriftSpecies[is]) {
              const double sech_yBd = 1. / cosh(yBd);
              const double sech_yTd = 1. / cosh(yTd);
              rhons[is][i][j][k] = rhoINIT[is] * sech_yBd * sech_yBd / FourPI;
              rhons[is][i][j][k] += rhoINIT[is] * sech_yTd * sech_yTd / FourPI;
            } else
              rhons[is][i][j][k] = rhoINIT[is] / FourPI;
          }
          // electric field
          Ex[i][j][k] = 0.0;
          Ey[i][j][k] = 0.0;
          Ez[i][j][k] = 0.0;
          // Magnetic field
          Bxn[i][j][k] = B0x * (-1.0 + tanh(yBd) - tanh(yTd));
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
    communicateNodeBC(nxn, nyn, nzn, Bxn, col->bcBx[0], col->bcBx[1],
                      col->bcBx[2], col->bcBx[3], col->bcBx[4], col->bcBx[5],
                      vct, this);
    communicateNodeBC(nxn, nyn, nzn, Byn, col->bcBy[0], col->bcBy[1],
                      col->bcBy[2], col->bcBy[3], col->bcBy[4], col->bcBy[5],
                      vct, this);
    communicateNodeBC(nxn, nyn, nzn, Bzn, col->bcBz[0], col->bcBz[1],
                      col->bcBz[2], col->bcBz[3], col->bcBz[4], col->bcBz[5],
                      vct, this);

    // initialize B on centers
#pragma omp parallel for collapse(3)
    for (int i = 0; i < nxc; i++)
      for (int j = 0; j < nyc; j++)
        for (int k = 0; k < nzc; k++) {
          const double xM = grid->getXN(i, j, k) - .5 * Lx;
          const double yB = grid->getYN(i, j, k) - .25 * Ly;
          const double yT = grid->getYN(i, j, k) - .75 * Ly;
          const double yBd = yB / delta;
          const double yTd = yT / delta;
          Bxc[i][j][k] = B0x * (-1.0 + tanh(yBd) - tanh(yTd));
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
    communicateCenterBC(nxc, nyc, nzc, Bxc, col->bcBx[0], col->bcBx[1],
                        col->bcBx[2], col->bcBx[3], col->bcBx[4], col->bcBx[5],
                        vct, this);
    communicateCenterBC(nxc, nyc, nzc, Byc, col->bcBy[0], col->bcBy[1],
                        col->bcBy[2], col->bcBy[3], col->bcBy[4], col->bcBy[5],
                        vct, this);
    communicateCenterBC(nxc, nyc, nzc, Bzc, col->bcBz[0], col->bcBz[1],
                        col->bcBz[2], col->bcBz[3], col->bcBz[4], col->bcBz[5],
                        vct, this);
    for (int is = 0; is < ns; is++)
      grid->interpN2C(rhocs, is, rhons);
  } else {
    init(); // use the fields from restart file
  }
}

void EMfields3D::initHumpPerturbation() {
  const Collective* col = &get_col();
  const VirtualTopology3D* vct = &get_vct();
  const Grid* grid = &get_grid();
  if (restart1 == 0) {
    // initialize
    if (get_vct().getCartesian_rank() == 0) {
      cout << "------------------------------------------" << endl;
      cout << "Initialize with Hump Pertubation" << endl;
      cout << "------------------------------------------" << endl;
      cout << "B0x                              = " << B0x << endl;
      cout << "B0y                              = " << B0y << endl;
      cout << "B0z                              = " << B0z << endl;
      cout << "Delta                            = " << delta << endl;
      for (int i = 0; i < ns; i++) {
        cout << "rho species " << i << " = " << rhoINIT[i];
        if (DriftSpecies[i])
          cout << " DRIFTING " << endl;
        else
          cout << " BACKGROUND " << endl;
      }
      cout << "-------------------------" << endl;
    }
#pragma omp parallel for collapse(3)
    for (int i = 0; i < nxn; i++)
      for (int j = 0; j < nyn; j++)
        for (int k = 0; k < nzn; k++) {
          const double xM = grid->getXN(i, j, k) - .5 * Lx;
          const double yM = grid->getYN(i, j, k) - .5 * Ly;
          const double zM = grid->getZN(i, j, k) - .5 * Lz;
          const double xMd = xM / delta;
          const double yMd = yM / delta;
          const double zMd = zM / delta;
          // initialize the density for species
          for (int is = 0; is < ns; is++) {
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
          Bxn[i][j][k] =
              B0x * (0.5 * sech_yMd * sech_yMd * sech_zMd * sech_zMd + 1.0);
          Byn[i][j][k] =
              B0y * (0.5 * sech_xMd * sech_xMd * sech_zMd * sech_zMd + 1.0);
          Bzn[i][j][k] =
              B0z * (0.5 * sech_xMd * sech_xMd * sech_yMd * sech_yMd + 1.0);
        }
    // communicate ghost
    communicateNodeBC(nxn, nyn, nzn, Bxn, col->bcBx[0], col->bcBx[1],
                      col->bcBx[2], col->bcBx[3], col->bcBx[4], col->bcBx[5],
                      vct, this);
    communicateNodeBC(nxn, nyn, nzn, Byn, col->bcBy[0], col->bcBy[1],
                      col->bcBy[2], col->bcBy[3], col->bcBy[4], col->bcBy[5],
                      vct, this);
    communicateNodeBC(nxn, nyn, nzn, Bzn, col->bcBz[0], col->bcBz[1],
                      col->bcBz[2], col->bcBz[3], col->bcBz[4], col->bcBz[5],
                      vct, this);

    // initialize B on centers
#pragma omp parallel for collapse(3)
    for (int i = 0; i < nxc; i++)
      for (int j = 0; j < nyc; j++)
        for (int k = 0; k < nzc; k++) {
          const double xM = grid->getXC(i, j, k) - .5 * Lx;
          const double yM = grid->getYC(i, j, k) - .5 * Ly;
          const double zM = grid->getZC(i, j, k) - .5 * Lz;
          const double xMd = xM / delta;
          const double yMd = yM / delta;
          const double zMd = zM / delta;
          const double sech_xMd = 1. / cosh(xMd);
          const double sech_yMd = 1. / cosh(yMd);
          const double sech_zMd = 1. / cosh(zMd);
          Bxc[i][j][k] =
              B0x * (0.5 * sech_yMd * sech_yMd * sech_zMd * sech_zMd + 1.0);
          Byc[i][j][k] =
              B0y * (0.5 * sech_xMd * sech_xMd * sech_zMd * sech_zMd + 1.0);
          Bzc[i][j][k] =
              B0z * (0.5 * sech_xMd * sech_xMd * sech_yMd * sech_yMd + 1.0);
        }
    // communicate ghost
    communicateCenterBC(nxc, nyc, nzc, Bxc, col->bcBx[0], col->bcBx[1],
                        col->bcBx[2], col->bcBx[3], col->bcBx[4], col->bcBx[5],
                        vct, this);
    communicateCenterBC(nxc, nyc, nzc, Byc, col->bcBy[0], col->bcBy[1],
                        col->bcBy[2], col->bcBy[3], col->bcBy[4], col->bcBy[5],
                        vct, this);
    communicateCenterBC(nxc, nyc, nzc, Bzc, col->bcBz[0], col->bcBz[1],
                        col->bcBz[2], col->bcBz[3], col->bcBz[4], col->bcBz[5],
                        vct, this);
    for (int is = 0; is < ns; is++)
      grid->interpN2C(rhocs, is, rhons);
  } else {
    init(); // use the fields from restart file
  }
}

/*! initialize GEM challenge with no Perturbation with dipole-like tail topology
 */
void EMfields3D::initGEMDipoleLikeTailNoPert() {
  const Grid* grid = &get_grid();
  // parameters controling the field topology
  // e.g., x1=Lx/5,x2=Lx/4 give 'separated' fields, x1=Lx/4,x2=Lx/3 give
  // 'reconnected' topology

  double x1 = Lx / 6.0; // minimal position of the gaussian peak
  double x2 = Lx / 4.0; // maximal position of the gaussian peak (the one closer
                        // to the center)
  double sigma = Lx / 15; // width of the gaussian
  double stretch_curve =
      2.0; // stretch the sin^2 function over the x dimension - also can
           // regulate the number of 'knots/reconnecitons points' if less than 1
  double skew_parameter = 0.50; // skew of the shape of the gaussian
  double pi = 3.1415927;
  double r1, r2, delta_x1x2;

  if (restart1 == 0) {

    // initialize
    if (get_vct().getCartesian_rank() == 0) {
      cout << "----------------------------------------------" << endl;
      cout << "Initialize GEM Challenge without Perturbation" << endl;
      cout << "----------------------------------------------" << endl;
      cout << "B0x                              = " << B0x << endl;
      cout << "B0y                              = " << B0y << endl;
      cout << "B0z                              = " << B0z << endl;
      cout << "Delta (current sheet thickness) = " << delta << endl;
      for (int i = 0; i < ns; i++) {
        cout << "rho species " << i << " = " << rhoINIT[i];
        if (DriftSpecies[i])
          cout << " DRIFTING " << endl;
        else
          cout << " BACKGROUND " << endl;
      }
      cout << "-------------------------" << endl;
    }

#pragma omp parallel for collapse(3) private(delta_x1x2, r1, r2)
    for (int i = 0; i < nxn; i++)
      for (int j = 0; j < nyn; j++)
        for (int k = 0; k < nzn; k++) {
          // initialize the density for species
          for (int is = 0; is < ns; is++) {
            if (DriftSpecies[is])
              rhons[is][i][j][k] =
                  ((rhoINIT[is] /
                    (cosh((grid->getYN(i, j, k) - Ly / 2) / delta) *
                     cosh((grid->getYN(i, j, k) - Ly / 2) / delta)))) /
                  FourPI;
            else
              rhons[is][i][j][k] = rhoINIT[is] / FourPI;
          }
          // electric field
          Ex[i][j][k] = 0.0;
          Ey[i][j][k] = 0.0;
          Ez[i][j][k] = 0.0;
          // Magnetic field

          delta_x1x2 = x1 - x2 *
                                (sin(((grid->getXN(i, j, k) - Lx / 2) / Lx *
                                      180.0 / stretch_curve) *
                                     (0.25 * FourPI) / 180.0)) *
                                (sin(((grid->getXN(i, j, k) - Lx / 2) / Lx *
                                      180.0 / stretch_curve) *
                                     (0.25 * FourPI) / 180.0));

          r1 = (grid->getYN(i, j, k) - (x1 + delta_x1x2)) *
               (1.0 - skew_parameter *
                          (sin(((grid->getXN(i, j, k) - Lx / 2) / Lx * 180.0) *
                               (0.25 * FourPI) / 180.0)) *
                          (sin(((grid->getXN(i, j, k) - Lx / 2) / Lx * 180.0) *
                               (0.25 * FourPI) / 180.0)));
          r2 = (grid->getYN(i, j, k) - ((Lx - x1) - delta_x1x2)) *
               (1.0 - skew_parameter *
                          (sin(((grid->getXN(i, j, k) - Lx / 2) / Lx * 180.0) *
                               (0.25 * FourPI) / 180.0)) *
                          (sin(((grid->getXN(i, j, k) - Lx / 2) / Lx * 180.0) *
                               (0.25 * FourPI) / 180.0)));

          // tail-like field topology
          Bxn[i][j][k] = B0x * 0.5 *
                         (-exp(-((r1) * (r1)) / (sigma * sigma)) +
                          exp(-((r2) * (r2)) / (sigma * sigma)));

          Byn[i][j][k] = B0y;
          // guide field
          Bzn[i][j][k] = B0z;
        }
    // initialize B on centers
#pragma omp parallel for collapse(3) private(delta_x1x2, r1, r2)
    for (int i = 0; i < nxc; i++)
      for (int j = 0; j < nyc; j++)
        for (int k = 0; k < nzc; k++) {
          // Magnetic field

          delta_x1x2 = x1 - x2 *
                                (sin(((grid->getXC(i, j, k) - Lx / 2) / Lx *
                                      180.0 / stretch_curve) *
                                     (0.25 * FourPI) / 180.0)) *
                                (sin(((grid->getXC(i, j, k) - Lx / 2) / Lx *
                                      180.0 / stretch_curve) *
                                     (0.25 * FourPI) / 180.0));

          r1 = (grid->getYC(i, j, k) - (x1 + delta_x1x2)) *
               (1.0 - skew_parameter *
                          (sin(((grid->getXC(i, j, k) - Lx / 2) / Lx * 180.0) *
                               (0.25 * FourPI) / 180.0)) *
                          (sin(((grid->getXC(i, j, k) - Lx / 2) / Lx * 180.0) *
                               (0.25 * FourPI) / 180.0)));
          r2 = (grid->getYC(i, j, k) - ((Lx - x1) - delta_x1x2)) *
               (1.0 - skew_parameter *
                          (sin(((grid->getXC(i, j, k) - Lx / 2) / Lx * 180.0) *
                               (0.25 * FourPI) / 180.0)) *
                          (sin(((grid->getXC(i, j, k) - Lx / 2) / Lx * 180.0) *
                               (0.25 * FourPI) / 180.0)));

          // tail-like field topology
          Bxc[i][j][k] = B0x * 0.5 *
                         (-exp(-((r1) * (r1)) / (sigma * sigma)) +
                          exp(-((r2) * (r2)) / (sigma * sigma)));

          Byc[i][j][k] = B0y;
          // guide field
          Bzc[i][j][k] = B0z;
        }
    for (int is = 0; is < ns; is++)
      grid->interpN2C(rhocs, is, rhons);
  } else {
    init(); // use the fields from restart file
  }
}

/*! initialize GEM challenge with no Perturbation */
void EMfields3D::initGEMnoPert() {
  const Collective* col = &get_col();
  const VirtualTopology3D* vct = &get_vct();
  const Grid* grid = &get_grid();
  if (restart1 == 0) {

    // initialize
    if (get_vct().getCartesian_rank() == 0) {
      cout << "----------------------------------------------" << endl;
      cout << "Initialize GEM Challenge without Perturbation" << endl;
      cout << "----------------------------------------------" << endl;
      cout << "B0x                              = " << B0x << endl;
      cout << "B0y                              = " << B0y << endl;
      cout << "B0z                              = " << B0z << endl;
      cout << "Delta (current sheet thickness) = " << delta << endl;
      for (int i = 0; i < ns; i++) {
        cout << "rho species " << i << " = " << rhoINIT[i];
        if (DriftSpecies[i])
          cout << " DRIFTING " << endl;
        else
          cout << " BACKGROUND " << endl;
      }
      cout << "-------------------------" << endl;
    }
#pragma omp parallel for collapse(3)
    for (int i = 0; i < nxn; i++)
      for (int j = 0; j < nyn; j++)
        for (int k = 0; k < nzn; k++) {
          // initialize the density for species
          for (int is = 0; is < ns; is++) {
            if (DriftSpecies[is])
              rhons[is][i][j][k] =
                  ((rhoINIT[is] /
                    (cosh((grid->getYN(i, j, k) - Ly / 2) / delta) *
                     cosh((grid->getYN(i, j, k) - Ly / 2) / delta)))) /
                  FourPI;
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
#pragma omp parallel for collapse(3)
    for (int i = 0; i < nxc; i++)
      for (int j = 0; j < nyc; j++)
        for (int k = 0; k < nzc; k++) {
          // Magnetic field
          Bxc[i][j][k] = B0x * tanh((grid->getYC(i, j, k) - Ly / 2) / delta);
          Byc[i][j][k] = B0y;
          // guide field
          Bzc[i][j][k] = B0z;
        }
    for (int is = 0; is < ns; is++)
      grid->interpN2C(rhocs, is, rhons);
  } else {
    init(); // use the fields from restart file
  }
}

/**
 * @brief Initialize the random-field test configuration.
 */
void EMfields3D::initRandomField() {
  const VirtualTopology3D* vct = &get_vct();
  const Grid* grid = &get_grid();
  double** modes_seed = newArr2(double, 7, 7);
  if (restart1 == 0) {
    // initialize
    if (get_vct().getCartesian_rank() == 0) {
      cout << "------------------------------------------" << endl;
      cout << "Initialize GEM Challenge with Pertubation" << endl;
      cout << "------------------------------------------" << endl;
      cout << "B0x                              = " << B0x << endl;
      cout << "B0y                              = " << B0y << endl;
      cout << "B0z                              = " << B0z << endl;
      cout << "Delta (current sheet thickness) = " << delta << endl;
      for (int i = 0; i < ns; i++) {
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
            my_file <<"modes_seed["<< m+3<<"][" << "\t" << n+3 << "] = " <<
  modes_seed[m+3][n+3] << endl;
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

#pragma omp parallel for collapse(3) private(kx, ky)
    for (int i = 0; i < nxn; i++)
      for (int j = 0; j < nyn; j++)
        for (int k = 0; k < nzn; k++) {
          // initialize the density for species
          for (int is = 0; is < ns; is++) {
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
            for (int n = -3; n < 4; n++) {

              kx = 2.0 * M_PI * m / Lx;
              ky = 2.0 * M_PI * n / Ly;
              Bxn[i][j][k] +=
                  -B0x * ky *
                  cos(grid->getXN(i, j, k) * kx + grid->getYN(i, j, k) * ky +
                      2.0 * M_PI * modes_seed[m + 3][n + 3]);
              Byn[i][j][k] +=
                  B0x * kx *
                  cos(grid->getXN(i, j, k) * kx + grid->getYN(i, j, k) * ky +
                      2.0 * M_PI * modes_seed[m + 3][n + 3]);
              // Bzn[i][j][k] +=
              // B0x*cos(grid->getXN(i,j,k)*kx+grid->getYN(i,j,k)*ky+2.0*M_PI*modes_seed[m+3][n+3]);
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
  } else {
    init(); // use the fields from restart file
  }
  delArr2(modes_seed, 7);
}

/**
 * @brief Initialize the force-free equilibrium configuration.
 */
void EMfields3D::initForceFree() {
  const VirtualTopology3D* vct = &get_vct();
  const Grid* grid = &get_grid();
  if (restart1 == 0) {

    // initialize
    if (get_vct().getCartesian_rank() == 0) {
      cout << "----------------------------------------" << endl;
      cout << "Initialize Force Free with Perturbation" << endl;
      cout << "----------------------------------------" << endl;
      cout << "B0x                              = " << B0x << endl;
      cout << "B0y                              = " << B0y << endl;
      cout << "B0z                              = " << B0z << endl;
      cout << "Delta (current sheet thickness) = " << delta << endl;
      for (int i = 0; i < ns; i++) {
        cout << "rho species " << i << " = " << rhoINIT[i] << endl;
      }
      cout << "Smoothing Factor = " << Smooth << endl;
      cout << "-------------------------" << endl;
    }
#pragma omp parallel for collapse(3)
    for (int i = 0; i < nxn; i++)
      for (int j = 0; j < nyn; j++)
        for (int k = 0; k < nzn; k++) {
          // initialize the density for species
          for (int is = 0; is < ns; is++) {
            rhons[is][i][j][k] = rhoINIT[is] / FourPI;
          }
          // electric field
          Ex[i][j][k] = 0.0;
          Ey[i][j][k] = 0.0;
          Ez[i][j][k] = 0.0;
          // Magnetic field
          Bxn[i][j][k] = B0x * tanh((grid->getYN(i, j, k) - Ly / 2) / delta);
          // add the initial GEM perturbation
          Bxn[i][j][k] += (B0x / 10.0) * (M_PI / Ly) *
                          cos(2 * M_PI * grid->getXN(i, j, k) / Lx) *
                          sin(M_PI * (grid->getYN(i, j, k) - Ly / 2) / Ly);
          Byn[i][j][k] =
              B0y - (B0x / 10.0) * (2 * M_PI / Lx) *
                        sin(2 * M_PI * grid->getXN(i, j, k) / Lx) *
                        cos(M_PI * (grid->getYN(i, j, k) - Ly / 2) / Ly);
          // Bz = B0z/cosh, localised on the sheet
          Bzn[i][j][k] = B0z / cosh((grid->getYN(i, j, k) - Ly / 2) / delta);
        }
#pragma omp parallel for collapse(3)
    for (int i = 0; i < nxc; i++)
      for (int j = 0; j < nyc; j++)
        for (int k = 0; k < nzc; k++) {
          Bxc[i][j][k] = B0x * tanh((grid->getYC(i, j, k) - Ly / 2) / delta);
          // add the perturbation
          Bxc[i][j][k] += (B0x / 10.0) * (M_PI / Ly) *
                          cos(2 * M_PI * grid->getXC(i, j, k) / Lx) *
                          sin(M_PI * (grid->getYC(i, j, k) - Ly / 2) / Ly);
          Byc[i][j][k] =
              B0y - (B0x / 10.0) * (2 * M_PI / Lx) *
                        sin(2 * M_PI * grid->getXC(i, j, k) / Lx) *
                        cos(M_PI * (grid->getYC(i, j, k) - Ly / 2) / Ly);
          // Bz = B0z/cosh, localised on the sheet
          Bzc[i][j][k] = B0z / cosh((grid->getYC(i, j, k) - Ly / 2) / delta);
        }

    for (int is = 0; is < ns; is++)
      grid->interpN2C(rhocs, is, rhons);

    // J = (c/4pi) curl(B) of the sheet and its perturbation, split among the
    // species by u0/v0/w0 as in GEMHarris with currentFromAmpere = 1
    initSpeciesCurrents(true, false);
  } else {
    init(); // use the fields from restart file
  }
}
/*! Beam test: zero fields, uniform species 0 and 1, species 2 inside an
 * ellipsoid */
void EMfields3D::initBEAM(double x_center, double y_center, double z_center,
                          double radius) {
  const Grid* grid = &get_grid();

  double distance;
  // initialize E and rhos on nodes
  if (restart1 == 0) {
#pragma omp parallel for collapse(3) private(distance)
    for (int i = 0; i < nxn; i++)
      for (int j = 0; j < nyn; j++)
        for (int k = 0; k < nzn; k++) {
          Ex[i][j][k] = 0.0;
          Ey[i][j][k] = 0.0;
          Ez[i][j][k] = 0.0;
          Bxn[i][j][k] = 0.0;
          Byn[i][j][k] = 0.0;
          Bzn[i][j][k] = 0.0;
          distance = (grid->getXN(i, j, k) - x_center) *
                         (grid->getXN(i, j, k) - x_center) / (radius * radius) +
                     (grid->getYN(i, j, k) - y_center) *
                         (grid->getYN(i, j, k) - y_center) / (radius * radius) +
                     (grid->getZN(i, j, k) - z_center) *
                         (grid->getZN(i, j, k) - z_center) /
                         (4 * radius * radius);
          // plasma
          rhons[0][i][j][k] =
              rhoINIT[0] / FourPI; // initialize with constant density
          // electrons
          rhons[1][i][j][k] = rhoINIT[1] / FourPI;
          // beam
          if (distance < 1.0)
            rhons[2][i][j][k] = rhoINIT[2] / FourPI;
          else
            rhons[2][i][j][k] = 0.0;
        }
    // initialize B on centers
#pragma omp parallel for collapse(3)
    for (int i = 0; i < nxc; i++)
      for (int j = 0; j < nyc; j++)
        for (int k = 0; k < nzc; k++) {
          // Magnetic field
          Bxc[i][j][k] = 0.0;
          Byc[i][j][k] = 0.0;
          Bzc[i][j][k] = 0.0;
        }
    for (int is = 0; is < ns; is++)
      grid->interpN2C(rhocs, is, rhons);
  } else {
    init(); // use the fields from restart file
  }
}

/**
 * @brief Initialize the 3D magnetic-dipole planetary configuration.
 */
void EMfields3D::initDipole() {
  const Collective* col = &get_col();
  const VirtualTopology3D* vct = &get_vct();
  const Grid* grid = &get_grid();

  // initialize
  if (vct->getCartesian_rank() == 0) {
    cout << "------------------------------------------" << endl;
    cout << "Initialise a Magnetic Dipole " << endl;
    cout << "------------------------------------------" << endl;
    cout << "B0x                              = " << B0x << endl;
    cout << "B0y                              = " << B0y << endl;
    cout << "B0z                              = " << B0z << endl;
    cout << "B1x   (external dipole field) - X  = " << B1x << endl;
    cout << "B1y                              = " << B1y << endl;
    cout << "B1z                              = " << B1z << endl;
    cout << "Planet_radius - no magnetic field inside the planet = "
         << Planet_radius << endl;
    cout << "Center dipole - X                = " << x_center_dipole << endl;
    cout << "Center dipole - Y                = " << y_center_dipole << endl;
    cout << "Center dipole - Z                = " << z_center_dipole << endl;
    cout << "Center planet - X                = " << x_center_planet << endl;
    cout << "Center planet - Y                = " << y_center_planet << endl;
    cout << "Center planet - Z                = " << z_center_planet << endl;
    cout << "Solar Wind drift velocity        = " << ue0 << endl;
  }

  double ebc[3];
  cross_product(ue0, ve0, we0, B0x, B0y, B0z, ebc);
  scale(ebc, -1.0, 3);

#pragma omp parallel for collapse(3)
  for (int i = 0; i < nxn; i++) {
    for (int j = 0; j < nyn; j++) {
      for (int k = 0; k < nzn; k++) {
        for (int is = 0; is < ns; is++) {
          rhons[is][i][j][k] = rhoINIT[is] / FourPI;
        }
        Ex[i][j][k] = ebc[0];
        Ey[i][j][k] = ebc[1];
        Ez[i][j][k] = ebc[2];

        double blp[3];
        // radius of the planet
        double a = Planet_radius;

        double x = grid->getXN(i, j, k);
        double y = grid->getYN(i, j, k);
        double z = grid->getZN(i, j, k);

        // Distance from planet center
        double r2 = ((x - x_center_planet) * (x - x_center_planet)) +
                    ((y - y_center_planet) * (y - y_center_planet)) +
                    ((z - z_center_planet) * (z - z_center_planet));
        // Distance from dipole center
        double r2d = ((x - x_center_dipole) * (x - x_center_dipole)) +
                     ((y - y_center_dipole) * (y - y_center_dipole)) +
                     ((z - z_center_dipole) * (z - z_center_dipole));

        // Compute dipolar field B_ext

        if (r2 > a * a) {
          // position from the dipole center
          const double x_displ = x - x_center_dipole;
          const double y_displ = y - y_center_dipole;
          const double z_displ = z - z_center_dipole;
          const double fac1 = -B1z * a * a * a / pow(r2d, 2.5);
          Bx_ext[i][j][k] = 3 * x_displ * z_displ * fac1;
          By_ext[i][j][k] = 3 * y_displ * z_displ * fac1;
          Bz_ext[i][j][k] =
              (2 * z_displ * z_displ - x_displ * x_displ - y_displ * y_displ) *
              fac1;
        } else { // no field inside the planet
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

  communicateCenterBC_P(nxc, nyc, nzc, Bxc, col->bcBx[0], col->bcBx[1],
                        col->bcBx[2], col->bcBx[3], col->bcBx[4], col->bcBx[5],
                        vct, this);
  communicateCenterBC_P(nxc, nyc, nzc, Byc, col->bcBy[0], col->bcBy[1],
                        col->bcBy[2], col->bcBy[3], col->bcBy[4], col->bcBy[5],
                        vct, this);
  communicateCenterBC_P(nxc, nyc, nzc, Bzc, col->bcBz[0], col->bcBz[1],
                        col->bcBz[2], col->bcBz[3], col->bcBz[4], col->bcBz[5],
                        vct, this);

  for (int is = 0; is < ns; is++)
    grid->interpN2C(rhocs, is, rhons);

  if (restart1 != 0) { // EM initialization from RESTART
    init();            // use the fields from restart file
  }
}

/**
 * @brief Initialize the 2D magnetic-dipole planetary configuration.
 */
void EMfields3D::initDipole2D() {
  const Collective* col = &get_col();
  const VirtualTopology3D* vct = &get_vct();
  const Grid* grid = &get_grid();

  // initialize
  if (vct->getCartesian_rank() == 0) {
    cout << "------------------------------------------" << endl;
    cout << "Initialise a 2D Magnetic Dipole on XZ Plane" << endl;
    cout << "------------------------------------------" << endl;
    cout << "B0x                              = " << B0x << endl;
    cout << "B0y                              = " << B0y << endl;
    cout << "B0z                              = " << B0z << endl;
    cout << "B1x   (external dipole field)    = " << B1x << endl;
    cout << "B1y                              = " << B1y << endl;
    cout << "B1z                              = " << B1z << endl;
    cout << "Planet_radius - no magnetic field inside the planet = "
         << Planet_radius << endl;
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

  double ebc[3];
  cross_product(ue0, ve0, we0, B0x, B0y, B0z, ebc);
  scale(ebc, -1.0, 3);

#pragma omp parallel for collapse(3)
  for (int i = 0; i < nxn; i++) {
    for (int j = 0; j < nyn; j++) {
      for (int k = 0; k < nzn; k++) {
        for (int is = 0; is < ns; is++) {
          rhons[is][i][j][k] = rhoINIT[is] / FourPI;
        }
        Ex[i][j][k] = ebc[0];
        Ey[i][j][k] = ebc[1];
        Ez[i][j][k] = ebc[2];

        double blp[3];
        double a = Planet_radius;

        double xc = x_center_dipole;
        double zc = z_center_dipole;

        double x = grid->getXN(i, j, k);
        double z = grid->getZN(i, j, k);

        // Distance from planet center (for interior mask)
        double r2 = ((x - x_center_planet) * (x - x_center_planet)) +
                    ((z - z_center_planet) * (z - z_center_planet));
        // Distance from dipole center (for field formula)
        double r2d = ((x - xc) * (x - xc)) + ((z - zc) * (z - zc));

        // Compute dipolar field B_ext

        if (r2 > a * a) {
          const double x_displ = x - xc;
          const double z_displ = z - zc;

          const double fac1 = -B1z * a * a / (r2d * r2d);

          Bx_ext[i][j][k] = 2 * x_displ * z_displ * fac1;
          By_ext[i][j][k] = 0.0;
          Bz_ext[i][j][k] = (z_displ * z_displ - x_displ * x_displ) * fac1;
        } else { // no field inside the planet
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

  communicateCenterBC_P(nxc, nyc, nzc, Bxc, col->bcBx[0], col->bcBx[1],
                        col->bcBx[2], col->bcBx[3], col->bcBx[4], col->bcBx[5],
                        vct, this);
  communicateCenterBC_P(nxc, nyc, nzc, Byc, col->bcBy[0], col->bcBy[1],
                        col->bcBy[2], col->bcBy[3], col->bcBy[4], col->bcBy[5],
                        vct, this);
  communicateCenterBC_P(nxc, nyc, nzc, Bzc, col->bcBz[0], col->bcBz[1],
                        col->bcBz[2], col->bcBz[3], col->bcBz[4], col->bcBz[5],
                        vct, this);

  for (int is = 0; is < ns; is++)
    grid->interpN2C(rhocs, is, rhons);

  if (restart1 != 0) { // EM initialization from RESTART
    init();            // use the fields from restart file
  }
}

#ifdef BATSRUS
/**
 * @brief Initialize fields from BATSRUS data.
 */
void EMfields3D::initBATSRUS() {
  const Collective* col = &get_col();
  const Grid* grid = &get_grid();
  cout << "------------------------------------------" << endl;
  cout << "         Initialize from BATSRUS          " << endl;
  cout << "------------------------------------------" << endl;

  // loop over species and cell centers: fill in charge density
  for (int is = 0; is < ns; is++)
    for (int i = 0; i < nxc; i++)
      for (int j = 0; j < nyc; j++)
        for (int k = 0; k < nzc; k++) {
          rhocs[is][i][j][k] = col->getFluidRhoCenter(i, j, k, is);
        }

  // loop over cell centers and fill in magnetic and electric fields
  for (int i = 0; i < nxc; i++)
    for (int j = 0; j < nyc; j++)
      for (int k = 0; k < nzc; k++) {
        col->setFluidFieldsCenter(&Ex[i][j][k], &Ey[i][j][k], &Ez[i][j][k],
                                  &Bxc[i][j][k], &Byc[i][j][k], &Bzc[i][j][k],
                                  i, j, k);
      }

  // interpolate from cell centers to nodes (corners of cells)
  for (int is = 0; is < ns; is++)
    grid->interpC2N(rhons[is], rhocs[is]);
  grid->interpC2N(Bxn, Bxc);
  grid->interpC2N(Byn, Byc);
  grid->interpC2N(Bzn, Bzc);
}
#endif
