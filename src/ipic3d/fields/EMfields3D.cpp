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
#include "MPIdata.h"
#include "Moments.h"
#include "Parameters.h"
#include "ParticleSoAHost.h"
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

namespace {
/** Mean square thermal speed (uth^2 + vth^2 + wth^2) / 3 of species s. */
double thermalSpeed2(const Collective* col, int s) {
  return (col->getUth(s) * col->getUth(s) + col->getVth(s) * col->getVth(s) +
          col->getWth(s) * col->getWth(s)) /
         3.0;
}
} // namespace

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
  divCleaningTol = col->getDivCleaningTol();
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
  qom = AlignedAllocPinned(double, ns); // copied to the device (d_qom)
  for (int i = 0; i < ns; i++)
    qom[i] = col->getQOM(i);
  // boundary conditions: EM fields
  bcEMfaceXright = col->getBcEMfaceXright();
  bcEMfaceXleft = col->getBcEMfaceXleft();
  bcEMfaceYright = col->getBcEMfaceYright();
  bcEMfaceYleft = col->getBcEMfaceYleft();
  bcEMfaceZright = col->getBcEMfaceZright();
  bcEMfaceZleft = col->getBcEMfaceZleft();
  // GEM challenge parameters
  B0x = col->getB0x();
  B0y = col->getB0y();
  B0z = col->getB0z();
  delta = col->getDelta();
  // the field boundary of this rank, decided once (FieldBoundary.h)
  {
    const int cells[3] = {nxc, nyc, nzc};
    double E0[3];
    driftElectricField(E0);
    bc_.setSpec(FieldBoundarySpec::build(*col, *vct, cells, E0, yStart, dy));
  }
  // the Krylov lengths (div(B): its rows drop a flat axis, from the spec)
  divEKrylovSize_ = checkedKrylovLength("div(E)", 1, nxc, nyc, nzc);
  maxwellKrylovSize_ = checkedKrylovLength("Maxwell", 3, nxn, nyn, nzn);
  divBKrylovSize_ = checkedKrylovLength("div(B)", 1, divBExtent(0),
                                        divBExtent(1), divBExtent(2));
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

  if (col->getFieldWriteMethod() == "pvtk" ||
      col->getFieldWriteMethod() == "nbcvtk") {
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

  if (_col.getFieldWriteMethod() == "pvtk" ||
      _col.getFieldWriteMethod() == "nbcvtk") {
    MPI_Type_free(&procview);
    MPI_Type_free(&xyzcomp);
    MPI_Type_free(&procviewXYZ);
    MPI_Type_free(&ghosttype);
  }

  gpuSolverFree();
}

void EMfields3D::adjustNonPeriodicHeatFlux(int is) {
  const VirtualTopology3D* vct = &get_vct();
  const size_t nodes = (size_t)nxn * nyn * nzn;

  // Double every component on the node plane `face` normal to dir, interior
  // [1, n-2] along the other two axes; k innermost (contiguous except on a z
  // face).  An edge node lies on two planes and is doubled twice.
  auto doublePlane = [&](int dir, int face) {
    int lo[3] = {1, 1, 1}, hi[3] = {nxn - 2, nyn - 2, nzn - 2};
    lo[dir] = hi[dir] = face;
    for (int c = 0; c < HeatFlux::ComponentCount; ++c) {
      double* a =
          heatFlux->fetch_arr() + HeatFlux::componentIndex(is, c) * nodes;
#pragma omp parallel for collapse(2) schedule(static)
      for (int i = lo[0]; i <= hi[0]; i++)
        for (int j = lo[1]; j <= hi[1]; j++)
          for (int k = lo[2]; k <= hi[2]; k++)
            a[((size_t)i * nyn + j) * nzn + k] *= 2;
    }
  };

  if (vct->getXleft_neighbor_P() == MPI_PROC_NULL)
    doublePlane(0, 1);
  if (vct->getYleft_neighbor_P() == MPI_PROC_NULL)
    doublePlane(1, 1);
  if (vct->getZleft_neighbor_P() == MPI_PROC_NULL)
    doublePlane(2, 1);
  if (vct->getXright_neighbor_P() == MPI_PROC_NULL)
    doublePlane(0, nxn - 2);
  if (vct->getYright_neighbor_P() == MPI_PROC_NULL)
    doublePlane(1, nyn - 2);
  if (vct->getZright_neighbor_P() == MPI_PROC_NULL)
    doublePlane(2, nzn - 2);
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

bool EMfields3D::hasPlanet() const {
  const std::string simCase = get_col().getCase();
  return simCase == "Dipole" || simCase == "Dipole2D";
}

bool EMfields3D::isPlanetNode(int i, int j, int k, bool planeXZ) const {
  const Grid* grid = &get_grid();
  const double xd = grid->getGlobalXN(i) - x_center_planet;
  const double yd = planeXZ ? 0.0 : grid->getGlobalYN(j) - y_center_planet;
  const double zd = grid->getGlobalZN(k) - z_center_planet;
  return xd * xd + yd * yd + zd * zd <= Planet_radius * Planet_radius;
}

PinnedVector<int> EMfields3D::planetNodes() const {
  const bool planeXZ = get_col().getCase() == "Dipole2D";
  if (planeXZ)
    assert_eq(nyn, 4);
  PinnedVector<int> nodes;
  // every local node, the ghosts too: a ghost is a copy of a node of the
  // neighbour, which holds the planet charge there as well
  for (int i = 0; i < nxn; i++)
    for (int j = 0; j < nyn; j++)
      for (int k = 0; k < nzn; k++)
        if (isPlanetNode(i, j, k, planeXZ))
          nodes.push_back((i * nyn + j) * nzn + k);
  return nodes;
}

double EMfields3D::planetChargeDensity(int is) const {
  return qom[is] / fabs(qom[is]) * rhoINIT[is] / FourPI;
}

void EMfields3D::ConstantChargePlanet() {
  const PinnedVector<int> nodes = planetNodes();
  for (int is = 0; is < ns; is++) {
    const double val = planetChargeDensity(is);
    double* rho = &rhons[is][0][0][0];
    for (int n : nodes)
      rho[n] = val;
  }
}

void EMfields3D::allocateHeatFlux() {
  heatFlux = std::make_unique<pinned_array4_double>(
      ns * HeatFlux::ComponentCount, nxn, nyn, nzn);
}

void EMfields3D::communicateGhostHeatFlux(int is) {
  const VirtualTopology3D* vct = &get_vct();

  for (int c = 0; c < HeatFlux::ComponentCount; ++c) {
    double*** component =
        convert_to_arr3((*heatFlux)[HeatFlux::componentIndex(is, c)]);
    communicateInterp(nxn, nyn, nzn, component, vct, this);
  }

  adjustNonPeriodicHeatFlux(is);

  for (int c = 0; c < HeatFlux::ComponentCount; ++c) {
    double*** component =
        convert_to_arr3((*heatFlux)[HeatFlux::componentIndex(is, c)]);
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

void EMfields3D::driftElectricField(double E0[3]) const {
  E0[0] = -(ve0 * B0z - we0 * B0y) / c;
  E0[1] = -(we0 * B0x - ue0 * B0z) / c;
  E0[2] = -(ue0 * B0y - ve0 * B0x) / c;
}

// =========================================================================
//  Zero-gradient ghost node on the X-right outflow face (Maxwell operator)
// =========================================================================

int EMfields3D::operatorCellEnd() const {
  return bc_.spec().outflowXright ? nxc - 1 : nxc - 2;
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

void EMfields3D::ownedNodeWeights(std::vector<double> w[3]) const {
  const VirtualTopology3D& vct = get_vct();
  int end[3];
  ownedNodeEnds(end[0], end[1], end[2]);
  const bool lowFace[3] = {vct.getXleft_neighbor() == MPI_PROC_NULL,
                           vct.getYleft_neighbor() == MPI_PROC_NULL,
                           vct.getZleft_neighbor() == MPI_PROC_NULL};
  const bool highFace[3] = {vct.getXright_neighbor() == MPI_PROC_NULL,
                            vct.getYright_neighbor() == MPI_PROC_NULL,
                            vct.getZright_neighbor() == MPI_PROC_NULL};
  for (int d = 0; d < 3; d++) {
    w[d].assign(end[d] + 1, 1.0); // w[d][0] unused: the owned nodes are 1..end
    if (lowFace[d])
      w[d][1] *= 0.5;
    if (highFace[d])
      w[d][end[d]] *= 0.5;
  }
}

double EMfields3D::getLocalEenergy() {
  std::vector<double> w[3];
  ownedNodeWeights(w);
  const int ni = (int)w[0].size(), nj = (int)w[1].size(), nk = (int)w[2].size();
  const double *wx = w[0].data(), *wy = w[1].data(), *wz = w[2].data();
  const double *ex = Ex.get_arr(), *ey = Ey.get_arr(), *ez = Ez.get_arr();
  double sum = 0.0; // sum of w |E|^2; k innermost, contiguous
#pragma omp parallel for collapse(2) reduction(+ : sum) schedule(static)
  for (int i = 1; i < ni; i++)
    for (int j = 1; j < nj; j++) {
      const size_t row = ((size_t)i * nyn + j) * nzn;
      double s = 0.0;
      for (int k = 1; k < nk; k++) {
        const size_t n = row + k;
        s += wz[k] * (ex[n] * ex[n] + ey[n] * ey[n] + ez[n] * ez[n]);
      }
      sum += wx[i] * wy[j] * s;
    }
  return .5 * dx * dy * dz * sum / FourPI;
}

double EMfields3D::getLocalBenergy() {
  const double *bx = Bxc.get_arr(), *by = Byc.get_arr(), *bz = Bzc.get_arr();
  const double *ex = Bx_ext.get_arr(), *ey = By_ext.get_arr(),
               *ez = Bz_ext.get_arr();
  // B_ext lives on the nodes: its value at the cell centre is the average of
  // the 8 corners, as interpN2C; n is the corner (i, j, k) of cell (i, j, k)
  const size_t sj = nzn, si = (size_t)nyn * nzn;
  auto atCentre = [si, sj](const double* f, size_t n) {
    return 0.125 * (f[n] + f[n + si] + f[n + sj] + f[n + 1] + f[n + si + sj] +
                    f[n + si + 1] + f[n + sj + 1] + f[n + si + sj + 1]);
  };
  double sum = 0.0; // sum of |B_c + B_ext|^2; k innermost, contiguous
#pragma omp parallel for collapse(2) reduction(+ : sum) schedule(static)
  for (int i = 1; i < nxc - 1; i++)
    for (int j = 1; j < nyc - 1; j++) {
      const size_t cell = ((size_t)i * nyc + j) * nzc;
      const size_t node = ((size_t)i * nyn + j) * nzn;
      double s = 0.0;
      for (int k = 1; k < nzc - 1; k++) {
        const double Bxt = bx[cell + k] + atCentre(ex, node + k);
        const double Byt = by[cell + k] + atCentre(ey, node + k);
        const double Bzt = bz[cell + k] + atCentre(ez, node + k);
        s += Bxt * Bxt + Byt * Byt + Bzt * Bzt;
      }
      sum += s;
    }
  return .5 * dx * dy * dz * sum / FourPI;
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
  std::vector<double> w[3];
  ownedNodeWeights(w);
  const int ni = (int)w[0].size(), nj = (int)w[1].size(), nk = (int)w[2].size();
  const double *wx = w[0].data(), *wy = w[1].data(), *wz = w[2].data();
  const size_t species = (size_t)is * nxn * nyn * nzn;
  const double* rho = rhons.get_arr() + species;
  const double *jx = Jxs.get_arr() + species, *jy = Jys.get_arr() + species,
               *jz = Jzs.get_arr() + species;
  double sum = 0.0; // sum of w |J|^2 / rho; k innermost, contiguous
#pragma omp parallel for collapse(2) reduction(+ : sum) schedule(static)
  for (int i = 1; i < ni; i++)
    for (int j = 1; j < nj; j++) {
      const size_t row = ((size_t)i * nyn + j) * nzn;
      double s = 0.0;
      for (int k = 1; k < nk; k++) {
        const size_t n = row + k;
        // where rho is 0 the current is 0 too
        s += fabs(rho[n]) > 1.e-20
                 ? wz[k] * (jx[n] * jx[n] + jy[n] * jy[n] + jz[n] * jz[n]) /
                       rho[n]
                 : 0.0;
      }
      sum += wx[i] * wy[j] * s;
    }
  const double localBenergy = 0.5 * dx * dy * dz * sum;
  double totalBenergy = 0.0;
  MPI_Allreduce(&localBenergy, &totalBenergy, 1, MPI_DOUBLE, MPI_SUM,
                (&get_vct())->getFieldComm());
  return (totalBenergy / qom[is]);
}

/*! Print info about electromagnetic field */
void EMfields3D::print(void) const {}

/*! destructor*/
EMfields3D::~EMfields3D() {
  AlignedFreePinned(qom);
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

    if (hasPlanet()) {
      ConstantChargePlanet();
    }
    // TaylorGreen: reset the species densities on the node planes of each
    // open face (bcEMface = 2) to sign(q) rhoINIT / 4pi
    else if (col->getCase().find("TaylorGreen") != std::string::npos) {
      ConstantChargeOpenBC();
    }

    // B on the centres: interpN2C reads the active nodes only.  The ghosts
    // of E, B and the cell B are filled by the GPU start-up
    // (gpuSolverSyncH2D, FieldBoundary).
    grid->interpN2C(Bxc, Bxn);
    grid->interpN2C(Byc, Byn);
    grid->interpN2C(Bzc, Bzn);

    for (int is = 0; is < ns; is++)
      grid->interpN2C(rhocs, is, rhons);
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

void EMfields3D::initSpeciesCurrents(bool fromAmpere) {
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

  for (int is = 0; is < ns; is++) {
    const double qs = (col->getQOM(is) > 0.0) ? 1.0 : -1.0;
    const double wU = (sumU != 0.0) ? qs * col->getU0(is) / sumU : 0.0;
    const double wV = (sumV != 0.0) ? qs * col->getV0(is) / sumV : 0.0;
    const double wW = (sumW != 0.0) ? qs * col->getW0(is) / sumW : 0.0;
    const double factU = wU * c / FourPI;
    const double factV = wV * c / FourPI;
    const double factW = wW * c / FourPI;

#pragma omp parallel for collapse(3)
    for (int i = 0; i < nxn; i++)
      for (int j = 0; j < nyn; j++)
        for (int k = 0; k < nzn; k++) {
          Jxs[is][i][j][k] = factU * tempXN[i][j][k];
          Jys[is][i][j][k] = factV * tempYN[i][j][k];
          Jzs[is][i][j][k] = factW * tempZN[i][j][k];
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

  // --- the perturbations as one flux function A_p (Bx = dA/dy,
  //     By = -dA/dx), from the box centre (xM, yM) ---
  // GEM (Birn et al. 2001): psiGEM cos(kxG xM) cos(kyG yM), psiGEM =
  // pertGEM B0x / kxG, so that delta By peaks at pertGEM B0x; kyG = pi/Ly
  // leaves By = 0 on the walls yM = +-Ly/2.  Hump:
  // A0 exp(-xM^2/deltax^2 - yM^2/deltay^2) cos(kx xM) cos(kyH yM).
  const double kxG = 2.0 * M_PI / Lx;
  const double kyG = M_PI / Ly;
  const double psiGEM = pertGEM * B0x / kxG;
  const double kx =
      (col->getKxHump() < 0.0) ? 2.0 * M_PI / Lx : col->getKxHump();
  const double kyH = (col->getKyHump() < 0.0) ? M_PI / Ly : col->getKyHump();
  const double A0 = (pertHump != 0.0) ? pertHump * B0x / kyH : 0.0;
  struct Flux {
    double a, dx, dy; // A_p, dA_p/dx, dA_p/dy
  };
  const auto perturbation = [&](double xM, double yM) {
    Flux f{0.0, 0.0, 0.0};
    if (pertGEM != 0.0) {
      const double cx = cos(kxG * xM), sx = sin(kxG * xM);
      const double cy = cos(kyG * yM), sy = sin(kyG * yM);
      f.a += psiGEM * cx * cy;
      f.dx -= psiGEM * kxG * sx * cy;
      f.dy -= psiGEM * kyG * cx * sy;
    }
    if (pertHump != 0.0) {
      const double g =
          exp(-xM * xM / (deltax * deltax) - yM * yM / (deltay * deltay));
      const double cx = cos(kx * xM), sx = sin(kx * xM);
      const double cy = cos(kyH * yM), sy = sin(kyH * yM);
      f.a += A0 * g * cx * cy;
      f.dx += A0 * g * cy * (-2.0 * xM / (deltax * deltax) * cx - kx * sx);
      f.dy += A0 * g * cx * (-2.0 * yM / (deltay * deltay) * cy - kyH * sy);
    }
    return f;
  };
  const auto xMN = [&](int i, int j, int k) {
    return grid->getXN(i, j, k) - 0.5 * Lx;
  };
  const auto yMN = [&](int i, int j, int k) {
    return grid->getYN(i, j, k) - 0.5 * Ly;
  };
  // node fields to the centres (interpN2C reads the active nodes only; the
  // ghosts of B are filled by the GPU start-up, gpuSolverSyncH2D)
  const auto nodesToCentres = [&](arr3_double xn, arr3_double yn,
                                  arr3_double zn, arr3_double xc,
                                  arr3_double yc, arr3_double zc) {
    grid->interpN2C(xc, xn);
    grid->interpN2C(yc, yn);
    grid->interpN2C(zc, zn);
  };

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

    // === B (Harris + perturbations), density and E = 0 on the nodes ===
#pragma omp parallel for collapse(3)
    for (int i = 0; i < nxn; i++)
      for (int j = 0; j < nyn; j++)
        for (int k = 0; k < nzn; k++) {
          const double xM = xMN(i, j, k), yM = yMN(i, j, k);
          const double sech = 1.0 / cosh(yM / delta);
          for (int is = 0; is < ns; is++)
            rhons[is][i][j][k] = DriftSpecies[is]
                                     ? rhoINIT[is] * sech * sech / FourPI
                                     : rhoINIT[is] / FourPI;
          Ex[i][j][k] = 0.0;
          Ey[i][j][k] = 0.0;
          Ez[i][j][k] = 0.0;
          const Flux f = perturbation(xM, yM);
          Bxn[i][j][k] = B0x * tanh(yM / delta) + f.dy;
          Byn[i][j][k] = B0y - f.dx;
          Bzn[i][j][k] = B0z;
        }
    nodesToCentres(Bxn, Byn, Bzn, Bxc, Byc, Bzc);
    for (int is = 0; is < ns; is++)
      grid->interpN2C(rhocs, is, rhons);

    // --- current initialization ---
    // Without Ampere, Jxs/Jys/Jzs stay at zero and particle init uses each
    // species' uniform u0/v0/w0 as drift velocities.
    initSpeciesCurrents(false);
    if (ampere) {
      // The Harris current J_z = -(c/4pi) (B0x/delta) sech^2(yM/delta),
      // analytic, is split by the w0 weights sign(q_s) w0_s / sum_r sign(q_r)
      // w0_r: each drifting species gets a uniform drift.  The Harris sheet
      // (E = 0) needs u_s = -2 c T_s / (q_s B0x delta), i.e. the shares
      // n_s T_s ~ rho_s vth_s^2 / |qom_s|; warn when the weights differ.
      double sumW = 0.0, sumP = 0.0;
      for (int is = 0; is < ns; is++) {
        sumW += (col->getQOM(is) > 0.0 ? 1.0 : -1.0) * col->getW0(is);
        if (DriftSpecies[is])
          sumP += rhoINIT[is] * thermalSpeed2(col, is) / fabs(col->getQOM(is));
      }
      std::vector<double> weight(ns, 0.0);
      for (int is = 0; is < ns && sumW != 0.0; is++) {
        weight[is] =
            (col->getQOM(is) > 0.0 ? 1.0 : -1.0) * col->getW0(is) / sumW;
        if (!DriftSpecies[is] || sumP <= 0.0)
          continue;
        const double harris =
            rhoINIT[is] * thermalSpeed2(col, is) / fabs(col->getQOM(is)) / sumP;
        if (fabs(weight[is] - harris) > 0.01 && vct->getCartesian_rank() == 0)
          cout << "WARNING: species " << is << " carries the share "
               << weight[is]
               << " of the Harris current (w0 weights), but the Harris "
                  "equilibrium needs "
               << harris << " (its share of n T); set the w0 in this ratio"
               << endl;
      }
#pragma omp parallel for collapse(3)
      for (int i = 0; i < nxn; i++)
        for (int j = 0; j < nyn; j++)
          for (int k = 0; k < nzn; k++) {
            const double sech = 1.0 / cosh(yMN(i, j, k) / delta);
            const double jHarris = -c / FourPI * B0x / delta * sech * sech;
            for (int is = 0; is < ns; is++)
              Jzs[is][i][j][k] = weight[is] * jHarris;
          }

      // The perturbation current (c/4pi) curl(dB), dB = (dA_p/dy, -dA_p/dx)
      // taken to the centres as B is (the ghost rules are linear), is
      // carried by the drifting (sheet) electrons in proportion to their
      // density.
      if (pertGEM != 0.0 || pertHump != 0.0) {
        bool sheetElectrons = false;
        for (int is = 0; is < ns; is++)
          sheetElectrons |= DriftSpecies[is] && col->getQOM(is) < 0.0;
        if (!sheetElectrons) {
          if (vct->getCartesian_rank() == 0)
            printf("ERROR: GEMHarris with currentFromAmpere = 1 and a "
                   "perturbation needs a drifting electron species (u0 or w0 "
                   "!= 0) to carry the perturbation current\n");
          fflush(stdout);
          MPIdata::exit(1);
        }
#pragma omp parallel for collapse(3)
        for (int i = 0; i < nxn; i++)
          for (int j = 0; j < nyn; j++)
            for (int k = 0; k < nzn; k++) {
              const Flux f = perturbation(xMN(i, j, k), yMN(i, j, k));
              tempXN[i][j][k] = f.dy;
              tempYN[i][j][k] = -f.dx;
              tempZN[i][j][k] = 0.0;
            }
        array3_double dBxc(nxc, nyc, nzc), dByc(nxc, nyc, nzc),
            dBzc(nxc, nyc, nzc);
        nodesToCentres(tempXN, tempYN, tempZN, dBxc, dByc, dBzc);
        // The ghost cells of dB, which the curl reads on the face nodes: the
        // halo and the codes of the cell B (Collective), the only host ghost
        // fill, as the GPU and its FieldBoundary do not exist yet.
        communicateCenterBC(nxc, nyc, nzc, dBxc, col->bcBx[0], col->bcBx[1],
                            col->bcBx[2], col->bcBx[3], col->bcBx[4],
                            col->bcBx[5], vct, this);
        communicateCenterBC(nxc, nyc, nzc, dByc, col->bcBy[0], col->bcBy[1],
                            col->bcBy[2], col->bcBy[3], col->bcBy[4],
                            col->bcBy[5], vct, this);
        communicateCenterBC(nxc, nyc, nzc, dBzc, col->bcBz[0], col->bcBz[1],
                            col->bcBz[2], col->bcBz[3], col->bcBz[4],
                            col->bcBz[5], vct, this);
        eqValue(0.0, tempXN, nxn, nyn, nzn);
        eqValue(0.0, tempYN, nxn, nyn, nzn);
        eqValue(0.0, tempZN, nxn, nyn, nzn);
        grid->curlC2N(tempXN, tempYN, tempZN, dBxc, dByc, dBzc);
#pragma omp parallel for collapse(3)
        for (int i = 0; i < nxn; i++)
          for (int j = 0; j < nyn; j++)
            for (int k = 0; k < nzn; k++) {
              double rhoE = 0.0;
              for (int is = 0; is < ns; is++)
                if (DriftSpecies[is] && col->getQOM(is) < 0.0)
                  rhoE += rhons[is][i][j][k];
              if (rhoE <= 0.0)
                continue;
              for (int is = 0; is < ns; is++) {
                if (!DriftSpecies[is] || col->getQOM(is) >= 0.0)
                  continue;
                const double f = rhons[is][i][j][k] / rhoE * c / FourPI;
                Jxs[is][i][j][k] += f * tempXN[i][j][k];
                Jys[is][i][j][k] += f * tempYN[i][j][k];
                Jzs[is][i][j][k] += f * tempZN[i][j][k];
              }
            }
      }
    }

    // --- spatially varying thermal speed ---
    // The sheet pressure is the Harris law p(A) = p0 exp(-2 A / (B0x delta))
    // of the perturbed flux function A = A_Harris + A_p (A_Harris =
    // B0x delta ln cosh(yM / delta)), so it is constant along the field
    // lines.  With the density n0 sech^2(yM / delta) kept, a drifting (sheet)
    // species has the temperature T_s exp(-2 A_p / (B0x delta)); the
    // background stays isothermal.  p**sn hold the thermal pressure
    // rho vth^2 that maxwellianFromSpeciesCurrents reads.  Unperturbed
    // (A_p = 0) this is the Harris equilibrium.
    if (col->getSpatiallyVaryingThermal()) {
      std::vector<double> uth2(ns), vth2(ns), wth2(ns);
      for (int is = 0; is < ns; is++) {
        uth2[is] = col->getUth(is) * col->getUth(is);
        vth2[is] = col->getVth(is) * col->getVth(is);
        wth2[is] = col->getWth(is) * col->getWth(is);
      }
      double range[2] = {1.0, -1.0}; // min and -max of the factor
#pragma omp parallel for collapse(3) reduction(min : range[ : 2])
      for (int i = 0; i < nxn; i++)
        for (int j = 0; j < nyn; j++)
          for (int k = 0; k < nzn; k++) {
            const double factor =
                exp(-2.0 * perturbation(xMN(i, j, k), yMN(i, j, k)).a /
                    (B0x * delta));
            range[0] = std::min(range[0], factor);
            range[1] = std::min(range[1], -factor);
            for (int is = 0; is < ns; is++) {
              const double rho = rhons[is][i][j][k];
              const double f = DriftSpecies[is] ? factor : 1.0;
              pXXsn[is][i][j][k] = f * uth2[is] * rho;
              pYYsn[is][i][j][k] = f * vth2[is] * rho;
              pZZsn[is][i][j][k] = f * wth2[is] * rho;
            }
          }
      MPI_Allreduce(MPI_IN_PLACE, range, 2, MPI_DOUBLE, MPI_MIN,
                    vct->getFieldComm());
      if (vct->getCartesian_rank() == 0)
        cout << "Sheet temperature from the Harris pressure law of the "
                "perturbed flux (spatiallyVaryingThermal = 1): factor in ["
             << range[0] << ", " << -range[1] << "]" << endl;
    }
  } else {
    init(); // restart
  }
}

void EMfields3D::initNullPoints() {
  const VirtualTopology3D* vct = &get_vct();
  const Grid* grid = &get_grid();
  if (restart1 == 0) {
    // div B = 0 for any box: B_x scaled by k_y/k_x = Lx/Ly and B_z by
    // k_y/k_z = Lz/Ly (1 in a cubic box)
    const double bxAmp = -B0x * (Lx / Ly), bzAmp = 2. * B0x * (Lz / Ly);
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
          Bxn[i][j][k] = bxAmp * cos(2. * M_PI * grid->getXN(i, j, k) / Lx) *
                         sin(2. * M_PI * grid->getYN(i, j, k) / Ly);
          Byn[i][j][k] = B0x * cos(2. * M_PI * grid->getYN(i, j, k) / Ly) *
                         (-2. * sin(2. * M_PI * grid->getZN(i, j, k) / Lz) +
                          sin(2. * M_PI * grid->getXN(i, j, k) / Lx));
          Bzn[i][j][k] = bzAmp * cos(2. * M_PI * grid->getZN(i, j, k) / Lz) *
                         sin(2. * M_PI * grid->getYN(i, j, k) / Ly);
        }

#pragma omp parallel for collapse(3)
    for (int i = 0; i < nxc; i++)
      for (int j = 0; j < nyc; j++)
        for (int k = 0; k < nzc; k++) {
          Bxc[i][j][k] = bxAmp * cos(2. * M_PI * grid->getXC(i, j, k) / Lx) *
                         sin(2. * M_PI * grid->getYC(i, j, k) / Ly);
          Byc[i][j][k] = B0x * cos(2. * M_PI * grid->getYC(i, j, k) / Ly) *
                         (-2. * sin(2. * M_PI * grid->getZC(i, j, k) / Lz) +
                          sin(2. * M_PI * grid->getXC(i, j, k) / Lx));
          Bzc[i][j][k] = bzAmp * cos(2. * M_PI * grid->getZC(i, j, k) / Lz) *
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
  // div u = 0 and div B = 0 for any box: u_y scaled by k_x/k_y = Ly/Lx, B_z
  // by (k_x + k_y)/k_z = Lz/Lx + Lz/Ly (1 and 2 in a cubic box)
  const double uyScale = Ly / Lx, bzAmp = -(Lz / Lx + Lz / Ly) * B0x;
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
            Jys[is][i][j][k] = -flowAmp * uyScale *
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
          Bzn[i][j][k] = bzAmp * sin(2. * M_PI * grid->getXN(i, j, k) / Lx) *
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
          Bzc[i][j][k] = bzAmp * sin(2. * M_PI * grid->getXC(i, j, k) / Lx) *
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
          // density: a drifting species carries both sheets (drift w0 in the
          // lower half, -w0 in the upper one: maxwellianDoubleHarris), the
          // others are uniform
          const double sech_yBd = 1. / cosh(yBd);
          const double sech_yTd = 1. / cosh(yTd);
          for (int is = 0; is < ns; is++)
            rhons[is][i][j][k] =
                DriftSpecies[is]
                    ? rhoINIT[is] *
                          (sech_yBd * sech_yBd + sech_yTd * sech_yTd) / FourPI
                    : rhoINIT[is] / FourPI;
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

    // B on the centres (the ghosts: the GPU start-up, gpuSolverSyncH2D)
    grid->interpN2C(Bxc, Bxn);
    grid->interpN2C(Byc, Byn);
    grid->interpN2C(Bzc, Bzn);
    for (int is = 0; is < ns; is++)
      grid->interpN2C(rhocs, is, rhons);
  } else {
    init(); // use the fields from restart file
  }
}

void EMfields3D::initDoublePeriodicHarrisWithGaussianHumpPerturbation() {
  const Grid* grid = &get_grid();
  // perturbation localized in X
  const double pertX = 0.4;
  const double deltax = 8. * delta;
  const double deltay = 4. * delta;
  const double yRatio = deltay / deltax; // By of the flux-derived humps
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
          // the X perturbations of the two sheets: the curl of the fluxes
          // +-B0x pertX deltay hump (Bx = dA/dy, By = -dA/dx), div-free
          const double xMdx = xM / deltax;
          const double yBdy = yB / deltay;
          const double yTdy = yT / deltay;
          const double humpB = exp(-xMdx * xMdx - yBdy * yBdy);
          Bxn[i][j][k] -= (B0x * pertX) * humpB * (2.0 * yBdy);
          Byn[i][j][k] += (B0x * pertX) * humpB * (2.0 * xMdx) * yRatio;
          const double humpT = exp(-xMdx * xMdx - yTdy * yTdy);
          Bxn[i][j][k] += (B0x * pertX) * humpT * (2.0 * yTdy);
          Byn[i][j][k] -= (B0x * pertX) * humpT * (2.0 * xMdx) * yRatio;

          // guide field
          Bzn[i][j][k] = B0z;
        }
    // initialize B on centers
#pragma omp parallel for collapse(3)
    for (int i = 0; i < nxc; i++)
      for (int j = 0; j < nyc; j++)
        for (int k = 0; k < nzc; k++) {
          const double xM = grid->getXC(i, j, k) - .5 * Lx;
          const double yB = grid->getYC(i, j, k) - .25 * Ly;
          const double yT = grid->getYC(i, j, k) - .75 * Ly;
          const double yBd = yB / delta;
          const double yTd = yT / delta;
          Bxc[i][j][k] = B0x * (-1.0 + tanh(yBd) - tanh(yTd));
          Byc[i][j][k] = B0y;
          // the X perturbations, as on the nodes
          const double xMdx = xM / deltax;
          const double yBdy = yB / deltay;
          const double yTdy = yT / deltay;
          const double humpB = exp(-xMdx * xMdx - yBdy * yBdy);
          Bxc[i][j][k] -= (B0x * pertX) * humpB * (2.0 * yBdy);
          Byc[i][j][k] += (B0x * pertX) * humpB * (2.0 * xMdx) * yRatio;
          const double humpT = exp(-xMdx * xMdx - yTdy * yTdy);
          Bxc[i][j][k] += (B0x * pertX) * humpT * (2.0 * yTdy);
          Byc[i][j][k] -= (B0x * pertX) * humpT * (2.0 * xMdx) * yRatio;
          // guide field
          Bzc[i][j][k] = B0z;
        }
    for (int is = 0; is < ns; is++)
      grid->interpN2C(rhocs, is, rhons);
  } else {
    init(); // use the fields from restart file
  }
}

void EMfields3D::initHumpPerturbation() {
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
  const double x1 = Lx / 6.0; // minimal position of the gaussian peak
  const double x2 = Lx / 4.0; // maximal position of the gaussian peak (the one
                              // closer to the center)
  const double sigma = Lx / 15; // width of the gaussian
  const double stretch_curve =
      2.0; // stretch the sin^2 function over the x dimension - also can
           // regulate the number of 'knots/reconnecitons points' if less than 1
  const double skew_parameter = 0.50; // skew of the shape of the gaussian

  // The tail field Bx = B0x/2 (exp(-r2^2/sigma^2) - exp(-r1^2/sigma^2)),
  // r_i = (y - y_i(x)) s(x), comes from the flux function (Bx = dA/dy)
  //   A = B0x sigma sqrt(pi) / (4 s) (erf(r2/sigma) - erf(r1/sigma)),
  // and By = B0y - dA/dx, so that div B = 0; with
  //   y1 = x1 + d(x), y2 = Lx - x1 - d(x), d = x1 - x2 sin^2(a x'),
  //   s = 1 - skew sin^2(b x'),  x' = x - Lx/2, a = pi/(Lx stretch),
  //   b = pi/Lx.
  const double aW = M_PI / (Lx * stretch_curve), bW = M_PI / Lx;
  const double K = B0x * sigma * sqrt(M_PI) / 4.0;
  auto field = [&](double x, double y, double& bx, double& by) {
    const double xp = x - Lx / 2;
    const double sa = sin(aW * xp), sb = sin(bW * xp);
    const double d = x1 - x2 * sa * sa;
    const double dd = -x2 * sin(2.0 * aW * xp) * aW; // d'(x)
    const double sk = 1.0 - skew_parameter * sb * sb;
    const double dsk = -skew_parameter * sin(2.0 * bW * xp) * bW; // s'(x)
    const double y1 = x1 + d, y2 = (Lx - x1) - d;
    const double u1 = (y - y1) * sk / sigma, u2 = (y - y2) * sk / sigma;
    const double g1 = exp(-u1 * u1), g2 = exp(-u2 * u2);
    bx = 0.5 * B0x * (g2 - g1);
    // dA/dx: the 1/s factor, then the erf arguments (y1' = d', y2' = -d')
    const double du1 = (-dd * sk + (y - y1) * dsk) / sigma;
    const double du2 = (dd * sk + (y - y2) * dsk) / sigma;
    const double dAdx = -K * dsk / (sk * sk) * (erf(u2) - erf(u1)) +
                        K / sk * (2.0 / sqrt(M_PI)) * (g2 * du2 - g1 * du1);
    by = B0y - dAdx;
  };

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
          // tail-like field topology, guide field
          field(grid->getXN(i, j, k), grid->getYN(i, j, k), Bxn[i][j][k],
                Byn[i][j][k]);
          Bzn[i][j][k] = B0z;
        }
    // initialize B on centers
#pragma omp parallel for collapse(3)
    for (int i = 0; i < nxc; i++)
      for (int j = 0; j < nyc; j++)
        for (int k = 0; k < nzc; k++) {
          field(grid->getXC(i, j, k), grid->getYC(i, j, k), Bxc[i][j][k],
                Byc[i][j][k]);
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
    // initialize B on centers
    grid->interpN2C(Bxc, Bxn);
    grid->interpN2C(Byc, Byn);
    grid->interpN2C(Bzc, Bzn);
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
    // species by u0/v0/w0
    initSpeciesCurrents(true);
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
  driftElectricField(ebc); // E0 = -(u0 x B0) / c

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

        // node coordinates from global indices: the same bits in every
        // decomposition
        double x = grid->getGlobalXN(i);
        double y = grid->getGlobalYN(j);
        double z = grid->getGlobalZN(k);

        // Distance from dipole center
        double r2d = ((x - x_center_dipole) * (x - x_center_dipole)) +
                     ((y - y_center_dipole) * (y - y_center_dipole)) +
                     ((z - z_center_dipole) * (z - z_center_dipole));

        // Compute dipolar field B_ext

        if (!isPlanetNode(i, j, k, false)) {
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
  driftElectricField(ebc); // E0 = -(u0 x B0) / c

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

        // node coordinates from global indices: the same bits in every
        // decomposition
        double x = grid->getGlobalXN(i);
        double z = grid->getGlobalZN(k);

        // Distance from dipole center (for field formula)
        double r2d = ((x - xc) * (x - xc)) + ((z - zc) * (z - zc));

        // Compute dipolar field B_ext

        if (!isPlanetNode(i, j, k, true)) {
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
