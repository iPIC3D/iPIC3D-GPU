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

/*******************************************************************************************************
  Grid3DCUCU.h  -  uniform cartesian 3D local grid for each process, including
 che guard cells
  -------------------

 *******************************************************************************************************/

#ifndef GRID_3D_CU_H
#define GRID_3D_CU_H

#include "arraysfwd.h"
#include "ipicfwd.h"
#include "math.h" // for floor

/**
 * Uniform cartesian local grid 3D
 *
 * @date Fri Jun 4 2008
 * @par Copyright:
 * @author Stefano Markidis, Giovanni Lapenta
 * @version 3.0
 *
 */
class Grid3DCU // :public Grid
{
public:
  /** constructor */
  Grid3DCU(CollectiveIO* col, VirtualTopology3D* vct);
  Grid3DCU(int nxc_, int nyc_, int nzc_, double dx_, double dy_, double dz_,
           double xStart_, double yStart_, double zStart_);
  /** destructor */
  ~Grid3DCU();
  /** allocate grid arrays for this domain */
  // void allocate(CollectiveIO * ptC, VirtualTopology3D * ptVCT);
  /** deallocate grid arrays for this domain */
  // void deallocate();
  /** print grid info (not used) */
  void print() const;
  /** calculate a derivative along a direction on nodes */
  // void derivN(arr3_double derN,
  //   const_arr4_double scFieldC, int ns, int dir)const;
  /** calculate gradient on nodes, given a scalar field defined on central
   * points  */
  void gradC2N(arr3_double gradXN, arr3_double gradYN, arr3_double gradZN,
               const_arr3_double scFieldC) const;
  /** calculate gradient on central points, given a scalar field defined on
   * nodes: cells i = 1..iHi (iHi < 0: nxc-2, the interior), j and k interior;
   * iHi = nxc-1 also evaluates the X-right ghost cell plane.  Not used. */
  void gradN2C(arr3_double gradXC, arr3_double gradYC, arr3_double gradZC,
               const_arr3_double scFieldN, int iHi = -1) const;
  /** calculate divergence on central points, given a vector field defined on
   * nodes: cells i = 1..iHi as for gradN2C */
  void divN2C(arr3_double divC, const_arr3_double vecFieldXN,
              const_arr3_double vecFieldYN, const_arr3_double vecFieldZN,
              int iHi = -1) const;
  /** calculate divergence on nodes, given a vector field defined on central
   * points (not used) */
  void divC2N(arr3_double divN, const_arr3_double vecFieldXC,
              const_arr3_double vecFieldYC, const_arr3_double vecFieldZC) const;
  /** calculate curl on nodes, given a vector field defined on central points */
  void curlC2N(arr3_double curlXN, arr3_double curlYN, arr3_double curlZN,
               const_arr3_double vecFieldXC, const_arr3_double vecFieldYC,
               const_arr3_double vecFieldZC) const;
  /** calculate curl on central points, given a vector field defined on nodes
   * (not used) */
  void curlN2C(arr3_double curlXC, arr3_double curlYC, arr3_double curlZC,
               const_arr3_double vecFieldXN, const_arr3_double vecFieldYN,
               const_arr3_double vecFieldZN) const;

  /** calculate divergence on central points, given a Tensor field defined on
   * nodes (not used) */
  void divSymmTensorN2C(arr3_double divCX, arr3_double divCY, arr3_double divCZ,
                        const_arr4_double pXX, const_arr4_double pXY,
                        const_arr4_double pXZ, const_arr4_double pYY,
                        const_arr4_double pYZ, const_arr4_double pZZ,
                        int ns) const;

  /** calculate laplacian on central points, given a scalar field defined on
   * central points for Poisson (not used) */
  // void lapC2Cpoisson(arr3_double lapC, arr3_double scFieldC)const;
  void lapC2Cpoisson(arr3_double lapC, arr3_double scFieldC,
                     EMfields3D* EMf) const;
  /** calculate laplacian on central points, given a scalar field defined on
   * central points (not used) */
  void lapC2C(arr3_double lapC, const_arr3_double scFieldC) const;

  /** calculate divergence on boundaries (not used) */
  void divBCleft(arr3_double divBC, const_arr3_double vectorX,
                 const_arr3_double vectorY, const_arr3_double vectorZ,
                 int leftActiveNode, int dirDER) const;
  /** calculate divergence on boundaries (not used) */
  void divBCright(arr3_double divBC, const_arr3_double vectorX,
                  const_arr3_double vectorY, const_arr3_double vectorZ,
                  int rightActiveNode, int dirDER) const;
  /** calculate derivative on boundaries */
  void derBC(arr3_double derBC, const_arr3_double vector, int leftActiveNode,
             int dirDER) const;

  /** interpolate on nodes from central points */
  void interpC2N(arr3_double vecFieldN, const_arr3_double vecFieldC) const;
  /** interpolate on central points from nodes */
  void interpN2C(arr3_double vecFieldC, const_arr3_double vecFieldN) const;
  /** interpolate on central points from nodes */
  void interpN2C(arr4_double vecFieldC, int ns,
                 const_arr4_double vecFieldN) const;

private:
  void init_derived_parameters();

  // /////////// PRIVATE VARIABLES //////////////
  //
  // Grid conventions:
  //
  //   [The purpose of this note is not so much to teach as
  //   to establish a consistent vocabulary and set of naming
  //   conventions for grid parameters.]
  //
  //   The purpose of iPic3D is to solve a Vlasov-Maxwell
  //   plasma problem.  Each problem is solved on a box-shaped
  //   domain and is defined by initial conditions and boundary
  //   conditions.
  //
  //   Conceptually, the full problem domain is solved on
  //   a "guarded" global grid.  The guarded global grid
  //   consists of proper grid cells (that partition the domain)
  //   surrounded by a layer of "guard" or "ghost" cells that are
  //   used to handle boundary conditions.  At the beginning of
  //   each cycle of the algorithm, the guard cells are populated
  //   with data based on the choice of boundary conditions of
  //   the problem.  Refer to the layer of proper cells just
  //   inside the guard cells as "(proper) boundary cells".
  //   Guard cell data is generally populated with data from
  //   boundary cells.  For periodic boundary conditions, guard
  //   cell data is simply copied from the appropriate boundary
  //   cell.  For conducting wall and outflow boundaries, each
  //   guard cell is populated based on the data from the
  //   neighboring boundary cell, probably with reflection of
  //   location and often with negation of some values, depending
  //   on the type of boundaries involved.
  //
  //   The corners of the mesh cells are referred to as
  //   "nodes".  In iPic3D, electromagnetic field quantities are
  //   canonically represented as sample values at the nodes, and
  //   the value of the electromagnetic field in the interior of
  //   a grid cell is determined by bilinear interpolation of
  //   the field values at the nodes.  Bilinear interpolation
  //   is also used when converting between a representation as
  //   cell-average quantities and a representation as nodal
  //   samples.  This maintains second-order accuracy in space.
  //
  //   To parallelize with MPI, the full domain of the problem
  //   is partitioned into "(proper) subdomains", one for
  //   each process, such that the global grid partitions the
  //   subdomains.  Refer to the grid cells within a subdomain
  //   as its "(proper) subgrid".  Refer to the outer layer of
  //   cells within each subgrid as "boundary cells".  The grid
  //   defined in each process is a "guarded subgrid", which
  //   consists of the proper subgrid surrounded by a layer of
  //   guard cells corresponding either to the boundary cells of
  //   the neighboring subdomain or to ghost cells of the guarded
  //   global grid.
  //
  //   The guard cells thus lie outside the proper physical
  //   subdomain for which the process is responsible.  Guard
  //   cells are used to handle boundary conditions and
  //   interprocess communication.  At the beginning of each
  //   cycle of the algorithm, the guard cells are repopulated.
  //   Guard cells outside the problem domain are populated using
  //   boundary conditions.  Any guard cell inside the problem
  //   domain is populated from the appropriate boundary cell of
  //   the appropriate process.
  //
  // Conventions for naming of grid quantities:
  //
  //   n = number
  //   x = x-direction
  //   y = y-direction
  //   z = z-direction
  //   c = cell of grid
  //   n = node of grid
  //
  //   nxc = number of x cells (in guarded grid)
  //   nxn = number of x nodes (in guarded grid)
  //
  //   nxc_g = number of x cells in guarded/"greater"/padded subgrid
  //   nxc_r = number of x cells in "real"/"restricted"/proper subgrid
  //
private:
  const VirtualTopology3D& _vct;
  /** number of cells - X direction, including + 2 (guard cells) */
  int nxc;
  /** number of nodes - X direction, including + 2 extra nodes for guard cells
   */
  int nxn;
  /** number of cell - Y direction, including + 2 (guard cells) */
  int nyc;
  /** number of nodes - Y direction, including + 2 extra nodes for guard cells
   */
  int nyn;
  /** number of cell - Z direction, including + 2 (guard cells) */
  int nzc;
  /** number of nodes - Z direction, including + 2 extra nodes for guard cells
   */
  int nzn;
  /** dx = space step - X direction */
  double dx;
  /** dy = space step - Y direction */
  double dy;
  /** dz = space step - Z direction */
  double dz;
  /** invdx = 1/dx */
  double invdx;
  /** invdy = 1/dy */
  double invdy;
  /** invdz = 1/dz */
  double invdz;
  /** volume of mesh cell*/
  double VOL;
  /** invol = inverse of volume*/
  double invVOL;
  // number of cells excluding guard cells
  // (i.e. restricted to proper subdomain)
  int nxc_r;
  int nyc_r;
  int nzc_r;
  // number of subdomain cells in a regular (untruncated) subdomain grid
  int num_cells_rr;
  /** node coordinate */
  pfloat* pfloat_node_xcoord;
  pfloat* pfloat_node_ycoord;
  pfloat* pfloat_node_zcoord;
  double* node_xcoord;
  double* node_ycoord;
  double* node_zcoord;
  /** center coordinate */
  double* center_xcoord;
  double* center_ycoord;
  double* center_zcoord;
  /** local grid boundaries coordinate of proper subdomain */
  double xStart, xEnd, yStart, yEnd, zStart, zEnd;
  /** global cell index of the first proper cell (local index 1) */
  int cellOffX = 0, cellOffY = 0, cellOffZ = 0;

public: // accessors (inline)
  const VirtualTopology3D& get_vct() const { return _vct; }
  int getNXC() const { return (nxc); }
  int getNXN() const { return (nxn); }
  int getNYC() const { return (nyc); }
  int getNYN() const { return (nyn); }
  int getNZC() const { return (nzc); }
  int getNZN() const { return (nzn); }
  int get_nxc_r() const { return nxc_r; }
  int get_nyc_r() const { return nyc_r; }
  int get_nzc_r() const { return nzc_r; }
  int get_num_cells_rr() const { return num_cells_rr; } // not used
  double getDX() const { return (dx); }
  double getDY() const { return (dy); }
  double getDZ() const { return (dz); }
  double get_invdx() const { return (invdx); }
  double get_invdy() const { return (invdy); }
  double get_invdz() const { return (invdz); }
  //
  // coordinate accessors
  //
  // calculated equivalents (not used)
  double calcXN(int X) const { return xStart + (X - 1) * dx; }
  double calcYN(int Y) const { return yStart + (Y - 1) * dy; }
  double calcZN(int Z) const { return zStart + (Z - 1) * dz; }
  // const pfloat &get_pfloat_XN(int X)const{ return pfloat_node_xcoord[X];}
  // const pfloat &get_pfloat_YN(int Y)const{ return pfloat_node_ycoord[Y];}
  // const pfloat &get_pfloat_ZN(int Z)const{ return pfloat_node_zcoord[Z];}
  const double& getXN(int X) const { return node_xcoord[X]; }
  const double& getYN(int Y) const { return node_ycoord[Y]; }
  const double& getZN(int Z) const { return node_zcoord[Z]; }
  const double& getXC(int X) const { return center_xcoord[X]; }
  const double& getYC(int Y) const { return center_ycoord[Y]; }
  const double& getZC(int Z) const { return center_zcoord[Z]; }
  //
  // The following could be eliminated in favor of the previous
  // unless we truly anticipate generalizing to a deformed
  // logically cartesian mesh.  See issue #40.
  //
  const double& getXN(int X, int Y, int Z) const { return node_xcoord[X]; }
  const double& getYN(int X, int Y, int Z) const { return node_ycoord[Y]; }
  const double& getZN(int X, int Y, int Z) const { return node_zcoord[Z]; }
  const double& getXC(int X, int Y, int Z) const { return center_xcoord[X]; }
  const double& getYC(int X, int Y, int Z) const { return center_ycoord[Y]; }
  const double& getZC(int X, int Y, int Z) const { return center_zcoord[Z]; }
  //
  double getXstart() const { return (xStart); }
  /** global cell index of local cell 1 in x, y, z */
  int getCellOffsetX() const { return cellOffX; }
  int getCellOffsetY() const { return cellOffY; }
  int getCellOffsetZ() const { return cellOffZ; }
  /** Coordinate of local node X (Y, Z) from its global node index,
   *  (cellOff + X - 1) * dx.  Every decomposition computes the same bits for
   *  the same node, while getXN (xStart + (X - 1) * dx) can differ in the last
   *  bit; the planet masks and the dipole field use these so that they agree
   *  in every decomposition. */
  double getGlobalXN(int X) const { return (cellOffX + X - 1) * dx; }
  double getGlobalYN(int Y) const { return (cellOffY + Y - 1) * dy; }
  double getGlobalZN(int Z) const { return (cellOffZ + Z - 1) * dz; }
  /** Centre of local cell X (Y, Z) from the global-index node coordinates. */
  double getGlobalXC(int X) const {
    return .5 * (getGlobalXN(X) + getGlobalXN(X + 1));
  }
  double getGlobalYC(int Y) const {
    return .5 * (getGlobalYN(Y) + getGlobalYN(Y + 1));
  }
  double getGlobalZC(int Z) const {
    return .5 * (getGlobalZN(Z) + getGlobalZN(Z + 1));
  }
  double getXend() const { return (xEnd); }
  double getYstart() const { return (yStart); }
  double getYend() const { return (yEnd); }
  double getZstart() const { return (zStart); }
  double getZend() const { return (zEnd); }
  double getVOL() const { return (VOL); }
  double getInvVOL() const { return (invVOL); }

  /** The cell of position (x, y, z), cell 0 the lower ghost cell and cell 1
   *  the first proper one: a position outside the guarded subdomain,
   *  infinite or NaN goes to the nearest edge cell (NaN to cell 0), clamped
   *  before the conversion to int. */
  void get_safe_cell_coordinates(int& cx, int& cy, int& cz, double x, double y,
                                 double z) const {
    cx = safe_cell((x - xStart) * invdx, nxc - 1);
    cy = safe_cell((y - yStart) * invdy, nyc - 1);
    cz = safe_cell((z - zStart) * invdz, nzc - 1);
  }

private:
  /** 1 + floor(pos) in [0, last]; !(c >= 0) also catches NaN. */
  static int safe_cell(double pos, int last) {
    const double c = 1 + floor(pos);
    if (!(c >= 0))
      return 0;
    return c > last ? last : int(c);
  }
};

typedef Grid3DCU Grid;

#endif // GRID_3D_CU_H
