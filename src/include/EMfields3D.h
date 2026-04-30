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

/**
 * @file EMfields3D.h
 * @brief Electromagnetic field storage and solver interface for one local domain.
 */

#ifndef EMfields3D_H
#define EMfields3D_H

#include "asserts.h"
#include "ipicfwd.h"
#include "Alloc.h"
#include "Basic.h"
#include "mpi.h"

#include "cudaTypeDef.cuh"

// dimension of vectors used in fieldForPcls
const int DFIELD_3or4=4; // 4 pads with garbage but is needed for alignment

/**
 * @brief Electromagnetic fields, sources, and implicit-solver work arrays for one MPI rank.
 *
 * The solver uses this class to initialize case-dependent fields, advance the
 * implicit Maxwell system, accumulate and reduce particle moments, and prepare
 * field buffers consumed by the GPU particle mover.
 */
class EMfields3D                // :public Field
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
    EMfields3D(Collective * col, Grid * grid, VirtualTopology3D *vct);
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
    void initBEAM(double x_center, double y_center, double z_center, double radius);
    /** @brief Initialize the standard GEM challenge configuration. */
    void initGEM();
    /** @brief Single Harris sheet with optional GEM/hump perturbations and optional Ampère current. */
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
     * @param deltaBoB Magnetic perturbation amplitude normalized by the background field.
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
    void AddPerturbationRho(double deltaBoB, double kx, double ky, double Bx_mod, double By_mod, double Bz_mod, double ne_mod, double ne_phase, double ni_mod, double ni_phase, double B0, Grid * grid);
    /**
     * @brief Add a perturbation to the electromagnetic field.
     *
     * @param deltaBoB Magnetic perturbation amplitude normalized by the background field.
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
    void AddPerturbation(double deltaBoB, double kx, double ky, double Ex_mod, double Ex_phase, double Ey_mod, double Ey_phase, double Ez_mod, double Ez_phase, double Bx_mod, double Bx_phase, double By_mod, double By_phase, double Bz_mod, double Bz_phase, double B0, Grid * grid);
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
    void PoissonImage(double *image, double *vector);
    /**
     * @brief Apply the Maxwell image operator used by the linear solver.
     *
     * @param im Output image vector.
     * @param vector Input Krylov vector.
     */
    void MaxwellImage(double *im, double *vector);
    /**
     * @brief Apply the local Maxwell image operator used by the preconditioner.
     *
     * @param im Output image vector.
     * @param vector Input Krylov vector.
     */
    void MaxwellImageLocal(double *im, double *vector);
    /**
     * @brief Build the Maxwell right-hand-side source term.
     *
     * @param bkrylov Output right-hand-side vector.
     */
    void MaxwellSource(double *bkrylov);
    /**
     * @brief Impose a constant charge inside the 3D planet sphere.
     *
     * @param R Planet radius.
     * @param x_center Planet-center x coordinate.
     * @param y_center Planet-center y coordinate.
     * @param z_center Planet-center z coordinate.
     */
    void ConstantChargePlanet(double R, double x_center, double y_center, double z_center);
    /**
     * @brief Impose a constant charge inside the 2D XZ-plane planet mask.
     *
     * @param R Planet radius.
     * @param x_center Planet-center x coordinate.
     * @param z_center Planet-center z coordinate.
     */
    void ConstantChargePlanet2DPlaneXZ(double R, double x_center, double z_center);
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
      const_arr3_double vectX, const_arr3_double vectY, const_arr3_double vectZ, int ns);
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
      const_arr3_double vectX, const_arr3_double vectY, const_arr3_double vectZ);
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

    /** @brief Populate the legacy nodal field buffer used by the particle mover. */
    void set_fieldForPcls();

    /**
     * @brief Pack field data into the cell-centered GPU mover buffer.
     *
     * The packed layout stores the four XY-plane corner nodes needed by the GPU
     * mover for each cell slab in Z.
     *
     * @param fieldForPclsOnCenter Output packed field buffer.
     */
    void set_fieldForPclsToCenter(cudaFieldType *fieldForPclsOnCenter);

    /**
     * @brief Communicate per-species moments before particle-to-grid reductions.
     *
     * @param ns Number of particle species.
     */
    void communicateGhostP2G(int ns);

    /**
     * @brief Adjust densities on non-periodic boundaries.
     *
     * @param is Species index.
     */
    void adjustNonPeriodicDensities(int is);


    /*! Perfect conductor boundary conditions LEFT wall */
    void perfectConductorLeft(arr3_double imageX, arr3_double imageY, arr3_double imageZ,
      const_arr3_double vectorX, const_arr3_double vectorY, const_arr3_double vectorZ,
      int dir);
    /*! Perfect conductor boundary conditions RIGHT wall */
    void perfectConductorRight(
      arr3_double imageX, arr3_double imageY, arr3_double imageZ,
      const_arr3_double vectorX,
      const_arr3_double vectorY,
      const_arr3_double vectorZ,
      int dir);
    /*! Perfect conductor boundary conditions for source LEFT wall */
    void perfectConductorLeftS(arr3_double vectorX, arr3_double vectorY, arr3_double vectorZ, int dir);
    /*! Perfect conductor boundary conditions for source RIGHT wall */
    void perfectConductorRightS(arr3_double vectorX, arr3_double vectorY, arr3_double vectorZ, int dir);

    /*! Calculate the sysceptibility tensor on the boundary */
    void sustensorRightX(double **susxx, double **susyx, double **suszx);
    void sustensorLeftX (double **susxx, double **susyx, double **suszx);
    void sustensorRightY(double **susxy, double **susyy, double **suszy);
    void sustensorLeftY (double **susxy, double **susyy, double **suszy);
    void sustensorRightZ(double **susxz, double **susyz, double **suszz);
    void sustensorLeftZ (double **susxz, double **susyz, double **suszz);

    /*** accessor methods ***/

    /*! get Potential array */
    arr3_double getPHI() {return PHI;}

    // field components defined on nodes
    //
    double getEx(int X, int Y, int Z) const { return Ex.get(X,Y,Z);}
    double getEy(int X, int Y, int Z) const { return Ey.get(X,Y,Z);}
    double getEz(int X, int Y, int Z) const { return Ez.get(X,Y,Z);}
    double getBx(int X, int Y, int Z) const { return Bxn.get(X,Y,Z);}
    double getBy(int X, int Y, int Z) const { return Byn.get(X,Y,Z);}
    double getBz(int X, int Y, int Z) const { return Bzn.get(X,Y,Z);}
    //
    const_arr4_pfloat get_fieldForPcls() { return fieldForPcls; }
    arr3_double getEx() { return Ex; }
    arr3_double getEy() { return Ey; }
    arr3_double getEz() { return Ez; }
    arr3_double getBx() { return Bxn; }
    arr3_double getBy() { return Byn; }
    arr3_double getBz() { return Bzn; }

    
    //for parallel vtk
    arr3_double getBxc(){return Bxc;};
    arr3_double getByc(){return Byc;};
    arr3_double getBzc(){return Bzc;};


    arr3_double getRHOc() { return rhoc; }
    arr3_double getRHOn() { return rhon; }
    double getRHOc(int X, int Y, int Z) const { return rhoc.get(X,Y,Z);}
    double getRHOn(int X, int Y, int Z) const { return rhon.get(X,Y,Z);}

    // densities per species:
    //
    double getRHOcs(int X,int Y,int Z,int is)const{return rhocs.get(is,X,Y,Z);}
    double getRHOns(int X,int Y,int Z,int is)const{return rhons.get(is,X,Y,Z);}
    arr4_double getRHOns(){return rhons;}
    arr4_double getRHOcs(){return rhocs;}
    // per-species 3D slice accessors for parallel HDF5 output
    arr3_double getRHOcs(int is){return arr3_double(rhocs.fetch_arr4()[is], nxc, nyc, nzc);}
    arr3_double getRHOns(int is){return arr3_double(rhons.fetch_arr4()[is], nxn, nyn, nzn);}


    double getBx_ext(int X, int Y, int Z) const{return Bx_ext.get(X,Y,Z);}
    double getBy_ext(int X, int Y, int Z) const{return By_ext.get(X,Y,Z);}
    double getBz_ext(int X, int Y, int Z) const{return Bz_ext.get(X,Y,Z);}
    
    arr3_double getBx_ext() { return Bx_ext; }
    arr3_double getBy_ext() { return By_ext; }
    arr3_double getBz_ext() { return Bz_ext; }


    //B_tot = B + B_ext
    arr3_double getBxTot() { addscale(1.0,Bxn,Bx_ext,Bx_tot,nxn,nyn,nzn); return Bx_tot; }
    arr3_double getByTot() { addscale(1.0,Byn,By_ext,By_tot,nxn,nyn,nzn); return By_tot; }
    arr3_double getBzTot() { addscale(1.0,Bzn,Bz_ext,Bz_tot,nxn,nyn,nzn); return Bz_tot; }
    double getBxTot(int X, int Y, int Z) const{return Bxn.get(X,Y,Z)+Bx_ext.get(X,Y,Z);;}
    double getByTot(int X, int Y, int Z) const{return Byn.get(X,Y,Z)+By_ext.get(X,Y,Z);}
    double getBzTot(int X, int Y, int Z) const{return Bzn.get(X,Y,Z)+Bz_ext.get(X,Y,Z);}

    arr4_double getpXXsn() { return pXXsn; }
    double getpXXsn(int X,int Y,int Z,int is)const{return pXXsn.get(is,X,Y,Z);}

    arr4_double getpXYsn() { return pXYsn; }
    double getpXYsn(int X,int Y,int Z,int is)const{return pXYsn.get(is,X,Y,Z);}

    arr4_double getpXZsn() { return pXZsn; }
    double getpXZsn(int X,int Y,int Z,int is)const{return pXZsn.get(is,X,Y,Z);}

    arr4_double getpYYsn() { return pYYsn; }
    double getpYYsn(int X,int Y,int Z,int is)const{return pYYsn.get(is,X,Y,Z);}

    arr4_double getpYZsn() { return pYZsn; }
    double getpYZsn(int X,int Y,int Z,int is)const{return pYZsn.get(is,X,Y,Z);}

    arr4_double getpZZsn() { return pZZsn; }
    double getpZZsn(int X,int Y,int Z,int is)const{return pZZsn.get(is,X,Y,Z);}

    // per-species 3D slice accessors (node grid) for parallel HDF5 / H5hut output
    arr3_double getpXXsn(int is){return arr3_double(pXXsn.fetch_arr4()[is], nxn, nyn, nzn);}
    arr3_double getpXYsn(int is){return arr3_double(pXYsn.fetch_arr4()[is], nxn, nyn, nzn);}
    arr3_double getpXZsn(int is){return arr3_double(pXZsn.fetch_arr4()[is], nxn, nyn, nzn);}
    arr3_double getpYYsn(int is){return arr3_double(pYYsn.fetch_arr4()[is], nxn, nyn, nzn);}
    arr3_double getpYZsn(int is){return arr3_double(pYZsn.fetch_arr4()[is], nxn, nyn, nzn);}
    arr3_double getpZZsn(int is){return arr3_double(pZZsn.fetch_arr4()[is], nxn, nyn, nzn);}

    double getJx(int X, int Y, int Z) const { return Jx.get(X,Y,Z);}
    double getJy(int X, int Y, int Z) const { return Jy.get(X,Y,Z);}
    double getJz(int X, int Y, int Z) const { return Jz.get(X,Y,Z);}
    arr3_double getJx() { return Jx; }
    arr3_double getJy() { return Jy; }
    arr3_double getJz() { return Jz; }
    arr4_double getJxs() { return Jxs; }
    arr4_double getJys() { return Jys; }
    arr4_double getJzs() { return Jzs; }
    // per-species 3D slice accessors (node grid) for parallel HDF5 output
    arr3_double getJxs(int is){return arr3_double(Jxs.fetch_arr4()[is], nxn, nyn, nzn);}
    arr3_double getJys(int is){return arr3_double(Jys.fetch_arr4()[is], nxn, nyn, nzn);}
    arr3_double getJzs(int is){return arr3_double(Jzs.fetch_arr4()[is], nxn, nyn, nzn);}

    double getJxs(int X,int Y,int Z,int is)const{return Jxs.get(is,X,Y,Z);}
    double getJys(int X,int Y,int Z,int is)const{return Jys.get(is,X,Y,Z);}
    double getJzs(int X,int Y,int Z,int is)const{return Jzs.get(is,X,Y,Z);}

    /*! get the electric field energy */
    double getEenergy();
    /*! get the magnetic field energy */
    double getBenergy();
    /*! get bulk kinetic energy */
    double getBulkEnergy(int is);

    /*! print electromagnetic fields info */
    void print(void) const;
    
    
    //get MPI Derived Datatype
    MPI_Datatype getYZFacetype(bool isCenterFlag){return isCenterFlag ?yzFacetypeC : yzFacetypeN;}
    MPI_Datatype getXZFacetype(bool isCenterFlag){return isCenterFlag ?xzFacetypeC : xzFacetypeN;}
    MPI_Datatype getXYFacetype(bool isCenterFlag){return isCenterFlag ?xyFacetypeC : xyFacetypeN;}
    MPI_Datatype getXEdgetype(bool isCenterFlag){return  isCenterFlag ?xEdgetypeC : xEdgetypeN;}
    MPI_Datatype getYEdgetype(bool isCenterFlag){return  isCenterFlag ?yEdgetypeC : yEdgetypeN;}
    MPI_Datatype getZEdgetype(bool isCenterFlag){return  isCenterFlag ?zEdgetypeC : zEdgetypeN;}
    MPI_Datatype getXEdgetype2(bool isCenterFlag){return  isCenterFlag ?xEdgetypeC2 : xEdgetypeN2;}
    MPI_Datatype getYEdgetype2(bool isCenterFlag){return  isCenterFlag ?yEdgetypeC2 : yEdgetypeN2;}
    MPI_Datatype getZEdgetype2(bool isCenterFlag){return  isCenterFlag ?zEdgetypeC2 : zEdgetypeN2;}
    MPI_Datatype getCornertype(bool isCenterFlag){return  isCenterFlag ?cornertypeC : cornertypeN;}



    MPI_Datatype getProcview(){return  procview;}
    MPI_Datatype getXYZeType(){return xyzcomp;}
    MPI_Datatype getProcviewXYZ(){return  procviewXYZ;}
    MPI_Datatype getGhostType(){return  ghosttype;}

    void freeDataType();
    bool isLittleEndian(){return lEndFlag;};

  public: // accessors
    const Collective& get_col()const{return _col;}
    const Grid& get_grid()const{return _grid;};
    const VirtualTopology3D& get_vct()const{return _vct;}
    /* ********************************* // VARIABLES ********************************* */
    
  private:
    // access to global data
    const Collective& _col;
    const Grid& _grid;
    const VirtualTopology3D&_vct;
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
    double *qom;
    /*! Boundary electron speed */
    double ue0, ve0, we0;


    // KEEP IN MEMORY GUARD CELLS ARE INCLUDED
    /*! number of cells - X direction, including + 2 (guard cells) */
    int nxc;
    /*! number of nodes - X direction, including + 2 extra nodes for guard cells */
    int nxn;
    /*! number of cell - Y direction, including + 2 (guard cells) */
    int nyc;
    /*! number of nodes - Y direction, including + 2 extra nodes for guard cells */
    int nyn;
    /*! number of cell - Z direction, including + 2 (guard cells) */
    int nzc;
    /*! number of nodes - Z direction, including + 2 extra nodes for guard cells */
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
    /** Characteristic length */
    double L_square;

    /*! PSI: magnetic potential (indexX, indexY, indexZ), defined on central points between nodes */
    array3_double PSI;

    /*! PHI: electric potential (indexX, indexY, indexZ), defined on central points between nodes */
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
    //array3_double arr;

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
    /*! SPECIES: charge density for each species, defined on central points of the cell */
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
    array3_double   Bx_ext;
    array3_double   By_ext;
    array3_double   Bz_ext;

    array3_double   Bx_tot;
    array3_double   By_tot;
    array3_double   Bz_tot;

    // external current, defined on nodes
    array3_double   Jx_ext;
    array3_double   Jy_ext;
    array3_double   Jz_ext;

    // pressure tensor components, defined on nodes
    array4_double pXXsn;
    array4_double pXYsn;
    array4_double pXZsn;
    array4_double pYYsn;
    array4_double pYZsn;
    array4_double pZZsn;

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

    /*! Boundary condition for electric field 0 = perfect conductor 1 = magnetic mirror */
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
    double *rhoINIT;
    /*! Drift of the species */
    bool *DriftSpecies;

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
    double *xkrylovPoisson_B;
    double *bkrylovPoisson_B;

    // Persistent arrays for calculateE (avoid per-cycle allocation)
    double *xkrylovMaxwell;
    double *bkrylovMaxwell;
    double *xkrylovPoisson_E;
    double *bkrylovPoisson_E;
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


    //MPI Derived Datatype for Center Halo Exchange
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

    //MPI Derived Datatype for Node Halo Exchange
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
    
    //for VTK output
    MPI_Datatype  procviewXYZ,xyzcomp,procview,ghosttype;
    bool lEndFlag;
    
    void OpenBoundaryInflowB(arr3_double vectorX, arr3_double vectorY, arr3_double vectorZ,

      int nx, int ny, int nz);
    void OpenBoundaryInflowE(arr3_double vectorX, arr3_double vectorY, arr3_double vectorZ,
      int nx, int ny, int nz);
    void OpenBoundaryInflowEImage(arr3_double imageX, arr3_double imageY, arr3_double imageZ,
      const_arr3_double vectorX, const_arr3_double vectorY, const_arr3_double vectorZ,
      int nx, int ny, int nz);
    void OpenBoundaryInflowESource(arr3_double vectorX, arr3_double vectorY, arr3_double vectorZ,
      int nx, int ny, int nz);
};

typedef EMfields3D Field;

#endif
