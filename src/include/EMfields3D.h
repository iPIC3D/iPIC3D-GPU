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

/*!************************************************************************* EMfields3D.h - ElectroMagnetic fields definition ------------------- begin : May 2008 copyright : KU Leuven developers : Stefano Markidis, Giovanni Lapenta ************************************************************************* */

#ifndef EMfields3D_H
#define EMfields3D_H

#include "asserts.h"
#include "ipicfwd.h"
#include "Alloc.h"
#include "Basic.h"
#include "mpi.h"

#include "cudaTypeDef.cuh"

#ifdef GPU_SOLVER
#include "GPUFieldArray.cuh"
#include "GPUHaloComm.cuh"
#endif

/*! Electromagnetic fields and sources defined for each local grid, and for an implicit maxwell's solver @date May 2008 @par Copyright: (C) 2008 KUL @author Stefano Markidis, Giovanni Lapenta. @version 3.0 */

// dimension of vectors used in fieldForPcls
const int DFIELD_3or4=4; // 4 pads with garbage but is needed for alignment

class EMfields3D                // :public Field
{
  public:
    /*! constructor */
    EMfields3D(Collective * col, Grid * grid, VirtualTopology3D *vct);
    /*! destructor */
    ~EMfields3D();

    /*! initialize the electromagnetic fields with constant values */
    void init();
    /*! init beam */
    void initBEAM(double x_center, double y_center, double z_center, double radius);
    /*! initialize GEM challenge */
    void initGEM();
    void initOriginalGEM();
    void initGEMDoubleHarris();
    void initDoublePeriodicHarrisWithGaussianHumpPerturbation();
    void initHumpPerturbation();
    /*! initialize GEM challenge with dipole-like tail without perturbation */
    void initGEMDipoleLikeTailNoPert();
    /*! initialize GEM challenge with no Perturbation */
    void initGEMnoPert();
#ifdef BATSRUS
    /*! initialize from BATSRUS */
    void initBATSRUS();
#endif
    /*! Random initial field */
    void initRandomField();
    /*! Init Force Free (JxB=0) */
    void initForceFree();
    /*! initialized with rotated magnetic field */
    void initEM_rotate(double B, double theta);
    /*! add a perturbattion to charge density */
    void AddPerturbationRho(double deltaBoB, double kx, double ky, double Bx_mod, double By_mod, double Bz_mod, double ne_mod, double ne_phase, double ni_mod, double ni_phase, double B0, Grid * grid);
    /*! add a perturbattion to the EM field */
    void AddPerturbation(double deltaBoB, double kx, double ky, double Ex_mod, double Ex_phase, double Ey_mod, double Ey_phase, double Ez_mod, double Ez_phase, double Bx_mod, double Bx_phase, double By_mod, double By_phase, double Bz_mod, double Bz_phase, double B0, Grid * grid);
    /*! Initialise a combination of magnetic dipoles */
    void initDipole();
    void initDipole2D();
    /*! Initialise magnetic nulls */
    void initNullPoints();
    /*! Initialise Taylor-Green flow */
    void initTaylorGreen();
    /*! Calculate Electric field using the implicit Maxwell solver */
    void calculateE(int cycle);
    /*! Image of Poisson Solver (for SOLVER) */
    void PoissonImage(double *image, double *vector);
    /*! Image of Maxwell Solver (for Solver) */
    void MaxwellImage(double *im, double *vector);
    /*! Image of Maxwell Solver without communication (for Solver preconditioner) */
    void MaxwellImageLocal(double *im, double *vector);
    /*! Maxwell source term (for SOLVER) */
    void MaxwellSource(double *bkrylov);
    /*! Impose a constant charge inside a spherical zone of the domain */
    void ConstantChargePlanet(double R, double x_center, double y_center, double z_center);
    void ConstantChargePlanet2DPlaneXZ(double R, double x_center, double z_center);
    /*! Impose a constant charge in the OpenBC boundaries */
    void ConstantChargeOpenBC();
    /*! Impose a constant charge in the OpenBC boundaries */
    void ConstantChargeOpenBCv2();
    /*! Calculate Magnetic field with the implicit solver: calculate B defined on nodes With E(n+ theta) computed, the magnetic field is evaluated from Faraday's law */
    void calculateB(int cycle);
    /*! Apply divergence cleaning: solve laplacian(PSI) = div(B), then B = B - grad(PSI) on boundary layers */
    void applyDivBCleaning();
    /*! fix B on the boundary for gem challange */
    void fixBcGEM();
    void fixBnGEM();
    /*! fix B on the boundary for gem challange */
    void fixBforcefree();

    /*! Calculate the three components of Pi(implicit pressure) cross image vector */
    void PIdot(arr3_double PIdotX, arr3_double PIdotY, arr3_double PIdotZ,
      const_arr3_double vectX, const_arr3_double vectY, const_arr3_double vectZ, int ns);
    /*! Calculate the three components of mu (implicit permeattivity) cross image vector */
    void MUdot(arr3_double MUdotX, arr3_double MUdotY, arr3_double MUdotZ,
      const_arr3_double vectX, const_arr3_double vectY, const_arr3_double vectZ);
    /*! Calculate rho hat, Jx hat, Jy hat, Jz hat */
    void calculateHatFunctions();


    /*! communicate ghost for densities and interp rho from node to center */
    void interpDensitiesN2C();
    /*! set to 0 all the densities fields */
    void setZeroDensities();
    /*! set to 0 primary moments */
    void setZeroPrimaryMoments();
    /*! set to 0 all densities derived from primary moments */
    void setZeroDerivedMoments();
    /*! Sum rhon over species */
    void sumOverSpecies();
    /*! Sum current over different species */
    void sumOverSpeciesJ();
    /*! Smoothing after the interpolation* */
    void smooth(arr3_double vector, int type);
    /*! SPECIES: Smoothing after the interpolation for species fields* */
    void smooth(double value, arr4_double vector, int is, int type);
    /*! smooth the electric field */
    void smoothE();

    /*! copy the field data to the array used to move the particles */
    void set_fieldForPcls();

    void set_fieldForPclsToCenter(cudaFieldType *fieldForPclsOnCenter);

    /*! communicate ghost for grid -> Particles interpolation */
    void communicateGhostP2G(int ns);

    /*! adjust densities on boundaries that are not periodic */
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

#ifdef GPU_SOLVER
    // ---- GPU Solver: lifecycle ----
    /** Allocate all GPU-resident field arrays (called once after construction). */
    void gpuSolverAllocate();
    /** Free all GPU-resident field arrays. */
    void gpuSolverFree();

    /** Return the dedicated solver CUDA stream. */
    cudaStream_t gpuSolverStream() const { return solverStream_; }
    /** Synchronise the solver stream (block host until all solver work is done). */
    void gpuSolverStreamSync() { cudaStreamSynchronize(solverStream_); }

    // ---- GPU Solver: host ↔ device synchronisation ----
    /** Copy all primary field arrays from host to device (for initialisation / restart). */
    void gpuSolverSyncH2D(cudaStream_t stream = 0);
    /** Copy field arrays from device to host (for I/O output). */
    void gpuSolverSyncD2H(cudaStream_t stream = 0);
    // ---- GPU Solver: GPU-aware MPI halo exchange ----
    /** Node-based halo exchange using GPU-aware MPI (operates on device pointer). */
    void gpuCommunicateNodeBC(int nx, int ny, int nz, GPUFieldArray3& gpuArr,
                              int bcFaceXright, int bcFaceXleft,
                              int bcFaceYright, int bcFaceYleft,
                              int bcFaceZright, int bcFaceZleft);
    /** Batched 3-field node-based halo exchange + BC in one MPI round. */
    void gpuCommunicateNodeBC_3(int nx, int ny, int nz,
                                GPUFieldArray3& a1, GPUFieldArray3& a2, GPUFieldArray3& a3,
                                int bcXR, int bcXL, int bcYR, int bcYL, int bcZR, int bcZL);
    /** Center-based halo exchange using GPU-aware MPI (operates on device pointer). */
    void gpuCommunicateCenterBC(int nx, int ny, int nz, GPUFieldArray3& gpuArr,
                                int bcFaceXright, int bcFaceXleft,
                                int bcFaceYright, int bcFaceYleft,
                                int bcFaceZright, int bcFaceZleft);
    /** Batched 3-field center-based halo exchange + BC in one MPI round. */
    void gpuCommunicateCenterBC_3(int nx, int ny, int nz,
                                  GPUFieldArray3& a1, GPUFieldArray3& a2, GPUFieldArray3& a3,
                                  int bcXR, int bcXL, int bcYR, int bcYL, int bcZR, int bcZL);
    /** Batched 9-field center-based halo exchange + BC in one MPI round (for fused triple Laplacian). */
    void gpuCommunicateCenterBC_9(int nx, int ny, int nz,
                                  GPUFieldArray3& a1, GPUFieldArray3& a2, GPUFieldArray3& a3,
                                  GPUFieldArray3& a4, GPUFieldArray3& a5, GPUFieldArray3& a6,
                                  GPUFieldArray3& a7, GPUFieldArray3& a8, GPUFieldArray3& a9,
                                  int bcXR, int bcXL, int bcYR, int bcYL, int bcZR, int bcZL);
    /** Batched 3-field center-based halo exchange + BC_P in one MPI round. */
    void gpuCommunicateCenterBC_P_3(int nx, int ny, int nz,
                                    GPUFieldArray3& a1, GPUFieldArray3& a2, GPUFieldArray3& a3,
                                    int bcXR, int bcXL, int bcYR, int bcYL, int bcZR, int bcZL);
    /** Batched 3-field node box-stencil halo exchange + BC in one MPI round. */
    void gpuCommunicateNodeBoxStencilBC_3(int nx, int ny, int nz,
                                          GPUFieldArray3& a1, GPUFieldArray3& a2, GPUFieldArray3& a3,
                                          int bcXR, int bcXL, int bcYR, int bcYL, int bcZR, int bcZL);
    /** Batched 3-field center-based halo exchange with per-field BCs (bc is int[6]). */
    void gpuCommunicateCenterBC_3mixed(int nx, int ny, int nz,
                                       GPUFieldArray3& a1, const int* bc1,
                                       GPUFieldArray3& a2, const int* bc2,
                                       GPUFieldArray3& a3, const int* bc3);
    /** Batched 3-field node-based halo exchange with per-field BCs (bc is int[6]). */
    void gpuCommunicateNodeBC_3mixed(int nx, int ny, int nz,
                                     GPUFieldArray3& a1, const int* bc1,
                                     GPUFieldArray3& a2, const int* bc2,
                                     GPUFieldArray3& a3, const int* bc3);
    /** Batched 3-field node box-stencil halo exchange with per-field BCs. */
    void gpuCommunicateNodeBoxStencilBC_3mixed(int nx, int ny, int nz,
                                               GPUFieldArray3& a1, const int* bc1,
                                               GPUFieldArray3& a2, const int* bc2,
                                               GPUFieldArray3& a3, const int* bc3);
    /** Node-based box-stencil (face-only) halo exchange for smoothing. */
    void gpuCommunicateNodeBoxStencilBC(int nx, int ny, int nz, GPUFieldArray3& gpuArr,
                                        int bcFaceXright, int bcFaceXleft,
                                        int bcFaceYright, int bcFaceYleft,
                                        int bcFaceZright, int bcFaceZleft);
    /** Particle-communicator centre halo exchange (for moments). */
    void gpuCommunicateCenterBC_P(int nx, int ny, int nz, GPUFieldArray3& gpuArr,
                                  int bcFaceXright, int bcFaceXleft,
                                  int bcFaceYright, int bcFaceYleft,
                                  int bcFaceZright, int bcFaceZleft);
    /** Additive (interpolating) node halo exchange for moments (ghost → shared nodes). */
    void gpuCommunicateInterp(int nx, int ny, int nz, GPUFieldArray3& gpuArr);
    /** Copy-style node halo exchange for moments (shared → ghost nodes). */
    void gpuCommunicateNode_P(int nx, int ny, int nz, GPUFieldArray3& gpuArr);
    /** Node-based box-stencil halo exchange using particle communicator (for smooth). */
    void gpuCommunicateNodeBoxStencilBC_P(int nx, int ny, int nz, GPUFieldArray3& gpuArr,
                                          int bcFaceXright, int bcFaceXleft,
                                          int bcFaceYright, int bcFaceYleft,
                                          int bcFaceZright, int bcFaceZleft);
    /** Batched 3-field node-based box-stencil halo exchange using particle communicator. */
    void gpuCommunicateNodeBoxStencilBC_P_3(int nx, int ny, int nz,
                                            GPUFieldArray3& a1, GPUFieldArray3& a2, GPUFieldArray3& a3,
                                            int bcXR, int bcXL, int bcYR, int bcYL, int bcZR, int bcZL);
    /** Center-based box-stencil halo exchange using particle communicator (for smooth). */
    void gpuCommunicateCenterBoxStencilBC_P(int nx, int ny, int nz, GPUFieldArray3& gpuArr,
                                            int bcFaceXright, int bcFaceXleft,
                                            int bcFaceYright, int bcFaceYleft,
                                            int bcFaceZright, int bcFaceZleft);
    /** Batched 3-field center-based box-stencil halo exchange using particle communicator. */
    void gpuCommunicateCenterBoxStencilBC_P_3(int nx, int ny, int nz,
                                              GPUFieldArray3& a1, GPUFieldArray3& a2, GPUFieldArray3& a3,
                                              int bcXR, int bcXL, int bcYR, int bcYL, int bcZR, int bcZL);

    // ---- GPU Solver: field-solver methods ----
    /** GPU version of calculateE. Falls back to CPU if GPU_SOLVER off. */
    void gpuCalculateE(int cycle);
    /** GPU version of calculateB. */
    void gpuCalculateB(int cycle);
    /** GPU version of calculateHatFunctions. */
    void gpuCalculateHatFunctions();
    /** GPU MaxwellImage: A*x callback for GMRES (operates on device Krylov vectors). */
    void gpuMaxwellImage(double* d_im, double* d_vector);
    /** GPU MaxwellSource: build RHS of Maxwell system (result in device Krylov vector). */
    void gpuMaxwellSource(double* d_bkrylov);
    /** GPU MUdot: compute μ·E for all species. */
    void gpuMUdot(GPUFieldArray3& MUdotX, GPUFieldArray3& MUdotY, GPUFieldArray3& MUdotZ,
                  GPUFieldArray3& vX, GPUFieldArray3& vY, GPUFieldArray3& vZ);
    /** GPU PIdot: compute π·v for one species (accumulative). */
    void gpuPIdot(GPUFieldArray3& PIX, GPUFieldArray3& PIY, GPUFieldArray3& PIZ,
                  GPUFieldArray3& vX, GPUFieldArray3& vY, GPUFieldArray3& vZ, int is);
    /** GPU smooth: SmoothNiter iterations of box smoothing. */
    void gpuSmooth(GPUFieldArray3& arr, int type);
    /** GPU smoothE: fused 3-component E-field smoothing. */
    void gpuSmoothE();
    /** GPU smooth3: fused 3-component node-field smoothing (for Jhat). */
    void gpuSmooth3(GPUFieldArray3& a1, GPUFieldArray3& a2, GPUFieldArray3& a3, int type);
    /** GPU perfect conductor BC: left boundary. */
    void gpuPerfectConductorLeft(GPUFieldArray3& imX, GPUFieldArray3& imY, GPUFieldArray3& imZ,
                                 GPUFieldArray3& vX, GPUFieldArray3& vY, GPUFieldArray3& vZ, int dir);
    /** GPU perfect conductor BC: right boundary. */
    void gpuPerfectConductorRight(GPUFieldArray3& imX, GPUFieldArray3& imY, GPUFieldArray3& imZ,
                                  GPUFieldArray3& vX, GPUFieldArray3& vY, GPUFieldArray3& vZ, int dir);
    /** GPU lapN2N: Laplacian node→node (gradN2C + halo comm + divC2N). */
    void gpuLapN2N(GPUFieldArray3& lapN, GPUFieldArray3& fieldN);
    /** Fused triple Laplacian: 3× lapN2N with ONE batched halo exchange. */
    void gpuLapN2N_3(GPUFieldArray3& lapA, GPUFieldArray3& fieldA,
                     GPUFieldArray3& lapB, GPUFieldArray3& fieldB,
                     GPUFieldArray3& lapC, GPUFieldArray3& fieldC);

    // ---- GPU Solver: moment post-processing on GPU ----
    /** D2D scatter: copy 10 packed moment arrays from the moment-kernel buffer
     *  (momentsSrc, layout [10][gridSize]) into the per-species slices of
     *  d_rhons, d_Jxs, d_Jys, d_Jzs, d_pXXsn, d_pXYsn, d_pXZsn,
     *  d_pYYsn, d_pYZsn, d_pZZsn.  Pure device-to-device, no host touch. */
    void gpuScatterMomentsD2D(double* momentsSrc, int species, cudaStream_t stream = 0);

    /** GPU communicateGhostP2G: full P2G ghost exchange for one species on GPU.
     *  Performs additive interpolation halo, adjustNonPeriodicDensities,
     *  and copy-style node halo exchange — all on device arrays. */
    void gpuCommunicateGhostP2G(int species);
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
    void gpuBatchedHaloExchange(double** h_fieldPtrs, int nFields,
                                int nx, int ny, int nz,
                                bool isCenterFlag, bool isFaceOnlyFlag,
                                bool needInterp, bool isParticle,
                                cudaStream_t stream);

    /** Allocate persistent GPU halo exchange buffers. */
    void gpuAllocateHaloBuffers();
    /** Free persistent GPU halo exchange buffers. */
    void gpuFreeHaloBuffers();

    /** GPU setZeroDerivedMoments: zero Jx/Jy/Jz, Jxh/Jyh/Jzh, rhon, rhoc, rhoh on device. */
    void gpuSetZeroDerivedMoments();

    /** GPU sumOverSpecies: rhon += sum_s(rhons_s) on device. */
    void gpuSumOverSpecies();

    /** GPU interpDensitiesN2C: rhoc = interpN2C(rhon) on device. */
    void gpuInterpDensitiesN2C();

    // ---- GPU Solver: open boundary conditions ----
    /** GPU open BC: zero source term on inflow faces. */
    void gpuOpenBoundaryInflowESource(double* dX, double* dY, double* dZ, int nx, int ny, int nz);
    /** GPU open BC: set image = vect - E_inj on inflow faces. */
    void gpuOpenBoundaryInflowEImage(double* imX, double* imY, double* imZ,
                                     const double* vX, const double* vY, const double* vZ,
                                     int nx, int ny, int nz);
    /** GPU open BC: SAL blend / Dirichlet on E inflow faces (post-solve). */
    void gpuOpenBoundaryInflowE(double* dX, double* dY, double* dZ, int nx, int ny, int nz);
    /** GPU open BC: SAL blend / extrapolation on center B inflow faces. */
    void gpuOpenBoundaryInflowB(double* dX, double* dY, double* dZ, int nx, int ny, int nz);

    // ---- GPU Solver: case-specific B fixes ----
    /** GPU fix center B for GEM case. */
    void gpuFixBcGEM();
    /** GPU fix node B for GEM case. */
    void gpuFixBnGEM();
    /** GPU fix center B for ForceFree case. */
    void gpuFixBforcefree();

    // ---- GPU Solver: ConstantChargePlanet ----
    /** GPU ConstantChargePlanet: set rhons inside sphere. */
    void gpuConstantChargePlanet(double R, double x_center, double y_center, double z_center);
    /** GPU ConstantChargePlanet 2D: set rhons inside circle in XZ plane. */
    void gpuConstantChargePlanet2DPlaneXZ(double R, double x_center, double z_center);

    // ---- GPU Solver: Poisson/divB correction ----
    /** GPU Poisson image operator (laplacian on centers). */
    void gpuPoissonImage(double* d_im, double* d_vec);
    /** GPU Poisson correction for div(E) cleaning. */
    void gpuPoissonCorrection(int cycle);
    /** GPU div(B) cleaning: solve lap(PSI)=div(B), correct B on boundary layers. */
    void gpuApplyDivBCleaning();

    // ---- GPU Solver: additional halo exchange ----
    /** Center box-stencil halo exchange using field communicator. */
    void gpuCommunicateCenterBoxStencilBC(int nx, int ny, int nz, GPUFieldArray3& gpuArr,
                                          int bcFaceXright, int bcFaceXleft,
                                          int bcFaceYright, int bcFaceYleft,
                                          int bcFaceZright, int bcFaceZleft);

    // ---- GPU Solver: accessor helpers ----
    GPUFieldArray3& gpuEx()  { return d_Ex;  }
    GPUFieldArray3& gpuEy()  { return d_Ey;  }
    GPUFieldArray3& gpuEz()  { return d_Ez;  }
    GPUFieldArray3& gpuExth(){ return d_Exth;}
    GPUFieldArray3& gpuEyth(){ return d_Eyth;}
    GPUFieldArray3& gpuEzth(){ return d_Ezth;}
    GPUFieldArray3& gpuBxn() { return d_Bxn; }
    GPUFieldArray3& gpuByn() { return d_Byn; }
    GPUFieldArray3& gpuBzn() { return d_Bzn; }
    GPUFieldArray3& gpuBxc() { return d_Bxc; }
    GPUFieldArray3& gpuByc() { return d_Byc; }
    GPUFieldArray3& gpuBzc() { return d_Bzc; }
    GPUFieldArray3& gpuBx_ext() { return d_Bx_ext; }
    GPUFieldArray3& gpuBy_ext() { return d_By_ext; }
    GPUFieldArray3& gpuBz_ext() { return d_Bz_ext; }
    GPUFieldArray4& gpuRhons()  { return d_rhons;   }
    GPUFieldArray4& gpuJxs()    { return d_Jxs;     }
    GPUFieldArray4& gpuJys()    { return d_Jys;     }
    GPUFieldArray4& gpuJzs()    { return d_Jzs;     }
    GPUFieldArray4& gpupXXsn()  { return d_pXXsn;   }
    GPUFieldArray4& gpupXYsn()  { return d_pXYsn;   }
    GPUFieldArray4& gpupXZsn()  { return d_pXZsn;   }
    GPUFieldArray4& gpupYYsn()  { return d_pYYsn;   }
    GPUFieldArray4& gpupYZsn()  { return d_pYZsn;   }
    GPUFieldArray4& gpupZZsn()  { return d_pZZsn;   }
    bool isGpuSolverAllocated() const { return gpuSolverAllocated_; }
#endif // GPU_SOLVER

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
    GPUFieldArray3 d_tempX,  d_tempY,  d_tempZ;
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
    double* d_qom = nullptr;

    // Reduction scratch buffer for GPU BLAS dot/norm operations
    double* d_blasScratch = nullptr;

    // GPU GMRES workspace (allocated on first use in gpuCalculateE)
    double* d_gmresV    = nullptr;  // [m+1][xkrylovlen]
    double* d_gmresW    = nullptr;  // [xkrylovlen]
    int     gmresVAlloc = 0;

    // ---- Persistent PINNED host buffers for GMRES reductions ----
    // Avoids per-call heap allocation and enables true async D→H DMA.
    static constexpr int GMRES_M = 20;
    static constexpr int GMRES_MP1 = GMRES_M + 1;
    double* h_gmresReduceLocal  = nullptr;  // pinned, [GMRES_MP1] for Arnoldi reductions
    double* h_gmresReduceGlobal = nullptr;  // pinned, [GMRES_MP1] for MPI_Allreduce output
    double* h_gmresH  = nullptr;  // pinned, [GMRES_MP1 * GMRES_M] Hessenberg matrix
    double* h_gmresG  = nullptr;  // pinned, [GMRES_MP1] residual vector
    double* h_gmresCS = nullptr;  // pinned, [GMRES_M] Givens cosines
    double* h_gmresSN = nullptr;  // pinned, [GMRES_M] Givens sines
    double* h_gmresY  = nullptr;  // pinned, [GMRES_MP1] back-substitution work

    // Dedicated non-blocking CUDA stream for the GPU field solver.
    // Avoids implicit serialisation with particle streams via the legacy default stream.
    cudaStream_t solverStream_ = 0;

    // ---- Persistent batched halo-exchange buffers ----
    // 6 directions: 0=XL 1=XR 2=YL 3=YR 4=ZL 5=ZR
    static constexpr int HALO_MAX_BATCH = 64;
    double* d_haloBuf_send_[6] = {};   // contiguous GPU send buffers
    double* d_haloBuf_recv_[6] = {};   // contiguous GPU recv buffers
    double** d_ptrArray_       = nullptr; // device array of field pointers for kernels
    double** h_ptrArray_       = nullptr; // pinned host staging for d_ptrArray_ H→D copies
    bool    haloBufsAllocated_ = false;

    // Flag tracking whether GPU solver arrays have been allocated
    bool gpuSolverAllocated_ = false;
#endif // GPU_SOLVER
};

typedef EMfields3D Field;

#endif
