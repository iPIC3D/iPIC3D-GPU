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

/*******************************************************************************************
  Particles3D.h  -  Class for particles of the same species, in a 2D space and 3 component velocity
  -------------------
developers: Stefano Markidis, Enrico Camporeale, Giovanni Lapenta, David Burgess
 ********************************************************************************************/

#ifndef Part2D_H
#define Part2D_H

#include "Particles3Dcomm.h"
//#include "TimeTasks.h"

/**
 * 
 * Class for particles of the same species, in a 2D space and 3 component velocity
 * 
 * @date Fri Jun 4 2007
 * @author Stefano Markidis, Giovanni Lapenta
 * @version 2.0
 *
 */
class Particles3D:public Particles3Dcomm {
  friend class moverParameter;
  public:
    /** constructor */
    //Particles3D();
    Particles3D(int species, CollectiveIO *col, VirtualTopology3D *vct, Grid * grid):
      Particles3Dcomm(species, col, vct, grid)
    {}
    /** destructor */
    ~Particles3D(){}
    /** Initial condition: uniform in space and maxwellian in velocity */
    void maxwellian(Field * EMf);
    /** Initial condition: uniform in space and maxwellian in velocity with velocity from Null Point currents */
    void maxwellianNullPoints(Field * EMf);
    /** Maxellian velocity from currents and uniform spatial distribution */
    void maxwellianDoubleHarris(Field * EMf);
    /** Maxellian velocity from currents and uniform spatial distribution */
    void maxwellianHumpPerturbation(Field * EMf);
    /** pitch_angle_energy initialization (Assume B on z only) for test particles */
    void pitch_angle_energy(Field * EMf);
    /** Force Free initialization (JxB=0) for particles */
    void force_free(Field * EMf);
   private:
    /** repopulate particles in a single cell */
    void populate_cell_with_particles(int i, int j, int k, double q,
      double dx_per_pcl, double dy_per_pcl, double dz_per_pcl);
   public:
    /** repopulate particles in boundary layer */
    void repopulate_particles();
    void repopulate_particlesInfo(bool* doRepopulateInjection, bool* doRepopulateInjectionSide, cudaCommonType* repopulateBoundary);
    void repopulate_particles_onlyInjection();
    /**Particles Open Boundary */
    void openbc_particles_outflow();
    void openbc_particles_outflowInfo(bool* doOpenBC, bool* applyOpenBC, cudaCommonType* delBdry, cudaCommonType* openBdry);

#ifdef BATSRUS
    /*! Initial condition: given a fluid model (BATSRUS) */
    void MaxwellianFromFluid(Field* EMf,Collective *col, int is);
    /*! Initiate dist. func. for a single cell form a fluid model (BATSRUS) */
    void MaxwellianFromFluidCell(Collective *col, int is, int i, int j, int k, int &ip, double *x, double *y, double *z, double *q, double *vx, double *vy, double *vz, longid* ParticleID);
#endif

};

#endif
