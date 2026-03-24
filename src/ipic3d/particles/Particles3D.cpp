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
  Particles3D.cpp  -  Class for particles of the same species, in a 3D space and 3component velocity
  -------------------
developers: Stefano Markidis, Giovanni Lapenta
 ********************************************************************************************/


#include <mpi.h>
#include <iostream>
#include <math.h>
#include <limits.h>
#include "asserts.h"
#include "VCtopology3D.h"
#include "Collective.h"
#include "Basic.h"
#include "Grid3DCU.h"
#include "Field.h"
#include "MPIdata.h"
#include "ipicdefs.h"
#include "TimeTasks.h"
#include "parallel.h"
#include "Particles3D.h"

#include "mic_particles.h"
#include "debug.h"
#include <complex>

using std::cout;
using std::cerr;
using std::endl;

#define min(a,b) (((a)<(b))?(a):(b));
#define max(a,b) (((a)>(b))?(a):(b));
#define MIN_VAL   1E-16
// particles processed together
#define P_SAME_TIME 2

// if true then mover cannot move particle 
// more than one processor subdomain
//
static bool cap_velocity(){return false;}

/**
 * 
 * Class for particles of the same species
 * @date Fri Jun 4 2009
 * @author Stefano Markidis, Giovanni Lapenta
 * @version 2.0
 *
 */

#ifdef BATSRUS
/** Maxellian random velocity and uniform spatial distribution */
void Particles3D::MaxwellianFromFluid(Field* EMf,Collective *col, int is){

  /*
   * Constuctiong the distrebution function from a Fluid model
   */

  // loop over grid cells and set position, velociy and charge of all particles indexed by counter
  // there are multiple (27 or so) particles per grid cell.
  int i,j,k,counter=0;
  for (i=1; i< grid->getNXC()-1;i++)
    for (j=1; j< grid->getNYC()-1;j++)
      for (k=1; k< grid->getNZC()-1;k++)
        MaxwellianFromFluidCell(col,is, i,j,k,counter,x,y,z,q,u,v,w,ParticleID);
}

void Particles3D::MaxwellianFromFluidCell(Collective *col, int is, int i, int j, int k, int &ip, double *x, double *y, double *z, double *q, double *vx, double *vy, double *vz, longid* ParticleID)
{
  /*
   * grid           : local grid object (in)
   * col            : collective (global) object (in)
   * is             : species index (in)
   * i,j,k          : grid cell index on proc (in)
   * ip             : particle number counter (inout)
   * x,y,z          : particle position (out)
   * q              : particle charge (out)
   * vx,vy,vz       : particle velocity (out)
   * ParticleID     : particle tracking ID (out)
   */

  // loop over particles inside grid cell i,j,k
  for (int ii=0; ii < npcelx; ii++)
    for (int jj=0; jj < npcely; jj++)
      for (int kk=0; kk < npcelz; kk++){
        // Assign particle positions: uniformly spaced. x_cellnode + dx_particle*(0.5+index_particle)
        fetchX(ip) = (ii + .5)*(dx/npcelx) + grid->getXN(i,j,k);
        fetchY(ip) = (jj + .5)*(dy/npcely) + grid->getYN(i,j,k);
        fetchZ(ip) = (kk + .5)*(dz/npcelz) + grid->getZN(i,j,k);
        // q = charge
        fetchQ(ip) =  (qom/fabs(qom))*(col->getFluidRhoCenter(i,j,k,is)/npcel)*(1.0/grid->getInvVOL());
        // u = X velocity
        sample_maxwellian(
          fetchU(ip),fetchV(ip),fetchW(ip),
          col->getFluidUthx(i,j,k,is),
          col->getFluidVthx(i,j,k,is),
          col->getFluidWthx(i,j,k,is),
          col->getFluidUx(i,j,k,is),
          col->getFluidVx(i,j,k,is),
          col->getFluidWx(i,j,k,is));
        ip++ ;
      }
}
#endif

/** Maxellian random velocity and uniform spatial distribution */
void Particles3D::maxwellian(Field * EMf)
{
  /* initialize random generator with different seed on different processor */
  srand(vct->getCartesian_rank() + 2);

  assert_eq(getNOP(),0);

  const double q_sgn = (qom / fabs(qom));
  // multipled by charge density gives charge per particle
  const double q_factor =  q_sgn * grid->getVOL() / npcel;

  for (int i = 1; i < grid->getNXC() - 1; i++)
  {
  for (int j = 1; j < grid->getNYC() - 1; j++)
  for (int k = 1; k < grid->getNZC() - 1; k++)
  {
    const double q = q_factor * EMf->getRHOcs(i, j, k, ns);
    for (int ii = 0; ii < npcelx; ii++)
    for (int jj = 0; jj < npcely; jj++)
    for (int kk = 0; kk < npcelz; kk++)
    {
      double u,v,w;
      sample_maxwellian(
        u,v,w,
        uth, vth, wth,
        u0, v0, w0);
      // could also sample positions randomly as in repopulate_particles();
      const double x = (ii + .5) * (dx / npcelx) + grid->getXN(i, j, k);
      const double y = (jj + .5) * (dy / npcely) + grid->getYN(i, j, k);
      const double z = (kk + .5) * (dz / npcelz) + grid->getZN(i, j, k);
      create_new_particle(u,v,w,q,x,y,z);
    }
  }
  }
}

/** Maxellian velocity from currents and uniform spatial distribution */
void Particles3D::maxwellianNullPoints(Field * EMf)
{
	/* initialize random generator with different seed on different processor */
	srand(vct->getCartesian_rank()+2);

	const double q_sgn = (qom / fabs(qom));
	const double q_factor =  q_sgn * grid->getVOL() / npcel;

	for (int i=1; i< grid->getNXC()-1;i++)
	for (int j=1; j< grid->getNYC()-1;j++)
	for (int k=1; k< grid->getNZC()-1;k++){
		const double q = q_factor * EMf->getRHOcs(i, j, k, ns);

		// determine the drift velocity from current X
		u0 = EMf->getJxs(i,j,k,ns)/EMf->getRHOns(i,j,k,ns);
		if (u0 > c){
			cout << "DRIFT VELOCITY x > c : B init field too high!" << endl;
			MPI_Abort(MPI_COMM_WORLD,2);
		}
		// determine the drift velocity from current Y
		v0 = EMf->getJys(i,j,k,ns)/EMf->getRHOns(i,j,k,ns);
		if (v0 > c){
			cout << "DRIFT VELOCITY y > c : B init field too high!" << endl;
			MPI_Abort(MPI_COMM_WORLD,2);
		}
		// determine the drift velocity from current Z
		w0 = EMf->getJzs(i,j,k,ns)/EMf->getRHOns(i,j,k,ns);
		if (w0 > c){
			cout << "DRIFT VELOCITY z > c : B init field too high!" << endl;
			MPI_Abort(MPI_COMM_WORLD,2);
		}
		for (int ii=0; ii < npcelx; ii++)
		for (int jj=0; jj < npcely; jj++)
		for (int kk=0; kk < npcelz; kk++){
			double u,v,w;
			sample_maxwellian(u, v, w, uth, vth, wth, u0, v0, w0);

			const double x = (ii + .5)*(dx/npcelx) + grid->getXN(i,j,k);
			const double y = (jj + .5)*(dy/npcely) + grid->getYN(i,j,k);
			const double z = (kk + .5)*(dz/npcelz) + grid->getZN(i,j,k);

			create_new_particle(u,v,w,q,x,y,z);
		}
	}
}


/** Maxellian random velocity and uniform spatial distribution - invert w0 for the upper current sheet */
void Particles3D::maxwellianDoubleHarris(Field * EMf)
{
  /* initialize random generator with different seed on different processor */
  srand(vct->getCartesian_rank() + 2);

  assert_eq(getNOP(),0);

  const double q_sgn = (qom / fabs(qom));
  const double Ly_upper = Ly/2.0;
  // multipled by charge density gives charge per particle
  const double q_factor =  q_sgn * grid->getVOL() / npcel;

  for (int i = 1; i < grid->getNXC() - 1; i++)
  {
  for (int j = 1; j < grid->getNYC() - 1; j++)
  for (int k = 1; k < grid->getNZC() - 1; k++)
  {
    const double q = q_factor * EMf->getRHOcs(i, j, k, ns);
    for (int ii = 0; ii < npcelx; ii++)
    for (int jj = 0; jj < npcely; jj++)
    for (int kk = 0; kk < npcelz; kk++)
    {

      // could also sample positions randomly as in repopulate_particles();
      const double x = (ii + .5) * (dx / npcelx) + grid->getXN(i, j, k);
      const double y = (jj + .5) * (dy / npcely) + grid->getYN(i, j, k);
      const double z = (kk + .5) * (dz / npcelz) + grid->getZN(i, j, k);

      double u,v,w;
      if(y> Ly_upper)  sample_maxwellian(u,v,w,uth, vth, wth,u0, v0, w0);//-1.0*w0
      else  sample_maxwellian(u,v,w,uth, vth, wth,u0, v0, w0);

      create_new_particle(u,v,w,q,x,y,z);
    }
  }
  }
}

/** Maxellian random velocity and uniform spatial distribution */
void Particles3D::maxwellianHumpPerturbation(Field * EMf)
{
  /* initialize random generator with different seed on different processor */
  srand(vct->getCartesian_rank() + 2);

  assert_eq(getNOP(),0);

  const double q_sgn = (qom / fabs(qom));
  // multipled by charge density gives charge per particle
  const double q_factor =  q_sgn * grid->getVOL() / npcel;

  for (int i = 1; i < grid->getNXC() - 1; i++)
  {
  for (int j = 1; j < grid->getNYC() - 1; j++)
  for (int k = 1; k < grid->getNZC() - 1; k++)
  {
    const double q = q_factor * EMf->getRHOcs(i, j, k, ns);
    for (int ii = 0; ii < npcelx; ii++)
    for (int jj = 0; jj < npcely; jj++)
    for (int kk = 0; kk < npcelz; kk++)
    {

      // could also sample positions randomly as in repopulate_particles();
      const double x = (ii + .5) * (dx / npcelx) + grid->getXN(i, j, k);
      const double y = (jj + .5) * (dy / npcely) + grid->getYN(i, j, k);
      const double z = (kk + .5) * (dz / npcelz) + grid->getZN(i, j, k);

      double u,v,w;
      sample_maxwellian(u,v,w,uth, vth, wth,u0, v0, w0);

      create_new_particle(u,v,w,q,x,y,z);
    }
  }
  }
}



/** pitch_angle_energy initialization (Assume B on z only) for test particles */
void Particles3D::pitch_angle_energy(Field * EMf) {

    /* initialize random generator with different seed on different processor */
    srand(vct->getCartesian_rank() + 3 + ns);
    assert_eq(getNOP(),0);

    double p0, pperp0, gyro_phase;

    const double q_factor =  (qom / fabs(qom)) * grid->getVOL() / npcel;

    long long counter=0;

    for (int i=1; i< grid->getNXC()-1;i++)
        for (int j=1; j< grid->getNYC()-1;j++)
            for (int k=1; k< grid->getNZC()-1;k++){

            	// q = charge following electron (species 0)
            	const double q = q_factor * EMf->getRHOcs(i, j, k, 0);

                for (int ii=0; ii < npcelx; ii++)
                    for (int jj=0; jj < npcely; jj++)
                        for (int kk=0; kk < npcelz; kk++){
                        	const double x= (ii + .5)*(dx/npcelx) + grid->getXN(i,j,k);
                        	const double y= (jj + .5)*(dy/npcely) + grid->getYN(i,j,k);
                        	const double z= (kk + .5)*(dz/npcelz) + grid->getZN(i,j,k);

                            // velocity - assumes B is along z
                            p0=sqrt((energy+1)*(energy+1)-1);
                            const double w =p0*cos(pitch_angle);
                            pperp0=p0*sin(pitch_angle);
                            gyro_phase = 2*M_PI* rand()/(double)RAND_MAX;
                            const double u=pperp0*cos(gyro_phase);
                            const double v=pperp0*sin(gyro_phase);
                            counter++ ;

                            create_new_particle(u,v,w,q,x,y,z);
                        }
            }
    const int num_ids = 1;
    longid id_list[num_ids] = {0};
    if (vct->getCartesian_rank() == 0){
    	cout << "------------------------------------------" << endl;
        cout << "Initialize Test Particle "<< ns << " with pitch angle "<< pitch_angle << ", energy " << energy << ", qom " << qom << ", npcel "<< counter<< endl;
        cout << "------------------------------------------" << endl;
    }
}


/** Force Free initialization (JxB=0) for particles */
void Particles3D::force_free(Field * EMf)
{
  eprintf("this function was not properly implemented and needs to be revised.");
#if 0
  /* initialize random generator */
  srand(vct->getCartesian_rank() + 1 + ns);
  for (int i = 1; i < grid->getNXC() - 1; i++)
  for (int j = 1; j < grid->getNYC() - 1; j++)
  for (int k = 1; k < grid->getNZC() - 1; k++)
  {
    for (int ii = 0; ii < npcelx; ii++)
    for (int jj = 0; jj < npcely; jj++)
    for (int kk = 0; kk < npcelz; kk++)
    {
      double x = (ii + .5) * (dx / npcelx) + grid->getXN(i, j, k);
      double y = (jj + .5) * (dy / npcely) + grid->getYN(i, j, k);
      double z = (kk + .5) * (dz / npcelz) + grid->getZN(i, j, k);
      // q = charge
      double q = (qom / fabs(qom)) * (EMf->getRHOcs(i, j, k, ns) / npcel) * (1.0 / invVOL);
      double shaperx = tanh((y - Ly / 2) / delta) / cosh((y - Ly / 2) / delta) / delta;
      double shaperz = 1.0 / (cosh((y - Ly / 2) / delta) * cosh((y - Ly / 2) / delta)) / delta;
      eprintf("shapery needs to be initialized.");
      eprintf("flvx etc. need to be initialized.");
      double shapery;
      // new drift velocity to satisfy JxB=0
      const double flvx = u0 * flvx * shaperx;
      const double flvz = w0 * flvz * shaperz;
      const double flvy = v0 * flvy * shapery;
      double u = c;
      double v = c;
      double w = c;
      while ((fabs(u) >= c) || (fabs(v) >= c) || (fabs(w) >= c))
      {
        sample_maxwellian(
          u, v, w,
          uth, vth, wth,
          flvx, flvy, flvz);
      }
      create_new_particle(u,v,w,q,x,y,z);
    }
  }
#endif
}

inline void Particles3D::populate_cell_with_particles(
  int i, int j, int k, double q_per_particle,
  double dx_per_pcl, double dy_per_pcl, double dz_per_pcl)
{
  const double cell_low_x = grid->getXN(i,j,k);
  const double cell_low_y = grid->getYN(i,j,k);
  const double cell_low_z = grid->getZN(i,j,k);
  for (int ii=0; ii < npcelx; ii++)
  for (int jj=0; jj < npcely; jj++)
  for (int kk=0; kk < npcelz; kk++)
  {
    double u,v,w,q,x,y,z;
    do {
      sample_maxwellian(u,v,w, uth,vth,wth, u0,v0,w0);
      x = (ii + sample_u_double())*dx_per_pcl + cell_low_x;
      y = (jj + sample_u_double())*dy_per_pcl + cell_low_y;
      z = (kk + sample_u_double())*dz_per_pcl + cell_low_z;
    } while ((x > Lx) || (y > Ly) || (z > Lz) || (x*y*z) < 0
             || sqrt(u*u + v*v + w*w) > c);
    create_new_particle(u,v,w,q_per_particle,x,y,z);
  }
}

// This could be generalized to use fluid moments
// to generate particles.
//
void Particles3D::repopulate_particles()
{
  using namespace BCparticles;

  // if this is not a boundary process then there is nothing to do
  if(!vct->isBoundaryProcess_P()) return;

  // if there are no reemission boundaries then no one has anything to do
  const bool repop_bndry_in_X = !vct->getPERIODICX_P() &&
        (bcPfaceXleft == REEMISSION || bcPfaceXright == REEMISSION);
  const bool repop_bndry_in_Y = !vct->getPERIODICY_P() &&
        (bcPfaceYleft == REEMISSION || bcPfaceYright == REEMISSION);
  const bool repop_bndry_in_Z = !vct->getPERIODICZ_P() &&
        (bcPfaceZleft == REEMISSION || bcPfaceZright == REEMISSION);
  const bool repopulation_boundary_exists =
        repop_bndry_in_X || repop_bndry_in_Y || repop_bndry_in_Z;

  if(!repopulation_boundary_exists) return;


  // boundaries to repopulate
  //
  const bool repopulateXleft = (vct->noXleftNeighbor_P() && bcPfaceXleft == REEMISSION);
  const bool repopulateYleft = (vct->noYleftNeighbor_P() && bcPfaceYleft == REEMISSION);
  const bool repopulateZleft = (vct->noZleftNeighbor_P() && bcPfaceZleft == REEMISSION);
  const bool repopulateXrght = (vct->noXrghtNeighbor_P() && bcPfaceXright == REEMISSION);
  const bool repopulateYrght = (vct->noYrghtNeighbor_P() && bcPfaceYright == REEMISSION);
  const bool repopulateZrght = (vct->noZrghtNeighbor_P() && bcPfaceZright == REEMISSION);
  const bool do_repopulate = 
       repopulateXleft || repopulateYleft || repopulateZleft
    || repopulateXrght || repopulateYrght || repopulateZrght;
  // if this process has no reemission boundaries then there is nothing to do
  if(!do_repopulate)
    return;

  // there are better ways to obtain these values...
  //
  double  FourPI =16*atan(1.0);
  const double q_per_particle
    = (qom/fabs(qom))*(Ninj/FourPI/npcel)*(1.0/grid->getInvVOL());

  const int nxc = grid->getNXC();
  const int nyc = grid->getNYC(); const int nzc = grid->getNZC();
  // number of cell layers to repopulate at boundary
  const int num_layers = 3;
  const double xLow = num_layers*dx;
  const double yLow = num_layers*dy;
  const double zLow = num_layers*dz;
  const double xHgh = Lx-xLow;
  const double yHgh = Ly-yLow;
  const double zHgh = Lz-zLow;
  if(repopulateXleft || repopulateXrght) assert_gt(nxc, 2*num_layers);
  if(repopulateYleft || repopulateYrght) assert_gt(nyc, 2*num_layers);
  if(repopulateZleft || repopulateZrght) assert_gt(nzc, 2*num_layers);

  // delete particles in repopulation layers
  //
  const int nop_orig = getNOP();
  int pidx = 0;
  while(pidx < getNOP())
  {
    // determine whether to delete the particle (using mode-aware accessors)
    const double xpcl = getX(pidx);
    const double ypcl = getY(pidx);
    const double zpcl = getZ(pidx);
    const bool delete_pcl =
      (repopulateXleft && xpcl < xLow) ||
      (repopulateYleft && ypcl < yLow) ||
      (repopulateZleft && zpcl < zLow) ||
      (repopulateXrght && xpcl > xHgh) ||
      (repopulateYrght && ypcl > yHgh) ||
      (repopulateZrght && zpcl > zHgh);
    if(delete_pcl)
      delete_particle(pidx);
    else
      pidx++;
  }
  const int nop_remaining = getNOP();

  const double dx_per_pcl = dx/npcelx;
  const double dy_per_pcl = dy/npcely;
  const double dz_per_pcl = dz/npcelz;

  // starting coordinate of upper layer
  const int upXstart = nxc-1-num_layers;
  const int upYstart = nyc-1-num_layers;
  const int upZstart = nzc-1-num_layers;

  // inject new particles.
  //
  {
    // we shrink the imagined boundaries of the array as we go along to ensure
    // that we never inject particles twice in a single mesh cell.
    //
    // initialize imagined boundaries to full subdomain excluding ghost cells.
    //
    int xbeg = 1;
    int xend = nxc-2;
    int ybeg = 1;
    int yend = nyc-2;
    int zbeg = 1;
    int zend = nzc-2;
    if (repopulateXleft)
    {
      //cout << "*** Repopulate Xleft species " << ns << " ***" << endl;
      for (int i=1; i<= num_layers; i++)
      for (int j=ybeg; j<=yend; j++)
      for (int k=zbeg; k<=zend; k++)
      {
        populate_cell_with_particles(i,j,k,q_per_particle,
          dx_per_pcl, dy_per_pcl, dz_per_pcl);
      }
      // these have all been filled, so never touch them again.
      xbeg += num_layers;
    }
    if (repopulateXrght)
    {      
      //cout << "*** Repopulate Xright species " << ns << " ***" << endl;
      for (int i=upXstart; i<=xend; i++)
      for (int j=ybeg; j<=yend; j++)
      for (int k=zbeg; k<=zend; k++)
      {
        populate_cell_with_particles(i,j,k,q_per_particle,
          dx_per_pcl, dy_per_pcl, dz_per_pcl);
      }
      // these have all been filled, so never touch them again.
      xend -= num_layers;
    }
    if (repopulateYleft)
    {     
      // cout << "*** Repopulate Yleft species " << ns << " ***" << endl;
      for (int i=xbeg; i<=xend; i++)
      for (int j=1; j<=num_layers; j++)
      for (int k=zbeg; k<=zend; k++)
      {
        populate_cell_with_particles(i,j,k,q_per_particle,
          dx_per_pcl, dy_per_pcl, dz_per_pcl);
      }
      // these have all been filled, so never touch them again.
      ybeg += num_layers;
    }
    if (repopulateYrght)
    {     
      // cout << "*** Repopulate Yright species " << ns << " ***" << endl;
      for (int i=xbeg; i<=xend; i++)
      for (int j=upYstart; j<=yend; j++)
      for (int k=zbeg; k<=zend; k++)
      {
        populate_cell_with_particles(i,j,k,q_per_particle,
          dx_per_pcl, dy_per_pcl, dz_per_pcl);
      }
      // these have all been filled, so never touch them again.
      yend -= num_layers;
    }
    if (repopulateZleft)
    {   
      //   cout << "*** Repopulate Zleft species " << ns << " ***" << endl;
      for (int i=xbeg; i<=xend; i++)
      for (int j=ybeg; j<=yend; j++)
      for (int k=1; k<=num_layers; k++)
      {
        populate_cell_with_particles(i,j,k,q_per_particle,
          dx_per_pcl, dy_per_pcl, dz_per_pcl);
      }
    }
    if (repopulateZrght)
    {   
      //   cout << "*** Repopulate Zright species " << ns << " ***" << endl;
      for (int i=xbeg; i<=xend; i++)
      for (int j=ybeg; j<=yend; j++)
      for (int k=upZstart; k<=zend; k++)
      {
        populate_cell_with_particles(i,j,k,q_per_particle,
          dx_per_pcl, dy_per_pcl, dz_per_pcl);
      }
    }
  }
  const int nop_final = getNOP();
  const int nop_deleted = nop_orig - nop_remaining;
  const int nop_created = nop_final - nop_remaining;

  //dprintf("change in # particles: %d - %d + %d = %d",nop_orig, nop_deleted, nop_created, nop_final);

  //if (vct->getCartesian_rank()==0){
  //  cout << "*** number of particles " << getNOP() << " ***" << endl;
  //}
}

void Particles3D::repopulate_particlesInfo(bool* doRepopulateInjection, bool* doRepopulateInjectionSide, cudaCommonType* repopulateBoundary) {
  
  using namespace BCparticles;

  *doRepopulateInjection = false;

  // if this is not a boundary process then there is nothing to do
  if(!vct->isBoundaryProcess_P()) return;

  // if there are no reemission boundaries then no one has anything to do
  const bool repop_bndry_in_X = !vct->getPERIODICX_P() &&
        (bcPfaceXleft == REEMISSION || bcPfaceXright == REEMISSION);
  const bool repop_bndry_in_Y = !vct->getPERIODICY_P() &&
        (bcPfaceYleft == REEMISSION || bcPfaceYright == REEMISSION);
  const bool repop_bndry_in_Z = !vct->getPERIODICZ_P() &&
        (bcPfaceZleft == REEMISSION || bcPfaceZright == REEMISSION);
  const bool repopulation_boundary_exists =
        repop_bndry_in_X || repop_bndry_in_Y || repop_bndry_in_Z;

  if(!repopulation_boundary_exists) return;

  // boundaries to repopulate
  //
  const bool repopulateXleft = (vct->noXleftNeighbor_P() && bcPfaceXleft == REEMISSION);
  const bool repopulateYleft = (vct->noYleftNeighbor_P() && bcPfaceYleft == REEMISSION);
  const bool repopulateZleft = (vct->noZleftNeighbor_P() && bcPfaceZleft == REEMISSION);
  const bool repopulateXrght = (vct->noXrghtNeighbor_P() && bcPfaceXright == REEMISSION);
  const bool repopulateYrght = (vct->noYrghtNeighbor_P() && bcPfaceYright == REEMISSION);
  const bool repopulateZrght = (vct->noZrghtNeighbor_P() && bcPfaceZright == REEMISSION);
  const bool do_repopulate = 
       repopulateXleft || repopulateYleft || repopulateZleft
    || repopulateXrght || repopulateYrght || repopulateZrght;

  if (!do_repopulate) return;
  
  *doRepopulateInjection = true; // we do repopulate

  const int nxc = grid->getNXC();
  const int nyc = grid->getNYC(); const int nzc = grid->getNZC();
  // number of cell layers to repopulate at boundary
  const int num_layers = 3;
  const double xLow = num_layers*dx;
  const double yLow = num_layers*dy;
  const double zLow = num_layers*dz;
  const double xHgh = Lx-xLow;
  const double yHgh = Ly-yLow;
  const double zHgh = Lz-zLow;
  if(repopulateXleft || repopulateXrght) assert_gt(nxc, 2*num_layers);
  if(repopulateYleft || repopulateYrght) assert_gt(nyc, 2*num_layers);
  if(repopulateZleft || repopulateZrght) assert_gt(nzc, 2*num_layers);

  doRepopulateInjectionSide[0] = repopulateXleft;
  doRepopulateInjectionSide[1] = repopulateXrght;
  doRepopulateInjectionSide[2] = repopulateYleft;
  doRepopulateInjectionSide[3] = repopulateYrght;
  doRepopulateInjectionSide[4] = repopulateZleft;
  doRepopulateInjectionSide[5] = repopulateZrght;

  repopulateBoundary[0] = xLow;
  repopulateBoundary[1] = xHgh;
  repopulateBoundary[2] = yLow;
  repopulateBoundary[3] = yHgh;
  repopulateBoundary[4] = zLow;
  repopulateBoundary[5] = zHgh;

}

void Particles3D::repopulate_particles_onlyInjection()
{
  using namespace BCparticles;

  // if this is not a boundary process then there is nothing to do
  if(!vct->isBoundaryProcess_P()) return;

  // if there are no reemission boundaries then no one has anything to do
  const bool repop_bndry_in_X = !vct->getPERIODICX_P() &&
        (bcPfaceXleft == REEMISSION || bcPfaceXright == REEMISSION);
  const bool repop_bndry_in_Y = !vct->getPERIODICY_P() &&
        (bcPfaceYleft == REEMISSION || bcPfaceYright == REEMISSION);
  const bool repop_bndry_in_Z = !vct->getPERIODICZ_P() &&
        (bcPfaceZleft == REEMISSION || bcPfaceZright == REEMISSION);
  const bool repopulation_boundary_exists =
        repop_bndry_in_X || repop_bndry_in_Y || repop_bndry_in_Z;

  if(!repopulation_boundary_exists) return;


  // boundaries to repopulate
  //
  const bool repopulateXleft = (vct->noXleftNeighbor_P() && bcPfaceXleft == REEMISSION);
  const bool repopulateYleft = (vct->noYleftNeighbor_P() && bcPfaceYleft == REEMISSION);
  const bool repopulateZleft = (vct->noZleftNeighbor_P() && bcPfaceZleft == REEMISSION);
  const bool repopulateXrght = (vct->noXrghtNeighbor_P() && bcPfaceXright == REEMISSION);
  const bool repopulateYrght = (vct->noYrghtNeighbor_P() && bcPfaceYright == REEMISSION);
  const bool repopulateZrght = (vct->noZrghtNeighbor_P() && bcPfaceZright == REEMISSION);
  const bool do_repopulate = 
       repopulateXleft || repopulateYleft || repopulateZleft
    || repopulateXrght || repopulateYrght || repopulateZrght;
  // if this process has no reemission boundaries then there is nothing to do
  if(!do_repopulate)
    return;

  // there are better ways to obtain these values...
  //
  double  FourPI =16*atan(1.0);
  const double q_per_particle
    = (qom/fabs(qom))*(Ninj/FourPI/npcel)*(1.0/grid->getInvVOL());

  const int nxc = grid->getNXC();
  const int nyc = grid->getNYC(); const int nzc = grid->getNZC();
  // number of cell layers to repopulate at boundary
  const int num_layers = 3;
  const double xLow = num_layers*dx;
  const double yLow = num_layers*dy;
  const double zLow = num_layers*dz;
  const double xHgh = Lx-xLow;
  const double yHgh = Ly-yLow;
  const double zHgh = Lz-zLow;
  if(repopulateXleft || repopulateXrght) assert_gt(nxc, 2*num_layers);
  if(repopulateYleft || repopulateYrght) assert_gt(nyc, 2*num_layers);
  if(repopulateZleft || repopulateZrght) assert_gt(nzc, 2*num_layers);


  const double dx_per_pcl = dx/npcelx;
  const double dy_per_pcl = dy/npcely;
  const double dz_per_pcl = dz/npcelz;

  // starting coordinate of upper layer
  const int upXstart = nxc-1-num_layers;
  const int upYstart = nyc-1-num_layers;
  const int upZstart = nzc-1-num_layers;

  // inject new particles.
  //
  {

    int xbeg = 1;
    int xend = nxc-2;
    int ybeg = 1;
    int yend = nyc-2;
    int zbeg = 1;
    int zend = nzc-2;
    if (repopulateXleft)
    {
      //cout << "*** Repopulate Xleft species " << ns << " ***" << endl;
      for (int i=1; i<= num_layers; i++)
      for (int j=ybeg; j<=yend; j++)
      for (int k=zbeg; k<=zend; k++)
      {
        populate_cell_with_particles(i,j,k,q_per_particle,
          dx_per_pcl, dy_per_pcl, dz_per_pcl);
      }
      // these have all been filled, so never touch them again.
      xbeg += num_layers;
    }
    if (repopulateXrght)
    {      
      //cout << "*** Repopulate Xright species " << ns << " ***" << endl;
      for (int i=upXstart; i<=xend; i++)
      for (int j=ybeg; j<=yend; j++)
      for (int k=zbeg; k<=zend; k++)
      {
        populate_cell_with_particles(i,j,k,q_per_particle,
          dx_per_pcl, dy_per_pcl, dz_per_pcl);
      }
      // these have all been filled, so never touch them again.
      xend -= num_layers;
    }
    if (repopulateYleft)
    {     
      // cout << "*** Repopulate Yleft species " << ns << " ***" << endl;
      for (int i=xbeg; i<=xend; i++)
      for (int j=1; j<=num_layers; j++)
      for (int k=zbeg; k<=zend; k++)
      {
        populate_cell_with_particles(i,j,k,q_per_particle,
          dx_per_pcl, dy_per_pcl, dz_per_pcl);
      }
      // these have all been filled, so never touch them again.
      ybeg += num_layers;
    }
    if (repopulateYrght)
    {     
      // cout << "*** Repopulate Yright species " << ns << " ***" << endl;
      for (int i=xbeg; i<=xend; i++)
      for (int j=upYstart; j<=yend; j++)
      for (int k=zbeg; k<=zend; k++)
      {
        populate_cell_with_particles(i,j,k,q_per_particle,
          dx_per_pcl, dy_per_pcl, dz_per_pcl);
      }
      // these have all been filled, so never touch them again.
      yend -= num_layers;
    }
    if (repopulateZleft)
    {   
      //   cout << "*** Repopulate Zleft species " << ns << " ***" << endl;
      for (int i=xbeg; i<=xend; i++)
      for (int j=ybeg; j<=yend; j++)
      for (int k=1; k<=num_layers; k++)
      {
        populate_cell_with_particles(i,j,k,q_per_particle,
          dx_per_pcl, dy_per_pcl, dz_per_pcl);
      }
    }
    if (repopulateZrght)
    {   
      //   cout << "*** Repopulate Zright species " << ns << " ***" << endl;
      for (int i=xbeg; i<=xend; i++)
      for (int j=ybeg; j<=yend; j++)
      for (int k=upZstart; k<=zend; k++)
      {
        populate_cell_with_particles(i,j,k,q_per_particle,
          dx_per_pcl, dy_per_pcl, dz_per_pcl);
      }
    }
  }

}


// Open BC for particles: duplicate particles on the boundary,.
// shift outside the box and update location to test if inside box
// if so, add to particle list
void Particles3D::openbc_particles_outflow()
{
  // if this is not a boundary process then there is nothing to do
  if(!vct->isBoundaryProcess_P()) return;

  //The below is OpenBC outflow for all other boundaries
  using namespace BCparticles;

  const bool openXleft = !vct->getPERIODICX_P() && vct->noXleftNeighbor_P() &&  bcPfaceXleft == OPENBCOut;
  const bool openYleft = !vct->getPERIODICY_P() && vct->noYleftNeighbor_P() &&  bcPfaceYleft == OPENBCOut;
  const bool openZleft = !vct->getPERIODICZ_P() && vct->noZleftNeighbor_P() &&  bcPfaceZleft == OPENBCOut;

  const bool openXright = !vct->getPERIODICX_P() && vct->noXrghtNeighbor_P() && bcPfaceXright == OPENBCOut;
  const bool openYright = !vct->getPERIODICY_P() && vct->noYrghtNeighbor_P() && bcPfaceYright == OPENBCOut;
  const bool openZright = !vct->getPERIODICZ_P() && vct->noZrghtNeighbor_P() && bcPfaceZright == OPENBCOut;

  if(!openXleft && !openYleft && !openZleft && !openXright && !openYright && !openZright)  return;

  const int num_layers = 3;
  assert_gt(nxc-2, (openXleft+openXright)*num_layers); //excluding 2 ghost cells, #of cells should be larger than total # of openBC layers
  assert_gt(nyc-2, (openYleft+openYright)*num_layers);
  assert_gt(nzc-2, (openZleft+openZright)*num_layers);

  const double xLow = num_layers*dx;
  const double yLow = num_layers*dy;
  const double zLow = num_layers*dz;
  const double xHgh = Lx-xLow;
  const double yHgh = Ly-yLow;
  const double zHgh = Lz-zLow;

  const bool   apply_openBC[6]    = {openXleft, openXright,openYleft, openYright,openZleft, openZright};
  const double delete_boundary[6] = {0, Lx,0, Ly,0, Lz};
  const double open_boundary[6]   = {xLow, xHgh,yLow, yHgh,zLow, zHgh};

  const int nop_orig = getNOP();
  const int capacity_out = roundup_to_multiple(nop_orig*0.1,DVECWIDTH);
  vector_SpeciesParticle injpcls(capacity_out);


  for(int dir_cnt=0;dir_cnt<6;dir_cnt++){

    if(apply_openBC[dir_cnt]){
    	  //dprintf( "*** OpenBC for Direction %d on particle species %d",dir_cnt, ns);

		  int pidx = 0;
		  int direction = dir_cnt/2;
		  double delbry  = delete_boundary[dir_cnt];
		  double openbry = open_boundary[dir_cnt];
		  double location;
		  while(pidx < getNOP())
		  {
		     // Read position for the appropriate direction (mode-aware)
		     if (direction == 0) location = getX(pidx);
		     else if (direction == 1) location = getY(pidx);
		     else location = getZ(pidx);

		     // delete the exiting particle if out of box on the direction of OpenBC
		     if((dir_cnt%2==0 && location<delbry) ||(dir_cnt%2==1 && location>delbry))
		       delete_particle(pidx);
		     else{
		       pidx++;

		       //copy the particle within open boundary to inject particle list if their shifted location after 1 time step is within simulation box
		       if ((dir_cnt%2==0 && location<openbry) ||(dir_cnt%2==1 && location>openbry)){
		    	   double injx=getX(pidx-1), injy=getY(pidx-1), injz=getZ(pidx-1);
		    	   double inju=getU(pidx-1), injv=getV(pidx-1), injw=getW(pidx-1);
		    	   double injq=getQ(pidx-1);

		    	   //shift 3 layers out, not mirror
		    	   if(direction == 0) injx = (dir_cnt%2==0) ?(injx-xLow):(injx+xLow);
		    	   if(direction == 1) injy = (dir_cnt%2==0) ?(injy-yLow):(injy+yLow);
		    	   if(direction == 2) injz = (dir_cnt%2==0) ?(injz-zLow):(injz+zLow);

		    	   injx = injx + inju*dt;
		    	   injy = injy + injv*dt;
		    	   injz = injz + injw*dt;

		    	   //Add particle if it enter that sub-domain or the domain box?
		    	   //assume create particle as long as it enters the domain box
		    	   if(injx>0 && injx<Lx && injy>0 && injy<Ly && injz>0 && injz<Lz){
		    		    injpcls.push_back(SpeciesParticle(inju,injv,injw,injq,injx,injy,injz,pclIDgenerator.generateID()));
		    	   }
		       }
		     }
		   }
	  }
  }

  //const int nop_remaining = getNOP();
  //const int nop_deleted = nop_orig - nop_remaining;
  const int nop_created = injpcls.size();

  //dprintf("change in # particles: %d - %d + %d = %d",nop_orig, nop_deleted, nop_created, nop_remaining);

  for(int outId=0;outId<nop_created;outId++) {
	  const SpeciesParticle& pcl = injpcls[outId];
	  add_new_particle(pcl.get_u(), pcl.get_v(), pcl.get_w(), pcl.get_q(),
	                   pcl.get_x(), pcl.get_y(), pcl.get_z(), pcl.get_t());
  }
}

void Particles3D::openbc_particles_outflowInfo(bool* doOpenBC, bool* applyOpenBC, cudaCommonType* delBdry, cudaCommonType* openBdry) {
  *doOpenBC = false;
  if(!vct->isBoundaryProcess_P()) return;

  using namespace BCparticles;

  const bool openXleft = !vct->getPERIODICX_P() && vct->noXleftNeighbor_P() &&  bcPfaceXleft == OPENBCOut;
  const bool openYleft = !vct->getPERIODICY_P() && vct->noYleftNeighbor_P() &&  bcPfaceYleft == OPENBCOut;
  const bool openZleft = !vct->getPERIODICZ_P() && vct->noZleftNeighbor_P() &&  bcPfaceZleft == OPENBCOut;

  const bool openXright = !vct->getPERIODICX_P() && vct->noXrghtNeighbor_P() && bcPfaceXright == OPENBCOut;
  const bool openYright = !vct->getPERIODICY_P() && vct->noYrghtNeighbor_P() && bcPfaceYright == OPENBCOut;
  const bool openZright = !vct->getPERIODICZ_P() && vct->noZrghtNeighbor_P() && bcPfaceZright == OPENBCOut;

  if(!openXleft && !openYleft && !openZleft && !openXright && !openYright && !openZright)  return;

  *doOpenBC = true; // if we got here, we are doing openBC

  applyOpenBC[0] = openXleft;
  applyOpenBC[1] = openXright;
  applyOpenBC[2] = openYleft;
  applyOpenBC[3] = openYright;
  applyOpenBC[4] = openZleft;
  applyOpenBC[5] = openZright;

  const int num_layers = 3;
  assert_gt(nxc-2, (openXleft+openXright)*num_layers); 
  assert_gt(nyc-2, (openYleft+openYright)*num_layers);
  assert_gt(nzc-2, (openZleft+openZright)*num_layers);

  const double xLow = num_layers*dx;
  const double yLow = num_layers*dy;
  const double zLow = num_layers*dz;
  const double xHgh = Lx-xLow;
  const double yHgh = Ly-yLow;
  const double zHgh = Lz-zLow;

  delBdry[0] = 0; delBdry[1] = Lx; delBdry[2] = 0; delBdry[3] = Ly; delBdry[4] = 0; delBdry[5] = Lz;
  openBdry[0] = xLow; openBdry[1] = xHgh; openBdry[2] = yLow; openBdry[3] = yHgh; openBdry[4] = zLow; openBdry[5] = zHgh;

}

