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
  Particles3Dcommcomm.h  -  Class for particles of the same species, in a 2D space and 3component velocity with communications methods
  -------------------
developers: Stefano Markidis, Giovanni Lapenta
 ********************************************************************************************/

#ifndef Part3DCOMM_H
#define Part3DCOMM_H

#include "ipicfwd.h"
#include "Alloc.h"
#include "Particle.h" // for ParticleType
// unfortunately this includes mpi.h, which includes 35000 lines:
#include "BlockCommunicator.h"
#include "aligned_vector.h"
#include "Larray.h"
#include "IDgenerator.h"

#include "cudaTypeDef.cuh"

namespace BCparticles
{
    enum Enum
    {
        EXIT = 0,
        PERFECT_MIRROR = 1,
        REEMISSION = 2,
        OPENBCOut = 3,
        OPENBCIn = 4
    };
}

/**
 * 
 * class for particles of the same species with communications methods
 * @date Fri Jun 4 2007
 * @author Stefano Markidis, Giovanni Lapenta
 * @version 2.0
 *
 */
class Particles3Dcomm // :public Particles
{
public:
  /** constructor */
  Particles3Dcomm(int species, CollectiveIO * col,
    VirtualTopology3D * vct, Grid * grid,
    StorageMode mode = StorageMode::SoA);

  void reserveSpace(int nop);
  void restartLoad();

  /** destructor */
  ~Particles3Dcomm();

 public: // handle boundary conditions
  // apply boundary conditions to all particles at the
  // end of a list of particles starting with index start
  // 
  // these are virtual so user can override these
  // to provide arbitrary custom boundary conditions
  virtual void apply_Xleft_BC(vector_SpeciesParticle& pcls, int start=0);
  virtual void apply_Yleft_BC(vector_SpeciesParticle& pcls, int start=0);
  virtual void apply_Zleft_BC(vector_SpeciesParticle& pcls, int start=0);
  virtual void apply_Xrght_BC(vector_SpeciesParticle& pcls, int start=0);
  virtual void apply_Yrght_BC(vector_SpeciesParticle& pcls, int start=0);
  virtual void apply_Zrght_BC(vector_SpeciesParticle& pcls, int start=0);
 private: // handle boundary conditions
  void apply_periodic_BC_global(vector_SpeciesParticle& pcl_list, int pstart);
  bool test_pcls_are_in_nonperiodic_domain(const vector_SpeciesParticle& pcls)const;
  bool test_pcls_are_in_domain(const vector_SpeciesParticle& pcls)const;
  bool test_outside_domain(const SpeciesParticle& pcl)const;
  bool test_outside_nonperiodic_domain(const SpeciesParticle& pcl)const;
  bool test_Xleft_of_domain(const SpeciesParticle& pcl)
  { return pcl.get_x() < 0.; }
  bool test_Xrght_of_domain(const SpeciesParticle& pcl)
  { return pcl.get_x() > Lx; }
  bool test_Yleft_of_domain(const SpeciesParticle& pcl)
  { return pcl.get_y() < 0.; }
  bool test_Yrght_of_domain(const SpeciesParticle& pcl)
  { return pcl.get_y() > Ly; }
  bool test_Zleft_of_domain(const SpeciesParticle& pcl)
  { return pcl.get_z() < 0.; }
  bool test_Zrght_of_domain(const SpeciesParticle& pcl)
  { return pcl.get_z() > Lz; }
  void apply_nonperiodic_BCs_global(vector_SpeciesParticle&, int pstart);
  bool test_all_pcls_are_in_subdomain();
  void apply_BCs_globally(vector_SpeciesParticle& pcl_list);
  void apply_BCs_locally(vector_SpeciesParticle& pcl_list,
    int direction, bool apply_shift, bool do_apply_BCs);
 private: // communicate particles between processes
  void flush_send();
  bool send_pcl_to_appropriate_buffer(SpeciesParticle& pcl, int count[6]);
  int handle_received_particles(int pclCommMode=0);
 public:
  int separate_and_send_particles();
  void recommunicate_particles_until_done(int min_num_iterations=3);
  void pad_capacities();
 private:
  void resize_AoS(int nop);
  void resize_SoA(int nop);
  void resizeActive(int nop);
 public:
  void copyParticlesToAoS();
  void copyParticlesToSoA();

 public:
  // --- Legacy conversion shims (compatibility) ---
  void convertParticlesToSynched();
  void convertParticlesToAoS();
  void convertParticlesToSoA();
  bool particlesAreSoA()const;

  // --- New SoA-primary helpers ---
  /** Pack SoA vectors into _pcls AoS buffer (for I/O or MPI bridging). */
  void packSoAToAoS();
  /** Clear the _pcls AoS buffer to free memory after use. */
  void clearAoS() { _pcls.clear(); }
  /** Append particles from an AoS buffer into the active storage. */
  void appendFromAoS(const SpeciesParticle* buf, int count);
  /** Get the storage mode of this instance. */
  StorageMode getStorageMode() const { return storageMode; }
  bool isSoAMode() const { return storageMode == StorageMode::SoA; }
  bool isAoSMode() const { return storageMode == StorageMode::AoS; }

  /** Clear all particles from the active storage (mode-aware). */
  void clearParticles() {
    if (storageMode == StorageMode::SoA) {
      u.clear(); v.clear(); w.clear(); q.clear();
      x.clear(); y.clear(); z.clear(); t.clear();
    } else {
      _pcls.clear();
    }
  }

  /*! sort particles for vectorized push (needs to be parallelized) */
  void sort_particles_serial();
  void sort_particles_serial_AoS();
  void sort_particles_serial_SoA();
  void sort_particles_parallel(int* cellCount, int* cellOffset);
  void sort_particles_parallel_AoS(int* globalCount, int* bucketOffset);
  void sort_particles_parallel_SoA(int* cellCount, int* cellOffset);

  // get accessors for optional arrays
  //
  //Larray<SpeciesParticle>& fetch_pcls(){ return _pcls; }
  //Larray<SpeciesParticle>& fetch_pclstmp(){ return _pclstmp; }

  // particle creation methods
  //
  void reserve_remaining_particle_IDs()
  {
    // reserve remaining particle IDs starting from getNOP()
    pclIDgenerator.reserve_particles_in_range(getNOP());
  }
  // create new particle (pushes to active storage based on storageMode)
  void create_new_particle(
    cudaParticleType u_, cudaParticleType v_, cudaParticleType w_, cudaParticleType q_,
    cudaParticleType x_, cudaParticleType y_, cudaParticleType z_)
  {
    const cudaParticleType t_ = pclIDgenerator.generateID();
    if (storageMode == StorageMode::SoA) {
      u.push_back(u_); v.push_back(v_); w.push_back(w_); q.push_back(q_);
      x.push_back(x_); y.push_back(y_); z.push_back(z_); t.push_back(t_);
    } else {
      _pcls.push_back(SpeciesParticle(u_,v_,w_,q_,x_,y_,z_,t_));
    }
  }
  // add particle with explicit ID to the active storage
  void add_new_particle(
    cudaParticleType u_, cudaParticleType v_, cudaParticleType w_, cudaParticleType q_,
    cudaParticleType x_, cudaParticleType y_, cudaParticleType z_, cudaParticleType t_)
  {
    if (storageMode == StorageMode::SoA) {
      u.push_back(u_); v.push_back(v_); w.push_back(w_); q.push_back(q_);
      x.push_back(x_); y.push_back(y_); z.push_back(z_); t.push_back(t_);
    } else {
      _pcls.push_back(SpeciesParticle(u_,v_,w_,q_,x_,y_,z_,t_));
    }
  }

  // swap-remove particle at index pidx from the active storage
  void delete_particle(int pidx)
  {
    if (storageMode == StorageMode::SoA) {
      const int last = getNOP() - 1;
      if (pidx != last) {
        u[pidx]=u[last]; v[pidx]=v[last]; w[pidx]=w[last]; q[pidx]=q[last];
        x[pidx]=x[last]; y[pidx]=y[last]; z[pidx]=z[last]; t[pidx]=t[last];
      }
      u.pop_back(); v.pop_back(); w.pop_back(); q.pop_back();
      x.pop_back(); y.pop_back(); z.pop_back(); t.pop_back();
    } else {
      _pcls[pidx]=_pcls.back();
      _pcls.pop_back();
    }
  }

  // inline get accessors
  //
  double get_dx(){return dx;}
  double get_dy(){return dy;}
  double get_dz(){return dz;}
  double get_invdx(){return inv_dx;}
  double get_invdy(){return inv_dy;}
  double get_invdz(){return inv_dz;}
  double get_xstart(){return xstart;}
  double get_ystart(){return ystart;}
  double get_zstart(){return zstart;}
  // Legacy compatibility shims for ParticleType
  ParticleType::Type get_particleType()const {
    return (storageMode == StorageMode::SoA) ? ParticleType::SoA : ParticleType::AoS;
  }
  void set_particleType(ParticleType::Type /*newType*/) {
    // No-op: storageMode is fixed at construction. Kept for API compatibility.
  }
  const SpeciesParticle& get_pcl(int pidx)const{ return _pcls[pidx]; }
  const vector_SpeciesParticle& get_pcl_list()const{ return _pcls; }
  vector_SpeciesParticle& get_pcl_array(){ return _pcls; }
  vector_SpeciesParticle* get_pcl_arrayPtr(){ return &_pcls; }
  const SpeciesParticle* get_pclptr(int id)const{ return &(_pcls[id]); }

  // Mode-safe AoS data accessors (assert AoS mode at call site)
  /** Raw pointer to AoS particle data. Asserts AoS mode. */
  SpeciesParticle* getAoSDataPtr() {
    assert(storageMode == StorageMode::AoS && "getAoSDataPtr called on SoA-mode instance");
    return _pcls.getList();
  }
  /** Number of particles in AoS storage. Asserts AoS mode. */
  int getAoSSize() const {
    assert(storageMode == StorageMode::AoS && "getAoSSize called on SoA-mode instance");
    return _pcls.size();
  }
  /** Set the AoS particle count. Asserts AoS mode. */
  void setAoSSize(int n) {
    assert(storageMode == StorageMode::AoS && "setAoSSize called on SoA-mode instance");
    _pcls.setSize(n);
  }
  /** AoS storage capacity. Asserts AoS mode. */
  int getAoSCapacity() const {
    assert(storageMode == StorageMode::AoS && "getAoSCapacity called on SoA-mode instance");
    return _pcls.capacity();
  }
  /** Reserve AoS storage. Asserts AoS mode. */
  void reserveAoS(int cap) {
    assert(storageMode == StorageMode::AoS && "reserveAoS called on SoA-mode instance");
    _pcls.reserve(cap);
  }
  // SoA bulk accessors — always valid in SoA mode
  const double *getUall()  const { return &u[0]; }
  const double *getVall()  const { return &v[0]; }
  const double *getWall()  const { return &w[0]; }
  const double *getQall()  const { return &q[0]; }
  const double *getXall()  const { return &x[0]; }
  const double *getYall()  const { return &y[0]; }
  const double *getZall()  const { return &z[0]; }
  const double *getParticleIDall() const{ return &t[0]; }
  // Mutable SoA data pointers (for direct D→H copies into host SoA vectors)
  double *getUallMut() { return &u[0]; }
  double *getVallMut() { return &v[0]; }
  double *getWallMut() { return &w[0]; }
  double *getQallMut() { return &q[0]; }
  double *getXallMut() { return &x[0]; }
  double *getYallMut() { return &y[0]; }
  double *getZallMut() { return &z[0]; }
  double *getTallMut() { return &t[0]; }
  /** Prepare SoA vectors to receive nop particles (reserve + resize). */
  void prepareSoAForNOP(int nop) {
    resize_SoA(nop);
  }
  // accessors for particle with index indexPart
  //
  int getNOP()  const {
    return (storageMode == StorageMode::SoA) ? (int)u.size() : (int)_pcls.size();
  }
  // set particle components (delegates to active storage)
  void setU(int i, cudaParticleType in){ if(isSoAMode()) u[i]=in; else _pcls[i].set_u(in); }
  void setV(int i, cudaParticleType in){ if(isSoAMode()) v[i]=in; else _pcls[i].set_v(in); }
  void setW(int i, cudaParticleType in){ if(isSoAMode()) w[i]=in; else _pcls[i].set_w(in); }
  void setQ(int i, cudaParticleType in){ if(isSoAMode()) q[i]=in; else _pcls[i].set_q(in); }
  void setX(int i, cudaParticleType in){ if(isSoAMode()) x[i]=in; else _pcls[i].set_x(in); }
  void setY(int i, cudaParticleType in){ if(isSoAMode()) y[i]=in; else _pcls[i].set_y(in); }
  void setZ(int i, cudaParticleType in){ if(isSoAMode()) z[i]=in; else _pcls[i].set_z(in); }
  void setT(int i, cudaParticleType in){ if(isSoAMode()) t[i]=in; else _pcls[i].set_t(in); }
  // fetch particle components (mutable reference)
  cudaParticleType& fetchU(int i){ return isSoAMode() ? u[i] : _pcls[i].fetch_u(); }
  cudaParticleType& fetchV(int i){ return isSoAMode() ? v[i] : _pcls[i].fetch_v(); }
  cudaParticleType& fetchW(int i){ return isSoAMode() ? w[i] : _pcls[i].fetch_w(); }
  cudaParticleType& fetchQ(int i){ return isSoAMode() ? q[i] : _pcls[i].fetch_q(); }
  cudaParticleType& fetchX(int i){ return isSoAMode() ? x[i] : _pcls[i].fetch_x(); }
  cudaParticleType& fetchY(int i){ return isSoAMode() ? y[i] : _pcls[i].fetch_y(); }
  cudaParticleType& fetchZ(int i){ return isSoAMode() ? z[i] : _pcls[i].fetch_z(); }
  cudaParticleType& fetchT(int i){ return isSoAMode() ? t[i] : _pcls[i].fetch_t(); }
  // get particle components (read-only)
  cudaParticleType getU(int i)const{ return isSoAMode() ? u[i] : _pcls[i].get_u(); }
  cudaParticleType getV(int i)const{ return isSoAMode() ? v[i] : _pcls[i].get_v(); }
  cudaParticleType getW(int i)const{ return isSoAMode() ? w[i] : _pcls[i].get_w(); }
  cudaParticleType getQ(int i)const{ return isSoAMode() ? q[i] : _pcls[i].get_q(); }
  cudaParticleType getX(int i)const{ return isSoAMode() ? x[i] : _pcls[i].get_x(); }
  cudaParticleType getY(int i)const{ return isSoAMode() ? y[i] : _pcls[i].get_y(); }
  cudaParticleType getZ(int i)const{ return isSoAMode() ? z[i] : _pcls[i].get_z(); }
  cudaParticleType getT(int i)const{ return isSoAMode() ? t[i] : _pcls[i].get_t(); }
  //int get_npmax() const {return npmax;}

  // computed get access
  //
  /** return the Kinetic energy */
  double getKe();
  /** return the maximum kinetic energy */
  double getMaxVelocity();
  /** return energy distribution */
  long long *getVelocityDistribution(int nBins, double maxVel);
  /** return the momentum */
  double getP();
  /** return the total charge (sum of particle weights q) */
  double getTotalQ();

public:
  // accessors
  //int get_ns()const{return ns;}
  // return number of this species
  int get_species_num()const{return ns;}
  int get_numpcls_in_bucket(int cx, int cy, int cz)const
  { return (*numpcls_in_bucket)[cx][cy][cz]; }
  int get_bucket_offset(int cx, int cy, int cz)const
  { return (*bucket_offset)[cx][cy][cz]; }

protected:
  // pointers to topology and grid information
  // (should be const)
  const Collective * col;
  const VirtualTopology3D * vct;
  const Grid * grid;
  //
  /** number of this species */
  int ns;
  /** maximum number of particles of this species on this domain. used for memory allocation */
  //int npmax;
  /** number of particles of this species on this domain */
  //int nop; // see getNOP();
  /** total number of particles */
  //long long np_tot;
  /** number of particles per cell */
  int npcel;
  /** number of particles per cell - X direction */
  int npcelx;
  /** number of particles per cell - Y direction */
  int npcely;
  /** number of particles per cell - Z direction */
  int npcelz;
  /** charge to mass ratio */
  double qom;
  /** recon thick */
  double delta;
  /** thermal velocity  - Direction X*/
  double uth;
  /** thermal velocity  - Direction Y*/
  double vth;
  /** thermal velocity  - Direction Z*/
  double wth;
  /** u0 Drift velocity - Direction X */
  double u0;
  /** v0 Drift velocity - Direction Y */
  double v0;
  /** w0 Drift velocity - Direction Z */
  double w0;
  // used to generate unique particle IDs
  doubleIDgenerator pclIDgenerator;

  // Legacy particleType kept for compatibility; storageMode is the authority
  ParticleType::Type particleType;
  StorageMode storageMode;
  //
  // AoS representation
  //
  //Larray<SpeciesParticle> _pcls;
  vector_SpeciesParticle_registered _pcls;
  //
  // particles data
  //
  // SoA representation  (pinned host memory for async GPU transfers)
  //
  // velocity components
  vector_double_registered u;
  vector_double_registered v;
  vector_double_registered w;
  // charge
  vector_double_registered q;
  // position
  vector_double_registered x;
  vector_double_registered y;
  vector_double_registered z;
  // subcycle time
  vector_double_registered t;
  // indicates whether this class is for tracking particles
  //bool TrackParticleID;
  bool isTestParticle;
  double pitch_angle;
  double energy;

  // structures for sorting particles
  //
  /** Average position data (used during particle push) **/
  //
  //Larray<double>& _xavg;
  //Larray<double>& _yavg;
  //Larray<double>& _zavg;
  //
  // alternate temporary storage for sorting particles
  //
  vector_SpeciesParticle_registered _pclstmp;
  //
  // references for buckets for serial sort.
  //
  array3_int* numpcls_in_bucket;
  array3_int* numpcls_in_bucket_now; // accumulator used during sorting
  //array3_int* bucket_size; // maximum number of particles in bucket
  array3_int* bucket_offset;

  /** rank of processor in which particle is created (for ID) */
  int BirthRank[2];
  /** number of variables to be stored in buffer for communication for each particle  */
  int nVar;
  /** time step */
  double dt;
  //
  // Copies of grid data (should just put pointer to Grid in this class)
  //
  /** Simulation domain lengths */
  double xstart, xend, ystart, yend, zstart, zend, invVOL;
  /** Lx = simulation box length - x direction   */
  double Lx;
  /** Ly = simulation box length - y direction   */
  double Ly;
  /** Lz = simulation box length - z direction   */
  double Lz;
  /** grid spacings */
  double dx, dy, dz;
  /** number of grid nodes */
  int nxn, nyn, nzn;
  /** number of grid cells */
  int nxc, nyc, nzc;
  // convenience values from grid
  double inv_dx;
  double inv_dy;
  double inv_dz;
  //
  // Communication variables
  //
  /** buffers for communication */
  //
  // communicator for this specie
  MPI_Comm mpi_comm;
  // send buffers
  //
  BlockCommunicator<SpeciesParticle> sendXleft;
  BlockCommunicator<SpeciesParticle> sendXrght;
  BlockCommunicator<SpeciesParticle> sendYleft;
  BlockCommunicator<SpeciesParticle> sendYrght;
  BlockCommunicator<SpeciesParticle> sendZleft;
  BlockCommunicator<SpeciesParticle> sendZrght;
  //
  // recv buffers
  //
  BlockCommunicator<SpeciesParticle> recvXleft;
  BlockCommunicator<SpeciesParticle> recvXrght;
  BlockCommunicator<SpeciesParticle> recvYleft;
  BlockCommunicator<SpeciesParticle> recvYrght;
  BlockCommunicator<SpeciesParticle> recvZleft;
  BlockCommunicator<SpeciesParticle> recvZrght;

  /** bool for communication verbose */
  bool cVERBOSE;
  /** Boundary condition on particles:
          <ul>
          <li>0 = exit</li>
          <li>1 = perfect mirror</li>
          <li>2 = riemission</li>
          <li>3 = periodic condition </li>
          </ul>
          */
public:
  /** Boundary Condition Particles: FaceXright */
  int bcPfaceXright;
  /** Boundary Condition Particles: FaceXleft */
  int bcPfaceXleft;
  /** Boundary Condition Particles: FaceYright */
  int bcPfaceYright;
  /** Boundary Condition Particles: FaceYleft */
  int bcPfaceYleft;
  /** Boundary Condition Particles: FaceYright */
  int bcPfaceZright;
  /** Boundary Condition Particles: FaceYleft */
  int bcPfaceZleft;
protected:
  //
  // Other variables
  //
  /** speed of light in vacuum */
  double c;
  /** restart variable for loading particles from restart file */
  int restart;
  /** Number of iteration of the mover*/
  int NiterMover;
  /** velocity of the injection of the particles */
  double Vinj;
  /** removed charge from species */
  double Q_removed;
  /** density of the injection of the particles */
  double Ninj;

 protected:

  // limits to apply to particle velocity
  //
  double umax;
  double vmax;
  double wmax;
  double umin;
  double vmin;
  double wmin;

};

// find the particles with particular IDs and print them
void print_pcls(vector_SpeciesParticle& pcls, int ns, longid* id_list, int num_ids);

// typedef Particles3Dcomm Particles; // Now defined in ipicfwd.h as ParticleSoAHost

#endif
