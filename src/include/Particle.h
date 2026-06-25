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

#ifndef _Particle_
#define _Particle_
#include "ipicdefs.h" // for longid
#include "cudaTypeDef.cuh"

// Depends on width of vector unit;
// need to be known at compile time.
//
#define AoS_PCLS_AT_A_TIME 2


template <class T>
class Larray;
// intended to occupy 64 bytes (8 doubles x 8 bytes = 64 bytes)
//
// particle for a specific species in a PIC simulation
class SpeciesParticle
{
  cudaParticleType u[3]; // velocity components (u, v, w) [normalized to speed of light c]
  cudaParticleType q;    // charge of the macroparticle [simulation units]
  cudaParticleType x[3]; // position components (x, y, z) [in cell-length units]
  cudaPclType_ID id;     // particle identifier
 public:
  __host__ __device__ SpeciesParticle(){}
  __host__ __device__ SpeciesParticle(
    cudaParticleType u_,  // x-component of velocity
    cudaParticleType v_,  // y-component of velocity
    cudaParticleType w_,  // z-component of velocity
    cudaParticleType q_,  // macroparticle charge
    cudaParticleType x_,  // x-position
    cudaParticleType y_,  // y-position
    cudaParticleType z_,  // z-position
    cudaPclType_ID id_)   // particle identifier
  {
    u[0]=u_;
    u[1]=v_;
    u[2]=w_;
    q=q_;
    x[0]=x_;
    x[1]=y_;
    x[2]=z_;
    id=id_;
  }
  // accessors
  // cudaParticleType component(int i){ return u[i]; } // a hack
  __host__ __device__ cudaParticleType get_u(int i)const{ return u[i]; }
  __host__ __device__ cudaParticleType get_q()const{ return q; }
  __host__ __device__ cudaParticleType get_x(int i)const{ return x[i]; }
  __host__ __device__ cudaPclType_ID get_id()const{ return id; }

  __host__ __device__ void set_u(cudaTypeSingle* in, int n=3) { for(int i=0;i<n;i++) u[i] = in[i]; }
  __host__ __device__ void set_u(int i, cudaTypeSingle in) { u[i] = in; }
  __host__ __device__ void set_q(cudaTypeSingle in) { q = in; }
  __host__ __device__ void set_x(int i, cudaTypeSingle in) { x[i] = in; }

  __host__ __device__ void set_u(cudaTypeDouble* in, int n=3) { for(int i=0;i<n;i++) u[i] = in[i]; }
  __host__ __device__ void set_u(int i, cudaTypeDouble in) { u[i] = in; }
  __host__ __device__ void set_q(cudaTypeDouble in) { q = in; }
  __host__ __device__ void set_x(int i, cudaTypeDouble in) { x[i] = in; }
  __host__ __device__ void set_id(cudaPclType_ID in){ id=in; }

  __host__ __device__ void set_x_u(cudaParticleType x, cudaParticleType y, cudaParticleType z, 
                                    cudaParticleType u, cudaParticleType v, cudaParticleType w){
    this->u[0] = u;
    this->u[1] = v;
    this->u[2] = w;

    this->x[0] = x;
    this->x[1] = y;
    this->x[2] = z;

  }
  
  longid get_ID()const{ return longid(id); }
  void set_ID(longid in){ id = cudaPclType_ID(in); }
  // alternative accessors
  __host__ __device__ cudaParticleType get_x()const{ return x[0]; }
  __host__ __device__ cudaParticleType get_y()const{ return x[1]; }
  __host__ __device__ cudaParticleType get_z()const{ return x[2]; }
  __host__ __device__ cudaParticleType get_u()const{ return u[0]; }
  __host__ __device__ cudaParticleType get_v()const{ return u[1]; }
  __host__ __device__ cudaParticleType get_w()const{ return u[2]; }
  __host__ __device__ cudaParticleType& fetch_x(){ return x[0]; }
  __host__ __device__ cudaParticleType& fetch_y(){ return x[1]; }
  __host__ __device__ cudaParticleType& fetch_z(){ return x[2]; }
  __host__ __device__ cudaParticleType& fetch_q(){ return q; }
  __host__ __device__ cudaParticleType& fetch_u(){ return u[0]; }
  __host__ __device__ cudaParticleType& fetch_v(){ return u[1]; }
  __host__ __device__ cudaParticleType& fetch_w(){ return u[2]; }
  __host__ __device__ cudaPclType_ID& fetch_id(){ return id; }

  __host__ __device__ void set_x(cudaTypeSingle in){ x[0]=in; }
  __host__ __device__ void set_y(cudaTypeSingle in){ x[1]=in; }
  __host__ __device__ void set_z(cudaTypeSingle in){ x[2]=in; }
  __host__ __device__ void set_u(cudaTypeSingle in){ u[0]=in; }
  __host__ __device__ void set_v(cudaTypeSingle in){ u[1]=in; }
  __host__ __device__ void set_w(cudaTypeSingle in){ u[2]=in; }
// double for compatibility
  __host__ __device__ void set_x(cudaTypeDouble in){ x[0]=in; }
  __host__ __device__ void set_y(cudaTypeDouble in){ x[1]=in; }
  __host__ __device__ void set_z(cudaTypeDouble in){ x[2]=in; }
  __host__ __device__ void set_u(cudaTypeDouble in){ u[0]=in; }
  __host__ __device__ void set_v(cudaTypeDouble in){ u[1]=in; }
  __host__ __device__ void set_w(cudaTypeDouble in){ u[2]=in; }

  __host__ __device__ void set_to_zero()
  {
    u[0] = 0; u[1] = 0; u[2] = 0; q = 0;
    x[0] = 0; x[1] = 0; x[2] = 0; id = 0;
  }
  __host__ __device__ void set(
    cudaParticleType _u, cudaParticleType _v, cudaParticleType _w, cudaParticleType _q,
    cudaParticleType _x, cudaParticleType _y, cudaParticleType _z, cudaPclType_ID _id
    )
  {
    u[0] = _u; u[1] = _v; u[2] = _w; q = _q;
    x[0] = _x; x[1] = _y; x[2] = _z; id = _id;
  }
};

#endif
