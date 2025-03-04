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

namespace ParticleType
{
  enum Type
  {
    AoS = 0,
    SoA,
    synched
  };
}


template <class T>
class Larray;
// intended to occupy 64 bytes
//
// particle for a specific species
class SpeciesParticle
{
  cudaParticleType u[3];
  cudaParticleType q;
  cudaParticleType x[3];
  cudaParticleType t;
 public:
  SpeciesParticle(){}
  SpeciesParticle(
    cudaParticleType u_,
    cudaParticleType v_,
    cudaParticleType w_,
    cudaParticleType q_,
    cudaParticleType x_,
    cudaParticleType y_,
    cudaParticleType z_,
    cudaParticleType t_)
  {
    u[0]=u_;
    u[1]=v_;
    u[2]=w_;
    q=q_;
    x[0]=x_;
    x[1]=y_;
    x[2]=z_;
    t=t_;
  }
  // accessors
  // cudaParticleType component(int i){ return u[i]; } // a hack
  __host__ __device__ cudaParticleType get_u(int i)const{ return u[i]; }
  __host__ __device__ cudaParticleType get_q()const{ return q; }
  __host__ __device__ cudaParticleType get_x(int i)const{ return x[i]; }
  __host__ __device__ cudaParticleType get_t()const{ return t; }

  __host__ __device__ void set_u(cudaTypeSingle* in, int n=3) { for(int i=0;i<n;i++) u[i] = in[i]; }
  __host__ __device__ void set_u(int i, cudaTypeSingle in) { u[i] = in; }
  __host__ __device__ void set_q(cudaTypeSingle in) { q = in; }
  __host__ __device__ void set_x(int i, cudaTypeSingle in) { x[i] = in; }
  __host__ __device__ void set_t(cudaTypeSingle in){ t=in; }

  __host__ __device__ void set_u(cudaTypeDouble* in, int n=3) { for(int i=0;i<n;i++) u[i] = in[i]; }
  __host__ __device__ void set_u(int i, cudaTypeDouble in) { u[i] = in; }
  __host__ __device__ void set_q(cudaTypeDouble in) { q = in; }
  __host__ __device__ void set_x(int i, cudaTypeDouble in) { x[i] = in; }
  __host__ __device__ void set_t(cudaTypeDouble in){ t=in; }

  __host__ __device__ void set_x_u(cudaParticleType x, cudaParticleType y, cudaParticleType z, 
                                    cudaParticleType u, cudaParticleType v, cudaParticleType w){
    this->u[0] = u;
    this->u[1] = v;
    this->u[2] = w;

    this->x[0] = x;
    this->x[1] = y;
    this->x[2] = z;

  }
  
  // tracking particles would actually use q for the ID
  longid get_ID()const{ return longid(t); }
  void set_ID(longid in){ t = cudaParticleType(in); }
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
  __host__ __device__ cudaParticleType& fetch_t(){ return t; }

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
    for(int i=0;i<8;i++) u[i]=0;
  }
  __host__ __device__ void set(
    cudaParticleType _u, cudaParticleType _v, cudaParticleType _w, cudaParticleType _q,
    cudaParticleType _x, cudaParticleType _y, cudaParticleType _z, cudaParticleType _t
    )
  {
    u[0] = _u; u[1] = _v; u[2] = _w; q = _q;
    x[0] = _x; x[1] = _y; x[2] = _z; t = _t;
  }
};

// to support SoA notation
//
// this class will simply be defined differently
// when underlying representation is SoA
//
//class FetchPclComponent
//{
//  int offset;
//  Larray<SpeciesParticle>& list;
// public:
//  FetchPclComponent( Larray<SpeciesParticle>& _list, int _offset)
//  : list(_list), offset(_offset)
//  { }
//  cudaParticleType operator[](int i)
//  {
//    return list[i].component(offset);
//    // return component(offset)[i];
//  }
//};

// intended to occupy 64 bytes
//
// dust particle for second-order-accuracy implicit advance
#if 0
class IDpcl
{
  int c[3]; // cell
  float q; // charge
  float x[3]; // position
  float t; // subcycle time
  float hdx[3]; // xavg = x + hdx
  float qom; // charge to mass ratio of particle
  float u[3];
  float m; // mass of particle
};
#endif

#endif
