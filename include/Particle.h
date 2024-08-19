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
class CUDA_ALIGN(64) SpeciesParticle
{
  cudaCommonType u[3];
  cudaCommonType q;
  cudaCommonType x[3];
  cudaCommonType t;
 public:
  SpeciesParticle(){}
  SpeciesParticle(
    cudaCommonType u_,
    cudaCommonType v_,
    cudaCommonType w_,
    cudaCommonType q_,
    cudaCommonType x_,
    cudaCommonType y_,
    cudaCommonType z_,
    cudaCommonType t_)
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
  // cudaCommonType component(int i){ return u[i]; } // a hack
  __host__ __device__ cudaCommonType get_u(int i)const{ return u[i]; }
  __host__ __device__ cudaCommonType get_q()const{ return q; }
  __host__ __device__ cudaCommonType get_x(int i)const{ return x[i]; }
  __host__ __device__ cudaCommonType get_t()const{ return t; }
  __host__ __device__ void set_u(cudaCommonType* in, int n=3) { for(int i=0;i<n;i++) u[i] = in[i]; }
  __host__ __device__ void set_u(int i, cudaCommonType in) { u[i] = in; }
  __host__ __device__ void set_q(cudaCommonType in) { q = in; }
  __host__ __device__ void set_x(int i, cudaCommonType in) { x[i] = in; }
  __host__ __device__ void set_t(cudaCommonType in){ t=in; }
  // tracking particles would actually use q for the ID
  longid get_ID()const{ return longid(t); }
  void set_ID(longid in){ t = cudaCommonType(in); }
  // alternative accessors
  __host__ __device__ cudaCommonType get_x()const{ return x[0]; }
  __host__ __device__ cudaCommonType get_y()const{ return x[1]; }
  __host__ __device__ cudaCommonType get_z()const{ return x[2]; }
  __host__ __device__ cudaCommonType get_u()const{ return u[0]; }
  __host__ __device__ cudaCommonType get_v()const{ return u[1]; }
  __host__ __device__ cudaCommonType get_w()const{ return u[2]; }
  __host__ __device__ cudaCommonType& fetch_x(){ return x[0]; }
  __host__ __device__ cudaCommonType& fetch_y(){ return x[1]; }
  __host__ __device__ cudaCommonType& fetch_z(){ return x[2]; }
  __host__ __device__ cudaCommonType& fetch_q(){ return q; }
  __host__ __device__ cudaCommonType& fetch_u(){ return u[0]; }
  __host__ __device__ cudaCommonType& fetch_v(){ return u[1]; }
  __host__ __device__ cudaCommonType& fetch_w(){ return u[2]; }
  __host__ __device__ cudaCommonType& fetch_t(){ return t; }
  __host__ __device__ void set_x(cudaCommonType in){ x[0]=in; }
  __host__ __device__ void set_y(cudaCommonType in){ x[1]=in; }
  __host__ __device__ void set_z(cudaCommonType in){ x[2]=in; }
  __host__ __device__ void set_u(cudaCommonType in){ u[0]=in; }
  __host__ __device__ void set_v(cudaCommonType in){ u[1]=in; }
  __host__ __device__ void set_w(cudaCommonType in){ u[2]=in; }
  __host__ __device__ void set_to_zero()
  {
    for(int i=0;i<8;i++) u[i]=0;
  }
  __host__ __device__ void set(
    cudaCommonType _u, cudaCommonType _v, cudaCommonType _w, cudaCommonType _q,
    cudaCommonType _x, cudaCommonType _y, cudaCommonType _z, cudaCommonType _t
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
//  cudaCommonType operator[](int i)
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
