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

#ifndef IPIC_DEFS_H
#define IPIC_DEFS_H

#include <cstdint>

typedef std::uint64_t longid;

// comment this out if OpenMP is not installed on your system.
#define USING_OMP

// uncomment the following line to use parallel hdf5
#define USING_PARALLEL_HDF5

// determine the width of the vector unit
//
#if defined(__MIC__)
const int VECBITS = 512;
#elif defined(__AVX__)
const int VECBITS = 256;
#elif defined(__SSE_)
const int VECBITS = 128;
#else
const int VECBITS = 64;
#endif
const int VECBYTES = VECBITS / 8;

// the number of doubles that fill a vector
const int DVECWIDTH = VECBYTES / sizeof(double);
const int SVECWIDTH = VECBYTES / sizeof(float);
// #define SINGLE_PRECISION_PCLS
//
//  single precision does not seem to help on the MIC
typedef double pfloat;
// #ifdef SINGLE_PRECISION_PCLS
//   typedef float pfloat;
//   #ifdef __MIC__
//     #define VECTOR_WIDTH 16
//   #else
//     #define VECTOR_WIDTH 8
//   #endif
// #else
//   #ifdef __MIC__
//     #define VECTOR_WIDTH 8
//   #else
//     #define VECTOR_WIDTH 4
//   #endif
//   typedef double pfloat;
// #endif

#endif // IPIC_DEFS_H
