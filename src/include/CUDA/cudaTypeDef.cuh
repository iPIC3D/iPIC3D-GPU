#ifndef _CUDA_TYPE_DEF_H_
#define _CUDA_TYPE_DEF_H_

#ifndef HIPIFLY
#include <cuda.h>
#include <cuda_runtime.h>
#include "cuda_fp16.h"
#define WARP_SIZE (32)
inline constexpr uint32_t WARP_FULL_MASK = 0xFFFFFFFF;
using warp_mask_t = uint32_t;
#else
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include "hipifly.hpp"
#define WARP_SIZE (64)
inline constexpr uint64_t WARP_FULL_MASK = 0xFFFFFFFFFFFFFFFF;
using warp_mask_t = uint64_t;
#endif

#include <iostream>
#include <sstream>

// ── Portable warp intrinsic helpers (defined once, used everywhere) ──
// Only available in device compilation units (.cu files compiled by nvcc/hipcc)
#if defined(__CUDACC__) || defined(__HIPCC__)

// Portable popcount: 32-bit on CUDA, 64-bit on HIP
__device__ __forceinline__ int warp_popcount(warp_mask_t mask) {
#ifndef HIPIFLY
    return __popc(mask);
#else
    return __popcll(mask);
#endif
}

// Portable warp-wide sum reduction (works for any WARP_SIZE)
template <typename T>
__device__ __forceinline__ T warp_reduce_sum(T val) {
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2)
        val += __shfl_down_sync(WARP_FULL_MASK, val, offset);
    return val;
}

#endif // __CUDACC__ || __HIPCC__

using cudaTypeSingle = float;
using cudaTypeDouble = double;

using cudaTypeHalf = __half;

using cudaCommonType = cudaTypeDouble;
using cudaParticleType = cudaTypeDouble;
using cudaFieldType = cudaTypeDouble; // type for the field array from host to device
using cudaMomentType = cudaTypeDouble; // MUST be DOUBLE now, type for the moment array from device to host

// ── Per-field particle types (all default to double; change individually for mixed precision) ──
using cudaPclType_U = cudaTypeDouble;  // velocity x
using cudaPclType_V = cudaTypeDouble;  // velocity y
using cudaPclType_W = cudaTypeDouble;  // velocity z
using cudaPclType_Q = cudaTypeDouble;  // charge
using cudaPclType_X = cudaTypeDouble;  // position x
using cudaPclType_Y = cudaTypeDouble;  // position y
using cudaPclType_Z = cudaTypeDouble;  // position z
using cudaPclType_T = cudaTypeDouble;  // subcycle time / particle ID

template <class T, int dim2, int dim3, int dim4>
using cudaTypeArray4 = T (*)[dim2][dim3][dim4];

template <class T, int dim2, int dim3>
using cudaTypeArray3 = T (*)[dim2][dim3];

template <class T,  int dim2>
using cudaTypeArray2 = T (*)[dim2];

template <class T>
using cudaTypeArray1 = T *;

/////////////////////////////////// CUDA API HOST call wrapper

#define ERROR_CHECK_C_LIKE false


#define cudaErrChk(call) cudaCheck((call), __FILE__, __LINE__)

__host__ inline void cudaCheck(cudaError_t code, const char *file, int line)
{
    if (code != cudaSuccess)
    {
#if ERROR_CHECK_C_LIKE == true
        std::cerr << "CUDA Check: " << cudaGetErrorString(code) << " File: " << file << " Line: " << line << std::endl;
        abort();
#else
        std::ostringstream oss;
        oss << "CUDA Check: " << cudaGetErrorString(code) << " File: " << file << " Line: " << line;
        throw std::runtime_error(oss.str());
#endif
    }
}
#undef ERROR_CHECK_C_LIKE

/////////////////////////////////// CUDA data alignment

#ifndef HIPIFLY

#if defined(__CUDACC__) // NVCC
    #define CUDA_ALIGN(n) __align__(n)
#else
    #define CUDA_ALIGN(n) 
#endif

#else

#if defined(__HIPCC__) // HIPCC
    #define CUDA_ALIGN(n) __align__(n)
#else
    #define CUDA_ALIGN(n)
#endif

#endif


/////////////////////////////////// CUDA type copy to device

template <typename T>
__host__ inline T* copyToDevice(T* objectOnHost, cudaStream_t stream = 0){
    if(objectOnHost == nullptr)throw std::runtime_error("CopyToDevice: can not copy a nullptr to device.");
    T* ptr = nullptr;
    cudaErrChk(cudaMalloc(&ptr, sizeof(T)));
    cudaErrChk(cudaMemcpyAsync(ptr, objectOnHost, sizeof(T), cudaMemcpyDefault, stream));

    cudaErrChk(cudaStreamSynchronize(stream));
    return ptr;
}

template <typename T>
__host__ inline T* copyArrayToDevice(T* objectOnHost, int numberOfElement, cudaStream_t stream = 0){
    if(objectOnHost == nullptr)throw std::runtime_error("CopyToDevice: can not copy a nullptr to device.");
    T* ptr = nullptr;
    cudaErrChk(cudaMalloc(&ptr, numberOfElement * sizeof(T)));
    cudaErrChk(cudaMemcpyAsync(ptr, objectOnHost, numberOfElement * sizeof(T), cudaMemcpyDefault, stream));

    cudaErrChk(cudaStreamSynchronize(stream));
    return ptr;
}

////////////////////////////////// One dimenstion to high dim index

/**
 * @brief Convert a 3D index tuple to a flat row-major index.
 * @param dim1 Extent of the first dimension.
 * @param dim2 Extent of the second dimension.
 * @param dim3 Extent of the third dimension.
 * @param index1 Index along the first dimension.
 * @param index2 Index along the second dimension.
 * @param index3 Index along the third dimension.
 * @return Flattened row-major index.
 */
__host__ __device__ inline uint32_t toOneDimIndex(uint32_t dim1, uint32_t dim2, uint32_t dim3,
                                     uint32_t index1, uint32_t index2, uint32_t index3){
    return (index1*dim2*dim3 + index2*dim3 + index3);
}

/**
 * @brief Convert a 4D index tuple to a flat row-major index.
 * @param dim1 Extent of the first dimension.
 * @param dim2 Extent of the second dimension.
 * @param dim3 Extent of the third dimension.
 * @param dim4 Extent of the fourth dimension.
 * @param index1 Index along the first dimension.
 * @param index2 Index along the second dimension.
 * @param index3 Index along the third dimension.
 * @param index4 Index along the fourth dimension.
 * @return Flattened row-major index.
 */
__host__ __device__ inline uint32_t toOneDimIndex(uint32_t dim1, uint32_t dim2, uint32_t dim3, uint32_t dim4,
                                         uint32_t index1, uint32_t index2, uint32_t index3, uint32_t index4){
    return (index1*dim2*dim3*dim4 + index2*dim3*dim4 + index3*dim4 + index4);
}

////////////////////////////////// Pinned memory allocation

__host__ inline void* allocateHostPinnedMem(size_t typeSize, size_t num){
    void* ptr = nullptr;
    cudaErrChk(cudaHostAlloc(&ptr, typeSize*num, cudaHostAllocDefault));
    return ptr;
}

template <typename T, typename... Args>
T* newHostPinnedObject(Args... args){
    T* ptr = (T*)allocateHostPinnedMem(sizeof(T), 1);
    return new(ptr) T(std::forward<Args>(args)...);
}

template <typename T, typename... Args>
T* newHostPinnedObjectArray(size_t num, Args... args){
    T* ptr = (T*)allocateHostPinnedMem(sizeof(T), num);
    for(size_t i = 0; i < num; i++){
        new(ptr + i) T(std::forward<Args>(args)...);
    }
    return ptr;
}

template <typename T>
void deleteHostPinnedObject(T* ptr){
    ptr->~T();
    cudaErrChk(cudaFreeHost(ptr));
}

template <typename T>
void deleteHostPinnedObjectArray(T* ptr, size_t num){
    for(size_t i = 0; i < num; i++){
        (ptr + i)->~T();
    }
    cudaErrChk(cudaFreeHost(ptr));
}

////////////////////////////////// Round up to
template <typename T>
__host__ __device__ inline T getGridSize(T threadNum, T blockSize) {
    return ((threadNum + blockSize - 1) / blockSize);
}

#endif
