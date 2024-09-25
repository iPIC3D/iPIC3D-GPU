#ifndef _CUDA_TYPE_DEF_H_
#define _CUDA_TYPE_DEF_H_

#include <cuda.h>
#include "cuda_fp16.h"
#include <iostream>
#include <sstream>

using cudaTypeSingle = float;
using cudaTypeDouble = double;

using cudaTypeHalf = __half;

using cudaCommonType = cudaTypeDouble;

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


#if defined(__CUDACC__) // NVCC
    #define CUDA_ALIGN(n) __align__(n)
#else
    #define CUDA_ALIGN(n) 
#endif

/////////////////////////////////// CUDA type copy to device

template <typename T>
__host__ inline T* copyToDevice(T* objectOnHost, cudaStream_t stream = 0){
    if(objectOnHost == nullptr)throw std::runtime_error("CopyToDevice: can not copy a nullptr to device.");
    T* ptr = nullptr;
    cudaErrChk(cudaMallocAsync(&ptr, sizeof(T), stream));
    cudaErrChk(cudaMemcpyAsync(ptr, objectOnHost, sizeof(T), cudaMemcpyDefault, stream));

    cudaErrChk(cudaStreamSynchronize(stream));
    return ptr;
}

template <typename T>
__host__ inline T* copyArrayToDevice(T* objectOnHost, int numberOfElement, cudaStream_t stream = 0){
    if(objectOnHost == nullptr)throw std::runtime_error("CopyToDevice: can not copy a nullptr to device.");
    T* ptr = nullptr;
    cudaErrChk(cudaMallocAsync(&ptr, numberOfElement * sizeof(T), stream));
    cudaErrChk(cudaMemcpyAsync(ptr, objectOnHost, numberOfElement * sizeof(T), cudaMemcpyDefault, stream));

    cudaErrChk(cudaStreamSynchronize(stream));
    return ptr;
}

////////////////////////////////// One dimenstion to high dim index

/**
 * @brief turn 3-dim index to one dim index
 */
__host__ __device__ inline uint32_t toOneDimIndex(uint32_t dim1, uint32_t dim2, uint32_t dim3,
                                     uint32_t index1, uint32_t index2, uint32_t index3){
    return (index1*dim2*dim3 + index2*dim3 + index3);
}

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