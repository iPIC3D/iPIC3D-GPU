#ifndef _CUDA_REDUCTION_H_
#define _CUDA_REDUCTION_H_

#include <iostream>
#include "cudaTypeDef.cuh"
#include <assert.h>

namespace cudaReduction
{

// ********************* max reduction *********************

template <typename T, unsigned int blockSize>
__device__ void warpReduceMax(volatile T* sdata, unsigned int tid) {

    if constexpr (blockSize >= 128 && WARP_SIZE == 64) sdata[tid] = max(sdata[tid], sdata[tid + 64]);
    if constexpr (blockSize >= 64) sdata[tid] = max(sdata[tid], sdata[tid + 32]);
    if constexpr (blockSize >= 32) sdata[tid] = max(sdata[tid], sdata[tid + 16]);
    if constexpr (blockSize >= 16) sdata[tid] = max(sdata[tid], sdata[tid + 8]);
    if constexpr (blockSize >= 8) sdata[tid] = max(sdata[tid], sdata[tid + 4]);
    if constexpr (blockSize >= 4) sdata[tid] = max(sdata[tid], sdata[tid + 2]);
    if constexpr (blockSize >= 2) sdata[tid] = max(sdata[tid], sdata[tid + 1]);
}


/**
 * @brief Reduce the input array to the maximum value, the blockSize should be 2^n and < 1024
 * @details The input array is reduced to the maximum value. But each block will have the one maximum value of its area, postprocess is needed.
 * 
 * @param g_idata input array, n is the size of the array
 * @param g_odata output array, the size of the array is GridSize, which means the number of blocks
 * @param n the size of the input array
 */
template <typename T, unsigned int blockSize>
__global__ void reduceMax(T* g_idata, T* g_odata, unsigned int n) {
    extern __shared__ T sdata[];
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * (blockSize * 2) + tid;
    unsigned int gridSize = blockSize * 2 * gridDim.x;

    sdata[tid] = g_idata[i];

    while (i < n) {
        sdata[tid] = max(sdata[tid], g_idata[i]);
        if (i + blockSize < n)
            sdata[tid] = max(sdata[tid], g_idata[i + blockSize]);
        i += gridSize;
    }

    __syncthreads();


    if constexpr (blockSize >= 512) { if (tid < 256) { sdata[tid] = max(sdata[tid], sdata[tid + 256]); } __syncthreads(); }
    if constexpr (blockSize >= 256) { if (tid < 128) { sdata[tid] = max(sdata[tid], sdata[tid + 128]); } __syncthreads(); }
    if constexpr (blockSize >= 128 && WARP_SIZE < 64) { if (tid < 64) { sdata[tid] = max(sdata[tid], sdata[tid + 64]); } __syncthreads(); }
    if (tid < WARP_SIZE) warpReduceMax<T, blockSize>(sdata, tid);


    if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}


/**
 * @brief Reduce the input array to one maximum value, the blockSize should be small enough for 32 threads
 * @details can only be luanched by one block, and 32 threads
 * 
 * @param g_idata input array, n is the size of the array
 * @param g_odata output value, the size is 1
 */
template <typename T>
__global__ void reduceMaxWarp(T* g_idata, T* g_odata, unsigned int n) {

    assert(blockDim.x == WARP_SIZE);
    assert(gridDim.x == 1);

    unsigned int tid = threadIdx.x;

    T maxValue; // the local variable of the thread
    if (tid < n)maxValue = g_idata[tid];
    else maxValue = g_idata[0];

    for(int i = tid + WARP_SIZE; i < n; i += WARP_SIZE){
        maxValue = max(maxValue, g_idata[i]);
    }

    constexpr unsigned int fullMask = 0xffffffff; // HIP requires uint64 mask
    // warp reduction
    for(int offset = WARP_SIZE / 2; offset > 0; offset /= 2){
        const T tempValue = __shfl_down_sync(fullMask, maxValue, offset);
        maxValue = max(maxValue, tempValue);
    }

    if(tid == 0) g_odata[0] = maxValue;
}


// ********************* min reduction *********************

template <typename T, unsigned int blockSize>
__device__ void warpReduceMin(volatile T* sdata, unsigned int tid) {
    if constexpr (blockSize >= 128 && WARP_SIZE == 64) sdata[tid] = min(sdata[tid], sdata[tid + 64]);
    if constexpr (blockSize >= 64) sdata[tid] = min(sdata[tid], sdata[tid + 32]);
    if constexpr (blockSize >= 32) sdata[tid] = min(sdata[tid], sdata[tid + 16]);
    if constexpr (blockSize >= 16) sdata[tid] = min(sdata[tid], sdata[tid + 8]);
    if constexpr (blockSize >= 8) sdata[tid] = min(sdata[tid], sdata[tid + 4]);
    if constexpr (blockSize >= 4) sdata[tid] = min(sdata[tid], sdata[tid + 2]);
    if constexpr (blockSize >= 2) sdata[tid] = min(sdata[tid], sdata[tid + 1]);
}

template <typename T, unsigned int blockSize>
__global__ void reduceMin(T* g_idata, T* g_odata, unsigned int n) {
    extern __shared__ T sdata[];
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * (blockSize * 2) + tid;
    unsigned int gridSize = blockSize * 2 * gridDim.x;

    sdata[tid] = g_idata[i];

    while (i < n) {
        sdata[tid] = min(sdata[tid], g_idata[i]);
        if (i + blockSize < n)
            sdata[tid] = min(sdata[tid], g_idata[i + blockSize]);
        i += gridSize;
    }

    __syncthreads();

    if constexpr (blockSize >= 512) { if (tid < 256) { sdata[tid] = min(sdata[tid], sdata[tid + 256]); } __syncthreads(); }
    if constexpr (blockSize >= 256) { if (tid < 128) { sdata[tid] = min(sdata[tid], sdata[tid + 128]); } __syncthreads(); }
    if constexpr (blockSize >= 128 && WARP_SIZE < 64) { if (tid < 64) { sdata[tid] = min(sdata[tid], sdata[tid + 64]); } __syncthreads(); }
    if (tid < WARP_SIZE) warpReduceMin<T, blockSize>(sdata, tid);

    if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}

template <typename T>
__global__ void reduceMinWarp(T* g_idata, T* g_odata, unsigned int n) {

    assert(blockDim.x == WARP_SIZE);
    assert(gridDim.x == 1);

    unsigned int tid = threadIdx.x;

    T minValue; // the local variable of the thread
    if (tid < n) minValue = g_idata[tid];
    else minValue = g_idata[0];

    for(int i = tid + WARP_SIZE; i < n; i += WARP_SIZE){
        minValue = min(minValue, g_idata[i]);
    }

    constexpr unsigned int fullMask = 0xffffffff;
    // warp reduction
    for(int offset = WARP_SIZE / 2; offset > 0; offset /= 2){
        const T tempValue = __shfl_down_sync(fullMask, minValue, offset);
        minValue = min(minValue, tempValue);
    }

    if(tid == 0) g_odata[0] = minValue;
}


// ********************* sum reduction *********************


template <typename T, unsigned int blockSize>
__device__ void warpReduceSum(volatile T* sdata, unsigned int tid) {
    if constexpr (blockSize >= 128 && WARP_SIZE == 64) sdata[tid] += sdata[tid + 64];
    if constexpr (blockSize >= 64) sdata[tid] += sdata[tid + 32];
    if constexpr (blockSize >= 32) sdata[tid] += sdata[tid + 16];
    if constexpr (blockSize >= 16) sdata[tid] += sdata[tid + 8];
    if constexpr (blockSize >= 8) sdata[tid] += sdata[tid + 4];
    if constexpr (blockSize >= 4) sdata[tid] += sdata[tid + 2];
    if constexpr (blockSize >= 2) sdata[tid] += sdata[tid + 1];
}


/**
 * @brief Reduce the input array to the sum, the blockSize should be 2^n and < 1024
 * @details The input array is reduced to the sum. But each block will have one sum value of its area, postprocess is needed.
 * 
 * @param g_idata input array, n is the size of the array
 * @param g_odata output array, the size of the array is GridSize, which means the number of blocks
 * @param n the size of the input array
 */
template <typename T, unsigned int blockSize>
__global__ void reduceSum(T* g_idata, T* g_odata, unsigned int n) {
    extern __shared__ T sdata[];
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * (blockSize * 2) + tid;
    unsigned int gridSize = blockSize * 2 * gridDim.x;

    sdata[tid] = 0;

    while (i < n) {
        sdata[tid] += g_idata[i];
        if (i + blockSize < n)
            sdata[tid] += g_idata[i + blockSize];
        i += gridSize;
    }

    __syncthreads();

    if constexpr (blockSize >= 512) { if (tid < 256) { sdata[tid] += sdata[tid + 256]; } __syncthreads(); }
    if constexpr (blockSize >= 256) { if (tid < 128) { sdata[tid] += sdata[tid + 128]; } __syncthreads(); }
    if constexpr (blockSize >= 128 && WARP_SIZE < 64) { if (tid < 64) { sdata[tid] += sdata[tid + 64]; } __syncthreads(); }
    if (tid < WARP_SIZE) warpReduceSum<T, blockSize>(sdata, tid);

    if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}


/**
 * @brief Reduce the input array to one sum value, the blockSize should be small enough for 32 threads
 * @details can only be luanched by one block, and 32 threads
 * 
 * @param g_idata input array, n is the size of the array
 * @param g_odata output value, the size is 1
 */
template <typename T>
__global__ void reduceSumWarp(T* g_idata, T* g_odata, unsigned int n) {

    assert(blockDim.x == WARP_SIZE);
    assert(gridDim.x == 1);

    unsigned int tid = threadIdx.x;

    T sumValue = 0; // the local variable of the thread
    if (tid < n)sumValue = g_idata[tid];

    for(int i = tid + WARP_SIZE; i < n; i += WARP_SIZE){
        sumValue += g_idata[i];
    }

    constexpr unsigned int fullMask = 0xffffffff;
    // warp reduction
    for(int offset = WARP_SIZE / 2; offset > 0; offset /= 2){
        sumValue += __shfl_down_sync(fullMask, sumValue, offset);
    }

    if(tid == 0) g_odata[0] = sumValue;

}


// ********************* pre and post process reduction *********************


enum class PreProcessType
{   
    none,
    minusConstThenEXP,
    multiply,
    multiplyEXP,
};

enum class PostProcessType
{
    logAdd,
    divide,
    divideEXP
};


// pre and post process reduction, it can be more elegent and flexible with cuda extended lambda

template <typename T, PreProcessType preProc, typename U>
__device__ __inline__ T preProcess(T value, U* preProcOprand, int eid) {
    if constexpr (preProc == PreProcessType::minusConstThenEXP) {
        return exp(value - *preProcOprand);
    } else if constexpr (preProc == PreProcessType::multiply) {
        return value * preProcOprand[eid];
    } else if constexpr (preProc == PreProcessType::multiplyEXP) {
        return value * exp(preProcOprand[eid]);
    } else {
        return value;
    }
}

template <typename T, PostProcessType postProc, typename U>
__device__ __inline__ T postProcess(T value, U* postProcOprand) {
    if constexpr (postProc == PostProcessType::logAdd) {
        return log(value) + *postProcOprand;
    } else if constexpr (postProc == PostProcessType::divide) {
        return value / *postProcOprand;
    } else if constexpr (postProc == PostProcessType::divideEXP) {
        return value / exp(*postProcOprand);
    }
}

/**
 * @brief with preprocess and weight
 */
template <typename T, unsigned int blockSize, PreProcessType preProc, typename U, typename V = int, bool ifWeight = false>
__global__ void reduceSumPreProcess(T* g_idata, T* g_odata, unsigned int n, U* preProcOprand, V* weight = nullptr) {
    extern __shared__ T sdata[];
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * (blockSize * 2) + tid;
    unsigned int gridSize = blockSize * 2 * gridDim.x;

    sdata[tid] = 0;

    while (i < n) {
        if constexpr (ifWeight) { // weighted
            sdata[tid] += preProcess<T, preProc, U>(g_idata[i], preProcOprand, i) * weight[i];
            if (i + blockSize < n)
                sdata[tid] += preProcess<T, preProc, U>(g_idata[i + blockSize], preProcOprand, i + blockSize) * weight[i + blockSize];
        } else { // not weighted
            sdata[tid] += preProcess<T, preProc, U>(g_idata[i], preProcOprand, i);
            if (i + blockSize < n)
                sdata[tid] += preProcess<T, preProc, U>(g_idata[i + blockSize], preProcOprand, i + blockSize);
        }
        i += gridSize;
    }

    __syncthreads();

    if constexpr (blockSize >= 512) { if (tid < 256) { sdata[tid] += sdata[tid + 256]; } __syncthreads(); }
    if constexpr (blockSize >= 256) { if (tid < 128) { sdata[tid] += sdata[tid + 128]; } __syncthreads(); }
    if constexpr (blockSize >= 128 && WARP_SIZE < 64) { if (tid < 64) { sdata[tid] += sdata[tid + 64]; } __syncthreads(); }
    if (tid < WARP_SIZE) warpReduceSum<T, blockSize>(sdata, tid);

    if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}


template <typename T, PostProcessType postProc, typename U>
__global__ void reduceSumWarpPostProcess(T* g_idata, T* g_odata, unsigned int n, U* postProcOprand) {

    assert(blockDim.x == WARP_SIZE);
    assert(gridDim.x == 1);

    unsigned int tid = threadIdx.x;

    T sumValue = 0; // the local variable of the thread
    if (tid < n)sumValue = g_idata[tid];

    for(int i = tid + WARP_SIZE; i < n; i += WARP_SIZE){
        sumValue += g_idata[i];
    }

    constexpr unsigned int fullMask = 0xffffffff;
    // warp reduction
    for(int offset = WARP_SIZE / 2; offset > 0; offset /= 2){
        sumValue += __shfl_down_sync(fullMask, sumValue, offset);
    }

    if(tid == 0) g_odata[0] = postProcess<T, postProc, U>(sumValue, postProcOprand);
}

} // namespace cudaReduction


#endif // _CUDA_GMM_REDUCTION_H_