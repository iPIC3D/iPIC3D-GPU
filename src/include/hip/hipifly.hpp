#ifndef _HIPIFLY_HPP_
#define _HIPIFLY_HPP_


// Error Handling
#define cudaError_t hipError_t
#define cudaSuccess hipSuccess
#define cudaGetErrorString hipGetErrorString
#define cudaGetLastError hipGetLastError

// Initialization and Device Management
#define cudaDeviceSynchronize hipDeviceSynchronize
#define cudaDeviceReset hipDeviceReset
#define cudaDeviceGetAttribute hipDeviceGetAttribute
#define cudaSetDevice hipSetDevice
#define cudaGetDevice hipGetDevice
#define cudaGetDeviceCount hipGetDeviceCount

// Memory Management
#define cudaMalloc hipMalloc
#define cudaFree hipFree
#define cudaMemcpy hipMemcpy
#define cudaMemcpyToSymbol hipMemcpyToSymbol
#define cudaMemcpyFromSymbol hipMemcpyFromSymbol
#define cudaMemcpyAsync hipMemcpyAsync
#define cudaMemset hipMemset
#define cudaHostAlloc hipHostMalloc
#define cudaFreeHost hipHostFree
#define cudaHostAllocDefault hipHostMallocDefault
#define cudaMallocAsync hipMallocAsync
#define cudaHostRegisterDefault hipHostRegisterDefault
#define cudaHostRegister hipHostRegister
#define cudaHostUnregister hipHostUnregister
#define cudaMemsetAsync hipMemsetAsync
#define cudaFreeAsync hipFreeAsync
#define cudaMallocHost hipHostMalloc // hipMallocHost is deprecated, and there is no cudaHostMalloc but cudaHostAlloc

// Memory Query
#define cudaMemGetInfo hipMemGetInfo

// Memory Types
#define cudaMemcpyHostToDevice hipMemcpyHostToDevice
#define cudaMemcpyDeviceToHost hipMemcpyDeviceToHost
#define cudaMemcpyDeviceToDevice hipMemcpyDeviceToDevice
#define cudaMemcpyDefault hipMemcpyDefault

// Stream Management
#define cudaStream_t hipStream_t
#define cudaStreamCreate hipStreamCreate
#define cudaStreamCreateWithFlags hipStreamCreateWithFlags
#define cudaStreamDestroy hipStreamDestroy
#define cudaStreamSynchronize hipStreamSynchronize
#define cudaStreamWaitEvent hipStreamWaitEvent
#define cudaStreamNonBlocking hipStreamNonBlocking

// Event Management
#define cudaEvent_t hipEvent_t
#define cudaEventCreate hipEventCreate
#define cudaEventDestroy hipEventDestroy
#define cudaEventRecord hipEventRecord
#define cudaEventSynchronize hipEventSynchronize
#define cudaEventElapsedTime hipEventElapsedTime
#define cudaEventDisableTiming hipEventDisableTiming
#define cudaEventCreateWithFlags hipEventCreateWithFlags
#define cudaEventBlockingSync hipEventBlockingSync

// Texture and Surface References
#define cudaTextureObject_t hipTextureObject_t
#define cudaCreateTextureObject hipCreateTextureObject
#define cudaDestroyTextureObject hipDestroyTextureObject
#define cudaResourceDesc hipResourceDesc
#define cudaTextureDesc hipTextureDesc

// Unified Memory Management
#define cudaMallocManaged hipMallocManaged
#define cudaMemPrefetchAsync hipMemPrefetchAsync

// Cooperative Groups
#define cudaLaunchCooperativeKernel hipLaunchCooperativeKernel

// Kernel Launch Configuration
// Note: hipLaunchKernel has the same signature as cudaLaunchKernel.
// hipLaunchKernelGGL is a different HIP-specific API — do NOT use it here.
#define cudaLaunchKernel hipLaunchKernel


// Warp primitives
// HIP __shfl_down / __shfl do not take a mask argument; drop it.
#define __shfl_down_sync(x, y, z) __shfl_down(y, z)
#define __shfl_sync(x, y, z) __shfl(y, z)



#endif