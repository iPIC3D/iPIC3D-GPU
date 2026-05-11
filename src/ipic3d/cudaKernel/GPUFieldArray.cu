/**
 * @file GPUFieldArray.cu
 * @brief Device kernels for GPUFieldArray utility operations.
 */

#include "GPUFieldArray.cuh"

// ---------------------------------------------------------------------------
//  Fill kernel: set every element of array to 'val'
// ---------------------------------------------------------------------------
template<class T>
__global__ void gpuFieldFillKernel(T* __restrict__ arr, T val, size_t n)
{
    size_t idx = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (idx < n) arr[idx] = val;
}

template<class T>
void GPUFieldArray3Impl<T>::fillKernelLaunch(T* d, T val, size_t n, cudaStream_t stream)
{
    const int blockSize = 256;
    const size_t gridSize = (n + blockSize - 1) / blockSize;
    gpuFieldFillKernel<T><<<gridSize, blockSize, 0, stream>>>(d, val, n);
    cudaErrChk(cudaGetLastError());
}

// ---------------------------------------------------------------------------
//  Explicit instantiations for supported element types.
//  Extend this list when new solver types are introduced.
// ---------------------------------------------------------------------------
template class GPUFieldArray3Impl<double>;
template class GPUFieldArray3Impl<float>;
