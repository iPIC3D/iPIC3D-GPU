/**
 * @file GPUFieldArray.cu
 * @brief Device kernels for GPUFieldArray utility operations.
 */

#include "GPUFieldArray.cuh"

// ---------------------------------------------------------------------------
//  Fill kernel: set every element to 'val'
// ---------------------------------------------------------------------------
__global__ void gpuFieldFillKernel(double* __restrict__ arr, double val, size_t n)
{
    size_t idx = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (idx < n) arr[idx] = val;
}

void GPUFieldArray3::fillKernelLaunch(double* d, double val, size_t n, cudaStream_t stream)
{
    const int blockSize = 256;
    const size_t gridSize = (n + blockSize - 1) / blockSize;
    gpuFieldFillKernel<<<gridSize, blockSize, 0, stream>>>(d, val, n);
    cudaErrChk(cudaGetLastError());
}
