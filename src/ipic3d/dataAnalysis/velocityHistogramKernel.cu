
#include "cudaTypeDef.cuh"
#include "velocityHistogram.cuh"

namespace velocityHistogram
{


__global__ void histogramKernel3D(const int nop, const histogramTypeIn *d1, const histogramTypeIn *d2, const histogramTypeIn *d3, 
                                    const histogramTypeIn *q,
                                    velocityHistogramCUDA3D *histogramCUDAPtr)
{

    int pidx = threadIdx.x + blockIdx.x * blockDim.x;
    int gridSize = gridDim.x * blockDim.x;

    extern __shared__ histogramTypeOut sHistogram[];
    auto gHistogram = histogramCUDAPtr[0].getHistogramCUDA();

    constexpr int tile = VELOCITY_HISTOGRAM3D_TILE;
    constexpr auto tileSize = tile * tile * tile;

    histogramTypeIn data[3];
    int dim[3];
    int dimSize[3] = {tile, tile, tile};
    histogramTypeOut qAbs;

    // size of this histogram
    const auto dim0Size = histogramCUDAPtr[0].getSize(0);
    const auto dim1Size = histogramCUDAPtr[0].getSize(1);
    const auto dim2Size = histogramCUDAPtr[0].getSize(2);

    // the dim sizes are multiply of tile
    for (int dim0 = 0; dim0 < dim0Size; dim0 += tile) // unroll if const size
    for (int dim1 = 0; dim1 < dim1Size; dim1 += tile)
    for (int dim2 = 0; dim2 < dim2Size; dim2 += tile)
    {
        dim[0] = dim0;
        dim[1] = dim1;
        dim[2] = dim2;

        // Initialize shared memory to zero
        for (int i = threadIdx.x; i < tileSize; i += blockDim.x)
        {
            sHistogram[i] = 0.0;
        }
        __syncthreads();

        for (int i = pidx; i < nop; i += gridSize)
        {
            data[0] = d1[i];
            data[1] = d2[i];
            data[2] = d3[i];
            const auto sIndex = histogramCUDAPtr[0].getIndexTiled(data, dim, dimSize);
            if (sIndex < 0) continue; // out of tile range

            qAbs = abs(q[i] * 10e6);
            atomicAdd(&sHistogram[sIndex], qAbs);
        }

        __syncthreads();

        for (int i = threadIdx.x; i < tileSize; i += blockDim.x)
        {
            atomicAdd(&gHistogram[dim0 + i % tile + (dim1 + i / tile % tile) * dim0Size + (dim2 + i / tile / tile) * dim0Size * dim1Size], sHistogram[i]);
        }
    }
}

/**
 * @brief reset and calculate the center of each histogram bin
 * @details this kernel is launched once for each histogram bin for all 3 histograms
 */
__global__ void resetBin(velocityHistogramCUDA3D* histogramCUDAPtr){
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int gridSize = gridDim.x * blockDim.x;

    auto histogram = histogramCUDAPtr->getHistogramCUDA();
    const auto histogramSize = histogramCUDAPtr->getLogicSize();

    for (int i = idx; i < histogramSize; i += gridSize){
        histogram[i] = 0.0;
        histogramCUDAPtr->centerOfBin(i);
    }
    
}


} // namespace velocityHistogram







