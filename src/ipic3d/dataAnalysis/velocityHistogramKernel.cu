
#include "cudaTypeDef.cuh"
#include "velocityHistogram.cuh"
#include "particleArraySoACUDA.cuh"

namespace velocityHistogram
{

using namespace particleArraySoA;


__global__ void velocityHistogramKernel(const int nop, const histogramTypeIn* d1, const histogramTypeIn* d2, const histogramTypeIn* q,
                                        velocityHistogramCUDA* histogramCUDAPtr){

    int pidx = threadIdx.x + blockIdx.x * blockDim.x;
    int gridSize = gridDim.x * blockDim.x;
    
    
    extern __shared__ histogramTypeOut sHistogram[];
    auto gHistogram = histogramCUDAPtr[0].getHistogramCUDA();

    constexpr int tile = VELOCITY_HISTOGRAM_TILE;
    const auto tileSize = tile * tile;
    // const int sharedMemSize = tileSize * sizeof(histogramTypeOut);

    histogramTypeIn d1d2[2];
    histogramTypeOut qAbs;

    // size of this histogram
    const auto dim0Size = histogramCUDAPtr[0].getSize(0);
    const auto dim1Size = histogramCUDAPtr[0].getSize(1);

    // the dim sizes are multiply of tile
    for(int dim0 = 0; dim0 < dim0Size; dim0+=tile)
    for(int dim1 = 0; dim1 < dim1Size; dim1+=tile){
        // Initialize shared memory to zero
        for (int i = threadIdx.x; i < tileSize; i += blockDim.x) {
            sHistogram[i] = 0.0;
        }
        __syncthreads();
        
        for(int i = pidx; i < nop; i += gridSize){
            
            d1d2[0] = d1[i];
            d1d2[1] = d2[i];

            const auto index = histogramCUDAPtr[0].getIndex(d1d2);
            if(index < 0)continue; // out of histogram range

            const auto index0 = index % dim0Size;
            const auto index1 = index / dim0Size;
            if(index0 < dim0 || index0 >= dim0 + tile || index1 < dim1 || index1 >= dim1 + tile)continue; // not this block
            const auto sIndex = (index0 - dim0) + (index1 - dim1) * tile;

            qAbs = abs(q[i] * 10e5); 
            atomicAdd(&sHistogram[sIndex], qAbs);
        }

        __syncthreads();

        for(int i = threadIdx.x; i < tileSize; i += blockDim.x){
            atomicAdd(&gHistogram[dim0 + i % tile + (dim1 + i / tile) * dim0Size], sHistogram[i]);
        }
        
    }


}

/**
 * @brief reset and calculate the center of each histogram bin
 * @details this kernel is launched once for each histogram bin for all 3 histograms
 */
__global__ void resetBinScaleMarkKernel(velocityHistogramCUDA* histogramCUDAPtr){
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if(idx >= histogramCUDAPtr->getLogicSize())return;

    
    histogramCUDAPtr[0].getHistogramCUDA()[idx] = 0.0; histogramCUDAPtr[0].centerOfBin(idx);
    histogramCUDAPtr[1].getHistogramCUDA()[idx] = 0.0; histogramCUDAPtr[1].centerOfBin(idx);
    histogramCUDAPtr[2].getHistogramCUDA()[idx] = 0.0; histogramCUDAPtr[2].centerOfBin(idx);

}



} // namespace velocityHistogram







