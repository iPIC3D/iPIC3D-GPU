#include "particleArraySoAView.cuh"
#include "velocityHistogram.cuh"

#include "cudaReduction.cuh"
#include "cudaTypeDef.cuh"
#include "dataAnalysisConfig.cuh"
#include <iostream>

namespace velocityHistogram {

__global__ void resetBin(velocityHistogramCUDA3D* histogramCUDAPtr);

__global__ void histogramKernel3D(const int nop, const histogramTypeIn* d1,
                                  const histogramTypeIn* d2,
                                  const histogramTypeIn* d3,
                                  const histogramTypeIn* q,
                                  velocityHistogramCUDA3D* histogramCUDAPtr);

__host__ void velocityHistogram3D::init(velocitySoA* pclArray,
                                        const int species,
                                        cudaStream_t stream) {

  getRange(pclArray, species, stream);
  histogramHostPtr->setHistogram(minArray, maxArray, binThisDim);
  cudaErrChk(cudaMemcpyAsync(histogramCUDAPtr, histogramHostPtr,
                             sizeof(velocityHistogramCUDA3D),
                             cudaMemcpyHostToDevice, stream));

  const int binNum = binThisDim[0] * binThisDim[1] * binThisDim[2];
  resetBin<<<getGridSize(binNum / 8, 256), 256, 0, stream>>>(histogramCUDAPtr);

  // shared memory size
  constexpr int tileSize = VELOCITY_HISTOGRAM3D_TILE *
                           VELOCITY_HISTOGRAM3D_TILE *
                           VELOCITY_HISTOGRAM3D_TILE;
  constexpr int sharedMemSize = sizeof(histogramTypeOut) * tileSize;
  if constexpr (sharedMemSize > 48 * 1024)
    throw std::runtime_error("Shared memory size exceeds the limit ...");
  if (binNum % tileSize != 0)
    throw std::runtime_error(
        "Adjust histogram resolution to multiply of tile ...");

  histogramKernel3D<<<getGridSize((int)pclArray->getNOP() / 128, 512), 512,
                      sharedMemSize, stream>>>(
      pclArray->getNOP(), pclArray->getElement(0), pclArray->getElement(1),
      pclArray->getElement(2), pclArray->getElement(3), histogramCUDAPtr);
}

/**
 * @brief Synchronous function to get the min and max for 3 dimensions, result
 * is stored in minArray and maxArray
 * @param pclArray Borrowed velocity SoA view for one species.
 * @param species Species index, kept for interface symmetry with the caller.
 * @param stream CUDA stream used by the reduction kernels.
 * @return `0` on success.
 */
__host__ int velocityHistogram3D::getRange(velocitySoA* pclArray,
                                           const int species,
                                           cudaStream_t stream) {

  if (HISTOGRAM_FIXED_RANGE == false) {
    using namespace cudaReduction;

    constexpr int blockSize = 256;
    auto blockNum = reduceBlockNum(pclArray->getNOP(), blockSize);

    for (int i = 0; i < 3; i++) { // UVW
      reduceMin<histogramTypeIn, blockSize>
          <<<blockNum, blockSize, blockSize * sizeof(histogramTypeIn),
             stream>>>(pclArray->getElement(i),
                       reductionTempArrayCUDA + i * reductionTempArraySize,
                       pclArray->getNOP());
      reduceMinWarp<histogramTypeIn><<<1, WARP_SIZE, 0, stream>>>(
          reductionTempArrayCUDA + i * reductionTempArraySize,
          reductionMinResultCUDA + i, blockNum);

      reduceMax<histogramTypeIn, blockSize>
          <<<blockNum, blockSize, blockSize * sizeof(histogramTypeIn),
             stream>>>(pclArray->getElement(i),
                       reductionTempArrayCUDA +
                           (i + 3) * reductionTempArraySize,
                       pclArray->getNOP());
      reduceMaxWarp<histogramTypeIn><<<1, WARP_SIZE, 0, stream>>>(
          reductionTempArrayCUDA + (i + 3) * reductionTempArraySize,
          reductionMaxResultCUDA + i, blockNum);
    }
    cudaErrChk(cudaMemcpyAsync(minArray, reductionMinResultCUDA,
                               sizeof(histogramTypeIn) * 3,
                               cudaMemcpyDeviceToHost, stream));
    cudaErrChk(cudaMemcpyAsync(maxArray, reductionMaxResultCUDA,
                               sizeof(histogramTypeIn) * 3,
                               cudaMemcpyDeviceToHost, stream));
    cudaErrChk(cudaStreamSynchronize(stream));

  } else {
    histogramTypeIn min = species == 0 || species == 2 ? MIN_VELOCITY_HIST_E
                                                       : MIN_VELOCITY_HIST_I;
    minArray[0] = min;
    minArray[1] = min;
    minArray[2] = min;
    histogramTypeIn max = species == 0 || species == 2 ? MAX_VELOCITY_HIST_E
                                                       : MAX_VELOCITY_HIST_I;
    maxArray[0] = max;
    maxArray[1] = max;
    maxArray[2] = max;
  }

  return 0;
}

} // namespace velocityHistogram
