#ifndef VELOCITY_HISTOGRAM_CUH
#define VELOCITY_HISTOGRAM_CUH

#include "cudaTypeDef.cuh"

#include "Particle.h"
#include "particleArrayCUDA.cuh"
#include "particleArraySoAView.cuh"

#include <fstream>
#include <iostream>
#include <type_traits>

#include "dataAnalysisConfig.cuh"

namespace histogram {

template <typename U, int dim, typename T = int> class histogramCUDA {

private:
  T* hostPtr;
  T* cudaPtr;

  U* scaleMark[dim];

  int bufferSize; // the physical size of current buffer, in elements
public:
  int size[dim]; // the logic size of each dimension, in elements
private:
  int logicSize; // the logic size of the whole histogram

  U min[dim], max[dim], resolution[dim];

public:
  /**
   * @param bufferSize the physical size of the buffer, in elements
   */
  __host__ histogramCUDA(int bufferSize) : bufferSize(bufferSize) {
    allocate();
  }

  /**
   * @param min the minimum value of each dimension
   * @param max the maximum value of each dimension
   * @param resolution the resolution of each dimension
   */
  __host__ void setHistogram(U* min, U* max, int* binThisDim) {

    for (int i = 0; i < dim; i++) {
      if (min[i] >= max[i] || binThisDim[i] <= 0) {
        std::cerr << "[!]Invalid histogram range or binThisDim" << std::endl;
        std::cerr << "[!]min: " << min[i] << " max: " << max[i]
                  << " binThisDim: " << binThisDim[i] << std::endl;
        return;
      }
    }

    logicSize = 1;
    for (int i = 0; i < dim; i++) {
      this->min[i] = min[i];
      this->max[i] = max[i];

      size[i] = binThisDim[i];
      this->resolution[i] = (max[i] - min[i]) / size[i];
      logicSize *= size[i];
    }

    if (bufferSize < logicSize) {
      cudaErrChk(cudaFreeHost(hostPtr));
      cudaErrChk(cudaFree(cudaPtr));
      bufferSize = logicSize;
      allocate();
    }
  }

  __host__ void copyHistogramAsync(cudaStream_t stream = 0) {
    cudaErrChk(cudaMemcpyAsync(hostPtr, cudaPtr, logicSize * sizeof(T),
                               cudaMemcpyDeviceToHost, stream));
  }

  __host__ T* getHistogram() { return hostPtr; }

  __host__ __device__ T* getHistogramCUDA() { return cudaPtr; }

  __host__ U** getScaleMarkCUDAPtrs() { return scaleMark; }

  __host__ __device__ int getLogicSize() { return logicSize; }

  __host__ void getSize(int* size) {
    for (int i = 0; i < dim; i++) {
      size[i] = this->size[i];
    }
  }

  __host__ __device__ int getSize(int index) { return size[index]; }

  __host__ U getMin(int index) { return min[index]; }

  __host__ U getMax(int index) { return max[index]; }

  __host__ U getResolution(int index) { return resolution[index]; }

  __device__ int getIndex(const U* data) {
    int index = 0;

    for (int i = dim - 1; i >= 0; i--) {
      // check the range
      if (data[i] < min[i] || data[i] > max[i]) {
        return -1;
      }

      auto tmp = (int)((data[i] - min[i]) / resolution[i]);
      if (tmp == size[i])
        tmp--; // the max value
      index += tmp;
      if (i != 0)
        index *= size[i - 1];
    }

    if (index >= logicSize)
      return -1;
    return index;
  }

  /**
   * @brief get the index of the bin, in the buffer
   * @param data the data to be histogramed, dim elements
   * @param tile the start index of the tile, dim elements
   * @param tileSize the size of the tile, dim elements
   *
   * @return the index of the bin, in Tile
   */
  __device__ int getIndexTiled(const U* data, const int* tile,
                               const int* tileSize) {
    int index = 0;

    for (int i = dim - 1; i >= 0; i--) {
      // check the range
      if (data[i] < min[i] || data[i] > max[i]) {
        return -1;
      }

      auto tmp =
          (int)((data[i] - min[i]) /
                resolution[i]); // the index in the whole histogram dimension
      if (tmp == size[i])
        tmp--; // the max value

      if (tmp < tile[i] || tmp >= tile[i] + tileSize[i])
        return -1; // out of the tile range

      index += tmp - tile[i];
      if (i != 0)
        index *= tileSize[i - 1];
    }

    auto tileBufferSize = 1;
    for (int i = 0; i < dim; i++) {
      tileBufferSize *= tileSize[i];
    }

    if (index >= tileBufferSize)
      return -1;
    return index;
  }

  /**
   * @brief get the center of the bin, for the scale mark, for GMM
   * @param index the index of the bin, in the buffer
   */
  __device__ void centerOfBin(int index) {
    int tmp = index;
    for (int i = 0; i < dim; i++) {
      scaleMark[i][index] = min[i] + (tmp % size[i] + 0.5) * resolution[i];
      tmp /= size[i];
    }
  }

private:
  __host__ void allocate() {
    cudaErrChk(cudaMallocHost((void**)&hostPtr, bufferSize * sizeof(T)));
    cudaErrChk(cudaMalloc((void**)&cudaPtr, bufferSize * sizeof(T)));

    for (int i = 0; i < dim; i++) {
      cudaErrChk(cudaMalloc((void**)&scaleMark[i], bufferSize * sizeof(U)));
    }
  }

public:
  __host__ ~histogramCUDA() {
    cudaErrChk(cudaFreeHost(hostPtr));
    cudaErrChk(cudaFree(cudaPtr));

    for (int i = 0; i < dim; i++) {
      cudaErrChk(cudaFree(scaleMark[i]));
    }
  }
};

} // namespace histogram

namespace velocityHistogram {

using histogramTypeIn = cudaParticleType;
using histogramTypeOut = cudaTypeSingle;

using velocityHistogramCUDA2D =
    histogram::histogramCUDA<histogramTypeIn, 2, histogramTypeOut>;
using velocityHistogramCUDA3D =
    histogram::histogramCUDA<histogramTypeIn, 3, histogramTypeOut>;

using velocitySoA = particleArraySoAView<histogramTypeIn, 4>;

using namespace DAConfig;

/**
 * @brief Histogram for one species
 */

class velocityHistogram3D {
private:
  // UVW
  velocityHistogramCUDA3D* histogramHostPtr;
  velocityHistogramCUDA3D* histogramCUDAPtr;

  int binThisDim[3] = {VELOCITY_HISTOGRAM3D_RES_1, VELOCITY_HISTOGRAM3D_RES_2,
                       VELOCITY_HISTOGRAM3D_RES_3};

  int reductionTempArraySize = 0;
  histogramTypeIn* reductionTempArrayCUDA;
  histogramTypeIn* reductionMinResultCUDA;
  histogramTypeIn* reductionMaxResultCUDA;

  histogramTypeIn minArray[3];
  histogramTypeIn maxArray[3];

  bool bigEndian;

  int reduceBlockNum(int dataSize, int blockSize) {
    constexpr int elementsPerThread = 128;
    if (dataSize < elementsPerThread)
      dataSize = elementsPerThread;
    auto blockNum = getGridSize(dataSize / elementsPerThread,
                                blockSize); // 4096 elements per thread
    blockNum = blockNum > 1024 ? 1024 : blockNum;

    if (reductionTempArraySize < blockNum) {
      cudaErrChk(cudaFree(reductionTempArrayCUDA));
      cudaErrChk(cudaMalloc(&reductionTempArrayCUDA,
                            sizeof(histogramTypeIn) * blockNum * 6));
      reductionTempArraySize = blockNum;
    }

    return blockNum;
  }

  /**
   * @brief get the Max and Min value of the given value set
   * @param pclArray Borrowed velocity SoA view for one species.
   * @param species Species index, used for output metadata.
   * @param stream CUDA stream used by the reduction kernels.
   * @return `0` on success.
   */
  int getRange(velocitySoA* pclArray, const int species, cudaStream_t stream);

public:
  /**
   * @brief Construct a velocity histogram accumulator.
   * @param initSize the initial size of the histogram buffer, in elements
   */
  velocityHistogram3D(int initSize) {

    const auto bufferSize = binThisDim[0] * binThisDim[1] * binThisDim[2];

    if (initSize < bufferSize) {
      std::cerr << "[!]Histogram initial size is too small: " << initSize
                << " vs " << binThisDim[0] << "x" << binThisDim[1] << std::endl;
      initSize = bufferSize;
    }

    histogramHostPtr = newHostPinnedObject<velocityHistogramCUDA3D>(initSize);
    cudaErrChk(cudaMalloc(&histogramCUDAPtr, sizeof(velocityHistogramCUDA3D)));

    if constexpr (HISTOGRAM_FIXED_RANGE == false) {
      reductionTempArraySize = 1024;
      cudaErrChk(
          cudaMalloc(&reductionTempArrayCUDA,
                     sizeof(histogramTypeIn) * reductionTempArraySize * 6));
      cudaErrChk(
          cudaMalloc(&reductionMinResultCUDA, sizeof(histogramTypeIn) * 6));
      reductionMaxResultCUDA = reductionMinResultCUDA + 3;
    }

    { // check the endian
      int test = 1;
      char* ptr = reinterpret_cast<char*>(&test);
      if (*ptr == 1) {
        bigEndian = false;
      } else {
        bigEndian = true;
      }
    }
  }

  /**
   * @brief Initiate the kernels for histograming, launch the kernels
   * @details It can be invoked after Moment in the main loop, for the output
   * and solver are on CPU
   * @param pclArray Borrowed velocity SoA view for one species.
   * @param species Species index, used for output naming and metadata.
   * @param stream CUDA stream used by the histogram kernels.
   */
  void init(velocitySoA* pclArray, const int species, cudaStream_t stream = 0);

  /**
   * @brief Wait for the histogram data to be ready, copy the data to host
   * @details It should be invoked after a previous Init, after this, can use
   * getVelocityHistogramCUDAArray to get the data writeToFile has the same
   * effect
   * @param stream CUDA stream used for the pending device-to-host copy.
   */
  void copyHistogramToHost(cudaStream_t stream = 0) {
    histogramHostPtr->copyHistogramAsync(stream);
    cudaErrChk(cudaStreamSynchronize(stream));
  }

  void writeToFile(std::string filePath, int cycleNum,
                   cudaStream_t stream = 0) {
    copyHistogramToHost(stream);

    std::string vtkType;
    if constexpr (std::is_same_v<histogramTypeOut, float>) {
      vtkType = "float";
    } else if constexpr (std::is_same_v<histogramTypeOut, double>) {
      vtkType = "double";
    } else if constexpr (std::is_same_v<histogramTypeOut, int>) {
      vtkType = "int";
    } else {
      throw std::runtime_error("Unsupported histogramTypeOut");
    }

    std::ostringstream ossFileName;
    ossFileName << filePath << "3D_" << cycleNum << ".vtk";

    std::ofstream vtkFile(ossFileName.str(), std::ios::binary);

    vtkFile << "# vtk DataFile Version 3.0\n";
    vtkFile << "Velocity Histogram\n";
    vtkFile << "BINARY\n";
    vtkFile << "DATASET STRUCTURED_POINTS\n";
    vtkFile << "DIMENSIONS " << histogramHostPtr->size[0] << " "
            << histogramHostPtr->size[1] << " " << histogramHostPtr->size[2]
            << "\n";
    vtkFile << "ORIGIN " << histogramHostPtr->getMin(0) << " "
            << histogramHostPtr->getMin(1) << " " << histogramHostPtr->getMin(2)
            << "\n";
    vtkFile << "SPACING " << histogramHostPtr->getResolution(0) << " "
            << histogramHostPtr->getResolution(1) << " "
            << histogramHostPtr->getResolution(2) << "\n";
    vtkFile << "POINT_DATA " << histogramHostPtr->getLogicSize() << "\n";
    vtkFile << "SCALARS scalars " << vtkType << " 1\n";
    vtkFile << "LOOKUP_TABLE default\n";

    auto histogramBuffer = histogramHostPtr->getHistogram();
    for (int j = 0; j < histogramHostPtr->getLogicSize(); j++) {
      histogramTypeOut value = histogramBuffer[j];

      if constexpr (sizeof(histogramTypeOut) == 4) {
        if (!bigEndian)
          *(uint32_t*)(&value) = __builtin_bswap32(*(uint32_t*)(&value));
      } else if constexpr (sizeof(histogramTypeOut) == 8) {
        if (!bigEndian)
          *(uint64_t*)(&value) = __builtin_bswap64(*(uint64_t*)(&value));
      }

      vtkFile.write(reinterpret_cast<char*>(&value), sizeof(histogramTypeOut));
    }

    vtkFile.close();
  }

  histogramTypeOut* getVelocityHistogramHostPtr() {
    return histogramHostPtr->getHistogram();
  }

  histogramTypeOut* getVelocityHistogramCUDAArray() {
    return histogramHostPtr->getHistogramCUDA();
  }

  histogramTypeIn** getHistogramScaleMark() {
    return histogramHostPtr->getScaleMarkCUDAPtrs();
  }

  ~velocityHistogram3D() {
    if constexpr (HISTOGRAM_FIXED_RANGE == false) {
      cudaErrChk(cudaFree(reductionTempArrayCUDA));
      cudaErrChk(cudaFree(reductionMinResultCUDA));
    }

    cudaErrChk(cudaFree(histogramCUDAPtr));
    deleteHostPinnedObject(histogramHostPtr);
  }
};

} // namespace velocityHistogram

#endif // VELOCITY_HISTOGRAM_CUH
