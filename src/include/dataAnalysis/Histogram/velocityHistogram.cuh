#ifndef _VELOCITY_HISTOGRAM_
#define _VELOCITY_HISTOGRAM_

#include "cudaTypeDef.cuh"


#include "Particle.h"
#include "particleArrayCUDA.cuh"
#include "particleArraySoACUDA.cuh"

#include <iostream>
#include <fstream>
#include <type_traits>

#include "dataAnalysisConfig.cuh"


namespace histogram{

template <typename U, int dim, typename T = int>
class histogramCUDA {

private:
    T* hostPtr;
    T* cudaPtr;

    U* scaleMark[dim];

    int bufferSize; // the physical size of current buffer, in elements
public:
    int size[dim];  // the logic size of each dimension, in elements
private:
    int logicSize;  // the logic size of the whole histogram

    U min[dim], max[dim], resolution[dim];

public:

    /**
     * @param bufferSize the physical size of the buffer, in elements
     */
    __host__ histogramCUDA(int bufferSize): bufferSize(bufferSize){
        allocate();
    }

    /**
     * @param min the minimum value of each dimension
     * @param max the maximum value of each dimension
     * @param resolution the resolution of each dimension
     */
    __host__ void setHistogram(U* min, U* max, int* binThisDim){
        
        for(int i=0; i<dim; i++){
            if(min[i] >= max[i] || binThisDim[i] <= 0){
                std::cerr << "[!]Invalid histogram range or binThisDim" << std::endl;
                std::cerr << "[!]min: " << min[i] << " max: " << max[i] << " binThisDim: " << binThisDim[i] << std::endl;
                return;
            }
        }

        logicSize = 1;
        for(int i=0; i<dim; i++){
            this->min[i] = min[i];
            this->max[i] = max[i];

            size[i] = binThisDim[i];
            this->resolution[i] = (max[i] - min[i]) / size[i];
            logicSize *= size[i];
        }

        if(bufferSize < logicSize){
            cudaErrChk(cudaFreeHost(hostPtr));
            cudaErrChk(cudaFree(cudaPtr));
            bufferSize = logicSize;
            allocate();
        }
        
    }

    __device__ void setHistogramDevice(U* min, U* max, int* binThisDim){
        
        for(int i=0; i<dim; i++){
            if(min[i] >= max[i] || binThisDim[i] <= 0){
                assert(0);
                return;
            }
        }

        logicSize = 1;
        for(int i=0; i<dim; i++){
            this->min[i] = min[i];
            this->max[i] = max[i];

            size[i] = binThisDim[i];
            this->resolution[i] = (max[i] - min[i]) / size[i];
            logicSize *= size[i];
        }

        if(bufferSize < logicSize){
            assert(0);
        }
        
    }

    __host__ void copyHistogramAsync(cudaStream_t stream = 0){
        cudaErrChk(cudaMemcpyAsync(hostPtr, cudaPtr, logicSize * sizeof(T), cudaMemcpyDeviceToHost, stream));
    }

    __host__ T* getHistogram(){
        return hostPtr;
    }

    __host__ __device__ T* getHistogramCUDA(){
        return cudaPtr;
    }

    __host__ U** getScaleMarkCUDAPtrs(){
        return scaleMark;
    }

    __host__ __device__ int getLogicSize(){
        return logicSize;
    }

    __host__ void getSize(int* size){
        for(int i=0; i<dim; i++){
            size[i] = this->size[i];
        }
    }

    __host__ __device__ int getSize(int index){
        return size[index];
    }

    __host__ U getMin(int index){
        return min[index];
    }

    __host__ U getMax(int index){
        return max[index];
    }

    __host__ U getResolution(int index){
        return resolution[index];
    }
    
    __device__ void addData(const U* data, const T count = 1){
        int index = 0;
        
        for(int i=dim-1; i>=0; i--){
            // check the range
            if(data[i] < min[i] || data[i] > max[i]){assert(0); return;}

            auto tmp = (int)((data[i] - min[i]) / resolution[i]);
            if(tmp == size[i])tmp--; // the max value
            index +=  tmp;
            if(i != 0)index *= size[i-1];
        }

        if(index >= logicSize)return;
        atomicAdd(&cudaPtr[index], count);
    }


    __device__ int getIndex(const U* data){
        int index = 0;
        
        for(int i=dim-1; i>=0; i--){
            // check the range
            if(data[i] < min[i] || data[i] > max[i]){return -1;}

            auto tmp = (int)((data[i] - min[i]) / resolution[i]);
            if(tmp == size[i])tmp--; // the max value
            index +=  tmp;
            if(i != 0)index *= size[i-1];
        }

        if(index >= logicSize)return -1;
        return index;
    }

    /**
     * @brief get the center of the bin
     * @param index the index of the bin, in the buffer
     */
    __device__ void centerOfBin(int index){
        int tmp = index;
        for(int i=0; i<dim; i++){
            scaleMark[i][index] = min[i] + (tmp % size[i] + 0.5) * resolution[i];
            tmp /= size[i];
        }
    }
private:

    __host__ void allocate(){
        cudaErrChk(cudaMallocHost((void**)&hostPtr, bufferSize * sizeof(T)));
        cudaErrChk(cudaMalloc((void**)&cudaPtr, bufferSize * sizeof(T)));

        for(int i=0; i<dim; i++){
            cudaErrChk(cudaMalloc((void**)&scaleMark[i], bufferSize * sizeof(U)));
        }
    }


public:

    __host__ ~histogramCUDA(){
        cudaErrChk(cudaFreeHost(hostPtr));
        cudaErrChk(cudaFree(cudaPtr));

        for(int i=0; i<dim; i++){
            cudaErrChk(cudaFree(scaleMark[i]));
        }
    }

};

}


namespace velocityHistogram{

using histogramTypeIn = cudaParticleType;
using histogramTypeOut = cudaTypeSingle;

using velocityHistogramCUDA = histogram::histogramCUDA<histogramTypeIn, 2, histogramTypeOut>;

using velocitySoA = particleArraySoA::particleArraySoACUDA<histogramTypeIn, 0, 3>;


/**
 * @brief Histogram for one species
 */
class velocityHistogram
{
private:
    // UVW
    velocityHistogramCUDA* histogramHostPtr;

    velocityHistogramCUDA* histogramCUDAPtr; // one buffer for 3 objects

    int binThisDim[2] = {VELOCITY_HISTOGRAM_RES, VELOCITY_HISTOGRAM_RES};

    int reductionTempArraySize = 0;
    histogramTypeIn* reductionTempArrayCUDA;
    histogramTypeIn* reductionMinResultCUDA;
    histogramTypeIn* reductionMaxResultCUDA;

    int cycleNum;


    bool bigEndian;

    int reduceBlockNum(int dataSize, int blockSize){
        constexpr int elementsPerThread = 128;
        if(dataSize < elementsPerThread)dataSize = elementsPerThread;
        auto blockNum = getGridSize(dataSize / elementsPerThread, blockSize); // 4096 elements per thread
        blockNum = blockNum > 1024 ? 1024 : blockNum;

        if(reductionTempArraySize < blockNum){
            cudaErrChk(cudaFree(reductionTempArrayCUDA));
            cudaErrChk(cudaMalloc(&reductionTempArrayCUDA, sizeof(histogramTypeIn)*blockNum * 6));
            reductionTempArraySize = blockNum;
        }

        return blockNum;
    }


    /**
     * @brief get the Max and Min value of the given value set
     */
    __host__ int getRange(velocitySoA* pclArray, const int species, cudaStream_t stream);

public:

    /**
     * @param initSize the initial size of the histogram buffer, in elements
     * @param path the path to store the output file, directory
     */
    __host__ velocityHistogram(int initSize) {
        // reduction temporary array
        reductionTempArraySize = 1024;
        cudaErrChk(cudaMalloc(&reductionTempArrayCUDA, sizeof(histogramTypeIn)*reductionTempArraySize * 6));

        // the range of the histogram
        cudaErrChk(cudaMalloc(&reductionMinResultCUDA, sizeof(histogramTypeIn)*6));
        reductionMaxResultCUDA = reductionMinResultCUDA + 3;

        if(initSize < binThisDim[0] * binThisDim[1]) {
            // throw std::runtime_error("Histogram initial size is too small");
            std::cerr << "[!]Histogram initial size is too small: " << initSize << " vs " << binThisDim[0] << "x" << binThisDim[1] << std::endl;
            initSize = binThisDim[0] * binThisDim[1];
        }
        histogramHostPtr = newHostPinnedObjectArray<velocityHistogramCUDA>(3, initSize);
        cudaErrChk(cudaMalloc(&histogramCUDAPtr, sizeof(velocityHistogramCUDA) * 3));
        cudaErrChk(cudaMemcpy(histogramCUDAPtr, histogramHostPtr, sizeof(velocityHistogramCUDA) * 3, cudaMemcpyDefault));
        
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
     * @details It can be invoked after Moment in the main loop, for the output and solver are on CPU
     */
    __host__ void init(velocitySoA* pclArray, int cycleNum, const int species, cudaStream_t stream = 0);

    /**
     * @brief Wait for the histogram data to be ready, copy the data to host
     * @details It should be invoked after a previous Init, after this, can use getVelocityHistogramCUDAArray to get the data
     *         writeToFile has the same effect
     */
    __host__ void copyHistogramToHost(cudaStream_t stream = 0){
        cudaErrChk(cudaStreamSynchronize(stream));
        
        for(int i=0; i<3; i++){
            histogramHostPtr[i].copyHistogramAsync(stream);
        }

        cudaErrChk(cudaStreamSynchronize(stream));
    }


    __host__ void writeToFile(std::string filePath, cudaStream_t stream = 0){
        copyHistogramToHost(stream);
        
        std::string items[3] = {"uv", "vw", "uw"};

        std::string vtkType;
        if constexpr (std::is_same_v<histogramTypeOut, float>){
            vtkType = "float";
        } else if constexpr (std::is_same_v<histogramTypeOut, double>){
            vtkType = "double";
        } else if constexpr (std::is_same_v<histogramTypeOut, int>){
            vtkType = "int";
        } else {
            throw std::runtime_error("Unsupported histogramTypeOut");
        }

        for(int i=0; i<3; i++){ // UV, VW, UW
            std::ostringstream ossFileName;
            ossFileName << filePath << items[i] << "_" << cycleNum << ".vtk";

            std::ofstream vtkFile(ossFileName.str(), std::ios::binary);

            vtkFile << "# vtk DataFile Version 3.0\n";
            vtkFile << "Velocity Histogram\n";
            vtkFile << "BINARY\n";  
            vtkFile << "DATASET STRUCTURED_POINTS\n";

            if constexpr (HISTOGRAM_OUTPUT_3D == true){

                switch (i)
                {
                case 0: // UV
                    vtkFile << "DIMENSIONS " << histogramHostPtr[i].size[0] << " " << histogramHostPtr[i].size[1] << " 1\n";
                    vtkFile << "ORIGIN " << histogramHostPtr[i].getMin(0) << " " << histogramHostPtr[i].getMin(1) << " 0\n"; 
                    vtkFile << "SPACING " << histogramHostPtr[i].getResolution(0) << " " << histogramHostPtr[i].getResolution(1) << " 1\n";  
                    break;

                case 1: // VW
                    vtkFile << "DIMENSIONS " << "1 " << histogramHostPtr[i].size[0] << " " << histogramHostPtr[i].size[1] << "\n";
                    vtkFile << "ORIGIN " << "0 " << histogramHostPtr[i].getMin(0) << " " << histogramHostPtr[i].getMin(1) << "\n";
                    vtkFile << "SPACING " << "1 " << histogramHostPtr[i].getResolution(0) << " " << histogramHostPtr[i].getResolution(1) << "\n";
                    break;

                case 2: // UW
                    vtkFile << "DIMENSIONS " << histogramHostPtr[i].size[0] << " " << "1 " << histogramHostPtr[i].size[1] << "\n";
                    vtkFile << "ORIGIN " << histogramHostPtr[i].getMin(0) << " " << "0 " << histogramHostPtr[i].getMin(1) << "\n";
                    vtkFile << "SPACING " << histogramHostPtr[i].getResolution(0) << " " << "1 " << histogramHostPtr[i].getResolution(1) << "\n";
                    break;
                
                default:
                    break;
                }

            } else {
                vtkFile << "DIMENSIONS " << histogramHostPtr[i].size[0] << " " << histogramHostPtr[i].size[1] << " 1\n";
                vtkFile << "ORIGIN " << histogramHostPtr[i].getMin(0) << " " << histogramHostPtr[i].getMin(1) << " 0\n"; 
                vtkFile << "SPACING " << histogramHostPtr[i].getResolution(0) << " " << histogramHostPtr[i].getResolution(1) << " 1\n";
            }


            vtkFile << "POINT_DATA " << histogramHostPtr[i].getLogicSize() << "\n";  
            vtkFile << "SCALARS scalars " << vtkType << " 1\n";  
            vtkFile << "LOOKUP_TABLE default\n";  

            auto histogramBuffer = histogramHostPtr[i].getHistogram();
            for (int j = 0; j < histogramHostPtr[i].getLogicSize(); j++) {
                histogramTypeOut value = histogramBuffer[j];

                if constexpr (sizeof(histogramTypeOut) == 4){
                    if(!bigEndian)*(uint32_t*)(&value) = __builtin_bswap32(*(uint32_t*)(&value));
                } else if constexpr (sizeof(histogramTypeOut) == 8){
                    if(!bigEndian)*(uint64_t*)(&value) = __builtin_bswap64(*(uint64_t*)(&value));
                }

                vtkFile.write(reinterpret_cast<char*>(&value), sizeof(histogramTypeOut));
            }

            vtkFile.close();
        }


    }

    __host__ histogramTypeOut* getVelocityHistogramHostPtr(const int i){
        return histogramHostPtr[i].getHistogram();
    }

    __host__ histogramTypeOut* getVelocityHistogramCUDAArray(const int i){
        return histogramHostPtr[i].getHistogramCUDA();
    }

    __host__ histogramTypeIn** getHistogramScaleMark(const int i){
        return histogramHostPtr[i].getScaleMarkCUDAPtrs();
    }

    ~velocityHistogram(){

        cudaErrChk(cudaFree(reductionTempArrayCUDA));
        cudaErrChk(cudaFree(reductionMinResultCUDA));

        cudaErrChk(cudaFree(histogramCUDAPtr));
        deleteHostPinnedObjectArray(histogramHostPtr, 3);
    }
};





    
}






#endif