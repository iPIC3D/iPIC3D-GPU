#include <iostream>
#include <vector>
#include <cstdlib>
#include <exception>
#include "cudaTypeDef.cuh"
#include "cudaReduction.cuh"

using dataType = float;

std::vector<dataType> generateDataset(size_t size) {
    std::vector<dataType> data(size);
    srand(static_cast<unsigned>(time(0)));
    for (size_t i = 0; i < size; ++i) {
        data[i] = static_cast<dataType>(rand()) / RAND_MAX * 100.0f; // Random floats in range [0, 100)
    }
    return data;
}

// CPU reduction for max
dataType cpuReduceMax(const std::vector<dataType>& data) {
    dataType maxVal = data[0];
    for (dataType val : data) {
        if (val > maxVal) maxVal = val;
    }
    return maxVal;
}

// CPU reduction for min
dataType cpuReduceMin(const std::vector<dataType>& data) {
    dataType minVal = data[0];
    for (dataType val : data) {
        if (val < minVal) minVal = val;
    }
    return minVal;
}



static int runTest(){

    const size_t size = 1 << 24;
    constexpr int blockSize = 256;
    constexpr int blockNum = 512;

    const std::vector<dataType> data = generateDataset(size);

    dataType result;

    dataType *deviceInput, *deviceOutput; dataType* reductionTemp;
    cudaErrChk(cudaMalloc(&deviceInput, size * sizeof(dataType)));
    cudaErrChk(cudaMalloc(&deviceOutput, sizeof(dataType)));
    cudaErrChk(cudaMemcpy(deviceInput, data.data(), size * sizeof(dataType), cudaMemcpyHostToDevice));

    cudaErrChk(cudaMalloc(&reductionTemp, blockNum * sizeof(dataType)));

    // max
    cudaReduction::reduceMax<dataType, blockSize><<<blockNum, blockSize, blockSize * sizeof(dataType)>>>(deviceInput, reductionTemp, size);
    cudaReduction::reduceMaxWarp<dataType><<<1, WARP_SIZE>>>(reductionTemp, deviceOutput, blockNum);
    cudaErrChk(cudaMemcpy(&result, deviceOutput, sizeof(dataType), cudaMemcpyDeviceToHost));
    // cpu reduction
    dataType cpuMax = cpuReduceMax(data);

    // check result
    if (result == cpuMax) {
        std::cout << "Max reduction test passed!" << std::endl;
    } else {
        std::cout << "Max reduction test failed!" << std::endl;
        return -1;
    }



    // min
    cudaReduction::reduceMin<dataType, blockSize><<<blockNum, blockSize, blockSize * sizeof(dataType)>>>(deviceInput, reductionTemp, size);
    cudaReduction::reduceMinWarp<dataType><<<1, WARP_SIZE>>>(reductionTemp, deviceOutput, blockNum);
    cudaErrChk(cudaMemcpy(&result, deviceOutput, sizeof(dataType), cudaMemcpyDeviceToHost));

    // cpu reduction
    dataType cpuMin = cpuReduceMin(data);

    // check result
    if (result == cpuMin) {
        std::cout << "Min reduction test passed!" << std::endl;
    } else {
        std::cout << "Min reduction test failed!" << std::endl;
        return -1;
    }
    
    return 0;
}

int main(){
    try {
        return runTest();
    } catch (const std::exception& error) {
        std::cerr << "reductionTest failed: " << error.what() << std::endl;
        return EXIT_FAILURE;
    }
}



