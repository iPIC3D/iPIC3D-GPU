
#include <cstdlib>
#include <exception>
#include <iostream>
#include <string>
#include <memory>
#include <random>

#include "dataAnalysis.cuh"
#include "dataAnalysisConfig.cuh"
#include "particleArraySoAView.cuh"
#include "velocityHistogram.cuh"


using namespace velocityHistogram;

using namespace DAConfig;

constexpr int nop = 5000000;



static int runTest(){
    int histogramSize = VELOCITY_HISTOGRAM3D_SIZE;

    velocitySoA pclArray(nop, 0);
    velocityHistogram::velocityHistogram3D histogram(histogramSize);


    // fill the pclArray with random data
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<cudaCommonType> dis(-1.0, 1.0);
    auto center = MIN_VELOCITY_HIST_E + MAX_VELOCITY_HIST_E;
    auto left = center - MIN_VELOCITY_HIST_E;
    auto right = MAX_VELOCITY_HIST_E - center;
    std::normal_distribution<cudaCommonType> normalDist1(center, right * 0.2);
    std::normal_distribution<cudaCommonType> normalDist2(center + right * 0.4, right * 0.1);

    auto uCPU = new cudaCommonType[nop];
    auto vCPU = new cudaCommonType[nop];
    auto wCPU = new cudaCommonType[nop];
    auto qCPU = new cudaCommonType[nop];

    auto uPtr = pclArray.getElement(0);  // U
    auto vPtr = pclArray.getElement(1);  // V
    auto wPtr = pclArray.getElement(2);  // W
    auto qPtr = pclArray.getElement(3);  // Q

    for(int i = 0; i < nop; i++){
        uCPU[i] = dis(gen) > 0.0 ? normalDist1(gen) : normalDist2(gen);
        vCPU[i] = dis(gen) > 0.0 ? normalDist1(gen) : normalDist2(gen);
        wCPU[i] = dis(gen) > 0.0 ? normalDist1(gen) : normalDist2(gen);
        qCPU[i] = dis(gen) / 1e10;
    }

    cudaErrChk(cudaMemcpy(uPtr, uCPU, nop * sizeof(cudaCommonType), cudaMemcpyHostToDevice));
    cudaErrChk(cudaMemcpy(vPtr, vCPU, nop * sizeof(cudaCommonType), cudaMemcpyHostToDevice));
    cudaErrChk(cudaMemcpy(wPtr, wCPU, nop * sizeof(cudaCommonType), cudaMemcpyHostToDevice));
    cudaErrChk(cudaMemcpy(qPtr, qCPU, nop * sizeof(cudaCommonType), cudaMemcpyHostToDevice));

    // CPU 3D histogram

    std::vector<cudaCommonType> cpuHist(histogramSize, 0);


    cudaCommonType minVal = MIN_VELOCITY_HIST_E;
    cudaCommonType maxVal = MAX_VELOCITY_HIST_E;
    cudaCommonType resolution1 = (maxVal - minVal) / VELOCITY_HISTOGRAM3D_RES_1;
    cudaCommonType resolution2 = (maxVal - minVal) / VELOCITY_HISTOGRAM3D_RES_2;
    cudaCommonType resolution3 = (maxVal - minVal) / VELOCITY_HISTOGRAM3D_RES_3;

    for (int i = 0; i < nop; i++){
        cudaCommonType uVal = uCPU[i];
        cudaCommonType vVal = vCPU[i];
        cudaCommonType wVal = wCPU[i];

        if(uVal >= minVal && uVal <= maxVal && vVal >= minVal && vVal <= maxVal && wVal >= minVal && wVal <= maxVal){
            int bin1 = static_cast<int>((uVal - minVal) / resolution1);
            if(bin1 >= VELOCITY_HISTOGRAM3D_RES_1) bin1 = VELOCITY_HISTOGRAM3D_RES_1 - 1;
            int bin2 = static_cast<int>((vVal - minVal) / resolution2);
            if(bin2 >= VELOCITY_HISTOGRAM3D_RES_2) bin2 = VELOCITY_HISTOGRAM3D_RES_2 - 1;
            int bin3 = static_cast<int>((wVal - minVal) / resolution3);
            if(bin3 >= VELOCITY_HISTOGRAM3D_RES_3) bin3 = VELOCITY_HISTOGRAM3D_RES_3 - 1;

            cpuHist[bin1 + bin2 * VELOCITY_HISTOGRAM3D_RES_1 + bin3 * VELOCITY_HISTOGRAM3D_RES_1 * VELOCITY_HISTOGRAM3D_RES_2] += std::abs(qCPU[i] * 1e7); // 10e6 in the kernel 
        }

    }

    // GPU histogram
    histogram.init(&pclArray, 0, 0);
    cudaErrChk(cudaDeviceSynchronize());
    histogram.copyHistogramToHost();

    auto histogramHostPtr = histogram.getVelocityHistogramHostPtr();

    // compare the results
    bool pass = true;
    cudaCommonType tolerance = 1e-6;

    for (int i = 0; i < histogramSize; i++){
        if (std::fabs(histogramHostPtr[i] - cpuHist[i]) > tolerance){
            std::cout << "Mismatch in UV histogram at bin " << i 
                      << ": GPU = " << histogramHostPtr[i] 
                      << ", CPU = " << cpuHist[i] << "\n";
            pass = false;
            break;
        }
    }

    if(pass){
        std::cout << "Test passed: CPU and GPU histograms match.\n";
    } else {
        std::cout << "Test failed: CPU and GPU histograms do not match.\n";
    }

    delete[] uCPU;
    delete[] vCPU;
    delete[] wCPU;
    delete[] qCPU;

    return !pass;

}

int main(){
    try {
        return runTest();
    } catch (const std::exception& error) {
        std::cerr << "histogram3DTest failed: " << error.what() << "\n";
        return EXIT_FAILURE;
    }
}






