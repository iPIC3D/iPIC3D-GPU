
#include <string>
#include <memory>
#include <random>

#include "dataAnalysis.cuh"
#include "dataAnalysisConfig.cuh"
#include "particleArraySoACUDA.cuh"
#include "velocityHistogram.cuh"


//using velocitySoA = particleArraySoA::particleArraySoACUDA<cudaCommonType, 0, 3>;
using namespace velocityHistogram;
using SoAElements = particleArraySoA::particleArraySoAElement;

constexpr int nop = 5000000;



int main(){
    int histogramSize = VELOCITY_HISTOGRAM_RES * VELOCITY_HISTOGRAM_RES;

    velocitySoA pclArray(nop, 0);
    velocityHistogram::velocityHistogram histogram(histogramSize);


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

    auto uPtr = pclArray.getElement(SoAElements::U);
    auto vPtr = pclArray.getElement(SoAElements::V);
    auto wPtr = pclArray.getElement(SoAElements::W);
    auto qPtr = pclArray.getElement(SoAElements::Q);

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

    // CPU histogram

    std::vector<cudaCommonType> cpuHistUV(histogramSize, 0);
    std::vector<cudaCommonType> cpuHistVW(histogramSize, 0);
    std::vector<cudaCommonType> cpuHistUW(histogramSize, 0);

    cudaCommonType minVal = MIN_VELOCITY_HIST_E;
    cudaCommonType maxVal = MAX_VELOCITY_HIST_E;
    cudaCommonType resolution = (maxVal - minVal) / VELOCITY_HISTOGRAM_RES;

    for (int i = 0; i < nop; i++){
        cudaCommonType uVal = uCPU[i];
        cudaCommonType vVal = vCPU[i];
        if(uVal >= minVal && uVal <= maxVal && vVal >= minVal && vVal <= maxVal){
            int binU = static_cast<int>((uVal - minVal) / resolution);
            if(binU >= VELOCITY_HISTOGRAM_RES) binU = VELOCITY_HISTOGRAM_RES - 1;
            int binV = static_cast<int>((vVal - minVal) / resolution);
            if(binV >= VELOCITY_HISTOGRAM_RES) binV = VELOCITY_HISTOGRAM_RES - 1;
            int index = binV * VELOCITY_HISTOGRAM_RES + binU;
            cpuHistUV[index] += std::fabs(qCPU[i] * 1e6);
        }

        cudaCommonType wVal = wCPU[i];
        if(vVal >= minVal && vVal <= maxVal && wVal >= minVal && wVal <= maxVal){
            int binV = static_cast<int>((vVal - minVal) / resolution);
            if(binV >= VELOCITY_HISTOGRAM_RES) binV = VELOCITY_HISTOGRAM_RES - 1;
            int binW = static_cast<int>((wVal - minVal) / resolution);
            if(binW >= VELOCITY_HISTOGRAM_RES) binW = VELOCITY_HISTOGRAM_RES - 1;
            int index = binW * VELOCITY_HISTOGRAM_RES + binV;
            cpuHistVW[index] += std::fabs(qCPU[i] * 1e6);
        }

        if(uVal >= minVal && uVal <= maxVal && wVal >= minVal && wVal <= maxVal){
            int binU = static_cast<int>((uVal - minVal) / resolution);
            if(binU >= VELOCITY_HISTOGRAM_RES) binU = VELOCITY_HISTOGRAM_RES - 1;
            int binW = static_cast<int>((wVal - minVal) / resolution);
            if(binW >= VELOCITY_HISTOGRAM_RES) binW = VELOCITY_HISTOGRAM_RES - 1;
            int index = binW * VELOCITY_HISTOGRAM_RES + binU;
            cpuHistUW[index] += std::fabs(qCPU[i] * 1e6);
        }
    }

    // GPU histogram
    histogram.init(&pclArray, 0, 0, 0);
    cudaErrChk(cudaDeviceSynchronize());
    histogram.copyHistogramToHost();

    auto histogramHostPtrUV = histogram.getVelocityHistogramHostPtr(0);
    auto histogramHostPtrVW = histogram.getVelocityHistogramHostPtr(1);
    auto histogramHostPtrUW = histogram.getVelocityHistogramHostPtr(2);

    // compare the results
    bool pass = true;
    cudaCommonType tolerance = 1e-6;

    for (int i = 0; i < histogramSize; i++){
        if (std::fabs(histogramHostPtrUV[i] - cpuHistUV[i]) > tolerance){
            std::cout << "Mismatch in UV histogram at bin " << i 
                      << ": GPU = " << histogramHostPtrUV[i] 
                      << ", CPU = " << cpuHistUV[i] << "\n";
            pass = false;
            break;
        }
        if (std::fabs(histogramHostPtrVW[i] - cpuHistVW[i]) > tolerance){
            std::cout << "Mismatch in VW histogram at bin " << i 
                      << ": GPU = " << histogramHostPtrVW[i] 
                      << ", CPU = " << cpuHistVW[i] << "\n";
            pass = false;
            break;
        }
        if (std::fabs(histogramHostPtrUW[i] - cpuHistUW[i]) > tolerance){
            std::cout << "Mismatch in UW histogram at bin " << i 
                      << ": GPU = " << histogramHostPtrUW[i] 
                      << ", CPU = " << cpuHistUW[i] << "\n";
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









