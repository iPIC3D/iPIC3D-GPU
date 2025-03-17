#ifndef _CUDA_GMM_UTILITY_CUH_
#define _CUDA_GMM_UTILITY_CUH_

#include <vector>
#include <string>
#include <fstream>
#include <ctime>
#include <initializer_list>
#include <stdexcept>
#include <memory>
#include <sstream>
#include <iomanip>

namespace cudaGMMWeight
{

template <typename T, int dataDim, typename U = int>
class GMMDataMultiDim{

private:
    int dim = dataDim;
    int numData;
    T* data[dataDim]; // pointers to the dimensions of the data points
    U* weight;
public:

    T maxVelocityArray[dataDim]; // the range of the data points

    // all dim in one array
    __host__ GMMDataMultiDim(int numData, T* data, U* weight, std::initializer_list<T> maxVelocityArray){ 
        this->numData = numData;
        for(int i = 0; i < dataDim; i++){
            this->data[i] = data + i*numData;
            this->maxVelocityArray[i] = *(maxVelocityArray.begin() + i);
        }
        this->weight = weight;
    }

    // all dim in separate arrays
    __host__ GMMDataMultiDim(int numData, T** data, U* weight, std::initializer_list<T> maxVelocityArray){
        this->numData = numData;
        for(int i = 0; i < dataDim; i++){
            this->data[i] = data[i];
            this->maxVelocityArray[i] = *(maxVelocityArray.begin() + i);
        }
        this->weight = weight;
    }

    __device__ __host__ T* getDim(int dim)const {
        return data[dim];
    }

    __device__ __host__ int getNumData()const {
        return numData;
    }

    __device__ __host__ int getDim()const {
        return dim;
    }

    __device__ __host__ U* getWeight()const {
        return weight;
    }

};

template <typename T>
struct GMMParam_s{
    int numComponents;
    int maxIteration;
    T threshold; // the threshold for the log likelihood
    // these 3 are optional, if not set, they will be initialized with the internal init functions
    T* weightInit;
    T* meanInit;
    T* coVarianceInit;
    
};

template <typename T>
using GMMParam_t = GMMParam_s<T>;


// store output GMM data of the previous DA cycle to be used as initial GMM parameters in next DA cycle
// at each DA cycle data in GMMParam_output_store are overwritten
template <typename T>
struct GMMParam_output_store{
    int numComponents;
    int maxIteration;
    T threshold; // the threshold for the log likelihood
    T weightVector[NUM_COMPONENT_GMM];
    T meanVector[NUM_COMPONENT_GMM * DATA_DIM_GMM];
    T coVarianceMatrix[NUM_COMPONENT_GMM * DATA_DIM_GMM * DATA_DIM_GMM ];
};


// result class. T is output parameter type, this can not be reused
template <typename T, int dataDim>
class GMMResult{

public:
    int simulationStep;
    int numComponents; // number of components

    std::unique_ptr<T[]> weight; // the weight of each component
    std::unique_ptr<T[]> mean; // the mean of each component
    std::unique_ptr<T[]> coVariance; // the coVariance of each component, full, row major

    int convergeStep; 
    T logLikelihoodFinal; // the final log likelihood
    // std::vector<T> logLikelihoodArray; // th elog lilelihood, in each iteration

private:

    std::string outputString() const {
        std::ostringstream outputStream;
        outputStream << std::fixed << std::setprecision(8);

        outputStream << "{\n";
        outputStream << "\"simulationStep\": " << simulationStep << ",\n";
        outputStream << "\"numComponent\": " << numComponents << ",\n";
        outputStream << "\"dataDim\": " << dataDim << ",\n";
        outputStream << "\"convergeStep\": " << convergeStep << ",\n";
        outputStream << "\"logLikeliHood\": " << logLikelihoodFinal << ",\n";

        outputStream << "\"components\": [\n";
        for (int i = 0; i < numComponents; i++) {
            outputStream << "  {\n";
            outputStream << "    \"weight\": " << weight[i] << ",\n";
            outputStream << "    \"mean\": [";
            for (int j = 0; j < dataDim; j++) {
                outputStream << mean[i * dataDim + j];
                if (j != dataDim - 1) {
                    outputStream << ", ";
                }
            }
            outputStream << "],\n";
            outputStream << "    \"coVariance\": [";
            for (int k = 0; k < dataDim * dataDim; k++) {
                outputStream << coVariance[i * dataDim * dataDim + k];
                if (k != dataDim * dataDim - 1) {
                    outputStream << ", ";
                }
            }
            outputStream << "]\n";
            outputStream << "  }";
            if (i != numComponents - 1) {
                outputStream << ",\n";
            } else {
                outputStream << "\n";
            }
        }
        outputStream << "]\n";
        outputStream << "}";
        
        return outputStream.str();
    }


public:

    GMMResult(const int simuStep, const int numComponents) : simulationStep(simuStep) , numComponents(numComponents){

        weight = std::make_unique<T[]>(numComponents);
        mean = std::make_unique<T[]>(numComponents*dataDim);
        coVariance = std::make_unique<T[]>(numComponents*dataDim*dataDim);

        // logLikelihoodArray = std::vector<T>();
    }

    GMMResult(GMMResult&&) noexcept = default;
    GMMResult& operator=(GMMResult&&) noexcept = default;

    GMMResult(const GMMResult&) = delete;
    GMMResult& operator=(const GMMResult&) = delete;


    static void outputResultArray(const std::vector<GMMResult<T, dataDim>>& resultArray, const std::string outputPath, const std::string metaData = "") {

        // open file or create file
        std::ofstream file(outputPath);
        if(!file.is_open()){
            throw std::runtime_error("GMMResult: Error: can not open file " + outputPath);
        }

        std::string outputString = "{\n";

        // date and time
        time_t now = time(0);
        tm* ltm = localtime(&now);
        outputString += "\"date\": \"" + std::to_string(1900 + ltm->tm_year) + "-" + std::to_string(1 + ltm->tm_mon) + "-" + std::to_string(ltm->tm_mday) + "\",\n";
        outputString += "\"time\": \"" + std::to_string(ltm->tm_hour) + ":" + std::to_string(ltm->tm_min) + ":" + std::to_string(ltm->tm_sec) + "\",\n";
        
        // meta data
        outputString += "\"metaData\": \"" + metaData + "\",\n";

        // out put to json file, create the string first, then write to file
        outputString += "\"results\": [\n";
        for(int i = 0; i < resultArray.size(); i++){// for each result
            outputString += resultArray[i].outputString();
            if(i != resultArray.size() - 1){
                outputString += ",";
            }
        }
        outputString += "]";

        outputString += "}\n";

        file << outputString;
        file.close();

    }

    ~GMMResult() = default;

};


} // namespace cudaGMMWeight

#endif