#ifndef _ARRAY_CUDA_H_
#define _ARRAY_CUDA_H_


#include <stdexcept>
#include <iostream>
#include "cudaTypeDef.cuh"
#include <cassert>



template <typename T, int sizeUnit = 64>
class arrayCUDA
{
using commonInt = uint32_t;

private:
    commonInt numberOfElement;
    T* arrayPtr;
    commonInt arraySize;

    const bool onHeap;

    /**
     * @brief Round a requested element count up to the internal allocation quantum.
     * @param size Requested element count.
     * @return Rounded element count aligned to `sizeUnit`.
     */
    commonInt roundUpToSizeUnit(commonInt size){
        commonInt rem = size % sizeUnit;
        return rem == 0 ? size : size + sizeUnit - rem;
    }

protected:
    cudaStream_t stream = 0;

public: // con-de-structor

    /**
     * @brief Create an empty device array with reserved capacity.
     * @param requiredSize Requested capacity in elements.
     */
    __host__ arrayCUDA(commonInt requiredSize): onHeap(false), numberOfElement(0){
        arraySize = roundUpToSizeUnit(requiredSize);
        cudaErrChk(cudaMalloc((void**)&arrayPtr, arraySize * sizeof(T)));
    }

    /**
     * @brief Create a device array from host data.
     * @param hostArrayPtr Host pointer holding the source elements.
     * @param size Number of valid source elements.
     */
    __host__ arrayCUDA(const T* hostArrayPtr, commonInt size): onHeap(false), numberOfElement(size){
        arraySize = roundUpToSizeUnit(size);
        cudaErrChk(cudaMalloc(&arrayPtr, arraySize * sizeof(T)));
        cudaErrChk(cudaMemcpyAsync(arrayPtr, hostArrayPtr, numberOfElement * sizeof(T), cudaMemcpyDefault, stream));

        cudaErrChk(cudaStreamSynchronize(stream));
    }

    /**
     * @brief Create a device array from host data with extra spare capacity.
     * @param hostArrayPtr Host pointer holding the source elements.
     * @param size Number of valid source elements.
     * @param expandIndex Capacity growth factor applied before allocation.
     */
    __host__ arrayCUDA(const T* hostArrayPtr, commonInt size, cudaTypeSingle expandIndex): onHeap(false), numberOfElement(size){
        size = size * expandIndex;
        arraySize = roundUpToSizeUnit(size);
        cudaErrChk(cudaMalloc((void**)&arrayPtr, arraySize * sizeof(T)));
        cudaErrChk(cudaMemcpyAsync(arrayPtr, hostArrayPtr, numberOfElement * sizeof(T), cudaMemcpyDefault, stream));
        cudaErrChk(cudaStreamSynchronize(stream));
    }

    __host__ ~arrayCUDA(){
        if(arrayPtr != nullptr){
            if(onHeap == false)cudaErrChk(cudaFree(arrayPtr));
            else { std::cerr << "arrayCUDA: the array is allocated on device heap!" << std::endl; assert(0);}
        }
    }

    // device heap array
    // __device__ arrayCUDA(commonInt requiredSize);
    // __device__ arrayCUDA(T* deviceArrayPtr, commonInt size);
    // __device__ ~arrayCUDA();

    __host__ void assignStream(cudaStream_t s){ stream = s; }

public: // utilities

    __host__ __device__ commonInt getNOE(){ return numberOfElement; }
    __host__ __device__ void setNOE(commonInt val){ numberOfElement = val; }
    __host__ __device__ commonInt getSize(){ return arraySize; }
    //__host__ __device__ void setSize(commonInt val){ arraySize = val; }

    __host__ __device__ T* getArray(){ return arrayPtr; }
    __host__ __device__ T* getElement(commonInt index){ return arrayPtr + index;}

    __host__ commonInt appendElement(T* p){
        if(numberOfElement >= (arraySize-1))assert(0);

        memcpy(arrayPtr + numberOfElement, p, sizeof(T));
        return numberOfElement++;
    }

    /**
     * @brief Expand the device allocation if `targetedSize` exceeds current capacity.
     * @param targetedSize Requested capacity in elements.
     * @return Final allocated capacity.
     */
    __host__ commonInt expand(commonInt targetedSize){
        if(targetedSize <= arraySize) return arraySize;

        arraySize = roundUpToSizeUnit(targetedSize);
        auto oldArray = arrayPtr;
        cudaErrChk(cudaMalloc(&arrayPtr, arraySize * sizeof(T)));
        cudaErrChk(cudaMemcpyAsync(arrayPtr, oldArray, numberOfElement * sizeof(T), cudaMemcpyDefault, stream));
        cudaErrChk(cudaStreamSynchronize(stream));
        cudaErrChk(cudaFree(oldArray));
        
        return arraySize;
    }
    /**
     * @brief Expand the device allocation using an explicit CUDA stream.
     * @param targetedSize Requested capacity in elements.
     * @param s CUDA stream used for the copy and synchronization.
     * @return Final allocated capacity.
     */
    __host__ commonInt expand(commonInt targetedSize, cudaStream_t s){
        if(targetedSize <= arraySize) return arraySize;
        arraySize = roundUpToSizeUnit(targetedSize);
        auto oldArray = arrayPtr;
        cudaErrChk(cudaMalloc(&arrayPtr, arraySize * sizeof(T)));
        cudaErrChk(cudaMemcpyAsync(arrayPtr, oldArray, numberOfElement * sizeof(T), cudaMemcpyDefault, s));
        cudaErrChk(cudaStreamSynchronize(s));
        cudaErrChk(cudaFree(oldArray));
        
        return arraySize;
    }

    /**
     * @brief Resize the device allocation while preserving the current element count.
     * @param targetedSize Requested capacity in elements.
     * @return Final allocated capacity.
     */
    __host__ commonInt resize(commonInt targetedSize){
        if(targetedSize <= numberOfElement) return arraySize;

        arraySize = roundUpToSizeUnit(targetedSize);
        auto oldArray = arrayPtr;
        cudaErrChk(cudaMalloc(&arrayPtr, arraySize * sizeof(T)));
        cudaErrChk(cudaMemcpyAsync(arrayPtr, oldArray, numberOfElement * sizeof(T), cudaMemcpyDefault, stream));
        cudaErrChk(cudaStreamSynchronize(stream));
        cudaErrChk(cudaFree(oldArray));

        return arraySize;
    }

    __host__ virtual arrayCUDA* copyToDevice(){
        arrayCUDA* ptr = nullptr;
        cudaErrChk(cudaMalloc((void**)&ptr, sizeof(arrayCUDA)));
        cudaErrChk(cudaMemcpyAsync(ptr, this, sizeof(arrayCUDA), cudaMemcpyDefault, stream));

        cudaErrChk(cudaStreamSynchronize(stream));
        return ptr;
    }

public: // on device implementation

    // __device__ commonInt expand(commonInt targetedSize);
    // __device__ commonInt resize(commonInt targetedSize);

};
















#endif
