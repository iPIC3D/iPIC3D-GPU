#ifndef _HASHED_SUM_H_
#define _HASHED_SUM_H_

#include "cuda.h"
#include "cudaTypeDef.cuh"


class hashedSum{

private:

    int bucketNum;
    int* bucket = nullptr;
    int sum;

private:

    __host__ __device__ int hash(int key){
        return key % bucketNum;
    }

public:

    __host__ hashedSum(int bucketNum):bucketNum(bucketNum){
        cudaErrChk(cudaMalloc(&bucket, bucketNum * sizeof(int)));
        resetBucket();
    }

    __device__ hashedSum(int bucketNum):bucketNum(bucketNum){
        bucket = new int[bucketNum];
        resetBucket();
    }

    __host__ __device__ hashedSum(int bucketNum, int* bucketCUDAPtr):bucketNum(bucketNum){
        bucket = bucketCUDAPtr;
        resetBucket();
    }

    __host__ __device__ int add(int key){
        return atomicAdd(&(bucket[hash(key)]), 1);
    }

    __host__ __device__ int getIndex(int key, int id){
        int index = 0;
        for(int i=0; i < hash(key); i++)index += bucket[i];
        index += bucket[hash(key)];

        return index;
    }
    __host__ __device__ int getSum(){
        if(sum != -1)return sum;
        
        int sum = 0;
        for(int i=0; i < bucketNum; i++)sum += bucket[i];
        return sum;
    }

    __host__ __device__ void resetBucket(){
        for(int i=0; i < bucketNum; i++)bucket[i] = 0;
        sum = -1;
    }

    __host__ __device__ void resize(int buckertNum){
        if(bucket != nullptr)delete[] bucket;
        this->bucketNum = buckertNum;
        bucket = new int[bucketNum];
    }

    __host__ __device__ ~hashedSum(){
        delete[] bucket;
    }

};









#endif