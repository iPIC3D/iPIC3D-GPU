#ifndef _HASHED_SUM_H_
#define _HASHED_SUM_H_

#include "cuda.h"
#include "cudaTypeDef.cuh"



class hashedSum{

private:

    int bucketNum;
    int bucket[10];

private:

    __host__ __device__ int hash(int key){
        return key % bucketNum;
    }

public:

    __host__ hashedSum(int bucketNum):bucketNum(bucketNum){
        // if(bucketNum > bucketSize)throw std::out_of_range("[!]hashedSum out of range");
        resetBucket();
    }

    // __device__ hashedSum(int bucketNum):bucketNum(bucketNum){
    //     bucket = new int[bucketNum];
    //     resetBucket();
    // }

    // __host__ hashedSum(int bucketNum, int* bucketCUDAPtr):bucketNum(bucketNum){
    //     bucket = bucketCUDAPtr;
    //     resetBucket();
    // }

    __device__ int add(int key);

    __host__ __device__ int getIndex(int key, int id){
        int index = 0;
        for(int i=0; i < hash(key); i++)index += bucket[i];
        index += id;

        return index;
    }
    __host__ __device__ int getSum(){
        int sum = 0;
        for(int i=0; i < bucketNum; i++)sum += bucket[i];
        return sum;
    }

    __host__ void resetBucket(){
        memset(bucket, 0, bucketNum * sizeof(int));
    }

    __host__ void resize(int buckertNum){
        this->bucketNum = buckertNum;
        resetBucket();
    }

    __host__ ~hashedSum(){
    }

};









#endif