#include "hashedSum.cuh"


__device__ int hashedSum::add(int key){
    return atomicAdd(&(bucket[hash(key)]), 1);
}




