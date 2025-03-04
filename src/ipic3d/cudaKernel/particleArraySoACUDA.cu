#include "particleArraySoACUDA.cuh"
#include "particleArrayCUDA.cuh"
#include "cudaTypeDef.cuh"




namespace particleArraySoA{

template<typename T, int startElement = 0, int stopElement = 6>
__global__ void particleToSoAKernel(SpeciesParticle* pclArray, int nop, particleArraySoACUDA<T, startElement, stopElement>* pclArraySoA){
    uint pidx = blockIdx.x * blockDim.x + threadIdx.x;
    uint gridSize = gridDim.x * blockDim.x;

    for (uint i = pidx; i < nop; i += gridSize){
        if constexpr (0 >= startElement && 0 <= stopElement) pclArraySoA->getElement(U)[i] = pclArray[i].get_u();
        if constexpr (1 >= startElement && 1 <= stopElement) pclArraySoA->getElement(V)[i] = pclArray[i].get_v();
        if constexpr (2 >= startElement && 2 <= stopElement) pclArraySoA->getElement(W)[i] = pclArray[i].get_w();
        if constexpr (3 >= startElement && 3 <= stopElement) pclArraySoA->getElement(Q)[i] = pclArray[i].get_q();
        if constexpr (4 >= startElement && 4 <= stopElement) pclArraySoA->getElement(X)[i] = pclArray[i].get_x();
        if constexpr (5 >= startElement && 5 <= stopElement) pclArraySoA->getElement(Y)[i] = pclArray[i].get_y();
        if constexpr (6 >= startElement && 6 <= stopElement) pclArraySoA->getElement(Z)[i] = pclArray[i].get_z();
    }
}



template<typename T, int startElement, int stopElement>
__host__ particleArraySoACUDA<T, startElement, stopElement>::particleArraySoACUDA(particleArrayCUDA* pclArray, cudaStream_t stream){
    nop = pclArray->getNOP();
    size = nop * 1.2;
    allocateMemory();
    auto objOnDevice = copyToDevice(this, stream);
    particleToSoAKernel<T, startElement, stopElement><<<getGridSize(nop / 64, 256), 256, 0, stream>>>(pclArray->getArray(), nop, objOnDevice);
    cudaErrChk(cudaStreamSynchronize(stream));
    cudaErrChk(cudaFree(objOnDevice));
}


template<typename T, int startElement, int stopElement>
__host__ void particleArraySoACUDA<T, startElement, stopElement>::updateFromAoS(particleArrayCUDA* pclArray, cudaStream_t stream){
    nop = pclArray->getNOP();

    if(!allocated){
        size = nop * 1.2;
        allocateMemory();
        allocated = true;
    }else if(allocated && size < nop){
        freeMemory();
        size = nop * 1.2;
        allocateMemory();
    }

    auto objOnDevice = copyToDevice(this, stream);
    particleToSoAKernel<T, startElement, stopElement><<<getGridSize(nop / 64, 256), 256, 0, stream>>>(pclArray->getArray(), nop, objOnDevice);
    cudaErrChk(cudaStreamSynchronize(stream));
    cudaErrChk(cudaFree(objOnDevice));
}


template class particleArraySoA::particleArraySoACUDA<cudaParticleType>;
template class particleArraySoA::particleArraySoACUDA<cudaParticleType, 0, 2>;
template class particleArraySoA::particleArraySoACUDA<cudaParticleType, 0, 3>;
                            
} // namespace particleArraySoA







