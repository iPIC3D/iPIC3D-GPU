#include "particleArraySoACUDA.cuh"
#include "particleArrayCUDA.cuh"
#include "cudaTypeDef.cuh"


// ── Runtime AoS→SoA scatter kernel (used after H→D AoS copies) ──

__global__ void scatterAoSToSoAKernel(particleArrayCUDA* pclsArray,
                                       uint32_t offset, uint32_t count)
{
    const uint32_t tidx = blockIdx.x * blockDim.x + threadIdx.x;
    if (tidx >= count) return;
    const uint32_t pidx = offset + tidx;
    const SpeciesParticle& pcl = pclsArray->getpcls()[pidx];
    pclsArray->getU()[pidx] = pcl.get_u();
    pclsArray->getV()[pidx] = pcl.get_v();
    pclsArray->getW()[pidx] = pcl.get_w();
    pclsArray->getQ()[pidx] = pcl.get_q();
    pclsArray->getX()[pidx] = pcl.get_x();
    pclsArray->getY()[pidx] = pcl.get_y();
    pclsArray->getZ()[pidx] = pcl.get_z();
    pclsArray->getT()[pidx] = pcl.get_t();
}


// ── Runtime SoA→AoS gather kernel (used before D→H AoS copies) ──

__global__ void gatherSoAToAoSKernel(particleArrayCUDA* pclsArray,
                                      uint32_t offset, uint32_t count)
{
    const uint32_t tidx = blockIdx.x * blockDim.x + threadIdx.x;
    if (tidx >= count) return;
    const uint32_t pidx = offset + tidx;
    SpeciesParticle& pcl = pclsArray->getpcls()[pidx];
    pcl.set_u(pclsArray->getU()[pidx]);
    pcl.set_v(pclsArray->getV()[pidx]);
    pcl.set_w(pclsArray->getW()[pidx]);
    pcl.set_q(pclsArray->getQ()[pidx]);
    pcl.set_x(pclsArray->getX()[pidx]);
    pcl.set_y(pclsArray->getY()[pidx]);
    pcl.set_z(pclsArray->getZ()[pidx]);
    pcl.set_t(pclsArray->getT()[pidx]);
}


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


template<typename T, int startElement, int stopElement>
__host__ void particleArraySoACUDA<T, startElement, stopElement>::updateFromSoA(particleArrayCUDA* pclArray){
    // Free any previously owned memory
    if (allocated) {
        freeMemory();
        allocated = false;
    }

    nop = pclArray->getNOP();
    size = 0; // non-owning view — no owned capacity

    // Borrow device pointers from particleArrayCUDA's persistent SoA
    cudaParticleType* soaPtrs[8] = {
        pclArray->getU(), pclArray->getV(), pclArray->getW(), pclArray->getQ(),
        pclArray->getX(), pclArray->getY(), pclArray->getZ(), pclArray->getT()
    };
    for (int i = startElement; i <= stopElement; i++) {
        elementPtr[i] = soaPtrs[i];
    }
}


template class particleArraySoA::particleArraySoACUDA<cudaParticleType>;
template class particleArraySoA::particleArraySoACUDA<cudaParticleType, 0, 2>;
template class particleArraySoA::particleArraySoACUDA<cudaParticleType, 0, 3>;
                            
} // namespace particleArraySoA







