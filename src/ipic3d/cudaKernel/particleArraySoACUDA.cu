#include "particleArraySoACUDA.cuh"
#include "particleArrayCUDA.cuh"
#include "cudaTypeDef.cuh"


namespace particleArraySoA{


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







