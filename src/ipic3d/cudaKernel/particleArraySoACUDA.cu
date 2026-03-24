#include "particleArraySoACUDA.cuh"
#include "particleArrayCUDA.cuh"
#include "cudaTypeDef.cuh"


// ── Runtime AoS→SoA scatter kernel (used after H→D AoS copies) ──
//
// Reads from an *external* AoS staging buffer on device, writes into the
// SoA arrays of pclsArray at [destOffset .. destOffset+count-1].
// The staging buffer indices are [0 .. count-1].

__global__ void scatterAoSToSoAKernel(const SpeciesParticle* __restrict__ aosStagingBuf,
                                       particleArrayCUDA* pclsArray,
                                       uint32_t destOffset, uint32_t count)
{
    const uint32_t tidx = blockIdx.x * blockDim.x + threadIdx.x;
    if (tidx >= count) return;
    const uint32_t pidx = destOffset + tidx;
    const SpeciesParticle& pcl = aosStagingBuf[tidx];
    pclsArray->getU()[pidx] = pcl.get_u();
    pclsArray->getV()[pidx] = pcl.get_v();
    pclsArray->getW()[pidx] = pcl.get_w();
    pclsArray->getQ()[pidx] = pcl.get_q();
    pclsArray->getX()[pidx] = pcl.get_x();
    pclsArray->getY()[pidx] = pcl.get_y();
    pclsArray->getZ()[pidx] = pcl.get_z();
    pclsArray->getT()[pidx] = pcl.get_t();
}


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







