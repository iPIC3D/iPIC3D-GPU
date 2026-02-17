
#include "cudaTypeDef.cuh"
#include "arrayCUDA.cuh"
#include "hashedSum.cuh"
#include "particleArrayCUDA.cuh"
#include "particleExchange.cuh"
#include "planetKernel.cuh"


// ═══════════════════════════════════════════════════════════════
//  Planet Extraction — compact PLANET particles into planetArray
// ═══════════════════════════════════════════════════════════════

/**
 * @brief Compact all particles flagged PLANET into the planetArray (device-only).
 *        Analogous to exitingKernel but writes to a device-only buffer.
 *        Uses hashedSum[PLANET_HASHEDSUM_INDEX] for scatter indices.
 */
__global__ void planetExtractionKernel(
    particleArrayCUDA* pclsArray,
    departureArrayType* departureArray,
    planetArray* planetArr,
    hashedSum* hashedSumArray)
{
    uint pidx = blockIdx.x * blockDim.x + threadIdx.x;
    if (pidx >= pclsArray->getNOP()) return;

    auto dep = departureArray->getArray()[pidx];
    if (dep.dest != departureArrayElementType::PLANET) return;

    int index = hashedSumArray[departureArrayElementType::PLANET_HASHEDSUM_INDEX]
                    .getIndex(pidx, dep.hashedId);

    memcpy(planetArr->getArray() + index,
           pclsArray->getpcls() + pidx,
           sizeof(SpeciesParticle));
}


// ═══════════════════════════════════════════════════════════════
//  Ion charge reduction
// ═══════════════════════════════════════════════════════════════

/**
 * @brief Sum |q| of all particles in one species' planetArray.
 *        Block-level reduction then atomicAdd into *chargeOut.
 */
__global__ void planetChargeReductionKernel(
    planetArray* planetArr, int count,
    cudaParticleType* chargeOut)
{
    extern __shared__ cudaParticleType sdata[];

    uint tid  = threadIdx.x;
    uint gid  = blockIdx.x * blockDim.x + threadIdx.x;

    // load
    cudaParticleType val = 0;
    if (gid < (uint)count) {
        val = fabs(planetArr->getArray()[gid].get_q());
    }
    sdata[tid] = val;
    __syncthreads();

    // tree reduction in shared mem
    for (uint s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }

    if (tid == 0) atomicAdd(chargeOut, sdata[0]);
}


// ═══════════════════════════════════════════════════════════════
//  Electron energy computation
// ═══════════════════════════════════════════════════════════════

/**
 * @brief For each electron planet particle compute kinetic energy and write
 *        into merged buffers at position (speciesOffset + localIndex).
 *        Energy: Ek = |q| / (2 * |qom|) * (u^2 + v^2 + w^2)
 */
__global__ void planetEnergyKernel(
    planetArray* planetArr, int count,
    cudaParticleType qom,
    cudaParticleType* energyBuf,
    uint32_t*         globalIdxBuf,
    int speciesOffset)
{
    uint gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= (uint)count) return;

    const SpeciesParticle& pcl = planetArr->getArray()[gid];

    const cudaParticleType u = pcl.get_u();
    const cudaParticleType v = pcl.get_v();
    const cudaParticleType w = pcl.get_w();
    const cudaParticleType absq = fabs(pcl.get_q());
    const cudaParticleType mass = absq / fabs(qom);  // |q| / |q/m| = m

    const int idx = speciesOffset + gid;
    energyBuf[idx]    = (cudaParticleType)0.5 * mass * (u * u + v * v + w * w);
    globalIdxBuf[idx] = (uint32_t)idx;
}


// ═══════════════════════════════════════════════════════════════
//  Bitonic sort (descending by energy, key-value)
// ═══════════════════════════════════════════════════════════════

/**
 * @brief Pad tail of arrays beyond realN with -inf keys so they
 *        sink to the end under descending sort.
 */
__global__ void bitonicPadKernel(
    cudaParticleType* keys, uint32_t* values,
    int realN, int paddedN)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x + realN;
    if (i >= paddedN) return;
    keys[i]   = (cudaParticleType)(-1e30);
    values[i] = 0xFFFFFFFFu;
}

/**
 * @brief One compare-and-swap step of bitonic sort.
 *        Produces a descending sequence when the outer loop completes.
 * @param j XOR distance for this step
 * @param k block size for this stage
 * @param n padded array length (power of 2)
 */
__global__ void bitonicSortStepKernel(
    cudaParticleType* __restrict__ keys,
    uint32_t*         __restrict__ values,
    int j, int k, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    int ixj = i ^ j;
    if (ixj <= i) return;  // one thread per pair

    // ascending = true when (i & k) != 0  →  descending when false
    bool ascending = ((i & k) != 0);

    if (( ascending && keys[i] > keys[ixj]) ||
        (!ascending && keys[i] < keys[ixj]))
    {
        // swap keys
        cudaParticleType tmpK = keys[i];
        keys[i]   = keys[ixj];
        keys[ixj] = tmpK;

        // swap values
        uint32_t tmpV = values[i];
        values[i]   = values[ixj];
        values[ixj] = tmpV;
    }
}


// ═══════════════════════════════════════════════════════════════
//  Charge cutoff — prefix-sum to find how many electrons to remove
// ═══════════════════════════════════════════════════════════════

/**
 * @brief Sequential prefix-sum of |q| in sorted (descending energy) order.
 *        Finds the cutoff index: electrons [0..cutoff-1] are deleted,
 *        electrons [cutoff..n-1] survive and will be reflected.
 *
 *        Single-thread kernel — planet particle counts are small (typically
 *        <100K), so a sequential scan is fast enough and avoids complexity.
 *
 * @param planetArrs  array of device pointers to per-species planetArrays
 * @param nSpecies    number of electron species
 * @param speciesOffsets  prefix offsets into the merged buffers per electron species
 * @param sortedGlobalIdx  sorted global indices (reordered by energy)
 * @param n  total number of electron planet particles
 * @param ionChargeTarget  device pointer to total |Q_ion| to match
 * @param cutoffIndex  output: first sorted index that SURVIVES (reflects)
 */
__global__ void chargeCutoffKernel(
    planetArray** planetArrs, int nSpecies, const int* speciesOffsets,
    const uint32_t* sortedGlobalIdx, int n,
    const cudaParticleType* ionChargeTarget,
    int* cutoffIndex)
{
    if (threadIdx.x != 0 || blockIdx.x != 0) return;

    const cudaParticleType ionCharge = *ionChargeTarget;

    // If no ion charge to match, all electrons survive (reflect all)
    if (ionCharge <= 0) {
        *cutoffIndex = 0;
        return;
    }

    cudaParticleType cumQ = 0;
    for (int i = 0; i < n; i++) {
        uint32_t gidx = sortedGlobalIdx[i];

        // Decode which species and local index from gidx
        // gidx = speciesOffset[s] + localIdx
        // Find species by checking which offset range gidx falls in
        int localIdx = (int)gidx;
        int speciesIdx = -1;
        for (int s = 0; s < nSpecies; s++) {
            int sStart = speciesOffsets[s];
            int sEnd   = (s + 1 < nSpecies) ? speciesOffsets[s + 1] : n;
            if ((int)gidx >= sStart && (int)gidx < sEnd) {
                speciesIdx = s;
                localIdx = (int)gidx - sStart;
                break;
            }
        }
        if (speciesIdx < 0) continue;  // should not happen

        cudaParticleType absq = fabs(planetArrs[speciesIdx]->getArray()[localIdx].get_q());
        cumQ += absq;

        if (cumQ >= ionCharge) {
            *cutoffIndex = i + 1;  // delete [0..i], reflect [i+1..n-1]
            return;
        }
    }
    *cutoffIndex = n;  // not enough electrons — delete all
}


// ═══════════════════════════════════════════════════════════════
//  Fused reflect + compact kernel (all electron species, device-only cutoff)
// ═══════════════════════════════════════════════════════════════

/**
 * @brief Each thread handles one position in [0, totalElecPlanet).
 *        If that position is >= *cutoffDevice (i.e. a survivor),
 *        reflect the particle in its planetArray, then write the
 *        reflected particle into a compact output buffer using an
 *        atomicAdd on a per-species counter.
 */
__global__ void planetReflectCompactKernel(
    planetArray** planetArrs, int nElecSpecies,
    const int* speciesOffsets,
    const uint32_t* sortedGlobalIdx,
    const int* cutoffDevice,
    int totalElecPlanet,
    SpeciesParticle* outputBuf,
    int* survivorCounters,
    cudaCommonType originX, cudaCommonType originY, cudaCommonType originZ,
    cudaCommonType sphereRadius, int doSphere)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= totalElecPlanet) return;

    // Read cutoff from device memory (written by chargeCutoffKernel on same stream)
    const int survivorStart = *cutoffDevice;

    // Only survivors (indices >= survivorStart in sorted order) are processed
    int sortedIdx = survivorStart + tid;
    if (sortedIdx >= totalElecPlanet) return;

    uint32_t gidx = sortedGlobalIdx[sortedIdx];
    if (gidx == 0xFFFFFFFFu) return;  // padding sentinel

    // Decode which electron species this global index belongs to
    int speciesIdx = -1;
    int localIdx   = -1;
    int speciesCount = 0;
    for (int s = 0; s < nElecSpecies; s++) {
        int sStart = speciesOffsets[s];
        int sEnd   = (s + 1 < nElecSpecies) ? speciesOffsets[s + 1] : totalElecPlanet;
        if ((int)gidx >= sStart && (int)gidx < sEnd) {
            speciesIdx = s;
            localIdx = (int)gidx - sStart;
            speciesCount = sEnd - sStart;
            break;
        }
    }
    if (speciesIdx < 0 || localIdx < 0 || localIdx >= speciesCount) return;

    // Read the original planet particle
    SpeciesParticle pcl = planetArrs[speciesIdx]->getArray()[localIdx];

    // ── Reflect ──
    const cudaCommonType eps = sphereRadius * (cudaCommonType)1e-4;

    if (doSphere == 1) { // 3D
        cudaCommonType dx = pcl.get_x() - originX;
        cudaCommonType dy = pcl.get_y() - originY;
        cudaCommonType dz = pcl.get_z() - originZ;
        cudaCommonType r  = sqrt(dx * dx + dy * dy + dz * dz);
        if (r < (cudaCommonType)1e-30) r = (cudaCommonType)1e-30;
        cudaCommonType invr = (cudaCommonType)1.0 / r;

        cudaCommonType nx = dx * invr;
        cudaCommonType ny = dy * invr;
        cudaCommonType nz = dz * invr;

        cudaCommonType vdotn = pcl.get_u() * nx + pcl.get_v() * ny + pcl.get_w() * nz;
        pcl.set_u(0, pcl.get_u() - (cudaCommonType)2.0 * vdotn * nx);
        pcl.set_u(1, pcl.get_v() - (cudaCommonType)2.0 * vdotn * ny);
        pcl.set_u(2, pcl.get_w() - (cudaCommonType)2.0 * vdotn * nz);

        pcl.set_x(0, originX + (sphereRadius + eps) * nx);
        pcl.set_x(1, originY + (sphereRadius + eps) * ny);
        pcl.set_x(2, originZ + (sphereRadius + eps) * nz);

    } else if (doSphere == 2) { // 2D (XZ plane)
        cudaCommonType dx = pcl.get_x() - originX;
        cudaCommonType dz = pcl.get_z() - originZ;
        cudaCommonType r  = sqrt(dx * dx + dz * dz);
        if (r < (cudaCommonType)1e-30) r = (cudaCommonType)1e-30;
        cudaCommonType invr = (cudaCommonType)1.0 / r;

        cudaCommonType nx = dx * invr;
        cudaCommonType nz = dz * invr;

        cudaCommonType vdotn = pcl.get_u() * nx + pcl.get_w() * nz;
        pcl.set_u(0, pcl.get_u() - (cudaCommonType)2.0 * vdotn * nx);
        pcl.set_u(2, pcl.get_w() - (cudaCommonType)2.0 * vdotn * nz);

        pcl.set_x(0, originX + (sphereRadius + eps) * nx);
        pcl.set_x(2, originZ + (sphereRadius + eps) * nz);
    }

    // ── Compact: atomicAdd to get a write slot in the output buffer ──
    int mySlot = atomicAdd(&survivorCounters[speciesIdx], 1);
    int outOffset = speciesOffsets[speciesIdx];

    memcpy(outputBuf + outOffset + mySlot,
           &pcl, sizeof(SpeciesParticle));
}
