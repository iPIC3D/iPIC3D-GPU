
#include "cudaTypeDef.cuh"
#include "arrayCUDA.cuh"
#include "hashedSum.cuh"
#include "particleArrayCUDA.cuh"
#include "particleExchange.cuh"
#include "planetKernel.cuh"


// ======= Planet extraction =======

constexpr cudaCommonType TWO_PI = (cudaCommonType)6.28318530717958647692;

/**
 * @brief Compact all particles flagged PLANET into the planet SoA buffer (device-only).
 *        Analogous to exitingKernel but writes to a device-only SoA buffer.
 *        Uses hashedSum[PLANET_HASHEDSUM_INDEX] for scatter indices.
 * @param pclsArray Source particle SoA buffer.
 * @param departureArray Per-particle destination metadata.
 * @param planetArr Destination device-side planet particle buffer.
 * @param hashedSumArray Prefix-sum helpers for destination indexing.
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

    // Gather SoA fields into AoS SpeciesParticle for planet buffer
    SpeciesParticle pcl;
    pcl.set_u(pclsArray->getU()[pidx]);
    pcl.set_v(pclsArray->getV()[pidx]);
    pcl.set_w(pclsArray->getW()[pidx]);
    pcl.set_q(pclsArray->getQ()[pidx]);
    pcl.set_x(pclsArray->getX()[pidx]);
    pcl.set_y(pclsArray->getY()[pidx]);
    pcl.set_z(pclsArray->getZ()[pidx]);
    pcl.set_t(pclsArray->getT()[pidx]);
    planetArr->getArray()[index] = pcl;
}


// ======= Ion charge reduction =======

/**
 * @brief Sum |q| of all particles in one species' planet SoA buffer.
 *        Block-level reduction then atomicAdd into *chargeOut.
 * @param planetArr Device buffer holding the extracted planet particles.
 * @param count Number of valid entries in `planetArr`.
 * @param chargeOut Device scalar accumulating the total absolute charge.
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


// ======= Electron energy staging =======

/**
 * @brief For each electron planet particle compute kinetic energy and write
 *        into merged buffers at position (speciesOffset + localIndex).
 *        Energy: Ek = |q| / (2 * |qom|) * (u^2 + v^2 + w^2)
 * @param planetArr Device buffer holding one electron species' planet particles.
 * @param count Number of valid entries in `planetArr`.
 * @param qom Charge-to-mass ratio of the species.
 * @param energyBuf Merged output buffer for kinetic-energy keys.
 * @param globalIdxBuf Merged output buffer for global particle indices.
 * @param speciesOffset Prefix offset of this species in the merged buffers.
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


// ======= Bitonic energy sort =======

/**
 * @brief Pad tail of arrays beyond realN with -inf keys so they
 *        sink to the end under descending sort.
 * @param keys Sort-key buffer to pad.
 * @param values Payload buffer aligned with `keys`.
 * @param realN Number of valid entries before padding.
 * @param paddedN Power-of-two length used by the bitonic sort.
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
 * @param keys Sort keys updated in place.
 * @param values Payload buffer permuted in lockstep with `keys`.
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
    if (ixj <= i) return;  // one thread handles each compare-and-swap pair

    // ascending = true when (i & k) != 0, descending otherwise
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


// ======= Charge cutoff =======

/**
 * @brief Sequential prefix-sum of |q| in sorted (descending energy) order.
 *        Finds the cutoff index: electrons [0..cutoff-1] are deleted,
 *        electrons [cutoff..n-1] survive and will be reflected.
 *
 *        Single-thread kernel: planet particle counts are small (typically
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
    *cutoffIndex = n;  // not enough electrons; delete all
}


// ======= Specular reflection and compaction =======

/**
 * @brief Specular (mirror) reflect surviving electrons and compact them
 *        into a contiguous output buffer, one segment per electron species.
 *        Reads cutoff from device memory (no host sync needed).
 *        Each thread maps to one entry in the sorted survivor range
 *        (sortedIdx = cutoff + tid); threads beyond the valid range
 *        early-return. Decodes species from the global index, applies
 *        v' = v - 2(v.n)n, places the particle on the sphere surface,
 *        and writes it to outputBuf[speciesOffset + atomicSlot].
 * @param planetArrs Device array of per-species planet buffers.
 * @param nElecSpecies Number of electron species present in `planetArrs`.
 * @param speciesOffsets Prefix offsets for each electron species in merged buffers.
 * @param sortedGlobalIdx Energy-sorted merged particle indices.
 * @param cutoffDevice Device pointer to the first surviving sorted index.
 * @param totalElecPlanet Total number of electron planet particles across species.
 * @param outputBuf Output AoS buffer receiving reflected survivors.
 * @param survivorCounters Per-species device counters for compacted survivors.
 * @param originX X coordinate of the planet center.
 * @param originY Y coordinate of the planet center.
 * @param originZ Z coordinate of the planet center.
 * @param sphereRadius Planet radius used for repositioning.
 * @param doSphere Geometry selector: 1 for full sphere, 2 for XZ-plane circle.
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

    // Reflect.
    const cudaCommonType eps = sphereRadius * (cudaCommonType)5e-2;  // small offset to prevent sticking to surface

    if (doSphere == 1) { // 3D
        const cudaCommonType dx = pcl.get_x() - originX;
        const cudaCommonType dy = pcl.get_y() - originY;
        const cudaCommonType dz = pcl.get_z() - originZ;
        cudaCommonType r  = sqrt(dx * dx + dy * dy + dz * dz);
        if (r < (cudaCommonType)1e-30) r = (cudaCommonType)1e-30;
        const cudaCommonType invr = (cudaCommonType)1.0 / r;

        const cudaCommonType nx = dx * invr;
        const cudaCommonType ny = dy * invr;
        const cudaCommonType nz = dz * invr;

        const cudaCommonType vdotn = pcl.get_u() * nx + pcl.get_v() * ny + pcl.get_w() * nz;
        pcl.set_u(0, pcl.get_u() - (cudaCommonType)2.0 * vdotn * nx);
        pcl.set_u(1, pcl.get_v() - (cudaCommonType)2.0 * vdotn * ny);
        pcl.set_u(2, pcl.get_w() - (cudaCommonType)2.0 * vdotn * nz);

        pcl.set_x(0, originX + (sphereRadius + eps) * nx);
        pcl.set_x(1, originY + (sphereRadius + eps) * ny);
        pcl.set_x(2, originZ + (sphereRadius + eps) * nz);

    } else if (doSphere == 2) { // 2D XZ-plane circle
        const cudaCommonType dx = pcl.get_x() - originX;
        const cudaCommonType dz = pcl.get_z() - originZ;
        cudaCommonType r  = sqrt(dx * dx + dz * dz);
        if (r < (cudaCommonType)1e-30) r = (cudaCommonType)1e-30;
        const cudaCommonType invr = (cudaCommonType)1.0 / r;

        const cudaCommonType nx = dx * invr;
        const cudaCommonType nz = dz * invr;

        const cudaCommonType vdotn = pcl.get_u() * nx + pcl.get_w() * nz;
        pcl.set_u(0, pcl.get_u() - (cudaCommonType)2.0 * vdotn * nx);
        pcl.set_u(2, pcl.get_w() - (cudaCommonType)2.0 * vdotn * nz);

        pcl.set_x(0, originX + (sphereRadius + eps) * nx);
        pcl.set_x(2, originZ + (sphereRadius + eps) * nz);
    }

    // Compact into the per-species output segment.
    int mySlot = atomicAdd(&survivorCounters[speciesIdx], 1);
    int outOffset = speciesOffsets[speciesIdx];

    memcpy(outputBuf + outOffset + mySlot,
           &pcl, sizeof(SpeciesParticle));
}


// ======= Local RNG helpers =======

/**
 * @brief Advance a simple xorshift32 PRNG state.
 * @param seed Current PRNG state.
 * @return Updated PRNG state after one xorshift step.
 */
__device__ inline uint32_t planetRngHash(uint32_t seed)
{
    seed ^= seed << 13;
    seed ^= seed >> 17;
    seed ^= seed << 5;
    return seed;
}

/**
 * @brief Return a uniform random variate in the half-open interval [0, 1).
 *
 * Precision is bounded by the mantissa width of `cudaCommonType`.
 * @param state PRNG state updated in place.
 * @return Pseudorandom variate in `[0, 1)`.
 */
__device__ inline cudaCommonType planetRngUniform(uint32_t& state)
{
    state = planetRngHash(state);
    return (cudaCommonType)(state) * (cudaCommonType)(1.0 / 4294967296.0);
}


// ======= Diffuse reflection and compaction =======

/**
 * @brief Same structure as planetReflectCompactKernel, but instead of
 *        specular (mirror) reflection, the velocity is randomised to a
 *        uniform direction on the OUTWARD hemisphere while preserving
 *        the original speed |v|.
 *
 *        NOTE: this is NOT identical to the legacy CPU function
 *        rotateAndCountParticlesInsideSphere(). The legacy code samples
 *        theta uniformly in [0, pi) (global frame) then rejects the inward
 *        hemisphere, producing a non-uniform p(Omega) ~ 1/sin(theta).
 *        This kernel samples cos(theta) uniformly in (0, 1) in the LOCAL
 *        frame (z_local = outward normal), giving a truly uniform
 *        distribution on the outward hemisphere.
 *
 *        The hemisphere sampling is done without a rejection loop:
 *        a random direction is generated in a local frame whose z-axis
 *        is the outward normal, then rotated to the global frame via an
 *        orthonormal basis (t1, t2, n).
 *        This avoids warp/wavefront divergence on both NVIDIA and AMD GPUs.
 *
 *        Position is reset to the sphere surface (unlike the legacy
 *        code which left it inside).
 *
 * @param planetArrs Device array of per-species planet buffers.
 * @param nElecSpecies Number of electron species present in `planetArrs`.
 * @param speciesOffsets Prefix offsets for each electron species in merged buffers.
 * @param sortedGlobalIdx Energy-sorted merged particle indices.
 * @param cutoffDevice Device pointer to the first surviving sorted index.
 * @param totalElecPlanet Total number of electron planet particles across species.
 * @param outputBuf Output AoS buffer receiving diffusely reflected survivors.
 * @param survivorCounters Per-species device counters for compacted survivors.
 * @param originX X coordinate of the planet center.
 * @param originY Y coordinate of the planet center.
 * @param originZ Z coordinate of the planet center.
 * @param sphereRadius Planet radius used for repositioning.
 * @param doSphere Geometry selector: 1 for full sphere, 2 for XZ-plane circle.
 * @param rngSeedBase Base seed for the per-thread PRNG (for example the cycle number)
 */
__global__ void planetDiffuseCompactKernel(
    planetArray** planetArrs, int nElecSpecies,
    const int* speciesOffsets,
    const uint32_t* sortedGlobalIdx,
    const int* cutoffDevice,
    int totalElecPlanet,
    SpeciesParticle* outputBuf,
    int* survivorCounters,
    cudaCommonType originX, cudaCommonType originY, cudaCommonType originZ,
    cudaCommonType sphereRadius, int doSphere,
    uint32_t rngSeedBase)
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

    // Compute original speed (invariant)
    const cudaCommonType uold = pcl.get_u();
    const cudaCommonType vold = pcl.get_v();
    const cudaCommonType wold = pcl.get_w();
    const cudaCommonType Vmod = sqrt(uold * uold + vold * vold + wold * wold);

    // Initialize the per-thread RNG state.
    uint32_t rngState = rngSeedBase ^ (uint32_t)((uint32_t)sortedIdx * 2654435761u + 1u);
    rngState = planetRngHash(rngState);
    if (rngState == 0u) rngState = 1u;  // xorshift32 absorbs at 0

    const cudaCommonType eps = sphereRadius * (cudaCommonType)5e-2;


    if (doSphere == 1) { // 3D sphere
        const cudaCommonType dx = pcl.get_x() - originX;
        const cudaCommonType dy = pcl.get_y() - originY;
        const cudaCommonType dz = pcl.get_z() - originZ;
        cudaCommonType r = sqrt(dx * dx + dy * dy + dz * dz);
        if (r < (cudaCommonType)1e-30) r = (cudaCommonType)1e-30;
        const cudaCommonType invr = (cudaCommonType)1.0 / r;

        // Outward unit normal n = (nx, ny, nz).
        const cudaCommonType nx = dx * invr;
        const cudaCommonType ny = dy * invr;
        const cudaCommonType nz = dz * invr;

        // Random direction on the outward hemisphere without rejection.
        const cudaCommonType cosTheta = planetRngUniform(rngState);
        const cudaCommonType sinTheta = sqrt((cudaCommonType)1.0 - cosTheta * cosTheta);
        const cudaCommonType phi = TWO_PI * planetRngUniform(rngState);
        const cudaCommonType cphi = cos(phi);
        const cudaCommonType sphi = sin(phi);

        // Velocity in the local frame where z_local aligns with the normal.
        const cudaCommonType vl_x = Vmod * sinTheta * cphi;
        const cudaCommonType vl_y = Vmod * sinTheta * sphi;
        const cudaCommonType vl_z = Vmod * cosTheta;

        // Build an orthonormal basis (t1, t2, n).
        cudaCommonType t1x, t1y, t1z;
        if (fabs(nx) < (cudaCommonType)0.9) {
            const cudaCommonType len = sqrt(nz * nz + ny * ny);
            const cudaCommonType invLen = (cudaCommonType)1.0 / len;
            t1x = (cudaCommonType)0.0;
            t1y =  nz * invLen;
            t1z = -ny * invLen;
        } else {
            const cudaCommonType len = sqrt(nz * nz + nx * nx);
            const cudaCommonType invLen = (cudaCommonType)1.0 / len;
            t1x = -nz * invLen;
            t1y =  (cudaCommonType)0.0;
            t1z =  nx * invLen;
        }
        // t2 = cross(n, t1).
        const cudaCommonType t2x = ny * t1z - nz * t1y;
        const cudaCommonType t2y = nz * t1x - nx * t1z;
        const cudaCommonType t2z = nx * t1y - ny * t1x;

        // Rotate back to the global frame: v = vl_x * t1 + vl_y * t2 + vl_z * n.
        pcl.set_u(0, vl_x * t1x + vl_y * t2x + vl_z * nx);
        pcl.set_u(1, vl_x * t1y + vl_y * t2y + vl_z * ny);
        pcl.set_u(2, vl_x * t1z + vl_y * t2z + vl_z * nz);

        pcl.set_x(0, originX + (sphereRadius + eps) * nx);
        pcl.set_x(1, originY + (sphereRadius + eps) * ny);
        pcl.set_x(2, originZ + (sphereRadius + eps) * nz);

    } else if (doSphere == 2) { // 2D XZ-plane circle
        const cudaCommonType dx = pcl.get_x() - originX;
        const cudaCommonType dz = pcl.get_z() - originZ;
        cudaCommonType r = sqrt(dx * dx + dz * dz);
        if (r < (cudaCommonType)1e-30) r = (cudaCommonType)1e-30;
        const cudaCommonType invr = (cudaCommonType)1.0 / r;

        const cudaCommonType nx = dx * invr;
        const cudaCommonType nz = dz * invr;

        const cudaCommonType alpha = ((cudaCommonType)3.14159265358979323846)
                                     * (planetRngUniform(rngState) - (cudaCommonType)0.5);
        const cudaCommonType ca = cos(alpha);
        const cudaCommonType sa = sin(alpha);
        pcl.set_u(0, Vmod * (ca * nx - sa * nz));
        pcl.set_u(2, Vmod * (ca * nz + sa * nx));

        pcl.set_x(0, originX + (sphereRadius + eps) * nx);
        pcl.set_x(2, originZ + (sphereRadius + eps) * nz);
    }

    // Compact into the per-species output segment.
    int mySlot = atomicAdd(&survivorCounters[speciesIdx], 1);
    int outOffset = speciesOffsets[speciesIdx];

    memcpy(outputBuf + outOffset + mySlot,
           &pcl, sizeof(SpeciesParticle));
}
