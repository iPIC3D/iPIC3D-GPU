
#include "cudaTypeDef.cuh"
#include "arrayCUDA.cuh"
#include "hashedSum.cuh"
#include "particleArrayCUDA.cuh"
#include "particleExchange.cuh"
#include "planetKernel.cuh"


// ═══════════════════════════════════════════════════════════════
//  Planet Extraction — compact PLANET particles into planetArray
// ═══════════════════════════════════════════════════════════════

constexpr cudaCommonType TWO_PI = (cudaCommonType)6.28318530717958647692;

/**
 * @brief Compact all particles flagged PLANET into the planet SoA buffer (device-only).
 *        Analogous to exitingKernel but writes to a device-only SoA buffer.
 *        Uses hashedSum[PLANET_HASHEDSUM_INDEX] for scatter indices.
 */
__global__ void planetExtractionKernel(
    particleArrayCUDA* pclsArray,
    departureArrayType* departureArray,
    ParticleSoADevice* planetSoA,
    hashedSum* hashedSumArray)
{
    uint pidx = blockIdx.x * blockDim.x + threadIdx.x;
    if (pidx >= pclsArray->getNOP()) return;

    auto dep = departureArray->getArray()[pidx];
    if (dep.dest != departureArrayElementType::PLANET) return;

    int index = hashedSumArray[departureArrayElementType::PLANET_HASHEDSUM_INDEX]
                .getIndex(pidx, dep.hashedId);

    // Direct SoA-to-SoA copy — no AoS intermediary
    planetSoA->u[index] = pclsArray->getU()[pidx];
    planetSoA->v[index] = pclsArray->getV()[pidx];
    planetSoA->w[index] = pclsArray->getW()[pidx];
    planetSoA->q[index] = pclsArray->getQ()[pidx];
    planetSoA->x[index] = pclsArray->getX()[pidx];
    planetSoA->y[index] = pclsArray->getY()[pidx];
    planetSoA->z[index] = pclsArray->getZ()[pidx];
    planetSoA->t[index] = pclsArray->getT()[pidx];
}


// ═══════════════════════════════════════════════════════════════
//  Ion charge reduction
// ═══════════════════════════════════════════════════════════════

/**
 * @brief Sum |q| of all particles in one species' planet SoA buffer.
 *        Block-level reduction then atomicAdd into *chargeOut.
 */
__global__ void planetChargeReductionKernel(
    ParticleSoADevice* planetSoA, int count,
    cudaParticleType* chargeOut)
{
    extern __shared__ cudaParticleType sdata[];

    uint tid  = threadIdx.x;
    uint gid  = blockIdx.x * blockDim.x + threadIdx.x;

    // load
    cudaParticleType val = 0;
    if (gid < (uint)count) {
        val = fabs(planetSoA->q[gid]);
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
    ParticleSoADevice* planetSoA, int count,
    cudaParticleType qom,
    cudaParticleType* energyBuf,
    uint32_t*         globalIdxBuf,
    int speciesOffset)
{
    uint gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= (uint)count) return;

    const cudaParticleType pclU = planetSoA->u[gid];
    const cudaParticleType pclV = planetSoA->v[gid];
    const cudaParticleType pclW = planetSoA->w[gid];
    const cudaParticleType absq = fabs(planetSoA->q[gid]);
    const cudaParticleType mass = absq / fabs(qom);  // |q| / |q/m| = m

    const int idx = speciesOffset + gid;
    energyBuf[idx]    = (cudaParticleType)0.5 * mass * (pclU * pclU + pclV * pclV + pclW * pclW);
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
    ParticleSoADevice** planetSoAArrs, int nSpecies, const int* speciesOffsets,
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

        cudaParticleType absq = fabs(planetSoAArrs[speciesIdx]->q[localIdx]);
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
 * @brief Specular (mirror) reflect surviving electrons and compact them
 *        into a contiguous output buffer, one segment per electron species.
 *        Reads cutoff from device memory (no host sync needed).
 *        Each thread maps to one entry in the sorted survivor range
 *        (sortedIdx = cutoff + tid); threads beyond the valid range
 *        early-return.  Decodes species from the global index, applies
 *        v' = v - 2(v·n̂)n̂, places the particle on the sphere surface,
 *        and writes it to outputBuf[speciesOffset + atomicSlot].
 */
__global__ void planetReflectCompactKernel(
    ParticleSoADevice** planetSoAArrs, int nElecSpecies,
    const int* speciesOffsets,
    const uint32_t* sortedGlobalIdx,
    const int* cutoffDevice,
    int totalElecPlanet,
    ParticleSoADevice* outputSoA,
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

    // Read the original planet particle from SoA
    ParticleSoADevice* srcSoA = planetSoAArrs[speciesIdx];
    cudaCommonType pclU = srcSoA->u[localIdx];
    cudaCommonType pclV = srcSoA->v[localIdx];
    cudaCommonType pclW = srcSoA->w[localIdx];
    cudaCommonType pclQ = srcSoA->q[localIdx];
    cudaCommonType pclX = srcSoA->x[localIdx];
    cudaCommonType pclY = srcSoA->y[localIdx];
    cudaCommonType pclZ = srcSoA->z[localIdx];
    cudaCommonType pclT = srcSoA->t[localIdx];

    // ── Reflect ──
    const cudaCommonType eps = sphereRadius * (cudaCommonType)5e-2;  // small offset to prevent sticking to surface

    if (doSphere == 1) { // 3D
        const cudaCommonType dx = pclX - originX;
        const cudaCommonType dy = pclY - originY;
        const cudaCommonType dz = pclZ - originZ;
        cudaCommonType r  = sqrt(dx * dx + dy * dy + dz * dz);
        if (r < (cudaCommonType)1e-30) r = (cudaCommonType)1e-30;
        const cudaCommonType invr = (cudaCommonType)1.0 / r;

        const cudaCommonType nx = dx * invr;
        const cudaCommonType ny = dy * invr;
        const cudaCommonType nz = dz * invr;

        const cudaCommonType vdotn = pclU * nx + pclV * ny + pclW * nz;
        pclU = pclU - (cudaCommonType)2.0 * vdotn * nx;
        pclV = pclV - (cudaCommonType)2.0 * vdotn * ny;
        pclW = pclW - (cudaCommonType)2.0 * vdotn * nz;

        pclX = originX + (sphereRadius + eps) * nx;
        pclY = originY + (sphereRadius + eps) * ny;
        pclZ = originZ + (sphereRadius + eps) * nz;

    } else if (doSphere == 2) { // 2D (XZ plane)
        const cudaCommonType dx = pclX - originX;
        const cudaCommonType dz = pclZ - originZ;
        cudaCommonType r  = sqrt(dx * dx + dz * dz);
        if (r < (cudaCommonType)1e-30) r = (cudaCommonType)1e-30;
        const cudaCommonType invr = (cudaCommonType)1.0 / r;

        const cudaCommonType nx = dx * invr;
        const cudaCommonType nz = dz * invr;

        const cudaCommonType vdotn = pclU * nx + pclW * nz;
        pclU = pclU - (cudaCommonType)2.0 * vdotn * nx;
        pclW = pclW - (cudaCommonType)2.0 * vdotn * nz;

        pclX = originX + (sphereRadius + eps) * nx;
        pclZ = originZ + (sphereRadius + eps) * nz;
    }

    // ── Compact: atomicAdd to get a write slot in the SoA output buffer ──
    int mySlot = atomicAdd(&survivorCounters[speciesIdx], 1);
    int writeIdx = speciesOffsets[speciesIdx] + mySlot;

    outputSoA->u[writeIdx] = pclU;
    outputSoA->v[writeIdx] = pclV;
    outputSoA->w[writeIdx] = pclW;
    outputSoA->q[writeIdx] = pclQ;
    outputSoA->x[writeIdx] = pclX;
    outputSoA->y[writeIdx] = pclY;
    outputSoA->z[writeIdx] = pclZ;
    outputSoA->t[writeIdx] = pclT;
}


// ═══════════════════════════════════════════════════════════════
//  Simple GPU-compatible hash-based PRNG (xorshift32)
//  Portable across CUDA and HIP — no vendor-specific intrinsics.
// ═══════════════════════════════════════════════════════════════

__device__ inline uint32_t planetRngHash(uint32_t seed)
{
    seed ^= seed << 13;
    seed ^= seed >> 17;
    seed ^= seed << 5;
    return seed;
}

/// Return a uniform value in (0, 1).  Precision is limited by the
/// mantissa width of cudaCommonType (24 bits for float, 32 bits used
/// here map into the 52-bit double mantissa).
__device__ inline cudaCommonType planetRngUniform(uint32_t& state)
{
    state = planetRngHash(state);
    return (cudaCommonType)(state) * (cudaCommonType)(1.0 / 4294967296.0);
}


// ═══════════════════════════════════════════════════════════════
//  Fused DIFFUSE reflect + compact kernel  (isotropic scattering)
//  Branch-free hemisphere sampling — no rejection loop, no warp
//  divergence.  Portable across CUDA (warp 32) and HIP (wave 64).
// ═══════════════════════════════════════════════════════════════

/**
 * @brief Same structure as planetReflectCompactKernel, but instead of
 *        specular (mirror) reflection, the velocity is randomised to a
 *        uniform direction on the OUTWARD hemisphere while preserving
 *        the original speed |v|.
 *
 *        NOTE: this is NOT identical to the legacy CPU function
 *        rotateAndCountParticlesInsideSphere().  The legacy code samples
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
 * @param rngSeedBase  base seed for the per-thread PRNG (e.g. cycle number)
 */
__global__ void planetDiffuseCompactKernel(
    ParticleSoADevice** planetSoAArrs, int nElecSpecies,
    const int* speciesOffsets,
    const uint32_t* sortedGlobalIdx,
    const int* cutoffDevice,
    int totalElecPlanet,
    ParticleSoADevice* outputSoA,
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

    // Read the original planet particle from SoA
    ParticleSoADevice* srcSoA = planetSoAArrs[speciesIdx];
    cudaCommonType pclU = srcSoA->u[localIdx];
    cudaCommonType pclV = srcSoA->v[localIdx];
    cudaCommonType pclW = srcSoA->w[localIdx];
    cudaCommonType pclQ = srcSoA->q[localIdx];
    cudaCommonType pclX = srcSoA->x[localIdx];
    cudaCommonType pclY = srcSoA->y[localIdx];
    cudaCommonType pclZ = srcSoA->z[localIdx];
    cudaCommonType pclT = srcSoA->t[localIdx];

    // Compute original speed (invariant)
    const cudaCommonType Vmod = sqrt(pclU * pclU + pclV * pclV + pclW * pclW);

    // ── Initialise per-thread RNG ──
    uint32_t rngState = rngSeedBase ^ (uint32_t)((uint32_t)sortedIdx * 2654435761u + 1u);
    rngState = planetRngHash(rngState);
    if (rngState == 0u) rngState = 1u;  // xorshift32 absorbs at 0

    const cudaCommonType eps = sphereRadius * (cudaCommonType)5e-2;


    if (doSphere == 1) { // ── 3D ──
        const cudaCommonType dx = pclX - originX;
        const cudaCommonType dy = pclY - originY;
        const cudaCommonType dz = pclZ - originZ;
        cudaCommonType r = sqrt(dx * dx + dy * dy + dz * dz);
        if (r < (cudaCommonType)1e-30) r = (cudaCommonType)1e-30;
        const cudaCommonType invr = (cudaCommonType)1.0 / r;

        // Outward unit normal  n = (nx, ny, nz)
        const cudaCommonType nx = dx * invr;
        const cudaCommonType ny = dy * invr;
        const cudaCommonType nz = dz * invr;

        // Random direction on the outward hemisphere (no rejection)
        const cudaCommonType cosTheta = planetRngUniform(rngState);
        const cudaCommonType sinTheta = sqrt((cudaCommonType)1.0 - cosTheta * cosTheta);
        const cudaCommonType phi = TWO_PI * planetRngUniform(rngState);
        const cudaCommonType cphi = cos(phi);
        const cudaCommonType sphi = sin(phi);

        // Velocity in LOCAL frame  (z_local = n)
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
        // t2 = cross(n, t1)
        const cudaCommonType t2x = ny * t1z - nz * t1y;
        const cudaCommonType t2y = nz * t1x - nx * t1z;
        const cudaCommonType t2z = nx * t1y - ny * t1x;

        // Rotate to global frame:  v = vl_x * t1 + vl_y * t2 + vl_z * n
        pclU = vl_x * t1x + vl_y * t2x + vl_z * nx;
        pclV = vl_x * t1y + vl_y * t2y + vl_z * ny;
        pclW = vl_x * t1z + vl_y * t2z + vl_z * nz;

        pclX = originX + (sphereRadius + eps) * nx;
        pclY = originY + (sphereRadius + eps) * ny;
        pclZ = originZ + (sphereRadius + eps) * nz;

    } else if (doSphere == 2) { // ── 2D (XZ plane) ──
        const cudaCommonType dx = pclX - originX;
        const cudaCommonType dz = pclZ - originZ;
        cudaCommonType r = sqrt(dx * dx + dz * dz);
        if (r < (cudaCommonType)1e-30) r = (cudaCommonType)1e-30;
        const cudaCommonType invr = (cudaCommonType)1.0 / r;

        const cudaCommonType nx = dx * invr;
        const cudaCommonType nz = dz * invr;

        const cudaCommonType alpha = ((cudaCommonType)3.14159265358979323846)
                                     * (planetRngUniform(rngState) - (cudaCommonType)0.5);
        const cudaCommonType ca = cos(alpha);
        const cudaCommonType sa = sin(alpha);
        pclU = Vmod * (ca * nx - sa * nz);
        pclW = Vmod * (ca * nz + sa * nx);

        pclX = originX + (sphereRadius + eps) * nx;
        pclZ = originZ + (sphereRadius + eps) * nz;
    }

    // ── Compact: atomicAdd to get a write slot in the SoA output buffer ──
    int mySlot = atomicAdd(&survivorCounters[speciesIdx], 1);
    int writeIdx = speciesOffsets[speciesIdx] + mySlot;

    outputSoA->u[writeIdx] = pclU;
    outputSoA->v[writeIdx] = pclV;
    outputSoA->w[writeIdx] = pclW;
    outputSoA->q[writeIdx] = pclQ;
    outputSoA->x[writeIdx] = pclX;
    outputSoA->y[writeIdx] = pclY;
    outputSoA->z[writeIdx] = pclZ;
    outputSoA->t[writeIdx] = pclT;
}
