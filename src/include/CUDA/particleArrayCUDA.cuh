#ifndef _PARTICLE_ARRAY_CUDA_H_
#define _PARTICLE_ARRAY_CUDA_H_

#include <stdexcept>
#include <iostream>
#include "cudaTypeDef.cuh"
#include "Particle.h"
#include "Particles3D.h"
#include "arrayCUDA.cuh"
#include "ParticleSoADevice.cuh"

/**
 * @brief GPU particle container — pure SoA.
 *
 * Owns 8 separate device arrays (one per particle field) via ParticleSoADevice.
 * All compute kernels (mover, moment, sort, merge, planet, data-analysis)
 * read and write through the SoA accessors getU(), getV(), ... getT().
 *
 * No persistent AoS allocation.  Small AoS staging buffers for H↔D transfer
 * of exchange / planet / exosphere particles live outside this class
 * (see incomingStagingHostPtr, exitingArray, planetArray in c_Solver).
 */
class particleArrayCUDA
{
private:
    uint32_t numberOfElement;   // current particle count
    uint32_t soaCapacity;       // allocated capacity (elements per SoA field)
    uint32_t initialNOP;
    ParticleSoADevice soa;      // 8 device pointers + nop + capacity
    cudaStream_t stream;

    static uint32_t roundUpSoA(uint32_t n) {
        constexpr uint32_t align = 32;
        return n + align - 1 - ((n + align - 1) % align);
    }

    __host__ void allocateSoA(uint32_t cap) {
        soa.capacity = cap;
        cudaErrChk(cudaMalloc(&soa.u, cap * sizeof(cudaParticleType)));
        cudaErrChk(cudaMalloc(&soa.v, cap * sizeof(cudaParticleType)));
        cudaErrChk(cudaMalloc(&soa.w, cap * sizeof(cudaParticleType)));
        cudaErrChk(cudaMalloc(&soa.q, cap * sizeof(cudaParticleType)));
        cudaErrChk(cudaMalloc(&soa.x, cap * sizeof(cudaParticleType)));
        cudaErrChk(cudaMalloc(&soa.y, cap * sizeof(cudaParticleType)));
        cudaErrChk(cudaMalloc(&soa.z, cap * sizeof(cudaParticleType)));
        cudaErrChk(cudaMalloc(&soa.t, cap * sizeof(cudaParticleType)));
    }

    __host__ void freeSoA() {
        cudaFree(soa.u); cudaFree(soa.v); cudaFree(soa.w); cudaFree(soa.q);
        cudaFree(soa.x); cudaFree(soa.y); cudaFree(soa.z); cudaFree(soa.t);
        soa = {};
    }

public:
    /**
     * @brief Construct from host Particles3D (SoA-mode) — direct SoA H→D.
     *
     * The source Particles3D should be in SoA storage mode (the default for
     * outputPart[] used at init time).  No device AoS buffer is allocated.
     * If the source is in AoS mode, scatters on host into SoA staging arrays.
     */
    __host__ particleArrayCUDA(Particles3D* p3D, cudaTypeSingle expand = 1.2, cudaStream_t deviceStream = 0)
        : numberOfElement(p3D->getNOP())
        , soaCapacity(0)
        , initialNOP(0)
        , soa{}
        , stream(deviceStream)
    {
        const uint32_t nop = p3D->getNOP();
        const uint32_t cap = roundUpSoA(static_cast<uint32_t>(nop * expand));
        soaCapacity = cap;
        allocateSoA(cap);
        soa.nop = nop;

        if (p3D->isSoAMode()) {
            // Direct SoA H→D: 8 memcpy from host SoA vectors
            if (nop > 0) {
                const size_t bytes = nop * sizeof(double);
                cudaErrChk(cudaMemcpyAsync(soa.u, p3D->getUall(), bytes, cudaMemcpyDefault, deviceStream));
                cudaErrChk(cudaMemcpyAsync(soa.v, p3D->getVall(), bytes, cudaMemcpyDefault, deviceStream));
                cudaErrChk(cudaMemcpyAsync(soa.w, p3D->getWall(), bytes, cudaMemcpyDefault, deviceStream));
                cudaErrChk(cudaMemcpyAsync(soa.q, p3D->getQall(), bytes, cudaMemcpyDefault, deviceStream));
                cudaErrChk(cudaMemcpyAsync(soa.x, p3D->getXall(), bytes, cudaMemcpyDefault, deviceStream));
                cudaErrChk(cudaMemcpyAsync(soa.y, p3D->getYall(), bytes, cudaMemcpyDefault, deviceStream));
                cudaErrChk(cudaMemcpyAsync(soa.z, p3D->getZall(), bytes, cudaMemcpyDefault, deviceStream));
                cudaErrChk(cudaMemcpyAsync(soa.t, p3D->getParticleIDall(), bytes, cudaMemcpyDefault, deviceStream));
                cudaErrChk(cudaStreamSynchronize(deviceStream));
            }
        } else {
            // AoS-mode source: scatter on host via temporary pinned staging buffer.
            if (nop > 0) {
                const SpeciesParticle* src = p3D->getAoSDataPtr();
                const size_t fieldBytes = nop * sizeof(cudaParticleType);
                cudaParticleType* stageBuf = nullptr;
                cudaErrChk(cudaHostAlloc(&stageBuf, fieldBytes, cudaHostAllocDefault));

                auto copyField = [&](cudaParticleType* dst, auto getter) {
                    for (uint32_t i = 0; i < nop; i++) stageBuf[i] = getter(src[i]);
                    cudaErrChk(cudaMemcpyAsync(dst, stageBuf, fieldBytes, cudaMemcpyHostToDevice, deviceStream));
                    cudaErrChk(cudaStreamSynchronize(deviceStream));
                };
                copyField(soa.u, [](const SpeciesParticle& p){ return p.get_u(); });
                copyField(soa.v, [](const SpeciesParticle& p){ return p.get_v(); });
                copyField(soa.w, [](const SpeciesParticle& p){ return p.get_w(); });
                copyField(soa.q, [](const SpeciesParticle& p){ return p.get_q(); });
                copyField(soa.x, [](const SpeciesParticle& p){ return p.get_x(); });
                copyField(soa.y, [](const SpeciesParticle& p){ return p.get_y(); });
                copyField(soa.z, [](const SpeciesParticle& p){ return p.get_z(); });
                copyField(soa.t, [](const SpeciesParticle& p){ return p.get_t(); });

                cudaErrChk(cudaFreeHost(stageBuf));
            }
        }
    }

    __host__ ~particleArrayCUDA() {
        freeSoA();
    }

    __host__ particleArrayCUDA* copyToDevice() {
        particleArrayCUDA* ptr = nullptr;
        cudaErrChk(cudaMalloc((void**)&ptr, sizeof(particleArrayCUDA)));
        cudaErrChk(cudaMemcpyAsync(ptr, this, sizeof(particleArrayCUDA), cudaMemcpyDefault, stream));
        cudaErrChk(cudaStreamSynchronize(stream));
        return ptr;
    }

    // ── Particle count ──

    __host__ __device__ __forceinline__ uint32_t getNOP()  const { return numberOfElement; }
    __host__ __device__ __forceinline__ uint32_t getNOE()  const { return numberOfElement; }

    __host__ void setNOE(uint32_t val) {
        numberOfElement = val;
        soa.nop = val;
    }

    __host__ __device__ void setInitialNOP(uint32_t n) { initialNOP = n; }
    __host__ __device__ __forceinline__ uint32_t getInitialNOP() const { return initialNOP; }

    // ── Capacity ──

    /** Allocated capacity per SoA field (in elements). */
    __host__ __device__ __forceinline__ uint32_t getCapacity() const { return soaCapacity; }
    /** Alias kept for callers that used getSize() to mean "allocated capacity". */
    __host__ __device__ __forceinline__ uint32_t getSize()     const { return soaCapacity; }

    // ── SoA accessors ──

    __host__ __device__ __forceinline__ cudaParticleType* getU() { return soa.u; }
    __host__ __device__ __forceinline__ cudaParticleType* getV() { return soa.v; }
    __host__ __device__ __forceinline__ cudaParticleType* getW() { return soa.w; }
    __host__ __device__ __forceinline__ cudaParticleType* getQ() { return soa.q; }
    __host__ __device__ __forceinline__ cudaParticleType* getX() { return soa.x; }
    __host__ __device__ __forceinline__ cudaParticleType* getY() { return soa.y; }
    __host__ __device__ __forceinline__ cudaParticleType* getZ() { return soa.z; }
    __host__ __device__ __forceinline__ cudaParticleType* getT() { return soa.t; }

    __host__ __device__ __forceinline__ ParticleSoADevice*       getSoA()       { return &soa; }
    __host__ __device__ __forceinline__ const ParticleSoADevice* getSoA() const { return &soa; }

    // ── Stream ──

    __host__ void assignStream(cudaStream_t s) { stream = s; }
    __host__ cudaStream_t getStream() const { return stream; }

    // ── Expand: grows SoA allocations only ──

    __host__ uint32_t expand(uint32_t targetSize, cudaStream_t deviceStream) {
        if (targetSize <= soaCapacity) return soaCapacity;
        uint32_t newCap = roundUpSoA(targetSize);
        auto expandField = [&](cudaParticleType*& field) {
            cudaParticleType* newPtr = nullptr;
            cudaErrChk(cudaMalloc(&newPtr, newCap * sizeof(cudaParticleType)));
            if (soa.nop > 0) {
                cudaErrChk(cudaMemcpyAsync(newPtr, field, soa.nop * sizeof(cudaParticleType), cudaMemcpyDefault, deviceStream));
            }
            cudaErrChk(cudaStreamSynchronize(deviceStream));
            cudaErrChk(cudaFree(field));
            field = newPtr;
        };
        expandField(soa.u); expandField(soa.v); expandField(soa.w); expandField(soa.q);
        expandField(soa.x); expandField(soa.y); expandField(soa.z); expandField(soa.t);
        soaCapacity = newCap;
        soa.capacity = newCap;
        return soaCapacity;
    }
};

/**
 * @brief Device kernel: scatter AoS particles from a staging buffer into the
 *        main SoA arrays of a particleArrayCUDA at a given offset.
 *
 * Used after H→D AoS copies (incoming MPI particles, repopulated, exosphere)
 * to populate the SoA arrays that all GPU compute kernels read from.
 *
 * @param aosStagingBuf  device pointer to AoS staging buffer (source)
 * @param pclsArray      device-resident particleArrayCUDA (destination SoA)
 * @param destOffset     first particle index in SoA to write to
 * @param count          number of particles to scatter
 */
__global__ void scatterAoSToSoAKernel(const SpeciesParticle* __restrict__ aosStagingBuf,
                                       particleArrayCUDA* pclsArray,
                                       uint32_t destOffset, uint32_t count);

#endif