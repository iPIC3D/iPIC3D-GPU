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
 * @brief GPU particle container — dual AoS+SoA during transition, pure SoA at end.
 *
 * Inherits arrayCUDA<SpeciesParticle> for backward compatibility (AoS).
 * Also maintains 8 separate device arrays (SoA) via ParticleSoADevice.
 * New code should use the SoA accessors (getU(), getV(), ...).
 * The AoS parent (getpcls()) will be removed once all kernels are migrated.
 */
class particleArrayCUDA : public arrayCUDA<SpeciesParticle, 32>
{
private:
    uint32_t initialNOP;
    ParticleSoADevice soa;  // SoA device pointers + count + capacity

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

    /**
     * @brief Scatter AoS device data into SoA device arrays via a temporary host buffer.
     *
     * Called once at construction to populate SoA from the AoS parent.
     */
    __host__ void scatterAoSToSoA(uint32_t nop, cudaStream_t deviceStream) {
        if (nop == 0) return;
        // Allocate temporary pinned host buffer, copy AoS from device
        SpeciesParticle* hostBuf = nullptr;
        cudaErrChk(cudaHostAlloc(&hostBuf, nop * sizeof(SpeciesParticle), cudaHostAllocDefault));
        cudaErrChk(cudaMemcpyAsync(hostBuf, getArray(), nop * sizeof(SpeciesParticle), cudaMemcpyDeviceToHost, deviceStream));
        cudaErrChk(cudaStreamSynchronize(deviceStream));

        // Scatter into 8 pinned staging arrays
        cudaParticleType* stageBuf = nullptr;
        cudaErrChk(cudaHostAlloc(&stageBuf, nop * sizeof(cudaParticleType), cudaHostAllocDefault));

        auto copyField = [&](cudaParticleType* dst, auto getter) {
            for (uint32_t i = 0; i < nop; i++) stageBuf[i] = getter(hostBuf[i]);
            cudaErrChk(cudaMemcpyAsync(dst, stageBuf, nop * sizeof(cudaParticleType), cudaMemcpyHostToDevice, deviceStream));
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
        cudaErrChk(cudaFreeHost(hostBuf));
    }

public:
    /**
     * @brief Construct from host Particles3D — populates both AoS and SoA on device.
     */
    __host__ particleArrayCUDA(Particles3D* p3D, cudaTypeSingle expand = 1.2, cudaStream_t deviceStream = 0)
        : arrayCUDA(p3D->get_pclptr(0), p3D->getNOP(), expand)
        , initialNOP(0)
        , soa{}
    {
        assignStream(deviceStream);
        uint32_t nop = p3D->getNOP();
        uint32_t cap = roundUpSoA(static_cast<uint32_t>(nop * expand));
        allocateSoA(cap);
        soa.nop = nop;
        scatterAoSToSoA(nop, deviceStream);
    }

    __host__ ~particleArrayCUDA() {
        freeSoA();
    }

    __host__ particleArrayCUDA* copyToDevice() override {
        particleArrayCUDA* ptr = nullptr;
        cudaErrChk(cudaMalloc((void**)&ptr, sizeof(particleArrayCUDA)));
        cudaErrChk(cudaMemcpyAsync(ptr, this, sizeof(particleArrayCUDA), cudaMemcpyDefault, stream));
        cudaErrChk(cudaStreamSynchronize(stream));
        return ptr;
    }

    // ── Particle count (unified: keeps AoS parent and SoA in sync) ──

    __host__ __device__ __forceinline__ uint32_t getNOP()     { return getNOE(); }
    __host__ __device__ __forceinline__ uint32_t getSoANOP()  { return soa.nop; }

    __host__ void setNOE(uint32_t val) {
        arrayCUDA::setNOE(val);
        soa.nop = val;
    }

    __host__ __device__ void setInitialNOP(uint32_t n) { initialNOP = n; }
    __host__ __device__ __forceinline__ uint32_t getInitialNOP()       { return initialNOP; }

    // ── SoA accessors (new — use these in migrated kernels) ──

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

    // ── Expand: grows both AoS and SoA allocations ──

    __host__ uint32_t expand(uint32_t targetSize, cudaStream_t deviceStream) {
        // Expand AoS parent
        arrayCUDA::expand(targetSize, deviceStream);
        // Expand SoA if needed
        if (targetSize > soa.capacity) {
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
            soa.capacity = newCap;
        }
        return getSize();
    }

    // ── Legacy AoS accessors (kept during transition, will be removed) ──

    __host__ __device__ __forceinline__ SpeciesParticle* getpcls()              { return getArray(); }
    __host__ __device__ __forceinline__ SpeciesParticle* getpcl(uint32_t index) { return getElement(index); }
};

/**
 * @brief Device kernel: scatter AoS particles into SoA arrays for a given range.
 *
 * Used after H→D AoS copies (incoming MPI particles, merge re-upload) to
 * populate the SoA arrays that all GPU compute kernels now read from.
 * Reads from getpcls()[offset..offset+count-1], writes to getU/V/W/Q/X/Y/Z/T
 * at the same indices.
 *
 * @param pclsArray  device-resident particleArrayCUDA (has both AoS and SoA pointers)
 * @param offset     first particle index to scatter
 * @param count      number of particles to scatter
 */
__global__ void scatterAoSToSoAKernel(particleArrayCUDA* pclsArray,
                                       uint32_t offset, uint32_t count);

/**
 * @brief Device kernel: gather SoA fields back into AoS particles for a given range.
 *
 * Used before D→H AoS copies (output staging, merge-prep download) to
 * synchronise the AoS parent from the authoritative SoA arrays.
 * Reads from getU/V/W/Q/X/Y/Z/T, writes to getpcls()[offset..offset+count-1].
 *
 * @param pclsArray  device-resident particleArrayCUDA (has both AoS and SoA pointers)
 * @param offset     first particle index to gather
 * @param count      number of particles to gather
 */
__global__ void gatherSoAToAoSKernel(particleArrayCUDA* pclsArray,
                                      uint32_t offset, uint32_t count);

#endif