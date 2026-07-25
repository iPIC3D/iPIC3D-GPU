#ifndef PARTICLE_ARRAY_CUDA_CUH
#define PARTICLE_ARRAY_CUDA_CUH

#include "Particle.h"
#include "ParticleIDGenerator.cuh"
#include "ParticleSoADevice.cuh"
#include "ParticleSoAHost.h"
#include "arrayCUDA.cuh"
#include "cudaTypeDef.cuh"
#include <iostream>
#include <stdexcept>
#include <type_traits>

/**
 * @brief GPU particle container — pure SoA.
 *
 * Owns separate device arrays for particle fields via ParticleSoADevice.
 * The ID array is allocated only when particle tracking is enabled.
 *
 * No persistent AoS allocation.  Small AoS staging buffers for H↔D transfer
 * of exchange / planet / exosphere particles live outside this class
 * (see incomingStagingHostPtr, exitingArray, planetArray in c_Solver).
 */
class particleArrayCUDA {
private:
  uint32_t initialNOP;
  ParticleSoADevice soa; // device field pointers + metadata
  cudaStream_t stream;

  static uint32_t roundUpSoA(uint32_t size) {
    constexpr uint32_t align = 64;
    return size + align - 1 - ((size + align - 1) % align);
  }

  __host__ void allocateSoA(uint32_t cap, bool trackParticleID) {
    soa.capacity = cap;
    soa.trackParticleID = trackParticleID;
    cudaErrChk(cudaMalloc(&soa.u, cap * sizeof(cudaPclType_U)));
    cudaErrChk(cudaMalloc(&soa.v, cap * sizeof(cudaPclType_V)));
    cudaErrChk(cudaMalloc(&soa.w, cap * sizeof(cudaPclType_W)));
    cudaErrChk(cudaMalloc(&soa.q, cap * sizeof(cudaPclType_Q)));
    cudaErrChk(cudaMalloc(&soa.x, cap * sizeof(cudaPclType_X)));
    cudaErrChk(cudaMalloc(&soa.y, cap * sizeof(cudaPclType_Y)));
    cudaErrChk(cudaMalloc(&soa.z, cap * sizeof(cudaPclType_Z)));
    if (trackParticleID)
      cudaErrChk(cudaMalloc(&soa.id, cap * sizeof(cudaPclType_ID)));
    else
      soa.id = nullptr;
  }

  __host__ void freeSoA() {
    cudaFree(soa.u);
    cudaFree(soa.v);
    cudaFree(soa.w);
    cudaFree(soa.q);
    cudaFree(soa.x);
    cudaFree(soa.y);
    cudaFree(soa.z);
    cudaFree(soa.id);
    soa = {};
  }

public:
  /**
   * @brief Construct from ParticleSoAHost — direct SoA H→D.
   * @param pSoA Source host-side SoA container.
   * @param expand Capacity growth factor applied to the initial allocation.
   * @param deviceStream CUDA stream used for the initial H2D copies.
   */
  __host__ particleArrayCUDA(ParticleSoAHost* pSoA, cudaTypeSingle expand = 1.2,
                             cudaStream_t deviceStream = 0)
      : initialNOP(0), soa{}, stream(deviceStream) {
    const uint32_t nop = pSoA->getNOP();
    const uint32_t cap = roundUpSoA(static_cast<uint32_t>(nop * expand));
    allocateSoA(cap, pSoA->tracksParticleID());
    soa.nop = nop;

    if (nop > 0) {
      // ParticleSoAHost stores typed SoA fields; copy each field by its own
      // element size.
      cudaErrChk(cudaMemcpyAsync(soa.u, pSoA->getUall(),
                                 nop * sizeof(cudaPclType_U), cudaMemcpyDefault,
                                 deviceStream));
      cudaErrChk(cudaMemcpyAsync(soa.v, pSoA->getVall(),
                                 nop * sizeof(cudaPclType_V), cudaMemcpyDefault,
                                 deviceStream));
      cudaErrChk(cudaMemcpyAsync(soa.w, pSoA->getWall(),
                                 nop * sizeof(cudaPclType_W), cudaMemcpyDefault,
                                 deviceStream));
      cudaErrChk(cudaMemcpyAsync(soa.q, pSoA->getQall(),
                                 nop * sizeof(cudaPclType_Q), cudaMemcpyDefault,
                                 deviceStream));
      cudaErrChk(cudaMemcpyAsync(soa.x, pSoA->getXall(),
                                 nop * sizeof(cudaPclType_X), cudaMemcpyDefault,
                                 deviceStream));
      cudaErrChk(cudaMemcpyAsync(soa.y, pSoA->getYall(),
                                 nop * sizeof(cudaPclType_Y), cudaMemcpyDefault,
                                 deviceStream));
      cudaErrChk(cudaMemcpyAsync(soa.z, pSoA->getZall(),
                                 nop * sizeof(cudaPclType_Z), cudaMemcpyDefault,
                                 deviceStream));
      if (soa.trackParticleID)
        cudaErrChk(cudaMemcpyAsync(soa.id, pSoA->getParticleIDall(),
                                   nop * sizeof(cudaPclType_ID),
                                   cudaMemcpyDefault, deviceStream));
      cudaErrChk(cudaStreamSynchronize(deviceStream));
    }
  }

  __host__ ~particleArrayCUDA() { freeSoA(); }

  /**
   * @brief Copy this host-side metadata object to device memory.
   * @return Device pointer to the copied metadata object.
   */
  __host__ particleArrayCUDA* copyToDevice() {
    particleArrayCUDA* ptr = nullptr;
    cudaErrChk(cudaMalloc((void**)&ptr, sizeof(particleArrayCUDA)));
    cudaErrChk(cudaMemcpyAsync(ptr, this, sizeof(particleArrayCUDA),
                               cudaMemcpyDefault, stream));
    cudaErrChk(cudaStreamSynchronize(stream));
    return ptr;
  }

  // ── Particle count ──

  __host__ __device__ __forceinline__ uint32_t getNOP() const {
    return soa.nop;
  }
  __host__ __device__ __forceinline__ uint32_t getNOE() const {
    return soa.nop;
  }

  __host__ void setNOE(uint32_t val) { soa.nop = val; }

  __host__ __device__ void setInitialNOP(uint32_t n) { initialNOP = n; }
  __host__ __device__ __forceinline__ uint32_t getInitialNOP() const {
    return initialNOP;
  }

  // ── Capacity ──

  /** Allocated capacity per SoA field (in elements). */
  __host__ __device__ __forceinline__ uint32_t getCapacity() const {
    return soa.capacity;
  }
  /** Alias kept for callers that used getSize() to mean "allocated capacity".
   */
  __host__ __device__ __forceinline__ uint32_t getSize() const {
    return soa.capacity;
  }
  __host__ __device__ __forceinline__ bool tracksParticleID() const {
    return soa.trackParticleID;
  }

  // ── SoA accessors ──

  __host__ __device__ __forceinline__ cudaPclType_U* getU() { return soa.u; }
  __host__ __device__ __forceinline__ cudaPclType_V* getV() { return soa.v; }
  __host__ __device__ __forceinline__ cudaPclType_W* getW() { return soa.w; }
  __host__ __device__ __forceinline__ cudaPclType_Q* getQ() { return soa.q; }
  __host__ __device__ __forceinline__ cudaPclType_X* getX() { return soa.x; }
  __host__ __device__ __forceinline__ cudaPclType_Y* getY() { return soa.y; }
  __host__ __device__ __forceinline__ cudaPclType_Z* getZ() { return soa.z; }
  __host__ __device__ __forceinline__ cudaPclType_ID* getID() { return soa.id; }

  __host__ __device__ __forceinline__ ParticleSoADevice* getSoA() {
    return &soa;
  }
  __host__ __device__ __forceinline__ const ParticleSoADevice* getSoA() const {
    return &soa;
  }

  // ── Stream ──

  __host__ void assignStream(cudaStream_t s) { stream = s; }
  __host__ cudaStream_t getStream() const { return stream; }

  /**
   * @brief Expand the SoA capacity while preserving the existing particles.
   * @param targetSize Requested capacity in elements.
   * @param deviceStream CUDA stream used for the staged field copies.
   * @return Final allocated SoA capacity.
   */
  __host__ uint32_t expand(uint32_t targetSize, cudaStream_t deviceStream) {
    if (targetSize <= soa.capacity)
      return soa.capacity;
    uint32_t newCap = roundUpSoA(targetSize);
    uint32_t nop = soa.nop;

    // Batch 2 fields at a time: malloc pair -> enqueue both copies -> sync ->
    // free old pair -> repeat for next pair. Peak overhead = 2 extra field
    // buffers
    auto expandPair = [&](auto*& fieldA, auto*& fieldB) {
      using FA = std::remove_pointer_t<std::decay_t<decltype(fieldA)>>;
      using FB = std::remove_pointer_t<std::decay_t<decltype(fieldB)>>;
      FA* newA = nullptr;
      FB* newB = nullptr;
      cudaErrChk(cudaMalloc(&newA, newCap * sizeof(FA)));
      cudaErrChk(cudaMalloc(&newB, newCap * sizeof(FB)));
      if (nop > 0) {
        cudaErrChk(cudaMemcpyAsync(newA, fieldA, nop * sizeof(FA),
                                   cudaMemcpyDefault, deviceStream));
        cudaErrChk(cudaMemcpyAsync(newB, fieldB, nop * sizeof(FB),
                                   cudaMemcpyDefault, deviceStream));
      }
      cudaErrChk(cudaStreamSynchronize(deviceStream));
      cudaFree(fieldA);
      fieldA = newA;
      cudaFree(fieldB);
      fieldB = newB;
    };
    auto expandOne = [&](auto*& field) {
      using FT = std::remove_pointer_t<std::decay_t<decltype(field)>>;
      FT* newField = nullptr;
      cudaErrChk(cudaMalloc(&newField, newCap * sizeof(FT)));
      if (nop > 0) {
        cudaErrChk(cudaMemcpyAsync(newField, field, nop * sizeof(FT),
                                   cudaMemcpyDefault, deviceStream));
      }
      cudaErrChk(cudaStreamSynchronize(deviceStream));
      cudaFree(field);
      field = newField;
    };
    expandPair(soa.u, soa.v); // group 1: u, v
    expandPair(soa.w, soa.q); // group 2: w, q
    expandPair(soa.x, soa.y); // group 3: x, y
    if (soa.trackParticleID)
      expandPair(soa.z, soa.id); // group 4: z, id
    else
      expandOne(soa.z);

    soa.capacity = newCap;
    return soa.capacity;
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
 * @param particleIDGenerator Shared species-local ID generator
 */
__global__ void
scatterAoSToSoAKernel(const SpeciesParticle* __restrict__ aosStagingBuf,
                      particleArrayCUDA* pclsArray, uint32_t destOffset,
                      uint32_t count, ParticleIDGenerator particleIDGenerator);

#endif // PARTICLE_ARRAY_CUDA_CUH
