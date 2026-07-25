#ifndef PARTICLE_ID_GENERATOR_CUH
#define PARTICLE_ID_GENERATOR_CUH

#include <algorithm>
#include <cstdint>
#include <limits>
#include <stdexcept>

#include <mpi.h>

#include "cudaTypeDef.cuh"

/**
 * @brief Copyable host/device handle for particle-ID allocation.
 *
 * IDs are rank/species-striped:
 *
 *   id = (sequence * mpiSize + mpiRank) * speciesCount + speciesIndex
 *
 * UINT64_MAX is reserved as the invalid/missing-ID sentinel.
 */
struct ParticleIDGenerator {
  using counter_type = unsigned long long;

  counter_type* nextSequence = nullptr;
  int mpiRank = 0;
  int mpiSize = 1;
  int speciesIndex = 0;
  int speciesCount = 1;

  __host__ __device__ cudaPclType_ID
  idFromSequence(counter_type sequence) const {
    return static_cast<cudaPclType_ID>(
        (sequence * static_cast<counter_type>(mpiSize) +
         static_cast<counter_type>(mpiRank)) *
            static_cast<counter_type>(speciesCount) +
        static_cast<counter_type>(speciesIndex));
  }

#if defined(__CUDACC__) || defined(__HIPCC__)
  __device__ __forceinline__ counter_type
  reserveSequenceBlock(counter_type count) const {
    return atomicAdd(nextSequence, count);
  }

  __device__ __forceinline__ cudaPclType_ID generateID() const {
    return idFromSequence(reserveSequenceBlock(1));
  }
#endif
};

/**
 * @brief Host-side owner for one species' device-visible ID counter.
 */
class ParticleIDGeneratorState {
public:
  using counter_type = ParticleIDGenerator::counter_type;

  ParticleIDGeneratorState() = default;

  ParticleIDGeneratorState(const ParticleIDGeneratorState&) = delete;
  ParticleIDGeneratorState& operator=(const ParticleIDGeneratorState&) = delete;

  ParticleIDGeneratorState(ParticleIDGeneratorState&& other) noexcept {
    moveFrom(other);
  }

  ParticleIDGeneratorState&
  operator=(ParticleIDGeneratorState&& other) noexcept {
    if (this != &other) {
      release();
      moveFrom(other);
    }
    return *this;
  }

  ~ParticleIDGeneratorState() { release(); }

  void initialize(int mpiRank, int mpiSize, int speciesIndex,
                  int speciesCount) {
    mpiRank_ = mpiRank;
    mpiSize_ = mpiSize > 0 ? mpiSize : 1;
    speciesCount_ = speciesCount > 0 ? speciesCount : 1;
    if (speciesIndex < 0 || speciesIndex >= speciesCount_)
      throw std::runtime_error(
          "Particle ID generator received invalid species index");
    speciesIndex_ = speciesIndex;
    hostNextSequence_ = 0;
  }

  void ensureDeviceCounter(cudaStream_t stream = 0) {
    if (deviceNextSequence_ == nullptr)
      cudaErrChk(cudaMalloc(&deviceNextSequence_, sizeof(counter_type)));
    syncDeviceCounter(stream);
  }

  void seedFromExistingIDs(const cudaPclType_ID* ids, int count, MPI_Comm comm,
                           cudaStream_t stream = 0) {
    int localHasIDs = 0;
    cudaPclType_ID localMax = 0;
    for (int i = 0; i < count; ++i) {
      const cudaPclType_ID id = ids[i];
      if (id != PARTICLE_ID_INVALID) {
        localHasIDs = 1;
        localMax = std::max(localMax, id);
      }
    }

    int globalHasIDs = 0;
    MPI_Allreduce(&localHasIDs, &globalHasIDs, 1, MPI_INT, MPI_MAX, comm);

    cudaPclType_ID globalMax = 0;
    MPI_Allreduce(&localMax, &globalMax, 1, MPI_UINT64_T, MPI_MAX, comm);

    counter_type nextSequence = 0;
    if (globalHasIDs) {
      const cudaPclType_ID rankSequence =
          (globalMax - static_cast<cudaPclType_ID>(speciesIndex_)) /
          static_cast<cudaPclType_ID>(speciesCount_);
      nextSequence = static_cast<counter_type>(
                         rankSequence / static_cast<cudaPclType_ID>(mpiSize_)) +
                     1;
      if (nextSequence > sequenceCapacity())
        throw std::runtime_error(
            "Particle ID generator exceeded uint64 ID range");
    }

    hostNextSequence_ = std::max(hostNextSequence_, nextSequence);
    if (deviceNextSequence_ != nullptr)
      syncDeviceCounter(stream);
  }

  counter_type reserveHostSequenceBlock(counter_type count,
                                        cudaStream_t stream = 0) {
    const counter_type base = hostNextSequence_;
    const counter_type capacity = sequenceCapacity();
    if (hostNextSequence_ > capacity || count > capacity - hostNextSequence_)
      throw std::runtime_error(
          "Particle ID generator exhausted uint64 ID range");
    hostNextSequence_ += count;
    if (deviceNextSequence_ != nullptr)
      syncDeviceCounter(stream);
    return base;
  }

  cudaPclType_ID generateHostID(cudaStream_t stream = 0) {
    return idFromSequence(reserveHostSequenceBlock(1, stream));
  }

  __host__ cudaPclType_ID idFromSequence(counter_type sequence) const {
    return handle().idFromSequence(sequence);
  }

  ParticleIDGenerator handle() const {
    ParticleIDGenerator h;
    h.nextSequence = deviceNextSequence_;
    h.mpiRank = mpiRank_;
    h.mpiSize = mpiSize_;
    h.speciesIndex = speciesIndex_;
    h.speciesCount = speciesCount_;
    return h;
  }

private:
  static constexpr cudaPclType_ID maxValidID() {
    return PARTICLE_ID_INVALID - static_cast<cudaPclType_ID>(1);
  }

  counter_type sequenceCapacity() const {
    const cudaPclType_ID species = static_cast<cudaPclType_ID>(speciesIndex_);
    if (species > maxValidID())
      return 0;
    const cudaPclType_ID maxRankSequence =
        (maxValidID() - species) / static_cast<cudaPclType_ID>(speciesCount_);
    const cudaPclType_ID rank = static_cast<cudaPclType_ID>(mpiRank_);
    if (rank > maxRankSequence)
      return 0;
    return static_cast<counter_type>((maxRankSequence - rank) /
                                     static_cast<cudaPclType_ID>(mpiSize_)) +
           1;
  }

  void syncDeviceCounter(cudaStream_t stream) {
    cudaErrChk(cudaMemcpyAsync(deviceNextSequence_, &hostNextSequence_,
                               sizeof(counter_type), cudaMemcpyHostToDevice,
                               stream));
    cudaErrChk(cudaStreamSynchronize(stream));
  }

  void release() {
    if (deviceNextSequence_ != nullptr) {
      cudaFree(deviceNextSequence_);
      deviceNextSequence_ = nullptr;
    }
  }

  void moveFrom(ParticleIDGeneratorState& other) {
    deviceNextSequence_ = other.deviceNextSequence_;
    hostNextSequence_ = other.hostNextSequence_;
    mpiRank_ = other.mpiRank_;
    mpiSize_ = other.mpiSize_;
    speciesIndex_ = other.speciesIndex_;
    speciesCount_ = other.speciesCount_;
    other.deviceNextSequence_ = nullptr;
    other.hostNextSequence_ = 0;
    other.mpiRank_ = 0;
    other.mpiSize_ = 1;
    other.speciesIndex_ = 0;
    other.speciesCount_ = 1;
  }

  counter_type* deviceNextSequence_ = nullptr;
  counter_type hostNextSequence_ = 0;
  int mpiRank_ = 0;
  int mpiSize_ = 1;
  int speciesIndex_ = 0;
  int speciesCount_ = 1;
};

#endif // PARTICLE_ID_GENERATOR_CUH
