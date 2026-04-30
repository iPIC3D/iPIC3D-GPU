#ifndef _PARTICLE_ARRAY_SOA_VIEW_H_
#define _PARTICLE_ARRAY_SOA_VIEW_H_

#include "cudaTypeDef.cuh"

class particleArrayCUDA; // forward declaration

/**
 * @brief Lightweight SoA view for data analysis kernels.
 *
 * Template params:
 *   T — element type (e.g. double, float)
 *   N — number of fields (e.g. 4 for u,v,w,q)
 *
 * Two construction modes:
 *   1. Owning (unit tests):     particleArraySoAView(nop) — allocates N device arrays
 *   2. Non-owning (production): borrowFrom(particleArrayCUDA*) — zero-copy pointer borrow
 */
template<typename T, int N>
class particleArraySoAView {
    static_assert(N >= 1 && N <= 8, "N must be between 1 and 8");

private:
    int nop;
    T* ptrs[N];
    bool owning;

    __host__ void allocateMemory() {
        for (int i = 0; i < N; i++)
            cudaErrChk(cudaMalloc(&ptrs[i], nop * sizeof(T)));
    }

    __host__ void freeMemory() {
        for (int i = 0; i < N; i++)
            cudaFree(ptrs[i]);
    }

public:
    // ── Non-owning default (for later borrowFrom) ──
    __host__ particleArraySoAView() : nop(0), ptrs{}, owning(false) {}

    // ── Owning constructor (unit tests) ──
    __host__ particleArraySoAView(int nop, cudaStream_t stream = 0)
        : nop(nop), ptrs{}, owning(true) {
        allocateMemory();
    }

    __host__ ~particleArraySoAView() {
        if (owning) freeMemory();
    }

    // Non-copyable
    particleArraySoAView(const particleArraySoAView&) = delete;
    particleArraySoAView& operator=(const particleArraySoAView&) = delete;

    // ── Borrow from particleArrayCUDA (zero-copy, production path) ──
    // Borrows fields [0..N-1] in order: u, v, w, q, x, y, z, t.
    // Declared here, defined in the .cu file where particleArrayCUDA is complete.
    __host__ void borrowFrom(particleArrayCUDA* pclArray);

    // ── Accessors ──
    __host__ __device__ T*  getElement(int i) const { return ptrs[i]; }
    __host__ __device__ int getNOP()          const { return nop; }
};

#endif
