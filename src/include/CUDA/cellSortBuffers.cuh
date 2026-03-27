#ifndef _CELL_SORT_BUFFERS_H_
#define _CELL_SORT_BUFFERS_H_

#include "cudaTypeDef.cuh"
#include "gridCUDA.cuh"
#include "particleArrayCUDA.cuh"
#include <cstdint>
#include <algorithm>  // std::swap

// ============================================================================
// Prefix-sum kernel configuration (shared between host buffer sizing and kernels)
// ============================================================================
static constexpr int SORT_SCAN_BLOCK_SIZE        = 256;
static constexpr int SORT_SCAN_ELEMENTS_PER_BLOCK = 2 * SORT_SCAN_BLOCK_SIZE;  // 512

// ============================================================================
// CellSortBuffers — per-species temporary device arrays for counting sort
// ============================================================================
//
// Allocated once per species during initCUDA(), reused every sort cycle.
// Cell-related buffers are fixed (grid doesn't change); sorted_indices
// grows via ensure_capacity() when particle count exceeds current allocation.
//
struct CellSortBuffers {

    int*          cell_counts;        // [num_cells] — histogram
    int*          cell_offsets;       // [num_cells] — working copy of prefix sum (mutated by stage 3)
    int*          cell_start_offsets; // [num_cells] — preserved exclusive prefix sum
    unsigned int* sorted_indices;    // [max_particles] — scatter destination per particle
    int*          block_sums;        // [num_scan_blocks] — temp for multi-block prefix sum

    int      num_cells       = 0;
    int      num_scan_blocks = 0;
    uint32_t max_particles   = 0;
    bool     allocated       = false;

    // ── Allocate all buffers ──
    __host__ void allocate(uint32_t max_pcl, int ncells, cudaStream_t s) {
        num_cells     = ncells;
        max_particles = max_pcl;
        num_scan_blocks = (ncells + SORT_SCAN_ELEMENTS_PER_BLOCK - 1)
                        / SORT_SCAN_ELEMENTS_PER_BLOCK;

        cudaErrChk(cudaMalloc(&cell_counts,        ncells  * sizeof(int)));
        cudaErrChk(cudaMalloc(&cell_offsets,        ncells  * sizeof(int)));
        cudaErrChk(cudaMalloc(&cell_start_offsets,  ncells  * sizeof(int)));
        cudaErrChk(cudaMalloc(&sorted_indices,      max_pcl * sizeof(unsigned int)));
        cudaErrChk(cudaMalloc(&block_sums,          num_scan_blocks * sizeof(int)));

        allocated = true;
    }

    // ── Free all buffers ──
    __host__ void free() {
        if (!allocated) return;
        cudaFree(cell_counts);
        cudaFree(cell_offsets);
        cudaFree(cell_start_offsets);
        cudaFree(sorted_indices);
        cudaFree(block_sums);
        cell_counts = cell_offsets = cell_start_offsets = block_sums = nullptr;
        sorted_indices = nullptr;
        allocated = false;
    }

    // ── Zero histogram before each sort ──
    __host__ void zero_async(cudaStream_t s) {
        cudaErrChk(cudaMemsetAsync(cell_counts, 0,
                                   num_cells * sizeof(int), s));
    }

    // ── Grow sorted_indices if particle count exceeds capacity ──
    __host__ void ensure_capacity(uint32_t needed, cudaStream_t s) {
        if (needed <= max_particles) return;
        // Free old sorted_indices and re-allocate (contents are transient)
        cudaFree(sorted_indices);
        max_particles = needed;
        cudaErrChk(cudaMalloc(&sorted_indices, needed * sizeof(unsigned int)));
    }
};


// ============================================================================
// SortScratchBuffer — single device buffer for cycling pointer swap
// ============================================================================
//
// Sized to hold nop elements of the largest particle field type (double).
// After scatter-and-swap, the scratch pointer and an SoA field pointer
// exchange ownership.  The particleArrayCUDA destructor frees whatever
// pointers remain in soa.*, and CellSorter::free() frees scratch.ptr.
//
struct SortScratchBuffer {

    void*   ptr            = nullptr;
    size_t  capacity_bytes = 0;

    // ── Ensure scratch can hold at least needed_bytes ──
    __host__ void ensure_capacity(size_t needed_bytes) {
        if (needed_bytes <= capacity_bytes) return;
        if (ptr) cudaFree(ptr);
        cudaErrChk(cudaMalloc(&ptr, needed_bytes));
        capacity_bytes = needed_bytes;
    }

    // ── Free the scratch allocation ──
    __host__ void free() {
        if (ptr) { cudaFree(ptr); ptr = nullptr; capacity_bytes = 0; }
    }

    // ── Typed accessor for kernel launch ──
    template <typename T>
    T* as() { return static_cast<T*>(ptr); }
};


// ============================================================================
// CellSorter — per-species sort orchestrator
// ============================================================================
//
// Owns a CellSortBuffers + SortScratchBuffer for one species.
// sort() drives all 4 stages on the caller's CUDA stream — fully async,
// no implicit device synchronization.
//
// After sort(), the SoA field pointers in hostPtr->getSoA() have been swapped
// with the scratch buffer.  The caller must re-sync the host object to device:
//   cudaMemcpyAsync(pclsArrayCUDAPtr[species], pclsArrayHostPtr[species],
//                    sizeof(particleArrayCUDA), cudaMemcpyDefault, stream);
//
struct CellSorter {

    CellSortBuffers   buffers;
    SortScratchBuffer scratch;
    int  num_cells   = 0;
    bool initialized = false;

    // ── Pending state between enqueueSortAsync() and finishSort() ──
    uint32_t           pending_num_to_sort = 0;
    uint32_t           pending_nop         = 0;
    particleArrayCUDA* pending_hostPtr     = nullptr;
    bool               sort_pending        = false;

    // ── One-time initialization ──
    __host__ void init(const grid3DCUDA& grid, uint32_t initial_capacity, cudaStream_t s) {
        num_cells = grid.nxc * grid.nyc * grid.nzc;
        buffers.allocate(initial_capacity, num_cells, s);
        // Scratch must hold nop doubles (8 bytes each) for the cycling swap.
        scratch.ensure_capacity(initial_capacity * sizeof(double));
        initialized = true;
    }

    // ── Phase A: Resize buffers (MUST be called with no kernels in flight) ──
    //
    // Ensures sort buffers and scratch can hold the current particle count.
    // May call cudaMalloc/cudaFree — caller MUST guarantee no concurrent
    // kernel is using these buffers.
    __host__ void prepareBuffers(particleArrayCUDA* hostPtr, cudaStream_t s);

    // ── Phase B: Enqueue sort stages 1-3 (non-blocking) ──
    //
    // Enqueues histogram, prefix sum, and sorted-indices kernels on stream s.
    // Returns immediately (~20μs host time). Caller may overlap CPU work
    // (e.g. MPI exchange) while the GPU processes these stages.
    //
    // prepareBuffers() MUST have been called first.
    __host__ void enqueueSortAsync(particleArrayCUDA* hostPtr,
                                   const grid3DCUDA*  deviceGrid,
                                   uint32_t           num_to_sort,
                                   cudaStream_t       s);

    // ── Phase C: Complete sort (sync + scatter + pointer swap) ──
    //
    // Synchronizes stream s (waits for stages 1-3), then runs stage 4
    // (scatter per SoA array + cycling pointer swap on host).
    // After return, SoA pointers in hostPtr->getSoA() are updated.
    __host__ void finishSort(cudaStream_t s);

    // ── Convenience: all 3 phases in sequence (backward compat) ──
    __host__ void sort(particleArrayCUDA* hostPtr,
                       const grid3DCUDA*  deviceGrid,
                       uint32_t           num_to_sort,
                       cudaStream_t       s);

    // ── Accessors for the cell-aware moment kernel / merging kernel ──
    __host__ const int* getCellStartOffsets() const { return buffers.cell_start_offsets; }
    __host__ int*       getCellCounts()       const { return buffers.cell_counts; }
    __host__ int        getNumCells()         const { return num_cells; }

    // ── Release GPU memory ──
    __host__ void free() {
        buffers.free();
        scratch.free();
        initialized = false;
    }
};

#endif // _CELL_SORT_BUFFERS_H_
