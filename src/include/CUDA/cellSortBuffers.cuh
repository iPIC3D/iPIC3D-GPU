#ifndef _CELL_SORT_BUFFERS_H_
#define _CELL_SORT_BUFFERS_H_

#include "cudaTypeDef.cuh"
#include "gridCUDA.cuh"
#include "particleArrayCUDA.cuh"
#include <cstdint>
#include <algorithm>  // std::swap

// ======= Cell-sort buffers and host orchestration =======
//
// These types support the 4-stage GPU counting-sort pipeline:
//   Stage 1: build a cell histogram in `cell_counts`
//   Stage 2: exclusive-scan that histogram into `cell_offsets`
//   Stage 3: reserve one sorted destination per particle in `sorted_indices`
//   Stage 4: scatter each SoA field through the permutation using `scratch`
//
// The host-side `CellSorter` interface exposes the work in 3 host phases:
//   prepareBuffers()   -> resize temporary allocations if needed
//   enqueueSortAsync() -> launch GPU stages 1-3 asynchronously
//   finishSort()       -> wait, then execute stage 4 and swap SoA pointers

// ======= Shared scan configuration =======

/**
 * @brief CUDA block size used by the Stage 2 prefix-sum kernels.
 */
static constexpr int SORT_SCAN_BLOCK_SIZE        = 256;

/**
 * @brief Number of histogram entries scanned per block in the multi-block path.
 *
 * Each block processes two values per thread, so one tile contains
 * `2 * SORT_SCAN_BLOCK_SIZE` entries.
 */
static constexpr int SORT_SCAN_ELEMENTS_PER_BLOCK = 2 * SORT_SCAN_BLOCK_SIZE;  // 512

// ======= Per-species temporary sort buffers =======

/**
 * @brief Device-resident scratch arrays reused by one species' cell sorter.
 *
 * The grid determines `num_cells`, so the histogram and prefix-sum buffers
 * are allocated once at initialization. Only `sorted_indices` needs to grow
 * with particle count.
 */
struct CellSortBuffers {

    int*          cell_counts;        // [num_cells] Stage 1 histogram.
    int*          cell_offsets;       // [num_cells] Stage 2 output, then mutated by Stage 3 atomics.
    int*          cell_start_offsets; // [num_cells] Preserved Stage 2 output for cell-aware kernels.
    unsigned int* sorted_indices;     // [max_particles] Stage 3 permutation for Stage 4 scatter.
    int*          block_sums;         // [num_scan_blocks] Stage 2 temporary storage for block totals.

    int      num_cells       = 0;
    int      num_scan_blocks = 0;
    uint32_t max_particles   = 0;
    bool     allocated       = false;

    /**
     * @brief Allocate every fixed-size and particle-count-dependent sort buffer.
     *
     * `ncells` determines the histogram and prefix-sum storage. `max_pcl`
     * determines the initial capacity of the Stage 3 permutation buffer.
     */
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

    /**
     * @brief Release all device allocations owned by this buffer bundle.
     */
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

    /**
     * @brief Reset the Stage 1 histogram before launching a new sort.
     */
    __host__ void zero_async(cudaStream_t s) {
        cudaErrChk(cudaMemsetAsync(cell_counts, 0,
                                   num_cells * sizeof(int), s));
    }

    /**
     * @brief Grow the Stage 3 permutation buffer when particle count increases.
     *
     * Existing contents are transient and do not need to be preserved.
     */
    __host__ void ensure_capacity(uint32_t needed, cudaStream_t s) {
        if (needed <= max_particles) return;
        // Free old sorted_indices and re-allocate (contents are transient)
        cudaFree(sorted_indices);
        max_particles = needed;
        cudaErrChk(cudaMalloc(&sorted_indices, needed * sizeof(unsigned int)));
    }
};


// ======= Stage 4 scratch buffer =======

/**
 * @brief Single temporary device buffer used by the Stage 4 scatter-and-swap.
 *
 * The buffer is sized for the largest SoA field type. During Stage 4 each
 * field is scattered into `ptr`, then the field pointer and `ptr` exchange
 * ownership so the sorted allocation becomes the live SoA storage.
 */
struct SortScratchBuffer {

    void*   ptr            = nullptr;
    size_t  capacity_bytes = 0;

    /**
     * @brief Ensure the scatter scratch buffer is large enough for one SoA field.
     */
    __host__ void ensure_capacity(size_t needed_bytes) {
        if (needed_bytes <= capacity_bytes) return;
        if (ptr) cudaFree(ptr);
        cudaErrChk(cudaMalloc(&ptr, needed_bytes));
        capacity_bytes = needed_bytes;
    }

    /**
     * @brief Release the scratch allocation used by Stage 4.
     */
    __host__ void free() {
        if (ptr) { cudaFree(ptr); ptr = nullptr; capacity_bytes = 0; }
    }

    /**
     * @brief Return the scratch allocation cast to the field type of one scatter kernel.
     */
    template <typename T>
    T* as() { return static_cast<T*>(ptr); }
};


// ======= CellSorter host-side controller =======

/**
 * @brief Host-side orchestrator for one species' 4-stage GPU cell sort.
 *
 * Owns the temporary buffers needed by stages 1-4 and exposes a split API
 * that lets the caller overlap GPU stages 1-3 with CPU work such as MPI.
 *
 * After Stage 4, the SoA field pointers inside `hostPtr->getSoA()` have been
 * swapped with the scratch allocation. The caller must copy the updated
 * metadata object back to device memory before later kernels consume it.
 */
struct CellSorter {

    CellSortBuffers   buffers;
    SortScratchBuffer scratch;
    int  num_cells   = 0;
    bool initialized = false;

    // Pending host-side state cached after stages 1-3 have been enqueued.
    uint32_t           pending_num_to_sort = 0;
    uint32_t           pending_nop         = 0;
    particleArrayCUDA* pending_hostPtr     = nullptr;
    bool               sort_pending        = false;

    /**
     * @brief Allocate the per-species sort state shared by all later sort calls.
     */
    __host__ void init(const grid3DCUDA& grid, uint32_t initial_capacity, cudaStream_t s) {
        num_cells = grid.nxc * grid.nyc * grid.nzc;
        buffers.allocate(initial_capacity, num_cells, s);
        // Scratch must hold one full SoA field for the cycling swap.
        scratch.ensure_capacity(initial_capacity * sizeof(double));
        initialized = true;
    }

    /**
     * @brief Host phase A: resize the reusable buffers before a new sort.
     *
     * This phase does not launch kernels. It may call `cudaMalloc()` or
     * `cudaFree()`, so the caller must ensure no in-flight kernel still uses
     * the previous allocations.
     */
    __host__ void prepareBuffers(particleArrayCUDA* hostPtr, cudaStream_t s);

    /**
     * @brief Host phase B: enqueue GPU stages 1-3 asynchronously on stream `s`.
     *
     * Stage 1 builds the histogram, Stage 2 scans it into cell offsets, and
     * Stage 3 reserves a sorted destination for each particle. The call returns
     * immediately after launch so the caller can overlap CPU work while the GPU
     * executes these stages.
     *
     * `prepareBuffers()` must be called first.
     */
    __host__ void enqueueSortAsync(particleArrayCUDA* hostPtr,
                                   const grid3DCUDA*  deviceGrid,
                                   uint32_t           num_to_sort,
                                   cudaStream_t       s);

    /**
     * @brief Host phase C: wait for stages 1-3, then execute GPU Stage 4.
     *
     * This phase synchronizes the stream, scatters every SoA field through the
     * permutation built in Stage 3, and swaps the live SoA pointers to the
     * freshly sorted allocations.
     */
    __host__ void finishSort(cudaStream_t s);

    /**
     * @brief Run host phases A-C sequentially for a complete synchronous sort.
     */
    __host__ void sort(particleArrayCUDA* hostPtr,
                       const grid3DCUDA*  deviceGrid,
                       uint32_t           num_to_sort,
                       cudaStream_t       s);

    // Accessors used by later cell-aware kernels after Stage 2 has completed.
    __host__ const int* getCellStartOffsets() const { return buffers.cell_start_offsets; }
    __host__ int*       getCellCounts()       const { return buffers.cell_counts; }
    __host__ int        getNumCells()         const { return num_cells; }

    /**
     * @brief Release all GPU memory owned by this per-species sorter.
     */
    __host__ void free() {
        buffers.free();
        scratch.free();
        initialized = false;
    }
};

#endif // _CELL_SORT_BUFFERS_H_
