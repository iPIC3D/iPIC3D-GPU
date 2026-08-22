#ifndef CELL_SORT_BUFFERS_CUH
#define CELL_SORT_BUFFERS_CUH

#include "cudaTypeDef.cuh"
#include "gridCUDA.cuh"
#include "particleArrayCUDA.cuh"
#include <algorithm> // std::swap
#include <cstdint>

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
static constexpr int SORT_SCAN_BLOCK_SIZE = 256;

/**
 * @brief Number of histogram entries scanned per block in the multi-block path.
 *
 * Each block processes two values per thread, so one tile contains
 * `2 * SORT_SCAN_BLOCK_SIZE` entries.
 */
static constexpr int SORT_SCAN_ELEMENTS_PER_BLOCK =
    2 * SORT_SCAN_BLOCK_SIZE; // 512

#if defined(IPIC3D_GPU_CYCLE_DIAGNOSTICS)

/** Maximum number of bounded reduction blocks used by sorter diagnostics. */
inline constexpr int CELL_SORT_DIAGNOSTIC_MAX_BLOCKS = 256;

/** Number of sortable SoA fields, including the optional particle ID. */
inline constexpr int CELL_SORT_DIAGNOSTIC_MAX_FIELDS = 8;
inline constexpr int CELL_SORT_DIAGNOSTIC_SORT_BUFFER_COUNT = 6;
inline constexpr int CELL_SORT_DIAGNOSTIC_ALLOCATION_METADATA_COUNT = 6;

/**
 * One block-local Stage-4 result.  The actual scatter destination is filled
 * with a diagnostic bit pattern before each field is scattered; any matching
 * entries afterward were not written by the scatter kernel.
 */
struct CellSortStage4DiagnosticPartial {
  std::uint64_t prefixPoisonCount;
  std::uint64_t firstPrefixPoison;
  std::uint64_t lastPrefixPoison;
  std::uint64_t tailPoisonCount;
  std::uint64_t firstTailPoison;
  std::uint64_t lastTailPoison;
  std::uint64_t prefixValueMismatchCount;
  std::uint64_t firstPrefixValueMismatchSource;
  std::uint64_t firstPrefixValueMismatchDestination;
  std::uint64_t tailValueMismatchCount;
  std::uint64_t firstTailValueMismatch;
};

#endif // IPIC3D_GPU_CYCLE_DIAGNOSTICS

// ======= Per-species temporary sort buffers =======

/**
 * @brief Device-resident scratch arrays reused by one species' cell sorter.
 *
 * The grid determines `num_cells`, so the histogram and prefix-sum buffers
 * are allocated once at initialization. Only `sorted_indices` needs to grow
 * with particle count.
 */
struct CellSortBuffers {

  int* cell_counts;  // [num_cells] Stage 1 histogram.
  int* cell_offsets; // [num_cells] Stage 2 output, then mutated by Stage 3
                     // atomics.
  int* cell_start_offsets;      // [num_cells] Preserved Stage 2 output for
                                // cell-aware kernels.
  unsigned int* sorted_indices; // [max_particles] Stage 3 permutation for Stage
                                // 4 scatter.
  int* block_sums; // [num_scan_blocks] Stage 2 temporary storage for block
                   // totals.
#if defined(IPIC3D_GPU_CYCLE_DIAGNOSTICS)
  // Immutable snapshot of the raw Phase-1 tile totals, captured before the
  // in-place Phase-2 scan destroys `block_sums`.
  int* diagnostic_phase1_block_sums = nullptr; // [num_scan_blocks]
#endif

  int num_cells = 0;
  int num_scan_blocks = 0;
  uint32_t max_particles = 0;
  bool allocated = false;

  /**
   * @brief Allocate every fixed-size and particle-count-dependent sort buffer.
   *
   * `ncells` determines the histogram and prefix-sum storage. `max_pcl`
   * determines the initial capacity of the Stage 3 permutation buffer.
   */
  __host__ void allocate(uint32_t max_pcl, int ncells, cudaStream_t s) {
    num_cells = ncells;
    max_particles = max_pcl;
    num_scan_blocks = (ncells + SORT_SCAN_ELEMENTS_PER_BLOCK - 1) /
                      SORT_SCAN_ELEMENTS_PER_BLOCK;

    cudaErrChk(cudaMalloc(&cell_counts, ncells * sizeof(int)));
    cudaErrChk(cudaMalloc(&cell_offsets, ncells * sizeof(int)));
    cudaErrChk(cudaMalloc(&cell_start_offsets, ncells * sizeof(int)));
    cudaErrChk(cudaMalloc(&sorted_indices, max_pcl * sizeof(unsigned int)));
    cudaErrChk(cudaMalloc(&block_sums, num_scan_blocks * sizeof(int)));
#if defined(IPIC3D_GPU_CYCLE_DIAGNOSTICS)
    cudaErrChk(cudaMalloc(&diagnostic_phase1_block_sums,
                          num_scan_blocks * sizeof(int)));
#endif

    allocated = true;
  }

  /**
   * @brief Release all device allocations owned by this buffer bundle.
   */
  __host__ void free() {
    if (!allocated)
      return;
    cudaFree(cell_counts);
    cudaFree(cell_offsets);
    cudaFree(cell_start_offsets);
    cudaFree(sorted_indices);
    cudaFree(block_sums);
#if defined(IPIC3D_GPU_CYCLE_DIAGNOSTICS)
    cudaFree(diagnostic_phase1_block_sums);
    diagnostic_phase1_block_sums = nullptr;
#endif
    cell_counts = cell_offsets = cell_start_offsets = block_sums = nullptr;
    sorted_indices = nullptr;
    allocated = false;
  }

  /**
   * @brief Reset the Stage 1 histogram before launching a new sort.
   */
  __host__ void zero_async(cudaStream_t s) {
    cudaErrChk(cudaMemsetAsync(cell_counts, 0, num_cells * sizeof(int), s));
  }

  /**
   * @brief Grow the Stage 3 permutation buffer when particle count increases.
   *
   * Existing contents are transient and do not need to be preserved.
   */
  __host__ void ensure_capacity(uint32_t needed, cudaStream_t s) {
    if (needed <= max_particles)
      return;
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

  void* ptr = nullptr;
  size_t capacity_bytes = 0;

  /**
   * @brief Ensure the scatter scratch buffer is large enough for one SoA field.
   */
  __host__ void ensure_capacity(size_t needed_bytes) {
    if (needed_bytes <= capacity_bytes)
      return;
    if (ptr)
      cudaFree(ptr);
    cudaErrChk(cudaMalloc(&ptr, needed_bytes));
    capacity_bytes = needed_bytes;
  }

  /**
   * @brief Release the scratch allocation used by Stage 4.
   */
  __host__ void free() {
    if (ptr) {
      cudaFree(ptr);
      ptr = nullptr;
      capacity_bytes = 0;
    }
  }

  /**
   * @brief Return the scratch allocation cast to the field type of one scatter
   * kernel.
   */
  template <typename T> T* as() { return static_cast<T*>(ptr); }
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

  CellSortBuffers buffers;
  SortScratchBuffer scratch;
  int num_cells = 0;
  bool initialized = false;

  // Pending host-side state cached after stages 1-3 have been enqueued.
  uint32_t pending_num_to_sort = 0;
  uint32_t pending_nop = 0;
  particleArrayCUDA* pending_hostPtr = nullptr;
  bool sort_pending = false;

#if defined(IPIC3D_GPU_CYCLE_DIAGNOSTICS)
  // The diagnostic reducer owns this bounded output allocation.  It is armed
  // only for the selected cycle, immediately before finishSort().
  CellSortStage4DiagnosticPartial* diagnostic_stage4_partials = nullptr;
  int diagnostic_stage4_partial_stride = 0;
  int diagnostic_stage4_blocks = 0;
  int diagnostic_stage4_field_count = 0;
  bool diagnostic_stage4_armed = false;
  std::uintptr_t diagnostic_enqueue_pointers[CELL_SORT_DIAGNOSTIC_MAX_FIELDS]{};
  std::uintptr_t diagnostic_enqueue_scratch = 0;
  int diagnostic_enqueue_field_count = 0;
  bool diagnostic_enqueue_snapshot_valid = false;
  std::uintptr_t
      diagnostic_enqueue_sort_buffers[CELL_SORT_DIAGNOSTIC_SORT_BUFFER_COUNT]{};
  std::uint64_t diagnostic_enqueue_allocation_metadata
      [CELL_SORT_DIAGNOSTIC_ALLOCATION_METADATA_COUNT]{};
  std::uint32_t diagnostic_requested_num_to_sort = 0;
  bool diagnostic_num_to_sort_was_clamped = false;
#endif

  /**
   * @brief Allocate the per-species sort state shared by all later sort calls.
   */
  __host__ void init(const grid3DCUDA& grid, uint32_t initial_capacity,
                     cudaStream_t s) {
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
                                 const grid3DCUDA* deviceGrid,
                                 uint32_t num_to_sort, cudaStream_t s);

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
  __host__ void sort(particleArrayCUDA* hostPtr, const grid3DCUDA* deviceGrid,
                     uint32_t num_to_sort, cudaStream_t s);

  // Accessors used by later cell-aware kernels after Stage 2 has completed.
  __host__ const int* getCellStartOffsets() const {
    return buffers.cell_start_offsets;
  }
  __host__ int* getCellCounts() const { return buffers.cell_counts; }
  __host__ int getNumCells() const { return num_cells; }

#if defined(IPIC3D_GPU_CYCLE_DIAGNOSTICS)
  // Raw diagnostic access is intentionally unavailable in production builds.
  __host__ const int* diagnosticCellOffsets() const {
    return buffers.cell_offsets;
  }
  __host__ const unsigned int* diagnosticSortedIndices() const {
    return buffers.sorted_indices;
  }
  __host__ const int* diagnosticBlockSums() const { return buffers.block_sums; }
  __host__ const int* diagnosticPhase1BlockSums() const {
    return buffers.diagnostic_phase1_block_sums;
  }
  __host__ int diagnosticNumScanBlocks() const {
    return buffers.num_scan_blocks;
  }
  __host__ std::uint32_t diagnosticIndexCapacity() const {
    return buffers.max_particles;
  }
  __host__ void* diagnosticScratchPointer() const { return scratch.ptr; }
  __host__ std::size_t diagnosticScratchBytes() const {
    return scratch.capacity_bytes;
  }
  __host__ bool diagnosticSortPending() const { return sort_pending; }
  __host__ std::uint32_t diagnosticPendingNumToSort() const {
    return pending_num_to_sort;
  }
  __host__ std::uint32_t diagnosticPendingNOP() const { return pending_nop; }
  __host__ const particleArrayCUDA* diagnosticPendingHostPointer() const {
    return pending_hostPtr;
  }
  __host__ std::uint32_t diagnosticRequestedNumToSort() const {
    return diagnostic_requested_num_to_sort;
  }
  __host__ bool diagnosticNumToSortWasClamped() const {
    return diagnostic_num_to_sort_was_clamped;
  }

  // Implemented in cellSortKernel.cu so the report describes the translation
  // unit that actually contains the histogram kernel, not merely its caller.
  __host__ int diagnosticCompiledWarpSize() const;
  __host__ int diagnosticCompiledWarpMaskBytes() const;
  __host__ int diagnosticAlgorithmVersion() const;
  __host__ int diagnosticEnqueuePointerMutationMask(
      const particleArrayCUDA& particles) const;
  __host__ int diagnosticEnqueueSortBufferMutationMask() const;
  __host__ int diagnosticEnqueueAllocationMetadataMutationMask(
      const particleArrayCUDA& particles) const;
  __host__ std::uint64_t
  diagnosticCurrentAllocationMetadata(int field,
                                      const particleArrayCUDA& particles) const;
  __host__ std::uintptr_t diagnosticEnqueuePointer(int field) const {
    return diagnostic_enqueue_pointers[field];
  }
  __host__ std::uintptr_t diagnosticEnqueueScratchPointer() const {
    return diagnostic_enqueue_scratch;
  }
  __host__ std::uintptr_t diagnosticEnqueueSortBufferPointer(int buffer) const {
    return diagnostic_enqueue_sort_buffers[buffer];
  }
  __host__ std::uint64_t diagnosticEnqueueAllocationMetadata(int field) const {
    return diagnostic_enqueue_allocation_metadata[field];
  }

  /** Arm exact per-field scratch coverage checks for the pending Stage 4. */
  __host__ void
  armStage4Diagnostics(CellSortStage4DiagnosticPartial* devicePartials,
                       int partialStride) {
    diagnostic_stage4_partials = devicePartials;
    diagnostic_stage4_partial_stride = partialStride;
    diagnostic_stage4_blocks = 0;
    diagnostic_stage4_field_count = 0;
    diagnostic_stage4_armed =
        sort_pending && devicePartials != nullptr && partialStride > 0;
  }

  __host__ int diagnosticStage4Blocks() const {
    return diagnostic_stage4_blocks;
  }
  __host__ int diagnosticStage4FieldCount() const {
    return diagnostic_stage4_field_count;
  }
#endif

  /**
   * @brief Release all GPU memory owned by this per-species sorter.
   */
  __host__ void free() {
    buffers.free();
    scratch.free();
    initialized = false;
  }
};

#endif // CELL_SORT_BUFFERS_CUH
