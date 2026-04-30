// ======= Cell-based particle sorting =======
//
// GPU counting-sort pipeline for the cell-sorted particle path.
//
// Algorithm (4 stages):
//   Stage 1: Position -> cell index (on-the-fly) + histogram
//   Stage 2: Exclusive prefix sum of histogram -> cell offsets
//   Stage 3: Recompute cell -> atomicAdd on offsets -> sorted_indices
//   Stage 4: scatter_partial per SoA array + cycling pointer swap
//
// Host-side execution is split into:
//   prepareBuffers()   -> resize reusable temporary storage
//   enqueueSortAsync() -> launch GPU stages 1-3
//   finishSort()       -> wait, then launch/complete GPU stage 4
//
// All kernels use grid-stride loops and handle arbitrary particle/cell counts.
// Partial sort: only particles[0 .. num_to_sort) are sorted; the tail
// [num_to_sort .. nop) is copied as-is by the scatter kernel.

#include "cellSortBuffers.cuh"
#include <vector>
#include <cmath>
#include <cstdio>

// ======= Cell index helper =======

/**
 * @brief Map one particle position to the flattened guarded-cell index.
 *
 * Delegates to `grid3DCUDA::get_safe_cell()` so the cell classification stays
 * aligned with the mover and moment kernels, including the NaN-safe clamping
 * path used before `floor()`. The linearization is
 * `cx + cy * nxc + cz * nxc * nyc` with x as the fastest-varying index.
 */
__device__ __forceinline__
int compute_cell_idx(cudaPclType_X xp, cudaPclType_Y yp, cudaPclType_Z zp,
                     const grid3DCUDA* g)
{
    int cx, cy, cz;
    g->get_safe_cell(xp, yp, zp, cx, cy, cz);
    return cx + cy * g->nxc + cz * g->nxc * g->nyc;
}


// ======= Warp-aggregated histogram helper =======

/**
 * @brief Aggregate equal-cell contributions within a warp before issuing atomics.
 *
 * Threads in the active warp are partitioned by `cell`. One leader per group
 * performs a single `atomicAdd()` carrying the population count of that group.
 * `WARP_SIZE` and `warp_mask_t` come from `cudaTypeDef.cuh`.
 */
__device__ __forceinline__
void warp_aggregated_atomic_inc(int* histogram, int cell)
{
    const warp_mask_t active = __activemask();
    const int lane = threadIdx.x & (WARP_SIZE - 1);
    warp_mask_t remaining = active;

    while (remaining != 0) {
        int leader = __ffs(remaining) - 1;
        int leader_cell = __shfl_sync(active, cell, leader);
        warp_mask_t match = __ballot_sync(active, cell == leader_cell) & remaining;
        if (lane == leader) {
            atomicAdd(&histogram[leader_cell], __popc(match));
        }
        remaining &= ~match;
    }
}


// ======= Stage 1: Histogram =======

static constexpr int SORT_HISTOGRAM_SHMEM_LIMIT = 1024;
static constexpr int SORT_BLOCK_SIZE = 256;

/**
 * @brief Build the cell histogram with one shared-memory accumulator per block.
 *
 * This path is used when the full histogram fits in shared memory. Each block
 * clears a private histogram, accumulates with warp-aggregated atomics, then
 * flushes its totals to the global histogram.
 */
__global__ void cell_sort_histogram_small_kernel(
    const cudaPclType_X* __restrict__ x,
    const cudaPclType_Y* __restrict__ y,
    const cudaPclType_Z* __restrict__ z,
    int*                 __restrict__ cell_counts,
    const grid3DCUDA*    __restrict__ grid,
    int                  num_cells,
    uint32_t             num_to_sort)
{
    extern __shared__ int shmem_hist[];

    const size_t stride = size_t(gridDim.x) * blockDim.x;

    // Collaborative zero of shared histogram
    for (int i = threadIdx.x; i < num_cells; i += blockDim.x)
        shmem_hist[i] = 0;
    __syncthreads();

    // Grid-stride loop over particles to sort
    for (size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
         tid < num_to_sort; tid += stride)
    {
        int cell = compute_cell_idx(x[tid], y[tid], z[tid], grid);
        warp_aggregated_atomic_inc(shmem_hist, cell);
    }
    __syncthreads();

    // Flush shared histogram to global
    for (int i = threadIdx.x; i < num_cells; i += blockDim.x) {
        atomicAdd(&cell_counts[i], shmem_hist[i]);
    }
}


/**
 * @brief Build the cell histogram directly in global memory for large grids.
 */
__global__ void cell_sort_histogram_warp_agg_kernel(
    const cudaPclType_X* __restrict__ x,
    const cudaPclType_Y* __restrict__ y,
    const cudaPclType_Z* __restrict__ z,
    int*                 __restrict__ cell_counts,
    const grid3DCUDA*    __restrict__ grid,
    uint32_t             num_to_sort)
{
    const size_t stride = size_t(gridDim.x) * blockDim.x;

    for (size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
         tid < num_to_sort; tid += stride)
    {
        int cell = compute_cell_idx(x[tid], y[tid], z[tid], grid);
        warp_aggregated_atomic_inc(cell_counts, cell);
    }
}


/**
 * @brief Launch the Stage 1 histogram kernel variant selected by grid size.
 */
__host__ inline void launch_cell_sort_histogram(
    const cudaPclType_X* x,
    const cudaPclType_Y* y,
    const cudaPclType_Z* z,
    int*                 cell_counts,
    const grid3DCUDA*    deviceGrid,
    int                  num_cells,
    uint32_t             num_to_sort,
    cudaStream_t         stream)
{
    if (num_to_sort == 0) return;
    const int blocks = getGridSize(num_to_sort, (uint32_t)SORT_BLOCK_SIZE);

    if (num_cells <= SORT_HISTOGRAM_SHMEM_LIMIT) {
        const size_t shmem = num_cells * sizeof(int);
        cell_sort_histogram_small_kernel<<<blocks, SORT_BLOCK_SIZE, shmem, stream>>>(
            x, y, z, cell_counts, deviceGrid, num_cells, num_to_sort);
    } else {
        cell_sort_histogram_warp_agg_kernel<<<blocks, SORT_BLOCK_SIZE, 0, stream>>>(
            x, y, z, cell_counts, deviceGrid, num_to_sort);
    }
}


// ======= Stage 2: Exclusive prefix sum =======

/**
 * @brief Run an in-place Blelloch exclusive scan on a shared-memory array.
 *
 * `data` must hold a power-of-two number of elements. Work is distributed
 * across all threads in the block, so the helper also handles `n` values
 * larger than `2 * blockDim.x`.
 *
 * The algorithm has two tree passes:
 *   1. up-sweep: accumulate subtree sums toward the root
 *   2. down-sweep: propagate exclusive prefixes back to every leaf
 */
__device__ __forceinline__
void blelloch_scan_shared(int* data, int n)
{
    const int tid = threadIdx.x;
    const int nthreads = blockDim.x;

    // Up-sweep (reduce)
    int offset = 1;
    for (int d = n >> 1; d > 0; d >>= 1) {
        __syncthreads();
        for (int i = tid; i < d; i += nthreads) {
            int ai = offset * (2 * i + 1) - 1;
            int bi = offset * (2 * i + 2) - 1;
            data[bi] += data[ai];
        }
        offset *= 2;
    }

    // Clear last for exclusive scan
    if (tid == 0) data[n - 1] = 0;

    // Down-sweep
    for (int d = 1; d < n; d *= 2) {
        offset >>= 1;
        __syncthreads();
        for (int i = tid; i < d; i += nthreads) {
            int ai = offset * (2 * i + 1) - 1;
            int bi = offset * (2 * i + 2) - 1;
            int t = data[ai];
            data[ai] = data[bi];
            data[bi] += t;
        }
    }
    __syncthreads();
}

/**
 * @brief Scan the full histogram in one block when it fits in one scan tile.
 *
 * The kernel loads the histogram into shared memory, pads the tail up to the
 * next power of two, runs the in-place Blelloch scan, then writes the valid
 * prefix values back to `cell_offsets`.
 */
__global__ void cell_sort_prefix_sum_single_kernel(
    int*       __restrict__ cell_offsets,
    const int* __restrict__ cell_counts,
    int                     num_cells,
    int                     padded_size)  // power of 2 >= num_cells
{
    extern __shared__ int sdata[];
    const int tid = threadIdx.x;

    for (int i = tid; i < padded_size; i += blockDim.x)
        sdata[i] = (i < num_cells) ? cell_counts[i] : 0;
    __syncthreads();

    blelloch_scan_shared(sdata, padded_size);

    for (int i = tid; i < num_cells; i += blockDim.x)
        cell_offsets[i] = sdata[i];
}

// ======= Stage 2: Multi-block helpers =======

/**
 * @brief Stage 2 phase 1: scan one tile locally and emit its total.
 *
 * Each block loads one `SORT_SCAN_ELEMENTS_PER_BLOCK` tile, writes the tile's
 * exclusive scan to `output`, and stores the tile sum in `block_sums` so
 * phase 2 can build prefixes between tiles.
 */
__global__ void cell_sort_prefix_sum_phase1_kernel(
    int*       __restrict__ output,
    int*       __restrict__ block_sums,
    const int* __restrict__ input,
    int                     num_elements)
{
    extern __shared__ int sdata[];
    const int tid         = threadIdx.x;
    const int block_start = blockIdx.x * SORT_SCAN_ELEMENTS_PER_BLOCK;

    const int idx1 = block_start + tid;
    const int idx2 = block_start + tid + SORT_SCAN_BLOCK_SIZE;

    int val1 = (idx1 < num_elements) ? input[idx1] : 0;
    int val2 = (idx2 < num_elements) ? input[idx2] : 0;
    sdata[tid]                       = val1;
    sdata[tid + SORT_SCAN_BLOCK_SIZE] = val2;
    __syncthreads();

    // Block total via warp reduction
    int local_sum = val1 + val2;
    local_sum = warp_reduce_sum(local_sum);

    constexpr int num_warps = SORT_SCAN_BLOCK_SIZE / WARP_SIZE;
    __shared__ int warp_sums[num_warps];
    int warp_id = tid / WARP_SIZE;
    int lane    = tid & (WARP_SIZE - 1);
    if (lane == 0) warp_sums[warp_id] = local_sum;
    __syncthreads();

    __shared__ int block_total;
    if (tid < num_warps) {
        local_sum = warp_sums[tid];
        for (int off = num_warps / 2; off > 0; off >>= 1)
            local_sum += __shfl_down_sync(static_cast<warp_mask_t>((1ull << num_warps) - 1), local_sum, off);
        if (tid == 0) block_total = local_sum;
    }
    __syncthreads();

    // Blelloch scan
    blelloch_scan_shared(sdata, SORT_SCAN_ELEMENTS_PER_BLOCK);

    if (idx1 < num_elements) output[idx1] = sdata[tid];
    if (idx2 < num_elements) output[idx2] = sdata[tid + SORT_SCAN_BLOCK_SIZE];

    if (tid == 0) block_sums[blockIdx.x] = block_total;
}

/**
 * @brief Stage 2 phase 2: exclusive-scan the per-tile totals from phase 1.
 */
__global__ void cell_sort_prefix_sum_phase2_kernel(
    int* __restrict__ block_sums,
    int               num_blocks,
    int               padded_size)  // power of 2 >= num_blocks
{
    extern __shared__ int sdata[];
    const int tid = threadIdx.x;

    for (int i = tid; i < padded_size; i += blockDim.x)
        sdata[i] = (i < num_blocks) ? block_sums[i] : 0;
    __syncthreads();

    blelloch_scan_shared(sdata, padded_size);

    for (int i = tid; i < num_blocks; i += blockDim.x)
        block_sums[i] = sdata[i];
}

/**
 * @brief Stage 2 phase 3: add the per-tile prefix to every local phase-1 scan.
 */
__global__ void cell_sort_prefix_sum_phase3_kernel(
    int*       __restrict__ data,
    const int* __restrict__ block_sums,
    int                     num_elements)
{
    __shared__ int prefix;
    if (threadIdx.x == 0) prefix = block_sums[blockIdx.x];
    __syncthreads();

    const int idx1 = blockIdx.x * SORT_SCAN_ELEMENTS_PER_BLOCK + threadIdx.x;
    const int idx2 = idx1 + SORT_SCAN_BLOCK_SIZE;
    if (idx1 < num_elements) data[idx1] += prefix;
    if (idx2 < num_elements) data[idx2] += prefix;
}


// ======= Stage 2 launcher =======

/**
 * @brief Return the smallest power of two greater than or equal to `v`.
 */
__host__ inline int next_power_of_2(int v) {
    int p = 1;
    while (p < v) p *= 2;
    return p;
}

/**
 * @brief Launch the single-block or three-phase Stage 2 prefix-sum path.
 *
 * Small histograms are scanned entirely in one block. Larger histograms are
 * handled as:
 *   1. phase 1: local tile scans + tile totals
 *   2. phase 2: scan the tile totals
 *   3. phase 3: add each scanned tile prefix back to its tile
 */
__host__ inline void launch_cell_sort_prefix_sum(
    int*         cell_offsets,
    const int*   cell_counts,
    int*         block_sums,
    int          num_cells,
    cudaStream_t stream)
{
    if (num_cells == 0) return;

    if (num_cells <= SORT_SCAN_ELEMENTS_PER_BLOCK) {
        int padded = next_power_of_2(num_cells);
        size_t shmem = padded * sizeof(int);
        cell_sort_prefix_sum_single_kernel<<<1, SORT_SCAN_BLOCK_SIZE, shmem, stream>>>(
            cell_offsets, cell_counts, num_cells, padded);
        return;
    }

    // ======= Phase 1: Scan each tile and record tile totals =======
    int nblocks = (num_cells + SORT_SCAN_ELEMENTS_PER_BLOCK - 1) / SORT_SCAN_ELEMENTS_PER_BLOCK;
    size_t phase1_shmem = SORT_SCAN_ELEMENTS_PER_BLOCK * sizeof(int);

    cell_sort_prefix_sum_phase1_kernel<<<nblocks, SORT_SCAN_BLOCK_SIZE, phase1_shmem, stream>>>(
        cell_offsets, block_sums, cell_counts, num_cells);

    // ======= Phase 2: Scan the tile totals in one block =======
    int phase2_padded = next_power_of_2(nblocks);
    constexpr int MAX_PHASE2_SINGLE_BLOCK = 8192;
    if (phase2_padded > MAX_PHASE2_SINGLE_BLOCK) {
        fprintf(stderr,
            "FATAL: cell_sort prefix_sum phase2 requires %d blocks (max %d). "
            "Grid has %d cells — reduce subdomain size.\n",
            nblocks, MAX_PHASE2_SINGLE_BLOCK, num_cells);
        abort();
    }
    size_t phase2_shmem = phase2_padded * sizeof(int);
    cell_sort_prefix_sum_phase2_kernel<<<1, SORT_SCAN_BLOCK_SIZE, phase2_shmem, stream>>>(
        block_sums, nblocks, phase2_padded);

    // ======= Phase 3: Add the scanned tile prefixes back to each tile =======
    cell_sort_prefix_sum_phase3_kernel<<<nblocks, SORT_SCAN_BLOCK_SIZE, 0, stream>>>(
        cell_offsets, block_sums, num_cells);
}


// ======= Stage 3: Sorted indices =======

/**
 * @brief Recompute each particle cell and reserve its destination slot.
 *
 * `cell_offsets` starts as the Stage 2 exclusive prefix sum. Each particle
 * atomically increments its cell counter and receives the unique scatter index
 * it will use in Stage 4.
 */
__global__ void cell_sort_sorted_indices_kernel(
    unsigned int*        __restrict__ sorted_indices,
    int*                 __restrict__ cell_offsets,   // mutated by atomics
    const cudaPclType_X* __restrict__ x,
    const cudaPclType_Y* __restrict__ y,
    const cudaPclType_Z* __restrict__ z,
    const grid3DCUDA*    __restrict__ grid,
    uint32_t             num_to_sort)
{
    const size_t stride = size_t(gridDim.x) * blockDim.x;

    for (size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
         tid < num_to_sort; tid += stride)
    {
        int cell = compute_cell_idx(x[tid], y[tid], z[tid], grid);
        int dest = atomicAdd(&cell_offsets[cell], 1);
        sorted_indices[tid] = static_cast<unsigned int>(dest);
    }
}

/**
 * @brief Launch Stage 3 to fill the permutation used by the scatter pass.
 */
__host__ inline void launch_cell_sort_sorted_indices(
    unsigned int*        sorted_indices,
    int*                 cell_offsets,
    const cudaPclType_X* x,
    const cudaPclType_Y* y,
    const cudaPclType_Z* z,
    const grid3DCUDA*    deviceGrid,
    uint32_t             num_to_sort,
    cudaStream_t         stream)
{
    if (num_to_sort == 0) return;
    const int blocks = getGridSize(num_to_sort, (uint32_t)SORT_BLOCK_SIZE);
    cell_sort_sorted_indices_kernel<<<blocks, SORT_BLOCK_SIZE, 0, stream>>>(
        sorted_indices, cell_offsets, x, y, z, deviceGrid, num_to_sort);
}


// ======= Stage 4: Scatter =======

/**
 * @brief Scatter the sorted prefix and copy the unsorted tail unchanged.
 *
 * Entries `[0, num_to_sort)` are written to their Stage 3 destinations, while
 * `[num_to_sort, nop)` are copied in place so the unsorted tail remains valid.
 */
template <typename T>
__global__ void cell_sort_scatter_partial_kernel(
    T*                   __restrict__ dst,
    const T*             __restrict__ src,
    const unsigned int*  __restrict__ sorted_indices,
    uint32_t             num_to_sort,
    uint32_t             nop)
{
    const size_t stride = size_t(gridDim.x) * blockDim.x;

    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x;
         i < nop; i += stride)
    {
        if (i < num_to_sort)
            dst[sorted_indices[i]] = src[i];   // scattered write (sorted prefix)
        else
            dst[i] = src[i];                   // coalesced identity copy (tail)
    }
}

/**
 * @brief Launch the Stage 4 scatter for one SoA field buffer.
 */
template <typename T>
__host__ inline void launch_cell_sort_scatter_partial(
    T*                  dst,
    const T*            src,
    const unsigned int* sorted_indices,
    uint32_t            num_to_sort,
    uint32_t            nop,
    cudaStream_t        stream)
{
    if (nop == 0) return;
    const int blocks = getGridSize(nop, (uint32_t)SORT_BLOCK_SIZE);
    cell_sort_scatter_partial_kernel<T><<<blocks, SORT_BLOCK_SIZE, 0, stream>>>(
        dst, src, sorted_indices, num_to_sort, nop);
}


// ======= CellSorter buffer preparation =======

/**
 * @brief Host phase A: ensure all temporary sort buffers are large enough.
 *
 * This prepares the reusable storage needed by GPU stages 1-4 but does not
 * launch any kernels.
 */
__host__ void CellSorter::prepareBuffers(
    particleArrayCUDA* hostPtr,
    cudaStream_t       s)
{
    if (!initialized) return;
    const uint32_t nop = hostPtr->getNOP();
    if (nop == 0) return;

    // Grow sorted_indices if needed (may cudaMalloc/cudaFree)
    buffers.ensure_capacity(nop, s);

    // Scratch must hold capacity (not nop) because after the pointer-swap
    // cycle scratch.ptr becomes a SoA field buffer.  All SoA fields must
    // hold up to soa.capacity elements for future particle additions.
    const uint32_t cap = hostPtr->getCapacity();
    scratch.ensure_capacity(cap * sizeof(double));
}


// ======= CellSorter staging =======

/**
 * @brief Host phase B: enqueue GPU stages 1-3 without waiting for Stage 4.
 */
__host__ void CellSorter::enqueueSortAsync(
    particleArrayCUDA* hostPtr,
    const grid3DCUDA*  deviceGrid,
    uint32_t           num_to_sort,
    cudaStream_t       s)
{
    if (!initialized) return;

    const uint32_t nop = hostPtr->getNOP();
    if (num_to_sort == 0 || nop == 0) { sort_pending = false; return; }
    if (num_to_sort > nop) num_to_sort = nop;  // safety clamp

    // Save state for finishSort()
    pending_num_to_sort = num_to_sort;
    pending_nop         = nop;
    pending_hostPtr     = hostPtr;
    sort_pending        = true;

    ParticleSoADevice* soa = hostPtr->getSoA();

    // ======= Stage 1: Histogram =======
    buffers.zero_async(s);
    launch_cell_sort_histogram(
        soa->x, soa->y, soa->z,
        buffers.cell_counts, deviceGrid, num_cells, num_to_sort, s);

    // ======= Stage 2: Prefix sum =======
    launch_cell_sort_prefix_sum(
        buffers.cell_offsets, buffers.cell_counts,
        buffers.block_sums,
        num_cells, s);

    // Preserve the Stage 2 output before Stage 3 mutates `cell_offsets`.
    cudaErrChk(cudaMemcpyAsync(
        buffers.cell_start_offsets, buffers.cell_offsets,
        num_cells * sizeof(int), cudaMemcpyDeviceToDevice, s));

    // ======= Stage 3: Sorted indices =======
    launch_cell_sort_sorted_indices(
        buffers.sorted_indices, buffers.cell_offsets,
        soa->x, soa->y, soa->z, deviceGrid, num_to_sort, s);
}

// ======= CellSorter completion =======

/**
 * @brief Host phase C: synchronize stages 1-3, then execute Stage 4.
 */
__host__ void CellSorter::finishSort(cudaStream_t s)
{
    if (!sort_pending) return;
    sort_pending = false;

    const uint32_t num_to_sort = pending_num_to_sort;
    const uint32_t nop         = pending_nop;
    ParticleSoADevice* soa     = pending_hostPtr->getSoA();

    // ======= Required barrier =======
    // Stages 1-3 must complete on the GPU before we scatter.
    // Also protects against host-pinned DMA races.
    cudaErrChk(cudaStreamSynchronize(s));

    const unsigned int* idx = buffers.sorted_indices;

    // ======= Stage 4: Scatter one field at a time and swap pointers =======
    // Scatter field_ptr -> scratch, then swap the two pointers so that
    // scratch now points to the old (unsorted) allocation and field_ptr
    // points to the freshly-sorted data.
    auto scatter_and_swap = [&](auto*& field_ptr) {
        using FT = std::remove_pointer_t<std::remove_reference_t<decltype(*field_ptr)>>;
        launch_cell_sort_scatter_partial<FT>(static_cast<FT*>(scratch.ptr), field_ptr, idx, num_to_sort, nop, s);
        // Exchange: field_ptr <- scratch (sorted), scratch <- field_ptr (old)
        field_ptr = static_cast<FT*>(std::exchange(scratch.ptr, static_cast<void*>(field_ptr)));
    };

    scatter_and_swap(soa->u);
    scatter_and_swap(soa->v);
    scatter_and_swap(soa->w);
    scatter_and_swap(soa->q);
    scatter_and_swap(soa->x);
    scatter_and_swap(soa->y);
    scatter_and_swap(soa->z);
    scatter_and_swap(soa->t);
}

// ======= CellSorter convenience wrapper =======

/**
 * @brief Execute host phases A-C in sequence for a complete sort.
 */
__host__ void CellSorter::sort(
    particleArrayCUDA* hostPtr,
    const grid3DCUDA*  deviceGrid,
    uint32_t           num_to_sort,
    cudaStream_t       s)
{
    prepareBuffers(hostPtr, s);
    enqueueSortAsync(hostPtr, deviceGrid, num_to_sort, s);
    finishSort(s);
}
