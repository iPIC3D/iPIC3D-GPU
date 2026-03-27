// ============================================================================
// GPU kernels for cell-based particle sorting (counting sort).
//
// Algorithm (4 stages):
//   Stage 1: Position → cell index (on-the-fly) + histogram
//   Stage 2: Exclusive prefix sum of histogram → cell offsets
//   Stage 3: Recompute cell → atomicAdd on offsets → sorted_indices
//   Stage 4: scatter_partial per SoA array + cycling pointer swap
//
// All kernels use grid-stride loops and handle arbitrary particle/cell counts.
// Partial sort: only particles[0 .. num_to_sort) are sorted; the tail
// [num_to_sort .. nop) is copied as-is by the scatter kernel.
// ============================================================================

#include "cellSortBuffers.cuh"
#include <vector>
#include <cmath>
#include <cstdio>

// ============================================================================
// Device helper: position → flat cell index
// ============================================================================
//
// Delegates to grid3DCUDA::get_safe_cell() so that the cell assignment is
// *exactly* the same as in the mover (get_safe_cell_and_weights) and moment
// kernels.  In particular this picks up the NaN-safe floating-point clamping
// (make_grid_position_safe) that runs before floor() when
// suppress_runaway_particle_instability is true.
//
// Linearization: cx + cy * nxc + cz * nxc * nyc   (x-fastest)
// Domain: guarded cells [0, nxc) × [0, nyc) × [0, nzc)
//
__device__ __forceinline__
int compute_cell_idx(cudaPclType_X xp, cudaPclType_Y yp, cudaPclType_Z zp,
                     const grid3DCUDA* g)
{
    int cx, cy, cz;
    g->get_safe_cell(xp, yp, zp, cx, cy, cz);
    return cx + cy * g->nxc + cz * g->nxc * g->nyc;
}


// ============================================================================
// Warp-aggregated atomic increment (portable fallback)
// ============================================================================
//
// Uses ballot + shfl to aggregate threads with the same cell index,
// so only one atomicAdd per unique value per warp.
//
// Note: WARP_SIZE and warp_mask_t are defined in cudaTypeDef.cuh for portability.

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
            atomicAdd(&histogram[leader_cell], warp_popcount(match));
        }
        remaining &= ~match;
    }
}


// ============================================================================
// Stage 1a: Histogram — shared memory (small grids, num_cells ≤ 1024)
// ============================================================================
static constexpr int SORT_HISTOGRAM_SHMEM_LIMIT = 1024;
static constexpr int SORT_BLOCK_SIZE = 256;

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
        if (shmem_hist[i] > 0)
            atomicAdd(&cell_counts[i], shmem_hist[i]);
    }
}


// ============================================================================
// Stage 1b: Histogram — warp-aggregated global atomics (large grids)
// ============================================================================
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


// ============================================================================
// Launcher: Stage 1
// ============================================================================
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


// ============================================================================
// Stage 2: Exclusive Prefix Sum (Blelloch scan)
// ============================================================================

// ── Shared memory Blelloch scan (in-place, handles n > 2*blockDim.x) ──
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

// ── Single-block scan (num_cells ≤ SORT_SCAN_ELEMENTS_PER_BLOCK = 512) ──
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

// ── Phase 1: block-local scan, store block total ──
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

// ── Phase 2: scan block totals (single block) ──
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

// ── Phase 3: add block prefix to each element ──
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


// ============================================================================
// Launcher: Stage 2
// ============================================================================
__host__ inline int next_power_of_2(int v) {
    int p = 1;
    while (p < v) p *= 2;
    return p;
}

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

    // Multi-block: 3 phases
    int nblocks = (num_cells + SORT_SCAN_ELEMENTS_PER_BLOCK - 1) / SORT_SCAN_ELEMENTS_PER_BLOCK;
    size_t phase1_shmem = SORT_SCAN_ELEMENTS_PER_BLOCK * sizeof(int);

    cell_sort_prefix_sum_phase1_kernel<<<nblocks, SORT_SCAN_BLOCK_SIZE, phase1_shmem, stream>>>(
        cell_offsets, block_sums, cell_counts, num_cells);

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

    cell_sort_prefix_sum_phase3_kernel<<<nblocks, SORT_SCAN_BLOCK_SIZE, 0, stream>>>(
        cell_offsets, block_sums, num_cells);
}


// ============================================================================
// Stage 3: Compute sorted indices (recompute cell from position)
// ============================================================================
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

// ============================================================================
// Launcher: Stage 3
// ============================================================================
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


// ============================================================================
// Stage 4: Scatter with partial sort (sorted prefix + identity tail)
// ============================================================================
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

// ============================================================================
// Launcher: Stage 4 (one array)
// ============================================================================
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


// ============================================================================
// CellSorter::prepareBuffers() — resize sort buffers (no kernels in flight!)
// ============================================================================
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


// ============================================================================
// CellSorter::enqueueSortAsync() — stages 1-3, non-blocking
// ============================================================================
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

    // ── Stage 1: Histogram ──
    buffers.zero_async(s);
    launch_cell_sort_histogram(
        soa->x, soa->y, soa->z,
        buffers.cell_counts, deviceGrid, num_cells, num_to_sort, s);

    // ── Stage 2: Prefix sum (Blelloch scan) ──
    launch_cell_sort_prefix_sum(
        buffers.cell_offsets, buffers.cell_counts,
        buffers.block_sums,
        num_cells, s);

    // Preserve prefix sum for cell-aware moment kernel
    cudaErrChk(cudaMemcpyAsync(
        buffers.cell_start_offsets, buffers.cell_offsets,
        num_cells * sizeof(int), cudaMemcpyDeviceToDevice, s));

    // ── Stage 3: Sorted indices (recomputes cell from position) ──
    launch_cell_sort_sorted_indices(
        buffers.sorted_indices, buffers.cell_offsets,
        soa->x, soa->y, soa->z, deviceGrid, num_to_sort, s);
}


// ============================================================================
// CellSorter::finishSort() — sync + stage 4 scatter & pointer swap
// ============================================================================
__host__ void CellSorter::finishSort(cudaStream_t s)
{
    if (!sort_pending) return;
    sort_pending = false;

    const uint32_t num_to_sort = pending_num_to_sort;
    const uint32_t nop         = pending_nop;
    ParticleSoADevice* soa     = pending_hostPtr->getSoA();

    // ── Required barrier ──
    // Stages 1-3 must complete on the GPU before we scatter.
    // Also protects against host-pinned DMA races (see original comment).
    cudaErrChk(cudaStreamSynchronize(s));

    const unsigned int* idx = buffers.sorted_indices;

    // Scatter field_ptr → scratch, then swap the two pointers so that
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


// ============================================================================
// CellSorter::sort() — convenience wrapper (backward compat)
// ============================================================================
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
