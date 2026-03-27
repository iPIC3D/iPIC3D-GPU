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
static constexpr int SORT_WARP_SIZE = 32;

__device__ __forceinline__
void warp_aggregated_atomic_inc(int* histogram, int cell)
{
    const unsigned int active = __activemask();
    const int lane = threadIdx.x & (SORT_WARP_SIZE - 1);
    unsigned int remaining = active;

    while (remaining != 0) {
        int leader = __ffs(remaining) - 1;
        int leader_cell = __shfl_sync(active, cell, leader);
        unsigned int match = __ballot_sync(active, cell == leader_cell) & remaining;
        if (lane == leader) {
            atomicAdd(&histogram[leader_cell], __popc(match));
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
    for (int off = SORT_WARP_SIZE / 2; off > 0; off >>= 1)
        local_sum += __shfl_down_sync(0xFFFFFFFF, local_sum, off);

    constexpr int num_warps = SORT_SCAN_BLOCK_SIZE / SORT_WARP_SIZE;
    __shared__ int warp_sums[num_warps];
    int warp_id = tid / SORT_WARP_SIZE;
    int lane    = tid & (SORT_WARP_SIZE - 1);
    if (lane == 0) warp_sums[warp_id] = local_sum;
    __syncthreads();

    __shared__ int block_total;
    if (tid < num_warps) {
        local_sum = warp_sums[tid];
        for (int off = num_warps / 2; off > 0; off >>= 1)
            local_sum += __shfl_down_sync((1u << num_warps) - 1, local_sum, off);
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
// CellSorter::sort() — orchestrate all 4 stages
// ============================================================================
//
// After this call, soa field pointers in hostPtr->getSoA() have been swapped
// with scratch allocations.  Caller must re-sync to device.
//
__host__ void CellSorter::sort(
    particleArrayCUDA* hostPtr,
    const grid3DCUDA*  deviceGrid,
    uint32_t           num_to_sort,
    cudaStream_t       s)
{
    if (!initialized) return;

    const uint32_t nop = hostPtr->getNOP();
    if (num_to_sort == 0 || nop == 0) return;
    if (num_to_sort > nop) num_to_sort = nop;  // safety clamp

    // Ensure buffers are large enough
    buffers.ensure_capacity(nop, s);
    // Use capacity (not nop) because after the pointer-swap cycle scratch.ptr
    // becomes one of the SoA field buffers.  All SoA fields must be able to
    // hold up to soa.capacity elements so that future particle additions
    // (MPI exchange, injection) don't overflow the swapped-in buffer.
    const uint32_t cap = hostPtr->getCapacity();
    scratch.ensure_capacity(cap * sizeof(double));

    ParticleSoADevice* soa = hostPtr->getSoA();

    // ══════ DEBUG VALIDATION (compile-time toggle) ══════
    // Set to true to enable per-stage stream syncs + host-side validation.
    // WARNING: adds 4 extra cudaStreamSynchronize barriers — use only for debugging.
    constexpr bool SORT_DEBUG = false;

    double pre_sum_x = 0.0;
    if constexpr (SORT_DEBUG) {
        // Sync stream so we can safely read device data
        cudaErrChk(cudaStreamSynchronize(s));
        // Sum of x positions before sort (partial: first min(nop,256) particles)
        const int ncheck = (nop < 256u) ? nop : 256;
        std::vector<double> hx(ncheck);
        cudaErrChk(cudaMemcpy(hx.data(), soa->x, ncheck * sizeof(double), cudaMemcpyDefault));
        for (int k = 0; k < ncheck; k++) pre_sum_x += hx[k];
        printf("[SORT_DEBUG] nop=%u  num_to_sort=%u  num_cells=%d  cap=%u\n",
               nop, num_to_sort, num_cells, cap);
        printf("[SORT_DEBUG] SoA ptrs: u=%p v=%p w=%p q=%p x=%p y=%p z=%p t=%p  scratch=%p\n",
               (void*)soa->u, (void*)soa->v, (void*)soa->w, (void*)soa->q,
               (void*)soa->x, (void*)soa->y, (void*)soa->z, (void*)soa->t,
               scratch.ptr);
    }

    // ── Stage 1: Histogram ──
    buffers.zero_async(s);
    launch_cell_sort_histogram(
        soa->x, soa->y, soa->z,
        buffers.cell_counts, deviceGrid, num_cells, num_to_sort, s);

    if constexpr (SORT_DEBUG) {
        cudaErrChk(cudaStreamSynchronize(s));
        cudaErrChk(cudaPeekAtLastError());
        // Verify histogram sums to num_to_sort
        std::vector<int> h_counts(num_cells);
        cudaErrChk(cudaMemcpy(h_counts.data(), buffers.cell_counts,
                              num_cells * sizeof(int), cudaMemcpyDefault));
        long long histsum = 0;
        int negcount = 0;
        for (int k = 0; k < num_cells; k++) {
            histsum += h_counts[k];
            if (h_counts[k] < 0) negcount++;
        }
        if (histsum != (long long)num_to_sort || negcount > 0)
            printf("[SORT_DEBUG] HISTOGRAM BUG: sum(cell_counts)=%lld  expected=%u  neg=%d\n",
                   histsum, num_to_sort, negcount);
        else
            printf("[SORT_DEBUG] Histogram OK: sum=%lld\n", histsum);
    }

    // ── Stage 2: Prefix sum (Blelloch scan) ──
    launch_cell_sort_prefix_sum(
        buffers.cell_offsets, buffers.cell_counts,
        buffers.block_sums,
        num_cells, s);

    if constexpr (SORT_DEBUG) {
        cudaErrChk(cudaStreamSynchronize(s));
        cudaErrChk(cudaPeekAtLastError());
        // Verify prefix sum: offsets[0]==0, offsets[last]+counts[last]==num_to_sort
        std::vector<int> h_offsets(num_cells);
        std::vector<int> h_counts(num_cells);
        cudaErrChk(cudaMemcpy(h_offsets.data(), buffers.cell_offsets,
                              num_cells * sizeof(int), cudaMemcpyDefault));
        cudaErrChk(cudaMemcpy(h_counts.data(), buffers.cell_counts,
                              num_cells * sizeof(int), cudaMemcpyDefault));
        bool ps_ok = (h_offsets[0] == 0) &&
                     (h_offsets[num_cells-1] + h_counts[num_cells-1] == (int)num_to_sort);
        // Check monotonicity
        bool mono = true;
        for (int k = 1; k < num_cells; k++) {
            if (h_offsets[k] < h_offsets[k-1]) { mono = false; break; }
        }
        if (!ps_ok || !mono)
            printf("[SORT_DEBUG] PREFIX SUM BUG: offsets[0]=%d  last_offset+last_count=%d  expected=%u  mono=%d\n",
                   h_offsets[0], h_offsets[num_cells-1] + h_counts[num_cells-1], num_to_sort, mono);
        else
            printf("[SORT_DEBUG] Prefix sum OK\n");
    }

    // Preserve prefix sum
    cudaErrChk(cudaMemcpyAsync(
        buffers.cell_start_offsets, buffers.cell_offsets,
        num_cells * sizeof(int), cudaMemcpyDeviceToDevice, s));

    // ── Stage 3: Sorted indices (recomputes cell from position) ──
    launch_cell_sort_sorted_indices(
        buffers.sorted_indices, buffers.cell_offsets,
        soa->x, soa->y, soa->z, deviceGrid, num_to_sort, s);

    if constexpr (SORT_DEBUG) {
        cudaErrChk(cudaStreamSynchronize(s));
        cudaErrChk(cudaPeekAtLastError());
        // Verify sorted_indices: all in [0, nop), no duplicates (via seen-bitmap)
        std::vector<unsigned int> h_idx(num_to_sort);
        cudaErrChk(cudaMemcpy(h_idx.data(), buffers.sorted_indices,
                              num_to_sort * sizeof(unsigned int), cudaMemcpyDefault));
        int oob = 0;
        unsigned int maxidx = 0;
        std::vector<uint8_t> seen(nop, 0);
        int dups = 0;
        for (uint32_t k = 0; k < num_to_sort; k++) {
            unsigned int v = h_idx[k];
            if (v >= nop) { oob++; }
            else {
                if (seen[v]) dups++;
                seen[v] = 1;
            }
            if (v > maxidx) maxidx = v;
        }
        if (oob > 0 || dups > 0)
            printf("[SORT_DEBUG] SORTED_INDICES BUG: oob=%d dups=%d maxidx=%u nop=%u\n",
                   oob, dups, maxidx, nop);
        else
            printf("[SORT_DEBUG] Sorted indices OK: max=%u\n", maxidx);
    }

    // ── Stage 4: Sequential scatter-and-swap for all 8 SoA arrays ──
    //
    // scatter_and_swap: scatter field into scratch, then exchange pointers.
    // Stream ordering guarantees the scatter completes before the next kernel
    // reads from the (now-swapped) scratch allocation.
    //
    // ── Required barrier ──
    // The SCATTER_AND_SWAP below modifies host-pinned memory (soa->u, etc.)
    // that a prior cudaMemcpyAsync H2D (e.g. the struct sync in
    // MoverAwaitAndPclExchange) may still be reading via DMA.  Stream
    // ordering only sequences GPU-side execution; the host is free to race
    // ahead.  This single sync ensures all prior DMA reads from the host
    // struct have completed before we mutate the SoA pointer fields.
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

    if constexpr (SORT_DEBUG) {
        cudaErrChk(cudaStreamSynchronize(s));
        cudaErrChk(cudaPeekAtLastError());
        // Verify x positions after sort: same partial sum
        const int ncheck = (nop < 256u) ? nop : 256;
        std::vector<double> hx(ncheck);
        cudaErrChk(cudaMemcpy(hx.data(), soa->x, ncheck * sizeof(double), cudaMemcpyDefault));
        double post_sum_x = 0.0;
        for (int k = 0; k < ncheck; k++) post_sum_x += hx[k];
        // After sort, the first 256 particles are in different cells, so partial sum WILL differ.
        // But we can check for NaN or unreasonable values.
        int nan_count = 0;
        for (int k = 0; k < ncheck; k++)
            if (std::isnan(hx[k]) || std::isinf(hx[k])) nan_count++;
        printf("[SORT_DEBUG] Post-sort x[0..%d]: pre_partial_sum=%.6e  post_partial_sum=%.6e  NaN/Inf=%d\n",
               ncheck-1, pre_sum_x, post_sum_x, nan_count);
        printf("[SORT_DEBUG] Post-sort SoA ptrs: u=%p v=%p w=%p q=%p x=%p y=%p z=%p t=%p  scratch=%p\n",
               (void*)soa->u, (void*)soa->v, (void*)soa->w, (void*)soa->q,
               (void*)soa->x, (void*)soa->y, (void*)soa->z, (void*)soa->t,
               scratch.ptr);
    }
}
