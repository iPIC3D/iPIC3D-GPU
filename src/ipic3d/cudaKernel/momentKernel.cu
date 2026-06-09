#include "momentKernel.cuh"

using commonType = cudaTypeDouble; // calculation type


// ======= Particle-based moment deposition =======

__device__ __forceinline__ void depositParticleMomentsToNodes(
    uint32_t pidx,
    particleArrayCUDA* pclsArray,
    grid3DCUDA* grid,
    cudaTypeArray1<cudaMomentType> moments,
    commonType inv_dx,
    commonType inv_dy,
    commonType inv_dz)
{
    const int nxn = grid->nxn;
    const int nyn = grid->nyn;
    const int nzn = grid->nzn;
    const commonType xstart = grid->xStart;
    const commonType ystart = grid->yStart;
    const commonType zstart = grid->zStart;

    // Load particle data from SoA
    const commonType ui = pclsArray->getU()[pidx];
    const commonType vi = pclsArray->getV()[pidx];
    const commonType wi = pclsArray->getW()[pidx];
    const commonType xpcl = pclsArray->getX()[pidx];
    const commonType ypcl = pclsArray->getY()[pidx];
    const commonType zpcl = pclsArray->getZ()[pidx];
    const commonType qi = pclsArray->getQ()[pidx];
    const commonType velmoments[10] = {
        1.0, // charge density
        ui,  // momentum density
        vi,
        wi,
        ui * ui, // second-order moments
        ui * vi,
        ui * wi,
        vi * vi,
        vi * wi,
        wi * wi
    };

    //
    // compute the weights to distribute the moments
    //
    int ix = 2 + int(floor((xpcl - xstart) * inv_dx));
    int iy = 2 + int(floor((ypcl - ystart) * inv_dy));
    int iz = 2 + int(floor((zpcl - zstart) * inv_dz));
    // Safety clamp: prevent negative indices (would wrap to huge uint32_t in toOneDimIndex)
    // and cap at nxn-1/nyn-1/nzn-1 to avoid OOB on the moments array.
    if (ix < 1) ix = 1; if (ix > nxn - 1) ix = nxn - 1;
    if (iy < 1) iy = 1; if (iy > nyn - 1) iy = nyn - 1;
    if (iz < 1) iz = 1; if (iz > nzn - 1) iz = nzn - 1;
    const commonType xi0 = xpcl - grid->getXN(ix-1);
    const commonType eta0 = ypcl - grid->getYN(iy - 1);
    const commonType zeta0 = zpcl - grid->getZN(iz - 1);
    const commonType xi1 = grid->getXN(ix) - xpcl;
    const commonType eta1 = grid->getYN(iy) - ypcl;
    const commonType zeta1 = grid->getZN(iz) - zpcl;
    const commonType invVOLqi = grid->invVOL * qi;
    const commonType weight0 = invVOLqi * xi0;
    const commonType weight1 = invVOLqi * xi1;
    const commonType weight00 = weight0 * eta0;
    const commonType weight01 = weight0 * eta1;
    const commonType weight10 = weight1 * eta0;
    const commonType weight11 = weight1 * eta1;
    const commonType weights[8] = {
        weight00 * zeta0 * grid->invVOL, // weight000
        weight00 * zeta1 * grid->invVOL, // weight001
        weight01 * zeta0 * grid->invVOL, // weight010
        weight01 * zeta1 * grid->invVOL, // weight011
        weight10 * zeta0 * grid->invVOL, // weight100
        weight10 * zeta1 * grid->invVOL, // weight101
        weight11 * zeta0 * grid->invVOL, // weight110
        weight11 * zeta1 * grid->invVOL  // weight111
    };

    uint32_t posIndex[8];
    posIndex[0] = toOneDimIndex(nxn, nyn, nzn, ix, iy, iz);
    posIndex[1] = toOneDimIndex(nxn, nyn, nzn, ix, iy, iz-1);
    posIndex[2] = toOneDimIndex(nxn, nyn, nzn, ix, iy-1, iz);
    posIndex[3] = toOneDimIndex(nxn, nyn, nzn, ix, iy-1, iz-1);
    posIndex[4] = toOneDimIndex(nxn, nyn, nzn, ix-1, iy, iz);
    posIndex[5] = toOneDimIndex(nxn, nyn, nzn, ix-1, iy, iz-1);
    posIndex[6] = toOneDimIndex(nxn, nyn, nzn, ix-1, iy-1, iz);
    posIndex[7] = toOneDimIndex(nxn, nyn, nzn, ix-1, iy-1, iz-1);
    const uint32_t oneDensity = nxn * nyn * nzn;
    for (int m = 0; m < 10; m++)    // 10 densities
    for (int c = 0; c < 8; c++)     // 8 grid nodes
    {
        atomicAdd(&moments[oneDensity*m + posIndex[c]], velmoments[m] * weights[c]); // device scope atomic, should be system scope if p2p direct access
    }
}

/**
 * @brief Deposit moments for the stayed-particle prefix plus newly appended particles.
 *
 * The kernel iterates with a grid-stride loop over the active particle range
 * and atomically accumulates the 10 velocity moments onto the surrounding
 * eight nodes.
 *
 * @param appendCount Device pointer to the number of appended particles.
 * @param momentParam Device-side moment parameter bundle for one species.
 * @param grid Device-side grid descriptor.
 * @param moments Packed moment output buffer for this species.
 */
__global__ void momentKernelStayed(const uint32_t* appendCount, momentParameter* momentParam,
                                grid3DCUDA* grid,
                                cudaTypeArray1<cudaMomentType> moments)
{

    const uint tidx = blockIdx.x * blockDim.x + threadIdx.x;
    const uint gridSize = blockDim.x * gridDim.x;
    auto pclsArray = momentParam->pclsArray;
    const uint totPcl = pclsArray->getNOP() + *appendCount;

    for(uint pidx = tidx; pidx < totPcl; pidx += gridSize )
    {
        if(momentParam->departureArray->getArray()[pidx].dest != 0)continue; // skip particles already marked for departure

        depositParticleMomentsToNodes(pidx, pclsArray, grid, moments,
                                      grid->invdx, grid->invdy, grid->invdz);
    }
}
/**
 * @brief Deposit moments for the unsorted tail appended after compaction.
 *
 * @param momentParam Device-side moment parameter bundle for one species.
 * @param grid Device-side grid descriptor.
 * @param moments Packed moment output buffer for this species.
 * @param stayedParticle Number of already-compacted stayed particles at the
 *        front of the SoA buffer.
 */
__global__ void momentKernelNew(momentParameter* momentParam,
                                grid3DCUDA* grid,
                                cudaTypeArray1<cudaMomentType> moments,
                                int stayedParticle)
{

    uint pidx = stayedParticle + blockIdx.x * blockDim.x + threadIdx.x;
    auto pclsArray = momentParam->pclsArray;
    if(pidx >= pclsArray->getNOP())return;

    depositParticleMomentsToNodes(pidx, pclsArray, grid, moments,
                                  grid->invdx, grid->invdy, grid->invdz);
}

__global__ void heatFluxKernelUnsorted(
    momentParameter* momentParam,
    grid3DCUDA* grid,
    const cudaTypeArray1<cudaMomentType> bulkMoments,
    cudaTypeArray1<cudaMomentType> heatFlux,
    cudaMomentType qom,
    cudaMomentType rhoFloor)
{
    if (qom == 0.0) return;

    const uint pidx = blockIdx.x * blockDim.x + threadIdx.x;
    auto pclsArray = momentParam->pclsArray;
    if (pidx >= pclsArray->getNOP()) return;

    const commonType inv_dx = 1.0 / grid->dx;
    const commonType inv_dy = 1.0 / grid->dy;
    const commonType inv_dz = 1.0 / grid->dz;
    const int nxn = grid->nxn;
    const int nyn = grid->nyn;
    const int nzn = grid->nzn;
    const commonType xstart = grid->xStart;
    const commonType ystart = grid->yStart;
    const commonType zstart = grid->zStart;
    const uint32_t oneDensity = nxn * nyn * nzn;

    const commonType ui = pclsArray->getU()[pidx];
    const commonType vi = pclsArray->getV()[pidx];
    const commonType wi = pclsArray->getW()[pidx];
    const commonType xpcl = pclsArray->getX()[pidx];
    const commonType ypcl = pclsArray->getY()[pidx];
    const commonType zpcl = pclsArray->getZ()[pidx];
    const commonType qi = pclsArray->getQ()[pidx];

    int ix = 2 + int(floor((xpcl - xstart) * inv_dx));
    int iy = 2 + int(floor((ypcl - ystart) * inv_dy));
    int iz = 2 + int(floor((zpcl - zstart) * inv_dz));
    if (ix < 1) ix = 1; if (ix > nxn - 1) ix = nxn - 1;
    if (iy < 1) iy = 1; if (iy > nyn - 1) iy = nyn - 1;
    if (iz < 1) iz = 1; if (iz > nzn - 1) iz = nzn - 1;

    const commonType xi0 = xpcl - grid->getXN(ix - 1);
    const commonType eta0 = ypcl - grid->getYN(iy - 1);
    const commonType zeta0 = zpcl - grid->getZN(iz - 1);
    const commonType xi1 = grid->getXN(ix) - xpcl;
    const commonType eta1 = grid->getYN(iy) - ypcl;
    const commonType zeta1 = grid->getZN(iz) - zpcl;
    const commonType invVOLqi = grid->invVOL * qi;
    const commonType weight0 = invVOLqi * xi0;
    const commonType weight1 = invVOLqi * xi1;
    const commonType weight00 = weight0 * eta0;
    const commonType weight01 = weight0 * eta1;
    const commonType weight10 = weight1 * eta0;
    const commonType weight11 = weight1 * eta1;

    const commonType weights[8] = {
        weight00 * zeta0 * grid->invVOL,
        weight00 * zeta1 * grid->invVOL,
        weight01 * zeta0 * grid->invVOL,
        weight01 * zeta1 * grid->invVOL,
        weight10 * zeta0 * grid->invVOL,
        weight10 * zeta1 * grid->invVOL,
        weight11 * zeta0 * grid->invVOL,
        weight11 * zeta1 * grid->invVOL
    };

    uint32_t posIndex[8];
    posIndex[0] = toOneDimIndex(nxn, nyn, nzn, ix, iy, iz);
    posIndex[1] = toOneDimIndex(nxn, nyn, nzn, ix, iy, iz - 1);
    posIndex[2] = toOneDimIndex(nxn, nyn, nzn, ix, iy - 1, iz);
    posIndex[3] = toOneDimIndex(nxn, nyn, nzn, ix, iy - 1, iz - 1);
    posIndex[4] = toOneDimIndex(nxn, nyn, nzn, ix - 1, iy, iz);
    posIndex[5] = toOneDimIndex(nxn, nyn, nzn, ix - 1, iy, iz - 1);
    posIndex[6] = toOneDimIndex(nxn, nyn, nzn, ix - 1, iy - 1, iz);
    posIndex[7] = toOneDimIndex(nxn, nyn, nzn, ix - 1, iy - 1, iz - 1);

    for (int c = 0; c < 8; ++c) {
        const uint32_t node = posIndex[c];
        const commonType rho = bulkMoments[node];
        if (fabs(rho) <= rhoFloor) continue;

        const commonType ux = bulkMoments[oneDensity + node] / rho;
        const commonType uy = bulkMoments[2 * oneDensity + node] / rho;
        const commonType uz = bulkMoments[3 * oneDensity + node] / rho;
        const commonType cx = ui - ux;
        const commonType cy = vi - uy;
        const commonType cz = wi - uz;
        const commonType cx2 = cx * cx;
        const commonType cy2 = cy * cy;
        const commonType cz2 = cz * cz;
        const commonType massWeight = weights[c] / qom;

        const commonType q[10] = {
            cx2 * cx,
            cx2 * cy,
            cx2 * cz,
            cx * cy2,
            cx * cy * cz,
            cx * cz2,
            cy2 * cy,
            cy2 * cz,
            cy * cz2,
            cz2 * cz
        };

        for (int m = 0; m < 10; ++m)
            atomicAdd(&heatFlux[oneDensity * m + node], q[m] * massWeight);
    }
}


// ======= Cell-aware sorted moment deposition =======

/**
 * @brief Deposit moments from the cell-sorted SoA prefix with one warp per cell.
 *
 * Particles belonging to one cell are assumed contiguous in
 * `cell_start_offsets`. Contributions are reduced within the warp so the
 * kernel performs one atomic add per `(moment, node)` pair instead of one
 * per particle contribution.
 *
 * @param cell_start_offsets Device array of per-cell particle-start offsets.
 * @param num_cells Number of populated cells in the sorted prefix.
 * @param num_to_sort Number of particles in the sorted prefix.
 * @param pclsArray Device-side particle SoA container.
 * @param grid Device-side grid descriptor.
 * @param moments Packed moment output buffer for this species.
 */
__global__ void cellAwareMomentKernel(
    const int*                    __restrict__ cell_start_offsets,
    int                           num_cells,
    uint32_t                      num_to_sort,
    particleArrayCUDA*            pclsArray,
    grid3DCUDA*                   grid,
    cudaTypeArray1<cudaMomentType> moments)
{
    const int global_tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int warp_id    = global_tid / WARP_SIZE;   // one warp per cell
    const int lane       = global_tid & (WARP_SIZE - 1);

    if (warp_id >= num_cells) return;

    const int cell = warp_id;
    const int cell_begin = cell_start_offsets[cell];
    const int cell_end   = (cell < num_cells - 1)
                         ? cell_start_offsets[cell + 1]
                         : static_cast<int>(num_to_sort);
    const int cell_count = cell_end - cell_begin;
    if (cell_count <= 0) return;

    // Recover node indices from the flat cell index.
    const int nxc = grid->nxc;
    const int nyc = grid->nyc;
    const int nxn = grid->nxn;
    const int nyn = grid->nyn;
    const int nzn = grid->nzn;

    const int cz  = cell / (nxc * nyc);
    const int rem = cell - cz * (nxc * nyc);
    const int cy  = rem / nxc;
    const int cx  = rem - cy * nxc;

    // Node index = cell index + 1 (ghost cell offset)
    const int ix = cx + 1;
    const int iy = cy + 1;
    const int iz = cz + 1;

    // 8 surrounding node flat indices (shared by all particles in this cell)
    const uint32_t oneDensity = nxn * nyn * nzn;
    uint32_t posIndex[8];
    posIndex[0] = toOneDimIndex(nxn, nyn, nzn, ix,   iy,   iz  );
    posIndex[1] = toOneDimIndex(nxn, nyn, nzn, ix,   iy,   iz-1);
    posIndex[2] = toOneDimIndex(nxn, nyn, nzn, ix,   iy-1, iz  );
    posIndex[3] = toOneDimIndex(nxn, nyn, nzn, ix,   iy-1, iz-1);
    posIndex[4] = toOneDimIndex(nxn, nyn, nzn, ix-1, iy,   iz  );
    posIndex[5] = toOneDimIndex(nxn, nyn, nzn, ix-1, iy,   iz-1);
    posIndex[6] = toOneDimIndex(nxn, nyn, nzn, ix-1, iy-1, iz  );
    posIndex[7] = toOneDimIndex(nxn, nyn, nzn, ix-1, iy-1, iz-1);

    // Cache grid constants used by the weight computation.
    const commonType inv_vol = grid->invVOL;
    const commonType _dx     = grid->dx;
    const commonType _dy     = grid->dy;
    const commonType _dz     = grid->dz;
    const commonType xstart  = grid->xStart;
    const commonType ystart  = grid->yStart;
    const commonType zstart  = grid->zStart;

    // Node positions for this cell (precomputed once, shared across all particles)
    // getXN(ix-1) = xStart + (ix-2)*dx,  getXN(ix) = xStart + (ix-1)*dx
    const commonType xn_lo = xstart + (ix - 2) * _dx;
    const commonType xn_hi = xstart + (ix - 1) * _dx;
    const commonType yn_lo = ystart + (iy - 2) * _dy;
    const commonType yn_hi = ystart + (iy - 1) * _dy;
    const commonType zn_lo = zstart + (iz - 2) * _dz;
    const commonType zn_hi = zstart + (iz - 1) * _dz;

    // SoA pointers (read once from device struct)
    const commonType* __restrict__ px = pclsArray->getX();
    const commonType* __restrict__ py = pclsArray->getY();
    const commonType* __restrict__ pz = pclsArray->getZ();
    const commonType* __restrict__ pu = pclsArray->getU();
    const commonType* __restrict__ pv = pclsArray->getV();
    const commonType* __restrict__ pw = pclsArray->getW();
    const commonType* __restrict__ pq = pclsArray->getQ();

    // Process particles in warp-sized batches.
    for (int batch = 0; batch < cell_count; batch += WARP_SIZE) {
        const int local_idx = batch + lane;
        const bool active = local_idx < cell_count;
        const int pidx = cell_begin + local_idx;

        // Load particle data (inactive lanes contribute zero)
        commonType ui = 0, vi = 0, wi = 0, qi = 0;
        commonType xpcl = 0, ypcl = 0, zpcl = 0;

        if (active) {
            ui   = pu[pidx];
            vi   = pv[pidx];
            wi   = pw[pidx];
            qi   = pq[pidx];
            xpcl = px[pidx];
            ypcl = py[pidx];
            zpcl = pz[pidx];
        }

        // Trilinear weights using the same arithmetic as momentKernelStayed.
        const commonType xi0   = xpcl - xn_lo;
        const commonType xi1   = xn_hi - xpcl;
        const commonType eta0  = ypcl - yn_lo;
        const commonType eta1  = yn_hi - ypcl;
        const commonType zeta0 = zpcl - zn_lo;
        const commonType zeta1 = zn_hi - zpcl;

        const commonType invVOLqi = inv_vol * qi;
        const commonType w0  = invVOLqi * xi0;
        const commonType w1  = invVOLqi * xi1;
        const commonType w00 = w0 * eta0;
        const commonType w01 = w0 * eta1;
        const commonType w10 = w1 * eta0;
        const commonType w11 = w1 * eta1;

        const commonType vm[10] = {
            active ? 1.0 : 0.0, // charge density
            ui,
            vi,
            wi,
            ui * ui,
            ui * vi,
            ui * wi,
            vi * vi,
            vi * wi,
            wi * wi
        };
        const commonType wt[8] = {
            w00 * zeta0 * inv_vol,
            w00 * zeta1 * inv_vol,
            w01 * zeta0 * inv_vol,
            w01 * zeta1 * inv_vol,
            w10 * zeta0 * inv_vol,
            w10 * zeta1 * inv_vol,
            w11 * zeta0 * inv_vol,
            w11 * zeta1 * inv_vol
        };

        // Warp reduction + single atomicAdd per (moment, node) pair
        for (int m = 0; m < 10; m++) {
            for (int c = 0; c < 8; c++) {
                commonType val = vm[m] * wt[c];
                val = warp_reduce_sum(val);
                if (lane == 0)
                    atomicAdd(&moments[oneDensity * m + posIndex[c]], val);
            }
        }
    }
}
