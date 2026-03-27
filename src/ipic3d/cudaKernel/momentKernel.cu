

#include "ipichdf5.h"
#include "EMfields3D.h"
#include "Collective.h"
#include "Basic.h"
#include "Com3DNonblk.h"
#include "VCtopology3D.h"
#include "Grid3DCU.h"
#include "CG.h"
#include "GMRES.h"
#include "ParticleSoAHost.h"
#include "Moments.h"
#include "Parameters.h"
#include "ompdefs.h"
#include "debug.h"
#include "string.h"
#include "mic_particles.h"
#include "TimeTasks.h"
#include "ipicmath.h" // for roundup_to_multiple
#include "Alloc.h"
#include "asserts.h"

#include "cudaTypeDef.cuh"
#include "momentKernel.cuh"
#include "gridCUDA.cuh"
#include "particleArrayCUDA.cuh"
#include "particleExchange.cuh"

using commonType = cudaTypeDouble; // calculation type


/**
 * @brief moment kernel, one particle per thread
 * @details the moment kernel should be launched in species
 *          if these're 4 species, launch 4 times in different streams
 * 
 * @param grid 
 * @param _pcls the particles of a species
 * @param moments array4, [x][y][z][density], 
 *                  here[nxn][nyn][nzn][10], must be 0 before kernel launch
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
        if(momentParam->departureArray->getArray()[pidx].dest != 0)continue; // return the exiting particles, which are out of current domian

        // can be const
        const commonType& inv_dx = grid->invdx;
        const commonType& inv_dy = grid->invdy;
        const commonType& inv_dz = grid->invdz;
        const int& nxn = grid->nxn; // nxn
        const int& nyn = grid->nyn;
        const int& nzn = grid->nzn;
        const commonType& xstart = grid->xStart; // x start
        const commonType& ystart = grid->yStart;
        const commonType& zstart = grid->zStart;
        

        // Load particle data from SoA
        const commonType ui = pclsArray->getU()[pidx];
        const commonType vi = pclsArray->getV()[pidx];
        const commonType wi = pclsArray->getW()[pidx];
        const commonType xpcl = pclsArray->getX()[pidx];
        const commonType ypcl = pclsArray->getY()[pidx];
        const commonType zpcl = pclsArray->getZ()[pidx];
        const commonType qi = pclsArray->getQ()[pidx];
        const commonType uui = ui * ui;
        const commonType uvi = ui * vi;
        const commonType uwi = ui * wi;
        const commonType vvi = vi * vi;
        const commonType vwi = vi * wi;
        const commonType wwi = wi * wi;
        commonType velmoments[10];
        velmoments[0] = 1.; // charge density
        velmoments[1] = ui; // momentum density
        velmoments[2] = vi;
        velmoments[3] = wi;
        velmoments[4] = uui; // second time momentum
        velmoments[5] = uvi;
        velmoments[6] = uwi;
        velmoments[7] = vvi;
        velmoments[8] = vwi;
        velmoments[9] = wwi;

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
        commonType weights[8]; // put the invVOL here
        weights[0] = weight00 * zeta0 * grid->invVOL; // weight000
        weights[1] = weight00 * zeta1 * grid->invVOL; // weight001
        weights[2] = weight01 * zeta0 * grid->invVOL; // weight010
        weights[3] = weight01 * zeta1 * grid->invVOL; // weight011
        weights[4] = weight10 * zeta0 * grid->invVOL; // weight100
        weights[5] = weight10 * zeta1 * grid->invVOL; // weight101
        weights[6] = weight11 * zeta0 * grid->invVOL; // weight110
        weights[7] = weight11 * zeta1 * grid->invVOL; // weight111


        uint32_t posIndex[8];
        posIndex[0] = toOneDimIndex(nxn, nyn, nzn, ix, iy, iz);
        posIndex[1] = toOneDimIndex(nxn, nyn, nzn, ix, iy, iz-1);
        posIndex[2] = toOneDimIndex(nxn, nyn, nzn, ix, iy-1, iz);
        posIndex[3] = toOneDimIndex(nxn, nyn, nzn, ix, iy-1, iz-1);
        posIndex[4] = toOneDimIndex(nxn, nyn, nzn, ix-1, iy, iz);
        posIndex[5] = toOneDimIndex(nxn, nyn, nzn, ix-1, iy, iz-1);
        posIndex[6] = toOneDimIndex(nxn, nyn, nzn, ix-1, iy-1, iz);
        posIndex[7] = toOneDimIndex(nxn, nyn, nzn, ix-1, iy-1, iz-1);
        uint32_t oneDensity = nxn * nyn * nzn;
        for (int m = 0; m < 10; m++)    // 10 densities
        for (int c = 0; c < 8; c++)     // 8 grid nodes
        {
            atomicAdd(&moments[oneDensity*m + posIndex[c]], velmoments[m] * weights[c]); // device scope atomic, should be system scope if p2p direct access
        }
    }
}



__global__ void momentKernelNew(momentParameter* momentParam,
                                grid3DCUDA* grid,
                                cudaTypeArray1<cudaMomentType> moments,
                                int stayedParticle)
{

    uint pidx = stayedParticle + blockIdx.x * blockDim.x + threadIdx.x;
    auto pclsArray = momentParam->pclsArray;
    if(pidx >= pclsArray->getNOP())return;

    // can be shared
    const commonType inv_dx = 1.0 / grid->dx;
    const commonType inv_dy = 1.0 / grid->dy;
    const commonType inv_dz = 1.0 / grid->dz;
    const int nxn = grid->nxn; // nxn
    const int nyn = grid->nyn;
    const int nzn = grid->nzn;
    const commonType xstart = grid->xStart; // x start
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
    const commonType uui = ui * ui;
    const commonType uvi = ui * vi;
    const commonType uwi = ui * wi;
    const commonType vvi = vi * vi;
    const commonType vwi = vi * wi;
    const commonType wwi = wi * wi;
    commonType velmoments[10];
    velmoments[0] = 1.; // charge density
    velmoments[1] = ui; // momentum density
    velmoments[2] = vi;
    velmoments[3] = wi;
    velmoments[4] = uui; // second time momentum
    velmoments[5] = uvi;
    velmoments[6] = uwi;
    velmoments[7] = vvi;
    velmoments[8] = vwi;
    velmoments[9] = wwi;

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
    commonType weights[8]; // put the invVOL here
    weights[0] = weight00 * zeta0 * grid->invVOL; // weight000
    weights[1] = weight00 * zeta1 * grid->invVOL; // weight001
    weights[2] = weight01 * zeta0 * grid->invVOL; // weight010
    weights[3] = weight01 * zeta1 * grid->invVOL; // weight011
    weights[4] = weight10 * zeta0 * grid->invVOL; // weight100
    weights[5] = weight10 * zeta1 * grid->invVOL; // weight101
    weights[6] = weight11 * zeta0 * grid->invVOL; // weight110
    weights[7] = weight11 * zeta1 * grid->invVOL; // weight111


    uint32_t posIndex[8];
    posIndex[0] = toOneDimIndex(nxn, nyn, nzn, ix, iy, iz);
    posIndex[1] = toOneDimIndex(nxn, nyn, nzn, ix, iy, iz-1);
    posIndex[2] = toOneDimIndex(nxn, nyn, nzn, ix, iy-1, iz);
    posIndex[3] = toOneDimIndex(nxn, nyn, nzn, ix, iy-1, iz-1);
    posIndex[4] = toOneDimIndex(nxn, nyn, nzn, ix-1, iy, iz);
    posIndex[5] = toOneDimIndex(nxn, nyn, nzn, ix-1, iy, iz-1);
    posIndex[6] = toOneDimIndex(nxn, nyn, nzn, ix-1, iy-1, iz);
    posIndex[7] = toOneDimIndex(nxn, nyn, nzn, ix-1, iy-1, iz-1);
    uint32_t oneDensity = nxn * nyn * nzn;
    for (int m = 0; m < 10; m++)    // 10 densities
    for (int c = 0; c < 8; c++)     // 8 grid nodes
    {
        atomicAdd(&moments[oneDensity*m + posIndex[c]], velmoments[m] * weights[c]); // device scope atomic, should be system scope if p2p direct access
    }


}


// ============================================================================
// Cell-aware moment kernel for sorted particles (warp-per-cell)
// ============================================================================
//
// One warp processes all particles in one cell.  Particles are contiguous in
// the sorted SoA prefix at indices [cell_start_offsets[cell], cell_end).
// Per-particle contributions are warp-reduced via __shfl_down_sync before a
// single atomicAdd per (moment, node) pair — 80 atomicAdds per CELL instead
// of 80 per particle.
//
// Launch: <<<(num_cells * WARP_SIZE + blockDim - 1) / blockDim, blockDim>>>
//
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

    // ── Recover node indices from flat cell index ──
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

    // ── Grid constants for weight computation ──
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

    // ── Process particles in warp-sized batches ──
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

        // 10 velocity moments
        const commonType vm0  = active ? 1.0 : 0.0;  // charge density
        const commonType vm1  = ui;
        const commonType vm2  = vi;
        const commonType vm3  = wi;
        const commonType vm4  = ui * ui;
        const commonType vm5  = ui * vi;
        const commonType vm6  = ui * wi;
        const commonType vm7  = vi * vi;
        const commonType vm8  = vi * wi;
        const commonType vm9  = wi * wi;

        // Trilinear weights — exact same arithmetic as momentKernelStayed
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

        const commonType wt0 = w00 * zeta0 * inv_vol;
        const commonType wt1 = w00 * zeta1 * inv_vol;
        const commonType wt2 = w01 * zeta0 * inv_vol;
        const commonType wt3 = w01 * zeta1 * inv_vol;
        const commonType wt4 = w10 * zeta0 * inv_vol;
        const commonType wt5 = w10 * zeta1 * inv_vol;
        const commonType wt6 = w11 * zeta0 * inv_vol;
        const commonType wt7 = w11 * zeta1 * inv_vol;

        // Store velmoments and weights in arrays for the reduction loop
        const commonType vm[10] = {vm0, vm1, vm2, vm3, vm4, vm5, vm6, vm7, vm8, vm9};
        const commonType wt[8]  = {wt0, wt1, wt2, wt3, wt4, wt5, wt6, wt7};

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

