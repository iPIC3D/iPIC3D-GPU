

#include "ipichdf5.h"
#include "EMfields3D.h"
#include "Collective.h"
#include "Basic.h"
#include "Com3DNonblk.h"
#include "VCtopology3D.h"
#include "Grid3DCU.h"
#include "CG.h"
#include "GMRES.h"
#include "Particles3Dcomm.h"
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
#include "Particles3D.h"

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

