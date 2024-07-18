

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

using commonType = cudaCommonType;


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
__global__ void momentKernel(momentParameter* momentParam,
                            grid3DCUDA* grid,
                            cudaTypeArray1<commonType> moments)
{

    uint pidx = blockIdx.x * blockDim.x + threadIdx.x;
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

    //initialize moments to 0, now we give the responsibility to host
    // if(momentParam->nop >= nxn * nyn * nzn){
    //     if(pidx < (nxn * nyn * nzn)){
    //         for(int i = 0; i < 10; i++)moments[pidx * 10 + i] = 0;
    //     }
    // }else{


    // }
    

    const SpeciesParticle &pcl = pclsArray->getpcls()[pidx];
    // compute the quadratic moments of velocity
    const commonType ui = pcl.get_u();
    const commonType vi = pcl.get_v();
    const commonType wi = pcl.get_w();
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
    const int ix = 2 + int(floor((pcl.get_x() - xstart) * inv_dx));
    const int iy = 2 + int(floor((pcl.get_y() - ystart) * inv_dy));
    const int iz = 2 + int(floor((pcl.get_z() - zstart) * inv_dz));
    const commonType xi0 = pcl.get_x() - grid->getXN(ix-1); // calculate here
    const commonType eta0 = pcl.get_y() - grid->getYN(iy - 1);
    const commonType zeta0 = pcl.get_z() - grid->getZN(iz - 1);
    const commonType xi1 = grid->getXN(ix) - pcl.get_x();
    const commonType eta1 = grid->getYN(iy) - pcl.get_y();
    const commonType zeta1 = grid->getZN(iz) - pcl.get_z();
    const commonType qi = pcl.get_q();
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

/*
    // add particle to moments
    cudaTypeArray1<commonType> momentsArray[8]; // 
    auto moments00 = moments[ix][iy];
    auto moments01 = moments[ix][iy - 1];
    auto moments10 = moments[ix - 1][iy];
    auto moments11 = moments[ix - 1][iy - 1];
    momentsArray[0] = moments00[iz];     // moments000
    momentsArray[1] = moments00[iz - 1]; // moments001
    momentsArray[2] = moments01[iz];     // moments010
    momentsArray[3] = moments01[iz - 1]; // moments011
    momentsArray[4] = moments10[iz];     // moments100
    momentsArray[5] = moments10[iz - 1]; // moments101
    momentsArray[6] = moments11[iz];     // moments110
    momentsArray[7] = moments11[iz - 1]; // moments111

    for (int m = 0; m < 10; m++)    // 10 densities
    for (int c = 0; c < 8; c++)     // 8 grid nodes
    {
        // weights a for particle to grid interpolation
        // add the contribution of one particle to the grid, should be atomic 
        //momentsArray[c][m] += velmoments[m] * weights[c];
        atomicAdd(&momentsArray[c][m], velmoments[m] * weights[c]); // device scope atomic, should be system scope if p2p direct access
    }
*/

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

    {
    // reduction, the kernel is launched on specieses, replace this by copy to host
    // can be done in previous step, multiple the index to the weights
    // rhons[is][i][j][k] += invVOL * moments[i][j][k][0]; // charge density
    // Jxs[is][i][j][k] += invVOL * moments[i][j][k][1];   // current density
    // Jys[is][i][j][k] += invVOL * moments[i][j][k][2];
    // Jzs[is][i][j][k] += invVOL * moments[i][j][k][3];
    // pXXsn[is][i][j][k] += invVOL * moments[i][j][k][4]; // pressure density
    // pXYsn[is][i][j][k] += invVOL * moments[i][j][k][5];
    // pXZsn[is][i][j][k] += invVOL * moments[i][j][k][6];
    // pYYsn[is][i][j][k] += invVOL * moments[i][j][k][7];
    // pYZsn[is][i][j][k] += invVOL * moments[i][j][k][8];
    // pZZsn[is][i][j][k] += invVOL * moments[i][j][k][9];

    
    //communicateGhostP2G(i);
    }


}
