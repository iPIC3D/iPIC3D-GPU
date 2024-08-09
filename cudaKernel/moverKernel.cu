

#include <iostream>
#include <math.h>
#include <limits.h>
#include "asserts.h"
#include "VCtopology3D.h"
#include "Collective.h"
#include "Basic.h"
#include "Grid3DCU.h"
#include "Field.h"
#include "ipicdefs.h"
#include "TimeTasks.h"
#include "parallel.h"
#include "Particles3D.h"

#include "mic_particles.h"
#include "debug.h"
#include <complex>

#include "cudaTypeDef.cuh"
#include "moverKernel.cuh"
#include "gridCUDA.cuh"
#include "particleArrayCUDA.cuh"
#include "hashedSum.cuh"

using commonType = cudaCommonType;

__device__ constexpr bool cap_velocity() { return false; }

__host__ __device__ void get_field_components_for_cell(
    const commonType *field_components[8],
    cudaTypeArray1<commonType> fieldForPcls, grid3DCUDA *grid,
    int cx, int cy, int cz);

__global__ void moverKernel(moverParameter *moverParam,
                            cudaTypeArray1<commonType> fieldForPcls,
                            grid3DCUDA *grid)
{
    uint pidx = blockIdx.x * blockDim.x + threadIdx.x;

    auto pclsArray = moverParam->pclsArray;
    if(pidx >= pclsArray->getNOP())return;
    
    const commonType dto2 = .5 * moverParam->dt,
                     qdto2mc = moverParam->qom * dto2 / moverParam->c;

    // copy the particle
    SpeciesParticle *pcl = pclsArray->getpcls() + pidx;

    const commonType xorig = pcl->get_x();
    const commonType yorig = pcl->get_y();
    const commonType zorig = pcl->get_z();
    const commonType uorig = pcl->get_u();
    const commonType vorig = pcl->get_v();
    const commonType worig = pcl->get_w();
    commonType xavg = xorig;
    commonType yavg = yorig;
    commonType zavg = zorig;
    commonType uavg, vavg, wavg;
    commonType uavg_old = uorig;
    commonType vavg_old = vorig;
    commonType wavg_old = worig;

    int innter = 0;
    const commonType PC_err_2 = 1E-12;  // square of error tolerance
    commonType currErr = PC_err_2 + 1.; // initialize to a larger value

    // calculate the average velocity iteratively
    while (currErr > PC_err_2 && innter < moverParam->NiterMover)
    {

        // compute weights for field components
        //
        commonType weights[8];
        int cx, cy, cz;
        grid->get_safe_cell_and_weights(xavg, yavg, zavg, cx, cy, cz, weights);

        const commonType *field_components[8];
        get_field_components_for_cell(field_components, fieldForPcls, grid, cx, cy, cz);

        commonType sampled_field[8];
        for (int i = 0; i < 8; i++)
            sampled_field[i] = 0;
        commonType &Bxl = sampled_field[0];
        commonType &Byl = sampled_field[1];
        commonType &Bzl = sampled_field[2];
        commonType &Exl = sampled_field[0 + moverParam->DFIELD_3or4];
        commonType &Eyl = sampled_field[1 + moverParam->DFIELD_3or4];
        commonType &Ezl = sampled_field[2 + moverParam->DFIELD_3or4];
        const int num_field_components = 2 * moverParam->DFIELD_3or4;
        for (int c = 0; c < 8; c++) // grid node
        {
            const commonType *field_components_c = field_components[c];
            const commonType weights_c = weights[c];

            for (int i = 0; i < num_field_components; i++) // field items
            {
                sampled_field[i] += weights_c * field_components_c[i];
            }
        }
        const commonType Omx = qdto2mc * Bxl;
        const commonType Omy = qdto2mc * Byl;
        const commonType Omz = qdto2mc * Bzl;

        // end interpolation
        const commonType omsq = (Omx * Omx + Omy * Omy + Omz * Omz);
        const commonType denom = 1.0 / (1.0 + omsq);
        // solve the position equation
        const commonType ut = uorig + qdto2mc * Exl;
        const commonType vt = vorig + qdto2mc * Eyl;
        const commonType wt = worig + qdto2mc * Ezl;
        // const commonType udotb = ut * Bxl + vt * Byl + wt * Bzl;
        const commonType udotOm = ut * Omx + vt * Omy + wt * Omz;
        // solve the velocity equation
        uavg = (ut + (vt * Omz - wt * Omy + udotOm * Omx)) * denom;
        vavg = (vt + (wt * Omx - ut * Omz + udotOm * Omy)) * denom;
        wavg = (wt + (ut * Omy - vt * Omx + udotOm * Omz)) * denom;
        // update average position
        xavg = xorig + uavg * dto2;
        yavg = yorig + vavg * dto2;
        zavg = zorig + wavg * dto2;

        innter++;
        currErr = ((uavg_old - uavg) * (uavg_old - uavg) + (vavg_old - vavg) * (vavg_old - vavg) + (wavg_old - wavg) * (wavg_old - wavg)) /
                  (uavg_old * uavg_old + vavg_old * vavg_old + wavg_old * wavg_old);
        // capture the new velocity for the next iteration
        uavg_old = uavg;
        vavg_old = vavg;
        wavg_old = wavg;

    } // end of iteration

    // update the final position and velocity
    if (cap_velocity()) //used to limit the speed of particles under c
    {
        auto umax = moverParam->umax;
        auto vmax = moverParam->vmax;
        auto wmax = moverParam->wmax;
        auto umin = moverParam->umin;
        auto vmin = moverParam->vmin;
        auto wmin = moverParam->wmin;

        bool cap = (abs(uavg) > umax || abs(vavg) > vmax || abs(wavg) > wmax) ? true : false;
        // we could do something more smooth or sophisticated
        if (cap)
        {
            if (uavg > umax)
                uavg = umax;
            else if (uavg < umin)
                uavg = umin;
            if (vavg > vmax)
                vavg = vmax;
            else if (vavg < vmin)
                vavg = vmin;
            if (wavg > wmax)
                wavg = wmax;
            else if (wavg < wmin)
                wavg = wmin;
        }
    }
    //
    pcl->set_x(xorig + uavg * moverParam->dt);
    pcl->set_y(yorig + vavg * moverParam->dt);
    pcl->set_z(zorig + wavg * moverParam->dt);
    pcl->set_u(2.0 * uavg - uorig);
    pcl->set_v(2.0 * vavg - vorig);
    pcl->set_w(2.0 * wavg - worig);

    // prepare the departure array

    prepareDepartureArray(pcl, moverParam->departureArray, grid, moverParam->hashedSumArray, pidx);
    
}

__host__ __device__ void get_field_components_for_cell(
    const commonType *field_components[8],
    const cudaTypeArray1<commonType> fieldForPcls, grid3DCUDA *grid,
    int cx, int cy, int cz)
{
    // interface to the right of cell
    const int ix = cx + 1;
    const int iy = cy + 1;
    const int iz = cz + 1;
/*
    auto field0 = fieldForPcls[ix];
    auto field1 = fieldForPcls[cx];
    auto field00 = field0[iy];
    auto field01 = field0[cy];
    auto field10 = field1[iy];
    auto field11 = field1[cy];
    field_components[0] = field00[iz]; // field000
    field_components[1] = field00[cz]; // field001
    field_components[2] = field01[iz]; // field010
    field_components[3] = field01[cz]; // field011
    field_components[4] = field10[iz]; // field100
    field_components[5] = field10[cz]; // field101
    field_components[6] = field11[iz]; // field110
    field_components[7] = field11[cz]; // field111
*/
    
    field_components[0] = fieldForPcls + toOneDimIndex(grid->nxn, grid->nyn, grid->nzn, 8, ix, iy, iz, 0); // field000
    field_components[1] = fieldForPcls + toOneDimIndex(grid->nxn, grid->nyn, grid->nzn, 8, ix, iy, cz, 0); // field001
    field_components[2] = fieldForPcls + toOneDimIndex(grid->nxn, grid->nyn, grid->nzn, 8, ix, cy, iz, 0); // field010
    field_components[3] = fieldForPcls + toOneDimIndex(grid->nxn, grid->nyn, grid->nzn, 8, ix, cy, cz, 0); // field011
    field_components[4] = fieldForPcls + toOneDimIndex(grid->nxn, grid->nyn, grid->nzn, 8, cx, iy, iz, 0); // field100
    field_components[5] = fieldForPcls + toOneDimIndex(grid->nxn, grid->nyn, grid->nzn, 8, cx, iy, cz, 0); // field101
    field_components[6] = fieldForPcls + toOneDimIndex(grid->nxn, grid->nyn, grid->nzn, 8, cx, cy, iz, 0); // field110
    field_components[7] = fieldForPcls + toOneDimIndex(grid->nxn, grid->nyn, grid->nzn, 8, cx, cy, cz, 0); // field111
}



__device__ void prepareDepartureArray(SpeciesParticle* pcl, departureArrayType* departureArray, grid3DCUDA* grid, hashedSum* hashedSumArray, uint32_t pidx){

    departureArrayElementType element;

    if(pcl->get_x() < grid->xStart)
    {
        element.dest = 1;
    }
    else if(pcl->get_x() > grid->xEnd)
    {
        element.dest = 2;
    }
    else if(pcl->get_y() < grid->yStart)
    {
        element.dest = 3;
    }
    else if(pcl->get_y() > grid->yEnd)
    {
        element.dest = 4;
    }
    else if(pcl->get_z() < grid->zStart)
    {
        element.dest = 5;
    }
    else if(pcl->get_z() > grid->zEnd)
    {
        element.dest = 6;
    }
    else element.dest = 0;

    if(element.dest != 0){
        element.hashedId = hashedSumArray[element.dest - 1].add(pidx);
    }else{
        element.hashedId = 0;
    }

    departureArray->getArray()[pidx] = element;
}



/*
__device__ void openbc_particles_outflow(moverOutflowParameter* param)
{
    // if this is not a boundary process then there is nothing to do
    if (!param->isBoundaryProcess_P)
        return;

    if (!param->openXleft && !param->openYleft && !param->openZleft && !param->openXright && !param->openYright && !param->openZright)
        return;

    const int num_layers = 3;

    const double xLow = num_layers * dx;
    const double yLow = num_layers * dy;
    const double zLow = num_layers * dz;
    const double xHgh = Lx - xLow;
    const double yHgh = Ly - yLow;
    const double zHgh = Lz - zLow;

    const bool apply_openBC[6] = {param->openXleft, param->openXright, param->openYleft, param->openYright, param->openZleft, param->openZright};
    const double delete_boundary[6] = {0, Lx, 0, Ly, 0, Lz}; // the bound of the whole problem domain box
    const double open_boundary[6] = {xLow, xHgh, yLow, yHgh, zLow, zHgh};

    const int nop_orig = param->nop;
    const int capacity_out = roundup_to_multiple(nop_orig * 0.1, DVECWIDTH);
    vector_SpeciesParticle injpcls(capacity_out);

    for (int dir_cnt = 0; dir_cnt < 6; dir_cnt++)
    {

        if (apply_openBC[dir_cnt])
        {
            // dprintf( "*** OpenBC for Direction %d on particle species %d",dir_cnt, ns);

            int pidx = 0;
            int direction = dir_cnt / 2;
            double delbry = delete_boundary[dir_cnt]; // the real problem domain box
            double openbry = open_boundary[dir_cnt]; // the inner box
            double location;
            while (pidx < param->nop)
            {
                SpeciesParticle &pcl = _pcls[pidx];
                location = pcl.get_x(direction);

                // delete the exiting particle if out of box on the direction of OpenBC
                if ((dir_cnt % 2 == 0 && location < delbry) || (dir_cnt % 2 == 1 && location > delbry))
                    delete_particle(pidx);
                else
                {
                    pidx++;

                    // copy the particle within open boundary to inject particle list if their shifted location after 1 time step is within simulation box
                    if ((dir_cnt % 2 == 0 && location < openbry) || (dir_cnt % 2 == 1 && location > openbry)) // if the pcl is in the interlayer of the direction
                    {
                        double injx = pcl.get_x(0), injy = pcl.get_x(1), injz = pcl.get_x(2);
                        double inju = pcl.get_u(0), injv = pcl.get_u(1), injw = pcl.get_u(2);
                        double injq = pcl.get_q();

                        // shift 3 layers out, not mirror
                        if (direction == 0)
                            injx = (dir_cnt % 2 == 0) ? (injx - xLow) : (injx + xLow);
                        if (direction == 1)
                            injy = (dir_cnt % 2 == 0) ? (injy - yLow) : (injy + yLow);
                        if (direction == 2)
                            injz = (dir_cnt % 2 == 0) ? (injz - zLow) : (injz + zLow);

                        injx = injx + inju * dt;
                        injy = injy + injv * dt;
                        injz = injz + injw * dt;

                        // Add particle if it enter that sub-domain or the domain box?
                        // assume create particle as long as it enters the domain box
                        if (injx > 0 && injx < Lx && injy > 0 && injy < Ly && injz > 0 && injz < Lz)
                        {
                            injpcls.push_back(SpeciesParticle(inju, injv, injw, injq, injx, injy, injz, pclIDgenerator.generateID()));
                        }
                    }
                }
            }
        }
    }

    // const int nop_remaining = getNOP();
    // const int nop_deleted = nop_orig - nop_remaining;
    const int nop_created = injpcls.size();

    // dprintf("change in # particles: %d - %d + %d = %d",nop_orig, nop_deleted, nop_created, nop_remaining);

    for (int outId = 0; outId < nop_created; outId++)
        _pcls.push_back(injpcls[outId]);
}

*/