

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

using commonType = cudaParticleType;

__device__ constexpr bool cap_velocity() { return false; }

// __host__ __device__ void get_field_components_for_cell(
//     const cudaFieldType *field_components[8],
//     cudaTypeArray1<cudaFieldType> fieldForPcls, grid3DCUDA *grid,
//     int cx, int cy, int cz);

__device__ void prepareDepartureArray(SpeciesParticle* pcl, 
                                    departureArrayType* departureArray, 
                                    grid3DCUDA* grid, 
                                    hashedSum* hashedSumArray, 
                                    uint32_t pidx);

__global__ void moverKernel(moverParameter *moverParam,
                            cudaTypeArray1<cudaFieldType> fieldForPcls,
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
    const cudaTypeDouble PC_err_2 = 1E-12;  // square of error tolerance
    cudaTypeDouble currErr = PC_err_2 + 1.; // initialize to a larger value

    // calculate the average velocity iteratively
    while (currErr > PC_err_2 && innter < moverParam->NiterMover)
    {

        // compute weights for field components
        //
        commonType weights[8];
        int cx, cy, cz;
        grid->get_safe_cell_and_weights(xavg, yavg, zavg, cx, cy, cz, weights);

        commonType sampled_field[6];
        for (int i = 0; i < 6; i++)
            sampled_field[i] = 0;
        commonType &Bxl = sampled_field[0];
        commonType &Byl = sampled_field[1];
        commonType &Bzl = sampled_field[2];
        commonType &Exl = sampled_field[3];
        commonType &Eyl = sampled_field[4];
        commonType &Ezl = sampled_field[5];

        // target previous cell
        const int previousIndex = (cx * (grid->nyn - 1) + cy) * grid->nzn + cz; // previous cell index

        assert(previousIndex < 24 * (grid->nzn * (grid->nyn - 1) * (grid->nxn - 1)));

        for (int c = 0; c < 8; c++) // grid node
        {
            // 4 from previous and 4 from itself

            for (int i = 0; i < 6; i++) // field items
            {
                sampled_field[i] += weights[c] * fieldForPcls[previousIndex * 24 + c * 6 + i];
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


    pcl->set_x_u(   xorig + uavg * moverParam->dt,  yorig + vavg * moverParam->dt,  zorig + wavg * moverParam->dt,
                    2.0f * uavg - uorig,            2.0f * vavg - vorig,            2.0f * wavg - worig);



    // prepare the departure array

    // prepareDepartureArray(pcl, moverParam->departureArray, grid, moverParam->hashedSumArray, pidx);
    
}

// __host__ __device__ void get_field_components_for_cell(
//     const cudaFieldType *field_components[8],
//     const cudaTypeArray1<cudaFieldType> fieldForPcls, grid3DCUDA *grid,
//     int cx, int cy, int cz)
// {
//     // interface to the right of cell
//     const int ix = cx + 1;
//     const int iy = cy + 1;
//     const int iz = cz + 1;
    
//     constexpr int ratio = sizeof(cudaCommonType) / sizeof(cudaFieldType);

//     field_components[0] = fieldForPcls + ratio * toOneDimIndex(grid->nxn, grid->nyn, grid->nzn, 8, ix, iy, iz, 0); // field000
//     field_components[1] = fieldForPcls + ratio * toOneDimIndex(grid->nxn, grid->nyn, grid->nzn, 8, ix, iy, cz, 0); // field001
//     field_components[2] = fieldForPcls + ratio * toOneDimIndex(grid->nxn, grid->nyn, grid->nzn, 8, ix, cy, iz, 0); // field010
//     field_components[3] = fieldForPcls + ratio * toOneDimIndex(grid->nxn, grid->nyn, grid->nzn, 8, ix, cy, cz, 0); // field011
//     field_components[4] = fieldForPcls + ratio * toOneDimIndex(grid->nxn, grid->nyn, grid->nzn, 8, cx, iy, iz, 0); // field100
//     field_components[5] = fieldForPcls + ratio * toOneDimIndex(grid->nxn, grid->nyn, grid->nzn, 8, cx, iy, cz, 0); // field101
//     field_components[6] = fieldForPcls + ratio * toOneDimIndex(grid->nxn, grid->nyn, grid->nzn, 8, cx, cy, iz, 0); // field110
//     field_components[7] = fieldForPcls + ratio * toOneDimIndex(grid->nxn, grid->nyn, grid->nzn, 8, cx, cy, cz, 0); // field111
// }

// __global__ void castingField(grid3DCUDA *grid, cudaTypeArray1<cudaCommonType> fieldForPcls){
//     uint pidx = blockIdx.x * blockDim.x + threadIdx.x;
//     if(pidx >= (grid->nxn * grid->nyn * grid->nzn))return;

//     //cudaFieldType temp[8];

//     auto pointDoublePtr = fieldForPcls + pidx * 8;
//     cudaFieldType* pointSinglePtr = (cudaFieldType*)pointDoublePtr;

//     for(int i=0; i < 8; i++){
//         pointSinglePtr[i] = (cudaFieldType)pointDoublePtr[i];
//     }

//     // for(int i=0; i < 8; i++){
//     //     pointSinglePtr[i] = temp[i];
//     // }
    
// }



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


