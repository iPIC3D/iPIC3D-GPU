

#include <iostream>
#include <math.h>
#include <limits.h>
#include <cstdint>
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

// ====== Inline helper functions for mover kernels ======

// Mark particle for deletion and register in the departure hashed sum.
__device__ __forceinline__ void markForDeletion(
    moverParameter *moverParam, uint32_t pidx)
{
    moverParam->departureArray->getArray()[pidx].dest = departureArrayElementType::DELETE;
    moverParam->departureArray->getArray()[pidx].hashedId =
        moverParam->hashedSumArray[departureArrayElementType::DELETE_HASHEDSUM_INDEX].add(pidx);
}

// Cap particle velocity to v2max = 0.9999^2.
// Rescales finite superluminal velocities; marks NaN/Inf for deletion.
// Returns true if the particle was deleted (caller should return).
__device__ __forceinline__ bool capVelocityOrDelete(
    SpeciesParticle *pcl, moverParameter *moverParam, uint32_t pidx)
{
    const commonType unew = pcl->get_u();
    const commonType vnew = pcl->get_v();
    const commonType wnew = pcl->get_w();
    const commonType v2 = unew * unew + vnew * vnew + wnew * wnew;
    constexpr commonType v2max = 0.9999 * 0.9999; // max allowed |v/c|^2
    if (!(v2 < v2max)) {
        if (isfinite(v2)) {
            const commonType scale = sqrt(v2max / v2);
            pcl->set_u(unew * scale);
            pcl->set_v(vnew * scale);
            pcl->set_w(wnew * scale);
        } else {
            markForDeletion(moverParam, pidx);
            return true;
        }
    }
    return false;
}

// Sample nFields components of the EM field at position (x,y,z).
// Fills weights[8], updates cell coordinates (cx,cy,cz), and writes
// sampled_field[0..nFields-1] with interpolated values.
template <int nFields>
__device__ __forceinline__ void sampleFieldsAtPosition(
    commonType x, commonType y, commonType z,
    grid3DCUDA *grid,
    cudaTypeArray1<cudaFieldType> fieldForPcls,
    commonType weights[8], int &cx, int &cy, int &cz,
    commonType sampled_field[6])
{
    grid->get_safe_cell_and_weights(x, y, z, cx, cy, cz, weights);
    const int previousIndex = (cx * (grid->nyn - 1) + cy) * grid->nzn + cz;
    assert(previousIndex < 24 * (grid->nzn * (grid->nyn - 1) * (grid->nxn - 1)));
    for (int i = 0; i < nFields; i++)
        sampled_field[i] = 0;
    for (int c = 0; c < 8; c++)
        for (int i = 0; i < nFields; i++)
            sampled_field[i] += weights[c] * fieldForPcls[previousIndex * 24 + c * 6 + i];
}

// Compute predictor-corrector convergence error (relative squared velocity change).
__device__ __forceinline__ commonType computePCError(
    commonType uavg, commonType vavg, commonType wavg,
    commonType uavg_old, commonType vavg_old, commonType wavg_old)
{
    return ((uavg_old - uavg) * (uavg_old - uavg) +
            (vavg_old - vavg) * (vavg_old - vavg) +
            (wavg_old - wavg) * (wavg_old - wavg)) /
           (uavg_old * uavg_old + vavg_old * vavg_old + wavg_old * wavg_old);
}

// ====== End inline helper functions ======

// __host__ __device__ void get_field_components_for_cell(
//     const cudaFieldType *field_components[8],
//     cudaTypeArray1<cudaFieldType> fieldForPcls, grid3DCUDA *grid,
//     int cx, int cy, int cz);

__device__ void prepareDepartureArray(SpeciesParticle* pcl, 
                                    moverParameter *moverParam,
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

    if(moverParam->departureArray->getArray()[pidx].dest != 0){ // deleted during the merging
        prepareDepartureArray(pcl, moverParam, moverParam->departureArray, grid, moverParam->hashedSumArray, pidx);
        return;
    }


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

        // sample E and B field components at current average position
        commonType weights[8];
        int cx, cy, cz;
        commonType sampled_field[6];
        sampleFieldsAtPosition<6>(xavg, yavg, zavg, grid, fieldForPcls,
                                  weights, cx, cy, cz, sampled_field);

        commonType &Bxl = sampled_field[0];
        commonType &Byl = sampled_field[1];
        commonType &Bzl = sampled_field[2];
        commonType &Exl = sampled_field[3];
        commonType &Eyl = sampled_field[4];
        commonType &Ezl = sampled_field[5];
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
        currErr = computePCError(uavg, vavg, wavg, uavg_old, vavg_old, wavg_old);
        // capture the new velocity for the next iteration
        uavg_old = uavg;
        vavg_old = vavg;
        wavg_old = wavg;

    } // end of iteration

    // update the final position and velocity
    // if (cap_velocity()) //used to limit the speed of particles under c
    // {
    //     auto umax = moverParam->umax;
    //     auto vmax = moverParam->vmax;
    //     auto wmax = moverParam->wmax;
    //     auto umin = moverParam->umin;
    //     auto vmin = moverParam->vmin;
    //     auto wmin = moverParam->wmin;
    //
    //     bool cap = (abs(uavg) > umax || abs(vavg) > vmax || abs(wavg) > wmax) ? true : false;
    //     if (cap)
    //     {
    //         if (uavg > umax)      uavg = umax;
    //         else if (uavg < umin) uavg = umin;
    //         if (vavg > vmax)      vavg = vmax;
    //         else if (vavg < vmin) vavg = vmin;
    //         if (wavg > wmax)      wavg = wmax;
    //         else if (wavg < wmin) wavg = wmin;
    //     }
    // }
    pcl->set_x_u(   xorig + uavg * moverParam->dt,  yorig + vavg * moverParam->dt,  zorig + wavg * moverParam->dt,
                    2.0f * uavg - uorig,            2.0f * vavg - vorig,            2.0f * wavg - worig);

    // Cap velocity to prevent superluminal particles.
    // The non-relativistic Boris pusher has no intrinsic speed-of-light limit,
    // so extreme E fields can produce |v| >= c.
    if (capVelocityOrDelete(pcl, moverParam, pidx))
        return;

    // prepare the departure array

    prepareDepartureArray(pcl, moverParam, moverParam->departureArray, grid, moverParam->hashedSumArray, pidx);
    
}





// select kernel in /src/ipic3d/main/iPIC3Dlib.cu
__global__ void moverSubcyclesKernel(moverParameter *moverParam,
        cudaTypeArray1<cudaFieldType> fieldForPcls,
        grid3DCUDA *grid)
{
    uint pidx = blockIdx.x * blockDim.x + threadIdx.x;

    auto pclsArray = moverParam->pclsArray;
    if(pidx >= pclsArray->getNOP())return;

    // copy the particle
    SpeciesParticle *pcl = pclsArray->getpcls() + pidx;

    if(moverParam->departureArray->getArray()[pidx].dest != 0){ // deleted during the merging
        prepareDepartureArray(pcl, moverParam, moverParam->departureArray, grid, moverParam->hashedSumArray, pidx);
        return;
    }


    // first step: evaluate local B magnitude

    commonType weights[8];
    int cx, cy, cz;
    commonType sampled_field[6];
    sampleFieldsAtPosition<3>(pcl->get_x(), pcl->get_y(), pcl->get_z(),
                              grid, fieldForPcls, weights, cx, cy, cz, sampled_field);

    // evaluate local B field magnitude
    const commonType B_mag = sqrt(sampled_field[0] * sampled_field[0] + sampled_field[1] * sampled_field[1] + sampled_field[2] * sampled_field[2]);

    // evaluate dt_substep and number of sub cycles

    commonType dt_sub = M_PI * moverParam->c / (4 * fabs(moverParam->qom) * B_mag);
    const int sub_cycles = (int)(moverParam->dt / dt_sub) + 1;
    dt_sub = moverParam->dt / (commonType)(sub_cycles);
    
    const commonType dto2_sub = .5 * dt_sub;
    const commonType qdto2mc_sub = moverParam->qom * dto2_sub / moverParam->c;

    // Safety: check initial velocity before entering subcycle loop.
    // If v² >= 1 or NaN (e.g. from MPI injection or previous-cycle noise),
    // gamma0 = 1/sqrt(1-v²) would produce NaN, cascading through the mover.
    // After subcycle 0, the velocity cap (S2) at the end of each subcycle
    // guarantees v² < v2max < 1, so this check is only needed once.
    {
        const commonType v2_init = pcl->get_u()*pcl->get_u()
                                 + pcl->get_v()*pcl->get_v()
                                 + pcl->get_w()*pcl->get_w();
        if (!(v2_init < 1.0)) {
            markForDeletion(moverParam, pidx);
            return;
        }
    }

    //start subcycling
    for(int cyc_cnt = 0; cyc_cnt < sub_cycles; cyc_cnt++)
    {
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

        const commonType vorig_sq = uorig*uorig + vorig*vorig + worig*worig;
        const commonType gamma0 = 1.0 / (sqrt(1.0 - vorig_sq));
        commonType gamma1;

        int innter = 0;
        const cudaTypeDouble PC_err_2 = 1E-12;  // square of error tolerance
        cudaTypeDouble currErr = PC_err_2 + 1.; // initialize to a larger value

        // calculate the average velocity iteratively - predictor corrector
        // uses dt_subcycle
        while (currErr > PC_err_2 && innter < moverParam->NiterMover)
        {

            // sample E and B field components at current average position
            sampleFieldsAtPosition<6>(xavg, yavg, zavg, grid, fieldForPcls,
                                      weights, cx, cy, cz, sampled_field);

            commonType &Bxl = sampled_field[0];
            commonType &Byl = sampled_field[1];
            commonType &Bzl = sampled_field[2];
            commonType &Exl = sampled_field[3];
            commonType &Eyl = sampled_field[4];
            commonType &Ezl = sampled_field[5];
            const commonType Omx = qdto2mc_sub * Bxl;
            const commonType Omy = qdto2mc_sub * Byl;
            const commonType Omz = qdto2mc_sub * Bzl;

            // end interpolation
            const commonType omsq = (Omx * Omx + Omy * Omy + Omz * Omz);
            commonType denom = 1.0 / (1.0 + omsq);
            // solve the position equation
            const commonType ut = uorig * gamma0 + qdto2mc_sub * Exl;
            const commonType vt = vorig * gamma0 + qdto2mc_sub * Eyl;
            const commonType wt = worig * gamma0 + qdto2mc_sub * Ezl;

            gamma1 = sqrt(1.0 + ut*ut + vt*vt + wt*wt);
			Bxl /= gamma1;
            Byl /= gamma1;
            Bzl /= gamma1;
            denom /= gamma1;

            // const commonType udotb = ut * Bxl + vt * Byl + wt * Bzl;
            const commonType udotOm = ut * Omx + vt * Omy + wt * Omz;
            // solve the velocity equation
            uavg = (ut + (vt * Omz - wt * Omy + udotOm * Omx)) * denom;
            vavg = (vt + (wt * Omx - ut * Omz + udotOm * Omy)) * denom;
            wavg = (wt + (ut * Omy - vt * Omx + udotOm * Omz)) * denom;
            // update average position
            xavg = xorig + uavg * dto2_sub;
            yavg = yorig + vavg * dto2_sub;
            zavg = zorig + wavg * dto2_sub;

            currErr = computePCError(uavg, vavg, wavg, uavg_old, vavg_old, wavg_old);
            // capture the new velocity for the next iteration
            uavg_old = uavg;
            vavg_old = vavg;
            wavg_old = wavg;

            innter++;

        } // end of iteration

        // relativistic velocity update
        const commonType ut = uorig * gamma0;
        const commonType vt = vorig * gamma0;
        const commonType wt = worig * gamma0;

        const commonType velt_sq = ut*ut + vt*vt + wt*wt;
        const commonType velavg_sq = uavg*uavg + vavg*vavg + wavg*wavg;
        const commonType velt_velavg = ut*uavg + vt*vavg + wt*wavg;

        const commonType cfa = 1.0 - velavg_sq;
        const commonType cfb = -2.0 * (-velt_velavg + gamma0 * velavg_sq);
        const commonType cfc = -1.0 - gamma0 * gamma0 * velavg_sq + 2.0 * gamma0 * velt_velavg - velt_sq;
        
        const commonType delta_rel = cfb * cfb - 4.0 * cfa * cfc;

         // update velocity
        if (delta_rel < 0.0){

            //cout << "Relativity violated: gamma0=" << gamma0 << ",  vavg_sq=" << vavg_sq;
            pcl->set_x_u(   xorig + uavg * dt_sub,  yorig + vavg * dt_sub,  zorig + wavg * dt_sub,
                (2.0*gamma1)*uavg - uorig*gamma0, (2.0*gamma1)*vavg - vorig*gamma0, (2.0*gamma1)*wavg - worig*gamma0);
        }
        else{
            const commonType gamma1 = ( -cfb + sqrt(delta_rel)) / 2.0 / cfa;
            pcl->set_x_u(   xorig + uavg * dt_sub,  yorig + vavg * dt_sub,  zorig + wavg * dt_sub,
                (1.0 + gamma0/gamma1)*uavg - ut/gamma1, (1.0 + gamma0/gamma1)*vavg - vt/gamma1, (1.0 + gamma0/gamma1)*wavg - wt/gamma1);
        }

        // Cap velocity to prevent superluminal particles.
        // If |v|^2 >= 1, the next subcycle's gamma = 1/sqrt(1-v^2) -> NaN.
        if (capVelocityOrDelete(pcl, moverParam, pidx))
            return;

    } // end iteration over subcycles
    
    // prepare the departure array

    prepareDepartureArray(pcl, moverParam, moverParam->departureArray, grid, moverParam->hashedSumArray, pidx);

}

__device__ uint32_t deleteAppendOpenBCOutflow(SpeciesParticle* pcl, moverParameter *moverParam, departureArrayType* departureArray, grid3DCUDA* grid, hashedSum* hashedSumArray) {

    if (!moverParam->doOpenBC) return 0;

    auto& delBdry = moverParam->deleteBoundary;
    auto& openBdry = moverParam->openBoundary;
    
    for (int side = 0; side < 6; side++) {

        if (!moverParam->applyOpenBC[side]) continue;

        const auto direction = side / 2; // x,y,z
        const auto location = pcl->get_x(direction);
        const bool leftRight = side % 2; // 0: left, 1: right

        // delete boundary
        if( (leftRight == 0 && location < delBdry[side]) || (leftRight == 1 && location > delBdry[side]) ) {
            // delete the particle
            return departureArrayElementType::DELETE;
        }

        // open boundary
        if ( (leftRight == 0 && location < openBdry[side]) || (leftRight == 1 && location > openBdry[side]) ) {
            SpeciesParticle newPcl = *pcl;
            
            // update the position
            newPcl.set_x(direction, newPcl.get_x(direction) + (leftRight==0?-1:1) * openBdry[direction*2]);
            newPcl.fetch_x() += newPcl.get_u() * moverParam->dt;
            newPcl.fetch_y() += newPcl.get_v() * moverParam->dt;
            newPcl.fetch_z() += newPcl.get_w() * moverParam->dt;

            newPcl.set_t(114514.0);

            // if the new particle is still in the domain
            
            if (
                newPcl.get_x() > delBdry[0] && newPcl.get_x() < delBdry[1] &&
                newPcl.get_y() > delBdry[2] && newPcl.get_y() < delBdry[3] &&
                newPcl.get_z() > delBdry[4] && newPcl.get_z() < delBdry[5]
                //newPcl.get_x() > grid->xStart && newPcl.get_x() < grid->xEnd &&
                //newPcl.get_y() > grid->yStart && newPcl.get_y() < grid->yEnd &&
                //newPcl.get_z() > grid->zStart && newPcl.get_z() < grid->zEnd
            ) {
                departureArrayElementType element;
                const auto index = moverParam->pclsArray->getNOP() + atomicAdd(&moverParam->appendCountAtomic, 1);
                // check memory overflow
                if (index >= moverParam->pclsArray->getSize()) {
                    printf("Memory overflow in open boundary outflow (index=%u, size=%u)\n",
                           index, moverParam->pclsArray->getSize());
                    // Cannot append — drop this duplicate and mark the original
                    // particle for deletion so it doesn't corrupt hashed sums.
                    return departureArrayElementType::DELETE;
                }
                memcpy(moverParam->pclsArray->getpcls() + index, &newPcl, sizeof(SpeciesParticle));
                if(newPcl.get_x() < grid->xStart)
                {
                    element.dest = departureArrayElementType::XLOW;
                }
                else if(newPcl.get_x() > grid->xEnd)
                {
                    element.dest = departureArrayElementType::XHIGH;
                }
                else if(newPcl.get_y() < grid->yStart)
                {
                    element.dest = departureArrayElementType::YLOW;
                }
                else if(newPcl.get_y() > grid->yEnd)
                {
                    element.dest = departureArrayElementType::YHIGH;
                }
                else if(newPcl.get_z() < grid->zStart)
                {
                    element.dest = departureArrayElementType::ZLOW;
                }
                else if(newPcl.get_z() > grid->zEnd)
                {
                    element.dest = departureArrayElementType::ZHIGH;
                }
                else element.dest = departureArrayElementType::STAY;

                if(element.dest != 0){
                    element.hashedId = hashedSumArray[element.dest - 1].add(index);
                }else{
                    element.hashedId = 0;
                }
            
                departureArray->getArray()[index] = element;
            }
        }

    }

    return 0;

}

__device__ uint32_t deleteRepopulateInjection(SpeciesParticle* pcl, moverParameter *moverParam, grid3DCUDA *grid) {
    if (!moverParam->doRepopulateInjection) return 0;

    auto& doRepopulateInjectionSide = moverParam->doRepopulateInjectionSide;
    auto& repopulateBoundary = moverParam->repopulateBoundary;

    if (
        (doRepopulateInjectionSide[0] && pcl->get_x() < repopulateBoundary[0]) ||
        (doRepopulateInjectionSide[1] && pcl->get_x() > repopulateBoundary[1]) ||
        (doRepopulateInjectionSide[2] && pcl->get_y() < repopulateBoundary[2]) ||
        (doRepopulateInjectionSide[3] && pcl->get_y() > repopulateBoundary[3]) ||
        (doRepopulateInjectionSide[4] && pcl->get_z() < repopulateBoundary[4]) ||
        (doRepopulateInjectionSide[5] && pcl->get_z() > repopulateBoundary[5])
    ) { // In the repoopulate layers
        return departureArrayElementType::DELETE;
    } else {
        return 0;
    }

}

__device__ uint32_t deleteInsideSphere(SpeciesParticle* pcl, moverParameter *moverParam, grid3DCUDA *grid) {
    
    if (moverParam->doSphere == 0) return 0;

    if(moverParam->doSphere == 1){ // 3D sphere
        const auto& sphereOrigin = moverParam->sphereOrigin;
        const auto& sphereRadius = moverParam->sphereRadius;

        const auto dx = pcl->get_x() - sphereOrigin[0];
        const auto dy = pcl->get_y() - sphereOrigin[1];
        const auto dz = pcl->get_z() - sphereOrigin[2];

        if (dx*dx + dy*dy + dz*dz < sphereRadius*sphereRadius) {
            return departureArrayElementType::PLANET;
        }
    } else if(moverParam->doSphere == 2){ // 2D sphere
        const auto& sphereOrigin = moverParam->sphereOrigin;
        const auto& sphereRadius = moverParam->sphereRadius;

        const auto dx = pcl->get_x() - sphereOrigin[0];
        const auto dz = pcl->get_z() - sphereOrigin[2];

        if (dx*dx + dz*dz < sphereRadius*sphereRadius) {
            return departureArrayElementType::PLANET;
        }
    }

    return 0;

}


__device__ void prepareDepartureArray(SpeciesParticle* pcl, moverParameter *moverParam, departureArrayType* departureArray, grid3DCUDA* grid, hashedSum* hashedSumArray, uint32_t pidx){

    if(departureArray->getArray()[pidx].dest != 0) {
        departureArray->getArray()[pidx].hashedId = 
            hashedSumArray[departureArray->getArray()[pidx].dest - 1].add(pidx);
        
            return;
    }

    // Safety: NaN positions bypass all comparison-based boundary checks
    // (IEEE 754: NaN < x and NaN > x are both false), so a NaN particle
    // would fall through to STAY and deposit NaN into the moments array.
    if (!isfinite(pcl->get_x()) || !isfinite(pcl->get_y()) || !isfinite(pcl->get_z())) {
        departureArrayElementType element;
        element.dest = departureArrayElementType::DELETE;
        element.hashedId = hashedSumArray[departureArrayElementType::DELETE_HASHEDSUM_INDEX].add(pidx);
        departureArray->getArray()[pidx] = element;
        return;
    }
    
    departureArrayElementType element;

    do {

        // OpenBC_outflow
        element.dest = deleteAppendOpenBCOutflow(pcl, moverParam, departureArray, grid, hashedSumArray);
        if(element.dest != 0)break;

        // INJECT
        element.dest = deleteRepopulateInjection(pcl, moverParam, grid);
        if(element.dest != 0)break;

        // sphere
        element.dest = deleteInsideSphere(pcl, moverParam, grid);
        if(element.dest != 0)break;

        // Exiting

        if(pcl->get_x() < grid->xStart)
        {
            element.dest = moverParam->isExitBC[0]
                ? departureArrayElementType::DELETE
                : departureArrayElementType::XLOW;
        }
        else if(pcl->get_x() > grid->xEnd)
        {
            element.dest = moverParam->isExitBC[1]
                ? departureArrayElementType::DELETE
                : departureArrayElementType::XHIGH;
        }
        else if(pcl->get_y() < grid->yStart)
        {
            element.dest = moverParam->isExitBC[2]
                ? departureArrayElementType::DELETE
                : departureArrayElementType::YLOW;
        }
        else if(pcl->get_y() > grid->yEnd)
        {
            element.dest = moverParam->isExitBC[3]
                ? departureArrayElementType::DELETE
                : departureArrayElementType::YHIGH;
        }
        else if(pcl->get_z() < grid->zStart)
        {
            element.dest = moverParam->isExitBC[4]
                ? departureArrayElementType::DELETE
                : departureArrayElementType::ZLOW;
        }
        else if(pcl->get_z() > grid->zEnd)
        {
            element.dest = moverParam->isExitBC[5]
                ? departureArrayElementType::DELETE
                : departureArrayElementType::ZHIGH;
        }
        else element.dest = departureArrayElementType::STAY;

    }while (0);

    if(element.dest != 0){
        element.hashedId = hashedSumArray[element.dest - 1].add(pidx);
    }else{
        element.hashedId = 0;
    }

    departureArray->getArray()[pidx] = element;
}

