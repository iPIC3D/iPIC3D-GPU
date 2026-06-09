#include <assert.h>
#include <cstdint>
#include <math.h>

#include "cudaTypeDef.cuh"
#include "gridCUDA.cuh"
#include "hashedSum.cuh"
#include "moverKernel.cuh"
#include "particleArrayCUDA.cuh"
#include "particleExchange.cuh"

using commonType = cudaParticleType;

constexpr cudaTypeDouble PC_err_2 = 1E-12;  // square of error tolerance

__device__ constexpr bool cap_velocity() { return false; }

// ======= Inline mover helpers =======

/**
 * @brief Mark a particle for deletion and register it in the delete hashed sum.
 *
 * The particle state itself is left untouched; only the departure metadata is
 * updated so later compaction and communication stages can remove it safely.
 *
 * @param moverParam Device-side mover parameter bundle.
 * @param pidx Particle index in the species SoA buffer.
 */
__device__ __forceinline__ void markForDeletion(
    moverParameter *moverParam, uint32_t pidx)
{
    moverParam->departureArray->getArray()[pidx].dest = departureArrayElementType::DELETE;
    moverParam->departureArray->getArray()[pidx].hashedId =
        moverParam->hashedSumArray[departureArrayElementType::DELETE_HASHEDSUM_INDEX].add(pidx);
}

/**
 * @brief Clamp finite superluminal velocities and delete non-finite particles.
 *
 * Operates on register-resident velocity components. Finite velocities with
 * |v/c| above the safety cap are rescaled back below it; NaN and Inf values
 * are marked for deletion because they would poison later gamma and moment
 * calculations.
 *
 * @param ufinal In-out x-velocity component.
 * @param vfinal In-out y-velocity component.
 * @param wfinal In-out z-velocity component.
 * @param moverParam Device-side mover parameter bundle.
 * @param pidx Particle index in the species SoA buffer.
 * @return True if the particle was deleted and the caller should return.
 */
__device__ __forceinline__ bool capVelocityOrDelete(
    commonType& ufinal, commonType& vfinal, commonType& wfinal,
    moverParameter *moverParam, uint32_t pidx)
{
    const commonType v2 = ufinal * ufinal + vfinal * vfinal + wfinal * wfinal;
    constexpr commonType v2max = 0.9999 * 0.9999; // max allowed |v/c|^2
    if (!(v2 < v2max)) {
        if (isfinite(v2)) {
            const commonType scale = sqrt(v2max / v2);
            ufinal *= scale;
            vfinal *= scale;
            wfinal *= scale;
        } else {
            markForDeletion(moverParam, pidx);
            return true;
        }
    }
    return false;
}

/**
 * @brief Interpolate particle fields from the packed node-centered field buffer.
 *
 * Updates the owning cell indices and trilinear weights, then accumulates the
 * first @p nFields components into @p sampled_field.
 *
 * @param x Particle x position.
 * @param y Particle y position.
 * @param z Particle z position.
 * @param grid Device-side grid descriptor.
 * @param fieldForPcls Packed field buffer consumed by the mover.
 * @param weights Output trilinear weights for the enclosing cell.
 * @param cx Output cell index in x.
 * @param cy Output cell index in y.
 * @param cz Output cell index in z.
 * @param sampled_field Output sampled field values.
 */
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
    assert(previousIndex >= 0 && previousIndex < (grid->nxn - 1) * (grid->nyn - 1) * grid->nzn);
    for (int i = 0; i < nFields; i++)
        sampled_field[i] = 0;
    for (int c = 0; c < 8; c++)
        for (int i = 0; i < nFields; i++)
            sampled_field[i] += weights[c] * fieldForPcls[previousIndex * 24 + c * 6 + i];
}

/**
 * @brief Compute the predictor-corrector convergence metric.
 *
 * The return value is the relative squared change between two successive
 * average-velocity estimates.
 *
 * @param uavg Current averaged x velocity.
 * @param vavg Current averaged y velocity.
 * @param wavg Current averaged z velocity.
 * @param uavg_old Previous averaged x velocity.
 * @param vavg_old Previous averaged y velocity.
 * @param wavg_old Previous averaged z velocity.
 * @return Relative squared change between successive averaged velocities.
 */
__device__ __forceinline__ commonType computePCError(
    commonType uavg, commonType vavg, commonType wavg,
    commonType uavg_old, commonType vavg_old, commonType wavg_old)
{
    return ((uavg_old - uavg) * (uavg_old - uavg) +
            (vavg_old - vavg) * (vavg_old - vavg) +
            (wavg_old - wavg) * (wavg_old - wavg)) /
           (uavg_old * uavg_old + vavg_old * vavg_old + wavg_old * wavg_old);
}

// ======= End inline mover helpers =======

/**
 * @brief Classify a moved particle and update its departure metadata.
 *
 * @param xpcl Particle x position after the mover.
 * @param ypcl Particle y position after the mover.
 * @param zpcl Particle z position after the mover.
 * @param pclsArray Device-side particle SoA container.
 * @param pidx Particle index in the species SoA buffer.
 * @param moverParam Device-side mover parameter bundle.
 * @param departureArray Device-side departure metadata array.
 * @param grid Device-side grid descriptor.
 * @param hashedSumArray Device-side hashed-sum buckets for departure compaction.
 */
__device__ void prepareDepartureArray(commonType xpcl, commonType ypcl, commonType zpcl,
                                    particleArrayCUDA* pclsArray, uint32_t pidx,
                                    moverParameter *moverParam,
                                    departureArrayType* departureArray, 
                                    grid3DCUDA* grid, 
                                    hashedSum* hashedSumArray);

/**
 * @brief Advance one particle over the full solver time step with the standard mover.
 *
 * The kernel performs the predictor-corrector push, clamps or deletes invalid
 * velocities, writes the final SoA state, and classifies the particle for
 * staying, exchange, deletion, or planet handling.
 *
 * @param moverParam Device-side mover parameter bundle for one species.
 * @param fieldForPcls Packed mover field buffer.
 * @param grid Device-side grid descriptor.
 */
__global__ void moverKernel(moverParameter *moverParam,
                            cudaTypeArray1<cudaFieldType> fieldForPcls,
                            grid3DCUDA *grid)
{
    uint pidx = blockIdx.x * blockDim.x + threadIdx.x;

    auto pclsArray = moverParam->pclsArray;
    if(pidx >= pclsArray->getNOP())return;
    
    const commonType dto2 = .5 * moverParam->dt,
                     qdto2mc = moverParam->qom * dto2 / moverParam->c;

    // Early exit for particles already deleted (e.g. during merging)
    if(moverParam->departureArray->getArray()[pidx].dest != 0){
        prepareDepartureArray(pclsArray->getX()[pidx], pclsArray->getY()[pidx], pclsArray->getZ()[pidx],
                              pclsArray, pidx, moverParam, moverParam->departureArray, grid, moverParam->hashedSumArray);
        return;
    }

    // Load particle state from SoA into registers
    const commonType xorig = pclsArray->getX()[pidx];
    const commonType yorig = pclsArray->getY()[pidx];
    const commonType zorig = pclsArray->getZ()[pidx];
    const commonType uorig = pclsArray->getU()[pidx];
    const commonType vorig = pclsArray->getV()[pidx];
    const commonType worig = pclsArray->getW()[pidx];
    commonType xavg = xorig;
    commonType yavg = yorig;
    commonType zavg = zorig;
    commonType uavg, vavg, wavg;
    commonType uavg_old = uorig;
    commonType vavg_old = vorig;
    commonType wavg_old = worig;

    int innter = 0;
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

    // Compute final position and velocity in registers
    commonType xfinal = xorig + uavg * moverParam->dt;
    commonType yfinal = yorig + vavg * moverParam->dt;
    commonType zfinal = zorig + wavg * moverParam->dt;
    commonType ufinal = 2.0 * uavg - uorig;
    commonType vfinal = 2.0 * vavg - vorig;
    commonType wfinal = 2.0 * wavg - worig;

    // Cap velocity to prevent superluminal particles.
    // The non-relativistic Boris pusher has no intrinsic speed-of-light limit,
    // so extreme E fields can produce |v| >= c.
    if (capVelocityOrDelete(ufinal, vfinal, wfinal, moverParam, pidx))
        return;

    // Write final state to SoA
    pclsArray->getX()[pidx] = xfinal;
    pclsArray->getY()[pidx] = yfinal;
    pclsArray->getZ()[pidx] = zfinal;
    pclsArray->getU()[pidx] = ufinal;
    pclsArray->getV()[pidx] = vfinal;
    pclsArray->getW()[pidx] = wfinal;

    // prepare the departure array
    prepareDepartureArray(xfinal, yfinal, zfinal, pclsArray, pidx,
                          moverParam, moverParam->departureArray, grid, moverParam->hashedSumArray);
    
}

/**
 * @brief Advance one particle with adaptive subcycling based on local magnetic field strength.
 *
 * The full step is split into smaller substeps when the local gyrofrequency is
 * large. State is kept in registers across the subcycle loop and written back
 * only once at the end.
 *
 * @param moverParam Device-side mover parameter bundle for one species.
 * @param fieldForPcls Packed mover field buffer.
 * @param grid Device-side grid descriptor.
 */
__global__ void moverSubcyclesKernel(moverParameter *moverParam,
        cudaTypeArray1<cudaFieldType> fieldForPcls,
        grid3DCUDA *grid)
{
    uint pidx = blockIdx.x * blockDim.x + threadIdx.x;

    auto pclsArray = moverParam->pclsArray;
    if(pidx >= pclsArray->getNOP())return;

    // Early exit for particles already deleted (e.g. during merging)
    if(moverParam->departureArray->getArray()[pidx].dest != 0){
        prepareDepartureArray(pclsArray->getX()[pidx], pclsArray->getY()[pidx], pclsArray->getZ()[pidx],
                              pclsArray, pidx, moverParam, moverParam->departureArray, grid, moverParam->hashedSumArray);
        return;
    }

    // Load particle state from SoA and keep it in registers across all subcycles.
    commonType cur_x = pclsArray->getX()[pidx];
    commonType cur_y = pclsArray->getY()[pidx];
    commonType cur_z = pclsArray->getZ()[pidx];
    commonType cur_u = pclsArray->getU()[pidx];
    commonType cur_v = pclsArray->getV()[pidx];
    commonType cur_w = pclsArray->getW()[pidx];

    // first step: evaluate local B magnitude
    commonType weights[8];
    int cx, cy, cz;
    commonType sampled_field[6];
    sampleFieldsAtPosition<3>(cur_x, cur_y, cur_z,
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
        const commonType v2_init = cur_u*cur_u + cur_v*cur_v + cur_w*cur_w;
        if (!(v2_init < 1.0)) {
            markForDeletion(moverParam, pidx);
            return;
        }
    }

    // Start subcycling; keep the state in registers across the loop.
    for(int cyc_cnt = 0; cyc_cnt < sub_cycles; cyc_cnt++)
    {
        const commonType xorig = cur_x;
        const commonType yorig = cur_y;
        const commonType zorig = cur_z;
        const commonType uorig = cur_u;
        const commonType vorig = cur_v;
        const commonType worig = cur_w;
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

         // update velocity in registers
        if (delta_rel < 0.0){
            cur_x = xorig + uavg * dt_sub;
            cur_y = yorig + vavg * dt_sub;
            cur_z = zorig + wavg * dt_sub;
            cur_u = (2.0*gamma1)*uavg - uorig*gamma0;
            cur_v = (2.0*gamma1)*vavg - vorig*gamma0;
            cur_w = (2.0*gamma1)*wavg - worig*gamma0;
        }
        else{
            const commonType gamma1_rel = ( -cfb + sqrt(delta_rel)) / 2.0 / cfa;
            cur_x = xorig + uavg * dt_sub;
            cur_y = yorig + vavg * dt_sub;
            cur_z = zorig + wavg * dt_sub;
            cur_u = (1.0 + gamma0/gamma1_rel)*uavg - ut/gamma1_rel;
            cur_v = (1.0 + gamma0/gamma1_rel)*vavg - vt/gamma1_rel;
            cur_w = (1.0 + gamma0/gamma1_rel)*wavg - wt/gamma1_rel;
        }

        // Cap velocity to prevent superluminal particles.
        // If |v|^2 >= 1, the next subcycle's gamma = 1/sqrt(1-v^2) -> NaN.
        if (capVelocityOrDelete(cur_u, cur_v, cur_w, moverParam, pidx))
            return;

    } // end iteration over subcycles
    
    // Write final state to SoA once
    pclsArray->getX()[pidx] = cur_x;
    pclsArray->getY()[pidx] = cur_y;
    pclsArray->getZ()[pidx] = cur_z;
    pclsArray->getU()[pidx] = cur_u;
    pclsArray->getV()[pidx] = cur_v;
    pclsArray->getW()[pidx] = cur_w;

    // prepare the departure array
    prepareDepartureArray(cur_x, cur_y, cur_z, pclsArray, pidx,
                          moverParam, moverParam->departureArray, grid, moverParam->hashedSumArray);

}

// ======= Boundary classification helpers =======

/**
 * @brief Apply open-boundary outflow logic and optionally append a translated duplicate.
 *
 * When a particle crosses an open-boundary layer, the original particle can be
 * deleted while a translated duplicate is appended and immediately classified
 * for exchange on the opposite side.
 *
 * @param xpcl Particle x position after the mover.
 * @param ypcl Particle y position after the mover.
 * @param zpcl Particle z position after the mover.
 * @param pclsArray Device-side particle SoA container.
 * @param pidx Particle index in the species SoA buffer.
 * @param moverParam Device-side mover parameter bundle.
 * @param departureArray Device-side departure metadata array.
 * @param grid Device-side grid descriptor.
 * @param hashedSumArray Device-side hashed-sum buckets for departure compaction.
 * @return Departure destination for the original particle, or 0 when this
 *         helper does not claim it.
 */
__device__ uint32_t deleteAppendOpenBCOutflow(commonType xpcl, commonType ypcl, commonType zpcl,
    particleArrayCUDA* pclsArray, uint32_t pidx,
    moverParameter *moverParam, departureArrayType* departureArray, grid3DCUDA* grid, hashedSum* hashedSumArray) {

    if (!moverParam->doOpenBC) return 0;

    auto& delBdry = moverParam->deleteBoundary;
    auto& openBdry = moverParam->openBoundary;
    
    const commonType pos[3] = {xpcl, ypcl, zpcl};

    for (int side = 0; side < 6; side++) {

        if (!moverParam->applyOpenBC[side]) continue;

        const auto direction = side / 2; // x,y,z
        const auto location = pos[direction];
        const bool leftRight = side % 2; // 0: left, 1: right

        // delete boundary
        if( (leftRight == 0 && location < delBdry[side]) || (leftRight == 1 && location > delBdry[side]) ) {
            // delete the particle
            return departureArrayElementType::DELETE;
        }

        // open boundary
        if ( (leftRight == 0 && location < openBdry[side]) || (leftRight == 1 && location > openBdry[side]) ) {
            // Read velocity and charge from SoA for particle duplication
            const commonType vel[3] = {pclsArray->getU()[pidx], pclsArray->getV()[pidx], pclsArray->getW()[pidx]};
            const commonType charge = pclsArray->getQ()[pidx];

            // Compute new particle position
            commonType newPos[3] = {pos[0], pos[1], pos[2]};
            newPos[direction] += (leftRight==0 ? -1 : 1) * openBdry[direction*2];
            newPos[0] += vel[0] * moverParam->dt;
            newPos[1] += vel[1] * moverParam->dt;
            newPos[2] += vel[2] * moverParam->dt;

            // if the new particle is still in the domain
            if (
                newPos[0] > delBdry[0] && newPos[0] < delBdry[1] &&
                newPos[1] > delBdry[2] && newPos[1] < delBdry[3] &&
                newPos[2] > delBdry[4] && newPos[2] < delBdry[5]
            ) {
                departureArrayElementType element;
                const auto index = pclsArray->getNOP() + atomicAdd(&moverParam->appendCountAtomic, 1);
                // check memory overflow
                if (index >= pclsArray->getSize()) {
                    printf("Memory overflow in open boundary outflow (index=%u, size=%u)\n",
                           index, pclsArray->getSize());
                    // Cannot append: drop this duplicate and mark the original
                    // particle for deletion so it doesn't corrupt hashed sums.
                    return departureArrayElementType::DELETE;
                }
                // Write new particle to SoA arrays
                pclsArray->getX()[index] = newPos[0];
                pclsArray->getY()[index] = newPos[1];
                pclsArray->getZ()[index] = newPos[2];
                pclsArray->getU()[index] = vel[0];
                pclsArray->getV()[index] = vel[1];
                pclsArray->getW()[index] = vel[2];
                pclsArray->getQ()[index] = charge;
                pclsArray->getT()[index] = 114514.0;

                if(newPos[0] < grid->xStart)
                {
                    element.dest = departureArrayElementType::XLOW;
                }
                else if(newPos[0] > grid->xEnd)
                {
                    element.dest = departureArrayElementType::XHIGH;
                }
                else if(newPos[1] < grid->yStart)
                {
                    element.dest = departureArrayElementType::YLOW;
                }
                else if(newPos[1] > grid->yEnd)
                {
                    element.dest = departureArrayElementType::YHIGH;
                }
                else if(newPos[2] < grid->zStart)
                {
                    element.dest = departureArrayElementType::ZLOW;
                }
                else if(newPos[2] > grid->zEnd)
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

/**
 * @brief Delete particles that enter repopulation-injection depletion layers.
 *
 * @param xpcl Particle x position after the mover.
 * @param ypcl Particle y position after the mover.
 * @param zpcl Particle z position after the mover.
 * @param moverParam Device-side mover parameter bundle.
 * @param grid Device-side grid descriptor.
 * @return `DELETE` when the particle should be removed, otherwise 0.
 */
__device__ uint32_t deleteRepopulateInjection(commonType xpcl, commonType ypcl, commonType zpcl,
    moverParameter *moverParam, grid3DCUDA *grid) {
    if (!moverParam->doRepopulateInjection) return 0;

    auto& doRepopulateInjectionSide = moverParam->doRepopulateInjectionSide;
    auto& repopulateBoundary = moverParam->repopulateBoundary;

    if (
        (doRepopulateInjectionSide[0] && xpcl < repopulateBoundary[0]) ||
        (doRepopulateInjectionSide[1] && xpcl > repopulateBoundary[1]) ||
        (doRepopulateInjectionSide[2] && ypcl < repopulateBoundary[2]) ||
        (doRepopulateInjectionSide[3] && ypcl > repopulateBoundary[3]) ||
        (doRepopulateInjectionSide[4] && zpcl < repopulateBoundary[4]) ||
        (doRepopulateInjectionSide[5] && zpcl > repopulateBoundary[5])
    ) { // In the repopulate layers
        return departureArrayElementType::DELETE;
    } else {
        return 0;
    }

}

/**
 * @brief Tag particles that crossed into the absorbing planet region.
 *
 * Supports both the full 3D sphere and the 2D XZ-plane circle used by the
 * dipole-2D configuration.
 *
 * @param xpcl Particle x position after the mover.
 * @param ypcl Particle y position after the mover.
 * @param zpcl Particle z position after the mover.
 * @param moverParam Device-side mover parameter bundle.
 * @param grid Device-side grid descriptor.
 * @return `PLANET` when the particle is inside the body, otherwise 0.
 */
__device__ uint32_t deleteInsideSphere(commonType xpcl, commonType ypcl, commonType zpcl,
    moverParameter *moverParam, grid3DCUDA *grid) {
    
    if (moverParam->doSphere == 0) return 0;

    if(moverParam->doSphere == 1){ // 3D sphere
        const auto& sphereOrigin = moverParam->sphereOrigin;
        const auto& sphereRadius = moverParam->sphereRadius;

        const auto dx = xpcl - sphereOrigin[0];
        const auto dy = ypcl - sphereOrigin[1];
        const auto dz = zpcl - sphereOrigin[2];

        if (dx*dx + dy*dy + dz*dz < sphereRadius*sphereRadius) {
            return departureArrayElementType::PLANET;
        }
    } else if(moverParam->doSphere == 2){ // 2D sphere
        const auto& sphereOrigin = moverParam->sphereOrigin;
        const auto& sphereRadius = moverParam->sphereRadius;

        const auto dx = xpcl - sphereOrigin[0];
        const auto dz = zpcl - sphereOrigin[2];

        if (dx*dx + dz*dz < sphereRadius*sphereRadius) {
            return departureArrayElementType::PLANET;
        }
    }

    return 0;

}
// ======= Departure classification =======

/**
 * @brief Finalize departure metadata for one moved particle.
 *
 * The routine applies, in order, open-boundary outflow handling, repopulation
 * injection deletion, planet absorption, and regular domain-exit checks. Any
 * non-staying destination is registered in the matching hashed-sum bucket.
 *
 * @param xpcl Particle x position after the mover.
 * @param ypcl Particle y position after the mover.
 * @param zpcl Particle z position after the mover.
 * @param pclsArray Device-side particle SoA container.
 * @param pidx Particle index in the species SoA buffer.
 * @param moverParam Device-side mover parameter bundle.
 * @param departureArray Device-side departure metadata array.
 * @param grid Device-side grid descriptor.
 * @param hashedSumArray Device-side hashed-sum buckets for departure compaction.
 */
__device__ void prepareDepartureArray(commonType xpcl, commonType ypcl, commonType zpcl,
    particleArrayCUDA* pclsArray, uint32_t pidx,
    moverParameter *moverParam, departureArrayType* departureArray, grid3DCUDA* grid, hashedSum* hashedSumArray){

    if(departureArray->getArray()[pidx].dest != 0) {
        departureArray->getArray()[pidx].hashedId = 
            hashedSumArray[departureArray->getArray()[pidx].dest - 1].add(pidx);
        
            return;
    }

    // Safety: NaN positions bypass all comparison-based boundary checks
    // (IEEE 754: NaN < x and NaN > x are both false), so a NaN particle
    // would fall through to STAY and deposit NaN into the moments array.
    if (!isfinite(xpcl) || !isfinite(ypcl) || !isfinite(zpcl)) {
        departureArrayElementType element;
        element.dest = departureArrayElementType::DELETE;
        element.hashedId = hashedSumArray[departureArrayElementType::DELETE_HASHEDSUM_INDEX].add(pidx);
        departureArray->getArray()[pidx] = element;
        return;
    }
    
    departureArrayElementType element;

    do {

        // OpenBC_outflow
        element.dest = deleteAppendOpenBCOutflow(xpcl, ypcl, zpcl, pclsArray, pidx, moverParam, departureArray, grid, hashedSumArray);
        if(element.dest != 0)break;

        // INJECT
        element.dest = deleteRepopulateInjection(xpcl, ypcl, zpcl, moverParam, grid);
        if(element.dest != 0)break;

        // sphere
        element.dest = deleteInsideSphere(xpcl, ypcl, zpcl, moverParam, grid);
        if(element.dest != 0)break;

        // Exiting

        if(xpcl < grid->xStart)
        {
            element.dest = moverParam->isExitBC[0]
                ? departureArrayElementType::DELETE
                : departureArrayElementType::XLOW;
        }
        else if(xpcl > grid->xEnd)
        {
            element.dest = moverParam->isExitBC[1]
                ? departureArrayElementType::DELETE
                : departureArrayElementType::XHIGH;
        }
        else if(ypcl < grid->yStart)
        {
            element.dest = moverParam->isExitBC[2]
                ? departureArrayElementType::DELETE
                : departureArrayElementType::YLOW;
        }
        else if(ypcl > grid->yEnd)
        {
            element.dest = moverParam->isExitBC[3]
                ? departureArrayElementType::DELETE
                : departureArrayElementType::YHIGH;
        }
        else if(zpcl < grid->zStart)
        {
            element.dest = moverParam->isExitBC[4]
                ? departureArrayElementType::DELETE
                : departureArrayElementType::ZLOW;
        }
        else if(zpcl > grid->zEnd)
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
