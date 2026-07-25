

#include "cudaTypeDef.cuh"
#include "particleControlKernel.cuh"

#include "gridCUDA.cuh"
#include "particleArrayCUDA.cuh"
#include "particleExchange.cuh"

// ======= Merge kernel =======

/**
 * @brief Merge close-velocity particle pairs inside a cell-sorted SoA buffer.
 *
 * Each warp owns one cell. The kernel looks for the closest not-yet-deleted
 * partner for each lead particle, merges charges and phase-space coordinates
 * conservatively, and marks the absorbed particle for deletion.
 * @param cellOffsetList Per-cell starting offsets in the sorted particle
 * buffer.
 * @param cellBinCountList Per-cell particle counts.
 * @param grid Device grid metadata used to compute cell statistics.
 * @param pclArray Cell-sorted particle SoA buffer.
 * @param departureArray Per-particle destination metadata used for deletion
 * flags.
 */
__global__ void mergingKernel(int* cellOffsetList, int* cellBinCountList,
                              grid3DCUDA* grid, particleArrayCUDA* pclArray,
                              departureArrayType* departureArray) {

  const uint pid = blockIdx.x * blockDim.x + threadIdx.x;
  const uint warpId = pid / WARP_SIZE;
  const auto& cellId = warpId;
  const uint laneId = pid % WARP_SIZE;

  // return if pid > number of particle rounded up to warpsize
  const int nop = pclArray->getNOP();
  // SoA field pointers
  auto soaU = pclArray->getU();
  auto soaV = pclArray->getV();
  auto soaW = pclArray->getW();
  auto soaQ = pclArray->getQ();
  auto soaX = pclArray->getX();
  auto soaY = pclArray->getY();
  auto soaZ = pclArray->getZ();
  auto dArray = departureArray->getArray();
  const uint cellNum = ((grid->nxc) * (grid->nyc) * (grid->nzc));
  if (cellId >= cellNum)
    return;

  // cell offset for this warp
  const int cellOffset = cellOffsetList[cellId];
  // number of particles in this cell
  const int numPIC = cellBinCountList[cellId];

  const int initialPIC = pclArray->getInitialNOP() /
                         ((grid->nxc - 2) * (grid->nyc - 2) * (grid->nzc - 2));

  if (numPIC <= initialPIC)
    return; // already at or below the target occupancy

  int cellMergeCount = 0; // number of particles merged in this cell

  // main loop for one cell, pushing right
  for (int p = 0; p < numPIC; p++) {
    const int mainPId = cellOffset + p;

    constexpr cudaParticleType threshold =
        0.009;                       // threshold for merging, percentage of
                                     // the velocity of the main particle
    cudaParticleType minNorm = 1e10; // minimum norm
    int minPId = -1;                 // minimum particle id

    // each thread calculate the VV norm between the particles it holds and the
    // main loop particle keep the smallest one
    for (int i = laneId; i < numPIC; i += WARP_SIZE) {
      const int pId = cellOffset + i;
      if (pId <= mainPId)
        continue; // ignore self, and the past particles

      // ignore if the particle is marked for deletion
      if (dArray[pId].dest != 0)
        continue;

      // calculate the VV norm
      const auto u1 = soaU[mainPId];
      const auto v1 = soaV[mainPId];
      const auto w1 = soaW[mainPId];

      const auto u2 = soaU[pId];
      const auto v2 = soaV[pId];
      const auto w2 = soaW[pId];

      const auto norm = (u1 - u2) * (u1 - u2) + (v1 - v2) * (v1 - v2) +
                        (w1 - w2) * (w1 - w2); // not distance, reduce the sqrt

      if (norm < minNorm) {
        minNorm = norm;
        minPId = pId;
      }
    }

    // warp reduce
    auto localNorm = minNorm;
    auto localPId = minPId;
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
      auto otherVal = __shfl_down_sync(WARP_FULL_MASK, localNorm, offset);
      auto otherLane = __shfl_down_sync(WARP_FULL_MASK, localPId, offset);

      if (otherVal < localNorm) {
        localNorm = otherVal;
        localPId = otherLane;
      }
    }

    // lane 0 holds the minimum norm
    if (laneId == 0) {
      const auto u1 = soaU[mainPId];
      const auto v1 = soaV[mainPId];
      const auto w1 = soaW[mainPId];

      if (localNorm < (threshold * (u1 * u1 + v1 * v1 + w1 * w1))) { // merge!
        // if (true) { // merge!
        dArray[localPId].dest = departureArrayElementType::DELETE;

        const auto q1 = soaQ[mainPId];
        const auto x1 = soaX[mainPId];
        const auto y1 = soaY[mainPId];
        const auto z1 = soaZ[mainPId];

        const auto u2 = soaU[localPId];
        const auto v2 = soaV[localPId];
        const auto w2 = soaW[localPId];
        const auto q2 = soaQ[localPId];
        const auto x2 = soaX[localPId];
        const auto y2 = soaY[localPId];
        const auto z2 = soaZ[localPId];

        const auto newQ = q1 + q2;
        soaU[mainPId] = (u1 * q1 + u2 * q2) / newQ;
        soaV[mainPId] = (v1 * q1 + v2 * q2) / newQ;
        soaW[mainPId] = (w1 * q1 + w2 * q2) / newQ;
        soaQ[mainPId] = newQ;
        soaX[mainPId] = (x1 * q1 + x2 * q2) / newQ;
        soaY[mainPId] = (y1 * q1 + y2 * q2) / newQ;
        soaZ[mainPId] = (z1 * q1 + z2 * q2) / newQ;

        cellMergeCount++;
      }
    }
    // return this warp if reaches the initial number of particles
    cellMergeCount = __shfl_sync(WARP_FULL_MASK, cellMergeCount, 0);
    if ((numPIC - cellMergeCount) <= initialPIC)
      return;
  }
}

using commonType = cudaParticleType;

// ======= Particle splitting kernels =======

/**
 * @brief Split a subset of particles when fewer new particles are needed than
 * currently exist.
 *
 * The launch uses `deltaPcl` threads. Each thread selects one source particle,
 * halves its charge, offsets the original and clone within the same cell, and
 * writes the new particle at the end of the SoA buffer.
 * @param moverParam Mover context holding the particle buffer to expand.
 * @param grid Device grid metadata used to keep daughter particles in-cell.
 */
template <>
__global__ void particleSplittingKernel<false>(moverParameter* moverParam,
                                               grid3DCUDA* grid) {
  const uint tidx = blockIdx.x * blockDim.x + threadIdx.x;
  auto pclsArray = moverParam->pclsArray;
  const uint deltaPcl = pclsArray->getInitialNOP() - pclsArray->getNOP();
  if (tidx >= deltaPcl)
    return;

  // grid properties
  const commonType& inv_dx = grid->invdx;
  const commonType& inv_dy = grid->invdy;
  const commonType& inv_dz = grid->invdz;
  const commonType& xstart = grid->xStart;
  const commonType& ystart = grid->yStart;
  const commonType& zstart = grid->zStart;

  // batch must be >= 1 --> there is no safety check, it must be checked before
  // launching the kernel
  const uint batch = pclsArray->getNOP() / deltaPcl;

  // generate random idx in [0,batch-1] to select the particle to split
  // based on LCRNG
  uint idxRNG = 0;
  if (batch > 1) {
    const uint seed = (1313492u + tidx);
    // seed ^= (seed >> 7);
    idxRNG = (seed * deltaPcl + batch - 1) % batch;
  }
  // select particle to split
  const uint pidx = tidx * batch + idxRNG;

  // copy the particle from SoA
  auto soaU = pclsArray->getU();
  auto soaV = pclsArray->getV();
  auto soaW = pclsArray->getW();
  auto soaQ = pclsArray->getQ();
  auto soaX = pclsArray->getX();
  auto soaY = pclsArray->getY();
  auto soaZ = pclsArray->getZ();
  auto soaID = pclsArray->getID();
  const bool trackParticleID = pclsArray->tracksParticleID();

  const auto x0 = soaX[pidx];
  const auto y0 = soaY[pidx];
  const auto z0 = soaZ[pidx];

  // index of the grid point to the right of the particle
  const int ix = 2 + int(floor((x0 - xstart) * inv_dx));
  const int iy = 2 + int(floor((y0 - ystart) * inv_dy));
  const int iz = 2 + int(floor((z0 - zstart) * inv_dz));
  // distance particle - grid point to the left
  const commonType xi0 = x0 - grid->getXN(ix - 1);
  const commonType yi0 = y0 - grid->getYN(iy - 1);
  const commonType zi0 = z0 - grid->getZN(iz - 1);
  // distance particle - grid point to the right
  const commonType xi1 = grid->getXN(ix) - x0;
  const commonType yi1 = grid->getYN(iy) - y0;
  const commonType zi1 = grid->getZN(iz) - z0;

  // select the lowest distance to ensure keeping the particles in the cell
  cudaTypeDouble delta = xi0;
  if (yi0 < delta)
    delta = yi0;
  if (zi0 < delta)
    delta = zi0;
  if (xi1 < delta)
    delta = xi1;
  if (yi1 < delta)
    delta = yi1;
  if (zi1 < delta)
    delta = zi1;

  delta /= 20;
  // Update original particle position in SoA
  soaX[pidx] = x0 - delta;
  soaY[pidx] = y0 - delta;
  soaZ[pidx] = z0 - delta;

  // Update charge consistently for both daughter particles.
  const auto q = soaQ[pidx];
  soaQ[pidx] = 0.5 * q;

  // Write new split particle to SoA at the end of the array
  const auto index = pclsArray->getNOP() + tidx;
  // check memory overflow
  if (index >= moverParam->pclsArray->getSize()) {
    printf("Memory overflow in open boundary outflow\n");
    //__trap();
    return;
  }
  soaU[index] = soaU[pidx];
  soaV[index] = soaV[pidx];
  soaW[index] = soaW[pidx];
  soaQ[index] = 0.5 * q;
  soaX[index] = x0 + delta;
  soaY[index] = y0 + delta;
  soaZ[index] = z0 + delta;
  if (trackParticleID)
    soaID[index] = moverParam->particleIDGenerator.generateID();
}
/**
 * @brief Split every existing particle multiple times when the deficit exceeds
 * the current population.
 *
 * Each thread repeatedly clones the particle it owns until the requested
 * particle deficit has been filled.
 * @param moverParam Mover context holding the particle buffer to expand.
 * @param grid Device grid metadata used to keep daughter particles in-cell.
 */
template <>
__global__ void particleSplittingKernel<true>(moverParameter* moverParam,
                                              grid3DCUDA* grid) {
  const uint tidx = blockIdx.x * blockDim.x + threadIdx.x;
  auto pclsArray = moverParam->pclsArray;
  if (tidx >= pclsArray->getNOP())
    return;

  const uint deltaPcl = pclsArray->getInitialNOP() - pclsArray->getNOP();

  // it is assumed deltaPcl >= pclsArray->getNOP() - no safety check
  const uint splittingTimes = deltaPcl / pclsArray->getNOP();

  // grid properties
  const commonType& inv_dx = grid->invdx;
  const commonType& inv_dy = grid->invdy;
  const commonType& inv_dz = grid->invdz;
  const commonType& xstart = grid->xStart;
  const commonType& ystart = grid->yStart;
  const commonType& zstart = grid->zStart;

  // SoA field pointers
  auto soaU = pclsArray->getU();
  auto soaV = pclsArray->getV();
  auto soaW = pclsArray->getW();
  auto soaQ = pclsArray->getQ();
  auto soaX = pclsArray->getX();
  auto soaY = pclsArray->getY();
  auto soaZ = pclsArray->getZ();
  auto soaID = pclsArray->getID();
  const bool trackParticleID = pclsArray->tracksParticleID();

  for (int i = 0; i < splittingTimes; i++) {
    const uint pidx = i * pclsArray->getNOP() + tidx;

    const auto x0 = soaX[pidx];
    const auto y0 = soaY[pidx];
    const auto z0 = soaZ[pidx];

    // index of the grid point to the right of the particle
    const int ix = 2 + int(floor((x0 - xstart) * inv_dx));
    const int iy = 2 + int(floor((y0 - ystart) * inv_dy));
    const int iz = 2 + int(floor((z0 - zstart) * inv_dz));
    // distance particle - grid point to the left
    const commonType xi0 = x0 - grid->getXN(ix - 1);
    const commonType yi0 = y0 - grid->getYN(iy - 1);
    const commonType zi0 = z0 - grid->getZN(iz - 1);
    // distance particle - grid point to the right
    const commonType xi1 = grid->getXN(ix) - x0;
    const commonType yi1 = grid->getYN(iy) - y0;
    const commonType zi1 = grid->getZN(iz) - z0;

    // select the lowest distance to ensure keeping the particles in the cell
    cudaTypeDouble delta = xi0;
    if (yi0 < delta)
      delta = yi0;
    if (zi0 < delta)
      delta = zi0;
    if (xi1 < delta)
      delta = xi1;
    if (yi1 < delta)
      delta = yi1;
    if (zi1 < delta)
      delta = zi1;

    delta /= 20;
    // Update original particle position in SoA
    soaX[pidx] = x0 - delta;
    soaY[pidx] = y0 - delta;
    soaZ[pidx] = z0 - delta;

    // Update charge consistently for both daughter particles.
    const auto q = soaQ[pidx];
    soaQ[pidx] = 0.5 * q;

    // Write new split particle to SoA at the end of the array
    const auto index = pclsArray->getNOP() + pidx;
    // check memory overflow
    if (index >= moverParam->pclsArray->getSize()) {
      printf("Memory overflow in open boundary outflow\n");
      //__trap();
      return;
    }
    soaU[index] = soaU[pidx];
    soaV[index] = soaV[pidx];
    soaW[index] = soaW[pidx];
    soaQ[index] = 0.5 * q;
    soaX[index] = x0 + delta;
    soaY[index] = y0 + delta;
    soaZ[index] = z0 + delta;
    if (trackParticleID)
      soaID[index] = moverParam->particleIDGenerator.generateID();
  }
}
