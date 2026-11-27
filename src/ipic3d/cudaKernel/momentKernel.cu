#include "momentKernel.cuh"

using commonType = cudaTypeDouble; // calculation type

// ======= Nodes and weights, shared by the deposits =======

/** The 8 nodes of the cell whose upper corner is node (ix, iy, iz), in the
 *  order of nodeWeights. */
__device__ __forceinline__ void cellNodes(const grid3DCUDA& grid, int ix,
                                          int iy, int iz,
                                          uint32_t posIndex[8]) {
  const int nxn = grid.nxn;
  const int nyn = grid.nyn;
  const int nzn = grid.nzn;
  posIndex[0] = toOneDimIndex(nxn, nyn, nzn, ix, iy, iz);
  posIndex[1] = toOneDimIndex(nxn, nyn, nzn, ix, iy, iz - 1);
  posIndex[2] = toOneDimIndex(nxn, nyn, nzn, ix, iy - 1, iz);
  posIndex[3] = toOneDimIndex(nxn, nyn, nzn, ix, iy - 1, iz - 1);
  posIndex[4] = toOneDimIndex(nxn, nyn, nzn, ix - 1, iy, iz);
  posIndex[5] = toOneDimIndex(nxn, nyn, nzn, ix - 1, iy, iz - 1);
  posIndex[6] = toOneDimIndex(nxn, nyn, nzn, ix - 1, iy - 1, iz);
  posIndex[7] = toOneDimIndex(nxn, nyn, nzn, ix - 1, iy - 1, iz - 1);
}

/** The weights of charge qi on the nodes posIndex of cellNodes, qi invVOL^2
 *  times the trilinear volume: an upper node gets the distances to the lower
 *  node planes of the cell (xi0, eta0, zeta0), a lower node those to the
 *  upper ones (xi1, eta1, zeta1); the labels below name the factors. */
__device__ __forceinline__ void nodeWeights(commonType invVOL, commonType qi,
                                            commonType xi0, commonType xi1,
                                            commonType eta0, commonType eta1,
                                            commonType zeta0, commonType zeta1,
                                            commonType weights[8]) {
  const commonType invVOLqi = invVOL * qi;
  const commonType weight0 = invVOLqi * xi0;
  const commonType weight1 = invVOLqi * xi1;
  const commonType weight00 = weight0 * eta0;
  const commonType weight01 = weight0 * eta1;
  const commonType weight10 = weight1 * eta0;
  const commonType weight11 = weight1 * eta1;
  weights[0] = weight00 * zeta0 * invVOL; // weight000
  weights[1] = weight00 * zeta1 * invVOL; // weight001
  weights[2] = weight01 * zeta0 * invVOL; // weight010
  weights[3] = weight01 * zeta1 * invVOL; // weight011
  weights[4] = weight10 * zeta0 * invVOL; // weight100
  weights[5] = weight10 * zeta1 * invVOL; // weight101
  weights[6] = weight11 * zeta0 * invVOL; // weight110
  weights[7] = weight11 * zeta1 * invVOL; // weight111
}

/** The 10 deposited moments of velocity (u, v, w): the charge density factor
 *  one, then u, v, w, then uu, uv, uw, vv, vw, ww. */
__device__ __forceinline__ void velocityMoments(commonType one, commonType u,
                                                commonType v, commonType w,
                                                commonType vm[10]) {
  vm[0] = one;
  vm[1] = u;
  vm[2] = v;
  vm[3] = w;
  vm[4] = u * u;
  vm[5] = u * v;
  vm[6] = u * w;
  vm[7] = v * v;
  vm[8] = v * w;
  vm[9] = w * w;
}

// ======= Particle-based moment deposition =======

/** The nodes and weights of particle pidx of an unsorted deposit. */
__device__ __forceinline__ void
particleNodeWeights(uint32_t pidx, particleArrayCUDA* pclsArray,
                    const grid3DCUDA& grid, uint32_t posIndex[8],
                    commonType weights[8]) {
  commonType xpcl = pclsArray->getX()[pidx];
  commonType ypcl = pclsArray->getY()[pidx];
  commonType zpcl = pclsArray->getZ()[pidx];
  // A position outside the guarded subdomain, or NaN, is moved to its edge,
  // as the mover and the cell sorter do: the cell below then contains the
  // position, its distances to the node planes stay in [0, dx] and the
  // float-to-int conversion never sees a NaN.
  grid.make_position_safe(xpcl, ypcl, zpcl);

  int ix = 2 + int(floor((xpcl - grid.xStart) * grid.invdx));
  int iy = 2 + int(floor((ypcl - grid.yStart) * grid.invdy));
  int iz = 2 + int(floor((zpcl - grid.zStart) * grid.invdz));
  // The clamped position gives 1 <= ix <= nxn-1 (and so on); the clamp
  // below only guards against round-off at the edges of that range, which
  // would index outside the moments array.
  if (ix < 1)
    ix = 1;
  if (ix > grid.nxn - 1)
    ix = grid.nxn - 1;
  if (iy < 1)
    iy = 1;
  if (iy > grid.nyn - 1)
    iy = grid.nyn - 1;
  if (iz < 1)
    iz = 1;
  if (iz > grid.nzn - 1)
    iz = grid.nzn - 1;

  cellNodes(grid, ix, iy, iz, posIndex);
  const commonType xi0 = xpcl - grid.getXN(ix - 1);
  const commonType eta0 = ypcl - grid.getYN(iy - 1);
  const commonType zeta0 = zpcl - grid.getZN(iz - 1);
  const commonType xi1 = grid.getXN(ix) - xpcl;
  const commonType eta1 = grid.getYN(iy) - ypcl;
  const commonType zeta1 = grid.getZN(iz) - zpcl;
  nodeWeights(grid.invVOL, pclsArray->getQ()[pidx], xi0, xi1, eta0, eta1, zeta0,
              zeta1, weights);
}

__device__ __forceinline__ void
depositParticleMomentsToNodes(uint32_t pidx, particleArrayCUDA* pclsArray,
                              const grid3DCUDA& grid,
                              cudaTypeArray1<cudaMomentType> moments) {
  const commonType ui = pclsArray->getU()[pidx];
  const commonType vi = pclsArray->getV()[pidx];
  const commonType wi = pclsArray->getW()[pidx];
  uint32_t posIndex[8];
  commonType weights[8];
  particleNodeWeights(pidx, pclsArray, grid, posIndex, weights);
  commonType velmoments[10];
  velocityMoments(1.0, ui, vi, wi, velmoments);

  const uint32_t oneDensity = grid.nxn * grid.nyn * grid.nzn;
  for (int m = 0; m < 10; m++)  // 10 densities
    for (int c = 0; c < 8; c++) // 8 grid nodes
    {
      atomicAdd(&moments[oneDensity * m + posIndex[c]],
                velmoments[m] *
                    weights[c]); // device scope atomic, should be system
                                 // scope if p2p direct access
    }
}

/** The deposit of the stayed particles (see momentKernel.cuh). */
__global__ void momentKernelStayed(particleArrayCUDA* pclsArray,
                                   departureArrayType* departureArray,
                                   const uint32_t* appendCount,
                                   const KERNEL_CONSTANT grid3DCUDA grid,
                                   cudaTypeArray1<cudaMomentType> moments) {

  const uint tidx = blockIdx.x * blockDim.x + threadIdx.x;
  const uint gridSize = blockDim.x * gridDim.x;
  const uint totPcl = pclsArray->getNOP() + *appendCount;

  for (uint pidx = tidx; pidx < totPcl; pidx += gridSize) {
    if (departureArray->getArray()[pidx].dest != 0)
      continue; // skip particles already marked for departure

    depositParticleMomentsToNodes(pidx, pclsArray, grid, moments);
  }
}
/** The deposit of an unsorted tail (see momentKernel.cuh). */
__global__ void momentKernelNew(particleArrayCUDA* pclsArray,
                                const KERNEL_CONSTANT grid3DCUDA grid,
                                cudaTypeArray1<cudaMomentType> moments,
                                int stayedParticle) {

  uint pidx = stayedParticle + blockIdx.x * blockDim.x + threadIdx.x;
  if (pidx >= pclsArray->getNOP())
    return;

  depositParticleMomentsToNodes(pidx, pclsArray, grid, moments);
}

/** The heat flux of an unsorted species (see momentKernel.cuh). */
__global__ void
heatFluxKernelUnsorted(particleArrayCUDA* pclsArray,
                       const KERNEL_CONSTANT grid3DCUDA grid,
                       const cudaTypeArray1<cudaMomentType> bulkMoments,
                       cudaTypeArray1<cudaMomentType> heatFlux,
                       cudaMomentType qom, cudaMomentType rhoFloor) {
  if (qom == 0.0)
    return;

  const uint pidx = blockIdx.x * blockDim.x + threadIdx.x;
  if (pidx >= pclsArray->getNOP())
    return;

  const commonType ui = pclsArray->getU()[pidx];
  const commonType vi = pclsArray->getV()[pidx];
  const commonType wi = pclsArray->getW()[pidx];
  uint32_t posIndex[8];
  commonType weights[8];
  particleNodeWeights(pidx, pclsArray, grid, posIndex, weights);

  const uint32_t oneDensity = grid.nxn * grid.nyn * grid.nzn;
  for (int c = 0; c < 8; ++c) {
    const uint32_t node = posIndex[c];
    const commonType rho = bulkMoments[node];
    if (fabs(rho) <= rhoFloor)
      continue;

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

    const commonType q[10] = {cx2 * cx,     cx2 * cy, cx2 * cz, cx * cy2,
                              cx * cy * cz, cx * cz2, cy2 * cy, cy2 * cz,
                              cy * cz2,     cz2 * cz};

    for (int m = 0; m < 10; ++m)
      atomicAdd(&heatFlux[oneDensity * m + node], q[m] * massWeight);
  }
}

// ======= Cell-aware sorted moment deposition =======

/** The deposit of the cell-sorted prefix, one warp per cell (see
 *  momentKernel.cuh). */
__global__ void
cellAwareMomentKernel(const int* __restrict__ cell_start_offsets, int num_cells,
                      uint32_t num_to_sort, particleArrayCUDA* pclsArray,
                      const KERNEL_CONSTANT grid3DCUDA grid,
                      cudaTypeArray1<cudaMomentType> moments) {
  const int global_tid = blockIdx.x * blockDim.x + threadIdx.x;
  const int warp_id = global_tid / WARP_SIZE; // one warp per cell
  const int lane = global_tid & (WARP_SIZE - 1);

  if (warp_id >= num_cells)
    return;

  const int cell = warp_id;
  const int cell_begin = cell_start_offsets[cell];
  const int cell_end = (cell < num_cells - 1) ? cell_start_offsets[cell + 1]
                                              : static_cast<int>(num_to_sort);
  const int cell_count = cell_end - cell_begin;
  if (cell_count <= 0)
    return;

  // Recover node indices from the flat cell index.
  const int nxc = grid.nxc;
  const int nyc = grid.nyc;

  const int cz = cell / (nxc * nyc);
  const int rem = cell - cz * (nxc * nyc);
  const int cy = rem / nxc;
  const int cx = rem - cy * nxc;

  // The upper corner node of the cell (cellNodes)
  const int ix = cx + 1;
  const int iy = cy + 1;
  const int iz = cz + 1;

  // 8 surrounding node flat indices (shared by all particles in this cell)
  const uint32_t oneDensity = grid.nxn * grid.nyn * grid.nzn;
  uint32_t posIndex[8];
  cellNodes(grid, ix, iy, iz, posIndex);

  // The node planes of the cell, shared by its particles
  const commonType xn_lo = grid.getXN(ix - 1);
  const commonType xn_hi = grid.getXN(ix);
  const commonType yn_lo = grid.getYN(iy - 1);
  const commonType yn_hi = grid.getYN(iy);
  const commonType zn_lo = grid.getZN(iz - 1);
  const commonType zn_hi = grid.getZN(iz);

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
      ui = pu[pidx];
      vi = pv[pidx];
      wi = pw[pidx];
      qi = pq[pidx];
      xpcl = px[pidx];
      ypcl = py[pidx];
      zpcl = pz[pidx];
      // A particle outside this cell (the sorter puts one outside the
      // subdomain, or NaN, into an edge cell) is moved onto it, so that its
      // distances to the node planes stay in [0, dx].
      xpcl = fmin(fmax(xpcl, xn_lo), xn_hi);
      ypcl = fmin(fmax(ypcl, yn_lo), yn_hi);
      zpcl = fmin(fmax(zpcl, zn_lo), zn_hi);
    }

    commonType wt[8];
    nodeWeights(grid.invVOL, qi, xpcl - xn_lo, xn_hi - xpcl, ypcl - yn_lo,
                yn_hi - ypcl, zpcl - zn_lo, zn_hi - zpcl, wt);

    commonType vm[10];
    velocityMoments(active ? 1.0 : 0.0, ui, vi, wi, vm);

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
