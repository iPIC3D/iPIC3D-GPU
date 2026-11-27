#ifndef GRID_CUDA_CUH
#define GRID_CUDA_CUH

#include "Grid3DCU.h"
#include "cudaTypeDef.cuh"

/** The grid of the rank for the GPU kernels, which take it by value
 *  (KERNEL_CONSTANT); built on the host from Grid3DCU. */
class grid3DCUDA {

public:
  /** cells and nodes per direction, ghost cells included (nxn = nxc + 1) */
  int nxc, nxn, nyc, nyn, nzc, nzn;
  /** cell size and its inverse */
  cudaParticleType dx, dy, dz, invdx, invdy, invdz;
  /** 1 / (dx dy dz) */
  cudaParticleType invVOL;

  /** local grid boundaries coordinate of proper subdomain */
  cudaParticleType xStart, xEnd, yStart, yEnd, zStart, zEnd;

private:
  /** start of the guarded subdomain, ghost cell included: xStart - dx */
  cudaParticleType xStart_g, yStart_g, zStart_g;
  /** margin, in cells, of get_safe_cell_position at the guarded edges */
  cudaCommonType epsilon;
  cudaCommonType nxc_minus_epsilon;
  cudaCommonType nyc_minus_epsilon;
  cudaCommonType nzc_minus_epsilon;

  /** The position in cells of the guarded subdomain, made safe.  Its floor
   *  is the cell of both get_safe_cell_index (the cell sorter) and
   *  get_safe_cell_and_weights (the interpolation), so that they agree. */
  __host__ __device__ void
  get_safe_cell_position(cudaParticleType xpos, cudaParticleType ypos,
                         cudaParticleType zpos, cudaParticleType& cx_pos,
                         cudaParticleType& cy_pos,
                         cudaParticleType& cz_pos) const {
    cx_pos = (xpos - xStart_g) * invdx;
    cy_pos = (ypos - yStart_g) * invdy;
    cz_pos = (zpos - zStart_g) * invdz;
    // a position outside the guarded subdomain goes to its edge; NaN to the
    // lower one, since !(x >= epsilon) is true for NaN and x < epsilon is not
    if (!(cx_pos >= epsilon))
      cx_pos = epsilon;
    if (!(cy_pos >= epsilon))
      cy_pos = epsilon;
    if (!(cz_pos >= epsilon))
      cz_pos = epsilon;
    if (cx_pos > nxc_minus_epsilon)
      cx_pos = nxc_minus_epsilon;
    if (cy_pos > nyc_minus_epsilon)
      cy_pos = nyc_minus_epsilon;
    if (cz_pos > nzc_minus_epsilon)
      cz_pos = nzc_minus_epsilon;
  }

public:
  grid3DCUDA() = default;

  __host__ explicit grid3DCUDA(const Grid3DCU* grid) {
    nxc = grid->getNXC();
    nyc = grid->getNYC();
    nzc = grid->getNZC();

    nxn = grid->getNXN();
    nyn = grid->getNYN();
    nzn = grid->getNZN();

    dx = grid->getDX();
    dy = grid->getDY();
    dz = grid->getDZ();

    invdx = grid->get_invdx();
    invdy = grid->get_invdy();
    invdz = grid->get_invdz();

    invVOL = grid->getInvVOL();

    xStart = grid->getXstart();
    yStart = grid->getYstart();
    zStart = grid->getZstart();

    xEnd = grid->getXend();
    yEnd = grid->getYend();
    zEnd = grid->getZend();

    xStart_g = xStart - dx;
    yStart_g = yStart - dy;
    zStart_g = zStart - dz;

    // small, but floor(nxc - epsilon) is still the last cell, nxc - 1
    epsilon = (nxc + nyc + nzc) * 1e-15;
    nxc_minus_epsilon = nxc - epsilon;
    nyc_minus_epsilon = nyc - epsilon;
    nzc_minus_epsilon = nzc - epsilon;
  }

  /** The flattened index cx + cy nxc + cz nxc nyc (x fastest) of the cell of
   *  get_safe_cell_position: the cells of the cell sorter. */
  __host__ __device__ int get_safe_cell_index(cudaParticleType xpos,
                                              cudaParticleType ypos,
                                              cudaParticleType zpos) const {
    cudaParticleType cx_pos, cy_pos, cz_pos;
    get_safe_cell_position(xpos, ypos, zpos, cx_pos, cy_pos, cz_pos);
    return int(floor(cx_pos)) + int(floor(cy_pos)) * nxc +
           int(floor(cz_pos)) * nxc * nyc;
  }

  /** The cell of get_safe_cell_position and the trilinear weights of the
   *  position on its 8 nodes. */
  __host__ __device__ void
  get_safe_cell_and_weights(cudaParticleType xpos, cudaParticleType ypos,
                            cudaParticleType zpos, int& cx, int& cy, int& cz,
                            cudaParticleType weights[8]) const {
    cudaParticleType cx_pos, cy_pos, cz_pos;
    get_safe_cell_position(xpos, ypos, zpos, cx_pos, cy_pos, cz_pos);
    cx = int(floor(cx_pos));
    cy = int(floor(cy_pos));
    cz = int(floor(cz_pos));

    // fraction of distance from the left
    const cudaParticleType w0x = cx_pos - cx;
    const cudaParticleType w0y = cy_pos - cy;
    const cudaParticleType w0z = cz_pos - cz;
    // fraction of the distance from the right of the cell
    const cudaParticleType w1x = 1.f - w0x;
    const cudaParticleType w1y = 1.f - w0y;
    const cudaParticleType w1z = 1.f - w0z;

    weights[0] = w1x * w1y * w1z; // weight111
    weights[1] = w0x * w1y * w1z; // weight011
    weights[2] = w0x * w0y * w1z; // weight001
    weights[3] = w1x * w0y * w1z; // weight101
    weights[4] = w1x * w1y * w0z; // weight110
    weights[5] = w0x * w1y * w0z; // weight010
    weights[6] = w0x * w0y * w0z; // weight000
    weights[7] = w1x * w0y * w0z; // weight100
  }

  /** The position of node index (node 1 is at xStart, node 0 the ghost
   *  node); signed, so that index 0 does not wrap. */
  __device__ double getXN(int index) const { return xStart + (index - 1) * dx; }
  __device__ double getYN(int index) const { return yStart + (index - 1) * dy; }
  __device__ double getZN(int index) const { return zStart + (index - 1) * dz; }

  /** Move a position into the guarded subdomain, the range that
   *  get_safe_cell_position allows: a position outside it, or NaN, goes to
   *  its edge (NaN to the lower one).  A position inside is unchanged.
   *  fmin/fmax, no branch. */
  __host__ __device__ void make_position_safe(cudaParticleType& xpos,
                                              cudaParticleType& ypos,
                                              cudaParticleType& zpos) const {
    xpos = fmin(fmax(xpos, xStart_g + epsilon * dx),
                xStart_g + nxc_minus_epsilon * dx);
    ypos = fmin(fmax(ypos, yStart_g + epsilon * dy),
                yStart_g + nyc_minus_epsilon * dy);
    zpos = fmin(fmax(zpos, zStart_g + epsilon * dz),
                zStart_g + nzc_minus_epsilon * dz);
  }
};
static_assert(std::is_trivially_copyable_v<grid3DCUDA>,
              "a kernel parameter (KERNEL_CONSTANT) is copied bytewise");

#endif // GRID_CUDA_CUH
