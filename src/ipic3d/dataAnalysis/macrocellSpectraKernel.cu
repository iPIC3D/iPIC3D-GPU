// ======= Macrocell (v_par, v_perp) spectra kernel =======
//
// One CUDA block processes one macrocell. The block iterates the macrocell's
// cells in (ix, iy, iz) order; for each cell, the 8 corner B values are
// preloaded once into shared memory and shared by all particles of that cell.
// A shared-memory mini-histogram (vperp x vpar) accumulates atomics with
// minimal contention, then is flushed once to the global per-macrocell slice
// at the end of the block.

#include "cudaTypeDef.cuh"
#include "gridCUDA.cuh"
#include "macrocellSpectra.cuh"

using macrocellSpectra::macrocellHistType;

/** Bins along v_par and v_perp, and of a macrocell. */
inline constexpr int kBinsVpar = DAConfig::MACROCELL_BINS_VPAR;
inline constexpr int kBinsVperp = DAConfig::MACROCELL_BINS_VPERP;
inline constexpr int kBins = macrocellSpectra::macrocellSpectra2D::Nb;

namespace {

__device__ __forceinline__ int binIndex(cudaCommonType v, cudaCommonType vmin,
                                        cudaCommonType vmax, int bins,
                                        cudaCommonType invRes) {
  if (!(v >= vmin) || v > vmax)
    return -1; // outside, or NaN
  int b = (int)((v - vmin) * invRes);
  if (b == bins)
    b = bins - 1;
  if (b < 0 || b >= bins)
    return -1;
  return b;
}

/** The cells of a macrocell along an axis, as interior indices (-1 and
 *  interiorCells are the ghost cells). The first and the last macrocell also
 *  take the ghost cell next to them: a particle exactly on the upper boundary
 *  of the rank has its index, one on the lower boundary can round into it. */
__device__ __forceinline__ void macrocellAxisRange(int macrocell,
                                                   int requestedSize,
                                                   int interiorCells,
                                                   int& start, int& length) {
  start = macrocell * requestedSize;
  const int end = start + requestedSize < interiorCells ? start + requestedSize
                                                        : interiorCells + 1;
  if (macrocell == 0)
    start = -1;
  length = end - start;
}

} // namespace

__global__ void macrocellSpectraKernel(
    const cudaCommonType* __restrict__ x, const cudaCommonType* __restrict__ y,
    const cudaCommonType* __restrict__ z, const cudaCommonType* __restrict__ u,
    const cudaCommonType* __restrict__ v, const cudaCommonType* __restrict__ w,
    const cudaCommonType* __restrict__ q,
    const int* __restrict__ cellStartOffsets,
    const int* __restrict__ cellCounts, int Nx_int, int Ny_int, int Nz_int,
    int Cx, int Cy, int Cz, const cudaFieldType* __restrict__ fieldForPcls,
    const grid3DCUDA* __restrict__ grid,
    macrocellHistType* __restrict__ histOut, cudaCommonType vmax) {
  // One block per macrocell, the grid of the macrocells.
  const int mx = blockIdx.x;
  const int my = blockIdx.y;
  const int mz = blockIdx.z;
  const int m = (mz * gridDim.y + my) * gridDim.x + mx;

  int sx, lx, sy, ly, sz, lz;
  macrocellAxisRange(mx, Cx, Nx_int, sx, lx);
  macrocellAxisRange(my, Cy, Ny_int, sy, ly);
  macrocellAxisRange(mz, Cz, Nz_int, sz, lz);

  extern __shared__ macrocellHistType shHist[];
  __shared__ cudaCommonType shB[24]; // 8 corners x 3 B components.

  // Bin geometry (vpar in [-vmax,+vmax], vperp in [0,vmax]).
  const cudaCommonType vparMin = -vmax;
  const cudaCommonType vparMax = vmax;
  const cudaCommonType vperpMin = 0.0;
  const cudaCommonType vperpMax = vmax;
  const cudaCommonType invResVpar =
      (cudaCommonType)kBinsVpar / (vparMax - vparMin);
  const cudaCommonType invResVperp =
      (cudaCommonType)kBinsVperp / (vperpMax - vperpMin);

  // Zero the mini-histogram.
  for (int i = threadIdx.x; i < kBins; i += blockDim.x)
    shHist[i] = (macrocellHistType)0;
  __syncthreads();

  const int nxc = grid->nxc;
  const int nyc = grid->nyc;
  const int nzn = grid->nzn;
  const int nyn = grid->nyn;

  // Iterate cells inside the macrocell. Interior indices are 0-based; the
  // sorter cell index uses the guarded layout (1-based for interior), whose
  // ghost cells the first and the last macrocell take too.
  for (int iz = 0; iz < lz; ++iz)
    for (int iy = 0; iy < ly; ++iy)
      for (int ix = 0; ix < lx; ++ix) {
        const int cx = sx + ix + 1; // the guarded index
        const int cy = sy + iy + 1;
        const int cz = sz + iz + 1;
        const int c = cx + cy * nxc + cz * nxc * nyc;

        const int n = cellCounts[c];
        if (n == 0) // the same for the block; touches no shared memory
          continue;
        const int s = cellStartOffsets[c];

        // Preload 8 corners x 3 B components from fieldForPcls.
        // previousIndex layout matches GridFields in moverKernel.cu:
        //   previousIndex = (cx*(nyn-1) + cy)*nzn + cz
        //   field[previousIndex*24 + corner*6 + comp]
        const int previousIndex = (cx * (nyn - 1) + cy) * nzn + cz;
        if (threadIdx.x < 24) {
          const int corner = threadIdx.x / 3;
          const int comp = threadIdx.x % 3;
          shB[corner * 3 + comp] = static_cast<cudaCommonType>(
              fieldForPcls[previousIndex * 24 + corner * 6 + comp]);
        }
        __syncthreads();

        // Stride over the cell's particles.
        for (uint32_t p = (uint32_t)s + threadIdx.x;
             p < (uint32_t)s + (uint32_t)n; p += blockDim.x) {
          const cudaCommonType xp = x[p];
          const cudaCommonType yp = y[p];
          const cudaCommonType zp = z[p];

          int ccx, ccy, ccz;
          cudaCommonType weights[8];
          grid->get_safe_cell_and_weights(xp, yp, zp, ccx, ccy, ccz, weights);

          // the sorter takes the same cell (get_safe_cell): another one, or
          // a position that is not finite, means no valid B here
          if (ccx != cx || ccy != cy || ccz != cz ||
              !(isfinite_fast(xp) && isfinite_fast(yp) && isfinite_fast(zp)))
            continue;

          cudaCommonType Bx = 0, By = 0, Bz = 0;
#pragma unroll
          for (int kc = 0; kc < 8; ++kc) {
            Bx += weights[kc] * shB[kc * 3 + 0];
            By += weights[kc] * shB[kc * 3 + 1];
            Bz += weights[kc] * shB[kc * 3 + 2];
          }
          const cudaCommonType Bmag = sqrt(Bx * Bx + By * By + Bz * Bz);
          if (!(Bmag > DAConfig::MACROCELL_BMIN))
            continue;

          const cudaCommonType charge = q[p];
          if (!isfinite_fast(charge)) // as the velocity histogram
            continue;
          const cudaCommonType up = u[p], vp = v[p], wp = w[p];
          const cudaCommonType vpar = (up * Bx + vp * By + wp * Bz) / Bmag;
          const cudaCommonType v2 = up * up + vp * vp + wp * wp;
          cudaCommonType vperp2 = v2 - vpar * vpar;
          if (vperp2 < 0)
            vperp2 = 0;
          const cudaCommonType vperp = sqrt(vperp2);

          const int ip =
              binIndex(vpar, vparMin, vparMax, kBinsVpar, invResVpar);
          const int ie =
              binIndex(vperp, vperpMin, vperpMax, kBinsVperp, invResVperp);
          if (ip < 0 || ie < 0)
            continue;

          // weight |q| * VELOCITY_WEIGHT_SCALE, as the velocity histogram
          const macrocellHistType wgt =
              (macrocellHistType)fabs(charge * DAConfig::VELOCITY_WEIGHT_SCALE);
          atomicAdd(&shHist[ie * kBinsVpar + ip], wgt);
        }
        __syncthreads();
      }

  // Flush the mini-histogram to the global per-macrocell slice.
  macrocellHistType* gSlice = histOut + (size_t)m * (size_t)kBins;
  for (int i = threadIdx.x; i < kBins; i += blockDim.x) {
    gSlice[i] = shHist[i];
  }
}
