// ======= Macrocell (v_par, v_perp) spectra kernel =======
//
// One CUDA block processes one slice of the cells of a macrocell (the whole
// macrocell unless the macrocells are few, macrocell::slicesFor). The block
// iterates the cells of its slice in (ix, iy, iz) order; for each cell, the 8
// corner B values are preloaded once into shared memory and shared by all
// particles of that cell. A shared-memory mini-histogram (vperp x vpar)
// accumulates atomics with minimal contention, then is flushed once to the
// bins of the macrocell in global memory at the end of the block: written
// with one slice per macrocell, added with several.

#include "cudaTypeDef.cuh"
#include "gridCUDA.cuh"
#include "macrocellSpectra.cuh"

using macrocellSpectra::macrocellHistType;

namespace {

/** Bins along v_par and v_perp, and of a macrocell. */
constexpr int kBinsVpar = DAConfig::MACROCELL_BINS_VPAR;
constexpr int kBinsVperp = DAConfig::MACROCELL_BINS_VPERP;
constexpr int kBins = macrocellSpectra::kSpectraBins;

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

} // namespace

__global__ void macrocellSpectraKernel(
    const cudaCommonType* __restrict__ x, const cudaCommonType* __restrict__ y,
    const cudaCommonType* __restrict__ z, const cudaCommonType* __restrict__ u,
    const cudaCommonType* __restrict__ v, const cudaCommonType* __restrict__ w,
    const cudaCommonType* __restrict__ q,
    const int* __restrict__ cellStartOffsets,
    const int* __restrict__ cellCounts, macrocell::MacrocellGeometry geometry,
    int slices, const cudaFieldType* __restrict__ fieldForPcls,
    const grid3DCUDA* __restrict__ grid,
    macrocellHistType* __restrict__ histOut, cudaCommonType vmax) {
  // One block per slice of a macrocell: its cells [first, end).
  const int m = blockIdx.x / slices;
  const macrocell::MacrocellBox box = geometry.box(m);
  const int lx = box.length[0], ly = box.length[1];
  int first, end;
  macrocell::sliceOf(box.cells(), blockIdx.x % slices, slices, first, end);

  extern __shared__ macrocellHistType shHist[];
  __shared__ cudaCommonType shB[macrocellSpectra::kCornerB];

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

  // Iterate the cells of the slice. Interior indices are 0-based; the
  // sorter cell index uses the guarded layout (1-based for interior), whose
  // ghost cells are in no macrocell.
  for (int k = first; k < end; ++k) {
    const int cx = box.start[0] + k % lx + 1; // the guarded index
    const int cy = box.start[1] + k / lx % ly + 1;
    const int cz = box.start[2] + k / (lx * ly) + 1;
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
    if (threadIdx.x < macrocellSpectra::kCornerB) {
      const int corner = threadIdx.x / 3;
      const int comp = threadIdx.x % 3;
      shB[corner * 3 + comp] = static_cast<cudaCommonType>(
          fieldForPcls[previousIndex * 24 + corner * 6 + comp]);
    }
    __syncthreads();

    // Stride over the cell's particles.
    for (uint32_t p = (uint32_t)s + threadIdx.x; p < (uint32_t)s + (uint32_t)n;
         p += blockDim.x) {
      const cudaCommonType xp = x[p];
      const cudaCommonType yp = y[p];
      const cudaCommonType zp = z[p];

      // the weights in cell (cx, cy, cz): the sort put the particle
      // there with the same arithmetic (get_safe_cell), and a position
      // that is not finite in a ghost cell, in no macrocell
      int ccx, ccy, ccz;
      cudaCommonType weights[8];
      grid->get_safe_cell_and_weights(xp, yp, zp, ccx, ccy, ccz, weights);

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

      const int ip = binIndex(vpar, vparMin, vparMax, kBinsVpar, invResVpar);
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

  // Flush the mini-histogram to the bins of the macrocell in global memory:
  // written with one slice, added to them (zeroed by the launch) with
  // several.
  macrocellHistType* gBins = histOut + (size_t)m * (size_t)kBins;
  for (int i = threadIdx.x; i < kBins; i += blockDim.x) {
    if (slices == 1)
      gBins[i] = shHist[i];
    else if (shHist[i] != 0)
      atomicAdd(&gBins[i], shHist[i]);
  }
}
