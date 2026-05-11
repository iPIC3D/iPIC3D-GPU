// ======= Macrocell (v_par, v_perp) spectra kernel =======
//
// One CUDA block processes one macrocell. The block iterates the macrocell's
// cells in (ix, iy, iz) order; for each cell, the 8 corner B values are
// preloaded once into shared memory and shared by all particles of that cell.
// A shared-memory mini-histogram (vperp x vpar) accumulates atomics with
// minimal contention, then is flushed once to the global per-macrocell slice
// at the end of the block.

#include "macrocellSpectra.cuh"
#include "cudaTypeDef.cuh"
#include "gridCUDA.cuh"

using macrocellSpectra::macrocellHistType;

namespace {

__device__ __forceinline__
int binIndex(cudaCommonType v, cudaCommonType vmin, cudaCommonType vmax,
             int bins, cudaCommonType invRes)
{
    if (!(v >= vmin) || v >  vmax) return -1;     // NaN-safe lower clamp
    int b = (int)((v - vmin) * invRes);
    if (b == bins) b = bins - 1;
    if (b < 0 || b >= bins) return -1;
    return b;
}

__device__ __forceinline__
void macrocellAxisRange(int macrocell, int requestedSize, int interiorCells,
                        int& start, int& length)
{
    start  = macrocell * requestedSize;
    const int remaining = interiorCells - start;
    length = (requestedSize < remaining) ? requestedSize : remaining;
}

} // namespace


__global__ void macrocellSpectraKernel(
    const cudaCommonType* __restrict__ x,
    const cudaCommonType* __restrict__ y,
    const cudaCommonType* __restrict__ z,
    const cudaCommonType* __restrict__ u,
    const cudaCommonType* __restrict__ v,
    const cudaCommonType* __restrict__ w,
    const cudaCommonType* __restrict__ q,
    const int* __restrict__ cellStartOffsets,
    const int* __restrict__ cellCounts,
    int Nx_int, int Ny_int, int Nz_int,
    int Cx, int Cy, int Cz,
    int Mx, int My, int Mz,
    const cudaFieldType* __restrict__ fieldForPcls,
    const grid3DCUDA*     __restrict__ grid,
    macrocellHistType* __restrict__ histOut,
    cudaCommonType vmax,
    cudaCommonType bMin)
{
    // One block per macrocell.
    const int mx = blockIdx.x;
    const int my = blockIdx.y;
    const int mz = blockIdx.z;
    if (mx >= Mx || my >= My || mz >= Mz) return;
    const int m = (mz * My + my) * Mx + mx;

    int sx, lx, sy, ly, sz, lz;
    macrocellAxisRange(mx, Cx, Nx_int, sx, lx);
    macrocellAxisRange(my, Cy, Ny_int, sy, ly);
    macrocellAxisRange(mz, Cz, Nz_int, sz, lz);

    constexpr int binsVpar  = DAConfig::MACROCELL_BINS_VPAR;
    constexpr int binsVperp = DAConfig::MACROCELL_BINS_VPERP;
    const int Nb = binsVpar * binsVperp;

    extern __shared__ macrocellHistType shHist[];
    __shared__ cudaCommonType shB[24]; // 8 corners x 3 B components.

    // Bin geometry (vpar in [-vmax,+vmax], vperp in [0,vmax]).
    const cudaCommonType vparMin   = -vmax;
    const cudaCommonType vparMax   =  vmax;
    const cudaCommonType vperpMin  = 0.0;
    const cudaCommonType vperpMax  = vmax;
    const cudaCommonType invResVpar  = (cudaCommonType)binsVpar  / (vparMax  - vparMin);
    const cudaCommonType invResVperp = (cudaCommonType)binsVperp / (vperpMax - vperpMin);

    // Zero the mini-histogram.
    for (int i = threadIdx.x; i < Nb; i += blockDim.x) shHist[i] = (macrocellHistType)0;
    __syncthreads();

    const int nxc = grid->nxc;
    const int nyc = grid->nyc;
    const int nzn = grid->nzn;
    const int nyn = grid->nyn;

    // Iterate cells inside the macrocell. Interior indices are 0-based; the
    // sorter cell index uses the guarded layout (1-based for interior).
    for (int iz = 0; iz < lz; ++iz)
    for (int iy = 0; iy < ly; ++iy)
    for (int ix = 0; ix < lx; ++ix)
    {
        const int cx = sx + ix + 1;   // skip ghost layer
        const int cy = sy + iy + 1;
        const int cz = sz + iz + 1;
        const int c  = cx + cy * nxc + cz * nxc * nyc;

        const int n = cellCounts[c];
        if (n == 0) {
            __syncthreads();
            continue;
        }
        const int s = cellStartOffsets[c];

        // Preload 8 corners x 3 B components from fieldForPcls.
        // previousIndex layout matches sampleFieldsAtPosition() in moverKernel.cu:
        //   previousIndex = (cx*(nyn-1) + cy)*nzn + cz
        //   field[previousIndex*24 + corner*6 + comp]
        const int previousIndex = (cx * (nyn - 1) + cy) * nzn + cz;
        if (threadIdx.x < 24) {
            const int corner = threadIdx.x / 3;
            const int comp   = threadIdx.x % 3;
            shB[corner * 3 + comp] = static_cast<cudaCommonType>(
                fieldForPcls[previousIndex * 24 + corner * 6 + comp]
            );
        }
        __syncthreads();

        // Stride over the cell's particles.
        for (uint32_t p = (uint32_t)s + threadIdx.x;
             p < (uint32_t)s + (uint32_t)n;
             p += blockDim.x)
        {
            const cudaCommonType xp = x[p];
            const cudaCommonType yp = y[p];
            const cudaCommonType zp = z[p];

            int ccx, ccy, ccz;
            cudaCommonType weights[8];
            grid->get_safe_cell_and_weights(xp, yp, zp, ccx, ccy, ccz, weights);

            // If the particle landed in another cell (rare floating-point
            // edge case at the border) skip: B preload would be wrong.
            if (ccx != cx || ccy != cy || ccz != cz) continue;

            cudaCommonType Bx = 0, By = 0, Bz = 0;
            #pragma unroll
            for (int kc = 0; kc < 8; ++kc) {
                Bx += weights[kc] * shB[kc * 3 + 0];
                By += weights[kc] * shB[kc * 3 + 1];
                Bz += weights[kc] * shB[kc * 3 + 2];
            }
            const cudaCommonType Bmag = sqrt(Bx*Bx + By*By + Bz*Bz);
            if (!(Bmag > bMin)) continue;

            const cudaCommonType up = u[p], vp = v[p], wp = w[p];
            const cudaCommonType vpar  = (up*Bx + vp*By + wp*Bz) / Bmag;
            const cudaCommonType v2    = up*up + vp*vp + wp*wp;
            cudaCommonType vperp2      = v2 - vpar * vpar;
            if (vperp2 < 0) vperp2 = 0;
            const cudaCommonType vperp = sqrt(vperp2);

            const int ip = binIndex(vpar,  vparMin,  vparMax,  binsVpar,  invResVpar);
            const int ie = binIndex(vperp, vperpMin, vperpMax, binsVperp, invResVperp);
            if (ip < 0 || ie < 0) continue;

            // Weight by |q| (matches velocityHistogramKernel scaling).
            const macrocellHistType wgt = (macrocellHistType)fabs((double)q[p] * 1.0e7);
            atomicAdd(&shHist[ie * binsVpar + ip], wgt);
        }
        __syncthreads();
    }

    // Flush the mini-histogram to the global per-macrocell slice.
    macrocellHistType* gSlice = histOut + (size_t)m * (size_t)Nb;
    for (int i = threadIdx.x; i < Nb; i += blockDim.x) {
        gSlice[i] = shHist[i];
    }
}
