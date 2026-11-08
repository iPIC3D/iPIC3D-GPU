#ifndef GPU_POINT_LAUNCH_CUH
#define GPU_POINT_LAUNCH_CUH

/**
 * @file GPUPointLaunch.cuh
 * @brief Launch shape of the 3D point kernels (stencils, MUdot, the local
 *        Maxwell operator): one thread per point of a box of a row-major
 *        [i][j][k] array (k contiguous).
 *
 * The contiguous index k runs on threadIdx.x, so the threads of a warp read
 * consecutive addresses; j runs on y and i on z.  A block has up to 256
 * threads and shrinks along an axis with fewer points, so the degenerate axis
 * of a 2D run (y for x-z, z for x-y: one cell, two nodes) leaves no thread
 * idle.
 * Kernels recover the point as
 *   i = iLo + blockIdx.z * blockDim.z + threadIdx.z
 *   j = jLo + blockIdx.y * blockDim.y + threadIdx.y
 *   k = kLo + blockIdx.x * blockDim.x + threadIdx.x
 * and return outside [iLo..iHi] x [jLo..jHi] x [kLo..kHi].
 *
 * Warp width: a k row has at most 32 points.  A 64-lane AMD wavefront then
 * reads two contiguous 256-B runs of doubles, at most one cache line more
 * than a single 64-point row, and avoids the idle lanes a 64-wide row leaves
 * when nk is not a multiple of 64 (96 cells per rank).  The point kernels use
 * no warp operations, so results do not depend on the warp width.
 */

#include "cudaTypeDef.cuh"

/** Threads of a full block. */
inline constexpr int kPointBlockThreads = 256;
/** Most points along k (one row). */
inline constexpr int kPointRowMax = 32;
/** Largest blockDim.z on NVIDIA (AMD allows 1024). */
inline constexpr int kPointBlockZMax = 64;
static_assert(kPointBlockThreads % WARP_SIZE == 0,
              "a full point block must hold whole warps (wavefronts)");

struct PointLaunch {
  dim3 grid, block;
};

/** Launch covering ni x nj x nk points (each count >= 1). */
inline PointLaunch pointLaunch(int ni, int nj, int nk) {
  // smallest power of two >= n, at most cap
  auto pow2 = [](int n, int cap) {
    int b = 1;
    while (b < n && b < cap)
      b <<= 1;
    return b;
  };
  const int bk = pow2(nk, kPointRowMax);
  const int bj = pow2(nj, kPointBlockThreads / bk);
  const int biMax = kPointBlockThreads / (bk * bj);
  const int bi = pow2(ni, biMax < kPointBlockZMax ? biMax : kPointBlockZMax);
  PointLaunch L;
  L.block = dim3(bk, bj, bi);
  L.grid = dim3((nk + bk - 1) / bk, (nj + bj - 1) / bj, (ni + bi - 1) / bi);
  return L;
}

#endif // GPU_POINT_LAUNCH_CUH
