#ifndef GPU_POINT_LAUNCH_CUH
#define GPU_POINT_LAUNCH_CUH

/**
 * @file GPUPointLaunch.cuh
 * @brief Launch shape of the 3D point kernels (stencils, MUdot, the Maxwell
 *        image kernels): one thread per point of a box of a row-major
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
 * Kernels that take a PointBox (up to two runs of points per axis, e.g. the
 * two boundary layers of a subdomain) recover the point with pointOf instead;
 * a PointList (up to three boxes, e.g. the boundary layers of all three axes)
 * is enumerated by one 1D launch, also through pointOf.
 *
 * Warp width: a k row has at most 32 points.  A 64-lane AMD wavefront then
 * reads two contiguous 256-B runs of doubles, at most one cache line more
 * than a single 64-point row, and avoids the idle lanes a 64-wide row leaves
 * when nk is not a multiple of 64 (96 cells per rank).  The point kernels use
 * no warp operations, so results do not depend on the warp width.
 */

#include "cudaTypeDef.cuh"

#include <cstdint>
#include <stdexcept>

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

/** The points of one axis of a launch box: n points, the first n1 of them
 *  from lo on, the others after a jump of gap points. */
struct PointSpan {
  int lo, n1, gap, n;
  /** The t-th point, 0 <= t < n (a select, no branch). */
  __host__ __device__ int at(int t) const {
    return lo + t + (t >= n1 ? gap : 0);
  }
};

/** The points [a..b]; none when b < a. */
inline PointSpan pointRange(int a, int b) {
  const int n = b >= a ? b - a + 1 : 0;
  return {a, n, 0, n};
}

/** The points [a1..b1] and [a2..b2], with b1 < a2. */
inline PointSpan pointRuns(int a1, int b1, int a2, int b2) {
  const int n1 = b1 - a1 + 1;
  return {a1, n1, a2 - b1 - 1, n1 + b2 - a2 + 1};
}

/** A launch box: the points (i, j, k) of three spans. */
struct PointBox {
  PointSpan i, j, k;
  bool empty() const { return i.n <= 0 || j.n <= 0 || k.n <= 0; }
};

/** Launch covering a non-empty box. */
inline PointLaunch pointLaunch(const PointBox& b) {
  return pointLaunch(b.i.n, b.j.n, b.k.n);
}

/** Division by a positive int d with a precomputed reciprocal: for
 *  0 <= n < 2^31, n / d = umulhi(n, mul) >> shift (d = 1: n itself). */
struct FastDiv {
  int d;
  unsigned mul, shift;
  FastDiv(int divisor = 1) : d(divisor), mul(0), shift(0) {
    if (d > 1) {
      int log2 = 0; // ceil(log2(d))
      while ((1u << log2) < static_cast<unsigned>(d))
        log2++;
      const unsigned p = 31 + log2;
      mul = static_cast<unsigned>(((std::uint64_t(1) << p) + d - 1) / d);
      shift = p - 32;
    }
  }
  /** n / d from the reciprocal (mul, shift) of d. */
  __host__ __device__ static int divide(int n, int d, unsigned mul,
                                        unsigned shift) {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
    const unsigned hi = __umulhi(static_cast<unsigned>(n), mul);
#else
    const unsigned hi = static_cast<unsigned>(
        (std::uint64_t(static_cast<unsigned>(n)) * mul) >> 32);
#endif
    return d > 1 ? static_cast<int>(hi >> shift) : n;
  }
  __host__ __device__ int div(int n) const { return divide(n, d, mul, shift); }
};

/** The points of up to three boxes, numbered one box after the other and,
 *  inside a box, k fastest, then j, then i: one 1D launch covers them all. */
struct PointList {
  static constexpr int kMaxBoxes = 3;
  PointBox box[kMaxBoxes];
  int first[kMaxBoxes + 1]; // box b holds the points [first[b], first[b+1])
  FastDiv nk[kMaxBoxes], nj[kMaxBoxes]; // point counts along k and j
  int count() const { return first[kMaxBoxes]; }
  bool empty() const { return count() == 0; }
  /** The p-th point, 0 <= p < count(): the box from two comparisons (an
   *  empty box takes no p), the point from two divisions, no branch.  The
   *  fields of the box are selected among values read for all three boxes,
   *  never indexed by the box: a per-thread index into a kernel argument
   *  can make a compiler copy the argument to per-thread memory. */
  __host__ __device__ void at(int p, int& i, int& j, int& k) const {
    const bool in1 = p >= first[1], in2 = p >= first[2];
    const auto pick = [in1, in2](auto x0, auto x1, auto x2) {
      return in2 ? x2 : in1 ? x1 : x0;
    };
    const auto span = [&pick](const PointSpan& s0, const PointSpan& s1,
                              const PointSpan& s2, int t) {
      return pick(s0.lo, s1.lo, s2.lo) + t +
             (t >= pick(s0.n1, s1.n1, s2.n1) ? pick(s0.gap, s1.gap, s2.gap)
                                             : 0);
    };
    const int q = p - pick(0, first[1], first[2]); // first[0] = 0
    const int dk = pick(nk[0].d, nk[1].d, nk[2].d);
    const int dj = pick(nj[0].d, nj[1].d, nj[2].d);
    const int r = FastDiv::divide(q, dk, pick(nk[0].mul, nk[1].mul, nk[2].mul),
                                  pick(nk[0].shift, nk[1].shift, nk[2].shift));
    const int ti = FastDiv::divide(r, dj, pick(nj[0].mul, nj[1].mul, nj[2].mul),
                                   pick(nj[0].shift, nj[1].shift, nj[2].shift));
    i = span(box[0].i, box[1].i, box[2].i, ti);
    j = span(box[0].j, box[1].j, box[2].j, r - ti * dj);
    k = span(box[0].k, box[1].k, box[2].k, q - r * dk);
  }
};

/** The list of the n <= 3 boxes, empty ones left out. */
inline PointList pointList(const PointBox* boxes, int n) {
  if (n < 0 || n > PointList::kMaxBoxes)
    throw std::invalid_argument("pointList takes at most 3 boxes");
  PointList L{};
  int m = 0;
  for (int s = 0; s < n; s++) {
    const PointBox& b = boxes[s];
    if (b.empty())
      continue;
    L.box[m] = b;
    L.nk[m] = FastDiv(b.k.n);
    L.nj[m] = FastDiv(b.j.n);
    L.first[m + 1] = L.first[m] + b.i.n * b.j.n * b.k.n;
    m++;
  }
  for (; m < PointList::kMaxBoxes; m++)
    L.first[m + 1] = L.first[m];
  return L;
}

/** 1D launch covering a non-empty list. */
inline PointLaunch pointLaunch(const PointList& L) {
  PointLaunch launch;
  launch.block = dim3(kPointBlockThreads);
  launch.grid = dim3((L.count() + kPointBlockThreads - 1) / kPointBlockThreads);
  return launch;
}

#if defined(__CUDACC__) || defined(__HIPCC__)
/** The point (i, j, k) of this thread in box b (3D launch); false outside. */
__device__ __forceinline__ bool pointOf(const PointBox& b, int& i, int& j,
                                        int& k) {
  const int ti = blockIdx.z * blockDim.z + threadIdx.z;
  const int tj = blockIdx.y * blockDim.y + threadIdx.y;
  const int tk = blockIdx.x * blockDim.x + threadIdx.x;
  if (ti >= b.i.n || tj >= b.j.n || tk >= b.k.n)
    return false;
  i = b.i.at(ti);
  j = b.j.at(tj);
  k = b.k.at(tk);
  return true;
}

/** The point (i, j, k) of this thread in list L (1D launch); false past it. */
__device__ __forceinline__ bool pointOf(const PointList& L, int& i, int& j,
                                        int& k) {
  const int p = blockIdx.x * blockDim.x + threadIdx.x;
  if (p >= L.first[PointList::kMaxBoxes])
    return false;
  L.at(p, i, j, k);
  return true;
}
#endif

#endif // GPU_POINT_LAUNCH_CUH
