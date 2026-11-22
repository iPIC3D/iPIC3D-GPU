#ifndef GPU_CLEANING_CUH
#define GPU_CLEANING_CUH

/**
 * @file GPUCleaning.cuh
 * @brief Operators and fused kernels of the divergence cleanings
 *        (EMfields3D::gpuPoissonCorrection for div(E),
 *        EMfields3D::gpuApplyDivBCleaning for div(B)).
 *
 * Both solve div(grad u) = r with the divergence and gradient of the field
 * solver, so that subtracting grad u removes the divergence to the solver
 * tolerance:
 *   div(E): phi on the cells, divN2C(gradC2N phi) = divN2C(E) - 4 pi rho,
 *           then E -= gradC2N phi on the nodes.  The ghost cells of phi
 *           (GhostSet::Phi, filled after the exchange of every image): the
 *           conductor image on a perfect conductor, odd (phi = 0 on the
 *           wall, so the correction has no tangential part there), 0 on the
 *           other physical faces.
 *   div(B): psi on the nodes, divC2N(gradN2C psi) = divC2N(Bc), then
 *           Bc -= gradN2C psi on the cells.  The nodes of a physical face are
 *           identity rows; the operator reads psi = 0 there on the physical
 *           faces other than perfect conductors (the correction has no
 *           tangential part on the face) and psi of the next inner node on
 *           a perfect conductor (zero normal gradient: B_n in the wall cell,
 *           and so at the wall, is kept).
 * div∘grad is the 27-point operator  sum_a δ_a² μ_b² μ_c²  (δ² the
 * (1, -2, 1)/h² difference, μ² the (1/4, 1/2, 1/4) average); a flat axis (a
 * single cell, periodic onto the rank) drops out.  The vectors are packed
 * rows p = (i nj + j) nk + k, 0 <= i < ni: the points of the array without
 * its ghost layer, one row plane for the two node planes of a twin axis;
 * faces in the order Xleft, Xright, Yleft, Yright, Zleft, Zright.
 */

#include "GPUBlas.cuh"
#include "GPUPointLaunch.cuh"
#include "cudaTypeDef.cuh"

#include <cstdint>
#include <type_traits>
#include <vector>

/** Division by d >= 1 fixed at launch, for 0 <= n < 2^31: n / d =
 *  (umulhi(n, mul) + n) >> shift, the round-up reciprocal 2^(32+shift) / d
 *  with its 33rd bit implicit (Granlund-Montgomery).  Unlike FastDiv
 *  (GPUPointLaunch.cuh), d = 1 needs no select (mul 1, shift 0), and the
 *  sum is one IMAD.HI. */
struct RowDiv {
  int d;
  unsigned mul, shift;
  RowDiv(int divisor = 1) : d(divisor) {
    unsigned l = 0; // ceil(log2(d))
    while ((1u << l) < static_cast<unsigned>(d))
      l++;
    mul = static_cast<unsigned>((std::uint64_t(1) << (32 + l)) / d + 1);
    shift = l;
  }
  __host__ __device__ __forceinline__ int div(int n) const {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
    const unsigned hi = __umulhi(static_cast<unsigned>(n), mul);
#else
    const unsigned hi = static_cast<unsigned>(
        (std::uint64_t(static_cast<unsigned>(n)) * mul) >> 32);
#endif
    return static_cast<int>((hi + static_cast<unsigned>(n)) >> shift);
  }
};

/** The 27-point operator of a cleaning solve on its packed rows. */
struct Lap27Op {
  int ni, nj, nk;
  RowDiv divNk, divNj;       // packed row -> (i, j, k) (lap27RowOf)
  cudaSolverType w[2][2][2]; // weight of the offset (a, b, c) by a, b, c != 0
  cudaSolverType invh2[3];   // 1 / h² per axis
  cudaSolverType diag;       // value of the identity rows (of -L)
  bool flat[3];              // a flat axis: δ² = 0, μ² = 1 (no offset read)
  bool twin[3];    // a flat node axis: its two interior planes are one periodic
                   // node, held by one row plane and unpacked into both
  bool faceRow[6]; // the boundary plane of the rows is a physical face of
                   // identity rows (the nodes of div(B))
  bool faceCopy[6]; // on that plane the operator reads the reading row
                    // (zero normal gradient) instead of 0
};

/** The flat axes of op as a mask, bit a for axis a: the template argument
 *  FLAT of the stencil kernels (the Chebyshev steps, the unpack and the
 *  image), which drops the offsets and weights along the flat axes at
 *  compile time. */
inline int lap27FlatMask(const Lap27Op& op) {
  return op.flat[0] | op.flat[1] << 1 | op.flat[2] << 2;
}

/** f(std::integral_constant<int, FLAT>()) with FLAT = lap27FlatMask(op): the
 *  host side of a launch with the flat axes as a template argument (one
 *  instance per mask, 0..7). */
template <int FLAT = 0, typename F>
void lap27FlatDispatch(const Lap27Op& op, F&& f) {
  if constexpr (FLAT < 8) {
    if (lap27FlatMask(op) == FLAT)
      f(std::integral_constant<int, FLAT>());
    else
      lap27FlatDispatch<FLAT + 1>(op, f);
  }
}

/** The packed-row kernels (here and the Chebyshev steps) run one thread per
 *  row in a 1D launch of kRowBlock threads: in a 3D launch the node rows
 *  (cells + 1 per axis) would leave most of the last block of every k-row
 *  idle. */
constexpr int kRowBlock = kPointBlockThreads;
template <typename Op> inline int rowBlocks(const Op& op) {
  return (op.ni * op.nj * op.nk + kRowBlock - 1) / kRowBlock;
}

/** The operator for nPacked rows per axis (1 along a flat axis, also on the
 *  nodes; a flat axis takes no face rows, aborts otherwise).  The identity
 *  rows take the middle of the spectrum of the other rows (lap27EigBounds). */
Lap27Op lap27Op(const int nPacked[3], const double invh2[3], const bool flat[3],
                const bool faceRow[6], const bool faceCopy[6], bool nodes);

/** Rows of the image as points of the unpacked array (index 1..n-2 along an
 *  axis of extent n with its ghosts): all of them, and the split of the
 *  overlapped exchange, the interior (2..n-3 along every split axis, reads no
 *  ghost) and the shells (the rest).  split is false without overlap, without
 *  a non-flat axis, or with fewer than 3 rows along a non-flat axis. */
struct Lap27Rows {
  PointList all, interior, shells;
  bool split;
};
Lap27Rows lap27Rows(const int n[3], const bool flat[3], bool overlap);

/** Exact extreme eigenvalues of the communication-free -L of the
 *  Chebyshev preconditioner (zero beyond the block, the face rules on the
 *  face planes).  Along every axis μ² = 1 - T, T = -h² δ² / 4, with the same
 *  end rules, so the eigenvalues are
 *    Λ(t) = sum_a 4 invh2_a t_a prod_{b != a} (1 - t_b)
 *  over the eigenvalues t_a of the 1D T (multilinear in t: the extremes are
 *  at the extreme 1D values).  min is the smallest non-zero eigenvalue,
 *  singular says that 0 is one; max = 0 for a zero operator (every axis
 *  flat). */
struct Lap27Eig {
  double min, max;
  bool singular;
};
Lap27Eig lap27EigBounds(const Lap27Op& op);

/** Krylov → array: the packed rows into the points 1..n-2 of the array,
 *  with the face rules on the face planes (0, or the next inner row); the row
 *  plane of a twin axis into both of its node planes.  imageOnly: only the
 *  rows that gpuLap27ToKrylov reads from f or the halo exchange sends, those
 *  within 2 rows (3 beyond a face plane) of the boundary along some split
 *  axis when an axis is flat; all rows in 3D. */
void gpuSolver2PhysLap27(cudaSolverType* f, const cudaSolverType* vec, int ny,
                         int nz, const Lap27Op& op, cudaStream_t stream,
                         bool imageOnly = false);

/** Image of the cleaning solve on the rows of the list: im = L f, from the
 *  unpacked f with valid ghosts along the non-flat axes (the rows of
 *  gpuSolver2PhysLap27 imageOnly suffice), vec the packed vector unpacked
 *  into f; a face row gets -diag * vec.  With a flat axis, the rows whose
 *  stencil reads neither a ghost nor a face plane read vec. */
void gpuLap27ToKrylov(cudaSolverType* im, const cudaSolverType* f,
                      const cudaSolverType* vec, int ny, int nz,
                      const Lap27Op& op, const PointList& rows,
                      cudaStream_t stream);

/** Right-hand side of div(B): b = divC2N(Bc) on the node rows of op (from
 *  the centres with extents (., nyc, nzc)), 0 on the face rows. */
void gpuDivC2NToKrylov(cudaSolverType* b, const cudaSolverType* bx,
                       const cudaSolverType* by, const cudaSolverType* bz,
                       int nyc, int nzc, const cudaSolverType invd[3],
                       const Lap27Op& op, cudaStream_t stream);

/** Right-hand side of div(E): b = divN2C(E) + rhoScale * rho on the cell
 *  rows of op (E on the nodes with extents (., nyn, nzn), rho on the
 *  centres). */
void gpuDivN2CRhoToKrylov(cudaSolverType* b, const cudaSolverType* ex,
                          const cudaSolverType* ey, const cudaSolverType* ez,
                          const cudaSolverType* rho, cudaSolverType rhoScale,
                          int nyn, int nzn, const cudaSolverType invd[3],
                          const Lap27Op& op, cudaStream_t stream);

/** B -= gradN2C psi on the interior centres (1..n-2 of nxc x nyc x nzc);
 *  reads the nodes 1..nxn-2 only. */
void gpuSubGradN2C(cudaSolverType* bx, cudaSolverType* by, cudaSolverType* bz,
                   const cudaSolverType* psi, int nxc, int nyc, int nzc,
                   const cudaSolverType invd[3], cudaStream_t stream);

/** E -= gradC2N phi on the interior nodes (1..n-2 of nxn x nyn x nzn);
 *  reads phi with its ghost cells, which must be valid. */
void gpuSubGradC2N(cudaSolverType* ex, cudaSolverType* ey, cudaSolverType* ez,
                   const cudaSolverType* phi, int nxn, int nyn, int nzn,
                   const cudaSolverType invd[3], cudaStream_t stream);

/**
 * The null space of a cleaning operator, which a right-hand side must not
 * touch (EMfields3D::gpuRemoveCleaningNullSpace): the constants (mean), and
 * the checkerboards (-1)^(i_a + i_b) f(i_c) of two checkerboard axes a, b,
 * for any profile f along the third axis (both gradients map them to 0).  An
 * axis carries the checkerboard (-1)^i when its end rules do: periodic with
 * an even cell count, or odd ghosts on both faces (oddWalls: the cells of
 * div(E) between two perfect conductors), whatever the count.  With three
 * such axes the three families share the triple checkerboard: P = P_ab +
 * P_ac + P_bc - 2 P_abc (projections onto Fourier subsets).
 *
 * A projection takes signed sums over the rows the rank owns (sums[s].box: a
 * row shared with the right neighbour, or its periodic image, belongs to the
 * neighbour), one per plane normal to sums[s].axis (index l along it; or
 * one total, axis -1): local slots local..local+lines of a buffer of
 * localSize; they add up over the ranks in a global buffer of globalSize,
 * plane l at global + first + l.
 * gpuCleaningNullSubtract turns the global sums into the terms subtracted
 * from the rows in box (the rows that are not face rows).
 */
struct CleaningNullSpace {
  struct Sum {
    SignedBox box;
    int axis, lines, local, global, first;
    double count; // unique points of one plane (of all rows for the mean)
  };
  SignedBox box;
  Sum sums[kMaxSignedTerms - 1]; // the mean and three families
  int nSums, localSize, globalSize;
  int N[3];
  bool periodic[3];
};

/** The null space of op (node or cell rows) on a rank whose packed rows
 *  start at the global index offset (N global cells per axis; sharedRight:
 *  the last row along an axis belongs to the right neighbour; oddWalls: the
 *  axes with odd ghosts on both global faces; mean: constants in the null
 *  space).  Every rank passes the same global flags. */
CleaningNullSpace cleaningNullSpace(const Lap27Op& op, bool nodes,
                                    const int N[3], const int offset[3],
                                    const bool periodic[3],
                                    const bool sharedRight[3],
                                    const bool oddWalls[3], bool mean);

/** One pass of the null-space removal on one rank, before the sum over the
 *  ranks: the signed sums of b into the localSize device slots d_sums, copied
 *  to as many page-locked host slots h_sums, then placed in global
 *  (globalSize, zero elsewhere).  Synchronises stream. */
void gpuCleaningNullSums(const cudaSolverType* d_b, const CleaningNullSpace& ns,
                         cudaSolverType* d_sums, cudaSolverType* h_sums,
                         std::vector<cudaSolverType>& global,
                         cudaStream_t stream);

/** After the sum over the ranks: b loses the terms of the global sums
 *  (d_coef receives the line coefficients, at most one per row along each
 *  axis, from as many page-locked host slots h_coef).  Synchronises stream. */
void gpuCleaningNullSubtract(cudaSolverType* d_b, const CleaningNullSpace& ns,
                             const std::vector<cudaSolverType>& global,
                             cudaSolverType* d_coef, cudaSolverType* h_coef,
                             cudaStream_t stream);

#if defined(__CUDACC__) || defined(__HIPCC__)
/** Whether axis a is flat in the mask FLAT (lap27FlatMask). */
__host__ __device__ constexpr bool lap27Flat(int FLAT, int a) {
  return (FLAT >> a) & 1;
}

/** L at a point from v(a, b, c), the value at the offset (a, b, c) in
 *  {-1, 0, 1}³: the values summed by weight class, then weighted; the
 *  offsets along a flat axis of FLAT (weight 0) are neither read nor
 *  weighted.  A class sum starts at its first offset (a, b, c <= 0), not at
 *  0 + that value: the two differ only in the sign of a zero sum, which the
 *  weighted sum (from +0, so never -0) does not see. */
template <int FLAT, typename T, typename V>
__device__ __forceinline__ T lap27(const V& v, const Lap27Op& op) {
  T cls[2][2][2];
#pragma unroll
  for (int a = -1; a <= 1; a++)
#pragma unroll
    for (int b = -1; b <= 1; b++)
#pragma unroll
      for (int c = -1; c <= 1; c++) {
        if ((a && lap27Flat(FLAT, 0)) || (b && lap27Flat(FLAT, 1)) ||
            (c && lap27Flat(FLAT, 2)))
          continue;
        if (a <= 0 && b <= 0 && c <= 0)
          cls[a != 0][b != 0][c != 0] = v(a, b, c);
        else
          cls[a != 0][b != 0][c != 0] += v(a, b, c);
      }
  T s = T(0);
#pragma unroll
  for (int a = 0; a < 2; a++)
#pragma unroll
    for (int b = 0; b < 2; b++)
#pragma unroll
      for (int c = 0; c < 2; c++)
        if (!((a && lap27Flat(FLAT, 0)) || (b && lap27Flat(FLAT, 1)) ||
              (c && lap27Flat(FLAT, 2))))
          s += op.w[a][b][c] * cls[a][b][c];
  return s;
}

/** The packed row p: r = p / nk (its (i, j) line) and x = (i, j, k), with
 *  no division along a flat axis of FLAT (its single row 0). */
template <int FLAT>
__device__ __forceinline__ void lap27RowOf(int p, const RowDiv& nk,
                                           const RowDiv& nj, int& r, int x[3]) {
  if constexpr (lap27Flat(FLAT, 2))
    r = p;
  else if constexpr (lap27Flat(FLAT, 0) && lap27Flat(FLAT, 1))
    r = 0;
  else
    r = nk.div(p);
  if constexpr (lap27Flat(FLAT, 0))
    x[0] = 0;
  else if constexpr (lap27Flat(FLAT, 1))
    x[0] = r;
  else
    x[0] = nj.div(r);
  x[1] = lap27Flat(FLAT, 1) ? 0 : r - x[0] * nj.d;
  x[2] = lap27Flat(FLAT, 2) ? 0 : p - r * nk.d;
}

#endif

#endif // GPU_CLEANING_CUH
