// =========================================================================
//  GPUChebyshev.cu
//
//  CUDA kernels for the Chebyshev semi-iterative solver.  The generic step
//  kernels operate on flat Krylov vectors (the Maxwell one interleaves
//  Ex,Ey,Ez at interior nodes); the Poisson step kernels evaluate the
//  communication-free operator of the cleaning solves (the 27-point
//  -div∘grad, Lap27Op; the 7-point PoissonLocalOperator is not used) on the
//  packed rows in the same pass, and gpuChebyshevCleaning chains them.
// =========================================================================

#include "GPUChebyshev.cuh"
#include "GPUCleaning.cuh"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <type_traits>
#include <utility>

// =========================================================================
//  Step 1 kernel:  z = b/theta,  y = coeff_b*b + coeff_Ab*Ab
// =========================================================================
template <typename T>
__global__ void k_chebyshevStep1(T* __restrict__ d_y, T* __restrict__ d_z,
                                 const T* __restrict__ d_b,
                                 const T* __restrict__ d_Ab, T invTheta,
                                 T coeff_b, T coeff_Ab, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n)
    return;
  T bv = d_b[idx];
  T abv = d_Ab[idx];
  d_z[idx] = bv * invTheta;
  d_y[idx] = coeff_b * bv + coeff_Ab * abv;
}

void gpuChebyshevStep1(cudaSolverType* d_y, cudaSolverType* d_z,
                       const cudaSolverType* d_b, const cudaSolverType* d_Ab,
                       cudaSolverType theta, cudaSolverType delta,
                       cudaSolverType rho, int n, cudaStream_t stream) {
  constexpr int BLK = 256;
  int nblk = (n + BLK - 1) / BLK;
  cudaSolverType invTheta = 1.0 / theta;
  cudaSolverType coeff_b = 4.0 * rho / delta; // 2*rho/delta * 2
  cudaSolverType coeff_Ab =
      -2.0 * rho / (delta * theta); // 2*rho/delta * (-1/theta)
  k_chebyshevStep1<<<nblk, BLK, 0, stream>>>(d_y, d_z, d_b, d_Ab, invTheta,
                                             coeff_b, coeff_Ab, n);
  cudaErrChk(cudaGetLastError());
}

// =========================================================================
//  Step N kernel:  w = fY*y + fB*b + fAy*Ay + fZ*z
// =========================================================================
template <typename T>
__global__ void
k_chebyshevStepN(T* __restrict__ d_w, const T* __restrict__ d_y,
                 const T* __restrict__ d_z, const T* __restrict__ d_b,
                 const T* __restrict__ d_Ay, T fY, T fB, T fAy, T fZ, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n)
    return;
  d_w[idx] = fY * d_y[idx] + fB * d_b[idx] + fAy * d_Ay[idx] + fZ * d_z[idx];
}

void gpuChebyshevStepN(cudaSolverType* d_w, const cudaSolverType* d_y,
                       const cudaSolverType* d_z, const cudaSolverType* d_b,
                       const cudaSolverType* d_Ay, cudaSolverType delta,
                       cudaSolverType sigma, cudaSolverType rho,
                       cudaSolverType rhoOld, int n, cudaStream_t stream) {
  constexpr int BLK = 256;
  int nblk = (n + BLK - 1) / BLK;
  cudaSolverType fY = rho * 2.0 * sigma;
  cudaSolverType fB = rho * 2.0 / delta;
  cudaSolverType fAy = -rho * 2.0 / delta; // from  b - A(y)
  cudaSolverType fZ = -rho * rhoOld;
  k_chebyshevStepN<<<nblk, BLK, 0, stream>>>(d_w, d_y, d_z, d_b, d_Ay, fY, fB,
                                             fAy, fZ, n);
  cudaErrChk(cudaGetLastError());
}

// =========================================================================
//  Steps of the cleaning preconditioner: A = the communication-free local
//  operator, evaluated in the step kernel on the packed rows: one kernel per
//  step, no unpacked array, no ghost layer, one thread per row (1D launch of
//  kRowBlock threads).
// =========================================================================

/** -∇² y at the packed cell p = (i nj + j) nk + k (7-point, not used
 *  currently).  Branch-free: every thread loads the six neighbours; outside
 *  the interior the neighbour index is the cell itself and its weight the
 *  face factor g, inside the weight is 1. */
template <typename T>
__device__ __forceinline__ T negLapLocal(const T* __restrict__ y, int i, int j,
                                         int k, int p,
                                         const PoissonLocalOperator& op) {
  const int sx = op.nj * op.nk, sy = op.nk;
  const bool inXm = i > 0, inXp = i < op.ni - 1;
  const bool inYm = j > 0, inYp = j < op.nj - 1;
  const bool inZm = k > 0, inZp = k < op.nk - 1;
  const T f = y[p];
  const T xm = (inXm ? T(1) : op.g[0]) * y[inXm ? p - sx : p];
  const T xp = (inXp ? T(1) : op.g[1]) * y[inXp ? p + sx : p];
  const T ym = (inYm ? T(1) : op.g[2]) * y[inYm ? p - sy : p];
  const T yp = (inYp ? T(1) : op.g[3]) * y[inYp ? p + sy : p];
  const T zm = (inZm ? T(1) : op.g[4]) * y[inZm ? p - 1 : p];
  const T zp = (inZp ? T(1) : op.g[5]) * y[inZp ? p + 1 : p];
  return -((xm - T(2) * f + xp) * op.invdx2 + (ym - T(2) * f + yp) * op.invdy2 +
           (zm - T(2) * f + zp) * op.invdz2);
}

/** The packed row p of PoissonLocalOperator as (i, j, k) (not used
 *  currently, unit test only). */
__device__ __forceinline__ void rowOf(const PoissonLocalOperator& op, int p,
                                      int& i, int& j, int& k) {
  k = p % op.nk;
  j = (p / op.nk) % op.nj;
  i = p / (op.nk * op.nj);
}

// ---- Lap27Op: the step kernels of the cleaning solves ----------------------

/** Per-launch constants of the Lap27Op step kernels, derived on the host
 *  (lap27Step) so that the kernels evaluate no face rule: per axis a, the
 *  row index whose - / + neighbour reads 0 (zLo, zHi: the block edge, or the
 *  row next to a zero face plane), reads the row itself (cLo, cHi: next to a
 *  copy face plane), and the face planes of identity rows (fLo, fHi); -1 for
 *  none.  A flat axis has no face rows (lap27Op). */
struct Lap27Step {
  int n;         // rows
  RowDiv nk, nj; // op.divNk, op.divNj
  int sx, sy;    // row strides along i, j (sy = nk.d: reading nk.d
                 // instead adds instructions to the step kernels)
  int zLo[3], zHi[3];
  int cLo[3], cHi[3];
  int fLo[3], fHi[3];
  cudaSolverType w[2][2][2]; // op.w
  cudaSolverType negDiag;    // L on a face row: -op.diag
  int negDiagHi;             // its high word
};

/** The template argument FACES of the step kernels: the block has no face
 *  rows (an interior rank), face rows reading 0 beyond, or also copy faces. */
enum { kNoFaceRows = 0, kZeroFaceRows = 1, kCopyFaceRows = 2 };

static int lap27Faces(const Lap27Op& op) {
  bool row = false, copy = false;
  for (int f = 0; f < 6; f++) {
    row = row || op.faceRow[f];
    copy = copy || op.faceCopy[f];
  }
  return copy ? kCopyFaceRows : row ? kZeroFaceRows : kNoFaceRows;
}

static Lap27Step lap27Step(const Lap27Op& op) {
  Lap27Step s{};
  const int n[3] = {op.ni, op.nj, op.nk};
  s.n = op.ni * op.nj * op.nk;
  s.nk = op.divNk;
  s.nj = op.divNj;
  s.sx = op.nj * op.nk;
  s.sy = op.nk;
  for (int a = 0; a < 3; a++) {
    const int last = n[a] - 1;
    const bool rl = op.faceRow[2 * a], rh = op.faceRow[2 * a + 1];
    const bool cl = op.faceCopy[2 * a], ch = op.faceCopy[2 * a + 1];
    s.fLo[a] = rl ? 0 : -1;
    s.fHi[a] = rh ? last : -1;
    s.zLo[a] = !rl ? 0 : cl ? -1 : 1;
    s.zHi[a] = !rh ? last : ch ? -1 : last - 1;
    s.cLo[a] = rl && cl ? 1 : -1;
    s.cHi[a] = rh && ch ? last - 1 : -1;
  }
  std::memcpy(s.w, op.w, sizeof s.w);
  s.negDiag = -op.diag;
  long long bits;
  std::memcpy(&bits, &s.negDiag, sizeof bits);
  s.negDiagHi = (int)(bits >> 32);
  return s;
}

/** L y at the row p = x of the communication-free operator: a neighbour
 *  beyond the block reads 0 (block-Jacobi cut, or the zero ghost of a cell
 *  face), one on a face plane 0 or the row itself (faceCopy); a face row
 *  gives -diag y.  yp = y[p], FLAT = lap27FlatMask(op), FACES =
 *  lap27Faces(op).  Bitwise the sums of lap27 (GPUCleaning.cuh), in its
 *  order:
 *  - a neighbour that reads 0 is not loaded and, after the first offset of
 *    its class, not added (c + 0 differs from c only for c = -0, a sign the
 *    weighted sum does not see, as in lap27);
 *  - the k neighbours are immediate offsets of the (i, j) column address,
 *    except next to a copy face (FACES = kCopyFaceRows);
 *  - a face row takes the path of the other rows, no branch: every
 *    neighbour reads 0 and the row itself has weight -diag, so the sum is
 *    fma(-diag, y, +0) = -diag y but for the sign of a zero, set from the
 *    sign bits. */
template <int FLAT, int FACES, typename T>
__device__ __forceinline__ T lap27StepL(const T* __restrict__ y, T yp, int p,
                                        const int x[3], const Lap27Step& s) {
  constexpr bool fl[3] = {lap27Flat(FLAT, 0), lap27Flat(FLAT, 1),
                          lap27Flat(FLAT, 2)};
  bool face = false;
  if constexpr (FACES != kNoFaceRows) {
#pragma unroll
    for (int a = 0; a < 3; a++)
      if (!fl[a])
        face = face | (x[a] == s.fLo[a]) | (x[a] == s.fHi[a]);
  }
  bool zL[3] = {}, zH[3] = {}; // the - / + neighbour along a reads 0
  int oL[3] = {}, oH[3] = {};  // its offset
  const int stride[3] = {s.sx, s.sy, 1};
#pragma unroll
  for (int a = 0; a < 3; a++) {
    if (fl[a])
      continue;
    if constexpr (FACES == kNoFaceRows) {
      zL[a] = x[a] == 0;
      zH[a] = x[a] == s.zHi[a];
    } else {
      zL[a] = face | (x[a] == s.zLo[a]);
      zH[a] = face | (x[a] == s.zHi[a]);
    }
    if constexpr (FACES == kCopyFaceRows) {
      oL[a] = x[a] == s.cLo[a] ? 0 : -stride[a];
      oH[a] = x[a] == s.cHi[a] ? 0 : stride[a];
    } else {
      oL[a] = -stride[a];
      oH[a] = stride[a];
    }
  }
  const auto off = [&](int a, int d) {
    return d < 0 ? oL[a] : d > 0 ? oH[a] : 0;
  };
  const auto v = [&](int a, int b, int c) -> T {
    if constexpr (FACES != kCopyFaceRows)
      return (y + (p + off(0, a) + off(1, b)))[c];
    else
      return y[p + off(0, a) + off(1, b) + off(2, c)];
  };
  T cls[2][2][2];
#pragma unroll
  for (int a = -1; a <= 1; a++)
#pragma unroll
    for (int b = -1; b <= 1; b++)
#pragma unroll
      for (int c = -1; c <= 1; c++) {
        if ((a && fl[0]) || (b && fl[1]) || (c && fl[2]))
          continue;
        T& cl = cls[a != 0][b != 0][c != 0];
        const bool zero = (a < 0 && zL[0]) || (a > 0 && zH[0]) ||
                          (b < 0 && zL[1]) || (b > 0 && zH[1]) ||
                          (c < 0 && zL[2]) || (c > 0 && zH[2]);
        if (!a && !b && !c)
          cl = yp;
        else if (a <= 0 && b <= 0 && c <= 0)
          cl = zero ? T(0) : v(a, b, c);
        else if (!zero)
          cl = __dadd_rn(cl, v(a, b, c));
      }
  T w0 = s.w[0][0][0];
  if constexpr (FACES != kNoFaceRows)
    w0 = face ? s.negDiag : w0;
  T sum = __fma_rn(w0, cls[0][0][0], T(0));
#pragma unroll
  for (int a = 0; a < 2; a++)
#pragma unroll
    for (int b = 0; b < 2; b++)
#pragma unroll
      for (int c = 0; c < 2; c++)
        if ((a || b || c) && !((a && fl[0]) || (b && fl[1]) || (c && fl[2])))
          sum = __fma_rn(s.w[a][b][c], cls[a][b][c], sum);
  if constexpr (FACES != kNoFaceRows) {
    const int hi = __double2hiint(yp), lo = __double2loint(yp);
    const int sign = (hi ^ s.negDiagHi) & int(0x80000000u);
    if (face && ((hi & 0x7fffffff) | lo) == 0)
      sum = __hiloint2double(__double2hiint(sum) | sign, __double2loint(sum));
  }
  return sum;
}

/** v times outScale = +-1: the sign bit of v flipped by neg (0 or
 *  0x80000000). */
template <typename T> __device__ __forceinline__ T lap27StepSign(T v, int neg) {
  return __hiloint2double(__double2hiint(v) ^ neg, __double2loint(v));
}

// The intrinsics fix the roundings on every compiler: step 1 fuses coeffB b
// and rounds the operator term, step N rounds fB b, then fuses fY y, the
// operator term and fZ z.  On HIP __dmul_rn and __dadd_rn are a plain * and
// +, which the compiler may contract with an adjacent + or * into an FMA;
// here every __dmul_rn feeds an __fma_rn or a store and every __dadd_rn adds
// a loaded value to a class sum, so each stays rounded on its own.
// coeffL = -coeffAb and fL = -fAy are the coefficients of L y.
template <int FLAT, int FACES, typename T>
__global__ void k_chebyshevLap27Step1(T* __restrict__ out, T* __restrict__ z,
                                      const T* __restrict__ b, Lap27Step s,
                                      T invTheta, T coeffB, T coeffL, int neg) {
  const int p = blockIdx.x * blockDim.x + threadIdx.x;
  if (p >= s.n)
    return;
  int r, x[3];
  lap27RowOf<FLAT>(p, s.nk, s.nj, r, x);
  const unsigned q = p;
  const T bv = b[q];
  z[q] = __dmul_rn(bv, invTheta);
  const T l = lap27StepL<FLAT, FACES>(b, bv, p, x, s);
  out[q] = lap27StepSign(__fma_rn(coeffB, bv, __dmul_rn(coeffL, l)), neg);
}

// out and z may be the same array (in-place update), so neither is
// __restrict__; y and b never alias them.
template <int FLAT, int FACES, typename T>
__global__ void k_chebyshevLap27StepN(T* out, const T* __restrict__ y,
                                      const T* z, const T* __restrict__ b,
                                      Lap27Step s, T fY, T fB, T fL, T fZ,
                                      int neg) {
  const int p = blockIdx.x * blockDim.x + threadIdx.x;
  if (p >= s.n)
    return;
  int r, x[3];
  lap27RowOf<FLAT>(p, s.nk, s.nj, r, x);
  const unsigned q = p;
  const T yp = y[q];
  const T l = lap27StepL<FLAT, FACES>(y, yp, p, x, s);
  const T t = __fma_rn(fY, yp, __dmul_rn(fB, b[q]));
  out[q] = lap27StepSign(__fma_rn(fZ, z[q], __fma_rn(fL, l, t)), neg);
}

/** f(flat, faces) with the integral constants FLAT = lap27FlatMask(op) and
 *  FACES = lap27Faces(op): 22 instances (all axes flat: no face rows). */
template <typename F> void lap27StepDispatch(const Lap27Op& op, F&& f) {
  const int faces = lap27Faces(op);
  lap27FlatDispatch(op, [&](auto flat) {
    using Faces0 = std::integral_constant<int, kNoFaceRows>;
    if constexpr (decltype(flat)::value == 7)
      f(flat, Faces0());
    else if (faces == kNoFaceRows)
      f(flat, Faces0());
    else if (faces == kZeroFaceRows)
      f(flat, std::integral_constant<int, kZeroFaceRows>());
    else
      f(flat, std::integral_constant<int, kCopyFaceRows>());
  });
}

/** The sign flip of outScale, which must be +1 or -1. */
static int lap27StepNeg(cudaSolverType outScale) {
  if (outScale != 1.0 && outScale != -1.0) {
    fprintf(stderr, "Chebyshev cleaning step: outScale %g is not +-1\n",
            outScale);
    abort();
  }
  return std::signbit(outScale) ? int(0x80000000u) : 0;
}

// ---- PoissonLocalOperator (not used currently, unit test only) -----------

template <typename T, typename Op>
__global__ void k_chebyshevPoissonStep1(T* __restrict__ out, T* __restrict__ z,
                                        const T* __restrict__ b, Op op,
                                        T invTheta, T coeffB, T coeffAb,
                                        T outScale) {
  const int p = blockIdx.x * blockDim.x + threadIdx.x;
  if (p >= op.ni * op.nj * op.nk)
    return;
  int i, j, k;
  rowOf(op, p, i, j, k);
  const T bv = b[p];
  z[p] = bv * invTheta;
  out[p] = outScale * (coeffB * bv + coeffAb * negLapLocal(b, i, j, k, p, op));
}

// out and z may be the same array (in-place update), so neither is
// __restrict__; y and b never alias them.
template <typename T, typename Op>
__global__ void k_chebyshevPoissonStepN(T* out, const T* __restrict__ y,
                                        const T* z, const T* __restrict__ b,
                                        Op op, T fY, T fB, T fAy, T fZ,
                                        T outScale) {
  const int p = blockIdx.x * blockDim.x + threadIdx.x;
  if (p >= op.ni * op.nj * op.nk)
    return;
  int i, j, k;
  rowOf(op, p, i, j, k);
  out[p] = outScale * (fY * y[p] + fB * b[p] +
                       fAy * negLapLocal(y, i, j, k, p, op) + fZ * z[p]);
}

template <typename Op>
void gpuChebyshevPoissonStep1(cudaSolverType* out, cudaSolverType* z,
                              const cudaSolverType* b, const Op& op,
                              cudaSolverType theta, cudaSolverType delta,
                              cudaSolverType rho, cudaSolverType outScale,
                              cudaStream_t stream) {
  if constexpr (std::is_same_v<Op, Lap27Op>) {
    const Lap27Step s = lap27Step(op);
    const int neg = lap27StepNeg(outScale);
    lap27StepDispatch(op, [&](auto flat, auto faces) {
      k_chebyshevLap27Step1<decltype(flat)::value, decltype(faces)::value>
          <<<rowBlocks(op), kRowBlock, 0, stream>>>(
              out, z, b, s, 1.0 / theta, 4.0 * rho / delta,
              2.0 * rho / (delta * theta), neg);
    });
  } else {
    k_chebyshevPoissonStep1<<<rowBlocks(op), kRowBlock, 0, stream>>>(
        out, z, b, op, 1.0 / theta, 4.0 * rho / delta,
        -2.0 * rho / (delta * theta), outScale);
  }
  cudaErrChk(cudaGetLastError());
}

template <typename Op>
void gpuChebyshevPoissonStepN(cudaSolverType* out, const cudaSolverType* y,
                              const cudaSolverType* z, const cudaSolverType* b,
                              const Op& op, cudaSolverType delta,
                              cudaSolverType sigma, cudaSolverType rho,
                              cudaSolverType rhoOld, cudaSolverType outScale,
                              cudaStream_t stream) {
  if constexpr (std::is_same_v<Op, Lap27Op>) {
    const Lap27Step s = lap27Step(op);
    const int neg = lap27StepNeg(outScale);
    lap27StepDispatch(op, [&](auto flat, auto faces) {
      k_chebyshevLap27StepN<decltype(flat)::value, decltype(faces)::value>
          <<<rowBlocks(op), kRowBlock, 0, stream>>>(
              out, y, z, b, s, rho * 2.0 * sigma, rho * 2.0 / delta,
              rho * 2.0 / delta, -rho * rhoOld, neg);
    });
  } else {
    k_chebyshevPoissonStepN<<<rowBlocks(op), kRowBlock, 0, stream>>>(
        out, y, z, b, op, rho * 2.0 * sigma, rho * 2.0 / delta,
        -rho * 2.0 / delta, -rho * rhoOld, outScale);
  }
  cudaErrChk(cudaGetLastError());
}

#define INSTANTIATE_POISSON_STEPS(Op)                                          \
  template void gpuChebyshevPoissonStep1<Op>(                                  \
      cudaSolverType*, cudaSolverType*, const cudaSolverType*, const Op&,      \
      cudaSolverType, cudaSolverType, cudaSolverType, cudaSolverType,          \
      cudaStream_t);                                                           \
  template void gpuChebyshevPoissonStepN<Op>(                                  \
      cudaSolverType*, const cudaSolverType*, const cudaSolverType*,           \
      const cudaSolverType*, const Op&, cudaSolverType, cudaSolverType,        \
      cudaSolverType, cudaSolverType, cudaSolverType, cudaStream_t);
INSTANTIATE_POISSON_STEPS(Lap27Op)
INSTANTIATE_POISSON_STEPS(PoissonLocalOperator) // not used (unit test only)
#undef INSTANTIATE_POISSON_STEPS

void gpuChebyshevCleaning(cudaSolverType* x, const cudaSolverType* b,
                          const Lap27Op& op, double eigMin, double eigMax,
                          int steps, cudaSolverType* y, cudaSolverType* z,
                          cudaStream_t stream) {
  const int n = op.ni * op.nj * op.nk;
  if (steps <= 0) {
    gpuEq(x, b, n, stream);
    return;
  }
  const double theta = 0.5 * (eigMin + eigMax), delta = 0.5 * (eigMax - eigMin);
  if (!(eigMax > 0.0)) {
    gpuScaleCopy(x, b, -1.0, n, stream);
    return;
  }
  if (delta <= 1e-12 * theta) {
    gpuScaleCopy(x, b, -1.0 / theta, n, stream);
    return;
  }
  const double sigma = theta / delta;
  double rhoOld = 1.0 / sigma;
  double rho = 1.0 / (2.0 * sigma - rhoOld);
  // x₀ = 0, so r₀ = b.  Steps 0 and 1: z = b/θ, y = f(b, (-L) b)
  gpuChebyshevPoissonStep1(steps == 1 ? x : y, z, b, op, theta, delta, rho,
                           steps == 1 ? -1.0 : 1.0, stream);
  // Steps 2..steps: z ← f(y, z, b, (-L) y) in place, then swap y and z
  for (int step = 2; step <= steps; step++) {
    rhoOld = rho;
    rho = 1.0 / (2.0 * sigma - rhoOld);
    const bool last = step == steps;
    gpuChebyshevPoissonStepN(last ? x : z, y, z, b, op, delta, sigma, rho,
                             rhoOld, last ? -1.0 : 1.0, stream);
    std::swap(y, z);
  }
}
