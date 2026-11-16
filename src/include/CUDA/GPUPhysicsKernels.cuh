#ifndef GPU_PHYSICS_KERNELS_CUH
#define GPU_PHYSICS_KERNELS_CUH

/**
 * @file GPUPhysicsKernels.cuh
 * @brief GPU kernel declarations for iPIC3D physics operations:
 *        MUdot, PIdot, and the planet-charge, Laplacian and block-Jacobi
 *        kernels.  The boundary conditions of the fields are in
 *        GPUFieldBoundary.cuh.
 *
 * All host wrapper functions are callable from .cpp translation units
 * (they do not use <<<>>> syntax directly).
 */

#include "GPUPointLaunch.cuh"
#include "cudaTypeDef.cuh"
#include <cstddef>

// =========================================================================
//  MUdot: magnetised susceptibility response  μ·E
// =========================================================================

/** Species per MUdot launch; more species run in several launches. */
inline constexpr int kMUdotMaxSpecies = 16;

/**
 * @brief MUdot = μ·v, the implicit susceptibility of all species applied to v
 * on the interior nodes of points, written to MUdotX/Y/Z (the other nodes
 * untouched).
 *
 * With b = B + B_ext and, per species, beta_s = (q/m)_s dt / (2c) and
 * d_s = (4π/2) delt dt/c (q/m)_s rho_s / (1 + beta_s^2 b^2):
 *   μ·v = Σ_s d_s (v + beta_s v×b + beta_s^2 (v·b) b)
 *       = A v + C1 v×b + C2 (v·b) b,
 *   A = Σ_s d_s,  C1 = Σ_s d_s beta_s,  C2 = Σ_s d_s beta_s^2.
 * One kernel per kMUdotMaxSpecies species reads v, B, B_ext and each rho_s
 * once.
 *
 * @param MUdotX/Y/Z  Device output (node-based [nxn][nyn][nzn]).
 * @param vX/Y/Z      Input vector (node-based).
 * @param Bxn/Byn/Bzn B at nodes, without B_ext (the kernel adds it).
 * @param Bx_ext/By_ext/Bz_ext External B at nodes.
 * @param rhons       Species densities: ns contiguous node slices.
 * @param qom         Host array of the ns charge-to-mass ratios.
 * @param ns          Number of species.
 * @param dt,c,FourPI,delt Time step, light speed, 4π and c θ dt.
 * @param nxn,nyn,nzn Grid dimensions (node-based, including ghosts).
 * @param points      Nodes to compute: a PointBox (3D launch) or a PointList
 *                    (1D launch).
 * @param stream      CUDA stream.
 */
template <typename Points>
void gpuMUdot(cudaSolverType* MUdotX, cudaSolverType* MUdotY,
              cudaSolverType* MUdotZ, const cudaSolverType* vX,
              const cudaSolverType* vY, const cudaSolverType* vZ,
              const cudaSolverType* Bxn, const cudaSolverType* Byn,
              const cudaSolverType* Bzn, const cudaSolverType* Bx_ext,
              const cudaSolverType* By_ext, const cudaSolverType* Bz_ext,
              const cudaSolverType* rhons, const double* qom, int ns, double dt,
              double c, double FourPI, double delt, int nxn, int nyn, int nzn,
              const Points& points, cudaStream_t stream = 0);

/**
 * @brief One species' term of MUdot = Σ_s ( denom_s * rot_s(E) );
 * firstSpecies writes, later species add.  Not used currently: gpuMUdot
 * covers all species in one kernel; kept as the per-species reference.
 *
 * @param MUdotX/Y/Z  Device output (node-based [nxn][nyn][nzn]).
 * @param vX/Y/Z      Input vector (node-based).
 * @param Bxn/Byn/Bzn B at nodes, without B_ext (the kernel adds it).
 * @param Bx_ext/By_ext/Bz_ext External B at nodes.
 * @param rhons_is    Species density at nodes (single species slice).
 * @param beta        = 0.5 * qom[is] * dt / c
 * @param prefactor   = FourPI/2 * delt * dt/c * qom[is]
 * @param nxn,nyn,nzn Grid dimensions (node-based, including ghosts).
 * @param firstSpecies If true, writes output; if false, accumulates (+=).
 * @param stream      CUDA stream.
 */
void gpuMUdotSpecies(cudaSolverType* MUdotX, cudaSolverType* MUdotY,
                     cudaSolverType* MUdotZ, const cudaSolverType* vX,
                     const cudaSolverType* vY, const cudaSolverType* vZ,
                     const cudaSolverType* Bxn, const cudaSolverType* Byn,
                     const cudaSolverType* Bzn, const cudaSolverType* Bx_ext,
                     const cudaSolverType* By_ext, const cudaSolverType* Bz_ext,
                     const cudaSolverType* rhons_is, cudaSolverType beta,
                     cudaSolverType prefactor, int nxn, int nyn, int nzn,
                     bool firstSpecies, cudaStream_t stream = 0);

// =========================================================================
//  PIdot: single-species cyclotron rotation (accumulative +=)
// =========================================================================

/**
 * @brief Compute PIdot for one species (accumulates into output).
 *
 * PIdot_X += rot(v) where rot is the single-species cyclotron rotation
 * with denom = 1/(1 + omc^2).
 *
 * @param PIX/PIY/PIZ Device output (node-based, accumulated).
 * @param vX/Y/Z      Input vector (node-based).
 * @param Bxn/Byn/Bzn B at nodes, without B_ext (the kernel adds it).
 * @param Bx_ext/By_ext/Bz_ext External B at nodes.
 * @param beta         = 0.5 * qom[is] * dt / c
 * @param nxn,nyn,nzn  Grid dimensions.
 * @param stream       CUDA stream.
 */
void gpuPIdotSpecies(cudaSolverType* PIX, cudaSolverType* PIY,
                     cudaSolverType* PIZ, const cudaSolverType* vX,
                     const cudaSolverType* vY, const cudaSolverType* vZ,
                     const cudaSolverType* Bxn, const cudaSolverType* Byn,
                     const cudaSolverType* Bzn, const cudaSolverType* Bx_ext,
                     const cudaSolverType* By_ext, const cudaSolverType* Bz_ext,
                     cudaSolverType beta, int nxn, int nyn, int nzn,
                     cudaStream_t stream = 0);

// =========================================================================
//  ConstantChargePlanet kernels
// =========================================================================

/** Set rhons = val inside sphere of radius R. */
void gpuConstantChargePlanetKernel(cudaSolverType* rhons, cudaSolverType val,
                                   cudaSolverType R, cudaSolverType xc,
                                   cudaSolverType yc, cudaSolverType zc,
                                   cudaSolverType xStart, cudaSolverType yStart,
                                   cudaSolverType zStart, cudaSolverType dx,
                                   cudaSolverType dy, cudaSolverType dz,
                                   int nxn, int nyn, int nzn,
                                   cudaStream_t stream = 0);

/** 2D version: set rhons inside circle in XZ plane. */
void gpuConstantChargePlanet2DKernel(cudaSolverType* rhons, cudaSolverType val,
                                     cudaSolverType R, cudaSolverType xc,
                                     cudaSolverType zc, cudaSolverType xStart,
                                     cudaSolverType zStart, cudaSolverType dx,
                                     cudaSolverType dz, int nxn, int nyn,
                                     int nzn, cudaStream_t stream = 0);

// =========================================================================
//  Center-to-center 7-point Laplacian (not used currently: the cleaning
//  solves use the 27-point operators of GPUCleaning.cuh)
// =========================================================================

/** 7-point center Laplacian of the interior cells of fC (face ghosts set by
 *  the caller), written packed into the Krylov vector lapK, p = (i nj + j) nk
 *  + k over the interior cells (layout of gpuPhys2Solver1).  Used only by
 *  the unit tests and gpuPoissonImage (not used). */
void gpuLapC2CToKrylov(cudaSolverType* lapK, const cudaSolverType* fC, int nxc,
                       int nyc, int nzc, cudaSolverType invdx2,
                       cudaSolverType invdy2, cudaSolverType invdz2,
                       cudaStream_t stream = 0);

/** 7-point center Laplacian on the cell array.  Used only by the unit tests
 *  (not used). */
void gpuLapC2CKernel(cudaSolverType* lapC, const cudaSolverType* fC, int nxc,
                     int nyc, int nzc, cudaSolverType invdx2,
                     cudaSolverType invdy2, cudaSolverType invdz2,
                     cudaStream_t stream = 0);

/** 7-point center Laplacian that never reads the ghost layer.  The value
 *  beyond each face is ghostFactor[f] * (adjacent interior cell), with
 *  f = Xleft, Xright, Yleft, Yright, Zleft, Zright and factor 0 (zero
 *  ghost), +1 (copy) or -1 (antisymmetric).  Used only by the unit tests
 *  and gpuPoissonImageLocal (not used). */
void gpuLapC2CLocalKernel(cudaSolverType* lapC, const cudaSolverType* fC,
                          int nxc, int nyc, int nzc, cudaSolverType invdx2,
                          cudaSolverType invdy2, cudaSolverType invdz2,
                          const int ghostFactor[6], cudaStream_t stream = 0);

// =========================================================================
//  Block-Jacobi preconditioner: solve D_i z_i = r_i per node
// =========================================================================

/**
 * @brief Point-block Jacobi preconditioner for the Maxwell operator.
 *
 * At each interior node, builds the 3×3 diagonal block D_i of the
 * discretised Maxwell operator (including the MUdot tensor summed over
 * all species) and solves D_i z_i = r_i via Cramer's rule.
 *
 * Completely communication-free.  Not used: FGMRESBlockJacobi applies the
 * precomputed inverse (gpuPrecomputeBlockJacobiInv,
 * gpuApplyBlockJacobiInvKrylov).
 *
 * @param zX/zY/zZ   Output solution (node-based).
 * @param rX/rY/rZ   Input RHS (node-based).
 * @param Bxn/Byn/Bzn   B at nodes, without B_ext (the kernel adds it).
 * @param Bx_ext/By_ext/Bz_ext  External B at nodes.
 * @param rhons      Per-species density, flat [ns][nxn*nyn*nzn].
 * @param d_qom      Charge-to-mass ratio per species [ns].
 * @param ns         Number of species.
 * @param dt,c_val,delt,FourPI  Physical constants.
 * @param diagScalar  1 + δ²/2 * cΣ
 * @param wx,wy,wz    1 + δ²/(2 hα²) for α = x,y,z
 * @param nxn,nyn,nzn  Node grid dimensions.
 * @param stream      CUDA stream.
 */
void gpuBlockJacobiPrecondKernel(
    cudaSolverType* zX, cudaSolverType* zY, cudaSolverType* zZ,
    const cudaSolverType* rX, const cudaSolverType* rY,
    const cudaSolverType* rZ, const cudaSolverType* Bxn,
    const cudaSolverType* Byn, const cudaSolverType* Bzn,
    const cudaSolverType* Bx_ext, const cudaSolverType* By_ext,
    const cudaSolverType* Bz_ext, const cudaSolverType* rhons,
    const cudaSolverType* d_qom, int ns, cudaSolverType dt,
    cudaSolverType c_val, cudaSolverType delt, cudaSolverType FourPI,
    cudaSolverType diagScalar, cudaSolverType wx, cudaSolverType wy,
    cudaSolverType wz, int nxn, int nyn, int nzn, cudaStream_t stream = 0);

/**
 * Precompute D^{-1} at each interior node and store 9 entries per node.
 * Layout: Dinv[(row*3+col) * nodeSlice + nodeIdx]
 */
void gpuPrecomputeBlockJacobiInv(
    cudaSolverType* Dinv, const cudaSolverType* Bxn, const cudaSolverType* Byn,
    const cudaSolverType* Bzn, const cudaSolverType* Bx_ext,
    const cudaSolverType* By_ext, const cudaSolverType* Bz_ext,
    const cudaSolverType* rhons, const cudaSolverType* d_qom, int ns,
    cudaSolverType dt, cudaSolverType c_val, cudaSolverType delt,
    cudaSolverType FourPI, cudaSolverType diagScalar, cudaSolverType wx,
    cudaSolverType wy, cudaSolverType wz, int nxn, int nyn, int nzn,
    cudaStream_t stream = 0);

/**
 * Fast D^{-1} application operating directly on Krylov vectors.
 * zKrylov = D^{-1} * rKrylov  (no pack/unpack needed).
 */
void gpuApplyBlockJacobiInvKrylov(cudaSolverType* zKrylov,
                                  const cudaSolverType* rKrylov,
                                  const cudaSolverType* Dinv, int nxn, int nyn,
                                  int nzn, cudaStream_t stream = 0);

#endif // GPU_PHYSICS_KERNELS_CUH
