/**
 * @file GPUPhysicsKernels.cuh
 * @brief GPU kernel declarations for iPIC3D physics operations:
 *        MUdot, PIdot, perfect-conductor rows (implicit susceptibility at
 *        the face), and the open-boundary, outflow, case-B-fix,
 *        planet-charge, Laplacian and block-Jacobi kernels.
 *
 * All host wrapper functions are callable from .cpp translation units
 * (they do not use <<<>>> syntax directly).
 */

#ifndef GPU_PHYSICS_KERNELS_CUH
#define GPU_PHYSICS_KERNELS_CUH

#include "OutflowBoundary.h"
#include "cudaTypeDef.cuh"
#include <cstddef>

// =========================================================================
//  MUdot: magnetised susceptibility response  μ·E
//  One species per call; the caller (gpuMUdot) loops over species.
// =========================================================================

/**
 * @brief One species' term of MUdot = Σ_s ( denom_s * rot_s(E) );
 * firstSpecies writes, later species add.
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
//  Perfect conductor boundary conditions (fused susceptibility + rows)
// =========================================================================

/**
 * @brief Image rows of the Maxwell operator on a perfect-conductor face.
 *
 * Face node plane: index 1 (high = false) or n-2 (high = true) along dir
 * (0 X, 1 Y, 2 Z), tangential interior.  Tangential components: identity.
 * Normal component n: Ampere's law without curl B, chi_n . E = E^n_n - 4 pi
 * theta dt Jhat_n, written linearly in the Krylov vector:
 *   image_n = vect_n + (chi_na vect_a + chi_nb vect_b) / chi_nn
 * (chi: implicit susceptibility at the node).  The rest of the equation is in
 * the RHS row, gpuPerfectConductorSource.
 *
 * @param rhons  All-species density [ns][nxn][nyn][nzn] (contiguous).
 * @param d_qom  Device array of the species q/m (ns elements).
 */
void gpuPerfectConductorImage(
    cudaSolverType* imageX, cudaSolverType* imageY, cudaSolverType* imageZ,
    const cudaSolverType* vectX, const cudaSolverType* vectY,
    const cudaSolverType* vectZ, const cudaSolverType* Bxn,
    const cudaSolverType* Byn, const cudaSolverType* Bzn,
    const cudaSolverType* Bx_ext, const cudaSolverType* By_ext,
    const cudaSolverType* Bz_ext, const cudaSolverType* rhons,
    const cudaSolverType* d_qom, int ns, cudaSolverType dt, cudaSolverType c,
    cudaSolverType FourPI, cudaSolverType delt, int nxn, int nyn, int nzn,
    int dir, bool high, cudaStream_t stream = 0);

/**
 * @brief RHS rows of the Maxwell system on a perfect-conductor face (same
 * plane as gpuPerfectConductorImage): tangential components ebc = -(u0 x B0),
 * normal component (E^n_n - 4 pi theta dt Jhat_n) / chi_nn, with En / Jn the
 * normal components of E^n and Jhat.
 */
void gpuPerfectConductorSource(
    cudaSolverType* vectorX, cudaSolverType* vectorY, cudaSolverType* vectorZ,
    const cudaSolverType* En, const cudaSolverType* Jn,
    const cudaSolverType* Bxn, const cudaSolverType* Byn,
    const cudaSolverType* Bzn, const cudaSolverType* Bx_ext,
    const cudaSolverType* By_ext, const cudaSolverType* Bz_ext,
    const cudaSolverType* rhons, const cudaSolverType* d_qom, int ns,
    cudaSolverType dt, cudaSolverType c, cudaSolverType th,
    cudaSolverType FourPI, cudaSolverType delt, cudaSolverType ebc0,
    cudaSolverType ebc1, cudaSolverType ebc2, int nxn, int nyn, int nzn,
    int dir, bool high, cudaStream_t stream = 0);

// =========================================================================
//  Non-periodic particle faces: doubles boundary-face values (10 moment
//  arrays per species).
// =========================================================================

/**
 * @brief Double the node moments on the non-periodic particle faces.
 *
 * For each non-periodic boundary face (where the neighbour is MPI_PROC_NULL),
 * doubles the values on the face plane (index 1 or n-2) of every array:
 * 10 per species (rhons, Jxs, Jys, Jzs, pXXsn, pXYsn, pXZsn, pYYsn, pYZsn,
 * pZZsn), several species per call.
 *
 * @param nptrs      Number of arrays.
 * @param d_devPtrs  Device array of nptrs pointers.
 * @param nxn,nyn,nzn  Grid dimensions (node-based, including ghosts).
 * @param xLeftNull  true if X-left neighbour is MPI_PROC_NULL.
 * @param xRightNull true if X-right neighbour is MPI_PROC_NULL.
 * @param yLeftNull  true if Y-left neighbour is MPI_PROC_NULL.
 * @param yRightNull true if Y-right neighbour is MPI_PROC_NULL.
 * @param zLeftNull  true if Z-left neighbour is MPI_PROC_NULL.
 * @param zRightNull true if Z-right neighbour is MPI_PROC_NULL.
 * @param stream     CUDA stream.
 */
void gpuAdjustNonPeriodicDensities(int nptrs, cudaSolverType* const* d_devPtrs,
                                   int nxn, int nyn, int nzn, bool xLeftNull,
                                   bool xRightNull, bool yLeftNull,
                                   bool yRightNull, bool zLeftNull,
                                   bool zRightNull, cudaStream_t stream = 0);

// =========================================================================
//  Open boundary face/layer kernels
// =========================================================================

/** Set 3 arrays to (cx, cy, cz) on a face plane interior [1..n-2]: the RHS
 *  rows of the open inflow faces. */
void gpuOpenBCSetFace3(cudaSolverType* X, cudaSolverType* Y, cudaSolverType* Z,
                       cudaSolverType cx, cudaSolverType cy, cudaSolverType cz,
                       int dir, int faceIdx, int nx, int ny, int nz,
                       cudaStream_t stream = 0);

/** Identity rows, image = vect, on a face plane interior [1..n-2]: the image
 *  rows of the open inflow faces (their RHS rows hold E0). */
void gpuOpenBCCopyFace3(cudaSolverType* imX, cudaSolverType* imY,
                        cudaSolverType* imZ, const cudaSolverType* vX,
                        const cudaSolverType* vY, const cudaSolverType* vZ,
                        int dir, int faceIdx, int nx, int ny, int nz,
                        cudaStream_t stream = 0);

/** SAL blend: v = v*sal + target*(1-sal) on boundary layers. */
void gpuSALBlendLayers3(cudaSolverType* X, cudaSolverType* Y, cudaSolverType* Z,
                        cudaSolverType tgtX, cudaSolverType tgtY,
                        cudaSolverType tgtZ, int dir, int layerStart,
                        int layerEnd, cudaSolverType invNLayers, int ascending,
                        int nx, int ny, int nz, cudaStream_t stream = 0,
                        int skipComp = -1);

/** Set 3 arrays to constant on boundary layers. */
void gpuSetConstLayers3(cudaSolverType* X, cudaSolverType* Y, cudaSolverType* Z,
                        cudaSolverType cx, cudaSolverType cy, cudaSolverType cz,
                        int dir, int layerStart, int layerEnd, int nx, int ny,
                        int nz, cudaStream_t stream = 0);

/** Extrapolate: copy from reference plane to boundary layers. */
void gpuExtrapolateLayers3(cudaSolverType* X, cudaSolverType* Y,
                           cudaSolverType* Z, int dir, int layerStart,
                           int layerEnd, int refLayer, int nx, int ny, int nz,
                           cudaStream_t stream = 0);

/** Neutralize species node densities on X layers [layerStart, layerEnd]:
 *  every layer takes the densities of reference plane refLayer, with the
 *  positive and the negative species rescaled to the same magnitude
 *  0.5*(rho_+ + |rho_-|). rhons: [ns][nx][ny][nz]; qom: [ns]. */
void gpuNeutralizeLayersX(cudaSolverType* rhons, const cudaSolverType* qom,
                          int ns, int layerStart, int layerEnd, int refLayer,
                          int nx, int ny, int nz, cudaStream_t stream = 0);

// =========================================================================
//  Ghost rows of the X-right ghost cell plane (outflow face closure of the
//  Maxwell operator, see OutflowBoundary.h)
// =========================================================================

/** Up to kMax cell arrays handled by one gpuGhostRowsXright call. */
struct CellArrayBatch {
  static constexpr int kMax = 10;
  cudaSolverType* p[kMax];
  int n;
};

/** Fill the y and z ghost rows of the X-right ghost cell plane i = nxc-1 of
 *  every array in the batch. rule[0..3] = y-left, y-right, z-left, z-right.
 *  The y rows are filled first, then the z rows over the full y range, so
 *  the corners follow both rules. */
void gpuGhostRowsXright(const CellArrayBatch& batch, int nxc, int nyc, int nzc,
                        const int rule[4], cudaStream_t stream = 0);

// =========================================================================
//  Fix B kernels for GEM / ForceFree
// =========================================================================

/** Fix center B for GEM (tanh profile on Y-boundary layers). */
void gpuFixBcGEMKernel(cudaSolverType* Bxc, cudaSolverType* Byc,
                       cudaSolverType* Bzc, cudaSolverType B0x,
                       cudaSolverType B0y, cudaSolverType B0z,
                       cudaSolverType yStart, cudaSolverType dy,
                       cudaSolverType LyH, cudaSolverType delta, int side,
                       int nxc, int nyc, int nzc, cudaStream_t stream = 0);

/** Fix node B for GEM (tanh profile using center Y coordinates). */
void gpuFixBnGEMKernel(cudaSolverType* Bxn, cudaSolverType* Byn,
                       cudaSolverType* Bzn, cudaSolverType B0x,
                       cudaSolverType B0y, cudaSolverType B0z,
                       cudaSolverType yStart, cudaSolverType dy,
                       cudaSolverType LyH, cudaSolverType delta, int side,
                       int nxn, int nyn, int nzn, int nyc,
                       cudaStream_t stream = 0);

/** Fix center B for ForceFree (tanh Bx, 1/cosh Bz on Y boundaries). */
void gpuFixBforcefreeKernel(cudaSolverType* Bxc, cudaSolverType* Byc,
                            cudaSolverType* Bzc, cudaSolverType B0x,
                            cudaSolverType B0y, cudaSolverType B0z,
                            cudaSolverType yStart, cudaSolverType dy,
                            cudaSolverType LyH, cudaSolverType delta, int side,
                            int nxc, int nyc, int nzc, cudaStream_t stream = 0);

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
//  Center-to-center Laplacian (Poisson solver)
// =========================================================================

/** 7-point center Laplacian. */
void gpuLapC2CKernel(cudaSolverType* lapC, const cudaSolverType* fC, int nxc,
                     int nyc, int nzc, cudaSolverType invdx2,
                     cudaSolverType invdy2, cudaSolverType invdz2,
                     cudaStream_t stream = 0);

/** 7-point center Laplacian that never reads the ghost layer.  The value
 *  beyond each face is ghostFactor[f] * (adjacent interior cell), with
 *  f = Xleft, Xright, Yleft, Yright, Zleft, Zright and factor 0 (zero
 *  ghost), +1 (copy) or -1 (antisymmetric). */
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
