/**
 * @file GPUPhysicsKernels.cuh
 * @brief GPU kernel declarations for iPIC3D physics operations:
 *        MUdot, PIdot, perfect conductor BCs, and sustensor computation.
 *
 * All host wrapper functions are callable from .cpp translation units
 * (they do not use <<<>>> syntax directly).
 *
 * Compile with -DGPU_SOLVER to activate.
 */

#ifndef GPU_PHYSICS_KERNELS_CUH
#define GPU_PHYSICS_KERNELS_CUH

#include "cudaTypeDef.cuh"
#include <cstddef>

#ifdef GPU_SOLVER

// =========================================================================
//  MUdot: magnetised susceptibility response  μ·E
//  Sums over all species.  Output arrays are zeroed then accumulated.
// =========================================================================

/**
 * @brief Compute MUdot = Σ_s ( denom_s * rot_s(E) ) for one species.
 *
 * @param MUdotX/Y/Z  Device output (node-based [nxn][nyn][nzn]).
 * @param vX/Y/Z      Input vector (node-based).
 * @param Bxn/Byn/Bzn Total B at nodes.
 * @param Bx_ext/By_ext/Bz_ext External B at nodes.
 * @param rhons_is    Species density at nodes (single species slice).
 * @param beta        = 0.5 * qom[is] * dt / c
 * @param prefactor   = FourPI/2 * delt * dt/c * qom[is]
 * @param nxn,nyn,nzn Grid dimensions (node-based, including ghosts).
 * @param firstSpecies If true, writes output; if false, accumulates (+=).
 * @param stream      CUDA stream.
 */
void gpuMUdotSpecies(cudaSolverType* MUdotX, cudaSolverType* MUdotY, cudaSolverType* MUdotZ,
                     const cudaSolverType* vX, const cudaSolverType* vY, const cudaSolverType* vZ,
                     const cudaSolverType* Bxn, const cudaSolverType* Byn, const cudaSolverType* Bzn,
                     const cudaSolverType* Bx_ext, const cudaSolverType* By_ext, const cudaSolverType* Bz_ext,
                     const cudaSolverType* rhons_is,
                     cudaSolverType beta, cudaSolverType prefactor,
                     int nxn, int nyn, int nzn,
                     bool firstSpecies,
                     cudaStream_t stream = 0);

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
 * @param Bxn/Byn/Bzn Total B at nodes.
 * @param Bx_ext/By_ext/Bz_ext External B at nodes.
 * @param beta         = 0.5 * qom[ns] * dt / c
 * @param nxn,nyn,nzn  Grid dimensions.
 * @param stream       CUDA stream.
 */
void gpuPIdotSpecies(cudaSolverType* PIX, cudaSolverType* PIY, cudaSolverType* PIZ,
                     const cudaSolverType* vX, const cudaSolverType* vY, const cudaSolverType* vZ,
                     const cudaSolverType* Bxn, const cudaSolverType* Byn, const cudaSolverType* Bzn,
                     const cudaSolverType* Bx_ext, const cudaSolverType* By_ext, const cudaSolverType* Bz_ext,
                     cudaSolverType beta,
                     int nxn, int nyn, int nzn,
                     cudaStream_t stream = 0);

// =========================================================================
//  Perfect conductor boundary conditions (fused sustensor + application)
// =========================================================================

/**
 * @brief GPU perfect conductor BC on the LEFT boundary for direction dir.
 *
 * dir = 0 → X-left (i=1),  dir = 1 → Y-left (j=1),  dir = 2 → Z-left (k=1).
 *
 * Computes the sustensor on the boundary face and applies the correction:
 *   imageX[bnd] = vectX[bnd] - (E[bnd] - sus_yx*vY - sus_zx*vZ - Jh*dt*th*4π) / sus_xx
 *   (with appropriate permutation for Y and Z directions)
 *
 * @param imageX/Y/Z  Device arrays to write the corrected image.
 * @param vectX/Y/Z   Input Krylov iterate (node-based).
 * @param Ex/Ey/Ez     Electric field at nodes.
 * @param Jxh/Jyh/Jzh Hat current at nodes.
 * @param Bxn/Byn/Bzn B at nodes.
 * @param Bx_ext/By_ext/Bz_ext External B.
 * @param rhons        All-species density [ns][nxn][nyn][nzn] (contiguous).
 * @param qom          Species q/m array (host, ns elements).  Copied internally.
 * @param ns           Number of species.
 * @param dt,c,th,FourPI,delt  Physical constants.
 * @param nxn,nyn,nzn  Grid dimensions.
 * @param dir          Direction (0=X, 1=Y, 2=Z).
 * @param stream       CUDA stream.
 */
void gpuPerfectConductorLeft(
    cudaSolverType* imageX, cudaSolverType* imageY, cudaSolverType* imageZ,
    const cudaSolverType* vectX, const cudaSolverType* vectY, const cudaSolverType* vectZ,
    const cudaSolverType* Ex, const cudaSolverType* Ey, const cudaSolverType* Ez,
    const cudaSolverType* Jxh, const cudaSolverType* Jyh, const cudaSolverType* Jzh,
    const cudaSolverType* Bxn, const cudaSolverType* Byn, const cudaSolverType* Bzn,
    const cudaSolverType* Bx_ext, const cudaSolverType* By_ext, const cudaSolverType* Bz_ext,
    const cudaSolverType* rhons,
    const cudaSolverType* d_qom, int ns,
    cudaSolverType dt, cudaSolverType c, cudaSolverType th, cudaSolverType FourPI, cudaSolverType delt,
    int nxn, int nyn, int nzn,
    int dir,
    cudaStream_t stream = 0);

/** Same as gpuPerfectConductorLeft but for the RIGHT boundary. */
void gpuPerfectConductorRight(
    cudaSolverType* imageX, cudaSolverType* imageY, cudaSolverType* imageZ,
    const cudaSolverType* vectX, const cudaSolverType* vectY, const cudaSolverType* vectZ,
    const cudaSolverType* Ex, const cudaSolverType* Ey, const cudaSolverType* Ez,
    const cudaSolverType* Jxh, const cudaSolverType* Jyh, const cudaSolverType* Jzh,
    const cudaSolverType* Bxn, const cudaSolverType* Byn, const cudaSolverType* Bzn,
    const cudaSolverType* Bx_ext, const cudaSolverType* By_ext, const cudaSolverType* Bz_ext,
    const cudaSolverType* rhons,
    const cudaSolverType* d_qom, int ns,
    cudaSolverType dt, cudaSolverType c, cudaSolverType th, cudaSolverType FourPI, cudaSolverType delt,
    int nxn, int nyn, int nzn,
    int dir,
    cudaStream_t stream = 0);

/**
 * @brief GPU perfect conductor BC for source term (LEFT boundary).
 *
 * Simpler version used in MaxwellSource:
 *   vectorX[bnd] = 0 (tangential components zeroed).
 */
void gpuPerfectConductorLeftS(
    cudaSolverType* vectorX, cudaSolverType* vectorY, cudaSolverType* vectorZ,
    int nxn, int nyn, int nzn,
    int dir,
    cudaStream_t stream = 0);

/** Same as gpuPerfectConductorLeftS but for the RIGHT boundary. */
void gpuPerfectConductorRightS(
    cudaSolverType* vectorX, cudaSolverType* vectorY, cudaSolverType* vectorZ,
    int nxn, int nyn, int nzn,
    int dir,
    cudaStream_t stream = 0);

// =========================================================================
//  adjustNonPeriodicDensities: doubles boundary-face values for
//  non-periodic BCs (10 moment arrays per species).
// =========================================================================

/**
 * @brief GPU version of adjustNonPeriodicDensities for one species.
 *
 * For each non-periodic boundary face (where the neighbour is MPI_PROC_NULL),
 * doubles the 10 per-species moment values on that face:
 *   rhons, Jxs, Jys, Jzs, pXXsn, pXYsn, pXZsn, pYYsn, pYZsn, pZZsn
 *
 * @param ptrs       Array of 10 device pointers (one per moment quantity).
 * @param nxn,nyn,nzn  Grid dimensions (node-based, including ghosts).
 * @param xLeftNull  true if X-left neighbour is MPI_PROC_NULL.
 * @param xRightNull true if X-right neighbour is MPI_PROC_NULL.
 * @param yLeftNull  true if Y-left neighbour is MPI_PROC_NULL.
 * @param yRightNull true if Y-right neighbour is MPI_PROC_NULL.
 * @param zLeftNull  true if Z-left neighbour is MPI_PROC_NULL.
 * @param zRightNull true if Z-right neighbour is MPI_PROC_NULL.
 * @param stream     CUDA stream.
 */
void gpuAdjustNonPeriodicDensities(
    int nptrs,
    cudaSolverType* const* d_devPtrs,
    int nxn, int nyn, int nzn,
    bool xLeftNull, bool xRightNull,
    bool yLeftNull, bool yRightNull,
    bool zLeftNull, bool zRightNull,
    cudaStream_t stream = 0);

// =========================================================================
//  Open boundary face/layer kernels
// =========================================================================

/** Zero 3 arrays on a face plane interior [1..n-2]. */
void gpuOpenBCZeroFace3(cudaSolverType* X, cudaSolverType* Y, cudaSolverType* Z,
    int dir, int faceIdx, int nx, int ny, int nz, cudaStream_t stream = 0);

/** Set image = vect - injE on a face plane interior. */
void gpuOpenBCImageDiffFace3(cudaSolverType* imX, cudaSolverType* imY, cudaSolverType* imZ,
    const cudaSolverType* vX, const cudaSolverType* vY, const cudaSolverType* vZ,
    cudaSolverType injE0, cudaSolverType injE1, cudaSolverType injE2,
    int dir, int faceIdx, int nx, int ny, int nz, cudaStream_t stream = 0);

/** SAL blend: v = v*sal + target*(1-sal) on boundary layers. */
void gpuSALBlendLayers3(cudaSolverType* X, cudaSolverType* Y, cudaSolverType* Z,
    cudaSolverType tgtX, cudaSolverType tgtY, cudaSolverType tgtZ,
    int dir, int layerStart, int layerEnd,
    cudaSolverType invNLayers, int ascending,
    int nx, int ny, int nz, cudaStream_t stream = 0);

/** Set 3 arrays to constant on boundary layers. */
void gpuSetConstLayers3(cudaSolverType* X, cudaSolverType* Y, cudaSolverType* Z,
    cudaSolverType cx, cudaSolverType cy, cudaSolverType cz,
    int dir, int layerStart, int layerEnd,
    int nx, int ny, int nz, cudaStream_t stream = 0);

/** Extrapolate: copy from reference plane to boundary layers. */
void gpuExtrapolateLayers3(cudaSolverType* X, cudaSolverType* Y, cudaSolverType* Z,
    int dir, int layerStart, int layerEnd, int refLayer,
    int nx, int ny, int nz, cudaStream_t stream = 0);

// =========================================================================
//  Fix B kernels for GEM / ForceFree
// =========================================================================

/** Fix center B for GEM (tanh profile on Y-boundary layers). */
void gpuFixBcGEMKernel(cudaSolverType* Bxc, cudaSolverType* Byc, cudaSolverType* Bzc,
    cudaSolverType B0x, cudaSolverType B0y, cudaSolverType B0z,
    cudaSolverType yStart, cudaSolverType dy, cudaSolverType LyH, cudaSolverType delta,
    int side, int nxc, int nyc, int nzc, cudaStream_t stream = 0);

/** Fix node B for GEM (tanh profile using center Y coordinates). */
void gpuFixBnGEMKernel(cudaSolverType* Bxn, cudaSolverType* Byn, cudaSolverType* Bzn,
    cudaSolverType B0x, cudaSolverType B0y, cudaSolverType B0z,
    cudaSolverType yStart, cudaSolverType dy, cudaSolverType LyH, cudaSolverType delta,
    int side, int nxn, int nyn, int nzn, int nyc, cudaStream_t stream = 0);

/** Fix center B for ForceFree (tanh Bx, 1/cosh Bz on Y boundaries). */
void gpuFixBforcefreeKernel(cudaSolverType* Bxc, cudaSolverType* Byc, cudaSolverType* Bzc,
    cudaSolverType B0x, cudaSolverType B0y, cudaSolverType B0z,
    cudaSolverType yStart, cudaSolverType dy, cudaSolverType LyH, cudaSolverType delta,
    int side, int nxc, int nyc, int nzc, cudaStream_t stream = 0);

// =========================================================================
//  ConstantChargePlanet kernels
// =========================================================================

/** Set rhons = val inside sphere of radius R. */
void gpuConstantChargePlanetKernel(cudaSolverType* rhons, cudaSolverType val,
    cudaSolverType R, cudaSolverType xc, cudaSolverType yc, cudaSolverType zc,
    cudaSolverType xStart, cudaSolverType yStart, cudaSolverType zStart,
    cudaSolverType dx, cudaSolverType dy, cudaSolverType dz,
    int nxn, int nyn, int nzn, cudaStream_t stream = 0);

/** 2D version: set rhons inside circle in XZ plane. */
void gpuConstantChargePlanet2DKernel(cudaSolverType* rhons, cudaSolverType val,
    cudaSolverType R, cudaSolverType xc, cudaSolverType zc,
    cudaSolverType xStart, cudaSolverType zStart, cudaSolverType dx, cudaSolverType dz,
    int nxn, int nyn, int nzn, cudaStream_t stream = 0);

// =========================================================================
//  Boundary layer subtraction (divB cleaning)
// =========================================================================

/** B -= gradPSI on boundary layers. */
void gpuSubBoundaryLayers3(cudaSolverType* BxN, cudaSolverType* ByN, cudaSolverType* BzN,
    const cudaSolverType* gX, const cudaSolverType* gY, const cudaSolverType* gZ,
    int dir, int layerStart, int layerEnd,
    int nxn, int nyn, int nzn, cudaStream_t stream = 0);

// =========================================================================
//  Center-to-center Laplacian (Poisson solver)
// =========================================================================

/** 7-point center Laplacian. */
void gpuLapC2CKernel(cudaSolverType* lapC, const cudaSolverType* fC,
    int nxc, int nyc, int nzc,
    cudaSolverType invdx2, cudaSolverType invdy2, cudaSolverType invdz2, cudaStream_t stream = 0);

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
 * Completely communication-free.
 *
 * @param zX/zY/zZ   Output solution (node-based).
 * @param rX/rY/rZ   Input RHS (node-based).
 * @param Bxn/Byn/Bzn   Total B at nodes.
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
    const cudaSolverType* rX, const cudaSolverType* rY, const cudaSolverType* rZ,
    const cudaSolverType* Bxn, const cudaSolverType* Byn, const cudaSolverType* Bzn,
    const cudaSolverType* Bx_ext, const cudaSolverType* By_ext, const cudaSolverType* Bz_ext,
    const cudaSolverType* rhons, const cudaSolverType* d_qom,
    int ns,
    cudaSolverType dt, cudaSolverType c_val, cudaSolverType delt, cudaSolverType FourPI,
    cudaSolverType diagScalar, cudaSolverType wx, cudaSolverType wy, cudaSolverType wz,
    int nxn, int nyn, int nzn,
    cudaStream_t stream = 0);

/**
 * Precompute D^{-1} at each interior node and store 9 entries per node.
 * Layout: Dinv[(row*3+col) * nodeSlice + nodeIdx]
 */
void gpuPrecomputeBlockJacobiInv(
    cudaSolverType* Dinv,
    const cudaSolverType* Bxn, const cudaSolverType* Byn, const cudaSolverType* Bzn,
    const cudaSolverType* Bx_ext, const cudaSolverType* By_ext, const cudaSolverType* Bz_ext,
    const cudaSolverType* rhons, const cudaSolverType* d_qom,
    int ns, cudaSolverType dt, cudaSolverType c_val, cudaSolverType delt, cudaSolverType FourPI,
    cudaSolverType diagScalar, cudaSolverType wx, cudaSolverType wy, cudaSolverType wz,
    int nxn, int nyn, int nzn, cudaStream_t stream = 0);

/**
 * Fast D^{-1} application operating directly on Krylov vectors.
 * zKrylov = D^{-1} * rKrylov  (no pack/unpack needed).
 */
void gpuApplyBlockJacobiInvKrylov(
    cudaSolverType* zKrylov, const cudaSolverType* rKrylov, const cudaSolverType* Dinv,
    int nxn, int nyn, int nzn, cudaStream_t stream = 0);

#endif // GPU_SOLVER
#endif // GPU_PHYSICS_KERNELS_CUH
