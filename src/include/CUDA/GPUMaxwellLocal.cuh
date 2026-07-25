/**
 * @file GPUMaxwellLocal.cuh
 * @brief Communication-free (local) Maxwell image operator kernels.
 *
 * Provides two highly fused CUDA kernels that together compute the
 * Maxwell matrix-vector product  A·E  WITHOUT any MPI communication.
 * The fused kernels themselves do not enforce boundary conditions; the caller
 * may apply boundary image corrections after unpacking. Ghost cells are
 * assumed to be zero.
 *
 * Intended use: GPU-local preconditioner for the GMRES Maxwell solver.
 *
 * The operator computed is:
 *   A·E = Δt² [ -∇²E - ∇(∇·(μ̂·E)) ] + μ̂·E + E
 *
 * Pipeline (called from EMfields3D::gpuMaxwellImageLocal):
 *   1. gpuSolver2Phys3          — unpack Krylov → vectX/Y/Z (existing)
 *   2. gpuMUdot                 — D = μ̂·E  (existing, ns species)
 *   3. gpuMaxwellLocalCenterOps — fused: 3×gradN2C + divN2C → 10 center arrays
 *   4. gpuMaxwellLocalNodeFused — fused: 3×divC2N + gradC2N + arithmetic
 *                                  + phys2solver packing → d_im
 *
 * Total kernel launches: 2 + ns  (solver2phys, ns×MUdot, center, node).
 *
 * Compile with -DGPU_SOLVER to activate.
 */

#ifndef GPU_MAXWELL_LOCAL_CUH
#define GPU_MAXWELL_LOCAL_CUH

#include "cudaTypeDef.cuh"

#ifdef GPU_SOLVER

// =========================================================================
//  Fused center-level operations
// =========================================================================

/**
 * @brief Fused gradN2C(Ex,Ey,Ez) + divN2C(Dx,Dy,Dz) in a single kernel.
 *
 * For each interior center cell (i,j,k) with 1 ≤ i ≤ nxc-2, etc.:
 *   - Reads the 8 corner nodes of (vectX,vectY,vectZ,Dx,Dy,Dz)
 *   - Computes 9 center-gradient components (3 per E-component)
 *   - Computes 1 center divergence of D
 *
 * Center ghost cells (i=0, nxc-1, etc.) remain zero — caller must
 * ensure they were zeroed before this call.
 *
 * @param gExX..gEzZ  9 center-sized output arrays for ∇(Ex), ∇(Ey), ∇(Ez).
 * @param divD        1 center-sized output for ∇·D.
 * @param vX,vY,vZ    Node-based input E-field components.
 * @param dX,dY,dZ    Node-based input D = μ̂·E components.
 * @param nxc,nyc,nzc Center grid dimensions (including ghost cells).
 * @param invdx,invdy,invdz  Inverse grid spacings.
 * @param stream      CUDA stream.
 */
void gpuMaxwellLocalCenterOps(
    cudaSolverType* gExX, cudaSolverType* gExY, cudaSolverType* gExZ,
    cudaSolverType* gEyX, cudaSolverType* gEyY, cudaSolverType* gEyZ,
    cudaSolverType* gEzX, cudaSolverType* gEzY, cudaSolverType* gEzZ,
    cudaSolverType* divD, const cudaSolverType* vX, const cudaSolverType* vY,
    const cudaSolverType* vZ, const cudaSolverType* dX,
    const cudaSolverType* dY, const cudaSolverType* dZ, int nxc, int nyc,
    int nzc, cudaSolverType invdx, cudaSolverType invdy, cudaSolverType invdz,
    cudaStream_t stream = 0);

// =========================================================================
//  Fused node-level operations + Krylov packing
// =========================================================================

/**
 * @brief Fused divC2N(lap) + gradC2N(divD) + arithmetic + phys2solver.
 *
 * For each interior node (i,j,k) with 1 ≤ i ≤ nxn-2, etc.:
 *   - Reads the 8 surrounding center cells for 9 gradient arrays
 *     → computes lapX, lapY, lapZ via divC2N stencil
 *   - Reads the 8 surrounding center cells of divD
 *     → computes gdivX, gdivY, gdivZ via gradC2N stencil
 *   - Reads vectX/Y/Z and Dx/Dy/Dz at the node
 *   - Computes: im_c = dt²·(-lap_c - gdiv_c) + D_c + E_c
 *   - Packs the 3 components directly into the Krylov output vector
 *
 * @param d_im        Output Krylov vector (length 3·(nxn-2)·(nyn-2)·(nzn-2)).
 * @param gExX..gEzZ  9 center-sized arrays (∇E components from center kernel).
 * @param divD        1 center-sized array (∇·D from center kernel).
 * @param vX,vY,vZ    Node-based E-field components.
 * @param dX,dY,dZ    Node-based D = μ̂·E components.
 * @param nxn,nyn,nzn Node grid dimensions (including ghost cells).
 * @param invdx,invdy,invdz  Inverse grid spacings.
 * @param dt2         = delt * delt  (squared time-step factor).
 * @param stream      CUDA stream.
 */
void gpuMaxwellLocalNodeFused(
    cudaSolverType* d_im, const cudaSolverType* gExX,
    const cudaSolverType* gExY, const cudaSolverType* gExZ,
    const cudaSolverType* gEyX, const cudaSolverType* gEyY,
    const cudaSolverType* gEyZ, const cudaSolverType* gEzX,
    const cudaSolverType* gEzY, const cudaSolverType* gEzZ,
    const cudaSolverType* divD, const cudaSolverType* vX,
    const cudaSolverType* vY, const cudaSolverType* vZ,
    const cudaSolverType* dX, const cudaSolverType* dY,
    const cudaSolverType* dZ, int nxn, int nyn, int nzn, cudaSolverType invdx,
    cudaSolverType invdy, cudaSolverType invdz, cudaSolverType dt2,
    cudaStream_t stream = 0);

#endif // GPU_SOLVER

#endif // GPU_MAXWELL_LOCAL_CUH
