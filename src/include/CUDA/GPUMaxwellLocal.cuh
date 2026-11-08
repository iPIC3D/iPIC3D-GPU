/**
 * @file GPUMaxwellLocal.cuh
 * @brief Fused kernels of the Maxwell image operator and their launch sets.
 *
 * Two fused CUDA kernels that together compute the Maxwell matrix-vector
 * product  A·E  on the interior nodes.  They do not enforce boundary
 * conditions; the caller writes the boundary rows into the Krylov image.
 *
 * gpuMaxwellLocalCenterOps computes the 10 cell arrays of the operator,
 * gpuMaxwellLocalNodeFused the image from them.  Both serve the full image
 * (EMfields3D::gpuMaxwellImage), whose ghost cells come from the halo
 * exchange and the face BCs, and the communication-free operator
 * (EMfields3D::gpuMaxwellImageLocal, the Richardson sweeps of the
 * block-Jacobi preconditioner, BlockJacobiSweeps > 1), whose ghost cells are
 * zero apart from the X-right ghost cell plane on an outflow face.  Each
 * launch covers a PointBox (3D launch) or a PointList (1D launch);
 * the images launch the PointLists of maxwellImageBoxes, built once.
 *
 * The operator computed is:
 *   A·E = Δt² [ -∇²E - ∇(∇·(μ̂·E)) ] + μ̂·E + E
 *
 * Pipeline:
 *   1. gpuSolver2Phys3          — unpack Krylov → vectX/Y/Z
 *   2. gpuMUdot                 — D = μ̂·E  (all species, one kernel)
 *   3. gpuMaxwellLocalCenterOps — fused: 3×gradN2C + divN2C → 10 center arrays
 *      (full image: halo exchange and face BCs of the 10 arrays)
 *   4. gpuMaxwellLocalNodeFused — fused: 3×divC2N + gradC2N + arithmetic
 *                                  + Krylov packing → d_im
 */

#ifndef GPU_MAXWELL_LOCAL_CUH
#define GPU_MAXWELL_LOCAL_CUH

#include "GPUPointLaunch.cuh"
#include "cudaTypeDef.cuh"

// =========================================================================
//  Launch sets
// =========================================================================

/** Flat index of the cell (i, j, k) read by gpuMaxwellLocalNodeFused:
 *  base + i·si + j·sj + k·sk.  The natural strides of an [nxc][nyc][nzc]
 *  array are {0, nyc·nzc, nzc, 1}. */
struct CellIndex {
  int base, si, sj, sk;
};

/**
 * Launch sets of the Maxwell image on one subdomain, built once
 * (EMfields3D::gpuSolverAllocate).  Each is a PointList, launched 1D: one
 * box for the whole and interior sets, up to three for the shells.  The 1D
 * launch keeps every thread busy where a 3D launch of 32-wide k rows idles
 * part of them (a quarter for 96 cells: 97 interior nodes along k).
 *
 * The split operator relies on two properties of the halo exchange of the
 * cell arrays: it reads and writes only the layers {0, 1, nc-2, nc-1} of
 * each axis, and the face BCs after it write whole planes in the order X, Y,
 * Z (which keeps the ghosts of a flat axis equal to its cell).
 *
 * Whole operator (blocking exchange, local operator, gpuMUdot):
 *   mudot   D = μ·E on the interior nodes [1..nn-2]
 *   cells   the 10 cell arrays on the interior cells [1..nc-2], along x up to
 *           the X-right ghost plane nxc-1 on an outflow face
 *   nodes   the image on the interior nodes [1..nn-2]
 *   cellIndex  natural strides
 *
 * Split operator (split, input HaloOverlap): the halo exchange sends only the
 * cells next to the faces, and the node kernel reads ghost cells only on the
 * nodes next to the faces.  Along each split axis, the shells:
 *   mudotShells  nodes {1, 2, nn-3, nn-2}: the D the cell shells read
 *   cellShells   cells {1, nc-2} (and nxc-1 on an outflow face): what the
 *                exchange sends
 *   nodeShells   nodes {1, nn-2}: they read the ghost cells
 * and the interiors mudotInterior [3..nn-4], cellInterior [2..nc-3],
 * nodeInterior [2..nn-3], which read no ghost cell and write no cell the
 * exchange reads: they run while it is in flight.  The shell box of the s-th
 * split axis takes its shell, the interior of the split axes before it and
 * the whole range of the others, so the shells and the interior cover each
 * point once.  The shell boxes of a stage form one PointList (mudotShells,
 * cellShells, nodeShells): one launch per stage, which fills the GPU where
 * one launch per thin box would not.
 *
 * A flat axis (a single interior cell, periodic onto this rank: y of an x-z
 * run) is not split.  Its ghost cells equal that cell, so the split image
 * reads the cell itself instead (splitCellIndex: stride 0, base moved to the
 * cell) and every launch takes the whole axis.  The exchange copies that
 * cell into the ghosts for every cell of the other axes, so with a flat axis
 * D and the cell arrays are computed whole before it starts (mudotShells and
 * cellShells hold the whole box, mudotInterior and cellInterior are empty)
 * and only the image overlaps it.
 */
struct MaxwellImageBoxes {
  PointList mudot, cells, nodes;
  CellIndex cellIndex;
  bool split;
  PointList mudotShells, cellShells, nodeShells;
  PointList mudotInterior, cellInterior, nodeInterior;
  CellIndex splitCellIndex;
};

/**
 * Boxes of the Maxwell image of a subdomain of nCell[a] cells (ghosts
 * included) per axis.
 * @param cellEndX  last cell of the cell arrays along x: nCell[0] - 1 on an
 *                  X-right outflow face, else nCell[0] - 2
 * @param flat      per axis: a single interior cell, periodic onto this rank
 * @param overlap   build the split operator; it applies (split) when there is
 *                  a split axis and every split axis has at least 3 interior
 *                  cells
 */
MaxwellImageBoxes maxwellImageBoxes(const int nCell[3], int cellEndX,
                                    const bool flat[3], bool overlap);

// =========================================================================
//  Fused center-level operations
// =========================================================================

/**
 * @brief Fused gradN2C(Ex,Ey,Ez) + divN2C(Dx,Dy,Dz) in a single kernel.
 *
 * For each center cell (i,j,k) of points:
 *   - Reads the 8 corner nodes of (vectX,vectY,vectZ,Dx,Dy,Dz)
 *   - Computes 9 center-gradient components (3 per E-component)
 *   - Computes 1 center divergence of D
 *
 * The other cells are not written.
 *
 * @param gExX..gEzZ  9 center-sized output arrays for ∇(Ex), ∇(Ey), ∇(Ez).
 * @param divD        1 center-sized output for ∇·D.
 * @param vX,vY,vZ    Node-based input E-field components.
 * @param dX,dY,dZ    Node-based input D = μ̂·E components.
 * @param nxc,nyc,nzc Center grid dimensions (including ghost cells).
 * @param invdx,invdy,invdz  Inverse grid spacings.
 * @param points      Cells to compute: a PointBox (3D launch) or a
 *                    PointList (1D launch).
 * @param stream      CUDA stream.
 */
template <typename Points>
void gpuMaxwellLocalCenterOps(
    cudaSolverType* gExX, cudaSolverType* gExY, cudaSolverType* gExZ,
    cudaSolverType* gEyX, cudaSolverType* gEyY, cudaSolverType* gEyZ,
    cudaSolverType* gEzX, cudaSolverType* gEzY, cudaSolverType* gEzZ,
    cudaSolverType* divD, const cudaSolverType* vX, const cudaSolverType* vY,
    const cudaSolverType* vZ, const cudaSolverType* dX,
    const cudaSolverType* dY, const cudaSolverType* dZ, int nxc, int nyc,
    int nzc, cudaSolverType invdx, cudaSolverType invdy, cudaSolverType invdz,
    const Points& points, cudaStream_t stream = 0);

// =========================================================================
//  Fused node-level operations + Krylov packing
// =========================================================================

/**
 * @brief Fused divC2N(lap) + gradC2N(divD) + arithmetic + Krylov packing.
 *
 * For each node (i,j,k) of points:
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
 * @param points      Interior nodes to compute: a PointBox (3D launch) or a
 *                    PointList (1D launch).
 * @param cell        Indexing of the center arrays.
 * @param stream      CUDA stream.
 */
template <typename Points>
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
    const Points& points, const CellIndex& cell, cudaStream_t stream = 0);

#endif // GPU_MAXWELL_LOCAL_CUH
