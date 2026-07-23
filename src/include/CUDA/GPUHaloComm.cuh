/**
 * @file GPUHaloComm.cuh
 * @brief GPU-aware MPI halo communication for iPIC3D GPU field solver.
 *
 * Mirrors the functionality of Com3DNonblk.cpp / BcFields3D.cpp but operates
 * directly on device-resident GPUFieldArray3 buffers via GPU-aware MPI.
 * MPI derived datatypes (already defined in EMfields3D for the host arrays)
 * are reused because the memory layout is identical.
 *
 * Local self-copies (periodic, same-rank neighbours) and boundary-condition
 * application (BCface) are implemented as CUDA kernels.
 *
 * Compile with -DGPU_SOLVER to activate.
 */

#ifndef GPU_HALO_COMM_CUH
#define GPU_HALO_COMM_CUH

#include "GPUFieldArray.cuh"
#include <mpi.h>

// =========================================================================
//  CUDA kernels for local self-copy (periodic same-rank neighbours)
// =========================================================================

/** Source-plane offset: 0 for center exchange, 1 for node-copy exchange. */

/** Copy ghost faces for periodic X self-neighbour. */
__global__ void gpuSelfCopyFaceX(cudaSolverType* __restrict__ arr,
                                  int nx, int ny, int nz, int offset);

/** Copy ghost faces for periodic Y self-neighbour. */
__global__ void gpuSelfCopyFaceY(cudaSolverType* __restrict__ arr,
                                  int nx, int ny, int nz, int offset);

/** Copy ghost faces for periodic Z self-neighbour. */
__global__ void gpuSelfCopyFaceZ(cudaSolverType* __restrict__ arr,
                                  int nx, int ny, int nz, int offset);

// =========================================================================
//  CUDA kernels for local edge self-copy (periodic)
// =========================================================================

/** Copy ghost edges along X when X is periodic self-neighbour (Z and Y exist). */
__global__ void gpuSelfCopyEdgeX(cudaSolverType* __restrict__ arr,
                                  int nx, int ny, int nz, int offset,
                                  bool hasZright, bool hasZleft,
                                  bool hasYright, bool hasYleft);

/** Copy ghost edges along Y when Y is periodic self-neighbour. */
__global__ void gpuSelfCopyEdgeY(cudaSolverType* __restrict__ arr,
                                  int nx, int ny, int nz, int offset,
                                  bool hasXright, bool hasXleft,
                                  bool hasZright, bool hasZleft);

/** Copy ghost edges along Z when Z is periodic self-neighbour. */
__global__ void gpuSelfCopyEdgeZ(cudaSolverType* __restrict__ arr,
                                  int nx, int ny, int nz, int offset,
                                  bool hasYright, bool hasYleft,
                                  bool hasXright, bool hasXleft);

// =========================================================================
//  CUDA kernels for local corner self-copy (periodic)
// =========================================================================

__global__ void gpuSelfCopyCornerX(cudaSolverType* __restrict__ arr,
                                    int nx, int ny, int nz, int offset,
                                    bool hasYleft, bool hasYright,
                                    bool hasZleft, bool hasZright);

__global__ void gpuSelfCopyCornerY(cudaSolverType* __restrict__ arr,
                                    int nx, int ny, int nz, int offset,
                                    bool hasXleft, bool hasXright,
                                    bool hasZleft, bool hasZright);

__global__ void gpuSelfCopyCornerZ(cudaSolverType* __restrict__ arr,
                                    int nx, int ny, int nz, int offset,
                                    bool hasYleft, bool hasYright,
                                    bool hasXleft, bool hasXright);

// =========================================================================
//  Batched self-copy kernels (multiple fields in one launch)
// =========================================================================

/** Batched face self-copy: blockIdx.z selects the field. */
__global__ void gpuBatchSelfCopyFaceX(cudaSolverType* const* __restrict__ fields, int nx, int ny, int nz,
                                       int offset);
__global__ void gpuBatchSelfCopyFaceY(cudaSolverType* const* __restrict__ fields, int nx, int ny, int nz,
                                       int offset);
__global__ void gpuBatchSelfCopyFaceZ(cudaSolverType* const* __restrict__ fields, int nx, int ny, int nz,
                                       int offset);

/** Batched edge self-copy: blockIdx.y selects the field. */
__global__ void gpuBatchSelfCopyEdgeX(cudaSolverType* const* __restrict__ fields, int nx, int ny, int nz,
                                       int offset,
                                       bool hasZright, bool hasZleft, bool hasYright, bool hasYleft);
__global__ void gpuBatchSelfCopyEdgeY(cudaSolverType* const* __restrict__ fields, int nx, int ny, int nz,
                                       int offset,
                                       bool hasXright, bool hasXleft, bool hasZright, bool hasZleft);
__global__ void gpuBatchSelfCopyEdgeZ(cudaSolverType* const* __restrict__ fields, int nx, int ny, int nz,
                                       int offset,
                                       bool hasYright, bool hasYleft, bool hasXright, bool hasXleft);

/** Batched corner self-copy: blockIdx.x selects the field. */
__global__ void gpuBatchSelfCopyCornerX(cudaSolverType* const* __restrict__ fields, int nx, int ny, int nz,
                                         int offset,
                                         bool hasYleft, bool hasYright, bool hasZleft, bool hasZright);
__global__ void gpuBatchSelfCopyCornerY(cudaSolverType* const* __restrict__ fields, int nx, int ny, int nz,
                                         int offset,
                                         bool hasXleft, bool hasXright, bool hasZleft, bool hasZright);
__global__ void gpuBatchSelfCopyCornerZ(cudaSolverType* const* __restrict__ fields, int nx, int ny, int nz,
                                         int offset,
                                         bool hasYleft, bool hasYright, bool hasXleft, bool hasXright);

// =========================================================================
//  CUDA kernels for boundary condition application on faces
// =========================================================================

/** Apply BC on the Xleft face (ghost layer i=0). */
__global__ void gpuBCfaceXleft(cudaSolverType* __restrict__ arr,
                                int nx, int ny, int nz, int bcType);

/** Apply BC on the Xright face (ghost layer i=nx-1). */
__global__ void gpuBCfaceXright(cudaSolverType* __restrict__ arr,
                                 int nx, int ny, int nz, int bcType);

/** Apply BC on the Yleft face (ghost layer j=0). */
__global__ void gpuBCfaceYleft(cudaSolverType* __restrict__ arr,
                                int nx, int ny, int nz, int bcType);

/** Apply BC on the Yright face (ghost layer j=ny-1). */
__global__ void gpuBCfaceYright(cudaSolverType* __restrict__ arr,
                                 int nx, int ny, int nz, int bcType);

/** Apply BC on the Zleft face (ghost layer k=0). */
__global__ void gpuBCfaceZleft(cudaSolverType* __restrict__ arr,
                                int nx, int ny, int nz, int bcType);

/** Apply BC on the Zright face (ghost layer k=nz-1). */
__global__ void gpuBCfaceZright(cudaSolverType* __restrict__ arr,
                                 int nx, int ny, int nz, int bcType);

// =========================================================================
//  CUDA kernels for particle boundary condition (BCface_P)
// =========================================================================
// BCface_P is identical to BCface for field solver purposes.
// Re-use the same kernels.

// =========================================================================
//  CUDA kernels for additive interpolation (communicateInterp)
// =========================================================================

__global__ void gpuAddFaceX(cudaSolverType* __restrict__ arr, int nx, int ny, int nz,
                             bool hasXright, bool hasXleft);
__global__ void gpuAddFaceY(cudaSolverType* __restrict__ arr, int nx, int ny, int nz,
                             bool hasYright, bool hasYleft);
__global__ void gpuAddFaceZ(cudaSolverType* __restrict__ arr, int nx, int ny, int nz,
                             bool hasZright, bool hasZleft);

__global__ void gpuAddEdgeZ(cudaSolverType* __restrict__ arr, int nx, int ny, int nz,
                             bool hasXright, bool hasXleft,
                             bool hasYright, bool hasYleft);
__global__ void gpuAddEdgeY(cudaSolverType* __restrict__ arr, int nx, int ny, int nz,
                             bool hasXright, bool hasXleft,
                             bool hasZright, bool hasZleft);
__global__ void gpuAddEdgeX(cudaSolverType* __restrict__ arr, int nx, int ny, int nz,
                             bool hasYright, bool hasYleft,
                             bool hasZright, bool hasZleft);

__global__ void gpuAddCorner(cudaSolverType* __restrict__ arr, int nx, int ny, int nz,
                              bool hasXright, bool hasXleft,
                              bool hasYright, bool hasYleft,
                              bool hasZright, bool hasZleft);

// =========================================================================
//  Host-callable wrapper: GPU-aware halo exchange
// =========================================================================

#include "ipicfwd.h"
class EMfields3D;

/**
 * @brief Apply face boundary conditions on GPU array.
 *
 * GPU counterpart of BCface() in BcFields3D.cpp.
 */
void gpuBCface(int nx, int ny, int nz,
               GPUFieldArray3& gpuArr,
               int bcFaceXright, int bcFaceXleft,
               int bcFaceYright, int bcFaceYleft,
               int bcFaceZright, int bcFaceZleft,
               const VirtualTopology3D* vct,
               cudaStream_t stream = 0);

/**
 * @brief Apply face boundary conditions (particle version) on GPU array.
 *
 * GPU counterpart of BCface_P() in BcFields3D.cpp.
 * For now identical to gpuBCface.
 */
void gpuBCface_P(int nx, int ny, int nz,
                 GPUFieldArray3& gpuArr,
                 int bcFaceXright, int bcFaceXleft,
                 int bcFaceYright, int bcFaceYleft,
                 int bcFaceZright, int bcFaceZleft,
                 const VirtualTopology3D* vct,
                 cudaStream_t stream = 0);

#endif // GPU_HALO_COMM_CUH
