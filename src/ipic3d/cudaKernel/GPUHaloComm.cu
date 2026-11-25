/**
 * @file GPUHaloComm.cu
 * @brief The halo entry of the field arrays: an exchange with the
 *        neighbours (gpuBatchedHaloExchange, GPUBatchedHaloComm.cu), then
 *        the ghosts of the physical faces from the rules of a GhostSet
 *        (FieldBoundary::fill, one launch); and finishE, the boundary of a
 *        solved E.
 *
 * The set fixes the geometry (cells: send offset 0; nodes: offset 1) and the
 * communicator (field or particle topology); facesOnly exchanges the 6 face
 * boxes only (box stencils).
 */

#include "EMfields3D.h"
#include "Profiler.h"

void EMfields3D::exchangeAndFill(GhostSet s, cudaSolverType* const* p, int n,
                                 bool facesOnly) {
  profiling::Range range(profiling::Region::HaloExchange);
  const int* e = bc_.extents(s);
  gpuBatchedHaloExchange(p, n, e[0], e[1], e[2], bc_.setOnCells(s), facesOnly,
                         false, bc_.setParticle(s), solverStream_);
  bc_.fill(s, p, n, solverStream_);
}

void EMfields3D::beginExchange(GhostSet s, cudaSolverType* const* p, int n,
                               bool facesOnly) {
  profiling::Range range(profiling::Region::HaloBegin);
  const int* e = bc_.extents(s);
  gpuBatchedHaloBeginExchange(p, n, e[0], e[1], e[2], bc_.setOnCells(s),
                              facesOnly, bc_.setParticle(s), solverStream_);
}

void EMfields3D::endExchangeAndFill(GhostSet s, cudaSolverType* const* p,
                                    int n) {
  profiling::Range range(profiling::Region::HaloEnd);
  gpuBatchedHaloEndExchange();
  bc_.fill(s, p, n, solverStream_);
}

void EMfields3D::finishE(cudaSolverType* const* E, int n) {
  profiling::Range range(profiling::Region::FinishE);
  exchangeAndFill(GhostSet::ENode, E, n);
  for (int i = 0; i + 2 < n; i += 3)
    bc_.layers(LayerSet::E, E[i], E[i + 1], E[i + 2], solverStream_);
}
