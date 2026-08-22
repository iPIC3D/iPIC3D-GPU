/*
 * Compile-time guarded GPU state diagnostics used to localize the first bad
 * particle, moment, or field in a timestep.  The implementation owns bounded
 * persistent reduction buffers; full simulation arrays are never copied to
 * the host.
 */

#ifndef GPU_CYCLE_DIAGNOSTICS_CUH
#define GPU_CYCLE_DIAGNOSTICS_CUH

#if defined(IPIC3D_GPU_CYCLE_DIAGNOSTICS)

#include "cudaTypeDef.cuh"
#include "mpi.h"

#include <cstddef>
#include <cstdint>

class particleArrayCUDA;
class grid3DCUDA;
struct CellSorter;

/** Inclusive particle-domain bounds for the local and global simulation box. */
struct GPUCycleDiagnosticParticleBounds {
  double localMin[3];
  double localMax[3];
  double globalMin[3];
  double globalMax[3];
};

/**
 * A rectangular portion of a row-major three-dimensional device array.
 * lo is inclusive and hi is exclusive.  origin is the physical coordinate of
 * element (0,0,0), and spacing is the coordinate increment per index.
 */
struct GPUCycleDiagnosticRegion {
  int nx;
  int ny;
  int nz;
  int lo[3];
  int hi[3];
  double origin[3];
  double spacing[3];
};

/** True for producer-state diagnostics on the configured cycle. */
inline bool gpuCycleDiagnosticsStateCycle(int cycle) {
  constexpr int target = IPIC3D_GPU_DIAGNOSTIC_CYCLE;
  return target < 0 || cycle == target;
}

/**
 * Maxwell-source diagnostics include the configured producer cycle as a
 * baseline and the following cycle, whose RHS consumes that producer state.
 */
inline bool gpuCycleDiagnosticsSourceCycle(int cycle) {
  constexpr int target = IPIC3D_GPU_DIAGNOSTIC_CYCLE;
  return target < 0 || cycle == target || cycle == target + 1;
}

/**
 * Reusable diagnostic reducer.  Storage is allocated lazily on its first
 * enabled checkpoint, then retained for the owning solver object's lifetime.
 */
class GPUCycleDiagnostics {
public:
  GPUCycleDiagnostics();
  ~GPUCycleDiagnostics();

  GPUCycleDiagnostics(const GPUCycleDiagnostics&) = delete;
  GPUCycleDiagnostics& operator=(const GPUCycleDiagnostics&) = delete;

  /** Report one SoA particle range, including bounds and offending records. */
  void reportParticleRange(const char* stage, int cycle, int species,
                           const char* rangeName,
                           const particleArrayCUDA& particles,
                           std::uint32_t begin, std::uint32_t count,
                           const GPUCycleDiagnosticParticleBounds& bounds,
                           cudaStream_t stream, MPI_Comm comm);

  /**
   * Report max-absolute value, signed value, location, and nonfinite count for
   * a set of arrays sharing one logical region.  scale[f] is applied only
   * while reducing and does not modify the input array.
   */
  void reportArraySet(const char* stage, int cycle, int species,
                      const char* const* componentNames,
                      const cudaSolverType* const* deviceArrays,
                      const double* scale, int fieldCount,
                      const GPUCycleDiagnosticRegion& region,
                      cudaStream_t stream, MPI_Comm comm);

  /**
   * Validate the preserved Stage 1 histogram, raw Phase-1 tile totals, the
   * completed Stage 2 scan, Stage 3 counters and permutation, then arm exact
   * poison checks for the pending Stage 4.
   */
  void reportCellSorterBeforeScatter(int cycle, int species, CellSorter& sorter,
                                     const particleArrayCUDA& particles,
                                     const grid3DCUDA& hostGrid,
                                     const grid3DCUDA* deviceGrid,
                                     bool expectFullSort, cudaStream_t stream,
                                     MPI_Comm comm);

  /**
   * Report per-field Stage-4 write coverage, pointer rotation, and whether the
   * resulting sorted particle ranges contain particles from their claimed
   * cells.  Must follow reportCellSorterBeforeScatter() and finishSort().
   */
  void reportCellSorterAfterScatter(int cycle, int species,
                                    const CellSorter& sorter,
                                    const particleArrayCUDA& particles,
                                    const grid3DCUDA& hostGrid,
                                    const grid3DCUDA* deviceGrid,
                                    cudaStream_t stream, MPI_Comm comm);

private:
  struct Impl;
  Impl* impl_;
};

#endif // IPIC3D_GPU_CYCLE_DIAGNOSTICS

#endif // GPU_CYCLE_DIAGNOSTICS_CUH
