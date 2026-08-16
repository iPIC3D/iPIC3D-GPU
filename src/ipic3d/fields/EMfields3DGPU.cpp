/* iPIC3D was originally developed by Stefano Markidis and Giovanni Lapenta.
 * This release was contributed by Alec Johnson and Ivy Bo Peng.
 * Publications that use results from iPIC3D need to properly cite
 * 'S. Markidis, G. Lapenta, and Rizwan-uddin. "Multi-scale simulations of
 * plasma with iPIC3D." Mathematics and Computers in Simulation 80.7 (2010):
 * 1509-1519.'
 *
 *        Copyright 2015 KTH Royal Institute of Technology
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *         http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

// =========================================================================
//  EMfields3DGPU.cpp  –  GPU solver methods for EMfields3D
//
//  Extracted from EMfields3D.cpp for modularity.  All code here lives
//  inside  #ifdef GPU_SOLVER  and is compiled only when that flag is on.
// =========================================================================

#ifdef GPU_SOLVER

#include "Collective.h"
#include "Com3DNonblk.h"
#include "EMfields3D.h"
#include "Grid3DCU.h"
#include "Parameters.h"
#include "TimeTasks.h"
#include "VCtopology3D.h"
#include "errors.h"
#include <mpi.h>

#include "GPUBlas.cuh"
#include "GPUChebyshev.cuh"
#include "GPUMaxwellLocal.cuh"
#include "GPUPhysicsKernels.cuh"
#include "GPUSolverMPITypes.h"
#include "GPUStencils.cuh"
#include "cudaTypeDef.cuh"

#ifdef HALO_OVERLAP
// Forward declarations for BC face functions (defined in GPUHaloComm.cu).
// Cannot include GPUHaloComm.cuh here because it contains __global__ decls
// and this file is compiled by the host compiler.
void gpuBCface(int nx, int ny, int nz, GPUFieldArray3& gpuArr, int bcFaceXright,
               int bcFaceXleft, int bcFaceYright, int bcFaceYleft,
               int bcFaceZright, int bcFaceZleft, const VirtualTopology3D* vct,
               cudaStream_t stream);
void gpuBCface_P(int nx, int ny, int nz, GPUFieldArray3& gpuArr,
                 int bcFaceXright, int bcFaceXleft, int bcFaceYright,
                 int bcFaceYleft, int bcFaceZright, int bcFaceZleft,
                 const VirtualTopology3D* vct, cudaStream_t stream);
#endif // HALO_OVERLAP

#include <algorithm>
#include <chrono>
#include <climits>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <vector>

// =========================================================================
//  GPU Solver: allocation / deallocation / synchronisation
// =========================================================================

void EMfields3D::gpuSolverAllocate() {
  if (gpuSolverAllocated_)
    return;

  const VirtualTopology3D* vct = &get_vct();
  const Collective* col = &get_col();
  const size_t centerPoints = (size_t)nxc * nyc * nzc;
  const size_t nodePoints = (size_t)nxn * nyn * nzn;
  if (centerPoints > INT_MAX || nodePoints > INT_MAX) {
    cerr << "ERROR: GPU field solver uses signed-int stencil indexing; rank "
         << vct->getCartesian_rank() << " has center/node volumes "
         << centerPoints << "/" << nodePoints << " exceeding INT_MAX" << endl;
    MPI_Abort(vct->getFieldComm(), EXIT_FAILURE);
    std::abort();
  }

  // Boundary kernels index n_layers_sal directly.  Validate active physical
  // boundary branches against their local array extent.
  auto validateLayers = [&](bool active, int extent, int reserved,
                            const char* operation) {
    if (!active)
      return;
    const int minLayers = yes_sal ? 1 : 0;
    const int maxLayers = extent - reserved;
    if (n_layers_sal >= minLayers && n_layers_sal <= maxLayers)
      return;
    cerr << "ERROR: invalid n_layers_sal=" << n_layers_sal << " for "
         << operation << " on rank " << vct->getCartesian_rank()
         << "; valid local range is [" << minLayers << "," << maxLayers
         << "] for extent " << extent << endl;
    MPI_Abort(vct->getFieldComm(), EXIT_FAILURE);
    std::abort();
  };

  const bool xLeftOpen =
      vct->getXleft_neighbor() == MPI_PROC_NULL && bcEMfaceXleft == 2;
  const bool xRightOpen =
      vct->getXright_neighbor() == MPI_PROC_NULL && bcEMfaceXright == 2;
  const bool yLeftOpen =
      vct->getYleft_neighbor() == MPI_PROC_NULL && bcEMfaceYleft == 2;
  const bool yRightOpen =
      vct->getYright_neighbor() == MPI_PROC_NULL && bcEMfaceYright == 2;
  const bool zLeftOpen =
      vct->getZleft_neighbor() == MPI_PROC_NULL && bcEMfaceZleft == 2;
  const bool zRightOpen =
      vct->getZright_neighbor() == MPI_PROC_NULL && bcEMfaceZright == 2;

  validateLayers(xLeftOpen && col->getBcPfaceXleft() == 2, nxn, 1,
                 "electric X-left open boundary");
  validateLayers(xRightOpen && col->getBcPfaceXright() == 3, nxn, 2,
                 "electric X-right open boundary");
  validateLayers(yLeftOpen && col->getBcPfaceYleft() == 2, nyn, yes_sal ? 1 : 2,
                 "electric Y-left open boundary");
  validateLayers(yRightOpen && col->getBcPfaceYright() == 2, nyn,
                 yes_sal ? 1 : 2, "electric Y-right open boundary");
  validateLayers(zLeftOpen && col->getBcPfaceZleft() == 2, nzn, yes_sal ? 1 : 2,
                 "electric Z-left open boundary");
  validateLayers(zRightOpen && col->getBcPfaceZright() == 2, nzn,
                 yes_sal ? 1 : 2, "electric Z-right open boundary");

  // Center-B open-boundary extrapolation is active only for extents greater
  // than 10 and requires one untouched reference layer beyond the corrected
  // range.
  validateLayers(xLeftOpen && nxc > 10, nxc, 1,
                 "magnetic X-left open boundary");
  validateLayers(xRightOpen && nxc > 10, nxc, 2,
                 "magnetic X-right open boundary");
  validateLayers(yLeftOpen && nyc > 10, nyc, yes_sal ? 1 : 2,
                 "magnetic Y-left open boundary");
  validateLayers(yRightOpen && nyc > 10, nyc, yes_sal ? 1 : 2,
                 "magnetic Y-right open boundary");
  validateLayers(zLeftOpen && nzc > 10, nzc, yes_sal ? 1 : 2,
                 "magnetic Z-left open boundary");
  validateLayers(zRightOpen && nzc > 10, nzc, yes_sal ? 1 : 2,
                 "magnetic Z-right open boundary");

  if (divBCorrection) {
    validateLayers(xLeftOpen || xRightOpen, nxn, 0,
                   "magnetic divergence correction in X");
    validateLayers(yLeftOpen || yRightOpen, nyn, 0,
                   "magnetic divergence correction in Y");
    validateLayers(zLeftOpen || zRightOpen, nzn, 0,
                   "magnetic divergence correction in Z");
  }

  // Raw CUDA workspaces below coexist with RAII field wrappers.  Mark the
  // aggregate live up front so an allocation exception rolls all completed
  // pieces back through the normal teardown path.
  gpuSolverAllocated_ = true;
  struct AllocationRollback {
    EMfields3D* fields;
    ~AllocationRollback() {
      if (fields)
        fields->gpuSolverFree();
    }
  } rollback{this};

  // ---- Electric field (node-based) ----
  d_Ex = GPUFieldArray3(nxn, nyn, nzn);
  d_Ey = GPUFieldArray3(nxn, nyn, nzn);
  d_Ez = GPUFieldArray3(nxn, nyn, nzn);
  d_Exth = GPUFieldArray3(nxn, nyn, nzn);
  d_Eyth = GPUFieldArray3(nxn, nyn, nzn);
  d_Ezth = GPUFieldArray3(nxn, nyn, nzn);

  // ---- Magnetic field ----
  d_Bxc = GPUFieldArray3(nxc, nyc, nzc);
  d_Byc = GPUFieldArray3(nxc, nyc, nzc);
  d_Bzc = GPUFieldArray3(nxc, nyc, nzc);
  d_Bxn = GPUFieldArray3(nxn, nyn, nzn);
  d_Byn = GPUFieldArray3(nxn, nyn, nzn);
  d_Bzn = GPUFieldArray3(nxn, nyn, nzn);

  // ---- Charge / current densities (node-based, summed over species) ----
  d_rhon = GPUFieldArray3(nxn, nyn, nzn);
  d_rhoc = GPUFieldArray3(nxc, nyc, nzc);
  d_rhoh = GPUFieldArray3(nxc, nyc, nzc);
  d_Jx = GPUFieldArray3(nxn, nyn, nzn);
  d_Jy = GPUFieldArray3(nxn, nyn, nzn);
  d_Jz = GPUFieldArray3(nxn, nyn, nzn);
  d_Jxh = GPUFieldArray3(nxn, nyn, nzn);
  d_Jyh = GPUFieldArray3(nxn, nyn, nzn);
  d_Jzh = GPUFieldArray3(nxn, nyn, nzn);

  // ---- Per-species densities and currents (node-based) ----
  d_rhons = GPUFieldArray4(ns, nxn, nyn, nzn);
  d_Jxs = GPUFieldArray4(ns, nxn, nyn, nzn);
  d_Jys = GPUFieldArray4(ns, nxn, nyn, nzn);
  d_Jzs = GPUFieldArray4(ns, nxn, nyn, nzn);

  // ---- Pressure tensor (node-based, species-indexed) ----
  d_pXXsn = GPUFieldArray4(ns, nxn, nyn, nzn);
  d_pXYsn = GPUFieldArray4(ns, nxn, nyn, nzn);
  d_pXZsn = GPUFieldArray4(ns, nxn, nyn, nzn);
  d_pYYsn = GPUFieldArray4(ns, nxn, nyn, nzn);
  d_pYZsn = GPUFieldArray4(ns, nxn, nyn, nzn);
  d_pZZsn = GPUFieldArray4(ns, nxn, nyn, nzn);

  // ---- Potentials (center-based) ----
  d_PHI = GPUFieldArray3(nxc, nyc, nzc);
  d_PSI = GPUFieldArray3(nxc, nyc, nzc);

  // ---- External B (node-based) ----
  d_Bx_ext = GPUFieldArray3(nxn, nyn, nzn);
  d_By_ext = GPUFieldArray3(nxn, nyn, nzn);
  d_Bz_ext = GPUFieldArray3(nxn, nyn, nzn);

  // ---- Temporary / work arrays ----
  d_tempXC = GPUFieldArray3(nxc, nyc, nzc);
  d_tempYC = GPUFieldArray3(nxc, nyc, nzc);
  d_tempZC = GPUFieldArray3(nxc, nyc, nzc);
  d_tempXN = GPUFieldArray3(nxn, nyn, nzn);
  d_tempYN = GPUFieldArray3(nxn, nyn, nzn);
  d_tempZN = GPUFieldArray3(nxn, nyn, nzn);
  d_tempC = GPUFieldArray3(nxc, nyc, nzc);
  d_tempX = GPUFieldArray3(nxn, nyn, nzn);
  d_tempY = GPUFieldArray3(nxn, nyn, nzn);
  d_tempZ = GPUFieldArray3(nxn, nyn, nzn);
  d_temp2X = GPUFieldArray3(nxn, nyn, nzn);
  d_temp2Y = GPUFieldArray3(nxn, nyn, nzn);
  d_temp2Z = GPUFieldArray3(nxn, nyn, nzn);
  d_imageX = GPUFieldArray3(nxn, nyn, nzn);
  d_imageY = GPUFieldArray3(nxn, nyn, nzn);
  d_imageZ = GPUFieldArray3(nxn, nyn, nzn);
  d_Dx = GPUFieldArray3(nxn, nyn, nzn);
  d_Dy = GPUFieldArray3(nxn, nyn, nzn);
  d_Dz = GPUFieldArray3(nxn, nyn, nzn);
  d_vectX = GPUFieldArray3(nxn, nyn, nzn);
  d_vectY = GPUFieldArray3(nxn, nyn, nzn);
  d_vectZ = GPUFieldArray3(nxn, nyn, nzn);
  d_divC = GPUFieldArray3(nxc, nyc, nzc);

  // ---- divB cleaning work arrays ----
  d_divBwork = GPUFieldArray3(nxc, nyc, nzc);
  d_gradPSIX = GPUFieldArray3(nxn, nyn, nzn);
  d_gradPSIY = GPUFieldArray3(nxn, nyn, nzn);
  d_gradPSIZ = GPUFieldArray3(nxn, nyn, nzn);

  // ---- Krylov vectors ----
  const int nMaxwellKrylov = maxwellKrylovSize_;
  const int nPoissonKrylov = poissonKrylovSize_;
  d_xkrylovMaxwell = GPUKrylovVector(nMaxwellKrylov);
  d_bkrylovMaxwell = GPUKrylovVector(nMaxwellKrylov);
  d_xkrylovPoisson_B = GPUKrylovVector(nPoissonKrylov);
  d_bkrylovPoisson_B = GPUKrylovVector(nPoissonKrylov);
  d_xkrylovPoisson_E = GPUKrylovVector(nPoissonKrylov);
  d_bkrylovPoisson_E = GPUKrylovVector(nPoissonKrylov);

  // ---- calculateE work arrays ----
  d_divE_work = GPUFieldArray3(nxc, nyc, nzc);
  d_gradPHIX_work = GPUFieldArray3(nxn, nyn, nzn);
  d_gradPHIY_work = GPUFieldArray3(nxn, nyn, nzn);
  d_gradPHIZ_work = GPUFieldArray3(nxn, nyn, nzn);

  // ---- Poisson image work arrays ----
  d_poissonTemp = GPUFieldArray3(nxc, nyc, nzc);
  d_poissonIm = GPUFieldArray3(nxc, nyc, nzc);

  // ---- Smooth temp buffer ----
  d_smoothTemp = GPUFieldArray3(nxn, nyn, nzn);

  // ---- Device copy of qom ----
  cudaErrChk(cudaMalloc(&d_qom, ns * sizeof(cudaSolverType)));
  cudaErrChk(cudaMemcpy(d_qom, qom, ns * sizeof(cudaSolverType),
                        cudaMemcpyHostToDevice));

  // ---- BLAS reduction scratch ----
  // Need (m+2) doubles for gpuBatchedDotNorm in GMRES (m=20 → 22 doubles).
  // Also used as 1-double scratch for individual gpuDot/gpuNorm2 calls.
  cudaErrChk(
      cudaMalloc(&d_blasScratch, (GMRES_M + 2) * sizeof(cudaSolverType)));

  // ---- Pinned host buffers for GMRES reductions ----
  cudaErrChk(cudaHostAlloc(&h_gmresReduceLocal,
                           GMRES_MP1 * sizeof(cudaSolverType),
                           cudaHostAllocDefault));
  cudaErrChk(cudaHostAlloc(&h_gmresReduceGlobal,
                           GMRES_MP1 * sizeof(cudaSolverType),
                           cudaHostAllocDefault));
  cudaErrChk(cudaHostAlloc(&h_gmresH,
                           (size_t)GMRES_MP1 * GMRES_M * sizeof(cudaSolverType),
                           cudaHostAllocDefault));
  cudaErrChk(cudaHostAlloc(&h_gmresG, GMRES_MP1 * sizeof(cudaSolverType),
                           cudaHostAllocDefault));
  cudaErrChk(cudaHostAlloc(&h_gmresCS, GMRES_M * sizeof(cudaSolverType),
                           cudaHostAllocDefault));
  cudaErrChk(cudaHostAlloc(&h_gmresSN, GMRES_M * sizeof(cudaSolverType),
                           cudaHostAllocDefault));
  cudaErrChk(cudaHostAlloc(&h_gmresY, GMRES_MP1 * sizeof(cudaSolverType),
                           cudaHostAllocDefault));

  // ---- Persistent GMRES workspace ----
  const int nGMRESKrylov = std::max(nMaxwellKrylov, nPoissonKrylov);
  cudaErrChk(cudaMalloc(&d_gmresV, (size_t)GMRES_MP1 * nGMRESKrylov *
                                       sizeof(cudaSolverType)));
  cudaErrChk(
      cudaMalloc(&d_gmresW, (size_t)nGMRESKrylov * sizeof(cudaSolverType)));
  gmresVAlloc = static_cast<size_t>(GMRES_MP1) * nGMRESKrylov;

  // ---- FGMRES workspace (allocated on first FGMRES use, then reused) ----
  d_fgmresZ = nullptr;
  fgmresZAlloc = 0;

  // ---- Chebyshev workspace (4 Krylov-sized vectors, lazy alloc) ----
  d_chebY = nullptr;
  d_chebW = nullptr;
  d_chebZ = nullptr;
  d_chebTmp = nullptr;
  chebAlloc = 0;

  // ---- Dedicated non-blocking solver stream ----
  cudaErrChk(cudaStreamCreateWithFlags(&solverStream_, cudaStreamNonBlocking));

  // ---- Persistent batched halo-exchange buffers ----
  gpuAllocateHaloBuffers();

  rollback.fields = nullptr;
}

void EMfields3D::gpuSolverFree() {
  if (!gpuSolverAllocated_) {
    // Halo communication can be used without allocating the full GPU solver.
    gpuFreeHaloBuffers();
    return;
  }

  // No device allocation may be released while solver work can reference it.
  // gpuFreeHaloBuffers() owns the active split-exchange guard.
  if (solverStream_)
    cudaErrChk(cudaStreamSynchronize(solverStream_));

  // Free the shared communication storage while haloBuffersReady_ is still
  // available to fence any asynchronous final unpack.
  gpuFreeHaloBuffers();

  // Electric field
  d_Ex.free();
  d_Ey.free();
  d_Ez.free();
  d_Exth.free();
  d_Eyth.free();
  d_Ezth.free();

  // Magnetic field
  d_Bxc.free();
  d_Byc.free();
  d_Bzc.free();
  d_Bxn.free();
  d_Byn.free();
  d_Bzn.free();

  // Charge / current densities
  d_rhon.free();
  d_rhoc.free();
  d_rhoh.free();
  d_Jx.free();
  d_Jy.free();
  d_Jz.free();
  d_Jxh.free();
  d_Jyh.free();
  d_Jzh.free();

  // Per-species
  d_rhons.free();
  d_Jxs.free();
  d_Jys.free();
  d_Jzs.free();
  d_pXXsn.free();
  d_pXYsn.free();
  d_pXZsn.free();
  d_pYYsn.free();
  d_pYZsn.free();
  d_pZZsn.free();

  // Potentials
  d_PHI.free();
  d_PSI.free();

  // External B
  d_Bx_ext.free();
  d_By_ext.free();
  d_Bz_ext.free();

  // Temporary arrays
  d_tempXC.free();
  d_tempYC.free();
  d_tempZC.free();
  d_tempXN.free();
  d_tempYN.free();
  d_tempZN.free();
  d_tempC.free();
  d_tempX.free();
  d_tempY.free();
  d_tempZ.free();
  d_temp2X.free();
  d_temp2Y.free();
  d_temp2Z.free();
  d_imageX.free();
  d_imageY.free();
  d_imageZ.free();
  d_Dx.free();
  d_Dy.free();
  d_Dz.free();
  d_vectX.free();
  d_vectY.free();
  d_vectZ.free();
  d_divC.free();

  // divB cleaning
  d_divBwork.free();
  d_gradPSIX.free();
  d_gradPSIY.free();
  d_gradPSIZ.free();

  // Krylov
  d_xkrylovMaxwell.free();
  d_bkrylovMaxwell.free();
  d_xkrylovPoisson_B.free();
  d_bkrylovPoisson_B.free();
  d_xkrylovPoisson_E.free();
  d_bkrylovPoisson_E.free();

  // calculateE work
  d_divE_work.free();
  d_gradPHIX_work.free();
  d_gradPHIY_work.free();
  d_gradPHIZ_work.free();

  // Poisson image
  d_poissonTemp.free();
  d_poissonIm.free();

  // Smooth temp
  d_smoothTemp.free();

  // qom device copy
  if (d_qom) {
    cudaFree(d_qom);
    d_qom = nullptr;
  }
  if (d_blasScratch) {
    cudaFree(d_blasScratch);
    d_blasScratch = nullptr;
  }
  if (d_gmresV) {
    cudaFree(d_gmresV);
    d_gmresV = nullptr;
  }
  if (d_gmresW) {
    cudaFree(d_gmresW);
    d_gmresW = nullptr;
  }
  gmresVAlloc = 0;

  // Free Chebyshev workspace
  if (d_chebY) {
    cudaFree(d_chebY);
    d_chebY = nullptr;
  }
  if (d_chebW) {
    cudaFree(d_chebW);
    d_chebW = nullptr;
  }
  if (d_chebZ) {
    cudaFree(d_chebZ);
    d_chebZ = nullptr;
  }
  if (d_chebTmp) {
    cudaFree(d_chebTmp);
    d_chebTmp = nullptr;
  }
  chebAlloc = 0;

  // Free FGMRES workspace
  if (d_fgmresZ) {
    cudaFree(d_fgmresZ);
    d_fgmresZ = nullptr;
  }
  fgmresZAlloc = 0;

  // Free Block-Jacobi scratch
  if (d_bjScratch1) {
    cudaFree(d_bjScratch1);
    d_bjScratch1 = nullptr;
  }
  if (d_bjScratch2) {
    cudaFree(d_bjScratch2);
    d_bjScratch2 = nullptr;
  }
  bjScratchAlloc = 0;

  // Free pinned GMRES host buffers
  if (h_gmresReduceLocal) {
    cudaFreeHost(h_gmresReduceLocal);
    h_gmresReduceLocal = nullptr;
  }
  if (h_gmresReduceGlobal) {
    cudaFreeHost(h_gmresReduceGlobal);
    h_gmresReduceGlobal = nullptr;
  }
  if (h_gmresH) {
    cudaFreeHost(h_gmresH);
    h_gmresH = nullptr;
  }
  if (h_gmresG) {
    cudaFreeHost(h_gmresG);
    h_gmresG = nullptr;
  }
  if (h_gmresCS) {
    cudaFreeHost(h_gmresCS);
    h_gmresCS = nullptr;
  }
  if (h_gmresSN) {
    cudaFreeHost(h_gmresSN);
    h_gmresSN = nullptr;
  }
  if (h_gmresY) {
    cudaFreeHost(h_gmresY);
    h_gmresY = nullptr;
  }

  // Destroy solver stream
  if (solverStream_) {
    cudaErrChk(cudaStreamDestroy(solverStream_));
    solverStream_ = 0;
  }

  gpuSolverAllocated_ = false;
}

// =========================================================================
//  Persistent batched halo-exchange buffer management
// =========================================================================

void EMfields3D::gpuAllocateHaloBuffers() {
  if (haloBufsAllocated_)
    return;

  // Topology, layouts, counts, and launch geometry are immutable for this
  // EMfields3D instance.  Build them once before allocating their storage.
  gpuInitializeHaloPlans();
  haloBufsAllocated_ = true;

  try {
    // Node extents dominate center extents.  Each cached maximum covers face,
    // edge, and corner phases for one direction and one field.
    const HaloGeometryPlan& nodePlan = haloGeometryPlans_[HALO_NODE_OFFSET1];
    for (int d = 0; d < 6; ++d) {
      const size_t bytes = (size_t)nodePlan.maxBufferElements[d] *
                           HALO_MAX_BATCH * sizeof(cudaSolverType);
      cudaErrChk(cudaMalloc(&d_haloBuf_send_[d], bytes));
      cudaErrChk(cudaMalloc(&d_haloBuf_recv_[d], bytes));
    }
    cudaErrChk(
        cudaMalloc(&d_ptrArray_, HALO_MAX_BATCH * sizeof(cudaSolverType*)));
    cudaErrChk(cudaHostAlloc(&h_ptrArray_,
                             HALO_MAX_BATCH * sizeof(cudaSolverType*),
                             cudaHostAllocDefault));

#ifdef HALO_OVERLAP
    // Split exchange resources belong to the halo subsystem, so tests and
    // communication-only users do not need the full field solver.
    int leastPriority = 0;
    int greatestPriority = 0;
    cudaErrChk(
        cudaDeviceGetStreamPriorityRange(&leastPriority, &greatestPriority));
    cudaErrChk(cudaStreamCreateWithPriority(&haloStream_, cudaStreamNonBlocking,
                                            greatestPriority));
    cudaErrChk(
        cudaEventCreateWithFlags(&haloInputReady_, cudaEventDisableTiming));
    cudaErrChk(
        cudaEventCreateWithFlags(&haloPhaseReady_, cudaEventDisableTiming));
    cudaErrChk(
        cudaEventCreateWithFlags(&haloBuffersReady_, cudaEventDisableTiming));
#endif
  } catch (...) {
    gpuFreeHaloBuffers();
    throw;
  }
}

void EMfields3D::gpuFreeHaloBuffers() {
  if (!haloBufsAllocated_)
    return;

#ifdef HALO_OVERLAP
  if (haloExchange_.active) {
    cerr << "ERROR: freeing shared GPU halo buffers during an active exchange"
         << endl;
    MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    std::abort();
  }
  // A blocking or split exchange may have returned after queueing its final
  // unpack.  The shared buffers and device pointer table remain live until the
  // unified availability event completes.
  if (haloBuffersReady_)
    cudaErrChk(cudaEventSynchronize(haloBuffersReady_));
#endif

  for (int d = 0; d < 6; ++d) {
    if (d_haloBuf_send_[d]) {
      cudaFree(d_haloBuf_send_[d]);
      d_haloBuf_send_[d] = nullptr;
    }
    if (d_haloBuf_recv_[d]) {
      cudaFree(d_haloBuf_recv_[d]);
      d_haloBuf_recv_[d] = nullptr;
    }
  }
  if (d_ptrArray_) {
    cudaFree(d_ptrArray_);
    d_ptrArray_ = nullptr;
  }
  if (h_ptrArray_) {
    cudaFreeHost(h_ptrArray_);
    h_ptrArray_ = nullptr;
  }

#ifdef HALO_OVERLAP
  if (haloInputReady_) {
    cudaErrChk(cudaEventDestroy(haloInputReady_));
    haloInputReady_ = nullptr;
  }
  if (haloPhaseReady_) {
    cudaErrChk(cudaEventDestroy(haloPhaseReady_));
    haloPhaseReady_ = nullptr;
  }
  if (haloBuffersReady_) {
    cudaErrChk(cudaEventDestroy(haloBuffersReady_));
    haloBuffersReady_ = nullptr;
  }
  if (haloStream_) {
    cudaErrChk(cudaStreamDestroy(haloStream_));
    haloStream_ = 0;
  }
#endif

  haloBufsAllocated_ = false;
}

void EMfields3D::gpuSolverSyncH2D(cudaStream_t stream) {
  // Electric field
  d_Ex.copyFromHostAsync(Ex.fetch_arr(), stream);
  d_Ey.copyFromHostAsync(Ey.fetch_arr(), stream);
  d_Ez.copyFromHostAsync(Ez.fetch_arr(), stream);
  d_Exth.copyFromHostAsync(Exth.fetch_arr(), stream);
  d_Eyth.copyFromHostAsync(Eyth.fetch_arr(), stream);
  d_Ezth.copyFromHostAsync(Ezth.fetch_arr(), stream);

  // Magnetic field
  d_Bxc.copyFromHostAsync(Bxc.fetch_arr(), stream);
  d_Byc.copyFromHostAsync(Byc.fetch_arr(), stream);
  d_Bzc.copyFromHostAsync(Bzc.fetch_arr(), stream);
  d_Bxn.copyFromHostAsync(Bxn.fetch_arr(), stream);
  d_Byn.copyFromHostAsync(Byn.fetch_arr(), stream);
  d_Bzn.copyFromHostAsync(Bzn.fetch_arr(), stream);

  // Charge / current densities
  d_rhon.copyFromHostAsync(rhon.fetch_arr(), stream);
  d_rhoc.copyFromHostAsync(rhoc.fetch_arr(), stream);
  d_rhoh.copyFromHostAsync(rhoh.fetch_arr(), stream);
  d_Jx.copyFromHostAsync(Jx.fetch_arr(), stream);
  d_Jy.copyFromHostAsync(Jy.fetch_arr(), stream);
  d_Jz.copyFromHostAsync(Jz.fetch_arr(), stream);
  d_Jxh.copyFromHostAsync(Jxh.fetch_arr(), stream);
  d_Jyh.copyFromHostAsync(Jyh.fetch_arr(), stream);
  d_Jzh.copyFromHostAsync(Jzh.fetch_arr(), stream);

  // Per-species densities
  d_rhons.copyFromHostAsync(rhons.fetch_arr(), stream);
  d_Jxs.copyFromHostAsync(Jxs.fetch_arr(), stream);
  d_Jys.copyFromHostAsync(Jys.fetch_arr(), stream);
  d_Jzs.copyFromHostAsync(Jzs.fetch_arr(), stream);

  // Pressure tensor
  d_pXXsn.copyFromHostAsync(pXXsn.fetch_arr(), stream);
  d_pXYsn.copyFromHostAsync(pXYsn.fetch_arr(), stream);
  d_pXZsn.copyFromHostAsync(pXZsn.fetch_arr(), stream);
  d_pYYsn.copyFromHostAsync(pYYsn.fetch_arr(), stream);
  d_pYZsn.copyFromHostAsync(pYZsn.fetch_arr(), stream);
  d_pZZsn.copyFromHostAsync(pZZsn.fetch_arr(), stream);

  // Potentials
  d_PHI.copyFromHostAsync(PHI.fetch_arr(), stream);
  d_PSI.copyFromHostAsync(PSI.fetch_arr(), stream);

  // External B
  d_Bx_ext.copyFromHostAsync(Bx_ext.fetch_arr(), stream);
  d_By_ext.copyFromHostAsync(By_ext.fetch_arr(), stream);
  d_Bz_ext.copyFromHostAsync(Bz_ext.fetch_arr(), stream);

  cudaStreamSynchronize(stream);
}

void EMfields3D::gpuSolverSyncD2H(cudaStream_t stream) {
  // Only copy field data needed for I/O or particle mover
  d_Ex.copyToHostAsync(Ex.fetch_arr(), stream);
  d_Ey.copyToHostAsync(Ey.fetch_arr(), stream);
  d_Ez.copyToHostAsync(Ez.fetch_arr(), stream);
  d_Exth.copyToHostAsync(Exth.fetch_arr(), stream);
  d_Eyth.copyToHostAsync(Eyth.fetch_arr(), stream);
  d_Ezth.copyToHostAsync(Ezth.fetch_arr(), stream);

  d_Bxc.copyToHostAsync(Bxc.fetch_arr(), stream);
  d_Byc.copyToHostAsync(Byc.fetch_arr(), stream);
  d_Bzc.copyToHostAsync(Bzc.fetch_arr(), stream);
  d_Bxn.copyToHostAsync(Bxn.fetch_arr(), stream);
  d_Byn.copyToHostAsync(Byn.fetch_arr(), stream);
  d_Bzn.copyToHostAsync(Bzn.fetch_arr(), stream);

  d_rhon.copyToHostAsync(rhon.fetch_arr(), stream);
  d_rhoc.copyToHostAsync(rhoc.fetch_arr(), stream);
  d_Jx.copyToHostAsync(Jx.fetch_arr(), stream);
  d_Jy.copyToHostAsync(Jy.fetch_arr(), stream);
  d_Jz.copyToHostAsync(Jz.fetch_arr(), stream);

  // Per-species 4D arrays: copy species-by-species to match the
  // per-species cudaHostRegister granularity.  Bulk cudaMemcpyAsync
  // on a host buffer whose sub-regions are separately pinned can
  // cause "invalid argument" errors.
  {
    const size_t speciesSlice = (size_t)nxn * nyn * nzn;
    const size_t sliceBytes = speciesSlice * sizeof(cudaSolverType);
    for (int is = 0; is < ns; is++) {
      cudaErrChk(cudaMemcpyAsync(rhons.fetch_arr() + is * speciesSlice,
                                 d_rhons.speciesPtr(is), sliceBytes,
                                 cudaMemcpyDeviceToHost, stream));
      cudaErrChk(cudaMemcpyAsync(Jxs.fetch_arr() + is * speciesSlice,
                                 d_Jxs.speciesPtr(is), sliceBytes,
                                 cudaMemcpyDeviceToHost, stream));
      cudaErrChk(cudaMemcpyAsync(Jys.fetch_arr() + is * speciesSlice,
                                 d_Jys.speciesPtr(is), sliceBytes,
                                 cudaMemcpyDeviceToHost, stream));
      cudaErrChk(cudaMemcpyAsync(Jzs.fetch_arr() + is * speciesSlice,
                                 d_Jzs.speciesPtr(is), sliceBytes,
                                 cudaMemcpyDeviceToHost, stream));
      cudaErrChk(cudaMemcpyAsync(pXXsn.fetch_arr() + is * speciesSlice,
                                 d_pXXsn.speciesPtr(is), sliceBytes,
                                 cudaMemcpyDeviceToHost, stream));
      cudaErrChk(cudaMemcpyAsync(pXYsn.fetch_arr() + is * speciesSlice,
                                 d_pXYsn.speciesPtr(is), sliceBytes,
                                 cudaMemcpyDeviceToHost, stream));
      cudaErrChk(cudaMemcpyAsync(pXZsn.fetch_arr() + is * speciesSlice,
                                 d_pXZsn.speciesPtr(is), sliceBytes,
                                 cudaMemcpyDeviceToHost, stream));
      cudaErrChk(cudaMemcpyAsync(pYYsn.fetch_arr() + is * speciesSlice,
                                 d_pYYsn.speciesPtr(is), sliceBytes,
                                 cudaMemcpyDeviceToHost, stream));
      cudaErrChk(cudaMemcpyAsync(pYZsn.fetch_arr() + is * speciesSlice,
                                 d_pYZsn.speciesPtr(is), sliceBytes,
                                 cudaMemcpyDeviceToHost, stream));
      cudaErrChk(cudaMemcpyAsync(pZZsn.fetch_arr() + is * speciesSlice,
                                 d_pZZsn.speciesPtr(is), sliceBytes,
                                 cudaMemcpyDeviceToHost, stream));
    }
  }

  d_PHI.copyToHostAsync(PHI.fetch_arr(), stream);
  d_PSI.copyToHostAsync(PSI.fetch_arr(), stream);

  cudaStreamSynchronize(stream);
}

// =========================================================================
//  GPU Solver: physics method implementations
// =========================================================================

void EMfields3D::gpuMUdot(GPUFieldArray3& MUdotX, GPUFieldArray3& MUdotY,
                          GPUFieldArray3& MUdotZ, GPUFieldArray3& vX,
                          GPUFieldArray3& vY, GPUFieldArray3& vZ) {
  for (int is = 0; is < ns; is++) {
    double beta = 0.5 * qom[is] * dt / c;
    double prefactor = FourPI / 2.0 * delt * dt / c * qom[is];
    gpuMUdotSpecies(MUdotX.devPtr(), MUdotY.devPtr(), MUdotZ.devPtr(),
                    vX.devPtr(), vY.devPtr(), vZ.devPtr(), d_Bxn.devPtr(),
                    d_Byn.devPtr(), d_Bzn.devPtr(), d_Bx_ext.devPtr(),
                    d_By_ext.devPtr(), d_Bz_ext.devPtr(),
                    d_rhons.speciesPtr(is), beta, prefactor, nxn, nyn, nzn,
                    /*firstSpecies=*/(is == 0), solverStream_);
  }
}

void EMfields3D::gpuPIdot(GPUFieldArray3& PIX, GPUFieldArray3& PIY,
                          GPUFieldArray3& PIZ, GPUFieldArray3& vX,
                          GPUFieldArray3& vY, GPUFieldArray3& vZ, int is) {
  double beta = 0.5 * qom[is] * dt / c;
  gpuPIdotSpecies(PIX.devPtr(), PIY.devPtr(), PIZ.devPtr(), vX.devPtr(),
                  vY.devPtr(), vZ.devPtr(), d_Bxn.devPtr(), d_Byn.devPtr(),
                  d_Bzn.devPtr(), d_Bx_ext.devPtr(), d_By_ext.devPtr(),
                  d_Bz_ext.devPtr(), beta, nxn, nyn, nzn, solverStream_);
}

void EMfields3D::gpuSmooth(GPUFieldArray3& arr, int type) {
  if (Smooth == 1.0)
    return;

  const double alpha = Smooth;
  const double beta3D = (1.0 - alpha) / 6.0;

  int nx, ny, nz;
  if (type == 0) {
    nx = nxc;
    ny = nyc;
    nz = nzc;
  } else {
    nx = nxn;
    ny = nyn;
    nz = nzn;
  }

  size_t fieldSize = (size_t)nx * ny * nz;
  bool isCenter = (type == 0);

  for (int icount = 1; icount < SmoothNiter + 1; icount++) {
#ifdef HALO_OVERLAP
    cudaSolverType* ptr1[1] = {arr.devPtr()};
    gpuBatchedHaloBeginExchange(ptr1, 1, nx, ny, nz, isCenter, true, true,
                                solverStream_);
    gpuSmoothStep_interior(d_smoothTemp.devPtr(), arr.devPtr(), nx, ny, nz,
                           alpha, beta3D, solverStream_);
    gpuBatchedHaloEndExchange();
    gpuBCface_P(nx, ny, nz, arr, 2, 2, 2, 2, 2, 2, &_vct, solverStream_);
    gpuSmoothStep_boundary(d_smoothTemp.devPtr(), arr.devPtr(), nx, ny, nz,
                           alpha, beta3D, solverStream_);
#else
    if (type == 0)
      gpuCommunicateCenterBoxStencilBC_P(nx, ny, nz, arr, 2, 2, 2, 2, 2, 2);
    else
      gpuCommunicateNodeBoxStencilBC_P(nx, ny, nz, arr, 2, 2, 2, 2, 2, 2);
    gpuSmoothStep(d_smoothTemp.devPtr(), arr.devPtr(), nx, ny, nz, alpha,
                  beta3D, solverStream_);
#endif
    gpuEq(arr.devPtr(), d_smoothTemp.devPtr(), fieldSize, solverStream_);
  }
}

void EMfields3D::gpuSmoothE() {
  if (Smooth == 1.0)
    return;

  const Collective* col = &get_col();
  const double alpha = Smooth;
  const double beta3D = (1.0 - alpha) / 6.0;
  size_t nodeSize = (size_t)nxn * nyn * nzn;

  for (int icount = 1; icount < SmoothNiter + 1; icount++) {
#ifdef HALO_OVERLAP
    cudaSolverType* eptrs[3] = {d_Ex.devPtr(), d_Ey.devPtr(), d_Ez.devPtr()};
    gpuBatchedHaloBeginExchange(eptrs, 3, nxn, nyn, nzn, false, true, false,
                                solverStream_);
    gpuSmoothStep_interior(d_tempX.devPtr(), d_Ex.devPtr(), nxn, nyn, nzn,
                           alpha, beta3D, solverStream_);
    gpuSmoothStep_interior(d_tempY.devPtr(), d_Ey.devPtr(), nxn, nyn, nzn,
                           alpha, beta3D, solverStream_);
    gpuSmoothStep_interior(d_tempZ.devPtr(), d_Ez.devPtr(), nxn, nyn, nzn,
                           alpha, beta3D, solverStream_);
    gpuBatchedHaloEndExchange();
    gpuBCface(nxn, nyn, nzn, d_Ex, col->bcEx[0], col->bcEx[1], col->bcEx[2],
              col->bcEx[3], col->bcEx[4], col->bcEx[5], &_vct, solverStream_);
    gpuBCface(nxn, nyn, nzn, d_Ey, col->bcEy[0], col->bcEy[1], col->bcEy[2],
              col->bcEy[3], col->bcEy[4], col->bcEy[5], &_vct, solverStream_);
    gpuBCface(nxn, nyn, nzn, d_Ez, col->bcEz[0], col->bcEz[1], col->bcEz[2],
              col->bcEz[3], col->bcEz[4], col->bcEz[5], &_vct, solverStream_);
    gpuSmoothStep_boundary(d_tempX.devPtr(), d_Ex.devPtr(), nxn, nyn, nzn,
                           alpha, beta3D, solverStream_);
    gpuSmoothStep_boundary(d_tempY.devPtr(), d_Ey.devPtr(), nxn, nyn, nzn,
                           alpha, beta3D, solverStream_);
    gpuSmoothStep_boundary(d_tempZ.devPtr(), d_Ez.devPtr(), nxn, nyn, nzn,
                           alpha, beta3D, solverStream_);
#else
    // Batched: 3 fields in 1 MPI round instead of 3 sequential exchanges
    gpuCommunicateNodeBoxStencilBC_3mixed(nxn, nyn, nzn, d_Ex, col->bcEx, d_Ey,
                                          col->bcEy, d_Ez, col->bcEz);
    gpuSmoothStep(d_tempX.devPtr(), d_Ex.devPtr(), nxn, nyn, nzn, alpha, beta3D,
                  solverStream_);
    gpuSmoothStep(d_tempY.devPtr(), d_Ey.devPtr(), nxn, nyn, nzn, alpha, beta3D,
                  solverStream_);
    gpuSmoothStep(d_tempZ.devPtr(), d_Ez.devPtr(), nxn, nyn, nzn, alpha, beta3D,
                  solverStream_);
#endif

    gpuEq(d_Ex.devPtr(), d_tempX.devPtr(), nodeSize, solverStream_);
    gpuEq(d_Ey.devPtr(), d_tempY.devPtr(), nodeSize, solverStream_);
    gpuEq(d_Ez.devPtr(), d_tempZ.devPtr(), nodeSize, solverStream_);
  }
}

void EMfields3D::gpuSmooth3(GPUFieldArray3& a1, GPUFieldArray3& a2,
                            GPUFieldArray3& a3, int type) {
  if (Smooth == 1.0)
    return;

  const double alpha = Smooth;
  const double beta3D = (1.0 - alpha) / 6.0;

  int nx, ny, nz;
  if (type == 0) {
    nx = nxc;
    ny = nyc;
    nz = nzc;
  } else {
    nx = nxn;
    ny = nyn;
    nz = nzn;
  }

  size_t fieldSize = (size_t)nx * ny * nz;
  bool isCenter = (type == 0);

  for (int icount = 1; icount < SmoothNiter + 1; icount++) {
#ifdef HALO_OVERLAP
    cudaSolverType* s3ptrs[3] = {a1.devPtr(), a2.devPtr(), a3.devPtr()};
    gpuBatchedHaloBeginExchange(s3ptrs, 3, nx, ny, nz, isCenter, true, true,
                                solverStream_);
    gpuSmoothStep_interior(d_temp2X.devPtr(), a1.devPtr(), nx, ny, nz, alpha,
                           beta3D, solverStream_);
    gpuSmoothStep_interior(d_temp2Y.devPtr(), a2.devPtr(), nx, ny, nz, alpha,
                           beta3D, solverStream_);
    gpuSmoothStep_interior(d_temp2Z.devPtr(), a3.devPtr(), nx, ny, nz, alpha,
                           beta3D, solverStream_);
    gpuBatchedHaloEndExchange();
    gpuBCface_P(nx, ny, nz, a1, 2, 2, 2, 2, 2, 2, &_vct, solverStream_);
    gpuBCface_P(nx, ny, nz, a2, 2, 2, 2, 2, 2, 2, &_vct, solverStream_);
    gpuBCface_P(nx, ny, nz, a3, 2, 2, 2, 2, 2, 2, &_vct, solverStream_);
    gpuSmoothStep_boundary(d_temp2X.devPtr(), a1.devPtr(), nx, ny, nz, alpha,
                           beta3D, solverStream_);
    gpuSmoothStep_boundary(d_temp2Y.devPtr(), a2.devPtr(), nx, ny, nz, alpha,
                           beta3D, solverStream_);
    gpuSmoothStep_boundary(d_temp2Z.devPtr(), a3.devPtr(), nx, ny, nz, alpha,
                           beta3D, solverStream_);
#else
    // Batched: 3 fields in 1 MPI round
    if (type == 0)
      gpuCommunicateCenterBoxStencilBC_P_3(nx, ny, nz, a1, a2, a3, 2, 2, 2, 2,
                                           2, 2);
    else
      gpuCommunicateNodeBoxStencilBC_P_3(nx, ny, nz, a1, a2, a3, 2, 2, 2, 2, 2,
                                         2);
    gpuSmoothStep(d_temp2X.devPtr(), a1.devPtr(), nx, ny, nz, alpha, beta3D,
                  solverStream_);
    gpuSmoothStep(d_temp2Y.devPtr(), a2.devPtr(), nx, ny, nz, alpha, beta3D,
                  solverStream_);
    gpuSmoothStep(d_temp2Z.devPtr(), a3.devPtr(), nx, ny, nz, alpha, beta3D,
                  solverStream_);
#endif

    gpuEq(a1.devPtr(), d_temp2X.devPtr(), fieldSize, solverStream_);
    gpuEq(a2.devPtr(), d_temp2Y.devPtr(), fieldSize, solverStream_);
    gpuEq(a3.devPtr(), d_temp2Z.devPtr(), fieldSize, solverStream_);
  }
}

void EMfields3D::gpuPerfectConductorLeft(GPUFieldArray3& imX,
                                         GPUFieldArray3& imY,
                                         GPUFieldArray3& imZ,
                                         GPUFieldArray3& vX, GPUFieldArray3& vY,
                                         GPUFieldArray3& vZ, int dir) {
  ::gpuPerfectConductorLeft(
      imX.devPtr(), imY.devPtr(), imZ.devPtr(), vX.devPtr(), vY.devPtr(),
      vZ.devPtr(), d_Ex.devPtr(), d_Ey.devPtr(), d_Ez.devPtr(), d_Jxh.devPtr(),
      d_Jyh.devPtr(), d_Jzh.devPtr(), d_Bxn.devPtr(), d_Byn.devPtr(),
      d_Bzn.devPtr(), d_Bx_ext.devPtr(), d_By_ext.devPtr(), d_Bz_ext.devPtr(),
      d_rhons.devPtr(), d_qom, ns, dt, c, th, FourPI, delt, nxn, nyn, nzn, dir,
      solverStream_);
}

void EMfields3D::gpuPerfectConductorRight(
    GPUFieldArray3& imX, GPUFieldArray3& imY, GPUFieldArray3& imZ,
    GPUFieldArray3& vX, GPUFieldArray3& vY, GPUFieldArray3& vZ, int dir) {
  ::gpuPerfectConductorRight(
      imX.devPtr(), imY.devPtr(), imZ.devPtr(), vX.devPtr(), vY.devPtr(),
      vZ.devPtr(), d_Ex.devPtr(), d_Ey.devPtr(), d_Ez.devPtr(), d_Jxh.devPtr(),
      d_Jyh.devPtr(), d_Jzh.devPtr(), d_Bxn.devPtr(), d_Byn.devPtr(),
      d_Bzn.devPtr(), d_Bx_ext.devPtr(), d_By_ext.devPtr(), d_Bz_ext.devPtr(),
      d_rhons.devPtr(), d_qom, ns, dt, c, th, FourPI, delt, nxn, nyn, nzn, dir,
      solverStream_);
}

// =========================================================================
//  Fused triple Laplacian: 3 independent lap(fieldN) with ONE halo exchange
//  Uses 9 center-sized scratch arrays:
//    fieldA → d_tempXC / d_tempYC / d_tempZC
//    fieldB → d_divC   / d_poissonTemp / d_poissonIm
//    fieldC → d_divBwork / d_divE_work / d_tempC
// =========================================================================

void EMfields3D::gpuLapN2N_3(GPUFieldArray3& lapA, GPUFieldArray3& fieldA,
                             GPUFieldArray3& lapB, GPUFieldArray3& fieldB,
                             GPUFieldArray3& lapC, GPUFieldArray3& fieldC) {
  const Grid* grid = &get_grid();
  double _invdx = grid->get_invdx();
  double _invdy = grid->get_invdy();
  double _invdz = grid->get_invdz();

  // Gradient A → d_tempXC, d_tempYC, d_tempZC
  gpuGradN2C(d_tempXC.devPtr(), d_tempYC.devPtr(), d_tempZC.devPtr(),
             fieldA.devPtr(), nxc, nyc, nzc, _invdx, _invdy, _invdz,
             solverStream_);

  // Gradient B → d_divC, d_poissonTemp, d_poissonIm  (center-sized scratch)
  gpuGradN2C(d_divC.devPtr(), d_poissonTemp.devPtr(), d_poissonIm.devPtr(),
             fieldB.devPtr(), nxc, nyc, nzc, _invdx, _invdy, _invdz,
             solverStream_);

  // Gradient C → d_divBwork, d_divE_work, d_tempC  (center-sized scratch)
  gpuGradN2C(d_divBwork.devPtr(), d_divE_work.devPtr(), d_tempC.devPtr(),
             fieldC.devPtr(), nxc, nyc, nzc, _invdx, _invdy, _invdz,
             solverStream_);

#ifdef HALO_OVERLAP
  // ---- Begin halo exchange: pack faces + post MPI ----
  cudaSolverType* ptrs9[9] = {
      d_tempXC.devPtr(),   d_tempYC.devPtr(),      d_tempZC.devPtr(),
      d_divC.devPtr(),     d_poissonTemp.devPtr(), d_poissonIm.devPtr(),
      d_divBwork.devPtr(), d_divE_work.devPtr(),   d_tempC.devPtr()};
  gpuBatchedHaloBeginExchange(ptrs9, 9, nxc, nyc, nzc, true, false, false,
                              solverStream_);

  // ---- Interior divC2N while MPI is in flight ----
  gpuDivC2N_interior(lapA.devPtr(), d_tempXC.devPtr(), d_tempYC.devPtr(),
                     d_tempZC.devPtr(), nxn, nyn, nzn, _invdx, _invdy, _invdz,
                     solverStream_);
  gpuDivC2N_interior(lapB.devPtr(), d_divC.devPtr(), d_poissonTemp.devPtr(),
                     d_poissonIm.devPtr(), nxn, nyn, nzn, _invdx, _invdy,
                     _invdz, solverStream_);
  gpuDivC2N_interior(lapC.devPtr(), d_divBwork.devPtr(), d_divE_work.devPtr(),
                     d_tempC.devPtr(), nxn, nyn, nzn, _invdx, _invdy, _invdz,
                     solverStream_);

  // ---- End halo exchange: MPI_Waitall + unpack + edges/corners ----
  gpuBatchedHaloEndExchange();

  // ---- BC face application (type 1 on all faces) ----
  gpuBCface(nxc, nyc, nzc, d_tempXC, 1, 1, 1, 1, 1, 1, &_vct, solverStream_);
  gpuBCface(nxc, nyc, nzc, d_tempYC, 1, 1, 1, 1, 1, 1, &_vct, solverStream_);
  gpuBCface(nxc, nyc, nzc, d_tempZC, 1, 1, 1, 1, 1, 1, &_vct, solverStream_);
  gpuBCface(nxc, nyc, nzc, d_divC, 1, 1, 1, 1, 1, 1, &_vct, solverStream_);
  gpuBCface(nxc, nyc, nzc, d_poissonTemp, 1, 1, 1, 1, 1, 1, &_vct,
            solverStream_);
  gpuBCface(nxc, nyc, nzc, d_poissonIm, 1, 1, 1, 1, 1, 1, &_vct, solverStream_);
  gpuBCface(nxc, nyc, nzc, d_divBwork, 1, 1, 1, 1, 1, 1, &_vct, solverStream_);
  gpuBCface(nxc, nyc, nzc, d_divE_work, 1, 1, 1, 1, 1, 1, &_vct, solverStream_);
  gpuBCface(nxc, nyc, nzc, d_tempC, 1, 1, 1, 1, 1, 1, &_vct, solverStream_);

  // ---- Boundary divC2N (ghost + BC data now available) ----
  gpuDivC2N_boundary(lapA.devPtr(), d_tempXC.devPtr(), d_tempYC.devPtr(),
                     d_tempZC.devPtr(), nxn, nyn, nzn, _invdx, _invdy, _invdz,
                     solverStream_);
  gpuDivC2N_boundary(lapB.devPtr(), d_divC.devPtr(), d_poissonTemp.devPtr(),
                     d_poissonIm.devPtr(), nxn, nyn, nzn, _invdx, _invdy,
                     _invdz, solverStream_);
  gpuDivC2N_boundary(lapC.devPtr(), d_divBwork.devPtr(), d_divE_work.devPtr(),
                     d_tempC.devPtr(), nxn, nyn, nzn, _invdx, _invdy, _invdz,
                     solverStream_);

#else
  // ---- Original blocking path ----
  gpuCommunicateCenterBC_9(nxc, nyc, nzc, d_tempXC, d_tempYC, d_tempZC, d_divC,
                           d_poissonTemp, d_poissonIm, d_divBwork, d_divE_work,
                           d_tempC, 1, 1, 1, 1, 1, 1);

  // Divergence A
  gpuDivC2N(lapA.devPtr(), d_tempXC.devPtr(), d_tempYC.devPtr(),
            d_tempZC.devPtr(), nxn, nyn, nzn, _invdx, _invdy, _invdz,
            solverStream_);

  // Divergence B
  gpuDivC2N(lapB.devPtr(), d_divC.devPtr(), d_poissonTemp.devPtr(),
            d_poissonIm.devPtr(), nxn, nyn, nzn, _invdx, _invdy, _invdz,
            solverStream_);

  // Divergence C
  gpuDivC2N(lapC.devPtr(), d_divBwork.devPtr(), d_divE_work.devPtr(),
            d_tempC.devPtr(), nxn, nyn, nzn, _invdx, _invdy, _invdz,
            solverStream_);
#endif
}

// =========================================================================
//  GPU MaxwellImage:  im = A * vector  (Krylov ↔ Krylov)
// =========================================================================

void EMfields3D::gpuMaxwellImage(cudaSolverType* d_im,
                                 cudaSolverType* d_vector) {
  const VirtualTopology3D* vct = &get_vct();
  const Grid* grid = &get_grid();
  double _invdx = grid->get_invdx();
  double _invdy = grid->get_invdy();
  double _invdz = grid->get_invdz();

  size_t nodeSize = (size_t)nxn * nyn * nzn;

  // Zero work arrays (9 memsets batched)
  cudaSolverType* zptrs[9] = {
      d_imageX.devPtr(), d_imageY.devPtr(), d_imageZ.devPtr(),
      d_tempX.devPtr(),  d_tempY.devPtr(),  d_tempZ.devPtr(),
      d_Dx.devPtr(),     d_Dy.devPtr(),     d_Dz.devPtr()};
  gpuSetAll0_N(zptrs, 9, nodeSize, solverStream_);

  // Krylov → physical space
  gpuSolver2Phys3(d_vectX.devPtr(), d_vectY.devPtr(), d_vectZ.devPtr(),
                  d_vector, nxn, nyn, nzn, solverStream_);

  // Laplacian: image = -lap(vect)  (fused: 3 Laps with 1 halo exchange)
  gpuLapN2N_3(d_imageX, d_vectX, d_imageY, d_vectY, d_imageZ, d_vectZ);
  gpuNeg3(d_imageX.devPtr(), d_imageY.devPtr(), d_imageZ.devPtr(), nodeSize,
          solverStream_);

  // MUdot: D = μ·vect
  gpuMUdot(d_Dx, d_Dy, d_Dz, d_vectX, d_vectY, d_vectZ);

  // div(D) on centers
  gpuDivN2C(d_divC.devPtr(), d_Dx.devPtr(), d_Dy.devPtr(), d_Dz.devPtr(), nxc,
            nyc, nzc, _invdx, _invdy, _invdz, solverStream_);

#ifdef HALO_OVERLAP
  // ---- Begin halo on divC ----
  cudaSolverType* ptr1[1] = {d_divC.devPtr()};
  gpuBatchedHaloBeginExchange(ptr1, 1, nxc, nyc, nzc, true, false, false,
                              solverStream_);

  // ---- Interior gradC2N while MPI is in flight ----
  gpuGradC2N_interior(d_tempX.devPtr(), d_tempY.devPtr(), d_tempZ.devPtr(),
                      d_divC.devPtr(), nxn, nyn, nzn, _invdx, _invdy, _invdz,
                      solverStream_);

  // ---- End halo ----
  gpuBatchedHaloEndExchange();
  gpuBCface(nxc, nyc, nzc, d_divC, 2, 2, 2, 2, 2, 2, &_vct, solverStream_);

  // ---- Boundary gradC2N ----
  gpuGradC2N_boundary(d_tempX.devPtr(), d_tempY.devPtr(), d_tempZ.devPtr(),
                      d_divC.devPtr(), nxn, nyn, nzn, _invdx, _invdy, _invdz,
                      solverStream_);
#else
  // Communicate divC
  gpuCommunicateCenterBC(nxc, nyc, nzc, d_divC, 2, 2, 2, 2, 2, 2);

  // grad(divC) on nodes
  gpuGradC2N(d_tempX.devPtr(), d_tempY.devPtr(), d_tempZ.devPtr(),
             d_divC.devPtr(), nxn, nyn, nzn, _invdx, _invdy, _invdz,
             solverStream_);
#endif

  // image -= temp  (fused triple)
  gpuSub3(d_imageX.devPtr(), d_tempX.devPtr(), d_imageY.devPtr(),
          d_tempY.devPtr(), d_imageZ.devPtr(), d_tempZ.devPtr(), nodeSize,
          solverStream_);

  // Scale by delt²  (fused triple)
  gpuScale3(d_imageX.devPtr(), d_imageY.devPtr(), d_imageZ.devPtr(),
            delt * delt, nodeSize, solverStream_);

  // Add ε·E: image += D + vect  (fused: 6 gpuSum → 1 gpuSumAddTwo3)
  gpuSumAddTwo3(d_imageX.devPtr(), d_Dx.devPtr(), d_vectX.devPtr(),
                d_imageY.devPtr(), d_Dy.devPtr(), d_vectY.devPtr(),
                d_imageZ.devPtr(), d_Dz.devPtr(), d_vectZ.devPtr(), nodeSize,
                solverStream_);

  // Perfect conductor BCs
  if (vct->getXleft_neighbor() == MPI_PROC_NULL && bcEMfaceXleft == 0)
    gpuPerfectConductorLeft(d_imageX, d_imageY, d_imageZ, d_vectX, d_vectY,
                            d_vectZ, 0);
  if (vct->getXright_neighbor() == MPI_PROC_NULL && bcEMfaceXright == 0)
    gpuPerfectConductorRight(d_imageX, d_imageY, d_imageZ, d_vectX, d_vectY,
                             d_vectZ, 0);
  if (vct->getYleft_neighbor() == MPI_PROC_NULL && bcEMfaceYleft == 0)
    gpuPerfectConductorLeft(d_imageX, d_imageY, d_imageZ, d_vectX, d_vectY,
                            d_vectZ, 1);
  if (vct->getYright_neighbor() == MPI_PROC_NULL && bcEMfaceYright == 0)
    gpuPerfectConductorRight(d_imageX, d_imageY, d_imageZ, d_vectX, d_vectY,
                             d_vectZ, 1);
  if (vct->getZleft_neighbor() == MPI_PROC_NULL && bcEMfaceZleft == 0)
    gpuPerfectConductorLeft(d_imageX, d_imageY, d_imageZ, d_vectX, d_vectY,
                            d_vectZ, 2);
  if (vct->getZright_neighbor() == MPI_PROC_NULL && bcEMfaceZright == 0)
    gpuPerfectConductorRight(d_imageX, d_imageY, d_imageZ, d_vectX, d_vectY,
                             d_vectZ, 2);

  // OpenBC: apply inflow BCs to GMRES image if enabled
  if (get_col().getApplyInflowBcsEImage())
    gpuOpenBoundaryInflowEImage(
        d_imageX.devPtr(), d_imageY.devPtr(), d_imageZ.devPtr(),
        d_vectX.devPtr(), d_vectY.devPtr(), d_vectZ.devPtr(), nxn, nyn, nzn);

  // Physical → Krylov space
  gpuPhys2Solver3(d_im, d_imageX.devPtr(), d_imageY.devPtr(), d_imageZ.devPtr(),
                  nxn, nyn, nzn, solverStream_);
}

// =========================================================================
//  GPU MaxwellImageLocal:  communication-free  im = A * vector
//
//  Computes a local approximation to gpuMaxwellImage:
//    • NO MPI communication  (halo exchanges skipped)
//    • Ghost cells are treated as zero
//    • Physical boundary image corrections are still applied locally
//
//  Intended for use as a GPU-local preconditioner in FGMRES.
//
//  Pipeline (2 + ns kernel launches):
//    1. gpuSolver2Phys3         — unpack Krylov → vectX/Y/Z
//    2. gpuMUdot                — D = μ̂·E  (ns species sub-launches)
//    3. gpuMaxwellLocalCenterOps — fused 3×gradN2C + divN2C
//    4. gpuMaxwellLocalNodeFused — fused 3×divC2N + gradC2N
//                                  + arithmetic + phys2solver → d_im
//
//  Scratch arrays used (all pre-allocated in gpuSolverAllocate):
//    Node-sized:  d_vectX/Y/Z, d_Dx/Y/Z           (input/work)
//    Center-sized: d_tempXC/YC/ZC                   (grad Ex)
//                  d_divC, d_poissonTemp, d_poissonIm (grad Ey)
//                  d_divBwork, d_divE_work, d_tempC   (grad Ez)
//                  d_imageX (first centSize elems)     (div D)
// =========================================================================

void EMfields3D::gpuMaxwellImageLocal(cudaSolverType* d_im,
                                      cudaSolverType* d_vector) {
  const VirtualTopology3D* vct = &get_vct();
  const Grid* grid = &get_grid();
  double _invdx = grid->get_invdx();
  double _invdy = grid->get_invdy();
  double _invdz = grid->get_invdz();

  size_t centSize = (size_t)nxc * nyc * nzc;

  // ---- Step 0: Zero center-sized scratch arrays ----
  // Ghost cells of center arrays MUST be zero since we skip communication.
  // gpuGradN2C / gpuDivN2C only write interior cells; ghost cells must be
  // pre-zeroed so the subsequent divC2N / gradC2N stencils read zeros at
  // subdomain boundaries.
  cudaErrChk(cudaMemsetAsync(d_tempXC.devPtr(), 0,
                             centSize * sizeof(cudaSolverType), solverStream_));
  cudaErrChk(cudaMemsetAsync(d_tempYC.devPtr(), 0,
                             centSize * sizeof(cudaSolverType), solverStream_));
  cudaErrChk(cudaMemsetAsync(d_tempZC.devPtr(), 0,
                             centSize * sizeof(cudaSolverType), solverStream_));
  cudaErrChk(cudaMemsetAsync(d_divC.devPtr(), 0,
                             centSize * sizeof(cudaSolverType), solverStream_));
  cudaErrChk(cudaMemsetAsync(d_poissonTemp.devPtr(), 0,
                             centSize * sizeof(cudaSolverType), solverStream_));
  cudaErrChk(cudaMemsetAsync(d_poissonIm.devPtr(), 0,
                             centSize * sizeof(cudaSolverType), solverStream_));
  cudaErrChk(cudaMemsetAsync(d_divBwork.devPtr(), 0,
                             centSize * sizeof(cudaSolverType), solverStream_));
  cudaErrChk(cudaMemsetAsync(d_divE_work.devPtr(), 0,
                             centSize * sizeof(cudaSolverType), solverStream_));
  cudaErrChk(cudaMemsetAsync(d_tempC.devPtr(), 0,
                             centSize * sizeof(cudaSolverType), solverStream_));
  // 10th center array: reuse first centSize doubles of d_imageX (node-sized, >
  // centSize)
  cudaErrChk(cudaMemsetAsync(d_imageX.devPtr(), 0,
                             centSize * sizeof(cudaSolverType), solverStream_));

  // ---- Step 1: Krylov → physical space ----
  gpuSolver2Phys3(d_vectX.devPtr(), d_vectY.devPtr(), d_vectZ.devPtr(),
                  d_vector, nxn, nyn, nzn, solverStream_);

  // ---- Step 2: MUdot: D = μ̂·E ----
  // Purely local (no communication), loops over species internally.
  gpuMUdot(d_Dx, d_Dy, d_Dz, d_vectX, d_vectY, d_vectZ);

  // ---- Step 3: Fused center-level operations ----
  // Computes 3×gradN2C(Ex,Ey,Ez) → 9 center arrays
  //        + divN2C(Dx,Dy,Dz)    → 1 center array
  // All in a single kernel launch.
  gpuMaxwellLocalCenterOps(
      d_tempXC.devPtr(), d_tempYC.devPtr(), d_tempZC.devPtr(),       // grad(Ex)
      d_divC.devPtr(), d_poissonTemp.devPtr(), d_poissonIm.devPtr(), // grad(Ey)
      d_divBwork.devPtr(), d_divE_work.devPtr(), d_tempC.devPtr(),   // grad(Ez)
      d_imageX.devPtr(), // div(D) — stored in first centSize of d_imageX
      d_vectX.devPtr(), d_vectY.devPtr(), d_vectZ.devPtr(), d_Dx.devPtr(),
      d_Dy.devPtr(), d_Dz.devPtr(), nxc, nyc, nzc, _invdx, _invdy, _invdz,
      solverStream_);

  // ---- Step 4: Fused node-level operations + Krylov packing ----
  // For each interior node:
  //   lapX = divC2N(grad(Ex))       — from 3 center arrays
  //   lapY = divC2N(grad(Ey))       — from 3 center arrays
  //   lapZ = divC2N(grad(Ez))       — from 3 center arrays
  //   gdivX/Y/Z = gradC2N(div(D))  — from 1 center array
  //   im = dt²·(-lap - gdiv) + D + E
  // Output is packed into Krylov space first, then boundary corrections are
  // applied after unpacking to node arrays.
  gpuMaxwellLocalNodeFused(
      d_im, d_tempXC.devPtr(), d_tempYC.devPtr(), d_tempZC.devPtr(), // grad(Ex)
      d_divC.devPtr(), d_poissonTemp.devPtr(), d_poissonIm.devPtr(), // grad(Ey)
      d_divBwork.devPtr(), d_divE_work.devPtr(), d_tempC.devPtr(),   // grad(Ez)
      d_imageX.devPtr(),                                             // div(D)
      d_vectX.devPtr(), d_vectY.devPtr(), d_vectZ.devPtr(), d_Dx.devPtr(),
      d_Dy.devPtr(), d_Dz.devPtr(), nxn, nyn, nzn, _invdx, _invdy, _invdz,
      delt * delt, solverStream_);

  // Match the CPU local operator: enforce boundary image corrections on the
  // unpacked node fields, then repack to Krylov space.
  gpuSolver2Phys3(d_imageX.devPtr(), d_imageY.devPtr(), d_imageZ.devPtr(), d_im,
                  nxn, nyn, nzn, solverStream_);

  if (vct->getXleft_neighbor() == MPI_PROC_NULL && bcEMfaceXleft == 0)
    gpuPerfectConductorLeft(d_imageX, d_imageY, d_imageZ, d_vectX, d_vectY,
                            d_vectZ, 0);
  if (vct->getXright_neighbor() == MPI_PROC_NULL && bcEMfaceXright == 0)
    gpuPerfectConductorRight(d_imageX, d_imageY, d_imageZ, d_vectX, d_vectY,
                             d_vectZ, 0);
  if (vct->getYleft_neighbor() == MPI_PROC_NULL && bcEMfaceYleft == 0)
    gpuPerfectConductorLeft(d_imageX, d_imageY, d_imageZ, d_vectX, d_vectY,
                            d_vectZ, 1);
  if (vct->getYright_neighbor() == MPI_PROC_NULL && bcEMfaceYright == 0)
    gpuPerfectConductorRight(d_imageX, d_imageY, d_imageZ, d_vectX, d_vectY,
                             d_vectZ, 1);
  if (vct->getZleft_neighbor() == MPI_PROC_NULL && bcEMfaceZleft == 0)
    gpuPerfectConductorLeft(d_imageX, d_imageY, d_imageZ, d_vectX, d_vectY,
                            d_vectZ, 2);
  if (vct->getZright_neighbor() == MPI_PROC_NULL && bcEMfaceZright == 0)
    gpuPerfectConductorRight(d_imageX, d_imageY, d_imageZ, d_vectX, d_vectY,
                             d_vectZ, 2);

  if (get_col().getApplyInflowBcsEImage())
    gpuOpenBoundaryInflowEImage(
        d_imageX.devPtr(), d_imageY.devPtr(), d_imageZ.devPtr(),
        d_vectX.devPtr(), d_vectY.devPtr(), d_vectZ.devPtr(), nxn, nyn, nzn);

  gpuPhys2Solver3(d_im, d_imageX.devPtr(), d_imageY.devPtr(), d_imageZ.devPtr(),
                  nxn, nyn, nzn, solverStream_);
}

// =========================================================================
//  GPU Chebyshev: estimate λ_max via power iteration (Rayleigh quotient)
// =========================================================================

cudaSolverType EMfields3D::gpuEstimateMaxEigenvalue(
    void (EMfields3D::*GpuImage)(cudaSolverType*, cudaSolverType*), int n,
    int nIter, MPI_Comm fieldcomm) {
  // Lazy allocate Chebyshev workspace (needed for d_chebTmp as scratch)
  if (chebAlloc < n) {
    if (d_chebY)
      cudaFree(d_chebY);
    if (d_chebW)
      cudaFree(d_chebW);
    if (d_chebZ)
      cudaFree(d_chebZ);
    if (d_chebTmp)
      cudaFree(d_chebTmp);
    cudaErrChk(cudaMalloc(&d_chebY, (size_t)n * sizeof(cudaSolverType)));
    cudaErrChk(cudaMalloc(&d_chebW, (size_t)n * sizeof(cudaSolverType)));
    cudaErrChk(cudaMalloc(&d_chebZ, (size_t)n * sizeof(cudaSolverType)));
    cudaErrChk(cudaMalloc(&d_chebTmp, (size_t)n * sizeof(cudaSolverType)));
    chebAlloc = n;
  }

  // Initialize v = 1/sqrt(n) (uniform vector)
  gpuEqValue(d_chebY, 1.0 / sqrt((double)n), n, solverStream_);

  double lambda = 0.0;
  for (int iter = 0; iter < nIter; iter++) {
    // w = A(v)
    (this->*GpuImage)(d_chebTmp, d_chebY);

    // Rayleigh quotient: λ = (v, w) / (v, v)
    // Since v is normalized, (v,v) ≈ local_sum; need global via MPI
    gpuDot_async(d_chebY, d_chebTmp, n, d_blasScratch, &h_gmresReduceLocal[0],
                 solverStream_);
    gpuNorm2_async(d_chebTmp, n, d_blasScratch, &h_gmresReduceLocal[1],
                   solverStream_);
    cudaErrChk(cudaStreamSynchronize(solverStream_));

    double localBuf[2] = {h_gmresReduceLocal[0], h_gmresReduceLocal[1]};
    double globalBuf[2];
    MPI_Allreduce(localBuf, globalBuf, 2, mpiTypeOf<cudaSolverType>(), MPI_SUM,
                  fieldcomm);

    double vw = globalBuf[0]; // (v, Av)
    double ww = globalBuf[1]; // ||Av||²

    if (ww < 1e-30)
      break;
    lambda = vw; // since v is normalized: (v,v) = 1 globally → λ = (v,Av)

    // v = w / ||w||
    gpuScaleCopy(d_chebY, d_chebTmp, 1.0 / sqrt(ww), n, solverStream_);
  }

  return lambda;
}

// =========================================================================
//  GPU Chebyshev Solve: full solver (with MPI communication)
//
//  Solves  A·x = b  via Chebyshev semi-iteration with initial guess x₀.
//  The recurrence operates on the residual  r₀ = b - A·x₀  and produces
//  a correction  δx ≈ A⁻¹·r₀.  Final result: x = x₀ + δx.
//
//  The polynomial applies to the NEGATED operator -A (positive eigenvalues
//  mapped to the standard Chebyshev interval), matching the formulation
//  in the Poisson Chebyshev solver.
// =========================================================================

void EMfields3D::gpuChebyshevSolve(
    cudaSolverType* d_x, int n, cudaSolverType* d_b,
    void (EMfields3D::*GpuImage)(cudaSolverType*, cudaSolverType*), int maxIter,
    cudaSolverType eigMin, cudaSolverType eigMax, MPI_Comm fieldcomm) {
  const VirtualTopology3D* vct = &get_vct();

  // ---- Lazy workspace allocation ----
  if (chebAlloc < n) {
    if (d_chebY)
      cudaFree(d_chebY);
    if (d_chebW)
      cudaFree(d_chebW);
    if (d_chebZ)
      cudaFree(d_chebZ);
    if (d_chebTmp)
      cudaFree(d_chebTmp);
    cudaErrChk(cudaMalloc(&d_chebY, (size_t)n * sizeof(cudaSolverType)));
    cudaErrChk(cudaMalloc(&d_chebW, (size_t)n * sizeof(cudaSolverType)));
    cudaErrChk(cudaMalloc(&d_chebZ, (size_t)n * sizeof(cudaSolverType)));
    cudaErrChk(cudaMalloc(&d_chebTmp, (size_t)n * sizeof(cudaSolverType)));
    chebAlloc = n;
  }

  // ---- Chebyshev parameters ----
  double theta = (eigMin + eigMax) * 0.5;
  double delta = (eigMax - eigMin) * 0.5;
  double sigma = theta / delta;

  double rhoOld = 1.0 / sigma;
  double rho = 1.0 / (2.0 * sigma - rhoOld);

  // ---- Compute residual r₀ = b - A·x₀ ----
  // Store r₀ in d_chebW (temporary).  d_chebTmp is used for A·x₀.
  (this->*GpuImage)(d_chebTmp, d_x);                    // tmp = A·x₀
  gpuSubRes(d_chebW, d_b, d_chebTmp, n, solverStream_); // w = b - A·x₀ = r₀

  // ---- Optional: print initial residual norm ----
  gpuNorm2_async(d_chebW, n, d_blasScratch, &h_gmresReduceLocal[0],
                 solverStream_);
  cudaErrChk(cudaStreamSynchronize(solverStream_));
  double localNorm = h_gmresReduceLocal[0];
  double globalNorm;
  MPI_Allreduce(&localNorm, &globalNorm, 1, mpiTypeOf<cudaSolverType>(),
                MPI_SUM, fieldcomm);
  double initial_error = sqrt(globalNorm);
  if (vct->getCartesian_rank() == 0)
    printf("  [Chebyshev] Initial residual: %g  (eigMin=%.4g, eigMax=%.4g, %d "
           "steps)\n",
           initial_error, eigMin, eigMax, maxIter);

  // Pointer aliasing for the Chebyshev recurrence:
  //   d_r0  = d_chebW   (residual, read-only after step 0)
  //   pY/pZ = d_chebY/Z (current/previous iterate, rotated via std::swap)
  //   pAy   = d_chebTmp (operator output A(y), always this buffer)
  // stepN writes w into pZ in-place (element-wise kernel, safe aliasing),
  // then swap(pZ, pY) rotates: new_z = old_y, new_y = w.

  double* d_r0 = d_chebW;  // residual (read-only after this)
  double* pY = d_chebY;    // current iterate
  double* pZ = d_chebZ;    // previous iterate
  double* pAy = d_chebTmp; // operator output A(.) — always this buffer

  // ---- Step 0+1: z = r0/theta,  y = f(r0, A(r0)) ----
  (this->*GpuImage)(pAy, d_r0); // Ay = A(r0)
  gpuChebyshevStep1(pY, pZ, d_r0, pAy, theta, delta, rho, n, solverStream_);
  // Now: pY = y₁,  pZ = z₀ = r0/theta

  // ---- Steps 2..maxIter ----
  for (int step = 2; step <= maxIter; step++) {
    rhoOld = rho;
    rho = 1.0 / (2.0 * sigma - rhoOld);

    // Ay = A(y)
    (this->*GpuImage)(pAy, pY);

    // stepN overwrites pZ in-place (element-wise, safe) then swap rotates
    gpuChebyshevStepN(pZ, // output: overwrite z in-place with w
                      pY, pZ, d_r0, pAy, delta, sigma, rho, rhoOld, n,
                      solverStream_);

    // Rotate: old_z now holds w.  We need z←y, y←w.
    // After writing w into pZ:  pZ has w, pY has old y.
    // We want: new_z = old_y, new_y = w.
    // So: swap(pZ, pY) → pZ = old_y, pY = w.  ✓
    std::swap(pZ, pY);
  }

  // ---- Apply correction: x = x₀ + y  (y ≈ A⁻¹·r₀) ----
  // The Chebyshev iteration solves A·y = r₀, giving y ≈ A⁻¹·r₀.
  // x_new = x₀ + y = x₀ + A⁻¹·(b - A·x₀)
  gpuAddscale(1.0, d_x, pY, n, solverStream_);

  // ---- Print final residual ----
  (this->*GpuImage)(pAy, d_x);
  gpuSubRes(d_chebW, d_b, pAy, n, solverStream_);
  gpuNorm2_async(d_chebW, n, d_blasScratch, &h_gmresReduceLocal[0],
                 solverStream_);
  cudaErrChk(cudaStreamSynchronize(solverStream_));
  localNorm = h_gmresReduceLocal[0];
  MPI_Allreduce(&localNorm, &globalNorm, 1, mpiTypeOf<cudaSolverType>(),
                MPI_SUM, fieldcomm);
  double final_error = sqrt(globalNorm);
  if (vct->getCartesian_rank() == 0)
    printf("  [Chebyshev] Final residual: %g  (reduction: %.2e)\n", final_error,
           final_error / (initial_error + 1e-30));
}

// =========================================================================
//  GPU MaxwellSource:  build RHS of Maxwell system
// =========================================================================

void EMfields3D::gpuMaxwellSource(cudaSolverType* d_bkrylov) {
  const Collective* col = &get_col();
  const VirtualTopology3D* vct = &get_vct();
  const Grid* grid = &get_grid();
  double _invdx = grid->get_invdx();
  double _invdy = grid->get_invdy();
  double _invdz = grid->get_invdz();

  size_t nodeSize = (size_t)nxn * nyn * nzn;
  size_t centSize = (size_t)nxc * nyc * nzc;

  // Zero work arrays (batched) — tempC is centSize, others are nodeSize
  d_tempC.setAll(0.0, solverStream_);
  cudaSolverType* zptrs2[9] = {
      d_tempX.devPtr(),  d_tempY.devPtr(),  d_tempZ.devPtr(),
      d_tempXN.devPtr(), d_tempYN.devPtr(), d_tempZN.devPtr(),
      d_temp2X.devPtr(), d_temp2Y.devPtr(), d_temp2Z.devPtr()};
  gpuSetAll0_N(zptrs2, 9, nodeSize, solverStream_);

  // Communicate Bc ghost cells (batched 3-field, mixed BCs)
  gpuCommunicateCenterBC_3mixed(nxc, nyc, nzc, d_Bxc, col->bcBx, d_Byc,
                                col->bcBy, d_Bzc, col->bcBz);

  // Case-specific B fixes (before curl, matching CPU MaxwellSource order)
  {
    const string& simCase = col->getCase();
    if (simCase == "ForceFree")
      gpuFixBforcefree();
    // CPU MaxwellSource intentionally does not call fixBnGEM here: the source
    // curl below reads cell-centered B only.
  }

  // OpenBC: apply inflow BCs on center B
  gpuOpenBoundaryInflowB(d_Bxc.devPtr(), d_Byc.devPtr(), d_Bzc.devPtr(), nxc,
                         nyc, nzc);

  // Case-specific center B fix after OpenBC
  {
    const string& simCase = col->getCase();
    if (simCase == "GEM" || simCase == "GEMnoPert" ||
        simCase == "GEMDoubleHarris")
      gpuFixBcGEM();
  }

  // curl(Bc) → tempXN/YN/ZN
  gpuCurlC2N(d_tempXN.devPtr(), d_tempYN.devPtr(), d_tempZN.devPtr(),
             d_Bxc.devPtr(), d_Byc.devPtr(), d_Bzc.devPtr(), nxn, nyn, nzn,
             _invdx, _invdy, _invdz, solverStream_);

  // temp2 = -4π/c * Jhat  (fused triple)
  gpuScaleCopy3(d_temp2X.devPtr(), d_Jxh.devPtr(), d_temp2Y.devPtr(),
                d_Jyh.devPtr(), d_temp2Z.devPtr(), d_Jzh.devPtr(), -FourPI / c,
                nodeSize, solverStream_);

  // temp2 += curl(B)  (fused triple)
  gpuSum3(d_temp2X.devPtr(), d_tempXN.devPtr(), d_temp2Y.devPtr(),
          d_tempYN.devPtr(), d_temp2Z.devPtr(), d_tempZN.devPtr(), nodeSize,
          solverStream_);

  // temp2 *= delt  (fused triple)
  gpuScale3(d_temp2X.devPtr(), d_temp2Y.devPtr(), d_temp2Z.devPtr(), delt,
            nodeSize, solverStream_);

  // Communicate rhoh
  gpuCommunicateCenterBC_P(nxc, nyc, nzc, d_rhoh, 2, 2, 2, 2, 2, 2);

  // grad(rhoh) → tempX/Y/Z
  gpuGradC2N(d_tempX.devPtr(), d_tempY.devPtr(), d_tempZ.devPtr(),
             d_rhoh.devPtr(), nxn, nyn, nzn, _invdx, _invdy, _invdz,
             solverStream_);

  // temp *= -delt² * 4π  (fused triple)
  gpuScale3(d_tempX.devPtr(), d_tempY.devPtr(), d_tempZ.devPtr(),
            -delt * delt * FourPI, nodeSize, solverStream_);

  // Add E and curl+current parts  (fused: 6 gpuSum → 1 gpuSumAddTwo3)
  gpuSumAddTwo3(d_tempX.devPtr(), d_Ex.devPtr(), d_temp2X.devPtr(),
                d_tempY.devPtr(), d_Ey.devPtr(), d_temp2Y.devPtr(),
                d_tempZ.devPtr(), d_Ez.devPtr(), d_temp2Z.devPtr(), nodeSize,
                solverStream_);

  // CPU perfectConductor*S uses ebc = -(ue0,ve0,we0) x (B0x,B0y,B0z).
  const cudaSolverType ebc0 = -(ve0 * B0z - we0 * B0y);
  const cudaSolverType ebc1 = -(we0 * B0x - ue0 * B0z);
  const cudaSolverType ebc2 = -(ue0 * B0y - ve0 * B0x);

  // Perfect conductor BCs for source
  if (vct->getXleft_neighbor() == MPI_PROC_NULL && bcEMfaceXleft == 0)
    gpuPerfectConductorLeftS(d_tempX.devPtr(), d_tempY.devPtr(),
                             d_tempZ.devPtr(), ebc0, ebc1, ebc2, nxn, nyn, nzn,
                             0, solverStream_);
  if (vct->getXright_neighbor() == MPI_PROC_NULL && bcEMfaceXright == 0)
    gpuPerfectConductorRightS(d_tempX.devPtr(), d_tempY.devPtr(),
                              d_tempZ.devPtr(), ebc0, ebc1, ebc2, nxn, nyn, nzn,
                              0, solverStream_);
  if (vct->getYleft_neighbor() == MPI_PROC_NULL && bcEMfaceYleft == 0)
    gpuPerfectConductorLeftS(d_tempX.devPtr(), d_tempY.devPtr(),
                             d_tempZ.devPtr(), ebc0, ebc1, ebc2, nxn, nyn, nzn,
                             1, solverStream_);
  if (vct->getYright_neighbor() == MPI_PROC_NULL && bcEMfaceYright == 0)
    gpuPerfectConductorRightS(d_tempX.devPtr(), d_tempY.devPtr(),
                              d_tempZ.devPtr(), ebc0, ebc1, ebc2, nxn, nyn, nzn,
                              1, solverStream_);
  if (vct->getZleft_neighbor() == MPI_PROC_NULL && bcEMfaceZleft == 0)
    gpuPerfectConductorLeftS(d_tempX.devPtr(), d_tempY.devPtr(),
                             d_tempZ.devPtr(), ebc0, ebc1, ebc2, nxn, nyn, nzn,
                             2, solverStream_);
  if (vct->getZright_neighbor() == MPI_PROC_NULL && bcEMfaceZright == 0)
    gpuPerfectConductorRightS(d_tempX.devPtr(), d_tempY.devPtr(),
                              d_tempZ.devPtr(), ebc0, ebc1, ebc2, nxn, nyn, nzn,
                              2, solverStream_);

  // OpenBC: zero source on inflow boundary nodes
  gpuOpenBoundaryInflowESource(d_tempX.devPtr(), d_tempY.devPtr(),
                               d_tempZ.devPtr(), nxn, nyn, nzn);

  // Physical → Krylov space
  gpuPhys2Solver3(d_bkrylov, d_tempX.devPtr(), d_tempY.devPtr(),
                  d_tempZ.devPtr(), nxn, nyn, nzn, solverStream_);
}

// =========================================================================
//  GPU GMRES:  restarted GMRES(m) on device Krylov vectors
//
//  Performance-critical design choices:
//    • All host-side small arrays (H, g, cs, sn, y, reduction buffers)
//      are PINNED and PERSISTENT — allocated once in gpuSolverInit().
//    • D→H reduction copies use async memcpy into pinned memory.
//    • Stream syncs are batched: ONE sync per Arnoldi step (before
//      MPI_Allreduce) plus ONE for the post-ortho norm.
//    • Device workspace (V, w) is allocated persistently in
//    gpuSolverAllocate().
// =========================================================================

static void
gpuGMRES_impl(EMfields3D* field,
              void (EMfields3D::*GpuImage)(cudaSolverType*, cudaSolverType*),
              cudaSolverType* d_x, int n, cudaSolverType* d_b, int m,
              int max_iter, cudaSolverType tol, cudaSolverType* d_scratch,
              cudaSolverType* const d_gmresV, cudaSolverType* const d_gmresW,
              const size_t gmresVAlloc, MPI_Comm fieldcomm, cudaStream_t stream,
              // Persistent PINNED host buffers (from EMfields3D members)
              cudaSolverType* h_reduceLocal, cudaSolverType* h_reduceGlobal,
              cudaSolverType* H,  // [mp1 * m]
              cudaSolverType* g,  // [mp1]
              cudaSolverType* cs, // [m]
              cudaSolverType* sn, // [m]
              cudaSolverType* y)  // [mp1]
{
  const int mp1 = m + 1;
  const size_t requiredV = static_cast<size_t>(mp1) * n;

  // GMRES workspace is allocated persistently in gpuSolverAllocate().
  if (!d_gmresV || !d_gmresW || gmresVAlloc < requiredV) {
    eprintf("Persistent GPU GMRES workspace too small: allocated %zu, required "
            "%zu",
            gmresVAlloc, requiredV);
    abort();
  }

  // r = b - A*x  →  V[0]
  (field->*GpuImage)(d_gmresW, d_x);     // w = A*x
  gpuEq(d_gmresV, d_b, n, stream);       // V[0] = b
  gpuSub(d_gmresV, d_gmresW, n, stream); // V[0] = b - A*x

  // ||r||₂  (async D→H into pinned buffer, one sync)
  gpuNorm2_async(d_gmresV, n, d_scratch, &h_reduceLocal[0], stream);
  cudaErrChk(cudaStreamSynchronize(stream));
  double initial_error;
  MPI_Allreduce(&h_reduceLocal[0], &initial_error, 1,
                mpiTypeOf<cudaSolverType>(), MPI_SUM, fieldcomm);
  initial_error = sqrt(initial_error);
  if (initial_error < 1e-30)
    return;

  int gmresRank;
  MPI_Comm_rank(fieldcomm, &gmresRank);

  // Compute ||b|| for status print (async, combined sync)
  gpuNorm2_async(d_b, n, d_scratch, &h_reduceLocal[0], stream);
  cudaErrChk(cudaStreamSynchronize(stream));
  double normb;
  MPI_Allreduce(&h_reduceLocal[0], &normb, 1, mpiTypeOf<cudaSolverType>(),
                MPI_SUM, fieldcomm);
  normb = sqrt(normb);
  if (normb == 0.0)
    normb = 1.0;
  if (gmresRank == 0)
    printf("Initial residual: %g norm b vector (source) = %g\n", initial_error,
           normb);

  gpuScale(d_gmresV, 1.0 / initial_error, n, stream);
  double error = initial_error;

  for (int restart = 0; restart < max_iter; restart++) {
    // Zero persistent host arrays
    memset(H, 0, (size_t)mp1 * m * sizeof(cudaSolverType));
    memset(g, 0, mp1 * sizeof(cudaSolverType));
    memset(cs, 0, m * sizeof(cudaSolverType));
    memset(sn, 0, m * sizeof(cudaSolverType));
    g[0] = error;
    int kEnd = m - 1;

    for (int k = 0; k < m; k++) {
      // w = A * V[k]
      (field->*GpuImage)(d_gmresW, d_gmresV + (size_t)k * n);

      // ---- Batched Arnoldi: fuse all k+1 dot products + norm² ----
      // ONE kernel launch → ONE async D→H copy → ONE sync → ONE MPI_Allreduce
      gpuBatchedDotNorm(d_gmresW, d_gmresV, (size_t)n, k, (size_t)n, d_scratch,
                        stream);
      cudaErrChk(cudaMemcpyAsync(h_reduceLocal, d_scratch,
                                 (k + 2) * sizeof(cudaSolverType),
                                 cudaMemcpyDeviceToHost, stream));
      cudaErrChk(cudaStreamSynchronize(stream));
      MPI_Allreduce(h_reduceLocal, h_reduceGlobal, k + 2,
                    mpiTypeOf<cudaSolverType>(), MPI_SUM, fieldcomm);

      // Store H[j][k] and apply orthogonalisation updates
      for (int j = 0; j <= k; j++) {
        double h_jk = h_reduceGlobal[j];
        H[j * m + k] = h_jk;
        gpuAddscale(-h_jk, d_gmresW, d_gmresV + (size_t)j * n, n, stream);
      }

      // H[k+1][k] = ||w_perp||  (post-orthogonalisation norm)
      // Async D→H, single sync
      gpuNorm2_async(d_gmresW, n, d_scratch, &h_reduceLocal[0], stream);
      cudaErrChk(cudaStreamSynchronize(stream));
      double global_wNorm;
      MPI_Allreduce(&h_reduceLocal[0], &global_wNorm, 1,
                    mpiTypeOf<cudaSolverType>(), MPI_SUM, fieldcomm);
      H[(k + 1) * m + k] = sqrt(global_wNorm);

      // Re-orthogonalise if needed (matches CPU GMRES)
      double av = sqrt(h_reduceGlobal[k + 1]); // pre-ortho ||w||
      const double delta = 0.001;
      if (av + delta * H[(k + 1) * m + k] == av) {
        for (int j = 0; j <= k; j++) {
          gpuDot_async(d_gmresW, d_gmresV + (size_t)j * n, n, d_scratch,
                       &h_reduceLocal[0], stream);
          cudaErrChk(cudaStreamSynchronize(stream));
          double htmp;
          MPI_Allreduce(&h_reduceLocal[0], &htmp, 1,
                        mpiTypeOf<cudaSolverType>(), MPI_SUM, fieldcomm);
          H[j * m + k] += htmp;
          gpuAddscale(-htmp, d_gmresW, d_gmresV + (size_t)j * n, n, stream);
        }
        gpuNorm2_async(d_gmresW, n, d_scratch, &h_reduceLocal[0], stream);
        cudaErrChk(cudaStreamSynchronize(stream));
        MPI_Allreduce(&h_reduceLocal[0], &global_wNorm, 1,
                      mpiTypeOf<cudaSolverType>(), MPI_SUM, fieldcomm);
        H[(k + 1) * m + k] = sqrt(global_wNorm);
      }

      // V[k+1] = w / H[k+1][k]
      if (H[(k + 1) * m + k] > 1e-30)
        gpuScaleCopy(d_gmresV + (size_t)(k + 1) * n, d_gmresW,
                     1.0 / H[(k + 1) * m + k], n, stream);
      else
        gpuEq(d_gmresV + (size_t)(k + 1) * n, d_gmresW, n, stream);

      // Apply previous Givens rotations
      for (int j = 0; j < k; j++) {
        double h0 = H[j * m + k];
        double h1 = H[(j + 1) * m + k];
        H[j * m + k] = cs[j] * h0 + sn[j] * h1;
        H[(j + 1) * m + k] = -sn[j] * h0 + cs[j] * h1;
      }

      // Compute new Givens rotation
      double h_kk = H[k * m + k];
      double h_k1k = H[(k + 1) * m + k];
      double r_val = sqrt(h_kk * h_kk + h_k1k * h_k1k);
      cs[k] = h_kk / r_val;
      sn[k] = h_k1k / r_val;
      H[k * m + k] = r_val;
      H[(k + 1) * m + k] = 0.0;

      double g_k = g[k];
      g[k] = cs[k] * g_k;
      g[k + 1] = -sn[k] * g_k;

      error = fabs(g[k + 1]);

      if (error / initial_error < tol) {
        kEnd = k;
        break; // Givens suggests convergence — verify with true residual
      }
    } // end inner loop

    // ---- Back-substitution and solution update ----
    for (int i = kEnd; i >= 0; i--) {
      y[i] = g[i];
      for (int j = i + 1; j <= kEnd; j++)
        y[i] -= H[i * m + j] * y[j];
      y[i] /= H[i * m + i];
    }
    for (int j = 0; j <= kEnd; j++)
      gpuAddscale(y[j], d_x, d_gmresV + (size_t)j * n, n, stream);

    // ---- True residual check (handles affine operators correctly) ----
    (field->*GpuImage)(d_gmresW, d_x);
    gpuEq(d_gmresV, d_b, n, stream);
    gpuSub(d_gmresV, d_gmresW, n, stream);

    gpuNorm2_async(d_gmresV, n, d_scratch, &h_reduceLocal[0], stream);
    cudaErrChk(cudaStreamSynchronize(stream));
    MPI_Allreduce(&h_reduceLocal[0], &error, 1, mpiTypeOf<cudaSolverType>(),
                  MPI_SUM, fieldcomm);
    error = sqrt(error);

    if (error / initial_error < tol) {
      if (gmresRank == 0)
        printf(
            "GMRES converged at restart # %d; iteration #%d with error: %g\n",
            restart, kEnd, error / initial_error);
      return;
    }
    gpuScale(d_gmresV, 1.0 / error, n, stream);
  }
  if (gmresRank == 0)
    std::cout << "  [GMRES] WARNING: did not converge after " << max_iter
              << " restarts" << std::endl;
}

// =========================================================================
//  GPU FGMRES workspace: allocate Z only on first actual FGMRES use
// =========================================================================

void EMfields3D::gpuEnsureFGMRESWorkspace(int m, int n) {
  const int nMaxwellKrylov = maxwellKrylovSize_;
  const int nPoissonKrylov = poissonKrylovSize_;
  const int nGMRESKrylov = std::max(nMaxwellKrylov, nPoissonKrylov);

  if (m > GMRES_M || n > nGMRESKrylov) {
    eprintf("Requested FGMRES workspace exceeds persistent sizing: requested "
            "m=%d n=%d, max m=%d n=%d",
            m, n, GMRES_M, nGMRESKrylov);
    abort();
  }

  const size_t required = static_cast<size_t>(GMRES_M) * nGMRESKrylov;
  if (!d_fgmresZ) {
    cudaErrChk(cudaMalloc(&d_fgmresZ, required * sizeof(cudaSolverType)));
    fgmresZAlloc = required;
    return;
  }

  if (fgmresZAlloc < required) {
    eprintf("Persistent GPU FGMRES Z workspace too small: allocated %zu, "
            "required %zu",
            fgmresZAlloc, required);
    abort();
  }
}

// =========================================================================
//  GPU FGMRES(m):  restarted right-preconditioned GMRES
//
//  Parameterised implementation shared by all FGMRES variants.
//  Differs from gpuGMRES_impl by using a Z-basis (Z[k] = M⁻¹ V[k])
//  and recomputing the true residual after each restart.
// =========================================================================

static void
gpuFGMRES_impl(EMfields3D* field,
               void (EMfields3D::*GpuImage)(cudaSolverType*, cudaSolverType*),
               void (EMfields3D::*GpuPrecond)(cudaSolverType*, cudaSolverType*),
               cudaSolverType* d_x, int n, cudaSolverType* d_b, int m,
               int max_iter, cudaSolverType tol, cudaSolverType* d_scratch,
               cudaSolverType* const d_gmresV, cudaSolverType* const d_gmresW,
               const size_t gmresVAlloc, cudaSolverType* const d_fgmresZ,
               const size_t fgmresZAlloc, MPI_Comm fieldcomm,
               cudaStream_t stream, cudaSolverType* h_reduceLocal,
               cudaSolverType* h_reduceGlobal, cudaSolverType* H,
               cudaSolverType* g, cudaSolverType* cs, cudaSolverType* sn,
               cudaSolverType* y, const char* label) {
  const int mp1 = m + 1;
  const size_t requiredV = static_cast<size_t>(mp1) * n;
  const size_t requiredZ = static_cast<size_t>(m) * n;

  // V[(m+1)*n] and w[n] are shared with GMRES and allocated persistently.
  if (!d_gmresV || !d_gmresW || gmresVAlloc < requiredV) {
    eprintf("Persistent GPU GMRES workspace too small: allocated %zu, required "
            "%zu",
            gmresVAlloc, requiredV);
    abort();
  }
  if (!d_fgmresZ || fgmresZAlloc < requiredZ) {
    eprintf("Persistent GPU FGMRES Z workspace too small: allocated %zu, "
            "required %zu",
            fgmresZAlloc, requiredZ);
    abort();
  }

  // r = b - A*x → V[0]
  (field->*GpuImage)(d_gmresW, d_x);
  gpuEq(d_gmresV, d_b, n, stream);
  gpuSub(d_gmresV, d_gmresW, n, stream);

  // ||r||₂
  gpuNorm2_async(d_gmresV, n, d_scratch, &h_reduceLocal[0], stream);
  cudaErrChk(cudaStreamSynchronize(stream));
  double initial_error;
  MPI_Allreduce(&h_reduceLocal[0], &initial_error, 1,
                mpiTypeOf<cudaSolverType>(), MPI_SUM, fieldcomm);
  initial_error = sqrt(initial_error);
  if (initial_error < 1e-30)
    return;

  int rank;
  MPI_Comm_rank(fieldcomm, &rank);

  gpuNorm2_async(d_b, n, d_scratch, &h_reduceLocal[0], stream);
  cudaErrChk(cudaStreamSynchronize(stream));
  double normb;
  MPI_Allreduce(&h_reduceLocal[0], &normb, 1, mpiTypeOf<cudaSolverType>(),
                MPI_SUM, fieldcomm);
  normb = sqrt(normb);
  if (normb == 0.0)
    normb = 1.0;
  if (rank == 0)
    printf("  [%s] Initial residual: %g  norm(b) = %g\n", label, initial_error,
           normb);

  double rho_tol = initial_error * tol;
  gpuScale(d_gmresV, 1.0 / initial_error, n, stream);
  double error = initial_error;

  for (int restart = 0; restart < max_iter; restart++) {
    memset(H, 0, (size_t)mp1 * m * sizeof(cudaSolverType));
    memset(g, 0, mp1 * sizeof(cudaSolverType));
    memset(cs, 0, m * sizeof(cudaSolverType));
    memset(sn, 0, m * sizeof(cudaSolverType));
    g[0] = error;

    int kk = 0;

    for (int k = 0; k < m && error > rho_tol; k++) {
      kk = k;

      // Z[k] = M⁻¹ V[k]
      (field->*GpuPrecond)(d_fgmresZ + (size_t)k * n, d_gmresV + (size_t)k * n);

      // w = A * Z[k]
      (field->*GpuImage)(d_gmresW, d_fgmresZ + (size_t)k * n);

      // Batched Arnoldi: fuse k+1 dot products + ||w||²
      gpuBatchedDotNorm(d_gmresW, d_gmresV, (size_t)n, k, (size_t)n, d_scratch,
                        stream);
      cudaErrChk(cudaMemcpyAsync(h_reduceLocal, d_scratch,
                                 (k + 2) * sizeof(cudaSolverType),
                                 cudaMemcpyDeviceToHost, stream));
      cudaErrChk(cudaStreamSynchronize(stream));
      MPI_Allreduce(h_reduceLocal, h_reduceGlobal, k + 2,
                    mpiTypeOf<cudaSolverType>(), MPI_SUM, fieldcomm);

      for (int j = 0; j <= k; j++) {
        double h_jk = h_reduceGlobal[j];
        H[j * m + k] = h_jk;
        gpuAddscale(-h_jk, d_gmresW, d_gmresV + (size_t)j * n, n, stream);
      }

      gpuNorm2_async(d_gmresW, n, d_scratch, &h_reduceLocal[0], stream);
      cudaErrChk(cudaStreamSynchronize(stream));
      double global_wNorm;
      MPI_Allreduce(&h_reduceLocal[0], &global_wNorm, 1,
                    mpiTypeOf<cudaSolverType>(), MPI_SUM, fieldcomm);
      H[(k + 1) * m + k] = sqrt(global_wNorm);

      // Re-orthogonalise if needed
      double av = sqrt(h_reduceGlobal[k + 1]);
      const double delta_reorth = 0.001;
      if (av + delta_reorth * H[(k + 1) * m + k] == av) {
        for (int j = 0; j <= k; j++) {
          gpuDot_async(d_gmresW, d_gmresV + (size_t)j * n, n, d_scratch,
                       &h_reduceLocal[0], stream);
          cudaErrChk(cudaStreamSynchronize(stream));
          double htmp;
          MPI_Allreduce(&h_reduceLocal[0], &htmp, 1,
                        mpiTypeOf<cudaSolverType>(), MPI_SUM, fieldcomm);
          H[j * m + k] += htmp;
          gpuAddscale(-htmp, d_gmresW, d_gmresV + (size_t)j * n, n, stream);
        }
        gpuNorm2_async(d_gmresW, n, d_scratch, &h_reduceLocal[0], stream);
        cudaErrChk(cudaStreamSynchronize(stream));
        MPI_Allreduce(&h_reduceLocal[0], &global_wNorm, 1,
                      mpiTypeOf<cudaSolverType>(), MPI_SUM, fieldcomm);
        H[(k + 1) * m + k] = sqrt(global_wNorm);
      }

      if (H[(k + 1) * m + k] > 1e-30)
        gpuScaleCopy(d_gmresV + (size_t)(k + 1) * n, d_gmresW,
                     1.0 / H[(k + 1) * m + k], n, stream);
      else
        gpuEq(d_gmresV + (size_t)(k + 1) * n, d_gmresW, n, stream);

      for (int j = 0; j < k; j++) {
        double h0 = H[j * m + k];
        double h1 = H[(j + 1) * m + k];
        H[j * m + k] = cs[j] * h0 + sn[j] * h1;
        H[(j + 1) * m + k] = -sn[j] * h0 + cs[j] * h1;
      }

      double h_kk = H[k * m + k];
      double h_k1k = H[(k + 1) * m + k];
      double r_val = sqrt(h_kk * h_kk + h_k1k * h_k1k);
      cs[k] = h_kk / r_val;
      sn[k] = h_k1k / r_val;
      H[k * m + k] = r_val;
      H[(k + 1) * m + k] = 0.0;

      double g_k = g[k];
      g[k] = cs[k] * g_k;
      g[k + 1] = -sn[k] * g_k;

      error = fabs(g[k + 1]);
    }

    // Back-substitution
    {
      int kEnd = (error <= rho_tol) ? kk : m - 1;
      for (int i = kEnd; i >= 0; i--) {
        y[i] = g[i];
        for (int j = i + 1; j <= kEnd; j++)
          y[i] -= H[i * m + j] * y[j];
        y[i] /= H[i * m + i];
      }
      for (int j = 0; j <= kEnd; j++)
        gpuAddscale(y[j], d_x, d_fgmresZ + (size_t)j * n, n, stream);
    }

    // True residual check (mandatory for flexible preconditioning)
    (field->*GpuImage)(d_gmresW, d_x);
    gpuEq(d_gmresV, d_b, n, stream);
    gpuSub(d_gmresV, d_gmresW, n, stream);

    gpuNorm2_async(d_gmresV, n, d_scratch, &h_reduceLocal[0], stream);
    cudaErrChk(cudaStreamSynchronize(stream));
    MPI_Allreduce(&h_reduceLocal[0], &error, 1, mpiTypeOf<cudaSolverType>(),
                  MPI_SUM, fieldcomm);
    error = sqrt(error);

    if (error <= rho_tol) {
      if (rank == 0)
        printf("  [%s] Converged at restart #%d, iteration #%d, error: %g\n",
               label, restart, kk, error / initial_error);
      return;
    }
    gpuScale(d_gmresV, 1.0 / error, n, stream);
  }

  if (rank == 0)
    printf("  [%s] WARNING: did not converge after %d restarts, error: %g\n",
           label, max_iter, error / initial_error);
}

// =========================================================================
//  GPU Block-Jacobi preconditioner (communication-free)
//
//  Approximately solves  A·x ≈ b  using a 3×3 block-diagonal D and the
//  communication-free local operator A_local.
//
//  • sweeps == 1  →  x = ω D⁻¹ b  (original damped Jacobi)
//  • sweeps > 1   →  Chebyshev semi-iteration on D⁻¹ A_local x = D⁻¹ b
//
//  The Chebyshev recurrence replaces the former Richardson (damped Jacobi)
//  sweeps with a 3-term polynomial recurrence whose coefficients are
//  derived from analytic eigenvalue bounds of D⁻¹ A_local:
//
//      λ_min ≈ 1 / diagScalar   (stencil diagonal analysis)
//      λ_max ≤ 2                (Gershgorin bound)
//
//  For the same number of sweeps the Chebyshev polynomial reduces the
//  error by the minimax factor  T_k(σ)⁻¹  vs  (1−ω/λ_max)^k  for
//  Richardson, giving ≈ 2-3× faster residual reduction per sweep.
//
//  D^{-1} is precomputed once per field solve (9 doubles/node) and applied
//  via a fused Krylov kernel (no pack/unpack overhead).
// =========================================================================

void EMfields3D::gpuBlockJacobiPrecond(cudaSolverType* d_x,
                                       cudaSolverType* d_b) {
  const Grid* grid = &get_grid();
  const int n = maxwellKrylovSize_;
  double invdx = grid->get_invdx();
  double invdy = grid->get_invdy();
  double invdz = grid->get_invdz();

  // Diagonal block coefficients
  double cSigma = invdx * invdx + invdy * invdy + invdz * invdz;
  double delt2h = delt * delt * 0.5;
  double diagScalar = 1.0 + delt2h * cSigma;
  double wx = 1.0 + delt2h * invdx * invdx;
  double wy = 1.0 + delt2h * invdy * invdy;
  double wz = 1.0 + delt2h * invdz * invdz;

  size_t nodeSlice = (size_t)nxn * nyn * nzn;

  // ---- Lazy-allocate D^{-1} storage ----
  if (blockJacobiDinvAlloc < (int)nodeSlice) {
    if (d_blockJacobiDinv)
      cudaFree(d_blockJacobiDinv);
    cudaErrChk(
        cudaMalloc(&d_blockJacobiDinv, 9 * nodeSlice * sizeof(cudaSolverType)));
    blockJacobiDinvAlloc = (int)nodeSlice;
    blockJacobiDinvStale = true;
  }

  // ---- Precompute D^{-1} once per solver invocation ----
  if (blockJacobiDinvStale) {
    gpuPrecomputeBlockJacobiInv(
        d_blockJacobiDinv, d_Bxn.devPtr(), d_Byn.devPtr(), d_Bzn.devPtr(),
        d_Bx_ext.devPtr(), d_By_ext.devPtr(), d_Bz_ext.devPtr(),
        d_rhons.devPtr(), d_qom, ns, dt, c, delt, FourPI, diagScalar, wx, wy,
        wz, nxn, nyn, nzn, solverStream_);
    blockJacobiDinvStale = false;
  }

  double omega = blockJacobiOmega;
  int nSweeps = blockJacobiSweeps;

  // ---- Step 0: z = D⁻¹ b  (fused Krylov kernel, no pack/unpack) ----
  gpuApplyBlockJacobiInvKrylov(d_x, d_b, d_blockJacobiDinv, nxn, nyn, nzn,
                               solverStream_);

  if (nSweeps <= 1) {
    // Single sweep: scale by ω (original damped-Jacobi behavior)
    if (omega != 1.0) {
      gpuScale(d_x, omega, (size_t)n, solverStream_);
    }
    return;
  }

  // ==================================================================
  //  Richardson (weighted Jacobi) refinement  (nSweeps > 1)
  //
  //  x = D⁻¹ b  (already computed above)
  //  for s = 2 .. nSweeps:
  //      r = b − A_local(x)
  //      x += ω · D⁻¹ r
  //
  //  Unlike the Chebyshev three-term recurrence, Richardson iteration
  //  does not amplify errors at subdomain-boundary nodes where A_local
  //  reads zeros from ghost cells.  The convergence factor per sweep
  //  is max|1 − ω·λ| over eigenvalues λ of D⁻¹A_local, which stays
  //  ≤ 1 for ω ∈ (0, 2/λ_max) — no divergence even when some λ ≈ 0.
  // ==================================================================

  // ---- Lazy-allocate dedicated BJ scratch vectors ----
  if (bjScratchAlloc < n) {
    if (d_bjScratch1)
      cudaFree(d_bjScratch1);
    if (d_bjScratch2)
      cudaFree(d_bjScratch2);
    d_bjScratch1 = nullptr;
    d_bjScratch2 = nullptr;
    cudaErrChk(cudaMalloc(&d_bjScratch1, (size_t)n * sizeof(cudaSolverType)));
    cudaErrChk(cudaMalloc(&d_bjScratch2, (size_t)n * sizeof(cudaSolverType)));
    bjScratchAlloc = n;
  }

  for (int step = 1; step < nSweeps; step++) {
    // r = b − A_local(x)
    gpuMaxwellImageLocal(d_bjScratch1, d_x); // s1 = A_local(x)
    gpuSubRes(d_bjScratch2, d_b, d_bjScratch1, (size_t)n,
              solverStream_); // s2 = b − s1

    // x += ω · D⁻¹ r
    gpuApplyBlockJacobiInvKrylov(d_bjScratch1, d_bjScratch2, d_blockJacobiDinv,
                                 nxn, nyn, nzn,
                                 solverStream_); // s1 = D⁻¹ r
    gpuAddscale(omega, d_x, d_bjScratch1, (size_t)n,
                solverStream_); // x += ω·s1
  }
}

// =========================================================================
//  GPU FGMRES(m) with Block-Jacobi preconditioner
// =========================================================================

void EMfields3D::gpuFGMRES_BlockJacobiPrecond(cudaSolverType* d_x, int n,
                                              cudaSolverType* d_b, int m,
                                              int max_iter, cudaSolverType tol,
                                              MPI_Comm fieldcomm) {
  // Diagnostic: print diagScalar so user can evaluate regime
  {
    const Grid* grid = &get_grid();
    double _invdx = grid->get_invdx(), _invdy = grid->get_invdy(),
           _invdz = grid->get_invdz();
    double _cSigma = _invdx * _invdx + _invdy * _invdy + _invdz * _invdz;
    double _diagScalar = 1.0 + delt * delt * 0.5 * _cSigma;
    int rank;
    MPI_Comm_rank(fieldcomm, &rank);
    if (rank == 0)
      printf("  [BlockJacobi] diagScalar=%.2f  sweeps=%d  (preconditioner "
             "helps when diagScalar >> 10)\n",
             _diagScalar, blockJacobiSweeps);
  }

  blockJacobiDinvStale = true;
  gpuEnsureFGMRESWorkspace(m, n);

  gpuFGMRES_impl(this, &EMfields3D::gpuMaxwellImage,
                 &EMfields3D::gpuBlockJacobiPrecond, d_x, n, d_b, m, max_iter,
                 tol, d_blasScratch, d_gmresV, d_gmresW, gmresVAlloc, d_fgmresZ,
                 fgmresZAlloc, fieldcomm, solverStream_, h_gmresReduceLocal,
                 h_gmresReduceGlobal, h_gmresH, h_gmresG, h_gmresCS, h_gmresSN,
                 h_gmresY, "FGMRES+BlockJacobi");
}

// =========================================================================
//  GPU calculateE: full E-field solver
// =========================================================================

void EMfields3D::gpuCalculateE(int cycle) {
  const Collective* col = &get_col();
  const VirtualTopology3D* vct = &get_vct();

  if (vct->getCartesian_rank() == 0)
    cout << "*** E CALCULATION [GPU] ***" << endl;

  const int nMaxwell = maxwellKrylovSize_;
  size_t nodeSize = (size_t)nxn * nyn * nzn;

  // Divergence cleaning on E (Poisson correction)
  gpuPoissonCorrection(cycle);

  if (vct->getCartesian_rank() == 0)
    cout << "*** MAXWELL SOLVER [GPU] ***" << endl;

  // Build RHS
  gpuMaxwellSource(d_bkrylovMaxwell.devPtr());

  // Initial guess: pack current E into x
  gpuPhys2Solver3(d_xkrylovMaxwell.devPtr(), d_Ex.devPtr(), d_Ey.devPtr(),
                  d_Ez.devPtr(), nxn, nyn, nzn, solverStream_);

  MPI_Comm fieldcomm = vct->getFieldComm();

  if (SolverType == "Chebyshev") {
    // Chebyshev semi-iterative solver
    double eigMin = (chebEigMin > 0.0) ? chebEigMin : 1.0;
    double eigMax = chebEigMax;
    if (eigMax <= 0.0) {
      eigMax = gpuEstimateMaxEigenvalue(&EMfields3D::gpuMaxwellImage, nMaxwell,
                                        20, fieldcomm);
    }
    gpuChebyshevSolve(d_xkrylovMaxwell.devPtr(), nMaxwell,
                      d_bkrylovMaxwell.devPtr(), &EMfields3D::gpuMaxwellImage,
                      chebMaxIter, eigMin, eigMax, fieldcomm);
  } else if (SolverType == "FGMRESBlockJacobi") {
    // FGMRES(20) with communication-free block-Jacobi preconditioner
    if (vct->getCartesian_rank() == 0)
      cout << "*** MAXWELL SOLVER [GPU FGMRES+BlockJacobi] ***" << endl;
    gpuFGMRES_BlockJacobiPrecond(d_xkrylovMaxwell.devPtr(), nMaxwell,
                                 d_bkrylovMaxwell.devPtr(), 20, 200, GMREStol,
                                 fieldcomm);
  } else {
    // Default: GMRES(20) solver
    gpuGMRES_impl(this, &EMfields3D::gpuMaxwellImage, d_xkrylovMaxwell.devPtr(),
                  nMaxwell, d_bkrylovMaxwell.devPtr(), 20, 200, GMREStol,
                  d_blasScratch, d_gmresV, d_gmresW, gmresVAlloc, fieldcomm,
                  solverStream_, h_gmresReduceLocal, h_gmresReduceGlobal,
                  h_gmresH, h_gmresG, h_gmresCS, h_gmresSN, h_gmresY);
  }
  // Krylov → physical: Exth, Eyth, Ezth
  gpuSolver2Phys3(d_Exth.devPtr(), d_Eyth.devPtr(), d_Ezth.devPtr(),
                  d_xkrylovMaxwell.devPtr(), nxn, nyn, nzn, solverStream_);

  // E^{n+1} = beta*E + alfa*Eth  where alfa=1/th, beta=-(1-th)/th
  gpuAddscale2_3(1.0 / th, -(1.0 - th) / th, d_Ex.devPtr(), d_Exth.devPtr(),
                 d_Ey.devPtr(), d_Eyth.devPtr(), d_Ez.devPtr(), d_Ezth.devPtr(),
                 nodeSize, solverStream_);

  // Smooth E
  gpuSmoothE();

  // Communicate final E fields (batched: 2 × 3 fields in 1 MPI round each)
  gpuCommunicateNodeBC_3mixed(nxn, nyn, nzn, d_Exth, col->bcEx, d_Eyth,
                              col->bcEy, d_Ezth, col->bcEz);
  gpuCommunicateNodeBC_3mixed(nxn, nyn, nzn, d_Ex, col->bcEx, d_Ey, col->bcEy,
                              d_Ez, col->bcEz);

  // OpenBC Inflow on solved E fields
  gpuOpenBoundaryInflowE(d_Exth.devPtr(), d_Eyth.devPtr(), d_Ezth.devPtr(), nxn,
                         nyn, nzn);
  gpuOpenBoundaryInflowE(d_Ex.devPtr(), d_Ey.devPtr(), d_Ez.devPtr(), nxn, nyn,
                         nzn);
}

// =========================================================================
//  GPU calculateB: Faraday update
// =========================================================================

void EMfields3D::gpuCalculateB(int cycle) {
  const Collective* col = &get_col();
  const VirtualTopology3D* vct = &get_vct();
  const Grid* grid = &get_grid();
  const string& simCase = col->getCase();
  const bool isGemCase = simCase == "GEM" || simCase == "GEMnoPert" ||
                         simCase == "GEMDoubleHarris";
  const bool isForceFreeCase = simCase == "ForceFree";
  double _invdx = grid->get_invdx();
  double _invdy = grid->get_invdy();
  double _invdz = grid->get_invdz();

  if (vct->getCartesian_rank() == 0)
    cout << "*** B CALCULATION [GPU] ***" << endl;

  size_t centSize = (size_t)nxc * nyc * nzc;

  // curl(Eth) → tempXC/YC/ZC
  gpuCurlN2C(d_tempXC.devPtr(), d_tempYC.devPtr(), d_tempZC.devPtr(),
             d_Exth.devPtr(), d_Eyth.devPtr(), d_Ezth.devPtr(), nxc, nyc, nzc,
             _invdx, _invdy, _invdz, solverStream_);

  // B^{n+1} = B^n - c*dt * curl(Eth)
  gpuAddscale3(-c * dt, d_Bxc.devPtr(), d_tempXC.devPtr(), d_Byc.devPtr(),
               d_tempYC.devPtr(), d_Bzc.devPtr(), d_tempZC.devPtr(), centSize,
               solverStream_);

  // Communicate center B ghost cells (batched: 3 fields in 1 MPI round)
#ifdef HALO_OVERLAP
  {
    // C2N at node (i,j,k) reads centers (i-1..i, j-1..j, k-1..k).
    // Start with the halo-independent core, then exclude every node whose
    // stencil can intersect a center layer modified after the halo exchange.
    int safeILo = 2, safeIHi = nxn - 3;
    int safeJLo = 2, safeJHi = nyn - 3;
    int safeKLo = 2, safeKHi = nzn - 3;

    if (vct->getXleft_neighbor() == MPI_PROC_NULL && bcEMfaceXleft == 2 &&
        nxc > 10)
      safeILo = std::max(safeILo, n_layers_sal + 2);
    if (vct->getXright_neighbor() == MPI_PROC_NULL && bcEMfaceXright == 2 &&
        nxc > 10)
      safeIHi = std::min(safeIHi, nxn - n_layers_sal - 3);
    if (vct->getYleft_neighbor() == MPI_PROC_NULL && bcEMfaceYleft == 2 &&
        nyc > 10)
      safeJLo = std::max(safeJLo, n_layers_sal + 2);
    if (vct->getYright_neighbor() == MPI_PROC_NULL && bcEMfaceYright == 2 &&
        nyc > 10)
      safeJHi = std::min(safeJHi, nyn - n_layers_sal - 3);
    if (vct->getZleft_neighbor() == MPI_PROC_NULL && bcEMfaceZleft == 2 &&
        nzc > 10)
      safeKLo = std::max(safeKLo, n_layers_sal + 2);
    if (vct->getZright_neighbor() == MPI_PROC_NULL && bcEMfaceZright == 2 &&
        nzc > 10)
      safeKHi = std::min(safeKHi, nzn - n_layers_sal - 3);

    // The case-specific fixes do not modify all three components to the same
    // depth.  Preserve a separate Y-safe box per component so work that is
    // genuinely independent is not needlessly delayed:
    //   GEM:       Bx/Bz change on 3 layers, By only on the ghost layer.
    //   ForceFree: Bz changes on 3 layers, Bx/By only on the ghost layer.
    int bxSafeJLo = safeJLo, bxSafeJHi = safeJHi;
    int bySafeJLo = safeJLo, bySafeJHi = safeJHi;
    int bzSafeJLo = safeJLo, bzSafeJHi = safeJHi;
    const bool lowYPhysical = vct->getYleft_neighbor() == MPI_PROC_NULL;
    const bool highYPhysical = vct->getYright_neighbor() == MPI_PROC_NULL;
    if (isGemCase) {
      if (lowYPhysical) {
        bxSafeJLo = std::max(bxSafeJLo, 4);
        bzSafeJLo = std::max(bzSafeJLo, 4);
      }
      if (highYPhysical) {
        bxSafeJHi = std::min(bxSafeJHi, nyn - 5);
        bzSafeJHi = std::min(bzSafeJHi, nyn - 5);
      }
    } else if (isForceFreeCase) {
      if (lowYPhysical)
        bzSafeJLo = std::max(bzSafeJLo, 4);
      if (highYPhysical)
        bzSafeJHi = std::min(bzSafeJHi, nyn - 5);
    }

    cudaSolverType* bptrs[3] = {d_Bxc.devPtr(), d_Byc.devPtr(), d_Bzc.devPtr()};
    gpuBatchedHaloBeginExchange(bptrs, 3, nxc, nyc, nzc, true, false, false,
                                solverStream_);

    // This safe core reads neither in-flight halo cells nor center layers that
    // the open/SAL or case-specific fixups below will change.
    gpuInterpC2N_range(d_Bxn.devPtr(), d_Bxc.devPtr(), nxn, nyn, nzn, safeILo,
                       safeIHi, bxSafeJLo, bxSafeJHi, safeKLo, safeKHi,
                       solverStream_);
    gpuInterpC2N_range(d_Byn.devPtr(), d_Byc.devPtr(), nxn, nyn, nzn, safeILo,
                       safeIHi, bySafeJLo, bySafeJHi, safeKLo, safeKHi,
                       solverStream_);
    gpuInterpC2N_range(d_Bzn.devPtr(), d_Bzc.devPtr(), nxn, nyn, nzn, safeILo,
                       safeIHi, bzSafeJLo, bzSafeJHi, safeKLo, safeKHi,
                       solverStream_);

    gpuBatchedHaloEndExchange();

    // Mixed BC face application
    gpuBCface(nxc, nyc, nzc, d_Bxc, col->bcBx[0], col->bcBx[1], col->bcBx[2],
              col->bcBx[3], col->bcBx[4], col->bcBx[5], &_vct, solverStream_);
    gpuBCface(nxc, nyc, nzc, d_Byc, col->bcBy[0], col->bcBy[1], col->bcBy[2],
              col->bcBy[3], col->bcBy[4], col->bcBy[5], &_vct, solverStream_);
    gpuBCface(nxc, nyc, nzc, d_Bzc, col->bcBz[0], col->bcBz[1], col->bcBz[2],
              col->bcBz[3], col->bcBz[4], col->bcBz[5], &_vct, solverStream_);

    // Open boundary conditions on center-based B
    gpuOpenBoundaryInflowB(d_Bxc.devPtr(), d_Byc.devPtr(), d_Bzc.devPtr(), nxc,
                           nyc, nzc);

    // Case-specific fixes on center-based B
    if (isGemCase)
      gpuFixBcGEM();
    if (isForceFreeCase)
      gpuFixBforcefree();

    // Complete exactly the nodes excluded above after halo data and every
    // center-B fixup are visible.  The range and complement phases are
    // disjoint and together cover the same domain as a full gpuInterpC2N call.
    gpuInterpC2N_complement(d_Bxn.devPtr(), d_Bxc.devPtr(), nxn, nyn, nzn,
                            safeILo, safeIHi, bxSafeJLo, bxSafeJHi, safeKLo,
                            safeKHi, solverStream_);
    gpuInterpC2N_complement(d_Byn.devPtr(), d_Byc.devPtr(), nxn, nyn, nzn,
                            safeILo, safeIHi, bySafeJLo, bySafeJHi, safeKLo,
                            safeKHi, solverStream_);
    gpuInterpC2N_complement(d_Bzn.devPtr(), d_Bzc.devPtr(), nxn, nyn, nzn,
                            safeILo, safeIHi, bzSafeJLo, bzSafeJHi, safeKLo,
                            safeKHi, solverStream_);
  }
#else
  gpuCommunicateCenterBC_3mixed(nxc, nyc, nzc, d_Bxc, col->bcBx, d_Byc,
                                col->bcBy, d_Bzc, col->bcBz);

  // Open boundary conditions on center-based B
  gpuOpenBoundaryInflowB(d_Bxc.devPtr(), d_Byc.devPtr(), d_Bzc.devPtr(), nxc,
                         nyc, nzc);

  // Case-specific fixes on center-based B
  if (isGemCase)
    gpuFixBcGEM();
  if (isForceFreeCase)
    gpuFixBforcefree();

  // Interpolate center → node
  gpuInterpC2N(d_Bxn.devPtr(), d_Bxc.devPtr(), nxn, nyn, nzn, solverStream_);
  gpuInterpC2N(d_Byn.devPtr(), d_Byc.devPtr(), nxn, nyn, nzn, solverStream_);
  gpuInterpC2N(d_Bzn.devPtr(), d_Bzc.devPtr(), nxn, nyn, nzn, solverStream_);
#endif

  // Communicate node B ghost cells (batched: 3 fields in 1 MPI round)
  gpuCommunicateNodeBC_3mixed(nxn, nyn, nzn, d_Bxn, col->bcBx, d_Byn, col->bcBy,
                              d_Bzn, col->bcBz);

  // Case-specific fixes on node-based B
  if (isGemCase)
    gpuFixBnGEM();

  // Divergence cleaning: lap(PSI) = div(B), B = B - grad(PSI)
  if (divBCorrection && cycle % divBCorrectionCycle == 0)
    gpuApplyDivBCleaning();
}

// =========================================================================
//  GPU calculateHatFunctions: compute Jhat and rhohat
// =========================================================================

void EMfields3D::gpuCalculateHatFunctions() {
  const Grid* grid = &get_grid();
  double _invdx = grid->get_invdx();
  double _invdy = grid->get_invdy();
  double _invdz = grid->get_invdz();

  size_t nodeSize = (size_t)nxn * nyn * nzn;
  size_t centSize = (size_t)nxc * nyc * nzc;

  // Smooth rhoc
  gpuSmooth(d_rhoc, 0);

  // Initialise Jxh/Jyh/Jzh = 0
  cudaSolverType* jptrs[3] = {d_Jxh.devPtr(), d_Jyh.devPtr(), d_Jzh.devPtr()};
  gpuSetAll0_N(jptrs, 3, nodeSize, solverStream_);

  for (int is = 0; is < ns; is++) {
    // divSymmTensorN2C for this species
    gpuDivSymmTensorN2C(d_tempXC.devPtr(), d_tempYC.devPtr(), d_tempZC.devPtr(),
                        d_pXXsn.speciesPtr(is), d_pXYsn.speciesPtr(is),
                        d_pXZsn.speciesPtr(is), d_pYYsn.speciesPtr(is),
                        d_pYZsn.speciesPtr(is), d_pZZsn.speciesPtr(is), nxc,
                        nyc, nzc, _invdx, _invdy, _invdz, solverStream_);

    // Scale by -dt/2 (fused triple)
    gpuScale3(d_tempXC.devPtr(), d_tempYC.devPtr(), d_tempZC.devPtr(),
              -dt / 2.0, centSize, solverStream_);

#ifdef HALO_OVERLAP
    // --- overlap: begin face exchange for 3 centre fields ---
    cudaSolverType* hatPtrs[3] = {d_tempXC.devPtr(), d_tempYC.devPtr(),
                                  d_tempZC.devPtr()};
    gpuBatchedHaloBeginExchange(hatPtrs, 3, nxc, nyc, nzc,
                                /*offsetZero=*/true, /*faceOnly=*/false,
                                /*isParticle=*/true, solverStream_);
    // --- interior interpC2N while faces in flight ---
    gpuInterpC2N_interior(d_tempXN.devPtr(), d_tempXC.devPtr(), nxn, nyn, nzn,
                          solverStream_);
    gpuInterpC2N_interior(d_tempYN.devPtr(), d_tempYC.devPtr(), nxn, nyn, nzn,
                          solverStream_);
    gpuInterpC2N_interior(d_tempZN.devPtr(), d_tempZC.devPtr(), nxn, nyn, nzn,
                          solverStream_);
    // --- end exchange: wait + unpack + edges + corners ---
    gpuBatchedHaloEndExchange();
    // BC
    gpuBCface_P(nxc, nyc, nzc, d_tempXC, 2, 2, 2, 2, 2, 2, &get_vct(),
                solverStream_);
    gpuBCface_P(nxc, nyc, nzc, d_tempYC, 2, 2, 2, 2, 2, 2, &get_vct(),
                solverStream_);
    gpuBCface_P(nxc, nyc, nzc, d_tempZC, 2, 2, 2, 2, 2, 2, &get_vct(),
                solverStream_);
    // --- boundary interpC2N ---
    gpuInterpC2N_boundary(d_tempXN.devPtr(), d_tempXC.devPtr(), nxn, nyn, nzn,
                          solverStream_);
    gpuInterpC2N_boundary(d_tempYN.devPtr(), d_tempYC.devPtr(), nxn, nyn, nzn,
                          solverStream_);
    gpuInterpC2N_boundary(d_tempZN.devPtr(), d_tempZC.devPtr(), nxn, nyn, nzn,
                          solverStream_);
#else
    // Communicate (batched: 3 fields in 1 MPI round)
    gpuCommunicateCenterBC_P_3(nxc, nyc, nzc, d_tempXC, d_tempYC, d_tempZC, 2,
                               2, 2, 2, 2, 2);

    // Interpolate C → N
    gpuInterpC2N(d_tempXN.devPtr(), d_tempXC.devPtr(), nxn, nyn, nzn,
                 solverStream_);
    gpuInterpC2N(d_tempYN.devPtr(), d_tempYC.devPtr(), nxn, nyn, nzn,
                 solverStream_);
    gpuInterpC2N(d_tempZN.devPtr(), d_tempZC.devPtr(), nxn, nyn, nzn,
                 solverStream_);
#endif

    // Add species current: tempN += Jxs[is] (fused triple)
    gpuSum3(d_tempXN.devPtr(), d_Jxs.speciesPtr(is), d_tempYN.devPtr(),
            d_Jys.speciesPtr(is), d_tempZN.devPtr(), d_Jzs.speciesPtr(is),
            nodeSize, solverStream_);

    // PIdot: Jhat += π(tempN)
    gpuPIdot(d_Jxh, d_Jyh, d_Jzh, d_tempXN, d_tempYN, d_tempZN, is);
  }

  // Smooth Jhat (batched: 3 fields in 1 MPI round per iteration)
  gpuSmooth3(d_Jxh, d_Jyh, d_Jzh, 1);

  // rhohat = rhoc - dt*theta*div(Jhat)
  gpuDivN2C(d_tempXC.devPtr(), d_Jxh.devPtr(), d_Jyh.devPtr(), d_Jzh.devPtr(),
            nxc, nyc, nzc, _invdx, _invdy, _invdz, solverStream_);
  gpuScale(d_tempXC.devPtr(), -dt * th, centSize, solverStream_);
  gpuSum(d_tempXC.devPtr(), d_rhoc.devPtr(), centSize, solverStream_);
  gpuEq(d_rhoh.devPtr(), d_tempXC.devPtr(), centSize, solverStream_);

  // Communicate rhoh
  gpuCommunicateCenterBC_P(nxc, nyc, nzc, d_rhoh, 2, 2, 2, 2, 2, 2);
}

// =========================================================================
//  GPU moment-processing: D2D scatter from packed moment buffer
// =========================================================================

void EMfields3D::gpuScatterMomentsD2D(cudaMomentType* momentsSrc, int species,
                                      cudaStream_t stream) {
  const size_t gridSize = (size_t)nxn * nyn * nzn;
  // momentsSrc layout: [rhons | Jxs | Jys | Jzs | pXXsn | pXYsn | pXZsn | pYYsn
  // | pYZsn | pZZsn] Each block is gridSize doubles, contiguous.
  cudaErrChk(cudaMemcpyAsync(
      d_rhons.speciesPtr(species), momentsSrc + 0 * gridSize,
      gridSize * sizeof(cudaSolverType), cudaMemcpyDeviceToDevice, stream));
  cudaErrChk(cudaMemcpyAsync(
      d_Jxs.speciesPtr(species), momentsSrc + 1 * gridSize,
      gridSize * sizeof(cudaSolverType), cudaMemcpyDeviceToDevice, stream));
  cudaErrChk(cudaMemcpyAsync(
      d_Jys.speciesPtr(species), momentsSrc + 2 * gridSize,
      gridSize * sizeof(cudaSolverType), cudaMemcpyDeviceToDevice, stream));
  cudaErrChk(cudaMemcpyAsync(
      d_Jzs.speciesPtr(species), momentsSrc + 3 * gridSize,
      gridSize * sizeof(cudaSolverType), cudaMemcpyDeviceToDevice, stream));
  cudaErrChk(cudaMemcpyAsync(
      d_pXXsn.speciesPtr(species), momentsSrc + 4 * gridSize,
      gridSize * sizeof(cudaSolverType), cudaMemcpyDeviceToDevice, stream));
  cudaErrChk(cudaMemcpyAsync(
      d_pXYsn.speciesPtr(species), momentsSrc + 5 * gridSize,
      gridSize * sizeof(cudaSolverType), cudaMemcpyDeviceToDevice, stream));
  cudaErrChk(cudaMemcpyAsync(
      d_pXZsn.speciesPtr(species), momentsSrc + 6 * gridSize,
      gridSize * sizeof(cudaSolverType), cudaMemcpyDeviceToDevice, stream));
  cudaErrChk(cudaMemcpyAsync(
      d_pYYsn.speciesPtr(species), momentsSrc + 7 * gridSize,
      gridSize * sizeof(cudaSolverType), cudaMemcpyDeviceToDevice, stream));
  cudaErrChk(cudaMemcpyAsync(
      d_pYZsn.speciesPtr(species), momentsSrc + 8 * gridSize,
      gridSize * sizeof(cudaSolverType), cudaMemcpyDeviceToDevice, stream));
  cudaErrChk(cudaMemcpyAsync(
      d_pZZsn.speciesPtr(species), momentsSrc + 9 * gridSize,
      gridSize * sizeof(cudaSolverType), cudaMemcpyDeviceToDevice, stream));
}

// =========================================================================
//  GPU CommunicateGhostP2G – ALL species batched
//  Processes species in chunks of HALO_MAX_BATCH / 10 . Each chunk runs the
//  full 3-phase sequence (additive halo → boundary adjust → copy halo)
//  independently.
// =========================================================================

void EMfields3D::gpuCommunicateGhostP2G_AllSpecies() {
  const VirtualTopology3D* vct = &get_vct();
  const int nFieldsPerSpecies = 10;
  const int speciesPerBatch = HALO_MAX_BATCH / nFieldsPerSpecies; // floor

  for (int isStart = 0; isStart < ns; isStart += speciesPerBatch) {
    const int isEnd = std::min(isStart + speciesPerBatch, ns);
    const int nsBatch = isEnd - isStart;
    const int nFields = nsBatch * nFieldsPerSpecies;

    // Gather device pointers for this chunk
    cudaSolverType* ptrs[HALO_MAX_BATCH];
    for (int is = 0; is < nsBatch; is++) {
      int off = is * nFieldsPerSpecies;
      int src = isStart + is;
      ptrs[off + 0] = d_rhons.speciesPtr(src);
      ptrs[off + 1] = d_Jxs.speciesPtr(src);
      ptrs[off + 2] = d_Jys.speciesPtr(src);
      ptrs[off + 3] = d_Jzs.speciesPtr(src);
      ptrs[off + 4] = d_pXXsn.speciesPtr(src);
      ptrs[off + 5] = d_pXYsn.speciesPtr(src);
      ptrs[off + 6] = d_pXZsn.speciesPtr(src);
      ptrs[off + 7] = d_pYYsn.speciesPtr(src);
      ptrs[off + 8] = d_pYZsn.speciesPtr(src);
      ptrs[off + 9] = d_pZZsn.speciesPtr(src);
    }

    // Phase 1: Batched additive (interpolating) halo exchange — ghost → shared
    // nodes
    gpuBatchedHaloExchange(ptrs, nFields, nxn, nyn, nzn,
                           /*offsetZero=*/true, /*isFaceOnly=*/false,
                           /*needInterp=*/true, /*isParticle=*/true,
                           solverStream_);

    // Phase 2: adjust non-periodic boundary densities
    gpuAdjustNonPeriodicDensities(nFields, d_ptrArray_, nxn, nyn, nzn,
                                  vct->getXleft_neighbor_P() == MPI_PROC_NULL,
                                  vct->getXright_neighbor_P() == MPI_PROC_NULL,
                                  vct->getYleft_neighbor_P() == MPI_PROC_NULL,
                                  vct->getYright_neighbor_P() == MPI_PROC_NULL,
                                  vct->getZleft_neighbor_P() == MPI_PROC_NULL,
                                  vct->getZright_neighbor_P() == MPI_PROC_NULL,
                                  solverStream_);

    // Phase 3: Batched copy-style node halo exchange — shared → ghost nodes
    gpuBatchedHaloExchange(ptrs, nFields, nxn, nyn, nzn,
                           /*offsetZero=*/false, /*isFaceOnly=*/false,
                           /*needInterp=*/false, /*isParticle=*/true,
                           solverStream_);
  }
}

// =========================================================================
//  GPU setZeroDerivedMoments
// =========================================================================

void EMfields3D::gpuSetZeroDerivedMoments() {
  // d_Jx/y/z:  gpuSumOverSpeciesJ() self-zeroes before accumulating, and is
  //            only called when writeJTot=true, the same condition under which
  //            getJx() is ever read.  No pre-zero needed here.
  // d_Jxh/y/zh: gpuCalculateHatFunctions() calls gpuSetAll0_N() on these
  //             before any accumulation.  No pre-zero needed here.
  // d_rhoc:    gpuSmooth() issues an MPI ghost exchange before the stencil
  //            reads ghost cells; even when Smooth==1.0 the stale ghost cells
  //            only flow into d_rhoh ghost cells which are overwritten by
  //            gpuCommunicateCenterBC_P.  No pre-zero needed here.
  // d_rhoh:    gpuEq(centSize) overwrites all elements, then
  //            gpuCommunicateCenterBC_P overwrites ghost cells.  No pre-zero
  //            needed.
  // d_rhon:    gpuSumOverSpecies() accumulates with +=, so must be pre-zeroed.
  d_rhon.setAll(0.0, solverStream_);
}

// =========================================================================
//  GPU sumOverSpecies: rhon = Σ_s rhons_s
// =========================================================================

void EMfields3D::gpuSumOverSpecies() {
  size_t nodeSize = (size_t)nxn * nyn * nzn;
  for (int is = 0; is < ns; is++)
    gpuSum(d_rhon.devPtr(), d_rhons.speciesPtr(is), nodeSize, solverStream_);
}

// =========================================================================
//  GPU sumOverSpeciesJ: J = sum_s(J_s)
// =========================================================================

void EMfields3D::gpuSumOverSpeciesJ() {
  size_t nodeSize = (size_t)nxn * nyn * nzn;

  d_Jx.setAll(0.0, solverStream_);
  d_Jy.setAll(0.0, solverStream_);
  d_Jz.setAll(0.0, solverStream_);

  for (int is = 0; is < ns; is++) {
    gpuSum(d_Jx.devPtr(), d_Jxs.speciesPtr(is), nodeSize, solverStream_);
    gpuSum(d_Jy.devPtr(), d_Jys.speciesPtr(is), nodeSize, solverStream_);
    gpuSum(d_Jz.devPtr(), d_Jzs.speciesPtr(is), nodeSize, solverStream_);
  }
}

// =========================================================================
//  GPU interpDensitiesN2C: rhoc = interp(rhon)
// =========================================================================

void EMfields3D::gpuInterpDensitiesN2C() {
  gpuInterpN2C(d_rhoc.devPtr(), d_rhon.devPtr(), nxc, nyc, nzc, solverStream_);
}

// =========================================================================
//  GPU OpenBoundaryInflowESource: zero source RHS on open inflow faces
// =========================================================================

void EMfields3D::gpuOpenBoundaryInflowESource(cudaSolverType* dX,
                                              cudaSolverType* dY,
                                              cudaSolverType* dZ, int nx,
                                              int ny, int nz) {
  const VirtualTopology3D* vct = &get_vct();
  const Collective* col = &get_col();

  if (vct->getXleft_neighbor() == MPI_PROC_NULL && bcEMfaceXleft == 2 &&
      col->getBcPfaceXleft() == 2)
    gpuOpenBCZeroFace3(dX, dY, dZ, 0, 1, nx, ny, nz, solverStream_);
  if (vct->getXright_neighbor() == MPI_PROC_NULL && bcEMfaceXright == 2 &&
      col->getBcPfaceXright() == 2)
    gpuOpenBCZeroFace3(dX, dY, dZ, 0, nx - 2, nx, ny, nz, solverStream_);
  if (vct->getYleft_neighbor() == MPI_PROC_NULL && bcEMfaceYleft == 2 &&
      col->getBcPfaceYleft() == 2)
    gpuOpenBCZeroFace3(dX, dY, dZ, 1, 1, nx, ny, nz, solverStream_);
  if (vct->getYright_neighbor() == MPI_PROC_NULL && bcEMfaceYright == 2 &&
      col->getBcPfaceYright() == 2)
    gpuOpenBCZeroFace3(dX, dY, dZ, 1, ny - 2, nx, ny, nz, solverStream_);
  if (vct->getZleft_neighbor() == MPI_PROC_NULL && bcEMfaceZleft == 2 &&
      col->getBcPfaceZleft() == 2)
    gpuOpenBCZeroFace3(dX, dY, dZ, 2, 1, nx, ny, nz, solverStream_);
  if (vct->getZright_neighbor() == MPI_PROC_NULL && bcEMfaceZright == 2 &&
      col->getBcPfaceZright() == 2)
    gpuOpenBCZeroFace3(dX, dY, dZ, 2, nz - 2, nx, ny, nz, solverStream_);
}

// =========================================================================
//  GPU OpenBoundaryInflowEImage: image = vect - injE on open inflow faces
// =========================================================================

void EMfields3D::gpuOpenBoundaryInflowEImage(
    cudaSolverType* imX, cudaSolverType* imY, cudaSolverType* imZ,
    const cudaSolverType* vX, const cudaSolverType* vY,
    const cudaSolverType* vZ, int nx, int ny, int nz) {
  const VirtualTopology3D* vct = &get_vct();
  const Collective* col = &get_col();

  // injE = -(ue0,ve0,we0) × (B0x,B0y,B0z)
  double injE[3];
  injE[0] = -(ve0 * B0z - we0 * B0y);
  injE[1] = -(we0 * B0x - ue0 * B0z);
  injE[2] = -(ue0 * B0y - ve0 * B0x);

  if (vct->getXleft_neighbor() == MPI_PROC_NULL && bcEMfaceXleft == 2 &&
      col->getBcPfaceXleft() == 2)
    gpuOpenBCImageDiffFace3(imX, imY, imZ, vX, vY, vZ, injE[0], injE[1],
                            injE[2], 0, 1, nx, ny, nz, solverStream_);
  // Xright image disabled (matches CPU)
  if (vct->getYleft_neighbor() == MPI_PROC_NULL && bcEMfaceYleft == 2 &&
      col->getBcPfaceYleft() == 2)
    gpuOpenBCImageDiffFace3(imX, imY, imZ, vX, vY, vZ, injE[0], injE[1],
                            injE[2], 1, 1, nx, ny, nz, solverStream_);
  if (vct->getYright_neighbor() == MPI_PROC_NULL && bcEMfaceYright == 2 &&
      col->getBcPfaceYright() == 2)
    gpuOpenBCImageDiffFace3(imX, imY, imZ, vX, vY, vZ, injE[0], injE[1],
                            injE[2], 1, ny - 2, nx, ny, nz, solverStream_);
  if (vct->getZleft_neighbor() == MPI_PROC_NULL && bcEMfaceZleft == 2 &&
      col->getBcPfaceZleft() == 2)
    gpuOpenBCImageDiffFace3(imX, imY, imZ, vX, vY, vZ, injE[0], injE[1],
                            injE[2], 2, 1, nx, ny, nz, solverStream_);
  if (vct->getZright_neighbor() == MPI_PROC_NULL && bcEMfaceZright == 2 &&
      col->getBcPfaceZright() == 2)
    gpuOpenBCImageDiffFace3(imX, imY, imZ, vX, vY, vZ, injE[0], injE[1],
                            injE[2], 2, nz - 2, nx, ny, nz, solverStream_);
}

// =========================================================================
//  GPU OpenBoundaryInflowE: SAL blend / Dirichlet on E inflow faces
// =========================================================================

void EMfields3D::gpuOpenBoundaryInflowE(cudaSolverType* dX, cudaSolverType* dY,
                                        cudaSolverType* dZ, int nx, int ny,
                                        int nz) {
  const VirtualTopology3D* vct = &get_vct();
  const Collective* col = &get_col();

  double injE[3];
  injE[0] = -(ve0 * B0z - we0 * B0y);
  injE[1] = -(we0 * B0x - ue0 * B0z);
  injE[2] = -(ue0 * B0y - ve0 * B0x);

  double invNL = (n_layers_sal > 0) ? 1.0 / (double)n_layers_sal : 1.0;

  if (yes_sal) {
    // SAL mode: blend E toward injE over n_layers_sal layers
    if (vct->getXleft_neighbor() == MPI_PROC_NULL && bcEMfaceXleft == 2 &&
        col->getBcPfaceXleft() == 2)
      gpuSALBlendLayers3(dX, dY, dZ, injE[0], injE[1], injE[2], 0, 0,
                         n_layers_sal, invNL, 1, nx, ny, nz, solverStream_);
    // Xright: outflow — copy from interior reference plane (bcPface==3)
    if (vct->getXright_neighbor() == MPI_PROC_NULL && bcEMfaceXright == 2 &&
        col->getBcPfaceXright() == 3)
      gpuExtrapolateLayers3(dX, dY, dZ, 0, nx - n_layers_sal - 1, nx - 1,
                            nx - 2 - n_layers_sal, nx, ny, nz, solverStream_);
    if (vct->getYleft_neighbor() == MPI_PROC_NULL && bcEMfaceYleft == 2 &&
        col->getBcPfaceYleft() == 2)
      gpuSALBlendLayers3(dX, dY, dZ, injE[0], injE[1], injE[2], 1, 0,
                         n_layers_sal, invNL, 1, nx, ny, nz, solverStream_);
    if (vct->getYright_neighbor() == MPI_PROC_NULL && bcEMfaceYright == 2 &&
        col->getBcPfaceYright() == 2)
      gpuSALBlendLayers3(dX, dY, dZ, injE[0], injE[1], injE[2], 1,
                         ny - n_layers_sal - 1, ny - 1, invNL, 0, nx, ny, nz,
                         solverStream_);
    if (vct->getZleft_neighbor() == MPI_PROC_NULL && bcEMfaceZleft == 2 &&
        col->getBcPfaceZleft() == 2)
      gpuSALBlendLayers3(dX, dY, dZ, injE[0], injE[1], injE[2], 2, 0,
                         n_layers_sal, invNL, 1, nx, ny, nz, solverStream_);
    if (vct->getZright_neighbor() == MPI_PROC_NULL && bcEMfaceZright == 2 &&
        col->getBcPfaceZright() == 2)
      gpuSALBlendLayers3(dX, dY, dZ, injE[0], injE[1], injE[2], 2,
                         nz - n_layers_sal - 1, nz - 1, invNL, 0, nx, ny, nz,
                         solverStream_);
  } else {
    // No SAL: Dirichlet inflow or extrapolation from interior
    if (vct->getXleft_neighbor() == MPI_PROC_NULL && bcEMfaceXleft == 2 &&
        col->getBcPfaceXleft() == 2)
      gpuSetConstLayers3(dX, dY, dZ, injE[0], injE[1], injE[2], 0, 0,
                         n_layers_sal, nx, ny, nz, solverStream_);
    if (vct->getXright_neighbor() == MPI_PROC_NULL && bcEMfaceXright == 2 &&
        col->getBcPfaceXright() == 3)
      gpuExtrapolateLayers3(dX, dY, dZ, 0, nx - n_layers_sal - 1, nx - 1,
                            nx - 2 - n_layers_sal, nx, ny, nz, solverStream_);
    if (vct->getYleft_neighbor() == MPI_PROC_NULL && bcEMfaceYleft == 2 &&
        col->getBcPfaceYleft() == 2)
      gpuExtrapolateLayers3(dX, dY, dZ, 1, 0, n_layers_sal, n_layers_sal + 1,
                            nx, ny, nz, solverStream_);
    if (vct->getYright_neighbor() == MPI_PROC_NULL && bcEMfaceYright == 2 &&
        col->getBcPfaceYright() == 2)
      gpuExtrapolateLayers3(dX, dY, dZ, 1, ny - n_layers_sal - 1, ny - 1,
                            ny - 2 - n_layers_sal, nx, ny, nz, solverStream_);
    if (vct->getZleft_neighbor() == MPI_PROC_NULL && bcEMfaceZleft == 2 &&
        col->getBcPfaceZleft() == 2)
      gpuExtrapolateLayers3(dX, dY, dZ, 2, 0, n_layers_sal, n_layers_sal + 1,
                            nx, ny, nz, solverStream_);
    if (vct->getZright_neighbor() == MPI_PROC_NULL && bcEMfaceZright == 2 &&
        col->getBcPfaceZright() == 2)
      gpuExtrapolateLayers3(dX, dY, dZ, 2, nz - n_layers_sal - 1, nz - 1,
                            nz - 2 - n_layers_sal, nx, ny, nz, solverStream_);
  }
}

// =========================================================================
//  GPU OpenBoundaryInflowB: SAL blend / extrapolation on center B
// =========================================================================

void EMfields3D::gpuOpenBoundaryInflowB(cudaSolverType* dX, cudaSolverType* dY,
                                        cudaSolverType* dZ, int nx, int ny,
                                        int nz) {
  const VirtualTopology3D* vct = &get_vct();
  double invNL = (n_layers_sal > 0) ? 1.0 / (double)n_layers_sal : 1.0;

  // Xleft
  if (vct->getXleft_neighbor() == MPI_PROC_NULL && bcEMfaceXleft == 2 &&
      nx > 10) {
    if (yes_sal)
      gpuSALBlendLayers3(dX, dY, dZ, B0x, B0y, B0z, 0, 0, n_layers_sal, invNL,
                         1, nx, ny, nz, solverStream_);
    else
      gpuSetConstLayers3(dX, dY, dZ, B0x, B0y, B0z, 0, 0, n_layers_sal, nx, ny,
                         nz, solverStream_);
  }
  // Xright: always extrapolation from interior
  if (vct->getXright_neighbor() == MPI_PROC_NULL && bcEMfaceXright == 2 &&
      nx > 10)
    gpuExtrapolateLayers3(dX, dY, dZ, 0, nx - n_layers_sal - 1, nx - 1,
                          nx - 2 - n_layers_sal, nx, ny, nz, solverStream_);
  // Yleft
  if (vct->getYleft_neighbor() == MPI_PROC_NULL && bcEMfaceYleft == 2 &&
      ny > 10) {
    if (yes_sal)
      gpuSALBlendLayers3(dX, dY, dZ, B0x, B0y, B0z, 1, 0, n_layers_sal, invNL,
                         1, nx, ny, nz, solverStream_);
    else
      gpuExtrapolateLayers3(dX, dY, dZ, 1, 0, n_layers_sal, n_layers_sal + 1,
                            nx, ny, nz, solverStream_);
  }
  // Yright
  if (vct->getYright_neighbor() == MPI_PROC_NULL && bcEMfaceYright == 2 &&
      ny > 10) {
    if (yes_sal)
      gpuSALBlendLayers3(dX, dY, dZ, B0x, B0y, B0z, 1, ny - n_layers_sal - 1,
                         ny - 1, invNL, 0, nx, ny, nz, solverStream_);
    else
      gpuExtrapolateLayers3(dX, dY, dZ, 1, ny - n_layers_sal - 1, ny - 1,
                            ny - 2 - n_layers_sal, nx, ny, nz, solverStream_);
  }
  // Zleft
  if (vct->getZleft_neighbor() == MPI_PROC_NULL && bcEMfaceZleft == 2 &&
      nz > 10) {
    if (yes_sal)
      gpuSALBlendLayers3(dX, dY, dZ, B0x, B0y, B0z, 2, 0, n_layers_sal, invNL,
                         1, nx, ny, nz, solverStream_);
    else
      gpuExtrapolateLayers3(dX, dY, dZ, 2, 0, n_layers_sal, n_layers_sal + 1,
                            nx, ny, nz, solverStream_);
  }
  // Zright
  if (vct->getZright_neighbor() == MPI_PROC_NULL && bcEMfaceZright == 2 &&
      nz > 10) {
    if (yes_sal)
      gpuSALBlendLayers3(dX, dY, dZ, B0x, B0y, B0z, 2, nz - n_layers_sal - 1,
                         nz - 1, invNL, 0, nx, ny, nz, solverStream_);
    else
      gpuExtrapolateLayers3(dX, dY, dZ, 2, nz - n_layers_sal - 1, nz - 1,
                            nz - 2 - n_layers_sal, nx, ny, nz, solverStream_);
  }
}

// =========================================================================
//  GPU fixBcGEM / fixBnGEM / fixBforcefree
// =========================================================================

void EMfields3D::gpuFixBcGEM() {
  const VirtualTopology3D* vct = &get_vct();
  double LyH = Ly / 2.0;
  if (vct->getYright_neighbor() == MPI_PROC_NULL)
    gpuFixBcGEMKernel(d_Bxc.devPtr(), d_Byc.devPtr(), d_Bzc.devPtr(), B0x, B0y,
                      B0z, yStart, dy, LyH, delta, 1, nxc, nyc, nzc,
                      solverStream_);
  if (vct->getYleft_neighbor() == MPI_PROC_NULL)
    gpuFixBcGEMKernel(d_Bxc.devPtr(), d_Byc.devPtr(), d_Bzc.devPtr(), B0x, B0y,
                      B0z, yStart, dy, LyH, delta, 0, nxc, nyc, nzc,
                      solverStream_);
}

void EMfields3D::gpuFixBnGEM() {
  const VirtualTopology3D* vct = &get_vct();
  double LyH = Ly / 2.0;
  if (vct->getYright_neighbor() == MPI_PROC_NULL)
    gpuFixBnGEMKernel(d_Bxn.devPtr(), d_Byn.devPtr(), d_Bzn.devPtr(), B0x, B0y,
                      B0z, yStart, dy, LyH, delta, 1, nxn, nyn, nzn, nyc,
                      solverStream_);
  if (vct->getYleft_neighbor() == MPI_PROC_NULL)
    gpuFixBnGEMKernel(d_Bxn.devPtr(), d_Byn.devPtr(), d_Bzn.devPtr(), B0x, B0y,
                      B0z, yStart, dy, LyH, delta, 0, nxn, nyn, nzn, nyc,
                      solverStream_);
}

void EMfields3D::gpuFixBforcefree() {
  const VirtualTopology3D* vct = &get_vct();
  double LyH = Ly / 2.0;
  if (vct->getYright_neighbor() == MPI_PROC_NULL)
    gpuFixBforcefreeKernel(d_Bxc.devPtr(), d_Byc.devPtr(), d_Bzc.devPtr(), B0x,
                           B0y, B0z, yStart, dy, LyH, delta, 1, nxc, nyc, nzc,
                           solverStream_);
  if (vct->getYleft_neighbor() == MPI_PROC_NULL)
    gpuFixBforcefreeKernel(d_Bxc.devPtr(), d_Byc.devPtr(), d_Bzc.devPtr(), B0x,
                           B0y, B0z, yStart, dy, LyH, delta, 0, nxc, nyc, nzc,
                           solverStream_);
}

// =========================================================================
//  GPU ConstantChargePlanet
// =========================================================================

void EMfields3D::gpuConstantChargePlanet(double R, double x_center,
                                         double y_center, double z_center) {
  for (int is = 0; is < ns; is++) {
    double ff = qom[is] / fabs(qom[is]);
    double val = ff * rhoINIT[is] / FourPI;
    gpuConstantChargePlanetKernel(d_rhons.speciesPtr(is), val, R, x_center,
                                  y_center, z_center, xStart, yStart, zStart,
                                  dx, dy, dz, nxn, nyn, nzn, solverStream_);
  }
}

void EMfields3D::gpuConstantChargePlanet2DPlaneXZ(double R, double x_center,
                                                  double z_center) {
  for (int is = 0; is < ns; is++) {
    double sign_q = qom[is] / fabs(qom[is]);
    double val = sign_q * rhoINIT[is] / FourPI;
    gpuConstantChargePlanet2DKernel(d_rhons.speciesPtr(is), val, R, x_center,
                                    z_center, xStart, zStart, dx, dz, nxn, nyn,
                                    nzn, solverStream_);
  }
}

// =========================================================================
//  GPU PoissonImage: A*x for Poisson solver (center Laplacian)
// =========================================================================

void EMfields3D::gpuPoissonImage(cudaSolverType* d_im, cudaSolverType* d_vec) {
  const Grid* grid = &get_grid();
  double _invdx = grid->get_invdx();
  double _invdy = grid->get_invdy();
  double _invdz = grid->get_invdz();

  d_poissonTemp.setAll(0.0, solverStream_);
  d_poissonIm.setAll(0.0, solverStream_);

  // Krylov → physical
  gpuSolver2Phys1(d_poissonTemp.devPtr(), d_vec, nxc, nyc, nzc, solverStream_);

  // Communicate ghost cells (center box stencil)
  gpuCommunicateCenterBoxStencilBC(nxc, nyc, nzc, d_poissonTemp, 1, 1, 1, 1, 1,
                                   1);

  // Laplacian
  gpuLapC2CKernel(d_poissonIm.devPtr(), d_poissonTemp.devPtr(), nxc, nyc, nzc,
                  _invdx * _invdx, _invdy * _invdy, _invdz * _invdz,
                  solverStream_);

  // Physical → Krylov
  gpuPhys2Solver1(d_im, d_poissonIm.devPtr(), nxc, nyc, nzc, solverStream_);
}

// =========================================================================
//  GPU PoissonImageLocal: -∇²·x (positive eigenvalues, NO communication)
//  Ghost cells remain zero → zero-Dirichlet local problem.
// =========================================================================

void EMfields3D::gpuPoissonImageLocal(cudaSolverType* d_im,
                                      cudaSolverType* d_vec) {
  const Grid* grid = &get_grid();
  double _invdx = grid->get_invdx();
  double _invdy = grid->get_invdy();
  double _invdz = grid->get_invdz();

  const int nPoisson = poissonKrylovSize_;

  d_poissonTemp.setAll(0.0, solverStream_);
  d_poissonIm.setAll(0.0, solverStream_);

  // Krylov → physical (ghost cells stay zero)
  gpuSolver2Phys1(d_poissonTemp.devPtr(), d_vec, nxc, nyc, nzc, solverStream_);

  // NO communication — ghost cells remain zero (zero-Dirichlet BCs)

  // Laplacian: d_poissonIm = ∇²·x
  gpuLapC2CKernel(d_poissonIm.devPtr(), d_poissonTemp.devPtr(), nxc, nyc, nzc,
                  _invdx * _invdx, _invdy * _invdy, _invdz * _invdz,
                  solverStream_);

  // Physical → Krylov, then negate to get -∇²·x (positive eigenvalues)
  gpuPhys2Solver1(d_im, d_poissonIm.devPtr(), nxc, nyc, nzc, solverStream_);
  gpuScale(d_im, -1.0, nPoisson, solverStream_);
}

// =========================================================================
//  Analytic eigenvalue bounds for local -∇² on center grid
//  Formula: λ_k = 4 sin²(k π / (2(N+1))) / h²  for mode k=1..N
//  Total eigenvalue = sum over x,y,z dimensions
//  Computed once per simulation.
// =========================================================================

void EMfields3D::computePoissonChebyshevEigenvalues() {
  if (poissonChebComputed)
    return;

  const Grid* grid = &get_grid();
  double invdx = grid->get_invdx();
  double invdy = grid->get_invdy();
  double invdz = grid->get_invdz();

  int Nx = nxc - 2; // interior center points in x
  int Ny = nyc - 2;
  int Nz = nzc - 2;

  double hx2_inv = invdx * invdx;
  double hy2_inv = invdy * invdy;
  double hz2_inv = invdz * invdz;

  auto eigBounds = [](int N, double h2_inv, double& eMin, double& eMax) {
    double sMin = sin(M_PI / (2.0 * (N + 1)));
    double sMax = sin(N * M_PI / (2.0 * (N + 1)));
    eMin = 4.0 * sMin * sMin * h2_inv;
    eMax = 4.0 * sMax * sMax * h2_inv;
  };

  double eMinX, eMaxX, eMinY, eMaxY, eMinZ, eMaxZ;
  eigBounds(Nx, hx2_inv, eMinX, eMaxX);
  eigBounds(Ny, hy2_inv, eMinY, eMaxY);
  eigBounds(Nz, hz2_inv, eMinZ, eMaxZ);

  double rawEigMin = eMinX + eMinY + eMinZ;
  double rawEigMax = eMaxX + eMaxY + eMaxZ;

  // Apply rescaling (as in chebyshevIterationAlpaka.hpp)
  poissonChebEigMin = rawEigMin * poissonChebRescaleEigMin;
  poissonChebEigMax = rawEigMax * poissonChebRescaleEigMax;

  poissonChebComputed = true;

  const VirtualTopology3D* vct = &get_vct();
  if (vct->getCartesian_rank() == 0)
    printf(
        "  [Poisson Chebyshev] Analytic eigenvalue bounds of local -nabla^2: "
        "raw [%.6g, %.6g], rescaled [%.6g, %.6g]  (rescaleMin=%.4g, "
        "rescaleMax=%.4g, "
        "interior grid %d x %d x %d)\n",
        rawEigMin, rawEigMax, poissonChebEigMin, poissonChebEigMax,
        poissonChebRescaleEigMin, poissonChebRescaleEigMax, Nx, Ny, Nz);
}

// =========================================================================
//  Chebyshev preconditioner for Poisson (communication-free)
//  Approximately solves (-∇²)·x = b via Chebyshev semi-iteration on
//  the local operator gpuPoissonImageLocal (positive eigenvalues).
//  Final result is NEGATED so that x ≈ (∇²)⁻¹ b.
// =========================================================================

void EMfields3D::gpuChebyshevPrecondPoisson(cudaSolverType* d_x,
                                            cudaSolverType* d_b) {
  const int n = poissonKrylovSize_;

  // ---- Lazy workspace allocation (shared with Maxwell Chebyshev) ----
  if (chebAlloc < n) {
    if (d_chebY)
      cudaFree(d_chebY);
    if (d_chebW)
      cudaFree(d_chebW);
    if (d_chebZ)
      cudaFree(d_chebZ);
    if (d_chebTmp)
      cudaFree(d_chebTmp);
    cudaErrChk(cudaMalloc(&d_chebY, (size_t)n * sizeof(cudaSolverType)));
    cudaErrChk(cudaMalloc(&d_chebW, (size_t)n * sizeof(cudaSolverType)));
    cudaErrChk(cudaMalloc(&d_chebZ, (size_t)n * sizeof(cudaSolverType)));
    cudaErrChk(cudaMalloc(&d_chebTmp, (size_t)n * sizeof(cudaSolverType)));
    chebAlloc = n;
  }

  int maxIter = poissonChebMaxIter;
  if (maxIter <= 0) {
    gpuEq(d_x, d_b, n, solverStream_);
    return;
  }

  // ---- Chebyshev parameters (positive eigenvalue bounds of -∇²) ----
  double eigMin = poissonChebEigMin;
  double eigMax = poissonChebEigMax;
  double theta = (eigMin + eigMax) * 0.5;
  double delta = (eigMax - eigMin) * 0.5;
  double sigma = theta / delta;

  double rhoOld = 1.0 / sigma;
  double rho = 1.0 / (2.0 * sigma - rhoOld);

  // ---- Preconditioner: x₀ = 0, so r₀ = b ----
  cudaSolverType* pY = d_chebY;
  cudaSolverType* pZ = d_chebZ;
  cudaSolverType* pAy = d_chebTmp;

  // Step 0+1: z = b/θ,  y = f(b, (-∇²)(b))
  gpuPoissonImageLocal(pAy, d_b);
  gpuChebyshevStep1(pY, pZ, d_b, pAy, theta, delta, rho, n, solverStream_);

  // Steps 2..maxIter
  for (int step = 2; step <= maxIter; step++) {
    rhoOld = rho;
    rho = 1.0 / (2.0 * sigma - rhoOld);

    gpuPoissonImageLocal(pAy, pY);
    gpuChebyshevStepN(pZ, pY, pZ, d_b, pAy, delta, sigma, rho, rhoOld, n,
                      solverStream_);
    std::swap(pZ, pY);
  }

  // Negate: Chebyshev solved (-∇²)⁻¹ b, we need (∇²)⁻¹ b = -(-∇²)⁻¹ b
  gpuScaleCopy(d_x, pY, -1.0, n, solverStream_);
}

// =========================================================================
//  FGMRES(m) for Poisson with Chebyshev preconditioner
// =========================================================================

void EMfields3D::gpuFGMRES_PoissonChebyshev(cudaSolverType* d_x, int n,
                                            cudaSolverType* d_b, int m,
                                            int max_iter, cudaSolverType tol,
                                            MPI_Comm fieldcomm) {
  computePoissonChebyshevEigenvalues();
  gpuEnsureFGMRESWorkspace(m, n);

  gpuFGMRES_impl(this, &EMfields3D::gpuPoissonImage,
                 &EMfields3D::gpuChebyshevPrecondPoisson, d_x, n, d_b, m,
                 max_iter, tol, d_blasScratch, d_gmresV, d_gmresW, gmresVAlloc,
                 d_fgmresZ, fgmresZAlloc, fieldcomm, solverStream_,
                 h_gmresReduceLocal, h_gmresReduceGlobal, h_gmresH, h_gmresG,
                 h_gmresCS, h_gmresSN, h_gmresY, "FGMRES+Poisson");
}

// =========================================================================
//  GPU PoissonCorrection (div(E) cleaning)
// =========================================================================

void EMfields3D::gpuPoissonCorrection(int cycle) {
  if (!PoissonCorrection || cycle % PoissonCorrectionCycle != 0)
    return;

  const Collective* col = &get_col();
  const VirtualTopology3D* vct = &get_vct();
  const Grid* grid = &get_grid();
  double _invdx = grid->get_invdx();
  double _invdy = grid->get_invdy();
  double _invdz = grid->get_invdz();

  const int nPoisson = poissonKrylovSize_;
  size_t nodeSize = (size_t)nxn * nyn * nzn;
  size_t centSize = (size_t)nxc * nyc * nzc;

  // Zero work arrays
  d_xkrylovPoisson_E.setZero(solverStream_);
  d_divE_work.setAll(0.0, solverStream_);
  d_tempC.setAll(0.0, solverStream_);
  d_gradPHIX_work.setAll(0.0, solverStream_);
  d_gradPHIY_work.setAll(0.0, solverStream_);
  d_gradPHIZ_work.setAll(0.0, solverStream_);

  // div(E) on centers
  gpuDivN2C(d_divE_work.devPtr(), d_Ex.devPtr(), d_Ey.devPtr(), d_Ez.devPtr(),
            nxc, nyc, nzc, _invdx, _invdy, _invdz, solverStream_);

  // tempC = -4π * rhoc
  gpuScaleCopy(d_tempC.devPtr(), d_rhoc.devPtr(), -FourPI, centSize,
               solverStream_);

  // divE_work += tempC
  gpuSum(d_divE_work.devPtr(), d_tempC.devPtr(), centSize, solverStream_);

  // RHS → Krylov
  gpuPhys2Solver1(d_bkrylovPoisson_E.devPtr(), d_divE_work.devPtr(), nxc, nyc,
                  nzc, solverStream_);

  if (vct->getCartesian_rank() == 0)
    cout << "*** DIVERGENCE CLEANING div(E) using FGMRES+Chebyshev [GPU] ***"
         << endl;

  // Solve
  MPI_Comm fieldcomm = vct->getFieldComm();
  gpuFGMRES_PoissonChebyshev(d_xkrylovPoisson_E.devPtr(), nPoisson,
                             d_bkrylovPoisson_E.devPtr(), 20, 200, GMREStol,
                             fieldcomm);

  // Solution → physical
  gpuSolver2Phys1(d_PHI.devPtr(), d_xkrylovPoisson_E.devPtr(), nxc, nyc, nzc,
                  solverStream_);
  gpuCommunicateCenterBC(nxc, nyc, nzc, d_PHI, 2, 2, 2, 2, 2, 2);

  // grad(PHI)
  gpuGradC2N(d_gradPHIX_work.devPtr(), d_gradPHIY_work.devPtr(),
             d_gradPHIZ_work.devPtr(), d_PHI.devPtr(), nxn, nyn, nzn, _invdx,
             _invdy, _invdz, solverStream_);

  // E -= grad(PHI)
  gpuSub(d_Ex.devPtr(), d_gradPHIX_work.devPtr(), nodeSize, solverStream_);
  gpuSub(d_Ey.devPtr(), d_gradPHIY_work.devPtr(), nodeSize, solverStream_);
  gpuSub(d_Ez.devPtr(), d_gradPHIZ_work.devPtr(), nodeSize, solverStream_);
}

// =========================================================================
//  GPU applyDivBCleaning
// =========================================================================

void EMfields3D::gpuApplyDivBCleaning() {
  const Collective* col = &get_col();
  const VirtualTopology3D* vct = &get_vct();
  const Grid* grid = &get_grid();
  double _invdx = grid->get_invdx();
  double _invdy = grid->get_invdy();
  double _invdz = grid->get_invdz();

  const int nPoisson = poissonKrylovSize_;

  // Zero work arrays
  d_divBwork.setAll(0.0, solverStream_);
  d_gradPSIX.setAll(0.0, solverStream_);
  d_gradPSIY.setAll(0.0, solverStream_);
  d_gradPSIZ.setAll(0.0, solverStream_);
  d_xkrylovPoisson_B.setZero(solverStream_);

  // div(Bn) on centers
  gpuDivN2C(d_divBwork.devPtr(), d_Bxn.devPtr(), d_Byn.devPtr(), d_Bzn.devPtr(),
            nxc, nyc, nzc, _invdx, _invdy, _invdz, solverStream_);

  // RHS → Krylov
  gpuPhys2Solver1(d_bkrylovPoisson_B.devPtr(), d_divBwork.devPtr(), nxc, nyc,
                  nzc, solverStream_);

  if (vct->getCartesian_rank() == 0)
    cout << "*** DIVERGENCE CLEANING div(B)=0 using FGMRES+Chebyshev [GPU] ***"
         << endl;

  // Solve
  MPI_Comm fieldcomm = vct->getFieldComm();
  gpuFGMRES_PoissonChebyshev(d_xkrylovPoisson_B.devPtr(), nPoisson,
                             d_bkrylovPoisson_B.devPtr(), 20, 200, GMREStol,
                             fieldcomm);

  // Solution → physical
  gpuSolver2Phys1(d_PSI.devPtr(), d_xkrylovPoisson_B.devPtr(), nxc, nyc, nzc,
                  solverStream_);
  gpuCommunicateCenterBC(nxc, nyc, nzc, d_PSI, 2, 2, 2, 2, 2, 2);

  // grad(PSI)
  gpuGradC2N(d_gradPSIX.devPtr(), d_gradPSIY.devPtr(), d_gradPSIZ.devPtr(),
             d_PSI.devPtr(), nxn, nyn, nzn, _invdx, _invdy, _invdz,
             solverStream_);

  // Subtract grad(PSI) from Bn on boundary layers
  if (vct->getXleft_neighbor() == MPI_PROC_NULL && bcEMfaceXleft == 2)
    gpuSubBoundaryLayers3(d_Bxn.devPtr(), d_Byn.devPtr(), d_Bzn.devPtr(),
                          d_gradPSIX.devPtr(), d_gradPSIY.devPtr(),
                          d_gradPSIZ.devPtr(), 0, 0, n_layers_sal - 1, nxn, nyn,
                          nzn, solverStream_);
  if (vct->getXright_neighbor() == MPI_PROC_NULL && bcEMfaceXright == 2)
    gpuSubBoundaryLayers3(d_Bxn.devPtr(), d_Byn.devPtr(), d_Bzn.devPtr(),
                          d_gradPSIX.devPtr(), d_gradPSIY.devPtr(),
                          d_gradPSIZ.devPtr(), 0, nxn - n_layers_sal, nxn - 1,
                          nxn, nyn, nzn, solverStream_);
  if (vct->getYleft_neighbor() == MPI_PROC_NULL && bcEMfaceYleft == 2)
    gpuSubBoundaryLayers3(d_Bxn.devPtr(), d_Byn.devPtr(), d_Bzn.devPtr(),
                          d_gradPSIX.devPtr(), d_gradPSIY.devPtr(),
                          d_gradPSIZ.devPtr(), 1, 0, n_layers_sal - 1, nxn, nyn,
                          nzn, solverStream_);
  if (vct->getYright_neighbor() == MPI_PROC_NULL && bcEMfaceYright == 2)
    gpuSubBoundaryLayers3(d_Bxn.devPtr(), d_Byn.devPtr(), d_Bzn.devPtr(),
                          d_gradPSIX.devPtr(), d_gradPSIY.devPtr(),
                          d_gradPSIZ.devPtr(), 1, nyn - n_layers_sal, nyn - 1,
                          nxn, nyn, nzn, solverStream_);
  if (vct->getZleft_neighbor() == MPI_PROC_NULL && bcEMfaceZleft == 2)
    gpuSubBoundaryLayers3(d_Bxn.devPtr(), d_Byn.devPtr(), d_Bzn.devPtr(),
                          d_gradPSIX.devPtr(), d_gradPSIY.devPtr(),
                          d_gradPSIZ.devPtr(), 2, 0, n_layers_sal - 1, nxn, nyn,
                          nzn, solverStream_);
  if (vct->getZright_neighbor() == MPI_PROC_NULL && bcEMfaceZright == 2)
    gpuSubBoundaryLayers3(d_Bxn.devPtr(), d_Byn.devPtr(), d_Bzn.devPtr(),
                          d_gradPSIX.devPtr(), d_gradPSIY.devPtr(),
                          d_gradPSIZ.devPtr(), 2, nzn - n_layers_sal, nzn - 1,
                          nxn, nyn, nzn, solverStream_);

  // Communicate corrected Bn (batched: 3 fields in 1 MPI round)
  gpuCommunicateNodeBC_3mixed(nxn, nyn, nzn, d_Bxn, col->bcBx, d_Byn, col->bcBy,
                              d_Bzn, col->bcBz);

  // Recompute center B from corrected node B
  gpuInterpN2C(d_Bxc.devPtr(), d_Bxn.devPtr(), nxc, nyc, nzc, solverStream_);
  gpuInterpN2C(d_Byc.devPtr(), d_Byn.devPtr(), nxc, nyc, nzc, solverStream_);
  gpuInterpN2C(d_Bzc.devPtr(), d_Bzn.devPtr(), nxc, nyc, nzc, solverStream_);

  // Communicate corrected center B (batched: 3 fields in 1 MPI round)
  gpuCommunicateCenterBC_3mixed(nxc, nyc, nzc, d_Bxc, col->bcBx, d_Byc,
                                col->bcBy, d_Bzc, col->bcBz);
}

#endif // GPU_SOLVER
