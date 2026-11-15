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
//  EMfields3DGPU.cpp  –  field solver of EMfields3D, on the GPU: device
//  arrays, host <-> device sync, Maxwell operators and solvers, the div(E)
//  and div(B) cleanings, moment post-processing and the field boundary
//  conditions.
// =========================================================================

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

#include <algorithm>
#include <chrono>
#include <climits>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
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

  // Boundary kernels index their layer count directly (n_layers_sal, or
  // layersXright() on X-right).  Validate active physical boundary branches
  // against their local array extent.
  auto validateLayers = [&](bool active, int layers, int extent, int reserved,
                            const char* operation) {
    if (!active)
      return;
    const int minLayers = yes_sal ? 1 : 0;
    const int maxLayers = extent - reserved;
    if (layers >= minLayers && layers <= maxLayers)
      return;
    cerr << "ERROR: invalid boundary layer count " << layers
         << " (n_layers_sal=" << n_layers_sal << ") for " << operation
         << " on rank " << vct->getCartesian_rank()
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

  validateLayers(xLeftOpen && col->getBcPfaceXleft() == 2, n_layers_sal, nxn, 1,
                 "electric X-left open boundary");
  validateLayers(xRightOpen && col->getBcPfaceXright() == 3, layersXright(),
                 nxn, 2, "electric X-right open boundary");
  validateLayers(yLeftOpen && col->getBcPfaceYleft() == 2, n_layers_sal, nyn,
                 yes_sal ? 1 : 2, "electric Y-left open boundary");
  validateLayers(yRightOpen && col->getBcPfaceYright() == 2, n_layers_sal, nyn,
                 yes_sal ? 1 : 2, "electric Y-right open boundary");
  validateLayers(zLeftOpen && col->getBcPfaceZleft() == 2, n_layers_sal, nzn,
                 yes_sal ? 1 : 2, "electric Z-left open boundary");
  validateLayers(zRightOpen && col->getBcPfaceZright() == 2, n_layers_sal, nzn,
                 yes_sal ? 1 : 2, "electric Z-right open boundary");

  // Center-B open-boundary extrapolation is active only for extents greater
  // than 10 and requires one untouched reference layer beyond the corrected
  // range.
  validateLayers(xLeftOpen && nxc > 10, n_layers_sal, nxc, 1,
                 "magnetic X-left open boundary");
  validateLayers(xRightOpen && nxc > 10, layersXright(), nxc, 2,
                 "magnetic X-right open boundary");
  validateLayers(yLeftOpen && nyc > 10, n_layers_sal, nyc, yes_sal ? 1 : 2,
                 "magnetic Y-left open boundary");
  validateLayers(yRightOpen && nyc > 10, n_layers_sal, nyc, yes_sal ? 1 : 2,
                 "magnetic Y-right open boundary");
  validateLayers(zLeftOpen && nzc > 10, n_layers_sal, nzc, yes_sal ? 1 : 2,
                 "magnetic Z-left open boundary");
  validateLayers(zRightOpen && nzc > 10, n_layers_sal, nzc, yes_sal ? 1 : 2,
                 "magnetic Z-right open boundary");

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

  // The device memory estimate (Collective::Print, scripts/ipic_preflight.py)
  // counts the arrays allocated below: keep it in step with them.

  // ---- Electric field (node-based) ----
  d_Ex = GPUFieldArray3(nxn, nyn, nzn);
  d_Ey = GPUFieldArray3(nxn, nyn, nzn);
  d_Ez = GPUFieldArray3(nxn, nyn, nzn);
  d_Exth = GPUFieldArray3(nxn, nyn, nzn);
  d_Eyth = GPUFieldArray3(nxn, nyn, nzn);
  d_Ezth = GPUFieldArray3(nxn, nyn, nzn);
  if (smoothingOn()) { // without smoothing the mover samples Exth itself
    d_ExthSmooth = GPUFieldArray3(nxn, nyn, nzn);
    d_EythSmooth = GPUFieldArray3(nxn, nyn, nzn);
    d_EzthSmooth = GPUFieldArray3(nxn, nyn, nzn);
  }

  // ---- Magnetic field ----
  d_Bxc = GPUFieldArray3(nxc, nyc, nzc);
  d_Byc = GPUFieldArray3(nxc, nyc, nzc);
  d_Bzc = GPUFieldArray3(nxc, nyc, nzc);
  d_Bxn = GPUFieldArray3(nxn, nyn, nzn);
  d_Byn = GPUFieldArray3(nxn, nyn, nzn);
  d_Bzn = GPUFieldArray3(nxn, nyn, nzn);

  // ---- Charge / current densities summed over species ----
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

  // ---- Potentials of the cleanings: PHI on the centres (copied to the host
  // for output), PSI on the nodes (div(B) cleaning only) ----
  d_PHI = GPUFieldArray3(nxc, nyc, nzc);
  if (divBCorrection)
    d_PSI = GPUFieldArray3(nxn, nyn, nzn);

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
  d_divD = GPUFieldArray3(nxc, nyc, nzc);
  d_Dx = GPUFieldArray3(nxn, nyn, nzn);
  d_Dy = GPUFieldArray3(nxn, nyn, nzn);
  d_Dz = GPUFieldArray3(nxn, nyn, nzn);
  d_vectX = GPUFieldArray3(nxn, nyn, nzn);
  d_vectY = GPUFieldArray3(nxn, nyn, nzn);
  d_vectZ = GPUFieldArray3(nxn, nyn, nzn);
  d_divC = GPUFieldArray3(nxc, nyc, nzc);
  d_divBwork = GPUFieldArray3(nxc, nyc, nzc);
  d_divE_work = GPUFieldArray3(nxc, nyc, nzc);
  d_poissonTemp = GPUFieldArray3(nxc, nyc, nzc);
  d_poissonIm = GPUFieldArray3(nxc, nyc, nzc);

  // ---- Krylov vectors ----
  const int nMaxwellKrylov = maxwellKrylovSize_;
  d_xkrylovMaxwell = GPUKrylovVector(nMaxwellKrylov);
  d_bkrylovMaxwell = GPUKrylovVector(nMaxwellKrylov);

  // ---- Smooth temp buffer ----
  d_smoothTemp = GPUFieldArray3(nxn, nyn, nzn);

  // ---- Device copy of qom ----
  cudaErrChk(cudaMalloc(&d_qom, ns * sizeof(cudaSolverType)));
  cudaErrChk(cudaMemcpy(d_qom, qom, ns * sizeof(cudaSolverType),
                        cudaMemcpyHostToDevice));

  // ---- BLAS reduction scratch ----
  // k + 2 <= KRYLOV_M_MAX + 1 values for gpuBatchedDotNorm in an Arnoldi
  // step, 1 + the fields of cleaningAtRoundOff, 1 for gpuDot_async /
  // gpuNorm2_async.
  cudaErrChk(
      cudaMalloc(&d_blasScratch, (KRYLOV_M_MAX + 2) * sizeof(cudaSolverType)));

  // ---- Pinned host buffers for GMRES reductions ----
  constexpr size_t mp1 = KRYLOV_M_MAX + 1, word = sizeof(cudaSolverType);
  cudaErrChk(
      cudaHostAlloc(&h_gmresReduceLocal, mp1 * word, cudaHostAllocDefault));
  cudaErrChk(
      cudaHostAlloc(&h_gmresReduceGlobal, mp1 * word, cudaHostAllocDefault));
  cudaErrChk(cudaHostAlloc(&h_gmresH, mp1 * KRYLOV_M_MAX * word,
                           cudaHostAllocDefault));
  cudaErrChk(cudaHostAlloc(&h_gmresG, mp1 * word, cudaHostAllocDefault));
  cudaErrChk(
      cudaHostAlloc(&h_gmresCS, KRYLOV_M_MAX * word, cudaHostAllocDefault));
  cudaErrChk(
      cudaHostAlloc(&h_gmresSN, KRYLOV_M_MAX * word, cudaHostAllocDefault));
  cudaErrChk(cudaHostAlloc(&h_gmresY, mp1 * word, cudaHostAllocDefault));

  // ---- Persistent GMRES workspace ----
  // the longest Krylov vectors: the Maxwell ones (3 x interior nodes)
  const int nGMRESKrylov = maxwellKrylovSize_;
  cudaErrChk(cudaMalloc(&d_gmresV, (size_t)GMRES_MP1 * nGMRESKrylov *
                                       sizeof(cudaSolverType)));
  cudaErrChk(
      cudaMalloc(&d_gmresW, (size_t)nGMRESKrylov * sizeof(cudaSolverType)));
  gmresVAlloc = static_cast<size_t>(GMRES_MP1) * nGMRESKrylov;

  // ---- FGMRES workspace (allocated on first FGMRES use, then reused) ----
  d_fgmresZ = nullptr;
  fgmresZAlloc = 0;

  // ---- Chebyshev solver workspace (4 Maxwell vectors, lazy alloc) ----
  d_chebY = nullptr;
  d_chebW = nullptr;
  d_chebZ = nullptr;
  d_chebTmp = nullptr;
  chebAlloc = 0;

  // ---- Dedicated non-blocking solver stream ----
  cudaErrChk(cudaStreamCreateWithFlags(&solverStream_, cudaStreamNonBlocking));

  // ---- Persistent batched halo-exchange buffers ----
  gpuAllocateHaloBuffers();

  // ---- Launch sets of the Maxwell image and the cleaning solves ----
  {
    const int nCell[3] = {nxc, nyc, nzc};
    const int left[3] = {vct->getXleft_neighbor(), vct->getYleft_neighbor(),
                         vct->getZleft_neighbor()};
    const int right[3] = {vct->getXright_neighbor(), vct->getYright_neighbor(),
                          vct->getZright_neighbor()};
    bool flat[3];
    for (int a = 0; a < 3; a++)
      flat[a] = flatAxis(a);
    imageBoxes_ =
        maxwellImageBoxes(nCell, operatorCellEnd(), flat, haloOverlap_);

    // The cleaning solves (GPUCleaning.cuh), when on.  Null spaces: the
    // constants for div(E) with every axis periodic and for div(B) when every
    // face is periodic or a perfect conductor; the checkerboards of even
    // periodic axes.
    if (PoissonCorrection || divBCorrection) {
      const Grid* grid = &get_grid();
      const double invh2[3] = {grid->get_invdx() * grid->get_invdx(),
                               grid->get_invdy() * grid->get_invdy(),
                               grid->get_invdz() * grid->get_invdz()};
      const bool periodic[3] = {vct->getPERIODICX(), vct->getPERIODICY(),
                                vct->getPERIODICZ()};
      const int N[3] = {col->getNxc(), col->getNyc(), col->getNzc()};
      const int offset[3] = {grid->getCellOffsetX(), grid->getCellOffsetY(),
                             grid->getCellOffsetZ()};
      if (PoissonCorrection) {
        const bool none[6] = {}, notShared[3] = {};
        const int cellRows[3] = {nxc - 2, nyc - 2, nzc - 2};
        CleaningSolve& s = divE_;
        s.op = lap27Op(cellRows, invh2, flat, none, none, false);
        s.rows = lap27Rows(nCell, flat, haloOverlap_);
        s.null = cleaningNullSpace(s.op, false, N, offset, periodic, notShared,
                                   periodic[0] && periodic[1] && periodic[2]);
        s.cheb = cleaningChebBounds(s.op, "div(E)");
        s.buf = &d_PHI;
        s.cells = true;
        s.name = "div(E)";
        s.label = "FGMRES+div(E)";
        s.x = GPUKrylovVector(divEKrylovSize_);
        s.b = GPUKrylovVector(divEKrylovSize_);
      }
      if (divBCorrection) {
        const int nB[3] = {divBExtent(0), divBExtent(1), divBExtent(2)};
        const int nodeRows[3] = {nB[0] - 2, nB[1] - 2, nB[2] - 2};
        const int bcEM[6] = {bcEMfaceXleft,  bcEMfaceXright, bcEMfaceYleft,
                             bcEMfaceYright, bcEMfaceZleft,  bcEMfaceZright};
        bool faceRow[6], meanNull = true, sharedRight[3];
        for (int f = 0; f < 6; f++)
          faceRow[f] = (f % 2 ? right : left)[f / 2] == MPI_PROC_NULL;
        for (int a = 0; a < 3; a++) {
          meanNull = meanNull && (periodic[a] ||
                                  (bcEM[2 * a] == 0 && bcEM[2 * a + 1] == 0));
          // the last node row is the right neighbour's (a twin axis: one row)
          sharedRight[a] = right[a] != MPI_PROC_NULL && !flat[a];
        }
        CleaningSolve& s = divB_;
        s.op = lap27Op(nodeRows, invh2, flat, faceRow, conductorFace_, true);
        s.rows = lap27Rows(nB, flat, haloOverlap_);
        s.null = cleaningNullSpace(s.op, true, N, offset, periodic, sharedRight,
                                   meanNull);
        s.cheb = cleaningChebBounds(s.op, "div(B)");
        s.buf = &d_PSI;
        s.cells = false;
        s.name = "div(B)";
        s.label = "FGMRES+div(B)";
        s.x = GPUKrylovVector(divBKrylovSize_);
        s.b = GPUKrylovVector(divBKrylovSize_);
      }
      cudaErrChk(cudaMalloc(&d_cleanLines_,
                            2 * cleanLinesSlots() * sizeof(cudaSolverType)));
      // the Chebyshev iterates, for the longer solve (an unset one has size 0)
      cleanChebLength_ = (int)std::max(divE_.b.size(), divB_.b.size());
      cudaErrChk(cudaMalloc(&d_cleanCheb_, 2 * (size_t)cleanChebLength_ *
                                               sizeof(cudaSolverType)));
    }
  }

  // The arrays are zeroed at allocation on the default stream, which the
  // non-blocking solver stream does not wait for: finish before first use.
  cudaErrChk(cudaDeviceSynchronize());

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
  d_ExthSmooth.free();
  d_EythSmooth.free();
  d_EzthSmooth.free();

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
  d_divD.free();
  d_Dx.free();
  d_Dy.free();
  d_Dz.free();
  d_vectX.free();
  d_vectY.free();
  d_vectZ.free();
  d_divC.free();

  d_divBwork.free();
  d_divE_work.free();
  d_poissonTemp.free();
  d_poissonIm.free();

  // Krylov
  d_xkrylovMaxwell.free();
  d_bkrylovMaxwell.free();
  divE_.x.free();
  divE_.b.free();
  divB_.x.free();
  divB_.b.free();

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

  if (d_cleanLines_) {
    cudaFree(d_cleanLines_);
    d_cleanLines_ = nullptr;
  }
  if (d_cleanCheb_) {
    cudaFree(d_cleanCheb_);
    d_cleanCheb_ = nullptr;
  }
  cleanChebLength_ = 0;

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
  if (d_blockJacobiDinv) {
    cudaFree(d_blockJacobiDinv);
    d_blockJacobiDinv = nullptr;
  }
  blockJacobiDinvAlloc = 0;
  blockJacobiDinvStale = true;

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

    // Split exchange resources (HaloOverlap) belong to the halo subsystem, so
    // tests and communication-only users do not need the full field solver.
    if (haloOverlap_) {
      int leastPriority = 0;
      int greatestPriority = 0;
      cudaErrChk(
          cudaDeviceGetStreamPriorityRange(&leastPriority, &greatestPriority));
      cudaErrChk(cudaStreamCreateWithPriority(
          &haloStream_, cudaStreamNonBlocking, greatestPriority));
      cudaErrChk(
          cudaEventCreateWithFlags(&haloInputReady_, cudaEventDisableTiming));
      cudaErrChk(
          cudaEventCreateWithFlags(&haloPhaseReady_, cudaEventDisableTiming));
      cudaErrChk(
          cudaEventCreateWithFlags(&haloBuffersReady_, cudaEventDisableTiming));
    }
  } catch (...) {
    gpuFreeHaloBuffers();
    throw;
  }
}

void EMfields3D::gpuFreeHaloBuffers() {
  if (!haloBufsAllocated_)
    return;

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

  haloBufsAllocated_ = false;
}

void EMfields3D::gpuSolverSyncH2D(cudaStream_t stream) {
  // Electric field
  d_Ex.copyFromHostAsync(Ex.fetch_arr(), stream);
  d_Ey.copyFromHostAsync(Ey.fetch_arr(), stream);
  d_Ez.copyFromHostAsync(Ez.fetch_arr(), stream);

  // Magnetic field
  d_Bxc.copyFromHostAsync(Bxc.fetch_arr(), stream);
  d_Byc.copyFromHostAsync(Byc.fetch_arr(), stream);
  d_Bzc.copyFromHostAsync(Bzc.fetch_arr(), stream);
  d_Bxn.copyFromHostAsync(Bxn.fetch_arr(), stream);
  d_Byn.copyFromHostAsync(Byn.fetch_arr(), stream);
  d_Bzn.copyFromHostAsync(Bzn.fetch_arr(), stream);

  // Charge / current densities
  d_rhon.copyFromHostAsync(rhon.fetch_arr(), stream);
  d_Jx.copyFromHostAsync(Jx.fetch_arr(), stream);
  d_Jy.copyFromHostAsync(Jy.fetch_arr(), stream);
  d_Jz.copyFromHostAsync(Jz.fetch_arr(), stream);

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

  // Potential
  d_PHI.copyFromHostAsync(PHI.fetch_arr(), stream);

  // External B
  d_Bx_ext.copyFromHostAsync(Bx_ext.fetch_arr(), stream);
  d_By_ext.copyFromHostAsync(By_ext.fetch_arr(), stream);
  d_Bz_ext.copyFromHostAsync(Bz_ext.fetch_arr(), stream);

  cudaStreamSynchronize(stream);
}

void EMfields3D::gpuSolverSyncD2H(cudaStream_t stream) {
  // Fields and moments needed on the host by output, diagnostics, restart
  // checkpoints and Catalyst
  d_Ex.copyToHostAsync(Ex.fetch_arr(), stream);
  d_Ey.copyToHostAsync(Ey.fetch_arr(), stream);
  d_Ez.copyToHostAsync(Ez.fetch_arr(), stream);

  d_Bxc.copyToHostAsync(Bxc.fetch_arr(), stream);
  d_Byc.copyToHostAsync(Byc.fetch_arr(), stream);
  d_Bzc.copyToHostAsync(Bzc.fetch_arr(), stream);
  d_Bxn.copyToHostAsync(Bxn.fetch_arr(), stream);
  d_Byn.copyToHostAsync(Byn.fetch_arr(), stream);
  d_Bzn.copyToHostAsync(Bzn.fetch_arr(), stream);

  d_rhon.copyToHostAsync(rhon.fetch_arr(), stream);
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

  cudaStreamSynchronize(stream);
}

// =========================================================================
//  GPU Solver: physics method implementations
// =========================================================================

void EMfields3D::gpuMUdot(GPUFieldArray3& MUdotX, GPUFieldArray3& MUdotY,
                          GPUFieldArray3& MUdotZ, GPUFieldArray3& vX,
                          GPUFieldArray3& vY, GPUFieldArray3& vZ) {
  gpuMUdot(MUdotX, MUdotY, MUdotZ, vX, vY, vZ, imageBoxes_.mudot);
}

template <typename Points>
void EMfields3D::gpuMUdot(GPUFieldArray3& MUdotX, GPUFieldArray3& MUdotY,
                          GPUFieldArray3& MUdotZ, GPUFieldArray3& vX,
                          GPUFieldArray3& vY, GPUFieldArray3& vZ,
                          const Points& points) {
  ::gpuMUdot(MUdotX.devPtr(), MUdotY.devPtr(), MUdotZ.devPtr(), vX.devPtr(),
             vY.devPtr(), vZ.devPtr(), d_Bxn.devPtr(), d_Byn.devPtr(),
             d_Bzn.devPtr(), d_Bx_ext.devPtr(), d_By_ext.devPtr(),
             d_Bz_ext.devPtr(), d_rhons.devPtr(), qom, ns, dt, c, FourPI, delt,
             nxn, nyn, nzn, points, solverStream_);
}

template void EMfields3D::gpuMUdot<PointBox>(GPUFieldArray3&, GPUFieldArray3&,
                                             GPUFieldArray3&, GPUFieldArray3&,
                                             GPUFieldArray3&, GPUFieldArray3&,
                                             const PointBox&);
template void EMfields3D::gpuMUdot<PointList>(GPUFieldArray3&, GPUFieldArray3&,
                                              GPUFieldArray3&, GPUFieldArray3&,
                                              GPUFieldArray3&, GPUFieldArray3&,
                                              const PointList&);

void EMfields3D::gpuPIdot(GPUFieldArray3& PIX, GPUFieldArray3& PIY,
                          GPUFieldArray3& PIZ, GPUFieldArray3& vX,
                          GPUFieldArray3& vY, GPUFieldArray3& vZ, int is) {
  double beta = 0.5 * qom[is] * dt / c;
  gpuPIdotSpecies(PIX.devPtr(), PIY.devPtr(), PIZ.devPtr(), vX.devPtr(),
                  vY.devPtr(), vZ.devPtr(), d_Bxn.devPtr(), d_Byn.devPtr(),
                  d_Bzn.devPtr(), d_Bx_ext.devPtr(), d_By_ext.devPtr(),
                  d_Bz_ext.devPtr(), beta, nxn, nyn, nzn, solverStream_);
}

// =========================================================================
//  Smoothing passes: rhoc (gpuSmooth), Jhat (gpuSmooth3), E (gpuSmoothE)
// =========================================================================

template <EMfields3D::HaloMode M, int N, typename Begin, typename End,
          typename Fix>
void EMfields3D::gpuSmoothPasses(GPUFieldArray3* const src[N],
                                 GPUFieldArray3* const dst[N],
                                 GPUFieldArray3* const scratch[N], int nx,
                                 int ny, int nz, Begin&& begin, End&& end,
                                 Fix&& fix) {
  const double alpha = Smooth;
  const double beta3D = (1.0 - alpha) / 6.0;
  const int passes = SmoothNiter;
  // Pass p writes dst when passes - 1 - p (+ flip) is even, the scratch
  // otherwise, so the last pass writes dst.  In place with an odd count the
  // first pass would write the array it reads: flip the parity, end in the
  // scratch and copy back once.
  const int flip = (src[0] == dst[0] && passes % 2 == 1) ? 1 : 0;
  GPUFieldArray3* const* in = src;
  for (int p = 0; p < passes; p++) {
    GPUFieldArray3* const* out =
        (passes - 1 - p + flip) % 2 == 0 ? dst : scratch;
    SmoothFields<N> f;
    for (int c = 0; c < N; c++) {
      f.out[c] = out[c]->devPtr();
      f.in[c] = in[c]->devPtr();
    }
    if constexpr (M == HaloMode::Overlap) {
      begin(in);
      gpuSmoothFields<N>(f, nx, ny, nz, alpha, beta3D, SmoothRegion::Interior,
                         solverStream_);
      end(in);
      gpuSmoothFields<N>(f, nx, ny, nz, alpha, beta3D, SmoothRegion::Shell,
                         solverStream_);
    } else {
      end(in);
      gpuSmoothFields<N>(f, nx, ny, nz, alpha, beta3D, SmoothRegion::All,
                         solverStream_);
    }
    fix(out);
    in = out;
  }
  if (in[0] != dst[0])
    for (int c = 0; c < N; c++)
      gpuEq(dst[c]->devPtr(), in[c]->devPtr(), (size_t)nx * ny * nz,
            solverStream_);
}

template <EMfields3D::HaloMode M>
void EMfields3D::gpuSmooth(GPUFieldArray3& arr, int type) {
  if (!smoothingOn())
    return;
  const bool isCenter = (type == 0);
  const int nx = isCenter ? nxc : nxn;
  const int ny = isCenter ? nyc : nyn;
  const int nz = isCenter ? nzc : nzn;

  // the node-sized d_smoothTemp viewed with the shape of arr
  GPUFieldArray3 tmp =
      GPUFieldArray3::wrapDevice(d_smoothTemp.devPtr(), nx, ny, nz);
  GPUFieldArray3* const fields[1] = {&arr};
  GPUFieldArray3* const scratch[1] = {&tmp};
  auto begin = [&](GPUFieldArray3* const* in) {
    cudaSolverType* ptr[1] = {in[0]->devPtr()};
    gpuBatchedHaloBeginExchange(ptr, 1, nx, ny, nz, isCenter, true, true,
                                solverStream_);
  };
  auto end = [&](GPUFieldArray3* const* in) {
    if constexpr (M == HaloMode::Overlap) {
      gpuBatchedHaloEndExchange();
      gpuBCface_P(nx, ny, nz, *in[0], 2, 2, 2, 2, 2, 2, &_vct, solverStream_);
    } else if (isCenter) {
      gpuCommunicateCenterBoxStencilBC_P(nx, ny, nz, *in[0], 2, 2, 2, 2, 2, 2);
    } else {
      gpuCommunicateNodeBoxStencilBC_P(nx, ny, nz, *in[0], 2, 2, 2, 2, 2, 2);
    }
  };
  gpuSmoothPasses<M, 1>(fields, fields, scratch, nx, ny, nz, begin, end,
                        [](GPUFieldArray3* const*) {});
}

template <EMfields3D::HaloMode M>
void EMfields3D::gpuSmoothE(GPUFieldArray3* const src[3],
                            GPUFieldArray3* const dst[3]) {
  // Every physical face continues every component evenly about the face node;
  // the face values the Maxwell rows impose are restored after each pass
  constexpr int m = BC_FACE_NODE_MIRROR;
  const int mirror[6] = {m, m, m, m, m, m};

  GPUFieldArray3* const scratch[3] = {&d_tempX, &d_tempY, &d_tempZ};
  auto begin = [&](GPUFieldArray3* const* in) {
    cudaSolverType* ptrs[3] = {in[0]->devPtr(), in[1]->devPtr(),
                               in[2]->devPtr()};
    gpuBatchedHaloBeginExchange(ptrs, 3, nxn, nyn, nzn, false, true, false,
                                solverStream_);
  };
  auto end = [&](GPUFieldArray3* const* in) {
    if constexpr (M == HaloMode::Overlap) {
      gpuBatchedHaloEndExchange();
      for (int c = 0; c < 3; c++)
        gpuBCface(nxn, nyn, nzn, *in[c], m, m, m, m, m, m, &_vct,
                  solverStream_);
    } else {
      // Batched: 3 fields in 1 MPI round instead of 3 sequential exchanges
      gpuCommunicateNodeBoxStencilBC_3mixed(nxn, nyn, nzn, *in[0], mirror,
                                            *in[1], mirror, *in[2], mirror);
    }
  };
  auto fix = [&](GPUFieldArray3* const* out) {
    gpuImposeFaceE(out[0]->devPtr(), out[1]->devPtr(), out[2]->devPtr());
  };
  gpuSmoothPasses<M, 3>(src, dst, scratch, nxn, nyn, nzn, begin, end, fix);
}

void EMfields3D::gpuImposeFaceE(cudaSolverType* dX, cudaSolverType* dY,
                                cudaSolverType* dZ) {
  // the right-hand sides of the identity rows, in the order of the source
  // (gpuPerfectConductorSources, then gpuOpenBoundaryInflowESource)
  double ebc[3]; // -(u0 x B0) / c
  driftElectricField(ebc);
  const int n[3] = {nxn, nyn, nzn};
  for (int f = 0; f < 6; f++)
    if (conductorFace_[f])
      gpuOpenBCSetFace3(dX, dY, dZ, ebc[0], ebc[1], ebc[2], f / 2,
                        f % 2 ? n[f / 2] - 2 : 1, nxn, nyn, nzn, solverStream_,
                        true); // tangential components only
  gpuOpenBoundaryInflowESource(dX, dY, dZ, nxn, nyn, nzn);
}

template <EMfields3D::HaloMode M>
void EMfields3D::gpuSmooth3(GPUFieldArray3& a1, GPUFieldArray3& a2,
                            GPUFieldArray3& a3, int type) {
  if (!smoothingOn())
    return;
  const bool isCenter = (type == 0);
  const int nx = isCenter ? nxc : nxn;
  const int ny = isCenter ? nyc : nyn;
  const int nz = isCenter ? nzc : nzn;

  // the node-sized d_temp2X/Y/Z viewed with the shape of the fields
  GPUFieldArray3 t1 = GPUFieldArray3::wrapDevice(d_temp2X.devPtr(), nx, ny, nz);
  GPUFieldArray3 t2 = GPUFieldArray3::wrapDevice(d_temp2Y.devPtr(), nx, ny, nz);
  GPUFieldArray3 t3 = GPUFieldArray3::wrapDevice(d_temp2Z.devPtr(), nx, ny, nz);
  GPUFieldArray3* const fields[3] = {&a1, &a2, &a3};
  GPUFieldArray3* const scratch[3] = {&t1, &t2, &t3};
  auto begin = [&](GPUFieldArray3* const* in) {
    cudaSolverType* ptrs[3] = {in[0]->devPtr(), in[1]->devPtr(),
                               in[2]->devPtr()};
    gpuBatchedHaloBeginExchange(ptrs, 3, nx, ny, nz, isCenter, true, true,
                                solverStream_);
  };
  auto end = [&](GPUFieldArray3* const* in) {
    if constexpr (M == HaloMode::Overlap) {
      gpuBatchedHaloEndExchange();
      for (int c = 0; c < 3; c++)
        gpuBCface_P(nx, ny, nz, *in[c], 2, 2, 2, 2, 2, 2, &_vct, solverStream_);
    } else if (isCenter) { // batched: 3 fields in 1 MPI round
      gpuCommunicateCenterBoxStencilBC_P_3(nx, ny, nz, *in[0], *in[1], *in[2],
                                           2, 2, 2, 2, 2, 2);
    } else {
      gpuCommunicateNodeBoxStencilBC_P_3(nx, ny, nz, *in[0], *in[1], *in[2], 2,
                                         2, 2, 2, 2, 2);
    }
  };
  gpuSmoothPasses<M, 3>(fields, fields, scratch, nx, ny, nz, begin, end,
                        [](GPUFieldArray3* const*) {});
}

void EMfields3D::gpuPerfectConductorImages(cudaSolverType* d_im,
                                           GPUFieldArray3& vX,
                                           GPUFieldArray3& vY,
                                           GPUFieldArray3& vZ) {
  for (int f = 0; f < 6; f++) {
    if (!conductorFace_[f])
      continue;
    ::gpuPerfectConductorImage(
        d_im, vX.devPtr(), vY.devPtr(), vZ.devPtr(), d_Bxn.devPtr(),
        d_Byn.devPtr(), d_Bzn.devPtr(), d_Bx_ext.devPtr(), d_By_ext.devPtr(),
        d_Bz_ext.devPtr(), d_rhons.devPtr(), d_qom, ns, dt, c, FourPI, delt,
        nxn, nyn, nzn, f / 2, f % 2 == 1, solverStream_);
  }
}

void EMfields3D::gpuPerfectConductorSources(cudaSolverType* dX,
                                            cudaSolverType* dY,
                                            cudaSolverType* dZ) {
  // ebc = -(u0 x B0) / c: the tangential E on the walls.
  double ebc[3];
  driftElectricField(ebc);
  const cudaSolverType* E[3] = {d_Ex.devPtr(), d_Ey.devPtr(), d_Ez.devPtr()};
  const cudaSolverType* J[3] = {d_Jxh.devPtr(), d_Jyh.devPtr(), d_Jzh.devPtr()};
  for (int f = 0; f < 6; f++) {
    if (!conductorFace_[f])
      continue;
    const int dir = f / 2;
    ::gpuPerfectConductorSource(
        dX, dY, dZ, E[dir], J[dir], d_Bxn.devPtr(), d_Byn.devPtr(),
        d_Bzn.devPtr(), d_Bx_ext.devPtr(), d_By_ext.devPtr(), d_Bz_ext.devPtr(),
        d_rhons.devPtr(), d_qom, ns, dt, c, th, FourPI, delt, ebc[0], ebc[1],
        ebc[2], nxn, nyn, nzn, dir, f % 2 == 1, solverStream_);
  }
}

// =========================================================================
//  gpuMaxwellImage:  im = A * vector  (Krylov ↔ Krylov)
//
//    im = delt² (-lap E - grad div(μ·E)) + μ·E + E on the interior nodes,
//    then the boundary rows (perfect conductor, open inflow faces).
//  D = μ·E on the nodes (gpuMUdot).  Cell level: the 9 gradients of E and
//  div D in one kernel (gpuMaxwellLocalCenterOps), the X-right outflow ghost
//  cell plane included, exchanged between ranks in one halo exchange and set
//  by the face BCs.  Node level: gpuMaxwellLocalNodeFused (divC2N of the
//  gradients, gradC2N of the divergence, the arithmetic and the Krylov
//  packing in one pass), the kernel of the local operator too.  Every value
//  read is written first, so no work array is zeroed.
//
//  Split operator (HaloMode::Overlap, needs MaxwellImageBoxes::split): D and
//  the cell arrays on the boundary layers (with a flat axis: whole), the
//  exchange started, then D, the cell arrays and the image on the interior
//  while it is in flight (they read no value it writes), and the image on
//  the nodes next to the faces once it is complete.  Same values as the
//  blocking exchange.
// =========================================================================

EMfields3D::GpuImageFn EMfields3D::gpuMaxwellImageFn() const {
  return imageBoxes_.split ? &EMfields3D::gpuMaxwellImage<HaloMode::Overlap>
                           : &EMfields3D::gpuMaxwellImage<HaloMode::Blocking>;
}

void EMfields3D::gpuMaxwellImage(cudaSolverType* d_im,
                                 cudaSolverType* d_vector) {
  (this->*gpuMaxwellImageFn())(d_im, d_vector);
}

template <EMfields3D::HaloMode M>
void EMfields3D::gpuMaxwellImage(cudaSolverType* d_im,
                                 cudaSolverType* d_vector) {
  const Grid* grid = &get_grid();
  const double _invdx = grid->get_invdx();
  const double _invdy = grid->get_invdy();
  const double _invdz = grid->get_invdz();
  const MaxwellImageBoxes& box = imageBoxes_;
  if constexpr (M == HaloMode::Overlap) {
    // without split the shell and interior lists are empty
    if (!box.split) {
      cerr << "ERROR: gpuMaxwellImage<Overlap> on a subdomain that does not "
              "split (see MaxwellImageBoxes::split)"
           << endl;
      MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
      std::abort();
    }
  }

  // The 10 cell arrays: the 9 gradients of Ex, Ey, Ez and div(μ·E)
  GPUFieldArray3* cell[10] = {
      &d_tempXC,    &d_tempYC,   &d_tempZC,    &d_divC,  &d_poissonTemp,
      &d_poissonIm, &d_divBwork, &d_divE_work, &d_tempC, &d_divD};
  cudaSolverType* cp[10];
  for (int a = 0; a < 10; a++)
    cp[a] = cell[a]->devPtr();
  GPUFieldArray3* En[3] = {&d_vectX, &d_vectY, &d_vectZ};
  GPUFieldArray3* Dn[3] = {&d_Dx, &d_Dy, &d_Dz};

  // each stage launches a PointList of imageBoxes_ (built once)
  auto mudot = [&](const PointList& points) {
    gpuMUdot(d_Dx, d_Dy, d_Dz, d_vectX, d_vectY, d_vectZ, points);
  };
  auto cellOps = [&](const PointList& points) {
    gpuMaxwellLocalCenterOps(cp[0], cp[1], cp[2], cp[3], cp[4], cp[5], cp[6],
                             cp[7], cp[8], cp[9], d_vectX.devPtr(),
                             d_vectY.devPtr(), d_vectZ.devPtr(), d_Dx.devPtr(),
                             d_Dy.devPtr(), d_Dz.devPtr(), nxc, nyc, nzc,
                             _invdx, _invdy, _invdz, points, solverStream_);
  };
  // face BCs after the exchange: code 1 for the gradients, code 2 for div D
  // (an X-right outflow plane is kept)
  auto faceBCs = [&]() {
    const int bcGrad = operatorBCXright(1), bcDiv = operatorBCXright(2);
    for (int a = 0; a < 9; a++)
      gpuBCface(nxc, nyc, nzc, *cell[a], bcGrad, 1, 1, 1, 1, 1, &_vct,
                solverStream_);
    gpuBCface(nxc, nyc, nzc, d_divD, bcDiv, 2, 2, 2, 2, 2, &_vct,
              solverStream_);
  };
  // node level, straight into the Krylov image
  auto nodeOps = [&](const PointList& points, const CellIndex& index) {
    gpuMaxwellLocalNodeFused(
        d_im, cp[0], cp[1], cp[2], cp[3], cp[4], cp[5], cp[6], cp[7], cp[8],
        cp[9], d_vectX.devPtr(), d_vectY.devPtr(), d_vectZ.devPtr(),
        d_Dx.devPtr(), d_Dy.devPtr(), d_Dz.devPtr(), nxn, nyn, nzn, _invdx,
        _invdy, _invdz, delt * delt, points, index, solverStream_);
  };
  // boundary rows: perfect conductor (linear), open inflow faces (identity)
  auto boundaryRows = [&]() {
    gpuPerfectConductorImages(d_im, d_vectX, d_vectY, d_vectZ);
    if (get_col().getApplyInflowBcsEImage())
      gpuOpenBoundaryInflowEImage(d_im, d_vectX.devPtr(), d_vectY.devPtr(),
                                  d_vectZ.devPtr(), nxn, nyn, nzn);
  };

  // Krylov → physical space; X-right outflow: zero-gradient ghost nodes (the
  // cell stencils include the ghost cell plane)
  gpuSolver2Phys3(d_vectX.devPtr(), d_vectY.devPtr(), d_vectZ.devPtr(),
                  d_vector, nxn, nyn, nzn, solverStream_);
  gpuOutflowGhostNodes(En);

  if constexpr (M == HaloMode::Overlap) {
    // The D the cell shells read, then the cell arrays the exchange sends,
    // one launch each (with a flat axis: both whole)
    mudot(box.mudotShells);
    gpuOutflowGhostNodes(Dn);
    cellOps(box.cellShells);
    gpuOutflowGhostRows(cp, 10);
    gpuBatchedHaloBeginExchange(cp, 10, nxc, nyc, nzc, true, false, false,
                                solverStream_);
    // The interior, queued before the host completes the MPI in
    // EndExchange: it runs while the exchange is in flight
    mudot(box.mudotInterior);
    cellOps(box.cellInterior);
    nodeOps(box.nodeInterior, box.splitCellIndex);
    gpuBatchedHaloEndExchange();
    // The nodes next to the faces read the exchanged ghost cells
    faceBCs();
    nodeOps(box.nodeShells, box.splitCellIndex);
  } else {
    mudot(box.mudot);
    gpuOutflowGhostNodes(Dn);
    cellOps(box.cells);
    gpuOutflowGhostRows(cp, 10);
    gpuBatchedHaloExchange(cp, 10, nxc, nyc, nzc, true, false, false, false,
                           solverStream_);
    faceBCs();
    nodeOps(box.nodes, box.cellIndex);
  }
  boundaryRows();
}

template void
EMfields3D::gpuMaxwellImage<EMfields3D::HaloMode::Blocking>(cudaSolverType*,
                                                            cudaSolverType*);
template void
EMfields3D::gpuMaxwellImage<EMfields3D::HaloMode::Overlap>(cudaSolverType*,
                                                           cudaSolverType*);

// =========================================================================
//  gpuMaxwellImageLocal:  communication-free  im = A * vector
//
//  Computes a local approximation to gpuMaxwellImage:
//    • NO MPI communication  (halo exchanges skipped)
//    • Ghost cells are treated as zero
//    • Physical boundary image corrections are still applied locally
//
//  Intended for use as a GPU-local preconditioner in FGMRES.
//
//  Pipeline:
//    1. gpuSolver2Phys3         — unpack Krylov → vectX/Y/Z
//    2. gpuMUdot                — D = μ̂·E  (all species, one kernel)
//    3. gpuMaxwellLocalCenterOps — fused 3×gradN2C + divN2C
//    4. gpuMaxwellLocalNodeFused — fused 3×divC2N + gradC2N
//                                  + arithmetic + Krylov packing → d_im
//    5. boundary rows (conductor, open inflow faces) written into d_im
//
//  Scratch arrays used (all pre-allocated in gpuSolverAllocate):
//    Node-sized:  d_vectX/Y/Z, d_Dx/Y/Z           (input/work)
//    Center-sized: d_tempXC/YC/ZC                   (grad Ex)
//                  d_divC, d_poissonTemp, d_poissonIm (grad Ey)
//                  d_divBwork, d_divE_work, d_tempC   (grad Ez)
//                  d_divD                             (div D)
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
  // gpuMaxwellLocalCenterOps writes only the cells of its box; ghost cells
  // must be pre-zeroed so the node kernel reads zeros at subdomain
  // boundaries.
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
  cudaErrChk(cudaMemsetAsync(d_divD.devPtr(), 0,
                             centSize * sizeof(cudaSolverType), solverStream_));

  // ---- Step 1: Krylov → physical space ----
  gpuSolver2Phys3(d_vectX.devPtr(), d_vectY.devPtr(), d_vectZ.devPtr(),
                  d_vector, nxn, nyn, nzn, solverStream_);

  // ---- Step 2: MUdot: D = μ̂·E ----
  // Purely local (no communication), loops over species internally.
  // X-right outflow face: zero-gradient ghost nodes of E and D for the ghost
  // cell plane, whose ghost rows stay zero like every other ghost cell here.
  GPUFieldArray3* En[3] = {&d_vectX, &d_vectY, &d_vectZ};
  GPUFieldArray3* Dn[3] = {&d_Dx, &d_Dy, &d_Dz};
  gpuOutflowGhostNodes(En);
  gpuMUdot(d_Dx, d_Dy, d_Dz, d_vectX, d_vectY, d_vectZ, imageBoxes_.mudot);
  gpuOutflowGhostNodes(Dn);

  // ---- Step 3: Fused center-level operations ----
  // Computes 3×gradN2C(Ex,Ey,Ez) → 9 center arrays
  //        + divN2C(Dx,Dy,Dz)    → 1 center array
  // in a single kernel launch, on the cells of the full image.
  gpuMaxwellLocalCenterOps(
      d_tempXC.devPtr(), d_tempYC.devPtr(), d_tempZC.devPtr(),       // grad(Ex)
      d_divC.devPtr(), d_poissonTemp.devPtr(), d_poissonIm.devPtr(), // grad(Ey)
      d_divBwork.devPtr(), d_divE_work.devPtr(), d_tempC.devPtr(),   // grad(Ez)
      d_divD.devPtr(),                                               // div(D)
      d_vectX.devPtr(), d_vectY.devPtr(), d_vectZ.devPtr(), d_Dx.devPtr(),
      d_Dy.devPtr(), d_Dz.devPtr(), nxc, nyc, nzc, _invdx, _invdy, _invdz,
      imageBoxes_.cells, solverStream_);

  // ---- Step 4: Fused node-level operations + Krylov packing ----
  // For each interior node:
  //   lapX = divC2N(grad(Ex))       — from 3 center arrays
  //   lapY = divC2N(grad(Ey))       — from 3 center arrays
  //   lapZ = divC2N(grad(Ez))       — from 3 center arrays
  //   gdivX/Y/Z = gradC2N(div(D))  — from 1 center array
  //   im = delt²·(-lap - gdiv) + D + E
  // Output is packed into Krylov space; the boundary rows follow.
  gpuMaxwellLocalNodeFused(
      d_im, d_tempXC.devPtr(), d_tempYC.devPtr(), d_tempZC.devPtr(), // grad(Ex)
      d_divC.devPtr(), d_poissonTemp.devPtr(), d_poissonIm.devPtr(), // grad(Ey)
      d_divBwork.devPtr(), d_divE_work.devPtr(), d_tempC.devPtr(),   // grad(Ez)
      d_divD.devPtr(),                                               // div(D)
      d_vectX.devPtr(), d_vectY.devPtr(), d_vectZ.devPtr(), d_Dx.devPtr(),
      d_Dy.devPtr(), d_Dz.devPtr(), nxn, nyn, nzn, _invdx, _invdy, _invdz,
      delt * delt, imageBoxes_.nodes, imageBoxes_.cellIndex, solverStream_);

  // Boundary rows as in the full image, written into the Krylov image
  gpuPerfectConductorImages(d_im, d_vectX, d_vectY, d_vectZ);
  if (get_col().getApplyInflowBcsEImage())
    gpuOpenBoundaryInflowEImage(d_im, d_vectX.devPtr(), d_vectY.devPtr(),
                                d_vectZ.devPtr(), nxn, nyn, nzn);
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
//  The recurrence applies A itself, whose eigenvalues lie in
//  [eigMin, eigMax] > 0.  (The cleaning preconditioner, gpuChebyshevCleaning,
//  instead uses -L and negates its result.)
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

  // ---- Print the initial residual norm ----
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
//  gpuMaxwellSource:  build RHS of Maxwell system
// =========================================================================

void EMfields3D::gpuMaxwellSource(cudaSolverType* d_bkrylov, int cycle) {
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

  // Case-specific B fix before the curl (gpuFixBnGEM is not needed: the source
  // curl below reads cell-centered B only)
  if (isForceFreeCase())
    gpuFixBforcefree();

  // Open faces: the exchange above reset their ghost layer, restore it.  The
  // layers themselves were treated once, after Faraday's law (gpuCalculateB).
  gpuOpenBoundaryInflowB(d_Bxc.devPtr(), d_Byc.devPtr(), d_Bzc.devPtr(), nxc,
                         nyc, nzc, true);

  // Case-specific center B fix after OpenBC
  if (isGemCase())
    gpuFixBcGEM();

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

  // Communicate rhoh (on an X-right outflow face rhohat is already 0 from the
  // reference node to the ghost cell, gpuZeroOutflowRhohat)
  gpuCommunicateCenterBC_P(nxc, nyc, nzc, d_rhoh, 2, 2, 2, 2, 2, 2);

  // grad(rhoh) → tempX/Y/Z
  gpuGradC2N(d_tempX.devPtr(), d_tempY.devPtr(), d_tempZ.devPtr(),
             d_rhoh.devPtr(), nxn, nyn, nzn, _invdx, _invdy, _invdz,
             solverStream_);

#if defined(IPIC3D_GPU_CYCLE_DIAGNOSTICS)
  if (gpuCycleDiagnosticsSourceCycle(cycle)) {
    // Only interior nodes enter the Krylov RHS.  Scratch ghost layers from
    // curl/grad stencils are intentionally excluded from this comparison.
    const GPUCycleDiagnosticRegion rhsRegion = {
        nxn,
        nyn,
        nzn,
        {1, 1, 1},
        {nxn - 1, nyn - 1, nzn - 1},
        {grid->getXN(0), grid->getYN(0), grid->getZN(0)},
        {grid->getDX(), grid->getDY(), grid->getDZ()}};
    static const char* const xyzNames[3] = {"x", "y", "z"};
    const cudaSolverType* electric[3] = {d_Ex.devPtr(), d_Ey.devPtr(),
                                         d_Ez.devPtr()};
    const cudaSolverType* curlB[3] = {d_tempXN.devPtr(), d_tempYN.devPtr(),
                                      d_tempZN.devPtr()};
    const cudaSolverType* jhat[3] = {d_Jxh.devPtr(), d_Jyh.devPtr(),
                                     d_Jzh.devPtr()};
    const cudaSolverType* gradRhohat[3] = {d_tempX.devPtr(), d_tempY.devPtr(),
                                           d_tempZ.devPtr()};
    const double electricScale[3] = {1.0, 1.0, 1.0};
    const double curlScale[3] = {delt, delt, delt};
    const double jhatScale[3] = {-delt * FourPI / c, -delt * FourPI / c,
                                 -delt * FourPI / c};
    const double gradScale[3] = {-delt * delt * FourPI, -delt * delt * FourPI,
                                 -delt * delt * FourPI};
    gpuCycleDiagnostics_.reportArraySet(
        "maxwell_source_E_contribution", cycle, -1, xyzNames, electric,
        electricScale, 3, rhsRegion, solverStream_, vct->getFieldComm());
    gpuCycleDiagnostics_.reportArraySet(
        "maxwell_source_curlB_contribution", cycle, -1, xyzNames, curlB,
        curlScale, 3, rhsRegion, solverStream_, vct->getFieldComm());
    gpuCycleDiagnostics_.reportArraySet(
        "maxwell_source_Jhat_contribution", cycle, -1, xyzNames, jhat,
        jhatScale, 3, rhsRegion, solverStream_, vct->getFieldComm());
    gpuCycleDiagnostics_.reportArraySet(
        "maxwell_source_grad_rhohat_contribution", cycle, -1, xyzNames,
        gradRhohat, gradScale, 3, rhsRegion, solverStream_,
        vct->getFieldComm());
  }
#endif

  // temp *= -delt² * 4π  (fused triple)
  gpuScale3(d_tempX.devPtr(), d_tempY.devPtr(), d_tempZ.devPtr(),
            -delt * delt * FourPI, nodeSize, solverStream_);

  // Add E and curl+current parts  (fused: 6 gpuSum → 1 gpuSumAddTwo3)
  gpuSumAddTwo3(d_tempX.devPtr(), d_Ex.devPtr(), d_temp2X.devPtr(),
                d_tempY.devPtr(), d_Ey.devPtr(), d_temp2Y.devPtr(),
                d_tempZ.devPtr(), d_Ez.devPtr(), d_temp2Z.devPtr(), nodeSize,
                solverStream_);

  // Perfect conductor BCs for source (RHS rows of the conductor faces)
  gpuPerfectConductorSources(d_tempX.devPtr(), d_tempY.devPtr(),
                             d_tempZ.devPtr());

  // OpenBC: RHS rows of the open inflow faces (E0; identity image rows)
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
//      are PINNED and PERSISTENT — allocated once in gpuSolverAllocate().
//    • D→H reduction copies use async memcpy into pinned memory.
//    • Stream syncs are batched: ONE sync per Arnoldi step (before
//      MPI_Allreduce) plus ONE for the post-ortho norm.
//    • Device workspace (V, w) is allocated persistently in
//    gpuSolverAllocate().
// =========================================================================

// A NaN or Inf residual means the field or the operator is no longer finite:
// stop instead of iterating to max_iter and applying the result.
static void gmresCheckFinite(double err, const char* label, int restart) {
  if (!std::isfinite(err)) {
    eprintf("%s: non-finite residual (%g) at restart %d", label, err, restart);
    abort();
  }
}

// =========================================================================
//  Field-solver guard
//
//  The values tested are global (reduced over the field communicator), so
//  every rank takes the same decision.  The run stops with MPI_Abort, not a
//  normal finalize: the final checkpoint of CallFinalize would otherwise
//  overwrite a good restart slot with the corrupted state.
// =========================================================================

void EMfields3D::checkMaxwellSource(double normB, int cycle) {
  if (normB < 0.0) // not computed (Chebyshev iteration)
    return;
  const double limit = get_col().getSolverSourceGrowthLimit();
  const double previous = previousMaxwellSourceNorm_;
  previousMaxwellSourceNorm_ = normB;
  const bool finite = std::isfinite(normB);
  // previous == 0: first solve of the run, or a zero source
  const bool jump = limit > 0.0 && previous > 0.0 && normB > limit * previous;
  if (finite && !jump)
    return;
  std::ostringstream reason;
  reason << "the Maxwell right-hand side norm ||b|| is " << normB
         << " at cycle " << cycle;
  if (finite)
    reason << ", " << normB / previous << " times the " << previous
           << " of the previous cycle (SolverSourceGrowthLimit = " << limit
           << ")";
  stopForSolverFailure(reason.str());
}

void EMfields3D::checkSolverConverged(bool converged, const char* solve,
                                      int cycle) {
  if (converged || !get_col().getSolverStopOnNonConvergence())
    return;
  std::ostringstream reason;
  reason << "the " << solve << " solve did not converge at cycle " << cycle
         << " (SolverStopOnNonConvergence = 1)";
  stopForSolverFailure(reason.str());
}

void EMfields3D::stopForSolverFailure(const std::string& reason) const {
  const VirtualTopology3D* vct = &get_vct();
  if (vct->getCartesian_rank() == 0) {
    std::cout
        << "\n"
        << "==============================================================="
           "==========\n"
        << "  SOLVER GUARD: " << reason << ".\n"
        << "  Stopping before this solution is used. No further output "
           "or checkpoint\n"
        << "  is written; the restart slots keep the checkpoints written "
           "before.\n"
        << "==============================================================="
           "==========\n"
        << std::flush;
  }
  // Let rank 0 print before any rank aborts the job.
  MPI_Barrier(vct->getFieldComm());
  MPI_Abort(vct->getFieldComm(), EXIT_FAILURE);
  std::abort(); // not reached
}

// Returns ||b|| and whether the true residual fell below tol * ||r0|| within
// max_iter restarts (or r0 = 0).
static EMfields3D::KrylovSolveStatus
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
  gmresCheckFinite(initial_error, "GMRES", 0);

  // ||b||: printed, and returned for the field-solver guard
  gpuNorm2_async(d_b, n, d_scratch, &h_reduceLocal[0], stream);
  cudaErrChk(cudaStreamSynchronize(stream));
  double normb;
  MPI_Allreduce(&h_reduceLocal[0], &normb, 1, mpiTypeOf<cudaSolverType>(),
                MPI_SUM, fieldcomm);
  EMfields3D::KrylovSolveStatus status;
  status.normB = sqrt(normb);
  if (initial_error < 1e-30)
    return status;

  int gmresRank;
  MPI_Comm_rank(fieldcomm, &gmresRank);
  if (gmresRank == 0)
    printf("Initial residual: %g norm b vector (source) = %g\n", initial_error,
           status.normB);

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

      // Store H[j][k]; w -= sum_j h_jk V[j] in one pass
      for (int j = 0; j <= k; j++)
        H[j * m + k] = h_reduceGlobal[j];
      gpuSubCombination(d_gmresW, d_gmresV, (size_t)n, k + 1, (size_t)n,
                        h_reduceGlobal, stream);

      // H[k+1][k] = ||w_perp||  (post-orthogonalisation norm)
      // Async D→H, single sync
      gpuNorm2_async(d_gmresW, n, d_scratch, &h_reduceLocal[0], stream);
      cudaErrChk(cudaStreamSynchronize(stream));
      double global_wNorm;
      MPI_Allreduce(&h_reduceLocal[0], &global_wNorm, 1,
                    mpiTypeOf<cudaSolverType>(), MPI_SUM, fieldcomm);
      H[(k + 1) * m + k] = sqrt(global_wNorm);

      // Re-orthogonalise when the projections cancelled ||w|| down to
      // round-off of its pre-orthogonalisation norm
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

    // ---- True residual check (the Arnoldi estimate can drift) ----
    (field->*GpuImage)(d_gmresW, d_x);
    gpuEq(d_gmresV, d_b, n, stream);
    gpuSub(d_gmresV, d_gmresW, n, stream);

    gpuNorm2_async(d_gmresV, n, d_scratch, &h_reduceLocal[0], stream);
    cudaErrChk(cudaStreamSynchronize(stream));
    MPI_Allreduce(&h_reduceLocal[0], &error, 1, mpiTypeOf<cudaSolverType>(),
                  MPI_SUM, fieldcomm);
    error = sqrt(error);
    gmresCheckFinite(error, "GMRES", restart);

    if (error / initial_error < tol) {
      if (gmresRank == 0)
        printf(
            "GMRES converged at restart # %d; iteration #%d with error: %g\n",
            restart, kEnd, error / initial_error);
      return status;
    }
    gpuScale(d_gmresV, 1.0 / error, n, stream);
  }
  if (gmresRank == 0)
    std::cout << "  [GMRES] WARNING: did not converge after " << max_iter
              << " restarts, error: " << error / initial_error << std::endl;
  status.converged = false;
  return status;
}

// =========================================================================
//  GPU FGMRES workspace: allocate Z only on first actual FGMRES use
// =========================================================================

void EMfields3D::gpuEnsureFGMRESWorkspace(int m, int n) {
  // the longest Krylov vectors: the Maxwell ones (3 x interior nodes)
  const int nGMRESKrylov = maxwellKrylovSize_;

  // m vectors of length n in the space of GMRES_M vectors of the longest
  // length (the cleaning solves use CLEANING_M shorter vectors)
  const size_t required = static_cast<size_t>(GMRES_M) * nGMRESKrylov;
  if (m > KRYLOV_M_MAX || n > nGMRESKrylov ||
      static_cast<size_t>(m) * n > required) {
    eprintf("Requested FGMRES workspace exceeds persistent sizing: requested "
            "m=%d n=%d, at most m=%d and m*n=%zu",
            m, n, KRYLOV_M_MAX, required);
    abort();
  }

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
//  Differs from gpuGMRES_impl by the Z basis (Z[k] = M⁻¹ V[k]); the
//  solution update uses Z.
// =========================================================================

// Returns ||b|| and whether the true residual fell below tol * ||r0|| within
// max_iter restarts (or r0 = 0).  zeroGuess: start from x = 0, without the
// image of x.
static EMfields3D::KrylovSolveStatus
gpuFGMRES_impl(EMfields3D* field,
               void (EMfields3D::*GpuImage)(cudaSolverType*, cudaSolverType*),
               void (EMfields3D::*GpuPrecond)(cudaSolverType*, cudaSolverType*),
               cudaSolverType* d_x, int n, cudaSolverType* d_b, bool zeroGuess,
               int m, int max_iter, cudaSolverType tol,
               cudaSolverType* d_scratch, cudaSolverType* const d_gmresV,
               cudaSolverType* const d_gmresW, const size_t gmresVAlloc,
               cudaSolverType* const d_fgmresZ, const size_t fgmresZAlloc,
               MPI_Comm fieldcomm, cudaStream_t stream,
               cudaSolverType* h_reduceLocal, cudaSolverType* h_reduceGlobal,
               cudaSolverType* H, cudaSolverType* g, cudaSolverType* cs,
               cudaSolverType* sn, cudaSolverType* y, const char* label) {
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
  if (zeroGuess) {
    cudaErrChk(
        cudaMemsetAsync(d_x, 0, (size_t)n * sizeof(cudaSolverType), stream));
    gpuEq(d_gmresV, d_b, n, stream);
  } else {
    (field->*GpuImage)(d_gmresW, d_x);
    gpuEq(d_gmresV, d_b, n, stream);
    gpuSub(d_gmresV, d_gmresW, n, stream);
  }

  // ||r||₂
  gpuNorm2_async(d_gmresV, n, d_scratch, &h_reduceLocal[0], stream);
  cudaErrChk(cudaStreamSynchronize(stream));
  double initial_error;
  MPI_Allreduce(&h_reduceLocal[0], &initial_error, 1,
                mpiTypeOf<cudaSolverType>(), MPI_SUM, fieldcomm);
  initial_error = sqrt(initial_error);
  gmresCheckFinite(initial_error, label, 0);

  // ||b||: printed, and returned for the field-solver guard (r0 = b with
  // zeroGuess)
  EMfields3D::KrylovSolveStatus status;
  status.normB = initial_error;
  if (!zeroGuess) {
    gpuNorm2_async(d_b, n, d_scratch, &h_reduceLocal[0], stream);
    cudaErrChk(cudaStreamSynchronize(stream));
    double normb;
    MPI_Allreduce(&h_reduceLocal[0], &normb, 1, mpiTypeOf<cudaSolverType>(),
                  MPI_SUM, fieldcomm);
    status.normB = sqrt(normb);
  }
  if (initial_error < 1e-30)
    return status;

  int rank;
  MPI_Comm_rank(fieldcomm, &rank);
  if (rank == 0)
    printf("  [%s] Initial residual: %g  norm(b) = %g\n", label, initial_error,
           status.normB);

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

      // w -= sum_j h_jk V[j] in one pass
      for (int j = 0; j <= k; j++)
        H[j * m + k] = h_reduceGlobal[j];
      gpuSubCombination(d_gmresW, d_gmresV, (size_t)n, k + 1, (size_t)n,
                        h_reduceGlobal, stream);

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
    gmresCheckFinite(error, label, restart);

    if (error <= rho_tol) {
      if (rank == 0)
        printf("  [%s] Converged at restart #%d, iteration #%d, error: %g\n",
               label, restart, kk, error / initial_error);
      return status;
    }
    gpuScale(d_gmresV, 1.0 / error, n, stream);
  }

  if (rank == 0)
    printf("  [%s] WARNING: did not converge after %d restarts, error: %g\n",
           label, max_iter, error / initial_error);
  status.converged = false;
  return status;
}

// =========================================================================
//  GPU Block-Jacobi preconditioner (communication-free)
//
//  Approximately solves  A·x ≈ b  using a 3×3 block-diagonal D and the
//  communication-free local operator A_local.
//
//  • sweeps == 1  →  x = ω D⁻¹ b  (damped Jacobi)
//  • sweeps > 1   →  x = D⁻¹ b, then sweeps − 1 Richardson (weighted Jacobi)
//                    steps x += ω D⁻¹ (b − A_local x)
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
    // Single sweep: scale by ω (damped Jacobi)
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
  //  The convergence factor per sweep
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

EMfields3D::KrylovSolveStatus EMfields3D::gpuFGMRES_BlockJacobiPrecond(
    cudaSolverType* d_x, int n, cudaSolverType* d_b, int m, int max_iter,
    cudaSolverType tol, MPI_Comm fieldcomm) {
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

  return gpuFGMRES_impl(
      this, gpuMaxwellImageFn(), &EMfields3D::gpuBlockJacobiPrecond, d_x, n,
      d_b, false, m, max_iter, tol, d_blasScratch, d_gmresV, d_gmresW,
      gmresVAlloc, d_fgmresZ, fgmresZAlloc, fieldcomm, solverStream_,
      h_gmresReduceLocal, h_gmresReduceGlobal, h_gmresH, h_gmresG, h_gmresCS,
      h_gmresSN, h_gmresY, "FGMRES+BlockJacobi");
}

// =========================================================================
//  Solver steps: the instantiation of the run's halo mode (HaloOverlap)
// =========================================================================

void EMfields3D::gpuCalculateE(int cycle) {
  if (haloOverlap_)
    gpuCalculateE<HaloMode::Overlap>(cycle);
  else
    gpuCalculateE<HaloMode::Blocking>(cycle);
}

void EMfields3D::gpuCalculateB(int cycle) {
  // A cleaning cycle changes the whole cell B before the interpolation to
  // the nodes, which the overlapped variant starts early: it runs blocking
  // (both variants give the same values)
  if (haloOverlap_ && !isDivBCleaningCycle(cycle))
    gpuCalculateB<HaloMode::Overlap>(cycle);
  else
    gpuCalculateB<HaloMode::Blocking>(cycle);
  // E^n is kept through the mover and the output of this cycle (checkpoints,
  // field files and diagnostics see E^n with B^n and the particles at n);
  // after Faraday's law Exth is free, so theta = 1 is a buffer swap
  gpuAdvanceE();
}

void EMfields3D::gpuCalculateHatFunctions(int cycle) {
  if (haloOverlap_)
    gpuCalculateHatFunctions<HaloMode::Overlap>(cycle);
  else
    gpuCalculateHatFunctions<HaloMode::Blocking>(cycle);
}

// =========================================================================
//  gpuCalculateE: full E-field solver
// =========================================================================

template <EMfields3D::HaloMode M> void EMfields3D::gpuCalculateE(int cycle) {
  const Collective* col = &get_col();
  const VirtualTopology3D* vct = &get_vct();

  if (vct->getCartesian_rank() == 0)
    cout << "*** E CALCULATION [GPU] ***" << endl;

  const int nMaxwell = maxwellKrylovSize_;

#if defined(IPIC3D_GPU_CYCLE_DIAGNOSTICS)
  auto reportElectricField = [&](const char* stage, bool includeGhosts) {
    if (!gpuCycleDiagnosticsStateCycle(cycle))
      return;
    const Grid& grid = get_grid();
    const GPUCycleDiagnosticRegion region = {
        nxn,
        nyn,
        nzn,
        {includeGhosts ? 0 : 1, includeGhosts ? 0 : 1, includeGhosts ? 0 : 1},
        {includeGhosts ? nxn : nxn - 1, includeGhosts ? nyn : nyn - 1,
         includeGhosts ? nzn : nzn - 1},
        {grid.getXN(0), grid.getYN(0), grid.getZN(0)},
        {grid.getDX(), grid.getDY(), grid.getDZ()}};
    // E(n+theta) as the mover samples it (smoothed when smoothing is on)
    static const char* const names[3] = {"ExForPcls", "EyForPcls", "EzForPcls"};
    const cudaSolverType* arrays[3] = {gpuExForPcls().devPtr(),
                                       gpuEyForPcls().devPtr(),
                                       gpuEzForPcls().devPtr()};
    const double scales[3] = {1.0, 1.0, 1.0};
    gpuCycleDiagnostics_.reportArraySet(stage, cycle, -1, names, arrays, scales,
                                        3, region, solverStream_,
                                        vct->getFieldComm());
  };
#endif

  // Divergence cleaning on E (Poisson correction)
  gpuPoissonCorrection(cycle);

  if (vct->getCartesian_rank() == 0)
    cout << "*** MAXWELL SOLVER [GPU] ***" << endl;

  // Build RHS
  gpuMaxwellSource(d_bkrylovMaxwell.devPtr(), cycle);

  // Initial guess: pack current E into x
  gpuPhys2Solver3(d_xkrylovMaxwell.devPtr(), d_Ex.devPtr(), d_Ey.devPtr(),
                  d_Ez.devPtr(), nxn, nyn, nzn, solverStream_);

  MPI_Comm fieldcomm = vct->getFieldComm();

  // The guard reads the status of the Krylov solvers after the solve, so it
  // adds no work to it; the Chebyshev iteration reports nothing (unchecked).
  KrylovSolveStatus status;
  const char* solve = "Maxwell GMRES";
  if (SolverType == "Chebyshev") {
    // Chebyshev semi-iterative solver
    double eigMin = (chebEigMin > 0.0) ? chebEigMin : 1.0;
    double eigMax = chebEigMax;
    if (eigMax <= 0.0) {
      eigMax = gpuEstimateMaxEigenvalue(gpuMaxwellImageFn(), nMaxwell, 20,
                                        fieldcomm);
    }
    gpuChebyshevSolve(d_xkrylovMaxwell.devPtr(), nMaxwell,
                      d_bkrylovMaxwell.devPtr(), gpuMaxwellImageFn(),
                      chebMaxIter, eigMin, eigMax, fieldcomm);
  } else if (SolverType == "FGMRESBlockJacobi") {
    // FGMRES(20) with communication-free block-Jacobi preconditioner
    if (vct->getCartesian_rank() == 0)
      cout << "*** MAXWELL SOLVER [GPU FGMRES+BlockJacobi] ***" << endl;
    status = gpuFGMRES_BlockJacobiPrecond(d_xkrylovMaxwell.devPtr(), nMaxwell,
                                          d_bkrylovMaxwell.devPtr(), 20, 200,
                                          GMREStol, fieldcomm);
    solve = "Maxwell FGMRES+BlockJacobi";
  } else {
    // Default: GMRES(20) solver
    status =
        gpuGMRES_impl(this, gpuMaxwellImageFn(), d_xkrylovMaxwell.devPtr(),
                      nMaxwell, d_bkrylovMaxwell.devPtr(), 20, 200, GMREStol,
                      d_blasScratch, d_gmresV, d_gmresW, gmresVAlloc, fieldcomm,
                      solverStream_, h_gmresReduceLocal, h_gmresReduceGlobal,
                      h_gmresH, h_gmresG, h_gmresCS, h_gmresSN, h_gmresY);
  }
  // Before the solution is unpacked: a ||b|| jump explains a failed solve,
  // so it is tested first.
  checkMaxwellSource(status.normB, cycle);
  checkSolverConverged(status.converged, solve, cycle);

  // Krylov → physical: Exth, Eyth, Ezth
  gpuSolver2Phys3(d_Exth.devPtr(), d_Eyth.devPtr(), d_Ezth.devPtr(),
                  d_xkrylovMaxwell.devPtr(), nxn, nyn, nzn, solverStream_);

#if defined(IPIC3D_GPU_CYCLE_DIAGNOSTICS)
  if (gpuCycleDiagnosticsStateCycle(cycle)) {
    const Grid& grid = get_grid();
    const GPUCycleDiagnosticRegion activeNodes = {
        nxn,
        nyn,
        nzn,
        {1, 1, 1},
        {nxn - 1, nyn - 1, nzn - 1},
        {grid.getXN(0), grid.getYN(0), grid.getZN(0)},
        {grid.getDX(), grid.getDY(), grid.getDZ()}};
    static const char* const names[3] = {"Exth", "Eyth", "Ezth"};
    const cudaSolverType* arrays[3] = {d_Exth.devPtr(), d_Eyth.devPtr(),
                                       d_Ezth.devPtr()};
    const double scales[3] = {1.0, 1.0, 1.0};
    gpuCycleDiagnostics_.reportArraySet("Eth_after_solver_unpack", cycle, -1,
                                        names, arrays, scales, 3, activeNodes,
                                        solverStream_, vct->getFieldComm());
  }
#endif

  // The mover samples E(n+theta), the push the implicit moments assume:
  // smoothed into ExthSmooth when smoothing is on, Exth itself otherwise.
  // Faraday's law uses the unsmoothed Exth.  E (d_Ex) stays E^n until
  // gpuAdvanceE.
  GPUFieldArray3* const solved[3] = {&d_Exth, &d_Eyth, &d_Ezth};
  if (smoothingOn()) {
    GPUFieldArray3* const smoothed[3] = {&d_ExthSmooth, &d_EythSmooth,
                                         &d_EzthSmooth};
    gpuSmoothE<M>(solved, smoothed);

#if defined(IPIC3D_GPU_CYCLE_DIAGNOSTICS)
    reportElectricField("Eth_after_smoothing", false);
#endif

    // Exth and ExthSmooth in one MPI round
    GPUFieldArray3* const both[6] = {solved[0],   solved[1],   solved[2],
                                     smoothed[0], smoothed[1], smoothed[2]};
    const int* const bc[6] = {col->bcEx, col->bcEy, col->bcEz,
                              col->bcEx, col->bcEy, col->bcEz};
    gpuCommunicateNodeBC_6mixed(nxn, nyn, nzn, both, bc);
    gpuOpenBoundaryInflowE(d_Exth.devPtr(), d_Eyth.devPtr(), d_Ezth.devPtr(),
                           nxn, nyn, nzn);
    gpuOpenBoundaryInflowE(d_ExthSmooth.devPtr(), d_EythSmooth.devPtr(),
                           d_EzthSmooth.devPtr(), nxn, nyn, nzn);
  } else {
    gpuCommunicateNodeBC_3mixed(nxn, nyn, nzn, d_Exth, col->bcEx, d_Eyth,
                                col->bcEy, d_Ezth, col->bcEz);
    gpuOpenBoundaryInflowE(d_Exth.devPtr(), d_Eyth.devPtr(), d_Ezth.devPtr(),
                           nxn, nyn, nzn);
  }

#if defined(IPIC3D_GPU_CYCLE_DIAGNOSTICS)
  reportElectricField("Eth_after_inflow_boundary", true);
#endif
}

// =========================================================================
//  gpuAdvanceE: E^n -> E(n+1)
// =========================================================================

void EMfields3D::gpuAdvanceE() {
  GPUFieldArray3& sx = gpuExForPcls();
  GPUFieldArray3& sy = gpuEyForPcls();
  GPUFieldArray3& sz = gpuEzForPcls();
  if (th == 1.0) {
    // E(n+1) = S(E(n+theta)): swap the storage; E^n's becomes the work array
    // the next solve (or smoothing) overwrites before any read
    d_Ex.swap(sx);
    d_Ey.swap(sy);
    d_Ez.swap(sz);
  } else {
    // E(n+1) = (S(E(n+theta)) - (1-theta) E^n) / theta on every node: the
    // source carries its halo and open-boundary values, and the ghost and
    // layer values of E^n are those of the previous step
    gpuAddscale2_3(1.0 / th, -(1.0 - th) / th, d_Ex.devPtr(), sx.devPtr(),
                   d_Ey.devPtr(), sy.devPtr(), d_Ez.devPtr(), sz.devPtr(),
                   (size_t)nxn * nyn * nzn, solverStream_);
  }
}

// =========================================================================
//  gpuCalculateB: Faraday update
// =========================================================================

template <EMfields3D::HaloMode M> void EMfields3D::gpuCalculateB(int cycle) {
  const Collective* col = &get_col();
  const VirtualTopology3D* vct = &get_vct();
  const Grid* grid = &get_grid();
  const bool gemCase = isGemCase(), forceFreeCase = isForceFreeCase();
  double _invdx = grid->get_invdx();
  double _invdy = grid->get_invdy();
  double _invdz = grid->get_invdz();

  if (vct->getCartesian_rank() == 0)
    cout << "*** B CALCULATION [GPU] ***" << endl;

  size_t centSize = (size_t)nxc * nyc * nzc;

#if defined(IPIC3D_GPU_CYCLE_DIAGNOSTICS)
  auto reportMagneticField = [&](const char* stage) {
    if (!gpuCycleDiagnosticsStateCycle(cycle))
      return;
    const GPUCycleDiagnosticRegion activeCenters = {
        nxc,
        nyc,
        nzc,
        {1, 1, 1},
        {nxc - 1, nyc - 1, nzc - 1},
        {grid->getXC(0), grid->getYC(0), grid->getZC(0)},
        {grid->getDX(), grid->getDY(), grid->getDZ()}};
    static const char* const names[3] = {"Bx", "By", "Bz"};
    const cudaSolverType* arrays[3] = {d_Bxc.devPtr(), d_Byc.devPtr(),
                                       d_Bzc.devPtr()};
    const double scales[3] = {1.0, 1.0, 1.0};
    gpuCycleDiagnostics_.reportArraySet(stage, cycle, -1, names, arrays, scales,
                                        3, activeCenters, solverStream_,
                                        vct->getFieldComm());
  };
  reportMagneticField("B_before_faraday_update");
#endif

  // curl(Eth) → tempXC/YC/ZC
  gpuCurlN2C(d_tempXC.devPtr(), d_tempYC.devPtr(), d_tempZC.devPtr(),
             d_Exth.devPtr(), d_Eyth.devPtr(), d_Ezth.devPtr(), nxc, nyc, nzc,
             _invdx, _invdy, _invdz, solverStream_);

  // B^{n+1} = B^n - c*dt * curl(Eth)
  gpuAddscale3(-c * dt, d_Bxc.devPtr(), d_tempXC.devPtr(), d_Byc.devPtr(),
               d_tempYC.devPtr(), d_Bzc.devPtr(), d_tempZC.devPtr(), centSize,
               solverStream_);

#if defined(IPIC3D_GPU_CYCLE_DIAGNOSTICS)
  reportMagneticField("B_after_faraday_update");
#endif

  // Communicate center B ghost cells (batched: 3 fields in 1 MPI round)
  if constexpr (M == HaloMode::Overlap) {
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
      safeIHi = std::min(safeIHi, nxn - layersXright() - 3);
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
    if (gemCase) {
      if (lowYPhysical) {
        bxSafeJLo = std::max(bxSafeJLo, 4);
        bzSafeJLo = std::max(bzSafeJLo, 4);
      }
      if (highYPhysical) {
        bxSafeJHi = std::min(bxSafeJHi, nyn - 5);
        bzSafeJHi = std::min(bzSafeJHi, nyn - 5);
      }
    } else if (forceFreeCase) {
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
    gpuFixBcCase();

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
  } else {
    gpuCommunicateCenterBC_3mixed(nxc, nyc, nzc, d_Bxc, col->bcBx, d_Byc,
                                  col->bcBy, d_Bzc, col->bcBz);

    // Open boundary conditions on center-based B
    gpuOpenBoundaryInflowB(d_Bxc.devPtr(), d_Byc.devPtr(), d_Bzc.devPtr(), nxc,
                           nyc, nzc);

    // Case-specific fixes on center-based B
    gpuFixBcCase();

    // div(B) cleaning of the cell B, before its interpolation to the nodes
    // (gpuCalculateB(int) runs every cleaning cycle Blocking)
    if (isDivBCleaningCycle(cycle))
      gpuApplyDivBCleaning(cycle);

    // Interpolate center → node
    gpuInterpC2N(d_Bxn.devPtr(), d_Bxc.devPtr(), nxn, nyn, nzn, solverStream_);
    gpuInterpC2N(d_Byn.devPtr(), d_Byc.devPtr(), nxn, nyn, nzn, solverStream_);
    gpuInterpC2N(d_Bzn.devPtr(), d_Bzc.devPtr(), nxn, nyn, nzn, solverStream_);
  }

  // Communicate node B ghost cells (batched: 3 fields in 1 MPI round)
  gpuCommunicateNodeBC_3mixed(nxn, nyn, nzn, d_Bxn, col->bcBx, d_Byn, col->bcBy,
                              d_Bzn, col->bcBz);

  // Case-specific fixes on node-based B
  if (gemCase)
    gpuFixBnGEM();
}

// =========================================================================
//  gpuCalculateHatFunctions: compute Jhat and rhohat
// =========================================================================

template <EMfields3D::HaloMode M>
void EMfields3D::gpuCalculateHatFunctions(int cycle) {
  const Grid* grid = &get_grid();
  double _invdx = grid->get_invdx();
  double _invdy = grid->get_invdy();
  double _invdz = grid->get_invdz();

  size_t nodeSize = (size_t)nxn * nyn * nzn;
  size_t centSize = (size_t)nxc * nyc * nzc;

  // Smooth rhoc
  gpuSmooth<M>(d_rhoc, 0);

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

    if constexpr (M == HaloMode::Overlap) {
      // --- overlap: begin halo exchange for 3 centre fields ---
      cudaSolverType* hatPtrs[3] = {d_tempXC.devPtr(), d_tempYC.devPtr(),
                                    d_tempZC.devPtr()};
      gpuBatchedHaloBeginExchange(hatPtrs, 3, nxc, nyc, nzc,
                                  /*offsetZero=*/true, /*faceOnly=*/false,
                                  /*isParticle=*/true, solverStream_);
      // --- interior interpC2N while the exchange is in flight ---
      gpuInterpC2N_interior(d_tempXN.devPtr(), d_tempXC.devPtr(), nxn, nyn, nzn,
                            solverStream_);
      gpuInterpC2N_interior(d_tempYN.devPtr(), d_tempYC.devPtr(), nxn, nyn, nzn,
                            solverStream_);
      gpuInterpC2N_interior(d_tempZN.devPtr(), d_tempZC.devPtr(), nxn, nyn, nzn,
                            solverStream_);
      // --- end exchange: MPI faces, edges and corners + unpack ---
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
    } else {
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
    }

    // Add species current: tempN += Jxs[is] (fused triple)
    gpuSum3(d_tempXN.devPtr(), d_Jxs.speciesPtr(is), d_tempYN.devPtr(),
            d_Jys.speciesPtr(is), d_tempZN.devPtr(), d_Jzs.speciesPtr(is),
            nodeSize, solverStream_);

    // PIdot: Jhat += π(tempN)
    gpuPIdot(d_Jxh, d_Jyh, d_Jzh, d_tempXN, d_tempYN, d_tempZN, is);
  }

#if defined(IPIC3D_GPU_CYCLE_DIAGNOSTICS)
  auto reportJhat = [&](const char* stage) {
    if (!gpuCycleDiagnosticsStateCycle(cycle))
      return;
    static const char* const jhatNames[3] = {"Jhat_x", "Jhat_y", "Jhat_z"};
    const cudaSolverType* jhatArrays[3] = {d_Jxh.devPtr(), d_Jyh.devPtr(),
                                           d_Jzh.devPtr()};
    const double scales[3] = {1.0, 1.0, 1.0};
    const GPUCycleDiagnosticRegion activeNodes = {
        nxn,
        nyn,
        nzn,
        {1, 1, 1},
        {nxn - 1, nyn - 1, nzn - 1},
        {grid->getXN(0), grid->getYN(0), grid->getZN(0)},
        {grid->getDX(), grid->getDY(), grid->getDZ()}};
    gpuCycleDiagnostics_.reportArraySet(stage, cycle, -1, jhatNames, jhatArrays,
                                        scales, 3, activeNodes, solverStream_,
                                        get_vct().getParticleComm());
  };
  reportJhat("Jhat_after_forming");
#endif

  // Smooth Jhat (batched: 3 fields in 1 MPI round per iteration)
  gpuSmooth3<M>(d_Jxh, d_Jyh, d_Jzh, 1);

#if defined(IPIC3D_GPU_CYCLE_DIAGNOSTICS)
  reportJhat("Jhat_after_smoothing");
#endif

  // rhohat = rhoc - dt*theta*div(Jhat)
  gpuDivN2C(d_tempXC.devPtr(), d_Jxh.devPtr(), d_Jyh.devPtr(), d_Jzh.devPtr(),
            nxc, nyc, nzc, _invdx, _invdy, _invdz, solverStream_);
  gpuScale(d_tempXC.devPtr(), -dt * th, centSize, solverStream_);
  gpuSum(d_tempXC.devPtr(), d_rhoc.devPtr(), centSize, solverStream_);
  gpuEq(d_rhoh.devPtr(), d_tempXC.devPtr(), centSize, solverStream_);

  // X-right outflow face: no predicted charge beyond the reference node
  gpuZeroOutflowRhohat();

#if defined(IPIC3D_GPU_CYCLE_DIAGNOSTICS)
  if (gpuCycleDiagnosticsStateCycle(cycle)) {
    static const char* const rhohatName[1] = {"rhohat"};
    const cudaSolverType* rhohatArray[1] = {d_rhoh.devPtr()};
    const double scale[1] = {1.0};
    const GPUCycleDiagnosticRegion activeCenters = {
        nxc,
        nyc,
        nzc,
        {1, 1, 1},
        {nxc - 1, nyc - 1, nzc - 1},
        {grid->getXC(0), grid->getYC(0), grid->getZC(0)},
        {grid->getDX(), grid->getDY(), grid->getDZ()}};
    gpuCycleDiagnostics_.reportArraySet(
        "rhohat_after_forming", cycle, -1, rhohatName, rhohatArray, scale, 1,
        activeCenters, solverStream_, get_vct().getParticleComm());
  }
#endif

  // Communicate rhoh
  gpuCommunicateCenterBC_P(nxc, nyc, nzc, d_rhoh, 2, 2, 2, 2, 2, 2);
}

// =========================================================================
//  GPU moment-processing: D2D scatter from packed moment buffer
// =========================================================================

#if defined(IPIC3D_GPU_CYCLE_DIAGNOSTICS)
void EMfields3D::gpuDiagnosticReportSpeciesMoments(const char* stage, int cycle,
                                                   int species,
                                                   cudaStream_t stream) {
  static const char* const momentNames[10] = {
      "rho", "Jx", "Jy", "Jz", "Pxx", "Pxy", "Pxz", "Pyy", "Pyz", "Pzz"};
  const cudaSolverType* momentArrays[10] = {
      d_rhons.speciesPtr(species), d_Jxs.speciesPtr(species),
      d_Jys.speciesPtr(species),   d_Jzs.speciesPtr(species),
      d_pXXsn.speciesPtr(species), d_pXYsn.speciesPtr(species),
      d_pXZsn.speciesPtr(species), d_pYYsn.speciesPtr(species),
      d_pYZsn.speciesPtr(species), d_pZZsn.speciesPtr(species)};
  const double scales[10] = {1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0};
  const Grid& grid = get_grid();
  const GPUCycleDiagnosticRegion nodeRegion = {
      nxn,
      nyn,
      nzn,
      {0, 0, 0},
      {nxn, nyn, nzn},
      {grid.getXN(0), grid.getYN(0), grid.getZN(0)},
      {grid.getDX(), grid.getDY(), grid.getDZ()}};
  gpuCycleDiagnostics_.reportArraySet(stage, cycle, species, momentNames,
                                      momentArrays, scales, 10, nodeRegion,
                                      stream, get_vct().getParticleComm());
}
#endif

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
//  gpuCommunicateGhostP2G_AllSpecies – ALL species batched
//  Processes species in chunks of HALO_MAX_BATCH / 10 . Each chunk runs the
//  full 3-phase sequence (additive halo → boundary adjust → copy halo)
//  independently.
// =========================================================================

void EMfields3D::gpuCommunicateGhostP2G_AllSpecies(int cycle) {
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

#if defined(IPIC3D_GPU_CYCLE_DIAGNOSTICS)
    if (gpuCycleDiagnosticsStateCycle(cycle)) {
      // This is deliberately before boundary-density adjustment and the
      // copy-style halo: it isolates the additive ghost-to-shared operation.
      for (int is = isStart; is < isEnd; ++is)
        gpuDiagnosticReportSpeciesMoments("moments_after_additive_halo", cycle,
                                          is, solverStream_);
    }
#endif

    // Phase 2: adjust the moments on the non-periodic boundary faces
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
//  gpuSetZeroDerivedMoments
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
//  gpuNeutralizeOutflowLayer: neutral rhons on the X-right outflow layer
// =========================================================================

void EMfields3D::gpuNeutralizeOutflowLayer() {
  if (!ownsOutflowFaceXright())
    return;
  const int L = layersXright();
  gpuNeutralizeLayersX(d_rhons.devPtr(), d_qom, ns, nxn - 1 - L, nxn - 1,
                       nxn - 2 - L, nxn, nyn, nzn, solverStream_);
}

// =========================================================================
//  gpuCopyOutflowLayerMoments: J_s, P_s on the X-right outflow layer
// =========================================================================

void EMfields3D::gpuCopyOutflowLayerMoments() {
  if (!ownsOutflowFaceXright())
    return;
  const int L = layersXright();
  for (int is = 0; is < ns; is++) {
    cudaSolverType* m[9] = {
        d_Jxs.speciesPtr(is),   d_Jys.speciesPtr(is),   d_Jzs.speciesPtr(is),
        d_pXXsn.speciesPtr(is), d_pXYsn.speciesPtr(is), d_pXZsn.speciesPtr(is),
        d_pYYsn.speciesPtr(is), d_pYZsn.speciesPtr(is), d_pZZsn.speciesPtr(is)};
    for (int c = 0; c < 9; c += 3)
      gpuExtrapolateLayers3(m[c], m[c + 1], m[c + 2], 0, nxn - 1 - L, nxn - 1,
                            nxn - 2 - L, nxn, nyn, nzn, solverStream_);
  }
}

// =========================================================================
//  gpuZeroOutflowRhohat: no predicted charge beyond x_s
// =========================================================================

void EMfields3D::gpuZeroOutflowRhohat() {
  if (!ownsOutflowFaceXright())
    return;
  // The cells from the one straddling x_s (node nxn-2-L) to the ghost cell,
  // nxc-1-L .. nxc-1, are the last L+1 x planes of the [i][j][k] cell array:
  // one contiguous block.
  const int first = nxc - 1 - layersXright();
  const size_t plane = (size_t)nyc * nzc;
  cudaErrChk(cudaMemsetAsync(d_rhoh.devPtr() + first * plane, 0,
                             (nxc - first) * plane * sizeof(cudaSolverType),
                             solverStream_));
}

void EMfields3D::gpuOutflowCopyE(cudaSolverType* dX, cudaSolverType* dY,
                                 cudaSolverType* dZ) {
  if (!ownsOutflowFaceXright())
    return;
  const int L = fieldCopyLayersXright();
  gpuExtrapolateLayers3(dX, dY, dZ, 0, nxn - 1 - L, nxn - 1, nxn - 2 - L, nxn,
                        nyn, nzn, solverStream_);
}

// =========================================================================
//  Ghost node on the X-right outflow face (Maxwell operator closure, see
//  OutflowBoundary.h)
// =========================================================================

void EMfields3D::gpuOutflowGhostNodes(GPUFieldArray3* const a[3]) {
  if (!ownsOutflowFaceXright())
    return;
  gpuExtrapolateLayers3(a[0]->devPtr(), a[1]->devPtr(), a[2]->devPtr(), 0,
                        nxn - 1, nxn - 1, nxn - 2, nxn, nyn, nzn,
                        solverStream_);
}

void EMfields3D::gpuOutflowGhostRows(cudaSolverType* const a[], int n) {
  if (!ownsOutflowFaceXright())
    return;
  if (n > CellArrayBatch::kMax) {
    cerr << "ERROR: gpuOutflowGhostRows takes at most " << CellArrayBatch::kMax
         << " arrays, got " << n << endl;
    MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    std::abort();
  }
  CellArrayBatch batch;
  batch.n = n;
  for (int i = 0; i < n; i++)
    batch.p[i] = a[i];
  int rule[4];
  outflowGhostRowRules(rule);
  gpuGhostRowsXright(batch, nxc, nyc, nzc, rule, solverStream_);
}

// =========================================================================
//  gpuSumOverSpecies: rhon = Σ_s rhons_s
// =========================================================================

void EMfields3D::gpuSumOverSpecies() {
  size_t nodeSize = (size_t)nxn * nyn * nzn;
  for (int is = 0; is < ns; is++)
    gpuSum(d_rhon.devPtr(), d_rhons.speciesPtr(is), nodeSize, solverStream_);
}

// =========================================================================
//  gpuSumOverSpeciesJ: J = sum_s(J_s)
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
//  gpuInterpDensitiesN2C: rhoc = interp(rhon)
// =========================================================================

void EMfields3D::gpuInterpDensitiesN2C() {
  gpuInterpN2C(d_rhoc.devPtr(), d_rhon.devPtr(), nxc, nyc, nzc, solverStream_);
}

// =========================================================================
//  gpuOpenBoundaryInflowESource: RHS rows (E0) of the open inflow faces
// =========================================================================

void EMfields3D::gpuOpenBoundaryInflowESource(cudaSolverType* dX,
                                              cudaSolverType* dY,
                                              cudaSolverType* dZ, int nx,
                                              int ny, int nz) {
  if (!get_col().getApplyInflowBcsEImage())
    return;
  double injE[3]; // E0 = -(u0 x B0) / c
  driftElectricField(injE);
  const int n[3] = {nx, ny, nz};
  for (int f = 0; f < 6; f++)
    if (openInflowFace_[f])
      gpuOpenBCSetFace3(dX, dY, dZ, injE[0], injE[1], injE[2], f / 2,
                        f % 2 ? n[f / 2] - 2 : 1, nx, ny, nz, solverStream_);
}

// =========================================================================
//  gpuOpenBoundaryInflowEImage: identity rows on the open inflow faces
// =========================================================================

void EMfields3D::gpuOpenBoundaryInflowEImage(cudaSolverType* d_im,
                                             const cudaSolverType* vX,
                                             const cudaSolverType* vY,
                                             const cudaSolverType* vZ, int nx,
                                             int ny, int nz) {
  const int n[3] = {nx, ny, nz};
  for (int f = 0; f < 6; f++)
    if (openInflowFace_[f])
      gpuOpenBCCopyFace3(d_im, vX, vY, vZ, f / 2, f % 2 ? n[f / 2] - 2 : 1, nx,
                         ny, nz, solverStream_);
}

// =========================================================================
//  gpuOpenBoundaryInflowE: SAL blend / Dirichlet on E inflow faces
// =========================================================================

void EMfields3D::gpuOpenBoundaryInflowE(cudaSolverType* dX, cudaSolverType* dY,
                                        cudaSolverType* dZ, int nx, int ny,
                                        int nz) {
  const VirtualTopology3D* vct = &get_vct();
  const Collective* col = &get_col();

  double injE[3]; // E0 = -(u0 x B0) / c
  driftElectricField(injE);

  double invNL = (n_layers_sal > 0) ? 1.0 / (double)n_layers_sal : 1.0;

  if (yes_sal) {
    // SAL mode: blend E toward injE over n_layers_sal layers
    if (vct->getXleft_neighbor() == MPI_PROC_NULL && bcEMfaceXleft == 2 &&
        col->getBcPfaceXleft() == 2)
      gpuSALBlendLayers3(dX, dY, dZ, injE[0], injE[1], injE[2], 0, 0,
                         n_layers_sal, invNL, 1, nx, ny, nz, solverStream_);
    // Xright outflow face: gpuOutflowCopyE (reference-plane copy, or only
    // the ghost node with OutflowLayerFields = solved)
    gpuOutflowCopyE(dX, dY, dZ);
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
    gpuOutflowCopyE(dX, dY, dZ);
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
//  gpuOpenBoundaryInflowB: SAL blend / extrapolation on center B
// =========================================================================

void EMfields3D::gpuOpenBoundaryInflowB(cudaSolverType* dX, cudaSolverType* dY,
                                        cudaSolverType* dZ, int nx, int ny,
                                        int nz, bool ghostOnly) {
  const VirtualTopology3D* vct = &get_vct();
  double invNL = (n_layers_sal > 0) ? 1.0 / (double)n_layers_sal : 1.0;
  // With sal_tangential_B the SAL leaves B normal to the face alone: blending
  // it towards B0 creates div(B) in the layer (see README)
  auto skip = [&](int dir) { return sal_tangential_B ? dir : -1; };
  // The treated layers: [0, leftEnd] on a left face, [rightStart(n, L), n - 1]
  // on a right face, or only the ghost layer.  The SAL weight of the ghost
  // layer is 0 in both ranges, so the ghost value does not depend on them.
  const int leftEnd = ghostOnly ? 0 : n_layers_sal;
  auto rightStart = [&](int n, int L) { return ghostOnly ? n - 1 : n - L - 1; };
  // The "> 10" gates, meant for the flat direction of a 2D run (which is
  // periodic and never reaches them), skip wall ranks with at most 8 cells
  // along the normal (Collective::Print warns)

  // Xleft
  if (vct->getXleft_neighbor() == MPI_PROC_NULL && bcEMfaceXleft == 2 &&
      nx > 10) {
    if (yes_sal)
      gpuSALBlendLayers3(dX, dY, dZ, B0x, B0y, B0z, 0, 0, leftEnd, invNL, 1, nx,
                         ny, nz, solverStream_, skip(0));
    else
      gpuSetConstLayers3(dX, dY, dZ, B0x, B0y, B0z, 0, 0, leftEnd, nx, ny, nz,
                         solverStream_);
  }
  // Xright: always extrapolation from interior
  if (vct->getXright_neighbor() == MPI_PROC_NULL && bcEMfaceXright == 2 &&
      nx > 10) {
    const int L = fieldCopyLayersXright();
    gpuExtrapolateLayers3(dX, dY, dZ, 0, rightStart(nx, L), nx - 1, nx - 2 - L,
                          nx, ny, nz, solverStream_);
  }
  // Yleft
  if (vct->getYleft_neighbor() == MPI_PROC_NULL && bcEMfaceYleft == 2 &&
      ny > 10) {
    if (yes_sal)
      gpuSALBlendLayers3(dX, dY, dZ, B0x, B0y, B0z, 1, 0, leftEnd, invNL, 1, nx,
                         ny, nz, solverStream_, skip(1));
    else
      gpuExtrapolateLayers3(dX, dY, dZ, 1, 0, leftEnd, n_layers_sal + 1, nx, ny,
                            nz, solverStream_);
  }
  // Yright
  if (vct->getYright_neighbor() == MPI_PROC_NULL && bcEMfaceYright == 2 &&
      ny > 10) {
    if (yes_sal)
      gpuSALBlendLayers3(dX, dY, dZ, B0x, B0y, B0z, 1,
                         rightStart(ny, n_layers_sal), ny - 1, invNL, 0, nx, ny,
                         nz, solverStream_, skip(1));
    else
      gpuExtrapolateLayers3(dX, dY, dZ, 1, rightStart(ny, n_layers_sal), ny - 1,
                            ny - 2 - n_layers_sal, nx, ny, nz, solverStream_);
  }
  // Zleft
  if (vct->getZleft_neighbor() == MPI_PROC_NULL && bcEMfaceZleft == 2 &&
      nz > 10) {
    if (yes_sal)
      gpuSALBlendLayers3(dX, dY, dZ, B0x, B0y, B0z, 2, 0, leftEnd, invNL, 1, nx,
                         ny, nz, solverStream_, skip(2));
    else
      gpuExtrapolateLayers3(dX, dY, dZ, 2, 0, leftEnd, n_layers_sal + 1, nx, ny,
                            nz, solverStream_);
  }
  // Zright
  if (vct->getZright_neighbor() == MPI_PROC_NULL && bcEMfaceZright == 2 &&
      nz > 10) {
    if (yes_sal)
      gpuSALBlendLayers3(dX, dY, dZ, B0x, B0y, B0z, 2,
                         rightStart(nz, n_layers_sal), nz - 1, invNL, 0, nx, ny,
                         nz, solverStream_, skip(2));
    else
      gpuExtrapolateLayers3(dX, dY, dZ, 2, rightStart(nz, n_layers_sal), nz - 1,
                            nz - 2 - n_layers_sal, nx, ny, nz, solverStream_);
  }
}

// =========================================================================
//  gpuFixBcGEM / gpuFixBnGEM / gpuFixBforcefree
// =========================================================================

void EMfields3D::gpuFixBcGEM(bool ghostOnly) {
  const VirtualTopology3D* vct = &get_vct();
  double LyH = Ly / 2.0;
  if (vct->getYright_neighbor() == MPI_PROC_NULL)
    gpuFixBcGEMKernel(d_Bxc.devPtr(), d_Byc.devPtr(), d_Bzc.devPtr(), B0x, B0y,
                      B0z, yStart, dy, LyH, delta, 1, nxc, nyc, nzc, ghostOnly,
                      solverStream_);
  if (vct->getYleft_neighbor() == MPI_PROC_NULL)
    gpuFixBcGEMKernel(d_Bxc.devPtr(), d_Byc.devPtr(), d_Bzc.devPtr(), B0x, B0y,
                      B0z, yStart, dy, LyH, delta, 0, nxc, nyc, nzc, ghostOnly,
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

void EMfields3D::gpuFixBforcefree(bool ghostOnly) {
  const VirtualTopology3D* vct = &get_vct();
  double LyH = Ly / 2.0;
  if (vct->getYright_neighbor() == MPI_PROC_NULL)
    gpuFixBforcefreeKernel(d_Bxc.devPtr(), d_Byc.devPtr(), d_Bzc.devPtr(), B0x,
                           B0y, B0z, yStart, dy, LyH, delta, 1, nxc, nyc, nzc,
                           ghostOnly, solverStream_);
  if (vct->getYleft_neighbor() == MPI_PROC_NULL)
    gpuFixBforcefreeKernel(d_Bxc.devPtr(), d_Byc.devPtr(), d_Bzc.devPtr(), B0x,
                           B0y, B0z, yStart, dy, LyH, delta, 0, nxc, nyc, nzc,
                           ghostOnly, solverStream_);
}

// =========================================================================
//  gpuConstantChargePlanet / gpuConstantChargePlanet2DPlaneXZ
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
  // x-z plane: one cell in y, the kernel writes the node planes j = 1, 2
  assert_eq(nyn, 4);
  for (int is = 0; is < ns; is++) {
    double sign_q = qom[is] / fabs(qom[is]);
    double val = sign_q * rhoINIT[is] / FourPI;
    gpuConstantChargePlanet2DKernel(d_rhons.speciesPtr(is), val, R, x_center,
                                    z_center, xStart, zStart, dx, dz, nxn, nyn,
                                    nzn, solverStream_);
  }
}

// =========================================================================
//  Ghost rule of the Poisson operator at physical (non-periodic) faces,
//  in gpuBCface order: Xright, Xleft, Yright, Yleft, Zright, Zleft.
//  0 = antisymmetric (ghost = -cell), 1 = zero ghost, 2 = copy (ghost = cell).
//  Zero ghosts on every face: the phi of the div(E) solve, and the 7-point
//  operators (gpuPoissonImage, gpuPoissonImageLocal; not used currently).
// =========================================================================

static constexpr int kPoissonBCface[6] = {1, 1, 1, 1, 1, 1};

// =========================================================================
//  gpuPoissonImage (not used currently)
// =========================================================================

void EMfields3D::gpuPoissonImage(cudaSolverType* d_im, cudaSolverType* d_vec) {
  const Grid* grid = &get_grid();
  double _invdx = grid->get_invdx();
  double _invdy = grid->get_invdy();
  double _invdz = grid->get_invdz();

  // Krylov → physical (interior; the 7-point stencil reads only the face
  // ghosts, which the exchange or the face BCs set)
  gpuSolver2Phys1(d_poissonTemp.devPtr(), d_vec, nxc, nyc, nzc, solverStream_);

  // Communicate ghost cells (center box stencil)
  gpuCommunicateCenterBoxStencilBC(nxc, nyc, nzc, d_poissonTemp,
                                   kPoissonBCface[0], kPoissonBCface[1],
                                   kPoissonBCface[2], kPoissonBCface[3],
                                   kPoissonBCface[4], kPoissonBCface[5]);

  // Laplacian, written straight into the Krylov image
  gpuLapC2CToKrylov(d_im, d_poissonTemp.devPtr(), nxc, nyc, nzc,
                    _invdx * _invdx, _invdy * _invdy, _invdz * _invdz,
                    solverStream_);
}

// =========================================================================
//  Ghost rules of the communication-free 7-point operator (of
//  gpuPoissonImageLocal, not used currently)
//
//  The local operator is the Laplacian of gpuPoissonImage restricted to this
//  rank.  Beyond each face the value is
//  factor × (adjacent interior cell), chosen to match the real operator
//  wherever that is known locally:
//    • physical face (no neighbour): the rule of gpuPoissonImage, from
//      kPoissonBCface (code 0 → factor -1, 1 → 0, 2 → +1);
//    • periodic wrap onto this rank with ONE cell along the direction: the
//      real ghost is the cell itself (factor +1).  Both faces copy, so the
//      direction contributes nothing ("flat", e.g. y of a 2D x-z run);
//    • neighbouring rank, or periodic wrap onto this rank with more than
//      one cell: factor 0, the block-Jacobi zero-Dirichlet cut.
// =========================================================================

struct PoissonLocalGhostRules {
  int nCell[3];  // interior cells per direction on this rank
  int factor[6]; // ghost = factor * cell: Xleft, Xright, Yleft, Yright, ...
};

static PoissonLocalGhostRules
poissonLocalGhostRules(const VirtualTopology3D* vct, int nxc, int nyc,
                       int nzc) {
  PoissonLocalGhostRules rules;
  rules.nCell[0] = nxc - 2;
  rules.nCell[1] = nyc - 2;
  rules.nCell[2] = nzc - 2;
  const int rank = vct->getCartesian_rank();
  const int neighbor[6] = {vct->getXleft_neighbor(), vct->getXright_neighbor(),
                           vct->getYleft_neighbor(), vct->getYright_neighbor(),
                           vct->getZleft_neighbor(), vct->getZright_neighbor()};
  for (int f = 0; f < 6; f++) {
    const int dir = f / 2;
    if (neighbor[f] == MPI_PROC_NULL) {
      // kPoissonBCface lists the right face of each direction first
      const int bc = kPoissonBCface[2 * dir + (f % 2 == 0 ? 1 : 0)];
      rules.factor[f] = (bc == 0) ? -1 : (bc == 2) ? 1 : 0;
    } else {
      // a single cell wrapping onto this rank copies (flat); else the cut
      rules.factor[f] = neighbor[f] == rank && rules.nCell[dir] == 1;
    }
  }
  return rules;
}

/** The communication-free 7-point -∇² on the packed interior cells (not
 *  used currently: the cleaning preconditioners use Lap27Op). */
[[maybe_unused]] static PoissonLocalOperator
poissonLocalOperator(const Grid* grid, const PoissonLocalGhostRules& r) {
  PoissonLocalOperator op;
  op.ni = r.nCell[0];
  op.nj = r.nCell[1];
  op.nk = r.nCell[2];
  op.invdx2 = grid->get_invdx() * grid->get_invdx();
  op.invdy2 = grid->get_invdy() * grid->get_invdy();
  op.invdz2 = grid->get_invdz() * grid->get_invdz();
  for (int f = 0; f < 6; f++)
    op.g[f] = r.factor[f];
  return op;
}

// =========================================================================
//  gpuPoissonImageLocal: -∇²·x (positive eigenvalues, NO communication)
//  The ghost layer is never read: gpuLapC2CLocalKernel applies the face
//  rules of poissonLocalGhostRules instead.  Not used currently.
// =========================================================================

void EMfields3D::gpuPoissonImageLocal(cudaSolverType* d_im,
                                      cudaSolverType* d_vec) {
  const Grid* grid = &get_grid();
  double _invdx = grid->get_invdx();
  double _invdy = grid->get_invdy();
  double _invdz = grid->get_invdz();

  const int nPoisson = divEKrylovSize_;
  const PoissonLocalGhostRules rules =
      poissonLocalGhostRules(&get_vct(), nxc, nyc, nzc);

  // Krylov → physical (interior only; no communication, ghosts unused)
  gpuSolver2Phys1(d_poissonTemp.devPtr(), d_vec, nxc, nyc, nzc, solverStream_);

  // Laplacian with the local face rules: d_poissonIm = ∇²·x (interior)
  gpuLapC2CLocalKernel(d_poissonIm.devPtr(), d_poissonTemp.devPtr(), nxc, nyc,
                       nzc, _invdx * _invdx, _invdy * _invdy, _invdz * _invdz,
                       rules.factor, solverStream_);

  // Physical → Krylov, then negate to get -∇²·x (positive eigenvalues)
  gpuPhys2Solver1(d_im, d_poissonIm.devPtr(), nxc, nyc, nzc, solverStream_);
  gpuScale(d_im, -1.0, nPoisson, solverStream_);
}

// =========================================================================
//  The cleaning solves (GPUCleaning.cuh): image, preconditioner and FGMRES
//  of the solve cleaning_ points to
// =========================================================================

template <EMfields3D::HaloMode M>
void EMfields3D::gpuCleaningImage(cudaSolverType* d_im, cudaSolverType* d_vec) {
  const CleaningSolve& s = *cleaning_;
  cudaSolverType* f = s.buf->devPtr();
  const int nx = s.cells ? nxc : nxn, ny = s.cells ? nyc : nyn,
            nz = s.cells ? nzc : nzn;
  // Unpack only the rows the image reads from f (it reads the others from
  // d_vec); the face planes of the div(B) rows take their rules here, before
  // the exchange carries them into the edge ghosts
  gpuSolver2PhysLap27(f, d_vec, ny, nz, s.op, solverStream_, true);
  cudaSolverType* ptr[1] = {f};
  if constexpr (M == HaloMode::Overlap) {
    gpuBatchedHaloBeginExchange(ptr, 1, nx, ny, nz, s.cells, false, false,
                                solverStream_);
    // the rows that read no ghost, while the exchange is in flight
    gpuLap27ToKrylov(d_im, f, d_vec, ny, nz, s.op, s.rows.interior,
                     solverStream_);
    gpuBatchedHaloEndExchange();
    gpuLap27ToKrylov(d_im, f, d_vec, ny, nz, s.op, s.rows.shells,
                     solverStream_);
  } else {
    gpuBatchedHaloExchange(ptr, 1, nx, ny, nz, s.cells, false, false, false,
                           solverStream_);
    gpuLap27ToKrylov(d_im, f, d_vec, ny, nz, s.op, s.rows.all, solverStream_);
  }
}

void EMfields3D::gpuCleaningPrecond(cudaSolverType* d_x, cudaSolverType* d_b) {
  const CleaningSolve& s = *cleaning_;
  gpuChebyshevCleaning(d_x, d_b, s.op, s.cheb.min, s.cheb.max,
                       poissonChebMaxIter, d_cleanCheb_,
                       d_cleanCheb_ + cleanChebLength_, solverStream_);
}

// The interval of the preconditioner (gpuChebyshevCleaning): the exact bounds
// of the local operator (lap27EigBounds), eigMin floored at eigMax /
// PoissonChebEigRatio, then the rescale factors.  The floor: on [eigMin,
// eigMax] the residual of the polynomial is about 1/cosh(2k sqrt(eigMin /
// eigMax)), so with the exact minimum of a large block it does almost
// nothing; with the floor it clusters the upper spectrum, and FGMRES takes the
// modes below it (README, "Solves of the ∇·E and ∇·B corrections").
EMfields3D::ChebBounds EMfields3D::cleaningChebBounds(const Lap27Op& op,
                                                      const char* label) {
  const Lap27Eig e = lap27EigBounds(op);
  double eigMin = e.min;
  if (poissonChebEigRatio > 0.0 && e.max / poissonChebEigRatio > eigMin)
    eigMin = e.max / poissonChebEigRatio;
  ChebBounds cb;
  cb.min = eigMin * poissonChebRescaleEigMin;
  cb.max = e.max * poissonChebRescaleEigMax;
  if (!(cb.min <= cb.max)) {
    cerr << "ERROR: the " << label << " Chebyshev interval [" << cb.min << ", "
         << cb.max << "] is inverted: lower PoissonChebRescaleEigMin "
         << "or raise PoissonChebRescaleEigMax" << endl;
    MPI_Abort(get_vct().getFieldComm(), EXIT_FAILURE);
    std::abort();
  }
  // the bounds differ between ranks (rows, physical faces): their range
  double range[4] = {eigMin, -eigMin, e.max, -e.max};
  MPI_Allreduce(MPI_IN_PLACE, range, 4, MPI_DOUBLE, MPI_MIN,
                get_vct().getFieldComm());
  if (get_vct().getCartesian_rank() == 0)
    printf("  [%s Chebyshev] exact eigenvalue bounds of the local operator "
           "(rows %d x %d x %d): min %.6g%s, max %.6g; eigMin floor "
           "eigMax/%g, rescaled [%.6g, %.6g]; over all ranks floored min in "
           "[%.6g, %.6g], max in [%.6g, %.6g]\n",
           label, op.ni, op.nj, op.nk, e.min,
           e.singular ? " (non-zero; singular)" : "", e.max,
           poissonChebEigRatio, cb.min, cb.max, range[0], -range[1], range[2],
           -range[3]);
  return cb;
}

bool EMfields3D::gpuCleaningSolve(CleaningSolve& s,
                                  const std::vector<RoundOffTerm>& scale,
                                  int cycle) {
  const int n = (int)s.b.size();
  cudaSolverType* b = s.b.devPtr();
  gpuRemoveCleaningNullSpace(b, s.null);
  const bool master = get_vct().getCartesian_rank() == 0;
  if (cleaningAtRoundOff(b, n, scale)) {
    if (master)
      cout << "*** DIVERGENCE CLEANING " << s.name
           << ": at round-off, skipped ***" << endl;
    return false;
  }
  if (master)
    cout << "*** DIVERGENCE CLEANING " << s.name
         << " using FGMRES+Chebyshev [GPU] ***" << endl;
  cleaning_ = &s;
  constexpr int maxRestarts = 4000 / CLEANING_M; // about 4000 iterations
  gpuEnsureFGMRESWorkspace(CLEANING_M, n);
  const KrylovSolveStatus status = gpuFGMRES_impl(
      this,
      s.rows.split ? &EMfields3D::gpuCleaningImage<HaloMode::Overlap>
                   : &EMfields3D::gpuCleaningImage<HaloMode::Blocking>,
      &EMfields3D::gpuCleaningPrecond, s.x.devPtr(), n, b, true, CLEANING_M,
      maxRestarts, divCleaningTol, d_blasScratch, d_gmresV, d_gmresW,
      gmresVAlloc, d_fgmresZ, fgmresZAlloc, get_vct().getFieldComm(),
      solverStream_, h_gmresReduceLocal, h_gmresReduceGlobal, h_gmresH,
      h_gmresG, h_gmresCS, h_gmresSN, h_gmresY, s.label);
  checkSolverConverged(status.converged,
                       (std::string(s.name) + " cleaning FGMRES").c_str(),
                       cycle);
  return true;
}

// The null space (CleaningNullSpace): per pass, the signed sums on the GPU,
// one copy to the host and one MPI_Allreduce, then the terms.  Two passes:
// the first leaves eps times the removed components, which can be most of
// what remains (a quiet start with a net charge and E = 0).
void EMfields3D::gpuRemoveCleaningNullSpace(cudaSolverType* d_b,
                                            const CleaningNullSpace& ns) {
  if (ns.nSums == 0)
    return;
  std::vector<cudaSolverType> global;
  for (int pass = 0; pass < 2; pass++) {
    gpuCleaningNullSums(d_b, ns, d_cleanLines_, global, solverStream_);
    MPI_Allreduce(MPI_IN_PLACE, global.data(), ns.globalSize,
                  mpiTypeOf<cudaSolverType>(), MPI_SUM,
                  get_vct().getFieldComm());
    gpuCleaningNullSubtract(d_b, ns, global, d_cleanLines_ + cleanLinesSlots(),
                            solverStream_);
  }
}

bool EMfields3D::cleaningAtRoundOff(const cudaSolverType* d_b, int n,
                                    const std::vector<RoundOffTerm>& fields) {
  // the squared norms in one batch: slot 0 of the scratch and pinned buffers
  // for b, slot 1 + f for field f
  if ((int)fields.size() > KRYLOV_M_MAX) {
    cerr << "ERROR: cleaningAtRoundOff takes at most " << KRYLOV_M_MAX
         << " fields" << endl;
    MPI_Abort(get_vct().getFieldComm(), EXIT_FAILURE);
    std::abort();
  }
  gpuNorm2_async(d_b, n, d_blasScratch, h_gmresReduceLocal, solverStream_);
  for (size_t f = 0; f < fields.size(); f++)
    gpuNorm2_async(fields[f].data, fields[f].n, d_blasScratch + 1 + f,
                   h_gmresReduceLocal + 1 + f, solverStream_);
  cudaErrChk(cudaStreamSynchronize(solverStream_));
  double norm2[2] = {h_gmresReduceLocal[0], 0.0};
  for (size_t f = 0; f < fields.size(); f++)
    norm2[1] += fields[f].weight * h_gmresReduceLocal[1 + f];
  MPI_Allreduce(MPI_IN_PLACE, norm2, 2, MPI_DOUBLE, MPI_SUM,
                get_vct().getFieldComm());
  const double floor = 1e3 * std::numeric_limits<cudaSolverType>::epsilon();
  return norm2[0] <= floor * floor * norm2[1];
}

// =========================================================================
//  gpuPoissonCorrection (div(E) cleaning)
// =========================================================================

void EMfields3D::gpuPoissonCorrection(int cycle) {
  if (!PoissonCorrection || cycle % PoissonCorrectionCycle != 0)
    return;

  const Grid* grid = &get_grid();
  const cudaSolverType invd[3] = {grid->get_invdx(), grid->get_invdy(),
                                  grid->get_invdz()};

  // b = divN2C(E) - 4π rho on the centres; a b at the round-off of div(E)
  // and of the species charge leaves nothing to clean
  gpuDivN2CRhoToKrylov(divE_.b.devPtr(), d_Ex.devPtr(), d_Ey.devPtr(),
                       d_Ez.devPtr(), d_rhoc.devPtr(), -FourPI, nyn, nzn, invd,
                       divE_.op, solverStream_);
  const size_t nodes = (size_t)nxn * nyn * nzn;
  const double invh2 =
      invd[0] * invd[0] + invd[1] * invd[1] + invd[2] * invd[2];
  std::vector<RoundOffTerm> scale;
  for (const GPUFieldArray3* e : {&d_Ex, &d_Ey, &d_Ez})
    scale.push_back({e->devPtr(), nodes, invh2});
  for (int is = 0; is < ns; is++)
    scale.push_back({d_rhons.speciesPtr(is), nodes, FourPI * FourPI});
  // phi = 0 in the ghost cells of the physical faces: the unpack writes the
  // interior and no exchange writes these ghosts, so they stay 0 in the solve
  gpuBCface(nxc, nyc, nzc, d_PHI, kPoissonBCface[0], kPoissonBCface[1],
            kPoissonBCface[2], kPoissonBCface[3], kPoissonBCface[4],
            kPoissonBCface[5], &_vct, solverStream_);
  if (!gpuCleaningSolve(divE_, scale, cycle)) {
    d_PHI.setAll(0.0, solverStream_); // no correction (phi is written out)
    return;
  }

  // phi on the centres, then E -= gradC2N phi on the nodes
  gpuSolver2PhysLap27(d_PHI.devPtr(), divE_.x.devPtr(), nyc, nzc, divE_.op,
                      solverStream_);
  gpuCommunicateCenterBC(nxc, nyc, nzc, d_PHI, kPoissonBCface[0],
                         kPoissonBCface[1], kPoissonBCface[2],
                         kPoissonBCface[3], kPoissonBCface[4],
                         kPoissonBCface[5]);
  gpuSubGradC2N(d_Ex.devPtr(), d_Ey.devPtr(), d_Ez.devPtr(), d_PHI.devPtr(),
                nxn, nyn, nzn, invd, solverStream_);
}

// =========================================================================
//  gpuApplyDivBCleaning (div(B) cleaning)
//
//  Faraday's law keeps divC2N(Bc) at round-off (divC2N∘curlN2C = 0), so the
//  cleaning removes what the boundary treatments of Bc create.  It projects
//  the cell B, the field Faraday's law advances: the node B is its
//  interpolation (gpuCalculateB interpolates after this), and a field without
//  divergence is left alone.  The GEM and ForceFree wall layers are the
//  exception: the case fixes reset them (gpuFixBnGEM the GEM node planes after
//  the interpolation, the next gpuMaxwellSource the cell layers of both).
// =========================================================================

void EMfields3D::gpuApplyDivBCleaning(int cycle) {
  const Collective* col = &get_col();
  const Grid* grid = &get_grid();
  const cudaSolverType invd[3] = {grid->get_invdx(), grid->get_invdy(),
                                  grid->get_invdz()};

  // b = divC2N(Bc) on the nodes (0 on the face rows); at round-off, about
  // eps |Bc| / h, there is nothing to clean
  gpuDivC2NToKrylov(divB_.b.devPtr(), d_Bxc.devPtr(), d_Byc.devPtr(),
                    d_Bzc.devPtr(), nyc, nzc, invd, divB_.op, solverStream_);
  const size_t cells = (size_t)nxc * nyc * nzc;
  const double invh2 =
      invd[0] * invd[0] + invd[1] * invd[1] + invd[2] * invd[2];
  if (!gpuCleaningSolve(divB_,
                        {{d_Bxc.devPtr(), cells, invh2},
                         {d_Byc.devPtr(), cells, invh2},
                         {d_Bzc.devPtr(), cells, invh2}},
                        cycle))
    return;

  // psi on the nodes with the face rules (Bc -= gradN2C psi reads no ghost
  // node), then the ghost cells as after Faraday's law: exchange and face
  // rules, the ghost layers of the open faces and of the case fixes (only
  // the ghosts, so that the cleaned layers reach the interpolation)
  gpuSolver2PhysLap27(d_PSI.devPtr(), divB_.x.devPtr(), nyn, nzn, divB_.op,
                      solverStream_);
  gpuSubGradN2C(d_Bxc.devPtr(), d_Byc.devPtr(), d_Bzc.devPtr(), d_PSI.devPtr(),
                nxc, nyc, nzc, invd, solverStream_);
  gpuCommunicateCenterBC_3mixed(nxc, nyc, nzc, d_Bxc, col->bcBx, d_Byc,
                                col->bcBy, d_Bzc, col->bcBz);
  gpuOpenBoundaryInflowB(d_Bxc.devPtr(), d_Byc.devPtr(), d_Bzc.devPtr(), nxc,
                         nyc, nzc, true);
  gpuFixBcCase(true);
}

bool EMfields3D::isGemCase() const {
  const string& simCase = get_col().getCase();
  return simCase == "GEM" || simCase == "GEMnoPert" ||
         simCase == "GEMDoubleHarris";
}

bool EMfields3D::isForceFreeCase() const {
  return get_col().getCase() == "ForceFree";
}

void EMfields3D::gpuFixBcCase(bool ghostOnly) {
  if (isGemCase())
    gpuFixBcGEM(ghostOnly);
  else if (isForceFreeCase())
    gpuFixBforcefree(ghostOnly);
}
