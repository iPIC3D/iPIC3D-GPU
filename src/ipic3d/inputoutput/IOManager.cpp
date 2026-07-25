/**
 * @file IOManager.cpp
 * @brief Modular I/O manager implementation for iPIC3D.
 */

#include "IOManager.h"
#include "Alloc.h" // newArr3, newArr4, delArr3, delArr4
#include "Collective.h"
#include "EMfields3D.h"
#include "Grid3DCU.h"
#include "MPIdata.h"
#include "OutputTagConfig.h"
#include "ParallelIO.h" // free VTK/H5hut/PHDF5 write functions
#include "Parameters.h"
#include "ParticleSoAHost.h"
#include "RestartMeshMetadata.h"
#include "RestartParticleCellMetadata.h"
#include "VCtopology3D.h"
#include "debug.h" // eprintf, warning_printf

#include <iostream>
#include <string>

#ifndef NO_HDF5
#include "OutputWrapperFPP.h"
#endif

#ifdef USE_ADIOS2
#include "ADIOS2IO.hpp"
#endif

using std::string;

// ======= Construction and destruction =======

IOManager::IOManager() = default;

IOManager::~IOManager() {
  // Defensive diagnostic: if finalize() was not called (e.g. early abort),
  // NBCVTK split-collective writes may still be in flight. We cannot safely
  // issue MPI calls from a destructor (MPI may already be finalized, and
  // these are collective operations), so just warn the user that the last
  // output cycle may be truncated.
  if (fieldreqcounter_ > 0 || momentreqcounter_ > 0) {
    dprintf("IOManager: ~IOManager() found %d field and %d moment NBCVTK "
            "writes still pending; finalize() was not called. "
            "Last output cycle may be truncated.\n",
            fieldreqcounter_, momentreqcounter_);
  }
#ifndef NO_HDF5
  delete outputWrapperFPP_;
#endif
#ifdef USE_ADIOS2
  delete adiosManager_;
#endif
  // Free VTK write buffers (delArr needs the first two dimensions)
  if (fieldwritebuffer_ && fieldBufDim0_ > 0) {
    delArr4(fieldwritebuffer_, fieldBufDim0_, localWriteNy_, localWriteNx_);
  }
  if (momentwritebuffer_ && momentBufDim0_ > 0) {
    delArr3(momentwritebuffer_, momentBufDim0_, localWriteNy_);
  }
}

// ======= Initialization =======

void IOManager::init(Collective* col, VCtopology3D* vct, Grid3DCU* grid,
                     EMfields3D* EMf, ParticleSoAHost** outputPart, int ns,
                     ParticleSoAHost** testpart, int nstestpart,
                     int first_cycle) {
  // Store non-owning pointers
  col_ = col;
  vct_ = vct;
  grid_ = grid;
  EMf_ = EMf;
  outputPart_ = outputPart;
  testpart_ = testpart;
  ns_ = ns;
  nstestpart_ = nstestpart;
  first_cycle_ = first_cycle;
  restart_cycle_ = col->getRestartOutputCycle();

  const string writeMethod = col->getWriteMethod();

  // Step 1: determine the field backend from WriteMethod.
  if (writeMethod == "shdf5")
    fieldBackend_ = FieldBackend::SHDF5;
  else if (writeMethod == "pvtk")
    fieldBackend_ = FieldBackend::PVTK;
  else if (writeMethod == "nbcvtk")
    fieldBackend_ = FieldBackend::NBCVTK;
  else if (writeMethod == "phdf5")
    fieldBackend_ = FieldBackend::PARALLEL_HDF5;
  else if (writeMethod == "H5hut")
    fieldBackend_ = FieldBackend::H5HUT;
  else if (writeMethod == "adios2")
    fieldBackend_ = FieldBackend::ADIOS2;
  else {
    warning_printf("Unknown WriteMethod '%s'. "
                   "Valid options: shdf5, pvtk, nbcvtk, phdf5, H5hut, adios2",
                   writeMethod.c_str());
    fieldBackend_ = FieldBackend::NONE;
  }

  // Step 2: determine the particle backend.
#ifdef USE_ADIOS2
  particleBackend_ = ParticleBackend::ADIOS2;
#else
  if (writeMethod == "H5hut")
    particleBackend_ = ParticleBackend::H5HUT;
  else
    particleBackend_ = ParticleBackend::SHDF5;
#endif

  // Step 3: determine the restart backend.
#ifdef USE_ADIOS2
  restartBackend_ = RestartBackend::ADIOS2;
#else
  restartBackend_ = RestartBackend::SHDF5;
#endif

  // Step 4: validate backend availability against compile flags.
#ifdef NO_HDF5
  if (fieldBackend_ == FieldBackend::SHDF5 ||
      fieldBackend_ == FieldBackend::PARALLEL_HDF5 ||
      fieldBackend_ == FieldBackend::H5HUT) {
    eprintf("WriteMethod '%s' requires HDF5 (compile with USE_HDF5=ON)",
            writeMethod.c_str());
  }
  if (particleBackend_ == ParticleBackend::SHDF5 ||
      particleBackend_ == ParticleBackend::H5HUT) {
    warning_printf(
        "Selected particle backend requires HDF5 (compile with USE_HDF5=ON). "
        "Particle output disabled.");
    particleBackend_ = ParticleBackend::NONE;
  }
  if (restartBackend_ == RestartBackend::SHDF5) {
    warning_printf(
        "Selected restart backend requires HDF5 (compile with USE_HDF5=ON). "
        "Restart output disabled.");
    restartBackend_ = RestartBackend::NONE;
  }
#endif

  if (restartBackend_ != RestartBackend::NONE &&
      (restart_cycle_ > 0 || col->getCallFinalize())) {
    const RestartMeshMetadata restartMesh =
        makeCurrentRestartMeshMetadata(col, vct, grid, ns);
    restartSlots_.init(
        col->getRestartDirName(), RestartSlotManager::backendName(),
        vct->getCartesian_rank(), MPIdata::get_nprocs(), restartMesh);
  }
#ifndef USE_ADIOS2
  if (fieldBackend_ == FieldBackend::ADIOS2) {
    eprintf(
        "WriteMethod 'adios2' requires ADIOS2 (compile with USE_ADIOS2=ON)");
  }
  if (particleBackend_ == ParticleBackend::ADIOS2) {
    eprintf("Selected particle backend requires ADIOS2 (compile with "
            "USE_ADIOS2=ON)");
  }
  if (restartBackend_ == RestartBackend::ADIOS2) {
    eprintf("Selected restart backend requires ADIOS2 (compile with "
            "USE_ADIOS2=ON)");
  }
#endif

  // ======= HDF5 backend setup =======
#ifndef NO_HDF5
  {
    bool needHDF5 = false;
    if (fieldBackend_ == FieldBackend::SHDF5)
      needHDF5 = true;
    if (particleBackend_ == ParticleBackend::SHDF5)
      needHDF5 = true;
    if (restartBackend_ == RestartBackend::SHDF5)
      needHDF5 = true;
    // PHDF5 and H5hut use free functions; no OutputWrapperFPP is needed.

    if (needHDF5) {
      outputWrapperFPP_ = new OutputWrapperFPP;
      outputWrapperFPP_->init_output_files(col, vct, grid, EMf, outputPart, ns,
                                           testpart, nstestpart);
    }
  }
#endif

  // ======= ADIOS2 backend setup =======
#ifdef USE_ADIOS2
  if (particleBackend_ == ParticleBackend::ADIOS2 ||
      restartBackend_ == RestartBackend::ADIOS2 ||
      fieldBackend_ == FieldBackend::ADIOS2) {
    using namespace std::string_literals;
    string particleTag =
        col->getParticlesOutputCycle() ? col->getPclOutputTag() : ""s;

    adiosManager_ = new ADIOS2IO::ADIOS2Manager();
    adiosManager_->initOutputFiles(""s, particleTag, 0, col, vct, grid, EMf,
                                   outputPart, ns, testpart, nstestpart);
  }
#endif

  // ======= VTK write-buffer setup =======
  // Compute local write sizes: interior nodes + boundary node for upper
  // processes
  localWriteNx_ = grid->getNXN() - 3 + (vct->isXupper() ? 1 : 0);
  localWriteNy_ = grid->getNYN() - 3 + (vct->isYupper() ? 1 : 0);
  localWriteNz_ = grid->getNZN() - 3 + (vct->isZupper() ? 1 : 0);

  if (!col->field_output_is_off()) {
    const OutputTagConfig& cfg = col->getOutputConfig();

    if (fieldBackend_ == FieldBackend::PVTK) {
      // Vector buffer needed for B, E, and J vector writes
      if (cfg.needsAnyField()) {
        fieldBufDim0_ = localWriteNz_;
        fieldwritebuffer_ =
            newArr4(float, fieldBufDim0_, localWriteNy_, localWriteNx_, 3);
      }
      // Scalar buffer needed for rho, pressure tensor
      if (cfg.needsAnyMoments()) {
        momentBufDim0_ = localWriteNz_;
        momentwritebuffer_ =
            newArr3(float, momentBufDim0_, localWriteNy_, localWriteNx_);
      }
    } else if (fieldBackend_ == FieldBackend::NBCVTK) {
      fieldreqcounter_ = 0;
      momentreqcounter_ = 0;
      int nFieldWrites = cfg.countFieldWrites();
      int nMomentWrites = cfg.countMomentWrites();
      if (nFieldWrites > 0) {
        fieldBufDim0_ = localWriteNz_ * nFieldWrites;
        fieldwritebuffer_ =
            newArr4(float, fieldBufDim0_, localWriteNy_, localWriteNx_, 3);
        fieldreqArr_.resize(nFieldWrites);
        fieldfhArr_.resize(nFieldWrites);
        fieldstsArr_.resize(nFieldWrites);
      }
      if (nMomentWrites > 0) {
        momentBufDim0_ = localWriteNz_ * nMomentWrites;
        momentwritebuffer_ =
            newArr3(float, momentBufDim0_, localWriteNy_, localWriteNx_);
        momentreqArr_.resize(nMomentWrites);
        momentfhArr_.resize(nMomentWrites);
        momentstsArr_.resize(nMomentWrites);
      }
    }
  }
}

// ======= Field output =======

void IOManager::writeFields(int cycle) {

  const OutputTagConfig& cfg = col_->getOutputConfig();

  switch (fieldBackend_) {

  // Serial HDF5 (one file per process).
  case FieldBackend::SHDF5:
#ifndef NO_HDF5
    outputWrapperFPP_->append_field_moment_output(cfg, cycle);
#endif
    break;

  // Blocking collective MPI-IO VTK.
  case FieldBackend::PVTK:
    if (cfg.needsAnyField())
      WriteFieldsVTK(grid_, EMf_, col_, vct_, col_->getFieldOutputTag(), cycle,
                     fieldwritebuffer_);
    if (cfg.needsAnyMoments())
      WriteMomentsVTK(grid_, EMf_, col_, vct_, col_->getMomentsOutputTag(),
                      cycle, momentwritebuffer_);
    if ((!cfg.JSpecies.empty() || cfg.writeJTot) && fieldwritebuffer_)
      WriteMomentsJVTK(grid_, EMf_, col_, vct_, cycle, fieldwritebuffer_);
    break;

  // Non-blocking collective MPI-IO VTK.
  case FieldBackend::NBCVTK:
    // Complete any pending writes from the previous cycle
    if (!fieldreqArr_.empty()) {
      if (fieldreqcounter_ > 0) {
        for (int si = 0; si < fieldreqcounter_; si++) {
          int ec = MPI_File_write_all_end(fieldfhArr_[si],
                                          &fieldwritebuffer_[si][0][0][0],
                                          &fieldstsArr_[si]);
          if (ec != MPI_SUCCESS) {
            char es[100];
            int len, cls;
            MPI_Error_class(ec, &cls);
            MPI_Error_string(cls, es, &len);
            dprintf("MPI_Waitall error at field output cycle %d  %d  %s\n",
                    cycle, si, es);
          } else {
            MPI_File_close(&fieldfhArr_[si]);
          }
        }
      }
      fieldreqcounter_ = WriteFieldsVTKNonblk(
          grid_, EMf_, col_, vct_, cycle, fieldwritebuffer_,
          fieldreqArr_.data(), fieldfhArr_.data());
    }

    if (!momentreqArr_.empty()) {
      if (momentreqcounter_ > 0) {
        for (int si = 0; si < momentreqcounter_; si++) {
          int ec = MPI_File_write_all_end(momentfhArr_[si],
                                          &momentwritebuffer_[si][0][0],
                                          &momentstsArr_[si]);
          if (ec != MPI_SUCCESS) {
            char es[100];
            int len, cls;
            MPI_Error_class(ec, &cls);
            MPI_Error_string(cls, es, &len);
            dprintf("MPI_Waitall error at moments output cycle %d  %d %s\n",
                    cycle, si, es);
          } else {
            MPI_File_close(&momentfhArr_[si]);
          }
        }
      }
      momentreqcounter_ = WriteMomentsVTKNonblk(
          grid_, EMf_, col_, vct_, cycle, momentwritebuffer_,
          momentreqArr_.data(), momentfhArr_.data());
    }
    break;

  // Parallel HDF5.
  case FieldBackend::PARALLEL_HDF5:
#ifndef NO_HDF5
    WriteOutputParallel(grid_, EMf_, col_, vct_, cycle, cfg);
#endif
    break;

  // H5hut.
  case FieldBackend::H5HUT:
#ifndef NO_HDF5
    WriteFieldsH5hut(ns_, grid_, EMf_, col_, vct_, cycle, cfg);
#endif
    break;

  // ADIOS2 field output.
  case FieldBackend::ADIOS2:
#ifdef USE_ADIOS2
    adiosManager_->appendFieldOutput(cycle);
#endif
    break;

  case FieldBackend::NONE:
    break;
  }
}

// ======= Particle output =======

void IOManager::writeParticles(int cycle) {
  switch (particleBackend_) {

  case ParticleBackend::ADIOS2:
#ifdef USE_ADIOS2
    adiosManager_->appendParticleOutput(cycle);
#endif
    break;

  case ParticleBackend::SHDF5:
#ifndef NO_HDF5
    if (outputWrapperFPP_)
      outputWrapperFPP_->append_output(col_->getPclOutputTag().c_str(), cycle,
                                       0);
#endif
    break;

  case ParticleBackend::H5HUT:
#ifndef NO_HDF5
    WritePartclH5hut(ns_, grid_, outputPart_, col_, vct_, cycle);
#endif
    break;

  case ParticleBackend::NONE:
    break;
  }
}

// ======= Test-particle output =======

void IOManager::writeTestParticles(int cycle) {
  if (nstestpart_ == 0)
    return;

  // SoA data is authoritative; no conversion is needed because all writers use
  // SoA accessors.

#ifndef NO_HDF5
  if (outputWrapperFPP_)
    outputWrapperFPP_->append_output("testpartpos + testpartvel+ testparttag",
                                     cycle, 0);
#endif
}

// ======= Restart output =======

void IOManager::writeRestart(int cycle) {
  if (restartBackend_ == RestartBackend::NONE)
    return;

  RestartWriteTarget target = restartSlots_.beginWrite(cycle);

  switch (restartBackend_) {

  case RestartBackend::ADIOS2:
#ifdef USE_ADIOS2
    adiosManager_->writeRestartOutput(cycle, target.dataDir);
#endif
    break;

  case RestartBackend::SHDF5:
#ifndef NO_HDF5
    if (outputWrapperFPP_)
      outputWrapperFPP_->append_restart(cycle, target.dataDir);
#endif
    break;

  case RestartBackend::NONE:
    break;
  }

  // Do not publish a slot until every rank has closed its new rank file.
#ifndef NO_MPI
  MPI_Barrier(MPIdata::get_PicGlobalComm());
#endif

  restartSlots_.publishIfRoot(target);
  restartSlots_.completeLocal(target);
}

// ======= Finalization =======

void IOManager::finalize() {
  // Drain any in-flight NBCVTK split-collective writes from the last output
  // cycle. Must run before MPI_Finalize and is collective on the file's
  // communicator; c_Solver::Finalize() guarantees both.
  drainNBCVTKPending();
#ifdef USE_ADIOS2
  if (adiosManager_)
    adiosManager_->closeOutputFiles();
#endif
  // OutputWrapperFPP has no explicit close; the destructor handles cleanup.
}

void IOManager::drainNBCVTKPending() {
  if (fieldBackend_ != FieldBackend::NBCVTK)
    return;

  if (fieldreqcounter_ > 0 && !fieldreqArr_.empty()) {
    for (int si = 0; si < fieldreqcounter_; ++si) {
      int ec = MPI_File_write_all_end(
          fieldfhArr_[si], &fieldwritebuffer_[si][0][0][0], &fieldstsArr_[si]);
      if (ec != MPI_SUCCESS) {
        char es[100];
        int len, cls;
        MPI_Error_class(ec, &cls);
        MPI_Error_string(cls, es, &len);
        dprintf(
            "MPI_File_write_all_end error during NBCVTK field drain  %d  %s\n",
            si, es);
      } else {
        MPI_File_close(&fieldfhArr_[si]);
      }
    }
    fieldreqcounter_ = 0;
  }

  if (momentreqcounter_ > 0 && !momentreqArr_.empty()) {
    for (int si = 0; si < momentreqcounter_; ++si) {
      int ec = MPI_File_write_all_end(
          momentfhArr_[si], &momentwritebuffer_[si][0][0], &momentstsArr_[si]);
      if (ec != MPI_SUCCESS) {
        char es[100];
        int len, cls;
        MPI_Error_class(ec, &cls);
        MPI_Error_string(cls, es, &len);
        dprintf(
            "MPI_File_write_all_end error during NBCVTK moment drain  %d  %s\n",
            si, es);
      } else {
        MPI_File_close(&momentfhArr_[si]);
      }
    }
    momentreqcounter_ = 0;
  }
}

// ======= Scheduling query =======

bool IOManager::needsParticleSync(int cycle) const {
  if (restart_cycle_ > 0 && cycle % restart_cycle_ == 0)
    return true;
  if (!col_->particle_output_is_off() &&
      cycle % col_->getParticlesOutputCycle() == 0)
    return true;
  // Also sync particles when diagnostics (ConservedQuantities) are due
  if (col_->getDiagnosticsOutputCycle() > 0 &&
      cycle % col_->getDiagnosticsOutputCycle() == 0)
    return true;
  return false;
}

bool IOManager::needsRestartParticleSync(int cycle) const {
  return restartBackend_ != RestartBackend::NONE && restart_cycle_ > 0 &&
         cycle % restart_cycle_ == 0;
}

void IOManager::setRestartParticleCellMetadata(
    const RestartParticleCellMetadata* metadata) {
#ifndef NO_HDF5
  if (outputWrapperFPP_)
    outputWrapperFPP_->setRestartParticleCellMetadata(metadata);
#endif
#ifdef USE_ADIOS2
  if (adiosManager_)
    adiosManager_->setRestartParticleCellMetadata(metadata);
#endif
}
