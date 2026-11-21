/**
 * @file IOManager.cpp
 * @brief Modular I/O manager implementation for iPIC3D.
 */

#include "IOManager.h"
#include "Alloc.h" // newArr3, newArr4, delArr3, delArr4
#include "BenchmarkMode.h"
#include "Collective.h"
#include "EMfields3D.h"
#include "Grid3DCU.h"
#include "MPIdata.h"
#include "OutputTagConfig.h"
#include "ParallelIO.h" // free VTK/PHDF5 write functions
#include "Parameters.h"
#include "ParticleSoAHost.h"
#include "RestartMeshMetadata.h"
#include "RestartParticleCellMetadata.h"
#include "VCtopology3D.h"
#include "debug.h" // eprintf, warning_printf

#include <iostream>
#include <string>
#include <vector>

#ifndef NO_HDF5
#include "OutputWrapperFPP.h"
#endif

#ifdef USE_ADIOS2
#include "ADIOS2IO.hpp"
#endif

using std::string;

namespace {

/** The particle and checkpoint backends of this build, by preference, by
 *  their name in the input file (ParticlesWriteMethod, RestartWriteMethod). */
std::vector<string> builtFileBackends() {
  std::vector<string> built;
#ifdef USE_ADIOS2
  built.push_back("adios2");
#endif
#ifndef NO_HDF5
  built.push_back("shdf5");
#endif
  return built;
}

/** The backend of the input key, which asked for requested ("": the first
 *  of the build): requested if the build has it, else the first of the
 *  build, else none (""). With warn, a backend replaced or none is warned
 *  of, the data (what) written with another one or not at all; the run goes
 *  on. */
string chooseFileBackend(const char* key, const string& requested, bool warn,
                         const char* what) {
  const std::vector<string> built = builtFileBackends();
  if (built.empty()) {
    if (warn)
      warning_printf("this build has neither ADIOS2 nor HDF5 (compile with "
                     "USE_ADIOS2=ON or USE_HDF5=ON): no %s are written",
                     what);
    return "";
  }
  for (const string& name : built)
    if (name == requested)
      return name;
  if (warn && !requested.empty())
    warning_printf("%s = %s is not in this build (compile with %s): the %s "
                   "are written with %s",
                   key, requested.c_str(),
                   requested == "adios2" ? "USE_ADIOS2=ON" : "USE_HDF5=ON",
                   what, built.front().c_str());
  return built.front();
}

/** The particle or restart backend of a name of builtFileBackends(). */
template <typename Backend> Backend fileBackend(const string& name) {
  return name == "adios2"  ? Backend::ADIOS2
         : name == "shdf5" ? Backend::SHDF5
                           : Backend::NONE;
}

} // namespace

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
  if constexpr (!BenchmarkConfig::DISK_OUTPUT_ENABLED)
    return;

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

  const string writeMethod = col->getFieldWriteMethod();

  // Step 1: the field backend, FieldWriteMethod.
  if (writeMethod == "shdf5")
    fieldBackend_ = FieldBackend::SHDF5;
  else if (writeMethod == "pvtk")
    fieldBackend_ = FieldBackend::PVTK;
  else if (writeMethod == "nbcvtk")
    fieldBackend_ = FieldBackend::NBCVTK;
  else if (writeMethod == "phdf5")
    fieldBackend_ = FieldBackend::PARALLEL_HDF5;
  else if (writeMethod == "adios2")
    fieldBackend_ = FieldBackend::ADIOS2;
  else {
    // Fatal: carrying on would silently drop all field output.
    eprintf("Unknown FieldWriteMethod '%s' in the input file. "
            "Valid options: shdf5, pvtk, nbcvtk, phdf5, adios2",
            writeMethod.c_str());
  }

  // Steps 2 and 3: the particle and checkpoint backends,
  // ParticlesWriteMethod and RestartWriteMethod (chooseFileBackend); rank 0
  // warns when it writes such data.
  const bool rank0 = vct->getCartesian_rank() == 0;
  particleBackend_ = fileBackend<ParticleBackend>(chooseFileBackend(
      "ParticlesWriteMethod", col->getParticlesWriteMethod(),
      rank0 && col->particle_output_requested(), "particles"));
  restartBackend_ = fileBackend<RestartBackend>(
      chooseFileBackend("RestartWriteMethod", col->getRestartWriteMethod(),
                        rank0 && restart_cycle_ > 0, "checkpoints"));

  // Step 4: validate backend availability against compile flags.
#ifdef NO_HDF5
  if (fieldBackend_ == FieldBackend::SHDF5 ||
      fieldBackend_ == FieldBackend::PARALLEL_HDF5) {
    eprintf("FieldWriteMethod '%s' requires HDF5 (compile with USE_HDF5=ON)",
            writeMethod.c_str());
  }
#endif

  if (restartBackend_ != RestartBackend::NONE &&
      (restart_cycle_ > 0 || col->getCallFinalize())) {
    const RestartMeshMetadata restartMesh =
        makeCurrentRestartMeshMetadata(col, vct, grid, ns);
    // Continue from the slot the restart read ("" for a fresh run): the next
    // checkpoint goes to the other slot. The backend by its manifest name.
    restartSlots_.init(col->getRestartDirName(),
                       restartBackend_ == RestartBackend::ADIOS2 ? "adios2"
                                                                 : "hdf5",
                       vct->getCartesian_rank(), MPIdata::get_nprocs(),
                       restartMesh, col->getRestartReadSlot());
  }
#ifndef USE_ADIOS2
  if (fieldBackend_ == FieldBackend::ADIOS2) {
    eprintf("FieldWriteMethod 'adios2' requires ADIOS2 (compile with "
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
    // PHDF5 uses free functions; no OutputWrapperFPP is needed.

    if (needHDF5) {
      outputWrapperFPP_ = new OutputWrapperFPP;
      outputWrapperFPP_->init_output_files(
          col, vct, grid, EMf, outputPart, ns, testpart, nstestpart,
          fieldBackend_ == FieldBackend::SHDF5 ||
              (particleBackend_ == ParticleBackend::SHDF5 &&
               col->particle_output_requested()),
          restartBackend_ == RestartBackend::SHDF5);
    }
  }
#endif

  // ======= ADIOS2 backend setup =======
#ifdef USE_ADIOS2
  if (particleBackend_ == ParticleBackend::ADIOS2 ||
      restartBackend_ == RestartBackend::ADIOS2 ||
      fieldBackend_ == FieldBackend::ADIOS2) {
    using namespace std::string_literals;
    const string particleTag = particleBackend_ == ParticleBackend::ADIOS2 &&
                                       col->particle_output_requested()
                                   ? col->getPclOutputTag()
                                   : ""s;

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
  if constexpr (!BenchmarkConfig::DISK_OUTPUT_ENABLED)
    return;

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
  if constexpr (!BenchmarkConfig::DISK_OUTPUT_ENABLED)
    return;

  switch (particleBackend_) {

  case ParticleBackend::ADIOS2:
#ifdef USE_ADIOS2
    adiosManager_->appendParticleOutput(cycle);
#endif
    break;

  case ParticleBackend::SHDF5:
#ifndef NO_HDF5
    if (outputWrapperFPP_)
      outputWrapperFPP_->append_particles(col_->getPclOutputTag(), cycle,
                                          col_->getParticlesOutputSpecies());
#endif
    break;

  case ParticleBackend::NONE:
    break;
  }
}

// ======= Test-particle output =======

void IOManager::writeTestParticles(int cycle) {
  if constexpr (!BenchmarkConfig::DISK_OUTPUT_ENABLED)
    return;

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
  if constexpr (!BenchmarkConfig::DISK_OUTPUT_ENABLED)
    return;

  if (restartBackend_ == RestartBackend::NONE)
    return;

  // Rank 0 removes the slot's manifest; only then may the ranks replace
  // their files, so an interrupted write leaves a slot marked incomplete.
  RestartWriteTarget target = restartSlots_.beginWrite(cycle);
#ifndef NO_MPI
  MPI_Barrier(MPIdata::get_PicGlobalComm());
#endif
  restartSlots_.prepareRankFile(target);

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
  if constexpr (!BenchmarkConfig::DISK_OUTPUT_ENABLED)
    return;

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

bool IOManager::needsFieldOutput(int cycle) const {
  if constexpr (!BenchmarkConfig::DISK_OUTPUT_ENABLED)
    return false;

  if (fieldBackend_ == FieldBackend::NONE || !Parameters::get_doWriteOutput() ||
      col_->field_output_is_off())
    return false;

  const OutputTagConfig& cfg = col_->getOutputConfig();
  if (!cfg.needsAnyField() && !cfg.needsAnyMoments())
    return false;

  return cycle % col_->getFieldOutputCycle() == 0 || cycle == first_cycle_;
}

bool IOManager::needsParticleOutput(int cycle) const {
  if constexpr (!BenchmarkConfig::DISK_OUTPUT_ENABLED)
    return false;

  return particleBackend_ != ParticleBackend::NONE &&
         col_->particle_output_requested() && Parameters::get_doWriteOutput() &&
         cycle % col_->getParticlesOutputCycle() == 0;
}

bool IOManager::needsParticleSync(int cycle) const {
  if constexpr (!BenchmarkConfig::DISK_OUTPUT_ENABLED)
    return false;

  return needsRestartParticleSync(cycle) || needsParticleOutput(cycle) ||
         col_->diagnostics_due(cycle);
}

std::vector<ParticleOutputArrays> IOManager::hostParticles(int cycle) const {
  if (needsRestartParticleSync(cycle))
    return std::vector<ParticleOutputArrays>(ns_, {true, true, true, true});
  std::vector<ParticleOutputArrays> arrays(ns_);
  // the kinetic energy, momentum and charge of every species
  if (col_->diagnostics_due(cycle))
    for (ParticleOutputArrays& species : arrays)
      species.velocity = species.charge = true;
  if (needsParticleOutput(cycle))
    for (const int s : col_->getParticlesOutputSpecies())
      arrays[s] |= col_->getParticlesOutputArrays();
  return arrays;
}

bool IOManager::needsRestartParticleSync(int cycle) const {
  if constexpr (!BenchmarkConfig::DISK_OUTPUT_ENABLED)
    return false;

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
