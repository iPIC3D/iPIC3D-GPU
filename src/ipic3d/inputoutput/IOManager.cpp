/*
 * IOManager.cpp - Modular I/O manager implementation for iPIC3D
 *
 * Each write*() method dispatches to the selected backend.
 * Backend-specific code is isolated behind its own preprocessor guard,
 * so compiling without HDF5 or ADIOS2 simply disables those paths —
 * no entanglement between unrelated backends.
 */

#include "IOManager.h"
#include "Collective.h"
#include "VCtopology3D.h"
#include "Grid3DCU.h"
#include "EMfields3D.h"
#include "Particles3D.h"
#include "Alloc.h"         // newArr3, newArr4, delArr3, delArr4
#include "ParallelIO.h"    // free VTK/H5hut/PHDF5 write functions
#include "debug.h"         // eprintf, warning_printf
#include "Parameters.h"

#include <string>
#include <iostream>

#ifndef NO_HDF5
#include "OutputWrapperFPP.h"
#endif

#ifdef USE_ADIOS2
#include "ADIOS2IO.hpp"
#endif

using std::string;

// ---------------------------------------------------------------------------
// Construction / Destruction
// ---------------------------------------------------------------------------

IOManager::IOManager() = default;

IOManager::~IOManager() {
#ifndef NO_HDF5
    delete outputWrapperFPP_;
#endif
#ifdef USE_ADIOS2
    delete adiosManager_;
#endif
    // Free VTK write buffers (delArr needs the first two dimensions). 
    // Note: the original code never freed these — this fixes that leak.
    if (fieldwritebuffer_ && grid_) {
        int dim0 = (fieldBackend_ == FieldBackend::NBCVTK)
                       ? (grid_->getNZN() - 3) * 4
                       : (grid_->getNZN() - 3);
        delArr4(fieldwritebuffer_, dim0, grid_->getNYN() - 3, grid_->getNXN() - 3);
    }
    if (momentwritebuffer_ && grid_) {
        int dim0 = (fieldBackend_ == FieldBackend::NBCVTK)
                       ? (grid_->getNZN() - 3) * 14
                       : (grid_->getNZN() - 3);
        delArr3(momentwritebuffer_, dim0, grid_->getNYN() - 3);
    }
}

// ---------------------------------------------------------------------------
// Initialisation
// ---------------------------------------------------------------------------

void IOManager::init(Collective* col, VCtopology3D* vct, Grid3DCU* grid,
                     EMfields3D* EMf, Particles3D* outputPart, int ns,
                     Particles3D* testpart, int nstestpart, int first_cycle)
{
    // Store non-owning pointers
    col_         = col;
    vct_         = vct;
    grid_        = grid;
    EMf_         = EMf;
    outputPart_  = outputPart;
    testpart_    = testpart;
    ns_          = ns;
    nstestpart_  = nstestpart;
    first_cycle_ = first_cycle;
    restart_cycle_ = col->getRestartOutputCycle();

    const string writeMethod = col->getWriteMethod();

    // ---- 1. Determine FIELD backend from WriteMethod ----
    if      (writeMethod == "shdf5")  fieldBackend_ = FieldBackend::SHDF5;
    else if (writeMethod == "pvtk")   fieldBackend_ = FieldBackend::PVTK;
    else if (writeMethod == "nbcvtk") fieldBackend_ = FieldBackend::NBCVTK;
    else if (writeMethod == "phdf5")  fieldBackend_ = FieldBackend::PARALLEL_HDF5;
    else if (writeMethod == "H5hut")  fieldBackend_ = FieldBackend::H5HUT;
    else if (writeMethod == "adios2") fieldBackend_ = FieldBackend::ADIOS2;
    else {
        warning_printf("Unknown WriteMethod '%s'. "
                       "Valid options: shdf5, pvtk, nbcvtk, phdf5, H5hut, adios2",
                       writeMethod.c_str());
        fieldBackend_ = FieldBackend::NONE;
    }

    // ---- 2. Determine PARTICLE backend ----
#ifdef USE_ADIOS2
    particleBackend_ = ParticleBackend::ADIOS2;
#else
    if (writeMethod == "H5hut")
        particleBackend_ = ParticleBackend::H5HUT;
    else
        particleBackend_ = ParticleBackend::SHDF5;
#endif

    // ---- 3. Determine RESTART backend ----
#ifdef USE_ADIOS2
    restartBackend_ = RestartBackend::ADIOS2;
#else
    restartBackend_ = RestartBackend::SHDF5;
#endif

    // ---- 4. Validate backend availability against compile flags ----
#ifdef NO_HDF5
    if (fieldBackend_ == FieldBackend::SHDF5 ||
        fieldBackend_ == FieldBackend::PARALLEL_HDF5 ||
        fieldBackend_ == FieldBackend::H5HUT) {
        eprintf("WriteMethod '%s' requires HDF5 (compile with USE_HDF5=ON)",
                writeMethod.c_str());
    }
    if (particleBackend_ == ParticleBackend::SHDF5 ||
        particleBackend_ == ParticleBackend::H5HUT) {
        eprintf("Selected particle backend requires HDF5 (compile with USE_HDF5=ON)");
    }
    if (restartBackend_ == RestartBackend::SHDF5) {
        eprintf("Selected restart backend requires HDF5 (compile with USE_HDF5=ON)");
    }
#endif
#ifndef USE_ADIOS2
    if (fieldBackend_ == FieldBackend::ADIOS2) {
        eprintf("WriteMethod 'adios2' requires ADIOS2 (compile with USE_ADIOS2=ON)");
    }
    if (particleBackend_ == ParticleBackend::ADIOS2) {
        eprintf("Selected particle backend requires ADIOS2 (compile with USE_ADIOS2=ON)");
    }
    if (restartBackend_ == RestartBackend::ADIOS2) {
        eprintf("Selected restart backend requires ADIOS2 (compile with USE_ADIOS2=ON)");
    }
#endif

    // ---- 5. Create HDF5 backend (OutputWrapperFPP) if any path needs it ----
#ifndef NO_HDF5
    {
        bool needHDF5 = false;
        if (fieldBackend_ == FieldBackend::SHDF5)       needHDF5 = true;
        if (particleBackend_ == ParticleBackend::SHDF5)  needHDF5 = true;
        if (restartBackend_ == RestartBackend::SHDF5)    needHDF5 = true;
        // PHDF5 and H5hut use free functions — no OutputWrapperFPP needed

        if (needHDF5) {
            outputWrapperFPP_ = new OutputWrapperFPP;
            outputWrapperFPP_->init_output_files(
                col, vct, grid, EMf, outputPart, ns, testpart, nstestpart);
        }
    }
#endif

    // ---- 6. Create ADIOS2 backend if any path needs it ----
#ifdef USE_ADIOS2
    if (particleBackend_ == ParticleBackend::ADIOS2 ||
        restartBackend_  == RestartBackend::ADIOS2  ||
        fieldBackend_    == FieldBackend::ADIOS2)
    {
        using namespace std::string_literals;
        string particleTag = col->getParticlesOutputCycle()
                                 ? col->getPclOutputTag() : ""s;

        adiosManager_ = new ADIOS2IO::ADIOS2Manager();
        adiosManager_->initOutputFiles(
            ""s, particleTag, 0,
            col, vct, grid, EMf, outputPart, ns, testpart, nstestpart);
    }
#endif

    // ---- 7. Allocate VTK write buffers if needed ----
    if (!col->field_output_is_off()) {
        if (fieldBackend_ == FieldBackend::PVTK) {
            if (!col->getFieldOutputTag().empty())
                fieldwritebuffer_ = newArr4(float,
                    grid->getNZN()-3, grid->getNYN()-3, grid->getNXN()-3, 3);
            if (!col->getMomentsOutputTag().empty())
                momentwritebuffer_ = newArr3(float,
                    grid->getNZN()-3, grid->getNYN()-3, grid->getNXN()-3);
        }
        else if (fieldBackend_ == FieldBackend::NBCVTK) {
            fieldreqcounter_  = 0;
            momentreqcounter_ = 0;
            if (!col->getFieldOutputTag().empty())
                fieldwritebuffer_ = newArr4(float,
                    (grid->getNZN()-3)*4, grid->getNYN()-3, grid->getNXN()-3, 3);
            if (!col->getMomentsOutputTag().empty())
                momentwritebuffer_ = newArr3(float,
                    (grid->getNZN()-3)*14, grid->getNYN()-3, grid->getNXN()-3);
        }
    }
}

// ---------------------------------------------------------------------------
// Field output
// ---------------------------------------------------------------------------

void IOManager::writeFields(int cycle) {
    switch (fieldBackend_) {

    // -- Serial HDF5 (one file per process) --
    case FieldBackend::SHDF5:
#ifndef NO_HDF5
        if (!col_->getFieldOutputTag().empty())
            outputWrapperFPP_->append_output(
                col_->getFieldOutputTag().c_str(), cycle);
        if (!col_->getMomentsOutputTag().empty())
            outputWrapperFPP_->append_output(
                col_->getMomentsOutputTag().c_str(), cycle);
#endif
        break;

    // -- Blocking collective MPI-IO VTK --
    case FieldBackend::PVTK:
        if (!col_->getFieldOutputTag().empty())
            WriteFieldsVTK(grid_, EMf_, col_, vct_,
                           col_->getFieldOutputTag(), cycle, fieldwritebuffer_);
        if (!col_->getMomentsOutputTag().empty())
            WriteMomentsVTK(grid_, EMf_, col_, vct_,
                            col_->getMomentsOutputTag(), cycle, momentwritebuffer_);
        break;

    // -- Non-blocking collective MPI-IO VTK --
    case FieldBackend::NBCVTK:
        // Complete any pending writes from the previous cycle
        if (!col_->getFieldOutputTag().empty()) {
            if (fieldreqcounter_ > 0) {
                for (int si = 0; si < fieldreqcounter_; si++) {
                    int ec = MPI_File_write_all_end(
                        fieldfhArr_[si],
                        &fieldwritebuffer_[si][0][0][0],
                        &fieldstsArr_[si]);
                    if (ec != MPI_SUCCESS) {
                        char es[100]; int len, cls;
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
                grid_, EMf_, col_, vct_, cycle,
                fieldwritebuffer_, fieldreqArr_, fieldfhArr_);
        }

        if (!col_->getMomentsOutputTag().empty()) {
            if (momentreqcounter_ > 0) {
                for (int si = 0; si < momentreqcounter_; si++) {
                    int ec = MPI_File_write_all_end(
                        momentfhArr_[si],
                        &momentwritebuffer_[si][0][0],
                        &momentstsArr_[si]);
                    if (ec != MPI_SUCCESS) {
                        char es[100]; int len, cls;
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
                grid_, EMf_, col_, vct_, cycle,
                momentwritebuffer_, momentreqArr_, momentfhArr_);
        }
        break;

    // -- Parallel HDF5 --
    case FieldBackend::PARALLEL_HDF5:
#ifndef NO_HDF5
        WriteOutputParallel(grid_, EMf_, outputPart_, col_, vct_, cycle);
#endif
        break;

    // -- H5hut --
    case FieldBackend::H5HUT:
#ifndef NO_HDF5
        WriteFieldsH5hut(ns_, grid_, EMf_, col_, vct_, cycle);
#endif
        break;

    // -- ADIOS2 field output --
    case FieldBackend::ADIOS2:
#ifdef USE_ADIOS2
        adiosManager_->appendFieldOutput(cycle);
#endif
        break;

    case FieldBackend::NONE:
        break;
    }
}

// ---------------------------------------------------------------------------
// Particle output
// ---------------------------------------------------------------------------

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
            outputWrapperFPP_->append_output(
                col_->getPclOutputTag().c_str(), cycle, 0);
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

// ---------------------------------------------------------------------------
// Test-particle output (currently HDF5-only)
// ---------------------------------------------------------------------------

void IOManager::writeTestParticles(int cycle) {
    if (nstestpart_ == 0) return;

    // Convert test particles (CPU-only, no CUDA sync needed)
    for (int i = 0; i < nstestpart_; i++) {
        testpart_[i].set_particleType(ParticleType::Type::AoS);
        testpart_[i].convertParticlesToSynched();
    }

#ifndef NO_HDF5
    if (outputWrapperFPP_)
        outputWrapperFPP_->append_output(
            "testpartpos + testpartvel+ testparttag", cycle, 0);
#endif
}

// ---------------------------------------------------------------------------
// Restart output
// ---------------------------------------------------------------------------

void IOManager::writeRestart(int cycle) {
    switch (restartBackend_) {

    case RestartBackend::ADIOS2:
#ifdef USE_ADIOS2
        adiosManager_->appendRestartOutput(cycle);
#endif
        break;

    case RestartBackend::SHDF5:
#ifndef NO_HDF5
        if (outputWrapperFPP_)
            outputWrapperFPP_->append_restart(cycle);
#endif
        break;

    case RestartBackend::NONE:
        break;
    }
}

// ---------------------------------------------------------------------------
// Finalise
// ---------------------------------------------------------------------------

void IOManager::finalize() {
#ifdef USE_ADIOS2
    if (adiosManager_)
        adiosManager_->closeOutputFiles();
#endif
    // OutputWrapperFPP has no explicit close — destructor handles cleanup.
}

// ---------------------------------------------------------------------------
// Scheduling query
// ---------------------------------------------------------------------------

bool IOManager::needsParticleSync(int cycle) const {
    if (restart_cycle_ > 0 && cycle % restart_cycle_ == 0)
        return true;
    if (!col_->particle_output_is_off() &&
        cycle % col_->getParticlesOutputCycle() == 0)
        return true;
    return false;
}
