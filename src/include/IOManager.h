/**
 * @file IOManager.h
 * @brief Modular I/O manager for iPIC3D.
 *
 * Decouples field, particle, and restart I/O so each concern can select its
 * backend independently.
 */

#ifndef IO_MANAGER_H
#define IO_MANAGER_H

#include "ipicfwd.h"
#include "arraysfwd.h"        // arr3_double, array4_double
#include "aligned_vector.h"   // vector_double
#include "RestartSlotManager.h"
#include <string>
#include <vector>

#ifndef NO_MPI
#include "mpi.h"
#endif

// Forward declarations; keep backend headers out of this translation unit.
#ifndef NO_HDF5
class OutputWrapperFPP;
#endif

#ifdef USE_ADIOS2
namespace ADIOS2IO { class ADIOS2Manager; }
#endif

class IOManager {
public:
    // ======= Backend enums =======
    enum class FieldBackend    { NONE, SHDF5, PVTK, NBCVTK, PARALLEL_HDF5, H5HUT, ADIOS2 };
    enum class ParticleBackend { NONE, SHDF5, H5HUT, ADIOS2 };
    enum class RestartBackend  { NONE, SHDF5, ADIOS2 };

    IOManager();
    ~IOManager();

    // Non-copyable, non-movable.
    IOManager(const IOManager&) = delete;
    IOManager& operator=(const IOManager&) = delete;

    /**
     * @brief Initialise backends based on compile-time options and runtime config.
     *
     * Must be called once, after Collective/Grid/EMfields/Particles are ready.
     * The pointers are stored but not owned; the caller (`c_Solver`) keeps ownership.
     *
     * @param col Collective input/configuration object.
     * @param vct MPI topology descriptor.
     * @param grid Local grid descriptor.
     * @param EMf Electromagnetic-field container.
     * @param outputPart Array of per-species host particle containers.
     * @param ns Number of particle species.
     * @param testpart Array of host test-particle containers.
     * @param nstestpart Number of test-particle species.
     * @param first_cycle First simulation cycle to consider for output scheduling.
     */
    void init(Collective* col, VCtopology3D* vct, Grid3DCU* grid,
              EMfields3D* EMf, ParticleSoAHost** outputPart, int ns,
              ParticleSoAHost** testpart, int nstestpart, int first_cycle);

    // ======= Write methods =======

    /**
     * @brief Write field data (E, B, J, rho, moments).
     *
     * @param cycle Simulation cycle being written.
     */
    void writeFields(int cycle);

    /**
     * @brief Write particle data (position, velocity, charge, ID).
     *
     * @param cycle Simulation cycle being written.
     */
    void writeParticles(int cycle);

    /**
     * @brief Write test-particle data.
     *
     * @param cycle Simulation cycle being written.
     */
    void writeTestParticles(int cycle);

    /**
     * @brief Write a restart checkpoint containing fields and particles.
     *
     * @param cycle Restart label stored in the checkpoint. This is the loop
     *              cycle that a restart will execute first.
     */
    void writeRestart(int cycle);

    // ======= Restart read methods =======

    /**
     * @brief Read EM fields and species densities from a restart checkpoint.
     *
     * Delegates to RestartReader::readFields using the restart directory
     * and last cycle stored in the Collective configuration.
     *
     * @param vct MPI topology descriptor.
     * @param grid Local grid descriptor.
     * @param Bxn Restart target array for magnetic field Bx on nodes.
     * @param Byn Restart target array for magnetic field By on nodes.
     * @param Bzn Restart target array for magnetic field Bz on nodes.
     * @param Ex Restart target array for electric field Ex on nodes.
     * @param Ey Restart target array for electric field Ey on nodes.
     * @param Ez Restart target array for electric field Ez on nodes.
     * @param rhons Restart target arrays for per-species charge density.
     * @param ns Number of particle species.
     */
    void readFieldRestart(
        const VCtopology3D* vct, const Grid3DCU* grid,
        arr3_double Bxn, arr3_double Byn, arr3_double Bzn,
        arr3_double Ex,  arr3_double Ey,  arr3_double Ez,
        array4_double* rhons, int ns);

    /**
     * @brief Read particle data from a restart checkpoint.
     *
     * Delegates to RestartReader::readParticles using the restart directory
     * and last cycle stored in the Collective configuration.
     *
     * @param vct MPI topology descriptor.
     * @param species_number Species index to read.
     * @param u Restart target vector for x-velocity.
     * @param v Restart target vector for y-velocity.
     * @param w Restart target vector for z-velocity.
     * @param q Restart target vector for particle charge/weight.
     * @param x Restart target vector for x-position.
     * @param y Restart target vector for y-position.
     * @param z Restart target vector for z-position.
     * @param t Restart target vector for particle tag/time.
     */
    void readParticlesRestart(
        const VCtopology3D* vct, int species_number,
        vector_double& u, vector_double& v, vector_double& w,
        vector_double& q,
        vector_double& x, vector_double& y, vector_double& z,
        vector_double& t);

    /**
     * @brief Close output files and release backend resources.
     *
     * Called from c_Solver::Finalize() after the last restart has been written.
     */
    void finalize();

    // ======= GPU-host copy scheduling queries =======

    /**
     * @brief Will the given cycle require particle data on the host?
     *
     * Used by `c_Solver::outputCopyAsync()` to decide whether to schedule
     * a GPU-to-host memcpy for the next cycle.
     *
     * @param cycle Simulation cycle to query.
     */
    bool needsParticleSync(int cycle) const;

    // ======= Accessor helpers =======
    FieldBackend    getFieldBackend()    const { return fieldBackend_; }
    ParticleBackend getParticleBackend() const { return particleBackend_; }
    RestartBackend  getRestartBackend()  const { return restartBackend_; }

private:
    // ======= Backend selection =======
    FieldBackend    fieldBackend_    = FieldBackend::NONE;
    ParticleBackend particleBackend_ = ParticleBackend::NONE;
    RestartBackend  restartBackend_  = RestartBackend::NONE;
    RestartSlotManager restartSlots_;

    // ======= Owned backend objects =======
#ifndef NO_HDF5
    OutputWrapperFPP* outputWrapperFPP_ = nullptr;
#endif
#ifdef USE_ADIOS2
    ADIOS2IO::ADIOS2Manager* adiosManager_ = nullptr;
#endif

    // ======= Owned VTK write buffers =======
    float**** fieldwritebuffer_  = nullptr;
    float***  momentwritebuffer_ = nullptr;

    // Local write sizes (interior nodes + boundary node for upper processes)
    int localWriteNx_ = 0;
    int localWriteNy_ = 0;
    int localWriteNz_ = 0;

    // Allocated first-dimension sizes for VTK write buffers (for deallocation)
    int fieldBufDim0_  = 0;
    int momentBufDim0_ = 0;

    // NBCVTK non-blocking state (dynamically sized based on OutputTagConfig)
    std::vector<MPI_Request> fieldreqArr_;
    std::vector<MPI_File>    fieldfhArr_;
    std::vector<MPI_Status>  fieldstsArr_;
    int         fieldreqcounter_  = 0;

    std::vector<MPI_Request> momentreqArr_;
    std::vector<MPI_File>    momentfhArr_;
    std::vector<MPI_Status>  momentstsArr_;
    int         momentreqcounter_  = 0;

    /**
     * @brief Complete any in-flight NBCVTK split-collective writes and close their files.
     *
     * Safe to call when the field backend is not NBCVTK (no-op)
     * (resets the per-buffer counters after draining).
     */
    void drainNBCVTKPending();

    // ======= Registered non-owning pointers =======
    Collective*   col_        = nullptr;
    VCtopology3D* vct_        = nullptr;
    Grid3DCU*     grid_       = nullptr;
    EMfields3D*   EMf_        = nullptr;
    ParticleSoAHost**  outputPart_ = nullptr;
    ParticleSoAHost**  testpart_   = nullptr;
    int ns_        = 0;
    int nstestpart_= 0;
    int first_cycle_   = 0;
    int restart_cycle_ = 0;
};

#endif // IO_MANAGER_H
