/*
 * IOManager.h - Modular I/O manager for iPIC3D
 *
 * Decouples field, particle, and restart I/O so each concern can
 * independently use any available backend (HDF5, VTK, ADIOS2, H5hut, etc.).
 *
 * Design rationale:
 *   - WriteMethod (from input file) determines the FIELD output backend.
 *   - Particle output backend is selected independently:
 *       USE_ADIOS2 → ADIOS2;   H5hut WriteMethod → H5hut;   else → serial HDF5.
 *   - Restart output backend is selected independently:
 *       USE_ADIOS2 → ADIOS2;   else → serial HDF5.
 *   - No preprocessor guard leaks: each backend's code is self-contained
 *     behind its own #ifdef / #ifndef.
 */

#ifndef IO_MANAGER_H
#define IO_MANAGER_H

#include "ipicfwd.h"
#include "arraysfwd.h"        // arr3_double, array4_double
#include "aligned_vector.h"   // vector_double
#include <string>

#ifndef NO_MPI
#include "mpi.h"
#endif

// Forward declarations — no backend headers pulled into the translation unit
#ifndef NO_HDF5
class OutputWrapperFPP;
#endif

#ifdef USE_ADIOS2
namespace ADIOS2IO { class ADIOS2Manager; }
#endif

class IOManager {
public:
    // ---- backend enums (field, particle, restart are independent) ----
    enum class FieldBackend    { NONE, SHDF5, PVTK, NBCVTK, PARALLEL_HDF5, H5HUT, ADIOS2 };
    enum class ParticleBackend { NONE, SHDF5, H5HUT, ADIOS2 };
    enum class RestartBackend  { NONE, SHDF5, ADIOS2 };

    IOManager();
    ~IOManager();

    // Non-copyable, non-movable
    IOManager(const IOManager&) = delete;
    IOManager& operator=(const IOManager&) = delete;

    /**
     * @brief Initialise backends based on compile-time options and runtime config.
     *
     * Must be called once, after Collective/Grid/EMfields/Particles are ready.
     * The pointers are stored but NOT owned — the caller (c_Solver) keeps ownership.
     */
    void init(Collective* col, VCtopology3D* vct, Grid3DCU* grid,
              EMfields3D* EMf, ParticleSoAHost* outputPart, int ns,
              ParticleSoAHost* testpart, int nstestpart, int first_cycle);

    // ---- write methods (no cycle-gating — caller decides when to call) ----

    /** Write field data (E, B, J, rho, moments). */
    void writeFields(int cycle);

    /** Write particle data (position, velocity, charge, ID). */
    void writeParticles(int cycle);

    /** Write test-particle data. */
    void writeTestParticles(int cycle);

    /** Write restart checkpoint (fields + particles). */
    void writeRestart(int cycle);

    // ---- restart READ methods (dispatch to RestartReader) ----

    /**
     * @brief Read EM fields and species densities from a restart checkpoint.
     *
     * Delegates to RestartReader::readFields using the restart directory
     * and last cycle stored in the Collective configuration.
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

    // ---- queries used by c_Solver for GPU-host copy scheduling ----

    /**
     * @brief Will the given cycle require particle data on the host?
     *
     * Used by c_Solver::outputCopyAsync() to decide whether to schedule
     * a GPU→host memcpy for the NEXT cycle.
     */
    bool needsParticleSync(int cycle) const;

    // ---- accessor helpers ----
    FieldBackend    getFieldBackend()    const { return fieldBackend_; }
    ParticleBackend getParticleBackend() const { return particleBackend_; }
    RestartBackend  getRestartBackend()  const { return restartBackend_; }

private:
    // ---- backend selection ----
    FieldBackend    fieldBackend_    = FieldBackend::NONE;
    ParticleBackend particleBackend_ = ParticleBackend::NONE;
    RestartBackend  restartBackend_  = RestartBackend::NONE;

    // ---- backend objects (owned) ----
#ifndef NO_HDF5
    OutputWrapperFPP* outputWrapperFPP_ = nullptr;
#endif
#ifdef USE_ADIOS2
    ADIOS2IO::ADIOS2Manager* adiosManager_ = nullptr;
#endif

    // ---- VTK / NBCVTK write buffers (owned) ----
    float**** fieldwritebuffer_  = nullptr;
    float***  momentwritebuffer_ = nullptr;

    // Local write sizes (interior nodes + boundary node for upper processes)
    int localWriteNx_ = 0;
    int localWriteNy_ = 0;
    int localWriteNz_ = 0;

    // NBCVTK non-blocking state
    MPI_Request fieldreqArr_[4]   = {};
    MPI_File    fieldfhArr_[4]    = {};
    MPI_Status  fieldstsArr_[4]   = {};
    int         fieldreqcounter_  = 0;

    MPI_Request momentreqArr_[14]  = {};
    MPI_File    momentfhArr_[14]   = {};
    MPI_Status  momentstsArr_[14]  = {};
    int         momentreqcounter_  = 0;

    // ---- registered (non-owning) pointers ----
    Collective*   col_        = nullptr;
    VCtopology3D* vct_        = nullptr;
    Grid3DCU*     grid_       = nullptr;
    EMfields3D*   EMf_        = nullptr;
    ParticleSoAHost*  outputPart_ = nullptr;
    ParticleSoAHost*  testpart_   = nullptr;
    int ns_        = 0;
    int nstestpart_= 0;
    int first_cycle_   = 0;
    int restart_cycle_ = 0;
};

#endif // IO_MANAGER_H
