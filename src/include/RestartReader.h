/*
 * RestartReader.h - Modular restart-file reader for iPIC3D
 *
 * Extracted from Collective.cpp to separate I/O concerns from
 * configuration/parameter management.  All checkpoint-reading
 * logic (ADIOS2 and HDF5 backends) lives here.
 *
 * Usage:
 *   - Standalone via static helpers (readLastCycle) during early init
 *   - Through IOManager::readFieldRestart / readParticlesRestart
 *   - Through Collective thin wrappers (backward compatibility)
 */

#ifndef RESTART_READER_H
#define RESTART_READER_H

#include "arraysfwd.h"        // arr3_double, array4_double
#include "aligned_vector.h"   // vector_double
#include "RestartSlotManager.h"
#include <string>

// Forward declarations
class VCtopology3D;
class Grid3DCU;
typedef Grid3DCU Grid;

class RestartReader {
public:

    // ---------------------------------------------------------------
    // Static helpers (usable before IOManager / RestartReader exist)
    // ---------------------------------------------------------------

    /**
     * @brief Read the checkpoint cycle label from the restart directory.
     *
     * Resolves A/B restart metadata when present, otherwise falls back to the
     * legacy flat layout. Returns the checkpoint cycle label.
     *
     * @param restartDir  Path to the directory containing restart files.
     * @return            The loop cycle to execute first after restart.
     */
    static int readLastCycle(const std::string& restartDir);

    /**
     * @brief Resolve the checkpoint directory and cycle label to read.
     *
     * New restart layouts use RestartDirName/restart_A or restart_B plus
     * latest_restart.json metadata.  If that metadata is absent, this falls
     * back to the legacy flat layout directly under RestartDirName.
     *
     * @param restartDir  User-provided restart root directory.
     * @return            Checkpoint metadata, including the directory that
     *                    contains rank-local restart files.
     */
    static RestartCheckpoint resolveLatestCheckpoint(
        const std::string& restartDir);

    // ---------------------------------------------------------------
    // Field restart
    // ---------------------------------------------------------------

    /**
     * @brief Read EM fields (B, E) and species densities from a restart file.
     *
     * ADIOS2 and HDF5 backends both store active-node data only and place it
     * into the guarded node arrays at offset [1][1][1]. Ghost nodes are
     * rebuilt by the field communication step after restart loading.
     *
     * @param vct         Cartesian topology (provides rank).
     * @param grid        Local grid (provides NXN, NYN, NZN).
     * @param Bxn,Byn,Bzn Magnetic field node arrays (output).
     * @param Ex,Ey,Ez    Electric field node arrays (output).
     * @param rhons       Species density array (output).
     * @param ns          Number of species.
     * @param restartDir  Path to restart directory.
     * @param last_cycle  Expected restart cycle label (validated against file).
     */
    static void readFields(
        const VCtopology3D* vct,
        const Grid* grid,
        arr3_double Bxn, arr3_double Byn, arr3_double Bzn,
        arr3_double Ex,  arr3_double Ey,  arr3_double Ez,
        array4_double* rhons, int ns,
        const std::string& restartDir, int last_cycle);

    // ---------------------------------------------------------------
    // Particle restart
    // ---------------------------------------------------------------

    /**
     * @brief Read particle data (position, velocity, charge, ID) from
     *        a restart file.
     *
     * ADIOS2 backend: reads from restart_<rank>.bp (variables named
     *   part<i>PositionX, etc.).
     * HDF5 backend: reads from restart<rank>.hdf (datasets under
     *   /particles/species_<i>/{x,y,z,u,v,w,q,ID}/cycle_N).
     *
     * Vectors are resized to the particle count found in the file,
     * with capacity rounded up to DVECWIDTH.
     *
     * @param vct             Cartesian topology (provides rank).
     * @param species_number  Species index.
     * @param u,v,w           Velocity components (output).
     * @param q               Charge per particle (output).
     * @param x,y,z           Position components (output).
     * @param t               Particle ID (output, stored as double).
     * @param restartDir      Path to restart directory.
     * @param last_cycle      Expected restart cycle label.
     */
    static void readParticles(
        const VCtopology3D* vct,
        int species_number,
        vector_double& u, vector_double& v, vector_double& w,
        vector_double& q,
        vector_double& x, vector_double& y, vector_double& z,
        vector_double& t,
        const std::string& restartDir, int last_cycle);
};

#endif // RESTART_READER_H
