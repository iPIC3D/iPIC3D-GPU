/*
 * RestartReader.cpp - Restart-file reader implementation for iPIC3D
 *
 * Contains all checkpoint-reading logic, extracted from Collective.cpp.
 * Supports ADIOS2 (BP5) and HDF5 backends via compile-time guards.
 */

#include "RestartReader.h"

#include "VCtopology3D.h"
#include "Grid3DCU.h"
#include "MPIdata.h"
#include "debug.h"          // eprintf
#include "ipichdf5.h"       // HDF5 headers (guarded by NO_HDF5)
#include "ipicdefs.h"       // DVECWIDTH
#include "ipicmath.h"       // roundup_to_multiple
#include "CUDA/cudaTypeDef.cuh"  // cudaCommonType

#include <sstream>
#include <string>
#include <iostream>

#ifdef USE_ADIOS2
#include "adios2.h"
#endif

using std::string;
using std::stringstream;

namespace {

int readLegacyLastCycle(const std::string& restartDir)
{
    int last_cycle = -1;

#ifdef USE_ADIOS2
    // ---- ADIOS2 legacy layout: read from RestartDirName/restart_0.bp ----
    string filePath = restartDir + "/restart_0.bp";

    adios2::ADIOS adios;
    adios2::IO    io     = adios.DeclareIO("restartCycleRead");
    io.SetEngine("BP5");
    adios2::Engine engine = io.Open(filePath, adios2::Mode::Read);

    auto stepNum = engine.Steps();

    for (unsigned int step = 0;
         engine.BeginStep() == adios2::StepStatus::OK; ++step)
    {
        if (step < stepNum - 1) {   // advance to the last step
            engine.EndStep();
            continue;
        }
        engine.Get("cycle", last_cycle);
        engine.EndStep();
        break;
    }
    engine.Close();

#elif !defined(NO_HDF5)
    // ---- HDF5 legacy layout: read from RestartDirName/restart0.hdf ----
    string filePath = restartDir + "/restart0.hdf";

    hid_t file_id = H5Fopen(filePath.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
    if (file_id < 0)
        eprintf("ERROR: could not open HDF5 restart file: %s", filePath.c_str());

    hid_t dataset_id = H5Dopen2(file_id, "/last_cycle", H5P_DEFAULT);
    if (dataset_id < 0) {
        H5Fclose(file_id);
        eprintf("ERROR: could not open /last_cycle dataset in %s", filePath.c_str());
    }

    H5Dread(dataset_id, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT,
            &last_cycle);
    H5Dclose(dataset_id);
    H5Fclose(file_id);

#else
    eprintf("Restart requires compiling with USE_ADIOS2 or HDF5 (without NO_HDF5).");
#endif

    return last_cycle;
}

} // namespace

// ===========================================================================
// readLastCycle  —  retrieve the restart cycle label from the newest checkpoint
// ===========================================================================

RestartCheckpoint RestartReader::resolveLatestCheckpoint(
    const std::string& restartDir)
{
    const std::string backend = RestartSlotManager::backendName();
    if (backend.empty()) {
        eprintf("Restart requires compiling with USE_ADIOS2 or HDF5 (without NO_HDF5).");
    }

#if !defined(USE_ADIOS2) && !defined(NO_HDF5)
    if (MPIdata::get_rank() == 0) {
        printf("\n");
        printf("=========================================================================\n");
        printf("  WARNING: HDF5 restart is a Beta feature. Use with caution!\n");
        printf("=========================================================================\n");
        printf("\n");
    }
#endif

    RestartCheckpoint checkpoint =
        RestartSlotManager::resolveLatest(restartDir, backend,
                                          MPIdata::get_nprocs());
    if (checkpoint.found) {
        if (MPIdata::get_rank() == 0) {
            std::cout << "[*] Restart checkpoint = restart_"
                      << checkpoint.slot
                      << ", cycle label = " << checkpoint.cycle << std::endl;
        }
        return checkpoint;
    }

    checkpoint.found = true;
    checkpoint.legacy = true;
    checkpoint.rootDir = restartDir;
    checkpoint.dataDir = restartDir;
    checkpoint.backend = backend;
    checkpoint.nranks = MPIdata::get_nprocs();
    checkpoint.cycle = readLegacyLastCycle(restartDir);

    if (MPIdata::get_rank() == 0) {
        std::cout << "[*] Restart checkpoint = legacy flat layout"
                  << ", cycle label = " << checkpoint.cycle << std::endl;
    }

    return checkpoint;
}

int RestartReader::readLastCycle(const std::string& restartDir)
{
    return resolveLatestCheckpoint(restartDir).cycle;
}

// ===========================================================================
// readFields  —  EM fields + species charge densities
// ===========================================================================

void RestartReader::readFields(
    const VCtopology3D* vct,
    const Grid* grid,
    arr3_double Bxn, arr3_double Byn, arr3_double Bzn,
    arr3_double Ex,  arr3_double Ey,  arr3_double Ez,
    array4_double* rhons_, int ns,
    const std::string& restartDir, int last_cycle)
{
#ifdef USE_ADIOS2
    // ---- ADIOS2 restart read (includes ghost cells) ----
    const int nxn = grid->getNXN();
    const int nyn = grid->getNYN();
    const int nzn = grid->getNZN();

    if (vct->getCartesian_rank() == 0)
        printf("LOADING EM FIELD FROM RESTART FILE in %s/restart.bp\n",
               restartDir.c_str());

    stringstream ss;
    ss << vct->getCartesian_rank();
    string name_file = restartDir + "/restart_" + ss.str() + ".bp";

    adios2::ADIOS  adios;
    adios2::IO     ioField    = adios.DeclareIO("Field");
    ioField.SetEngine("BP5");
    adios2::Engine engineField = ioField.Open(name_file, adios2::Mode::Read);

    auto stepNum = engineField.Steps();

    for (unsigned int step = 0;
         engineField.BeginStep() == adios2::StepStatus::OK; ++step)
    {
        if (step < stepNum - 1) {
            engineField.EndStep();
            continue;
        }

        // Validate cycle
        int lastCycle = -1;
        engineField.Get<int>("cycle", lastCycle, adios2::Mode::Sync);
        if (lastCycle != last_cycle) {
            engineField.EndStep();
            engineField.Close();
            printf("last_cycle = %d\n", lastCycle);
            printf("last_cycle = %d\n", last_cycle);
            eprintf("restart cycle label in file does not match the selected checkpoint label");
        } else {
            if (MPIdata::get_rank() == 0)
                std::cout << "[*] Fields Restarting from cycle label: "
                          << lastCycle << std::endl;
        }

        // B field
        engineField.Get<cudaCommonType>("Bx", (cudaCommonType*)Bxn.get_arr(),
                                        adios2::Mode::Deferred);
        engineField.Get<cudaCommonType>("By", (cudaCommonType*)Byn.get_arr(),
                                        adios2::Mode::Deferred);
        engineField.Get<cudaCommonType>("Bz", (cudaCommonType*)Bzn.get_arr(),
                                        adios2::Mode::Deferred);
        // E field
        engineField.Get<cudaCommonType>("Ex", (cudaCommonType*)Ex.get_arr(),
                                        adios2::Mode::Deferred);
        engineField.Get<cudaCommonType>("Ey", (cudaCommonType*)Ey.get_arr(),
                                        adios2::Mode::Deferred);
        engineField.Get<cudaCommonType>("Ez", (cudaCommonType*)Ez.get_arr(),
                                        adios2::Mode::Deferred);
        // Species density
        for (int i = 0; i < ns; i++) {
            engineField.Get<cudaCommonType>(
                "rhosSpecies" + std::to_string(i),
                (cudaCommonType*)&((*rhons_)[i][0][0][0]),
                adios2::Mode::Deferred);
        }

        engineField.EndStep();
        break;
    }
    engineField.Close();

#elif !defined(NO_HDF5)
    // ---- HDF5 restart read (ghost cells NOT stored — place into interior) ----
    if (vct->getCartesian_rank() == 0) {
        printf("\n");
        printf("=========================================================================\n");
        printf("  WARNING: HDF5 restart is a Beta feature. Use with caution!\n");
        printf("=========================================================================\n");
        printf("\n");
        printf("LOADING EM FIELD FROM HDF5 RESTART FILE in %s/restart<rank>.hdf\n",
               restartDir.c_str());
    }

    const int nxn     = grid->getNXN();
    const int nyn     = grid->getNYN();
    const int nzn     = grid->getNZN();
    const int nxn_int = nxn - 2;
    const int nyn_int = nyn - 2;
    const int nzn_int = nzn - 2;
    const int nels    = nxn_int * nyn_int * nzn_int;

    stringstream ss;
    ss << vct->getCartesian_rank();
    string name_file = restartDir + "/restart" + ss.str() + ".hdf";

    hid_t file_id = H5Fopen(name_file.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
    if (file_id < 0)
        eprintf("ERROR: could not open HDF5 restart file: %s",
                name_file.c_str());

    string cycle_str = "cycle_" + std::to_string(last_cycle);

    if (MPIdata::get_rank() == 0)
        std::cout << "[*] Fields Restarting (HDF5) from cycle label: "
                  << last_cycle << std::endl;

    // Lambda: read interior-only 3D field, place at [i+1][j+1][k+1]
    auto readField3D = [&](const string& dsPath, arr3_double& arr) {
        double* buf = new double[nels];
        herr_t status = H5LTread_dataset_double(file_id, dsPath.c_str(), buf);
        if (status < 0) {
            delete[] buf;
            H5Fclose(file_id);
            eprintf("ERROR: could not read dataset %s from %s",
                    dsPath.c_str(), name_file.c_str());
        }
        for (int i = 0; i < nxn_int; i++)
            for (int j = 0; j < nyn_int; j++)
                for (int k = 0; k < nzn_int; k++)
                    arr[i+1][j+1][k+1] =
                        buf[i * nyn_int * nzn_int + j * nzn_int + k];
        delete[] buf;
    };

    // B field
    readField3D("/fields/Bx/" + cycle_str, Bxn);
    readField3D("/fields/By/" + cycle_str, Byn);
    readField3D("/fields/Bz/" + cycle_str, Bzn);

    // E field
    readField3D("/fields/Ex/" + cycle_str, Ex);
    readField3D("/fields/Ey/" + cycle_str, Ey);
    readField3D("/fields/Ez/" + cycle_str, Ez);

    // Species density (rhos)
    for (int i = 0; i < ns; i++) {
        string dsPath = "/moments/species_" + std::to_string(i)
                      + "/rho/" + cycle_str;
        double* buf = new double[nels];
        herr_t status = H5LTread_dataset_double(file_id, dsPath.c_str(), buf);
        if (status < 0) {
            delete[] buf;
            H5Fclose(file_id);
            eprintf("ERROR: could not read dataset %s from %s",
                    dsPath.c_str(), name_file.c_str());
        }
        for (int ix = 0; ix < nxn_int; ix++)
            for (int iy = 0; iy < nyn_int; iy++)
                for (int iz = 0; iz < nzn_int; iz++)
                    (*rhons_)[i][ix+1][iy+1][iz+1] =
                        buf[ix * nyn_int * nzn_int + iy * nzn_int + iz];
        delete[] buf;
    }

    H5Fclose(file_id);

#else
    eprintf("Restart requires compiling with USE_ADIOS2 or HDF5 (without NO_HDF5).");
#endif
}

// ===========================================================================
// readParticles  —  position, velocity, charge and ID for one species
// ===========================================================================

void RestartReader::readParticles(
    const VCtopology3D* vct,
    int species_number,
    vector_double& u, vector_double& v, vector_double& w,
    vector_double& q,
    vector_double& x, vector_double& y, vector_double& z,
    vector_double& t,
    const std::string& restartDir, int last_cycle)
{
#ifdef USE_ADIOS2
    // ---- ADIOS2 particle restart read ----
    if (vct->getCartesian_rank() == 0){
        printf("LOADING PARTICLE FROM RESTART FILE in %s/restart.bp\n",
               restartDir.c_str());
        printf("\n");
    }

    stringstream ss;
    ss << vct->getCartesian_rank();
    string name_file = restartDir + "/restart_" + ss.str() + ".bp";

    adios2::ADIOS  adios;
    adios2::IO     ioParticle    = adios.DeclareIO("Particles");
    ioParticle.SetEngine("BP5");
    adios2::Engine engineParticle = ioParticle.Open(name_file, adios2::Mode::Read);

    auto stepNum = engineParticle.Steps();

    for (unsigned int step = 0;
         engineParticle.BeginStep() == adios2::StepStatus::OK; ++step)
    {
        if (step < stepNum - 1) {
            engineParticle.EndStep();
            continue;
        }

        // Validate cycle
        int lastCycle = -1;
        engineParticle.Get<int>("cycle", lastCycle, adios2::Mode::Sync);
        if (lastCycle != last_cycle) {
            printf("last_cycle = %d\n", lastCycle);
            printf("last_cycle = %d\n", last_cycle);
            eprintf("restart cycle label in file does not match the selected checkpoint label");
        } else {
            if (MPIdata::get_rank() == 0)
                std::cout << "[*] Particle Restarting from cycle label: "
                          << lastCycle << std::endl;
        }

        // Determine particle count from variable shape
        string specStr = std::to_string(species_number);
        auto varX = ioParticle.InquireVariable<cudaCommonType>(
            "part" + specStr + "PositionX");
        int nop = varX.Shape()[0];

        const int padded_nop = roundup_to_multiple(nop, DVECWIDTH);
        u.reserve(padded_nop);  v.reserve(padded_nop);
        w.reserve(padded_nop);  q.reserve(padded_nop);
        x.reserve(padded_nop);  y.reserve(padded_nop);
        z.reserve(padded_nop);  t.reserve(padded_nop);

        u.resize(nop);  v.resize(nop);
        w.resize(nop);  q.resize(nop);
        x.resize(nop);  y.resize(nop);
        z.resize(nop);  t.resize(nop);

        // Read velocity
        engineParticle.Get<cudaCommonType>(
            "part" + specStr + "VelocityU", &u[0], adios2::Mode::Deferred);
        engineParticle.Get<cudaCommonType>(
            "part" + specStr + "VelocityV", &v[0], adios2::Mode::Deferred);
        engineParticle.Get<cudaCommonType>(
            "part" + specStr + "VelocityW", &w[0], adios2::Mode::Deferred);

        // Read charge
        engineParticle.Get<cudaCommonType>(
            "part" + specStr + "charge", &q[0], adios2::Mode::Deferred);

        // Read position
        engineParticle.Get<cudaCommonType>(
            "part" + specStr + "PositionX", &x[0], adios2::Mode::Deferred);
        engineParticle.Get<cudaCommonType>(
            "part" + specStr + "PositionY", &y[0], adios2::Mode::Deferred);
        engineParticle.Get<cudaCommonType>(
            "part" + specStr + "PositionZ", &z[0], adios2::Mode::Deferred);

        // Read particle ID
        engineParticle.Get<cudaCommonType>(
            "part" + specStr + "ID", &t[0], adios2::Mode::Deferred);

        engineParticle.EndStep();
        break;
    }
    engineParticle.Close();

#elif !defined(NO_HDF5)
    // ---- HDF5 particle restart read ----
    if (vct->getCartesian_rank() == 0 && species_number == 0) {
        printf("\n");
        printf("=========================================================================\n");
        printf("  WARNING: HDF5 restart is a Beta feature. Use with caution!\n");
        printf("=========================================================================\n");
        printf("\n");
        printf("LOADING PARTICLES FROM HDF5 RESTART FILE in %s/restart<rank>.hdf\n",
               restartDir.c_str());
    }

    stringstream ss;
    ss << vct->getCartesian_rank();
    string name_file = restartDir + "/restart" + ss.str() + ".hdf";

    hid_t file_id = H5Fopen(name_file.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
    if (file_id < 0)
        eprintf("ERROR: could not open HDF5 restart file: %s",
                name_file.c_str());

    string cycle_str   = "cycle_" + std::to_string(last_cycle);
    string species_str = std::to_string(species_number);

    if (MPIdata::get_rank() == 0 && species_number == 0){
        std::cout << "[*] Particle Restarting (HDF5) from cycle label: "
                  << last_cycle << std::endl;
        printf("\n");
    }
    

    // Determine particle count from the x dataset dimensions
    string xPath = "/particles/species_" + species_str + "/x/" + cycle_str;
    hid_t ds_id = H5Dopen2(file_id, xPath.c_str(), H5P_DEFAULT);
    if (ds_id < 0) {
        H5Fclose(file_id);
        eprintf("ERROR: could not open dataset %s from %s",
                xPath.c_str(), name_file.c_str());
    }
    hid_t   space_id = H5Dget_space(ds_id);
    hsize_t dims[1];
    H5Sget_simple_extent_dims(space_id, dims, NULL);
    int nop = (int)dims[0];
    H5Sclose(space_id);
    H5Dclose(ds_id);

    const int padded_nop = roundup_to_multiple(nop, DVECWIDTH);
    u.reserve(padded_nop);  v.reserve(padded_nop);
    w.reserve(padded_nop);  q.reserve(padded_nop);
    x.reserve(padded_nop);  y.reserve(padded_nop);
    z.reserve(padded_nop);  t.reserve(padded_nop);

    u.resize(nop);  v.resize(nop);
    w.resize(nop);  q.resize(nop);
    x.resize(nop);  y.resize(nop);
    z.resize(nop);  t.resize(nop);

    // Lambda: read a 1D particle dataset
    auto readPclDataset = [&](const string& varName, double* dest) {
        string dsPath = "/particles/species_" + species_str + "/"
                      + varName + "/" + cycle_str;
        herr_t status = H5LTread_dataset_double(file_id, dsPath.c_str(), dest);
        if (status < 0) {
            H5Fclose(file_id);
            eprintf("ERROR: could not read dataset %s from %s",
                    dsPath.c_str(), name_file.c_str());
        }
    };

    // Position
    readPclDataset("x", &x[0]);
    readPclDataset("y", &y[0]);
    readPclDataset("z", &z[0]);

    // Velocity
    readPclDataset("u", &u[0]);
    readPclDataset("v", &v[0]);
    readPclDataset("w", &w[0]);

    // Charge
    readPclDataset("q", &q[0]);

    // Particle ID — stored as long in HDF5, convert to double
    {
        string dsPath = "/particles/species_" + species_str + "/ID/"
                      + cycle_str;
        long* id_buf = new long[nop];
        herr_t status = H5LTread_dataset(file_id, dsPath.c_str(),
                                         H5T_NATIVE_LONG, id_buf);
        if (status < 0) {
            delete[] id_buf;
            H5Fclose(file_id);
            eprintf("ERROR: could not read dataset %s from %s",
                    dsPath.c_str(), name_file.c_str());
        }
        for (int p = 0; p < nop; p++)
            t[p] = (double)id_buf[p];
        delete[] id_buf;
    }

    H5Fclose(file_id);

#else
    eprintf("Restart requires compiling with USE_ADIOS2 or HDF5 (without NO_HDF5).");
#endif
}
