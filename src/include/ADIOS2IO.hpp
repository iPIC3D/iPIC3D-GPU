#ifndef _ADIOS2_HPP_
#define _ADIOS2_HPP_

#include <string>
#include <vector>
#include <unordered_map>
#include <functional>

#include "ipicfwd.h"
#include "cudaTypeDef.cuh"
#include "particleArrayCUDA.cuh"
#include "VCtopology3D.h"
#include "Grid3DCU.h"
#include "EMfields3D.h"
#include "Collective.h"
#include "ParticleSoAHost.h"

#include "adios2.h"

using std::string;

namespace ADIOS2IO {

using optionFuncType = std::function<void(adios2::IO&, adios2::Engine&)>;

class ADIOS2Manager {

private:
    // adios2
    adios2::ADIOS adios;

    adios2::IO ioField;
    adios2::Engine engineField;
    std::vector <optionFuncType> fieldOptions;

    adios2::IO ioParticle;
    adios2::Engine engineParticle;
    std::vector <optionFuncType> particleOptions;
    
    adios2::IO ioRestart;
    adios2::Engine engineRestart;
    std::vector <optionFuncType> restartOptions;

    std::unordered_map<std::string, optionFuncType> outputTagOptions;

    // general
    int cartisianRank;
    string saveDirName;
    string restartDirName;

    bool open = false; // open flag
    int lastCycle = -1; // last output simulation cycle
    int outputCount = 0; // output times count

    // field
    string fieldTag;

    // particle
    string particleTag;
    int sample;

    // restart
    string restartTag;


    // pointer registration
    Collective *col;
    VCtopology3D *vct;
    Grid3DCU *grid;
    EMfields3D *EMf;
    ParticleSoAHost **part; // now we only copy from the CPU buffer
    // particleArrayCUDA **pclsArrayHostPtr;
    ParticleSoAHost **testpart;
    int ns;
    int nstestpart;
    

public:

    ADIOS2Manager() {
        outputTagOptions = {
            // particle
            {"position", std::bind(&ADIOS2Manager::_particlePosition, this, std::placeholders::_1, std::placeholders::_2)},
            {"velocity", std::bind(&ADIOS2Manager::_particleVelocity, this, std::placeholders::_1, std::placeholders::_2)},
            {"q", std::bind(&ADIOS2Manager::_particleCharge, this, std::placeholders::_1, std::placeholders::_2)},
            {"ID", std::bind(&ADIOS2Manager::_particleID, this, std::placeholders::_1, std::placeholders::_2)},
            // field
            {"proc_topology", std::bind(&ADIOS2Manager::_procTopology, this, std::placeholders::_1, std::placeholders::_2)},
            {"E", std::bind(&ADIOS2Manager::_E, this, std::placeholders::_1, std::placeholders::_2)},
            {"B", std::bind(&ADIOS2Manager::_B, this, std::placeholders::_1, std::placeholders::_2)},
            {"Js", std::bind(&ADIOS2Manager::_Js, this, std::placeholders::_1, std::placeholders::_2)},
            {"rhos", std::bind(&ADIOS2Manager::_rhos, this, std::placeholders::_1, std::placeholders::_2)},
            {"pressure", std::bind(&ADIOS2Manager::_pressure, this, std::placeholders::_1, std::placeholders::_2)},
        };
    }

/**
 * @brief Create or open the ADIOS2 output files and bind solver data sources.
 *
 * @details Registers field and particle sources, configures the output layout,
 *          and opens the files needed by subsequent append calls.
 * @param fieldTag Base tag for field-output variables.
 * @param particleTag Base tag for particle-output variables.
 * @param sample Output sampling stride used by the backend.
 * @param col Solver collective/configuration object.
 * @param vct MPI topology for rank-local domain information.
 * @param grid Local grid descriptor.
 * @param EMf Field container used as the data source.
 * @param outputPart Regular particle species to serialize.
 * @param ns Number of regular particle species.
 * @param testpart Test-particle species to serialize.
 * @param nstestpart Number of test-particle species.
 */
void initOutputFiles(string fieldTag, string particleTag, int sample,
                     Collective* col, VCtopology3D* vct, Grid3DCU* grid,
                     EMfields3D* EMf, ParticleSoAHost** outputPart, int ns,
                     ParticleSoAHost** testpart, int nstestpart);

/**
 * @brief Append the output data to the output files, the interface 
 * 
 * @param cycle simulation cycle
 */
void appendOutput(int cycle);

void closeOutputFiles();

// void loadRestart(iPic3D::c_Solver& KCode);

public:
/**
 * @brief Append one field-output step to the ADIOS2 stream.
 * @param cycle Simulation cycle being written.
 */
void appendFieldOutput(int cycle); 

/**
 * @brief Append one particle-output step to the ADIOS2 stream.
 * @param cycle Simulation cycle being written.
 */
void appendParticleOutput(int cycle);

/**
 * @brief Append one restart/checkpoint step to the ADIOS2 stream.
 * @param cycle Simulation cycle being written.
 */
void appendRestartOutput(int cycle);

private:

/**
 * @brief Create or query an ADIOS2 variable and optionally update its selection.
 * @param io ADIOS2 IO object that owns the variable definition.
 * @param name Variable name.
 * @param shape Global variable shape for array variables.
 * @param start Local starting offset within `shape`.
 * @param count Local extent within `shape`.
 * @param constantDims Whether the variable has constant dimensions across steps.
 * @return ADIOS2 variable handle, newly defined or previously queried.
 */
template < typename T >
adios2::Variable<T> _variableHelper(adios2::IO &io, const std::string &name, const adios2::Dims &shape = adios2::Dims(), 
                                    const adios2::Dims &start = adios2::Dims(), const adios2::Dims &count = adios2::Dims(),
                                    const bool constantDims = false) {
    
    auto var = io.InquireVariable<T>(name);

    if(var){ // variable exists
        if (!shape.empty()) { // is array
            var.SetShape(shape);
            var.SetSelection({start, count});
        }
    } 
    else { // first time define
        var = shape.empty() ? io.DefineVariable<T>(name) : io.DefineVariable<T>(name, shape, start, count, constantDims);
        if (!var) throw std::runtime_error("Failed to define variable: " + name);
        
    }

    return var;            
}

// tag mapping
/* Field
    collective
    total_topology 
    proc_topology
    B --> to write all B components
    E --> to write all E components
    phi --> scalar vector
    Jall --> to write all J (current density) components
    Jsall --> to write all Js (current densities for each species) components
    rho -> net charge density
    rhos -> charge densities for each species
    pressure -> pressure tensor for each species
    k_energy -> kinetic energy for each species
    B_energy -> energy of magnetic field
    E_energy -> energy of electric field
*/

void _procTopology(adios2::IO &io, adios2::Engine &engine){
    int coord[3] = {vct->getCoordinates(0), vct->getCoordinates(1), vct->getCoordinates(2)};
    auto varCoord = _variableHelper<int>(io, "cartesian_coord", {3}, {0}, {3});
    engine.Put<int>(varCoord, coord, adios2::Mode::Sync);

    auto varRank = _variableHelper<int>(io, "cartesian_rank");
    engine.Put<int>(varRank, vct->getCartesian_rank(), adios2::Mode::Sync);

    int xleft = vct->getXleft_neighbor();
    auto varXleft = _variableHelper<int>(io, "Xleft_neighbor");
    engine.Put<int>(varXleft, xleft, adios2::Mode::Sync);

    int xright = vct->getXright_neighbor();
    auto varXright = _variableHelper<int>(io, "Xright_neighbor");
    engine.Put<int>(varXright, xright, adios2::Mode::Sync);

    int yleft = vct->getYleft_neighbor();
    auto varYleft = _variableHelper<int>(io, "Yleft_neighbor");
    engine.Put<int>(varYleft, yleft, adios2::Mode::Sync);

    int yright = vct->getYright_neighbor();
    auto varYright = _variableHelper<int>(io, "Yright_neighbor");
    engine.Put<int>(varYright, yright, adios2::Mode::Sync);

    int zleft = vct->getZleft_neighbor();
    auto varZleft = _variableHelper<int>(io, "Zleft_neighbor");
    engine.Put<int>(varZleft, zleft, adios2::Mode::Sync);

    int zright = vct->getZright_neighbor();
    auto varZright = _variableHelper<int>(io, "Zright_neighbor");
    engine.Put<int>(varZright, zright, adios2::Mode::Sync);
}

// Note that the ghost cells are also included in the output
void _E(adios2::IO &io, adios2::Engine &engine){
    const adios2::Dims shape = {static_cast<unsigned long>(grid->getNXN()), static_cast<unsigned long>(grid->getNYN()), static_cast<unsigned long>(grid->getNZN())};

    auto ex = _variableHelper<cudaCommonType>(io, "Ex", shape, {0, 0, 0}, shape);
    auto ey = _variableHelper<cudaCommonType>(io, "Ey", shape, {0, 0, 0}, shape);
    auto ez = _variableHelper<cudaCommonType>(io, "Ez", shape, {0, 0, 0}, shape);

    engine.Put<cudaCommonType>(ex, EMf->getEx().get_arr(), adios2::Mode::Deferred);
    engine.Put<cudaCommonType>(ey, EMf->getEy().get_arr(), adios2::Mode::Deferred);
    engine.Put<cudaCommonType>(ez, EMf->getEz().get_arr(), adios2::Mode::Deferred);
}

void _B(adios2::IO &io, adios2::Engine &engine){
    const adios2::Dims shape = {static_cast<unsigned long>(grid->getNXN()), static_cast<unsigned long>(grid->getNYN()), static_cast<unsigned long>(grid->getNZN())};

    auto bx = _variableHelper<cudaCommonType>(io, "Bx", shape, {0, 0, 0}, shape);
    auto by = _variableHelper<cudaCommonType>(io, "By", shape, {0, 0, 0}, shape);
    auto bz = _variableHelper<cudaCommonType>(io, "Bz", shape, {0, 0, 0}, shape);

    // Store only the evolved B (Bxn/Byn/Bzn), not B_tot = Bxn + Bx_ext.
    // Bx_ext is recomputed from the analytic expression at init, so storing
    // B_tot would cause Bx_ext to be double-counted on restart.
    engine.Put<cudaCommonType>(bx, EMf->getBx().get_arr(), adios2::Mode::Deferred);
    engine.Put<cudaCommonType>(by, EMf->getBy().get_arr(), adios2::Mode::Deferred);
    engine.Put<cudaCommonType>(bz, EMf->getBz().get_arr(), adios2::Mode::Deferred);
}

void _rhos(adios2::IO &io, adios2::Engine &engine){
    const adios2::Dims shape = {static_cast<unsigned long>(grid->getNXN()), static_cast<unsigned long>(grid->getNYN()), static_cast<unsigned long>(grid->getNZN())};

    for (int i = 0; i < ns; i++) {
        auto var = _variableHelper<cudaCommonType>(io, "rhosSpecies" + std::to_string(i), shape, {0, 0, 0}, shape);
        engine.Put<cudaCommonType>(var, (cudaCommonType*)(EMf->getRHOns()[i]), adios2::Mode::Deferred);
    }
}

void _Js(adios2::IO &io, adios2::Engine &engine){
    const adios2::Dims shape = {static_cast<unsigned long>(grid->getNXN()), static_cast<unsigned long>(grid->getNYN()), static_cast<unsigned long>(grid->getNZN())};

    for (int i = 0; i < ns; i++) {
        auto jx = _variableHelper<cudaCommonType>(io, "JxsSpecies" + std::to_string(i), shape, {0, 0, 0}, shape);
        auto jy = _variableHelper<cudaCommonType>(io, "JysSpecies" + std::to_string(i), shape, {0, 0, 0}, shape);
        auto jz = _variableHelper<cudaCommonType>(io, "JzsSpecies" + std::to_string(i), shape, {0, 0, 0}, shape);

        engine.Put<cudaCommonType>(jx, (cudaCommonType*)(EMf->getJxs()[i]), adios2::Mode::Deferred);
        engine.Put<cudaCommonType>(jy, (cudaCommonType*)(EMf->getJys()[i]), adios2::Mode::Deferred);
        engine.Put<cudaCommonType>(jz, (cudaCommonType*)(EMf->getJzs()[i]), adios2::Mode::Deferred);
    }
}

void _pressure(adios2::IO &io, adios2::Engine &engine){
    const adios2::Dims shape = {static_cast<unsigned long>(grid->getNXN()), static_cast<unsigned long>(grid->getNYN()), static_cast<unsigned long>(grid->getNZN())};

    for (int i = 0; i < ns; i++) {
        auto pxx = _variableHelper<cudaCommonType>(io, "pXXSpecies" + std::to_string(i), shape, {0, 0, 0}, shape);
        auto pxy = _variableHelper<cudaCommonType>(io, "pXYSpecies" + std::to_string(i), shape, {0, 0, 0}, shape);
        auto pxz = _variableHelper<cudaCommonType>(io, "pXZSpecies" + std::to_string(i), shape, {0, 0, 0}, shape);
        auto pyy = _variableHelper<cudaCommonType>(io, "pYYSpecies" + std::to_string(i), shape, {0, 0, 0}, shape);
        auto pyz = _variableHelper<cudaCommonType>(io, "pYZSpecies" + std::to_string(i), shape, {0, 0, 0}, shape);
        auto pzz = _variableHelper<cudaCommonType>(io, "pZZSpecies" + std::to_string(i), shape, {0, 0, 0}, shape);

        engine.Put<cudaCommonType>(pxx, (cudaCommonType*)(EMf->getpXXsn()[i]), adios2::Mode::Deferred);
        engine.Put<cudaCommonType>(pxy, (cudaCommonType*)(EMf->getpXYsn()[i]), adios2::Mode::Deferred);
        engine.Put<cudaCommonType>(pxz, (cudaCommonType*)(EMf->getpXZsn()[i]), adios2::Mode::Deferred);
        engine.Put<cudaCommonType>(pyy, (cudaCommonType*)(EMf->getpYYsn()[i]), adios2::Mode::Deferred);
        engine.Put<cudaCommonType>(pyz, (cudaCommonType*)(EMf->getpYZsn()[i]), adios2::Mode::Deferred);
        engine.Put<cudaCommonType>(pzz, (cudaCommonType*)(EMf->getpZZsn()[i]), adios2::Mode::Deferred);
 
    }
}


/* Particle
    position -> particle position (x,y)
    velocity -> particle velocity (u,v,w)
    q -> particle charge
    ID -> particle ID (note: TrackParticleID has to be set true in Collective)
*/
void _particlePosition(adios2::IO &io, adios2::Engine &engine){
    for (int i = 0; i < ns; i++) {
        const unsigned long sizeNOP = static_cast<unsigned long>(part[i]->getNOP());

        auto x = _variableHelper<cudaCommonType>(io, "part" + std::to_string(i) + "PositionX", {sizeNOP}, {0}, {sizeNOP});
        auto y = _variableHelper<cudaCommonType>(io, "part" + std::to_string(i) + "PositionY", {sizeNOP}, {0}, {sizeNOP});
        auto z = _variableHelper<cudaCommonType>(io, "part" + std::to_string(i) + "PositionZ", {sizeNOP}, {0}, {sizeNOP});

        engine.Put<cudaCommonType>(x, part[i]->getXall(), adios2::Mode::Deferred);
        engine.Put<cudaCommonType>(y, part[i]->getYall(), adios2::Mode::Deferred);
        engine.Put<cudaCommonType>(z, part[i]->getZall(), adios2::Mode::Deferred);
    }
}

void _particleVelocity(adios2::IO &io, adios2::Engine &engine){
    for (int i = 0; i < ns; i++) {
        const unsigned long sizeNOP = static_cast<unsigned long>(part[i]->getNOP());

        auto u = _variableHelper<cudaCommonType>(io, "part" + std::to_string(i) + "VelocityU", {sizeNOP}, {0}, {sizeNOP});
        auto v = _variableHelper<cudaCommonType>(io, "part" + std::to_string(i) + "VelocityV", {sizeNOP}, {0}, {sizeNOP});
        auto w = _variableHelper<cudaCommonType>(io, "part" + std::to_string(i) + "VelocityW", {sizeNOP}, {0}, {sizeNOP});

        engine.Put<cudaCommonType>(u, part[i]->getUall(), adios2::Mode::Deferred);
        engine.Put<cudaCommonType>(v, part[i]->getVall(), adios2::Mode::Deferred);
        engine.Put<cudaCommonType>(w, part[i]->getWall(), adios2::Mode::Deferred);
    }
}

void _particleCharge(adios2::IO &io, adios2::Engine &engine){
    for (int i = 0; i < ns; i++) {
        const unsigned long sizeNOP = static_cast<unsigned long>(part[i]->getNOP());

        auto var = _variableHelper<cudaCommonType>(io, "part" + std::to_string(i) + "charge", {sizeNOP}, {0}, {sizeNOP});

        engine.Put<cudaCommonType>(var, part[i]->getQall(), adios2::Mode::Deferred);
    }
}

void _particleID(adios2::IO &io, adios2::Engine &engine){
    for (int i = 0; i < ns; i++) {
        const unsigned long sizeNOP = static_cast<unsigned long>(part[i]->getNOP());

        auto var = _variableHelper<cudaCommonType>(io, "part" + std::to_string(i) + "ID", {sizeNOP}, {0}, {sizeNOP});

        engine.Put<cudaCommonType>(var, part[i]->getParticleIDall(), adios2::Mode::Deferred);
    }
}

// restart, all of the above


};


}






#endif
