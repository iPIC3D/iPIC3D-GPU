#ifndef _ADIOS2_HPP_
#define _ADIOS2_HPP_

#include <string>
#include <vector>
#include <unordered_map>
#include <functional>
#include <cstddef>
#include <stdexcept>

#include "ipicfwd.h"
#include "cudaTypeDef.cuh"
#include "particleArrayCUDA.cuh"
#include "VCtopology3D.h"
#include "Grid3DCU.h"
#include "EMfields3D.h"
#include "Collective.h"
#include "ParticleSoAHost.h"
#include "RestartParticleCellMetadata.h"

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
    int restartWriteCount = 0;

    struct ActiveNodeLayout {
        std::size_t nx = 0;
        std::size_t ny = 0;
        std::size_t nz = 0;

        std::size_t count() const { return nx * ny * nz; }
        adios2::Dims dims() const { return {nx, ny, nz}; }
    };

    ActiveNodeLayout activeNodeLayout;
    std::vector<std::vector<cudaCommonType>> activeNodeWriteBuffers;


    // pointer registration
    Collective *col;
    VCtopology3D *vct;
    Grid3DCU *grid;
    EMfields3D *EMf;
    ParticleSoAHost **part; // now we only copy from the CPU buffer
    // particleArrayCUDA **pclsArrayHostPtr;
    ParticleSoAHost **testpart;
    const RestartParticleCellMetadata* restartParticleCellMetadata = nullptr;
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
            {"particle_cell_metadata", std::bind(&ADIOS2Manager::_particleCellMetadata, this, std::placeholders::_1, std::placeholders::_2)},
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
 * @brief Write one restart/checkpoint into the selected slot directory.
 * @param cycle Restart label stored in the checkpoint; the loop cycle to
 *              execute first after restart.
 * @param restartDir Directory containing this checkpoint slot's rank files.
 */
void writeRestartOutput(int cycle, const string& restartDir);

void setRestartParticleCellMetadata(
    const RestartParticleCellMetadata* metadata) {
    restartParticleCellMetadata = metadata;
}

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

ActiveNodeLayout _makeActiveNodeLayout() const {
    if (!grid) {
        throw std::runtime_error("ADIOS2 restart active-node layout requested before grid registration");
    }

    const int nx = grid->getNXN() - 2;
    const int ny = grid->getNYN() - 2;
    const int nz = grid->getNZN() - 2;

    if (nx <= 0 || ny <= 0 || nz <= 0) {
        throw std::runtime_error("ADIOS2 restart active-node layout has non-positive extent");
    }

    return {
        static_cast<std::size_t>(nx),
        static_cast<std::size_t>(ny),
        static_cast<std::size_t>(nz)
    };
}

void _prepareActiveNodeWriteBuffers(std::size_t slots) {
    activeNodeLayout = _makeActiveNodeLayout();
    if (activeNodeWriteBuffers.size() < slots) {
        activeNodeWriteBuffers.resize(slots);
    }

    const std::size_t count = activeNodeLayout.count();
    for (std::size_t i = 0; i < slots; ++i) {
        activeNodeWriteBuffers[i].resize(count);
    }
}

std::vector<cudaCommonType>& _activeNodeWriteBuffer(std::size_t slot) {
    if (slot >= activeNodeWriteBuffers.size() ||
        activeNodeWriteBuffers[slot].size() != activeNodeLayout.count()) {
        throw std::runtime_error("ADIOS2 restart active-node write buffer is not prepared");
    }
    return activeNodeWriteBuffers[slot];
}

void _packActiveNodes(arr3_double src, std::vector<cudaCommonType>& dst) const {
    std::size_t index = 0;
    for (std::size_t ix = 0; ix < activeNodeLayout.nx; ++ix) {
        for (std::size_t iy = 0; iy < activeNodeLayout.ny; ++iy) {
            for (std::size_t iz = 0; iz < activeNodeLayout.nz; ++iz) {
                dst[index++] = static_cast<cudaCommonType>(src[ix + 1][iy + 1][iz + 1]);
            }
        }
    }
}

void _putActiveNodeArray(adios2::IO &io, adios2::Engine &engine,
                         const std::string &name, arr3_double src,
                         std::size_t bufferSlot) {
    auto &buffer = _activeNodeWriteBuffer(bufferSlot);
    _packActiveNodes(src, buffer);

    const adios2::Dims shape = activeNodeLayout.dims();
    auto var = _variableHelper<cudaCommonType>(io, name, shape, {0, 0, 0}, shape);
    engine.Put<cudaCommonType>(var, buffer.data(), adios2::Mode::Deferred);
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

void _E(adios2::IO &io, adios2::Engine &engine){
    _prepareActiveNodeWriteBuffers(3);
    _putActiveNodeArray(io, engine, "Ex", EMf->getEx(), 0);
    _putActiveNodeArray(io, engine, "Ey", EMf->getEy(), 1);
    _putActiveNodeArray(io, engine, "Ez", EMf->getEz(), 2);
    engine.PerformPuts();
}

void _B(adios2::IO &io, adios2::Engine &engine){
    // Store only the evolved B (Bxn/Byn/Bzn), not B_tot = Bxn + Bx_ext.
    // Bx_ext is recomputed from the analytic expression at init, so storing
    // B_tot would cause Bx_ext to be double-counted on restart.
    _prepareActiveNodeWriteBuffers(3);
    _putActiveNodeArray(io, engine, "Bx", EMf->getBx(), 0);
    _putActiveNodeArray(io, engine, "By", EMf->getBy(), 1);
    _putActiveNodeArray(io, engine, "Bz", EMf->getBz(), 2);
    engine.PerformPuts();
}

void _rhos(adios2::IO &io, adios2::Engine &engine){
    _prepareActiveNodeWriteBuffers(1);
    for (int i = 0; i < ns; i++) {
        _putActiveNodeArray(io, engine, "rhosSpecies" + std::to_string(i),
                            EMf->getRHOns(i), 0);
        engine.PerformPuts();
    }
}

void _Js(adios2::IO &io, adios2::Engine &engine){
    _prepareActiveNodeWriteBuffers(3);
    for (int i = 0; i < ns; i++) {
        _putActiveNodeArray(io, engine, "JxsSpecies" + std::to_string(i),
                            EMf->getJxs(i), 0);
        _putActiveNodeArray(io, engine, "JysSpecies" + std::to_string(i),
                            EMf->getJys(i), 1);
        _putActiveNodeArray(io, engine, "JzsSpecies" + std::to_string(i),
                            EMf->getJzs(i), 2);
        engine.PerformPuts();
    }
}

void _pressure(adios2::IO &io, adios2::Engine &engine){
    _prepareActiveNodeWriteBuffers(6);
    for (int i = 0; i < ns; i++) {
        _putActiveNodeArray(io, engine, "pXXSpecies" + std::to_string(i),
                            EMf->getpXXsn(i), 0);
        _putActiveNodeArray(io, engine, "pXYSpecies" + std::to_string(i),
                            EMf->getpXYsn(i), 1);
        _putActiveNodeArray(io, engine, "pXZSpecies" + std::to_string(i),
                            EMf->getpXZsn(i), 2);
        _putActiveNodeArray(io, engine, "pYYSpecies" + std::to_string(i),
                            EMf->getpYYsn(i), 3);
        _putActiveNodeArray(io, engine, "pYZSpecies" + std::to_string(i),
                            EMf->getpYZsn(i), 4);
        _putActiveNodeArray(io, engine, "pZZSpecies" + std::to_string(i),
                            EMf->getpZZsn(i), 5);
        engine.PerformPuts();
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

void _particleCellMetadata(adios2::IO &io, adios2::Engine &engine) {
    if (!restartParticleCellMetadata || !restartParticleCellMetadata->valid) {
        throw std::runtime_error("Restart particle cell metadata was not prepared before ADIOS2 restart write");
    }

    auto dims = _variableHelper<int>(io, "activeCellDims", {3}, {0}, {3});
    engine.Put<int>(dims, restartParticleCellMetadata->activeCellDims.data(),
                    adios2::Mode::Sync);

    const unsigned long activeCells =
        static_cast<unsigned long>(restartParticleCellMetadata->activeCellCount());
    const adios2::Dims shape = {activeCells};
    for (int i = 0; i < ns; i++) {
        const auto& speciesMetadata = restartParticleCellMetadata->species[i];

        auto offsets = _variableHelper<int>(
            io, "part" + std::to_string(i) + "CellOffsets",
            shape, {0}, shape);
        auto counts = _variableHelper<int>(
            io, "part" + std::to_string(i) + "CellCounts",
            shape, {0}, shape);

        engine.Put<int>(offsets, speciesMetadata.cellOffsets.data(),
                        adios2::Mode::Deferred);
        engine.Put<int>(counts, speciesMetadata.cellCounts.data(),
                        adios2::Mode::Deferred);
    }
    engine.PerformPuts();
}

// restart, all of the above


};


}






#endif
