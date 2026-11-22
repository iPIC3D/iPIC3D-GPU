#ifndef ADIOS2_IO_HPP
#define ADIOS2_IO_HPP

#include <cstddef>
#include <functional>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <vector>

#include "Collective.h"
#include "EMfields3D.h"
#include "Grid3DCU.h"
#include "ParticleSample.h"
#include "ParticleSoAHost.h"
#include "RestartParticleCellMetadata.h"
#include "VCtopology3D.h"
#include "cudaTypeDef.cuh"
#include "ipicfwd.h"
#include "particleArrayCUDA.cuh"

#include "adios2.h"

using std::string;

namespace ADIOS2IO {

/** Writes the data of a tag into a step: of the given species, from their
 *  views, for the particle data (the particle output writes those of
 *  ParticlesOutputSpecies, the particles it keeps of each; a restart all the
 *  particles of every species); the other tags write every species
 *  (std::bind discards the arguments). */
using optionFuncType = std::function<void(
    adios2::IO&, adios2::Engine&, const std::vector<int>& species,
    const std::vector<ParticleView>& views)>;

class ADIOS2Manager {

private:
  // adios2
  adios2::ADIOS adios;

  adios2::IO ioField;
  adios2::Engine engineField;
  std::vector<optionFuncType> fieldOptions;

  adios2::IO ioParticle;
  adios2::Engine engineParticle;
  std::vector<optionFuncType> particleOptions;

  adios2::IO ioRestart;
  adios2::Engine engineRestart;
  std::vector<optionFuncType> restartOptions;

  std::unordered_map<std::string, optionFuncType> outputTagOptions;

  // general
  int cartisianRank;
  string saveDirName;
  string restartDirName;

  bool open = false; // open flag

  // field
  string fieldTag;

  // particle
  string particleTag;

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
  Collective* col;
  VCtopology3D* vct;
  Grid3DCU* grid;
  EMfields3D* EMf;
  ParticleSoAHost** part; // host particle buffers
  // particleArrayCUDA **pclsArrayHostPtr;
  ParticleSoAHost** testpart;
  const RestartParticleCellMetadata* restartParticleCellMetadata = nullptr;
  int ns;
  int nstestpart;

public:
  ADIOS2Manager() {
    outputTagOptions = {
        // particle
        {"position", std::bind(&ADIOS2Manager::_particlePosition, this,
                               std::placeholders::_1, std::placeholders::_2,
                               std::placeholders::_3, std::placeholders::_4)},
        {"velocity", std::bind(&ADIOS2Manager::_particleVelocity, this,
                               std::placeholders::_1, std::placeholders::_2,
                               std::placeholders::_3, std::placeholders::_4)},
        {"q", std::bind(&ADIOS2Manager::_particleCharge, this,
                        std::placeholders::_1, std::placeholders::_2,
                        std::placeholders::_3, std::placeholders::_4)},
        {"ID", std::bind(&ADIOS2Manager::_particleID, this,
                         std::placeholders::_1, std::placeholders::_2,
                         std::placeholders::_3, std::placeholders::_4)},
        {"particle_cell_metadata",
         std::bind(&ADIOS2Manager::_particleCellMetadata, this,
                   std::placeholders::_1, std::placeholders::_2)},
        // field
        {"proc_topology",
         std::bind(&ADIOS2Manager::_procTopology, this, std::placeholders::_1,
                   std::placeholders::_2)},
        {"E", std::bind(&ADIOS2Manager::_E, this, std::placeholders::_1,
                        std::placeholders::_2)},
        {"B", std::bind(&ADIOS2Manager::_B, this, std::placeholders::_1,
                        std::placeholders::_2)},
        {"Js", std::bind(&ADIOS2Manager::_Js, this, std::placeholders::_1,
                         std::placeholders::_2)},
        {"rhos", std::bind(&ADIOS2Manager::_rhos, this, std::placeholders::_1,
                           std::placeholders::_2)},
        {"pressure", std::bind(&ADIOS2Manager::_pressure, this,
                               std::placeholders::_1, std::placeholders::_2)},
    };
  }

  /**
   * @brief Create or open the ADIOS2 output files and bind solver data sources.
   *
   * @details Registers field and particle sources, configures the output
   * layout, and opens the files needed by subsequent append calls.
   * @param fieldTag Base tag for field-output variables.
   * @param particleTag Base tag for particle-output variables.
   * @param col Solver collective/configuration object.
   * @param vct MPI topology for rank-local domain information.
   * @param grid Local grid descriptor.
   * @param EMf Field container used as the data source.
   * @param outputPart Regular particle species to serialize.
   * @param ns Number of regular particle species.
   * @param testpart Test-particle species to serialize.
   * @param nstestpart Number of test-particle species.
   */
  void initOutputFiles(string fieldTag, string particleTag, Collective* col,
                       VCtopology3D* vct, Grid3DCU* grid, EMfields3D* EMf,
                       ParticleSoAHost** outputPart, int ns,
                       ParticleSoAHost** testpart, int nstestpart);

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
   * @param views The particles written of each species (of every species,
   * those of ParticlesOutputSpecies read).
   */
  void appendParticleOutput(int cycle, const std::vector<ParticleView>& views);

  /**
   * @brief Write one restart/checkpoint into the selected slot directory.
   * @param cycle Restart label stored in the checkpoint; the loop cycle to
   *              execute first after restart.
   * @param restartDir Directory containing this checkpoint slot's rank files.
   */
  void writeRestartOutput(int cycle, const string& restartDir);

  void
  setRestartParticleCellMetadata(const RestartParticleCellMetadata* metadata) {
    restartParticleCellMetadata = metadata;
  }

private:
  /**
   * @brief Create or query an ADIOS2 variable and optionally update its
   * selection.
   * @param io ADIOS2 IO object that owns the variable definition.
   * @param name Variable name.
   * @param shape Global variable shape for array variables.
   * @param start Local starting offset within `shape`.
   * @param count Local extent within `shape`.
   * @param constantDims Whether the variable has constant dimensions across
   * steps.
   * @return ADIOS2 variable handle, newly defined or previously queried.
   */
  template <typename T>
  adios2::Variable<T>
  _variableHelper(adios2::IO& io, const std::string& name,
                  const adios2::Dims& shape = adios2::Dims(),
                  const adios2::Dims& start = adios2::Dims(),
                  const adios2::Dims& count = adios2::Dims(),
                  const bool constantDims = false) {

    auto var = io.InquireVariable<T>(name);

    if (var) {              // variable exists
      if (!shape.empty()) { // is array
        var.SetShape(shape);
        var.SetSelection({start, count});
      }
    } else { // first time define
      var = shape.empty()
                ? io.DefineVariable<T>(name)
                : io.DefineVariable<T>(name, shape, start, count, constantDims);
      if (!var)
        throw std::runtime_error("Failed to define variable: " + name);
    }

    return var;
  }

  ActiveNodeLayout _makeActiveNodeLayout() const {
    if (!grid) {
      throw std::runtime_error("ADIOS2 restart active-node layout requested "
                               "before grid registration");
    }

    const int nx = grid->getNXN() - 2;
    const int ny = grid->getNYN() - 2;
    const int nz = grid->getNZN() - 2;

    if (nx <= 0 || ny <= 0 || nz <= 0) {
      throw std::runtime_error(
          "ADIOS2 restart active-node layout has non-positive extent");
    }

    return {static_cast<std::size_t>(nx), static_cast<std::size_t>(ny),
            static_cast<std::size_t>(nz)};
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
      throw std::runtime_error(
          "ADIOS2 restart active-node write buffer is not prepared");
    }
    return activeNodeWriteBuffers[slot];
  }

  void _packActiveNodes(arr3_double src,
                        std::vector<cudaCommonType>& dst) const {
    std::size_t index = 0;
    for (std::size_t ix = 0; ix < activeNodeLayout.nx; ++ix) {
      for (std::size_t iy = 0; iy < activeNodeLayout.ny; ++iy) {
        for (std::size_t iz = 0; iz < activeNodeLayout.nz; ++iz) {
          dst[index++] =
              static_cast<cudaCommonType>(src[ix + 1][iy + 1][iz + 1]);
        }
      }
    }
  }

  void _putActiveNodeArray(adios2::IO& io, adios2::Engine& engine,
                           const std::string& name, arr3_double src,
                           std::size_t bufferSlot) {
    auto& buffer = _activeNodeWriteBuffer(bufferSlot);
    _packActiveNodes(src, buffer);

    const adios2::Dims shape = activeNodeLayout.dims();
    auto var =
        _variableHelper<cudaCommonType>(io, name, shape, {0, 0, 0}, shape);
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

  void _procTopology(adios2::IO& io, adios2::Engine& engine) {
    int coord[3] = {vct->getCoordinates(0), vct->getCoordinates(1),
                    vct->getCoordinates(2)};
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

  void _E(adios2::IO& io, adios2::Engine& engine) {
    _prepareActiveNodeWriteBuffers(3);
    _putActiveNodeArray(io, engine, "Ex", EMf->getEx(), 0);
    _putActiveNodeArray(io, engine, "Ey", EMf->getEy(), 1);
    _putActiveNodeArray(io, engine, "Ez", EMf->getEz(), 2);
    engine.PerformPuts();
  }

  void _B(adios2::IO& io, adios2::Engine& engine) {
    // Store only the evolved B (Bxn/Byn/Bzn), not B_tot = Bxn + Bx_ext.
    // Bx_ext is recomputed from the analytic expression at init, so storing
    // B_tot would cause Bx_ext to be double-counted on restart.
    _prepareActiveNodeWriteBuffers(3);
    _putActiveNodeArray(io, engine, "Bx", EMf->getBx(), 0);
    _putActiveNodeArray(io, engine, "By", EMf->getBy(), 1);
    _putActiveNodeArray(io, engine, "Bz", EMf->getBz(), 2);
    engine.PerformPuts();
  }

  void _rhos(adios2::IO& io, adios2::Engine& engine) {
    _prepareActiveNodeWriteBuffers(1);
    for (int i = 0; i < ns; i++) {
      _putActiveNodeArray(io, engine, "rhosSpecies" + std::to_string(i),
                          EMf->getRHOns(i), 0);
      engine.PerformPuts();
    }
  }

  void _Js(adios2::IO& io, adios2::Engine& engine) {
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

  void _pressure(adios2::IO& io, adios2::Engine& engine) {
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
      ID -> particle ID for tracked species
  */
  // RestartReader reads these variables of a checkpoint as cudaCommonType
  static_assert(std::is_same_v<cudaPclType_X, cudaCommonType> &&
                    std::is_same_v<cudaPclType_Y, cudaCommonType> &&
                    std::is_same_v<cudaPclType_Z, cudaCommonType> &&
                    std::is_same_v<cudaPclType_U, cudaCommonType> &&
                    std::is_same_v<cudaPclType_V, cudaCommonType> &&
                    std::is_same_v<cudaPclType_W, cudaCommonType> &&
                    std::is_same_v<cudaPclType_Q, cudaCommonType>,
                "the particle variables have the type the restart reads");

  void _particlePosition(adios2::IO& io, adios2::Engine& engine,
                         const std::vector<int>& species,
                         const std::vector<ParticleView>& views) {
    for (const int i : species) {
      const ParticleView& p = views[i];
      const adios2::Dims n = {p.nop};
      const string name = "part" + std::to_string(i);
      auto x =
          _variableHelper<cudaPclType_X>(io, name + "PositionX", n, {0}, n);
      auto y =
          _variableHelper<cudaPclType_Y>(io, name + "PositionY", n, {0}, n);
      auto z =
          _variableHelper<cudaPclType_Z>(io, name + "PositionZ", n, {0}, n);
      engine.Put<cudaPclType_X>(x, nonNull(p.x, p.nop), adios2::Mode::Deferred);
      engine.Put<cudaPclType_Y>(y, nonNull(p.y, p.nop), adios2::Mode::Deferred);
      engine.Put<cudaPclType_Z>(z, nonNull(p.z, p.nop), adios2::Mode::Deferred);
    }
  }

  void _particleVelocity(adios2::IO& io, adios2::Engine& engine,
                         const std::vector<int>& species,
                         const std::vector<ParticleView>& views) {
    for (const int i : species) {
      const ParticleView& p = views[i];
      const adios2::Dims n = {p.nop};
      const string name = "part" + std::to_string(i);
      auto u =
          _variableHelper<cudaPclType_U>(io, name + "VelocityU", n, {0}, n);
      auto v =
          _variableHelper<cudaPclType_V>(io, name + "VelocityV", n, {0}, n);
      auto w =
          _variableHelper<cudaPclType_W>(io, name + "VelocityW", n, {0}, n);
      engine.Put<cudaPclType_U>(u, nonNull(p.u, p.nop), adios2::Mode::Deferred);
      engine.Put<cudaPclType_V>(v, nonNull(p.v, p.nop), adios2::Mode::Deferred);
      engine.Put<cudaPclType_W>(w, nonNull(p.w, p.nop), adios2::Mode::Deferred);
    }
  }

  void _particleCharge(adios2::IO& io, adios2::Engine& engine,
                       const std::vector<int>& species,
                       const std::vector<ParticleView>& views) {
    for (const int i : species) {
      const ParticleView& p = views[i];
      const adios2::Dims n = {p.nop};
      auto var = _variableHelper<cudaPclType_Q>(
          io, "part" + std::to_string(i) + "charge", n, {0}, n);
      engine.Put<cudaPclType_Q>(var, nonNull(p.q, p.nop),
                                adios2::Mode::Deferred);
    }
  }

  void _particleID(adios2::IO& io, adios2::Engine& engine,
                   const std::vector<int>& species,
                   const std::vector<ParticleView>& views) {
    for (const int i : species) {
      const ParticleView& p = views[i];
      if (!p.hasID)
        continue;
      const adios2::Dims n = {p.nop};
      auto var = _variableHelper<cudaPclType_ID>(
          io, "part" + std::to_string(i) + "ID", n, {0}, n);
      engine.Put<cudaPclType_ID>(var, nonNull(p.id, p.nop),
                                 adios2::Mode::Deferred);
    }
  }

  void _particleCellMetadata(adios2::IO& io, adios2::Engine& engine) {
    if (!restartParticleCellMetadata || !restartParticleCellMetadata->valid) {
      throw std::runtime_error("Restart particle cell metadata was not "
                               "prepared before ADIOS2 restart write");
    }

    auto dims = _variableHelper<int>(io, "activeCellDims", {3}, {0}, {3});
    engine.Put<int>(dims, restartParticleCellMetadata->activeCellDims.data(),
                    adios2::Mode::Sync);

    const unsigned long activeCells = static_cast<unsigned long>(
        restartParticleCellMetadata->activeCellCount());
    const adios2::Dims shape = {activeCells};
    for (int i = 0; i < ns; i++) {
      const auto& speciesMetadata = restartParticleCellMetadata->species[i];

      auto offsets = _variableHelper<int>(
          io, "part" + std::to_string(i) + "CellOffsets", shape, {0}, shape);
      auto counts = _variableHelper<int>(
          io, "part" + std::to_string(i) + "CellCounts", shape, {0}, shape);

      engine.Put<int>(offsets, speciesMetadata.cellOffsets.data(),
                      adios2::Mode::Deferred);
      engine.Put<int>(counts, speciesMetadata.cellCounts.data(),
                      adios2::Mode::Deferred);
    }
    engine.PerformPuts();
  }

  // restart, all of the above
};

} // namespace ADIOS2IO

#endif // ADIOS2_IO_HPP
