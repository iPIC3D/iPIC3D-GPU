/* iPIC3D was originally developed by Stefano Markidis and Giovanni Lapenta.
 * This release was contributed by Alec Johnson and Ivy Bo Peng.
 * Publications that use results from iPIC3D need to properly cite
 * 'S. Markidis, G. Lapenta, and Rizwan-uddin. "Multi-scale simulations of
 * plasma with iPIC3D." Mathematics and Computers in Simulation 80.7 (2010):
 * 1509-1519.'
 *
 *        Copyright 2015 KTH Royal Institute of Technology
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *         http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "BenchmarkMode.h"
#include "BuildInfo.h"
#include "Collective.h"
#include "EMfields3D.h"
#include "Grid3DCU.h"
#include "IOManager.h"
#include "MPIdata.h"
#include "Parameters.h"
#include "ParticleArrayConfig.h"
#include "ParticleCommInjection.h"
#include "Profiler.h"
#include "VCtopology3D.h"
#include "debug.h"
#include "iPic3D.h"
#include "ipicdefs.h"
#include "mpi.h"
#include "outputPrepare.h"
#include <iomanip>

#include "GPUFieldPacking.cuh"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <future>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <unistd.h>
#include <vector>

#include "ExosphereIonization.h"

#include "cudaTypeDef.cuh"
#include "injectionKernel.cuh"
#include "momentKernel.cuh"
#include "moverKernel.cuh"
#include "moverPush.cuh"
#include "particleArrayCUDA.cuh"
#include "particleControlKernel.cuh"
#include "particleExchange.cuh"
#include "particleOutputSelect.cuh"
#include "particleSums.cuh"

#if defined(USE_CATALYST) && IPIC3D_COMPILE_DISK_OUTPUT
#include "Adaptor.h"
#endif

// ======= Solver Constants =======
constexpr double AUX_BUFFER_FRACTION =
    0.1; // exiting/filler/staging as fraction of NOP
constexpr double PLANET_BUFFER_FRACTION =
    0.05; // planet array initial fraction of NOP
// Shift per cycle of the parents the particle splitting takes: 2^32 over the
// golden ratio, so that the shifts of successive cycles are far apart.
constexpr uint32_t SPLIT_PARENT_STRIDE = 2654435761u;
constexpr int INITIAL_PLANET_BUF_CAP =
    1024; // initial planet cross-species buffer capacity
// smallest density of the heat flux (the bulk velocity J / rho)
constexpr cudaMomentType HEAT_FLUX_RHO_FLOOR = 1.0e-30;
// Salt of the keys of ParticlesOutputTrackedFraction and
// ParticlesOutputMaxPerCell (particleOutputSelect.cuh),
// times species + 1: the species are chosen independently.
constexpr std::uint64_t PARTICLE_SELECT_SALT = 0x9e3779b97f4a7c15ULL;

using namespace iPic3D;

/**
 * @brief Convert the input-file case string into the internal CaseType enum.
 *
 * This centralizes case parsing so the solver can branch on a stable enum
 * instead of repeatedly comparing strings throughout the runtime path.
 * @param s Case string read from the input configuration.
 * @return Parsed solver case identifier.
 */
static CaseType parseCaseType(const std::string& s) {
  if (s == "GEMnoPert")
    return CaseType::GEMnoPert;
  if (s == "ForceFree")
    return CaseType::ForceFree;
  if (s == "GEM")
    return CaseType::GEM;
  if (s == "GEMDoubleHarris")
    return CaseType::GEMDoubleHarris;
  if (s == "BATSRUS")
    return CaseType::BATSRUS;
  if (s == "Dipole")
    return CaseType::Dipole;
  if (s == "Dipole2D")
    return CaseType::Dipole2D;
  if (s == "NullPoints")
    return CaseType::NullPoints;
  if (s == "TaylorGreen")
    return CaseType::TaylorGreen;
  if (s == "HumpPert")
    return CaseType::HumpPert;
  if (s == "RandomCase")
    return CaseType::RandomCase;
  if (s == "GEMHarris")
    return CaseType::GEMHarris;
  return CaseType::Default;
}

/**
 * @brief Copy the input file into the output folder, so the folder holds the
 *        input next to the build identity of the run. A fresh run keeps the
 *        input's name; a restart adds ".restart<cycle>" so every run segment
 *        keeps its own input. A failure is reported, not fatal.
 * @return The copy's path, or an empty string if nothing was copied.
 */
static std::string copyInputToOutput(const std::string& input,
                                     const std::string& outputDir, bool restart,
                                     int firstCycle) {
  namespace fs = std::filesystem;
  const fs::path src(input);
  fs::path name = src.filename();
  if (restart)
    name = src.stem().string() + ".restart" + std::to_string(firstCycle) +
           src.extension().string();
  const fs::path dst = fs::path(outputDir) / name;
  std::error_code ec;
  if (fs::exists(dst) && fs::equivalent(src, dst, ec))
    return dst.string();
  fs::copy_file(src, dst, fs::copy_options::overwrite_existing, ec);
  if (ec) {
    cerr << "WARNING: could not copy the input file " << input << " to "
         << dst.string() << ": " << ec.message() << endl;
    return std::string();
  }
  return dst.string();
}

/**
 * @brief Build identity (BuildInfo.h) and run environment, as printed at
 *        startup and written at the top of SimulationData.txt.
 */
static std::string provenanceReport(const Collective& col,
                                    const std::string& inputCopy, int nprocs,
                                    int firstCycle) {
  std::ostringstream out;
  out << "---------------------------\n"
      << "-  Build and run          -\n"
      << "---------------------------\n";
  iPic3D::build::describe(out);

  char when[64] = "unknown";
  const std::time_t now = std::time(nullptr);
  std::strftime(when, sizeof(when), "%Y-%m-%d %H:%M:%S %z",
                std::localtime(&now));
  char host[256] = "unknown";
  gethostname(host, sizeof(host) - 1);
  char mpiVersion[MPI_MAX_LIBRARY_VERSION_STRING] = "unknown";
  int mpiVersionLength = 0;
  MPI_Get_library_version(mpiVersion, &mpiVersionLength);
  std::string mpiLibrary(mpiVersion);
  mpiLibrary = mpiLibrary.substr(0, mpiLibrary.find_first_of("\r\n,"));

  out << "Run started    : " << when << "\n"
      << "Host (rank 0)  : " << host << "\n"
      << "MPI ranks      : " << nprocs << " (" << mpiLibrary << ")\n";
  if (iPic3D::build::openmpThreads() > 0)
    out << "OpenMP threads : " << iPic3D::build::openmpThreads()
        << " per rank\n";
  int deviceCount = 0;
  if (cudaGetDeviceCount(&deviceCount) == cudaSuccess && deviceCount > 0) {
    cudaDeviceProp prop;
    int runtime = 0, driver = 0;
    cudaGetDeviceProperties(&prop, 0);
    cudaRuntimeGetVersion(&runtime);
    cudaDriverGetVersion(&driver);
    out << "GPUs (rank 0)  : " << deviceCount << " x " << prop.name
        << ", runtime " << runtime << ", driver " << driver << "\n";
  }
  out << "Input file     : " << col.getinputfile();
  if (!inputCopy.empty())
    out << " (copy: " << inputCopy << ")";
  out << "\n"
      << "Start          : ";
  if (col.getRestart_status() != 0) {
    out << "restart at cycle " << firstCycle << " from "
        << col.getRestartReadDirName() << "\n";
    // slot-selection notes: fallbacks and a newer slot about to be replaced
    std::istringstream notes(col.getRestartSelectionReport());
    for (std::string line; std::getline(notes, line);)
      out << "Restart note   : " << line << "\n";
  } else {
    out << "fresh run\n";
  }
  return out.str();
}
// MPIdata* iPic3D::c_Solver::mpi=0;
c_Solver::c_Solver()
    : col(0), vct(0), grid(0), EMf(0), particlesCommInj(nullptr),
      particlesHost(nullptr), testpart(nullptr), exosphereIonization(nullptr),
      ioManager(0), heatFluxCalculationEnabled_(false),
      heatFluxOutputEnabled_(false), heatFluxScheduledCycle_(-1),
      heatFluxStream(nullptr), heatFluxReadDoneEvt(nullptr),
      heatFluxHostDoneEvt(nullptr), heatFluxCUDAPtr(nullptr),
      heatFluxBulkCUDAPtr(nullptr), Ke(0), BulkEnergy(0), momentum(0), Qtot(0),
      Qremoved(0) {}

/**
 * @brief Destroy the solver and release host-side runtime objects.
 *
 * GPU allocations are released separately in deInitCUDA(). This destructor
 * handles the remaining owning pointers managed by c_Solver.
 */
c_Solver::~c_Solver() {
#if defined(USE_CATALYST) && IPIC3D_COMPILE_DISK_OUTPUT
  Adaptor::Finalize();
#endif

  // I/O backends retain non-owning pointers to fields, particles, topology,
  // grid, and configuration, so release them before their data sources.
  delete ioManager;

  if (particlesCommInj) {
    for (int i = 0; i < ns; i++)
      delete particlesCommInj[i];
    delete[] particlesCommInj;
  }

  if (testpart) {
    for (int i = 0; i < nstestpart; i++)
      delete testpart[i];
    delete[] testpart;
  }

  if (particlesHost) {
    for (int i = 0; i < ns; i++)
      delete particlesHost[i];
    delete[] particlesHost;
  }

  delete exosphereIonization;

  delete[] Ke;
  delete[] BulkEnergy;
  delete[] momentum;
  delete[] Qtot;
  delete[] Qremoved;

  // EMfields3D::freeDataType() dereferences its borrowed configuration, grid,
  // and topology references; keep those owners alive through field teardown.
  delete EMf;
  delete grid;
  delete vct;
  delete col;
}

/**
 * @brief Initialize MPI-side solver state, host particle containers, and GPU
 * data.
 *
 * This method owns the full startup sequence: input parsing, topology and grid
 * construction, field initialization, particle initialization or restart load,
 * I/O backend setup, and final CUDA-side allocation.
 * @param argc Command-line argument count forwarded to `Collective`.
 * @param argv Command-line argument vector forwarded to `Collective`.
 * @return `0` on success, nonzero on initialization failure.
 */
int c_Solver::Init(int argc, char** argv) {
  profiling::Range range(profiling::Region::Init);
  // the phases of the initialisation: each start closes the previous one
  profiling::Phase phase;
  phase.start(profiling::Region::InitInput);
#if defined(__MIC__)
  assert_eq(DVECWIDTH, 8);
#endif
  // initialized MPI environment
  // nprocs = number of processors
  // myrank = rank of tha process*/
  Parameters::init_parameters();
  // mpi = &MPIdata::instance();
  nprocs = MPIdata::get_nprocs();
  myrank = MPIdata::get_rank();

  col = new Collective(argc, argv); // Every proc loads the parameters of
                                    // simulation from class Collective
  restart_cycle = col->getRestartOutputCycle();
  SaveDirName = col->getSaveDirName();
  RestartDirName = col->getRestartDirName();
  restart_status = col->getRestart_status();
  ns =
      col->getNs(); // get the number of particle species involved in simulation
  // Restart labels identify the loop cycle to resume. Periodic checkpoints are
  // written before that cycle completes, so a checkpoint labeled N must restart
  // by executing cycle N again. Fresh runs start at cycle 0.
  first_cycle = (restart_status != 0) ? col->getLast_cycle() : 0;
  // initialize the virtual cartesian topology
  vct = new VCtopology3D(*col);
  // Check if we can map the processes into a matrix ordering defined in
  // Collective.cpp
  const int localTopologyMismatch = nprocs != vct->getNprocs();
  int topologyMismatch = localTopologyMismatch;
#ifndef NO_MPI
  int anyTopologyMismatch = 0;
  MPI_Allreduce(&topologyMismatch, &anyTopologyMismatch, 1, MPI_INT, MPI_MAX,
                MPIdata::get_PicGlobalComm());
  topologyMismatch = anyTopologyMismatch;
#endif
  if (topologyMismatch) {
    if (myrank == 0) {
      if (localTopologyMismatch) {
        cerr << "Error: " << nprocs << " processes cannot be mapped into "
             << vct->getXLEN() << "x" << vct->getYLEN() << "x" << vct->getZLEN()
             << " matrix; change XLEN, YLEN, and ZLEN." << endl;
      } else {
        cerr << "Error: another MPI rank reported inconsistent topology "
                "dimensions; ensure XLEN, YLEN, and ZLEN are identical on "
                "all ranks."
             << endl;
      }
    }
    return 1;
  }
  // We create a new communicator with a 3D virtual Cartesian topology
  vct->setup_vctopology(MPIdata::get_PicGlobalComm());
  {
    stringstream num_proc_ss;
    num_proc_ss << vct->getCartesian_rank();
    num_proc_str = num_proc_ss.str();
  }
#ifdef BATSRUS
  // set index offset for each processor
  col->setGlobalStartIndex(vct);
#endif

  // Print the initial settings to stdout and a file
  if (myrank == 0) {
#if IPIC3D_COMPILE_DISK_OUTPUT
    // Fresh runs clear old output. Restart runs must preserve restart files but
    // still need writable output directories for settings/proc files.
    if (restart_status == 0) {
      checkOutputFolder(SaveDirName);
      if (RestartDirName != SaveDirName)
        checkOutputFolder(RestartDirName);
    } else {
      ensureOutputFolder(SaveDirName);
      if (RestartDirName != SaveDirName)
        ensureOutputFolder(RestartDirName);
    }
#endif

    std::string inputCopy;
#if IPIC3D_COMPILE_DISK_OUTPUT
    inputCopy = copyInputToOutput(col->getinputfile(), SaveDirName,
                                  restart_status != 0, first_cycle);
#endif
    const std::string provenance =
        provenanceReport(*col, inputCopy, nprocs, first_cycle);
    cout << provenance;

    MPIdata::instance().Print();
    vct->Print();
    col->Print();
#if IPIC3D_COMPILE_DISK_OUTPUT
    col->save(provenance);
#else
    cout << "Benchmark mode " << BENCHMARK_MODE
         << ": disk output and output-related device-to-host copies are "
            "disabled at compile time."
         << endl;
#endif
  }
#ifndef NO_MPI
  MPI_Barrier(MPIdata::get_PicGlobalComm());
#endif
  // The GPU of this rank, before the host arrays that are copied to and from
  // it are allocated (EMfields3D, the particles)
  phase.start(profiling::Region::InitSelectDevice);
  if (selectDevice() != 0)
    return 1;
  phase.start(profiling::Region::InitFields);
  // Create the local grid
  grid = new Grid3DCU(col, vct);        // Create the local grid
  EMf = new EMfields3D(col, grid, vct); // Create Electromagnetic Fields Object

  // Resolve case string once and store as enum + derived flags
  caseType_ = parseCaseType(col->getCase());
  doPlanet_ =
      (caseType_ == CaseType::Dipole || caseType_ == CaseType::Dipole2D);

  switch (caseType_) {
  case CaseType::GEMnoPert:
    EMf->initGEMnoPert();
    break;
  case CaseType::ForceFree:
    EMf->initForceFree();
    break;
  case CaseType::GEM:
    EMf->initGEM();
    break;
  case CaseType::GEMDoubleHarris:
    EMf->initGEMDoubleHarris();
    break;
#ifdef BATSRUS
  case CaseType::BATSRUS:
    EMf->initBATSRUS();
    break;
#endif
  case CaseType::Dipole:
    EMf->initDipole();
    break;
  case CaseType::Dipole2D:
    EMf->initDipole2D();
    break;
  case CaseType::NullPoints:
    EMf->initNullPoints();
    break;
  case CaseType::TaylorGreen:
    EMf->initTaylorGreen();
    break;
  case CaseType::HumpPert:
    EMf->initHumpPerturbation();
    break;
  case CaseType::GEMHarris:
    EMf->initGEMHarris();
    break;
  case CaseType::RandomCase:
    EMf->initRandomField();
    if (myrank == 0) {
      cout << "Case is " << col->getCase() << "\n";
      cout << "total # of particle per cell is " << col->getNpcel(0) << "\n";
    }
    break;
  default:
    if (myrank == 0) {
      cout << " =========================================================== "
           << endl;
      cout << " WARNING: The case '" << col->getCase()
           << "' was not recognized. " << endl;
      cout << "          Runing simulation with the default initialization. "
           << endl;
      cout << " =========================================================== "
           << endl;
    }
    EMf->init();
    break;
  }

  // ======= Allocate particlesHost[]: lightweight SoA host mirrors =======
  phase.start(profiling::Region::InitParticles);
  particlesHost = new ParticleSoAHost*[ns];
  for (int i = 0; i < ns; i++) {
    particlesHost[i] = new ParticleSoAHost(i, col, vct, grid);
    particlesHost[i]->setRelativistic(doPlanet_); // moverSubcyclesKernel

    if (col->getRestart_status() != 0) { // restart
      particlesHost[i]->restartLoad();
    }
    // Fresh start: maxwellian methods below call prepareSoAForNOP() internally
  }

  // Initial condition for PARTICLES (skipped when restarting)
  if (restart_status == 0) {
    for (int i = 0; i < ns; i++) {
      switch (caseType_) {
#ifdef BATSRUS
      case CaseType::BATSRUS:
        eprintf("BATSRUS not supported on ParticleSoAHost");
        break;
#endif
      case CaseType::ForceFree:
      case CaseType::NullPoints:
      case CaseType::TaylorGreen:
        particlesHost[i]->maxwellianFromSpeciesCurrents(EMf);
        break;
      case CaseType::GEMDoubleHarris:
        particlesHost[i]->maxwellianDoubleHarris(EMf);
        break;
      case CaseType::HumpPert:
        particlesHost[i]->maxwellianHumpPerturbation(EMf);
        break;
      case CaseType::GEMHarris:
        if (col->getCurrentFromAmpere())
          particlesHost[i]->maxwellianFromSpeciesCurrents(
              EMf, col->getSpatiallyVaryingThermal());
        else
          particlesHost[i]->maxwellian(EMf);
        break;
      default:
        particlesHost[i]->maxwellian(EMf);
        break;
      }
    }
  }

  for (int i = 0; i < ns; i++)
    particlesHost[i]->reserve_remaining_particle_IDs();

  // ======= Allocate test particles, if configured =======
  nstestpart = col->getNsTestPart();

  // ======= Allocate particlesCommInj[]: MPI exchange + injection engine
  // =======
  particlesCommInj = new ParticleCommInjection*[ns];
  for (int i = 0; i < ns; i++) {
    particlesCommInj[i] = new ParticleCommInjection(*particlesHost[i]);
    const auto totalPcl =
        col->getNpcel(i) * grid->getNXN() * grid->getNYN() * grid->getNZN();
    particlesCommInj[i]->reserveCommBuffer(
        static_cast<int>(totalPcl * AUX_BUFFER_FRACTION));
  }

  if (nstestpart > 0) {
    testpart = new ParticleSoAHost*[nstestpart];
    for (int i = 0; i < nstestpart; i++) {
      testpart[i] = new ParticleSoAHost(
          i + ns, col, vct,
          grid); // species id for test particles is increased by ns
      testpart[i]->pitch_angle_energy(EMf);
      testpart[i]->reserve_remaining_particle_IDs();
    }
  }

  // ======= Initialize modular I/O manager =======
  phase.start(profiling::Region::InitOutput);
#if IPIC3D_COMPILE_DISK_OUTPUT
  ioManager = new IOManager;
  if (Parameters::get_doWriteOutput() || restart_cycle > 0 ||
      col->getCallFinalize()) {
    ioManager->init(col, vct, grid, EMf, particlesHost, ns, testpart,
                    nstestpart, first_cycle);
  }

  Ke = new double[ns];
  BulkEnergy = new double[ns];
  momentum = new double[ns];
  Qtot = new double[ns];
  cq = SaveDirName + "/ConservedQuantities.txt";
  // Restarts append to the existing file; fresh runs start a new one.
  if (myrank == 0 && restart_status == 0) {
    ofstream my_file(cq.c_str());
    my_file.close();
  }

  Qremoved = new double[ns];
#endif

  // ======= Exosphere ionization source =======
  numSolarWindSpecies = col->getNumSolarWindSpecies();
  numPlanetarySpecies = ns - numSolarWindSpecies;
  if (col->getEnableExosphereInjection() && numPlanetarySpecies > 0) {
    exosphereIonization = new ExosphereIonization(col, grid, vct);
    exosphereTaskFutures.reserve(numPlanetarySpecies);
  } else {
    exosphereIonization = nullptr;
  }

#if defined(USE_CATALYST) && IPIC3D_COMPILE_DISK_OUTPUT
  Adaptor::Initialize(col, (int)(grid->getXstart() / grid->getDX()),
                      (int)(grid->getYstart() / grid->getDY()),
                      (int)(grid->getZstart() / grid->getDZ()), grid->getNXN(),
                      grid->getNYN(), grid->getNZN(), grid->getDX(),
                      grid->getDY(), grid->getDZ());
#endif

#if IPIC3D_COMPILE_DISK_OUTPUT
  // Create or open the particle number file csv
  pclNumCSV = std::ofstream(SaveDirName + "/particleNum" +
                                std::to_string(myrank) + ".csv",
                            std::ios::app);
  pclNumCSV << "cycle,";
  for (int i = 0; i < ns - 1; i++) {
    pclNumCSV << "species" << i << ",";
  }
  pclNumCSV << "species" << ns - 1 << std::endl;
#endif
  phase.stop();

  if (initCUDA() != 0)
    return 1;
#if IPIC3D_COMPILE_DISK_OUTPUT
  if (Parameters::get_doWriteOutput() || restart_cycle > 0 ||
      col->getCallFinalize())
    ioManager->setRestartParticleCellMetadata(&restartParticleCellMetadata_);
  // the host particle copies and the particle sums that the outputs and the
  // energy diagnostics of the first cycle read
  if (ioManager->needsParticleSync(first_cycle) ||
      col->diagnostics_due(first_cycle)) {
    outputCopyAsync(first_cycle - 1);
    awaitOutputCopy();
  }
#endif

  return 0;
}

/**
 * @brief Select the GPU assigned to this MPI rank.
 *
 * Called before the first page-locked host allocation (the host field and
 * particle arrays): such an allocation starts CUDA on the device that is
 * current then, so a later selection would leave a second context on device
 * 0.
 * @return 0, or 1 when a rank of the run cannot be mapped to a device.
 */
int c_Solver::selectDevice() {
  MPI_Comm sharedComm = MPI_COMM_NULL;
  int sharedRank = 0, sharedSize = 0;
  int deviceCount = 0;
  MPI_Comm_split_type(MPIdata::get_PicGlobalComm(), MPI_COMM_TYPE_SHARED, 0,
                      MPI_INFO_NULL, &sharedComm);
  MPI_Comm_rank(sharedComm, &sharedRank);
  MPI_Comm_size(sharedComm, &sharedSize);

  const cudaError_t countStatus = cudaGetDeviceCount(&deviceCount);
  const char* failureReason = nullptr;
  if (countStatus != cudaSuccess) {
    failureReason = cudaGetErrorString(countStatus);
  } else if (deviceCount <= 0) {
    failureReason = "no visible GPU devices";
  } else if (sharedSize > deviceCount && sharedSize % deviceCount != 0) {
    failureReason =
        "local MPI rank count is not divisible by the GPU device count";
  } else {
    // Use one rank per device when possible; otherwise assign equal-sized
    // contiguous groups of local ranks to each device.
    const int ranksPerDevice =
        sharedSize > deviceCount ? sharedSize / deviceCount : 1;
    cudaDeviceOnNode = sharedRank / ranksPerDevice;
    const cudaError_t setStatus = cudaSetDevice(cudaDeviceOnNode);
    if (setStatus != cudaSuccess)
      failureReason = cudaGetErrorString(setStatus);
  }

  if (failureReason) {
    cerr << "Error: cannot map MPI rank " << MPIdata::get_rank()
         << " to a GPU device (local ranks=" << sharedSize
         << ", visible GPU devices=" << deviceCount << "): " << failureReason
         << "." << endl;
  }

  MPI_Comm_free(&sharedComm);
  const int mappingFailure = failureReason ? 1 : 0;
  int anyMappingFailure = 0;
  MPI_Allreduce(&mappingFailure, &anyMappingFailure, 1, MPI_INT, MPI_MAX,
                MPIdata::get_PicGlobalComm());
  if (anyMappingFailure)
    return 1;
#ifndef NDEBUG
  if (sharedRank == 0)
    cout << "[*]GPU assignment: shared comm size: " << sharedSize
         << " visible GPU devices: " << deviceCount << endl;
#endif
  return 0;
}

/**
 * @brief Allocate and initialize all GPU-side solver resources.
 *
 * This includes stream creation, particle/moment/device metadata allocation,
 * pinned host registrations, sorting buffers, and planet-boundary workspaces.
 */
int c_Solver::initCUDA() {
  profiling::Range range(profiling::Region::InitDevice);
  const bool heatFluxRequested = Parameters::get_doWriteOutput() &&
                                 !col->field_output_is_off() &&
                                 col->getOutputConfig().needsAnyHeatFlux();

  // Mode 0 calculates heat flux only when a real field-output backend consumes
  // it. Mode 1 retains the requested device calculation while compiling out
  // its D2H, ghost-preparation, and write stages. Mode 2 enables neither path.
  heatFluxOutputEnabled_ = false;
  if constexpr (BenchmarkConfig::DISK_OUTPUT_ENABLED) {
    heatFluxOutputEnabled_ =
        heatFluxRequested && ioManager &&
        ioManager->getFieldBackend() != IOManager::FieldBackend::NONE;
  }
  heatFluxCalculationEnabled_ =
      heatFluxOutputEnabled_ ||
      (BenchmarkConfig::MODE == 1 && heatFluxRequested);
  heatFluxScheduledCycle_ = -1;

  // ======= Create per-species streams and async launcher state =======
  streams = new cudaStream_t[ns * 2];
  stayedParticle = new int[ns];
  exitingResults = new std::future<int>[ns];
  for (int i = 0; i < ns; i++) {
    cudaErrChk(cudaStreamCreate(streams + i));
    cudaErrChk(cudaStreamCreate(streams + i + ns));
    stayedParticle[i] = 0;
  }
  cudaErrChk(cudaStreamCreate(&planetStream));
  // Dedicated stream for the output D->H copies and the particle sums of the
  // energy diagnostics.  Isolating it from the per-species streams[] removes
  // implicit ordering between the output copy and the next cycle's mover/sort
  // kernels, which is re-established explicitly via
  // cycleEndEvent/eventOutputCopy below.
  cudaErrChk(cudaStreamCreate(&outputStream));
  cudaErrChk(cudaMalloc(&particleSumsCUDAPtr, ns * sizeof(ParticleSums)));
  cudaErrChk(cudaMallocHost(&particleSumsHostPtr, ns * sizeof(ParticleSums)));
  cudaErrChk(cudaMalloc(&particleSampleCountCUDAPtr, ns * sizeof(int)));
  cudaErrChk(cudaMallocHost(&particleSampleCountHostPtr, ns * sizeof(int)));
  particleFractionBlockSums.resize(ns);
  // The heat flux overlaps the field solve: below it, ordered by events only
  if (heatFluxCalculationEnabled_)
    heatFluxStream = createPriorityStream(StreamPriority::Diagnostic);

  for (int i = 0; i < ns; i++)
    particlesHost[i]->initializeParticleIDDeviceCounter(streams[i]);

  {
    // ======= Allocate device-resident particle containers and staging buffers
    // =======
    pclsArrayHostPtr = new particleArrayCUDA*[ns];
    pclsArrayCUDAPtr = new particleArrayCUDA*[ns];
    departureArrayHostPtr = new departureArrayType*[ns];
    departureArrayCUDAPtr = new departureArrayType*[ns];

    hashedSumArrayHostPtr = new hashedSum*[ns];
    hashedSumArrayCUDAPtr = new hashedSum*[ns];
    exitingArrayHostPtr = new exitingArray*[ns];
    exitingArrayCUDAPtr = new exitingArray*[ns];
    fillerBufferArrayHostPtr = new fillerBuffer*[ns];
    fillerBufferArrayCUDAPtr = new fillerBuffer*[ns];
    incomingStagingHostPtr = new arrayCUDA<SpeciesParticle>*[ns];
    incomingStagingCUDAPtr = new arrayCUDA<SpeciesParticle>*[ns];

    for (int i = 0; i < ns; i++) {
      // particleArrayCUDA performs the initial SoA H2D copy in its constructor
      pclsArrayHostPtr[i] = newHostPinnedObject<particleArrayCUDA>(
          particlesHost[i], ParticleArrayConfig::HEADROOM, streams[i]);
      pclsArrayCUDAPtr[i] = pclsArrayHostPtr[i]->copyToDevice();

      departureArrayHostPtr[i] = newHostPinnedObject<departureArrayType>(
          pclsArrayHostPtr[i]->getSize()); // same length
      departureArrayCUDAPtr[i] = departureArrayHostPtr[i]->copyToDevice();
      cudaErrChk(cudaMemsetAsync(departureArrayHostPtr[i]->getArray(), 0,
                                 departureArrayHostPtr[i]->getSize() *
                                     sizeof(departureArrayElementType),
                                 streams[i]));

      // hashedSumArrayHostPtr[i] = new hashedSum[8]{ //
      //   hashedSum(5), hashedSum(5), hashedSum(5), hashedSum(5),
      //   hashedSum(5), hashedSum(5), hashedSum(10), hashedSum(10)
      // };

      hashedSumArrayHostPtr[i] = newHostPinnedObjectArray<hashedSum>(
          departureArrayElementType::HASHED_SUM_NUM, 10);

      hashedSumArrayCUDAPtr[i] = copyArrayToDevice(
          hashedSumArrayHostPtr[i], departureArrayElementType::HASHED_SUM_NUM);

      exitingArrayHostPtr[i] = newHostPinnedObject<exitingArray>(
          AUX_BUFFER_FRACTION * pclsArrayHostPtr[i]->getNOP());
      exitingArrayCUDAPtr[i] = exitingArrayHostPtr[i]->copyToDevice();
      fillerBufferArrayHostPtr[i] = newHostPinnedObject<fillerBuffer>(
          AUX_BUFFER_FRACTION * pclsArrayHostPtr[i]->getNOP());
      fillerBufferArrayCUDAPtr[i] = fillerBufferArrayHostPtr[i]->copyToDevice();

      // AoS staging buffer for incoming H2D particle transfers
      // (MPI exchange, repopulated particles, and exosphere injection).
      // Sized from the initial NOP and expanded on demand.
      incomingStagingHostPtr[i] =
          newHostPinnedObject<arrayCUDA<SpeciesParticle>>(static_cast<uint32_t>(
              AUX_BUFFER_FRACTION * pclsArrayHostPtr[i]->getNOP()));
      incomingStagingCUDAPtr[i] = incomingStagingHostPtr[i]->copyToDevice();
    }
  }

  // ======= Kernel parameters =======
  // The grid, and per species the mover and injection parameters.
  gridCUDA = grid3DCUDA(grid);
  moverCountersHostPtr = newHostPinnedObjectArray<MoverCounters>(ns);
  moverCountersCUDAPtr = copyArrayToDevice(moverCountersHostPtr, ns);
  for (int i = 0; i < ns; i++) {
    moverParameter& mover = moverParams.emplace_back(
        particlesHost[i], pclsArrayCUDAPtr[i], departureArrayCUDAPtr[i],
        hashedSumArrayCUDAPtr[i], fillerBufferArrayCUDAPtr[i],
        moverCountersCUDAPtr + i);
    mover.reflectTol = kWallTolerance *
                       std::min({grid->getDX(), grid->getDY(), grid->getDZ()});
    if (doPlanet_) {
      const bool sphere2D = caseType_ == CaseType::Dipole2D; // the XZ plane
      mover.doSphere = sphere2D ? 2 : 1;
      mover.sphereOrigin[0] = col->getx_center_planet();
      mover.sphereOrigin[1] = sphere2D ? 0.0 : col->gety_center_planet();
      mover.sphereOrigin[2] = col->getz_center_planet();
      mover.sphereRadius = col->getPlanet_radius();
    }

    injectionParams.push_back(particlesCommInj[i]->makeInjectionParameter());
  }

  // ======= Allocate device moment buffers =======
  auto gridSize = grid->getNXN() * grid->getNYN() * grid->getNZN();
  momentsCUDAPtr = new cudaTypeArray1<cudaMomentType>[ns];
  // for(int i=0; i<ns; i++)cudaMallocAsync(&(momentsCUDAPtr[i]),
  // gridSize*10*sizeof(cudaMomentType), streams[i]);
  for (int i = 0; i < ns; i++)
    cudaErrChk(cudaMalloc(&(momentsCUDAPtr[i]),
                          gridSize * 10 * sizeof(cudaMomentType)));

  heatFluxCUDAPtr = nullptr;
  heatFluxBulkCUDAPtr = nullptr;
  if (heatFluxCalculationEnabled_) {
    heatFluxCUDAPtr = new cudaTypeArray1<cudaMomentType>[ns];
    heatFluxBulkCUDAPtr = new cudaTypeArray1<cudaMomentType>[ns];
    for (int i = 0; i < ns; i++) {
      cudaErrChk(cudaMalloc(&(heatFluxCUDAPtr[i]),
                            gridSize * HeatFlux::ComponentCount *
                                sizeof(cudaMomentType)));
      cudaErrChk(cudaMalloc(&(heatFluxBulkCUDAPtr[i]),
                            gridSize * 4 * sizeof(cudaMomentType)));
    }
  }

  // cudaMallocAsync(&fieldForPclCUDAPtr, gridSize*8*sizeof(cudaCommonType), 0);

  const int fieldSize =
      grid->getNZN() * (grid->getNYN() - 1) * (grid->getNXN() - 1);

  // cudaMallocAsync(&fieldForPclCUDAPtr, fieldSize * 24 *
  // sizeof(cudaFieldType), 0);
  cudaErrChk(
      cudaMalloc(&fieldForPclCUDAPtr, fieldSize * 24 * sizeof(cudaFieldType)));

  // Clear any stale CUDA error before GPU solver init
  cudaGetLastError();

  // Allocate GPU solver field arrays and perform initial H2D sync.
  EMf->gpuSolverAllocate();
  EMf->gpuSolverSyncH2D(EMf->gpuSolverStream());
  cudaErrChk(
      cudaEventCreateWithFlags(&solverDoneEvent, cudaEventDisableTiming));

  // momentsPipelineDoneEvt is recorded at the end of every MomentsAwait() GPU
  // pipeline (after gpuCalculateHatFunctions). heatFluxStream waits on it
  // before the D2D copy of d_rhons/d_Jxs into heatFluxBulkCUDAPtr, ensuring
  // current-cycle values with ghost cells filled are used.
  // Pre-record here (on the now-synchronized solverStream_) so that
  // ScheduleHeatFlux on cycle 0 — which runs after the pre-loop MomentsAwait()
  // has already re-recorded the event — never observes an uninitialised handle.
  if (heatFluxCalculationEnabled_) {
    cudaErrChk(cudaEventCreateWithFlags(&momentsPipelineDoneEvt,
                                        cudaEventDisableTiming));
    cudaErrChk(cudaEventRecord(momentsPipelineDoneEvt, EMf->gpuSolverStream()));
  }

  // the host heat flux of the output: ScheduleHeatFlux copies it from the
  // device
  if (heatFluxOutputEnabled_)
    EMf->allocateHeatFlux();

  threadPoolPtr = new ThreadPool(ns, "iPIC3D mover");
  cudaErrChk(cudaEventCreateWithFlags(&event0, cudaEventDisableTiming));
  cudaErrChk(cudaEventCreateWithFlags(
      &eventOutputCopy, cudaEventDisableTiming | cudaEventBlockingSync));
  // Per-species persistent events. cycleEndEvent[i] is recorded at the end
  // of MoverAwaitAndPclExchange on streams[i]. moverHashedReadyEvt[i]
  // (mover wrote hashedSums) and auxHashedReadyEvt[i] (exitingKernel done)
  // are the per-species events of cudaLauncherAsync.
  cycleEndEvent = new cudaEvent_t[ns];
  moverHashedReadyEvt = new cudaEvent_t[ns];
  auxHashedReadyEvt = new cudaEvent_t[ns];
  stayedMomentsDoneEvt = new cudaEvent_t[ns];
  heatFluxReadDoneEvt =
      heatFluxCalculationEnabled_ ? new cudaEvent_t[ns] : nullptr;
  heatFluxHostDoneEvt = nullptr;
#if IPIC3D_COMPILE_DISK_OUTPUT
  if (heatFluxOutputEnabled_)
    heatFluxHostDoneEvt = new cudaEvent_t[ns];
#endif
  for (int i = 0; i < ns; i++) {
    cudaErrChk(cudaEventCreateWithFlags(
        &cycleEndEvent[i], cudaEventDisableTiming | cudaEventBlockingSync));
    cudaErrChk(cudaEventCreateWithFlags(&moverHashedReadyEvt[i],
                                        cudaEventDisableTiming));
    cudaErrChk(cudaEventCreateWithFlags(&auxHashedReadyEvt[i],
                                        cudaEventDisableTiming));
    cudaErrChk(cudaEventCreateWithFlags(&stayedMomentsDoneEvt[i],
                                        cudaEventDisableTiming));
    if (heatFluxCalculationEnabled_) {
      cudaErrChk(cudaEventCreateWithFlags(&heatFluxReadDoneEvt[i],
                                          cudaEventDisableTiming));
      cudaErrChk(cudaEventRecord(heatFluxReadDoneEvt[i], heatFluxStream));
    }
#if IPIC3D_COMPILE_DISK_OUTPUT
    if (heatFluxOutputEnabled_) {
      cudaErrChk(cudaEventCreateWithFlags(&heatFluxHostDoneEvt[i],
                                          cudaEventDisableTiming |
                                              cudaEventBlockingSync));
      cudaErrChk(cudaEventRecord(heatFluxHostDoneEvt[i], heatFluxStream));
    }
#endif
    // Pre-record cycleEndEvent[i] on streams[i] so MomentsAwait() / output-
    // CopyAsync() are well-defined even if invoked before any moment pipeline
    // has run. Both real producers (CalculateMoments and MoverAwaitAndPcl-
    // Exchange) re-record this event after their moment scatter so subsequent
    // waits correctly observe the latest cycle's moments.
    cudaErrChk(cudaEventRecord(cycleEndEvent[i], streams[i]));
  }
  // Pre-record eventOutputCopy on outputStream once, so that the very first
  // cycle's cudaStreamWaitEvent on it (in cudaLauncherAsync / sortSpecies)
  // is well-defined and resolves immediately. This eliminates the need for a
  // "first-cycle" guard flag; Init() schedules and drains an explicit first-
  // cycle copy later when configured outputs need host particle data.
  cudaErrChk(cudaEventRecord(eventOutputCopy, outputStream));

  initParticleControl();

  // ======= Initialize per-species cell sorters =======
  sortingCycle_ = col->getSortingCycle();
  sortThisCycle_ = false;
  cellSorters = new CellSorter[ns];
  if (sortingCycle_ > 0) {
    for (int i = 0; i < ns; i++) {
      cellSorters[i].init(gridCUDA, pclsArrayHostPtr[i]->getCapacity(),
                          streams[i]);
    }
  }

  // ======= Allocate planet quasi-neutral boundary-condition buffers =======
  {
    planetArrayHostPtr = new planetArray*[ns];
    planetArrayCUDAPtr = new planetArray*[ns];
    planetPclCount = new int[ns];

    for (int i = 0; i < ns; i++) {
      if (doPlanet_) {
        planetArrayHostPtr[i] = newHostPinnedObject<planetArray>(
            (uint32_t)(PLANET_BUFFER_FRACTION * pclsArrayHostPtr[i]->getNOP()));
        planetArrayCUDAPtr[i] = planetArrayHostPtr[i]->copyToDevice();
      } else {
        planetArrayHostPtr[i] = nullptr;
        planetArrayCUDAPtr[i] = nullptr;
      }
      planetPclCount[i] = 0;
    }

    // Build the compact map from electron-subset index to global species index.
    planetElecSpeciesCount = 0;
    for (int i = 0; i < ns; i++)
      if (col->getQOM(i) < 0)
        planetElecSpeciesCount++;

    planetElecSpeciesMap = new int[planetElecSpeciesCount];
    {
      int idx = 0;
      for (int i = 0; i < ns; i++)
        if (col->getQOM(i) < 0)
          planetElecSpeciesMap[idx++] = i;
    }

    // Cross-species device buffers.
    planetBufCapacity = doPlanet_ ? INITIAL_PLANET_BUF_CAP : 0;
    if (doPlanet_) {
      cudaErrChk(cudaMalloc(&planetEnergyBuf,
                            planetBufCapacity * sizeof(cudaParticleType)));
      cudaErrChk(cudaMalloc(&planetGlobalIdxBuf,
                            planetBufCapacity * sizeof(uint32_t)));
      cudaErrChk(cudaMalloc(
          &planetWarpChargeBuf,
          planetWarpChargeCount(planetBufCapacity, planetElecSpeciesCount) *
              sizeof(cudaParticleType)));
      cudaErrChk(cudaMalloc(&planetBCStateDevice, sizeof(PlanetBCState)));
      cudaErrChk(cudaMemset(planetBCStateDevice, 0, sizeof(PlanetBCState)));
      cudaErrChk(
          cudaEventCreateWithFlags(&planetDoneEvent, cudaEventDisableTiming));
      cudaErrChk(cudaMalloc(&planetCutoffDevice, sizeof(int)));

      // Device array of per-electron-species planetArray device pointers.
      cudaErrChk(cudaMalloc(&planetArrayCUDAPtrDevice,
                            planetElecSpeciesCount * sizeof(planetArray*)));
      PinnedVector<planetArray*> tmpPtrs(planetElecSpeciesCount);
      for (int e = 0; e < planetElecSpeciesCount; e++)
        tmpPtrs[e] = planetArrayCUDAPtr[planetElecSpeciesMap[e]];
      cudaErrChk(cudaMemcpy(planetArrayCUDAPtrDevice, tmpPtrs.data(),
                            planetElecSpeciesCount * sizeof(planetArray*),
                            cudaMemcpyHostToDevice));

      cudaErrChk(cudaMalloc(&planetElecOffsetsDevice,
                            planetElecSpeciesCount * sizeof(int)));
    } else {
      planetEnergyBuf = nullptr;
      planetGlobalIdxBuf = nullptr;
      planetWarpChargeBuf = nullptr;
      planetBCStateDevice = nullptr;
      planetCutoffDevice = nullptr;
      planetArrayCUDAPtrDevice = nullptr;
      planetElecOffsetsDevice = nullptr;
    }

    // Persistent buffers reused by processPlanetParticles().
    const int elecCount =
        planetElecSpeciesCount > 0 ? planetElecSpeciesCount : 1;
    cudaErrChk(cudaHostAlloc(&planetElecOffsets, elecCount * sizeof(int),
                             cudaHostAllocDefault));
    cudaErrChk(cudaHostAlloc(&planetSurvivorCount, elecCount * sizeof(int),
                             cudaHostAllocDefault));

    if (doPlanet_) {
      cudaErrChk(
          cudaMalloc(&planetSurvivorCountDevice, elecCount * sizeof(int)));
      planetReflectedBufCapacity = INITIAL_PLANET_BUF_CAP;
      cudaErrChk(cudaMalloc(&planetReflectedBuf, planetReflectedBufCapacity *
                                                     sizeof(SpeciesParticle)));
    } else {
      planetSurvivorCountDevice = nullptr;
      planetReflectedBuf = nullptr;
      planetReflectedBufCapacity = 0;
    }
    planetRngCycleCounter = 0;

    // Planet BC diagnostics: accumulated on the device by the planet kernels
    // and written every planetDiagCycle cycles by WritePlanetDiagnostics().
#if IPIC3D_COMPILE_DISK_OUTPUT
    planetDiagCycle =
        doPlanet_ ? std::max(col->getPlanetDiagnosticsCycle(), 0) : 0;
#else
    planetDiagCycle = 0;
#endif
    planetDiagChargeDevice = nullptr;
    planetDiagChargeHost = nullptr;
    planetBCStateHost = nullptr;
    planetDiagChargePrev.assign(ns + planetElecSpeciesCount, 0.0);
    planetBCStatePrev = PlanetBCState{};
    planetDiagHitCount.assign(ns, 0.0);
    planetDiagAbsorbedCount.assign(ns, 0.0);
    planetDiagRankCycles = 0.0;
    planetDiagCyclesAccumulated = 0;
    if (planetDiagCycle > 0) {
      const int nCharge = ns + planetElecSpeciesCount;
      cudaErrChk(cudaMalloc(&planetDiagChargeDevice,
                            nCharge * sizeof(cudaParticleType)));
      cudaErrChk(cudaMemset(planetDiagChargeDevice, 0,
                            nCharge * sizeof(cudaParticleType)));
      cudaErrChk(cudaHostAlloc(&planetDiagChargeHost,
                               nCharge * sizeof(cudaParticleType),
                               cudaHostAllocDefault));
      cudaErrChk(cudaHostAlloc(&planetBCStateHost, sizeof(PlanetBCState),
                               cudaHostAllocDefault));

      planetDiagFile = SaveDirName + "/PlanetBC.txt";
      if (myrank == 0) {
        // Restarts append to an existing file; fresh runs start a new one.
        std::ifstream existing(planetDiagFile);
        const bool appendToExisting =
            restart_status != 0 &&
            existing.peek() != std::ifstream::traits_type::eof();
        existing.close();
        if (!appendToExisting) {
          const int cw = 16; // column width
          ofstream my_file(planetDiagFile.c_str());
          my_file << std::left << std::setw(8) << "Cycle" << std::setw(10)
                  << "NCycles";
          for (const char* name : {"Nhit", "Nabs", "Qhit", "Qabs"})
            for (int is = 0; is < ns; is++)
              my_file << std::setw(cw)
                      << (std::string(name) + "[" + std::to_string(is) + "]");
          my_file << std::setw(cw) << "Qres_sum" << std::setw(cw) << "Qres_max"
                  << std::setw(cw) << "Qres_min" << std::setw(cw) << "Ecut_mean"
                  << std::setw(cw) << "RankCycles" << std::setw(cw)
                  << "ElecLimited" << endl;
        }
      }
    }
  }

  cudaErrChk(cudaDeviceSynchronize());

  if (MPIdata::get_rank() == 0)
    std::cout << "CUDA Init finished" << std::endl;

  return 0;
}

/**
 * @brief Release all CUDA-side resources owned by the solver.
 *
 * This mirrors initCUDA(): streams, device descriptors, pinned registrations,
 * sort buffers, and planet-boundary workspaces are all torn down here.
 */
int c_Solver::deInitCUDA() {

  cudaEventDestroy(event0);
  cudaEventDestroy(eventOutputCopy);
  for (int i = 0; i < ns; i++) {
    cudaEventDestroy(cycleEndEvent[i]);
    cudaEventDestroy(moverHashedReadyEvt[i]);
    cudaEventDestroy(auxHashedReadyEvt[i]);
    cudaEventDestroy(stayedMomentsDoneEvt[i]);
    if (heatFluxCalculationEnabled_)
      cudaEventDestroy(heatFluxReadDoneEvt[i]);
#if IPIC3D_COMPILE_DISK_OUTPUT
    if (heatFluxOutputEnabled_)
      cudaEventDestroy(heatFluxHostDoneEvt[i]);
#endif
  }
  delete[] cycleEndEvent;
  delete[] moverHashedReadyEvt;
  delete[] auxHashedReadyEvt;
  delete[] stayedMomentsDoneEvt;
  cudaEventDestroy(solverDoneEvent);
  if (heatFluxCalculationEnabled_)
    cudaEventDestroy(momentsPipelineDoneEvt);
  delete[] heatFluxReadDoneEvt;
  delete[] heatFluxHostDoneEvt;

  delete threadPoolPtr;

  cudaFree(fieldForPclCUDAPtr);

  // ======= Release per-species host/device objects =======
  for (int i = 0; i < ns; i++) {

    // Destroy host-pinned wrapper objects.

    deleteHostPinnedObject(pclsArrayHostPtr[i]);
    deleteHostPinnedObject(departureArrayHostPtr[i]);
    deleteHostPinnedObjectArray(hashedSumArrayHostPtr[i],
                                departureArrayElementType::HASHED_SUM_NUM);
    deleteHostPinnedObject(exitingArrayHostPtr[i]);
    deleteHostPinnedObject(fillerBufferArrayHostPtr[i]);
    deleteHostPinnedObject(incomingStagingHostPtr[i]);

    // Release device-side object storage.

    cudaFree(pclsArrayCUDAPtr[i]);
    cudaFree(departureArrayCUDAPtr[i]);
    cudaFree(hashedSumArrayCUDAPtr[i]);
    cudaFree(exitingArrayCUDAPtr[i]);
    cudaFree(fillerBufferArrayCUDAPtr[i]);
    cudaFree(incomingStagingCUDAPtr[i]);

    cudaFree(momentsCUDAPtr[i]);
    if (heatFluxCalculationEnabled_) {
      cudaFree(heatFluxCUDAPtr[i]);
      cudaFree(heatFluxBulkCUDAPtr[i]);
    }
  }

  // ======= Release pointer arrays and bookkeeping =======
  delete[] pclsArrayHostPtr;
  delete[] pclsArrayCUDAPtr;
  delete[] departureArrayHostPtr;
  delete[] departureArrayCUDAPtr;
  delete[] hashedSumArrayHostPtr;
  delete[] hashedSumArrayCUDAPtr;
  delete[] exitingArrayHostPtr;
  delete[] exitingArrayCUDAPtr;
  delete[] fillerBufferArrayHostPtr;
  delete[] fillerBufferArrayCUDAPtr;
  delete[] incomingStagingHostPtr;
  delete[] incomingStagingCUDAPtr;
  deleteHostPinnedObjectArray(moverCountersHostPtr, ns);
  cudaFree(moverCountersCUDAPtr);
  moverParams.clear(); // they point to the device buffers freed above
  injectionParams.clear();
  delete[] momentsCUDAPtr;
  delete[] heatFluxCUDAPtr;
  delete[] heatFluxBulkCUDAPtr;

  // ======= Release planet quasi-neutral boundary-condition buffers =======
  for (int i = 0; i < ns; i++) {
    if (planetArrayHostPtr[i])
      deleteHostPinnedObject(planetArrayHostPtr[i]);
    if (planetArrayCUDAPtr[i])
      cudaFree(planetArrayCUDAPtr[i]);
  }
  delete[] planetArrayHostPtr;
  delete[] planetArrayCUDAPtr;
  delete[] planetPclCount;
  delete[] planetElecSpeciesMap;
  if (planetEnergyBuf)
    cudaFree(planetEnergyBuf);
  if (planetGlobalIdxBuf)
    cudaFree(planetGlobalIdxBuf);
  if (planetWarpChargeBuf)
    cudaFree(planetWarpChargeBuf);
  if (planetBCStateDevice)
    cudaFree(planetBCStateDevice);
  if (doPlanet_)
    cudaEventDestroy(planetDoneEvent);
  if (planetDiagChargeDevice)
    cudaFree(planetDiagChargeDevice);
  if (planetDiagChargeHost)
    cudaFreeHost(planetDiagChargeHost);
  if (planetBCStateHost)
    cudaFreeHost(planetBCStateHost);
  if (planetCutoffDevice)
    cudaFree(planetCutoffDevice);
  if (planetArrayCUDAPtrDevice)
    cudaFree(planetArrayCUDAPtrDevice);
  if (planetElecOffsetsDevice)
    cudaFree(planetElecOffsetsDevice);
  cudaFreeHost(planetElecOffsets);
  cudaFreeHost(planetSurvivorCount);
  if (planetSurvivorCountDevice)
    cudaFree(planetSurvivorCountDevice);
  if (planetReflectedBuf)
    cudaFree(planetReflectedBuf);

  // ======= Release cell sorters =======
  for (int i = 0; i < ns; i++)
    cellSorters[i].free();
  delete[] cellSorters;

  // ======= Destroy streams =======
  for (int i = 0; i < ns * 2; i++)
    cudaStreamDestroy(streams[i]);
  cudaStreamDestroy(planetStream);
  cudaStreamDestroy(outputStream);
  cudaErrChk(cudaFree(particleSumsCUDAPtr));
  cudaErrChk(cudaFreeHost(particleSumsHostPtr));
  cudaErrChk(cudaFree(particleSampleCountCUDAPtr));
  cudaErrChk(cudaFreeHost(particleSampleCountHostPtr));
  for (GrowingDeviceArray<int>& blockSums : particleFractionBlockSums)
    blockSums.release();
  particleFractionIdx.release();
  particleFractionStage.release();
  if (heatFluxCalculationEnabled_)
    cudaStreamDestroy(heatFluxStream);
  delete[] streams;
  delete[] stayedParticle;
  delete[] exitingResults;

  return 0;
}

/**
 * @brief Fill heatFluxBulkCUDAPtr[species] from the field solver's moment
 * arrays (D2D).
 *
 * Reads d_rhons/d_Jxs/d_Jys/d_Jzs on the device. The caller MUST ensure
 * heatFluxStream has already waited on momentsPipelineDoneEvt so that the
 * ghost-exchange and hat-function pipeline on solverStream_ has completed.
 *
 * Layout of heatFluxBulkCUDAPtr[s]:
 *   [0*gridSize .. 1*gridSize)  →  rhons[s]   (nxn*nyn*nzn doubles)
 *   [1*gridSize .. 2*gridSize)  →  Jxs[s]
 *   [2*gridSize .. 3*gridSize)  →  Jys[s]
 *   [3*gridSize .. 4*gridSize)  →  Jzs[s]
 *
 * d_rhons.speciesPtr(s) = d_ptr_ + s*nxn*nyn*nzn  (same flat layout).
 * cudaSolverType == cudaMomentType == double, so no conversion is needed.
 *
 * @param species  Species index.
 * @param stream   CUDA stream carrying the asynchronous copies
 * (heatFluxStream).
 */
void c_Solver::copyHeatFluxBulkD2D(int species, cudaStream_t stream) {
  const size_t gridSize =
      (size_t)grid->getNXN() * grid->getNYN() * grid->getNZN();
  cudaErrChk(cudaMemcpyAsync(heatFluxBulkCUDAPtr[species] + 0 * gridSize,
                             EMf->gpuRhons().speciesPtr(species),
                             gridSize * sizeof(cudaMomentType),
                             cudaMemcpyDeviceToDevice, stream));
  cudaErrChk(cudaMemcpyAsync(heatFluxBulkCUDAPtr[species] + 1 * gridSize,
                             EMf->gpuJxs().speciesPtr(species),
                             gridSize * sizeof(cudaMomentType),
                             cudaMemcpyDeviceToDevice, stream));
  cudaErrChk(cudaMemcpyAsync(heatFluxBulkCUDAPtr[species] + 2 * gridSize,
                             EMf->gpuJys().speciesPtr(species),
                             gridSize * sizeof(cudaMomentType),
                             cudaMemcpyDeviceToDevice, stream));
  cudaErrChk(cudaMemcpyAsync(heatFluxBulkCUDAPtr[species] + 3 * gridSize,
                             EMf->gpuJzs().speciesPtr(species),
                             gridSize * sizeof(cudaMomentType),
                             cudaMemcpyDeviceToDevice, stream));
}

void c_Solver::copyHeatFluxD2H(int species, cudaStream_t stream) {
#if IPIC3D_COMPILE_DISK_OUTPUT
  const auto gridSize = grid->getNXN() * grid->getNYN() * grid->getNZN();
  cudaErrChk(cudaMemcpyAsync(
      EMf->getHeatFluxSpeciesPtr(species), heatFluxCUDAPtr[species],
      gridSize * HeatFlux::ComponentCount * sizeof(cudaMomentType),
      cudaMemcpyDefault, stream));
#else
  (void)species;
  (void)stream;
  return;
#endif
}

/**
 * @brief Recompute all particle moments directly from the current GPU particle
 * state.
 *
 * This path zeroes the device moment buffers, launches the full moment kernel
 * for every species, scatters the results into the field solver's device
 * arrays, and then waits for completion via MomentsAwait().
 */
void c_Solver::CalculateMoments() {
  profiling::Range range(profiling::Region::InitialMoments);

  // With sorting enabled, cell-sort first and use the cell-aware moment
  // kernel
  const bool useSortedMoments = (sortingCycle_ >= 1);
  if (useSortedMoments)
    sortSpecies(std::vector<int>(ns, 1));

  auto gridSize = grid->getNXN() * grid->getNYN() * grid->getNZN();
  for (int i = 0; i < ns; i++) {
    cudaErrChk(cudaMemsetAsync(momentsCUDAPtr[i], 0,
                               gridSize * 10 * sizeof(cudaMomentType),
                               streams[i]));
    // Particle data is already resident on the device; only the kernel launch
    // is needed here.
    const uint32_t nop = pclsArrayHostPtr[i]->getNOP();
    if (useSortedMoments) {
      if (nop > 0) {
        const int numCells = cellSorters[i].getNumCells();
        const int threads = numCells * WARP_SIZE;
        cellAwareMomentKernel<<<getGridSize(threads, DEFAULT_BLOCK_SIZE),
                                DEFAULT_BLOCK_SIZE, 0, streams[i]>>>(
            cellSorters[i].getCellStartOffsets(), numCells, nop,
            pclsArrayCUDAPtr[i], gridCUDA, momentsCUDAPtr[i]);
      }
    } else {
      momentKernelNew<<<(nop / DEFAULT_BLOCK_SIZE + 1), DEFAULT_BLOCK_SIZE, 0,
                        streams[i]>>>(pclsArrayCUDAPtr[i], gridCUDA,
                                      momentsCUDAPtr[i], 0);
    }
    // Scatter the packed moments into the field solver's per-species arrays
    EMf->gpuScatterMomentsD2D(momentsCUDAPtr[i], i, streams[i]);
    // Record cycleEndEvent[i] on streams[i] so the per-species event-based
    // sync inside MomentsAwait() actually waits for this initial moment
    // pipeline. Without this record, MomentsAwait would wait only for the
    // record made in initCUDA and read the moments before these kernels
    // finish.
    cudaErrChk(cudaEventRecord(cycleEndEvent[i], streams[i]));
  }

  // Synchronize all species before the field-side ghost exchange and
  // reductions.
  MomentsAwait(-1);
}

/**
 * @brief Advance the electric field solver for one cycle.
 * @param cycle Simulation cycle being advanced.
 */
void c_Solver::CalculateField(int cycle) {
  // E(n+theta) in d_Exth (Faraday) and, smoothed, in gpuExForPcls() (mover);
  // d_Ex keeps E^n until CalculateB
  EMf->gpuCalculateE(cycle);
  // Record event so particle packing can wait for solver completion
  cudaErrChk(cudaEventRecord(solverDoneEvent, EMf->gpuSolverStream()));
}

bool c_Solver::needsHeatFluxCalculation(int cycle) const {
#if IPIC3D_COMPILE_DIAGNOSTIC_CALCULATIONS
  if (!heatFluxCalculationEnabled_)
    return false;
  return (cycle % col->getFieldOutputCycle() == 0 || cycle == first_cycle);
#else
  (void)cycle;
  return false;
#endif
}

void c_Solver::ScheduleHeatFlux(int cycle) {
#if IPIC3D_COMPILE_DIAGNOSTIC_CALCULATIONS
  if (!needsHeatFluxCalculation(cycle)) {
    heatFluxScheduledCycle_ = -1;
    return;
  }

  heatFluxScheduledCycle_ = cycle;
  const auto gridSize = grid->getNXN() * grid->getNYN() * grid->getNZN();

  // Gate: heatFluxStream must not start reading d_rhons/d_Jxs/d_Jys/d_Jzs
  // until the full MomentsAwait() GPU pipeline on solverStream_ has completed
  // (gpuCommunicateGhostP2G_AllSpecies + gpuCalculateHatFunctions).
  // momentsPipelineDoneEvt is recorded at the tail of that pipeline, so this
  // single stream-wait covers all ns species simultaneously.
  cudaErrChk(cudaStreamWaitEvent(heatFluxStream, momentsPipelineDoneEvt, 0));

  for (int s = 0; s < ns; ++s) {
    // the particles too: the sort of a checkpoint or of a capped particle
    // output (outputCopyAsync) runs on their stream with no host wait
    cudaErrChk(cudaStreamWaitEvent(heatFluxStream, cycleEndEvent[s], 0));
    copyHeatFluxBulkD2D(s, heatFluxStream);
    cudaErrChk(cudaMemsetAsync(heatFluxCUDAPtr[s], 0,
                               gridSize * HeatFlux::ComponentCount *
                                   sizeof(cudaMomentType),
                               heatFluxStream));
    const uint32_t nop = pclsArrayHostPtr[s]->getNOP();
    if (nop > 0) {
      heatFluxKernelUnsorted<<<getGridSize((int)nop, DEFAULT_BLOCK_SIZE),
                               DEFAULT_BLOCK_SIZE, 0, heatFluxStream>>>(
          pclsArrayCUDAPtr[s], gridCUDA, heatFluxBulkCUDAPtr[s],
          heatFluxCUDAPtr[s], (cudaMomentType)col->getQOM(s),
          HEAT_FLUX_RHO_FLOOR);
    }
    cudaErrChk(cudaEventRecord(heatFluxReadDoneEvt[s], heatFluxStream));
  }

#if IPIC3D_COMPILE_DISK_OUTPUT
  if (heatFluxOutputEnabled_) {
    // the deferred field outputs still writing the host heat flux first
    ioManager->beforeFieldCopy();
    for (int s = 0; s < ns; ++s) {
      copyHeatFluxD2H(s, heatFluxStream);
      cudaErrChk(cudaEventRecord(heatFluxHostDoneEvt[s], heatFluxStream));
    }
  }
#endif
#else
  (void)cycle;
  return;
#endif
}

void c_Solver::finishHeatFluxForOutput(int cycle) {
#if IPIC3D_COMPILE_DISK_OUTPUT
  if (!heatFluxOutputEnabled_ || heatFluxScheduledCycle_ != cycle)
    return;

  profiling::Range range(profiling::Region::HeatFluxHalo);

  for (int s = 0; s < ns; ++s)
    cudaErrChk(cudaEventSynchronize(heatFluxHostDoneEvt[s]));

  for (int s = 0; s < ns; ++s)
    EMf->communicateGhostHeatFlux(s);

  heatFluxScheduledCycle_ = -1;
#else
  (void)cycle;
  return;
#endif
}

/**
 * @brief Pack the current fields into the mover/interpolation device buffer
 * (gpuPackFieldForPclsToCenter).
 *
 * Macrocell spectra consumes this same packed B layout during data analysis,
 * before the mover path refreshes it for the current-cycle E solve. The caller
 * can request a host-side wait when the buffer must be ready immediately.
 */
void c_Solver::refreshFieldForPclsDeviceBuffer(bool synchronize) {
  profiling::Range range(profiling::Region::FieldPack);
  // The mover samples E(n+theta) (gpuExForPcls: smoothed when smoothing is
  // on) and B^n at the nodes plus the external field.
  //
  // Re-record solverDoneEvent here to capture the *current* tail of
  // solverStream_, not just the E-solver tail from the last CalculateField().
  // This is necessary because:
  //   - The mover path calls this after CalculateField(), so only E work is
  //     outstanding — the re-record is a cheap no-op in that case.
  //   - The data analysis (macrocell spectra) calls this BEFORE
  //     CalculateField(), meaning the previous cycle's CalculateB() and
  //     MomentsAwait() solver-stream work may still be queued. Without a fresh
  //     record here the wait below only gates on the previous E-solve record,
  //     letting gpuPackFieldForPclsToCenter race with writes to Bxn/Byn/Bzn
  //     from CalculateB(prev cycle).
  //   - On the first cycle the event has never been recorded; recording it
  //     against the current (empty or H2D-initialized) stream makes the
  //     wait formally safe instead of a no-op.
  cudaErrChk(cudaEventRecord(solverDoneEvent, EMf->gpuSolverStream()));
  cudaErrChk(cudaStreamWaitEvent(streams[0], solverDoneEvent, 0));
  {
    const int ncells =
        (grid->getNXN() - 1) * (grid->getNYN() - 1) * grid->getNZN();
    const int gridDim = (ncells + kPackFieldBlock - 1) / kPackFieldBlock;
    gpuPackFieldForPclsToCenter<<<gridDim, kPackFieldBlock, 0, streams[0]>>>(
        fieldForPclCUDAPtr, EMf->gpuExForPcls().devPtr(),
        EMf->gpuEyForPcls().devPtr(), EMf->gpuEzForPcls().devPtr(),
        EMf->gpuBxn().devPtr(), EMf->gpuByn().devPtr(), EMf->gpuBzn().devPtr(),
        EMf->gpuBx_ext().devPtr(), EMf->gpuBy_ext().devPtr(),
        EMf->gpuBz_ext().devPtr(), grid->getNXN(), grid->getNYN(),
        grid->getNZN());
  }
  cudaErrChk(cudaEventRecord(event0, streams[0]));
  if (synchronize) {
    cudaErrChk(cudaStreamSynchronize(streams[0]));
  }
}

/**
 * Cells of the rank whose centre is outside the planet of the mover
 * (doSphere 1: a sphere, 2: a circle in the x-z plane, 0: no planet). The
 * first mover removes the particles the input file loads in the other cells.
 */
static int cellsOutsidePlanet(const Grid3DCU& grid,
                              const moverParameter& mover) {
  const int nxc = grid.getNXC(), nyc = grid.getNYC(), nzc = grid.getNZC();
  if (mover.doSphere == 0)
    return (nxc - 2) * (nyc - 2) * (nzc - 2);
  const double r2 = mover.sphereRadius * mover.sphereRadius;
  int count = 0;
  for (int i = 1; i < nxc - 1; i++)
    for (int j = 1; j < nyc - 1; j++)
      for (int k = 1; k < nzc - 1; k++) {
        const double dx = grid.getXC(i) - mover.sphereOrigin[0];
        const double dy =
            mover.doSphere == 1 ? grid.getYC(j) - mover.sphereOrigin[1] : 0.0;
        const double dz = grid.getZC(k) - mover.sphereOrigin[2];
        count += dx * dx + dy * dy + dz * dz >= r2;
      }
  return count;
}

/**
 * @brief Set up the control of the particle count (ParticleSplitting,
 * ParticleMerging).
 *
 * The control keeps the particle count of a species on this rank close to its
 * reference count: the particles per cell of the species times the cells of
 * the rank outside the planet, which is what the input file loads there and
 * the planet leaves. The species from ns_solar_wind on (planetary) are not
 * controlled.
 */
void c_Solver::initParticleControl() {
  pclControlReference_.assign(ns, 0);
  splitGeometry_.assign(ns, SplitGeometry{});
  cyclesSinceMerge_.assign(ns, 0);
  mergeSpecies_ = -1;
  if (!col->getParticleSplitting() && !col->getParticleMerging())
    return;

  const grid3DCUDA& g = gridCUDA;
  const uint32_t cells = (g.nxc - 2) * (g.nyc - 2) * (g.nzc - 2);
  const uint32_t keptCells = cellsOutsidePlanet(*grid, moverParams[0]);
  const int domainCells[3] = {col->getNxc(), col->getNyc(), col->getNzc()};
  const cudaParticleType start[3] = {g.xStart, g.yStart, g.zStart};
  const cudaParticleType d[3] = {g.dx, g.dy, g.dz};
  for (int i = 0; i < col->getNumSolarWindSpecies(); i++) {
    const uint32_t npcel = col->getNpcel(i);
    pclControlReference_[i] = npcel * keptCells;
    // The two particles of a split are one loading interval apart.
    const int perCell[3] = {col->getNpcelx(i), col->getNpcely(i),
                            col->getNpcelz(i)};
    SplitGeometry& split = splitGeometry_[i];
    for (int a = 0; a < 3; a++) {
      split.start[a] = start[a];
      split.d[a] = d[a];
      split.invd[a] = 1.0 / d[a];
      split.halfSpacing[a] = 0.5 * d[a] / std::max(perCell[a], 1);
      if (domainCells[a] > 1)
        split.axis[split.axisCount++] = a;
    }

    const uint32_t loaded = pclsArrayHostPtr[i]->getNOP();
    if (restart_status == 0 && loaded != npcel * cells)
      std::cout << "WARNING: particle control: rank " << MPIdata::get_rank()
                << " species " << i << " loads " << loaded
                << " particles, not the particles per cell times the cells ("
                << npcel * cells << ") that its reference count assumes"
                << std::endl;
  }
}

/**
 * @brief Split particles of a species that fell below
 * ParticleSplittingThreshold times its reference count on this rank, up to
 * that count.
 *
 * A particle is split once per cycle at most, so a species below half its
 * reference count doubles every cycle at most. The particle arrays keep the
 * room the mover's open-boundary copies need: a split stops at capacity /
 * ParticleArrayConfig::RESIZE_THRESHOLD, the count at which the exchange grows
 * them.
 *
 * Runs on the species' stream, before the mover. The species merged in this
 * cycle is not split: the merge has flagged particles for deletion, and the
 * twin of one of them would keep half its charge.
 */
void c_Solver::splitParticles(int species, int cycle) {
  particleArrayCUDA* pcls = pclsArrayHostPtr[species];
  const uint32_t reference = pclControlReference_[species];
  const uint32_t nop = pcls->getNOP();
  // nop >= reference also stops a species that is not controlled, and a
  // threshold above 1
  if (nop == 0 || nop >= reference ||
      nop >= col->getParticleSplittingThreshold() * reference ||
      splitGeometry_[species].axisCount == 0 || species == mergeSpecies_)
    return;
  const uint32_t room = static_cast<uint32_t>(
      pcls->getCapacity() / ParticleArrayConfig::RESIZE_THRESHOLD);
  if (room <= nop)
    return;
  profiling::Range range(profiling::Region::ParticleSplit, species);
  const uint32_t count = std::min({reference - nop, nop, room - nop});

  std::cout << "Particle splitting: rank " << MPIdata::get_rank() << " species "
            << species << " particles " << nop << " + " << count << std::endl;
  particleSplittingKernel<<<getGridSize((int)count, DEFAULT_BLOCK_SIZE),
                            DEFAULT_BLOCK_SIZE, 0, streams[species]>>>(
      *pcls->getSoA(), count, (uint32_t)cycle * SPLIT_PARENT_STRIDE,
      splitGeometry_[species],
      particlesHost[species]->getParticleIDGenerator());
  pcls->setNOE(nop + count);
  cudaErrChk(cudaMemcpyAsync(pclsArrayCUDAPtr[species], pcls,
                             sizeof(particleArrayCUDA), cudaMemcpyDefault,
                             streams[species]));
}

/**
 * @brief Choose the species to merge in the next mover: among the species
 * above ParticleMergingThreshold times their reference count on this rank,
 * the one merged longest ago. One species is merged per cycle.
 */
void c_Solver::scheduleMerging() {
  mergeSpecies_ = -1;
  for (int i = 0; i < ns; i++) {
    cyclesSinceMerge_[i]++;
    const uint32_t reference = pclControlReference_[i];
    if (reference > 0 &&
        pclsArrayHostPtr[i]->getNOP() >
            col->getParticleMergingThreshold() * reference &&
        (mergeSpecies_ < 0 ||
         cyclesSinceMerge_[i] > cyclesSinceMerge_[mergeSpecies_]))
      mergeSpecies_ = i;
  }
}

/**
 * @brief Sort a species by cell and merge particles in its cells that hold
 * more than the particles per cell of the input file.
 *
 * The merged-away particles are flagged in the departure array; the mover
 * that follows skips them and its compaction removes them.
 */
void c_Solver::mergeParticles(int species, bool waitForHeatFlux) {
  profiling::Range range(profiling::Region::ParticleMerge, species);
  particleArrayCUDA* pcls = pclsArrayHostPtr[species];
  cudaStream_t stream = streams[species];
  std::cout << "Particle merging: rank " << MPIdata::get_rank() << " species "
            << species << " particles " << pcls->getNOP() << std::endl;

  cudaErrChk(cudaStreamSynchronize(stream));
  // The previous cycle's output copy and the heat flux read the particles on
  // other streams.
  cudaErrChk(cudaStreamWaitEvent(stream, eventOutputCopy, 0));
  if (waitForHeatFlux)
    cudaErrChk(cudaStreamWaitEvent(stream, heatFluxReadDoneEvt[species], 0));
  sortSpeciesByCell(species);

  const CellSorter& sorter = cellSorters[species];
  const int numCells = sorter.getNumCells();
  particleMergingKernel<<<getGridSize(numCells * WARP_SIZE, DEFAULT_BLOCK_SIZE),
                          DEFAULT_BLOCK_SIZE, 0, stream>>>(
      sorter.getCellStartOffsets(), sorter.getCellCounts(), numCells,
      col->getNpcel(species), *pcls->getSoA(),
      departureArrayHostPtr[species]->getArray());
  cyclesSinceMerge_[species] = 0;
}

/**
 * @brief Launch the asynchronous GPU mover pipeline for one species.
 *
 * The method advances particles, extracts exit/planet populations, compacts the
 * stayed prefix, prepares the CPU-side communication buffer, and optionally
 * accumulates moments for the stayed particles in the unsorted pipeline.
 *
 * @param species Species index to advance.
 * @param cycle Simulation cycle being advanced.
 * @param doMomentsInLauncher Whether the stayed-prefix moments are accumulated
 * here.
 * @param waitForHeatFlux Wait until the heat-flux kernel has read this
 * species' particles.
 * @return Number of particles removed from the stayed prefix
 *         (MPI exiting + deleted + planet-removed).
 */
int c_Solver::cudaLauncherAsync(const int species, const int cycle,
                                const bool doMomentsInLauncher,
                                const bool waitForHeatFlux) {
  cudaSetDevice(
      cudaDeviceOnNode); // Required when multiple MPI ranks share a node.
  profiling::Range range(profiling::Region::Mover, species);

  // Explicit edge: gate every SoA-mutating kernel on this species' stream
  // behind the previous cycle's output D->H drain. Required because output
  // copies run on outputStream, which is not ordered with streams[species].
  // Pre-recorded at init so this resolves immediately on the first cycle.
  // Aux stream streams[species+ns] is transitively
  // ordered via moverHashedReadyEvt[species] later in this function.
  cudaErrChk(cudaStreamWaitEvent(streams[species], eventOutputCopy, 0));
  cudaErrChk(cudaStreamWaitEvent(streams[species + ns], eventOutputCopy, 0));
  if (waitForHeatFlux) {
    cudaErrChk(
        cudaStreamWaitEvent(streams[species], heatFluxReadDoneEvt[species], 0));
    cudaErrChk(cudaStreamWaitEvent(streams[species + ns],
                                   heatFluxReadDoneEvt[species], 0));
  }

  // Persistent per-species events (created in initCUDA, destroyed in
  // deInitCUDA):
  //   moverHashedReadyEvt[species]: recorded on streams[species] after the
  //     mover writes hashedSums
  //   auxHashedReadyEvt[species]: recorded on streams[species+ns] after
  //     exitingKernel completes
  cudaEvent_t& event1 = moverHashedReadyEvt[species];
  cudaEvent_t& event2 = auxHashedReadyEvt[species];

  if (col->getParticleSplitting())
    splitParticles(species, cycle);

  // ======= Launch mover kernels after field data is available =======
  cudaErrChk(cudaStreamWaitEvent(streams[species], event0, 0));
  // Empty species (e.g. planetary species before exosphere injection has
  // populated them) have NOP=0. Skip kernels that iterate per-particle;
  // downstream consumers already correctly handle zero exit/stayed counts.
  // Note: moments memset must still run so the moment scatter
  // (gpuScatterMomentsD2D) copies zeros.
  const uint32_t nop = pclsArrayHostPtr[species]->getNOP();
  const moverParameter& mover = moverParams[species];
  if (nop > 0) {
    if (doPlanet_)
      moverSubcyclesKernel<<<getGridSize((int)nop, kMoverBlockSize),
                             kMoverBlockSize, 0, streams[species]>>>(
          mover, fieldForPclCUDAPtr, gridCUDA);
    else
      moverKernel<<<getGridSize((int)nop, kMoverBlockSize), kMoverBlockSize, 0,
                    streams[species]>>>(mover, fieldForPclCUDAPtr, gridCUDA);

    // The wall pass: the particles the mover left at a reflecting wall, then
    // the list is emptied for the next cycle.
    if (mover.doReflect) {
      const int wallCapacity = fillerBufferArrayHostPtr[species]->getSize();
      if (wallCapacity > 0) {
        const int wallGrid = getGridSize(wallCapacity, kMoverBlockSize);
        if (doPlanet_)
          moverSubcyclesWallsKernel<<<wallGrid, kMoverBlockSize, 0,
                                      streams[species]>>>(
              mover, fieldForPclCUDAPtr, gridCUDA);
        else
          moverWallsKernel<<<wallGrid, kMoverBlockSize, 0, streams[species]>>>(
              mover, fieldForPclCUDAPtr, gridCUDA);
      }
      cudaErrChk(cudaMemsetAsync(&moverCountersCUDAPtr[species].wallCount, 0,
                                 sizeof(uint32_t), streams[species]));
    }
  }

  cudaErrChk(cudaEventRecord(event1, streams[species]));
  // Unsorted pipeline: compute moments for stayed particles right after mover
  // (overlaps with hashedSum D2H + exitingKernel on streams[species+ns]).
  // Sorted pipeline: moments deferred to MoverAwaitAndPclExchange after sort.
  if (doMomentsInLauncher) {
    const auto gridSize = grid->getNXN() * grid->getNYN() * grid->getNZN();
    cudaErrChk(cudaMemsetAsync(momentsCUDAPtr[species], 0,
                               gridSize * 10 * sizeof(cudaMomentType),
                               streams[species]));
    if (nop > 0)
      momentKernelStayed<<<getGridSize((int)nop, DEFAULT_BLOCK_SIZE),
                           DEFAULT_BLOCK_SIZE, 0, streams[species]>>>(
          pclsArrayCUDAPtr[species], departureArrayCUDAPtr[species],
          &moverCountersCUDAPtr[species].openBCAppendCounters.accepted,
          gridCUDA, momentsCUDAPtr[species]);
  }
  // Mark that momentKernelStayed has finished reading the accepted OpenBC
  // append count and pclsArrayCUDAPtr->nop_. This event is consumed below
  // before the OpenBC path resets the counters and rewrites particle metadata.
  // It is recorded unconditionally so the wait is well-defined on every cycle
  // (in the sorted path it just chains the empty cycle's mover, which is
  // cheap).
  cudaErrChk(cudaEventRecord(stayedMomentsDoneEvt[species], streams[species]));

  // Copy 8 hashedSums to host: 6 directions + delete + planet (XLOW..PLANET)
  cudaErrChk(cudaStreamWaitEvent(streams[species + ns], event1, 0));
  cudaErrChk(cudaMemcpyAsync(
      hashedSumArrayHostPtr[species], hashedSumArrayCUDAPtr[species],
      (departureArrayElementType::PLANET_HASHEDSUM_INDEX + 1) *
          sizeof(hashedSum),
      cudaMemcpyDefault, streams[species + ns]));

  // The host waits for the mover's counts: with open boundaries first in the
  // append accounting below
  profiling::Phase counts;
  counts.start(profiling::Region::MoverCounts, species);

  // Copy OpenBC append accounting to host.
  if (mover.doOpenBC) {
    // This D->H read is non-destructive; it only needs mover ordering (event1
    // wait above) and may run in parallel with momentKernelStayed.
    cudaErrChk(
        cudaMemcpyAsync(&moverCountersHostPtr[species].openBCAppendCounters,
                        &moverCountersCUDAPtr[species].openBCAppendCounters,
                        sizeof(OpenBCAppendCounters), cudaMemcpyDefault,
                        streams[species + ns]));
    // The next two operations are destructive for momentKernelStayed:
    //   - cudaMemsetAsync zeros the accepted count, which the kernel reads as
    //     `*appendCount` to compute totPcl = nop + appendCount.
    //   - The H->D rewrite of pclsArrayCUDAPtr below mutates pclsArray->nop_,
    //     which the kernel reads via pclsArray->getNOP().
    // Without the wait below, both can land while the kernel is still
    // iterating, silently dropping (memset wins) or double-counting (H->D
    // wins) the OpenBC-appended particles in the moment deposition.
    cudaErrChk(cudaStreamWaitEvent(streams[species + ns],
                                   stayedMomentsDoneEvt[species], 0));
    cudaErrChk(
        cudaMemsetAsync(&moverCountersCUDAPtr[species].openBCAppendCounters, 0,
                        sizeof(OpenBCAppendCounters), streams[species + ns]));
    cudaErrChk(cudaStreamSynchronize(streams[species + ns]));

    const uint32_t newPclAfterOBC =
        pclsArrayHostPtr[species]->getNOP() +
        moverCountersHostPtr[species].openBCAppendCounters.accepted;
    pclsArrayHostPtr[species]->setNOE(newPclAfterOBC);
    cudaErrChk(cudaMemcpyAsync(
        pclsArrayCUDAPtr[species], pclsArrayHostPtr[species],
        sizeof(particleArrayCUDA), cudaMemcpyDefault, streams[species + ns]));
  }

  // ======= Host-side accounting after mover completion =======
  cudaErrChk(cudaStreamSynchronize(streams[species + ns]));
  counts.stop();
  // the rest of the task: buffers, extraction, compaction, exiting D->H
  profiling::Range extract(profiling::Region::ParticleExtract, species);
  // cudaErrChk(cudaStreamSynchronize(streams[species]));
  int count_exiting = 0; // exiting particle number
  for (int i = 0; i < departureArrayElementType::DELETE_HASHEDSUM_INDEX; i++)
    count_exiting += hashedSumArrayHostPtr[species][i].getSum();
  const int count_deleted =
      hashedSumArrayHostPtr[species]
                           [departureArrayElementType::DELETE_HASHEDSUM_INDEX]
                               .getSum(); // deleted particle number
  const int count_removed_planet =
      hashedSumArrayHostPtr[species]
                           [departureArrayElementType::PLANET_HASHEDSUM_INDEX]
                               .getSum(); // planet particle number
  planetPclCount[species] = count_removed_planet;
  const int hole = count_exiting + count_deleted + count_removed_planet;
  // if (count_deleted > 0){
  //   std::cout << " Particle holes myrank: "<< MPIdata::get_rank() << "
  //   species: " << species<< " hole: " << hole << " deleted: "<< count_deleted
  //   << std::endl;
  // }
  if (count_exiting > exitingArrayHostPtr[species]->getSize()) {
    // Expand the exiting AoS buffer before extracting particles into it.
    exitingArrayHostPtr[species]->expand(count_exiting *
                                             ParticleArrayConfig::RESIZE_FACTOR,
                                         streams[species + ns]);
    cudaErrChk(cudaMemcpyAsync(
        exitingArrayCUDAPtr[species], exitingArrayHostPtr[species],
        sizeof(exitingArray), cudaMemcpyDefault, streams[species + ns]));
  }

  // The filler buffer of the compaction/sorting kernels.  On a rank with
  // reflecting walls it also holds the wall list of the mover, which keeps
  // its initial share of the particles when the species grows.
  const int fillerNeeded =
      mover.doReflect
          ? std::max(hole, static_cast<int>(AUX_BUFFER_FRACTION * nop))
          : hole;
  if (fillerNeeded > fillerBufferArrayHostPtr[species]->getSize()) {
    fillerBufferArrayHostPtr[species]->expand(
        fillerNeeded * ParticleArrayConfig::RESIZE_FACTOR,
        streams[species + ns]);
    cudaErrChk(cudaMemcpyAsync(
        fillerBufferArrayCUDAPtr[species], fillerBufferArrayHostPtr[species],
        sizeof(fillerBuffer), cudaMemcpyDefault, streams[species + ns]));
  }

  // Planet extraction must run before exitingKernel, which overwrites
  // departureArray[].hashedId with HOLE hashes for front-region particles.
  // planetExtractionKernel needs the original PLANET hashedId to scatter
  // correctly.
  if (count_removed_planet > 0) {
    // Previous-cycle planet work that was not followed by a host sync (see
    // processPlanetParticles) may still be reading this buffer.
    cudaErrChk(cudaStreamWaitEvent(streams[species + ns], planetDoneEvent, 0));
    if ((uint32_t)count_removed_planet >
        planetArrayHostPtr[species]->getSize()) {
      planetArrayHostPtr[species]->expand(
          count_removed_planet * ParticleArrayConfig::RESIZE_FACTOR,
          streams[species + ns]);
      cudaErrChk(cudaMemcpyAsync(
          planetArrayCUDAPtr[species], planetArrayHostPtr[species],
          sizeof(planetArray), cudaMemcpyDefault, streams[species + ns]));
    }
    planetExtractionKernel<<<getGridSize(
                                 (int)pclsArrayHostPtr[species]->getNOP(),
                                 DEFAULT_BLOCK_SIZE),
                             DEFAULT_BLOCK_SIZE, 0, streams[species + ns]>>>(
        pclsArrayCUDAPtr[species], departureArrayCUDAPtr[species],
        planetArrayCUDAPtr[species], hashedSumArrayCUDAPtr[species]);
  }

  // Re-read NOP: the OpenBC block above may have grown it via setNOE().
  const uint32_t nopAfterOBC = pclsArrayHostPtr[species]->getNOP();
  if (nopAfterOBC > 0)
    exitingKernel<<<getGridSize((int)nopAfterOBC, DEFAULT_BLOCK_SIZE),
                    DEFAULT_BLOCK_SIZE, 0, streams[species + ns]>>>(
        pclsArrayCUDAPtr[species], departureArrayCUDAPtr[species],
        exitingArrayCUDAPtr[species], hashedSumArrayCUDAPtr[species]);

  cudaErrChk(cudaEventRecord(event2, streams[species + ns]));

  // Prepare comm buffer and copy exiting particles D→H.
  particlesCommInj[species]->clearCommBuffer();
  if (count_exiting > 0) {
    particlesCommInj[species]->prepareCommBufferForNOP(count_exiting);
    cudaErrChk(cudaMemcpyAsync(particlesCommInj[species]->getCommPclsDataMut(),
                               exitingArrayHostPtr[species]->getArray(),
                               count_exiting * sizeof(SpeciesParticle),
                               cudaMemcpyDefault, streams[species + ns]));
  }

  // Compact the stayed prefix into the front of the SoA arrays.
  // (compactParticles1/2 perform hole-and-filler compaction only; no sorting.)
  cudaErrChk(cudaStreamWaitEvent(streams[species], event2, 0));
  const uint32_t stayedCount = pclsArrayHostPtr[species]->getNOP() - hole;
  if (hole > 0)
    compactParticles1<<<getGridSize(hole, SMALL_BLOCK_SIZE), SMALL_BLOCK_SIZE,
                        0, streams[species]>>>(
        pclsArrayCUDAPtr[species], departureArrayCUDAPtr[species],
        fillerBufferArrayCUDAPtr[species],
        hashedSumArrayCUDAPtr[species] +
            departureArrayElementType::FILLER_HASHEDSUM_INDEX,
        hole);
  // Guard against the catastrophic edge case hole == NOP (every particle
  // departs in a single step). Launching with zero work would produce a
  // grid size of 0 and a CUDA invalid-configuration error.
  if (stayedCount > 0)
    compactParticles2<<<getGridSize((int)stayedCount, DEFAULT_BLOCK_SIZE),
                        DEFAULT_BLOCK_SIZE, 0, streams[species]>>>(
        pclsArrayCUDAPtr[species], departureArrayCUDAPtr[species],
        fillerBufferArrayCUDAPtr[species],
        hashedSumArrayCUDAPtr[species] +
            departureArrayElementType::HOLE_HASHEDSUM_INDEX,
        stayedCount);

  cudaErrChk(cudaStreamSynchronize(
      streams[species + ns])); // exiting D→H complete, comm buffer ready

  return hole; // Number of exiting + deleted + planet particles
}

/**
 * @brief Launch the per-cycle particle mover workflow for all species.
 *
 * This prepares the field interpolation buffer, decides whether the current
 * cycle uses the sorted or unsorted moment pipeline, and enqueues one async
 * mover task per species on the solver thread pool.
 * @param cycle Simulation cycle being advanced.
 * @return Always false.
 */
bool c_Solver::ParticlesMoverMomentAsync(int cycle) {
  // ======= Decide whether to use the sorted particle pipeline this cycle
  // =======
  sortThisCycle_ = (sortingCycle_ > 0) && (cycle % sortingCycle_ == 0);
  const bool doMomentsInLauncher = !sortThisCycle_;
  if (MPIdata::get_rank() == 0)
    printf("  [Cycle %d] Particle sorting: %s\n", cycle,
           sortThisCycle_ ? "ON" : "OFF");

  // Refresh the shared field-interpolation buffer for the mover. The per-
  // species mover streams consume it through cudaStreamWaitEvent(event0).
  refreshFieldForPclsDeviceBuffer(false);
  const bool waitForHeatFlux =
      heatFluxCalculationEnabled_ && (heatFluxScheduledCycle_ == cycle);

  // The movers of the other species start while this thread sorts and merges
  // the species to merge; its mover follows the merge.
  for (int i = 0; i < ns; i++)
    if (i != mergeSpecies_)
      exitingResults[i] =
          threadPoolPtr->enqueue(&c_Solver::cudaLauncherAsync, this, i, cycle,
                                 doMomentsInLauncher, waitForHeatFlux);
  if (mergeSpecies_ >= 0) {
    mergeParticles(mergeSpecies_, waitForHeatFlux);
    exitingResults[mergeSpecies_] = threadPoolPtr->enqueue(
        &c_Solver::cudaLauncherAsync, this, mergeSpecies_, cycle,
        doMomentsInLauncher, waitForHeatFlux);
  }

  return (false);
}

/**
 * @brief Process planet-hit particles across all species on the dedicated
 * planet stream.
 *
 * Ions are always absorbed and their charge is reduced on the GPU. Electrons of
 * all species are ranked together by the kinetic energy of one physical
 * electron and absorbed from the most energetic down, until the absorbed charge
 * best matches the ion charge plus the imbalance carried from previous cycles;
 * the rest are reflected back into the outgoing communication buffers. The
 * leftover imbalance stays on the GPU for the next cycle. Intermediate work
 * stays on the GPU until the final small host-side synchronization needed for
 * reflected-particle counts.
 */
void c_Solver::processPlanetParticles() {
  const bool doDiag = planetDiagCycle > 0;
  if (doDiag)
    planetDiagCyclesAccumulated++;

  // ======= Step 1: Check whether any planet particles were collected =======
  int totalIonPlanet = 0;
  int totalElecPlanet = 0;
  for (int i = 0; i < ns; i++) {
    if (planetPclCount[i] == 0)
      continue;
    if (col->getQOM(i) > 0)
      totalIonPlanet += planetPclCount[i];
    else if (col->getQOM(i) < 0)
      totalElecPlanet += planetPclCount[i];
  }
  // Nothing hit the surface: the carried charge is unchanged.
  if (totalIonPlanet == 0 && totalElecPlanet == 0) {
    return;
  }
  profiling::Range range(profiling::Region::PlanetAbsorption);

  if (doDiag) {
    planetDiagRankCycles += 1.0;
    for (int i = 0; i < ns; i++) {
      planetDiagHitCount[i] += planetPclCount[i];
      // Ions are always absorbed; electrons are added after the cutoff.
      if (col->getQOM(i) > 0)
        planetDiagAbsorbedCount[i] += planetPclCount[i];
    }
  }
  // Device addresses of the two sections of planetDiagChargeDevice: hit |Q|
  // per species [0, ns) and absorbed |Q| per electron species [ns, ...).
  // nullptr (diagnostics off) makes the kernels skip the atomics.
  cudaParticleType* const diagHitChargeDevice =
      doDiag ? planetDiagChargeDevice : nullptr;
  cudaParticleType* const diagAbsorbedElecChargeDevice =
      doDiag ? planetDiagChargeDevice + ns : nullptr;

  // ======= Step 2: Reduce removed ion charge on the GPU =======
  // ionChargeCycle is zero here: chargeCutoffKernel consumes and clears it.
  for (int i = 0; i < ns; i++) {
    if (col->getQOM(i) <= 0 || planetPclCount[i] == 0)
      continue;
    const int blockSize = DEFAULT_BLOCK_SIZE;
    const int gridSz = getGridSize(planetPclCount[i], blockSize);
    planetChargeReductionKernel<<<gridSz, blockSize,
                                  blockSize * sizeof(cudaParticleType),
                                  planetStream>>>(
        planetArrayCUDAPtr[i], planetPclCount[i],
        &planetBCStateDevice->ionChargeCycle,
        diagHitChargeDevice ? diagHitChargeDevice + i : nullptr);
  }
  // No host sync is needed here: later kernels consume ionChargeCycle
  // directly.

  // ======= Step 3: Resize shared work buffers as needed =======
  // Before the electrons are described to the kernels: the buffer behind
  // sortedGlobalIdx can move here. Nothing grows when no electron hit.
  // Round up to the next power of 2 for bitonic sort.
  int nPad = 1;
  while (nPad < totalElecPlanet)
    nPad <<= 1;

  if (nPad > planetBufCapacity) {
    if (planetEnergyBuf)
      cudaFree(planetEnergyBuf);
    if (planetGlobalIdxBuf)
      cudaFree(planetGlobalIdxBuf);
    if (planetWarpChargeBuf)
      cudaFree(planetWarpChargeBuf);
    planetBufCapacity = nPad;
    cudaErrChk(cudaMalloc(&planetEnergyBuf,
                          planetBufCapacity * sizeof(cudaParticleType)));
    cudaErrChk(
        cudaMalloc(&planetGlobalIdxBuf, planetBufCapacity * sizeof(uint32_t)));
    cudaErrChk(cudaMalloc(
        &planetWarpChargeBuf,
        planetWarpChargeCount(planetBufCapacity, planetElecSpeciesCount) *
            sizeof(cudaParticleType)));
  }

  // Expand the reflected-particle output buffer if needed.
  if (totalElecPlanet > planetReflectedBufCapacity) {
    if (planetReflectedBuf)
      cudaFree(planetReflectedBuf);
    planetReflectedBufCapacity = totalElecPlanet * 2;
    cudaErrChk(cudaMalloc(&planetReflectedBuf, planetReflectedBufCapacity *
                                                   sizeof(SpeciesParticle)));
  }

  // The electrons that hit, merged over the electron species, as the kernels
  // take them.
  PlanetElectrons electrons;
  electrons.planetArrs = planetArrayCUDAPtrDevice;
  electrons.speciesOffsets = planetElecOffsetsDevice;
  electrons.sortedGlobalIdx = planetGlobalIdxBuf;
  electrons.nSpecies = planetElecSpeciesCount;
  electrons.total = totalElecPlanet;

  if (totalElecPlanet == 0) {
    // No electron to balance against: carry the absorbed ion charge over.
    chargeCutoffKernel<<<1, 1, 0, planetStream>>>(
        electrons, planetEnergyBuf, planetWarpChargeBuf, planetBCStateDevice,
        diagAbsorbedElecChargeDevice, planetCutoffDevice);
    // No host sync on this path: the next cycle's planet extraction waits on
    // this event before overwriting the planet buffers.
    cudaErrChk(cudaEventRecord(planetDoneEvent, planetStream));
    return;
  }

  // ======= Step 4: Compute per-electron energy and build species offsets
  // =======
  int offset = 0;
  for (int e = 0; e < planetElecSpeciesCount; e++) {
    int specIdx = planetElecSpeciesMap[e];
    planetElecOffsets[e] = offset;
    if (planetPclCount[specIdx] > 0) {
      planetEnergyKernel<<<getGridSize(planetPclCount[specIdx],
                                       DEFAULT_BLOCK_SIZE),
                           DEFAULT_BLOCK_SIZE, 0, planetStream>>>(
          planetArrayCUDAPtr[specIdx], planetPclCount[specIdx],
          (cudaParticleType)col->getQOM(specIdx), (cudaParticleType)col->getC(),
          planetEnergyBuf, planetGlobalIdxBuf, offset,
          diagHitChargeDevice ? diagHitChargeDevice + specIdx : nullptr);
    }
    offset += planetPclCount[specIdx];
  }

  // Copy offsets to device for the cutoff and reflection kernels. The device
  // pointer array planetArrayCUDAPtrDevice needs no refresh: a planet buffer
  // that grows keeps its device object.
  cudaErrChk(cudaMemcpyAsync(planetElecOffsetsDevice, planetElecOffsets,
                             planetElecSpeciesCount * sizeof(int),
                             cudaMemcpyHostToDevice, planetStream));

  // ======= Step 5: Sort electron planet particles by descending energy =======
  if (nPad > totalElecPlanet) {
    bitonicPadKernel<<<getGridSize(nPad - totalElecPlanet, DEFAULT_BLOCK_SIZE),
                       DEFAULT_BLOCK_SIZE, 0, planetStream>>>(
        planetEnergyBuf, planetGlobalIdxBuf, totalElecPlanet, nPad);
  }
  // One thread per compare-and-swap pair.
  for (int k = 2; k <= nPad; k <<= 1) {
    for (int j = k >> 1; j > 0; j >>= 1) {
      bitonicSortStepKernel<<<getGridSize(nPad / 2, DEFAULT_BLOCK_SIZE),
                              DEFAULT_BLOCK_SIZE, 0, planetStream>>>(
          planetEnergyBuf, planetGlobalIdxBuf, j, k, nPad);
    }
  }

  // ======= Step 6: Find the energy cutoff that balances the absorbed charge
  // =======
  // All threads gather the charges into sums per warp of the sorted order;
  // the single-thread cutoff kernel then steps over whole warps. It always
  // writes planetCutoffDevice and updates the carried charge.
  planetWarpChargeKernel<<<getGridSize(totalElecPlanet, DEFAULT_BLOCK_SIZE),
                           DEFAULT_BLOCK_SIZE, 0, planetStream>>>(
      electrons, planetWarpChargeBuf);
  chargeCutoffKernel<<<1, 1, 0, planetStream>>>(
      electrons, planetEnergyBuf, planetWarpChargeBuf, planetBCStateDevice,
      diagAbsorbedElecChargeDevice, planetCutoffDevice);
  // No host sync is needed here: the reflection kernel reads planetCutoffDevice
  // directly.

  // ======= Step 7: Reflect and compact surviving electrons =======
  // Diffuse by default (planetReflectionType = 1): Lambert (cosine-law)
  // direction on the outward hemisphere, speed kept; specular with 0.
  const moverParameter& planet = moverParams[0]; // the same for all species
  PlanetReflection reflection;
  reflection.originX = planet.sphereOrigin[0];
  reflection.originY = planet.sphereOrigin[1];
  reflection.originZ = planet.sphereOrigin[2];
  reflection.radius = planet.sphereRadius;
  reflection.doSphere = planet.doSphere;
  reflection.diffuse = col->getPlanetReflectionType() == 1;
  reflection.rngSeed = planetRngCycleCounter++;

  // Zero the per-species atomic survivor counters.
  cudaErrChk(cudaMemsetAsync(planetSurvivorCountDevice, 0,
                             planetElecSpeciesCount * sizeof(int),
                             planetStream));

  // Launch one thread per electron candidate; each checks the device-side
  // cutoff.
  planetReflectCompactKernel<<<getGridSize(totalElecPlanet, DEFAULT_BLOCK_SIZE),
                               DEFAULT_BLOCK_SIZE, 0, planetStream>>>(
      electrons, planetCutoffDevice, reflection, planetReflectedBuf,
      planetSurvivorCountDevice);

  // ======= Step 8: Copy survivor counts back to the host =======
  cudaErrChk(cudaMemcpyAsync(planetSurvivorCount, planetSurvivorCountDevice,
                             planetElecSpeciesCount * sizeof(int),
                             cudaMemcpyDeviceToHost, planetStream));
  cudaErrChk(
      cudaStreamSynchronize(planetStream)); // Counts are needed on host for
                                            // buffer growth and D2H copies.
  if (doDiag) {
    for (int e = 0; e < planetElecSpeciesCount; e++) {
      const int specIdx = planetElecSpeciesMap[e];
      planetDiagAbsorbedCount[specIdx] +=
          planetPclCount[specIdx] - planetSurvivorCount[e];
    }
  }

  // ======= Step 9: Copy reflected particles into the per-species comm buffers
  // =======
  // A reflected electron can lie outside the local subdomain (the planet can
  // straddle ranks, and the electron is placed up to 2R from the centre): the
  // exchange in MoverAwaitAndPclExchange() forwards it to its owner rank like
  // any exiting particle. One placed beyond a domain face gets that face's
  // particle boundary condition: exit (0) deletes it; mirror (1) mirrors its
  // position and normal velocity; open inflow (2) mirrors its position and
  // draws a thermal velocity into the domain; open outflow (3) leaves it
  // outside, and the exchange then removes it as unrecoverable, with a
  // warning.
  for (int e = 0; e < planetElecSpeciesCount; e++) {
    if (planetSurvivorCount[e] == 0)
      continue;
    int specIdx = planetElecSpeciesMap[e];
    const int nRefl = planetSurvivorCount[e];
    const int commOffset = particlesCommInj[specIdx]->getCommNOP();
    // Grow the communication buffer before appending reflected particles.
    particlesCommInj[specIdx]->prepareCommBufferForNOP(commOffset + nRefl);
    // Copy the compact reflected AoS block into the next comm-buffer slot
    // range.
    cudaErrChk(cudaMemcpyAsync(
        particlesCommInj[specIdx]->getCommPclsDataMut() + commOffset,
        planetReflectedBuf + planetElecOffsets[e],
        nRefl * sizeof(SpeciesParticle), cudaMemcpyDeviceToHost, planetStream));
  }

  // Complete all reflected-particle D2H copies before MPI exchange starts.
  cudaErrChk(cudaStreamSynchronize(planetStream));
}

/**
 * @brief Wait for mover tasks, exchange particles through MPI, and finalize
 * moments.
 *
 * This method joins the async mover futures, optionally processes planet hits,
 * overlaps MPI exchange with GPU sorting, appends incoming particles, and
 * finishes the moment accumulation path used by the current cycle.
 * @param cycle Simulation cycle being finalized.
 * @return Always false.
 */
bool c_Solver::MoverAwaitAndPclExchange(int cycle) {
  // ======= Phase 3A: await mover and compaction futures =======
  bool openBCAppendRejected = false;
  {
    profiling::Range wait(profiling::Region::MoverWait);
    for (int i = 0; i < ns; i++) {
      const int x = awaitTask(exitingResults[i], // holes
                              "particle mover of species " + std::to_string(i));
      const auto& appendCounters = moverCountersHostPtr[i].openBCAppendCounters;
      if (appendCounters.rejected != 0) {
        openBCAppendRejected = true;
        const uint32_t postAppendNOP = pclsArrayHostPtr[i]->getNOP();
        const uint32_t appendBase = postAppendNOP - appendCounters.accepted;
        std::cerr << "[OpenBC append overflow] rank=" << MPIdata::get_rank()
                  << " cycle=" << cycle << " species=" << i
                  << " base_nop=" << appendBase
                  << " capacity=" << pclsArrayHostPtr[i]->getCapacity()
                  << " accepted=" << appendCounters.accepted
                  << " rejected=" << appendCounters.rejected << std::endl;
      }
      stayedParticle[i] = pclsArrayHostPtr[i]->getNOP() - x;
    }
  }

  if (openBCAppendRejected) {
    std::cerr << "ERROR on rank " << MPIdata::get_rank()
              << ": the device particle arrays have no room left for the "
                 "particles that enter through the open-outflow faces; build "
                 "with a larger PARTICLE_ARRAY_HEADROOM (now "
              << ParticleArrayConfig::HEADROOM
              << ", the room before the first particle exchange) or "
                 "PARTICLE_ARRAY_RESIZE_THRESHOLD (now "
              << ParticleArrayConfig::RESIZE_THRESHOLD << ", after each one)"
              << std::endl;
#ifndef NO_MPI
    MPI_Abort(MPIdata::get_PicGlobalComm(), EXIT_FAILURE);
#endif
    std::abort();
  }

  // ======= Planet processing on the dedicated planet stream =======
  if (doPlanet_)
    processPlanetParticles();

  // ======= Phase 3B: sync, pre-expand SoA, update NOP, prepare sort =======
  // Pre-expand the SoA for stayed + injected + estimated MPI incoming, set
  // NOP, and launch the GPU injection kernels, which write the SoA tail.

  // Per-species injection count (0 unless this rank owns an open-inflow face
  // and the species has rhoINJECT > 0).
  // Declared at function scope so Phase 3G can also use it.
  std::vector<int> injectedBCS(ns, 0);
  for (int i = 0; i < ns; i++)
    injectedBCS[i] = injectionParams[i].totalInjected;
  {
    profiling::Range injection(profiling::Region::ParticleInjection);

    for (int i = 0; i < ns; i++) {
      // Sync compaction kernels so expand() can safely realloc.
      {
        profiling::Range wait(profiling::Region::CompactionWait, i);
        cudaErrChk(cudaStreamSynchronize(streams[i]));
      }

      // Estimate: stayed + injected + exiting-as-proxy-for-incoming.
      const uint32_t estimatedMPI = particlesCommInj[i]->getCommNOP();
      const uint32_t estimatedTotal =
          (uint32_t)stayedParticle[i] + injectedBCS[i] + estimatedMPI;

      // Pre-expand SoA if needed (one allocation, before any kernel launch).
      if ((estimatedTotal * ParticleArrayConfig::RESIZE_THRESHOLD) >=
          pclsArrayHostPtr[i]->getSize()) {
        pclsArrayHostPtr[i]->expand(
            static_cast<uint32_t>(estimatedTotal *
                                  ParticleArrayConfig::RESIZE_FACTOR),
            streams[i]);
        departureArrayHostPtr[i]->expand(pclsArrayHostPtr[i]->getSize(),
                                         streams[i]);
        cudaErrChk(cudaMemcpyAsync(
            departureArrayCUDAPtr[i], departureArrayHostPtr[i],
            sizeof(departureArrayType), cudaMemcpyDefault, streams[i]));
      }

      // Set host NOP to stayed + injected (injection count is known).
      // MPI particles will be appended on top in Phase 3G.
      pclsArrayHostPtr[i]->setNOE(stayedParticle[i] + injectedBCS[i]);
    }

    // ======= Launch GPU injection kernels (async on per-species stream)
    // =======
    for (int i = 0; i < ns; i++) {
      if (injectedBCS[i] > 0) {
        // Per-cycle seed: deterministic but different each cycle+species.
        unsigned long long rngSeed =
            (unsigned long long)cycle * 6364136223846793005ULL +
            (unsigned long long)i;

        // Sync host metadata to device (SoA pointers may have changed after
        // expand).
        cudaErrChk(cudaMemcpyAsync(pclsArrayCUDAPtr[i], pclsArrayHostPtr[i],
                                   sizeof(particleArrayCUDA), cudaMemcpyDefault,
                                   streams[i]));

        injectionKernel<<<getGridSize(injectedBCS[i], DEFAULT_BLOCK_SIZE),
                          DEFAULT_BLOCK_SIZE, 0, streams[i]>>>(
            pclsArrayCUDAPtr[i], injectionParams[i],
            (uint32_t)stayedParticle[i], rngSeed);
      }
    }
  }

  if (sortThisCycle_) {
    profiling::Range sortStart(profiling::Region::SortStart);
    for (int i = 0; i < ns; i++) {
      cellSorters[i].prepareBuffers(pclsArrayHostPtr[i], streams[i]);
    }

    // ======= Phase 3C: enqueue sort stages 1-3 for all species =======
    // Histogram, prefix sum, and sorted-index generation overlap with MPI
    // below.
    for (int i = 0; i < ns; i++) {
      cellSorters[i].enqueueSortAsync(
          pclsArrayHostPtr[i], gridCUDA,
          stayedParticle[i] + injectedBCS[i], // sort stayed + injected
          streams[i]);
    }
  }

  // ======= Phase 3D: perform MPI exchange while sort stages 1-3 run =======
  for (int i = 0; i < ns; i++) {
    profiling::Range mpi(profiling::Region::ParticleMPI, i);
    particlesCommInj[i]->separateAndSendParticles();
    particlesCommInj[i]->recommunicateParticlesUntilDone(1);
  }

  // ======= Exosphere ionization =======
  injectExosphereParticles();

  // ======= Phase 3E: finish sorting for all species =======
  // After this, the host particleArrayCUDA objects contain the sorted SoA
  // pointers.
  if (sortThisCycle_) {
    profiling::Range sortFinish(profiling::Region::SortFinish);
    for (int i = 0; i < ns; i++) {
#if defined(IPIC3D_GPU_CYCLE_DIAGNOSTICS)
      // Stages 1-3 have completed logically on this stream, and all their
      // independent buffers are still intact.  This collective diagnostic
      // synchronizes/reduces them and arms exact Stage-4 scratch poisoning.
      // Every rank participates, including ranks with no local particles.
      if (gpuCycleDiagnosticsStateCycle(cycle))
        gpuCycleDiagnostics_.reportCellSorterBeforeScatter(
            cycle, i, cellSorters[i], *pclsArrayHostPtr[i], gridCUDA, true,
            streams[i], vct->getParticleComm());
#endif
      cellSorters[i].finishSort(streams[i]);
#if defined(IPIC3D_GPU_CYCLE_DIAGNOSTICS)
      // Runs on the same stream after every real scatter kernel.  Reports
      // untouched poisoned slots, validates pointer rotation, and checks that
      // each sorted cell range contains particles classified to that cell.
      if (gpuCycleDiagnosticsStateCycle(cycle))
        gpuCycleDiagnostics_.reportCellSorterAfterScatter(
            cycle, i, cellSorters[i], *pclsArrayHostPtr[i], gridCUDA,
            streams[i], vct->getParticleComm());
#endif
      // Re-sync the device-side metadata after the host-side pointer swap.
      cudaErrChk(cudaMemcpyAsync(pclsArrayCUDAPtr[i], pclsArrayHostPtr[i],
                                 sizeof(particleArrayCUDA), cudaMemcpyDefault,
                                 streams[i]));
    }
  }

  // ======= Phase 3F: sync all streams before mutating SoA allocations =======
  // expand() may allocate, copy, and free SoA buffers, so no kernels may be
  // active.

  // Unsorted pipeline: wait for mover/compaction kernels before any expand() or
  // tail moments.
  {
    profiling::Range sync(profiling::Region::ParticleSync);
    for (int i = 0; i < ns; i++)
      cudaErrChk(cudaStreamSynchronize(streams[i]));
  }

  // ======= Phase 3G: expand, append incoming particles, and finalize moments
  // =======
  const auto momentGridSize = grid->getNXN() * grid->getNYN() * grid->getNZN();

  for (int i = 0; i < ns; i++) {
    profiling::Range append(profiling::Region::ParticleAppend, i);

    // Total particles = stayed + GPU-injected + MPI incoming.
    const int sortedCount = stayedParticle[i] + injectedBCS[i];
    auto newPclNum = sortedCount + particlesCommInj[i]->getCommNOP();

    // Expand SoA arrays if needed.
    // Safe point: Phase 3F guarantees no kernels are using these buffers.
    if ((newPclNum * ParticleArrayConfig::RESIZE_THRESHOLD) >=
        pclsArrayHostPtr[i]->getSize()) {
      pclsArrayHostPtr[i]->expand(
          newPclNum * ParticleArrayConfig::RESIZE_FACTOR, streams[i]);
      departureArrayHostPtr[i]->expand(pclsArrayHostPtr[i]->getSize(),
                                       streams[i]);
      cudaErrChk(cudaMemcpyAsync(
          departureArrayCUDAPtr[i], departureArrayHostPtr[i],
          sizeof(departureArrayType), cudaMemcpyDefault, streams[i]));
    }

    // Copy incoming AoS particles to the device staging buffer.
    const int incomingCount = particlesCommInj[i]->getCommNOP();
    if (incomingCount > 0) {
      if (static_cast<uint32_t>(incomingCount) >
          incomingStagingHostPtr[i]->getSize()) {
        incomingStagingHostPtr[i]->expand(
            incomingCount * ParticleArrayConfig::RESIZE_FACTOR, streams[i]);
        cudaErrChk(cudaMemcpyAsync(
            incomingStagingCUDAPtr[i], incomingStagingHostPtr[i],
            sizeof(arrayCUDA<SpeciesParticle>), cudaMemcpyDefault, streams[i]));
      }
      cudaErrChk(cudaMemcpyAsync(incomingStagingHostPtr[i]->getArray(),
                                 particlesCommInj[i]->getCommPclsData(),
                                 incomingCount * sizeof(SpeciesParticle),
                                 cudaMemcpyDefault, streams[i]));
    }

    // Update NOP to the new total and copy the host metadata back to the
    // device.
    pclsArrayHostPtr[i]->setNOE(newPclNum);
    cudaErrChk(cudaMemcpyAsync(pclsArrayCUDAPtr[i], pclsArrayHostPtr[i],
                               sizeof(particleArrayCUDA), cudaMemcpyDefault,
                               streams[i]));

    // Scatter the incoming AoS particles into the SoA tail after
    // stayed+injected.
    if (incomingCount > 0)
      scatterAoSToSoAKernel<<<getGridSize(incomingCount, DEFAULT_BLOCK_SIZE),
                              DEFAULT_BLOCK_SIZE, 0, streams[i]>>>(
          incomingStagingHostPtr[i]->getArray(), pclsArrayCUDAPtr[i],
          (uint32_t)sortedCount, (uint32_t)incomingCount,
          particlesHost[i]->getParticleIDGenerator());

#if defined(IPIC3D_GPU_CYCLE_DIAGNOSTICS)
    if (gpuCycleDiagnosticsStateCycle(cycle)) {
      const GPUCycleDiagnosticParticleBounds diagnosticBounds = {
          {grid->getXstart(), grid->getYstart(), grid->getZstart()},
          {grid->getXend(), grid->getYend(), grid->getZend()},
          {0.0, 0.0, 0.0},
          {col->getLx(), col->getLy(), col->getLz()}};

      // These calls are collective even when a rank has an empty range.  The
      // shared species stream makes the incoming AoS scatter visible before
      // the tail is inspected, and both inspections finish before deposition.
      gpuCycleDiagnostics_.reportParticleRange(
          "post_incoming_scatter_pre_moments", cycle, i, "sorted_prefix",
          *pclsArrayHostPtr[i], 0, static_cast<std::uint32_t>(sortedCount),
          diagnosticBounds, streams[i], vct->getParticleComm());
      gpuCycleDiagnostics_.reportParticleRange(
          "post_incoming_scatter_pre_moments", cycle, i, "incoming_tail",
          *pclsArrayHostPtr[i], static_cast<std::uint32_t>(sortedCount),
          static_cast<std::uint32_t>(incomingCount), diagnosticBounds,
          streams[i], vct->getParticleComm());
    }
#endif

    // Finalize moments according to the active sorted/unsorted pipeline.
    profiling::Range deposit(profiling::Region::MomentDeposit, i);
    if (sortThisCycle_) {
      // Sorted path: cell-aware moments for the stayed prefix, flat kernel for
      // the incoming tail.
      cudaErrChk(cudaMemsetAsync(momentsCUDAPtr[i], 0,
                                 momentGridSize * 10 * sizeof(cudaMomentType),
                                 streams[i]));

      const uint32_t numSorted = sortedCount;
      if (numSorted > 0) {
        const int numCells = cellSorters[i].getNumCells();
        const int threads = numCells * WARP_SIZE; // one warp per cell
        cellAwareMomentKernel<<<getGridSize(threads, DEFAULT_BLOCK_SIZE),
                                DEFAULT_BLOCK_SIZE, 0, streams[i]>>>(
            cellSorters[i].getCellStartOffsets(), numCells, numSorted,
            pclsArrayCUDAPtr[i], gridCUDA, momentsCUDAPtr[i]);
      }

      const int tailCount = newPclNum - sortedCount;
      if (tailCount > 0)
        momentKernelNew<<<getGridSize(tailCount, SMALL_BLOCK_SIZE),
                          SMALL_BLOCK_SIZE, 0, streams[i]>>>(
            pclsArrayCUDAPtr[i], gridCUDA, momentsCUDAPtr[i], sortedCount);
    } else {
      // Unsorted path: the stayed prefix was already handled in
      // cudaLauncherAsync(). Only the injected + incoming tail needs
      // momentKernelNew.
      const int tailCount = newPclNum - stayedParticle[i];
      if (tailCount > 0)
        momentKernelNew<<<getGridSize(tailCount, SMALL_BLOCK_SIZE),
                          SMALL_BLOCK_SIZE, 0, streams[i]>>>(
            pclsArrayCUDAPtr[i], gridCUDA, momentsCUDAPtr[i],
            stayedParticle[i]);
    }

    // Reset hashed sums and departure flags for the next cycle.
    for (int j = 0; j < departureArrayElementType::HASHED_SUM_NUM; j++)
      hashedSumArrayHostPtr[i][j].resetBucket();
    cudaErrChk(cudaMemcpyAsync(
        hashedSumArrayCUDAPtr[i], hashedSumArrayHostPtr[i],
        (departureArrayElementType::HASHED_SUM_NUM) * sizeof(hashedSum),
        cudaMemcpyDefault, streams[i]));
    cudaErrChk(cudaMemsetAsync(departureArrayHostPtr[i]->getArray(), 0,
                               departureArrayHostPtr[i]->getSize() *
                                   sizeof(departureArrayElementType),
                               streams[i]));
  }

  // the rest: the moment scatter of every species
  profiling::Range scatter(profiling::Region::MomentScatter);
  for (int i = 0; i < ns; i++) {
#if defined(IPIC3D_GPU_CYCLE_DIAGNOSTICS)
    if (gpuCycleDiagnosticsStateCycle(cycle)) {
      static const char* const momentNames[10] = {
          "rho", "Jx", "Jy", "Jz", "Pxx", "Pxy", "Pxz", "Pyy", "Pyz", "Pzz"};
      const cudaSolverType* packedMoments[10];
      for (int component = 0; component < 10; ++component)
        packedMoments[component] =
            momentsCUDAPtr[i] + component * momentGridSize;
      const double scales[10] = {1.0, 1.0, 1.0, 1.0, 1.0,
                                 1.0, 1.0, 1.0, 1.0, 1.0};
      const GPUCycleDiagnosticRegion nodeRegion = {
          grid->getNXN(),
          grid->getNYN(),
          grid->getNZN(),
          {0, 0, 0},
          {grid->getNXN(), grid->getNYN(), grid->getNZN()},
          {grid->getXN(0), grid->getYN(0), grid->getZN(0)},
          {grid->getDX(), grid->getDY(), grid->getDZ()}};
      gpuCycleDiagnostics_.reportArraySet(
          "moments_before_d2d_scatter", cycle, i, momentNames, packedMoments,
          scales, 10, nodeRegion, streams[i], vct->getParticleComm());
    }
#endif
    // Scatter the packed moments into the field solver's per-species arrays
    EMf->gpuScatterMomentsD2D(momentsCUDAPtr[i], i, streams[i]);
#if defined(IPIC3D_GPU_CYCLE_DIAGNOSTICS)
    if (gpuCycleDiagnosticsStateCycle(cycle))
      EMf->gpuDiagnosticReportSpeciesMoments("moments_after_d2d_scatter", cycle,
                                             i, streams[i]);
#endif
    // Cycle-end marker on streams[i]. Happens-after every in-place SoA write
    // of this cycle (compactParticles2 / scatterAoSToSoAKernel) and after the
    // moment kernels and the moment scatter on the same stream. Consumers:
    //   - MomentsAwait()    : per-species cudaEventSynchronize (host)
    //   - outputCopyAsync() : cudaStreamWaitEvent on outputStream
    //   - cudaLauncherAsync (next cycle, indirectly via eventOutputCopy)
    //   - ScheduleHeatFlux() and the data analysis (startAnalysis):
    //     cudaStreamWaitEvent on their streams, also after the sort of a
    //     checkpoint or of a capped particle output (outputCopyAsync), which
    //     records the event again
    cudaErrChk(cudaEventRecord(cycleEndEvent[i], streams[i]));
  }

  return (false);
}

/**
 * @brief Advance the magnetic field solver for one cycle.
 *
 * This assumes the electric field has already been updated for the same cycle.
 * @param cycle Simulation cycle being advanced.
 */
void c_Solver::CalculateB(int cycle) {
  // B(n+1) in d_Bxc, d_Bxn, ..., then E^n -> E(n+1) in d_Ex
  EMf->gpuCalculateB(cycle);
}

/**
 * @brief Synchronize moment accumulation and prepare field-side derived
 * quantities.
 *
 * After the per-species moment kernels and scatters complete, this method
 * schedules particle merging if it is on (after a mover only), communicates
 * ghost moments, applies special boundary charge sources, and computes the
 * field-solver derived moment quantities.
 */
void c_Solver::MomentsAwait(int cycle) {
  // Wait for all per-species moment kernels and scatters to finish.
  // cycleEndEvent[i] is recorded on streams[i] right after the scatter,
  // so a per-species cudaEventSynchronize is sufficient and avoids a global
  // device sync that would also drain unrelated streams (planetStream is
  // already host-synced inside processPlanetParticles; outputStream of the
  // previous cycle, if any, must NOT be waited on here — the next cycle's
  // mover/sort already explicitly waits on eventOutputCopy).
  {
    profiling::Range wait(profiling::Region::MomentKernelsWait);
    for (int i = 0; i < ns; ++i)
      cudaErrChk(cudaEventSynchronize(cycleEndEvent[i]));
  }

  // Not in the pre-loop call (cycle -1): before the first mover a planet rank
  // still holds the particles loaded inside the planet.
  if (col->getParticleMerging() && cycle >= 0)
    scheduleMerging();

  // Phase 1: ghost exchange of the species in batches of HALO_MAX_BATCH / 10
  // (2 halo exchanges per batch instead of 2 per species)
  EMf->gpuCommunicateGhostP2G_AllSpecies(cycle);

  // Phase 2: zero derived quantities, planet charge fix, outflow-layer
  // neutralization, sum over species, interp N→C
  {
    profiling::Range derived(profiling::Region::DerivedMoments);
    EMf->gpuSetZeroDerivedMoments();
    // Enforce constant charge inside planet BEFORE summing over species,
    // so that rhon (and downstream rhoc, rhoh) includes the planet fix.
    if (doPlanet_)
      EMf->gpuConstantChargePlanet();
    // Neutral charge, and the reference plane's current and second moments,
    // on the X-right outflow layer: the field solve sees a neutral layer,
    // whether its field is then copied from the reference plane or kept as
    // solved (OutflowLayerFields).
    EMf->gpuNeutralizeOutflowLayer();
    EMf->gpuCopyOutflowLayerMoments();
    EMf->gpuSumOverSpecies();
    if (col->getOutputConfig().needsJTotComputation())
      EMf->gpuSumOverSpeciesJ();
    EMf->gpuInterpDensitiesN2C();
  }

  // Phase 3: hat functions (Jhat, rhohat)
  EMf->gpuCalculateHatFunctions(cycle);

  // Record momentsPipelineDoneEvt at the tail of the solver stream so that
  // ScheduleHeatFlux() (next cycle, step 2) can gate its D2D copy of
  // d_rhons/d_Jxs/d_Jys/d_Jzs until the ghost exchange and hat functions
  // have committed all writes to those arrays.  Without this gate, the D2D
  // copy could read partially-updated ghost cells from gpuCommunicateGhostP2G.
  if (heatFluxCalculationEnabled_)
    cudaErrChk(cudaEventRecord(momentsPipelineDoneEvt, EMf->gpuSolverStream()));
}

/**
 * @brief Append the current per-species particle counts to the rank-local CSV
 * log.
 * @param cycle Simulation cycle associated with the recorded counts.
 */
void c_Solver::writeParticleNum(int cycle) {
#if IPIC3D_COMPILE_DISK_OUTPUT
  if (!pclNumCSV.is_open() || !pclNumCSV.good()) {
    // Reopen if stream is in a bad state
    pclNumCSV.close();
    pclNumCSV.clear();
    pclNumCSV.open(col->getSaveDirName() + "/particleNum" +
                       std::to_string(myrank) + ".csv",
                   std::ios::app);
  }
  pclNumCSV << cycle << ",";
  for (int i = 0; i < ns - 1; i++) {
    pclNumCSV << pclsArrayHostPtr[i]->getNOP() << ",";
  }
  pclNumCSV << pclsArrayHostPtr[ns - 1]->getNOP() << std::endl;
  pclNumCSV.flush();
#else
  (void)cycle;
  return;
#endif
}

/**
 * @brief Write restart, field, particle, and test-particle outputs for one
 * cycle.
 *
 * Particle and restart outputs wait on outputCopyAsync() so the host-side SoA
 * mirrors are up to date before the IOManager consumes them.
 * @param cycle Simulation cycle being written.
 */
void c_Solver::WriteOutput(int cycle) {
#if IPIC3D_COMPILE_DISK_OUTPUT
  // complete the deferred outputs written since the last cycle
  {
    profiling::Range poll(profiling::Region::OutputPoll);
    ioManager->poll();
  }

  const bool fieldOutputDue = ioManager->needsFieldOutput(cycle);
  const bool particleOutputDue = ioManager->needsParticleOutput(cycle);
  const bool restartOutputDue = ioManager->needsRestartParticleSync(cycle);

#ifdef USE_CATALYST
  const bool catalystNeedsHostData =
      Adaptor::RequestDataDescription(col->getDt() * cycle, cycle);
#endif

  // ---- Copy to the host the field arrays the host consumers read ----
  {
    // the field output (its tags) or a checkpoint (every array)
    FieldOutputArrays fields = ioManager->hostFields(cycle);
#ifdef USE_CATALYST
    if (catalystNeedsHostData)
      fields = FieldOutputArrays::all();
#endif
    // the energy diagnostics: E, the cell B, the species density and current
    // (the bulk energy)
    if (col->diagnostics_due(cycle))
      fields.E = fields.cellB = fields.speciesRho = fields.speciesJ = true;
    if (fields.any()) {
      // the deferred outputs still writing from the host arrays first
      ioManager->beforeFieldCopy();
      // gpuSolverSyncD2H internally synchronises the solver stream after
      // issuing all D2H copies, so no separate cudaStreamSynchronize is needed.
      EMf->gpuSolverSyncD2H(fields, EMf->gpuSolverStream());
    }
  }

#ifdef USE_CATALYST
  // RequestDataDescription() has already decided whether the GPU host mirror
  // needed refreshing.  This consumes that exact request without asking the
  // Catalyst pipeline a second time.
  {
    profiling::Range catalyst(profiling::Region::Catalyst);
    Adaptor::CoProcess(EMf);
  }
#endif

  {
    profiling::Range diagnostics(profiling::Region::Diagnostics);
    WriteConserved(cycle);
    WritePlanetDiagnostics(cycle);
  }

  // ======= Restart checkpoint =======
  if (restartOutputDue) {
    // Periodic restarts are written before the current iteration is completed.
    // The checkpoint label remains the triggering loop cycle, so a checkpoint
    // labeled N restarts by executing cycle N again: it holds E^N (d_Ex is
    // advanced only in CalculateB), B^N and the particles at N.
    // eventOutputCopy is pre-recorded once at init on outputStream and re-
    // recorded by every outputCopyAsync() that actually issues copies. The
    // wait below is therefore always well-defined. First-cycle output
    // is primed during Init() when host particle data is required.
    awaitOutputCopy();
    prepareActiveRestartParticleCellMetadata();
    // SoA data is already in host vectors after outputCopyAsync — no conversion
    // needed
    ioManager->writeRestart(cycle);
  }

  if (!Parameters::get_doWriteOutput())
    return;

  // ======= Field output =======
  if (fieldOutputDue) {
    finishHeatFluxForOutput(cycle);
    ioManager->writeFields(cycle);
  }

  // ======= Particle output =======
  if (particleOutputDue) {
    // See comment above: the event is pre-recorded so this is always safe.
    awaitOutputCopy();
    // SoA data is already in host vectors after outputCopyAsync — no conversion
    // needed
    ioManager->writeParticles(cycle);
  }

  // ======= Test-particle output =======
  if (nstestpart > 0 && !col->testparticle_output_is_off() &&
      cycle % col->getTestParticlesOutputCycle() == 0) {
    ioManager->writeTestParticles(cycle);
  }
#else
  (void)cycle;
  return;
#endif
}

void c_Solver::ensureRestartParticleCellMetadataBuffers() {
  const int guardedNx = grid->getNXC();
  const int guardedNy = grid->getNYC();
  const int guardedNz = grid->getNZC();
  const int guardedCells = guardedNx * guardedNy * guardedNz;

  restartGuardedCellOffsets_.resize(ns);
  restartGuardedCellCounts_.resize(ns);
  for (int s = 0; s < ns; ++s) {
    restartGuardedCellOffsets_[s].resize(guardedCells);
    restartGuardedCellCounts_[s].resize(guardedCells);
  }

  restartParticleCellMetadata_.resize(ns, guardedNx - 2, guardedNy - 2,
                                      guardedNz - 2);
}

void c_Solver::copyRestartParticleCellMetadataFromDevice(int species,
                                                         cudaStream_t stream) {
  const int guardedCells = grid->getNXC() * grid->getNYC() * grid->getNZC();
  if (pclsArrayHostPtr[species]->getNOP() == 0) {
    std::fill(restartGuardedCellOffsets_[species].begin(),
              restartGuardedCellOffsets_[species].end(), 0);
    std::fill(restartGuardedCellCounts_[species].begin(),
              restartGuardedCellCounts_[species].end(), 0);
    return;
  }

  cudaErrChk(cudaMemcpyAsync(restartGuardedCellOffsets_[species].data(),
                             cellSorters[species].getCellStartOffsets(),
                             guardedCells * sizeof(int), cudaMemcpyDeviceToHost,
                             stream));
  cudaErrChk(cudaMemcpyAsync(restartGuardedCellCounts_[species].data(),
                             cellSorters[species].getCellCounts(),
                             guardedCells * sizeof(int), cudaMemcpyDeviceToHost,
                             stream));
}

void c_Solver::prepareActiveRestartParticleCellMetadata() {
  ensureRestartParticleCellMetadataBuffers();

  const int guardedNx = grid->getNXC();
  const int guardedNy = grid->getNYC();
  const int guardedNz = grid->getNZC();
  const int expectedActiveCells =
      restartParticleCellMetadata_.activeCellCount();

  for (int s = 0; s < ns; ++s) {
    auto& activeOffsets = restartParticleCellMetadata_.species[s].cellOffsets;
    auto& activeCounts = restartParticleCellMetadata_.species[s].cellCounts;

    int activeIndex = 0;
    for (int gz = 1; gz < guardedNz - 1; ++gz) {
      for (int gy = 1; gy < guardedNy - 1; ++gy) {
        for (int gx = 1; gx < guardedNx - 1; ++gx) {
          const int guardedIndex =
              gx + gy * guardedNx + gz * guardedNx * guardedNy;
          activeOffsets[activeIndex] =
              restartGuardedCellOffsets_[s][guardedIndex];
          activeCounts[activeIndex] =
              restartGuardedCellCounts_[s][guardedIndex];
          ++activeIndex;
        }
      }
    }
    if (activeIndex != expectedActiveCells) {
      eprintf("Restart particle active-cell metadata size mismatch");
    }
  }

  restartParticleCellMetadata_.valid = true;
}

void c_Solver::awaitOutputCopy() {
  profiling::Range range(profiling::Region::OutputCopyWait);
  cudaErrChk(cudaEventSynchronize(eventOutputCopy));
}

/**
 * @brief Schedule the asynchronous GPU-to-host particle copies of the next
 * cycle's outputs, and the particle sums of its energy diagnostics.
 *
 * The main loop calls this at the end of cycle, for cycle + 1, Init for the
 * first cycle; Finalize with finalCheckpoint, for the final checkpoint only
 * (every array of every particle).
 * @param cycle Current simulation cycle (not read with finalCheckpoint).
 * @param finalCheckpoint Only the copy of the final checkpoint.
 */
void c_Solver::outputCopyAsync(int cycle, bool finalCheckpoint) {
#if IPIC3D_COMPILE_DISK_OUTPUT
  const int next = cycle + 1;
  // Return before issuing any work unless the next cycle reads host
  // particle data or the particle sums.
  const bool diagnostics = !finalCheckpoint && col->diagnostics_due(next);
  const bool hostCopy = finalCheckpoint || ioManager->needsParticleSync(next);
  if (!diagnostics && !hostCopy)
    return;
  // the deferred outputs still writing from the host particle buffers, the
  // samples and the cell metadata first
  if (hostCopy)
    ioManager->beforeParticleCopy();
  const bool restartSync =
      finalCheckpoint || ioManager->needsRestartParticleSync(next);
  // every particle of the arrays the outputs read; the particles selected of
  // the species with ParticlesOutputTrackedFraction or
  // ParticlesOutputMaxPerCell
  std::vector<ParticleOutputArrays> host(ns), sampled(ns);
  if (finalCheckpoint) {
    host.assign(ns, ParticleOutputArrays::all());
  } else if (hostCopy) {
    host = ioManager->hostParticles(next);
    sampled = ioManager->sampledParticles(next);
  }
  if (restartSync)
    ensureRestartParticleCellMetadataBuffers();

  // Explicit edge: wait for every species' end-of-cycle event before reading
  // particle SoA from device. cycleEndEvent[s] is recorded on streams[s] at
  // the end of MoverAwaitAndPclExchange and so happens-after every in-place
  // SoA write of the current cycle. First-cycle output priming uses the
  // events pre-recorded during CUDA initialization. Using a
  // dedicated outputStream isolates the D->H copies from the per-species
  // streams used by the next cycle's mover/sort, so those kernels can run
  // ahead in parallel with output as long as they wait on eventOutputCopy.
  // A checkpoint (its cell metadata) and the cap of the particles written
  // per cell need the species sorted by cell: the cycle's sort leaves the
  // particles that arrived unsorted. The fraction needs no sort.
  for (int s = 0; s < ns; ++s) {
    const bool sample = sampled[s].any();
    const bool capped = sample && col->particleOutputMaxPerCell(s) > 0;
    if (restartSync || sample) {
      cudaErrChk(cudaStreamWaitEvent(streams[s], cycleEndEvent[s], 0));
      cudaErrChk(cudaStreamWaitEvent(streams[s], eventOutputCopy, 0));

      if ((restartSync || capped) && pclsArrayHostPtr[s]->getNOP() > 0)
        sortSpeciesByCell(s);

      if (restartSync)
        copyRestartParticleCellMetadataFromDevice(s, streams[s]);
      if (sample)
        countSampledParticlesAsync(s);
      cudaErrChk(cudaEventRecord(cycleEndEvent[s], streams[s]));
    }
    cudaErrChk(cudaStreamWaitEvent(outputStream, cycleEndEvent[s], 0));
  }

  // the full copies first: queued behind the counts too (cycleEndEvent),
  // they start as soon as these end, while the host sizes the samples
  {
    profiling::Range copies(profiling::Region::ParticleCopyD2H);
    if (diagnostics)
      sumParticlesAsync();
    copyParticlesToHostAsync(host);
    copySampledParticlesAsync(sampled);
  }
  cudaErrChk(cudaEventRecord(eventOutputCopy, outputStream));
#else
  (void)cycle;
  (void)finalCheckpoint;
  return;
#endif
}

void c_Solver::copyParticlesToHostAsync(
    const std::vector<ParticleOutputArrays>& host) {
  // Direct GPU SoA -> host SoA transfers of the arrays of every particle
  // that the outputs of the cycle read (IOManager::hostParticles): the mover
  // waits for them. The other host arrays stay as they were, read by no
  // output then.
  for (int i = 0; i < ns; i++) {
    const ParticleOutputArrays& arrays = host[i];
    if (!arrays.any())
      continue;
    const uint32_t nop = pclsArrayHostPtr[i]->getNOP();
    // Resize the host SoA vectors to receive the current particle count.
    ParticleSoAHost& to = *particlesHost[i];
    particleArrayCUDA& from = *pclsArrayHostPtr[i];
    to.prepareSoAForNOP(nop);
    if (nop == 0)
      continue;
    auto copy = [&](auto* dst, const auto* src) {
      cudaErrChk(cudaMemcpyAsync(dst, src, nop * sizeof(*src),
                                 cudaMemcpyDefault, outputStream));
    };
    if (arrays.velocity) {
      copy(to.getUallMut(), from.getU());
      copy(to.getVallMut(), from.getV());
      copy(to.getWallMut(), from.getW());
    }
    if (arrays.charge)
      copy(to.getQallMut(), from.getQ());
    if (arrays.position) {
      copy(to.getXallMut(), from.getX());
      copy(to.getYallMut(), from.getY());
      copy(to.getZallMut(), from.getZ());
    }
    if (arrays.id && to.tracksParticleID())
      copy(to.getIDallMut(), from.getID());
  }
}

namespace {

/** The keys of the particles p of species s: of their IDs if they have
 *  some, else of their coordinates. */
ParticleSelectKeys selectKeys(particleArrayCUDA& p, int s) {
  return {p.tracksParticleID() ? p.getID() : nullptr,
          p.getX(),
          p.getY(),
          p.getZ(),
          p.getU(),
          PARTICLE_SELECT_SALT * (std::uint64_t)(s + 1)};
}

// the cap scans its cells into the sort's block sums
static_assert(kSelectScanBlock >= SORT_SCAN_ELEMENTS_PER_BLOCK,
              "CellSorter::block_sums holds the block totals of the cap");

} // namespace

// The particles of a species that its particle output writes
// (particleOutputSelect.cuh). With a cap, from the species sorted by cell:
// the sort's cell counts and starts, its free buffers for the rest
// (cell_offsets: the first output slot of each cell, sorted_indices: the
// indices written, scratch: one array at a time). With the fraction, from
// the particles in their order, with buffers of its own.
void c_Solver::countSampledParticlesAsync(int s) {
  // on the species' stream (after its sort with a cap): only these wait on
  // the host
  particleSampleCountHostPtr[s] = 0;
  particleArrayCUDA& pcls = *pclsArrayHostPtr[s];
  const uint32_t nop = pcls.getNOP();
  const int maxPerCell = col->particleOutputMaxPerCell(s);
  const double fraction = col->particleOutputFraction(s);
  if (nop == 0 || (maxPerCell == 0 && fraction == 0.0)) // a fraction 0: none
    return;
  if (maxPerCell > 0) {
    CellSorter& sorter = cellSorters[s];
    launchCapCount(sorter.getCellCounts(), sorter.getNumCells(), maxPerCell,
                   sorter.buffers.cell_offsets, sorter.buffers.block_sums,
                   particleSampleCountCUDAPtr + s, streams[s]);
  } else {
    GrowingDeviceArray<int>& blockSums = particleFractionBlockSums[s];
    blockSums.reserve((nop + kSelectScanBlock - 1) / kSelectScanBlock);
    launchFractionCount(selectKeys(pcls, s), nop, selectLimit(fraction),
                        blockSums.ptr, particleSampleCountCUDAPtr + s,
                        streams[s]);
  }
  cudaErrChk(cudaMemcpyAsync(particleSampleCountHostPtr + s,
                             particleSampleCountCUDAPtr + s, sizeof(int),
                             cudaMemcpyDeviceToHost, streams[s]));
}

void c_Solver::copySampledParticlesAsync(
    const std::vector<ParticleOutputArrays>& sampled) {
  // the numbers of particles written, which size the samples and the
  // buffers of the fraction (grown once, for every species)
  std::uint32_t largestFraction = 0;
  for (int s = 0; s < ns; s++) {
    if (!sampled[s].any())
      continue;
    {
      profiling::Range wait(profiling::Region::SampleCountWait, s);
      cudaErrChk(cudaStreamSynchronize(streams[s])); // the count
    }
    const std::uint32_t n = particleSampleCountHostPtr[s];
    ioManager->particleSample(s).prepare(n, sampled[s],
                                         particlesHost[s]->tracksParticleID());
    if (col->particleOutputMaxPerCell(s) == 0)
      largestFraction = std::max(largestFraction, n);
  }
  particleFractionIdx.reserve(largestFraction);
  particleFractionStage.reserve(largestFraction);

  // per species the indices of the particles written (with a cap also
  // their charges), then their arrays through a device stage; the species
  // run in turn on outputStream, so they share the buffers of the fraction
  for (int s = 0; s < ns; s++) {
    const ParticleOutputArrays& arrays = sampled[s];
    const std::uint32_t n = particleSampleCountHostPtr[s];
    if (!arrays.any() || n == 0)
      continue;
    ParticleSample& sample = ioManager->particleSample(s);
    particleArrayCUDA& pcls = *pclsArrayHostPtr[s];
    const ParticleSelectKeys keys = selectKeys(pcls, s);
    const int maxPerCell = col->particleOutputMaxPerCell(s);
    CellSorter& sorter = cellSorters[s];
    // the cap: the sort's free buffers (scratch: the capacity of a SoA field)
    std::uint32_t* const idx = maxPerCell > 0 ? sorter.buffers.sorted_indices
                                              : particleFractionIdx.ptr;
    void* const stage =
        maxPerCell > 0 ? sorter.scratch.ptr : (void*)particleFractionStage.ptr;
    cudaPclType_Q* const qStage =
        arrays.charge ? static_cast<cudaPclType_Q*>(stage) : nullptr;
    if (maxPerCell > 0) { // the charges of each cell
      launchCapSelect(sorter.getCellCounts(), sorter.getCellStartOffsets(),
                      sorter.buffers.cell_offsets, sorter.getNumCells(),
                      maxPerCell, keys, pcls.getQ(), idx, qStage, outputStream);
    } else { // the fraction: charges q / f
      const double fraction = col->particleOutputFraction(s);
      launchFractionSelect(keys, pcls.getNOP(), selectLimit(fraction),
                           particleFractionBlockSums[s].ptr, idx, outputStream);
      if (qStage)
        launchGatherScaled(qStage, pcls.getQ(), idx, n, 1.0 / fraction,
                           outputStream);
    }
    auto toHost = [&](auto* dst) {
      cudaErrChk(cudaMemcpyAsync(dst, stage, n * sizeof(*dst),
                                 cudaMemcpyDeviceToHost, outputStream));
    };
    auto copy = [&](auto* dst, const auto* src) {
      using T = std::remove_const_t<std::remove_pointer_t<decltype(src)>>;
      static_assert(sizeof(T) <= sizeof(double), "the stage holds doubles");
      launchGather<T>(static_cast<T*>(stage), src, idx, n, outputStream);
      toHost(dst);
    };
    if (qStage)
      toHost(sample.q());
    if (arrays.position) {
      copy(sample.x(), pcls.getX());
      copy(sample.y(), pcls.getY());
      copy(sample.z(), pcls.getZ());
    }
    if (arrays.velocity) {
      copy(sample.u(), pcls.getU());
      copy(sample.v(), pcls.getV());
      copy(sample.w(), pcls.getW());
    }
    if (arrays.id) // only for a species with IDs (IOManager::sampledParticles)
      copy(sample.id(), pcls.getID());
  }
}

void c_Solver::sumParticlesAsync() {
  cudaErrChk(cudaMemsetAsync(particleSumsCUDAPtr, 0, ns * sizeof(ParticleSums),
                             outputStream));
  for (int s = 0; s < ns; s++) {
    particleArrayCUDA& pcls = *pclsArrayHostPtr[s];
    // the relativistic push of the planet cases (moverSubcyclesKernel)
    launchParticleSums(pcls.getU(), pcls.getV(), pcls.getW(), pcls.getQ(),
                       pcls.getNOP(), col->getQOM(s), col->getC(), doPlanet_,
                       particleSumsCUDAPtr + s, outputStream);
  }
  cudaErrChk(cudaMemcpyAsync(particleSumsHostPtr, particleSumsCUDAPtr,
                             ns * sizeof(ParticleSums), cudaMemcpyDeviceToHost,
                             outputStream));
}

/**
 * @brief Append conserved-quantity diagnostics to the rank-selected text file.
 * @param cycle Simulation cycle being written.
 */
void c_Solver::WriteConserved(int cycle) {
#if IPIC3D_COMPILE_DISK_OUTPUT
  if (col->diagnostics_due(cycle)) {
    // The particle sums of the cycle, computed on the GPU by
    // outputCopyAsync(cycle - 1) on outputStream (primed in Init for the
    // first cycle): eventOutputCopy follows them. The mover launched just
    // before WriteOutput() only waits for it on the GPU, so wait here.
    awaitOutputCopy();
    std::vector<ParticleSums> sums(ns);
    MPI_Allreduce(particleSumsHostPtr, sums.data(), kParticleSumsValues * ns,
                  MPI_DOUBLE, MPI_SUM, vct->getParticleComm());
    Eenergy = EMf->getEenergy();
    Benergy = EMf->getBenergy();
    TOTenergy = 0.0;
    double TOTcharge = 0.0;
    double P[3] = {0.0, 0.0, 0.0}; // total momentum vector
    auto norm = [](const double a[3]) {
      return std::sqrt(a[0] * a[0] + a[1] * a[1] + a[2] * a[2]);
    };
    for (int is = 0; is < ns; is++) {
      Ke[is] = sums[is].energy;
      BulkEnergy[is] = EMf->getBulkEnergy(is);
      TOTenergy += Ke[is];
      momentum[is] = norm(sums[is].momentum);
      for (int c = 0; c < 3; c++)
        P[c] += sums[is].momentum[c];
      Qtot[is] = sums[is].charge;
      TOTcharge += Qtot[is];
    }
    TOTmomentum = norm(P);
    if (myrank == (nprocs - 1)) {
      const int cw = 16; // column width
      std::ostringstream h;
      h << std::left << std::setw(8) << "Cycle" << std::setw(cw)
        << "Total_Energy" << std::setw(cw) << "Momentum" << std::setw(cw)
        << "Eenergy" << std::setw(cw) << "Benergy" << std::setw(cw) << "Kenergy"
        << std::setw(cw) << "Total_Charge";
      for (int is = 0; is < ns; is++)
        h << std::setw(cw) << ("Ke[" + std::to_string(is) + "]");
      for (int is = 0; is < ns; is++)
        h << std::setw(cw) << ("BulkE[" + std::to_string(is) + "]");
      for (int is = 0; is < ns; is++)
        h << std::setw(cw) << ("Mom[" + std::to_string(is) + "]");
      for (int is = 0; is < ns; is++)
        h << std::setw(cw) << ("Q[" + std::to_string(is) + "]");
      h << std::setw(cw) << "Px" << std::setw(cw) << "Py" << std::setw(cw)
        << "Pz";
      const std::string header = h.str();
      // The header starts a new file: a fresh run, or a restart without the
      // file.  A file with other columns (a restart of a run written by
      // another version) is kept under another name, not appended to.
      std::string existingHeader;
      {
        std::ifstream existing(cq);
        std::getline(existing, existingHeader);
      }
      if (!existingHeader.empty() && existingHeader != header) {
        const std::string kept = cq + ".before_cycle_" + std::to_string(cycle);
        std::rename(cq.c_str(), kept.c_str());
        std::cout << "ConservedQuantities.txt had other columns; kept as "
                  << kept << std::endl;
        existingHeader.clear();
      }
      ofstream my_file(cq.c_str(), fstream::app);
      my_file << std::scientific << std::setprecision(6);
      if (existingHeader.empty())
        my_file << header << endl;
      my_file << std::left << std::setw(8) << cycle << std::setw(cw)
              << (Eenergy + Benergy + TOTenergy) << std::setw(cw) << TOTmomentum
              << std::setw(cw) << Eenergy << std::setw(cw) << Benergy
              << std::setw(cw) << TOTenergy << std::setw(cw) << TOTcharge;
      for (int is = 0; is < ns; is++)
        my_file << std::setw(cw) << Ke[is];
      for (int is = 0; is < ns; is++)
        my_file << std::setw(cw) << BulkEnergy[is];
      for (int is = 0; is < ns; is++)
        my_file << std::setw(cw) << momentum[is];
      for (int is = 0; is < ns; is++)
        my_file << std::setw(cw) << Qtot[is];
      for (int c = 0; c < 3; c++)
        my_file << std::setw(cw) << P[c];
      my_file << endl;
      my_file.close();
    }
  }
#else
  (void)cycle;
  return;
#endif
}

/**
 * @brief Append planet boundary-condition diagnostics to PlanetBC.txt.
 *
 * The planet kernels accumulate the diagnostics on the device as they run, so
 * the per-cycle cost is a few extra atomics. Everything else is paid here,
 * once every planetDiagCycle cycles: one small D2H copy, two MPI reductions
 * and one line written by rank 0. This runs from WriteOutput(), while the
 * mover kernels of the cycle are in flight; the device accumulators are
 * cumulative and differenced on the host, so no kernel (memset) has to queue
 * behind the mover. A row written at cycle N covers the planet hits processed
 * since the previous row, up to cycle N-1. At the end of the run Finalize()
 * writes the hits not yet reported in one more row, labelled with the resume
 * cycle (last cycle + 1) like the final restart checkpoint.
 *
 * Columns, per species s (charges are |q| sums in code units, as Q[s] in
 * ConservedQuantities.txt):
 *  - Nhit[s], Qhit[s]: macroparticles / charge that entered the planet;
 *  - Nabs[s], Qabs[s]: absorbed by the surface (all ions; electrons above the
 *    energy cutoff); the difference was reflected.
 * Global columns:
 *  - Qres_sum/max/min: charge carried to the next cycle, summed over ranks and
 *    its extremes over ranks (> 0: ion charge not yet neutralised);
 *  - Ecut_mean: mean per-electron energy separating absorbed from reflected
 *    electrons, an estimate of the surface potential barrier (nan if none);
 *  - RankCycles: rank-cycles in which something hit the planet;
 *  - ElecLimited: rank-cycles in which every electron that hit was absorbed
 *    and positive charge was still left.
 * @param cycle Simulation cycle being written.
 * @param endOfRun Write regardless of the cadence (end of the run).
 */
void c_Solver::WritePlanetDiagnostics(int cycle, bool endOfRun) {
#if IPIC3D_COMPILE_DISK_OUTPUT
  // planetDiagCyclesAccumulated is identical on all ranks, so every rank takes
  // the same branch before the collective calls below.
  if (planetDiagCycle <= 0 || planetDiagCyclesAccumulated == 0 ||
      (!endOfRun && cycle % planetDiagCycle != 0))
    return;

  // ======= Snapshot the cumulative device accumulators =======
  const int nCharge = ns + planetElecSpeciesCount;
  cudaErrChk(cudaMemcpyAsync(planetDiagChargeHost, planetDiagChargeDevice,
                             nCharge * sizeof(cudaParticleType),
                             cudaMemcpyDeviceToHost, planetStream));
  cudaErrChk(cudaMemcpyAsync(planetBCStateHost, planetBCStateDevice,
                             sizeof(PlanetBCState), cudaMemcpyDeviceToHost,
                             planetStream));
  cudaErrChk(cudaStreamSynchronize(planetStream));

  // Window values = current snapshot - previous snapshot.
  std::vector<double> charge(nCharge);
  for (int k = 0; k < nCharge; k++) {
    charge[k] = planetDiagChargeHost[k] - planetDiagChargePrev[k];
    planetDiagChargePrev[k] = planetDiagChargeHost[k];
  }
  const PlanetBCState& cur = *planetBCStateHost;
  const double cutoffEnergySum =
      cur.cutoffEnergySum - planetBCStatePrev.cutoffEnergySum;
  // Unsigned differences stay correct across a counter wrap-around.
  const uint32_t cutoffSamples =
      cur.cutoffSamples - planetBCStatePrev.cutoffSamples;
  const uint32_t electronLimitedCycles =
      cur.electronLimitedCycles - planetBCStatePrev.electronLimitedCycles;
  planetBCStatePrev = cur;

  // ======= Pack the local values and reduce them over ranks =======
  // Layout: [Nhit(ns) | Nabs(ns) | Qhit(ns) | Qabs(ns) | Qres, EcutSum,
  //          EcutSamples, RankCycles, ElecLimited]
  const int nTail = 5;
  std::vector<double> local(4 * ns + nTail, 0.0);
  std::vector<double> global(4 * ns + nTail, 0.0);
  for (int is = 0; is < ns; is++) {
    local[is] = planetDiagHitCount[is];
    local[ns + is] = planetDiagAbsorbedCount[is];
    local[2 * ns + is] = charge[is];
    if (col->getQOM(is) > 0) // ions are always absorbed
      local[3 * ns + is] = charge[is];
  }
  for (int e = 0; e < planetElecSpeciesCount; e++)
    local[3 * ns + planetElecSpeciesMap[e]] = charge[ns + e];
  const double residual = cur.residualCharge;
  double* tail = local.data() + 4 * ns;
  tail[0] = residual;
  tail[1] = cutoffEnergySum;
  tail[2] = cutoffSamples;
  tail[3] = planetDiagRankCycles;
  tail[4] = electronLimitedCycles;
  double residualExtremes[2] = {residual, -residual};
  double globalExtremes[2] = {residual, -residual};
#ifndef NO_MPI
  MPI_Reduce(local.data(), global.data(), (int)local.size(), MPI_DOUBLE,
             MPI_SUM, 0, MPIdata::get_PicGlobalComm());
  MPI_Reduce(residualExtremes, globalExtremes, 2, MPI_DOUBLE, MPI_MAX, 0,
             MPIdata::get_PicGlobalComm());
#else
  global = local;
#endif

  const int nCycles = planetDiagCyclesAccumulated;
  std::fill(planetDiagHitCount.begin(), planetDiagHitCount.end(), 0.0);
  std::fill(planetDiagAbsorbedCount.begin(), planetDiagAbsorbedCount.end(),
            0.0);
  planetDiagRankCycles = 0.0;
  planetDiagCyclesAccumulated = 0;

  // ======= Rank 0 appends one row =======
  if (myrank == 0) {
    const int cw = 16; // column width
    const double* gtail = global.data() + 4 * ns;
    const double cutoffMean = gtail[2] > 0
                                  ? gtail[1] / gtail[2]
                                  : std::numeric_limits<double>::quiet_NaN();
    ofstream my_file(planetDiagFile.c_str(), fstream::app);
    my_file << std::left << std::setw(8) << cycle << std::setw(10) << nCycles;
    for (int k = 0; k < 2 * ns; k++) // counts
      my_file << std::setw(cw) << static_cast<long long>(global[k]);
    my_file << std::scientific << std::setprecision(6);
    for (int k = 2 * ns; k < 4 * ns; k++) // charges
      my_file << std::setw(cw) << global[k];
    my_file << std::setw(cw) << gtail[0] << std::setw(cw) << globalExtremes[0]
            << std::setw(cw) << -globalExtremes[1] << std::setw(cw)
            << cutoffMean << std::setw(cw) << static_cast<long long>(gtail[3])
            << std::setw(cw) << static_cast<long long>(gtail[4]) << endl;
  }
#else
  (void)cycle;
  (void)endOfRun;
#endif
}

/**
 * @brief Write one velocity-distribution snapshot for every species.
 *
 * Output cadence is controlled by the caller; this method only performs the
 * per-species histogram calculation and file append.
 * @param cycle Simulation cycle being written.
 */
void c_Solver::WriteVelocityDistribution(int cycle) {
#if IPIC3D_COMPILE_DISK_OUTPUT
  for (int is = 0; is < ns; is++) {
    double maxVel = particlesHost[is]->getMaxVelocity();
    long long* VelocityDist =
        particlesHost[is]->getVelocityDistribution(nDistributionBins, maxVel);
    if (myrank == 0) {
      ofstream my_file(ds.c_str(), fstream::app);
      my_file << cycle << "\t" << is << "\t" << maxVel;
      for (int i = 0; i < nDistributionBins; i++)
        my_file << "\t" << VelocityDist[i];
      my_file << endl;
      my_file.close();
    }
    delete[] VelocityDist;
  }
#else
  (void)cycle;
  return;
#endif
}

/**
 * @brief Write field and moment traces at a regular grid of virtual satellite
 * points.
 *
 * The trace file samples electromagnetic fields, selected current sums, and
 * charge densities at `nsat^3` probe locations in the local domain.
 */
void c_Solver::WriteVirtualSatelliteTraces() {
#if IPIC3D_COMPILE_DISK_OUTPUT
  if (ns <= 2)
    return;
  assert_eq(ns, 4);

  ofstream my_file(cqsat.c_str(), fstream::app);
  const int nx0 = grid->get_nxc_r();
  const int ny0 = grid->get_nyc_r();
  const int nz0 = grid->get_nzc_r();
  for (int isat = 0; isat < nsat; isat++) {
    for (int jsat = 0; jsat < nsat; jsat++) {
      for (int ksat = 0; ksat < nsat; ksat++) {
        int index1 = 1 + isat * nx0 / nsat + nx0 / nsat / 2;
        int index2 = 1 + jsat * ny0 / nsat + ny0 / nsat / 2;
        int index3 = 1 + ksat * nz0 / nsat + nz0 / nsat / 2;
        my_file << EMf->getBx(index1, index2, index3) << "\t"
                << EMf->getBy(index1, index2, index3) << "\t"
                << EMf->getBz(index1, index2, index3) << "\t";
        my_file << EMf->getEx(index1, index2, index3) << "\t"
                << EMf->getEy(index1, index2, index3) << "\t"
                << EMf->getEz(index1, index2, index3) << "\t";
        my_file << EMf->getJxs(index1, index2, index3, 0) +
                       EMf->getJxs(index1, index2, index3, 2)
                << "\t"
                << EMf->getJys(index1, index2, index3, 0) +
                       EMf->getJys(index1, index2, index3, 2)
                << "\t"
                << EMf->getJzs(index1, index2, index3, 0) +
                       EMf->getJzs(index1, index2, index3, 2)
                << "\t";
        my_file << EMf->getJxs(index1, index2, index3, 1) +
                       EMf->getJxs(index1, index2, index3, 3)
                << "\t"
                << EMf->getJys(index1, index2, index3, 1) +
                       EMf->getJys(index1, index2, index3, 3)
                << "\t"
                << EMf->getJzs(index1, index2, index3, 1) +
                       EMf->getJzs(index1, index2, index3, 3)
                << "\t";
        my_file << EMf->getRHOns(index1, index2, index3, 0) +
                       EMf->getRHOns(index1, index2, index3, 2)
                << "\t";
        my_file << EMf->getRHOns(index1, index2, index3, 1) +
                       EMf->getRHOns(index1, index2, index3, 3)
                << "\t";
      }
    }
  }
  my_file << endl;
  my_file.close();
#else
  return;
#endif
}

/**
 * @brief Finalize output backends, optionally write the last restart, and free
 * CUDA state.
 */
void c_Solver::Finalize() {
  profiling::Range range(profiling::Region::Finalize);

#if IPIC3D_COMPILE_DISK_OUTPUT
  pclNumCSV.close();

  // Planet hits processed since the last PlanetBC.txt row, the last cycle's
  // included, labelled with the resume cycle like the final checkpoint.
  {
    profiling::Range diagnostics(profiling::Region::Diagnostics);
    WritePlanetDiagnostics(col->getNcycles() + first_cycle, true);
  }

  if (ioManager &&
      ioManager->getRestartBackend() != IOManager::RestartBackend::NONE &&
      col->getCallFinalize() && Parameters::get_doWriteOutput() &&
      col->getRestartOutputCycle() > 0) {
    // Sync EM fields from device to host before final restart write
    ioManager->beforeFieldCopy();
    EMf->gpuSolverSyncD2H(FieldOutputArrays::all(), EMf->gpuSolverStream());

    outputCopyAsync(LastCycle() - 1, true);
    awaitOutputCopy();
    // Final restart data is written after the last loop iteration has
    // completed, so the resume label is the next loop cycle.
    prepareActiveRestartParticleCellMetadata();
    ioManager->writeRestart(col->getNcycles() + first_cycle);
  }

  if (ioManager) {
    profiling::Range output(profiling::Region::FinalizeOutput);
    ioManager->finalize();
  }
#endif

  profiling::Range device(profiling::Region::FinalizeDevice);
  deInitCUDA();
}

/**
 * @brief Sort the host particle SoA of every species by cell (serial
 * counting sort).
 *
 * This path is separate from the GPU cell sorter and only touches the host
 * ParticleSoAHost containers.  Not used: the GPU cell sort orders the
 * particles.
 */
void c_Solver::sortParticles() {

  for (int species_idx = 0; species_idx < ns; species_idx++)
    particlesHost[species_idx]->sort_particles_serial();
}

/**
 * @brief Sort the particles of a species by cell on its stream.
 *
 * The sorters are created in initCUDA() only with periodic sorting
 * (SortingCycle > 0); the other callers (analysis cycles, restart output,
 * particle merging) create theirs here on first use.
 */
void c_Solver::sortSpeciesByCell(int species) {
  profiling::Range range(profiling::Region::ParticleSort, species);
  particleArrayCUDA* pcls = pclsArrayHostPtr[species];
  CellSorter& sorter = cellSorters[species];
  if (!sorter.initialized)
    sorter.init(gridCUDA, pcls->getCapacity(), streams[species]);
  sorter.sort(pcls, gridCUDA, pcls->getNOP(), streams[species]);
  // the sort swapped the SoA pointers on the host
  cudaErrChk(cudaMemcpyAsync(pclsArrayCUDAPtr[species], pcls,
                             sizeof(particleArrayCUDA), cudaMemcpyDefault,
                             streams[species]));
}

/**
 * @brief Sort by cell the species with a nonzero entry in mask, on the GPU,
 * independently of the mover's periodic sorting (CalculateMoments and the
 * macrocell analyses of the data analysis, spectra and histograms, need
 * it).
 */
void c_Solver::sortSpecies(const std::vector<int>& mask) {
  for (int i = 0; i < ns; i++) {
    if (!mask[i])
      continue;
    // Ensure no kernels are in flight on this species' stream
    cudaErrChk(cudaStreamSynchronize(streams[i]));
    // Explicit edge: also wait for the previous cycle's output D->H drain
    // before sort kernels mutate the SoA. The host sync above only waits for
    // streams[i]; outputStream is independent.
    cudaErrChk(cudaStreamWaitEvent(streams[i], eventOutputCopy, 0));
    sortSpeciesByCell(i);
  }

  // Wait until every species has completed its full sort pipeline.
  profiling::Range wait(profiling::Region::SortWait);
  for (int i = 0; i < ns; i++)
    if (mask[i])
      cudaErrChk(cudaStreamSynchronize(streams[i]));
}

/**
 * @brief Pad host particle capacities for regular and test-particle species.
 */
void c_Solver::pad_particle_capacities() {
  for (int i = 0; i < ns; i++)
    particlesHost[i]->padCapacities();

  for (int i = 0; i < nstestpart; i++)
    testpart[i]->padCapacities();
}

/**
 * @brief Return the first cycle after the current simulation window.
 */
int c_Solver::LastCycle() { return (col->getNcycles() + first_cycle); }

/**
 * @brief Inject photoionized exosphere particles into host particle buffers.
 *
 * For each planetary species (index >= numSolarWindSpecies), this method
 * samples new macro-particles from the Chamberlain neutral density profile via
 * ExosphereIonization, then appends them to the AoS comm buffer
 * particlesCommInj[i]. The particles will be copied to the GPU by the
 * subsequent H2D transfer.
 *
 * Memory-aware injection: before sampling, the method queries GPU free memory
 * and computes a per-species particle budget to prevent out-of-memory
 * conditions. The current implementation applies a fixed low-memory heuristic
 * and keeps a portion of the currently free GPU memory in reserve for field
 * arrays, moments, and other runtime allocations.
 *
 * Task-based parallelism: each planetary species is submitted as an independent
 * task to the thread pool (same pool used by the mover). Each task uses its own
 * RNG, particle buffer, and particlesCommInj[i] — no cross-species contention.
 * All futures are collected before returning, so
 * particlesCommInj[i].getCommNOP() is finalized for the subsequent H2D copy
 * loop.
 */
void c_Solver::injectExosphereParticles() {
  if (exosphereIonization == nullptr)
    return;
  profiling::Range range(profiling::Region::Exosphere);

  // ======= Compute a memory-aware per-species injection budget =======
  // Only activate a budget when GPU memory is actually scarce or a hard cap
  // is configured.  When memory is plentiful, maxParticlesPerSpecies stays 0
  // (unlimited), so sampleIonizedParticles takes its unbudgeted path
  // (hasBudget == false, no per-cell budget check).
  const int numPlanetSpecies = ns - numSolarWindSpecies;
  int maxParticlesPerSpecies = 0; // 0 = unlimited (fast path)

  // (1) Configurable hard cap from input file (0 = unlimited)
  const int configCap = col->getMaxExosphereParticlesPerSpecies();
  if (configCap > 0)
    maxParticlesPerSpecies = configCap;

  // (2) GPU memory-based dynamic cap — only when free memory is low
  size_t gpuFree = 0, gpuTotal = 0;
  if (cudaMemGetInfo(&gpuFree, &gpuTotal) == cudaSuccess && gpuTotal > 0) {
    const double freeRatio = static_cast<double>(gpuFree) / gpuTotal;

    // Only impose a memory budget when <30% of GPU memory remains.
    // Above this threshold, pass 0 (unlimited) for zero overhead.
    constexpr double memoryPressureThreshold = 0.30;
    if (freeRatio < memoryPressureThreshold) {
      // Keep 20% of current free memory as safety margin
      const size_t usableBytes = static_cast<size_t>(gpuFree * 0.80);
      const size_t bytesPerParticle =
          static_cast<size_t>(sizeof(SpeciesParticle) * 1.5);
      const int gpuBudgetPerSpecies =
          (numPlanetSpecies > 0)
              ? static_cast<int>(usableBytes / bytesPerParticle) /
                    numPlanetSpecies
              : static_cast<int>(usableBytes / bytesPerParticle);

      // Take the tighter of config cap and GPU budget
      if (maxParticlesPerSpecies > 0)
        maxParticlesPerSpecies =
            std::min(maxParticlesPerSpecies, gpuBudgetPerSpecies);
      else
        maxParticlesPerSpecies = gpuBudgetPerSpecies;
    }
  }

  // ======= Enqueue one sampling task per planetary species =======
  // Each task: (1) samples particles via thread-safe RNG, (2) appends to
  // particlesCommInj[i] AoS comm buffer. No shared mutable state between tasks
  // — safe for concurrent execution. numSolarWindSpecies, numPlanetarySpecies,
  // and exosphereTaskFutures are persistent class members initialized once in
  // Init() — no per-call recomputation or allocation.
  exosphereTaskFutures.clear();

  for (int i = numSolarWindSpecies; i < ns; i++) {
    exosphereTaskFutures.push_back(
        threadPoolPtr->enqueue([this, i, maxParticlesPerSpecies]() {
          profiling::Range range(profiling::Region::ExosphereSample, i);
          // sampleIonizedParticles is thread-safe for different species
          // indices: each species uses its own std::mt19937_64 RNG, particle
          // buffer, and charge accumulator (no global rand() or shared mutable
          // state).
          const std::vector<SpeciesParticle>& exosphereParticles =
              exosphereIonization->sampleIonizedParticles(
                  i, maxParticlesPerSpecies);

          if (exosphereParticles.empty())
            return;

          // Append exosphere particles to AoS comm buffer (after MPI-incoming +
          // repopulated) Each task operates on its own particlesCommInj[i] — no
          // cross-species contention.
          particlesCommInj[i]->appendFromAoS(
              exosphereParticles.data(),
              static_cast<int>(exosphereParticles.size()));
        }));
  }

  // Wait for every species task before the subsequent H2D copy loop runs.
  for (size_t k = 0; k < exosphereTaskFutures.size(); k++)
    awaitTask(exosphereTaskFutures[k],
              "exosphere injection of species " +
                  std::to_string(numSolarWindSpecies + (int)k));
}
