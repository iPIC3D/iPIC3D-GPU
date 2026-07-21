/* iPIC3D was originally developed by Stefano Markidis and Giovanni Lapenta. 
 * This release was contributed by Alec Johnson and Ivy Bo Peng.
 * Publications that use results from iPIC3D need to properly cite  
 * 'S. Markidis, G. Lapenta, and Rizwan-uddin. "Multi-scale simulations of 
 * plasma with iPIC3D." Mathematics and Computers in Simulation 80.7 (2010): 1509-1519.'
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


#include "mpi.h"
#include "MPIdata.h"
#include "iPic3D.h"
#include "TimeTasks.h"
#include <iomanip>
#include "ipicdefs.h"
#include "debug.h"
#include "Parameters.h"
#include "VCtopology3D.h"
#include "Collective.h"
#include "Grid3DCU.h"
#include "EMfields3D.h"
#include "ParticleCommInjection.h"
#include "Timing.h"
#include "outputPrepare.h"
#include "IOManager.h"

#ifdef GPU_SOLVER
#include "GPUFieldPacking.cuh"
#endif


#include <algorithm>
#include <chrono>
#include <fstream>
#include <future>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

// ======= Timing Debugging =======
// Set to 1 to enable per-phase timing printfs
// (launcher, await, MPI, planet, exchange total).
#define ENABLE_SOA_TIMING 0

#include "ExosphereIonization.h"

#include "cudaTypeDef.cuh"
#include "momentKernel.cuh"
#include "particleArrayCUDA.cuh"
#include "moverKernel.cuh"
#include "particleExchange.cuh"
#include "dataAnalysis.cuh"
#include "particleControlKernel.cuh"
#include "injectionKernel.cuh"


#ifdef USE_CATALYST
#include "Adaptor.h"
#endif

// ======= Particle Count Control =======
// Set to true to enable particle merging.
// The current implementation is functional but not fully optimized.
constexpr bool PARTICLE_MERGING = false;
// Set to true to enable particle splitting.
constexpr bool PARTICLE_SPLITTING = false; 

// ======= Solver Constants =======
constexpr double INITIAL_CAPACITY_FACTOR = 1.4;   // SoA array initial over-allocation
constexpr double AUX_BUFFER_FRACTION     = 0.1;   // exiting/filler/staging as fraction of NOP
constexpr double PLANET_BUFFER_FRACTION  = 0.05;  // planet array initial fraction of NOP
constexpr double EXPAND_GROWTH_FACTOR    = 1.5;   // growth factor when expanding arrays
constexpr double EXPAND_THRESHOLD_FACTOR = 1.2;   // trigger expand when NOP * this >= capacity
constexpr double SPLIT_THRESHOLD         = 0.95;  // split when NOP < this * initialNOP
constexpr double MERGE_THRESHOLD         = 1.05;  // merge when NOP > this * initialNOP
constexpr int    DEFAULT_BLOCK_SIZE      = 256;    // default CUDA block size
constexpr int    SMALL_BLOCK_SIZE        = 128;    // smaller CUDA block size (sorting kernels)
constexpr int    INITIAL_PLANET_BUF_CAP  = 1024;   // initial planet cross-species buffer capacity

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
  if (s == "GEMnoPert")       return CaseType::GEMnoPert;
  if (s == "ForceFree")       return CaseType::ForceFree;
  if (s == "GEM")             return CaseType::GEM;
  if (s == "GEMDoubleHarris") return CaseType::GEMDoubleHarris;
  if (s == "BATSRUS")         return CaseType::BATSRUS;
  if (s == "Dipole")          return CaseType::Dipole;
  if (s == "Dipole2D")        return CaseType::Dipole2D;
  if (s == "NullPoints")      return CaseType::NullPoints;
  if (s == "TaylorGreen")     return CaseType::TaylorGreen;
  if (s == "HumpPert")        return CaseType::HumpPert;
  if (s == "RandomCase")      return CaseType::RandomCase;
  if (s == "GEMHarris")       return CaseType::GEMHarris;
  return CaseType::Default;
}
//MPIdata* iPic3D::c_Solver::mpi=0;
/**
 * @brief Destroy the solver and release host-side runtime objects.
 *
 * GPU allocations are released separately in deInitCUDA(). This destructor
 * handles the remaining owning pointers managed by c_Solver.
 */
c_Solver::~c_Solver()
{
  // EMf must be deleted first: ~EMfields3D() calls freeDataType() which
  // dereferences _col, _grid, and _vct (stored as const refs bound to
  // *col, *grid, *vct). Deleting col/vct/grid first would leave those
  // refs dangling.
  delete EMf;  // field (destructor uses _col/_grid/_vct refs → must be first)
  delete col;  // configuration parameters ("collectiveIO")
  delete vct;  // process topology
  delete grid; // grid
  delete ioManager; // I/O backends (HDF5, ADIOS2, VTK buffers)

  // delete particles
  //
  if(particlesCommInj) // exchange particles
  {
    for (int i = 0; i < ns; i++)
      delete particlesCommInj[i];
    delete[] particlesCommInj;
  }

  if(particlesHost) // lightweight SoA host mirror
  {
    for (int i = 0; i < ns; i++)
      delete particlesHost[i];
    delete[] particlesHost;
  }

#ifdef USE_CATALYST
  Adaptor::Finalize();
#endif
  delete [] Ke;
  delete [] BulkEnergy;
  delete [] momentum;
  delete [] Qtot;
  delete [] Qremoved;

  if (testpart) {
    for (int i = 0; i < nstestpart; i++)
      delete testpart[i];
    delete[] testpart;
  }

  delete exosphereIonization;
  delete my_clock;
}

/**
 * @brief Initialize MPI-side solver state, host particle containers, and GPU data.
 *
 * This method owns the full startup sequence: input parsing, topology and grid
 * construction, field initialization, particle initialization or restart load,
 * I/O backend setup, and final CUDA-side allocation.
 * @param argc Command-line argument count forwarded to `Collective`.
 * @param argv Command-line argument vector forwarded to `Collective`.
 * @return `0` on success, nonzero on initialization failure.
 */
int c_Solver::Init(int argc, char **argv) {
  #if defined(__MIC__)
  assert_eq(DVECWIDTH,8);
  #endif
  // get MPI data
  //
  // c_Solver is not a singleton, so the following line was pulled out.
  //MPIdata::init(&argc, &argv);
  //
  // initialized MPI environment
  // nprocs = number of processors
  // myrank = rank of tha process*/
  Parameters::init_parameters();
  //mpi = &MPIdata::instance();
  nprocs = MPIdata::get_nprocs();
  myrank = MPIdata::get_rank();

  col = new Collective(argc, argv); // Every proc loads the parameters of simulation from class Collective
  restart_cycle = col->getRestartOutputCycle();
  SaveDirName = col->getSaveDirName();
  RestartDirName = col->getRestartDirName();
  restart_status = col->getRestart_status();
  ns = col->getNs();            // get the number of particle species involved in simulation
  // Restart labels identify the loop cycle to resume. Periodic checkpoints are
  // written before that cycle completes, so a checkpoint labeled N must restart
  // by executing cycle N again. Fresh runs keep the historical start at cycle 0.
  first_cycle = (restart_status != 0) ? col->getLast_cycle() : 0;
  // initialize the virtual cartesian topology
  vct = new VCtopology3D(*col);
  // Check if we can map the processes into a matrix ordering defined in Collective.cpp
  if (nprocs != vct->getNprocs()) {
    if (myrank == 0) {
      cerr << "Error: " << nprocs << " processes cant be mapped into " << vct->getXLEN() << "x" << vct->getYLEN() << "x" << vct->getZLEN() << " matrix: Change XLEN,YLEN, ZLEN in method VCtopology3D.init()" << endl;
      MPIdata::instance().finalize_mpi();
      return (1);
    }
  }
  // We create a new communicator with a 3D virtual Cartesian topology
    vct->setup_vctopology(MPIdata::get_PicGlobalComm());
  {
    stringstream num_proc_ss;
    num_proc_ss << vct->getCartesian_rank();
    num_proc_str = num_proc_ss.str();
  }
  // initialize the central cell index

#ifdef BATSRUS
  // set index offset for each processor
  col->setGlobalStartIndex(vct);
#endif

  // Print the initial settings to stdout and a file
  if (myrank == 0) {
    // Fresh runs clear old output. Restart runs must preserve restart files but
    // still need writable output directories for settings/proc files.
    if (restart_status == 0) {
      checkOutputFolder(SaveDirName);
      if (RestartDirName != SaveDirName) checkOutputFolder(RestartDirName);
    } else {
      ensureOutputFolder(SaveDirName);
      if (RestartDirName != SaveDirName) ensureOutputFolder(RestartDirName);
    }
    
    MPIdata::instance().Print();
    vct->Print();
    col->Print();
    col->save();
  }
#ifndef NO_MPI
  MPI_Barrier(MPIdata::get_PicGlobalComm());
#endif
  // Create the local grid
  grid = new Grid3DCU(col, vct);  // Create the local grid
  EMf = new EMfields3D(col, grid, vct);  // Create Electromagnetic Fields Object

  // Resolve case string once and store as enum + derived flags
  caseType_ = parseCaseType(col->getCase());
  doPlanet_ = (caseType_ == CaseType::Dipole || caseType_ == CaseType::Dipole2D);

  switch (caseType_) {
    case CaseType::GEMnoPert:       EMf->initGEMnoPert(); break;
    case CaseType::ForceFree:       EMf->initForceFree(); break;
    case CaseType::GEM:             EMf->initGEM(); break;
    case CaseType::GEMDoubleHarris: EMf->initGEMDoubleHarris(); break;
#ifdef BATSRUS
    case CaseType::BATSRUS:         EMf->initBATSRUS(); break;
#endif
    case CaseType::Dipole:          EMf->initDipole(); break;
    case CaseType::Dipole2D:        EMf->initDipole2D(); break;
    case CaseType::NullPoints:      EMf->initNullPoints(); break;
    case CaseType::TaylorGreen:     EMf->initTaylorGreen(); break;
    case CaseType::HumpPert:        EMf->initHumpPerturbation(); break;
    case CaseType::GEMHarris:       EMf->initGEMHarris(); break;
    case CaseType::RandomCase:
      EMf->initRandomField();
      if (myrank==0) {
        cout << "Case is " << col->getCase() <<"\n";
        cout <<"total # of particle per cell is " << col->getNpcel(0) << "\n";
      }
      break;
    default:
      if (myrank==0) {
        cout << " =========================================================== " << endl;
        cout << " WARNING: The case '" << col->getCase() << "' was not recognized. " << endl;
        cout << "          Runing simulation with the default initialization. " << endl;
        cout << " =========================================================== " << endl;
      }
      EMf->init();
      break;
  }

  // ======= Allocate particlesHost[]: lightweight SoA host mirrors =======
  particlesHost = new ParticleSoAHost*[ns];
  for (int i = 0; i < ns; i++)
  {
    particlesHost[i] = new ParticleSoAHost(i, col, vct, grid);

    if (col->getRestart_status() != 0) { // restart
      particlesHost[i]->restartLoad();
    }
    // Fresh start: maxwellian methods below call prepareSoAForNOP() internally
  }

  // Initial condition for PARTICLES (skipped when restarting)
  if (restart_status == 0) {
    for (int i = 0; i < ns; i++)
    {
      switch (caseType_) {
        case CaseType::ForceFree:       particlesHost[i]->force_free(EMf); break;
#ifdef BATSRUS
        case CaseType::BATSRUS:         eprintf("BATSRUS not supported on ParticleSoAHost"); break;
#endif
        case CaseType::NullPoints:      particlesHost[i]->maxwellianNullPoints(EMf); break;
        case CaseType::TaylorGreen:     particlesHost[i]->maxwellianNullPoints(EMf); break;
        case CaseType::GEMDoubleHarris: particlesHost[i]->maxwellianDoubleHarris(EMf); break;
        case CaseType::HumpPert:        particlesHost[i]->maxwellianHumpPerturbation(EMf); break;
        case CaseType::GEMHarris:
          if (col->getCurrentFromAmpere()) {
            if (col->getSpatiallyVaryingThermal())
              particlesHost[i]->maxwellianAmpereVaryingThermal(EMf);
            else
              particlesHost[i]->maxwellianNullPoints(EMf);
          } else {
            particlesHost[i]->maxwellian(EMf);
          }
          break;
        default:                        particlesHost[i]->maxwellian(EMf); break;
      }
    }
  }

  for (int i = 0; i < ns; i++)
    particlesHost[i]->reserve_remaining_particle_IDs();

  // ======= Allocate test particles, if configured =======
  nstestpart = col->getNsTestPart();

  // ======= Allocate particlesCommInj[]: MPI exchange + injection engine =======
  particlesCommInj = new ParticleCommInjection*[ns];
  for (int i = 0; i < ns; i++)
  {
    particlesCommInj[i] = new ParticleCommInjection(*particlesHost[i]);
    const auto totalPcl = col->getNpcel(i) * grid->getNXN() * grid->getNYN() * grid->getNZN();
    particlesCommInj[i]->reserveCommBuffer(static_cast<int>(totalPcl * AUX_BUFFER_FRACTION));
  }

  if(nstestpart>0){
    testpart = new ParticleSoAHost*[nstestpart];
    for (int i = 0; i < nstestpart; i++)
    {
      testpart[i] = new ParticleSoAHost(i+ns,col,vct,grid);//species id for test particles is increased by ns
      testpart[i]->pitch_angle_energy(EMf);
      testpart[i]->reserve_remaining_particle_IDs();
    }
  }

  // ======= Initialize modular I/O manager =======
  ioManager = new IOManager;
  if (Parameters::get_doWriteOutput() || restart_cycle > 0 || col->getCallFinalize()) {
      ioManager->init(col, vct, grid, EMf, particlesHost, ns, testpart, nstestpart, first_cycle);
  }

  Ke = new double[ns];
  BulkEnergy = new double[ns];
  momentum = new double[ns];
  Qtot = new double[ns];
  cq = SaveDirName + "/ConservedQuantities.txt";
  if (myrank == 0) {
    ofstream my_file(cq.c_str());
    my_file.close();
  }
  

  Qremoved = new double[ns];

  // ======= Exosphere ionization source =======
  numSolarWindSpecies = col->getNumSolarWindSpecies();
  numPlanetarySpecies = ns - numSolarWindSpecies;
  if (col->getEnableExosphereInjection() && numPlanetarySpecies > 0) {
    exosphereIonization = new ExosphereIonization(col, grid, vct);
    exosphereTaskFutures.reserve(numPlanetarySpecies);
  } else {
    exosphereIonization = nullptr;
  }

#ifdef USE_CATALYST
  Adaptor::Initialize(col, \
		  (int)(grid->getXstart()/grid->getDX()), \
		  (int)(grid->getYstart()/grid->getDY()), \
		  (int)(grid->getZstart()/grid->getDZ()), \
		  grid->getNXN(),
		  grid->getNYN(),
		  grid->getNZN(),
		  grid->getDX(),
		  grid->getDY(),
		  grid->getDZ());
#endif

  // Create or open the particle number file csv
  pclNumCSV = std::ofstream(SaveDirName + "/particleNum" + std::to_string(myrank) + ".csv", std::ios::app); 
  pclNumCSV << "cycle,";
  for(int i=0; i<ns-1; i++){
    pclNumCSV << "species" << i << ",";
  }
  pclNumCSV << "species" << ns-1 << std::endl;

  initCUDA();
  if (Parameters::get_doWriteOutput() || restart_cycle > 0 || col->getCallFinalize()) {
    ioManager->setRestartParticleCellMetadata(&restartParticleCellMetadata_);
    if (ioManager->needsParticleSync(first_cycle)) {
      outputCopyAsync(first_cycle - 1);
      cudaErrChk(cudaEventSynchronize(eventOutputCopy));
    }
  }

  my_clock = new Timing(myrank);

  return 0;
}

/**
 * @brief Allocate and initialize all CUDA-side solver resources.
 *
 * This includes stream creation, particle/moment/device metadata allocation,
 * pinned host registrations, sorting buffers, and planet-boundary workspaces.
 */
int c_Solver::initCUDA(){
  heatFluxEnabled_ = Parameters::get_doWriteOutput()
                  && col->getOutputConfig().needsAnyHeatFlux();
  heatFluxScheduledCycle_ = -1;

  // ======= Select the GPU assigned to this MPI rank =======
  {
    MPI_Comm sharedComm; int sharedRank, sharedSize; int deviceOnNode;
    MPI_Comm_split_type(MPIdata::get_PicGlobalComm(), MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &sharedComm); 
    MPI_Comm_rank(sharedComm, &sharedRank);             // rank in the node
    MPI_Comm_size(sharedComm, &sharedSize);             // total processes in this node
    cudaErrChk(cudaGetDeviceCount(&deviceOnNode));      // GPU on the node
    
    if(sharedSize <= deviceOnNode){ // process <= device
      cudaDeviceOnNode = sharedRank;
    }else{
      if(sharedSize % deviceOnNode != 0){ // if proc is not a multiple of device
        cerr << "Error: Can not map process to device on node. " << "Global COMM rank: " << MPIdata::get_rank() <<  
            " Shared COMM size: " << sharedSize << " Device Number in Node: " << deviceOnNode << endl;
        MPIdata::instance().finalize_mpi();
        return (1);
      }
      int procPerDevice = sharedSize / deviceOnNode;
      cudaDeviceOnNode = sharedRank / procPerDevice;
    }
    cudaErrChk(cudaSetDevice(cudaDeviceOnNode)); 
#ifndef NDEBUG
    if(sharedRank == 0)
    cout << "[*]GPU assignment: shared comm size: " << sharedSize << " GPU device on the node: " << deviceOnNode << endl;
#endif
  }

  // ======= Create per-species streams and async launcher state =======
  streams = new cudaStream_t[ns*2]; stayedParticle = new int[ns]; exitingResults = new std::future<int>[ns];
  for(int i=0; i<ns; i++){ cudaErrChk(cudaStreamCreate(streams+i)); cudaErrChk(cudaStreamCreate(streams+i+ns)); stayedParticle[i] = 0; }
  cudaErrChk(cudaStreamCreate(&planetStream));
  // Dedicated streams for output D->H and field-interpolation H->D copies.
  // Isolating these from the per-species streams[] removes implicit ordering
  // between the output copy and the next cycle's mover/sort kernels, which is
  // re-established explicitly via cycleEndEvent/eventOutputCopy below.
  cudaErrChk(cudaStreamCreate(&outputStream));
  cudaErrChk(cudaStreamCreate(&fieldH2DStream));
  if (heatFluxEnabled_)
    cudaErrChk(cudaStreamCreate(&heatFluxStream));

  for (int i = 0; i < ns; i++)
    particlesHost[i]->initializeParticleIDDeviceCounter(streams[i]);

  {
    // ======= Allocate device-resident particle containers and staging buffers =======
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

    for(int i=0; i<ns; i++){
      // particleArrayCUDA performs the initial SoA H2D copy in its constructor
      pclsArrayHostPtr[i] = newHostPinnedObject<particleArrayCUDA>(particlesHost[i], INITIAL_CAPACITY_FACTOR, streams[i]);
      pclsArrayHostPtr[i]->setInitialNOP(pclsArrayHostPtr[i]->getNOP());
      pclsArrayCUDAPtr[i] = pclsArrayHostPtr[i]->copyToDevice();

      departureArrayHostPtr[i] = newHostPinnedObject<departureArrayType>(pclsArrayHostPtr[i]->getSize()); // same length
      departureArrayCUDAPtr[i] = departureArrayHostPtr[i]->copyToDevice();
      cudaErrChk(cudaMemsetAsync(departureArrayHostPtr[i]->getArray(), 0, departureArrayHostPtr[i]->getSize() * sizeof(departureArrayElementType), streams[i]));

      // hashedSumArrayHostPtr[i] = new hashedSum[8]{ // 
      //   hashedSum(5), hashedSum(5), hashedSum(5), hashedSum(5), 
      //   hashedSum(5), hashedSum(5), hashedSum(10), hashedSum(10)
      // };

      hashedSumArrayHostPtr[i] = newHostPinnedObjectArray<hashedSum>(departureArrayElementType::HASHED_SUM_NUM, 10);

      hashedSumArrayCUDAPtr[i] = copyArrayToDevice(hashedSumArrayHostPtr[i], departureArrayElementType::HASHED_SUM_NUM);
      
      exitingArrayHostPtr[i] = newHostPinnedObject<exitingArray>(AUX_BUFFER_FRACTION * pclsArrayHostPtr[i]->getNOP());
      exitingArrayCUDAPtr[i] = exitingArrayHostPtr[i]->copyToDevice();
      fillerBufferArrayHostPtr[i] = newHostPinnedObject<fillerBuffer>(AUX_BUFFER_FRACTION * pclsArrayHostPtr[i]->getNOP());
      fillerBufferArrayCUDAPtr[i] = fillerBufferArrayHostPtr[i]->copyToDevice();

      // AoS staging buffer for incoming H2D particle transfers
      // (MPI exchange, repopulated particles, and exosphere injection).
      // Sized from the initial NOP and expanded on demand.
      incomingStagingHostPtr[i] = newHostPinnedObject<arrayCUDA<SpeciesParticle>>(static_cast<uint32_t>(AUX_BUFFER_FRACTION * pclsArrayHostPtr[i]->getNOP()));
      incomingStagingCUDAPtr[i] = incomingStagingHostPtr[i]->copyToDevice();

    }
  }

  // ======= Allocate one shared device grid descriptor =======
  grid3DCUDAHostPtr = newHostPinnedObject<grid3DCUDA>(grid);
  grid3DCUDACUDAPtr = copyToDevice(grid3DCUDAHostPtr, 0);


  // ======= Build per-species mover parameters =======
  // Scalar species parameters are copied from ParticleSoAHost.
  moverParamHostPtr = new moverParameter*[ns];
  moverParamCUDAPtr = new moverParameter*[ns];
  for(int i=0; i<ns; i++){
    moverParamHostPtr[i] = newHostPinnedObject<moverParameter>(particlesHost[i], pclsArrayCUDAPtr[i], departureArrayCUDAPtr[i], hashedSumArrayCUDAPtr[i]);

    // Initialize mover flags for open boundaries, repopulation, and planet handling.
    particlesHost[i]->openbc_particles_outflowInfo(&moverParamHostPtr[i]->doOpenBC, moverParamHostPtr[i]->applyOpenBC, moverParamHostPtr[i]->deleteBoundary, moverParamHostPtr[i]->openBoundary);
    moverParamHostPtr[i]->appendCountAtomic = 0;

    // GPU-side EXIT BC: particles exiting via an EXIT face are marked DELETE
    // on the GPU to avoid sending them through MPI exchange at all.
    // Only applies on boundary ranks (where the neighbor is MPI_PROC_NULL).
    particlesHost[i]->fillExitBCFlags(moverParamHostPtr[i]->isExitBC);

    if(col->getRHOinject(i)>0.0)
    particlesHost[i]->repopulate_particlesInfo(&moverParamHostPtr[i]->doRepopulateInjection, moverParamHostPtr[i]->doRepopulateInjectionSide, moverParamHostPtr[i]->repopulateBoundary);
    else moverParamHostPtr[i]->doRepopulateInjection = false;

    if (caseType_ == CaseType::Dipole) {
      moverParamHostPtr[i]->doSphere = 1;
      moverParamHostPtr[i]->sphereOrigin[0] = col->getx_center_planet();
      moverParamHostPtr[i]->sphereOrigin[1] = col->gety_center_planet();
      moverParamHostPtr[i]->sphereOrigin[2] = col->getz_center_planet();
      moverParamHostPtr[i]->sphereRadius = col->getPlanet_radius();
    } else if (caseType_ == CaseType::Dipole2D) {
      moverParamHostPtr[i]->doSphere = 2;
      moverParamHostPtr[i]->sphereOrigin[0] = col->getx_center_planet();
      moverParamHostPtr[i]->sphereOrigin[1] = 0.0;
      moverParamHostPtr[i]->sphereOrigin[2] = col->getz_center_planet();
      moverParamHostPtr[i]->sphereRadius = col->getPlanet_radius();
    } else {
      moverParamHostPtr[i]->doSphere = 0;
    }

    moverParamCUDAPtr[i] = copyToDevice(moverParamHostPtr[i], streams[i]);
  }

  momentParamHostPtr = new momentParameter*[ns];
  momentParamCUDAPtr = new momentParameter*[ns];
  for(int i=0; i<ns; i++){
    momentParamHostPtr[i] = newHostPinnedObject<momentParameter>(pclsArrayCUDAPtr[i], departureArrayCUDAPtr[i]);
    momentParamCUDAPtr[i] = copyToDevice(momentParamHostPtr[i], streams[i]);
  }

  // ======= Build per-species GPU injection parameters =======
  injectionParamHostPtr = new injectionParameter*[ns];
  injectionParamCUDAPtr = new injectionParameter*[ns];
  for (int i = 0; i < ns; i++) {
    injectionParamHostPtr[i] = newHostPinnedObject<injectionParameter>();
    particlesCommInj[i]->fillInjectionParameter(injectionParamHostPtr[i]);
    injectionParamCUDAPtr[i] = copyToDevice(injectionParamHostPtr[i], streams[i]);
  }



  // ======= Allocate device moment buffers =======
  auto gridSize = grid->getNXN() * grid->getNYN() * grid->getNZN();
  momentsCUDAPtr = new cudaTypeArray1<cudaMomentType>[ns];
  //for(int i=0; i<ns; i++)cudaMallocAsync(&(momentsCUDAPtr[i]), gridSize*10*sizeof(cudaMomentType), streams[i]);
  for(int i=0; i<ns; i++) cudaErrChk(cudaMalloc(&(momentsCUDAPtr[i]), gridSize*10*sizeof(cudaMomentType)));

  heatFluxCUDAPtr = nullptr;
  heatFluxBulkCUDAPtr = nullptr;
  if (heatFluxEnabled_) {
    heatFluxCUDAPtr = new cudaTypeArray1<cudaMomentType>[ns];
    heatFluxBulkCUDAPtr = new cudaTypeArray1<cudaMomentType>[ns];
    for (int i = 0; i < ns; i++) {
      cudaErrChk(cudaMalloc(&(heatFluxCUDAPtr[i]),
                            gridSize * HeatFlux::ComponentCount * sizeof(cudaMomentType)));
      cudaErrChk(cudaMalloc(&(heatFluxBulkCUDAPtr[i]),
                            gridSize * 4 * sizeof(cudaMomentType)));
    }
  }

#ifndef GPU_SOLVER
  { // Register the 10 host-side moment arrays per species as pinned memory.
    // NOTE: when GPU_SOLVER is ON, pinning is deferred to after
    // gpuSolverSyncH2D to avoid conflicting with cudaMemcpyAsync
    // on the full 4D arrays (cudaHostRegister per-species slices
    // + whole-array async copy = "invalid argument").
    for(int i=0; i<ns; i++)
      registerMomentsPinnedMemory(i);
  }
#endif

  // cudaMallocAsync(&fieldForPclCUDAPtr, gridSize*8*sizeof(cudaCommonType), 0);

  const int fieldSize = grid->getNZN() * (grid->getNYN() - 1) * (grid->getNXN() - 1);

  //cudaMallocAsync(&fieldForPclCUDAPtr, fieldSize * 24 * sizeof(cudaFieldType), 0);
  cudaErrChk(cudaMalloc(&fieldForPclCUDAPtr, fieldSize * 24 * sizeof(cudaFieldType)));

#ifdef GPU_SOLVER
  // When GPU solver is ON, field packing is done on GPU — no need for
  // the pinned host staging buffer.
  fieldForPclHostPtr = nullptr;

  // Clear any stale CUDA error before GPU solver init
  cudaGetLastError();

  // Allocate GPU solver field arrays and perform initial H2D sync.
  EMf->gpuSolverAllocate();
  EMf->gpuSolverSyncH2D(EMf->gpuSolverStream());
  cudaErrChk(cudaEventCreateWithFlags(&solverDoneEvent, cudaEventDisableTiming));

  // momentsPipelineDoneEvt is recorded at the end of every MomentsAwait() GPU
  // pipeline (after gpuCalculateHatFunctions). heatFluxStream waits on it
  // before the D2D copy of d_rhons/d_Jxs into heatFluxBulkCUDAPtr, ensuring
  // current-cycle values with ghost cells filled are used.
  // Pre-record here (on the now-synchronized solverStream_) so that
  // ScheduleHeatFlux on cycle 0 — which runs after the pre-loop MomentsAwait()
  // has already re-recorded the event — never observes an uninitialised handle.
  if (heatFluxEnabled_) {
    cudaErrChk(cudaEventCreateWithFlags(&momentsPipelineDoneEvt, cudaEventDisableTiming));
    cudaErrChk(cudaEventRecord(momentsPipelineDoneEvt, EMf->gpuSolverStream()));
  }

  // Now safe to pin moment memory (after the initial H2D sync is done)
  for(int i=0; i<ns; i++)
    registerMomentsPinnedMemory(i);
#else
  cudaErrChk(cudaHostAlloc((void**)&fieldForPclHostPtr, fieldSize * 24 * sizeof(cudaFieldType), 0));
#endif

  if (heatFluxEnabled_)
    registerHeatFluxPinnedMemory();

  threadPoolPtr = new ThreadPool(ns);
  cudaErrChk(cudaEventCreateWithFlags(&event0, cudaEventDisableTiming));
  cudaErrChk(cudaEventCreateWithFlags(&eventOutputCopy, cudaEventDisableTiming|cudaEventBlockingSync));
  // Per-species persistent events. cycleEndEvent[i] is recorded at the end
  // of MoverAwaitAndPclExchange on streams[i]. moverHashedReadyEvt[i] /
  // auxHashedReadyEvt[i] replace the previously per-call event1/event2
  // created inside cudaLauncherAsync.
  cycleEndEvent        = new cudaEvent_t[ns];
  moverHashedReadyEvt  = new cudaEvent_t[ns];
  auxHashedReadyEvt    = new cudaEvent_t[ns];
  stayedMomentsDoneEvt = new cudaEvent_t[ns];
  heatFluxReadDoneEvt  = heatFluxEnabled_ ? new cudaEvent_t[ns] : nullptr;
  heatFluxHostDoneEvt  = heatFluxEnabled_ ? new cudaEvent_t[ns] : nullptr;
  for (int i = 0; i < ns; i++) {
    cudaErrChk(cudaEventCreateWithFlags(&cycleEndEvent[i],        cudaEventDisableTiming|cudaEventBlockingSync));
    cudaErrChk(cudaEventCreateWithFlags(&moverHashedReadyEvt[i],  cudaEventDisableTiming));
    cudaErrChk(cudaEventCreateWithFlags(&auxHashedReadyEvt[i],    cudaEventDisableTiming));
    cudaErrChk(cudaEventCreateWithFlags(&stayedMomentsDoneEvt[i], cudaEventDisableTiming));
    if (heatFluxEnabled_) {
      cudaErrChk(cudaEventCreateWithFlags(&heatFluxReadDoneEvt[i], cudaEventDisableTiming));
      cudaErrChk(cudaEventCreateWithFlags(&heatFluxHostDoneEvt[i], cudaEventDisableTiming|cudaEventBlockingSync));
      cudaErrChk(cudaEventRecord(heatFluxReadDoneEvt[i], heatFluxStream));
      cudaErrChk(cudaEventRecord(heatFluxHostDoneEvt[i], heatFluxStream));
    }
    // Pre-record cycleEndEvent[i] on streams[i] so MomentsAwait() / output-
    // CopyAsync() are well-defined even if invoked before any moment pipeline
    // has run. Both real producers (CalculateMoments and MoverAwaitAndPcl-
    // Exchange) re-record this event after their copyMomentsD2H so subsequent
    // waits correctly observe the latest cycle's moments.
    cudaErrChk(cudaEventRecord(cycleEndEvent[i], streams[i]));
  }
  // Pre-record eventOutputCopy on outputStream once, so that the very first
  // cycle's cudaStreamWaitEvent on it (in cudaLauncherAsync / sortAllSpecies)
  // is well-defined and resolves immediately. This eliminates the need for a
  // "first-cycle" guard flag; Init() schedules and drains an explicit first-
  // cycle copy later when configured outputs need host particle data.
  cudaErrChk(cudaEventRecord(eventOutputCopy, outputStream));

  // ======= Allocate merge bookkeeping =======
  toBeMerged = new int[2 * ns];
  for(int i=0;i<2*ns;i++){
    toBeMerged[i] = 0;
  }
  //memset(toBeMerged, 0, 2 * ns * sizeof(int));

  // ======= Initialize per-species cell sorters =======
  sortingCycle_ = col->getSortingCycle();
  sortThisCycle_ = false;
  cellSorters = new CellSorter[ns];
  if (sortingCycle_ > 0) {
    for (int i = 0; i < ns; i++) {
      cellSorters[i].init(*grid3DCUDAHostPtr,
                          pclsArrayHostPtr[i]->getCapacity(),
                          streams[i]);
    }
  }

  dataAnalysis::dataAnalysisPipeline::createOutputDirectory(
      myrank, ns, vct, restart_status != 0, col->getVelocitySpectra());

  // ======= Allocate planet quasi-neutral boundary-condition buffers =======
  {
    planetArrayHostPtr = new planetArray*[ns];
    planetArrayCUDAPtr = new planetArray*[ns];
    planetPclCount     = new int[ns];

    for (int i = 0; i < ns; i++) {
      if (doPlanet_) {
        planetArrayHostPtr[i] = newHostPinnedObject<planetArray>((uint32_t)(PLANET_BUFFER_FRACTION * pclsArrayHostPtr[i]->getNOP()));
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
      if (col->getQOM(i) < 0) planetElecSpeciesCount++;

    planetElecSpeciesMap = new int[planetElecSpeciesCount];
    {
      int idx = 0;
      for (int i = 0; i < ns; i++)
        if (col->getQOM(i) < 0) planetElecSpeciesMap[idx++] = i;
    }

    // Cross-species device buffers.
    planetBufCapacity = doPlanet_ ? INITIAL_PLANET_BUF_CAP : 0;
    if (doPlanet_) {
      cudaErrChk(cudaMalloc(&planetEnergyBuf,    planetBufCapacity * sizeof(cudaParticleType)));
      cudaErrChk(cudaMalloc(&planetGlobalIdxBuf,  planetBufCapacity * sizeof(uint32_t)));
      cudaErrChk(cudaMalloc(&planetIonChargeDevice, sizeof(cudaParticleType)));
      cudaErrChk(cudaMalloc(&planetCutoffDevice,    sizeof(int)));

      // Device array of per-electron-species planetArray device pointers.
      cudaErrChk(cudaMalloc(&planetArrayCUDAPtrDevice, planetElecSpeciesCount * sizeof(planetArray*)));
      planetArray** tmpPtrs = new planetArray*[planetElecSpeciesCount];
      for (int e = 0; e < planetElecSpeciesCount; e++)
        tmpPtrs[e] = planetArrayCUDAPtr[planetElecSpeciesMap[e]];
      cudaErrChk(cudaMemcpy(planetArrayCUDAPtrDevice, tmpPtrs, planetElecSpeciesCount * sizeof(planetArray*), cudaMemcpyHostToDevice));
      delete[] tmpPtrs;

      cudaErrChk(cudaMalloc(&planetElecOffsetsDevice, planetElecSpeciesCount * sizeof(int)));
    } else {
      planetEnergyBuf = nullptr;
      planetGlobalIdxBuf = nullptr;
      planetIonChargeDevice = nullptr;
      planetCutoffDevice = nullptr;
      planetArrayCUDAPtrDevice = nullptr;
      planetElecOffsetsDevice = nullptr;
    }

    // Persistent buffers reused by processPlanetParticles().
    const int elecCount = planetElecSpeciesCount > 0 ? planetElecSpeciesCount : 1;
    cudaErrChk(cudaHostAlloc(&planetElecOffsets, elecCount * sizeof(int), cudaHostAllocDefault));
    cudaErrChk(cudaHostAlloc(&planetTmpPtrs, elecCount * sizeof(planetArray*), cudaHostAllocDefault));
    cudaErrChk(cudaHostAlloc(&planetSurvivorCount, elecCount * sizeof(int), cudaHostAllocDefault));

    if (doPlanet_) {
      cudaErrChk(cudaMalloc(&planetSurvivorCountDevice, elecCount * sizeof(int)));
      planetReflectedBufCapacity = INITIAL_PLANET_BUF_CAP;
      cudaErrChk(cudaMalloc(&planetReflectedBuf, planetReflectedBufCapacity * sizeof(SpeciesParticle)));
    } else {
      planetSurvivorCountDevice = nullptr;
      planetReflectedBuf = nullptr;
      planetReflectedBufCapacity = 0;
    }
    planetRngCycleCounter = 0;
  }

  cudaErrChk(cudaDeviceSynchronize());

  if(MPIdata::get_rank() == 0)std::cout << "CUDA Init finished" << std::endl;

  return 0;

}


/**
 * @brief Release all CUDA-side resources owned by the solver.
 *
 * This mirrors initCUDA(): streams, device descriptors, pinned registrations,
 * sort buffers, and planet-boundary workspaces are all torn down here.
 */
int c_Solver::deInitCUDA(){

  cudaEventDestroy(event0);
  cudaEventDestroy(eventOutputCopy);
  for (int i = 0; i < ns; i++) {
    cudaEventDestroy(cycleEndEvent[i]);
    cudaEventDestroy(moverHashedReadyEvt[i]);
    cudaEventDestroy(auxHashedReadyEvt[i]);
    cudaEventDestroy(stayedMomentsDoneEvt[i]);
    if (heatFluxEnabled_) {
      cudaEventDestroy(heatFluxReadDoneEvt[i]);
      cudaEventDestroy(heatFluxHostDoneEvt[i]);
    }
  }
  delete[] cycleEndEvent;
  delete[] moverHashedReadyEvt;
  delete[] auxHashedReadyEvt;
  delete[] stayedMomentsDoneEvt;
#ifdef GPU_SOLVER
  cudaEventDestroy(solverDoneEvent);
  if (heatFluxEnabled_)
    cudaEventDestroy(momentsPipelineDoneEvt);
#endif
  delete[] heatFluxReadDoneEvt;
  delete[] heatFluxHostDoneEvt;

  delete threadPoolPtr;

  deleteHostPinnedObject(grid3DCUDAHostPtr);
  cudaFree(grid3DCUDACUDAPtr);

  cudaFree(fieldForPclCUDAPtr);
#ifdef GPU_SOLVER
  // fieldForPclHostPtr was not allocated when GPU_SOLVER is ON.
#else
  cudaFreeHost(fieldForPclHostPtr);
#endif

  // ======= Release per-species host/device objects =======
  for(int i=0; i<ns; i++){

    // Destroy host-pinned wrapper objects.

    deleteHostPinnedObject(pclsArrayHostPtr[i]);
    deleteHostPinnedObject(departureArrayHostPtr[i]);
    deleteHostPinnedObjectArray(hashedSumArrayHostPtr[i], departureArrayElementType::HASHED_SUM_NUM);
    deleteHostPinnedObject(exitingArrayHostPtr[i]);
    deleteHostPinnedObject(fillerBufferArrayHostPtr[i]);
    deleteHostPinnedObject(incomingStagingHostPtr[i]);

    deleteHostPinnedObject(moverParamHostPtr[i]);
    deleteHostPinnedObject(momentParamHostPtr[i]);
    deleteHostPinnedObject(injectionParamHostPtr[i]);

    // Release device-side object storage.

    cudaFree(pclsArrayCUDAPtr[i]);
    cudaFree(departureArrayCUDAPtr[i]);
    cudaFree(hashedSumArrayCUDAPtr[i]);
    cudaFree(exitingArrayCUDAPtr[i]);
    cudaFree(fillerBufferArrayCUDAPtr[i]);
    cudaFree(incomingStagingCUDAPtr[i]);

    cudaFree(moverParamCUDAPtr[i]);
    cudaFree(momentParamCUDAPtr[i]);
    cudaFree(injectionParamCUDAPtr[i]);

    cudaFree(momentsCUDAPtr[i]);
    if (heatFluxEnabled_) {
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
  delete[] moverParamHostPtr;
  delete[] moverParamCUDAPtr;
  delete[] momentParamHostPtr;
  delete[] momentParamCUDAPtr;
  delete[] injectionParamHostPtr;
  delete[] injectionParamCUDAPtr;
  delete[] momentsCUDAPtr;
  delete[] heatFluxCUDAPtr;
  delete[] heatFluxBulkCUDAPtr;
  delete[] toBeMerged;

  // ======= Release planet quasi-neutral boundary-condition buffers =======
  for (int i = 0; i < ns; i++) {
    if (planetArrayHostPtr[i]) deleteHostPinnedObject(planetArrayHostPtr[i]);
    if (planetArrayCUDAPtr[i]) cudaFree(planetArrayCUDAPtr[i]);
  }
  delete[] planetArrayHostPtr;
  delete[] planetArrayCUDAPtr;
  delete[] planetPclCount;
  delete[] planetElecSpeciesMap;
  if (planetEnergyBuf)           cudaFree(planetEnergyBuf);
  if (planetGlobalIdxBuf)        cudaFree(planetGlobalIdxBuf);
  if (planetIonChargeDevice)     cudaFree(planetIonChargeDevice);
  if (planetCutoffDevice)        cudaFree(planetCutoffDevice);
  if (planetArrayCUDAPtrDevice) cudaFree(planetArrayCUDAPtrDevice);
  if (planetElecOffsetsDevice)   cudaFree(planetElecOffsetsDevice);
  cudaFreeHost(planetElecOffsets);
  cudaFreeHost(planetTmpPtrs);
  cudaFreeHost(planetSurvivorCount);
  if (planetSurvivorCountDevice) cudaFree(planetSurvivorCountDevice);
  if (planetReflectedBuf)        cudaFree(planetReflectedBuf);

  // ======= Release cell sorters =======
  for (int i = 0; i < ns; i++) cellSorters[i].free();
  delete[] cellSorters;

  // ======= Destroy streams =======
  for(int i=0; i<ns*2; i++)cudaStreamDestroy(streams[i]);
  cudaStreamDestroy(planetStream);
  cudaStreamDestroy(outputStream);
  cudaStreamDestroy(fieldH2DStream);
  if (heatFluxEnabled_)
    cudaStreamDestroy(heatFluxStream);
  delete[] streams;
  delete[] stayedParticle;
  delete[] exitingResults;

  { // Unregister the host-side pinned moment arrays.
    for (int i = 0; i < ns; i++)
      unregisterMomentsPinnedMemory(i);
  }
  if (heatFluxEnabled_)
    unregisterHeatFluxPinnedMemory();

  return 0;
}


/**
 * @brief Asynchronously copy one species' 10 moment arrays from device to host.
 *
 * The destination buffers are the corresponding EMfields3D host arrays for the
 * species. Callers are responsible for synchronizing the stream later.
 * @param species Species index whose moment arrays are copied.
 * @param stream CUDA stream that carries the asynchronous copies.
 */
void c_Solver::copyMomentsD2H(int species, cudaStream_t stream) {
  const auto gridSize = grid->getNXN() * grid->getNYN() * grid->getNZN();
  cudaErrChk(cudaMemcpyAsync((void*)&(EMf->getRHOns().get(species,0,0,0)),  momentsCUDAPtr[species]+0*gridSize, gridSize*sizeof(cudaMomentType), cudaMemcpyDefault, stream));
  cudaErrChk(cudaMemcpyAsync((void*)&(EMf->getJxs().get(species,0,0,0)),    momentsCUDAPtr[species]+1*gridSize, gridSize*sizeof(cudaMomentType), cudaMemcpyDefault, stream));
  cudaErrChk(cudaMemcpyAsync((void*)&(EMf->getJys().get(species,0,0,0)),    momentsCUDAPtr[species]+2*gridSize, gridSize*sizeof(cudaMomentType), cudaMemcpyDefault, stream));
  cudaErrChk(cudaMemcpyAsync((void*)&(EMf->getJzs().get(species,0,0,0)),    momentsCUDAPtr[species]+3*gridSize, gridSize*sizeof(cudaMomentType), cudaMemcpyDefault, stream));
  cudaErrChk(cudaMemcpyAsync((void*)&(EMf->getpXXsn().get(species,0,0,0)),  momentsCUDAPtr[species]+4*gridSize, gridSize*sizeof(cudaMomentType), cudaMemcpyDefault, stream));
  cudaErrChk(cudaMemcpyAsync((void*)&(EMf->getpXYsn().get(species,0,0,0)),  momentsCUDAPtr[species]+5*gridSize, gridSize*sizeof(cudaMomentType), cudaMemcpyDefault, stream));
  cudaErrChk(cudaMemcpyAsync((void*)&(EMf->getpXZsn().get(species,0,0,0)),  momentsCUDAPtr[species]+6*gridSize, gridSize*sizeof(cudaMomentType), cudaMemcpyDefault, stream));
  cudaErrChk(cudaMemcpyAsync((void*)&(EMf->getpYYsn().get(species,0,0,0)),  momentsCUDAPtr[species]+7*gridSize, gridSize*sizeof(cudaMomentType), cudaMemcpyDefault, stream));
  cudaErrChk(cudaMemcpyAsync((void*)&(EMf->getpYZsn().get(species,0,0,0)),  momentsCUDAPtr[species]+8*gridSize, gridSize*sizeof(cudaMomentType), cudaMemcpyDefault, stream));
  cudaErrChk(cudaMemcpyAsync((void*)&(EMf->getpZZsn().get(species,0,0,0)),  momentsCUDAPtr[species]+9*gridSize, gridSize*sizeof(cudaMomentType), cudaMemcpyDefault, stream));
}

void c_Solver::copyHeatFluxBulkH2D(int species, cudaStream_t stream) {
  const auto gridSize = grid->getNXN() * grid->getNYN() * grid->getNZN();
  cudaErrChk(cudaMemcpyAsync(heatFluxBulkCUDAPtr[species] + 0 * gridSize,
                             (void*)&(EMf->getRHOns().get(species,0,0,0)),
                             gridSize * sizeof(cudaMomentType),
                             cudaMemcpyDefault, stream));
  cudaErrChk(cudaMemcpyAsync(heatFluxBulkCUDAPtr[species] + 1 * gridSize,
                             (void*)&(EMf->getJxs().get(species,0,0,0)),
                             gridSize * sizeof(cudaMomentType),
                             cudaMemcpyDefault, stream));
  cudaErrChk(cudaMemcpyAsync(heatFluxBulkCUDAPtr[species] + 2 * gridSize,
                             (void*)&(EMf->getJys().get(species,0,0,0)),
                             gridSize * sizeof(cudaMomentType),
                             cudaMemcpyDefault, stream));
  cudaErrChk(cudaMemcpyAsync(heatFluxBulkCUDAPtr[species] + 3 * gridSize,
                             (void*)&(EMf->getJzs().get(species,0,0,0)),
                             gridSize * sizeof(cudaMomentType),
                             cudaMemcpyDefault, stream));
}

/**
 * @brief Fill heatFluxBulkCUDAPtr[species] from live GPU solver moment arrays (D2D).
 *
 * GPU_SOLVER=ON path: replaces the stale-host copyHeatFluxBulkH2D by reading
 * d_rhons/d_Jxs/d_Jys/d_Jzs directly on the device. The caller MUST ensure
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
 * @param stream   CUDA stream carrying the asynchronous copies (heatFluxStream).
 */
void c_Solver::copyHeatFluxBulkD2D(int species, cudaStream_t stream) {
#ifdef GPU_SOLVER
  const size_t gridSize = (size_t)grid->getNXN() * grid->getNYN() * grid->getNZN();
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
#else
  // Should never be called when GPU_SOLVER is off; guard against accidental use.
  (void)species; (void)stream;
  eprintf("copyHeatFluxBulkD2D called but GPU_SOLVER is not compiled in");
#endif
}

void c_Solver::copyHeatFluxD2H(int species, cudaStream_t stream) {
  const auto gridSize = grid->getNXN() * grid->getNYN() * grid->getNZN();
  cudaErrChk(cudaMemcpyAsync(EMf->getHeatFluxSpeciesPtr(species),
                             heatFluxCUDAPtr[species],
                             gridSize * HeatFlux::ComponentCount * sizeof(cudaMomentType),
                             cudaMemcpyDefault, stream));
}

/**
 * @brief Register one species' host-side moment arrays as pinned memory.
 *
 * Pinning these buffers enables faster asynchronous D2H copies from the CUDA
 * moment buffers into the EMfields3D storage used by the field solver.
 * @param species Species index whose host moment arrays are pinned.
 */
void c_Solver::registerMomentsPinnedMemory(int species) {
  const auto gridSize = grid->getNXN() * grid->getNYN() * grid->getNZN();
  cudaErrChk(cudaHostRegister((void*)&(EMf->getRHOns().get(species,0,0,0)), gridSize*sizeof(cudaCommonType), cudaHostRegisterDefault));
  cudaErrChk(cudaHostRegister((void*)&(EMf->getJxs().get(species,0,0,0)), gridSize*sizeof(cudaCommonType), cudaHostRegisterDefault));
  cudaErrChk(cudaHostRegister((void*)&(EMf->getJys().get(species,0,0,0)), gridSize*sizeof(cudaCommonType), cudaHostRegisterDefault));
  cudaErrChk(cudaHostRegister((void*)&(EMf->getJzs().get(species,0,0,0)), gridSize*sizeof(cudaCommonType), cudaHostRegisterDefault));
  cudaErrChk(cudaHostRegister((void*)&(EMf->getpXXsn().get(species,0,0,0)), gridSize*sizeof(cudaCommonType), cudaHostRegisterDefault));
  cudaErrChk(cudaHostRegister((void*)&(EMf->getpXYsn().get(species,0,0,0)), gridSize*sizeof(cudaCommonType), cudaHostRegisterDefault));
  cudaErrChk(cudaHostRegister((void*)&(EMf->getpXZsn().get(species,0,0,0)), gridSize*sizeof(cudaCommonType), cudaHostRegisterDefault));
  cudaErrChk(cudaHostRegister((void*)&(EMf->getpYYsn().get(species,0,0,0)), gridSize*sizeof(cudaCommonType), cudaHostRegisterDefault));
  cudaErrChk(cudaHostRegister((void*)&(EMf->getpYZsn().get(species,0,0,0)), gridSize*sizeof(cudaCommonType), cudaHostRegisterDefault));
  cudaErrChk(cudaHostRegister((void*)&(EMf->getpZZsn().get(species,0,0,0)), gridSize*sizeof(cudaCommonType), cudaHostRegisterDefault));
}

/**
 * @brief Unregister one species' host-side moment arrays from pinned memory.
 * @param species Species index whose host moment arrays are unpinned.
 */
void c_Solver::unregisterMomentsPinnedMemory(int species) {
  cudaErrChk(cudaHostUnregister((void*)&(EMf->getRHOns().get(species,0,0,0))));
  cudaErrChk(cudaHostUnregister((void*)&(EMf->getJxs().get(species,0,0,0))));
  cudaErrChk(cudaHostUnregister((void*)&(EMf->getJys().get(species,0,0,0))));
  cudaErrChk(cudaHostUnregister((void*)&(EMf->getJzs().get(species,0,0,0))));
  cudaErrChk(cudaHostUnregister((void*)&(EMf->getpXXsn().get(species,0,0,0))));
  cudaErrChk(cudaHostUnregister((void*)&(EMf->getpXYsn().get(species,0,0,0))));
  cudaErrChk(cudaHostUnregister((void*)&(EMf->getpXZsn().get(species,0,0,0))));
  cudaErrChk(cudaHostUnregister((void*)&(EMf->getpYYsn().get(species,0,0,0))));
  cudaErrChk(cudaHostUnregister((void*)&(EMf->getpYZsn().get(species,0,0,0))));
  cudaErrChk(cudaHostUnregister((void*)&(EMf->getpZZsn().get(species,0,0,0))));
}

void c_Solver::registerHeatFluxPinnedMemory() {
  const auto gridSize = grid->getNXN() * grid->getNYN() * grid->getNZN();
  const size_t bytes = static_cast<size_t>(ns) *
                       static_cast<size_t>(HeatFlux::ComponentCount) *
                       static_cast<size_t>(gridSize) *
                       sizeof(cudaCommonType);
  cudaErrChk(cudaHostRegister((void*)EMf->getHeatFluxRaw(), bytes,
                              cudaHostRegisterDefault));
}

void c_Solver::unregisterHeatFluxPinnedMemory() {
  cudaErrChk(cudaHostUnregister((void*)EMf->getHeatFluxRaw()));
}


/**
 * @brief Recompute all particle moments directly from the current GPU particle state.
 *
 * This path zeroes the device moment buffers, launches the full moment kernel
 * for every species, copies the results back to host field arrays, and then
 * waits for completion via MomentsAwait().
 */
void c_Solver::CalculateMoments() {

  // timeTasks_set_main_task(TimeTasks::MOMENTS);

  auto gridSize = grid->getNXN() * grid->getNYN() * grid->getNZN();
  for(int i=0; i<ns; i++){
    cudaErrChk(cudaMemsetAsync(momentsCUDAPtr[i], 0, gridSize*10*sizeof(cudaMomentType), streams[i]));
    // Particle data is already resident on the device; only the kernel launch is needed here.
    momentKernelNew<<<(pclsArrayHostPtr[i]->getNOP()/DEFAULT_BLOCK_SIZE + 1), DEFAULT_BLOCK_SIZE, 0, streams[i] >>>(momentParamCUDAPtr[i], grid3DCUDACUDAPtr, momentsCUDAPtr[i], 0);
#ifdef GPU_SOLVER
    // D2D scatter: packed moments → per-field GPU solver arrays (no host touch)
    EMf->gpuScatterMomentsD2D(momentsCUDAPtr[i], i, streams[i]);
#else
    copyMomentsD2H(i, streams[i]);
#endif
    // Record cycleEndEvent[i] on streams[i] so the per-species event-based
    // sync inside MomentsAwait() actually waits for this initial moment
    // pipeline. Without this record, MomentsAwait would observe a never-
    // recorded event (which makes cudaEventSynchronize return immediately)
    // and proceed to use stale / undefined host moment buffers in cycle 0.
    cudaErrChk(cudaEventRecord(cycleEndEvent[i], streams[i]));
  }

  // Synchronize all species before the field-side ghost exchange and reductions.
  MomentsAwait();

}


/**
 * @brief Advance the electric field solver for one cycle.
 * @param cycle Simulation cycle being advanced.
 */
void c_Solver::CalculateField(int cycle) {
  timeTasks_set_main_task(TimeTasks::FIELDS);

#ifdef GPU_SOLVER
  // Full GPU field solver — results stay in device arrays (d_Ex, d_Exth, ...)
  EMf->gpuCalculateE(cycle);
  // Record event so particle packing can wait for solver completion
  cudaErrChk(cudaEventRecord(solverDoneEvent, EMf->gpuSolverStream()));
#else
  // Legacy CPU field solver
  EMf->calculateE(cycle);
#endif
}

bool c_Solver::needsHeatFluxOutput(int cycle) const {
  if (!heatFluxEnabled_) return false;
  if (!Parameters::get_doWriteOutput()) return false;
  if (col->field_output_is_off()) return false;
  return (cycle % col->getFieldOutputCycle() == 0 || cycle == first_cycle);
}

void c_Solver::ScheduleHeatFlux(int cycle) {
  if (!needsHeatFluxOutput(cycle)) {
    heatFluxScheduledCycle_ = -1;
    return;
  }

  heatFluxScheduledCycle_ = cycle;
  const auto gridSize = grid->getNXN() * grid->getNYN() * grid->getNZN();
  constexpr cudaMomentType rhoFloor = 1.0e-30;

#ifdef GPU_SOLVER
  // Gate: heatFluxStream must not start reading d_rhons/d_Jxs/d_Jys/d_Jzs
  // until the full MomentsAwait() GPU pipeline on solverStream_ has completed
  // (gpuCommunicateGhostP2G_AllSpecies + gpuCalculateHatFunctions).
  // momentsPipelineDoneEvt is recorded at the tail of that pipeline, so this
  // single stream-wait covers all ns species simultaneously.
  cudaErrChk(cudaStreamWaitEvent(heatFluxStream, momentsPipelineDoneEvt, 0));
#endif

  for (int s = 0; s < ns; ++s) {
#ifdef GPU_SOLVER
    // D2D path: copy directly from live GPU solver moment arrays, avoiding the
    // stale-host bug and the heatFluxStream/solverStream_ host-buffer race.
    copyHeatFluxBulkD2D(s, heatFluxStream);
#else
    copyHeatFluxBulkH2D(s, heatFluxStream);
#endif
    cudaErrChk(cudaMemsetAsync(heatFluxCUDAPtr[s], 0,
                               gridSize * HeatFlux::ComponentCount *
                               sizeof(cudaMomentType),
                               heatFluxStream));
    const uint32_t nop = pclsArrayHostPtr[s]->getNOP();
    if (nop > 0) {
      heatFluxKernelUnsorted<<<getGridSize((int)nop, DEFAULT_BLOCK_SIZE),
                                DEFAULT_BLOCK_SIZE, 0, heatFluxStream>>>(
          momentParamCUDAPtr[s], grid3DCUDACUDAPtr,
          heatFluxBulkCUDAPtr[s], heatFluxCUDAPtr[s],
          (cudaMomentType)col->getQOM(s), rhoFloor);
    }
    cudaErrChk(cudaEventRecord(heatFluxReadDoneEvt[s], heatFluxStream));
  }

  for (int s = 0; s < ns; ++s) {
    copyHeatFluxD2H(s, heatFluxStream);
    cudaErrChk(cudaEventRecord(heatFluxHostDoneEvt[s], heatFluxStream));
  }
}

void c_Solver::finishHeatFluxForOutput(int cycle) {
  if (!heatFluxEnabled_ || heatFluxScheduledCycle_ != cycle) return;

  timeTasks_set_main_task(TimeTasks::MOMENTS);

  for (int s = 0; s < ns; ++s)
    cudaErrChk(cudaEventSynchronize(heatFluxHostDoneEvt[s]));

  for (int s = 0; s < ns; ++s)
    EMf->communicateGhostHeatFlux(s);

  heatFluxScheduledCycle_ = -1;
}

/**
 * @brief Pack the current mover/interpolation field buffer and copy it to the GPU.
 *
 * Macrocell spectra consumes this same packed B layout during data analysis,
 * before the mover path refreshes it for the current-cycle E solve. The caller
 * can request a host-side wait when the buffer must be ready immediately.
 */
void c_Solver::refreshFieldForPclsDeviceBuffer(bool synchronizeCopy)
{
#ifdef GPU_SOLVER
  // ---- GPU path: pack fields entirely on the GPU ----
  // gpuCalculateE / gpuCalculateB produce results directly in d_Ex etc.
  // No H2D sync needed — launch the GPU packing kernel immediately.
  //
  // Re-record solverDoneEvent here to capture the *current* tail of
  // solverStream_, not just the E-solver tail from the last CalculateField().
  // This is necessary because:
  //   - The mover path calls this after CalculateField(), so only E work is
  //     outstanding — the re-record is a cheap no-op in that case.
  //   - The analysis path (sortAllSpecies → refreshFieldForPclsDeviceBuffer)
  //     calls this BEFORE CalculateField(), meaning the previous cycle's
  //     CalculateB() and MomentsAwait() solver-stream work may still be
  //     queued. Without a fresh record here the wait below only gates on the
  //     previous E-solve record, letting gpuPackFieldForPclsToCenter race
  //     with writes to Bxn/Byn/Bzn from CalculateB(prev cycle).
  //   - On cycle 0 the event has never been recorded; re-recording it against
  //     the current (empty or H2D-initialized) stream makes the wait formally
  //     safe instead of a no-op.
  cudaErrChk(cudaEventRecord(solverDoneEvent, EMf->gpuSolverStream()));
  cudaErrChk(cudaStreamWaitEvent(streams[0], solverDoneEvent, 0));
  {
    const int ncells = (grid->getNXN() - 1) * (grid->getNYN() - 1) * grid->getNZN();
    const int blockSize = 256;
    const int gridDim   = (ncells + blockSize - 1) / blockSize;
    gpuPackFieldForPclsToCenter<<<gridDim, blockSize, 0, streams[0]>>>(
        fieldForPclCUDAPtr,
        EMf->gpuEx().devPtr(),     EMf->gpuEy().devPtr(),     EMf->gpuEz().devPtr(),
        EMf->gpuBxn().devPtr(),    EMf->gpuByn().devPtr(),    EMf->gpuBzn().devPtr(),
        EMf->gpuBx_ext().devPtr(), EMf->gpuBy_ext().devPtr(), EMf->gpuBz_ext().devPtr(),
        grid->getNXN(), grid->getNYN(), grid->getNZN());
  }
  cudaErrChk(cudaEventRecord(event0, streams[0]));
  if (synchronizeCopy) {
    cudaErrChk(cudaStreamSynchronize(streams[0]));
  }
#else
  // ---- CPU path: pack on host, then H2D copy on dedicated fieldH2DStream ----
  EMf->set_fieldForPclsToCenter(fieldForPclHostPtr);

  const size_t fieldValues =
      static_cast<size_t>(grid->getNZN()) *
      static_cast<size_t>(grid->getNYN() - 1) *
      static_cast<size_t>(grid->getNXN() - 1) * 24u;

  cudaErrChk(cudaMemcpyAsync(fieldForPclCUDAPtr, fieldForPclHostPtr,
                             fieldValues * sizeof(cudaFieldType),
                             cudaMemcpyDefault, fieldH2DStream));
  cudaErrChk(cudaEventRecord(event0, fieldH2DStream));
  if (synchronizeCopy) {
    cudaErrChk(cudaStreamSynchronize(fieldH2DStream));
  }
#endif
}

/**
 * @brief Launch the asynchronous GPU mover pipeline for one species.
 *
 * The method advances particles, extracts exit/planet populations, compacts the
 * stayed prefix, prepares the CPU-side communication buffer, and optionally
 * accumulates moments for the stayed particles in the unsorted pipeline.
 *
 * @param species Species index to advance.
 * @param doMomentsInLauncher Whether the stayed-prefix moments are accumulated here.
 * @return Number of particles removed from the stayed prefix
 *         (MPI exiting + deleted + planet-removed).
 */
int c_Solver::cudaLauncherAsync(const int species, const bool doMomentsInLauncher,
                                const bool waitForHeatFlux){
  cudaSetDevice(cudaDeviceOnNode); // Required when multiple MPI ranks share a node.
#if ENABLE_SOA_TIMING
  auto _tL0 = std::chrono::high_resolution_clock::now();
#endif

  // Explicit edge: gate every SoA-mutating kernel on this species' stream
  // behind the previous cycle's output D->H drain. Required because output
  // copies now run on the dedicated outputStream and are no longer ordered
  // implicitly with streams[species]. Pre-recorded at init so this resolves
  // immediately on cycle 0. Aux stream streams[species+ns] is transitively
  // ordered via moverHashedReadyEvt[species] later in this function.
  cudaErrChk(cudaStreamWaitEvent(streams[species],    eventOutputCopy, 0));
  cudaErrChk(cudaStreamWaitEvent(streams[species+ns], eventOutputCopy, 0));
  if (waitForHeatFlux) {
    cudaErrChk(cudaStreamWaitEvent(streams[species], heatFluxReadDoneEvt[species], 0));
    cudaErrChk(cudaStreamWaitEvent(streams[species+ns], heatFluxReadDoneEvt[species], 0));
  }

  // Persistent per-species events (created once in initCUDA, destroyed in
  // deInitCUDA) replacing what used to be per-call event1/event2.
  //   moverHashedReadyEvt[species] : recorded on streams[species]    after mover writes hashedSums
  //   auxHashedReadyEvt[species]   : recorded on streams[species+ns] after exitingKernel completes
  cudaEvent_t& event1 = moverHashedReadyEvt[species];
  cudaEvent_t& event2 = auxHashedReadyEvt[species];

  // ======= Particle count control: optional splitting =======
  //std::cout << "myrank: "<<MPIdata::get_rank() <<" pclsArrayHostPtr[species]->getInitialNOP(): " << pclsArrayHostPtr[species]->getInitialNOP() <<
  //          " pclsArrayHostPtr[species]->getNOP() " << pclsArrayHostPtr[species]->getNOP() << std::endl;
  if constexpr(PARTICLE_SPLITTING)
  {
    // Guard: splitting requires at least one source particle to duplicate.
    // The `multiple times` branch below divides by getNOP(), so an empty
    // species (e.g. catastrophic depletion) would otherwise trigger a
    // divide-by-zero and a zero-block kernel launch.
    if(pclsArrayHostPtr[species]->getNOP() > 0 &&
       pclsArrayHostPtr[species]->getNOP() < SPLIT_THRESHOLD * pclsArrayHostPtr[species]->getInitialNOP()){
      const uint32_t deltaPcl = pclsArrayHostPtr[species]->getInitialNOP() - pclsArrayHostPtr[species]->getNOP();
      if(deltaPcl < pclsArrayHostPtr[species]->getNOP()){
        std::cout << "Particle splitting basic myrank: "<< MPIdata::get_rank() << " species " << species <<" number particles: " << pclsArrayHostPtr[species]->getNOP() <<
                  " delta: " << deltaPcl <<std::endl;
        particleSplittingKernel<false><<<getGridSize((int)deltaPcl, DEFAULT_BLOCK_SIZE), DEFAULT_BLOCK_SIZE, 0, streams[species]>>>(moverParamCUDAPtr[species], grid3DCUDACUDAPtr);
        pclsArrayHostPtr[species]->setNOE(pclsArrayHostPtr[species]->getInitialNOP());
        cudaErrChk(cudaMemcpyAsync(pclsArrayCUDAPtr[species], pclsArrayHostPtr[species], sizeof(particleArrayCUDA), cudaMemcpyDefault, streams[species]));
        //cudaErrChk(cudaStreamSynchronize(streams[species+ns]));
      }
      else{ 
        // in this case the final number of particles will be < pclsArrayHostPtr[species]->getInitialNOP() 
        // worst case scenario will be pclsArrayHostPtr[species]->getInitialNOP() - (pclsArrayHostPtr[species]->getNOP() - 1)
        const int splittingTimes = deltaPcl / pclsArrayHostPtr[species]->getNOP();
        std::cout << "Particle splitting multipleTimesKernel myrank: "<< MPIdata::get_rank() << " species " << species <<" number particles: " << pclsArrayHostPtr[species]->getNOP() <<
                  " delta: " << deltaPcl <<std::endl;
        particleSplittingKernel<true><<<getGridSize((int)pclsArrayHostPtr[species]->getNOP(), DEFAULT_BLOCK_SIZE), DEFAULT_BLOCK_SIZE, 0, streams[species]>>>(moverParamCUDAPtr[species], grid3DCUDACUDAPtr);
        pclsArrayHostPtr[species]->setNOE( (splittingTimes + 1) * pclsArrayHostPtr[species]->getNOP());
        cudaErrChk(cudaMemcpyAsync(pclsArrayCUDAPtr[species], pclsArrayHostPtr[species], sizeof(particleArrayCUDA), cudaMemcpyDefault, streams[species]));
        
        //std::cerr << "Particle control multiple time splitting not yet implemented "<<std::endl;
      } 
    }
  }
  
  // ======= Launch mover kernels after field data is available =======
#if ENABLE_SOA_TIMING
  auto _tL1 = std::chrono::high_resolution_clock::now(); // after splitting, before mover launch
#endif
  cudaErrChk(cudaStreamWaitEvent(streams[species], event0, 0));
  // Empty species (e.g. planetary species before exosphere injection has
  // populated them) have NOP=0. Skip kernels that iterate per-particle;
  // downstream consumers already correctly handle zero exit/stayed counts.
  // Note: moments memset must still run so the D2H copy reads zeros.
  const uint32_t nop = pclsArrayHostPtr[species]->getNOP();
  if (nop > 0) {
    if (doPlanet_)
      moverSubcyclesKernel<<<getGridSize((int)nop, DEFAULT_BLOCK_SIZE), DEFAULT_BLOCK_SIZE, 0, streams[species]>>>(moverParamCUDAPtr[species], fieldForPclCUDAPtr, grid3DCUDACUDAPtr);
    else
      moverKernel<<<getGridSize((int)nop, DEFAULT_BLOCK_SIZE), DEFAULT_BLOCK_SIZE, 0, streams[species]>>>(moverParamCUDAPtr[species], fieldForPclCUDAPtr, grid3DCUDACUDAPtr);
  }

  cudaErrChk(cudaEventRecord(event1, streams[species]));
  // Unsorted pipeline: compute moments for stayed particles right after mover
  // (overlaps with hashedSum D2H + exitingKernel on streams[species+ns]).
  // Sorted pipeline: moments deferred to MoverAwaitAndPclExchange after sort.
  if (doMomentsInLauncher) {
    const auto gridSize = grid->getNXN() * grid->getNYN() * grid->getNZN();
    cudaErrChk(cudaMemsetAsync(momentsCUDAPtr[species], 0, gridSize*10*sizeof(cudaMomentType), streams[species]));
    if (nop > 0)
      momentKernelStayed<<<getGridSize((int)nop, DEFAULT_BLOCK_SIZE), DEFAULT_BLOCK_SIZE, 0, streams[species]>>>(
          &(moverParamCUDAPtr[species]->appendCountAtomic), momentParamCUDAPtr[species], grid3DCUDACUDAPtr, momentsCUDAPtr[species]);
  }
  // Mark that momentKernelStayed has finished reading appendCountAtomic and
  // pclsArrayCUDAPtr->nop_. This event is consumed below before the OpenBC
  // path resets the device counter and rewrites the particle metadata; it is
  // recorded unconditionally so the wait is well-defined on every cycle (in
  // the sorted path it just chains the empty cycle's mover, which is cheap).
  cudaErrChk(cudaEventRecord(stayedMomentsDoneEvt[species], streams[species]));

  // Copy 8 hashedSums to host: 6 directions + delete + planet (XLOW..PLANET)
  cudaErrChk(cudaStreamWaitEvent(streams[species+ns], event1, 0));
  cudaErrChk(cudaMemcpyAsync(hashedSumArrayHostPtr[species], hashedSumArrayCUDAPtr[species], 
    (departureArrayElementType::PLANET_HASHEDSUM_INDEX + 1)*sizeof(hashedSum), cudaMemcpyDefault, streams[species+ns]));

  // Copy OpenBC appended particle number to host
  if (moverParamHostPtr[species]->doOpenBC) {
    // The D->H read of appendCountAtomic is non-destructive; it only needs
    // mover ordering (event1 wait above) and may run in parallel with
    // momentKernelStayed.
    cudaErrChk(cudaMemcpyAsync(&moverParamHostPtr[species]->appendCountAtomic, &moverParamCUDAPtr[species]->appendCountAtomic, 
                                sizeof(uint32_t), cudaMemcpyDefault, streams[species+ns]));
    // The next two operations are destructive for momentKernelStayed:
    //   - cudaMemsetAsync zeros appendCountAtomic, which the kernel reads as
    //     `*appendCount` to compute totPcl = nop + appendCount.
    //   - The H->D rewrite of pclsArrayCUDAPtr below mutates pclsArray->nop_,
    //     which the kernel reads via momentParam->pclsArray->getNOP().
    // Without the wait below, both can land while the kernel is still
    // iterating, silently dropping (memset wins) or double-counting (H->D
    // wins) the OpenBC-appended particles in the moment deposition.
    cudaErrChk(cudaStreamWaitEvent(streams[species+ns], stayedMomentsDoneEvt[species], 0));
    cudaErrChk(cudaMemsetAsync(&moverParamCUDAPtr[species]->appendCountAtomic, 0, sizeof(uint32_t), streams[species+ns]));
    cudaErrChk(cudaStreamSynchronize(streams[species+ns]));

    const uint32_t newPclAfterOBC = pclsArrayHostPtr[species]->getNOP() + moverParamHostPtr[species]->appendCountAtomic;
    pclsArrayHostPtr[species]->setNOE(newPclAfterOBC);
    cudaErrChk(cudaMemcpyAsync(pclsArrayCUDAPtr[species], pclsArrayHostPtr[species], sizeof(particleArrayCUDA), cudaMemcpyDefault, streams[species+ns]));
  }


  // ======= Host-side accounting after mover completion =======
#if ENABLE_SOA_TIMING
  auto _tL2 = std::chrono::high_resolution_clock::now(); // before hashedSum sync
#endif
  cudaErrChk(cudaStreamSynchronize(streams[species+ns]));
#if ENABLE_SOA_TIMING
  auto _tL3 = std::chrono::high_resolution_clock::now(); // after hashedSum sync
#endif
  //cudaErrChk(cudaStreamSynchronize(streams[species]));
  int count_exiting = 0; // exiting particle number
  for(int i=0; i<departureArrayElementType::DELETE_HASHEDSUM_INDEX; i++)count_exiting += hashedSumArrayHostPtr[species][i].getSum();
  const int count_deleted = hashedSumArrayHostPtr[species][departureArrayElementType::DELETE_HASHEDSUM_INDEX].getSum(); // deleted particle number
  const int count_removed_planet = hashedSumArrayHostPtr[species][departureArrayElementType::PLANET_HASHEDSUM_INDEX].getSum(); // planet particle number
  planetPclCount[species] = count_removed_planet;
  const int hole = count_exiting + count_deleted + count_removed_planet;
  //if (count_deleted > 0){
  //  std::cout << " Particle holes myrank: "<< MPIdata::get_rank() << " species: " << species<< " hole: " << hole << " deleted: "<< count_deleted << std::endl;
  //}
  if(count_exiting > exitingArrayHostPtr[species]->getSize()){ 
    // Expand the exiting AoS buffer before extracting particles into it.
    exitingArrayHostPtr[species]->expand(count_exiting * EXPAND_GROWTH_FACTOR, streams[species+ns]);
    cudaErrChk(cudaMemcpyAsync(exitingArrayCUDAPtr[species], exitingArrayHostPtr[species], 
                                sizeof(exitingArray), cudaMemcpyDefault, streams[species+ns]));
  }

  if(hole > fillerBufferArrayHostPtr[species]->getSize()){
    // Expand the filler buffer used by the compaction/sorting kernels.
    fillerBufferArrayHostPtr[species]->expand(hole * EXPAND_GROWTH_FACTOR, streams[species+ns]);
    cudaErrChk(cudaMemcpyAsync(fillerBufferArrayCUDAPtr[species], fillerBufferArrayHostPtr[species], 
                                sizeof(fillerBuffer), cudaMemcpyDefault, streams[species+ns]));
  }

  // Planet extraction must run before exitingKernel, which overwrites
  // departureArray[].hashedId with HOLE hashes for front-region particles.
  // planetExtractionKernel needs the original PLANET hashedId to scatter correctly.
  if (count_removed_planet > 0) {
    if ((uint32_t)count_removed_planet > planetArrayHostPtr[species]->getSize()) {
      planetArrayHostPtr[species]->expand(count_removed_planet * EXPAND_GROWTH_FACTOR, streams[species+ns]);
      cudaErrChk(cudaMemcpyAsync(planetArrayCUDAPtr[species], planetArrayHostPtr[species],
                                  sizeof(planetArray), cudaMemcpyDefault, streams[species+ns]));
    }
    planetExtractionKernel<<<getGridSize((int)pclsArrayHostPtr[species]->getNOP(), DEFAULT_BLOCK_SIZE), DEFAULT_BLOCK_SIZE, 0, streams[species+ns]>>>(
        pclsArrayCUDAPtr[species], departureArrayCUDAPtr[species],
        planetArrayCUDAPtr[species], hashedSumArrayCUDAPtr[species]);
  }

#if ENABLE_SOA_TIMING
  auto _tL4 = std::chrono::high_resolution_clock::now(); // before exitingKernel
#endif
  // Re-read NOP: the OpenBC block above may have grown it via setNOE().
  const uint32_t nopAfterOBC = pclsArrayHostPtr[species]->getNOP();
  if (nopAfterOBC > 0)
    exitingKernel<<<getGridSize((int)nopAfterOBC, DEFAULT_BLOCK_SIZE), DEFAULT_BLOCK_SIZE, 0, streams[species+ns]>>>(pclsArrayCUDAPtr[species],
                  departureArrayCUDAPtr[species], exitingArrayCUDAPtr[species], hashedSumArrayCUDAPtr[species]);

  cudaErrChk(cudaEventRecord(event2, streams[species+ns]));

  // Prepare comm buffer and copy exiting particles D→H.
  particlesCommInj[species]->clearCommBuffer();
  if (count_exiting > 0) {
    particlesCommInj[species]->prepareCommBufferForNOP(count_exiting);
    cudaErrChk(cudaMemcpyAsync(particlesCommInj[species]->getCommPclsDataMut(),
                                exitingArrayHostPtr[species]->getArray(),
                                count_exiting * sizeof(SpeciesParticle),
                                cudaMemcpyDefault, streams[species+ns]));
  }

  // Compact the stayed prefix into the front of the SoA arrays.
  // (compactParticles1/2 perform hole-and-filler compaction only; no sorting.)
  cudaErrChk(cudaStreamWaitEvent(streams[species], event2, 0));
  const uint32_t stayedCount = pclsArrayHostPtr[species]->getNOP() - hole;
  if (hole > 0)
    compactParticles1<<<getGridSize(hole, SMALL_BLOCK_SIZE), SMALL_BLOCK_SIZE, 0, streams[species]>>>(pclsArrayCUDAPtr[species], departureArrayCUDAPtr[species],
                                                          fillerBufferArrayCUDAPtr[species], hashedSumArrayCUDAPtr[species]+departureArrayElementType::FILLER_HASHEDSUM_INDEX, hole);
  // Guard against the catastrophic edge case hole == NOP (every particle
  // departs in a single step). Launching with zero work would produce a
  // grid size of 0 and a CUDA invalid-configuration error.
  if (stayedCount > 0)
    compactParticles2<<<getGridSize((int)stayedCount, DEFAULT_BLOCK_SIZE), DEFAULT_BLOCK_SIZE, 0, streams[species]>>>(pclsArrayCUDAPtr[species], departureArrayCUDAPtr[species],
                                                          fillerBufferArrayCUDAPtr[species], hashedSumArrayCUDAPtr[species]+departureArrayElementType::HOLE_HASHEDSUM_INDEX, stayedCount);

  // event1/event2 are now persistent (initCUDA / deInitCUDA); no per-call destroy.
  cudaErrChk(cudaStreamSynchronize(streams[species+ns])); // exiting D→H complete, comm buffer ready

#if ENABLE_SOA_TIMING
  auto _tL5 = std::chrono::high_resolution_clock::now();
  if (MPIdata::get_rank() == 0) {
    printf("  [SoA launcher s%d: split=%.2f moverLaunch=%.2f hashedSync=%.2f "
           "expand+planet=%.2f exitKernel+D2H=%.2f total=%.2f ms  "
           "nop=%u exit=%d del=%d planet=%d]\n",
           species,
           std::chrono::duration<double, std::milli>(_tL1 - _tL0).count(),
           std::chrono::duration<double, std::milli>(_tL2 - _tL1).count(),
           std::chrono::duration<double, std::milli>(_tL3 - _tL2).count(),
           std::chrono::duration<double, std::milli>(_tL4 - _tL3).count(),
           std::chrono::duration<double, std::milli>(_tL5 - _tL4).count(),
           std::chrono::duration<double, std::milli>(_tL5 - _tL0).count(),
           pclsArrayHostPtr[species]->getNOP(), count_exiting, count_deleted, count_removed_planet);
  }
#endif

  return hole; // Number of exiting + deleted + planet particles
}

/**
 * @brief Launch the per-cycle particle mover workflow for all species.
 *
 * This prepares the field interpolation buffer, decides whether the current
 * cycle uses the sorted or unsorted moment pipeline, and enqueues one async
 * mover task per species on the solver thread pool.
 * @param cycle Simulation cycle being advanced.
 * @return Always `false`; retained for legacy caller compatibility.
 */
bool c_Solver::ParticlesMoverMomentAsync(int cycle)
{
  timeTasks_set_main_task(TimeTasks::PARTICLES);

  // ======= Decide whether to use the sorted particle pipeline this cycle =======
  sortThisCycle_ = (sortingCycle_ > 0) && (cycle % sortingCycle_ == 0);
  const bool doMomentsInLauncher = !sortThisCycle_;
  if (MPIdata::get_rank() == 0)
    printf("  [Cycle %d] Particle sorting: %s\n", cycle, sortThisCycle_ ? "ON" : "OFF");

  // Refresh the shared field-interpolation buffer for the mover. The per-
  // species mover streams consume it through cudaStreamWaitEvent(event0).
  refreshFieldForPclsDeviceBuffer(false);
  const bool waitForHeatFlux = heatFluxEnabled_ && (heatFluxScheduledCycle_ == cycle);

  for(int i=0; i<ns; i++){
    if (i != mergeIdx){
      exitingResults[i] = threadPoolPtr->enqueue(&c_Solver::cudaLauncherAsync, this, i, doMomentsInLauncher, waitForHeatFlux);
      toBeMerged[2 * i + 1] +=1;
    }
  }

  if (mergeIdx >= 0 && mergeIdx < ns) 
  {
    const auto& i = mergeIdx;
    std::cout << " Particle merging myrank: "<< MPIdata::get_rank() << " species: " << i << std::endl;

    // GPU cell sort — all NOP particles (merge requires cell ordering)
    const uint32_t mergeNop = pclsArrayHostPtr[i]->getNOP();
    cudaErrChk(cudaStreamSynchronize(streams[i]));  // ensure no kernels in flight
    // Explicit edge: also wait for the previous cycle's output D->H drain
    // before mutating SoA via sort/merge. The host sync above only waits for
    // streams[i]; outputStream is independent.
    cudaErrChk(cudaStreamWaitEvent(streams[i], eventOutputCopy, 0));
    if (waitForHeatFlux)
      cudaErrChk(cudaStreamWaitEvent(streams[i], heatFluxReadDoneEvt[i], 0));
    // Lazy-init: cellSorters[i] is only constructed in c_Solver::Init() when
    // sortingCycle_ > 0. The merging path however needs the sorter even when
    // periodic sorting is disabled (sortingCycle_ == 0); initialize on first
    // use to avoid undefined behavior in prepareBuffers/enqueueSortAsync.
    if (!cellSorters[i].initialized) {
      cellSorters[i].init(*grid3DCUDAHostPtr,
                          pclsArrayHostPtr[i]->getCapacity(),
                          streams[i]);
    }
    cellSorters[i].prepareBuffers(pclsArrayHostPtr[i], streams[i]);
    cellSorters[i].enqueueSortAsync(pclsArrayHostPtr[i], grid3DCUDACUDAPtr,
                                     mergeNop, streams[i]);
    cellSorters[i].finishSort(streams[i]);
    // Re-sync SoA pointers to device (sort did a pointer swap on host)
    cudaErrChk(cudaMemcpyAsync(pclsArrayCUDAPtr[i], pclsArrayHostPtr[i],
                                sizeof(particleArrayCUDA), cudaMemcpyDefault, streams[i]));

    // Merging kernel using CellSorter's device buffers
    const int totalCells = cellSorters[i].getNumCells();
    mergingKernel<<<getGridSize(totalCells * WARP_SIZE, DEFAULT_BLOCK_SIZE), DEFAULT_BLOCK_SIZE, 0, streams[i]>>>(
        const_cast<int*>(cellSorters[i].getCellStartOffsets()),
        cellSorters[i].getCellCounts(),
        grid3DCUDACUDAPtr, pclsArrayCUDAPtr[i], departureArrayCUDAPtr[i]);

    // Treat merge species as unsorted cycle: momentKernelStayed runs in launcher
    exitingResults[i] = threadPoolPtr->enqueue(&c_Solver::cudaLauncherAsync, this, i, true, waitForHeatFlux);

    toBeMerged[2 * i + 1] = 0;
    mergeIdx = -1; // merged
  }

  return (false);
}

/**
 * @brief Process planet-hit particles across all species on the dedicated planet stream.
 *
 * Ions are reduced to a total removed charge, electrons are energy-ranked, and
 * only the charge-balanced survivor subset is reflected back into the outgoing
 * communication buffers. Intermediate work stays on the GPU until the final
 * small host-side synchronization needed for reflected-particle counts.
 */
void c_Solver::processPlanetParticles()
{
#if ENABLE_SOA_TIMING
  auto _tP0 = std::chrono::high_resolution_clock::now();
#endif
  // ======= Step 1: Check whether any planet particles were collected =======
  int totalIonPlanet  = 0;
  int totalElecPlanet = 0;
  for (int i = 0; i < ns; i++) {
    if (planetPclCount[i] == 0) continue;
    if (col->getQOM(i) > 0)
      totalIonPlanet += planetPclCount[i];
    else
      totalElecPlanet += planetPclCount[i];
  }
  if (totalElecPlanet == 0) {
#if ENABLE_SOA_TIMING
    if (MPIdata::get_rank() == 0)
      printf("  [SoA planet: skip (0 elec) %.2f ms]\n",
             std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - _tP0).count());
#endif
    return;
  }

#if ENABLE_SOA_TIMING
  auto _tP1 = std::chrono::high_resolution_clock::now();
#endif
  // ======= Step 2: Reduce removed ion charge on the GPU =======
  cudaErrChk(cudaMemsetAsync(planetIonChargeDevice, 0, sizeof(cudaParticleType), planetStream));
  for (int i = 0; i < ns; i++) {
    if (col->getQOM(i) <= 0 || planetPclCount[i] == 0) continue;
    const int blockSize = DEFAULT_BLOCK_SIZE;
    const int gridSz = getGridSize(planetPclCount[i], blockSize);
    planetChargeReductionKernel<<<gridSz, blockSize, blockSize * sizeof(cudaParticleType), planetStream>>>(
        planetArrayCUDAPtr[i], planetPclCount[i], planetIonChargeDevice);
  }
  // No host sync is needed here: later kernels consume planetIonChargeDevice directly.

  // ======= Step 3: Resize shared work buffers as needed =======
  // Round up to the next power of 2 for bitonic sort.
  int nPad = 1;
  while (nPad < totalElecPlanet) nPad <<= 1;

  if (nPad > planetBufCapacity) {
    if (planetEnergyBuf)    cudaFree(planetEnergyBuf);
    if (planetGlobalIdxBuf) cudaFree(planetGlobalIdxBuf);
    planetBufCapacity = nPad;
    cudaErrChk(cudaMalloc(&planetEnergyBuf,    planetBufCapacity * sizeof(cudaParticleType)));
    cudaErrChk(cudaMalloc(&planetGlobalIdxBuf,  planetBufCapacity * sizeof(uint32_t)));
  }

  // Expand the reflected-particle output buffer if needed.
  if (totalElecPlanet > planetReflectedBufCapacity) {
    if (planetReflectedBuf) cudaFree(planetReflectedBuf);
    planetReflectedBufCapacity = totalElecPlanet * 2;
    cudaErrChk(cudaMalloc(&planetReflectedBuf, planetReflectedBufCapacity * sizeof(SpeciesParticle)));
  }

  // ======= Step 4: Compute per-electron energy and build species offsets =======
  int offset = 0;
  for (int e = 0; e < planetElecSpeciesCount; e++) {
    int specIdx = planetElecSpeciesMap[e];
    planetElecOffsets[e] = offset;
    if (planetPclCount[specIdx] > 0) {
      planetEnergyKernel<<<getGridSize(planetPclCount[specIdx], DEFAULT_BLOCK_SIZE), DEFAULT_BLOCK_SIZE, 0, planetStream>>>(
          planetArrayCUDAPtr[specIdx], planetPclCount[specIdx],
          (cudaParticleType)col->getQOM(specIdx),
          planetEnergyBuf, planetGlobalIdxBuf,
          offset);
    }
    offset += planetPclCount[specIdx];
  }

  // Copy offsets to device for the cutoff and reflection kernels.
  if (planetElecSpeciesCount > 0) {
    cudaErrChk(cudaMemcpyAsync(planetElecOffsetsDevice, planetElecOffsets,
                                planetElecSpeciesCount * sizeof(int), cudaMemcpyHostToDevice, planetStream));
  }

  // Refresh the device pointer array in case any per-species planet buffer expanded.
  {
    for (int e = 0; e < planetElecSpeciesCount; e++)
      planetTmpPtrs[e] = planetArrayCUDAPtr[planetElecSpeciesMap[e]];
    cudaErrChk(cudaMemcpyAsync(planetArrayCUDAPtrDevice, planetTmpPtrs,
                                planetElecSpeciesCount * sizeof(planetArray*), cudaMemcpyHostToDevice, planetStream));
  }

  // ======= Step 5: Sort electron planet particles by descending energy =======
  if (nPad > totalElecPlanet) {
    bitonicPadKernel<<<getGridSize(nPad - totalElecPlanet, DEFAULT_BLOCK_SIZE), DEFAULT_BLOCK_SIZE, 0, planetStream>>>(
        planetEnergyBuf, planetGlobalIdxBuf, totalElecPlanet, nPad);
  }
  for (int k = 2; k <= nPad; k <<= 1) {
    for (int j = k >> 1; j > 0; j >>= 1) {
      bitonicSortStepKernel<<<getGridSize(nPad, DEFAULT_BLOCK_SIZE), DEFAULT_BLOCK_SIZE, 0, planetStream>>>(
          planetEnergyBuf, planetGlobalIdxBuf, j, k, nPad);
    }
  }

  // ======= Step 6: Find the energy cutoff that matches the removed ion charge =======
  cudaErrChk(cudaMemsetAsync(planetCutoffDevice, 0, sizeof(int), planetStream));
  chargeCutoffKernel<<<1, 1, 0, planetStream>>>(
      planetArrayCUDAPtrDevice, planetElecSpeciesCount, planetElecOffsetsDevice,
      planetGlobalIdxBuf, totalElecPlanet,
      planetIonChargeDevice, planetCutoffDevice);
  // No host sync is needed here: the reflection kernel reads planetCutoffDevice directly.

  // ======= Step 7: Reflect and compact surviving electrons =======
  const int doSphere = moverParamHostPtr[0]->doSphere;
  const cudaCommonType originX = moverParamHostPtr[0]->sphereOrigin[0];
  const cudaCommonType originY = moverParamHostPtr[0]->sphereOrigin[1];
  const cudaCommonType originZ = moverParamHostPtr[0]->sphereOrigin[2];
  const cudaCommonType radius  = moverParamHostPtr[0]->sphereRadius;

  // Zero the per-species atomic survivor counters.
  cudaErrChk(cudaMemsetAsync(planetSurvivorCountDevice, 0,
                              planetElecSpeciesCount * sizeof(int), planetStream));

  // Launch one thread per electron candidate; each checks the device-side cutoff.
  const int reflectionType = col->getPlanetReflectionType();
  if (reflectionType == 1) {
    // Diffuse scattering matches the legacy isotropic reflection path.
    planetDiffuseCompactKernel<<<getGridSize(totalElecPlanet, DEFAULT_BLOCK_SIZE), DEFAULT_BLOCK_SIZE, 0, planetStream>>>(
        planetArrayCUDAPtrDevice, planetElecSpeciesCount,
        planetElecOffsetsDevice,
        planetGlobalIdxBuf,
        planetCutoffDevice,
        totalElecPlanet,
        planetReflectedBuf,
        planetSurvivorCountDevice,
        originX, originY, originZ, radius, doSphere,
        planetRngCycleCounter);
  } else {
    // Specular reflection is the default path.
    planetReflectCompactKernel<<<getGridSize(totalElecPlanet, DEFAULT_BLOCK_SIZE), DEFAULT_BLOCK_SIZE, 0, planetStream>>>(
        planetArrayCUDAPtrDevice, planetElecSpeciesCount,
        planetElecOffsetsDevice,
        planetGlobalIdxBuf,
        planetCutoffDevice,
        totalElecPlanet,
        planetReflectedBuf,
        planetSurvivorCountDevice,
        originX, originY, originZ, radius, doSphere);
  }
  planetRngCycleCounter++;

  // ======= Step 8: Copy survivor counts back to the host =======
  cudaErrChk(cudaMemcpyAsync(planetSurvivorCount, planetSurvivorCountDevice,
                              planetElecSpeciesCount * sizeof(int), cudaMemcpyDeviceToHost, planetStream));
  cudaErrChk(cudaStreamSynchronize(planetStream)); // Counts are needed on host for buffer growth and D2H copies.
#if ENABLE_SOA_TIMING
  auto _tP2 = std::chrono::high_resolution_clock::now();
#endif

  // ======= Step 9: Copy reflected particles into the per-species comm buffers =======
  for (int e = 0; e < planetElecSpeciesCount; e++) {
    if (planetSurvivorCount[e] == 0) continue;
    int specIdx = planetElecSpeciesMap[e];
    const int nRefl = planetSurvivorCount[e];
    const int commOffset = particlesCommInj[specIdx]->getCommNOP();
    // Grow the communication buffer before appending reflected particles.
    particlesCommInj[specIdx]->prepareCommBufferForNOP(commOffset + nRefl);
    // Copy the compact reflected AoS block into the next comm-buffer slot range.
    cudaErrChk(cudaMemcpyAsync(
        particlesCommInj[specIdx]->getCommPclsDataMut() + commOffset,
        planetReflectedBuf + planetElecOffsets[e],
        nRefl * sizeof(SpeciesParticle),
        cudaMemcpyDeviceToHost, planetStream));
  }

  // Complete all reflected-particle D2H copies before MPI exchange starts.
  cudaErrChk(cudaStreamSynchronize(planetStream));
#if ENABLE_SOA_TIMING
  auto _tP3 = std::chrono::high_resolution_clock::now();
  if (MPIdata::get_rank() == 0) {
    printf("  [SoA planet: setup=%.2f GPU(sort+reflect+sync)=%.2f D2H=%.2f total=%.2f ms  ions=%d elec=%d]\n",
           std::chrono::duration<double, std::milli>(_tP1 - _tP0).count(),
           std::chrono::duration<double, std::milli>(_tP2 - _tP1).count(),
           std::chrono::duration<double, std::milli>(_tP3 - _tP2).count(),
           std::chrono::duration<double, std::milli>(_tP3 - _tP0).count(),
           totalIonPlanet, totalElecPlanet);
  }
#endif
}

/**
 * @brief Wait for mover tasks, exchange particles through MPI, and finalize moments.
 *
 * This method joins the async mover futures, optionally processes planet hits,
 * overlaps MPI exchange with GPU sorting, appends incoming particles, and
 * finishes the moment accumulation path used by the current cycle.
 * @param cycle Simulation cycle being finalized.
 * @return Always `false`; retained for legacy caller compatibility.
 */
bool c_Solver::MoverAwaitAndPclExchange(int cycle)
{
#if ENABLE_SOA_TIMING
  auto _t0 = std::chrono::high_resolution_clock::now();
#endif

  // ======= Phase 3A: await mover and compaction futures =======
  for (int i = 0; i < ns; i++){ 
#if ENABLE_SOA_TIMING
    auto _ta = std::chrono::high_resolution_clock::now();
#endif
    auto x = exitingResults[i].get(); // holes
#if ENABLE_SOA_TIMING
    auto _tb = std::chrono::high_resolution_clock::now();
#endif
    stayedParticle[i] = pclsArrayHostPtr[i]->getNOP() - x;
#if ENABLE_SOA_TIMING
    if (MPIdata::get_rank() == 0)
      printf("  [SoA await s%d: %.2f ms  holes=%d stayed=%d]\n", i,
             std::chrono::duration<double, std::milli>(_tb - _ta).count(), x, stayedParticle[i]);
#endif
  }
#if ENABLE_SOA_TIMING
  auto _t1 = std::chrono::high_resolution_clock::now();
#endif

  // ======= Planet processing on the dedicated planet stream =======
  if (doPlanet_)
    processPlanetParticles();
#if ENABLE_SOA_TIMING
  auto _t2 = std::chrono::high_resolution_clock::now();
#endif

  // ======= Phase 3B: sync, pre-expand SoA, update NOP, prepare sort =======
  // After this phase no GPU kernels are in flight.  SoA arrays are large
  // enough for stayed + injection + estimated MPI incoming, so that a
  // future GPU injection kernel can write directly to the SoA tail.

  // Compute per-species injection count (cheap, cached).
  // Declared at function scope so Phase 3G can also use it.
  // std::vector with runtime size `ns` replaces the non-standard VLA form
  // `int injectedBCS[ns];` (GCC extension, not valid ISO C++).
  std::vector<int> injectedBCS(ns, 0);
  for (int i = 0; i < ns; i++) {
    if (moverParamHostPtr[i]->doRepopulateInjection)
      injectedBCS[i] = particlesCommInj[i]->computeInjectionCount();
  }
  {

    for (int i = 0; i < ns; i++) {
      // Sync compaction kernels so expand() can safely realloc.
      cudaErrChk(cudaStreamSynchronize(streams[i]));

      // Estimate: stayed + injected + exiting-as-proxy-for-incoming.
      const uint32_t estimatedMPI = particlesCommInj[i]->getCommNOP();
      const uint32_t estimatedTotal = (uint32_t)stayedParticle[i] + injectedBCS[i] + estimatedMPI;

      // Pre-expand SoA if needed (one allocation, before any kernel launch).
      if ((estimatedTotal * EXPAND_THRESHOLD_FACTOR) >= pclsArrayHostPtr[i]->getSize()) {
        pclsArrayHostPtr[i]->expand(
            static_cast<uint32_t>(estimatedTotal * EXPAND_GROWTH_FACTOR), streams[i]);
        departureArrayHostPtr[i]->expand(pclsArrayHostPtr[i]->getSize(), streams[i]);
        cudaErrChk(cudaMemcpyAsync(departureArrayCUDAPtr[i], departureArrayHostPtr[i],
                    sizeof(departureArrayType), cudaMemcpyDefault, streams[i]));
      }

      // Set host NOP to stayed + injected (injection count is known).
      // MPI particles will be appended on top in Phase 3G.
      pclsArrayHostPtr[i]->setNOE(stayedParticle[i] + injectedBCS[i]);
    }


    // ======= Launch GPU injection kernels (async on per-species stream) =======
    for (int i = 0; i < ns; i++) {
      if (injectedBCS[i] > 0) {
        // Per-cycle seed: deterministic but different each cycle+species.
        unsigned long long rngSeed =
            (unsigned long long)cycle * 6364136223846793005ULL +
            (unsigned long long)i;

        // Sync host metadata to device (SoA pointers may have changed after expand).
        cudaErrChk(cudaMemcpyAsync(pclsArrayCUDAPtr[i], pclsArrayHostPtr[i],
                    sizeof(particleArrayCUDA), cudaMemcpyDefault, streams[i]));

        injectionKernel<<<getGridSize(injectedBCS[i], DEFAULT_BLOCK_SIZE),
                          DEFAULT_BLOCK_SIZE, 0, streams[i]>>>(
            pclsArrayCUDAPtr[i],
            injectionParamCUDAPtr[i],
            (uint32_t)stayedParticle[i],
            rngSeed);
      }
    }

    if (sortThisCycle_) {
      for (int i = 0; i < ns; i++) {
        cellSorters[i].prepareBuffers(pclsArrayHostPtr[i], streams[i]);
      }

      // ======= Phase 3C: enqueue sort stages 1-3 for all species =======
      // Histogram, prefix sum, and sorted-index generation overlap with MPI below.
      for (int i = 0; i < ns; i++) {
        cellSorters[i].enqueueSortAsync(pclsArrayHostPtr[i],
                                         grid3DCUDACUDAPtr,
                                         stayedParticle[i] + injectedBCS[i],  // sort stayed + injected
                                         streams[i]);
      }
    }
  }
#if ENABLE_SOA_TIMING
  auto _t2b = std::chrono::high_resolution_clock::now();
#endif

  // ======= Phase 3D: perform MPI exchange while sort stages 1-3 run =======
  for (int i = 0; i < ns; i++)
  {
#if ENABLE_SOA_TIMING
    auto _mpi0 = std::chrono::high_resolution_clock::now();
#endif
    particlesCommInj[i]->separateAndSendParticles();
#if ENABLE_SOA_TIMING
    auto _mpi1 = std::chrono::high_resolution_clock::now();
#endif
    particlesCommInj[i]->recommunicateParticlesUntilDone(1);
#if ENABLE_SOA_TIMING
    auto _mpi2 = std::chrono::high_resolution_clock::now();
    if (MPIdata::get_rank() == 0)
      printf("  [SoA MPI s%d: separate=%.2f recomm=%.2f total=%.2f ms  commNOP=%d]\n", i,
             std::chrono::duration<double, std::milli>(_mpi1 - _mpi0).count(),
             std::chrono::duration<double, std::milli>(_mpi2 - _mpi1).count(),
             std::chrono::duration<double, std::milli>(_mpi2 - _mpi0).count(),
             particlesCommInj[i]->getCommNOP());
#endif
  }
#if ENABLE_SOA_TIMING
  auto _t3 = std::chrono::high_resolution_clock::now();
#endif

  // ======= Exosphere ionization =======
  injectExosphereParticles();
#if ENABLE_SOA_TIMING
  auto _t4 = std::chrono::high_resolution_clock::now();
  if (MPIdata::get_rank() == 0)
    printf("  [SoA exosphere: %.2f ms]\n",
           std::chrono::duration<double, std::milli>(_t4 - _t3).count());
#endif

  // ======= Phase 3E: finish sorting for all species =======
  // After this, the host particleArrayCUDA objects contain the sorted SoA pointers.
  if (sortThisCycle_) {
    for (int i = 0; i < ns; i++) {
      cellSorters[i].finishSort(streams[i]);
      // Re-sync the device-side metadata after the host-side pointer swap.
      cudaErrChk(cudaMemcpyAsync(pclsArrayCUDAPtr[i], pclsArrayHostPtr[i],
                                  sizeof(particleArrayCUDA), cudaMemcpyDefault, streams[i]));
    }
  } 

  // ======= Phase 3F: sync all streams before mutating SoA allocations =======
  // expand() may allocate, copy, and free SoA buffers, so no kernels may be active.

  // Unsorted pipeline: wait for mover/compaction kernels before any expand() or tail moments.
  for (int i = 0; i < ns; i++) 
    cudaErrChk(cudaStreamSynchronize(streams[i]));

    #if ENABLE_SOA_TIMING
  auto _t4b = std::chrono::high_resolution_clock::now();
#endif

  // ======= Phase 3G: expand, append incoming particles, and finalize moments =======
  const auto momentGridSize = grid->getNXN() * grid->getNYN() * grid->getNZN();

  for (int i = 0; i < ns; i++) {

    // Total particles = stayed + GPU-injected + MPI incoming.
    const int sortedCount = stayedParticle[i] + injectedBCS[i];
    auto newPclNum = sortedCount + particlesCommInj[i]->getCommNOP();

    // Expand SoA arrays if needed.
    // Safe point: Phase 3F guarantees no kernels are using these buffers.
    if ((newPclNum * EXPAND_THRESHOLD_FACTOR) >= pclsArrayHostPtr[i]->getSize()) {
      pclsArrayHostPtr[i]->expand(newPclNum * EXPAND_GROWTH_FACTOR, streams[i]);
      departureArrayHostPtr[i]->expand(pclsArrayHostPtr[i]->getSize(), streams[i]);
      cudaErrChk(cudaMemcpyAsync(departureArrayCUDAPtr[i], departureArrayHostPtr[i], sizeof(departureArrayType), cudaMemcpyDefault, streams[i]));
    }

    // Copy incoming AoS particles to the device staging buffer.
    const int incomingCount = particlesCommInj[i]->getCommNOP();
    if (incomingCount > 0) {
      if (static_cast<uint32_t>(incomingCount) > incomingStagingHostPtr[i]->getSize()) {
        incomingStagingHostPtr[i]->expand(incomingCount * EXPAND_GROWTH_FACTOR, streams[i]);
        cudaErrChk(cudaMemcpyAsync(incomingStagingCUDAPtr[i], incomingStagingHostPtr[i],
                    sizeof(arrayCUDA<SpeciesParticle>), cudaMemcpyDefault, streams[i]));
      }
      cudaErrChk(cudaMemcpyAsync(incomingStagingHostPtr[i]->getArray(),
                particlesCommInj[i]->getCommPclsData(),
                incomingCount * sizeof(SpeciesParticle),
                cudaMemcpyDefault, streams[i]));
    }

    // Update NOP to the new total and copy the host metadata back to the device.
    pclsArrayHostPtr[i]->setNOE(newPclNum);
    cudaErrChk(cudaMemcpyAsync(pclsArrayCUDAPtr[i], pclsArrayHostPtr[i],
                                sizeof(particleArrayCUDA), cudaMemcpyDefault, streams[i]));

    // Scatter the incoming AoS particles into the SoA tail after stayed+injected.
    if (incomingCount > 0)
      scatterAoSToSoAKernel<<<getGridSize(incomingCount, DEFAULT_BLOCK_SIZE), DEFAULT_BLOCK_SIZE, 0, streams[i]>>>(
          incomingStagingHostPtr[i]->getArray(),
          pclsArrayCUDAPtr[i], (uint32_t)sortedCount, (uint32_t)incomingCount,
          particlesHost[i]->getParticleIDGenerator());

    // Finalize moments according to the active sorted/unsorted pipeline.
    if (sortThisCycle_) {
      // Sorted path: cell-aware moments for the stayed prefix, flat kernel for the incoming tail.
      cudaErrChk(cudaMemsetAsync(momentsCUDAPtr[i], 0,
                                  momentGridSize * 10 * sizeof(cudaMomentType), streams[i]));

      const uint32_t numSorted = sortedCount;
      if (numSorted > 0) {
        const int numCells = cellSorters[i].getNumCells();
        const int warps = numCells;
        const int threads = warps * WARP_SIZE;
        cellAwareMomentKernel<<<getGridSize(threads, DEFAULT_BLOCK_SIZE), DEFAULT_BLOCK_SIZE, 0, streams[i]>>>(
            cellSorters[i].getCellStartOffsets(),
            numCells,
            numSorted,
            pclsArrayCUDAPtr[i],
            grid3DCUDACUDAPtr,
            momentsCUDAPtr[i]);
      }

      const int tailCount = newPclNum - sortedCount;
      if (tailCount > 0)
        momentKernelNew<<<getGridSize(tailCount, SMALL_BLOCK_SIZE), SMALL_BLOCK_SIZE, 0, streams[i]>>>(
            momentParamCUDAPtr[i], grid3DCUDACUDAPtr, momentsCUDAPtr[i], sortedCount);
    } else {
      // Unsorted path: the stayed prefix was already handled in cudaLauncherAsync().
      // Only the injected + incoming tail needs momentKernelNew.
      const int tailCount = newPclNum - stayedParticle[i];
      if (tailCount > 0)
        momentKernelNew<<<getGridSize(tailCount, SMALL_BLOCK_SIZE), SMALL_BLOCK_SIZE, 0, streams[i]>>>(
            momentParamCUDAPtr[i], grid3DCUDACUDAPtr, momentsCUDAPtr[i], stayedParticle[i]);
    }

    // Reset hashed sums and departure flags for the next cycle.
    for (int j = 0; j < departureArrayElementType::HASHED_SUM_NUM; j++)
      hashedSumArrayHostPtr[i][j].resetBucket();
    cudaErrChk(cudaMemcpyAsync(hashedSumArrayCUDAPtr[i], hashedSumArrayHostPtr[i],
                                (departureArrayElementType::HASHED_SUM_NUM) * sizeof(hashedSum),
                                cudaMemcpyDefault, streams[i]));
    cudaErrChk(cudaMemsetAsync(departureArrayHostPtr[i]->getArray(), 0,
                                departureArrayHostPtr[i]->getSize() * sizeof(departureArrayElementType),
                                streams[i]));
  }

#if ENABLE_SOA_TIMING
  auto _t5a = std::chrono::high_resolution_clock::now();
#endif
  for (int i = 0; i < ns; i++) {
#ifdef GPU_SOLVER
    // D2D scatter: packed moments → per-field GPU solver arrays (no host touch)
    EMf->gpuScatterMomentsD2D(momentsCUDAPtr[i], i, streams[i]);
#else
    copyMomentsD2H(i, streams[i]);
#endif
    // Cycle-end marker on streams[i]. Happens-after every in-place SoA write
    // of this cycle (compactParticles2 / scatterAoSToSoAKernel) and after the
    // moment kernels and the moment D->H copy on the same stream. Consumers:
    //   - MomentsAwait()    : per-species cudaEventSynchronize (host)
    //   - outputCopyAsync() : cudaStreamWaitEvent on outputStream
    //   - cudaLauncherAsync (next cycle, indirectly via eventOutputCopy)
    cudaErrChk(cudaEventRecord(cycleEndEvent[i], streams[i]));
  }

#if ENABLE_SOA_TIMING
  auto _t5 = std::chrono::high_resolution_clock::now();
  if (MPIdata::get_rank() == 0) {
    printf("  [SoA exchange TOTAL: await=%.2f planet=%.2f sort_prep+enqueue=%.2f MPI=%.2f "
           "exosphere=%.2f sort_finish+sync=%.2f expand+H2D+moments=%.2f momD2H=%.2f total=%.2f ms]\n",
           std::chrono::duration<double, std::milli>(_t1 - _t0).count(),
           std::chrono::duration<double, std::milli>(_t2 - _t1).count(),
           std::chrono::duration<double, std::milli>(_t2b - _t2).count(),
           std::chrono::duration<double, std::milli>(_t3 - _t2b).count(),
           std::chrono::duration<double, std::milli>(_t4 - _t3).count(),
           std::chrono::duration<double, std::milli>(_t4b - _t4).count(),
           std::chrono::duration<double, std::milli>(_t5a - _t4b).count(),
           std::chrono::duration<double, std::milli>(_t5 - _t5a).count(),
           std::chrono::duration<double, std::milli>(_t5 - _t0).count());
  }
#endif

  return (false);
}

/**
 * @brief Advance the magnetic field solver for one cycle.
 *
 * This assumes the electric field has already been updated for the same cycle.
 * @param cycle Simulation cycle being advanced.
 */
void c_Solver::CalculateB(int cycle) {
  timeTasks_set_main_task(TimeTasks::FIELDS);

  auto tB0 = std::chrono::high_resolution_clock::now();
#ifdef GPU_SOLVER
  // Full GPU B solver — results stay in device arrays (d_Bxc, d_Bxn, ...)
  EMf->gpuCalculateB(cycle);
#else
  // Legacy CPU B solver
  EMf->calculateB(cycle);
#endif
  auto tB1 = std::chrono::high_resolution_clock::now();
  //if (MPIdata::get_rank() == 0) {
  //  double ms = std::chrono::duration<double, std::milli>(tB1 - tB0).count();
  //  std::cout << "[CalculateB] total: " << ms << " ms" << std::endl;
  //}
}

/**
 * @brief Synchronize moment accumulation and prepare field-side derived quantities.
 *
 * After all D2H moment copies complete, this method optionally schedules
 * particle merging, communicates ghost moments, applies special boundary
 * charge sources, and computes the field-solver derived moment quantities.
 */
void c_Solver::MomentsAwait() {

  timeTasks_set_main_task(TimeTasks::MOMENTS);

  // Wait for all per-species moment kernels and D->H copies to finish.
  // cycleEndEvent[i] is recorded on streams[i] right after copyMomentsD2H(i),
  // so a per-species cudaEventSynchronize is sufficient and avoids a global
  // device sync that would also drain unrelated streams (planetStream is
  // already host-synced inside processPlanetParticles; outputStream of the
  // previous cycle, if any, must NOT be waited on here — the next cycle's
  // mover/sort already explicitly waits on eventOutputCopy).
  for (int i = 0; i < ns; ++i)
    cudaErrChk(cudaEventSynchronize(cycleEndEvent[i]));

  if constexpr(PARTICLE_MERGING)
  {
    // Mark species whose particle count exceeds the merge threshold.
    for(int i = 0; i < ns; i++) {
      if(pclsArrayHostPtr[i]->getNOP() > MERGE_THRESHOLD * pclsArrayHostPtr[i]->getInitialNOP()) {
        toBeMerged[2 * i] = 1;
      }
      else{
        toBeMerged[2 * i] = 0;
      }
    }
    // Select the species that has gone the longest without merging.
    mergeIdx = -1;
    int mergeCountFromLast = -1;
    for(int i=0;i<ns;i++){
        if( (toBeMerged[2 * i] == 1) && (toBeMerged[2 * i + 1] > mergeCountFromLast) ){
          mergeIdx = i;
          mergeCountFromLast = toBeMerged[2 * i + 1];
        }
    }

    if (mergeIdx >= 0 && mergeIdx < ns){
      // GPU sort + merge happens in ParticlesMoverMomentAsync — no D→H needed here.
      // Just record that this species is scheduled for merging.
    }
  }
  else
  {
    mergeIdx = -1; 
  }

#ifdef GPU_SOLVER
  // ---- Full GPU moment-processing pipeline (no host touch) ----
  // Phase 1: all-species ghost exchange batched (2 halo exchanges instead of 2*ns)
  auto tGhostStart = std::chrono::high_resolution_clock::now();
  EMf->gpuCommunicateGhostP2G_AllSpecies();
  auto tGhostEnd = std::chrono::high_resolution_clock::now();
  //if (myrank == 0) {
  //  double ms = std::chrono::duration<double, std::milli>(tGhostEnd - tGhostStart).count();
  //  std::cout << "[MomentsAwait] gpuCommunicateGhostP2G_AllSpecies: " << ms << " ms" << std::endl;
  //}

  // Phase 2: zero derived quantities, planet charge fix, sum over species, interp N→C
  EMf->gpuSetZeroDerivedMoments();
  // Enforce constant charge inside planet BEFORE summing over species,
  // so that rhon (and downstream rhoc, rhoh) includes the planet fix.
  if (col->getCase() == "Dipole") {
    EMf->gpuConstantChargePlanet(col->getPlanet_radius(),
        col->getx_center_planet(), col->gety_center_planet(), col->getz_center_planet());
  } else if (col->getCase() == "Dipole2D") {
    EMf->gpuConstantChargePlanet2DPlaneXZ(col->getPlanet_radius(),
        col->getx_center_planet(), col->getz_center_planet());
  }
  EMf->gpuSumOverSpecies();
  if (col->getOutputConfig().needsJTotComputation())
    EMf->gpuSumOverSpeciesJ();
  EMf->gpuInterpDensitiesN2C();

  // Phase 3: hat functions (Jhat, rhohat) — already GPU-implemented
  EMf->gpuCalculateHatFunctions();

  // Record momentsPipelineDoneEvt at the tail of the solver stream so that
  // ScheduleHeatFlux() (next cycle, step 2) can gate its D2D copy of
  // d_rhons/d_Jxs/d_Jys/d_Jzs until the ghost exchange and hat functions
  // have committed all writes to those arrays.  Without this gate, the D2D
  // copy could read partially-updated ghost cells from gpuCommunicateGhostP2G.
  if (heatFluxEnabled_)
    cudaErrChk(cudaEventRecord(momentsPipelineDoneEvt, EMf->gpuSolverStream()));

  auto tEnd = std::chrono::high_resolution_clock::now();
  //if (myrank == 0) {
  //  double ghostMs  = std::chrono::duration<double, std::milli>(tGhostEnd - tGhostStart).count();
  //  double phaseMs  = std::chrono::duration<double, std::milli>(tEnd - tGhostEnd).count();
  //  double totalMs  = std::chrono::duration<double, std::milli>(tEnd - t0).count();
  //  std::cout << "[MomentsAwait] ghost total: " << ghostMs << " ms, phase2+3: " << phaseMs
  //            << " ms, total: " << totalMs << " ms" << std::endl;
  //}
#else
  // ---- Legacy CPU moment-processing pipeline ----
  for (int i = 0; i < ns; i++)
    EMf->communicateGhostP2G(i);

  EMf->setZeroDerivedMoments();
  // Fill the planet interior with the constant charge used by the legacy boundary model.
  if (caseType_ == CaseType::Dipole) {
    EMf->ConstantChargePlanet(col->getPlanet_radius(),col->getx_center_planet(),col->gety_center_planet(),col->getz_center_planet());
  } else if (caseType_ == CaseType::Dipole2D) {
    EMf->ConstantChargePlanet2DPlaneXZ(col->getPlanet_radius(),col->getx_center_planet(),col->getz_center_planet());
  }
  // Legacy OpenBC constant-charge path is intentionally left disabled here.
  //EMf->ConstantChargeOpenBC();
  // Accumulate species moments into total moments.
  EMf->sumOverSpecies();
  // Total J (Jx/Jy/Jz) is computed on demand inside each output writer
  // via sumOverSpeciesJ(), which zeroes before accumulating.
  // Interpolate nodal densities to cell centers.
  EMf->interpDensitiesN2C();
  // Compute the hat quantities required by the implicit field solve.
  EMf->calculateHatFunctions();
#endif
}

/**
 * @brief Append the current per-species particle counts to the rank-local CSV log.
 * @param cycle Simulation cycle associated with the recorded counts.
 */
void c_Solver::writeParticleNum(int cycle) {
  if (!pclNumCSV.is_open() || !pclNumCSV.good()) {
    // Reopen if stream is in a bad state
    pclNumCSV.close();
    pclNumCSV.clear();
    pclNumCSV.open(col->getSaveDirName() + "/particleNum" + std::to_string(myrank) + ".csv", std::ios::app);
  }
  pclNumCSV << cycle << ",";
  for(int i=0; i<ns-1; i++){
    pclNumCSV << pclsArrayHostPtr[i]->getNOP() << ",";
  }
  pclNumCSV << pclsArrayHostPtr[ns-1]->getNOP() << std::endl;
  pclNumCSV.flush();
}


/**
 * @brief Write restart, field, particle, and test-particle outputs for one cycle.
 *
 * Particle and restart outputs wait on outputCopyAsync() so the host-side SoA
 * mirrors are up to date before the IOManager consumes them.
 * @param cycle Simulation cycle being written.
 */
void c_Solver::WriteOutput(int cycle) {

#ifdef GPU_SOLVER
  // ---- GPU solver: sync device → host only when I/O actually needs host arrays ----
  {
    bool needFieldSync = false;
    // Diagnostics (energy conservation) need host E, B, rhons, Jxs
    if (col->getDiagnosticsOutputCycle() > 0 && cycle % col->getDiagnosticsOutputCycle() == 0)
      needFieldSync = true;
    // Restart checkpoint needs host fields
    if (restart_cycle > 0 && cycle % restart_cycle == 0)
      needFieldSync = true;
    // Field file output needs host fields
    if (Parameters::get_doWriteOutput() && !col->field_output_is_off() &&
        (cycle % col->getFieldOutputCycle() == 0 || cycle == first_cycle))
      needFieldSync = true;
    if (needFieldSync) {
      // gpuSolverSyncD2H internally synchronises the solver stream after
      // issuing all D2H copies, so no separate cudaStreamSynchronize is needed.
      EMf->gpuSolverSyncD2H(EMf->gpuSolverStream());
    }
  }
#endif

#ifdef USE_CATALYST
  Adaptor::CoProcess(col->getDt()*cycle, cycle, EMf);
#endif

  WriteConserved(cycle);

  // ======= Restart checkpoint =======
  if (restart_cycle > 0 && cycle % restart_cycle == 0) {
    // Periodic restarts are written before the current iteration is completed.
    // The checkpoint label remains the triggering loop cycle, so a checkpoint
    // labeled N restarts by executing cycle N again.
    // eventOutputCopy is pre-recorded once at init on outputStream and re-
    // recorded by every outputCopyAsync() that actually issues copies. The
    // synchronize below is therefore always well-defined. First-cycle output
    // is primed during Init() when host particle data is required.
    cudaErrChk(cudaEventSynchronize(eventOutputCopy));
    prepareActiveRestartParticleCellMetadata();
    // SoA data is already in host vectors after outputCopyAsync — no conversion needed
    ioManager->writeRestart(cycle);
  }

  if (!Parameters::get_doWriteOutput()) return;

  // ======= Field output =======
  if (!col->field_output_is_off() &&
      (cycle % col->getFieldOutputCycle() == 0 || cycle == first_cycle)) {
    finishHeatFluxForOutput(cycle);
    ioManager->writeFields(cycle);
  }

  // ======= Particle output =======
  if (!col->particle_output_is_off() &&
      cycle % col->getParticlesOutputCycle() == 0) {
    // See comment above: the event is pre-recorded so this is always safe.
    cudaErrChk(cudaEventSynchronize(eventOutputCopy));
    // SoA data is already in host vectors after outputCopyAsync — no conversion needed
    ioManager->writeParticles(cycle);
  }

  // ======= Test-particle output =======
  if (nstestpart > 0 && !col->testparticle_output_is_off() &&
      cycle % col->getTestParticlesOutputCycle() == 0) {
    ioManager->writeTestParticles(cycle);
  }
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

  restartParticleCellMetadata_.resize(
      ns, guardedNx - 2, guardedNy - 2, guardedNz - 2);
}

void c_Solver::copyRestartParticleCellMetadataFromDevice(
    int species, cudaStream_t stream)
{
  const int guardedCells = grid->getNXC() * grid->getNYC() * grid->getNZC();
  if (pclsArrayHostPtr[species]->getNOP() == 0) {
    std::fill(restartGuardedCellOffsets_[species].begin(),
              restartGuardedCellOffsets_[species].end(), 0);
    std::fill(restartGuardedCellCounts_[species].begin(),
              restartGuardedCellCounts_[species].end(), 0);
    return;
  }

  cudaErrChk(cudaMemcpyAsync(
      restartGuardedCellOffsets_[species].data(),
      cellSorters[species].getCellStartOffsets(),
      guardedCells * sizeof(int), cudaMemcpyDeviceToHost, stream));
  cudaErrChk(cudaMemcpyAsync(
      restartGuardedCellCounts_[species].data(),
      cellSorters[species].getCellCounts(),
      guardedCells * sizeof(int), cudaMemcpyDeviceToHost, stream));
}

void c_Solver::prepareActiveRestartParticleCellMetadata() {
  ensureRestartParticleCellMetadataBuffers();

  const int guardedNx = grid->getNXC();
  const int guardedNy = grid->getNYC();
  const int guardedNz = grid->getNZC();
  const int expectedActiveCells =
      restartParticleCellMetadata_.activeCellCount();

  for (int s = 0; s < ns; ++s) {
    auto& activeOffsets =
        restartParticleCellMetadata_.species[s].cellOffsets;
    auto& activeCounts =
        restartParticleCellMetadata_.species[s].cellCounts;

    int activeIndex = 0;
    for (int gz = 1; gz < guardedNz - 1; ++gz) {
      for (int gy = 1; gy < guardedNy - 1; ++gy) {
        for (int gx = 1; gx < guardedNx - 1; ++gx) {
          const int guardedIndex = gx + gy * guardedNx
                                 + gz * guardedNx * guardedNy;
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

/**
 * @brief Schedule asynchronous GPU-to-host particle copies for a future output cycle.
 *
 * The solver calls this one cycle ahead of output. Passing `cycle == -1`
 * forces the copy path used during final restart writing.
 * @param cycle Current simulation cycle used to predict the next output step.
 */
void c_Solver::outputCopyAsync(int cycle) {
  if (ioManager->needsParticleSync(cycle + 1)) {
    const bool restartSync = ioManager->needsRestartParticleSync(cycle + 1);
    if (restartSync) {
      ensureRestartParticleCellMetadataBuffers();
    }

    // Explicit edge: wait for every species' end-of-cycle event before reading
    // particle SoA from device. cycleEndEvent[s] is recorded on streams[s] at
    // the end of MoverAwaitAndPclExchange and so happens-after every in-place
    // SoA write of the current cycle. First-cycle output priming uses the
    // events pre-recorded during CUDA initialization. Using a
    // dedicated outputStream isolates the D->H copies from the per-species
    // streams used by the next cycle's mover/sort, so those kernels can run
    // ahead in parallel with output as long as they wait on eventOutputCopy.
    for (int s = 0; s < ns; ++s) {
      if (restartSync) {
        cudaErrChk(cudaStreamWaitEvent(streams[s], cycleEndEvent[s], 0));
        cudaErrChk(cudaStreamWaitEvent(streams[s], eventOutputCopy, 0));

        const uint32_t nop = pclsArrayHostPtr[s]->getNOP();
        if (nop > 0) {
          if (!cellSorters[s].initialized) {
            cellSorters[s].init(*grid3DCUDAHostPtr,
                                pclsArrayHostPtr[s]->getCapacity(),
                                streams[s]);
          }
          cellSorters[s].prepareBuffers(pclsArrayHostPtr[s], streams[s]);
          cellSorters[s].enqueueSortAsync(pclsArrayHostPtr[s],
                                          grid3DCUDACUDAPtr,
                                          nop, streams[s]);
          cellSorters[s].finishSort(streams[s]);
          cudaErrChk(cudaMemcpyAsync(
              pclsArrayCUDAPtr[s], pclsArrayHostPtr[s],
              sizeof(particleArrayCUDA), cudaMemcpyDefault, streams[s]));
        }

        copyRestartParticleCellMetadataFromDevice(s, streams[s]);
        cudaErrChk(cudaEventRecord(cycleEndEvent[s], streams[s]));
      }
      cudaErrChk(cudaStreamWaitEvent(outputStream, cycleEndEvent[s], 0));
    }

    for (int i = 0; i < ns; i++) {
      const uint32_t nop = pclsArrayHostPtr[i]->getNOP();
      // Resize the host SoA vectors to receive the current particle count.
      particlesHost[i]->prepareSoAForNOP(nop);
      if (nop == 0) continue;
      // Direct GPU SoA -> host SoA transfer with no AoS intermediary.
      cudaErrChk(cudaMemcpyAsync(particlesHost[i]->getUallMut(), pclsArrayHostPtr[i]->getU(), nop * sizeof(cudaPclType_U), cudaMemcpyDefault, outputStream));
      cudaErrChk(cudaMemcpyAsync(particlesHost[i]->getVallMut(), pclsArrayHostPtr[i]->getV(), nop * sizeof(cudaPclType_V), cudaMemcpyDefault, outputStream));
      cudaErrChk(cudaMemcpyAsync(particlesHost[i]->getWallMut(), pclsArrayHostPtr[i]->getW(), nop * sizeof(cudaPclType_W), cudaMemcpyDefault, outputStream));
      cudaErrChk(cudaMemcpyAsync(particlesHost[i]->getQallMut(), pclsArrayHostPtr[i]->getQ(), nop * sizeof(cudaPclType_Q), cudaMemcpyDefault, outputStream));
      cudaErrChk(cudaMemcpyAsync(particlesHost[i]->getXallMut(), pclsArrayHostPtr[i]->getX(), nop * sizeof(cudaPclType_X), cudaMemcpyDefault, outputStream));
      cudaErrChk(cudaMemcpyAsync(particlesHost[i]->getYallMut(), pclsArrayHostPtr[i]->getY(), nop * sizeof(cudaPclType_Y), cudaMemcpyDefault, outputStream));
      cudaErrChk(cudaMemcpyAsync(particlesHost[i]->getZallMut(), pclsArrayHostPtr[i]->getZ(), nop * sizeof(cudaPclType_Z), cudaMemcpyDefault, outputStream));
      if (particlesHost[i]->tracksParticleID()) {
        cudaErrChk(cudaMemcpyAsync(particlesHost[i]->getIDallMut(), pclsArrayHostPtr[i]->getID(), nop * sizeof(cudaPclType_ID), cudaMemcpyDefault, outputStream));
      }
    }
    cudaErrChk(cudaEventRecord(eventOutputCopy, outputStream));
  }
}

/**
 * @brief Append conserved-quantity diagnostics to the rank-selected text file.
 * @param cycle Simulation cycle being written.
 */
void c_Solver::WriteConserved(int cycle) {
  if(col->getDiagnosticsOutputCycle() > 0 && cycle % col->getDiagnosticsOutputCycle() == 0)
  {
    // particlesHost[*]->getKe/getP/getTotalQ() iterate over the host SoA
    // u/v/w/q arrays, which are populated only by the async D->H copies in
    // outputCopyAsync() on outputStream and signalled by eventOutputCopy.
    // The mover launched just before WriteOutput() only inserts a stream-side
    // wait on eventOutputCopy (cudaStreamWaitEvent in cudaLauncherAsync), it
    // does NOT block the host. Without the synchronize below, diagnostics on
    // cycle i>first_cycle can read torn / partially-copied host buffers from
    // outputCopyAsync(i-1), producing nonsense conserved quantities.
    // eventOutputCopy is pre-recorded at init and first-cycle output is
    // explicitly primed when configured outputs need host particle data.
    cudaErrChk(cudaEventSynchronize(eventOutputCopy));
    Eenergy = EMf->getEenergy();
    Benergy = EMf->getBenergy();
    TOTenergy = 0.0;
    TOTmomentum = 0.0;
    double TOTcharge = 0.0;
    for (int is = 0; is < ns; is++) {
      Ke[is] = particlesHost[is]->getKe();
      BulkEnergy[is] = EMf->getBulkEnergy(is);
      TOTenergy += Ke[is];
      momentum[is] = particlesHost[is]->getP();
      TOTmomentum += momentum[is];
      Qtot[is] = particlesHost[is]->getTotalQ();
      TOTcharge += Qtot[is];
    }
    if (myrank == (nprocs-1)) {
      const int cw = 16; // column width
      ofstream my_file(cq.c_str(), fstream::app);
      my_file << std::scientific << std::setprecision(6);
      if(cycle == 0) {
        my_file << std::left
                << std::setw(8)  << "Cycle"
                << std::setw(cw) << "Total_Energy"
                << std::setw(cw) << "Momentum"
                << std::setw(cw) << "Eenergy"
                << std::setw(cw) << "Benergy"
                << std::setw(cw) << "Kenergy"
                << std::setw(cw) << "Total_Charge";
        for (int is = 0; is < ns; is++) my_file << std::setw(cw) << ("Ke[" + std::to_string(is) + "]");
        for (int is = 0; is < ns; is++) my_file << std::setw(cw) << ("BulkE[" + std::to_string(is) + "]");
        for (int is = 0; is < ns; is++) my_file << std::setw(cw) << ("Mom[" + std::to_string(is) + "]");
        for (int is = 0; is < ns; is++) my_file << std::setw(cw) << ("Q[" + std::to_string(is) + "]");
        my_file << endl;
      }
      my_file << std::left << std::setw(8) << cycle
              << std::setw(cw) << (Eenergy + Benergy + TOTenergy)
              << std::setw(cw) << TOTmomentum
              << std::setw(cw) << Eenergy
              << std::setw(cw) << Benergy
              << std::setw(cw) << TOTenergy
              << std::setw(cw) << TOTcharge;
      for (int is = 0; is < ns; is++) my_file << std::setw(cw) << Ke[is];
      for (int is = 0; is < ns; is++) my_file << std::setw(cw) << BulkEnergy[is];
      for (int is = 0; is < ns; is++) my_file << std::setw(cw) << momentum[is];
      for (int is = 0; is < ns; is++) my_file << std::setw(cw) << Qtot[is];
      my_file << endl;
      my_file.close();
    }
  }
}

/**
 * @brief Write one velocity-distribution snapshot for every species.
 *
 * Output cadence is controlled by the caller; this method only performs the
 * per-species histogram calculation and file append.
 * @param cycle Simulation cycle being written.
 */
void c_Solver::WriteVelocityDistribution(int cycle)
{
  for (int is = 0; is < ns; is++) {
    double maxVel = particlesHost[is]->getMaxVelocity();
    long long *VelocityDist = particlesHost[is]->getVelocityDistribution(nDistributionBins, maxVel);
    if (myrank == 0) {
      ofstream my_file(ds.c_str(), fstream::app);
      my_file << cycle << "\t" << is << "\t" << maxVel;
      for (int i = 0; i < nDistributionBins; i++)
        my_file << "\t" << VelocityDist[i];
      my_file << endl;
      my_file.close();
    }
    delete [] VelocityDist;
  }
}

/**
 * @brief Write field and moment traces at a regular grid of virtual satellite points.
 *
 * The trace file samples electromagnetic fields, selected current sums, and
 * charge densities at `nsat^3` probe locations in the local domain.
 */
void c_Solver::WriteVirtualSatelliteTraces()
{
  if(ns <= 2) return;
  assert_eq(ns,4);

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
        my_file << EMf->getBx(index1, index2, index3) << "\t" << EMf->getBy(index1, index2, index3) << "\t" << EMf->getBz(index1, index2, index3) << "\t";
        my_file << EMf->getEx(index1, index2, index3) << "\t" << EMf->getEy(index1, index2, index3) << "\t" << EMf->getEz(index1, index2, index3) << "\t";
        my_file << EMf->getJxs(index1, index2, index3, 0) + EMf->getJxs(index1, index2, index3, 2) << "\t" << EMf->getJys(index1, index2, index3, 0) + EMf->getJys(index1, index2, index3, 2) << "\t" << EMf->getJzs(index1, index2, index3, 0) + EMf->getJzs(index1, index2, index3, 2) << "\t";
        my_file << EMf->getJxs(index1, index2, index3, 1) + EMf->getJxs(index1, index2, index3, 3) << "\t" << EMf->getJys(index1, index2, index3, 1) + EMf->getJys(index1, index2, index3, 3) << "\t" << EMf->getJzs(index1, index2, index3, 1) + EMf->getJzs(index1, index2, index3, 3) << "\t";
        my_file << EMf->getRHOns(index1, index2, index3, 0) + EMf->getRHOns(index1, index2, index3, 2) << "\t";
        my_file << EMf->getRHOns(index1, index2, index3, 1) + EMf->getRHOns(index1, index2, index3, 3) << "\t";
      }}}
  my_file << endl;
  my_file.close();
}

/**
 * @brief Finalize output backends, optionally write the last restart, and free CUDA state.
 */
void c_Solver::Finalize() {

  pclNumCSV.close();

  if (col->getCallFinalize() && Parameters::get_doWriteOutput() && col->getRestartOutputCycle() > 0)
  {
    // Sync EM fields from device to host before final restart write
#ifdef GPU_SOLVER
    EMf->gpuSolverSyncD2H(EMf->gpuSolverStream());
#else
    cudaErrChk(cudaDeviceSynchronize());
#endif

    outputCopyAsync(-1);
    cudaErrChk(cudaEventSynchronize(eventOutputCopy));
    // Final restart data is written after the last loop iteration has
    // completed, so the resume label is the next loop cycle.
    prepareActiveRestartParticleCellMetadata();
    ioManager->writeRestart(col->getNcycles() + first_cycle);
  }

  ioManager->finalize();

  deInitCUDA();

  // Stop profiling after all runtime work has completed.
  my_clock->stopTiming();
}

/**
 * @brief Perform the legacy host-side particle sort for all species.
 *
 * This path is separate from the GPU cell sorter and only touches the host
 * ParticleSoAHost containers.
 */
void c_Solver::sortParticles() {

  for(int species_idx=0; species_idx<ns; species_idx++)
    particlesHost[species_idx]->sort_particles_serial();

}

/**
 * @brief Perform a full GPU cell sort for every species.
 *
 * This is used by the analysis path when cell-sorted order is required
 * independently of the mover's periodic sorting cadence.
 */
void c_Solver::sortAllSpecies() {
  for (int i = 0; i < ns; i++) {
    // Ensure no kernels are in flight on this species' stream
    cudaErrChk(cudaStreamSynchronize(streams[i]));
    // Explicit edge: also wait for the previous cycle's output D->H drain
    // before sort kernels mutate the SoA. The host sync above only waits for
    // streams[i]; outputStream is independent.
    cudaErrChk(cudaStreamWaitEvent(streams[i], eventOutputCopy, 0));

    // Lazy-init: cellSorters[i] is only constructed in c_Solver::Init() when
    // sortingCycle_ > 0. Analysis cycles can however call sortAllSpecies()
    // even with periodic sorting disabled (sortingCycle_ == 0), in which
    // case the sorter buffers are still null. Initialize on first use here
    // to avoid undefined behavior in prepareBuffers/enqueueSortAsync.
    if (!cellSorters[i].initialized) {
      cellSorters[i].init(*grid3DCUDAHostPtr,
                          pclsArrayHostPtr[i]->getCapacity(),
                          streams[i]);
    }

    const uint32_t nop = pclsArrayHostPtr[i]->getNOP();
    cellSorters[i].prepareBuffers(pclsArrayHostPtr[i], streams[i]);
    cellSorters[i].enqueueSortAsync(pclsArrayHostPtr[i], grid3DCUDACUDAPtr,
                                     nop, streams[i]);
    cellSorters[i].finishSort(streams[i]);

    // Re-sync host struct to device (SoA pointers were swapped on host)
    cudaErrChk(cudaMemcpyAsync(pclsArrayCUDAPtr[i], pclsArrayHostPtr[i],
                                sizeof(particleArrayCUDA), cudaMemcpyDefault, streams[i]));
  }

  // Wait until every species has completed its full sort pipeline.
  for (int i = 0; i < ns; i++) {
    cudaErrChk(cudaStreamSynchronize(streams[i]));
  }

  if constexpr (DAConfig::MACROCELL_SPECTRA_ENABLE) {
    if (col->getVelocitySpectra()) {
    bool anyMacrocellSpecies = false;
    for (int s = 0; s < ns && !anyMacrocellSpecies; ++s) {
      anyMacrocellSpecies = col->getVelocitySpectraSpecies(s);
    }

    if (col->getMacrocellNx() > 0 &&
        col->getMacrocellNy() > 0 &&
        col->getMacrocellNz() > 0 &&
        anyMacrocellSpecies) {
      // Data analysis starts immediately after sortAllSpecies(). Refresh and
      // drain the packed field buffer here so macrocell spectra sees the current
      // B field, including on cycle 0 before the mover path has packed it once.
      refreshFieldForPclsDeviceBuffer(true);
    }
    } // getVelocitySpectra()
  }
}

/**
 * @brief Pad host particle capacities for regular and test-particle species.
 */
void c_Solver::pad_particle_capacities()
{
  for (int i = 0; i < ns; i++)
    particlesHost[i]->padCapacities();

  for (int i = 0; i < nstestpart; i++)
    testpart[i]->padCapacities();
}


/**
 * @brief Return the first cycle after the current simulation window.
 */
int c_Solver::LastCycle() {
    return (col->getNcycles() + first_cycle);
}

/**
 * @brief Inject photoionized exosphere particles into host particle buffers.
 *
 * For each planetary species (index >= numSolarWindSpecies), this method samples
 * new macro-particles from the Chamberlain neutral density profile via
 * ExosphereIonization, then appends them to the AoS comm buffer particlesCommInj[i].
 * The particles will be copied to the GPU by the subsequent H2D transfer.
 *
 * Memory-aware injection: before sampling, the method queries GPU free memory
 * and computes a per-species particle budget to prevent out-of-memory conditions.
 * The current implementation applies a fixed low-memory heuristic and keeps a
 * portion of the currently free GPU memory in reserve for field arrays, moments,
 * and other runtime allocations.
 *
 * Task-based parallelism: each planetary species is submitted as an independent
 * task to the thread pool (same pool used by the mover). Each task uses its own
 * RNG, particle buffer, and particlesCommInj[i] — no cross-species contention. All futures
 * are collected before returning, so particlesCommInj[i].getCommNOP() is finalized for the
 * subsequent H2D copy loop.
 */
void c_Solver::injectExosphereParticles()
{
  if (exosphereIonization == nullptr) return;

  // ======= Compute a memory-aware per-species injection budget =======
  // Only activate a budget when GPU memory is actually scarce or a hard cap
  // is configured.  When memory is plentiful, maxParticlesPerSpecies stays 0
  // (unlimited), so sampleIonizedParticles follows the original zero-overhead
  // code path (hasBudget == false, no per-cell budget check).
  const int numPlanetSpecies = ns - numSolarWindSpecies;
  int maxParticlesPerSpecies = 0;  // 0 = unlimited (fast path)

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
      const size_t bytesPerParticle = static_cast<size_t>(sizeof(SpeciesParticle) * 1.5);
      const int gpuBudgetPerSpecies = (numPlanetSpecies > 0)
          ? static_cast<int>(usableBytes / bytesPerParticle) / numPlanetSpecies
          : static_cast<int>(usableBytes / bytesPerParticle);

      // Take the tighter of config cap and GPU budget
      if (maxParticlesPerSpecies > 0)
        maxParticlesPerSpecies = std::min(maxParticlesPerSpecies, gpuBudgetPerSpecies);
      else
        maxParticlesPerSpecies = gpuBudgetPerSpecies;
    }
  }

  // ======= Enqueue one sampling task per planetary species =======
  // Each task: (1) samples particles via thread-safe RNG, (2) appends to particlesCommInj[i] AoS comm buffer.
  // No shared mutable state between tasks — safe for concurrent execution.
  // numSolarWindSpecies, numPlanetarySpecies, and exosphereTaskFutures are persistent
  // class members initialized once in Init() — no per-call recomputation or allocation.
  exosphereTaskFutures.clear();

  for (int i = numSolarWindSpecies; i < ns; i++) {
    exosphereTaskFutures.push_back(
      threadPoolPtr->enqueue([this, i, maxParticlesPerSpecies]() {
        // sampleIonizedParticles is thread-safe for different species indices:
        // each species uses its own std::mt19937_64 RNG, particle buffer, and
        // charge accumulator (no global rand() or shared mutable state).
        const std::vector<SpeciesParticle>& exosphereParticles =
            exosphereIonization->sampleIonizedParticles(i, maxParticlesPerSpecies);

        if (exosphereParticles.empty()) return;

        // Append exosphere particles to AoS comm buffer (after MPI-incoming + repopulated)
        // Each task operates on its own particlesCommInj[i] — no cross-species contention.
        particlesCommInj[i]->appendFromAoS(exosphereParticles.data(),
                                          static_cast<int>(exosphereParticles.size()));
      })
    );
  }

  // Wait for every species task before the subsequent H2D copy loop runs.
  for (auto& future : exosphereTaskFutures) {
    future.get();
  }
}
