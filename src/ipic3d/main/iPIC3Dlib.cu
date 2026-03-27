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
#include "ompdefs.h"
#include "VCtopology3D.h"
#include "Collective.h"
#include "Grid3DCU.h"
#include "EMfields3D.h"
#include "ParticleCommInjection.h"
#include "Timing.h"
#include "ParallelIO.h"
#include "outputPrepare.h"
#include "IOManager.h"
//
#ifndef NO_HDF5
#include "WriteOutputParallel.h"
#include "OutputWrapperFPP.h"
#endif

#include <iostream>
#include <fstream>
#include <sstream>
#include <chrono>

// Set to 1 to enable per-phase timing printfs (launcher, await, MPI, planet, exchange TOTAL)
#define ENABLE_SOA_TIMING 0

#include "Moments.h" // for debugging

#include "ExosphereIonization.h"
#include <cstring>  // std::memcpy

#include "cudaTypeDef.cuh"
#include "momentKernel.cuh"
#include "particleArrayCUDA.cuh"
#include "moverKernel.cuh"
#include "particleExchange.cuh"
#include "dataAnalysis.cuh"
#include "thread"
#include "future"
#include "particleControlKernel.cuh"


#ifdef USE_CATALYST
#include "Adaptor.h"
#endif

using namespace iPic3D;
//MPIdata* iPic3D::c_Solver::mpi=0;



c_Solver::~c_Solver()
{
  delete col; // configuration parameters ("collectiveIO")
  delete vct; // process topology
  delete grid; // grid
  delete EMf; // field
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
  delete [] momentum;
  delete [] Qtot;
  delete [] Qremoved;
  delete exosphereIonization;
  delete my_clock;
}

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
  first_cycle = col->getLast_cycle() + 1; // get the last cycle from the restart
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
    //check and create the output directory, only if it is not a restart run
    if(restart_status == 0){checkOutputFolder(SaveDirName); if(RestartDirName != SaveDirName)checkOutputFolder(RestartDirName); }
    
    MPIdata::instance().Print();
    vct->Print();
    col->Print();
    col->save();
  }
  // Create the local grid
  grid = new Grid3DCU(col, vct);  // Create the local grid
  EMf = new EMfields3D(col, grid, vct);  // Create Electromagnetic Fields Object

  if      (col->getCase()=="GEMnoPert") 		EMf->initGEMnoPert();
  else if (col->getCase()=="ForceFree") 		EMf->initForceFree();
  else if (col->getCase()=="GEM")       		EMf->initGEM();
  else if (col->getCase()=="GEMDoubleHarris")  	        EMf->initGEMDoubleHarris();
#ifdef BATSRUS
  else if (col->getCase()=="BATSRUS")   		EMf->initBATSRUS();
#endif
  else if (col->getCase()=="Dipole")    		EMf->initDipole();
  else if (col->getCase()=="Dipole2D")  		EMf->initDipole2D();
  else if (col->getCase()=="NullPoints")             	EMf->initNullPoints();
  else if (col->getCase()=="TaylorGreen")               EMf->initTaylorGreen();
  else if (col->getCase()=="HumpPert")                  EMf->initHumpPerturbation();
  else if (col->getCase()=="RandomCase") {
    EMf->initRandomField();
    if (myrank==0) {
      cout << "Case is " << col->getCase() <<"\n";
      cout <<"total # of particle per cell is " << col->getNpcel(0) << "\n";
    }
  }
  else {
    if (myrank==0) {
      cout << " =========================================================== " << endl;
      cout << " WARNING: The case '" << col->getCase() << "' was not recognized. " << endl;
      cout << "          Runing simulation with the default initialization. " << endl;
      cout << " =========================================================== " << endl;
    }
    EMf->init();
  }

  // ===== Allocate particlesHost[] — lightweight SoA host mirror (no communicator) =====
  particlesHost = new ParticleSoAHost*[ns];
  for (int i = 0; i < ns; i++)
  {
    particlesHost[i] = new ParticleSoAHost(i, col, vct, grid);
    const auto totalPcl = col->getNpcel(i) * grid->getNXN() * grid->getNYN() * grid->getNZN();

    if (col->getRestart_status() == 0) {
      particlesHost[i]->reserveSpace(totalPcl);
      particlesHost[i]->clearParticles();
    } else { // restart
      particlesHost[i]->restartLoad();
    }
  }

  // Initial condition for PARTICLES (skipped when restarting)
  if (restart_status == 0) {
    for (int i = 0; i < ns; i++)
    {
      if      (col->getCase()=="ForceFree")        particlesHost[i]->force_free(EMf);
#ifdef BATSRUS
      else if (col->getCase()=="BATSRUS")          eprintf("BATSRUS not supported on ParticleSoAHost");
#endif
      else if (col->getCase()=="NullPoints")       particlesHost[i]->maxwellianNullPoints(EMf);
      else if (col->getCase()=="TaylorGreen")      particlesHost[i]->maxwellianNullPoints(EMf);
      else if (col->getCase()=="GEMDoubleHarris")  particlesHost[i]->maxwellianDoubleHarris(EMf);
      else if (col->getCase()=="HumpPert")         particlesHost[i]->maxwellianHumpPerturbation(EMf);
      else                                         particlesHost[i]->maxwellian(EMf);
      particlesHost[i]->reserve_remaining_particle_IDs();
    }
  }

  //allocate test particles if any
  nstestpart = col->getNsTestPart();

  // ===== Allocate particlesCommInj[] — MPI exchange + injection engine (AoS comm buffer) =====
  particlesCommInj = new ParticleCommInjection*[ns];
  for (int i = 0; i < ns; i++)
  {
    particlesCommInj[i] = new ParticleCommInjection(*particlesHost[i]);
    const auto totalPcl = col->getNpcel(i) * grid->getNXN() * grid->getNYN() * grid->getNZN();
    particlesCommInj[i]->reserveCommBuffer(static_cast<int>(totalPcl * 0.1));
  }

  if(nstestpart>0){
    testpart = new ParticleSoAHost*[nstestpart];
    for (int i = 0; i < nstestpart; i++)
    {
      testpart[i] = new ParticleSoAHost(i+ns,col,vct,grid);//species id for test particles is increased by ns
      testpart[i]->pitch_angle_energy(EMf);
    }
  }

  // ---- Initialise modular I/O manager ----
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

  // ── Exosphere ionization source ──
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

  my_clock = new Timing(myrank);

  return 0;
}

/**
 * @brief CUDA initilaize 
 */
int c_Solver::initCUDA(){

  // Set device for this MPI process
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
  
	// init the streams according to the species
  streams = new cudaStream_t[ns*2]; stayedParticle = new int[ns]; exitingResults = new std::future<int>[ns];
  for(int i=0; i<ns; i++){ cudaErrChk(cudaStreamCreate(streams+i)); cudaErrChk(cudaStreamCreate(streams+i+ns)); stayedParticle[i] = 0; }
  cudaErrChk(cudaStreamCreate(&planetStream));
	{ 
    // init arrays on device, pointers are device pointer, copied
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
      // the constructor will copy particles from host to device
      pclsArrayHostPtr[i] = newHostPinnedObject<particleArrayCUDA>(particlesHost[i], 1.4, streams[i]); // use the oputputPart as the initial pcls
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
      
      exitingArrayHostPtr[i] = newHostPinnedObject<exitingArray>(0.1 * pclsArrayHostPtr[i]->getNOP());
      exitingArrayCUDAPtr[i] = exitingArrayHostPtr[i]->copyToDevice();
      fillerBufferArrayHostPtr[i] = newHostPinnedObject<fillerBuffer>(0.1 * pclsArrayHostPtr[i]->getNOP());
      fillerBufferArrayCUDAPtr[i] = fillerBufferArrayHostPtr[i]->copyToDevice();

      // AoS staging buffer for incoming H→D particle transfers (MPI + repopulated + exosphere).
      // Sized at 10% of initial NOP — will be expanded dynamically if needed.
      incomingStagingHostPtr[i] = newHostPinnedObject<arrayCUDA<SpeciesParticle>>(static_cast<uint32_t>(0.1 * pclsArrayHostPtr[i]->getNOP()));
      incomingStagingCUDAPtr[i] = incomingStagingHostPtr[i]->copyToDevice();

    }
  }

  // one grid for all species
  grid3DCUDAHostPtr = newHostPinnedObject<grid3DCUDA>(grid);
  grid3DCUDACUDAPtr = copyToDevice(grid3DCUDAHostPtr, 0);


  // kernelParams — scalar species parameters come from particlesHost (ParticleSoAHost)
  moverParamHostPtr = new moverParameter*[ns];
  moverParamCUDAPtr = new moverParameter*[ns];
  for(int i=0; i<ns; i++){
    moverParamHostPtr[i] = newHostPinnedObject<moverParameter>(particlesHost[i], pclsArrayCUDAPtr[i], departureArrayCUDAPtr[i], hashedSumArrayCUDAPtr[i]);

    // init the moverParam for OpenBC, repopulateInjection, sphere
    particlesHost[i]->openbc_particles_outflowInfo(&moverParamHostPtr[i]->doOpenBC, moverParamHostPtr[i]->applyOpenBC, moverParamHostPtr[i]->deleteBoundary, moverParamHostPtr[i]->openBoundary);
    moverParamHostPtr[i]->appendCountAtomic = 0;

    // GPU-side EXIT BC: particles exiting via an EXIT face are marked DELETE
    // on the GPU to avoid sending them through MPI exchange at all.
    // Only applies on boundary ranks (where the neighbor is MPI_PROC_NULL).
    particlesHost[i]->fillExitBCFlags(moverParamHostPtr[i]->isExitBC);

    if(col->getRHOinject(i)>0.0)
    particlesHost[i]->repopulate_particlesInfo(&moverParamHostPtr[i]->doRepopulateInjection, moverParamHostPtr[i]->doRepopulateInjectionSide, moverParamHostPtr[i]->repopulateBoundary);
    else moverParamHostPtr[i]->doRepopulateInjection = false;

    if (col->getCase()=="Dipole") {
      moverParamHostPtr[i]->doSphere = 1;
      moverParamHostPtr[i]->sphereOrigin[0] = col->getx_center_planet();
      moverParamHostPtr[i]->sphereOrigin[1] = col->gety_center_planet();
      moverParamHostPtr[i]->sphereOrigin[2] = col->getz_center_planet();
      moverParamHostPtr[i]->sphereRadius = col->getL_square();
    } else if (col->getCase()=="Dipole2D") {
      moverParamHostPtr[i]->doSphere = 2;
      moverParamHostPtr[i]->sphereOrigin[0] = col->getx_center_planet();
      moverParamHostPtr[i]->sphereOrigin[1] = 0.0;
      moverParamHostPtr[i]->sphereOrigin[2] = col->getz_center_planet();
      moverParamHostPtr[i]->sphereRadius = col->getL_square();
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



  // simple device buffer, allocate one dimension array on device memory
  auto gridSize = grid->getNXN() * grid->getNYN() * grid->getNZN();
  momentsCUDAPtr = new cudaTypeArray1<cudaMomentType>[ns];
  //for(int i=0; i<ns; i++)cudaMallocAsync(&(momentsCUDAPtr[i]), gridSize*10*sizeof(cudaMomentType), streams[i]);
  for(int i=0; i<ns; i++)cudaMalloc(&(momentsCUDAPtr[i]), gridSize*10*sizeof(cudaMomentType));

  { // register the 10 densities to host pinned memory
    for(int i=0; i<ns; i++)
      registerMomentsPinnedMemory(i);
  }

  // cudaMallocAsync(&fieldForPclCUDAPtr, gridSize*8*sizeof(cudaCommonType), 0);

  const int fieldSize = grid->getNZN() * (grid->getNYN() - 1) * (grid->getNXN() - 1);

  //cudaMallocAsync(&fieldForPclCUDAPtr, fieldSize * 24 * sizeof(cudaFieldType), 0);
  cudaMalloc(&fieldForPclCUDAPtr, fieldSize * 24 * sizeof(cudaFieldType));

  cudaErrChk(cudaHostAlloc((void**)&fieldForPclHostPtr, fieldSize * 24 * sizeof(cudaFieldType), 0));

  threadPoolPtr = new ThreadPool(ns);
  cudaErrChk(cudaEventCreateWithFlags(&event0, cudaEventDisableTiming));
  cudaErrChk(cudaEventCreateWithFlags(&eventOutputCopy, cudaEventDisableTiming|cudaEventBlockingSync));

  // merging
  toBeMerged = new int[2 * ns];
  for(int i=0;i<2*ns;i++){
    toBeMerged[i] = 0;
  }
  //memset(toBeMerged, 0, 2 * ns * sizeof(int));

  cudaErrChk(cudaHostAlloc(&cellCountHostPtr, sizeof(int) * grid->getNXC() * grid->getNYC() * grid->getNZC(), cudaHostAllocDefault));
  cudaErrChk(cudaHostAlloc(&cellOffsetHostPtr, sizeof(int) * grid->getNXC() * grid->getNYC() * grid->getNZC(), cudaHostAllocDefault));

  cudaErrChk(cudaMalloc(&cellCountCUDAPtr, sizeof(int) * grid->getNXC() * grid->getNYC() * grid->getNZC()));
  cudaErrChk(cudaMalloc(&cellOffsetCUDAPtr, sizeof(int) * grid->getNXC() * grid->getNYC() * grid->getNZC()));

  // ── Cell sorter (counting sort) per species ──
  cellSorters = new CellSorter[ns];
  for (int i = 0; i < ns; i++) {
    cellSorters[i].init(*grid3DCUDAHostPtr,
                        pclsArrayHostPtr[i]->getCapacity(),
                        streams[i]);
  }

  dataAnalysis::dataAnalysisPipeline::createOutputDirectory(myrank, ns, vct);

  // ── Planet quasi-neutral BC allocations ──
  {
    const bool doPlanet = (col->getCase() == "Dipole" || col->getCase() == "Dipole2D");

    planetArrayHostPtr = new planetArray*[ns];
    planetArrayCUDAPtr = new planetArray*[ns];
    planetPclCount     = new int[ns];

    for (int i = 0; i < ns; i++) {
      if (doPlanet) {
        planetArrayHostPtr[i] = newHostPinnedObject<planetArray>((uint32_t)(0.05 * pclsArrayHostPtr[i]->getNOP()));
        planetArrayCUDAPtr[i] = planetArrayHostPtr[i]->copyToDevice();
      } else {
        planetArrayHostPtr[i] = nullptr;
        planetArrayCUDAPtr[i] = nullptr;
      }
      planetPclCount[i] = 0;
    }

    // Build electron species map
    planetElecSpeciesCount = 0;
    for (int i = 0; i < ns; i++)
      if (col->getQOM(i) < 0) planetElecSpeciesCount++;

    planetElecSpeciesMap = new int[planetElecSpeciesCount];
    {
      int idx = 0;
      for (int i = 0; i < ns; i++)
        if (col->getQOM(i) < 0) planetElecSpeciesMap[idx++] = i;
    }

    // Cross-species device buffers
    planetBufCapacity = doPlanet ? 1024 : 0;
    if (doPlanet) {
      cudaErrChk(cudaMalloc(&planetEnergyBuf,    planetBufCapacity * sizeof(cudaParticleType)));
      cudaErrChk(cudaMalloc(&planetGlobalIdxBuf,  planetBufCapacity * sizeof(uint32_t)));
      cudaErrChk(cudaMalloc(&planetIonChargeDevice, sizeof(cudaParticleType)));
      cudaErrChk(cudaMalloc(&planetCutoffDevice,    sizeof(int)));

      // Device array of device pointers to per-electron-species planetArrays
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

    // Persistent host/device buffers for processPlanetParticles
    const int elecCount = planetElecSpeciesCount > 0 ? planetElecSpeciesCount : 1;
    cudaErrChk(cudaHostAlloc(&planetElecOffsets, elecCount * sizeof(int), cudaHostAllocDefault));
    cudaErrChk(cudaHostAlloc(&planetTmpPtrs, elecCount * sizeof(planetArray*), cudaHostAllocDefault));
    cudaErrChk(cudaHostAlloc(&planetSurvivorCount, elecCount * sizeof(int), cudaHostAllocDefault));

    if (doPlanet) {
      cudaErrChk(cudaMalloc(&planetSurvivorCountDevice, elecCount * sizeof(int)));
      planetReflectedBufCapacity = 1024;
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


int c_Solver::deInitCUDA(){

  cudaEventDestroy(event0);
  cudaEventDestroy(eventOutputCopy);

  delete threadPoolPtr;

  deleteHostPinnedObject(grid3DCUDAHostPtr);
  cudaFree(grid3DCUDACUDAPtr);

  cudaFree(fieldForPclCUDAPtr);
  cudaFreeHost(fieldForPclHostPtr);

  // release device objects
  for(int i=0; i<ns; i++){

    // ==================  delete host object, deconstruct ==================

    deleteHostPinnedObject(pclsArrayHostPtr[i]);
    deleteHostPinnedObject(departureArrayHostPtr[i]);
    deleteHostPinnedObjectArray(hashedSumArrayHostPtr[i], departureArrayElementType::HASHED_SUM_NUM);
    deleteHostPinnedObject(exitingArrayHostPtr[i]);
    deleteHostPinnedObject(fillerBufferArrayHostPtr[i]);
    deleteHostPinnedObject(incomingStagingHostPtr[i]);

    deleteHostPinnedObject(moverParamHostPtr[i]);
    deleteHostPinnedObject(momentParamHostPtr[i]);


    // ==================  cudaFree device object mem ==================

    cudaFree(pclsArrayCUDAPtr[i]);
    cudaFree(departureArrayCUDAPtr[i]);
    cudaFree(hashedSumArrayCUDAPtr[i]);
    cudaFree(exitingArrayCUDAPtr[i]);
    cudaFree(fillerBufferArrayCUDAPtr[i]);
    cudaFree(incomingStagingCUDAPtr[i]);

    cudaFree(moverParamCUDAPtr[i]);
    cudaFree(momentParamCUDAPtr[i]);

    cudaFree(momentsCUDAPtr[i]);
    
  }


  // delete ptr arrays
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
  delete[] momentsCUDAPtr;
  delete[] toBeMerged;

  // ── Planet quasi-neutral BC cleanup ──
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

  // ── Cell sorters cleanup ──
  for (int i = 0; i < ns; i++) cellSorters[i].free();
  delete[] cellSorters;

  // delete streams
  for(int i=0; i<ns*2; i++)cudaStreamDestroy(streams[i]);
  cudaStreamDestroy(planetStream);
  delete[] streams;
  delete[] stayedParticle;
  delete[] exitingResults;

  { // unregister the pinned mem
    for (int i = 0; i < ns; i++)
      unregisterMomentsPinnedMemory(i);
  }

  return 0;
}


// ---------------------------------------------------------------------------
// CUDA helper: async-copy 10 moment arrays from device to host for one species
// ---------------------------------------------------------------------------
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

// ---------------------------------------------------------------------------
// CUDA helper: register 10 moment arrays as pinned memory for one species
// ---------------------------------------------------------------------------
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

// ---------------------------------------------------------------------------
// CUDA helper: unregister 10 moment arrays from pinned memory for one species
// ---------------------------------------------------------------------------
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


void c_Solver::CalculateMoments() {

  // timeTasks_set_main_task(TimeTasks::MOMENTS);

  // sum moments
  auto gridSize = grid->getNXN() * grid->getNYN() * grid->getNZN();
  for(int i=0; i<ns; i++){
    cudaErrChk(cudaMemsetAsync(momentsCUDAPtr[i], 0, gridSize*10*sizeof(cudaMomentType), streams[i]));  // set moments to 0
    // copy the particles to device---- already there...by initliazation or Mover
    // launch the moment kernel
    momentKernelNew<<<(pclsArrayHostPtr[i]->getNOP()/256 + 1), 256, 0, streams[i] >>>(momentParamCUDAPtr[i], grid3DCUDACUDAPtr, momentsCUDAPtr[i], 0);
    copyMomentsD2H(i, streams[i]);
  }

  // synchronize
  MomentsAwait();

}


//! MAXWELL SOLVER for Efield
void c_Solver::CalculateField(int cycle) {
  timeTasks_set_main_task(TimeTasks::FIELDS);

  // calculate the E field
  EMf->calculateE(cycle);
}



/*  -------------- */
/*!  Particle mover */
/*  -------------- */
int c_Solver::cudaLauncherAsync(const int species){
  cudaSetDevice(cudaDeviceOnNode); // a must on multi-device node
#if ENABLE_SOA_TIMING
  auto _tL0 = std::chrono::high_resolution_clock::now();
#endif

  cudaEvent_t event1, event2;
  cudaErrChk(cudaEventCreateWithFlags(&event1, cudaEventDisableTiming));
  cudaErrChk(cudaEventCreateWithFlags(&event2, cudaEventDisableTiming));

  
  // particle number control 
  // splitting
  //std::cout << "myrank: "<<MPIdata::get_rank() <<" pclsArrayHostPtr[species]->getInitialNOP(): " << pclsArrayHostPtr[species]->getInitialNOP() <<
  //          " pclsArrayHostPtr[species]->getNOP() " << pclsArrayHostPtr[species]->getNOP() << std::endl;
  constexpr bool PARTICLE_SPLITTING = false; // set to true to enable particle splitting
  if constexpr(PARTICLE_SPLITTING)
  {
    if(pclsArrayHostPtr[species]->getNOP() < 0.95 * pclsArrayHostPtr[species]->getInitialNOP()){
      const uint32_t deltaPcl = pclsArrayHostPtr[species]->getInitialNOP() - pclsArrayHostPtr[species]->getNOP();
      if(deltaPcl < pclsArrayHostPtr[species]->getNOP()){
        std::cout << "Particle splitting basic myrank: "<< MPIdata::get_rank() << " species " << species <<" number particles: " << pclsArrayHostPtr[species]->getNOP() <<
                  " delta: " << deltaPcl <<std::endl;
        particleSplittingKernel<false><<<getGridSize((int)deltaPcl, 256), 256, 0, streams[species]>>>(moverParamCUDAPtr[species], grid3DCUDACUDAPtr);
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
        particleSplittingKernel<true><<<getGridSize((int)pclsArrayHostPtr[species]->getNOP(), 256), 256, 0, streams[species]>>>(moverParamCUDAPtr[species], grid3DCUDACUDAPtr);
        pclsArrayHostPtr[species]->setNOE( (splittingTimes + 1) * pclsArrayHostPtr[species]->getNOP());
        cudaErrChk(cudaMemcpyAsync(pclsArrayCUDAPtr[species], pclsArrayHostPtr[species], sizeof(particleArrayCUDA), cudaMemcpyDefault, streams[species]));
        
        //std::cerr << "Particle control multiple time splitting not yet implemented "<<std::endl;
      } 
    }
  }
  
  // Mover
  // wait to field values copied to device
#if ENABLE_SOA_TIMING
  auto _tL1 = std::chrono::high_resolution_clock::now(); // after splitting, before mover launch
#endif
  cudaErrChk(cudaStreamWaitEvent(streams[species], event0, 0));
  if (col->getCase()=="Dipole" || col->getCase()=="Dipole2D")
    moverSubcyclesKernel<<<getGridSize((int)pclsArrayHostPtr[species]->getNOP(), 256), 256, 0, streams[species]>>>(moverParamCUDAPtr[species], fieldForPclCUDAPtr, grid3DCUDACUDAPtr);
  else
    moverKernel<<<getGridSize((int)pclsArrayHostPtr[species]->getNOP(), 256), 256, 0, streams[species]>>>(moverParamCUDAPtr[species], fieldForPclCUDAPtr, grid3DCUDACUDAPtr);
  
  cudaErrChk(cudaEventRecord(event1, streams[species]));
  // NOTE: momentKernelStayed REMOVED — moments are now computed after sorting
  // in MoverAwaitAndPclExchange via cellAwareMomentKernel.

  // Copy 8 hashedSums to host: 6 directions + delete + planet (XLOW..PLANET)
  cudaErrChk(cudaStreamWaitEvent(streams[species+ns], event1, 0));
  cudaErrChk(cudaMemcpyAsync(hashedSumArrayHostPtr[species], hashedSumArrayCUDAPtr[species], 
    (departureArrayElementType::PLANET_HASHEDSUM_INDEX + 1)*sizeof(hashedSum), cudaMemcpyDefault, streams[species+ns]));

  // Copy OpenBC appended particle number to host
  if (moverParamHostPtr[species]->doOpenBC) {
    cudaErrChk(cudaMemcpyAsync(&moverParamHostPtr[species]->appendCountAtomic, &moverParamCUDAPtr[species]->appendCountAtomic, 
                                sizeof(uint32_t), cudaMemcpyDefault, streams[species+ns]));
    cudaErrChk(cudaMemsetAsync(&moverParamCUDAPtr[species]->appendCountAtomic, 0, sizeof(uint32_t), streams[species+ns]));
    cudaErrChk(cudaStreamSynchronize(streams[species+ns]));

    const uint32_t newPclAfterOBC = pclsArrayHostPtr[species]->getNOP() + moverParamHostPtr[species]->appendCountAtomic;
    pclsArrayHostPtr[species]->setNOE(newPclAfterOBC);
    cudaErrChk(cudaMemcpyAsync(pclsArrayCUDAPtr[species], pclsArrayHostPtr[species], sizeof(particleArrayCUDA), cudaMemcpyDefault, streams[species+ns]));
  }


  // After Mover
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
    // expand the exiting AoS device buffer
    exitingArrayHostPtr[species]->expand(count_exiting * 1.5, streams[species+ns]);
    cudaErrChk(cudaMemcpyAsync(exitingArrayCUDAPtr[species], exitingArrayHostPtr[species], 
                                sizeof(exitingArray), cudaMemcpyDefault, streams[species+ns]));
  }

  if(hole > fillerBufferArrayHostPtr[species]->getSize()){
    // prepare the fillerBuffer
    fillerBufferArrayHostPtr[species]->expand(hole * 1.5, streams[species+ns]);
    cudaErrChk(cudaMemcpyAsync(fillerBufferArrayCUDAPtr[species], fillerBufferArrayHostPtr[species], 
                                sizeof(fillerBuffer), cudaMemcpyDefault, streams[species+ns]));
  }

  // Planet extraction — MUST run before exitingKernel, which overwrites
  // departureArray[].hashedId with HOLE hashes for front-region particles.
  // planetExtractionKernel needs the original PLANET hashedId to scatter correctly.
  if (count_removed_planet > 0) {
    if ((uint32_t)count_removed_planet > planetArrayHostPtr[species]->getSize()) {
      planetArrayHostPtr[species]->expand(count_removed_planet * 1.5, streams[species+ns]);
      cudaErrChk(cudaMemcpyAsync(planetArrayCUDAPtr[species], planetArrayHostPtr[species],
                                  sizeof(planetArray), cudaMemcpyDefault, streams[species+ns]));
    }
    planetExtractionKernel<<<getGridSize((int)pclsArrayHostPtr[species]->getNOP(), 256), 256, 0, streams[species+ns]>>>(
        pclsArrayCUDAPtr[species], departureArrayCUDAPtr[species],
        planetArrayCUDAPtr[species], hashedSumArrayCUDAPtr[species]);
  }

#if ENABLE_SOA_TIMING
  auto _tL4 = std::chrono::high_resolution_clock::now(); // before exitingKernel
#endif
  exitingKernel<<<getGridSize((int)pclsArrayHostPtr[species]->getNOP(), 256), 256, 0, streams[species+ns]>>>(pclsArrayCUDAPtr[species], 
                departureArrayCUDAPtr[species], exitingArrayCUDAPtr[species], hashedSumArrayCUDAPtr[species]);

  cudaErrChk(cudaEventRecord(event2, streams[species+ns]));

  // Prepare comm buffer and copy exiting particles D→H (single AoS memcpy)
  particlesCommInj[species]->clearCommBuffer();
  if (count_exiting > 0) {
    particlesCommInj[species]->prepareCommBufferForNOP(count_exiting);
    cudaErrChk(cudaMemcpyAsync(particlesCommInj[species]->getCommPclsDataMut(),
                                exitingArrayHostPtr[species]->getArray(),
                                count_exiting * sizeof(SpeciesParticle),
                                cudaMemcpyDefault, streams[species+ns]));
  }

  // Sorting, the first cycle, x might be 0
  cudaErrChk(cudaStreamWaitEvent(streams[species], event2, 0));
  if (hole > 0) 
  sortingKernel1<<<getGridSize(hole, 128), 128, 0, streams[species]>>>(pclsArrayCUDAPtr[species], departureArrayCUDAPtr[species], 
                                                          fillerBufferArrayCUDAPtr[species], hashedSumArrayCUDAPtr[species]+departureArrayElementType::FILLER_HASHEDSUM_INDEX, hole);
  sortingKernel2<<<getGridSize((int)(pclsArrayHostPtr[species]->getNOP()-hole), 256), 256, 0, streams[species]>>>(pclsArrayCUDAPtr[species], departureArrayCUDAPtr[species], 
                                                          fillerBufferArrayCUDAPtr[species], hashedSumArrayCUDAPtr[species]+departureArrayElementType::HOLE_HASHEDSUM_INDEX, pclsArrayHostPtr[species]->getNOP()-hole);

  cudaErrChk(cudaEventDestroy(event1));
  cudaErrChk(cudaEventDestroy(event2));
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

bool c_Solver::ParticlesMoverMomentAsync()
{
  // move all species of particles
  
  timeTasks_set_main_task(TimeTasks::PARTICLES);
  // Should change this to add background field
  //EMf->set_fieldForPcls();
  EMf->set_fieldForPclsToCenter(fieldForPclHostPtr);

  auto gridSize = grid->getNXN() * grid->getNYN() * grid->getNZN();
  //! copy fieldForPcls to device, for every species 
  cudaErrChk(cudaMemcpyAsync(fieldForPclCUDAPtr, fieldForPclHostPtr, (grid->getNZN() * (grid->getNYN() - 1) * (grid->getNXN() - 1)) * 24 * sizeof(cudaFieldType), cudaMemcpyDefault, streams[0]));
    // castingField<<<gridSize/256 + 1, 256, 0, streams[0]>>>(grid3DCUDACUDAPtr, fieldForPclCUDAPtr);
  cudaErrChk(cudaEventRecord(event0, streams[0]));

  for(int i=0; i<ns; i++){
    if (i != mergeIdx){
      exitingResults[i] = threadPoolPtr->enqueue(&c_Solver::cudaLauncherAsync, this, i);
      toBeMerged[2 * i + 1] +=1;
    }
  }

  if (mergeIdx >= 0 && mergeIdx < ns) 
  {
    const auto& i = mergeIdx;
    std::cout << " Particle merging myrank: "<< MPIdata::get_rank() << " species: " << i << std::endl;

    cudaErrChk(cudaStreamSynchronize(streams[i])); // wait for the D→H SoA copy
    // sort (SoA-native — no AoS conversion)
    particlesHost[i]->sort_particles_parallel(cellCountHostPtr, cellOffsetHostPtr);

    const int totalCells = grid->getNXC() * grid->getNYC() * grid->getNZC();
    cudaErrChk(cudaMemcpyAsync(cellCountCUDAPtr, cellCountHostPtr, totalCells*sizeof(int), cudaMemcpyDefault, streams[i]));
    cudaErrChk(cudaMemcpyAsync(cellOffsetCUDAPtr, cellOffsetHostPtr, totalCells*sizeof(int), cudaMemcpyDefault, streams[i]));

    // Direct host SoA → GPU SoA upload (no AoS intermediary)
    const uint32_t mergeNop = pclsArrayHostPtr[i]->getNOP();
    if (mergeNop > 0) {
      const size_t bytes = mergeNop * sizeof(double);
      cudaErrChk(cudaMemcpyAsync(pclsArrayHostPtr[i]->getU(), particlesHost[i]->getUall(), bytes, cudaMemcpyDefault, streams[i]));
      cudaErrChk(cudaMemcpyAsync(pclsArrayHostPtr[i]->getV(), particlesHost[i]->getVall(), bytes, cudaMemcpyDefault, streams[i]));
      cudaErrChk(cudaMemcpyAsync(pclsArrayHostPtr[i]->getW(), particlesHost[i]->getWall(), bytes, cudaMemcpyDefault, streams[i]));
      cudaErrChk(cudaMemcpyAsync(pclsArrayHostPtr[i]->getQ(), particlesHost[i]->getQall(), bytes, cudaMemcpyDefault, streams[i]));
      cudaErrChk(cudaMemcpyAsync(pclsArrayHostPtr[i]->getX(), particlesHost[i]->getXall(), bytes, cudaMemcpyDefault, streams[i]));
      cudaErrChk(cudaMemcpyAsync(pclsArrayHostPtr[i]->getY(), particlesHost[i]->getYall(), bytes, cudaMemcpyDefault, streams[i]));
      cudaErrChk(cudaMemcpyAsync(pclsArrayHostPtr[i]->getZ(), particlesHost[i]->getZall(), bytes, cudaMemcpyDefault, streams[i]));
      cudaErrChk(cudaMemcpyAsync(pclsArrayHostPtr[i]->getT(), particlesHost[i]->getParticleIDall(), bytes, cudaMemcpyDefault, streams[i]));
    }

    // merge
    mergingKernel<<<getGridSize(totalCells * WARP_SIZE, 256), 256, 0, streams[i]>>>(cellOffsetCUDAPtr, cellCountCUDAPtr, 
        grid3DCUDACUDAPtr, pclsArrayCUDAPtr[i], departureArrayCUDAPtr[i]);

    exitingResults[i] = threadPoolPtr->enqueue(&c_Solver::cudaLauncherAsync, this, i);

    toBeMerged[2 * i + 1] = 0;
    mergeIdx = -1; // merged
  }

  return (false);
}

// ═══════════════════════════════════════════════════════════════════════
//  Planet quasi-neutral BC: cross-species processing on planetStream
//  Single host sync at the end; all intermediate results stay on device.
// ═══════════════════════════════════════════════════════════════════════
void c_Solver::processPlanetParticles()
{
#if ENABLE_SOA_TIMING
  auto _tP0 = std::chrono::high_resolution_clock::now();
#endif
  // ── Step 1: Check if any planet particles exist ──
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
  // ── Step 2: Reduce ion charge on GPU (result stays on device) ──
  cudaErrChk(cudaMemsetAsync(planetIonChargeDevice, 0, sizeof(cudaParticleType), planetStream));
  for (int i = 0; i < ns; i++) {
    if (col->getQOM(i) <= 0 || planetPclCount[i] == 0) continue;
    const int blockSize = 256;
    const int gridSz = getGridSize(planetPclCount[i], blockSize);
    planetChargeReductionKernel<<<gridSz, blockSize, blockSize * sizeof(cudaParticleType), planetStream>>>(
        planetArrayCUDAPtr[i], planetPclCount[i], planetIonChargeDevice);
  }
  // No host sync — ionChargeDevice is read by chargeCutoffKernel via device pointer

  // ── Step 3: Expand cross-species buffers if needed ──
  // Round up to next power of 2 for bitonic sort
  int nPad = 1;
  while (nPad < totalElecPlanet) nPad <<= 1;

  if (nPad > planetBufCapacity) {
    if (planetEnergyBuf)    cudaFree(planetEnergyBuf);
    if (planetGlobalIdxBuf) cudaFree(planetGlobalIdxBuf);
    planetBufCapacity = nPad;
    cudaErrChk(cudaMalloc(&planetEnergyBuf,    planetBufCapacity * sizeof(cudaParticleType)));
    cudaErrChk(cudaMalloc(&planetGlobalIdxBuf,  planetBufCapacity * sizeof(uint32_t)));
  }

  // Expand reflected output buffer if needed (upper bound = totalElecPlanet)
  if (totalElecPlanet > planetReflectedBufCapacity) {
    if (planetReflectedBuf) cudaFree(planetReflectedBuf);
    planetReflectedBufCapacity = totalElecPlanet * 2;
    cudaErrChk(cudaMalloc(&planetReflectedBuf, planetReflectedBufCapacity * sizeof(SpeciesParticle)));
  }

  // ── Step 4: Compute energy per electron planet particle ──
  int offset = 0;
  for (int e = 0; e < planetElecSpeciesCount; e++) {
    int specIdx = planetElecSpeciesMap[e];
    planetElecOffsets[e] = offset;
    if (planetPclCount[specIdx] > 0) {
      planetEnergyKernel<<<getGridSize(planetPclCount[specIdx], 256), 256, 0, planetStream>>>(
          planetArrayCUDAPtr[specIdx], planetPclCount[specIdx],
          (cudaParticleType)col->getQOM(specIdx),
          planetEnergyBuf, planetGlobalIdxBuf,
          offset);
    }
    offset += planetPclCount[specIdx];
  }

  // Copy offsets to device (used by chargeCutoffKernel and planetReflectCompactKernel)
  if (planetElecSpeciesCount > 0) {
    cudaErrChk(cudaMemcpyAsync(planetElecOffsetsDevice, planetElecOffsets,
                                planetElecSpeciesCount * sizeof(int), cudaMemcpyHostToDevice, planetStream));
  }

  // Update device array of planetArray device pointers (in case pointers changed due to expand)
  {
    for (int e = 0; e < planetElecSpeciesCount; e++)
      planetTmpPtrs[e] = planetArrayCUDAPtr[planetElecSpeciesMap[e]];
    cudaErrChk(cudaMemcpyAsync(planetArrayCUDAPtrDevice, planetTmpPtrs,
                                planetElecSpeciesCount * sizeof(planetArray*), cudaMemcpyHostToDevice, planetStream));
  }

  // ── Step 5: Bitonic sort (descending by energy) ──
  if (nPad > totalElecPlanet) {
    bitonicPadKernel<<<getGridSize(nPad - totalElecPlanet, 256), 256, 0, planetStream>>>(
        planetEnergyBuf, planetGlobalIdxBuf, totalElecPlanet, nPad);
  }
  for (int k = 2; k <= nPad; k <<= 1) {
    for (int j = k >> 1; j > 0; j >>= 1) {
      bitonicSortStepKernel<<<getGridSize(nPad, 256), 256, 0, planetStream>>>(
          planetEnergyBuf, planetGlobalIdxBuf, j, k, nPad);
    }
  }

  // ── Step 6: Find cutoff (result stays on device) ──
  cudaErrChk(cudaMemsetAsync(planetCutoffDevice, 0, sizeof(int), planetStream));
  chargeCutoffKernel<<<1, 1, 0, planetStream>>>(
      planetArrayCUDAPtrDevice, planetElecSpeciesCount, planetElecOffsetsDevice,
      planetGlobalIdxBuf, totalElecPlanet,
      planetIonChargeDevice, planetCutoffDevice);
  // No host sync — cutoffDevice is read by planetReflectCompactKernel via device pointer

  // ── Step 7: Fused reflect + compact (all electron species in one kernel launch) ──
  const int doSphere = moverParamHostPtr[0]->doSphere;
  const cudaCommonType originX = moverParamHostPtr[0]->sphereOrigin[0];
  const cudaCommonType originY = moverParamHostPtr[0]->sphereOrigin[1];
  const cudaCommonType originZ = moverParamHostPtr[0]->sphereOrigin[2];
  const cudaCommonType radius  = moverParamHostPtr[0]->sphereRadius;

  // Zero per-species atomic counters
  cudaErrChk(cudaMemsetAsync(planetSurvivorCountDevice, 0,
                              planetElecSpeciesCount * sizeof(int), planetStream));

  // Launch: totalElecPlanet threads; each checks if it is a survivor via device cutoff
  const int reflectionType = col->getPlanetReflectionType();
  if (reflectionType == 1) {
    // Diffuse (isotropic) scattering — matches legacy rotateAndCountParticlesInsideSphere
    planetDiffuseCompactKernel<<<getGridSize(totalElecPlanet, 256), 256, 0, planetStream>>>(
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
    // Specular (mirror) reflection — default
    planetReflectCompactKernel<<<getGridSize(totalElecPlanet, 256), 256, 0, planetStream>>>(
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

  // ── Step 8: D2H survivor counts (one small transfer) ──
  cudaErrChk(cudaMemcpyAsync(planetSurvivorCount, planetSurvivorCountDevice,
                              planetElecSpeciesCount * sizeof(int), cudaMemcpyDeviceToHost, planetStream));
  cudaErrChk(cudaStreamSynchronize(planetStream)); // ONLY sync: need counts on host for resize + D2H
#if ENABLE_SOA_TIMING
  auto _tP2 = std::chrono::high_resolution_clock::now();
#endif

  // ── Step 9: D2H reflected particles into AoS comm buffer (appended after exiting) ──
  for (int e = 0; e < planetElecSpeciesCount; e++) {
    if (planetSurvivorCount[e] == 0) continue;
    int specIdx = planetElecSpeciesMap[e];
    const int nRefl = planetSurvivorCount[e];
    const int commOffset = particlesCommInj[specIdx]->getCommNOP();
    // Grow comm buffer to accommodate reflected particles
    particlesCommInj[specIdx]->prepareCommBufferForNOP(commOffset + nRefl);
    // Single AoS D→H from reflected device buffer into comm buffer at offset
    cudaErrChk(cudaMemcpyAsync(
        particlesCommInj[specIdx]->getCommPclsDataMut() + commOffset,
        planetReflectedBuf + planetElecOffsets[e],
        nRefl * sizeof(SpeciesParticle),
        cudaMemcpyDeviceToHost, planetStream));
  }

  // Sync to ensure all D2H are complete before MPI exchange
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

bool c_Solver::MoverAwaitAndPclExchange()
{
#if ENABLE_SOA_TIMING
  auto _t0 = std::chrono::high_resolution_clock::now();
#endif

  // ═══════════════════════════════════════════════════════════════════════
  //  Phase 3A: Await mover+compaction futures
  // ═══════════════════════════════════════════════════════════════════════
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

  // ── Planet processing on planetStream (blocks internally, completes before returning) ──
  const bool doPlanet = (col->getCase() == "Dipole" || col->getCase() == "Dipole2D");
  if (doPlanet)
    processPlanetParticles();
#if ENABLE_SOA_TIMING
  auto _t2 = std::chrono::high_resolution_clock::now();
#endif

  // ═══════════════════════════════════════════════════════════════════════
  //  Phase 3B: Sync all streams → update NOP → prepare sort buffers
  //  SAFE: after this sync, no GPU kernels are in flight for any species.
  //  We can safely resize sort buffers (which may cudaMalloc/cudaFree).
  // ═══════════════════════════════════════════════════════════════════════
  for (int i = 0; i < ns; i++) {
    // Update host NOP to stayed count (compacted prefix only)
    pclsArrayHostPtr[i]->setNOE(stayedParticle[i]);
    // Sync stream to ensure mover+compaction kernels have completed
    cudaErrChk(cudaStreamSynchronize(streams[i]));
    // Resize sort buffers if needed (may cudaMalloc/cudaFree — safe: no kernels)
    cellSorters[i].prepareBuffers(pclsArrayHostPtr[i], streams[i]);
  }

  // ═══════════════════════════════════════════════════════════════════════
  //  Phase 3C: Enqueue sort stages 1-3 for all species (non-blocking)
  //  GPU processes histogram + prefix sum + sorted indices while CPU
  //  proceeds to MPI exchange below.
  // ═══════════════════════════════════════════════════════════════════════
  for (int i = 0; i < ns; i++) {
    cellSorters[i].enqueueSortAsync(pclsArrayHostPtr[i],
                                     grid3DCUDACUDAPtr,
                                     stayedParticle[i],  // sort only stayed prefix
                                     streams[i]);
  }
#if ENABLE_SOA_TIMING
  auto _t2b = std::chrono::high_resolution_clock::now();
#endif

  // ═══════════════════════════════════════════════════════════════════════
  //  Phase 3D: MPI exchange (CPU-side, overlaps with GPU sort stages 1-3)
  // ═══════════════════════════════════════════════════════════════════════
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
#endif
    if (moverParamHostPtr[i]->doRepopulateInjection) {
      particlesCommInj[i]->repopulateParticlesOnlyInjection();
    }
#if ENABLE_SOA_TIMING
    auto _mpi3 = std::chrono::high_resolution_clock::now();
    if (MPIdata::get_rank() == 0)
      printf("  [SoA MPI s%d: separate=%.2f recomm=%.2f inject=%.2f total=%.2f ms  commNOP=%d]\n", i,
             std::chrono::duration<double, std::milli>(_mpi1 - _mpi0).count(),
             std::chrono::duration<double, std::milli>(_mpi2 - _mpi1).count(),
             std::chrono::duration<double, std::milli>(_mpi3 - _mpi2).count(),
             std::chrono::duration<double, std::milli>(_mpi3 - _mpi0).count(),
             particlesCommInj[i]->getCommNOP());
#endif
  }
#if ENABLE_SOA_TIMING
  auto _t3 = std::chrono::high_resolution_clock::now();
#endif

  // ── Exosphere ionization ──
  injectExosphereParticles();
#if ENABLE_SOA_TIMING
  auto _t4 = std::chrono::high_resolution_clock::now();
  if (MPIdata::get_rank() == 0)
    printf("  [SoA exosphere: %.2f ms]\n",
           std::chrono::duration<double, std::milli>(_t4 - _t3).count());
#endif

  // ═══════════════════════════════════════════════════════════════════════
  //  Phase 3E: Finish sort (sync + stage 4 scatter/swap) for all species
  //  After this, SoA pointers in hostPtr are updated (sorted data).
  // ═══════════════════════════════════════════════════════════════════════
  for (int i = 0; i < ns; i++) {
    cellSorters[i].finishSort(streams[i]);
    // Re-sync host struct to device (SoA pointers were swapped on host)
    cudaErrChk(cudaMemcpyAsync(pclsArrayCUDAPtr[i], pclsArrayHostPtr[i],
                                sizeof(particleArrayCUDA), cudaMemcpyDefault, streams[i]));
  }

  // ═══════════════════════════════════════════════════════════════════════
  //  Phase 3F: Sync all streams — ensures stage 4 scatter is FULLY done
  //  SAFE: after this, no GPU kernels are running. We can expand() safely
  //  (which does cudaMalloc + cudaMemcpy + cudaFree on SoA buffers).
  // ═══════════════════════════════════════════════════════════════════════
  for (int i = 0; i < ns; i++) {
    cudaErrChk(cudaStreamSynchronize(streams[i]));
  }
#if ENABLE_SOA_TIMING
  auto _t4b = std::chrono::high_resolution_clock::now();
#endif

  // ═══════════════════════════════════════════════════════════════════════
  //  Phase 3G: Per-species: expand + H2D incoming + moments + D2H
  //  Memory mutations (expand) happen first per species, then all kernels.
  // ═══════════════════════════════════════════════════════════════════════
  const auto momentGridSize = grid->getNXN() * grid->getNYN() * grid->getNZN();

  for (int i = 0; i < ns; i++) {

    // Total particles = stayed (sorted prefix) + incoming (MPI + repopulated + exosphere)
    auto newPclNum = stayedParticle[i] + particlesCommInj[i]->getCommNOP();

    // ── Expand SoA arrays if needed (may cudaMalloc + cudaFree) ──
    // SAFE: Phase 3F sync guarantees no kernels are using these buffers.
    if ((newPclNum * 1.2) >= pclsArrayHostPtr[i]->getSize()) {
      pclsArrayHostPtr[i]->expand(newPclNum * 1.5, streams[i]);
      departureArrayHostPtr[i]->expand(pclsArrayHostPtr[i]->getSize(), streams[i]);
      cudaErrChk(cudaMemcpyAsync(departureArrayCUDAPtr[i], departureArrayHostPtr[i], sizeof(departureArrayType), cudaMemcpyDefault, streams[i]));
    }

    // ── H2D incoming particles via staging buffer ──
    const int incomingCount = particlesCommInj[i]->getCommNOP();
    if (incomingCount > 0) {
      if (static_cast<uint32_t>(incomingCount) > incomingStagingHostPtr[i]->getSize()) {
        incomingStagingHostPtr[i]->expand(incomingCount * 1.5, streams[i]);
        cudaErrChk(cudaMemcpyAsync(incomingStagingCUDAPtr[i], incomingStagingHostPtr[i],
                    sizeof(arrayCUDA<SpeciesParticle>), cudaMemcpyDefault, streams[i]));
      }
      cudaErrChk(cudaMemcpyAsync(incomingStagingHostPtr[i]->getArray(),
                particlesCommInj[i]->getCommPclsData(),
                incomingCount * sizeof(SpeciesParticle),
                cudaMemcpyDefault, streams[i]));
    }

    // ── Update NOP to total (stayed + incoming) and sync to device ──
    pclsArrayHostPtr[i]->setNOE(newPclNum);
    cudaErrChk(cudaMemcpyAsync(pclsArrayCUDAPtr[i], pclsArrayHostPtr[i],
                                sizeof(particleArrayCUDA), cudaMemcpyDefault, streams[i]));

    // ── Scatter incoming AoS → SoA at offset=stayedParticle[i] ──
    if (incomingCount > 0)
      scatterAoSToSoAKernel<<<getGridSize(incomingCount, 256), 256, 0, streams[i]>>>(
          incomingStagingHostPtr[i]->getArray(),
          pclsArrayCUDAPtr[i], (uint32_t)stayedParticle[i], (uint32_t)incomingCount);

    // ── Zero moments array ──
    cudaErrChk(cudaMemsetAsync(momentsCUDAPtr[i], 0,
                                momentGridSize * 10 * sizeof(cudaMomentType), streams[i]));

    // ── Cell-aware moment kernel for sorted prefix [0, stayedParticle[i]) ──
    const uint32_t numSorted = stayedParticle[i];
    if (numSorted > 0) {
      const int numCells = cellSorters[i].getNumCells();
      const int warps = numCells;
      const int threads = warps * 32;
      cellAwareMomentKernel<<<getGridSize(threads, 256), 256, 0, streams[i]>>>(
          cellSorters[i].getCellStartOffsets(),
          numCells,
          numSorted,
          pclsArrayCUDAPtr[i],
          grid3DCUDACUDAPtr,
          momentsCUDAPtr[i]);
    }

    // ── Flat moment kernel for unsorted tail [stayedParticle[i], newPclNum) ──
    const int tailCount = newPclNum - stayedParticle[i];
    if (tailCount > 0)
      momentKernelNew<<<getGridSize(tailCount, 128), 128, 0, streams[i]>>>(
          momentParamCUDAPtr[i], grid3DCUDACUDAPtr, momentsCUDAPtr[i], stayedParticle[i]);

    // ── Reset hashedSum + departureArray ──
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
    copyMomentsD2H(i, streams[i]);
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

//! MAXWELL SOLVER for Bfield (assuming Efield has already been calculated)
void c_Solver::CalculateB(int cycle) {
  timeTasks_set_main_task(TimeTasks::FIELDS);
  // calculate the B field
  EMf->calculateB(cycle);
}

void c_Solver::MomentsAwait() {

  timeTasks_set_main_task(TimeTasks::MOMENTS);

  // synchronize
  cudaErrChk(cudaDeviceSynchronize());

  constexpr bool PARTICLE_MERGING = false; // set to true to enable particle merging, false to disable. Note that the merging process is not fully optimized yet, so it might cause performance drop if enabled. Use with caution.
  if constexpr(PARTICLE_MERGING)
  {
    // check which one to merge
    for(int i = 0; i < ns; i++) {
      if(pclsArrayHostPtr[i]->getNOP() > 1.05 * pclsArrayHostPtr[i]->getInitialNOP()) {
        toBeMerged[2 * i] = 1;
      }
      else{
        toBeMerged[2 * i] = 0;
      }
    }
    // select spcecies to merge: the one that has not been merged for most cycles among the species that require merging
    mergeIdx = -1;
    int mergeCountFromLast = -1;
    for(int i=0;i<ns;i++){
        if( (toBeMerged[2 * i] == 1) && (toBeMerged[2 * i + 1] > mergeCountFromLast) ){
          mergeIdx = i;
          mergeCountFromLast = toBeMerged[2 * i + 1];
        }
    }

    if (mergeIdx >= 0 && mergeIdx < ns){
      const auto& i = mergeIdx; 

      // Ensure host SoA vectors have enough capacity
      const uint32_t mergeNopCopy = pclsArrayHostPtr[i]->getNOP();
      particlesHost[i]->prepareSoAForNOP(mergeNopCopy);

      // Direct GPU SoA → host SoA (no AoS intermediary)
      if (mergeNopCopy > 0) {
        const size_t bytes = mergeNopCopy * sizeof(double);
        cudaErrChk(cudaMemcpyAsync(particlesHost[i]->getUallMut(), pclsArrayHostPtr[i]->getU(), bytes, cudaMemcpyDefault, streams[i]));
        cudaErrChk(cudaMemcpyAsync(particlesHost[i]->getVallMut(), pclsArrayHostPtr[i]->getV(), bytes, cudaMemcpyDefault, streams[i]));
        cudaErrChk(cudaMemcpyAsync(particlesHost[i]->getWallMut(), pclsArrayHostPtr[i]->getW(), bytes, cudaMemcpyDefault, streams[i]));
        cudaErrChk(cudaMemcpyAsync(particlesHost[i]->getQallMut(), pclsArrayHostPtr[i]->getQ(), bytes, cudaMemcpyDefault, streams[i]));
        cudaErrChk(cudaMemcpyAsync(particlesHost[i]->getXallMut(), pclsArrayHostPtr[i]->getX(), bytes, cudaMemcpyDefault, streams[i]));
        cudaErrChk(cudaMemcpyAsync(particlesHost[i]->getYallMut(), pclsArrayHostPtr[i]->getY(), bytes, cudaMemcpyDefault, streams[i]));
        cudaErrChk(cudaMemcpyAsync(particlesHost[i]->getZallMut(), pclsArrayHostPtr[i]->getZ(), bytes, cudaMemcpyDefault, streams[i]));
        cudaErrChk(cudaMemcpyAsync(particlesHost[i]->getTallMut(), pclsArrayHostPtr[i]->getT(), bytes, cudaMemcpyDefault, streams[i]));
      }
    }
  }
  else
  {
    mergeIdx = -1; 
  }

  for (int i = 0; i < ns; i++)
  {
    EMf->communicateGhostP2G(i);
  }

  EMf->setZeroDerivedMoments();
  // Fill with constant charge the planet
  if (col->getCase()=="Dipole") {
    EMf->ConstantChargePlanet(col->getL_square(),col->getx_center_planet(),col->gety_center_planet(),col->getz_center_planet());
  }else if(col->getCase()=="Dipole2D") {
	EMf->ConstantChargePlanet2DPlaneXZ(col->getL_square(),col->getx_center_planet(),col->getz_center_planet());
  }
  // Set a constant charge in the OpenBC boundaries
  //EMf->ConstantChargeOpenBC();
  // sum all over the species
  EMf->sumOverSpecies();
  // calculate densities on centers from nodes
  EMf->interpDensitiesN2C();
  // calculate the hat quantities for the implicit method
  EMf->calculateHatFunctions();
}

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


void c_Solver::WriteOutput(int cycle) {

#ifdef USE_CATALYST
  Adaptor::CoProcess(col->getDt()*cycle, cycle, EMf);
#endif

  WriteConserved(cycle);

  // ---- Restart checkpoint ----
  if (restart_cycle > 0 && cycle % restart_cycle == 0) {
    cudaErrChk(cudaEventSynchronize(eventOutputCopy));
    // SoA data is already in host vectors after outputCopyAsync — no conversion needed
    ioManager->writeRestart(cycle);
  }

  if (!Parameters::get_doWriteOutput()) return;

  // ---- Field output ----
  if (!col->field_output_is_off() &&
      (cycle % col->getFieldOutputCycle() == 0 || cycle == first_cycle)) {
    ioManager->writeFields(cycle);
  }

  // ---- Particle output ----
  if (!col->particle_output_is_off() &&
      cycle % col->getParticlesOutputCycle() == 0) {
    cudaErrChk(cudaEventSynchronize(eventOutputCopy));
    // SoA data is already in host vectors after outputCopyAsync — no conversion needed
    ioManager->writeParticles(cycle);
  }

  // ---- Test-particle output ----
  if (nstestpart > 0 && !col->testparticle_output_is_off() &&
      cycle % col->getTestParticlesOutputCycle() == 0) {
    ioManager->writeTestParticles(cycle);
  }
}

void c_Solver::outputCopyAsync(int cycle) { // -1 to enable
  if (ioManager->needsParticleSync(cycle + 1)) {
    for (int i = 0; i < ns; i++) {
      const uint32_t nop = pclsArrayHostPtr[i]->getNOP();
      // Prepare host SoA vectors to receive nop particles
      particlesHost[i]->prepareSoAForNOP(nop);
      if (nop == 0) continue;
      const size_t bytes = nop * sizeof(double);
      // Direct GPU SoA → host SoA: 8 async copies, no AoS intermediary
      cudaErrChk(cudaMemcpyAsync(particlesHost[i]->getUallMut(), pclsArrayHostPtr[i]->getU(), bytes, cudaMemcpyDefault, streams[0]));
      cudaErrChk(cudaMemcpyAsync(particlesHost[i]->getVallMut(), pclsArrayHostPtr[i]->getV(), bytes, cudaMemcpyDefault, streams[0]));
      cudaErrChk(cudaMemcpyAsync(particlesHost[i]->getWallMut(), pclsArrayHostPtr[i]->getW(), bytes, cudaMemcpyDefault, streams[0]));
      cudaErrChk(cudaMemcpyAsync(particlesHost[i]->getQallMut(), pclsArrayHostPtr[i]->getQ(), bytes, cudaMemcpyDefault, streams[0]));
      cudaErrChk(cudaMemcpyAsync(particlesHost[i]->getXallMut(), pclsArrayHostPtr[i]->getX(), bytes, cudaMemcpyDefault, streams[0]));
      cudaErrChk(cudaMemcpyAsync(particlesHost[i]->getYallMut(), pclsArrayHostPtr[i]->getY(), bytes, cudaMemcpyDefault, streams[0]));
      cudaErrChk(cudaMemcpyAsync(particlesHost[i]->getZallMut(), pclsArrayHostPtr[i]->getZ(), bytes, cudaMemcpyDefault, streams[0]));
      cudaErrChk(cudaMemcpyAsync(particlesHost[i]->getTallMut(), pclsArrayHostPtr[i]->getT(), bytes, cudaMemcpyDefault, streams[0]));
    }
    cudaErrChk(cudaEventRecord(eventOutputCopy, streams[0]));
  }
}

// write the conserved quantities
void c_Solver::WriteConserved(int cycle) {
  if(col->getDiagnosticsOutputCycle() > 0 && cycle % col->getDiagnosticsOutputCycle() == 0)
  {
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

void c_Solver::WriteVelocityDistribution(int cycle)
{
  // Velocity distribution
  //if(cycle % col->getVelocityDistributionOutputCycle() == 0)
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
}

// This seems to record values at a grid of sample points
//
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

void c_Solver::Finalize() {

  pclNumCSV.close();

  if (col->getCallFinalize() && Parameters::get_doWriteOutput() && col->getRestartOutputCycle() > 0)
  {
    outputCopyAsync(-1);
    cudaErrChk(cudaEventSynchronize(eventOutputCopy));
    convertOutputParticlesToSynched();
    ioManager->writeRestart((col->getNcycles() + first_cycle) - 1);
  }

  ioManager->finalize();

  deInitCUDA();

  // stop profiling
  my_clock->stopTiming();
}

//! place the particles into new cells according to their current position
void c_Solver::sortParticles() {

  for(int species_idx=0; species_idx<ns; species_idx++)
    particlesHost[species_idx]->sort_particles_serial();

}

// ────────────────────────────────────────────────────────────────────────────
// GPU cell-based counting sort for all species
// DEPRECATED: sorting is now integrated into MoverAwaitAndPclExchange.
// Kept for backward compatibility only — should not be called in the main loop.
// ────────────────────────────────────────────────────────────────────────────
void c_Solver::SortParticlesGPU() {
  // No-op: sorting is now performed inside MoverAwaitAndPclExchange
  // (Phases 3B–3F) using the split enqueueSortAsync/finishSort API.
}

void c_Solver::pad_particle_capacities()
{
  for (int i = 0; i < ns; i++)
    particlesHost[i]->padCapacities();

  for (int i = 0; i < nstestpart; i++)
    testpart[i]->padCapacities();
}

// No-op: particle data is always SoA.
void c_Solver::convertParticlesToSoA()
{
}

// No-op: particle data is always SoA.
void c_Solver::convertParticlesToAoS()
{
}

// All I/O reads SoA vectors directly — no conversion needed for SoA-mode instances.
// Kept for backward compat but is now a no-op for the normal output path.
void c_Solver::convertOutputParticlesToSynched()
{
  // No-op: SoA-mode particlesHost/testpart already have authoritative data in SoA vectors.
  // All I/O writers use getXall()/getUall() which read SoA directly.
}


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
 * A configurable safety margin (memoryReserveFraction) keeps a fraction of GPU
 * memory free for field arrays, moments, and other allocations.
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

  // ── Compute memory-aware particle budget per species ──
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

  // Enqueue one task per planetary species.
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

  // Wait for all species to complete before the H2D copy loop runs.
  for (auto& future : exosphereTaskFutures) {
    future.get();
  }
}
