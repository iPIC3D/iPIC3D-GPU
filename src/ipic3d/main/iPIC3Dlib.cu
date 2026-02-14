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
#include "ipicdefs.h"
#include "debug.h"
#include "Parameters.h"
#include "ompdefs.h"
#include "VCtopology3D.h"
#include "Collective.h"
#include "Grid3DCU.h"
#include "EMfields3D.h"
#include "Particles3D.h"
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

#include "Moments.h" // for debugging


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
  if(part) // exchange particles
  {
    for (int i = 0; i < ns; i++)
    {
      // placement delete
      part[i].~Particles3D();
    }
    free(part);
  }

  if(outputPart) // initial and output particles
  {
    for (int i = 0; i < ns; i++)
    {
      // placement delete
      outputPart[i].~Particles3D();
    }
    free(outputPart);
  }

#ifdef USE_CATALYST
  Adaptor::Finalize();
#endif
  delete [] Ke;
  delete [] momentum;
  delete [] Qremoved;
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

  // Allocation of particles
  part = (Particles3D*) malloc(sizeof(Particles3D)*ns);
  for (int i = 0; i < ns; i++)
  {
    new(&part[i]) Particles3D(i,col,vct,grid);
    const auto totalPcl = col->getNpcel(i) * grid->getNXN() * grid->getNYN() * grid->getNZN();
    part[i].reserveSpace(totalPcl * 0.1); // reserve the size for exchange
    part[i].get_pcl_array().clear();
  }

  outputPart = (Particles3D*) malloc(sizeof(Particles3D)*ns);
  for (int i = 0; i < ns; i++)
  {
    new(&outputPart[i]) Particles3D(i,col,vct,grid);
    const auto totalPcl = col->getNpcel(i) * grid->getNXN() * grid->getNYN() * grid->getNZN();

    if (col->getRestart_status() == 0){
      outputPart[i].reserveSpace(totalPcl); 
      outputPart[i].get_pcl_array().clear();
    } else { // restart
      outputPart[i].restartLoad();
    }
  }

  // Initial Condition for PARTICLES if you are not starting from RESTART
  if (restart_status == 0) {
    for (int i = 0; i < ns; i++)
    {
      if      (col->getCase()=="ForceFree") 		outputPart[i].force_free(EMf);
#ifdef BATSRUS
      else if (col->getCase()=="BATSRUS")   		outputPart[i].MaxwellianFromFluid(EMf,col,i);
#endif
      else if (col->getCase()=="NullPoints")    	outputPart[i].maxwellianNullPoints(EMf);
      else if (col->getCase()=="TaylorGreen")           outputPart[i].maxwellianNullPoints(EMf); // Flow is initiated from the current prescribed on the grid.
      else if (col->getCase()=="GEMDoubleHarris")  	outputPart[i].maxwellianDoubleHarris(EMf);
      else                                  		outputPart[i].maxwellian(EMf);
      outputPart[i].reserve_remaining_particle_IDs();
    }
  }

  //allocate test particles if any
  nstestpart = col->getNsTestPart();
  if(nstestpart>0){
	  testpart = (Particles3D*) malloc(sizeof(Particles3D)*nstestpart);
	  for (int i = 0; i < nstestpart; i++)
	  {
	     new(&testpart[i]) Particles3D(i+ns,col,vct,grid);//species id for test particles is increased by ns
	     testpart[i].pitch_angle_energy(EMf);
	   }
  }

  // ---- Initialise modular I/O manager ----
  ioManager = new IOManager;
  if (Parameters::get_doWriteOutput() || restart_cycle > 0 || col->getCallFinalize()) {
      ioManager->init(col, vct, grid, EMf, outputPart, ns, testpart, nstestpart, first_cycle);
  }

  Ke = new double[ns];
  BulkEnergy = new double[ns];
  momentum = new double[ns];
  cq = SaveDirName + "/ConservedQuantities.txt";
  if (myrank == 0) {
    ofstream my_file(cq.c_str());
    my_file.close();
  }
  

  Qremoved = new double[ns];

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

    for(int i=0; i<ns; i++){
      // the constructor will copy particles from host to device
      pclsArrayHostPtr[i] = newHostPinnedObject<particleArrayCUDA>(outputPart+i, 1.4, streams[i]); // use the oputputPart as the initial pcls
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

    }
  }

  // one grid for all species
  grid3DCUDAHostPtr = newHostPinnedObject<grid3DCUDA>(grid);
  grid3DCUDACUDAPtr = copyToDevice(grid3DCUDAHostPtr, 0);


  // kernelParams 
  moverParamHostPtr = new moverParameter*[ns];
  moverParamCUDAPtr = new moverParameter*[ns];
  for(int i=0; i<ns; i++){
    moverParamHostPtr[i] = newHostPinnedObject<moverParameter>(outputPart+i, pclsArrayCUDAPtr[i], departureArrayCUDAPtr[i], hashedSumArrayCUDAPtr[i]);

    // init the moverParam for OpenBC, repopulateInjection, sphere
    outputPart[i].openbc_particles_outflowInfo(&moverParamHostPtr[i]->doOpenBC, moverParamHostPtr[i]->applyOpenBC, moverParamHostPtr[i]->deleteBoundary, moverParamHostPtr[i]->openBoundary);
    moverParamHostPtr[i]->appendCountAtomic = 0;

    if(col->getRHOinject(i)>0.0)
    outputPart[i].repopulate_particlesInfo(&moverParamHostPtr[i]->doRepopulateInjection, moverParamHostPtr[i]->doRepopulateInjectionSide, moverParamHostPtr[i]->repopulateBoundary);
    else moverParamHostPtr[i]->doRepopulateInjection = false;

    if (col->getCase()=="Dipole") {
      moverParamHostPtr[i]->doSphere = 1;
      moverParamHostPtr[i]->sphereOrigin[0] = col->getx_center();
      moverParamHostPtr[i]->sphereOrigin[1] = col->gety_center();
      moverParamHostPtr[i]->sphereOrigin[2] = col->getz_center();
      moverParamHostPtr[i]->sphereRadius = col->getL_square();
    } else if (col->getCase()=="Dipole2D") {
      moverParamHostPtr[i]->doSphere = 2;
      moverParamHostPtr[i]->sphereOrigin[0] = col->getx_center();
      moverParamHostPtr[i]->sphereOrigin[1] = 0.0;
      moverParamHostPtr[i]->sphereOrigin[2] = col->getz_center();
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
    for(int i=0; i<ns; i++){
      cudaErrChk(cudaHostRegister((void*)&(EMf->getRHOns().get(i,0,0,0)), gridSize*sizeof(cudaCommonType), cudaHostRegisterDefault));
      cudaErrChk(cudaHostRegister((void*)&(EMf->getJxs().get(i,0,0,0)), gridSize*sizeof(cudaCommonType), cudaHostRegisterDefault));
      cudaErrChk(cudaHostRegister((void*)&(EMf->getJys().get(i,0,0,0)), gridSize*sizeof(cudaCommonType), cudaHostRegisterDefault));
      cudaErrChk(cudaHostRegister((void*)&(EMf->getJzs().get(i,0,0,0)), gridSize*sizeof(cudaCommonType), cudaHostRegisterDefault));
      cudaErrChk(cudaHostRegister((void*)&(EMf->getpXXsn().get(i,0,0,0)), gridSize*sizeof(cudaCommonType), cudaHostRegisterDefault));
      cudaErrChk(cudaHostRegister((void*)&(EMf->getpXYsn().get(i,0,0,0)), gridSize*sizeof(cudaCommonType), cudaHostRegisterDefault));
      cudaErrChk(cudaHostRegister((void*)&(EMf->getpXZsn().get(i,0,0,0)), gridSize*sizeof(cudaCommonType), cudaHostRegisterDefault));
      cudaErrChk(cudaHostRegister((void*)&(EMf->getpYYsn().get(i,0,0,0)), gridSize*sizeof(cudaCommonType), cudaHostRegisterDefault));
      cudaErrChk(cudaHostRegister((void*)&(EMf->getpYZsn().get(i,0,0,0)), gridSize*sizeof(cudaCommonType), cudaHostRegisterDefault));
      cudaErrChk(cudaHostRegister((void*)&(EMf->getpZZsn().get(i,0,0,0)), gridSize*sizeof(cudaCommonType), cudaHostRegisterDefault));
    }

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


  dataAnalysis::dataAnalysisPipeline::createOutputDirectory(myrank, ns, vct);

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

    deleteHostPinnedObject(moverParamHostPtr[i]);
    deleteHostPinnedObject(momentParamHostPtr[i]);


    // ==================  cudaFree device object mem ==================

    cudaFree(pclsArrayCUDAPtr[i]);
    cudaFree(departureArrayCUDAPtr[i]);
    cudaFree(hashedSumArrayCUDAPtr[i]);
    cudaFree(exitingArrayCUDAPtr[i]);
    cudaFree(fillerBufferArrayCUDAPtr[i]);

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
  delete[] moverParamHostPtr;
  delete[] moverParamCUDAPtr;
  delete[] momentParamHostPtr;
  delete[] momentParamCUDAPtr;
  delete[] momentsCUDAPtr;
  delete[] toBeMerged;


  // delete streams
  for(int i=0; i<ns*2; i++)cudaStreamDestroy(streams[i]);
  delete[] streams;
  delete[] stayedParticle;
  delete[] exitingResults;

  { // unregister the pinned mem
    for (int i = 0; i < ns; i++) {

      cudaErrChk(cudaHostUnregister((void*)&(EMf->getRHOns().get(i,0,0,0))));
      cudaErrChk(cudaHostUnregister((void*)&(EMf->getJxs().get(i,0,0,0))));
      cudaErrChk(cudaHostUnregister((void*)&(EMf->getJys().get(i,0,0,0))));
      cudaErrChk(cudaHostUnregister((void*)&(EMf->getJzs().get(i,0,0,0))));
      cudaErrChk(cudaHostUnregister((void*)&(EMf->getpXXsn().get(i,0,0,0))));
      cudaErrChk(cudaHostUnregister((void*)&(EMf->getpXYsn().get(i,0,0,0))));
      cudaErrChk(cudaHostUnregister((void*)&(EMf->getpXZsn().get(i,0,0,0))));
      cudaErrChk(cudaHostUnregister((void*)&(EMf->getpYYsn().get(i,0,0,0))));
      cudaErrChk(cudaHostUnregister((void*)&(EMf->getpYZsn().get(i,0,0,0))));
      cudaErrChk(cudaHostUnregister((void*)&(EMf->getpZZsn().get(i,0,0,0))));
    }
  }

  return 0;
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
    // copy moments back to 10 densities
    cudaErrChk(cudaMemcpyAsync((void*)&(EMf->getRHOns().get(i,0,0,0)),  momentsCUDAPtr[i]+0*gridSize, gridSize*sizeof(cudaMomentType), cudaMemcpyDefault, streams[i]));
    cudaErrChk(cudaMemcpyAsync((void*)&(EMf->getJxs().get(i,0,0,0)),    momentsCUDAPtr[i]+1*gridSize, gridSize*sizeof(cudaMomentType), cudaMemcpyDefault, streams[i]));
    cudaErrChk(cudaMemcpyAsync((void*)&(EMf->getJys().get(i,0,0,0)),    momentsCUDAPtr[i]+2*gridSize, gridSize*sizeof(cudaMomentType), cudaMemcpyDefault, streams[i]));
    cudaErrChk(cudaMemcpyAsync((void*)&(EMf->getJzs().get(i,0,0,0)),    momentsCUDAPtr[i]+3*gridSize, gridSize*sizeof(cudaMomentType), cudaMemcpyDefault, streams[i]));
    cudaErrChk(cudaMemcpyAsync((void*)&(EMf->getpXXsn().get(i,0,0,0)),  momentsCUDAPtr[i]+4*gridSize, gridSize*sizeof(cudaMomentType), cudaMemcpyDefault, streams[i]));
    cudaErrChk(cudaMemcpyAsync((void*)&(EMf->getpXYsn().get(i,0,0,0)),  momentsCUDAPtr[i]+5*gridSize, gridSize*sizeof(cudaMomentType), cudaMemcpyDefault, streams[i]));
    cudaErrChk(cudaMemcpyAsync((void*)&(EMf->getpXZsn().get(i,0,0,0)),  momentsCUDAPtr[i]+6*gridSize, gridSize*sizeof(cudaMomentType), cudaMemcpyDefault, streams[i]));
    cudaErrChk(cudaMemcpyAsync((void*)&(EMf->getpYYsn().get(i,0,0,0)),  momentsCUDAPtr[i]+7*gridSize, gridSize*sizeof(cudaMomentType), cudaMemcpyDefault, streams[i]));
    cudaErrChk(cudaMemcpyAsync((void*)&(EMf->getpYZsn().get(i,0,0,0)),  momentsCUDAPtr[i]+8*gridSize, gridSize*sizeof(cudaMomentType), cudaMemcpyDefault, streams[i]));
    cudaErrChk(cudaMemcpyAsync((void*)&(EMf->getpZZsn().get(i,0,0,0)),  momentsCUDAPtr[i]+9*gridSize, gridSize*sizeof(cudaMomentType), cudaMemcpyDefault, streams[i]));

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

  cudaEvent_t event1, event2;
  cudaErrChk(cudaEventCreateWithFlags(&event1, cudaEventDisableTiming));
  cudaErrChk(cudaEventCreateWithFlags(&event2, cudaEventDisableTiming));
  auto gridSize = grid->getNXN() * grid->getNYN() * grid->getNZN();

  
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
  cudaErrChk(cudaStreamWaitEvent(streams[species], event0, 0));
  if (col->getCase()=="Dipole" || col->getCase()=="Dipole2D")
    moverSubcyclesKernel<<<getGridSize((int)pclsArrayHostPtr[species]->getNOP(), 256), 256, 0, streams[species]>>>(moverParamCUDAPtr[species], fieldForPclCUDAPtr, grid3DCUDACUDAPtr);
  else
    moverKernel<<<getGridSize((int)pclsArrayHostPtr[species]->getNOP(), 256), 256, 0, streams[species]>>>(moverParamCUDAPtr[species], fieldForPclCUDAPtr, grid3DCUDACUDAPtr);
  
  cudaErrChk(cudaEventRecord(event1, streams[species]));
  // Moment stayed
  cudaErrChk(cudaMemsetAsync(momentsCUDAPtr[species], 0, gridSize*10*sizeof(cudaMomentType), streams[species]));  // set moments to 0
  momentKernelStayed<<<getGridSize((int)pclsArrayHostPtr[species]->getNOP(), 256), 256, 0, streams[species] >>>
                          (&(moverParamCUDAPtr[species]->appendCountAtomic), momentParamCUDAPtr[species], grid3DCUDACUDAPtr, momentsCUDAPtr[species]);

  // Copy 7 exiting hashedSum to host
  cudaErrChk(cudaStreamWaitEvent(streams[species+ns], event1, 0));
  cudaErrChk(cudaMemcpyAsync(hashedSumArrayHostPtr[species], hashedSumArrayCUDAPtr[species], 
    departureArrayElementType::DELETE*sizeof(hashedSum), cudaMemcpyDefault, streams[species+ns]));

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
  cudaErrChk(cudaStreamSynchronize(streams[species+ns]));
  //cudaErrChk(cudaStreamSynchronize(streams[species]));
  int x = 0; // exiting particle number
  for(int i=0; i<departureArrayElementType::DELETE_HASHEDSUM_INDEX; i++)x += hashedSumArrayHostPtr[species][i].getSum();
  const int y = hashedSumArrayHostPtr[species][departureArrayElementType::DELETE_HASHEDSUM_INDEX].getSum(); // deleted particle number
  const int hole = x + y;
  //if (y > 0){
  //  std::cout << " Particle holes myrank: "<< MPIdata::get_rank() << " species: " << species<< " hole: " << hole << " deleted: "<< y << std::endl;
  //}
  if(x > exitingArrayHostPtr[species]->getSize()){ 
    // prepare the exitingArray
    exitingArrayHostPtr[species]->expand(x * 1.5, streams[species+ns]);
    cudaErrChk(cudaMemcpyAsync(exitingArrayCUDAPtr[species], exitingArrayHostPtr[species], 
                                sizeof(exitingArray), cudaMemcpyDefault, streams[species+ns]));
  }

  if(hole > fillerBufferArrayHostPtr[species]->getSize()){
    // prepare the fillerBuffer
    fillerBufferArrayHostPtr[species]->expand(hole * 1.5, streams[species+ns]);
    cudaErrChk(cudaMemcpyAsync(fillerBufferArrayCUDAPtr[species], fillerBufferArrayHostPtr[species], 
                                sizeof(fillerBuffer), cudaMemcpyDefault, streams[species+ns]));
  }

  if(x > part[species].get_pcl_list().capacity()){
    // expand the host array
    auto pclArray = part[species].get_pcl_arrayPtr();
    pclArray->reserve(x * 1.5);
  }

  exitingKernel<<<getGridSize((int)pclsArrayHostPtr[species]->getNOP(), 256), 256, 0, streams[species+ns]>>>(pclsArrayCUDAPtr[species], 
                departureArrayCUDAPtr[species], exitingArrayCUDAPtr[species], hashedSumArrayCUDAPtr[species]);
  cudaErrChk(cudaEventRecord(event2, streams[species+ns]));
  // Copy exiting particle to host
  cudaErrChk(cudaMemcpyAsync(part[species].get_pcl_array().getList(), exitingArrayHostPtr[species]->getArray(), 
                              x*sizeof(SpeciesParticle), cudaMemcpyDefault, streams[species+ns]));
  part[species].get_pcl_array().setSize(x);

  // Sorting, the first cycle, x might be 0
  cudaErrChk(cudaStreamWaitEvent(streams[species], event2, 0));
  if (hole > 0) 
  sortingKernel1<<<getGridSize(hole, 128), 128, 0, streams[species]>>>(pclsArrayCUDAPtr[species], departureArrayCUDAPtr[species], 
                                                          fillerBufferArrayCUDAPtr[species], hashedSumArrayCUDAPtr[species]+departureArrayElementType::FILLER_HASHEDSUM_INDEX, hole);
  sortingKernel2<<<getGridSize((int)(pclsArrayHostPtr[species]->getNOP()-hole), 256), 256, 0, streams[species]>>>(pclsArrayCUDAPtr[species], departureArrayCUDAPtr[species], 
                                                          fillerBufferArrayCUDAPtr[species], hashedSumArrayCUDAPtr[species]+departureArrayElementType::HOLE_HASHEDSUM_INDEX, pclsArrayHostPtr[species]->getNOP()-hole);

  cudaErrChk(cudaEventDestroy(event1));
  cudaErrChk(cudaEventDestroy(event2));
  cudaErrChk(cudaStreamSynchronize(streams[species+ns])); // exiting particle copied
  return hole; // Number of exiting + deleted particles
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

    cudaErrChk(cudaStreamSynchronize(streams[i])); // wait for the copy
    // sort
    outputPart[i].sort_particles_parallel(cellCountHostPtr, cellOffsetHostPtr);

    const int totalCells = grid->getNXC() * grid->getNYC() * grid->getNZC();
    cudaErrChk(cudaMemcpyAsync(cellCountCUDAPtr, cellCountHostPtr, totalCells*sizeof(int), cudaMemcpyDefault, streams[i]));
    cudaErrChk(cudaMemcpyAsync(cellOffsetCUDAPtr, cellOffsetHostPtr, totalCells*sizeof(int), cudaMemcpyDefault, streams[i]));

    cudaErrChk(cudaMemcpyAsync(pclsArrayHostPtr[i]->getpcls(), outputPart[i].get_pcl_array().getList(), 
                              pclsArrayHostPtr[i]->getNOP()*sizeof(SpeciesParticle), cudaMemcpyDefault, streams[i]));

    // merge
    mergingKernel<<<getGridSize(totalCells * WARP_SIZE, 256), 256, 0, streams[i]>>>(cellOffsetCUDAPtr, cellCountCUDAPtr, 
        grid3DCUDACUDAPtr, pclsArrayCUDAPtr[i], departureArrayCUDAPtr[i]);

    exitingResults[i] = threadPoolPtr->enqueue(&c_Solver::cudaLauncherAsync, this, i);

    toBeMerged[2 * i + 1] = 0;
    mergeIdx = -1; // merged
  }

  return (false);
}

bool c_Solver::MoverAwaitAndPclExchange()
{

  for (int i = 0; i < ns; i++){ 
    auto x = exitingResults[i].get(); // holes
    stayedParticle[i] = pclsArrayHostPtr[i]->getNOP() - x;
  }
  // exiting particles are copied back

  for (int i = 0; i < ns; i++)  // communicate each species
  {
    auto a = part[i].separate_and_send_particles();
    part[i].recommunicate_particles_until_done(1);
    // injection
    if (moverParamHostPtr[i]->doRepopulateInjection) { // Now part contains incoming particles and the injection
      part[i].repopulate_particles_onlyInjection();
    }
  }


  for(int i=0; i<ns; i++){

    // Copy repopulate particle and the incoming particles to device
    const auto oldPclNum = pclsArrayHostPtr[i]->getNOP();
    pclsArrayHostPtr[i]->setNOE(stayedParticle[i]); // After the Sorting
    auto newPclNum = stayedParticle[i] + part[i].getNOP();

    // now the host array contains the entering particles
    if((newPclNum * 1.2) >= pclsArrayHostPtr[i]->getSize()){ // not enough size, expand the device array size
      pclsArrayHostPtr[i]->expand(newPclNum * 1.5, streams[i]);
      departureArrayHostPtr[i]->expand(pclsArrayHostPtr[i]->getSize(), streams[i]);
      cudaErrChk(cudaMemcpyAsync(departureArrayCUDAPtr[i], departureArrayHostPtr[i], sizeof(departureArrayType), cudaMemcpyDefault, streams[i]));
    }
    // now enough size on device pcls array, copy particles
    cudaErrChk(cudaMemcpyAsync(pclsArrayHostPtr[i]->getpcls() + pclsArrayHostPtr[i]->getNOP(), 
              (void*)&(part[i].get_pcl_list()[0]), 
              part[i].getNOP()*sizeof(SpeciesParticle),
              cudaMemcpyDefault, streams[i]));

              
    pclsArrayHostPtr[i]->setNOE(newPclNum); 
    cudaErrChk(cudaMemcpyAsync(pclsArrayCUDAPtr[i], pclsArrayHostPtr[i], sizeof(particleArrayCUDA), cudaMemcpyDefault, streams[i]));    
    // moment for new particles, incoming, repopulate
    if(pclsArrayHostPtr[i]->getNOP() - stayedParticle[i] > 0)
    momentKernelNew<<<getGridSize(pclsArrayHostPtr[i]->getNOP() - stayedParticle[i], 128u), 128, 0, streams[i] >>>
                      (momentParamCUDAPtr[i], grid3DCUDACUDAPtr, momentsCUDAPtr[i], stayedParticle[i]);

    // reset the hashedSum, no need for departureArray it will be cleared in Mover
    for(int j=0; j<departureArrayElementType::HASHED_SUM_NUM; j++)hashedSumArrayHostPtr[i][j].resetBucket();
    cudaErrChk(cudaMemcpyAsync(hashedSumArrayCUDAPtr[i], hashedSumArrayHostPtr[i], (departureArrayElementType::HASHED_SUM_NUM) * sizeof(hashedSum), cudaMemcpyDefault, streams[i]));

    cudaErrChk(cudaMemsetAsync(departureArrayHostPtr[i]->getArray(), 0, departureArrayHostPtr[i]->getSize() * sizeof(departureArrayElementType), streams[i]));

  }

  auto gridSize = grid->getNXN() * grid->getNYN() * grid->getNZN();
  
  for(int i=0; i<ns; i++){ // copy moments back to 10 densities
    cudaErrChk(cudaMemcpyAsync((void*)&(EMf->getRHOns().get(i,0,0,0)),  momentsCUDAPtr[i]+0*gridSize, gridSize*sizeof(cudaMomentType), cudaMemcpyDefault, streams[i]));
    cudaErrChk(cudaMemcpyAsync((void*)&(EMf->getJxs().get(i,0,0,0)),    momentsCUDAPtr[i]+1*gridSize, gridSize*sizeof(cudaMomentType), cudaMemcpyDefault, streams[i]));
    cudaErrChk(cudaMemcpyAsync((void*)&(EMf->getJys().get(i,0,0,0)),    momentsCUDAPtr[i]+2*gridSize, gridSize*sizeof(cudaMomentType), cudaMemcpyDefault, streams[i]));
    cudaErrChk(cudaMemcpyAsync((void*)&(EMf->getJzs().get(i,0,0,0)),    momentsCUDAPtr[i]+3*gridSize, gridSize*sizeof(cudaMomentType), cudaMemcpyDefault, streams[i]));
    cudaErrChk(cudaMemcpyAsync((void*)&(EMf->getpXXsn().get(i,0,0,0)),  momentsCUDAPtr[i]+4*gridSize, gridSize*sizeof(cudaMomentType), cudaMemcpyDefault, streams[i]));
    cudaErrChk(cudaMemcpyAsync((void*)&(EMf->getpXYsn().get(i,0,0,0)),  momentsCUDAPtr[i]+5*gridSize, gridSize*sizeof(cudaMomentType), cudaMemcpyDefault, streams[i]));
    cudaErrChk(cudaMemcpyAsync((void*)&(EMf->getpXZsn().get(i,0,0,0)),  momentsCUDAPtr[i]+6*gridSize, gridSize*sizeof(cudaMomentType), cudaMemcpyDefault, streams[i]));
    cudaErrChk(cudaMemcpyAsync((void*)&(EMf->getpYYsn().get(i,0,0,0)),  momentsCUDAPtr[i]+7*gridSize, gridSize*sizeof(cudaMomentType), cudaMemcpyDefault, streams[i]));
    cudaErrChk(cudaMemcpyAsync((void*)&(EMf->getpYZsn().get(i,0,0,0)),  momentsCUDAPtr[i]+8*gridSize, gridSize*sizeof(cudaMomentType), cudaMemcpyDefault, streams[i]));
    cudaErrChk(cudaMemcpyAsync((void*)&(EMf->getpZZsn().get(i,0,0,0)),  momentsCUDAPtr[i]+9*gridSize, gridSize*sizeof(cudaMomentType), cudaMemcpyDefault, streams[i]));
  }


  return (false);
}

//! MAXWELL SOLVER for Bfield (assuming Efield has already been calculated)
void c_Solver::CalculateB() {
  timeTasks_set_main_task(TimeTasks::FIELDS);
  // calculate the B field
  EMf->calculateB();
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

      if(outputPart[i].get_pcl_array().capacity() < pclsArrayHostPtr[i]->getNOP()){
        auto pclArray = outputPart[i].get_pcl_arrayPtr();
        pclArray->reserve(pclsArrayHostPtr[i]->getNOP() * 1.2);
      }
      cudaErrChk(cudaMemcpyAsync(outputPart[i].get_pcl_array().getList(), pclsArrayHostPtr[i]->getpcls(), 
                                pclsArrayHostPtr[i]->getNOP()*sizeof(SpeciesParticle), cudaMemcpyDefault, streams[i]));
      outputPart[i].get_pcl_array().setSize(pclsArrayHostPtr[i]->getNOP()); 
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
  // sum all over the species
  EMf->sumOverSpecies();
  // Fill with constant charge the planet
  if (col->getCase()=="Dipole") {
    EMf->ConstantChargePlanet(col->getL_square(),col->getx_center(),col->gety_center(),col->getz_center());
  }else if(col->getCase()=="Dipole2D") {
	EMf->ConstantChargePlanet2DPlaneXZ(col->getL_square(),col->getx_center(),col->getz_center());
  }
  // Set a constant charge in the OpenBC boundaries
  //EMf->ConstantChargeOpenBC();
  // calculate densities on centers from nodes
  EMf->interpDensitiesN2C();
  // calculate the hat quantities for the implicit method
  EMf->calculateHatFunctions();
}

void c_Solver::writeParticleNum(int cycle) {
  pclNumCSV << cycle << ",";
  for(int i=0; i<ns-1; i++){
    pclNumCSV << pclsArrayHostPtr[i]->getNOP() << ",";
  }
  pclNumCSV << pclsArrayHostPtr[ns-1]->getNOP() << std::endl;
}


void c_Solver::WriteOutput(int cycle) {

#ifdef USE_CATALYST
  Adaptor::CoProcess(col->getDt()*cycle, cycle, EMf);
#endif

  WriteConserved(cycle);

  // ---- Restart checkpoint ----
  if (restart_cycle > 0 && cycle % restart_cycle == 0) {
    cudaErrChk(cudaEventSynchronize(eventOutputCopy));
    convertOutputParticlesToSynched();
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
    for (int i = 0; i < ns; i++) {
      outputPart[i].set_particleType(ParticleType::Type::AoS);
      outputPart[i].convertParticlesToSynched();
    }
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
      if (outputPart[i].get_pcl_array().capacity() < pclsArrayHostPtr[i]->getNOP()) {
        auto pclArray = outputPart[i].get_pcl_arrayPtr();
        pclArray->reserve(pclsArrayHostPtr[i]->getNOP() * 1.2);
      }
      cudaErrChk(cudaMemcpyAsync(
          outputPart[i].get_pcl_array().getList(),
          pclsArrayHostPtr[i]->getpcls(),
          pclsArrayHostPtr[i]->getNOP() * sizeof(SpeciesParticle),
          cudaMemcpyDefault, streams[0]));
      outputPart[i].get_pcl_array().setSize(pclsArrayHostPtr[i]->getNOP());
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
    for (int is = 0; is < ns; is++) {
      Ke[is] = outputPart[is].getKe();
      BulkEnergy[is] = EMf->getBulkEnergy(is);
      TOTenergy += Ke[is];
      momentum[is] = outputPart[is].getP();
      TOTmomentum += momentum[is];
    }
    if (myrank == (nprocs-1)) {
      ofstream my_file(cq.c_str(), fstream::app);
      if(cycle == 0)my_file << "\t" << "\t" << "\t" << "Total_Energy" << "\t" << "Momentum" << "\t" << "Eenergy" << "\t" << "Benergy" << "\t" << "Kenergy" << "\t" << "Kenergy(species)" << "\t" << "BulkEnergy(species)" << endl;
      my_file << cycle << "\t" << "\t" << (Eenergy + Benergy + TOTenergy) << "\t" << TOTmomentum << "\t" << Eenergy << "\t" << Benergy << "\t" << TOTenergy;
      for (int is = 0; is < ns; is++) my_file << "\t" << Ke[is];
      for (int is = 0; is < ns; is++) my_file << "\t" << BulkEnergy[is];
      my_file << endl;
      my_file.close();
    }
  }
}
/* write the conserved quantities
void c_Solver::WriteConserved(int cycle) {
  if(col->getDiagnosticsOutputCycle() > 0 && cycle % col->getDiagnosticsOutputCycle() == 0)
  {
	if(cycle==0)buf_counter=0;
    Eenergy[buf_counter] = EMf->getEenergy();
    Benergy[buf_counter] = EMf->getBenergy();
    Kenergy[buf_counter] = 0.0;
    TOTmomentum[buf_counter] = 0.0;
    for (int is = 0; is < ns; is++) {
      Ke[is] = part[is].getKe();
      Kenergy[buf_counter] += Ke[is];
      momentum[is] = part[is].getP();
      TOTmomentum[buf_counter] += momentum[is];
    }
    outputcycle[buf_counter] = cycle;
    buf_counter ++;

    //Flush out result if this is the last cycle or the buffer is full
    if(buf_counter==OUTPUT_BUFSIZE || cycle==(LastCycle()-1)){
    	if (myrank == (nprocs-1)) {
    		ofstream my_file(cq.c_str(), fstream::app);
    		stringstream ss;
      //if(cycle/OUTPUT_BUFSIZE == 0)
      //my_file  << "Cycle" << "\t" << "Total_Energy" 				 << "\t" << "Momentum" << "\t" << "Eenergy" <<"\t" << "Benergy" << "\t" << "Kenergy" << endl;
    		for(int bufid=0;bufid<OUTPUT_BUFSIZE;bufid++)
    			ss << outputcycle[bufid] << "\t" << (Eenergy[bufid]+Benergy[bufid]+Kenergy[bufid])<< "\t" << TOTmomentum[bufid] << "\t" << Eenergy[bufid] << "\t" << Benergy[bufid] << "\t" << Kenergy[bufid] << endl;

    		my_file << ss;
    		my_file.close();
    	}
    	buf_counter = 0;
    }
  }
}*/

void c_Solver::WriteVelocityDistribution(int cycle)
{
  // Velocity distribution
  //if(cycle % col->getVelocityDistributionOutputCycle() == 0)
  {
    for (int is = 0; is < ns; is++) {
      double maxVel = outputPart[is].getMaxVelocity();
      long long *VelocityDist = outputPart[is].getVelocityDistribution(nDistributionBins, maxVel);
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
    part[species_idx].sort_particles_serial();

}

void c_Solver::pad_particle_capacities()
{
  for (int i = 0; i < ns; i++)
    part[i].pad_capacities();

  for (int i = 0; i < nstestpart; i++)
    testpart[i].pad_capacities();
}

// convert particle to struct of arrays (assumed by I/O)
void c_Solver::convertParticlesToSoA()
{
  for (int i = 0; i < ns; i++)
    part[i].convertParticlesToSoA();
}

// convert particle to array of structs (used in computing)
void c_Solver::convertParticlesToAoS()
{
  for (int i = 0; i < ns; i++)
    part[i].convertParticlesToAoS();
}

// convert particle to array of structs (used in computing)
void c_Solver::convertOutputParticlesToSynched()
{
  for (int i = 0; i < ns; i++){
    outputPart[i].set_particleType(ParticleType::Type::AoS); // otherwise the output is not synched
    outputPart[i].convertParticlesToSynched();
  }

  for (int i = 0; i < nstestpart; i++){
    testpart[i].set_particleType(ParticleType::Type::AoS); // otherwise the output is not synched
    testpart[i].convertParticlesToSynched();
  }
}


int c_Solver::LastCycle() {
    return (col->getNcycles() + first_cycle);
}
