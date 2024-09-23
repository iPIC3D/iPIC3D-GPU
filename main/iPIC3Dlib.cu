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
//
#ifndef NO_HDF5
#include "WriteOutputParallel.h"
#include "OutputWrapperFPP.h"
#endif

#include <iostream>
#include <fstream>
#include <sstream>

#include "Moments.h" // for debugging

#if CUDA_ON == true
#include "cudaTypeDef.cuh"
#include "cuda.h"
#include "momentKernel.cuh"
#include "particleArrayCUDA.cuh"
#include "moverKernel.cuh"
#include "particleExchange.cuh"
#include "thread"
#include "future"
#endif

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
#ifndef NO_HDF5
  delete outputWrapperFPP;
#endif
  // delete particles
  //
  if(part)
  {
    for (int i = 0; i < ns; i++)
    {
      // placement delete
      part[i].~Particles3D();
    }
    free(part);
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
    //check and create the output directory
    checkOutputFolder(SaveDirName);
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
  }

  // Initial Condition for PARTICLES if you are not starting from RESTART
  if (restart_status == 0) {
    for (int i = 0; i < ns; i++)
    {
      if      (col->getCase()=="ForceFree") 		part[i].force_free(EMf);
#ifdef BATSRUS
      else if (col->getCase()=="BATSRUS")   		part[i].MaxwellianFromFluid(EMf,col,i);
#endif
      else if (col->getCase()=="NullPoints")    	part[i].maxwellianNullPoints(EMf);
      else if (col->getCase()=="TaylorGreen")           part[i].maxwellianNullPoints(EMf); // Flow is initiated from the current prescribed on the grid.
      else if (col->getCase()=="GEMDoubleHarris")  	part[i].maxwellianDoubleHarris(EMf);
      else                                  		part[i].maxwellian(EMf);
      part[i].reserve_remaining_particle_IDs();
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

  if ( Parameters::get_doWriteOutput()){
		#ifndef NO_HDF5
	  	if(col->getWriteMethod() == "shdf5" || col->getCallFinalize() || restart_cycle>0 ||
			  (col->getWriteMethod()=="pvtk" && !col->particle_output_is_off()) )
		{
			  outputWrapperFPP = new OutputWrapperFPP;
			  fetch_outputWrapperFPP().init_output_files(col,vct,grid,EMf,part,ns,testpart,nstestpart);
		}
		#endif
	  if(!col->field_output_is_off()){
		  if(col->getWriteMethod()=="pvtk"){
			  if(!(col->getFieldOutputTag()).empty())
				  fieldwritebuffer = newArr4(float,(grid->getNZN()-3),grid->getNYN()-3,grid->getNXN()-3,3);
			  if(!(col->getMomentsOutputTag()).empty())
				  momentwritebuffer=newArr3(float,(grid->getNZN()-3), grid->getNYN()-3, grid->getNXN()-3);
		  }
		  else if(col->getWriteMethod()=="nbcvtk"){
		    momentreqcounter=0;
		    fieldreqcounter = 0;
			  if(!(col->getFieldOutputTag()).empty())
				  fieldwritebuffer = newArr4(float,(grid->getNZN()-3)*4,grid->getNYN()-3,grid->getNXN()-3,3);
			  if(!(col->getMomentsOutputTag()).empty())
				  momentwritebuffer=newArr3(float,(grid->getNZN()-3)*14, grid->getNYN()-3, grid->getNXN()-3);
		  }
	  }
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

#if CUDA_ON == true
  initCUDA();
#endif

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
  streams = new cudaStream_t[ns*2]; stayedParticle = new int[ns];
  for(int i=0; i<ns; i++){ cudaErrChk(cudaStreamCreate(streams+i)); cudaErrChk(cudaStreamCreate(streams+i+ns)); stayedParticle[i] = 0; }

	{// init arrays on device, pointers are device pointer, copied
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
      pclsArrayHostPtr[i] = new particleArrayCUDA(part+i, streams[i]);
      pclsArrayCUDAPtr[i] = pclsArrayHostPtr[i]->copyToDevice();
      // clear the host pclArray
      part[i].get_pcl_array().clear(); // 0

      departureArrayHostPtr[i] = new departureArrayType(pclsArrayHostPtr[i]->getSize()); // same length
      departureArrayCUDAPtr[i] = departureArrayHostPtr[i]->copyToDevice();
      cudaErrChk(cudaMemsetAsync(departureArrayHostPtr[i]->getArray(), 0, departureArrayHostPtr[i]->getSize() * sizeof(departureArrayElementType), streams[i]));

      hashedSumArrayHostPtr[i] = new hashedSum[8]{ // 
        hashedSum(5), hashedSum(5), hashedSum(5), hashedSum(5), 
        hashedSum(5), hashedSum(5), hashedSum(10), hashedSum(10)
      };
      hashedSumArrayCUDAPtr[i] = copyArrayToDevice(hashedSumArrayHostPtr[i], 8);
      
      exitingArrayHostPtr[i] = new exitingArray(0.1 * pclsArrayHostPtr[i]->getNOP());
      exitingArrayCUDAPtr[i] = exitingArrayHostPtr[i]->copyToDevice();
      fillerBufferArrayHostPtr[i] = new fillerBuffer(0.1 * pclsArrayHostPtr[i]->getNOP());
      fillerBufferArrayCUDAPtr[i] = fillerBufferArrayHostPtr[i]->copyToDevice();

    }
  }

  // one grid for all species
  grid3DCUDAHostPtr = new grid3DCUDA(grid);
  grid3DCUDACUDAPtr = copyToDevice(grid3DCUDAHostPtr, 0);


  // kernelParams 
  moverParamHostPtr = new moverParameter*[ns];
  moverParamCUDAPtr = new moverParameter*[ns];
  for(int i=0; i<ns; i++){
    moverParamHostPtr[i] = new moverParameter(part+i, pclsArrayCUDAPtr[i], departureArrayCUDAPtr[i], hashedSumArrayCUDAPtr[i]);
    moverParamCUDAPtr[i] = copyToDevice(moverParamHostPtr[i], streams[i]);
  }

  momentParamHostPtr = new momentParameter*[ns];
  momentParamCUDAPtr = new momentParameter*[ns];
  for(int i=0; i<ns; i++){
    momentParamHostPtr[i] = new momentParameter(pclsArrayCUDAPtr[i], departureArrayCUDAPtr[i]);
    momentParamCUDAPtr[i] = copyToDevice(momentParamHostPtr[i], streams[i]);
  }

  // simple device buffer, allocate one dimension array on device memory
  auto gridSize = grid->getNXN() * grid->getNYN() * grid->getNZN();
  momentsCUDAPtr = new cudaTypeArray1<cudaCommonType>[ns];
  for(int i=0; i<ns; i++)cudaMallocAsync(&(momentsCUDAPtr[i]), gridSize*10*sizeof(cudaCommonType), streams[i]);

  cudaMallocAsync(&fieldForPclCUDAPtr, gridSize*8*sizeof(cudaCommonType), 0);

  cudaDeviceSynchronize();

  if(MPIdata::get_rank() == 0)std::cout << "CUDA Init finished" << std::endl;

  return 0;

}


void c_Solver::CalculateMoments(bool isInit) {

  timeTasks_set_main_task(TimeTasks::MOMENTS);

#if MOMENT_CUDA_ON == false
  pad_particle_capacities();
  // vectorized assumes that particles are sorted by mesh cell
  if(Parameters::get_VECTORIZE_MOMENTS())
  {
    switch(Parameters::get_MOMENTS_TYPE())
    {
      case Parameters::SoA:
        // since particles are sorted,
        // we can vectorize interpolation of particles to grid
        convertParticlesToSoA();
        sortParticles();
        EMf->sumMoments_vectorized(part);
        break;
      case Parameters::AoS:
        convertParticlesToAoS();
        sortParticles();
        EMf->sumMoments_vectorized_AoS(part);
        break;
      default:
        unsupported_value_error(Parameters::get_MOMENTS_TYPE());
    }
  }
  else
  {
    if(Parameters::get_SORTING_PARTICLES())
      sortParticles();
    switch(Parameters::get_MOMENTS_TYPE())
    {
      case Parameters::SoA:
        EMf->setZeroPrimaryMoments();
        convertParticlesToSoA();
        EMf->sumMoments(part);
        break;
      case Parameters::AoS:
        EMf->setZeroPrimaryMoments(); // clear the data to 0
        convertParticlesToAoS(); // convert 
        EMf->sumMoments_AoS(part); // sum up the 10 densities of each particles of each species
        // then calculate the weight according to their position
        // map the 10 momentum to the grid(node) with the weight
        
        break;
      case Parameters::AoSintr:
        EMf->setZeroPrimaryMoments();
        convertParticlesToAoS();
        EMf->sumMoments_AoS_intr(part);
        break;
      default:
        unsupported_value_error(Parameters::get_MOMENTS_TYPE());
    }
  }
#else

  // sum moments
  if(isInit){
    auto gridSize = grid->getNXN() * grid->getNYN() * grid->getNZN();
    for(int i=0; i<ns; i++){
      cudaErrChk(cudaMemsetAsync(momentsCUDAPtr[i], 0, gridSize*10*sizeof(cudaCommonType), streams[i]));  // set moments to 0
      // copy the particles to device---- already there...by initliazation or Mover
      // launch the moment kernel
      momentKernelNew<<<(pclsArrayHostPtr[i]->getNOP()/256 + 1), 256, 0, streams[i] >>>(momentParamCUDAPtr[i], grid3DCUDACUDAPtr, momentsCUDAPtr[i], 0);
      // copy moments back to 10 densities
      cudaErrChk(cudaMemcpyAsync((void*)&(EMf->getRHOns().get(i,0,0,0)),  momentsCUDAPtr[i]+0*gridSize, gridSize*sizeof(cudaCommonType), cudaMemcpyDefault, streams[i]));
      cudaErrChk(cudaMemcpyAsync((void*)&(EMf->getJxs().get(i,0,0,0)),    momentsCUDAPtr[i]+1*gridSize, gridSize*sizeof(cudaCommonType), cudaMemcpyDefault, streams[i]));
      cudaErrChk(cudaMemcpyAsync((void*)&(EMf->getJys().get(i,0,0,0)),    momentsCUDAPtr[i]+2*gridSize, gridSize*sizeof(cudaCommonType), cudaMemcpyDefault, streams[i]));
      cudaErrChk(cudaMemcpyAsync((void*)&(EMf->getJzs().get(i,0,0,0)),    momentsCUDAPtr[i]+3*gridSize, gridSize*sizeof(cudaCommonType), cudaMemcpyDefault, streams[i]));
      cudaErrChk(cudaMemcpyAsync((void*)&(EMf->getpXXsn().get(i,0,0,0)),  momentsCUDAPtr[i]+4*gridSize, gridSize*sizeof(cudaCommonType), cudaMemcpyDefault, streams[i]));
      cudaErrChk(cudaMemcpyAsync((void*)&(EMf->getpXYsn().get(i,0,0,0)),  momentsCUDAPtr[i]+5*gridSize, gridSize*sizeof(cudaCommonType), cudaMemcpyDefault, streams[i]));
      cudaErrChk(cudaMemcpyAsync((void*)&(EMf->getpXZsn().get(i,0,0,0)),  momentsCUDAPtr[i]+6*gridSize, gridSize*sizeof(cudaCommonType), cudaMemcpyDefault, streams[i]));
      cudaErrChk(cudaMemcpyAsync((void*)&(EMf->getpYYsn().get(i,0,0,0)),  momentsCUDAPtr[i]+7*gridSize, gridSize*sizeof(cudaCommonType), cudaMemcpyDefault, streams[i]));
      cudaErrChk(cudaMemcpyAsync((void*)&(EMf->getpYZsn().get(i,0,0,0)),  momentsCUDAPtr[i]+8*gridSize, gridSize*sizeof(cudaCommonType), cudaMemcpyDefault, streams[i]));
      cudaErrChk(cudaMemcpyAsync((void*)&(EMf->getpZZsn().get(i,0,0,0)),  momentsCUDAPtr[i]+9*gridSize, gridSize*sizeof(cudaCommonType), cudaMemcpyDefault, streams[i]));

    }
  }
  // synchronize
  cudaErrChk(cudaDeviceSynchronize());
  
#endif

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

//! MAXWELL SOLVER for Efield
void c_Solver::CalculateField(int cycle) {
  timeTasks_set_main_task(TimeTasks::FIELDS);

  // calculate the E field
  EMf->calculateE(cycle);
}


int c_Solver::cudaLauncherAsync(const int species){
  cudaSetDevice(cudaDeviceOnNode); // a must on multi-device node

  cudaEvent_t event1, event2;
  cudaErrChk(cudaEventCreateWithFlags(&event1, cudaEventDisableTiming));
  cudaErrChk(cudaEventCreateWithFlags(&event2, cudaEventDisableTiming));
  auto gridSize = grid->getNXN() * grid->getNYN() * grid->getNZN();

  // Mover
  moverKernel<<<getGridSize((int)pclsArrayHostPtr[species]->getNOP(), 256), 256, 0, streams[species]>>>(moverParamCUDAPtr[species], fieldForPclCUDAPtr, grid3DCUDACUDAPtr);
  cudaErrChk(cudaEventRecord(event1, streams[species]));
  // Moment stayed
  cudaErrChk(cudaMemsetAsync(momentsCUDAPtr[species], 0, gridSize*10*sizeof(cudaCommonType), streams[species]));  // set moments to 0
  momentKernelStayed<<<getGridSize((int)pclsArrayHostPtr[species]->getNOP(), 256), 256, 0, streams[species] >>>
                          (momentParamCUDAPtr[species], grid3DCUDACUDAPtr, momentsCUDAPtr[species]);

  // Copy 6 exiting hashedSum to host
  cudaErrChk(cudaStreamWaitEvent(streams[species+ns], event1));
  cudaErrChk(cudaMemcpyAsync(hashedSumArrayHostPtr[species], hashedSumArrayCUDAPtr[species], 
                              6*sizeof(hashedSum), cudaMemcpyDefault, streams[species+ns]));

  // After Mover
  cudaErrChk(cudaStreamSynchronize(streams[species+ns]));
  int x = 0;
  for(int i=0; i<6; i++)x += hashedSumArrayHostPtr[species][i].getSum();
  if(x > exitingArrayHostPtr[species]->getSize()){ 
    // prepare the exitingArray
    exitingArrayHostPtr[species]->expand(x * 1.5, streams[species+ns]);
    fillerBufferArrayHostPtr[species]->expand(x * 1.5, streams[species+ns]);

    cudaErrChk(cudaMemcpyAsync(exitingArrayCUDAPtr[species], exitingArrayHostPtr[species], 
                                sizeof(exitingArray), cudaMemcpyDefault, streams[species+ns]));
    cudaErrChk(cudaMemcpyAsync(fillerBufferArrayCUDAPtr[species], fillerBufferArrayHostPtr[species], 
                                sizeof(fillerBuffer), cudaMemcpyDefault, streams[species+ns]));
  }

  if(x > part[species].get_pcl_list().capacity()){
    // expand the host array
    auto pclArray = part[species].get_pcl_array();
    pclArray.reserve(x * 1.5);
  }
  exitingKernel<<<getGridSize((int)pclsArrayHostPtr[species]->getNOP(), 256), 256, 0, streams[species+ns]>>>(pclsArrayCUDAPtr[species], 
                departureArrayCUDAPtr[species], exitingArrayCUDAPtr[species], hashedSumArrayCUDAPtr[species]);
  cudaErrChk(cudaEventRecord(event2, streams[species+ns]));
  // Copy exiting particle to host
  cudaErrChk(cudaMemcpyAsync(part[species].get_pcl_array().getList(), exitingArrayHostPtr[species]->getArray(), 
                              x*sizeof(SpeciesParticle), cudaMemcpyDefault, streams[species+ns]));
  part[species].get_pcl_array().setSize(x);

  // Sorting
  cudaErrChk(cudaStreamWaitEvent(streams[species], event2));
  sortingKernel1<<<getGridSize(x, 128), 128, 0, streams[species]>>>(pclsArrayCUDAPtr[species], departureArrayCUDAPtr[species], 
                                                          fillerBufferArrayCUDAPtr[species], hashedSumArrayCUDAPtr[species]+7, x);
  sortingKernel2<<<getGridSize((int)(pclsArrayHostPtr[species]->getNOP()-x), 256), 256, 0, streams[species]>>>(pclsArrayCUDAPtr[species], departureArrayCUDAPtr[species], 
                                                          fillerBufferArrayCUDAPtr[species], hashedSumArrayCUDAPtr[species]+6, pclsArrayHostPtr[species]->getNOP()-x);

  
  cudaErrChk(cudaStreamSynchronize(streams[species+ns])); // exiting particle copied
  return x;
}

//! MAXWELL SOLVER for Bfield (assuming Efield has already been calculated)
void c_Solver::CalculateB() {
  timeTasks_set_main_task(TimeTasks::FIELDS);
  // calculate the B field
  EMf->calculateB();
}

/*  -------------- */
/*!  Particle mover */
/*  -------------- */
bool c_Solver::ParticlesMover()
{
  // move all species of particles
  {
    timeTasks_set_main_task(TimeTasks::PARTICLES);
    // Should change this to add background field
    EMf->set_fieldForPcls();

#if MOVER_CUDA_ON==false
    pad_particle_capacities();
    for (int i = 0; i < ns; i++)  // move each species
    {
      // #pragma omp task inout(part[i]) in(grid) target_device(booster)
      // should merely pass EMf->get_fieldForPcls() rather than EMf.
      // use the Predictor Corrector scheme to move particles
      switch(Parameters::get_MOVER_TYPE())
      {
        case Parameters::SoA:
          part[i].mover_PC(EMf);
          break;
        case Parameters::AoS:
          part[i].mover_PC_AoS(EMf);
          break;
        case Parameters::AoS_Relativistic:
        	part[i].mover_PC_AoS_Relativistic(EMf);
        	break;
        case Parameters::AoSintr:
          part[i].mover_PC_AoS_vec_intr(EMf);
          break;
        case Parameters::AoSvec:
          part[i].mover_PC_AoS_vec(EMf);
          break;
        default:
          unsupported_value_error(Parameters::get_MOVER_TYPE());
      }

	  //Should integrate BC into separate_and_send_particles
	  part[i].openbc_particles_outflow();
	  part[i].separate_and_send_particles();

    }
#else
    auto gridSize = grid->getNXN() * grid->getNYN() * grid->getNZN();
    //! copy fieldForPcls to device, for every species 
    cudaErrChk(cudaMemcpyAsync(fieldForPclCUDAPtr, (void*)&(EMf->get_fieldForPcls().get(0,0,0,0)), gridSize*8*sizeof(cudaCommonType), cudaMemcpyDefault, streams[0]));
    cudaErrChk(cudaStreamSynchronize(streams[0]));
    
    std::future<int> results[ns];
    for(int i=0; i<ns; i++){
      //part[i].get_pcl_array().clear(); // clear the host pclArray
      results[i] = std::async(&c_Solver::cudaLauncherAsync, this, i);
    }

    for (int i = 0; i < ns; i++){ //  it can be better
      auto x = results[i].get();
      stayedParticle[i] = pclsArrayHostPtr[i]->getNOP() - x;

      // part[i].openbc_particles_outflow();
      auto a = part[i].separate_and_send_particles();
    }
#endif

    for (int i = 0; i < ns; i++)  // communicate each species
    {
      part[i].recommunicate_particles_until_done(1);
    }
  }

  /* -------------------------------------- */
  /* Repopulate the buffer zone at the edge */
  /* -------------------------------------- */

  // for (int i=0; i < ns; i++) {
  //   if (col->getRHOinject(i)>0.0)
  //     part[i].repopulate_particles();
  // }

  /* --------------------------------------- */
  /* Remove particles from depopulation area */
  /* --------------------------------------- */
  // if (col->getCase()=="Dipole") {
  //   for (int i=0; i < ns; i++)
  //     Qremoved[i] = part[i].deleteParticlesInsideSphere(col->getL_square(),col->getx_center(),col->gety_center(),col->getz_center());
  // }else if (col->getCase()=="Dipole2D") {
	// for (int i=0; i < ns; i++)
	//   Qremoved[i] = part[i].deleteParticlesInsideSphere2DPlaneXZ(col->getL_square(),col->getx_center(),col->getz_center());
  // }

#if MOVER_CUDA_ON==true
  for(int i=0; i<ns; i++){

    // now the host array contains the entering particles
    if((part[i].getNOP() + stayedParticle[i]) >= pclsArrayHostPtr[i]->getSize()){ // not enough size, expand the device array size
      pclsArrayHostPtr[i]->expand((part[i].getNOP() + stayedParticle[i]) * 1.5, streams[i]);
      departureArrayHostPtr[i]->expand(pclsArrayHostPtr[i]->getSize(), streams[i]);
      cudaErrChk(cudaMemcpyAsync(departureArrayCUDAPtr[i], departureArrayHostPtr[i], sizeof(departureArrayType), cudaMemcpyDefault, streams[i]));
    }
    // now enough size on device pcls array, copy particles
    cudaErrChk(cudaMemcpyAsync(pclsArrayHostPtr[i]->getpcls() + stayedParticle[i], 
              (void*)&(part[i].get_pcl_list()[0]), 
              part[i].getNOP()*sizeof(SpeciesParticle),
              cudaMemcpyDefault, streams[i]));
    // update counter
    pclsArrayHostPtr[i]->setNOE(stayedParticle[i] + part[i].getNOP()); 
    // copy the new object to device, device has new copy of the object now
    cudaErrChk(cudaMemcpyAsync(pclsArrayCUDAPtr[i], pclsArrayHostPtr[i], sizeof(particleArrayCUDA), cudaMemcpyDefault, streams[i]));
    
    // moment for entering particle
    momentKernelNew<<<getGridSize(part[i].getNOP(), 128), 128, 0, streams[i] >>>
                      (momentParamCUDAPtr[i], grid3DCUDACUDAPtr, momentsCUDAPtr[i], stayedParticle[i]);

    // reset the hashedSum, no need for departureArray it will be cleared in Mover
    for(int j=0; j<8; j++)hashedSumArrayHostPtr[i][j].resetBucket();
    cudaErrChk(cudaMemcpyAsync(hashedSumArrayCUDAPtr[i], hashedSumArrayHostPtr[i], 8 * sizeof(hashedSum), cudaMemcpyDefault, streams[i]));

  }

  auto gridSize = grid->getNXN() * grid->getNYN() * grid->getNZN();
  
  for(int i=0; i<ns; i++){ // copy moments back to 10 densities
    cudaErrChk(cudaMemcpyAsync((void*)&(EMf->getRHOns().get(i,0,0,0)),  momentsCUDAPtr[i]+0*gridSize, gridSize*sizeof(cudaCommonType), cudaMemcpyDefault, streams[i]));
    cudaErrChk(cudaMemcpyAsync((void*)&(EMf->getJxs().get(i,0,0,0)),    momentsCUDAPtr[i]+1*gridSize, gridSize*sizeof(cudaCommonType), cudaMemcpyDefault, streams[i]));
    cudaErrChk(cudaMemcpyAsync((void*)&(EMf->getJys().get(i,0,0,0)),    momentsCUDAPtr[i]+2*gridSize, gridSize*sizeof(cudaCommonType), cudaMemcpyDefault, streams[i]));
    cudaErrChk(cudaMemcpyAsync((void*)&(EMf->getJzs().get(i,0,0,0)),    momentsCUDAPtr[i]+3*gridSize, gridSize*sizeof(cudaCommonType), cudaMemcpyDefault, streams[i]));
    cudaErrChk(cudaMemcpyAsync((void*)&(EMf->getpXXsn().get(i,0,0,0)),  momentsCUDAPtr[i]+4*gridSize, gridSize*sizeof(cudaCommonType), cudaMemcpyDefault, streams[i]));
    cudaErrChk(cudaMemcpyAsync((void*)&(EMf->getpXYsn().get(i,0,0,0)),  momentsCUDAPtr[i]+5*gridSize, gridSize*sizeof(cudaCommonType), cudaMemcpyDefault, streams[i]));
    cudaErrChk(cudaMemcpyAsync((void*)&(EMf->getpXZsn().get(i,0,0,0)),  momentsCUDAPtr[i]+6*gridSize, gridSize*sizeof(cudaCommonType), cudaMemcpyDefault, streams[i]));
    cudaErrChk(cudaMemcpyAsync((void*)&(EMf->getpYYsn().get(i,0,0,0)),  momentsCUDAPtr[i]+7*gridSize, gridSize*sizeof(cudaCommonType), cudaMemcpyDefault, streams[i]));
    cudaErrChk(cudaMemcpyAsync((void*)&(EMf->getpYZsn().get(i,0,0,0)),  momentsCUDAPtr[i]+8*gridSize, gridSize*sizeof(cudaCommonType), cudaMemcpyDefault, streams[i]));
    cudaErrChk(cudaMemcpyAsync((void*)&(EMf->getpZZsn().get(i,0,0,0)),  momentsCUDAPtr[i]+9*gridSize, gridSize*sizeof(cudaCommonType), cudaMemcpyDefault, streams[i]));
  }


#endif

  /* --------------------------------------- */
  /* Test Particles mover 					 */
  /* --------------------------------------- */
  for (int i = 0; i < nstestpart; i++)  // move each species
  {
	switch(Parameters::get_MOVER_TYPE())
	{
	  case Parameters::SoA:
		  testpart[i].mover_PC(EMf);
		break;
	  case Parameters::AoS:
		  testpart[i].mover_PC_AoS(EMf);
		break;
	  case Parameters::AoS_Relativistic:
		  testpart[i].mover_PC_AoS_Relativistic(EMf);
		break;
	  case Parameters::AoSintr:
		  testpart[i].mover_PC_AoS_vec_intr(EMf);
		break;
	  case Parameters::AoSvec:
		  testpart[i].mover_PC_AoS_vec(EMf);
		break;
	  default:
		unsupported_value_error(Parameters::get_MOVER_TYPE());
	}

	testpart[i].openbc_delete_testparticles();
	testpart[i].separate_and_send_particles();
  }

  for (int i = 0; i < nstestpart; i++)
  {
	  testpart[i].recommunicate_particles_until_done(1);
  }

  return (false);
}

void c_Solver::WriteOutput(int cycle) {

#ifdef USE_CATALYST
  Adaptor::CoProcess(col->getDt()*cycle, cycle, EMf);
#endif

  WriteConserved(cycle);
  WriteRestart(cycle);

  if(!Parameters::get_doWriteOutput())  return;


  if (col->getWriteMethod() == "nbcvtk"){//Non-blocking collective MPI-IO

	  if(!col->field_output_is_off() && (cycle%(col->getFieldOutputCycle()) == 0 || cycle == first_cycle) ){
		  if(!(col->getFieldOutputTag()).empty()){

			  if(fieldreqcounter>0){
			          
			          //MPI_Waitall(fieldreqcounter,&fieldreqArr[0],&fieldstsArr[0]);
				  for(int si=0;si< fieldreqcounter;si++){
				    int error_code = MPI_File_write_all_end(fieldfhArr[si],&fieldwritebuffer[si][0][0][0],&fieldstsArr[si]);//fieldstsArr[si].MPI_ERROR;
					  if (error_code != MPI_SUCCESS) {
						  char error_string[100];
						  int length_of_error_string, error_class;
						  MPI_Error_class(error_code, &error_class);
						  MPI_Error_string(error_class, error_string, &length_of_error_string);
						  dprintf("MPI_Waitall error at field output cycle %d  %d  %s\n",cycle, si, error_string);
					  }else{
						  MPI_File_close(&(fieldfhArr[si]));
					  }
				  }
			  }
			  fieldreqcounter = WriteFieldsVTKNonblk(grid, EMf, col, vct,cycle,fieldwritebuffer,fieldreqArr,fieldfhArr);
		  }

		  if(!(col->getMomentsOutputTag()).empty()){

			  if(momentreqcounter>0){
			    //MPI_Waitall(momentreqcounter,&momentreqArr[0],&momentstsArr[0]);
				  for(int si=0;si< momentreqcounter;si++){
				    int error_code = MPI_File_write_all_end(momentfhArr[si],&momentwritebuffer[si][0][0],&momentstsArr[si]);//momentstsArr[si].MPI_ERROR;
					  if (error_code != MPI_SUCCESS) {
						  char error_string[100];
						  int length_of_error_string, error_class;
						  MPI_Error_class(error_code, &error_class);
						  MPI_Error_string(error_class, error_string, &length_of_error_string);
						  dprintf("MPI_Waitall error at moments output cycle %d  %d %s\n",cycle, si, error_string);
					  }else{
						  MPI_File_close(&(momentfhArr[si]));
					  }
				  }
			  }
			  momentreqcounter = WriteMomentsVTKNonblk(grid, EMf, col, vct,cycle,momentwritebuffer,momentreqArr,momentfhArr);
		  }
	  }

	  //Particle information is still in hdf5
	  	WriteParticles(cycle);
	  //Test Particle information is still in hdf5
	    WriteTestParticles(cycle);

  }else if (col->getWriteMethod() == "pvtk"){//Blocking collective MPI-IO
	  if(!col->field_output_is_off() && (cycle%(col->getFieldOutputCycle()) == 0 || cycle == first_cycle) ){
		  if(!(col->getFieldOutputTag()).empty()){
			  //WriteFieldsVTK(grid, EMf, col, vct, col->getFieldOutputTag() ,cycle);//B + E + Je + Ji + rho
			  WriteFieldsVTK(grid, EMf, col, vct, col->getFieldOutputTag() ,cycle, fieldwritebuffer);//B + E + Je + Ji + rho
		  }
		  if(!(col->getMomentsOutputTag()).empty()){
			  WriteMomentsVTK(grid, EMf, col, vct, col->getMomentsOutputTag() ,cycle, momentwritebuffer);
		  }
	  }

	  //Particle information is still in hdf5
	  	WriteParticles(cycle);
	  //Test Particle information is still in hdf5
	    WriteTestParticles(cycle);

  }else{

		#ifdef NO_HDF5
			eprintf("The selected output option must be compiled with HDF5");

		#else
			if (col->getWriteMethod() == "H5hut"){

			  if (!col->field_output_is_off() && cycle%(col->getFieldOutputCycle())==0)
				WriteFieldsH5hut(ns, grid, EMf, col, vct, cycle);
			  if (!col->particle_output_is_off() && cycle%(col->getParticlesOutputCycle())==0)
				WritePartclH5hut(ns, grid, part, col, vct, cycle);

			}else if (col->getWriteMethod() == "phdf5"){

			  if (!col->field_output_is_off() && cycle%(col->getFieldOutputCycle())==0)
				WriteOutputParallel(grid, EMf, part, col, vct, cycle);

			  if (!col->particle_output_is_off() && cycle%(col->getParticlesOutputCycle())==0)
			  {
				if(MPIdata::get_rank()==0)
				  warning_printf("WriteParticlesParallel() is not yet implemented.");
			  }

			}else if (col->getWriteMethod() == "shdf5"){

					WriteFields(cycle);

					WriteParticles(cycle);

					WriteTestParticles(cycle);

			}else{
			  warning_printf(
				"Invalid output option. Options are: H5hut, phdf5, shdf5, pvtk");
			  invalid_value_error(col->getWriteMethod().c_str());
			}
		#endif
  	  }
}

void c_Solver::WriteRestart(int cycle)
{
#ifndef NO_HDF5
  if (restart_cycle>0 && cycle%restart_cycle==0){
	  convertParticlesToSynched();
	  fetch_outputWrapperFPP().append_restart(cycle);
  }
#endif
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
      Ke[is] = part[is].getKe();
      BulkEnergy[is] = EMf->getBulkEnergy(is);
      TOTenergy += Ke[is];
      momentum[is] = part[is].getP();
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
      double maxVel = part[is].getMaxVelocity();
      long long *VelocityDist = part[is].getVelocityDistribution(nDistributionBins, maxVel);
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

void c_Solver::WriteFields(int cycle) {

#ifndef NO_HDF5
  if(col->field_output_is_off())   return;

  if(cycle % (col->getFieldOutputCycle()) == 0 || cycle == first_cycle)
  {
	  if(!(col->getFieldOutputTag()).empty())
		  	  fetch_outputWrapperFPP().append_output((col->getFieldOutputTag()).c_str(), cycle);//E+B+Js
	  if(!(col->getMomentsOutputTag()).empty())
		  	  fetch_outputWrapperFPP().append_output((col->getMomentsOutputTag()).c_str(), cycle);//rhos+pressure
  }
#endif
}

void c_Solver::WriteParticles(int cycle)
{
#ifndef NO_HDF5
  if(col->particle_output_is_off() || cycle%(col->getParticlesOutputCycle())!=0) return;

  // this is a hack
  for (int i = 0; i < ns; i++)
    part[i].convertParticlesToSynched();

  fetch_outputWrapperFPP().append_output((col->getPclOutputTag()).c_str(), cycle, 0);//"position + velocity + q "
#endif
}

void c_Solver::WriteTestParticles(int cycle)
{
#ifndef NO_HDF5
  if(nstestpart == 0 || col->testparticle_output_is_off() || cycle%(col->getTestParticlesOutputCycle())!=0) return;

  // this is a hack
  for (int i = 0; i < nstestpart; i++)
    testpart[i].convertParticlesToSynched();

  fetch_outputWrapperFPP().append_output("testpartpos + testpartvel+ testparttag", cycle, 0); // + testpartcharge
#endif
}

// This needs to be separated into methods that save particles
// and methods that save field data
//
void c_Solver::Finalize() {
  if (col->getCallFinalize() && Parameters::get_doWriteOutput())
  {
    #ifndef NO_HDF5
    convertParticlesToSynched();
    fetch_outputWrapperFPP().append_restart((col->getNcycles() + first_cycle) - 1);
    #endif
  }

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
void c_Solver::convertParticlesToSynched()
{
  for (int i = 0; i < ns; i++)
    part[i].convertParticlesToSynched();

  for (int i = 0; i < nstestpart; i++)
    testpart[i].convertParticlesToSynched();
}


int c_Solver::LastCycle() {
    return (col->getNcycles() + first_cycle);
}
