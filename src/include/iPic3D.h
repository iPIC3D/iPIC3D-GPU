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

/***************************************************************************
  iPIC3D.cpp  -  Main file for 3D simulation
  -------------------
 ************************************************************************** */

#ifndef _IPIC3D_H_
#define _IPIC3D_H_

class Timing;

#ifndef NO_MPI
#include "mpi.h"
#endif
#include "ipicfwd.h"
#include "assert.h"
#include <string>
using std::string;

#include "cudaTypeDef.cuh"
#include "moverKernel.cuh"
#include "momentKernel.cuh"
#include "particleArrayCUDA.cuh"
#include "gridCUDA.cuh"
#include "cellSortBuffers.cuh"
#include "particleExchange.cuh"
#include "planetKernel.cuh"
#include "threadPool.hpp"
#include "ExosphereIonization.h"
#include "ParticleSoAHost.h"
#include "ParticleCommInjection.h"

#include <fstream>

class IOManager;   // modular I/O manager (see IOManager.h)

namespace iPic3D {
  class c_Solver;

  // Simulation case types — resolved once from the input-file string in Init()
  enum class CaseType {
    GEMnoPert,
    ForceFree,
    GEM,
    GEMDoubleHarris,
    BATSRUS,
    Dipole,
    Dipole2D,
    NullPoints,
    TaylorGreen,
    HumpPert,
    RandomCase,
    Default         // unknown string → default initialisation
  };
}

namespace dataAnalysis {
  class dataAnalysisPipelineImpl;
}

namespace iPic3D {

  class c_Solver {

  friend dataAnalysis::dataAnalysisPipelineImpl;

  public:
    ~c_Solver();
    c_Solver():
      col(0),
      vct(0),
      grid(0),
      EMf(0),
      particlesCommInj(nullptr),
      particlesHost(nullptr),
      ioManager(0),
      Ke(0),
      BulkEnergy(0),
      momentum(0),
      Qtot(0),
      Qremoved(0),
      my_clock(0)
    {}


    int Init(int argc, char **argv);
    int initCUDA();
    int deInitCUDA();
    
    void CalculateMoments();
    void CalculateField(int cycle);
    int cudaLauncherAsync(int species, bool doMomentsInLauncher);
    bool ParticlesMoverMomentAsync(int cycle);
    bool MoverAwaitAndPclExchange(int cycle);
    void processPlanetParticles();
    void injectExosphereParticles();
    void sortAllSpecies();
    void CalculateB(int cycle);
    void MomentsAwait();

    //
    // output methods
    //
    void writeParticleNum(int cycle);
    void WriteConserved(int cycle);
    void WriteVelocityDistribution(int cycle);
    void WriteVirtualSatelliteTraces();
    void outputCopyAsync(int cycle);
    void WriteOutput(int cycle);
    void Finalize();

    int FirstCycle() { return (first_cycle); }
    int get_myrank() { return (myrank); }
    int LastCycle();

  private:
    void pad_particle_capacities();
    void sortParticles();
    void copyMomentsD2H(int species, cudaStream_t stream);
    void registerMomentsPinnedMemory(int species);
    void unregisterMomentsPinnedMemory(int species);

  private:
    //static MPIdata * mpi;
    Collective    *col; // the input parameters
    VCtopology3D  *vct; // mpi topology 
    Grid3DCU      *grid; // 3d cartesion grid, local grid
    EMfields3D    *EMf; // 
    ParticleCommInjection **particlesCommInj; // MPI exchange + injection engine (per species)
    ParticleSoAHost **particlesHost; // lightweight SoA host mirror (no communicator)
    ParticleSoAHost **testpart;
    ExosphereIonization *exosphereIonization; // exosphere photoionization source (CPU sampling)
    int numSolarWindSpecies;                     // cached: col->getNumSolarWindSpecies()
    int numPlanetarySpecies;                     // cached: ns - numSolarWindSpecies
    std::vector<std::future<void>> exosphereTaskFutures; // persistent future buffer (avoids per-call allocation)
    double        *Ke; // kinetic energy of each species, the normal one, added up
    double        *BulkEnergy; // bulk kinetic energy of each species, consider the bulk motion
    double        *momentum; // an array of doubles, total momentum of all particle species
    double        *Qtot; // total charge per species (sum of particle weights)
    double        *Qremoved; // array of double, with species length, removed charges from the depopulation area
    Timing        *my_clock;
    std::ofstream pclNumCSV;

    IOManager     *ioManager; // modular I/O manager (owns backends)


    int cudaDeviceOnNode; // the device this rank should use
    cudaStream_t*       streams;
    cudaStream_t        planetStream;  // dedicated stream for planet BC processing

    std::future<int>* exitingResults;
    int* stayedParticle; // stayed particles for each species

	//! Host pointers of objects, to be copied to device, for management later
	  particleArrayCUDA**   pclsArrayHostPtr;       // array of pointer, point to objects on host
    departureArrayType**  departureArrayHostPtr;  // for every species
    hashedSum**           hashedSumArrayHostPtr;      // species * 8
    exitingArray**        exitingArrayHostPtr;        // species
    arrayCUDA<SpeciesParticle>** incomingStagingHostPtr;  // per-species AoS staging for H→D incoming particles
    fillerBuffer**        fillerBufferArrayHostPtr;   // species
    grid3DCUDA* 		      grid3DCUDAHostPtr;      // one grid, used in all specieses
    moverParameter**      moverParamHostPtr;		  // for every species
    momentParameter**     momentParamHostPtr;		  // for every species

    int* cellCountHostPtr;
    int* cellOffsetHostPtr;

    CellSorter* cellSorters;  // per-species GPU cell sorter (counting sort)

    int  sortingCycle_;    // cached from col->getSortingCycle() (0=disabled)
    bool sortThisCycle_;   // true when this cycle uses the sorted pipeline

    CaseType caseType_;  // resolved once in Init() from col->getCase()
    bool     doPlanet_;  // true when caseType_ is Dipole or Dipole2D
    
	//! CUDA pointers of objects, have been copied to device
    particleArrayCUDA**   pclsArrayCUDAPtr;           // array of pointer, point to pclsArray on device
    departureArrayType**  departureArrayCUDAPtr;      // for every species
    hashedSum**           hashedSumArrayCUDAPtr;      // species * 8
    exitingArray**        exitingArrayCUDAPtr;        // species
    arrayCUDA<SpeciesParticle>** incomingStagingCUDAPtr;  // per-species device copy of staging metadata
    fillerBuffer**        fillerBufferArrayCUDAPtr;   // species
    grid3DCUDA* 		      grid3DCUDACUDAPtr;    	    // one grid, used in all specieses
    moverParameter**      moverParamCUDAPtr;		      // for every species
    momentParameter**     momentParamCUDAPtr;		      // for every species

    int* cellCountCUDAPtr;
    int* cellOffsetCUDAPtr;

	//! simple device buffers
    // [10][nxn][nyn][nzn], a piece of cuda memory to hold the moment
    cudaTypeArray1<cudaMomentType>* momentsCUDAPtr; // for every species
    // [nxn][nyn][nzn][2*4], a piece of cuda memory to hold E and B from host
    cudaTypeArray1<cudaFieldType> fieldForPclCUDAPtr; // for all species

    cudaTypeArray1<cudaFieldType> fieldForPclHostPtr;
    
    ThreadPool *threadPoolPtr;

    cudaEvent_t event0, eventOutputCopy;

    //bool verbose;
    string SaveDirName;
    string RestartDirName;
    string cqsat;
    string cq;
    string ds;
    string num_proc_str;
    int restart_cycle;
    int restart_status;
    int first_cycle;
    int ns;
    int nstestpart;
    int nprocs;
    int myrank;
    int nsat;
    int nDistributionBins;
    double Eenergy;
    double Benergy;
    double TOTenergy;
    double TOTmomentum;
    int mergeIdx = -1;
    int* toBeMerged;

    //! Planet quasi-neutral BC data structures
    planetArray**       planetArrayHostPtr;      // per species, host-pinned metadata
    planetArray**       planetArrayCUDAPtr;      // per species, device metadata
    int*                planetPclCount;           // per species, planet particle count this cycle

    // Cross-species planet processing buffers (device)
    cudaParticleType*   planetEnergyBuf;          // merged electron energies
    uint32_t*           planetGlobalIdxBuf;       // encodes species + local index
    cudaParticleType*   planetIonChargeDevice;    // single-element device buffer for reduction output
    int*                planetCutoffDevice;       // single-element device buffer for cutoff index output
    planetArray**       planetArrayCUDAPtrDevice; // device array of device pointers (for chargeCutoffKernel)
    int*                planetElecOffsetsDevice;  // device array of per-electron-species offsets
    int                 planetBufCapacity;        // current allocation size of merged buffers
    int                 planetElecSpeciesCount;   // number of electron species
    int*                planetElecSpeciesMap;     // maps electron index [0..nElec-1] → species index

    // Persistent host buffers for processPlanetParticles (avoid per-call allocation)
    int*                planetElecOffsets;         // [planetElecSpeciesCount] species offsets into merged buffers
    planetArray**       planetTmpPtrs;             // [planetElecSpeciesCount] temp pointer array
    int*                planetSurvivorCount;       // [planetElecSpeciesCount] survivor counts per electron species (host)
    int*                planetSurvivorCountDevice; // [planetElecSpeciesCount] per-species atomic counters (device)
    SpeciesParticle*    planetReflectedBuf;        // device buffer for compact reflected particles
    int                 planetReflectedBufCapacity;// capacity (in particles) of planetReflectedBuf
    uint32_t            planetRngCycleCounter;     // incremented each cycle for diffuse-scatter RNG seed

  };

}

#endif
