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

  /**
   * @brief Simulation case identifiers resolved once during solver initialization.
   */
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
    GEMHarris,
    Default         // unknown string → default initialisation
  };
}

namespace dataAnalysis {
  class dataAnalysisPipelineImpl;
}

namespace iPic3D {

  /**
   * @brief Top-level iPIC3D solver orchestrator.
   *
   * The solver owns MPI/topology state, fields, particle containers, CUDA
   * runtime objects, and output backends. Runtime control flows through the
   * methods declared here and implemented in `iPIC3Dlib.cu`.
   */
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
      testpart(nullptr),
      exosphereIonization(nullptr),
      ioManager(0),
      Ke(0),
      BulkEnergy(0),
      momentum(0),
      Qtot(0),
      Qremoved(0),
      my_clock(0)
    {}

    // ======= Initialization and lifecycle =======
    /**
     * @brief Initialize the full solver state from input arguments.
     *
     * @param argc Command-line argument count.
     * @param argv Command-line argument vector.
     * @return Zero on success; non-zero on failure.
     */
    int Init(int argc, char **argv);
    /**
     * @brief Allocate and initialize all CUDA-side solver resources.
     *
     * @return Zero on success; non-zero on failure.
     */
    int initCUDA();
    /**
     * @brief Release all CUDA-side solver resources.
     *
     * @return Zero on success; non-zero on failure.
     */
    int deInitCUDA();

    // ======= Particle and field advance =======
    void CalculateMoments();
    /**
     * @brief Advance the electric field solve for one cycle.
     *
     * @param cycle Current simulation cycle.
     */
    void CalculateField(int cycle);
    /**
     * @brief Launch the GPU mover pipeline for one species.
     *
     * @param species Species index.
     * @param doMomentsInLauncher True when moments are deposited in the same async path.
     * @return Status code from the launch path.
     */
    int cudaLauncherAsync(int species, bool doMomentsInLauncher);
    /**
     * @brief Launch mover and moment work for all species asynchronously.
     *
     * @param cycle Current simulation cycle.
     * @return True when the mover pipeline completed successfully.
     */
    bool ParticlesMoverMomentAsync(int cycle);
    /**
     * @brief Wait for mover completion, exchange particles, and handle injections.
     *
     * @param cycle Current simulation cycle.
     * @return True when the exchange path completed successfully.
     */
    bool MoverAwaitAndPclExchange(int cycle);
    void processPlanetParticles();
    void injectExosphereParticles();
    void sortAllSpecies();
    /**
     * @brief Advance the magnetic field for one cycle.
     *
     * @param cycle Current simulation cycle.
     */
    void CalculateB(int cycle);
    void MomentsAwait();

    // ======= Output and diagnostics =======
    /**
     * @brief Write per-species particle counts for one cycle.
     *
     * @param cycle Current simulation cycle.
     */
    void writeParticleNum(int cycle);
    /**
     * @brief Write conserved-quantity diagnostics for one cycle.
     *
     * @param cycle Current simulation cycle.
     */
    void WriteConserved(int cycle);
    /**
     * @brief Write the velocity-distribution diagnostic for one cycle.
     *
     * @param cycle Current simulation cycle.
     */
    void WriteVelocityDistribution(int cycle);
    void WriteVirtualSatelliteTraces();
    /**
     * @brief Schedule or execute GPU-to-host copies needed for output.
     *
     * @param cycle Current simulation cycle, or -1 for the final flush path.
     */
    void outputCopyAsync(int cycle);
    /**
     * @brief Write configured outputs for one cycle.
     *
     * @param cycle Current simulation cycle.
     */
    void WriteOutput(int cycle);
    void Finalize();

    int FirstCycle() { return (first_cycle); }
    int get_myrank() { return (myrank); }
    int LastCycle();

  private:
    void pad_particle_capacities();
    void sortParticles();
    /**
     * @brief Copy one species' moment buffer from device to host.
     *
     * @param species Species index.
     * @param stream CUDA stream used for the asynchronous copy.
     */
    void copyMomentsD2H(int species, cudaStream_t stream);
    /**
     * @brief Register one species' host moment arrays as pinned memory.
     *
     * @param species Species index.
     */
    void registerMomentsPinnedMemory(int species);
    /**
     * @brief Unregister one species' host moment arrays from pinned memory.
     *
     * @param species Species index.
     */
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

    // ======= Host-side metadata objects mirrored to the device =======
	  particleArrayCUDA**   pclsArrayHostPtr;       // array of pointers to host-resident metadata objects
    departureArrayType**  departureArrayHostPtr;  // for every species
    hashedSum**           hashedSumArrayHostPtr;      // species * 8
    exitingArray**        exitingArrayHostPtr;        // species
    arrayCUDA<SpeciesParticle>** incomingStagingHostPtr;  // per-species AoS staging for H→D incoming particles
    fillerBuffer**        fillerBufferArrayHostPtr;   // species
    grid3DCUDA* 		      grid3DCUDAHostPtr;      // one shared grid descriptor for all species
    moverParameter**      moverParamHostPtr;		  // for every species
    momentParameter**     momentParamHostPtr;		  // for every species
    injectionParameter**  injectionParamHostPtr;  // for every species (GPU injection)

    CellSorter* cellSorters;  // per-species GPU cell sorter (counting sort)

    int  sortingCycle_;    // cached from col->getSortingCycle() (0=disabled)
    bool sortThisCycle_;   // true when this cycle uses the sorted pipeline

    CaseType caseType_;  // resolved once in Init() from col->getCase()
    bool     doPlanet_;  // true when caseType_ is Dipole or Dipole2D
    
    // ======= Device-side metadata objects =======
    particleArrayCUDA**   pclsArrayCUDAPtr;           // array of pointers to device-resident particle metadata
    departureArrayType**  departureArrayCUDAPtr;      // for every species
    hashedSum**           hashedSumArrayCUDAPtr;      // species * 8
    exitingArray**        exitingArrayCUDAPtr;        // species
    arrayCUDA<SpeciesParticle>** incomingStagingCUDAPtr;  // per-species device copy of staging metadata
    fillerBuffer**        fillerBufferArrayCUDAPtr;   // species
    grid3DCUDA* 		      grid3DCUDACUDAPtr;    	    // one shared grid descriptor for all species
    moverParameter**      moverParamCUDAPtr;		      // for every species
    momentParameter**     momentParamCUDAPtr;		      // for every species
    injectionParameter**  injectionParamCUDAPtr;      // for every species (GPU injection)

    // ======= Shared device buffers =======
    // [10][nxn][nyn][nzn] packed moment storage per species.
    cudaTypeArray1<cudaMomentType>* momentsCUDAPtr; // for every species
    // Packed field-interpolation buffer copied from host for mover kernels.
    cudaTypeArray1<cudaFieldType> fieldForPclCUDAPtr; // for all species

    cudaTypeArray1<cudaFieldType> fieldForPclHostPtr;
    
    ThreadPool *threadPoolPtr;

    cudaEvent_t event0, eventOutputCopy;
    // True once outputCopyAsync() has actually scheduled a D->H copy and
    // recorded eventOutputCopy. Used by WriteOutput() to guard against
    // synchronizing on a never-recorded event on the very first cycle, and to
    // know whether the host SoA mirrors hold copy-back data or just the
    // initial / restart-loaded particle state.
    bool outputCopyEverRecorded_ = false;

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

    // ======= Planet quasi-neutral boundary-condition state =======
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
