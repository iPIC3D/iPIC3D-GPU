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


#include "MPIdata.h"
#include "iPic3D.h"
#include "debug.h"
#include "TimeTasks.h"
#include <stdio.h>
#include <chrono>

#include "dataAnalysis.cuh"

using namespace iPic3D;

int main(int argc, char **argv) {

 MPIdata::init(&argc, &argv);
 {

  iPic3D::c_Solver KCode;
  KCode.Init(argc, argv); //! load param from file, init the grid, fields
  dataAnalysis::dataAnalysisPipeline DA(KCode); // has to be created after KCode.Init()


  timeTasks.resetCycle(); //reset timer
  KCode.CalculateMoments();
  for (int i = KCode.FirstCycle(); i < KCode.LastCycle(); i++) {

    if (KCode.get_myrank() == 0)
      printf(" ======= Cycle %d ======= \n",i);
    
    auto start = std::chrono::high_resolution_clock::now();
    timeTasks.resetCycle();

    KCode.writeParticleNum(i);

    // Sort all species on GPU before data-analysis cycles
    if (dataAnalysis::dataAnalysisPipeline::isAnalysisCycle(i)) {
      KCode.sortAllSpecies();
    }
    auto t_sort = std::chrono::high_resolution_clock::now();

    // DA analysis runs async on GPU while CalculateField runs on CPU.
    // t_field is sampled *before* DA.waitForAnalysis() so that field= reflects
    // only the field solver and not particle-count-dependent analysis wait time.
    DA.startAnalysis(i);
    KCode.CalculateField(i); // E field
    auto t_field = std::chrono::high_resolution_clock::now();
    DA.waitForAnalysis();
    auto t_da_wait = std::chrono::high_resolution_clock::now();

    KCode.ParticlesMoverMomentAsync(i); // launch Mover kernels (moments computed post-sort)
    // some spare CPU cycles
    KCode.WriteOutput(i);
    auto t_mover = std::chrono::high_resolution_clock::now();

    KCode.MoverAwaitAndPclExchange(i); // includes sort + moments
    auto t_exchange = std::chrono::high_resolution_clock::now();

    KCode.CalculateB(i); 
    auto t_bfield = std::chrono::high_resolution_clock::now();

    KCode.MomentsAwait(); 
    auto t_moments = std::chrono::high_resolution_clock::now();

    KCode.outputCopyAsync(i); // copy output data to host, for next output
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsed = end - start;
    if (KCode.get_myrank() == 0) {
      std::cout<< "Execution time cycle: "<< elapsed.count() << " ms"
               << "  [sort=" << std::chrono::duration<double, std::milli>(t_sort - start).count()
               << " field=" << std::chrono::duration<double, std::milli>(t_field - t_sort).count()
               << " da_wait=" << std::chrono::duration<double, std::milli>(t_da_wait - t_field).count()
               << " mover+out=" << std::chrono::duration<double, std::milli>(t_mover - t_da_wait).count()
               << " exchange+sort+moments=" << std::chrono::duration<double, std::milli>(t_exchange - t_mover).count()
               << " B=" << std::chrono::duration<double, std::milli>(t_bfield - t_exchange).count()
               << " momentsAwait=" << std::chrono::duration<double, std::milli>(t_moments - t_bfield).count()
               << " outCopy=" << std::chrono::duration<double, std::milli>(end - t_moments).count()
               << "]" << std::endl;
    }

#ifdef LOG_TASKS_TOTAL_TIME
    timeTasks.print_cycle_times(i); // print out total time for all tasks
#endif
  }

#ifdef LOG_TASKS_TOTAL_TIME
    timeTasks.print_tasks_total_times();
#endif

  KCode.Finalize();
 }
 // close MPI
 MPIdata::instance().finalize_mpi();

 return 0;
}
