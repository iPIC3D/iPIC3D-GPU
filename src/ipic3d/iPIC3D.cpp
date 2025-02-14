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

#include <ipic3d_cali.h>

#include "MPIdata.h"
#include "TimeTasks.h"
#include "debug.h"
#include "iPic3D.h"
#include <stdio.h>

#include "dataAnalysis.cuh"

using namespace iPic3D;

int main(int argc, char **argv) {

  MPIdata::init(&argc, &argv);
  {
#if CUDA_ON == true
    if (MPIdata::get_rank() == 0)
      std::cout << "The Software was built for GPU" << std::endl;
#endif

    iPic3D::c_Solver KCode;
    CALI_MARK_BEGIN("main");
    KCode.Init(argc, argv); //! load param from file, init the grid, fields

    timeTasks.resetCycle(); // reset timer
    KCode.CalculateMoments(true);
    CALI_MARK_LOOP_BEGIN(ipic3d_main_loop, "ipic3d_main_loop");
    for (int i = KCode.FirstCycle(); i < KCode.LastCycle(); i++) {
      CALI_MARK_ITERATION_BEGIN(ipic3d_main_loop, i);

      if (KCode.get_myrank() == 0)
        printf(" ======= Cycle %d ======= \n", i);

      timeTasks.resetCycle();

      CALI_MARK_BEGIN("em_field_and_gmm");
      auto analysisFuture = dataAnalysis::startAnalysis(KCode, i);

      KCode.CalculateField(i); // E field

      dataAnalysis::waitForAnalysis(analysisFuture);
      CALI_MARK_END("em_field_and_gmm");

      KCode.ParticlesMover(); // use the fields to calculate the new v and x for
                              // particles
      KCode.CalculateB();     // B field
      KCode.CalculateMoments(
          false); // the charge intense, current intense and pressure tensor,
      // calculated from particles position and celocity, then mapped to
      // node(grid) for further solving
      //  some are mapped to cell center

      // KCode.WriteOutput(i);
      // print out total time for all tasks
#ifdef LOG_TASKS_TOTAL_TIME
      timeTasks.print_cycle_times(i);
#endif
      CALI_MARK_ITERATION_END(ipic3d_main_loop);
    }
    CALI_MARK_LOOP_END(ipic3d_main_loop);

#ifdef LOG_TASKS_TOTAL_TIME
    timeTasks.print_tasks_total_times();
#endif

    KCode.Finalize();
    CALI_MARK_END("main");
  }
  // close MPI
  MPIdata::instance().finalize_mpi();

  return 0;
}
