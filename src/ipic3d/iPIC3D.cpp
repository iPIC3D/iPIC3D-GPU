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

#include "iPic3D.h"
#include "BenchmarkMode.h"
#include "MPIdata.h"
#include "Profiler.h"
#include "Timing.h"
#include "debug.h"
#include <stdio.h>

#if IPIC3D_COMPILE_DIAGNOSTIC_CALCULATIONS
#include "dataAnalysis.cuh"
#endif

using namespace iPic3D;

#if IPIC3D_COMPILE_DIAGNOSTIC_CALCULATIONS
using Analysis = dataAnalysis::dataAnalysisPipeline;
#else
/** The data analysis when it is not compiled (BENCHMARK_MODE 2). */
struct Analysis {};
#endif

/**
 * One cycle of the time loop: its stages in turn, each a profiler region that
 * the cycle timer opens and times, from its start to the start of the next
 * (Timing::stage).
 */
static void runCycle(c_Solver& KCode, Analysis& DA, Timing& timing, int i) {
  using profiling::Region;
  if (KCode.get_myrank() == 0)
    printf(" ======= Cycle %d ======= \n", i);
  timing.startCycle(i);

#if IPIC3D_COMPILE_DISK_OUTPUT
  timing.stage(Region::ParticleCountLog);
  KCode.writeParticleNum(i);
#endif
#if IPIC3D_COMPILE_DIAGNOSTIC_CALCULATIONS
  // The macrocell analyses need the particles sorted by cell (the spectra
  // B packed too)
  timing.stage(Region::AnalysisPrepare);
  DA.prepareAnalysis(i);
  timing.stage(Region::HeatFlux);
  KCode.ScheduleHeatFlux(i);
  // DA analysis runs on its own host thread and GPU stream, of the lowest
  // priority, while CalculateField runs the field solve on the solver
  // stream, of high priority. The mover, which changes the particles the
  // analysis reads, starts after DA.waitForAnalysis().
  timing.stage(Region::AnalysisStart);
  DA.startAnalysis(i);
#else
  (void)DA;
#endif

  timing.stage(Region::FieldSolveE);
  KCode.CalculateField(i);

#if IPIC3D_COMPILE_DIAGNOSTIC_CALCULATIONS
  timing.stage(Region::AnalysisWait);
  DA.waitForAnalysis();
#endif

  // Launch the mover kernels (moments: in the launcher, or after the sort
  // on sorting cycles).
  timing.stage(Region::MoverLaunch);
  KCode.ParticlesMoverMomentAsync(i);

#if IPIC3D_COMPILE_DISK_OUTPUT
  // Use otherwise-idle CPU time for output while the GPU mover runs.
  timing.stage(Region::Output);
  KCode.WriteOutput(i);
#endif

  timing.stage(Region::ParticleExchange);
  KCode.MoverAwaitAndPclExchange(i); // includes sort + moments

  timing.stage(Region::FieldSolveB);
  KCode.CalculateB(i);

  timing.stage(Region::MomentsAwait);
  KCode.MomentsAwait(i);

#if IPIC3D_COMPILE_DISK_OUTPUT
  // There is no future cycle after the final loop iteration.  Avoid
  // predicting and copying output for a cycle that will never be written.
  if (i + 1 < KCode.LastCycle()) {
    timing.stage(Region::OutputCopy);
    KCode.outputCopyAsync(i); // copy output data for the next cycle
  }
#endif

  timing.endCycle();
}

/**
 * The time loop with the data analysis it runs: the analysis set-up, the
 * initial moments, the cycles and, on return, the release of the analysis
 * (its thread, GPU buffers and record files), before the finalisation.
 */
static void runLoop(c_Solver& KCode, Timing& timing) {
#if IPIC3D_COMPILE_DIAGNOSTIC_CALCULATIONS
  Analysis DA(KCode); // has to be created after KCode.Init()
#else
  Analysis DA;
#endif

  KCode.CalculateMoments();
  timing.markLoopStart();

  // The steady state: the cycles after the first (warm-up) one, which the
  // profiler records apart (CrayPat records only them)
  const int firstCycle = KCode.FirstCycle();
  const int lastCycle = KCode.LastCycle();
  for (int i = firstCycle; i < lastCycle; i++) {
    runCycle(KCode, DA, timing, i);
    if (i == firstCycle && i + 1 < lastCycle)
      timing.startSteadyState();
  }
  timing.endLoop();
}

static int runSimulation(int argc, char** argv) {
  Timing timing;
  timing.markProgramStart();

  iPic3D::c_Solver KCode;
  //! load param from file, init the grid, fields
  const int initStatus = KCode.Init(argc, argv);
  if (initStatus != 0)
    return initStatus;
  timing.markInitEnd();

  runLoop(KCode, timing);

  KCode.Finalize();
  timing.markEnd();
  timing.printSummary();
  profiling::reportStageTimers(timing.steadyCycles());
  return 0;
}

int main(int argc, char** argv) {
  // CrayPat records only the steady-state cycles (runSimulation)
  profiling::initRecording();
  MPIdata::init(&argc, &argv);
  const int status = runSimulation(argc, argv);

  // close MPI
  MPIdata::instance().finalize_mpi();

  return status;
}
