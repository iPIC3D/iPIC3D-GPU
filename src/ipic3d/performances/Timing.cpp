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

#include "Timing.h"
#include "MPIdata.h"
#include "ipicdefs.h"
#include "stdio.h"
#include <chrono>
#include <cmath>
#include <mpi.h>

double Timing::tProgStart = 0.0;
double Timing::tInitEnd = 0.0;
double Timing::tLoopStart = 0.0;
double Timing::tLoopEnd = 0.0;
double Timing::cycleMsSum = 0.0;
double Timing::cycleMsSumSq = 0.0;
int Timing::nCycles = 0;

void Timing::markProgramStart() { tProgStart = MPI_Wtime(); }
void Timing::markInitEnd() { tInitEnd = MPI_Wtime(); }
void Timing::markLoopStart() { tLoopStart = MPI_Wtime(); }
void Timing::markLoopEnd() { tLoopEnd = MPI_Wtime(); }

void Timing::recordCycle(int rank, TimePoint start, TimePoint sort,
                         TimePoint field, TimePoint daWait, TimePoint mover,
                         TimePoint exchange, TimePoint bField,
                         TimePoint moments, TimePoint end) {
  const auto ms = [](TimePoint a, TimePoint b) {
    return std::chrono::duration<double, std::milli>(b - a).count();
  };

  const double totalMs = ms(start, end);
  cycleMsSum += totalMs;
  cycleMsSumSq += totalMs * totalMs;
  nCycles++;

  if (rank == 0) {
    printf("Execution time cycle: %g ms  [sort=%g field=%g da_wait=%g "
           "mover+out=%g exchange+sort+moments=%g B=%g momentsAwait=%g "
           "outCopy=%g]\n",
           totalMs, ms(start, sort), ms(sort, field), ms(field, daWait),
           ms(daWait, mover), ms(mover, exchange), ms(exchange, bField),
           ms(bField, moments), ms(moments, end));
  }
}
/**
 *
 * series of methods for timing and profiling
 * @date Fri Jun 4 2007
 * @author Stefano Markidis, Giovanni Lapenta
 * @version 2.0
 *
 */

/** default constructor */
Timing::Timing() {}

/** constructor with the initialization of the log file */
Timing::Timing(int my_rank) {
  rank_id = my_rank;
  // initialize the logger
  // MPE_Init_log();
  // start the timer
  startTiming();
  // get event ID, from MPE
  // event1a = MPE_Log_get_event_number();
  // event1b = MPE_Log_get_event_number();
  // event2a = MPE_Log_get_event_number();
  // event2b = MPE_Log_get_event_number();
  // event3a = MPE_Log_get_event_number();
  // event3b = MPE_Log_get_event_number();
  // described the events
  // if (my_rank==0){
  // MPE_Describe_state(event1a,event1b,"Mover","red"); // the mover is red in
  // the visualizer MPE_Describe_state(event2a,event2b,"Field","blue"); // the
  // mover is blue in the visualizer MPE_Describe_state(event3a,event3b,"Interp
  // P->G","yellow"); // the interpolation particle->Grid is yellow in the
  // visualizer
  // }
  former_MPI_Barrier(MPIdata::get_PicGlobalComm());
  // start the log
  // MPE_Start_log();
}

/** start the timer */
void Timing::startTiming() {
  ttick = MPI_Wtick();
  former_MPI_Barrier(MPIdata::get_PicGlobalComm());
  tstart = MPI_Wtime();
}
/** stop the timer */
void Timing::stopTiming() {
  former_MPI_Barrier(MPIdata::get_PicGlobalComm());
  tend = MPI_Wtime();
  texecution = tend - tstart;
  if (rank_id == 0) {
    // replace %g with %11.3e?
    printf("\n\n*** SIMULATION ENDED SUCESSFULLY ***\n");
    // Phase breakdown, available when main() set the program-phase marks.
    // Note: stopTiming() runs at the tail of Finalize(), so tend also closes
    // the finalize span.
    if (tProgStart > 0.0 && tInitEnd > 0.0 && tLoopStart > 0.0 &&
        tLoopEnd > 0.0) {
      const double totalSec = tend - tProgStart;
      const double avgMs = nCycles > 0 ? cycleMsSum / nCycles : 0.0;
      double stddevMs = 0.0;
      if (nCycles > 1) {
        const double var =
            (cycleMsSumSq - cycleMsSum * cycleMsSum / nCycles) / (nCycles - 1);
        stddevMs = var > 0.0 ? std::sqrt(var) : 0.0;
      }
      printf(" Total execution    : %g sec (%g hours)\n"
             " Initialization     : %g sec\n"
             " Pre-cycle setup    : %g sec (DA pipeline + initial moments)\n"
             " Cycle loop         : %g sec over %d cycles\n"
             "                      avg = %g ms/cycle, stddev = %g ms\n"
             " Finalize           : %g sec\n",
             totalSec, totalSec / 3600, tInitEnd - tProgStart,
             tLoopStart - tInitEnd, tLoopEnd - tLoopStart, nCycles, avgMs,
             stddevMs, tend - tLoopEnd);
    }
    printf(" Simulation Time    : %g sec (%g hours) [end of Init -> end of "
           "Finalize, i.e. pre-cycle setup + cycle loop + finalize]\n",
           texecution, texecution / 3600);
    printf("***\n\n");
  }
  // close the log file
  // MPE_Finish_log("iPIC3D_LOG");
}

/** start timing the mover */
void Timing::start_mover() {
  // MPE_Log_event(event1a,0,"start mover");
}
/** stop timing the mover */
void Timing::stop_mover() {
  // MPE_Log_event(event1b,0,"end mover");
}
/** start timing the field solver */
void Timing::start_field() {
  // MPE_Log_event(event2a,0,"start Field solver");
}
/** stop timing the field solver */
void Timing::stop_field() {
  // MPE_Log_event(event2b,0,"stop Field solver");
}
/** start timing the interpolation Particle -> Grid */
void Timing::start_interpP2G() {
  // MPE_Log_event(event3a,0,"start interpolation");
}
/** stop timing the interpolation Particle -> Grid*/
void Timing::stop_interpP2G() {
  // MPE_Log_event(event3b,0,"stop interpolation");
}
/** get the elapsed time from start_timng and stop_timing */
double Timing::getExecutionTime() { return (texecution); }
/** print to screen the elapsed time */
void Timing::Print() {
  printf("Execution Time: %g sec (%g hours)\n", texecution, texecution / 3600);
  // cout << "Execution Time: " << texecution << " sec" << " (" << texecution /
  // 3600 << " hours)" << endl;
}
/** print to screen the elapsed time from t_start to the call to print
 * function*/
void Timing::Print_OnAir() {
  printf("Execution Time: %g sec\n", MPI_Wtime() - tstart);
  // cout << "Execution Time: " << MPI_Wtime() - tstart << " sec" << endl;
}
