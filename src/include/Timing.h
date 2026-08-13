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

/*******************************************************************************************
  Timing.h  -  series of methods for timing and profiling PARSEK
  -------------------
developers: Stefano Markidis, Enrico Camporeale, Giovanni Lapenta, David Burgess
 ********************************************************************************************/

#ifndef TIMING_H
#define TIMING_H

#include <chrono>

// #include "MPIdata.h"

// #include <iostream>

// using std::cout;
// using std::endl;
/**
 *
 * series of methods for timing and profiling PARSEK
 * @date Fri Jun 4 2007
 * @author Stefano Markidis, Giovanni Lapenta
 * @version 2.0
 *
 */
class Timing {
public:
  /** default constructor */
  Timing();
  /** default constructor */
  Timing(int my_rank);
  /** start timing the mover */
  void start_mover();
  /** stop timing the mover */
  void stop_mover();
  /** start timing the field solver */
  void start_field();
  /** stop timing the field solver */
  void stop_field();
  /** start timing the interpolation Particle -> Grid */
  void start_interpP2G();
  /** stop timing the interpolation Particle -> Grid */
  void stop_interpP2G();
  /** start the timer */
  void startTiming();
  /** stop the timer */
  void stopTiming();
  /** get the execution time */
  double getExecutionTime();
  /** print the execution time */
  void Print();
  /** print the execution time without stopping the clock*/
  void Print_OnAir();

  // Program-phase marks (wall clock), set from main(). stopTiming() uses
  // them to print the timing breakdown alongside the simulation time.
  /** mark program start (right after MPI init) */
  static void markProgramStart();
  /** mark end of solver Init() */
  static void markInitEnd();
  /** mark start of the cycle loop (after initial moments) */
  static void markLoopStart();
  /** per-cycle wall-clock marks captured in the main loop */
  using TimePoint = std::chrono::high_resolution_clock::time_point;
  /** accumulate one cycle's stats for avg/stddev and print the rank-0
   * per-cycle breakdown */
  static void recordCycle(int rank, TimePoint start, TimePoint sort,
                          TimePoint field, TimePoint daWait, TimePoint mover,
                          TimePoint exchange, TimePoint bField,
                          TimePoint moments, TimePoint end);
  /** mark end of the cycle loop (before Finalize()) */
  static void markLoopEnd();

private:
  /** rank of the processor */
  int rank_id;
  /** wall-clock time when startTiming method is called */
  double tstart;
  /** wall-clock time when endTiming method is called */
  double tend;
  /** execution time */
  double texecution;
  /** time precision */
  double ttick;
  /** events to write the logging: particle mover, field solver. it can be
   * extended with other events */
  int event1a, event1b, event2a, event2b, event3a, event3b, event4a, event4b;

  /** program-phase marks (0 = never set) and per-cycle statistics */
  static double tProgStart;
  static double tInitEnd;
  static double tLoopStart;
  static double tLoopEnd;
  static double cycleMsSum;
  static double cycleMsSumSq;
  static int nCycles;
};

#endif // TIMING_H
