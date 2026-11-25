#ifndef TIMING_H
#define TIMING_H

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

/**
 * @file Timing.h
 * @brief The run timer, always on: the phases of the program, the time of
 * every cycle and of its stages (the `Execution time cycle` line of rank 0),
 * the steady state, and the summary at the end of the run.
 *
 * Every time is host wall time on a monotonic clock (std::chrono::
 * steady_clock). The timer opens the profiler regions of the time loop
 * (Profiler.h): each cycle, each of its stages, and the steady state, which
 * is the profiler's recording window. The steady state is the cycles after
 * the first (warm-up) cycle; it starts and ends at a timing boundary
 * (boundary(): the GPU drained, then all ranks met), so its interval holds
 * all of its GPU work and is the same on every rank.
 */

#include "Profiler.h"

#include <chrono>

/** The stages of a cycle: the profiler regions from kFirstCycleStage to
 *  kLastCycleStage, in loop order (Profiler.h). */
inline constexpr profiling::Region kFirstCycleStage =
    profiling::Region::ParticleCountLog;
inline constexpr profiling::Region kLastCycleStage =
    profiling::Region::OutputCopy;
/** Number of cycle stages. */
inline constexpr int kCycleStages =
    static_cast<int>(kLastCycleStage) - static_cast<int>(kFirstCycleStage) + 1;

class Timing {
public:
  using Clock = std::chrono::steady_clock;

  void markProgramStart() { programStart_ = Clock::now(); }
  void markInitEnd() { initEnd_ = Clock::now(); }
  void markLoopStart() { loopStart_ = Clock::now(); }
  void markEnd() { end_ = Clock::now(); }

  /** Start a cycle: the Cycle region, indexed by its number. */
  void startCycle(int cycle);
  /** Start the given stage of the cycle, a region from kFirstCycleStage to
   *  kLastCycleStage: it ends the previous stage and lasts until the next one
   *  or the end of the cycle. */
  void stage(profiling::Region region);
  /** End the cycle: rank 0 prints its `Execution time cycle` line. */
  void endCycle();

  /** Start the steady state, after the first cycle: a boundary, then the
   *  profiler's recording window and the SteadyState region (collective). */
  void startSteadyState();
  /** End the time loop: a boundary, then the end of the steady state if it
   *  started (collective). */
  void endLoop();

  /** Number of steady-state cycles completed. */
  int steadyCycles() const { return steadyCycles_; }

  /** The summary of the run, printed by rank 0 (collective). */
  void printSummary() const;

private:
  /** Drain the GPU of this rank, then wait for all ranks: the edges of the
   *  steady state and the end of the loop (collective). */
  static void boundary();
  /** End the open stage, if any, at the given time, where the next one
   *  starts. */
  void endStage(Clock::time_point now);

  Clock::time_point programStart_{}, initEnd_{}, loopStart_{}, loopEnd_{},
      end_{}, steadyStart_{}, steadyEnd_{};
  /// the start of the cycle, and of its open stage (the first one starts
  /// with the cycle)
  Clock::time_point cycleStart_{}, stageStart_{};
  /// the regions of the steady state, the cycle and its stage
  profiling::Phase steadyRegion_, cycleRegion_, stageRegion_;
  bool inSteadyState_ = false;

  /// the open stage, -1 for none, and the stages of the current cycle
  int stage_ = -1;
  double stageMs_[kCycleStages] = {};

  // all cycles
  int cycles_ = 0;
  double cycleMsSum_ = 0.0, cycleMsSumSq_ = 0.0;
  // steady-state cycles
  int steadyCycles_ = 0;
  double steadyMsSum_ = 0.0, steadyMsSumSq_ = 0.0;
  double steadyMsMin_ = 0.0, steadyMsMax_ = 0.0;
  double steadyStageMsSum_[kCycleStages] = {};
};

#endif // TIMING_H
