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
#include "cudaTypeDef.cuh"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <mpi.h>

/** Cycle stages per line of the summary. */
inline constexpr int kStagesPerSummaryLine = 3;
/** Seconds per hour, milliseconds per second. */
inline constexpr double kSecondsPerHour = 3600.0;
inline constexpr double kMsPerSecond = 1.0e3;

namespace {

double seconds(Timing::Clock::time_point a, Timing::Clock::time_point b) {
  return std::chrono::duration<double>(b - a).count();
}

double milliseconds(Timing::Clock::time_point a, Timing::Clock::time_point b) {
  return std::chrono::duration<double, std::milli>(b - a).count();
}

/** Sample standard deviation of n values from their sum and sum of squares. */
double stddev(int n, double sum, double sumSq) {
  if (n < 2)
    return 0.0;
  const double var = (sumSq - sum * sum / n) / (n - 1);
  return var > 0.0 ? std::sqrt(var) : 0.0;
}

/** Name of the cycle stage of the given index: its region's. */
const char* stageName(int stage) {
  return profiling::regionName(static_cast<profiling::Region>(
      static_cast<int>(kFirstCycleStage) + stage));
}

} // namespace

void Timing::boundary() {
  cudaErrChk(cudaDeviceSynchronize());
  MPI_Barrier(MPIdata::get_PicGlobalComm());
}

void Timing::startCycle(int cycle) {
  cycleRegion_.start(profiling::Region::Cycle, cycle);
  std::fill(std::begin(stageMs_), std::end(stageMs_), 0.0);
  cycleStart_ = stageStart_ = Clock::now();
}

void Timing::stage(profiling::Region region) {
  const int index =
      static_cast<int>(region) - static_cast<int>(kFirstCycleStage);
  assert(index >= 0 && index < kCycleStages && "not a cycle stage");
  endStage(Clock::now());
  stage_ = index;
  stageRegion_.start(region);
}

void Timing::endStage(Clock::time_point now) {
  stageRegion_.stop();
  // each stage ends where the next starts: the stages add up to the cycle
  if (stage_ >= 0) {
    stageMs_[stage_] += milliseconds(stageStart_, now);
    stageStart_ = now;
  }
  stage_ = -1;
}

void Timing::endCycle() {
  const Clock::time_point now = Clock::now();
  endStage(now);
  cycleRegion_.stop();
  const double totalMs = milliseconds(cycleStart_, now);
  cycles_++;
  cycleMsSum_ += totalMs;
  cycleMsSumSq_ += totalMs * totalMs;
  if (inSteadyState_) {
    profiling::endSteadyCycle();
    steadyMsMin_ =
        steadyCycles_ == 0 ? totalMs : std::min(steadyMsMin_, totalMs);
    steadyMsMax_ =
        steadyCycles_ == 0 ? totalMs : std::max(steadyMsMax_, totalMs);
    steadyCycles_++;
    steadyMsSum_ += totalMs;
    steadyMsSumSq_ += totalMs * totalMs;
    for (int s = 0; s < kCycleStages; s++)
      steadyStageMsSum_[s] += stageMs_[s];
  }

  if (MPIdata::get_rank() == 0) {
    std::printf("Execution time cycle: %g ms  [", totalMs);
    for (int s = 0; s < kCycleStages; s++)
      std::printf("%s%s=%g", s == 0 ? "" : " ", stageName(s), stageMs_[s]);
    std::printf("]\n");
  }
}

void Timing::startSteadyState() {
  boundary();
  profiling::recordSteadyState(true);
  steadyRegion_.start(profiling::Region::SteadyState);
  inSteadyState_ = true;
  steadyStart_ = Clock::now();
}

void Timing::endLoop() {
  // the GPU work of the last cycle belongs to the loop
  boundary();
  loopEnd_ = Clock::now();
  if (inSteadyState_) {
    steadyEnd_ = loopEnd_;
    inSteadyState_ = false;
    steadyRegion_.stop();
    profiling::recordSteadyState(false);
  }
}

void Timing::printSummary() const {
  enum Span {
    Total,
    Init,
    PreCycle,
    Loop,
    Steady,
    Finalize,
    Simulation,
    Spans
  };
  const double local[Spans] = {
      seconds(programStart_, end_),
      seconds(programStart_, initEnd_),
      seconds(initEnd_, loopStart_),
      seconds(loopStart_, loopEnd_),
      steadyCycles_ > 0 ? seconds(steadyStart_, steadyEnd_) : 0.0,
      seconds(loopEnd_, end_),
      seconds(initEnd_, end_)};
  double stageMs[kCycleStages];
  for (int s = 0; s < kCycleStages; s++)
    stageMs[s] = steadyCycles_ > 0 ? steadyStageMsSum_[s] / steadyCycles_ : 0.0;

  const MPI_Comm comm = MPIdata::get_PicGlobalComm();
  double maxSpan[Spans], stageSum[kCycleStages], stageMax[kCycleStages];
  MPI_Reduce(local, maxSpan, Spans, MPI_DOUBLE, MPI_MAX, 0, comm);
  MPI_Reduce(stageMs, stageSum, kCycleStages, MPI_DOUBLE, MPI_SUM, 0, comm);
  MPI_Reduce(stageMs, stageMax, kCycleStages, MPI_DOUBLE, MPI_MAX, 0, comm);
  if (MPIdata::get_rank() != 0)
    return;

  const int nprocs = MPIdata::get_nprocs();
  std::printf("\n\n*** SIMULATION ENDED SUCESSFULLY ***\n");
  std::printf(
      " Total execution    : %g sec (%g hours)\n"
      " Initialization     : %g sec\n"
      " Pre-cycle setup    : %g sec (data-analysis setup + initial moments)\n"
      " Cycle loop         : %g sec over %d cycles\n"
      "                      avg = %g ms/cycle, stddev = %g ms\n",
      maxSpan[Total], maxSpan[Total] / kSecondsPerHour, maxSpan[Init],
      maxSpan[PreCycle], maxSpan[Loop], cycles_,
      cycles_ > 0 ? cycleMsSum_ / cycles_ : 0.0,
      stddev(cycles_, cycleMsSum_, cycleMsSumSq_));
  if (steadyCycles_ > 0) {
    std::printf(
        " Steady state       : %g sec over %d cycles (all but the first)\n"
        "                      avg = %g ms/cycle; rank 0: stddev = %g "
        "ms, min = %g ms, max = %g ms\n"
        "                      stages, ms/cycle, mean (max) over the "
        "ranks:",
        maxSpan[Steady], steadyCycles_,
        kMsPerSecond * maxSpan[Steady] / steadyCycles_,
        stddev(steadyCycles_, steadyMsSum_, steadyMsSumSq_), steadyMsMin_,
        steadyMsMax_);
    for (int s = 0; s < kCycleStages; s++)
      std::printf("%s%s %.3g (%.3g)",
                  s % kStagesPerSummaryLine == 0 ? "\n                      "
                                                 : ", ",
                  stageName(s), stageSum[s] / nprocs, stageMax[s]);
    std::printf("\n");
  } else {
    std::printf(" Steady state       : none (one cycle or less)\n");
  }
  std::printf(" Finalize           : %g sec\n"
              " Simulation Time    : %g sec (%g hours) [end of Init -> end of "
              "Finalize, i.e. pre-cycle setup + cycle loop + finalize]\n"
              " Times are the maximum over the %d ranks; the cycle lines and "
              "their statistics are those of rank 0.\n"
              "***\n\n",
              maxSpan[Finalize], maxSpan[Simulation],
              maxSpan[Simulation] / kSecondsPerHour, nprocs);
  std::fflush(stdout);
}
