/**
 * @file Profiler.cpp
 * @brief The out-of-line parts of the profiler backends (Profiler.h): the NVTX
 * thread names and recording window, CrayPat's recording switch, the stage
 * timers and their report. Each part is compiled only in the build that
 * enables its backend.
 */

#include "Profiler.h"

#if defined(IPIC3D_PROFILING_NVTX)
#include <cuda_profiler_api.h>
#include <sys/syscall.h>
#include <unistd.h>
#endif
#if defined(IPIC3D_CRAYPAT)
#include <pat_api.h>
#endif
#if defined(IPIC3D_TIME_TASKS)
#include "MPIdata.h"
#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <mpi.h>
#endif

namespace profiling {

#if defined(IPIC3D_TIME_TASKS)
/** Nanoseconds per millisecond and per second. */
inline constexpr double kNanosecondsPerMs = 1.0e6;
inline constexpr double kNanosecondsPerSecond = 1.0e9;
#endif

#if defined(IPIC3D_PROFILING_NVTX)
void NvtxTracer::nameThread(const char* name) noexcept {
  nvtxNameOsThreadA(static_cast<std::uint32_t>(syscall(SYS_gettid)), name);
}

void NvtxTracer::record(bool on) noexcept {
  // a failure (no profiler attached is not one) leaves the capture as it was
  if (on)
    (void)cudaProfilerStart();
  else
    (void)cudaProfilerStop();
}
#endif

#if defined(IPIC3D_CRAYPAT)
void CrayPat::record(bool on) noexcept {
  (void)PAT_record(on ? PAT_STATE_ON : PAT_STATE_OFF);
}
#endif

#if defined(IPIC3D_TIME_TASKS)

namespace {

/** The time and the calls of every region, added by any thread. Static
 *  storage: zero before the first region. */
std::atomic<std::int64_t> totalNanoseconds[kRegionCount];
std::atomic<std::int64_t> totalCalls[kRegionCount];

/** The totals of every region at an instant. */
struct Totals {
  std::int64_t nanoseconds[kRegionCount];
  std::int64_t calls[kRegionCount];
};

Totals totals() {
  Totals t;
  for (int r = 0; r < kRegionCount; ++r) {
    t.nanoseconds[r] = totalNanoseconds[r].load(std::memory_order_relaxed);
    t.calls[r] = totalCalls[r].load(std::memory_order_relaxed);
  }
  return t;
}

/** The totals at the start and the end of the steady state. */
Totals steadyStart{}, steadyEnd{};

/** The time of every region in each steady-state cycle: the totals at the
 *  start of the current cycle, the time of the last cycle that ended, and the
 *  sum of the squared times of the cycles before it. A cycle's time joins the
 *  sum when the next one ends, or with the rest of the steady state after the
 *  last cycle, so that the cycles add up to the steady state. */
std::int64_t cycleStartNanoseconds[kRegionCount];
std::int64_t cycleNanoseconds[kRegionCount];
double cycleSquaresSum[kRegionCount];

/** Add the time of the last cycle that ended to the sum of the squares (0 at
 *  first: no cycle ended). */
void addCycleSquares() {
  for (int r = 0; r < kRegionCount; ++r) {
    const double ns = static_cast<double>(cycleNanoseconds[r]);
    cycleSquaresSum[r] += ns * ns;
  }
}

} // namespace

void StageTimer::add(Region region, std::int64_t nanoseconds) noexcept {
  const int r = static_cast<int>(region);
  totalNanoseconds[r].fetch_add(nanoseconds, std::memory_order_relaxed);
  totalCalls[r].fetch_add(1, std::memory_order_relaxed);
}

void StageTimer::mark(bool on) {
  if (on) {
    steadyStart = totals();
    for (int r = 0; r < kRegionCount; ++r)
      cycleStartNanoseconds[r] = steadyStart.nanoseconds[r];
    return;
  }
  steadyEnd = totals();
  for (int r = 0; r < kRegionCount; ++r)
    cycleNanoseconds[r] += steadyEnd.nanoseconds[r] - cycleStartNanoseconds[r];
  addCycleSquares();
}

void StageTimer::endCycle() {
  addCycleSquares();
  for (int r = 0; r < kRegionCount; ++r) {
    const std::int64_t now =
        totalNanoseconds[r].load(std::memory_order_relaxed);
    cycleNanoseconds[r] = now - cycleStartNanoseconds[r];
    cycleStartNanoseconds[r] = now;
  }
}

void StageTimer::report(int steadyCycles) {
  // run: seconds and calls; steady: ms per cycle, its variance over the
  // cycles and calls per cycle, from the closing boundary of the steady
  // state, before the finalisation
  const Totals run = totals();
  const double perCycle = steadyCycles > 0 ? 1.0 / steadyCycles : 0.0;
  double runSeconds[kRegionCount], runCalls[kRegionCount],
      steadyMs[kRegionCount], steadyVariance[kRegionCount],
      steadyCalls[kRegionCount];
  for (int r = 0; r < kRegionCount; ++r) {
    runSeconds[r] = run.nanoseconds[r] / kNanosecondsPerSecond;
    runCalls[r] = static_cast<double>(run.calls[r]);
    const double steadyNanoseconds = static_cast<double>(
        steadyEnd.nanoseconds[r] - steadyStart.nanoseconds[r]);
    steadyMs[r] = steadyNanoseconds / kNanosecondsPerMs * perCycle;
    // the variance over the cycles, (sum of the squares - n mean^2) / (n - 1),
    // in ms^2 (rounding can leave a tiny negative sum)
    const double squaresAboutMean =
        std::max(0.0, cycleSquaresSum[r] -
                          steadyNanoseconds * steadyNanoseconds * perCycle);
    steadyVariance[r] = steadyCycles > 1
                            ? squaresAboutMean / (steadyCycles - 1) /
                                  (kNanosecondsPerMs * kNanosecondsPerMs)
                            : 0.0;
    steadyCalls[r] =
        static_cast<double>(steadyEnd.calls[r] - steadyStart.calls[r]) *
        perCycle;
  }

  const MPI_Comm comm = MPIdata::get_PicGlobalComm();
  const int nprocs = MPIdata::get_nprocs();
  double runMax[kRegionCount], runCallsMax[kRegionCount],
      steadyMin[kRegionCount], steadyMax[kRegionCount], steadySum[kRegionCount],
      steadyVarianceSum[kRegionCount], steadyCallsSum[kRegionCount];
  MPI_Reduce(runSeconds, runMax, kRegionCount, MPI_DOUBLE, MPI_MAX, 0, comm);
  MPI_Reduce(runCalls, runCallsMax, kRegionCount, MPI_DOUBLE, MPI_MAX, 0, comm);
  MPI_Reduce(steadyMs, steadyMin, kRegionCount, MPI_DOUBLE, MPI_MIN, 0, comm);
  MPI_Reduce(steadyMs, steadyMax, kRegionCount, MPI_DOUBLE, MPI_MAX, 0, comm);
  MPI_Reduce(steadyMs, steadySum, kRegionCount, MPI_DOUBLE, MPI_SUM, 0, comm);
  MPI_Reduce(steadyVariance, steadyVarianceSum, kRegionCount, MPI_DOUBLE,
             MPI_SUM, 0, comm);
  MPI_Reduce(steadyCalls, steadyCallsSum, kRegionCount, MPI_DOUBLE, MPI_SUM, 0,
             comm);
  if (MPIdata::get_rank() != 0)
    return;

  std::printf(
      "\n=== Stage timers (TIME_TASKS) ===\n"
      "Host wall time of each region, the regions it contains included,\n"
      "summed over its calls and over the threads that ran it (the mover of\n"
      "each species runs on its own thread). Steady state: the %d cycles\n"
      "after the first, in ms per cycle: minimum, mean and maximum over the\n"
      "%d ranks, and the standard deviation over the cycles, root mean\n"
      "square over the ranks; run: the whole run in s, maximum over the\n"
      "ranks.\n",
      steadyCycles, nprocs);
  std::printf("Stage %-20s %11s %11s %11s %11s %11s %11s\n", "region",
              "calls/cycle", "min ms", "mean ms", "max ms", "std ms",
              "run max s");
  for (int r = 0; r < kRegionCount; ++r)
    if (runCallsMax[r] > 0.0) // the regions that ran on some rank
      std::printf("Stage %-20s %11.2f %11.3f %11.3f %11.3f %11.3f %11.3f\n",
                  kRegionNames[r], steadyCallsSum[r] / nprocs, steadyMin[r],
                  steadySum[r] / nprocs, steadyMax[r],
                  std::sqrt(steadyVarianceSum[r] / nprocs), runMax[r]);
  std::fflush(stdout);
}

#endif // IPIC3D_TIME_TASKS

} // namespace profiling
