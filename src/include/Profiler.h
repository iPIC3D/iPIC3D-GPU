#ifndef PROFILER_H
#define PROFILER_H

/**
 * @file Profiler.h
 * @brief The profiler of iPIC3D: named regions of the code (the stages of the
 * cycle, the field solver, the particle pipeline, the output, the data
 * analysis), seen by the GPU profilers and by the stage timers of the run, and
 * the recording window of the steady-state cycles.
 *
 * A region is opened by a Range object and closed when it goes out of scope:
 *
 *     profiling::Range range(profiling::Region::MaxwellSolve);
 *     profiling::Range range(profiling::Region::Mover, species); // indexed
 *
 * A Phase opens and closes regions by calls, one at a time, where a scope
 * cannot (the stages of the time loop, the phases of the initialisation).
 *
 * Three backends, each a class with one implementation per build, chosen by
 * CMake at compile time (no run-time choice, no virtual call):
 *  - Tracer, PROFILING=ON: a region is an NVTX range of the domain "iPIC3D"
 *    on a CUDA build (Nsight Systems, Nsight Compute) and a ROCTX range on a
 *    HIP build (rocprofv3, rocprof); the index of an indexed region is the
 *    NVTX payload (ROCTX has none: a region keeps its one name);
 *  - StageTimer, TIME_TASKS=ON: the host wall time of every region, summed
 *    over its calls and the threads that ran it, reported at the end of the
 *    run;
 *  - CrayPat, CRAYPAT=ON: the executable records only the steady-state cycles
 *    when run under pat_run.
 * A backend that is off is a class of empty inline functions: with the three
 * off, Range and Phase are empty objects and the optimised code holds nothing
 * of the profiler.
 *
 * A region measures the host thread that opens it. The GPU work it launches
 * runs later, on its stream; Nsight Systems and rocprofv3 attach the kernels
 * to the range that launched them, and a host timer sees the GPU time only
 * where the host waits for it.
 */

#include <cstdint>

#if defined(IPIC3D_PROFILING_NVTX)
#include <nvtx3/nvToolsExt.h>
#elif defined(IPIC3D_PROFILING_ROCTX_LEGACY)
#include <roctracer/roctx.h>
#elif defined(IPIC3D_PROFILING_ROCTX)
#include <rocprofiler-sdk-roctx/roctx.h>
#endif
#if defined(IPIC3D_TIME_TASKS)
#include <chrono>
#endif
#include <optional>
#include <type_traits>

namespace profiling {

/** The name of each region, in the order of Region: the NVTX or ROCTX range
 *  name and the row of the stage timers. */
inline constexpr const char* kRegionNames[] = {
    // the run
    "Init", "InitInput", "InitSelectDevice", "InitFields", "InitParticles",
    "InitOutput", "InitDevice", "InitialMoments", "SteadyState", "Cycle",
    "Finalize", "FinalizeOutput", "FinalizeDevice",
    // the stages of a cycle
    "ParticleCountLog", "AnalysisPrepare", "HeatFlux", "AnalysisStart",
    "FieldSolveE", "AnalysisWait", "MoverLaunch", "Output", "ParticleExchange",
    "FieldSolveB", "MomentsAwait", "OutputCopy",
    // field solver
    "PoissonCorrection", "MaxwellSource", "MaxwellSolve", "EigenvalueEstimate",
    "KrylovIteration", "CleaningIteration", "KrylovReduce", "MaxwellImage",
    "CleaningImage", "Preconditioner", "SmoothE", "FinishE", "Faraday",
    "DivBCleaning", "HaloExchange", "HaloBegin", "HaloEnd",
    // moments
    "MomentKernelsWait", "MomentHalo", "DerivedMoments", "HatFunctions",
    // particles
    "FieldPack", "ParticleMerge", "Mover", "ParticleSplit", "MoverCounts",
    "ParticleExtract", "MoverWait", "PlanetAbsorption", "ParticleInjection",
    "CompactionWait", "SortStart", "ParticleMPI", "Exosphere",
    "ExosphereSample", "SortFinish", "ParticleSync", "ParticleAppend",
    "MomentDeposit", "MomentScatter", "ParticleSort", "SortWait",
    // output
    "OutputPoll", "FieldCopyD2H", "HeatFluxHalo", "Catalyst", "Diagnostics",
    "OutputCopyWait", "WriteRestart", "WriteFields", "WriteParticles",
    "WriteTestParticles", "OutputWait", "OutputJob", "OutputFinish",
    "ParticleCopyD2H", "SampleCountWait",
    // data analysis
    "AnalysisInit", "AnalysisRun", "VelocityHistogram", "GMMFit",
    "GMMIteration", "MacrocellSpectra", "AnalysisSync", "AnalysisWrite",
    "AnalysisFinalize"};
/** Number of regions. */
inline constexpr int kRegionCount =
    sizeof(kRegionNames) / sizeof(kRegionNames[0]);
/** Name of the NVTX domain of the ranges, a row of the Nsight Systems
 *  timeline. */
inline constexpr const char* kDomainName = "iPIC3D";
/** Lowest value and span of an NVTX colour channel, 48 to 239: visible on
 *  light and dark timelines. */
inline constexpr std::uint32_t kColorChannelMin = 48;
inline constexpr std::uint32_t kColorChannelRange = 192;
/** The hash of a region name that gives its colour: FNV-1a (offset basis,
 *  prime), then the MurmurHash3 finaliser (its two multipliers). */
inline constexpr std::uint32_t kFnvOffsetBasis = 2166136261u;
inline constexpr std::uint32_t kFnvPrime = 16777619u;
inline constexpr std::uint32_t kMurmurMix1 = 0x85ebca6bu;
inline constexpr std::uint32_t kMurmurMix2 = 0xc2b2ae35u;

/**
 * The regions of the code, in the order of kRegionNames and of the
 * stage-timer report. Each is opened at one or more places; the comment names
 * the code it covers.
 */
enum class Region : std::uint8_t {
  // ---- the run (iPIC3D.cpp, Timing, c_Solver::Init, c_Solver::Finalize)
  Init,             ///< c_Solver::Init
  InitInput,        ///< input file, topology, run record
  InitSelectDevice, ///< GPU of the rank, CUDA context (selectDevice)
  InitFields,       ///< grid, field arrays, initial or restart fields
  InitParticles,    ///< initial or restart particles, exchange buffers
  InitOutput,       ///< output backends, diagnostics files, exosphere source
  InitDevice,       ///< GPU arrays, streams and events (initCUDA)
  InitialMoments,   ///< moments of the initial particles (CalculateMoments)
  SteadyState,      ///< the cycles after the warm-up (first) cycle
  Cycle,            ///< one cycle of the time loop, indexed by its number
  Finalize,         ///< final diagnostics and checkpoint, output, GPU release
  FinalizeOutput,   ///< deferred outputs completed, output files closed
  FinalizeDevice,   ///< GPU arrays, streams and events released (deInitCUDA)
  // ---- the stages of a cycle, in loop order, opened in turn by the cycle
  //      timer (Timing::stage, runCycle in iPIC3D.cpp): kept together, from
  //      ParticleCountLog to OutputCopy (Timing.h)
  ParticleCountLog, ///< particle counts to particleNum<rank>.csv
  AnalysisPrepare,  ///< sort and field pack for the data analysis
  HeatFlux,         ///< heat-flux kernels and copies, on their stream
  AnalysisStart,    ///< launch of the data-analysis thread
  FieldSolveE,      ///< E(n+theta): source, Krylov solve, smoothing, halo
  AnalysisWait,     ///< wait for the data-analysis thread
  MoverLaunch,      ///< field pack, merging, launch of the mover tasks
  Output,           ///< diagnostics, checkpoint, field and particle output
  ParticleExchange, ///< mover wait, planet, MPI exchange, sort, moments
  FieldSolveB,      ///< Faraday's law, B halo, E(n+1)
  MomentsAwait,     ///< moment wait, moment halo, derived moments
  OutputCopy,       ///< device-to-host copies for the next cycle's output
  // ---- field solver (EMfields3DGPU.cpp, GPUHaloComm.cu)
  PoissonCorrection,  ///< div(E) correction before the solve
  MaxwellSource,      ///< right-hand side of the Maxwell system
  MaxwellSolve,       ///< Krylov (or Chebyshev) solve of the Maxwell system
  EigenvalueEstimate, ///< power iteration for the Chebyshev bounds
  KrylovIteration,    ///< one step of the Maxwell solve, indexed by it
  CleaningIteration,  ///< one step of a div(E) or div(B) cleaning solve
  KrylovReduce,       ///< dot products of any solve: stream wait, Allreduce
  MaxwellImage,       ///< application of the Maxwell operator
  CleaningImage,      ///< application of a cleaning operator
  Preconditioner,     ///< cleaning Chebyshev or block-Jacobi preconditioner
  SmoothE,            ///< smoothing of E(n+theta) for the mover
  FinishE,            ///< halo and open-face layers of E(n+theta)
  Faraday,            ///< curl of E and update of the cell B
  DivBCleaning,       ///< div(B) cleaning of the cell B
  HaloExchange,       ///< field halo exchange and ghost fill (blocking)
  HaloBegin,          ///< start of an overlapped field halo exchange
  HaloEnd,            ///< end of an overlapped field halo exchange
  // ---- moments (iPIC3Dlib.cu, EMfields3DGPU.cpp)
  MomentKernelsWait, ///< host wait for the moment kernels of every species
  MomentHalo,        ///< ghost exchange of the species moments
  DerivedMoments,    ///< sums over species, planet and outflow-layer charge
  HatFunctions,      ///< implicit source terms (rhohat, Jhat)
  // ---- particles (iPIC3Dlib.cu)
  FieldPack,         ///< field buffer of the mover
  ParticleMerge,     ///< sort and merging of one species, indexed by it
  Mover,             ///< mover task of one species (worker thread)
  ParticleSplit,     ///< particle splitting of one species
  MoverCounts,       ///< wait for the mover's departure counts
  ParticleExtract,   ///< exiting and planet extraction, compaction
  MoverWait,         ///< main-thread wait for the mover tasks
  PlanetAbsorption,  ///< planet charge balance and reflection
  ParticleInjection, ///< compaction wait, array room, inflow injection
  CompactionWait,    ///< wait for the compaction of one species
  SortStart,         ///< cell sort stages 1-3, before the exchange
  ParticleMPI,       ///< MPI particle exchange of one species
  Exosphere,         ///< exosphere ionisation source
  ExosphereSample,   ///< sampling of one species (mover worker thread)
  SortFinish,        ///< cell sort scatter
  ParticleSync,      ///< wait for the particle streams before the append
  ParticleAppend,    ///< incoming particles and moments of one species
  MomentDeposit,     ///< moment kernels of one species (in ParticleAppend)
  MomentScatter,     ///< moments into the field-solver arrays
  ParticleSort,      ///< cell sort outside the mover pipeline
  SortWait,          ///< wait for the cell sorts of the analysis or moments
  // ---- output (iPIC3Dlib.cu, IOManager.cpp, AsyncOutput.cpp)
  OutputPoll,         ///< completion of deferred output jobs
  FieldCopyD2H,       ///< field arrays to the host
  HeatFluxHalo,       ///< heat-flux copy wait and ghost exchange
  Catalyst,           ///< in-situ visualisation (USE_CATALYST)
  Diagnostics,        ///< conserved quantities, planet diagnostics
  OutputCopyWait,     ///< wait for the copies and sums of OutputCopy
  WriteRestart,       ///< checkpoint
  WriteFields,        ///< field output
  WriteParticles,     ///< particle output
  WriteTestParticles, ///< test-particle output
  OutputWait,         ///< main-thread wait for an output job
  OutputJob,          ///< an output job on its writer thread
  OutputFinish,       ///< blocking completion of an output (checkpoint wait)
  ParticleCopyD2H,    ///< particle copies, sampling and sums for the output
  SampleCountWait,    ///< wait for the counts of one species' sample
  // ---- data analysis (dataAnalysis.cu; the analysis thread from AnalysisRun
  //      to AnalysisWrite)
  AnalysisInit,      ///< set-up: analyses, GPU buffers, record files
  AnalysisRun,       ///< the analysis of one cycle
  VelocityHistogram, ///< velocity histograms of one species
  GMMFit,            ///< Gaussian-mixture fits of one species
  GMMIteration,      ///< one EM step of the fits, indexed by it
  MacrocellSpectra,  ///< energy spectra of one species
  AnalysisSync,      ///< wait for the GPU work of one species' analyses
  AnalysisWrite,     ///< record files of the cycle
  AnalysisFinalize,  ///< release: thread, GPU buffers, record files
  Count
};
static_assert(kRegionCount == static_cast<int>(Region::Count),
              "one name in kRegionNames per region, in the order of Region");

/** Name of a region: the NVTX or ROCTX range name and the stage-timer row. */
constexpr const char* regionName(Region region) {
  return kRegionNames[static_cast<int>(region)];
}

/** A colour of the region that does not change between runs: the bytes of a
 *  hash of its name spread over the RGB channels (FNV-1a, then the
 *  MurmurHash3 finaliser). */
constexpr std::uint32_t regionColor(Region region) {
  std::uint32_t hash = kFnvOffsetBasis;
  for (const char* c = regionName(region); *c != '\0'; ++c)
    hash = (hash ^ static_cast<unsigned char>(*c)) * kFnvPrime;
  hash = (hash ^ (hash >> 16)) * kMurmurMix1;
  hash = (hash ^ (hash >> 13)) * kMurmurMix2;
  hash ^= hash >> 16;
  std::uint32_t argb = 0xff000000u;
  for (int shift = 16; shift >= 0; shift -= 8)
    argb |= (kColorChannelMin +
             (((hash >> shift) & 0xffu) * kColorChannelRange >> 8))
            << shift;
  return argb;
}

// ===========================================================================
//  Backend: tracing (PROFILING). One static interface: push a region (with an
//  index), pop it, name the calling thread, open or close the recording
//  window of the tool.
// ===========================================================================

#if defined(IPIC3D_PROFILING_NVTX)

/** NVTX ranges of the domain kDomainName, with the region names registered
 *  and the colours computed once per process (registered strings are the
 *  cheapest NVTX messages). */
class NvtxTracer {
public:
  static void push(Region region) noexcept {
    const nvtxEventAttributes_t event = attributes(region);
    (void)nvtxDomainRangePushEx(registry().domain, &event);
  }
  static void push(Region region, std::int64_t index) noexcept {
    nvtxEventAttributes_t event = attributes(region);
    event.payloadType = NVTX_PAYLOAD_TYPE_INT64;
    event.payload.llValue = index;
    (void)nvtxDomainRangePushEx(registry().domain, &event);
  }
  static void pop() noexcept { (void)nvtxDomainRangePop(registry().domain); }
  /** The calling OS thread's name in the timeline (Profiler.cpp). */
  static void nameThread(const char* name) noexcept;
  /** cudaProfilerStart (on) or cudaProfilerStop (off): Nsight Systems
   *  (--capture-range=cudaProfilerApi) and Nsight Compute
   *  (--profile-from-start off) profile in between only (Profiler.cpp). */
  static void record(bool on) noexcept;

private:
  struct Registry {
    nvtxDomainHandle_t domain;
    nvtxStringHandle_t names[kRegionCount];
    std::uint32_t colors[kRegionCount];
    Registry() : domain(nvtxDomainCreateA(kDomainName)) {
      for (int r = 0; r < kRegionCount; ++r) {
        names[r] = nvtxDomainRegisterStringA(domain, kRegionNames[r]);
        colors[r] = regionColor(static_cast<Region>(r));
      }
    }
  };
  static const Registry& registry() noexcept {
    static const Registry instance;
    return instance;
  }
  static nvtxEventAttributes_t attributes(Region region) noexcept {
    const Registry& r = registry();
    nvtxEventAttributes_t event{};
    event.version = NVTX_VERSION;
    event.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
    event.colorType = NVTX_COLOR_ARGB;
    event.color = r.colors[static_cast<int>(region)];
    event.messageType = NVTX_MESSAGE_TYPE_REGISTERED;
    event.message.registered = r.names[static_cast<int>(region)];
    return event;
  }
};
using Tracer = NvtxTracer;

#elif defined(IPIC3D_PROFILING_ROCTX)

/** ROCTX ranges (rocprofiler-sdk-roctx, or the legacy roctracer roctx): no
 *  domain, colour or payload. The index is left out, so that a region keeps
 *  one name the tools can sum; no recording window, the SteadyState region
 *  marks it. */
class RoctxTracer {
public:
  static void push(Region region) noexcept {
    (void)roctxRangePushA(regionName(region));
  }
  static void push(Region region, std::int64_t) noexcept { push(region); }
  static void pop() noexcept { (void)roctxRangePop(); }
  static void nameThread(const char* name) noexcept {
#if defined(IPIC3D_PROFILING_ROCTX_LEGACY)
    (void)name; // the legacy roctx names no thread
#else
    (void)roctxNameOsThread(name);
#endif
  }
  static void record(bool) noexcept {}
};
using Tracer = RoctxTracer;

#else

/** No tracing. */
class NullTracer {
public:
  static void push(Region) noexcept {}
  static void push(Region, std::int64_t) noexcept {}
  static void pop() noexcept {}
  static void nameThread(const char*) noexcept {}
  static void record(bool) noexcept {}
};
using Tracer = NullTracer;

#endif

// ===========================================================================
//  Backend: stage timers (TIME_TASKS). A Range times itself (start, stop) into
//  the totals of its region; the totals are kept at the edges of the steady
//  state (mark) and at the end of each of its cycles (endCycle), and printed
//  at the end (report).
// ===========================================================================

#if defined(IPIC3D_TIME_TASKS)

/** The host wall time of the regions (totals and report in Profiler.cpp). */
class StageTimer {
public:
  void start(Region region) noexcept {
    region_ = region;
    start_ = std::chrono::steady_clock::now();
  }
  void stop() const noexcept {
    add(region_, std::chrono::duration_cast<std::chrono::nanoseconds>(
                     std::chrono::steady_clock::now() - start_)
                     .count());
  }
  /** Keep the totals at the start (on) or the end (off) of the steady
   *  state (main thread). */
  static void mark(bool on);
  /** The end of a steady-state cycle (main thread): the time of each region
   *  in the cycle, for its standard deviation over the cycles. */
  static void endCycle();
  /** Print the totals, over the run and per steady-state cycle; collective
   *  over the ranks of the run. */
  static void report(int steadyCycles);

private:
  /** Add one call of a region that lasted the given time (any thread). */
  static void add(Region region, std::int64_t nanoseconds) noexcept;
  Region region_;
  std::chrono::steady_clock::time_point start_;
};

#else

/** No stage timers. */
class StageTimer {
public:
  void start(Region) noexcept {}
  void stop() const noexcept {}
  static void mark(bool) {}
  static void endCycle() {}
  static void report(int) {}
};

#endif

// ===========================================================================
//  Backend: CrayPat (CRAYPAT). Its recording state, switched on the main
//  thread, covers every thread; without pat_run it is the stub library's
//  PAT_record, which does nothing.
// ===========================================================================

#if defined(IPIC3D_CRAYPAT)
struct CrayPat {
  /** PAT_record on or off (Profiler.cpp). */
  static void record(bool on) noexcept;
};
#else
struct CrayPat {
  static void record(bool) noexcept {}
};
#endif

// ===========================================================================
//  The profiler: regions, thread names, recording window, report
// ===========================================================================

/** A region open from the construction to the destruction, on the calling
 *  thread. Ranges of one thread nest; do not move one to another thread. */
class [[nodiscard]] Range {
public:
  explicit Range(Region region) noexcept {
    Tracer::push(region);
    timer_.start(region);
  }
  /** An indexed region: the species, the cycle, the iteration. */
  Range(Region region, std::int64_t index) noexcept {
    Tracer::push(region, index);
    timer_.start(region);
  }
  // With every region backend off the destructor is the implicit one,
  // trivial: a Range then leaves no clean-up in the code around it
#if defined(IPIC3D_PROFILING_NVTX) || defined(IPIC3D_PROFILING_ROCTX) ||       \
    defined(IPIC3D_TIME_TASKS)
  ~Range() {
    timer_.stop();
    Tracer::pop();
  }
#endif
  Range(const Range&) = delete;
  Range& operator=(const Range&) = delete;

private:
  StageTimer timer_;
};

/** A region opened and closed by calls rather than by a scope, one at a time
 *  (the phases of the initialisation, the stages of the time loop): start
 *  closes the open region, if any, and opens the given one; stop and the
 *  destructor close it. */
class LivePhase {
public:
  void start(Region region) { range_.emplace(region); }
  void start(Region region, std::int64_t index) {
    range_.emplace(region, index);
  }
  void stop() { range_.reset(); }

private:
  std::optional<Range> range_;
};

/** A phase of a Range that does nothing: no state to keep. */
class NullPhase {
public:
  void start(Region) noexcept {}
  void start(Region, std::int64_t) noexcept {}
  void stop() noexcept {}
};

/** The phase holder of the build: empty when a Range does nothing (its
 *  destructor trivial, every region backend off). */
using Phase = std::conditional_t<std::is_trivially_destructible_v<Range>,
                                 NullPhase, LivePhase>;

/** Name the calling OS thread in the tracing tool (Nsight Systems,
 *  rocprofv3); call it first thing in a new thread. */
inline void nameThread(const char* name) noexcept { Tracer::nameThread(name); }

/** At the start of main: CrayPat records nothing until recordSteadyState. */
inline void initRecording() noexcept { CrayPat::record(false); }

/**
 * Open (on) or close (off) the recording window of the steady-state cycles,
 * on the main thread, at the timing boundaries that also delimit the reported
 * steady-state time (Timing): the stage timers report it apart from the whole
 * run, cudaProfilerStart/Stop bound it for the Nsight tools, and CrayPat
 * records only inside it.
 */
inline void recordSteadyState(bool on) {
  StageTimer::mark(on);
  Tracer::record(on);
  CrayPat::record(on);
}

/** The end of a steady-state cycle, on the main thread (Timing): the stage
 *  timers keep the time of each region in the cycle. */
inline void endSteadyCycle() { StageTimer::endCycle(); }

/** Print the stage timers, over the run and per steady-state cycle;
 *  collective over the ranks of the run. */
inline void reportStageTimers(int steadyCycles) {
  StageTimer::report(steadyCycles);
}

} // namespace profiling

#endif // PROFILER_H
