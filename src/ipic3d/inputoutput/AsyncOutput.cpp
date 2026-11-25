/**
 * @file AsyncOutput.cpp
 * @brief Writer lanes, the completion of the written jobs, backpressure on
 * the host buffers they read (AsyncOutput.h).
 */

#include "AsyncOutput.h"
#include "Profiler.h"
#include "threadPool.hpp"

#include <chrono>
#include <sys/resource.h>
#include <sys/syscall.h>
#include <unistd.h>

namespace {

/** Niceness of a writer thread: it yields to the main thread, the mover
 *  launchers and the analysis, which the cycle waits for. */
constexpr int kWriterNiceness = 10;

double secondsSince(std::chrono::steady_clock::time_point start) {
  return std::chrono::duration<double>(std::chrono::steady_clock::now() - start)
      .count();
}

/** The write of a job, on its writer thread or in submit. */
void writeJob(OutputJob& job) {
  profiling::Range range(profiling::Region::OutputJob);
  job.write();
}

/** The blocking completion of a written job, on the main thread. */
void finishJob(OutputJob& job) {
  profiling::Range range(profiling::Region::OutputFinish);
  job.finish(true);
}

} // namespace

AsyncOutput::AsyncOutput(bool async)
    : async_(async), lanes_((size_t)OutputLane::Count) {}

AsyncOutput::~AsyncOutput() {
  // join the writers, which finish their queued jobs, before the jobs go
  lanes_.clear();
}

void AsyncOutput::submit(OutputLane lane, std::unique_ptr<OutputJob> job,
                         HostBuffers reads) {
  if (!async_) {
    writeJob(*job);
    finishJob(*job);
    return;
  }
  std::unique_ptr<ThreadPool>& thread = lanes_[(size_t)lane];
  if (!thread) {
    thread = std::make_unique<ThreadPool>(1, "iPIC3D output");
    // a failure leaves the default priority
    thread->enqueue([] {
      setpriority(PRIO_PROCESS, (id_t)syscall(SYS_gettid), kWriterNiceness);
    });
  }
  OutputJob* const writer = job.get();
  std::shared_future<void> written =
      thread->enqueue([writer] { writeJob(*writer); }).share();
  pending_.push_back({lane, reads, std::move(job), std::move(written)});
}

void AsyncOutput::poll() {
  while (!pending_.empty()) {
    Pending& front = pending_.front();
    if (front.written.wait_for(std::chrono::seconds(0)) !=
        std::future_status::ready)
      return;
    awaitTask(front.written, front.job->name()); // its error, if any
    if (!front.job->finish(false))
      return;
    pending_.pop_front();
  }
}

void AsyncOutput::waitForReaders(HostBuffers buffers) {
  for (const Pending& entry : pending_)
    if (entry.reads & buffers)
      await(entry);
}

void AsyncOutput::waitLane(OutputLane lane) {
  for (const Pending& entry : pending_)
    if (entry.lane == lane)
      await(entry);
}

void AsyncOutput::drain() {
  while (!pending_.empty()) {
    Pending& front = pending_.front();
    await(front);
    const auto start = std::chrono::steady_clock::now();
    finishJob(*front.job);
    waited_ += secondsSince(start);
    pending_.pop_front();
  }
}

void AsyncOutput::await(const Pending& entry) {
  profiling::Range range(profiling::Region::OutputWait);
  const auto start = std::chrono::steady_clock::now();
  awaitTask(entry.written, entry.job->name());
  waited_ += secondsSince(start);
}
