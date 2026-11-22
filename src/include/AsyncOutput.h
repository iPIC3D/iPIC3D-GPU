#ifndef ASYNC_OUTPUT_H
#define ASYNC_OUTPUT_H

/**
 * @file AsyncOutput.h
 * @brief Deferred output: jobs that write host copies of the device data to
 * disk on writer threads (lanes) while the simulation goes on, and complete
 * on the main thread at a later cycle.
 *
 * A job writes on its lane and makes no MPI call there (MPI runs with
 * MPI_THREAD_FUNNELED); its finish() runs on the main thread at the polls
 * after the write and may use MPI (the barrier of a checkpoint). A lane is
 * one thread: its jobs write one at a time, in the order submitted, so a file
 * keeps the order of its cycles and a library that is not thread-safe (HDF5,
 * an ADIOS2 instance) is used by one thread.
 *
 * The host buffers a job reads (HostBuffers) keep its data until it has
 * written them: the next copy into them waits for the job (backpressure). A
 * write can thus run until the next copy into its buffers.
 */

#include <deque>
#include <future>
#include <memory>
#include <string>
#include <vector>

class ThreadPool;

/** The writer threads: each serialises the library its jobs use. */
enum class OutputLane { Hdf5, Adios2, Count };

/** A deferred output. */
class OutputJob {
public:
  virtual ~OutputJob() = default;
  /** What is written, for error messages. */
  virtual std::string name() const = 0;
  /** Writes to disk, on the job's lane: no MPI call. */
  virtual void write() = 0;
  /** On the main thread once write() has returned, at every poll until it
   *  returns true: the MPI steps that complete the output. With wait, it
   *  blocks until done. */
  virtual bool finish(bool wait) {
    (void)wait;
    return true;
  }
};

/** The sets of host buffers, filled by GPU copies, that an output job reads:
 *  one bit per set, which the caller defines. */
using HostBuffers = unsigned;

class AsyncOutput {
public:
  /** Deferred with async; otherwise a job writes and finishes in submit(),
   *  on the main thread, as a synchronous write. */
  explicit AsyncOutput(bool async);
  ~AsyncOutput();
  AsyncOutput(const AsyncOutput&) = delete;
  AsyncOutput& operator=(const AsyncOutput&) = delete;

  bool deferred() const { return async_; }

  /** Writes the job on its lane, after the jobs submitted to it before; the
   *  caller keeps the buffers it reads (reads) until it has written them
   *  (waitForReaders). */
  void submit(OutputLane lane, std::unique_ptr<OutputJob> job,
              HostBuffers reads);

  /** Main thread, once a cycle: completes the written jobs in the order
   *  submitted; a writer's error is thrown again here. */
  void poll();

  /** Before buffers are overwritten: waits until the jobs that read any of
   *  them have written. */
  void waitForReaders(HostBuffers buffers);

  /** Waits until the jobs of lane have written (before the main thread uses
   *  the lane's library itself). */
  void waitLane(OutputLane lane);

  /** Waits for every job and completes it (blocking finish). */
  void drain();

  /** Seconds the main thread has waited for the writers. */
  double waitedSeconds() const { return waited_; }

private:
  /** A job submitted and not completed. */
  struct Pending {
    OutputLane lane;
    HostBuffers reads;
    std::unique_ptr<OutputJob> job;
    std::shared_future<void> written;
  };

  /** Waits until the job of entry has written, timed; its error is thrown
   *  again with the job name. */
  void await(const Pending& entry);

  bool async_;
  std::vector<std::unique_ptr<ThreadPool>> lanes_; // created on first use
  std::deque<Pending> pending_; // submitted, not completed, in order
  double waited_ = 0.0;
};

#endif // ASYNC_OUTPUT_H
