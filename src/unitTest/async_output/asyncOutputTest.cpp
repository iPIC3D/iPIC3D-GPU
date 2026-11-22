/**
 * @file asyncOutputTest.cpp
 * @brief Unit test of the deferred output (AsyncOutput): the jobs of a lane
 *        written in the order submitted, on a thread other than the main
 *        one, and completed on the main thread in the order submitted, a
 *        written job behind an unwritten one included; a copy into host
 *        buffers kept waiting by the jobs that read them and by no other
 *        (waitForReaders), a lane's library by the jobs of that lane
 *        (waitLane); a finish() polled until done, and blocking in drain();
 *        the error of a writer thrown again with the name of its job; the
 *        synchronous mode; the destructor waiting for the writers. CPU only.
 */

#include "AsyncOutput.h"

#include <algorithm>
#include <chrono>
#include <exception>
#include <future>
#include <iostream>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace {

/** The host buffer sets of the test jobs. */
constexpr HostBuffers kParticles = 1u << 0, kFields = 1u << 1;
/** How long a gated job is kept from writing while the main thread waits. */
constexpr std::chrono::milliseconds kGateDelay(50);
/** Polls that a finish done only by a blocking call would take. */
constexpr int kNeverPolled = 1000000;

int failures = 0;

void check(bool ok, const std::string& what) {
  std::cout << "Test " << (ok ? "passed" : "failed") << ": " << what << "\n";
  failures += !ok;
}

/** What the jobs of a test did: their ids in the order written and
 *  completed, and whether a write ran on the main thread or a finish off it.
 */
struct Record {
  std::mutex mutex;
  std::vector<int> written, finished;
  const std::thread::id mainThread = std::this_thread::get_id();
  bool writeOnMain = false, finishOffMain = false;

  std::vector<int> writtenNow() {
    std::lock_guard<std::mutex> lock(mutex);
    return written;
  }
  bool hasWritten(int id) {
    const std::vector<int> ids = writtenNow();
    return std::find(ids.begin(), ids.end(), id) != ids.end();
  }
};

/** A job that waits for its gate (if valid), then fails or records its
 *  write; its finish is done after `polls` calls that do not wait. */
class TestJob final : public OutputJob {
public:
  TestJob(int id, Record& record) : id_(id), record_(record) {}
  std::string name() const override { return "job " + std::to_string(id_); }
  void write() override {
    if (gate.valid())
      gate.wait();
    if (fail)
      throw std::runtime_error("injected failure");
    std::lock_guard<std::mutex> lock(record_.mutex);
    record_.written.push_back(id_);
    record_.writeOnMain |= std::this_thread::get_id() == record_.mainThread;
  }
  bool finish(bool wait) override {
    if (!wait && polls-- > 0)
      return false;
    std::lock_guard<std::mutex> lock(record_.mutex);
    record_.finished.push_back(id_);
    record_.finishOffMain |= std::this_thread::get_id() != record_.mainThread;
    return true;
  }

  std::shared_future<void> gate;
  int polls = 0;
  bool fail = false;

private:
  int id_;
  Record& record_;
};

/** A test job to submit; its gate, polls and fail may be set first. */
std::unique_ptr<TestJob> job(int id, Record& record) {
  return std::make_unique<TestJob>(id, record);
}

/** A test job that writes once gate is opened. */
std::unique_ptr<TestJob> gatedJob(int id, Record& record,
                                  std::promise<void>& gate) {
  std::unique_ptr<TestJob> gated = job(id, record);
  gated->gate = gate.get_future().share();
  return gated;
}

/** Opens the gate after kGateDelay, on another thread. */
std::thread openLater(std::promise<void>& gate) {
  return std::thread([&gate] {
    std::this_thread::sleep_for(kGateDelay);
    gate.set_value();
  });
}

void testOrder() {
  Record record;
  AsyncOutput output(true);
  const std::vector<int> ids = {0, 1, 2, 3, 4, 5, 6, 7};
  for (const int id : ids)
    output.submit(id % 2 ? OutputLane::Adios2 : OutputLane::Hdf5,
                  job(id, record), 0);
  output.drain();
  std::vector<int> lane[2];
  for (const int id : record.written)
    lane[id % 2].push_back(id);
  check(lane[0] == std::vector<int>({0, 2, 4, 6}) &&
            lane[1] == std::vector<int>({1, 3, 5, 7}),
        "the jobs of a lane are written in the order submitted");
  check(record.finished == ids, "jobs are completed in the order submitted");
  check(!record.writeOnMain && !record.finishOffMain,
        "jobs are written off the main thread and completed on it");
  check(output.deferred() && output.waitedSeconds() > 0.0,
        "the wait for the writers is timed");
}

void testBackpressure() {
  Record record;
  AsyncOutput output(true);
  std::promise<void> gate;
  output.submit(OutputLane::Hdf5, gatedJob(0, record, gate), kParticles);
  output.submit(OutputLane::Adios2, job(1, record), kFields);
  output.waitForReaders(kFields);
  check(record.writtenNow() == std::vector<int>({1}),
        "a copy waits for the jobs that read its buffers, not for the others");
  output.waitLane(OutputLane::Adios2);
  output.poll();
  check(record.finished.empty(),
        "a written job is not completed before an unwritten one submitted "
        "earlier");
  std::thread opener = openLater(gate);
  output.waitForReaders(kParticles);
  check(record.hasWritten(0), "the copy goes on once its readers have written");
  opener.join();
  output.poll();
  check(record.finished == std::vector<int>({0, 1}),
        "a poll completes the written jobs");

  std::promise<void> laneGate;
  output.submit(OutputLane::Hdf5, gatedJob(2, record, laneGate), 0);
  output.waitForReaders(kParticles | kFields);
  output.waitLane(OutputLane::Adios2);
  check(!record.hasWritten(2),
        "a job waits for neither the buffers it does not read nor another "
        "lane");
  opener = openLater(laneGate);
  output.waitLane(OutputLane::Hdf5);
  check(record.hasWritten(2), "a lane's library waits for the lane's jobs");
  opener.join();
  output.drain();
}

void testFinish() {
  Record record;
  AsyncOutput output(true);
  std::unique_ptr<TestJob> polled = job(0, record);
  polled->polls = 2;
  output.submit(OutputLane::Hdf5, std::move(polled), 0);
  output.submit(OutputLane::Hdf5, job(1, record), 0);
  output.waitLane(OutputLane::Hdf5);
  output.poll();
  output.poll();
  check(record.finished.empty(),
        "a job whose finish is not done stays pending, and the jobs after it");
  output.poll();
  check(record.finished == std::vector<int>({0, 1}),
        "a finish is polled until done");
  std::unique_ptr<TestJob> neverPolled = job(2, record);
  neverPolled->polls = kNeverPolled;
  output.submit(OutputLane::Hdf5, std::move(neverPolled), 0);
  output.drain();
  check(record.finished == std::vector<int>({0, 1, 2}),
        "drain completes a job with a blocking finish");
}

/** The error of the failed job 7, as thrown again by wait. */
template <typename Wait> std::string errorOf(Wait wait) {
  Record record;
  AsyncOutput output(true);
  std::unique_ptr<TestJob> failing = job(7, record);
  failing->fail = true;
  output.submit(OutputLane::Adios2, std::move(failing), kParticles);
  try {
    wait(output);
  } catch (const std::runtime_error& error) {
    return error.what();
  }
  return "no error";
}

void testError() {
  const std::string expected = "job 7: injected failure";
  check(errorOf([](AsyncOutput& output) { output.drain(); }) == expected,
        "drain throws a writer's error again with the name of its job");
  check(errorOf([](AsyncOutput& output) {
          output.waitForReaders(kParticles);
        }) == expected,
        "so does the wait of a copy into the buffers the job reads");
  check(errorOf([](AsyncOutput& output) {
          output.waitLane(OutputLane::Adios2);
        }) == expected,
        "so does the wait for its lane");
  check(errorOf([](AsyncOutput& output) {
          for (;;) { // until the job has failed
            output.poll();
            std::this_thread::yield();
          }
        }) == expected,
        "so does a poll");
}

void testSynchronous() {
  Record record;
  AsyncOutput output(false);
  std::unique_ptr<TestJob> neverPolled = job(0, record);
  neverPolled->polls = kNeverPolled;
  output.submit(OutputLane::Hdf5, std::move(neverPolled), kParticles | kFields);
  check(record.written == std::vector<int>({0}) && record.writeOnMain &&
            record.finished == std::vector<int>({0}),
        "not deferred, a job is written and completed by submit");
  output.waitForReaders(kParticles | kFields);
  output.waitLane(OutputLane::Hdf5);
  output.poll();
  output.drain();
  check(!output.deferred() && output.waitedSeconds() == 0.0,
        "not deferred, nothing is waited for");
}

void testDestructor() {
  Record record;
  std::promise<void> gate;
  std::thread opener;
  {
    AsyncOutput output(true);
    output.submit(OutputLane::Hdf5, gatedJob(0, record, gate), 0);
    output.submit(OutputLane::Hdf5, job(1, record), 0);
    opener = openLater(gate);
  }
  check(record.written == std::vector<int>({0, 1}),
        "the destructor waits for the writers to write their jobs");
  opener.join();
}

} // namespace

int main() {
  try {
    testOrder();
    testBackpressure();
    testFinish();
    testError();
    testSynchronous();
    testDestructor();
  } catch (const std::exception& error) {
    std::cerr << "asyncOutputTest failed: " << error.what() << "\n";
    failures++;
  }
  return failures == 0 ? 0 : 1;
}
