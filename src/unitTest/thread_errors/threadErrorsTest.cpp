/**
 * @file threadErrorsTest.cpp
 * @brief An error on rank 1 ends every rank, whatever thread raises it,
 *        while rank 0 waits in a collective.
 *
 * Cases (--case <name>), run on 2 ranks; rank 1 throws, rank 0 waits in
 * MPI_Barrier for it:
 *   task     in a ThreadPool task, thrown again on the main thread by
 *            awaitTask, as the analysis, mover and injection tasks
 *   thread   on a secondary thread, where nothing catches it
 * runCase.cmake requires the error of rank 1 (std::terminate; MPIdata), the
 * report of MPI_Abort and the end of the run: a rank that stopped alone would
 * leave rank 0 in the barrier under a launcher that does not kill the job.
 */

#include <cstdio>
#include <stdexcept>
#include <string>
#include <thread>

#include "MPIdata.h"
#include "threadPool.hpp"

namespace {

[[noreturn]] void fail() { throw std::runtime_error("injected failure"); }

void raise(const std::string& kind) {
  if (kind == "task") {
    ThreadPool pool(1);
    std::future<int> result = pool.enqueue([]() -> int { fail(); });
    awaitTask(result, "test task");
  } else if (kind == "thread") {
    std::thread worker(fail);
    worker.join();
  } else {
    std::printf("unknown case %s\n", kind.c_str());
  }
}

} // namespace

int main(int argc, char** argv) {
  MPIdata::init(&argc, &argv);
  std::string kind;
  for (int i = 1; i + 1 < argc; i++)
    if (std::string(argv[i]) == "--case")
      kind = argv[i + 1];
  if (MPIdata::get_nprocs() < 2)
    std::printf("run on 2 ranks\n");
  else if (MPIdata::get_rank() == 1)
    raise(kind);
  MPI_Barrier(MPI_COMM_WORLD); // rank 0 waits here for rank 1
  std::printf("rank %d: no error ended the run\n", MPIdata::get_rank());
  MPIdata::finalize_mpi();
  return 0;
}
