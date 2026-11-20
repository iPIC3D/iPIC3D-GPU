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

#ifndef NO_MPI
#include <mpi.h>
#endif
#include "MPIdata.h"
#include "ompdefs.h" // for omp_get_max_threads
#include <assert.h>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <thread>

// code to check that init() is called before instance()
//
// no need for this to have more than file scope
int MPIdata::rank = -1;
int MPIdata::nprocs = -1;
MPI_Comm MPIdata::PIC_COMM = MPI_COMM_NULL;
static bool MPIdata_is_initialized = false;
bool MPIdata_assert_initialized() {
  assert(MPIdata_is_initialized);
  return true;
}

MPIdata& MPIdata::instance() {
  // This is executed on the first call to check that
  // MPIdata has first been initialized.
  static bool check = MPIdata_assert_initialized();
  static MPIdata* instance = new MPIdata;
  // After the first call, this is the only line
  // that is actually executed.
  return *instance;
}

static std::thread::id mainThread;

/**
 * std::terminate: an exception that no one catches, on any thread (one of a
 * worker is thrown again by the main thread that waits for the task), or a
 * throwing destructor. It prints the error and ends every rank: a rank that
 * stops alone leaves the others waiting in its collectives. With no handler
 * GCC and Clang call it before unwinding the stack, so no destructor waits on
 * a thread or a collective first (only the frames up to a noexcept function
 * or an OpenMP region are unwound). It builds no string: the error may be a
 * failed allocation (the rethrow takes the emergency pool of the C++ runtime
 * then). MPI_Abort from a worker thread, while the main thread may be in an
 * MPI call, is beyond MPI_THREAD_FUNNELED; Open MPI and MPICH accept it.
 */
[[noreturn]] static void abortAllRanks() {
  const char* error = "std::terminate called";
  // keeps the exception, and so its what(), alive
  const std::exception_ptr exception = std::current_exception();
  if (exception) {
    try {
      std::rethrow_exception(exception);
    } catch (const std::exception& e) {
      error = e.what();
    } catch (...) {
      error = "an exception of unknown type";
    }
  }
  const bool onMainThread = std::this_thread::get_id() == mainThread;
  std::fflush(stdout);
  std::fprintf(stderr, "[!] Fatal error on rank %d%s: %s\n",
               MPIdata::get_rank(),
               onMainThread ? "" : " (on a secondary thread)", error);
  std::fflush(stderr);
#ifndef NO_MPI
  int initialized = 0, finalized = 0;
  MPI_Initialized(&initialized);
  MPI_Finalized(&finalized);
  if (initialized && !finalized)
    MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
#endif
  std::abort();
}

void MPIdata::init(int* argc, char*** argv) {
  assert(!MPIdata_is_initialized);
  mainThread = std::this_thread::get_id();
  std::set_terminate(abortAllRanks);

#ifdef NO_MPI
  rank = 0;
  nprocs = 1;
#else // NO_MPI
  /* Initialize the MPI API: only the main thread makes MPI calls, while
   * the thread pool and OpenMP run others */
  int provided;
  MPI_Init_thread(argc, argv, MPI_THREAD_FUNNELED, &provided);

  MPI_Comm_dup(MPI_COMM_WORLD, &PIC_COMM);
  MPI_Comm_rank(PIC_COMM, &rank);
  MPI_Comm_size(PIC_COMM, &nprocs);
  if (rank == 0 && provided < MPI_THREAD_FUNNELED)
    std::fprintf(stderr,
                 "[!] Warning: the MPI library gives the thread level %d, "
                 "below MPI_THREAD_FUNNELED (%d) that this run needs\n",
                 provided, MPI_THREAD_FUNNELED);

#endif // NO_MPI

  MPIdata_is_initialized = true;
}

void MPIdata::exit(int code) {
  finalize_mpi();
  ::exit(code);
}

void MPIdata::finalize_mpi() {
#ifndef NO_MPI
  MPI_Finalize();
#endif
}

void MPIdata::Print(void) {
  printf("\n"
         "Number of processes = %d\n"
         "-------------------------\n"
         "Number of threads = %d\n"
         "-------------------------\n",
         get_nprocs(), omp_get_max_threads());
}

// extern MPIdata *mpi; // instantiated in iPIC3D.cpp
