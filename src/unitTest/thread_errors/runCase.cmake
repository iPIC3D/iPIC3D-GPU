# Run the cases of threadErrorsTest on 2 ranks through the launcher of the unit
# tests (cmake -P, with -DLAUNCH=<launch.sh> -DEXE=<test>) and require of each
#   - the error of rank 1, as the std::terminate handler prints it,
#   - the end by MPI_Abort: its report by the MPI library (Open MPI: "MPI_ABORT
#     was invoked", MPICH and its derivatives: "called MPI_Abort") or, under
#     srun, the exit code it gave rank 1: a launcher that kills the job when a
#     rank dies on a signal would hide a rank that only called abort(),
#   - the end of the run within the timeout.

function(run_case name expect)
  execute_process(
    COMMAND sh ${LAUNCH} 2 ${EXE} --case ${name}
    OUTPUT_VARIABLE output
    ERROR_VARIABLE output
    RESULT_VARIABLE result
    TIMEOUT 40)
  message("${output}")
  if(result MATCHES "timeout")
    message(FATAL_ERROR "${name}: the run did not end: a rank waits for the failed one")
  endif()
  if(output MATCHES "no error ended the run")
    message(FATAL_ERROR "${name}: rank 1 raised no error")
  endif()
  if(NOT output MATCHES "Fatal error on rank 1[^\n]*${expect}")
    message(FATAL_ERROR "${name}: no fatal-error message of rank 1 with '${expect}'")
  endif()
  if(NOT output MATCHES
     "MPI_ABORT was invoked|called MPI_Abort|task 1: Exited with exit code")
    message(FATAL_ERROR "${name}: rank 1 did not end the run with MPI_Abort")
  endif()
endfunction()

run_case(task "test task: injected failure")
run_case(thread "injected failure")
