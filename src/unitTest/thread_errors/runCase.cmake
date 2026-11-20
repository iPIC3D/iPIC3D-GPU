# Run one case of threadErrorsTest on 2 ranks (cmake -P, with -DMPIEXEC=...,
# -DNUMPROC_FLAG=..., -DPREFLAGS=..., -DPOSTFLAGS=..., -DEXE=..., -DCASE=...,
# -DEXPECT=<the error of rank 1>) and require
#   - the error of rank 1, as the std::terminate handler prints it,
#   - the report of MPI_Abort by the MPI library (Open MPI: "MPI_ABORT was
#     invoked", MPICH and its derivatives: "called MPI_Abort"): a launcher
#     that kills the job when a rank dies on a signal would hide a rank that
#     only called abort(),
#   - the end of the run within the timeout.
separate_arguments(PREFLAGS)
separate_arguments(POSTFLAGS)
execute_process(
  COMMAND ${MPIEXEC} ${NUMPROC_FLAG} 2 ${PREFLAGS} ${EXE} ${POSTFLAGS} --case
          ${CASE}
  OUTPUT_VARIABLE output
  ERROR_VARIABLE output
  RESULT_VARIABLE result
  TIMEOUT 50)
message("${output}")
if(result MATCHES "timeout")
  message(FATAL_ERROR "the run did not end: a rank waits for the failed one")
endif()
if(output MATCHES "no error ended the run")
  message(FATAL_ERROR "rank 1 raised no error")
endif()
if(NOT output MATCHES "Fatal error on rank 1[^\n]*${EXPECT}")
  message(FATAL_ERROR "no fatal-error message of rank 1 with '${EXPECT}'")
endif()
if(NOT output MATCHES "MPI_ABORT was invoked|called MPI_Abort")
  message(FATAL_ERROR "rank 1 did not end the run with MPI_Abort")
endif()
