# The registration of the unit tests: every test starts through launch.sh,
# which makes it a job step inside a Slurm allocation.
include_guard(GLOBAL)

# out: the test environment launch.sh reads outside Slurm, the MPI launcher
# CMake found and its flags.
function(unit_test_environment out)
  string(REPLACE ";" " " pre "${MPIEXEC_PREFLAGS}")
  string(REPLACE ";" " " post "${MPIEXEC_POSTFLAGS}")
  set(${out}
      "IPIC3D_MPIEXEC=${MPIEXEC_EXECUTABLE}"
      "IPIC3D_MPIEXEC_NUMPROC_FLAG=${MPIEXEC_NUMPROC_FLAG}"
      "IPIC3D_MPIEXEC_PREFLAGS=${pre}"
      "IPIC3D_MPIEXEC_POSTFLAGS=${post}"
      PARENT_SCOPE)
endfunction()

# add_unit_test(<name> TARGET <target> TIMEOUT <seconds> [RANKS <n>] [GPU]
#               [ARGS <arguments>...])
#   RANKS  MPI ranks of the test; without it, a program without MPI
#   GPU    the test uses the GPU: ctest -j runs one such test at a time
function(add_unit_test name)
  cmake_parse_arguments(PARSE_ARGV 1 TEST "GPU" "TARGET;TIMEOUT;RANKS" "ARGS")
  set(ranks serial)
  set(processors 1)
  if(TEST_RANKS)
    set(ranks ${TEST_RANKS})
    set(processors ${TEST_RANKS})
  endif()
  add_test(NAME ${name}
           COMMAND sh ${CMAKE_CURRENT_FUNCTION_LIST_DIR}/launch.sh ${ranks}
                   $<TARGET_FILE:${TEST_TARGET}> ${TEST_ARGS})
  unit_test_environment(environment)
  set_tests_properties(${name} PROPERTIES TIMEOUT ${TEST_TIMEOUT}
                                          PROCESSORS ${processors}
                                          ENVIRONMENT "${environment}")
  if(TEST_GPU)
    set_tests_properties(${name} PROPERTIES RESOURCE_LOCK ipic3d_gpu)
  endif()
endfunction()
