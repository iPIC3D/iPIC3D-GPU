# The backends of the profiler (src/include/Profiler.h), chosen at compile
# time by three independent options:
#
#   PROFILING   the profiler regions are NVTX ranges on a CUDA build (the NVTX3
#               headers of the CUDA toolkit; cudaProfilerStart/Stop bound the
#               steady-state cycles) and ROCTX ranges on a HIP build
#               (rocprofiler-sdk-roctx, else the legacy roctracer roctx64, of
#               the ROCm of the HIP compiler)
#   CRAYPAT     CrayPat records only the steady-state cycles: PAT_record from
#               pat_api.h and the API stub library of perftools
#               (perftools-base sets CRAYPAT_ROOT); pat_run supplies the
#               PAT_record that runs
#   TIME_TASKS  host stage timers, reported at the end of the run
#
# The INTERFACE target ipic3d_profiling carries their compile definitions,
# include paths and libraries; with the three options off it is empty and the
# profiler compiles to nothing. BENCHMARK_MODE 1 and 2 set TIME_TASKS on, and
# 2 sets the other two off (CMakeLists.txt). Included after the device
# language is enabled.

add_library(ipic3d_profiling INTERFACE)
set(IPIC3D_PROFILER_BACKEND "none")

if(PROFILING)
  if(HIP_ON)
    # ROCTX of the ROCm of the HIP compiler only: the include folder of another
    # ROCm (ROCM_PATH, /opt/rocm) holds its own HIP headers, which would come
    # before the compiler's and do not compile with it. A ROCTX installed
    # elsewhere is given by the cache variables IPIC3D_ROCTX_SDK_INCLUDE_DIR
    # and IPIC3D_ROCTX_SDK_LIBRARY (or the _LEGACY_ ones).
    set(ipic3d_rocm "${CMAKE_HIP_COMPILER_ROCM_ROOT}")
    find_path(IPIC3D_ROCTX_SDK_INCLUDE_DIR NAMES rocprofiler-sdk-roctx/roctx.h
              PATHS "${ipic3d_rocm}" PATH_SUFFIXES include NO_DEFAULT_PATH)
    find_library(IPIC3D_ROCTX_SDK_LIBRARY NAMES rocprofiler-sdk-roctx
                 PATHS "${ipic3d_rocm}" PATH_SUFFIXES lib lib64 NO_DEFAULT_PATH)
    if(IPIC3D_ROCTX_SDK_INCLUDE_DIR AND IPIC3D_ROCTX_SDK_LIBRARY)
      set(ipic3d_roctx SDK)
      set(ipic3d_roctx_name "rocprofiler-sdk-roctx")
    else()
      find_path(IPIC3D_ROCTX_LEGACY_INCLUDE_DIR NAMES roctracer/roctx.h
                PATHS "${ipic3d_rocm}" PATH_SUFFIXES include NO_DEFAULT_PATH)
      find_library(IPIC3D_ROCTX_LEGACY_LIBRARY NAMES roctx64
                   PATHS "${ipic3d_rocm}" PATH_SUFFIXES lib lib64
                   NO_DEFAULT_PATH)
      if(NOT IPIC3D_ROCTX_LEGACY_INCLUDE_DIR OR NOT IPIC3D_ROCTX_LEGACY_LIBRARY)
        message(FATAL_ERROR
                "PROFILING=ON on a HIP build needs the ROCTX of the HIP "
                "compiler's ROCm, ${ipic3d_rocm}: it has neither "
                "rocprofiler-sdk-roctx/roctx.h with librocprofiler-sdk-roctx "
                "nor roctracer/roctx.h with libroctx64. Install one there, or "
                "give one of that ROCm version installed elsewhere with "
                "IPIC3D_ROCTX_SDK_INCLUDE_DIR and IPIC3D_ROCTX_SDK_LIBRARY (or "
                "the _LEGACY_ ones)")
      endif()
      set(ipic3d_roctx LEGACY)
      set(ipic3d_roctx_name "legacy roctracer")
      target_compile_definitions(ipic3d_profiling INTERFACE
                                 IPIC3D_PROFILING_ROCTX_LEGACY)
    endif()
    target_compile_definitions(ipic3d_profiling INTERFACE
                               IPIC3D_PROFILING_ROCTX)
    target_include_directories(ipic3d_profiling INTERFACE
                               "${IPIC3D_ROCTX_${ipic3d_roctx}_INCLUDE_DIR}")
    target_link_libraries(ipic3d_profiling INTERFACE
                          "${IPIC3D_ROCTX_${ipic3d_roctx}_LIBRARY}")
    set(IPIC3D_PROFILER_BACKEND
        "ROCTX (${ipic3d_roctx_name}: ${IPIC3D_ROCTX_${ipic3d_roctx}_LIBRARY})")
    mark_as_advanced(IPIC3D_ROCTX_SDK_INCLUDE_DIR IPIC3D_ROCTX_SDK_LIBRARY
                     IPIC3D_ROCTX_LEGACY_INCLUDE_DIR IPIC3D_ROCTX_LEGACY_LIBRARY)
  else()
    # NVTX3 is header-only; it loads the tool's injection library at run time
    find_path(IPIC3D_NVTX_INCLUDE_DIR NAMES nvtx3/nvToolsExt.h
              HINTS ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES} ENV NVTX3_ROOT
                    ENV CUDAToolkit_ROOT ENV CUDA_HOME
              PATH_SUFFIXES include)
    if(NOT IPIC3D_NVTX_INCLUDE_DIR)
      message(FATAL_ERROR
              "PROFILING=ON on a CUDA build needs the NVTX3 headers "
              "(nvtx3/nvToolsExt.h) of the CUDA toolkit; set NVTX3_ROOT")
    endif()
    target_compile_definitions(ipic3d_profiling INTERFACE IPIC3D_PROFILING_NVTX)
    target_include_directories(ipic3d_profiling INTERFACE
                               "${IPIC3D_NVTX_INCLUDE_DIR}")
    target_link_libraries(ipic3d_profiling INTERFACE ${CMAKE_DL_LIBS})
    set(IPIC3D_PROFILER_BACKEND "NVTX (${IPIC3D_NVTX_INCLUDE_DIR})")
    mark_as_advanced(IPIC3D_NVTX_INCLUDE_DIR)
  endif()
endif()

if(CRAYPAT)
  find_path(IPIC3D_CRAYPAT_INCLUDE_DIR NAMES pat_api.h
            HINTS ENV CRAYPAT_ROOT PATH_SUFFIXES include)
  find_library(IPIC3D_CRAYPAT_LIBRARY NAMES _pat_stubs_api
               HINTS ENV CRAYPAT_ROOT PATH_SUFFIXES lib64 lib)
  if(NOT IPIC3D_CRAYPAT_INCLUDE_DIR OR NOT IPIC3D_CRAYPAT_LIBRARY)
    message(FATAL_ERROR
            "CRAYPAT=ON needs pat_api.h and the stub library _pat_stubs_api "
            "of CrayPat: load perftools-base (it sets CRAYPAT_ROOT) before "
            "configuring")
  endif()
  target_compile_definitions(ipic3d_profiling INTERFACE IPIC3D_CRAYPAT)
  target_include_directories(ipic3d_profiling INTERFACE
                             "${IPIC3D_CRAYPAT_INCLUDE_DIR}")
  target_link_libraries(ipic3d_profiling INTERFACE "${IPIC3D_CRAYPAT_LIBRARY}")
  mark_as_advanced(IPIC3D_CRAYPAT_INCLUDE_DIR IPIC3D_CRAYPAT_LIBRARY)
endif()

if(TIME_TASKS)
  target_compile_definitions(ipic3d_profiling INTERFACE IPIC3D_TIME_TASKS)
endif()
