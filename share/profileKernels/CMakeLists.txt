# Kernel profiles: one executable per GPU kernel, and the measured ceilings of
# the rooflines (README.md).
#
# Added by the main CMakeLists.txt with -DBUILD_PROFILE_KERNELS=ON.  The
# executables launch kernels of the iPIC3D library themselves, which works
# only when the library is linked into them: with the shared library, nvcc
# hides the launch functions of its kernels from other binaries (CUDA 13), or
# the launch reaches a second copy of the static CUDA runtime and does nothing
# (CUDA 12).  The kernels are the objects of the iPIC3Dlib target, compiled
# with the flags of the solver.

if(BUILD_SHARED_LIBS)
  message(FATAL_ERROR
          "BUILD_PROFILE_KERNELS needs -DBUILD_SHARED_LIBS=OFF: the kernel "
          "profiles launch kernels of the iPIC3D library, which works only "
          "when the library is linked into them (README.md)")
endif()

# the kernel profiles, and the measured ceilings of the rooflines
set(kernel_profiles profileMover profileCellAwareMoment profileMomentKernelNew
                    profileCeilings)

add_library(kernelProfile STATIC KernelProfile.cu)
target_link_libraries(kernelProfile PUBLIC iPIC3Dlib)
target_include_directories(kernelProfile PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

foreach(profile IN LISTS kernel_profiles)
  add_executable(${profile} ${profile}.cu)
  target_link_libraries(${profile} PRIVATE kernelProfile)
  set_target_properties(${profile} PROPERTIES RUNTIME_OUTPUT_DIRECTORY
                                              ${CMAKE_BINARY_DIR})
endforeach()

set(profile_targets kernelProfile ${kernel_profiles})
if(HIP_ON)
  foreach(target IN LISTS profile_targets)
    get_target_property(sources ${target} SOURCES)
    set_source_files_properties(${sources} PROPERTIES LANGUAGE HIP)
    set_property(TARGET ${target} PROPERTY HIP_ARCHITECTURES ${HIP_ARCH})
  endforeach()
  # the device code of the static library is linked into each executable
  foreach(profile IN LISTS kernel_profiles)
    target_link_options(${profile} PRIVATE -fgpu-rdc --hip-link)
  endforeach()
else()
  foreach(target IN LISTS profile_targets)
    set_target_properties(${target} PROPERTIES CUDA_SEPARABLE_COMPILATION ON
                                               CUDA_ARCHITECTURES ${CUDA_ARCH})
  endforeach()
endif()
