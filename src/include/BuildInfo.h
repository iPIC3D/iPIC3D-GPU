/**
 * @file BuildInfo.h
 * @brief Identity of the iPIC3D build that produced a run.
 *
 * CMake records the git commit, whether the working tree had uncommitted
 * changes (and a hash of them), the build type, the build options and the
 * compilers (cmake/GenerateBuildInfo.cmake). The identity is written to the run
 * log and SimulationData.txt, to the restart metadata (a restart with a
 * different build gives a warning) and inside every output file, so a run and
 * each of its files name the code that made them. Builds without the generated
 * header (e.g. the unit tests) report "unknown".
 */
#ifndef IPIC3D_BUILD_INFO_H
#define IPIC3D_BUILD_INFO_H

#include <ostream>
#include <string>

namespace iPic3D {
namespace build {

/** git describe of the build commit, with "-dirty" if the working tree had
 *  uncommitted changes, e.g. "9ab73ad6c0" or "9ab73ad6c0-dirty". */
std::string version();
/** Full commit hash, or "unknown". */
std::string commit();
/** 1: built from a tree with uncommitted changes, 0: clean, -1: unknown. */
int dirty();
/** One line that identifies the binary: commit, hash of the uncommitted
 *  changes, build type, options and compilers. Compared on restart. */
std::string signature();
/** Multi-line description, every line starting with prefix; ends with a
 *  warning paragraph if the build is dirty. */
void describe(std::ostream& out, const std::string& prefix = "");
/** OpenMP threads per rank, 0 if built without OpenMP. Here because the CUDA
 *  sources are not compiled with the OpenMP flags. */
int openmpThreads();
/** Legacy VTK dataset field data (FIELD block, binary strings) holding
 *  "ipic3d_version" and "ipic3d_build"; goes between the dataset geometry and
 *  POINT_DATA of a BINARY STRUCTURED_POINTS file. */
std::string vtkFieldData();

} // namespace build
} // namespace iPic3D

#endif // IPIC3D_BUILD_INFO_H
