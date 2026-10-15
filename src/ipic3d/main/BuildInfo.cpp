#include "BuildInfo.h"

#ifdef _OPENMP
#include <omp.h>
#endif

#if defined(__has_include)
#if __has_include("ipic3d_build_info.h")
#include "ipic3d_build_info.h"
#endif
#endif

// Builds without the generated header (the unit tests compile the sources
// with their own CMake projects) know nothing about their origin.
#ifndef IPIC3D_GIT_COMMIT
#define IPIC3D_GIT_COMMIT "unknown"
#define IPIC3D_GIT_DESCRIBE "unknown"
#define IPIC3D_GIT_BRANCH "unknown"
#define IPIC3D_GIT_DATE "unknown"
#define IPIC3D_GIT_DIRTY -1
#define IPIC3D_GIT_DIRTY_FILES 0
#define IPIC3D_GIT_DIRTY_HASH ""
#define IPIC3D_BUILD_TYPE "unknown"
#define IPIC3D_BUILD_OPTIONS "unknown"
#define IPIC3D_COMPILERS "unknown"
#endif

namespace iPic3D {
namespace build {

namespace {

/** Legacy VTK binary string: a length header (one byte 11xxxxxx below 64
 *  characters, two bytes 10xxxxxx xxxxxxxx below 16384) and the characters. */
std::string vtkBinaryString(std::string s) {
  if (s.size() >= 16384)
    s.resize(16383);
  std::string out;
  if (s.size() < 64) {
    out += static_cast<char>(0xC0 | s.size());
  } else {
    out += static_cast<char>(0x80 | (s.size() >> 8));
    out += static_cast<char>(s.size() & 0xFF);
  }
  return out + s;
}

} // namespace

std::string version() {
  return std::string(IPIC3D_GIT_DESCRIBE) +
         (IPIC3D_GIT_DIRTY == 1 ? "-dirty" : "");
}

std::string commit() { return IPIC3D_GIT_COMMIT; }

int dirty() { return IPIC3D_GIT_DIRTY; }

std::string signature() {
  std::string s = std::string("commit=") + IPIC3D_GIT_COMMIT;
  if (IPIC3D_GIT_DIRTY == 1)
    s += std::string(" dirty=") + IPIC3D_GIT_DIRTY_HASH;
  return s + " build=" + IPIC3D_BUILD_TYPE + " " + IPIC3D_BUILD_OPTIONS +
         " compilers=" + IPIC3D_COMPILERS;
}

void describe(std::ostream& out, const std::string& prefix) {
  const char* state = IPIC3D_GIT_DIRTY == 1   ? "uncommitted changes"
                      : IPIC3D_GIT_DIRTY == 0 ? "clean"
                                              : "unknown";
  out << prefix << "iPIC3D version : " << version() << "\n"
      << prefix << "Commit         : " << IPIC3D_GIT_COMMIT << " ("
      << IPIC3D_GIT_DATE << ", branch " << IPIC3D_GIT_BRANCH << ")\n"
      << prefix << "Working tree   : " << state;
  if (IPIC3D_GIT_DIRTY == 1)
    out << ", " << IPIC3D_GIT_DIRTY_FILES << " file(s), state "
        << IPIC3D_GIT_DIRTY_HASH;
  out << "\n"
      << prefix << "Build type     : " << IPIC3D_BUILD_TYPE << "\n"
      << prefix << "Build options  : " << IPIC3D_BUILD_OPTIONS << "\n"
      << prefix << "Compilers      : " << IPIC3D_COMPILERS << "\n";
  if (IPIC3D_GIT_DIRTY == 1)
    out << prefix << "WARNING: this binary was built from a working tree "
        << "with uncommitted changes;\n"
        << prefix << "         the commit alone does not reproduce it.\n";
}

int openmpThreads() {
#ifdef _OPENMP
  return omp_get_max_threads();
#else
  return 0;
#endif
}

std::string vtkFieldData() {
  return "FIELD FieldData 2\nipic3d_version 1 1 string\n" +
         vtkBinaryString(version()) + "\nipic3d_build 1 1 string\n" +
         vtkBinaryString(signature()) + "\n";
}

} // namespace build
} // namespace iPic3D
