/**
 * @file restartOutputFiles.cpp
 * @brief The serial-HDF5 output files on a restart (restartOutputFiles.h).
 */

#include "restartOutputFiles.h"

#ifndef NO_HDF5

#include "hdf5.h"
#include "hdf5_hl.h"
#include "outputPrepare.h" // filesByRank

#include <cstdio>
#include <cstring>

namespace {

/** The HDF5 error stack is not printed while one lives: datasets that may
 *  be missing are looked up, not reported. */
class QuietHDF5 {
public:
  QuietHDF5() {
    H5Eget_auto2(H5E_DEFAULT, &func_, &data_);
    H5Eset_auto2(H5E_DEFAULT, nullptr, nullptr);
  }
  ~QuietHDF5() { H5Eset_auto2(H5E_DEFAULT, func_, data_); }
  QuietHDF5(const QuietHDF5&) = delete;
  QuietHDF5& operator=(const QuietHDF5&) = delete;

private:
  H5E_auto2_t func_ = nullptr;
  void* data_ = nullptr;
};

/** The file at path open for reading for the scope; id < 0 if it cannot be
 *  opened. */
struct ReadOnlyFile {
  explicit ReadOnlyFile(const std::string& path)
      : id(H5Fopen(path.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT)) {}
  ~ReadOnlyFile() {
    if (id >= 0)
      H5Fclose(id);
  }
  ReadOnlyFile(const ReadOnlyFile&) = delete;
  ReadOnlyFile& operator=(const ReadOnlyFile&) = delete;
  const hid_t id;
};

/** No neighbour, MPI_PROC_NULL: a negative rank that differs between MPI
 *  libraries, compared as -1 (coordinates and sizes are never negative). */
int comparable(int value) { return value < 0 ? -1 : value; }

/** An integer dataset and the values it must hold. */
struct ExpectedInts {
  const char* name;
  std::vector<int> values;
};

/** Whether the file at path holds every dataset of expected, of rank 1 with
 *  those values (comparable); nothing if it cannot be opened. */
std::optional<bool> holdsInts(const std::string& path,
                              const std::vector<ExpectedInts>& expected) {
  const QuietHDF5 quiet;
  const ReadOnlyFile file(path);
  if (file.id < 0)
    return std::nullopt;
  for (const ExpectedInts& dataset : expected) {
    const hsize_t n = dataset.values.size();
    int rank = 0;
    hsize_t size = 0;
    H5T_class_t type;
    std::size_t bytes = 0;
    std::vector<int> stored(n);
    // the rank first: the size of a dataset of another rank would overrun
    if (H5LTget_dataset_ndims(file.id, dataset.name, &rank) < 0 || rank != 1 ||
        H5LTget_dataset_info(file.id, dataset.name, &size, &type, &bytes) < 0 ||
        size != n || type != H5T_INTEGER ||
        H5LTread_dataset_int(file.id, dataset.name, stored.data()) < 0)
      return false;
    for (hsize_t k = 0; k < n; k++)
      if (comparable(stored[k]) != comparable(dataset.values[k]))
        return false;
  }
  return true;
}

/** The datasets cycle_<N>, N >= first, that H5Lvisit meets, by path. */
struct CycleDatasets {
  int first;
  std::vector<std::string> paths;
};

herr_t collectCycle(hid_t, const char* path, const H5L_info_t*, void* data) {
  CycleDatasets& found = *static_cast<CycleDatasets*>(data);
  // a link inside a group cycle_<N> collected (visited right after it) goes
  // with the group
  if (!found.paths.empty()) {
    const std::string& last = found.paths.back();
    if (std::strncmp(path, last.c_str(), last.size()) == 0 &&
        path[last.size()] == '/')
      return 0;
  }
  const char* slash = std::strrchr(path, '/');
  int cycle = 0;
  char rest = 0; // a name with anything after the number is another one
  if (std::sscanf(slash ? slash + 1 : path, "cycle_%d%c", &cycle, &rest) == 1 &&
      cycle >= found.first)
    found.paths.emplace_back(path);
  return 0;
}

} // namespace

std::vector<std::pair<int, std::string>>
rankOutputFiles(const std::string& folder) {
  return filesByRank(folder, "proc([0-9]{1,9})\\.hdf");
}

std::optional<bool> writtenAt(const std::string& path,
                              const RankTopology& topology) {
  const std::array<int, 6>& n = topology.neighbours;
  return holdsInts(
      path, {{"/topology/cartesian_coord",
              {topology.coordinates.begin(), topology.coordinates.end()}},
             {"/topology/Xleft_neighbor", {n[0]}},
             {"/topology/Xright_neighbor", {n[1]}},
             {"/topology/Yleft_neighbor", {n[2]}},
             {"/topology/Yright_neighbor", {n[3]}},
             {"/topology/Zleft_neighbor", {n[4]}},
             {"/topology/Zright_neighbor", {n[5]}}});
}

std::optional<bool> sameDecomposition(const std::string& path, int xlen,
                                      int ylen, int zlen) {
  return holdsInts(path, {{"/topology/XLEN", {xlen}},
                          {"/topology/YLEN", {ylen}},
                          {"/topology/ZLEN", {zlen}}});
}

int removeCyclesFrom(const std::string& path, int firstCycle) {
  const hid_t file = H5Fopen(path.c_str(), H5F_ACC_RDWR, H5P_DEFAULT);
  if (file < 0)
    return -1;
  // collected first: no link is removed while the links are visited
  CycleDatasets found{firstCycle, {}};
  // H5Lvisit and H5L_info_t: the versions of the HDF5 library (1.8 on)
  herr_t status =
      H5Lvisit(file, H5_INDEX_NAME, H5_ITER_NATIVE, collectCycle, &found);
  for (std::size_t k = 0; status >= 0 && k < found.paths.size(); k++)
    status = H5Ldelete(file, found.paths[k].c_str(), H5P_DEFAULT);
  if (H5Fclose(file) < 0 || status < 0)
    return -1;
  return static_cast<int>(found.paths.size());
}

#endif // NO_HDF5
