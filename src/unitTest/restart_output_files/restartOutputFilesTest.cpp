/**
 * @file restartOutputFilesTest.cpp
 * @brief The serial-HDF5 output files on a restart (restartOutputFiles):
 *        removeCyclesFrom removes the datasets cycle_<N> from the restart
 *        cycle on in every group and keeps the others; writtenAt and
 *        sameDecomposition recognise the topology of a rank's file and the
 *        decomposition of settings.hdf; rankOutputFiles lists the rank files
 *        of a folder. CPU only.
 */

#include "restartOutputFiles.h"

#include "hdf5.h"

#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <unistd.h>
#include <utility>
#include <vector>

namespace fs = std::filesystem;

namespace {

int failures = 0;

void check(bool ok, const std::string& what) {
  std::cout << "Test " << (ok ? "passed" : "failed") << ": " << what << "\n";
  failures += !ok;
}

/** A new HDF5 file at path, closed at the end of the scope. */
struct NewFile {
  explicit NewFile(const fs::path& path)
      : id(H5Fcreate(path.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT)) {}
  ~NewFile() { H5Fclose(id); }
  const hid_t id;
};

/** The int dataset at path (its groups created) holding values. */
void put(hid_t file, const std::string& path, const std::vector<int>& values) {
  const hid_t links = H5Pcreate(H5P_LINK_CREATE);
  H5Pset_create_intermediate_group(links, 1);
  const hsize_t size = values.size();
  const hid_t space = H5Screate_simple(1, &size, nullptr);
  const hid_t dataset = H5Dcreate2(file, path.c_str(), H5T_NATIVE_INT, space,
                                   links, H5P_DEFAULT, H5P_DEFAULT);
  H5Dwrite(dataset, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT,
           values.data());
  H5Dclose(dataset);
  H5Sclose(space);
  H5Pclose(links);
}

/** Whether the dataset at path (its groups existing) is in the file. */
bool has(const fs::path& file, const std::string& path) {
  const hid_t id = H5Fopen(file.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
  const bool found = H5Lexists(id, path.c_str(), H5P_DEFAULT) > 0;
  H5Fclose(id);
  return found;
}

void testRemoveCycles(const fs::path& dir) {
  const fs::path file = dir / "proc0.hdf";
  {
    NewFile out(file);
    for (const char* path :
         {"/particles/species_0/x/cycle_0", "/particles/species_0/x/cycle_4",
          "/particles/species_0/x/cycle_8", "/particles/species_1/ID/cycle_4",
          "/fields/Bx/cycle_12", "/moments/rho/cycle_3", "/notes/cycle_4x",
          "/topology/cartesian_coord", "/other/cycle_9/cycle_10"})
      put(out.id, path, {1, 2, 3});
  }
  check(removeCyclesFrom(file.string(), 4) == 5,
        "removeCyclesFrom 4: the four datasets of cycles 4, 8 and 12, and the "
        "group cycle_9 with the dataset cycle_10 in it");
  check(!has(file, "/particles/species_0/x/cycle_4") &&
            !has(file, "/particles/species_0/x/cycle_8") &&
            !has(file, "/particles/species_1/ID/cycle_4") &&
            !has(file, "/fields/Bx/cycle_12") && !has(file, "/other/cycle_9"),
        "removeCyclesFrom 4: those removed, in every group");
  check(has(file, "/particles/species_0/x/cycle_0") &&
            has(file, "/moments/rho/cycle_3") && has(file, "/notes/cycle_4x") &&
            has(file, "/topology/cartesian_coord"),
        "removeCyclesFrom 4: the earlier cycles and the other datasets kept");
  check(removeCyclesFrom(file.string(), 4) == 0,
        "removeCyclesFrom again: nothing more");
  // the failed open of the missing file not printed by HDF5 (it would be in
  // a run, before the run stops)
  H5E_auto2_t printer = nullptr;
  void* printerData = nullptr;
  H5Eget_auto2(H5E_DEFAULT, &printer, &printerData);
  H5Eset_auto2(H5E_DEFAULT, nullptr, nullptr);
  check(removeCyclesFrom((dir / "missing.hdf").string(), 0) == -1,
        "removeCyclesFrom of a missing file: -1");
  H5Eset_auto2(H5E_DEFAULT, printer, printerData);
}

void testTopology(const fs::path& dir) {
  const RankTopology rank{{1, 0, 2}, {3, 5, -2, -2, 7, 9}};
  const fs::path file = dir / "proc5.hdf";
  {
    NewFile out(file);
    put(out.id, "/topology/cartesian_coord", {1, 0, 2});
    const char* names[6] = {"Xleft",  "Xright", "Yleft",
                            "Yright", "Zleft",  "Zright"};
    for (int k = 0; k < 6; k++)
      put(out.id, std::string("/topology/") + names[k] + "_neighbor",
          {rank.neighbours[k]});
  }
  check(writtenAt(file.string(), rank) == true,
        "writtenAt: the rank's topology");
  RankTopology otherMpi = rank; // MPI_PROC_NULL of another MPI library
  otherMpi.neighbours[2] = otherMpi.neighbours[3] = -1;
  check(writtenAt(file.string(), otherMpi) == true,
        "writtenAt: no neighbour, whatever its negative value");
  RankTopology moved = rank;
  moved.coordinates[2] = 0;
  check(writtenAt(file.string(), moved) == false,
        "writtenAt: other coordinates");
  RankTopology rewired = rank;
  rewired.neighbours[1] = 4;
  check(writtenAt(file.string(), rewired) == false,
        "writtenAt: another neighbour");
  const fs::path bare = dir / "proc6.hdf";
  {
    NewFile out(bare);
    put(out.id, "/fields/Bx/cycle_0", {0});
  }
  check(writtenAt(bare.string(), rank) == false,
        "writtenAt: a file without topology");
  const fs::path text = dir / "proc7.hdf";
  std::ofstream(text) << "not an HDF5 file";
  check(!writtenAt(text.string(), rank).has_value() &&
            !writtenAt((dir / "missing.hdf").string(), rank).has_value(),
        "writtenAt: nothing for a file that cannot be opened");
}

void testDecomposition(const fs::path& dir) {
  const fs::path file = dir / "settings.hdf";
  {
    NewFile out(file);
    put(out.id, "/topology/XLEN", {2});
    put(out.id, "/topology/YLEN", {2});
    put(out.id, "/topology/ZLEN", {1});
  }
  check(sameDecomposition(file.string(), 2, 2, 1) == true,
        "sameDecomposition: 2 x 2 x 1");
  check(sameDecomposition(file.string(), 4, 2, 1) == false,
        "sameDecomposition: not 4 x 2 x 1");
  check(!sameDecomposition((dir / "missing.hdf").string(), 4, 2, 1).has_value(),
        "sameDecomposition: nothing for a missing file");
  const fs::path bare = dir / "bare.hdf";
  {
    NewFile out(bare);
  }
  check(sameDecomposition(bare.string(), 2, 2, 1) == false,
        "sameDecomposition: a file without topology");
  const fs::path text = dir / "text.hdf";
  std::ofstream(text) << "not an HDF5 file";
  check(!sameDecomposition(text.string(), 2, 2, 1).has_value(),
        "sameDecomposition: nothing for a file that cannot be opened");
}

void testRankFiles(const fs::path& dir) {
  for (const char* name :
       {"proc0.hdf", "proc12.hdf", "proc3.hdf.bak", "procx.hdf", "settings.hdf",
        "proc7_before_cycle4.hdf"})
    std::ofstream(dir / name).put('x');
  const std::vector<std::pair<int, std::string>> expected = {
      {0, (dir / "proc0.hdf").string()}, {12, (dir / "proc12.hdf").string()}};
  check(rankOutputFiles(dir.string()) == expected,
        "rankOutputFiles: proc0.hdf and proc12.hdf, by rank");
  check(rankOutputFiles((dir / "missing").string()).empty(),
        "rankOutputFiles: a missing folder");
}

} // namespace

int main() {
  const fs::path dir = fs::temp_directory_path() /
                       ("restartOutputFilesTest_" + std::to_string(getpid()));
  const char* parts[] = {"cycles", "topology", "decomposition", "ranks"};
  for (const char* part : parts)
    fs::create_directories(dir / part);
  testRemoveCycles(dir / parts[0]);
  testTopology(dir / parts[1]);
  testDecomposition(dir / parts[2]);
  testRankFiles(dir / parts[3]);
  fs::remove_all(dir);
  return failures == 0 ? 0 : 1;
}
