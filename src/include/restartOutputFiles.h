#ifndef RESTART_OUTPUT_FILES_H
#define RESTART_OUTPUT_FILES_H

/**
 * @file restartOutputFiles.h
 * @brief The serial-HDF5 output files of a run (proc<rank>.hdf and
 * settings.hdf, OutputWrapperFPP) on a restart in its folder: the datasets of
 * the cycles the restart writes again are removed, so that each file holds
 * one history; files of another decomposition (a remapped restart) are
 * recognised, to be kept aside (keepAside).
 */

#include <array>
#include <optional>
#include <string>
#include <utility>
#include <vector>

/** The place of a rank in its decomposition, as the proc_topology output
 *  writes it into the rank's file: its Cartesian coordinates and its
 *  neighbours (X left, X right, Y left, Y right, Z left, Z right). */
struct RankTopology {
  std::array<int, 3> coordinates;
  std::array<int, 6> neighbours;
};

/** The output files proc<rank>.hdf in folder, by rank. */
std::vector<std::pair<int, std::string>>
rankOutputFiles(const std::string& folder);

/** Whether the output file at path was written by the rank at topology (its
 *  /topology datasets; no neighbour, MPI_PROC_NULL, counts as any negative
 *  value, which differs between MPI libraries): false without them, nothing
 *  if the file cannot be opened. */
std::optional<bool> writtenAt(const std::string& path,
                              const RankTopology& topology);

/** Whether settings.hdf at path was written on the decomposition xlen ylen
 *  zlen (its /topology datasets): false without the datasets, nothing if
 *  the file is missing or cannot be opened. */
std::optional<bool> sameDecomposition(const std::string& path, int xlen,
                                      int ylen, int zlen);

/** Remove from the HDF5 file at path every dataset cycle_<N> with N >=
 *  firstCycle, in whatever group (a group cycle_<N> with what it holds):
 *  the outputs a restart from firstCycle writes again. HDF5 may keep the
 *  space they took (h5repack frees it).
 *  @return how many, -1 if the file cannot be opened or a removal fails. */
int removeCyclesFrom(const std::string& path, int firstCycle);

#endif // RESTART_OUTPUT_FILES_H
