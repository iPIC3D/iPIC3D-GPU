/* iPIC3D was originally developed by Stefano Markidis and Giovanni Lapenta.
 * This release was contributed by Alec Johnson and Ivy Bo Peng.
 * Publications that use results from iPIC3D need to properly cite
 * 'S. Markidis, G. Lapenta, and Rizwan-uddin. "Multi-scale simulations of
 * plasma with iPIC3D." Mathematics and Computers in Simulation 80.7 (2010):
 * 1509-1519.'
 *
 *        Copyright 2015 KTH Royal Institute of Technology
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *         http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "OutputWrapperFPP.h"
#include "EMfields3D.h"
#include "Grid3DCU.h"
#include "MPIdata.h"
#include "OutputTagConfig.h"
#include "ParticleSoAHost.h"
#include "RestartParticleCellMetadata.h"
#include "VCtopology3D.h"
#include "mpi.h"
#include "outputPrepare.h" // keepAside
#include "restartOutputFiles.h"

#include <algorithm>
#include <cstdio>
#include <filesystem>
#include <optional>
#include <string>
#include <vector>

#ifndef NO_HDF5
namespace {

/** The error of this rank; the ranks then stop together (MPIdata::exit),
 *  each with its files closed. */
void reportError(const std::string& message) {
  std::fprintf(stderr, "ERROR on rank %d: %s\n", MPIdata::get_rank(),
               message.c_str());
}

/**
 * @brief The serial-HDF5 output files in folder (settings.hdf and every
 * proc<rank>.hdf; ownFile the one of this rank) before a restart from
 * firstCycle writes into them (collective). Of this decomposition: every
 * rank removes from its file the datasets of firstCycle and later, which
 * the restart writes again. Of another one (settings.hdf, a file of a rank
 * this run lacks, or the topology in a rank's file says so; a remapped
 * restart): rank 0 keeps all of them aside as *_before_cycle<firstCycle>,
 * and the run starts new files. A rank that cannot read or rewrite a file
 * the run would continue says so and every rank stops, none while it
 * writes its own.
 */
void prepareRestartOutputs(const std::string& folder,
                           const std::string& ownFile, const VCtopology3D* vct,
                           int firstCycle) {
  const MPI_Comm comm = MPIdata::get_PicGlobalComm();
  const bool rank0 = MPIdata::get_rank() == 0;
  const std::string settings = folder + "/settings.hdf";
  // the files: 0 of this decomposition, 1 one of them cannot be read, 2 of
  // another decomposition (kept aside as they are, whether read or not)
  int state = 0;
  std::vector<std::string> files; // rank 0: the output files of the folder
  if (rank0) {
    // settings.hdf missing or unreadable tells nothing: the rank files do
    if (!sameDecomposition(settings, vct->getXLEN(), vct->getYLEN(),
                           vct->getZLEN())
             .value_or(true))
      state = 2;
    files.push_back(settings);
    for (const auto& [rank, path] : rankOutputFiles(folder)) {
      if (rank >= vct->getNprocs())
        state = 2;
      files.push_back(path);
    }
  }
  const RankTopology mine{
      {vct->getCoordinates(0), vct->getCoordinates(1), vct->getCoordinates(2)},
      {vct->getXleft_neighbor(), vct->getXright_neighbor(),
       vct->getYleft_neighbor(), vct->getYright_neighbor(),
       vct->getZleft_neighbor(), vct->getZright_neighbor()}};
  std::error_code error;
  const bool ownExists = std::filesystem::exists(ownFile, error);
  bool unreadable = false;
  if (ownExists) {
    const std::optional<bool> written = writtenAt(ownFile, mine);
    unreadable = !written;
    state = std::max(state, unreadable ? 1 : !*written ? 2 : 0);
  }
  MPI_Allreduce(MPI_IN_PLACE, &state, 1, MPI_INT, MPI_MAX, comm);
  if (state == 1) { // a file that the run would continue
    if (unreadable)
      reportError("cannot open " + ownFile +
                  " to compare its decomposition with this run's (held by "
                  "another program, damaged, or not readable)");
    MPIdata::exit(1);
  }

  if (state == 2) {
    if (rank0) {
      // a rename that fails throws: the run ends with its message
      keepAside(files, firstCycle);
      std::fprintf(stderr,
                   "[!] Warning: the serial-HDF5 output files in %s come "
                   "from another decomposition or periodicity, or do not "
                   "record theirs: they are kept as *_before_cycle%d*.hdf, "
                   "and new ones start\n",
                   folder.c_str(), firstCycle);
    }
    MPI_Barrier(comm); // the files kept aside before a rank starts its own
    return;
  }

  // every rank ends its removal, its file closed, before any stops: a rank
  // stopped while it rewrites its file would leave it damaged
  const int removed = ownExists ? removeCyclesFrom(ownFile, firstCycle) : 0;
  if (removed < 0)
    reportError("cannot remove the datasets of cycle " +
                std::to_string(firstCycle) + " and later from " + ownFile);
  int sums[2] = {removed < 0, std::max(removed, 0)}; // failed ranks, datasets
  MPI_Allreduce(MPI_IN_PLACE, sums, 2, MPI_INT, MPI_SUM, comm);
  if (sums[0] > 0)
    MPIdata::exit(1);
  if (rank0 && sums[1] > 0)
    std::printf("Restart at cycle %d: %d datasets of cycle %d and later "
                "removed from %s/proc*.hdf (the restart writes its outputs "
                "again)\n",
                firstCycle, sums[1], firstCycle, folder.c_str());
}

} // namespace
#endif // NO_HDF5

void OutputWrapperFPP::init_output_files(Collective* col, VCtopology3D* vct,
                                         Grid3DCU* grid, EMfields3D* EMf,
                                         ParticleSoAHost** part, int ns,
                                         ParticleSoAHost** testpart,
                                         int nstestpart, bool outputFiles,
                                         bool restartFiles, int firstCycle) {
#ifndef NO_HDF5
  // The set-up of the HDF5 library here, on the main thread: the library
  // does it at its first call, with MPI calls in a parallel HDF5, and a
  // rank's first call may otherwise be a deferred write on a writer thread,
  // which may make no MPI call (AsyncOutput.h).
  H5open();
  col_ = col;
  cartesian_rank = vct->getCartesian_rank();
  stringstream num_proc_ss;
  num_proc_ss << cartesian_rank;
  string num_proc_str = num_proc_ss.str();
  SaveDirName = col->getSaveDirName();
  RestartDirName = col->getRestartDirName();
  int restart_status = col->getRestart_status();
  output_file = SaveDirName + "/proc" + num_proc_str + ".hdf";

  // Initialize the output (simulation results and restart file)
  hdf5_agent.set_simulation_pointers(EMf, grid, vct, col);

  for (int i = 0; i < ns; ++i) {
    hdf5_agent.set_simulation_pointers_part(part[i]);
  }
  for (int i = 0; i < nstestpart; ++i) {
    hdf5_agent.set_simulation_pointers_part(testpart[i]);
  }

  // Add the HDF5 output agent to the Output Manager's list
  output_mgr.push_back(&hdf5_agent);

  if (outputFiles) {
    // before rank 0 writes settings.hdf again: it tells the decomposition
    if (restart_status != 0)
      prepareRestartOutputs(SaveDirName, output_file, vct, firstCycle);
    if (cartesian_rank == 0) {
      hdf5_agent.open(SaveDirName + "/settings.hdf");
      output_mgr.output("collective + total_topology + proc_topology", 0);
      hdf5_agent.close();
    }

    if (restart_status == 0) {
      hdf5_agent.open(output_file);
    } else {
      hdf5_agent.open_append(output_file);
    }
    output_mgr.output("proc_topology ", 0);
    hdf5_agent.close();
  }

  if (restartFiles &&
      (col->getCallFinalize() || col->getRestartOutputCycle() > 0)) {

    if (cartesian_rank == 0) {
      hdf5_agent.open(RestartDirName + "/settings.hdf");
      output_mgr.output("collective + total_topology + proc_topology", 0);
      hdf5_agent.close();
    }
  }

#endif
}

void OutputWrapperFPP::append_particles(
    const ParticleOutputArrays& arrays, int cycle,
    const std::vector<int>& species, const std::vector<ParticleView>& views) {
#ifndef NO_HDF5
  hdf5_agent.open_append(output_file);
  hdf5_agent.writeParticles(arrays, cycle, species, views);
  hdf5_agent.close();
#endif
}

void OutputWrapperFPP::append_test_particles(const ParticleOutputArrays& arrays,
                                             int cycle) {
#ifndef NO_HDF5
  hdf5_agent.open_append(output_file);
  hdf5_agent.writeTestParticles(arrays, cycle);
  hdf5_agent.close();
#endif
}

void OutputWrapperFPP::append_field_moment_output(const OutputTagConfig& cfg,
                                                  int cycle) {
#ifndef NO_HDF5
  hdf5_agent.open_append(output_file);
  hdf5_agent.outputFieldsMoments(cfg, cycle);
  hdf5_agent.close();
#endif
}

void OutputWrapperFPP::setRestartParticleCellMetadata(
    const RestartParticleCellMetadata* metadata) {
#ifndef NO_HDF5
  hdf5_agent.set_restart_particle_cell_metadata(metadata);
#endif
}

void OutputWrapperFPP::append_restart(int cycle, const string& restartDir) {
#ifndef NO_HDF5
  stringstream num_proc_ss;
  num_proc_ss << cartesian_rank;
  restart_file = restartDir + "/restart" + num_proc_ss.str() + ".hdf";

  hdf5_agent.open(restart_file);
  output_mgr.output("proc_topology ", cycle);
  output_mgr.output("Eall + Ball + rhos + Jsall + pressure", cycle);
  hdf5_agent.writeAllParticles(cycle); // the IDs of the species with IDs
  output_mgr.output("particle_cell_metadata", cycle);
  output_mgr.output("last_cycle", cycle);
  hdf5_agent.close();
#endif
}
