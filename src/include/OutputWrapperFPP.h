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

#ifndef OUTPUT_WRAPPER_FPP_H
#define OUTPUT_WRAPPER_FPP_H
// ===
// OutputWrapperFPP: output wrapper for file-per-process output
//
//   This class should provide a mechanism to avoid having
//   repeatedly opening and closing the same file.
// ===
#include "PSKOutput.h"
#include "ipicfwd.h"
#ifndef NO_HDF5
#include "PSKhdf5adaptor.h"
#endif

struct OutputTagConfig;
struct RestartParticleCellMetadata;

using namespace PSK;

class OutputWrapperFPP {
private:
#ifndef NO_HDF5
  PSK::OutputManager<PSK::OutputAdaptor> output_mgr; // Create an Output Manager
  myOutputAgent<PSK::HDF5OutputAdaptor>
      hdf5_agent; // Create an Output Agent for HDF5 output
#endif            // NO_HDF5
  int cartesian_rank;
  const Collective* col_;
  string SaveDirName;
  string RestartDirName;
  string output_file;
  string restart_file;

public:
  /** With outputFiles (fields, particles or test particles in HDF5), also
   *  start the output file of the rank and, on rank 0, settings.hdf; a
   *  restart from firstCycle continues the files of its folder without
   *  their datasets of firstCycle and later, which it writes again, or keeps
   *  the files of another decomposition aside. With restartFiles (HDF5
   *  checkpoints), rank 0 writes settings.hdf in the restart root too. */
  void init_output_files(Collective* col, VCtopology3D* vct, Grid3DCU* grid,
                         EMfields3D* EMf, ParticleSoAHost** part, int ns,
                         ParticleSoAHost** testpart, int nstestpart,
                         bool outputFiles, bool restartFiles, int firstCycle);
  /** The particle arrays of arrays of the given species at a cycle, from
   *  their views (the particles the output writes), into the output file of
   *  the rank. */
  void append_particles(const ParticleOutputArrays& arrays, int cycle,
                        const std::vector<int>& species,
                        const std::vector<ParticleView>& views);
  /** The same arrays of every test particle at a cycle, into the output
   *  file of the rank. */
  void append_test_particles(const ParticleOutputArrays& arrays, int cycle);
  void append_field_moment_output(const OutputTagConfig& cfg, int cycle);
  void
  setRestartParticleCellMetadata(const RestartParticleCellMetadata* metadata);
  // `cycle` is the restart label: the loop cycle to execute first after
  // restart.
  void append_restart(int cycle, const string& restartDir);
};

#endif // OUTPUT_WRAPPER_FPP_H
