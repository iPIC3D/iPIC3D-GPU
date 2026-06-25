#ifdef USE_ADIOS2

#include "ADIOS2IO.hpp"
#include "VCtopology3D.h"
#include "Grid3DCU.h"
#include "EMfields3D.h"
#include "ParticleSoAHost.h"
#include "Collective.h"

#include "mpi.h"
#include "adios2.h"

#include "MPIdata.h"

#include <chrono>
#include <algorithm> 

namespace ADIOS2IO {

using namespace std;


void ADIOS2Manager::initOutputFiles(string fieldTag, string particleTag, int sample,
                                    Collective* col_in, VCtopology3D* vct_in, Grid3DCU* grid_in,
                                    EMfields3D* EMf_in, ParticleSoAHost** outputPart_in, int ns_in,
                                    ParticleSoAHost** testpart_in, int nstestpart_in) {

    if (open) {
        closeOutputFiles();
    }

    fieldOptions.clear();
    particleOptions.clear();
    restartOptions.clear();

    this->cartisianRank = vct_in->getCartesian_rank();
    this->saveDirName = col_in->getSaveDirName();
    this->restartDirName = col_in->getRestartDirName();
    if (col_in->getRestartOutputCycle()) {
        this->restartTag =
            "proc_topology+E+B+rhos+Js+pressure+position+velocity+q"s;
        if (col_in->anyRegularParticleID()) {
            this->restartTag += "+ID";
        }
        this->restartTag += "+particle_cell_metadata";
    } else {
        this->restartTag = ""s;
    }

    this->fieldTag = fieldTag;
    this->particleTag = particleTag;
    this->sample = sample;

    this->col = col_in;
    this->vct = vct_in;
    this->grid = grid_in;
    this->EMf = EMf_in;

    this->part = outputPart_in;
    this->ns = ns_in;
    this->testpart = testpart_in;
    this->nstestpart = nstestpart_in;


    // ADIOS2
    this->adios = adios2::ADIOS(MPIdata::get_PicGlobalComm());

    // open files
    if (!fieldTag.empty()) { throw runtime_error("Field output is not supported yet"); 
        this->ioField = adios.DeclareIO("FieldOutput");
        this->ioField.SetEngine("BP5");
        auto filePath = saveDirName + "/field_" + to_string(cartisianRank) + ".bp";
        engineField = ioField.Open(filePath, adios2::Mode::Write);

    }

    if (!particleTag.empty()) {
        this->ioParticle = adios.DeclareIO("ParticleOutput");
        this->ioParticle.SetEngine("BP5");
        auto filePath = saveDirName + "/particle_" + to_string(cartisianRank) + ".bp";
        engineParticle = ioParticle.Open(filePath, col->getRestart_status() == 0 ? adios2::Mode::Write : adios2::Mode::Append, MPI_COMM_SELF);

        // parse the tag and prepae the map
        particleTag.erase(remove(particleTag.begin(), particleTag.end(), ' '), particleTag.end());
        vector<string> tags;
        stringstream ss(particleTag);
        string tag;
        while (getline(ss, tag, '+')) {
            tags.push_back(tag);
        }

        // find the function in the map and register it to vector
        for (auto tag : tags) {
            if (outputTagOptions.find(tag) != outputTagOptions.end()) {
                particleOptions.push_back(outputTagOptions[tag]);
            } else {
                throw runtime_error("Particle output tag is not supported: " + tag);
            }
        }


    }


    if (!restartTag.empty()) {
        // parse the tag and prepae the map
        restartTag.erase(remove(restartTag.begin(), restartTag.end(), ' '), restartTag.end());
        vector<string> tags;
        stringstream ss(restartTag);
        string tag;
        while (getline(ss, tag, '+')) {
            tags.push_back(tag);
        }
        // find the function in the map and register it to vector
        for (auto tag : tags) {
            if (outputTagOptions.find(tag) != outputTagOptions.end()) {
                restartOptions.push_back(outputTagOptions[tag]);
            } else {
                throw runtime_error("Restart output tag is not supported: " + tag);
            }
        }

    }


    open = true;

}



void ADIOS2Manager::appendFieldOutput(int cycle) {
    throw runtime_error("Field output is not supported yet");
}

void ADIOS2Manager::appendParticleOutput(int cycle) {
    if (particleOptions.empty()) return;

    engineParticle.BeginStep();

    auto cycleVar = _variableHelper<int>(ioParticle, "cycle");
    engineParticle.Put<int>(cycleVar, cycle);

    // auto timeVar = _variableHelper<int>(ioParticle, "IOTimeMS");
    // auto start = chrono::high_resolution_clock::now();

    for (auto option : particleOptions) {
        option(ioParticle, engineParticle);
    }

    // engineParticle.PerformPuts(); // do the heavy job here

    // auto stop = chrono::high_resolution_clock::now();
    // auto duration = chrono::duration_cast<chrono::milliseconds>(stop - start);
    // engineParticle.Put<int>(timeVar, duration.count());

    engineParticle.EndStep();

}


void ADIOS2Manager::writeRestartOutput(int cycle, const string& restartDir) {
    if (restartOptions.empty()) return;

    const string ioName = "RestartOutput_" + to_string(cartisianRank) + "_"
                        + to_string(restartWriteCount++);
    adios2::IO ioRestart = adios.DeclareIO(ioName);
    ioRestart.SetEngine("BP5");

    const string filePath = restartDir + "/restart_"
                          + to_string(cartisianRank) + ".bp";
    adios2::Engine engineRestart =
        ioRestart.Open(filePath, adios2::Mode::Write, MPI_COMM_SELF);
    engineRestart.BeginStep();

    auto cycleVar = _variableHelper<int>(ioRestart, "cycle");
    engineRestart.Put<int>(cycleVar, cycle);
    for (auto option : restartOptions) {
        option(ioRestart, engineRestart);
    }

    engineRestart.EndStep();
    engineRestart.Close();

}


void ADIOS2Manager::appendOutput(int cycle) {
    if (!open) throw runtime_error("Output files are not open");


    appendParticleOutput(cycle);

    outputCount++;
    lastCycle = cycle;

}


void ADIOS2Manager::closeOutputFiles() {

    if (!open) return;

    if (!fieldTag.empty()) {
        engineField.Close();
    }

    if (!particleTag.empty()) {
        engineParticle.Close();
    }

    open = false;
}







} // end namespace ADIOS2IO

#endif
