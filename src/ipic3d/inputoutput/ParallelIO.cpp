/* iPIC3D was originally developed by Stefano Markidis and Giovanni Lapenta. 
 * This release was contributed by Alec Johnson and Ivy Bo Peng.
 * Publications that use results from iPIC3D need to properly cite  
 * 'S. Markidis, G. Lapenta, and Rizwan-uddin. "Multi-scale simulations of 
 * plasma with iPIC3D." Mathematics and Computers in Simulation 80.7 (2010): 1509-1519.'
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

#include <mpi.h>
#include <fstream>

#include "ParallelIO.h"
#include "OutputTagConfig.h"
#include "MPIdata.h"
#include "debug.h"
#include "TimeTasks.h"
#include "Collective.h"
#include "Grid3DCU.h"
#include "VCtopology3D.h"
#include "ParticleSoAHost.h"
#include "EMfields3D.h"
#include "math.h"
#include <algorithm>
#include <iomanip>
#include <iostream>
#include <sstream>

/*! Function used to write the EM fields using the parallel HDF5 library */
void WriteOutputParallel(Grid3DCU *grid, EMfields3D *EMf, CollectiveIO *col, VCtopology3D *vct, int cycle, const OutputTagConfig& cfg){

#ifdef PHDF5
  timeTasks_set_task(TimeTasks::WRITE_FIELDS);

  stringstream filenmbr;
  string       filename;

  /* ------------------- */
  /* Setup the file name */
  /* ------------------- */

  filenmbr << setfill('0') << setw(5) << cycle;
  filename = col->getSaveDirName() + "/" + col->getSimName() + "_" + filenmbr.str() + ".h5";

  /* ---------------------------------------------------------------------------- */
  /* Define the number of cells in the globa and local mesh and set the mesh size */
  /* ---------------------------------------------------------------------------- */

  int nxc = grid->getNXC();
  int nyc = grid->getNYC();
  int nzc = grid->getNZC();

  int    dglob[3] = { col ->getNxc()  , col ->getNyc()  , col ->getNzc()   };
  int    dlocl[3] = { nxc-2,            nyc-2,            nzc-2 };
  double L    [3] = { col ->getLx ()  , col ->getLy ()  , col ->getLz ()   };

  /* --------------------------------------- */
  /* Declare and open the parallel HDF5 file */
  /* --------------------------------------- */

  PHDF5fileClass outputfile(filename, 3, vct->getCoordinates(), vct->getFieldComm());

  const bool bp = (col->getParticlesOutputCycle() > 0);

  outputfile.CreatePHDF5file(L, dglob, dlocl, bp);

  /* ----------------------------------------------------------------- */
  /* All quantities on the node grid: global = Nxc+1,                  */
  /* interior local = nxc-2, upper-boundary local = nxc-1, data from 1 */
  /*                                                                   */
  /* This matches the VTK convention:                                  */
  /*   VTK lx = nxn - 3 = nxc - 2  (interior)                         */
  /*   VTK lx = nxn - 3 + 1 = nxc - 1  (upper boundary)              */
  /* ----------------------------------------------------------------- */
  const bool xu = vct->isXupper();
  const bool yu = vct->isYupper();
  const bool zu = vct->isZupper();
  const hsize_t gdim[3] = { (hsize_t)(col->getNxc()+1),
                            (hsize_t)(col->getNyc()+1),
                            (hsize_t)(col->getNzc()+1) };
  const hsize_t ldim[3] = { (hsize_t)(nxc-2 + (xu?1:0)),
                            (hsize_t)(nyc-2 + (yu?1:0)),
                            (hsize_t)(nzc-2 + (zu?1:0)) };
  const hsize_t foff[3] = { (hsize_t)(vct->getCoordinates(0)*(nxc-2)),
                            (hsize_t)(vct->getCoordinates(1)*(nyc-2)),
                            (hsize_t)(vct->getCoordinates(2)*(nzc-2)) };

  const int ns = col->getNs();

  // --- B field ---
  if (cfg.writeB) {
    outputfile.WritePHDF5dataset("Fields", "Bx", EMf->getBxTot(), 1,1,1, gdim, ldim, foff);
    outputfile.WritePHDF5dataset("Fields", "By", EMf->getByTot(), 1,1,1, gdim, ldim, foff);
    outputfile.WritePHDF5dataset("Fields", "Bz", EMf->getBzTot(), 1,1,1, gdim, ldim, foff);
  }

  // --- E field ---
  if (cfg.writeE) {
    outputfile.WritePHDF5dataset("Fields", "Ex", EMf->getEx(), 1,1,1, gdim, ldim, foff);
    outputfile.WritePHDF5dataset("Fields", "Ey", EMf->getEy(), 1,1,1, gdim, ldim, foff);
    outputfile.WritePHDF5dataset("Fields", "Ez", EMf->getEz(), 1,1,1, gdim, ldim, foff);
  }

  // --- Per-species rho ---
  for (int si : cfg.rhoSpecies) {
    outputfile.WritePHDF5dataset("Fields", "rho_"+std::to_string(si),
      EMf->getRHOns(si), 1,1,1, gdim, ldim, foff, 4*3.1415926535897);
  }

  // --- Total rho ---
  if (cfg.writeRhoTot) {
    outputfile.WritePHDF5dataset("Fields", "rho_tot",
      EMf->getRHOn(), 1,1,1, gdim, ldim, foff);
  }

  // --- Per-species J ---
  for (int si : cfg.JSpecies) {
    string s = std::to_string(si);
    outputfile.WritePHDF5dataset("Fields", "Jx_"+s, EMf->getJxs(si), 1,1,1, gdim, ldim, foff);
    outputfile.WritePHDF5dataset("Fields", "Jy_"+s, EMf->getJys(si), 1,1,1, gdim, ldim, foff);
    outputfile.WritePHDF5dataset("Fields", "Jz_"+s, EMf->getJzs(si), 1,1,1, gdim, ldim, foff);
  }

  // --- Total J ---
  if (cfg.writeJTot) {
    EMf->sumOverSpeciesJ();
    outputfile.WritePHDF5dataset("Fields", "Jx_tot", EMf->getJx(), 1,1,1, gdim, ldim, foff);
    outputfile.WritePHDF5dataset("Fields", "Jy_tot", EMf->getJy(), 1,1,1, gdim, ldim, foff);
    outputfile.WritePHDF5dataset("Fields", "Jz_tot", EMf->getJz(), 1,1,1, gdim, ldim, foff);
  }

  // --- Per-species pressure tensor ---
  {
    struct PDesc { const char* name; const std::set<int>& species;
                   arr3_double (EMfields3D::*getter)(int); };
    PDesc pdescs[] = {
      {"pXX", cfg.PXXSpecies, &EMfields3D::getpXXsn},
      {"pXY", cfg.PXYSpecies, &EMfields3D::getpXYsn},
      {"pXZ", cfg.PXZSpecies, &EMfields3D::getpXZsn},
      {"pYY", cfg.PYYSpecies, &EMfields3D::getpYYsn},
      {"pYZ", cfg.PYZSpecies, &EMfields3D::getpYZsn},
      {"pZZ", cfg.PZZSpecies, &EMfields3D::getpZZsn},
    };
    for (auto &pd : pdescs)
      for (int si : pd.species)
        outputfile.WritePHDF5dataset("Fields",
          string(pd.name)+"_"+std::to_string(si),
          (EMf->*(pd.getter))(si), 1,1,1, gdim, ldim, foff);
  }

  // --- Total pressure tensor ---
  if (cfg.needsAnyPTot()) {
    const int nxn = grid->getNXN();
    const int nyn = grid->getNYN();
    const int nzn = grid->getNZN();
    // Temporary buffer sized for full local node array (including ghosts).
    // We write from index 1 via the WritePHDF5dataset(i0=1) overload.
    arr3_double tmp(nxn, nyn, nzn);

    auto writePTot = [&](const char* name, bool doWrite,
                         double (EMfields3D::*g4)(int,int,int,int) const) {
      if (!doWrite) return;
      for (int i = 0; i < (int)ldim[0]; i++)
        for (int j = 0; j < (int)ldim[1]; j++)
          for (int k = 0; k < (int)ldim[2]; k++) {
            double sum = 0.0;
            for (int s = 0; s < ns; s++)
              sum += (EMf->*g4)(i+1, j+1, k+1, s);
            tmp[i+1][j+1][k+1] = sum;
          }
      outputfile.WritePHDF5dataset("Fields", string(name)+"_tot",
                                   tmp, 1,1,1, gdim, ldim, foff);
    };

    writePTot("pXX", cfg.writePXXTot, &EMfields3D::getpXXsn);
    writePTot("pXY", cfg.writePXYTot, &EMfields3D::getpXYsn);
    writePTot("pXZ", cfg.writePXZTot, &EMfields3D::getpXZsn);
    writePTot("pYY", cfg.writePYYTot, &EMfields3D::getpYYsn);
    writePTot("pYZ", cfg.writePYZTot, &EMfields3D::getpYZsn);
    writePTot("pZZ", cfg.writePZZTot, &EMfields3D::getpZZsn);
  }

  // --- Per-species heat-flux tensor ---
  for (int c = 0; c < HeatFlux::ComponentCount; ++c) {
    for (int si : cfg.heatFluxSpecies[c]) {
      outputfile.WritePHDF5dataset("Fields",
        string(HeatFlux::ComponentNames[c]) + "_" + std::to_string(si),
        EMf->getHeatFluxComponent(si, c), 1,1,1, gdim, ldim, foff);
    }
  }

  // --- Total heat-flux tensor ---
  if (cfg.needsAnyHeatFluxTot()) {
    const int nxn = grid->getNXN();
    const int nyn = grid->getNYN();
    const int nzn = grid->getNZN();
    arr3_double tmp(nxn, nyn, nzn);

    for (int c = 0; c < HeatFlux::ComponentCount; ++c) {
      if (!cfg.writeHeatFluxTot[c]) continue;
      for (int i = 0; i < (int)ldim[0]; i++)
        for (int j = 0; j < (int)ldim[1]; j++)
          for (int k = 0; k < (int)ldim[2]; k++) {
            double sum = 0.0;
            for (int s = 0; s < ns; s++)
              sum += EMf->getHeatFlux(i+1, j+1, k+1, s, c);
            tmp[i+1][j+1][k+1] = sum;
          }
      outputfile.WritePHDF5dataset("Fields",
        string(HeatFlux::ComponentNames[c]) + "_tot",
        tmp, 1,1,1, gdim, ldim, foff);
    }
  }

  outputfile.ClosePHDF5file();

#else  
  eprintf(
    " The input file requests the use of the Parallel HDF5 functions,\n"
    " but the code has been compiled using the sequential HDF5 library.\n"
    " Recompile the code using the parallel HDF5 options\n"
    " or change the input file options. ");
#endif

}

/*! Function to write the EM fields using the H5hut library. */
void WriteFieldsH5hut(int nspec, Grid3DCU *grid, EMfields3D *EMf, CollectiveIO *col, VCtopology3D *vct, int cycle, const OutputTagConfig& cfg){
  if(col->field_output_is_off())
    return;
#ifdef USEH5HUT
  timeTasks_set_task(TimeTasks::WRITE_FIELDS);

  H5output file;

  string filename = col->getSaveDirName() + "/" + col->getSimName();

  file.SetNameCycle(filename, cycle);

  file.OpenFieldsFile("Node", nspec, col->getNxc()+1, col->getNyc()+1, col->getNzc()+1, vct->getCoordinates(), vct->getDims(), vct->getFieldComm());

  const int nxn = grid->getNXN();
  const int nyn = grid->getNYN();
  const int nzn = grid->getNZN();
  const int ns  = col->getNs();

  // --- B field (includes B_ext) ---
  if (cfg.writeB) {
    file.WriteFields(EMf->getBxTot(), "Bx", nxn, nyn, nzn);
    file.WriteFields(EMf->getByTot(), "By", nxn, nyn, nzn);
    file.WriteFields(EMf->getBzTot(), "Bz", nxn, nyn, nzn);
  }

  // --- E field ---
  if (cfg.writeE) {
    file.WriteFields(EMf->getEx(), "Ex", nxn, nyn, nzn);
    file.WriteFields(EMf->getEy(), "Ey", nxn, nyn, nzn);
    file.WriteFields(EMf->getEz(), "Ez", nxn, nyn, nzn);
  }

  // --- Per-species rho ---
  for (int si : cfg.rhoSpecies) {
    file.WriteFields(EMf->getRHOns(si), "rho_" + std::to_string(si), nxn, nyn, nzn);
  }

  // --- Total rho ---
  if (cfg.writeRhoTot) {
    file.WriteFields(EMf->getRHOn(), "rho_tot", nxn, nyn, nzn);
  }

  // --- Per-species J ---
  for (int si : cfg.JSpecies) {
    string s = std::to_string(si);
    file.WriteFields(EMf->getJxs(si), "Jx_" + s, nxn, nyn, nzn);
    file.WriteFields(EMf->getJys(si), "Jy_" + s, nxn, nyn, nzn);
    file.WriteFields(EMf->getJzs(si), "Jz_" + s, nxn, nyn, nzn);
  }

  // --- Total J ---
  if (cfg.writeJTot) {
    EMf->sumOverSpeciesJ();
    file.WriteFields(EMf->getJx(), "Jx_tot", nxn, nyn, nzn);
    file.WriteFields(EMf->getJy(), "Jy_tot", nxn, nyn, nzn);
    file.WriteFields(EMf->getJz(), "Jz_tot", nxn, nyn, nzn);
  }

  // --- Per-species pressure tensor ---
  {
    struct PDesc { const char* name; const std::set<int>& species;
                   arr3_double (EMfields3D::*getter)(int); };
    PDesc pdescs[] = {
      {"pXX", cfg.PXXSpecies, &EMfields3D::getpXXsn},
      {"pXY", cfg.PXYSpecies, &EMfields3D::getpXYsn},
      {"pXZ", cfg.PXZSpecies, &EMfields3D::getpXZsn},
      {"pYY", cfg.PYYSpecies, &EMfields3D::getpYYsn},
      {"pYZ", cfg.PYZSpecies, &EMfields3D::getpYZsn},
      {"pZZ", cfg.PZZSpecies, &EMfields3D::getpZZsn},
    };
    for (auto &pd : pdescs)
      for (int si : pd.species)
        file.WriteFields((EMf->*(pd.getter))(si),
          string(pd.name) + "_" + std::to_string(si), nxn, nyn, nzn);
  }

  // --- Total pressure tensor ---
  if (cfg.needsAnyPTot()) {
    arr3_double tmp(nxn, nyn, nzn);

    auto writePTot = [&](const char* name, bool doWrite,
                         double (EMfields3D::*g4)(int,int,int,int) const) {
      if (!doWrite) return;
      for (int i = 0; i < nxn; i++)
        for (int j = 0; j < nyn; j++)
          for (int k = 0; k < nzn; k++) {
            double sum = 0.0;
            for (int s = 0; s < ns; s++)
              sum += (EMf->*g4)(i, j, k, s);
            tmp[i][j][k] = sum;
          }
      file.WriteFields(tmp, string(name) + "_tot", nxn, nyn, nzn);
    };

    writePTot("pXX", cfg.writePXXTot, &EMfields3D::getpXXsn);
    writePTot("pXY", cfg.writePXYTot, &EMfields3D::getpXYsn);
    writePTot("pXZ", cfg.writePXZTot, &EMfields3D::getpXZsn);
    writePTot("pYY", cfg.writePYYTot, &EMfields3D::getpYYsn);
    writePTot("pYZ", cfg.writePYZTot, &EMfields3D::getpYZsn);
    writePTot("pZZ", cfg.writePZZTot, &EMfields3D::getpZZsn);
  }

  // --- Per-species heat-flux tensor ---
  for (int c = 0; c < HeatFlux::ComponentCount; ++c) {
    for (int si : cfg.heatFluxSpecies[c]) {
      file.WriteFields(EMf->getHeatFluxComponent(si, c),
        string(HeatFlux::ComponentNames[c]) + "_" + std::to_string(si),
        nxn, nyn, nzn);
    }
  }

  // --- Total heat-flux tensor ---
  if (cfg.needsAnyHeatFluxTot()) {
    arr3_double tmp(nxn, nyn, nzn);
    for (int c = 0; c < HeatFlux::ComponentCount; ++c) {
      if (!cfg.writeHeatFluxTot[c]) continue;
      for (int i = 0; i < nxn; i++)
        for (int j = 0; j < nyn; j++)
          for (int k = 0; k < nzn; k++) {
            double sum = 0.0;
            for (int s = 0; s < ns; s++)
              sum += EMf->getHeatFlux(i, j, k, s, c);
            tmp[i][j][k] = sum;
          }
      file.WriteFields(tmp, string(HeatFlux::ComponentNames[c]) + "_tot",
                       nxn, nyn, nzn);
    }
  }

  file.CloseFieldsFile();

#else  
  eprintf(
    " The input file requests the use of the Parallel HDF5 functions,\n"
    " but the code has been compiled using the sequential HDF5 library.\n"
    " Recompile the code using the parallel HDF5 options\n"
    " or change the input file options. ");
#endif

}

/*! Function to write the particles using the H5hut library. */
void WritePartclH5hut(int nspec, Grid3DCU *grid, ParticleSoAHost **part, CollectiveIO *col, VCtopology3D *vct, int cycle){
#ifdef USEH5HUT
  timeTasks_set_task(TimeTasks::WRITE_PARTICLES);

  H5output file;

  string filename = col->getSaveDirName() + "/" + col->getSimName();

  file.SetNameCycle(filename, cycle);

  /* ------------------- */
  /* Write the particles */
  /* ------------------- */

  file.OpenPartclFile(nspec, vct->getFieldComm());
  for (int i=0; i<nspec; i++){
    // SoA data is authoritative — no conversion needed
    file.WriteParticles(i, part[i]->getNOP(),
                           part[i]->getQall(),
                           part[i]->getXall(),
                           part[i]->getYall(),
                           part[i]->getZall(),
                           part[i]->getUall(),
                           part[i]->getVall(),
                           part[i]->getWall(),
                           vct->getFieldComm());
  }
  file.ClosePartclFile();

#else  
  eprintf(
    " The input file requests the use of the Parallel HDF5 functions,\n"
    " but the code has been compiled using the sequential HDF5 library.\n"
    " Recompile the code using the parallel HDF5 options\n"
    " or change the input file options. ");
#endif

}

#if 0
void ReadPartclH5hut(int nspec, ParticleSoAHost **part, Collective *col, VCtopology3D *vct, Grid3DCU *grid){
#ifdef USEH5HUT

  H5input infile;
  double L[3] = {col->getLx(), col->getLy(), col->getLz()};

  infile.SetNameCycle(col->getinitfile(), col->getLast_cycle());
  infile.OpenPartclFile(nspec);

  infile.ReadParticles(vct->getCartesian_rank(), vct->getNproc(), vct->getDims(), L, vct->getFieldComm());

  for (int s = 0; s < nspec; s++){
    part[s]->allocate(s, infile.GetNp(s), col, vct, grid);

    infile.DumpPartclX(part[s]->getXref(), s);
    infile.DumpPartclY(part[s]->getYref(), s);
    infile.DumpPartclZ(part[s]->getZref(), s);
    infile.DumpPartclU(part[s]->getUref(), s);
    infile.DumpPartclV(part[s]->getVref(), s);
    infile.DumpPartclW(part[s]->getWref(), s);
    infile.DumpPartclQ(part[s]->getQref(), s);
  }
  infile.ClosePartclFile();

//--- TEST PARTICLE LECTURE:
//  for (int s = 0; s < nspec; s++){
//    for (int n = 0; n < part[s].getNOP(); n++){
//      double ix = part[s].getX(n);
//      double iy = part[s].getY(n);
//      double iz = part[s].getZ(n);
//      if (ix<=0 || iy<=0 || iz <=0) {
//        cout << " ERROR: This particle has negative position. " << endl;
//        cout << "        n = " << n << "/" << part[s].getNOP();
//        cout << "       ix = " << ix;
//        cout << "       iy = " << iy;
//        cout << "       iz = " << iz;
//      }
//    }
//  }
//--- END TEST

#endif
}
#endif

#if 0
void ReadFieldsH5hut(int nspec, EMfields3D *EMf, Collective *col, VCtopology3D *vct, Grid3DCU *grid){
#ifdef USEH5HUT

  H5input infile;

  infile.SetNameCycle(col->getinitfile(), col->getLast_cycle());

  infile.OpenFieldsFile("Node", nspec, col->getNxc()+1,
                                       col->getNyc()+1,
                                       col->getNzc()+1,
                                       vct->getCoordinates(),
                                       vct->getDims(),
                                       vct->getFieldComm());

  infile.ReadFields(EMf->getEx(), "Ex", grid->getNXN(), grid->getNYN(), grid->getNZN());
  infile.ReadFields(EMf->getEy(), "Ey", grid->getNXN(), grid->getNYN(), grid->getNZN());
  infile.ReadFields(EMf->getEz(), "Ez", grid->getNXN(), grid->getNYN(), grid->getNZN());
  infile.ReadFields(EMf->getBx(), "Bx", grid->getNXN(), grid->getNYN(), grid->getNZN());
  infile.ReadFields(EMf->getBy(), "By", grid->getNXN(), grid->getNYN(), grid->getNZN());
  infile.ReadFields(EMf->getBz(), "Bz", grid->getNXN(), grid->getNYN(), grid->getNZN());

  for (int is = 0; is < nspec; is++){
    std::stringstream  ss;
    ss << is;
    std::string s_is = ss.str();
    infile.ReadFields(EMf->getRHOns(is), "rho_"+s_is, grid->getNXN(), grid->getNYN(), grid->getNZN());
  }

  infile.CloseFieldsFile();

  // initialize B on centers
    MPI_Barrier(MPIdata::get_PicGlobalComm());

  // Comm ghost nodes for B-field
  communicateNodeBC(grid->getNXN(), grid->getNYN(), grid->getNZN(), EMf->getBx(), col->bcBx[0],col->bcBx[1],col->bcBx[2],col->bcBx[3],col->bcBx[4],col->bcBx[5], vct);
  communicateNodeBC(grid->getNXN(), grid->getNYN(), grid->getNZN(), EMf->getBy(), col->bcBy[0],col->bcBy[1],col->bcBy[2],col->bcBy[3],col->bcBy[4],col->bcBy[5], vct);
  communicateNodeBC(grid->getNXN(), grid->getNYN(), grid->getNZN(), EMf->getBz(), col->bcBz[0],col->bcBz[1],col->bcBz[2],col->bcBz[3],col->bcBz[4],col->bcBz[5], vct);

  grid->interpN2C(EMf->getBxc(), EMf->getBx());
  grid->interpN2C(EMf->getByc(), EMf->getBy());
  grid->interpN2C(EMf->getBzc(), EMf->getBz());

  // Comm ghost cells for B-field
  communicateNodeBC(grid->getNXN(), grid->getNYN(), grid->getNZN(), EMf->getBx(), col->bcBx[0],col->bcBx[1],col->bcBx[2],col->bcBx[3],col->bcBx[4],col->bcBx[5], vct);
  communicateNodeBC(grid->getNXN(), grid->getNYN(), grid->getNZN(), EMf->getBy(), col->bcBy[0],col->bcBy[1],col->bcBy[2],col->bcBy[3],col->bcBy[4],col->bcBy[5], vct);
  communicateNodeBC(grid->getNXN(), grid->getNYN(), grid->getNZN(), EMf->getBz(), col->bcBz[0],col->bcBz[1],col->bcBz[2],col->bcBz[3],col->bcBz[4],col->bcBz[5], vct);

  // communicate E
  communicateNodeBC(grid->getNXN(), grid->getNYN(), grid->getNZN(), EMf->getEx(), col->bcBx[0],col->bcBx[1],col->bcBx[2],col->bcBx[3],col->bcBx[4],col->bcBx[5], vct);
  communicateNodeBC(grid->getNXN(), grid->getNYN(), grid->getNZN(), EMf->getEy(), col->bcBy[0],col->bcBy[1],col->bcBy[2],col->bcBy[3],col->bcBy[4],col->bcBy[5], vct);
  communicateNodeBC(grid->getNXN(), grid->getNYN(), grid->getNZN(), EMf->getEz(), col->bcBz[0],col->bcBz[1],col->bcBz[2],col->bcBz[3],col->bcBz[4],col->bcBz[5], vct);

  for (int is = 0; is < nspec; is++)
    grid->interpN2C(EMf->getRHOcs(), is, EMf->getRHOns());

//---READ FROM THE CELLS:
//
//  infile.OpenFieldsFile("Cell", nspec, col->getNxc(),
//                                       col->getNyc(),
//                                       col->getNzc(),
//                                       vct->getCoordinates(),
//                                       vct->getDims(),
//                                       vct->getComm());
//
//  infile.ReadFields(EMf->getExc(), "Exc", grid->getNXC(), grid->getNYC(), grid->getNZC());
//  infile.ReadFields(EMf->getEyc(), "Eyc", grid->getNXC(), grid->getNYC(), grid->getNZC());
//  infile.ReadFields(EMf->getEzc(), "Ezc", grid->getNXC(), grid->getNYC(), grid->getNZC());
//  infile.ReadFields(EMf->getBxc(), "Bxc", grid->getNXC(), grid->getNYC(), grid->getNZC());
//  infile.ReadFields(EMf->getByc(), "Byc", grid->getNXC(), grid->getNYC(), grid->getNZC());
//  infile.ReadFields(EMf->getBzc(), "Bzc", grid->getNXC(), grid->getNYC(), grid->getNZC());
//
//  for (int is = 0; is < nspec; is++){
//    std::stringstream  ss;
//    ss << is;
//    std::string s_is = ss.str();
//    infile.ReadFields(EMf->getRHOcs(is, 0), "rhoc_"+s_is, grid->getNXC(), grid->getNYC(), grid->getNZC());
//  }
//
//  infile.CloseFieldsFile();
//
//  // initialize B on nodes
//  grid->interpC2N(EMf->getBx(), EMf->getBxc());
//  grid->interpC2N(EMf->getBy(), EMf->getByc());
//  grid->interpC2N(EMf->getBz(), EMf->getBzc());
//
//  for (int is = 0; is < nspec; is++)
//    grid->interpC2N(EMf->getRHOns(), is, EMf->getRHOcs());
//
//---END READ FROM THE CELLS

#endif
}
#endif


#include <functional>

// ─── VTK MPI-IO helpers ──────────────────────────────────────────────

/** Common grid geometry for VTK writers. */
struct VTKGridInfo {
	int nxn, nyn, nzn;     // local node counts (excluding ghosts, +1 for upper boundary)
	int dimX, dimY, dimZ;  // global node counts
	double spaceX, spaceY, spaceZ;
	int nPoints;
	int cycle;             // simulation cycle (used by nonblocking writers)
	int lx, ly, lz;       // local write sizes (interior nodes excluding 3 ghost layers)
};

static VTKGridInfo getVTKGridInfo(Grid3DCU *grid, CollectiveIO *col, VCtopology3D *vct) {
	VTKGridInfo g;
	g.nxn  = grid->getNXN() + (vct->isXupper() ? 1 : 0);
	g.nyn  = grid->getNYN() + (vct->isYupper() ? 1 : 0);
	g.nzn  = grid->getNZN() + (vct->isZupper() ? 1 : 0);
	g.dimX = col->getNxc() + 1;
	g.dimY = col->getNyc() + 1;
	g.dimZ = col->getNzc() + 1;
	g.spaceX = g.dimX > 1 ? col->getLx() / (g.dimX - 1) : col->getLx();
	g.spaceY = g.dimY > 1 ? col->getLy() / (g.dimY - 1) : col->getLy();
	g.spaceZ = g.dimZ > 1 ? col->getLz() / (g.dimZ - 1) : col->getLz();
	g.nPoints = g.dimX * g.dimY * g.dimZ;
	g.cycle = 0;
	g.lx = g.nxn - 3;
	g.ly = g.nyn - 3;
	g.lz = g.nzn - 3;
	return g;
}

/**
 * @brief Write a 3-component vector VTK file via MPI-IO.
 */
static void writeVectorVTK(
	Grid3DCU *grid, EMfields3D *EMf, CollectiveIO *col, VCtopology3D *vct,
	const std::string &filepath, const std::string &vectorName,
	const std::string &description,
	float**** buf, const VTKGridInfo &g,
	std::function<void(float****,int,int,int)> fillFn)
{
	const int lx = g.nxn - 3, ly = g.nyn - 3, lz = g.nzn - 3;

	fillFn(buf, lx, ly, lz);

	if (EMf->isLittleEndian()) {
		for (int iz = 0; iz < lz; iz++)
			for (int iy = 0; iy < ly; iy++)
				for (int ix = 0; ix < lx; ix++) {
					ByteSwap((unsigned char*)&buf[iz][iy][ix][0], 4);
					ByteSwap((unsigned char*)&buf[iz][iy][ix][1], 4);
					ByteSwap((unsigned char*)&buf[iz][iy][ix][2], 4);
				}
	}

	char header[1024];
	sprintf(header,
		"# vtk DataFile Version 2.0\n"
		"%s from iPIC3D\n"
		"BINARY\n"
		"DATASET STRUCTURED_POINTS\n"
		"DIMENSIONS %d %d %d\n"
		"ORIGIN 0 0 0\n"
		"SPACING %f %f %f\n"
		"POINT_DATA %d\n"
		"VECTORS %s float\n",
		description.c_str(),
		g.dimX, g.dimY, g.dimZ,
		g.spaceX, g.spaceY, g.spaceZ,
		g.nPoints,
		vectorName.c_str());

	int nelem = strlen(header);
	MPI_Offset disp = nelem * sizeof(char);
	MPI_File fh;
	MPI_Status status;

	MPI_File_open(vct->getFieldComm(), filepath.c_str(),
		MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &fh);
	if (vct->getCartesian_rank() == 0)
		MPI_File_write(fh, header, nelem, MPI_BYTE, &status);

	int ec = MPI_File_set_view(fh, disp, EMf->getXYZeType(),
		EMf->getProcviewXYZ(), "native", MPI_INFO_NULL);
	if (ec != MPI_SUCCESS) {
		char es[100]; int len, cls;
		MPI_Error_class(ec, &cls); MPI_Error_string(cls, es, &len);
		dprintf("Error in MPI_File_set_view: %s\n", es);
	}
	ec = MPI_File_write_all(fh, buf[0][0][0], lx * ly * lz, EMf->getXYZeType(), &status);
	if (ec != MPI_SUCCESS) {
		char es[100]; int len, cls;
		MPI_Error_class(ec, &cls); MPI_Error_string(cls, es, &len);
		dprintf("Error in MPI_File_write_all: %s\n", es);
	}
	MPI_File_close(&fh);
}

/**
 * @brief Write a scalar VTK file via MPI-IO.
 */
static void writeScalarVTK(
	Grid3DCU *grid, EMfields3D *EMf, CollectiveIO *col, VCtopology3D *vct,
	const std::string &filepath, const std::string &scalarName,
	const std::string &description,
	float*** buf, const VTKGridInfo &g,
	std::function<void(float***,int,int,int)> fillFn)
{
	const int lx = g.nxn - 3, ly = g.nyn - 3, lz = g.nzn - 3;

	fillFn(buf, lx, ly, lz);

	if (EMf->isLittleEndian()) {
		for (int iz = 0; iz < lz; iz++)
			for (int iy = 0; iy < ly; iy++)
				for (int ix = 0; ix < lx; ix++)
					ByteSwap((unsigned char*)&buf[iz][iy][ix], 4);
	}

	char header[1024];
	sprintf(header,
		"# vtk DataFile Version 2.0\n"
		"%s from iPIC3D\n"
		"BINARY\n"
		"DATASET STRUCTURED_POINTS\n"
		"DIMENSIONS %d %d %d\n"
		"ORIGIN 0 0 0\n"
		"SPACING %f %f %f\n"
		"POINT_DATA %d\n"
		"SCALARS %s float\n"
		"LOOKUP_TABLE default\n",
		description.c_str(),
		g.dimX, g.dimY, g.dimZ,
		g.spaceX, g.spaceY, g.spaceZ,
		g.nPoints,
		scalarName.c_str());

	int nelem = strlen(header);
	MPI_Offset disp = nelem * sizeof(char);
	MPI_File fh;
	MPI_Status status;

	MPI_File_open(vct->getFieldComm(), filepath.c_str(),
		MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &fh);
	if (vct->getCartesian_rank() == 0)
		MPI_File_write(fh, header, nelem, MPI_BYTE, &status);

	int ec = MPI_File_set_view(fh, disp, MPI_FLOAT,
		EMf->getProcview(), "native", MPI_INFO_NULL);
	if (ec != MPI_SUCCESS) {
		char es[100]; int len, cls;
		MPI_Error_class(ec, &cls); MPI_Error_string(cls, es, &len);
		dprintf("Error in MPI_File_set_view: %s\n", es);
	}
	ec = MPI_File_write_all(fh, buf[0][0], lx * ly * lz, MPI_FLOAT, &status);
	if (ec != MPI_SUCCESS) {
		char es[100]; int len, cls;
		MPI_Error_class(ec, &cls); MPI_Error_string(cls, es, &len);
		dprintf("Error in MPI_File_write_all: %s\n", es);
	}
	MPI_File_close(&fh);
}

/** Build a VTK output file path. */
static std::string vtkPath(CollectiveIO *col, const std::string &tag, int cycle) {
	ostringstream ss;
	ss << col->getSaveDirName() << "/" << col->getSimName() << "_" << tag << "_" << cycle << ".vtk";
	return ss.str();
}

// ─── WriteFieldsVTK (new: uses OutputTagConfig, B and E only) ────────

void WriteFieldsVTK(Grid3DCU *grid, EMfields3D *EMf, CollectiveIO *col, VCtopology3D *vct, const string & outputTag ,int cycle,float**** fieldwritebuffer){

	const OutputTagConfig &cfg = col->getOutputConfig();
	VTKGridInfo g = getVTKGridInfo(grid, col, vct);

	if (cfg.writeB) {
		writeVectorVTK(grid, EMf, col, vct,
			vtkPath(col, "B", cycle), "B", "Magnetic Field",
			fieldwritebuffer, g,
			[&](float**** buf, int lx, int ly, int lz) {
				for (int iz = 0; iz < lz; iz++)
					for (int iy = 0; iy < ly; iy++)
						for (int ix = 0; ix < lx; ix++) {
							buf[iz][iy][ix][0] = (float)EMf->getBxTot(ix+1, iy+1, iz+1);
							buf[iz][iy][ix][1] = (float)EMf->getByTot(ix+1, iy+1, iz+1);
							buf[iz][iy][ix][2] = (float)EMf->getBzTot(ix+1, iy+1, iz+1);
						}
			});
	}

	if (cfg.writeE) {
		writeVectorVTK(grid, EMf, col, vct,
			vtkPath(col, "E", cycle), "E", "Electric Field",
			fieldwritebuffer, g,
			[&](float**** buf, int lx, int ly, int lz) {
				for (int iz = 0; iz < lz; iz++)
					for (int iy = 0; iy < ly; iy++)
						for (int ix = 0; ix < lx; ix++) {
							buf[iz][iy][ix][0] = (float)EMf->getEx(ix+1, iy+1, iz+1);
							buf[iz][iy][ix][1] = (float)EMf->getEy(ix+1, iy+1, iz+1);
							buf[iz][iy][ix][2] = (float)EMf->getEz(ix+1, iy+1, iz+1);
						}
			});
	}
}

// ─── WriteMomentsVTK (new: uses OutputTagConfig, numeric species naming) ──

void WriteMomentsVTK(Grid3DCU *grid, EMfields3D *EMf, CollectiveIO *col, VCtopology3D *vct, const string & outputTag ,int cycle, float*** momentswritebuffer){

	const OutputTagConfig &cfg = col->getOutputConfig();
	VTKGridInfo g = getVTKGridInfo(grid, col, vct);
	const int ns = col->getNs();

	// --- Per-species rho ---
	for (int si : cfg.rhoSpecies) {
		writeScalarVTK(grid, EMf, col, vct,
			vtkPath(col, "rho" + std::to_string(si), cycle),
			"rho" + std::to_string(si),
			"Species " + std::to_string(si) + " density",
			momentswritebuffer, g,
			[&](float*** buf, int lx, int ly, int lz) {
				for (int iz = 0; iz < lz; iz++)
					for (int iy = 0; iy < ly; iy++)
						for (int ix = 0; ix < lx; ix++)
							buf[iz][iy][ix] = (float)(EMf->getRHOns(ix+1, iy+1, iz+1, si) * 4.0 * 3.1415926535897);
			});
	}

	// --- Total rho ---
	if (cfg.writeRhoTot) {
		writeScalarVTK(grid, EMf, col, vct,
			vtkPath(col, "rho_tot", cycle), "rho_tot",
			"Total charge density",
			momentswritebuffer, g,
			[&](float*** buf, int lx, int ly, int lz) {
				for (int iz = 0; iz < lz; iz++)
					for (int iy = 0; iy < ly; iy++)
						for (int ix = 0; ix < lx; ix++)
							buf[iz][iy][ix] = (float)(EMf->getRHOn(ix+1, iy+1, iz+1) * 4.0 * 3.1415926535897);
			});
	}

	// --- Per-species J and Total J ---
	// J is a vector quantity and uses the field (vector) write buffer.
	// These are written by WriteMomentsJVTK(), called separately from IOManager.

	// --- Per-species pressure tensor components ---
	struct PDesc {
		const char* name;
		const std::set<int>& species;
		bool writeTot;
		std::function<double(int,int,int,int)> getter;
	};
	PDesc pdescs[] = {
		{"PXX", cfg.PXXSpecies, cfg.writePXXTot, [&](int x,int y,int z,int s){ return EMf->getpXXsn(x,y,z,s); }},
		{"PXY", cfg.PXYSpecies, cfg.writePXYTot, [&](int x,int y,int z,int s){ return EMf->getpXYsn(x,y,z,s); }},
		{"PXZ", cfg.PXZSpecies, cfg.writePXZTot, [&](int x,int y,int z,int s){ return EMf->getpXZsn(x,y,z,s); }},
		{"PYY", cfg.PYYSpecies, cfg.writePYYTot, [&](int x,int y,int z,int s){ return EMf->getpYYsn(x,y,z,s); }},
		{"PYZ", cfg.PYZSpecies, cfg.writePYZTot, [&](int x,int y,int z,int s){ return EMf->getpYZsn(x,y,z,s); }},
		{"PZZ", cfg.PZZSpecies, cfg.writePZZTot, [&](int x,int y,int z,int s){ return EMf->getpZZsn(x,y,z,s); }},
	};

	for (auto &pd : pdescs) {
		// Per-species
		for (int si : pd.species) {
			std::string tag = std::string(pd.name) + std::to_string(si);
			writeScalarVTK(grid, EMf, col, vct,
				vtkPath(col, tag, cycle), tag,
				"Species " + std::to_string(si) + " pressure " + pd.name,
				momentswritebuffer, g,
				[&](float*** buf, int lx, int ly, int lz) {
					for (int iz = 0; iz < lz; iz++)
						for (int iy = 0; iy < ly; iy++)
							for (int ix = 0; ix < lx; ix++)
								buf[iz][iy][ix] = (float)pd.getter(ix+1, iy+1, iz+1, si);
				});
		}
		// Total (sum over all species on-the-fly)
		if (pd.writeTot) {
			std::string tag = std::string(pd.name) + "_tot";
			writeScalarVTK(grid, EMf, col, vct,
				vtkPath(col, tag, cycle), tag,
				"Total pressure " + std::string(pd.name),
				momentswritebuffer, g,
				[&](float*** buf, int lx, int ly, int lz) {
					for (int iz = 0; iz < lz; iz++)
						for (int iy = 0; iy < ly; iy++)
							for (int ix = 0; ix < lx; ix++) {
								double sum = 0.0;
								for (int s = 0; s < ns; s++)
									sum += pd.getter(ix+1, iy+1, iz+1, s);
								buf[iz][iy][ix] = (float)sum;
							}
			});
		}
	}

	// --- Per-species and total heat-flux tensor components ---
	for (int c = 0; c < HeatFlux::ComponentCount; ++c) {
		const std::string name = HeatFlux::ComponentNames[c];
		for (int si : cfg.heatFluxSpecies[c]) {
			std::string tag = name + std::to_string(si);
			writeScalarVTK(grid, EMf, col, vct,
				vtkPath(col, tag, cycle), tag,
				"Species " + std::to_string(si) + " heat flux " + name,
				momentswritebuffer, g,
				[&](float*** buf, int lx, int ly, int lz) {
					for (int iz = 0; iz < lz; iz++)
						for (int iy = 0; iy < ly; iy++)
							for (int ix = 0; ix < lx; ix++)
								buf[iz][iy][ix] = (float)EMf->getHeatFlux(ix+1, iy+1, iz+1, si, c);
				});
		}
		if (cfg.writeHeatFluxTot[c]) {
			std::string tag = name + "_tot";
			writeScalarVTK(grid, EMf, col, vct,
				vtkPath(col, tag, cycle), tag,
				"Total heat flux " + name,
				momentswritebuffer, g,
				[&](float*** buf, int lx, int ly, int lz) {
					for (int iz = 0; iz < lz; iz++)
						for (int iy = 0; iy < ly; iy++)
							for (int ix = 0; ix < lx; ix++) {
								double sum = 0.0;
								for (int s = 0; s < ns; s++)
									sum += EMf->getHeatFlux(ix+1, iy+1, iz+1, s, c);
								buf[iz][iy][ix] = (float)sum;
							}
				});
		}
	}
}

/**
 * @brief Write per-species and total J as vector VTK files.
 *
 * Called from IOManager after field writes (needs the vector write buffer).
 */
void WriteMomentsJVTK(Grid3DCU *grid, EMfields3D *EMf, CollectiveIO *col, VCtopology3D *vct,
                      int cycle, float**** fieldwritebuffer)
{
	const OutputTagConfig &cfg = col->getOutputConfig();
	VTKGridInfo g = getVTKGridInfo(grid, col, vct);
	const int ns = col->getNs();

	// Per-species J
	for (int si : cfg.JSpecies) {
		writeVectorVTK(grid, EMf, col, vct,
			vtkPath(col, "J" + std::to_string(si), cycle),
			"J" + std::to_string(si),
			"Species " + std::to_string(si) + " current",
			fieldwritebuffer, g,
			[&](float**** buf, int lx, int ly, int lz) {
				for (int iz = 0; iz < lz; iz++)
					for (int iy = 0; iy < ly; iy++)
						for (int ix = 0; ix < lx; ix++) {
							buf[iz][iy][ix][0] = (float)EMf->getJxs(ix+1, iy+1, iz+1, si);
							buf[iz][iy][ix][1] = (float)EMf->getJys(ix+1, iy+1, iz+1, si);
							buf[iz][iy][ix][2] = (float)EMf->getJzs(ix+1, iy+1, iz+1, si);
						}
			});
	}

	// Total J
	if (cfg.writeJTot) {
		writeVectorVTK(grid, EMf, col, vct,
			vtkPath(col, "J_tot", cycle), "J_tot",
			"Total current",
			fieldwritebuffer, g,
			[&](float**** buf, int lx, int ly, int lz) {
				for (int iz = 0; iz < lz; iz++)
					for (int iy = 0; iy < ly; iy++)
						for (int ix = 0; ix < lx; ix++) {
							buf[iz][iy][ix][0] = (float)EMf->getJx(ix+1, iy+1, iz+1);
							buf[iz][iy][ix][1] = (float)EMf->getJy(ix+1, iy+1, iz+1);
							buf[iz][iy][ix][2] = (float)EMf->getJz(ix+1, iy+1, iz+1);
						}
			});
	}
}


// ─── Non-blocking VTK helpers (shared grid setup) ────────────────────────────

static void nbcVectorWrite(EMfields3D *EMf, CollectiveIO *col, VCtopology3D *vct,
                           float**** fieldwritebuffer, int counter,
                           const VTKGridInfo &g, const std::string &tag,
                           const std::string &desc,
                           std::function<void(float****,int,int,int)> fill,
                           MPI_File *fhArr)
{
	float**** buf = &fieldwritebuffer[counter * g.lz];
	fill(buf, g.lx, g.ly, g.lz);

	if (EMf->isLittleEndian()) {
		for (int iz = 0; iz < g.lz; iz++)
			for (int iy = 0; iy < g.ly; iy++)
				for (int ix = 0; ix < g.lx; ix++) {
					ByteSwap((unsigned char*)&buf[iz][iy][ix][0], 4);
					ByteSwap((unsigned char*)&buf[iz][iy][ix][1], 4);
					ByteSwap((unsigned char*)&buf[iz][iy][ix][2], 4);
				}
	}

	char header[1024];
	sprintf(header, "# vtk DataFile Version 2.0\n"
	                "%s from iPIC3D\n"
	                "BINARY\n"
	                "DATASET STRUCTURED_POINTS\n"
	                "DIMENSIONS %d %d %d\n"
	                "ORIGIN 0 0 0\n"
	                "SPACING %f %f %f\n"
	                "POINT_DATA %d\n"
	                "VECTORS %s float\n",
	                desc.c_str(), g.dimX, g.dimY, g.dimZ,
	                g.spaceX, g.spaceY, g.spaceZ, g.nPoints, tag.c_str());

	int nelem = strlen(header);
	MPI_Offset disp = nelem * sizeof(char);

	std::string filename = vtkPath(col, tag, 0); // cycle filled below
	{
		std::ostringstream ss;
		ss << col->getSaveDirName() << "/" << col->getSimName() << "_" << tag << "_" << g.cycle << ".vtk";
		filename = ss.str();
	}

	MPI_File_open(vct->getFieldComm(), filename.c_str(),
	              MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &fhArr[counter]);

	if (vct->getCartesian_rank() == 0) {
		MPI_Status status;
		MPI_File_write(fhArr[counter], header, nelem, MPI_BYTE, &status);
	}

	int ec = MPI_File_set_view(fhArr[counter], disp, EMf->getXYZeType(),
	                           EMf->getProcviewXYZ(), "native", MPI_INFO_NULL);
	if (ec != MPI_SUCCESS) {
		char es[100]; int len, cls;
		MPI_Error_class(ec, &cls);
		MPI_Error_string(cls, es, &len);
		dprintf("Error in MPI_File_set_view: %s\n", es);
	}

	ec = MPI_File_write_all_begin(fhArr[counter], buf[0][0][0],
	                              g.lx * g.ly * g.lz, EMf->getXYZeType());
	if (ec != MPI_SUCCESS) {
		char es[100]; int len, cls;
		MPI_Error_class(ec, &cls);
		MPI_Error_string(cls, es, &len);
		dprintf("Error in MPI_File_iwrite: %s\n", es);
	}
}

static void nbcScalarWrite(EMfields3D *EMf, CollectiveIO *col, VCtopology3D *vct,
                            float*** momentswritebuffer, int counter,
                            const VTKGridInfo &g, const std::string &tag,
                            const std::string &desc,
                            std::function<void(float***,int,int,int)> fill,
                            MPI_File *fhArr)
{
	float*** buf = &momentswritebuffer[counter * g.lz];
	fill(buf, g.lx, g.ly, g.lz);

	if (EMf->isLittleEndian()) {
		for (int iz = 0; iz < g.lz; iz++)
			for (int iy = 0; iy < g.ly; iy++)
				for (int ix = 0; ix < g.lx; ix++)
					ByteSwap((unsigned char*)&buf[iz][iy][ix], 4);
	}

	char header[1024];
	sprintf(header, "# vtk DataFile Version 2.0\n"
	                "%s from iPIC3D\n"
	                "BINARY\n"
	                "DATASET STRUCTURED_POINTS\n"
	                "DIMENSIONS %d %d %d\n"
	                "ORIGIN 0 0 0\n"
	                "SPACING %f %f %f\n"
	                "POINT_DATA %d\n"
	                "SCALARS %s float\n"
	                "LOOKUP_TABLE default\n",
	                desc.c_str(), g.dimX, g.dimY, g.dimZ,
	                g.spaceX, g.spaceY, g.spaceZ, g.nPoints, tag.c_str());

	int nelem = strlen(header);
	MPI_Offset disp = nelem * sizeof(char);

	std::ostringstream ss;
	ss << col->getSaveDirName() << "/" << col->getSimName() << "_" << tag << "_" << g.cycle << ".vtk";

	MPI_File_open(vct->getFieldComm(), ss.str().c_str(),
	              MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &fhArr[counter]);

	if (vct->getCartesian_rank() == 0) {
		MPI_Status status;
		MPI_File_write(fhArr[counter], header, nelem, MPI_BYTE, &status);
	}

	int ec = MPI_File_set_view(fhArr[counter], disp, MPI_FLOAT,
	                           EMf->getProcview(), "native", MPI_INFO_NULL);
	if (ec != MPI_SUCCESS) {
		char es[100]; int len, cls;
		MPI_Error_class(ec, &cls);
		MPI_Error_string(cls, es, &len);
		dprintf("Error in MPI_File_set_view: %s\n", es);
	}

	ec = MPI_File_write_all_begin(fhArr[counter], buf[0][0],
	                              g.lx * g.ly * g.lz, MPI_FLOAT);
	if (ec != MPI_SUCCESS) {
		char es[100]; int len, cls;
		MPI_Error_class(ec, &cls);
		MPI_Error_string(cls, es, &len);
		dprintf("Error in MPI_File_iwrite: %s\n", es);
	}
}

// ─── WriteFieldsVTKNonblk (new: OutputTagConfig, B + E + per-species J + J_tot) ──

int WriteFieldsVTKNonblk(Grid3DCU *grid, EMfields3D *EMf, CollectiveIO *col, VCtopology3D *vct, int cycle,
                          float**** fieldwritebuffer, MPI_Request requestArr[], MPI_File fhArr[])
{
	const OutputTagConfig &cfg = col->getOutputConfig();
	VTKGridInfo g = getVTKGridInfo(grid, col, vct);
	g.cycle = cycle;
	int counter = 0;

	if (cfg.writeB) {
		nbcVectorWrite(EMf, col, vct, fieldwritebuffer, counter, g, "B", "Magnetic Field",
			[&](float**** buf, int lx, int ly, int lz) {
				for (int iz = 0; iz < lz; iz++)
					for (int iy = 0; iy < ly; iy++)
						for (int ix = 0; ix < lx; ix++) {
							buf[iz][iy][ix][0] = (float)EMf->getBxTot(ix+1, iy+1, iz+1);
							buf[iz][iy][ix][1] = (float)EMf->getByTot(ix+1, iy+1, iz+1);
							buf[iz][iy][ix][2] = (float)EMf->getBzTot(ix+1, iy+1, iz+1);
						}
			}, fhArr);
		counter++;
	}

	if (cfg.writeE) {
		nbcVectorWrite(EMf, col, vct, fieldwritebuffer, counter, g, "E", "Electric Field",
			[&](float**** buf, int lx, int ly, int lz) {
				for (int iz = 0; iz < lz; iz++)
					for (int iy = 0; iy < ly; iy++)
						for (int ix = 0; ix < lx; ix++) {
							buf[iz][iy][ix][0] = (float)EMf->getEx(ix+1, iy+1, iz+1);
							buf[iz][iy][ix][1] = (float)EMf->getEy(ix+1, iy+1, iz+1);
							buf[iz][iy][ix][2] = (float)EMf->getEz(ix+1, iy+1, iz+1);
						}
			}, fhArr);
		counter++;
	}

	for (int si : cfg.JSpecies) {
		nbcVectorWrite(EMf, col, vct, fieldwritebuffer, counter, g,
			"J" + std::to_string(si), "Species " + std::to_string(si) + " current",
			[&](float**** buf, int lx, int ly, int lz) {
				for (int iz = 0; iz < lz; iz++)
					for (int iy = 0; iy < ly; iy++)
						for (int ix = 0; ix < lx; ix++) {
							buf[iz][iy][ix][0] = (float)EMf->getJxs(ix+1, iy+1, iz+1, si);
							buf[iz][iy][ix][1] = (float)EMf->getJys(ix+1, iy+1, iz+1, si);
							buf[iz][iy][ix][2] = (float)EMf->getJzs(ix+1, iy+1, iz+1, si);
						}
			}, fhArr);
		counter++;
	}

	if (cfg.writeJTot) {
		nbcVectorWrite(EMf, col, vct, fieldwritebuffer, counter, g, "J_tot", "Total current",
			[&](float**** buf, int lx, int ly, int lz) {
				for (int iz = 0; iz < lz; iz++)
					for (int iy = 0; iy < ly; iy++)
						for (int ix = 0; ix < lx; ix++) {
							buf[iz][iy][ix][0] = (float)EMf->getJx(ix+1, iy+1, iz+1);
							buf[iz][iy][ix][1] = (float)EMf->getJy(ix+1, iy+1, iz+1);
							buf[iz][iy][ix][2] = (float)EMf->getJz(ix+1, iy+1, iz+1);
						}
			}, fhArr);
		counter++;
	}

	return counter;
}

// ─── WriteMomentsVTKNonblk (new: OutputTagConfig, numeric species naming) ─────

int WriteMomentsVTKNonblk(Grid3DCU *grid, EMfields3D *EMf, CollectiveIO *col, VCtopology3D *vct, int cycle,
                           float*** momentswritebuffer, MPI_Request requestArr[], MPI_File fhArr[])
{
	const OutputTagConfig &cfg = col->getOutputConfig();
	VTKGridInfo g = getVTKGridInfo(grid, col, vct);
	g.cycle = cycle;
	const int ns = col->getNs();
	int counter = 0;

	// Per-species rho
	for (int si : cfg.rhoSpecies) {
		nbcScalarWrite(EMf, col, vct, momentswritebuffer, counter, g,
			"rho" + std::to_string(si), "Species " + std::to_string(si) + " density",
			[&](float*** buf, int lx, int ly, int lz) {
				for (int iz = 0; iz < lz; iz++)
					for (int iy = 0; iy < ly; iy++)
						for (int ix = 0; ix < lx; ix++)
							buf[iz][iy][ix] = (float)(EMf->getRHOns(ix+1, iy+1, iz+1, si) * 4.0 * 3.1415926535897);
			}, fhArr);
		counter++;
	}

	// Total rho
	if (cfg.writeRhoTot) {
		nbcScalarWrite(EMf, col, vct, momentswritebuffer, counter, g,
			"rho_tot", "Total charge density",
			[&](float*** buf, int lx, int ly, int lz) {
				for (int iz = 0; iz < lz; iz++)
					for (int iy = 0; iy < ly; iy++)
						for (int ix = 0; ix < lx; ix++)
							buf[iz][iy][ix] = (float)(EMf->getRHOn(ix+1, iy+1, iz+1) * 4.0 * 3.1415926535897);
			}, fhArr);
		counter++;
	}

	// Per-species and total pressure tensor components
	struct PDesc {
		const char* name;
		const std::set<int>& species;
		bool writeTot;
		std::function<double(int,int,int,int)> getter;
	};
	PDesc pdescs[] = {
		{"PXX", cfg.PXXSpecies, cfg.writePXXTot, [&](int x,int y,int z,int s){ return EMf->getpXXsn(x,y,z,s); }},
		{"PXY", cfg.PXYSpecies, cfg.writePXYTot, [&](int x,int y,int z,int s){ return EMf->getpXYsn(x,y,z,s); }},
		{"PXZ", cfg.PXZSpecies, cfg.writePXZTot, [&](int x,int y,int z,int s){ return EMf->getpXZsn(x,y,z,s); }},
		{"PYY", cfg.PYYSpecies, cfg.writePYYTot, [&](int x,int y,int z,int s){ return EMf->getpYYsn(x,y,z,s); }},
		{"PYZ", cfg.PYZSpecies, cfg.writePYZTot, [&](int x,int y,int z,int s){ return EMf->getpYZsn(x,y,z,s); }},
		{"PZZ", cfg.PZZSpecies, cfg.writePZZTot, [&](int x,int y,int z,int s){ return EMf->getpZZsn(x,y,z,s); }},
	};

	for (auto &pd : pdescs) {
		for (int si : pd.species) {
			std::string tag = std::string(pd.name) + std::to_string(si);
			nbcScalarWrite(EMf, col, vct, momentswritebuffer, counter, g,
				tag, "Species " + std::to_string(si) + " pressure " + pd.name,
				[&](float*** buf, int lx, int ly, int lz) {
					for (int iz = 0; iz < lz; iz++)
						for (int iy = 0; iy < ly; iy++)
							for (int ix = 0; ix < lx; ix++)
								buf[iz][iy][ix] = (float)pd.getter(ix+1, iy+1, iz+1, si);
				}, fhArr);
			counter++;
		}
		if (pd.writeTot) {
			std::string tag = std::string(pd.name) + "_tot";
			nbcScalarWrite(EMf, col, vct, momentswritebuffer, counter, g,
				tag, "Total pressure " + std::string(pd.name),
				[&](float*** buf, int lx, int ly, int lz) {
					for (int iz = 0; iz < lz; iz++)
						for (int iy = 0; iy < ly; iy++)
							for (int ix = 0; ix < lx; ix++) {
								double sum = 0.0;
								for (int s = 0; s < ns; s++)
									sum += pd.getter(ix+1, iy+1, iz+1, s);
								buf[iz][iy][ix] = (float)sum;
							}
				}, fhArr);
			counter++;
		}
	}

	// Per-species and total heat-flux tensor components
	for (int c = 0; c < HeatFlux::ComponentCount; ++c) {
		const std::string name = HeatFlux::ComponentNames[c];
		for (int si : cfg.heatFluxSpecies[c]) {
			std::string tag = name + std::to_string(si);
			nbcScalarWrite(EMf, col, vct, momentswritebuffer, counter, g,
				tag, "Species " + std::to_string(si) + " heat flux " + name,
				[&](float*** buf, int lx, int ly, int lz) {
					for (int iz = 0; iz < lz; iz++)
						for (int iy = 0; iy < ly; iy++)
							for (int ix = 0; ix < lx; ix++)
								buf[iz][iy][ix] = (float)EMf->getHeatFlux(ix+1, iy+1, iz+1, si, c);
				}, fhArr);
			counter++;
		}
		if (cfg.writeHeatFluxTot[c]) {
			std::string tag = name + "_tot";
			nbcScalarWrite(EMf, col, vct, momentswritebuffer, counter, g,
				tag, "Total heat flux " + name,
				[&](float*** buf, int lx, int ly, int lz) {
					for (int iz = 0; iz < lz; iz++)
						for (int iy = 0; iy < ly; iy++)
							for (int ix = 0; ix < lx; ix++) {
								double sum = 0.0;
								for (int s = 0; s < ns; s++)
									sum += EMf->getHeatFlux(ix+1, iy+1, iz+1, s, c);
								buf[iz][iy][ix] = (float)sum;
							}
				}, fhArr);
			counter++;
		}
	}

	return counter;
}



void ByteSwap(unsigned char * b, int n)
{
   int i = 0;
   int j = n-1;
   while (i<j)
   {
      std::swap(b[i], b[j]);
      i++, j--;
   }
}

void WriteTestPclsVTK(int nspec, Grid3DCU *grid, ParticleSoAHost **testpart, EMfields3D *EMf,
		CollectiveIO *col, VCtopology3D *vct, const string & tag, int cycle,MPI_Request *testpartMPIReq, MPI_File *fh){
	/* the below is nonblocking collective IO
	 * const int nop = testpart[0].getNOP();

  	if(cycle>0){
  		MPI_Wait(headerReq, status);
  		MPI_Wait(dataReq, status);
  		MPI_Wait(footReq, status);
//  		MPI_File_close(&fh);
//  		int error_code=status->MPI_ERROR;
//  		if (error_code != MPI_SUCCESS) {
//  			char error_string[100];
//  			int length_of_error_string, error_class;
//
//  			MPI_Error_class(error_code, &error_class);
//  			MPI_Error_string(error_class, error_string, &length_of_error_string);
//  			dprintf("MPI_Wait error: %s\n", error_string);
//  		}
  	}else{
  		pclbuffersize = nop*3*1.2;
  		testpclPos = new float[pclbuffersize];
  	}

  	if(nop>pclbuffersize){
  		pclbuffersize = nop*3*1.2;
  		delete testpclPos;
  		testpclPos = new float[pclbuffersize];
  	}

//  	for (int pidx = 0; pidx < nop; pidx++) {
//  		testpclPos[pidx*3+0]=(float)(testpart[0]).getX(pidx);
//  		testpclPos[pidx*3+1]=(float)(testpart[0]).getY(pidx);
//  		testpclPos[pidx*3+2]=(float)(testpart[0]).getZ(pidx);
//  	}

  	//write to parallel vtk pvtu files
  	int  is=0;
	ostringstream filename;
	filename << col->getSaveDirName() << "/" << col->getSimName() << "_testparticle"<< testpart[is].get_species_num() << "_cycle" << cycle << ".vtu";
	MPI_File_open(vct->getComm(),filename.str().c_str(), MPI_MODE_CREATE|MPI_MODE_WRONLY, MPI_INFO_NULL, &fh);
	MPI_File_set_view(fh, 0, MPI_BYTE, MPI_BYTE, "native", MPI_INFO_NULL);

	  ofstream myfile;
	  myfile.open ("example.vtu");
	  myfile <<  "<?xml version=\"1.0\"?>\n"
				"<VTKFile type=\"UnstructuredGrid\" version=\"0.1\" byte_order=\"LittleEndian\">\n"
			    "  <UnstructuredGrid>\n"
				"    <Piece NumberOfPoints=\"1\" NumberOfCells=\"1\">\n"
				"		<Cells>\n"
				"			<DataArray type=\"UInt8\" Name=\"connectivity\" format=\"ascii\">0 1</DataArray>\n"
				"			<DataArray type=\"UInt8\" Name=\"offsets\" 		format=\"ascii\">1</DataArray>\n"
				"			<DataArray type=\"UInt8\" Name=\"types\"    	format=\"ascii\">1</DataArray>\n"
				"		</Cells>\n"
				"		<Points>\n"
				"        	<DataArray type=\"Float32\" NumberOfComponents=\"3\" format=\"acscii\">\n" <<
				(testpart[0]).getX(100) << (testpart[0]).getY(100)  << (testpart[0]).getZ(100) <<
				 "			</DataArray>\n"
				  	  					"		</Points>\n"
				  	  					"	</Piece>\n"
				  	  					"	</UnstructuredGrid>\n"
				  	  					"</VTKFile>";
	  myfile.close();

  	char header[8192];
  	sprintf(header, "<?xml version=\"1.0\"?>\n"
  					"<VTKFile type=\"UnstructuredGrid\" version=\"0.1\" byte_order=\"%s\">\n"
  				    "  <UnstructuredGrid>\n"
  					"    <Piece NumberOfPoints=\"%d\" NumberOfCells=\"1\">\n"
  					"		<Cells>\n"
  					"			<DataArray type=\"UInt8\" Name=\"connectivity\" format=\"ascii\">0 1</DataArray>\n"
  					"			<DataArray type=\"UInt8\" Name=\"offsets\" 		format=\"ascii\">1</DataArray>\n"
  					"			<DataArray type=\"UInt8\" Name=\"types\"    	format=\"ascii\">1</DataArray>\n"
  					"		</Cells>\n"
  					"		<Points>\n"
  					"        	<DataArray type=\"Float32\" NumberOfComponents=\"3\" format=\"binary\">\n",
  					(EMf->isLittleEndian() ?"LittleEndian":"BigEndian"),3);

  	int nelem = strlen(header);
  	int charsize=sizeof(char);
  	MPI_Offset disp = nelem*charsize;

  	//MPI_File_iwrite(fh, header, nelem, MPI_BYTE, headerReq);
  	MPI_File_write(fh, header, nelem, MPI_BYTE, status);

  	int err = MPI_File_set_view(fh, disp, MPI_FLOAT, MPI_FLOAT, "native", MPI_INFO_NULL);
  	if(err){
  		          dprintf("Error in MPI_File_set_view\n");
  		      }

	//dprintf("testpart[is].getNOP() = %d, sizeof(SpeciesParticle)=%d, u = %f, x = %f ",testpart[0].getNOP(), sizeof(SpeciesParticle), pcl.get_u(), pcl.get_x());
	//const SpeciesParticle *pclptr = testpart[is].get_pclptr(0);
	//const SpeciesParticle * pclptr = (SpeciesParticle * )temppcl;
	//dprintf("temppcl u = %f, x= %f",pclptr->get_u() , pclptr->get_x());
//	MPI_Datatype particleType;
//	MPI_Type_vector (10,3,sizeof(SpeciesParticle),MPI_DOUBLE,&particleType);//testpart[0].getNOP()
//	MPI_Type_commit(&particleType);

  	//MPI_File_iwrite(fh, pclptr, 1, particleType, dataReq);
  	testpclPos[0]=1.1;testpclPos[1]=2.1;testpclPos[2]=3.1;
  	testpclPos[3]=1.1;testpclPos[4]=2.1;testpclPos[5]=3.1;
  	testpclPos[6]=1.1;testpclPos[7]=2.1;testpclPos[8]=3.1;
  	MPI_File_write_all(fh, testpclPos, 9, MPI_FLOAT, status);
  	 int tcount=0;
	  MPI_Get_count(status, MPI_FLOAT, &tcount);
	  dprintf(" wrote %i MPI_FLOAT",  tcount);
	int error_code=status->MPI_ERROR;
	if (error_code != MPI_SUCCESS) {
		char error_string[100];
		int length_of_error_string, error_class;

		MPI_Error_class(error_code, &error_class);
		MPI_Error_string(error_class, error_string, &length_of_error_string);
		dprintf("MPI_File_write error: %s\n", error_string);
	}

  	char foot[8192];
  	sprintf(foot, "			</DataArray>\n"
  	  					"		</Points>\n"
  	  					"	</Piece>\n"
  	  					"	</UnstructuredGrid>\n"
  	  					"</VTKFile>");
  	//nelem = strlen(foot);
  	//MPI_File_set_view(fh, disp+9, MPI_BYTE, MPI_BYTE, "native", MPI_INFO_NULL);
  	//MPI_File_iwrite(fh, foot, nelem, MPI_BYTE, footReq);
  	//MPI_File_write(fh, foot, nelem, MPI_BYTE, status);

  	if(cycle==LastCycle()){
  		MPI_Wait(headerReq, status);
  		MPI_Wait(dataReq, status);
  		MPI_Wait(footReq, status);
  	    MPI_File_close(&fh);
  	}
	 * */
	MPI_Status  *status;
	if(cycle>0){dprintf("Previous writing done");
		MPI_Wait(testpartMPIReq, status);dprintf("Previous writing done");
		int error_code=status->MPI_ERROR;
		if (error_code != MPI_SUCCESS) {
			char error_string[100];
			int length_of_error_string, error_class;

			MPI_Error_class(error_code, &error_class);
			MPI_Error_string(error_class, error_string, &length_of_error_string);
			dprintf("MPI_Wait error: %s\n", error_string);
		}
		else{
			if (vct->getCartesian_rank()==0) MPI_File_close(fh);
			dprintf("Previous writing done");
		}
	}

	//buffering if no full
	const int timesteps = 10;
	int nopStep[timesteps];



	//write to parallel vtk pvtu files

	int 		 is=0;

	char header[12048];
	sprintf(header, "<?xml version=\"1.0\"?>\n"
					"<VTKFile type=\"UnstructuredGrid\" version=\"0.1\" byte_order=\"%s\">\n"
				    "  <UnstructuredGrid>\n"
					"    <Piece NumberOfPoints=\"%d\" NumberOfCells=\"1\">\n"
					"		<Cells>\n"
					"			<DataArray type=\"UInt8\" Name=\"connectivity\" format=\"ascii\">0 1</DataArray>\n"
					"			<DataArray type=\"UInt8\" Name=\"offsets\" 		format=\"ascii\">0</DataArray>\n"
					"			<DataArray type=\"UInt8\" Name=\"types\"    	format=\"ascii\">11</DataArray>\n"
					"		</Cells>\n"
					"		<Points>\n"
					"        	<DataArray type=\"Float32\" NumberOfComponents=\"3\" format=\"ascii\">\n"
					"			</DataArray>\n"
					"		</Points>\n"
					"	</Piece>\n"
					"	</UnstructuredGrid>\n"
					"</VTKFile>", (EMf->isLittleEndian() ?"LittleEndian":"BigEndian"),testpart[is]->getNOP());

	int nelem = strlen(header);
	int charsize=sizeof(char);
	MPI_Offset disp = nelem*charsize;

	ostringstream filename;
	filename << col->getSaveDirName() << "/" << col->getSimName() << "_testparticle"<< testpart[is]->get_species_num() << "_cycle" << cycle << ".vtu";
	MPI_File_open(vct->getFieldComm(),filename.str().c_str(), MPI_MODE_CREATE|MPI_MODE_WRONLY, MPI_INFO_NULL, fh);

	MPI_File_set_view(*fh, 0, MPI_BYTE, MPI_BYTE, "native", MPI_INFO_NULL);
	if (vct->getCartesian_rank()==0){

		MPI_File_iwrite(*fh, header, nelem, MPI_BYTE, testpartMPIReq);

	}

	/*
	 *
<VTKFile type="UnstructuredGrid" version="0.1" byte_order="LittleEndian">
  <UnstructuredGrid>
    <Piece NumberOfPoints="9" NumberOfCells="1">
        <Cells>
        <DataArray type="Int32" Name="connectivity" format="ascii">
          0 1 2 3 4 5 6 7 8
        </DataArray>
		<DataArray type="Int32" Name="offsets" format="ascii">
		 0
		</DataArray>

		<DataArray type="UInt8" Name="types" format="ascii">
			1
		</DataArray>
        </Cells>

      <PointData Scalars="testpartID">
        <DataArray type="UInt16" Name="testpartID" format="ascii">
          0 1 2 3 4 5 6 7 8
        </DataArray>

        <DataArray type="Float32" Name="testpartVelocity" NumberOfComponents="3" format="ascii">
          -1 -1 -1
          -1 -1 -1
          -1 -1 -1
          -1 -1 -1
          -1 -1 -1
          -1 -1 -1
          -1 -1 -1
          -1 -1 -1
          -1 -1 -1
        </DataArray>
      </PointData>

      <Points>
        <DataArray type="Float32" NumberOfComponents="3" format="ascii">
          0 0 0 0 0 1 0 0 2
          0 1 0 0 1 1 0 1 2
          0 2 0 0 2 1 0 2 2
        </DataArray>
      </Points>
    </Piece>
  </UnstructuredGrid>
</VTKFile>
	 * */

}
