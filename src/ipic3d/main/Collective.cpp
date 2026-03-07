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
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include "input_array.h"
#include "Collective.h"
#include "ConfigFile.h"
#include "limits.h" // for INT_MAX
#include "MPIdata.h"
#include "debug.h"
#include "asserts.h" // for assert_ge
#include "string.h"
#include "RestartReader.h"

// order must agree with Enum in Collective.h
static const char *enumNames[] =
{
  "default",
  "initial",
  "final",
  // used by ImplSusceptMode
  "explPredict",
  "implPredict",
  // marker for last enumerated symbol of this class
  "NUMBER_OF_ENUMS",
  "INVALID_ENUM"
};

int Collective::read_enum_parameter(const char* option_name, const char* default_value,
  const ConfigFile& config)
{
  string enum_name = config.read < string >(option_name,default_value);
  // search the list (could use std::map)
  //
  for(int i=0;i<NUMBER_OF_ENUMS;i++)
  {
    if(!strcmp(enum_name.c_str(),enumNames[i]))
      return i;
  }
  // could not find enum, so issue error and quit.
  if(!MPIdata::get_rank())
  {
    eprintf("in input file %s there is an invalid option %s\n",
      inputfile.c_str(), enum_name.c_str());
  }
  MPIdata::exit(1);
  // this is a better way
  return INVALID_ENUM;
}

const char* Collective::get_name_of_enum(int in)
{
  assert_ge(in, 0);
  assert_lt(in, NUMBER_OF_ENUMS);
  return enumNames[in];
}

/*! Read the input file from text file and put the data in a collective wrapper: if it's a restart read from input file basic sim data and load particles and EM field from restart file */
void Collective::ReadInput(string inputfile) {
  using namespace std;
  int test_verbose;
  // Loading the input file 
  ConfigFile config(inputfile);
  // the following variables are ALWAYS taken from inputfile, even if restarting 
  {

#ifdef BATSRUS
    if(RESTART1)
    {
      cout<<" The fluid interface can not handle RESTART yet, aborting!\n"<<flush;
      abort();
    }
#endif

    dt = config.read < double >("dt");
    ncycles = config.read < int >("ncycles");
    th = config.read < double >("th",1.0);

    Smooth = config.read < double >("Smooth",1.0);
    SmoothNiter = config.read < int >("SmoothNiter",6);

    SaveDirName = config.read < string > ("SaveDirName","data");
    RestartDirName = config.read < string > ("RestartDirName","data");
    ns = config.read < int >("ns");
    nstestpart = config.read < int >("nsTestPart", 0);
    NpMaxNpRatio = config.read < double >("NpMaxNpRatio",1.5);
    assert_ge(NpMaxNpRatio, 1.);
    // mode parameters for second order in time
    PushWithBatTime = config.read < double >("PushWithBatTime",0);
    PushWithEatTime = config.read < double >("PushWithEatTime",1);
    ImplSusceptTime = config.read < double >("ImplSusceptTime",0);
    ImplSusceptMode = read_enum_parameter("ImplSusceptMode", "initial",config);
    switch(ImplSusceptMode)
    {
      // values not yet supported:
      case explPredict:
      case implPredict:
      default:
        unsupported_value_error(ImplSusceptMode);
      // supported values:
      case initial:
        ;
    }
    // GEM Challenge 
    B0x = config.read <double>("B0x",0.0);
    B0y = config.read <double>("B0y",0.0);
    B0z = config.read <double>("B0z",0.0);

    // Earth parameters
    B1x = 0.0;
    B1y = 0.0;
    B1z = 0.0;
    B1x = config.read <double>("B1x",0.0);
    B1y = config.read <double>("B1y",0.0);
    B1z = config.read <double>("B1z",0.0);

    delta = config.read < double >("delta",0.5);

    Case              = config.read<string>("Case");
    wmethod           = config.read<string>("WriteMethod");
    SimName           = config.read<string>("SimulationName");
    PoissonCorrection = config.read<string>("PoissonCorrection","no");
    PoissonCorrectionCycle = config.read<int>("PoissonCorrectionCycle",10);
    divBCorrection = config.read<string>("divBCorrection","no");
    divBCorrectionCycle = config.read<int>("divBCorrectionCycle",10);

    rhoINIT = std::make_unique<double[]>(ns);
    array_double rhoINIT0 = config.read < array_double > ("rhoINIT");
    rhoINIT[0] = rhoINIT0.a;
    if (ns > 1)
      rhoINIT[1] = rhoINIT0.b;
    if (ns > 2)
      rhoINIT[2] = rhoINIT0.c;
    if (ns > 3)
      rhoINIT[3] = rhoINIT0.d;
    if (ns > 4)
      rhoINIT[4] = rhoINIT0.e;
    if (ns > 5)
      rhoINIT[5] = rhoINIT0.f;

    rhoINJECT =std::make_unique<double[]>(ns);
    array_double rhoINJECT0 = config.read<array_double>( "rhoINJECT" );
    rhoINJECT[0]=rhoINJECT0.a;
    if (ns > 1)
      rhoINJECT[1]=rhoINJECT0.b;
    if (ns > 2)
      rhoINJECT[2]=rhoINJECT0.c;
    if (ns > 3)
      rhoINJECT[3]=rhoINJECT0.d;
    if (ns > 4)
      rhoINJECT[4]=rhoINJECT0.e;
    if (ns > 5)
      rhoINJECT[5]=rhoINJECT0.f;

    // take the tolerance of the solvers
    CGtol = config.read < double >("CGtol",1e-3);
    GMREStol = config.read < double >("GMREStol",1e-3);
    NiterMover = config.read < int >("NiterMover",3);
    // solver selection: "GMRES" (default) or "Chebyshev"
    SolverType = config.read < string >("SolverType","GMRES");
    ChebyshevMaxIter = config.read < int >("ChebyshevMaxIter",20);
    ChebyshevEigMin = config.read < double >("ChebyshevEigMin",0.0);
    ChebyshevEigMax = config.read < double >("ChebyshevEigMax",0.0);
    // Block-Jacobi preconditioner parameters
    BlockJacobiSweeps = config.read < int >("BlockJacobiSweeps",1);
    BlockJacobiOmega = config.read < double >("BlockJacobiOmega",1.0);
    // Poisson Chebyshev preconditioner parameters (divergence cleaning)
    PoissonChebMaxIter = config.read < int >("PoissonChebMaxIter",10);
    PoissonChebRescaleEigMin = config.read < double >("PoissonChebRescaleEigMin",1.0);
    PoissonChebRescaleEigMax = config.read < double >("PoissonChebRescaleEigMax",1.0);
    // take the injection of the particless
    Vinj = config.read < double >("Vinj",0.0);

    // take the output cycles
    FieldOutputCycle = config.read < int >("FieldOutputCycle",100);
    ParticlesOutputCycle = config.read < int >("ParticlesOutputCycle",0);
    FieldOutputTag     =   config.read <string>("FieldOutputTag","");
    ParticlesOutputTag =   config.read <string>("ParticlesOutputTag","");
    MomentsOutputTag   =   config.read <string>("MomentsOutputTag","");
    TestParticlesOutputCycle = config.read < int >("TestPartOutputCycle",0);
    testPartFlushCycle = config.read < int >("TestParticlesOutputCycle",10);
    RestartOutputCycle = config.read < int >("RestartOutputCycle",5000);
    DiagnosticsOutputCycle = config.read < int >("DiagnosticsOutputCycle", FieldOutputCycle);
    ParaviewScriptPath     =   config.read <string>("ParaviewScriptPath", "");
    CallFinalize = config.read < bool >("CallFinalize", true);
  }

  //read everything from input file, if restart is true, overwrite the setting - bug fixing

  restart_status = 0;
  last_cycle = -1;
  c = config.read < double >("c",1.0);

#ifdef BATSRUS
  // set grid size and resolution based on the initial file from fluid code
  Lx =  getFluidLx();
  Ly =  getFluidLy();
  Lz =  getFluidLz();
  nxc = getFluidNxc();
  nyc = getFluidNyc();
  nzc = getFluidNzc();
#else
  Lx = config.read < double >("Lx",10.0);
  Ly = config.read < double >("Ly",10.0);
  Lz = config.read < double >("Lz",10.0);
  nxc = config.read < int >("nxc",64);
  nyc = config.read < int >("nyc",64);
  nzc = config.read < int >("nzc",64);
#endif
  XLEN = config.read < int >("XLEN",1);
  YLEN = config.read < int >("YLEN",1);
  ZLEN = config.read < int >("ZLEN",1);
  PERIODICX = config.read < bool >("PERIODICX",true);
  PERIODICY = config.read < bool >("PERIODICY",true);
  PERIODICZ = config.read < bool >("PERIODICZ",true);

  PERIODICX_P = config.read < bool >("PERIODICX_P",PERIODICX);
  PERIODICY_P = config.read < bool >("PERIODICY_P",PERIODICY);
  PERIODICZ_P = config.read < bool >("PERIODICZ_P",PERIODICZ);

  x_center_dipole = config.read < double >("x_center_dipole",5.0);
  y_center_dipole = config.read < double >("y_center_dipole",5.0);
  z_center_dipole = config.read < double >("z_center_dipole",5.0);
  x_center_planet = config.read < double >("x_center_planet",5.0);
  y_center_planet = config.read < double >("y_center_planet",5.0);
  z_center_planet = config.read < double >("z_center_planet",5.0);
  L_square = config.read < double >("L_square",5.0);

  // ── Planet reflection model: 0=specular (default), 1=diffuse (isotropic) ──
  planetReflectionType = config.read<int>("planetReflectionType", 0);

  // ── Exosphere / planet species parameters ──
  numSolarWindSpecies       = config.read<int>("ns_solar_wind", ns);       // default: all species are SW
  numPlanetarySpecies       = config.read<int>("ns_planetary", 0);        // default: no planetary species
  enableExosphereInjection  = config.read<int>("AddExosphereInjection", 0); // 0=off, 1=on
  maxInjectionRadius        = config.read<double>("RmaxExosphereInjection", 3.0);
  maxExosphereParticlesPerSpecies = config.read<int>("MaxExosphereParticlesPerSpecies", 0); // 0 = unlimited

  // Allocate arrays for planetary neutral species parameters (at least 1 to avoid null)
  const int numNeutralSpecies = std::max(numPlanetarySpecies, 1);
  neutralSurfaceDensity     = std::make_unique<double[]>(numNeutralSpecies);
  exosphericScaleHeight     = std::make_unique<double[]>(numNeutralSpecies);
  photoionizationFrequency  = std::make_unique<double[]>(numNeutralSpecies);
  macroParticleWeightRatio  = std::make_unique<double[]>(numNeutralSpecies);

  // Initialize to safe defaults
  for (int i = 0; i < numNeutralSpecies; i++) {
    neutralSurfaceDensity[i]    = 0.0;
    exosphericScaleHeight[i]    = 1.0;
    photoionizationFrequency[i] = 0.0;
    macroParticleWeightRatio[i] = 1.0;
  }

  if (numPlanetarySpecies > 0) {
    array_double NeutralSurfaceDensity0    = config.read<array_double>("NeutralSurfaceDensity");
    array_double ExosphericScaleHeight0    = config.read<array_double>("ExosphericScaleHeight");
    array_double PhotoionizationFrequency0 = config.read<array_double>("PhotoionizationFrequency");
    array_double MacroParticleWeightRatio0 = config.read<array_double>("MacroParticleWeightRatio");
    neutralSurfaceDensity[0] = NeutralSurfaceDensity0.a;  exosphericScaleHeight[0] = ExosphericScaleHeight0.a;  photoionizationFrequency[0] = PhotoionizationFrequency0.a;  macroParticleWeightRatio[0] = MacroParticleWeightRatio0.a;
    if (numPlanetarySpecies > 1) { neutralSurfaceDensity[1] = NeutralSurfaceDensity0.b; exosphericScaleHeight[1] = ExosphericScaleHeight0.b; photoionizationFrequency[1] = PhotoionizationFrequency0.b; macroParticleWeightRatio[1] = MacroParticleWeightRatio0.b; }
    if (numPlanetarySpecies > 2) { neutralSurfaceDensity[2] = NeutralSurfaceDensity0.c; exosphericScaleHeight[2] = ExosphericScaleHeight0.c; photoionizationFrequency[2] = PhotoionizationFrequency0.c; macroParticleWeightRatio[2] = MacroParticleWeightRatio0.c; }
    if (numPlanetarySpecies > 3) { neutralSurfaceDensity[3] = NeutralSurfaceDensity0.d; exosphericScaleHeight[3] = ExosphericScaleHeight0.d; photoionizationFrequency[3] = PhotoionizationFrequency0.d; macroParticleWeightRatio[3] = MacroParticleWeightRatio0.d; }
    if (numPlanetarySpecies > 4) { neutralSurfaceDensity[4] = NeutralSurfaceDensity0.e; exosphericScaleHeight[4] = ExosphericScaleHeight0.e; photoionizationFrequency[4] = PhotoionizationFrequency0.e; macroParticleWeightRatio[4] = MacroParticleWeightRatio0.e; }
    if (numPlanetarySpecies > 5) { neutralSurfaceDensity[5] = NeutralSurfaceDensity0.f; exosphericScaleHeight[5] = ExosphericScaleHeight0.f; photoionizationFrequency[5] = PhotoionizationFrequency0.f; macroParticleWeightRatio[5] = MacroParticleWeightRatio0.f; }
  }


  uth = std::make_unique<double[]>(ns);
  vth = std::make_unique<double[]>(ns);
  wth = std::make_unique<double[]>(ns);
  u0 = std::make_unique<double[]>(ns);
  v0 = std::make_unique<double[]>(ns);
  w0 = std::make_unique<double[]>(ns);

  array_double uth0 = config.read < array_double > ("uth");
  array_double vth0 = config.read < array_double > ("vth");
  array_double wth0 = config.read < array_double > ("wth");
  array_double u00 = config.read < array_double > ("u0");
  array_double v00 = config.read < array_double > ("v0");
  array_double w00 = config.read < array_double > ("w0");

  uth[0] = uth0.a;
  vth[0] = vth0.a;
  wth[0] = wth0.a;
  u0[0] = u00.a;
  v0[0] = v00.a;
  w0[0] = w00.a;
  if (ns > 1) {
    uth[1] = uth0.b;
    vth[1] = vth0.b;
    wth[1] = wth0.b;
    u0[1] = u00.b;
    v0[1] = v00.b;
    w0[1] = w00.b;
  }
  if (ns > 2) {
    uth[2] = uth0.c;
    vth[2] = vth0.c;
    wth[2] = wth0.c;
    u0[2] = u00.c;
    v0[2] = v00.c;
    w0[2] = w00.c;
  }
  if (ns > 3) {
    uth[3] = uth0.d;
    vth[3] = vth0.d;
    wth[3] = wth0.d;
    u0[3] = u00.d;
    v0[3] = v00.d;
    w0[3] = w00.d;
  }
  if (ns > 4) {
    uth[4] = uth0.e;
    vth[4] = vth0.e;
    wth[4] = wth0.e;
    u0[4] = u00.e;
    v0[4] = v00.e;
    w0[4] = w00.e;
  }
  if (ns > 5) {
    uth[5] = uth0.f;
    vth[5] = vth0.f;
    wth[5] = wth0.f;
    u0[5] = u00.f;
    v0[5] = v00.f;
    w0[5] = w00.f;
  }

  if (nstestpart > 0) {
		array_double pitch_angle0 = config.read < array_double > ("pitch_angle");
		array_double energy0 	  = config.read < array_double > ("energy");
		pitch_angle = std::make_unique<double[]>(nstestpart);
		energy      = std::make_unique<double[]>(nstestpart);
		if (nstestpart > 0) {
			pitch_angle[0] = pitch_angle0.a;
			energy[0] 	   = energy0.a;
		}
		if (nstestpart > 1) {
			pitch_angle[1] = pitch_angle0.b;
			energy[1] 	   = energy0.b;
		}
		if (nstestpart > 2) {
			pitch_angle[2] = pitch_angle0.c;
			energy[2] 	   = energy0.c;
		}
		if (nstestpart > 3) {
			pitch_angle[3] = pitch_angle0.d;
			energy[3] 	   = energy0.d;
		}
		if (nstestpart > 4) {
			pitch_angle[4] = pitch_angle0.e;
			energy[4] 	   = energy0.e;
		}
		if (nstestpart > 5) {
			pitch_angle[5] = pitch_angle0.f;
			energy[5] 	   = energy0.f;
		}
		if (nstestpart > 6) {
			pitch_angle[6] = pitch_angle0.g;
			energy[6] 	   = energy0.g;
		}
		if (nstestpart > 7) {
			pitch_angle[7] = pitch_angle0.h;
			energy[7] 	   = energy0.h;
		}
  }


  npcelx = std::make_unique<int[]>(ns+nstestpart);
  npcely = std::make_unique<int[]>(ns+nstestpart);
  npcelz = std::make_unique<int[]>(ns+nstestpart);
  qom = std::make_unique<double[]>(ns+nstestpart);
  array_int npcelx0 = config.read < array_int > ("npcelx");
  array_int npcely0 = config.read < array_int > ("npcely");
  array_int npcelz0 = config.read < array_int > ("npcelz");
  array_double qom0 = config.read < array_double > ("qom");
  npcelx[0] = npcelx0.a;
  npcely[0] = npcely0.a;
  npcelz[0] = npcelz0.a;
  qom[0]	  = qom0.a;
  int ns_tot =ns+nstestpart;
  if (ns_tot > 1) {
    npcelx[1] = npcelx0.b;
    npcely[1] = npcely0.b;
    npcelz[1] = npcelz0.b;
    qom[1]	= qom0.b;
  }
  if (ns_tot > 2) {
    npcelx[2] = npcelx0.c;
    npcely[2] = npcely0.c;
    npcelz[2] = npcelz0.c;
    qom[2] 	= qom0.c;
  }
  if (ns_tot > 3) {
    npcelx[3] = npcelx0.d;
    npcely[3] = npcely0.d;
    npcelz[3] = npcelz0.d;
    qom[3] 	= qom0.d;
  }
  if (ns_tot > 4) {
    npcelx[4] = npcelx0.e;
    npcely[4] = npcely0.e;
    npcelz[4] = npcelz0.e;
    qom[4] 	= qom0.e;
  }
  if (ns_tot > 5) {
    npcelx[5] = npcelx0.f;
    npcely[5] = npcely0.f;
    npcelz[5] = npcelz0.f;
    qom[5] 	= qom0.f;
  }
  if (ns_tot > 6) {
    npcelx[6] = npcelx0.g;
    npcely[6] = npcely0.g;
    npcelz[6] = npcelz0.g;
    qom[6] 	= qom0.g;
  }
  if (ns_tot > 7) {
    npcelx[7] = npcelx0.h;
    npcely[7] = npcely0.h;
    npcelz[7] = npcelz0.h;
    qom[7] 	= qom0.h;
  }
  if (ns_tot > 8) {
    npcelx[8] = npcelx0.i;
    npcely[8] = npcely0.i;
    npcelz[8] = npcelz0.i;
    qom[8] 	= qom0.i;
  }
  if (ns_tot > 9) {
    npcelx[9] = npcelx0.j;
    npcely[9] = npcely0.j;
    npcelz[9] = npcelz0.j;
    qom[9] 	= qom0.j;
  }
  if (ns_tot > 10) {
    npcelx[10] = npcelx0.k;
    npcely[10] = npcely0.k;
    npcelz[10] = npcelz0.k;
    qom[10] 	 = qom0.k;
  }
  if (ns_tot > 11) {
    npcelx[11] = npcelx0.l;
    npcely[11] = npcely0.l;
    npcelz[11] = npcelz0.l;
    qom[11] 	 = qom0.l;
  }



  //verbose = config.read < bool > ("verbose",false);

  // PHI Electrostatic Potential
  bcPHIfaceXright = config.read < int >("bcPHIfaceXright",1);
  bcPHIfaceXleft  = config.read < int >("bcPHIfaceXleft",1);
  bcPHIfaceYright = config.read < int >("bcPHIfaceYright",1);
  bcPHIfaceYleft  = config.read < int >("bcPHIfaceYleft",1);
  bcPHIfaceZright = config.read < int >("bcPHIfaceZright",1);
  bcPHIfaceZleft  = config.read < int >("bcPHIfaceZleft",1);

  // EM field boundary condition
  bcEMfaceXright = config.read < int >("bcEMfaceXright");
  bcEMfaceXleft  = config.read < int >("bcEMfaceXleft");
  bcEMfaceYright = config.read < int >("bcEMfaceYright");
  bcEMfaceYleft  = config.read < int >("bcEMfaceYleft");
  bcEMfaceZright = config.read < int >("bcEMfaceZright");
  bcEMfaceZleft  = config.read < int >("bcEMfaceZleft");

  // EM field absorbing boundary condition parameters
  yes_sal  = config.read < int >("yes_sal",0);
  n_layers_sal  = config.read < int >("n_layers_sal",3);


  /*  ------------------------------------------------------------------- */
  /*  Electric and Magnetic field boundary conditions for BCface          */
  /*  ------------------------------------------------------------------- */
  /*  bcEM* == 0 : Perfect Electric Conductor (PEC)                       */
  /*    - E_tangential = 0   (Dirichlet, bc=1)                            */
  /*    - E_normal     = free (Neumann,  bc=2)                            */
  /*    - B_tangential = free (Neumann,  bc=2)                            */
  /*    - B_normal     = 0   (Dirichlet, bc=1)                            */
  /*  bcEM* != 0 : Perfect Magnetic Conductor (PMC, "perfect mirror")     */
  /*    - E_tangential = free (Neumann,  bc=2)                            */
  /*    - E_normal     = 0   (Dirichlet, bc=1)                            */
  /*    - B_tangential = 0   (Dirichlet, bc=1)                            */
  /*    - B_normal     = free (Neumann,  bc=2)                            */
  /*                                                                      */
  /*  Convention: bc=1 -> Dirichlet (value fixed to 0)                    */
  /*             bc=2 -> Neumann   (derivative fixed, value free)         */
  /*                                                                      */
  /*  Note: E and B always get complementary bc values on every face.     */
  /*  Face indices: 0=Xright, 1=Xleft, 2=Yright, 3=Yleft, 4=Zright, 5=Zleft */
  /*  ------------------------------------------------------------------- */

  /* Ex Bx component: normal on X-faces (bc=2 for PEC), tangential on Y/Z-faces (bc=1 for PEC) */
  bcEx[0] = bcEMfaceXright == 0 ? 2 : 1;   bcBx[0] = bcEMfaceXright == 0 ? 1 : 2;
  bcEx[1] = bcEMfaceXleft  == 0 ? 2 : 1;   bcBx[1] = bcEMfaceXleft  == 0 ? 1 : 2;
  bcEx[2] = bcEMfaceYright == 0 ? 1 : 2;   bcBx[2] = bcEMfaceYright == 0 ? 2 : 1;
  bcEx[3] = bcEMfaceYleft  == 0 ? 1 : 2;   bcBx[3] = bcEMfaceYleft  == 0 ? 2 : 1;
  bcEx[4] = bcEMfaceZright == 0 ? 1 : 2;   bcBx[4] = bcEMfaceZright == 0 ? 2 : 1;
  bcEx[5] = bcEMfaceZleft  == 0 ? 1 : 2;   bcBx[5] = bcEMfaceZleft  == 0 ? 2 : 1;
  /* Ey By component: tangential on X-faces (bc=1 for PEC), normal on Y-faces (bc=2 for PEC), tangential on Z-faces */
  bcEy[0] = bcEMfaceXright == 0 ? 1 : 2;   bcBy[0] = bcEMfaceXright == 0 ? 2 : 1;
  bcEy[1] = bcEMfaceXleft  == 0 ? 1 : 2;   bcBy[1] = bcEMfaceXleft  == 0 ? 2 : 1;
  bcEy[2] = bcEMfaceYright == 0 ? 2 : 1;   bcBy[2] = bcEMfaceYright == 0 ? 1 : 2;
  bcEy[3] = bcEMfaceYleft  == 0 ? 2 : 1;   bcBy[3] = bcEMfaceYleft  == 0 ? 1 : 2;
  bcEy[4] = bcEMfaceZright == 0 ? 1 : 2;   bcBy[4] = bcEMfaceZright == 0 ? 2 : 1;
  bcEy[5] = bcEMfaceZleft  == 0 ? 1 : 2;   bcBy[5] = bcEMfaceZleft  == 0 ? 2 : 1;
  /* Ez Bz component: tangential on X/Y-faces (bc=1 for PEC), normal on Z-faces (bc=2 for PEC) */
  bcEz[0] = bcEMfaceXright == 0 ? 1 : 2;   bcBz[0] = bcEMfaceXright == 0 ? 2 : 1;
  bcEz[1] = bcEMfaceXleft  == 0 ? 1 : 2;   bcBz[1] = bcEMfaceXleft  == 0 ? 2 : 1;
  bcEz[2] = bcEMfaceYright == 0 ? 1 : 2;   bcBz[2] = bcEMfaceYright == 0 ? 2 : 1;
  bcEz[3] = bcEMfaceYleft  == 0 ? 1 : 2;   bcBz[3] = bcEMfaceYleft  == 0 ? 2 : 1;
  bcEz[4] = bcEMfaceZright == 0 ? 2 : 1;   bcBz[4] = bcEMfaceZright == 0 ? 1 : 2;
  bcEz[5] = bcEMfaceZleft  == 0 ? 2 : 1;   bcBz[5] = bcEMfaceZleft  == 0 ? 1 : 2;

  // Particles Boundary condition
  bcPfaceXright = config.read < int >("bcPfaceXright",1);
  bcPfaceXleft  = config.read < int >("bcPfaceXleft",1);
  bcPfaceYright = config.read < int >("bcPfaceYright",1);
  bcPfaceYleft  = config.read < int >("bcPfaceYleft",1);
  bcPfaceZright = config.read < int >("bcPfaceZright",1);
  bcPfaceZleft  = config.read < int >("bcPfaceZleft",1);

  // E field inflow BC: master switch for applying inflow BCs in GMRes image
  applyInflowBcsEImage = config.read<int>("ApplyInflowBcsEImage", 1);

  if (RESTART1) {               // you are restarting 
    RestartDirName = config.read < string > ("RestartDirName","data");
    restart_status = 1;

    // Delegate cycle reading to RestartReader (in inputoutput/)
    last_cycle = RestartReader::readLastCycle(RestartDirName);
  }

  /*
  TrackParticleID = new bool[ns];
  array_bool TrackParticleID0 = config.read < array_bool > ("TrackParticleID");
  TrackParticleID[0] = TrackParticleID0.a;
  if (ns > 1)
    TrackParticleID[1] = TrackParticleID0.b;
  if (ns > 2)
    TrackParticleID[2] = TrackParticleID0.c;
  if (ns > 3)
    TrackParticleID[3] = TrackParticleID0.d;
  if (ns > 4)
    TrackParticleID[4] = TrackParticleID0.e;
  if (ns > 5)
    TrackParticleID[5] = TrackParticleID0.f;
    */
}

bool Collective::field_output_is_off()const
{
  return (FieldOutputCycle <= 0);
}

bool Collective::particle_output_is_off()const
{
  return getParticlesOutputCycle() <= 0;
}
bool Collective::testparticle_output_is_off()const
{
  return getTestParticlesOutputCycle() <= 0;
}



void Collective::read_field_restart(
    const VCtopology3D* vct,
    const Grid* grid,
    arr3_double Bxn, arr3_double Byn, arr3_double Bzn,
    arr3_double Ex, arr3_double Ey, arr3_double Ez,
    array4_double* rhons_, int ns)const
{
    // Delegate to RestartReader (implementation in inputoutput/RestartReader.cpp)
    RestartReader::readFields(vct, grid, Bxn, Byn, Bzn, Ex, Ey, Ez,
                              rhons_, ns, getRestartDirName(), last_cycle);
}

void Collective::read_particles_restart(
    const VCtopology3D* vct,
    int species_number,
    vector_double& u,
    vector_double& v,
    vector_double& w,
    vector_double& q,
    vector_double& x,
    vector_double& y,
    vector_double& z,
    vector_double& t)const
{
    // Delegate to RestartReader (implementation in inputoutput/RestartReader.cpp)
    RestartReader::readParticles(vct, species_number, u, v, w, q, x, y, z, t,
                                 getRestartDirName(), last_cycle);
}



/*! constructor */
Collective::Collective(int argc, char **argv) {
  if (argc < 2) {
    inputfile = "inputfile";
    RESTART1 = false;
  }
  else if (argc < 3) {
    inputfile = argv[1];
    RESTART1 = false;
  }
  else {
    if (strcmp(argv[1], "restart") == 0) {
      inputfile = argv[2];
      RESTART1 = true;
    }
    else if (strcmp(argv[2], "restart") == 0) {
      inputfile = argv[1];
      RESTART1 = true;
    }
    else {
      cout << "Error: syntax error in mpirun arguments. Did you mean to 'restart' ?" << endl;
      return;
    }

    if(MPIdata::get_rank() == 0)std::cout << "Restarting..." << endl;
  }
  ReadInput(inputfile);
  init_derived_parameters();
}

void Collective::init_derived_parameters()
{
  /*! fourpi = 4 greek pi */
  fourpi = 16.0 * atan(1.0);
  /*! dx = space step - X direction */
  dx = Lx / (double) nxc;
  /*! dy = space step - Y direction */
  dy = Ly / (double) nyc;
  /*! dz = space step - Z direction */
  dz = Lz / (double) nzc;
  /*! npcel = number of particles per cell */
  npcel = std::make_unique<int[]>(ns+nstestpart);
  /*! np = number of particles of different species */
  //np = new int[ns];
  /*! npMax = maximum number of particles of different species */
  //npMax = new int[ns];

  /* quantities per process */

  // check that procs divides grid
  // (this restriction should be removed).
  //
  if(0==MPIdata::get_rank())
  {
    fflush(stdout);
    bool xerror = false;
    bool yerror = false;
    bool zerror = false;
    if(nxc % XLEN) xerror=true;
    if(nyc % YLEN) yerror=true;
    if(nzc % ZLEN) zerror=true;
    if(xerror) warning_printf("XLEN=%d does not divide nxc=%d\n", XLEN,nxc);
    if(yerror) warning_printf("YLEN=%d does not divide nyc=%d\n", YLEN,nyc);
    if(zerror) warning_printf("ZLEN=%d does not divide nzc=%d\n", ZLEN,nzc);
    fflush(stdout);
    bool error = (xerror||yerror||zerror);
    // Comment out this check if your postprocessing code does not
    // require the field output subarrays to be the same size.
    // Alternatively, you could modify the output routine to pad
    // with zeros...
    //if(error)
    //{
    //  eprintf("For WriteMethod=default processor dimensions "
    //          "must divide mesh cell dimensions");
    //}
  }

  int num_cells_r = nxc*nyc*nzc;
  //num_procs = XLEN*YLEN*ZLEN;
  //ncells_rs = nxc_rs*nyc_rs*nzc_rs;

  for (int i = 0; i < (ns+nstestpart); i++)
  {
    npcel[i] = npcelx[i] * npcely[i] * npcelz[i];
    //np[i] = npcel[i] * num_cells;
    //nop_rs[i] = npcel[i] * ncells_rs;
    //maxnop_rs[i] = NpMaxNpRatio * nop_rs[i];
    // INT_MAX is about 2 billion, surely enough
    // to index the particles in a single MPI process:
    //assert_le(NpMaxNpRatio * npcel[i] * ncells_proper_per_proc , double(INT_MAX));
    //double npMaxi = (NpMaxNpRatio * np[i]);
    //npMax[i] = (int) npMaxi;
  }
}

/*! Print Simulation Parameters */
void Collective::Print() {
  cout << endl;
  cout << "Simulation Parameters" << endl;
  cout << "---------------------" << endl;
  cout << "Number of species    = " << ns << endl;
  for (int i = 0; i < ns; i++)
    cout << "qom[" << i << "] = " << qom[i] << endl;
  cout << "x-Length                 = " << Lx << endl;
  cout << "y-Length                 = " << Ly << endl;
  cout << "z-Length                 = " << Lz << endl;
  cout << "Number of cells (x)      = " << nxc << endl;
  cout << "Number of cells (y)      = " << nyc << endl;
  cout << "Number of cells (z)      = " << nzc << endl;
  cout << "Time step                = " << dt << endl;
  cout << "Number of cycles         = " << ncycles << endl;
  cout << "Results saved in  : " << SaveDirName << endl;
  cout << "Case type         : " << Case << endl;
  cout << "Simulation name   : " << SimName << endl;
  cout << "Smoothing         : " << (Smooth == 1.0 ? "off" : "on") << " (alpha=" << Smooth << ", Niter=" << SmoothNiter << ")" << endl;
  cout << "---------------------" << endl;
  cout << "EM Field Boundary Conditions" << endl;
  cout << "---------------------" << endl;
  // helper lambda to decode bcEMface codes: 0=perfect conductor, 2=open/inflow
  auto bcName = [](int code) -> const char* {
    switch(code) {
      case 0: return "perfect conductor";
      case 1: return "Dirichlet (first order)";
      case 2: return "open/inflow (Neumann)";
      default: return "unknown";
    }
  };
  cout << "Xleft  : " << bcEMfaceXleft  << " (" << bcName(bcEMfaceXleft)  << ")" << endl;
  cout << "Xright : " << bcEMfaceXright << " (" << bcName(bcEMfaceXright) << ")" << endl;
  cout << "Yleft  : " << bcEMfaceYleft  << " (" << bcName(bcEMfaceYleft)  << ")" << endl;
  cout << "Yright : " << bcEMfaceYright << " (" << bcName(bcEMfaceYright) << ")" << endl;
  cout << "Zleft  : " << bcEMfaceZleft  << " (" << bcName(bcEMfaceZleft)  << ")" << endl;
  cout << "Zright : " << bcEMfaceZright << " (" << bcName(bcEMfaceZright) << ")" << endl;
  cout << "SAL (absorbing layer): " << (yes_sal ? "yes" : "no");
  if (yes_sal) cout << ", n_layers=" << n_layers_sal;
  cout << endl;
  cout << "---------------------" << endl;
  cout << "Particle Boundary Conditions" << endl;
  cout << "---------------------" << endl;
  auto bcPName = [](int code) -> const char* {
    switch(code) {
      case 0: return "exit";
      case 1: return "perfect mirror";
      case 2: return "reemission";
      case 3: return "open BC outflow";
      case 4: return "open BC inflow";
      default: return "unknown";
    }
  };
  cout << "Xleft  : " << bcPfaceXleft  << " (" << bcPName(bcPfaceXleft)  << ")" << endl;
  cout << "Xright : " << bcPfaceXright << " (" << bcPName(bcPfaceXright) << ")" << endl;
  cout << "Yleft  : " << bcPfaceYleft  << " (" << bcPName(bcPfaceYleft)  << ")" << endl;
  cout << "Yright : " << bcPfaceYright << " (" << bcPName(bcPfaceYright) << ")" << endl;
  cout << "Zleft  : " << bcPfaceZleft  << " (" << bcPName(bcPfaceZleft)  << ")" << endl;
  cout << "Zright : " << bcPfaceZright << " (" << bcPName(bcPfaceZright) << ")" << endl;
  cout << "E inflow BCs in GMRes : " << (applyInflowBcsEImage ? "yes" : "no")
       << " (applied per-face where bcPface == 2)" << endl;
  cout << "---------------------" << endl;
  cout << "Field Corrections" << endl;
  cout << "---------------------" << endl;
  cout << "Poisson div(E) correction  : " << PoissonCorrection;
  if (PoissonCorrection == "yes") cout << ", every " << PoissonCorrectionCycle << " cycles (in calculateE)";
  cout << endl;
  cout << "div(B) cleaning            : " << divBCorrection;
  if (divBCorrection == "yes") cout << ", every " << divBCorrectionCycle << " cycles (in calculateB)";
  cout << endl;
  cout << "---------------------" << endl;
  cout << "Planet Boundary" << endl;
  cout << "---------------------" << endl;
  cout << "Reflection type            : " << planetReflectionType
       << (planetReflectionType == 0 ? " (specular)" : " (diffuse/isotropic)") << endl;
  cout << "---------------------" << endl;
  cout << "Exosphere Ionization" << endl;
  cout << "---------------------" << endl;
  if (enableExosphereInjection && numPlanetarySpecies > 0) {
    cout << "Status                     : enabled" << endl;
    cout << "Solar wind species         : " << numSolarWindSpecies << " (indices 0.." << numSolarWindSpecies-1 << ")" << endl;
    cout << "Planetary species          : " << numPlanetarySpecies << " (indices " << numSolarWindSpecies << ".." << ns-1 << ")" << endl;
    cout << "Max injection radius       : " << maxInjectionRadius << " d_i" << endl;
    cout << "Planet radius (L_square)   : " << L_square << " d_i" << endl;
    for (int i = 0; i < numPlanetarySpecies; i++) {
      int globalIdx = numSolarWindSpecies + i;
      cout << "  Species " << globalIdx << " (neutral " << i << "):" << endl;
      cout << "    NeutralSurfaceDensity    = " << neutralSurfaceDensity[i] << " n_sw" << endl;
      cout << "    ExosphericScaleHeight    = " << exosphericScaleHeight[i] << " d_i" << endl;
      cout << "    PhotoionizationFrequency = " << photoionizationFrequency[i] << " wci" << endl;
      cout << "    MacroParticleWeightRatio = " << macroParticleWeightRatio[i] << endl;
      cout << "    qom                      = " << qom[globalIdx] << endl;
      cout << "    uth/vth/wth              = " << uth[globalIdx] << " / " << vth[globalIdx] << " / " << wth[globalIdx] << endl;
    }
  } else {
    cout << "Status                     : disabled" << endl;
  }
  cout << "---------------------" << endl;
  cout << "Check Simulation Constraints" << endl;
  cout << "---------------------" << endl;
  cout << "Accuracy Constraint:  " << endl;
  for (int i = 0; i < ns; i++) {
    cout << "u_th < dx/dt species " << i << ".....";
    if (uth[i] < (dx / dt))
      cout << "OK" << endl;
    else
      cout << "NOT SATISFIED. STOP THE SIMULATION." << endl;

    cout << "v_th < dy/dt species " << i << "......";
    if (vth[i] < (dy / dt))
      cout << "OK" << endl;
    else
      cout << "NOT SATISFIED. STOP THE SIMULATION." << endl;
  }
  cout << endl;
  cout << "Finite Grid Stability Constraint:  ";
  cout << endl;
  for (int is = 0; is < ns; is++) {
    if (uth[is] * dt / dx > .1)
      cout << "OK u_th*dt/dx (species " << is << ") = " << uth[is] * dt / dx << " > .1" << endl;
    else
      cout << "WARNING. u_th*dt/dx (species " << is << ") = " << uth[is] * dt / dx << " < .1" << endl;

    if (vth[is] * dt / dy > .1)
      cout << "OK v_th*dt/dy (species " << is << ") = " << vth[is] * dt / dy << " > .1" << endl;
    else
      cout << "WARNING. v_th*dt/dy (species " << is << ") = " << vth[is] * dt / dy << " < .1"  << endl;

  }


}
/*! Print Simulation Parameters */
void Collective::save() {
  string temp;
  temp = SaveDirName + "/SimulationData.txt";
  ofstream my_file(temp.c_str());
  my_file << "---------------------------" << endl;
  my_file << "-  Simulation Parameters  -" << endl;
  my_file << "---------------------------" << endl;

  my_file << "Number of species    = " << ns << endl;
  for (int i = 0; i < ns; i++)
    my_file << "qom[%d] = " << qom[i] << endl;
  my_file << "---------------------------" << endl;
  my_file << "x-Length                 = " << Lx << endl;
  my_file << "y-Length                 = " << Ly << endl;
  my_file << "z-Length                 = " << Lz << endl;
  my_file << "Number of cells (x)      = " << nxc << endl;
  my_file << "Number of cells (y)      = " << nyc << endl;
  my_file << "Number of cells (z)      = " << nzc << endl;
  my_file << "---------------------------" << endl;
  my_file << "Time step                = " << dt << endl;
  my_file << "Number of cycles         = " << ncycles << endl;
  my_file << "---------------------------" << endl;
  for (int is = 0; is < ns; is++){
    my_file << "rho init species   " << is << " = " << rhoINIT[is] << endl;
    my_file << "rho inject species " << is << " = " << rhoINJECT[is]  << endl;
  }
  my_file << "current sheet thickness  = " << delta << endl;
  my_file << "B0x                      = " << B0x << endl;
  my_file << "BOy                      = " << B0y << endl;
  my_file << "B0z                      = " << B0z << endl;
  my_file << "---------------------------" << endl;
  my_file << "Smooth                   = " << Smooth << endl;
  my_file << "SmoothNiter              = " << SmoothNiter<< endl;
  my_file << "GMRES error tolerance    = " << GMREStol << endl;
  my_file << "CG error tolerance       = " << CGtol << endl;
  my_file << "Mover error tolerance    = " << NiterMover << endl;
  my_file << "---------------------------" << endl;
  my_file << "Results saved in: " << SaveDirName << endl;
  my_file << "Restart saved in: " << RestartDirName << endl;
  my_file << "---------------------" << endl;
  my_file.close();

}

