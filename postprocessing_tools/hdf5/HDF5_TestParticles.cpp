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

//output all test particles to legacy Polydata format

#include "hdf5.h"
//#include "../../include/Alloc.h"
//#include "math.h"
#include <iostream>
#include <fstream>
#include <string>
#include <sstream>

using std::string;
using std::stringstream;
using std::ofstream;
using std::cout;
using std::endl;



int main (int argc, char **argv) {
	if(argc==1){
		cout << "Provide initial cycle, step cycle, and last cycle in the command line" << endl;
		cout << "ie, HDF5_TestParticles 0 100 1000" << endl;
		return(-1);
	}
	
	// cycle we want to open
	int start_cycle;
	int step_cycle;
	int end_cycle;
	// 3 inputs first cycle, step, and end cycle
	sscanf(argv[1],"%d",&start_cycle);
	sscanf(argv[2],"%d",&step_cycle);
	sscanf(argv[3],"%d",&end_cycle);
	
	
	// hdf stuff 
	hid_t    file_id;
	hid_t    dataset_id;
	herr_t   status;
	// Open the  settings file 
	file_id = H5Fopen("settings.hdf", H5F_ACC_RDWR, H5P_DEFAULT);
	if (file_id < 0){
		cout << "couldn't open file: settings.hdf" << endl;
		return -1;
	}
	// First read the topology
	int nproc;
    dataset_id = H5Dopen2(file_id, "/topology/Nprocs", H5P_DEFAULT);  // HDF 1.8.8
	status = H5Dread(dataset_id, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT,&nproc);
	status = H5Dclose(dataset_id);
	// read nxc
	int nxc;
	dataset_id = H5Dopen2(file_id, "/collective/Nxc", H5P_DEFAULT); // HDF 1.8.8
	status = H5Dread(dataset_id, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT,&nxc);
	status = H5Dclose(dataset_id);
	// read nyc
	int nyc;
	dataset_id = H5Dopen2(file_id, "/collective/Nyc", H5P_DEFAULT); // HDF 1.8.8
	status = H5Dread(dataset_id, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT,&nyc);
	status = H5Dclose(dataset_id);
	// read nyc
	int nzc;
	dataset_id = H5Dopen2(file_id, "/collective/Nzc", H5P_DEFAULT); // HDF 1.8.8
	status = H5Dread(dataset_id, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT,&nzc);
	status = H5Dclose(dataset_id);
	// read ns
	int ns;
	dataset_id = H5Dopen2(file_id, "/collective/Ns", H5P_DEFAULT); // HDF 1.8.8
	status = H5Dread(dataset_id, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT,&ns);
	status = H5Dclose(dataset_id);

	// read NsTestPart
    int NsTestPart;
    dataset_id = H5Dopen2(file_id, "/collective/NsTestPart", H5P_DEFAULT); // HDF 1.8.8
    status = H5Dread(dataset_id, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT,&NsTestPart);
    status = H5Dclose(dataset_id);

    int pcl[NsTestPart];
    int pclx, pcly, pclz,maxPcl=0;
    stringstream ss;

    for(int si=0;si<NsTestPart;si++){
    	ss.str("");
    	ss << "/collective/testspecies_" << (si+ns) << "/Npcelx";
    	dataset_id = H5Dopen2(file_id, ss.str().c_str(), H5P_DEFAULT); // HDF 1.8.8
    	status = H5Dread(dataset_id, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT,&pclx);//cout << "pclx=" << pclx<<endl;
    	status = H5Dclose(dataset_id);

    	ss.str("");
    	ss << "/collective/testspecies_" << (si+ns) << "/Npcely";
    	dataset_id = H5Dopen2(file_id, ss.str().c_str(), H5P_DEFAULT); // HDF 1.8.8
    	status = H5Dread(dataset_id, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT,&pcly);//cout << "pcly=" << pcly<<endl;
    	status = H5Dclose(dataset_id);

    	ss.str("");
    	ss << "/collective/testspecies_" << (si+ns) << "/Npcelz";
    	dataset_id = H5Dopen2(file_id, ss.str().c_str(), H5P_DEFAULT); // HDF 1.8.8
    	status = H5Dread(dataset_id, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT,&pclz);//cout << "pclz=" << pclz<<endl;
    	status = H5Dclose(dataset_id);

    	pcl[si]=pclx*pcly*pclz*nxc*nyc*nzc; //cout << "pcl[" << si << "]" <<pcl[si] <<endl;

    	if(pcl[si]>maxPcl) maxPcl=pcl[si];
    }
    //cout << "maxPcl=" << maxPcl <<endl;

	double *X= new double[maxPcl];
	double* Y = new double[maxPcl];
	double* Z = new double[maxPcl];
//	double* U= new double[maxPcl];
//	double* V = new double[maxPcl];
//	double* W = new double[maxPcl];
//	double* Q= new double[maxPcl];

	// at this point you can close settings
	status = H5Fclose(file_id);
	cout << " Successful in reading settings.hdf" << endl;

	for (int i=0; i < nproc; i++){
		ss.str("");ss <<  "proc"  << i << ".hdf";
		file_id = H5Fopen(ss.str().c_str(), H5F_ACC_RDWR, H5P_DEFAULT);
		if (file_id< 0){
			cout << "couldn't open file:  "<< ss.str() << endl;
			return -1;
		}
		status = H5Fclose(file_id);
	} 
	cout << " Successful in opening and closing all the proc*.hdf the first time" << endl;


	int nop;
	hid_t datatype, dataspace;
	hsize_t  dims_out[1];

	for (int i_cycle=start_cycle; i_cycle < (end_cycle+1); i_cycle+=step_cycle){
		
		cout << "****** CYCLE " << i_cycle << "******" <<  endl;
		stringstream cc;

		for(int si=0;si<NsTestPart;si++){
			int totwritepcl=0;

			cc.str("");cc << "testparticle" << (ns+si) << "_cycle" << i_cycle << ".vtk";cout << "Start writing " << cc.str() << endl;
			ofstream my_file(cc.str().c_str());

			my_file << "# vtk DataFile Version 1.0" << endl;
			my_file << "Test Particle from iPIC3D" << endl;
			my_file << "ASCII" << endl;
			my_file << "DATASET POLYDATA" << endl;
			my_file << "POINTS " << pcl[si] << " float " << endl;

			//my_file << "X   Y   Z  U   V   W    Q" << endl;

			for (int i=0; i < nproc; i++){
				ss.str("");ss <<  "proc"  << i << ".hdf";
				file_id = H5Fopen(ss.str().c_str(), H5F_ACC_RDWR, H5P_DEFAULT);
				if (file_id < 0){
					cout << "couldn't open file:  "<< ss.str() << endl;
					return -1;
				}

				cc.str("");cc << "/testparticles/species_" << (si+ns) << "/x/cycle_" << i_cycle;//cout << "reading data"<< cc.str() << endl;
				dataset_id = H5Dopen2(file_id,cc.str().c_str(), H5P_DEFAULT); // HDF 1.8.8
				datatype  = H5Dget_type(dataset_id);
				dataspace = H5Dget_space(dataset_id);
				status = H5Sget_simple_extent_dims(dataspace, dims_out, NULL);
				nop=dims_out[0];totwritepcl += nop;//cout << "nop =" << nop <<endl;
				status = H5Dread(dataset_id, datatype, H5S_ALL, H5S_ALL, H5P_DEFAULT, X);
				status = H5Dclose(dataset_id);

				cc.str("");cc << "/testparticles/species_" << (si+ns) << "/y/cycle_" << i_cycle;
				dataset_id = H5Dopen2(file_id,cc.str().c_str(), H5P_DEFAULT);
				status = H5Dread(dataset_id, datatype, H5S_ALL, H5S_ALL, H5P_DEFAULT, Y);
				status = H5Dclose(dataset_id);

				cc.str("");cc << "/testparticles/species_" << (si+ns) << "/z/cycle_" << i_cycle;
				dataset_id = H5Dopen2(file_id,cc.str().c_str(), H5P_DEFAULT);
				status = H5Dread(dataset_id, datatype, H5S_ALL, H5S_ALL, H5P_DEFAULT, Z);
				status = H5Dclose(dataset_id);

//						cc.str("");cc << "/testparticles/species_" << (si+ns) << "/u/cycle_" << i_cycle;
//						dataset_id = H5Dopen2(file_id,cc.str().c_str(), H5P_DEFAULT);
//						status = H5Dread(dataset_id, datatype, H5S_ALL, H5S_ALL, H5P_DEFAULT, U);
//						status = H5Dclose(dataset_id);
//
//						cc.str("");cc << "/testparticles/species_" << (si+ns) << "/v/cycle_" << i_cycle;
//						dataset_id = H5Dopen2(file_id,cc.str().c_str(), H5P_DEFAULT);
//						status = H5Dread(dataset_id, datatype, H5S_ALL, H5S_ALL, H5P_DEFAULT, V);
//						status = H5Dclose(dataset_id);
//
//						cc.str("");cc << "/testparticles/species_" << (si+ns) << "/w/cycle_" << i_cycle;
//						dataset_id = H5Dopen2(file_id,cc.str().c_str(), H5P_DEFAULT);
//						status = H5Dread(dataset_id, datatype, H5S_ALL, H5S_ALL, H5P_DEFAULT, W);
//						status = H5Dclose(dataset_id);
//
//
//						cc.str("");cc << "/testparticles/species_" << (si+ns) << "/q/cycle_" << i_cycle;
//						dataset_id = H5Dopen2(file_id,cc.str().c_str(), H5P_DEFAULT);
//						status = H5Dread(dataset_id, datatype, H5S_ALL, H5S_ALL, H5P_DEFAULT, Q);
// 	  					status = H5Dclose(dataset_id);


				for(int i=0;i<nop;i++){
					my_file << X[i] <<"     "<< Y[i]<<"      "<< Z[i] << endl;
				}
				status = H5Fclose(file_id);
			}

			if(totwritepcl!=pcl[si]){
				cout << "totwritepcl!=pcl[si]"<< endl;
				return -1;
			}

			my_file << "VERTICES 1 "<< totwritepcl+1 << endl;
			my_file << totwritepcl <<" ";
			for(int i=0;i<totwritepcl;i++)
				my_file << i <<" ";
			my_file <<endl;

			my_file.close();


		}
	}

		

	
		delete[] X;
		delete[] Y;
		delete[] Z;
//		delete[] U;
//		delete[] V;
//		delete[] W;
//		delete[] Q;
	
	return(0);
}




























