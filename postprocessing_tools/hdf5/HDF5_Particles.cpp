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
		cout << "ie, HDF5_particles 0 100 1000" << endl;
		return(-1);
	}
	
	// cycle we want to open
	int start_cycle;
	int step_cycle;
	int end_cycle;
	int procID;
	// 3 inputs first cycle, step, and end cycle
	sscanf(argv[1],"%d",&start_cycle);
	sscanf(argv[2],"%d",&step_cycle);
	sscanf(argv[3],"%d",&end_cycle);
	sscanf(argv[4],"%d",&procID);
	
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
//    int NsTestPart;
//    dataset_id = H5Dopen2(file_id, "/collective/NsTestPart", H5P_DEFAULT); // HDF 1.8.8
//    status = H5Dread(dataset_id, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT,&NsTestPart);
//    status = H5Dclose(dataset_id);

    int pcl[ns];
    int pclx, pcly, pclz,maxPcl=0;
    stringstream ss;

    for(int si=0;si<ns;si++){
    	ss.str("");
    	ss << "/collective/species_" << (si) << "/Npcelx";
    	dataset_id = H5Dopen2(file_id, ss.str().c_str(), H5P_DEFAULT); // HDF 1.8.8
    	status = H5Dread(dataset_id, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT,&pclx);//cout << "pclx=" << pclx<<endl;
    	status = H5Dclose(dataset_id);

    	ss.str("");
    	ss << "/collective/species_" << (si) << "/Npcely";
    	dataset_id = H5Dopen2(file_id, ss.str().c_str(), H5P_DEFAULT); // HDF 1.8.8
    	status = H5Dread(dataset_id, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT,&pcly);//cout << "pcly=" << pcly<<endl;
    	status = H5Dclose(dataset_id);

    	ss.str("");
    	ss << "/collective/species_" << (si) << "/Npcelz";
    	dataset_id = H5Dopen2(file_id, ss.str().c_str(), H5P_DEFAULT); // HDF 1.8.8
    	status = H5Dread(dataset_id, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT,&pclz);//cout << "pclz=" << pclz<<endl;
    	status = H5Dclose(dataset_id);

    	if( (pclx*pcly*pclz*nxc*nyc*nzc)>maxPcl) maxPcl=pclx*pcly*pclz*nxc*nyc*nzc;
    }
    //cout << "maxPcl=" << maxPcl <<endl;

	// at this point you can close settings
	status = H5Fclose(file_id);
	cout << " Successful in reading settings.hdf" << endl;

	int nop;
	int buffersize=maxPcl;
	hid_t datatype, dataspace;
	hsize_t  dims_out[1];
	double *X= new double[buffersize];
	double* Y = new double[buffersize];
	double* Z = new double[buffersize];
	double* U= new double[buffersize];
	double* V = new double[buffersize];
	double* W = new double[buffersize];
	double* Q= new double[buffersize];

	for (int i_cycle=start_cycle; i_cycle < (end_cycle+1); i_cycle+=step_cycle){
		
		cout << "****** CYCLE " << i_cycle << "******" <<  endl;
		stringstream cc;


		ss.str("");ss <<  "restart"  << procID << ".hdf";
		file_id = H5Fopen(ss.str().c_str(), H5F_ACC_RDWR, H5P_DEFAULT);
		if (file_id < 0){
			cout << "couldn't open file:  "<< ss.str() << endl;
			return -1;
		}

		for(int si=0;si<ns;si++){

			cc.str("");cc << "/particles/species_" << (si) << "/x/cycle_" << i_cycle;//cout << "reading data"<< cc.str() << endl;
			dataset_id = H5Dopen2(file_id,cc.str().c_str(), H5P_DEFAULT); // HDF 1.8.8
			datatype  = H5Dget_type(dataset_id);
			dataspace = H5Dget_space(dataset_id);
			status = H5Sget_simple_extent_dims(dataspace, dims_out, NULL);
			nop=dims_out[0]; cout << "nop =" << nop <<endl;

			if(nop>buffersize){
				delete[] X;
				delete[] Y;
				delete[] Z;
				delete[] U;
				delete[] V;
				delete[] W;
				delete[] Q;

				buffersize = 1.2*nop;
				X= new double[buffersize];
				Y= new double[buffersize];
				Z= new double[buffersize];
				U= new double[buffersize];
				V= new double[buffersize];
				W= new double[buffersize];
				Q= new double[buffersize];

				cout << "Buffer resized to "<< buffersize << endl;
			}

			status = H5Dread(dataset_id, datatype, H5S_ALL, H5S_ALL, H5P_DEFAULT, X);
			status = H5Dclose(dataset_id);

			cc.str("");cc << "/particles/species_" << (si) << "/y/cycle_" << i_cycle;
			dataset_id = H5Dopen2(file_id,cc.str().c_str(), H5P_DEFAULT);
			status = H5Dread(dataset_id, datatype, H5S_ALL, H5S_ALL, H5P_DEFAULT, Y);
			status = H5Dclose(dataset_id);

			cc.str("");cc << "/particles/species_" << (si) << "/z/cycle_" << i_cycle;
			dataset_id = H5Dopen2(file_id,cc.str().c_str(), H5P_DEFAULT);
			status = H5Dread(dataset_id, datatype, H5S_ALL, H5S_ALL, H5P_DEFAULT, Z);
			status = H5Dclose(dataset_id);

			cc.str("");cc << "/particles/species_" << (si) << "/u/cycle_" << i_cycle;
			dataset_id = H5Dopen2(file_id,cc.str().c_str(), H5P_DEFAULT);
			status = H5Dread(dataset_id, datatype, H5S_ALL, H5S_ALL, H5P_DEFAULT, U);
			status = H5Dclose(dataset_id);

			cc.str("");cc << "/particles/species_" << (si) << "/v/cycle_" << i_cycle;
			dataset_id = H5Dopen2(file_id,cc.str().c_str(), H5P_DEFAULT);
			status = H5Dread(dataset_id, datatype, H5S_ALL, H5S_ALL, H5P_DEFAULT, V);
			status = H5Dclose(dataset_id);

			cc.str("");cc << "/particles/species_" << (si) << "/w/cycle_" << i_cycle;
			dataset_id = H5Dopen2(file_id,cc.str().c_str(), H5P_DEFAULT);
			status = H5Dread(dataset_id, datatype, H5S_ALL, H5S_ALL, H5P_DEFAULT, W);
			status = H5Dclose(dataset_id);


			cc.str("");cc << "/particles/species_" << (si) << "/q/cycle_" << i_cycle;
			dataset_id = H5Dopen2(file_id,cc.str().c_str(), H5P_DEFAULT);
			status = H5Dread(dataset_id, datatype, H5S_ALL, H5S_ALL, H5P_DEFAULT, Q);
			status = H5Dclose(dataset_id);

			cc.str("");cc << "species" << (si) << "_cycle" << i_cycle << "_proc"<< procID << ".vtk";cout << "Start writing " << cc.str() << endl;
			ofstream my_file(cc.str().c_str());

//			my_file <<  "<?xml version=\"1.0\"?>\n"
//							"<VTKFile type=\"UnstructuredGrid\" version=\"0.1\" byte_order=\"LittleEndian\">\n"
//						    "  <UnstructuredGrid>\n"
//							"    <Piece NumberOfPoints=\""<< nop << "\" NumberOfCells=\"1\">\n"
//							"		<Cells>\n"
//							"			<DataArray type=\"UInt8\" Name=\"connectivity\" format=\"ascii\">0 1</DataArray>\n"
//							"			<DataArray type=\"UInt8\" Name=\"offsets\" 		format=\"ascii\">1</DataArray>\n"
//							"			<DataArray type=\"UInt8\" Name=\"types\"    	format=\"ascii\">1</DataArray>\n"
//							"		</Cells>\n"
//							"		<Points>\n"
//							"        	<DataArray type=\"Float32\" NumberOfComponents=\"3\" format=\"acscii\">\n";

			my_file << "# vtk DataFile Version 3.1 " << endl;
			my_file << "Particle Speces "<< si<< " from iPIC3D" << endl;
			my_file << "ASCII" << endl;
			my_file << "DATASET UNSTRUCTURED_GRID" << endl;
			my_file << "POINTS " << nop << " float " << endl;


			for(int i=0;i<nop;i++){
				my_file << X[i] <<"     "<< Y[i]<<"      "<< Z[i] <<"\n";
			}

			my_file <<   "POINT_DATA " << nop << "\n"
			  	  	<<   "SCALARS q float "	  << "\n"
			  	    <<   "LOOKUP_TABLE default\n"<< "\n";

			for(int i=0;i<nop;i++){
				my_file << Q[i] <<"\n";
			}

			my_file <<   "VECTORS Velocity float \n";

			for(int i=0;i<nop;i++){
				my_file << U[i] <<"     "<< V[i]<<"      "<< W[i] <<"\n";
			}

			my_file.close();

		}
		status = H5Fclose(file_id);

	}

		

	
		delete[] X;
		delete[] Y;
		delete[] Z;
		delete[] U;
		delete[] V;
		delete[] W;
		delete[] Q;
	
	return(0);
}




























