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

//Extract Particle Distribution Function

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

	int si;
	double upstreamXmin = 42.0;
	double upstreamXmax = 42.5;
	double shockXmin = 44.0;
	double shockXmax = 44.5;
	double downstreamXmin = 45.5;
	double downstreamXmax = 46.0;

	sscanf(argv[1],"%d",&si);

	if(argc > 2){
	sscanf(argv[2],"%lf",&upstreamXmin);
	sscanf(argv[3],"%lf",&upstreamXmax);
	sscanf(argv[4],"%lf",&shockXmin);
	sscanf(argv[5],"%lf",&shockXmax);
	sscanf(argv[6],"%lf",&downstreamXmin);
	sscanf(argv[7],"%lf",&downstreamXmax);
	}

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

	status = H5Fclose(file_id);

	int nop;
	int buffersize=maxPcl*2.0;
	hid_t datatype, dataspace;
	hsize_t  dims_out[1];
	double *X= new double[buffersize];
	double* Y = new double[buffersize];
	double* Z = new double[buffersize];
	double* U= new double[buffersize];
	double* V = new double[buffersize];
	double* W = new double[buffersize];
	double* Q= new double[buffersize];

	int procID[] = {671, 672, 735, 736, 799, 800, 863, 864};
	const int len = sizeof(procID)/sizeof(int);cout << "process number = " << len <<endl;
	int i_cycle  = 120000;
	stringstream cc;

	cc.str("");cc << "upstream_" << upstreamXmin << "_" << upstreamXmax << "species" << si << "_cyc"<< i_cycle << ".vtk";
	ofstream upstreamfile(cc.str().c_str());

	cc.str("");cc << "shock_" << shockXmin << "_" << shockXmax << "species" << si << "_cyc"<< i_cycle << ".vtk";
	ofstream shockfile(cc.str().c_str());

	cc.str("");cc << "downstream_" << downstreamXmin << "_" << downstreamXmax << "species" << si << "_cyc"<< i_cycle << ".vtk";
	ofstream downstreamfile(cc.str().c_str());

	/*
	upstreamfile << "# vtk DataFile Version 3.1 " << endl;
	upstreamfile << "Particle Speces "<< si<< " from iPIC3D" << endl;
	upstreamfile << "ASCII" << endl;
	upstreamfile << "DATASET UNSTRUCTURED_GRID" << endl;
	upstreamfile << "POINTS " << nop << " float " << endl;

	shockfile << "# vtk DataFile Version 3.1 " << endl;
	shockfile << "Particle Speces "<< si<< " from iPIC3D" << endl;
	shockfile << "ASCII" << endl;
	shockfile << "DATASET UNSTRUCTURED_GRID" << endl;
	shockfile << "POINTS " << nop << " float " << endl;

	downstreamfile << "# vtk DataFile Version 3.1 " << endl;
	downstreamfile << "Particle Speces "<< si<< " from iPIC3D" << endl;
	downstreamfile << "ASCII" << endl;
	downstreamfile << "DATASET UNSTRUCTURED_GRID" << endl;
	downstreamfile << "POINTS " << nop << " float " << endl;

	upstreamfile << "Px Py Pz Vx Vy Vz Q" << endl;
	shockfile << "Px Py Pz Vx Vy Vz Q" << endl;
	downstreamfile << "Px Py Pz Vx Vy Vz Q" << endl;
	*/

	int upstreamnop = 0;
	int shocknop = 0;
	int downstreamnop = 0;
	double upstreamVxmin=1.0;
	double upstreamVxmax=-1.0;
	double upstreamVymin=1.0;
	double upstreamVymax=-1.0;
	double upstreamVzmin=1.0;
	double upstreamVzmax=-1.0;

	double shockVxmin=1.0;
	double shockVxmax=-1.0;
	double shockVymin=1.0;
	double shockVymax=-1.0;
	double shockVzmin=1.0;
	double shockVzmax=-1.0;

	double downstreamVxmin=1.0;
	double downstreamVxmax=-1.0;
	double downstreamVymin=1.0;
	double downstreamVymax=-1.0;
	double downstreamVzmin=1.0;
	double downstreamVzmax=-1.0;


for(int procid = 0;procid<len;procid++){
	ss.str("");ss <<  "restart"  << procID[procid] << ".hdf";
	file_id = H5Fopen(ss.str().c_str(), H5F_ACC_RDWR, H5P_DEFAULT);
		if (file_id < 0){
			cout << "couldn't open file:  "<< ss.str() << endl;
			return -1;
		}
	cout << "reading file "<< ss.str() << endl;


	cc.str("");cc << "/particles/species_" << (si) << "/x/cycle_" << i_cycle;
	dataset_id = H5Dopen2(file_id,cc.str().c_str(), H5P_DEFAULT); // HDF 1.8.8
	datatype  = H5Dget_type(dataset_id);
	dataspace = H5Dget_space(dataset_id);
	status = H5Sget_simple_extent_dims(dataspace, dims_out, NULL);
	nop=dims_out[0]; cout << "species_ "<< si << " nop =" << nop <<endl;

	if(nop>buffersize){
		delete[] X;
		delete[] Y;
		delete[] Z;
		delete[] U;
		delete[] V;
		delete[] W;
		delete[] Q;

		buffersize = 2.0*nop;
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

	status = H5Fclose(file_id);

	for(int i=0;i<nop;i++){
		if(X[i]>=upstreamXmin && X[i]<=upstreamXmax){
			upstreamfile << X[i] <<" "<< Y[i]<<" "<< Z[i]<<" " << U[i]<<" "<< V[i]<<" "<< W[i]<<" "<< Q[i] <<"\n";
			//upstreamfile << X[i] <<" "<< Y[i]<<" "<< Z[i]<<" " <<W[i] << "\n";//
			upstreamnop ++;
			if(U[i] < upstreamVxmin){
				upstreamVxmin = U[i];
			}
			if(U[i] > upstreamVxmax){
				upstreamVxmax = U[i];
			}

			if(V[i] < upstreamVymin){
				upstreamVymin = V[i];
			}
			if(V[i] > upstreamVymax){
				upstreamVymax = V[i];
			}

			if(W[i] < upstreamVzmin){
				upstreamVzmin = W[i];
			}
			if(W[i] > upstreamVzmax){
				upstreamVzmax = W[i];
			}
		}else if(X[i]>=shockXmin && X[i]<=shockXmax){
			shockfile << X[i] <<" "<< Y[i]<<" "<< Z[i]<<" " << U[i]<<" "<< V[i]<<" "<< W[i]<<" "<< Q[i] <<"\n";
			//shockfile << X[i] <<" "<< Y[i]<<" "<< Z[i]<<" " <<  W[i] <<"\n";
			shocknop ++;
			if(U[i] < shockVxmin){
				shockVxmin = U[i];
			}
			if(U[i] > shockVxmax){
				shockVxmax = U[i];
			}


			if(V[i] < shockVymin){
				shockVymin = V[i];
			}
			if(V[i] > shockVymax){
				shockVymax = V[i];
			}


			if(W[i] < shockVzmin){
				shockVzmin = W[i];
			}
			if(W[i] > shockVzmax){
				shockVzmax = W[i];
			}
		}else if(X[i]>=downstreamXmin && X[i]<=downstreamXmax){
			downstreamfile << X[i] <<" "<< Y[i]<<" "<< Z[i]<<" " << U[i]<<" "<< V[i]<<" "<< W[i]<<" "<< Q[i] <<"\n";
			//downstreamfile << X[i] <<" "<< Y[i]<<" "<< Z[i]<<" "<< W[i] <<"\n";
			downstreamnop ++;
			if(U[i] < downstreamVxmin){
				downstreamVxmin = U[i];
			}
			if(U[i] > downstreamVxmax){
				downstreamVxmax = U[i];
			}

			if(V[i] < downstreamVymin){
				downstreamVymin = V[i];
			}
			if(V[i] > downstreamVymax){
				downstreamVymax = V[i];
			}

			if(W[i] < downstreamVzmin){
				downstreamVzmin = W[i];
			}
			if(W[i] > downstreamVzmax){
				downstreamVzmax = W[i];
			}
		}

	}

}
	upstreamfile.close();
	shockfile.close();
	downstreamfile.close();
	
	cout << "upstreamnop =  "<< upstreamnop << ", upstreamVxmin =  "<< upstreamVxmin << ", upstreamVxmax =  "<< upstreamVxmax <<
			", upstreamVymin =  "<< upstreamVymin << ", upstreamVymax =  " << upstreamVymax <<
			", upstreamVzmin =  "<< upstreamVzmin << ", upstreamVzmax =  " << upstreamVzmax << endl;

	cout << "shocknop =  "<< shocknop << ", shockVxmin =  "<< shockVxmin << ", shockVxmax =  "<< shockVxmax <<
			", shockVymin =  "<< shockVymin << ", shockVymax =  "<< shockVymax <<
			", shockVzmin =  "<< shockVzmin << ", shockVzmax =  "<< shockVzmax << endl;

	cout << "downstreamnop =  "<< downstreamnop << ", downstreamVxmin =  "<< downstreamVxmin << ", downstreamVxmax =  "<< downstreamVxmax <<
			", downstreamVymin =  "<< downstreamVymin << ", downstreamVymax =  "<< downstreamVymax<<
			", downstreamVzmin =  "<< downstreamVzmin << ", downstreamVzmax =  "<< downstreamVzmax<< endl;

		delete[] X;
		delete[] Y;
		delete[] Z;
		delete[] U;
		delete[] V;
		delete[] W;
		delete[] Q;
	
	return(0);
}




























