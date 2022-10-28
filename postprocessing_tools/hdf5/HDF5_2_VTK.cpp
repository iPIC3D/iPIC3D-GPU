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

/***************************************************************************
 convHDF5.cpp  -  Convert program to open iPIC3D Output
 -------------------
 begin                : Jun 2008
 copyright            : (C) 2004 by Stefano Markidis, Giovanni Lapenta
 ************************************************************************** */

#include "hdf5.h"
#include "../../include/Alloc.h"
#include "math.h"


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
		cout << "ie, HDF5_2_VTK 0 100 1000" << endl;
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
	int XLEN;
	dataset_id = H5Dopen2(file_id, "/topology/XLEN", H5P_DEFAULT); // HDF 1.8.8
	status = H5Dread(dataset_id, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT,&XLEN);
	status = H5Dclose(dataset_id);
	int YLEN;
	dataset_id = H5Dopen2(file_id, "/topology/YLEN", H5P_DEFAULT); // HDF 1.8.8
	status = H5Dread(dataset_id, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT,&YLEN);
	status = H5Dclose(dataset_id);
	int ZLEN;
	dataset_id = H5Dopen2(file_id, "/topology/ZLEN", H5P_DEFAULT); // HDF 1.8.8
	status = H5Dread(dataset_id, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT,&ZLEN);
	status = H5Dclose(dataset_id);
	
	// read Lx	
	double Lx;
	dataset_id = H5Dopen2(file_id, "/collective/Lx", H5P_DEFAULT); // HDF 1.8.8
	status = H5Dread(dataset_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT,&Lx);
	status = H5Dclose(dataset_id);
	// read Ly
	double Ly;	
	dataset_id = H5Dopen2(file_id, "/collective/Ly", H5P_DEFAULT); // HDF 1.8.8
	status = H5Dread(dataset_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT,&Ly);
	status = H5Dclose(dataset_id);
	// read Lz
	double Lz;	
	dataset_id = H5Dopen2(file_id, "/collective/Lz", H5P_DEFAULT); // HDF 1.8.8
	status = H5Dread(dataset_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT,&Lz);
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
	// at this point you can close settings
	status = H5Dclose(file_id);
	cout << " Successful in reading settings.hdf" << endl;
	// prepare to read the proc files
	hid_t    *proc_file_id = new hid_t[nproc];
	string temp;
	int* cartesian_cor= new int[3];
	
    // this should be changed if higher than 30 procs per dimension
    int mappa[30][30][30];
	for (int i=0; i < nproc; i++){
		stringstream ss;
		ss << i;
		temp = "proc" + ss.str() + ".hdf";
		//temp = "proc" << ss.c_str() <<".hdf";
		proc_file_id[i] = H5Fopen(temp.c_str(), H5F_ACC_RDWR, H5P_DEFAULT);
		if (proc_file_id[i] < 0){
			cout << "couldn't open file:  "<< temp << endl;
			return -1;
		}
		// read the position in the topology
		dataset_id = H5Dopen2(proc_file_id[i], "/topology/cartesian_coord", H5P_DEFAULT); // HDF 1.8.8
		status = H5Dread(dataset_id, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT,cartesian_cor);
		mappa[cartesian_cor[0]][cartesian_cor[1]][cartesian_cor[2]] = i;
		cout << "file" << i << " in topology[" << cartesian_cor[0] << "][" << cartesian_cor[1] << "][" << cartesian_cor[2]  <<"]" << endl;
		status = H5Dclose(dataset_id);
		H5Fclose(proc_file_id[i]);
		
	} 
	cout << " Successful in opening and closing all the proc*.hdf the first time" << endl;
	// grid points
	int nxn = nxc/XLEN;
	int nyn = nyc/YLEN;
	int nzn = nzc/ZLEN;
	double dx = Lx/(nxc-1);
    if (nxc==1)
        dx = Lx;
	double dy = Ly/(nyc-1);
    if (nyc==1)
        dy = Ly;
	double dz = Lz/(nzc-1);
    if (nzc==1)
        dz = Lz;
	// auxiliary variables
	double *temp_storageX = new double[(nxn+1)*(nyn+1)*(nzn+1)];
	double *temp_storageY = new double[(nxn+1)*(nyn+1)*(nzn+1)];
	double *temp_storageZ = new double[(nxn+1)*(nyn+1)*(nzn+1)];
	double*** BX= newArr3(double,nxn*XLEN,nyn*YLEN,nzn*ZLEN);
	double*** BY = newArr3(double,nxn*XLEN,nyn*YLEN,nzn*ZLEN);
	double*** BZ = newArr3(double,nxn*XLEN,nyn*YLEN,nzn*ZLEN);
	double*** EX= newArr3(double,nxn*XLEN,nyn*YLEN,nzn*ZLEN);
	double*** EY = newArr3(double,nxn*XLEN,nyn*YLEN,nzn*ZLEN);
	double*** EZ = newArr3(double,nxn*XLEN,nyn*YLEN,nzn*ZLEN);
	double*** JX= newArr3(double,nxn*XLEN,nyn*YLEN,nzn*ZLEN);
	double*** JY = newArr3(double,nxn*XLEN,nyn*YLEN,nzn*ZLEN);
	double*** JZ = newArr3(double,nxn*XLEN,nyn*YLEN,nzn*ZLEN);
	double*** rho = newArr3(double,nxn*XLEN,nyn*YLEN,nzn*ZLEN);
	double Bmod;
	double Epar;
	double Eper;
	double Eperx;
	double Epery;
	double Eperz;
	int node;
	// each different cycle write a .vtk file
	for (int i_cycle=start_cycle; i_cycle < (end_cycle+1); i_cycle+=step_cycle){
		
		cout << "****** CYCLE " << i_cycle << "******" <<  endl;
		stringstream cc;
		cc << i_cycle;
		// prepare the file
		//int nxn = nxc/XLEN + 1;
		//int nyn = nyc/YLEN + 1;
		//int nzn = nzc/ZLEN + 1;
		
		// B
		temp = "B_cycle"+ cc.str();
		temp += ".vtk";
		ofstream my_file(temp.c_str());
		// E
		temp = "E_cycle"+ cc.str();
		temp += ".vtk";
		ofstream my_fileE(temp.c_str());
		// Epar
		temp = "Epar_cycle"+ cc.str();
		temp += ".vtk";
		ofstream my_fileEpar(temp.c_str());
		// Eper
		temp = "Eper_cycle"+ cc.str();
		temp += ".vtk";
		ofstream my_fileEper(temp.c_str());
		// Je
		temp = "Je_cycle"+ cc.str();
		temp += ".vtk";
		ofstream my_fileJe(temp.c_str());
		// Ji
		temp = "Ji_cycle"+ cc.str();
		temp += ".vtk";
		ofstream my_fileJi(temp.c_str());
		// rhoe
		temp = "rhoe_cycle"+ cc.str();
		temp += ".vtk";
		ofstream my_filerhoe(temp.c_str());
		// rhoi
		temp = "rhoi_cycle"+ cc.str();
		temp += ".vtk";
		ofstream my_filerhoi(temp.c_str());
		// Ve
		temp = "Ve_cycle"+ cc.str();
		temp += ".vtk";
		ofstream my_fileVe(temp.c_str());
		// Vi
		temp = "Vi_cycle"+ cc.str();
		temp += ".vtk";
		ofstream my_fileVi(temp.c_str());
		
		
		// B
		my_file << "# vtk DataFile Version 1.0" << endl;
		my_file << "Magnetic Field from iPIC3D" << endl;
		my_file << "ASCII" << endl;
		my_file << "DATASET STRUCTURED_POINTS" << endl;
		my_file << "DIMENSIONS " << nxn*XLEN << " " << nyn*YLEN << " " << nzn*ZLEN << endl;
		my_file << "ORIGIN 0 0 0" << endl;
		my_file << "SPACING " << dx << " " << dy << " " << dz << endl;
		
		my_file << "POINT_DATA " << nxn*nyn*nzn*nproc << endl;
		my_file << "VECTORS B float" << endl;
	    cout << "1/8 done with B mesh" << endl;
		
		// E
		my_fileE << "# vtk DataFile Version 1.0" << endl;
		my_fileE << "Electric Field from iPIC3D" << endl;
		my_fileE << "ASCII" << endl;
		my_fileE << "DATASET STRUCTURED_POINTS" << endl;
		my_fileE << "DIMENSIONS " << nxn*XLEN << " " << nyn*YLEN << " " << nzn*ZLEN << endl;
		my_fileE << "ORIGIN 0 0 0" << endl;
		my_fileE << "SPACING " << dx << " " << dy << " " << dz << endl;
		
		my_fileE << "POINT_DATA " << nxn*nyn*nzn*nproc << endl;
		my_fileE << "VECTORS E float" << endl;
		
		// Epar
		my_fileEpar << "# vtk DataFile Version 1.0" << endl;
		my_fileEpar << "Parallel Electric Field from iPIC3D" << endl;
		my_fileEpar << "ASCII" << endl;
		my_fileEpar << "DATASET STRUCTURED_POINTS" << endl;
		my_fileEpar << "DIMENSIONS " << nxn*XLEN << " " << nyn*YLEN << " " << nzn*ZLEN << endl;
		my_fileEpar << "ORIGIN 0 0 0" << endl;
		my_fileEpar << "SPACING " << dx << " " << dy << " " << dz << endl;
		
		my_fileEpar << "POINT_DATA " << nxn*nyn*nzn*nproc << endl;
		my_fileEpar << "SCALARS Epar float" << endl;
		my_fileEpar << "LOOKUP_TABLE default" << endl;
		// Eper
		my_fileEper << "# vtk DataFile Version 1.0" << endl;
		my_fileEper << "Parallel Electric Field from iPIC3D" << endl;
		my_fileEper << "ASCII" << endl;
		my_fileEper << "DATASET STRUCTURED_POINTS" << endl;
		my_fileEper << "DIMENSIONS " << nxn*XLEN << " " << nyn*YLEN << " " << nzn*ZLEN << endl;
		my_fileEper << "ORIGIN 0 0 0" << endl;
		my_fileEper << "SPACING " << dx << " " << dy << " " << dz << endl;
		
		my_fileEper << "POINT_DATA " << nxn*nyn*nzn*nproc << endl;
		my_fileEper << "SCALARS Eper float" << endl;
		my_fileEper << "LOOKUP_TABLE default" << endl;
		cout << "2/8 done with E, Epar, Eperp meshes" << endl;
		// Je
		my_fileJe << "# vtk DataFile Version 1.0" << endl;
		my_fileJe << "Electron current from iPIC3D" << endl;
		my_fileJe << "ASCII" << endl;
		my_fileJe << "DATASET STRUCTURED_POINTS" << endl;
		my_fileJe << "DIMENSIONS " << nxn*XLEN << " " << nyn*YLEN << " " << nzn*ZLEN << endl;
		my_fileJe << "ORIGIN 0 0 0" << endl;
		my_fileJe << "SPACING " << dx << " " << dy << " " << dz << endl;
		
		my_fileJe << "POINT_DATA " << nxn*nyn*nzn*nproc << endl;
		my_fileJe << "VECTORS Je float" << endl;
		cout << "3/8 done with Je mesh" << endl;
		
		// Ji
		
		my_fileJi << "# vtk DataFile Version 1.0" << endl;
		my_fileJi << "Ion current from iPIC3D" << endl;
		my_fileJi << "ASCII" << endl;
		my_fileJi << "DATASET STRUCTURED_POINTS" << endl;
		my_fileJi << "DIMENSIONS " << nxn*XLEN << " " << nyn*YLEN << " " << nzn*ZLEN << endl;
		my_fileJi << "ORIGIN 0 0 0" << endl;
		my_fileJi << "SPACING " << dx << " " << dy << " " << dz << endl;
		
		my_fileJi << "POINT_DATA " << nxn*nyn*nzn*nproc << endl;
		my_fileJi << "VECTORS Ji float" << endl;
		cout << "4/8 done with Ji mesh" << endl;
		
		// rhoe
		my_filerhoe << "# vtk DataFile Version 1.0" << endl;
		my_filerhoe << "electron density from iPIC3D" << endl;
		my_filerhoe << "ASCII" << endl;
		my_filerhoe << "DATASET STRUCTURED_POINTS" << endl;
		my_filerhoe << "DIMENSIONS " << nxn*XLEN << " " << nyn*YLEN << " " << nzn*ZLEN << endl;
		my_filerhoe << "ORIGIN 0 0 0" << endl;
		my_filerhoe << "SPACING " << dx << " " << dy << " " << dz << endl;
		
		my_filerhoe << "POINT_DATA " << nxn*nyn*nzn*nproc << endl;
		my_filerhoe << "SCALARS rhoe float" << endl;
		my_filerhoe << "LOOKUP_TABLE default" << endl;
		cout << "5/8 done with rhoe mesh" << endl;
		// end rhoe
		
		// rhoi
		my_filerhoi << "# vtk DataFile Version 1.0" << endl;
		my_filerhoi << "ion density from iPIC3D" << endl;
		my_filerhoi << "ASCII" << endl;
		my_filerhoi << "DATASET STRUCTURED_POINTS" << endl;
		my_filerhoi << "DIMENSIONS " << nxn*XLEN << " " << nyn*YLEN << " " << nzn*ZLEN << endl;
		my_filerhoi << "ORIGIN 0 0 0" << endl;
		my_filerhoi << "SPACING " << dx << " " << dy << " " << dz << endl;
		
		my_filerhoi << "POINT_DATA " << nxn*nyn*nzn*nproc << endl;
		my_filerhoi << "SCALARS rhoi float" << endl;
		my_filerhoi << "LOOKUP_TABLE default" << endl;
		// end rhoi
		cout << "6/8 done with rhoi mesh" << endl;
		
		
		// Ve
		my_fileVe << "# vtk DataFile Version 1.0" << endl;
		my_fileVe << "Electron average velocity from iPIC3D" << endl;
		my_fileVe << "ASCII" << endl;
		my_fileVe << "DATASET STRUCTURED_POINTS" << endl;
		my_fileVe << "DIMENSIONS " << nxn*XLEN << " " << nyn*YLEN << " " << nzn*ZLEN << endl;
		my_fileVe << "ORIGIN 0 0 0" << endl;
		my_fileVe << "SPACING " << dx << " " << dy << " " << dz << endl;
		
		my_fileVe << "POINT_DATA " << nxn*nyn*nzn*nproc << endl;
		my_fileVe << "VECTORS Ve float" << endl;
		cout << "7/8 done with Ve mesh" << endl;
		
		// Vi
		my_fileVi << "# vtk DataFile Version 1.0" << endl;
		my_fileVi << "Ion average velocity from iPIC3D" << endl;
		my_fileVi << "ASCII" << endl;
		my_fileVi << "DATASET STRUCTURED_POINTS" << endl;
		my_fileVi << "DIMENSIONS " << nxn*XLEN << " " << nyn*YLEN << " " << nzn*ZLEN << endl;
		my_fileVi << "ORIGIN 0 0 0" << endl;
		my_fileVi << "SPACING " << dx << " " << dy << " " << dz << endl;
		
		my_fileVi << "POINT_DATA " << nxn*nyn*nzn*nproc << endl;
		my_fileVi << "VECTORS Vi float" << endl;
		cout << "8/8 done with Vi mesh" << endl;
		cout << endl;
		
		
		
		
		
		cout << "1/8 Reading and writing B" << endl;
		int proc = 0;
		for (int i=0; i < XLEN;i++)
			for (int j=0; j < YLEN;j++)
				for (int k=0; k < ZLEN;k++){
					stringstream ss;
					ss << mappa[i][j][k];
					temp = "proc" + ss.str() + ".hdf";
					//cout << "opening file ->" << temp << " mappa: " << mappa[i][j][k] <<  endl;
					proc_file_id[mappa[i][j][k]] = H5Fopen(temp.c_str(), H5F_ACC_RDWR, H5P_DEFAULT); 
					if (proc_file_id[mappa[i][j][k]] < 0){
						cout << "couldn't open file:  "<< temp << endl;
						return -1;
					}
					// read data  
					temp = "/fields/Bx/cycle_"+ cc.str(); 
					dataset_id = H5Dopen2(proc_file_id[mappa[i][j][k]],temp.c_str(), H5P_DEFAULT); // HDF 1.8.8
					status = H5Dread(dataset_id, H5T_NATIVE_DOUBLE, H5S_ALL,H5S_ALL,H5P_DEFAULT,temp_storageX); status = H5Dclose(dataset_id);
					temp = "/fields/By/cycle_"+ cc.str(); 
					dataset_id = H5Dopen2(proc_file_id[mappa[i][j][k]],temp.c_str(), H5P_DEFAULT); // HDF 1.8.8
					status = H5Dread(dataset_id, H5T_NATIVE_DOUBLE, H5S_ALL,H5S_ALL,H5P_DEFAULT,temp_storageY); status = H5Dclose(dataset_id);
					temp = "/fields/Bz/cycle_"+ cc.str(); 
					dataset_id = H5Dopen2(proc_file_id[mappa[i][j][k]],temp.c_str(), H5P_DEFAULT); // HDF 1.8.8
					status = H5Dread(dataset_id, H5T_NATIVE_DOUBLE, H5S_ALL,H5S_ALL,H5P_DEFAULT,temp_storageZ); status = H5Dclose(dataset_id);
					node=0;
					for (int ii=0; ii < (nxn+1);ii++)
						for (int jj=0; jj < (nyn+1);jj++)
							for (int kk=0; kk < (nzn+1);kk++){
								if (ii!= nxn && jj!= nyn && kk!=nzn){  
									BX[ii + nxn*i][jj + nyn*j][kk + nzn*k] = temp_storageX[node];
									BY[ii + nxn*i][jj + nyn*j][kk + nzn*k] = temp_storageY[node];
									BZ[ii + nxn*i][jj + nyn*j][kk + nzn*k] = temp_storageZ[node];
								}
								node++;
							}
					// close the file
					status = H5Fclose(proc_file_id[mappa[i][j][k]]);
					proc++;
				}
		
		// write to disc
		for (int kk=0; kk < nzn*ZLEN;kk++)
			for (int jj=0; jj < nyn*YLEN;jj++)
				for (int ii=0; ii < nxn*XLEN;ii++){
                    if (fabs(BX[ii][jj][kk]) < 1E-16)
                        BX[ii][jj][kk] = 0.0;
                    if (fabs(BY[ii][jj][kk]) < 1E-16)
                        BY[ii][jj][kk] = 0.0;
                    if (fabs(BZ[ii][jj][kk]) < 1E-16)
                        BZ[ii][jj][kk] = 0.0;
					my_file << BX[ii][jj][kk] << " " << BY[ii][jj][kk] << " " << BZ[ii][jj][kk] << endl;
					// my_file << temp_storageX[node] << " " << temp_storageY[node] << " " << temp_storageZ[node] << endl;
				}
		my_file.close();
		
		// write Electric field
		cout << "2/8 Reading and writing E, Epar, Eperp" << endl;
		proc = 0;
		for (int i=0; i < XLEN;i++)
			for (int j=0; j < YLEN;j++)
				for (int k=0; k < ZLEN;k++){
					stringstream ss;
					ss << mappa[i][j][k];
					temp = "proc" + ss.str() + ".hdf";
					proc_file_id[mappa[i][j][k]] = H5Fopen(temp.c_str(), H5F_ACC_RDWR, H5P_DEFAULT); 
					// read data
					
					temp = "/fields/Ex/cycle_"+ cc.str(); 
					dataset_id = H5Dopen2(proc_file_id[mappa[i][j][k]],temp.c_str(), H5P_DEFAULT);  // HDF 1.8.8
					status = H5Dread(dataset_id, H5T_NATIVE_DOUBLE, H5S_ALL,H5S_ALL,H5P_DEFAULT,temp_storageX); status = H5Dclose(dataset_id);
					temp = "/fields/Ey/cycle_"+ cc.str(); 
					dataset_id = H5Dopen2(proc_file_id[mappa[i][j][k]],temp.c_str(), H5P_DEFAULT);  // HDF 1.8.8
					status = H5Dread(dataset_id, H5T_NATIVE_DOUBLE, H5S_ALL,H5S_ALL,H5P_DEFAULT,temp_storageY); status = H5Dclose(dataset_id);
					temp = "/fields/Ez/cycle_"+ cc.str(); 
					dataset_id = H5Dopen2(proc_file_id[mappa[i][j][k]],temp.c_str(), H5P_DEFAULT);  // HDF 1.8.8
					status = H5Dread(dataset_id, H5T_NATIVE_DOUBLE, H5S_ALL,H5S_ALL,H5P_DEFAULT,temp_storageZ); status = H5Dclose(dataset_id);
					node=0;
					for (int ii=0; ii < (nxn+1);ii++)
						for (int jj=0; jj < (nyn+1);jj++)
							for (int kk=0; kk < (nzn+1);kk++){
								if (ii!= nxn && jj!= nyn && kk!=nzn){  
									EX[ii + nxn*i][jj + nyn*j][kk + nzn*k] = temp_storageX[node];
									EY[ii + nxn*i][jj + nyn*j][kk + nzn*k] = temp_storageY[node];
									EZ[ii + nxn*i][jj + nyn*j][kk + nzn*k] = temp_storageZ[node];
								}
								node++;
							}
					// close the file
					H5Fclose(proc_file_id[mappa[i][j][k]]);
					// go to other proc
					proc++;
				}
		for (int kk=0; kk < nzn*ZLEN;kk++)
			for (int jj=0; jj < nyn*YLEN;jj++)
				for (int ii=0; ii < nxn*XLEN;ii++){
                    if (fabs(EX[ii][jj][kk]) < 1E-16)
                        EX[ii][jj][kk] = 0.0;
                    if (fabs(EY[ii][jj][kk]) < 1E-16)
                        EY[ii][jj][kk] = 0.0;
                    if (fabs(EZ[ii][jj][kk]) < 1E-16)
                        EZ[ii][jj][kk] = 0.0;
					my_fileE << EX[ii][jj][kk] << " " << EY[ii][jj][kk] << " " << EZ[ii][jj][kk] << endl;
					// my_file << temp_storageX[node] << " " << temp_storageY[node] << " " << temp_storageZ[node] << endl;
				}
		my_fileE.close();
		for (int kk=0; kk < nzn*ZLEN;kk++)
			for (int jj=0; jj < nyn*YLEN;jj++)
				for (int ii=0; ii < nxn*XLEN;ii++){
					Bmod = sqrt(BX[ii][jj][kk] * BX[ii][jj][kk] + BY[ii][jj][kk] * BY[ii][jj][kk] + BZ[ii][jj][kk] * BZ[ii][jj][kk]);
					Epar = (EX[ii][jj][kk] * BX[ii][jj][kk] + EY[ii][jj][kk] * BY[ii][jj][kk] + EZ[ii][jj][kk] * BZ[ii][jj][kk] ) / Bmod ;
					Eperx = EX[ii][jj][kk] -Epar* BX[ii][jj][kk] / Bmod ;
					Epery = EY[ii][jj][kk] -Epar* BY[ii][jj][kk] / Bmod ;
					Eperz = EZ[ii][jj][kk] -Epar* BZ[ii][jj][kk] / Bmod ;
					Eper = sqrt(Eperx * Eperx + Epery * Epery + Eperz * Eperz);
					my_fileEpar << Epar << endl;
					my_fileEper << Eper << endl;
					
				}
		my_fileEpar.close();
		my_fileEper.close();
		// Je
		// write electron current
		cout << "3/8 Reading and writing Je" << endl;
		proc = 0;
		for (int i=0; i < XLEN;i++)
			for (int j=0; j < YLEN;j++)
				for (int k=0; k < ZLEN;k++){
					stringstream ss;
					ss << mappa[i][j][k];
					temp = "proc" + ss.str() + ".hdf";
					proc_file_id[mappa[i][j][k]] = H5Fopen(temp.c_str(), H5F_ACC_RDWR, H5P_DEFAULT); 
					// read data
					// species 0: current sheet electrons
					temp = "/moments/species_0/Jx/cycle_"+ cc.str(); 
					dataset_id = H5Dopen2(proc_file_id[mappa[i][j][k]],temp.c_str(), H5P_DEFAULT);  // HDF 1.8.8
					status = H5Dread(dataset_id, H5T_NATIVE_DOUBLE, H5S_ALL,H5S_ALL,H5P_DEFAULT,temp_storageX); status = H5Dclose(dataset_id);
					temp = "/moments/species_0/Jy/cycle_"+ cc.str(); 
					dataset_id = H5Dopen2(proc_file_id[mappa[i][j][k]],temp.c_str(), H5P_DEFAULT);  // HDF 1.8.8
					status = H5Dread(dataset_id, H5T_NATIVE_DOUBLE, H5S_ALL,H5S_ALL,H5P_DEFAULT,temp_storageY); status = H5Dclose(dataset_id);
					temp = "/moments/species_0/Jz/cycle_"+ cc.str(); 
					dataset_id = H5Dopen2(proc_file_id[mappa[i][j][k]],temp.c_str(), H5P_DEFAULT);  // HDF 1.8.8
					status = H5Dread(dataset_id, H5T_NATIVE_DOUBLE, H5S_ALL,H5S_ALL,H5P_DEFAULT,temp_storageZ); status = H5Dclose(dataset_id);
					node=0;
					for (int ii=0; ii < (nxn+1);ii++)
						for (int jj=0; jj < (nyn+1);jj++)
							for (int kk=0; kk < (nzn+1);kk++){
								if (ii!= nxn && jj!= nyn && kk!=nzn){  
									JX[ii + nxn*i][jj + nyn*j][kk + nzn*k] = temp_storageX[node];
									JY[ii + nxn*i][jj + nyn*j][kk + nzn*k] = temp_storageY[node];
									JZ[ii + nxn*i][jj + nyn*j][kk + nzn*k] = temp_storageZ[node];
								}
								node++;
							}
									
					
					// close the file
					H5Fclose(proc_file_id[mappa[i][j][k]]);
					// go to other proc
					proc++;
				}
		for (int kk=0; kk < nzn*ZLEN;kk++)
			for (int jj=0; jj < nyn*YLEN;jj++)
				for (int ii=0; ii < nxn*XLEN;ii++){
                    if (fabs(JX[ii][jj][kk]) < 1E-16)
                        JX[ii][jj][kk] = 0.0;
                    if (fabs(JY[ii][jj][kk]) < 1E-16)
                        JY[ii][jj][kk] = 0.0;
                    if (fabs(JZ[ii][jj][kk]) < 1E-16)
                        JZ[ii][jj][kk] = 0.0;
					my_fileJe << JX[ii][jj][kk] << " " << JY[ii][jj][kk] << " " << JZ[ii][jj][kk] << endl;
				}
		my_fileJe.close();
		// end Je
		// rhoe
		// write charge density
		cout << "4/8 Reading and writing rhoe" << endl;
		proc = 0;
		for (int i=0; i < XLEN;i++)
			for (int j=0; j < YLEN;j++)
				for (int k=0; k < ZLEN;k++){
					stringstream ss;
					ss << mappa[i][j][k];
					temp = "proc" + ss.str() + ".hdf";
					proc_file_id[mappa[i][j][k]] = H5Fopen(temp.c_str(), H5F_ACC_RDWR, H5P_DEFAULT); 
					// read data
					temp = "/moments/species_0/rho/cycle_"+ cc.str(); 
					dataset_id = H5Dopen2(proc_file_id[mappa[i][j][k]],temp.c_str(), H5P_DEFAULT); // HDF 1.8.8
					status = H5Dread(dataset_id, H5T_NATIVE_DOUBLE, H5S_ALL,H5S_ALL,H5P_DEFAULT,temp_storageX); status = H5Dclose(dataset_id);
					node=0;
					for (int ii=0; ii < (nxn+1);ii++)
						for (int jj=0; jj < (nyn+1);jj++)
							for (int kk=0; kk < (nzn+1);kk++){
								if (ii!= nxn && jj!= nyn && kk!=nzn){  
									rho[ii + nxn*i][jj + nyn*j][kk + nzn*k] = temp_storageX[node];
								}
								node++;
							}
					
					
					
					// close the file
					H5Fclose(proc_file_id[mappa[i][j][k]]);
					// go to other proc
					proc++;
				}
		//cout << "WRITING VECTOR rhoe TO VTK FILE" << endl;
		for (int kk=0; kk < nzn*ZLEN;kk++)
			for (int jj=0; jj < nyn*YLEN;jj++)
				for (int ii=0; ii < nxn*XLEN;ii++){
					my_filerhoe << 4*3.1415*rho[ii][jj][kk]  << endl;
				}
		my_filerhoe.close();
		// end rhoe
        
		// start Ve
		cout << "5/8 writing Ve" << endl;
		for (int kk=0; kk < nzn*ZLEN;kk++)
			for (int jj=0; jj < nyn*YLEN;jj++)
				for (int ii=0; ii < nxn*XLEN;ii++){
					my_fileVe << (JX[ii][jj][kk]/rho[ii][jj][kk]) << " " << (JY[ii][jj][kk]/rho[ii][jj][kk]) << " " << (JZ[ii][jj][kk]/rho[ii][jj][kk]) << endl;
				}
		my_fileVe.close();
		// end Ve
		
		
		// Ji
		// write ion current
		cout << "6/8 Reading and writing Ji" << endl;
		proc = 0;
		for (int i=0; i < XLEN;i++)
			for (int j=0; j < YLEN;j++)
				for (int k=0; k < ZLEN;k++){
					stringstream ss;
					ss << mappa[i][j][k];
					temp = "proc" + ss.str() + ".hdf";
					proc_file_id[mappa[i][j][k]] = H5Fopen(temp.c_str(), H5F_ACC_RDWR, H5P_DEFAULT); 
					// read data
					// species 1: current sheet ions
					temp = "/moments/species_1/Jx/cycle_"+ cc.str(); 
					dataset_id = H5Dopen2(proc_file_id[mappa[i][j][k]],temp.c_str(), H5P_DEFAULT); // HDF 1.8.8
					status = H5Dread(dataset_id, H5T_NATIVE_DOUBLE, H5S_ALL,H5S_ALL,H5P_DEFAULT,temp_storageX); status = H5Dclose(dataset_id);
					temp = "/moments/species_1/Jy/cycle_"+ cc.str(); 
					dataset_id = H5Dopen2(proc_file_id[mappa[i][j][k]],temp.c_str(), H5P_DEFAULT); // HDF 1.8.8
					status = H5Dread(dataset_id, H5T_NATIVE_DOUBLE, H5S_ALL,H5S_ALL,H5P_DEFAULT,temp_storageY); status = H5Dclose(dataset_id);
					temp = "/moments/species_1/Jz/cycle_"+ cc.str(); 
					dataset_id = H5Dopen2(proc_file_id[mappa[i][j][k]],temp.c_str(), H5P_DEFAULT); // HDF 1.8.8
					status = H5Dread(dataset_id, H5T_NATIVE_DOUBLE, H5S_ALL,H5S_ALL,H5P_DEFAULT,temp_storageZ); status = H5Dclose(dataset_id);
					node=0;
					for (int ii=0; ii < (nxn+1);ii++)
						for (int jj=0; jj < (nyn+1);jj++)
							for (int kk=0; kk < (nzn+1);kk++){
								if (ii!= nxn && jj!= nyn && kk!=nzn){
                                    if (fabs(JX[ii][jj][kk]) < 1E-16)
                                        JX[ii][jj][kk] = 0.0;
                                    if (fabs(JY[ii][jj][kk]) < 1E-16)
                                        JY[ii][jj][kk] = 0.0;
                                    if (fabs(JZ[ii][jj][kk]) < 1E-16)
                                        JZ[ii][jj][kk] = 0.0;
									JX[ii + nxn*i][jj + nyn*j][kk + nzn*k] = temp_storageX[node];
									JY[ii + nxn*i][jj + nyn*j][kk + nzn*k] = temp_storageY[node];
									JZ[ii + nxn*i][jj + nyn*j][kk + nzn*k] = temp_storageZ[node];
								}
								node++;
							}
					
					
					
					// close the file
					H5Fclose(proc_file_id[mappa[i][j][k]]);
					// go to other proc
					proc++;
				}
		
		for (int kk=0; kk < nzn*ZLEN;kk++)
			for (int jj=0; jj < nyn*YLEN;jj++)
				for (int ii=0; ii < nxn*XLEN;ii++){
                    
					my_fileJi << JX[ii][jj][kk] << " " << JY[ii][jj][kk] << " " << JZ[ii][jj][kk] << endl;
				}
		my_fileJi.close();
		// end Ji
		
		
				
		// rhoi
		// write charge density
		cout << "7/8 Reading and writing rhoi" << endl;
		proc = 0;
		for (int i=0; i < XLEN;i++)
			for (int j=0; j < YLEN;j++)
				for (int k=0; k < ZLEN;k++){
					stringstream ss;
					ss << mappa[i][j][k];
					temp = "proc" + ss.str() + ".hdf";
					proc_file_id[mappa[i][j][k]] = H5Fopen(temp.c_str(), H5F_ACC_RDWR, H5P_DEFAULT); 
					// read data
					temp = "/moments/species_1/rho/cycle_"+ cc.str(); 
					dataset_id = H5Dopen2(proc_file_id[mappa[i][j][k]],temp.c_str(), H5P_DEFAULT); // HDF 1.8.8
					status = H5Dread(dataset_id, H5T_NATIVE_DOUBLE, H5S_ALL,H5S_ALL,H5P_DEFAULT,temp_storageX); status = H5Dclose(dataset_id);
					node=0;
					for (int ii=0; ii < (nxn+1);ii++)
						for (int jj=0; jj < (nyn+1);jj++)
							for (int kk=0; kk < (nzn+1);kk++){
								if (ii!= nxn && jj!= nyn && kk!=nzn){  
									rho[ii + nxn*i][jj + nyn*j][kk + nzn*k]  = temp_storageX[node];
									
								}
								node++;
							}
					
					
					
					// close the file
					H5Fclose(proc_file_id[mappa[i][j][k]]);
					// go to other proc
					proc++;
				}
		//cout << "WRITING VECTOR rhoi TO VTK FILE" << endl;
		for (int kk=0; kk < nzn*ZLEN;kk++)
			for (int jj=0; jj < nyn*YLEN;jj++)
				for (int ii=0; ii < nxn*XLEN;ii++){
					my_filerhoi << 4*3.1415*rho[ii][jj][kk]  << endl;
				}
		my_filerhoi.close();
		// end rhoi
		
		// start Ve
		cout << "8/8 writing Vi" << endl;
		for (int kk=0; kk < nzn*ZLEN;kk++)
			for (int jj=0; jj < nyn*YLEN;jj++)
				for (int ii=0; ii < nxn*XLEN;ii++){
					my_fileVi << (JX[ii][jj][kk]/rho[ii][jj][kk]) << " " << (JY[ii][jj][kk]/rho[ii][jj][kk]) << " " << (JZ[ii][jj][kk]/rho[ii][jj][kk]) << endl;
				}
		my_fileVi.close();
		// end Vi
		cout << endl;
		
	} // end of simulation cycle
	
	delete[] proc_file_id;
	delete[] temp_storageX;
	delete[] temp_storageY;
	delete[] temp_storageZ;
	delArr3(BX,nxn*XLEN,nyn*YLEN);
	delArr3(BY,nxn*XLEN,nyn*YLEN);
	delArr3(BZ,nxn*XLEN,nyn*YLEN);
	delArr3(EX,nxn*XLEN,nyn*YLEN);
	delArr3(EY,nxn*XLEN,nyn*YLEN);
	delArr3(EZ,nxn*XLEN,nyn*YLEN);
	delArr3(JX,nxn*XLEN,nyn*YLEN);
	delArr3(JY,nxn*XLEN,nyn*YLEN);
	delArr3(JZ,nxn*XLEN,nyn*YLEN);
	delArr3(rho,nxn*XLEN,nyn*YLEN);
	
	
	return(0);
}




























