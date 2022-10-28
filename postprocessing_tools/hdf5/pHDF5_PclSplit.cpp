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

#include "mpi.h"
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


  int rank, size;

  MPI_Init (&argc, &argv);/* starts MPI */
  MPI_Comm_rank (MPI_COMM_WORLD, &rank);/* get current process id */
  MPI_Comm_size (MPI_COMM_WORLD, &size);/* get number of processes */

	// cycle we want to open
	int start_cycle;
	int step_cycle;
	int end_cycle;
	// 3 inputs first cycle, step, and end cycle
	sscanf(argv[1],"%d",&start_cycle);
	sscanf(argv[2],"%d",&step_cycle);
	sscanf(argv[3],"%d",&end_cycle);

	
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

	// at this point you can close settings
	status = H5Fclose(file_id);

	cout << " Successful in reading settings.hdf, XLEN=" << XLEN << ", YLEN=" << YLEN <<", ZLEN=" << ZLEN  << endl;
	// prepare to read the proc files
	hid_t    proc_file_id;
	string temp;
	int* cartesian_cor= new int[3];
	
    // this should be changed if higher than 30 procs per dimension
	const int xproc = XLEN;
	const int yproc = YLEN;
	const int zproc = ZLEN;
	int nop0, nop1;
	hid_t datatype, dataspace;
	hsize_t  dims_out[1];
    int mappa[xproc][yproc][zproc];
    
    for(int icycle=start_cycle;icycle<=end_cycle;icycle=icycle+step_cycle)
	for (int xi=0; xi < XLEN; xi++)
		for (int yi=0; yi < YLEN; yi++)
			for (int zi=0; zi < ZLEN; zi++){

		stringstream ss;
		ss << "restart" << (xi*(YLEN*ZLEN) + yi*ZLEN + ZLEN) << ".hdf";
		proc_file_id = H5Fopen(ss.str().c_str(), H5F_ACC_RDWR, H5P_DEFAULT);
		if (proc_file_id  < 0){
			cout << "couldn't open file:  "<< ss.str() << endl;
			return -1;
		}
		// read the position in the topology
		dataset_id = H5Dopen2(proc_file_id, "/topology/cartesian_coord", H5P_DEFAULT); // HDF 1.8.8
		status = H5Dread(dataset_id, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT,cartesian_cor);
		status = H5Dclose(dataset_id);

		ss.str("");
		ss << "/particles/species_0/x/cycle_" << icycle;
		dataset_id = H5Dopen2(proc_file_id,ss.str().c_str(), H5P_DEFAULT); // HDF 1.8.8
		datatype  = H5Dget_type(dataset_id);
		dataspace = H5Dget_space(dataset_id);
		status = H5Sget_simple_extent_dims(dataspace, dims_out, NULL);
		status = H5Dclose(dataset_id);
		nop0=dims_out[0];

		ss.str("");
		ss << "/particles/species_1/x/cycle_" << icycle;
		dataset_id = H5Dopen2(proc_file_id,ss.str().c_str(), H5P_DEFAULT); // HDF 1.8.8
		datatype  = H5Dget_type(dataset_id);
		dataspace = H5Dget_space(dataset_id);
		status = H5Sget_simple_extent_dims(dataspace, dims_out, NULL);
		status = H5Dclose(dataset_id);
		nop1=dims_out[0];

		cout << ss.str() << " at [" << cartesian_cor[0] << "][" << cartesian_cor[1] << "][" << cartesian_cor[2] << "], species_0: " << nop0 << ", species_1: " << nop1 ;


		H5Fclose(proc_file_id);
			}

	MPI_Finalize();	
	return(0);
}




























