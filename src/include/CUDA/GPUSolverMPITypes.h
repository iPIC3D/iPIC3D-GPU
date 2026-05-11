/**
 * @file GPUSolverMPITypes.h
 * @brief Provides mpiTypeOf<T>() helper to map cudaSolverType (and float)
 *        to the corresponding MPI_Datatype.
 *
 * Include this from any .cu or .cpp file that performs MPI operations
 * using cudaSolverType buffers.
 */
#pragma once
#include <mpi.h>
#include "cudaTypeDef.cuh"

/// Returns the MPI_Datatype corresponding to the template parameter T.
/// Specialised for double and float; any other type yields a link error.
template<typename T> inline MPI_Datatype mpiTypeOf() = delete;

template<> inline MPI_Datatype mpiTypeOf<double>() { return MPI_DOUBLE; }
template<> inline MPI_Datatype mpiTypeOf<float>()  { return MPI_FLOAT; }
