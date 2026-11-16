#ifndef GPU_HALO_COMM_CUH
#define GPU_HALO_COMM_CUH

/**
 * @file GPUHaloComm.cuh
 * @brief GPU-aware MPI halo communication for iPIC3D GPU field solver.
 *
 * The plan of the exchanges of device-resident field arrays: an exchange
 * sends every ghost box (6 faces, 12 edges, 8 corners) straight to the
 * neighbour across it, through contiguous device buffers and CUDA-aware MPI
 * (GPUBatchedHaloComm.cu).  The ghost cells and nodes of the physical faces
 * are the boundary rules of FieldBoundary (GPUFieldBoundary.cuh), applied
 * after the exchange (EMfields3D::exchangeAndFill, GPUHaloComm.cu).
 */

#include "GPUFieldArray.cuh"
#include <mpi.h>

// =========================================================================
//  Halo exchange plan (GPUBatchedHaloComm.cu)
// =========================================================================

/** One box of a halo exchange in an (nx, ny, nz) array: li x lj x lk points
 *  from the linear index field, k fastest.  Pack copies the send box of a
 *  remote neighbour into the neighbour's message, unpack the message into
 *  the ghost box at field; point e of field f is element
 *  message * nFields + f * length + offset + e of the buffer.  A self copy
 *  (self >= 0, a ghost box along an axis periodic on one process) copies the
 *  box at field into the ghost box at self. */
struct HaloBox {
  int block;   // first block of the box in the launch
  int points;  // li * lj * lk
  int lj, lk;  // extents along j and k
  int field;   // origin: send box (pack), ghost box (unpack), source (self)
  int self;    // ghost box origin of a self copy, -1 for the buffer
  int message; // per-field offset of the neighbour's message in the buffer
  int length;  // per-field length of that message
  int offset;  // per-field offset of the box in that message
};

/** The boxes of one copy launch, passed by value: blockIdx.x walks the
 *  boxes, blockIdx.y the fields. */
struct HaloCopyTable {
  int boxes = 0;
  int blocks = 0;
  int ny = 0, nz = 0;
  HaloBox box[26] = {};
};

/** A remote neighbour of an exchange: one message for all its boxes. */
struct HaloPeer {
  int rank = MPI_PROC_NULL;
  int offset = 0; // per-field offset of the message in the buffers
  int length = 0; // per-field length of the message
};

/** The exchange of one communicator, array geometry and extent, built once
 *  when the halo buffers are allocated. */
struct HaloPlan {
  MPI_Comm comm = MPI_COMM_NULL;
  bool present[6] = {}; // face neighbours XL XR YL YR ZL ZR exist
  int peers = 0;        // distinct remote neighbours
  HaloPeer peer[26] = {};
  int bufferPoints = 0; // per field: the length of all messages
  HaloCopyTable pack;   // remote send boxes, self copies of interior boxes
  HaloCopyTable unpack; // remote ghost boxes
  HaloCopyTable fill;   // self copies reading ghost boxes the unpack writes
};

#endif // GPU_HALO_COMM_CUH
