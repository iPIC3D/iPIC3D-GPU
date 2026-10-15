/**
 * @file OutflowBoundary.h
 * @brief Codes shared by the CPU and GPU field solvers for the X-right open
 *        outflow face (bcEMfaceXright = 2, bcPfaceXright = 3).
 *
 * The Maxwell operator closes an outflow face with a zero-gradient ghost
 * node: the ghost node plane of its node work arrays, which is not part of
 * the Krylov vector, copies the face plane, and the node-to-cell stencils
 * also evaluate the X-right ghost cell plane from it, so the face row is the
 * interior operator applied to that continuation. The halo BC of the
 * operator's cell arrays keeps that plane (BC_FACE_KEEP), and its ghost rows,
 * the y/z ghost cells the stencils do not reach, follow one rule per side
 * (GhostRowRule).
 */
#ifndef OUTFLOW_BOUNDARY_H
#define OUTFLOW_BOUNDARY_H

/** Face boundary code that leaves a ghost layer unchanged (BCface, gpuBCface;
 *  no case of their switch matches it). */
constexpr int BC_FACE_KEEP = -1;

/** Rule for the ghost rows of the X-right ghost cell plane, per side. */
enum GhostRowRule : int {
  GHOST_ROW_KEEP = 0,     ///< physical face: left to that face's BC
  GHOST_ROW_PERIODIC = 1, ///< own periodic image: copy the partner row
  GHOST_ROW_ADJACENT = 2  ///< neighbour rank: copy the adjacent row
};

#endif // OUTFLOW_BOUNDARY_H
