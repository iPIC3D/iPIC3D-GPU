#ifndef FIELD_BOUNDARY_H
#define FIELD_BOUNDARY_H

/**
 * @file FieldBoundary.h
 * @brief The field boundary conditions: every type that defines them.
 *
 *  - The spec (FieldBoundarySpec::build): what each face of this rank is,
 *    decided once at construction of the fields.  The one place where the
 *    solver reads the boundary inputs (bcEMface*, bcPface*, the absorbing
 *    layer, the open-face options, ForceWallB) and the topology (the
 *    neighbours, the periodic and the flat axes).
 *  - The conductor image (Image, ImageWalls): the parity of every quantity
 *    the field solve reads about a perfect conductor.
 *  - The four mechanisms every boundary treatment of the fields is one of:
 *    - ghost rules (GhostRule, GhostSet): the ghost cells or nodes of the
 *      physical faces, from interior values, a constant or the conductor
 *      wall value; they complement the halo exchange;
 *    - boundary rows (RowFaces): the identity rows of the face nodes (a
 *      conductor's E_t, an inflow face's E0) and the face values imposed on
 *      E; a conductor's normal row is the interior row, read through the
 *      conductor image;
 *    - layer operations (LayerOp, LayerSet): the layers next to open faces
 *      (absorbing blend, E0/B0, copies) and the GEM/ForceFree wall layers;
 *    - moment fold: the image of the deposit on the face planes.
 * The GPU plan that derives the rules, rows and layer lists from the spec,
 * and the calls the field solver makes (its only boundary entry):
 * GPUFieldBoundary.cuh.
 *
 * Faces have one order everywhere in the GPU field solver: XL, XR, YL, YR,
 * ZL, ZR (face f: axis f / 2, high side f % 2).
 */

#include "cudaTypeDef.cuh" // cudaSolverType

class Collective;
class VCtopology3D;
typedef VCtopology3D VirtualTopology3D; // as in ipicfwd.h

// =========================================================================
//  Faces and the spec
// =========================================================================

/** Faces of a subdomain, axis by axis, low side first. */
enum Face : int { FACE_XL, FACE_XR, FACE_YL, FACE_YR, FACE_ZL, FACE_ZR };
inline constexpr int kFaces = 6;
/** Axis of face f (0 x, 1 y, 2 z). */
constexpr int faceAxis(int f) { return f / 2; }
/** True for the high face of its axis. */
constexpr bool faceHigh(int f) { return f % 2 == 1; }

/** What a face of this rank is, for the fields. */
enum class FaceKind : int {
  Neighbour, ///< an MPI neighbour or a periodic image: the halo fills it
  Conductor, ///< bcEMface = 0: a perfect electric conductor at rest
  Mirror,    ///< bcEMface = 1: Collective's codes, nothing more
  Open,      ///< bcEMface = 2 (inflow, outflow and lateral open faces)
};

/** The GEM/ForceFree wall layers: B reset to the case's equilibrium profile
 *  every cycle (input ForceWallB, off by default: None). */
enum class CaseWall : int { None, Gem, ForceFree };

/** One face of this rank. */
struct FaceSpec {
  FaceKind kind = FaceKind::Neighbour; // in the field topology
  int bcEM = 0;                        // bcEMface of the face (input)
  int bcP = 0;                         // bcPface of the face (input)
  bool particlePhysical = false;       // no neighbour in the particle topology
  bool physical() const { return kind != FaceKind::Neighbour; }
  bool conductor() const { return kind == FaceKind::Conductor; }
  bool open() const { return kind == FaceKind::Open; }
};

/** The boundary of this rank's fields, fixed for the whole run. */
struct FieldBoundarySpec {
  FaceSpec face[kFaces];
  int cells[3] = {0, 0, 0}; // local cell extents, ghosts included
  int nodes[3] = {0, 0, 0}; // local node extents, ghosts included
  bool periodic[3] = {};    // the global axes (the topology)
  // one cell, periodic onto this rank: the operators drop the axis (the
  // flat direction of a 2D run)
  bool flat[3] = {};

  // ---- open faces (bcEMface = 2) ----
  bool sal = false;            // absorbing layer (yes_sal)
  int salLayers = 0;           // n_layers_sal
  bool salTangentialB = false; // the absorbing layer leaves B_n alone
  bool inflowRows = false;     // ApplyInflowBcsEImage: identity rows, E0
  double E0[3] = {0, 0, 0};    // -(u0 x B0) / c of species 0
  double B0[3] = {0, 0, 0};
  // X-right open outflow face (bcEMface = 2, bcPface = 3) of this rank
  bool outflowXright = false;
  // its layers L: node planes nxn-1-L..nxn-1 (cells nxc-1-L..nxc-1) take
  // the values of plane nxn-2-L (nxc-2-L).  The neutralized charge, the
  // copied moments and, with OutflowLayerFields = copy, E and B share
  // L = max(3, n_layers_sal), so that the L cells between the reference
  // plane and the face contain the 3-cell particle open layer
  int outflowLayers = 0;
  // layers over which E and B are copied after a solve: outflowLayers,
  // except with OutflowLayerFields = solved, where it is 0 (only the ghost
  // planes take the face values, the zero-gradient continuation the Maxwell
  // operator assumes)
  int outflowCopyLayers = 0;
  // GhostRowRule of the y and z ghost rows of its ghost cell plane (y-left,
  // y-right, z-left, z-right): a physical face keeps its own rule, a
  // periodic image on this rank gives the exact partner row, and a
  // neighbour rank's row, whose face-plane nodes the operator does not
  // exchange, is stood in for by the adjacent row
  int outflowRows[4] = {0, 0, 0, 0};

  // ---- case walls ----
  CaseWall caseWall = CaseWall::None;
  double yStart = 0, dy = 0, halfLy = 0, delta = 0; // the wall profiles

  /** Face f carries an inflow face's rows and layers: open, bcPface = 2,
   *  not X-right (an open X-right face keeps its operator rows). */
  bool inflow(int f) const {
    return face[f].open() && face[f].bcP == 2 && f != FACE_XR;
  }
  /** The B layers of open face f are treated: the local cell count along
   *  its normal, ghosts included, is above 10 (an inherited gate). */
  bool openBLayers(int f) const {
    return face[f].open() && cells[faceAxis(f)] > 10;
  }
  /** Those B layers copy a reference layer: X-right, and without the
   *  absorbing layer the open faces other than X-left and the inflow faces
   *  (which hold B0). */
  bool openBCopy(int f) const {
    return f == FACE_XR || !(sal || f == FACE_XL || inflow(f));
  }
  bool anyConductor() const {
    for (const FaceSpec& s : face)
      if (s.conductor())
        return true;
    return false;
  }
  /** Both global faces of axis a are perfect conductors: the same on every
   *  rank, whether or not it owns them (the null spaces of the cleanings). */
  bool conductorAxis(int a) const {
    return !periodic[a] && face[2 * a].bcEM == 0 && face[2 * a + 1].bcEM == 0;
  }

  /** The boundary of this rank from the input and the topology.  E0 is the
   *  drift field of species 0; cells the local cell extents (ghosts
   *  included). */
  static FieldBoundarySpec build(const Collective& col,
                                 const VirtualTopology3D& vct,
                                 const int cells[3], const double E0[3],
                                 double yStart, double dy);
};

// =========================================================================
//  Ghost rules
// =========================================================================

/** The value of a ghost point, from the points 1, 2 or m layers inward along
 *  the face normal (f(1), f(2), f(m)).  Collective's face codes (bcEx, ...)
 *  are 0 OddCell, 1 Zero, 2 EvenCell. */
enum GhostRule : int {
  GHOST_KEEP,      ///< left as it is (the exchange, or a stencil, wrote it)
  GHOST_ZERO,      ///< 0
  GHOST_EVEN_CELL, ///< f(1): zero gradient, or even about a cell face
  GHOST_ODD_CELL,  ///< -f(1): odd about a cell face
  GHOST_EVEN_NODE, ///< f(2): even about the face node
  GHOST_ODD_NODE,  ///< 2 f(1) - f(2): odd about the face node's value
  GHOST_WALL,      ///< 2 W - f(1): odd about the conductor wall value W
  GHOST_CONST,     ///< a constant (E0, B0, a case-wall profile)
  GHOST_COPY,      ///< f(m): a copy of interior layer m
};

/** The conductor image: the reflection in the wall with charge conjugation
 *  (the image charge is opposite), under which Maxwell's equations and the
 *  Lorentz force are invariant, whatever the particles do at the wall.  A
 *  polar vector (E, J, Jhat, mu E) keeps its normal component and reverses
 *  the tangential ones, an axial vector (B) the reverse, a charge-like scalar
 *  (rho, rhohat, div E, div(mu E)) changes sign.  Every quantity the field
 *  solve reads takes this image: its ghosts, and 0 on the face plane where
 *  it is odd (the even ones keep the folded face value: real plus image is
 *  twice the half cell's deposit). */
enum class Image : int { Polar, Axial, Scalar };
/** Component c of a quantity of kind k is odd about a face of axis n. */
constexpr bool imageOdd(Image k, int c, int n) {
  return k == Image::Scalar || ((c == n) == (k == Image::Axial));
}

/** The cells next to the conductor walls: lo[a], hi[a] the index of those
 *  cells for the low and the high face of axis a (1 and n-2), -1 where there
 *  is no conductor.  A node->cell stencil in such a cell reads the face node
 *  plane through the conductor image: a charge-like scalar as 0, a polar
 *  vector's components tangential to the face as 0, its normal component as
 *  stored (GPUStencils.cuh, imagePlanes).  The default has none. */
struct ImageWalls {
  int lo[3] = {-1, -1, -1};
  int hi[3] = {-1, -1, -1};
};

/** One face rule of one array, as the fill kernel reads it.  rank: the index
 *  of the pass that writes it in its set (codes, then the open-face,
 *  case-wall and conductor-wall overrides); a ghost point on several faces
 *  composes their rules in (rank, axis) order. */
struct GhostRuleDev {
  const cudaSolverType* wall = nullptr; // GHOST_WALL: 2 W per cell of the
                                        // face's ghost plane
  cudaSolverType value = 0;             // GHOST_CONST
  int kind = GHOST_KEEP;                // GhostRule
  int rank = 0;                         // order of composition at edges
  int copy = 0;                         // GHOST_COPY: the layer m
};

/** Most arrays filled by one launch. */
inline constexpr int kGhostBatchMax = 10;

/** The arrays of one fill (blockIdx.y selects one).  Array a takes the rules
 *  of slot a % slots of the set's rule table (device memory,
 *  [slots][kFaces], built once): the kernels index no kernel argument by a
 *  run-time value. */
struct GhostBatch {
  cudaSolverType* p[kGhostBatchMax] = {};
  int n = 0;
  int slots = 1;
  const GhostRuleDev* rules = nullptr;
};

/** Points of the planes of the active faces of a box, each point once: face
 *  f's slab is its plane minus the planes of the active faces before it, so
 *  edges and corners belong to the first of their faces.  Ghost shells: the
 *  ghost planes (index 0, n-1), whole.  Node planes: the face node planes
 *  (index 1, n-2), interior [1, n-2] (rows and moment fold).  On face f the
 *  in-plane axes are the slow one (y on x faces, x otherwise) and the fast
 *  one (z, or y on z faces): consecutive threads run along k on x and y
 *  faces. */
struct FaceSlabs {
  int n[3] = {0, 0, 0}; // array extents
  bool active[kFaces] = {};
  int plane[kFaces] = {};  // index of face f's plane along its axis
  int slowLo[kFaces] = {}; // slab f: slow axis from slowLo,
  int fastLo[kFaces] = {}; // fast axis [fastLo, fastLo + fastN)
  int fastN[kFaces] = {};
  int start[kFaces + 1] = {}; // slab f: threads [start[f], start[f+1])
  int count() const { return start[kFaces]; }
};

/** The ghost sets: the arrays a fill applies to, with their rules. */
enum class GhostSet : int {
  ENode,          ///< E: the face codes (conductor: image of E)
  ENodeSmooth,    ///< E smoothing passes: even about the face node
  BNode,          ///< node B: the face codes (conductor: image of B)
  BNodeConductor, ///< node B: the conductor faces only (after the GEM planes)
  BCell,          ///< cell B: codes, open-face and case-wall ghosts, and
                  ///< the conductor wall (without a case wall)
  BCellWall,      ///< cell B: the conductor wall when there is a case wall,
                  ///< after its interior layers (LayerSet::CaseCells)
  BCellStart,     ///< cell B at start-up: codes and wall ghosts
  BCellSource,    ///< cell B before the source curl: the ForceFree ghosts,
                  ///< the open faces, the GEM ghosts (no codes)
  OpCells,        ///< the operator's 9 gradients of E and div(mu E)
  OpCellsLocal,   ///< the same for the local (block-Jacobi) operator
  CellChargeP,    ///< the charge of the field solve (rho_c, rhohat), particle
                  ///< topology: the image (odd) on conductors, even elsewhere
  CellCurrentP,   ///< the pressure term of Jhat, div(Pi), particle topology:
                  ///< the image on conductors (tangential odd), even
                  ///< elsewhere
  NodeCurrentP,   ///< Jhat (smoothing), particle topology: the image on
                  ///< conductors (Jhat_t odd), even about the face node
                  ///< elsewhere
  Phi,            ///< the div(E) cleaning potential: the image (odd) on
                  ///< conductors, 0 half a cell outside the other faces
  Psi,            ///< the div(B) cleaning potential: no rule (the geometry
                  ///< of its exchange)
  Count
};

/** Rule for the ghost rows of the X-right ghost cell plane, per side (the
 *  outflow closure of the Maxwell operator). */
enum GhostRowRule : int {
  GHOST_ROW_KEEP = 0,     ///< physical face: left to that face's rule
  GHOST_ROW_PERIODIC = 1, ///< own periodic image: copy the partner row
  GHOST_ROW_ADJACENT = 2  ///< neighbour rank: copy the adjacent row
};

// =========================================================================
//  Boundary rows of the Maxwell system
// =========================================================================

/** The special face rows of this rank: per face, none, a perfect conductor
 *  (E_t identity rows with source 0; E_n keeps the interior row), or an
 *  inflow face with identity rows (E0).  At edges and corners the inflow
 *  rows win over the conductor ones. */
enum RowFace : int { ROW_NONE, ROW_CONDUCTOR, ROW_INFLOW };
struct RowFaces {
  FaceSlabs planes;      // node planes of the faces with rows
  int kind[kFaces] = {}; // RowFace
  cudaSolverType E0[3] = {0, 0, 0};
};

// =========================================================================
//  Layer operations
// =========================================================================

enum LayerKind : int {
  LAYER_BLEND, ///< absorbing layer: v sal + target (1 - sal)
  LAYER_SET,   ///< the components of mask set to target
  LAYER_COPY,  ///< the reference layer ref copied to the layers
};

/** One operation on the layers [first, last] normal to axis, whole along
 *  the other axes. */
struct LayerOp {
  int kind = LAYER_SET;
  int axis = 0, first = 0, last = 0;
  int ref = 0; // LAYER_COPY
  cudaSolverType target[3] = {0, 0, 0};
  int mask = 7;                 // LAYER_SET, LAYER_BLEND: components written
  cudaSolverType invLayers = 1; // LAYER_BLEND: 1 / n_layers_sal
  int ascending = 1;            // LAYER_BLEND: the weight grows with the layer
};

/** The layer sets: ordered operations on three node or cell arrays. */
enum class LayerSet : int {
  E,              ///< E after the solve: inflow E0 (or blend), X-right copy
  B,              ///< cell B after Faraday's law: open-face layers
  CaseCells,      ///< cell B: the GEM/ForceFree wall layers
  CaseNodes,      ///< node B: the GEM wall planes
  OutflowNodes,   ///< operator node arrays: the X-right ghost node plane
  OutflowMoments, ///< species J, P: the X-right outflow layer
  Count
};

#endif // FIELD_BOUNDARY_H
