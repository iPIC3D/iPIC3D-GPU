#ifndef GPU_FIELD_BOUNDARY_CUH
#define GPU_FIELD_BOUNDARY_CUH

/**
 * @file GPUFieldBoundary.cuh
 * @brief The GPU side of the field boundary conditions: the plan built once
 *        from the FieldBoundarySpec, the calls the field solver makes at a
 *        boundary (FieldBoundary, its only boundary entry; the halo exchange
 *        of a set is EMfields3D::exchangeAndFill), and the kernels that
 *        apply them.  The types that define the boundary conditions:
 *        FieldBoundary.h.
 *
 * Each mechanism is one launch for all faces and all arrays of a call: one
 * fill of the ghost points (fill), one pass of the boundary rows (imageRows,
 * rowValues, zeroTangential), the moment fold (foldMoments); the layer
 * operations run in their order (layers).  The plan is fixed for the run:
 * the hot loop only passes the cached tables to the kernels.
 *
 * A set is a sequence of whole-plane passes (face codes, then the open-face,
 * case-wall and conductor-wall overrides): a ghost point on several faces
 * composes their rules in the order of those passes (GhostRuleDev::rank),
 * then of the axes.
 */

#include "FieldBoundary.h"
#include "GPUFieldArray.cuh"
#include "cudaTypeDef.cuh"

// =========================================================================
//  Kernel launchers
// =========================================================================

/** Ghost planes (index 0, n-1, whole) of the faces with active[f]. */
FaceSlabs ghostShell(const int n[3], const bool active[kFaces]);
/** Face node planes (index 1, n-2, interior [1, n-2]) of active faces. */
FaceSlabs nodePlanes(const int n[3], const bool active[kFaces]);

/** Fill the ghost points of shell for the arrays of batch. */
void gpuFillGhosts(const GhostBatch& batch, const FaceSlabs& shell,
                   cudaStream_t stream);
/** The identity rows of the face nodes into the Krylov image im: im = v for
 *  the components of a conductor's E_t and of an inflow face (node extents
 *  of rows.planes).  A conductor's normal row is left as the operator wrote
 *  it, the interior row read through the conductor image. */
void gpuBoundaryImageRows(const RowFaces& rows, cudaSolverType* im,
                          const cudaSolverType* const v[3],
                          cudaStream_t stream);
/** Values on the face planes of the node vector v: E0 on the inflow faces
 *  (with inflow), then 0 for the components tangential to a conductor. */
void gpuBoundaryFaceValues(const RowFaces& rows, cudaSolverType* const v[3],
                           bool inflow, cudaStream_t stream);
/** Apply op to the node or cell arrays X, Y, Z of extents n. */
void gpuLayerOp(const LayerOp& op, cudaSolverType* X, cudaSolverType* Y,
                cudaSolverType* Z, const int n[3], cudaStream_t stream);
/** Multiply the moments on the face node planes of the particle-physical
 *  faces by 2 per face (the image of the deposit): d_ptrs holds nFields
 *  device pointers. */
void gpuFoldMoments(const FaceSlabs& planes, cudaSolverType* const* d_ptrs,
                    int nFields, cudaStream_t stream);
/** 2 W per cell of the ghost plane of conductor face f, W the mean of the 4
 *  face nodes of the cell face in the node field Bn (its normal component;
 *  node extents nodes). */
void gpuConductorWallSum(cudaSolverType* sum, const cudaSolverType* Bn,
                         const int nodes[3], int f, cudaStream_t stream);
/** Neutral species densities on the X layers [first, last] (the X-right
 *  outflow layer): every layer takes the densities of plane ref, with the
 *  positive and negative species rescaled to the same magnitude. */
void gpuNeutralizeLayersX(cudaSolverType* rhons, const cudaSolverType* qom,
                          int ns, int first, int last, int ref, int nx, int ny,
                          int nz, cudaStream_t stream);
/** Fill the y and z ghost rows of the X-right ghost cell plane i = nxc-1 of
 *  the cell arrays of batch (its rules unused; GhostRowRule rule[0..3] =
 *  y-left, y-right, z-left, z-right).  The y rows first, then the z rows
 *  over the full y range. */
void gpuGhostRowsXright(const GhostBatch& batch, int nxc, int nyc, int nzc,
                        const int rule[4], cudaStream_t stream);

// =========================================================================
//  The plan
// =========================================================================

/** The GPU plan of the field boundary conditions: the rules of every ghost
 *  set, the rows and the layer lists, derived once from the spec, and the
 *  calls that apply them.  Every boundary treatment of the field solver is
 *  one of these calls, the case walls and the X-right outflow face
 *  included; EMfields3D adds the halo exchange before a fill
 *  (exchangeAndFill, finishE in GPUHaloComm.cu).  The face rows of the
 *  cleaning operators (lap27Op) are built from the spec. */
class FieldBoundary {
public:
  FieldBoundary() = default;
  ~FieldBoundary();
  FieldBoundary(const FieldBoundary&) = delete;
  FieldBoundary& operator=(const FieldBoundary&) = delete;

  /** The spec, at construction of the fields (host only). */
  void setSpec(const FieldBoundarySpec& spec) { spec_ = spec; }
  const FieldBoundarySpec& spec() const { return spec_; }

  /** The GPU plan, once: the rules, rows and layer lists. */
  void buildPlan();
  /** Free the device data of the plan (the wall values, the rule tables);
   *  the plan has to be built again before use. */
  void releaseDevice();
  /** Conductor faces: the wall values W from the node B (start-up and
   *  restart).  Before any fill of BCell, BCellWall or BCellStart. */
  void initWall(const cudaSolverType* const Bn[3], cudaStream_t stream);

  /** Geometry and topology of a set's arrays, for the halo exchange: cells
   *  or nodes, their extents with ghosts, the particle or the field
   *  communicator. */
  bool setOnCells(GhostSet s) const { return set(s).cells; }
  const int* extents(GhostSet s) const {
    return set(s).cells ? spec_.cells : spec_.nodes;
  }
  bool setParticle(GhostSet s) const { return set(s).particle; }
  /** Fill the ghost points of the n arrays p (array i takes the rules of
   *  slot i modulo the set's slot count). */
  void fill(GhostSet s, cudaSolverType* const* p, int n,
            cudaStream_t stream) const;

  /** The identity rows of the Maxwell image im (Krylov layout) of the node
   *  vector v: im = v for a conductor's E_t and on the inflow faces. */
  void imageRows(cudaSolverType* im, const cudaSolverType* const v[3],
                 cudaStream_t stream) const;
  /** The values of those rows written into the node field E: E0 on the
   *  inflow faces, 0 for the components tangential to a conductor (the
   *  right-hand side, every smoothing pass of E, E^n after the div(E)
   *  correction). */
  void rowValues(cudaSolverType* const E[3], cudaStream_t stream) const;
  /** The conductor image of the polar node vector v on the face planes: its
   *  components tangential to a conductor 0 (E, the solved E, Jhat).  No-op
   *  without a conductor. */
  void zeroTangential(cudaSolverType* const v[3], cudaStream_t stream) const;

  /** The operations of set s, in order, on X, Y, Z. */
  void layers(LayerSet s, cudaSolverType* X, cudaSolverType* Y,
              cudaSolverType* Z, cudaStream_t stream) const;

  /** The cell B before the curl of the Maxwell source: the ForceFree wall
   *  layers, the ghosts of BCellSource (the ForceFree ghosts, the ghost
   *  layers of the open faces, the GEM ghosts), the GEM wall layers, and the
   *  conductor wall after them (BCellWall, only with a case wall). */
  void sourceCellB(cudaSolverType* const Bc[3], cudaStream_t stream) const;
  /** The cell B after its BCell fill: the case-wall layers, then the
   *  conductor wall over them (BCellWall); no-op without a case wall. */
  void caseWallCells(cudaSolverType* const Bc[3], cudaStream_t stream) const;
  /** The node B after its BNode fill: the GEM wall planes (B_y of the ghost
   *  node included), then the conductor images again (BNodeConductor);
   *  no-op without the GEM case wall. */
  void caseWallNodes(cudaSolverType* const Bn[3], cudaStream_t stream) const;

  /** The moment fold on the particle-physical faces (d_ptrs: nFields device
   *  pointers of node arrays). */
  void foldMoments(cudaSolverType* const* d_ptrs, int nFields,
                   cudaStream_t stream) const;
  /** X-right outflow face: the ghost rows of the operator's cell arrays. */
  void outflowGhostRows(cudaSolverType* const* p, int n,
                        cudaStream_t stream) const;
  /** X-right outflow face: neutral species densities rhons (ns species, qom
   *  on the device) on the layers of LayerSet::OutflowMoments. */
  void neutralizeOutflow(cudaSolverType* rhons, const cudaSolverType* qom,
                         int ns, cudaStream_t stream) const;
  /** X-right outflow face: rhohat = 0 on every cell beyond the reference
   *  node x_s (the cell straddling it, the layer cells and the ghost cell),
   *  so that the charge the solve sees ends at x_s like the post-solve field
   *  copy (LayerSet::E).  Without it the solve builds the field of the
   *  straddling half cell, the copy removes it every cycle, and the Gauss
   *  residual left behind spreads inward.  With OutflowLayerFields = solved
   *  nothing is copied, and the layer field is that of a neutral layer. */
  void zeroOutflowCharge(cudaSolverType* rhoh, cudaStream_t stream) const;

  /** The plan, read by the unit tests: the slot count of set s, the rule of
   *  (slot, face), the ghost points it fills, and the face node planes of
   *  the moment fold. */
  int slots(GhostSet s) const { return set(s).slots; }
  const GhostRuleDev& rule(GhostSet s, int slot, int f) const {
    return set(s).rule[slot][f];
  }
  const FaceSlabs& shell(GhostSet s) const { return set(s).shell; }
  const FaceSlabs& momentPlanes() const { return momentPlanes_; }

  /** The conductor walls, through whose image the node->cell stencils of
   *  the field solve read the face planes: the cell charge rho_c and the
   *  operator's div(mu E). */
  const ImageWalls& imageWalls() const { return imageWalls_; }

private:
  struct SetPlan {
    bool cells = false, particle = false;
    int slots = 1;
    GhostRuleDev rule[kGhostBatchMax][kFaces]; // per slot (host copy)
    const GhostRuleDev* d_rule = nullptr;      // the same, on the device
    FaceSlabs shell;
  };
  struct LayerList {
    /** At most one operation per face; a case wall has two per y face. */
    static constexpr int kMaxOps = kFaces;
    bool cells = false;
    int n = 0;
    LayerOp op[kMaxOps];
  };

  FieldBoundarySpec spec_;
  bool planned_ = false;
  SetPlan sets_[static_cast<int>(GhostSet::Count)];
  GhostRuleDev* d_rules_ = nullptr; // the rule tables of all sets
  LayerList layers_[static_cast<int>(LayerSet::Count)];
  RowFaces rows_;                         // conductor and inflow rows
  FaceSlabs momentPlanes_;                // particle-physical face node planes
  ImageWalls imageWalls_;                 // the wall cells of the conductors
  cudaSolverType caseProfile_[2][4] = {}; // the case-wall profiles per side
  GPUFieldArray3 wall_[kFaces];           // 2 W, conductor faces

  /** A ghost set or a layer list of the plan (abort unless it is built). */
  const SetPlan& set(GhostSet s) const {
    requirePlan("a ghost set");
    return sets_[static_cast<int>(s)];
  }
  const LayerList& list(LayerSet s) const {
    requirePlan("a layer set");
    return layers_[static_cast<int>(s)];
  }
  /** Abort unless the plan is built (what: what was used). */
  void requirePlan(const char* what) const;
};

#endif // GPU_FIELD_BOUNDARY_CUH
