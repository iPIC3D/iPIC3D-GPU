// =========================================================================
//  FieldBoundary.cpp - the field boundary conditions: the spec of this rank
//  (FieldBoundary.h) and the GPU plan derived from it (GPUFieldBoundary.cuh).
//
//  Everything the solver does at a boundary is decided here, once: the ghost
//  rule of every array on every face, the special rows of the Maxwell system
//  on the face nodes, and the ordered layer operations of the open faces and
//  the case walls.  A ghost set is a sequence of passes whose rules compose
//  at edges in pass order (GhostRuleDev::rank).
// =========================================================================

#include "FieldBoundary.h"

#include "Collective.h"
#include "GPUFieldBoundary.cuh"
#include "VCtopology3D.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <mpi.h>
#include <string>

// =========================================================================
//  The spec
// =========================================================================

FieldBoundarySpec FieldBoundarySpec::build(const Collective& col,
                                           const VirtualTopology3D& vct,
                                           const int cells[3],
                                           const double E0[3], double yStart,
                                           double dy) {
  FieldBoundarySpec s;
  const int neighbour[kFaces] = {
      vct.getXleft_neighbor(), vct.getXright_neighbor(),
      vct.getYleft_neighbor(), vct.getYright_neighbor(),
      vct.getZleft_neighbor(), vct.getZright_neighbor()};
  const int neighbourP[kFaces] = {
      vct.getXleft_neighbor_P(), vct.getXright_neighbor_P(),
      vct.getYleft_neighbor_P(), vct.getYright_neighbor_P(),
      vct.getZleft_neighbor_P(), vct.getZright_neighbor_P()};
  const int bcEM[kFaces] = {col.getBcEMfaceXleft(), col.getBcEMfaceXright(),
                            col.getBcEMfaceYleft(), col.getBcEMfaceYright(),
                            col.getBcEMfaceZleft(), col.getBcEMfaceZright()};
  const int bcP[kFaces] = {col.getBcPfaceXleft(), col.getBcPfaceXright(),
                           col.getBcPfaceYleft(), col.getBcPfaceYright(),
                           col.getBcPfaceZleft(), col.getBcPfaceZright()};
  for (int f = 0; f < kFaces; f++) {
    FaceSpec& face = s.face[f];
    face.bcEM = bcEM[f];
    face.bcP = bcP[f];
    face.particlePhysical = neighbourP[f] == MPI_PROC_NULL;
    face.kind = neighbour[f] != MPI_PROC_NULL ? FaceKind::Neighbour
                : bcEM[f] == 0                ? FaceKind::Conductor
                : bcEM[f] == 2                ? FaceKind::Open
                                              : FaceKind::Mirror;
  }
  const bool periodic[3] = {vct.getPERIODICX(), vct.getPERIODICY(),
                            vct.getPERIODICZ()};
  const int self = vct.getCartesian_rank();
  for (int a = 0; a < 3; a++) {
    s.cells[a] = cells[a];
    s.nodes[a] = cells[a] + 1;
    s.periodic[a] = periodic[a];
    s.flat[a] = cells[a] - 2 == 1 && neighbour[2 * a] == self &&
                neighbour[2 * a + 1] == self;
  }

  s.sal = col.getYes_sal() != 0;
  s.salLayers = col.getN_layers_sal();
  s.salTangentialB = col.getSal_tangential_B() != 0;
  s.inflowRows = col.getApplyInflowBcsEImage() != 0;
  for (int c = 0; c < 3; c++)
    s.E0[c] = E0[c];
  s.B0[0] = col.getB0x();
  s.B0[1] = col.getB0y();
  s.B0[2] = col.getB0z();
  s.outflowXright = s.face[FACE_XR].open() &&
                    s.face[FACE_XR].bcP == BCparticles::OPEN_OUTFLOW;
  s.outflowLayers = s.outflowXright
                        ? std::max(BCparticles::kOpenFaceLayers, s.salLayers)
                        : s.salLayers;
  s.outflowCopyLayers =
      s.outflowXright && col.getOutflowLayerSolved() ? 0 : s.outflowLayers;
  for (int r = 0; r < 4; r++) {
    const int nb = neighbour[FACE_YL + r];
    s.outflowRows[r] = nb == MPI_PROC_NULL ? GHOST_ROW_KEEP
                       : nb == self        ? GHOST_ROW_PERIODIC
                                           : GHOST_ROW_ADJACENT;
  }

  const std::string& simCase = col.getCase();
  // the case's B forced on the wall layers only with ForceWallB
  const bool gem = simCase == "GEM" || simCase == "GEMnoPert" ||
                   simCase == "GEMDoubleHarris";
  s.caseWall = !col.getForceWallB()     ? CaseWall::None
               : gem                    ? CaseWall::Gem
               : simCase == "ForceFree" ? CaseWall::ForceFree
                                        : CaseWall::None;
  s.yStart = yStart;
  s.dy = dy;
  s.halfLy = col.getLy() / 2.0;
  s.delta = col.getDelta();

  // The open-face layer operations index their layer count directly
  // (n_layers_sal, or outflowLayers on X-right): it has to fit the local
  // extent, with one untouched reference layer beyond a copy
  auto validate = [&](bool active, int layers, int extent, int reserved,
                      const char* field, int f) {
    static const char* const kFaceName[kFaces] = {
        "X-left", "X-right", "Y-left", "Y-right", "Z-left", "Z-right"};
    const int minLayers = s.sal ? 1 : 0, maxLayers = extent - reserved;
    if (!active || (layers >= minLayers && layers <= maxLayers))
      return;
    std::cerr << "ERROR: invalid boundary layer count " << layers
              << " (n_layers_sal=" << s.salLayers << ") for " << field << " "
              << kFaceName[f] << " open boundary on rank " << self
              << "; valid local range is [" << minLayers << "," << maxLayers
              << "] for extent " << extent << std::endl;
    MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
  };
  for (int f = 0; f < kFaces; f++) {
    const int a = faceAxis(f);
    // E: the inflow faces blend toward or set E0, the X-right outflow face
    // copies a reference plane
    if (f == FACE_XR)
      validate(s.outflowXright, s.outflowLayers, s.nodes[a], 2, "electric", f);
    else
      validate(s.inflow(f), s.salLayers, s.nodes[a], 1, "electric", f);
    // B, only for extents greater than 10 (openBLayers)
    validate(s.openBLayers(f), f == FACE_XR ? s.outflowLayers : s.salLayers,
             s.cells[a], s.openBCopy(f) ? 2 : 1, "magnetic", f);
  }
  return s;
}

// =========================================================================
//  The physics table: ghost rules
// =========================================================================

namespace {

constexpr GhostRuleDev kKeep{};
GhostRuleDev ruleOf(int kind) {
  GhostRuleDev r;
  r.kind = kind;
  return r;
}
GhostRuleDev constRule(cudaSolverType value) {
  GhostRuleDev r;
  r.kind = GHOST_CONST;
  r.value = value;
  return r;
}
GhostRuleDev copyRule(int layer) {
  GhostRuleDev r;
  r.kind = GHOST_COPY;
  r.copy = layer;
  return r;
}
/** The ghost of the conductor image, odd or even (imageOdd): a node array
 *  about the face node (its odd components are 0 there), a cell array about
 *  the face. */
GhostRuleDev imageNode(bool odd) {
  return ruleOf(odd ? GHOST_ODD_NODE : GHOST_EVEN_NODE);
}
GhostRuleDev imageCell(bool odd) {
  return ruleOf(odd ? GHOST_ODD_CELL : GHOST_EVEN_CELL);
}

/** The GEM and ForceFree wall profiles on side s (0 low, 1 high): {B0x
 *  tanh(a0), B0z / cosh(a0), B0z / cosh(a1), B0z / cosh(a2)}, a_m the
 *  profile argument at the centre of the m-th cell from the ghost cell. */
void caseWallProfile(const FieldBoundarySpec& S, int side,
                     cudaSolverType out[4]) {
  const int nyc = S.cells[1];
  double a[3];
  for (int m = 0; m < 3; m++) {
    const int j = side == 0 ? m : nyc - 1 - m;
    a[m] = (S.yStart + (j - 0.5) * S.dy - S.halfLy) / S.delta;
  }
  out[0] = S.B0[0] * std::tanh(a[0]);
  for (int m = 0; m < 3; m++)
    out[1 + m] = S.B0[2] / std::cosh(a[m]);
}

/** The n arrays p as the batch of one launch (at most kGhostBatchMax). */
GhostBatch batchOf(cudaSolverType* const* p, int n) {
  if (n > kGhostBatchMax) {
    std::cerr << "ERROR: FieldBoundary: a launch takes at most "
              << kGhostBatchMax << " arrays, got " << n << "\n";
    MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
  }
  GhostBatch b;
  b.n = n;
  for (int i = 0; i < n; i++)
    b.p[i] = p[i];
  return b;
}

/** E on a physical face (component c, face axis n): a perfect conductor is
 *  the image of E in the wall, E_n even and E_t odd about the face node (E_t
 *  = 0 there); elsewhere Collective's codes, E_n 0 and E_t a copy of the
 *  face node. */
GhostRuleDev eNodeRule(const FaceSpec& face, int c, int n) {
  if (face.conductor())
    return imageNode(imageOdd(Image::Polar, c, n));
  return ruleOf(c == n ? GHOST_ZERO : GHOST_EVEN_CELL);
}

/** Node B: on a conductor the image of B in the wall, B_t even and B_n odd
 *  about the face value (frozen there); elsewhere Collective's codes, B_n a
 *  copy of the face node and B_t 0. */
GhostRuleDev bNodeRule(const FaceSpec& face, int c, int n) {
  if (face.conductor())
    return imageNode(imageOdd(Image::Axial, c, n));
  return ruleOf(c == n ? GHOST_EVEN_CELL : GHOST_ZERO);
}

/** Cell B, Collective's codes: on a conductor B_n 0 (replaced by the wall
 *  rule) and B_t a copy of the first cell; elsewhere B_n a copy, B_t 0. */
GhostRuleDev bCellCode(const FaceSpec& face, int c, int n) {
  if (face.conductor())
    return ruleOf(c == n ? GHOST_ZERO : GHOST_EVEN_CELL);
  return ruleOf(c == n ? GHOST_EVEN_CELL : GHOST_ZERO);
}

} // namespace

// =========================================================================
//  The plan
// =========================================================================

FieldBoundary::~FieldBoundary() { releaseDevice(); }

void FieldBoundary::releaseDevice() {
  if (d_rules_)
    cudaErrChk(cudaFree(d_rules_));
  d_rules_ = nullptr;
  PinnedVector<GhostRuleDev>().swap(rules_);
  for (SetPlan& P : sets_) {
    P.rule = nullptr;
    P.d_rule = nullptr;
  }
  for (GPUFieldArray3& w : wall_)
    w.free();
  planned_ = false;
}

void FieldBoundary::requirePlan(const char* what) const {
  if (planned_)
    return;
  std::cerr << "ERROR: FieldBoundary: " << what
            << " used before buildPlan (or after releaseDevice)\n";
  MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
}

void FieldBoundary::buildPlan() {
  if (planned_)
    return;
  const FieldBoundarySpec& S = spec_;
  const int* cells = S.cells;
  const int* nodes = S.nodes;

  // ---- conductor wall values (2 W per cell of the face's ghost plane) ----
  for (int f = 0; f < kFaces; f++)
    if (S.face[f].conductor()) {
      const int a = faceAxis(f);
      wall_[f] =
          GPUFieldArray3(1, cells[a == 0 ? 1 : 0], cells[a == 2 ? 1 : 2]);
    }

  // ---- case-wall profiles ----
  if (S.caseWall != CaseWall::None)
    for (int side = 0; side < 2; side++)
      caseWallProfile(S, side, caseProfile_[side]);

  // ---- ghost sets ----
  // The rules of all sets in one host table, a part of it per set.
  const size_t rulesPerSet = (size_t)kGhostBatchMax * kFaces;
  rules_.assign(std::size(sets_) * rulesPerSet, kKeep);
  for (size_t s = 0; s < std::size(sets_); s++)
    sets_[s].rule = rules_.data() + s * rulesPerSet;
  // A set is a sequence of passes: a pass gives the rule of (slot, face),
  // GHOST_KEEP for none, and the last rule of a face wins; its rank, the
  // index of its pass, orders the composition at edges
  // (GhostRuleDev::rank).  The passes are a fold over the lambdas, each of
  // its own type: resolved and inlined at compile time.
  auto define = [&](GhostSet id, bool onCells, bool particle, int slots,
                    const auto&... passes) {
    SetPlan& P = sets_[static_cast<int>(id)];
    P.cells = onCells;
    P.particle = particle;
    P.slots = slots;
    bool active[kFaces] = {};
    for (int slot = 0; slot < slots; slot++)
      for (int f = 0; f < kFaces; f++) {
        GhostRuleDev r = kKeep;
        int rank = 0;
        const auto take = [&](const GhostRuleDev& p) {
          if (p.kind != GHOST_KEEP) {
            r = p;
            r.rank = rank;
          }
          rank++;
        };
        (take(passes(slot, f)), ...);
        P.rule[slot * kFaces + f] = r;
        active[f] = active[f] || r.kind != GHOST_KEEP;
      }
    P.shell = ghostShell(onCells ? cells : nodes, active);
  };
  // the faces a rule may act on: physical in the set's topology
  auto physical = [&](int f) { return S.face[f].physical(); };
  auto physicalP = [&](int f) { return S.face[f].particlePhysical; };
  auto conductor = [&](int f) { return S.face[f].conductor(); };

  // E at start-up, E(n+theta) and the smoothed E after the solve, E^n after
  // the div(E) correction
  define(GhostSet::ENode, false, false, 3, [&](int c, int f) {
    return physical(f) ? eNodeRule(S.face[f], c, faceAxis(f)) : kKeep;
  });
  // the smoothing passes of E: every component even about the face node
  define(GhostSet::ENodeSmooth, false, false, 1, [&](int, int f) {
    return physical(f) ? ruleOf(GHOST_EVEN_NODE) : kKeep;
  });
  define(GhostSet::BNode, false, false, 3, [&](int c, int f) {
    return physical(f) ? bNodeRule(S.face[f], c, faceAxis(f)) : kKeep;
  });
  define(GhostSet::BNodeConductor, false, false, 3, [&](int c, int f) {
    return conductor(f) ? bNodeRule(S.face[f], c, faceAxis(f)) : kKeep;
  });

  // Cell B.  The passes: Collective's codes, the ghost layer of the open
  // faces, the case walls, the conductor wall (B_n odd about W).
  const auto codes = [&](int c, int f) {
    return physical(f) ? bCellCode(S.face[f], c, faceAxis(f)) : kKeep;
  };
  // the ghost layer of the open-face B layers (LayerSet::B): the absorbing
  // blend's weight is 0 there (B0; B_n kept with salTangentialB), an inflow
  // face without it holds B0, the others copy the reference layer
  const auto openGhosts = [&](int c, int f) {
    if (!S.openBLayers(f))
      return kKeep;
    if (S.openBCopy(f))
      return copyRule((f == FACE_XR ? S.outflowCopyLayers : S.salLayers) + 1);
    if (S.sal && S.salTangentialB && c == faceAxis(f))
      return kKeep;
    return constRule(S.B0[c]);
  };
  // the case walls on the y faces: the wall profile in the ghost cells (on
  // a conductor wall the wall rule of B_y comes after it: BCell's last
  // pass, or BCellWall with a case wall)
  const auto caseGhosts = [&](CaseWall which) {
    return [&, which](int c, int f) {
      if (S.caseWall != which || faceAxis(f) != 1 || !physical(f))
        return kKeep;
      const cudaSolverType* p = caseProfile_[faceHigh(f) ? 1 : 0];
      if (c == 1)
        return constRule(S.B0[1]);
      if (c == 0)
        return constRule(p[0]);
      return constRule(which == CaseWall::Gem ? S.B0[2] : p[1]);
    };
  };
  const auto wallGhosts = [&](int c, int f) {
    if (!conductor(f) || c != faceAxis(f))
      return kKeep;
    GhostRuleDev r = ruleOf(GHOST_WALL);
    r.wall = wall_[f].devPtr();
    return r;
  };
  // With a case wall its interior layers (LayerSet::CaseCells) come between
  // the case ghosts and the conductor wall, which then is BCellWall.
  const bool caseWall = S.caseWall != CaseWall::None;
  const auto wallUnlessCase = [&](int c, int f) {
    return caseWall ? kKeep : wallGhosts(c, f);
  };
  const auto wallIfCase = [&](int c, int f) {
    return caseWall ? wallGhosts(c, f) : kKeep;
  };
  define(GhostSet::BCell, true, false, 3, codes, openGhosts,
         caseGhosts(CaseWall::Gem), caseGhosts(CaseWall::ForceFree),
         wallUnlessCase);
  define(GhostSet::BCellWall, true, false, 3, wallIfCase);
  define(GhostSet::BCellStart, true, false, 3, codes, wallGhosts);
  // before the source curl: the ForceFree ghosts, the open faces, the GEM
  // ghosts (no codes; the wall is BCellWall)
  define(GhostSet::BCellSource, true, false, 3, caseGhosts(CaseWall::ForceFree),
         openGhosts, caseGhosts(CaseWall::Gem));

  // The operator's 10 cell arrays: the 9 gradients grad_d E_c (a = 3 c + d)
  // and div(mu E) (a = 9).  On a conductor their image: grad_d of the polar
  // E_c (the derivative along the normal reverses the parity of E_c) and the
  // charge-like div(mu E), odd; the face rows are then the interior rows of
  // the image, whose Gauss term sums by parts up to the wall.  Other physical
  // faces: gradients 0, div(mu E) a copy.  An X-right outflow face keeps the
  // ghost plane its stencils computed.
  const auto opCode = [&](int a, int f) {
    if (f == FACE_XR && S.outflowXright)
      return kKeep;
    const int n = faceAxis(f);
    if (conductor(f))
      return imageCell(a == 9
                           ? imageOdd(Image::Scalar, 0, n)
                           : imageOdd(Image::Polar, a / 3, n) != (a % 3 == n));
    return ruleOf(a == 9 ? GHOST_EVEN_CELL : GHOST_ZERO);
  };
  define(GhostSet::OpCells, true, false, 10,
         [&](int a, int f) { return physical(f) ? opCode(a, f) : kKeep; });
  // The local operator: no exchange, every ghost cell 0 (the neighbours'
  // too), then the conductor faces as in OpCells
  define(
      GhostSet::OpCellsLocal, true, false, 10,
      [&](int, int f) {
        return f == FACE_XR && S.outflowXright ? kKeep : ruleOf(GHOST_ZERO);
      },
      [&](int a, int f) { return conductor(f) ? opCode(a, f) : kKeep; });

  // The derived moments the field solve reads (particle topology): the
  // conductor image on a conductor, even on the other particle-physical
  // faces (cells about the face, nodes about the face node).
  //  - The charge (rho_c, rhohat): odd.
  //  - The pressure term -dt/2 div(Pi) of Jhat (cells): a current, its
  //    tangential components odd.  The ghost enters only the interpolation
  //    to the face node, where the image gives 0, as for Jhat_t.
  //  - Jhat (smoothing): a current, as E.  Even about the face node
  //    elsewhere, the rule of the E smoothing, so that the smoothing
  //    conserves the total and is self-adjoint in the lumped inner
  //    product.  With Jhat_t = 0 on a conductor face (zeroTangential) its odd
  //    ghost cancels node 2 in every pass, so Jhat_t stays 0 there as E_t
  //    does: the Jhat smoothing is the adjoint of the E smoothing.
  const auto defineMoment = [&](GhostSet id, bool onCells, int slots, Image k) {
    define(id, onCells, true, slots, [&](int c, int f) {
      if (!conductor(f) && !physicalP(f))
        return kKeep;
      const bool odd = conductor(f) && imageOdd(k, c, faceAxis(f));
      return onCells ? imageCell(odd) : imageNode(odd);
    });
  };
  defineMoment(GhostSet::CellChargeP, true, 1, Image::Scalar);
  defineMoment(GhostSet::CellCurrentP, true, 3, Image::Polar);
  defineMoment(GhostSet::NodeCurrentP, false, 3, Image::Polar);
  // The cleaning potentials.  phi (cells) is charge-like: the conductor
  // image is odd, so phi = 0 on the wall (an equipotential) and the
  // correction gradC2N phi has no tangential part there; 0 half a cell
  // outside the other physical faces.  psi (nodes) has no rule: the physical
  // face nodes are identity rows of the div(B) operator, which reads no
  // ghost beyond them; the set gives its exchange the geometry.
  define(GhostSet::Phi, true, false, 1, [&](int, int f) {
    return conductor(f)  ? imageCell(imageOdd(Image::Scalar, 0, faceAxis(f)))
           : physical(f) ? ruleOf(GHOST_ZERO)
                         : kKeep;
  });
  define(GhostSet::Psi, false, false, 1);

  // the rule tables on the device
  const size_t ruleBytes = rules_.size() * sizeof(GhostRuleDev);
  cudaErrChk(cudaMalloc(&d_rules_, ruleBytes));
  cudaErrChk(
      cudaMemcpy(d_rules_, rules_.data(), ruleBytes, cudaMemcpyHostToDevice));
  for (SetPlan& P : sets_)
    P.d_rule = d_rules_ + (P.rule - rules_.data());

  // ---- rows of the Maxwell system ----
  {
    bool active[kFaces];
    for (int f = 0; f < kFaces; f++) {
      rows_.kind[f] = S.face[f].conductor()         ? ROW_CONDUCTOR
                      : S.inflowRows && S.inflow(f) ? ROW_INFLOW
                                                    : ROW_NONE;
      active[f] = rows_.kind[f] != ROW_NONE;
    }
    for (int c = 0; c < 3; c++)
      rows_.E0[c] = S.E0[c];
    rows_.planes = nodePlanes(nodes, active);
  }
  {
    bool active[kFaces];
    for (int f = 0; f < kFaces; f++)
      active[f] = S.face[f].particlePhysical;
    momentPlanes_ = nodePlanes(nodes, active);
  }
  // the cells next to the conductors: cell 1 and cell n-2
  for (int a = 0; a < 3; a++) {
    imageWalls_.lo[a] = conductor(2 * a) ? 1 : -1;
    imageWalls_.hi[a] = conductor(2 * a + 1) ? cells[a] - 2 : -1;
  }

  // ---- layer lists ----
  const int N = S.salLayers;
  const cudaSolverType invN = N > 0 ? 1.0 / (double)N : 1.0;
  auto range = [&](int f, int extent, int layers, LayerOp& op) {
    op.axis = faceAxis(f);
    op.first = faceHigh(f) ? extent - layers - 1 : 0;
    op.last = faceHigh(f) ? extent - 1 : layers;
  };
  auto blend = [&](int f, int extent, const double* target, int mask) {
    LayerOp op;
    op.kind = LAYER_BLEND;
    range(f, extent, N, op);
    for (int c = 0; c < 3; c++)
      op.target[c] = target[c];
    op.mask = mask;
    op.invLayers = invN;
    op.ascending = faceHigh(f) ? 0 : 1;
    return op;
  };
  auto set3 = [&](int f, int extent, int layers, const double* target) {
    LayerOp op;
    op.kind = LAYER_SET;
    range(f, extent, layers, op);
    for (int c = 0; c < 3; c++)
      op.target[c] = target[c];
    return op;
  };
  auto copy = [&](int f, int extent, int layers) {
    LayerOp op;
    op.kind = LAYER_COPY;
    range(f, extent, layers, op);
    op.ref = faceHigh(f) ? extent - 2 - layers : layers + 1;
    return op;
  };
  // B and CaseCells are on the cells, the other sets on the nodes
  for (int id = 0; id < static_cast<int>(LayerSet::Count); id++) {
    layers_[id].n = 0;
    layers_[id].cells = id == static_cast<int>(LayerSet::B) ||
                        id == static_cast<int>(LayerSet::CaseCells);
  }
  auto push = [&](LayerSet id, const LayerOp& op) {
    LayerList& L = layers_[static_cast<int>(id)];
    if (L.n >= LayerList::kMaxOps) {
      std::cerr << "ERROR: FieldBoundary: too many layer operations\n";
      MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }
    L.op[L.n++] = op;
  };

  // E after the solve: the inflow faces hold E0 (or blend toward it) on the
  // node layers 0..N, X-left first, then the X-right outflow copy, then the
  // Y and Z inflow faces
  {
    auto inflowE = [&](int f) {
      if (!S.inflow(f))
        return;
      const int extent = nodes[faceAxis(f)];
      push(LayerSet::E,
           S.sal ? blend(f, extent, S.E0, 7) : set3(f, extent, N, S.E0));
    };
    inflowE(FACE_XL);
    if (S.outflowXright)
      push(LayerSet::E, copy(FACE_XR, nodes[0], S.outflowCopyLayers));
    for (int f = FACE_YL; f < kFaces; f++)
      inflowE(f);
  }
  // cell B after Faraday's law, on the open faces with layers (openBLayers)
  for (int f = 0; f < kFaces; f++) {
    if (!S.openBLayers(f))
      continue;
    const int a = faceAxis(f), extent = cells[a];
    const int mask = S.salTangentialB ? 7 & ~(1 << a) : 7;
    if (S.openBCopy(f))
      push(LayerSet::B,
           copy(f, extent, f == FACE_XR ? S.outflowCopyLayers : N));
    else if (S.sal)
      push(LayerSet::B, blend(f, extent, S.B0, mask));
    else
      push(LayerSet::B, set3(f, extent, N, S.B0));
  }
  // the case walls: the interior wall layers (the ghosts are BCell rules);
  // the y-right side first, as the case kernels did
  if (S.caseWall != CaseWall::None)
    for (int f : {FACE_YR, FACE_YL}) {
      if (!physical(f))
        continue;
      const cudaSolverType* p = caseProfile_[faceHigh(f) ? 1 : 0];
      const int ny = cells[1], nyn = nodes[1];
      const int j1 = faceHigh(f) ? ny - 2 : 1, j2 = faceHigh(f) ? ny - 3 : 2;
      auto layer = [&](int first, int last, cudaSolverType x, cudaSolverType z,
                       int mask) {
        LayerOp op;
        op.kind = LAYER_SET;
        op.axis = 1;
        op.first = first;
        op.last = last;
        op.target[0] = x;
        op.target[2] = z;
        op.mask = mask;
        return op;
      };
      if (S.caseWall == CaseWall::Gem) {
        push(LayerSet::CaseCells,
             layer(std::min(j1, j2), std::max(j1, j2), p[0], S.B0[2], 5));
        // node B: the wall planes j0..j2 (B_x, B_z) and the ghost's B_y
        // (on a conductor wall BNodeConductor sets its image after it)
        const int n0 = faceHigh(f) ? nyn - 1 : 0,
                  n2 = faceHigh(f) ? nyn - 3 : 2;
        push(LayerSet::CaseNodes,
             layer(std::min(n0, n2), std::max(n0, n2), p[0], S.B0[2], 5));
        LayerOp by = layer(n0, n0, 0, 0, 2);
        by.target[1] = S.B0[1];
        push(LayerSet::CaseNodes, by);
      } else {
        push(LayerSet::CaseCells, layer(j1, j1, 0, p[2], 4));
        push(LayerSet::CaseCells, layer(j2, j2, 0, p[3], 4));
      }
    }
  // the X-right outflow face: the operator's ghost node plane (a copy of the
  // face plane) and the moments on the outflow layer
  if (S.outflowXright) {
    push(LayerSet::OutflowNodes, copy(FACE_XR, nodes[0], 0));
    push(LayerSet::OutflowMoments, copy(FACE_XR, nodes[0], S.outflowLayers));
  }
  planned_ = true;
}

void FieldBoundary::initWall(const cudaSolverType* const Bn[3],
                             cudaStream_t stream) {
  requirePlan("initWall");
  for (int f = 0; f < kFaces; f++)
    if (spec_.face[f].conductor())
      gpuConductorWallSum(wall_[f].devPtr(), Bn[faceAxis(f)], spec_.nodes, f,
                          stream);
}

void FieldBoundary::fill(GhostSet s, cudaSolverType* const* p, int n,
                         cudaStream_t stream) const {
  const SetPlan& P = set(s);
  if (P.shell.count() == 0 || n == 0)
    return;
  GhostBatch b = batchOf(p, n);
  b.slots = P.slots;
  b.rules = P.d_rule;
  gpuFillGhosts(b, P.shell, stream);
}

void FieldBoundary::imageRows(cudaSolverType* im,
                              const cudaSolverType* const v[3],
                              cudaStream_t stream) const {
  requirePlan("imageRows");
  gpuBoundaryImageRows(rows_, im, v, stream);
}

void FieldBoundary::rowValues(cudaSolverType* const E[3],
                              cudaStream_t stream) const {
  requirePlan("rowValues");
  gpuBoundaryFaceValues(rows_, E, true, stream);
}

void FieldBoundary::zeroTangential(cudaSolverType* const v[3],
                                   cudaStream_t stream) const {
  requirePlan("zeroTangential");
  if (spec_.anyConductor()) // else nothing to write
    gpuBoundaryFaceValues(rows_, v, false, stream);
}

void FieldBoundary::layers(LayerSet s, cudaSolverType* X, cudaSolverType* Y,
                           cudaSolverType* Z, cudaStream_t stream) const {
  const LayerList& L = list(s);
  const int* n = L.cells ? spec_.cells : spec_.nodes;
  for (int i = 0; i < L.n; i++)
    gpuLayerOp(L.op[i], X, Y, Z, n, stream);
}

void FieldBoundary::sourceCellB(cudaSolverType* const Bc[3],
                                cudaStream_t stream) const {
  if (spec_.caseWall == CaseWall::ForceFree)
    layers(LayerSet::CaseCells, Bc[0], Bc[1], Bc[2], stream);
  fill(GhostSet::BCellSource, Bc, 3, stream);
  if (spec_.caseWall == CaseWall::Gem)
    layers(LayerSet::CaseCells, Bc[0], Bc[1], Bc[2], stream);
  fill(GhostSet::BCellWall, Bc, 3, stream);
}

void FieldBoundary::caseWallCells(cudaSolverType* const Bc[3],
                                  cudaStream_t stream) const {
  layers(LayerSet::CaseCells, Bc[0], Bc[1], Bc[2], stream);
  fill(GhostSet::BCellWall, Bc, 3, stream);
}

void FieldBoundary::caseWallNodes(cudaSolverType* const Bn[3],
                                  cudaStream_t stream) const {
  if (spec_.caseWall != CaseWall::Gem)
    return;
  layers(LayerSet::CaseNodes, Bn[0], Bn[1], Bn[2], stream);
  fill(GhostSet::BNodeConductor, Bn, 3, stream);
}

void FieldBoundary::foldMoments(cudaSolverType* const* d_ptrs, int nFields,
                                cudaStream_t stream) const {
  requirePlan("foldMoments");
  gpuFoldMoments(momentPlanes_, d_ptrs, nFields, stream);
}

void FieldBoundary::outflowGhostRows(cudaSolverType* const* p, int n,
                                     cudaStream_t stream) const {
  if (!spec_.outflowXright)
    return;
  gpuGhostRowsXright(batchOf(p, n), spec_.cells[0], spec_.cells[1],
                     spec_.cells[2], spec_.outflowRows, stream);
}

void FieldBoundary::neutralizeOutflow(cudaSolverType* rhons,
                                      const cudaSolverType* qom, int ns,
                                      cudaStream_t stream) const {
  const LayerList& L = list(LayerSet::OutflowMoments);
  if (L.n == 0)
    return;
  const LayerOp& op = L.op[0];
  gpuNeutralizeLayersX(rhons, qom, ns, op.first, op.last, op.ref,
                       spec_.nodes[0], spec_.nodes[1], spec_.nodes[2], stream);
}

void FieldBoundary::zeroOutflowCharge(cudaSolverType* rhoh,
                                      cudaStream_t stream) const {
  if (!spec_.outflowXright)
    return;
  // the cells nxc-1-L .. nxc-1, from the one straddling x_s (node
  // nxn-2-L): the last L+1 x planes of the [i][j][k] cell array
  const int* n = spec_.cells;
  const int first = n[0] - 1 - spec_.outflowLayers;
  const size_t plane = (size_t)n[1] * n[2];
  cudaErrChk(cudaMemsetAsync(rhoh + first * plane, 0,
                             (n[0] - first) * plane * sizeof(cudaSolverType),
                             stream));
}
