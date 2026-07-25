#ifndef RESTART_REMAP_PLAN_H
#define RESTART_REMAP_PLAN_H

#include "RestartMeshMetadata.h"

#include <array>
#include <vector>

struct RestartInterval {
  int begin = 0;
  int end = 0;

  int length() const { return end - begin; }
  bool empty() const { return end <= begin; }
};

struct RestartBox3D {
  RestartInterval x;
  RestartInterval y;
  RestartInterval z;

  bool empty() const { return x.empty() || y.empty() || z.empty(); }
  int nx() const { return x.length(); }
  int ny() const { return y.length(); }
  int nz() const { return z.length(); }
};

struct RestartRankBox {
  int rank = 0;
  std::array<int, 3> coord = {{0, 0, 0}};
  RestartBox3D activeCells;
  RestartBox3D activeNodes;
  RestartBox3D ownedActiveNodes;
};

struct RestartNodeCopy {
  int sourceRank = 0;
  RestartBox3D global;
  RestartBox3D sourceLocal;
  RestartBox3D destinationLocal;
};

struct RestartCellCopy {
  int sourceRank = 0;
  RestartBox3D global;
  RestartBox3D sourceLocal;
  RestartBox3D destinationLocal;
};

class RestartRemapPlan {
public:
  RestartRemapPlan(const RestartMeshMetadata& source,
                   const RestartMeshMetadata& destination, int destinationRank);

  const RestartMeshMetadata& sourceMesh() const { return source_; }
  const RestartMeshMetadata& destinationMesh() const { return destination_; }
  const RestartRankBox& destinationBox() const { return destinationBox_; }

  const std::vector<RestartNodeCopy>& nodeCopies() const { return nodeCopies_; }

  const std::vector<RestartCellCopy>& cellCopies() const { return cellCopies_; }

  int destinationRank() const { return destinationRank_; }

private:
  RestartMeshMetadata source_;
  RestartMeshMetadata destination_;
  int destinationRank_ = 0;
  RestartRankBox destinationBox_;
  std::vector<RestartNodeCopy> nodeCopies_;
  std::vector<RestartCellCopy> cellCopies_;
};

RestartRankBox restartRankBoxFor(const RestartMeshMetadata& mesh, int rank);

#endif // RESTART_REMAP_PLAN_H
