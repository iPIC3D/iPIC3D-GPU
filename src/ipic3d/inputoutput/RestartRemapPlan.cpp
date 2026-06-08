#include "RestartRemapPlan.h"

#include "debug.h"

#include <algorithm>

namespace {

RestartInterval makeInterval(int begin, int end)
{
    RestartInterval interval;
    interval.begin = begin;
    interval.end = end;
    return interval;
}

RestartInterval intersectInterval(RestartInterval a, RestartInterval b)
{
    return makeInterval(std::max(a.begin, b.begin),
                        std::min(a.end, b.end));
}

RestartBox3D intersectBox(RestartBox3D a, RestartBox3D b)
{
    RestartBox3D box;
    box.x = intersectInterval(a.x, b.x);
    box.y = intersectInterval(a.y, b.y);
    box.z = intersectInterval(a.z, b.z);
    return box;
}

RestartInterval localInterval(RestartInterval global,
                              RestartInterval owner)
{
    return makeInterval(global.begin - owner.begin,
                        global.end - owner.begin);
}

RestartBox3D localBox(RestartBox3D global, RestartBox3D owner)
{
    RestartBox3D box;
    box.x = localInterval(global.x, owner.x);
    box.y = localInterval(global.y, owner.y);
    box.z = localInterval(global.z, owner.z);
    return box;
}

bool isPowerOfTwo(int value)
{
    return value > 0 && (value & (value - 1)) == 0;
}

bool isPowerOfTwoMultiple(int destination, int source)
{
    if (source <= 0 || destination <= 0) return false;
    if (destination % source != 0) return false;
    return isPowerOfTwo(destination / source);
}

int ceilDiv(int numerator, int denominator)
{
    return (numerator + denominator - 1) / denominator;
}

void validateMesh(const RestartMeshMetadata& mesh, const char* label)
{
    if (!mesh.valid) {
        eprintf("ERROR: restart %s mesh metadata is missing or invalid",
                label);
    }
    if (mesh.nranks != mesh.xlen * mesh.ylen * mesh.zlen) {
        eprintf("ERROR: restart %s mesh topology is inconsistent", label);
    }
    if (mesh.nxc < mesh.xlen || mesh.nyc < mesh.ylen ||
        mesh.nzc < mesh.zlen) {
        eprintf("ERROR: restart %s mesh has fewer cells than ranks in a direction",
                label);
    }
}

void validateCompatibleMeshes(const RestartMeshMetadata& source,
                              const RestartMeshMetadata& destination)
{
    validateMesh(source, "source");
    validateMesh(destination, "destination");

    if (source.nxc != destination.nxc ||
        source.nyc != destination.nyc ||
        source.nzc != destination.nzc) {
        eprintf("ERROR: restart remap requires identical global cell counts");
    }

    if (!isPowerOfTwoMultiple(destination.xlen, source.xlen) ||
        !isPowerOfTwoMultiple(destination.ylen, source.ylen) ||
        !isPowerOfTwoMultiple(destination.zlen, source.zlen)) {
        eprintf("ERROR: restart remap requires destination topology "
                "XLEN/YLEN/ZLEN to be source topology times 2^p in each direction");
    }

    if (!source.fieldsStoreActiveNodesOnly ||
        !source.particlesStoreActiveCellsOnly ||
        !source.particlesSortedByActiveCell) {
        eprintf("ERROR: restart remap requires active-node fields and "
                "active-cell-sorted particle restart data");
    }
}

RestartInterval rankCellInterval(int globalCells, int ranks, int coord)
{
    const int regular = ceilDiv(globalCells, ranks);
    const int begin = coord * regular;
    const int count = (coord == ranks - 1)
        ? globalCells - regular * (ranks - 1)
        : regular;
    return makeInterval(begin, begin + count);
}

std::array<int, 3> coordFromRank(const RestartMeshMetadata& mesh, int rank)
{
    std::array<int, 3> coord = {{0, 0, 0}};
    coord[0] = rank / (mesh.ylen * mesh.zlen);
    const int rem = rank - coord[0] * mesh.ylen * mesh.zlen;
    coord[1] = rem / mesh.zlen;
    coord[2] = rem - coord[1] * mesh.zlen;
    return coord;
}

} // namespace

int restartRankFromCoord(const RestartMeshMetadata& mesh,
                         int x, int y, int z)
{
    return (x * mesh.ylen + y) * mesh.zlen + z;
}

RestartRankBox restartRankBoxFor(const RestartMeshMetadata& mesh, int rank)
{
    if (rank < 0 || rank >= mesh.nranks) {
        eprintf("ERROR: restart rank %d is outside source topology", rank);
    }

    RestartRankBox box;
    box.rank = rank;
    box.coord = coordFromRank(mesh, rank);

    box.activeCells.x = rankCellInterval(mesh.nxc, mesh.xlen, box.coord[0]);
    box.activeCells.y = rankCellInterval(mesh.nyc, mesh.ylen, box.coord[1]);
    box.activeCells.z = rankCellInterval(mesh.nzc, mesh.zlen, box.coord[2]);

    box.activeNodes.x =
        makeInterval(box.activeCells.x.begin, box.activeCells.x.end + 1);
    box.activeNodes.y =
        makeInterval(box.activeCells.y.begin, box.activeCells.y.end + 1);
    box.activeNodes.z =
        makeInterval(box.activeCells.z.begin, box.activeCells.z.end + 1);

    box.ownedActiveNodes = box.activeNodes;
    if (box.coord[0] != mesh.xlen - 1) box.ownedActiveNodes.x.end -= 1;
    if (box.coord[1] != mesh.ylen - 1) box.ownedActiveNodes.y.end -= 1;
    if (box.coord[2] != mesh.zlen - 1) box.ownedActiveNodes.z.end -= 1;

    return box;
}

RestartRemapPlan::RestartRemapPlan(const RestartMeshMetadata& source,
                                   const RestartMeshMetadata& destination,
                                   int destinationRank)
    : source_(source),
      destination_(destination),
      destinationRank_(destinationRank)
{
    validateCompatibleMeshes(source_, destination_);

    destinationBox_ = restartRankBoxFor(destination_, destinationRank_);

    nodeCopies_.reserve(8);
    cellCopies_.reserve(8);

    for (int sourceRank = 0; sourceRank < source_.nranks; ++sourceRank) {
        const RestartRankBox sourceBox =
            restartRankBoxFor(source_, sourceRank);

        const RestartBox3D nodeGlobal =
            intersectBox(sourceBox.ownedActiveNodes,
                         destinationBox_.activeNodes);
        if (!nodeGlobal.empty()) {
            RestartNodeCopy copy;
            copy.sourceRank = sourceRank;
            copy.global = nodeGlobal;
            copy.sourceLocal = localBox(nodeGlobal, sourceBox.activeNodes);
            copy.destinationLocal =
                localBox(nodeGlobal, destinationBox_.activeNodes);
            nodeCopies_.push_back(copy);
        }

        const RestartBox3D cellGlobal =
            intersectBox(sourceBox.activeCells,
                         destinationBox_.activeCells);
        if (!cellGlobal.empty()) {
            RestartCellCopy copy;
            copy.sourceRank = sourceRank;
            copy.global = cellGlobal;
            copy.sourceLocal = localBox(cellGlobal, sourceBox.activeCells);
            copy.destinationLocal =
                localBox(cellGlobal, destinationBox_.activeCells);
            cellCopies_.push_back(copy);
        }
    }

    if (nodeCopies_.empty()) {
        eprintf("ERROR: restart remap generated no active-node reads");
    }
    if (cellCopies_.empty()) {
        eprintf("ERROR: restart remap generated no active-cell reads");
    }
}
