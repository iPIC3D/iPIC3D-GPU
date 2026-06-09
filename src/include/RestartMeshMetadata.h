#ifndef RESTART_MESH_METADATA_H
#define RESTART_MESH_METADATA_H

class Collective;
class VCtopology3D;
class Grid3DCU;

struct RestartMeshMetadata {
    bool valid = false;

    int xlen = 1;
    int ylen = 1;
    int zlen = 1;
    int nranks = 1;

    int nxc = 0;
    int nyc = 0;
    int nzc = 0;
    int ns = 0;

    double lx = 0.0;
    double ly = 0.0;
    double lz = 0.0;
    double dx = 0.0;
    double dy = 0.0;
    double dz = 0.0;

    bool periodicX = false;
    bool periodicY = false;
    bool periodicZ = false;
    bool periodicParticleX = false;
    bool periodicParticleY = false;
    bool periodicParticleZ = false;

    bool fieldsStoreActiveNodesOnly = true;
    bool particlesStoreActiveCellsOnly = true;
    bool particlesSortedByActiveCell = true;
};

RestartMeshMetadata makeCurrentRestartMeshMetadata(
    const Collective* col,
    const VCtopology3D* vct,
    const Grid3DCU* grid,
    int ns);

#endif // RESTART_MESH_METADATA_H
