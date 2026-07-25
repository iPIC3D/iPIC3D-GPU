#include "RestartMeshMetadata.h"

#include "Collective.h"
#include "Grid3DCU.h"
#include "VCtopology3D.h"
#include "debug.h"

RestartMeshMetadata makeCurrentRestartMeshMetadata(const Collective* col,
                                                   const VCtopology3D* vct,
                                                   const Grid3DCU* grid,
                                                   int ns) {
  if (!col || !vct || !grid) {
    eprintf("ERROR: cannot build restart mesh metadata from null input");
  }

  RestartMeshMetadata metadata;
  metadata.valid = true;

  metadata.xlen = vct->getXLEN();
  metadata.ylen = vct->getYLEN();
  metadata.zlen = vct->getZLEN();
  metadata.nranks = vct->getNprocs();

  metadata.nxc = col->getNxc();
  metadata.nyc = col->getNyc();
  metadata.nzc = col->getNzc();
  metadata.ns = ns;

  metadata.lx = col->getLx();
  metadata.ly = col->getLy();
  metadata.lz = col->getLz();
  metadata.dx = grid->getDX();
  metadata.dy = grid->getDY();
  metadata.dz = grid->getDZ();

  metadata.periodicX = vct->getPERIODICX();
  metadata.periodicY = vct->getPERIODICY();
  metadata.periodicZ = vct->getPERIODICZ();
  metadata.periodicParticleX = vct->getPERIODICX_P();
  metadata.periodicParticleY = vct->getPERIODICY_P();
  metadata.periodicParticleZ = vct->getPERIODICZ_P();

  return metadata;
}
