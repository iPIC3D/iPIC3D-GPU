#ifndef HOST_FIELD_LOOPS_H
#define HOST_FIELD_LOOPS_H

/**
 * @file HostFieldLoops.h
 * @brief OpenMP loop from the local nodes of the host field arrays, which are
 *        [i][j][k] with k contiguous, into an output buffer ordered with x
 *        fastest: the order of the VTK STRUCTURED_POINTS files.
 */

#include <algorithm>
#include <cstddef>

/** Fill out[((iz*ly + iy)*lx + ix)*NC + c], 0 <= ix < lx (and so on), with
 *  store(v[c]), where sample(ix + 1, iy + 1, iz + 1, v) writes the NC values
 *  at that local node (index 0 is the ghost layer).
 *
 *  The source is contiguous along k, the output along x.  z therefore runs
 *  innermost in tiles of kNodeTileZ nodes: a thread reads kNodeTileZ
 *  consecutive k of one (i, j), a cache line of doubles, and writes
 *  kNodeTileZ sequential output rows.  The threads share the (z tile, y)
 *  pairs. */
template <int NC, typename T, typename Sample, typename Store>
void fillNodesXFastest(T* out, int lx, int ly, int lz, Sample&& sample,
                       Store&& store) {
  constexpr int kNodeTileZ = 8;
  const int tilesZ = (lz + kNodeTileZ - 1) / kNodeTileZ;
#pragma omp parallel for collapse(2) schedule(static)
  for (int tz = 0; tz < tilesZ; tz++)
    for (int iy = 0; iy < ly; iy++) {
      const int z0 = tz * kNodeTileZ;
      const int z1 = std::min(lz, z0 + kNodeTileZ);
      for (int ix = 0; ix < lx; ix++)
        for (int iz = z0; iz < z1; iz++) {
          double v[NC];
          sample(ix + 1, iy + 1, iz + 1, v);
          T* const o = out + (((size_t)iz * ly + iy) * lx + ix) * NC;
          for (int c = 0; c < NC; c++)
            o[c] = store(v[c]);
        }
    }
}

#endif // HOST_FIELD_LOOPS_H
