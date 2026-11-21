#ifndef MACROCELL_PARTITION_CUH
#define MACROCELL_PARTITION_CUH

#include <algorithm>
#include <string>

#include "cudaTypeDef.cuh"

/**
 * @file macrocellPartition.cuh
 * @brief The macrocells of a rank, of the macrocell spectra and of the
 * macrocell velocity histograms (each analysis on its own tiling): the
 * tiling of the cells, the cells of a macrocell as the kernels walk them (in
 * slices, by several blocks, when the macrocells are few), and the JSON lines
 * that describe them.
 */
namespace macrocell {

/** Blocks per multiprocessor that a macrocell kernel launches at least, its
 *  macrocells cut in slices if needed (slicesFor). */
inline constexpr int kSliceBlocksPerSM = 4;

/** The cells of a macrocell, as interior indices: the first along each axis
 *  and the number along each axis. */
struct MacrocellBox {
  int start[3], length[3];

  __host__ __device__ int cells() const {
    return length[0] * length[1] * length[2];
  }
};

/** The cells of a macrocell along an axis, as interior indices: the
 *  requested size, the remaining cells for the last one (the tiling of
 *  MacrocellPartition). */
__host__ __device__ inline void macrocellAxisRange(int macrocell,
                                                   int requestedSize,
                                                   int interiorCells,
                                                   int& start, int& length) {
  start = macrocell * requestedSize;
  length = start + requestedSize <= interiorCells ? requestedSize
                                                  : interiorCells - start;
}

/** The macrocells of a rank as the kernels walk them (by value). */
struct MacrocellGeometry {
  int interior[3]; ///< interior cells per axis
  int size[3];     ///< requested macrocell size per axis
  int count[3];    ///< macrocells per axis

  /// The cells of macrocell m, of linear id m = (mz * My + my) * Mx + mx (x
  /// fastest, the layout of the kernels and of the files).
  __host__ __device__ MacrocellBox box(int m) const {
    const int index[3] = {m % count[0], m / count[0] % count[1],
                          m / (count[0] * count[1])};
    MacrocellBox b;
    for (int a = 0; a < 3; a++)
      macrocellAxisRange(index[a], size[a], interior[a], b.start[a],
                         b.length[a]);
    return b;
  }
};

/** Slice `slice` of `slices` of n cells, in order: [first, end), the slices
 *  one after the other, their sizes differing by one at most (some empty
 *  with fewer cells than slices). */
__host__ __device__ inline void sliceOf(int n, int slice, int slices,
                                        int& first, int& end) {
  if (slices == 1) { // no 64-bit division (slow on GPUs)
    first = 0;
    end = n;
    return;
  }
  first = (int)((long long)n * slice / slices);
  end = (int)((long long)n * (slice + 1) / slices);
}

/**
 * @brief The tiling of the interior cells of a rank in macrocells.
 *
 * Given the interior cell extent `(Nx_int, Ny_int, Nz_int)` and the requested
 * macrocell size `(Cx, Cy, Cz)`, each axis is cut in macrocells of the
 * requested size; when the extent is not a multiple of it, the last macrocell
 * along that axis has the remaining cells, fewer: on a rank with fewer cells
 * than the size along an axis (a truncated last rank), the only one.
 *
 * The tiles cover the interior cells only. The cell sort can put a particle
 * that lies exactly on a face of the rank in a ghost cell: it is in no
 * macrocell, and the macrocell analyses leave it out.
 */
struct MacrocellPartition {
  int Cx = 0, Cy = 0, Cz = 0;             ///< requested macrocell size
  int Nx_int = 0, Ny_int = 0, Nz_int = 0; ///< interior cell extents
  int Mx = 0, My = 0, Mz = 0;             ///< macrocells per axis
  int M = 0;                              ///< total macrocells = Mx*My*Mz

  /// Subdomain origin in the global (whole-domain) interior cell index space.
  int globalOffsetX = 0, globalOffsetY = 0, globalOffsetZ = 0;

  /**
   * @brief Build the tiling.
   * @return false if a size or an extent is not positive.
   */
  bool build(int Nx_int_, int Ny_int_, int Nz_int_, int Cx_, int Cy_, int Cz_,
             int globalOffX, int globalOffY, int globalOffZ);

  /// The macrocells as the kernels walk them.
  MacrocellGeometry geometry() const {
    return {{Nx_int, Ny_int, Nz_int}, {Cx, Cy, Cz}, {Mx, My, Mz}};
  }

  /// The cells of the largest macrocell.
  int largestCells() const {
    return std::min(Cx, Nx_int) * std::min(Cy, Ny_int) * std::min(Cz, Nz_int);
  }
};

/**
 * @brief The slices of each macrocell of a kernel of blocksPerSlice blocks
 * per slice of a macrocell: enough for kSliceBlocksPerSM blocks per
 * multiprocessor of the current device, at most the cells of the largest
 * macrocell (one slice when the macrocells alone are enough).
 */
int slicesFor(int macrocells, int blocksPerSlice, int largestCells);

/**
 * @brief For each row along x of the cells of slice `slice` of `slices` of
 * macrocell m (its linear id; its cells x fastest, sliceOf), visit(begin,
 * end) with the particles [begin, end) of the cells of the slice in the row,
 * among the particles sorted by cell: cellStart and cellCount of the cell
 * sort, on the guarded grid of interior + 2 cells per axis. The cells of a
 * row are consecutive, so are their particles. The same calls for every
 * thread of a block; one slice: the rows of the macrocell.
 */
template <typename Visit>
__device__ inline void
forEachMacrocellRow(const MacrocellGeometry& g, int m, int slice, int slices,
                    const int* cellStart, const int* cellCount, Visit visit) {
  const MacrocellBox b = g.box(m);
  int c, end;
  sliceOf(b.cells(), slice, slices, c, end);
  const int nxc = g.interior[0] + 2, nyc = g.interior[1] + 2;
  // the first cell of the slice: x in row (iy, iz) of the macrocell
  const int row = c / b.length[0];
  int x = c - row * b.length[0], iy = row % b.length[1], iz = row / b.length[1];
  while (c < end) {
    // cells x to x + n - 1 of the row, the guarded index of the first and
    // the last of them
    const int n = b.length[0] - x < end - c ? b.length[0] - x : end - c;
    const int first = (b.start[0] + x + 1) + (b.start[1] + iy + 1) * nxc +
                      (b.start[2] + iz + 1) * nxc * nyc;
    const int last = first + n - 1;
    visit(cellStart[first], cellStart[last] + cellCount[last]);
    c += n;
    x = 0; // the next row
    if (++iy == b.length[1]) {
      iy = 0;
      iz++;
    }
  }
}

/** The JSON members of the regions of the analysis records of a rank:
 *  "regions": "rank", or "regions": "macrocells" and the tiling (part); the
 *  records of a file need the same (RecordFile layout). */
std::string regionsDescription(const MacrocellPartition* part);

} // namespace macrocell

#endif // MACROCELL_PARTITION_CUH
