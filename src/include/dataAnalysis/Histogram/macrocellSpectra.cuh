#ifndef MACROCELL_SPECTRA_CUH
#define MACROCELL_SPECTRA_CUH

#include <memory>
#include <string>
#include <vector>

#include "cudaTypeDef.cuh"
#include "dataAnalysisConfig.cuh"

// Forward declarations to keep this header lightweight.
class particleArrayCUDA;
class grid3DCUDA;
class VCtopology3D;

namespace macrocellSpectra {

// Histogram element type for per-macrocell (v_par, v_perp) bins.
// Kept as `cudaTypeSingle` for parity with
// `velocityHistogram::histogramTypeOut`.
using macrocellHistType = cudaTypeSingle;

// ======= Host-side macrocell tiling =======

/**
 * @brief One axis-aligned tile in the interior cell index space.
 *
 * Coordinates are 0-based interior cell indices, i.e. they exclude the
 * one-cell guard layer that surrounds every MPI subdomain in iPIC3D.
 */
struct MacrocellRange {
  int start; ///< first interior cell index covered by the macrocell
  int size;  ///< number of cells (>= 1)
};

/**
 * @brief Build the per-axis tiling of an MPI subdomain into macrocells.
 *
 * Given an interior cell extent `(Nx_int, Ny_int, Nz_int)` and the requested
 * macrocell size `(Cx, Cy, Cz)`, the partition tiles each axis in macrocells
 * of size `Cx` (resp. `Cy`, `Cz`); when the extent is not divisible by the
 * macrocell size, the last macrocell along that axis carries the remainder
 * (its `size` is then `< Cx`).
 *
 * The tiles cover the interior cells; the kernel adds to the first and the
 * last macrocell of each axis the ghost cell next to it: a particle exactly
 * on the upper boundary of the rank has its index, one on the lower boundary
 * can round into it.
 *
 * The class is host-side. The JSON writer uses the full per-axis tiling, while
 * the CUDA kernel receives the scalar extents and requested macrocell sizes.
 */
struct MacrocellPartition {
  int Cx = 0, Cy = 0, Cz = 0;             ///< requested macrocell size
  int Nx_int = 0, Ny_int = 0, Nz_int = 0; ///< interior cell extents
  int Mx = 0, My = 0, Mz = 0;             ///< macrocells per axis
  int M = 0;                              ///< total macrocells = Mx*My*Mz

  /// Per-axis tilings (sizes Mx, My, Mz).
  std::vector<MacrocellRange> rangeX, rangeY, rangeZ;

  /// Subdomain origin in the global (whole-domain) interior cell index space.
  int globalOffsetX = 0, globalOffsetY = 0, globalOffsetZ = 0;

  /**
   * @brief Build the tiling.
   * @return false if any size is non-positive or any C* > N*_int.
   */
  bool build(int Nx_int_, int Ny_int_, int Nz_int_, int Cx_, int Cy_, int Cz_,
             int globalOffX, int globalOffY, int globalOffZ);

  /// Linear macrocell id (x fastest), the layout of the kernel and of the
  /// files. Not used currently.
  int linearId(int mx, int my, int mz) const {
    return (mz * My + my) * Mx + mx;
  }

  /// The inverse of linearId. Not used currently.
  void unflatten(int m, int& mx, int& my, int& mz) const {
    mx = m % Mx;
    my = (m / Mx) % My;
    mz = m / (Mx * My);
  }
};

// ======= Device-side flat 2D (v_par, v_perp) spectra accumulator =======

/**
 * @brief Per-species, per-macrocell (v_par, v_perp) histogram accumulator.
 *
 * Layout: a single flat float buffer of size
 *     M * (MACROCELL_BINS_VPERP * MACROCELL_BINS_VPAR)
 * with vperp as the outer index and vpar as the inner index inside one
 * macrocell slice. Bin metadata (range and resolution) is uniform across
 * macrocells for the active species.
 *
 * The ranges of species s are
 *   - vpar  in [-vmax_s, +vmax_s]
 *   - vperp in [0,       +vmax_s]
 * with vmax_s its VelocityHistogramMax.
 *
 * Output files (one per enabled species per MPI rank):
 *   <subdomainDir>/species_<S>.bin   — flat binary, one record per analysis
 *                                      cycle appended in order.
 *   <subdomainDir>/species_<S>.json  — fully self-contained metadata; the
 *                                      "records" array lists every cycle that
 *                                      has been successfully written.
 *
 * Binary record layout (record K covers cycle records[K]):
 *   [ M*Nb float32 ]
 *   macrocell m = (mz*My + my)*Mx + mx  at offset  m * Nb * 4  within the
 * record. Within each macrocell slice: row-major [vperp][vpar] (vperp outer,
 * vpar inner). Byte offset of macrocell m at record K: K * record_size_bytes +
 * m * bytes_per_macrocell
 */
class macrocellSpectra2D {
public:
  /// Number of bins per macrocell.
  static constexpr int Nb =
      DAConfig::MACROCELL_BINS_VPAR * DAConfig::MACROCELL_BINS_VPERP;
  static_assert(Nb * sizeof(macrocellHistType) <= 48 * 1024,
                "the bins of a macrocell fit the default shared memory");

  /**
   * @param part The macrocells of this rank.
   * @param vmax Velocity range of each species.
   */
  macrocellSpectra2D(const MacrocellPartition& part,
                     std::vector<cudaCommonType> vmax);

  /// Async memset of the device histogram buffer to zero.
  void reset(cudaStream_t stream);

  /**
   * @brief Launch the fused interpolate-and-bin kernel for one species.
   *
   * The borrowed pointers must come from `c_Solver`:
   *   - `pclsHostPtr->getX()/Y()/Z()/U()/V()/W()/Q()` device pointers
   *   - `fieldForPclsCUDA`: packed mover field buffer (B in components 0..2)
   *   - `gridDevicePtr`:    device-resident `grid3DCUDA`
   *   - `cellStartOffsetsCUDA`, `cellCountsCUDA`: from the per-species
   *      `CellSorter`. Particles must already be cell-sorted.
   */
  void launch(particleArrayCUDA* pclsHostPtr, cudaFieldType* fieldForPclsCUDA,
              const grid3DCUDA* gridDevicePtr, const int* cellStartOffsetsCUDA,
              const int* cellCountsCUDA, int species, cudaStream_t stream);

  /**
   * @brief Register a species' output file context.
   *
   * Must be called once per enabled species before the first writeToFile().
   *
   * Fresh start (isRestart == false):
   *   - The binary file is created/truncated.
   *   - The JSON is written with an empty "records" array.
   *
   * Restart (isRestart == true):
   *   - The existing JSON is read to restore the "records" list, without the
   *     cycles from firstCycle on, which the restart writes again. Files of
   *     other macrocells, bins or velocity range are kept as
   *     *_before_cycle<firstCycle>.*, and start again (the caller warns).
   *   - The binary file is truncated to the records kept (fewer if it is
   *     shorter), so the JSON always remains the authoritative source of
   *     truth.
   *
   * @param subdomainDir  Path to the per-rank subdomain directory (trailing /).
   * @param species       Species index (used as the file name suffix).
   * @param rank          MPI rank stored in the JSON metadata.
   * @param isRestart     true when continuing a previous simulation.
   * @param firstCycle    the first cycle of this run.
   * @param vct           VCT topology pointer; its coordinates are snapshotted
   *                      into the JSON and are not accessed after this call.
   * @return true when the records of a restart were kept aside (another
   *         layout or velocity range)
   */
  bool initSpeciesFile(const std::string& subdomainDir, int species, int rank,
                       bool isRestart, int firstCycle, const VCtopology3D* vct);

  /**
   * @brief D->H copy + append one record to the species binary file.
   *
   * The binary is opened in append mode. After a successful write the
   * in-memory records list is updated and the JSON is rewritten atomically
   * via a temp-file + rename to guarantee a consistent on-disk state even
   * across unexpected process termination.
   *
   * initSpeciesFile() must have been called for this species beforehand.
   *
   * @param species Species index (must match a prior initSpeciesFile call).
   * @param cycle   Simulation cycle being saved.
   * @param stream  CUDA stream used for the D->H copy.
   */
  void writeToFile(int species, int cycle, cudaStream_t stream);

  /// Not used currently.
  const MacrocellPartition& partition() const { return part_; }

  ~macrocellSpectra2D();

  macrocellSpectra2D(const macrocellSpectra2D&) = delete;
  macrocellSpectra2D& operator=(const macrocellSpectra2D&) = delete;

private:
  MacrocellPartition part_;
  std::vector<cudaCommonType> vmax_; ///< velocity range of each species

  // Flat histogram buffer (device + pinned host mirror).
  // Sized M * Nb elements; shared and reused across species sequentially.
  macrocellHistType* dHist_ = nullptr;
  macrocellHistType* hHist_ = nullptr;
  size_t numFloats_ = 0; ///< == part_.M * Nb

  // ======= Per-species output context =======

  /**
   * @brief Runtime state for one enabled species' output files.
   *
   * Populated by initSpeciesFile() and updated by writeToFile().
   */
  struct SpeciesFileContext {
    std::string binPath;      ///< path of the .bin file
    std::string jsonPath;     ///< path of the .json file
    std::vector<int> records; ///< simulation cycles written so far
    int species = -1;         ///< species index
    int rank = 0;             ///< MPI rank
    bool initialized = false;
    // Snapshot of VCT topology taken at initSpeciesFile() time.
    int cartesianRank = -1;
    int coords[3] = {0, 0, 0};
    int topology[3] = {0, 0, 0};
  };

  /// Indexed by species id; disabled species keep initialized=false.
  std::vector<SpeciesFileContext> speciesCtx_;

  /**
   * @brief Atomically rewrite the JSON for ctx (temp-file + rename).
   *
   * Called after every successful binary append so the JSON "records" array
   * always reflects the true content of the binary file.
   */
  void writeJson_(const SpeciesFileContext& ctx) const;

  size_t recordSizeBytes_() const;
  size_t bytesPerMacrocell_() const;

  SpeciesFileContext makeSpeciesContext_(const std::string& subdomainDir,
                                         int species, int rank,
                                         const VCtopology3D* vct) const;

  void createEmptyBinary_(const std::string& path) const;
  bool reconcileRestart_(SpeciesFileContext& ctx, int firstCycle) const;
  void appendBinaryRecord_(const SpeciesFileContext& ctx) const;

  /**
   * @brief Parse the "records" array from the content of a JSON written by
   * writeJson_.
   *
   * Returns an empty vector if the array is absent or malformed.
   */
  static std::vector<int> parseRecordsFromJson_(const std::string& content);
};

} // namespace macrocellSpectra

// ======= Device-side kernel declarations =======

__global__ void macrocellSpectraKernel(
    // Particle SoA
    const cudaCommonType* __restrict__ x, const cudaCommonType* __restrict__ y,
    const cudaCommonType* __restrict__ z, const cudaCommonType* __restrict__ u,
    const cudaCommonType* __restrict__ v, const cudaCommonType* __restrict__ w,
    const cudaCommonType* __restrict__ q,
    // Cell sort
    const int* __restrict__ cellStartOffsets,
    const int* __restrict__ cellCounts,
    // Interior cell extents and requested macrocell sizes
    int Nx_int, int Ny_int, int Nz_int, int Cx, int Cy, int Cz,
    // Field & grid
    const cudaFieldType* __restrict__ fieldForPcls,
    const grid3DCUDA* __restrict__ grid,
    // Histogram parameters
    macrocellSpectra::macrocellHistType* __restrict__ histOut, // [M * Nb]
    cudaCommonType vmax);

#endif // MACROCELL_SPECTRA_CUH
