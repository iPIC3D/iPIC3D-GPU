#ifndef _MACROCELL_SPECTRA_H_
#define _MACROCELL_SPECTRA_H_

#include <string>
#include <vector>
#include <memory>

#include "cudaTypeDef.cuh"
#include "dataAnalysisConfig.cuh"

// Forward declarations to keep this header lightweight.
class particleArrayCUDA;
class grid3DCUDA;
class VCtopology3D;

namespace macrocellSpectra {

using namespace DAConfig;

// Histogram element type for per-macrocell (v_par, v_perp) bins.
// Kept as `cudaTypeSingle` for parity with `velocityHistogram::histogramTypeOut`.
using macrocellHistType = cudaTypeSingle;

// ======= Host-side macrocell tiling =======

/**
 * @brief One axis-aligned tile in the interior cell index space.
 *
 * Coordinates are 0-based interior cell indices, i.e. they exclude the
 * one-cell guard layer that surrounds every MPI subdomain in iPIC3D.
 */
struct MacrocellRange {
    int start;   ///< first interior cell index covered by the macrocell
    int size;    ///< number of cells (>= 1)
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
 * Halo cells are never included.
 *
 * The class is purely host-side: device-resident copies of the per-axis
 * tilings are owned by `macrocellSpectra2D`.
 */
struct MacrocellPartition {
    int Cx = 0, Cy = 0, Cz = 0;             ///< requested macrocell size
    int Nx_int = 0, Ny_int = 0, Nz_int = 0; ///< interior cell extents
    int Mx = 0, My = 0, Mz = 0;             ///< macrocells per axis
    int M  = 0;                             ///< total macrocells = Mx*My*Mz

    /// Per-axis tilings (sizes Mx, My, Mz).
    std::vector<MacrocellRange> rangeX, rangeY, rangeZ;

    /// Subdomain origin in the global (whole-domain) interior cell index space.
    int globalOffsetX = 0, globalOffsetY = 0, globalOffsetZ = 0;

    /**
     * @brief Build the tiling.
     * @return false if any size is non-positive or any C* > N*_int.
     */
    bool build(int Nx_int_, int Ny_int_, int Nz_int_,
               int Cx_, int Cy_, int Cz_,
               int globalOffX, int globalOffY, int globalOffZ);

    /// Linear macrocell id used everywhere (x fastest).
    int linearId(int mx, int my, int mz) const {
        return (mz * My + my) * Mx + mx;
    }

    void unflatten(int m, int& mx, int& my, int& mz) const {
        mx =  m % Mx;
        my = (m / Mx) % My;
        mz =  m / (Mx * My);
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
 * The ranges follow the existing fixed-range convention used by
 * `velocityHistogram3D`:
 *   - vpar  in [-vmax_s, +vmax_s]
 *   - vperp in [0,       +vmax_s]
 * with vmax_s = MAX_VELOCITY_HIST_E for species 0/2 and
 *      vmax_s = MAX_VELOCITY_HIST_I otherwise.
 */
class macrocellSpectra2D {
public:
    /// Number of bins per macrocell.
    static constexpr int Nb = MACROCELL_BINS_VPAR * MACROCELL_BINS_VPERP;

    macrocellSpectra2D(const MacrocellPartition& part);

    /// Async memset of the device histogram buffer to zero.
    void reset(cudaStream_t stream);

    /**
     * @brief Launch the fused interpolate-and-bin kernel for one species.
     *
     * The borrowed pointers must come from `c_Solver`:
     *   - `pclsHostPtr->getX()/Y()/Z()/U()/V()/W()` device pointers
     *   - `fieldForPclsCUDA`: packed mover field buffer (B in components 0..2)
     *   - `gridDevicePtr`:    device-resident `grid3DCUDA`
     *   - `cellStartOffsetsCUDA`, `cellCountsCUDA`: from the per-species
     *      `CellSorter`. Particles must already be cell-sorted.
     */
    void launch(particleArrayCUDA* pclsHostPtr,
                cudaCommonType*    fieldForPclsCUDA,   // cudaFieldType*
                const grid3DCUDA*  gridDevicePtr,
                const int*         cellStartOffsetsCUDA,
                const int*         cellCountsCUDA,
                int                species,
                cudaStream_t       stream);

    /**
     * @brief Copy histograms to host and write per-macrocell binary + JSON.
     *
     * Files are written under
     *   <subdomainDir>/species_<species>/cycle_<NNNNNN>/
     *       mc_<mx>_<my>_<mz>.bin
     *       mc_<mx>_<my>_<mz>.json
     */
    void writeToFile(const std::string&        subdomainDir,
                     int                       subdomainRank,
                     const VCtopology3D*  vct,
                     int                       species,
                     int                       cycle,
                     cudaStream_t              stream);

    /// Write the constant per-subdomain partition.json once at startup.
    void writePartitionMetadata(const std::string&       subdomainDir,
                                int                      subdomainRank,
                                const VCtopology3D* vct) const;

    const MacrocellPartition& partition() const { return part_; }

    ~macrocellSpectra2D();

    macrocellSpectra2D(const macrocellSpectra2D&)            = delete;
    macrocellSpectra2D& operator=(const macrocellSpectra2D&) = delete;

private:
    MacrocellPartition part_;

    // Device-resident axis tilings: 2 ints per macrocell (start, size).
    int* dRangeX_ = nullptr;
    int* dRangeY_ = nullptr;
    int* dRangeZ_ = nullptr;

    // Flat histogram buffer (device + pinned host mirror).
    macrocellHistType* dHist_ = nullptr;
    macrocellHistType* hHist_ = nullptr;
    size_t numFloats_ = 0;   ///< == part_.M * Nb

    // Returns vmax for the given species using the fixed-range convention.
    static cudaCommonType vmaxForSpecies(int species);
};

} // namespace macrocellSpectra


// ======= Device-side kernel declarations =======

__global__ void macrocellSpectraKernel(
    // Particle SoA
    const cudaCommonType* __restrict__ x,
    const cudaCommonType* __restrict__ y,
    const cudaCommonType* __restrict__ z,
    const cudaCommonType* __restrict__ u,
    const cudaCommonType* __restrict__ v,
    const cudaCommonType* __restrict__ w,
    const cudaCommonType* __restrict__ q,
    // Cell sort
    const int* __restrict__ cellStartOffsets,
    const int* __restrict__ cellCounts,
    // Macrocell axis tilings (2 ints per element: start, size)
    const int* __restrict__ rangeX,
    const int* __restrict__ rangeY,
    const int* __restrict__ rangeZ,
    int Mx, int My, int Mz,
    // Field & grid
    const cudaCommonType* __restrict__ fieldForPcls,  // packed cudaFieldType
    const grid3DCUDA*     __restrict__ grid,
    // Histogram parameters
    macrocellSpectra::macrocellHistType* __restrict__ histOut, // [M * Nb]
    int   binsVpar, int binsVperp,
    cudaCommonType vmax,
    cudaCommonType bMin);


#endif // _MACROCELL_SPECTRA_H_
