#ifndef DATA_ANALYSIS_CONFIG_CUH
#define DATA_ANALYSIS_CONFIG_CUH

#include "BenchmarkMode.h"
#include "cudaTypeDef.cuh"

#include <climits>

/**
 * @file dataAnalysisConfig.cuh
 * @brief Compile-time constants of the data analysis, which a build includes
 * with BENCHMARK_MODE < 2: whether it writes its results and the sizes that
 * fix GPU buffers and shared memory. Which analyses run, how often, their
 * ranges and the GMM settings are input-file options (DataAnalysisInput.h).
 */

namespace DAConfig {

/** The analyses write their results (BENCHMARK_MODE == 0). */
inline constexpr bool DATA_ANALYSIS_OUTPUT =
    BenchmarkConfig::DISK_OUTPUT_ENABLED;

/** Output subfolders, inside DataAnalysisOutputDir. */
inline constexpr const char* HISTOGRAM_SUBDIR = "velocityHistogram/";
inline constexpr const char* GMM_SUBDIR = "velocityGMM/";
inline constexpr const char* MACROCELL_SPECTRA_SUBDIR = "macrocellSpectra/";

/** Weight of a particle in the velocity histograms and spectra: |q| times
 *  this. */
inline constexpr double VELOCITY_WEIGHT_SCALE = 1e7;

/** Bins of the 3D velocity histogram along u, v, w. */
inline constexpr int VELOCITY_HISTOGRAM3D_RES_1 = 100;
inline constexpr int VELOCITY_HISTOGRAM3D_RES_2 = 100;
inline constexpr int VELOCITY_HISTOGRAM3D_RES_3 = 100;
inline constexpr int VELOCITY_HISTOGRAM3D_SIZE = VELOCITY_HISTOGRAM3D_RES_1 *
                                                 VELOCITY_HISTOGRAM3D_RES_2 *
                                                 VELOCITY_HISTOGRAM3D_RES_3;
/** Bins of the 3D velocity histogram of a macrocell along u, v, w
 *  (VelocityHistogramMacrocells): multiples of 16, the tile of the kernel. */
inline constexpr int MACROCELL_HISTOGRAM3D_RES_1 = 32;
inline constexpr int MACROCELL_HISTOGRAM3D_RES_2 = 32;
inline constexpr int MACROCELL_HISTOGRAM3D_RES_3 = 32;

/** Bins of the macrocell spectra along v_par and v_perp. */
inline constexpr int MACROCELL_BINS_VPAR = 64;
inline constexpr int MACROCELL_BINS_VPERP = 64;
/** Macrocells of a rank at most, of the macrocell histograms and of the
 *  spectra: an int indexes the bins of all of them (sparseBins::SparseBins);
 *  the histograms also take their bins in GPU memory, 4 bytes each. */
inline constexpr int MAX_HISTOGRAM_MACROCELLS =
    INT_MAX / (MACROCELL_HISTOGRAM3D_RES_1 * MACROCELL_HISTOGRAM3D_RES_2 *
               MACROCELL_HISTOGRAM3D_RES_3);
inline constexpr int MAX_SPECTRA_MACROCELLS =
    INT_MAX / (MACROCELL_BINS_VPAR * MACROCELL_BINS_VPERP);
/** A particle where |B| is at most this is left out of the spectra (v_par
 *  divides by |B|). */
inline constexpr cudaCommonType MACROCELL_BMIN = 1e-12;

/** Dimension of the GMM data (u, v, w); the analysis supports 3 only. */
inline constexpr int DATA_DIM_GMM = 3;

} // namespace DAConfig

#endif // DATA_ANALYSIS_CONFIG_CUH
