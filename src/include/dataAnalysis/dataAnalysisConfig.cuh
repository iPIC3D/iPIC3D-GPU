#ifndef _DATA_ANALYSIS_CONFIG_H_
#define _DATA_ANALYSIS_CONFIG_H_

#include <string>
#include "cudaTypeDef.cuh"

namespace DAConfig {


// General configuration
inline constexpr bool DATA_ANALYSIS_ENABLED = false;
inline constexpr bool VELOCITY_HISTOGRAM_ENABLE = DATA_ANALYSIS_ENABLED && true;
inline constexpr bool GMM_ENABLE = VELOCITY_HISTOGRAM_ENABLE && true;

inline const std::string DATA_ANALYSIS_OUTPUT_DIR = "./";
inline constexpr int DATA_ANALYSIS_EVERY_CYCLE = 50; 

// Histogram configuration
inline constexpr bool HISTOGRAM_OUTPUT = VELOCITY_HISTOGRAM_ENABLE && true;
inline const std::string HISTOGRAM_OUTPUT_DIR = DATA_ANALYSIS_OUTPUT_DIR + "velocityHistogram/";
// For 2D
inline constexpr int VELOCITY_HISTOGRAM2D_RES = 200; // must be multiply of VELOCITY_HISTOGRAM_TILE
inline constexpr int VELOCITY_HISTOGRAM2D_TILE = 100;
// For 3D
inline constexpr int VELOCITY_HISTOGRAM3D_RES_1 = 100; 
inline constexpr int VELOCITY_HISTOGRAM3D_RES_2 = 100;
inline constexpr int VELOCITY_HISTOGRAM3D_RES_3 = 100;
inline constexpr int VELOCITY_HISTOGRAM3D_SIZE = VELOCITY_HISTOGRAM3D_RES_1 * VELOCITY_HISTOGRAM3D_RES_2 * VELOCITY_HISTOGRAM3D_RES_3;
inline constexpr int VELOCITY_HISTOGRAM3D_TILE = 20;

inline constexpr bool HISTOGRAM_FIXED_RANGE = true; // edit the range in velocityHistogram::getRange --> moved here
inline constexpr cudaCommonType MIN_VELOCITY_HIST_E = -0.2;
inline constexpr cudaCommonType MAX_VELOCITY_HIST_E = 0.2;
inline constexpr cudaCommonType MIN_VELOCITY_HIST_I = -0.09;
inline constexpr cudaCommonType MAX_VELOCITY_HIST_I = 0.09;

// GMM configuration

inline constexpr bool GMM_OUTPUT = GMM_ENABLE && true;
inline const std::string GMM_OUTPUT_DIR = DATA_ANALYSIS_OUTPUT_DIR + "velocityGMM/";
inline constexpr int DATA_DIM_GMM = 3; // only works with DATA_DIM = 3 now
inline constexpr int NUM_COMPONENT_GMM = 10; // number of components used in GMM - array with length NUM_ANALYSIS_GMM
inline constexpr int MAX_ITERATION_GMM = 100;
inline constexpr cudaCommonType  THRESHOLD_CONVERGENCE_GMM = 1e-6;
inline constexpr bool START_WITH_LAST_PARAMETERS_GMM = true; // start GMM iteration with output paramters of last GMM step as initial parameters
inline constexpr bool NORMALIZE_DATA_FOR_GMM = true;    // normalize data before GMM such that velocities are in range -1;1 --> the original velocity domain is assumed to be symmetric wrt 0
inline constexpr bool CHECK_COVMATRIX_GMM = true;   // safety check on the cov-matrix --> ensures variances > EPS_COVMATRIX_GMM
inline constexpr cudaCommonType TOL_COVMATRIX_GMM = 1e-9;  // tol used to ensure cov-matrix determinant > 0
inline constexpr cudaCommonType EPS_COVMATRIX_GMM = 1e-4;   // minimum value that elements on the cov-matrix main diagonal can assume (assume data normalized in range -1,1)
inline constexpr bool PRUNE_COMPONENTS_GMM = true; // remove GMM components with weight < PRUNE_THRESHOLD_GMM --> remove one componet at a time
inline constexpr cudaCommonType PRUNE_THRESHOLD_GMM = 0.005;

constexpr bool checkDAEnabled(){
    static_assert(!GMM_ENABLE || VELOCITY_HISTOGRAM_ENABLE, "GMM requires velocity histogram to be enabled");
    static_assert(!VELOCITY_HISTOGRAM_ENABLE || (VELOCITY_HISTOGRAM2D_RES % VELOCITY_HISTOGRAM2D_TILE) == 0, "VELOCITY_HISTOGRAM_RES must be multiply of VELOCITY_HISTOGRAM_TILE");

    static_assert(!VELOCITY_HISTOGRAM_ENABLE || (VELOCITY_HISTOGRAM3D_RES_1 % VELOCITY_HISTOGRAM3D_TILE) == 0, "Adjust histogram resolution to multiply of tile ...");
    static_assert(!VELOCITY_HISTOGRAM_ENABLE || (VELOCITY_HISTOGRAM3D_RES_2 % VELOCITY_HISTOGRAM3D_TILE) == 0, "Adjust histogram resolution to multiply of tile ...");
    static_assert(!VELOCITY_HISTOGRAM_ENABLE || (VELOCITY_HISTOGRAM3D_RES_3 % VELOCITY_HISTOGRAM3D_TILE) == 0, "Adjust histogram resolution to multiply of tile ...");

    return true;
}

inline auto discard = checkDAEnabled();

inline std::string DA_2D_PLANE_NAME[3] = {"uv", "vw", "uw"};

} // namespace DAConfig


#endif