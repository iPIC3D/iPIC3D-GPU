#ifndef _DATA_ANALYSIS_CONFIG_H_
#define _DATA_ANALYSIS_CONFIG_H_

#include <string>
#include "cudaTypeDef.cuh"

#pragma once

// General configuration
inline constexpr bool DATA_ANALYSIS_ENABLED = true;
inline constexpr bool VELOCITY_HISTOGRAM_ENABLE = true;
inline constexpr bool GMM_ENABLE = true;

inline const std::string DATA_ANALYSIS_OUTPUT_DIR = "./";
inline constexpr int DATA_ANALYSIS_EVERY_CYCLE = 100; // 0 to disable

// Histogram configuration
inline constexpr int VELOCITY_HISTOGRAM_RES = 100; // must be multiply of VELOCITY_HISTOGRAM_TILE
inline constexpr int VELOCITY_HISTOGRAM_TILE = 100;
inline constexpr bool HISTOGRAM_OUTPUT = true;
inline const std::string HISTOGRAM_OUTPUT_DIR = DATA_ANALYSIS_OUTPUT_DIR + "velocityHistogram/";

inline constexpr bool HISTOGRAM_FIXED_RANGE = true; // edit the range in velocityHistogram::getRange --> moved here
inline constexpr cudaCommonType MIN_VELOCITY_HIST_E = -0.2;
inline constexpr cudaCommonType MAX_VELOCITY_HIST_E = 0.2;
inline constexpr cudaCommonType MIN_VELOCITY_HIST_I = -0.09;
inline constexpr cudaCommonType MAX_VELOCITY_HIST_I = 0.09;

inline constexpr bool HISTOGRAM_OUTPUT_3D = false; // the vtk file format, if false the 3 planes are on the same surface in paraview

// GMM configuration
inline constexpr bool GMM_OUTPUT = true;
inline constexpr bool OUTPUT_ALL_END_GMM = true;
inline const std::string GMM_OUTPUT_DIR = DATA_ANALYSIS_OUTPUT_DIR + "velocityGMM/";
inline constexpr int DATA_DIM_GMM = 2; // only works with DATA_DIM = 2 now
inline constexpr int NUM_ANALYSIS_GMM = 1; // number of GMM analysis
inline constexpr int NUM_COMPONENT_GMM[NUM_ANALYSIS_GMM] = {4}; // number of components used in GMM - array with length NUM_ANALYSIS_GMM
inline constexpr int TOTAL_COMPONENT_GMM = 4; // must be the sum of NUM_COMPONENT_GMM
inline constexpr int MAX_ITERATION_GMM = 100;
inline constexpr cudaCommonType  THRESHOLD_CONVERGENCE_GMM = 1e-6;
inline constexpr bool START_WITH_LAST_PARAMETERS_GMM = true; // start GMM iteration with output paramters of last GMM step as initial parameters
inline constexpr bool CHECK_COVMATRIX_GMM = true;  // safety check on the cov-matrix --> ensures variances > EPS_COVMATRIX_GMM
inline constexpr bool NORMALIZE_DATA_FOR_GMM = true; // normalize data before GMM such that velocities are in range -1;1
inline constexpr bool REMOVE_MEAN_GMM = false;  // remove mean from data before GMM
inline constexpr bool FILTER_WEIGHTS_GMM = false;  // filter GMM data removing data with low weight -> if weight * threshold < max(weight) --> weight = 0
inline constexpr int WEIGHTS_THRESHOLD_GMM = 100;  // threshold in filtering GMM data 
inline constexpr cudaCommonType TOLL_COVMATRIX_GMM = 1e-10;
inline constexpr cudaCommonType EPS_COVMATRIX_GMM = 1e-4;


template<int N>
struct checkSumNumComponentsGMM{
    enum {value = NUM_COMPONENT_GMM[N - 1] + checkSumNumComponentsGMM<N - 1>::value};
};
template <>
struct checkSumNumComponentsGMM<0>{
    enum {value = 0};
};

constexpr bool checkDAEnabled(){
    static_assert(!GMM_ENABLE || VELOCITY_HISTOGRAM_ENABLE, "GMM requires velocity histogram to be enabled");
    static_assert(!GMM_ENABLE || checkSumNumComponentsGMM<NUM_ANALYSIS_GMM>::value == TOTAL_COMPONENT_GMM, "Error in NUM_COMPONENT_GMM setup" );

    static_assert(!VELOCITY_HISTOGRAM_ENABLE || (VELOCITY_HISTOGRAM_RES % VELOCITY_HISTOGRAM_TILE) == 0, "VELOCITY_HISTOGRAM_RES must be multiply of VELOCITY_HISTOGRAM_TILE");

    return true;
}

inline auto discard = checkDAEnabled();



#endif