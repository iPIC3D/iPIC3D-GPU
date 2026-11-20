#ifndef DATA_ANALYSIS_INPUT_H
#define DATA_ANALYSIS_INPUT_H

#include <string>
#include <vector>

/**
 * @file DataAnalysisInput.h
 * @brief The data-analysis options of the input file, read by Collective.
 *
 * Every DataAnalysisCycle cycles the analysis thread computes, per species,
 * the 3D velocity histogram (VelocityHistogram), its Gaussian-mixture fit
 * (VelocityGMM, the compressed form of the distribution) and the macrocell
 * (v_par, v_perp) spectra (VelocitySpectra). The member initialisers are the
 * defaults of the input file.
 */

/** Default fixed histogram range [-v, v] of a species with qom < 0. */
inline constexpr double kHistogramVmaxElectrons = 0.2;
/** Default fixed histogram range [-v, v] of the other species. */
inline constexpr double kHistogramVmaxIons = 0.09;

struct DataAnalysisInput {
  /** DataAnalysisCycle: analyse every this many cycles; 0 disables. */
  int cycle = 50;
  /** DataAnalysisOutputDir: parent folder of the analysis outputs (read
   *  with a '/' at the end). */
  std::string outputDir = "./";

  // ======= Velocity histograms =======
  /** VelocityHistogram: 3D (u, v, w) histogram of every species. */
  bool velocityHistogram = false;
  /** VelocityHistogramFixedRange: bin [VelocityHistogramMin,
   *  VelocityHistogramMax] on the three axes; otherwise the range of the
   *  velocities of the species at that cycle (velocityHistogram3D::init). */
  bool histogramFixedRange = true;
  /** VelocityHistogramMin / Max, per species (max > 0). The macrocell
   *  spectra bin v_par in [-max, max] and v_perp in [0, max]. */
  std::vector<double> histogramMin, histogramMax;

  // ======= Gaussian-mixture fit of the histograms =======
  /** VelocityGMM: fit every histogram (needs VelocityHistogram). */
  bool gmm = false;
  /** GMMMaxIterations: EM iterations at most, plus one when the last one
   *  prunes a component or resets a non-finite mean. */
  int gmmMaxIterations = 100;
  /** GMMConvergenceThreshold: stop when the log-likelihood per unit weight
   *  of the histogram gains less, or decreases (from the 7th iteration on). */
  double gmmConvergenceThreshold = 1e-5;
  /** GMMWarmStart: start from the fit of the previous analysis. */
  bool gmmWarmStart = true;
  /** GMMNormalize: fit the velocities divided by the largest velocity of
   *  the bins. */
  bool gmmNormalize = true;
  /** GMMCheckCovariance: reset non-finite covariance elements and raise
   *  the variances to GMMMinVariance. */
  bool gmmCheckCovariance = true;
  /** GMMCovarianceTolerance: smallest pivot of the Cholesky factorisation,
   *  in the units of the fit (normalised with GMMNormalize). */
  double gmmCovarianceTolerance = 1e-9;
  /** GMMMinVariance: with GMMCheckCovariance, the smallest variance of each
   *  velocity given the previous ones (a Cholesky pivot), so also on the
   *  diagonal, in the units of the fit (the default suits GMMNormalize). */
  double gmmMinVariance = 1e-4;
  /** GMMPruneComponents: drop components of small weight, the smallest
   *  first, one at a time. */
  bool gmmPruneComponents = true;
  /** GMMPruneThreshold: weight below which a component is dropped. */
  double gmmPruneThreshold = 0.005;

  // ======= Macrocell (v_par, v_perp) spectra =======
  /** VelocitySpectra: needs the three macrocell sizes, at most the cells of
   *  a rank, and a species in VelocitySpectraSpecies. */
  bool velocitySpectra = false;
  /** MacrocellNx, MacrocellNy, MacrocellNz: macrocell size in cells. */
  int macrocell[3] = {0, 0, 0};
  /** VelocitySpectraSpecies: per-species mask, nonzero = on. */
  std::vector<int> spectraSpecies;

  /** True when an analysis runs at some cycle. */
  bool any() const {
    return cycle > 0 && (velocityHistogram || velocitySpectra);
  }
  /** True when the analyses run at this cycle. */
  bool runsAt(int cycleNumber) const {
    return any() && cycleNumber % cycle == 0;
  }
};

#endif // DATA_ANALYSIS_INPUT_H
