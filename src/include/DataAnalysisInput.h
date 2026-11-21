#ifndef DATA_ANALYSIS_INPUT_H
#define DATA_ANALYSIS_INPUT_H

#include <string>
#include <vector>

/**
 * @file DataAnalysisInput.h
 * @brief The data-analysis options of the input file, read by Collective.
 *
 * Every DataAnalysisCycle cycles the analysis thread computes, per species,
 * the 3D velocity histogram of each rank or of each of its macrocells
 * (VelocityHistogram, VelocityHistogramMacrocells), its Gaussian-mixture fit
 * (VelocityGMM, the compressed form of the distribution) and the macrocell
 * (v_par, v_perp) spectra (VelocitySpectra). The member initialisers are the
 * defaults of the input file.
 */

/** Default fixed histogram range [-v, v] of a species with qom < 0. */
inline constexpr double kHistogramVmaxElectrons = 0.2;
/** Default fixed histogram range [-v, v] of the other species. */
inline constexpr double kHistogramVmaxIons = 0.09;
/** Most components of a GMM fit: one per thread of a block of the per-fit
 *  GMM kernels (cudaGMMWeight::kGMMThreads). */
inline constexpr int kGMMMaxComponents = 256;

struct DataAnalysisInput {
  /** DataAnalysisCycle: analyse every this many cycles; 0 disables. */
  int cycle = 50;
  /** DataAnalysisOutputDir: parent folder of the analysis outputs (read
   *  with a '/' at the end). */
  std::string outputDir = "./";

  // ======= Velocity histograms =======
  /** VelocityHistogram: 3D (u, v, w) histogram of every species. */
  bool velocityHistogram = false;
  /** VelocityHistogramMacrocells: a histogram (and GMM fit) per macrocell;
   *  otherwise one per rank. */
  bool histogramMacrocells = false;
  /** VelocityHistogramMacrocellNx, Ny, Nz: macrocell size in cells of the
   *  macrocell histograms and their GMM fits, at most the cells of a rank
   *  and at most DAConfig::MAX_HISTOGRAM_MACROCELLS macrocells on it; each
   *  defaults to the common size MacrocellNx, Ny, Nz. */
  int histogramMacrocellSize[3] = {0, 0, 0};
  /** VelocityHistogramFixedRange: bin [VelocityHistogramMin,
   *  VelocityHistogramMax] on the three axes; otherwise the range of the
   *  velocities of the species at that cycle in the rank or in the macrocell
   *  (velocityHistogram::binGridOf). */
  bool histogramFixedRange = true;
  /** VelocityHistogramMin / Max, per species (max > 0). The macrocell
   *  spectra bin v_par in [-max, max] and v_perp in [0, max]. */
  std::vector<double> histogramMin, histogramMax;

  // ======= Gaussian-mixture fit of the histograms =======
  /** VelocityGMM: fit every histogram (needs VelocityHistogram). */
  bool gmm = false;
  /** GMMComponents: Gaussians of each fit, 1 to kGMMMaxComponents. The
   *  GPU memory of a fit grows with them, 8 bytes per component and
   *  non-empty bin and 8 more per bin: up to 8 MB per component and 8 MB
   *  for the histogram of a rank, fitted in one piece. The last fit of each
   * species and region stays on the host, 104 bytes per component. */
  int gmmComponents = 10;
  /** GMMMaxIterations: EM iterations at most, plus one when the last one
   *  prunes a component, which gives its weight back to the others. */
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
  /** GMMPruneThreshold: weight below which a component is dropped; with
   *  pruning, below 1 / GMMComponents (the weights of a cold start). */
  double gmmPruneThreshold = 0.005;

  // ======= Macrocell (v_par, v_perp) spectra =======
  /** VelocitySpectra: needs its three macrocell sizes and a species in
   *  VelocitySpectraSpecies. */
  bool velocitySpectra = false;
  /** VelocitySpectraMacrocellNx, Ny, Nz: macrocell size in cells of the
   *  spectra, at most the cells of a rank and at most
   *  DAConfig::MAX_SPECTRA_MACROCELLS macrocells on it; each defaults to the
   *  common size MacrocellNx, Ny, Nz. */
  int spectraMacrocellSize[3] = {0, 0, 0};
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
