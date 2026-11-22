/**
 * @file gmmTest.cu
 * @brief Unit test of the Gaussian-mixture fit (cudaGMM.cuh) on the bin
 *        centres of a synthetic 3D velocity histogram, weighted by a known
 *        mixture.
 *
 * The cases share one GMM object on a non-blocking stream, as the analysis
 * does for its fits, and pass only the bins of nonzero weight. Cases
 * (--case <name>):
 *   onestep     one EM step from a fixed start with correlated covariances
 *               against a double-precision EM step on the host
 *   onesteplarge  the same on more points than the most blocks of the
 *               M-step statistics take in one pass (their block cap)
 *   correlated  two components, one with correlated u and v: the fit
 *               recovers the weights, means and covariances, and stops on the
 *               convergence threshold before the iteration limit
 *   truth       started at the truth with a large threshold: the fit runs the
 *               least steps before it may converge
 *   normalize   12 EM steps on the velocities and on the velocities divided
 *               by 1.7: the same components and log-likelihood in physical
 *               units, and the points given back
 *   prune       the same data with a third component started far from the
 *               data, fitted on normalised velocities: the third component is
 *               pruned (flagged inactive) and the other two are recovered in
 *               physical units
 *   steps       the prune case stopped by the iteration limit at the first
 *               pruning: one more step, weights summing to 1; at the limit
 *               the pruning raises no other small component
 *   reproducible  the same fit twice, and with a new GMM object: the same
 *               bits
 *   production  ten components from random means, on normalised velocities,
 *               with the pruning and the threshold of the input defaults: a
 *               finite fit as good as the two-component one
 *   line        all the weight on the bins with u = v (a singular covariance):
 *               the fit stays finite, recovers the mean and the covariance of
 *               the weight on the line, and writes a positive covariance
 *   floor       two diagonal-neighbour bins, singular along u = -v and with no
 *               spread in w, with and without normalisation: the written
 *               covariance has the determinant of the floor, GMMMinVariance^3
 *               in the units of the fit
 *   degenerate  one point for three components; points of no weight (the
 *               weights kept); a point of infinite weight (one step, the
 *               components finite); a non-positive start without the covariance
 *               check; no point, too many components or responsibilities,
 *               fits that do not cover the points, several fits without
 *               offsets, no EM iteration, no initial components, a NaN
 *               initial mean and, normalised, a scale not positive and
 *               finite refused
 *   batch       several fits in one batch (two mixtures, one with more points
 *               than a pass of the statistics, one point, no point, no
 *               weight), normalised and not, run to convergence and stopped
 *               by a pruning at the iteration limit: each fit gives the bits
 *               of the same fit alone
 *   manyfits    300 small random fits in one batch (some without points): the
 *               bits of the fits alone, finite, weights summing to 1
 *   components  one and the most components (a thread each in the per-fit
 *               kernels): one EM step against the host; one component run to
 *               convergence gives the moments of the points; a batch with
 *               the most components, pruning: the bits of the fits alone,
 *               finite, weights summing to 1
 */

#include <algorithm>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "cudaGMM.cuh"

namespace {

using real = double;
using weight_t = float;
using GMMModel = cudaGMMWeight::GMM<real, 3, weight_t>;
using GMMPoints = cudaGMMWeight::GMMDataMultiDim<real, 3, weight_t>;

/** Bins per axis of the histogram on [-1, 1]^3, and of the one-step cases:
 *  68^3 bins are more points than one pass of the M-step statistics. */
constexpr int kGrid = 40;
constexpr int kOneStepGrid = 12;
constexpr int kOneStepLargeGrid = 68;
constexpr int kStatisticsPass = cudaGMMWeight::kMaxStatisticsBlocks *
                                cudaGMMWeight::kGMMThreads *
                                cudaGMMWeight::kStatisticsPointsPerThread;
/** Total weight of the histogram: the counts of the analysis are large. */
constexpr real kTotalWeight = 1e7;
/** Fit parameters. */
constexpr int kMaxIterations = 1000;
constexpr real kThreshold = 1e-9;
constexpr real kCovarianceTolerance = 1e-9;
constexpr real kMinVariance = 1e-4;
constexpr real kPruneThreshold = 0.005;
/** Accepted errors of the fitted weights, means and covariances: the mixture
 *  is cut at the edges of the histogram, which moves its moments by about
 *  1e-4 (the moments of the bin centres match those of the Gaussians). */
constexpr real kWeightError = 1e-3, kMeanError = 1e-3;
constexpr real kCovarianceError = 1e-3;
/** Accepted relative error of one EM step against the host, and between the
 *  fits with and without normalisation (rounding). */
constexpr real kOneStepError = 1e-10;
constexpr real kNormalizeError = 1e-9;
/** EM steps of the normalize case. */
constexpr int kNormalizeSteps = 12;
/** The scale of the normalize case: not a power of 2, so that the scaling
 *  of the points and of the components rounds. */
constexpr real kNormalizeScale = 1.7;
/** A threshold any step of the truth case meets. */
constexpr real kLargeThreshold = 1.0;
/** Accepted error of the sum of the weights (kMinComponentWeight floors). */
constexpr real kWeightSumError = 1e-6;
/** Variance along u = v of the line case: that of the product of the two
 *  Gaussians of variance 0.04 in u and v. */
constexpr real kLineVariance = 0.02;
/** The floor case: the distance of its bins along u and v (a bin of 100 on
 *  [-1, 1]), and the accepted relative error of its determinant. */
constexpr real kFloorBin = 0.02;
constexpr real kFloorError = 1e-6;

/** EM steps of the reproducible case, and the input defaults of the
 *  production case. */
constexpr int kReproducibleSteps = 30;
constexpr real kProductionThreshold = 1e-5;
constexpr int kProductionIterations = 100;
constexpr int kProductionComponents = 10;
/** The production fit may miss the two-component optimum by this, per unit
 *  weight (local optima). */
constexpr real kProductionLoss = 1e-2;
/** The iteration limit of the batch case stopped at its first pruning. */
constexpr int kPruneLimit = cudaGMMWeightKernel::kPruneInterval;
/** The steps case: the start weight of a component on the first Gaussian,
 *  below the threshold after one step. */
constexpr real kSmallStartWeight = 2e-4;
/** The manyfits case: its fits, the most points of one, every how many fits
 *  one has no point and one is fitted alone. */
constexpr int kManyFits = 300;
constexpr int kManyFitsMaxPoints = 200;
constexpr int kManyFitsEmptyEvery = 50;
constexpr int kManyFitsAloneEvery = 23;
/** The components case: the most components of a fit (one per thread of a
 *  block of the per-fit kernels), and the variance of its drawn starts, broad
 *  enough for every component to take weight. */
constexpr int kMostComponents = cudaGMMWeight::kGMMThreads;
constexpr real kDrawnStartVariance = 0.25;
/** The prune threshold of its pruning batch: below 1 / kMostComponents, as
 *  the input requires. */
constexpr real kMostComponentsPruneThreshold = 0.002;
/** Variance of the drawn starts of the production case. */
constexpr real kProductionStartVariance = 0.04;

/** The stream of the fits: non-blocking, as in the analysis. */
cudaStream_t stream;
int failures = 0;

void check(bool ok, const char* what, double value) {
  std::printf("  [%s] %s (%.3e)\n", ok ? "ok" : "FAIL", what, value);
  if (!ok)
    failures++;
}

struct Gaussian {
  real weight;
  real mean[3];
  real cov[9];        // row major
  bool active = true; // of a fit: not pruned
};

/** Determinant of a row-major 3x3 matrix. */
real determinant(const real* c) {
  return c[0] * (c[4] * c[8] - c[5] * c[7]) -
         c[1] * (c[3] * c[8] - c[5] * c[6]) +
         c[2] * (c[3] * c[7] - c[4] * c[6]);
}

/** Leading principal minors positive: a covariance a density can use. */
bool positiveDefinite(const Gaussian& g) {
  const real* c = g.cov;
  return c[0] > 0 && c[0] * c[4] - c[1] * c[3] > 0 && determinant(c) > 0;
}

/** Density of a 3D Gaussian at x. */
real density(const Gaussian& g, const real* x) {
  const real* c = g.cov;
  const real det = determinant(c);
  const real inv[9] = {c[4] * c[8] - c[5] * c[7], c[2] * c[7] - c[1] * c[8],
                       c[1] * c[5] - c[2] * c[4], c[5] * c[6] - c[3] * c[8],
                       c[0] * c[8] - c[2] * c[6], c[2] * c[3] - c[0] * c[5],
                       c[3] * c[7] - c[4] * c[6], c[1] * c[6] - c[0] * c[7],
                       c[0] * c[4] - c[1] * c[3]};
  real d[3], q = 0;
  for (int i = 0; i < 3; i++)
    d[i] = x[i] - g.mean[i];
  for (int i = 0; i < 3; i++)
    for (int j = 0; j < 3; j++)
      q += d[i] * inv[i * 3 + j] * d[j] / det;
  return std::exp(-0.5 * q) / std::sqrt(std::pow(2 * M_PI, 3) * det);
}

/** Bin centres of the histogram (u, v, w arrays) and their weights. */
struct Histogram {
  // page-locked: copied to the device
  PinnedVector<real> point[3];
  PinnedVector<weight_t> weight;
  size_t size() const { return weight.size(); }
};

/** The non-empty bins of the histogram of the mixture; with `onLine` only
 *  the bins with u = v. */
Histogram histogramOf(const std::vector<Gaussian>& mixture, bool onLine,
                      int grid = kGrid) {
  const real bin = 2.0 / grid;
  Histogram h;
  for (int k = 0; k < grid; k++)
    for (int j = 0; j < grid; j++)
      for (int i = 0; i < grid; i++) {
        const real x[3] = {-1 + (i + 0.5) * bin, -1 + (j + 0.5) * bin,
                           -1 + (k + 0.5) * bin};
        real value = 0;
        if (!onLine || i == j)
          for (const Gaussian& g : mixture)
            value += g.weight * density(g, x);
        const weight_t w = (weight_t)(value * bin * bin * bin * kTotalWeight);
        if (w <= 0)
          continue;
        for (int d = 0; d < 3; d++)
          h.point[d].push_back(x[d]);
        h.weight.push_back(w);
      }
  return h;
}

struct Result {
  std::vector<Gaussian> components;
  int steps;
  real logLikelihood;
  real pointsError; // largest change of a point by the fit
  real weightSum() const {
    real sum = 0;
    for (const Gaussian& g : components)
      sum += g.weight;
    return sum;
  }
};

/**
 * Fit each histogram starting from its start (in physical units) with the
 * given GMM, all in one batch; with scales > 0 (all or none) the fit of each
 * runs on the velocities divided by its scale (GMMNormalize). The steps of a
 * result are those of its fit.
 */
std::vector<Result>
fitBatch(GMMModel& gmm, const std::vector<const Histogram*>& histograms,
         const std::vector<std::vector<Gaussian>>& starts,
         const std::vector<real>& scales, bool prune,
         int maxIterations = kMaxIterations, bool checkCovariance = true,
         real threshold = kThreshold, real pruneThreshold = kPruneThreshold) {
  const int fits = (int)histograms.size();
  const int numComponents = (int)starts[0].size();
  std::vector<int> offset(fits + 1, 0);
  for (int f = 0; f < fits; f++)
    offset[f + 1] = offset[f] + (int)histograms[f]->size();
  const int n = offset[fits];
  // the uploads on the stream of the fit, which does not wait for the
  // legacy stream
  real* point[3];
  for (int d = 0; d < 3; d++) {
    cudaErrChk(cudaMalloc(&point[d], n * sizeof(real)));
    for (int f = 0; f < fits; f++)
      cudaErrChk(cudaMemcpyAsync(point[d] + offset[f],
                                 histograms[f]->point[d].data(),
                                 histograms[f]->size() * sizeof(real),
                                 cudaMemcpyHostToDevice, stream));
  }
  weight_t* weight;
  cudaErrChk(cudaMalloc(&weight, n * sizeof(weight_t)));
  for (int f = 0; f < fits; f++)
    cudaErrChk(cudaMemcpyAsync(weight + offset[f], histograms[f]->weight.data(),
                               histograms[f]->size() * sizeof(weight_t),
                               cudaMemcpyHostToDevice, stream));

  std::vector<real> w, mean, cov, fitScale;
  for (int f = 0; f < fits; f++) {
    for (const Gaussian& g : starts[f]) {
      w.push_back(g.weight);
      mean.insert(mean.end(), g.mean, g.mean + 3);
      cov.insert(cov.end(), g.cov, g.cov + 9);
    }
    fitScale.push_back(scales[f] > 0 ? scales[f] : 1); // in [-1, 1]
  }
  cudaGMMWeight::GMMParam_t<real> param = {.numComponents = numComponents,
                                           .maxIteration = maxIterations,
                                           .threshold = threshold,
                                           .weightInit = w.data(),
                                           .meanInit = mean.data(),
                                           .coVarianceInit = cov.data(),
                                           .normalize = scales[0] > 0,
                                           .scale = fitScale.data(),
                                           .checkCovariance = checkCovariance,
                                           .covarianceTolerance =
                                               kCovarianceTolerance,
                                           .minVariance = kMinVariance,
                                           .prune = prune,
                                           .pruneThreshold = pruneThreshold};

  // page-locked: config copies it to the device
  const PinnedVector<GMMPoints> data(1, GMMPoints(n, point, weight));
  gmm.config(&param, data.data(), fits, offset.data());
  gmm.initGMM();
  PinnedVector<real> back[3];
  for (int d = 0; d < 3; d++) {
    back[d].resize(n);
    cudaErrChk(cudaMemcpyAsync(back[d].data(), point[d], n * sizeof(real),
                               cudaMemcpyDeviceToHost, stream));
  }
  cudaErrChk(cudaStreamSynchronize(stream));
  std::vector<Result> results(fits);
  for (int f = 0; f < fits; f++) {
    const auto r = gmm.getGMMResult(0, f);
    Result& result = results[f];
    result.steps = r.convergeStep;
    result.logLikelihood = r.logLikelihoodFinal;
    result.pointsError = 0;
    for (int d = 0; d < 3; d++)
      for (int i = 0; i < (int)histograms[f]->size(); i++)
        result.pointsError =
            std::max(result.pointsError, std::fabs(back[d][offset[f] + i] -
                                                   histograms[f]->point[d][i]));
    for (int c = 0; c < numComponents; c++) {
      Gaussian g;
      g.active = r.active[c];
      g.weight = r.weight[c];
      for (int d = 0; d < 3; d++)
        g.mean[d] = r.mean[c * 3 + d];
      for (int e = 0; e < 9; e++)
        g.cov[e] = r.coVariance[c * 9 + e];
      result.components.push_back(g);
    }
  }
  for (int d = 0; d < 3; d++)
    cudaErrChk(cudaFree(point[d]));
  cudaErrChk(cudaFree(weight));
  return results;
}

/** One fit of fitBatch: scale 0 for none. */
Result fit(GMMModel& gmm, const Histogram& h,
           const std::vector<Gaussian>& start, real scale, bool prune,
           int maxIterations = kMaxIterations, bool checkCovariance = true,
           real threshold = kThreshold, real pruneThreshold = kPruneThreshold) {
  return fitBatch(gmm, {&h}, {start}, {scale}, prune, maxIterations,
                  checkCovariance, threshold, pruneThreshold)[0];
}

/** Largest errors of the fitted components against the true ones. */
void compare(const std::vector<Gaussian>& fitted,
             const std::vector<Gaussian>& truth) {
  real errW = 0, errMean = 0, errCov = 0;
  for (size_t c = 0; c < truth.size(); c++) {
    errW = std::max(errW, std::fabs(fitted[c].weight - truth[c].weight));
    for (int d = 0; d < 3; d++)
      errMean =
          std::max(errMean, std::fabs(fitted[c].mean[d] - truth[c].mean[d]));
    for (int e = 0; e < 9; e++)
      errCov = std::max(errCov, std::fabs(fitted[c].cov[e] - truth[c].cov[e]));
  }
  check(errW < kWeightError, "weights recovered", errW);
  check(errMean < kMeanError, "means recovered", errMean);
  check(errCov < kCovarianceError, "covariances recovered", errCov);
}

/** Two components; the first has correlated u and v. */
std::vector<Gaussian> twoComponents() {
  return {
      {0.6, {-0.3, -0.1, 0.0}, {0.04, 0.03, 0, 0.03, 0.04, 0, 0, 0, 0.02}},
      {0.4, {0.35, 0.25, -0.2}, {0.02, 0, 0.01, 0, 0.03, 0, 0.01, 0, 0.02}}};
}

/** A start away from the truth: shifted means, round covariances. */
Gaussian startNear(const Gaussian& g, real shift) {
  Gaussian s = {0.5, {g.mean[0] + shift, g.mean[1] - shift, g.mean[2]}, {}};
  for (int d = 0; d < 3; d++)
    s.cov[d * 4] = 0.05;
  return s;
}

/** One EM step on the host, in double: the responsibilities of the start,
 *  then the weights, means and covariances about the new means; the
 *  log-likelihood of the start into logLikelihood. */
std::vector<Gaussian> hostEMStep(const Histogram& h,
                                 const std::vector<Gaussian>& start,
                                 real& logLikelihood) {
  const size_t n = h.size(), numComponents = start.size();
  std::vector<real> gamma(n * numComponents);
  logLikelihood = 0;
  for (size_t i = 0; i < n; i++) {
    const real x[3] = {h.point[0][i], h.point[1][i], h.point[2][i]};
    real p = 0;
    for (size_t k = 0; k < numComponents; k++)
      p += gamma[i * numComponents + k] =
          start[k].weight * density(start[k], x);
    for (size_t k = 0; k < numComponents; k++)
      gamma[i * numComponents + k] /= p;
    logLikelihood += h.weight[i] * std::log(p);
  }
  std::vector<Gaussian> next(numComponents);
  real total = 0;
  for (size_t k = 0; k < numComponents; k++) {
    Gaussian& g = next[k];
    real sum = 0, first[3] = {};
    for (size_t i = 0; i < n; i++) {
      const real hg = h.weight[i] * gamma[i * numComponents + k];
      sum += hg;
      for (int d = 0; d < 3; d++)
        first[d] += hg * h.point[d][i];
    }
    for (int d = 0; d < 3; d++)
      g.mean[d] = first[d] / sum;
    real second[9] = {};
    for (size_t i = 0; i < n; i++) {
      const real hg = h.weight[i] * gamma[i * numComponents + k];
      for (int a = 0; a < 3; a++)
        for (int b = 0; b < 3; b++)
          second[a * 3 + b] +=
              hg * (h.point[a][i] - g.mean[a]) * (h.point[b][i] - g.mean[b]);
    }
    for (int e = 0; e < 9; e++)
      g.cov[e] = second[e] / sum;
    g.weight = sum;
    total += sum;
  }
  for (Gaussian& g : next)
    g.weight /= total;
  return next;
}

/** Largest error of an EM step of the GPU against the host: relative of the
 *  weights, of the means, of the covariances relative to the first
 *  variance. */
real stepError(const std::vector<Gaussian>& fitted,
               const std::vector<Gaussian>& host) {
  real err = 0;
  for (size_t c = 0; c < host.size(); c++) {
    const Gaussian &a = fitted[c], &b = host[c];
    err = std::max(err, std::fabs(a.weight - b.weight) / b.weight);
    for (int d = 0; d < 3; d++)
      err = std::max(err, std::fabs(a.mean[d] - b.mean[d]));
    for (int e = 0; e < 9; e++)
      err = std::max(err, std::fabs(a.cov[e] - b.cov[e]) / b.cov[0]);
  }
  return err;
}

/** One EM step of the GPU from start against that of the host: its steps,
 *  components and log-likelihood. @return the step of the host */
std::vector<Gaussian> checkOneStep(GMMModel& gmm, const Histogram& h,
                                   const std::vector<Gaussian>& start) {
  real hostLogLikelihood;
  const std::vector<Gaussian> host = hostEMStep(h, start, hostLogLikelihood);
  const Result r = fit(gmm, h, start, 0, false, 1);
  const real err = stepError(r.components, host);
  check(r.steps == 1 && r.components.size() == start.size(),
        "one EM step, every component written", r.steps);
  check(err < kOneStepError, "the EM step of the host", err);
  check(std::fabs(r.logLikelihood - hostLogLikelihood) <
            kOneStepError * std::fabs(hostLogLikelihood),
        "the log-likelihood of the host",
        std::fabs(r.logLikelihood / hostLogLikelihood - 1));
  return host;
}

int testOneStep(GMMModel& gmm, int grid) {
  failures = 0;
  const std::vector<Gaussian> truth = twoComponents();
  const Histogram h = histogramOf(truth, false, grid);
  std::vector<Gaussian> start = {startNear(truth[0], 0.1),
                                 startNear(truth[1], -0.1)};
  // correlated starts: the off-diagonal terms of the factor in the E step
  start[0].cov[1] = start[0].cov[3] = 0.02;
  start[1].cov[5] = start[1].cov[7] = -0.015;
  std::printf("  %zu points\n", h.size());
  if (grid == kOneStepLargeGrid)
    check((int)h.size() > kStatisticsPass,
          "more points than a pass of the statistics", h.size());
  checkOneStep(gmm, h, start);
  return failures;
}

int testOneStepSmall(GMMModel& gmm) {
  std::printf("onestep\n");
  return testOneStep(gmm, kOneStepGrid);
}

int testOneStepLarge(GMMModel& gmm) {
  std::printf("onesteplarge\n");
  return testOneStep(gmm, kOneStepLargeGrid);
}

int testCorrelated(GMMModel& gmm) {
  std::printf("correlated\n");
  failures = 0;
  const std::vector<Gaussian> truth = twoComponents();
  const Result r =
      fit(gmm, histogramOf(truth, false),
          {startNear(truth[0], 0.1), startNear(truth[1], -0.1)}, 0, false);
  std::printf("  %d EM steps, log-likelihood %.6e\n", r.steps, r.logLikelihood);
  check(std::isfinite(r.logLikelihood), "log-likelihood finite",
        r.logLikelihood);
  compare(r.components, truth);
  check(r.steps < kMaxIterations, "converged before the iteration limit",
        r.steps);
  return failures;
}

int testNormalize(GMMModel& gmm) {
  std::printf("normalize\n");
  failures = 0;
  const std::vector<Gaussian> truth = twoComponents();
  const Histogram h = histogramOf(truth, false);
  const std::vector<Gaussian> start = {startNear(truth[0], 0.1),
                                       startNear(truth[1], -0.1)};
  // a fixed number of steps, before the convergence, for the same steps
  const Result plain = fit(gmm, h, start, 0, false, kNormalizeSteps);
  const Result scaled =
      fit(gmm, h, start, kNormalizeScale, false, kNormalizeSteps);
  real err = 0;
  for (size_t c = 0; c < truth.size(); c++) {
    const Gaussian &a = plain.components[c], &b = scaled.components[c];
    err = std::max(err, std::fabs(a.weight - b.weight));
    for (int d = 0; d < 3; d++)
      err = std::max(err, std::fabs(a.mean[d] - b.mean[d]));
    for (int e = 0; e < 9; e++)
      err = std::max(err, std::fabs(a.cov[e] - b.cov[e]));
  }
  std::printf("  %d and %d EM steps\n", plain.steps, scaled.steps);
  check(plain.steps == kNormalizeSteps && scaled.steps == kNormalizeSteps,
        "both fits run the same steps", scaled.steps);
  check(err < kNormalizeError, "the same components", err);
  check(std::fabs(scaled.logLikelihood - plain.logLikelihood) <
            kNormalizeError * std::fabs(plain.logLikelihood),
        "the same log-likelihood in physical units",
        std::fabs(scaled.logLikelihood / plain.logLikelihood - 1));
  check(scaled.pointsError <= kNormalizeError, "the points given back",
        scaled.pointsError);
  return failures;
}

int testTruth(GMMModel& gmm) {
  std::printf("truth\n");
  failures = 0;
  const std::vector<Gaussian> truth = twoComponents();
  const Result r = fit(gmm, histogramOf(truth, false), truth, 0, false,
                       kMaxIterations, true, kLargeThreshold);
  check(r.steps == cudaGMMWeightKernel::kMinStepsToConverge + 1,
        "the least steps before the convergence", r.steps);
  check(r.components[0].active && r.components[1].active,
        "every component active without pruning", 0);
  return failures;
}

/** The two components and one started far from the data. */
std::vector<Gaussian> startWithFar(const std::vector<Gaussian>& truth) {
  Gaussian far = {0.2, {0.9, -0.9, 0.9}, {}};
  for (int d = 0; d < 3; d++)
    far.cov[d * 4] = 1e-3;
  return {startNear(truth[0], 0.1), startNear(truth[1], -0.1), far};
}

int testPrune(GMMModel& gmm) {
  std::printf("prune\n");
  failures = 0;
  const std::vector<Gaussian> truth = twoComponents();
  const Result r =
      fit(gmm, histogramOf(truth, false), startWithFar(truth), 2.0, true);
  std::printf("  %d EM steps, weights %.4f %.4f %.3e\n", r.steps,
              r.components[0].weight, r.components[1].weight,
              r.components[2].weight);
  // an active component keeps kMinComponentWeight at least
  check(r.components[2].weight < 10 * cudaGMMWeightKernel::kPrunedWeight,
        "far component pruned", r.components[2].weight);
  check(!r.components[2].active && r.components[0].active &&
            r.components[1].active,
        "the pruned component flagged inactive, the others active", 0);
  compare({r.components[0], r.components[1]}, truth);
  check(std::fabs(r.weightSum() - 1) < kWeightSumError, "weights sum to 1",
        r.weightSum() - 1);
  return failures;
}

int testSteps(GMMModel& gmm) {
  std::printf("steps\n");
  failures = 0;
  const std::vector<Gaussian> truth = twoComponents();
  const int limit = kPruneLimit;
  const Result r = fit(gmm, histogramOf(truth, false), startWithFar(truth), 2.0,
                       true, limit);
  std::printf("  %d EM steps for %d at most, weights %.4f %.4f %.3e\n", r.steps,
              limit, r.components[0].weight, r.components[1].weight,
              r.components[2].weight);
  check(r.steps == limit + 1, "one step after the pruning at the limit",
        r.steps);
  check(r.components[2].weight < 10 * cudaGMMWeightKernel::kPrunedWeight,
        "far component pruned", r.components[2].weight);
  check(std::fabs(r.weightSum() - 1) < kWeightSumError, "weights sum to 1",
        r.weightSum() - 1);
  check(std::isfinite(r.logLikelihood), "log-likelihood finite",
        r.logLikelihood);

  // the pruning at the limit raises no other component: a small one on the
  // first Gaussian, below the threshold after the one step of the limit 1,
  // stays below it in the final step (raised to kRaisedShare / 4 it would
  // take a share of the first one)
  std::vector<Gaussian> start = startWithFar(truth);
  Gaussian small = truth[0];
  small.weight = kSmallStartWeight;
  start.push_back(small);
  const Result one = fit(gmm, histogramOf(truth, false), start, 2.0, true, 1);
  std::printf("  limit 1: %d EM steps, weights %.4f %.4f %.3e %.3e\n",
              one.steps, one.components[0].weight, one.components[1].weight,
              one.components[2].weight, one.components[3].weight);
  check(one.steps == 2 &&
            one.components[2].weight <
                10 * cudaGMMWeightKernel::kPrunedWeight &&
            one.components[3].weight < kPruneThreshold,
        "the pruning at the limit raises no other component",
        one.components[3].weight);
  return failures;
}

int testLine(GMMModel& gmm) {
  std::printf("line\n");
  failures = 0;
  // the weight of a Gaussian on the bins with u = v
  const Gaussian g = {
      1.0, {0.1, 0.1, -0.2}, {0.04, 0.0, 0, 0.0, 0.04, 0, 0, 0, 0.02}};
  Gaussian start = {1.0, {0, 0, 0}, {}};
  for (int d = 0; d < 3; d++)
    start.cov[d * 4] = 0.05;
  const Result r = fit(gmm, histogramOf({g}, true), {start}, 0, false);
  const Gaussian& f = r.components[0];
  bool finite = std::isfinite(r.logLikelihood);
  for (int e = 0; e < 9; e++)
    finite = finite && std::isfinite(f.cov[e]);
  real errMean = 0;
  for (int d = 0; d < 3; d++)
    errMean = std::max(errMean, std::fabs(f.mean[d] - g.mean[d]));
  // u = v: var u, var v and cov uv are the variance along the line
  const real expected[9] = {
      kLineVariance, kLineVariance, 0, kLineVariance, kLineVariance, 0, 0, 0,
      g.cov[8]};
  real errCov = 0;
  for (int e = 0; e < 9; e++)
    errCov = std::max(errCov, std::fabs(f.cov[e] - expected[e]));
  std::printf("  %d EM steps, log-likelihood %.6e, var u %.4e, cov uv %.4e\n",
              r.steps, r.logLikelihood, f.cov[0], f.cov[1]);
  check(finite, "fit finite", r.logLikelihood);
  check(errMean < kMeanError, "mean recovered", errMean);
  check(errCov < kCovarianceError, "covariance on the line recovered", errCov);
  check(positiveDefinite(f), "the written covariance is positive", f.cov[0]);
  return failures;
}

int testFloor(GMMModel& gmm) {
  std::printf("floor\n");
  failures = 0;
  Histogram two;
  for (int b = 0; b < 2; b++) {
    two.point[0].push_back(b * kFloorBin);
    two.point[1].push_back(b * kFloorBin);
    two.point[2].push_back(0);
    two.weight.push_back(1000);
  }
  Gaussian start = {1.0, {0.01, 0.01, 0}, {}};
  for (int d = 0; d < 3; d++)
    start.cov[d * 4] = 0.05;
  for (const real scale : {0.0, 2.0}) {
    const Result r = fit(gmm, two, {start}, scale, false);
    const real floor = (scale > 0 ? scale * scale : 1) * kMinVariance;
    const real ratio = determinant(r.components[0].cov) / std::pow(floor, 3);
    std::printf("  scale %g: %d EM steps, det / floor^3 %.6f\n", scale, r.steps,
                ratio);
    check(std::isfinite(r.logLikelihood) && ratio > 1 - kFloorError &&
              ratio < 1 + kFloorError,
          scale > 0 ? "normalised: the determinant of the floor"
                    : "the determinant of the floor",
          ratio);
  }
  return failures;
}

int testDegenerate(GMMModel& gmm) {
  std::printf("degenerate\n");
  failures = 0;
  Gaussian start = {1.0 / 3, {0, 0, 0}, {}};
  for (int d = 0; d < 3; d++)
    start.cov[d * 4] = 0.05;
  std::vector<Gaussian> three = {start, start, start};
  for (int c = 0; c < 3; c++)
    three[c].mean[c] = 0.2;
  // one point: the components collapse on it
  Histogram one;
  for (int d = 0; d < 3; d++)
    one.point[d].push_back(0.1 * (d + 1));
  one.weight.push_back(1000);
  const Result r = fit(gmm, one, three, 0, true);
  bool finite = std::isfinite(r.logLikelihood), positive = true;
  for (const Gaussian& g : r.components) {
    for (int d = 0; d < 3; d++)
      finite = finite && std::isfinite(g.mean[d]);
    positive = positive && positiveDefinite(g);
  }
  std::printf("  one point: %d EM steps, log-likelihood %.6e\n", r.steps,
              r.logLikelihood);
  check(finite, "one point: fit finite", r.logLikelihood);
  check(positive, "one point: covariances positive", 0);
  check(std::fabs(r.weightSum() - 1) < kWeightSumError,
        "one point: weights sum to 1", r.weightSum() - 1);
  // points of no weight: nothing to fit, the EM stops at once
  Histogram none = histogramOf(twoComponents(), false, kOneStepGrid);
  std::fill(none.weight.begin(), none.weight.end(), weight_t(0));
  const Result z = fit(gmm, none, three, 0, true);
  check(z.steps == 1 && z.logLikelihood == 0,
        "no weight: one step, log-likelihood 0", z.steps);
  real kept = 0;
  for (const Gaussian& g : z.components)
    kept = std::max(kept, std::fabs(g.weight - start.weight));
  check(kept < kWeightSumError, "no weight: the weights kept", kept);
  // a point of infinite weight (a bin whose sum overflowed the float): the
  // EM stops at once, the components finite
  Histogram overflowed = histogramOf(twoComponents(), false, kOneStepGrid);
  overflowed.weight[0] = INFINITY;
  const Result u = fit(gmm, overflowed, three, 0, true);
  bool componentsFinite = true;
  for (const Gaussian& g : u.components) {
    componentsFinite &= std::isfinite(g.weight);
    for (int d = 0; d < 3; d++)
      componentsFinite &= std::isfinite(g.mean[d]);
    for (int e = 0; e < 9; e++)
      componentsFinite &= std::isfinite(g.cov[e]);
  }
  check(u.steps == 1 && componentsFinite,
        "a point of infinite weight: one step, the components finite", u.steps);

  // no covariance check: a start that is not positive definite (|cov uv| >
  // the variances) is factored with the pivot floor: one EM step stays
  // finite, with positive covariances
  std::vector<Gaussian> skew = three;
  skew[0].cov[1] = skew[0].cov[3] = 0.08;
  const Histogram h = histogramOf(twoComponents(), false);
  const Result c = fit(gmm, h, skew, 0, false, 1, false);
  bool positiveFit = std::isfinite(c.logLikelihood);
  for (const Gaussian& g : c.components)
    positiveFit = positiveFit && positiveDefinite(g);
  check(positiveFit, "no covariance check: a non-positive start fits finite",
        c.logLikelihood);

  // no point, too many components or responsibilities, fits that do not
  // cover the points, no iteration, a scale not positive and finite, no or
  // non-finite initial components: refused (before any work on the points)
  // fits: those of the offsets, or given without offsets; the initial means
  // meanValue, or no initial components
  auto refused = [&](int numComponents, int numPoints,
                     const std::vector<int>& offsets = {}, int fits = 0,
                     int maxIteration = 1, real scale = 1, real meanValue = 0,
                     bool initial = true) {
    if (fits == 0)
      fits = offsets.empty() ? 1 : (int)offsets.size() - 1;
    std::vector<real> w(fits * numComponents, 1.0 / numComponents),
        mean(fits * numComponents * 3, meanValue),
        cov(fits * numComponents * 9, 0), scales(fits, scale);
    cudaGMMWeight::GMMParam_t<real> param = {
        .numComponents = numComponents,
        .maxIteration = maxIteration,
        .threshold = kThreshold,
        .weightInit = initial ? w.data() : nullptr,
        .meanInit = mean.data(),
        .coVarianceInit = cov.data(),
        .normalize = true,
        .scale = scales.data(),
        .checkCovariance = true,
        .covarianceTolerance = kCovarianceTolerance,
        .minVariance = kMinVariance,
        .prune = false,
        .pruneThreshold = kPruneThreshold};
    real* none[3] = {nullptr, nullptr, nullptr};
    const PinnedVector<GMMPoints> data(1, GMMPoints(numPoints, none, nullptr));
    try {
      gmm.config(&param, data.data(), fits,
                 offsets.empty() ? nullptr : offsets.data());
    } catch (const std::invalid_argument&) {
      return true;
    }
    cudaErrChk(cudaStreamSynchronize(stream)); // the queued copy of data
    return false;
  };
  check(refused(2, 0), "no point refused", 0);
  check(refused(cudaGMMWeight::kGMMThreads + 1, 10),
        "more components than threads refused", 0);
  check(refused(2, 10, {0, 4, 9}) && refused(2, 10, {1, 4, 10}) &&
            refused(2, 10, {0, 6, 4, 10}),
        "fits that do not cover the points in order refused", 0);
  check(refused(2, 10, {}, 3), "several fits without offsets refused", 0);
  check(refused(2, 10, {}, 1, 0), "no EM iteration refused", 0);
  check(refused(2, 10, {}, 1, 1, 0) && refused(2, 10, {}, 1, 1, NAN) &&
            refused(2, 10, {}, 1, 1, INFINITY),
        "normalised, a scale 0, NaN or infinite refused", 0);
  check(refused(cudaGMMWeight::kGMMThreads,
                INT_MAX / cudaGMMWeight::kGMMThreads + 1),
        "more than INT_MAX responsibilities refused", 0);
  check(refused(2, 10, {}, 1, 1, 1, 0, false), "no initial components refused",
        0);
  check(refused(2, 10, {}, 1, 1, 1, NAN), "a NaN initial mean refused", 0);
  return failures;
}

/** The fits are equal bit for bit (the parameters of the components; not
 *  the padding of a Gaussian). */
bool sameFit(const Result& a, const Result& b) {
  bool same = a.steps == b.steps && a.logLikelihood == b.logLikelihood &&
              a.components.size() == b.components.size();
  for (size_t c = 0; same && c < a.components.size(); c++) {
    const Gaussian &x = a.components[c], &y = b.components[c];
    same = std::memcmp(&x.weight, &y.weight, sizeof(x.weight)) == 0 &&
           std::memcmp(x.mean, y.mean, sizeof(x.mean)) == 0 &&
           std::memcmp(x.cov, y.cov, sizeof(x.cov)) == 0 &&
           x.active == y.active;
  }
  return same;
}

int testReproducible(GMMModel& gmm) {
  std::printf("reproducible\n");
  failures = 0;
  const std::vector<Gaussian> truth = twoComponents();
  const Histogram h = histogramOf(truth, false);
  const std::vector<Gaussian> start = startWithFar(truth);
  const Result a = fit(gmm, h, start, 2.0, true, kReproducibleSteps);
  const Result b = fit(gmm, h, start, 2.0, true, kReproducibleSteps);
  check(sameFit(a, b), "the same fit twice: the same bits", a.steps);
  GMMModel fresh(stream);
  check(sameFit(a, fit(fresh, h, start, 2.0, true, kReproducibleSteps)),
        "a new GMM object: the same bits", a.steps);
  return failures;
}

/** A start as the analysis draws it: means in the unit ball, round
 *  covariances of the given variance, equal weights. */
std::vector<Gaussian> drawnStart(int components, real variance,
                                 std::mt19937& gen) {
  std::uniform_real_distribution<real> uniform(-1, 1);
  std::vector<Gaussian> start(components);
  for (Gaussian& g : start) {
    g = {real(1) / components, {}, {}};
    real r2;
    do {
      r2 = 0;
      for (int d = 0; d < 3; d++) {
        g.mean[d] = uniform(gen);
        r2 += g.mean[d] * g.mean[d];
      }
    } while (r2 > 1);
    for (int d = 0; d < 3; d++)
      g.cov[d * 4] = variance;
  }
  return start;
}

int testProduction(GMMModel& gmm) {
  std::printf("production\n");
  failures = 0;
  const std::vector<Gaussian> truth = twoComponents();
  const Histogram h = histogramOf(truth, false);
  std::mt19937 gen(2024);
  const std::vector<Gaussian> start =
      drawnStart(kProductionComponents, kProductionStartVariance, gen);
  const Result r = fit(gmm, h, start, 1.0, true, kProductionIterations, true,
                       kProductionThreshold);
  const Result two = fit(
      gmm, h, {startNear(truth[0], 0.1), startNear(truth[1], -0.1)}, 0, false);
  real total = 0;
  for (weight_t w : h.weight)
    total += w;
  int active = 0;
  for (const Gaussian& g : r.components)
    active += g.weight > 10 * cudaGMMWeightKernel::kPrunedWeight;
  const real loss = (two.logLikelihood - r.logLikelihood) / total;
  std::printf("  %d EM steps, %d active components, log-likelihood %.6e "
              "(two components %.6e)\n",
              r.steps, active, r.logLikelihood, two.logLikelihood);
  check(std::isfinite(r.logLikelihood), "log-likelihood finite",
        r.logLikelihood);
  check(std::fabs(r.weightSum() - 1) < kWeightSumError, "weights sum to 1",
        r.weightSum() - 1);
  check(loss < kProductionLoss, "as good as the two-component fit", loss);
  return failures;
}

int testBatch(GMMModel& gmm) {
  std::printf("batch\n");
  failures = 0;
  const std::vector<Gaussian> truth = twoComponents();
  std::vector<Gaussian> other = truth; // shifted and narrower
  for (Gaussian& g : other) {
    g.mean[0] *= -0.5;
    g.mean[2] += 0.1;
    for (real& c : g.cov)
      c *= 0.5;
  }
  const Histogram a = histogramOf(truth, false);
  const Histogram b = histogramOf(other, false, kOneStepLargeGrid);
  Histogram one, none;
  for (int d = 0; d < 3; d++)
    one.point[d].push_back(0.1 * (d + 1));
  one.weight.push_back(1000);
  Histogram weightless = histogramOf(truth, false, kOneStepGrid);
  std::fill(weightless.weight.begin(), weightless.weight.end(), weight_t(0));
  const std::vector<const Histogram*> histograms = {&a, &b, &one, &none,
                                                    &weightless};
  const std::vector<std::vector<Gaussian>> starts = {
      startWithFar(truth), startWithFar(other), startWithFar(truth),
      startWithFar(truth), startWithFar(other)};
  check((int)b.size() > kStatisticsPass,
        "a fit with more points than a pass of the statistics", b.size());

  // normalised and not, to convergence and stopped at a pruning at the limit
  const std::vector<real> scaled = {2.0, 1.5, 3.0, 2.0, 1.0},
                          plain(histograms.size(), 0);
  const struct {
    const char* name;
    const std::vector<real>& scales;
    int maxIterations;
  } runs[] = {{"normalised", scaled, kMaxIterations},
              {"not normalised", plain, kMaxIterations},
              {"limit at the first pruning", scaled, kPruneLimit}};
  for (const auto& run : runs) {
    const std::vector<Result> batch =
        fitBatch(gmm, histograms, starts, run.scales, true, run.maxIterations);
    bool same = true;
    std::printf("  %s: EM steps", run.name);
    for (size_t f = 0; f < histograms.size(); f++) {
      std::printf(" %d", batch[f].steps);
      if (histograms[f]->size() > 0)
        same &= sameFit(batch[f], fit(gmm, *histograms[f], starts[f],
                                      run.scales[f], true, run.maxIterations));
    }
    std::printf("\n");
    check(same, "each fit of the batch: the bits of the fit alone", 0);
    check(batch[3].steps == 0, "the fit without points: no step",
          batch[3].steps);
    check(batch[4].steps == 1 && batch[4].logLikelihood == 0,
          "the fit without weight: one step, log-likelihood 0", batch[4].steps);
  }
  return failures;
}

int testManyFits(GMMModel& gmm) {
  std::printf("manyfits\n");
  failures = 0;
  std::mt19937 gen(97531);
  std::uniform_real_distribution<real> unit(0.0, 1.0);
  std::normal_distribution<real> normal(0.0, 1.0);
  std::vector<Histogram> histograms(kManyFits);
  std::vector<std::vector<Gaussian>> starts(kManyFits);
  std::vector<real> scales(kManyFits);
  for (int f = 0; f < kManyFits; f++) {
    Histogram& h = histograms[f];
    const int points = f % kManyFitsEmptyEvery == 0
                           ? 0
                           : 1 + (int)(unit(gen) * kManyFitsMaxPoints);
    const real center[3] = {unit(gen) - 0.5, unit(gen) - 0.5, unit(gen) - 0.5};
    const real sigma = 0.05 + 0.15 * unit(gen);
    real largest = 0;
    for (int i = 0; i < points; i++) {
      for (int d = 0; d < 3; d++) {
        h.point[d].push_back(center[d] + sigma * normal(gen));
        largest = std::max(largest, std::fabs(h.point[d].back()));
      }
      h.weight.push_back((weight_t)(1 + 99 * unit(gen)));
    }
    scales[f] = std::max(largest, real(1e-3));
    for (int c = 0; c < 3; c++) {
      Gaussian g = {1.0 / 3, {}, {}};
      for (int d = 0; d < 3; d++) {
        g.mean[d] = center[d] + sigma * normal(gen);
        g.cov[d * 4] = sigma * sigma;
      }
      starts[f].push_back(g);
    }
  }
  std::vector<const Histogram*> pointers;
  for (const Histogram& h : histograms)
    pointers.push_back(&h);
  const std::vector<Result> batch =
      fitBatch(gmm, pointers, starts, scales, true, kProductionIterations, true,
               kProductionThreshold);
  bool same = true, finite = true, sums = true, empty = true;
  int mostSteps = 0;
  for (int f = 0; f < kManyFits; f++) {
    if (histograms[f].size() == 0) {
      empty &= batch[f].steps == 0;
      continue;
    }
    mostSteps = std::max(mostSteps, batch[f].steps);
    finite &= std::isfinite(batch[f].logLikelihood);
    sums &= std::fabs(batch[f].weightSum() - 1) < kWeightSumError;
    if (f % kManyFitsAloneEvery == 0)
      same &= sameFit(batch[f],
                      fit(gmm, histograms[f], starts[f], scales[f], true,
                          kProductionIterations, true, kProductionThreshold));
  }
  std::printf("  %d fits, %d EM steps at most\n", kManyFits, mostSteps);
  check(same, "fits of the batch: the bits of the fits alone", 0);
  check(finite, "every fit finite", 0);
  check(sums, "the weights of every fit sum to 1", 0);
  check(empty, "the fits without points: no step", 0);
  check(mostSteps <= kProductionIterations + 1, "the iteration limit",
        mostSteps);
  return failures;
}

int testComponents(GMMModel& gmm) {
  std::printf("components\n");
  failures = 0;
  const std::vector<Gaussian> truth = twoComponents();
  const Histogram h = histogramOf(truth, false, kOneStepGrid);
  std::mt19937 gen(8642);
  for (int components : {1, kMostComponents}) {
    const std::vector<Gaussian> start =
        drawnStart(components, kDrawnStartVariance, gen);
    std::printf("  %d components, one EM step\n", components);
    const std::vector<Gaussian> host = checkOneStep(gmm, h, start);
    if (components == 1) {
      // the first step gives the moments of the points, which do not move
      const Result c = fit(gmm, h, start, 2.0, true);
      std::printf("  1 component to convergence: %d EM steps\n", c.steps);
      check(c.steps == cudaGMMWeightKernel::kMinStepsToConverge + 1,
            "converged at the least steps", c.steps);
      check(c.components[0].weight == 1, "weight 1", c.components[0].weight);
      check(stepError(c.components, host) < kOneStepError,
            "the moments of the points", stepError(c.components, host));
    }
  }

  // a batch with the most components, normalised, pruning
  std::vector<Gaussian> other = truth;
  for (Gaussian& g : other)
    g.mean[1] = -g.mean[1];
  const Histogram b = histogramOf(other, false, kOneStepGrid);
  const std::vector<std::vector<Gaussian>> starts = {
      drawnStart(kMostComponents, kDrawnStartVariance, gen),
      drawnStart(kMostComponents, kDrawnStartVariance, gen)};
  const std::vector<real> scales = {1.5, 2.0};
  const std::vector<Result> batch =
      fitBatch(gmm, {&h, &b}, starts, scales, true, kProductionIterations, true,
               kProductionThreshold, kMostComponentsPruneThreshold);
  bool same = true, finite = true, sums = true;
  for (int f = 0; f < 2; f++) {
    int active = 0;
    for (const Gaussian& g : batch[f].components)
      active += g.weight > 10 * cudaGMMWeightKernel::kPrunedWeight;
    std::printf("  %d components, fit %d: %d EM steps, %d left\n",
                kMostComponents, f, batch[f].steps, active);
    same &=
        sameFit(batch[f], fit(gmm, f == 0 ? h : b, starts[f], scales[f], true,
                              kProductionIterations, true, kProductionThreshold,
                              kMostComponentsPruneThreshold));
    finite &= std::isfinite(batch[f].logLikelihood);
    sums &= std::fabs(batch[f].weightSum() - 1) < kWeightSumError;
  }
  check(same, "fits of the batch: the bits of the fits alone", 0);
  check(finite, "every fit finite", 0);
  check(sums, "the weights of every fit sum to 1", 0);
  return failures;
}

} // namespace

int main(int argc, char** argv) {
  const std::pair<const char*, int (*)(GMMModel&)> cases[] = {
      {"onestep", testOneStepSmall},  {"onesteplarge", testOneStepLarge},
      {"correlated", testCorrelated}, {"truth", testTruth},
      {"normalize", testNormalize},   {"prune", testPrune},
      {"steps", testSteps},           {"reproducible", testReproducible},
      {"production", testProduction}, {"line", testLine},
      {"floor", testFloor},           {"degenerate", testDegenerate},
      {"batch", testBatch},           {"manyfits", testManyFits},
      {"components", testComponents}};
  std::string only;
  for (int i = 1; i + 1 < argc; i++)
    if (std::string(argv[i]) == "--case")
      only = argv[i + 1];
  int bad = 0, ran = 0;
  try {
    stream = createPriorityStream(StreamPriority::Diagnostic);
    {
      GMMModel gmm(stream);
      for (const auto& c : cases)
        if (only.empty() || only == c.first) {
          bad += c.second(gmm);
          ran++;
        }
    }
    cudaErrChk(cudaStreamDestroy(stream));
  } catch (const std::exception& error) {
    std::printf("gmmTest failed: %s\n", error.what());
    return 1;
  }
  if (ran == 0) {
    std::printf("unknown case %s\n", only.c_str());
    return 2;
  }
  std::printf("%s\n", bad == 0 ? "PASSED" : "FAILED");
  return bad == 0 ? 0 : 1;
}
