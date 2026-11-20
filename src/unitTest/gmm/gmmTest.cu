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
 *               by 2: the same components and log-likelihood in physical
 *               units, and the points given back
 *   prune       the same data with a third component started far from the
 *               data, fitted on normalised velocities: the third component is
 *               pruned and the other two are recovered in physical units
 *   steps       the prune case stopped by the iteration limit at the first
 *               pruning: one more step, weights summing to 1
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
 *               weights kept); a NaN mean (reset); a non-positive start
 *               without the covariance check; the JSON of a non-finite fit;
 *               no point and too many components refused
 */

#include <algorithm>
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
  real cov[9]; // row major
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
  std::vector<real> point[3];
  std::vector<weight_t> weight;
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
 * Fit the histogram starting from `start` (in physical units) with the given
 * GMM; with `scale` > 0 the fit runs on the velocities divided by it
 * (GMMNormalize).
 */
Result fit(GMMModel& gmm, const Histogram& h,
           const std::vector<Gaussian>& start, real scale, bool prune,
           int maxIterations = kMaxIterations, bool checkCovariance = true,
           real threshold = kThreshold) {
  const int n = (int)h.size(), numComponents = (int)start.size();
  // the uploads on the stream of the fit, which does not wait for the
  // legacy stream
  real* point[3];
  for (int d = 0; d < 3; d++) {
    cudaErrChk(cudaMalloc(&point[d], n * sizeof(real)));
    cudaErrChk(cudaMemcpyAsync(point[d], h.point[d].data(), n * sizeof(real),
                               cudaMemcpyHostToDevice, stream));
  }
  weight_t* weight;
  cudaErrChk(cudaMalloc(&weight, n * sizeof(weight_t)));
  cudaErrChk(cudaMemcpyAsync(weight, h.weight.data(), n * sizeof(weight_t),
                             cudaMemcpyHostToDevice, stream));

  std::vector<real> w(numComponents), mean(numComponents * 3),
      cov(numComponents * 9);
  for (int c = 0; c < numComponents; c++) {
    w[c] = start[c].weight;
    for (int d = 0; d < 3; d++)
      mean[c * 3 + d] = start[c].mean[d];
    for (int e = 0; e < 9; e++)
      cov[c * 9 + e] = start[c].cov[e];
  }
  cudaGMMWeight::GMMParam_t<real> param = {.numComponents = numComponents,
                                           .maxIteration = maxIterations,
                                           .threshold = threshold,
                                           .weightInit = w.data(),
                                           .meanInit = mean.data(),
                                           .coVarianceInit = cov.data(),
                                           .normalize = scale > 0,
                                           // the points lie in [-1, 1]
                                           .scale = scale > 0 ? scale : 1,
                                           .checkCovariance = checkCovariance,
                                           .covarianceTolerance =
                                               kCovarianceTolerance,
                                           .minVariance = kMinVariance,
                                           .prune = prune,
                                           .pruneThreshold = kPruneThreshold};

  GMMPoints data(n, point, weight);
  Result result;
  gmm.config(&param, &data);
  result.steps = gmm.initGMM();
  auto r = gmm.getGMMResult(0, result.steps);
  result.logLikelihood = r.logLikelihoodFinal;
  result.pointsError = 0;
  std::vector<real> back(n);
  for (int d = 0; d < 3; d++) {
    cudaErrChk(cudaMemcpyAsync(back.data(), point[d], n * sizeof(real),
                               cudaMemcpyDeviceToHost, stream));
    cudaErrChk(cudaStreamSynchronize(stream));
    for (int i = 0; i < n; i++)
      result.pointsError =
          std::max(result.pointsError, std::fabs(back[i] - h.point[d][i]));
  }
  for (int c = 0; c < numComponents; c++) {
    Gaussian g;
    g.weight = r.weight[c];
    for (int d = 0; d < 3; d++)
      g.mean[d] = r.mean[c * 3 + d];
    for (int e = 0; e < 9; e++)
      g.cov[e] = r.coVariance[c * 9 + e];
    result.components.push_back(g);
  }
  for (int d = 0; d < 3; d++)
    cudaErrChk(cudaFree(point[d]));
  cudaErrChk(cudaFree(weight));
  return result;
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

int testOneStep(GMMModel& gmm, int grid) {
  failures = 0;
  const std::vector<Gaussian> truth = twoComponents();
  const Histogram h = histogramOf(truth, false, grid);
  std::vector<Gaussian> start = {startNear(truth[0], 0.1),
                                 startNear(truth[1], -0.1)};
  // correlated starts: the off-diagonal terms of the factor in the E step
  start[0].cov[1] = start[0].cov[3] = 0.02;
  start[1].cov[5] = start[1].cov[7] = -0.015;
  real hostLogLikelihood;
  const std::vector<Gaussian> host = hostEMStep(h, start, hostLogLikelihood);
  const Result r = fit(gmm, h, start, 0, false, 1);
  real err = 0;
  for (size_t c = 0; c < host.size(); c++) {
    const Gaussian &a = r.components[c], &b = host[c];
    err = std::max(err, std::fabs(a.weight - b.weight) / b.weight);
    for (int d = 0; d < 3; d++)
      err = std::max(err, std::fabs(a.mean[d] - b.mean[d]));
    for (int e = 0; e < 9; e++)
      err = std::max(err, std::fabs(a.cov[e] - b.cov[e]) / b.cov[0]);
  }
  std::printf("  %zu points\n", h.size());
  if (grid == kOneStepLargeGrid)
    check((int)h.size() > kStatisticsPass,
          "more points than a pass of the statistics", h.size());
  check(r.steps == 1, "one EM step", r.steps);
  check(err < kOneStepError, "the EM step of the host", err);
  check(std::fabs(r.logLikelihood - hostLogLikelihood) <
            kOneStepError * std::fabs(hostLogLikelihood),
        "the log-likelihood of the host",
        std::fabs(r.logLikelihood / hostLogLikelihood - 1));
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
  const Result scaled = fit(gmm, h, start, 2.0, false, kNormalizeSteps);
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
  check(r.steps == cudaGMMWeight::kMinStepsToConverge + 1,
        "the least steps before the convergence", r.steps);
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
  compare({r.components[0], r.components[1]}, truth);
  check(std::fabs(r.weightSum() - 1) < kWeightSumError, "weights sum to 1",
        r.weightSum() - 1);
  return failures;
}

int testSteps(GMMModel& gmm) {
  std::printf("steps\n");
  failures = 0;
  const std::vector<Gaussian> truth = twoComponents();
  const int limit = cudaGMMWeight::kPruneInterval;
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

  // a NaN mean: reset, then fitted
  const Histogram h = histogramOf(twoComponents(), false);
  std::vector<Gaussian> broken = three;
  broken[2].mean[1] = NAN;
  const Result m = fit(gmm, h, broken, 0, false);
  bool meansFinite = std::isfinite(m.logLikelihood);
  for (const Gaussian& g : m.components)
    for (int d = 0; d < 3; d++)
      meansFinite = meansFinite && std::isfinite(g.mean[d]);
  check(meansFinite, "a NaN mean: reset, the fit finite", m.logLikelihood);

  // no covariance check: a start that is not positive definite (|cov uv| >
  // the variances) is factored with the pivot floor: one EM step stays
  // finite, with positive covariances
  std::vector<Gaussian> skew = three;
  skew[0].cov[1] = skew[0].cov[3] = 0.08;
  const Result c = fit(gmm, h, skew, 0, false, 1, false);
  bool positiveFit = std::isfinite(c.logLikelihood);
  for (const Gaussian& g : c.components)
    positiveFit = positiveFit && positiveDefinite(g);
  check(positiveFit, "no covariance check: a non-positive start fits finite",
        c.logLikelihood);

  // the JSON of a non-finite fit: null, not nan or inf
  cudaGMMWeight::GMMResult<real, 3> broken3(0, 1);
  broken3.weight[0] = NAN;
  broken3.logLikelihoodFinal = INFINITY;
  broken3.convergeStep = 1;
  for (int e = 0; e < 3; e++)
    broken3.mean[e] = 0;
  for (int e = 0; e < 9; e++)
    broken3.coVariance[e] = e % 4 ? 0 : 1;
  const std::string json = broken3.outputString();
  check(json.find("null") != std::string::npos &&
            json.find("nan") == std::string::npos &&
            json.find("inf") == std::string::npos,
        "the JSON of a non-finite fit has null", 0);

  // no point, too many components: refused
  auto refused = [&](int numComponents, int numPoints) {
    std::vector<real> w(numComponents, 1.0 / numComponents),
        mean(numComponents * 3, 0), cov(numComponents * 9, 0);
    cudaGMMWeight::GMMParam_t<real> param = {.numComponents = numComponents,
                                             .maxIteration = 1,
                                             .threshold = kThreshold,
                                             .weightInit = w.data(),
                                             .meanInit = mean.data(),
                                             .coVarianceInit = cov.data(),
                                             .normalize = false,
                                             .scale = 1,
                                             .checkCovariance = true,
                                             .covarianceTolerance =
                                                 kCovarianceTolerance,
                                             .minVariance = kMinVariance,
                                             .prune = false,
                                             .pruneThreshold = kPruneThreshold};
    real* none[3] = {nullptr, nullptr, nullptr};
    GMMPoints data(numPoints, none, nullptr);
    try {
      gmm.config(&param, &data);
    } catch (const std::invalid_argument&) {
      return true;
    }
    return false;
  };
  check(refused(2, 0), "no point refused", 0);
  check(refused(cudaGMMWeight::kGMMThreads + 1, 10),
        "more components than threads refused", 0);
  return failures;
}

/** The fits are equal bit for bit. */
bool sameFit(const Result& a, const Result& b) {
  bool same = a.steps == b.steps && a.logLikelihood == b.logLikelihood &&
              a.components.size() == b.components.size();
  for (size_t c = 0; same && c < a.components.size(); c++)
    same =
        std::memcmp(&a.components[c], &b.components[c], sizeof(Gaussian)) == 0;
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

int testProduction(GMMModel& gmm) {
  std::printf("production\n");
  failures = 0;
  const std::vector<Gaussian> truth = twoComponents();
  const Histogram h = histogramOf(truth, false);
  // means drawn in the unit ball, diagonal covariances, equal weights
  std::mt19937 gen(2024);
  std::uniform_real_distribution<real> uniform(-1, 1);
  std::vector<Gaussian> start(kProductionComponents);
  for (Gaussian& g : start) {
    g = {1.0 / kProductionComponents, {}, {}};
    real r2;
    do {
      r2 = 0;
      for (int d = 0; d < 3; d++) {
        g.mean[d] = uniform(gen);
        r2 += g.mean[d] * g.mean[d];
      }
    } while (r2 > 1);
    for (int d = 0; d < 3; d++)
      g.cov[d * 4] = 0.04;
  }
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

} // namespace

int main(int argc, char** argv) {
  const std::pair<const char*, int (*)(GMMModel&)> cases[] = {
      {"onestep", testOneStepSmall},  {"onesteplarge", testOneStepLarge},
      {"correlated", testCorrelated}, {"truth", testTruth},
      {"normalize", testNormalize},   {"prune", testPrune},
      {"steps", testSteps},           {"reproducible", testReproducible},
      {"production", testProduction}, {"line", testLine},
      {"floor", testFloor},           {"degenerate", testDegenerate}};
  std::string only;
  for (int i = 1; i + 1 < argc; i++)
    if (std::string(argv[i]) == "--case")
      only = argv[i + 1];
  int bad = 0, ran = 0;
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
  if (ran == 0) {
    std::printf("unknown case %s\n", only.c_str());
    return 2;
  }
  std::printf("%s\n", bad == 0 ? "PASSED" : "FAILED");
  return bad == 0 ? 0 : 1;
}
