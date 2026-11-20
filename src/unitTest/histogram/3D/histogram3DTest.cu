/**
 * @file histogram3DTest.cu
 * @brief Unit test of velocityHistogram3D: the histogram against a CPU
 *        reference, its reproducibility, and the compacted non-empty bins.
 *
 * The histograms run on a non-blocking stream, as in the analysis. Cases:
 *   - a small set, then two Gaussian populations of 5e6 particles (more than
 *     a batch of the accumulation; the buffers grow), against the CPU, bit
 *     for bit against the integer sums of the CPU, run twice and in another
 *     order bit for bit, compacted;
 *   - no particle; all particles in one bin over two batches; a narrow beam
 *     over a few bins; charges zero or non-finite; non-finite and
 *     out-of-range values, range edges; one huge finite charge (no NaN);
 *   - over the range of the velocities: the Gaussians with infinite and NaN
 *     velocities, bit for bit, twice and in another order, compacted; an
 *     axis without spread; a cold beam, also outside the fixed range (all
 *     its weight in the middle bins); a width that overflows (the fixed
 *     range); no particle; every velocity NaN; one particle;
 *   - the VTK file: header, a point per bin at its centre, big-endian bins.
 */

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#include <unistd.h>

#include "cudaTypeDef.cuh"
#include "dataAnalysisConfig.cuh"
#include "particleArraySoAView.cuh"
#include "velocityHistogram.cuh"

using namespace velocityHistogram;
using namespace DAConfig;

namespace {

/** The fixed range of the histogram on the three axes. */
constexpr cudaCommonType kVmin = -0.2, kVmax = 0.2;
/** Particles of the large cases (more than a batch) and of the others. */
constexpr int kLargeParticles = 5000000;
constexpr int kSmallParticles = 100000;
/** Accepted error of a bin, relative (the float rounding of the output). */
constexpr double kRelTolerance = 1e-6;
/** Accepted error of a bin centre. */
constexpr double kCenterTolerance = 1e-12;
constexpr int kBins[3] = {VELOCITY_HISTOGRAM3D_RES_1,
                          VELOCITY_HISTOGRAM3D_RES_2,
                          VELOCITY_HISTOGRAM3D_RES_3};
/** Particles of the first, small set (the buffers grow after it). */
constexpr int kFirstParticles = 50000;
static_assert(kLargeParticles > kBatchParticles, "two batches");

/** The stream of the histograms: non-blocking, as in the analysis. */
cudaStream_t stream;

struct Particles {
  std::vector<cudaCommonType> u, v, w, q;
  int size() const { return (int)q.size(); }
  void push(cudaCommonType pu, cudaCommonType pv, cudaCommonType pw,
            cudaCommonType pq) {
    u.push_back(pu);
    v.push_back(pv);
    w.push_back(pw);
    q.push_back(pq);
  }
};

/** The binned range of each axis. */
struct Range {
  cudaCommonType lo[3] = {kVmin, kVmin, kVmin}, hi[3] = {kVmax, kVmax, kVmax};
};

/** The range of the finite velocities, the fixed width about their value on
 *  an axis without spread, the fixed range on an axis without finite
 *  velocities or whose width overflows (as the GPU without fixedRange). */
Range dataRange(const Particles& p) {
  Range range;
  const std::vector<cudaCommonType>* axis[3] = {&p.u, &p.v, &p.w};
  for (int d = 0; d < 3; d++) {
    cudaCommonType lo = INFINITY, hi = -INFINITY;
    for (cudaCommonType x : *axis[d])
      if (std::isfinite(x)) {
        lo = std::min(lo, x);
        hi = std::max(hi, x);
      }
    if (lo == hi) {
      hi = lo + (kVmax - kVmin) / 2;
      lo -= (kVmax - kVmin) / 2;
    }
    if (hi - lo > 0 && std::isfinite(hi - lo)) {
      range.lo[d] = lo;
      range.hi[d] = hi;
    }
  }
  return range;
}

/** The bin of particle i as the GPU defines it (times the inverse
 *  resolution, max in the last bin), -1 outside the range or for a
 *  non-finite charge, which weighs 0. */
int cpuBin(const Particles& p, int i, const Range& r) {
  const cudaCommonType velocity[3] = {p.u[i], p.v[i], p.w[i]};
  int bin[3];
  for (int d = 0; d < 3; d++) {
    if (!(velocity[d] >= r.lo[d] && velocity[d] <= r.hi[d]))
      return -1;
    const int b =
        (int)((velocity[d] - r.lo[d]) * (kBins[d] / (r.hi[d] - r.lo[d])));
    bin[d] = std::min(b, kBins[d] - 1);
  }
  return std::isfinite(p.q[i])
             ? bin[0] + kBins[0] * (bin[1] + kBins[1] * bin[2])
             : -1;
}

/** The histogram on the CPU, summed in double. */
std::vector<double> cpuHistogram(const Particles& p, const Range& r = {}) {
  std::vector<double> hist(VELOCITY_HISTOGRAM3D_SIZE, 0.0);
  for (int i = 0; i < p.size(); i++) {
    const int bin = cpuBin(p, i, r);
    if (bin >= 0)
      hist[bin] += std::fabs(p.q[i]) * VELOCITY_WEIGHT_SCALE;
  }
  return hist;
}

/** The histogram of the GPU bit for bit: the same fixed-point weights and
 *  integer sums, on the CPU. */
std::vector<histogramTypeOut> cpuFixedPointHistogram(const Particles& p,
                                                     const Range& r = {}) {
  double largest = 0, count = 0;
  for (cudaCommonType q : p.q)
    if (std::isfinite(q)) {
      largest = std::max(largest, (double)std::fabs(q));
      count += 1;
    }
  // as weightScaleKernel
  double toFixed = kFixedPointOne / largest / count;
  if (!std::isfinite(toFixed))
    toFixed = 0;
  const double toOutput =
      largest / kFixedPointOne * VELOCITY_WEIGHT_SCALE * count;
  std::vector<unsigned long long> sum(VELOCITY_HISTOGRAM3D_SIZE, 0);
  for (int i = 0; i < p.size(); i++) {
    const int bin = cpuBin(p, i, r);
    if (bin >= 0) // __double2ull_rn: to nearest, ties to even
      sum[bin] +=
          (unsigned long long)std::nearbyint(std::fabs(p.q[i]) * toFixed);
  }
  std::vector<histogramTypeOut> hist(VELOCITY_HISTOGRAM3D_SIZE);
  for (int i = 0; i < VELOCITY_HISTOGRAM3D_SIZE; i++)
    hist[i] = (histogramTypeOut)((double)sum[i] * toOutput);
  return hist;
}

/** The GPU histogram of the particles (kVmin, kVmax: the fixed range). */
std::vector<histogramTypeOut> gpuHistogram(velocityHistogram3D& histogram,
                                           const Particles& p) {
  velocitySoA soa(p.size(), 0);
  // the uploads on the stream of the histogram, which does not wait for the
  // legacy stream
  const std::vector<cudaCommonType>* fields[4] = {&p.u, &p.v, &p.w, &p.q};
  for (int f = 0; f < 4; f++)
    if (p.size() > 0)
      cudaErrChk(cudaMemcpyAsync(soa.getElement(f), fields[f]->data(),
                                 p.size() * sizeof(cudaCommonType),
                                 cudaMemcpyHostToDevice, stream));
  histogram.init(&soa, kVmin, kVmax, stream);
  histogram.copyHistogramToHost(stream);
  const histogramTypeOut* host = histogram.getVelocityHistogramHostPtr();
  return std::vector<histogramTypeOut>(host, host + VELOCITY_HISTOGRAM3D_SIZE);
}

bool report(bool ok, const std::string& what) {
  std::cout << "Test " << (ok ? "passed" : "failed") << ": " << what << "\n";
  return ok;
}

bool compare(const std::vector<histogramTypeOut>& gpu,
             const std::vector<double>& cpu, const std::string& what) {
  double maxError = 0;
  int wrong = 0;
  for (int i = 0; i < VELOCITY_HISTOGRAM3D_SIZE; i++) {
    const double error = std::fabs(gpu[i] - cpu[i]);
    maxError = std::max(maxError, cpu[i] > 0 ? error / cpu[i] : error);
    wrong += !(error <= kRelTolerance * cpu[i]); // also NaN
  }
  std::ostringstream detail;
  detail << " (largest relative error " << std::scientific << maxError << ", "
         << wrong << " wrong bins)";
  return report(wrong == 0, what + detail.str());
}

bool sameBits(const std::vector<histogramTypeOut>& a,
              const std::vector<histogramTypeOut>& b, const std::string& what) {
  return report(
      std::memcmp(a.data(), b.data(), a.size() * sizeof(histogramTypeOut)) == 0,
      what);
}

/** Every non-empty bin of the last histogram, in increasing index, with its
 *  centre and content. */
bool checkCompaction(velocityHistogram3D& histogram,
                     const std::vector<histogramTypeOut>& gpu,
                     const Range& r = {}) {
  const int numNonEmpty = histogram.compactNonEmpty(stream);
  std::vector<int> nonEmpty;
  for (int i = 0; i < VELOCITY_HISTOGRAM3D_SIZE; i++)
    if (gpu[i] > 0)
      nonEmpty.push_back(i);
  bool ok = numNonEmpty == (int)nonEmpty.size();
  if (ok && numNonEmpty > 0) {
    std::vector<cudaCommonType> center[3];
    for (int d = 0; d < 3; d++) {
      center[d].resize(numNonEmpty);
      cudaErrChk(cudaMemcpy(center[d].data(), histogram.getNonEmptyCenters()[d],
                            numNonEmpty * sizeof(cudaCommonType),
                            cudaMemcpyDeviceToHost));
    }
    std::vector<histogramTypeOut> weight(numNonEmpty);
    cudaErrChk(cudaMemcpy(weight.data(), histogram.getNonEmptyWeights(),
                          numNonEmpty * sizeof(weight[0]),
                          cudaMemcpyDeviceToHost));
    for (int m = 0; m < numNonEmpty && ok; m++) {
      int index = nonEmpty[m];
      ok = weight[m] == gpu[index];
      for (int d = 0; d < 3; d++) {
        const cudaCommonType expected =
            r.lo[d] + (index % kBins[d] + 0.5) * (r.hi[d] - r.lo[d]) / kBins[d];
        ok = ok && std::fabs(center[d][m] - expected) < kCenterTolerance;
        index /= kBins[d];
      }
    }
  }
  return report(ok, std::to_string(numNonEmpty) +
                        " compacted non-empty bins of " +
                        std::to_string(nonEmpty.size()));
}

/** The VTK file of the last histogram: its header (a point per bin at its
 *  centre) and its bins, big-endian. */
bool checkVtk(velocityHistogram3D& histogram,
              const std::vector<histogramTypeOut>& gpu, const Range& r = {}) {
  const std::filesystem::path dir =
      std::filesystem::temp_directory_path() /
      ("histogram3DTest_vtk_" + std::to_string(getpid()));
  std::filesystem::create_directories(dir);
  histogram.writeToFile((dir / "h_").string(), 7, stream);
  std::ifstream in(dir / "h_3D_7.vtk", std::ios::binary);
  const std::string text((std::istreambuf_iterator<char>(in)),
                         std::istreambuf_iterator<char>());
  in.close();
  std::filesystem::remove_all(dir);
  auto values = [&](const std::string& key) {
    std::vector<double> v(3, NAN);
    const size_t pos = text.find("\n" + key + " ");
    if (pos != std::string::npos) {
      std::istringstream line(text.substr(pos + key.size() + 2));
      line >> v[0] >> v[1] >> v[2];
    }
    return v;
  };
  const std::vector<double> dims = values("DIMENSIONS"),
                            origin = values("ORIGIN"),
                            spacing = values("SPACING");
  bool ok = true;
  for (int d = 0; d < 3; d++) {
    const double res = (r.hi[d] - r.lo[d]) / kBins[d];
    ok = ok && dims[d] == kBins[d] &&
         std::fabs(origin[d] - (r.lo[d] + 0.5 * res)) <= 1e-15 &&
         std::fabs(spacing[d] - res) <= 1e-15;
  }
  const std::string table = "LOOKUP_TABLE default\n";
  const size_t data = text.find(table) + table.size();
  ok = ok && text.find(table) != std::string::npos &&
       text.size() >= data + gpu.size() * sizeof(histogramTypeOut);
  for (size_t i = 0; ok && i < gpu.size(); i++) {
    char bytes[sizeof(histogramTypeOut)];
    for (size_t b = 0; b < sizeof(histogramTypeOut); b++) // big-endian
      bytes[b] = text[data + (i + 1) * sizeof(histogramTypeOut) - 1 - b];
    ok = std::memcmp(bytes, &gpu[i], sizeof(histogramTypeOut)) == 0;
  }
  return report(ok, "the VTK file: header and big-endian bins");
}

/** Two Gaussian populations; random charges. */
Particles twoGaussians(unsigned seed) {
  Particles p;
  std::mt19937 gen(seed);
  std::uniform_real_distribution<cudaCommonType> dis(-1.0, 1.0);
  std::normal_distribution<cudaCommonType> core(0.0, 0.2 * kVmax),
      beam(0.4 * kVmax, 0.1 * kVmax);
  auto draw = [&]() { return dis(gen) > 0.0 ? core(gen) : beam(gen); };
  for (int i = 0; i < kLargeParticles; i++) {
    const cudaCommonType pu = draw(), pv = draw(), pw = draw();
    p.push(pu, pv, pw, -(1.0 + 0.5 * dis(gen)) * 1e-6);
  }
  return p;
}

/** The same particles in another order. */
Particles shuffle(const Particles& p, unsigned seed) {
  std::vector<int> order(p.size());
  for (int i = 0; i < p.size(); i++)
    order[i] = i;
  std::shuffle(order.begin(), order.end(), std::mt19937(seed));
  Particles out;
  for (int i : order)
    out.push(p.u[i], p.v[i], p.w[i], p.q[i]);
  return out;
}

int runTest() {
  stream = createPriorityStream(StreamPriority::Diagnostic);
  velocityHistogram3D histogram(true);
  bool pass = true;

  // a small set first: the scratch and compaction buffers grow after it
  const Particles random = twoGaussians(12345);
  Particles first;
  for (int i = 0; i < kFirstParticles; i++)
    first.push(random.u[i], random.v[i], random.w[i], random.q[i]);
  const auto firstGpu = gpuHistogram(histogram, first);
  pass &= sameBits(firstGpu, cpuFixedPointHistogram(first),
                   "a small set: the integer sums of the CPU give the same "
                   "bits");
  pass &= checkCompaction(histogram, firstGpu);

  // two Gaussian populations, more particles than a batch
  const auto gpu = gpuHistogram(histogram, random);
  pass &= compare(gpu, cpuHistogram(random), "two Gaussians, 5e6 particles");
  pass &= sameBits(gpu, cpuFixedPointHistogram(random),
                   "the integer sums of the CPU give the same bits");
  const auto again = gpuHistogram(histogram, random);
  pass &= sameBits(gpu, again, "two runs give the same bits");
  pass &= checkCompaction(histogram, again);
  pass &= checkVtk(histogram, again);
  pass &= sameBits(gpu, gpuHistogram(histogram, shuffle(random, 4321)),
                   "the particles in another order give the same bits");

  // no particle
  const Particles none;
  const auto empty = gpuHistogram(histogram, none);
  pass &= compare(empty, cpuHistogram(none), "no particle");
  pass &= checkCompaction(histogram, empty);

  // a cold beam: every particle in one bin, over two batches
  Particles cold;
  for (int i = 0; i < kLargeParticles; i++)
    cold.push(0.0123, -0.05, 0.1, -(1.0 + (i % 7) * 0.1) * 1e-6);
  const auto coldGpu = gpuHistogram(histogram, cold);
  pass &= compare(coldGpu, cpuHistogram(cold), "one bin, 5e6 particles");
  pass &= sameBits(coldGpu, cpuFixedPointHistogram(cold),
                   "one bin: the integer sums of the CPU give the same bits");

  // a narrow beam (sigma 3/4 of a bin): a few bins per warp, the shared
  // adds of a bin in contention, with carries between their 32-bit words
  Particles narrow;
  {
    std::mt19937 gen(2468);
    std::normal_distribution<cudaCommonType> beam(0.01, 0.003);
    std::uniform_real_distribution<cudaCommonType> charge(0.5e-6, 1.5e-6);
    for (int i = 0; i < kLargeParticles; i++)
      narrow.push(beam(gen), beam(gen), beam(gen), -charge(gen));
  }
  const auto narrowGpu = gpuHistogram(histogram, narrow);
  pass &= compare(narrowGpu, cpuHistogram(narrow), "a narrow beam");
  pass &= sameBits(narrowGpu, cpuFixedPointHistogram(narrow),
                   "a narrow beam: the integer sums of the CPU give the same "
                   "bits");

  // charges all zero or non-finite: nothing to weigh
  Particles uncharged;
  for (int i = 0; i < kSmallParticles; i++)
    uncharged.push(0.01, 0.02, 0.03,
                   i % 3 ? 0.0 : std::numeric_limits<double>::quiet_NaN());
  pass &= compare(gpuHistogram(histogram, uncharged), cpuHistogram(uncharged),
                  "charges zero or non-finite");

  // non-finite and out-of-range velocities, non-finite charges, the edges
  Particles bad;
  {
    const cudaCommonType nan = std::numeric_limits<cudaCommonType>::quiet_NaN();
    const cudaCommonType inf = std::numeric_limits<cudaCommonType>::infinity();
    std::mt19937 gen(678);
    std::uniform_real_distribution<cudaCommonType> inRange(kVmin, kVmax);
    for (int i = 0; i < kSmallParticles; i++) {
      cudaCommonType pv[3] = {inRange(gen), inRange(gen), inRange(gen)};
      cudaCommonType pq = -1e-6;
      const bool upper = (i / 8) % 2;
      switch (i % 8) {
      case 0:
        pv[i % 3] = nan;
        break;
      case 1:
        pv[i % 3] = upper ? inf : -inf;
        break;
      case 2:
        pv[i % 3] = upper ? 1.5 * kVmax : 1.5 * kVmin;
        break;
      case 3:
        pq = nan;
        break;
      case 4:
        pq = upper ? inf : -inf;
        break;
      case 5:
        pv[i % 3] = kVmax; // max is in the last bin
        break;
      case 6:
        pv[i % 3] = kVmin;
        break;
      default:
        break;
      }
      bad.push(pv[0], pv[1], pv[2], pq);
    }
  }
  pass &= compare(gpuHistogram(histogram, bad), cpuHistogram(bad),
                  "non-finite and out-of-range values, range edges");

  // one huge finite charge: the others round to 0 (the fixed point), its bin
  // overflows the float output; no NaN anywhere
  Particles huge = first;
  huge.q[0] = -1e300;
  const auto hugeGpu = gpuHistogram(histogram, huge);
  bool finiteOrInf = true;
  for (histogramTypeOut x : hugeGpu)
    finiteOrInf = finiteOrInf && !std::isnan(x);
  pass &= report(finiteOrInf, "one huge charge: no NaN");
  pass &= sameBits(hugeGpu, cpuFixedPointHistogram(huge),
                   "one huge charge: the integer sums of the CPU give the "
                   "same bits");

  // over the range of the velocities: infinite velocities left out of it, and
  // an axis without spread on the fixed range
  velocityHistogram3D ranged(false);
  Particles wide = twoGaussians(999);
  for (int i = 0; i < wide.size(); i += 1000) {
    wide.u[i] = i % 2000 ? std::numeric_limits<double>::infinity() : 0.9;
    wide.w[i] = std::numeric_limits<double>::quiet_NaN();
  }
  const Range range = dataRange(wide);
  const auto rangedGpu = gpuHistogram(ranged, wide);
  pass &=
      compare(rangedGpu, cpuHistogram(wide, range), "range of the velocities");
  pass &= sameBits(rangedGpu, cpuFixedPointHistogram(wide, range),
                   "range of the velocities: the integer sums of the CPU "
                   "give the same bits");
  pass &= sameBits(rangedGpu, gpuHistogram(ranged, wide),
                   "range of the velocities: two runs give the same bits");
  pass &= checkCompaction(ranged, rangedGpu, range);
  pass &= checkVtk(ranged, rangedGpu, range);
  pass &= sameBits(rangedGpu, gpuHistogram(ranged, shuffle(wide, 8765)),
                   "range of the velocities: the particles in another order "
                   "give the same bits");
  for (int i = 0; i < wide.size(); i++)
    wide.v[i] = 0.05;
  const Range noSpread = dataRange(wide);
  const auto noSpreadGpu = gpuHistogram(ranged, wide);
  pass &= compare(noSpreadGpu, cpuHistogram(wide, noSpread),
                  "range of the velocities, an axis without spread");
  pass &= checkCompaction(ranged, noSpreadGpu, noSpread);
  pass &= sameBits(gpuHistogram(ranged, cold),
                   cpuFixedPointHistogram(cold, dataRange(cold)),
                   "range of the velocities, a cold beam: one bin, bit for "
                   "bit");
  // a cold beam outside the fixed range: every particle in a middle bin of
  // the fixed width about its velocity (the rounding of the range picks one
  // of the two)
  Particles drifting = cold;
  for (int i = 0; i < drifting.size(); i++)
    drifting.w[i] = 2 * kVmax;
  const auto driftingGpu = gpuHistogram(ranged, drifting);
  double total = 0, expected = 0;
  bool middle = true;
  for (int i = 0; i < VELOCITY_HISTOGRAM3D_SIZE; i++) {
    total += driftingGpu[i];
    middle &= driftingGpu[i] == 0 ||
              std::abs(i / (kBins[0] * kBins[1]) - kBins[2] / 2 + 0.5) < 1;
  }
  for (cudaCommonType q : drifting.q)
    expected += std::fabs(q) * VELOCITY_WEIGHT_SCALE;
  pass &=
      report(middle && std::fabs(total - expected) <= kRelTolerance * expected,
             "range of the velocities, a cold beam outside the fixed "
             "range: all its weight, in the middle bins of w");
  // a width that overflows: the fixed range on that axis
  Particles overflowing = first;
  overflowing.u[0] = 1e308;
  overflowing.u[1] = -1e308;
  const Range overflow = dataRange(overflowing);
  pass &= report(overflow.lo[0] == kVmin && overflow.hi[0] == kVmax,
                 "a width that overflows: the reference takes the fixed "
                 "range");
  pass &= sameBits(gpuHistogram(ranged, overflowing),
                   cpuFixedPointHistogram(overflowing, overflow),
                   "range of the velocities, a width that overflows: the "
                   "fixed range on that axis, bit for bit");

  // the range of the velocities of no particle and of NaN velocities: the
  // fixed range on every axis; of one particle: the fixed width about it
  pass &= compare(gpuHistogram(ranged, none), cpuHistogram(none),
                  "range of the velocities, no particle");
  Particles undefined = bad;
  for (int i = 0; i < undefined.size(); i++)
    undefined.u[i] = undefined.v[i] = undefined.w[i] =
        std::numeric_limits<cudaCommonType>::quiet_NaN();
  pass &= compare(gpuHistogram(ranged, undefined), cpuHistogram(undefined),
                  "range of the velocities, every velocity NaN");
  Particles single;
  single.push(0.05, -0.1, 0.15, -1e-6);
  pass &= compare(gpuHistogram(ranged, single),
                  cpuHistogram(single, dataRange(single)),
                  "range of the velocities, one particle");

  return pass ? 0 : 1;
}

} // namespace

int main() {
  try {
    return runTest();
  } catch (const std::exception& error) {
    std::cerr << "histogram3DTest failed: " << error.what() << "\n";
    return EXIT_FAILURE;
  }
}
