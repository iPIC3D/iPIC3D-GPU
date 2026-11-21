/**
 * @file histogram3DTest.cu
 * @brief Unit test of the velocity histograms of a rank (velocityHistogram3D)
 *        and of its macrocells (macrocellHistogram3D, the tiling and its
 *        slices): against a CPU reference, their reproducibility, and the
 *        compacted non-empty bins.
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
 *     its weight in the middle bin), and spread over a few units in the
 *     last place (no spread: the same histogram); a width that overflows
 *     (the fixed range); no particle; every velocity NaN (the fixed range);
 *     one particle;
 *   - the records of the non-empty bins (32-bit index lists or bitmaps) and
 *     their restart;
 *   - macrocell histograms (macrocellHistogram3D), bit for bit against the
 *     integer sums of the CPU per macrocell: particles sorted by cell over a
 *     3D tiling with remainders, the particles of the ghost cells left out
 *     (a dense cell, a cold beam, an empty macrocell, non-finite and
 *     out-of-range values), a 2D one and a rank with fewer cells than a
 *     macrocell along two axes (a truncated last rank: one macrocell of its
 *     cells) and a rank of 288 macrocells (more compaction blocks than the
 *     threads of its scan), on the fixed range and on the range of each
 *     macrocell, in another order within the cells; no particle, also with
 *     the cells of a stale sort; the grids, the compaction by region, the
 *     records of the non-empty bins (16-bit index lists or bitmaps), none
 *     without one, and their restart;
 *   - the tiling of every extent up to 40 cells in macrocells of every size
 *     up to 45 and its slices, and the macrocell histograms of ranks of many
 *     shapes (one cell, one cell thick, macrocells of one cell, of the rank,
 *     of one cell less, larger than the rank; one block per macrocell or
 *     several, in slices of its cells, when the macrocells are few) bit for
 *     bit against the CPU.
 */

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <limits>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#include <unistd.h>

#include "cudaTypeDef.cuh"
#include "dataAnalysisConfig.cuh"
#include "macrocellPartition.cuh"
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
/** Bins of a macrocell histogram along each axis. */
constexpr int kMacrocellBins3[3] = {MACROCELL_HISTOGRAM3D_RES_1,
                                    MACROCELL_HISTOGRAM3D_RES_2,
                                    MACROCELL_HISTOGRAM3D_RES_3};
/** The ranks of the macrocell cases: interior cells and macrocell size (the
 *  3D one with remainders on every axis), the particles of a cell at most,
 *  of the dense cell and of the cold-beam cell. */
constexpr int kRank3DInterior[3] = {7, 5, 3}, kRank3DSize[3] = {3, 2, 2};
constexpr int kRank2DInterior[3] = {8, 6, 1}, kRank2DSize[3] = {4, 3, 1};
constexpr int kSmallRankInterior[3] = {5, 3, 2}, kSmallRankSize[3] = {4, 4, 4};
constexpr int kManyRankInterior[3] = {24, 24, 4}, kManyRankSize[3] = {2, 2, 2};
constexpr int kManyCellParticles = 20;
/** The tiling sweep: every interior up to this many cells, every macrocell
 *  size up to this (above the interior: one macrocell). */
constexpr int kSweepMaxInterior = 40, kSweepMaxSize = 45;
/** The ranks of the shapes sweep, {interior, macrocell size}: one cell, a
 *  rank one cell thick, a macrocell of one cell, of the whole rank, of one
 *  cell less (a last macrocell of one cell), above the rank on some or every
 *  axis, and 1600 macrocells (one slice each on a GPU of up to 400
 *  multiprocessors); and the particles of a cell at most. */
constexpr int kShapes[][2][3] = {
    {{1, 1, 1}, {1, 1, 1}},  {{1, 1, 1}, {4, 4, 4}},  {{9, 1, 1}, {4, 1, 1}},
    {{1, 7, 1}, {1, 3, 1}},  {{1, 1, 6}, {1, 1, 5}},  {{6, 5, 4}, {6, 5, 4}},
    {{6, 5, 4}, {5, 4, 3}},  {{6, 5, 4}, {1, 5, 2}},  {{6, 5, 4}, {7, 2, 9}},
    {{13, 2, 3}, {4, 3, 1}}, {{3, 11, 2}, {3, 1, 2}}, {{40, 40, 1}, {1, 1, 1}}};
constexpr int kShapeCellParticles = 12;
constexpr int kCellParticles = 60;
constexpr int kDenseCellParticles = 5000;
constexpr int kColdCellParticles = 2000;
/** The first particle and the particles of every cell of a stale sort. */
constexpr int kStaleCell = 1 << 28;
/** The macrocells of the 3D case: without particles, of a cold beam, of NaN
 *  velocities, of charges 1e-9 of the others; and its dense cell, guarded
 *  (2, 2, 1), in macrocell 0. */
constexpr int kEmptyMacrocell = 4, kColdMacrocell = 7, kNanMacrocell = 13;
constexpr int kTinyMacrocell = 2;
constexpr int kDenseCell =
    2 + (kRank3DInterior[0] + 2) * (2 + (kRank3DInterior[1] + 2) * 1);
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

/** The range of the finite velocities over the given bins, the fixed width
 *  with their middle at the centre of bin bins / 2 on an axis without spread
 *  (bins narrower than kMinRelativeBinWidth of the largest |velocity|), the
 *  fixed range on an
 *  axis without finite velocities or whose width or inverse resolution
 *  overflows (as the GPU without fixedRange). */
Range dataRange(const Particles& p, const int* bins = kBins) {
  Range range;
  const std::vector<cudaCommonType>* axis[3] = {&p.u, &p.v, &p.w};
  for (int d = 0; d < 3; d++) {
    cudaCommonType lo = INFINITY, hi = -INFINITY;
    for (cudaCommonType x : *axis[d])
      if (std::isfinite(x)) {
        lo = std::min(lo, x);
        hi = std::max(hi, x);
      }
    if (lo > hi) // none finite
      continue;
    if (!(hi - lo > bins[d] * kMinRelativeBinWidth * std::max(hi, -lo))) {
      // the middle in the centre of bin bins / 2
      const cudaCommonType width = kVmax - kVmin;
      const cudaCommonType middle = lo + (hi - lo) / 2;
      lo = middle - width / 2 - (bins[d] % 2 ? 0 : width / (2 * bins[d]));
      hi = lo + width;
    }
    if (hi - lo > 0 && std::isfinite(hi - lo) &&
        std::isfinite(bins[d] / (hi - lo))) {
      range.lo[d] = lo;
      range.hi[d] = hi;
    }
  }
  return range;
}

/** The bin of particle i as the GPU defines it (times the inverse
 *  resolution, max in the last bin), -1 outside the range or for a
 *  non-finite charge, which weighs 0. */
int cpuBin(const Particles& p, int i, const Range& r, const int* bins = kBins) {
  const cudaCommonType velocity[3] = {p.u[i], p.v[i], p.w[i]};
  int bin[3];
  for (int d = 0; d < 3; d++) {
    if (!(velocity[d] >= r.lo[d] && velocity[d] <= r.hi[d]))
      return -1;
    const int b =
        (int)((velocity[d] - r.lo[d]) * (bins[d] / (r.hi[d] - r.lo[d])));
    bin[d] = std::min(b, bins[d] - 1);
  }
  return std::isfinite(p.q[i]) ? bin[0] + bins[0] * (bin[1] + bins[1] * bin[2])
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

/** The fixed-point scale of the charges of the particles, as
 *  fixedPointScale: to fixed point and to the output. */
void cpuWeightScale(const Particles& p, double& toFixed, double& toOutput) {
  double largest = 0, count = 0;
  for (cudaCommonType q : p.q)
    if (std::isfinite(q)) {
      largest = std::max(largest, (double)std::fabs(q));
      count += 1;
    }
  toFixed = kFixedPointOne / largest / count;
  if (!std::isfinite(toFixed))
    toFixed = 0;
  toOutput = largest / kFixedPointOne * VELOCITY_WEIGHT_SCALE * count;
}

/** The histogram of the GPU bit for bit: the same fixed-point weights and
 *  integer sums, on the CPU. */
std::vector<histogramTypeOut> cpuFixedPointHistogram(const Particles& p,
                                                     const Range& r = {},
                                                     const int* bins = kBins) {
  double toFixed, toOutput;
  cpuWeightScale(p, toFixed, toOutput);
  std::vector<unsigned long long> sum((size_t)bins[0] * bins[1] * bins[2], 0);
  for (int i = 0; i < p.size(); i++) {
    const int bin = cpuBin(p, i, r, bins);
    if (bin >= 0) // __double2ull_rn: to nearest, ties to even
      sum[bin] +=
          (unsigned long long)std::nearbyint(std::fabs(p.q[i]) * toFixed);
  }
  std::vector<histogramTypeOut> hist(sum.size());
  for (size_t i = 0; i < sum.size(); i++)
    hist[i] = (histogramTypeOut)((double)sum[i] * toOutput);
  return hist;
}

/** The particles into soa, on the stream of the histograms, which does not
 *  wait for the legacy stream. */
void upload(velocitySoA& soa, const Particles& p) {
  const std::vector<cudaCommonType>* fields[4] = {&p.u, &p.v, &p.w, &p.q};
  for (int f = 0; f < 4; f++)
    if (p.size() > 0)
      cudaErrChk(cudaMemcpyAsync(soa.getElement(f), fields[f]->data(),
                                 p.size() * sizeof(cudaCommonType),
                                 cudaMemcpyHostToDevice, stream));
}

/** The first bins of the last init() of h, on the host. */
std::vector<histogramTypeOut> hostBins(histogramRegions& h, size_t bins) {
  std::vector<histogramTypeOut> host(bins);
  cudaErrChk(cudaMemcpyAsync(host.data(), h.deviceBins(),
                             sizeof(histogramTypeOut) * bins,
                             cudaMemcpyDeviceToHost, stream));
  cudaErrChk(cudaStreamSynchronize(stream));
  return host;
}

/** The GPU histogram of the particles (kVmin, kVmax: the fixed range). */
std::vector<histogramTypeOut> gpuHistogram(velocityHistogram3D& histogram,
                                           const Particles& p) {
  velocitySoA soa(p.size(), 0);
  upload(soa, p);
  histogram.init(&soa, kVmin, kVmax, stream);
  return hostBins(histogram, VELOCITY_HISTOGRAM3D_SIZE);
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
  return report(a.size() == b.size() &&
                    std::memcmp(a.data(), b.data(),
                                a.size() * sizeof(histogramTypeOut)) == 0,
                what);
}

/** The non-empty bins of the last init() of h: region after region (the
 *  offsets of the regions), in increasing index, their contents, and their
 *  centres on the ranges of their regions (bins per axis). */
bool checkCompaction(histogramRegions& h,
                     const std::vector<histogramTypeOut>& gpu,
                     const std::vector<Range>& ranges, const int* bins) {
  const int regionBins = bins[0] * bins[1] * bins[2];
  const int total = h.compactNonEmpty(stream);
  const std::vector<int>& offsets = h.getRegionOffsets();
  std::vector<cudaCommonType> center[3];
  std::vector<histogramTypeOut> weight(total);
  histogramTypeIn* const* centers =
      total > 0 ? h.getNonEmptyCenters(stream) : nullptr;
  for (int d = 0; d < 3; d++) {
    center[d].resize(total);
    if (total > 0)
      cudaErrChk(cudaMemcpy(center[d].data(), centers[d],
                            total * sizeof(cudaCommonType),
                            cudaMemcpyDeviceToHost));
  }
  if (total > 0)
    cudaErrChk(cudaMemcpy(weight.data(), h.getNonEmptyWeights(),
                          total * sizeof(histogramTypeOut),
                          cudaMemcpyDeviceToHost));
  bool ok = (int)offsets.size() == h.numRegions() + 1 && offsets[0] == 0;
  int next = 0;
  for (int m = 0; ok && m < h.numRegions(); m++) {
    ok = offsets[m] == next;
    for (int b = 0; ok && b < regionBins; b++) {
      const histogramTypeOut value = gpu[(size_t)m * regionBins + b];
      if (!(value > 0))
        continue;
      ok = next < total && weight[next] == value;
      int index = b;
      for (int d = 0; d < 3; d++) {
        const Range& r = ranges[m];
        const cudaCommonType expected =
            r.lo[d] + (index % bins[d] + 0.5) * (r.hi[d] - r.lo[d]) / bins[d];
        ok = ok && std::fabs(center[d][next] - expected) < kCenterTolerance;
        index /= bins[d];
      }
      next++;
    }
  }
  ok = ok && offsets.back() == total && next == total;
  return report(ok, std::to_string(total) +
                        " compacted non-empty bins, region after region");
}

/** A temporary folder of the test, removed with the object (also when a
 *  check throws). */
struct TempDir {
  std::filesystem::path path;
  explicit TempDir(const std::string& name)
      : path(std::filesystem::temp_directory_path() /
             (name + "_" + std::to_string(getpid()))) {
    std::filesystem::create_directories(path);
  }
  ~TempDir() {
    std::error_code error;
    std::filesystem::remove_all(path, error);
  }
};

/** A record of a histograms file, read back (ok: whole). */
struct ReadRecord {
  sparseBins::SparseRecordHeader header;
  std::vector<double> ranges;
  std::vector<std::uint32_t> counts;
  std::vector<histogramTypeOut> weights;
  std::vector<std::uint32_t> bins;
  bool ok = false;
};

/** The bins of the weights of a record from its index bytes (sparseIndex.h):
 *  per region its list or its bitmap; false when they do not add up. */
bool decodeIndex(const std::vector<std::uint32_t>& counts,
                 const std::vector<std::uint8_t>& index, int regionBins,
                 std::vector<std::uint32_t>& bins) {
  const size_t listBytes = sparseBins::binIndexBytes(regionBins),
               bitmapBytes = sparseBins::bitmapBytes(regionBins);
  size_t at = 0;
  for (const std::uint32_t count : counts) {
    if (sparseBins::isBitmap(count, regionBins)) {
      if (at + bitmapBytes > index.size())
        return false;
      std::uint32_t found = 0;
      for (int b = 0; b < regionBins; b++)
        if (index[at + b / sparseBins::kBinsPerBitmapByte] >>
                (b % sparseBins::kBinsPerBitmapByte) &
            1) {
          bins.push_back(b);
          found++;
        }
      if (found != count)
        return false;
      at += bitmapBytes;
      continue;
    }
    for (std::uint32_t i = 0; i < count; i++, at += listBytes) {
      if (at + listBytes > index.size())
        return false;
      std::uint16_t shortBin;
      std::uint32_t bin;
      if (listBytes == 2) {
        std::memcpy(&shortBin, &index[at], listBytes);
        bin = shortBin;
      } else {
        std::memcpy(&bin, &index[at], listBytes);
      }
      bins.push_back(bin);
    }
  }
  return at == index.size();
}

/** The records of a histograms file of regions of regionBins bins, up to
 *  the first that is not whole. */
std::vector<ReadRecord> readRecords(const std::string& path, int regionBins) {
  std::ifstream in(path, std::ios::binary);
  const std::string data((std::istreambuf_iterator<char>(in)),
                         std::istreambuf_iterator<char>());
  std::vector<ReadRecord> records;
  for (size_t at = 0; at + sizeof(ReadRecord::header) <= data.size();) {
    ReadRecord r;
    std::memcpy(&r.header, data.data() + at, sizeof(r.header));
    const size_t R = r.header.regions, n = r.header.nonEmpty;
    const size_t arrays = sizeof(r.header) + (6 * sizeof(double) + 4) * R +
                          sizeof(histogramTypeOut) * n + r.header.indexBytes;
    // its arrays and the padding to the alignment, nothing more
    const size_t padded =
        (arrays + kRecordAlignment - 1) / kRecordAlignment * kRecordAlignment;
    if (r.header.bytes > data.size() - at || r.header.bytes != padded)
      break;
    size_t offset = at + sizeof(r.header);
    auto take = [&](auto& values, size_t count) {
      values.resize(count);
      std::memcpy(values.data(), data.data() + offset,
                  count * sizeof(values[0]));
      offset += count * sizeof(values[0]);
    };
    take(r.ranges, 6 * R);
    take(r.counts, R);
    take(r.weights, n);
    std::vector<std::uint8_t> index;
    take(index, r.header.indexBytes);
    r.ok = decodeIndex(r.counts, index, regionBins, r.bins);
    records.push_back(r);
    at += r.header.bytes;
  }
  return records;
}

/** A record holds the histograms of the GPU: its header, the range and the
 *  non-empty bins of every region, their contents and indices. */
bool sameRecord(const ReadRecord& r, int cycle, int species,
                const std::vector<histogramTypeOut>& gpu,
                const std::vector<Range>& ranges, int regionBins) {
  const int R = (int)ranges.size();
  bool ok = r.ok && r.header.magic == kHistogramRecordMagic &&
            r.header.cycle == cycle && r.header.species == species &&
            (int)r.header.regions == R &&
            r.header.bytes % kRecordAlignment == 0;
  size_t next = 0;
  for (int m = 0; ok && m < R; m++) {
    for (int e = 0; e < 3; e++)
      ok = ok && r.ranges[6 * m + e] == ranges[m].lo[e] &&
           r.ranges[6 * m + 3 + e] == ranges[m].hi[e];
    std::uint32_t count = 0;
    for (int b = 0; ok && b < regionBins; b++) {
      const histogramTypeOut value = gpu[(size_t)m * regionBins + b];
      if (!(value > 0))
        continue;
      ok = next < r.weights.size() && r.weights[next] == value &&
           r.bins[next] == (std::uint32_t)b;
      next++;
      count++;
    }
    ok = ok && r.counts[m] == count;
  }
  return ok && next == r.header.nonEmpty;
}

/** The records of the last init() of h (ranges: of its regions): two
 *  records read back and the JSON, none without a non-empty bin; on a
 *  restart the same layout continued without the records from its cycle on,
 *  another kept aside. */
bool checkRecords(histogramRegions& h, const std::vector<histogramTypeOut>& gpu,
                  const std::vector<Range>& ranges, int regionBins,
                  const std::string& what) {
  namespace fs = std::filesystem;
  const TempDir temp("histogram3DTest_records");
  const fs::path& dir = temp.path;
  const std::string stem = (dir / "subDomain0_histograms").string(),
                    bin = stem + ".bin";
  std::vector<std::string> layout;
  const std::string json = h.describe("", layout);
  RecordFile file(kHistogramRecordMagic);
  bool ok = !file.open(stem, json, layout, false, 0);
  h.writeRecord(file, 3, 7, stream);
  h.writeRecord(file, 1, 9, stream);
  file.flush();
  const std::vector<ReadRecord> records = readRecords(bin, regionBins);
  std::ifstream in(stem + ".json");
  const std::string text((std::istreambuf_iterator<char>(in)),
                         std::istreambuf_iterator<char>());
  in.close();
  // a histogram without a non-empty bin writes no record
  if (std::none_of(gpu.begin(), gpu.end(),
                   [](histogramTypeOut w) { return w > 0; }))
    return report(ok && records.empty() && fs::file_size(bin) == 0 &&
                      text == json,
                  what + ": no record without a non-empty bin");
  ok = ok && records.size() == 2 &&
       sameRecord(records[0], 7, 3, gpu, ranges, regionBins) &&
       sameRecord(records[1], 9, 1, gpu, ranges, regionBins) &&
       fs::file_size(bin) == records[0].header.bytes * 2 && text == json &&
       json.find("\"record_magic\": " + std::to_string(kHistogramRecordMagic) +
                 ",") != std::string::npos;
  const bool written = ok;

  // a restart at cycle 9: the same layout continues, another is kept aside
  RecordFile again(kHistogramRecordMagic);
  const bool continued = !again.open(stem, json, layout, true, 9) && ok &&
                         fs::file_size(bin) == records[0].header.bytes;
  std::ofstream(stem + ".json", std::ios::trunc) << "{}\n";
  const bool aside =
      again.open(stem, json, layout, true, 9) && fs::file_size(bin) == 0 &&
      fs::exists(dir / "subDomain0_histograms_before_cycle9.bin");
  const bool file1 = report(written, what + ": the records of the non-empty "
                                            "bins, the JSON of the layout");
  const bool restart =
      report(continued && aside, what + ": a restart continues the same "
                                        "layout, keeps another aside");
  return file1 && restart;
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
  pass &= checkCompaction(histogram, firstGpu, {Range()}, kBins);

  // two Gaussian populations, more particles than a batch
  const auto gpu = gpuHistogram(histogram, random);
  pass &= compare(gpu, cpuHistogram(random), "two Gaussians, 5e6 particles");
  pass &= sameBits(gpu, cpuFixedPointHistogram(random),
                   "the integer sums of the CPU give the same bits");
  const auto again = gpuHistogram(histogram, random);
  pass &= sameBits(gpu, again, "two runs give the same bits");
  pass &= checkCompaction(histogram, again, {Range()}, kBins);
  pass &= checkRecords(histogram, again, {Range()}, VELOCITY_HISTOGRAM3D_SIZE,
                       "a rank");
  pass &= sameBits(gpu, gpuHistogram(histogram, shuffle(random, 4321)),
                   "the particles in another order give the same bits");

  // no particle
  const Particles none;
  const auto empty = gpuHistogram(histogram, none);
  pass &= compare(empty, cpuHistogram(none), "no particle");
  pass &= checkCompaction(histogram, empty, {Range()}, kBins);

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
  pass &= checkCompaction(ranged, rangedGpu, {range}, kBins);
  pass &= checkRecords(ranged, rangedGpu, {range}, VELOCITY_HISTOGRAM3D_SIZE,
                       "a rank, range of the velocities");
  pass &= sameBits(rangedGpu, gpuHistogram(ranged, shuffle(wide, 8765)),
                   "range of the velocities: the particles in another order "
                   "give the same bits");
  for (int i = 0; i < wide.size(); i++)
    wide.v[i] = 0.05;
  const Range noSpread = dataRange(wide);
  const auto noSpreadGpu = gpuHistogram(ranged, wide);
  pass &= compare(noSpreadGpu, cpuHistogram(wide, noSpread),
                  "range of the velocities, an axis without spread");
  pass &= checkCompaction(ranged, noSpreadGpu, {noSpread}, kBins);
  pass &= sameBits(gpuHistogram(ranged, cold),
                   cpuFixedPointHistogram(cold, dataRange(cold)),
                   "range of the velocities, a cold beam: one bin, bit for "
                   "bit");
  // the same beam spread over a few units in the last place along u (one
  // push in a uniform field): no spread either, the fixed width about its
  // middle
  Particles ulps = cold;
  for (int i = 0; i < ulps.size(); i++)
    for (int k = 0; k < i % 4; k++)
      ulps.u[i] = std::nextafter(ulps.u[i], 1.0);
  const auto ulpsGpu = gpuHistogram(ranged, ulps);
  const BinGrid& ulpsGrid = ranged.getGrid(0);
  pass &= report(
      std::fabs(ulpsGrid.max[0] - ulpsGrid.min[0] - (kVmax - kVmin)) < 1e-12 &&
          ulpsGrid.min[0] < ulps.u[0] && ulps.u[3] < ulpsGrid.max[0],
      "range of the velocities, a cold beam spread over 3 units "
      "in the last place: the fixed width about it");
  pass &= sameBits(ulpsGpu, cpuFixedPointHistogram(ulps, dataRange(ulps)),
                   "range of the velocities, a cold beam spread over 3 units "
                   "in the last place: bit for bit");
  // a cold beam outside the fixed range: all its weight in bin bins / 2 of
  // w, whose centre is its velocity; spread over a few units in the last
  // place, the same histogram
  Particles drifting = cold;
  for (int i = 0; i < drifting.size(); i++)
    drifting.w[i] = 2 * kVmax;
  const auto driftingGpu = gpuHistogram(ranged, drifting);
  double total = 0, expected = 0;
  bool middle = true;
  for (int i = 0; i < VELOCITY_HISTOGRAM3D_SIZE; i++) {
    total += driftingGpu[i];
    middle &= driftingGpu[i] == 0 || i / (kBins[0] * kBins[1]) == kBins[2] / 2;
  }
  for (cudaCommonType q : drifting.q)
    expected += std::fabs(q) * VELOCITY_WEIGHT_SCALE;
  pass &=
      report(middle && std::fabs(total - expected) <= kRelTolerance * expected,
             "range of the velocities, a cold beam outside the fixed "
             "range: all its weight in the middle bin of w");
  Particles driftingUlps = drifting;
  for (int i = 0; i < driftingUlps.size(); i++)
    for (int k = 0; k < i % 4; k++)
      driftingUlps.w[i] = std::nextafter(driftingUlps.w[i], 1.0);
  pass &= sameBits(gpuHistogram(ranged, driftingUlps), driftingGpu,
                   "range of the velocities, the beam outside the fixed range "
                   "spread over 3 units in the last place: the histogram of "
                   "the beam without spread");
  // a width that overflows: the fixed range on that axis
  Particles overflowing = first;
  overflowing.u[0] = 1e308;
  overflowing.u[1] = -1e308;
  const Range overflow = dataRange(overflowing);
  pass &= sameBits(gpuHistogram(ranged, overflowing),
                   cpuFixedPointHistogram(overflowing, overflow),
                   "range of the velocities, a width that overflows: the "
                   "fixed range on that axis, bit for bit");
  pass &= report(ranged.getGrid(0).min[0] == kVmin &&
                     ranged.getGrid(0).max[0] == kVmax,
                 "range of the velocities, a width that overflows: the "
                 "fixed range on that axis");

  // the range of the velocities of no particle and of NaN velocities: the
  // fixed range on every axis; of one particle: the fixed width about it
  auto fixedGrid = [&](const BinGrid& g) {
    bool fixed = true;
    for (int d = 0; d < 3; d++)
      fixed &= g.min[d] == kVmin && g.max[d] == kVmax;
    return fixed;
  };
  pass &= compare(gpuHistogram(ranged, none), cpuHistogram(none),
                  "range of the velocities, no particle");
  pass &= report(fixedGrid(ranged.getGrid(0)),
                 "range of the velocities, no particle: the fixed range");
  Particles undefined = bad;
  for (int i = 0; i < undefined.size(); i++)
    undefined.u[i] = undefined.v[i] = undefined.w[i] =
        std::numeric_limits<cudaCommonType>::quiet_NaN();
  pass &= compare(gpuHistogram(ranged, undefined), cpuHistogram(undefined),
                  "range of the velocities, every velocity NaN");
  pass &= report(fixedGrid(ranged.getGrid(0)),
                 "range of the velocities, every velocity NaN: the fixed "
                 "range");
  Particles single;
  single.push(0.05, -0.1, 0.15, -1e-6);
  pass &= compare(gpuHistogram(ranged, single),
                  cpuHistogram(single, dataRange(single)),
                  "range of the velocities, one particle");

  return pass ? 0 : 1;
}

// ======= Macrocell histograms =======

/** A rank tiled in macrocells, its particles sorted by cell over the guarded
 *  grid (ghost cells too, in no macrocell), as the cell sort leaves them. */
struct SortedRank {
  macrocell::MacrocellPartition part;
  Particles particles;                   // cell after cell
  std::vector<int> macrocellOf;          // per particle, -1 in a ghost cell
  std::vector<int> cellStart, cellCount; // per guarded cell
};

/** The rank: count(cell, macrocell) particles in each guarded cell (x
 *  fastest), the k-th drawn by draw(particles, cell, macrocell, k); the
 *  macrocell of a ghost cell is -1. */
SortedRank
sortedRank(const int interior[3], const int size[3],
           const std::function<int(int, int)>& count,
           const std::function<void(Particles&, int, int, int)>& draw) {
  SortedRank rank;
  rank.part.build(interior[0], interior[1], interior[2], size[0], size[1],
                  size[2], 0, 0, 0);
  const int counts[3] = {rank.part.Mx, rank.part.My, rank.part.Mz};
  // the macrocell of a guarded cell along an axis, -1 for a ghost cell
  auto along = [&](int cell, int a) {
    return cell >= 1 && cell <= interior[a] ? (cell - 1) / size[a] : -1;
  };
  for (int cz = 0; cz < interior[2] + 2; cz++)
    for (int cy = 0; cy < interior[1] + 2; cy++)
      for (int cx = 0; cx < interior[0] + 2; cx++) {
        const int c = cx + (interior[0] + 2) * (cy + (interior[1] + 2) * cz);
        const int mx = along(cx, 0), my = along(cy, 1), mz = along(cz, 2);
        const int m = mx < 0 || my < 0 || mz < 0
                          ? -1
                          : mx + counts[0] * (my + counts[1] * mz);
        if (m >= rank.part.M)
          throw std::logic_error("sortedRank: a macrocell beyond the tiling");
        const int n = count(c, m);
        rank.cellStart.push_back(rank.particles.size());
        rank.cellCount.push_back(n);
        for (int k = 0; k < n; k++) {
          draw(rank.particles, c, m, k);
          rank.macrocellOf.push_back(m);
        }
      }
  return rank;
}

/** The particles of a rank in another order within each cell. */
SortedRank shuffleCells(const SortedRank& rank, unsigned seed) {
  SortedRank out = rank;
  std::mt19937 gen(seed);
  for (size_t c = 0; c < rank.cellStart.size(); c++) {
    std::vector<int> order(rank.cellCount[c]);
    for (int k = 0; k < rank.cellCount[c]; k++)
      order[k] = rank.cellStart[c] + k;
    std::shuffle(order.begin(), order.end(), gen);
    for (int k = 0; k < rank.cellCount[c]; k++) {
      const int to = rank.cellStart[c] + k, from = order[k];
      out.particles.u[to] = rank.particles.u[from];
      out.particles.v[to] = rank.particles.v[from];
      out.particles.w[to] = rank.particles.w[from];
      out.particles.q[to] = rank.particles.q[from];
    }
  }
  return out;
}

/** The macrocell histograms of the CPU: the range of each macrocell (the
 *  fixed one, or dataRange of its particles) and its bins, the fixed-point
 *  sums of the GPU with the scale of the charges of the macrocell. */
struct MacrocellReference {
  std::vector<Range> range;
  std::vector<histogramTypeOut> bins;
};

MacrocellReference cpuMacrocellHistograms(const SortedRank& rank,
                                          bool fixedRange) {
  const int M = rank.part.M;
  std::vector<Particles> of(M);
  const Particles& p = rank.particles;
  for (int i = 0; i < p.size(); i++)
    if (rank.macrocellOf[i] >= 0)
      of[rank.macrocellOf[i]].push(p.u[i], p.v[i], p.w[i], p.q[i]);
  MacrocellReference ref;
  for (int m = 0; m < M; m++) {
    ref.range.push_back(fixedRange ? Range{}
                                   : dataRange(of[m], kMacrocellBins3));
    const std::vector<histogramTypeOut> bins =
        cpuFixedPointHistogram(of[m], ref.range[m], kMacrocellBins3);
    ref.bins.insert(ref.bins.end(), bins.begin(), bins.end());
  }
  return ref;
}

/** The largest error of the bins of every macrocell against their sums in
 *  double, relative to the bin. */
double macrocellDoubleError(const SortedRank& rank,
                            const MacrocellReference& ref,
                            const std::vector<histogramTypeOut>& gpu) {
  const Particles& p = rank.particles;
  std::vector<double> sum(gpu.size(), 0.0);
  for (int i = 0; i < p.size(); i++) {
    const int m = rank.macrocellOf[i];
    const int bin = m < 0 ? -1 : cpuBin(p, i, ref.range[m], kMacrocellBins3);
    if (bin >= 0)
      sum[(size_t)m * kMacrocellBins + bin] +=
          std::fabs(p.q[i]) * VELOCITY_WEIGHT_SCALE;
  }
  double error = 0;
  for (size_t b = 0; b < sum.size(); b++)
    if (sum[b] > 0)
      error = std::max(error, std::fabs(gpu[b] - sum[b]) / sum[b]);
  return error;
}

/** The GPU macrocell histograms of the rank (kVmin, kVmax: the fixed range). */
std::vector<histogramTypeOut> gpuMacrocellHistograms(macrocellHistogram3D& h,
                                                     const SortedRank& rank) {
  velocitySoA soa(rank.particles.size(), 0);
  upload(soa, rank.particles);
  int *cellStart, *cellCount;
  const size_t cells = rank.cellStart.size();
  cudaErrChk(cudaMalloc(&cellStart, cells * sizeof(int)));
  cudaErrChk(cudaMalloc(&cellCount, cells * sizeof(int)));
  cudaErrChk(cudaMemcpyAsync(cellStart, rank.cellStart.data(),
                             cells * sizeof(int), cudaMemcpyHostToDevice,
                             stream));
  cudaErrChk(cudaMemcpyAsync(cellCount, rank.cellCount.data(),
                             cells * sizeof(int), cudaMemcpyHostToDevice,
                             stream));
  h.init(&soa, cellStart, cellCount, kVmin, kVmax, stream);
  std::vector<histogramTypeOut> bins =
      hostBins(h, (size_t)rank.part.M * kMacrocellBins);
  cudaErrChk(cudaFree(cellStart));
  cudaErrChk(cudaFree(cellCount));
  return bins;
}

/** The grid of every macrocell of the last init(): the range of the
 *  reference, its bins and resolution. */
bool checkMacrocellGrids(const macrocellHistogram3D& h,
                         const MacrocellReference& ref) {
  bool ok = h.numRegions() == (int)ref.range.size();
  for (int m = 0; ok && m < h.numRegions(); m++) {
    const BinGrid& g = h.getGrid(m);
    for (int d = 0; d < 3; d++)
      ok = ok && g.bins[d] == kMacrocellBins3[d] &&
           g.min[d] == ref.range[m].lo[d] && g.max[d] == ref.range[m].hi[d] &&
           g.resolution[d] ==
               (ref.range[m].hi[d] - ref.range[m].lo[d]) / kMacrocellBins3[d];
  }
  return report(ok, "the grid of every macrocell");
}

/** Run the macrocell histograms of a rank, fixed range and range of each
 *  macrocell, against the CPU. */
bool checkMacrocellRank(const SortedRank& rank, const std::string& what) {
  bool pass = true;
  for (bool fixedRange : {true, false}) {
    const std::string name =
        what + (fixedRange ? ", fixed range" : ", range of each macrocell");
    macrocellHistogram3D h(rank.part, fixedRange);
    const MacrocellReference ref = cpuMacrocellHistograms(rank, fixedRange);
    const auto gpu = gpuMacrocellHistograms(h, rank);
    pass &= sameBits(gpu, ref.bins,
                     name + ": the integer sums of the CPU give the same bits");
    const double error = macrocellDoubleError(rank, ref, gpu);
    std::ostringstream detail;
    detail << std::scientific << error;
    pass &= report(error <= kRelTolerance,
                   name +
                       ": every bin within the float rounding of its sum "
                       "in double (" +
                       detail.str() + ")");
    pass &= checkMacrocellGrids(h, ref);
    pass &= checkCompaction(h, gpu, ref.range, kMacrocellBins3);
    pass &= checkRecords(h, gpu, ref.range, kMacrocellBins, name);
    pass &= sameBits(gpuMacrocellHistograms(h, shuffleCells(rank, 5)), gpu,
                     name + ": another order in the cells gives the same bits");
  }
  return pass;
}

/** The tiling of every interior up to kSweepMaxInterior cells in macrocells
 *  of every size up to kSweepMaxSize: ceil(N / C) macrocells of C cells in
 *  order, the last of the remaining ones, every cell in one; no tiling of a
 *  size or an extent that is not positive; the slices of as many cells in up
 *  to kSweepMaxSize slices: in order, every cell in one, their sizes within
 *  one; and the slices of a kernel: one for a macrocell of one cell or for
 *  many macrocells, several for one large macrocell, at most its cells. */
bool checkTilingSweep() {
  bool ok = true;
  for (int n = 1; n <= kSweepMaxInterior; n++)
    for (int c = 1; c <= kSweepMaxSize; c++) {
      macrocell::MacrocellPartition part;
      ok = ok && part.build(n, 1, 1, c, 1, 1, 0, 0, 0) &&
           part.Mx == (n + c - 1) / c && part.M == part.Mx;
      int next = 0;
      for (int m = 0; ok && m < part.Mx; m++) {
        int start, length;
        macrocell::macrocellAxisRange(m, c, n, start, length);
        ok = start == next && length >= 1 &&
             (m == part.Mx - 1 ? length <= c : length == c);
        next = start + length;
      }
      ok = ok && next == n;
    }
  macrocell::MacrocellPartition part;
  ok = ok && !part.build(0, 1, 1, 1, 1, 1, 0, 0, 0) &&
       !part.build(1, 1, 1, 1, 0, 1, 0, 0, 0) &&
       !part.build(1, 1, 1, 1, 1, -2, 0, 0, 0);
  for (int n = 1; n <= kSweepMaxInterior; n++)
    for (int slices = 1; slices <= kSweepMaxSize; slices++) {
      int next = 0, smallest = n, largest = 0;
      for (int s = 0; ok && s < slices; s++) {
        int first, end;
        macrocell::sliceOf(n, s, slices, first, end);
        ok = first == next && end >= first;
        smallest = std::min(smallest, end - first);
        largest = std::max(largest, end - first);
        next = end;
      }
      ok = ok && next == n && largest - smallest <= 1;
    }
  ok = ok && macrocell::slicesFor(1, kMacrocellTiles, 1) == 1 &&
       macrocell::slicesFor(1 << 20, 1, 1 << 20) == 1 &&
       macrocell::slicesFor(1, 1, 1 << 20) > 1 &&
       macrocell::slicesFor(1, 1, 3) <= 3;
  return report(ok, "tiling of 1 to " + std::to_string(kSweepMaxInterior) +
                        " cells in macrocells of 1 to " +
                        std::to_string(kSweepMaxSize) +
                        ": every cell in one, only the last smaller; no "
                        "tiling of a size or extent below 1; slices in "
                        "order, within one cell of each other, at most the "
                        "cells of a macrocell");
}

/** The macrocell histograms of ranks of the shapes of kShapes (particles in
 *  the ghost cells too) against the CPU, both ranges: the walk of the cells
 *  of every macrocell. */
bool checkShapes(std::mt19937& gen) {
  std::normal_distribution<cudaCommonType> normal(0.0, 1.0);
  std::uniform_int_distribution<int> count(0, kShapeCellParticles);
  bool pass = true;
  for (const auto& shape : kShapes) {
    const SortedRank rank = sortedRank(
        shape[0], shape[1], [&](int, int) { return count(gen); },
        [&](Particles& p, int, int m, int) {
          p.push(0.01 * (m % 7) - 0.03 + 0.02 * normal(gen), 0.02 * normal(gen),
                 0.01 * normal(gen), -1e-6);
        });
    std::ostringstream name;
    name << "interior " << shape[0][0] << "x" << shape[0][1] << "x"
         << shape[0][2] << ", macrocells of " << shape[1][0] << "x"
         << shape[1][1] << "x" << shape[1][2] << " (" << rank.part.M
         << ", slices "
         << macrocell::slicesFor(rank.part.M, 1, rank.part.largestCells())
         << " and "
         << macrocell::slicesFor(rank.part.M, kMacrocellTiles,
                                 rank.part.largestCells())
         << ")";
    for (bool fixedRange : {true, false}) {
      macrocellHistogram3D h(rank.part, fixedRange);
      pass &= sameBits(gpuMacrocellHistograms(h, rank),
                       cpuMacrocellHistograms(rank, fixedRange).bins,
                       name.str() +
                           (fixedRange ? ", fixed range" : ", own ranges") +
                           ": the bits of the CPU");
    }
  }
  return pass;
}

int runMacrocellTest() {
  bool pass = true;
  const cudaCommonType nan = std::numeric_limits<cudaCommonType>::quiet_NaN();
  const cudaCommonType inf = std::numeric_limits<cudaCommonType>::infinity();

  // 3D: 3 x 3 x 2 macrocells with remainders; per macrocell a drifting
  // Gaussian (another range), a dense cell, a cold beam alone in a
  // macrocell (no spread), one without particles, one of NaN velocities
  // (the fixed range), and out-of-range and non-finite values elsewhere
  std::mt19937 gen(24680);
  std::normal_distribution<cudaCommonType> normal(0.0, 1.0);
  std::uniform_real_distribution<cudaCommonType> unit(0.0, 1.0);
  const SortedRank rank3D = sortedRank(
      kRank3DInterior, kRank3DSize,
      [&](int c, int m) {
        if (m == kEmptyMacrocell)
          return 0;
        if (m == kColdMacrocell)
          return c % 3 == 0 ? kColdCellParticles : 0;
        if (c == kDenseCell)
          return kDenseCellParticles;
        return (int)(unit(gen) * kCellParticles);
      },
      [&](Particles& p, int c, int m, int k) {
        // macrocell 2: charges 1e-9 of the others (a low density), binned
        // with the precision of its own
        const cudaCommonType q =
            -(1.0 + 0.5 * unit(gen)) * (m == kTinyMacrocell ? 1e-15 : 1e-6);
        if (m == kColdMacrocell) {
          p.push(0.0123, -0.05, 0.1, q);
          return;
        }
        if (m == kNanMacrocell) {
          p.push(nan, nan, nan, q);
          return;
        }
        const cudaCommonType drift = 0.02 * (m % 5) - 0.04;
        cudaCommonType v[3];
        for (int d = 0; d < 3; d++)
          v[d] = drift + (0.01 + 0.005 * (m % 3)) * normal(gen);
        cudaCommonType charge = q;
        switch ((c + k) % 97) {
        case 0:
          v[k % 3] = nan;
          break;
        case 1:
          v[k % 3] = k % 2 ? inf : -inf;
          break;
        case 2:
          v[k % 3] = k % 2 ? 1.5 * kVmax : 1.5 * kVmin;
          break;
        case 3:
          charge = nan;
          break;
        case 4:
          v[k % 3] = kVmax;
          break;
        default:
          break;
        }
        p.push(v[0], v[1], v[2], charge);
      });
  std::cout << "3D rank: " << rank3D.part.M << " macrocells, "
            << rank3D.particles.size() << " particles\n";
  if (rank3D.part.M != 3 * 3 * 2)
    throw std::logic_error("the 3D case needs its 3 x 3 x 2 macrocells");
  pass &= checkMacrocellRank(rank3D, "3D macrocells");

  // 2D: 2 x 2 x 1 macrocells
  const SortedRank rank2D = sortedRank(
      kRank2DInterior, kRank2DSize,
      [&](int, int) { return (int)(unit(gen) * kCellParticles); },
      [&](Particles& p, int, int m, int) {
        p.push(0.03 * m - 0.05 + 0.02 * normal(gen), 0.02 * normal(gen),
               0.01 * normal(gen), -1e-6);
      });
  pass &= checkMacrocellRank(rank2D, "2D macrocells");

  // a rank with fewer cells than a macrocell along y and z: one macrocell of
  // its cells along them; along x the last macrocell has the one remaining
  // cell
  const SortedRank small = sortedRank(
      kSmallRankInterior, kSmallRankSize,
      [&](int, int) { return (int)(unit(gen) * kCellParticles); },
      [&](Particles& p, int, int m, int) {
        p.push(0.04 * m - 0.02 + 0.02 * normal(gen), 0.02 * normal(gen),
               0.01 * normal(gen), -1e-6);
      });
  // the cells of the macrocells along an axis
  auto cellsOf = [](int macrocells, int size, int interior) {
    std::vector<int> cells;
    for (int m = 0; m < macrocells; m++) {
      int start, length;
      macrocell::macrocellAxisRange(m, size, interior, start, length);
      cells.push_back(length);
    }
    return cells;
  };
  const macrocell::MacrocellPartition& part = small.part;
  pass &= report(
      part.M == 2 &&
          cellsOf(part.Mx, part.Cx, part.Nx_int) == std::vector<int>{4, 1} &&
          cellsOf(part.My, part.Cy, part.Ny_int) == std::vector<int>{3} &&
          cellsOf(part.Mz, part.Cz, part.Nz_int) == std::vector<int>{2},
      "a rank smaller than a macrocell: macrocells of 4 and 1 "
      "cells along x, one of its 3 and 2 cells along y and z");
  pass &= checkMacrocellRank(small, "a rank smaller than a macrocell");

  // many macrocells: more compaction blocks than the threads of its scan
  // (each scans a strip of several)
  const SortedRank many = sortedRank(
      kManyRankInterior, kManyRankSize,
      [&](int, int) { return (int)(unit(gen) * kManyCellParticles); },
      [&](Particles& p, int, int m, int) {
        p.push(0.001 * (m % 50) - 0.025 + 0.02 * normal(gen),
               0.02 * normal(gen), 0.01 * normal(gen), -1e-6);
      });
  if (many.part.M *
          getGridSize(kMacrocellBins, sparseBins::kCompactBlockBins) <=
      2 * sparseBins::kScanThreads)
    throw std::logic_error("the many-macrocell case needs more compaction "
                           "blocks than twice the scan threads");
  pass &= checkMacrocellRank(many, "many macrocells");

  pass &= checkTilingSweep();
  pass &= checkShapes(gen);

  // no particle: every bin empty, the fixed range everywhere
  const SortedRank emptyRank = sortedRank(
      kRank3DInterior, kRank3DSize, [](int, int) { return 0; },
      [](Particles&, int, int, int) {});
  pass &= checkMacrocellRank(emptyRank, "no particle");

  // no particle, and the cells of a stale sort (the cell sort leaves them so
  // for a species without particles): the fixed range, no particle read
  SortedRank stale = emptyRank;
  std::fill(stale.cellStart.begin(), stale.cellStart.end(), kStaleCell);
  std::fill(stale.cellCount.begin(), stale.cellCount.end(), kStaleCell);
  const MacrocellReference fixedEmpty = cpuMacrocellHistograms(stale, true);
  for (bool fixedRange : {true, false}) {
    macrocellHistogram3D h(stale.part, fixedRange);
    pass &= sameBits(gpuMacrocellHistograms(h, stale), fixedEmpty.bins,
                     std::string("no particle, stale cells, ") +
                         (fixedRange ? "fixed range"
                                     : "range of each "
                                       "macrocell") +
                         ": no bin");
    pass &= checkMacrocellGrids(h, fixedEmpty);
  }
  return pass ? 0 : 1;
}

} // namespace

int main() {
  stream = createPriorityStream(StreamPriority::Diagnostic);
  int failed = EXIT_FAILURE;
  try {
    const int rank = runTest();
    const int macrocells = runMacrocellTest();
    failed = rank || macrocells;
  } catch (const std::exception& error) {
    std::cerr << "histogram3DTest failed: " << error.what() << "\n";
  }
  cudaErrChk(cudaStreamDestroy(stream));
  return failed;
}
