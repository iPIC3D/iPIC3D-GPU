/**
 * @file particleOutputSelectTest.cu
 * @brief Unit test of the selection of the particles that the particle
 *        output writes (particleOutputSelect.cuh), against a CPU reference.
 *        The key limit of a fraction. The fraction (no sort): exactly the
 *        particles with key <= limit, in their order, their charges q / f,
 *        about f of them. The cap, on cells of every kind (empty, below, at
 *        and above the cap, a dense one): in every cell min(n, K) particles,
 *        those of smallest key, by ID or, without IDs, by coordinates (equal
 *        coordinates, equal keys: still K written); the gathered arrays; the
 *        cell's charge carried by its written particles (unequal charges);
 *        the same set whatever the order of the particles in their cells;
 *        every particle of a cell written with the probability K / count
 *        over many salts and a current of the cell estimated without bias;
 *        a cap above the threads of a block; more than a million cells (the
 *        scan of the block totals in several chunks) without charges; no
 *        cells or particles; nothing written past the last particle.
 */

#include "cudaTypeDef.cuh"
#include "particleOutputSelect.cuh"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <exception>
#include <iostream>
#include <random>
#include <set>
#include <string>
#include <vector>

namespace {

constexpr int kMaxPerCell = 8;
/** A cap above the threads of the block of a cell (kCapBlock). */
constexpr int kLargeMaxPerCell = 200;
/** A cap above every count. */
constexpr int kNoCap = 1000000;
constexpr std::uint64_t kSalt = 0x9e3779b97f4a7c15ULL;
/** Cell counts of the main layout: empty, below, at and above the cap, a
 *  dense cell. */
constexpr int kCounts[] = {0,
                           1,
                           kMaxPerCell - 1,
                           kMaxPerCell,
                           kMaxPerCell + 1,
                           2 * kMaxPerCell,
                           10 * kMaxPerCell,
                           1000};
/** Repeats of kCounts in the main layout (cells), in random order. */
constexpr int kRepeats = 60;
/** The fraction: on kFractionParticles particles. */
constexpr double kFraction = 0.17;
/** A fraction whose f 2^64 is not an integer (f below 2^-11). */
constexpr double kSmallFraction = 1e-5;
constexpr int kFractionParticles = 300007;
/** The inclusion test: one cell of kInclusionCount particles writing
 *  kInclusionKept, over kInclusionTrials salts, within kSigmas standard
 *  deviations. */
constexpr int kInclusionCount = 100;
constexpr int kInclusionKept = 10;
constexpr int kInclusionTrials = 2000;
constexpr double kSigmas = 5.0;
/** The scan test: more cells than one chunk of the block totals covers. */
constexpr int kScanCells = 1200000;
constexpr int kScanMaxPerCell = 2;
/** A cell of particles with equal coordinates (equal keys without IDs). */
constexpr int kEqualCount = 20;
/** Charges summed in another order than on the device. */
constexpr double kChargeTolerance = 1e-13;

int failures = 0;

void check(bool ok, const std::string& what) {
  std::cout << "Test " << (ok ? "passed" : "failed") << ": " << what << "\n";
  failures += !ok;
}

/** The device's mix64 and keys (particleOutputSelect.cu). */
std::uint64_t mix64(std::uint64_t z) {
  z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
  z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
  return z ^ (z >> 31);
}

std::uint64_t bits(double d) {
  std::uint64_t b;
  std::memcpy(&b, &d, sizeof b);
  return b;
}

/** Particles cell after cell (a species sorted by cell): the arrays the keys
 *  and the gathers read; host buffers in page-locked memory, copied to and
 *  from the device. */
struct Particles {
  PinnedVector<double> x, y, z, u, q;
  PinnedVector<cudaPclType_ID> id;
  PinnedVector<int> count, start;
  std::size_t size() const { return q.size(); }

  std::uint64_t key(std::size_t i, std::uint64_t salt, bool byID) const {
    if (byID)
      return mix64(id[i] ^ salt);
    std::uint64_t h = mix64(salt ^ bits(x[i]));
    h = mix64(h ^ bits(y[i]));
    h = mix64(h ^ bits(z[i]));
    return mix64(h ^ bits(u[i]));
  }
};

/** count[c] particles in cell c: random coordinates and charges (unequal,
 *  of one sign), distinct random IDs. */
Particles makeParticles(const std::vector<int>& counts, std::mt19937_64& gen) {
  std::uniform_real_distribution<double> unit(0.0, 1.0), charge(-2e-3, -1e-3);
  Particles p;
  std::set<cudaPclType_ID> ids;
  for (const int n : counts) {
    p.start.push_back((int)p.size());
    p.count.push_back(n);
    for (int j = 0; j < n; j++) {
      for (auto* a : {&p.x, &p.y, &p.z, &p.u})
        a->push_back(unit(gen));
      p.q.push_back(charge(gen));
      cudaPclType_ID id;
      do
        id = gen() >> 1;
      while (!ids.insert(id).second);
      p.id.push_back(id);
    }
  }
  return p;
}

template <typename T> T* toDevice(const PinnedVector<T>& host) {
  T* device = nullptr;
  cudaErrChk(
      cudaMalloc(&device, std::max<std::size_t>(host.size(), 1) * sizeof(T)));
  if (!host.empty())
    cudaErrChk(cudaMemcpy(device, host.data(), host.size() * sizeof(T),
                          cudaMemcpyHostToDevice));
  return device;
}

template <typename T>
PinnedVector<T> fromDevice(const T* device, std::size_t n) {
  PinnedVector<T> host(n);
  if (n > 0)
    cudaErrChk(
        cudaMemcpy(host.data(), device, n * sizeof(T), cudaMemcpyDeviceToHost));
  return host;
}

/** The particles on the device, and the buffers of a selection. */
struct Device {
  int *count, *start, *offset, *blockSums, *total;
  double *x, *y, *z, *u, *q;
  cudaPclType_ID* id;
  std::uint32_t* idx;
  double* stage;
  std::size_t slots; // of idx and stage: a guard slot after every particle
  cudaStream_t stream;
  // the total and the guard slots on the host
  PinnedVector<int> totalHost = PinnedVector<int>(1);
  PinnedVector<std::uint32_t> idxGuard = PinnedVector<std::uint32_t>(1);
  PinnedVector<std::uint64_t> stageGuard = PinnedVector<std::uint64_t>(1);

  explicit Device(const Particles& p) : slots(p.size() + 1) {
    const int cells = std::max<int>(p.count.size(), 1);
    count = toDevice(p.count);
    start = toDevice(p.start);
    x = toDevice(p.x), y = toDevice(p.y), z = toDevice(p.z), u = toDevice(p.u);
    q = toDevice(p.q);
    id = toDevice(p.id);
    const std::size_t blocks =
        (std::max<std::size_t>(cells, p.size()) + kSelectScanBlock - 1) /
            kSelectScanBlock +
        1;
    cudaErrChk(cudaMalloc(&offset, cells * sizeof(int)));
    cudaErrChk(cudaMalloc(&blockSums, blocks * sizeof(int)));
    cudaErrChk(cudaMalloc(&total, sizeof(int)));
    cudaErrChk(cudaMalloc(&idx, slots * sizeof(std::uint32_t)));
    cudaErrChk(cudaMalloc(&stage, slots * sizeof(double)));
    cudaErrChk(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  }
  ~Device() {
    (void)cudaStreamDestroy(stream);
    for (void* ptr : {(void*)count, (void*)start, (void*)x, (void*)y, (void*)z,
                      (void*)u, (void*)q, (void*)id, (void*)offset,
                      (void*)blockSums, (void*)total, (void*)idx, (void*)stage})
      (void)cudaFree(ptr);
  }
  ParticleSelectKeys keys(bool byID, std::uint64_t salt) const {
    return {byID ? id : nullptr, x, y, z, u, salt};
  }
  int readTotal() {
    cudaErrChk(cudaMemcpyAsync(totalHost.data(), total, sizeof(int),
                               cudaMemcpyDeviceToHost, stream));
    cudaErrChk(cudaStreamSynchronize(stream));
    return totalHost[0];
  }
  /** idx and the stage set to all ones bits, which the slot after the last
   *  particle written must keep (guardIntact). */
  void poison() {
    cudaErrChk(
        cudaMemsetAsync(idx, 0xFF, slots * sizeof(std::uint32_t), stream));
    cudaErrChk(cudaMemsetAsync(stage, 0xFF, slots * sizeof(double), stream));
  }
  bool guardIntact(int written) {
    cudaErrChk(cudaMemcpyAsync(idxGuard.data(), idx + written,
                               sizeof(std::uint32_t), cudaMemcpyDeviceToHost,
                               stream));
    cudaErrChk(cudaMemcpyAsync(stageGuard.data(), stage + written,
                               sizeof(double), cudaMemcpyDeviceToHost, stream));
    cudaErrChk(cudaStreamSynchronize(stream));
    return idxGuard[0] == UINT32_MAX && stageGuard[0] == UINT64_MAX;
  }
};

/** The output of a selection: the first slot of each cell (with a cap), the
 *  indices written, their charges and the other arrays (when asked). */
struct Selection {
  int total = 0;
  bool guard = false; // nothing written past the last particle
  PinnedVector<int> offset;
  PinnedVector<std::uint32_t> idx;
  PinnedVector<double> x, u, q;
  PinnedVector<cudaPclType_ID> id;
};

/** x, u and the IDs of the particles written, through the stage. */
void gatherArrays(Device& d, Selection& s) {
  launchGather<double>(d.stage, d.x, d.idx, s.total, d.stream);
  cudaErrChk(cudaStreamSynchronize(d.stream));
  s.x = fromDevice(d.stage, s.total);
  launchGather<double>(d.stage, d.u, d.idx, s.total, d.stream);
  cudaErrChk(cudaStreamSynchronize(d.stream));
  s.u = fromDevice(d.stage, s.total);
  launchGather<cudaPclType_ID>((cudaPclType_ID*)d.stage, d.id, d.idx, s.total,
                               d.stream);
  cudaErrChk(cudaStreamSynchronize(d.stream));
  s.id = fromDevice((cudaPclType_ID*)d.stage, s.total);
}

/** The cap on the species sorted by cell (on the device of p), the phases
 *  of copySampledParticlesAsync; with full the charges and the other arrays
 *  through the stage, else the indices only. */
Selection selectCapped(Device& d, const Particles& p, int maxPerCell,
                       std::uint64_t salt, bool byID, bool full = true) {
  const int cells = (int)p.count.size();
  const ParticleSelectKeys keys = d.keys(byID, salt);
  Selection s;
  d.poison();
  launchCapCount(d.count, cells, maxPerCell, d.offset, d.blockSums, d.total,
                 d.stream);
  s.total = d.readTotal();
  launchCapSelect(d.count, d.start, d.offset, cells, maxPerCell, keys, d.q,
                  d.idx, full ? d.stage : nullptr, d.stream);
  cudaErrChk(cudaStreamSynchronize(d.stream));
  s.offset = fromDevice(d.offset, cells);
  s.idx = fromDevice(d.idx, s.total);
  if (full) {
    s.q = fromDevice(d.stage, s.total);
    gatherArrays(d, s);
  }
  s.guard = d.guardIntact(s.total);
  return s;
}

Selection selectCapped(const Particles& p, int maxPerCell, std::uint64_t salt,
                       bool byID, bool full = true) {
  Device d(p);
  return selectCapped(d, p, maxPerCell, salt, byID, full);
}

/** The fraction f > 0 (by ID), on the particles in their order. */
Selection selectFraction(const Particles& p, double fraction,
                         std::uint64_t salt) {
  Device d(p);
  const std::uint64_t limit = selectLimit(fraction);
  const ParticleSelectKeys keys = d.keys(true, salt);
  Selection s;
  d.poison();
  launchFractionCount(keys, p.size(), limit, d.blockSums, d.total, d.stream);
  s.total = d.readTotal();
  launchFractionSelect(keys, p.size(), limit, d.blockSums, d.idx, d.stream);
  cudaErrChk(cudaStreamSynchronize(d.stream));
  s.idx = fromDevice(d.idx, s.total);
  gatherArrays(d, s);
  launchGatherScaled(d.stage, d.q, d.idx, s.total, 1.0 / fraction, d.stream);
  cudaErrChk(cudaStreamSynchronize(d.stream));
  s.q = fromDevice(d.stage, s.total);
  s.guard = d.guardIntact(s.total);
  return s;
}

/** A capped selection against the reference, cell by cell: min(n, K) of
 *  smallest key, the offsets, the gathered values, the cell's charge; why:
 *  the first difference. */
bool matchesReference(const Particles& p, const Selection& s, int K,
                      std::uint64_t salt, bool byID, std::string& why) {
  if (!s.guard) {
    why = "a value written past the last particle";
    return false;
  }
  long expected = 0;
  for (std::size_t c = 0; c < p.count.size(); c++) {
    if (s.offset[c] != expected) {
      why = "offset of cell " + std::to_string(c);
      return false;
    }
    const int n = p.count[c];
    std::vector<std::uint64_t> keys;
    double cellCharge = 0.0;
    for (int j = 0; j < n; j++) {
      keys.push_back(p.key(p.start[c] + j, salt, byID));
      cellCharge += p.q[p.start[c] + j];
    }
    const int kept = std::min(n, K);
    expected += kept;
    std::set<std::uint32_t> got(s.idx.begin() + s.offset[c],
                                s.idx.begin() + s.offset[c] + kept);
    if ((int)got.size() != kept ||
        (kept && (*got.begin() < (std::uint32_t)p.start[c] ||
                  *got.rbegin() >= (std::uint32_t)(p.start[c] + n)))) {
      why = "indices of cell " + std::to_string(c) + " (" + std::to_string(n) +
            ")";
      return false;
    }
    // the kept-th smallest key t: every key below t written, none above
    if (kept > 0) {
      std::vector<std::uint64_t> sorted = keys;
      std::nth_element(sorted.begin(), sorted.begin() + kept - 1, sorted.end());
      const std::uint64_t t = sorted[kept - 1];
      for (int j = 0; j < n; j++) {
        const bool in = got.count(p.start[c] + j) != 0;
        if ((keys[j] < t && !in) || (keys[j] > t && in)) {
          why = "written set of cell " + std::to_string(c);
          return false;
        }
      }
    }
    if (s.q.empty()) // the indices only
      continue;
    double keptCharge = 0.0, written = 0.0;
    for (int o = s.offset[c]; o < s.offset[c] + kept; o++) {
      keptCharge += p.q[s.idx[o]];
      written += s.q[o];
    }
    for (int o = s.offset[c]; o < s.offset[c] + kept; o++) {
      const std::uint32_t i = s.idx[o];
      const double expectedQ =
          n > K ? p.q[i] * cellCharge / keptCharge : p.q[i];
      if (std::abs(s.q[o] - expectedQ) >
              kChargeTolerance * std::abs(expectedQ) ||
          (n <= K && s.q[o] != p.q[i]) || s.x[o] != p.x[i] ||
          s.u[o] != p.u[i] || s.id[o] != p.id[i]) {
        why = "gathered values of output " + std::to_string(o);
        return false;
      }
    }
    if (kept > 0 && std::abs(written - cellCharge) >
                        kChargeTolerance * std::abs(cellCharge)) {
      why = "charge of cell " + std::to_string(c);
      return false;
    }
  }
  if (s.total != expected) {
    why =
        "total " + std::to_string(s.total) + " != " + std::to_string(expected);
    return false;
  }
  return true;
}

/** check of a capped selection against the reference, with the first
 *  difference in the message. */
void checkReference(const Particles& p, const Selection& s, int K, bool byID,
                    const std::string& what) {
  std::string why;
  const bool ok = matchesReference(p, s, K, kSalt, byID, why);
  check(ok, what + (why.empty() ? "" : " — " + why));
}

void testLimit() {
  const std::uint64_t half = selectLimit(0.5);
  const std::uint64_t one = selectLimit(1.0);
  const std::uint64_t tiny = selectLimit(1e-30);
  const std::uint64_t small = selectLimit(kSmallFraction);
  check(half == (1ULL << 63) - 1 && one == UINT64_MAX && tiny == 0 &&
            small == (std::uint64_t)std::floor(std::ldexp(kSmallFraction, 64)),
        "the key limit, u = key / 2^64 < f: f = 0.5 admits key < 2^63, 1 "
        "every key, 1e-30 key 0, f 2^64 not an integer the keys up to its "
        "floor");
}

/** No cells, only empty cells, no particles: nothing written. */
void testEmpty() {
  std::mt19937_64 gen(16);
  const Particles noCells;
  const Particles emptyCells = makeParticles({0, 0, 0}, gen);
  const int capped = selectCapped(noCells, kMaxPerCell, kSalt, true).total +
                     selectCapped(emptyCells, kMaxPerCell, kSalt, true).total;
  const int fraction = selectFraction(noCells, kFraction, kSalt).total;
  check(capped == 0 && fraction == 0,
        "no cells, empty cells, no particles: nothing written");
}

Particles mainLayout(std::mt19937_64& gen, bool equalCell) {
  std::vector<int> counts;
  for (int r = 0; r < kRepeats; r++)
    counts.insert(counts.end(), std::begin(kCounts), std::end(kCounts));
  std::shuffle(counts.begin(), counts.end(), gen);
  Particles p = makeParticles(counts, gen);
  if (equalCell) { // particles with equal coordinates: equal keys
    p.start.push_back((int)p.size());
    p.count.push_back(kEqualCount);
    for (int j = 0; j < kEqualCount; j++) {
      for (auto* a : {&p.x, &p.y, &p.z, &p.u})
        a->push_back(0.5);
      p.q.push_back(-1e-3);
      p.id.push_back(1000000000ULL + j);
    }
  }
  return p;
}

/** The cap K on the main layout, by ID or by coordinates (with a cell of
 *  equal keys): against the reference, whatever the order of the particles
 *  in their cells, and a cap above every count. */
void testCells(bool byID, int K) {
  std::mt19937_64 gen(byID ? 11 : 12);
  const bool equalCell = !byID;
  const Particles p = mainLayout(gen, equalCell);
  const std::string by = byID ? "by ID" : "by coordinates";
  const Selection s = selectCapped(p, K, kSalt, byID);
  checkReference(p, s, K, byID,
                 "cap " + std::to_string(K) + " (" + by +
                     "): in every cell min(n, K) of smallest key, the arrays "
                     "gathered, the cell's charge");

  // the same set whatever the order of the particles in their cells
  Particles shuffled = p;
  for (std::size_t c = 0; c < p.count.size(); c++) {
    std::vector<int> order(p.count[c]);
    for (int j = 0; j < p.count[c]; j++)
      order[j] = p.start[c] + j;
    std::shuffle(order.begin(), order.end(), gen);
    for (int j = 0; j < p.count[c]; j++) {
      const int to = p.start[c] + j, from = order[j];
      for (auto pair : {std::make_pair(&shuffled.x, &p.x),
                        std::make_pair(&shuffled.y, &p.y),
                        std::make_pair(&shuffled.z, &p.z),
                        std::make_pair(&shuffled.u, &p.u),
                        std::make_pair(&shuffled.q, &p.q)})
        (*pair.first)[to] = (*pair.second)[from];
      shuffled.id[to] = p.id[from];
    }
  }
  const Selection t = selectCapped(shuffled, K, kSalt, byID);
  bool same = t.total == s.total;
  for (std::size_t c = 0; same && c < p.count.size(); c++) {
    const int kept =
        (c + 1 < p.count.size() ? s.offset[c + 1] : s.total) - s.offset[c];
    std::multiset<cudaPclType_ID> a(s.id.begin() + s.offset[c],
                                    s.id.begin() + s.offset[c] + kept);
    std::multiset<cudaPclType_ID> b(t.id.begin() + t.offset[c],
                                    t.id.begin() + t.offset[c] + kept);
    // without IDs, the particles of equal coordinates are interchangeable
    same = a == b || (equalCell && c + 1 == p.count.size());
  }
  check(same, "the same particles written whatever their order in the cells (" +
                  by + ", cap " + std::to_string(K) + ")");

  checkReference(p, selectCapped(p, kNoCap, kSalt, byID), kNoCap, byID,
                 "a cap above every count writes every particle, charges "
                 "unchanged (" +
                     by + ")");
}

/** The fraction: exactly the particles with key <= limit, in their order,
 *  q / f; about f of them; f = 1 every particle. */
void testFraction() {
  std::mt19937_64 gen(15);
  const Particles p = makeParticles({kFractionParticles}, gen);
  const std::uint64_t limit = selectLimit(kFraction);
  const Selection s = selectFraction(p, kFraction, kSalt);
  std::vector<std::uint32_t> expected;
  for (std::size_t i = 0; i < p.size(); i++)
    if (p.key(i, kSalt, true) <= limit)
      expected.push_back(i);
  bool same = s.guard && s.total == (int)expected.size() &&
              std::equal(expected.begin(), expected.end(), s.idx.begin());
  for (int o = 0; same && o < s.total; o++)
    same = s.q[o] == p.q[s.idx[o]] * (1.0 / kFraction) &&
           s.id[o] == p.id[s.idx[o]] && s.x[o] == p.x[s.idx[o]];
  check(same, "fraction " + std::to_string(kFraction) +
                  ": the particles with key <= limit, in their order, "
                  "charges q / f");
  const double mean = kFraction * p.size();
  const double sd = std::sqrt(p.size() * kFraction * (1 - kFraction));
  check(std::abs(s.total - mean) < kSigmas * sd,
        "about f of the particles (" + std::to_string(s.total) + ", expected " +
            std::to_string(mean) + " +- " + std::to_string(sd) + ")");
  const Selection all = selectFraction(p, 1.0, kSalt);
  check(all.guard && all.total == (int)p.size(), "fraction 1: every particle");
}

/** Over many salts: every particle written with the probability kept /
 *  count, and the written particles estimate the current sum q u of the
 *  cell without bias (velocities independent of the charges). */
void testInclusion() {
  std::mt19937_64 gen(13);
  const Particles p = makeParticles({kInclusionCount}, gen);
  std::vector<int> times(kInclusionCount, 0);
  double truth = 0.0;
  for (int i = 0; i < kInclusionCount; i++)
    truth += p.q[i] * p.u[i];
  std::vector<double> estimates;
  Device d(p);
  for (int trial = 0; trial < kInclusionTrials; trial++) {
    const Selection s =
        selectCapped(d, p, kInclusionKept, mix64(trial + 1), true);
    double estimate = 0.0;
    for (int o = 0; o < s.total; o++) {
      times[s.idx[o]]++;
      estimate += s.q[o] * s.u[o];
    }
    estimates.push_back(estimate);
  }
  const double prob = (double)kInclusionKept / kInclusionCount;
  const double mean = kInclusionTrials * prob;
  const double sd = std::sqrt(kInclusionTrials * prob * (1 - prob));
  int worst = 0;
  for (const int t : times)
    worst = std::max(worst, (int)std::abs(t - mean));
  check(worst < kSigmas * sd,
        "every particle written with the probability " + std::to_string(prob) +
            " (largest deviation " + std::to_string(worst) + ", sd " +
            std::to_string(sd) + ")");
  double m = 0.0, v = 0.0;
  for (const double e : estimates)
    m += e / kInclusionTrials;
  for (const double e : estimates)
    v += (e - m) * (e - m) / (kInclusionTrials - 1);
  const double se = std::sqrt(v / kInclusionTrials);
  check(std::abs(m - truth) < kSigmas * se,
        "the written particles estimate the current sum q u of the cell "
        "without bias (" +
            std::to_string(m) + " vs " + std::to_string(truth) + ", se " +
            std::to_string(se) + ")");
}

/** More cells than one chunk of the block totals: the counts and scan,
 *  the indices only. */
void testManyCells() {
  std::mt19937_64 gen(14);
  std::uniform_int_distribution<int> small(0, 3);
  std::vector<int> counts(kScanCells);
  for (int& n : counts)
    n = small(gen);
  const Particles p = makeParticles(counts, gen);
  checkReference(p, selectCapped(p, kScanMaxPerCell, kSalt, true, false),
                 kScanMaxPerCell, true,
                 std::to_string(kScanCells) +
                     " cells: the scan of the counts and the written "
                     "particles, no charges");
}

} // namespace

int main() {
  try {
    testLimit();
    testEmpty();
    testCells(true, kMaxPerCell);
    testCells(false, kMaxPerCell);
    testCells(true, kLargeMaxPerCell);
    testFraction();
    testInclusion();
    testManyCells();
  } catch (const std::exception& error) {
    std::cerr << "particleOutputSelectTest failed: " << error.what() << "\n";
    failures++;
  }
  return failures == 0 ? 0 : 1;
}
