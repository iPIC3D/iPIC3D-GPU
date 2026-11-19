/**
 * @file particleControlTest.cu
 * @brief Unit test of the particle merging and splitting kernels
 *        (particleControlKernel.cu).
 *
 * Cases (--case <name>):
 *   merge-conservation  random cells of drifting thermal particles with
 *                       unequal charges, half of them in pairs of equal
 *                       velocity: per cell, the charge, the momentum and the
 *                       centre of charge are kept; the energy does not grow;
 *                       a particle takes part in one merge at most (it
 *                       gains the charge of one particle merged away, and
 *                       each of those is gained once); no cell goes below
 *                       the target and a cell at or below it is untouched
 *   merge-host          the merged pairs are those of the same rule applied
 *                       on the host, one particle at a time; also cold cells,
 *                       where all candidates are equal and the first is taken
 *   merge-frame         the particles merged away do not change with the
 *                       frame: the same cells with a drift of 25 thermal
 *                       speeds
 *   merge-chain         three particles of equal velocity and target 1: one
 *                       merge, and the particle merged away takes no other
 *                       (the charge is kept)
 *   merge-nodes         a pair on a line along a grid axis with equal
 *                       velocities leaves the ten node moments unchanged;
 *                       thermal cells: the charge of the eight nodes of a
 *                       cell is kept in sum, and a node changes by much less
 *                       than its shot noise (reported)
 *   split               3D and 2D: the two particles have half the charge and
 *                       the velocity of the parent, are symmetric about it
 *                       along one listed axis, in its cell; the parents are
 *                       distinct and move with the shift; the other
 *                       particles are untouched; the ten node moments are
 *                       unchanged; the new IDs are distinct
 *   split-merge         a cell split, then merged down to its first count,
 *                       gives back its particles
 */

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <iterator>
#include <random>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "particleControlKernel.cuh"

using real = cudaParticleType;

namespace {

/** Threads per block of the launches: a multiple of WARP_SIZE. */
constexpr int kBlockSize = 256;
/** Round-off allowed on a conserved sum, relative to its scale. */
constexpr real kRoundOff = 1e-12;
/** Cell size and thermal speed of the test particles. */
constexpr real kCell = 0.25, kVth = 0.02;

int failures = 0;

void check(bool ok, const char* what, double value = 0) {
  std::printf("  [%s] %s (%.3e)\n", ok ? "ok" : "FAIL", what, value);
  if (!ok)
    failures++;
}

/** Particles of one species on the host, sorted by cell where it matters. */
struct Particles {
  std::vector<real> u, v, w, q, x, y, z;
  std::vector<cudaPclType_ID> id;
  size_t size() const { return q.size(); }
  void resize(size_t n) {
    for (auto* f : {&u, &v, &w, &q, &x, &y, &z})
      f->resize(n);
    id.resize(n, PARTICLE_ID_INVALID);
  }
};

/** Departure flags of the particles; empty for none. */
using Flags = std::vector<departureArrayElementType>;

bool flagged(const Flags& flags, size_t i) {
  return !flags.empty() && flags[i].dest != 0;
}

template <typename T> T* toDevice(const std::vector<T>& host, size_t capacity) {
  T* device;
  cudaErrChk(cudaMalloc(&device, std::max<size_t>(capacity, 1) * sizeof(T)));
  if (!host.empty())
    cudaErrChk(cudaMemcpy(device, host.data(), host.size() * sizeof(T),
                          cudaMemcpyHostToDevice));
  return device;
}

template <typename T>
void fromDevice(std::vector<T>& host, T* device, size_t n) {
  host.resize(n);
  if (n > 0)
    cudaErrChk(
        cudaMemcpy(host.data(), device, n * sizeof(T), cudaMemcpyDeviceToHost));
  cudaErrChk(cudaFree(device));
}

/** Device copy of the particles with room for capacity. */
ParticleSoADevice upload(const Particles& p, size_t capacity, bool trackID) {
  ParticleSoADevice soa{};
  soa.u = toDevice(p.u, capacity);
  soa.v = toDevice(p.v, capacity);
  soa.w = toDevice(p.w, capacity);
  soa.q = toDevice(p.q, capacity);
  soa.x = toDevice(p.x, capacity);
  soa.y = toDevice(p.y, capacity);
  soa.z = toDevice(p.z, capacity);
  soa.id = trackID ? toDevice(p.id, capacity) : nullptr;
  soa.nop = (uint32_t)p.size();
  soa.capacity = (uint32_t)capacity;
  soa.trackParticleID = trackID;
  return soa;
}

/** The first n particles back on the host; frees the device copy. */
Particles download(ParticleSoADevice soa, size_t n) {
  Particles p;
  fromDevice(p.u, soa.u, n);
  fromDevice(p.v, soa.v, n);
  fromDevice(p.w, soa.w, n);
  fromDevice(p.q, soa.q, n);
  fromDevice(p.x, soa.x, n);
  fromDevice(p.y, soa.y, n);
  fromDevice(p.z, soa.z, n);
  if (soa.trackParticleID)
    fromDevice(p.id, soa.id, n);
  return p;
}

/** Cells in a row along x; cell c spans [c, c + 1) kCell, y and z in
 *  [0, kCell). */
struct Cells {
  std::vector<int> start, count;
  int size() const { return (int)count.size(); }
};

/**
 * Thermal particles with the drift, charges between 0.5 and 1.5 (negative),
 * filling the cells with the given counts. With pairs, every other particle
 * repeats the velocity of the one before it.
 */
Particles thermalCells(const std::vector<int>& counts, Cells& cells, real drift,
                       bool pairs, unsigned seed) {
  std::mt19937_64 rng(seed);
  std::normal_distribution<real> normal(0.0, kVth);
  std::uniform_real_distribution<real> uniform(0.0, 1.0);
  Particles p;
  cells.start.clear();
  cells.count = counts;
  for (int c = 0; c < (int)counts.size(); c++) {
    cells.start.push_back((int)p.size());
    for (int i = 0; i < counts[c]; i++) {
      const size_t n = p.size();
      p.resize(n + 1);
      p.q[n] = -(0.5 + uniform(rng));
      p.x[n] = (c + uniform(rng)) * kCell;
      p.y[n] = uniform(rng) * kCell;
      p.z[n] = uniform(rng) * kCell;
      if (pairs && i % 2 == 1) {
        p.u[n] = p.u[n - 1];
        p.v[n] = p.v[n - 1];
        p.w[n] = p.w[n - 1];
      } else {
        p.u[n] = drift + normal(rng);
        p.v[n] = normal(rng);
        p.w[n] = normal(rng);
      }
    }
  }
  return p;
}

/** Cell counts around the warp sizes and the target, and large cells. */
std::vector<int> mixedCounts(int target, unsigned seed) {
  std::vector<int> counts = {0,
                             1,
                             target - 1,
                             target,
                             target + 1,
                             WARP_SIZE - 1,
                             WARP_SIZE,
                             WARP_SIZE + 1,
                             2 * WARP_SIZE + 3,
                             5 * WARP_SIZE,
                             1000};
  std::mt19937 rng(seed);
  std::uniform_int_distribution<int> any(0, 4 * target);
  for (int c = 0; c < 600; c++)
    counts.push_back(any(rng));
  return counts;
}

/** Run the merging kernel; the particles after it and the departure flags. */
Particles merge(const Particles& before, const Cells& cells, int target,
                Flags& flags) {
  const size_t n = before.size();
  ParticleSoADevice soa = upload(before, n, false);
  int* start = toDevice(cells.start, cells.size());
  int* count = toDevice(cells.count, cells.size());
  flags.assign(n, departureArrayElementType{});
  departureArrayElementType* departure = toDevice(flags, n);
  const int threads = cells.size() * WARP_SIZE;
  particleMergingKernel<<<(threads + kBlockSize - 1) / kBlockSize,
                          kBlockSize>>>(start, count, cells.size(), target, soa,
                                        departure);
  cudaErrChk(cudaDeviceSynchronize());
  fromDevice(flags, departure, n);
  cudaErrChk(cudaFree(start));
  cudaErrChk(cudaFree(count));
  return download(soa, n);
}

/**
 * The merging rule on the host, one particle at a time: partner[i] is the
 * particle merged into i, or -1; gone[i] is set for a particle merged away.
 */
void hostMerge(const Particles& p, const Cells& cells, int target,
               std::vector<int>& partner, std::vector<char>& gone) {
  partner.assign(p.size(), -1);
  gone.assign(p.size(), 0);
  for (int c = 0; c < cells.size(); c++) {
    const int first = cells.start[c], count = cells.count[c];
    if (count <= target)
      continue;
    real q = 0, qu = 0, qv = 0, qw = 0, qv2 = 0;
    for (int i = first; i < first + count; i++) {
      q += p.q[i];
      qu += p.q[i] * p.u[i];
      qv += p.q[i] * p.v[i];
      qw += p.q[i] * p.w[i];
      qv2 += p.q[i] * (p.u[i] * p.u[i] + p.v[i] * p.v[i] + p.w[i] * p.w[i]);
    }
    const real maxDv2 =
        MERGE_VELOCITY_TOLERANCE *
        std::max((qv2 - (qu * qu + qv * qv + qw * qw) / q) / q, (real)0);
    int merged = 0;
    for (int lead = first; lead < first + count - 1; lead++) {
      if (gone[lead])
        continue;
      real best = maxDv2;
      int bestIdx = -1;
      for (int i = lead + 1; i < first + count; i++) {
        const real du = p.u[i] - p.u[lead], dv = p.v[i] - p.v[lead],
                   dw = p.w[i] - p.w[lead];
        const real dv2 = du * du + dv * dv + dw * dw;
        if (!gone[i] && dv2 <= maxDv2 && (bestIdx < 0 || dv2 < best)) {
          best = dv2;
          bestIdx = i;
        }
      }
      if (bestIdx < 0)
        continue;
      partner[lead] = bestIdx;
      gone[bestIdx] = 1;
      if (count - ++merged <= target)
        break;
    }
  }
}

/** Sums of a cell over the particles not flagged: charge, q v, q x, q v^2. */
struct CellSums {
  real q = 0, p[3] = {0, 0, 0}, x[3] = {0, 0, 0}, energy = 0;
  int alive = 0;
};

CellSums cellSums(const Particles& p, int first, int count,
                  const Flags& flags) {
  CellSums s;
  for (int i = first; i < first + count; i++) {
    if (flagged(flags, i))
      continue;
    s.alive++;
    s.q += p.q[i];
    s.p[0] += p.q[i] * p.u[i];
    s.p[1] += p.q[i] * p.v[i];
    s.p[2] += p.q[i] * p.w[i];
    s.x[0] += p.q[i] * p.x[i];
    s.x[1] += p.q[i] * p.y[i];
    s.x[2] += p.q[i] * p.z[i];
    s.energy += p.q[i] * (p.u[i] * p.u[i] + p.v[i] * p.v[i] + p.w[i] * p.w[i]);
  }
  return s;
}

/**
 * The ten moments (charge, three q v, six q v v) on the eight nodes of each
 * cell of a grid of cell size kCell from the origin, of the particles not
 * flagged: moments[10 * (8 * cell + node) + m], cells in a row along x.
 */
std::vector<real> nodeMoments(const Particles& p, int numCells,
                              const Flags& flags) {
  std::vector<real> moments(80 * (size_t)numCells, 0.0);
  for (size_t i = 0; i < p.size(); i++) {
    if (flagged(flags, i))
      continue;
    const real pos[3] = {p.x[i] / kCell, p.y[i] / kCell, p.z[i] / kCell};
    const int cell =
        std::min(std::max((int)std::floor(pos[0]), 0), numCells - 1);
    const real f[3] = {pos[0] - cell, pos[1], pos[2]};
    const real vm[10] = {1.0,
                         p.u[i],
                         p.v[i],
                         p.w[i],
                         p.u[i] * p.u[i],
                         p.u[i] * p.v[i],
                         p.u[i] * p.w[i],
                         p.v[i] * p.v[i],
                         p.v[i] * p.w[i],
                         p.w[i] * p.w[i]};
    for (int node = 0; node < 8; node++) {
      real weight = p.q[i];
      for (int a = 0; a < 3; a++)
        weight *= (node >> a) & 1 ? f[a] : 1 - f[a];
      for (int m = 0; m < 10; m++)
        moments[10 * (8 * (size_t)cell + node) + m] += weight * vm[m];
    }
  }
  return moments;
}

/** Largest |a - b| over the ten moments, each over the scale of that moment. */
real momentChange(const std::vector<real>& a, const std::vector<real>& b) {
  real change = 0;
  for (int k = 0; k < 10; k++) {
    real scale = 0, diff = 0;
    for (size_t i = k; i < a.size(); i += 10) {
      scale = std::max(scale, std::fabs(a[i]));
      diff = std::max(diff, std::fabs(a[i] - b[i]));
    }
    if (scale > 0)
      change = std::max(change, diff / scale);
  }
  return change;
}

// ======= Merging =======

int testMergeConservation() {
  std::printf("merge-conservation\n");
  failures = 0;
  const int target = 40;
  Cells cells;
  const Particles before =
      thermalCells(mixedCounts(target, 11), cells, 5 * kVth, true, 12);
  Flags flags, none;
  const Particles after = merge(before, cells, target, flags);

  real errQ = 0, errP = 0, errX = 0, energyGain = 0;
  int below = 0, touched = 0, merges = 0, badFlag = 0, notOnePair = 0;
  for (int c = 0; c < cells.size(); c++) {
    const int first = cells.start[c], count = cells.count[c];
    const CellSums a = cellSums(before, first, count, none);
    const CellSums b = cellSums(after, first, count, flags);
    merges += count - b.alive;
    for (int i = first; i < first + count; i++)
      if (flags[i].dest != 0 &&
          flags[i].dest != departureArrayElementType::DELETE)
        badFlag++;
    if (count <= target) {
      for (int i = first; i < first + count; i++)
        if (flags[i].dest != 0 || after.q[i] != before.q[i] ||
            after.u[i] != before.u[i] || after.x[i] != before.x[i])
          touched++;
      continue;
    }
    if (b.alive < target)
      below++;
    const real qScale = std::fabs(a.q);
    errQ = std::max(errQ, std::fabs(a.q - b.q) / qScale);
    for (int k = 0; k < 3; k++) {
      errP = std::max(errP, std::fabs(a.p[k] - b.p[k]) / (qScale * kVth));
      errX = std::max(errX, std::fabs(a.x[k] - b.x[k]) / (qScale * kCell));
    }
    energyGain = std::max(energyGain, (b.energy - a.energy) / a.energy);

    // A particle that changed gained the charge of exactly one particle
    // merged away, and each of those is gained once.
    std::multiset<real> goneCharges;
    for (int i = first; i < first + count; i++)
      if (flags[i].dest != 0)
        goneCharges.insert(before.q[i]);
    for (int i = first; i < first + count; i++) {
      if (flags[i].dest != 0 || after.q[i] == before.q[i])
        continue;
      const real q2 = after.q[i] - before.q[i];
      auto it = goneCharges.lower_bound(q2 - 1e-12);
      if (it == goneCharges.end() || std::fabs(*it - q2) > 1e-12) {
        notOnePair++;
        continue;
      }
      goneCharges.erase(it);
    }
    notOnePair += (int)goneCharges.size();
  }
  std::printf("  %zu particles in %d cells, %d merges\n", before.size(),
              cells.size(), merges);
  check(merges > 1000, "merges happen", merges);
  check(errQ < kRoundOff, "charge of each cell kept", errQ);
  check(errP < kRoundOff, "momentum of each cell kept", errP);
  check(errX < kRoundOff, "centre of charge of each cell kept", errX);
  check(energyGain < kRoundOff, "energy of no cell grows", energyGain);
  check(notOnePair == 0, "a particle takes part in one merge at most",
        notOnePair);
  check(below == 0, "no cell below the target", below);
  check(touched == 0, "cells at or below the target untouched", touched);
  check(badFlag == 0, "only DELETE flags", badFlag);
  return failures;
}

/** The pairs the kernel merged against those of the host rule. */
int comparePairs(const Particles& before, const Cells& cells, int target,
                 const char* what) {
  Flags flags;
  const Particles after = merge(before, cells, target, flags);
  std::vector<int> partner;
  std::vector<char> gone;
  hostMerge(before, cells, target, partner, gone);
  int wrongGone = 0, merges = 0;
  real errValue = 0;
  for (size_t i = 0; i < before.size(); i++) {
    if ((flags[i].dest != 0) != (gone[i] != 0))
      wrongGone++;
    if (partner[i] < 0)
      continue;
    merges++;
    const int j = partner[i];
    const real q = before.q[i] + before.q[j];
    const real f1 = before.q[i] / q, f2 = before.q[j] / q;
    errValue = std::max(
        {errValue, std::fabs(after.q[i] - q),
         std::fabs(after.u[i] - (f1 * before.u[i] + f2 * before.u[j])) / kVth,
         std::fabs(after.x[i] - (f1 * before.x[i] + f2 * before.x[j])) / kCell,
         std::fabs(after.z[i] - (f1 * before.z[i] + f2 * before.z[j])) /
             kCell});
  }
  std::printf("  %s: %d merges\n", what, merges);
  check(merges > 100, "merges happen", merges);
  check(wrongGone == 0, "same particles merged away as on the host", wrongGone);
  check(errValue < kRoundOff, "merged particles as on the host", errValue);
  return merges;
}

int testMergeHost() {
  std::printf("merge-host\n");
  failures = 0;
  const int target = 40;
  Cells cells;
  comparePairs(thermalCells(mixedCounts(target, 21), cells, 0.0, false, 22),
               cells, target, "thermal");
  comparePairs(thermalCells(mixedCounts(target, 23), cells, 3 * kVth, true, 24),
               cells, target, "thermal with pairs, drifting");
  // every particle of a cell moves with one velocity: all candidates tie
  Particles cold = thermalCells(mixedCounts(target, 25), cells, 0.0, false, 26);
  for (int c = 0; c < cells.size(); c++)
    for (int i = cells.start[c] + 1; i < cells.start[c] + cells.count[c]; i++) {
      cold.u[i] = cold.u[cells.start[c]];
      cold.v[i] = cold.v[cells.start[c]];
      cold.w[i] = cold.w[cells.start[c]];
    }
  comparePairs(cold, cells, target, "cold");
  return failures;
}

int testMergeFrame() {
  std::printf("merge-frame\n");
  failures = 0;
  const int target = 40;
  Cells cells;
  const std::vector<int> counts = mixedCounts(target, 31);
  // 25 thermal speeds; adding it rounds off the low bits of the velocities
  const real drift = 0.5;
  const Particles rest = thermalCells(counts, cells, 0.0, false, 32);
  Particles moving = rest;
  for (auto& u : moving.u)
    u += drift;
  Flags flagsRest, flagsMoving;
  merge(rest, cells, target, flagsRest);
  merge(moving, cells, target, flagsMoving);
  int merges = 0, differ = 0;
  for (size_t i = 0; i < rest.size(); i++) {
    merges += flagsRest[i].dest != 0;
    differ += (flagsRest[i].dest != 0) != (flagsMoving[i].dest != 0);
  }
  std::printf("  drift %.0f thermal speeds: %d merges at rest\n", drift / kVth,
              merges);
  check(merges > 100, "merges happen", merges);
  // a pair at the tolerance can change side with the round-off of the drift
  check(differ <= merges / 500, "same merges in the moving frame", differ);
  return failures;
}

int testMergeChain() {
  std::printf("merge-chain\n");
  failures = 0;
  Particles p;
  p.resize(3);
  for (int i = 0; i < 3; i++) {
    p.q[i] = -1.0 - i;
    p.u[i] = 0.3;
    p.v[i] = -0.1;
    p.w[i] = 0.2;
    p.x[i] = p.y[i] = p.z[i] = 0.1 * (i + 1) * kCell;
  }
  Cells cells;
  cells.start = {0};
  cells.count = {3};
  Flags flags;
  const Particles after = merge(p, cells, 1, flags);
  const CellSums s = cellSums(after, 0, 3, flags);
  check(s.alive == 2, "one merge among three particles", s.alive);
  check(std::fabs(s.q + 6.0) < kRoundOff, "charge kept", s.q);
  check(flags[0].dest == 0, "the first particle stays", flags[0].dest);
  return failures;
}

int testMergeNodes() {
  std::printf("merge-nodes\n");
  failures = 0;
  // Pairs of equal velocity on a line along each axis in turn, one per cell.
  const int numCells = 300;
  Cells cells;
  Particles line =
      thermalCells(std::vector<int>(numCells, 2), cells, 2 * kVth, true, 41);
  for (int c = 0; c < numCells; c++) {
    const int i = cells.start[c], along = c % 3;
    if (along != 0)
      line.x[i + 1] = line.x[i];
    if (along != 1)
      line.y[i + 1] = line.y[i];
    if (along != 2)
      line.z[i + 1] = line.z[i];
  }
  Flags flags, none;
  Particles after = merge(line, cells, 1, flags);
  int merges = 0;
  for (const auto& f : flags)
    merges += f.dest != 0;
  check(merges == numCells, "every pair merged", merges);
  const real exact = momentChange(nodeMoments(line, numCells, none),
                                  nodeMoments(after, numCells, flags));
  check(exact < kRoundOff,
        "pair along a grid axis, equal velocities: ten node moments kept",
        exact);

  // Thermal cells: the charge of the eight nodes of a cell changes, their
  // sum does not.
  const int target = 64;
  const Particles thermal = thermalCells(std::vector<int>(numCells, 2 * target),
                                         cells, 0.0, false, 42);
  after = merge(thermal, cells, target, flags);
  const std::vector<real> a = nodeMoments(thermal, numCells, none),
                          b = nodeMoments(after, numCells, flags);
  real sumErr = 0, largest = 0, squares = 0;
  merges = 0;
  for (const auto& f : flags)
    merges += f.dest != 0;
  for (int c = 0; c < numCells; c++) {
    real sa = 0, sb = 0;
    for (int node = 0; node < 8; node++) {
      const real qa = a[10 * (8 * c + node)], qb = b[10 * (8 * c + node)];
      sa += qa;
      sb += qb;
      largest = std::max(largest, std::fabs(qa - qb));
      squares += (qa - qb) * (qa - qb);
    }
    sumErr = std::max(sumErr, std::fabs(sa - sb) / std::fabs(sa));
  }
  // a node holds target / 4 particles of mean charge 1: its shot noise is
  // about sqrt(target / 4) charges
  const real shotNoise = std::sqrt(target / 4.0);
  std::printf("  thermal cells of %d particles: %.2f merges per cell; node "
              "charge change over shot noise: rms %.4f, largest %.3f\n",
              2 * target, merges / (double)numCells,
              std::sqrt(squares / (8.0 * numCells)) / shotNoise,
              largest / shotNoise);
  check(sumErr < kRoundOff, "thermal cells: node charge sum kept", sumErr);
  check(largest < 0.5 * shotNoise,
        "thermal cells: node charge change below half the shot noise",
        largest / shotNoise);
  return failures;
}

// ======= Splitting =======

/** Split count particles; the particles after it (nop + count of them). */
Particles split(const Particles& before, uint32_t count, uint32_t shift,
                const SplitGeometry& geometry, bool trackID) {
  ParticleSoADevice soa = upload(before, before.size() + count, trackID);
  ParticleIDGenerator generator;
  ParticleIDGenerator::counter_type firstNew = 1000;
  cudaErrChk(cudaMalloc(&generator.nextSequence, sizeof(firstNew)));
  cudaErrChk(cudaMemcpy(generator.nextSequence, &firstNew, sizeof(firstNew),
                        cudaMemcpyHostToDevice));
  if (count > 0)
    particleSplittingKernel<<<(count + kBlockSize - 1) / kBlockSize,
                              kBlockSize>>>(soa, count, shift, geometry,
                                            generator);
  cudaErrChk(cudaDeviceSynchronize());
  cudaErrChk(cudaFree(generator.nextSequence));
  return download(soa, before.size() + count);
}

SplitGeometry splitGeometry(bool flatZ) {
  SplitGeometry g{};
  for (int a = 0; a < 3; a++) {
    g.start[a] = 0.0;
    g.d[a] = kCell;
    g.invd[a] = 1.0 / kCell;
    g.halfSpacing[a] = 0.5 * kCell / 4;
    if (a < 2 || !flatZ)
      g.axis[g.axisCount++] = a;
  }
  return g;
}

/** Checks of one split of count particles. Returns the set of parents. */
std::set<int> checkSplit(const Particles& before, int numCells, uint32_t count,
                         uint32_t shift, bool flatZ) {
  const SplitGeometry geometry = splitGeometry(flatZ);
  const Particles after = split(before, count, shift, geometry, true);
  const size_t nop = before.size();
  const std::vector<real>* const pos0[3] = {&before.x, &before.y, &before.z};
  const std::vector<real>* const pos1[3] = {&after.x, &after.y, &after.z};

  // A parent is the particle whose charge halved; its twin is found by the
  // charge, the velocity and the two positions kept.
  std::set<int> parents;
  std::set<cudaPclType_ID> ids;
  int bad = 0, changed = 0, flatSplit = 0;
  real errCentre = 0;
  std::vector<int> twinOf(nop, -1);
  for (size_t t = nop; t < nop + count; t++) {
    ids.insert(after.id[t]);
    // the twin has the velocity of its parent: find it among the halved
    for (size_t i = 0; i < nop; i++) {
      if (twinOf[i] >= 0 || after.q[i] != 0.5 * before.q[i] ||
          after.q[t] != after.q[i] || after.u[t] != before.u[i] ||
          after.v[t] != before.v[i] || after.w[t] != before.w[i])
        continue;
      twinOf[i] = (int)t;
      break;
    }
  }
  for (size_t i = 0; i < nop; i++) {
    if (twinOf[i] < 0) {
      if (after.q[i] != before.q[i] || after.x[i] != before.x[i] ||
          after.y[i] != before.y[i] || after.z[i] != before.z[i] ||
          after.u[i] != before.u[i] || after.v[i] != before.v[i] ||
          after.w[i] != before.w[i] || after.id[i] != before.id[i])
        changed++;
      continue;
    }
    parents.insert((int)i);
    const int t = twinOf[i];
    int moved = 0;
    for (int a = 0; a < 3; a++) {
      const real x0 = (*pos0[a])[i], x1 = (*pos1[a])[i], x2 = (*pos1[a])[t];
      if (x1 == x0 && x2 == x0)
        continue;
      moved++;
      if (a == 2 && flatZ)
        flatSplit++;
      errCentre = std::max(errCentre, std::fabs(0.5 * (x1 + x2) - x0) / kCell);
      // apart, by no more than the spacing, in the cell of the parent
      if (!(x1 < x0 && x0 < x2) ||
          x2 - x1 > 2 * geometry.halfSpacing[a] * (1 + 1e-12) ||
          std::floor(x1 / kCell) != std::floor(x0 / kCell) ||
          std::floor(x2 / kCell) != std::floor(x0 / kCell))
        bad++;
    }
    if (moved != 1 || after.u[i] != before.u[i] || after.v[i] != before.v[i] ||
        after.w[i] != before.w[i] || after.id[i] != before.id[i])
      bad++;
  }
  const Flags none;
  const real moments = momentChange(nodeMoments(before, numCells, none),
                                    nodeMoments(after, numCells, none));
  std::printf("  %s, %u of %zu split, shift %u\n", flatZ ? "2D" : "3D", count,
              nop, shift);
  check(parents.size() == count, "every twin has its own parent",
        parents.size());
  check(bad == 0,
        "twins: half charge, velocity kept, apart along one axis, "
        "in the parent's cell",
        bad);
  check(errCentre < kRoundOff, "twins symmetric about the parent", errCentre);
  check(flatSplit == 0, "no split along an axis that is not listed", flatSplit);
  check(changed == 0, "other particles untouched", changed);
  check(moments < kRoundOff, "ten node moments kept", moments);
  check(ids.size() == count && ids.count(PARTICLE_ID_INVALID) == 0,
        "new IDs distinct", ids.size());
  return parents;
}

int testSplit() {
  std::printf("split\n");
  failures = 0;
  const int numCells = 50;
  Cells cells;
  // distinct velocities: a twin identifies its parent
  const Particles p =
      thermalCells(std::vector<int>(numCells, 37), cells, 3 * kVth, false, 51);
  const uint32_t nop = (uint32_t)p.size();
  const std::set<int> a = checkSplit(p, numCells, nop / 7, 0, false);
  const std::set<int> b = checkSplit(p, numCells, nop / 7, 12345, false);
  checkSplit(p, numCells, nop / 3, 2654435761u, true);
  checkSplit(p, numCells, nop, 99, false);
  checkSplit(p, numCells, 1, 7, true);
  std::vector<int> common;
  std::set_intersection(a.begin(), a.end(), b.begin(), b.end(),
                        std::back_inserter(common));
  check(common.size() < a.size() / 2, "the shift changes the parents",
        common.size());
  return failures;
}

int testSplitMerge() {
  std::printf("split-merge\n");
  failures = 0;
  const int count = 150;
  Cells cells;
  // one cell, far from the tolerance: no pair of the original particles is
  // close, so that only the twins merge
  Particles p = thermalCells({count}, cells, 0.0, false, 61);
  for (int i = 0; i < count; i++) {
    p.u[i] = 10 * kVth * std::cos(0.7 * i) * (1 + i);
    p.v[i] = 10 * kVth * std::sin(1.3 * i) * (1 + i);
  }
  const Particles twins = split(p, count, 5, splitGeometry(false), false);
  cells.count = {2 * count};
  Flags flags;
  const Particles merged = merge(twins, cells, count, flags);
  int alive = 0;
  real err = 0;
  for (int i = 0; i < 2 * count; i++)
    alive += flags[i].dest == 0;
  // the parents are the first particles and absorb their twins
  for (int i = 0; i < count; i++) {
    if (flags[i].dest != 0) {
      err = 1;
      continue;
    }
    err = std::max({err, std::fabs(merged.q[i] - p.q[i]),
                    std::fabs(merged.x[i] - p.x[i]) / kCell,
                    std::fabs(merged.y[i] - p.y[i]) / kCell,
                    std::fabs(merged.z[i] - p.z[i]) / kCell,
                    std::fabs(merged.u[i] - p.u[i]) / kVth,
                    std::fabs(merged.v[i] - p.v[i]) / kVth,
                    std::fabs(merged.w[i] - p.w[i]) / kVth});
  }
  check(alive == count, "back to the first count", alive);
  check(err < kRoundOff, "the particles before the split are back", err);
  return failures;
}

} // namespace

int main(int argc, char** argv) {
  const std::pair<const char*, int (*)()> cases[] = {
      {"merge-conservation", testMergeConservation},
      {"merge-host", testMergeHost},
      {"merge-frame", testMergeFrame},
      {"merge-chain", testMergeChain},
      {"merge-nodes", testMergeNodes},
      {"split", testSplit},
      {"split-merge", testSplitMerge}};
  std::string only;
  for (int i = 1; i + 1 < argc; i++)
    if (std::string(argv[i]) == "--case")
      only = argv[i + 1];
  int bad = 0, ran = 0;
  for (const auto& c : cases)
    if (only.empty() || only == c.first) {
      bad += c.second();
      ran++;
    }
  if (ran == 0) {
    std::printf("unknown case %s\n", only.c_str());
    return 2;
  }
  std::printf("%s\n", bad == 0 ? "PASSED" : "FAILED");
  return bad == 0 ? 0 : 1;
}
