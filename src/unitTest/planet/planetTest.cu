/**
 * @file planetTest.cu
 * @brief Unit test for the planet boundary-condition kernels defined in
 *        planetKernel.cu.
 *
 * What is tested
 * ──────────────
 *  1. Ion charge reduction (planetChargeReductionKernel)
 *     → per-cycle ion charge and diagnostic accumulator
 *  2. Energy computation  (planetEnergyKernel)
 *     → energy of ONE physical electron, (gamma-1) c^2 / |qom|, independent of
 *       the macroparticle weight |q|; diagnostic hit-charge accumulation
 *       (including a multi-block launch)
 *  3. Bitonic sort         (bitonicPadKernel + bitonicSortStepKernel)
 *     → descending energy order; a light fast electron outranks a heavy slow
 *       one
 *  4. Charge cutoff        (chargeCutoffKernel) over four cycles
 *     → nearest-match cutoff, charge carried across cycles (both signs),
 *       ions-only cycle, conservation ∑|q_ion| - ∑|q_absorbed| == residual,
 *       diagnostics
 *  5. Reflect + compact    (planetReflectCompactKernel)
 *     → surviving particles are outside the planet sphere after reflection
 *
 * Test data (c = 1)
 * ─────────────────
 *  • ion species (global 0, qom = +1): 5 particles, q = +2.0 → |Q_ion| = 10
 *  • electron species A (global 1, qom = -1): 4 particles
 *  • electron species B (global 2, qom = -2): 3 particles
 *
 *    pcl   gidx   q      u      per-electron energy
 *    A0     0   -0.25   0.40   0.0910895   (light and fast)
 *    A1     1   -4.00   0.30   0.0482848   (heavy and slower)
 *    A2     2   -1.00   0.20   0.0206207
 *    A3     3   -0.50   0.10   0.0050378
 *    B0     4   -2.50   0.50   0.0773503
 *    B1     5   -1.50   0.35   0.0337606
 *    B2     6   -1.00   0.05   0.0006262
 *
 *  Sorted (descending energy): A0 B0 A1 B1 A2 A3 B2
 *  |q| in that order:          0.25 2.5 4 1.5 1 0.5 1   (prefix 0.25 2.75 6.75
 *                                                         8.25 9.25 9.75 10.75)
 *  Weighting the energy by |q| (the old key) would rank A1 above A0.
 *
 *  Cycles (an electron is absorbed while it brings the absorbed charge closer
 *  to target = residual + ion charge):
 *    1: ions 10, target 10    → absorb 6 (9.75),  residual +0.25
 *       (the old ">= target" rule absorbed all 7: 10.75)
 *    2: ions 3, no electrons  → absorb 0,         residual +3.25 (limited)
 *    3: no ions, target 3.25  → absorb 2 (2.75),  residual +0.50
 *    4: ions 2,  target 2.5   → absorb 2 (2.75),  residual -0.25
 */

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <vector>

// planetKernel.cuh pulls in cudaTypeDef, particleArrayCUDA, particleExchange,
// hashedSum, arrayCUDA — in the correct order.
#include "planetKernel.cuh"

// ─── helpers ───────────────────────────────────────────────────

static constexpr int BLOCK = 256;

/** Next power of two >= n */
static int nextPow2(int n) {
  int p = 1;
  while (p < n)
    p <<= 1;
  return p;
}

/** Reference per-electron kinetic energy (gamma - 1) c^2 / |qom|, computed
 *  directly in long double. */
static cudaParticleType hostEnergy(const SpeciesParticle& p,
                                   cudaParticleType qom, cudaParticleType c) {
  const long double v2 = (long double)p.get_u() * p.get_u() +
                         (long double)p.get_v() * p.get_v() +
                         (long double)p.get_w() * p.get_w();
  const long double c2 = (long double)c * c;
  const long double gamma = 1.0L / sqrtl(1.0L - v2 / c2);
  return (cudaParticleType)((gamma - 1.0L) * c2 / fabsl((long double)qom));
}

static bool closeRel(cudaParticleType a, cudaParticleType b,
                     cudaParticleType tol = 1e-12) {
  return fabs(a - b) <= tol * std::max((cudaParticleType)1.0, fabs(b));
}

// ─── printing helpers ──────────────────────────────────────────

static void printParticle(const char* tag, int idx, const SpeciesParticle& p,
                          cudaParticleType energy = -1.0) {
  std::cout << "  " << std::left << std::setw(12) << tag << " [" << std::setw(2)
            << idx << "]"
            << "  pos=(" << std::setw(8) << p.get_x() << ", " << std::setw(8)
            << p.get_y() << ", " << std::setw(8) << p.get_z() << ")"
            << "  vel=(" << std::setw(8) << p.get_u() << ", " << std::setw(8)
            << p.get_v() << ", " << std::setw(8) << p.get_w() << ")"
            << "  q=" << std::setw(8) << p.get_q();
  if (energy >= 0.0)
    std::cout << "  E=" << energy;
  std::cout << "\n";
}

static void printSeparator(const char* title) {
  std::cout << "\n═══════════════════════════════════════════════════\n"
            << " " << title
            << "\n═══════════════════════════════════════════════════\n";
}

static void check(bool ok, const std::string& what, bool& allPassed) {
  if (ok) {
    std::cout << "  ok    " << what << "\n";
  } else {
    std::cerr << "  *** FAIL: " << what << " ***\n";
    allPassed = false;
  }
}

static PlanetBCState readState(const PlanetBCState* d_state) {
  PlanetBCState h;
  cudaErrChk(
      cudaMemcpy(&h, d_state, sizeof(PlanetBCState), cudaMemcpyDeviceToHost));
  return h;
}

static void setIonChargeCycle(PlanetBCState* d_state, cudaParticleType q) {
  cudaErrChk(cudaMemcpy(&d_state->ionChargeCycle, &q, sizeof(cudaParticleType),
                        cudaMemcpyHostToDevice));
}

// ═══════════════════════════════════════════════════════════════
//  MAIN
// ═══════════════════════════════════════════════════════════════

static int runTest() {
  bool allPassed = true;

  // ─── Planet sphere parameters ──────────────────────────────
  const cudaCommonType originX = 5.0;
  const cudaCommonType originY = 5.0;
  const cudaCommonType originZ = 5.0;
  const cudaCommonType sphereRadius = 1.0;
  const int doSphere = 1; // 3-D

  // ─── Species parameters ────────────────────────────────────
  const cudaParticleType c = 1.0;
  const cudaParticleType qomElec[2] = {-1.0, -2.0};
  const int ns = 3; // ion, electron A, electron B
  const int elecGlobal[2] = {1, 2};
  const int nElecSpecies = 2;

  // ────────────────────────────────────────────────────────────
  //  Particles (all inside the sphere at (5.3, 5, 5), moving along +x)
  // ────────────────────────────────────────────────────────────
  const int nIonPlanet = 5;
  SpeciesParticle ionPcls[nIonPlanet];
  for (int i = 0; i < nIonPlanet; i++)
    ionPcls[i].set(/*u*/ 0.01, /*v*/ 0.0, /*w*/ 0.0, /*q*/ 2.0,
                   /*x*/ 5.3, /*y*/ 5.0, /*z*/ 5.0, (cudaParticleType)i);

  const int nElec0 = 4;
  SpeciesParticle elec0Pcls[nElec0];
  {
    const cudaParticleType qs[] = {-0.25, -4.0, -1.0, -0.5};
    const cudaParticleType us[] = {0.40, 0.30, 0.20, 0.10};
    for (int i = 0; i < nElec0; i++)
      elec0Pcls[i].set(us[i], 0.0, 0.0, qs[i], 5.3, 5.0, 5.0,
                       (cudaParticleType)i);
  }
  const int nElec1 = 3;
  SpeciesParticle elec1Pcls[nElec1];
  {
    const cudaParticleType qs[] = {-2.5, -1.5, -1.0};
    const cudaParticleType us[] = {0.50, 0.35, 0.05};
    for (int i = 0; i < nElec1; i++)
      elec1Pcls[i].set(us[i], 0.0, 0.0, qs[i], 5.3, 5.0, 5.0,
                       (cudaParticleType)i);
  }
  const int totalElecPlanet = nElec0 + nElec1; // 7

  auto pclOf = [&](int gidx) -> const SpeciesParticle& {
    return gidx < nElec0 ? elec0Pcls[gidx] : elec1Pcls[gidx - nElec0];
  };

  // ─── Expected values ───────────────────────────────────────
  const cudaParticleType expectedIonCharge = 10.0;
  const uint32_t expectedSortedGidx[] = {0, 4, 1, 5, 2, 3, 6};
  std::vector<cudaParticleType> expectedEnergy(totalElecPlanet);
  for (int g = 0; g < totalElecPlanet; g++)
    expectedEnergy[g] = hostEnergy(pclOf(g), qomElec[g < nElec0 ? 0 : 1], c);

  // ════════════════════════════════════════════════════════════
  //  Device buffers
  // ════════════════════════════════════════════════════════════
  planetArray ionPA(ionPcls, nIonPlanet);
  planetArray* d_ionPA = ionPA.copyToDevice();
  planetArray elec0PA(elec0Pcls, nElec0);
  planetArray* d_elec0PA = elec0PA.copyToDevice();
  planetArray elec1PA(elec1Pcls, nElec1);
  planetArray* d_elec1PA = elec1PA.copyToDevice();

  planetArray* h_elecPtrs[2] = {d_elec0PA, d_elec1PA};
  planetArray** d_elecPtrs;
  cudaErrChk(cudaMalloc(&d_elecPtrs, nElecSpecies * sizeof(planetArray*)));
  cudaErrChk(cudaMemcpy(d_elecPtrs, h_elecPtrs,
                        nElecSpecies * sizeof(planetArray*),
                        cudaMemcpyHostToDevice));

  int h_speciesOffsets[2] = {0, nElec0};
  int* d_speciesOffsets;
  cudaErrChk(cudaMalloc(&d_speciesOffsets, nElecSpecies * sizeof(int)));
  cudaErrChk(cudaMemcpy(d_speciesOffsets, h_speciesOffsets,
                        nElecSpecies * sizeof(int), cudaMemcpyHostToDevice));

  PlanetBCState* d_state;
  cudaErrChk(cudaMalloc(&d_state, sizeof(PlanetBCState)));
  cudaErrChk(cudaMemset(d_state, 0, sizeof(PlanetBCState)));

  // Diagnostics layout as in the solver: hit |Q| per global species, then
  // absorbed |Q| per electron species.
  cudaParticleType* d_diag;
  cudaErrChk(
      cudaMalloc(&d_diag, (ns + nElecSpecies) * sizeof(cudaParticleType)));
  cudaErrChk(
      cudaMemset(d_diag, 0, (ns + nElecSpecies) * sizeof(cudaParticleType)));

  // ════════════════════════════════════════════════════════════
  //  TEST 1: Ion charge reduction
  // ════════════════════════════════════════════════════════════
  printSeparator("TEST 1 — Ion charge reduction");
  planetChargeReductionKernel<<<(nIonPlanet + BLOCK - 1) / BLOCK, BLOCK,
                                BLOCK * sizeof(cudaParticleType)>>>(
      d_ionPA, nIonPlanet, &d_state->ionChargeCycle, d_diag + 0);
  cudaErrChk(cudaDeviceSynchronize());
  {
    const PlanetBCState st = readState(d_state);
    cudaParticleType diagIon = 0;
    cudaErrChk(cudaMemcpy(&diagIon, d_diag, sizeof(cudaParticleType),
                          cudaMemcpyDeviceToHost));
    std::cout << "  ion |Q| this cycle = " << st.ionChargeCycle
              << "  diagnostic = " << diagIon << "  (expected "
              << expectedIonCharge << ")\n";
    check(closeRel(st.ionChargeCycle, expectedIonCharge),
          "ion charge reduced into state", allPassed);
    check(closeRel(diagIon, expectedIonCharge),
          "ion charge reduced into diagnostic accumulator", allPassed);
  }

  // ════════════════════════════════════════════════════════════
  //  TEST 2: Per-electron energies + hit-charge accumulation
  // ════════════════════════════════════════════════════════════
  printSeparator("TEST 2 — Per-electron energy (weight independent)");

  const int paddedN = nextPow2(totalElecPlanet);
  cudaParticleType* d_energyBuf;
  uint32_t* d_globalIdxBuf;
  cudaErrChk(cudaMalloc(&d_energyBuf, paddedN * sizeof(cudaParticleType)));
  cudaErrChk(cudaMalloc(&d_globalIdxBuf, paddedN * sizeof(uint32_t)));

  planetEnergyKernel<<<(nElec0 + BLOCK - 1) / BLOCK, BLOCK>>>(
      d_elec0PA, nElec0, qomElec[0], c, d_energyBuf, d_globalIdxBuf,
      /*speciesOffset=*/0, d_diag + elecGlobal[0]);
  planetEnergyKernel<<<(nElec1 + BLOCK - 1) / BLOCK, BLOCK>>>(
      d_elec1PA, nElec1, qomElec[1], c, d_energyBuf, d_globalIdxBuf,
      /*speciesOffset=*/nElec0, d_diag + elecGlobal[1]);
  cudaErrChk(cudaDeviceSynchronize());

  std::vector<cudaParticleType> h_energy(totalElecPlanet);
  std::vector<uint32_t> h_gidx(totalElecPlanet);
  cudaErrChk(cudaMemcpy(h_energy.data(), d_energyBuf,
                        totalElecPlanet * sizeof(cudaParticleType),
                        cudaMemcpyDeviceToHost));
  cudaErrChk(cudaMemcpy(h_gidx.data(), d_globalIdxBuf,
                        totalElecPlanet * sizeof(uint32_t),
                        cudaMemcpyDeviceToHost));
  {
    bool energyPass = true;
    for (int i = 0; i < totalElecPlanet; i++) {
      std::cout << "    gidx=" << h_gidx[i] << "  q=" << std::setw(6)
                << pclOf(i).get_q() << "  E=" << std::setw(12) << h_energy[i]
                << "  (expected " << expectedEnergy[i] << ")\n";
      energyPass &=
          closeRel(h_energy[i], expectedEnergy[i]) && h_gidx[i] == (uint32_t)i;
    }
    check(energyPass, "energies match (gamma-1) c^2 / |qom|", allPassed);

    cudaParticleType diagElec[2];
    cudaErrChk(cudaMemcpy(diagElec, d_diag + elecGlobal[0],
                          2 * sizeof(cudaParticleType),
                          cudaMemcpyDeviceToHost));
    std::cout << "  hit |Q|: A = " << diagElec[0]
              << " (expected 5.75), B = " << diagElec[1] << " (expected 5)\n";
    check(closeRel(diagElec[0], 5.75) && closeRel(diagElec[1], 5.0),
          "electron hit charge accumulated per species", allPassed);
  }

  // Same velocity, weights 40x apart → identical energy keys.
  {
    SpeciesParticle twin[2];
    twin[0].set(0.3, 0.1, -0.2, -0.1, 5.3, 5.0, 5.0, 0.0);
    twin[1].set(0.3, 0.1, -0.2, -4.0, 5.3, 5.0, 5.0, 1.0);
    planetArray twinPA(twin, 2);
    planetArray* d_twinPA = twinPA.copyToDevice();
    cudaParticleType* d_twinE;
    uint32_t* d_twinIdx;
    cudaErrChk(cudaMalloc(&d_twinE, 2 * sizeof(cudaParticleType)));
    cudaErrChk(cudaMalloc(&d_twinIdx, 2 * sizeof(uint32_t)));
    planetEnergyKernel<<<1, BLOCK>>>(d_twinPA, 2, -1.0, c, d_twinE, d_twinIdx,
                                     0, nullptr);
    cudaErrChk(cudaDeviceSynchronize());
    cudaParticleType twinE[2];
    cudaErrChk(cudaMemcpy(twinE, d_twinE, 2 * sizeof(cudaParticleType),
                          cudaMemcpyDeviceToHost));
    check(twinE[0] == twinE[1],
          "energy independent of macroparticle weight (|q| 0.1 vs 4)",
          allPassed);
    cudaFree(d_twinE);
    cudaFree(d_twinIdx);
    cudaFree(d_twinPA);
  }

  // Multi-block launch: hit charge summed over several blocks and warps.
  {
    const int nBig = 2000; // 8 blocks of 256, last one partial
    std::vector<SpeciesParticle> big(nBig);
    for (int i = 0; i < nBig; i++)
      big[i].set(0.01 * (i % 7), 0.0, 0.0, -0.5, 5.3, 5.0, 5.0,
                 (cudaParticleType)i);
    planetArray bigPA(big.data(), nBig);
    planetArray* d_bigPA = bigPA.copyToDevice();
    cudaParticleType *d_bigE, *d_bigQ;
    uint32_t* d_bigIdx;
    cudaErrChk(cudaMalloc(&d_bigE, nBig * sizeof(cudaParticleType)));
    cudaErrChk(cudaMalloc(&d_bigIdx, nBig * sizeof(uint32_t)));
    cudaErrChk(cudaMalloc(&d_bigQ, sizeof(cudaParticleType)));
    cudaErrChk(cudaMemset(d_bigQ, 0, sizeof(cudaParticleType)));
    planetEnergyKernel<<<(nBig + BLOCK - 1) / BLOCK, BLOCK>>>(
        d_bigPA, nBig, -1.0, c, d_bigE, d_bigIdx, 0, d_bigQ);
    cudaErrChk(cudaDeviceSynchronize());
    cudaParticleType bigQ = 0;
    cudaErrChk(cudaMemcpy(&bigQ, d_bigQ, sizeof(cudaParticleType),
                          cudaMemcpyDeviceToHost));
    std::cout << "  multi-block hit |Q| = " << bigQ << "  (expected 1000)\n";
    check(bigQ == 1000.0, "multi-block hit-charge accumulation", allPassed);
    cudaFree(d_bigE);
    cudaFree(d_bigIdx);
    cudaFree(d_bigQ);
    cudaFree(d_bigPA);
  }

  // ════════════════════════════════════════════════════════════
  //  TEST 3: Bitonic sort (descending by energy)
  // ════════════════════════════════════════════════════════════
  printSeparator("TEST 3 — Bitonic sort (descending energy)");
  if (paddedN > totalElecPlanet) {
    bitonicPadKernel<<<(paddedN - totalElecPlanet + BLOCK - 1) / BLOCK,
                       BLOCK>>>(d_energyBuf, d_globalIdxBuf, totalElecPlanet,
                                paddedN);
  }
  for (int k = 2; k <= paddedN; k <<= 1)
    for (int j = k >> 1; j > 0; j >>= 1)
      bitonicSortStepKernel<<<(paddedN + BLOCK - 1) / BLOCK, BLOCK>>>(
          d_energyBuf, d_globalIdxBuf, j, k, paddedN);
  cudaErrChk(cudaDeviceSynchronize());

  cudaErrChk(cudaMemcpy(h_energy.data(), d_energyBuf,
                        totalElecPlanet * sizeof(cudaParticleType),
                        cudaMemcpyDeviceToHost));
  cudaErrChk(cudaMemcpy(h_gidx.data(), d_globalIdxBuf,
                        totalElecPlanet * sizeof(uint32_t),
                        cudaMemcpyDeviceToHost));
  {
    bool sortPass = true;
    for (int i = 0; i < totalElecPlanet; i++) {
      std::cout << "    rank " << i << "  gidx=" << h_gidx[i]
                << "  |q|=" << std::setw(5) << fabs(pclOf(h_gidx[i]).get_q())
                << "  E=" << h_energy[i] << "\n";
      sortPass &= h_gidx[i] == expectedSortedGidx[i];
      if (i > 0)
        sortPass &= h_energy[i] <= h_energy[i - 1];
    }
    check(sortPass, "descending order A0 B0 A1 B1 A2 A3 B2", allPassed);
    check(h_gidx[0] == 0 && h_gidx[2] == 1,
          "light fast electron (A0) outranks heavy slow one (A1)", allPassed);
  }

  // ════════════════════════════════════════════════════════════
  //  TEST 4: Charge cutoff with carry-over across cycles
  // ════════════════════════════════════════════════════════════
  printSeparator("TEST 4 — Charge cutoff & carry-over");

  int* d_cutoff;
  cudaErrChk(cudaMalloc(&d_cutoff, sizeof(int)));

  struct CycleCase {
    const char* label;
    cudaParticleType ionCharge; // set before the cycle (cycle 1: from TEST 1)
    int nElec;                  // electrons that hit this cycle
    int expectedCutoff;
    cudaParticleType expectedResidual;
  };
  const CycleCase cycles[] = {
      {"cycle 1: ions 10, 7 electrons", -1.0, totalElecPlanet, 6, 0.25},
      {"cycle 2: ions 3, no electrons", 3.0, 0, 0, 3.25},
      {"cycle 3: no ions, 7 electrons", 0.0, totalElecPlanet, 2, 0.5},
      {"cycle 4: ions 2, 7 electrons", 2.0, totalElecPlanet, 2, -0.25},
  };

  cudaParticleType totalIon = 0, totalAbsorbed = 0;
  int h_cutoff = 0;
  for (const CycleCase& cc : cycles) {
    if (cc.ionCharge >= 0)
      setIonChargeCycle(d_state, cc.ionCharge);
    totalIon += readState(d_state).ionChargeCycle;

    chargeCutoffKernel<<<1, 1>>>(d_elecPtrs, nElecSpecies, d_speciesOffsets,
                                 d_globalIdxBuf, d_energyBuf, cc.nElec, d_state,
                                 d_diag + ns, d_cutoff);
    cudaErrChk(cudaDeviceSynchronize());
    cudaErrChk(
        cudaMemcpy(&h_cutoff, d_cutoff, sizeof(int), cudaMemcpyDeviceToHost));
    const PlanetBCState st = readState(d_state);

    cudaParticleType absorbed = 0;
    for (int i = 0; i < h_cutoff; i++)
      absorbed += fabs(pclOf(h_gidx[i]).get_q());
    totalAbsorbed += absorbed;

    std::cout << "  " << cc.label << " → cutoff " << h_cutoff
              << ", absorbed |q| " << absorbed << ", residual "
              << st.residualCharge << "\n";
    check(h_cutoff == cc.expectedCutoff &&
              closeRel(st.residualCharge, cc.expectedResidual) &&
              st.ionChargeCycle == 0,
          std::string(cc.label) + ": cutoff, residual, ion charge cleared",
          allPassed);
  }

  {
    const PlanetBCState st = readState(d_state);
    std::cout << "  ∑|q_ion| = " << totalIon
              << "  ∑|q_absorbed electrons| = " << totalAbsorbed
              << "  residual = " << st.residualCharge << "\n";
    check(closeRel(totalIon - totalAbsorbed, st.residualCharge),
          "charge conserved: ∑ ion - ∑ absorbed electron == residual",
          allPassed);

    cudaParticleType absorbedBySpecies[2];
    cudaErrChk(cudaMemcpy(absorbedBySpecies, d_diag + ns,
                          2 * sizeof(cudaParticleType),
                          cudaMemcpyDeviceToHost));
    std::cout << "  absorbed |Q|: A = " << absorbedBySpecies[0]
              << " (expected 6.25), B = " << absorbedBySpecies[1]
              << " (expected 9)\n";
    check(closeRel(absorbedBySpecies[0], 6.25) &&
              closeRel(absorbedBySpecies[1], 9.0),
          "absorbed charge per electron species", allPassed);

    // Cutoff energies: between A3/B2 (cycle 1) and B0/A1 (cycles 3 and 4).
    const cudaParticleType expectedCutSum =
        0.5 * (expectedEnergy[3] + expectedEnergy[6]) +
        2 * 0.5 * (expectedEnergy[4] + expectedEnergy[1]);
    std::cout << "  cutoff samples = " << st.cutoffSamples
              << "  energy sum = " << st.cutoffEnergySum << " (expected "
              << expectedCutSum
              << ")  electron-limited cycles = " << st.electronLimitedCycles
              << "\n";
    check(st.cutoffSamples == 3 &&
              closeRel(st.cutoffEnergySum, expectedCutSum) &&
              st.electronLimitedCycles == 1,
          "cutoff-energy and electron-limited diagnostics", allPassed);
  }

  // ════════════════════════════════════════════════════════════
  //  TEST 5: Reflect + compact (cutoff from cycle 4: 2)
  // ════════════════════════════════════════════════════════════
  printSeparator("TEST 5 — Reflect & compact surviving electrons");

  SpeciesParticle* d_outputBuf;
  cudaErrChk(
      cudaMalloc(&d_outputBuf, totalElecPlanet * sizeof(SpeciesParticle)));
  // Unused tail slots of each species segment are copied back below too.
  cudaErrChk(
      cudaMemset(d_outputBuf, 0, totalElecPlanet * sizeof(SpeciesParticle)));
  int* d_survivorCounters;
  cudaErrChk(cudaMalloc(&d_survivorCounters, nElecSpecies * sizeof(int)));
  cudaErrChk(cudaMemset(d_survivorCounters, 0, nElecSpecies * sizeof(int)));

  planetReflectCompactKernel<<<(totalElecPlanet + BLOCK - 1) / BLOCK, BLOCK>>>(
      d_elecPtrs, nElecSpecies, d_speciesOffsets, d_globalIdxBuf, d_cutoff,
      totalElecPlanet, d_outputBuf, d_survivorCounters, originX, originY,
      originZ, sphereRadius, doSphere);
  cudaErrChk(cudaDeviceSynchronize());

  int h_survivorCounters[2] = {0, 0};
  cudaErrChk(cudaMemcpy(h_survivorCounters, d_survivorCounters,
                        nElecSpecies * sizeof(int), cudaMemcpyDeviceToHost));
  std::vector<SpeciesParticle> h_outputBuf(totalElecPlanet);
  cudaErrChk(cudaMemcpy(h_outputBuf.data(), d_outputBuf,
                        totalElecPlanet * sizeof(SpeciesParticle),
                        cudaMemcpyDeviceToHost));

  std::cout << "  Survivors per species: A=" << h_survivorCounters[0]
            << "  B=" << h_survivorCounters[1] << "\n";
  check(h_survivorCounters[0] == 3 && h_survivorCounters[1] == 2,
        "survivors A1 A2 A3 and B1 B2", allPassed);

  bool reflectPass = true;
  for (int s = 0; s < nElecSpecies; s++) {
    for (int i = 0; i < h_survivorCounters[s]; i++) {
      const SpeciesParticle& p = h_outputBuf[h_speciesOffsets[s] + i];
      const cudaCommonType dx = p.get_x() - originX;
      const cudaCommonType dy = p.get_y() - originY;
      const cudaCommonType dz = p.get_z() - originZ;
      const cudaCommonType r = sqrt(dx * dx + dy * dy + dz * dz);
      printParticle(s == 0 ? "reflected A" : "reflected B", i, p);
      // Outside the sphere, and u reversed along the (1,0,0) normal.
      reflectPass &= r >= sphereRadius && p.get_u() < 0.0;
    }
  }
  check(reflectPass, "survivors outside the sphere with u reversed", allPassed);

  // ════════════════════════════════════════════════════════════
  //  Summary
  // ════════════════════════════════════════════════════════════
  printSeparator("SUMMARY");
  std::cout << (allPassed ? "  ALL TESTS PASSED\n\n"
                          : "  SOME TESTS FAILED\n\n");

  // ─── Cleanup ───────────────────────────────────────────────
  cudaFree(d_state);
  cudaFree(d_diag);
  cudaFree(d_energyBuf);
  cudaFree(d_globalIdxBuf);
  cudaFree(d_cutoff);
  cudaFree(d_outputBuf);
  cudaFree(d_survivorCounters);
  cudaFree(d_speciesOffsets);
  cudaFree(d_elecPtrs);
  // d_ionPA, d_elec0PA, d_elec1PA freed by cudaFree (raw device structs)
  cudaFree(d_ionPA);
  cudaFree(d_elec0PA);
  cudaFree(d_elec1PA);
  // Host arrayCUDA destructors free internal device arrays automatically

  return allPassed ? 0 : 1;
}

int main() {
  try {
    return runTest();
  } catch (const std::exception& error) {
    std::cerr << "planetTest failed: " << error.what() << "\n";
    return EXIT_FAILURE;
  }
}
