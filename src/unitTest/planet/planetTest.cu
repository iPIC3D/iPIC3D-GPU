/**
 * @file planetTest.cu
 * @brief Unit test for the planet boundary-condition kernels defined in
 *        planetKernel.cu.
 *
 * What is tested
 * ──────────────
 *  1. Energy computation  (planetEnergyKernel)
 *  2. Bitonic sort         (bitonicPadKernel + bitonicSortStepKernel)
 *     → particles are printed and verified to be in descending energy order
 *  3. Ion charge reduction (planetChargeReductionKernel)
 *  4. Charge cutoff        (chargeCutoffKernel)
 *     → quasi-neutrality: ∑|q_removed_elec| == ∑|q_ions|
 *  5. Reflect + compact    (planetReflectCompactKernel)
 *     → surviving particles are outside the planet sphere after reflection
 *
 * Test data
 * ─────────
 *  • 1 ion  species   :  5 particles,  q = +2.0 each  → total |Q_ion| = 10.0
 *  • 2 electron species:
 *      – species 0: 4 particles  (various q < 0, various velocities)
 *      – species 1: 3 particles  (various q < 0, various velocities)
 *    Total electron |q| = 11.5 > 10.0, so some electrons must be removed.
 *
 *  Expected sorted energies (descending):
 *    31.25  24.0  9.0  3.0  2.0  0.5  0.25
 *
 *  Expected cutoff index = 5  (remove sorted indices 0..4,
 *                               survivors at sorted indices 5..6)
 *  Removed ∑|q| = 2.5 + 3.0 + 2.0 + 1.5 + 1.0 = 10.0 == ion charge ✓
 */

#include <iostream>
#include <iomanip>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <algorithm>
#include <numeric>
#include <cstring>

// planetKernel.cuh pulls in cudaTypeDef, particleArrayCUDA, particleExchange,
// hashedSum, arrayCUDA — in the correct order.
#include "planetKernel.cuh"


// ─── helpers ───────────────────────────────────────────────────

static constexpr int BLOCK = 256;

/** Next power of two >= n */
static int nextPow2(int n) {
    int p = 1;
    while (p < n) p <<= 1;
    return p;
}

/** Compute kinetic energy on host for verification */
static cudaParticleType hostEnergy(const SpeciesParticle& p, cudaParticleType qom) {
    cudaParticleType absq = fabs(p.get_q());
    cudaParticleType mass = absq / fabs(qom);
    cudaParticleType v2   = p.get_u() * p.get_u()
                          + p.get_v() * p.get_v()
                          + p.get_w() * p.get_w();
    return 0.5 * mass * v2;
}

// ─── printing helpers ──────────────────────────────────────────

static void printParticle(const char* tag, int idx, const SpeciesParticle& p,
                          cudaParticleType energy = -1.0) {
    std::cout << "  " << std::left << std::setw(12) << tag
              << " [" << std::setw(2) << idx << "]"
              << "  pos=(" << std::setw(8) << p.get_x()
              << ", "      << std::setw(8) << p.get_y()
              << ", "      << std::setw(8) << p.get_z() << ")"
              << "  vel=(" << std::setw(8) << p.get_u()
              << ", "      << std::setw(8) << p.get_v()
              << ", "      << std::setw(8) << p.get_w() << ")"
              << "  q=" << std::setw(8) << p.get_q();
    if (energy >= 0.0) std::cout << "  E=" << energy;
    std::cout << "\n";
}

static void printSeparator(const char* title) {
    std::cout << "\n═══════════════════════════════════════════════════\n"
              << " " << title
              << "\n═══════════════════════════════════════════════════\n";
}

// ═══════════════════════════════════════════════════════════════
//  MAIN
// ═══════════════════════════════════════════════════════════════

int main() {
    bool allPassed = true;

    // ─── Planet sphere parameters ──────────────────────────────
    const cudaCommonType originX = 5.0;
    const cudaCommonType originY = 5.0;
    const cudaCommonType originZ = 5.0;
    const cudaCommonType sphereRadius = 1.0;
    const int doSphere = 1;  // 3-D

    // ─── Species parameters ────────────────────────────────────
    const cudaParticleType qomIon  =  1.0;   // ion   charge-to-mass ratio
    const cudaParticleType qomElec = -1.0;    // electron charge-to-mass ratio
    const int nIonSpecies  = 1;
    const int nElecSpecies = 2;

    // ────────────────────────────────────────────────────────────
    //  1. Create ion particles (inside the sphere)
    // ────────────────────────────────────────────────────────────
    const int nIonPlanet = 5;
    SpeciesParticle ionPcls[nIonPlanet];
    for (int i = 0; i < nIonPlanet; i++) {
        // All at (5.3, 5.0, 5.0), inside sphere (r=0.3 < 1.0)
        // Velocity doesn't matter for ions in this test
        ionPcls[i].set(
            /*u*/ 1.0, /*v*/ 0.0, /*w*/ 0.0,
            /*q*/ 2.0,
            /*x*/ 5.3, /*y*/ 5.0, /*z*/ 5.0,
            /*t*/ (cudaParticleType)i);
    }

    // ────────────────────────────────────────────────────────────
    //  2. Create electron particles (inside the sphere)
    // ────────────────────────────────────────────────────────────
    //  Electron species 0:  4 particles
    //    idx  q       vel            mass=|q|/|qom|  E = 0.5*m*v²
    //     0  -3.0    (4, 0, 0)      3.0              24.00
    //     1  -2.0    (3, 0, 0)      2.0               9.00
    //     2  -1.0    (2, 0, 0)      1.0               2.00
    //     3  -0.5    (1, 0, 0)      0.5               0.25
    const int nElec0 = 4;
    SpeciesParticle elec0Pcls[nElec0];
    {
        cudaParticleType qs[]  = { -3.0, -2.0, -1.0, -0.5 };
        cudaParticleType us[]  = {  4.0,  3.0,  2.0,  1.0 };
        for (int i = 0; i < nElec0; i++) {
            elec0Pcls[i].set(
                us[i], 0.0, 0.0,   // velocity
                qs[i],              // charge
                5.3, 5.0, 5.0,     // position
                (cudaParticleType)i);
        }
    }

    //  Electron species 1:  3 particles
    //    idx  q       vel            mass=|q|/|qom|  E = 0.5*m*v²
    //     0  -2.5    (5, 0, 0)      2.5              31.25
    //     1  -1.5    (2, 0, 0)      1.5               3.00
    //     2  -1.0    (1, 0, 0)      1.0               0.50
    const int nElec1 = 3;
    SpeciesParticle elec1Pcls[nElec1];
    {
        cudaParticleType qs[]  = { -2.5, -1.5, -1.0 };
        cudaParticleType us[]  = {  5.0,  2.0,  1.0 };
        for (int i = 0; i < nElec1; i++) {
            elec1Pcls[i].set(
                us[i], 0.0, 0.0,
                qs[i],
                5.3, 5.0, 5.0,
                (cudaParticleType)i);
        }
    }

    const int totalElecPlanet = nElec0 + nElec1;  // 7

    // ─── Expected values ───────────────────────────────────────
    // Merged energy buffer layout:  [E0_0, E0_1, E0_2, E0_3, E1_0, E1_1, E1_2]
    //                                 24    9     2    0.25  31.25  3.0   0.5
    // Descending sort:  31.25 24 9 3 2 0.5 0.25
    // Sorted indices:    4    0  1 5 2  6   3
    // Prefix-sum |q|:   2.5  5.5 7.5 9 10  --  --   → cutoff = 5
    const cudaParticleType expectedIonCharge = 10.0;
    const int              expectedCutoff    = 5;
    const cudaParticleType expectedRemovedQ  = 10.0;  // 2.5+3+2+1.5+1
    const cudaParticleType expectedSortedEnergies[] =
        { 31.25, 24.0, 9.0, 3.0, 2.0, 0.5, 0.25 };

    // ════════════════════════════════════════════════════════════
    //  Allocate device buffers via arrayCUDA (= planetArray)
    // ════════════════════════════════════════════════════════════

    // Ion planetArray (1 species)
    planetArray ionPA(ionPcls, nIonPlanet);
    planetArray* d_ionPA = ionPA.copyToDevice();

    // Electron planetArrays (2 species)
    planetArray elec0PA(elec0Pcls, nElec0);
    planetArray* d_elec0PA = elec0PA.copyToDevice();

    planetArray elec1PA(elec1Pcls, nElec1);
    planetArray* d_elec1PA = elec1PA.copyToDevice();

    // Array of device pointers to electron planetArrays (for chargeCutoff / reflectCompact)
    planetArray* h_elecPtrs[2] = { d_elec0PA, d_elec1PA };
    planetArray** d_elecPtrs;
    cudaErrChk(cudaMalloc(&d_elecPtrs, nElecSpecies * sizeof(planetArray*)));
    cudaErrChk(cudaMemcpy(d_elecPtrs, h_elecPtrs,
                           nElecSpecies * sizeof(planetArray*),
                           cudaMemcpyHostToDevice));

    // Species offsets into merged buffers:  species0 starts at 0, species1 at nElec0
    int h_speciesOffsets[2] = { 0, nElec0 };
    int* d_speciesOffsets;
    cudaErrChk(cudaMalloc(&d_speciesOffsets, nElecSpecies * sizeof(int)));
    cudaErrChk(cudaMemcpy(d_speciesOffsets, h_speciesOffsets,
                           nElecSpecies * sizeof(int), cudaMemcpyHostToDevice));

    // ════════════════════════════════════════════════════════════
    //  TEST 1: Ion charge reduction
    // ════════════════════════════════════════════════════════════
    printSeparator("TEST 1 — Ion charge reduction");

    cudaParticleType* d_ionCharge;
    cudaErrChk(cudaMalloc(&d_ionCharge, sizeof(cudaParticleType)));
    cudaErrChk(cudaMemset(d_ionCharge, 0, sizeof(cudaParticleType)));

    int gridIon = (nIonPlanet + BLOCK - 1) / BLOCK;
    planetChargeReductionKernel<<<gridIon, BLOCK, BLOCK * sizeof(cudaParticleType)>>>(
        d_ionPA, nIonPlanet, d_ionCharge);
    cudaErrChk(cudaDeviceSynchronize());

    cudaParticleType h_ionCharge = 0;
    cudaErrChk(cudaMemcpy(&h_ionCharge, d_ionCharge,
                           sizeof(cudaParticleType), cudaMemcpyDeviceToHost));

    std::cout << "  Total ion |Q|  = " << h_ionCharge
              << "  (expected " << expectedIonCharge << ")\n";

    if (fabs(h_ionCharge - expectedIonCharge) > 1e-10) {
        std::cerr << "  *** FAIL: ion charge mismatch ***\n";
        allPassed = false;
    } else {
        std::cout << "  PASS\n";
    }

    // ════════════════════════════════════════════════════════════
    //  TEST 2: Electron energy computation
    // ════════════════════════════════════════════════════════════
    printSeparator("TEST 2 — Electron energy computation");

    cudaParticleType* d_energyBuf;
    uint32_t*         d_globalIdxBuf;
    cudaErrChk(cudaMalloc(&d_energyBuf,    totalElecPlanet * sizeof(cudaParticleType)));
    cudaErrChk(cudaMalloc(&d_globalIdxBuf, totalElecPlanet * sizeof(uint32_t)));

    // Species 0
    int grid0 = (nElec0 + BLOCK - 1) / BLOCK;
    planetEnergyKernel<<<grid0, BLOCK>>>(
        d_elec0PA, nElec0, qomElec,
        d_energyBuf, d_globalIdxBuf, /*speciesOffset=*/0);

    // Species 1
    int grid1 = (nElec1 + BLOCK - 1) / BLOCK;
    planetEnergyKernel<<<grid1, BLOCK>>>(
        d_elec1PA, nElec1, qomElec,
        d_energyBuf, d_globalIdxBuf, /*speciesOffset=*/nElec0);

    cudaErrChk(cudaDeviceSynchronize());

    // Copy back and verify
    std::vector<cudaParticleType> h_energy(totalElecPlanet);
    std::vector<uint32_t>         h_gidx(totalElecPlanet);
    cudaErrChk(cudaMemcpy(h_energy.data(), d_energyBuf,
                           totalElecPlanet * sizeof(cudaParticleType),
                           cudaMemcpyDeviceToHost));
    cudaErrChk(cudaMemcpy(h_gidx.data(), d_globalIdxBuf,
                           totalElecPlanet * sizeof(uint32_t),
                           cudaMemcpyDeviceToHost));

    std::cout << "  Merged energy buffer (before sort):\n";
    // Also compute expected energies on host
    std::vector<cudaParticleType> expectedEnergiesUnsorted(totalElecPlanet);
    for (int i = 0; i < nElec0; i++)
        expectedEnergiesUnsorted[i] = hostEnergy(elec0Pcls[i], qomElec);
    for (int i = 0; i < nElec1; i++)
        expectedEnergiesUnsorted[nElec0 + i] = hostEnergy(elec1Pcls[i], qomElec);

    bool energyPass = true;
    for (int i = 0; i < totalElecPlanet; i++) {
        int species = (i < nElec0) ? 0 : 1;
        int local   = (i < nElec0) ? i : i - nElec0;
        std::cout << "    gidx=" << std::setw(2) << h_gidx[i]
                  << "  species=" << species << "  local=" << local
                  << "  E=" << std::setw(10) << h_energy[i]
                  << "  (expected " << expectedEnergiesUnsorted[i] << ")\n";

        if (fabs(h_energy[i] - expectedEnergiesUnsorted[i]) > 1e-10) {
            std::cerr << "  *** FAIL: energy mismatch at index " << i << " ***\n";
            energyPass = false;
        }
    }
    if (energyPass) std::cout << "  PASS\n";
    else allPassed = false;

    // ════════════════════════════════════════════════════════════
    //  TEST 3: Bitonic sort (descending by energy)
    // ════════════════════════════════════════════════════════════
    printSeparator("TEST 3 — Bitonic sort (descending energy)");

    int paddedN = nextPow2(totalElecPlanet);

    // Pad
    if (paddedN > totalElecPlanet) {
        int padGrid = (paddedN - totalElecPlanet + BLOCK - 1) / BLOCK;
        bitonicPadKernel<<<padGrid, BLOCK>>>(
            d_energyBuf, d_globalIdxBuf, totalElecPlanet, paddedN);
        cudaErrChk(cudaDeviceSynchronize());
    }

    // Bitonic sort steps
    for (int k = 2; k <= paddedN; k <<= 1) {
        for (int j = k >> 1; j > 0; j >>= 1) {
            int sortGrid = (paddedN + BLOCK - 1) / BLOCK;
            bitonicSortStepKernel<<<sortGrid, BLOCK>>>(
                d_energyBuf, d_globalIdxBuf, j, k, paddedN);
            cudaErrChk(cudaDeviceSynchronize());
        }
    }

    // Copy sorted data back
    cudaErrChk(cudaMemcpy(h_energy.data(), d_energyBuf,
                           totalElecPlanet * sizeof(cudaParticleType),
                           cudaMemcpyDeviceToHost));
    cudaErrChk(cudaMemcpy(h_gidx.data(), d_globalIdxBuf,
                           totalElecPlanet * sizeof(uint32_t),
                           cudaMemcpyDeviceToHost));

    std::cout << "  Sorted electron particles (descending energy):\n";
    std::cout << "  " << std::setw(6) << "rank" << std::setw(8) << "gidx"
              << std::setw(10) << "species" << std::setw(8) << "local"
              << std::setw(14) << "energy" << std::setw(14) << "expected"
              << std::setw(10) << "|q|" << "\n";
    std::cout << "  " << std::string(70, '-') << "\n";

    bool sortPass = true;
    for (int i = 0; i < totalElecPlanet; i++) {
        int gidx = (int)h_gidx[i];
        int species = (gidx < nElec0) ? 0 : 1;
        int local   = (gidx < nElec0) ? gidx : gidx - nElec0;
        cudaParticleType absq = (species == 0)
            ? fabs(elec0Pcls[local].get_q())
            : fabs(elec1Pcls[local].get_q());

        std::cout << "  " << std::setw(6) << i
                  << std::setw(8) << gidx
                  << std::setw(10) << species
                  << std::setw(8) << local
                  << std::setw(14) << h_energy[i]
                  << std::setw(14) << expectedSortedEnergies[i]
                  << std::setw(10) << absq
                  << "\n";

        if (fabs(h_energy[i] - expectedSortedEnergies[i]) > 1e-10) {
            std::cerr << "  *** FAIL: sort order mismatch at rank " << i << " ***\n";
            sortPass = false;
        }
        // Also check descending order
        if (i > 0 && h_energy[i] > h_energy[i - 1]) {
            std::cerr << "  *** FAIL: not descending at rank " << i << " ***\n";
            sortPass = false;
        }
    }
    if (sortPass) std::cout << "  PASS\n";
    else allPassed = false;

    // ════════════════════════════════════════════════════════════
    //  TEST 4: Charge cutoff + quasi-neutrality
    // ════════════════════════════════════════════════════════════
    printSeparator("TEST 4 — Charge cutoff & quasi-neutrality");

    int* d_cutoff;
    cudaErrChk(cudaMalloc(&d_cutoff, sizeof(int)));
    cudaErrChk(cudaMemset(d_cutoff, 0, sizeof(int)));

    chargeCutoffKernel<<<1, 1>>>(
        d_elecPtrs, nElecSpecies, d_speciesOffsets,
        d_globalIdxBuf, totalElecPlanet,
        d_ionCharge, d_cutoff);
    cudaErrChk(cudaDeviceSynchronize());

    int h_cutoff = 0;
    cudaErrChk(cudaMemcpy(&h_cutoff, d_cutoff,
                           sizeof(int), cudaMemcpyDeviceToHost));

    std::cout << "  Cutoff index  = " << h_cutoff
              << "  (expected " << expectedCutoff << ")\n";

    // Compute removed charge from sorted order
    cudaParticleType removedQ = 0;
    std::cout << "\n  Particles REMOVED (highest energy, indices 0.."
              << h_cutoff - 1 << "):\n";
    for (int i = 0; i < h_cutoff && i < totalElecPlanet; i++) {
        int gidx    = (int)h_gidx[i];
        int species = (gidx < nElec0) ? 0 : 1;
        int local   = (gidx < nElec0) ? gidx : gidx - nElec0;
        const SpeciesParticle& p = (species == 0) ? elec0Pcls[local] : elec1Pcls[local];
        cudaParticleType absq = fabs(p.get_q());
        removedQ += absq;
        printParticle(species == 0 ? "elec0" : "elec1", local, p, h_energy[i]);
    }

    std::cout << "\n  Particles in SPHERE (survivors, indices "
              << h_cutoff << ".." << totalElecPlanet - 1 << "):\n";
    for (int i = h_cutoff; i < totalElecPlanet; i++) {
        int gidx    = (int)h_gidx[i];
        int species = (gidx < nElec0) ? 0 : 1;
        int local   = (gidx < nElec0) ? gidx : gidx - nElec0;
        const SpeciesParticle& p = (species == 0) ? elec0Pcls[local] : elec1Pcls[local];
        printParticle(species == 0 ? "elec0" : "elec1", local, p, h_energy[i]);
    }

    std::cout << "\n  ∑|q| removed electrons = " << removedQ
              << "  (expected " << expectedRemovedQ << ")\n";
    std::cout << "  ∑|q| ions              = " << h_ionCharge << "\n";

    bool cutoffPass = (h_cutoff == expectedCutoff);
    bool neutralityPass = (fabs(removedQ - h_ionCharge) < 1e-10);

    if (!cutoffPass) {
        std::cerr << "  *** FAIL: cutoff index mismatch ***\n";
        allPassed = false;
    }
    if (!neutralityPass) {
        std::cerr << "  *** FAIL: quasi-neutrality violated — "
                  << "removed |q|=" << removedQ
                  << " != ion |Q|=" << h_ionCharge << " ***\n";
        allPassed = false;
    }
    if (cutoffPass && neutralityPass) {
        std::cout << "  PASS — quasi-neutrality satisfied: "
                  << "removed electron charge matches ion charge\n";
    }

    // ════════════════════════════════════════════════════════════
    //  TEST 5: Reflect + compact
    // ════════════════════════════════════════════════════════════
    printSeparator("TEST 5 — Reflect & compact surviving electrons");

    SpeciesParticle* d_outputBuf;
    cudaErrChk(cudaMalloc(&d_outputBuf, totalElecPlanet * sizeof(SpeciesParticle)));

    int* d_survivorCounters;
    cudaErrChk(cudaMalloc(&d_survivorCounters, nElecSpecies * sizeof(int)));
    cudaErrChk(cudaMemset(d_survivorCounters, 0, nElecSpecies * sizeof(int)));

    int reflectGrid = (totalElecPlanet + BLOCK - 1) / BLOCK;
    planetReflectCompactKernel<<<reflectGrid, BLOCK>>>(
        d_elecPtrs, nElecSpecies,
        d_speciesOffsets,
        d_globalIdxBuf,
        d_cutoff,
        totalElecPlanet,
        d_outputBuf,
        d_survivorCounters,
        originX, originY, originZ,
        sphereRadius, doSphere);
    cudaErrChk(cudaDeviceSynchronize());

    // Read survivor counts
    int h_survivorCounters[2] = {0, 0};
    cudaErrChk(cudaMemcpy(h_survivorCounters, d_survivorCounters,
                           nElecSpecies * sizeof(int), cudaMemcpyDeviceToHost));

    int totalSurvivors = h_survivorCounters[0] + h_survivorCounters[1];
    std::cout << "  Survivors per species: s0=" << h_survivorCounters[0]
              << "  s1=" << h_survivorCounters[1]
              << "  total=" << totalSurvivors << "\n";

    // Copy output buffer back
    std::vector<SpeciesParticle> h_outputBuf(totalElecPlanet);
    cudaErrChk(cudaMemcpy(h_outputBuf.data(), d_outputBuf,
                           totalElecPlanet * sizeof(SpeciesParticle),
                           cudaMemcpyDeviceToHost));

    bool reflectPass = true;
    const cudaCommonType eps = sphereRadius * 1e-4;

    // Check species 0 survivors
    std::cout << "\n  Reflected survivors (species 0):\n";
    for (int i = 0; i < h_survivorCounters[0]; i++) {
        const SpeciesParticle& p = h_outputBuf[h_speciesOffsets[0] + i];
        cudaCommonType dx = p.get_x() - originX;
        cudaCommonType dy = p.get_y() - originY;
        cudaCommonType dz = p.get_z() - originZ;
        cudaCommonType r  = sqrt(dx*dx + dy*dy + dz*dz);
        std::cout << "    pos=(" << p.get_x() << ", " << p.get_y() << ", " << p.get_z()
                  << ")  vel=(" << p.get_u() << ", " << p.get_v() << ", " << p.get_w()
                  << ")  q=" << p.get_q() << "  r=" << r << "\n";
        if (r < sphereRadius) {
            std::cerr << "    *** FAIL: reflected particle still inside sphere (r="
                      << r << " < " << sphereRadius << ") ***\n";
            reflectPass = false;
        }
    }

    // Check species 1 survivors
    std::cout << "  Reflected survivors (species 1):\n";
    for (int i = 0; i < h_survivorCounters[1]; i++) {
        const SpeciesParticle& p = h_outputBuf[h_speciesOffsets[1] + i];
        cudaCommonType dx = p.get_x() - originX;
        cudaCommonType dy = p.get_y() - originY;
        cudaCommonType dz = p.get_z() - originZ;
        cudaCommonType r  = sqrt(dx*dx + dy*dy + dz*dz);
        std::cout << "    pos=(" << p.get_x() << ", " << p.get_y() << ", " << p.get_z()
                  << ")  vel=(" << p.get_u() << ", " << p.get_v() << ", " << p.get_w()
                  << ")  q=" << p.get_q() << "  r=" << r << "\n";
        if (r < sphereRadius) {
            std::cerr << "    *** FAIL: reflected particle still inside sphere (r="
                      << r << " < " << sphereRadius << ") ***\n";
            reflectPass = false;
        }
    }

    // Verify reflected velocity is reversed along normal
    // For particles originally at (5.3,5,5) the normal is (1,0,0)
    // so velocity should have u -> -u, v and w unchanged.
    std::cout << "\n  Velocity reflection check:\n";
    for (int s = 0; s < nElecSpecies; s++) {
        for (int i = 0; i < h_survivorCounters[s]; i++) {
            const SpeciesParticle& p = h_outputBuf[h_speciesOffsets[s] + i];
            // The original velocity u was positive; after specular reflection
            // along (1,0,0) it should be negative.
            if (p.get_u() >= 0.0) {
                std::cerr << "    *** FAIL: species " << s << " particle " << i
                          << " has u=" << p.get_u()
                          << " >= 0 — expected negative after reflection ***\n";
                reflectPass = false;
            } else {
                std::cout << "    species " << s << " pcl " << i
                          << ": u=" << p.get_u() << " (reflected, OK)\n";
            }
        }
    }

    if (reflectPass) std::cout << "  PASS\n";
    else allPassed = false;

    // ════════════════════════════════════════════════════════════
    //  Summary
    // ════════════════════════════════════════════════════════════
    printSeparator("SUMMARY");
    if (allPassed) {
        std::cout << "  ALL TESTS PASSED\n\n";
    } else {
        std::cout << "  SOME TESTS FAILED\n\n";
    }

    // ─── Cleanup ───────────────────────────────────────────────
    cudaFree(d_ionCharge);
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
