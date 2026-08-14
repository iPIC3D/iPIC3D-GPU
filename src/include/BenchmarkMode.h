#ifndef IPIC3D_BENCHMARK_MODE_H
#define IPIC3D_BENCHMARK_MODE_H

// Direct/non-CMake builds retain the normal runtime behavior.
#ifndef BENCHMARK_MODE
#define BENCHMARK_MODE 0
#endif

// Capability aliases for preprocessor guards. Use the namespaced constexpr
// values below in ordinary C++ expressions.
#define IPIC3D_COMPILE_DISK_OUTPUT (BENCHMARK_MODE == 0)
#define IPIC3D_COMPILE_DIAGNOSTIC_CALCULATIONS (BENCHMARK_MODE < 2)

static_assert(BENCHMARK_MODE >= 0 && BENCHMARK_MODE <= 2,
              "BENCHMARK_MODE must be 0, 1, or 2");

namespace BenchmarkConfig {

inline constexpr int MODE = BENCHMARK_MODE;
inline constexpr bool DISK_OUTPUT_ENABLED = IPIC3D_COMPILE_DISK_OUTPUT;
// Data-analysis and heat-flux diagnostic calculations remain part of level-1
// benchmarks even though their host-transfer and serialization stages are
// removed. Level 2 measures the simulation without those calculations.
inline constexpr bool DIAGNOSTIC_CALCULATIONS_ENABLED =
    IPIC3D_COMPILE_DIAGNOSTIC_CALCULATIONS;
inline constexpr bool DATA_ANALYSIS_ENABLED = DIAGNOSTIC_CALCULATIONS_ENABLED;

} // namespace BenchmarkConfig

#endif // IPIC3D_BENCHMARK_MODE_H
