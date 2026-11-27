/**
 * @file profileMomentKernelNew.cu
 * @brief Profile of momentKernelNew, the moment deposit of one thread per
 * particle that the solver runs on the unsorted tail of the particle arrays
 * (the particles received from the neighbours), on particles in random order.
 *
 * --tail F deposits the last fraction F of the particles, as the solver
 * deposits the tail after the sorted particles; by default all of them.
 */

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <string>

#include "KernelProfile.cuh"
#include "momentKernel.cuh"

int main(int argc, char** argv) {
  using namespace profileKernels;
  KernelProfile profile(argc, argv, "momentKernelNew", Order::Shuffled,
                        "--tail", 0);
  const std::uint32_t n = profile.particleCount();
  const std::uint32_t tail = std::max<std::uint32_t>(
      1, (std::uint32_t)std::llround(profile.options().tail * n));
  const std::uint32_t stayed = n - tail;

  cudaMomentType* moments = profile.moments();

  char fraction[32];
  std::snprintf(fraction, sizeof(fraction), "tail=%g", profile.options().tail);
  profile.time(
      fraction, tail, profiling::Region::MomentDeposit,
      [&] { profile.clearMoments(); },
      [&] {
        momentKernelNew<<<getGridSize((int)tail, SMALL_BLOCK_SIZE),
                          SMALL_BLOCK_SIZE, 0, profile.stream()>>>(
            profile.pclsArrayDevice(), profile.gridCUDA(), moments,
            (int)stayed);
      });
  return profile.finish(profile.checkDepositedCharge(stayed, tail));
}
