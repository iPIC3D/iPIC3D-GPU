#ifndef PARTICLE_SUMS_CUH
#define PARTICLE_SUMS_CUH

/**
 * @file particleSums.cuh
 * @brief The kinetic energy, momentum and charge of the particles of a species
 * on a rank, summed on the GPU for the energy diagnostics
 * (ConservedQuantities.txt), so that they need no copy of the particles to
 * the host.
 */

#include "cudaReduction.cuh"
#include "cudaTypeDef.cuh"

#include <algorithm>
#include <cstddef>
#include <cstdint>

/** Threads of a block of the sums (a block size of blockReduce). */
constexpr unsigned int kParticleSumsBlock = 256;
/** Most blocks of a launch; a grid-stride loop covers the other particles. */
constexpr unsigned int kParticleSumsMaxBlocks = 1024;
/** The doubles of a ParticleSums (an MPI reduction of them as doubles). */
constexpr int kParticleSumsValues = 5;

/** The sums of the particles of a species on a rank. */
struct ParticleSums {
  double energy;      // kinetic energy
  double momentum[3]; // momentum vector
  double charge;      // total charge
};
static_assert(sizeof(ParticleSums) == kParticleSumsValues * sizeof(double),
              "ParticleSums is reduced as kParticleSumsValues doubles");

/**
 * @brief Adds to *sums the sums of the nop particles of the arrays (a species):
 * with the relativistic push the kinetic energy (gamma - 1) m c^2 and the
 * momentum gamma m v, otherwise 1/2 m v^2 and m v; m = q / qom (the terms
 * are summed with q, and a thread's sums divided by qom once). In the
 * relativistic case a particle with |v| >= c (possible only in loaded data,
 * the mover deletes it) is left out of them. The charge sums every particle.
 * The same sums as ParticleSoAHost::getLocalKinetic plus the charge, in another
 * order (atomicAdd of the block sums).
 */
template <unsigned int blockSize>
__global__ void particleSumsKernel(const cudaPclType_U* __restrict__ u,
                                   const cudaPclType_V* __restrict__ v,
                                   const cudaPclType_W* __restrict__ w,
                                   const cudaPclType_Q* __restrict__ q,
                                   uint32_t nop, double invQom, double invC2,
                                   bool relativistic,
                                   ParticleSums* __restrict__ sums) {
  __shared__ double sdata[blockSize];
  double energy = 0.0, px = 0.0, py = 0.0, pz = 0.0, charge = 0.0;
  for (size_t p = blockIdx.x * (size_t)blockSize + threadIdx.x; p < nop;
       p += (size_t)blockSize * gridDim.x) {
    const double ux = u[p], uy = v[p], uz = w[p], qp = q[p];
    charge += qp;
    const double v2 = ux * ux + uy * uy + uz * uz;
    // (gamma - 1) m c^2 = gamma^2 / (gamma + 1) m v^2, free of the
    // cancellation of gamma - 1 at small v
    double gamma = 1.0, keFactor = 0.5;
    if (relativistic) {
      if (!(v2 * invC2 < 1.0))
        continue;
      gamma = 1.0 / sqrt(1.0 - v2 * invC2);
      keFactor = gamma * gamma / (gamma + 1.0);
    }
    const double qGamma = gamma * qp;
    energy += keFactor * qp * v2;
    px += qGamma * ux;
    py += qGamma * uy;
    pz += qGamma * uz;
  }

  const double partial[kParticleSumsValues] = {
      energy * invQom, px * invQom, py * invQom, pz * invQom, charge};
  double* const total[kParticleSumsValues] = {
      &sums->energy, &sums->momentum[0], &sums->momentum[1], &sums->momentum[2],
      &sums->charge};
  for (int k = 0; k < kParticleSumsValues; k++) {
    if (k > 0)
      __syncthreads(); // blockReduce reuses sdata
    const double blockTotal = cudaReduction::blockReduce<blockSize>(
        partial[k], sdata, cudaReduction::SumOp());
    if (threadIdx.x == 0)
      atomicAdd(total[k], blockTotal);
  }
}

/**
 * @brief Adds the sums of the nop particles of a species (device arrays) to
 * the device ParticleSums *sums, on stream; the caller zeroes *sums first.
 */
inline void launchParticleSums(const cudaPclType_U* u, const cudaPclType_V* v,
                               const cudaPclType_W* w, const cudaPclType_Q* q,
                               uint32_t nop, double qom, double c,
                               bool relativistic, ParticleSums* sums,
                               cudaStream_t stream) {
  if (nop == 0)
    return;
  const unsigned int blocks = (unsigned int)std::min<size_t>(
      ((size_t)nop + kParticleSumsBlock - 1) / kParticleSumsBlock,
      kParticleSumsMaxBlocks);
  particleSumsKernel<kParticleSumsBlock>
      <<<blocks, kParticleSumsBlock, 0, stream>>>(
          u, v, w, q, nop, 1.0 / qom, 1.0 / (c * c), relativistic, sums);
  cudaErrChk(cudaGetLastError());
}

#endif // PARTICLE_SUMS_CUH
