#include "particleControlKernel.cuh"

// ======= Merging =======

/**
 * Velocity variance of the particles [first, first + count) of the warp's
 * cell, weighted by charge: the trace, about the mean velocity. Every lane
 * returns the same value; it is 0 for a cell without charge.
 */
__device__ __forceinline__ cudaParticleType cellVelocityVariance(
    const ParticleSoADevice& pcls, int first, int count, int lane) {
  cudaParticleType q = 0, qu = 0, qv = 0, qw = 0, qv2 = 0;
  for (int i = lane; i < count; i += WARP_SIZE) {
    const int p = first + i;
    const cudaParticleType qp = pcls.q[p];
    const cudaParticleType u = pcls.u[p];
    const cudaParticleType v = pcls.v[p];
    const cudaParticleType w = pcls.w[p];
    q += qp;
    qu += qp * u;
    qv += qp * v;
    qw += qp * w;
    qv2 += qp * (u * u + v * v + w * w);
  }
  q = warp_reduce_sum(q);
  qu = warp_reduce_sum(qu);
  qv = warp_reduce_sum(qv);
  qw = warp_reduce_sum(qw);
  qv2 = warp_reduce_sum(qv2);
  // the sums of a lane other than 0 are partial: take lane 0's
  q = __shfl_sync(WARP_FULL_MASK, q, 0);
  qu = __shfl_sync(WARP_FULL_MASK, qu, 0);
  qv = __shfl_sync(WARP_FULL_MASK, qv, 0);
  qw = __shfl_sync(WARP_FULL_MASK, qw, 0);
  qv2 = __shfl_sync(WARP_FULL_MASK, qv2, 0);
  const cudaParticleType variance =
      (qv2 - (qu * qu + qv * qv + qw * qw) / q) / q;
  // 0 for the round-off of a cold cell and for the NaN of a cell without
  // charge (fmax returns its other argument)
  return fmax(variance, (cudaParticleType)0);
}

/** Merge particle `from` into particle `into`: total charge, centre of
 *  charge, mean velocity. */
__device__ __forceinline__ void mergeInto(const ParticleSoADevice& pcls,
                                          int into, int from) {
  const cudaParticleType q1 = pcls.q[into];
  const cudaParticleType q2 = pcls.q[from];
  const cudaParticleType q = q1 + q2;
  // share of each particle; a pair without charge keeps its mid-point
  const cudaParticleType f1 = q != 0 ? q1 / q : (cudaParticleType)0.5;
  const cudaParticleType f2 = 1 - f1;
  pcls.q[into] = q;
  pcls.x[into] = f1 * pcls.x[into] + f2 * pcls.x[from];
  pcls.y[into] = f1 * pcls.y[into] + f2 * pcls.y[from];
  pcls.z[into] = f1 * pcls.z[into] + f2 * pcls.z[from];
  pcls.u[into] = f1 * pcls.u[into] + f2 * pcls.u[from];
  pcls.v[into] = f1 * pcls.v[into] + f2 * pcls.v[from];
  pcls.w[into] = f1 * pcls.w[into] + f2 * pcls.w[from];
}

/**
 * The lanes share the particles of the cell: particle i of the cell belongs
 * to lane i % WARP_SIZE. A departure flag is written and read by the lane of
 * its particle only, and a particle is modified (by lane 0) only after its
 * last read: the lanes exchange information through shuffles alone.
 */
__global__ void particleMergingKernel(const int* cellStart,
                                      const int* cellCount, int numCells,
                                      int targetPerCell, ParticleSoADevice pcls,
                                      departureArrayElementType* departure) {
  const uint32_t thread = blockIdx.x * blockDim.x + threadIdx.x;
  const int cell = thread / WARP_SIZE;
  const int lane = thread % WARP_SIZE;
  if (cell >= numCells)
    return;
  const int count = cellCount[cell];
  if (count <= targetPerCell)
    return;
  const int first = cellStart[cell];

  const cudaParticleType maxDv2 =
      MERGE_VELOCITY_TOLERANCE * cellVelocityVariance(pcls, first, count, lane);

  int merged = 0;
  for (int lead = 0; lead < count - 1; lead++) {
    const int leadIdx = first + lead;

    // A lead merged away earlier takes no partner: its charge is gone with
    // it. The lane of the lead knows.
    const int leadLane = lead % WARP_SIZE;
    int absorbed = lane == leadLane ? departure[leadIdx].dest != 0 : 0;
    absorbed = __shfl_sync(WARP_FULL_MASK, absorbed, leadLane);
    if (absorbed)
      continue;

    // The nearest particle in velocity, within the tolerance (inclusive),
    // among the later particles of this lane that are still there; the first
    // of equals.
    const cudaParticleType u1 = pcls.u[leadIdx];
    const cudaParticleType v1 = pcls.v[leadIdx];
    const cudaParticleType w1 = pcls.w[leadIdx];
    cudaParticleType best = maxDv2;
    int bestIdx = -1;
    // first particle of this lane after the lead
    for (int i = lead + 1 + ((lane - lead - 1) & (WARP_SIZE - 1)); i < count;
         i += WARP_SIZE) {
      const int p = first + i;
      const cudaParticleType du = pcls.u[p] - u1;
      const cudaParticleType dv = pcls.v[p] - v1;
      const cudaParticleType dw = pcls.w[p] - w1;
      const cudaParticleType dv2 = du * du + dv * dv + dw * dw;
      if ((dv2 < best || (bestIdx < 0 && dv2 == best)) &&
          departure[p].dest == 0) {
        best = dv2;
        bestIdx = i;
      }
    }

    // Most leads have no particle within the tolerance.
    if (__ballot_sync(WARP_FULL_MASK, bestIdx >= 0) == 0)
      continue;

    // The nearest over the warp, the first of equals, to every lane.
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
      const cudaParticleType otherBest =
          __shfl_down_sync(WARP_FULL_MASK, best, offset);
      const int otherIdx = __shfl_down_sync(WARP_FULL_MASK, bestIdx, offset);
      if (otherIdx >= 0 && (bestIdx < 0 || otherBest < best ||
                            (otherBest == best && otherIdx < bestIdx))) {
        best = otherBest;
        bestIdx = otherIdx;
      }
    }
    bestIdx = __shfl_sync(WARP_FULL_MASK, bestIdx, 0);

    if (lane == bestIdx % WARP_SIZE)
      departure[first + bestIdx].dest = departureArrayElementType::DELETE;
    if (lane == 0)
      mergeInto(pcls, leadIdx, first + bestIdx);
    merged++;
    if (count - merged <= targetPerCell)
      return;
  }
}

// ======= Splitting =======

__global__ void particleSplittingKernel(ParticleSoADevice pcls, uint32_t count,
                                        uint32_t shift, SplitGeometry geometry,
                                        ParticleIDGenerator idGenerator) {
  const uint32_t t = blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= count)
    return;
  const uint32_t nop = pcls.nop;
  // count <= nop: the parents of two threads differ
  const uint32_t parent = (uint32_t)(((uint64_t)t * nop / count + shift) % nop);
  const uint32_t twin = nop + t;

  // Along one axis the weight of a node is linear inside the cell: the pair
  // at +delta and -delta deposits what the parent did.
  const int a = geometry.axis[(parent + shift) % (uint32_t)geometry.axisCount];
  cudaParticleType* const position[3] = {pcls.x, pcls.y, pcls.z};
  const cudaParticleType xa = position[a][parent];
  const cudaParticleType lowerFace =
      geometry.start[a] +
      floor((xa - geometry.start[a]) * geometry.invd[a]) * geometry.d[a];
  const cudaParticleType toFace =
      fmin(xa - lowerFace, lowerFace + geometry.d[a] - xa);
  // both particles stay in the cell of the parent
  const cudaParticleType delta =
      fmax(fmin(geometry.halfSpacing[a], (cudaParticleType)0.5 * toFace),
           (cudaParticleType)0);

  const cudaParticleType halfCharge = (cudaParticleType)0.5 * pcls.q[parent];
  pcls.q[parent] = halfCharge;
  pcls.q[twin] = halfCharge;
  pcls.x[twin] = pcls.x[parent];
  pcls.y[twin] = pcls.y[parent];
  pcls.z[twin] = pcls.z[parent];
  pcls.u[twin] = pcls.u[parent];
  pcls.v[twin] = pcls.v[parent];
  pcls.w[twin] = pcls.w[parent];
  position[a][parent] = xa - delta;
  position[a][twin] = xa + delta;
  if (pcls.trackParticleID)
    pcls.id[twin] = idGenerator.generateID();
}
