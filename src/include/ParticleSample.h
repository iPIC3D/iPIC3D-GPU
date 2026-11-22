#ifndef PARTICLE_SAMPLE_H
#define PARTICLE_SAMPLE_H

/**
 * @file ParticleSample.h
 * @brief The particle arrays that a particle writer writes (ParticleView),
 * and the host copy of the particles of a species that
 * ParticlesOutputTrackedFraction or ParticlesOutputMaxPerCell selects
 * (ParticleSample, particleOutputSelect.cuh).
 */

#include "OutputTagConfig.h" // ParticleOutputArrays
#include "ParticleSoAHost.h"
#include "cudaTypeDef.cuh"

#include <cassert>
#include <cstddef>
#include <type_traits>
#include <vector>

/** The particles of a species that a particle writer writes: the host copy
 *  of all of them (ParticleSoAHost) or of those that
 *  ParticlesOutputTrackedFraction or ParticlesOutputMaxPerCell selects
 *  (ParticleSample). An array the output does not write may be null,
 *  as may every array with no particle; hasID: the species has IDs. */
struct ParticleView {
  std::size_t nop = 0;
  const cudaPclType_X* x = nullptr;
  const cudaPclType_Y* y = nullptr;
  const cudaPclType_Z* z = nullptr;
  const cudaPclType_U* u = nullptr;
  const cudaPclType_V* v = nullptr;
  const cudaPclType_W* w = nullptr;
  const cudaPclType_Q* q = nullptr;
  const cudaPclType_ID* id = nullptr;
  bool hasID = false;
};

/** The array p of n values, or for n = 0 a valid address if p is null (a
 *  species without particles, or an array never filled): a writer's data. */
template <typename T> const T* nonNull(const T* p, std::size_t n) {
  static const T none{};
  assert(p || n == 0);
  return p ? p : &none;
}

/** Every particle of the host copy p. */
inline ParticleView viewOf(const ParticleSoAHost& p) {
  ParticleView view;
  view.nop = p.getNOP();
  view.x = p.getXall();
  view.y = p.getYall();
  view.z = p.getZall();
  view.u = p.getUall();
  view.v = p.getVall();
  view.w = p.getWall();
  view.q = p.getQall();
  view.id = p.getParticleIDall();
  view.hasID = p.tracksParticleID();
  return view;
}

/** Every particle of the ns host copies part[s]. */
inline std::vector<ParticleView> viewsOf(ParticleSoAHost* const* part, int ns) {
  std::vector<ParticleView> views(ns);
  for (int s = 0; s < ns; s++)
    views[s] = viewOf(*part[s]);
  return views;
}

/** The particles of a species that ParticlesOutputTrackedFraction or
 *  ParticlesOutputMaxPerCell selects, in page-locked memory: copied from the
 *  device, then written by the particle output. The arrays only grow. */
class ParticleSample {
public:
  /** n particles in the arrays of `arrays`: room for them; an array that
   *  grows loses its content. ids: the species has IDs (ParticleView). */
  void prepare(std::size_t n, const ParticleOutputArrays& arrays, bool ids) {
    nop_ = n;
    arrays_ = arrays;
    ids_ = ids;
    auto room = [n](auto& a) {
      using Array = std::decay_t<decltype(a)>;
      if (a.size() >= n)
        return;
      a = Array();          // the old array freed first
      a = Array(n + n / 4); // a margin: the counts change between outputs
    };
    if (arrays.position) {
      room(x_);
      room(y_);
      room(z_);
    }
    if (arrays.velocity) {
      room(u_);
      room(v_);
      room(w_);
    }
    if (arrays.charge)
      room(q_);
    if (arrays.id)
      room(id_);
  }

  cudaPclType_X* x() { return x_.data(); }
  cudaPclType_Y* y() { return y_.data(); }
  cudaPclType_Z* z() { return z_.data(); }
  cudaPclType_U* u() { return u_.data(); }
  cudaPclType_V* v() { return v_.data(); }
  cudaPclType_W* w() { return w_.data(); }
  cudaPclType_Q* q() { return q_.data(); }
  cudaPclType_ID* id() { return id_.data(); }

  /** The arrays of the last prepare. */
  ParticleView view() const {
    ParticleView view;
    view.nop = nop_;
    if (arrays_.position) {
      view.x = x_.data();
      view.y = y_.data();
      view.z = z_.data();
    }
    if (arrays_.velocity) {
      view.u = u_.data();
      view.v = v_.data();
      view.w = w_.data();
    }
    if (arrays_.charge)
      view.q = q_.data();
    if (arrays_.id)
      view.id = id_.data();
    view.hasID = ids_;
    return view;
  }

private:
  std::size_t nop_ = 0;
  ParticleOutputArrays arrays_;
  bool ids_ = false;
  PinnedVector<cudaPclType_X> x_;
  PinnedVector<cudaPclType_Y> y_;
  PinnedVector<cudaPclType_Z> z_;
  PinnedVector<cudaPclType_U> u_;
  PinnedVector<cudaPclType_V> v_;
  PinnedVector<cudaPclType_W> w_;
  PinnedVector<cudaPclType_Q> q_;
  PinnedVector<cudaPclType_ID> id_;
};

#endif // PARTICLE_SAMPLE_H
