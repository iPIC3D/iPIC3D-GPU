#ifndef BC_PARTICLES_H
#define BC_PARTICLES_H

/** The particle boundary conditions of a face (bcPface* in the input file). */
namespace BCparticles {
enum Enum {
  EXIT = 0,
  PERFECT_MIRROR = 1,
  OPEN_INFLOW = 2,
  OPEN_OUTFLOW = 3,
};

/** Cells of the particle layer of an open face.  An open-inflow face empties
 *  its layer in the mover and refills it (ParticleSoAHost::
 *  repopulate_particlesInfo, ParticleCommInjection::injectionFaces); an
 *  open-outflow face makes the plasma beyond it from its layer
 *  (ParticleSoAHost::openbc_particles_outflowInfo), and the field outflow
 *  layer contains it (FieldBoundarySpec::outflowLayers). */
inline constexpr int kOpenFaceLayers = 3;
} // namespace BCparticles

#endif // BC_PARTICLES_H
