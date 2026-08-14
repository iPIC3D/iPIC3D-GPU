#ifndef ADAPTOR_H
#define ADAPTOR_H

// For iPic3D arrays
#include "Alloc.h"
// Access to simulation parameters
#include "Collective.h"
// Access to physical quantities
#include "EMfields3D.h"

namespace Adaptor {
void Initialize(const Collective* sim_params, const int start_x,
                const int start_y, const int start_z, const int nx,
                const int ny, const int nz, const double dx, const double dy,
                const double dz);

void Finalize();

/**
 * Submit this timestep to Catalyst and retain an accepted request for
 * CoProcess().  The return value is true only when satisfying that request
 * requires the host B/rho arrays to be current.
 */
bool RequestDataDescription(double time, unsigned int timeStep);

/** Consume the request retained by RequestDataDescription(), if any. */
void CoProcess(EMfields3D* EMf);
} // namespace Adaptor

#endif // ADAPTOR_H
