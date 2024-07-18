#ifndef FEADAPTOR_HEADER
#define FEADAPTOR_HEADER

// For iPic3D arrays
#include "../include/Alloc.h"
// Access to simulation parameters
#include "../include/Collective.h"
// Access to physical quantities
#include "../include/EMfields3D.h"

namespace Adaptor {
void Initialize(const Collective *sim_params, const int start_x,
                const int start_y, const int start_z, const int nx,
                const int ny, const int nz, const double dx, const double dy,
                const double dz);

void Finalize();

void CoProcess(double time, unsigned int timeStep, EMfields3D *EMf);
} // namespace Adaptor

#endif
