#ifndef RESTART_PARTICLE_CELL_METADATA_H
#define RESTART_PARTICLE_CELL_METADATA_H

#include <array>
#include <vector>

/**
 * @brief Rank-local active-cell particle layout written with restart files.
 *
 * The particle sorter classifies particles on the guarded local cell grid.
 * Restart metadata stores only active cells, flattened with x as the fastest
 * varying index:
 *   active = ix + iy * nx + iz * nx * ny
 */
struct RestartParticleCellMetadata {
    struct Species {
        std::vector<int> cellOffsets;
        std::vector<int> cellCounts;
    };

    std::array<int, 3> activeCellDims = {{0, 0, 0}};
    std::vector<Species> species;
    bool valid = false;

    int activeCellCount() const {
        return activeCellDims[0] * activeCellDims[1] * activeCellDims[2];
    }

    void resize(int speciesCount, int nx, int ny, int nz) {
        activeCellDims = {{nx, ny, nz}};
        species.resize(speciesCount);

        const int cells = activeCellCount();
        for (Species& entry : species) {
            entry.cellOffsets.resize(cells);
            entry.cellCounts.resize(cells);
        }
        valid = false;
    }
};

#endif // RESTART_PARTICLE_CELL_METADATA_H
