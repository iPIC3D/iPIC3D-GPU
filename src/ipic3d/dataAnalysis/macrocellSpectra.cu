#include "macrocellSpectra.cuh"

#include "cudaTypeDef.cuh"
#include "gridCUDA.cuh"
#include "particleArrayCUDA.cuh"
#include "VCtopology3D.h"
#include "outputPrepare.h"

#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <stdexcept>
#include <cassert>
#include <filesystem>

namespace macrocellSpectra {

// ======= MacrocellPartition =======

bool MacrocellPartition::build(int Nx_int_, int Ny_int_, int Nz_int_,
                               int Cx_, int Cy_, int Cz_,
                               int globalOffX, int globalOffY, int globalOffZ)
{
    if (Cx_ <= 0 || Cy_ <= 0 || Cz_ <= 0)            return false;
    if (Nx_int_ <= 0 || Ny_int_ <= 0 || Nz_int_ <= 0) return false;
    if (Cx_ > Nx_int_ || Cy_ > Ny_int_ || Cz_ > Nz_int_) return false;

    Cx = Cx_; Cy = Cy_; Cz = Cz_;
    Nx_int = Nx_int_; Ny_int = Ny_int_; Nz_int = Nz_int_;
    globalOffsetX = globalOffX;
    globalOffsetY = globalOffY;
    globalOffsetZ = globalOffZ;

    auto tile = [](int N, int C, std::vector<MacrocellRange>& out) {
        const int M = (N + C - 1) / C;
        out.clear();
        out.reserve(M);
        for (int m = 0; m < M; ++m) {
            const int start = m * C;
            const int size  = (start + C <= N) ? C : (N - start);
            out.push_back({start, size});
        }
        return M;
    };

    Mx = tile(Nx_int, Cx, rangeX);
    My = tile(Ny_int, Cy, rangeY);
    Mz = tile(Nz_int, Cz, rangeZ);
    M  = Mx * My * Mz;
    return true;
}


// ======= macrocellSpectra2D =======

cudaCommonType macrocellSpectra2D::vmaxForSpecies(int species) {
    return (species == 0 || species == 2) ? MAX_VELOCITY_HIST_E
                                          : MAX_VELOCITY_HIST_I;
}

namespace {

// Helper: copy a host vector<MacrocellRange> to a device int array of size 2*N
// laid out as [start0, size0, start1, size1, ...].
int* uploadRanges(const std::vector<MacrocellRange>& ranges) {
    const size_t bytes = ranges.size() * 2 * sizeof(int);
    std::vector<int> flat;
    flat.reserve(ranges.size() * 2);
    for (const auto& r : ranges) {
        flat.push_back(r.start);
        flat.push_back(r.size);
    }
    int* dPtr = nullptr;
    cudaErrChk(cudaMalloc(&dPtr, bytes));
    cudaErrChk(cudaMemcpy(dPtr, flat.data(), bytes, cudaMemcpyHostToDevice));
    return dPtr;
}

} // namespace


macrocellSpectra2D::macrocellSpectra2D(const MacrocellPartition& part)
    : part_(part)
{
    if (part_.M <= 0) {
        throw std::runtime_error("macrocellSpectra2D: empty partition");
    }

    dRangeX_ = uploadRanges(part_.rangeX);
    dRangeY_ = uploadRanges(part_.rangeY);
    dRangeZ_ = uploadRanges(part_.rangeZ);

    numFloats_ = static_cast<size_t>(part_.M) * static_cast<size_t>(Nb);
    cudaErrChk(cudaMalloc(&dHist_, numFloats_ * sizeof(float)));
    cudaErrChk(cudaMallocHost((void**)&hHist_, numFloats_ * sizeof(float)));
}


macrocellSpectra2D::~macrocellSpectra2D() {
    if (dRangeX_) cudaFree(dRangeX_);
    if (dRangeY_) cudaFree(dRangeY_);
    if (dRangeZ_) cudaFree(dRangeZ_);
    if (dHist_)   cudaFree(dHist_);
    if (hHist_)   cudaFreeHost(hHist_);
}


void macrocellSpectra2D::reset(cudaStream_t stream) {
    cudaErrChk(cudaMemsetAsync(dHist_, 0, numFloats_ * sizeof(float), stream));
}


void macrocellSpectra2D::launch(particleArrayCUDA* pclsHostPtr,
                                cudaCommonType*    fieldForPclsCUDA,
                                const grid3DCUDA*  gridDevicePtr,
                                const int*         cellStartOffsetsCUDA,
                                const int*         cellCountsCUDA,
                                int                species,
                                cudaStream_t       stream)
{
    if (pclsHostPtr->getNOP() == 0) return;

    const cudaCommonType vmax = vmaxForSpecies(species);

    dim3 grid(part_.Mx, part_.My, part_.Mz);
    constexpr int BLOCK_SIZE = 128;
    // Shared memory: mini-histogram (Nb floats) + 8 corners x 3 B components.
    const size_t shmemBytes = Nb * sizeof(float)
                            + 24 * sizeof(cudaCommonType);

    macrocellSpectraKernel<<<grid, BLOCK_SIZE, shmemBytes, stream>>>(
        pclsHostPtr->getX(),
        pclsHostPtr->getY(),
        pclsHostPtr->getZ(),
        pclsHostPtr->getU(),
        pclsHostPtr->getV(),
        pclsHostPtr->getW(),
        pclsHostPtr->getQ(),
        cellStartOffsetsCUDA,
        cellCountsCUDA,
        dRangeX_, dRangeY_, dRangeZ_,
        part_.Mx, part_.My, part_.Mz,
        fieldForPclsCUDA,
        gridDevicePtr,
        dHist_,
        MACROCELL_BINS_VPAR, MACROCELL_BINS_VPERP,
        vmax,
        MACROCELL_BMIN);
}


// ======= Output =======

void macrocellSpectra2D::writePartitionMetadata(const std::string&       subdomainDir,
                                                int                      subdomainRank,
                                                const VCtopology3D* vct) const
{
    const std::string path = subdomainDir + "partition.json";
    std::ofstream f(path);
    if (!f.is_open())
        throw std::runtime_error("macrocellSpectra: cannot write " + path);

    auto writeAxisTiling = [&](const char* name,
                               const std::vector<MacrocellRange>& r) {
        f << "    \"" << name << "\": [";
        for (size_t i = 0; i < r.size(); ++i) {
            f << "[" << r[i].start << "," << r[i].size << "]";
            if (i + 1 < r.size()) f << ",";
        }
        f << "]";
    };

    f << "{\n"
      << "  \"subdomain_rank\": " << subdomainRank << ",\n"
      << "  \"cartesian_coords\": ["
      << vct->getCoordinates(0) << "," << vct->getCoordinates(1) << ","
      << vct->getCoordinates(2) << "],\n"
      << "  \"topology\": ["
      << vct->getXLEN() << "," << vct->getYLEN() << "," << vct->getZLEN() << "],\n"
      << "  \"subdomain_interior_cells\": ["
      << part_.Nx_int << "," << part_.Ny_int << "," << part_.Nz_int << "],\n"
      << "  \"subdomain_global_offset_cells\": ["
      << part_.globalOffsetX << "," << part_.globalOffsetY << ","
      << part_.globalOffsetZ << "],\n"
      << "  \"macrocell_size_request\": ["
      << part_.Cx << "," << part_.Cy << "," << part_.Cz << "],\n"
      << "  \"macrocells_per_axis\": ["
      << part_.Mx << "," << part_.My << "," << part_.Mz << "],\n"
      << "  \"axis_tiling\": {\n";
    writeAxisTiling("x", part_.rangeX); f << ",\n";
    writeAxisTiling("y", part_.rangeY); f << ",\n";
    writeAxisTiling("z", part_.rangeZ); f << "\n  },\n"
      << "  \"bins\": {\"vpar\": " << MACROCELL_BINS_VPAR
      << ", \"vperp\": " << MACROCELL_BINS_VPERP << "},\n"
      << "  \"ranges\": {\n"
      << "    \"e\": {\"vpar\": [" << MIN_VELOCITY_HIST_E << ","
                                   << MAX_VELOCITY_HIST_E << "], \"vperp\": [0,"
                                   << MAX_VELOCITY_HIST_E << "]},\n"
      << "    \"i\": {\"vpar\": [" << MIN_VELOCITY_HIST_I << ","
                                   << MAX_VELOCITY_HIST_I << "], \"vperp\": [0,"
                                   << MAX_VELOCITY_HIST_I << "]}\n"
      << "  },\n"
      << "  \"binary_layout\": \"row-major, vperp outer, vpar inner, float32\",\n"
      << "  \"endian\": \"little\"\n"
      << "}\n";
    f.close();
}


void macrocellSpectra2D::writeToFile(const std::string&        subdomainDir,
                                     int                       subdomainRank,
                                     const VCtopology3D*       vct,
                                     int                       species,
                                     int                       cycle,
                                     cudaStream_t              stream)
{
    // D->H copy of the full per-species histogram block.
    cudaErrChk(cudaMemcpyAsync(hHist_, dHist_,
                               numFloats_ * sizeof(float),
                               cudaMemcpyDeviceToHost, stream));
    cudaErrChk(cudaStreamSynchronize(stream));

    std::ostringstream cycDir;
    cycDir << subdomainDir << "species_" << species << "/cycle_"
           << std::setw(6) << std::setfill('0') << cycle << "/";
    const std::string cycDirStr = cycDir.str();

    // Create directories (idempotent).
    std::error_code ec;
    std::filesystem::create_directories(cycDirStr, ec);
    if (ec)
        throw std::runtime_error("macrocellSpectra: cannot create dir "
                                 + cycDirStr + " (" + ec.message() + ")");

    const cudaCommonType vmax = vmaxForSpecies(species);

    for (int mz = 0; mz < part_.Mz; ++mz)
    for (int my = 0; my < part_.My; ++my)
    for (int mx = 0; mx < part_.Mx; ++mx)
    {
        const int  m  = part_.linearId(mx, my, mz);
        const auto rx = part_.rangeX[mx];
        const auto ry = part_.rangeY[my];
        const auto rz = part_.rangeZ[mz];

        std::ostringstream base;
        base << cycDirStr << "mc_" << mx << "_" << my << "_" << mz;
        const std::string binPath  = base.str() + ".bin";
        const std::string jsonPath = base.str() + ".json";

        // Binary
        std::ofstream bin(binPath, std::ios::binary);
        if (!bin.is_open())
            throw std::runtime_error("macrocellSpectra: cannot open " + binPath);
        bin.write(reinterpret_cast<const char*>(hHist_ + size_t(m) * Nb),
                  Nb * sizeof(float));
        bin.close();

        // JSON sidecar
        std::ofstream js(jsonPath);
        if (!js.is_open())
            throw std::runtime_error("macrocellSpectra: cannot open " + jsonPath);
        js << "{\n"
           << "  \"cycle\": " << cycle << ",\n"
           << "  \"species\": " << species << ",\n"
           << "  \"mpi_rank\": " << subdomainRank << ",\n"
           << "  \"mpi_cartesian_rank\": " << (vct ? vct->getCartesian_rank() : -1) << ",\n"
           << "  \"mpi_coords\": ["
              << (vct ? vct->getCoordinates(0) : -1) << ","
              << (vct ? vct->getCoordinates(1) : -1) << ","
              << (vct ? vct->getCoordinates(2) : -1) << "],\n"
           << "  \"mpi_topology\": ["
              << (vct ? vct->getXLEN() : -1) << ","
              << (vct ? vct->getYLEN() : -1) << ","
              << (vct ? vct->getZLEN() : -1) << "],\n"
           << "  \"macrocell_index_in_subdomain\": ["
              << mx << "," << my << "," << mz << "],\n"
           << "  \"macrocell_size_cells\": ["
              << rx.size << "," << ry.size << "," << rz.size << "],\n"
           << "  \"macrocell_offset_in_subdomain_cells\": ["
              << rx.start << "," << ry.start << "," << rz.start << "],\n"
           << "  \"macrocell_offset_in_global_cells\": ["
              << (part_.globalOffsetX + rx.start) << ","
              << (part_.globalOffsetY + ry.start) << ","
              << (part_.globalOffsetZ + rz.start) << "],\n"
           << "  \"bins\": [" << MACROCELL_BINS_VPERP << ","
                              << MACROCELL_BINS_VPAR  << "],\n"
           << "  \"ranges\": {\"vpar\": [" << -vmax << "," << vmax
              << "], \"vperp\": [0," << vmax << "]},\n"
           << "  \"binary_file\": \"mc_" << mx << "_" << my << "_" << mz << ".bin\",\n"
           << "  \"endian\": \"little\"\n"
           << "}\n";
        js.close();
    }
}

} // namespace macrocellSpectra
