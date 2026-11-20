#include "macrocellSpectra.cuh"

#include "BuildInfo.h"
#include "VCtopology3D.h"
#include "cudaTypeDef.cuh"
#include "gridCUDA.cuh"
#include "outputPrepare.h"
#include "particleArrayCUDA.cuh"

#include <algorithm>
#include <cmath>
#include <cstdlib> // std::strtol, std::strtod
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <utility>

namespace macrocellSpectra {

/** Threads per block of the spectra kernel. */
inline constexpr int kSpectraThreads = 128;

// ======= MacrocellPartition =======

bool MacrocellPartition::build(int Nx_int_, int Ny_int_, int Nz_int_, int Cx_,
                               int Cy_, int Cz_, int globalOffX, int globalOffY,
                               int globalOffZ) {
  if (Cx_ <= 0 || Cy_ <= 0 || Cz_ <= 0)
    return false;
  if (Nx_int_ <= 0 || Ny_int_ <= 0 || Nz_int_ <= 0)
    return false;
  if (Cx_ > Nx_int_ || Cy_ > Ny_int_ || Cz_ > Nz_int_)
    return false;

  Cx = Cx_;
  Cy = Cy_;
  Cz = Cz_;
  Nx_int = Nx_int_;
  Ny_int = Ny_int_;
  Nz_int = Nz_int_;
  globalOffsetX = globalOffX;
  globalOffsetY = globalOffY;
  globalOffsetZ = globalOffZ;

  auto tile = [](int N, int C, std::vector<MacrocellRange>& out) {
    const int M = (N + C - 1) / C;
    out.clear();
    out.reserve(M);
    for (int m = 0; m < M; ++m) {
      const int start = m * C;
      const int size = (start + C <= N) ? C : (N - start);
      out.push_back({start, size});
    }
    return M;
  };

  Mx = tile(Nx_int, Cx, rangeX);
  My = tile(Ny_int, Cy, rangeY);
  Mz = tile(Nz_int, Cz, rangeZ);
  M = Mx * My * Mz;
  return true;
}

// ======= macrocellSpectra2D =======

namespace {

bool isRecordSeparator(char c) {
  return c == ' ' || c == '\n' || c == '\r' || c == '\t' || c == ',';
}

} // namespace

macrocellSpectra2D::macrocellSpectra2D(const MacrocellPartition& part,
                                       std::vector<cudaCommonType> vmax)
    : part_(part), vmax_(std::move(vmax)) {
  if (part_.M <= 0) {
    throw std::runtime_error("macrocellSpectra2D: empty partition");
  }

  numFloats_ = static_cast<size_t>(part_.M) * static_cast<size_t>(Nb);
  cudaErrChk(cudaMalloc(&dHist_, numFloats_ * sizeof(macrocellHistType)));
  hHist_ = static_cast<macrocellHistType*>(
      allocateHostPinnedMem(sizeof(macrocellHistType), numFloats_));
}

macrocellSpectra2D::~macrocellSpectra2D() {
  if (dHist_)
    cudaFree(dHist_);
  if (hHist_)
    cudaFreeHost(hHist_);
}

size_t macrocellSpectra2D::recordSizeBytes_() const {
  return numFloats_ * sizeof(macrocellHistType);
}

size_t macrocellSpectra2D::bytesPerMacrocell_() const {
  return static_cast<size_t>(Nb) * sizeof(macrocellHistType);
}

void macrocellSpectra2D::reset(cudaStream_t stream) {
  cudaErrChk(cudaMemsetAsync(dHist_, 0, numFloats_ * sizeof(macrocellHistType),
                             stream));
}

void macrocellSpectra2D::launch(particleArrayCUDA* pclsHostPtr,
                                cudaFieldType* fieldForPclsCUDA,
                                const grid3DCUDA* gridDevicePtr,
                                const int* cellStartOffsetsCUDA,
                                const int* cellCountsCUDA, int species,
                                cudaStream_t stream) {
  if (pclsHostPtr->getNOP() == 0)
    return;

  dim3 grid(part_.Mx, part_.My, part_.Mz);
  // Dynamic shared memory is only the per-block mini-histogram; B corners use
  // a small statically aligned shared array in the kernel.
  const size_t shmemBytes = Nb * sizeof(macrocellHistType);

  macrocellSpectraKernel<<<grid, kSpectraThreads, shmemBytes, stream>>>(
      pclsHostPtr->getX(), pclsHostPtr->getY(), pclsHostPtr->getZ(),
      pclsHostPtr->getU(), pclsHostPtr->getV(), pclsHostPtr->getW(),
      pclsHostPtr->getQ(), cellStartOffsetsCUDA, cellCountsCUDA, part_.Nx_int,
      part_.Ny_int, part_.Nz_int, part_.Cx, part_.Cy, part_.Cz,
      fieldForPclsCUDA, gridDevicePtr, dHist_, vmax_[species]);
  cudaErrChk(cudaGetLastError());
}

// ======= Output =======

// -----------------------------------------------------------------------
// parseRecordsFromJson_
// -----------------------------------------------------------------------
std::vector<int>
macrocellSpectra2D::parseRecordsFromJson_(const std::string& content) {
  std::vector<int> result;

  // Locate the "records" key and the following array.
  const std::string key = "\"records\"";
  auto pos = content.find(key);
  if (pos == std::string::npos)
    return result;
  pos += key.size();

  pos = content.find(':', pos);
  if (pos == std::string::npos)
    return result;
  pos = content.find('[', pos);
  if (pos == std::string::npos)
    return result;
  ++pos;

  // Parse comma-separated integers until ']'.
  while (pos < content.size()) {
    // Skip whitespace and commas.
    while (pos < content.size() && isRecordSeparator(content[pos]))
      ++pos;
    if (pos >= content.size() || content[pos] == ']')
      break;

    char* endPtr = nullptr;
    long val = std::strtol(content.c_str() + pos, &endPtr, 10);
    if (endPtr == content.c_str() + pos)
      break; // no digits found
    result.push_back(static_cast<int>(val));
    pos = static_cast<size_t>(endPtr - content.c_str());
  }

  return result;
}

// -----------------------------------------------------------------------
// writeJson_  — full rewrite via temp-file + atomic rename
// -----------------------------------------------------------------------
void macrocellSpectra2D::writeJson_(const SpeciesFileContext& ctx) const {
  const std::string tmpPath = ctx.jsonPath + ".tmp";
  std::ofstream f(tmpPath);
  if (!f.is_open())
    throw std::runtime_error("macrocellSpectra: cannot write " + tmpPath);

  // Helper: write one axis tiling as  "name": [[s0,n0],[s1,n1],...]
  auto writeAxisTiling = [&](const char* name,
                             const std::vector<MacrocellRange>& r) {
    f << "    \"" << name << "\": [";
    for (size_t i = 0; i < r.size(); ++i) {
      f << "[" << r[i].start << "," << r[i].size << "]";
      if (i + 1 < r.size())
        f << ",";
    }
    f << "]";
  };

  f << std::setprecision(17); // full double precision for velocity ranges

  f << "{\n"
    << "  \"subdomain_rank\": " << ctx.rank << ",\n"
    << "  \"cartesian_rank\": " << ctx.cartesianRank << ",\n"
    << "  \"cartesian_coords\": [" << ctx.coords[0] << "," << ctx.coords[1]
    << "," << ctx.coords[2] << "],\n"
    << "  \"topology\": [" << ctx.topology[0] << "," << ctx.topology[1] << ","
    << ctx.topology[2] << "],\n"
    << "  \"species\": " << ctx.species << ",\n"
    << "  \"ipic3d_version\": \"" << iPic3D::build::version() << "\",\n"
    << "  \"ipic3d_build\": \"" << iPic3D::build::signature() << "\",\n"
    << "  \"vmax\": " << vmax_[ctx.species] << ",\n"
    << "  \"subdomain_interior_cells\": [" << part_.Nx_int << ","
    << part_.Ny_int << "," << part_.Nz_int << "],\n"
    << "  \"subdomain_global_offset_cells\": [" << part_.globalOffsetX << ","
    << part_.globalOffsetY << "," << part_.globalOffsetZ << "],\n"
    << "  \"macrocell_size_request\": [" << part_.Cx << "," << part_.Cy << ","
    << part_.Cz << "],\n"
    << "  \"macrocells_per_axis\": [" << part_.Mx << "," << part_.My << ","
    << part_.Mz << "],\n"
    << "  \"total_macrocells\": " << part_.M << ",\n"
    << "  \"axis_tiling\": {\n";
  writeAxisTiling("x", part_.rangeX);
  f << ",\n";
  writeAxisTiling("y", part_.rangeY);
  f << ",\n";
  writeAxisTiling("z", part_.rangeZ);
  f << "\n  },\n"
    << "  \"bins\": {\"vpar\": " << DAConfig::MACROCELL_BINS_VPAR
    << ", \"vperp\": " << DAConfig::MACROCELL_BINS_VPERP << "},\n"
    << "  \"ranges\": {\n"
    << "    \"vpar\":  [" << -vmax_[ctx.species] << "," << vmax_[ctx.species]
    << "],\n"
    << "    \"vperp\": [0," << vmax_[ctx.species] << "]\n"
    << "  },\n"
    << "  \"dtype\": \"float32\",\n"
    << "  \"endian\": \"little\",\n"
    << "  \"bytes_per_macrocell\": " << bytesPerMacrocell_() << ",\n"
    << "  \"record_size_bytes\": " << recordSizeBytes_() << ",\n"
    << "  \"binary_layout\": \"per record: M macrocells contiguous; "
       "within macrocell: row-major [vperp][vpar] (vperp outer, vpar inner); "
       "macrocell linear id m = (mz*My + my)*Mx + mx\",\n"
    << "  \"byte_offset_formula\": "
       "\"record_index * record_size_bytes + m * bytes_per_macrocell\",\n"
    << "  \"records\": [";
  for (size_t i = 0; i < ctx.records.size(); ++i) {
    f << ctx.records[i];
    if (i + 1 < ctx.records.size())
      f << ",";
  }
  f << "]\n}\n";

  f.close();
  if (!f) // badbit can be set after close if the flush failed
    throw std::runtime_error("macrocellSpectra: write failed for " + tmpPath);

  // Atomic rename: on POSIX, rename() over an existing file is atomic.
  std::error_code ec;
  std::filesystem::rename(tmpPath, ctx.jsonPath, ec);
  if (ec)
    throw std::runtime_error("macrocellSpectra: rename failed " + tmpPath +
                             " -> " + ctx.jsonPath + " (" + ec.message() + ")");
}

macrocellSpectra2D::SpeciesFileContext
macrocellSpectra2D::makeSpeciesContext_(const std::string& subdomainDir,
                                        int species, int rank,
                                        const VCtopology3D* vct) const {
  SpeciesFileContext ctx;
  ctx.species = species;
  ctx.rank = rank;
  ctx.binPath = subdomainDir + "species_" + std::to_string(species) + ".bin";
  ctx.jsonPath = subdomainDir + "species_" + std::to_string(species) + ".json";

  if (vct) {
    ctx.cartesianRank = vct->getCartesian_rank();
    ctx.coords[0] = vct->getCoordinates(0);
    ctx.coords[1] = vct->getCoordinates(1);
    ctx.coords[2] = vct->getCoordinates(2);
    ctx.topology[0] = vct->getXLEN();
    ctx.topology[1] = vct->getYLEN();
    ctx.topology[2] = vct->getZLEN();
  }

  return ctx;
}

void macrocellSpectra2D::createEmptyBinary_(const std::string& path) const {
  std::ofstream bin(path, std::ios::binary | std::ios::trunc);
  if (!bin.is_open()) {
    throw std::runtime_error("macrocellSpectra: cannot create " + path);
  }
}

bool macrocellSpectra2D::reconcileRestart_(SpeciesFileContext& ctx,
                                           int firstCycle) const {
  if (std::filesystem::exists(ctx.jsonPath)) {
    std::ifstream f(ctx.jsonPath);
    if (!f.is_open())
      throw std::runtime_error("macrocellSpectra: cannot read " + ctx.jsonPath);
    const std::string content((std::istreambuf_iterator<char>(f)),
                              std::istreambuf_iterator<char>());
    f.close();
    // the value of a number of the JSON, NaN if absent
    auto number = [&](const std::string& key) {
      const size_t pos = content.find("\"" + key + "\":");
      return pos == std::string::npos
                 ? NAN
                 : std::strtod(content.c_str() + pos + key.size() + 3, nullptr);
    };
    // the tiling as writeJson_ writes it
    auto has = [&](const std::string& key, int a, int b, int c) {
      return content.find("\"" + key + "\": [" + std::to_string(a) + "," +
                          std::to_string(b) + "," + std::to_string(c) + "]") !=
             std::string::npos;
    };
    const std::string bins =
        "\"bins\": {\"vpar\": " +
        std::to_string(DAConfig::MACROCELL_BINS_VPAR) +
        ", \"vperp\": " + std::to_string(DAConfig::MACROCELL_BINS_VPERP) + "}";
    if (number("record_size_bytes") != (double)recordSizeBytes_() ||
        number("vmax") != (double)vmax_[ctx.species] ||
        content.find(bins) == std::string::npos ||
        !has("subdomain_interior_cells", part_.Nx_int, part_.Ny_int,
             part_.Nz_int) ||
        !has("macrocell_size_request", part_.Cx, part_.Cy, part_.Cz)) {
      // records of another layout cannot continue: kept aside, start again
      keepAside({ctx.jsonPath, ctx.binPath}, firstCycle);
      createEmptyBinary_(ctx.binPath);
      return true;
    }
    ctx.records = parseRecordsFromJson_(content);
    // the restart writes again the cycles from firstCycle on
    const auto again =
        std::find_if(ctx.records.begin(), ctx.records.end(),
                     [&](int cycle) { return cycle >= firstCycle; });
    ctx.records.erase(again, ctx.records.end());
  } else {
    std::cerr << "[macrocellSpectra] rank=" << ctx.rank
              << " species=" << ctx.species
              << ": JSON not found on restart; starting fresh.\n";
  }

  if (!std::filesystem::exists(ctx.binPath)) {
    std::cerr << "[macrocellSpectra] rank=" << ctx.rank
              << " species=" << ctx.species
              << ": binary not found on restart; creating new file.\n";
    ctx.records.clear();
    createEmptyBinary_(ctx.binPath);
    return false;
  }

  // JSON is authoritative: keep the binary records it lists, which drops
  // the later cycles and incomplete trailing writes
  const size_t actualBytes = std::filesystem::file_size(ctx.binPath);
  const size_t completeInBin = actualBytes / recordSizeBytes_();
  if (completeInBin < ctx.records.size()) {
    std::cerr << "[macrocellSpectra] rank=" << ctx.rank
              << " species=" << ctx.species << ": the binary holds "
              << completeInBin << " records of the " << ctx.records.size()
              << " listed; keeping those.\n";
    ctx.records.resize(completeInBin);
  }
  const size_t keptBytes = ctx.records.size() * recordSizeBytes_();
  if (actualBytes == keptBytes)
    return false;
  std::error_code ec;
  std::filesystem::resize_file(ctx.binPath, keptBytes, ec);
  if (ec) {
    throw std::runtime_error("macrocellSpectra: cannot resize " + ctx.binPath +
                             " (" + ec.message() + ")");
  }
  return false;
}

void macrocellSpectra2D::appendBinaryRecord_(
    const SpeciesFileContext& ctx) const {
  // std::ios::app positions the write pointer at the end before every write,
  // which is correct for sequential appends.
  std::ofstream bin(ctx.binPath, std::ios::binary | std::ios::app);
  if (!bin.is_open()) {
    throw std::runtime_error("macrocellSpectra: cannot open " + ctx.binPath);
  }

  bin.write(reinterpret_cast<const char*>(hHist_),
            static_cast<std::streamsize>(recordSizeBytes_()));
  bin.close();

  if (!bin) {
    throw std::runtime_error("macrocellSpectra: write failed for " +
                             ctx.binPath);
  }
}

// -----------------------------------------------------------------------
// initSpeciesFile
// -----------------------------------------------------------------------
bool macrocellSpectra2D::initSpeciesFile(const std::string& subdomainDir,
                                         int species, int rank, bool isRestart,
                                         int firstCycle,
                                         const VCtopology3D* vct) {
  if (species < 0) {
    throw std::runtime_error("macrocellSpectra: invalid species " +
                             std::to_string(species));
  }

  SpeciesFileContext ctx =
      makeSpeciesContext_(subdomainDir, species, rank, vct);
  bool keptAside = false;
  if (isRestart) {
    keptAside = reconcileRestart_(ctx, firstCycle);
  } else {
    createEmptyBinary_(ctx.binPath);
  }

  if (species >= static_cast<int>(speciesCtx_.size())) {
    speciesCtx_.resize(static_cast<size_t>(species) + 1);
  }
  ctx.initialized = true;
  speciesCtx_[static_cast<size_t>(species)] = std::move(ctx);
  writeJson_(speciesCtx_[static_cast<size_t>(species)]);
  return keptAside;
}

// -----------------------------------------------------------------------
// writeToFile  — D->H copy + binary append + JSON rewrite
// -----------------------------------------------------------------------
void macrocellSpectra2D::writeToFile(int species, int cycle,
                                     cudaStream_t stream) {
  if (species < 0 || species >= static_cast<int>(speciesCtx_.size()) ||
      !speciesCtx_[static_cast<size_t>(species)].initialized) {
    throw std::runtime_error(
        "macrocellSpectra: initSpeciesFile not called for species " +
        std::to_string(species));
  }
  SpeciesFileContext& ctx = speciesCtx_[static_cast<size_t>(species)];

  // ---- D->H copy of the full per-species histogram block ----
  cudaErrChk(cudaMemcpyAsync(hHist_, dHist_, recordSizeBytes_(),
                             cudaMemcpyDeviceToHost, stream));
  cudaErrChk(cudaStreamSynchronize(stream));

  appendBinaryRecord_(ctx);

  // ---- Update in-memory records and rewrite JSON atomically ----
  // JSON is updated only after the binary write has completed, so
  // records.size() * record_size_bytes == actual binary file size.
  ctx.records.push_back(cycle);
  writeJson_(ctx);
}

} // namespace macrocellSpectra
