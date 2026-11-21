#include "macrocellPartition.cuh"

#include <algorithm>
#include <ostream>
#include <sstream>

namespace macrocell {

bool MacrocellPartition::build(int Nx_int_, int Ny_int_, int Nz_int_, int Cx_,
                               int Cy_, int Cz_, int globalOffX, int globalOffY,
                               int globalOffZ) {
  if (Cx_ <= 0 || Cy_ <= 0 || Cz_ <= 0)
    return false;
  if (Nx_int_ <= 0 || Ny_int_ <= 0 || Nz_int_ <= 0)
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

  Mx = (Nx_int + Cx - 1) / Cx;
  My = (Ny_int + Cy - 1) / Cy;
  Mz = (Nz_int + Cz - 1) / Cz;
  M = Mx * My * Mz;
  return true;
}

int slicesFor(int macrocells, int blocksPerSlice, int largestCells) {
  int device = 0;
  cudaDeviceProp prop;
  cudaErrChk(cudaGetDevice(&device));
  cudaErrChk(cudaGetDeviceProperties(&prop, device));
  const long long wanted =
      (long long)kSliceBlocksPerSM * prop.multiProcessorCount;
  const long long blocks = (long long)std::max(macrocells, 1) * blocksPerSlice;
  return (int)std::max(
      1LL, std::min((wanted + blocks - 1) / blocks, (long long)largestCells));
}

namespace {

/** The JSON members of the tiling, from "subdomain_interior_cells" to the
 *  "axis_tiling" object, one per line, each ending with ",\n". */
void writeMacrocellTiling(std::ostream& out, const MacrocellPartition& part) {
  // one axis as  "name": [[start,size],...] (macrocellAxisRange)
  auto axis = [&](const char* name, int macrocells, int size, int interior) {
    out << "    \"" << name << "\": [";
    for (int i = 0; i < macrocells; ++i) {
      int start, length;
      macrocellAxisRange(i, size, interior, start, length);
      out << (i ? "," : "") << "[" << start << "," << length << "]";
    }
    out << "]";
  };
  out << "  \"subdomain_interior_cells\": [" << part.Nx_int << ","
      << part.Ny_int << "," << part.Nz_int << "],\n"
      << "  \"subdomain_global_offset_cells\": [" << part.globalOffsetX << ","
      << part.globalOffsetY << "," << part.globalOffsetZ << "],\n"
      << "  \"macrocell_size_request\": [" << part.Cx << "," << part.Cy << ","
      << part.Cz << "],\n"
      << "  \"macrocells_per_axis\": [" << part.Mx << "," << part.My << ","
      << part.Mz << "],\n"
      << "  \"total_macrocells\": " << part.M << ",\n"
      << "  \"axis_tiling\": {\n";
  axis("x", part.Mx, part.Cx, part.Nx_int);
  out << ",\n";
  axis("y", part.My, part.Cy, part.Ny_int);
  out << ",\n";
  axis("z", part.Mz, part.Cz, part.Nz_int);
  out << "\n  },\n";
}

} // namespace

std::string regionsDescription(const MacrocellPartition* part) {
  std::ostringstream out;
  out << "  \"regions\": \"" << (part ? "macrocells" : "rank") << "\",\n";
  if (part)
    writeMacrocellTiling(out, *part);
  return out.str();
}

} // namespace macrocell
