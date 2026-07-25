#include "RestartSlotManager.h"

#include "debug.h"

#include <filesystem>
#include <fstream>
#include <iomanip>
#include <regex>
#include <sstream>
#include <string>

namespace fs = std::filesystem;

namespace {

std::string readTextFile(const fs::path& path) {
  std::ifstream in(path);
  if (!in)
    return std::string();
  std::ostringstream buffer;
  buffer << in.rdbuf();
  return buffer.str();
}

bool extractObject(const std::string& text, const std::string& key,
                   std::string& object) {
  const std::string quotedKey = "\"" + key + "\"";
  const std::size_t keyPos = text.find(quotedKey);
  if (keyPos == std::string::npos)
    return false;

  const std::size_t bracePos = text.find('{', keyPos + quotedKey.size());
  if (bracePos == std::string::npos)
    return false;

  int depth = 0;
  for (std::size_t i = bracePos; i < text.size(); ++i) {
    if (text[i] == '{')
      ++depth;
    if (text[i] == '}') {
      --depth;
      if (depth == 0) {
        object = text.substr(bracePos, i - bracePos + 1);
        return true;
      }
    }
  }
  return false;
}

bool extractString(const std::string& text, const std::string& key,
                   std::string& value) {
  const std::regex expr("\"" + key + "\"\\s*:\\s*\"([^\"]*)\"");
  std::smatch match;
  if (!std::regex_search(text, match, expr))
    return false;
  value = match[1].str();
  return true;
}

bool extractLongLong(const std::string& text, const std::string& key,
                     long long& value) {
  const std::regex expr("\"" + key + "\"\\s*:\\s*(-?[0-9]+)");
  std::smatch match;
  if (!std::regex_search(text, match, expr))
    return false;
  value = std::stoll(match[1].str());
  return true;
}

bool extractDouble(const std::string& text, const std::string& key,
                   double& value) {
  const std::regex expr(
      "\"" + key +
      "\"\\s*:\\s*(-?(?:[0-9]+(?:\\.[0-9]*)?|\\.[0-9]+)(?:[eE][-+]?[0-9]+)?)");
  std::smatch match;
  if (!std::regex_search(text, match, expr))
    return false;
  value = std::stod(match[1].str());
  return true;
}

bool extractBool(const std::string& text, const std::string& key, bool& value) {
  const std::regex expr("\"" + key + "\"\\s*:\\s*(true|false|0|1)");
  std::smatch match;
  if (!std::regex_search(text, match, expr))
    return false;

  const std::string token = match[1].str();
  value = (token == "true" || token == "1");
  return true;
}

void writeBool(std::ostream& out, bool value) {
  out << (value ? "true" : "false");
}

bool parseMeshMetadata(const std::string& block, RestartMeshMetadata& mesh) {
  std::string meshBlock;
  if (!extractObject(block, "mesh", meshBlock))
    return false;

  long long tmp = 0;
  RestartMeshMetadata parsed;

  if (!extractLongLong(meshBlock, "xlen", tmp))
    return false;
  parsed.xlen = static_cast<int>(tmp);
  if (!extractLongLong(meshBlock, "ylen", tmp))
    return false;
  parsed.ylen = static_cast<int>(tmp);
  if (!extractLongLong(meshBlock, "zlen", tmp))
    return false;
  parsed.zlen = static_cast<int>(tmp);
  if (!extractLongLong(meshBlock, "nranks", tmp))
    return false;
  parsed.nranks = static_cast<int>(tmp);

  if (!extractLongLong(meshBlock, "nxc", tmp))
    return false;
  parsed.nxc = static_cast<int>(tmp);
  if (!extractLongLong(meshBlock, "nyc", tmp))
    return false;
  parsed.nyc = static_cast<int>(tmp);
  if (!extractLongLong(meshBlock, "nzc", tmp))
    return false;
  parsed.nzc = static_cast<int>(tmp);
  if (!extractLongLong(meshBlock, "ns", tmp))
    return false;
  parsed.ns = static_cast<int>(tmp);

  extractDouble(meshBlock, "lx", parsed.lx);
  extractDouble(meshBlock, "ly", parsed.ly);
  extractDouble(meshBlock, "lz", parsed.lz);
  extractDouble(meshBlock, "dx", parsed.dx);
  extractDouble(meshBlock, "dy", parsed.dy);
  extractDouble(meshBlock, "dz", parsed.dz);

  extractBool(meshBlock, "periodicX", parsed.periodicX);
  extractBool(meshBlock, "periodicY", parsed.periodicY);
  extractBool(meshBlock, "periodicZ", parsed.periodicZ);
  extractBool(meshBlock, "periodicParticleX", parsed.periodicParticleX);
  extractBool(meshBlock, "periodicParticleY", parsed.periodicParticleY);
  extractBool(meshBlock, "periodicParticleZ", parsed.periodicParticleZ);
  extractBool(meshBlock, "fieldsStoreActiveNodesOnly",
              parsed.fieldsStoreActiveNodesOnly);
  extractBool(meshBlock, "particlesStoreActiveCellsOnly",
              parsed.particlesStoreActiveCellsOnly);
  extractBool(meshBlock, "particlesSortedByActiveCell",
              parsed.particlesSortedByActiveCell);

  parsed.valid = parsed.xlen > 0 && parsed.ylen > 0 && parsed.zlen > 0 &&
                 parsed.nranks == parsed.xlen * parsed.ylen * parsed.zlen &&
                 parsed.nxc > 0 && parsed.nyc > 0 && parsed.nzc > 0 &&
                 parsed.ns >= 0;
  if (!parsed.valid)
    return false;

  mesh = parsed;
  return true;
}

bool parseCheckpointBlock(const std::string& block, const std::string& rootDir,
                          const std::string& backend,
                          RestartCheckpoint& checkpoint) {
  std::string slot;
  long long generation = 0;
  long long cycle = -1;
  long long nranks = 0;
  std::string savedBackend;

  if (!extractString(block, "slot", slot))
    return false;
  if (!extractLongLong(block, "cycle", cycle))
    return false;
  extractLongLong(block, "generation", generation);
  extractLongLong(block, "nranks", nranks);
  extractString(block, "backend", savedBackend);

  if (slot != "A" && slot != "B")
    return false;
  if (!savedBackend.empty() && savedBackend != backend)
    return false;

  checkpoint.found = true;
  checkpoint.rootDir = rootDir;
  checkpoint.dataDir = RestartSlotManager::slotDir(rootDir, slot);
  checkpoint.slot = slot;
  checkpoint.backend = savedBackend.empty() ? backend : savedBackend;
  checkpoint.cycle = static_cast<int>(cycle);
  checkpoint.nranks = static_cast<int>(nranks);
  checkpoint.generation = generation;
  if (!parseMeshMetadata(block, checkpoint.mesh)) {
    eprintf("ERROR: restart checkpoint restart_%s is missing or has "
            "invalid mesh metadata",
            slot.c_str());
  }
  return true;
}

bool checkpointMatchesManifest(const RestartCheckpoint& checkpoint) {
  const fs::path manifestPath = fs::path(checkpoint.dataDir) / "manifest.json";
  const std::string manifest = readTextFile(manifestPath);
  if (manifest.empty()) {
    eprintf("ERROR: restart checkpoint restart_%s is missing %s",
            checkpoint.slot.c_str(), manifestPath.string().c_str());
  }

  RestartCheckpoint manifestCheckpoint;
  if (!parseCheckpointBlock(manifest, checkpoint.rootDir, checkpoint.backend,
                            manifestCheckpoint)) {
    return false;
  }

  if (manifestCheckpoint.slot != checkpoint.slot)
    return false;
  if (manifestCheckpoint.cycle != checkpoint.cycle)
    return false;
  if (manifestCheckpoint.generation != checkpoint.generation)
    return false;
  if (manifestCheckpoint.nranks != checkpoint.nranks)
    return false;
  return true;
}

bool checkpointHasAllRankFiles(const RestartCheckpoint& checkpoint) {
  for (int rank = 0; rank < checkpoint.nranks; ++rank) {
    const fs::path rankFile =
        fs::path(checkpoint.dataDir) /
        RestartSlotManager::rankFileName(checkpoint.backend, rank);
    if (!fs::exists(rankFile))
      return false;
  }
  return true;
}

bool validateCheckpoint(const RestartCheckpoint& checkpoint,
                        const std::string& backend) {
  if (!checkpoint.found)
    return false;
  if (checkpoint.slot != "A" && checkpoint.slot != "B")
    return false;
  if (checkpoint.backend != backend)
    return false;
  if (checkpoint.cycle < 0)
    return false;
  if (checkpoint.nranks <= 0)
    return false;
  if (!checkpoint.mesh.valid)
    return false;
  if (!fs::is_directory(checkpoint.dataDir))
    return false;
  if (!checkpointMatchesManifest(checkpoint))
    return false;
  return checkpointHasAllRankFiles(checkpoint);
}

void writeTextAtomically(const fs::path& path, const std::string& text) {
  fs::create_directories(path.parent_path());

  const fs::path tmpPath = path.string() + ".tmp";
  {
    std::ofstream out(tmpPath, std::ios::trunc);
    if (!out) {
      eprintf("ERROR: could not write restart metadata file: %s",
              tmpPath.c_str());
    }
    out << text;
    out.close();
    if (!out) {
      eprintf("ERROR: failed while writing restart metadata file: %s",
              tmpPath.c_str());
    }
  }

  std::error_code ec;
  fs::rename(tmpPath, path, ec);
  if (ec) {
    eprintf("ERROR: could not publish restart metadata file: %s", path.c_str());
  }
}

void writeMeshObject(std::ostream& out, const RestartMeshMetadata& mesh,
                     const std::string& indent) {
  out << indent << "{\n"
      << indent << "  \"xlen\": " << mesh.xlen << ",\n"
      << indent << "  \"ylen\": " << mesh.ylen << ",\n"
      << indent << "  \"zlen\": " << mesh.zlen << ",\n"
      << indent << "  \"nranks\": " << mesh.nranks << ",\n"
      << indent << "  \"nxc\": " << mesh.nxc << ",\n"
      << indent << "  \"nyc\": " << mesh.nyc << ",\n"
      << indent << "  \"nzc\": " << mesh.nzc << ",\n"
      << indent << "  \"ns\": " << mesh.ns << ",\n"
      << std::setprecision(17) << indent << "  \"lx\": " << mesh.lx << ",\n"
      << indent << "  \"ly\": " << mesh.ly << ",\n"
      << indent << "  \"lz\": " << mesh.lz << ",\n"
      << indent << "  \"dx\": " << mesh.dx << ",\n"
      << indent << "  \"dy\": " << mesh.dy << ",\n"
      << indent << "  \"dz\": " << mesh.dz << ",\n"
      << indent << "  \"periodicX\": ";
  writeBool(out, mesh.periodicX);
  out << ",\n" << indent << "  \"periodicY\": ";
  writeBool(out, mesh.periodicY);
  out << ",\n" << indent << "  \"periodicZ\": ";
  writeBool(out, mesh.periodicZ);
  out << ",\n" << indent << "  \"periodicParticleX\": ";
  writeBool(out, mesh.periodicParticleX);
  out << ",\n" << indent << "  \"periodicParticleY\": ";
  writeBool(out, mesh.periodicParticleY);
  out << ",\n" << indent << "  \"periodicParticleZ\": ";
  writeBool(out, mesh.periodicParticleZ);
  out << ",\n" << indent << "  \"fieldsStoreActiveNodesOnly\": ";
  writeBool(out, mesh.fieldsStoreActiveNodesOnly);
  out << ",\n" << indent << "  \"particlesStoreActiveCellsOnly\": ";
  writeBool(out, mesh.particlesStoreActiveCellsOnly);
  out << ",\n" << indent << "  \"particlesSortedByActiveCell\": ";
  writeBool(out, mesh.particlesSortedByActiveCell);
  out << "\n" << indent << "}";
}

std::string checkpointJson(const RestartWriteTarget& target) {
  std::ostringstream out;
  out << "{\n"
      << "  \"slot\": \"" << target.slot << "\",\n"
      << "  \"generation\": " << target.generation << ",\n"
      << "  \"cycle\": " << target.cycle << ",\n"
      << "  \"backend\": \"" << target.backend << "\",\n"
      << "  \"nranks\": " << target.nranks << ",\n"
      << "  \"mesh\": ";
  writeMeshObject(out, target.mesh, "  ");
  out << "\n"
      << "}\n";
  return out.str();
}

void writeCheckpointObject(std::ostream& out,
                           const RestartCheckpoint& checkpoint,
                           const std::string& indent) {
  out << indent << "{\n"
      << indent << "  \"slot\": \"" << checkpoint.slot << "\",\n"
      << indent << "  \"generation\": " << checkpoint.generation << ",\n"
      << indent << "  \"cycle\": " << checkpoint.cycle << ",\n"
      << indent << "  \"backend\": \"" << checkpoint.backend << "\",\n"
      << indent << "  \"nranks\": " << checkpoint.nranks << ",\n"
      << indent << "  \"mesh\": ";
  writeMeshObject(out, checkpoint.mesh, indent + "  ");
  out << "\n";
  out << indent << "}";
}

std::string latestJson(const RestartWriteTarget& target) {
  RestartCheckpoint latest;
  latest.found = true;
  latest.rootDir = target.rootDir;
  latest.dataDir = target.dataDir;
  latest.slot = target.slot;
  latest.backend = target.backend;
  latest.cycle = target.cycle;
  latest.nranks = target.nranks;
  latest.generation = target.generation;
  latest.mesh = target.mesh;

  std::ostringstream out;
  out << "{\n"
      << "  \"latest\": ";
  writeCheckpointObject(out, latest, "  ");
  out << ",\n"
      << "  \"previous\": ";
  if (target.previous.found) {
    writeCheckpointObject(out, target.previous, "  ");
    out << "\n";
  } else {
    out << "null\n";
  }
  out << "}\n";
  return out.str();
}

} // namespace

void RestartSlotManager::init(const std::string& rootDir,
                              const std::string& backend, int rank, int nranks,
                              const RestartMeshMetadata& mesh) {
  rootDir_ = rootDir;
  backend_ = backend;
  rank_ = rank;
  nranks_ = nranks;
  mesh_ = mesh;
  latest_ = resolveLatest(rootDir_, backend_);
  initialized_ = true;
}

RestartWriteTarget RestartSlotManager::beginWrite(int cycle) {
  if (!initialized_) {
    eprintf("ERROR: RestartSlotManager used before init()");
  }

  RestartWriteTarget target;
  target.rootDir = rootDir_;
  target.backend = backend_;
  target.cycle = cycle;
  target.rank = rank_;
  target.nranks = nranks_;
  target.previous = latest_;
  target.mesh = mesh_;

  if (!latest_.found) {
    target.slot = "A";
    target.generation = 1;
  } else {
    target.slot = latest_.slot == "A" ? "B" : "A";
    target.generation = latest_.generation + 1;
  }

  target.dataDir = slotDir(rootDir_, target.slot);
  target.rankFilePath =
      (fs::path(target.dataDir) / rankFileName(backend_, rank_)).string();

  fs::create_directories(target.dataDir);
  std::error_code ec;
  fs::remove_all(target.rankFilePath, ec);
  if (ec) {
    eprintf("ERROR: could not remove old restart rank file: %s",
            target.rankFilePath.c_str());
  }

  return target;
}

void RestartSlotManager::publishIfRoot(const RestartWriteTarget& target) const {
  if (rank_ != 0)
    return;

  writeTextAtomically(fs::path(target.dataDir) / "manifest.json",
                      checkpointJson(target));
  writeTextAtomically(fs::path(target.rootDir) / "latest_restart.json",
                      latestJson(target));
}

void RestartSlotManager::completeLocal(const RestartWriteTarget& target) {
  latest_.found = true;
  latest_.rootDir = target.rootDir;
  latest_.dataDir = target.dataDir;
  latest_.slot = target.slot;
  latest_.backend = target.backend;
  latest_.cycle = target.cycle;
  latest_.nranks = target.nranks;
  latest_.generation = target.generation;
  latest_.mesh = target.mesh;
}

RestartCheckpoint
RestartSlotManager::resolveLatest(const std::string& rootDir,
                                  const std::string& backend) {
  const fs::path latestPath = fs::path(rootDir) / "latest_restart.json";
  const std::string latestText = readTextFile(latestPath);
  if (latestText.empty())
    return RestartCheckpoint();

  std::string latestBlock;
  RestartCheckpoint latest;
  if (extractObject(latestText, "latest", latestBlock) &&
      parseCheckpointBlock(latestBlock, rootDir, backend, latest) &&
      validateCheckpoint(latest, backend)) {
    return latest;
  }

  std::string previousBlock;
  RestartCheckpoint previous;
  if (extractObject(latestText, "previous", previousBlock) &&
      parseCheckpointBlock(previousBlock, rootDir, backend, previous) &&
      validateCheckpoint(previous, backend)) {
    return previous;
  }

  return RestartCheckpoint();
}

RestartCheckpoint RestartSlotManager::readManifest(const std::string& dataDir,
                                                   const std::string& backend) {
  const fs::path manifestPath = fs::path(dataDir) / "manifest.json";
  const std::string manifest = readTextFile(manifestPath);
  if (manifest.empty()) {
    eprintf("ERROR: restart directory %s does not contain manifest.json. "
            "Legacy flat restarts are unsupported.",
            dataDir.c_str());
  }

  fs::path rootPath = fs::path(dataDir).parent_path();
  RestartCheckpoint checkpoint;
  if (!parseCheckpointBlock(manifest, rootPath.string(), backend, checkpoint)) {
    eprintf("ERROR: malformed restart manifest in %s",
            manifestPath.string().c_str());
  }
  checkpoint.dataDir = dataDir;
  return checkpoint;
}

std::string RestartSlotManager::backendName() {
#ifdef USE_ADIOS2
  return "adios2";
#elif !defined(NO_HDF5)
  return "hdf5";
#else
  return "";
#endif
}

std::string RestartSlotManager::slotDir(const std::string& rootDir,
                                        const std::string& slot) {
  return (fs::path(rootDir) / ("restart_" + slot)).string();
}

std::string RestartSlotManager::rankFileName(const std::string& backend,
                                             int rank) {
  if (backend == "adios2") {
    return "restart_" + std::to_string(rank) + ".bp";
  } else if (backend == "hdf5") {
    return "restart" + std::to_string(rank) + ".hdf";
  } else {
    eprintf("ERROR: unknown restart backend: %s", backend.c_str());
    return "";
  }
}
