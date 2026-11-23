#include "outputPrepare.h"

#include <algorithm>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <regex>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace fs = std::filesystem;

namespace {

/** The folder at relativePath (from the current directory), created if
 *  missing; an error if it is something else. */
fs::path outputFolder(const std::string& relativePath) {
  const fs::path folder = fs::current_path() / relativePath;
  if (!fs::exists(folder))
    fs::create_directories(folder);
  else if (!fs::is_directory(folder))
    throw std::runtime_error("Output path " + relativePath +
                             " is not a directory");
  return folder;
}

} // namespace

void checkOutputFolder(const std::string& relativePath) {
  for (const auto& entry : fs::directory_iterator(outputFolder(relativePath)))
    fs::remove_all(entry);
}

void ensureOutputFolder(const std::string& relativePath) {
  outputFolder(relativePath);
}

bool keepAside(const std::vector<std::string>& paths, int cycle) {
  auto aside = [cycle](const fs::path& file, int copy) {
    fs::path name = file;
    name.replace_filename(file.stem().string() + "_before_cycle" +
                          std::to_string(cycle) +
                          (copy > 1 ? "_" + std::to_string(copy) : "") +
                          file.extension().string());
    return name;
  };
  // a suffix free for every file: a later restart from the same cycle keeps
  // its records too
  int copy = 1;
  while (std::any_of(paths.begin(), paths.end(), [&](const std::string& path) {
    return fs::exists(aside(path, copy));
  }))
    copy++;
  bool kept = false;
  for (const std::string& path : paths)
    if (fs::exists(path)) {
      fs::rename(path, aside(path, copy));
      kept = true;
    }
  return kept;
}

std::string recordsStem(const std::string& folder, const std::string& name,
                        int rank) {
  return (fs::path(folder) / ("subDomain" + std::to_string(rank) + "_" + name))
      .string();
}

std::vector<std::pair<int, std::string>>
filesByRank(const std::string& folder, const std::string& pattern) {
  std::vector<std::pair<int, std::string>> files;
  std::error_code error;
  if (!fs::is_directory(folder, error))
    return files;
  const std::regex name(pattern);
  std::smatch match;
  // a folder that cannot be listed throws: the run ends with its message
  for (const fs::directory_entry& entry : fs::directory_iterator(folder)) {
    const std::string file = entry.path().filename().string();
    if (std::regex_match(file, match, name))
      files.emplace_back(std::stoi(match[1].str()), entry.path().string());
  }
  std::sort(files.begin(), files.end());
  return files;
}

int keepAsideRecords(const std::string& folder, const std::string& name,
                     int cycle) {
  std::set<int> ranks; // a rank's .json and .bin together
  for (const auto& [rank, path] : filesByRank(
           folder, "subDomain([0-9]{1,9})_" + name + "\\.(json|bin)"))
    ranks.insert(rank);
  int kept = 0;
  for (const int rank : ranks) {
    const std::string stem = recordsStem(folder, name, rank);
    kept += keepAside({stem + ".json", stem + ".bin"}, cycle);
  }
  return kept;
}

bool RecordFile::open(const std::string& stem, const std::string& json,
                      const std::vector<std::string>& layout, bool isRestart,
                      int firstCycle) {
  binPath = stem + ".bin";
  const std::string jsonPath = stem + ".json";
  bool keptAside = false;
  if (isRestart && fs::exists(binPath)) {
    std::ifstream in(jsonPath);
    const std::string old((std::istreambuf_iterator<char>(in)),
                          std::istreambuf_iterator<char>());
    // the records need the layout and the same format (the "endian" and
    // "record_" lines of describeRecords)
    std::vector<std::string> needed = layout;
    std::istringstream lines(json);
    for (std::string line; std::getline(lines, line);)
      if (line.rfind("  \"record_", 0) == 0 ||
          line.rfind("  \"endian\"", 0) == 0)
        needed.push_back(line);
    if (std::any_of(needed.begin(), needed.end(), [&](const std::string& line) {
          return old.find(line) == std::string::npos;
        })) {
      keepAside({jsonPath, binPath}, firstCycle);
      keptAside = true;
    } else {
      dropRecordsFrom(binPath, magic, firstCycle);
    }
  }
  if (!isRestart || keptAside || !fs::exists(binPath)) {
    std::ofstream create(binPath, std::ios::binary | std::ios::trunc);
    if (!create)
      throw std::runtime_error("RecordFile: cannot create " + binPath);
  }
  // the description replaced at once: never a partial one beside records
  const std::string temporary = jsonPath + ".tmp";
  std::ofstream out(temporary, std::ios::trunc);
  out << json;
  out.close();
  if (!out)
    throw std::runtime_error("RecordFile: cannot write " + temporary);
  fs::rename(temporary, jsonPath);
  return keptAside;
}

bool RecordFile::cut(const std::string& stem, int firstCycle) const {
  const std::string bin = stem + ".bin";
  std::uint32_t first = magic; // of a missing or empty file
  std::ifstream(bin, std::ios::binary)
      .read(reinterpret_cast<char*>(&first), sizeof(first));
  if (first != magic)
    return keepAside({stem + ".json", bin}, firstCycle);
  dropRecordsFrom(bin, magic, firstCycle);
  return false;
}

void RecordFile::end() {
  pending.resize((pending.size() + kRecordAlignment - 1) / kRecordAlignment *
                     kRecordAlignment,
                 0);
  const std::uint64_t bytes = pending.size() - recordStart;
  std::memcpy(pending.data() + recordStart + kRecordSizeOffset, &bytes,
              sizeof(bytes));
}

void RecordFile::flush() {
  if (pending.empty())
    return;
  if (binPath.empty())
    throw std::logic_error("RecordFile: records added before open()");
  // appended: the records in the file are never written again
  std::error_code missing, ignored;
  const std::uintmax_t before = fs::file_size(binPath, missing);
  std::ofstream out(binPath, std::ios::binary | std::ios::app);
  out.write(pending.data(), (std::streamsize)pending.size());
  out.close();
  if (!out) { // the part written taken back: the file ends as it did
    fs::resize_file(binPath, missing ? 0 : before, ignored);
    throw std::runtime_error("RecordFile: cannot write " + binPath);
  }
  pending.clear();
}

void dropRecordsFrom(const std::string& path, std::uint32_t magic,
                     int firstCycle) {
  std::error_code error;
  const std::uintmax_t size = fs::file_size(path, error);
  if (error) // no file
    return;
  std::ifstream in(path, std::ios::binary);
  std::uintmax_t keep = 0; // the bytes of the whole records kept
  char header[kRecordHeaderBytes];
  while (keep + kRecordHeaderBytes <= size && in.seekg(keep) &&
         in.read(header, kRecordHeaderBytes)) {
    std::uint32_t first;
    std::int32_t cycle;
    std::uint64_t bytes;
    std::memcpy(&first, header, sizeof(first));
    std::memcpy(&cycle, header + sizeof(first), sizeof(cycle));
    std::memcpy(&bytes, header + kRecordSizeOffset, sizeof(bytes));
    if (first != magic || bytes < kRecordHeaderBytes || bytes > size - keep ||
        cycle >= firstCycle)
      break;
    keep += bytes;
  }
  in.close();
  if (keep < size) {
    fs::resize_file(path, keep, error);
    if (error)
      throw std::runtime_error("dropRecordsFrom: cannot resize " + path + " (" +
                               error.message() + ")");
  }
}

bool bigEndianHost() {
  const int test = 1;
  return *reinterpret_cast<const char*>(&test) != 1;
}

std::string describeRecords(const std::string& members, std::uint32_t magic,
                            const std::vector<RecordField>& header,
                            const std::vector<RecordField>& arrays,
                            const std::string& content) {
  auto fields = [](const std::vector<RecordField>& list) {
    std::string text;
    for (const RecordField& f : list)
      text += std::string(text.empty() ? "" : ", ") + "[\"" + f.name +
              "\", \"" + f.dtype + "\"" +
              (f.shape.empty() ? "" : ", [" + f.shape + "]") + "]";
    return "[" + text + "]";
  };
  return "{\n" + members + "  \"endian\": \"" +
         (bigEndianHost() ? "big" : "little") + "\",\n" +
         "  \"record_magic\": " + std::to_string(magic) + ",\n" +
         "  \"record_alignment\": " + std::to_string(kRecordAlignment) + ",\n" +
         "  \"record_header\": " + fields(header) + ",\n" +
         "  \"record_arrays\": " + fields(arrays) + ",\n" +
         "  \"content\": \"" + content + "\"\n}\n";
}
