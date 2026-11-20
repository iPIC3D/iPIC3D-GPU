/**
 * @file analysisRecordsTest.cpp
 * @brief Unit test of the files of the data-analysis records on a restart
 *        (outputPrepare): appendJsonRecord and dropJsonRecordsFrom, also
 *        after an append cut short at every byte; keepAside of a JSON and
 *        binary pair; dropVtkFrom. CPU only.
 */

#include "outputPrepare.h"

#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <string>
#include <vector>

#include <unistd.h>

namespace fs = std::filesystem;

namespace {

int failures = 0;

void check(bool ok, const std::string& what) {
  std::cout << "Test " << (ok ? "passed" : "failed") << ": " << what << "\n";
  failures += !ok;
}

std::string read(const fs::path& path) {
  std::ifstream in(path, std::ios::binary);
  return std::string((std::istreambuf_iterator<char>(in)),
                     std::istreambuf_iterator<char>());
}

void write(const fs::path& path, const std::string& text) {
  std::ofstream(path, std::ios::binary | std::ios::trunc) << text;
}

/** A record of several lines, as a GMM record. */
std::string record(int cycle) {
  return "{\n  \"cycle\": " + std::to_string(cycle) +
         ",\n  \"weight\": [0.5, 0.5]\n}";
}

/** The file of the records of these cycles. */
std::string recordsOf(const fs::path& path, const std::vector<int>& cycles) {
  fs::remove(path);
  for (int cycle : cycles)
    appendJsonRecord(path.string(), cycle, record(cycle));
  return read(path);
}

/** The file after dropJsonRecordsFrom, "" when removed. */
std::string dropped(const fs::path& path, const std::string& content,
                    int firstCycle) {
  write(path, content);
  dropJsonRecordsFrom(path.string(), firstCycle);
  return fs::exists(path) ? read(path) : "";
}

void testJsonRecords(const fs::path& dir) {
  const fs::path json = dir / "species0.json";
  const std::string upTo10 = recordsOf(json, {0, 10});
  const std::string upTo20 = recordsOf(json, {0, 10, 20});
  const std::string upTo30 = recordsOf(json, {0, 10, 20, 30});
  check(upTo10 == "{\n\"0\": " + record(0) + ",\n\"10\": " + record(10) + "\n}",
        "appendJsonRecord: a JSON object of the records");
  check(dropped(json, upTo30, 20) == upTo10 &&
            dropped(json, upTo30, 25) == upTo20 &&
            dropped(json, upTo30, 31) == upTo30,
        "dropJsonRecordsFrom: the records from the cycle on");
  check(dropped(json, upTo30, 0).empty(),
        "dropJsonRecordsFrom: no record left, the file removed");
  fs::remove(json);
  dropJsonRecordsFrom(json.string(), 10);
  check(!fs::exists(json), "dropJsonRecordsFrom: a missing file left so");

  // the append of 30 cut short at every byte (over the closing "\n}" of
  // upTo20): a restart from 30 or before drops it
  const size_t start = upTo20.size() - 2;
  const std::string append = upTo30.substr(start);
  bool repaired = true;
  for (size_t k = 0; k <= append.size(); k++) {
    std::string cut = upTo20.substr(0, start) + append.substr(0, k);
    if (k < 2)
      cut += std::string("\n}").substr(k);
    repaired &=
        dropped(json, cut, 30) == upTo20 && dropped(json, cut, 20) == upTo10;
  }
  check(repaired, "an append cut short at any byte: dropped with its cycle");
  // the first append cut short: the file removed, or completed when only
  // its closing "\n}" is missing
  const std::string only0 = recordsOf(json, {0});
  bool removed = true;
  for (size_t k = 0; k <= only0.size(); k++) {
    const std::string result = dropped(json, only0.substr(0, k), 5);
    removed &= k + 2 >= only0.size() ? result == only0 : result.empty();
  }
  check(removed, "the first append cut short: the file removed, or "
                 "completed when its record is whole");
  // the records appended again after a drop
  write(json, upTo30);
  dropJsonRecordsFrom(json.string(), 20);
  appendJsonRecord(json.string(), 20, record(20));
  check(read(json) == upTo20, "the records appended again after a drop");
}

void testKeepAside(const fs::path& dir) {
  const std::string json = (dir / "species_0.json").string();
  const std::string bin = (dir / "species_0.bin").string();
  auto aside = [&](const std::string& suffix, const char* extension) {
    return dir / ("species_0_before_cycle20" + suffix + extension);
  };
  write(json, "first json");
  write(bin, "first bin");
  keepAside({json, bin}, 20);
  check(!fs::exists(json) && !fs::exists(bin) &&
            read(aside("", ".json")) == "first json" &&
            read(aside("", ".bin")) == "first bin",
        "keepAside: the pair renamed");
  // again from the same cycle, the binary missing: the next free suffix
  write(json, "second json");
  keepAside({json, bin}, 20);
  check(read(aside("_2", ".json")) == "second json" &&
            !fs::exists(aside("_2", ".bin")) &&
            read(aside("", ".json")) == "first json",
        "keepAside again: _2, nothing overwritten");
  // the JSON missing: a suffix free for the pair (_2 is taken by the JSON)
  write(bin, "third bin");
  keepAside({json, bin}, 20);
  check(read(aside("_3", ".bin")) == "third bin" &&
            read(aside("_2", ".json")) == "second json",
        "keepAside of a pair: one suffix free for both");
}

void testDropVtk(const fs::path& dir) {
  const std::vector<std::string> kept = {"h_3D_0.vtk", "h_3D_10.vtk",
                                         "notes.txt", "x.vtk", "y_abc.vtk"};
  const std::vector<std::string> removed = {"h_3D_20.vtk", "s1_3D_100.vtk"};
  for (const auto& names : {kept, removed})
    for (const std::string& name : names)
      write(dir / name, name);
  dropVtkFrom(dir.string(), 20);
  bool ok = true;
  for (const std::string& name : kept)
    ok &= fs::exists(dir / name);
  for (const std::string& name : removed)
    ok &= !fs::exists(dir / name);
  check(ok, "dropVtkFrom: the histograms from the cycle on");
}

} // namespace

int main() {
  const fs::path dir = fs::temp_directory_path() /
                       ("analysisRecordsTest_" + std::to_string(getpid()));
  const char* parts[] = {"records", "aside", "vtk"};
  for (const char* part : parts)
    fs::create_directories(dir / part);
  try {
    testJsonRecords(dir / parts[0]);
    testKeepAside(dir / parts[1]);
    testDropVtk(dir / parts[2]);
  } catch (const std::exception& error) {
    std::cerr << "analysisRecordsTest failed: " << error.what() << "\n";
    failures++;
  }
  fs::remove_all(dir);
  return failures == 0 ? 0 : 1;
}
