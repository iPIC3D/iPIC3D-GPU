/**
 * @file analysisRecordsTest.cpp
 * @brief Unit test of the files of the data-analysis records on a restart
 *        (outputPrepare): keepAside of a JSON and binary pair, and of the
 *        records and mappings of every rank of a folder; the binary
 *        records of RecordFile: built in memory, padded, written at the
 *        flush; a restart that drops the later records, a record cut short
 *        at every byte or of another magic, the same layout continued,
 *        another layout or record format and records without their
 *        description kept aside; the cut of an analysis switched off (a file
 *        of another magic kept aside); the JSON of describeRecords. CPU
 *        only.
 */

#include "outputPrepare.h"

#include <cstdint>
#include <cstring>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <string>
#include <vector>

#include <unistd.h>

namespace fs = std::filesystem;

namespace {

/** The magic of the test records, of another file, and the cycles of the
 *  records written. */
constexpr std::uint32_t kMagic = 0x54534554, kOtherMagic = 0x4f544852;
constexpr int kRecordCycles[] = {0, 10, 20, 30};

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
  // nothing there: nothing kept
  check(!keepAside({json, bin}, 30), "keepAside of no file: false");

  // the records and mappings of every rank of a folder: ranks 0 (a pair and
  // its mapping), 7 (a binary and its mapping) and 9 (a mapping alone);
  // the records of another analysis left
  const fs::path folder = dir / "records";
  fs::create_directories(folder);
  for (const char* file :
       {"subDomain0_x.json", "subDomain0_x.bin", "subDomain7_x.bin",
        "vctMapping_subDomain0.txt", "vctMapping_subDomain7.txt",
        "vctMapping_subDomain9.txt", "subDomain0_other.bin"})
    write((folder / file).string(), file);
  bool ok = keepAsideRecords(folder.string() + "/", "x", 0) == 3;
  for (const char* file :
       {"subDomain0_x_before_cycle0.json", "subDomain0_x_before_cycle0.bin",
        "subDomain7_x_before_cycle0.bin",
        "vctMapping_subDomain0_before_cycle0.txt",
        "vctMapping_subDomain7_before_cycle0.txt",
        "vctMapping_subDomain9_before_cycle0.txt", "subDomain0_other.bin"})
    ok &= fs::exists(folder / file);
  ok &= !fs::exists(folder / "subDomain0_x.bin") &&
        !fs::exists(folder / "vctMapping_subDomain9.txt") &&
        keepAsideRecords((dir / "none").string(), "x", 0) == 0;
  check(ok, "keepAsideRecords: the files of every rank kept aside, those of "
            "another analysis left, no folder no file");
}

/** The header of the test records. */
struct TestHeader {
  std::uint32_t magic;
  std::int32_t cycle;
  std::uint64_t count;
  std::uint64_t unused;
  std::uint64_t bytes;
};

/** The bytes of a test record, composed by hand: its header, cycle + 1
 *  doubles and one byte (a size not a multiple of the alignment), zero
 *  padding. */
std::string expectedRecord(int cycle, std::uint32_t magic = kMagic) {
  const size_t data = sizeof(TestHeader) + sizeof(double) * (cycle + 1) + 1;
  const size_t bytes =
      (data + kRecordAlignment - 1) / kRecordAlignment * kRecordAlignment;
  const TestHeader header = {magic, cycle, (std::uint64_t)cycle + 1, 0, bytes};
  std::string record(bytes, '\0');
  std::memcpy(&record[0], &header, sizeof(header));
  for (int i = 0; i <= cycle; i++) {
    const double value = cycle;
    std::memcpy(&record[sizeof(header) + sizeof(double) * i], &value,
                sizeof(value));
  }
  record[data - 1] = 1;
  return record;
}

/** The same record built by a RecordFile (begin sets the magic). */
void addRecord(RecordFile& file, int cycle) {
  file.begin(TestHeader{0, cycle, (std::uint64_t)cycle + 1, 0, 0});
  const std::vector<double> values(cycle + 1, cycle);
  file.put(values.data(), values.size());
  const char odd = 1;
  file.put(&odd, 1);
  file.end();
}

void testRecordFile(const fs::path& dir) {
  const std::string stem = (dir / "subDomain0_test").string(),
                    bin = stem + ".bin";
  const std::string layout = "  \"regions\": \"rank\",\n";
  const std::string json = "{\n" + layout + "}\n";
  RecordFile file(kMagic);
  bool ok = !file.open(stem, json, {layout}, false, 0) &&
            read(stem + ".json") == json && fs::file_size(bin) == 0;
  std::string expected;
  for (int cycle : kRecordCycles) {
    addRecord(file, cycle);
    expected += expectedRecord(cycle);
  }
  ok &= fs::file_size(bin) == 0; // nothing written before flush
  file.flush();
  file.flush(); // nothing more
  check(ok && read(bin) == expected,
        "RecordFile: the records built, padded, their sizes in the headers, "
        "written at the flush");

  // the size of the records before a cycle
  auto before = [&](int cycle) {
    size_t bytes = 0;
    for (int c : kRecordCycles)
      bytes += c < cycle ? expectedRecord(c).size() : 0;
    return bytes;
  };
  const std::string all = read(bin);
  dropRecordsFrom(bin, kMagic, 20);
  check(read(bin) == all.substr(0, before(20)),
        "dropRecordsFrom: the records from the cycle on");
  // a record cut short at every byte, or of another magic
  const std::string next = expectedRecord(20),
                    other = expectedRecord(20, kOtherMagic);
  ok = true;
  for (size_t cut = 1; cut < next.size(); cut++) {
    write(bin, all.substr(0, before(20)) + next.substr(0, cut));
    dropRecordsFrom(bin, kMagic, 100);
    ok &= read(bin) == all.substr(0, before(20));
  }
  write(bin, all.substr(0, before(20)) + other);
  dropRecordsFrom(bin, kMagic, 100);
  ok &= read(bin) == all.substr(0, before(20));
  dropRecordsFrom((dir / "missing.bin").string(), kMagic, 0);
  ok &= !fs::exists(dir / "missing.bin");
  check(ok, "dropRecordsFrom: a record cut short at every byte or of another "
            "magic dropped, a missing file left so");

  // a restart: the same layout continues, another is kept aside
  write(bin, all);
  ok = !file.open(stem, json, {layout}, true, 30) &&
       read(bin) == all.substr(0, before(30));
  const std::string otherJson = "{\n  \"regions\": \"macrocells\",\n}\n";
  ok &= file.open(stem, otherJson, {"  \"regions\": \"macrocells\",\n"}, true,
                  30) &&
        fs::file_size(bin) == 0 && read(stem + ".json") == otherJson &&
        read((dir / "subDomain0_test_before_cycle30.bin").string()) ==
            all.substr(0, before(30)) &&
        read((dir / "subDomain0_test_before_cycle30.json").string()) == json;
  check(ok, "RecordFile::open on a restart: the same layout continued, "
            "another kept aside");

  // a restart: the same layout in another record format, and records
  // without their description, kept aside
  const std::string format1 = "{\n" + layout + "  \"record_magic\": 1,\n}\n",
                    format2 = "{\n" + layout + "  \"record_magic\": 2,\n}\n";
  write(bin, all);
  write(stem + ".json", format1);
  ok = file.open(stem, format2, {layout}, true, 40) &&
       fs::file_size(bin) == 0 &&
       read((dir / "subDomain0_test_before_cycle40.bin").string()) == all;
  write(bin, all);
  fs::remove(stem + ".json");
  ok &= file.open(stem, json, {layout}, true, 50) && fs::file_size(bin) == 0 &&
        read((dir / "subDomain0_test_before_cycle50.bin").string()) == all;
  check(ok, "RecordFile::open on a restart: another record format, and "
            "records without their description, kept aside");

  // an analysis switched off: its records cut from the cycle on, a file of
  // another magic (format or byte order) kept aside, not emptied
  write(bin, all);
  ok = !file.cut(stem, 20) && read(bin) == all.substr(0, before(20));
  write(bin, other + all);
  ok &= file.cut(stem, 20) && !fs::exists(bin) && !fs::exists(stem + ".json") &&
        read((dir / "subDomain0_test_before_cycle20.bin").string()) ==
            other + all &&
        fs::exists(dir / "subDomain0_test_before_cycle20.json");
  ok &= !file.cut((dir / "missing").string(), 0) &&
        !fs::exists(dir / "missing.bin");
  check(ok, "RecordFile::cut: the records from the cycle on, a file of "
            "another magic kept aside, a missing one left so");
}

void testDescribe() {
  const std::string json = describeRecords(
      "  \"rank\": 3,\n", 7, {{"magic", "uint32", ""}, {"cycle", "int32", ""}},
      {{"values", "float64", "\"regions\", 6"}}, "what they hold");
  check(json == std::string("{\n  \"rank\": 3,\n  \"endian\": \"") +
                    (bigEndianHost() ? "big" : "little") +
                    "\",\n  \"record_magic\": 7,\n  \"record_alignment\": " +
                    std::to_string(kRecordAlignment) +
                    ",\n  \"record_header\": [[\"magic\", \"uint32\"], "
                    "[\"cycle\", \"int32\"]],\n  \"record_arrays\": "
                    "[[\"values\", \"float64\", [\"regions\", 6]]],\n  "
                    "\"content\": \"what they hold\"\n}\n",
        "describeRecords: the members, then the format of the records");
}

} // namespace

int main() {
  const fs::path dir = fs::temp_directory_path() /
                       ("analysisRecordsTest_" + std::to_string(getpid()));
  const char* parts[] = {"aside", "binary"};
  for (const char* part : parts)
    fs::create_directories(dir / part);
  try {
    testKeepAside(dir / parts[0]);
    testRecordFile(dir / parts[1]);
    testDescribe();
  } catch (const std::exception& error) {
    std::cerr << "analysisRecordsTest failed: " << error.what() << "\n";
    failures++;
  }
  fs::remove_all(dir);
  return failures == 0 ? 0 : 1;
}
