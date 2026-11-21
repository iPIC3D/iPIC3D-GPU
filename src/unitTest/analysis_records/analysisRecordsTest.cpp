/**
 * @file analysisRecordsTest.cpp
 * @brief Unit test of the files of the data-analysis records on a restart
 *        (outputPrepare): keepAside of a JSON and binary pair, and of the
 *        records of every rank of a folder; the binary
 *        records of RecordFile: built in memory, padded, written at the
 *        flush; a flush that fails midway taken back, the records before as
 *        they were; a restart that drops the later records, a record cut short
 *        at every byte or of another magic, the same layout continued,
 *        another layout or record format and records without their
 *        description kept aside; the cut of an analysis switched off (a file
 *        of another magic kept aside); the JSON of describeRecords; the
 *        GMM records (gmmRecords): the fitted regions and their active
 *        components only; the index of sparse bins (sparseIndex): lists or
 *        bitmaps. CPU only.
 */

#include "DataAnalysisInput.h"
#include "gmmRecords.h"
#include "outputPrepare.h"
#include "sparseIndex.h"

#include <csignal>
#include <cstdint>
#include <cstring>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

#include <sys/resource.h>
#include <unistd.h>

namespace fs = std::filesystem;

namespace {

/** The magic of the test records, of another file, and the cycles of the
 *  records written. */
constexpr std::uint32_t kMagic = 0x54534554, kOtherMagic = 0x4f544852;
constexpr int kRecordCycles[] = {0, 10, 20, 30};
/** The bytes of the second record of a flush that the file can take before
 *  its write fails (testFailedFlush): its header and a part of its values. */
constexpr std::size_t kFailedRecordBytes = kRecordHeaderBytes + 12;
/** The components of the test fits of a GMM record. */
constexpr int kTestComponents = 3;

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

  // the records of every rank of a folder: ranks 0 (a pair), 7 (a binary)
  // and 9 (a JSON); the records of another analysis left
  const fs::path folder = dir / "records";
  fs::create_directories(folder);
  for (const char* file :
       {"subDomain0_x.json", "subDomain0_x.bin", "subDomain7_x.bin",
        "subDomain9_x.json", "subDomain0_other.bin"})
    write((folder / file).string(), file);
  bool ok = keepAsideRecords(folder.string() + "/", "x", 0) == 3;
  for (const char* file :
       {"subDomain0_x_before_cycle0.json", "subDomain0_x_before_cycle0.bin",
        "subDomain7_x_before_cycle0.bin", "subDomain9_x_before_cycle0.json",
        "subDomain0_other.bin"})
    ok &= fs::exists(folder / file);
  ok &= !fs::exists(folder / "subDomain0_x.bin") &&
        !fs::exists(folder / "subDomain9_x.json") &&
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

/** A flush that fails midway, the file at the size limit of the process as
 *  on a full disk, after a whole record of the cycle and a part of the next
 *  one: the records of the flushes before are as they were and the file ends
 *  with them; the records, still pending, are written by the next flush. */
void testFailedFlush(const fs::path& dir) {
  const std::string stem = (dir / "subDomain0_failed").string(),
                    bin = stem + ".bin";
  RecordFile file(kMagic);
  file.open(stem, "{\n}\n", {}, false, 0);
  addRecord(file, 0);
  addRecord(file, 10);
  file.flush();
  const std::string before = read(bin);
  addRecord(file, 20);
  addRecord(file, 30);

  struct rlimit unlimited, limited;
  getrlimit(RLIMIT_FSIZE, &unlimited);
  limited = unlimited;
  limited.rlim_cur =
      before.size() + expectedRecord(20).size() + kFailedRecordBytes;
  // the write fails at the limit, instead of the signal ending the test
  const auto handler = std::signal(SIGXFSZ, SIG_IGN);
  setrlimit(RLIMIT_FSIZE, &limited);
  bool refused = false;
  try {
    file.flush();
  } catch (const std::runtime_error&) {
    refused = true;
  }
  setrlimit(RLIMIT_FSIZE, &unlimited);
  std::signal(SIGXFSZ, handler);
  check(refused && before == expectedRecord(0) + expectedRecord(10) &&
            read(bin) == before,
        "RecordFile::flush that fails midway: an error, the file as before "
        "it, its records whole");
  file.flush();
  check(read(bin) == before + expectedRecord(20) + expectedRecord(30),
        "RecordFile::flush after one that failed: its records written once");
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

/** A GMM record: of the fitted regions only, increasing, and of their
 *  active components only, the upper triangle of their covariances, in
 *  GMMRecordValue; the JSON lists the arrays; no record without a fit, nor
 *  of regions out of range or not increasing; the last slot of 256. */
void testGMMRecord(const fs::path& dir) {
  using namespace dataAnalysis;
  const int K = kTestComponents, D = kGMMRecordDim;
  // two fits of 4 regions: region 1 with every component, region 3 with its
  // component 1 pruned; covariances not symmetric, to see which half is
  // written
  const bool all[kTestComponents] = {true, true, true},
             pruned[kTestComponents] = {true, false, true};
  double weight[2][kTestComponents], mean[2][kTestComponents * kGMMRecordDim],
      cov[2][kTestComponents * kGMMRecordDim * kGMMRecordDim];
  for (int f = 0; f < 2; f++) {
    for (int k = 0; k < K; k++)
      weight[f][k] = 0.125 * (f + 1) + k;
    for (int i = 0; i < K * D; i++)
      mean[f][i] = 100 * f + i + 0.5;
    for (int i = 0; i < K * D * D; i++)
      cov[f][i] = 1000 * f + i + 0.25;
  }
  const std::vector<GMMFitView> fits = {
      {1, 7, -12.5, all, weight[0], mean[0], cov[0]},
      {3, 9, -20.25, pruned, weight[1], mean[1], cov[1]}};
  const std::string stem = (dir / "subDomain0_gmm").string();
  RecordFile file(kGMMRecordMagic);
  file.open(stem, "{\n}\n", {}, false, 0);
  writeGMMRecord(file, 40, 2, 4, K, fits);
  file.flush();

  // the record expected: 2 fits, 5 components
  std::string expected;
  auto put = [&](const auto& value) {
    expected.append(reinterpret_cast<const char*>(&value), sizeof(value));
  };
  put(GMMRecordHeader{kGMMRecordMagic, 40, 2, 4, 2, 5, 0});
  const int slots[2][kTestComponents] = {{0, 1, 2}, {0, 2, -1}},
            counts[2] = {3, 2};
  auto eachComponent = [&](auto value) {
    for (int f = 0; f < 2; f++)
      for (int j = 0; j < counts[f]; j++)
        value(f, slots[f][j]);
  };
  put(-12.5);
  put(-20.25);
  eachComponent([&](int f, int k) { put((GMMRecordValue)weight[f][k]); });
  eachComponent([&](int f, int k) {
    for (int d = 0; d < D; d++)
      put((GMMRecordValue)mean[f][k * D + d]);
  });
  eachComponent([&](int f, int k) {
    for (int e : {0, 1, 2, 4, 5, 8})
      put((GMMRecordValue)cov[f][k * D * D + e]);
  });
  put(std::int32_t{7});
  put(std::int32_t{9});
  put(std::uint16_t{1});
  put(std::uint16_t{3});
  put(std::uint16_t{3});
  put(std::uint16_t{2});
  eachComponent([&](int, int k) { put((std::uint8_t)k); });
  expected.resize((expected.size() + kRecordAlignment - 1) / kRecordAlignment *
                      kRecordAlignment,
                  '\0');
  const std::uint64_t bytes = expected.size();
  std::memcpy(&expected[kRecordSizeOffset], &bytes, sizeof(bytes));
  check(read(stem + ".bin") == expected,
        "writeGMMRecord: the fitted regions, their active components, the "
        "upper triangle of the covariances");

  const std::string json = describeGMMRecords("  \"rank\": 0,\n");
  bool ok = true;
  for (const char* part :
       {"[\"log_likelihood\", \"float64\", [\"fits\"]]",
        "[\"covariances\", \"float32\", [\"components\", 6]]",
        "[\"slot\", \"uint8\", [\"components\"]]",
        "\"covariance_elements\": [\"uu\", \"uv\", \"uw\", \"vv\", "
        "\"vw\", \"ww\"]"})
    ok &= json.find(part) != std::string::npos;
  // refused: no fit, a region out of range, regions not increasing
  const std::vector<GMMFitView> refused[] = {
      {}, {{4, 7, -12.5, all, weight[0], mean[0], cov[0]}}, {fits[1], fits[0]}};
  for (const std::vector<GMMFitView>& wrong : refused)
    try {
      writeGMMRecord(file, 50, 0, 4, K, wrong);
      ok = false;
    } catch (const std::invalid_argument&) {
    }
  check(ok, "describeGMMRecords: the arrays of the records; writeGMMRecord "
            "refuses no fit, a region out of range, regions not increasing");

  // 256 components, the last one alone active: slot 255
  std::vector<double> one(kGMMMaxComponents, 1.0);
  std::unique_ptr<bool[]> last(new bool[kGMMMaxComponents]());
  last[kGMMMaxComponents - 1] = true;
  const std::vector<double> means(kGMMMaxComponents * D, 0.5),
      covs(kGMMMaxComponents * D * D, 0.25);
  RecordFile wide(kGMMRecordMagic);
  wide.open((dir / "subDomain1_gmm").string(), "{\n}\n", {}, false, 0);
  writeGMMRecord(
      wide, 0, 0, 1, kGMMMaxComponents,
      {{0, 1, 0.0, last.get(), one.data(), means.data(), covs.data()}});
  wide.flush();
  const std::string record = read((dir / "subDomain1_gmm.bin").string());
  GMMRecordHeader header;
  std::memcpy(&header, record.data(), sizeof(header));
  // the slot, last array: after the header, 8 + 4 (1 + 3 + 6) + 4 + 2 + 2
  check(header.components == 1 &&
            (std::uint8_t)record[sizeof(header) + 8 + 40 + 8] == 255,
        "writeGMMRecord: slot 255 of 256 components");
}

/** The index of sparse bins: per region the list of its bins or, when
 *  smaller, its bitmap (of any number of bins); nothing for an empty region;
 *  32-bit indices above kShortIndexBins bins; regions without bins
 *  refused. */
void testBinIndex() {
  using namespace sparseBins;
  // 64 bins per region, a bitmap of 8 bytes: 6 bins (12 bytes as a list) as
  // the bitmap, an empty region, 2 bins (4 bytes) as a list
  const std::vector<int> toy = {3, 4, 5, 20, 21, 36, 10, 50};
  std::vector<std::uint8_t> expected = {0x38, 0, 0x30, 0, 0x10, 0, 0, 0};
  for (const std::uint16_t bin : {std::uint16_t{10}, std::uint16_t{50}}) {
    const auto* bytes = reinterpret_cast<const std::uint8_t*>(&bin);
    expected.insert(expected.end(), bytes, bytes + sizeof(bin));
  }
  bool ok = encodeBinIndex({0, 6, 6, 8}, toy.data(), 64) == expected;

  // 32768 bins (a bitmap of 4096 bytes): 2048 bins as a list of 4096 bytes,
  // 2049 as the bitmap
  std::vector<int> bins(2049);
  std::iota(bins.begin(), bins.end(), 0);
  for (int& bin : bins)
    bin *= 15;
  const std::vector<std::uint8_t> list = encodeBinIndex({0, 2048}, bins.data(),
                                                        32768),
                                  bitmap = encodeBinIndex({0, 2049},
                                                          bins.data(), 32768);
  std::uint16_t second;
  std::memcpy(&second, &list[2], sizeof(second));
  ok &= !isBitmap(2048, 32768) && isBitmap(2049, 32768) &&
        list.size() == 4096 && second == 15 && bitmap.size() == 4096;
  int set = 0;
  auto bit = [&](int b) {
    return bitmap[b / kBinsPerBitmapByte] >> (b % kBinsPerBitmapByte) & 1;
  };
  for (int b = 0; b < 32768; b++)
    set += bit(b);
  ok &= set == 2049 && bit(15) && bit(30720);

  // more than kShortIndexBins bins: 32-bit indices
  const std::vector<int> far = {7, 70000};
  const std::vector<std::uint8_t> wide =
      encodeBinIndex({0, 2}, far.data(), 1 << 17);
  std::uint32_t last;
  std::memcpy(&last, &wide[4], sizeof(last));
  ok &= wide.size() == 8 && last == 70000;
  // 12 bins, a bitmap of 2 bytes: 2 bins listed (4 bytes) as the bitmap
  const std::vector<int> odd = {1, 11};
  ok &= encodeBinIndex({0, 2}, odd.data(), 12) ==
        std::vector<std::uint8_t>{0x02, 0x08};
  try {
    encodeBinIndex({0}, nullptr, 0);
    ok = false;
  } catch (const std::invalid_argument&) {
  }
  check(ok, "encodeBinIndex: a region as its bitmap or its list, whichever "
            "is smaller, an empty one as nothing, 32-bit indices above "
            "65536 bins, any number of bins per region");
}

} // namespace

int main() {
  const fs::path dir = fs::temp_directory_path() /
                       ("analysisRecordsTest_" + std::to_string(getpid()));
  const char* parts[] = {"aside", "binary", "failed", "gmm"};
  for (const char* part : parts)
    fs::create_directories(dir / part);
  try {
    testKeepAside(dir / parts[0]);
    testRecordFile(dir / parts[1]);
    testFailedFlush(dir / parts[2]);
    testDescribe();
    testGMMRecord(dir / parts[3]);
    testBinIndex();
  } catch (const std::exception& error) {
    std::cerr << "analysisRecordsTest failed: " << error.what() << "\n";
    failures++;
  }
  fs::remove_all(dir);
  return failures == 0 ? 0 : 1;
}
