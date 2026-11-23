#ifndef OUTPUT_PREPARE_H
#define OUTPUT_PREPARE_H

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

/** Binary records of the data analysis (RecordFile): the bytes of the
 *  header that starts every record, the offset of its size (uint64, the last
 *  field of the header), and the multiple of bytes of every record. */
inline constexpr std::size_t kRecordHeaderBytes = 32;
inline constexpr std::size_t kRecordSizeOffset = 24;
inline constexpr std::size_t kRecordAlignment = 8;

/** The output folder at relativePath (from the current directory), created
 *  if missing, emptied (checkOutputFolder) or kept as it is
 *  (ensureOutputFolder); an error if it is not a folder. */
void checkOutputFolder(const std::string& relativePath);
void ensureOutputFolder(const std::string& relativePath);
/** Rename the files that exist to <stem>_before_cycle<cycle><extension>,
 *  or with _2, _3, ... after the cycle when a file of that name exists, the
 *  same suffix for all (a JSON and its binary): records that a restart does
 *  not continue are kept, never overwritten. @return whether one existed. */
bool keepAside(const std::vector<std::string>& paths, int cycle);
/** The files in folder whose names match pattern, a regular expression
 *  whose first group is a rank (at most 9 digits, an int), as (rank, path)
 *  sorted by rank; none if folder is missing, filesystem_error if it cannot
 *  be listed. */
std::vector<std::pair<int, std::string>> filesByRank(const std::string& folder,
                                                     const std::string& pattern);
/** The records file of a rank in folder, without its extension (.json,
 *  .bin): <folder>/subDomain<rank>_<name>. */
std::string recordsStem(const std::string& folder, const std::string& name,
                        int rank);
/** Keep aside (keepAside) the records files (recordsStem: .json and .bin)
 *  of every rank that has one of them in folder. @return the ranks that
 *  had. */
int keepAsideRecords(const std::string& folder, const std::string& name,
                     int cycle);

// ======= Binary records of the data analysis =======
/** A field of a record: its name, its numpy dtype and, for an array, its
 *  shape as the members of a JSON array: names of header fields and numbers
 *  (e.g. "\"regions\", 6"). */
struct RecordField {
  std::string name, dtype, shape;
};

/** The numpy name of an arithmetic type: float32, int32, uint16, ... */
template <typename T> std::string dtypeName() {
  return std::string(std::is_floating_point_v<T> ? "float"
                     : std::is_signed_v<T>       ? "int"
                                                 : "uint") +
         std::to_string(8 * sizeof(T));
}

/** True on a big-endian host. */
bool bigEndianHost();

/**
 * @brief The JSON description of a records file: members (JSON members,
 * each line ending with ",\n": the rank, the regions, ...), then the format
 * of the records: endianness, magic, alignment, the fields of the header and
 * the arrays that follow it, in order, and content, a text on what they
 * hold.
 */
std::string describeRecords(const std::string& members, std::uint32_t magic,
                            const std::vector<RecordField>& header,
                            const std::vector<RecordField>& arrays,
                            const std::string& content);

/**
 * @brief A file of binary records appended in cycle order, <stem>.bin,
 * described by <stem>.json (describeRecords). A record starts with a header
 * of kRecordHeaderBytes: the magic of the file (uint32), the cycle (int32),
 * fields of its kind, and its size in bytes (uint64 at kRecordSizeOffset,
 * header and padding included, a multiple of kRecordAlignment); the records
 * are read by scanning the headers. The records of a cycle are built in
 * memory (begin, put, end) and written at once (flush), appended: the
 * records of the cycles before are never written again, so a run that ends
 * during a write leaves them whole, followed at most by records of that
 * cycle and one cut short, which the readers skip and a restart drops
 * (dropRecordsFrom).
 */
class RecordFile {
public:
  /** @param magic The first word of every record of the file. */
  explicit RecordFile(std::uint32_t magic) : magic(magic) {}

  /**
   * @brief Open the file and write its description: a fresh start creates
   * it empty; a restart keeps its records before firstCycle
   * (dropRecordsFrom), or, when the JSON on disk lacks a part of layout or a
   * line of the record format of json (records of another layout, format or
   * byte order), keeps both files aside and starts again.
   * @param json The description, written to <stem>.json.
   * @param layout Parts of json that records of this file need.
   * @return true when a restart kept the files aside
   */
  bool open(const std::string& stem, const std::string& json,
            const std::vector<std::string>& layout, bool isRestart,
            int firstCycle);

  /** Of an analysis switched off: cut the records of an earlier run from
   *  firstCycle on, which the restart would have written again
   *  (dropRecordsFrom); a file whose first word is not the magic (another
   *  format or byte order) is kept aside instead (keepAside).
   *  @return whether it was kept aside. */
  bool cut(const std::string& stem, int firstCycle) const;

  /** Start a record with its header: kRecordHeaderBytes, a member cycle
   *  after the magic and a member bytes at kRecordSizeOffset; its first word
   *  is set to the magic of the file. */
  template <typename Header> void begin(const Header& header) {
    static_assert(sizeof(Header) == kRecordHeaderBytes &&
                      offsetof(Header, cycle) == sizeof(magic) &&
                      offsetof(Header, bytes) == kRecordSizeOffset,
                  "a record header");
    recordStart = pending.size();
    put(&header, 1);
    std::memcpy(pending.data() + recordStart, &magic, sizeof(magic));
  }

  /** Add count values to the record, as their bytes. */
  template <typename T> void put(const T* values, std::size_t count) {
    const char* bytes = reinterpret_cast<const char*>(values);
    pending.insert(pending.end(), bytes, bytes + sizeof(T) * count);
  }

  /** End the record: padded to a multiple of kRecordAlignment bytes, its
   *  size written into its header. */
  void end();

  /** Write the records ended since the last flush, in one write. A write
   *  that fails (a full disk) throws runtime_error with its part of the file
   *  taken back, the records still pending. */
  void flush();

private:
  std::uint32_t magic;
  std::string binPath;       // of open()
  std::vector<char> pending; // records not written yet
  std::size_t recordStart = 0;
};

/** Cut a records file before its first record of cycle firstCycle or
 *  later, cut short or of another magic: the records a restart writes again
 *  (a missing file is left so). */
void dropRecordsFrom(const std::string& path, std::uint32_t magic,
                     int firstCycle);

#endif // OUTPUT_PREPARE_H
